// Axera NPU backend: Qwen3.5/3.6-35B-A3B with the sgl-kernel-axera NVFP4 kernels.
//
// The manifest's kernels (ax-dsl-kernels models/qwen35_35b_a3b/sgl):
//   decode          role decode: all 40 layers for one token in one call; runtime banks are indexed by block.
//   prefill.gdn     one 256-row gated-delta-net layer.
//   prefill.attn    one 256-row gated attention layer with paged KV and M-RoPE.
//   prefill.moe     one layer's MoE for a fixed 4096-row workspace.
//   prefill.padding zeros the workspace rows beyond the live token count.
// Prefill is layer-outer over up to 4096 tokens: for each layer, run 256-row attention chunks, clear the padded rows,
// then run the fixed 4096-row MoE. The decode and prefill kernels read the same NVFP4 weight banks.
// The state is this adapter's (the decode kernel's banks): KV cache paged [10][64 pages][K, V][256][2 x 256] bf16,
// delta-net states [30][32][128][128] f32, conv states q / k [30][3][2048], v [30][3][4096] bf16 (value heads in HF
// order). The prefill kernels write the recurrent states to a second set, the two swapped after each page. One
// sequence. Weights only from the GGUF: every projection and expert row is quantized to NVFP4 (E2M1 pairs, an E4M3
// scale per 16 inputs, the FP32 tensor scale2) in memory at load, one tensor / expert at a time, straight into the
// device banks. An offline bank pack is a debug fallback only (AXERA_SGL35_BANKS=1); the release ships no bank files.

#include "../model.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;          // cells per page = prefill rows per call
constexpr int WN = 4096;         // rows in the fixed MoE prefill workspace
constexpr int KH = 16, VH = 32;  // key / value heads of a delta-net layer
constexpr int REP = VH / KH;
constexpr int NB = 10;           // blocks of 4 layers

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

// GGUF's tiled value head of HF (grouped) head g
inline int tiled(int g) {
    return (g % REP) * KH + g / REP;
}

std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) { memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float)); });
    }
    axera_drop_mapped(w);
    return v;
}

class qwen35_moe_sgl final : public model {
  public:
    explicit qwen35_moe_sgl(const manifest & m) : man(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, AX = 1;  // AX: prefill position axes
    std::vector<int> ord;                   // layer -> its index among the delta-net / attention layers

    std::map<std::string, axera_cmm> banks;  // the decode kernel's inputs by name (weights, states, KV cache)
    std::map<std::string, size_t>    bank_n;  // their ABI byte sizes (axera_cmm_alloc rounds c->size up to 128)
    axera_cmm                        ST2, CQ2, CK2, CV2;  // the second recurrent-state set (prefill outputs)
    axera_cmm                        ONES, ZERO;           // global scales (1) and zero points (0) of the prefill kernels
    call                             dec;
    call                             pad;
    std::vector<call>                pre_attn, pre_moe;  // per layer
    axera_cmm DX, DA, DOUT, DPOS, INDPTR, INDICES, LASTLEN, CLOC, PX, PA, POS, VALID;
    // the recurrent state at the first cell of the current page (snap_cell), and the inputs of that page's cells so far
    std::vector<uint8_t>     snap;
    int64_t                  snap_cell = -1;
    int64_t                  state_cell = -1;  // tokens processed: the current recurrent state is before this cell
    std::vector<ggml_bf16_t> page_x;
    std::vector<int32_t>     page_p;  // [cells][4]
    output_head              head;

    // the fused prefill (roles prefill.gdn_moe / prefill.attn_moe, moe.slots, moe.combine): per layer and state parity
    // (0: the current recurrent states in, the second set out; 1 the reverse) its call, the IO entries rebound per page
    struct fcall : call {
        struct at {
            bool out = false;
            int  i   = -1;
        };
        at x, h2, base, ti, tw;  // the per-page IO entries
        AX_ENGINE_IO_BUFFER_T & operator[](const at & a) { return a.out ? out[a.i] : in[a.i]; }
    };
    bool                     fused = false;
    std::vector<fcall>       pre_f[2];
    std::vector<std::array<AX_ENGINE_IO_BUFFER_T, 4>> exp_w;  // per layer: e_gw, e_gs, e_dw, e_ds (bank slices)
    const kernel *           slots_k = nullptr, *comb_k = nullptr;
    axera_cmm                FPOS, XS, H2S, BASES, AOUT, TKI, TKW, SOUT, SLOTE, SLOTT, SLOTW, SNS, SGR, LOC, NTOK;

    void         init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void         init_fused();
    void         prefill_fused(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    void         experts(int il, int nt);
    void         snap_layer(int il, int set);
    void         fill_pack(const kernel & dk, const rope & rp);
    void         fill(const std::map<std::string, ggml_tensor *> & t, const kernel & dk, const rope & rp);
    axera_cmm &  B(const std::string & n) { return banks.at(n); }
    size_t       logical(const std::string & n) const {
        auto it = bank_n.find(n);
        return it == bank_n.end() ? 0 : it->second;
    }
    // entry idx of a bank (a layer's slice: [10 blocks] of a phase's, [30] of the delta-net states)
    AX_ENGINE_IO_BUFFER_T slice(const std::string & bank, int idx, uint32_t bytes) {
        const axera_cmm & c = B(bank);
        const size_t      n = bank.find("state_in") != std::string::npos ? 3 * NB : NB;
        const size_t      z = logical(bank);
        if (!z || z % n || z / n != bytes) {
            GGML_ABORT("axera qwen35_moe_sgl: slice(%s) idx=%d n=%zu size=%zu bytes=%u\n", bank.c_str(), idx, n, z,
                       bytes);
        }
        return axera_io(c.get(), (char *) c->vir + (size_t) idx * bytes, bytes);
    }
    // a sub-field (e.g. the first FP32 scale) of a bank's per-block row
    AX_ENGINE_IO_BUFFER_T part(const std::string & bank, int idx, uint32_t bytes) {
        const axera_cmm & c = B(bank);
        const size_t      n = bank.find("state_in") != std::string::npos ? 3 * NB : NB;
        const size_t      z = logical(bank);
        if (!z || z % n || z / n < bytes) {
            GGML_ABORT("axera qwen35_moe_sgl: part(%s) idx=%d n=%zu size=%zu bytes=%u\n", bank.c_str(), idx, n, z,
                       bytes);
        }
        return axera_io(c.get(), (char *) c->vir + (size_t) idx * (z / n), bytes);
    }
    void         states_swap();
    void         zero_state();
    void         save_state(int64_t cell);
    void         load_state();
    void         paging(int64_t hist, int64_t write_cell);
    void         prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    void         decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h);
};

// Debug-only fallback: allocate the decode kernel's runtime banks and load an offline NVFP4 pack. The pack contains
// one file per non-state input and is shared by decode and prefill; state and paged KV banks start zero. The release
// path is fill() below; this is used only when AXERA_SGL35_BANKS=1 and a pack is present.
void qwen35_moe_sgl::fill_pack(const kernel & dk, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|moe)_(.+)");
    std::string           dir = man.dir + "/banks/";
    if (!std::ifstream(dir + "b0_attn_w_q.bin", std::ios::binary)) {
        dir = man.dir + "/";
    }
    for (const auto & d : dk.desc->in) {
        bank_n[d.name] = d.bytes;
    }
    for (const auto & d : dk.desc->in) {
        std::smatch mt;
        const bool  layer_bank = std::regex_match(d.name, mt, re);
        if (!layer_bank) {
            if (d.name == "b3_kv") {
                banks[d.name] = axera_cmm_alloc(d.bytes);
            }
            continue;
        }
        axera_cmm & c = banks[d.name] = axera_cmm_alloc(d.bytes);
        const std::string n = mt[3];
        if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in" || n == "kv") {
            continue;
        }
        if (n == "cos" || n == "sin") {
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT((size_t) R == rp.inv_freq.size());
            for (int64_t b = 0; b < NB; ++b) {
                float * dst = (float *) c->vir + (size_t) b * P * R;
                for (int64_t q = 0; q < P; ++q) {
                    for (int64_t i = 0; i < R; ++i) {
                        const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                        dst[q * R + i] = n == "cos" ? (float) std::cos(th) : (float) std::sin(th);
                    }
                }
            }
            c->flush();
            continue;
        }
        std::ifstream f(dir + d.name + ".bin", std::ios::binary);
        if (!f) {
            GGML_ABORT("axera qwen35_moe_sgl: missing bank %s%s.bin\n", dir.c_str(), d.name.c_str());
        }
        f.read((char *) c->vir, d.bytes);
        if (!f || (size_t) f.gcount() != d.bytes) {
            GGML_ABORT("axera qwen35_moe_sgl: short bank %s%s.bin\n", dir.c_str(), d.name.c_str());
        }
        c->flush();
    }
}

// the decode kernel's banks from the published GGUF: b<phase>_attn_* (delta net, phases 0-2; gated attention, phase 3),
// b<phase>_moe_*, each [10 blocks, ...] (layer = 4 * block + phase), and the state / KV pools. Every projection and
// expert row is quantized here, in memory, to the kernels' NVFP4 layout (E2M1 pairs, one E4M3 scale per 16 inputs and
// the FP32 global scale2): the GGUF's own quantization is never copied, so Q8_0 is a faithful source without a second
// lossy requantization. One layer (and one expert) at a time; the mapped pages are dropped as soon as a tensor has
// been consumed.
void qwen35_moe_sgl::fill(const std::map<std::string, ggml_tensor *> & t, const kernel & dk, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|moe)_(.+)");
    std::vector<const ggml_tensor *> read;
    std::map<std::string, int>       dtype;  // AX_ENGINE dtype of each bank (3 = FP32, else BF16)
    const bool                       keep = axera_keep_pages();
    auto T = [&](int il, const std::string & g) -> const ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen35_35b_a3b: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    // allocate every bank; the state / KV pools stay zero, the RoPE table is generated
    for (const auto & d : dk.desc->in) {
        bank_n[d.name] = d.bytes;
        dtype[d.name]  = d.dtype;
    }
    for (const auto & d : dk.desc->in) {
        if (banks.count(d.name)) {
            continue;
        }
        std::smatch mt;
        if (!std::regex_match(d.name, mt, re)) {
            if (d.name == "b3_kv") {  // the KV cache pool: zero
                banks[d.name] = axera_cmm_alloc(d.bytes);
            }
            continue;
        }
        axera_cmm & c = banks[d.name] = axera_cmm_alloc(d.bytes);
        const std::string n = mt[3];
        if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in" || n == "kv") {
            continue;  // the state pools: zero
        }
        if (n == "cos_sin_cache") {  // [positions][cos | sin of ROT / 2] f32
            const int64_t P = d.shape[1], R = d.shape[2] / 2;
            GGML_ASSERT((size_t) R == rp.inv_freq.size() && d.dtype == 3);
            for (int64_t q = 0; q < P; ++q) {
                for (int64_t i = 0; i < R; ++i) {
                    const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                    ((float *) c->vir)[q * 2 * R + i]     = (float) std::cos(th);
                    ((float *) c->vir)[q * 2 * R + R + i] = (float) std::sin(th);
                }
            }
            c->flush();
            continue;
        }
        GGML_ASSERT(d.shape[0] == NB && d.bytes % NB == 0);
    }
    auto slot = [&](const std::string & n, int b) -> uint8_t * {
        if (!banks.count(n) || !bank_n.count(n)) {
            GGML_ABORT("axera qwen35_moe_sgl: bank %s is not in the decode ABI\n", n.c_str());
        }
        const axera_cmm & c = banks.at(n);
        const size_t      bb = bank_n.at(n) / NB;
        GGML_ASSERT(bb * NB == bank_n.at(n));
        return (uint8_t *) c->vir + (size_t) b * bb;
    };
    // floats into a bank slot (BF16 or FP32 by the ABI)
    auto put = [&](const std::string & n, int b, const std::vector<float> & v, float scale = 1.0f) {
        if (!bank_n.count(n) || !dtype.count(n)) {
            GGML_ABORT("axera qwen35_moe_sgl: bank %s is not in the decode ABI\n", n.c_str());
        }
        const size_t bb = bank_n.at(n) / NB;
        uint8_t *    d  = slot(n, b);
        if (dtype.at(n) == 3) {
            if (v.size() * 4 != bb) {
                GGML_ABORT("axera qwen35_moe_sgl: %s f32 %zu floats, bank slot %zu bytes\n", n.c_str(), v.size(), bb);
            }
            for (size_t i = 0; i < v.size(); ++i) {
                ((float *) d)[i] = v[i] * scale;
            }
        } else {
            if (v.size() * 2 != bb) {
                GGML_ABORT("axera qwen35_moe_sgl: %s bf16 %zu floats, bank slot %zu bytes\n", n.c_str(), v.size(), bb);
            }
            for (size_t i = 0; i < v.size(); ++i) {
                ((ggml_bf16_t *) d)[i] = ggml_fp32_to_bf16(v[i] * scale);
            }
        }
    };
    // rows r0 .. r0 + nr of a GGUF tensor -> this block's NVFP4 code / scale banks, its global scale at scale2 index
    auto put_nv = [&](const std::string & pre, const std::string & base, const ggml_tensor * w, int64_t r0, int64_t nr,
                      int b, int64_t row0, int64_t s2i, const std::vector<int64_t> * row_src = nullptr,
                      const std::vector<int64_t> * blk_src = nullptr, const std::string & sname = "",
                      const std::string & zname = "") -> float {
        const std::string qn = pre + base;
        const std::string sn = sname.empty() ? qn + "_scale" : sname;
        const std::string zn = zname.empty() ? qn + "_scale2" : zname;
        const int64_t     K  = w->ne[0];
        uint8_t *         q  = slot(qn, b);
        uint8_t *         s  = slot(sn, b);
        GGML_ASSERT((size_t) (row0 + nr) * (K / 2) <= bank_n.at(qn) / NB);
        GGML_ASSERT((size_t) (row0 + nr) * (K / 16) <= bank_n.at(sn) / NB);
        GGML_ASSERT((size_t) (s2i + 1) * sizeof(float) <= bank_n.at(zn) / NB);
        const float scale2 = row_src || blk_src
                                 ? axera_quantize_nvfp4_slice_perm(w, r0, nr, row_src, blk_src,
                                                                   q + (size_t) row0 * (K / 2), K / 2,
                                                                   s + (size_t) row0 * (K / 16), K / 16)
                                 : axera_quantize_nvfp4_slice(w, r0, nr, q + (size_t) row0 * (K / 2), K / 2,
                                                              s + (size_t) row0 * (K / 16), K / 16);
        ((float *) slot(zn, b))[s2i] = scale2;
        return scale2;
    };
    // GGUF tiles the 32 GDN value heads; the kernels want HF grouped order. row_src undoes it for output rows,
    // blk_src for the 16-input blocks of an input-column permutation.
    auto head_rows = [&](int64_t rows) {
        std::vector<int64_t> m(rows);
        const int64_t        hr = rows / VH;
        for (int64_t r = 0; r < rows; ++r) {
            m[r] = tiled((int) (r / hr)) * hr + r % hr;
        }
        return m;
    };
    auto head_blocks = [&](int64_t K) {
        std::vector<int64_t> m(K / 16);
        const int64_t        hb = K / VH / 16;
        for (int64_t g = 0; g < K / 16; ++g) {
            m[g] = tiled((int) (g / hb)) * hb + g % hb;
        }
        return m;
    };
    auto head_perm = [&](const std::vector<float> & v) {  // GGUF's tiled value-head order -> HF grouped order
        std::vector<float> u(VH);
        for (int h = 0; h < VH; ++h) {
            u[h] = v[tiled(h)];
        }
        return u;
    };

    for (int b = 0; b < NB; ++b) {
        for (int phase = 0; phase < 4; ++phase) {
            const int         il = 4 * b + phase;
            const std::string pa = "b" + std::to_string(phase) + "_attn_";
            const std::string pm = "b" + std::to_string(phase) + "_moe_";
            read.clear();
            if (phase < 3) {  // gated delta net
                put(pa + "input_norm_w", b, f32_of(T(il, "attn_norm.weight")));
                const ggml_tensor * qkv = T(il, "attn_qkv.weight");
                GGML_ASSERT(qkv->ne[0] % 16 == 0 && qkv->ne[1] == 2048 + 2048 + 4096);
                const std::vector<int64_t> vr = head_rows(4096);
                put_nv(pa, "w_q", qkv, 0, 2048, b, 0, 0);
                put_nv(pa, "w_k", qkv, 2048, 2048, b, 0, 0);
                put_nv(pa, "w_v", qkv, 4096, 4096, b, 0, 0, &vr);
                if (!keep) {
                    axera_drop_mapped(qkv);
                }
                {
                    const ggml_tensor * w = T(il, "attn_gate.weight");
                    const std::vector<int64_t> m = head_rows(ggml_nrows(w));
                    put_nv(pa, "w_z", w, 0, ggml_nrows(w), b, 0, 0, &m);
                }
                for (const auto & [base, g] : { std::pair<const char *, const char *>{ "w_a", "ssm_alpha.weight" },
                                                { "w_b", "ssm_beta.weight" } }) {
                    const ggml_tensor * w = T(il, g);
                    const std::vector<int64_t> m = head_rows(ggml_nrows(w));
                    put_nv(pa, base, w, 0, ggml_nrows(w), b, 0, 0, &m);
                }
                {
                    const ggml_tensor * w = T(il, "ssm_out.weight");
                    const std::vector<int64_t> m = head_blocks(w->ne[0]);
                    put_nv(pa, "w_o", w, 0, ggml_nrows(w), b, 0, 0, nullptr, &m);
                }
                const std::vector<float> a = f32_of(T(il, "ssm_a"));
                std::vector<float>       al(VH);
                for (int h = 0; h < VH; ++h) {
                    al[h] = logf(-a[tiled(h)]);  // GGUF ssm_a = -exp(A_log)
                }
                put(pa + "a_log", b, al);
                put(pa + "dt_bias", b, head_perm(f32_of(T(il, "ssm_dt.bias"))));
                put(pa + "o_norm_w", b, f32_of(T(il, "ssm_norm.weight")));
                const ggml_tensor *      w  = T(il, "ssm_conv1d.weight");
                const std::vector<float> cv = f32_of(w);
                const int64_t            tp = w->ne[0];
                for (char c : { 'q', 'k', 'v' }) {
                    const int64_t C  = (int64_t) bank_n.at(pa + std::string("w_") + c + "_conv") / NB / (tp * 2);
                    const int64_t v0 = w->ne[1] - C;
                    std::vector<float> u((size_t) C * tp);
                    for (int64_t r = 0; r < C; ++r) {
                        const int64_t hr  = C / VH;
                        const int64_t src = c == 'v' ? v0 + tiled((int) (r / hr)) * hr + r % hr
                                                     : (c == 'q' ? 0 : C) + r;
                        memcpy(&u[(size_t) r * tp], &cv[(size_t) src * tp], tp * sizeof(float));
                    }
                    put(pa + std::string("w_") + c + "_conv", b, u);
                }
            } else {  // gated attention
                put(pa + "w_norm_attn", b, f32_of(T(il, "attn_norm.weight")));
                put(pa + "w_norm_q", b, f32_of(T(il, "attn_q_norm.weight")), 1.0f / std::sqrt(256.0f));
                put(pa + "w_norm_k", b, f32_of(T(il, "attn_k_norm.weight")));
                for (const char * n : { "q", "k", "v", "o" }) {
                    const std::string g = std::string("attn_") + (n[0] == 'o' ? "output" : n) + ".weight";
                    const ggml_tensor * w = T(il, g);
                    put_nv(pa, std::string("w_") + n, w, 0, ggml_nrows(w), b, 0, 0, nullptr, nullptr,
                           pa + "w_scale_" + n, pa + "w_scale2_" + n);
                }
            }
            // MoE of this layer
            put(pm + "input_norm_w", b, f32_of(T(il, "post_attention_norm.weight")));
            put(pm + "shared_expert_gate", b, f32_of(T(il, "ffn_gate_inp_shexp.weight")));
            put(pm + "w_route", b, f32_of(T(il, "ffn_gate_inp.weight")));
            put(pm + "w_scale_route", b, std::vector<float>(256, 1.0f));
            ((float *) slot(pm + "w_scale2_route", b))[0] = 1.0f;
            {
                const ggml_tensor * g = T(il, "ffn_gate_exps.weight");
                const ggml_tensor * u = T(il, "ffn_up_exps.weight");
                GGML_ASSERT(g->ne[0] == u->ne[0] && g->ne[1] == u->ne[1] && g->ne[2] == u->ne[2]);
                const int64_t F = g->ne[1], E = g->ne[2];
                for (int64_t e = 0; e < E; ++e) {
                    put_nv(pm, "w_gate", g, e * F, F, b, e * 2 * F, e * 2, nullptr, nullptr, pm + "w_scale_gate",
                           pm + "w_scale2_gate");
                    put_nv(pm, "w_gate", u, e * F, F, b, e * 2 * F + F, e * 2 + 1, nullptr, nullptr,
                           pm + "w_scale_gate", pm + "w_scale2_gate");
                }
                if (!keep) {
                    axera_drop_mapped(g);
                    axera_drop_mapped(u);
                }
                const ggml_tensor * d = T(il, "ffn_down_exps.weight");
                const int64_t Rd = d->ne[1], Ed = d->ne[2];
                for (int64_t e = 0; e < Ed; ++e) {
                    put_nv(pm, "w_down", d, e * Rd, Rd, b, e * Rd, e, nullptr, nullptr, pm + "w_scale_down",
                           pm + "w_scale2_down");
                }
                if (!keep) {
                    axera_drop_mapped(d);
                }
            }
            for (const char * n : { "gate", "up", "down" }) {
                const std::string g = std::string("ffn_") + n + "_shexp.weight";
                const ggml_tensor * w = T(il, g);
                const std::string base = std::string("w_shared_expert_mlp_") + n;
                put_nv(pm, base, w, 0, ggml_nrows(w), b, 0, 0, nullptr, nullptr,
                       pm + "w_scale_shared_expert_mlp_" + n, pm + "w_scale2_shared_expert_mlp_" + n);
            }
            if (!keep) {
                for (const ggml_tensor * w : read) {
                    axera_drop_mapped(w);
                }
            }
        }
    }
    for (auto & [n, c] : banks) {
        c->flush();
    }
}

void qwen35_moe_sgl::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L == 4 * NB);
    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen35_35b_a3b: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        ord[il] = g ? ng++ : na++;
    }
    const kernel & dk = man.get("decode");
    const kernel & gk = man.get("prefill.gdn");
    const kernel & ak = man.get("prefill.attn");
    const kernel & mk = man.get("prefill.moe");
    banks.clear();
    {
        axera_timer tb("init: banks");
        const rope  rq(rope_node, names.find("inp_pos"), 1);
        // Release path: build every NVFP4 bank in memory from the GGUF rows. The offline pack is a debug fallback
        // only, explicit (AXERA_SGL35_BANKS=1) and absent from the release package.
        const char * want_pack = std::getenv("AXERA_SGL35_BANKS");
        std::string  pack_dir  = man.dir + "/banks/";
        std::ifstream probe(pack_dir + "b0_attn_w_q.bin", std::ios::binary);
        if (!probe) {
            pack_dir = man.dir + "/";
            probe.open(pack_dir + "b0_attn_w_q.bin", std::ios::binary);
        }
        if (want_pack && std::string(want_pack) != "0" && probe) {
            GGML_LOG_INFO("axera qwen35_35b_a3b: NVFP4 banks from the debug pack %s\n", pack_dir.c_str());
            fill_pack(dk, rq);
        } else {
            fill(t, dk, rq);
        }
    }
    ST2  = axera_cmm_alloc(B("b0_attn_state_in")->size);
    CQ2  = axera_cmm_alloc(B("b0_attn_q_conv_state_in")->size);
    CK2  = axera_cmm_alloc(B("b0_attn_k_conv_state_in")->size);
    CV2  = axera_cmm_alloc(B("b0_attn_v_conv_state_in")->size);
    snap.assign(ST2->size + CQ2->size + CK2->size + CV2->size, 0);
    snap_cell  = -1;
    state_cell = 0;
    page_x.clear();
    page_p.clear();
    ONES = axera_cmm_alloc(4096);
    for (int i = 0; i < 1024; ++i) {
        ((float *) ONES->vir)[i] = 1.0f;
    }
    ONES->flush();
    ZERO = axera_cmm_alloc(4096);
    const int NP = (int) logical("b3_kv") / (NB * 2 * PG * 2 * 256 * 2);  // pages
    DX      = axera_cmm_alloc(dk.in_bytes("x"));
    DA      = axera_cmm_alloc(dk.in_bytes("attention"));
    DOUT    = axera_cmm_alloc(dk.out_bytes("output"));
    DPOS    = axera_cmm_alloc(dk.in_bytes("position_ids"));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc((size_t) NP * sizeof(int32_t));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    PX      = axera_cmm_alloc((size_t) WN * H * sizeof(ggml_bf16_t));
    PA      = axera_cmm_alloc((size_t) WN * H * sizeof(ggml_bf16_t));
    POS     = axera_cmm_alloc(ak.in_bytes("position_ids"));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    AX      = (int) (ak.in_bytes("position_ids") / (PG * sizeof(int32_t)));
    GGML_ASSERT((AX == 1 || AX == 3) && (int64_t) NP * PG == man.context && dk.in_bytes("indices") == INDICES->size);
    for (int i = 0; i < NP; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    // decode: the banks, its own rows and paging
    dec   = {};
    dec.k = &dk;
    for (const auto & d : dk.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &DX : n == "attention" ? &DA : n == "position_ids" ? &DPOS : n == "indptr" ? &INDPTR :
                                n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN : n == "cache_location" ? &CLOC :
                                &B(n);
        dec.in.push_back(axera_io(*b, d.bytes));
    }
    dec.out.push_back(axera_io(DOUT, dk.out_bytes("output")));

    // prefill: per layer, the slices of the decode banks of its phase and block
    pre_attn.assign(L, {});
    pre_moe.assign(L, {});
    for (int il = 0; il < L; ++il) {
        const int         phase = il % 4, blk = il / 4;
        const std::string pa = "b" + std::to_string(phase) + "_attn_", pm = "b" + std::to_string(phase) + "_moe_";
        call &            c = pre_attn[il];
        c.k                 = phase == 3 ? &ak : &gk;
        for (const auto & d : c.k->desc->in) {
            const std::string & n = d.name;
            if (n == "x") {
                c.in.push_back(axera_io(PX, d.bytes));
            } else if (n == "position_ids") {
                c.in.push_back(axera_io(POS, d.bytes));
            } else if (n == "valid_tokens") {
                c.in.push_back(axera_io(VALID, d.bytes));
            } else if (n == "indptr" || n == "indices" || n == "cache_location") {
                c.in.push_back(axera_io(n == "indptr" ? INDPTR : n == "indices" ? INDICES : CLOC, d.bytes));
            } else if (n.find("scale2") != std::string::npos) {
                c.in.push_back(part(pa + n, blk, d.bytes));
            } else if (n.rfind("w_zeropoint_", 0) == 0) {
                c.in.push_back(axera_io(ZERO, d.bytes));
            } else if (n == "paged_kv_cache") {
                c.in.push_back(slice("b3_kv", ord[il], d.bytes));
            } else if (n == "cos_sin_cache") {
                c.in.push_back(axera_io(B("b3_attn_cos_sin_cache"), d.bytes));
            } else if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in") {
                c.in.push_back(slice("b0_attn_" + n, ord[il], d.bytes));
            } else {
                static const std::map<std::string, std::string> A = { { "x_norm_w", "w_norm_attn" }, { "q_norm_w", "w_norm_q" },
                                                                       { "k_norm_w", "w_norm_k" } };
                c.in.push_back(slice(pa + (A.count(n) ? A.at(n) : n), blk, d.bytes));
            }
        }
        for (const auto & d : c.k->desc->out) {
            const std::string & n = d.name;
            if (n == "output" || n == "residual_out") {
                c.out.push_back(axera_io(PA, d.bytes));
            } else {  // <x>_out: the second state set (swapped after the page)
                const std::string in = n.substr(0, n.size() - 4) + "_in";
                const axera_cmm & s2 = in == "state_in" ? ST2 : in == "q_conv_state_in" ? CQ2 : in == "k_conv_state_in" ? CK2 : CV2;
                c.out.push_back(axera_io(s2.get(), (char *) s2->vir + (size_t) ord[il] * d.bytes, d.bytes));
            }
        }
        call & m = pre_moe[il];
        m.k      = &mk;
        for (const auto & d : mk.desc->in) {
            const std::string & n = d.name;
            if (n == "x") {
                m.in.push_back(axera_io(PA, d.bytes));
            } else if (n == "w_scale2_route" || n.find("scale2_shared") != std::string::npos) {
                m.in.push_back(part(pm + n, blk, d.bytes));
            } else {  // w_up / w_scale_up / w_scale2_up: the fused gate | up tensor (rows 0:512 gate, 512:1024 up)
                std::string b = n == "w_up" ? "w_gate" : n == "w_scale_up" ? "w_scale_gate" : n == "w_scale2_up" ? "w_scale2_gate" : n;
                m.in.push_back(slice(pm + b, blk, d.bytes));
            }
        }
        m.out.push_back(axera_io(PX, mk.desc->out[0].bytes));
    }
    pad   = {};
    pad.k = &man.get("prefill.padding");
    for (const auto & d : pad.k->desc->in) {
        if (d.name == "x" || d.name == "activation") {
            pad.in.push_back(axera_io(PA, d.bytes));
        } else if (d.name == "token_count") {
            pad.in.push_back(axera_io(VALID, d.bytes));
        } else {
            GGML_ABORT("axera qwen35_moe_sgl: unexpected padding input %s\n", d.name.c_str());
        }
    }
    fused = man.has("prefill.gdn_moe") && man.has("prefill.attn_moe") && man.has("moe.slots") && man.has("moe.combine");
    if (fused) {
        init_fused();
    }
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen35_35b_a3b: %d layers, banks (weights + state) %.2f GiB, %lld cells\n", L, bb / 1073741824.0,
                  (long long) man.context);
}

// the prefill kernels' new recurrent states become the current ones (pointer swap of the pools, rebinding the IO)
void qwen35_moe_sgl::states_swap() {
    for (auto [n, s2] : std::initializer_list<std::pair<const char *, axera_cmm *>>{
             { "b0_attn_state_in", &ST2 }, { "b0_attn_q_conv_state_in", &CQ2 }, { "b0_attn_k_conv_state_in", &CK2 },
             { "b0_attn_v_conv_state_in", &CV2 } }) {
        axera_cmm & cur = B(n);
        auto *      a = cur.get(), *b = s2->get();
        std::swap(cur, *s2);
        // every IO bound to either pool now points to the other one
        auto rebind = [&](std::vector<AX_ENGINE_IO_BUFFER_T> & v) {
            for (auto & io : v) {
                const char * p = (const char *) io.pVirAddr;
                if (p >= (const char *) a->vir && p < (const char *) a->vir + a->size) {
                    io = axera_io(b, (char *) b->vir + (p - (const char *) a->vir), io.nSize);
                } else if (p >= (const char *) b->vir && p < (const char *) b->vir + b->size) {
                    io = axera_io(a, (char *) a->vir + (p - (const char *) b->vir), io.nSize);
                }
            }
        };
        rebind(dec.in);
        for (auto & c : pre_attn) {
            rebind(c.in);
            rebind(c.out);
        }
        for (auto & v : pre_f) {
            for (auto & c : v) {
                rebind(c.in);
                rebind(c.out);
            }
        }
    }
}

void qwen35_moe_sgl::zero_state() {
    axera_timer tl("state zero");
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        memset(B(n)->vir, 0, B(n)->size);
        B(n)->flush();
    }
    snap_cell  = -1;
    state_cell = 0;
    page_x.clear();
    page_p.clear();
}

void qwen35_moe_sgl::save_state(int64_t cell) {
    GGML_ASSERT(cell % PG == 0);
    axera_timer tl("state save");
    size_t o = 0;
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        B(n)->invalidate();
        memcpy(snap.data() + o, B(n)->vir, B(n)->size);
        o += B(n)->size;
    }
    snap_cell = cell;
    page_x.clear();
    page_p.clear();
}

void qwen35_moe_sgl::load_state() {
    axera_timer tl("state load");
    size_t o = 0;
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        memcpy(B(n)->vir, snap.data() + o, B(n)->size);
        B(n)->flush();
        o += B(n)->size;
    }
}

// the fused prefill's calls: per layer and state parity the layer kernel's IO (the decode banks' slices of the layer, its
// recurrent states in the set of the parity and out to the other), the routed experts' bank slices, the group buffers
void qwen35_moe_sgl::init_fused() {
    const kernel & gk = man.get("prefill.gdn_moe");
    const kernel & ak = man.get("prefill.attn_moe");
    slots_k           = &man.get("moe.slots");
    comb_k            = &man.get("moe.combine");
    const size_t xr   = (size_t) H * sizeof(ggml_bf16_t);  // bytes per row
    const size_t NR   = comb_k->in_bytes("base") / xr;     // rows of a group
    const size_t K    = gk.out_bytes("topk_i") / (PG * sizeof(int32_t));
    GGML_ASSERT(NR % PG == 0 && slots_k->in_bytes("h2") == NR * xr && comb_k->out_bytes("out") == NR * xr &&
                comb_k->in_bytes("loc") == K * NR * sizeof(int32_t) && gk.out_bytes("topk_w") == PG * K * sizeof(float));
    XS    = axera_cmm_alloc(NR * xr);
    H2S   = axera_cmm_alloc(NR * xr);
    BASES = axera_cmm_alloc(NR * xr);
    AOUT  = axera_cmm_alloc(PG * xr);
    TKI   = axera_cmm_alloc(NR * K * sizeof(int32_t));
    TKW   = axera_cmm_alloc(NR * K * sizeof(float));
    // slot rows: the combine kernel's, and one slots call's past them (a call's output is a whole window)
    SOUT          = axera_cmm_alloc(comb_k->in_bytes("sout") + slots_k->out_bytes("out"));
    const int NCL = (int) (2 * comb_k->in_bytes("sout") / slots_k->out_bytes("out") + 4);  // slots calls per layer, at most
    SLOTE         = axera_cmm_alloc((size_t) NCL * slots_k->in_bytes("slot_exp"));
    SLOTT         = axera_cmm_alloc((size_t) NCL * slots_k->in_bytes("slot_tok"));
    SLOTW         = axera_cmm_alloc((size_t) NCL * slots_k->in_bytes("rw"));
    SNS           = axera_cmm_alloc((size_t) NCL * sizeof(int32_t));
    SGR           = axera_cmm_alloc((size_t) NCL * sizeof(int32_t));
    LOC           = axera_cmm_alloc(comb_k->in_bytes("loc"));
    NTOK          = axera_cmm_alloc(sizeof(int32_t));

    auto state = [&](const std::string & in, int il, int set, uint32_t bytes) {  // a delta-net state slice of set 0 / 1
        if (set == 0) {
            return slice("b0_attn_" + in, ord[il], bytes);
        }
        const axera_cmm & s2 = in == "state_in" ? ST2 : in == "q_conv_state_in" ? CQ2 : in == "k_conv_state_in" ? CK2 : CV2;
        return axera_io(s2.get(), (char *) s2->vir + (size_t) ord[il] * bytes, bytes);
    };
    for (int par = 0; par < 2; ++par) {
        pre_f[par].assign(L, {});
        for (int il = 0; il < L; ++il) {
            const int         phase = il % 4, blk = il / 4;
            const std::string pa = "b" + std::to_string(phase) + "_attn_", pm = "b" + std::to_string(phase) + "_moe_";
            fcall &           c = pre_f[par][il];
            c.k                 = phase == 3 ? &ak : &gk;
            auto bind = [&](const io_desc & d, bool out) {
                const std::string &                 n = d.name;
                std::vector<AX_ENGINE_IO_BUFFER_T> & v = out ? c.out : c.in;
                const fcall::at                     here{ out, (int) v.size() };
                const bool st_in  = n == "state_in" || (n.size() > 9 && n.compare(n.size() - 9, 9, "_state_in") == 0);
                const bool st_out = n == "state_out" || (n.size() > 10 && n.compare(n.size() - 10, 10, "_state_out") == 0);
                if (n == "x") {
                    c.x = here;
                    v.push_back(axera_io(XS, PG * xr));
                } else if (n == "h2" || n == "base") {
                    (n == "h2" ? c.h2 : c.base) = here;
                    v.push_back(axera_io(n == "h2" ? H2S : BASES, PG * xr));
                } else if (n == "topk_i" || n == "topk_w") {
                    (n == "topk_i" ? c.ti : c.tw) = here;
                    v.push_back(axera_io(n == "topk_i" ? TKI : TKW, d.bytes));
                } else if (n == "output" || n == "residual_out") {  // the attention's own output: unused
                    v.push_back(axera_io(AOUT, d.bytes));
                } else if (st_in) {
                    v.push_back(state(n, il, par, d.bytes));
                } else if (st_out) {  // (an output the kernel writes in place, declared as either)
                    v.push_back(state(n.substr(0, n.size() - 4) + "_in", il, 1 - par, d.bytes));
                } else if (out) {
                    GGML_ABORT("axera qwen35_35b_a3b: %s: unexpected output %s\n", c.k->desc->role.c_str(), n.c_str());
                } else if (n == "position_ids") {  // (its own: the fused attention takes 3 M-RoPE axes)
                    if (!FPOS) {
                        FPOS = axera_cmm_alloc(d.bytes);
                    }
                    GGML_ASSERT(FPOS->size == d.bytes);
                    v.push_back(axera_io(FPOS, d.bytes));
                } else if (n == "valid_tokens") {
                    v.push_back(axera_io(VALID, d.bytes));
                } else if (n == "indptr" || n == "indices" || n == "cache_location") {
                    v.push_back(axera_io(n == "indptr" ? INDPTR : n == "indices" ? INDICES : CLOC, d.bytes));
                } else if (n.find("scale2") != std::string::npos) {
                    v.push_back(axera_io(ONES, d.bytes));
                } else if (n.rfind("w_zeropoint_", 0) == 0) {
                    v.push_back(axera_io(ZERO, d.bytes));
                } else if (n == "paged_kv_cache") {
                    v.push_back(slice("b3_kv", ord[il], d.bytes));
                } else if (n == "cos_sin_cache") {
                    v.push_back(axera_io(B("b3_attn_cos_sin_cache"), d.bytes));
                } else if (n == "moe_input_norm_w" || (phase == 3 && n == "input_norm_w")) {  // the post-attention norm
                    v.push_back(slice(pm + "input_norm_w", blk, d.bytes));
                } else if (n.find("shared_expert") != std::string::npos || n.find("route") != std::string::npos) {
                    v.push_back(slice(pm + n, blk, d.bytes));
                } else {
                    static const std::map<std::string, std::string> A = { { "x_norm_w", "w_norm_attn" }, { "q_norm_w", "w_norm_q" },
                                                                           { "k_norm_w", "w_norm_k" } };
                    v.push_back(slice(pa + (A.count(n) ? A.at(n) : n), blk, d.bytes));
                }
            };
            for (const auto & d : c.k->desc->in) {
                bind(d, false);
            }
            for (const auto & d : c.k->desc->out) {
                bind(d, true);
            }
            GGML_ASSERT(c.x.i >= 0 && c.h2.i >= 0 && c.base.i >= 0 && c.ti.i >= 0 && c.tw.i >= 0);
        }
    }
    exp_w.assign(L, {});
    for (int il = 0; il < L; ++il) {
        const std::string pm = "b" + std::to_string(il % 4) + "_moe_";
        exp_w[il] = { slice(pm + "w_gate", il / 4, slots_k->in_bytes("e_gw")), slice(pm + "w_scale_gate", il / 4, slots_k->in_bytes("e_gs")),
                      slice(pm + "w_down", il / 4, slots_k->in_bytes("e_dw")), slice(pm + "w_scale_down", il / 4, slots_k->in_bytes("e_ds")) };
    }
}

// layer il's recurrent state, in set 0 (the current one) or 1, into the restart point (as save_state would)
void qwen35_moe_sgl::snap_layer(int il, int set) {
    size_t o = 0;
    for (auto [n, s2] : std::initializer_list<std::pair<const char *, axera_cmm *>>{
             { "b0_attn_state_in", &ST2 }, { "b0_attn_q_conv_state_in", &CQ2 }, { "b0_attn_k_conv_state_in", &CK2 },
             { "b0_attn_v_conv_state_in", &CV2 } }) {
        const axera_cmm & c  = set == 0 ? B(n) : *s2;
        const size_t      lb = logical(n) / (3 * NB), lo = (size_t) ord[il] * lb;
        c->invalidate((char *) c->vir + lo, lb);
        memcpy(snap.data() + o + lo, (const char *) c->vir + lo, lb);
        o += c->size;
    }
}

// prefill with the fused kernels: groups of up to NR rows (pages) from cell c0 (a page's first), each layer by layer:
// its pages through the layer kernel (the states alternating between the two sets page by page), then its experts
void qwen35_moe_sgl::prefill_fused(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    const size_t xr = (size_t) H * sizeof(ggml_bf16_t);
    const int    NR = (int) (XS->size / xr);
    for (int g0 = 0; g0 < n; g0 += NR) {
        const int  rows = std::min(NR, n - g0), np = (rows + PG - 1) / PG, vl = rows - (np - 1) * PG;
        const bool part = vl < PG;  // (the last group's last page only)
        memcpy(XS->vir, x + (size_t) g0 * H, rows * xr);
        memset((char *) XS->vir + rows * xr, 0, ((size_t) np * PG - rows) * xr);
        XS->flush(XS->vir, (size_t) np * PG * xr);
        axera_timer tg("prefill (group)");
        for (int il = 0; il < L; ++il) {
            for (int p = 0; p < np; ++p) {
                const int     r0 = g0 + p * PG, v = p == np - 1 ? vl : PG;
                const int64_t cell = c0 + r0;
                if (part && p == np - 1 && il % 4 != 3) {  // the state before a partial page: a later batch restarts there
                    snap_layer(il, p % 2);
                }
                if (il % 4 == 3) {  // the attention's positions, all its axes (padding 0)
                    auto *    pp = (int32_t *) FPOS->vir;
                    const int ax = (int) (FPOS->size / (PG * sizeof(int32_t)));
                    GGML_ASSERT(ax <= 4);
                    for (int s = 0; s < ax; ++s) {
                        for (int r = 0; r < PG; ++r) {
                            pp[s * PG + r] = r < v ? p4[(size_t) (r0 + r) * 4 + s] : 0;
                            // (an index out of the kernel's tables hangs the NPU: stop here instead)
                            GGML_ASSERT(pp[s * PG + r] >= 0 && pp[s * PG + r] < man.context);
                        }
                    }
                    FPOS->flush();
                }
                paging(cell, cell);
                *(int32_t *) VALID->vir = v;
                VALID->flush();
                fcall &      c  = pre_f[p % 2][il];
                const size_t xo = (size_t) p * PG * xr, to = (size_t) p * PG * (TKI->size / NR);
                c[c.x]          = axera_io(XS.get(), (char *) XS->vir + xo, (uint32_t) (PG * xr));
                c[c.h2]         = axera_io(H2S.get(), (char *) H2S->vir + xo, (uint32_t) (PG * xr));
                c[c.base]       = axera_io(BASES.get(), (char *) BASES->vir + xo, (uint32_t) (PG * xr));
                c[c.ti]         = axera_io(TKI.get(), (char *) TKI->vir + to, (uint32_t) (PG * (TKI->size / NR)));
                c[c.tw]         = axera_io(TKW.get(), (char *) TKW->vir + to, (uint32_t) (PG * (TKW->size / NR)));
                axera_timer ta(il % 4 == 3 ? "prefill: attn_moe (page, layer)" : "prefill: gdn_moe (page, layer)");
                c.k->run(c.in, c.out, il % 4 == 3 ? "prefill attn_moe" : "prefill gdn_moe");
            }
            experts(il, rows);
            static const bool chk = getenv("GGML_AXERA_FUSED_CHECK") != nullptr;  // debug: each layer's output rows
            if (chk) {
                auto stat = [&](const axera_cmm & b, const char * nm) {
                    b->invalidate(b->vir, rows * xr);
                    const ggml_bf16_t * s = (const ggml_bf16_t *) b->vir;
                    float mx = 0; int64_t bad = 0;
                    for (int64_t i = 0; i < (int64_t) rows * H; ++i) {
                        const float f = ggml_bf16_to_fp32(s[i]);
                        if (!std::isfinite(f)) { bad++; } else { mx = std::max(mx, std::fabs(f)); }
                    }
                    fprintf(stderr, "axera fused: layer %d %s: max |x| %g, %lld non-finite\n", il, nm, mx, (long long) bad);
                    return bad;
                };
                stat(H2S, "h2"); stat(BASES, "base");
                if (stat(XS, "out")) {
                    GGML_ABORT("axera fused: layer %d: non-finite output\n", il);
                }
            }
        }
        if (np % 2) {  // the last page's states in the second set
            states_swap();
        }
        tg.stop();
        XS->invalidate(XS->vir, rows * xr);
        const int a = std::max(0, out0 - g0);
        if (a < rows) {
            axera_host_pool().run(rows - a, [&](int64_t i) {
                const int r = a + (int) i;
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) XS->vir + (size_t) r * H, h.data() + (size_t) (g0 + r - out0) * H, H);
            });
        }
        if (part) {  // the page's inputs so far: a later batch restarts the page with them
            snap_cell = c0 + g0 + (np - 1) * PG;
            page_x.assign(x + (size_t) (g0 + (np - 1) * PG) * H, x + (size_t) (g0 + rows) * H);
            page_p.assign(p4 + (size_t) (g0 + (np - 1) * PG) * 4, p4 + (size_t) (g0 + rows) * 4);
        } else if (g0 + rows == n) {  // a full last page: the next page starts clean
            save_state(c0 + n);
        }
    }
}

// layer il's routed experts for the group's first nt rows: their top-K (ids, normalized weights) grouped by expert into
// slots of G rows (experts with more than 64 picks in slots of 128 rows), the slots calls (a call: up to its window
// of rows), then the combine kernel adds every row's K slot rows to its base: the next layer's input, in XS
void qwen35_moe_sgl::experts(int il, int nt) {
    axera_timer   tr("prefill: moe route (host)");
    const size_t  NR = XS->size / ((size_t) H * sizeof(ggml_bf16_t));
    const int     K = (int) (TKI->size / (NR * sizeof(int32_t))), NE = 256, G1 = 64, G2 = 128;
    const int     SW = (int) (slots_k->out_bytes("out") / ((size_t) H * sizeof(ggml_bf16_t)));  // rows per slots call
    const int     SM = (int) (slots_k->in_bytes("slot_exp") / sizeof(int32_t));                 // slots per call, at most
    const int     GM = (int) (slots_k->in_bytes("slot_tok") / (SM * sizeof(int32_t)));          // rows per slot, at most
    const int64_t SR = comb_k->in_bytes("sout") / ((int64_t) H * sizeof(ggml_bf16_t));          // slot rows
    GGML_ASSERT(GM >= G2 && SW % G2 == 0);
    TKI->invalidate(TKI->vir, (size_t) nt * K * sizeof(int32_t));
    TKW->invalidate(TKW->vir, (size_t) nt * K * sizeof(float));
    const int32_t * ids = (const int32_t *) TKI->vir;
    const float *   wts = (const float *) TKW->vir;
    // picks grouped by expert (counting sort, row order within an expert)
    std::vector<int> cnt(NE, 0), first(NE + 1, 0), order((size_t) nt * K);
    for (int pk = 0; pk < nt * K; ++pk) {
        GGML_ASSERT(ids[pk] >= 0 && ids[pk] < NE);
        cnt[ids[pk]]++;
    }
    for (int e = 0; e < NE; ++e) {
        first[e + 1] = first[e] + cnt[e];
    }
    {
        std::vector<int> f(first.begin(), first.end() - 1);
        for (int pk = 0; pk < nt * K; ++pk) {
            order[f[ids[pk]]++] = pk;
        }
    }
    // two regions of slots (G2 first), each region's slots in calls of up to min(SM, SW / G) slots
    struct region {
        int G = 0, ns = 0, per = 0, call0 = 0, ncall = 0;
        int64_t row0 = 0;
    } rg[2];
    std::vector<char> big(NE, 0);
    std::vector<int>  slot0(NE, 0);
    for (int pass = 0; pass < 2; ++pass) {  // pass 1: no G2 slots (pass 0's rows over the combine kernel's)
        rg[0] = { G2 };
        rg[1] = { G1 };
        for (int e = 0; e < NE; ++e) {
            big[e]   = pass == 0 && cnt[e] > G1;
            region & r = rg[big[e] ? 0 : 1];
            slot0[e] = r.ns;
            r.ns += (cnt[e] + r.G - 1) / r.G;
        }
        rg[1].row0 = (int64_t) rg[0].ns * G2;
        if (rg[1].row0 + (int64_t) rg[1].ns * G1 <= SR) {
            break;
        }
        GGML_ASSERT(pass == 0 && "axera qwen35_35b_a3b: a layer's expert slots exceed the combine kernel's rows");
    }
    int ncall = 0;
    for (region & r : rg) {
        r.per   = std::min(SM, SW / r.G);
        r.call0 = ncall;
        r.ncall = (r.ns + r.per - 1) / r.per;
        ncall += r.ncall;
    }
    GGML_ASSERT((size_t) ncall * sizeof(int32_t) <= SNS->size);
    const size_t te = slots_k->in_bytes("slot_exp"), tt = slots_k->in_bytes("slot_tok"), tw = slots_k->in_bytes("rw");
    memset(SLOTE->vir, 0, ncall * te);  // padding: expert 0, row 0, weight 0
    memset(SLOTT->vir, 0, ncall * tt);
    memset(SLOTW->vir, 0, ncall * tw);
    auto * se  = (int32_t *) SLOTE->vir;
    auto * st  = (int32_t *) SLOTT->vir;
    auto * sw  = (float *) SLOTW->vir;
    auto * loc = (int32_t *) LOC->vir;  // [K][NR]
    for (const region & r : rg) {
        for (int c = 0; c < r.ncall; ++c) {
            ((int32_t *) SNS->vir)[r.call0 + c] = std::min(r.per, r.ns - c * r.per);
            ((int32_t *) SGR->vir)[r.call0 + c] = r.G;
        }
    }
    axera_host_pool().run(NE, [&](int64_t e) {
        const region & r = rg[big[e] ? 0 : 1];
        for (int j = 0; j < cnt[e]; ++j) {
            const int sl = slot0[e] + j / r.G, gg = j % r.G, pk = order[first[e] + j];
            const int ci = r.call0 + sl / r.per, js = sl % r.per;
            se[(size_t) ci * SM + js]                   = (int32_t) e;
            st[((size_t) ci * SM + js) * GM + gg]       = pk / K;
            sw[((size_t) ci * SM + js) * GM + gg]       = wts[pk];
            loc[(size_t) (pk % K) * NR + pk / K]        = (int32_t) (r.row0 + (int64_t) sl * r.G + gg);
        }
    });
    SLOTE->flush(SLOTE->vir, ncall * te);
    SLOTT->flush(SLOTT->vir, ncall * tt);
    SLOTW->flush(SLOTW->vir, ncall * tw);
    SNS->flush(SNS->vir, ncall * sizeof(int32_t));
    SGR->flush(SGR->vir, ncall * sizeof(int32_t));
    LOC->flush();
    *(int32_t *) NTOK->vir = nt;
    NTOK->flush();
    tr.stop();
    const kernel &                     sk = *slots_k;
    std::vector<AX_ENGINE_IO_BUFFER_T> in(sk.info->nInputSize), out(sk.info->nOutputSize);
    in[sk.input("h2")]   = axera_io(H2S);
    in[sk.input("e_gw")] = exp_w[il][0];
    in[sk.input("e_gs")] = exp_w[il][1];
    in[sk.input("e_dw")] = exp_w[il][2];
    in[sk.input("e_ds")] = exp_w[il][3];
    for (const region & r : rg) {
        for (int c = 0; c < r.ncall; ++c) {
            const int     ci = r.call0 + c;
            const int64_t o  = r.row0 + (int64_t) c * r.per * r.G;  // the call's first slot row
            in[sk.input("slot_exp")]   = axera_io(SLOTE.get(), (char *) SLOTE->vir + ci * te, (uint32_t) te);
            in[sk.input("slot_tok")]   = axera_io(SLOTT.get(), (char *) SLOTT->vir + ci * tt, (uint32_t) tt);
            in[sk.input("rw")]         = axera_io(SLOTW.get(), (char *) SLOTW->vir + ci * tw, (uint32_t) tw);
            in[sk.input("nslots")]     = axera_io(SNS.get(), (int32_t *) SNS->vir + ci, sizeof(int32_t));
            in[sk.input("group_rows")] = axera_io(SGR.get(), (int32_t *) SGR->vir + ci, sizeof(int32_t));
            out[sk.output("out")]      = axera_io(SOUT.get(), (char *) SOUT->vir + o * H * sizeof(ggml_bf16_t), sk.out_bytes("out"));
            axera_timer ts("prefill: moe slots");
            sk.run(in, out, "prefill moe slots");
        }
    }
    const kernel &                     ck = *comb_k;
    std::vector<AX_ENGINE_IO_BUFFER_T> ci(ck.info->nInputSize), co(ck.info->nOutputSize);
    ci[ck.input("base")]    = axera_io(BASES);
    ci[ck.input("loc")]     = axera_io(LOC);
    ci[ck.input("sout")]    = axera_io(SOUT, ck.in_bytes("sout"));
    ci[ck.input("ntokens")] = axera_io(NTOK);
    co[ck.output("out")]    = axera_io(XS);
    axera_timer tc("prefill: moe combine");
    ck.run(ci, co, "prefill moe combine");
}

// the history before the token(s) written from cell write_cell: hist cells in whole pages and a last one
void qwen35_moe_sgl::paging(int64_t hist, int64_t write_cell) {
    const int64_t pages          = (hist + PG - 1) / PG;
    ((int32_t *) INDPTR->vir)[0] = 0;
    ((int32_t *) INDPTR->vir)[1] = (int32_t) pages;
    *(int32_t *) LASTLEN->vir    = (int32_t) (pages ? hist - (pages - 1) * PG : 0);
    ((int32_t *) CLOC->vir)[0]   = (int32_t) (write_cell / PG);
    ((int32_t *) CLOC->vir)[1]   = (int32_t) (write_cell % PG);
    for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
        (*b)->flush();
    }
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16, their positions p4 [n][4]; the hidden states of
// the rows from out0 on into h
void qwen35_moe_sgl::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    const size_t xr = (size_t) H * sizeof(ggml_bf16_t);
    auto set_in = [&](call & c, const char * name, const axera_cmm & cm, size_t off) {
        for (size_t i = 0; i < c.k->desc->in.size(); ++i) {
            if (c.k->desc->in[i].name == name) {
                c.in[i] = axera_io(cm.get(), (char *) cm->vir + off, c.k->desc->in[i].bytes);
                return;
            }
        }
        GGML_ABORT("axera qwen35_moe_sgl: no input %s\n", name);
    };
    auto set_out = [&](call & c, const char * name, const axera_cmm & cm, size_t off) {
        for (size_t i = 0; i < c.k->desc->out.size(); ++i) {
            if (c.k->desc->out[i].name == name) {
                c.out[i] = axera_io(cm.get(), (char *) cm->vir + off, c.k->desc->out[i].bytes);
                return;
            }
        }
        GGML_ABORT("axera qwen35_moe_sgl: no output %s\n", name);
    };
    for (int r0 = 0; r0 < n; r0 += WN) {
        const int     live = std::min(WN, n - r0);
        const int64_t cell = c0 + r0;
        save_state(cell);
        // a batch that ends inside a page keeps that page's start as a restart point (per layer, at its own last page)
        const bool tail  = r0 + live == n && live % PG != 0;
        const int  lastp = tail ? live - live % PG : 0;
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) live * xr);
        memset((ggml_bf16_t *) PX->vir + (size_t) live * H, 0, (size_t) (WN - live) * xr);
        PX->flush();
        axera_timer tl("prefill (chunk)");
        for (int il = 0; il < L; ++il) {
            const bool gdn = il % 4 != 3;
            for (int s = 0; s < live; s += PG) {
                const int     v    = std::min(PG, live - s);
                const int64_t c    = cell + s;
                if (gdn && tail && lastp > 0 && s == lastp) {
                    snap_layer(il, 0);
                }
                set_in(pre_attn[il], "x", PX, (size_t) s * xr);
                if (gdn) {
                    set_out(pre_attn[il], "residual_out", PA, (size_t) s * xr);
                } else {
                    set_out(pre_attn[il], "output", PA, (size_t) s * xr);
                    auto * pp = (int32_t *) POS->vir;
                    for (int a = 0; a < AX; ++a) {
                        for (int r = 0; r < PG; ++r) {
                            pp[a * PG + r] = r < v ? p4[(size_t) (s + r) * 4 + a] : 0;
                        }
                    }
                    POS->flush();
                }
                paging(c, c);
                *(int32_t *) VALID->vir = v;
                VALID->flush();
                axera_timer ta(gdn ? "prefill: gdn (chunk, layer)" : "prefill: attn (chunk, layer)");
                pre_attn[il].k->run(pre_attn[il].in, pre_attn[il].out, gdn ? "prefill gdn" : "prefill attn");
                if (gdn) {
                    states_swap();
                }
            }
            if (live < WN) {
                *(int32_t *) VALID->vir = live;
                VALID->flush();
                axera_timer tp("prefill: padding");
                pad.k->run(pad.in, pad.out, "prefill padding");
            }
            axera_timer tm("prefill: moe (chunk, layer)");
            pre_moe[il].k->run(pre_moe[il].in, pre_moe[il].out, "prefill moe");
        }
        const bool     at_page = (cell + live) % PG == 0;
        if (at_page) {
            save_state(cell + live);
        }
        PX->invalidate(PX->vir, (size_t) live * xr);
        const int first = std::max(0, out0 - r0), last = std::min(live, out0 + (int) h.size() / H - r0);
        if (first < last) {
            axera_host_pool().run(last - first, [&](int64_t i) {
                const int r = first + (int) i;
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) PX->vir + (size_t) r * H,
                                      h.data() + (size_t) (r0 + r - out0) * H, H);
            });
        }
        if (!at_page) {  // the last partial page: from its start, where the per-layer state was snapshotted
            if (tail && lastp > 0) {
                snap_cell = cell + lastp;
            }
            page_x.assign(x + (size_t) (r0 + lastp) * H, x + (size_t) (r0 + live) * H);
            page_p.assign(p4 + (size_t) (r0 + lastp) * 4, p4 + (size_t) (r0 + live) * 4);
        }
    }
    state_cell = c0 + n;
}

// one token at cell c > 0, its history the c cells before it
void qwen35_moe_sgl::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h) {
    if (c % WN == 0) {  // a chunk's first cell: the state here is the restart point
        save_state(c);
    }
    memcpy(DX->vir, x, H * sizeof(ggml_bf16_t));
    DX->flush();
    for (int a = 0; a < AX && a < 4; ++a) {
        ((int32_t *) DPOS->vir)[a] = p4[a];
    }
    DPOS->flush();
    paging(c, c);
    {
        axera_timer tl("layers (decode)");
        dec.k->run(dec.in, dec.out, "decode");
    }
    DOUT->invalidate(DOUT->vir, H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) DOUT->vir, h, H);
    page_x.insert(page_x.end(), x, x + H);
    page_p.insert(page_p.end(), p4, p4 + 4);
    state_cell = c + 1;
}

ggml_status qwen35_moe_sgl::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_35b_a3b: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) {
            ++n0;
        }
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0           = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
        ++L_;
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera qwen35_35b_a3b: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, man.context, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera qwen35_35b_a3b: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera qwen35_35b_a3b: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t *      pos = (const int32_t *) inp_pos->data;
    const bool           mr  = inp_pos->ne[0] == 4 * (int64_t) T;  // M-RoPE: [4][T], else [T]
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
        }
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
    });
    std::vector<float> h((size_t) T * H);
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const int64_t      c0 = kv.cell_of(0);
    if (c0 == 0) {  // a new sequence
        zero_state();
    }
    if (T > 1 || c0 == 0) {
        if (c0 == state_cell && c0 % PG == 0) {  // sequential batch from a page start: the state is exactly here
            axera_timer tp("prefill (all pages)");
            prefill(c0, T, xb.data(), pb.data(), h, 0);
        } else {
            // a rewind (e.g. the server's prompt cache): from c0's page start and the state saved there
            const int64_t s0 = c0 - c0 % PG;
            const int     k  = (int) (c0 - s0);
            if (k > 0 && (snap_cell != s0 || (int) (page_x.size() / H) < k)) {
                GGML_LOG_ERROR("axera qwen35_moe_sgl: a prompt from cell %lld, inside a page whose state is not kept\n",
                               (long long) c0);
                return GGML_STATUS_FAILED;
            }
            page_x.resize((size_t) k * H);  // (a kept prefix, e.g. the server's prompt cache: the cells after it are gone)
            page_p.resize((size_t) k * 4);
            std::vector<ggml_bf16_t> xs;
            std::vector<int32_t>     ps;
            if (k > 0) {
                load_state();
                xs = page_x;
                ps = page_p;
            }
            xs.insert(xs.end(), xb.begin(), xb.end());
            ps.insert(ps.end(), pb.begin(), pb.end());
            axera_timer tp("prefill (all pages)");
            prefill(s0, k + T, xs.data(), ps.data(), h, k);
        }
    } else {
        if (c0 != state_cell) {
            GGML_LOG_ERROR("axera qwen35_moe_sgl: a decode from cell %lld, not the state's cell %lld\n", (long long) c0,
                           (long long) state_cell);
            return GGML_STATUS_FAILED;
        }
        decode(c0, xb.data(), pb.data(), h.data());
    }
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    head.run(g, names, h, T);
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_moe_sgl(const manifest & m) {
    return std::make_shared<qwen35_moe_sgl>(m);
}

}  // namespace axera
