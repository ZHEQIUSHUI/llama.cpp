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
// The state is this adapter's (the decode kernel's banks): KV cache paged [10][64 pages][K, V][256][2 x 256] bf16 (a
// long-context decode kernel instead takes one input b3_kv<j> [pages][K, V][256][2 x 256] per attention layer, its pools
// holding only the pages of llama.cpp's -c),
// delta-net states [30][32][128][128] f32, conv states q / k [30][3][2048], v [30][3][4096] bf16 (value heads in HF
// order). The prefill kernels write the recurrent states to a second set, the two swapped after each page. One
// sequence. Weights only from the GGUF: every projection and expert row is quantized to NVFP4 (E2M1 pairs, an E4M3
// scale per 16 inputs, the FP32 tensor scale2) in memory at load, one tensor / expert at a time, straight into the
// device banks. An offline bank pack is a debug fallback only (AXERA_SGL35_BANKS=1); the release ships no bank files.

#include "../model.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"
#include "vocab_embedding_1_axmodel.h"
#include "vocab_embedding_axmodel.h"
#include "fused_head_axmodel.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;          // cells per page = prefill rows per call
constexpr int WN = 4096;         // rows in the fixed MoE prefill workspace
constexpr int KH = 16, VH = 32;  // key / value heads of a delta-net layer
constexpr int REP = VH / KH;
constexpr int NB = 10;           // blocks of 4 layers
// the per-page runtime metadata, packed so a page writes and flushes one cache range instead of four
constexpr size_t META_INDPTR = 0, META_LASTLEN = 8, META_CLOC = 12, META_VALID = 20, META_BYTES = 24;

// the recurrent-state pools (kind 0: state, 1: q conv, 2: k conv, 3: v conv)
static const char * STATE_BANK[4] = { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in",
                                      "b0_attn_v_conv_state_in" };
static int state_kind(const std::string & n) {
    for (int k = 0; k < 4; ++k) {
        if (n == STATE_BANK[k]) {
            return k;
        }
    }
    return -1;
}

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    int                                ix = -1, iout = -1, ires = -1, ipos = -1;  // hot per-page IO entries
};

// GGML_AXERA_SGL35_BANK_SUM=1: print one FNV-1a hash per bank after load. The loader must produce the same bytes
// before and after a conversion optimisation; this is the cheap check that it did (debug only).
void axera_bank_checksums(const std::map<std::string, axera_cmm> & banks) {
    const char * e = std::getenv("GGML_AXERA_SGL35_BANK_SUM");
    if (!e || std::atoi(e) == 0) {
        return;
    }
    uint64_t all = 1469598103934665603ULL;
    size_t   nb  = 0;
    for (const auto & [n, c] : banks) {
        uint64_t h = 1469598103934665603ULL;
        const uint8_t * p = (const uint8_t *) c->vir;
        size_t i = 0;
        for (; i + 8 <= c->size; i += 8) {
            uint64_t v;
            memcpy(&v, p + i, 8);
            h = (h ^ v) * 1099511628211ULL;
            all = (all ^ v) * 1099511628211ULL;
        }
        for (; i < c->size; ++i) {
            h = (h ^ p[i]) * 1099511628211ULL;
            all = (all ^ p[i]) * 1099511628211ULL;
        }
        all ^= c->size;
        fprintf(stderr, "GGML_AXERA_SGL35_BANK_SUM bank %-40s %10zu %016llx\n", n.c_str(), c->size,
                (unsigned long long) h);
        nb += c->size;
    }
    fprintf(stderr, "GGML_AXERA_SGL35_BANK_SUM all %zu bytes %016llx\n", nb, (unsigned long long) all);
}

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
    call                             ver;  // role decode.verify: four rows, one rank-1 delta per row
    call                             app;  // role gdn.apply: advance the recurrent state by accepted deltas
    axera_cmm                        VX, VOUT, VPOSN, VPIDS, VINDPTR, VLOC, VNT, VNPREV, VST0;
    axera_cmm                        VDELTA[2][3];  // u / k / g per verification set
    axera_cmm                        VSNAP[2][3];   // q / k / v Conv history per verification set
    std::vector<uint8_t>             conv_entry;    // Conv state before the latest verify (j = 0)
    bool                             conv_entry_valid = false;
    int                              conv_set = -1, conv_n = 0;
    int                              delta_set = -1, delta_n = 0;
    bool                             delta_live = false;
    int64_t                          ver_c0 = -1;
    int                              ver_T = 0;
    call                             pad;
    std::array<int, 4>               dec_state = { -1, -1, -1, -1 };
    bool                             dec_state_dirty = false;  // dec.in's state entries follow the prefill's swaps
    std::vector<call>                pre_attn[2], pre_moe;  // per layer and recurrent-state parity
    int                              state_parity = 0;
    axera_cmm DX, DA, DOUT, DPOS, INDICES, PX, PA, XB, POS, META;
    // the token embedding as an NPU gather (the GGUF table converted to bf16 once), for the whole-prefill chunk
    // and a decode token; when absent compute() falls back to the graph's CPU GET_ROWS
    bool      emb = false;
    axera_cmm EMBW, EMBI;
    call      emb4k, emb1;
    // the upstream fused RMSNorm + projection head, for one-row decodes (see output_head::bind_fused)
    bool      fh_loaded = false;
    bool      fh_ok     = false;
    // the recurrent state at the first cell of the current page (snap_cell), and the inputs of that page's cells so far
    std::vector<uint8_t>     snap;
    int64_t                  snap_cell = -1;
    int                      snap_parity = -1;  // 1: the second state set holds snap_cell's state (no copy was made)
    int64_t                  state_cell = -1;  // tokens processed: the current recurrent state is before this cell
    bool                     state_dirty = false;  // a sequence has updated the recurrent-state pools
    std::vector<ggml_bf16_t> page_x;
    std::vector<int32_t>     page_p;  // [cells][4]
    std::vector<float>       hb;
    std::vector<int>         orows;
    std::vector<int32_t>     p4b;
    output_head              head;

    // the MTP block (blk.<L>.nextn.*): its own paged KV, the weight tensors in the kernels' layouts, the calls
    struct mtp_call {
        call      c;
        axera_cmm e, h, pids, pos, indptr, loc, ntok, hnext, hraw;
    };
    mtp_call                         mtp1, mtp8, mtpP;
    std::map<std::string, axera_cmm> mtp_w;
    axera_cmm                        MKV, MINDICES;
    int                              mtp_kv_pages = 0;
    size_t                           mtp_kv_bytes = 0;
    const void *                     mtp_bound = nullptr;

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
    // the paged KV of attention layer j: entry j of the shared pool b3_kv, or its own pool (inputs b3_kv<j>)
    std::vector<axera_cmm> KVL;
    int                    kv_pages = 0;
    int64_t                kv_cells = 0;
    AX_ENGINE_IO_BUFFER_T  kv_io(int j, uint32_t bytes) {
        if (KVL.empty()) {
            return slice("b3_kv", j, bytes);
        }
        return axera_io(KVL.at(j).get(), KVL[j]->vir, bytes);  // (the kernel touches only the pages up to the history)
    }
    void         states_swap();
    void         sync_dec_states();
    void         zero_state();
    void         save_state(int64_t cell);
    void         load_state();
    void         state_home();
    void         materialize_deltas();
    void         restore_conv(int accepted);
    void         bind_deltas(call & c, int out_set, int prev_set);
    void         bind_state();
    AX_ENGINE_IO_BUFFER_T state_io(const std::string & name);
    ggml_status  verify(const kv_view & kv, const ggml_bf16_t * xb, const int32_t * pb, int T, float * h);
    void         init_mtp(const std::map<std::string, ggml_tensor *> & t, int il, const rope & rq);
    ggml_status  compute_mtp(ggml_cgraph * g);
    void         paging(int64_t hist, int64_t write_cell);
    void         embed_tokens(const int32_t * toks, int n);
    void         prefill(int64_t c0, int n, const ggml_bf16_t * x, const float * xf, size_t xf_nb1,
                         const int32_t * toks,
                         const int32_t * p4, std::vector<float> & h, const std::vector<int> & out_rows);
    void         decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h);
    void         prewarm(ggml_cgraph * g) override;
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
    {
        axera_timer ta("fill: alloc");
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
                    banks[d.name] = axera_cmm_alloc_raw(d.bytes);
                }
                continue;
            }
            axera_cmm &       c = banks[d.name] = axera_cmm_alloc_raw(d.bytes);
            const std::string n = mt[3];
            if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in" ||
                n == "kv") {
                continue;  // the state pools: zero
            }
            if (n == "cos_sin_cache") {  // [positions][cos | sin of ROT / 2] f32
                const int64_t P = d.shape[1], R = d.shape[2] / 2;
                GGML_ASSERT((size_t) R == rp.inv_freq.size() && d.dtype == 3);
                memset(c->vir, 0, c->size);  // raw allocation: zero the alignment tail (this bank is not in the batch)
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
        // The state / KV pools need zeroes and the weight banks would otherwise be zeroed one at a time before the
        // fill overwrites them. Zero all banks in parallel once; the fill's final pass does the only cache flush.
        std::vector<axera_cmm *> bv;
        bv.reserve(banks.size());
        for (auto & [n, c] : banks) {
            if (n.find("cos_sin_cache") == std::string::npos) {  // already generated above
                bv.push_back(&c);
            }
        }
        axera_host_pool().run((int64_t) bv.size(), [&](int64_t i) { memset((*bv[i])->vir, 0, (*bv[i])->size); });
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
            std::vector<float> tmp(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                tmp[i] = v[i] * scale;
            }
            memcpy(d, tmp.data(), bb);
        } else {
            if (v.size() * 2 != bb) {
                GGML_ABORT("axera qwen35_moe_sgl: %s bf16 %zu floats, bank slot %zu bytes\n", n.c_str(), v.size(), bb);
            }
            std::vector<float> scaled(scale == 1.0f ? 0 : v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                if (scale != 1.0f) {
                    scaled[i] = v[i] * scale;
                }
            }
            std::vector<ggml_bf16_t> tmp(v.size());
            ggml_fp32_to_bf16_row(scale == 1.0f ? v.data() : scaled.data(), tmp.data(), v.size());
            memcpy(d, tmp.data(), bb);
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

    axera_timer tl("fill: layers");
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
                const ggml_tensor * d = T(il, "ffn_down_exps.weight");
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
    tl.stop();
    {
        axera_timer tf("fill: flush");
        for (auto & [n, c] : banks) {
            c->flush();
        }
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
    snap.assign(logical("b0_attn_state_in") + logical("b0_attn_q_conv_state_in") +
                    logical("b0_attn_k_conv_state_in") + logical("b0_attn_v_conv_state_in"), 0);
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
    const bool   split   = logical("b3_kv0") > 0;  // one KV input per attention layer
    const size_t page_kv = (size_t) 2 * PG * 2 * 256 * 2;
    const int    NP      = (int) (split ? logical("b3_kv0") / page_kv : logical("b3_kv") / (NB * page_kv));  // pages
    kv_pages             = NP;
    KVL.clear();
    if (split) {
        if (const int64_t cells = axera_kv_cells(); cells > 0) {  // llama.cpp's -c
            kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(NP, (cells + PG - 1) / PG));
        }
        if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
            kv_pages = std::max(1, std::min(NP, atoi(e)));
        }
        for (int j = 0; j < NB; ++j) {
            GGML_ASSERT(logical("b3_kv" + std::to_string(j)) == (size_t) NP * page_kv);
            KVL.push_back(axera_cmm_alloc((size_t) kv_pages * page_kv));
            memset(KVL[j]->vir, 0, KVL[j]->size);
            KVL[j]->flush();
        }
        GGML_LOG_INFO("axera qwen35_moe_sgl: KV pools of %d of the kernels' %d pages (%lld cells), %.2f GiB\n", kv_pages, NP,
                      (long long) kv_pages * PG, NB * (double) kv_pages * page_kv / 1073741824.0);
    }
    kv_cells = (int64_t) kv_pages * PG;
    DX      = axera_cmm_alloc(dk.in_bytes("x"));
    DA      = axera_cmm_alloc(dk.in_bytes("attention"));
    DOUT    = axera_cmm_alloc(dk.out_bytes("output"));
    DPOS    = axera_cmm_alloc(dk.in_bytes("position_ids"));
    INDICES = axera_cmm_alloc((size_t) NP * sizeof(int32_t));
    PX      = axera_cmm_alloc((size_t) WN * H * sizeof(ggml_bf16_t));
    PA      = axera_cmm_alloc((size_t) WN * H * sizeof(ggml_bf16_t));
    XB      = axera_cmm_alloc((size_t) WN * H * sizeof(ggml_bf16_t));
    POS     = axera_cmm_alloc((size_t) WN * ak.in_bytes("position_ids") / PG);
    META    = axera_cmm_alloc_nc(32);  // host-written scalars: the NPU sees each write without a cache flush
    AX      = (int) (ak.in_bytes("position_ids") / (PG * sizeof(int32_t)));
    GGML_ASSERT((AX == 1 || AX == 3) && (int64_t) NP * PG == man.context && dk.in_bytes("indices") == INDICES->size);
    for (int i = 0; i < NP; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    // The token embedding table is the one weight this adapter runs through the NPU's gather instead of the CPU's
    // GET_ROWS. The two kernel images are part of the adapter (not the manifest): 4096 padded rows for a prefill
    // chunk, one row for decode. The table is converted once; the output of the one-row kernel lands directly in DX.
    {
        axera_timer te("init: token embedding");
        auto it = t.find("token_embd.weight");
        if (it != t.end()) {
            const ggml_tensor * tw = it->second;
            const auto & k4 = man.load_memory("vocab_embedding_4096", std::vector<char>(
                (const char *) VOCAB_EMBEDDING_AXMODEL,
                (const char *) VOCAB_EMBEDDING_AXMODEL + VOCAB_EMBEDDING_AXMODEL_SIZE));
            const auto & k1 = man.load_memory("vocab_embedding_1", std::vector<char>(
                (const char *) VOCAB_EMBEDDING_1_AXMODEL,
                (const char *) VOCAB_EMBEDDING_1_AXMODEL + VOCAB_EMBEDDING_1_AXMODEL_SIZE));
            const size_t wb = k4.in_bytes("weight");
            const size_t ob = k4.out_bytes("output_flat");
            GGML_ASSERT(tw->ne[0] == H && wb == (size_t) tw->ne[0] * tw->ne[1] * sizeof(ggml_bf16_t));
            GGML_ASSERT(k4.in_bytes("input_flat") == WN * sizeof(int32_t) && ob == (size_t) WN * H * sizeof(ggml_bf16_t));
            GGML_ASSERT(k1.in_bytes("input_flat") == sizeof(int32_t) && k1.in_bytes("weight") == wb &&
                        k1.out_bytes("output_flat") == (size_t) H * sizeof(ggml_bf16_t));
            EMBW = axera_cmm_alloc(wb);
            if (tw->type == GGML_TYPE_BF16 && ggml_is_contiguous(tw) && ggml_nbytes(tw) == wb) {
                memcpy(EMBW->vir, tw->data, wb);
            } else {
                axera_for_rows_f32(tw, [&](int64_t r, const float * row) {
                    ggml_fp32_to_bf16_row(row, (ggml_bf16_t *) EMBW->vir + (size_t) r * H, H);
                });
            }
            EMBW->flush();
            axera_drop_mapped(tw);
            EMBI = axera_cmm_alloc(WN * sizeof(int32_t));
            emb4k = {};
            emb4k.k = &k4;
            emb4k.in.assign(k4.info->nInputSize, {});
            emb4k.out.assign(k4.info->nOutputSize, {});
            emb4k.in[k4.input("input_flat")] = axera_io(EMBI, EMBI->size);
            emb4k.in[k4.input("weight")]     = axera_io(EMBW, EMBW->size);
            emb4k.out[k4.output("output_flat")] = axera_io(XB, (uint32_t) ob);
            emb1 = {};
            emb1.k = &k1;
            emb1.in.assign(k1.info->nInputSize, {});
            emb1.out.assign(k1.info->nOutputSize, {});
            emb1.in[k1.input("input_flat")] = axera_io(EMBI, (uint32_t) k1.in_bytes("input_flat"));
            emb1.in[k1.input("weight")]     = axera_io(EMBW, EMBW->size);
            emb1.out[k1.output("output_flat")] = axera_io(DX, (uint32_t) k1.out_bytes("output_flat"));
            emb = true;
            GGML_LOG_INFO("axera qwen35_35b_a3b: token embedding %.2f GiB on the NPU\n", wb / 1073741824.0);
        }
    }

    // decode: the banks, its own rows and paging
    dec   = {};
    dec.k = &dk;
    for (const auto & d : dk.desc->in) {
        const std::string & n = d.name;
        const int st = n == "state_in" ? 0 : n == "q_conv_state_in" ? 1 : n == "k_conv_state_in" ? 2 :
                       n == "v_conv_state_in" ? 3 : -1;
        if (st >= 0) {
            dec_state[st] = (int) dec.in.size();
        }
        if (n == "indptr") {
            dec.in.push_back(axera_io(META.get(), (char *) META->vir + META_INDPTR, d.bytes));
        } else if (n == "last_page_len") {
            dec.in.push_back(axera_io(META.get(), (char *) META->vir + META_LASTLEN, d.bytes));
        } else if (n == "cache_location") {
            dec.in.push_back(axera_io(META.get(), (char *) META->vir + META_CLOC, d.bytes));
        } else {
            const axera_cmm * b = n == "x" ? &DX : n == "attention" ? &DA : n == "position_ids" ? &DPOS :
                                  n == "indices" ? &INDICES : nullptr;
            if (split && n.rfind("b3_kv", 0) == 0) {
                dec.in.push_back(kv_io(std::stoi(n.substr(5)), d.bytes));
            } else {
                dec.in.push_back(axera_io(b ? *b : B(n), d.bytes));
            }
        }
    }
    dec.out.push_back(axera_io(DOUT, dk.out_bytes("output")));

    // prefill: per layer and state parity, the slices of the decode banks of its phase and block. Both parities are
    // bound once: a page only swaps the four state pools and selects the other call, no IO scanning per layer.
    pre_attn[0].assign(L, {});
    pre_attn[1].assign(L, {});
    pre_moe.assign(L, {});
    auto meta = [&](size_t off, uint32_t bytes) {
        return axera_io(META.get(), (char *) META->vir + off, bytes);
    };
    auto state_second = [&](const std::string & in, int il, uint32_t bytes) {
        const axera_cmm & s2 = in == "state_in" ? ST2 : in == "q_conv_state_in" ? CQ2 : in == "k_conv_state_in" ? CK2 : CV2;
        return axera_io(s2.get(), (char *) s2->vir + (size_t) ord[il] * bytes, bytes);
    };
    for (int par = 0; par < 2; ++par) {
        for (int il = 0; il < L; ++il) {
            const int         phase = il % 4, blk = il / 4;
            const std::string pa = "b" + std::to_string(phase) + "_attn_";
            call &            c = pre_attn[par][il];
            c.k                 = phase == 3 ? &ak : &gk;
            for (const auto & d : c.k->desc->in) {
                const std::string & n = d.name;
                c.ix = n == "x" ? (int) c.in.size() : c.ix;
                if (n == "x") {
                    c.in.push_back(axera_io(PX, d.bytes));
                } else if (n == "position_ids") {
                    c.ipos = (int) c.in.size();
                    c.in.push_back(axera_io(POS, d.bytes));
                } else if (n == "valid_tokens") {
                    c.in.push_back(meta(META_VALID, d.bytes));
                } else if (n == "indptr") {
                    c.in.push_back(meta(META_INDPTR, d.bytes));
                } else if (n == "indices") {
                    c.in.push_back(axera_io(INDICES, d.bytes));
                } else if (n == "cache_location") {
                    c.in.push_back(meta(META_CLOC, d.bytes));
                } else if (n.find("scale2") != std::string::npos) {
                    c.in.push_back(part(pa + n, blk, d.bytes));
                } else if (n.rfind("w_zeropoint_", 0) == 0) {
                    c.in.push_back(axera_io(ZERO, d.bytes));
                } else if (n == "paged_kv_cache") {
                    c.in.push_back(kv_io(ord[il], d.bytes));
                } else if (n == "cos_sin_cache") {
                    c.in.push_back(axera_io(B("b3_attn_cos_sin_cache"), d.bytes));
                } else if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in") {
                    c.in.push_back(par == 0 ? slice("b0_attn_" + n, ord[il], d.bytes) : state_second(n, il, d.bytes));
                } else {
                    static const std::map<std::string, std::string> A = { { "x_norm_w", "w_norm_attn" }, { "q_norm_w", "w_norm_q" },
                                                                           { "k_norm_w", "w_norm_k" } };
                    c.in.push_back(slice(pa + (A.count(n) ? A.at(n) : n), blk, d.bytes));
                }
            }
            for (const auto & d : c.k->desc->out) {
                const std::string & n = d.name;
                if (n == "output" || n == "residual_out") {
                    (n == "output" ? c.iout : c.ires) = (int) c.out.size();
                    c.out.push_back(axera_io(PA, d.bytes));
                } else {  // <x>_out: the other state set
                    const std::string in = n.substr(0, n.size() - 4) + "_in";
                    c.out.push_back(par == 0 ? state_second(in, il, d.bytes) : slice("b0_attn_" + in, ord[il], d.bytes));
                }
            }
        }
    }
    for (int il = 0; il < L; ++il) {
        const int         blk = il / 4;
        const std::string pm = "b" + std::to_string(il % 4) + "_moe_";
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
            pad.in.push_back(meta(META_VALID, d.bytes));
        } else {
            GGML_ABORT("axera qwen35_moe_sgl: unexpected padding input %s\n", d.name.c_str());
        }
    }

    // verification (role decode.verify): four rows' positions, the entry state, the accepted-delta
    // prologue, one rank-1 delta per row, and the small Conv history snapshots
    ver = {};
    if (man.has("decode.verify")) {
        const kernel & vk = man.get("decode.verify");
        GGML_ASSERT(vk.in_bytes("x") == 4 * H * sizeof(ggml_bf16_t) &&
                    vk.out_bytes("output") == 4 * H * sizeof(ggml_bf16_t) && vk.in_bytes("ntok") == sizeof(int32_t));
        VX      = axera_cmm_alloc(vk.in_bytes("x"));
        VOUT    = axera_cmm_alloc(vk.out_bytes("output"));
        VPOSN   = axera_cmm_alloc(vk.in_bytes("positions"));
        VPIDS   = axera_cmm_alloc(vk.in_bytes("position_ids"));
        VINDPTR = axera_cmm_alloc(vk.in_bytes("indptr"));
        VLOC    = axera_cmm_alloc(vk.in_bytes("cache_location"));
        VNT     = axera_cmm_alloc(vk.in_bytes("ntok"));
        VNPREV  = axera_cmm_alloc(vk.in_bytes("nprev"));
        VST0    = axera_cmm_alloc(logical(STATE_BANK[0]));
        GGML_ASSERT(vk.in_bytes("indices") == INDICES->size);
        GGML_ASSERT(vk.in_bytes("du") == vk.in_bytes("pu") && vk.in_bytes("dk") == vk.in_bytes("pk") &&
                    vk.in_bytes("dg") == vk.in_bytes("pg"));
        static const char * DN[3] = { "du", "dk", "dg" };
        static const char * SN[3] = { "cqsnap", "cksnap", "cvsnap" };
        for (auto & set : VSNAP) {
            for (int k = 0; k < 3; ++k) {
                set[k] = axera_cmm_alloc(vk.in_bytes(SN[k]));
                GGML_ASSERT(vk.in_bytes(SN[k]) == 4 * logical(STATE_BANK[k + 1]));
            }
        }
        for (auto & set : VDELTA) {
            for (int k = 0; k < 3; ++k) {
                set[k] = axera_cmm_alloc(vk.in_bytes(DN[k]));
            }
        }
        ver.k = &vk;
        for (const auto & d : vk.desc->in) {
            const std::string & n = d.name;
            const axera_cmm *   b = n == "x" ? &VX : n == "output" ? &VOUT : n == "positions" ? &VPOSN :
                                    n == "position_ids" ? &VPIDS : n == "indptr" ? &VINDPTR : n == "ntok" ? &VNT :
                                    n == "cache_location" ? &VLOC : n == "indices" ? &INDICES : n == "nprev" ? &VNPREV : nullptr;
            if (b) {
                ver.in.push_back(axera_io(*b, d.bytes));
            } else if (state_kind(n) >= 0) {
                ver.in.push_back(state_io(n));
            } else if (int k = n == DN[0] || n == "pu" ? 0 : n == DN[1] || n == "pk" ? 1 : n == DN[2] || n == "pg" ? 2 : -1;
                       k >= 0) {
                ver.in.push_back(axera_io(VDELTA[0][k], d.bytes));
            } else if (int k = n == SN[0] ? 0 : n == SN[1] ? 1 : n == SN[2] ? 2 : -1; k >= 0) {
                ver.in.push_back(axera_io(VSNAP[0][k], d.bytes));
            } else if (n.rfind("b3_kv", 0) == 0) {
                ver.in.push_back(kv_io(std::stoi(n.substr(5)), d.bytes));
            } else if (banks.count(n) && bank_n.at(n) == d.bytes) {
                ver.in.push_back(axera_io(B(n), d.bytes));
            } else {
                GGML_ABORT("axera qwen35_moe_sgl: decode.verify: unexpected input %s\n", n.c_str());
            }
        }
        for (const auto & d : vk.desc->out) {
            const std::string & n = d.name;
            GGML_ASSERT(n == "output");
            ver.out.push_back(axera_io(VOUT, d.bytes));
        }

        app = {};
        GGML_ASSERT(man.has("gdn.apply") && "decode.verify needs the gdn.apply delta kernel");
        const kernel & ak = man.get("gdn.apply");
        GGML_ASSERT(ak.in_bytes("state_in") == logical(STATE_BANK[0]) &&
                    ak.out_bytes("state_out") == logical(STATE_BANK[0]) && ak.in_bytes("nprev") == sizeof(int32_t));
        GGML_ASSERT(ak.in_bytes("pu") == vk.in_bytes("pu") && ak.in_bytes("pk") == vk.in_bytes("pk") &&
                    ak.in_bytes("pg") == vk.in_bytes("pg"));
        app.k = &ak;
        for (const auto & d : ak.desc->in) {
            const std::string & n = d.name;
            const axera_cmm *   b = n == "state_in" ? &B(STATE_BANK[0]) : n == "nprev" ? &VNPREV :
                                    n == "pu" ? &VDELTA[0][0] : n == "pk" ? &VDELTA[0][1] : n == "pg" ? &VDELTA[0][2] : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen35_moe_sgl: gdn.apply: unexpected input %s\n", n.c_str());
            }
            app.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : ak.desc->out) {
            GGML_ASSERT(d.name == "state_out");
            app.out.push_back(axera_io(B(STATE_BANK[0]), d.bytes));
        }
        conv_entry.resize((size_t) (logical(STATE_BANK[1]) + logical(STATE_BANK[2]) + logical(STATE_BANK[3])));
        bind_state();
    }
    fused = man.has("prefill.gdn_moe") && man.has("prefill.attn_moe") && man.has("moe.slots") && man.has("moe.combine");
    if (fused) {
        init_fused();
    }
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    axera_bank_checksums(banks);
    GGML_LOG_INFO("axera qwen35_35b_a3b: %d layers, banks (weights + state) %.2f GiB, %lld cells\n", L, bb / 1073741824.0,
                  (long long) man.context);
}

// the prefill kernels' new recurrent states become the current ones. Both state parities' calls are bound once, so
// this is only the four pool swaps: no per-page scan or rebind of thousands of IO entries. The decode call's state
// entries follow at the next decode (a page does not read them).
void qwen35_moe_sgl::states_swap() {
    axera_timer ts("states swap");
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
        if (ver.k) {
            rebind(ver.in);
            rebind(app.in);
            rebind(app.out);
        }
        for (auto & par : pre_attn) {
            for (auto & c : par) {
                rebind(c.in);
                rebind(c.out);
            }
        }
        for (auto & v : pre_f) {
            for (auto & c : v) {
                rebind(c.in);
                rebind(c.out);
            }
        }
    }
    dec_state_dirty = true;
    state_parity ^= 1;
}

void qwen35_moe_sgl::sync_dec_states() {
    if (!dec_state_dirty) {
        return;
    }
    int st = 0;
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        if (dec_state[st] >= 0) {
            dec.in[dec_state[st]] = axera_io(B(n), dec.in[dec_state[st]].nSize);
        }
        ++st;
    }
    dec_state_dirty = false;
}

// Bind a verify / apply call's delta rows.  The recurrent state itself always
// lives in its original pool; only an accepted prefix of (u, k, g) is pending.
void qwen35_moe_sgl::bind_deltas(call & c, int out_set, int prev_set) {
    for (size_t i = 0; i < c.in.size(); ++i) {
        const std::string & n = c.k->desc->in[i].name;
        const int k = n == "du" || n == "pu" ? 0 : n == "dk" || n == "pk" ? 1 : n == "dg" || n == "pg" ? 2 : -1;
        if (k < 0) {
            continue;
        }
        const bool prev = n[0] == 'p';
        const int  set  = prev ? prev_set : out_set;
        GGML_ASSERT(set >= 0 && set < 2);
        c.in[i] = axera_io(VDELTA[set][k], c.k->desc->in[i].bytes);
    }
}

// Apply the accepted rows to the recurrent-state pool on the NPU.  Verify
// consumes them in its prologue; decode / prefill use this tiny kernel first.
void qwen35_moe_sgl::materialize_deltas() {
    if (delta_n <= 0) {
        delta_live = false;
        return;
    }
    GGML_ASSERT(app.k && delta_set >= 0 && delta_set < 2);
    axera_timer ta("gdn delta apply");
    bind_deltas(app, -1, delta_set);
    *(int32_t *) VNPREV->vir = delta_n;
    VNPREV->flush();
    app.k->run(app.in, app.out, "gdn apply");
    delta_n    = 0;
    delta_live = false;
}

// Restore the tiny Conv history at the state before cell c0 + accepted.  Row
// `accepted - 1` of the verify set is the history after that accepted token;
// accepted == 0 uses the pre-verify copy.
void qwen35_moe_sgl::restore_conv(int accepted) {
    if (!conv_entry_valid) {
        return;
    }
    GGML_ASSERT(accepted >= 0 && accepted <= conv_n);
    GGML_ASSERT(accepted == 0 || (conv_set >= 0 && conv_set < 2));
    size_t o = 0;
    for (int k = 0; k < 3; ++k) {
        axera_cmm & pool  = B(STATE_BANK[k + 1]);
        const size_t bytes = logical(STATE_BANK[k + 1]);
        const void * src = conv_entry.data() + o;
        if (accepted > 0) {
            void * p = (char *) VSNAP[conv_set][k]->vir + (size_t) (accepted - 1) * bytes;
            VSNAP[conv_set][k]->invalidate(p, bytes);
            src = p;
        }
        memcpy(pool->vir, src, bytes);
        pool->flush(pool->vir, bytes);
        o += bytes;
    }
    conv_entry_valid = false;
    conv_set         = -1;
    conv_n           = 0;
}

AX_ENGINE_IO_BUFFER_T qwen35_moe_sgl::state_io(const std::string & name) {
    const int kind = state_kind(name);
    GGML_ASSERT(kind >= 0);
    return axera_io(B(name), (uint32_t) logical(name));
}

void qwen35_moe_sgl::bind_state() {
    auto rebind = [&](call & c) {
        for (size_t i = 0; i < c.in.size(); ++i) {
            if (state_kind(c.k->desc->in[i].name) >= 0) {
                c.in[i] = state_io(c.k->desc->in[i].name);
            }
        }
    };
    rebind(dec);
    if (ver.k) {
        rebind(ver);
    }
}

// The recurrent state is always in the pools; bring the pending deltas in too.
void qwen35_moe_sgl::state_home() {
    materialize_deltas();
}

void qwen35_moe_sgl::zero_state() {
    page_x.reserve((size_t) PG * H);
    page_p.reserve((size_t) PG * 4);
    state_home();  // apply a verification prefix that the sampler accepted
    if (state_dirty) {  // CMM starts zero; a new sequence needs the memset only after an earlier one ran
        axera_timer tl("state zero");
        for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
            memset(B(n)->vir, 0, B(n)->size);
            B(n)->flush();
        }
    }
    delta_n          = 0;
    delta_live       = false;
    delta_set        = -1;
    conv_entry_valid = false;
    conv_set         = -1;
    conv_n           = 0;
    ver_c0           = -1;
    state_dirty = false;
    snap_cell  = -1;
    snap_parity = -1;
    state_cell = 0;
    page_x.clear();
    page_p.clear();
}

void qwen35_moe_sgl::save_state(int64_t cell) {
    GGML_ASSERT(cell % PG == 0);
    materialize_deltas();
    axera_timer tl("state save");
    if (cell == 0) {  // a fresh sequence's state is zero and load_state() restores it without a snapshot copy
        snap_cell = 0;
        snap_parity = -1;
        page_x.clear();
        page_p.clear();
        return;
    }
    size_t o = 0;
    for (int k = 0; k < 4; ++k) {
        axera_cmm & pool = B(STATE_BANK[k]);
        const size_t bytes = logical(STATE_BANK[k]);
        pool->invalidate(pool->vir, bytes);
        memcpy(snap.data() + o, pool->vir, bytes);
        o += bytes;
    }
    snap_cell = cell;
    snap_parity = -1;
    page_x.clear();
    page_p.clear();
}

void qwen35_moe_sgl::load_state() {
    axera_timer tl("state load");
    state_home();
    delta_n          = 0;
    delta_live       = false;
    conv_entry_valid = false;
    conv_set         = -1;
    conv_n           = 0;
    if (snap_cell == 0) {
        for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
            memset(B(n)->vir, 0, B(n)->size);
            B(n)->flush();
        }
        return;
    }
    if (snap_parity == 1) {  // the alternate set kept the page-start state: restore it into the current set
        size_t o = 0;
        for (auto [n, s2] : std::initializer_list<std::pair<const char *, axera_cmm *>>{
                 { "b0_attn_state_in", &ST2 }, { "b0_attn_q_conv_state_in", &CQ2 },
                 { "b0_attn_k_conv_state_in", &CK2 }, { "b0_attn_v_conv_state_in", &CV2 } }) {
            GGML_ASSERT((*s2)->size == B(n)->size);
            (*s2)->invalidate((*s2)->vir, (*s2)->size);
            memcpy(B(n)->vir, (*s2)->vir, B(n)->size);
            B(n)->flush();
            o += B(n)->size;
        }
        GGML_UNUSED(o);
        return;
    }
    size_t o = 0;
    for (int k = 0; k < 4; ++k) {
        axera_cmm & pool = B(STATE_BANK[k]);
        const size_t bytes = logical(STATE_BANK[k]);
        memcpy(pool->vir, snap.data() + o, bytes);
        pool->flush(pool->vir, bytes);
        o += bytes;
    }
}

// T (2 - 4) consecutive tokens from cell c0 through the verification kernel: the four rows' hidden
// states into h, one rank-1 state delta per row, and the small Conv snapshots.  The recurrent state
// stays in its pool at the entry cell; the accepted prefix is folded in by the next call.
ggml_status qwen35_moe_sgl::verify(const kv_view & kv, const ggml_bf16_t * xb, const int32_t * pb, int T, float * h) {
    GGML_ASSERT(ver.k && T >= 1 && T <= 4);
    const int64_t c0 = kv.cell_of(0);
    if (c0 % PG == 0) {
        materialize_deltas();
        save_state(c0);
    }
    {
        axera_timer tc("conv save");
        size_t      o = 0;
        for (int k = 0; k < 3; ++k) {
            axera_cmm &  pool  = B(STATE_BANK[k + 1]);
            const size_t bytes = logical(STATE_BANK[k + 1]);
            pool->invalidate(pool->vir, bytes);
            memcpy(conv_entry.data() + o, pool->vir, bytes);
            o += bytes;
        }
        conv_entry_valid = true;
    }
    memset(VX->vir, 0, VX->size);
    memcpy(VX->vir, xb, (size_t) T * H * sizeof(ggml_bf16_t));
    VX->flush();
    auto * pos = (int32_t *) VPOSN->vir;
    auto * pid = (int32_t *) VPIDS->vir;
    auto * loc = (int32_t *) VLOC->vir;
    const int rows = (int) (ver.k->in_bytes("positions") / sizeof(int32_t));
    const int axes = (int) (ver.k->in_bytes("position_ids") / (rows * sizeof(int32_t)));
    GGML_ASSERT(rows == 4 && axes >= 1 && axes <= 4);
    for (int r = 0; r < 4; ++r) {
        const int     rr = std::min(r, T - 1);
        const int64_t c  = c0 + rr;
        pos[r]           = (int32_t) c;
        for (int s = 0; s < axes; ++s) {
            pid[s * rows + r] = pb[(size_t) rr * 4 + std::min(s, 3)];
        }
        loc[2 * r]     = (int32_t) (c / PG);
        loc[2 * r + 1] = (int32_t) (c % PG);
    }
    const int64_t np = (c0 + T + PG - 1) / PG;
    GGML_ASSERT(np * PG <= man.context);
    ((int32_t *) VINDPTR->vir)[0] = 0;
    ((int32_t *) VINDPTR->vir)[1] = (int32_t) np;
    *(int32_t *) VNT->vir         = T;
    for (const axera_cmm * b : { &VPOSN, &VPIDS, &VLOC, &VINDPTR, &VNT }) {
        (*b)->flush();
    }
    const int stale    = delta_set >= 0 ? delta_set : 0;
    const int set      = 1 - (delta_live ? delta_set : stale);
    const int prev_set = delta_live ? delta_set : stale;
    static const char * SN[3] = { "cqsnap", "cksnap", "cvsnap" };
    bind_deltas(ver, set, prev_set);
    for (int k = 0; k < 3; ++k) {
        ver.in[ver.k->input(SN[k])] = axera_io(VSNAP[set][k], ver.k->in_bytes(SN[k]));
    }
    *(int32_t *) VNPREV->vir = delta_n;
    VNPREV->flush();
    ver.out[ver.k->output("output")] = axera_io(VOUT, ver.k->out_bytes("output"));
    {
        axera_timer tl("layers (verify)");
        ver.k->run(ver.in, ver.out, "verify");
    }
    VOUT->invalidate(VOUT->vir, (size_t) T * H * sizeof(ggml_bf16_t));
    for (int r = 0; r < T; ++r) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) VOUT->vir + (size_t) r * H, h + (size_t) r * H, H);
    }
    conv_set         = set;
    conv_n           = T;
    conv_entry_valid = true;
    for (int r = 0; r < T; ++r) {
        if (r > 0 && (c0 + r) % PG == 0) {  // a page's first cell inside the batch: the state before it
            axera_cmm &  state = B(STATE_BANK[0]);
            const size_t sb    = logical(STATE_BANK[0]);
            state->invalidate(state->vir, sb);
            memcpy(VST0->vir, state->vir, sb);
            const int saved_set = delta_set, saved_n = delta_n;
            const bool saved_live = delta_live;
            delta_set  = set;
            delta_n    = r;
            delta_live = true;
            materialize_deltas();
            restore_conv(r);
            save_state(c0 + r);
            memcpy(state->vir, VST0->vir, sb);
            state->flush();
            delta_set        = saved_set;
            delta_n          = saved_n;
            delta_live       = saved_live;
            conv_entry_valid = true;  // the bytes above were not overwritten
        }
        page_x.insert(page_x.end(), xb + (size_t) r * H, xb + (size_t) (r + 1) * H);
        page_p.insert(page_p.end(), pb + (size_t) r * 4, pb + (size_t) (r + 1) * 4);
    }
    delta_set  = set;
    delta_n    = T;
    delta_live = true;
    conv_set   = set;
    conv_n     = T;
    conv_entry_valid = true;
    ver_c0  = c0;
    ver_T   = T;
    state_cell = c0;  // the pool is at the entry cell until the next call accepts rows
    return GGML_STATUS_SUCCESS;
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
                    v.push_back(axera_io(META.get(), (char *) META->vir + META_VALID, d.bytes));
                } else if (n == "indptr") {
                    v.push_back(axera_io(META.get(), (char *) META->vir + META_INDPTR, d.bytes));
                } else if (n == "indices") {
                    v.push_back(axera_io(INDICES, d.bytes));
                } else if (n == "cache_location") {
                    v.push_back(axera_io(META.get(), (char *) META->vir + META_CLOC, d.bytes));
                } else if (n.find("scale2") != std::string::npos) {
                    v.push_back(axera_io(ONES, d.bytes));
                } else if (n.rfind("w_zeropoint_", 0) == 0) {
                    v.push_back(axera_io(ZERO, d.bytes));
                } else if (n == "paged_kv_cache") {
                    v.push_back(kv_io(ord[il], d.bytes));
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
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= kv_cells);
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
                *(int32_t *) ((char *) META->vir + META_VALID) = v;
                META->flush(META->vir, META_BYTES);
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
    ((int32_t *) ((char *) META->vir + META_INDPTR))[0] = 0;
    ((int32_t *) ((char *) META->vir + META_INDPTR))[1] = (int32_t) pages;
    *(int32_t *) ((char *) META->vir + META_LASTLEN)    = (int32_t) (pages ? hist - (pages - 1) * PG : 0);
    ((int32_t *) ((char *) META->vir + META_CLOC))[0]   = (int32_t) (write_cell / PG);
    ((int32_t *) ((char *) META->vir + META_CLOC))[1]   = (int32_t) (write_cell % PG);
}

// n token ids (1 .. 4096) through the 4096-row gather; the pad rows repeat the last id, the caller reads only its n
void qwen35_moe_sgl::embed_tokens(const int32_t * toks, int n) {
    GGML_ASSERT(emb && n > 0 && n <= WN);
    auto * ti = (int32_t *) EMBI->vir;
    for (int i = 0; i < WN; ++i) {
        ti[i] = toks[std::min(i, n - 1)];
    }
    EMBI->flush();
    axera_timer te("token embedding");
    emb4k.k->run(emb4k.in, emb4k.out, "token embedding (prefill)");
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16 (x) or f32 (xf, xf_nb1 bytes per row), their
// positions p4 [n][4]; the hidden states of the requested rows into the compact h (one [H] row per out_rows entry)
void qwen35_moe_sgl::prefill(int64_t c0, int n, const ggml_bf16_t * x, const float * xf, size_t xf_nb1,
                             const int32_t * toks, const int32_t * p4, std::vector<float> & h,
                             const std::vector<int> & out_rows) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= kv_cells);
    GGML_ASSERT((x != nullptr) + (xf != nullptr) + (toks != nullptr) == 1);
    const size_t xr = (size_t) H * sizeof(ggml_bf16_t);
    for (int r0 = 0; r0 < n; r0 += WN) {
        const int     live = std::min(WN, n - r0);
        const int64_t cell = c0 + r0;
        save_state(cell);
        // a batch that ends inside a page keeps that page's start as a restart point (per layer, at its own last page)
        const bool tail  = r0 + live == n && live % PG != 0;
        const int  lastp = tail ? live - live % PG : 0;
        {  // the first layer reads this padded chunk directly; the MoE writes the next layer's input to PX
            axera_timer tx("prefill: input copy + flush");
            if (toks) {
                embed_tokens(toks + r0, live);
            } else if (xf) {
                axera_host_pool().run(live, [&](int64_t i) {
                    ggml_fp32_to_bf16_row((const float *) ((const char *) xf + (size_t) (r0 + i) * xf_nb1),
                                          (ggml_bf16_t *) XB->vir + (size_t) i * H, H);
                });
            } else {
                memcpy(XB->vir, x + (size_t) r0 * H, (size_t) live * xr);
            }
            if (!toks) {
                memset((ggml_bf16_t *) XB->vir + (size_t) live * H, 0, (size_t) (WN - live) * xr);
                XB->flush();
            }
        }
        {  // every page's positions once, in its kernel layout; a page then only moves the pointer
            axera_timer tpos("prefill: positions");
            auto * pp = (int32_t *) POS->vir;
            const size_t page_ints = (size_t) AX * PG;
            const int pages_live = (live + PG - 1) / PG;
            for (int r = 0; r < pages_live * PG; ++r) {
                for (int a = 0; a < AX; ++a) {
                    pp[(size_t) (r / PG) * page_ints + (size_t) a * PG + r % PG] =
                        r < live ? p4[(size_t) r * 4 + a] : 0;
                }
            }
            POS->flush(POS->vir, (size_t) pages_live * page_ints * sizeof(int32_t));
        }
        axera_timer tl("prefill (chunk)");
        for (int il = 0; il < L; ++il) {
            const bool gdn = il % 4 != 3;
            for (int s = 0; s < live; s += PG) {
                const int     v    = std::min(PG, live - s);
                const int64_t c    = cell + s;
                call & pc = pre_attn[state_parity][il];
                const axera_cmm & src = il == 0 ? XB : PX;
                pc.in[pc.ix] = axera_io(src.get(), (char *) src->vir + (size_t) s * xr, (uint32_t) (PG * xr));
                if (gdn) {
                    pc.out[pc.ires] = axera_io(PA.get(), (char *) PA->vir + (size_t) s * xr, (uint32_t) (PG * xr));
                } else {
                    pc.out[pc.iout] = axera_io(PA.get(), (char *) PA->vir + (size_t) s * xr, (uint32_t) (PG * xr));
                    pc.in[pc.ipos]  = axera_io(POS.get(), (char *) POS->vir + (size_t) (s / PG) * AX * PG * sizeof(int32_t),
                                              pc.k->desc->in[pc.ipos].bytes);
                }
                {
                    axera_timer tpg("prefill: paging");
                    paging(c, c);
                    *(int32_t *) ((char *) META->vir + META_VALID) = v;
                    META->flush(META->vir, META_BYTES);
                }
                axera_timer ta(gdn ? "prefill: gdn (chunk, layer)" : "prefill: attn (chunk, layer)");
                pc.k->run(pc.in, pc.out, gdn ? "prefill gdn" : "prefill attn");
                if (gdn) {
                    states_swap();
                }
            }
            if (live < WN) {
                *(int32_t *) ((char *) META->vir + META_VALID) = live;
                META->flush(META->vir, META_BYTES);
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
        axera_timer tconv("prefill: output rows");
        std::vector<int> want;
        for (size_t i = 0; i < out_rows.size(); ++i) {
            if (out_rows[i] >= r0 && out_rows[i] < r0 + live) {
                want.push_back((int) i);
            }
        }
        if (!want.empty()) {
            axera_host_pool().run((int64_t) want.size(), [&](int64_t i) {
                const int slot = want[(size_t) i];
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) PX->vir + (size_t) (out_rows[slot] - r0) * H,
                                      h.data() + (size_t) slot * H, H);
            });
        }
        if (!at_page) {  // the last partial page: from its start, where the per-layer state was snapshotted
            if (tail && lastp > 0) {
                snap_cell = cell + lastp;
                snap_parity = 1;  // the state set the last page read from still holds its start state
            }
            page_x.resize((size_t) (live - lastp) * H);
            if (toks) {
                XB->invalidate(XB->vir, (size_t) live * xr);
                memcpy(page_x.data(), (const ggml_bf16_t *) XB->vir + (size_t) lastp * H,
                       page_x.size() * sizeof(ggml_bf16_t));
            } else if (xf) {
                for (int r = lastp; r < live; ++r) {
                    ggml_fp32_to_bf16_row((const float *) ((const char *) xf + (size_t) (r0 + r) * xf_nb1),
                                          page_x.data() + (size_t) (r - lastp) * H, H);
                }
            } else {
                memcpy(page_x.data(), x + (size_t) (r0 + lastp) * H, page_x.size() * sizeof(ggml_bf16_t));
            }
            page_p.assign(p4 + (size_t) (r0 + lastp) * 4, p4 + (size_t) (r0 + live) * 4);
        }
    }
    state_cell = c0 + n;
    state_dirty = true;
}

// one token at cell c > 0, its history the c cells before it
void qwen35_moe_sgl::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h) {
    {
        axera_timer ts("decode host: sync states");
        sync_dec_states();
    }
    if (c % WN == 0) {  // a chunk's first cell: the state here is the restart point
        save_state(c);
    }
    if (x) {
        axera_timer tx("decode host: x");
        memcpy(DX->vir, x, H * sizeof(ggml_bf16_t));
        DX->flush();
    } else {  // the NPU embedding wrote one row straight into DX
        axera_timer tx("decode host: x");
        DX->invalidate(DX->vir, H * sizeof(ggml_bf16_t));
        x = (const ggml_bf16_t *) DX->vir;
    }
    {
        axera_timer tp("decode host: positions");
        for (int a = 0; a < AX && a < 4; ++a) {
            ((int32_t *) DPOS->vir)[a] = p4[a];
        }
        DPOS->flush();
        paging(c, c);
        META->flush(META->vir, META_VALID);
    }
    {
        axera_timer tl("layers (decode)");
        dec.k->run(dec.in, dec.out, "decode");
    }
    // debug: the same call again immediately (the IO and state are unchanged): the second call's time separates a
    // kernel's real cost from the state that a different model's call in between leaves behind
    static const bool dec_twice = getenv("GGML_AXERA_DECODE_TWICE") != nullptr;
    if (dec_twice) {
        static int64_t n = 0, us = 0;
        const auto     t0 = std::chrono::steady_clock::now();
        dec.k->run(dec.in, dec.out, "decode (twice)");
        us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        if (++n % 8 == 0) {
            fprintf(stderr, "decode twice: second %.3f ms over %lld\n", us / 1e3 / n, (long long) n);
        }
    }
    {
        axera_timer to("decode host: output");
        if (h) {  // (a fused head reads the bf16 row on the NPU)
            DOUT->invalidate(DOUT->vir, H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) DOUT->vir, h, H);
        }
        page_x.insert(page_x.end(), x, x + H);
        page_p.insert(page_p.end(), p4, p4 + 4);
    }
    state_cell = c + 1;
    state_dirty = true;
}

void qwen35_moe_sgl::prewarm(ggml_cgraph * g) {
    // the name map is the adapter's only per-graph one-time host cost; llama.cpp prebuilds the next decode graph at
    // the end of a prompt's call, so building its map there keeps it out of the first generated token
    names.of(g);
}

ggml_status qwen35_moe_sgl::compute(ggml_cgraph * g) {
    static const bool ct_on = getenv("LLAMA_GRAPH_PROF") != nullptr;
    const int64_t     ct_0  = ct_on ? ggml_time_us() : 0;
    const auto &  t       = names.of(g);
    const int64_t ct_names = ct_on ? ggml_time_us() : 0;
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return compute_mtp(g);
    }
    axera_timer   t_pre1("compute pre (early)");
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
    GGML_ASSERT(x0 && x0->src[0]);
    ggml_tensor * xin = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    t_pre1.stop();
    axera_timer t_pre2(T > 1 ? "compute pre (prompt)" : "compute pre (decode)");
    // the layer count and the graph's ROPE node are model constants: scan for them once, not on every decode token
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data) {
        axera_timer tl("compute pre: layer count");
        int         L_ = 0;
        while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
            ++L_;
        }
        tl.stop();
        axera_timer trope("compute pre: rope scan");
        const ggml_tensor * rope_node = nullptr;
        for (int i = 0; i < g->n_nodes && !rope_node; i++) {
            rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
        }
        trope.stop();
        GGML_ASSERT(rope_node && "axera qwen35_35b_a3b: no ROPE node in the graph");
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    axera_timer temb("compute pre: embed");
    const ggml_tensor * inp_tokens = names.find("inp_tokens");
    const bool npu_emb = emb && inp_tokens && inp_tokens->type == GGML_TYPE_I32 && inp_tokens->ne[0] == T;
    // The table is a dropped file mapping here, so llama.cpp's CPU GET_ROWS faults a row in and walks the generic
    // backend (0.18 ms/token, measured); the one-row NPU gather costs more (0.32 ms). Read the bf16 copy the loader
    // already keeps on the NPU (EMBW) directly instead: one 4 KiB memcpy.
    const bool cpu_embed = npu_emb && T == 1;
    std::vector<ggml_bf16_t> xb1;
    bool direct_embed = false;
    if (!npu_emb || cpu_embed) {
        GGML_ASSERT(xin->type == GGML_TYPE_F32 && ggml_is_contiguous(xin));
        if (cpu_embed && emb) {  // the token's row of the NPU table's host-visible bf16 copy
            const int64_t id = ((const int32_t *) inp_tokens->data)[0];
            GGML_ASSERT(id >= 0 && (size_t) id * H * sizeof(ggml_bf16_t) + H * sizeof(ggml_bf16_t) <= EMBW->size);
            xb1.resize(H);
            memcpy(xb1.data(), (const ggml_bf16_t *) EMBW->vir + (size_t) id * H, H * sizeof(ggml_bf16_t));
            direct_embed = true;
        } else if (cpu_embed && xin->op == GGML_OP_GET_ROWS && xin->src[0] && xin->src[1] &&
            xin->src[0]->type == GGML_TYPE_F32 && xin->src[0]->ne[0] == H) {
            const ggml_tensor * w  = xin->src[0];
            const int64_t       id = ((const int32_t *) xin->src[1]->data)[0];
            GGML_ASSERT(id >= 0 && id < w->ne[1]);
            xb1.resize(H);
            ggml_fp32_to_bf16_row((const float *) ((const char *) w->data + id * w->nb[1]), xb1.data(), H);
            direct_embed = true;
        }
        if (!direct_embed && xin->op != GGML_OP_NONE) {
            int n0 = 0;
            while (n0 < g->n_nodes && g->nodes[n0] != x0) {
                ++n0;
            }
            cpu_compute(g, g->nodes, n0, "input (cpu)");
        }
        x0 = xin;
    }
    temb.stop();
    axera_timer tkv("compute pre: kv + cells");
    const int32_t * token_ids = npu_emb && !cpu_embed ? (const int32_t *) inp_tokens->data : nullptr;
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, man.context, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera qwen35_35b_a3b: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    {
        axera_timer tchk("prompt: cell checks");
        for (int i = 1; i < T; ++i) {
            if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
                GGML_LOG_ERROR("axera qwen35_35b_a3b: a batch of non-consecutive cells\n");
                return GGML_STATUS_FAILED;
            }
        }
    }
    tkv.stop();
    axera_timer tpos("compute pre: positions");
    const int32_t *      pos = (const int32_t *) inp_pos->data;
    const bool           mr  = inp_pos->ne[0] == 4 * (int64_t) T;  // M-RoPE: [4][T], else [T]
    p4b.resize((size_t) T * 4);
    std::vector<int32_t> & pb = p4b;
    {
        axera_timer tpos("prompt: positions fill");
        for (int r = 0; r < T; ++r) {
            for (int s = 0; s < 4; ++s) {
                pb[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
            }
        }
    }
    tpos.stop();
    axera_timer tout("compute pre: out rows");
    const int n_out = (int) logits->ne[1];
    GGML_ASSERT(n_out >= 0 && n_out <= T);
    const ggml_tensor * out_ids = n_out > 0 && n_out < T ? names.find("out_ids") : nullptr;
    if (out_ids && (out_ids->type != GGML_TYPE_I32 || out_ids->ne[0] != n_out)) {
        out_ids = nullptr;
    }
    if (n_out > 0 && n_out < T && !out_ids) {
        for (int i = 0; i < g->n_nodes; i++) {
            const ggml_tensor * node = g->nodes[i];
            if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
                node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
                out_ids = node->src[1];
            }
        }
    }
    GGML_ASSERT(n_out == 0 || n_out == T || out_ids);
    orows.resize((size_t) n_out);
    std::vector<int> & out_rows = orows;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        out_rows[(size_t) r] = row;
    }
    hb.resize((size_t) n_out * H);
    std::vector<float> & h = hb;
    tout.stop();
    t_pre2.stop();
    const int64_t ct_pre = ct_on ? ggml_time_us() : 0;
    bool combined_dec = false;
    if (n_out > 0) {
        head.bind(man.get("lm_head"), logits->src[0]);
        combined_dec = head.bind_combined(man.get("lm_head"), logits->src[0]);
        combined_dec = combined_dec && head.combined_ready(g, names);
        if (man.has("lm_head.verify")) {
            head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
        }
    }
    bool fused_dec = false;
    if (!combined_dec && T == 1 && n_out == 1) {
        if (!fh_loaded) {
            fh_loaded = true;
            const char * e = getenv("GGML_AXERA_FUSED_HEAD");
            if (!e || atoi(e) != 0) {
                const auto & fk = man.load_memory("lm_head_fused", std::vector<char>(
                    (const char *) FUSED_HEAD_AXMODEL, (const char *) FUSED_HEAD_AXMODEL + FUSED_HEAD_AXMODEL_SIZE));
                fh_ok = head.bind_fused(fk, logits->src[0]);
            }
        }
        fused_dec = fh_ok && head.fused_ready(g, names);
    }
    const int64_t ct_bind = ct_on ? ggml_time_us() : 0;
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const int64_t      c0 = kv.cell_of(0);
    if (ver_c0 >= 0 && c0 != 0) {
        if (c0 < ver_c0 || c0 > ver_c0 + ver_T) {
            GGML_LOG_ERROR("axera qwen35_moe_sgl: cell %lld is outside the last verify's %lld..%lld window\n",
                           (long long) c0, (long long) ver_c0, (long long) (ver_c0 + ver_T));
            return GGML_STATUS_FAILED;
        }
        // The sampler accepted c0 - ver_c0 rows.  The recurrent state pool is
        // still at the verify's entry cell; keep the accepted prefix pending
        // for the next verify prologue (or gdn.apply before decode / prefill).
        const int accepted = (int) (c0 - ver_c0);
        restore_conv(accepted);
        delta_n    = accepted;
        delta_live = accepted > 0;
        if (snap_cell > c0) {  // (no restart point of c0's page kept)
            snap_cell = -1;
            page_x.clear();
            page_p.clear();
        } else if (snap_cell >= 0) {
            page_x.resize((size_t) (c0 - snap_cell) * H);
            page_p.resize((size_t) (c0 - snap_cell) * 4);
        }
        state_cell = c0;
        ver_c0     = -1;
        ver_T      = 0;
    }
    if (c0 == 0) {
        ver_c0 = -1;
        ver_T  = 0;
    }
    if (c0 == 0) {  // a new sequence
        zero_state();
    }
    if (T > 1 && T <= 4 && ver.k && c0 > 0 && c0 + 4 <= man.context) {
        axera_timer tv("verify (all layers)");
        std::vector<ggml_bf16_t> vxb((size_t) T * H);
        if (token_ids) {
            embed_tokens(token_ids, T);
            XB->invalidate(XB->vir, (size_t) T * H * sizeof(ggml_bf16_t));
            memcpy(vxb.data(), XB->vir, vxb.size() * sizeof(ggml_bf16_t));
        } else {
            axera_host_pool().run(T, [&](int64_t r) {
                ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                                      vxb.data() + (size_t) r * H, H);
            });
        }
        const ggml_status st = verify(kv, vxb.data(), pb.data(), T, h.data());
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
    } else if (T > 1 || c0 == 0) {
        state_home();
        if (c0 == state_cell && c0 % PG == 0) {  // sequential batch from a page start: the state is exactly here
            axera_timer tp(T > 256 ? "prefill (prompt)" : "prefill (all pages)");
            if (token_ids) {
                prefill(c0, T, nullptr, nullptr, 0, token_ids, pb.data(), h, out_rows);
            } else if (direct_embed) {
                prefill(c0, T, xb1.data(), nullptr, 0, nullptr, pb.data(), h, out_rows);
            } else {
                prefill(c0, T, nullptr, (const float *) x0->data, x0->nb[1], nullptr, pb.data(), h, out_rows);
            }
        } else {
            // a rewind (e.g. the server's prompt cache): from c0's page start and the state saved there
            const int64_t s0 = c0 - c0 % PG;
            const int     k  = (int) (c0 - s0);
            if (snap_cell != s0 || (k > 0 && (int) (page_x.size() / H) < k)) {
                GGML_LOG_ERROR("axera qwen35_moe_sgl: a prompt from cell %lld, whose page-start state is not kept\n",
                               (long long) c0);
                return GGML_STATUS_FAILED;
            }
            page_x.resize((size_t) k * H);  // (a kept prefix, e.g. the server's prompt cache: the cells after it are gone)
            page_p.resize((size_t) k * 4);
            std::vector<ggml_bf16_t> xs;
            std::vector<int32_t>     ps;
            // k == 0 still needs the saved state when tokens after the page start have been decoded
            if (k > 0 || state_cell != s0) {
                load_state();
            }
            if (k > 0) {
                xs = page_x;
                ps = page_p;
            }
            std::vector<ggml_bf16_t> xb((size_t) T * H);
            axera_timer              txb("prompt: x bf16");
            if (token_ids) {
                embed_tokens(token_ids, T);
                XB->invalidate(XB->vir, (size_t) T * H * sizeof(ggml_bf16_t));
                memcpy(xb.data(), XB->vir, xb.size() * sizeof(ggml_bf16_t));
            } else {
                axera_host_pool().run(T, [&](int64_t r) {
                    ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                                          xb.data() + (size_t) r * H, H);
                });
            }
            xs.insert(xs.end(), xb.begin(), xb.end());
            ps.insert(ps.end(), pb.begin(), pb.end());
            std::vector<int> shifted(out_rows.size());
            for (size_t i = 0; i < out_rows.size(); ++i) {
                shifted[i] = out_rows[i] + k;
            }
            axera_timer tp(T > 256 ? "prefill (prompt)" : "prefill (all pages)");
            prefill(s0, k + T, xs.data(), nullptr, 0, nullptr, ps.data(), h, shifted);
        }
    } else {
        if (c0 != state_cell) {
            GGML_LOG_ERROR("axera qwen35_moe_sgl: a decode from cell %lld, not the state's cell %lld\n", (long long) c0,
                           (long long) state_cell);
            return GGML_STATUS_FAILED;
        }
        materialize_deltas();
        const bool direct_x = combined_dec && T == 1 && n_out == 1;
        float * hd = (fused_dec || direct_x) ? nullptr : h.data();
        if (direct_embed) {
            decode(c0, xb1.data(), pb.data(), hd);
        } else if (token_ids) {
            uint32_t one = (uint32_t) *token_ids;
            memcpy(EMBI->vir, &one, sizeof(one));
            EMBI->flush(EMBI->vir, sizeof(one));
            axera_timer te("token embedding (decode)");
            emb1.k->run(emb1.in, emb1.out, "token embedding (decode)");
            te.stop();
            decode(c0, nullptr, pb.data(), hd);
        } else {
            std::vector<ggml_bf16_t> xb((size_t) H);
            ggml_fp32_to_bf16_row((const float *) x0->data, xb.data(), H);
            decode(c0, xb.data(), pb.data(), hd);
        }
    }
    if (n_out > 0) {
        axera_timer th(T > 256 ? "output head (prompt)" : T > 1 ? "output head (warmup)" : "output head (decode)");
        if (combined_dec) {
            const bool direct_x = T == 1 && n_out == 1;
            head.run_combined(g, names, h, T, &out_rows, direct_x ? &DOUT : nullptr);
        } else if (fused_dec) {
            head.run_fused(g, names, DOUT);
        } else {
            head.run(g, names, h, T, nullptr, nullptr, &out_rows);
        }
    }
    if (warmup_active()) {  // the warmup's dummy sequence must not make the first real prompt clear the state
        zero_state();
    }
    tm_all.stop();
    if (ct_on && T == 1) {
        static int64_t cn = 0, ct_names_us = 0, ct_pre_us = 0, ct_bind_us = 0, ct_all_us = 0, ct_total_us = 0;
        const int64_t ct_end = ggml_time_us();
        ++cn;
        ct_names_us += ct_names - ct_0;
        ct_pre_us   += ct_pre - ct_names;
        ct_bind_us  += ct_bind - ct_pre;
        ct_all_us   += ct_end - ct_bind;
        ct_total_us += ct_end - ct_0;
        if (cn % 32 == 0) {
            fprintf(stderr, "adapter compute: total %.3f = names %.3f + pre %.3f + bind %.3f + graph %.3f ms/token (%lld)\n",
                    ct_total_us / 1e3 / cn, ct_names_us / 1e3 / cn, ct_pre_us / 1e3 / cn, ct_bind_us / 1e3 / cn,
                    ct_all_us / 1e3 / cn, (long long) cn);
        }
    }
    ggml_tensor * hn = names.find("h_nextn");  // the MTP block's input: every row's final norm
    if (hn && (hn->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        axera_timer tn("h_nextn");
        GGML_ASSERT(hn->type == GGML_TYPE_F32 && hn->ne[0] == H && hn->ne[1] == T);
        float         eps;
        const float * w = final_norm(hn, H, &eps);
        axera_host_pool().run(T, [&](int64_t r) {
            const float * x = h.data() + (size_t) r * H;
            float *       y = (float *) ((char *) hn->data + r * hn->nb[1]);
            double        ss = 0;
            for (int i = 0; i < H; ++i) {
                ss += (double) x[i] * x[i];
            }
            const float sc = 1.0f / sqrtf((float) (ss / H) + eps);
            for (int i = 0; i < H; ++i) {
                y[i] = x[i] * sc * w[i];
            }
        });
        if (ggml_backend_buffer_is_axera(hn->buffer)) {
            ((ggml_backend_axera_buffer_context *) hn->buffer->context)->flush(hn->data, ggml_nbytes(hn));
        }
    }
    return GGML_STATUS_SUCCESS;
}


// The MTP block (blk.<il>.nextn.*: eh_proj, one full-attention layer with its own paged KV, MoE + shared expert,
// shared head norm) as the kernel's runtime tensors. The attention, shared expert and eh_proj weights use the same
// NVFP4 encoding as the target's banks; the routed experts keep the GGUF's MXFP4 codes/scales (E8M0 per 32 inputs),
// with gate and up fused per expert as the kernel reads them.
void qwen35_moe_sgl::init_mtp(const std::map<std::string, ggml_tensor *> & t, int il, const rope & rq) {
    axera_timer    tb("init: mtp");
    const kernel & k1 = man.get("mtp.t1");
    const kernel & kp = man.get("mtp.prompt");
    GGML_ASSERT(k1.in_bytes("paged_kv") == kp.in_bytes("paged_kv"));
    if (man.has("mtp.m8")) {
        const kernel & k8 = man.get("mtp.m8");
        GGML_ASSERT(k8.in_bytes("paged_kv") == k1.in_bytes("paged_kv"));
    }
    const size_t mtp_page_kv = (size_t) 2 * PG * 2 * 256 * 2;
    GGML_ASSERT(k1.in_bytes("paged_kv") % mtp_page_kv == 0);
    const int mtp_np = (int) (k1.in_bytes("paged_kv") / mtp_page_kv);
    mtp_kv_pages     = mtp_np;
    if (kv_cells > 0) {
        mtp_kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(mtp_np, (kv_cells + PG - 1) / PG));
    }
    if (const char * e = getenv("GGML_AXERA_MTP_KV_PAGES")) {
        mtp_kv_pages = std::max(1, std::min(mtp_np, atoi(e)));
    }
    mtp_kv_bytes = (size_t) mtp_kv_pages * mtp_page_kv;
    // one buffer per weight input, shared by t1 / m8 / prompt (they have the same layout)
    for (const auto & d : k1.desc->in) {
        if (d.name.rfind("w_", 0) == 0 && !mtp_w.count(d.name)) {
            mtp_w[d.name] = axera_cmm_alloc(d.bytes);
        }
    }
    auto W = [&](const char * n) -> axera_cmm & {
        auto it = mtp_w.find(n);
        if (it == mtp_w.end()) {
            GGML_ABORT("axera qwen35_moe_sgl: MTP: no weight %s in the kernel\n", n);
        }
        return it->second;
    };
    auto Wn = [&](const char * n) -> const axera_cmm & { return W(n); };

    std::vector<const ggml_tensor *> read;
    const std::string                 pre = "blk." + std::to_string(il) + ".";
    auto T = [&](const char * n) -> const ggml_tensor * {
        const std::string nm = pre + n;
        auto              it = t.find(nm);
        if (it == t.end()) {
            GGML_ABORT("axera qwen35_moe_sgl: MTP: %s not in the graph\n", nm.c_str());
        }
        read.push_back(it->second);
        return it->second;
    };
    auto put_bf16 = [&](const char * n, const std::vector<float> & v, float scale = 1.0f) {
        const size_t bytes = v.size() * sizeof(ggml_bf16_t);
        GGML_ASSERT(W(n)->size >= bytes);
        std::vector<float> u(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            u[i] = v[i] * scale;
        }
        ggml_fp32_to_bf16_row(u.data(), (ggml_bf16_t *) W(n)->vir, (int64_t) v.size());
    };
    auto put_f32 = [&](const char * n, const std::vector<float> & v) {
        GGML_ASSERT(W(n)->size >= v.size() * sizeof(float));
        memcpy(W(n)->vir, v.data(), v.size() * sizeof(float));
    };
    auto put_nv = [&](const char * base, const ggml_tensor * w) {
        const std::string qn = std::string("w_") + base;
        const std::string sn = std::string("w_scale_") + base;
        const std::string zn = std::string("w_scale2_") + base;
        const int64_t     K  = w->ne[0], N = ggml_nrows(w);
        GGML_ASSERT((size_t) N * (K / 2) <= Wn(qn.c_str())->size && (size_t) N * (K / 16) <= Wn(sn.c_str())->size);
        const float s2 = axera_quantize_nvfp4_slice(w, 0, N, (uint8_t *) W(qn.c_str())->vir, K / 2,
                                                    (uint8_t *) W(sn.c_str())->vir, K / 16);
        GGML_ASSERT(Wn(zn.c_str())->size >= sizeof(float));
        memcpy(W(zn.c_str())->vir, &s2, sizeof(float));
    };

    // RMSNorm weights (the GGUF converter folded the +1 already) and the attention q norm's scale
    put_bf16("w_enorm", f32_of(T("nextn.enorm.weight")));
    put_bf16("w_hnorm", f32_of(T("nextn.hnorm.weight")));
    put_bf16("w_attn_norm", f32_of(T("attn_norm.weight")));
    put_bf16("w_moe_input_norm_w", f32_of(T("post_attention_norm.weight")));
    put_bf16("w_shared_head_norm", f32_of(T("nextn.shared_head_norm.weight")));
    put_f32("w_attn_q_norm", [&] {
        std::vector<float> v = f32_of(T("attn_q_norm.weight"));
        const float        s = 1.0f / std::sqrt(256.0f);
        for (float & x : v) {
            x *= s;
        }
        return v;
    }());
    put_bf16("w_attn_k_norm", f32_of(T("attn_k_norm.weight")));

    // the rotary table: [max positions][cos | sin of ROT / 2] f32, theta and freq scale of the graph's ROPE node
    {
        const int64_t R = (int64_t) rq.inv_freq.size();
        GGML_ASSERT(R == 32 && Wn("w_attn_cos_sin_cache")->size % (2 * R * sizeof(float)) == 0);
        const int64_t P = (int64_t) (Wn("w_attn_cos_sin_cache")->size / (2 * R * sizeof(float)));
        float *       d = (float *) W("w_attn_cos_sin_cache")->vir;
        for (int64_t q = 0; q < P; ++q) {
            for (int64_t i = 0; i < R; ++i) {
                const double th = (double) q * rq.freq_scale * rq.inv_freq[i];
                d[q * 2 * R + i]     = (float) std::cos(th);
                d[q * 2 * R + R + i] = (float) std::sin(th);
            }
        }
    }

    put_nv("eh", T("nextn.eh_proj.weight"));
    put_nv("q", T("attn_q.weight"));
    put_nv("k", T("attn_k.weight"));
    put_nv("v", T("attn_v.weight"));
    put_nv("o", T("attn_output.weight"));
    put_bf16("w_route", f32_of(T("ffn_gate_inp.weight")));
    put_f32("w_scale_route", std::vector<float>(256, 1.0f));
    put_f32("w_scale2_route", std::vector<float>(1, 1.0f));
    put_bf16("w_shared_expert_gate", f32_of(T("ffn_gate_inp_shexp.weight")));
    for (const char * n : { "gate", "up", "down" }) {
        const std::string base = std::string("shared_expert_mlp_") + n;
        put_nv(base.c_str(), T((std::string("ffn_") + n + "_shexp.weight").c_str()));
    }
    for (const char * n : { "q", "k", "v", "o" }) {
        std::vector<float> z(1, 0.0f);
        put_f32((std::string("w_zeropoint_") + n).c_str(), z);
    }

    // routed experts: the GGUF's MXFP4 rows as they are (codes [K / 2], E8M0 scales [K / 32]); gate | up rows
    {
        const ggml_tensor * g = T("ffn_gate_exps.weight");
        const ggml_tensor * u = T("ffn_up_exps.weight");
        const ggml_tensor * d = T("ffn_down_exps.weight");
        for (const ggml_tensor * w : { g, u, d }) {
            GGML_ASSERT(w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w) && w->ne[0] % 32 == 0);
            GGML_ASSERT(w->nb[2] == (size_t) (w->ne[1] * w->ne[0] / 32 * 17));
        }
        const int64_t E = g->ne[2], F = g->ne[1], K = g->ne[0];
        GGML_ASSERT(u->ne[0] == K && u->ne[1] == F && u->ne[2] == E && d->ne[0] == F && d->ne[1] == K && d->ne[2] == E);
        auto * qg = (uint8_t *) W("w_gate")->vir;
        auto * sg = (uint8_t *) W("w_scale_gate")->vir;
        auto * qd = (uint8_t *) W("w_down")->vir;
        auto * sd = (uint8_t *) W("w_scale_down")->vir;
        GGML_ASSERT(Wn("w_gate")->size >= (size_t) E * 2 * F * (K / 2) &&
                    Wn("w_scale_gate")->size >= (size_t) E * 2 * F * (K / 32) &&
                    Wn("w_down")->size >= (size_t) E * d->ne[1] * (d->ne[0] / 2) &&
                    Wn("w_scale_down")->size >= (size_t) E * d->ne[1] * (d->ne[0] / 32));
        axera_host_pool().run(E, [&](int64_t e) {
            const size_t gq = (size_t) 2 * F * (K / 2), gs = (size_t) 2 * F * (K / 32);
            const size_t dq = (size_t) d->ne[1] * (d->ne[0] / 2), ds = (size_t) d->ne[1] * (d->ne[0] / 32);
            axera_mxfp4_rows((const uint8_t *) g->data + (size_t) e * g->nb[2], F, K, qg + e * gq, sg + e * gs);
            axera_mxfp4_rows((const uint8_t *) u->data + (size_t) e * u->nb[2], F, K, qg + e * gq + F * (K / 2),
                             sg + e * gs + F * (K / 32));
            axera_mxfp4_rows((const uint8_t *) d->data + (size_t) e * d->nb[2], d->ne[1], d->ne[0], qd + e * dq,
                             sd + e * ds);
        });
        float * s2g = (float *) W("w_scale2_gate")->vir;
        float * s2d = (float *) W("w_scale2_down")->vir;
        for (int64_t e = 0; e < E; ++e) {
            s2g[2 * e] = s2g[2 * e + 1] = 1.0f;
            s2d[e]                      = 1.0f;
        }
        if (!axera_keep_pages()) {
            for (const ggml_tensor * w : { g, u, d }) {
                axera_drop_mapped(w);
            }
        }
    }
    for (auto & [n, c] : mtp_w) {
        c->flush();
    }

    // the calls: the weights shared, the rows / paging / outputs of each kernel variant
    auto bind = [&](mtp_call & mc, const kernel & k) {
        mc.c   = {};
        mc.c.k = &k;
        for (const auto & d : k.desc->in) {
            if (d.name.rfind("w_", 0) == 0) {
                auto it = mtp_w.find(d.name);
                if (it == mtp_w.end() || it->second->size < d.bytes) {
                    GGML_ABORT("axera qwen35_moe_sgl: MTP: %s %s: weight %s is %u bytes\n",
                               k.desc->role.c_str(), k.desc->file.c_str(), d.name.c_str(), d.bytes);
                }
                mc.c.in.push_back(axera_io(it->second, d.bytes));
                continue;
            }
            axera_cmm * b = d.name == "e" ? &mc.e : d.name == "h" ? &mc.h : d.name == "position_ids" ? &mc.pids :
                            d.name == "positions" ? &mc.pos : d.name == "indptr" ? &mc.indptr :
                            d.name == "indices" ? &MINDICES : d.name == "cache_location" ? &mc.loc :
                            d.name == "ntok" ? &mc.ntok : d.name == "paged_kv" ? &MKV : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen35_moe_sgl: MTP: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(b == &MKV ? mtp_kv_bytes : d.bytes);
                if (b == &MKV) {
                    memset((*b)->vir, 0, (*b)->size);
                    (*b)->flush();
                }
            }
            GGML_ASSERT(b == &MKV ? (*b)->size >= mtp_kv_bytes : (*b)->size >= d.bytes);
            mc.c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : k.desc->out) {
            axera_cmm * b = d.name == "hnext" ? &mc.hnext : d.name == "hraw" ? &mc.hraw :
                            d.name == "kv" || d.name == "paged_kv" ? &MKV : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen35_moe_sgl: MTP: %s: unexpected output %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(b == &MKV ? mtp_kv_bytes : d.bytes);
                if (b == &MKV) {
                    memset((*b)->vir, 0, (*b)->size);
                    (*b)->flush();
                }
            }
            GGML_ASSERT(b == &MKV ? (*b)->size >= mtp_kv_bytes : (*b)->size >= d.bytes);
            mc.c.out.push_back(axera_io(*b, d.bytes));
        }
    };
    bind(mtp1, k1);
    bind(mtpP, kp);
    if (man.has("mtp.m8")) {
        bind(mtp8, man.get("mtp.m8"));
    } else if (man.has("mtp.m4")) {  // a 4-row catch-up variant (the board rejects the 8-row one)
        bind(mtp8, man.get("mtp.m4"));
    }
    GGML_ASSERT(MKV && MKV->size >= mtp_kv_bytes && MINDICES &&
                MINDICES->size >= k1.in_bytes("indices"));
    GGML_LOG_INFO("axera qwen35_moe_sgl: MTP KV pool of %d of the kernels' %d pages, %.2f GiB\n", mtp_kv_pages,
                  mtp_np, mtp_kv_bytes / 1073741824.0);
    for (int i = 0; i < (int) (MINDICES->size / sizeof(int32_t)); ++i) {
        ((int32_t *) MINDICES->vir)[i] = i;
    }
    MINDICES->flush();
}

// A draft context's MTP graph: token embedding e (the token rows) with the target's final-norm hidden h of the
// previous cell through blk.<il>, at the draft context's own cells. h_nextn / the logits of the output rows.
ggml_status qwen35_moe_sgl::compute_mtp(ggml_cgraph * g) {
    ggml_tensor * hin  = names.find("mtp_h_input");
    ggml_tensor * hn   = names.find("h_nextn");
    ggml_tensor * pos  = names.find("inp_pos");
    ggml_tensor * kidx = names.find("attn_inp_k_idxs");
    ggml_tensor * kqm  = names.find("attn_inp_kq_mask");
    const ggml_tensor * embd = nullptr;
    int           il = -1;
    for (int i = 0; i < g->n_nodes && !embd; i++) {
        const char * n = g->nodes[i]->name;
        if (strncmp(n, "mtp_enorm-", 10) == 0) {
            il                    = atoi(n + 10);
            const ggml_tensor * m = g->nodes[i];
            const ggml_tensor * r = m->src[0]->op == GGML_OP_RMS_NORM ? m->src[0] : m->src[1];
            embd                  = r->op == GGML_OP_RMS_NORM ? r->src[0] : nullptr;
            if (embd && embd->op != GGML_OP_NONE) {
                int n0 = 0;
                while (n0 < g->n_nodes && g->nodes[n0] != r) {
                    ++n0;
                }
                cpu_compute(g, g->nodes, n0, "mtp input (cpu)");
            }
        }
    }
    if (!embd || !hn || !pos || !kidx || !kqm || il < 0 || !bound) {
        GGML_LOG_ERROR("axera qwen35_moe_sgl: an MTP graph before the target model's or of an unexpected shape "
                       "(embd %d h_nextn %d inp_pos %d k_idxs %d kq_mask %d layer %d target %d)\n",
                       !!embd, !!hn, !!pos, !!kidx, !!kqm, il, !!bound);
        return GGML_STATUS_FAILED;
    }
    const int T = (int) hin->ne[1];
    GGML_ASSERT(hin->type == GGML_TYPE_F32 && hin->ne[0] == H && embd->type == GGML_TYPE_F32 && embd->ne[0] == H &&
                embd->ne[1] == T && hn->ne[1] == T);
    const ggml_tensor * w0 = names.at("blk." + std::to_string(il) + ".nextn.eh_proj.weight");
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    if (!rope_node) {
        GGML_LOG_ERROR("axera qwen35_moe_sgl: MTP graph without a ROPE node\n");
        return GGML_STATUS_FAILED;
    }
    const rope rq(rope_node, pos, T);
    if (mtp_bound != w0->data) {
        init_mtp(names.t, il, rq);
        mtp_bound = w0->data;
    }
    const kv_view kv(kidx, kqm, man.context, PG);
    auto cell_of = [&](int r) -> int64_t {
        const int64_t c = kv.cell_of(r);
        if (c < 0 || c >= man.context) {
            GGML_LOG_ERROR("axera qwen35_moe_sgl: MTP row %d has cell %lld outside [0,%lld); k_idxs[%d] = %lld\n",
                           r, (long long) c, (long long) man.context, r,
                           (long long) ((const int64_t *) kidx->data)[r]);
            GGML_ABORT("axera qwen35_moe_sgl: MTP cell out of range\n");
        }
        return c;
    };
    static bool mtp_logged[9] = {};
    static int  mtp_log_prompt = -1;
    if (T >= 1 && T <= 8) {
        if (!mtp_logged[T]) {
            mtp_logged[T] = true;
            GGML_LOG_INFO("axera qwen35_moe_sgl: MTP T=%d cells %lld..%lld pages %lld\n", T,
                          (long long) cell_of(0), (long long) cell_of(T - 1),
                          (long long) (cell_of(T - 1) / PG + 1));
        }
    } else if (T != mtp_log_prompt) {
        mtp_log_prompt = T;
        GGML_LOG_INFO("axera qwen35_moe_sgl: MTP T=%d cells %lld..%lld pages %lld\n", T,
                      (long long) cell_of(0), (long long) cell_of(T - 1),
                      (long long) (cell_of(T - 1) / PG + 1));
    }
    auto row_bf16 = [&](const ggml_tensor * x, int r, ggml_bf16_t * d) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x->data + (size_t) r * x->nb[1]), d, H);
    };
    std::vector<float> h((size_t) T * H);
    std::vector<float> hr((size_t) T * H);  // the pre-shared-head-norm rows, for the combined head
    axera_timer tm(T == 1 ? "mtp (draft)" : T <= 8 ? "mtp (catch-up rows)" : "mtp (prompt)");
    const int  R8    = mtp8.c.k ? (int) (mtp8.e->size / (H * sizeof(ggml_bf16_t))) : 0;
    const int  paxes = 3;
    GGML_ASSERT(mtp1.pids && mtp1.pids->size >= (size_t) paxes * sizeof(int32_t));
    if (mtp8.c.k) {
        GGML_ASSERT(mtp8.pids && mtp8.pids->size >= (size_t) paxes * R8 * sizeof(int32_t));
    }

    const bool mr_pos = pos->ne[0] == 4 * (int64_t) T;
    auto fill_pids = [&](axera_cmm & c, int axes, int T_, int base, int pad_from) {
        auto * d = (int32_t *) c->vir;
        for (int s = 0; s < axes; ++s) {
            for (int r = 0; r < T_; ++r) {
                const int rr = base + std::min(r, pad_from);
                d[s * T_ + r] = mr_pos ? ((const int32_t *) pos->data)[(size_t) std::min(s, 3) * T + rr]
                                       : ((const int32_t *) pos->data)[rr];
            }
        }
        c->flush();
    };
    auto fill_pos = [&](axera_cmm & c, int T_, int row0, int pad_from, const kv_view &, auto & cell) {
        auto * d = (int32_t *) c->vir;
        for (int r = 0; r < T_; ++r) {
            d[r] = (int32_t) cell(row0 + std::min(r, pad_from));
        }
        c->flush();
    };
    auto fill_loc = [&](axera_cmm & c, int T_, int row0, int pad_from, const kv_view &, auto & cell) {
        auto * d = (int32_t *) c->vir;
        for (int r = 0; r < T_; ++r) {
            const int64_t c_ = cell(row0 + std::min(r, pad_from));
            d[2 * r]          = (int32_t) (c_ / PG);
            d[2 * r + 1]      = (int32_t) (c_ % PG);
        }
        c->flush();
    };
    auto set_ntok = [&](axera_cmm & c, int n) {
        *(int32_t *) c->vir = n;
        c->flush();
    };
    auto set_indptr = [&](axera_cmm & c, int64_t last_cell) {
        ((int32_t *) c->vir)[0] = 0;
        ((int32_t *) c->vir)[1] = (int32_t) (last_cell / PG + 1);
        c->flush();
    };
    auto rows_in = [&](axera_cmm & ce, axera_cmm & ch, int T_, int base, int pad_from) {
        for (int r = 0; r < T_; ++r) {
            const int rr = base + std::min(r, pad_from);
            row_bf16(embd, rr, (ggml_bf16_t *) ce->vir + (size_t) r * H);
            row_bf16(hin, rr, (ggml_bf16_t *) ch->vir + (size_t) r * H);
        }
        ce->flush();
        ch->flush();
    };

    if (T >= 2 && T <= R8 && !std::getenv("GGML_AXERA_MTP_T1")) {  // the rows accepted by a verification, in one call
        rows_in(mtp8.e, mtp8.h, R8, 0, T - 1);
        fill_pids(mtp8.pids, paxes, R8, 0, T - 1);
        fill_pos(mtp8.pos, R8, 0, T - 1, kv, cell_of);
        fill_loc(mtp8.loc, R8, 0, T - 1, kv, cell_of);
        set_ntok(mtp8.ntok, T);
        set_indptr(mtp8.indptr, cell_of(T - 1));
        mtp8.c.k->run(mtp8.c.in, mtp8.c.out, mtp8.c.k->desc->role.c_str());
        mtp8.hnext->invalidate(mtp8.hnext->vir, (size_t) T * H * sizeof(ggml_bf16_t));
        mtp8.hraw->invalidate(mtp8.hraw->vir, (size_t) T * H * sizeof(ggml_bf16_t));
        for (int r = 0; r < T; ++r) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtp8.hnext->vir + (size_t) r * H, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtp8.hraw->vir + (size_t) r * H, hr.data() + (size_t) r * H, H);
        }
    } else if (T <= 8 || std::getenv("GGML_AXERA_MTP_T1")) {  // a draft step: row by row
        for (int r = 0; r < T; ++r) {
            rows_in(mtp1.e, mtp1.h, 1, r, 0);
            fill_pids(mtp1.pids, paxes, 1, r, 0);
            fill_pos(mtp1.pos, 1, r, 0, kv, cell_of);
            fill_loc(mtp1.loc, 1, r, 0, kv, cell_of);
            set_ntok(mtp1.ntok, 1);
            set_indptr(mtp1.indptr, cell_of(r));
            mtp1.c.k->run(mtp1.c.in, mtp1.c.out, "mtp t1");
            mtp1.hnext->invalidate(mtp1.hnext->vir, H * sizeof(ggml_bf16_t));
            mtp1.hraw->invalidate(mtp1.hraw->vir, H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtp1.hnext->vir, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtp1.hraw->vir, hr.data() + (size_t) r * H, H);
        }
    } else {
        const int P = (int) (mtpP.e->size / (H * sizeof(ggml_bf16_t)));
        GGML_ASSERT(P > 0 && P <= 256);
        for (int it = 0; it < T; it += P) {
            const int n = std::min(P, T - it);
            rows_in(mtpP.e, mtpP.h, P, it, n - 1);  // padded rows repeat the last valid row
            fill_pids(mtpP.pids, paxes, P, it, n - 1);
            fill_pos(mtpP.pos, P, it, n - 1, kv, cell_of);
            fill_loc(mtpP.loc, P, it, n - 1, kv, cell_of);
            set_ntok(mtpP.ntok, n);
            set_indptr(mtpP.indptr, cell_of(it + n - 1));
            mtpP.c.k->run(mtpP.c.in, mtpP.c.out, "mtp prompt");
            mtpP.hnext->invalidate(mtpP.hnext->vir, (size_t) n * H * sizeof(ggml_bf16_t));
            mtpP.hraw->invalidate(mtpP.hraw->vir, (size_t) n * H * sizeof(ggml_bf16_t));
            for (int r = 0; r < n; ++r) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtpP.hnext->vir + (size_t) r * H,
                                      h.data() + (size_t) (it + r) * H, H);
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) mtpP.hraw->vir + (size_t) r * H,
                                      hr.data() + (size_t) (it + r) * H, H);
            }
        }
    }

    if (hn->flags & GGML_TENSOR_FLAG_COMPUTE) {
        for (int r = 0; r < T; ++r) {
            memcpy((char *) hn->data + (size_t) r * hn->nb[1], h.data() + (size_t) r * H, H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(hn->buffer)) {
            ((ggml_backend_axera_buffer_context *) hn->buffer->context)->flush(hn->data, ggml_nbytes(hn));
        }
    }
    ggml_tensor * logits = names.find("result_output");
    if (logits && logits->ne[1] > 0 && (logits->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        head.bind(man.get("lm_head"), logits->src[0]);
        if (T == 1 && man.has("lm_head.draft")) {
            if (!head.dk) {
                const std::string    f  = man.dir + "/draft_vocab.i32";
                FILE *               fp = fopen(f.c_str(), "rb");
                std::vector<int32_t> ids(man.get("lm_head.draft").out_bytes("out") / sizeof(float));
                GGML_ASSERT(fp && fread(ids.data(), sizeof(int32_t), ids.size(), fp) == ids.size());
                fclose(fp);
                axera_timer tb("init: draft head");
                head.bind_draft(man.get("lm_head.draft"), ids);
            }
            head.draft = true;
            const std::vector<char> normed(T, 1);  // (the draft kernel projects the MTP shared-head norm)
            head.run(g, names, h, T, &normed);
        } else {
            if (!head.bind_combined(man.get("lm_head"), logits->src[0])) {
                GGML_LOG_ERROR("axera qwen35_moe_sgl: the MTP head is not a combined lm_head\n");
                return GGML_STATUS_FAILED;
            }
            if (man.has("lm_head.verify")) {
                head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
            }
            head.run_combined(g, names, hr, T, nullptr, nullptr);
        }
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_moe_sgl(const manifest & m) {
    return std::make_shared<qwen35_moe_sgl>(m);
}

}  // namespace axera
