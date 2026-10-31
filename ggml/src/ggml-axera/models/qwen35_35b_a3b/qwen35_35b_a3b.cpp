// Axera NPU backend: Qwen3.5-35B-A3B (40 layers: 10 blocks of 3 gated delta net layers and one gated attention layer,
// each with a mixture of 256 experts, top 8, and a shared expert)
//
// The manifest's kernels (ax-dsl-kernels models/qwen35_35b_a3b, on sgl-kernel-axera):
//   decode   role decode: the whole model for one token in one call (a runtime loop over the 10 blocks, the layers'
//            weights, recurrent states and KV cache in banks [10, ...] per phase)
//   prefill  per page of 256 cells, layer by layer: role prefill.gdn or prefill.attn (one layer, 256 rows), then
//            role prefill.moe (the layer's experts, 256 rows); their weights are the decode banks' slices of the
//            layer. A batch that starts inside a page restarts at the page's first cell from the recurrent state
//            saved there, with the page's earlier inputs (kept) before its own
// The state is this adapter's (the decode kernel's banks): KV cache paged [10][64 pages][K, V][256][2 x 256] bf16,
// delta-net states [30][32][128][128] f32, conv states q / k [30][3][2048], v [30][3][4096] bf16 (value heads in HF
// order). The prefill kernels write the recurrent states to a second set, the two swapped after each page. One
// sequence. Weights only from the GGUF; the routed experts once, in the decode banks.

#include "../model.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"

#include <cmath>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;          // cells per page = prefill rows per call
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

// rows of an [N, K] weight of the GGUF in the kernels' MXFP4 layout (codes [K / 2], scales [K / 32]): an MXFP4 tensor's
// converted row by row as read, any other type quantized once
struct mx4 {
    const ggml_tensor * w;
    axera_cmm           q, s;
    int64_t             N = 0, K = 0;
    explicit mx4(const ggml_tensor * w) : w(w), N(ggml_nrows(w)), K(w->ne[0]) {
        if (w->type != GGML_TYPE_MXFP4) {
            axera_quantize_mxfp4(w, q, s);
        }
    }
    // row r: its codes and its scales (either may be null)
    void row(int64_t r, uint8_t * codes, uint8_t * scales) const {
        if (q) {
            if (codes) {
                memcpy(codes, (const uint8_t *) q->vir + r * (K / 2), K / 2);
            }
            if (scales) {
                memcpy(scales, (const uint8_t *) s->vir + r * (K / 32), K / 32);
            }
            return;
        }
        thread_local std::vector<uint8_t> c, sc;
        c.resize(K / 2);
        sc.resize(K / 32);
        axera_mxfp4_rows((const uint8_t *) w->data + r * w->nb[1], 1, K, codes ? codes : c.data(), scales ? scales : sc.data());
    }
};

std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) { memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float)); });
    }
    return v;
}

class qwen35_35b_a3b final : public model {
  public:
    explicit qwen35_35b_a3b(const manifest & m) : man(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, AX = 1;  // AX: prefill position axes
    std::vector<int> ord;                   // layer -> its index among the delta-net / attention layers

    std::map<std::string, axera_cmm> banks;  // the decode kernel's inputs by name (weights, states, KV cache)
    axera_cmm                        ST2, CQ2, CK2, CV2;  // the second recurrent-state set (prefill outputs)
    axera_cmm                        ONES, ZERO;           // global scales (1) and zero points (0) of the prefill kernels
    call                             dec;
    std::vector<call>                pre_attn, pre_moe;  // per layer
    axera_cmm DX, DA, DOUT, DPOS, INDPTR, INDICES, LASTLEN, CLOC, PX, PA, POS, VALID;
    // the recurrent state at the first cell of the current page (snap_cell), and the inputs of that page's cells so far
    std::vector<uint8_t>     snap;
    int64_t                  snap_cell = -1;
    std::vector<ggml_bf16_t> page_x;
    std::vector<int32_t>     page_p;  // [cells][4]
    output_head              head;

    void         init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void         fill(const std::map<std::string, ggml_tensor *> & t, const kernel & dk, const rope & rp);
    axera_cmm &  B(const std::string & n) { return banks.at(n); }
    void         states_swap();
    void         zero_state();
    void         save_state(int64_t cell);
    void         load_state();
    void         paging(int64_t hist, int64_t write_cell);
    void         prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    void         decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h);
};

// the decode kernel's banks from the GGUF: b<phase>_attn_* (delta net, phases 0-2; gated attention, phase 3),
// b<phase>_moe_*, each [10 blocks, ...] (layer = 4 * block + phase), and the state / KV pools
void qwen35_35b_a3b::fill(const std::map<std::string, ggml_tensor *> & t, const kernel & dk, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|moe)_(.+)");
    auto T = [&](int il, const std::string & g) -> const ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen35_35b_a3b: %s not in the graph\n", nm.c_str());
        }
        return t.at(nm);
    };
    for (const auto & d : dk.desc->in) {
        std::smatch mt;
        if (banks.count(d.name)) {
            continue;
        }
        if (!std::regex_match(d.name, mt, re)) {
            if (d.name == "b3_kv") {  // the KV cache pool: zero
                banks[d.name] = axera_cmm_alloc(d.bytes);
            }
            continue;
        }
        const int         phase = std::stoi(mt[1]);
        const bool        moe   = mt[2] == "moe";
        const std::string n     = mt[3];
        axera_cmm &       c     = banks[d.name] = axera_cmm_alloc(d.bytes);
        if (n == "state_in" || n == "q_conv_state_in" || n == "k_conv_state_in" || n == "v_conv_state_in" || n == "kv") {
            continue;  // the state pools: zero
        }
        if (n == "cos_sin_cache") {  // [positions][cos | sin of ROT / 2 frequencies] f32
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
        const size_t bb = d.bytes / NB;  // bytes per block
        for (int b = 0; b < NB; ++b) {
            const int il  = 4 * b + phase;
            uint8_t * dst = (uint8_t *) c->vir + (size_t) b * bb;
            auto      put = [&](const std::vector<float> & v, float scale = 1.0f) {
                GGML_ASSERT(v.size() * (d.dtype == 3 ? 4 : 2) == bb);
                for (size_t i = 0; i < v.size(); ++i) {
                    if (d.dtype == 3) {
                        ((float *) dst)[i] = v[i] * scale;
                    } else {
                        ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i] * scale);
                    }
                }
            };
            auto fill_v = [&](float v) { put(std::vector<float>(bb / (d.dtype == 3 ? 4 : 2), v)); };
            // MXFP4 rows of a weight into this input (codes or scales): rows r0 .. r0 + nr, value heads (of hr rows)
            // from row vrow on in HF order
            auto rows = [&](const mx4 & w, int64_t r0, int64_t nr, bool scales, int64_t vrow = -1, int64_t hr = 0) {
                const size_t rb = scales ? w.K / 32 : w.K / 2;
                GGML_ASSERT((size_t) nr * rb == bb);
                axera_host_pool().run(nr, [&](int64_t r) {
                    int64_t src = r0 + r;
                    if (vrow >= 0 && src >= vrow) {
                        const int64_t j = src - vrow;
                        src             = vrow + tiled((int) (j / hr)) * hr + j % hr;
                    }
                    w.row(src, scales ? nullptr : dst + r * rb, scales ? dst + r * rb : nullptr);
                });
            };
            auto cols = [&](const mx4 & w, bool scales) {  // value heads of the input (columns) in HF order
                const size_t rb = scales ? w.K / 32 : w.K / 2, cb = rb / VH;
                GGML_ASSERT((size_t) w.N * rb == bb);
                axera_host_pool().run(w.N, [&](int64_t r) {
                    thread_local std::vector<uint8_t> src;
                    src.resize(rb);
                    w.row(r, scales ? nullptr : src.data(), scales ? src.data() : nullptr);
                    for (int g = 0; g < VH; ++g) {
                        memcpy(dst + r * rb + g * cb, src.data() + tiled(g) * cb, cb);
                    }
                });
            };
            auto vec = [&](const std::string & g, bool heads, float scale = 1.0f) {
                std::vector<float> v = f32_of(T(il, g));
                if (heads) {
                    std::vector<float> u(v.size());
                    for (int h = 0; h < VH; ++h) {
                        u[h] = v[tiled(h)];
                    }
                    v = u;
                }
                put(v, scale);
            };
            if (n.find("scale2") != std::string::npos) {
                fill_v(1.0f);
            } else if (n.rfind("w_zeropoint_", 0) == 0) {
                memset(dst, 0, bb);
            } else if (moe) {
                if (n == "input_norm_w") {
                    vec("post_attention_norm.weight", false);
                } else if (n == "shared_expert_gate") {
                    vec("ffn_gate_inp_shexp.weight", false);
                } else if (n == "w_route") {
                    vec("ffn_gate_inp.weight", false);
                } else if (n == "w_scale_route") {
                    fill_v(1.0f);
                } else if (n == "w_gate" || n == "w_scale_gate") {  // per expert: gate rows, then up rows
                    const mx4     g(T(il, "ffn_gate_exps.weight")), u(T(il, "ffn_up_exps.weight"));
                    const bool    sc = n != "w_gate";
                    const int64_t E = g.w->ne[2], F = g.w->ne[1], rb = sc ? g.K / 32 : g.K / 2;
                    GGML_ASSERT((size_t) E * 2 * F * rb == bb);
                    axera_host_pool().run(E * 2 * F, [&](int64_t r) {
                        const int64_t e = r / (2 * F), j = r % (2 * F);
                        const mx4 &   w = j < F ? g : u;
                        w.row(e * F + j % F, sc ? nullptr : dst + r * rb, sc ? dst + r * rb : nullptr);
                    });
                } else if (n == "w_down" || n == "w_scale_down") {
                    const mx4 w(T(il, "ffn_down_exps.weight"));
                    rows(w, 0, w.N, n != "w_down");
                } else {
                    static const std::map<std::string, const char *> S = { { "gate", "ffn_gate_shexp.weight" },
                                                                            { "up", "ffn_up_shexp.weight" },
                                                                            { "down", "ffn_down_shexp.weight" } };
                    const size_t p = n.rfind('_');
                    if (n.find("shared_expert_mlp_") == std::string::npos || !S.count(n.substr(p + 1))) {
                        GGML_ABORT("axera qwen35_35b_a3b: unexpected MoE input %s\n", d.name.c_str());
                    }
                    const mx4 w(T(il, S.at(n.substr(p + 1))));
                    rows(w, 0, w.N, n.rfind("w_scale_", 0) == 0);
                }
            } else if (phase == 3) {  // gated attention
                static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                        { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
                const std::string p = n.substr(n.rfind('_') + 1);
                if (n == "w_norm_attn") {
                    vec("attn_norm.weight", false);
                } else if (n == "w_norm_q") {  // the attention scale folded in
                    vec("attn_q_norm.weight", false, 1.0f / std::sqrt((float) d.shape.back()));
                } else if (n == "w_norm_k") {
                    vec("attn_k_norm.weight", false);
                } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                    const mx4 w(T(il, W.at(p)));
                    rows(w, 0, w.N, n != "w_" + p);
                } else {
                    GGML_ABORT("axera qwen35_35b_a3b: unexpected attention input %s\n", d.name.c_str());
                }
            } else if (n == "input_norm_w") {  // gated delta net
                vec("attn_norm.weight", false);
            } else if (n == "o_norm_w") {
                vec("ssm_norm.weight", false);
            } else if (n == "a_log") {
                std::vector<float> a = f32_of(T(il, "ssm_a")), u(VH);
                for (int h = 0; h < VH; ++h) {
                    u[h] = logf(-a[tiled(h)]);  // GGUF ssm_a = -exp(A_log)
                }
                put(u);
            } else if (n == "dt_bias") {
                vec("ssm_dt.bias", true);
            } else if (n == "w_q_conv" || n == "w_k_conv" || n == "w_v_conv") {
                // [C][1][taps] in the checkpoint's tap order: channels q, k (C each) then v (the rest, value heads)
                const ggml_tensor *      w  = T(il, "ssm_conv1d.weight");
                const std::vector<float> v  = f32_of(w);
                const int64_t            tp = w->ne[0], C = d.shape[1], v0 = w->ne[1] - C;
                const int64_t            c0 = n[2] == 'q' ? 0 : n[2] == 'k' ? C : v0;
                std::vector<float>       u((size_t) C * tp);
                for (int64_t r = 0; r < C; ++r) {
                    const int64_t hr  = C / VH;
                    const int64_t src = n[2] == 'v' ? v0 + tiled((int) (r / hr)) * hr + r % hr : c0 + r;
                    memcpy(&u[(size_t) r * tp], &v[(size_t) src * tp], tp * sizeof(float));
                }
                put(u);
            } else if (n.rfind("w_", 0) == 0 && n.size() >= 3) {  // w_<p>, w_<p>_scale for p in q k v z a b o
                const char        p      = n[2];
                const std::string suffix = n.substr(3);
                const bool        scales = suffix == "_scale";
                GGML_ASSERT(suffix.empty() || scales);
                if (p == 'q' || p == 'k' || p == 'v') {
                    const mx4     w(T(il, "attn_qkv.weight"));
                    const int64_t nr = d.shape[1];  // rows q, k (nr each), then v (the rest)
                    if (p == 'v') {
                        rows(w, w.N - nr, nr, scales, w.N - nr, nr / VH);
                    } else {
                        rows(w, p == 'q' ? 0 : nr, nr, scales);
                    }
                } else if (p == 'z') {
                    const mx4 w(T(il, "attn_gate.weight"));
                    rows(w, 0, w.N, scales, 0, w.N / VH);
                } else if (p == 'a' || p == 'b') {
                    const mx4 w(T(il, p == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight"));
                    rows(w, 0, w.N, scales, 0, 1);
                } else if (p == 'o') {
                    const mx4 w(T(il, "ssm_out.weight"));
                    cols(w, scales);
                } else {
                    GGML_ABORT("axera qwen35_35b_a3b: unexpected delta-net input %s\n", d.name.c_str());
                }
            } else {
                GGML_ABORT("axera qwen35_35b_a3b: unexpected delta-net input %s\n", d.name.c_str());
            }
        }
        c->flush();
    }
}

void qwen35_35b_a3b::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
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
        fill(t, dk, rope(rope_node, names.find("inp_pos"), 1));
    }
    ST2  = axera_cmm_alloc(B("b0_attn_state_in")->size);
    CQ2  = axera_cmm_alloc(B("b0_attn_q_conv_state_in")->size);
    CK2  = axera_cmm_alloc(B("b0_attn_k_conv_state_in")->size);
    CV2  = axera_cmm_alloc(B("b0_attn_v_conv_state_in")->size);
    snap.assign(ST2->size + CQ2->size + CK2->size + CV2->size, 0);
    snap_cell = -1;
    page_x.clear();
    page_p.clear();
    ONES = axera_cmm_alloc(4096);
    for (int i = 0; i < 1024; ++i) {
        ((float *) ONES->vir)[i] = 1.0f;
    }
    ONES->flush();
    ZERO = axera_cmm_alloc(4096);
    const int NP = (int) B("b3_kv")->size / (NB * 2 * PG * 2 * 256 * 2);  // pages
    DX      = axera_cmm_alloc(dk.in_bytes("x"));
    DA      = axera_cmm_alloc(dk.in_bytes("attention"));
    DOUT    = axera_cmm_alloc(dk.out_bytes("output"));
    DPOS    = axera_cmm_alloc(dk.in_bytes("position_ids"));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc((size_t) NP * sizeof(int32_t));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    PX      = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    PA      = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
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
    auto slice = [&](const std::string & bank, int idx, uint32_t bytes) {
        const axera_cmm & c = B(bank);
        const size_t      n = bank.find("state_in") != std::string::npos ? 3 * NB : NB;
        GGML_ASSERT(c->size % n == 0 && c->size / n == bytes);
        return axera_io(c.get(), (char *) c->vir + (size_t) idx * bytes, bytes);
    };
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
                c.in.push_back(axera_io(ONES, d.bytes));
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
                m.in.push_back(axera_io(ONES, d.bytes));
            } else {  // w_up / w_scale_up / w_scale2_up: the fused gate | up tensor (rows 0:512 gate, 512:1024 up)
                std::string b = n == "w_up" ? "w_gate" : n == "w_scale_up" ? "w_scale_gate" : n == "w_scale2_up" ? "w_scale2_gate" : n;
                m.in.push_back(slice(pm + b, blk, d.bytes));
            }
        }
        m.out.push_back(axera_io(PX, mk.desc->out[0].bytes));
    }
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen35_35b_a3b: %d layers, banks (weights + state) %.2f GiB, %lld cells\n", L, bb / 1073741824.0,
                  (long long) man.context);
}

// the prefill kernels' new recurrent states become the current ones (pointer swap of the pools, rebinding the IO)
void qwen35_35b_a3b::states_swap() {
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
    }
}

void qwen35_35b_a3b::zero_state() {
    axera_timer tl("state zero");
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        memset(B(n)->vir, 0, B(n)->size);
        B(n)->flush();
    }
    snap_cell = -1;
    page_x.clear();
    page_p.clear();
}

void qwen35_35b_a3b::save_state(int64_t cell) {
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

void qwen35_35b_a3b::load_state() {
    axera_timer tl("state load");
    size_t o = 0;
    for (const char * n : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" }) {
        memcpy(B(n)->vir, snap.data() + o, B(n)->size);
        B(n)->flush();
        o += B(n)->size;
    }
}

// the history before the token(s) written from cell write_cell: hist cells in whole pages and a last one
void qwen35_35b_a3b::paging(int64_t hist, int64_t write_cell) {
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
void qwen35_35b_a3b::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (v < PG) {  // a partial last page: the state before it is the restart point of a later batch
            save_state(cell);
        }
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) PX->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        PX->flush();
        auto * pp = (int32_t *) POS->vir;
        for (int s = 0; s < AX; ++s) {
            for (int r = 0; r < PG; ++r) {
                pp[s * PG + r] = r < v ? p4[(size_t) (r0 + r) * 4 + s] : 0;
            }
        }
        POS->flush();
        paging(cell, cell);
        *(int32_t *) VALID->vir = v;
        VALID->flush();
        {
            axera_timer tl("prefill (page)");
            for (int il = 0; il < L; ++il) {
                pre_attn[il].k->run(pre_attn[il].in, pre_attn[il].out, il % 4 == 3 ? "prefill attn" : "prefill gdn");
                pre_moe[il].k->run(pre_moe[il].in, pre_moe[il].out, "prefill moe");
            }
        }
        states_swap();
        PX->invalidate(PX->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) PX->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
        if (v < PG) {  // the page's inputs so far: a later batch restarts the page with them
            page_x.assign(x + (size_t) r0 * H, x + (size_t) (r0 + v) * H);
            page_p.assign(p4 + (size_t) r0 * 4, p4 + (size_t) (r0 + v) * 4);
        } else if (r0 + PG == n) {  // a full last page: the next page starts clean
            save_state(cell + PG);
        }
    }
}

// one token at cell c > 0, its history the c cells before it
void qwen35_35b_a3b::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, float * h) {
    if (c % PG == 0) {  // a page's first cell: the state here is the restart point
        save_state(c);
    }
    memcpy(DX->vir, x, H * sizeof(ggml_bf16_t));
    DX->flush();
    ((int32_t *) DPOS->vir)[0] = p4[0];
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
}

ggml_status qwen35_35b_a3b::compute(ggml_cgraph * g) {
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
        // from the batch's page start: the page's earlier inputs (and the state saved there) before the batch's
        const int64_t s0 = c0 - c0 % PG;
        const int     k  = (int) (c0 - s0);
        if (k > 0 && (snap_cell != s0 || (int) (page_x.size() / H) < k)) {
            GGML_LOG_ERROR("axera qwen35_35b_a3b: a prompt from cell %lld, inside a page whose state is not kept\n", (long long) c0);
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
    } else {
        decode(c0, xb.data(), pb.data(), h.data());
    }
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    head.run(g, names, h, T);
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_35b_a3b(const manifest & m) {
    return std::make_shared<qwen35_35b_a3b>(m);
}

}  // namespace axera
