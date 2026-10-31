// Axera NPU backend: Qwen3.8-27B (64 layers: 16 blocks of 3 gated delta net layers and one gated attention layer,
// dense MLP)
//
// One path, the manifest's kernels (ax-dsl-kernels models/qwen38_27b):
//   prefill  role prefill: the whole model in one call per page of 256 cells (a runtime loop over the 16 blocks, the
//            layers' weights in banks [16, ...] indexed by the block), the last page of a batch padded (valid_tokens).
//            Its history is whole pages: a batch that starts inside a page restarts at the page's first cell from the
//            recurrent state saved there, with the page's earlier inputs (kept) before its own
//   decode   role decode.group: 16 consecutive layers per call (weights in packs), 4 calls per token, then lm_head
// The state is this adapter's (CMM pools in the prefill kernel's layouts, the decode group bound to each layer's part):
// the KV cache paged [16][64 pages][K, V][256][4 x 256] bf16, the delta-net states [48][48][128][128] f32 and conv
// states q / k [48][3][2048], v [48][3][6144] bf16 (value heads in HF order). One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <cmath>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;           // cells per page = prefill rows per call = the decode kernels' KV block
constexpr int KH = 16, VH = 48;   // key / value heads of a delta-net layer
constexpr int REP = VH / KH;

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

// GGUF's tiled value head of HF (grouped) head g
inline int tiled(int g) {
    return (g % REP) * KH + g / REP;
}

class qwen38_27b final : public model {
  public:
    explicit qwen38_27b(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, G = 0, ROT = 0, NBLK = 0;
    std::vector<int> ord;  // layer -> its index among the delta-net layers / the attention layers

    // the state pools; their sizes per layer
    axera_cmm KV, ST, CQ, CK, CV;
    size_t    kv_b = 0, st_b = 0, cq_b = 0, cv_b = 0;
    // the recurrent state at the first cell of the current page (snap_cell), and the inputs of that page's cells so far
    std::vector<uint8_t>     snap;
    int64_t                  snap_cell = -1;
    std::vector<ggml_bf16_t> page_x;  // [cells][H] from snap_cell
    std::vector<int32_t>     page_p;  // their M-RoPE positions [cells][4]

    std::vector<call>              dec;
    axera_cmm                      A, COS, SIN, CS, PN, MASK;
    call                           pre;
    std::map<std::string, axera_cmm> banks;
    axera_cmm                      PX, POS, INDPTR, INDICES, LASTLEN, CLOC, VALID, POUT;
    output_head                    head;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    void      zero_state();
    void      save_state(int64_t cell);
    void      load_state();
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
};

// rows of an [N, K] weight of the GGUF in the kernels' MXFP4 layout (codes [K / 2], scales [K / 32]): an MXFP4 tensor's
// converted row by row as read, any other type (a community GGUF: Q4_K, Q8_0, ...) quantized once
struct mx4 {
    const ggml_tensor * w;
    axera_cmm           q, s;
    int64_t             N = 0, K = 0;
    explicit mx4(const ggml_tensor * w) : w(w), N(w->ne[1]), K(w->ne[0]) {
        if (w->type != GGML_TYPE_MXFP4) {
            axera_quantize_mxfp4(w, q, s);
        }
    }
    // row r: its codes or its scales into dst
    void row(int64_t r, bool scales, uint8_t * dst) const {
        if (q) {
            memcpy(dst, scales ? (const uint8_t *) s->vir + r * (K / 32) : (const uint8_t *) q->vir + r * (K / 2),
                   scales ? K / 32 : K / 2);
            return;
        }
        thread_local std::vector<uint8_t> other;
        other.resize(scales ? K / 2 : K / 32);
        const uint8_t * src = (const uint8_t *) w->data + r * w->nb[1];
        axera_mxfp4_rows(src, 1, K, scales ? other.data() : dst, scales ? dst : other.data());
    }
};

// a [rows] or [rows][cols] f32 view of any small GGUF tensor
std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) { memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float)); });
    }
    return v;
}

axera_cmm qwen38_27b::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera qwen38_27b: prefill: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const int         nb    = L / 4;
    GGML_ASSERT(d.shape.size() >= 2 && d.shape[0] == nb && d.bytes % nb == 0);
    axera_cmm     c  = axera_cmm_alloc(d.bytes);
    const size_t  bb = d.bytes / nb;  // bytes per block
    auto          T  = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen38_27b: %s not in the graph\n", nm.c_str());
        }
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        const int il  = 4 * b + phase;
        uint8_t * dst = (uint8_t *) c->vir + (size_t) b * bb;
        auto put_f32 = [&](const std::vector<float> & v, float scale = 1.0f) {
            GGML_ASSERT(v.size() * sizeof(float) == bb || v.size() * sizeof(ggml_bf16_t) == bb);
            for (size_t i = 0; i < v.size(); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v[i] * scale;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i] * scale);
                }
            }
        };
        auto fill = [&](float v) {  // a block's global scales / zero points: every element
            for (size_t i = 0; i < bb / (d.dtype == 3 ? 4 : 2); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v);
                }
            }
        };
        // MXFP4 rows r0 .. of a weight (value heads of row size hr from vrow on in HF order); scales or codes
        auto rows = [&](const mx4 & w, int64_t r0, int64_t nr, bool scales, int64_t vrow = -1, int64_t hr = 0) {
            const size_t rb = scales ? w.K / 32 : w.K / 2;
            GGML_ASSERT((size_t) nr * rb == bb);
            axera_host_pool().run(nr, [&](int64_t r) {
                int64_t src = r0 + r;
                if (vrow >= 0 && src >= vrow) {
                    const int64_t j = src - vrow;
                    src             = vrow + tiled((int) (j / hr)) * hr + j % hr;
                }
                w.row(src, scales, dst + r * rb);
            });
        };
        auto cols = [&](const mx4 & w, bool scales) {  // value heads of the input (columns) in HF order
            const size_t rb = scales ? w.K / 32 : w.K / 2, cb = rb / VH;
            GGML_ASSERT((size_t) w.N * rb == bb);
            axera_host_pool().run(w.N, [&](int64_t r) {
                thread_local std::vector<uint8_t> src;
                src.resize(rb);
                w.row(r, scales, src.data());
                for (int g = 0; g < VH; ++g) {
                    memcpy(dst + r * rb + g * cb, src.data() + tiled(g) * cb, cb);
                }
            });
        };
        auto vec = [&](const char * g, bool heads, float scale = 1.0f) {  // a small tensor; per value head in HF order
            std::vector<float> v = f32_of(T(il, g));
            if (heads) {
                std::vector<float> u(v.size());
                for (int h = 0; h < VH; ++h) {
                    u[h] = v[tiled(h)];
                }
                v = u;
            }
            put_f32(v, scale);
        };
        if (mlp) {
            static const std::map<std::string, const char *> W = { { "1", "ffn_gate.weight" }, { "2", "ffn_up.weight" },
                                                                    { "3", "ffn_down.weight" } };
            if (n == "w_norm") {
                vec("post_attention_norm.weight", false);
            } else if (n == "input_permutation") {
                memset(dst, 0, bb);
            } else if (n.rfind("w_scale2_", 0) == 0) {
                fill(1.0f);
            } else if (n.rfind("w_zp_scale", 0) == 0) {
                fill(0.0f);
            } else if (n.size() == 2 && W.count(n.substr(1))) {
                const mx4 w(T(il, W.at(n.substr(1))));
                rows(w, 0, w.N, false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                const mx4 w(T(il, W.at(n.substr(7))));
                rows(w, 0, w.N, true);
            } else {
                GGML_ABORT("axera qwen38_27b: prefill: unexpected MLP input %s\n", d.name.c_str());
            }
            continue;
        }
        if (phase == 3) {  // gated attention
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                vec("attn_norm.weight", false);
            } else if (n == "w_norm_q") {  // the attention scale folded in
                vec("attn_q_norm.weight", false, 1.0f / std::sqrt((float) d.shape.back()));
            } else if (n == "w_norm_k") {
                vec("attn_k_norm.weight", false);
            } else if (n == "cos" || n == "sin") {  // [16384][ROT / 2] f32
                const int64_t P = d.shape[1], R = d.shape[2];
                GGML_ASSERT(R == ROT / 2 && d.dtype == 3);
                for (int64_t q = 0; q < P; ++q) {
                    for (int64_t i = 0; i < R; ++i) {
                        const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                        ((float *) dst)[q * R + i] = (float) (n == "cos" ? std::cos(th) : std::sin(th));
                    }
                }
            } else if (n.rfind("w_scale2_", 0) == 0) {
                fill(1.0f);
            } else if (n.rfind("w_zeropoint_", 0) == 0) {
                fill(0.0f);
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                const mx4 w(T(il, W.at(p)));
                rows(w, 0, w.N, n != "w_" + p);
            } else {
                GGML_ABORT("axera qwen38_27b: prefill: unexpected attention input %s\n", d.name.c_str());
            }
            continue;
        }
        // gated delta net: q / k / v rows of attn_qkv (q, k: key heads; v: value heads); z, a, b, out per value head
        if (n == "input_norm_w") {
            vec("attn_norm.weight", false);
        } else if (n == "o_norm_w") {
            vec("ssm_norm.weight", false);
        } else if (n == "a_log") {
            std::vector<float> a = f32_of(T(il, "ssm_a")), u(VH);
            for (int h = 0; h < VH; ++h) {
                u[h] = logf(-a[tiled(h)]);  // GGUF ssm_a = -exp(A_log)
            }
            put_f32(u);
        } else if (n == "dt_bias") {
            vec("ssm_dt.bias", true);
        } else if (n == "w_q_conv" || n == "w_k_conv" || n == "w_v_conv") {
            // [C][1][taps] bf16 in the checkpoint's tap order: channels q, k (C each) then v (the rest, value heads)
            const ggml_tensor * w  = T(il, "ssm_conv1d.weight");
            const std::vector<float> v = f32_of(w);
            const int64_t tp = w->ne[0], C = d.shape[1], v0 = w->ne[1] - C;
            const int64_t c0 = n[2] == 'q' ? 0 : n[2] == 'k' ? C : v0;
            std::vector<float> u((size_t) C * tp);
            for (int64_t r = 0; r < C; ++r) {
                const int64_t hr  = C / VH;
                const int64_t src = n[2] == 'v' ? v0 + tiled((int) (r / hr)) * hr + r % hr : c0 + r;
                memcpy(&u[(size_t) r * tp], &v[(size_t) src * tp], tp * sizeof(float));
            }
            put_f32(u);
        } else if (n.rfind("w_", 0) == 0 && n.size() >= 3) {
            // w_<p>, w_<p>_scale, w_<p>_scale2 for p in q k v z a b o
            const char        p      = n[2];
            const std::string suffix = n.substr(3);
            if (suffix == "_scale2") {
                fill(1.0f);
                continue;
            }
            const bool scales = suffix == "_scale";
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
            } else if (p == 'a' || p == 'b') {  // Q8_0 in the GGUF: requantized (the delta net takes all-FP4 projections)
                const mx4 w(T(il, p == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight"));
                rows(w, 0, w.N, scales, 0, 1);
            } else if (p == 'o') {
                const mx4 w(T(il, "ssm_out.weight"));
                cols(w, scales);
            } else {
                GGML_ABORT("axera qwen38_27b: prefill: unexpected delta-net input %s\n", d.name.c_str());
            }
        } else {
            GGML_ABORT("axera qwen38_27b: prefill: unexpected delta-net input %s\n", d.name.c_str());
        }
    }
    c->flush();
    return c;
}

void qwen38_27b::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen38_27b: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        ord[il] = g ? ng++ : na++;
    }
    const kernel & dk = man.get("decode.group");
    const kernel & pk = man.get("prefill");
    G                 = 0;
    for (const auto & d : dk.desc->out) {
        if (d.name.rfind("st", 0) == 0 || d.name.rfind("kv", 0) == 0) {
            G = std::max(G, atoi(d.name.c_str() + 2) + 1);
        }
    }
    GGML_ASSERT(G % 4 == 0 && L % G == 0);
    // the pools: sizes from the prefill kernel's (all layers), checked against the decode group's (one layer)
    auto in_of = [&](const kernel & k, const std::string & n) -> const io_desc & {
        for (const auto & d : k.desc->in) {
            if (d.name == n) {
                return d;
            }
        }
        for (const auto & d : k.desc->out) {
            if (d.name == n) {
                return d;
            }
        }
        GGML_ABORT("axera qwen38_27b: %s has no IO %s\n", k.desc->file.c_str(), n.c_str());
    };
    const io_desc & pkv = in_of(pk, "b3_kv"), &pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in"), &pcv = in_of(pk, "b0_attn_v_conv_state_in");
    GGML_ASSERT(pkv.shape[0] == na && pst.shape[0] == ng && pcq.shape[0] == ng && pcv.shape[0] == ng);
    kv_b = pkv.bytes / na;
    st_b = pst.bytes / ng;
    cq_b = pcq.bytes / ng;
    cv_b = pcv.bytes / ng;
    GGML_ASSERT(in_of(dk, "kv3").bytes == kv_b && in_of(dk, "st0").bytes == st_b && in_of(dk, "cq0").bytes == cq_b &&
                in_of(dk, "ck0").bytes == cq_b && in_of(dk, "cv0").bytes == cv_b);
    NBLK = (int) pkv.shape[1];
    GGML_ASSERT(pkv.shape[3] == PG && (int64_t) NBLK * PG == man.context);
    KV = axera_cmm_alloc(pkv.bytes);
    ST = axera_cmm_alloc(pst.bytes);
    CQ = axera_cmm_alloc(pcq.bytes);
    CK = axera_cmm_alloc(pcq.bytes);
    CV = axera_cmm_alloc(pcv.bytes);
    snap.assign(pst.bytes + 2 * pcq.bytes + pcv.bytes, 0);
    snap_cell = -1;
    page_x.clear();

    // decode: x / out A, cs = cos | sin, pn = cell | blocks | the K row of the cell (page * 2 * 256 + offset)
    const uint32_t csb = dk.in_bytes("cs");
    ROT  = (int) (csb / sizeof(ggml_bf16_t));
    A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    COS  = axera_cmm_alloc(csb / 2);
    SIN  = axera_cmm_alloc(csb / 2);
    CS   = axera_cmm_alloc(csb);
    PN   = axera_cmm_alloc(3 * sizeof(int32_t));
    MASK = axera_cmm_alloc(dk.in_bytes("mask"));
    GGML_ASSERT(dk.in_bytes("pn") == 3 * sizeof(int32_t) && dk.in_bytes("mask") == (size_t) man.context * sizeof(ggml_bf16_t));
    auto pool_io = [&](const std::string & n, int il) {
        if (n.rfind("kv", 0) == 0) {
            return axera_io(KV.get(), (char *) KV->vir + ord[il] * kv_b, (uint32_t) kv_b);
        }
        const axera_cmm & p = n[0] == 's' ? ST : n[1] == 'q' ? CQ : n[1] == 'k' ? CK : CV;
        const size_t      b = n[0] == 's' ? st_b : n[1] == 'v' ? cv_b : cq_b;
        return axera_io(p.get(), (char *) p->vir + ord[il] * b, (uint32_t) b);
    };
    dec.assign(L / G, {});
    for (int gi = 0; gi < L / G; ++gi) {
        call & c = dec[gi];
        c.k      = &dk;
        for (const auto & d : dk.desc->in) {
            if (dk.desc->packs.count(d.name)) {
                c.in.push_back(axera_io(prm.pack(t, *dk.desc, d.name, gi * G), d.bytes));
                continue;
            }
            const axera_cmm * b = d.name == "x" ? &A : d.name == "cs" ? &CS : d.name == "pn" ? &PN : d.name == "mask" ? &MASK : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen38_27b: decode.group: unexpected input %s\n", d.name.c_str());
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : dk.desc->out) {
            c.out.push_back(d.name == "out" ? axera_io(A, d.bytes) : pool_io(d.name, gi * G + atoi(d.name.c_str() + 2)));
        }
    }

    // prefill: the weight banks from the GGUF, the pools, one page's inputs
    const rope rp(rope_node, names.find("inp_pos"), 1);
    pre   = {};
    pre.k = &pk;
    PX      = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POUT    = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POS     = axera_cmm_alloc(pk.in_bytes("position_ids"));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    GGML_ASSERT(pk.in_bytes("x") == PX->size && pk.in_bytes("position_ids") == 3 * PG * sizeof(int32_t) &&
                pk.in_bytes("indices") == (size_t) NBLK * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();
    banks.clear();
    axera_timer tb("init: prefill banks");
    for (const auto & d : pk.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &PX : n == "position_ids" ? &POS : n == "indptr" ? &INDPTR : n == "indices" ? &INDICES :
                                n == "last_page_len" ? &LASTLEN : n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID :
                                n == "b3_kv" ? &KV : n == "b0_attn_state_in" ? &ST : n == "b0_attn_q_conv_state_in" ? &CQ :
                                n == "b0_attn_k_conv_state_in" ? &CK : n == "b0_attn_v_conv_state_in" ? &CV : nullptr;
        if (!b) {
            b = &(banks[n] = bank(t, d, rp));
        }
        GGML_ASSERT((*b)->size >= d.bytes);
        pre.in.push_back(axera_io(*b, d.bytes));
    }
    GGML_ASSERT(pk.desc->out.size() == 1 && pk.desc->out[0].name == "output");
    pre.out.push_back(axera_io(POUT, pk.desc->out[0].bytes));

    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen38_27b: %d layers (%d per decode call), prefill banks %.1f GiB, state %.2f GiB, %lld cells\n",
                  L, G, bb / 1073741824.0, (KV->size + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}

void qwen38_27b::zero_state() {
    axera_timer tl("state zero");
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memset((*p)->vir, 0, (*p)->size);
        (*p)->flush();
    }
    snap_cell = -1;
    page_x.clear();
    page_p.clear();
}

// the recurrent state (before cell `cell`, the first of its page) and the page's inputs from there
void qwen38_27b::save_state(int64_t cell) {
    GGML_ASSERT(cell % PG == 0);
    axera_timer tl("state save");
    size_t o = 0;
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        (*p)->invalidate();
        memcpy(snap.data() + o, (*p)->vir, (*p)->size);
        o += (*p)->size;
    }
    snap_cell = cell;
    page_x.clear();
    page_p.clear();
}

void qwen38_27b::load_state() {
    axera_timer tl("state load");
    size_t o = 0;
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memcpy((*p)->vir, snap.data() + o, (*p)->size);
        (*p)->flush();
        o += (*p)->size;
    }
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16 and their M-RoPE positions p4 [n][4]; the hidden
// states of the rows from out0 on into h
void qwen38_27b::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0);
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
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < PG; ++r) {
                pp[s * PG + r] = r < v ? p4[(size_t) (r0 + r) * 4 + s] : 0;
            }
        }
        POS->flush();
        ((int32_t *) INDPTR->vir)[0]  = 0;
        ((int32_t *) INDPTR->vir)[1]  = (int32_t) (cell / PG);
        *(int32_t *) LASTLEN->vir     = PG;
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (cell / PG);
        ((int32_t *) CLOC->vir)[1]    = 0;
        *(int32_t *) VALID->vir       = v;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        {
            axera_timer tl("prefill (page)");
            pre.k->run(pre.in, pre.out, "prefill");
        }
        axera_timer to("prefill out rows");
        POUT->invalidate(POUT->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
        if (v < PG) {  // the page's inputs so far: a later batch restarts the page with them
            page_x.assign(x + (size_t) r0 * H, x + (size_t) (r0 + v) * H);
            page_p.assign(p4 + (size_t) r0 * 4, p4 + (size_t) (r0 + v) * 4);
        } else if (r0 + PG == n) {  // a full last page: the next page starts clean
            save_state(cell + PG);
        }
    }
}

ggml_status qwen38_27b::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen38_27b: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
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
        axera_timer tc("input (cpu graph)");
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
    GGML_ASSERT(rope_node && "axera qwen38_27b: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    axera_timer   tkv("kv view");
    const kv_view kv(k_idxs, kqm, man.context, PG);
    tkv.stop();
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera qwen38_27b: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera qwen38_27b: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t *    pos  = (const int32_t *) inp_pos->data;
    const bool         mr   = inp_pos->ne[0] == 4 * (int64_t) T;  // M-RoPE: [4][T], else [T]
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
        }
    }
    const rope         rp(rope_node, inp_pos, T);
    GGML_ASSERT(rp.ROT == ROT);
    std::vector<float> h((size_t) T * H);
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const int64_t      c0 = kv.cell_of(0);
    if (c0 == 0) {  // a new sequence
        zero_state();
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
        });
    }
    if (T > 1) {
        // from the batch's page start: the page's earlier inputs (and the state saved there) before the batch's
        const int64_t s0 = c0 - c0 % PG;
        const int     k  = (int) (c0 - s0);
        if (k > 0 && (snap_cell != s0 || (int) (page_x.size() / H) < k)) {
            GGML_LOG_ERROR("axera qwen38_27b: a prompt from cell %lld, inside a page whose state is not kept\n", (long long) c0);
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
        // one token through the decode groups; at a page's first cell the state there is the restart point
        if (c0 % PG == 0) {
            save_state(c0);
        }
        memcpy(A->vir, xb.data(), H * sizeof(ggml_bf16_t));
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        rp.rows(0, 1, COS.get(), SIN.get());
        const int nblk = kv.mask_row(0, (ggml_bf16_t *) MASK->vir);
        MASK->flush(MASK->vir, (size_t) nblk * PG * sizeof(ggml_bf16_t));
        const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
        memcpy(CS->vir, COS->vir, half);
        memcpy((char *) CS->vir + half, SIN->vir, half);
        CS->flush();
        ((int32_t *) PN->vir)[0] = (int32_t) c0;
        ((int32_t *) PN->vir)[1] = nblk;
        ((int32_t *) PN->vir)[2] = (int32_t) ((c0 / PG) * 2 * PG + c0 % PG);
        PN->flush();
        {
            axera_timer tl("layers (decode)");
            for (auto & c : dec) {
                c.k->run(c.in, c.out, "decode group");
            }
        }
        A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, h.data(), H);
        page_x.insert(page_x.end(), xb.begin(), xb.end());
        page_p.insert(page_p.end(), pb.begin(), pb.end());
    }
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    head.run(g, names, h, T);
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen38_27b(const manifest & m) {
    return std::make_shared<qwen38_27b>(m);
}

}  // namespace axera
