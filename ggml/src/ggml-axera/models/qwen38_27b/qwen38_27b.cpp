// Axera NPU backend: Qwen3.8-27B (64 layers: 16 blocks of 3 gated delta net layers and one gated attention layer,
// dense MLP)
//
// One path, the manifest's kernels (ax-dsl-kernels models/qwen38_27b):
//   prefill  role prefill: the whole model in one call per page of 256 cells (a runtime loop over the 16 blocks, the
//            layers' weights in banks [16, ...] indexed by the block), the last page of a batch padded (valid_tokens).
//            Its history is whole pages: a batch that starts inside a page restarts at the page's first cell from the
//            recurrent state saved there, with the page's earlier inputs (kept) before its own
//   decode   role decode.group: 16 consecutive layers per call (weights in packs), 4 calls per token, then lm_head
//   verify   role decode.verify (speculative decoding: a batch of 2 - 4 tokens after the last cell): the decode
//            groups' layers for 4 rows per call, on the decode groups' packs, with the recurrent state after each row
//            (snapshots); a next batch that starts inside it (drafts rejected) continues from the snapshot there
//   mtp      roles mtp.t1 / mtp.prompt: the MTP block (blk.<L>.*, a draft context's graphs) on its own paged KV cache
// The state is this adapter's (CMM pools in the prefill kernel's layouts, the decode group bound to each layer's part):
// the KV cache paged [16][64 pages][K, V][256][4 x 256] bf16, the delta-net states [48][48][128][128] f32 and conv
// states q / k [48][3][2048], v [48][3][6144] bf16 (value heads in HF order); after a verification the current state of
// a delta-net layer is a snapshot's row (where[]). One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <array>
#include <cmath>
#include <cstring>
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
    // a second restart point kept: the page where the last prompt ended (a server's next turn restarts inside it)
    std::vector<uint8_t>     ksnap;
    int64_t                  ksnap_cell  = -1;
    bool                     snap_prompt = false;  // the current restart point is that page's
    std::vector<ggml_bf16_t> kpage_x;
    std::vector<int32_t>     kpage_p;
    std::vector<int32_t>     page_p;  // their M-RoPE positions [cells][4]

    std::vector<call>              dec;
    axera_cmm                      A, COS, SIN, CS, PN, MASK;
    // the current recurrent state of each delta-net layer: its pool part, or a verification snapshot's row (the
    // decode / verify calls' state outputs bound there); the snapshots of the verification groups, two sets
    struct place {
        const ggml_backend_axera_buffer_context * b = nullptr;
        char *                                    p = nullptr;
    };
    struct state_io {
        call * c;
        size_t i;
        int    kind, g;  // kind: 0 st, 1 cq, 2 ck, 3 cv; g: the delta-net layer
    };
    std::vector<place>                    where[4];
    std::vector<state_io>                 state_ios;
    std::vector<call>                     ver;  // per group (role decode.verify)
    std::vector<std::array<axera_cmm, 4>> snaps[2];
    axera_cmm                             VX, VCS, VPN, VMASK, VNT;
    int     cur_set = -1;              // the snapshot set holding the current state, -1: the pools
    int64_t ver_c0  = -1;              // the last verification's first cell, rows, snapshot set
    int     ver_T = 0, ver_set = -1;
    // the MTP block (draft context graphs): its kernels' IO, its KV cache
    call      mtp1, mtpp, mtp8;  // mtp8: role mtp.m8 (optional), up to 8 consecutive rows per call
    axera_cmm ME, MH, MCS, MPN, MMASK, MOUT, MRAW, PE, PH, PCS, PPN, PMASK, PNBLK, POUT2, PRAW, MKV;
    axera_cmm M8E, M8H, M8CS, M8PN, M8MASK, M8NT, M8OUT, M8RAW;
    const void * mtp_bound = nullptr;
    call                           pre;
    std::map<std::string, axera_cmm> banks;
    axera_cmm                      PX, POS, INDPTR, INDICES, LASTLEN, CLOC, VALID, POUT;
    output_head                    head;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_mtp(const std::map<std::string, ggml_tensor *> & t, int il);
    size_t part(int kind) const { return kind == 0 ? st_b : kind == 3 ? cv_b : cq_b; }
    const axera_cmm & pool(int kind) const { return kind == 0 ? ST : kind == 1 ? CQ : kind == 2 ? CK : CV; }
    void bind_state();
    void state_home();
    void state_at(int set, int row);
    ggml_status verify(const kv_view & kv, const rope & rp, const ggml_bf16_t * xb, const int32_t * pb, int T, float * h);
    ggml_status compute_mtp(ggml_cgraph * g);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    void      zero_state();
    void      save_state(int64_t cell);
    void      load_state();
    void      swap_kept();
    bool      restart_point(int64_t s0, int k);
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
    for (int k = 0; k < 4; ++k) {  // the current state: the pools
        where[k].assign(ng, {});
        for (int g = 0; g < ng; ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set = -1;
    ver_c0  = -1;
    // a call's state outputs st<j> / cq<j> / ck<j> / cv<j> (layer j of the group from layer l0): bound to where[]
    auto state_outs = [&](call & c, int l0) {
        for (size_t i = 0; i < c.k->desc->out.size(); ++i) {
            const std::string & n = c.k->desc->out[i].name;
            const int kind = n.rfind("st", 0) == 0 ? 0 : n.rfind("cq", 0) == 0 ? 1 : n.rfind("ck", 0) == 0 ? 2 :
                             n.rfind("cv", 0) == 0 ? 3 : -1;
            if (kind >= 0 && isdigit((unsigned char) n[2])) {
                state_ios.push_back({ &c, i, kind, ord[l0 + atoi(n.c_str() + 2)] });
            }
        }
    };
    state_ios.clear();
    for (int gi = 0; gi < L / G; ++gi) {
        state_outs(dec[gi], gi * G);
    }
    // verification: the decode groups' packs, x / out in place, 4 rows' cs / pn / mask, the snapshots
    ver.clear();
    if (man.has("decode.verify")) {
        const kernel & vk = man.get("decode.verify");
        GGML_ASSERT(vk.in_bytes("x") == 4 * H * sizeof(ggml_bf16_t) && vk.in_bytes("cs") == 4 * csb &&
                    vk.in_bytes("pn") == 4 * 3 * sizeof(int32_t) && vk.in_bytes("mask") == 4 * dk.in_bytes("mask"));
        VX    = axera_cmm_alloc(vk.in_bytes("x"));
        VCS   = axera_cmm_alloc(vk.in_bytes("cs"));
        VPN   = axera_cmm_alloc(vk.in_bytes("pn"));
        VMASK = axera_cmm_alloc(vk.in_bytes("mask"));
        VNT   = axera_cmm_alloc(sizeof(int32_t));
        static const char * SN[4] = { "ssnap", "cqsnap", "cksnap", "cvsnap" };
        for (auto & set : snaps) {
            set.clear();
            set.resize(L / G);
            for (auto & a : set) {
                for (int k = 0; k < 4; ++k) {
                    a[k] = axera_cmm_alloc(vk.out_bytes(SN[k]));
                    GGML_ASSERT(a[k]->size == 4 * (size_t) (G / 4 * 3) * part(k));
                }
            }
        }
        ver.assign(L / G, {});
        for (int gi = 0; gi < L / G; ++gi) {
            call & c = ver[gi];
            c.k      = &vk;
            for (const auto & d : vk.desc->in) {
                if (vk.desc->packs.count(d.name)) {  // the decode group's (the same layout)
                    GGML_ASSERT(dk.desc->packs.count(d.name) && dk.desc->packs.at(d.name) == vk.desc->packs.at(d.name));
                    size_t j = 0;
                    while (dk.desc->in[j].name != d.name) {
                        ++j;
                    }
                    c.in.push_back(dec[gi].in[j]);
                    continue;
                }
                const axera_cmm * b = d.name == "x" ? &VX : d.name == "cs" ? &VCS : d.name == "pn" ? &VPN :
                                      d.name == "mask" ? &VMASK : d.name == "ntok" ? &VNT : nullptr;
                if (!b) {
                    GGML_ABORT("axera qwen38_27b: decode.verify: unexpected input %s\n", d.name.c_str());
                }
                c.in.push_back(axera_io(*b, d.bytes));
            }
            for (const auto & d : vk.desc->out) {
                int k = -1;
                for (int j = 0; j < 4; ++j) {
                    k = d.name == SN[j] ? j : k;
                }
                c.out.push_back(d.name == "out" ? axera_io(VX, d.bytes) : k >= 0 ? axera_io(snaps[0][gi][k], d.bytes) :
                                                   pool_io(d.name, gi * G + atoi(d.name.c_str() + 2)));
            }
            state_outs(c, gi * G);
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

// the calls' state outputs to where the state is
void qwen38_27b::bind_state() {
    for (const auto & s : state_ios) {
        const place & w = where[s.kind][s.g];
        s.c->out[s.i]   = axera_io(w.b, w.p, (uint32_t) part(s.kind));
    }
}

// the current state into the pools (the prefill kernel's, the decode calls' default)
void qwen38_27b::state_home() {
    if (cur_set < 0) {
        return;
    }
    axera_timer tl("state home");
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            char * dst = (char *) pool(k)->vir + g * part(k);
            where[k][g].b->invalidate(where[k][g].p, part(k));
            memcpy(dst, where[k][g].p, part(k));
            where[k][g] = { pool(k).get(), dst };
        }
        pool(k)->flush();
    }
    cur_set = -1;
    bind_state();
}

// the current state: the snapshot after row `row` of the last verification (set `set`)
void qwen38_27b::state_at(int set, int row) {
    const int per = G / 4 * 3;  // delta-net layers per group
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            const axera_cmm & sn = snaps[set][g / per][k];
            where[k][g]          = { sn.get(), (char *) sn->vir + ((size_t) row * per + g % per) * part(k) };
        }
    }
    cur_set = set;
    bind_state();
}

void qwen38_27b::zero_state() {
    axera_timer tl("state zero");
    state_home();
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memset((*p)->vir, 0, (*p)->size);
        (*p)->flush();
    }
    snap_cell   = -1;
    ksnap_cell  = -1;
    snap_prompt = false;
    page_x.clear();
    page_p.clear();
}

void qwen38_27b::swap_kept() {
    std::swap(snap, ksnap);
    std::swap(snap_cell, ksnap_cell);
    std::swap(page_x, kpage_x);
    std::swap(page_p, kpage_p);
}

// the restart point of the page from s0 with its first k inputs (the current one, or the kept one made current)
bool qwen38_27b::restart_point(int64_t s0, int k) {
    if (snap_cell == s0 && (int) (page_x.size() / H) >= k) {
        return true;
    }
    if (ksnap_cell == s0 && (int) (kpage_x.size() / H) >= k) {
        swap_kept();
        snap_prompt = true;
        return true;
    }
    return false;
}

// the recurrent state (before cell `cell`, the first of its page) and the page's inputs from there
void qwen38_27b::save_state(int64_t cell) {
    GGML_ASSERT(cell % PG == 0);
    axera_timer tl("state save");
    if (snap_prompt && snap_cell >= 0) {  // the last prompt's page: kept
        swap_kept();
    }
    snap_prompt = false;
    snap.resize(ksnap.size() ? ksnap.size() : snap.size());
    size_t o = 0;
    for (int k = 0; k < 4; ++k) {  // from where the state is, in the pools' layout
        for (const place & w : where[k]) {
            w.b->invalidate(w.p, part(k));
            memcpy(snap.data() + o, w.p, part(k));
            o += part(k);
        }
    }
    snap_cell = cell;
    page_x.clear();
    page_p.clear();
}

void qwen38_27b::load_state() {
    axera_timer tl("state load");
    state_home();
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

// T (2 - 4) consecutive tokens from cell c0 > 0 (x rows xb, M-RoPE positions pb) through the verification groups, their
// hidden states into h: rows T..3 padded (the last row's positions and mask, later cells: KV rows nobody reads); the
// state after each row in the other snapshot set, the current state the last row's
ggml_status qwen38_27b::verify(const kv_view & kv, const rope & rp, const ggml_bf16_t * xb, const int32_t * pb, int T,
                               float * h) {
    const int64_t c0 = kv.cell_of(0);
    if (c0 % PG == 0) {  // a page's first cell: the state there is the restart point
        save_state(c0);
    }
    memset(VX->vir, 0, VX->size);
    memcpy(VX->vir, xb, (size_t) T * H * sizeof(ggml_bf16_t));
    VX->flush();
    const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
    auto *       cs   = (char *) VCS->vir;
    auto *       pn   = (int32_t *) VPN->vir;
    for (int r = 0; r < 4; ++r) {
        const int     rr = std::min(r, T - 1);
        const int64_t c  = c0 + r;
        rp.rows(rr, 1, COS.get(), SIN.get());
        memcpy(cs + r * 2 * half, COS->vir, half);
        memcpy(cs + r * 2 * half + half, SIN->vir, half);
        const int nblk = kv.mask_row(rr, (ggml_bf16_t *) VMASK->vir + (size_t) r * man.context);
        pn[3 * r]      = (int32_t) c;
        pn[3 * r + 1]  = std::max(nblk, (int) (c / PG + 1));
        pn[3 * r + 2]  = (int32_t) ((c / PG) * 2 * PG + c % PG);
    }
    *(int32_t *) VNT->vir = T;
    for (const axera_cmm * b : { &VCS, &VPN, &VMASK, &VNT }) {
        (*b)->flush();
    }
    const int set = cur_set == 0 ? 1 : 0;
    static const char * SN[4] = { "ssnap", "cqsnap", "cksnap", "cvsnap" };
    {
        axera_timer tl("layers (verify)");
        for (size_t gi = 0; gi < ver.size(); ++gi) {
            call & c = ver[gi];
            for (int k = 0; k < 4; ++k) {
                c.out[c.k->output(SN[k])] = axera_io(snaps[set][gi][k], snaps[set][gi][k]->size);
            }
            c.k->run(c.in, c.out, "verify group");
        }
    }
    VX->invalidate(VX->vir, (size_t) T * H * sizeof(ggml_bf16_t));
    for (int r = 0; r < T; ++r) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) VX->vir + (size_t) r * H, h + (size_t) r * H, H);
    }
    for (int r = 0; r < T; ++r) {
        if (r > 0 && (c0 + r) % PG == 0) {  // a page's first cell inside the batch: the state before it
            state_at(set, r - 1);
            save_state(c0 + r);
        }
        page_x.insert(page_x.end(), xb + (size_t) r * H, xb + (size_t) (r + 1) * H);
        page_p.insert(page_p.end(), pb + (size_t) r * 4, pb + (size_t) (r + 1) * 4);
    }
    state_at(set, T - 1);
    ver_c0  = c0;
    ver_T   = T;
    ver_set = set;
    return GGML_STATUS_SUCCESS;
}

ggml_status qwen38_27b::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return compute_mtp(g);
    }
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
    if (ver_c0 >= 0 && c0 > ver_c0 && c0 < ver_c0 + ver_T) {
        // the last verification's rows from c0 on rejected: its state after row c0 - 1, its inputs up to there
        state_at(ver_set, (int) (c0 - ver_c0 - 1));
        if (snap_cell > c0 && ksnap_cell >= 0 && ksnap_cell <= c0 && c0 < ksnap_cell + PG) {
            swap_kept();  // (a page start inside the rejected rows: the kept one, the page of c0)
            snap_prompt = true;
        }
        if (snap_cell > c0) {  // (no restart point of c0's page kept)
            snap_cell = -1;
            page_x.clear();
            page_p.clear();
        } else if (snap_cell >= 0) {
            page_x.resize((size_t) (c0 - snap_cell) * H);
            page_p.resize((size_t) (c0 - snap_cell) * 4);
        }
    }
    ver_c0 = -1;
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
    if (T > 1 && T <= 4 && !ver.empty() && c0 > 0 && c0 + 4 <= man.context) {
        axera_timer tv("verify (all groups)");
        const ggml_status st = verify(kv, rp, xb.data(), pb.data(), T, h.data());
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
    } else if (T > 1) {
        state_home();
        // from the batch's page start: the page's earlier inputs (and the state saved there) before the batch's
        const int64_t s0 = c0 - c0 % PG;
        const int     k  = (int) (c0 - s0);
        if (k > 0 && !restart_point(s0, k)) {
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
        snap_prompt = true;  // (its page's restart point outlives the decode's next pages)
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
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    head.run(g, names, h, T);
    return GGML_STATUS_SUCCESS;
}

// the MTP block's kernels: its packs from blk.<il>.* (both kernels the same layout), its paged KV cache, the IO
void qwen38_27b::init_mtp(const std::map<std::string, ggml_tensor *> & t, int il) {
    axera_timer    tb("init: mtp");
    const kernel & k1 = man.get("mtp.t1");
    const kernel & kp = man.get("mtp.prompt");
    GGML_ASSERT(k1.desc->packs == kp.desc->packs && k1.out_bytes("kv") == kp.out_bytes("kv"));
    MKV = axera_cmm_alloc(k1.out_bytes("kv"));
    auto bind = [&](call & c, const kernel & k, std::initializer_list<std::pair<const char *, axera_cmm *>> io) {
        c   = {};
        c.k = &k;
        auto of = [&](const std::string & n) -> axera_cmm * {
            for (const auto & [name, b] : io) {
                if (n == name) {
                    return b;
                }
            }
            return nullptr;
        };
        for (const auto & d : k.desc->in) {
            if (k.desc->packs.count(d.name)) {
                c.in.push_back(axera_io(prm.pack(t, *k.desc, d.name, il), d.bytes));
                continue;
            }
            axera_cmm * b = of(d.name);
            if (!b) {
                GGML_ABORT("axera qwen38_27b: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(d.bytes);
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : k.desc->out) {
            axera_cmm * b = d.name == std::string("kv") ? &MKV : of(d.name);
            if (!b) {
                GGML_ABORT("axera qwen38_27b: %s: unexpected output %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(d.bytes);
            }
            c.out.push_back(axera_io(*b, d.bytes));
        }
    };
    bind(mtp1, k1, { { "e", &ME }, { "h", &MH }, { "cs", &MCS }, { "pn", &MPN }, { "mask", &MMASK }, { "hnext", &MOUT },
                     { "hraw", &MRAW } });
    bind(mtpp, kp, { { "e", &PE }, { "h", &PH }, { "cs", &PCS }, { "pn", &PPN }, { "mask", &PMASK }, { "nblk", &PNBLK },
                     { "hnext", &POUT2 }, { "hraw", &PRAW } });
    GGML_ASSERT(MCS->size == (size_t) ROT * sizeof(ggml_bf16_t) && MMASK->size == (size_t) man.context * sizeof(ggml_bf16_t) &&
                PE->size % ((size_t) H * sizeof(ggml_bf16_t)) == 0);
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live = { &mtp1.in, &mtpp.in };
    if (man.has("mtp.m8")) {
        const kernel & k8 = man.get("mtp.m8");
        GGML_ASSERT(k8.desc->packs == k1.desc->packs && k8.out_bytes("kv") == k1.out_bytes("kv"));
        bind(mtp8, k8, { { "e", &M8E }, { "h", &M8H }, { "cs", &M8CS }, { "pn", &M8PN }, { "mask", &M8MASK }, { "ntok", &M8NT },
                         { "hnext", &M8OUT }, { "hraw", &M8RAW } });
        GGML_ASSERT(M8MASK->size == M8E->size / ((size_t) H * sizeof(ggml_bf16_t)) * MMASK->size);
        live.push_back(&mtp8.in);
    }
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
}

// a draft context's MTP graph: token x_p (its embedding row) with the target's final-norm hidden of p - 1 through the
// MTP block at the draft context's cells; h_nextn (the block's shared head norm) and the logits of the output rows
ggml_status qwen38_27b::compute_mtp(ggml_cgraph * g) {
    const auto &  t     = names.t;
    ggml_tensor * hin   = names.find("mtp_h_input");
    ggml_tensor * hn    = names.find("h_nextn");
    ggml_tensor * pos   = names.find("inp_pos");
    ggml_tensor * kidx  = names.find("attn_inp_k_idxs");
    ggml_tensor * kqm   = names.find("attn_inp_kq_mask");
    const ggml_tensor * embd = nullptr;
    int           il    = -1;
    for (int i = 0; i < g->n_nodes && !embd; i++) {  // the token rows: the input of the embedding norm mtp_enorm-<il>
        const char * n = g->nodes[i]->name;
        if (strncmp(n, "mtp_enorm-", 10) == 0) {
            il                    = atoi(n + 10);
            const ggml_tensor * m = g->nodes[i];
            const ggml_tensor * r = m->src[0]->op == GGML_OP_RMS_NORM ? m->src[0] : m->src[1];
            embd                  = r->op == GGML_OP_RMS_NORM ? r->src[0] : nullptr;
            if (embd && embd->op != GGML_OP_NONE) {  // (computed in this graph: the nodes before the norm on the CPU)
                int n0 = 0;
                while (n0 < g->n_nodes && g->nodes[n0] != r) {
                    ++n0;
                }
                cpu_compute(g, g->nodes, n0, "mtp input (cpu)");
            }
        }
    }
    if (!embd || !hn || !pos || !kidx || !kqm || il < 0 || !bound) {
        GGML_LOG_ERROR("axera qwen38_27b: an MTP graph before the target model's or of an unexpected shape (embd %d h_nextn %d "
                       "inp_pos %d k_idxs %d kq_mask %d layer %d target %d)\n", !!embd, !!hn, !!pos, !!kidx, !!kqm, il, !!bound);
        return GGML_STATUS_FAILED;
    }
    const int T = (int) hin->ne[1];
    GGML_ASSERT(hin->type == GGML_TYPE_F32 && hin->ne[0] == H && embd->type == GGML_TYPE_F32 && embd->ne[0] == H &&
                embd->ne[1] == T && hn->ne[1] == T);
    const ggml_tensor * w0 = t.at("blk." + std::to_string(il) + ".nextn.eh_proj.weight");
    if (mtp_bound != w0->data) {
        init_mtp(t, il);
        mtp_bound = w0->data;
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    const rope    rp(rope_node, pos, T);
    const kv_view kv(kidx, kqm, man.context, PG);
    auto row_bf16 = [&](const ggml_tensor * x, int r, ggml_bf16_t * d) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x->data + (size_t) r * x->nb[1]), d, H);
    };
    std::vector<float> h((size_t) T * H);
    const size_t       half = (ROT / 2) * sizeof(ggml_bf16_t);
    axera_timer        tm(T == 1 ? "mtp (draft)" : T <= 8 ? "mtp (catch-up rows)" : "mtp (prompt)");
    const int R8 = mtp8.k ? (int) (M8E->size / (H * sizeof(ggml_bf16_t))) : 0;
    if (T >= 2 && T <= R8) {  // a verification's accepted rows in one call (each row sees the rows before it)
        const size_t mrow = MMASK->size;
        auto *       pn   = (int32_t *) M8PN->vir;
        for (int r = 0; r < R8; ++r) {
            const int rr = std::min(r, T - 1);  // rows from T on: the last row's (the kernel leaves their KV alone)
            if (r < T) {
                row_bf16(embd, r, (ggml_bf16_t *) M8E->vir + (size_t) r * H);
                row_bf16(hin, r, (ggml_bf16_t *) M8H->vir + (size_t) r * H);
            }
            rp.rows(rr, 1, COS.get(), SIN.get());
            memcpy((char *) M8CS->vir + r * 2 * half, COS->vir, half);
            memcpy((char *) M8CS->vir + r * 2 * half + half, SIN->vir, half);
            auto * mk = (ggml_bf16_t *) ((char *) M8MASK->vir + r * mrow);
            std::fill(mk, mk + man.context, ggml_fp32_to_bf16(-3e38f));
            const int64_t c    = kv.cell_of(rr);
            const int     nblk = kv.mask_row(rr, mk);
            pn[3 * r]          = (int32_t) c;
            pn[3 * r + 1]      = std::max(nblk, (int) (c / PG + 1));
            pn[3 * r + 2]      = (int32_t) ((c / PG) * 2 * PG + c % PG);
        }
        *(int32_t *) M8NT->vir = T;
        for (const axera_cmm * b : { &M8E, &M8H, &M8CS, &M8PN, &M8MASK, &M8NT }) {
            (*b)->flush();
        }
        mtp8.k->run(mtp8.in, mtp8.out, "mtp m8");
        M8OUT->invalidate();
        for (int r = 0; r < T; ++r) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) M8OUT->vir + (size_t) r * H, h.data() + (size_t) r * H, H);
        }
    } else if (T <= 8) {  // a draft step, or a verification's accepted rows: row by row
        for (int r = 0; r < T; ++r) {
            row_bf16(embd, r, (ggml_bf16_t *) ME->vir);
            row_bf16(hin, r, (ggml_bf16_t *) MH->vir);
            rp.rows(r, 1, COS.get(), SIN.get());
            memcpy(MCS->vir, COS->vir, half);
            memcpy((char *) MCS->vir + half, SIN->vir, half);
            const int64_t c    = kv.cell_of(r);
            const int     nblk = kv.mask_row(r, (ggml_bf16_t *) MMASK->vir);
            auto *        pn   = (int32_t *) MPN->vir;
            pn[0]              = (int32_t) c;
            pn[1]              = std::max(nblk, (int) (c / PG + 1));
            pn[2]              = (int32_t) ((c / PG) * 2 * PG + c % PG);
            for (const axera_cmm * b : { &ME, &MH, &MCS, &MPN, &MMASK }) {
                (*b)->flush();
            }
            mtp1.k->run(mtp1.in, mtp1.out, "mtp t1");
            MOUT->invalidate();
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MOUT->vir, h.data() + (size_t) r * H, H);
        }
    } else {
        const int P = (int) (PE->size / (H * sizeof(ggml_bf16_t)));  // rows per call
        std::vector<ggml_bf16_t> mrow(man.context);
        for (int it = 0; it < T; it += P) {
            const int n = std::min(P, T - it);
            memset(PE->vir, 0, PE->size);
            memset(PH->vir, 0, PH->size);
            auto * pn   = (int32_t *) PPN->vir;
            auto * mk   = (ggml_bf16_t *) PMASK->vir;  // [pages][P][256]
            int    nmax = 0;
            for (int r = 0; r < P; ++r) {
                const int     rr = std::min(r, n - 1);  // padded rows: the last row's positions and mask, later cells
                const int64_t c  = kv.cell_of(it + rr) + (r - rr);
                GGML_ASSERT(c < man.context);
                if (r < n) {
                    row_bf16(embd, it + r, (ggml_bf16_t *) PE->vir + (size_t) r * H);
                    row_bf16(hin, it + r, (ggml_bf16_t *) PH->vir + (size_t) r * H);
                }
                rp.rows(it + rr, 1, COS.get(), SIN.get());
                memcpy((char *) PCS->vir + r * 2 * half, COS->vir, half);
                memcpy((char *) PCS->vir + r * 2 * half + half, SIN->vir, half);
                if (r < n) {  // (padded rows reuse the last real row's mask)
                    std::fill(mrow.begin(), mrow.end(), ggml_fp32_to_bf16(-3e38f));
                    nmax = std::max(nmax, kv.mask_row(it + rr, mrow.data()));
                }
                for (int b = 0; b < (int) (man.context / PG); ++b) {
                    memcpy(mk + ((size_t) b * P + r) * PG, mrow.data() + (size_t) b * PG, PG * sizeof(ggml_bf16_t));
                }
                pn[3 * r]     = (int32_t) c;
                pn[3 * r + 1] = (int32_t) (c / PG + 1);
                pn[3 * r + 2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
                nmax          = std::max(nmax, (int) (c / PG + 1));
            }
            *(int32_t *) PNBLK->vir = nmax;
            for (const axera_cmm * b : { &PE, &PH, &PCS, &PPN, &PMASK, &PNBLK }) {
                (*b)->flush();
            }
            mtpp.k->run(mtpp.in, mtpp.out, "mtp prompt");
            POUT2->invalidate();
            for (int r = 0; r < n; ++r) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT2->vir + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
            }
        }
    }
    if (hn->flags & GGML_TENSOR_FLAG_COMPUTE) {
        for (int r = 0; r < T; ++r) {
            memcpy((char *) hn->data + r * hn->nb[1], h.data() + (size_t) r * H, H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(hn->buffer)) {
            ((ggml_backend_axera_buffer_context *) hn->buffer->context)->flush(hn->data, ggml_nbytes(hn));
        }
    }
    ggml_tensor * logits = names.find("result_output");
    if (logits && logits->ne[1] > 0 && (logits->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        const std::vector<char> normed(T, 1);  // (the kernels' shared head norm)
        head.bind(man.get("lm_head"), logits->src[0]);
        head.run(g, names, h, T, &normed);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen38_27b(const manifest & m) {
    return std::make_shared<qwen38_27b>(m);
}

}  // namespace axera
