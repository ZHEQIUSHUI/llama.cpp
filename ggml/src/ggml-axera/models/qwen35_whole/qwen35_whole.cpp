// Axera NPU backend: the Qwen3.5 dense whole-model design (9B, 4B, 2B; 0.8B with MXFP8 weights from a Q8_0 GGUF). Every model of a manifest gets this
// adapter; the manifest names its kernels, and the kernel IO gives the model's layer counts and head sizes.
//
// One path, the manifest's kernels (ax-dsl-kernels models/qwen35_{2b,4b,9b}/whole):
//   prefill  role prefill: the whole model in one call per page of 256 cells (a runtime loop over the manifest's
//            blocks, the layers' weights in banks [blocks, ...] indexed by the block), the last page of a batch padded
//            (valid_tokens).
//            Its history is whole pages: a batch that starts inside a page restarts at the page's first cell from the
//            recurrent state saved there, with the page's earlier inputs (kept) before its own
//   decode   role decode.group: G consecutive layers per call (weights in packs), L / G calls per token, then lm_head
//   verify   role decode.verify, optional (speculative decoding): a batch of 2 - 4 tokens in one call
//   mtp      roles mtp.*, optional: a draft context's MTP block
// The state is this adapter's (CMM pools in the prefill kernel's layouts, the decode group bound to each layer's part):
// the KV cache paged [blocks][pages][K, V][256][kv heads x head dim] bf16, the delta-net states
// [layers][value heads][key dim][value dim] f32 and conv states q / k [layers][3][key heads x key dim], v
// [layers][3][value heads x value dim] bf16 (value heads in HF order); after a verification the current state of a
// delta-net layer is a snapshot's row (where[]). One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <regex>
#include <set>

namespace axera {
namespace {

constexpr int PG = 256;  // cells per page = prefill rows per call = the decode kernels' KV block

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

// the adapter blob in llama.cpp's sequence state (context checkpoints / prompt cache)
struct whole_state_header {
    char     magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t has_target;
    uint32_t has_mtp;
    int64_t  state_cell;
    int64_t  snap_cell;
    int64_t  ksnap_cell;
    int64_t  mtp_cell;
    uint32_t snap_prompt;
    uint32_t ncall;
    uint32_t pna;
    uint32_t nkv_pages;
    uint64_t kv_page_b;
    uint64_t kv_n;
    uint64_t st_b;
    uint64_t cq_b;
    uint64_t ck_b;
    uint64_t cv_b;
    uint64_t snap_n;
    uint64_t ksnap_n;
    uint64_t page_x_n;
    uint64_t page_p_n;
    uint64_t kpage_x_n;
    uint64_t kpage_p_n;
    uint64_t mtp_pages;
    uint64_t mtp_page_b;
    uint64_t mtp_n;
};

class qwen35_whole final : public model {
  public:
    explicit qwen35_whole(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size() const override;
    size_t       state_write(void * dst, size_t size) override;
    size_t       state_read(const void * src, size_t size) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, G = 0, ROT = 0, NBLK = 0;
    bool             fp8 = false;  // the prefill banks' projections: MXFP8 (else MXFP4), from the kernel IO shapes
    int              KH = 16, VH = 0, REP = 1;  // delta-net key / value heads (from the prefill kernels' IO)
    std::vector<int> ord;  // layer -> its index among the delta-net layers / the attention layers
    std::vector<int> state_group, state_local;  // a delta-net pool slot -> its verification group / local state

    // GGUF's tiled value head of HF (grouped) head g
    int tiled(int g) const { return (g % REP) * KH + g / REP; }

    // the state pools; their sizes per layer
    axera_cmm              ST, CQ, CK, CV;
    std::vector<axera_cmm> KVC;  // the KV cache: one buffer per prefill call's attention layers (a CMM allocation is < 4 GB)
    size_t    kv_b = 0, st_b = 0, cq_b = 0, cv_b = 0;
    // the pages of the cache the pools hold (llama.cpp's -c through the buffer type, GGML_AXERA_KV_PAGES overrides;
    // default: the kernels' context): the kernels keep their declared IO size but touch only the pages up to the runtime
    // page count, so a smaller context needs only its pages (a single-block prefill kernel only: a call of several
    // layers has fixed layer strides in its KV IO)
    int       kv_pages = 0;
    int64_t   kv_cells = 0;
    size_t    kv_alloc = 0;  // the bytes of one layer's pool
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
    std::vector<float>                    h_buf;      // the target graph's hidden rows
    std::vector<float>                    mtp_h_buf;  // the MTP graph's hidden rows
    axera_cmm                             VX, VCS, VPN, VMASK, VNT;
    int     cur_set = -1;              // the snapshot set holding the current state, -1: the pools
    int64_t ver_c0  = -1;              // the last verification's first cell, rows, snapshot set
    int     ver_T = 0, ver_set = -1;
    // the MTP block (draft context graphs): its kernels' IO, its KV cache
    call      mtp1, mtpp, mtppkv, mtp8;  // mtp8: role mtp.m8 (optional), up to 8 consecutive rows per call
    axera_cmm ME, MH, MCS, MPN, MMASK, MOUT, MRAW, PE, PH, PCS, PPN, PMASK, PNBLK, POUT2, PRAW, MKV;
    axera_cmm PKE, PKH, PKCS;  // the compact prompt kernels use their own row width (not mtp.prompt's 256)
    axera_cmm PK, PV;  // the K/V-only prompt pass' plain row outputs
    size_t    mtp_kv_row = 0;
    bool      mtp_h_f32 = false;  // the K/V prompt kernel takes FP32 h rows (the EF32 build)
    bool      mtp_e_f32 = false;  // the K/V prompt kernel takes FP32 token-embedding rows
    axera_cmm M8E, M8H, M8CS, M8PN, M8MASK, M8NT, M8OUT, M8RAW;
    const void * mtp_bound = nullptr;
    size_t       mtp_page_b = 0;  // bytes of one MTP KV page
    int64_t      mtp_cell  = 0;   // the next MTP KV cell
    std::vector<call>              pres;  // the prefill calls of a page (the kernel's blocks each)
    int                            ncall = 1;
    std::map<std::string, axera_cmm> banks;
    axera_cmm                      PX, POS, INDPTR, INDICES, LASTLEN, CLOC, VALID, POUT[2];
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
    int64_t   state_cell() const { return snap_cell < 0 ? 0 : snap_cell + (int64_t) (page_x.size() / std::max(1, H)); }
};

// rows of an [N, K] weight of the GGUF in the kernels' MX layout: MXFP8 codes [K] or MXFP4 codes [K / 2], and scales
// [K / 32]. A matching MXFP4 GGUF tensor is converted row by row as read; every other type (a community GGUF: Q4_K,
// Q8_0, ...) is quantized once, in the format the prefill kernel's bank IO selected.
struct mx4 {
    const ggml_tensor * w;
    axera_cmm           q, s;
    bool                fp8 = false;
    int64_t             N = 0, K = 0;
    mx4(const ggml_tensor * w, bool fp8) : w(w), fp8(fp8), N(w->ne[1]), K(w->ne[0]) {
        if (fp8) {
            axera_quantize_mxfp8(w, q, s);
        } else if (w->type != GGML_TYPE_MXFP4) {
            axera_quantize_mxfp4(w, q, s);
        }
    }
    // row r: its codes or its scales into dst
    void row(int64_t r, bool scales, uint8_t * dst) const {
        const size_t code = fp8 ? K : K / 2;
        if (q) {
            memcpy(dst, scales ? (const uint8_t *) s->vir + r * (K / 32) : (const uint8_t *) q->vir + r * code,
                   scales ? K / 32 : code);
            return;
        }
        GGML_ASSERT(!fp8 && w->type == GGML_TYPE_MXFP4);
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

axera_cmm qwen35_whole::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera qwen35_whole: prefill: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const bool        cs    = n == "cos" || n == "sin";
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(d.shape.size() >= 2 && nb > 0 && (nb == L / 4 || cs) && d.bytes % nb == 0);
    axera_cmm     c  = axera_cmm_alloc(d.bytes);
    const size_t  bb = d.bytes / nb;  // bytes per block
    std::vector<const ggml_tensor *> read;  // the GGUF tensors read: their mapped pages dropped when the bank is done
    auto          T  = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen35_whole: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
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
        // MX rows r0 .. of a weight (value heads of row size hr from vrow on in HF order); scales or codes
        auto rows = [&](const mx4 & w, int64_t r0, int64_t nr, bool scales, int64_t vrow = -1, int64_t hr = 0) {
            const size_t rb = scales ? w.K / 32 : w.fp8 ? w.K : w.K / 2;
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
            const size_t rb = scales ? w.K / 32 : w.fp8 ? w.K : w.K / 2, cb = rb / VH;
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
                const mx4 w(T(il, W.at(n.substr(1))), fp8);
                rows(w, 0, w.N, false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                const mx4 w(T(il, W.at(n.substr(7))), fp8);
                rows(w, 0, w.N, true);
            } else {
                GGML_ABORT("axera qwen35_whole: prefill: unexpected MLP input %s\n", d.name.c_str());
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
                const mx4 w(T(il, W.at(p)), fp8);
                rows(w, 0, w.N, n != "w_" + p);
            } else {
                GGML_ABORT("axera qwen35_whole: prefill: unexpected attention input %s\n", d.name.c_str());
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
                const mx4     w(T(il, "attn_qkv.weight"), fp8);
                const int64_t nr = d.shape[1];  // rows q, k (nr each), then v (the rest)
                if (p == 'v') {
                    rows(w, w.N - nr, nr, scales, w.N - nr, nr / VH);
                } else {
                    rows(w, p == 'q' ? 0 : nr, nr, scales);
                }
            } else if (p == 'z') {
                const mx4 w(T(il, "attn_gate.weight"), fp8);
                rows(w, 0, w.N, scales, 0, w.N / VH);
            } else if (p == 'a' || p == 'b') {  // small GGUF tensors: quantized to the bank's MX format like every projection
                const mx4 w(T(il, p == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight"), fp8);
                rows(w, 0, w.N, scales, 0, 1);
            } else if (p == 'o') {
                const mx4 w(T(il, "ssm_out.weight"), fp8);
                cols(w, scales);
            } else {
                GGML_ABORT("axera qwen35_whole: prefill: unexpected delta-net input %s\n", d.name.c_str());
            }
        } else {
            GGML_ABORT("axera qwen35_whole: prefill: unexpected delta-net input %s\n", d.name.c_str());
        }
    }
    c->flush();
    for (const ggml_tensor * w : read) {
        axera_drop_mapped(w);
    }
    return c;
}

void qwen35_whole::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen35_whole: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        ord[il] = g ? 0 : na++;  // the delta-net rows are laid out below, once the prefill call size is known
        if (g) {
            ++ng;
        }
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
        GGML_ABORT("axera qwen35_whole: %s has no IO %s\n", k.desc->file.c_str(), n.c_str());
    };
    const io_desc & pkv = in_of(pk, "b3_kv"), &pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in"), &pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const int pna = (int) pkv.shape[0], png = 3 * pna;  // the prefill kernel's layers per call
    GGML_ASSERT(pna > 0 && na % pna == 0);
    ncall = na / pna;
    GGML_ASSERT((pst.shape[0] == png || pst.shape[0] == ng || pst.shape[0] == L / 4 || pst.shape[0] == pna) &&
                pcq.shape[0] == pst.shape[0] && pcv.shape[0] == pst.shape[0]);
    // The common state bank (b0 [3 * pna]) is phase-major inside one call; calls follow one another. A call's
    // [3 * pna] rows are then one contiguous slice of the pool, which is what the prefill calls bind below. A build
    // with one bank per phase (b0 / b1 / b2, each [blocks]) keeps the phase-major pool instead.
    const bool common_state = pst.shape[0] == png || pst.shape[0] == ng;
    for (int il = 0; il < L; ++il) {
        if (t.count("blk." + std::to_string(il) + ".ssm_a") == 0) {
            continue;
        }
        const int b = il / 4, p = il % 4;
        ord[il] = common_state ? (b / pna) * png + p * pna + b % pna : p * (L / 4) + b;
    }
    state_group.assign(ng, -1);
    state_local.assign(ng, -1);
    for (int il = 0; il < L; ++il) {
        if (t.count("blk." + std::to_string(il) + ".ssm_a") == 0) {
            continue;
        }
        const int gi = (il / 4) / (G / 4), j = il - gi * G;
        state_group[ord[il]] = gi;
        state_local[ord[il]] = (j / 4) * 3 + j % 4;
    }
    // the delta-net head geometry: the state [layers, 1, VH, KD, VD], and conv1d's channels 2 * KH * KD + VH * VD
    GGML_ASSERT(pst.shape.size() == 5 && pcv.shape.size() >= 2);
    VH = (int) pst.shape[2];
    const int64_t KD = pst.shape[3], VD = pst.shape[4];
    const ggml_tensor * conv_w = t.at("blk.0.ssm_conv1d.weight");
    GGML_ASSERT(conv_w && conv_w->ne[1] == 2 * (int64_t) KH * KD + (int64_t) VH * VD);
    KH  = (int) ((conv_w->ne[1] - VH * VD) / (2 * KD));
    REP = VH / KH;
    GGML_ASSERT(KH > 0 && REP * KH == VH && conv_w->ne[1] == 2 * KH * KD + VH * VD);
    kv_b = pkv.bytes / pna;
    st_b = pst.bytes / pst.shape[0];
    cq_b = pcq.bytes / pcq.shape[0];
    cv_b = pcv.bytes / pcv.shape[0];
    GGML_ASSERT(in_of(dk, "kv3").bytes == kv_b && in_of(dk, "st0").bytes == st_b && in_of(dk, "cq0").bytes == cq_b &&
                in_of(dk, "ck0").bytes == cq_b && in_of(dk, "cv0").bytes == cv_b);
    NBLK = (int) pkv.shape[1];
    GGML_ASSERT(pkv.shape[3] == PG && (int64_t) NBLK * PG == man.context);
    kv_pages = NBLK;
    if (const int64_t cells = axera_kv_cells(); cells > 0 && pna == 1) {  // llama.cpp's -c (its own cache's cells)
        kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(NBLK, (cells + PG - 1) / PG));
    }
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        kv_pages = std::max(1, std::min(NBLK, atoi(e)));
    }
    GGML_ASSERT(kv_pages == NBLK || pna == 1);
    kv_cells = (int64_t) kv_pages * PG;
    kv_alloc = kv_b / NBLK * kv_pages;
    if (kv_pages < NBLK) {
        GGML_LOG_INFO("axera qwen35_whole: the KV pools hold %d of the kernels' %d pages (%lld cells)\n", kv_pages, NBLK, (long long) kv_cells);
    }
    KVC.resize(ncall);
    for (int k = 0; k < ncall; ++k) {
        KVC[k] = axera_cmm_alloc(kv_alloc * pna);
    }
    ST = axera_cmm_alloc((size_t) ng * st_b);
    CQ = axera_cmm_alloc((size_t) ng * cq_b);
    CK = axera_cmm_alloc((size_t) ng * cq_b);
    CV = axera_cmm_alloc((size_t) ng * cv_b);
    snap.assign((size_t) ng * (st_b + 2 * cq_b + cv_b), 0);
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
            const int a = ord[il];
            return axera_io(KVC[a / pna].get(), (char *) KVC[a / pna]->vir + (size_t) (a % pna) * kv_b, (uint32_t) kv_b);
        }
        const axera_cmm & p = n[0] == 's' ? ST : n[1] == 'q' ? CQ : n[1] == 'k' ? CK : CV;
        const size_t      b = n[0] == 's' ? st_b : n[1] == 'v' ? cv_b : cq_b;
        return axera_io(p.get(), (char *) p->vir + ord[il] * b, (uint32_t) b);
    };
    // the prefill kernel's weight banks [blocks, ..] from the GGUF (first: decode and verification calls may read
    // their row-split weights from them, inputs k_<bank>: the blocks of the call's G layers)
    const rope rp(rope_node, names.find("inp_pos"), 1);
    banks.clear();
    {
        axera_timer tb("init: prefill banks");
        // The prefill bank IO distinguishes the two MX layouts without relying on the GGUF type (Q8_0 is converted at
        // load): an MXFP8 code row is K bytes, an MXFP4 code row K / 2, while both scale rows are K / 32. The first
        // MLP gate bank is present in every whole-model prefill, in either format.
        const io_desc * code = nullptr, * scale = nullptr;
        for (const auto & d : pk.desc->in) {
            if (d.name == "b0_mlp_w1") {
                code = &d;
            } else if (d.name == "b0_mlp_w_scale1") {
                scale = &d;
            }
        }
        GGML_ASSERT(code && scale && code->shape.size() >= 2 && scale->shape.size() >= 2);
        const int64_t code_k = code->shape.back(), scale_k = scale->shape.back() * 32;
        GGML_ASSERT(code_k == scale_k || code_k * 2 == scale_k);
        fp8 = code_k == scale_k;
        GGML_LOG_INFO("axera qwen35_whole: prefill banks are MXFP%d (%lld codes, %lld scales per row)\n",
                      fp8 ? 8 : 4, (long long) code_k, (long long) scale->shape.back());
        static const std::set<std::string> pools = { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                                                     "valid_tokens", "b3_kv" };
        // b0 / b1 / b2: the recurrent state banks of the three delta-net phases (a build may use one common b0 for all)
        auto ends = [](const std::string & s, const char * suffix) {
            const size_t n = strlen(suffix);
            return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
        };
        auto state_pool = [&](const std::string & n) {
            return n.size() > 5 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                   (ends(n, "_attn_state_in") || ends(n, "_attn_q_conv_state_in") || ends(n, "_attn_k_conv_state_in") ||
                    ends(n, "_attn_v_conv_state_in"));
        };
        for (const auto & d : pk.desc->in) {
            if (!pools.count(d.name) && !state_pool(d.name)) {
                // the weights of all L / 4 blocks (a call binds its slice below); the cos / sin tables are the same
                // for every block, so the kernel's blocks of them are bound whole to every call
                const bool cs = ends(d.name, "_attn_cos") || ends(d.name, "_attn_sin");
                io_desc    dd = d;
                if (!cs && ncall > 1) {
                    dd.shape[0] *= ncall;
                    dd.bytes *= ncall;
                }
                banks[d.name] = bank(t, dd, rp);
            }
        }
    }
    auto bank_io = [&](const kernel & k, const io_desc & d, int gi) -> AX_ENGINE_IO_BUFFER_T {
        auto it = banks.find(d.name.substr(2));
        if (d.name.rfind("k_", 0) != 0 || it == banks.end()) {
            GGML_ABORT("axera qwen35_whole: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
        }
        const size_t bb = it->second->size / (L / 4);  // a block's bytes
        GGML_ASSERT(d.bytes == (size_t) (G / 4) * bb);
        return axera_io(it->second.get(), (char *) it->second->vir + (size_t) gi * d.bytes, d.bytes);
    };
    dec.assign(L / G, {});
    for (int gi = 0; gi < L / G; ++gi) {
        call & c = dec[gi];
        c.k      = &dk;
        for (const auto & d : dk.desc->in) {
            if (d.name.rfind("k_", 0) == 0) {
                c.in.push_back(bank_io(dk, d, gi));
                continue;
            }
            if (dk.desc->packs.count(d.name)) {
                c.in.push_back(axera_io(prm.pack(t, *dk.desc, d.name, gi * G), d.bytes));
                continue;
            }
            const axera_cmm * b = d.name == "x" ? &A : d.name == "cs" ? &CS : d.name == "pn" ? &PN : d.name == "mask" ? &MASK : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen35_whole: decode.group: unexpected input %s\n", d.name.c_str());
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : dk.desc->out) {
            c.out.push_back(d.name == "out" ? axera_io(A, d.bytes) : pool_io(d.name, gi * G + atoi(d.name.c_str() + 2)));
        }
        // this group's converted parameters, read only through its packs now: freed before the next group's
        // (the load's peak: the packs so far and one group's conversions, not every layer's)
        std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
        for (int gj = 0; gj <= gi; ++gj) {
            live.push_back(&dec[gj].in);
        }
        prm.release(live);
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
                if (d.name.rfind("k_", 0) == 0) {
                    c.in.push_back(bank_io(vk, d, gi));
                    continue;
                }
                const axera_cmm * b = d.name == "x" ? &VX : d.name == "cs" ? &VCS : d.name == "pn" ? &VPN :
                                      d.name == "mask" ? &VMASK : d.name == "ntok" ? &VNT : nullptr;
                if (!b) {
                    GGML_ABORT("axera qwen35_whole: decode.verify: unexpected input %s\n", d.name.c_str());
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

    // prefill: the weight banks (above), the pools, one page's inputs
    PX      = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POUT[0] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POUT[1] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
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
    GGML_ASSERT(pk.desc->out.size() == 1 && pk.desc->out[0].name == "output" && pk.desc->out[0].bytes == POUT[0]->size);
    pres.assign(ncall, {});
    auto ends = [](const std::string & s, const char * suffix) {
        const size_t z = strlen(suffix);
        return s.size() >= z && s.compare(s.size() - z, z, suffix) == 0;
    };
    for (int k = 0; k < ncall; ++k) {  // call k: blocks k * pna .. of the banks, its layers of the pools
        call & c = pres[k];
        c.k      = &pk;
        for (const auto & d : pk.desc->in) {
            const std::string & n = d.name;
            const axera_cmm *   b = n == "x" ? (k == 0 ? &PX : &POUT[(k - 1) & 1]) : n == "position_ids" ? &POS :
                                    n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                    n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID : nullptr;
            size_t off = 0;
            if (!b && n == "b3_kv") {
                b = &KVC[k];
            } else if (!b && n.size() > 5 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                       (ends(n, "_attn_state_in") || ends(n, "_attn_q_conv_state_in") || ends(n, "_attn_k_conv_state_in") ||
                        ends(n, "_attn_v_conv_state_in"))) {
                const int phase = n[1] - '0';
                const int kind  = n.rfind("_attn_state_in") != std::string::npos ? 0 :
                                  n.rfind("_attn_q_conv_state_in") != std::string::npos ? 1 :
                                  n.rfind("_attn_k_conv_state_in") != std::string::npos ? 2 : 3;
                b = &pool(kind);
                // A common bank (b0 [layers]) has the phase-major pool's own order: a call's slice follows its
                // predecessor's. A phase bank (bP [blocks]) is one contiguous phase of the pool, and the call's
                // blocks follow in that phase.
                if (d.shape[0] == png && ncall > 1) {
                    off = (size_t) k * d.bytes;
                } else if (d.shape[0] == ng) {
                    off = 0;
                } else if (d.shape[0] == L / 4) {
                    off = (size_t) phase * (L / 4) * part(kind);
                } else if (d.shape[0] == pna) {
                    off = (size_t) phase * (L / 4) * part(kind) + (size_t) k * d.bytes;
                } else {
                    GGML_ABORT("axera qwen35_whole: prefill state %s: shape0 %lld\n", n.c_str(), (long long) d.shape[0]);
                }
                if (d.bytes != (size_t) d.shape[0] * part(kind) || off + d.bytes > (*b)->size) {
                    GGML_ABORT("axera qwen35_whole: prefill state %s: shape0 %lld bytes %zu part %zu bank %zu off %zu\n", n.c_str(),
                               (long long) d.shape[0], (size_t) d.bytes, (size_t) part(kind), (*b)->size, off);
                }
            } else if (!b) {
                b   = &banks.at(n);
                // (cos / sin are bound whole; the weight banks hold every call's blocks)
                off = (*b)->size >= (size_t) ncall * d.bytes ? (size_t) k * d.bytes : 0;
            }
            GGML_ASSERT((*b)->size >= off + (n == "b3_kv" ? kv_alloc : d.bytes));
            c.in.push_back(axera_io(b->get(), (char *) (*b)->vir + off, (uint32_t) d.bytes));
        }
        c.out.push_back(axera_io(POUT[k & 1], pk.desc->out[0].bytes));
    }

    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen35_whole: %d layers (%d per decode call), prefill banks %.1f GiB, state %.2f GiB, %lld cells\n",
                  L, G, bb / 1073741824.0, (kv_b * na + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}

// the calls' state outputs to where the state is
void qwen35_whole::bind_state() {
    for (const auto & s : state_ios) {
        const place & w = where[s.kind][s.g];
        s.c->out[s.i]   = axera_io(w.b, w.p, (uint32_t) part(s.kind));
    }
}

// the current state into the pools (the prefill kernel's, the decode calls' default)
void qwen35_whole::state_home() {
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
void qwen35_whole::state_at(int set, int row) {
    const int per = G / 4 * 3;  // delta-net layers per group
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            GGML_ASSERT(k > 0 || (state_group[g] >= 0 && state_local[g] >= 0));
            const axera_cmm & sn = snaps[set][state_group[g]][k];
            where[k][g]          = { sn.get(), (char *) sn->vir + ((size_t) row * per + state_local[g]) * part(k) };
        }
    }
    cur_set = set;
    bind_state();
}

void qwen35_whole::zero_state() {
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

void qwen35_whole::swap_kept() {
    std::swap(snap, ksnap);
    std::swap(snap_cell, ksnap_cell);
    std::swap(page_x, kpage_x);
    std::swap(page_p, kpage_p);
}

// the restart point of the page from s0 with its first k inputs (the current one, or the kept one made current)
bool qwen35_whole::restart_point(int64_t s0, int k) {
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
void qwen35_whole::save_state(int64_t cell) {
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

void qwen35_whole::load_state() {
    axera_timer tl("state load");
    state_home();
    size_t o = 0;
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memcpy((*p)->vir, snap.data() + o, (*p)->size);
        (*p)->flush();
        o += (*p)->size;
    }
}

// llama.cpp's sequence state (prompt cache / context checkpoint). The private CMM pools are not the cache tensors
// llama.cpp serializes, so the state stream carries this adapter blob: the KV pages before the current cell, the
// recurrent state, and the current page's restart point (the recurrent state at the page start plus its inputs).
// Only the pages up to the current cell are copied, plus the current page: a later prompt can overwrite the pages
// after the checkpoint, and a restore rewrites the checkpoint's page from its restart point.
size_t qwen35_whole::state_size() const {
    if (KVC.empty() || !ST) {
        return 0;
    }
    const int    pna         = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    const int64_t cur_cell   = state_cell();
    const bool   has_target  = cur_cell > 0 || snap_cell >= 0 || !page_x.empty();
    const size_t nkv_pages   = has_target ? std::min<size_t>((size_t) kv_pages, (size_t) ((cur_cell + PG - 1) / PG)) : 0;
    const size_t mtp_pages   = MKV && mtp_page_b ? std::min<size_t>(MKV->size / mtp_page_b, (size_t) ((mtp_cell + PG - 1) / PG)) : 0;
    const bool   has_mtp     = mtp_pages > 0;
    if (!has_target && !has_mtp) {
        return 0;
    }
    size_t n = sizeof(whole_state_header);
    if (has_target) {
        n += (size_t) KVC.size() * pna * nkv_pages * (kv_b / NBLK);
        n += ST->size + CQ->size + CK->size + CV->size;
        n += snap.size() + ksnap.size() + page_x.size() * sizeof(ggml_bf16_t) + page_p.size() * sizeof(int32_t) +
             kpage_x.size() * sizeof(ggml_bf16_t) + kpage_p.size() * sizeof(int32_t);
    }
    n += mtp_pages * mtp_page_b;
    return n;
}

size_t qwen35_whole::state_write(void * dst, size_t size) {
    axera_timer tw("state write");
    const size_t total = state_size();
    if (!total || size < total) {
        return 0;
    }
    const int    pna       = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    const int64_t cur_cell = state_cell();
    const bool   has_target = cur_cell > 0 || snap_cell >= 0 || !page_x.empty();
    const size_t nkv_pages = has_target ? std::min<size_t>((size_t) kv_pages, (size_t) ((cur_cell + PG - 1) / PG)) : 0;
    const size_t mtp_pages = MKV && mtp_page_b ? std::min<size_t>(MKV->size / mtp_page_b, (size_t) ((mtp_cell + PG - 1) / PG)) : 0;
    if (has_target) {
        state_home();  // a verification leaves the current state in a snapshot row
    }

    whole_state_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "AXW35S1", sizeof(h.magic) - 1);
    h.version     = 1;
    h.kind        = 1;
    h.has_target  = has_target;
    h.has_mtp     = mtp_pages > 0;
    h.state_cell  = cur_cell;
    h.snap_cell   = snap_cell;
    h.ksnap_cell  = ksnap_cell;
    h.mtp_cell    = mtp_cell;
    h.snap_prompt = snap_prompt;
    h.ncall       = (uint32_t) KVC.size();
    h.pna         = (uint32_t) pna;
    h.nkv_pages   = (uint32_t) nkv_pages;
    h.kv_page_b   = kv_b / NBLK;
    h.kv_n        = (uint64_t) KVC.size() * pna * nkv_pages * (kv_b / NBLK);
    h.st_b        = ST->size;
    h.cq_b        = CQ->size;
    h.ck_b        = CK->size;
    h.cv_b        = CV->size;
    h.snap_n      = snap.size();
    h.ksnap_n     = ksnap.size();
    h.page_x_n    = page_x.size() * sizeof(ggml_bf16_t);
    h.page_p_n    = page_p.size() * sizeof(int32_t);
    h.kpage_x_n   = kpage_x.size() * sizeof(ggml_bf16_t);
    h.kpage_p_n   = kpage_p.size() * sizeof(int32_t);
    h.mtp_pages   = mtp_pages;
    h.mtp_page_b  = mtp_page_b;
    h.mtp_n       = mtp_pages * mtp_page_b;

    char * p = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (has_target) {
        const size_t page_n = nkv_pages * (kv_b / NBLK);
        for (const axera_cmm & c : KVC) {
            for (int j = 0; j < pna; ++j) {
                char * src = (char *) c->vir + (size_t) j * kv_alloc;
                c->invalidate(src, page_n);
                memcpy(p, src, page_n);
                p += page_n;
            }
        }
        for (const axera_cmm * b : { &ST, &CQ, &CK, &CV }) {
            (*b)->invalidate((*b)->vir, (*b)->size);
            memcpy(p, (*b)->vir, (*b)->size);
            p += (*b)->size;
        }
        const auto put = [&](const void * src, size_t n) {
            if (n) {
                memcpy(p, src, n);
                p += n;
            }
        };
        put(snap.data(), h.snap_n);
        put(ksnap.data(), h.ksnap_n);
        put(page_x.data(), h.page_x_n);
        put(page_p.data(), h.page_p_n);
        put(kpage_x.data(), h.kpage_x_n);
        put(kpage_p.data(), h.kpage_p_n);
    }
    if (h.mtp_n) {
        MKV->invalidate(MKV->vir, h.mtp_n);
        memcpy(p, MKV->vir, h.mtp_n);
        p += h.mtp_n;
    }
    return (size_t) (p - (char *) dst);
}

size_t qwen35_whole::state_read(const void * src, size_t size) {
    axera_timer tr("state read");
    if (size < sizeof(whole_state_header)) {
        return 0;
    }
    whole_state_header h;
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, "AXW35S1", sizeof(h.magic) - 1) != 0 || h.version != 1 || h.kind != 1) {
        return 0;
    }
    const int pna = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    if (h.has_target && (KVC.size() != h.ncall || pna != (int) h.pna || h.kv_page_b != kv_b / NBLK ||
                         h.nkv_pages > (uint32_t) kv_pages || h.st_b != ST->size || h.cq_b != CQ->size ||
                         h.ck_b != CK->size || h.cv_b != CV->size || h.kv_n != (uint64_t) h.ncall * h.pna * h.nkv_pages * h.kv_page_b ||
                         h.page_x_n % (H * sizeof(ggml_bf16_t)) || h.page_p_n % (4 * sizeof(int32_t)) ||
                         h.kpage_x_n % (H * sizeof(ggml_bf16_t)) || h.kpage_p_n % (4 * sizeof(int32_t)) ||
                         h.state_cell < 0 || h.state_cell > kv_cells)) {
        return 0;
    }
    const bool mtp_valid = h.mtp_n == 0 ? h.mtp_pages == 0 && !h.has_mtp :
                           h.has_mtp && MKV && mtp_page_b && h.mtp_page_b == mtp_page_b &&
                               h.mtp_pages <= MKV->size / mtp_page_b && h.mtp_pages * mtp_page_b == h.mtp_n &&
                               h.mtp_cell >= 0 && h.mtp_cell <= kv_cells;
    if (!mtp_valid) {
        return 0;
    }
    size_t need = sizeof(h);
    if (h.has_target) {
        need += h.kv_n + h.st_b + h.cq_b + h.ck_b + h.cv_b + h.snap_n + h.ksnap_n + h.page_x_n + h.page_p_n + h.kpage_x_n +
                h.kpage_p_n;
    }
    need += h.mtp_n;
    if (size < need) {
        return 0;
    }

    const char * p = (const char *) src + sizeof(h);
    if (h.has_target) {
        const size_t page_n = h.nkv_pages * h.kv_page_b;
        for (const axera_cmm & c : KVC) {
            for (int j = 0; j < (int) h.pna; ++j) {
                char * d = (char *) c->vir + (size_t) j * kv_alloc;
                memcpy(d, p, page_n);
                c->flush(d, page_n);
                p += page_n;
            }
        }
        for (const axera_cmm * b : { &ST, &CQ, &CK, &CV }) {
            memcpy((*b)->vir, p, (*b)->size);
            (*b)->flush((*b)->vir, (*b)->size);
            p += (*b)->size;
        }
        const auto get = [&](void * dst, size_t n) {
            if (n) {
                memcpy(dst, p, n);
                p += n;
            }
        };
        snap.resize(h.snap_n);
        ksnap.resize(h.ksnap_n);
        page_x.resize(h.page_x_n / sizeof(ggml_bf16_t));
        page_p.resize(h.page_p_n / sizeof(int32_t));
        kpage_x.resize(h.kpage_x_n / sizeof(ggml_bf16_t));
        kpage_p.resize(h.kpage_p_n / sizeof(int32_t));
        get(snap.data(), h.snap_n);
        get(ksnap.data(), h.ksnap_n);
        get(page_x.data(), h.page_x_n);
        get(page_p.data(), h.page_p_n);
        get(kpage_x.data(), h.kpage_x_n);
        get(kpage_p.data(), h.kpage_p_n);
        snap_cell   = h.snap_cell;
        ksnap_cell  = h.ksnap_cell;
        snap_prompt = h.snap_prompt;
        cur_set     = -1;
        ver_c0      = -1;
        ver_T       = 0;
        ver_set     = -1;
        for (int k = 0; k < 4; ++k) {
            for (size_t g = 0; g < where[k].size(); ++g) {
                where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
            }
        }
        bind_state();
    }
    if (h.mtp_n) {
        memcpy(MKV->vir, p, h.mtp_n);
        MKV->flush(MKV->vir, h.mtp_n);
        p += h.mtp_n;
    }
    mtp_cell = h.mtp_cell;
    return (size_t) (p - (const char *) src);
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16 and their M-RoPE positions p4 [n][4]; the hidden
// states of the rows from out0 on into h
void qwen35_whole::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0);
    GGML_ASSERT(c0 >= 0 && n >= 0 && c0 + n <= man.context);
    GGML_ASSERT(c0 + n <= kv_cells);  // (the pools' pages)
    for (int64_t r = 0; r < n; ++r) {
        for (int s = 0; s < 4; ++s) {
            GGML_ASSERT(p4[r * 4 + s] >= 0 && p4[r * 4 + s] < man.context);
        }
    }
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
            for (auto & c : pres) {
                c.k->run(c.in, c.out, "prefill");
            }
        }
        axera_timer       to("prefill out rows");
        const axera_cmm & po = POUT[(ncall - 1) & 1];
        po->invalidate(po->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
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
ggml_status qwen35_whole::verify(const kv_view & kv, const rope & rp, const ggml_bf16_t * xb, const int32_t * pb, int T,
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

ggml_status qwen35_whole::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return compute_mtp(g);
    }
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_whole: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
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
    // stop before a nextn / MTP block: its attn_norm is not a target-model layer
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight") &&
           !t.count("blk." + std::to_string(L_) + ".nextn.eh_proj.weight")) {
        ++L_;
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera qwen35_whole: no ROPE node in the graph");
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
        GGML_LOG_ERROR("axera qwen35_whole: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera qwen35_whole: a batch of non-consecutive cells\n");
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
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
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
    if (T > 1 && T <= 4 && !ver.empty() && c0 > 0 && c0 + 4 <= kv_cells) {
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
            GGML_LOG_ERROR("axera qwen35_whole: a prompt from cell %lld, inside a page whose state is not kept\n", (long long) c0);
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
        GGML_ASSERT(c0 >= 0 && c0 < kv_cells);
        if (c0 % PG == 0) {
            save_state(c0);
        }
        memcpy(A->vir, xb.data(), H * sizeof(ggml_bf16_t));
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        rp.rows(0, 1, COS.get(), SIN.get());
        const int nblk = kv.mask_row(0, (ggml_bf16_t *) MASK->vir);
        GGML_ASSERT(nblk > 0 && nblk <= NBLK);
        MASK->flush(MASK->vir, (size_t) nblk * PG * sizeof(ggml_bf16_t));
        const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
        memcpy(CS->vir, COS->vir, half);
        memcpy((char *) CS->vir + half, SIN->vir, half);
        CS->flush();
        ((int32_t *) PN->vir)[0] = (int32_t) c0;
        ((int32_t *) PN->vir)[1] = nblk;
        ((int32_t *) PN->vir)[2] = (int32_t) ((c0 / PG) * 2 * PG + c0 % PG);
        GGML_ASSERT(((int32_t *) PN->vir)[2] >= 0 && ((int32_t *) PN->vir)[2] < kv_pages * 2 * PG);
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
    if (man.has("lm_head.topk")) {
        head.bind_topk(man.get("lm_head.topk"));
    }
    const bool combined = head.bind_combined(man.get("lm_head"), logits->src[0]) && head.combined_ready(g, names);
    if (man.has("lm_head.verify")) {
        head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
    }
    if (combined) {
        head.run_combined(g, names, h, T, nullptr, nullptr);
    } else {
        head.run(g, names, h, T);
    }
    return GGML_STATUS_SUCCESS;
}

// the MTP block's kernels: its packs from blk.<il>.* (both kernels the same layout), its paged KV cache, the IO
void qwen35_whole::init_mtp(const std::map<std::string, ggml_tensor *> & t, int il) {
    axera_timer    tb("init: mtp");
    const kernel & k1 = man.get("mtp.t1");
    const kernel & kp = man.get("mtp.prompt");
    GGML_ASSERT(k1.desc->packs == kp.desc->packs && k1.out_bytes("kv") == kp.out_bytes("kv"));
    MKV = axera_cmm_alloc(k1.out_bytes("kv"));
    mtp_page_b = k1.out_bytes("kv") / NBLK;
    mtp_kv_row = (size_t) k1.out_bytes("kv") / NBLK / (2 * PG);
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
                GGML_ABORT("axera qwen35_whole: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(d.bytes);
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : k.desc->out) {
            axera_cmm * b = d.name == std::string("kv") ? &MKV : of(d.name);
            if (!b) {
                GGML_ABORT("axera qwen35_whole: %s: unexpected output %s\n", k.desc->role.c_str(), d.name.c_str());
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
    if (man.has("mtp.promptkv")) {
        const kernel & kpk = man.get("mtp.promptkv");
        bind(mtppkv, kpk, { { "e", &PKE }, { "h", &PKH }, { "cs", &PKCS }, { "kout", &PK }, { "vout", &PV } });
        const io_desc * ed = nullptr;
        for (const auto & d : kpk.desc->in) {
            if (d.name == "e") {
                ed = &d;
            }
        }
        GGML_ASSERT(ed && ed->shape.size() >= 2);
        const size_t P = (size_t) ed->shape[1];
        mtp_h_f32 = PKH->size == P * H * sizeof(float);
        mtp_e_f32 = PKE->size == P * H * sizeof(float);
        live.push_back(&mtppkv.in);
    }
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
ggml_status qwen35_whole::compute_mtp(ggml_cgraph * g) {
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
        GGML_LOG_ERROR("axera qwen35_whole: an MTP graph before the target model's or of an unexpected shape (embd %d h_nextn %d "
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
    mtp_cell = 0;
    for (int r = 0; r < T; ++r) {
        mtp_cell = std::max(mtp_cell, kv.cell_of(r) + 1);
    }
    auto row_bf16 = [&](const ggml_tensor * x, int r, ggml_bf16_t * d) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x->data + (size_t) r * x->nb[1]), d, H);
    };
    mtp_h_buf.resize((size_t) T * H);
    std::vector<float> & h = mtp_h_buf;
    std::vector<float> hr((size_t) T * H);  // the pre-head-norm rows, for the combined head
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
        for (int r = 0; r < 3 * R8; ++r) {  // (an index out of the kernel's range hangs the NPU: stop here instead)
            GGML_ASSERT(pn[r] >= 0 && pn[r] < 2 * man.context);
        }
        *(int32_t *) M8NT->vir = T;
        for (const axera_cmm * b : { &M8E, &M8H, &M8CS, &M8PN, &M8MASK, &M8NT }) {
            (*b)->flush();
        }
        mtp8.k->run(mtp8.in, mtp8.out, "mtp m8");
        M8OUT->invalidate();
        M8RAW->invalidate();
        for (int r = 0; r < T; ++r) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) M8OUT->vir + (size_t) r * H, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) M8RAW->vir + (size_t) r * H, hr.data() + (size_t) r * H, H);
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
            MRAW->invalidate();
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MOUT->vir, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MRAW->vir, hr.data() + (size_t) r * H, H);
        }
    } else {
        const bool kvonly = mtppkv.k != nullptr;  // all but the last row need only their K/V
        axera_cmm & pv_e  = kvonly ? PKE : PE;
        axera_cmm & pv_h  = kvonly ? PKH : PH;
        axera_cmm & pv_cs = kvonly ? PKCS : PCS;
        const size_t e_row = mtp_e_f32 ? sizeof(float) : sizeof(ggml_bf16_t);
        const int  P      = (int) (pv_e->size / (H * e_row));  // rows per call
        const int  n_rows = kvonly ? T - 1 : T;
        auto run_t1 = [&](int r) {
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
        };
        for (int it = 0; it < n_rows; it += P) {
            const int n = std::min(P, n_rows - it);
            if (n < P) {
                memset((char *) pv_e->vir + (size_t) n * H * e_row, 0, (size_t) (P - n) * H * e_row);
                memset((char *) pv_h->vir + (size_t) n * H * (mtp_h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)), 0,
                       (size_t) (P - n) * H * (mtp_h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)));
            }
            // mtp.promptkv has no pn input. Keep its cells in host memory instead of writing three
            // uncached CMM words per row and reading them back again for the K/V scatter.
            std::vector<int32_t> pn_host;
            if (kvonly) {
                pn_host.resize((size_t) 3 * P);
            }
            auto * pn = kvonly ? pn_host.data() : (int32_t *) PPN->vir;
            const int nmax = kvonly ? 0 : kv.chunk_mask((const int32_t *) pos->data, it, P, n, (ggml_bf16_t *) PMASK->vir);
            axera_host_pool().run((int64_t) 2 * n, [&](int64_t j) {
                const int rr = (int) (j % n);
                if (j < n) {
                    if (mtp_e_f32) {
                        memcpy((char *) pv_e->vir + (size_t) rr * H * sizeof(float),
                               (const char *) embd->data + (size_t) (it + rr) * embd->nb[1], H * sizeof(float));
                    } else {
                        row_bf16(embd, it + rr, (ggml_bf16_t *) pv_e->vir + (size_t) rr * H);
                    }
                } else {
                    if (mtp_h_f32) {
                        memcpy((char *) pv_h->vir + (size_t) rr * H * sizeof(float),
                               (const char *) hin->data + (size_t) (it + rr) * hin->nb[1], H * sizeof(float));
                    } else {
                        row_bf16(hin, it + rr, (ggml_bf16_t *) pv_h->vir + (size_t) rr * H);
                    }
                }
            });
            for (int r = 0; r < P; ++r) {
                const int     rr = std::min(r, n - 1);
                const int64_t c  = kv.cell_of(it + rr) + (r - rr);
                GGML_ASSERT(c < kv_cells);
                rp.rows(it + rr, 1, COS.get(), SIN.get());
                memcpy((char *) pv_cs->vir + r * 2 * half, COS->vir, half);
                memcpy((char *) pv_cs->vir + r * 2 * half + half, SIN->vir, half);
                pn[3 * r]     = (int32_t) c;
                pn[3 * r + 1] = (int32_t) (c / PG + 1);
                pn[3 * r + 2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
            }
            pv_e->flush(pv_e->vir, (size_t) n * H * e_row);
            pv_h->flush(pv_h->vir, (size_t) n * H * (mtp_h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)));
            pv_cs->flush(pv_cs->vir, (size_t) n * 2 * half);
            if (!kvonly) {
                PPN->flush();
            }
            if (kvonly) {
                mtppkv.k->run(mtppkv.in, mtppkv.out, "mtp prompt kv");
                PK->invalidate();
                PV->invalidate();
                int64_t pg0 = -1, pg1 = -1;
                for (int r = 0; r < n; ++r) {
                    const int64_t cell = pn[3 * r];
                    const int64_t page = cell / PG, cl = cell % PG;
                    memcpy((char *) MKV->vir + (size_t) ((page * 2 + 0) * PG + cl) * mtp_kv_row,
                           (char *) PK->vir + (size_t) r * mtp_kv_row, mtp_kv_row);
                    memcpy((char *) MKV->vir + (size_t) ((page * 2 + 1) * PG + cl) * mtp_kv_row,
                           (char *) PV->vir + (size_t) r * mtp_kv_row, mtp_kv_row);
                    pg0 = pg0 < 0 ? page : std::min(pg0, page);
                    pg1 = std::max(pg1, page);
                }
                if (pg1 >= 0) {
                    MKV->flush((char *) MKV->vir + (size_t) (pg0 * 2) * PG * mtp_kv_row,
                               (size_t) ((pg1 - pg0 + 1) * 2) * PG * mtp_kv_row);
                }
            } else {
                *(int32_t *) PNBLK->vir = nmax;
                PNBLK->flush();
                PMASK->flush(PMASK->vir, (size_t) nmax * P * PG * sizeof(ggml_bf16_t));
                mtpp.k->run(mtpp.in, mtpp.out, "mtp prompt");
                POUT2->invalidate();
                for (int r = 0; r < n; ++r) {
                    ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT2->vir + (size_t) r * H,
                                          h.data() + (size_t) (it + r) * H, H);
                }
            }
        }
        if (kvonly) {
            run_t1(T - 1);
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
        head.bind(man.get("lm_head"), logits->src[0]);
        if (man.has("lm_head.topk")) {
            head.bind_topk(man.get("lm_head.topk"));
        }
        if (T == 1 && man.has("lm_head.draft")) {  // a draft step: the draft vocabulary's head (draft_vocab.i32)
            if (!head.dk) {
                const std::string f  = man.dir + "/draft_vocab.i32";
                FILE *            fp = fopen(f.c_str(), "rb");
                GGML_ASSERT(fp);
                // the vocabulary is the file's, not the kernel's output shape: an argmax draft head has no full row
                fseek(fp, 0, SEEK_END);
                const long fsz = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                GGML_ASSERT(fsz > 0 && fsz % (long) sizeof(int32_t) == 0);
                std::vector<int32_t> ids((size_t) fsz / sizeof(int32_t));
                GGML_ASSERT(fread(ids.data(), sizeof(int32_t), ids.size(), fp) == ids.size());
                fclose(fp);
                axera_timer tb("init: draft head");
                head.bind_draft(man.get("lm_head.draft"), ids);
            }
            head.draft = true;
            const std::vector<char> normed(T, 1);  // (the draft kernel projects the MTP shared-head norm)
            head.run(g, names, h, T, &normed);
        } else {
            if (!head.bind_combined(man.get("lm_head"), logits->src[0])) {
                GGML_LOG_ERROR("axera qwen35_whole: the MTP head is not a combined lm_head\n");
                return GGML_STATUS_FAILED;
            }
            if (man.has("lm_head.verify")) {
                head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
            }
            // the draft context's catch-up rows are not host-sampled; publishing them can replace the target verify
            // rows when both graphs reuse the same host-visible logits buffer
            head.run_combined(g, names, hr, T, nullptr, nullptr, false);
        }
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_whole(const manifest & m) {
    return std::make_shared<qwen35_whole>(m);
}

}  // namespace axera
