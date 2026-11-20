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
#include "../../core/seqstate.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <regex>
#include <set>
#include <thread>

namespace axera {
namespace {

constexpr int PG = 256;           // cells per page = prefill rows per call = the decode kernels' KV block
constexpr int KH = 16, VH = 48;   // key / value heads of a delta-net layer
constexpr int REP = VH / KH;

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

// the adapter blob in llama.cpp's sequence state (context checkpoints / prompt cache). The recurrent state itself is
// core/seqstate's blob; this header only describes the KV pages and the MTP cache around it. AXW27S1 (version 2) is
// refused: its state layout is the old adapter-owned one.
struct state_header_27b {
    char     magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t has_target;
    uint32_t has_mtp;
    int64_t  state_cell;
    int64_t  mtp_cell;
    uint32_t ncall;
    uint32_t pna;
    uint32_t nkv_pages;
    uint64_t kv_page_b;
    uint64_t kvs_page_b;
    uint64_t kv_n;
    uint64_t kvs_n;
    uint64_t seq_n;
    uint64_t mtp_page0;
    uint64_t mtp_pages;
    uint64_t mtp_page_b;
    uint64_t mtp_n;
    uint32_t partial;
    uint32_t reserved;
};

// GGUF's tiled value head of HF (grouped) head g
inline int tiled(int g) {
    return (g % REP) * KH + g / REP;
}

class qwen38_27b final : public model, public seq_model {
  public:
    explicit qwen38_27b(const manifest & m) : man(m), prm(m) {}
    ~qwen38_27b() override { mtp_wait(); }
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size(uint32_t flags) const override;
    size_t       state_write(void * dst, size_t size, uint32_t flags) override;
    size_t       state_read(const void * src, size_t size, uint32_t flags) override;
    int          last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override;
    void     seq_state_load(const uint8_t * src) override;
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override;
    bool     seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) override;
    void     seq_verify_keep(int k) override;
    void     seq_verify_peek(int k, uint8_t * dst) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, G = 0, ROT = 0, NBLK = 0;
    std::vector<int> ord;  // layer -> its index among the delta-net layers / the attention layers

    // the state pools; their sizes per layer
    axera_cmm              KV, ST, CQ, CK, CV;
    std::vector<axera_cmm> KVC;  // the KV cache: one buffer per prefill call's attention layers (a CMM allocation is < 4 GB)
    std::vector<axera_cmm> KVSC;  // its scales (an int8 cache: a bf16 scale per row and head), the same split
    size_t    kv_b = 0, kvs_b = 0, st_b = 0, cq_b = 0, cv_b = 0;
    // the pages of the cache the pools hold (GGML_AXERA_KV_PAGES; default: the kernels' context): the kernels keep their
    // declared IO size but touch only the pages up to the runtime page count, so a smaller context needs only its pages
    int       kv_pages = 0;
    int64_t   kv_cells = 0;
    size_t    kv_alloc = 0, kvs_alloc = 0;  // the bytes of one layer's pool
    std::vector<call>              dec;
    int                            dec_result = 0;
    axera_cmm                      A, B, C, D, COS, SIN, CS, PN, MASK;
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
    bool    state_zero = true;         // the pools hold the zero state (CMM starts zero; a load sets it)
    int     ver_set = -1;              // the snapshot set of the pending verification
    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;
    size_t                    state_image_b = 0;
    int                       ver_rows = 0;
    // the current batch, for x == nullptr runs; the rows the kernel callbacks write the hidden states into
    const ggml_bf16_t * batch_rows = nullptr;
    const kv_view *     batch_kv = nullptr;
    const rope *        batch_rope = nullptr;
    float *             out_h = nullptr;
    std::vector<ggml_bf16_t> src_rows;
    // the MTP block (draft context graphs): its kernels' IO, its KV cache
    call      mtp1, mtpp, mtppkv, mtp8;  // mtp8: role mtp.m8 (optional), up to 8 consecutive rows per call
    axera_cmm ME, MH, MCS, MPN, MMASK, MOUT, MRAW, PE, PH, PCS, PPN, PMASK, PNBLK, POUT2, PRAW, MKV;
    axera_cmm PKE, PKH, PKCS;  // the compact prompt kernels use their own row width (not mtp.prompt's 256)
    axera_cmm PK, PV;  // the K/V-only prompt pass' plain row outputs (copied into the paged MKV)
    size_t    mtp_kv_row = 0;
    bool      mtp_h_f32 = false;  // the K/V prompt kernel takes FP32 h rows (the EF32 build)
    bool      mtp_e_f32 = false;  // the K/V prompt kernel takes FP32 token-embedding rows
    axera_cmm M8E, M8H, M8CS, M8PN, M8MASK, M8NT, M8OUT, M8RAW;
    const void * mtp_bound = nullptr;
    size_t       mtp_page_b = 0;  // bytes of one MTP KV page
    int64_t      mtp_cell  = 0;   // the next MTP KV cell
    // a verification's MTP catch-up (nothing on the host reads its result: it only extends the MTP block's KV) runs
    // on its own thread while the host goes on (sampling, the next draft's setup); any next graph waits for it
    mutable std::thread mtp_bg;
    void                mtp_wait() const {
        if (mtp_bg.joinable()) {
            mtp_bg.join();
        }
    }
    // the prefill kernel variants: 256 rows, and the 128 / 64-row tails that hold the last page of a prompt
    struct prefill_set {
        const kernel *    k = nullptr;
        int               rows = 0;
        std::vector<call> calls;  // the kernel's calls of a page (its blocks each)
        axera_cmm         X, POS, OUT[4];
        int               result = 0;  // the OUT[] buffer that holds the page's final rows
    };
    prefill_set                    pfs[5];  // the manifest's prefill variants, rows descending
    int                            ncall = 1;
    std::map<std::string, axera_cmm> banks;
    axera_cmm                      INDPTR, INDICES, LASTLEN, CLOC, VALID;
    output_head                    head;

    // The upstream fused dense path (sgl_kernel_axera.models.qwen38_27b.forward_qwen38_blocks): one prefill kernel and
    // one decode kernel carry whole 4-layer blocks in banks [KB, ...]. NB = L / 4 blocks in the model, KB = the blocks
    // one kernel call owns, NC = NB / KB calls per row batch. The 2B/4B manifests have no decode.group; the 27B
    // manifest keeps it and runs the older per-group path untouched.
    bool   fused = false;
    bool   nv4   = false;
    int    NB = 0, KB = 0, NC = 1;
    int    VH = 0, KHd = 0, KD = 0, VD = 0, REPf = 0;  // delta-net value/key heads and their dimensions
    size_t kv_blk_b = 0;                              // bytes of one block's KV pool (all pages)

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_fused(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_fused_phases(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_mtp(const std::map<std::string, ggml_tensor *> & t, int il);
    size_t part(int kind) const { return kind == 0 ? st_b : kind == 3 ? cv_b : cq_b; }
    const axera_cmm & pool(int kind) const { return kind == 0 ? ST : kind == 1 ? CQ : kind == 2 ? CK : CV; }
    void bind_state();
    void state_home();
    void state_at(int set, int row);
    ggml_status verify(int64_t c0, int row0, int T, const ggml_bf16_t * xb, const int32_t * pb, float * h);
    ggml_status compute_mtp(ggml_cgraph * g);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    axera_cmm bank_fused(const std::map<std::string, ggml_tensor *> & t, const io_desc & d);
    int       vtiled(int g) const { return (g % REPf) * KHd + g / REPf; }
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                      seq_snap * snap);
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
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(d.shape.size() >= 2 && nb > 0 && (nb == L / 4 || n == "cos" || n == "sin") && d.bytes % nb == 0);
    axera_cmm     c  = axera_cmm_alloc_bank(d.bytes);
    const size_t  bb = d.bytes / nb;  // bytes per block
    const bool keep_pages = axera_keep_pages();
    std::vector<const ggml_tensor *> read;  // the GGUF tensors of one block: their mapped pages dropped right after it
    auto          T  = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen38_27b: %s not in the graph\n", nm.c_str());
        }
        axera_page_in(t.at(nm));
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        axera_mapped_drop_guard drop(read, !keep_pages);
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
                if (b > 0) {  // the same table in every block: block 0's, copied
                    memcpy(dst, c->vir, bb);
                } else {  // (a 262144-position table is 8 M cos or sin: one row per pool item, not one thread for all)
                    const bool co = n == "cos";
                    GGML_ASSERT((size_t) (P * R) * sizeof(float) == bb);
                    axera_host_pool().run(P, [&](int64_t q) {
                        for (int64_t i = 0; i < R; ++i) {
                            const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                            ((float *) dst)[q * R + i] = (float) (co ? std::cos(th) : std::sin(th));
                        }
                    });
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
    if (keep_pages) {
        for (const ggml_tensor * w : read) {
            axera_drop_mapped(w);
        }
    }
    return c;
}

// Fill one fused upstream bank for all NB blocks: block b's four layers are 4*b + phase. Every source tensor is
// read straight from the GGUF; the delta-net rows and columns the converter tiled are put back in checkpoint order.
axera_cmm qwen38_27b::bank_fused(const std::map<std::string, ggml_tensor *> & t, const io_desc & d) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera qwen38_27b: fused: unexpected input %s\n", d.name.c_str());
    }
    const int         phase  = std::stoi(mt[1]);
    const bool        mlp    = mt[2] == "mlp";
    const std::string n      = mt[3];
    const bool        shared = d.name == "b3_attn_cos_sin_cache";
    const size_t      bb     = d.bytes / (shared ? 1 : (size_t) KB);
    axera_cmm         c      = axera_cmm_alloc(bb * (size_t) (shared ? 1 : NB));
    const size_t      stride = bb;
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen38_27b: fused: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    auto scale2_of = [&](const ggml_tensor * w, int64_t row) -> float {
        std::string nm = w->name;
        const std::string suf = ".weight";
        if (nm.size() >= suf.size() && nm.compare(nm.size() - suf.size(), suf.size(), suf) == 0) {
            nm.resize(nm.size() - suf.size());
        }
        nm += ".scale";
        auto it = t.find(nm);
        if (it == t.end() || it->second->type != GGML_TYPE_F32) {
            GGML_ABORT("axera qwen38_27b: fused: NVFP4 tensor %s has no F32 .scale sidecar\n", w->name);
        }
        const int64_t ne = ggml_nelements(it->second);
        GGML_ASSERT(ne == 1 || (row >= 0 && row < ne));
        return ((const float *) it->second->data)[ne == 1 ? 0 : row];
    };
    for (int b = 0; b < (shared ? 1 : NB); ++b) {
        const int il  = 4 * b + phase;
        uint8_t * dst = (uint8_t *) c->vir + (size_t) b * stride;
        auto put = [&](const std::vector<float> & v, float mul = 1.0f) {
            if (v.size() * (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t)) != bb) {
                GGML_ABORT("axera qwen38_27b: fused: bank %s: %zu values of dtype %d, kernel block wants %zu bytes\n",
                           d.name.c_str(), v.size(), d.dtype, bb);
            }
            for (size_t i = 0; i < v.size(); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v[i] * mul;
                } else {
                    GGML_ASSERT(d.dtype == 9);
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i] * mul);
                }
            }
        };
        auto zero = [&]() { memset(dst, 0, bb); };
        auto scalar = [&](float v) {
            for (size_t i = 0; i < bb / (d.dtype == 3 ? 4 : 2); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v);
                }
            }
        };
        auto nv = [&](const ggml_tensor * w, int64_t s0, int64_t nr, int64_t row0, bool codes, bool scales, int64_t K) {
            thread_local std::vector<uint8_t> tq, ts;
            uint8_t * qp = codes ? dst + (size_t) row0 * (K / 2) : nullptr;
            uint8_t * sp = scales ? dst + (size_t) row0 * (K / 16) : nullptr;
            if (!codes) {
                tq.resize((size_t) nr * (K / 2));
                qp = tq.data();
            }
            if (!scales) {
                ts.resize((size_t) nr * (K / 16));
                sp = ts.data();
            }
            axera_copy_nvfp4_slice(w, s0, nr, qp, K / 2, sp, K / 16);
        };
        auto nv_rows = [&](const ggml_tensor * w, int64_t row0, int64_t nr, bool codes, bool scales, int64_t K,
                           int64_t hr, int64_t vrow) {
            GGML_ASSERT(hr > 0 && nr % hr == 0);
            for (int64_t j = 0; j < nr; j += hr) {
                nv(w, vrow + (int64_t) vtiled((int) (j / hr)) * hr, hr, row0 + j, codes, scales, K);
            }
        };
        auto nv_cols = [&](const ggml_tensor * w, int64_t row0, int64_t nr, bool codes, bool scales, int64_t K) {
            thread_local std::vector<uint8_t> tq, ts;
            tq.resize((size_t) K / 2);
            ts.resize((size_t) K / 16);
            const size_t cq = (size_t) (K / 2) / VH, cs = (size_t) (K / 16) / VH;
            GGML_ASSERT((size_t) VH * cq == (size_t) (K / 2) && (size_t) VH * cs == (size_t) (K / 16));
            for (int64_t r = 0; r < nr; ++r) {
                axera_copy_nvfp4_slice(w, row0 + r, 1, tq.data(), K / 2, ts.data(), K / 16);
                for (int g = 0; g < VH; ++g) {
                    if (codes) {
                        memcpy(dst + (size_t) (row0 + r) * (K / 2) + (size_t) g * cq, tq.data() + vtiled(g) * cq, cq);
                    }
                    if (scales) {
                        memcpy(dst + (size_t) (row0 + r) * (K / 16) + (size_t) g * cs, ts.data() + vtiled(g) * cs, cs);
                    }
                }
            }
        };
        const bool codes  = d.dtype == 15;
        const bool scales = d.dtype == 11;
        GGML_ASSERT(codes || scales || d.dtype == 3 || d.dtype == 9 || d.dtype == 6);
        if (mlp) {
            if (n == "w_norm") {
                put(f32_of(T(il, "post_attention_norm.weight")));
            } else if (n == "input_permutation") {
                zero();
            } else if (n == "w1" || n == "w_scale1") {
                const ggml_tensor * w = T(il, "ffn_gate.weight");
                nv(w, 0, w->ne[1], 0, codes, scales, w->ne[0]);
            } else if (n == "w2" || n == "w_scale2") {
                const ggml_tensor * w = T(il, "ffn_up.weight");
                nv(w, 0, w->ne[1], 0, codes, scales, w->ne[0]);
            } else if (n == "w3" || n == "w_scale3") {
                const ggml_tensor * w = T(il, "ffn_down.weight");
                nv(w, 0, w->ne[1], 0, codes, scales, w->ne[0]);
            } else if (n.rfind("w_scale2_", 0) == 0) {
                const std::string p = n.substr(9);
                const char *      g = p == "1" ? "ffn_gate.weight" : p == "2" ? "ffn_up.weight" : "ffn_down.weight";
                scalar(scale2_of(T(il, g), 0));
            } else if (n.rfind("w_zp_scale", 0) == 0) {
                zero();
            } else {
                GGML_ABORT("axera qwen38_27b: fused: unexpected MLP input %s\n", d.name.c_str());
            }
        } else if (phase == 3) {
            const std::string p = n.size() > 2 ? n.substr(n.rfind('_') + 1) : n;
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (n == "w_norm_q") {
                put(f32_of(T(il, "attn_q_norm.weight")), 1.0f / std::sqrt((float) d.shape.back()));
            } else if (n == "w_norm_k") {
                put(f32_of(T(il, "attn_k_norm.weight")));
            } else if (n == "cos_sin_cache") {
                const int64_t P = d.shape.size() >= 3 ? d.shape[1] : 0;
                GGML_ASSERT(P > 0 && d.shape.back() == 64 && bb == (size_t) P * 64 * sizeof(ggml_bf16_t));
                std::vector<float> inv(32);
                for (int i = 0; i < 32; ++i) {
                    inv[i] = 1.0f / std::pow(10000000.0f, (float) (2 * i) / 64);
                }
                ggml_bf16_t * o = (ggml_bf16_t *) dst;
                for (int64_t q = 0; q < P; ++q) {
                    for (int i = 0; i < 32; ++i) {
                        const float th = (float) q * inv[i];
                        o[q * 64 + i]      = ggml_fp32_to_bf16(std::cos(th));
                        o[q * 64 + 32 + i] = ggml_fp32_to_bf16(std::sin(th));
                    }
                }
            } else if ((n.size() == 3 && n[0] == 'w' && n[1] == '_' &&
                        (n[2] == 'q' || n[2] == 'k' || n[2] == 'v' || n[2] == 'o')) ||
                       n.rfind("w_scale_", 0) == 0 || n.rfind("w_scale2_", 0) == 0 ||
                       n.rfind("w_zeropoint_", 0) == 0) {
                static const std::map<char, const char *> W = { { 'q', "attn_q.weight" }, { 'k', "attn_k.weight" },
                                                                { 'v', "attn_v.weight" }, { 'o', "attn_output.weight" } };
                if (W.count(n[2])) {
                    const ggml_tensor * w = T(il, W.at(n[2]));
                    nv(w, 0, w->ne[1], 0, codes, scales, w->ne[0]);
                } else if (n == "w_scale_q" || n == "w_scale_k" || n == "w_scale_v" || n == "w_scale_o") {
                    const ggml_tensor * w = T(il, W.at(n[8]));
                    nv(w, 0, w->ne[1], 0, codes, scales, w->ne[0]);
                } else if (n.rfind("w_scale2_", 0) == 0 && n.size() == 10) {
                    const ggml_tensor * w = T(il, W.at(n[9]));
                    scalar(scale2_of(w, 0));
                } else if (n.rfind("w_zeropoint_", 0) == 0) {
                    zero();
                } else {
                    GGML_ABORT("axera qwen38_27b: fused: unexpected attention input %s\n", d.name.c_str());
                }
            } else {
                GGML_ABORT("axera qwen38_27b: fused: unexpected attention input %s\n", d.name.c_str());
            }
        } else {
            if (n == "input_norm_w") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (n == "o_norm_w") {
                put(f32_of(T(il, "ssm_norm.weight")));
            } else if (n == "a_log") {
                const std::vector<float> a = f32_of(T(il, "ssm_a"));
                std::vector<float>       u(VH);
                for (int h = 0; h < VH; ++h) {
                    u[h] = std::log(-a[(size_t) vtiled(h)]);
                }
                put(u);
            } else if (n == "dt_bias") {
                const std::vector<float> v = f32_of(T(il, "ssm_dt.bias"));
                std::vector<float>       u(VH);
                for (int h = 0; h < VH; ++h) {
                    u[h] = v[(size_t) vtiled(h)];
                }
                put(u);
            } else if (n == "w_q_conv" || n == "w_k_conv" || n == "w_v_conv") {
                const ggml_tensor *      w = T(il, "ssm_conv1d.weight");
                const std::vector<float> v = f32_of(w);
                const int64_t            tp = w->ne[0], C = d.shape[1] * d.shape[2];
                const int64_t            v0 = w->ne[1] - C;
                const int64_t            c0 = n[2] == 'q' ? 0 : n[2] == 'k' ? C : v0;
                GGML_ASSERT(d.shape.size() == 4 && d.shape[2] == 1 && d.shape[3] == tp && C + c0 <= w->ne[1]);
                std::vector<float> u((size_t) C * tp);
                for (int64_t r = 0; r < C; ++r) {
                    const int64_t hr  = C / VH;
                    const int64_t src = n[2] == 'v' ? v0 + (int64_t) vtiled((int) (r / hr)) * hr + r % hr : c0 + r;
                    memcpy(&u[(size_t) r * tp], &v[(size_t) src * tp], tp * sizeof(float));
                }
                put(u);
            } else if (n == "w_a_scale2" || n == "w_b_scale2" || n == "w_o_scale2") {
                const char * g = n[2] == 'a' ? "ssm_alpha.weight" : n[2] == 'b' ? "ssm_beta.weight" : "ssm_out.weight";
                scalar(scale2_of(T(il, g), 0));
            } else if (n == "w_q_scale2" || n == "w_k_scale2" || n == "w_v_scale2" || n == "w_z_scale2") {
                const ggml_tensor * qkv = T(il, "attn_qkv.weight");
                const int64_t        kh = (int64_t) KHd * KD;
                const char           p  = n[2];
                scalar(p == 'z' ? scale2_of(T(il, "attn_gate.weight"), 0)
                                : scale2_of(qkv, p == 'q' ? 0 : p == 'k' ? kh : 2 * kh));
            } else if (n == "w_q" || n == "w_q_scale" || n == "w_k" || n == "w_k_scale" || n == "w_v" ||
                       n == "w_v_scale" || n == "w_z" || n == "w_z_scale") {
                const ggml_tensor * qkv = T(il, "attn_qkv.weight");
                const int64_t        K   = qkv->ne[0];
                const int64_t        kh  = (int64_t) KHd * KD;
                const int64_t        vh  = qkv->ne[1] - 2 * kh;
                GGML_ASSERT(kh * 2 + vh == qkv->ne[1] && vh % VH == 0);
                const bool  sc = n.size() > 6 && n.compare(n.size() - 6, 6, "_scale") == 0;
                const char  p  = n[2];
                if (p == 'z') {
                    const ggml_tensor * z = T(il, "attn_gate.weight");
                    GGML_ASSERT(z->ne[1] % VH == 0);
                    nv_rows(z, 0, z->ne[1], codes, scales, K, z->ne[1] / VH, 0);
                } else if (p == 'v') {
                    nv_rows(qkv, 0, vh, codes, scales, K, vh / VH, 2 * kh);
                } else {
                    nv(qkv, p == 'q' ? 0 : kh, kh, 0, codes, scales, K);
                }
            } else if (n == "w_a" || n == "w_a_scale" || n == "w_b" || n == "w_b_scale") {
                const ggml_tensor * w = T(il, n[2] == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight");
                nv_rows(w, 0, w->ne[1], codes, scales, w->ne[0], 1, 0);
            } else if (n == "w_o" || n == "w_o_scale") {
                const ggml_tensor * w = T(il, "ssm_out.weight");
                nv_cols(w, 0, w->ne[1], codes, scales, w->ne[0]);
            } else {
                GGML_ABORT("axera qwen38_27b: fused: unexpected delta-net input %s\n", d.name.c_str());
            }
        }
    }
    c->flush();
    for (const ggml_tensor * w : read) {
        axera_drop_mapped(w);
    }
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
    // A split prefill kernel binds one KV pool per block (b3_kv<block>): every
    // block's K+V is then one IO, below the runtime's 2 GiB IO limit, and a
    // single call can run the whole model.
    auto split_kv_index = [](const std::string & n) -> int {
        if (n.rfind("b3_kv", 0) != 0 || n.size() <= 5) {
            return -1;
        }
        for (size_t i = 5; i < n.size(); ++i) {
            if (!isdigit((unsigned char) n[i])) {
                return -1;
            }
        }
        return atoi(n.c_str() + 5);
    };
    bool split_kv = false;
    for (const auto & d : pk.desc->in) {
        split_kv = split_kv || split_kv_index(d.name) >= 0;
    }
    const io_desc & pkv = in_of(pk, split_kv ? "b3_kv0" : "b3_kv"), &pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in"), &pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const io_desc * pkvs = nullptr;  // an int8 cache: its scales
    for (const auto & d : pk.desc->in) {
        if (d.name == "b3_attn_kv_scale") {
            pkvs = &d;
        }
    }
    const int png = (int) pst.shape[0];  // GDN layers the prefill banks cover (3 per block)
    const int pna = split_kv ? png / 3 : (int) pkv.shape[0];  // the prefill kernel's blocks per call
    GGML_ASSERT(pna > 0 && na % pna == 0 && png == 3 * pna && pcq.shape[0] == png && pcv.shape[0] == png);
    ncall = na / pna;
    const int kv_layers = split_kv ? 1 : pna;  // each b3_kv<block> IO is one layer, a b3_kv bank is pna layers
    kv_b  = pkv.bytes / kv_layers;
    kvs_b = pkvs ? pkvs->bytes / kv_layers : 0;
    st_b  = pst.bytes / png;
    cq_b  = pcq.bytes / png;
    cv_b  = pcv.bytes / png;
    GGML_ASSERT(in_of(dk, "kv3").bytes == kv_b && in_of(dk, "st0").bytes == st_b && in_of(dk, "cq0").bytes == cq_b &&
                in_of(dk, "ck0").bytes == cq_b && in_of(dk, "cv0").bytes == cv_b);
    NBLK = (int) pkv.shape[split_kv ? 0 : 1];
    GGML_ASSERT(pkv.shape[split_kv ? 2 : 3] == PG && (int64_t) NBLK * PG == man.context);
    kv_pages = NBLK;
    if (const int64_t cells = axera_kv_cells(); cells > 0 && (split_kv || pna == 1)) {  // llama.cpp's -c (its own cache's cells); a
        kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(NBLK, (cells + PG - 1) / PG));  // multi-layer prefill
    }                                                                                            // call needs every page
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        kv_pages = std::max(1, std::min(NBLK, atoi(e)));
    }
    GGML_ASSERT(kv_pages == NBLK || pna == 1 || split_kv);  // (a multi-layer bank needs fixed layer strides in its IO)
    kv_cells  = (int64_t) kv_pages * PG;
    kv_alloc  = kv_b / NBLK * kv_pages;
    kvs_alloc = kvs_b / NBLK * kv_pages;
    if (kv_pages < NBLK) {
        GGML_LOG_INFO("axera qwen38_27b: the KV pools hold %d of the kernels' %d pages (%lld cells)\n", kv_pages, NBLK, (long long) kv_cells);
    }
    const int npool = split_kv || pna == 1 ? na : ncall;  // one pool per attention layer, or one per prefill call
    KVC.resize(npool);
    KVSC.resize(kvs_b ? npool : 0);
    for (int k = 0; k < npool; ++k) {
        KVC[k] = axera_cmm_alloc(kv_alloc * (split_kv ? 1 : pna));
        if (kvs_b) {
            KVSC[k] = axera_cmm_alloc(kvs_alloc * (split_kv ? 1 : pna));
        }
    }
    for (const auto & d : dk.desc->out) {  // the decode kernels' cache IO matches the prefill's (kv<j>, kvs<j>)
        if (d.name.rfind("kvs", 0) == 0) {
            GGML_ASSERT(kvs_b && d.bytes == kvs_b);
        } else if (d.name.rfind("kv", 0) == 0) {
            GGML_ASSERT(d.bytes == kv_b);
        }
    }
    ST = axera_cmm_alloc(st_b * ng);
    CQ = axera_cmm_alloc(cq_b * ng);
    CK = axera_cmm_alloc(cq_b * ng);
    CV = axera_cmm_alloc(cv_b * ng);
    state_image_b = ST->size + CQ->size + CK->size + CV->size;
    GGML_ASSERT(state_image_b == st_b * ng + 2 * cq_b * ng + cv_b * ng);

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
    const int kl = split_kv ? 1 : pna;  // layers per KV pool (a split pool is one block's)
    auto pool_io = [&](const std::string & n, int il) {
        if (n.rfind("kvs", 0) == 0) {
            const int a = ord[il];
            GGML_ASSERT(kvs_b);
            return axera_io(KVSC[a / kl].get(), (char *) KVSC[a / kl]->vir + (size_t) (a % kl) * kvs_b, (uint32_t) kvs_b);
        }
        if (n.rfind("kv", 0) == 0) {
            const int a = ord[il];
            return axera_io(KVC[a / kl].get(), (char *) KVC[a / kl]->vir + (size_t) (a % kl) * kv_b, (uint32_t) kv_b);
        }
        const axera_cmm & p = n[0] == 's' ? ST : n[1] == 'q' ? CQ : n[1] == 'k' ? CK : CV;
        const size_t      b = n[0] == 's' ? st_b : n[1] == 'v' ? cv_b : cq_b;
        return axera_io(p.get(), (char *) p->vir + ord[il] * b, (uint32_t) b);
    };
    // the prefill kernel's weight banks [blocks, ..] from the GGUF (first: decode and verification calls may read
    // their row-split weights from them, inputs k_<bank>: the 4 blocks of the call's 16 layers)
    const rope rp(rope_node, names.find("inp_pos"), 1);
    banks.clear();
    {
        axera_timer tb("init: prefill banks");
        static const std::set<std::string> pools = { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                                                     "valid_tokens", "b3_kv", "b3_attn_kv_scale", "b0_attn_state_in", "b0_attn_q_conv_state_in",
                                                     "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" };
        for (const auto & d : pk.desc->in) {
            if (!pools.count(d.name) && split_kv_index(d.name) < 0) {
                // the weights of all L / 4 blocks (a call's slice of them bound below); the cos / sin tables are the
                // same for every block: the kernel's blocks of them, bound whole to every call
                const bool cs = d.name.size() > 4 && (d.name.compare(d.name.size() - 4, 4, "_cos") == 0 ||
                                                      d.name.compare(d.name.size() - 4, 4, "_sin") == 0);
                io_desc dd = d;
                if (!cs && ncall > 1) {
                    dd.shape[0] *= ncall;
                    dd.bytes *= ncall;
                }
                banks[d.name] = bank(t, dd, rp);
            }
        }
        axera_bank_sum("qwen38_27b", banks);
    }
    auto bank_io = [&](const kernel & k, const io_desc & d, int gi) -> AX_ENGINE_IO_BUFFER_T {
        auto it = banks.find(d.name.substr(2));
        if (d.name.rfind("k_", 0) != 0 || it == banks.end()) {
            GGML_ABORT("axera qwen38_27b: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
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
                GGML_ABORT("axera qwen38_27b: decode.group: unexpected input %s\n", d.name.c_str());
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : dk.desc->out) {
            c.out.push_back(d.name == "out" ? axera_io(A, d.bytes) : pool_io(d.name, gi * G + atoi(d.name.c_str() + strspn(d.name.c_str(), "abcdefghijklmnopqrstuvwxyz"))));
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
                                                   pool_io(d.name, gi * G + atoi(d.name.c_str() + strspn(d.name.c_str(), "abcdefghijklmnopqrstuvwxyz"))));
            }
            state_outs(c, gi * G);
        }
    }

    // prefill: the weight banks (above), the pools, one page's inputs; three row variants share the banks and pools
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    GGML_ASSERT(pk.in_bytes("indices") == (size_t) NBLK * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();
    auto bind_prefill = [&](int idx, const char * role) {
        if (!man.has(role)) {
            return;
        }
        prefill_set &     p = pfs[idx];
        const kernel &    k = man.get(role);
        const int         rows = (int) (k.in_bytes("x") / (H * sizeof(ggml_bf16_t)));
        GGML_ASSERT(rows >= 2 && rows <= PG);
        p.k                 = &k;
        p.rows              = rows;
        p.X                 = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.OUT[0]            = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.OUT[1]            = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.POS               = axera_cmm_alloc(k.in_bytes("position_ids"));
        GGML_ASSERT(k.in_bytes("x") == p.X->size && k.in_bytes("position_ids") == 3 * (size_t) rows * sizeof(int32_t));
        GGML_ASSERT(k.desc->out.size() == 1 && k.desc->out[0].name == "output" && k.desc->out[0].bytes == p.OUT[0]->size);
        const io_desc & kv = in_of(k, split_kv ? "b3_kv0" : "b3_kv");
        const io_desc & st = in_of(k, "b0_attn_state_in");
        const int       pna_k = split_kv ? (int) st.shape[0] / 3 : (int) kv.shape[0];
        GGML_ASSERT(pna_k == pna && na % pna_k == 0);
        const int nc = na / pna_k;
        GGML_ASSERT(kv.bytes == kv_b && st.bytes == pst.bytes);
        p.calls.assign(nc, {});
        for (int c_i = 0; c_i < nc; ++c_i) {  // call c_i: blocks c_i * pna .. of the banks, its layers of the pools
            call & c = p.calls[c_i];
            c.k      = &k;
            for (const auto & d : k.desc->in) {
                const std::string & n = d.name;
                const int           ki = split_kv_index(n);
                const axera_cmm *   b = n == "x" ? (c_i == 0 ? &p.X : &p.OUT[(c_i - 1) & 1]) : n == "position_ids" ? &p.POS :
                                        n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                        n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID : nullptr;
                size_t off = 0;
                if (!b && (n == "b3_kv" || ki >= 0)) {
                    b = &KVC[ki >= 0 ? c_i * pna + ki : c_i];
                } else if (!b && n == "b3_attn_kv_scale") {
                    b = &KVSC[c_i];
                } else if (!b) {
                    b = n == "b0_attn_state_in" ? &ST : n == "b0_attn_q_conv_state_in" ? &CQ :
                        n == "b0_attn_k_conv_state_in" ? &CK : n == "b0_attn_v_conv_state_in" ? &CV : nullptr;
                    if (b) {
                        off = (size_t) c_i * d.bytes;  // (the pools: the prefill order of the layers, kind by kind)
                    } else {
                        b   = &banks.at(n);
                        off = (*b)->size >= (size_t) nc * d.bytes ? (size_t) c_i * d.bytes : 0;  // (cos / sin: whole)
                    }
                }
                const bool kvio = n == "b3_kv" || ki >= 0;
                GGML_ASSERT((*b)->size >= off + (kvio ? kv_alloc : n == "b3_attn_kv_scale" ? kvs_alloc : d.bytes));
                c.in.push_back(axera_io(b->get(), (char *) (*b)->vir + off, d.bytes));
            }
            c.out.push_back(axera_io(p.OUT[c_i & 1], k.desc->out[0].bytes));
        }
    };
    // the manifests carry the 256-row prefill and any tail variants (128 / 96 / 80 / 64 rows); the rows are the
    // kernel's x input, not a naming convention, and a missing role is simply not a candidate
    bind_prefill(0, "prefill");
    bind_prefill(1, "prefill128");
    bind_prefill(2, "prefill96");
    bind_prefill(3, "prefill80");
    bind_prefill(4, "prefill64");

    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
    ver_rows = man.has("decode.verify")
                   ? (int) (man.get("decode.verify").in_bytes("x") / (H * sizeof(ggml_bf16_t)))
                   : 0;
    seq_config cfg;
    cfg.kept_max = 1;
    // The 27B verify kernel materializes one 4-row snapshot per call; chaining several calls for a 5..32-row tail
    // changes the numerics (and the kernel count) of a path that used to be one prefill page, which flips greedy
    // continuations in long degraded chats. Keep the pre-migration path selection by default; the environment
    // variable still opts into a longer chain.
    cfg.verify_chain = [vr = ver_rows] {
        const char * e = getenv("GGML_AXERA_VERIFY_CHAIN");
        return e ? atoi(e) : vr;
    }();
    cfg.check = [] {
        const char * e = getenv("GGML_AXERA_SEQ_CHECK");
        return e && atoi(e) != 0;
    }();
    cfg.force_miss = [] {
        const char * e = getenv("GGML_AXERA_SEQ_FORCE_MISS");
        return e ? std::max(0, atoi(e)) : 0;
    }();
    seq = std::make_unique<seqstate>(*this, cfg);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen38_27b: %d layers (%d per decode call), prefill banks %.1f GiB, state %.2f GiB, %lld cells\n",
                  L, G, bb / 1073741824.0, ((kv_b + kvs_b) * na + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}

// The upstream fused dense path: one prefill kernel (256 rows) and one decode kernel (1 row), their banks carrying
// KB whole 4-layer blocks. The 2B/4B manifests use this path; the 27B manifest has decode.group and keeps init().
void qwen38_27b::init_fused(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L    = L_;
    H    = H_;
    fused = true;
    nv4   = true;
    GGML_ASSERT(L % 4 == 0);
    NB = L / 4;
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");
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
    // This builder registers the projections separately (w_q, w_k, w_v, w_z, w_a, w_b); the 27B sgl manifests fuse
    // them into w_k/w_a. The two layouts are filled by different bank mappers, chosen from the compiled IO.
    if (pk.find_input("b0_attn_w_q") < 0) {
        GGML_ABORT("axera qwen38_27b: fused prefill needs the per-projection delta-net banks\n");
    }
    const io_desc & pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in");
    const io_desc & pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const io_desc & pkv = in_of(pk, "b3_kv");
    KB = (int) in_of(pk, "b0_mlp_w_norm").shape[0];
    GGML_ASSERT(KB > 0 && NB % KB == 0);
    NC = NB / KB;
    GGML_ASSERT(pst.shape.size() == 5 && pst.bytes % (size_t) KB == 0);
    VH  = (int) pst.shape[2];
    KD  = (int) pst.shape[3];
    VD  = (int) pst.shape[4];
    st_b = pst.bytes / (size_t) KB;
    cq_b = pcq.bytes / (size_t) KB;
    cv_b = pcv.bytes / (size_t) KB;
    const io_desc & pq = in_of(pk, "b0_attn_w_q");
    GGML_ASSERT(pq.shape.size() == 3 && pq.shape[1] % KD == 0);
    KHd  = (int) (pq.shape[1] / KD);
    REPf = VH / KHd;
    GGML_ASSERT(KHd > 0 && REPf > 0 && VH % KHd == 0);

    GGML_ASSERT(pkv.shape.size() >= 5 && pkv.bytes % (size_t) KB == 0);
    const size_t kd = pkv.shape.size();
    GGML_ASSERT(kd >= 6 && pkv.shape[kd - 6] == KB && pkv.shape[kd - 4] == 2 && pkv.shape[kd - 3] == PG &&
                pkv.shape[kd - 1] == 256);
    kv_blk_b = pkv.bytes / (size_t) KB;
    // [..][KB][pages][2][PG][KV heads][256]
    kv_pages = (int) pkv.shape[kd - 5];
    NBLK = kv_pages;
    GGML_ASSERT((int64_t) NBLK * PG >= (int64_t) man.context);
    // A one-block bank has a fixed page stride and can hold only the pages the
    // requested context needs, like the released 27B prefill.  A multi-block
    // bank needs every page to keep the block stride in the compiled IO.
    if (const int64_t cells = axera_kv_cells(); cells > 0 && KB == 1) {
        kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(NBLK, (cells + PG - 1) / PG));
    }
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        kv_pages = std::max(1, std::min(NBLK, atoi(e)));
    }
    GGML_ASSERT(kv_pages == NBLK || KB == 1);
    kv_cells = (int64_t) kv_pages * PG;
    ROT  = 64;

    kv_b      = pkv.bytes;  // one call's KV bank (KB blocks)
    kv_alloc  = kv_b / (size_t) NBLK * (size_t) kv_pages;
    kvs_b     = 0;
    kvs_alloc = 0;
    KVC.resize(NC);
    for (int c = 0; c < NC; ++c) {
        KVC[c] = axera_cmm_alloc(kv_alloc);
    }
    KVSC.clear();
    ST = axera_cmm_alloc(st_b * (size_t) NB * 3);
    CQ = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CK = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CV = axera_cmm_alloc(cv_b * (size_t) NB * 3);
    state_image_b = ST->size + CQ->size + CK->size + CV->size;

    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(std::max(pk.in_bytes("last_page_len"), dk.in_bytes("last_page_len")));
    GGML_ASSERT(LASTLEN->size >= 2 * sizeof(int32_t));
    CLOC  = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID = axera_cmm_alloc(sizeof(int32_t));
    const int n_indices = (int) (pk.in_bytes("indices") / sizeof(int32_t));
    GGML_ASSERT(n_indices >= (int) ((man.context + PG - 1) / PG) && n_indices <= kv_pages);
    for (int i = 0; i < n_indices; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    A = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    B = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    C = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    D = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    banks.clear();
    {
        axera_timer tb("init: fused banks");
        for (const auto & d : pk.desc->in) {
            static const std::string rt[] = { "_state_in", "_state_out", "_conv_state_in", "_conv_state_out" };
            bool runtime = d.name == "x" || d.name == "position_ids" || d.name == "indptr" || d.name == "indices" ||
                           d.name == "last_page_len" || d.name == "cache_location" || d.name == "valid_tokens" ||
                           d.name == "b3_kv";
            for (const auto & suf : rt) {
                runtime = runtime || (d.name.size() > suf.size() &&
                                      d.name.compare(d.name.size() - suf.size(), suf.size(), suf) == 0);
            }
            if (!runtime) {
                banks[d.name] = bank_fused(t, d);
            }
        }
    }

    auto state_io = [&](const std::string & n, int ci) -> AX_ENGINE_IO_BUFFER_T {
        const int ph = n[1] - '0';
        const size_t per = n.find("q_conv_state") != std::string::npos ? cq_b :
                           n.find("k_conv_state") != std::string::npos ? cq_b :
                           n.find("v_conv_state") != std::string::npos ? cv_b : st_b;
        const axera_cmm * p = n.find("q_conv_state") != std::string::npos ? &CQ :
                              n.find("k_conv_state") != std::string::npos ? &CK :
                              n.find("v_conv_state") != std::string::npos ? &CV : &ST;
        GGML_ASSERT(ph >= 0 && ph < 3);
        const size_t off = ((size_t) ph * NB + (size_t) ci * KB) * per;
        return axera_io(p->get(), (char *) (*p)->vir + off, per * (size_t) KB);
    };
    auto bind = [&](const kernel & k, std::vector<call> & calls, bool decode) {
        calls.assign(NC, {});
        for (int ci = 0; ci < NC; ++ci) {
            call & c = calls[ci];
            c.k      = &k;
            for (const auto & d : k.desc->in) {
                const std::string & n = d.name;
                if (n == "x") {
                    c.in.push_back(axera_io(decode ? A : (ci == 0 ? pfs[0].X : pfs[0].OUT[(ci - 1) & 1]), d.bytes));
                } else if (n == "position_ids") {
                    c.in.push_back(axera_io(pfs[0].POS, d.bytes));
                } else if (n == "indptr") {
                    c.in.push_back(axera_io(INDPTR, d.bytes));
                } else if (n == "indices") {
                    c.in.push_back(axera_io(INDICES, d.bytes));
                } else if (n == "last_page_len") {
                    c.in.push_back(axera_io(LASTLEN, d.bytes));
                } else if (n == "cache_location") {
                    c.in.push_back(axera_io(CLOC, d.bytes));
                } else if (n == "valid_tokens") {
                    c.in.push_back(axera_io(VALID, d.bytes));
                } else if (n == "b3_kv") {
                    GGML_ASSERT(d.bytes == kv_b && KVC[ci]->size >= kv_alloc);
                    c.in.push_back(axera_io(KVC[ci], kv_b));
                } else if (n.size() > 2 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                           (n.find("_state_in") != std::string::npos || n.find("_state_out") != std::string::npos ||
                            n.find("_conv_state_in") != std::string::npos ||
                            n.find("_conv_state_out") != std::string::npos)) {
                    c.in.push_back(state_io(n, ci));
                } else {
                    auto it = banks.find(n);
                    if (it == banks.end()) {
                        GGML_ABORT("axera qwen38_27b: fused: %s: unexpected input %s\n", k.desc->role.c_str(), n.c_str());
                    }
                    const bool shared = n == "b3_attn_cos_sin_cache";
                    const size_t off  = shared ? 0 : (size_t) ci * d.bytes;
                    GGML_ASSERT(it->second->size >= off + d.bytes);
                    c.in.push_back(axera_io(it->second.get(), (char *) it->second->vir + off, d.bytes));
                }
            }
            for (const auto & d : k.desc->out) {
                GGML_ASSERT(d.name == "output");
                c.out.push_back(axera_io(decode ? A : pfs[0].OUT[ci & 1], d.bytes));
            }
        }
    };
    prefill_set & P = pfs[0];
    P.k             = &pk;
    P.rows          = PG;
    P.X             = axera_cmm_alloc(pk.in_bytes("x"));
    P.POS           = axera_cmm_alloc(pk.in_bytes("position_ids"));
    P.OUT[0]        = axera_cmm_alloc(pk.out_bytes("output"));
    P.OUT[1]        = axera_cmm_alloc(pk.out_bytes("output"));
    GGML_ASSERT(pk.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                pk.in_bytes("position_ids") == 3 * (size_t) PG * sizeof(int32_t) &&
                pk.out_bytes("output") == (size_t) PG * H * sizeof(ggml_bf16_t));
    bind(pk, P.calls, false);
    // Consecutive calls alternate OUT[0] and OUT[1]. With more than one call
    // (the 2-block grouping used by the 4 GiB KV limit) the page's final rows
    // are in the last call's buffer, not always OUT[0].
    P.result = (NC - 1) & 1;
    bind(dk, dec, true);

    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen38_27b: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        if (g) {
            const int b = il / 4, ph = il % 4;
            ord[il] = ph * NB + b;  // the phase-major GDN state pools
            ++ng;
        } else {
            ord[il] = il / 4;  // the fused KV bank's block
            ++na;
        }
    }
    G = L;
    for (int k = 0; k < 4; ++k) {
        where[k].assign(ng, {});
        for (int g = 0; g < ng; ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set       = -1;
    state_zero    = true;
    ver_rows      = 0;
    state_ios.clear();
    ver.clear();

    seq_config cfg;
    cfg.kept_max     = 1;
    cfg.verify_chain = 0;
    cfg.check        = [] { const char * e = getenv("GGML_AXERA_SEQ_CHECK"); return e && atoi(e) != 0; }();
    cfg.force_miss   = [] { const char * e = getenv("GGML_AXERA_SEQ_FORCE_MISS"); return e ? std::max(0, atoi(e)) : 0; }();
    seq              = std::make_unique<seqstate>(*this, cfg);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen38_27b: fused %d layers, %d blocks, %d per call, banks %.2f GiB, state %.2f GiB, %lld cells\n",
                  L, NB, KB, bb / 1073741824.0,
                  (double) (KVC.size() * kv_b + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}


// The 2-core build of the upstream fused graph is numerically wrong once a GQA phase follows a delta-net phase in
// the same axmodel (CRUN: cos 0.78 for the full graph, 0.89 for [GDN, GQA]; the board reproduces it). The fix that
// keeps the upstream kernels untouched is to compile each 4-layer block twice: one axmodel with the three delta-net
// stages, one with the attention stage. This adapter runs them in layer order, block by block, on the same weight
// banks / state pools / KV pools. The manifests name the four roles prefill.gdn, prefill.attn, decode.gdn,
// decode.attn.
void qwen38_27b::init_fused_phases(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                                   int L_, int H_) {
    (void) rope_node;
    L    = L_;
    H    = H_;
    fused = true;
    nv4   = true;
    GGML_ASSERT(L % 4 == 0);
    NB = L / 4;
    const kernel & kg = man.get("prefill.gdn");
    const kernel & ka = man.get("prefill.attn");
    const kernel & km = man.get("prefill.mlp");
    const kernel & dg = man.get("decode.gdn");
    const kernel & da = man.get("decode.attn");
    const kernel & dm = man.get("decode.mlp");
    // The 2-core fallback splits the phase-3 attention and MLP into separate
    // graphs; the attention stage is the one that diverged when fused with MLP.
    // Require all six roles so a half-updated package fails at load, not silently.
    auto in_of = [&](const kernel & k, const std::string & n) -> const io_desc & {
        for (const auto & d : k.desc->in) {
            if (d.name == n) return d;
        }
        for (const auto & d : k.desc->out) {
            if (d.name == n) return d;
        }
        GGML_ABORT("axera qwen38_27b: %s has no IO %s\n", k.desc->file.c_str(), n.c_str());
    };
    if (kg.find_input("b0_attn_w_q") < 0) {
        GGML_ABORT("axera qwen38_27b: phase prefill needs the per-projection delta-net banks\n");
    }
    KB = (int) in_of(kg, "b0_mlp_w_norm").shape[0];
    GGML_ASSERT(KB == 1);
    NC = NB;
    const io_desc & pst = in_of(kg, "b0_attn_state_in");
    const io_desc & pcq = in_of(kg, "b0_attn_q_conv_state_in");
    const io_desc & pcv = in_of(kg, "b0_attn_v_conv_state_in");
    GGML_ASSERT(pst.shape.size() == 5);
    VH  = (int) pst.shape[2];
    KD  = (int) pst.shape[3];
    VD  = (int) pst.shape[4];
    st_b = pst.bytes / (size_t) KB;
    cq_b = pcq.bytes / (size_t) KB;
    cv_b = pcv.bytes / (size_t) KB;
    const io_desc & pq = in_of(kg, "b0_attn_w_q");
    GGML_ASSERT(pq.shape.size() == 3 && pq.shape[1] % KD == 0);
    KHd  = (int) (pq.shape[1] / KD);
    REPf = VH / KHd;
    GGML_ASSERT(KHd > 0 && REPf > 0 && VH % KHd == 0);

    const io_desc & pkv = in_of(ka, "b3_kv");
    GGML_ASSERT(pkv.shape.size() >= 5 && pkv.bytes % (size_t) KB == 0);
    const size_t kd = pkv.shape.size();
    GGML_ASSERT(kd >= 6 && pkv.shape[kd - 6] == KB && pkv.shape[kd - 4] == 2 && pkv.shape[kd - 3] == PG &&
                pkv.shape[kd - 1] == 256);
    kv_blk_b = pkv.bytes;
    kv_b     = pkv.bytes;
    kv_alloc = kv_b;
    kv_pages = (int) pkv.shape[kd - 5];
    kv_cells = man.context;
    GGML_ASSERT((int64_t) kv_pages * PG >= (int64_t) man.context);
    NBLK = kv_pages;
    ROT  = 64;
    kvs_b = 0;
    KVSC.clear();
    KVC.resize(NB);
    for (int b = 0; b < NB; ++b) {
        KVC[b] = axera_cmm_alloc(kv_b);
    }
    ST = axera_cmm_alloc(st_b * (size_t) NB * 3);
    CQ = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CK = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CV = axera_cmm_alloc(cv_b * (size_t) NB * 3);
    state_image_b = ST->size + CQ->size + CK->size + CV->size;

    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(kg.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(std::max(kg.in_bytes("last_page_len"),
                                       std::max(dg.in_bytes("last_page_len"), da.in_bytes("last_page_len"))));
    CLOC  = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID = axera_cmm_alloc(sizeof(int32_t));
    const int n_indices = (int) (kg.in_bytes("indices") / sizeof(int32_t));
    GGML_ASSERT(n_indices >= (int) ((man.context + PG - 1) / PG) && n_indices <= kv_pages);
    for (int i = 0; i < n_indices; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    A = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    B = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    C = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    D = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    banks.clear();
    auto skip = [](const std::string & n) {
        static const std::set<std::string> rt = { "x", "position_ids", "indptr", "indices", "last_page_len",
                                                  "cache_location", "valid_tokens", "b3_kv", "residual" };
        if (rt.count(n)) return true;
        return n.size() > 2 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
               (n.find("_state_in") != std::string::npos || n.find("_conv_state_in") != std::string::npos);
    };
    for (const kernel * k : { &kg, &ka, &km, &dg, &da, &dm }) {
        for (const auto & d : k->desc->in) {
            if (!skip(d.name) && !banks.count(d.name)) {
                banks[d.name] = bank_fused(t, d);
            }
        }
    }

    prefill_set & P = pfs[0];
    P.k             = &kg;
    P.rows          = PG;
    P.X             = axera_cmm_alloc(kg.in_bytes("x"));
    P.POS           = axera_cmm_alloc(kg.in_bytes("position_ids"));
    for (int i = 0; i < 4; ++i) {
        P.OUT[i] = axera_cmm_alloc(kg.out_bytes("output"));
    }
    GGML_ASSERT(kg.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                kg.in_bytes("position_ids") == 3 * (size_t) PG * sizeof(int32_t) &&
                kg.out_bytes("output") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                ka.in_bytes("x") == kg.in_bytes("x") && ka.out_bytes("output") == kg.out_bytes("output") &&
                km.out_bytes("output") == kg.out_bytes("output") &&
                dm.out_bytes("output") == (size_t) H * sizeof(ggml_bf16_t));

    auto state_io = [&](const std::string & n, int b) -> AX_ENGINE_IO_BUFFER_T {
        const int ph = n[1] - '0';
        const size_t per = n.find("q_conv_state_in") != std::string::npos ? cq_b :
                           n.find("k_conv_state_in") != std::string::npos ? cq_b :
                           n.find("v_conv_state_in") != std::string::npos ? cv_b : st_b;
        const axera_cmm * p = n.find("q_conv_state_in") != std::string::npos ? &CQ :
                              n.find("k_conv_state_in") != std::string::npos ? &CK :
                              n.find("v_conv_state_in") != std::string::npos ? &CV : &ST;
        GGML_ASSERT(ph >= 0 && ph < 3 && b >= 0 && b < NB);
        const size_t off = ((size_t) ph * NB + (size_t) b) * per;
        return axera_io(p->get(), (char *) (*p)->vir + off, per);
    };
    auto bind_phase = [&](bool decode, std::vector<call> & calls, int & result) {
        calls.clear();
        // Three kernels per block (delta-net phases 0-2, attention, MLP) with a
        // four-buffer rotation: block input, GDN out, attention out, MLP out.
        // After NB=8 blocks the rotation returns to buffer 0, the page result.
        const kernel * ks[3] = { decode ? &dg : &kg, decode ? &da : &ka, decode ? &dm : &km };
        auto buf = [&](int i) -> axera_cmm & {
            if (decode) {
                return i == 0 ? A : i == 1 ? B : i == 2 ? C : D;
            }
            return P.OUT[i];
        };
        int cur = 0;
        for (int b = 0; b < NB; ++b) {
            for (int j = 0; j < 3; ++j) {
                const kernel * k = ks[j];
                const int ini  = (cur + j) & 3;
                const int outi = (cur + j + 1) & 3;
                call c;
                c.k = k;
                for (const auto & d : k->desc->in) {
                    const std::string & n = d.name;
                    if (n == "x") {
                        c.in.push_back(axera_io(buf(ini), d.bytes));
                    } else if (n == "residual") {
                        // MLP keeps the block input as its residual while
                        // attention has already replaced the activation.
                        c.in.push_back(axera_io(buf(cur), d.bytes));
                    } else if (n == "position_ids") {
                        c.in.push_back(axera_io(P.POS, d.bytes));
                    } else if (n == "indptr") {
                        c.in.push_back(axera_io(INDPTR, d.bytes));
                    } else if (n == "indices") {
                        c.in.push_back(axera_io(INDICES, d.bytes));
                    } else if (n == "last_page_len") {
                        c.in.push_back(axera_io(LASTLEN, d.bytes));
                    } else if (n == "cache_location") {
                        c.in.push_back(axera_io(CLOC, d.bytes));
                    } else if (n == "valid_tokens") {
                        c.in.push_back(axera_io(VALID, d.bytes));
                    } else if (n == "b3_kv") {
                        GGML_ASSERT(d.bytes == kv_b);
                        c.in.push_back(axera_io(KVC[b], d.bytes));
                    } else if (n.size() > 2 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                               (n.find("_state_in") != std::string::npos || n.find("_conv_state_in") != std::string::npos)) {
                        c.in.push_back(state_io(n, b));
                    } else {
                        auto it = banks.find(n);
                        if (it == banks.end()) {
                            GGML_ABORT("axera qwen38_27b: fused phases: %s: unexpected input %s\n",
                                       k->desc->role.c_str(), n.c_str());
                        }
                        const bool shared = n == "b3_attn_cos_sin_cache";
                        const size_t off  = shared ? 0 : (size_t) b * d.bytes;
                        GGML_ASSERT(it->second->size >= off + d.bytes);
                        c.in.push_back(axera_io(it->second.get(), (char *) it->second->vir + off, d.bytes));
                    }
                }
                for (const auto & d : k->desc->out) {
                    GGML_ASSERT(d.name == "output" &&
                                d.bytes == (decode ? (size_t) H * sizeof(ggml_bf16_t) : P.OUT[0]->size));
                    c.out.push_back(axera_io(buf(outi), d.bytes));
                }
                calls.push_back(std::move(c));
            }
            cur = (cur + 3) & 3;
        }
        result = cur;
    };
    bind_phase(false, P.calls, P.result);
    bind_phase(true, dec, dec_result);

    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen38_27b: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        if (g) {
            ord[il] = (il % 4) * NB + il / 4;
            ++ng;
        } else {
            ord[il] = il / 4;
            ++na;
        }
    }
    G = L;
    for (int k = 0; k < 4; ++k) {
        where[k].assign(ng, {});
        for (int g = 0; g < ng; ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set    = -1;
    state_zero = true;
    ver_rows   = 0;
    state_ios.clear();
    ver.clear();

    seq_config cfg;
    cfg.kept_max     = 1;
    cfg.verify_chain = 0;
    cfg.check        = [] { const char * e = getenv("GGML_AXERA_SEQ_CHECK"); return e && atoi(e) != 0; }();
    cfg.force_miss   = [] { const char * e = getenv("GGML_AXERA_SEQ_FORCE_MISS"); return e ? std::max(0, atoi(e)) : 0; }();
    seq              = std::make_unique<seqstate>(*this, cfg);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen38_27b: fused phases %d layers, %d blocks, GDN+attn kernels, banks %.2f GiB, "
                  "state %.2f GiB, %lld cells\n",
                  L, NB, bb / 1073741824.0,
                  (double) (KVC.size() * kv_b + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
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
    state_zero = false;
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

seq_caps qwen38_27b::seq_get_caps() const {
    seq_caps c;
    c.page            = PG;
    c.width           = H;
    c.image_bytes     = state_image_b;
    c.cells           = kv_cells;
    c.verify_rows     = ver_rows;
    // The verify kernel updates its entry state in place, so a fully rejected batch has no saved entry image to keep
    // zero rows from. Rewind to the page's restart point instead (one image load, correctness over the rare case).
    c.verify_min_keep = ver_rows > 0 ? 1 : 0;
    c.decode_at_zero  = true;  // the decode group runs cell 0 (the old path did, after a zeroed state)
    return c;
}

void qwen38_27b::seq_state_save(uint8_t * dst) {
    axera_timer tl("state save");
    size_t o = 0;
    if (cur_set < 0) {  // the pools: one contiguous copy per kind
        for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
            (*p)->invalidate((*p)->vir, (*p)->size);
            memcpy(dst + o, (*p)->vir, (*p)->size);
            o += (*p)->size;
        }
    } else {  // a verification snapshot: its rows are not contiguous in the pools' order
        for (int k = 0; k < 4; ++k) {
            for (const place & w : where[k]) {
                w.b->invalidate(w.p, part(k));
                memcpy(dst + o, w.p, part(k));
                o += part(k);
            }
        }
    }
    GGML_ASSERT(o == state_image_b);
    state_zero = false;
}

// src == nullptr: the zero state of a new sequence (and of the dummy warm-up batch's cleanup).
void qwen38_27b::seq_state_load(const uint8_t * src) {
    axera_timer tl("state load");
    const bool had_snapshot = cur_set >= 0;
    ver_set = -1;
    if (!src) {
        if (!state_zero || had_snapshot) {
            axera_timer tz("state zero");
            for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
                memset((*p)->vir, 0, (*p)->size);
                (*p)->flush();
            }
        }
        state_zero = true;
        if (MKV && mtp_page_b && mtp_cell > 0) {
            const size_t pages = std::min<size_t>(MKV->size / mtp_page_b, (size_t) ((mtp_cell + PG - 1) / PG));
            if (pages) {
                memset(MKV->vir, 0, pages * mtp_page_b);
                MKV->flush(MKV->vir, pages * mtp_page_b);
            }
        }
        mtp_cell = 0;
    } else {
        size_t o = 0;
        for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
            memcpy((*p)->vir, src + o, (*p)->size);
            (*p)->flush();
            o += (*p)->size;
        }
        GGML_ASSERT(o == state_image_b);
        state_zero = false;
    }
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set = -1;
    bind_state();
}

bool qwen38_27b::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    state_home();  // the prefill kernels read the pools
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    prefill(c0, n, xb, p4, h_buf, batch0, snap);
    return true;
}

bool qwen38_27b::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
    if (fused) {
        // the upstream decode kernel takes the token's three M-RoPE positions, a one-page table and the cell in its
        // page; the recurrent and conv states are updated in place.
        memcpy(A->vir, x, H * sizeof(ggml_bf16_t));
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        int32_t * pp = (int32_t *) pfs[0].POS->vir;
        for (int s = 0; s < 3; ++s) {
            pp[s] = p4 ? p4[s] : s < 2 ? (int32_t) c : 0;
        }
        pfs[0].POS->flush(pfs[0].POS->vir, 3 * sizeof(int32_t));
        ((int32_t *) INDPTR->vir)[0] = 0;
        ((int32_t *) INDPTR->vir)[1] = (int32_t) (c / PG + 1);
        ((int32_t *) LASTLEN->vir)[0] = (int32_t) (c % PG);
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (c / PG);
        ((int32_t *) CLOC->vir)[1]    = (int32_t) (c % PG);
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
            (*b)->flush();
        }
        {
            axera_timer tl("layers (decode)");
            for (auto & cc : dec) {
                cc.k->run(cc.in, cc.out, "dense decode");
            }
            state_zero = false;
        }
        const axera_cmm & out = dec_result == 0 ? A : dec_result == 1 ? B : dec_result == 2 ? C : D;
        out->invalidate(out->vir, H * sizeof(ggml_bf16_t));
        if (dec_result != 0) {
            memcpy(A->vir, out->vir, H * sizeof(ggml_bf16_t));
            A->flush(A->vir, H * sizeof(ggml_bf16_t));
        }
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, out_h, H);
        return true;
    }
    (void) p4;  // the row's M-RoPE positions come from the graph's rope view (batch_rope)
    memcpy(A->vir, x, H * sizeof(ggml_bf16_t));
    A->flush(A->vir, H * sizeof(ggml_bf16_t));
    batch_rope->rows(0, 1, COS.get(), SIN.get());
    const int nblk = batch_kv->mask_row(0, (ggml_bf16_t *) MASK->vir);
    MASK->flush(MASK->vir, (size_t) nblk * PG * sizeof(ggml_bf16_t));
    const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
    memcpy(CS->vir, COS->vir, half);
    memcpy((char *) CS->vir + half, SIN->vir, half);
    CS->flush();
    ((int32_t *) PN->vir)[0] = (int32_t) c;
    ((int32_t *) PN->vir)[1] = nblk;
    ((int32_t *) PN->vir)[2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
    PN->flush();
    {
        axera_timer tl("layers (decode)");
        for (auto & c : dec) {
            c.k->run(c.in, c.out, "decode group");
        }
        state_zero = false;
    }
    A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, out_h, H);
    return true;
}

void qwen38_27b::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

bool qwen38_27b::seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) {
    const uint16_t * xr = x;
    if (!xr) {
        src_rows.resize((size_t) n * H);
        seq_rows(batch_row0, n, (uint16_t *) src_rows.data());
        xr = (const uint16_t *) src_rows.data();
    }
    return verify(c, batch_row0, n, (const ggml_bf16_t *) xr, p4, out_h) == GGML_STATUS_SUCCESS;
}

void qwen38_27b::seq_verify_keep(int k) {
    GGML_ASSERT(ver_set >= 0 && k > 0 && k <= ver_rows);
    state_at(ver_set, k - 1);
}

void qwen38_27b::seq_verify_peek(int k, uint8_t * dst) {
    GGML_ASSERT(ver_set >= 0 && k > 0 && k <= ver_rows);
    const int    per = G / 4 * 3;  // delta-net layers per group
    size_t       o   = 0;
    for (int kind = 0; kind < 4; ++kind) {
        for (size_t g = 0; g < where[kind].size(); ++g) {
            const axera_cmm & sn = snaps[ver_set][g / per][kind];
            char *            p  = (char *) sn->vir + ((size_t) (k - 1) * per + g % per) * part(kind);
            sn->invalidate(p, part(kind));
            memcpy(dst + o, p, part(kind));
            o += part(kind);
        }
    }
    GGML_ASSERT(o == state_image_b);
}

// llama.cpp's sequence state (prompt cache / context checkpoint): see qwen35_whole. The private CMM pools are not the
// llama.cpp cache tensors, so the checkpoint carries the KV pages before the current cell, the recurrent state and the
// current page's restart point. The int8 KV caches' per-row scales travel with them (KVSC).
size_t qwen38_27b::state_size(uint32_t flags) const {
    mtp_wait();  // the MTP catch-up extends MKV and mtp_cell; size and bytes must be consistent
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    const bool   partial = flags & state_flags_partial_only;
    const size_t seq_n   = seq->size(partial);
    if (!seq_n) {
        return 0;
    }
    const int     pna       = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    const int64_t state_cell = std::max<int64_t>(0, seq->cell());
    const size_t  nkv_pages  = std::min<size_t>((size_t) kv_pages, (size_t) ((state_cell + PG - 1) / PG));
    size_t        mtp_pages  = 0;
    if (MKV && mtp_page_b && mtp_cell > 0) {
        mtp_pages = partial ? 1 : std::min<size_t>(MKV->size / mtp_page_b, (size_t) ((mtp_cell + PG - 1) / PG));
    }
    size_t n = sizeof(state_header_27b) + seq_n;
    if (!partial) {
        if (fused) {
            n += (size_t) KVC.size() * (size_t) KB * nkv_pages * kv_blk_b / (size_t) kv_pages;
        } else {
            n += (size_t) KVC.size() * pna * nkv_pages * (kv_b / NBLK);
            n += (size_t) KVSC.size() * pna * nkv_pages * (kvs_b / NBLK);
        }
    }
    n += mtp_pages * mtp_page_b;
    return n;
}

size_t qwen38_27b::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    mtp_wait();  // an async MTP catch-up writes MKV; serialize after it finishes
    const size_t total = state_size(flags);
    if (!total || size < total) {
        return 0;
    }
    const bool   partial = flags & state_flags_partial_only;
    const int    pna     = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    const size_t seq_n   = seq->size(partial);
    const size_t nkv_pages = partial ? 0 : std::min<size_t>((size_t) kv_pages, (size_t) ((std::max<int64_t>(0, seq->cell()) + PG - 1) / PG));
    size_t mtp_pages = 0, mtp_page0 = 0;
    if (MKV && mtp_page_b && mtp_cell > 0) {
        if (partial) {
            mtp_pages = 1;
            mtp_page0 = (size_t) ((mtp_cell - 1) / PG);
        } else {
            mtp_pages = std::min<size_t>(MKV->size / mtp_page_b, (size_t) ((mtp_cell + PG - 1) / PG));
        }
    }

    state_header_27b h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "AXW27S2", sizeof(h.magic) - 1);
    h.version     = 3;
    h.kind        = 2;
    h.has_target  = 1;
    h.has_mtp     = mtp_pages > 0;
    h.state_cell  = seq->cell();
    h.mtp_cell    = mtp_cell;
    h.ncall       = (uint32_t) KVC.size();
    h.pna         = (uint32_t) pna;
    h.nkv_pages   = (uint32_t) nkv_pages;
    h.kv_page_b   = fused ? kv_blk_b / (size_t) kv_pages : kv_b / NBLK;
    h.kvs_page_b  = kvs_b / NBLK;
    h.kv_n        = partial ? 0 : fused ? (uint64_t) KVC.size() * KB * nkv_pages * h.kv_page_b
                                      : (uint64_t) KVC.size() * pna * nkv_pages * (kv_b / NBLK);
    h.kvs_n       = partial ? 0 : fused ? 0 : (uint64_t) KVSC.size() * pna * nkv_pages * (kvs_b / NBLK);
    h.seq_n       = seq_n;
    h.mtp_page0   = mtp_page0;
    h.mtp_pages   = mtp_pages;
    h.mtp_page_b  = mtp_page_b;
    h.mtp_n       = mtp_pages * mtp_page_b;
    h.partial     = partial;

    char * p = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (!partial) {
        if (fused) {
            const size_t page_b = kv_blk_b / (size_t) kv_pages;
            const size_t page_n = nkv_pages * page_b;
            for (const axera_cmm & c : KVC) {
                for (int blk = 0; blk < KB; ++blk) {
                    char * src = (char *) c->vir + (size_t) blk * kv_blk_b;
                    c->invalidate(src, page_n);
                    memcpy(p, src, page_n);
                    p += page_n;
                }
            }
        } else {
            const size_t page_n = nkv_pages * (kv_b / NBLK);
            for (const axera_cmm & c : KVC) {
                for (int j = 0; j < pna; ++j) {
                    char * src = (char *) c->vir + (size_t) j * kv_alloc;
                    c->invalidate(src, page_n);
                    memcpy(p, src, page_n);
                    p += page_n;
                }
            }
            const size_t scale_n = nkv_pages * (kvs_b / NBLK);
            for (const axera_cmm & c : KVSC) {
                for (int j = 0; j < pna; ++j) {
                    char * src = (char *) c->vir + (size_t) j * kvs_alloc;
                    c->invalidate(src, scale_n);
                    memcpy(p, src, scale_n);
                    p += scale_n;
                }
            }
        }
    }
    if (seq->write((uint8_t *) p, seq_n, partial) != seq_n) {
        return 0;
    }
    p += seq_n;
    if (h.mtp_n) {
        char * src = (char *) MKV->vir + mtp_page0 * mtp_page_b;
        MKV->invalidate(src, h.mtp_n);
        memcpy(p, src, h.mtp_n);
        p += h.mtp_n;
    }
    return (size_t) (p - (char *) dst);
}

size_t qwen38_27b::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    mtp_wait();
    if (size < sizeof(state_header_27b)) {
        return 0;
    }
    const bool partial = flags & state_flags_partial_only;
    state_header_27b h;
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, "AXW27S2", sizeof(h.magic) - 1) != 0 || h.version != 3 || h.kind != 2 || !h.has_target ||
        h.partial != partial || !h.seq_n) {
        return 0;
    }
    const int pna = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    const bool kv_ok = fused ? h.kv_page_b == kv_blk_b / (size_t) kv_pages && h.kvs_n == 0 &&
                                  (partial ? h.nkv_pages != 0 || h.kv_n == 0
                                           : h.nkv_pages <= (uint32_t) kv_pages &&
                                                 h.kv_n == (uint64_t) h.ncall * KB * h.nkv_pages * h.kv_page_b)
                            : h.kv_page_b == kv_b / NBLK && h.kvs_page_b == kvs_b / NBLK &&
                                  (partial ? h.nkv_pages != 0 || h.kv_n == 0 || h.kvs_n == 0
                                           : h.nkv_pages <= (uint32_t) kv_pages &&
                                                 h.kv_n == (uint64_t) h.ncall * h.pna * h.nkv_pages * h.kv_page_b &&
                                                 h.kvs_n == (uint64_t) KVSC.size() * h.pna * h.nkv_pages * h.kvs_page_b);
    if (KVC.size() != h.ncall || pna != (int) h.pna || !kv_ok ||
        h.state_cell < -1 || h.state_cell > kv_cells) {
        return 0;
    }
    const bool mtp_valid = h.mtp_n == 0 ? h.mtp_pages == 0 && !h.has_mtp :
                           h.has_mtp && MKV && mtp_page_b && h.mtp_page_b == mtp_page_b &&
                               h.mtp_pages > 0 && h.mtp_pages <= MKV->size / mtp_page_b &&
                               h.mtp_page0 <= MKV->size / mtp_page_b - h.mtp_pages &&
                               h.mtp_pages * mtp_page_b == h.mtp_n && (!partial || h.mtp_pages == 1) &&
                               h.mtp_cell >= 0 && h.mtp_cell <= kv_cells;
    if (!mtp_valid) {
        return 0;
    }
    const size_t need = sizeof(h) + h.kv_n + h.kvs_n + h.seq_n + h.mtp_n;
    if (size < need) {
        return 0;
    }

    const char * p = (const char *) src + sizeof(h);
    if (!partial) {
        const size_t page_n = h.nkv_pages * h.kv_page_b;
        if (fused) {
            for (const axera_cmm & c : KVC) {
                for (int blk = 0; blk < KB; ++blk) {
                    char * d = (char *) c->vir + (size_t) blk * kv_blk_b;
                    memcpy(d, p, page_n);
                    c->flush(d, page_n);
                    p += page_n;
                }
            }
        } else {
            for (const axera_cmm & c : KVC) {
                for (int j = 0; j < (int) h.pna; ++j) {
                    char * d = (char *) c->vir + (size_t) j * kv_alloc;
                    memcpy(d, p, page_n);
                    c->flush(d, page_n);
                    p += page_n;
                }
            }
            const size_t scale_n = h.nkv_pages * h.kvs_page_b;
            for (const axera_cmm & c : KVSC) {
                for (int j = 0; j < (int) h.pna; ++j) {
                    char * d = (char *) c->vir + (size_t) j * kvs_alloc;
                    memcpy(d, p, scale_n);
                    c->flush(d, scale_n);
                    p += scale_n;
                }
            }
        }
    }
    if (seq->read((const uint8_t *) p, h.seq_n) != h.seq_n) {
        return 0;
    }
    p += h.seq_n;
    if (h.mtp_n) {
        char * dst = (char *) MKV->vir + (size_t) h.mtp_page0 * mtp_page_b;
        memcpy(dst, p, h.mtp_n);
        MKV->flush(dst, h.mtp_n);
        p += h.mtp_n;
    }
    mtp_cell = h.mtp_cell;
    state_zero = false;
    return (size_t) (p - (const char *) src);
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16 and their M-RoPE positions p4 [n][4]; the hidden
// states of the rows from out0 on into h
void qwen38_27b::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                         seq_snap * snap) {
    GGML_ASSERT(c0 % PG == 0);
    GGML_ASSERT(c0 + n <= kv_cells);  // (the pools' pages)
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (v < PG && snap && snap->cell == cell) {  // the state before the run's last (partial) page
            seq_state_save(snap->dst);
        }
        // the smallest row variant that holds this page's valid rows (the tails save the padding work)
        prefill_set * P = nullptr;
        for (int i = 4; i >= 0; --i) {
            if (pfs[i].k && v <= pfs[i].rows) {
                P = &pfs[i];
                break;
            }
        }
        GGML_ASSERT(P && v <= P->rows);
        memcpy(P->X->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) P->X->vir + (size_t) v * H, 0, (size_t) (P->rows - v) * H * sizeof(ggml_bf16_t));
        P->X->flush();
        // The three-kernel rotation starts with OUT[0] as the block input.
        memcpy(P->OUT[0]->vir, P->X->vir, P->X->size);
        P->OUT[0]->flush();
        auto * pp = (int32_t *) P->POS->vir;
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < P->rows; ++r) {
                const int rr = fused ? (r < v ? r : std::max(0, v - 1)) : (r < v ? r : 0);
                pp[s * P->rows + r] = p4[(size_t) (r0 + rr) * 4 + s];
            }
        }
        P->POS->flush();
        ((int32_t *) INDPTR->vir)[0]  = 0;
        ((int32_t *) INDPTR->vir)[1]  = (int32_t) (cell / PG);
        *(int32_t *) LASTLEN->vir     = PG;
        if (fused) {
            ((int32_t *) LASTLEN->vir)[1] = PG;
        }
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (cell / PG);
        ((int32_t *) CLOC->vir)[1]    = 0;
        *(int32_t *) VALID->vir       = v;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        {
            axera_timer tl("prefill (page)");
            for (auto & c : P->calls) {
                c.k->run(c.in, c.out, "prefill");
            }
            state_zero = false;
        }
        axera_timer       to("prefill out rows");
        const axera_cmm & po = P->OUT[P->result];
        po->invalidate(po->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
    }
}

// T (2 - 4) consecutive tokens from cell c0 through the verification groups, their hidden states into h at the batch
// rows. Rows T..3 are padded (the last row's positions and mask); the state after each row lands in the other snapshot
// set, one past the entry state that the kernel updates in place.
ggml_status qwen38_27b::verify(int64_t c0, int row0, int T, const ggml_bf16_t * xb, const int32_t * pb, float * h) {
    (void) pb;
    GGML_ASSERT(batch_kv && batch_rope && T >= 1 && T <= ver_rows);
    memset(VX->vir, 0, VX->size);
    memcpy(VX->vir, xb, (size_t) T * H * sizeof(ggml_bf16_t));
    VX->flush();
    const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
    auto *       cs   = (char *) VCS->vir;
    auto *       pn   = (int32_t *) VPN->vir;
    for (int r = 0; r < 4; ++r) {
        const int     rr = std::min(r, T - 1);
        const int64_t c  = c0 + r;
        batch_rope->rows(row0 + rr, 1, COS.get(), SIN.get());
        memcpy(cs + r * 2 * half, COS->vir, half);
        memcpy(cs + r * 2 * half + half, SIN->vir, half);
        const int nblk = batch_kv->mask_row(row0 + rr, (ggml_bf16_t *) VMASK->vir + (size_t) r * man.context);
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
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) VX->vir + (size_t) r * H,
                              h + (size_t) (row0 + r) * H, H);
    }
    state_at(set, T - 1);
    ver_set = set;
    return GGML_STATUS_SUCCESS;
}

ggml_status qwen38_27b::compute(ggml_cgraph * g) {
    mtp_wait();
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
        if (man.has("prefill.gdn")) {
            init_fused_phases(t, rope_node, L_, H_);
        } else if (man.has("decode") && !man.has("decode.group")) {
            init_fused(t, rope_node, L_, H_);
        } else {
            init(t, rope_node, L_, H_);
        }
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
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const int64_t      c0 = kv.cell_of(0);
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
        });
    }
    // common.cpp marks its warm-up batch with llama_set_dummy_batch(): its state is not needed. Drop it here, while
    // the batch is still being timed as warm-up, so the prompt's own zero state can skip the memset.
    const bool warmup = c0 == 0 && names.find("inp_mixed_slots_dummy");
    batch_rows      = xb.data();
    batch_kv        = &kv;
    batch_rope      = &rp;
    out_h           = h.data();
    last_seq_status = 0;
    const seq_status sst = seq->feed(c0, T, (const uint16_t *) xb.data(), pb.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) sst;
        if (sst == SEQ_MISS) {
            GGML_LOG_WARN("axera qwen38_27b: batch from cell %lld has no restart point (%s); the slot must replay\n",
                          (long long) c0, seq->why());
        } else {
            GGML_LOG_ERROR("axera qwen38_27b: sequence state rejected a batch from cell %lld: %s\n",
                           (long long) c0, seq->why());
        }
        return GGML_STATUS_FAILED;
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
    if (fused) {
        // the upstream rmsnorm_gemm head: an [8, K] bf16 tile, the final-norm weight and the NVFP4 output head.
        // result_output may be the output.scale MUL over the mm; graph_head walks to the mm and bind_norm_gemm takes
        // the per-tensor scale from output.scale.
        ggml_tensor * const hm = graph_head(names);
        GGML_ASSERT(hm && hm->op == GGML_OP_MUL_MAT && hm->src[0] && hm->src[0]->type == GGML_TYPE_NVFP4);
        float s2 = 1.0f;
        auto s2it = t.find("output.scale");
        if (s2it != t.end() && s2it->second->type == GGML_TYPE_F32 && ggml_nelements(s2it->second) == 1) {
            s2 = ((const float *) s2it->second->data)[0];
        }
        if (!head.bind_norm_gemm(man.get("lm_head"), hm->src[0], s2)) {
            GGML_ABORT("axera qwen38_27b: the lm_head IO is not the upstream rmsnorm_gemm head\n");
        }
        head.run(g, names, h, T);
    } else {
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
    }
    if (warmup) {  // the warm-up sequence must not make the first real prompt clear the state
        seq->forget();
        seq_state_load(nullptr);
    }
    return GGML_STATUS_SUCCESS;
}

// the MTP block's kernels: its packs from blk.<il>.* (both kernels the same layout), its paged KV cache, the IO
void qwen38_27b::init_mtp(const std::map<std::string, ggml_tensor *> & t, int il) {
    axera_timer    tb("init: mtp");
    const kernel & k1 = man.get("mtp.t1");
    const kernel & kp = man.get("mtp.prompt");
    GGML_ASSERT(k1.desc->packs == kp.desc->packs && k1.out_bytes("kv") == kp.out_bytes("kv"));
    MKV = axera_cmm_alloc(k1.out_bytes("kv") / NBLK * kv_pages);  // (paged like the layers' caches)
    mtp_page_b = k1.out_bytes("kv") / NBLK;
    mtp_kv_row = (size_t) k1.out_bytes("kv") / NBLK / (2 * PG);   // one K (or V) row: HKV * D bf16
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
            c.out.push_back(d.name == std::string("kv") ? axera_io(b->get(), (*b)->vir, (uint32_t) d.bytes) : axera_io(*b, d.bytes));
        }
    };
    bind(mtp1, k1, { { "e", &ME }, { "h", &MH }, { "cs", &MCS }, { "pn", &MPN }, { "mask", &MMASK }, { "hnext", &MOUT },
                     { "hraw", &MRAW } });
    bind(mtpp, kp, { { "e", &PE }, { "h", &PH }, { "cs", &PCS }, { "pn", &PPN }, { "mask", &PMASK }, { "nblk", &PNBLK },
                     { "hnext", &POUT2 }, { "hraw", &PRAW } });
    GGML_ASSERT(MCS->size == (size_t) ROT * sizeof(ggml_bf16_t) && MMASK->size == (size_t) man.context * sizeof(ggml_bf16_t) &&
                PE->size % ((size_t) H * sizeof(ggml_bf16_t)) == 0);
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live = { &mtp1.in, &mtpp.in };
    if (man.has("mtp.promptkv")) {  // the K/V-only prompt pass (the last row uses mtp1's full block)
        const kernel & kpk = man.get("mtp.promptkv");
        // its compact pack has only the eh_proj / K / V weights; its K/V rows are copied into MKV by the host
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
ggml_status qwen38_27b::compute_mtp(ggml_cgraph * g) {
    const auto &  t     = names.cur.t;
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
        // (GGML_AXERA_MTP_SYNC=1: in line) asynchronous when this graph has no output rows: neither its logits nor
        // its h_nextn rows are read
        static const bool  sync = getenv("GGML_AXERA_MTP_SYNC") && atoi(getenv("GGML_AXERA_MTP_SYNC")) != 0;
        const ggml_tensor * lg   = names.find("result_output");
        if (!sync && !(lg && lg->ne[1] > 0 && (lg->flags & GGML_TENSOR_FLAG_COMPUTE))) {
            tm.stop();
            mtp_bg = std::thread([this] {
                axera_timer tb("mtp (catch-up rows, async)");
                mtp8.k->run(mtp8.in, mtp8.out, "mtp m8");
            });
            return GGML_STATUS_SUCCESS;
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
            if (n < P) {  // (a full chunk overwrites every row)
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
                const int     rr = std::min(r, n - 1);  // padded rows: the last row's positions and mask, later cells
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
                for (int r = 0; r < n; ++r) {  // each row's K/V into its paged cell (K page then V page)
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
            // the last prompt row's full block: its h_nextn seeds the first draft step, and mtp1 writes its KV cell too
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
                GGML_LOG_ERROR("axera qwen38_27b: the MTP head is not a combined lm_head\n");
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

std::shared_ptr<model> make_qwen38_27b(const manifest & m) {
    return std::make_shared<qwen38_27b>(m);
}

}  // namespace axera
