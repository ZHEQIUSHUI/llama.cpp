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
#include "../../ops/llm.h"
#include "../../ops/gdn_bank.h"
#include "../../ops/gdn_mtp.h"
#include "../../ops/gdn_state.h"

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

using call = npu_call;

// the adapter blob in llama.cpp's sequence state (context checkpoints / prompt cache). The recurrent state itself is
// core/seqstate's blob; this header only describes the KV pages and the MTP cache around it. AXW27S1 (version 2) is
// refused: its state layout is the old adapter-owned one.
class qwen38_27b final : public model, public seq_model, gdn_state {
  public:
    explicit qwen38_27b(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size(uint32_t flags) const override;
    size_t       state_write(void * dst, size_t size, uint32_t flags) override;
    size_t       state_read(const void * src, size_t size, uint32_t flags) override;
    kv_state_format state_format() const;
    cmm_kv_pools    state_pools() const;
    int          last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override { state_save(dst); }
    void     seq_state_load(const uint8_t * src) override {
        if (state_load(src)) {
            mtp.clear();
        }
    }
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override;
    bool     seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) override;
    void     seq_verify_keep(int k) override { verify_keep(k); }
    void     seq_verify_peek(int k, uint8_t * dst) override { verify_peek(k, dst); }

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    std::vector<int> ord;  // layer -> its index among the delta-net layers / the attention layers

    // the state pools; their sizes per layer
    axera_cmm              KV;
    std::vector<axera_cmm> KVC;  // the KV cache: one buffer per prefill call's attention layers (a CMM allocation is < 4 GB)
    std::vector<axera_cmm> KVSC;  // its scales (an int8 cache: a bf16 scale per row and head), the same split
    size_t    kv_b = 0, kvs_b = 0;
    // the pages of the cache the pools hold (GGML_AXERA_KV_PAGES; default: the kernels' context): the kernels keep their
    // declared IO size but touch only the pages up to the runtime page count, so a smaller context needs only its pages
    size_t    kv_alloc = 0, kvs_alloc = 0;  // the bytes of one layer's pool
    int                            dec_result = 0;
    const axera_cmm *              dec_out = nullptr;
    axera_cmm                      A2, DIN, B, C, D;
    gdn_mtp mtp;  // the MTP block (draft context graphs)
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
    axera_cmm                      INDPTR, INDICES, LASTLEN, CLOC, VALID;
    output_head                    head;

    // The upstream fused dense path (sgl_kernel_axera.models.qwen38_27b.forward_qwen38_blocks): one prefill kernel and
    // one decode kernel carry whole 4-layer blocks in banks [KB, ...]. NB = L / 4 blocks in the model, KB = the blocks
    // one kernel call owns, NC = NB / KB calls per row batch. The 2B/4B manifests have no decode.group; the 27B
    // manifest keeps it and runs the older per-group path untouched.
    bool   fused = false;
    bool   nv4   = false;
    int    NB = 0, KB = 0, NC = 1;
    // delta-net value/key heads and their dimensions. VH hides the file-level constant of the same name in every
    // member function, so it starts at that value (KH * REP): the per-group path reads it; the fused paths set it
    // from the kernel's state shape.
    int    VH = KH * REP, KHd = 0, KD = 0, VD = 0, REPf = 0;
    size_t kv_blk_b = 0;                              // bytes of one block's KV pool (all pages)

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_fused(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void init_fused_phases(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    gdn_mtp_env mtp_env();
    axera_cmm bank_fused(const std::map<std::string, ggml_tensor *> & t, const io_desc & d);
    int       vtiled(int g) const { return (g % REPf) * KHd + g / REPf; }
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                      seq_snap * snap);
};

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
    // llama.cpp's -c sizes the pools unless a multi-layer prefill call needs every page
    kv_pages = llm_kv_pages(NBLK, PG, split_kv || pna == 1);
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
                banks[d.name] = gdn_prefill_bank({ "qwen38_27b", L, ROT, KH, VH, false }, t, dd, rp);
            }
        }
        axera_bank_sum("qwen38_27b", banks);
    }
    bind_decode("qwen38_27b", man, prm, t, L, H, ng, ord, pool_io);

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
    // The 27B verify kernel materializes one 4-row snapshot per call; chaining several calls for a 5..32-row tail
    // changes the numerics (and the kernel count) of a path that used to be one prefill page, which flips greedy
    // continuations in long degraded chats. Keep the pre-migration path selection by default; the environment
    // variable still opts into a longer chain.
    seq_config cfg = seq_config_env(1, env_int("GGML_AXERA_VERIFY_CHAIN", ver_rows));
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
    NBLK = (int) pkv.shape[kd - 5];
    GGML_ASSERT((int64_t) NBLK * PG >= (int64_t) man.context);
    // A one-block bank has a fixed page stride and can hold only the pages the
    // requested context needs, like the released 27B prefill.  A multi-block
    // bank needs every page to keep the block stride in the compiled IO.
    kv_pages = llm_kv_pages(NBLK, PG, KB == 1);
    GGML_ASSERT(kv_pages == NBLK || KB == 1);
    kv_cells = (int64_t) kv_pages * PG;
    ROT  = 64;

    kv_b      = pkv.bytes;  // one call's KV bank (KB blocks)
    kv_alloc  = getenv("GGML_AXERA_KV_FULL") ? kv_b : kv_b / (size_t) NBLK * (size_t) kv_pages;
    kvs_b     = 0;
    kvs_alloc = 0;
    KVC.resize(NC);
    for (int c = 0; c < NC; ++c) {
        KVC[c] = axera_cmm_alloc(kv_alloc);
        // A partially filled page is masked by the decode kernel, but a NaN in
        // its untouched slots survives the additive mask (NaN + -3e38).  A
        // fresh CMM pool can hold such patterns, so start it at a finite zero.
        memset(KVC[c]->vir, 0, KVC[c]->size);
        KVC[c]->flush();
    }
    KVSC.clear();
    ST = axera_cmm_alloc(st_b * (size_t) NB * 3);
    CQ = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CK = axera_cmm_alloc(cq_b * (size_t) NB * 3);
    CV = axera_cmm_alloc(cv_b * (size_t) NB * 3);
    // A fresh CMM pool is not guaranteed to hold zero, and the recurrent state
    // update reads it before writing.  Start from a finite zero like the KV.
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memset((*p)->vir, 0, (*p)->size);
        (*p)->flush();
    }
    state_image_b = ST->size + CQ->size + CK->size + CV->size;

    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(std::max(pk.in_bytes("last_page_len"), dk.in_bytes("last_page_len")));
    GGML_ASSERT(LASTLEN->size >= 2 * sizeof(int32_t));
    CLOC  = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID = axera_cmm_alloc(sizeof(int32_t));
    const int n_indices = (int) (pk.in_bytes("indices") / sizeof(int32_t));
    // The kernel's table input is sized for the manifest's maximum context;
    // a smaller -c allocates fewer KV pages but leaves this input unchanged.
    // A multi-block bank keeps every page (fixed block stride) even when the
    // requested context is shorter, so the table only has to cover the context.
    GGML_ASSERT(n_indices >= (int) ((man.context + PG - 1) / PG));
    GGML_ASSERT(KB > 1 || n_indices >= kv_pages);
    // Entries past the context-sized pool must not point outside it: the
    // compiled table is always 256k long, while a smaller -c allocates only
    // the first kv_pages pages.
    for (int i = 0; i < n_indices; ++i) {
        ((int32_t *) INDICES->vir)[i] = i < kv_pages ? i : 0;
    }
    INDICES->flush();

    A   = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    A2  = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    DIN = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
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
                    const axera_cmm & xb = decode ? (ci == 0 ? DIN : ((ci - 1) & 1 ? A2 : A)) :
                                                    (ci == 0 ? pfs[0].X : pfs[0].OUT[(ci - 1) & 1]);
                    c.in.push_back(axera_io(xb, d.bytes));
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
                c.out.push_back(axera_io(decode ? (ci & 1 ? A2 : A) : pfs[0].OUT[ci & 1], d.bytes));
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
    dec_out = ((NC - 1) & 1) ? &A2 : &A;

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

    seq = std::make_unique<seqstate>(*this, seq_config_env(1));
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
        const size_t per = n.find("q_conv_state") != std::string::npos ? cq_b :
                           n.find("k_conv_state") != std::string::npos ? cq_b :
                           n.find("v_conv_state") != std::string::npos ? cv_b : st_b;
        const axera_cmm * p = n.find("q_conv_state") != std::string::npos ? &CQ :
                              n.find("k_conv_state") != std::string::npos ? &CK :
                              n.find("v_conv_state") != std::string::npos ? &CV : &ST;
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
    dec_out = dec_result == 0 ? &A : dec_result == 1 ? &B : dec_result == 2 ? &C : &D;

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

    seq = std::make_unique<seqstate>(*this, seq_config_env(1));
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
// the current state into the pools (the prefill kernel's, the decode calls' default)
// the current state: the snapshot after row `row` of the last verification (set `set`)
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
    // The upstream single-row graph has no correct one-core variant for this shape yet; a temporary switch keeps
    // the model usable through the validated page kernel while that kernel is repaired.
    c.decode_kernel = getenv("GGML_AXERA_NO_DECODE_KERNEL") == nullptr;
    return c;
}

// src == nullptr: the zero state of a new sequence (and of the dummy warm-up batch's cleanup).
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
    if (fused && getenv("GGML_AXERA_DECODE_PREFILL") && atoi(getenv("GGML_AXERA_DECODE_PREFILL")) != 0) {
        // Debug: run one token through the (validated) prefill kernel, one valid row, and take row 0.
        prefill_set & P = pfs[0];
        memset(P.X->vir, 0, (size_t) P.rows * H * sizeof(ggml_bf16_t));
        memcpy(P.X->vir, x, H * sizeof(ggml_bf16_t));
        P.X->flush();
        int32_t * pp = (int32_t *) P.POS->vir;
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < P.rows; ++r) {
                pp[s * P.rows + r] = p4 ? p4[s] : (s < 2 ? (int32_t) c : 0);
            }
        }
        P.POS->flush();
        ((int32_t *) INDPTR->vir)[0]  = 0;
        ((int32_t *) INDPTR->vir)[1]  = (int32_t) (c / PG);
        ((int32_t *) LASTLEN->vir)[0] = 1;
        ((int32_t *) LASTLEN->vir)[1] = 1;
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (c / PG);
        ((int32_t *) CLOC->vir)[1]    = (int32_t) (c % PG);
        *(int32_t *) VALID->vir       = 1;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        for (auto & cc : P.calls) {
            cc.k->run(cc.in, cc.out, "dense decode via prefill");
        }
        state_zero = false;
        const axera_cmm & po = P.OUT[P.result];
        po->invalidate(po->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir, out_h, H);
        return true;
    }
    if (fused) {
        // the upstream decode kernel takes the token's three M-RoPE positions, a one-page table and the cell in its
        // page; the recurrent and conv states are updated in place.
        memcpy(DIN->vir, x, H * sizeof(ggml_bf16_t));
        DIN->flush(DIN->vir, H * sizeof(ggml_bf16_t));
        int32_t * pp = (int32_t *) pfs[0].POS->vir;
        for (int s = 0; s < 3; ++s) {
            pp[s] = p4 ? p4[s] : s < 2 ? (int32_t) c : 0;
        }
        pfs[0].POS->flush(pfs[0].POS->vir, 3 * sizeof(int32_t));
        // The upstream metadata kernel: history pages are ceil(c/PG) complete
        // pages and the tail length is the last of them. At c == 0 there is no
        // history page at all, so the decode kernel never touches the pool's
        // untouched rows (masking them is not enough when they hold NaNs).
        const int hp         = (int) ((c + PG - 1) / PG);
        const int tail       = c == 0 ? 0 : (int) (c - (int64_t) (hp - 1) * PG);
        ((int32_t *) INDPTR->vir)[0] = 0;
        ((int32_t *) INDPTR->vir)[1] = hp;
        ((int32_t *) LASTLEN->vir)[0] = tail;
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (c / PG);
        ((int32_t *) CLOC->vir)[1]    = (int32_t) (c % PG);
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
            (*b)->flush();
        }
        auto run_dec = [&]() -> const axera_cmm & {
            axera_timer tl("layers (decode)");
            for (auto & cc : dec) {
                cc.k->run(cc.in, cc.out, "dense decode");
            }
            state_zero = false;
            return *dec_out;
        };
        const axera_cmm * res = &run_dec();
        (*res)->invalidate((*res)->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) (*res)->vir, out_h, H);
        return true;
    }
    (void) p4;  // the row's M-RoPE positions come from the graph's rope view (batch_rope)
    decode(c, x, out_h);
    return true;
}

void qwen38_27b::seq_rows(int batch_row0, int n, uint16_t * dst) {
    rows(batch_row0, n, dst);
}

bool qwen38_27b::seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) {
    (void) p4;  // the rows' M-RoPE positions come from the graph's rope view (batch_rope)
    return verify_rows(c, n, x, batch_row0, man.context);
}

// llama.cpp's sequence state (prompt cache / context checkpoint): core/kv_state's blob of the KV pages before the
// current cell, core/seqstate's restart points (the recurrent state) and, its tail, the MTP cache (gdn_mtp)
kv_state_format qwen38_27b::state_format() const {
    kv_state_format f;
    f.magic   = "AXW27S2";
    f.version = 4;
    f.pages   = kv_pages;
    f.pg      = PG;
    f.cells   = kv_cells;
    return f;
}

cmm_kv_pools qwen38_27b::state_pools() const {
    cmm_kv_pools p;
    if (fused) {
        p.add(KVC, KB, kv_blk_b, kv_blk_b / (size_t) kv_pages);
    } else {
        const int pna = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
        p.add(KVC, pna, kv_alloc, kv_b / NBLK);
        p.add(KVSC, pna, kvs_alloc, kvs_b / NBLK);  // the int8 caches' scales
    }
    return p;
}

size_t qwen38_27b::state_size(uint32_t flags) const {
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache: size and bytes must agree
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    return kv_state_size(state_format(), state_pools(), *seq, flags & state_flags_partial_only, &mtp);
}

size_t qwen38_27b::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    return kv_state_write(state_format(), p, *seq, dst, size, flags & state_flags_partial_only, &mtp);
}

size_t qwen38_27b::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    mtp.wait();
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    const size_t n = kv_state_read(state_format(), p, *seq, src, size, flags & state_flags_partial_only, &mtp);
    if (n) {
        state_zero = false;
    }
    return n;
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
            axera_npu_batch nb;  // the page's dependent calls, waited for once
            for (auto & c : P->calls) {
                nb.add(*c.k, c.in, c.out, "prefill");
            }
            nb.wait();
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
ggml_status qwen38_27b::compute(ggml_cgraph * g) {
    mtp.wait();
    const auto &  t       = names.of(g);
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return mtp.compute(mtp_env(), g);
    }
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen38_27b: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0);
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = llm_layers(t);
    const ggml_tensor * rope_node = llm_rope_node(g);
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
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    int64_t     c0 = 0;
    if (feed(names, man.context, x0, inp_pos, rope_node, T, "qwen38_27b", c0) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    std::vector<float> & h = h_buf;
    // common.cpp marks its warm-up batch with llama_set_dummy_batch(): its state is not needed. Drop it here, while
    // the batch is still being timed as warm-up, so the prompt's own zero state can skip the memset.
    const bool warmup = c0 == 0 && names.find("inp_mixed_slots_dummy");
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
// a draft context's MTP graph: token x_p (its embedding row) with the target's final-norm hidden of p - 1 through the
// MTP block at the draft context's cells; h_nextn (the block's shared head norm) and the logits of the output rows
}  // namespace

std::shared_ptr<model> make_qwen38_27b(const manifest & m) {
    return std::make_shared<qwen38_27b>(m);
}


gdn_mtp_env qwen38_27b::mtp_env() {
    gdn_mtp_env e;
    e.who          = "qwen38_27b";
    e.man          = &man;
    e.prm          = &prm;
    e.names        = &names;
    e.head         = &head;
    e.COS          = &COS;
    e.SIN          = &SIN;
    e.H            = H;
    e.ROT          = ROT;
    e.NBLK         = NBLK;
    e.kv_pages     = kv_pages;
    e.kv_cells     = kv_cells;
    e.target_bound = bound != nullptr;
    e.paged        = true;
    e.async        = true;
    for (auto & c : dec) {
        e.live.push_back(&c.in);
    }
    return e;
}

}  // namespace axera
