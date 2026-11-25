// Axera NPU backend: the Qwen3.5 hybrid models (3 gated delta net layers + 1 gated attention layer per block, dense
// MLP) on the upstream fused kernels (sgl_kernel_axera forward_qwen38_blocks): one prefill kernel (256 rows) and one
// decode kernel (1 row) per call of KB whole 4-layer blocks, their weights in banks [KB, ...]; NC = NB / KB calls per
// row batch (NB = L / 4 blocks). The single- and dual-core 2B / 4B packages (manifest model qwen38_27b, roles decode
// and prefill without decode.group; the 27B's own per-group kernels are models/qwen38_27b). No verification, no MTP.
// The state: the KV bank of each call [KB][pages][2][256][KV heads][256] bf16, the delta-net states and conv states
// phase-major [3][NB]; core/seqstate's restart points of them. One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/seqstate.h"
#include "../../core/params.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"
#include "../../ops/gdn_state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;  // cells per page = prefill rows per call

using call = npu_call;

class qwen35_fused final : public model, public seq_model, gdn_state {
  public:
    explicit qwen35_fused(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t      state_size(uint32_t flags) const override;
    size_t      state_write(void * dst, size_t size, uint32_t flags) override;
    size_t      state_read(const void * src, size_t size, uint32_t flags) override;
    int         last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override { state_save(dst); }
    void     seq_state_load(const uint8_t * src) override { state_load(src); }
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override { rows(batch_row0, n, dst); }

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    output_head      head;

    int    NB = 0, KB = 0, NC = 1;
    int    VH = 0, KHd = 0, KD = 0, VD = 0, REPf = 0;  // delta-net value / key heads, their dims (the kernel's)
    size_t kv_blk_b = 0;                              // bytes of one block's KV pool (all pages)
    axera_cmm         A2, DIN;  // the decode calls' x / out rotation (DIN the first call's input)
    const axera_cmm * dec_out = nullptr;

    void      init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_);
    axera_cmm bank_fused(const std::map<std::string, ggml_tensor *> & t, const io_desc & d);
    int       vtiled(int g) const { return (g % REPf) * KHd + g / REPf; }
    kv_state_format state_format() const;
    cmm_kv_pools    state_pools() const;
};

// read straight from the GGUF; the delta-net rows and columns the converter tiled are put back in checkpoint order.
axera_cmm qwen35_fused::bank_fused(const std::map<std::string, ggml_tensor *> & t, const io_desc & d) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera qwen35_fused: unexpected input %s\n", d.name.c_str());
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
            GGML_ABORT("axera qwen35_fused: %s not in the graph\n", nm.c_str());
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
            GGML_ABORT("axera qwen35_fused: NVFP4 tensor %s has no F32 .scale sidecar\n", w->name);
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
                GGML_ABORT("axera qwen35_fused: bank %s: %zu values of dtype %d, kernel block wants %zu bytes\n",
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
                GGML_ABORT("axera qwen35_fused: unexpected MLP input %s\n", d.name.c_str());
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
                    GGML_ABORT("axera qwen35_fused: unexpected attention input %s\n", d.name.c_str());
                }
            } else {
                GGML_ABORT("axera qwen35_fused: unexpected attention input %s\n", d.name.c_str());
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
                GGML_ABORT("axera qwen35_fused: unexpected delta-net input %s\n", d.name.c_str());
            }
        }
    }
    c->flush();
    for (const ggml_tensor * w : read) {
        axera_drop_mapped(w);
    }
    return c;
}

void qwen35_fused::init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_) {
    L    = L_;
    H    = H_;
    GGML_ASSERT(L % 4 == 0);
    NB = L / 4;
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");
    auto in_of = [&](const kernel & k, const std::string & n) -> const io_desc & { return gdn_io(k, n, "qwen35_fused"); };
    // This builder registers the projections separately (w_q, w_k, w_v, w_z, w_a, w_b); the 27B sgl manifests fuse
    // them into w_k/w_a. The two layouts are filled by different bank mappers, chosen from the compiled IO.
    if (pk.find_input("b0_attn_w_q") < 0) {
        GGML_ABORT("axera qwen35_fused: fused prefill needs the per-projection delta-net banks\n");
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
                        GGML_ABORT("axera qwen35_fused: fused: %s: unexpected input %s\n", k.desc->role.c_str(), n.c_str());
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
    gdn_page_set & P = pfs[0];
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
            GGML_ABORT("axera qwen35_fused: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
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
    GGML_LOG_INFO("axera qwen35_fused: fused %d layers, %d blocks, %d per call, banks %.2f GiB, state %.2f GiB, %lld cells\n",
                  L, NB, KB, bb / 1073741824.0,
                  (double) (KVC.size() * kv_b + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}

seq_caps qwen35_fused::seq_get_caps() const {
    seq_caps c;
    c.page           = PG;
    c.width          = H;
    c.image_bytes    = state_image_b;
    c.cells          = kv_cells;
    c.decode_at_zero = true;
    return c;
}

// src == nullptr: the zero state of a new sequence (and of the dummy warm-up batch's cleanup).
bool qwen35_fused::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    gdn_prefill_opts o;
    o.pos_pad   = 2;     // the padding rows repeat the last valid row's positions
    o.seed_out0 = true;  // the calls' rotation starts with OUT[0] as the block input
    o.lastlen2  = true;
    prefill_pages(c0, n, xb, p4, h_buf, batch0, snap, o);
    return true;
}

bool qwen35_fused::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
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

// llama.cpp's sequence state (prompt cache / context checkpoint): core/kv_state's blob of the KV pages before the
// current cell and core/seqstate's restart points (the recurrent state)
kv_state_format qwen35_fused::state_format() const {
    kv_state_format f;
    f.magic   = "AXW27S2";  // (the blobs of the qwen38_27b adapter these packages ran on)
    f.version = 4;
    f.pages   = kv_pages;
    f.pg      = PG;
    f.cells   = kv_cells;
    return f;
}

cmm_kv_pools qwen35_fused::state_pools() const {
    cmm_kv_pools p;
    p.add(KVC, KB, kv_blk_b, kv_blk_b / (size_t) kv_pages);
    return p;
}

size_t qwen35_fused::state_size(uint32_t flags) const {
    return KVC.empty() || !ST || !seq ? 0 : kv_state_size(state_format(), state_pools(), *seq, flags & state_flags_partial_only);
}

size_t qwen35_fused::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    return kv_state_write(state_format(), p, *seq, dst, size, flags & state_flags_partial_only);
}

size_t qwen35_fused::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    const size_t n = kv_state_read(state_format(), p, *seq, src, size, flags & state_flags_partial_only);
    if (n) {
        state_zero = false;
    }
    return n;
}

ggml_status qwen35_fused::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_fused: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0);
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    const int L_ = llm_layers(t);
    const ggml_tensor * rope_node = llm_rope_node(g);
    GGML_ASSERT(rope_node && "axera qwen35_fused: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, L_, H_);
        bound = w0->data;
    }
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    int64_t     c0 = 0;
    if (feed(names, man.context, x0, inp_pos, rope_node, T, "qwen35_fused", c0) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    std::vector<float> & h = h_buf;
    // common.cpp marks its warm-up batch with llama_set_dummy_batch(): its state is not needed. Drop it here, while
    // the batch is still being timed as warm-up, so the prompt's own zero state can skip the memset.
    const bool warmup = c0 == 0 && names.find("inp_mixed_slots_dummy");
    axera_timer th("output head");
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
        GGML_ABORT("axera qwen35_fused: the lm_head IO is not the upstream rmsnorm_gemm head\n");
    }
    head.run(g, names, h, T);
    if (warmup) {  // the warm-up sequence must not make the first real prompt clear the state
        seq->forget();
        seq_state_load(nullptr);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_fused(const manifest & m) {
    return std::make_shared<qwen35_fused>(m);
}

}  // namespace axera
