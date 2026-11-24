#include "gdn_bank.h"
#include "llm.h"

#include <cmath>
#include <cstring>
#include <regex>
#include <vector>

namespace axera {

axera_cmm gdn_prefill_bank(const gdn_bank_shape & s, const std::map<std::string, ggml_tensor *> & t, const io_desc & d,
                           const rope & rp) {
    const int  L = s.L, ROT = s.ROT, VH = s.VH;
    const bool fp8   = s.fp8;
    auto       tiled = [&](int g) { return s.tiled(g); };
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera %s: prefill: unexpected input %s\n", s.who, d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const bool        cs    = n == "cos" || n == "sin";
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(d.shape.size() >= 2 && nb > 0 && (nb == L / 4 || cs) && d.bytes % nb == 0);
    axera_cmm     c  = axera_cmm_alloc_bank(d.bytes);
    const size_t  bb = d.bytes / nb;  // bytes per block
    std::vector<const ggml_tensor *> read;  // the GGUF tensors of one block: their mapped pages dropped right after it
    auto          T  = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera %s: %s not in the graph\n", s.who, nm.c_str());
        }
        axera_page_in(t.at(nm));
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        axera_mapped_drop_guard drop(read, true);
        const int il  = 4 * b + phase;
        uint8_t * dst = (uint8_t *) c->vir + (size_t) b * bb;
        auto put_f32 = [&](const std::vector<float> & v, float scale = 1.0f) {
            if (v.size() * sizeof(float) != bb && v.size() * sizeof(ggml_bf16_t) != bb) {
                GGML_ABORT("axera %s: bank %s: %zu values of dtype %d, the kernel's block takes %zu bytes (%zu / %d)\n",
                           s.who, d.name.c_str(), v.size(), d.dtype, bb, (size_t) d.bytes, nb);
            }
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
        auto rows = [&](const mx_weight & w, int64_t r0, int64_t nr, bool scales, int64_t vrow = -1, int64_t hr = 0) {
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
        auto cols = [&](const mx_weight & w, bool scales) {  // value heads of the input (columns) in HF order
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
                const mx_weight w(T(il, W.at(n.substr(1))), fp8);
                rows(w, 0, w.N, false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                const mx_weight w(T(il, W.at(n.substr(7))), fp8);
                rows(w, 0, w.N, true);
            } else {
                GGML_ABORT("axera %s: prefill: unexpected MLP input %s\n", s.who, d.name.c_str());
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
                const mx_weight w(T(il, W.at(p)), fp8);
                rows(w, 0, w.N, n != "w_" + p);
            } else {
                GGML_ABORT("axera %s: prefill: unexpected attention input %s\n", s.who, d.name.c_str());
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
                const mx_weight     w(T(il, "attn_qkv.weight"), fp8);
                const int64_t nr = d.shape[1];  // rows q, k (nr each), then v (the rest)
                if (p == 'v') {
                    rows(w, w.N - nr, nr, scales, w.N - nr, nr / VH);
                } else {
                    rows(w, p == 'q' ? 0 : nr, nr, scales);
                }
            } else if (p == 'z') {
                const mx_weight w(T(il, "attn_gate.weight"), fp8);
                rows(w, 0, w.N, scales, 0, w.N / VH);
            } else if (p == 'a' || p == 'b') {  // small GGUF tensors: quantized to the bank's MX format like every projection
                const mx_weight w(T(il, p == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight"), fp8);
                rows(w, 0, w.N, scales, 0, 1);
            } else if (p == 'o') {
                const mx_weight w(T(il, "ssm_out.weight"), fp8);
                cols(w, scales);
            } else {
                GGML_ABORT("axera %s: prefill: unexpected delta-net input %s\n", s.who, d.name.c_str());
            }
        } else {
            GGML_ABORT("axera %s: prefill: unexpected delta-net input %s\n", s.who, d.name.c_str());
        }
    }
    c->flush();
    return c;
}

}  // namespace axera
