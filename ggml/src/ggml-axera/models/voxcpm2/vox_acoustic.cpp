// Axera NPU backend: VoxCPM2 acoustic stages.
//
// The three acoustic modules are whole-model AXModels built by ax-dsl-kernels
// (models/voxcpm2/{dit,locenc,vae}) with the layer loop in the runtime and
// MXFP8 weight banks.  The host side here only binds the GGUF weights, carries
// the AudioVAE causal state between patches and keeps the Euler loop outside.

#include "vox_acoustic.h"

#include "../../core/manifest.h"
#include "../../core/mx.h"
#include "../../core/quant.h"
#include "../../core/runtime.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace axera {
namespace {

constexpr int64_t kPatch = 4;
constexpr int64_t kFeat = 64;
constexpr int64_t kHidden = 1024;
constexpr int64_t kVaeSamples = 7680;
constexpr int64_t kRopeDim = 128;

struct role {
    kernel *                              k = nullptr;
    std::vector<axera_cmm>                bufs;
    std::vector<AX_ENGINE_IO_BUFFER_T>    in, out;
    std::map<std::string, int>            idx;

    bool bind(kernel * kk) {
        k = kk;
        idx.clear();
        bufs.resize(k->info->nInputSize);
        in.resize(k->info->nInputSize);
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            bufs[i] = axera_cmm_alloc(k->info->pInputs[i].nSize);
            in[i]   = axera_io(bufs[i], k->info->pInputs[i].nSize);
            idx[k->info->pInputs[i].pName] = (int) i;
        }
        out.resize(k->info->nOutputSize);
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            outputs.emplace_back(axera_cmm_alloc(k->info->pOutputs[i].nSize));
            out[i] = axera_io(outputs.back(), k->info->pOutputs[i].nSize);
        }
        return true;
    }
    std::vector<axera_cmm> outputs;

    int find(const std::string & name) const {
        auto it = idx.find(name);
        return it == idx.end() ? -1 : it->second;
    }
    axera_cmm & buf(const std::string & name) { return bufs[find(name)]; }
};

std::vector<float> flat_f32(const ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    const auto *       tr = ggml_get_type_traits(t->type);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, out.size() * sizeof(float));
    } else {
        GGML_ASSERT(tr->to_float);
        tr->to_float(t->data, out.data(), (int64_t) out.size());
    }
    return out;
}

void put_bf16(axera_cmm & b, const float * src, size_t n) {
    GGML_ASSERT(n * 2 <= b->size);
    auto * dst = (ggml_bf16_t *) b->vir;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ggml_fp32_to_bf16(src[i]);
    }
    b->flush(b->vir, n * 2);
}

void put_f16(axera_cmm & b, const float * src, size_t n) {
    GGML_ASSERT(n * 2 <= b->size);
    auto * dst = (ggml_fp16_t *) b->vir;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }
    b->flush(b->vir, n * 2);
}

void put_mxfp8(axera_cmm & q, axera_cmm & s, const float * src, int64_t rows, int64_t cols) {
    GGML_ASSERT(cols % AXERA_MX_GROUP == 0);
    GGML_ASSERT((size_t) rows * cols <= q->size);
    GGML_ASSERT((size_t) rows * (cols / AXERA_MX_GROUP) <= s->size);
    auto * qd = (uint8_t *) q->vir;
    auto * sd = (uint8_t *) s->vir;
    for (int64_t r = 0; r < rows; ++r) {
        axera_quantize_row_mxfp8_f32(src + r * cols, qd + r * cols, sd + r * (cols / AXERA_MX_GROUP), cols);
    }
    q->flush(q->vir, (size_t) rows * cols);
    s->flush(s->vir, (size_t) rows * (cols / AXERA_MX_GROUP));
}

float bf16_to_f32(uint16_t v) { return axera_bf16_to_f32(v); }

struct acoustic final : vox_acoustic {
    role dit, locenc, vae;

    // DiT runtime inputs and output
    axera_cmm dit_x, dit_lm, dit_res, dit_cond, dit_t, dit_vel, dit_cos, dit_sin, dit_inv_freq;
    // LocEnc runtime inputs and output
    axera_cmm le_x, le_cos, le_sin, le_out;
    // VAE runtime input, output and the 15 state pairs
    axera_cmm vae_x, vae_out;
    std::vector<std::pair<axera_cmm, axera_cmm>> vae_state;  // (a, b), roles swap each call
    std::vector<std::string>                     vae_state_in, vae_state_out;
    bool                                         vae_first = true;

    std::string err;

    bool fail(const std::string & msg) {
        err = msg;
        return false;
    }

    const ggml_tensor * need(const std::map<std::string, ggml_tensor *> & t, const std::string & n) {
        auto it = t.find(n);
        if (it == t.end()) {
            fail("missing GGUF tensor " + n);
            return nullptr;
        }
        return it->second;
    }

    // [N, K] matrix (ggml ne0 = K, ne1 = N) -> one mxfp8 pair input.
    bool mxfp8_mat(const std::map<std::string, ggml_tensor *> & t, role & r, const std::string & wn,
                   const std::string & sn, const std::string & src) {
        const ggml_tensor * w = need(t, src);
        if (!w) return false;
        if (r.find(wn) < 0 || r.find(sn) < 0) return fail("kernel has no mxfp8 input " + wn);
        const int64_t N = ggml_nrows(w), K = w->ne[0];
        auto          f = flat_f32(w);
        put_mxfp8(r.buf(wn), r.buf(sn), f.data(), N, K);
        return true;
    }

    bool bf16_vec(const std::map<std::string, ggml_tensor *> & t, role & r, const std::string & dst,
                  const std::string & src) {
        const ggml_tensor * w = need(t, src);
        if (!w) return false;
        auto f = flat_f32(w);
        put_bf16(r.buf(dst), f.data(), f.size());
        return true;
    }

    // Banked transformer weights: b<phase>_<kind>_<field> holds three blocks,
    // row b of phase p is layer 4*b + p of the GGUF.
    bool bank(const std::map<std::string, ggml_tensor *> & t, role & r, const std::string & prefix,
              const std::string & kind, const std::string & field, const std::string & gguf_field) {
        const int64_t N = ggml_nrows(need(t, prefix + "0." + gguf_field));
        int64_t       K = 0;
        for (int p = 0; p < 4; ++p) {
            for (int b = 0; b < 3; ++b) {
                const ggml_tensor * w = need(t, prefix + std::to_string(4 * b + p) + "." + gguf_field);
                if (!w) return false;
                if (K == 0) {
                    K = w->ne[0];
                } else if (K != w->ne[0] || N != ggml_nrows(w)) {
                    return fail("bank shape mismatch in " + std::string(w->name));
                }
            }
        }
        if (K == 0) return fail("empty bank");
        std::vector<float> all((size_t) 3 * N * K);
        for (int p = 0; p < 4; ++p) {
            const std::string wn = "b" + std::to_string(p) + "_" + kind + "_" + field;
            const std::string sn = "b" + std::to_string(p) + "_" + kind + "_w_scale" +
                                   (field == "w_q" ? "_q" : field == "w_k" ? "_k" : field == "w_v" ? "_v" :
                                    field == "w_o" ? "_o" : field == "w1" ? "1" : field == "w2" ? "2" : "3");
            for (int b = 0; b < 3; ++b) {
                const ggml_tensor * w = t.at(prefix + std::to_string(4 * b + p) + "." + gguf_field);
                auto                f = flat_f32(w);
                memcpy(all.data() + (size_t) (b * N) * K, f.data(), (size_t) N * K * sizeof(float));
            }
            if (r.find(wn) < 0 || r.find(sn) < 0) return fail("kernel has no bank input " + wn);
            put_mxfp8(r.buf(wn), r.buf(sn), all.data(), 3 * N, K);
        }
        return true;
    }

    bool bank_vec(const std::map<std::string, ggml_tensor *> & t, role & r, const std::string & prefix,
                  const std::string & kind, const std::string & field, const std::string & gguf_field) {
        std::vector<float> all(3 * kHidden);
        for (int p = 0; p < 4; ++p) {
            for (int b = 0; b < 3; ++b) {
                const ggml_tensor * w = need(t, prefix + std::to_string(4 * b + p) + "." + gguf_field);
                if (!w) return false;
                auto f = flat_f32(w);
                memcpy(all.data() + (size_t) b * kHidden, f.data(), (size_t) kHidden * sizeof(float));
            }
            const std::string wn = "b" + std::to_string(p) + "_" + kind + "_" + field;
            if (r.find(wn) < 0) return fail("kernel has no bank vector " + wn);
            put_bf16(r.buf(wn), all.data(), all.size());
        }
        return true;
    }

    bool dit_weights(const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & factors);
    bool locenc_weights(const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & factors);
    bool vae_weights(const std::map<std::string, ggml_tensor *> & t);
    bool vae_states();
};

void rope_tables(int rows, const std::vector<float> & factors, float * cos, float * sin) {
    GGML_ASSERT((int) factors.size() == kRopeDim / 2);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < kRopeDim / 2; ++i) {
            const float inv = 1.0f / std::pow(10000.0f, (float) (2 * i) / (float) kRopeDim) / factors[i];
            const float a   = (float) r * inv;
            cos[r * (kRopeDim / 2) + i] = std::cos(a);
            sin[r * (kRopeDim / 2) + i] = std::sin(a);
        }
    }
}

bool acoustic::dit_weights(const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & factors) {
    struct g {
        const char * name, * tensor;
    };
    const g globals[] = {
        { "lm_w", "projections.lm_to_dit_proj.weight" },
        { "res_w", "projections.res_to_dit_proj.weight" },
        { "time_w1", "locdit.time_mlp.linear_1.weight" },
        { "time_w2", "locdit.time_mlp.linear_2.weight" },
        { "dtime_w1", "locdit.delta_time_mlp.linear_1.weight" },
        { "dtime_w2", "locdit.delta_time_mlp.linear_2.weight" },
        { "in_w", "locdit.in_proj.weight" },
        { "cond_w", "locdit.cond_proj.weight" },
        { "out_w", "locdit.out_proj.weight" },
    };
    for (const auto & e : globals) {
        const std::string w = e.name;
        const size_t      p = w.rfind("_w");
        const std::string s = p == std::string::npos ? w + "_s" : w.substr(0, p) + "_s" + w.substr(p + 2);
        if (!mxfp8_mat(t, dit, w, s, e.tensor)) return false;
    }
    const g biases[] = {
        { "lm_b", "projections.lm_to_dit_proj.bias" },
        { "res_b", "projections.res_to_dit_proj.bias" },
        { "in_b", "locdit.in_proj.bias" },
        { "cond_b", "locdit.cond_proj.bias" },
        { "out_b", "locdit.out_proj.bias" },
        { "time_b1", "locdit.time_mlp.linear_1.bias" },
        { "time_b2", "locdit.time_mlp.linear_2.bias" },
        { "dtime_b1", "locdit.delta_time_mlp.linear_1.bias" },
        { "dtime_b2", "locdit.delta_time_mlp.linear_2.bias" },
        { "final_norm", "locdit.norm.weight" },
    };
    for (const auto & e : biases) {
        if (!bf16_vec(t, dit, e.name, e.tensor)) return false;
    }
    struct f {
        const char * field, * tensor;
    };
    const f attn[] = {
        { "w_q", "attn_q.weight" }, { "w_k", "attn_k.weight" }, { "w_v", "attn_v.weight" },
        { "w_o", "attn_output.weight" },
    };
    const f mlp[] = {
        { "w1", "ffn_gate.weight" }, { "w2", "ffn_up.weight" }, { "w3", "ffn_down.weight" },
    };
    for (const auto & e : attn) {
        if (!bank(t, dit, "locdit.blk.", "attn", e.field, e.tensor)) return false;
    }
    for (const auto & e : mlp) {
        if (!bank(t, dit, "locdit.blk.", "mlp", e.field, e.tensor)) return false;
    }
    if (!bank_vec(t, dit, "locdit.blk.", "attn", "w_norm_attn", "attn_norm.weight")) return false;
    if (!bank_vec(t, dit, "locdit.blk.", "mlp", "w_norm", "ffn_norm.weight")) return false;
    rope_tables(22, factors, (float *) dit_cos->vir, (float *) dit_sin->vir);
    dit_cos->flush();
    dit_sin->flush();
    return true;
}

bool acoustic::locenc_weights(const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & factors) {
    if (!mxfp8_mat(t, locenc, "in_w", "in_s", "locenc.in_proj.weight")) return false;
    if (!bf16_vec(t, locenc, "in_b", "locenc.in_proj.bias")) return false;
    if (!bf16_vec(t, locenc, "cls_token", "locenc.cls_token.weight")) return false;
    if (!bf16_vec(t, locenc, "final_norm", "locenc.norm.weight")) return false;
    struct f {
        const char * field, * tensor;
    };
    const f attn[] = {
        { "w_q", "attn_q.weight" }, { "w_k", "attn_k.weight" }, { "w_v", "attn_v.weight" },
        { "w_o", "attn_output.weight" },
    };
    const f mlp[] = {
        { "w1", "ffn_gate.weight" }, { "w2", "ffn_up.weight" }, { "w3", "ffn_down.weight" },
    };
    for (const auto & e : attn) {
        if (!bank(t, locenc, "locenc.blk.", "attn", e.field, e.tensor)) return false;
    }
    for (const auto & e : mlp) {
        if (!bank(t, locenc, "locenc.blk.", "mlp", e.field, e.tensor)) return false;
    }
    if (!bank_vec(t, locenc, "locenc.blk.", "attn", "w_norm_attn", "attn_norm.weight")) return false;
    if (!bank_vec(t, locenc, "locenc.blk.", "mlp", "w_norm", "ffn_norm.weight")) return false;
    rope_tables(5, factors, (float *) le_cos->vir, (float *) le_sin->vir);
    le_cos->flush();
    le_sin->flush();
    return true;
}

bool acoustic::vae_weights(const std::map<std::string, ggml_tensor *> & t) {
    auto raw = [&](const std::string & n) {
        const ggml_tensor * w = need(t, n);
        return w ? flat_f32(w) : std::vector<float>();
    };
    auto set_mx = [&](const std::string & wn, const std::string & sn, const std::vector<float> & f, int64_t N,
                      int64_t K) {
        if (f.empty()) return false;
        if (vae.find(wn) < 0 || vae.find(sn) < 0) return fail("VAE kernel has no input " + wn);
        put_mxfp8(vae.buf(wn), vae.buf(sn), f.data(), N, K);
        return true;
    };
    auto set_bf = [&](const std::string & n, const std::vector<float> & f) {
        if (f.empty() || vae.find(n) < 0) return fail("VAE kernel has no input " + n);
        put_bf16(vae.buf(n), f.data(), f.size());
        return true;
    };
    auto set_f16 = [&](const std::string & n, const std::vector<float> & f) {
        if (f.empty() || vae.find(n) < 0) return fail("VAE kernel has no input " + n);
        put_f16(vae.buf(n), f.data(), f.size());
        return true;
    };

    // model.0 depthwise [64, 1, 7] -> w0 [7, 64]
    {
        const ggml_tensor * w = need(t, "audio_vae.decoder.model.0");
        if (!w) return false;
        const int64_t C = w->ne[2], K = w->ne[0];
        auto          f = flat_f32(w);
        std::vector<float> out((size_t) K * C);
        for (int64_t c = 0; c < C; ++c)
            for (int64_t k = 0; k < K; ++k) out[(size_t) k * C + c] = f[(size_t) c * K + k];
        if (!set_f16("w0", out)) return false;
        if (!set_bf("b0", raw("audio_vae.decoder.model.0.bias"))) return false;
    }
    // model.1 [2048, 64, 1]
    {
        const ggml_tensor * w = need(t, "audio_vae.decoder.model.1");
        if (!w) return false;
        auto f = flat_f32(w);
        if (!set_mx("w1", "s1", f, w->ne[2], w->ne[1])) return false;
        if (!set_bf("b1", raw("audio_vae.decoder.model.1.bias"))) return false;
    }
    // model.9 final conv [1, c, 7] -> wf [7, c]
    {
        const ggml_tensor * w = need(t, "audio_vae.decoder.model.9");
        if (!w) return false;
        const int64_t C = w->ne[1], K = w->ne[0];
        auto          f = flat_f32(w);
        std::vector<float> out((size_t) K * C);
        for (int64_t c = 0; c < C; ++c)
            for (int64_t k = 0; k < K; ++k) out[(size_t) k * C + c] = f[(size_t) c * K + k];
        if (!set_f16("wf", out)) return false;
        if (!set_bf("bf", raw("audio_vae.decoder.model.9.bias"))) return false;
    }
    if (!set_bf("alpha_f", raw("audio_vae.decoder.model.8.alpha"))) return false;

    for (int bi = 0; bi < 6; ++bi) {
        const std::string  n = std::to_string(bi + 2);
        const std::string  p = "audio_vae.decoder.model." + n + ".block.";
        static const int64_t chans[6]  = { 1024, 512, 256, 128, 64, 32 };
        static const int64_t ichs[6]   = { 2048, 1024, 512, 256, 128, 64 };
        static const int64_t rates[6]  = { 8, 6, 5, 2, 2, 2 };
        const int64_t        oc     = chans[bi];
        const int64_t        ic     = ichs[bi];
        const int64_t        stride = rates[bi];
        if (!set_bf("alpha" + n, raw(p + "0.alpha"))) return false;
        // transposed conv [ic, oc, 2*stride] -> lo/hi [stride*oc, ic]
        {
            const ggml_tensor * w = need(t, p + "1");
            if (!w) return false;
            const int64_t ks = w->ne[0];
            auto          f = flat_f32(w);
            std::vector<float> lo((size_t) stride * oc * ic), hi((size_t) stride * oc * ic);
            for (int64_t i = 0; i < ic; ++i)
                for (int64_t o = 0; o < oc; ++o)
                    for (int64_t q = 0; q < stride; ++q) {
                        lo[(size_t) (q * oc + o) * ic + i] = f[(size_t) (i * oc + o) * ks + q];
                        hi[(size_t) (q * oc + o) * ic + i] = f[(size_t) (i * oc + o) * ks + stride + q];
                    }
            if (!set_mx("t" + n + "_lo", "t" + n + "_slo", lo, stride * oc, ic)) return false;
            if (!set_mx("t" + n + "_hi", "t" + n + "_shi", hi, stride * oc, ic)) return false;
            if (!set_bf("t" + n + "_b", raw(p + "1.bias"))) return false;
        }
        for (int ri = 0; ri < 3; ++ri) {
            const std::string rp = p + std::to_string(ri + 2) + ".block.";
            const std::string rn = "r" + n + std::to_string(ri);
            if (!set_bf(rn + "_a1", raw(rp + "0.alpha"))) return false;
            {
                const ggml_tensor * w = need(t, rp + "1");
                if (!w) return false;
                const int64_t K = w->ne[0];
                auto          f = flat_f32(w);
                std::vector<float> out((size_t) K * oc);
                for (int64_t c = 0; c < oc; ++c)
                    for (int64_t k = 0; k < K; ++k) out[(size_t) k * oc + c] = f[(size_t) c * K + k];
                if (!set_f16(rn + "_dw", out)) return false;
                if (!set_bf(rn + "_dwb", raw(rp + "1.bias"))) return false;
            }
            if (!set_bf(rn + "_a2", raw(rp + "2.alpha"))) return false;
            {
                const ggml_tensor * w = need(t, rp + "3");
                if (!w) return false;
                auto f = flat_f32(w);
                if (!set_mx(rn + "_pw", rn + "_pws", f, oc, oc)) return false;
                if (!set_bf(rn + "_pwb", raw(rp + "3.bias"))) return false;
            }
        }
        {
            auto sc = raw("audio_vae.decoder.sr_cond_model." + n + ".scale_embed");
            auto bi_ = raw("audio_vae.decoder.sr_cond_model." + n + ".bias_embed");
            if (sc.size() < (size_t) 4 * ic || bi_.size() < (size_t) 4 * ic) return fail("sr_cond shape");
            std::vector<float> s(sc.begin() + 3 * ic, sc.begin() + 4 * ic);
            std::vector<float> b(bi_.begin() + 3 * ic, bi_.begin() + 4 * ic);
            if (!set_bf("sr_scale_" + n, s)) return false;
            if (!set_bf("sr_bias_" + n, b)) return false;
        }
    }
    return true;
}

bool acoustic::vae_states() {
    // Every `Xo` input is the output twin of the state input `X`; the initial
    // depthwise pair is si0/so0 (the only name that is not just + "o").
    for (const auto & name : vae.idx) {
        const std::string & n = name.first;
        if (n == "so0") {
            if (vae.find("si0") >= 0) {
                vae_state_out.push_back(n);
                vae_state_in.push_back("si0");
            }
            continue;
        }
        if (n.size() > 1 && n.back() == 'o') {
            std::string base = n.substr(0, n.size() - 1);
            if (vae.find(base) >= 0) {
                vae_state_out.push_back(n);
                vae_state_in.push_back(base);
            }
        }
    }
    if (vae_state_in.size() != 26) return fail("VAE state pairs: expected 26, got " + std::to_string(vae_state_in.size()));
    for (size_t i = 0; i < vae_state_in.size(); ++i) {
        const int i0 = vae.find(vae_state_in[i]);
        const int i1 = vae.find(vae_state_out[i]);
        if (vae.bufs[i0]->size != vae.bufs[i1]->size) return fail("VAE state size mismatch");
        vae_state.emplace_back(axera_cmm_alloc(vae.bufs[i0]->size), axera_cmm_alloc(vae.bufs[i0]->size));
        // The kernel's in/out IO entries are re-pointed at a rotating pair.
    }
    return true;
}

}  // namespace

vox_acoustic * vox_acoustic_open(const std::map<std::string, ggml_tensor *> & tensors,
                                 const std::vector<float> & factors, std::string * err) {
    auto a = std::make_unique<acoustic>();
    const manifest * m = kernels();
    if (!m) {
        if (err) *err = "no manifest";
        return nullptr;
    }
    if (!m->has("acoustic.dit") || !m->has("acoustic.locenc") || !m->has("acoustic.vae")) {
        if (err) *err = "manifest has no acoustic.dit / acoustic.locenc / acoustic.vae";
        return nullptr;
    }
    if (!a->dit.bind(&m->get("acoustic.dit")) || !a->locenc.bind(&m->get("acoustic.locenc")) ||
        !a->vae.bind(&m->get("acoustic.vae"))) {
        if (err) *err = "kernel load failed";
        return nullptr;
    }
    auto mk = [](axera_cmm & b, uint32_t bytes) { b = axera_cmm_alloc(bytes); };
    mk(a->dit_x, kPatch * kFeat * 2);
    mk(a->dit_lm, 2048 * 2);
    mk(a->dit_res, 2048 * 2);
    mk(a->dit_cond, kPatch * kFeat * 2);
    mk(a->dit_t, 4);
    mk(a->dit_vel, kPatch * kFeat * 2);
    mk(a->dit_cos, 22 * (kRopeDim / 2) * 4);
    mk(a->dit_sin, 22 * (kRopeDim / 2) * 4);
    mk(a->dit_inv_freq, (kHidden / 2) * 4);
    {
        auto * f = (float *) a->dit_inv_freq->vir;
        for (int i = 0; i < kHidden / 2; ++i) {
            f[i] = std::exp((float) i * (-std::log(10000.0f) / (float) (kHidden / 2 - 1)));
        }
        a->dit_inv_freq->flush();
    }
    mk(a->le_x, kPatch * kFeat * 2);
    mk(a->le_cos, 5 * (kRopeDim / 2) * 4);
    mk(a->le_sin, 5 * (kRopeDim / 2) * 4);
    mk(a->le_out, kHidden * 2);
    mk(a->vae_x, kPatch * kFeat * 2);
    mk(a->vae_out, kVaeSamples * 4);

    a->dit.in[a->dit.find("x")]         = axera_io(a->dit_x);
    a->dit.in[a->dit.find("lm_hidden")] = axera_io(a->dit_lm, 2048 * 2);
    a->dit.in[a->dit.find("res_hidden")] = axera_io(a->dit_res, 2048 * 2);
    a->dit.in[a->dit.find("cond")]      = axera_io(a->dit_cond);
    a->dit.in[a->dit.find("t")]         = axera_io(a->dit_t, 4);
    a->dit.in[a->dit.find("cos")]       = axera_io(a->dit_cos);
    a->dit.in[a->dit.find("sin")]       = axera_io(a->dit_sin);
    a->dit.in[a->dit.find("inv_freq")]  = axera_io(a->dit_inv_freq, (kHidden / 2) * 4);
    a->dit.out[0]                       = axera_io(a->dit_vel);

    a->locenc.in[a->locenc.find("x")]   = axera_io(a->le_x);
    a->locenc.in[a->locenc.find("cos")] = axera_io(a->le_cos);
    a->locenc.in[a->locenc.find("sin")] = axera_io(a->le_sin);
    a->locenc.out[0]                    = axera_io(a->le_out);

    a->vae.in[a->vae.find("x")] = axera_io(a->vae_x);
    a->vae.out[0]               = axera_io(a->vae_out);

    if (!a->dit_weights(tensors, factors) || !a->locenc_weights(tensors, factors) ||
        !a->vae_weights(tensors) || !a->vae_states()) {
        if (err) *err = a->err.empty() ? "weight binding failed" : a->err;
        return nullptr;
    }
    // A first VAE call sees zero state; the pair roles rotate afterwards.
    for (size_t i = 0; i < a->vae_state.size(); ++i) {
        a->vae.in[a->vae.find(a->vae_state_in[i])]  = axera_io(a->vae_state[i].first);
        a->vae.in[a->vae.find(a->vae_state_out[i])] = axera_io(a->vae_state[i].second);
    }
    return a.release();
}

void vox_acoustic_close(vox_acoustic * a) { delete static_cast<acoustic *>(a); }

bool vox_acoustic_dit(vox_acoustic * a_, const float * x, const float * lm, const float * res,
                      const float * cond, float t, float * vel) {
    auto * a = static_cast<acoustic *>(a_);
    if (!a) return false;
    auto put = [](axera_cmm & b, const float * src, size_t n) { put_bf16(b, src, n); };
    // Runtime arrays are [64, 4]; the kernel consumes [4, 64].
    float tmp[kPatch * kFeat];
    for (int c = 0; c < kFeat; ++c)
        for (int p = 0; p < kPatch; ++p) tmp[p * kFeat + c] = x[c * kPatch + p];
    put(a->dit_x, tmp, kPatch * kFeat);
    put(a->dit_lm, lm, 2048);
    put(a->dit_res, res, 2048);
    for (int c = 0; c < kFeat; ++c)
        for (int p = 0; p < kPatch; ++p) tmp[p * kFeat + c] = cond[c * kPatch + p];
    put(a->dit_cond, tmp, kPatch * kFeat);
    {
        auto * tbuf = (float *) a->dit_t->vir;
        tbuf[0] = t;
        a->dit_t->flush(a->dit_t->vir, 4);
    }
    a->dit.k->run(a->dit.in, a->dit.out, "voxcoustic dit");
    a->dit_vel->invalidate(a->dit_vel->vir, kPatch * kFeat * 2);
    const auto * o = (const ggml_bf16_t *) a->dit_vel->vir;
    for (int p = 0; p < kPatch; ++p)
        for (int c = 0; c < kFeat; ++c) vel[c * kPatch + p] = bf16_to_f32(o[p * kFeat + c].bits);
    return true;
}

bool vox_acoustic_locenc(vox_acoustic * a_, const float * x, float * hidden) {
    auto * a = static_cast<acoustic *>(a_);
    if (!a) return false;
    float tmp[kPatch * kFeat];
    for (int c = 0; c < kFeat; ++c)
        for (int p = 0; p < kPatch; ++p) tmp[p * kFeat + c] = x[c * kPatch + p];
    put_bf16(a->le_x, tmp, kPatch * kFeat);
    a->locenc.k->run(a->locenc.in, a->locenc.out, "voxcoustic locenc");
    a->le_out->invalidate(a->le_out->vir, kHidden * 2);
    const auto * o = (const ggml_bf16_t *) a->le_out->vir;
    for (int i = 0; i < kHidden; ++i) hidden[i] = bf16_to_f32(o[i].bits);
    return true;
}

bool vox_acoustic_vae(vox_acoustic * a_, const float * patch, float * pcm) {
    auto * a = static_cast<acoustic *>(a_);
    if (!a) return false;
    float tmp[kPatch * kFeat];
    for (int c = 0; c < kFeat; ++c)
        for (int p = 0; p < kPatch; ++p) tmp[p * kFeat + c] = patch[c * kPatch + p];
    put_bf16(a->vae_x, tmp, kPatch * kFeat);
    if (!a->vae_first) {
        for (size_t i = 0; i < a->vae_state.size(); ++i) {
            std::swap(a->vae.in[a->vae.find(a->vae_state_in[i])], a->vae.in[a->vae.find(a->vae_state_out[i])]);
        }
    }
    a->vae_first = false;
    a->vae.k->run(a->vae.in, a->vae.out, "voxcoustic vae");
    a->vae_out->invalidate(a->vae_out->vir, kVaeSamples * 4);
    memcpy(pcm, a->vae_out->vir, kVaeSamples * 4);
    return true;
}

void vox_acoustic_vae_reset(vox_acoustic * a_) {
    auto * a = static_cast<acoustic *>(a_);
    if (!a) return;
    a->vae_first = true;
    for (size_t i = 0; i < a->vae_state.size(); ++i) {
        memset(a->vae_state[i].first->vir, 0, a->vae_state[i].first->size);
        memset(a->vae_state[i].second->vir, 0, a->vae_state[i].second->size);
        a->vae_state[i].first->flush();
        a->vae_state[i].second->flush();
        a->vae.in[a->vae.find(a->vae_state_in[i])]  = axera_io(a->vae_state[i].first);
        a->vae.in[a->vae.find(a->vae_state_out[i])] = axera_io(a->vae_state[i].second);
    }
}

}  // namespace axera
