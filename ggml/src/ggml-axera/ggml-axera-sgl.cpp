// Axera NPU backend: the whole text model in the fused kernels of sglang_axera_opt (sgl-kernel-axera models/qwen38_27b
// prefill for Qwen3.8-27B, models/qwen36_35b_a3b decode for Qwen3.5-35B-A3B), GGML_AXERA_SGL=<dir> with the qwen35
// template
//
// One kernel call runs all layers (a runtime loop over blocks of three GDN and one gated attention layer, each with its
// MLP / MoE) on 256 rows (prefill) or 1 row (decode). Weights, recurrent state and the paged KV cache (pages of 256
// cells) are the kernels' inputs; <dir> holds the weight banks as the offline packers wrote them (pack_sgl27.py /
// pack_sgl35.py: <name>.bin of every bank, manifest.txt "<name> <bytes> <state>"), the kernels sgl<model>_r<rows>_v<valid>
// _h<history>_npu4.axmodel are in GGML_AXERA_KERNEL_DIR (loaded with the others at initialization). The state banks (GDN
// state pools, KV pages) are allocated here and cleared at position 0.
// The template (ggml-axera-q35-run.cpp) runs a prompt's whole 256-position chunks through the chunk kernels (27B), and
// single tokens through the decode kernel (35B); the state moves between the kernels' banks and the template's caches
// (axera_sgl_sync) whenever the other side runs next. The 27B decode kernel and the chunk kernels of fewer valid rows
// are not used (the former is slower than the template's, the latter compute NaN). (sgl-kernel-axera flash_attn.py
// project_token's NVFP4 path drops the banks offset, every block then reading block 0's attention Q/K/V weights: the
// decode kernels are built with it fixed, `if not banks and ...`.)

#include "ggml-axera-q35.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <set>
#include <fstream>
#include <tuple>
#include <sstream>

struct axera_sgl {
    struct variant {
        ggml_axera_kernel *                k = nullptr;
        int                                rows = 0, valid = 0;
        bool                               hist = false;
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        axera_cmm                          X, OUT, POS, INDPTR, INDICES, LPL, CLOC, ATT;  // ATT: decode workspace
    };
    std::vector<variant>             vars;
    std::map<std::string, axera_cmm> banks;
    std::map<std::string, size_t>    bytes;  // a bank's size (its CMM rounded up)
    std::vector<std::string>         state;  // banks cleared at position 0
    int                              H = 0, pages = 0;  // pages: the page table's entries (the context / 256)
    int64_t                          pages_kv = 0;      // pages of the KV bank
    int64_t                          next_pos = -1;
    // which side holds the current recurrent state (the kernels' banks or the template's caches) and the positions
    // whose KV each side holds (0 .. *_upto - 1)
    bool                             cur = false;
    int64_t                          sgl_upto = 0, tpl_upto = 0;
    int                              NK = 0, NV = 0, KVW = 0;  // set at the first state move, from the template's caches
};

bool axera_sgl_on() {
    static const bool on = std::getenv("GGML_AXERA_SGL") != nullptr;
    return on;
}

// the whole-model kernels' weight banks from the GGUF (GGML_AXERA_SGL=gguf): the Qwen3.8-27B prefill kernels'
// (models/qwen38_27b). MXFP4 -> NVFP4 is exact: the E2M1 codes are moved (ggml pairs element j with j + 16 in a byte,
// NVFP4 neighbours), each 32-element E8M0 scale serves both 16-element blocks as an E4M3 power of two under a
// power-of-two tensor scale. llama.cpp's conversion is undone exactly (value-head order, ssm_a = -exp(A_log)); the
// Q8_0 ssm_alpha / ssm_beta (48 rows) are requantized to NVFP4 as sgl_kernel_axera.quantization.quantize_nvfp4 does
// from bf16 (the GDN kernel takes all its projections in one format).

namespace sglcv {

struct nv4 {
    std::vector<uint8_t> q, s;  // [rows][K / 2] E2M1 pairs (even K low nibble), [rows][K / 16] E4M3
    float                s2 = 1.0f;
};

static const float E2M1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static uint8_t e4m3_pow2(int p) {  // 2^p, p in [-9, 8]: normal (exponent bias 7) or subnormal (2^-6 / 8 per step)
    return p >= -6 ? (uint8_t) ((p + 7) << 3) : (uint8_t) (1 << (p + 9));
}

static float e4m3_to_f32(uint8_t v) {
    const int e = (v >> 3) & 15, m = v & 7;
    const float f = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
    return (v & 0x80) ? -f : f;
}

static uint8_t f32_to_e4m3(float x) {  // round to nearest even, x >= 0, <= 448 (the callers clamp)
    uint8_t best = 0;
    float   bd   = INFINITY;
    for (int v = 0; v < 0x7f; ++v) {  // 0x7f is NaN
        const float d = fabsf(e4m3_to_f32((uint8_t) v) - x);
        if (d < bd || (d == bd && (v & 1) == 0)) {
            bd   = d;
            best = (uint8_t) v;
        }
    }
    return best;
}

static uint8_t f32_to_e2m1(float x) {  // ml_dtypes float4_e2m1fn: round to nearest even, saturating at 6
    const uint8_t sign = std::signbit(x) ? 8 : 0;
    const float   a    = fminf(fabsf(x), 6.0f);
    int           best = 0;
    float         bd   = INFINITY;
    for (int c = 0; c < 8; ++c) {
        const float d = fabsf(E2M1[c] - a);
        if (d < bd || (d == bd && (c & 1) == 0)) {
            bd   = d;
            best = c;
        }
    }
    return (uint8_t) (sign | best);
}

// rows src_rows of a GGUF MXFP4 [rows][K] tensor, the 32-element blocks of each row in the order col_blocks (empty:
// as stored) -> NVFP4
static nv4 from_mxfp4(const ggml_tensor * t, const std::vector<int> & src_rows, const std::vector<int> & col_blocks) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    const int64_t K = t->ne[0], nb = K / 32;
    auto blk = [&](int r, int64_t j) { return (const block_mxfp4 *) ((const char *) t->data + r * t->nb[1]) + j; };
    auto src = [&](int64_t j) { return col_blocks.empty() ? j : (int64_t) col_blocks[j]; };
    auto nonzero = [](const block_mxfp4 * b) {
        for (int i = 0; i < 16; ++i) {
            if (b->qs[i] & 0x77) {
                return true;
            }
        }
        return false;
    };
    int emax = 0;
    for (int r : src_rows) {
        for (int64_t j = 0; j < nb; ++j) {
            const block_mxfp4 * b = blk(r, j);
            if (b->e > emax && nonzero(b)) {
                emax = b->e;
            }
        }
    }
    const int g = (emax ? emax : 127) - 127 - 8;  // the largest block scale at 2^8, E4M3's largest power of two
    nv4       o;
    o.s2 = ldexpf(1.0f, g);
    o.q.resize(src_rows.size() * (size_t) K / 2);
    o.s.resize(src_rows.size() * (size_t) nb * 2);
    for (size_t r = 0; r < src_rows.size(); ++r) {
        for (int64_t j = 0; j < nb; ++j) {
            const block_mxfp4 * b = blk(src_rows[r], src(j));
            uint8_t             c[32];
            for (int i = 0; i < 16; ++i) {
                c[i]      = b->qs[i] & 15;
                c[i + 16] = b->qs[i] >> 4;
            }
            const int p = std::min(8, std::max(-9, (int) b->e - 127 - g));
            GGML_ASSERT(!nonzero(b) || (int) b->e - 127 - g >= -9);  // (no block below E4M3's range: checked)
            const uint8_t s = nonzero(b) ? e4m3_pow2(p) : 0;
            o.s[(r * nb + j) * 2] = o.s[(r * nb + j) * 2 + 1] = s;
            uint8_t * q = o.q.data() + r * (K / 2) + j * 16;
            for (int i = 0; i < 16; ++i) {
                q[i] = (uint8_t) (c[2 * i] | (c[2 * i + 1] << 4));
            }
        }
    }
    return o;
}

// bf16-valued rows [rows][K] (float) -> NVFP4 as quantize_nvfp4: tensor scale amax / (448 * 6), E4M3 block scales of
// the 16-element block amax / (6 * tensor scale) (at most 448), E2M1 codes of the values over the decoded block scale
static nv4 quantize_nvfp4(const std::vector<float> & w, int64_t rows, int64_t K) {
    float amax = 0.0f;
    for (float v : w) {
        amax = fmaxf(amax, fabsf(v));
    }
    nv4 o;
    o.s2         = amax ? (float) ((double) amax / (448.0 * 6.0)) : 1.0f;
    const float six_g = 6.0f * o.s2;
    o.q.assign(rows * K / 2, 0);
    o.s.assign(rows * K / 16, 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < K / 16; ++j) {
            const float * x  = w.data() + r * K + j * 16;
            float         bm = 0.0f;
            for (int i = 0; i < 16; ++i) {
                bm = fmaxf(bm, fabsf(x[i]));
            }
            const uint8_t s  = f32_to_e4m3(fminf(bm / six_g, 448.0f));
            const float   ds = e4m3_to_f32(s) * o.s2;
            o.s[r * (K / 16) + j] = s;
            uint8_t * q = o.q.data() + r * (K / 2) + j * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t lo = f32_to_e2m1(x[2 * i] / (ds > 0 ? ds : 1.0f));
                const uint8_t hi = f32_to_e2m1(x[2 * i + 1] / (ds > 0 ? ds : 1.0f));
                q[i] = (uint8_t) (lo | (hi << 4));
            }
        }
    }
    return o;
}

static std::vector<float> f32_rows(const ggml_tensor * t) {  // an F32 / Q8_0 tensor as [rows][ne0] float
    const int64_t K = t->ne[0], rows = ggml_nrows(t);
    std::vector<float> out((size_t) (K * rows));
    for (int64_t r = 0; r < rows; ++r) {
        const char * row = (const char *) t->data + r * t->nb[1];
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data() + r * K, row, K * sizeof(float));
        } else {
            GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
            const block_q8_0 * b = (const block_q8_0 *) row;
            for (int64_t j = 0; j < K / 32; ++j) {
                const float d = GGML_FP16_TO_FP32(b[j].d);
                for (int i = 0; i < 32; ++i) {
                    out[r * K + j * 32 + i] = (float) b[j].qs[i] * d;
                }
            }
        }
    }
    return out;
}

static std::vector<uint8_t> bf16_bytes(const std::vector<float> & v) {
    std::vector<uint8_t> o(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) {
        const ggml_bf16_t b = ggml_fp32_to_bf16(v[i]);
        memcpy(o.data() + 2 * i, &b, 2);
    }
    return o;
}

template <typename T> static std::vector<uint8_t> bytes_of(const std::vector<T> & v) {
    std::vector<uint8_t> o(v.size() * sizeof(T));
    memcpy(o.data(), v.data(), o.size());
    return o;
}

static std::vector<uint8_t> f32_bytes(float v) {
    return bytes_of(std::vector<float>{ v });
}

}  // namespace sglcv

// the fields of model layer il, as the kernels' banks name them (b<phase>_<attn|mlp>_<field>)
static std::map<std::string, std::vector<uint8_t>> axera_sgl_layer_fields(const std::map<std::string, ggml_tensor *> & t,
                                                                          int il, int NK, int NV, int KD, int VD) {
    using namespace sglcv;
    std::map<std::string, std::vector<uint8_t>> f;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto T   = [&](const std::string & n) { return t.at(p + n); };
    const int R = NV / NK;
    std::vector<int> hidx(NV);  // HF value head hb from the GGUF's position (vo * NK + kh, hb = kh * R + vo)
    for (int hb = 0; hb < NV; ++hb) {
        hidx[hb] = (hb % R) * NK + hb / R;
    }
    auto range = [](int a, int b) {
        std::vector<int> v;
        for (int i = a; i < b; ++i) {
            v.push_back(i);
        }
        return v;
    };
    auto vrows = [&](int base, int dim) {  // HF-ordered value-head rows base .. base + NV * dim of a GGUF tensor
        std::vector<int> v;
        for (int hb = 0; hb < NV; ++hb) {
            for (int d = 0; d < dim; ++d) {
                v.push_back(base + hidx[hb] * dim + d);
            }
        }
        return v;
    };
    auto put = [&](const std::string & pre, const nv4 & w, const char * scale_name, const char * scale2_name) {
        f[pre]         = w.q;
        f[scale_name]  = w.s;
        f[scale2_name] = f32_bytes(w.s2);
    };
    auto rows_f = [&](const std::vector<float> & x, const std::vector<int> & rows, int64_t K) {
        std::vector<float> o(rows.size() * K);
        for (size_t r = 0; r < rows.size(); ++r) {
            memcpy(o.data() + r * K, x.data() + rows[r] * K, K * sizeof(float));
        }
        return o;
    };
    auto round_bf16 = [](std::vector<float> v) {
        for (float & x : v) {
            x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        }
        return v;
    };
    // MLP
    f["mlp_w_norm"]            = bf16_bytes(f32_rows(T("post_attention_norm.weight")));
    f["mlp_input_permutation"] = bytes_of(std::vector<int32_t>{ 0 });
    const char * mlp[3]        = { "ffn_gate", "ffn_up", "ffn_down" };
    for (int j = 1; j <= 3; ++j) {
        const ggml_tensor * w = T(std::string(mlp[j - 1]) + ".weight");
        const nv4           q = from_mxfp4(w, range(0, (int) ggml_nrows(w)), {});
        f["mlp_w" + std::to_string(j)]         = q.q;
        f["mlp_w_scale" + std::to_string(j)]   = q.s;
        f["mlp_w_scale2_" + std::to_string(j)] = f32_bytes(q.s2);
        f["mlp_w_zp_scale" + std::to_string(j)] = bf16_bytes({ 0.0f });
    }
    if (t.count(p + "ssm_a")) {  // GDN
        const int kw = NK * KD;
        f["attn_input_norm_w"] = bf16_bytes(f32_rows(T("attn_norm.weight")));
        const ggml_tensor * qkv = T("attn_qkv.weight");
        put("attn_w_q", from_mxfp4(qkv, range(0, kw), {}), "attn_w_q_scale", "attn_w_q_scale2");
        put("attn_w_k", from_mxfp4(qkv, range(kw, 2 * kw), {}), "attn_w_k_scale", "attn_w_k_scale2");
        put("attn_w_v", from_mxfp4(qkv, vrows(2 * kw, VD), {}), "attn_w_v_scale", "attn_w_v_scale2");
        put("attn_w_z", from_mxfp4(T("attn_gate.weight"), vrows(0, VD), {}), "attn_w_z_scale", "attn_w_z_scale2");
        for (const char * ab : { "a", "b" }) {
            const ggml_tensor * w = T(std::string(*ab == 'a' ? "ssm_alpha" : "ssm_beta") + ".weight");
            const auto          x = round_bf16(rows_f(f32_rows(w), vrows(0, 1), w->ne[0]));
            put(std::string("attn_w_") + ab, quantize_nvfp4(x, NV, w->ne[0]), (std::string("attn_w_") + ab + "_scale").c_str(),
                (std::string("attn_w_") + ab + "_scale2").c_str());
        }
        std::vector<int> ob;  // ssm_out's input columns: value-head blocks of VD (VD / 32 MXFP4 blocks each)
        for (int hb = 0; hb < NV; ++hb) {
            for (int j = 0; j < VD / 32; ++j) {
                ob.push_back(hidx[hb] * (VD / 32) + j);
            }
        }
        const ggml_tensor * out = T("ssm_out.weight");
        put("attn_w_o", from_mxfp4(out, range(0, (int) ggml_nrows(out)), ob), "attn_w_o_scale", "attn_w_o_scale2");
        const ggml_tensor *      cw   = T("ssm_conv1d.weight");  // [channels][taps]
        const std::vector<float> conv = f32_rows(cw);
        const int64_t            taps = cw->ne[0];
        f["attn_w_q_conv"] = bf16_bytes(rows_f(conv, range(0, kw), taps));
        f["attn_w_k_conv"] = bf16_bytes(rows_f(conv, range(kw, 2 * kw), taps));
        f["attn_w_v_conv"] = bf16_bytes(rows_f(conv, vrows(2 * kw, VD), taps));
        const std::vector<float> a = f32_rows(T("ssm_a")), dt = f32_rows(T("ssm_dt.bias"));
        std::vector<float>       alog(NV), dtb(NV);
        for (int hb = 0; hb < NV; ++hb) {
            alog[hb] = logf(-a[hidx[hb]]);
            dtb[hb]  = dt[hidx[hb]];
        }
        f["attn_a_log"]    = bytes_of(alog);
        f["attn_dt_bias"]  = bf16_bytes(dtb);
        f["attn_o_norm_w"] = bf16_bytes(f32_rows(T("ssm_norm.weight")));
    } else {  // gated attention
        const char * names[4] = { "attn_q", "attn_k", "attn_v", "attn_output" };
        const char   x[4]     = { 'q', 'k', 'v', 'o' };
        for (int i = 0; i < 4; ++i) {
            const ggml_tensor * w = T(std::string(names[i]) + ".weight");
            const nv4           q = from_mxfp4(w, range(0, (int) ggml_nrows(w)), {});
            f[std::string("attn_w_") + x[i]]             = q.q;
            f[std::string("attn_w_scale_") + x[i]]       = q.s;
            f[std::string("attn_w_scale2_") + x[i]]      = f32_bytes(q.s2);
            f[std::string("attn_w_zeropoint_") + x[i]]   = bf16_bytes({ 0.0f });
        }
        f["attn_w_norm_attn"] = bf16_bytes(f32_rows(T("attn_norm.weight")));
        std::vector<float> qn = f32_rows(T("attn_q_norm.weight"));
        const float        sc = 1.0f / sqrtf((float) T("attn_q_norm.weight")->ne[0]);
        for (float & v : qn) {
            v *= sc;
        }
        f["attn_w_norm_q"] = bytes_of(qn);
        f["attn_w_norm_k"] = bf16_bytes(f32_rows(T("attn_k_norm.weight")));
    }
    return f;
}

// every weight bank (kernel input name -> its bytes, already allocated and zeroed) of an L-layer model from the GGUF
static void axera_sgl_fill_from_gguf(std::map<std::string, axera_cmm> & banks, const std::map<std::string, size_t> & bytes,
                                     const std::map<std::string, ggml_tensor *> & t, int L, double theta, int rot) {
    const ggml_tensor * qkv0 = t.at("blk.0.attn_qkv.weight");
    if (qkv0->type != GGML_TYPE_MXFP4) {
        GGML_ABORT("axera sgl: banks from the GGUF need its layer weights in MXFP4 (Qwen3.8-27B mxfp4), not %s\n",
                   ggml_type_name(qkv0->type));
    }
    const int           NV   = (int) t.at("blk.0.ssm_a")->ne[0];
    const int           VD   = (int) (t.at("blk.0.attn_gate.weight")->ne[1] / NV);
    const int           DK   = (int) t.at("blk.0.ssm_norm.weight")->ne[0];  // key dim = value dim (128) in Qwen3.5
    const int           NK   = (int) ((qkv0->ne[1] - NV * VD) / 2 / DK);
    const int           blocks = L / 4;
    GGML_LOG_INFO("axera sgl: banks from the GGUF: %d layers, GDN %d key / %d value heads\n", L, NK, NV);
    std::mutex mu;
    std::vector<std::string> missing;
    axera_host_pool().run(L, [&](int64_t il) {
        const auto f     = axera_sgl_layer_fields(t, (int) il, NK, NV, DK, VD);
        const int  phase = (int) il % 4, b = (int) il / 4;
        for (const auto & [field, data] : f) {
            const std::string name = "b" + std::to_string(phase) + "_" + field;
            auto              it   = banks.find(name);
            if (it == banks.end()) {
                std::lock_guard<std::mutex> lock(mu);
                missing.push_back(name);
                continue;
            }
            const size_t stride = bytes.at(name) / blocks;  // a block's rows, padded to 32 bytes
            GGML_ASSERT(data.size() <= stride && "axera sgl: a GGUF field larger than its bank row");
            memcpy((char *) it->second->vir + b * stride, data.data(), data.size());
        }
    });
    // RoPE: [8192][cos | sin] of the rotated half, one row shared by the blocks
    auto cs = banks.find("b3_attn_cos_sin_cache");
    if (cs != banks.end()) {
        const int64_t n = (int64_t) (bytes.at("b3_attn_cos_sin_cache") / (rot * sizeof(float)));
        auto *        o = (float *) cs->second->vir;
        for (int64_t i = 0; i < n; ++i) {
            for (int j = 0; j < rot / 2; ++j) {
                const float inv = powf((float) theta, (float) (2 * j) / (float) rot);
                const float a   = (float) i / inv;
                o[i * rot + j]           = cosf(a);
                o[i * rot + rot / 2 + j] = sinf(a);
            }
        }
    }
    for (auto & [name, b] : banks) {
        b->flush(b->vir, b->size);
    }
    std::sort(missing.begin(), missing.end());
    missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
    for (const auto & n : missing) {
        GGML_LOG_WARN("axera sgl: GGUF field without a bank: %s\n", n.c_str());
    }
}

// GGML_AXERA_SGL=gguf: the banks converted from the GGUF at load; =<dir>: read from <dir> (pack_sgl27.py, a check)
std::shared_ptr<axera_sgl> axera_sgl_load(int H, const std::map<std::string, ggml_tensor *> & t, int L, double theta, int rot) {
    const std::string src       = std::getenv("GGML_AXERA_SGL");
    const bool        from_gguf = src == "gguf";
    auto              s         = std::make_shared<axera_sgl>();
    s->H                        = H;
    static const std::set<std::string> meta = { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                                                "attention" };
    // the kernels; their weight / state inputs are the banks, shared by all
    std::vector<std::pair<std::string, ggml_axera_kernel *>> ks;
    for (const auto & f : axera_kernels().files) {
        int rows, valid, hist, model = 0, len = 0;
        if (sscanf(f.first.c_str(), "sgl%d_r%d_v%d_h%d%n", &model, &rows, &valid, &hist, &len) != 4 || (size_t) len != f.first.size()) {
            continue;
        }
        if (rows > 1 && valid != rows) {  // NaN (sglang runs a partial chunk through its per-layer kernels)
            GGML_LOG_WARN("axera sgl: %s skipped: a chunk of fewer valid rows computes NaN\n", f.first.c_str());
            continue;
        }
        ggml_axera_kernel * k = axera_kernels().get(f.first);
        if (!k) {
            GGML_ABORT("axera sgl: cannot load %s\n", f.first.c_str());
        }
        ks.emplace_back(f.first, k);
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            const AX_ENGINE_IOMETA_T & mt = k->info->pInputs[i];
            if (meta.count(mt.pName)) {
                continue;
            }
            auto it = s->bytes.find(mt.pName);
            GGML_ASSERT((it == s->bytes.end() || it->second == mt.nSize) && "axera sgl: kernels disagree on a bank size");
            s->bytes[mt.pName] = mt.nSize;
        }
    }
    if (ks.empty()) {
        GGML_ABORT("axera sgl: no sgl<model>_r<rows>_v<valid>_h<history> kernels in the kernel directory\n");
    }
    std::map<std::string, int> state;  // bank -> 1: recurrent state / KV pages (cleared, not loaded)
    for (const auto & [n, b] : s->bytes) {
        state[n] = n.find("_state_in") != std::string::npos || (n.size() > 3 && n.compare(n.size() - 3, 3, "_kv") == 0);
    }
    if (!from_gguf) {  // the banks' sizes from the packer's manifest must be the kernels'
        std::ifstream man(src + "/manifest.txt");
        std::string   name;
        size_t        n_bytes = 0;
        int           st      = 0;
        while (man >> name >> n_bytes >> st) {
            GGML_ASSERT(s->bytes.count(name) && s->bytes.at(name) == n_bytes && "axera sgl: manifest differs from the kernels");
        }
    }
    size_t cfree = 0, ctotal = 0;
    axera_cmm_memory(&cfree, &ctotal);
    GGML_LOG_WARN("axera sgl: %zu banks from %s, CMM free %.2f of %.2f GiB\n", s->bytes.size(), from_gguf ? "the GGUF" : src.c_str(),
                  cfree / 1073741824.0, ctotal / 1073741824.0);
    // the largest banks first, while the CMM has the holes for them (a bank is one IO: contiguous)
    std::vector<std::pair<size_t, std::string>> order;
    for (const auto & [n, b] : s->bytes) {
        order.emplace_back(b, n);
    }
    std::sort(order.rbegin(), order.rend());
    size_t total = 0;
    for (const auto & [n_bytes, n_name] : order) {
        axera_cmm b = axera_cmm_alloc(n_bytes);  // zeroed
        if (state[n_name]) {
            s->state.push_back(n_name);
        } else if (!from_gguf) {
            FILE * f = fopen((src + "/" + n_name + ".bin").c_str(), "rb");
            if (!f || fread(b->vir, 1, n_bytes, f) != n_bytes) {
                GGML_ABORT("axera sgl: cannot read %s/%s.bin (%zu bytes)\n", src.c_str(), n_name.c_str(), n_bytes);
            }
            fclose(f);
            b->flush(b->vir, n_bytes);
        }
        total += n_bytes;
        s->banks[n_name] = std::move(b);
    }
    if (from_gguf) {
        const auto t0 = std::chrono::steady_clock::now();
        axera_sgl_fill_from_gguf(s->banks, s->bytes, t, L, theta, rot);
        GGML_LOG_WARN("axera sgl: banks converted from the GGUF in %.1f s\n",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (const char * d = std::getenv("GGML_AXERA_SGL_DUMP")) {  // a check: the banks as files (pack_sgl27.py's)
            for (const auto & [n, b] : s->banks) {
                if (!state[n]) {
                    FILE * f = fopen((std::string(d) + "/" + n + ".bin").c_str(), "wb");
                    GGML_ASSERT(f);
                    fwrite(b->vir, 1, s->bytes.at(n), f);
                    fclose(f);
                }
            }
        }
    }
    for (const auto & [name, k] : ks) {
        int rows, valid, hist, model = 0;
        sscanf(name.c_str(), "sgl%d_r%d_v%d_h%d", &model, &rows, &valid, &hist);
        axera_sgl::variant v;
        v.k     = k;
        v.rows  = rows;
        v.valid = valid;
        v.hist  = hist != 0;
        const AX_ENGINE_IO_INFO_T * io = v.k->info;
        auto small = [&](axera_cmm & b, uint32_t n) {
            b = axera_cmm_alloc(n);
            return axera_io(b.get(), b->vir, n);
        };
        for (uint32_t i = 0; i < io->nInputSize; ++i) {
            const AX_ENGINE_IOMETA_T & mt = io->pInputs[i];
            const std::string          nm = mt.pName;
            auto                       it = s->banks.find(nm);
            if (it != s->banks.end()) {
                v.in.push_back(axera_io(it->second.get(), it->second->vir, mt.nSize));
            } else if (nm == "position_ids") {
                v.in.push_back(small(v.POS, mt.nSize));
            } else if (nm == "indptr") {
                v.in.push_back(small(v.INDPTR, mt.nSize));
            } else if (nm == "indices") {
                v.in.push_back(small(v.INDICES, mt.nSize));
                s->pages = (int) (mt.nSize / 4);
            } else if (nm == "last_page_len") {
                v.in.push_back(small(v.LPL, mt.nSize));
            } else if (nm == "cache_location") {
                v.in.push_back(small(v.CLOC, mt.nSize));
            } else if (nm == "attention") {
                v.in.push_back(small(v.ATT, mt.nSize));
            } else {
                GGML_ASSERT(nm == "x" && mt.nSize == (uint32_t) rows * H * 2 && "axera sgl: unknown kernel input");
                v.in.push_back(small(v.X, mt.nSize));
            }
        }
        for (uint32_t i = 0; i < io->nOutputSize; ++i) {
            const AX_ENGINE_IOMETA_T & mt = io->pOutputs[i];
            auto                       it = s->banks.find(mt.pName);
            if (it != s->banks.end()) {
                v.out.push_back(axera_io(it->second.get(), it->second->vir, mt.nSize));
            } else {
                GGML_ASSERT(!v.OUT && mt.nSize == (uint32_t) rows * H * 2 && "axera sgl: unknown kernel output");
                v.out.push_back(small(v.OUT, mt.nSize));
            }
        }
        GGML_ASSERT(v.X && v.OUT && v.POS && v.INDPTR && v.LPL && v.CLOC);
        s->vars.push_back(std::move(v));
    }
    GGML_LOG_WARN("axera sgl: %zu banks (%.2f GiB), %zu kernels, %d pages\n", s->banks.size(), total / 1073741824.0,
                  s->vars.size(), s->pages);
    return s;
}

static void axera_sgl_restart(axera_sgl & s) {
    for (const auto & n : s.state) {
        auto & b = s.banks.at(n);
        if (n.find("state") != std::string::npos) {  // the KV pages need no clearing: read up to the valid length
            memset(b->vir, 0, b->size);
            b->flush(b->vir, b->size);
        }
    }
    s.next_pos = 0;
    s.cur      = false;
    s.sgl_upto = s.tpl_upto = 0;
}

// the page metadata of n rows at position p (sgl_kernel_axera batch_metadata.prepare_attention_metadata): the history
// before p (its pages, the cells of its last page, the page table: identity, unused entries 0), the new rows' page
// p / 256 and cell p % 256
static void axera_sgl_meta(const axera_sgl & s, axera_sgl::variant & v, int64_t p, int n) {
    auto * ps = (int32_t *) v.POS->vir;
    for (int j = 0; j < n; ++j) {
        ps[j] = (int32_t) (p + j);
    }
    const int32_t hist = (int32_t) ((p + 255) / 256);
    ((int32_t *) v.INDPTR->vir)[0] = 0;
    ((int32_t *) v.INDPTR->vir)[1] = hist;
    ((int32_t *) v.LPL->vir)[0]    = p > 0 ? (int32_t) (p - (int64_t) (hist - 1) * 256) : 0;
    for (int j = 0; j < s.pages; ++j) {
        ((int32_t *) v.INDICES->vir)[j] = j < hist ? j : 0;
    }
    ((int32_t *) v.CLOC->vir)[0] = (int32_t) (p / 256);
    ((int32_t *) v.CLOC->vir)[1] = (int32_t) (p % 256);
    for (auto * b : { v.POS.get(), v.INDPTR.get(), v.LPL.get(), v.CLOC.get(), v.INDICES.get() }) {
        b->flush(b->vir, b->size);
    }
}

static void axera_sgl_call(axera_sgl::variant & v, int64_t p) {
    const int ret = axera_run_sync(v.k, v.in, v.out);
    if (ret != 0) {
        GGML_ABORT("axera sgl: RunSync 0x%x (rows %d valid %d, position %" PRId64 ")\n", ret, v.rows, v.valid, p);
    }
}

// the layers of T tokens at positions pos (consecutive, a single sequence): layer 0 input rows x0(r) -> h [T, H].
// Returns the tokens run (-1: failed); prefill_only: the chunk kernels only, stopping at a rest shorter than them all
int axera_sgl_layers(std::shared_ptr<axera_sgl> & sp, const int32_t * pos, int T, int H,
                     const std::function<const float *(int)> & x0, std::vector<float> & h, bool prefill_only) {
    GGML_ASSERT(sp && "axera sgl: the banks are loaded by axera_q35_init");
    axera_sgl & s = *sp;
    GGML_ASSERT(s.H == H);
    static const bool unaligned = [] {
        const char * e = std::getenv("GGML_AXERA_SGL_UNALIGNED");
        return e == nullptr || strcmp(e, "0") != 0;
    }();
    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        if (p == 0) {
            axera_sgl_restart(s);
        }
        if (p != s.next_pos || p >= (int64_t) s.pages * 256) {
            GGML_LOG_ERROR("axera sgl: position %" PRId64 " (expected %" PRId64 ", context %d)\n", p, s.next_pos,
                           s.pages * 256);
            s.next_pos = -1;
            return -1;
        }
        // the largest chunk kernel that fits from a page start, with history after the first page (a kernel built with
        // history fails without: inf)
        axera_sgl::variant * v = nullptr;
        for (auto & c : s.vars) {
            const bool hist_ok = c.hist == (p > 0);
            // also from inside a page, within it (the last chunk after a shorter one; GGML_AXERA_SGL_UNALIGNED=0: off)
            const bool start_ok = p % 256 == 0 || (unaligned && p % 256 + c.valid <= 256);
            if (start_ok && c.rows > 1 && hist_ok && c.valid <= T - it && p + c.valid <= (int64_t) s.pages * 256 &&
                (!v || c.valid > v->valid)) {
                v = &c;
            }
        }
        if (!v && prefill_only) {
            return it;
        }
        if (!v) {
            for (auto & c : s.vars) {
                v = c.rows == 1 ? &c : v;
            }
        }
        if (!v) {
            GGML_LOG_ERROR("axera sgl: no kernel for position %" PRId64 "\n", p);
            return -1;
        }
        const int n = v->rows == 1 ? 1 : v->valid;
        axera_timer tm(v->rows == 1 ? "sgl decode" : "sgl prefill chunk");
        auto * X = (ggml_bf16_t *) v->X->vir;
        axera_host_pool().run(n, [&](int64_t r) { ggml_fp32_to_bf16_row(x0(it + (int) r), X + (size_t) r * H, H); });
        v->X->flush(v->X->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        axera_sgl_meta(s, *v, p, n);
        axera_sgl_call(*v, p);
        v->OUT->invalidate(v->OUT->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        const auto * O = (const ggml_bf16_t *) v->OUT->vir;
        axera_host_pool().run(n, [&](int64_t r) {
            ggml_bf16_to_fp32_row(O + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
        });
        s.next_pos = p + n;
        s.sgl_upto = s.next_pos;
        s.cur      = true;
        it += n;
    }
    return T;
}

int64_t axera_sgl_next(const axera_sgl * s) {
    return s ? s->next_pos : -1;
}

bool axera_sgl_has_decode(const axera_sgl * s) {
    return s && std::any_of(s->vars.begin(), s->vars.end(), [](const axera_sgl::variant & v) { return v.rows == 1; });
}

// the recurrent state and the KV of positions p0 .. p0 + n - 1 (cells cell_of(i) of the template's cache, i = 0 .. n - 1)
// from the kernels' banks into the template's caches (to_tpl) or back. GDN value heads: the kernels' HF order (key head
// kh, its value heads vo: kh * R + vo, R = NV / NK), the template's GGUF order (vo * NK + kh); its conv state q | k | v
// in one row per step, the kernels' three pools. KV: pages [page][K, V][256 cells][heads x dim] bf16 against
// [cell][heads x dim] f16.
static void axera_sgl_move(axera_sgl & s, axera_q35 & m, int64_t p0, int64_t n, const std::function<int64_t(int64_t)> & cell_of,
                           bool to_tpl) {
    axera_timer tm(to_tpl ? "sgl state -> template" : "sgl state <- template");
    const int DK = 128, DV = 128, W = 3;
    int       nG = 0, nA = 0;
    for (const auto & ly : m.layers) {
        (ly.gdn ? nG : nA)++;
    }
    auto bank = [&](const char * name) -> char * {
        auto & b = s.banks.at(name);
        if (to_tpl) {
            b->invalidate(b->vir, b->size);
        }
        return (char *) b->vir;
    };
    char * st = bank("b0_attn_state_in");
    char * qc = bank("b0_attn_q_conv_state_in");
    char * kc = bank("b0_attn_k_conv_state_in");
    char * vc = bank("b0_attn_v_conv_state_in");
    char * kv = bank("b3_kv");
    if (!s.NV) {
        s.NV       = (int) (s.bytes.at("b0_attn_state_in") / ((size_t) nG * DK * DV * sizeof(float)));
        s.NK       = (int) (s.bytes.at("b0_attn_q_conv_state_in") / ((size_t) nG * W * DK * sizeof(ggml_bf16_t)));
        for (const auto & ly : m.layers) {
            if (!ly.gdn && !s.KVW) {
                s.KVW = (int) (ly.kv.at(0).row / sizeof(ggml_fp16_t));
            }
        }
        s.pages_kv = (int64_t) (s.bytes.at("b3_kv") / ((size_t) nA * 2 * 256 * s.KVW * sizeof(ggml_bf16_t)));
        GGML_LOG_INFO("axera sgl: %d GDN layers (%d key / %d value heads), %d attention layers (KV row %d), %" PRId64
                      " KV pages\n", nG, s.NK, s.NV, nA, s.KVW, s.pages_kv);
    }
    const int    NK = s.NK, NV = s.NV, R = NV / NK, KVW = s.KVW;
    const size_t head = (size_t) DK * DV * sizeof(float);
    int64_t      lo = n ? cell_of(0) : 0, hi = lo;  // the template's cells (a single sequence's: consecutive)
    for (int64_t t = 1; t < n; ++t) {
        lo = std::min(lo, cell_of(t));
        hi = std::max(hi, cell_of(t));
    }
    // one job per GDN layer's state / conv rows and per attention layer's K or V cache, in parallel
    std::vector<std::function<void()>> jobs;
    int                                g = 0, a = 0;
    for (auto & ly : m.layers) {
        if (ly.gdn) {
            GGML_ASSERT(ly.in_place && "axera sgl: the GDN state in the llama.cpp recurrent cache");
            for (auto & rg : ly.cache_state) {
                jobs.push_back([&, g, rg, row = ly.row] {
                    char * tp = rg.base + row * rg.stride;
                    if (!to_tpl) {
                        rg.b->invalidate(tp, rg.size);
                    }
                    if (!rg.conv) {
                        GGML_ASSERT(rg.size == NV * head);
                        for (int ha = 0; ha < NV; ++ha) {
                            char * sp = st + ((size_t) g * NV + (ha % NK) * R + ha / NK) * head;
                            memcpy(to_tpl ? tp + ha * head : sp, to_tpl ? sp : tp + ha * head, head);
                        }
                    } else {
                        const int C = 2 * NK * DK + NV * DV;
                        GGML_ASSERT(rg.size == (size_t) W * C * sizeof(ggml_bf16_t));
                        auto * d  = (ggml_bf16_t *) tp;
                        auto   cp = [&](ggml_bf16_t * t, char * k, size_t bytes) {
                            memcpy(to_tpl ? (void *) t : (void *) k, to_tpl ? (void *) k : (void *) t, bytes);
                        };
                        for (int r = 0; r < W; ++r) {
                            const size_t i = (size_t) g * W + r;
                            cp(d + r * C, qc + i * NK * DK * 2, NK * DK * 2);
                            cp(d + r * C + NK * DK, kc + i * NK * DK * 2, NK * DK * 2);
                            for (int ha = 0; ha < NV; ++ha) {
                                const int hb = (ha % NK) * R + ha / NK;
                                cp(d + r * C + 2 * NK * DK + ha * DV, vc + (i * NV * DV + hb * DV) * 2, DV * 2);
                            }
                        }
                    }
                    if (to_tpl) {
                        rg.b->flush(tp, rg.size);
                    }
                });
            }
            ++g;
            continue;
        }
        for (auto & kr : ly.kv) {
            GGML_ASSERT(kr.row == KVW * sizeof(ggml_fp16_t));
            jobs.push_back([&, a, kr, stream = ly.stream] {
                char * base = kr.base + stream * kr.stride;
                auto * pg   = (ggml_bf16_t *) kv + (size_t) a * s.pages_kv * 2 * 256 * KVW;
                if (!to_tpl && n) {
                    kr.b->invalidate(base + lo * kr.row, (size_t) (hi - lo + 1) * kr.row);
                }
                for (int64_t t = 0; t < n; ++t) {
                    const int64_t q    = p0 + t;
                    ggml_bf16_t * row  = pg + (((q / 256) * 2 + (kr.v ? 1 : 0)) * 256 + q % 256) * (size_t) KVW;
                    auto *        cell = (ggml_fp16_t *) (base + cell_of(t) * kr.row);
                    for (int i = 0; i < KVW; ++i) {
                        if (to_tpl) {
                            cell[i] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(row[i]));
                        } else {
                            row[i] = ggml_fp32_to_bf16(ggml_fp16_to_fp32(cell[i]));
                        }
                    }
                }
                if (to_tpl && n) {
                    kr.b->flush(base + lo * kr.row, (size_t) (hi - lo + 1) * kr.row);
                }
            });
        }
        ++a;
    }
    axera_host_pool().run((int64_t) jobs.size(), [&](int64_t j) { jobs[j](); });
    if (!to_tpl) {
        for (const char * name : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in",
                                   "b0_attn_v_conv_state_in", "b3_kv" }) {
            auto & b = s.banks.at(name);
            b->flush(b->vir, b->size);
        }
    }
}

// a prompt's chunks just ran here (positions p0 .. p0 + n - 1, batch tokens 0 .. n - 1 at cells cell_of): the state and
// their KV into the template's caches, which run next
void axera_sgl_to_q35(axera_sgl & s, axera_q35 & m, int64_t p0, int64_t n, const std::function<int64_t(int64_t)> & cell_of) {
    axera_sgl_move(s, m, p0, n, cell_of, true);
    s.cur      = false;
    s.tpl_upto = p0 + n;
}

// before a batch at position p runs on the template's kernels (sgl = false) or here (sgl = true): the state and the KV
// the side lacks moved to it (a single sequence at the cells of its positions: llama.cpp's for one sequence from 0)
void axera_sgl_sync(axera_sgl & s, axera_q35 & m, int64_t p, bool sgl) {
    auto ident = [](int64_t base) { return [base](int64_t t) { return base + t; }; };
    if (sgl && !s.cur) {
        axera_sgl_move(s, m, s.sgl_upto, p - s.sgl_upto, ident(s.sgl_upto), false);
        s.sgl_upto = p;
        s.next_pos = p;
        s.cur      = true;
    } else if (!sgl && s.cur) {
        axera_sgl_move(s, m, s.tpl_upto, p - s.tpl_upto, ident(s.tpl_upto), true);
        s.tpl_upto = p;
        s.cur      = false;
    }
}

// the template's kernels ran positions up to p - 1
void axera_sgl_template_ran(axera_sgl * s, int64_t p) {
    if (s) {
        s->tpl_upto = p;
        s->cur      = false;
    }
}

// one token at position p through the decode kernel: layer 0 input row x -> out [H]
bool axera_sgl_decode(axera_sgl & s, axera_q35 & m, int64_t p, const float * x, float * out) {
    if (p == 0) {
        axera_sgl_restart(s);
    }
    axera_sgl_sync(s, m, p, true);
    axera_sgl::variant * v = nullptr;
    for (auto & c : s.vars) {
        v = c.rows == 1 ? &c : v;
    }
    if (!v || p >= (int64_t) s.pages * 256 || p >= s.pages_kv * 256) {
        GGML_LOG_ERROR("axera sgl: no decode at position %" PRId64 " (context %d)\n", p, s.pages * 256);
        return false;
    }
    axera_timer tm("sgl decode");
    ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) v->X->vir, s.H);
    v->X->flush(v->X->vir, (size_t) s.H * sizeof(ggml_bf16_t));
    axera_sgl_meta(s, *v, p, 1);
    axera_sgl_call(*v, p);
    v->OUT->invalidate(v->OUT->vir, (size_t) s.H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) v->OUT->vir, out, s.H);
    s.next_pos = s.sgl_upto = p + 1;
    s.cur                   = true;
    return true;
}
