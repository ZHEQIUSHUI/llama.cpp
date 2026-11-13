// Axera NPU backend: the MiniCPM-OCR-1.0 SigLIP vision encoder in one four-core kernel.
//
// One call runs 27 layers and the final 2 x 2 merger for up to N patches (the release kernel is built
// for N = 1536; mtmd hands over one slice at a time).  The learned position table uses the processor's
// nearest bucket (floor(70 * i / size)), there is no RoPE, and the q/k/v projections are stacked into
// the qkv layout of q35_vit's existing layer functions.  Weights come from the mmproj GGUF.

#include "../model.h"
#include "../../core/patch_embed.h"
#include "../../ops/ops.h"

#include <cmath>
#include <cstring>

namespace axera {
namespace {

class minicpm_ocr_vit_full final : public model {
  public:
    explicit minicpm_ocr_vit_full(const manifest & m) : man(m), k(man.get("vit")) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest &                 man;
    const kernel &                   k;
    graph_names                      names;
    std::map<std::string, axera_cmm> buf;
    const void *                     bound = nullptr;
    int                              N = 0, QB = 0, BC = 0;
    patch_embed                      pemb;

    axera_cmm & io(const char * n, bool out = false) {
        auto & c = buf[n];
        if (!c) {
            c = axera_cmm_alloc(out ? k.out_bytes(n) : k.in_bytes(n));
        }
        return c;
    }
    void weights(int L, int H, int CP);
    void tables(int gh, int gw, int n, const ggml_tensor * pe, int H);
};

void to_bf16(const ggml_tensor * w, ggml_bf16_t * dst) {
    const int64_t ne = ggml_nelements(w);
    if (w->type == GGML_TYPE_BF16) {
        memcpy(dst, w->data, ne * sizeof(ggml_bf16_t));
    } else if (w->type == GGML_TYPE_F32) {
        ggml_fp32_to_bf16_row((const float *) w->data, dst, ne);
    } else {
        std::vector<float> f(ne);
        ggml_get_type_traits(w->type)->to_float(w->data, f.data(), ne);
        ggml_fp32_to_bf16_row(f.data(), dst, ne);
    }
}

void minicpm_ocr_vit_full::weights(int L, int H, int CP) {
    axera_timer tm("ocr vit weights (load)");
    auto stack = [&](const char * in, const char * suffix) {
        axera_cmm & c = io(in);
        const size_t abi = k.in_bytes(in);
        GGML_ASSERT(abi % (size_t) L == 0);
        const size_t per = abi / L;
        axera_host_pool().run(L, [&](int64_t l) {
            const ggml_tensor * w = names.at("v.blk." + std::to_string(l) + "." + suffix);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + l * per));
        });
        c->flush();
    };
    stack("ln1_w", "ln1.weight");
    stack("ln1_b", "ln1.bias");
    stack("w_o", "attn_out.weight");
    stack("b_o", "attn_out.bias");
    stack("ln2_w", "ln2.weight");
    stack("ln2_b", "ln2.bias");
    stack("w_up", "ffn_up.weight");
    stack("b_up", "ffn_up.bias");
    stack("w_down", "ffn_down.weight");
    stack("b_down", "ffn_down.bias");

    struct qkv { const char * in; const char * q; const char * kk; const char * v; size_t elem; };
    for (const qkv & x : {qkv{"w_qkv", "attn_q.weight", "attn_k.weight", "attn_v.weight", (size_t) H * H},
                          qkv{"b_qkv", "attn_q.bias", "attn_k.bias", "attn_v.bias", (size_t) H}}) {
        axera_cmm & c = io(x.in);
        const size_t abi = k.in_bytes(x.in);
        GGML_ASSERT(abi % (size_t) L == 0);
        const size_t per = abi / L;
        GGML_ASSERT(per == 3 * x.elem * sizeof(ggml_bf16_t));
        axera_host_pool().run(L, [&](int64_t l) {
            char * d = (char *) c->vir + l * per;
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.q), (ggml_bf16_t *) d);
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.kk), (ggml_bf16_t *) (d + x.elem * sizeof(ggml_bf16_t)));
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.v), (ggml_bf16_t *) (d + 2 * x.elem * sizeof(ggml_bf16_t)));
        });
        c->flush();
    }

    static const std::pair<const char *, const char *> whole[] = {
        { "b_patch", "v.patch_embd.bias" }, { "post_w", "v.post_ln.weight" }, { "post_b", "v.post_ln.bias" },
        { "mm_w", "mm.input_norm.weight" }, { "mm_b", "mm.input_norm.bias" },
        { "w0", "mm.up.weight" }, { "b0", "mm.up.bias" }, { "w2", "mm.down.weight" }, { "b2", "mm.down.bias" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        to_bf16(w, (ggml_bf16_t *) io(in)->vir);
        io(in)->flush();
    }
    const ggml_tensor * wp = names.at("v.patch_embd.weight");
    GGML_ASSERT((size_t) ggml_nelements(wp) == (size_t) H * CP);
    std::vector<float> pw((size_t) H * CP), zero((size_t) H * CP, 0.0f);
    if (wp->type == GGML_TYPE_F32) {
        memcpy(pw.data(), wp->data, pw.size() * sizeof(float));
    } else if (wp->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) wp->data, pw.data(), (int64_t) pw.size());
    } else {
        ggml_get_type_traits(wp->type)->to_float(wp->data, pw.data(), (int64_t) pw.size());
    }
    const int PW = (int) (k.in_bytes("w_patch") / sizeof(ggml_bf16_t) / H);
    GGML_ASSERT(pemb.init(pw.data(), zero.data(), H, CP, PW));
    const std::vector<float> & ws = pemb.weight_still();
    GGML_ASSERT(ws.size() * sizeof(ggml_bf16_t) == k.in_bytes("w_patch"));
    ggml_fp32_to_bf16_row(ws.data(), (ggml_bf16_t *) io("w_patch")->vir, (int64_t) ws.size());
    io("w_patch")->flush();
}

void minicpm_ocr_vit_full::tables(int gh, int gw, int n, const ggml_tensor * pe, int H) {
    axera_timer tm("ocr vit tables (cpu)");
    std::vector<int> py(N, 0), px(N, 0);
    for (int y = 0, i = 0; y < gh; y += 2) {
        for (int x = 0; x < gw; x += 2) {
            for (int d = 0; d < 4; ++d, ++i) {
                py[i] = y + d / 2;
                px[i] = x + d % 2;
            }
        }
    }
    const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
    const float * tab  = (const float *) pe->data;
    GGML_ASSERT(pe->type == GGML_TYPE_F32 && pe->ne[0] == H && side == 70);
    auto * pos = (ggml_bf16_t *) io("pos")->vir;
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pos + i * H;
        if (i >= n) {
            std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
            return;
        }
        const int y = (int) ((int64_t) py[i] * side / gh);
        const int x = (int) ((int64_t) px[i] * side / gw);
        ggml_fp32_to_bf16_row((const float *) (tab + ((int64_t) y * side + x) * H), d, H);
    });
    io("pos")->flush();

    *(int32_t *) io("nblk")->vir = (n + BC - 1) / BC;
    io("nblk")->flush();
    auto * km = (ggml_bf16_t *) io("kmask")->vir;
    axera_host_pool().run(N / BC, [&](int64_t b) {
        std::vector<ggml_bf16_t> row(BC);
        for (int c = 0; c < BC; ++c) {
            row[c] = ggml_fp32_to_bf16(b * BC + c < n ? 0.0f : -3e38f);
        }
        for (int r = 0; r < QB; ++r) {
            memcpy(km + ((size_t) b * QB + r) * BC, row.data(), BC * sizeof(ggml_bf16_t));
        }
    });
    io("kmask")->flush();
}

ggml_status minicpm_ocr_vit_full::compute(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     H = (int) pe->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0], CP = 3 * PS * PS;
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    if (N == 0) {
        N = (int) (k.in_bytes("px") / (CP * sizeof(ggml_bf16_t)));
        const AX_ENGINE_IOMETA_T & km = k.info->pInputs[k.input("kmask")];
        GGML_ASSERT(km.nShapeSize == 3);
        QB = (int) km.pShape[1];
        BC = (int) km.pShape[2];
        GGML_ASSERT(km.pShape[0] * BC == N);
    }
    if (raw->ne[3] != 1 || gw % 2 || gh % 2 || n > N) {
        GGML_LOG_ERROR("axera minicpm_ocr_vit_full: %d frame(s) of %d x %d patches, at most %d patches\n", (int) raw->ne[3], gw,
                       gh, N);
        return GGML_STATUS_FAILED;
    }
    const ggml_tensor * w0 = names.at("v.blk.0.ln1.weight");
    if (bound != w0->data) {
        weights(L, H, CP);
        bound = w0->data;
    }
    tables(gh, gw, n, pe, H);
    {
        axera_timer   tm("ocr vit patches (cpu)");
        const float * img = (const float *) raw->data;
        const int64_t W = raw->ne[0], HI = raw->ne[1];
        auto *        pxb = (ggml_bf16_t *) io("px")->vir;
        const int     PW = pemb.width();
        axera_host_pool().run(N, [&](int64_t i) {
            ggml_bf16_t * d = pxb + i * PW;
            if (i >= n) {
                std::fill(d, d + PW, ggml_fp32_to_bf16(0.0f));
                return;
            }
            const int y = (int) ((i / (2 * gw)) * 2 + (i % 4) / 2);
            const int x = (int) (((i / 4) % (gw / 2)) * 2 + i % 2);
            thread_local std::vector<float> p, row;
            p.resize(CP);
            row.resize(PW);
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < PS; ++ky) {
                    const float * src = img + c * W * HI + (int64_t) (y * PS + ky) * W + x * PS;
                    memcpy(p.data() + (c * PS + ky) * PS, src, PS * sizeof(float));
                }
            }
            pemb.row(p.data(), nullptr, row.data());
            ggml_fp32_to_bf16_row(row.data(), d, PW);
        });
        io("px")->flush();
    }
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), outs(k.info->nOutputSize);
    for (uint32_t i = 0; i < k.info->nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pInputs[i];
        in[i] = axera_io(io(m.pName), m.nSize);
    }
    for (uint32_t i = 0; i < k.info->nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pOutputs[i];
        outs[i] = axera_io(io(m.pName, true), m.nSize);
    }
    {
        axera_timer tm("ocr vit (one call)");
        k.run(in, outs, "vit");
    }
    axera_cmm &  o = io("out", true);
    const size_t bytes = (size_t) (n / 4) * 1024 * sizeof(float);
    const int    OUTD = (int) (k.out_bytes("out") / sizeof(float) / (N / 4));
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] == n / 4 && OUTD == 1024);
    o->invalidate(o->vir, bytes);
    memcpy(out->data, o->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_minicpm_ocr_vit_full(const manifest & m) {
    return std::make_shared<minicpm_ocr_vit_full>(m);
}

bool minicpm_ocr_vit_graph(const ggml_cgraph * g) {
    for (int i = 0; i < g->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && g->nodes[i]->src[s]; ++s) {
            const char * n = g->nodes[i]->src[s]->name;
            const char * a = strchr(n, '#');
            if (strncmp(a ? a + 1 : n, "v.blk.0.attn_q.weight", 21) == 0) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace axera
