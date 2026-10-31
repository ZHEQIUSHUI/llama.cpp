// Axera NPU backend: the Qwen3.5 / Qwen3.8 vision encoder (mtmd clip graph, projector qwen3vl: 27 layers of 1152, 16
// heads of 72; 27B, 35B-A3B) in one call of the manifest's role vit (ax-dsl-kernels models/<model>/vit, vit_full: patch
// embedding, every layer, the 2 x 2 merge and projector for N patches, an image padded to N; the projector's width
// from the kernel's output).
//
// The host orders the patches in 2 x 2 merge order (as clip_graph_qwen3vl), resizes the learned position table
// (bilinear, align corners), makes the 2-D RoPE tables and the key mask: those depend on the image's size only and are
// kept for the next image of the same size. The layers' weights from the mmproj GGUF, stacked [layers, ...] bf16 in
// CMM once.

#include "../model.h"
#include "../../ops/ops.h"

#include <cmath>

namespace axera {
namespace {

class qwen35_vit_full final : public model {
  public:
    explicit qwen35_vit_full(const manifest & m) : man(m), k(man.get("vit")) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest &                  man;
    const kernel &                    k;
    graph_names                       names;
    int                               N = 0, QB = 0, BC = 0, HD = 0;
    std::map<std::string, axera_cmm>  buf;          // the kernel's IO by name (weights, tables, scratch, outputs)
    const void *                      bound = nullptr;
    int                               gh_ = -1, gw_ = -1;  // the image size of the tables in buf
    std::vector<int>                  py, px;       // the patches in merge order

    axera_cmm & io(const char * n, bool out = false) {
        auto & c = buf[n];
        if (!c) {
            c = axera_cmm_alloc(out ? k.out_bytes(n) : k.in_bytes(n));
        }
        return c;
    }
    void weights(int L, int H);
    void tables(int gh, int gw, int n, const ggml_tensor * pe, int H);
};

// bf16 rows of a weight of any type into dst (ne elements)
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

void qwen35_vit_full::weights(int L, int H) {
    axera_timer tm("vit weights (load)");
    // stacked per layer: kernel input -> mmproj tensor of v.blk.<l>.
    static const std::pair<const char *, const char *> per_layer[] = {
        { "ln1_w", "ln1.weight" }, { "ln1_b", "ln1.bias" }, { "w_qkv", "attn_qkv.weight" }, { "b_qkv", "attn_qkv.bias" },
        { "w_o", "attn_out.weight" }, { "b_o", "attn_out.bias" }, { "ln2_w", "ln2.weight" }, { "ln2_b", "ln2.bias" },
        { "w_up", "ffn_up.weight" }, { "b_up", "ffn_up.bias" }, { "w_down", "ffn_down.weight" }, { "b_down", "ffn_down.bias" },
    };
    for (const auto & [in, t] : per_layer) {
        axera_cmm &  c   = io(in);
        const size_t per = c->size / L;
        GGML_ASSERT(per * L == k.in_bytes(in) || c->size >= k.in_bytes(in));
        axera_host_pool().run(L, [&](int64_t l) {
            const ggml_tensor * w = names.at("v.blk." + std::to_string(l) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in) / L);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + l * (k.in_bytes(in) / L)));
        });
        c->flush();
    }
    static const std::pair<const char *, const char *> whole[] = {
        { "b_patch", "v.patch_embd.bias" }, { "post_w", "v.post_ln.weight" }, { "post_b", "v.post_ln.bias" },
        { "w0", "mm.0.weight" }, { "b0", "mm.0.bias" }, { "w2", "mm.2.weight" }, { "b2", "mm.2.bias" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        to_bf16(w, (ggml_bf16_t *) io(in)->vir);
        io(in)->flush();
    }
    // the patch projection: w0 + w1 (both temporal frames are the image), [H][3][PS][PS] f32 -> bf16 [H, CP]
    const ggml_tensor * w0 = names.at("v.patch_embd.weight"), *w1 = names.at("v.patch_embd.weight.1");
    GGML_ASSERT(w0->type == GGML_TYPE_F32 && w1->type == GGML_TYPE_F32 &&
                (size_t) ggml_nelements(w0) * sizeof(ggml_bf16_t) == k.in_bytes("w_patch"));
    auto * d = (ggml_bf16_t *) io("w_patch")->vir;
    for (int64_t i = 0; i < ggml_nelements(w0); ++i) {
        d[i] = ggml_fp32_to_bf16(((const float *) w0->data)[i] + ((const float *) w1->data)[i]);
    }
    io("w_patch")->flush();
    (void) H;
}

// the tables of an image of gh x gw patches (n real): merge order, positions, RoPE, key mask
void qwen35_vit_full::tables(int gh, int gw, int n, const ggml_tensor * pe, int H) {
    axera_timer tm("vit tables (cpu)");
    py.assign(N, 0);
    px.assign(N, 0);
    for (int y = 0, i = 0; y < gh; y += 2) {
        for (int x = 0; x < gw; x += 2) {
            for (int d = 0; d < 4; ++d, ++i) {
                py[i] = y + d / 2;
                px[i] = x + d % 2;
            }
        }
    }
    // learned positions [side x side][H] (x fastest), bilinear with align corners to gh x gw
    const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
    const float * tab  = (const float *) pe->data;
    auto *        pos  = (ggml_bf16_t *) io("pos")->vir;
    GGML_ASSERT(pe->type == GGML_TYPE_F32 && side * side == pe->ne[1]);
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pos + i * H;
        if (i >= n) {
            std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
            return;
        }
        const float   sy = gh > 1 ? (float) py[i] * (side - 1) / (gh - 1) : 0.0f;
        const float   sx = gw > 1 ? (float) px[i] * (side - 1) / (gw - 1) : 0.0f;
        const int     y0 = (int) sy, x0 = (int) sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
        const float   fy = sy - y0, fx = sx - x0;
        const float * a = tab + ((int64_t) y0 * side + x0) * H, *b = tab + ((int64_t) y0 * side + x1) * H;
        const float * c = tab + ((int64_t) y1 * side + x0) * H, *e = tab + ((int64_t) y1 * side + x1) * H;
        for (int j = 0; j < H; ++j) {
            d[j] = ggml_fp32_to_bf16((1 - fy) * ((1 - fx) * a[j] + fx * b[j]) + fy * ((1 - fx) * c[j] + fx * e[j]));
        }
    });
    io("pos")->flush();
    // RoPE: frequency j of HD / 4 for the row, then HD / 4 for the column, 10000^(-4 j / HD)
    std::vector<float> inv(HD / 4);
    for (int q = 0; q < HD / 4; ++q) {
        inv[q] = (float) std::pow(10000.0, -4.0 * q / HD);
    }
    auto * cs = (ggml_bf16_t *) io("cos")->vir, *sn = (ggml_bf16_t *) io("sin")->vir;
    axera_host_pool().run(N, [&](int64_t i) {
        for (int j = 0; j < HD / 2; ++j) {
            const float a = (float) (j < HD / 4 ? py[i] : px[i]) * inv[j % (HD / 4)];
            cs[i * (HD / 2) + j] = ggml_fp32_to_bf16(std::cos(a));
            sn[i * (HD / 2) + j] = ggml_fp32_to_bf16(std::sin(a));
        }
    });
    io("cos")->flush();
    io("sin")->flush();
    // keys: the n patches in nblk blocks, the padding of the last hidden ([N / BC][QB][BC], the key row repeated)
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

ggml_status qwen35_vit_full::compute(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");  // [W, H, 3, frames] f32, normalized
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     H = (int) pe->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0], CP = 3 * PS * PS;
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    if (N == 0) {
        const AX_ENGINE_IOMETA_T & km = k.info->pInputs[k.input("kmask")];  // [N / BC, QB, BC]
        const AX_ENGINE_IOMETA_T & cm = k.info->pInputs[k.input("cos")];    // [N, HD / 2]
        GGML_ASSERT(km.nShapeSize == 3 && cm.nShapeSize == 2);
        N  = (int) cm.pShape[0];
        QB = (int) km.pShape[1];
        BC = (int) km.pShape[2];
        HD = 2 * (int) cm.pShape[1];
        GGML_ASSERT(km.pShape[0] * BC == N && k.in_bytes("px") == (size_t) N * CP * sizeof(ggml_bf16_t));
    }
    if (raw->ne[3] != 1 || gw % 2 || gh % 2 || n > N) {
        GGML_LOG_ERROR("axera qwen35_vit_full: %d frame(s) of %d x %d patches: still images of even patch counts, at most %d "
                       "patches (mtmd --image-max-tokens)\n", (int) raw->ne[3], gw, gh, N);
        return GGML_STATUS_FAILED;
    }
    const ggml_tensor * w0 = names.at("v.blk.0.ln1.weight");
    if (bound != w0->data) {
        weights(L, H);
        bound = w0->data;
        gh_ = gw_ = -1;
    }
    if (gh != gh_ || gw != gw_) {
        tables(gh, gw, n, pe, H);
        gh_ = gh;
        gw_ = gw;
    }
    {
        axera_timer   tm("vit patches (cpu)");
        const float * img = (const float *) raw->data;
        const int64_t W = raw->ne[0], HI = raw->ne[1];
        auto *        pxb = (ggml_bf16_t *) io("px")->vir;
        axera_host_pool().run(N, [&](int64_t i) {
            ggml_bf16_t * d = pxb + i * CP;
            if (i >= n) {
                std::fill(d, d + CP, ggml_fp32_to_bf16(0.0f));
                return;
            }
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < PS; ++ky) {
                    const float * row = img + c * W * HI + (int64_t) (py[i] * PS + ky) * W + px[i] * PS;
                    ggml_fp32_to_bf16_row(row, d + (c * PS + ky) * PS, PS);
                }
            }
        });
        io("px")->flush();
    }
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), outs(k.info->nOutputSize);
    for (uint32_t i = 0; i < k.info->nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pInputs[i];
        in[i]                        = axera_io(io(m.pName), m.nSize);
    }
    for (uint32_t i = 0; i < k.info->nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pOutputs[i];
        outs[i]                      = axera_io(io(m.pName, true), m.nSize);
    }
    {
        axera_timer tm("vit (one call)");
        k.run(in, outs, "vit");
    }
    axera_cmm &  o     = io("out", true);
    const int    OUTD  = (int) (k.out_bytes("out") / sizeof(float) / (N / 4));
    const size_t bytes = (size_t) (n / 4) * OUTD * sizeof(float);
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] == n / 4);
    o->invalidate(o->vir, bytes);
    memcpy(out->data, o->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_vit_full(const manifest & m) {
    return std::make_shared<qwen35_vit_full>(m);
}

}  // namespace axera
