// Axera NPU backend: the Qwen3.5 vision encoder (mtmd clip graph, projector qwen3vl): 9B / 27B / 35B-A3B (27 layers
// of 1152, 16 heads of 72) on the manifest's ViT kernels (ax-dsl-kernels q35_vit.py, CORES=4)
//
//   vit.embed<N>   patch projection + the position table              (by rows: N / 4 patches per core)
//   vit.pre<N>     LayerNorm1, q / k / v, RoPE                         (by rows)
//   vit.attn<N>    bidirectional attention over the image's patches    (by heads: NH / 4 per core)
//   vit.post<N>    o projection, LayerNorm2, MLP, residuals            (by rows)
//   vit.merge<N>   post LayerNorm, 2 x 2 merge, projector MLP          (by rows)
// N: the patch count of the kernels (the smallest set that holds the image; the padding patches hidden from the keys).
// The host orders the patches in 2 x 2 merge order (as clip_graph_qwen3vl), resizes the learned position table
// (bilinear, align corners) and makes the 2-D RoPE tables. Weights from the mmproj GGUF (bf16 in CMM, else a copy).

#include "../model.h"
#include "../../ops/ops.h"

#include <cmath>

namespace axera {
namespace {

class qwen35_vit final : public model {
  public:
    explicit qwen35_vit(const manifest & m) : man(m) {
        for (const auto & [role, d] : m.kernels) {
            int n = 0;
            if (sscanf(role.c_str(), "vit.attn%d", &n) == 1) {
                sizes.push_back(n);
            }
        }
        std::sort(sizes.begin(), sizes.end());
        GGML_ASSERT(!sizes.empty());
    }

    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest &                      man;
    graph_names                           names;
    std::vector<int>                      sizes;  // the patch counts of the kernel sets
    int                                   N = 0, QB = 0, BC = 0;
    const kernel *                        embed = nullptr, *pre = nullptr, *attn = nullptr, *post = nullptr, *merge = nullptr;
    axera_cmm                             PX, POS, COS, SIN, X[2], Q, K, V, ATT, NBLK, KMASK, OUT, WPATCH;
    const void *                          wpatch_of = nullptr;
    std::map<const void *, axera_cmm>     bf16;  // bf16 copies of the weights not in CMM as bf16, by their data

    AX_ENGINE_IO_BUFFER_T weight(const ggml_tensor * w);
    void run(const kernel & k, const std::map<std::string, AX_ENGINE_IO_BUFFER_T> & io, const char * what);
};

AX_ENGINE_IO_BUFFER_T qwen35_vit::weight(const ggml_tensor * w) {
    const int64_t ne = ggml_nelements(w);
    if (w->type == GGML_TYPE_BF16 && ggml_is_contiguous(w) && ggml_backend_buffer_is_axera(w->buffer)) {
        return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, (uint32_t) (ne * 2));
    }
    axera_cmm & c = bf16[w->data];
    if (!c) {
        c          = axera_cmm_alloc(ne * sizeof(ggml_bf16_t));
        auto * dst = (ggml_bf16_t *) c->vir;
        if (w->type == GGML_TYPE_F32) {
            ggml_fp32_to_bf16_row((const float *) w->data, dst, ne);
        } else if (w->type == GGML_TYPE_BF16) {
            memcpy(dst, w->data, ne * sizeof(ggml_bf16_t));
        } else {
            std::vector<float> f(ne);
            ggml_get_type_traits(w->type)->to_float(w->data, f.data(), ne);
            ggml_fp32_to_bf16_row(f.data(), dst, ne);
        }
        c->flush();
    }
    return axera_io(c, ne * 2);
}

// a call with its IO by name, each at the kernel's size (the buffers may be larger)
void qwen35_vit::run(const kernel & k, const std::map<std::string, AX_ENGINE_IO_BUFFER_T> & io, const char * what) {
    auto bind = [&](const AX_ENGINE_IOMETA_T & m) {
        auto it = io.find(m.pName);
        if (it == io.end() || it->second.nSize < m.nSize) {
            GGML_ABORT("axera qwen35_vit: %s: IO %s of %u bytes, %u given\n", what, m.pName, m.nSize,
                       it == io.end() ? 0u : it->second.nSize);
        }
        AX_ENGINE_IO_BUFFER_T b = it->second;
        b.nSize                 = m.nSize;
        return b;
    };
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), out(k.info->nOutputSize);
    for (uint32_t i = 0; i < k.info->nInputSize; ++i) {
        in[i] = bind(k.info->pInputs[i]);
    }
    for (uint32_t i = 0; i < k.info->nOutputSize; ++i) {
        out[i] = bind(k.info->pOutputs[i]);
    }
    axera_timer tm(what);
    k.run(in, out, what);
}

ggml_status qwen35_vit::compute(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");  // [W, H, 3, frames] f32, normalized
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     H = (int) pe->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0];
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    if (raw->ne[3] != 1 || gw % 2 || gh % 2) {
        GGML_LOG_ERROR("axera qwen35_vit: %d frame(s) of %d x %d patches: still images with even patch counts only\n",
                       (int) raw->ne[3], gw, gh);
        return GGML_STATUS_FAILED;
    }
    int Nn = 0;  // the smallest kernel set that holds the image
    for (int c : sizes) {
        Nn = Nn == 0 && c >= n ? c : Nn;
    }
    if (Nn == 0) {
        GGML_LOG_ERROR("axera qwen35_vit: %d patches (%d x %d), the kernels hold %d: a smaller image (mtmd "
                       "--image-max-tokens)\n", n, gw, gh, sizes.back());
        return GGML_STATUS_FAILED;
    }
    if (N != Nn) {
        N     = Nn;
        const std::string sn = std::to_string(N);
        embed = &man.get("vit.embed" + sn);
        pre   = &man.get("vit.pre" + sn);
        attn  = &man.get("vit.attn" + sn);
        post  = &man.get("vit.post" + sn);
        merge = &man.get("vit.merge" + sn);
        const AX_ENGINE_IOMETA_T & km = attn->info->pInputs[attn->input("kmask")];  // [N / BC, QB, BC]
        GGML_ASSERT(km.nShapeSize == 3 && km.pShape[0] * km.pShape[2] == N);
        QB    = km.pShape[1];
        BC    = km.pShape[2];
        PX    = axera_cmm_alloc(embed->in_bytes("px"));
        POS   = axera_cmm_alloc(embed->in_bytes("pos"));
        COS   = axera_cmm_alloc(pre->in_bytes("cos"));
        SIN   = axera_cmm_alloc(pre->in_bytes("sin"));
        X[0]  = axera_cmm_alloc(embed->out_bytes("out"));
        X[1]  = axera_cmm_alloc(embed->out_bytes("out"));
        Q     = axera_cmm_alloc(pre->out_bytes("q_out"));
        K     = axera_cmm_alloc(pre->out_bytes("k_out"));
        V     = axera_cmm_alloc(pre->out_bytes("v_out"));
        ATT   = axera_cmm_alloc(attn->out_bytes("out"));
        NBLK  = axera_cmm_alloc(attn->in_bytes("nblk"));
        KMASK = axera_cmm_alloc(attn->in_bytes("kmask"));
        OUT   = axera_cmm_alloc(merge->out_bytes("out"));
    }
    const AX_ENGINE_IOMETA_T & cm = pre->info->pInputs[pre->input("cos")];  // [N, HD / 2]
    const int HD = 2 * (int) cm.pShape[1], CP = 3 * PS * PS;
    const int OUTD = (int) (merge->out_bytes("out") / sizeof(float) / (N / 4));
    GGML_ASSERT(embed->in_bytes("px") == (size_t) N * CP * sizeof(ggml_bf16_t));
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] == n / 4);
    {  // the patch projection: w0 + w1 (both temporal frames are the image), [H][3][PS][PS] f32 -> bf16 [H, CP]
        const ggml_tensor * w0 = names.at("v.patch_embd.weight"), *w1 = names.at("v.patch_embd.weight.1");
        GGML_ASSERT(w0->type == GGML_TYPE_F32 && w1->type == GGML_TYPE_F32 && ggml_nelements(w0) == (int64_t) H * CP);
        if (wpatch_of != w0->data) {
            WPATCH = axera_cmm_alloc((size_t) H * CP * sizeof(ggml_bf16_t));
            auto * d = (ggml_bf16_t *) WPATCH->vir;
            for (int64_t i = 0; i < (int64_t) H * CP; ++i) {
                d[i] = ggml_fp32_to_bf16(((const float *) w0->data)[i] + ((const float *) w1->data)[i]);
            }
            WPATCH->flush();
            wpatch_of = w0->data;
        }
    }
    // the patches in merge order: groups of 2 x 2 (row-major over the groups), each (y, x) (y, x+1) (y+1, x) (y+1, x+1)
    std::vector<int> py(N, 0), px(N, 0);
    for (int y = 0, k = 0; y < gh; y += 2) {
        for (int x = 0; x < gw; x += 2) {
            for (int d = 0; d < 4; ++d, ++k) {
                py[k] = y + d / 2;
                px[k] = x + d % 2;
            }
        }
    }
    {
        axera_timer   tm("vit prepare (cpu)");
        const float * img = (const float *) raw->data;
        const int64_t W = raw->ne[0], HI = raw->ne[1];
        auto *        pxb = (ggml_bf16_t *) PX->vir;
        axera_host_pool().run(N, [&](int64_t k) {
            ggml_bf16_t * d = pxb + k * CP;
            if (k >= n) {
                std::fill(d, d + CP, ggml_fp32_to_bf16(0.0f));
                return;
            }
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < PS; ++ky) {
                    const float * row = img + c * W * HI + (int64_t) (py[k] * PS + ky) * W + px[k] * PS;
                    for (int kx = 0; kx < PS; ++kx) {
                        d[(c * PS + ky) * PS + kx] = ggml_fp32_to_bf16(row[kx]);
                    }
                }
            }
        });
        PX->flush();
        // learned positions [side x side][H] (x fastest), bilinear with align corners to gh x gw
        const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
        const float * tab  = (const float *) pe->data;
        auto *        pos  = (ggml_bf16_t *) POS->vir;
        GGML_ASSERT(pe->type == GGML_TYPE_F32 && side * side == pe->ne[1]);
        axera_host_pool().run(N, [&](int64_t k) {
            ggml_bf16_t * d = pos + k * H;
            if (k >= n) {
                std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
                return;
            }
            const float   sy = gh > 1 ? (float) py[k] * (side - 1) / (gh - 1) : 0.0f;
            const float   sx = gw > 1 ? (float) px[k] * (side - 1) / (gw - 1) : 0.0f;
            const int     y0 = (int) sy, x0 = (int) sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
            const float   fy = sy - y0, fx = sx - x0;
            const float * a = tab + ((int64_t) y0 * side + x0) * H, *b = tab + ((int64_t) y0 * side + x1) * H;
            const float * c = tab + ((int64_t) y1 * side + x0) * H, *e = tab + ((int64_t) y1 * side + x1) * H;
            for (int i = 0; i < H; ++i) {
                d[i] = ggml_fp32_to_bf16((1 - fy) * ((1 - fx) * a[i] + fx * b[i]) + fy * ((1 - fx) * c[i] + fx * e[i]));
            }
        });
        POS->flush();
        // RoPE: frequency j of HD / 4 for the row, then HD / 4 for the column, 10000^(-4 j / HD)
        auto * cs = (ggml_bf16_t *) COS->vir, *sn = (ggml_bf16_t *) SIN->vir;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < HD / 2; ++j) {
                const int   q = j % (HD / 4);
                const float a = (float) (j < HD / 4 ? py[k] : px[k]) * (float) std::pow(10000.0, -4.0 * q / HD);
                cs[k * (HD / 2) + j] = ggml_fp32_to_bf16(std::cos(a));
                sn[k * (HD / 2) + j] = ggml_fp32_to_bf16(std::sin(a));
            }
        }
        COS->flush();
        SIN->flush();
        // keys: the n patches in nblk blocks, the padding of the last hidden
        *(int32_t *) NBLK->vir = (n + BC - 1) / BC;
        NBLK->flush();
        auto * km = (ggml_bf16_t *) KMASK->vir;
        for (int b = 0; b < N / BC; ++b) {
            for (int r = 0; r < QB; ++r) {
                for (int c = 0; c < BC; ++c) {
                    km[((size_t) b * QB + r) * BC + c] = ggml_fp32_to_bf16(b * BC + c < n ? 0.0f : -3e38f);
                }
            }
        }
        KMASK->flush();
    }
    auto io  = [](const axera_cmm & c) { return axera_io(c); };
    auto blk = [&](int il, const char * nm) { return weight(names.at("v.blk." + std::to_string(il) + "." + nm)); };
    run(*embed, { { "px", io(PX) }, { "w_patch", io(WPATCH) }, { "b_patch", weight(names.at("v.patch_embd.bias")) },
                  { "pos", io(POS) }, { "out", io(X[0]) } }, "vit embed");
    int cur = 0;
    for (int il = 0; il < L; ++il) {
        run(*pre, { { "x", io(X[cur]) }, { "ln1_w", blk(il, "ln1.weight") }, { "ln1_b", blk(il, "ln1.bias") },
                    { "w_qkv", blk(il, "attn_qkv.weight") }, { "b_qkv", blk(il, "attn_qkv.bias") }, { "cos", io(COS) },
                    { "sin", io(SIN) }, { "q_out", io(Q) }, { "k_out", io(K) }, { "v_out", io(V) } }, "vit pre");
        run(*attn, { { "qs", io(Q) }, { "kc", io(K) }, { "vc", io(V) }, { "nblk", io(NBLK) }, { "kmask", io(KMASK) },
                     { "out", io(ATT) } }, "vit attn");
        run(*post, { { "x", io(X[cur]) }, { "attn", io(ATT) }, { "w_o", blk(il, "attn_out.weight") },
                     { "b_o", blk(il, "attn_out.bias") }, { "ln2_w", blk(il, "ln2.weight") }, { "ln2_b", blk(il, "ln2.bias") },
                     { "w_up", blk(il, "ffn_up.weight") }, { "b_up", blk(il, "ffn_up.bias") },
                     { "w_down", blk(il, "ffn_down.weight") }, { "b_down", blk(il, "ffn_down.bias") }, { "out", io(X[1 - cur]) } },
            "vit post");
        cur = 1 - cur;
    }
    run(*merge, { { "x", io(X[cur]) }, { "ln_w", weight(names.at("v.post_ln.weight")) },
                  { "ln_b", weight(names.at("v.post_ln.bias")) }, { "w0", weight(names.at("mm.0.weight")) },
                  { "b0", weight(names.at("mm.0.bias")) }, { "w2", weight(names.at("mm.2.weight")) },
                  { "b2", weight(names.at("mm.2.bias")) }, { "out", io(OUT) } }, "vit merge");
    OUT->invalidate();
    const size_t bytes = (size_t) (n / 4) * OUTD * sizeof(float);
    memcpy(out->data, OUT->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_vit(const manifest & m) {
    return std::make_shared<qwen35_vit>(m);
}

}  // namespace axera
