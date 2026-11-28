// Axera NPU backend: the Inkling-2-Mini vision tower (mtmd clip graph, projector inkling2) in one call of the
// manifest's role vit: patch embedding, class token, pre-LN, 23 layers with axial 2-D RoPE, exact GELU and
// LayerScale, the 2 x 2 merger and the bias-free projector. The host orders patches in 2 x 2-merge-block order,
// resizes the learned 32 x 32 position table bilinearly (align corners false), makes the RoPE table (width then
// height, pairs) and the key mask; the class token sits in the kernel's last row so the merger's groups of four
// patches stay aligned.  The layer weights come from the mmproj GGUF, stacked [layers, ...] bf16 in CMM once.

#include "../model.h"
#include "../../ops/ops.h"

#include <cmath>

namespace axera {
namespace {

class inkling2_vit_full final : public model {
  public:
    explicit inkling2_vit_full(const manifest & m) : man(m), k(man.get("vit")) {}
    ggml_status compute(ggml_cgraph * g) override;
    void        prewarm(ggml_cgraph * g) override;

  private:
    const manifest &                 man;
    const kernel &                   k;
    graph_names                      names;
    int                              N = 0, QB = 0, BC = 0, HD = 0;
    std::map<std::string, axera_cmm> buf;
    const void *                     bound = nullptr;
    int                              gh_ = -1, gw_ = -1;
    std::vector<int>                 py, px;

    axera_cmm & io(const char * n, bool out = false) {
        auto & c = buf[n];
        if (!c) {
            c = axera_cmm_alloc(out ? k.out_bytes(n) : k.in_bytes(n));
        }
        return c;
    }
    void shapes(int CP);
    void weights(int L, int H);
    void tables(int gh, int gw, int n, const ggml_tensor * pe, int H);
    void patches(const float * img, int64_t W, int64_t HI, int n, int H);
};

void to_bf16(const ggml_tensor * w, ggml_bf16_t * dst, int64_t ne) {
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

void inkling2_vit_full::weights(int L, int H) {
    axera_timer tm("inkling2 vit weights (load)");
    static const std::pair<const char *, const char *> per_layer[] = {
        { "ln1_w", "ln1.weight" }, { "ln1_b", "ln1.bias" }, { "ln2_w", "ln2.weight" }, { "ln2_b", "ln2.bias" },
        { "wq", "attn_q.weight" }, { "bq", "attn_q.bias" }, { "wk", "attn_k.weight" }, { "bk", "attn_k.bias" },
        { "wv", "attn_v.weight" }, { "bv", "attn_v.bias" }, { "wo", "attn_out.weight" }, { "bo", "attn_out.bias" },
        { "w_up", "ffn_up.weight" }, { "b_up", "ffn_up.bias" }, { "w_down", "ffn_down.weight" },
        { "b_down", "ffn_down.bias" }, { "ls1", "ls1.weight" }, { "ls2", "ls2.weight" },
    };
    for (const auto & [in, t] : per_layer) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / L;
        GGML_ASSERT(per * L == k.in_bytes(in));
        axera_host_pool().run(L, [&](int64_t l) {
            const ggml_tensor * w = names.at("v.blk." + std::to_string(l) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + l * per), ggml_nelements(w));
            axera_drop_mapped(w);
        });
        c->flush();
    }
    // The mmproj stores the patch projection as a conv kernel [kw, kh, ic, oc] for ggml_conv_2d; the
    // NPU kernel wants the HF linear layout [oc, (kh, kw, c)] with the channel fastest.
    {
        const ggml_tensor * w  = names.at("v.patch_embd.weight");
        const int           PS = 20, C = 3;
        GGML_ASSERT(w->ne[0] == PS && w->ne[1] == PS && w->ne[2] == C && w->ne[3] == H);
        std::vector<float> f((size_t) ggml_nelements(w));
        ggml_get_type_traits(w->type)->to_float(w->data, f.data(), ggml_nelements(w));
        auto * dst = (ggml_bf16_t *) io("w_patch")->vir;
        for (int oc = 0; oc < H; ++oc) {
            for (int kh = 0; kh < PS; ++kh) {
                for (int kw = 0; kw < PS; ++kw) {
                    for (int c = 0; c < C; ++c) {
                        const size_t si = (size_t) kw + PS * ((size_t) kh + PS * ((size_t) c + C * oc));
                        dst[((size_t) oc * PS + kh) * PS * C + (size_t) kw * C + c] = ggml_fp32_to_bf16(f[si]);
                    }
                }
            }
        }
        io("w_patch")->flush();
        axera_drop_mapped(w);
    }
    static const std::pair<const char *, const char *> whole[] = {
        { "pre_w", "v.pre_ln.weight" }, { "pre_b", "v.pre_ln.bias" },
        { "w0", "mm.0.weight" }, { "w2", "mm.2.weight" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        to_bf16(w, (ggml_bf16_t *) io(in)->vir, ggml_nelements(w));
        io(in)->flush();
        axera_drop_mapped(w);
    }
    axera_flush_file_pages();
    axera_keep_mapped(names.at("v.position_embd.weight"));
    axera_keep_mapped(names.at("v.class_embd"));
}

// order%s of an image of gh x gw patches (n real): merge-block order, class at row N-1
void inkling2_vit_full::tables(int gh, int gw, int n, const ggml_tensor * pe, int H) {
    axera_timer tm("inkling2 vit tables (cpu)");
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
    const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
    const float * tab  = (const float *) pe->data;
    auto *        pos  = (ggml_bf16_t *) io("pos")->vir;
    GGML_ASSERT(pe->type == GGML_TYPE_F32 && side * side == pe->ne[1]);
    const std::vector<float> cls = [&] {
        const ggml_tensor * c = names.at("v.class_embd");
        GGML_ASSERT(c->type == GGML_TYPE_F32 && c->ne[0] == H);
        return std::vector<float>((const float *) c->data, (const float *) c->data + H);
    }();
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pos + i * H;
        if (i == N - 1) {
            ggml_fp32_to_bf16_row(cls.data(), d, H);
            return;
        }
        if (i >= n) {
            std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
            return;
        }
        // bilinear, align corners false: s = (p + 0.5) * side / m - 0.5
        auto tap = [side](int p, int m, int & i0, int & i1, float & f) {
            float s = ((float) p + 0.5f) * (float) side / (float) m - 0.5f;
            s = std::max(0.0f, std::min(s, (float) (side - 1)));
            i0 = (int) s;
            i1 = std::min(i0 + 1, side - 1);
            f  = s - (float) i0;
        };
        int   y0, y1, x0, x1;
        float fy, fx;
        tap(py[i], gh, y0, y1, fy);
        tap(px[i], gw, x0, x1, fx);
        const float * a = tab + ((int64_t) y0 * side + x0) * H, *b = tab + ((int64_t) y0 * side + x1) * H;
        const float * c = tab + ((int64_t) y1 * side + x0) * H, *e = tab + ((int64_t) y1 * side + x1) * H;
        for (int j = 0; j < H; ++j) {
            d[j] = ggml_fp32_to_bf16((1 - fy) * ((1 - fx) * a[j] + fx * b[j]) + fy * ((1 - fx) * c[j] + fx * e[j]));
        }
    });
    io("pos")->flush();
    // RoPE angles: 16 width frequencies then 16 height, class at angle 0
    std::vector<float> inv(16);
    for (int q = 0; q < 16; ++q) {
        inv[q] = (float) std::pow(10000.0, -2.0 * q / 32.0);
    }
    const int HF = HD / 2;
    auto * cs = (ggml_bf16_t *) io("cos")->vir, *sn = (ggml_bf16_t *) io("sin")->vir;
    axera_host_pool().run(N, [&](int64_t i) {
        for (int j = 0; j < HF; ++j) {
            const float a = i == N - 1 ? 0.0f
                          : (float) (j < 16 ? px[i] + 1 : py[i] + 1) * inv[j % 16];
            cs[i * HF + j] = ggml_fp32_to_bf16(std::cos(a));
            sn[i * HF + j] = ggml_fp32_to_bf16(std::sin(a));
        }
    });
    io("cos")->flush();
    io("sin")->flush();
    *(int32_t *) io("nblk")->vir = (N + BC - 1) / BC;
    io("nblk")->flush();
    auto * km = (ggml_bf16_t *) io("kmask")->vir;
    axera_host_pool().run(N / BC, [&](int64_t b) {
        std::vector<ggml_bf16_t> row(BC);
        for (int c = 0; c < BC; ++c) {
            const int idx = (int) (b * BC + c);
            row[c] = ggml_fp32_to_bf16(idx < n || idx == N - 1 ? 0.0f : -3e38f);
        }
        for (int r = 0; r < QB; ++r) {
            memcpy(km + ((size_t) b * QB + r) * BC, row.data(), BC * sizeof(ggml_bf16_t));
        }
    });
    io("kmask")->flush();
}

void inkling2_vit_full::shapes(int CP) {
    if (N != 0) {
        return;
    }
    const AX_ENGINE_IOMETA_T & km = k.info->pInputs[k.input("kmask")];
    const AX_ENGINE_IOMETA_T & cm = k.info->pInputs[k.input("cos")];
    GGML_ASSERT(km.nShapeSize == 3 && cm.nShapeSize == 2);
    N  = (int) cm.pShape[0];
    QB = (int) km.pShape[1];
    BC = (int) km.pShape[2];
    HD = 2 * (int) cm.pShape[1];
    GGML_ASSERT(km.pShape[0] * BC == N && k.in_bytes("px") == (size_t) N * CP * sizeof(ggml_bf16_t));
}

void inkling2_vit_full::prewarm(ggml_cgraph * g) {
    names.of(g);
    if (!names.find("v.class_embd") || !names.find("v.position_embd.weight") || !names.find("v.patch_embd.weight") ||
        !names.find("v.blk.0.ln1.weight")) {
        return;
    }
    const int H  = (int) names.at("v.position_embd.weight")->ne[0];
    const int PS = 20;
    int       L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    const ggml_tensor * w0 = names.at("v.blk.0.ln1.weight");
    if (!w0->data || bound == w0->data) {
        return;
    }
    shapes(3 * PS * PS);
    weights(L, H);
    bound = w0->data;
    gh_ = gw_ = -1;
}

// px rows: one patch each, 20 x 20 x 3 with the channel innermost (the patch projection's learned order)
void inkling2_vit_full::patches(const float * img, int64_t W, int64_t HI, int n, int H) {
    axera_timer tm("inkling2 vit patches (cpu)");
    const int PS = 20, CP = 3 * PS * PS;
    auto *    pxb = (ggml_bf16_t *) io("px")->vir;
    GGML_ASSERT((int) (k.in_bytes("px") / sizeof(ggml_bf16_t) / N) == CP);
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pxb + i * CP;
        if (i >= n) {
            std::fill(d, d + CP, ggml_fp32_to_bf16(0.0f));
            return;
        }
        std::vector<float> p(CP);
        for (int ky = 0; ky < PS; ++ky) {
            for (int kx = 0; kx < PS; ++kx) {
                const int64_t s = (int64_t) (py[i] * PS + ky) * W + px[i] * PS + kx;
                for (int c = 0; c < 3; ++c) {
                    p[((ky * PS + kx) * 3) + c] = img[c * W * HI + s];
                }
            }
        }
        ggml_fp32_to_bf16_row(p.data(), d, CP);
    });
    io("px")->flush();
}

ggml_status inkling2_vit_full::compute(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");  // [W, H, 3, 1] f32, already normalized by mtmd
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     H = (int) pe->ne[0], PS = 20, CP = 3 * PS * PS;
    const int     gw_full = (int) raw->ne[0] / PS, gh_full = (int) raw->ne[1] / PS;
    // diagnostic: a top-left crop of crop_tok x crop_tok image tokens (2 x 2 patches each) reproduces the
    // simulator's real-weight crop check on the board with the same kernel
    const char *  crop_env = getenv("INKLING2_VIT_CROP");
    const int     crop_tok = crop_env ? atoi(crop_env) : 0;
    const int     gw = crop_tok > 0 ? crop_tok * 2 : gw_full;
    const int     gh = crop_tok > 0 ? crop_tok * 2 : gh_full;
    const int     n  = gw * gh;
    int           L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    shapes(CP);
    if (gw < 2 || gh < 2 || gw % 2 || gh % 2 || n > N - 1) {
        GGML_LOG_ERROR("axera inkling2_vit_full: %d x %d patches: even counts, at most %d patches\n", gw, gh, N - 1);
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
    patches((const float *) raw->data, raw->ne[0], raw->ne[1], n, H);
    if (const char * dump = getenv("INKLING2_VIT_IN_DUMP")) {
        // the exact NPU inputs (NPU (px, pos, cos, sin, kmask) + grid), for a bit-exact CRUN cross-check
        FILE * f = fopen(dump, "wb");
        if (f) {
            const int32_t hdr[4] = { 0x314b5649, N, gh, gw };
            fwrite(hdr, sizeof(hdr), 1, f);
            fwrite(&n, sizeof(n), 1, f);
            const std::pair<const char *, size_t> parts[] = {
                { "px", k.in_bytes("px") }, { "pos", k.in_bytes("pos") },
                { "cos", k.in_bytes("cos") }, { "sin", k.in_bytes("sin") },
                { "kmask", k.in_bytes("kmask") },
            };
            for (const auto & [name, bytes] : parts) {
                io(name)->invalidate(io(name)->vir, bytes);
                fwrite(io(name)->vir, 1, bytes, f);
            }
            fclose(f);
            GGML_LOG_INFO("axera inkling2_vit_full: wrote NPU inputs (%d x %d patches) to %s\n", gh, gw, dump);
        }
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
        axera_timer tm("inkling2 vit (one call)");
        k.run(in, outs, "vit");
    }
    axera_cmm &  o     = io("out", true);
    const int    OUTD  = (int) (k.out_bytes("out") / sizeof(float) / (N / 4));
    const size_t bytes = (size_t) (n / 4) * OUTD * sizeof(float);
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD);
    o->invalidate(o->vir, bytes);
    if (const char * dump = getenv("INKLING2_VIT_DUMP")) {
        FILE * f = fopen(dump, "wb");
        if (f) {
            fwrite(o->vir, 1, bytes, f);
            fclose(f);
            GGML_LOG_INFO("axera inkling2_vit_full: wrote %zu bytes of %d x %d embeddings to %s\n", bytes, n / 4, OUTD, dump);
        }
    }
    if (out->ne[1] != n / 4) {
        ggml_backend_buffer_t b = out->buffer;
        if (b) {
            memset(out->data, 0, ggml_nbytes(out));
        }
        if (b && ggml_backend_buffer_is_axera(b)) {
            ((ggml_backend_axera_buffer_context *) b->context)->flush(out->data, ggml_nbytes(out));
        }
        return GGML_STATUS_SUCCESS;
    }
    memcpy(out->data, o->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_inkling2_vit_full(const manifest & m) {
    return std::make_shared<inkling2_vit_full>(m);
}

}  // namespace axera
