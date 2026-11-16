// Axera NPU backend: MiniCPM-V 4.7's SigLIP vision tower in one four-core call.
//
// One kernel runs patch embedding, encoder layers 0..6, the window-attention
// vit_merger (four patches -> one token), layers 7..26, post LayerNorm and the
// final 2x2 DownsampleMLP.  The host gives it patches in 2-level order
// (2x2 tile of windows -> window -> 2x2 patches), which keeps both merger
// group boundaries consecutive; global attention is permutation equivariant.

#include "../model.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace axera {
namespace {

class minicpmv4_7_vit_full final : public model {
  public:
    explicit minicpmv4_7_vit_full(const manifest & m) : man(m), k(man.get("vit")) {}
    ggml_status compute(ggml_cgraph * g) override;
    void        prewarm(ggml_cgraph * g) override { GGML_ASSERT(prepare(g)); }

  private:
    const manifest &                 man;
    const kernel &                   k;
    graph_names                      names;
    std::map<std::string, axera_cmm> buf;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, outs;
    bool                             io_bound = false;
    const void *                     bound = nullptr;
    int                              N = 0, N2 = 0, QB = 0, BC = 0, QB2 = 0, BC2 = 0, HD = 0;
    int                              gh_ = -1, gw_ = -1;

    axera_cmm & io(const char * n) {
        auto & c = buf[n];
        if (!c) {
            c = axera_cmm_alloc(k.find_output(n) >= 0 ? k.out_bytes(n) : k.in_bytes(n));
        }
        return c;
    }

    void weights(int L, int H);
    void tables(int gh, int gw, int n, const ggml_tensor * pe, int H);
    void bind_io();
    bool prepare(ggml_cgraph * g);
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

void minicpmv4_7_vit_full::weights(int L, int H) {
    axera_timer tm("minicpmv4_7 vit weights (load)");
    auto stack = [&](const char * in, const char * suffix) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / L;
        GGML_ASSERT(per * L == k.in_bytes(in));
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

    struct qkv { const char * in; const char * q; const char * k; const char * v; size_t elem; };
    for (const qkv & x : {qkv{"w_qkv", "attn_q.weight", "attn_k.weight", "attn_v.weight", (size_t) H * H},
                          qkv{"b_qkv", "attn_q.bias", "attn_k.bias", "attn_v.bias", (size_t) H}}) {
        axera_cmm &  c   = io(x.in);
        const size_t per = k.in_bytes(x.in) / L;
        GGML_ASSERT(per * L == k.in_bytes(x.in) && per == 3 * x.elem * sizeof(ggml_bf16_t));
        axera_host_pool().run(L, [&](int64_t l) {
            char * d = (char *) c->vir + l * per;
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.q), (ggml_bf16_t *) d);
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.k),
                    (ggml_bf16_t *) (d + x.elem * sizeof(ggml_bf16_t)));
            to_bf16(names.at("v.blk." + std::to_string(l) + "." + x.v),
                    (ggml_bf16_t *) (d + 2 * x.elem * sizeof(ggml_bf16_t)));
        });
        c->flush();
    }

    struct cat3 { const char * in; const char * q; const char * k; const char * v; size_t elem; };
    for (const cat3 & x : {cat3{"wm_qkv", "v.vit_merger.attn_q.weight", "v.vit_merger.attn_k.weight",
                                "v.vit_merger.attn_v.weight", (size_t) H * H},
                           cat3{"bwm_qkv", "v.vit_merger.attn_q.bias", "v.vit_merger.attn_k.bias",
                                "v.vit_merger.attn_v.bias", (size_t) H}}) {
        axera_cmm & c = io(x.in);
        GGML_ASSERT(c->size == 3 * x.elem * sizeof(ggml_bf16_t));
        to_bf16(names.at(x.q), (ggml_bf16_t *) c->vir);
        to_bf16(names.at(x.k), (ggml_bf16_t *) ((char *) c->vir + x.elem * sizeof(ggml_bf16_t)));
        to_bf16(names.at(x.v), (ggml_bf16_t *) ((char *) c->vir + 2 * x.elem * sizeof(ggml_bf16_t)));
        c->flush();
    }

    static const std::pair<const char *, const char *> whole[] = {
        { "b_patch", "v.patch_embd.bias" }, { "post_w", "v.post_ln.weight" }, { "post_b", "v.post_ln.bias" },
        { "mm_w", "mm.input_norm.weight" }, { "mm_b", "mm.input_norm.bias" },
        { "w0", "mm.up.weight" }, { "b0", "mm.up.bias" }, { "w2", "mm.down.weight" }, { "b2", "mm.down.bias" },
        { "wm_ln_w", "v.vit_merger.ln1.weight" }, { "wm_ln_b", "v.vit_merger.ln1.bias" },
        { "wm_o", "v.vit_merger.attn_out.weight" }, { "bwm_o", "v.vit_merger.attn_out.bias" },
        { "wm_pre_w", "v.vit_merger.ds_ln.weight" }, { "wm_pre_b", "v.vit_merger.ds_ln.bias" },
        { "wm_up_w", "v.vit_merger.ds_ffn_up.weight" }, { "wm_up_b", "v.vit_merger.ds_ffn_up.bias" },
        { "wm_down_w", "v.vit_merger.ds_ffn_down.weight" }, { "wm_down_b", "v.vit_merger.ds_ffn_down.bias" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        to_bf16(w, (ggml_bf16_t *) io(in)->vir);
        io(in)->flush();
    }

    const ggml_tensor * wp = names.at("v.patch_embd.weight");
    GGML_ASSERT((size_t) ggml_nelements(wp) == (size_t) H * 3 * 14 * 14);
    const size_t nw = (size_t) ggml_nelements(wp);
    if (wp->type == GGML_TYPE_BF16) {
        memcpy(io("w_patch")->vir, wp->data, nw * sizeof(ggml_bf16_t));
    } else if (wp->type == GGML_TYPE_F32) {
        ggml_fp32_to_bf16_row((const float *) wp->data, (ggml_bf16_t *) io("w_patch")->vir, nw);
    } else {
        std::vector<float> f(nw);
        ggml_get_type_traits(wp->type)->to_float(wp->data, f.data(), nw);
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) io("w_patch")->vir, nw);
    }
    io("w_patch")->flush();
}

void minicpmv4_7_vit_full::tables(int gh, int gw, int n, const ggml_tensor * pe, int H) {
    axera_timer tm("minicpmv4_7 vit tables (cpu)");
    const int side = (int) std::lround(std::sqrt((double) pe->ne[1]));
    GGML_ASSERT(side * side == pe->ne[1]);
    auto row_f32 = [&](int64_t r, ggml_bf16_t * d) {
        const char * p = (const char *) pe->data + (size_t) r * pe->nb[1];
        if (pe->type == GGML_TYPE_F32) {
            ggml_fp32_to_bf16_row((const float *) p, d, H);
        } else if (pe->type == GGML_TYPE_BF16) {
            memcpy(d, p, (size_t) H * sizeof(ggml_bf16_t));
        } else {
            std::vector<float> f(H);
            ggml_get_type_traits(pe->type)->to_float(p, f.data(), H);
            ggml_fp32_to_bf16_row(f.data(), d, H);
        }
    };
    auto * pos = (ggml_bf16_t *) io("pos")->vir;
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pos + i * H;
        if (i >= n) {
            std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
            return;
        }
        // 2-level order: a 2x2 tile of windows (16 patches), then the four windows, then the 2x2 patches.
        // The vit_merger groups the inner four (a window) and the final merger groups four consecutive
        // merged tokens (a tile of windows); global attention is permutation equivariant.
        const int tile = (int) (i / 16), win = (int) ((i / 4) % 4), pat = (int) (i % 4);
        const int y    = (tile / (gw / 4)) * 4 + (win / 2) * 2 + pat / 2;
        const int x    = (tile % (gw / 4)) * 4 + (win % 2) * 2 + pat % 2;
        const int by = (int) ((int64_t) y * side / gh);
        const int bx = (int) ((int64_t) x * side / gw);
        row_f32((int64_t) by * side + bx, d);
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

    const int n4 = n / 4;
    *(int32_t *) io("nblk2")->vir = (n4 + BC2 - 1) / BC2;
    io("nblk2")->flush();
    auto * km2 = (ggml_bf16_t *) io("kmask2")->vir;
    axera_host_pool().run(N2 / BC2, [&](int64_t b) {
        std::vector<ggml_bf16_t> row(BC2);
        for (int c = 0; c < BC2; ++c) {
            row[c] = ggml_fp32_to_bf16(b * BC2 + c < n4 ? 0.0f : -3e38f);
        }
        for (int r = 0; r < QB2; ++r) {
            memcpy(km2 + ((size_t) b * QB2 + r) * BC2, row.data(), BC2 * sizeof(ggml_bf16_t));
        }
    });
    io("kmask2")->flush();
}

bool minicpmv4_7_vit_full::prepare(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    const int     H = (int) pe->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0], CP = 3 * PS * PS;
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    if (N == 0) {
        N = (int) (k.in_bytes("px") / (CP * sizeof(ggml_bf16_t)));
        const AX_ENGINE_IOMETA_T & km = k.info->pInputs[k.input("kmask")];
        const AX_ENGINE_IOMETA_T & k2 = k.info->pInputs[k.input("kmask2")];
        GGML_ASSERT(km.nShapeSize == 3 && k2.nShapeSize == 3);
        QB  = (int) km.pShape[1];
        BC  = (int) km.pShape[2];
        QB2 = (int) k2.pShape[1];
        BC2 = (int) k2.pShape[2];
        N2  = (int) (k2.pShape[0] * BC2);
        HD  = 72;
        GGML_ASSERT(km.pShape[0] * BC == N && N2 * 4 == N);
    }
    if (raw->ne[3] != 1 || gw % 4 || gh % 4 || n > N || L != 27) {
        GGML_LOG_ERROR("axera minicpmv4_7_vit_full: %d frame(s) of %d x %d patches, %d layers, at most %d patches\n",
                       (int) raw->ne[3], gw, gh, L, N);
        return false;
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
    if (!io_bound) {
        bind_io();
    }
    return true;
}

void minicpmv4_7_vit_full::bind_io() {
    in.resize(k.info->nInputSize);
    outs.resize(k.info->nOutputSize);
    for (uint32_t i = 0; i < k.info->nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pInputs[i];
        in[i]                        = axera_io(io(m.pName), m.nSize);
    }
    for (uint32_t i = 0; i < k.info->nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = k.info->pOutputs[i];
        outs[i]                      = axera_io(io(m.pName), m.nSize);
    }
    io_bound = true;
}

ggml_status minicpmv4_7_vit_full::compute(ggml_cgraph * g) {
    if (!prepare(g)) {
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * raw = names.at("inp_raw");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     PS = (int) names.at("v.patch_embd.weight")->ne[0], CP = 3 * PS * PS;
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    {
        axera_timer   tm("minicpmv4_7 vit patch pack (cpu)");
        const float * img = (const float *) raw->data;
        const int64_t W = raw->ne[0], HI = raw->ne[1];
        auto *        pxb = (ggml_bf16_t *) io("px")->vir;
        axera_host_pool().run(N, [&](int64_t i) {
            ggml_bf16_t * d = pxb + i * CP;
            if (i >= n) {
                std::fill(d, d + CP, ggml_fp32_to_bf16(0.0f));
                return;
            }
            const int tile = (int) (i / 16), win = (int) ((i / 4) % 4), pat = (int) (i % 4);
            const int y    = (tile / (gw / 4)) * 4 + (win / 2) * 2 + pat / 2;
            const int x    = (tile % (gw / 4)) * 4 + (win % 2) * 2 + pat % 2;
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < PS; ++ky) {
                    const float * row = img + c * W * HI + (int64_t) (y * PS + ky) * W + x * PS;
                    ggml_fp32_to_bf16_row(row, d + (c * PS + ky) * PS, PS);
                }
            }
        });
    }
    {
        axera_timer tm("minicpmv4_7 vit patch pack flush");
        io("px")->flush();
    }
    {
        axera_timer tm("minicpmv4_7 vit (one call)");
        k.run(in, outs, "vit");
    }

    axera_cmm & o     = io("out");
    const int   OUTD  = (int) (k.out_bytes("out") / sizeof(float) / (N2 / 4));
    const int   nout  = n / 16;
    GGML_ASSERT(OUTD == 2048 && out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] >= nout);
    const size_t bytes = (size_t) nout * OUTD * sizeof(float);
    {
        axera_timer tm("minicpmv4_7 vit output invalidate");
        o->invalidate(o->vir, bytes);
    }
    {
        axera_timer tm("minicpmv4_7 vit output memcpy");
        memcpy(out->data, o->vir, bytes);
    }

    // build_suffix() appends rows of v.tok_embd_sep in the order the graph's
    // views appear; reproduce that tail from the same mmproj tensor.
    const ggml_tensor * sep = names.find("v.tok_embd_sep");
    const int           tail = (int) out->ne[1] - nout;
    if (tail > 0 && sep) {
        std::vector<int64_t> rows;
        for (int i = 0; i < g->n_nodes; ++i) {
            const ggml_tensor * v = g->nodes[i];
            if (v->op == GGML_OP_VIEW && v->src[0] == sep) {
                rows.push_back((int64_t) (v->view_offs / sep->nb[1]));
            }
        }
        GGML_ASSERT((int) rows.size() == tail);
        float * dst = (float *) out->data + (size_t) nout * OUTD;
        for (int64_t r : rows) {
            const char * src = (const char *) sep->data + (size_t) r * sep->nb[1];
            if (sep->type == GGML_TYPE_F32) {
                memcpy(dst, src, (size_t) OUTD * sizeof(float));
            } else if (sep->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, OUTD);
            } else {
                ggml_get_type_traits(sep->type)->to_float(src, dst, OUTD);
            }
            dst += OUTD;
        }
    } else {
        GGML_ASSERT(tail == 0);
    }
    const size_t all = (size_t) out->ne[1] * OUTD * sizeof(float);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, all);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_minicpmv4_7_vit_full(const manifest & m) {
    return std::make_shared<minicpmv4_7_vit_full>(m);
}

}  // namespace axera
