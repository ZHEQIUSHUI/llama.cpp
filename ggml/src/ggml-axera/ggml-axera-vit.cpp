// Axera NPU backend: the Qwen3.5 vision encoder

#include "ggml-axera-impl.h"

// the Qwen3.5 vision encoder (mtmd clip graph, projector qwen3vl) on the q35_vit.py kernels
// qwen35_vit_{embed,pre,attn,post,merge}<N>_h<H>: N patches (padded), width H
struct axera_vit {
    int                 N = 0, H = 0, QB = 0, BC = 0;
    ggml_axera_kernel * embed = nullptr, *pre = nullptr, *attn = nullptr, *post = nullptr, *merge = nullptr;
    axera_cmm           PX, POS, COS, SIN, X[2], Q, K, V, ATT, NBLK, KMASK, OUT;
    std::map<const void *, axera_cmm> bf16;  // bf16 copies of non-bf16 weights, by their data
    axera_cmm           WPATCH;              // the two temporal patch kernels summed (a still image), bf16
    const void *        wpatch_of = nullptr;
};

// a graph is the vision encoder if it reads the first ViT layer's qkv weight
bool axera_vit_graph(const struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && cgraph->nodes[i]->src[s]; ++s) {
            if (axera_base_name(cgraph->nodes[i]->src[s]->name) == "v.blk.0.attn_qkv.weight") {
                return true;
            }
        }
    }
    return false;
}

// the clip graph's result (embeddings [OUT, n / 4]) from its input image: patches in 2 x 2 merge order (as
// clip_graph_qwen3vl), the learned position table bilinearly resized (align corners) and the 2-D RoPE tables on the
// host; patch projection, the layers and the merger on the NPU
enum ggml_status axera_vit_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
    std::map<std::string, ggml_tensor *> t;  // by name; the scheduler's input copies are named "AXERA#<name>#0"
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        t.emplace(axera_base_name(node->name), node);
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            t.emplace(axera_base_name(node->src[s]->name), node->src[s]);
        }
    }
    auto get = [&](const std::string & name) -> ggml_tensor * {
        auto it = t.find(name);
        if (it == t.end()) {
            GGML_ABORT("qwen35 vision: tensor %s not in the graph\n", name.c_str());
        }
        return it->second;
    };
    if (!ctx->vit) {
        ctx->vit = std::make_shared<axera_vit>();
    }
    axera_vit &   v   = *ctx->vit;
    ggml_tensor * raw = get("inp_raw");  // [W, H, 3, frames] f32, normalized
    ggml_tensor * pe  = get("v.position_embd.weight");
    ggml_tensor * out = cgraph->nodes[cgraph->n_nodes - 1];
    const int     H   = (int) pe->ne[0], PS = (int) get("v.patch_embd.weight")->ne[0];
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L   = 0;
    while (t.count("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    if (raw->ne[3] != 1 || gw % 2 || gh % 2) {
        GGML_ABORT("qwen35 vision: %d frame(s) of %d x %d patches: only still images with even patch counts\n",
                   (int) raw->ne[3], gw, gh);
    }
    // the smallest kernel patch count that holds the image
    int N = 0;
    for (int c : { 256, 512, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192 }) {
        if (N == 0 && c >= n && std::ifstream(axera_kernels().path("qwen35_vit_attn" + std::to_string(c) + "_h" + std::to_string(H))).good()) {
            N = c;
        }
    }
    if (N == 0) {
        GGML_ABORT("qwen35 vision: no qwen35_vit_*<N>_h%d kernels for %d patches (%d x %d); a smaller image (mtmd "
                   "--image-max-tokens) or the CPU (--no-mmproj-offload)\n", H, n, gw, gh);
    }
    auto name = [&](const char * k) { return std::string("qwen35_vit_") + k + std::to_string(N) + "_h" + std::to_string(H); };
    if (v.N != N || v.H != H) {
        v.N = N;
        v.H = H;
        v.embed = axera_kernels().get(name("embed"));
        v.pre   = axera_kernels().get(name("pre"));
        v.attn  = axera_kernels().get(name("attn"));
        v.post  = axera_kernels().get(name("post"));
        v.merge = axera_kernels().get(name("merge"));
        const AX_ENGINE_IOMETA_T & km = v.attn->info->pInputs[v.attn->input("kmask")];  // [N / BC, QB, BC]
        GGML_ASSERT(km.nShapeSize == 3 && km.pShape[0] * km.pShape[2] == N);
        v.QB = km.pShape[1];
        v.BC = km.pShape[2];
        v.PX    = axera_cmm_alloc(axera_input_size(v.embed, "px"));
        v.POS   = axera_cmm_alloc(axera_input_size(v.embed, "pos"));
        v.COS   = axera_cmm_alloc(axera_input_size(v.pre, "cos"));
        v.SIN   = axera_cmm_alloc(axera_input_size(v.pre, "sin"));
        v.X[0]  = axera_cmm_alloc(axera_output_size(v.embed, "out"));
        v.X[1]  = axera_cmm_alloc(axera_output_size(v.embed, "out"));
        v.Q     = axera_cmm_alloc(axera_output_size(v.pre, "q_out"));
        v.K     = axera_cmm_alloc(axera_output_size(v.pre, "k_out"));
        v.V     = axera_cmm_alloc(axera_output_size(v.pre, "v_out"));
        v.ATT   = axera_cmm_alloc(axera_output_size(v.attn, "out"));
        v.NBLK  = axera_cmm_alloc(axera_input_size(v.attn, "nblk"));
        v.KMASK = axera_cmm_alloc(axera_input_size(v.attn, "kmask"));
        v.OUT   = axera_cmm_alloc(axera_output_size(v.merge, "out"));
    }
    const AX_ENGINE_IOMETA_T & cm = v.pre->info->pInputs[v.pre->input("cos")];  // [N, HD / 2]
    const int HD = 2 * (int) cm.pShape[1], CP = 3 * PS * PS;
    const int OUTD = (int) (axera_output_size(v.merge, "out") / sizeof(float) / (N / 4));
    GGML_ASSERT(axera_input_size(v.embed, "px") == (size_t) N * CP * sizeof(ggml_bf16_t));
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] == n / 4);

    // weights: bf16 tensors in CMM as they are, others converted once
    auto weight = [&](const ggml_tensor * w) -> AX_ENGINE_IO_BUFFER_T {
        const int64_t ne = ggml_nelements(w);
        if (w->type == GGML_TYPE_BF16 && ggml_is_contiguous(w) && ggml_backend_buffer_is_axera(w->buffer)) {
            return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, (uint32_t) (ne * 2));
        }
        axera_cmm & c = v.bf16[w->data];
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
            c->flush(c->vir, c->size);
        }
        return axera_io(c.get(), c->vir, (uint32_t) (ne * 2));
    };
    {  // the patch projection: w0 + w1 (both temporal frames are the image), [H][3][PS][PS] f32 -> bf16 [H, CP]
        const ggml_tensor * w0 = get("v.patch_embd.weight"), *w1 = get("v.patch_embd.weight.1");
        GGML_ASSERT(w0->type == GGML_TYPE_F32 && w1->type == GGML_TYPE_F32 && ggml_nelements(w0) == (int64_t) H * CP);
        if (v.wpatch_of != w0->data) {
            v.WPATCH = axera_cmm_alloc((size_t) H * CP * sizeof(ggml_bf16_t));
            auto * d = (ggml_bf16_t *) v.WPATCH->vir;
            for (int64_t i = 0; i < (int64_t) H * CP; ++i) {
                d[i] = ggml_fp32_to_bf16(((const float *) w0->data)[i] + ((const float *) w1->data)[i]);
            }
            v.WPATCH->flush(v.WPATCH->vir, v.WPATCH->size);
            v.wpatch_of = w0->data;
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
        auto *        pxb = (ggml_bf16_t *) v.PX->vir;
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
        v.PX->flush(v.PX->vir, v.PX->size);
        // learned positions [side x side][H] (x fastest), bilinear with align corners to gh x gw
        const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
        const float * tab  = (const float *) pe->data;
        auto *        pos  = (ggml_bf16_t *) v.POS->vir;
        GGML_ASSERT(pe->type == GGML_TYPE_F32 && side * side == pe->ne[1]);
        axera_host_pool().run(N, [&](int64_t k) {
            ggml_bf16_t * d = pos + k * H;
            if (k >= n) {
                std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
                return;
            }
            const float sy = gh > 1 ? (float) py[k] * (side - 1) / (gh - 1) : 0.0f;
            const float sx = gw > 1 ? (float) px[k] * (side - 1) / (gw - 1) : 0.0f;
            const int   y0 = (int) sy, x0 = (int) sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
            const float fy = sy - y0, fx = sx - x0;
            const float * a = tab + ((int64_t) y0 * side + x0) * H, *b = tab + ((int64_t) y0 * side + x1) * H;
            const float * c = tab + ((int64_t) y1 * side + x0) * H, *e = tab + ((int64_t) y1 * side + x1) * H;
            for (int i = 0; i < H; ++i) {
                d[i] = ggml_fp32_to_bf16((1 - fy) * ((1 - fx) * a[i] + fx * b[i]) + fy * ((1 - fx) * c[i] + fx * e[i]));
            }
        });
        v.POS->flush(v.POS->vir, v.POS->size);
        // RoPE: frequency j of HD / 4 for the row, then HD / 4 for the column, 10000^(-4 j / HD)
        auto * cs = (ggml_bf16_t *) v.COS->vir, *sn = (ggml_bf16_t *) v.SIN->vir;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < HD / 2; ++j) {
                const int   q = j % (HD / 4);
                const float a = (float) (j < HD / 4 ? py[k] : px[k]) * (float) std::pow(10000.0, -4.0 * q / HD);
                cs[k * (HD / 2) + j] = ggml_fp32_to_bf16(std::cos(a));
                sn[k * (HD / 2) + j] = ggml_fp32_to_bf16(std::sin(a));
            }
        }
        v.COS->flush(v.COS->vir, v.COS->size);
        v.SIN->flush(v.SIN->vir, v.SIN->size);
        // keys: the n patches in nblk blocks, the padding of the last hidden
        *(int32_t *) v.NBLK->vir = (n + v.BC - 1) / v.BC;
        v.NBLK->flush(v.NBLK->vir, v.NBLK->size);
        auto * km = (ggml_bf16_t *) v.KMASK->vir;
        for (int b = 0; b < N / v.BC; ++b) {
            for (int r = 0; r < v.QB; ++r) {
                for (int c = 0; c < v.BC; ++c) {
                    km[((size_t) b * v.QB + r) * v.BC + c] = ggml_fp32_to_bf16(b * v.BC + c < n ? 0.0f : -3e38f);
                }
            }
        }
        v.KMASK->flush(v.KMASK->vir, v.KMASK->size);
    }
    // a kernel call with its IO by name
    using io_map = std::map<std::string, AX_ENGINE_IO_BUFFER_T>;
    auto cm_io = [](const axera_cmm & c) { return axera_io(c.get(), c->vir, (uint32_t) c->size); };
    auto run   = [&](ggml_axera_kernel * k, const io_map & io, const char * what) {
        // each IO at the kernel's size (CMM buffers may be larger)
        auto bind = [&](const AX_ENGINE_IOMETA_T & m) {
            auto it = io.find(m.pName);
            if (it == io.end() || it->second.nSize < m.nSize) {
                GGML_ABORT("qwen35 vision: %s: IO %s of %u bytes, %u given\n", what, m.pName, m.nSize,
                           it == io.end() ? 0u : it->second.nSize);
            }
            AX_ENGINE_IO_BUFFER_T b = it->second;
            b.nSize                 = m.nSize;
            return b;
        };
        std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), outs(k->info->nOutputSize);
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            in[i] = bind(k->info->pInputs[i]);
        }
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            outs[i] = bind(k->info->pOutputs[i]);
        }
        AX_ENGINE_IO_T s;
        memset(&s, 0, sizeof(s));
        s.pInputs     = in.data();
        s.nInputSize  = (uint32_t) in.size();
        s.pOutputs    = outs.data();
        s.nOutputSize = (uint32_t) outs.size();
        axera_timer tm(what);
        axera_warn_one_core(k);
        const int   ret = AX_ENGINE_RunSync(k->handle, &s);
        if (ret != 0) {
            GGML_ABORT("qwen35 vision: RunSync failed (0x%x) in %s\n", ret, what);
        }
    };
    auto blk = [&](int il, const char * n) { return weight(get("v.blk." + std::to_string(il) + "." + n)); };
    run(v.embed, { { "px", cm_io(v.PX) }, { "w_patch", cm_io(v.WPATCH) }, { "b_patch", weight(get("v.patch_embd.bias")) },
                   { "pos", cm_io(v.POS) }, { "out", cm_io(v.X[0]) } }, "vit embed");
    int cur = 0;
    for (int il = 0; il < L; ++il) {
        run(v.pre, { { "x", cm_io(v.X[cur]) }, { "ln1_w", blk(il, "ln1.weight") }, { "ln1_b", blk(il, "ln1.bias") },
                     { "w_qkv", blk(il, "attn_qkv.weight") }, { "b_qkv", blk(il, "attn_qkv.bias") }, { "cos", cm_io(v.COS) },
                     { "sin", cm_io(v.SIN) }, { "q_out", cm_io(v.Q) }, { "k_out", cm_io(v.K) }, { "v_out", cm_io(v.V) } },
            "vit pre");
        run(v.attn, { { "qs", cm_io(v.Q) }, { "kc", cm_io(v.K) }, { "vc", cm_io(v.V) }, { "nblk", cm_io(v.NBLK) },
                      { "kmask", cm_io(v.KMASK) }, { "out", cm_io(v.ATT) } }, "vit attn");
        run(v.post, { { "x", cm_io(v.X[cur]) }, { "attn", cm_io(v.ATT) }, { "w_o", blk(il, "attn_out.weight") },
                      { "b_o", blk(il, "attn_out.bias") }, { "ln2_w", blk(il, "ln2.weight") }, { "ln2_b", blk(il, "ln2.bias") },
                      { "w_up", blk(il, "ffn_up.weight") }, { "b_up", blk(il, "ffn_up.bias") },
                      { "w_down", blk(il, "ffn_down.weight") }, { "b_down", blk(il, "ffn_down.bias") },
                      { "out", cm_io(v.X[1 - cur]) } }, "vit post");
        cur = 1 - cur;
    }
    run(v.merge, { { "x", cm_io(v.X[cur]) }, { "ln_w", weight(get("v.post_ln.weight")) }, { "ln_b", weight(get("v.post_ln.bias")) },
                   { "w0", weight(get("mm.0.weight")) }, { "b0", weight(get("mm.0.bias")) }, { "w2", weight(get("mm.2.weight")) },
                   { "b2", weight(get("mm.2.bias")) }, { "out", cm_io(v.OUT) } }, "vit merge");
    v.OUT->invalidate(v.OUT->vir, v.OUT->size);
    const size_t bytes = (size_t) (n / 4) * OUTD * sizeof(float);
    memcpy(out->data, v.OUT->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}
