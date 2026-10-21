// Axera NPU backend: the Qwen3-ASR audio encoder

#include "ggml-axera-impl.h"

// the Qwen3-ASR audio encoder (mtmd clip graph, projector qwen3a): the transformer layers on the q3a_audio.py kernels
// (q35_vit.py's pre / attn / post on the audio tower: qwen3a_{pre,attn,post}<N>_h<H>, N rows padded, width H), the
// conv front-end before them and the post norm and projector after them as the graph's own nodes on the CPU backend
struct axera_audio {
    int                 N = 0, H = 0, QB = 0, BC = 0;
    ggml_axera_kernel * pre = nullptr, *attn = nullptr, *post = nullptr;
    axera_cmm           COS, SIN, X[2], Q, K, V, ATT, NBLK, KMASK;
    std::map<std::string, axera_cmm> cw;  // conv kernels in bf16 [OC][K] by name, for the front-end GEMMs
    std::map<std::string, std::pair<axera_cmm, axera_cmm>> cio;  // front-end GEMM x / out buffers by kernel
    std::map<std::string, axera_cmm> w;  // bf16 weights by name (q / k / v joined as attn_qkv), from bound's model
    const void *        bound = nullptr;  // data of a.blk.0.attn_q.weight
};

// a graph is the audio encoder if it reads the conv front-end's output projection
bool axera_audio_graph(const struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && cgraph->nodes[i]->src[s]; ++s) {
            if (axera_base_name(cgraph->nodes[i]->src[s]->name) == "a.conv_out.weight") {
                return true;
            }
        }
    }
    return false;
}

// the conv front-end, nodes [0, n): on the CPU, except the matrix products of the 3 x 3 convolutions with a
// qwen3a_conv_o<OC>_k<K>_r<R> kernel (q3a_conv.py): im2col rows [rows][K] f16 x the kernel [OC][K] f16 -> [OC][rows]
// f32 (ggml's layout of that product) on the NPU, rows padded to R
static void axera_audio_frontend(ggml_backend_axera_context * ctx, axera_audio & a, struct ggml_cgraph * cgraph, int n) {
    int i0 = 0;
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT || !node->src[1]->view_src) {
            continue;
        }
        const ggml_tensor * w  = node->src[1]->view_src;  // the conv kernel, reshaped to [K, OC]
        const ggml_tensor * x  = node->src[0];             // im2col, reshaped to [K, rows]
        const std::string   wn = axera_base_name(w->name);
        if (wn.rfind("a.conv2d.", 0) != 0 || w->type != GGML_TYPE_F16 || x->type != GGML_TYPE_F16 ||
            !ggml_is_contiguous(x) || node->type != GGML_TYPE_F32) {
            continue;
        }
        const int64_t K = x->ne[0], rows = x->ne[1], OC = node->ne[1];
        GGML_ASSERT(node->ne[0] == rows && node->src[1]->ne[0] == K && node->src[1]->ne[1] == OC);
        std::string kn;
        int64_t     R = 0;
        for (int64_t r : { 208, 416, 832, 1664, 800, 1600, 3200, 6400 }) {
            const std::string c = "qwen3a_conv_o" + std::to_string(OC) + "_k" + std::to_string(K) + "_r" + std::to_string(r);
            if (r >= rows && (R == 0 || r < R) && std::ifstream(axera_kernels().path(c)).good()) {
                kn = c;
                R  = r;
            }
        }
        if (R == 0) {
            continue;
        }
        if (i > i0) {
            axera_cpu_compute(ctx, cgraph, i0, i, "audio front-end (cpu)");
        }
        i0                    = i + 1;
        ggml_axera_kernel * k = axera_kernels().get(kn);
        axera_cmm &         W = a.cw[wn];
        if (!W) {
            W = axera_cmm_alloc((size_t) OC * K * sizeof(ggml_bf16_t));
            std::vector<float> f((size_t) OC * K);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) w->data, f.data(), OC * K);
            ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) W->vir, OC * K);
            W->flush(W->vir, W->size);
        }
        auto & io = a.cio[kn];
        if (!io.first) {
            io.first  = axera_cmm_alloc(axera_input_size(k, "x"));
            io.second = axera_cmm_alloc(axera_output_size(k, "out"));
        }
        GGML_ASSERT(io.first->size >= (size_t) R * K * sizeof(ggml_bf16_t) && io.second->size >= (size_t) OC * R * sizeof(float));
        {
            axera_timer tm("audio conv x (cpu)");
            auto *      xb = (ggml_bf16_t *) io.first->vir;
            axera_host_pool().run(R, [&](int64_t r) {
                ggml_bf16_t * d = xb + r * K;
                if (r >= rows) {
                    std::fill(d, d + K, ggml_fp32_to_bf16(0.0f));
                    return;
                }
                std::vector<float> f(K);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) ((const char *) x->data + r * x->nb[1]), f.data(), K);
                ggml_fp32_to_bf16_row(f.data(), d, K);
            });
            io.first->flush(io.first->vir, io.first->size);
        }
        std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), out(k->info->nOutputSize);
        in[k->input("w")]     = axera_io(W.get(), W->vir, axera_input_size(k, "w"));
        in[k->input("x")]     = axera_io(io.first.get(), io.first->vir, axera_input_size(k, "x"));
        out[k->output("out")] = axera_io(io.second.get(), io.second->vir, axera_output_size(k, "out"));
        {
            axera_timer tm("audio conv (npu)");
            const int   ret = axera_run_sync(k, in, out);
            if (ret != 0) {
                GGML_ABORT("qwen3 audio: RunSync failed (0x%x) in %s\n", ret, kn.c_str());
            }
        }
        io.second->invalidate(io.second->vir, io.second->size);
        for (int64_t o = 0; o < OC; ++o) {
            memcpy((char *) node->data + o * node->nb[1], (const float *) io.second->vir + o * R, rows * sizeof(float));
        }
    }
    if (n > i0) {
        axera_cpu_compute(ctx, cgraph, i0, n, "audio front-end (cpu)");
    }
}

enum ggml_status axera_audio_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
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
            GGML_ABORT("qwen3 audio: tensor %s not in the graph\n", name.c_str());
        }
        return it->second;
    };
    if (!ctx->audio) {
        ctx->audio = std::make_shared<axera_audio>();
    }
    axera_audio & a = *ctx->audio;
    int           L = 0;
    while (t.count("a.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    // the layers' input: the source of the NORM under layer 0's ln1, and their output: the source of the post norm's
    auto norm_input = [&](const std::string & w, int * idx) -> ggml_tensor * {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * mul = cgraph->nodes[i];
            if (mul->op != GGML_OP_MUL || !mul->src[1] || axera_base_name(mul->src[1]->name) != w) {
                continue;
            }
            ggml_tensor * norm = mul->src[0];
            GGML_ASSERT(norm->op == GGML_OP_NORM);
            for (int j = 0; j < cgraph->n_nodes; j++) {
                if (cgraph->nodes[j] == norm) {
                    *idx = j;
                }
            }
            return norm->src[0];
        }
        GGML_ABORT("qwen3 audio: no norm with weight %s\n", w.c_str());
    };
    int           in_norm = -1, post_norm = -1;
    ggml_tensor * x0      = norm_input("a.blk.0.ln1.weight", &in_norm);
    ggml_tensor * xl      = norm_input("a.post_ln.weight", &post_norm);
    const int     H = (int) x0->ne[0], n = (int) x0->ne[1];
    GGML_ASSERT(x0->type == GGML_TYPE_F32 && xl->type == GGML_TYPE_F32 && ggml_is_contiguous(x0) &&
                ggml_is_contiguous(xl) && ggml_are_same_shape(x0, xl) && in_norm >= 0 && post_norm > in_norm);
    int x0_node = -1;
    for (int i = 0; i < in_norm; i++) {
        x0_node = cgraph->nodes[i] == x0 ? i : x0_node;
    }
    GGML_ASSERT(x0_node >= 0 && "qwen3 audio: the layers' input is not a node before them");

    // the smallest kernel row count that holds the n rows; none: the whole graph on the CPU
    int N = 0;
    for (int c : { 256, 512, 1024, 2048, 4096 }) {
        if (N == 0 && c >= n && std::ifstream(axera_kernels().path("qwen3a_attn" + std::to_string(c) + "_h" + std::to_string(H))).good()) {
            N = c;
        }
    }
    if (N == 0) {
        GGML_LOG_WARN("%s: no qwen3a_*<N>_h%d kernels for %d rows: the audio encoder on the CPU\n", __func__, H, n);
        axera_cpu_compute(ctx, cgraph, 0, cgraph->n_nodes, "audio (cpu)");
        return GGML_STATUS_SUCCESS;
    }
    auto name = [&](const char * k) { return std::string("qwen3a_") + k + std::to_string(N) + "_h" + std::to_string(H); };
    if (a.N != N || a.H != H) {
        a.N    = N;
        a.H    = H;
        a.pre  = axera_kernels().get(name("pre"));
        a.attn = axera_kernels().get(name("attn"));
        a.post = axera_kernels().get(name("post"));
        const AX_ENGINE_IOMETA_T & km = a.attn->info->pInputs[a.attn->input("kmask")];  // [N / BC, QB, BC]
        GGML_ASSERT(km.nShapeSize == 3 && km.pShape[0] * km.pShape[2] == N);
        a.QB    = km.pShape[1];
        a.BC    = km.pShape[2];
        a.COS   = axera_cmm_alloc(axera_input_size(a.pre, "cos"));
        a.SIN   = axera_cmm_alloc(axera_input_size(a.pre, "sin"));
        a.X[0]  = axera_cmm_alloc(axera_input_size(a.pre, "x"));
        a.X[1]  = axera_cmm_alloc(axera_input_size(a.pre, "x"));
        a.Q     = axera_cmm_alloc(axera_output_size(a.pre, "q_out"));
        a.K     = axera_cmm_alloc(axera_output_size(a.pre, "k_out"));
        a.V     = axera_cmm_alloc(axera_output_size(a.pre, "v_out"));
        a.ATT   = axera_cmm_alloc(axera_output_size(a.attn, "out"));
        a.NBLK  = axera_cmm_alloc(axera_input_size(a.attn, "nblk"));
        a.KMASK = axera_cmm_alloc(axera_input_size(a.attn, "kmask"));
        // no RoPE in the audio tower: the pre kernel's rotation with cos 1, sin 0
        auto * cs = (ggml_bf16_t *) a.COS->vir, *sn = (ggml_bf16_t *) a.SIN->vir;
        for (size_t i = 0; i < a.COS->size / sizeof(ggml_bf16_t); ++i) {
            cs[i] = ggml_fp32_to_bf16(1.0f);
            sn[i] = ggml_fp32_to_bf16(0.0f);
        }
        a.COS->flush(a.COS->vir, a.COS->size);
        a.SIN->flush(a.SIN->vir, a.SIN->size);
    }
    GGML_ASSERT(axera_input_size(a.pre, "x") == (size_t) N * H * sizeof(ggml_bf16_t));

    // the layer weights in bf16, converted once per model (q / k / v rows joined for the pre kernel's w_qkv)
    if (a.bound != get("a.blk.0.attn_q.weight")->data) {
        a.w.clear();
        a.bound = get("a.blk.0.attn_q.weight")->data;
    }
    auto weight = [&](int il, const std::string & n) -> AX_ENGINE_IO_BUFFER_T {
        const std::string key = "a.blk." + std::to_string(il) + "." + n;
        axera_cmm &       c   = a.w[key];
        if (!c) {
            const bool                       qkv = n == "attn_qkv.weight" || n == "attn_qkv.bias";
            std::vector<const ggml_tensor *> parts;
            for (const char * p : { "attn_q", "attn_k", "attn_v" }) {
                if (qkv) {
                    parts.push_back(get("a.blk." + std::to_string(il) + "." + p + n.substr(8)));
                }
            }
            if (!qkv) {
                parts.push_back(get(key));
            }
            int64_t ne = 0;
            for (const ggml_tensor * p : parts) {
                ne += ggml_nelements(p);
            }
            c          = axera_cmm_alloc(ne * sizeof(ggml_bf16_t));
            auto * dst = (ggml_bf16_t *) c->vir;
            for (const ggml_tensor * p : parts) {
                const int64_t      m = ggml_nelements(p);
                std::vector<float> f(m);
                if (p->type == GGML_TYPE_F32) {
                    memcpy(f.data(), p->data, m * sizeof(float));
                } else {
                    ggml_get_type_traits(p->type)->to_float(p->data, f.data(), m);
                }
                ggml_fp32_to_bf16_row(f.data(), dst, m);
                dst += m;
            }
            c->flush(c->vir, c->size);
        }
        return axera_io(c.get(), c->vir, (uint32_t) c->size);
    };

    // the conv front-end, the per-chunk positions: the layers' input rows
    axera_audio_frontend(ctx, a, cgraph, x0_node + 1);
    {
        axera_timer tm("audio prepare (cpu)");
        auto *      x = (ggml_bf16_t *) a.X[0]->vir;
        ggml_fp32_to_bf16_row((const float *) x0->data, x, (int64_t) n * H);
        std::fill(x + (size_t) n * H, x + (size_t) N * H, ggml_fp32_to_bf16(0.0f));
        a.X[0]->flush(a.X[0]->vir, a.X[0]->size);
        *(int32_t *) a.NBLK->vir = (n + a.BC - 1) / a.BC;
        a.NBLK->flush(a.NBLK->vir, a.NBLK->size);
        auto * km = (ggml_bf16_t *) a.KMASK->vir;
        for (int b = 0; b < N / a.BC; ++b) {
            for (int r = 0; r < a.QB; ++r) {
                for (int c = 0; c < a.BC; ++c) {
                    km[((size_t) b * a.QB + r) * a.BC + c] = ggml_fp32_to_bf16(b * a.BC + c < n ? 0.0f : -3e38f);
                }
            }
        }
        a.KMASK->flush(a.KMASK->vir, a.KMASK->size);
    }
    // a kernel call with its IO by name
    using io_map = std::map<std::string, AX_ENGINE_IO_BUFFER_T>;
    auto cm_io = [](const axera_cmm & c) { return axera_io(c.get(), c->vir, (uint32_t) c->size); };
    auto run   = [&](ggml_axera_kernel * k, const io_map & io, const char * what) {
        std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), outs(k->info->nOutputSize);
        for (int o = 0; o < 2; ++o) {
            auto & bufs = o ? outs : in;
            for (size_t i = 0; i < bufs.size(); ++i) {
                const AX_ENGINE_IOMETA_T & m  = o ? k->info->pOutputs[i] : k->info->pInputs[i];
                auto                       it = io.find(m.pName);
                if (it == io.end() || it->second.nSize < m.nSize) {
                    GGML_ABORT("qwen3 audio: %s: IO %s of %u bytes, %u given\n", what, m.pName, m.nSize,
                               it == io.end() ? 0u : it->second.nSize);
                }
                bufs[i]       = it->second;
                bufs[i].nSize = m.nSize;
            }
        }
        axera_timer tm(what);
        const int   ret = axera_run_sync(k, in, outs);
        if (ret != 0) {
            GGML_ABORT("qwen3 audio: RunSync failed (0x%x) in %s\n", ret, what);
        }
    };
    int cur = 0;
    for (int il = 0; il < L; ++il) {
        run(a.pre, { { "x", cm_io(a.X[cur]) }, { "ln1_w", weight(il, "ln1.weight") }, { "ln1_b", weight(il, "ln1.bias") },
                     { "w_qkv", weight(il, "attn_qkv.weight") }, { "b_qkv", weight(il, "attn_qkv.bias") },
                     { "cos", cm_io(a.COS) }, { "sin", cm_io(a.SIN) }, { "q_out", cm_io(a.Q) }, { "k_out", cm_io(a.K) },
                     { "v_out", cm_io(a.V) } }, "audio pre");
        run(a.attn, { { "qs", cm_io(a.Q) }, { "kc", cm_io(a.K) }, { "vc", cm_io(a.V) }, { "nblk", cm_io(a.NBLK) },
                      { "kmask", cm_io(a.KMASK) }, { "out", cm_io(a.ATT) } }, "audio attn");
        run(a.post, { { "x", cm_io(a.X[cur]) }, { "attn", cm_io(a.ATT) }, { "w_o", weight(il, "attn_out.weight") },
                      { "b_o", weight(il, "attn_out.bias") }, { "ln2_w", weight(il, "ln2.weight") },
                      { "ln2_b", weight(il, "ln2.bias") }, { "w_up", weight(il, "ffn_up.weight") },
                      { "b_up", weight(il, "ffn_up.bias") }, { "w_down", weight(il, "ffn_down.weight") },
                      { "b_down", weight(il, "ffn_down.bias") }, { "out", cm_io(a.X[1 - cur]) } }, "audio post");
        cur = 1 - cur;
    }
    // the last layer's rows into the graph, then the post norm and the projector
    a.X[cur]->invalidate(a.X[cur]->vir, a.X[cur]->size);
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) a.X[cur]->vir, (float *) xl->data, (int64_t) n * H);
    axera_cpu_compute(ctx, cgraph, post_norm, cgraph->n_nodes, "audio projector (cpu)");
    ggml_tensor * out = cgraph->nodes[cgraph->n_nodes - 1];
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, ggml_nbytes(out));
    }
    return GGML_STATUS_SUCCESS;
}
