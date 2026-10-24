// Axera NPU backend: the Qwen3-ASR audio encoder

#include "ggml-axera-impl.h"

#include <mutex>
#include <set>

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
    // the conv front-end on the NPU (axera_audio_frontend_npu): its buffers by name, the model its weights are of
    std::map<std::string, axera_cmm> fe_io;
    const void *                     fe_bound = nullptr;  // data of a.conv_out.weight
};

// a tensor of any type as bf16 into dst, by rows on the host workers
static void axera_audio_to_bf16(const ggml_tensor * p, ggml_bf16_t * dst) {
    const int64_t K = p->ne[0], rows = ggml_nelements(p) / K;
    const auto *  tr = ggml_get_type_traits(p->type);
    axera_host_pool().run(rows, [&](int64_t r) {
        std::vector<float> f(K);
        const char *       src = (const char *) p->data + r * ggml_row_size(p->type, K);
        if (p->type == GGML_TYPE_F32) {
            memcpy(f.data(), src, K * sizeof(float));
        } else {
            tr->to_float(src, f.data(), K);
        }
        ggml_fp32_to_bf16_row(f.data(), dst + r * K, K);
    });
}

// the bf16 copies of the encoder layers' matrices made while the model loads (axera_audio_on_set), by their data
static std::mutex                                      g_audio_pre_mu;
static std::map<const void *, axera_cmm>               g_audio_pre;

// a tensor the loader set whole, or a source of an op the scheduler asks about (a mapped model's weights are never set:
// mtmd's warmup reserves the encoder's graph while loading): an audio encoder layer's matrix (a.blk.<i>.{attn_q,attn_k,attn_v,attn_out,ffn_up,
// ffn_down}.weight) converted to bf16 now, so that the first request does not wait for it
void axera_audio_on_set(const ggml_tensor * t) {
    static const char * names[] = { ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_out.weight",
                                    ".ffn_up.weight", ".ffn_down.weight" };
    if (strncmp(t->name, "a.blk.", 6) != 0 || ggml_n_dims(t) != 2 || !ggml_is_contiguous(t) ||
        (t->type != GGML_TYPE_F32 && !ggml_get_type_traits(t->type)->to_float)) {
        return;
    }
    const size_t n = strlen(t->name);
    bool         m = false;
    for (const char * s : names) {
        const size_t k = strlen(s);
        m = m || (n > k && strcmp(t->name + n - k, s) == 0);
    }
    if (!m) {
        return;
    }
    {
        static std::set<const void *> seen;  // (a weight converted once, whichever hook sees it first)
        std::lock_guard<std::mutex>   lock(g_audio_pre_mu);
        if (!seen.insert(t->data).second) {
            return;
        }
    }
    static std::once_flag kernels;  // the encoder's kernels loaded with the model too (the first one seen)
    std::call_once(kernels, [] {
        axera_timer tk("audio kernels at load (host)");
        std::vector<std::string> names;
        for (const auto & f : axera_kernels().files) {
            const std::string & n = f.first;
            if (n.rfind("qwen3a_fe", 0) == 0 || n.rfind("qwen3a_layer", 0) == 0 ||
                (n.rfind("mm_bf16_m", 0) == 0 && n.size() > 6 && n.compare(n.size() - 6, 6, "_k7680") == 0)) {
                names.push_back(n);
            }
        }
        for (const std::string & n : names) {
            axera_kernels().get(n);
        }
    });
    axera_timer tm("audio weights at load (host)");
    axera_cmm   c = axera_cmm_alloc(ggml_nelements(t) * sizeof(ggml_bf16_t));
    axera_audio_to_bf16(t, (ggml_bf16_t *) c->vir);
    c->flush(c->vir, c->size);
    std::lock_guard<std::mutex> lock(g_audio_pre_mu);
    g_audio_pre[t->data] = std::move(c);
}

// the load-time bf16 copy of p, taken out of the registry (nullptr if there is none)
static axera_cmm axera_audio_take_pre(const ggml_tensor * p) {
    std::lock_guard<std::mutex> lock(g_audio_pre_mu);
    auto                        it = g_audio_pre.find(p->data);
    if (it == g_audio_pre.end() || it->second->size != (size_t) ggml_nelements(p) * sizeof(ggml_bf16_t)) {
        return axera_cmm();
    }
    axera_cmm c = std::move(it->second);
    g_audio_pre.erase(it);
    return c;
}

// a graph is the audio encoder if it reads the conv front-end's output projection and the transformer layers (the
// Qwen3-TTS speaker encoder has a conv_out too)
bool axera_audio_graph(const struct ggml_cgraph * cgraph) {
    bool conv_out = false, layers = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && cgraph->nodes[i]->src[s]; ++s) {
            const std::string n = axera_base_name(cgraph->nodes[i]->src[s]->name);
            conv_out            = conv_out || n == "a.conv_out.weight";
            layers              = layers || n == "a.blk.0.attn_q.weight";
        }
    }
    return conv_out && layers;
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

// the conv front-end on the NPU when its kernels exist for the call's chunks (n of 100 frames, padded to 1 / 2 / 4 / 8):
// each 3 x 3 stride-2 convolution as one qwen3a_fe_o480_k<K>_r<R> call (q3a_fe.py: the product, bias and GELU, bf16
// out), its im2col rows written by the host as bf16 from the previous output; then conv_out on mm_bf16_m<M>_n<H>_k7680
// and its bias and the per-chunk positions on the host, into the layers' input x0. False (nothing written) without
// those kernels: the graph's front-end runs as before
static bool axera_audio_frontend_npu(axera_audio & a, const std::map<std::string, ggml_tensor *> & t, ggml_tensor * x0) {
    auto find = [&](const std::string & n) -> ggml_tensor * {
        auto it = t.find(n);
        return it == t.end() ? nullptr : it->second;
    };
    const ggml_tensor * inp = find("inp_raw");
    const ggml_tensor * w1 = find("a.conv2d.1.weight"), *w2 = find("a.conv2d.2.weight"), *w3 = find("a.conv2d.3.weight");
    const ggml_tensor * b1 = find("a.conv2d.1.bias"), *b2 = find("a.conv2d.2.bias"), *b3 = find("a.conv2d.3.bias");
    const ggml_tensor * wo = find("a.conv_out.weight"), *bo = find("a.conv_out.bias"), *pe = find("a.position_embd.weight");
    if (!inp || !w1 || !w2 || !w3 || !b1 || !b2 || !b3 || !wo || !pe || inp->type != GGML_TYPE_F32 || !ggml_is_contiguous(inp)) {
        return false;
    }
    const int64_t NF = inp->ne[0], MEL = inp->ne[1], CH = 100, H = x0->ne[0];
    const int64_t C = w1->ne[3];  // 480
    if (NF % CH != 0 || MEL != 128 || w1->ne[2] != 1 || w2->ne[2] != C || w3->ne[2] != C || wo->ne[0] != 16 * C ||
        wo->ne[1] != H || x0->ne[1] != NF / CH * 13 || b1->type != GGML_TYPE_F32 || b2->type != GGML_TYPE_F32 ||
        b3->type != GGML_TYPE_F32 || (bo && bo->type != GGML_TYPE_F32) || pe->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t nch = NF / CH;
    int64_t       n   = 0;
    for (int64_t c : { 1, 2, 4, 8 }) {
        n = n == 0 && c >= nch ? c : n;
    }
    const int64_t     R1 = 3200 * n, R2 = 800 * n, R3 = 208 * n, M = 13 * n, K1 = 16, K2 = 9 * C;
    const std::string k1n = "qwen3a_fe_o" + std::to_string(C) + "_k16_r" + std::to_string(R1);
    const std::string k2n = "qwen3a_fe_o" + std::to_string(C) + "_k" + std::to_string(K2) + "_r" + std::to_string(R2);
    const std::string k3n = "qwen3a_fe_o" + std::to_string(C) + "_k" + std::to_string(K2) + "_r" + std::to_string(R3);
    const std::string kon = "mm_bf16_m" + std::to_string(M) + "_n" + std::to_string(H) + "_k" + std::to_string(16 * C);
    // conv2d.2 with its im2col on chip (qwen3a_fe2_o<C>_n<n>) where it exists: conv2d.1's rows then ordered (chunk, h,
    // column parity, w / 2), its taps [OC][kh][kw][IC]
    const std::string k2i  = "qwen3a_fe2_o" + std::to_string(C) + "_n" + std::to_string(n);
    const bool        chip = n > 0 && std::ifstream(axera_kernels().path(k2i)).good();
    for (const std::string & k : { k1n, chip ? k2i : k2n, k3n, kon }) {
        if (n == 0 || !std::ifstream(axera_kernels().path(k)).good()) {
            return false;
        }
    }
    // the kernels' weights as bf16 rows [OC][IC * 9] (conv2d.1's 9 taps padded to 16), once per model
    auto conv_w = [&](const ggml_tensor * w, int64_t K) -> axera_cmm & {
        axera_cmm & c = a.w["fe:" + axera_base_name(w->name)];
        if (!c || a.fe_bound != wo->data) {
            const int64_t OC = w->ne[3], KK = w->ne[0] * w->ne[1] * w->ne[2];
            c = axera_cmm_alloc((size_t) OC * K * sizeof(ggml_bf16_t));
            auto * d = (ggml_bf16_t *) c->vir;
            axera_host_pool().run(OC, [&](int64_t o) {
                std::vector<float> f(KK);
                const char *       src = (const char *) w->data + o * w->nb[3];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(f.data(), src, KK * sizeof(float));
                } else {
                    ggml_get_type_traits(w->type)->to_float(src, f.data(), KK);
                }
                f.resize(K, 0.0f);
                ggml_fp32_to_bf16_row(f.data(), d + o * K, K);
            });
            c->flush(c->vir, c->size);
        }
        return c;
    };
    axera_cmm & W1 = conv_w(w1, K1);
    axera_cmm & W2 = chip ? a.w["fe:taps:a.conv2d.2.weight"] : conv_w(w2, K2);
    if (chip && (!W2 || a.fe_bound != wo->data)) {  // [OC][IC][3][3] -> [OC][3][3][IC]
        const int64_t IC = w2->ne[2];
        W2 = axera_cmm_alloc((size_t) C * K2 * sizeof(ggml_bf16_t));
        auto * d = (ggml_bf16_t *) W2->vir;
        axera_host_pool().run(C, [&](int64_t o) {
            std::vector<float> f(K2), g(K2);
            const char *       src = (const char *) w2->data + o * w2->nb[3];
            if (w2->type == GGML_TYPE_F32) {
                memcpy(f.data(), src, K2 * sizeof(float));
            } else {
                ggml_get_type_traits(w2->type)->to_float(src, f.data(), K2);
            }
            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t k = 0; k < 9; ++k) {
                    g[k * IC + ic] = f[ic * 9 + k];
                }
            }
            ggml_fp32_to_bf16_row(g.data(), d + o * K2, K2);
        });
        W2->flush(W2->vir, W2->size);
    }
    axera_cmm & W3 = conv_w(w3, K2);
    axera_cmm & WO = a.w["fe:a.conv_out.weight"];
    if (!WO || a.fe_bound != wo->data) {
        WO = axera_cmm_alloc((size_t) H * 16 * C * sizeof(ggml_bf16_t));
        const int64_t KO = wo->ne[0];
        auto *        d  = (ggml_bf16_t *) WO->vir;
        axera_host_pool().run(H, [&](int64_t r) {
            std::vector<float> f(KO);
            const char *       src = (const char *) wo->data + r * wo->nb[1];
            if (wo->type == GGML_TYPE_F32) {
                memcpy(f.data(), src, KO * sizeof(float));
            } else {
                ggml_get_type_traits(wo->type)->to_float(src, f.data(), KO);
            }
            ggml_fp32_to_bf16_row(f.data(), d + r * KO, KO);
        });
        WO->flush(WO->vir, WO->size);
    }
    a.fe_bound = wo->data;
    auto buf = [&](const std::string & k, size_t bytes) -> axera_cmm & {
        axera_cmm & b = a.fe_io[k];
        if (!b || b->size < bytes) {
            b = axera_cmm_alloc(bytes);
        }
        return b;
    };
    axera_cmm & X1 = buf("x1", (size_t) R1 * K1 * 2), &O1 = buf("o1", (size_t) C * R1 * 2);
    axera_cmm & X2 = buf("x2", chip ? 64 : (size_t) R2 * K2 * 2), &O2 = buf("o2", (size_t) C * R2 * 2);
    axera_cmm & X3 = buf("x3", (size_t) R3 * K2 * 2), &O3 = buf("o3", (size_t) C * R3 * 2);
    axera_cmm & XO = buf("xo", (size_t) M * 16 * C * 2), &OO = buf("oo", (size_t) M * H * 4);
    auto io = [](const axera_cmm & c, size_t bytes) { return axera_io(c.get(), c->vir, (uint32_t) bytes); };
    auto bias_io = [&](const ggml_tensor * b) -> AX_ENGINE_IO_BUFFER_T {
        axera_cmm & c = a.w["fe:" + axera_base_name(b->name)];
        if (!c) {
            c = axera_cmm_alloc(ggml_nbytes(b));
            memcpy(c->vir, b->data, ggml_nbytes(b));
            c->flush(c->vir, c->size);
        }
        return io(c, c->size);
    };
    auto run = [&](const std::string & kn, std::vector<std::pair<const char *, AX_ENGINE_IO_BUFFER_T>> ins,
                   const char * out_name, AX_ENGINE_IO_BUFFER_T out, const char * what) {
        ggml_axera_kernel * k = axera_kernels().get(kn);
        std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), outs(k->info->nOutputSize);
        GGML_ASSERT(ins.size() == in.size() && outs.size() == 1);
        for (auto & p : ins) {
            const int i = k->input(p.first);
            GGML_ASSERT(i >= 0 && p.second.nSize >= k->info->pInputs[i].nSize);
            in[i]       = p.second;
            in[i].nSize = k->info->pInputs[i].nSize;
        }
        const int o = k->output(out_name);
        GGML_ASSERT(o >= 0 && out.nSize >= k->info->pOutputs[o].nSize);
        outs[o]       = out;
        outs[o].nSize = k->info->pOutputs[o].nSize;
        axera_timer tm(what);
        const int   ret = axera_run_sync(k, in, outs);
        if (ret != 0) {
            GGML_ABORT("qwen3 audio: RunSync failed (0x%x) in %s\n", ret, kn.c_str());
        }
    };
    // a 3 x 3 stride-2 pad-1 im2col of a bf16 map [IC][n][IH][IW] (rows (n, oh, ow), columns (ic, kh, kw)) into dst
    auto im2col = [&](const ggml_bf16_t * src, int64_t IC, int64_t IH, int64_t IW, int64_t OH, int64_t OW, int64_t K,
                      ggml_bf16_t * dst, const char * what) {
        axera_timer tm(what);
        const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f);
        axera_host_pool().run(n * OH, [&](int64_t job) {
            const int64_t c = job / OH, oh = job % OH;
            ggml_bf16_t * row0 = dst + (c * OH + oh) * OW * K;
            if (c >= nch) {  // a padding chunk
                std::fill(row0, row0 + OW * K, zero);
                return;
            }
            for (int64_t ow = 0; ow < OW; ++ow) {
                ggml_bf16_t * d    = row0 + ow * K;
                const int64_t w0   = 2 * ow - 1;
                const bool    edge = w0 < 0 || w0 + 2 >= IW;
                for (int64_t ic = 0; ic < IC; ++ic) {
                    const ggml_bf16_t * s = src + (ic * n + c) * IH * IW;
                    for (int64_t kh = 0; kh < 3; ++kh, d += 3) {
                        const int64_t h = 2 * oh + kh - 1;
                        if (h < 0 || h >= IH) {
                            d[0] = d[1] = d[2] = zero;
                            continue;
                        }
                        const ggml_bf16_t * r = s + h * IW;
                        if (!edge) {
                            d[0] = r[w0];
                            d[1] = r[w0 + 1];
                            d[2] = r[w0 + 2];
                        } else {
                            for (int64_t kw = 0; kw < 3; ++kw) {
                                const int64_t w = w0 + kw;
                                d[kw]           = w >= 0 && w < IW ? r[w] : zero;
                            }
                        }
                    }
                }
                std::fill(d, row0 + (ow + 1) * K, zero);
            }
        });
    };
    // conv2d.1: the mel rows of each chunk, one input channel
    {
        axera_timer tm("audio fe im2col 1 (host)");
        const float * mel  = (const float *) inp->data;
        auto *        d    = (ggml_bf16_t *) X1->vir;
        axera_host_pool().run(n * 64, [&](int64_t job) {
            const int64_t c = job / 64, oh = job % 64;
            for (int64_t ow = 0; ow < 50; ++ow) {
                float v[16] = {};
                for (int64_t kh = 0; kh < 3; ++kh) {
                    for (int64_t kw = 0; kw < 3; ++kw) {
                        const int64_t h = 2 * oh + kh - 1, f = 2 * ow + kw - 1;
                        v[kh * 3 + kw] = c < nch && h >= 0 && h < MEL && f >= 0 && f < CH ? mel[h * NF + c * CH + f] : 0.0f;
                    }
                }
                const int64_t col = chip ? (ow % 2) * 25 + ow / 2 : ow;  // (the parity-split columns for fe2)
                ggml_fp32_to_bf16_row(v, d + ((c * 64 + oh) * 50 + col) * K1, K1);
            }
        });
        X1->flush(X1->vir, (size_t) R1 * K1 * 2);
    }
    run(k1n, { { "x", io(X1, (size_t) R1 * K1 * 2) }, { "w", io(W1, W1->size) }, { "b", bias_io(b1) } }, "out",
        io(O1, (size_t) C * R1 * 2), "audio fe conv 1 (npu)");
    if (chip) {  // conv2d.1's output read by conv2d.2's kernel directly
        run(k2i, { { "x", io(O1, (size_t) C * R1 * 2) }, { "w", io(W2, W2->size) }, { "b", bias_io(b2) } }, "out",
            io(O2, (size_t) C * R2 * 2), "audio fe conv 2 (npu)");
    } else {
        O1->invalidate(O1->vir, (size_t) C * R1 * 2);
        im2col((const ggml_bf16_t *) O1->vir, C, 64, 50, 32, 25, K2, (ggml_bf16_t *) X2->vir, "audio fe im2col 2 (host)");
        X2->flush(X2->vir, (size_t) R2 * K2 * 2);
        run(k2n, { { "x", io(X2, (size_t) R2 * K2 * 2) }, { "w", io(W2, W2->size) }, { "b", bias_io(b2) } }, "out",
            io(O2, (size_t) C * R2 * 2), "audio fe conv 2 (npu)");
    }
    O2->invalidate(O2->vir, (size_t) C * R2 * 2);
    im2col((const ggml_bf16_t *) O2->vir, C, 32, 25, 16, 13, K2, (ggml_bf16_t *) X3->vir, "audio fe im2col 3 (host)");
    X3->flush(X3->vir, (size_t) R3 * K2 * 2);
    run(k3n, { { "x", io(X3, (size_t) R3 * K2 * 2) }, { "w", io(W3, W3->size) }, { "b", bias_io(b3) } }, "out",
        io(O3, (size_t) C * R3 * 2), "audio fe conv 3 (npu)");
    O3->invalidate(O3->vir, (size_t) C * R3 * 2);
    // conv_out's rows (n, ow), columns (oc, oh): the [OW, OH, OC, n] map permuted to [OH, OC, OW, n] as the graph does
    {
        axera_timer   tm("audio fe conv_out rows (host)");
        const auto *  s = (const ggml_bf16_t *) O3->vir;
        auto *        d = (ggml_bf16_t *) XO->vir;
        axera_host_pool().run(M, [&](int64_t r) {
            const int64_t c = r / 13, ow = r % 13;
            for (int64_t oc = 0; oc < C; ++oc) {
                for (int64_t oh = 0; oh < 16; ++oh) {
                    d[r * 16 * C + oc * 16 + oh] = s[oc * R3 + (c * 16 + oh) * 13 + ow];
                }
            }
        });
        XO->flush(XO->vir, (size_t) M * 16 * C * 2);
    }
    run(kon, { { "x", io(XO, (size_t) M * 16 * C * 2) }, { "w", io(WO, WO->size) } }, "out", io(OO, (size_t) M * H * 4),
        "audio fe conv_out (npu)");
    OO->invalidate(OO->vir, (size_t) M * H * 4);
    {
        axera_timer   tm("audio fe positions (host)");
        const auto *  y = (const float *) OO->vir;
        auto *        x = (float *) x0->data;
        const float * bb = bo ? (const float *) bo->data : nullptr;
        for (int64_t r = 0; r < nch * 13; ++r) {
            const float * p = (const float *) ((const char *) pe->data + (r % 13) * pe->nb[1]);
            for (int64_t i = 0; i < H; ++i) {
                x[r * H + i] = (y[r * H + i] + (bb ? bb[i] : 0.0f)) + p[i];
            }
        }
    }
    return true;
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

    // the four-core layer kernel (q3a_layer.py, qwen3a_layer<NF>_h<H>) of the smallest row count that holds the n rows
    int NF = 0;
    for (int c : { 64, 128 }) {
        if (NF == 0 && c >= n && std::ifstream(axera_kernels().path("qwen3a_layer" + std::to_string(c) + "_h" + std::to_string(H))).good()) {
            NF = c;
        }
    }
    // else the smallest pre / attn / post kernel row count that holds them; none: the whole graph on the CPU
    int N = 0;
    for (int c : { 256, 512, 1024, 2048, 4096 }) {
        if (N == 0 && c >= n && std::ifstream(axera_kernels().path("qwen3a_attn" + std::to_string(c) + "_h" + std::to_string(H))).good()) {
            N = c;
        }
    }
    if (N == 0 && NF == 0) {
        GGML_LOG_WARN("%s: no qwen3a_*<N>_h%d kernels for %d rows: the audio encoder on the CPU\n", __func__, H, n);
        axera_cpu_compute(ctx, cgraph, 0, cgraph->n_nodes, "audio (cpu)");
        return GGML_STATUS_SUCCESS;
    }
    auto name = [&](const char * k) { return std::string("qwen3a_") + k + std::to_string(N) + "_h" + std::to_string(H); };
    if (N != 0 && (a.N != N || a.H != H)) {
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
    GGML_ASSERT(N == 0 || axera_input_size(a.pre, "x") == (size_t) N * H * sizeof(ggml_bf16_t));

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
            if (parts.size() == 1) {
                c = axera_audio_take_pre(parts[0]);  // (converted while loading)
            }
            if (!c) {
                c          = axera_cmm_alloc(ne * sizeof(ggml_bf16_t));
                auto * dst = (ggml_bf16_t *) c->vir;
                for (const ggml_tensor * p : parts) {
                    axera_cmm pre = axera_audio_take_pre(p);
                    if (pre) {
                        memcpy(dst, pre->vir, pre->size);
                    } else {
                        axera_audio_to_bf16(p, dst);
                    }
                    dst += ggml_nelements(p);
                }
            }
            c->flush(c->vir, c->size);
        }
        return axera_io(c.get(), c->vir, (uint32_t) c->size);
    };

    // the conv front-end, the per-chunk positions: the layers' input rows
    if (!axera_audio_frontend_npu(a, t, x0)) {
        axera_audio_frontend(ctx, a, cgraph, x0_node + 1);
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
    if (NF != 0) {  // the layers on the four-core layer kernel
        ggml_axera_kernel * k  = axera_kernels().get("qwen3a_layer" + std::to_string(NF) + "_h" + std::to_string(H));
        const size_t        xb = (size_t) NF * H * sizeof(ggml_bf16_t);
        for (int i = 0; i < 3; ++i) {
            axera_cmm & b = a.fe_io["lx" + std::to_string(i)];
            if (!b || b->size < xb) {
                b = axera_cmm_alloc(xb);
            }
        }
        axera_cmm & KM = a.fe_io["lmask"];
        if (!KM || KM->size < (size_t) NF * sizeof(ggml_bf16_t)) {
            KM = axera_cmm_alloc((size_t) NF * sizeof(ggml_bf16_t));
        }
        axera_cmm * LX[2] = { &a.fe_io["lx0"], &a.fe_io["lx1"] };
        {
            axera_timer tm("audio prepare (cpu)");
            auto *      x = (ggml_bf16_t *) (*LX[0])->vir;
            ggml_fp32_to_bf16_row((const float *) x0->data, x, (int64_t) n * H);
            std::fill(x + (size_t) n * H, x + (size_t) NF * H, ggml_fp32_to_bf16(0.0f));
            (*LX[0])->flush((*LX[0])->vir, xb);
            auto * km = (ggml_bf16_t *) KM->vir;
            for (int c = 0; c < NF; ++c) {
                km[c] = ggml_fp32_to_bf16(c < n ? 0.0f : -3e38f);
            }
            KM->flush(KM->vir, KM->size);
        }
        // GGML_AXERA_AUDIO_TWICE=1: every layer call run twice, the second into a spare buffer poisoned before, the two
        // outputs compared byte by byte (a race between the cores shows as a difference)
        static const bool twice = [] {
            const char * e = std::getenv("GGML_AXERA_AUDIO_TWICE");
            return e && strcmp(e, "1") == 0;
        }();
        static int n_twice = 0, n_diff = 0;
        int        cur     = 0;
        for (int il = 0; il < L; ++il) {
            io_map m = { { "x", cm_io(*LX[cur]) }, { "ln1_w", weight(il, "ln1.weight") }, { "ln1_b", weight(il, "ln1.bias") },
                         { "w_qkv", weight(il, "attn_qkv.weight") }, { "b_qkv", weight(il, "attn_qkv.bias") },
                         { "w_o", weight(il, "attn_out.weight") }, { "b_o", weight(il, "attn_out.bias") },
                         { "ln2_w", weight(il, "ln2.weight") }, { "ln2_b", weight(il, "ln2.bias") },
                         { "w_up", weight(il, "ffn_up.weight") }, { "b_up", weight(il, "ffn_up.bias") },
                         { "w_down", weight(il, "ffn_down.weight") }, { "b_down", weight(il, "ffn_down.bias") },
                         { "kmask", cm_io(KM) }, { "out", cm_io(*LX[1 - cur]) } };
            run(k, m, "audio layer (npu)");
            if (twice) {
                axera_cmm & S = a.fe_io["lx2"];
                memset(S->vir, 0xff, xb);  // (bf16 NaN)
                S->flush(S->vir, xb);
                m["out"] = cm_io(S);
                run(k, m, "audio layer, again (npu)");
                S->invalidate(S->vir, xb);
                (*LX[1 - cur])->invalidate((*LX[1 - cur])->vir, xb);
                ++n_twice;
                if (memcmp(S->vir, (*LX[1 - cur])->vir, (size_t) n * H * sizeof(ggml_bf16_t)) != 0) {
                    GGML_LOG_WARN("axera audio: layer %d gave different outputs on two runs (%d of %d calls)\n", il,
                                  ++n_diff, n_twice);
                }
            }
            cur = 1 - cur;
        }
        if (twice) {
            GGML_LOG_INFO("axera audio: %d layer calls run twice, %d differed\n", n_twice, n_diff);
        }
        (*LX[cur])->invalidate((*LX[cur])->vir, xb);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) (*LX[cur])->vir, (float *) xl->data, (int64_t) n * H);
        axera_cpu_compute(ctx, cgraph, post_norm, cgraph->n_nodes, "audio projector (cpu)");
        ggml_tensor * out = cgraph->nodes[cgraph->n_nodes - 1];
        if (ggml_backend_buffer_is_axera(out->buffer)) {
            ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, ggml_nbytes(out));
        }
        return GGML_STATUS_SUCCESS;
    }
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
