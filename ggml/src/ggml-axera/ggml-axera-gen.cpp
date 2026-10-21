// Axera NPU backend: graphs without a template of their own (mtmd's generation graphs: the Qwen3-TTS code predictor
// and codec decoder, its speaker encoder): run by the CPU backend from inside the backend, the matrix products that
// have an NPU kernel of their shape on the NPU

#include "ggml-axera-q35.h"  // axera_q35_context: the fused layer kernel's LMAX / BC

#include <set>
#include <sys/stat.h>

// bf16 copies of the weights the NPU reads (any GGUF type), by data and size (views of a 3-D weight are separate)
struct axera_gen {
    std::map<std::pair<const void *, size_t>, std::unique_ptr<ggml_backend_axera_buffer_context>> w;
    std::map<std::string, bool> have;   // kernel name -> file exists
    std::set<std::string>       names;  // profiling labels (axera_timer keeps the pointer)
    axera_cmm                   tmp[2];  // per-call bf16 copies of non-weight operands (src0, src1)
    // GEMVs of one input by several weights (q / k / v, gate / up) as one call: the weights joined by their data, the
    // output, and the results of the later nodes of a group until the graph reaches them
    std::map<std::vector<const void *>, axera_cmm>      wg;
    axera_cmm                                          gout;
    std::map<const ggml_tensor *, std::vector<float>>  pending;
    // 1-D IM2COL nodes not run on the CPU, by the NPU GEMM that reads them (which writes their rows as bf16 itself)
    std::map<const ggml_tensor *, const ggml_tensor *> im2col;
    // the Qwen3-TTS code predictor's layers on the template's fused decode layer kernel of its shapes (Qwen3-0.6B's
    // qwen3_attn[_mx8]_kvc_c<LMAX> as qwen3tts_cp_*): IO per layer, its 16 positions' KV resident in the kernel's kc / vc
    struct layer {
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        std::vector<axera_cmm>             own;
    };
    ggml_axera_kernel * cp      = nullptr;
    const void *        cp_bound = nullptr;  // data of a.gen.code.blk.0.ln1.weight
    int                 cp_LMAX = 0, cp_BC = 0, cp_H = 0, cp_ROT = 0;
    std::vector<layer>  cp_layers;
    axera_cmm           X, Y, POS, NBLK, MASK, COS, SIN;
    bool                cp_off = false;  // the kernel is missing or a graph did not fit: the layers stay as they are
    std::vector<uintptr_t>                      key;       // the last graph's nodes, and its analysis
    std::shared_ptr<struct axera_gen_analysis> analysis;
};

// ggml's MUL_MAT out [M][N] = src1 [M][K] . src0 [N][K]^T (out ne0 = N): src0 the weight or, for a convolution, the
// im2col rows
static bool axera_gen_mm_shape(const ggml_tensor * n, int64_t * M, int64_t * N, int64_t * K) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    auto convertible = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 || ggml_get_type_traits(t->type)->to_float;
    };
    if (n->op != GGML_OP_MUL_MAT || n->type != GGML_TYPE_F32 || !convertible(a) || !convertible(b) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(n) || a->ne[2] * a->ne[3] != 1 ||
        b->ne[2] * b->ne[3] != 1 || !ggml_backend_buffer_is_axera(n->buffer)) {
        return false;
    }
    *M = b->ne[1];
    *N = a->ne[1];
    *K = a->ne[0];
    return true;
}

// a model weight (or a view of one): a tensor in a weights buffer, or a leaf that is not a graph input
static bool axera_gen_is_weight(const ggml_tensor * t) {
    const ggml_tensor * s = t->view_src ? t->view_src : t;
    return (s->buffer && s->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
           (s->op == GGML_OP_NONE && !(s->flags & GGML_TENSOR_FLAG_INPUT) && s->data);
}

// the weight a tensor is made of by data movement only (reshape / view / transpose / permute / cont: e.g. a transposed
// convolution's kernel made contiguous on every run), nullptr if it reads anything else
static const ggml_tensor * axera_gen_const_root(const ggml_tensor * t) {
    for (int depth = 0; t && depth < 8; ++depth) {
        if (axera_gen_is_weight(t)) {
            return t->view_src ? t->view_src : t;
        }
        if (t->op != GGML_OP_CONT && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && t->op != GGML_OP_TRANSPOSE &&
            t->op != GGML_OP_PERMUTE) {
            return nullptr;
        }
        t = t->src[0];
    }
    return nullptr;
}

static bool axera_gen_have(axera_gen & g, const std::string & name) {
    auto it = g.have.find(name);
    if (it == g.have.end()) {
        struct stat st;
        it = g.have.emplace(name, stat(axera_kernels().path(name).c_str(), &st) == 0).first;
    }
    return it->second;
}

// rows [0, rows) of a 2-D tensor of any type as bf16 into dst, on the host workers
static void axera_gen_to_bf16(const ggml_tensor * t, int64_t rows, ggml_bf16_t * dst) {
    const int64_t K = t->ne[0];
    const auto *  tr = ggml_get_type_traits(t->type);
    axera_host_pool().run(rows, [&](int64_t r) {
        const char * src = (const char *) t->data + r * t->nb[1];
        if (t->type == GGML_TYPE_BF16) {
            memcpy(dst + r * K, src, K * sizeof(ggml_bf16_t));
            return;
        }
        std::vector<float>       f(K);  // (dst may be uncached: the row is written out whole)
        std::vector<ggml_bf16_t> h(K);
        if (t->type == GGML_TYPE_F32) {
            ggml_fp32_to_bf16_row((const float *) src, h.data(), K);
        } else {
            tr->to_float(src, f.data(), K);
            ggml_fp32_to_bf16_row(f.data(), h.data(), K);
        }
        memcpy(dst + r * K, h.data(), K * sizeof(ggml_bf16_t));
    });
}

// the rows of a CONT node read through the strides of its source view instead (a transposed weight: the CONT itself
// need not run), f32 / f16 / bf16 sources; false for any other
static bool axera_gen_cont_bf16(const ggml_tensor * a, int64_t rows, ggml_bf16_t * dst) {
    const ggml_tensor * v = a->src[0];
    if (a->op != GGML_OP_CONT || !v->data || ggml_nelements(v) != ggml_nelements(a) || v->ne[2] * v->ne[3] != 1 ||
        v->ne[0] != a->ne[0] || (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16)) {
        return false;
    }
    const int64_t K = a->ne[0], B = 64, nr = (rows + B - 1) / B, nc = (K + B - 1) / B;
    axera_host_pool().run(nr * nc, [&](int64_t job) {  // 64 x 64 blocks: a transposed source read a block at a time
        const int64_t r0 = job / nc * B, c0 = job % nc * B;
        for (int64_t r = r0; r < std::min(rows, r0 + B); ++r) {
            const char * row = (const char *) v->data + r * v->nb[1];
            for (int64_t c = c0; c < std::min(K, c0 + B); ++c) {
                const char * e = row + c * v->nb[0];
                dst[r * K + c] = v->type == GGML_TYPE_BF16 ? *(const ggml_bf16_t *) e :
                                 ggml_fp32_to_bf16(v->type == GGML_TYPE_F32 ? *(const float *) e :
                                                                              ggml_fp16_to_fp32(*(const ggml_fp16_t *) e));
            }
        }
    });
    return true;
}

// the bf16 [rows][K] of an operand: the tensor itself (bf16 in CMM), its cached copy (a weight), or converted per call
// into tmp
static std::pair<const ggml_backend_axera_buffer_context *, void *> axera_gen_bf16(axera_gen & g, const ggml_tensor * a,
                                                                                    int64_t rows, axera_cmm & tmp) {
    const int64_t K     = a->ne[0];
    const size_t  bytes = (size_t) rows * K * sizeof(ggml_bf16_t);
    if (a->type == GGML_TYPE_BF16 && ggml_backend_buffer_is_axera(a->buffer)) {
        return { (const ggml_backend_axera_buffer_context *) a->buffer->context, a->data };
    }
    const ggml_tensor * root = axera_gen_is_weight(a) ? nullptr : axera_gen_const_root(a);
    if (root) {  // a constant made of a weight: cached by the weight and its own shape (its data is the graph's)
        auto & c = g.w[{ (const char *) root->data + a->ne[0], ggml_nbytes(a) }];
        if (!c) {
            c = axera_cmm_alloc(bytes);
            if (!axera_gen_cont_bf16(a, rows, (ggml_bf16_t *) c->vir)) {
                axera_gen_to_bf16(a, rows, (ggml_bf16_t *) c->vir);
            }
            c->flush(c->vir, bytes);
        }
        return { c.get(), c->vir };
    }
    if (axera_gen_is_weight(a)) {
        auto & c = g.w[{ a->data, ggml_nbytes(a) }];
        if (!c) {
            c = axera_cmm_alloc(bytes);
            axera_gen_to_bf16(a, rows, (ggml_bf16_t *) c->vir);
            c->flush(c->vir, bytes);
        }
        return { c.get(), c->vir };
    }
    if (!tmp || tmp->size < bytes) {
        tmp = axera_cmm_alloc_nc(bytes);  // written by the host only: no cache to clean before the NPU reads it
    }
    axera_gen_to_bf16(a, rows, (ggml_bf16_t *) tmp->vir);
    tmp->flush(tmp->vir, bytes);
    return { tmp.get(), tmp->vir };
}

// a 1-D IM2COL (ggml's rows [OW][IC][KW] of a convolution's input) written as bf16 rows into tmp, on the host workers:
// for the NPU GEMM that reads it, instead of the CPU's f16 rows converted once more
static std::pair<const ggml_backend_axera_buffer_context *, void *> axera_gen_im2col_bf16(const ggml_tensor * ic,
                                                                                          axera_cmm & tmp) {
    const int32_t *     p  = (const int32_t *) ic->op_params;
    const int           s0 = p[0], p0 = p[2], d0 = p[4];
    const ggml_tensor * x  = ic->src[1];
    const int64_t       KW = ic->src[0]->ne[0], IC = x->ne[1], IW = x->ne[0], OW = ic->ne[1], K = IC * KW;
    const size_t        bytes = (size_t) OW * K * sizeof(ggml_bf16_t);
    if (!tmp || tmp->size < bytes) {
        tmp = axera_cmm_alloc_nc(bytes);  // written by the host only: no cache to clean before the NPU reads it
    }
    auto *        dst = (ggml_bf16_t *) tmp->vir;
    const int64_t RB  = 64;  // output rows per task, built in cached memory and copied out whole (tmp is uncached)
    axera_host_pool().run((OW + RB - 1) / RB, [&](int64_t b) {
        std::vector<float>       f(K);
        std::vector<ggml_bf16_t> h(K);
        for (int64_t ow = b * RB; ow < std::min(OW, (b + 1) * RB); ++ow) {
            for (int64_t c = 0; c < IC; ++c) {
                const float * src = (const float *) ((const char *) x->data + c * x->nb[1]);
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * s0 + kw * d0 - p0;
                    f[c * KW + kw]   = iw >= 0 && iw < IW ? src[iw] : 0.0f;
                }
            }
            ggml_fp32_to_bf16_row(f.data(), h.data(), K);
            memcpy(dst + ow * K, h.data(), K * sizeof(ggml_bf16_t));
        }
    });
    tmp->flush(tmp->vir, bytes);
    return { tmp.get(), tmp->vir };
}

// the conv1d kernel of a 1-D IM2COL's convolution (stride 1, no padding: mtmd's causal convolutions take their left
// context as input): conv1d_bf16_ic<IC>_oc<OC>_k<K>_d<D>_l<L> (x [IC, L] bf16, w [K, OC, IC] bf16 -> out [OC, L -
// (K - 1) D] f32, ggml's layout of the MUL_MAT after the IM2COL); empty if the convolution is not one
static std::string axera_gen_conv_name(const ggml_tensor * ic, int64_t OC) {
    const int32_t * p = (const int32_t *) ic->op_params;
    if (p[0] != 1 || p[2] != 0) {
        return "";
    }
    char name[96];
    snprintf(name, sizeof(name), "conv1d_bf16_ic%" PRId64 "_oc%" PRId64 "_k%" PRId64 "_d%d_l%" PRId64, ic->src[1]->ne[1], OC,
             ic->src[0]->ne[0], p[4], ic->src[1]->ne[0]);
    return name;
}

// the MUL_MAT n after a 1-D IM2COL ic as one conv1d kernel call: the input converted to bf16, the kernel's taps
// repacked once ([OC][IC][K] -> [K][OC][IC]), the output written into n
static void axera_gen_conv(axera_gen & g, ggml_tensor * n, const ggml_tensor * ic, const std::string & name) {
    const ggml_tensor * x  = ic->src[1];
    const ggml_tensor * w  = ic->src[0];
    const int64_t       KW = w->ne[0], IC = x->ne[1], L = x->ne[0], OC = n->ne[1];
    auto *              dctx = (const ggml_backend_axera_buffer_context *) n->buffer->context;
    axera_timer         tm(g.names.insert("gen " + name + " (npu)").first->c_str());
    auto &              wk = g.w[{ w->data, ggml_nbytes(w) + 1 }];  // (+1: not the GEMM's copy of the same weight)
    if (!wk) {
        GGML_ASSERT(ggml_is_contiguous(w) && w->ne[1] == IC && w->ne[2] == OC);
        wk = axera_cmm_alloc((size_t) KW * OC * IC * sizeof(ggml_bf16_t));
        auto *     d  = (ggml_bf16_t *) wk->vir;
        const auto tr = ggml_get_type_traits(w->type);
        axera_host_pool().run(OC, [&](int64_t o) {  // row o: [IC][K] of any type
            std::vector<float> f(IC * KW);
            const char *       src = (const char *) w->data + o * w->nb[2];
            if (w->type == GGML_TYPE_F32) {
                memcpy(f.data(), src, f.size() * sizeof(float));
            } else {
                tr->to_float(src, f.data(), (int64_t) f.size());
            }
            for (int64_t k = 0; k < KW; ++k) {
                for (int64_t i = 0; i < IC; ++i) {
                    d[(k * OC + o) * IC + i] = ggml_fp32_to_bf16(f[i * KW + k]);
                }
            }
        });
        wk->flush(wk->vir, wk->size);
    }
    const size_t xb = (size_t) IC * L * sizeof(ggml_bf16_t);
    if (!g.tmp[1] || g.tmp[1]->size < xb) {
        g.tmp[1] = axera_cmm_alloc_nc(xb);  // written by the host only: no cache to clean before the NPU reads it
    }
    axera_gen_to_bf16(x, IC, (ggml_bf16_t *) g.tmp[1]->vir);
    ggml_axera_kernel * k  = axera_kernels().get(name);
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    std::vector<AX_ENGINE_IO_BUFFER_T> in(2), out(1);
    in[ix]  = axera_io(g.tmp[1].get(), g.tmp[1]->vir, (uint32_t) xb);
    in[iw]  = axera_io(wk.get(), wk->vir, (uint32_t) wk->size);
    out[io] = axera_io(dctx, n->data, (uint32_t) ggml_nbytes(n));
    dctx->invalidate(n->data, ggml_nbytes(n));  // no dirty line of the output left to be written back over the NPU's
    if (axera_run_sync(k, in, out) != 0) {
        GGML_ABORT("%s: %s failed\n", __func__, name.c_str());
    }
    dctx->invalidate(n->data, ggml_nbytes(n));
}

// the GEMM reading a 1-D IM2COL of node i within the next nodes, if a GEMM or conv1d kernel takes its shape: the
// IM2COL is then left to it
static const ggml_tensor * axera_gen_im2col_user(axera_gen & g, struct ggml_cgraph * cgraph, int i) {
    const ggml_tensor * ic = cgraph->nodes[i];
    const int32_t *     p  = (const int32_t *) ic->op_params;
    if (p[6] != 0 || ic->src[1]->type != GGML_TYPE_F32 || ic->src[1]->ne[2] * ic->src[1]->ne[3] != 1) {
        return nullptr;
    }
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 8); j++) {
        const ggml_tensor * m = cgraph->nodes[j];
        int64_t             M, N, K;
        if (m->op == GGML_OP_MUL_MAT && (m->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            (m->src[0] == ic || m->src[0]->view_src == ic) && axera_gen_mm_shape(m, &M, &N, &K) && N == ic->ne[1] &&
            K == ic->ne[0]) {
            char name[96];
            snprintf(name, sizeof(name), "mm_bf16_m%" PRId64 "_n%" PRId64 "_k%" PRId64, M, N, K);
            const std::string conv = axera_gen_conv_name(ic, M);
            return axera_gen_have(g, name) || (!conv.empty() && axera_gen_have(g, conv)) ? m : nullptr;
        }
    }
    return nullptr;
}

// a CONT of a weight (axera_gen_const_root) whose only readers are offloaded MUL_MATs, which have its bf16 copy or
// make it from the CONT's source view: nothing reads the CONT's own data, it is not run
static bool axera_gen_const_skip(axera_gen & g, struct ggml_cgraph * cgraph, int i) {
    const ggml_tensor * n    = cgraph->nodes[i];
    const ggml_tensor * root = axera_gen_const_root(n);
    if (!root || (!g.w.count({ (const char *) root->data + n->ne[0], ggml_nbytes(n) }) &&
                  !(n->src[0]->data && (n->src[0]->type == GGML_TYPE_F32 || n->src[0]->type == GGML_TYPE_F16 ||
                                        n->src[0]->type == GGML_TYPE_BF16) && n->src[0]->ne[0] == n->ne[0] &&
                    n->src[0]->ne[2] * n->src[0]->ne[3] == 1))) {
        return false;  // (neither cached nor to be read through its source's strides)
    }
    int readers = 0;
    for (int j = i + 1; j < cgraph->n_nodes; j++) {
        const ggml_tensor * m = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC && m->src[s]; ++s) {
            if (m->src[s] == n || m->src[s]->view_src == n) {
                int64_t M, N, K;
                char    name[96];
                if (m->op != GGML_OP_MUL_MAT || s != 0 || m->src[0] != n || !axera_gen_mm_shape(m, &M, &N, &K)) {
                    return false;
                }
                snprintf(name, sizeof(name), "mm_bf16_m%" PRId64 "_n%" PRId64 "_k%" PRId64, M, N, K);
                if (!axera_gen_have(g, name)) {
                    return false;
                }
                ++readers;
            }
        }
    }
    return readers > 0;
}

// SnakeBeta as mtmd builds it, nodes i .. i + 4: x * alpha -> sin -> sqr -> * beta -> + x (x [T, C], alpha / beta one
// per channel), computed in one pass on the host workers after before() (the CPU nodes ahead); false if the nodes are
// not that
template <typename F> static bool axera_gen_snake(struct ggml_cgraph * cgraph, int i, F before) {
    if (i + 4 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor ** n = cgraph->nodes + i;
    const ggml_tensor * x = n[0]->src[0];
    const ggml_tensor * a = n[0]->src[1];
    const ggml_tensor * b = n[3]->src[1];
    for (int k = 0; k < 5; ++k) {
        if (!(n[k]->flags & GGML_TENSOR_FLAG_COMPUTE) || n[k]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (n[0]->op != GGML_OP_MUL || n[1]->op != GGML_OP_SIN || n[2]->op != GGML_OP_SQR || n[3]->op != GGML_OP_MUL ||
        n[4]->op != GGML_OP_ADD || n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] ||
        n[4]->src[0] != x || n[4]->src[1] != n[3] || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(n[4]) || x->ne[2] * x->ne[3] != 1 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        a->ne[0] != 1 || b->ne[0] != 1 || a->ne[1] != x->ne[1] || b->ne[1] != x->ne[1] || !ggml_is_contiguous(a) ||
        !ggml_is_contiguous(b)) {
        return false;
    }
    before();
    const int64_t T = x->ne[0], C = x->ne[1], TB = 4096;
    const int64_t nt = (T + TB - 1) / TB;
    axera_timer   tm("gen snake (host)");
    axera_host_pool().run(C * nt, [&](int64_t job) {
        const int64_t c = job / nt, t0 = job % nt * TB, t1 = std::min(T, t0 + TB);
        const float   al = ((const float *) a->data)[c], be = ((const float *) b->data)[c];
        const float * xr = (const float *) ((const char *) x->data + c * x->nb[1]);
        float *       y  = (float *) ((char *) n[4]->data + c * n[4]->nb[1]);
        for (int64_t t = t0; t < t1; ++t) {  // (vectorized: sin(v)^2 by range reduction and a polynomial)
            const float v  = al * xr[t];
            const float q  = floorf(v * 0.318309886f + 0.5f);  // v = q pi + r, |r| <= pi / 2
            const float r  = (v - q * 3.14159274f) + q * 8.74227766e-8f;
            const float r2 = r * r;
            const float sv = r * (1.0f + r2 * (-1.66666667e-1f + r2 * (8.33333310e-3f + r2 * (-1.98412701e-4f +
                                 r2 * (2.75573137e-6f + r2 * -2.50521084e-8f)))));
            y[t] = xr[t] + sv * sv * be;  // (the sign of sin(v) = (-1)^q sin(r) drops out of the square)
        }
    });
    return true;
}

// CONCAT along dim 0 of two f32 matrices (the causal convolutions' left context before their input), row by row on
// the host workers (the CPU's concat goes element by element); false if the node is not that
static bool axera_gen_concat(ggml_tensor * n) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    if (n->op != GGML_OP_CONCAT || ((const int32_t *) n->op_params)[0] != 0 || n->type != GGML_TYPE_F32 ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) || a->nb[0] != sizeof(float) ||
        b->nb[0] != sizeof(float) || n->ne[2] * n->ne[3] != 1 || a->ne[2] * a->ne[3] != 1 || b->ne[2] * b->ne[3] != 1 ||
        a->ne[1] != n->ne[1] || b->ne[1] != n->ne[1]) {
        return false;
    }
    axera_timer tm("gen concat (host)");
    axera_host_pool().run(n->ne[1], [&](int64_t r) {
        float * d = (float *) ((char *) n->data + r * n->nb[1]);
        memcpy(d, (const char *) a->data + r * a->nb[1], a->ne[0] * sizeof(float));
        memcpy(d + a->ne[0], (const char *) b->data + r * b->nb[1], b->ne[0] * sizeof(float));
    });
    return true;
}

// CONT of a transposed f32 matrix (the convolutions' [T, C] <-> [C, T] layouts) in 64 x 64 blocks on the host workers;
// false if the node is not that
static bool axera_gen_transpose(ggml_tensor * n) {
    const ggml_tensor * s = n->src[0];
    if (n->op != GGML_OP_CONT || n->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) ||
        n->ne[2] * n->ne[3] != 1 || s->ne[2] * s->ne[3] != 1 || s->nb[1] != sizeof(float) || s->nb[0] < s->nb[1] * s->ne[1]) {
        return false;
    }
    const int64_t R = n->ne[1], C = n->ne[0], B = 64, nr = (R + B - 1) / B, nc = (C + B - 1) / B;
    axera_timer   tm("gen transpose (host)");
    axera_host_pool().run(nr * nc, [&](int64_t job) {
        const int64_t r0 = job / nc * B, c0 = job % nc * B;
        for (int64_t r = r0; r < std::min(R, r0 + B); ++r) {
            float * d = (float *) ((char *) n->data + r * n->nb[1]);
            for (int64_t c = c0; c < std::min(C, c0 + B); ++c) {
                d[c] = *(const float *) ((const char *) s->data + c * s->nb[0] + r * s->nb[1]);
            }
        }
    });
    return true;
}

// COL2IM_1D (a transposed convolution's GEMM result [T_in][K * OC] scattered into [OC][T_out]) on the host workers,
// by blocks of output steps over all channels (reading whole rows of the GEMM result); false if not f32
static bool axera_gen_col2im(ggml_tensor * n) {
    const ggml_tensor * s = n->src[0];
    if (n->op != GGML_OP_COL2IM_1D || n->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) ||
        !ggml_is_contiguous(n)) {
        return false;
    }
    const int32_t * p = (const int32_t *) n->op_params;
    const int64_t   S = p[0], OC = p[1], P = p[2], KOC = s->ne[0], T_in = s->ne[1], K = KOC / OC, T = n->ne[0];
    const int64_t   B = 1024, nb = (T + B - 1) / B;
    axera_timer     tm("gen col2im (host)");
    axera_host_pool().run(nb, [&](int64_t blk) {
        const int64_t t0 = blk * B, t1 = std::min(T, t0 + B);
        auto *        d  = (float *) n->data;
        for (int64_t oc = 0; oc < OC; ++oc) {
            memset(d + oc * T + t0, 0, (t1 - t0) * sizeof(float));
        }
        // the input steps whose taps reach [t0, t1): t_in s + k - P in it, 0 <= k < K
        const int64_t lo = std::max<int64_t>(0, (t0 + P - K + 1 + S - 1) / S), hi = std::min(T_in - 1, (t1 - 1 + P) / S);
        for (int64_t ti = lo; ti <= hi; ++ti) {
            const float * row = (const float *) s->data + ti * KOC;
            const int64_t k0 = std::max<int64_t>(0, t0 + P - ti * S), k1 = std::min(K, t1 + P - ti * S);
            for (int64_t oc = 0; oc < OC; ++oc) {
                float *       o = d + oc * T + ti * S - P;
                const float * c = row + oc * K;
                for (int64_t k = k0; k < k1; ++k) {
                    o[k] += c[k];
                }
            }
        }
    });
    return true;
}

// a depthwise 1-D convolution (ggml_conv_1d_dw: IM2COL ic of x [T, 1, C] into f16 rows, then the MUL_MAT m of them by
// the kernel [K, 1, C]): m computed directly in f32 on the host workers, ic not needed. False if they are not that
static bool axera_gen_dwconv(const ggml_tensor * ic, ggml_tensor * m) {
    const int32_t *     p = (const int32_t *) ic->op_params;
    const ggml_tensor * w = ic->src[0];
    const ggml_tensor * x = ic->src[1];
    const int64_t       K = w->ne[0], C = w->ne[2], OW = ic->ne[1];
    if (p[6] != 0 || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || w->ne[1] != 1 || x->ne[1] != 1 ||
        x->ne[2] != C || x->ne[3] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) || m->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(m) || m->ne[0] != OW || m->ne[1] != 1 || m->ne[2] != C ||
        (w->type != GGML_TYPE_F32 && !ggml_get_type_traits(w->type)->to_float)) {
        return false;
    }
    const int s0 = p[0], p0 = p[2], d0 = p[4];
    const int64_t IW = x->ne[0];
    axera_timer tm("gen dwconv (host)");
    axera_host_pool().run(C, [&](int64_t c) {
        std::vector<float> k(K);
        const char *       wr = (const char *) w->data + c * w->nb[2];
        if (w->type == GGML_TYPE_F32) {
            memcpy(k.data(), wr, K * sizeof(float));
        } else {
            ggml_get_type_traits(w->type)->to_float(wr, k.data(), K);
        }
        const float * xr = (const float *) ((const char *) x->data + c * x->nb[2]);
        float *       o  = (float *) ((char *) m->data + c * m->nb[2]);
        for (int64_t t = 0; t < OW; ++t) {
            float acc = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                const int64_t iw = t * s0 + j * d0 - p0;
                acc += iw >= 0 && iw < IW ? k[j] * xr[iw] : 0.0f;
            }
            o[t] = acc;
        }
    });
    return true;
}

static const char * axera_gen_label(axera_gen & g, const char * fmt, int64_t M, int64_t N, int64_t K) {
    char s[96];
    snprintf(s, sizeof(s), fmt, M, N, K);
    return g.names.insert(s).first->c_str();
}

// one MUL_MAT on the NPU if a kernel takes its shape: GEMV kernels (mul_mat_bf16_[b<B>_]n<C>_k<K>) for up to 8 f32
// rows times a weight, else mm_bf16_m<M>_n<N>_k<K> (x [M, K], w [N, K] bf16 -> out [M, N] f32, written into the node).
// before() runs first (the CPU nodes ahead of it); false if no kernel takes it
template <typename F> static bool axera_gen_mm(ggml_backend_axera_context * ctx, axera_gen & g, ggml_tensor * n, F before) {
    int64_t M, N, K;
    if (!axera_gen_mm_shape(n, &M, &N, &K)) {
        return false;
    }
    auto ic = g.im2col.find(n);
    if (ic != g.im2col.end()) {  // a convolution: its conv1d kernel if there is one
        const std::string conv = axera_gen_conv_name(ic->second, M);
        if (!conv.empty() && axera_gen_have(g, conv)) {
            before();
            axera_gen_conv(g, n, ic->second, conv);
            return true;
        }
    }
    const ggml_tensor * a    = n->src[0];
    const ggml_tensor * b    = n->src[1];
    auto *              dctx = (const ggml_backend_axera_buffer_context *) n->buffer->context;
    char                name[96];
    snprintf(name, sizeof(name), "mm_bf16_m%" PRId64 "_n%" PRId64 "_k%" PRId64, M, N, K);
    const bool gemm = axera_gen_have(g, name);
    if (!gemm &&
        !(M <= 8 && b->type == GGML_TYPE_F32 && axera_gen_is_weight(a) && axera_kernels().mul_mat_chunk(N, K) > 0)) {
        return false;
    }
    before();
    axera_timer tm(axera_gen_label(g, gemm ? "gen mm (npu) m%" PRId64 " n%" PRId64 " k%" PRId64 :
                                             "gen gemv (npu) m%" PRId64 " n%" PRId64 " k%" PRId64, M, N, K));
    axera_timer tw(ic != g.im2col.end() ? "gen mm: im2col rows (host)" : "gen mm: w (host)");
    const auto w = ic != g.im2col.end() ? axera_gen_im2col_bf16(ic->second, g.tmp[0]) : axera_gen_bf16(g, a, N, g.tmp[0]);
    tw.stop();
    if (!gemm) {
        for (int64_t r = 0; r < M;) {
            const float * x    = (const float *) ((const char *) b->data + r * b->nb[1]);
            int           done = 0;
            for (int Bn : { 8, 4, 2 }) {
                if (done == 0 && r + Bn <= M) {
                    float * dsts[8];
                    for (int i = 0; i < Bn; ++i) {
                        dsts[i] = (float *) ((char *) n->data + (r + i) * n->nb[1]);
                    }
                    done = axera_gemv_batch(ctx, w.first, w.second, nullptr, N, K, x, Bn, dsts) ? Bn : 0;
                }
            }
            if (done == 0) {
                axera_gemv_raw_to(ctx, w.first, w.second, N, K, x, dctx, (float *) ((char *) n->data + r * n->nb[1]));
                done = 1;
            }
            r += done;
        }
        return true;
    }
    axera_timer         tx("gen mm: x (host)");
    const auto          x  = axera_gen_bf16(g, b, M, g.tmp[1]);
    tx.stop();
    axera_timer         tr("gen mm: run (npu)");
    ggml_axera_kernel * k  = axera_kernels().get(name);
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    std::vector<AX_ENGINE_IO_BUFFER_T> in(2), out(1);
    in[ix]  = axera_io(x.first, x.second, (uint32_t) (M * K * sizeof(ggml_bf16_t)));
    in[iw]  = axera_io(w.first, w.second, (uint32_t) (N * K * sizeof(ggml_bf16_t)));
    out[io] = axera_io(dctx, n->data, (uint32_t) ggml_nbytes(n));
    dctx->invalidate(n->data, ggml_nbytes(n));  // no dirty line of the output left to be written back over the NPU's
    if (axera_run_sync(k, in, out) != 0) {
        GGML_ABORT("%s: %s failed\n", __func__, name);
    }
    dctx->invalidate(n->data, ggml_nbytes(n));
    return true;
}

// a single-row GEMV and the later MUL_MATs of the same input row by weights of the same width (within the next nodes)
// as one GEMV of the joined weights: the first node's result written now, the others' kept in pending. False if the
// node has no such group or no kernel takes the joined rows
template <typename F>
static bool axera_gen_group(ggml_backend_axera_context * ctx, axera_gen & g, struct ggml_cgraph * cgraph, int i,
                            const std::set<const ggml_tensor *> & skip, F before) {
    ggml_tensor * n = cgraph->nodes[i];
    int64_t       M, N, K;
    auto          single = [&](const ggml_tensor * t) {
        int64_t m, nn, k;
        return t->op == GGML_OP_MUL_MAT && (t->flags & GGML_TENSOR_FLAG_COMPUTE) && axera_gen_mm_shape(t, &m, &nn, &k) &&
               m == 1 && t->src[1]->type == GGML_TYPE_F32 && axera_gen_is_weight(t->src[0]);
    };
    if (!single(n) || !axera_gen_mm_shape(n, &M, &N, &K)) {
        return false;
    }
    std::vector<ggml_tensor *> grp = { n };
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 96); j++) {
        ggml_tensor * t = cgraph->nodes[j];
        if (t->src[1] == n->src[1] && single(t) && t->src[0]->ne[0] == K && !g.pending.count(t) && !skip.count(t)) {
            grp.push_back(t);
        }
    }
    int64_t Nt = 0;
    std::vector<const void *> key;
    for (ggml_tensor * t : grp) {
        Nt += t->src[0]->ne[1];
        key.push_back(t->src[0]->data);
    }
    if (grp.size() < 2 || axera_kernels().mul_mat_chunk(Nt, K) == 0) {
        return false;
    }
    before();
    axera_timer tm(axera_gen_label(g, "gen gemv group (npu) m%" PRId64 " n%" PRId64 " k%" PRId64, 1, Nt, K));
    auto & w = g.wg[key];
    if (!w) {
        w        = axera_cmm_alloc((size_t) Nt * K * sizeof(ggml_bf16_t));
        auto * d = (ggml_bf16_t *) w->vir;
        for (ggml_tensor * t : grp) {
            axera_gen_to_bf16(t->src[0], t->src[0]->ne[1], d);
            d += t->src[0]->ne[1] * K;
        }
        w->flush(w->vir, (size_t) Nt * K * sizeof(ggml_bf16_t));
    }
    if (!g.gout || g.gout->size < (size_t) Nt * sizeof(float)) {
        g.gout = axera_cmm_alloc((size_t) Nt * sizeof(float));
    }
    auto * y = (float *) g.gout->vir;
    axera_gemv_raw_to(ctx, w.get(), w->vir, Nt, K, (const float *) n->src[1]->data, g.gout.get(), y);
    for (ggml_tensor * t : grp) {
        const int64_t r = t->src[0]->ne[1];
        if (t == n) {
            memcpy(n->data, y, r * sizeof(float));
        } else {
            g.pending[t].assign(y, y + r);
        }
        y += r;
    }
    return true;
}

// one code predictor layer call of a graph: the layer's input and output nodes, its index and position
struct axera_gen_cp_call {
    ggml_tensor * x, * out;
    int           il, pos;
};

// the code predictor's layer calls of a graph (layer il at position p: rms_norm(x) * a.gen.code.blk.<il>.ln1 ... out =
// x' + ffn_down(...)), the nodes only they need marked in skip; empty if the graph has none or a node outside them
// needs one inside
static std::vector<axera_gen_cp_call> axera_gen_cp_calls(struct ggml_cgraph * cgraph, std::set<const ggml_tensor *> & skip) {
    std::vector<axera_gen_cp_call> calls;
    std::map<int, int>             count;  // layer -> calls so far
    std::map<int, ggml_tensor *>   open;   // layer -> input of its call in progress
    auto layer_of = [](const ggml_tensor * w, const char * suffix) {
        const std::string n = axera_base_name(w->name);
        int               il, len = 0;
        char              tail[48];
        if (sscanf(n.c_str(), "a.gen.code.blk.%d.%47s%n", &il, tail, &len) == 2 && strcmp(tail, suffix) == 0) {
            return il;
        }
        return -1;
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        int il;
        if (n->op == GGML_OP_MUL && n->src[0]->op == GGML_OP_RMS_NORM && (il = layer_of(n->src[1], "ln1.weight")) >= 0) {
            if (open.count(il)) {  // the previous call of the layer has no output in the graph (position 0's last
                               // layer: only its K / V are read): a call that only writes them (out nullptr)
                calls.push_back({ open[il], nullptr, il, count[il]++ });
            }
            open[il] = n->src[0]->src[0];
        } else if (n->op == GGML_OP_ADD && n->src[1]->op == GGML_OP_MUL_MAT &&
                   (il = layer_of(n->src[1]->src[0], "ffn_down.weight")) >= 0 && open.count(il)) {
            calls.push_back({ open[il], n, il, count[il]++ });
            open.erase(il);
        }
    }
    for (const auto & o : open) {
        calls.push_back({ o.second, nullptr, o.first, count[o.first]++ });
    }
    if (calls.empty()) {
        return calls;
    }
    // inside a call: what its output needs, back to its input (and the weights)
    std::set<const ggml_tensor *> outs, ins;
    for (const auto & c : calls) {
        if (c.out) {
            outs.insert(c.out);
        }
        ins.insert(c.x);
    }
    std::vector<const ggml_tensor *> stack;
    for (const auto & c : calls) {
        for (int s = 0; c.out && s < GGML_MAX_SRC && c.out->src[s]; ++s) {
            stack.push_back(c.out->src[s]);
        }
    }
    while (!stack.empty()) {
        const ggml_tensor * t = stack.back();
        stack.pop_back();
        if (t->op == GGML_OP_NONE || ins.count(t) || outs.count(t) || !skip.insert(t).second) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            stack.push_back(t->src[s]);
        }
        if (t->view_src) {
            stack.push_back(t->view_src);
        }
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || skip.count(n) || outs.count(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            if (skip.count(n->src[s]) || (n->src[s]->view_src && skip.count(n->src[s]->view_src))) {
                if (axera_debug()) {
                    fprintf(stderr, "axera gen: %zu code predictor layer calls, but %s (%s) outside them reads %s (%s)\n",
                            calls.size(), n->name, ggml_op_desc(n), n->src[s]->name, ggml_op_desc(n->src[s]));
                }
                skip.clear();
                return {};
            }
        }
    }
    return calls;
}

// binds the fused layer kernel to the code predictor's weights (bf16 or MXFP8, converted once)
static bool axera_gen_cp_bind(axera_gen & g, const std::map<std::string, ggml_tensor *> & t, int n_layer, int H) {
    const ggml_tensor * w0 = axera_graph_find(t, "a.gen.code.blk.0.ln1.weight");
    if (!w0) {
        return false;
    }
    if (g.cp && g.cp_bound == w0->data && (int) g.cp_layers.size() == n_layer) {
        return true;
    }
    if (!g.cp) {
        // the run's weight format first (MXFP8 with GGML_AXERA_WEIGHTS=mxfp8: about 17% faster per layer call, the
        // generations end as with bf16), else the other one
        const char * bf = "qwen3tts_cp_attn_kvc_c4096", * mx = "qwen3tts_cp_attn_mx8_kvc_c4096";
        for (const char * name : { axera_weights_mx8() ? mx : bf, axera_weights_mx8() ? bf : mx }) {
            if (axera_kernels().has(name)) {
                g.cp = axera_kernels().get(name);
                break;
            }
        }
        if (!g.cp) {
            return false;
        }
        axera_q35_context(g.cp, g.cp_LMAX, g.cp_BC);
        const int ix = g.cp->input("x"), ic = g.cp->input("cos");
        if (ix < 0 || ic < 0 || g.cp->info->pInputs[ix].nSize != (uint32_t) H * sizeof(ggml_bf16_t)) {
            g.cp = nullptr;
            return false;
        }
        g.cp_H   = H;
        g.cp_ROT = (int) (g.cp->info->pInputs[ic].nSize / sizeof(ggml_bf16_t)) * 2;
        g.X      = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        g.Y      = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        g.POS    = axera_cmm_alloc(sizeof(int32_t));
        g.NBLK   = axera_cmm_alloc(sizeof(int32_t));
        g.MASK   = axera_cmm_alloc(g.cp_LMAX * sizeof(ggml_bf16_t));
        g.COS    = axera_cmm_alloc(g.cp_ROT / 2 * sizeof(ggml_bf16_t));
        g.SIN    = axera_cmm_alloc(g.cp_ROT / 2 * sizeof(ggml_bf16_t));
    }
    static const std::map<std::string, const char *> names = {
        { "ln1", "ln1.weight" },         { "ln2", "ln2.weight" },         { "q_norm", "attn_q_norm.weight" },
        { "k_norm", "attn_k_norm.weight" }, { "q", "attn_q.weight" },    { "k", "attn_k.weight" },
        { "v", "attn_v.weight" },        { "o", "attn_out.weight" },      { "gate", "ffn_gate.weight" },
        { "up", "ffn_up.weight" },       { "down", "ffn_down.weight" },
    };
    ggml_axera_kernel * k = g.cp;
    g.cp_layers.clear();
    g.cp_layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & ly = g.cp_layers[il];
        ly.in.assign(k->info->nInputSize, {});
        ly.out.assign(k->info->nOutputSize, {});
        auto get = [&](const std::string & n) -> const ggml_tensor * {
            auto it = names.find(n);
            const ggml_tensor * w = it == names.end() ? nullptr :
                axera_graph_find(t, ("a.gen.code.blk." + std::to_string(il) + "." + it->second).c_str());
            if (!w) {
                GGML_ABORT("axera gen: code predictor layer %d has no weight for kernel input %s\n", il, n.c_str());
            }
            return w;
        };
        std::map<std::string, std::pair<ggml_backend_axera_buffer_context *, ggml_backend_axera_buffer_context *>> mx;
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            const std::string n    = k->info->pInputs[i].pName;
            const uint32_t    size = k->info->pInputs[i].nSize;
            const std::map<std::string, axera_cmm *> fixed = {
                { "x", &g.X }, { "pos", &g.POS }, { "nblk", &g.NBLK }, { "mask", &g.MASK }, { "cos", &g.COS }, { "sin", &g.SIN },
            };
            auto f = fixed.find(n);
            if (f != fixed.end()) {
                ly.in[i] = axera_io(f->second->get(), (*f->second)->vir, size);
            } else if ((n.rfind("w_", 0) == 0 || n.rfind("s_", 0) == 0) && k->input(("s_" + n.substr(2)).c_str()) >= 0) {
                const std::string base = n.substr(2);
                if (!mx.count(base)) {
                    axera_cmm q, sc;
                    axera_quantize_mxfp8(get(base), q, sc);
                    mx[base] = { q.get(), sc.get() };
                    ly.own.push_back(std::move(q));
                    ly.own.push_back(std::move(sc));
                }
                auto * b = n[0] == 'w' ? mx[base].first : mx[base].second;
                GGML_ASSERT(b->size >= size);
                ly.in[i] = axera_io(b, b->vir, size);
            } else {  // a bf16 weight (w_<x> of the bf16 kernel) or an f32 norm weight
                const ggml_tensor * w  = get(n.rfind("w_", 0) == 0 ? n.substr(2) : n);
                const bool          bf = (size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == size;
                GGML_ASSERT(bf || (size_t) ggml_nelements(w) * sizeof(float) == size);
                ly.own.push_back(bf ? axera_to_bf16(w) : axera_cmm_alloc(size));
                auto * b = ly.own.back().get();
                if (!bf) {
                    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
                        memcpy((float *) b->vir + r * w->ne[0], row, w->ne[0] * sizeof(float));
                    });
                    b->flush(b->vir, size);
                }
                ly.in[i] = axera_io(b, b->vir, size);
            }
        }
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            const std::string n    = k->info->pOutputs[i].pName;
            const uint32_t    size = k->info->pOutputs[i].nSize;
            if (n == "out") {
                ly.out[i] = axera_io(g.Y.get(), g.Y->vir, size);
            } else {  // kc / vc: resident, the cells of this frame's positions written before they are read
                ly.own.push_back(axera_cmm_alloc(size));
                ly.out[i] = axera_io(ly.own.back().get(), ly.own.back()->vir, size);
            }
        }
    }
    g.cp_bound = w0->data;
    return true;
}

// one code predictor layer call on the NPU: out = layer il (x) at position pos
static void axera_gen_cp_run(axera_gen & g, const axera_gen_cp_call & c, float rope_theta) {
    const int H = g.cp_H;
    if (c.il == 0) {  // a new position: its mask cell, its RoPE rows (position 0: a new frame, every cell masked)
        auto * mask = (ggml_bf16_t *) g.MASK->vir;
        if (c.pos == 0) {
            std::fill(mask, mask + g.cp_LMAX, ggml_fp32_to_bf16(-3e38f));
        }
        mask[c.pos]                = ggml_fp32_to_bf16(0.0f);
        *(int32_t *) g.POS->vir  = c.pos;
        *(int32_t *) g.NBLK->vir = c.pos / g.cp_BC + 1;
        for (int i = 0; i < g.cp_ROT / 2; ++i) {
            const float a                    = (float) c.pos * powf(rope_theta, -2.0f * i / g.cp_ROT);
            ((ggml_bf16_t *) g.COS->vir)[i] = ggml_fp32_to_bf16(cosf(a));
            ((ggml_bf16_t *) g.SIN->vir)[i] = ggml_fp32_to_bf16(sinf(a));
        }
        g.MASK->flush(g.MASK->vir, (size_t) g.cp_BC * (c.pos / g.cp_BC + 1) * sizeof(ggml_bf16_t));
        g.POS->flush(g.POS->vir, sizeof(int32_t));
        g.NBLK->flush(g.NBLK->vir, sizeof(int32_t));
        g.COS->flush(g.COS->vir, g.COS->size);
        g.SIN->flush(g.SIN->vir, g.SIN->size);
    }
    ggml_fp32_to_bf16_row((const float *) c.x->data, (ggml_bf16_t *) g.X->vir, H);
    g.X->flush(g.X->vir, H * sizeof(ggml_bf16_t));
    auto & ly = g.cp_layers[c.il];
    if (axera_run_sync(g.cp, ly.in, ly.out) != 0) {
        GGML_ABORT("axera gen: code predictor layer %d failed\n", c.il);
    }
    if (c.out) {
        g.Y->invalidate(g.Y->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) g.Y->vir, (float *) c.out->data, H);
    }
}

// a graph of the llama.cpp model (the template's), not of mtmd
bool axera_llama_graph(const struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const char * s = cgraph->nodes[i]->name;
        if (strcmp(s, "attn_norm-0") == 0 || strcmp(s, "mtp_h_input") == 0 || strcmp(s, "result_output") == 0 ||
            strcmp(s, "result_norm") == 0) {
            return true;
        }
    }
    return false;
}

// GGML_AXERA_PROF=2: every CPU node timed on its own, by op and shape
static bool axera_gen_prof_nodes() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_PROF");
        return e && atoi(e) >= 2;
    }();
    return on;
}

// the analysis of a graph for axera_gen_compute: the code predictor's layer calls (by output node; those without one by
// their input), the nodes only they need, the layers' RoPE base
struct axera_gen_analysis {
    std::set<const ggml_tensor *>                            skip;
    std::vector<axera_gen_cp_call>                           calls;
    std::map<const ggml_tensor *, const axera_gen_cp_call *> call_of, after;
    float                                                    theta = 0.0f;
};

static void axera_gen_analyze(ggml_backend_axera_context * ctx, axera_gen & g, struct ggml_cgraph * cgraph,
                              axera_gen_analysis & an) {
    std::set<const ggml_tensor *> &  skip  = an.skip;
    std::vector<axera_gen_cp_call> & calls = an.calls;
    float &                          theta = an.theta;
    axera_timer ta("gen graph analysis (host)");
    if (!g.cp_off) {
        calls = axera_gen_cp_calls(cgraph, skip);
        int n_layer = 0;
        for (const auto & c : calls) {
            n_layer = std::max(n_layer, c.il + 1);
        }
        for (int i = 0; i < cgraph->n_nodes && !calls.empty() && theta == 0.0f; i++) {  // the layers' RoPE base
            const ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_ROPE && skip.count(n)) {
                memcpy(&theta, (const int32_t *) n->op_params + 5, sizeof(float));
            }
        }
        if (axera_debug()) {
            fprintf(stderr, "axera gen: %zu code predictor layer calls (%d layers), %zu nodes skipped, rope base %g\n",
                    calls.size(), n_layer, skip.size(), theta);
        }
        if (!calls.empty() && !axera_gen_cp_bind(g, axera_graph_names(ctx, cgraph), n_layer, (int) calls[0].x->ne[0])) {
            if (axera_debug()) {
                fprintf(stderr, "axera gen: no code predictor layer kernel\n");
            }
            g.cp_off = true;  // no kernel for it: the layers as they are from now on
            calls.clear();
            skip.clear();
        }
        std::set<const ggml_tensor *> outs;
        for (const auto & c : calls) {
            outs.insert(c.out);
        }
        for (const auto & c : calls) {  // (a call without output runs right after the call that writes its input)
            if (c.pos >= g.cp_LMAX || c.x->type != GGML_TYPE_F32 || ggml_nelements(c.x) != g.cp_H ||
                !ggml_is_contiguous(c.x) || (c.out && c.out->type != GGML_TYPE_F32) || (!c.out && !outs.count(c.x))) {
                calls.clear();
                skip.clear();
                break;
            }
        }
    }
    for (const auto & c : calls) {
        (c.out ? an.call_of[c.out] : an.after[c.x]) = &c;
    }
}

enum ggml_status axera_gen_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
    if (!ctx->gen) {
        ctx->gen = std::make_shared<axera_gen>();
    }
    axera_gen & g = *ctx->gen;

    // the code predictor's layers on the fused layer kernel, the nodes only they need skipped: analyzed once per graph
    // (mtmd builds the same graph in the same memory for every frame)
    std::vector<uintptr_t> key;
    key.reserve((size_t) cgraph->n_nodes * (3 + GGML_MAX_SRC));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        key.push_back((uintptr_t) n);
        key.push_back((uintptr_t) n->op);
        key.push_back((uintptr_t) n->flags);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            key.push_back((uintptr_t) n->src[s]);
        }
    }
    if (!g.analysis || key != g.key) {
        auto an = std::make_shared<axera_gen_analysis>();
        axera_gen_analyze(ctx, g, cgraph, *an);
        g.analysis = an;
        g.key      = std::move(key);
    }
    const axera_gen_analysis &            an    = *g.analysis;
    const std::set<const ggml_tensor *> & skip  = an.skip;
    const auto &                          call_of = an.call_of;
    const auto &                          after   = an.after;
    auto cp_run = [&](const axera_gen_cp_call & c) {
        axera_timer tm("gen code predictor layer (npu)");
        axera_gen_cp_run(g, c, an.theta);
        auto a = c.out ? after.find(c.out) : after.end();
        if (a != after.end()) {
            axera_gen_cp_run(g, *a->second, an.theta);
        }
    };

    std::vector<ggml_tensor *> run;  // CPU nodes not yet computed
    auto cpu = [&] {
        axera_cpu_compute_nodes(ctx, cgraph, run.data(), (int) run.size(), "gen (cpu)");
        run.clear();
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || skip.count(n)) {
            continue;
        }
        auto c = call_of.find(n);
        if (c != call_of.end()) {
            cpu();
            cp_run(*c->second);
            continue;
        }
        if (n->op == GGML_OP_IM2COL) {
            if (const ggml_tensor * m = axera_gen_im2col_user(g, cgraph, i)) {
                g.im2col[m] = n;
                continue;
            }
            if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
                cgraph->nodes[i + 1]->src[0] == n && cgraph->nodes[i + 1]->src[1] == n->src[0]) {
                cpu();
                if (axera_gen_dwconv(n, cgraph->nodes[i + 1])) {
                    i += 1;
                    continue;
                }
            }
        }
        if ((n->op == GGML_OP_CONCAT || n->op == GGML_OP_CONT || n->op == GGML_OP_COL2IM_1D) &&
            ggml_nelements(n) >= 65536 && !axera_gen_const_root(n)) {  // (large ones: the host workers' start costs)
            cpu();
            if (axera_gen_concat(n) || axera_gen_transpose(n) || axera_gen_col2im(n)) {
                continue;
            }
        }
        if (n->op == GGML_OP_CONT && axera_gen_const_skip(g, cgraph, i)) {  // a constant the NPU already has as bf16
            continue;
        }
        if (n->op == GGML_OP_MUL && !axera_gen_prof_nodes() && axera_gen_snake(cgraph, i, cpu)) {
            i += 4;
            continue;
        }
        auto p = g.pending.find(n);
        if (p != g.pending.end()) {  // computed with an earlier node of its group
            cpu();
            memcpy(n->data, p->second.data(), p->second.size() * sizeof(float));
            g.pending.erase(p);
            continue;
        }
        if (n->op == GGML_OP_MUL_MAT && (axera_gen_group(ctx, g, cgraph, i, skip, cpu) || axera_gen_mm(ctx, g, n, cpu))) {
            continue;
        }
        if (axera_gen_prof_nodes()) {
            cpu();
            char s[128];
            snprintf(s, sizeof(s), "gen cpu %s %s [%" PRId64 " %" PRId64 "]%s", ggml_op_desc(n),
                     ggml_type_name(n->src[0] ? n->src[0]->type : n->type), n->ne[0], n->ne[1],
                     n->op == GGML_OP_MUL_MAT ? (" k" + std::to_string(n->src[0]->ne[0])).c_str() : "");
            axera_cpu_compute_nodes(ctx, cgraph, &cgraph->nodes[i], 1, g.names.insert(s).first->c_str());
            continue;
        }
        run.push_back(n);
    }
    cpu();
    g.pending.clear();
    g.im2col.clear();
    return GGML_STATUS_SUCCESS;
}
