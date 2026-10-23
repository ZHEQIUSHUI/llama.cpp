// Axera NPU backend: graphs without a template of their own (mtmd's generation graphs: the Qwen3-TTS code predictor
// and codec decoder, its speaker encoder): run by the CPU backend from inside the backend, the matrix products that
// have an NPU kernel of their shape on the NPU

#include "ggml-axera-q35.h"  // axera_q35_context: the fused layer kernel's LMAX / BC

#include <set>
#include <sstream>
#include <sys/stat.h>

// a convolution's epilogue the conv1d kernel can compute (its _b / _br variant): the ADD of its bias (a [1, OC] view
// of an f32 weight) and, after it, the ADD of a residual [OW, OC] f32; their outputs written by the kernel
struct axera_gen_conv_epi {
    ggml_tensor *       cont = nullptr;  // ggml_conv_1d's CONT of its permuted result (a copy for one batch)
    ggml_tensor *       add = nullptr, *radd = nullptr;
    const ggml_tensor * bias = nullptr, *res = nullptr;
};

// a SnakeBeta whose output only a convolution reads (directly, or after the CONCAT of the convolution's left context,
// whose tail the graph also keeps as state): computed by the host straight into the conv1d kernel's bf16 input, and
// only that tail of the CONCAT in f32; the snake's own output and the CONCAT need not be written
struct axera_gen_snake_fuse {
    ggml_tensor *       concat = nullptr;  // nullptr: the IM2COL reads the snake's output
    const ggml_tensor * ic = nullptr, *mm = nullptr;
};

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
    std::set<const ggml_tensor *>                      done;  // nodes a kernel computed already (conv epilogues)
    const std::map<const ggml_tensor *, axera_gen_conv_epi> * epi = nullptr;  // the running graph's, by MUL_MAT
    const std::map<const ggml_tensor *, axera_gen_snake_fuse> * sfuse = nullptr;  // the running graph's, by snake start
    const ggml_tensor * prebf16 = nullptr;  // the conv input tmp[1] holds as bf16 already (a fused snake's)
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
    // or all layers of one position in one call (qwen3tts_cp_stack<layers>[_mx8]_c<LMAX>, the hidden state on chip
    // between them): its IO, the layers' K / V resident in it
    ggml_axera_kernel * cps = nullptr;
    layer               cps_io;
    // or one whole predictor position per call, its input projection and codebook head included
    // (qwen3tts_cp_step<layers>_mx8_h<HT>_c<LMAX>, q3tts_cp_step.py): IO bound once but the head (per call), the
    // step's embedding row E, cos | sin CS, pos | nblk PN, its own mask MASKX and context, its logits buffer;
    // GGML_AXERA_CP_STEP=0 off
    ggml_axera_kernel *                 cpx = nullptr;
    layer                               cpx_io;
    axera_cmm                           E, CS, PN, MASKX;
    ggml_backend_axera_buffer_context * cpx_lg = nullptr;
    int                                 cpx_e = -1, cpx_head = -1, cpx_logits = -1;
    int                                 cpx_LMAX = 0, cpx_BC = 0, cpx_ROT = 0;
    // or a whole frame in one call, the sampling on chip (qwen3tts_cp_frame<layers>_mx8_h<HT>_k<top_k>_c<LMAX>,
    // q3tts_cp_frame.py; GGML_AXERA_CP_FRAME=1): its IO bound once, the frame's inputs E0 / E1 / RND, the positions'
    // RoPE rows, pos | nblk and masks, its codes output
    ggml_axera_kernel *                 cpf = nullptr;
    layer                               cpf_io;
    axera_cmm                           F_E0, F_E1, F_RND, F_CS, F_PN, F_MASK;
    ggml_backend_axera_buffer_context * cpf_codes = nullptr, *cpf_lgs = nullptr;
    const ggml_tensor *                 cpf_bound = nullptr;
    int                                 cpf_topk  = 0;
    const ggml_tensor * cpx_bound = nullptr, *cpx_norm = nullptr;  // the weights the step kernel's packs hold
    axera_cmm           X, Y, POS, NBLK, MASK, COS, SIN;
    bool                cp_off = false;  // the kernel is missing or a graph did not fit: the layers stay as they are
    std::vector<uintptr_t>                      key;       // the last graph's nodes, and its analysis
    std::shared_ptr<struct axera_gen_analysis> analysis;
    // the analyses of the last few graphs by their key (the code generation and code2wav graphs alternate: neither
    // analyzed again when the other one ran in between)
    std::map<std::vector<uintptr_t>, std::shared_ptr<struct axera_gen_analysis>> analyses;
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
// repacked once ([OC][IC][K] -> [K][OC][IC]), the output written into n; or, with an epilogue e and the kernel's
// variant of it (<name>_br: bias and residual, <name>_b: bias), into the epilogue's last ADD, the ADDs marked done
static void axera_gen_conv(axera_gen & g, ggml_tensor * n, const ggml_tensor * ic, std::string name,
                           const axera_gen_conv_epi * e) {
    const bool br = e && e->radd && axera_gen_have(g, name + "_br");
    const bool bo = e && !br && axera_gen_have(g, name + "_b");
    name += br ? "_br" : bo ? "_b" : "";
    ggml_tensor * dst = br ? e->radd : bo ? e->add : e && e->cont ? e->cont : n;
    const ggml_tensor * x  = ic->src[1];
    const ggml_tensor * w  = ic->src[0];
    const int64_t       KW = w->ne[0], IC = x->ne[1], L = x->ne[0], OC = n->ne[1];
    auto *              dctx = (const ggml_backend_axera_buffer_context *) dst->buffer->context;
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
    if (g.prebf16 != x) {
        axera_timer tx("gen conv: x (host)");
        axera_gen_to_bf16(x, IC, (ggml_bf16_t *) g.tmp[1]->vir);
    }
    g.prebf16 = nullptr;
    ggml_axera_kernel * k  = axera_kernels().get(name);
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    const int           ib = k->input("b"), ir = k->input("res");
    const uint32_t      n_in = 2 + (br || bo) + br;
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == n_in && k->info->nOutputSize == 1);
    GGML_ASSERT((ib >= 0) == (br || bo) && (ir >= 0) == br);
    std::vector<AX_ENGINE_IO_BUFFER_T> in(n_in), out(1);
    in[ix]  = axera_io(g.tmp[1].get(), g.tmp[1]->vir, (uint32_t) xb);
    in[iw]  = axera_io(wk.get(), wk->vir, (uint32_t) wk->size);
    out[io] = axera_io(dctx, dst->data, (uint32_t) ggml_nbytes(dst));
    if (ib >= 0) {  // the bias, f32 (a CMM copy unless it is in CMM already)
        const ggml_tensor * b  = e->bias;
        const size_t        bs = ggml_nbytes(b);
        if (b->buffer && ggml_backend_buffer_is_axera(b->buffer)) {
            in[ib] = axera_io((const ggml_backend_axera_buffer_context *) b->buffer->context, b->data, (uint32_t) bs);
        } else {
            auto & c = g.w[{ b->data, bs + 2 }];  // (+2: not a bf16 copy of the same data)
            if (!c) {
                c = axera_cmm_alloc(bs);
                memcpy(c->vir, b->data, bs);
                c->flush(c->vir, bs);
            }
            in[ib] = axera_io(c.get(), c->vir, (uint32_t) bs);
        }
    }
    if (ir >= 0) {  // the residual, the host's f32 written back to DRAM first
        auto * rctx = (const ggml_backend_axera_buffer_context *) e->res->buffer->context;
        rctx->flush(e->res->data, ggml_nbytes(e->res));
        in[ir] = axera_io(rctx, e->res->data, (uint32_t) ggml_nbytes(e->res));
    }
    {
        axera_timer tc("gen conv: cache (host)");
        dctx->invalidate(dst->data, ggml_nbytes(dst));  // no dirty line of the output left to be written back over the NPU's
    }
    {
        axera_timer tr("gen conv: run (npu)");
        if (axera_run_sync(k, in, out) != 0) {
            GGML_ABORT("%s: %s failed\n", __func__, name.c_str());
        }
    }
    axera_timer tc("gen conv: cache (host)");
    dctx->invalidate(dst->data, ggml_nbytes(dst));
    if (e && e->cont) {
        g.done.insert(e->cont);
    }
    if (br || bo) {
        g.done.insert(e->add);
    }
    if (br) {
        g.done.insert(e->radd);
    }
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

// x + sin(al x)^2 be, sin by range reduction and a polynomial (vectorizes)
static inline float axera_snake_val(float x, float al, float be) {
    const float v  = al * x;
    const float q  = floorf(v * 0.318309886f + 0.5f);  // v = q pi + r, |r| <= pi / 2
    const float r  = (v - q * 3.14159274f) + q * 8.74227766e-8f;
    const float r2 = r * r;
    const float sv = r * (1.0f + r2 * (-1.66666667e-1f + r2 * (8.33333310e-3f + r2 * (-1.98412701e-4f +
                         r2 * (2.75573137e-6f + r2 * -2.50521084e-8f)))));
    return x + sv * sv * be;  // (the sign of sin(v) = (-1)^q sin(r) drops out of the square)
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
        for (int64_t t = t0; t < t1; ++t) {
            y[t] = axera_snake_val(xr[t], al, be);
        }
    });
    return true;
}

// the snake at node i fused into its convolution's input (axera_gen_snake_fuse f): bf16 rows [IC][L] into tmp[1] (the
// CONCAT's left context, then the snake), the CONCAT's last pad steps of each row in f32 (the state the graph keeps);
// false if the nodes are not a snake or the convolution has no kernel
template <typename F> static bool axera_gen_snake_fused(axera_gen & g, struct ggml_cgraph * cgraph, int i,
                                                       const axera_gen_snake_fuse & f, F before) {
    ggml_tensor ** n = cgraph->nodes + i;
    const ggml_tensor * x = n[0]->src[0];
    const ggml_tensor * a = n[0]->src[1];
    const ggml_tensor * b = n[3]->src[1];
    const std::string   conv = axera_gen_conv_name(f.ic, f.mm->ne[1]);
    if (i + 4 >= cgraph->n_nodes || n[4]->op != GGML_OP_ADD || conv.empty() || !axera_gen_have(g, conv) ||
        x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * left = f.concat ? f.concat->src[0] : nullptr;
    const int64_t       T = x->ne[0], C = x->ne[1], P = left ? left->ne[0] : 0, L = P + T;
    if ((left && (left->type != GGML_TYPE_F32 || !ggml_is_contiguous(left) || left->ne[1] != C ||
                  f.concat->ne[0] != L || !ggml_is_contiguous(f.concat))) ||
        f.ic->src[1] != (f.concat ? f.concat : n[4]) || f.ic->src[1]->ne[0] != L) {
        return false;
    }
    before();
    const size_t xb = (size_t) C * L * sizeof(ggml_bf16_t);
    if (!g.tmp[1] || g.tmp[1]->size < xb) {
        g.tmp[1] = axera_cmm_alloc_nc(xb);
    }
    auto *        dst = (ggml_bf16_t *) g.tmp[1]->vir;
    const int64_t TB = 4096, nt = (L + TB - 1) / TB;
    axera_timer   tm("gen snake + conv input (host)");
    axera_host_pool().run(C * nt, [&](int64_t job) {
        const int64_t c = job / nt, t0 = job % nt * TB, t1 = std::min(L, t0 + TB);
        const float   al = ((const float *) a->data)[c], be = ((const float *) b->data)[c];
        const float * xr = (const float *) ((const char *) x->data + c * x->nb[1]);
        const float * lr = left ? (const float *) ((const char *) left->data + c * left->nb[1]) : nullptr;
        float *       tail = f.concat ? (float *) ((char *) f.concat->data + c * f.concat->nb[1]) : nullptr;
        ggml_bf16_t   row[TB];
        float         v[TB];
        for (int64_t t = t0; t < t1; ++t) {
            v[t - t0] = t < P ? lr[t] : axera_snake_val(xr[t - P], al, be);
        }
        ggml_fp32_to_bf16_row(v, row, t1 - t0);
        memcpy(dst + c * L + t0, row, (t1 - t0) * sizeof(ggml_bf16_t));
        for (int64_t t = std::max(t0, L - P); tail && t < t1; ++t) {  // the state's steps
            tail[t] = v[t - t0];
        }
    });
    if (f.concat) {
        g.done.insert(f.concat);
    }
    g.prebf16 = f.ic->src[1];
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
            const auto e = g.epi ? g.epi->find(n) : decltype(g.epi->end())();
            axera_gen_conv(g, n, ic->second, conv, g.epi && e != g.epi->end() ? &e->second : nullptr);
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
    const auto  tw0 = std::chrono::steady_clock::now();
    const auto w = ic != g.im2col.end() ? axera_gen_im2col_bf16(ic->second, g.tmp[0]) : axera_gen_bf16(g, a, N, g.tmp[0]);
    tw.stop();
    static const bool trace = std::getenv("GGML_AXERA_TRACE") != nullptr;
    if (trace) {
        fprintf(stderr, "axera trace: mm %s m%" PRId64 " n%" PRId64 " k%" PRId64 " w %s (%s, %s, weight %d) %.3f ms\n", n->name, M, N,
                K, a->name, ggml_op_desc(a), ggml_type_name(a->type), (int) axera_gen_is_weight(a),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count());
    }
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

// binds a code predictor kernel's IO to the weights: the single-layer kernel to layer il, or (il < 0) the stacked one,
// each of its weight inputs and K / V outputs named <input>_<layer> (bf16 or MXFP8, converted once)
static void axera_gen_cp_bind_io(axera_gen & g, ggml_axera_kernel * k, const std::map<std::string, ggml_tensor *> & t,
                                 int il, axera_gen::layer & ly) {
    static const std::map<std::string, const char *> names = {
        { "ln1", "ln1.weight" },         { "ln2", "ln2.weight" },         { "q_norm", "attn_q_norm.weight" },
        { "k_norm", "attn_k_norm.weight" }, { "q", "attn_q.weight" },    { "k", "attn_k.weight" },
        { "v", "attn_v.weight" },        { "o", "attn_out.weight" },      { "gate", "ffn_gate.weight" },
        { "up", "ffn_up.weight" },       { "down", "ffn_down.weight" },
    };
    // an input name -> its name in the single-layer kernel, its layer and the suffix naming that layer
    auto split = [&](const std::string & n, std::string & base, int & l, std::string & sfx) {
        base = n, l = il, sfx = "";
        const size_t us = n.rfind('_');
        if (il < 0 && us != std::string::npos && us + 1 < n.size() &&
            n.find_first_not_of("0123456789", us + 1) == std::string::npos) {
            base = n.substr(0, us), l = atoi(n.c_str() + us + 1), sfx = n.substr(us);
        }
    };
    ly.in.assign(k->info->nInputSize, {});
    ly.out.assign(k->info->nOutputSize, {});
    ly.own.clear();
    auto get = [&](const std::string & n, int l) -> const ggml_tensor * {
        auto it = names.find(n);
        const ggml_tensor * w = it == names.end() || l < 0 ? nullptr :
            axera_graph_find(t, ("a.gen.code.blk." + std::to_string(l) + "." + it->second).c_str());
        if (!w) {
            GGML_ABORT("axera gen: code predictor layer %d has no weight for kernel input %s\n", l, n.c_str());
        }
        return w;
    };
    std::map<std::string, std::pair<ggml_backend_axera_buffer_context *, ggml_backend_axera_buffer_context *>> mx;
    const std::map<std::string, axera_cmm *> fixed = {
        { "x", &g.X }, { "pos", &g.POS }, { "nblk", &g.NBLK }, { "mask", &g.MASK }, { "cos", &g.COS }, { "sin", &g.SIN },
    };
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        auto              f    = fixed.find(n);
        std::string       base, sfx;
        int               l;
        split(n, base, l, sfx);
        if (f != fixed.end()) {
            ly.in[i] = axera_io(f->second->get(), (*f->second)->vir, size);
        } else if ((base.rfind("w_", 0) == 0 || base.rfind("s_", 0) == 0) &&
                   k->input(("s_" + base.substr(2) + sfx).c_str()) >= 0) {
            const std::string key = base.substr(2) + sfx;
            if (!mx.count(key)) {
                axera_cmm q, sc;
                axera_quantize_mxfp8(get(base.substr(2), l), q, sc);
                mx[key] = { q.get(), sc.get() };
                ly.own.push_back(std::move(q));
                ly.own.push_back(std::move(sc));
            }
            auto * b = base[0] == 'w' ? mx[key].first : mx[key].second;
            GGML_ASSERT(b->size >= size);
            ly.in[i] = axera_io(b, b->vir, size);
        } else {  // a bf16 weight (w_<x> of the bf16 kernel) or an f32 norm weight
            const ggml_tensor * w  = get(base.rfind("w_", 0) == 0 ? base.substr(2) : base, l);
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

// picks the code predictor's kernel once (the stacked one for n_layer layers of at least `positions` cells where
// stacking is possible and not off, GGML_AXERA_CP_STACK=0, else the single-layer one; the run's weight format first)
// and binds it to the weights
// GGML_AXERA_CP_VERIFY=1: the stacked kernel's result of each call compared with the single-layer kernel's (debug)
static bool axera_gen_cp_verify() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_CP_VERIFY");
        return e && atoi(e) != 0;
    }();
    return on;
}

static bool axera_gen_cp_bind(axera_gen & g, const std::map<std::string, ggml_tensor *> & t, int n_layer, int H,
                              int positions, bool stack) {
    const ggml_tensor * w0 = axera_graph_find(t, "a.gen.code.blk.0.ln1.weight");
    if (!w0) {
        return false;
    }
    if (g.cps && !stack) {
        return false;  // (a graph of other calls after a stacked one: not expected)
    }
    if ((g.cps || (g.cp && (int) g.cp_layers.size() == n_layer)) && g.cp_bound == w0->data) {
        return true;
    }
    if (!g.cp && !g.cps) {
        // MXFP8 with GGML_AXERA_WEIGHTS=mxfp8: about 17% faster per layer call, the generations end as with bf16
        const char * e = std::getenv("GGML_AXERA_CP_STACK");
        stack          = stack && !(e && strcmp(e, "0") == 0);
        for (bool mx8 : { axera_weights_mx8(), !axera_weights_mx8() }) {
            const std::string sn = "qwen3tts_cp_stack" + std::to_string(n_layer) + (mx8 ? "_mx8" : "");
            const int64_t     sc = stack ? axera_kernels().ctx_for(sn, positions) : 0;
            const std::string ln = mx8 ? "qwen3tts_cp_attn_mx8_kvc_c4096" : "qwen3tts_cp_attn_kvc_c4096";
            if (sc > 0) {
                g.cps = axera_kernels().get(sn + "_c" + std::to_string(sc));
                if (axera_gen_cp_verify() && axera_kernels().has(ln)) {
                    g.cp = axera_kernels().get(ln);
                }
                break;
            }
            if (axera_kernels().has(ln)) {
                g.cp = axera_kernels().get(ln);
                break;
            }
        }
        ggml_axera_kernel * k = g.cps ? g.cps : g.cp;
        if (!k) {
            return false;
        }
        axera_q35_context(k, g.cp_LMAX, g.cp_BC);
        const int ix = k->input("x"), ic = k->input("cos");
        if (ix < 0 || ic < 0 || k->info->pInputs[ix].nSize != (uint32_t) H * sizeof(ggml_bf16_t)) {
            g.cp = g.cps = nullptr;
            return false;
        }
        g.cp_H   = H;
        g.cp_ROT = (int) (k->info->pInputs[ic].nSize / sizeof(ggml_bf16_t)) * 2;
        g.X      = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        g.Y      = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        g.POS    = axera_cmm_alloc(sizeof(int32_t));
        g.NBLK   = axera_cmm_alloc(sizeof(int32_t));
        int mask_cells = g.cp_LMAX;
        if (g.cps && g.cp) {  // (verifying: the single-layer kernel's mask too)
            int lmax, bc;
            axera_q35_context(g.cp, lmax, bc);
            mask_cells = std::max(mask_cells, lmax);
        }
        g.MASK   = axera_cmm_alloc(mask_cells * sizeof(ggml_bf16_t));
        g.COS    = axera_cmm_alloc(g.cp_ROT / 2 * sizeof(ggml_bf16_t));
        g.SIN    = axera_cmm_alloc(g.cp_ROT / 2 * sizeof(ggml_bf16_t));
    }
    if (g.cps) {
        axera_gen_cp_bind_io(g, g.cps, t, -1, g.cps_io);
    }
    if (g.cp) {
        g.cp_layers.clear();
        g.cp_layers.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            axera_gen_cp_bind_io(g, g.cp, t, il, g.cp_layers[il]);
        }
    }
    g.cp_bound = w0->data;
    return true;
}

// one code predictor call on the NPU: out = layer il (x) at position pos, or with the stacked kernel all layers (il 0)
static void axera_gen_cp_run(axera_gen & g, const axera_gen_cp_call & c, float rope_theta) {
    const int H = g.cp_H;
    if (c.il == 0) {  // a new position: its mask cell, its RoPE rows (position 0: a new frame, every cell masked)
        auto * mask = (ggml_bf16_t *) g.MASK->vir;
        if (c.pos == 0) {
            std::fill(mask, mask + g.MASK->size / sizeof(ggml_bf16_t), ggml_fp32_to_bf16(-3e38f));
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
    auto & ly = g.cps ? g.cps_io : g.cp_layers[c.il];
    if (axera_run_sync(g.cps ? g.cps : g.cp, ly.in, ly.out) != 0) {
        GGML_ABORT("axera gen: code predictor layer %d failed\n", c.il);
    }
    if (g.cps && g.cp) {  // verifying: the same position through the single-layer kernel, layer by layer
        static int n_checked = 0;
        g.Y->invalidate(g.Y->vir, H * sizeof(ggml_bf16_t));
        std::vector<ggml_bf16_t> st((ggml_bf16_t *) g.Y->vir, (ggml_bf16_t *) g.Y->vir + H);
        for (size_t il = 0; il < g.cp_layers.size(); ++il) {
            if (il > 0) {
                memcpy(g.X->vir, g.Y->vir, H * sizeof(ggml_bf16_t));
                g.X->flush(g.X->vir, H * sizeof(ggml_bf16_t));
            }
            if (axera_run_sync(g.cp, g.cp_layers[il].in, g.cp_layers[il].out) != 0) {
                GGML_ABORT("axera gen: code predictor layer %zu failed\n", il);
            }
            g.Y->invalidate(g.Y->vir, H * sizeof(ggml_bf16_t));
        }
        double dot = 0, na = 0, nb = 0, mx = 0;
        for (int i = 0; i < H; ++i) {
            const double a = ggml_bf16_to_fp32(st[i]), b = ggml_bf16_to_fp32(((ggml_bf16_t *) g.Y->vir)[i]);
            dot += a * b, na += a * a, nb += b * b, mx = std::max(mx, fabs(a - b));
        }
        if (n_checked++ < 48) {
            fprintf(stderr, "axera gen: cp verify pos %d: stacked vs layers cos %.6f max diff %g (|stacked| %g |layers| %g)\n",
                    c.pos, dot / sqrt(na * nb + 1e-30), mx, sqrt(na), sqrt(nb));
        }
        memcpy(g.Y->vir, st.data(), H * sizeof(ggml_bf16_t));  // (the stacked result goes on)
        g.Y->flush(g.Y->vir, H * sizeof(ggml_bf16_t));
    }
    if (c.out && !g.cps) {  // (stacked: written when the graph reaches the output node)
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
    // stacked: the calls by the node they run at (layer 0's output: its input is still there, the graph's allocator
    // may reuse its memory later) and by the node their result is written into (the last layer's output, if any)
    std::map<const ggml_tensor *, const axera_gen_cp_call *> stack_at, stack_out;
    float                                                    theta = 0.0f;
    // one call per position with the step kernel (axera_gen::cpx): the step's embedding row (the projection's
    // input), its logits node (the head's MUL_MAT, nullptr at position 0) and head weight; by the node it runs at
    struct step {
        const ggml_tensor * e = nullptr, *head_w = nullptr;
        ggml_tensor *       logits = nullptr;
        int                 pos = 0;
    };
    std::vector<step>                                 steps;
    std::map<const ggml_tensor *, const step *>       step_at;
    std::map<const ggml_tensor *, axera_gen_conv_epi> epi;  // the convolutions' bias / residual ADDs, by MUL_MAT
    std::map<const ggml_tensor *, axera_gen_snake_fuse> sfuse;  // snakes fused into a conv input, by their first node
    // the whole frame on the frame kernel (on: none of the graph's nodes runs): the inputs mtmd sets (the talker's
    // hidden state, code 0, the draws), the weights the graph reads (heads, embedding tables, code 0's table), the
    // outputs it reads back (the codes, the next frame's embedding: the graph's last node)
    struct frame_io {
        bool                       on = false;
        int                        top_k = 0, n_codes = 0;
        const ggml_tensor *        h = nullptr, *code0 = nullptr, *head_w = nullptr, *embd_w = nullptr, *out_embd_w = nullptr;
        std::vector<const ggml_tensor *> rnd;
        ggml_tensor *              out_codes = nullptr, *out_embd = nullptr;
    } frame;
};

// whether the layer calls can be one call per position: each position's layers chained (layer il reads layer il - 1's
// output) and those outputs read by nothing but the next layer's skipped nodes; their outputs then marked in mids
static bool axera_gen_cp_stackable(struct ggml_cgraph * cgraph, const std::vector<axera_gen_cp_call> & calls,
                                   const std::set<const ggml_tensor *> & skip, int n_layer, int positions,
                                   std::set<const ggml_tensor *> & mids) {
    std::map<std::pair<int, int>, const axera_gen_cp_call *> at;
    for (const auto & c : calls) {
        at[{ c.pos, c.il }] = &c;
    }
    if (n_layer < 2 || (int) calls.size() != n_layer * positions || (int) at.size() != n_layer * positions) {
        return false;
    }
    for (int p = 0; p < positions; ++p) {
        for (int il = 1; il < n_layer; ++il) {
            const axera_gen_cp_call * prev = at.at({ p, il - 1 });
            if (!prev->out || at.at({ p, il })->x != prev->out) {
                return false;
            }
            mids.insert(prev->out);
        }
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || skip.count(n) || mids.count(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            if (mids.count(n->src[s]) || (n->src[s]->view_src && mids.count(n->src[s]->view_src))) {
                return false;
            }
        }
    }
    return true;
}

// GGML_AXERA_CP_STEP=0: the predictor's positions through the stacked layers kernel and the GEMVs, as before
static bool axera_gen_step_on() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_CP_STEP");
        return !(e && strcmp(e, "0") == 0);
    }();
    return on;
}

// the stacked positions of a graph as step kernel calls: per position its embedding row (the input projection's
// input, or layer 0's input without a projection) and, but at position 0, its codebook head (MUL_MAT of final_norm(the
// last layer's output)); the projection, the final norm and the head nodes added to skip. Empty (skip unchanged) unless
// every position has that shape, with one projection and one norm weight
static std::vector<axera_gen_analysis::step> axera_gen_cp_steps(struct ggml_cgraph * cgraph,
                                                                const std::vector<axera_gen_cp_call> & st,
                                                                std::set<const ggml_tensor *> & skip,
                                                                const ggml_tensor ** proj_w, const ggml_tensor ** proj_b,
                                                                const ggml_tensor ** norm_w) {
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            users[n->src[s]].push_back(n);
        }
    }
    auto only_user = [&](const ggml_tensor * t, ggml_op op) -> ggml_tensor * {
        auto it = users.find(t);
        return it != users.end() && it->second.size() == 1 && it->second[0]->op == op ? it->second[0] : nullptr;
    };
    std::vector<axera_gen_analysis::step> steps;
    std::set<const ggml_tensor *>         add;
    *proj_w = *proj_b = *norm_w = nullptr;
    for (size_t i = 0; i < st.size(); ++i) {
        const axera_gen_cp_call & c = st[i];
        axera_gen_analysis::step  s;
        s.pos                         = c.pos;
        const ggml_tensor * x         = c.x;
        const ggml_tensor * mm        = x->op == GGML_OP_ADD && x->src[0]->op == GGML_OP_MUL_MAT ? x->src[0] :
                                        x->op == GGML_OP_MUL_MAT ? x : nullptr;
        const bool          proj      = mm && axera_gen_is_weight(mm->src[0]) && mm->src[1]->ne[1] == 1 &&
                                        mm->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(mm->src[1]);
        const ggml_tensor * pb        = proj && x != mm ? x->src[1] : nullptr;
        if (i == 0) {
            *proj_w = proj ? mm->src[0] : nullptr;
            *proj_b = pb;
        } else if ((proj ? mm->src[0] : nullptr) != *proj_w || pb != *proj_b) {
            return {};
        }
        if (pb && (pb->type != GGML_TYPE_F32 || !axera_gen_is_weight(pb))) {
            return {};
        }
        s.e = proj ? mm->src[1] : x;
        if (proj) {
            add.insert(mm);
            add.insert(x);
        }
        if (c.out) {
            ggml_tensor * r = only_user(c.out, GGML_OP_RMS_NORM);
            ggml_tensor * m = r ? only_user(r, GGML_OP_MUL) : nullptr;
            ggml_tensor * h = m && m->src[0] == r ? only_user(m, GGML_OP_MUL_MAT) : nullptr;
            if (!h || h->src[1] != m || h->type != GGML_TYPE_F32 || !ggml_is_contiguous(h) || h->ne[1] != 1 ||
                (*norm_w && m->src[1] != *norm_w)) {
                return {};
            }
            *norm_w   = m->src[1];
            s.logits  = h;
            s.head_w  = h->src[0];
            add.insert(c.out);
            add.insert(r);
            add.insert(m);
            add.insert(h);
        }
        steps.push_back(s);
    }
    if (!*norm_w || (*norm_w)->type != GGML_TYPE_F32) {
        return {};
    }
    skip.insert(add.begin(), add.end());
    return steps;
}

// a member of a code predictor kernel's .layout: lines "<pack> - 0 <bytes>" (a pack input and its size) or "<member>
// <pack> <offset> <bytes> [blk <rows> <cols> <bw>]" (a weight's bytes in its pack, stored in column strips of bw)
struct axera_gen_member {
    std::string         name, pack;
    size_t              off = 0, bytes = 0;
    int64_t             rows = 0, cols = 0, bw = 0;
    const ggml_tensor * w    = nullptr;
    int                 kind = 0;  // 1 MXFP8 elements, 2 MXFP8 scales, 3 bf16, 4 f32
};

// reads the .layout of kernel name and finds every member's tensor (the layers' by their GGUF names, LP.w_pin / b_pin
// / norm the projection and the final norm): false, with a warning, if there is no layout or a member has no tensor of
// its size or does not fit its pack
static bool axera_gen_layout_read(const std::string & name, const std::map<std::string, ggml_tensor *> & t,
                                  const ggml_tensor * proj_w, const ggml_tensor * proj_b, const ggml_tensor * norm_w,
                                  std::vector<axera_gen_member> & members, std::map<std::string, size_t> & psize) {
    const std::string path = axera_kernels().path(name);
    std::ifstream     f(path.substr(0, path.size() - strlen(".axmodel")) + ".layout");
    if (!f.good()) {
        GGML_LOG_WARN("axera gen: %s has no layout file, not used\n", name.c_str());
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        axera_gen_member   m;
        std::string        blk;
        if (!(ls >> m.name >> m.pack >> m.off >> m.bytes)) {
            continue;
        }
        if (ls >> blk && blk == "blk") {
            ls >> m.rows >> m.cols >> m.bw;
        }
        if (m.pack == "-") {
            psize[m.name] = m.bytes;
        } else {
            members.push_back(m);
        }
    }
    static const std::map<std::string, const char *> names = {
        { "ln1", "ln1.weight" },         { "ln2", "ln2.weight" },         { "q_norm", "attn_q_norm.weight" },
        { "k_norm", "attn_k_norm.weight" }, { "q", "attn_q.weight" },    { "k", "attn_k.weight" },
        { "v", "attn_v.weight" },        { "o", "attn_out.weight" },      { "gate", "ffn_gate.weight" },
        { "up", "ffn_up.weight" },       { "down", "ffn_down.weight" },
    };
    auto unfit = [&](const char * why, const std::string & what) {
        GGML_LOG_WARN("axera gen: %s does not fit the model (%s %s), not used\n", name.c_str(), why, what.c_str());
        return false;
    };
    for (auto & m : members) {
        const size_t      dot = m.name.find('.');
        const std::string lj = m.name.substr(0, dot), par = dot == std::string::npos ? "" : m.name.substr(dot + 1);
        if (lj == "LP") {
            m.w = par == "w_pin" ? proj_w : par == "b_pin" ? proj_b : par == "norm" ? norm_w : nullptr;
        } else if (lj.size() > 1 && lj[0] == 'L') {
            const std::string wn = par.rfind("w_", 0) == 0 || par.rfind("s_", 0) == 0 ? par.substr(2) : par;
            auto              it = names.find(wn);
            m.w = it == names.end() ? nullptr :
                  axera_graph_find(t, ("a.gen.code.blk." + lj.substr(1) + "." + it->second).c_str());
        }
        if (!m.w) {
            return unfit("no tensor for member", m.name);
        }
        const int64_t ne = ggml_nelements(m.w);
        size_t        want;
        if ((par.rfind("w_", 0) == 0 || par.rfind("s_", 0) == 0) && lj != "LP") {
            m.kind = par[0] == 'w' ? 1 : 2;
            want   = m.w->ne[0] % AXERA_MX_GROUP ? 0 : m.kind == 1 ? (size_t) ne : (size_t) ne / AXERA_MX_GROUP;
        } else if (m.pack == "pb") {
            m.kind = 3;
            want   = (size_t) ne * sizeof(ggml_bf16_t);
        } else {
            m.kind = 4;
            want   = m.w->type == GGML_TYPE_F32 ? ggml_nbytes(m.w) : 0;
        }
        if (want == 0 || m.bytes != want) {
            return unfit("size of member", m.name);
        }
        if (!psize.count(m.pack) || m.off + m.bytes > psize[m.pack]) {
            return unfit("pack of member", m.name);
        }
        if (m.bw && (m.rows <= 0 || m.cols <= 0 || m.cols % m.bw || m.bytes % (size_t) (m.rows * m.cols))) {
            return unfit("strips of member", m.name);
        }
    }
    return true;
}

// whether every input of kernel k but the named ones is a pack of psize, of at least its size
static bool axera_gen_layout_inputs(const std::string & name, const ggml_axera_kernel * k,
                                    const std::map<std::string, size_t> & psize, const std::set<std::string> & others) {
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n  = k->info->pInputs[i].pName;
        auto              it = psize.find(n);
        if (!others.count(n) && (it == psize.end() || it->second < k->info->pInputs[i].nSize)) {
            GGML_LOG_WARN("axera gen: %s does not fit the model (input %s without its pack), not used\n", name.c_str(),
                          n.c_str());
            return false;
        }
    }
    return true;
}

// the packs allocated into io.own and filled once from the members' tensors (MXFP8 layer weights, the projection as
// bf16, the bias and norms as f32), by pack name
static std::map<std::string, ggml_backend_axera_buffer_context *> axera_gen_layout_fill(
        const std::vector<axera_gen_member> & members, const std::map<std::string, size_t> & psize, axera_gen::layer & io) {
    std::map<std::string, ggml_backend_axera_buffer_context *> packs;
    for (const auto & p : psize) {
        io.own.push_back(axera_cmm_alloc(std::max<size_t>(p.second, 64)));
        packs[p.first] = io.own.back().get();
        memset(packs[p.first]->vir, 0, packs[p.first]->size);
    }
    std::map<const ggml_tensor *, std::pair<axera_cmm, axera_cmm>> mx;  // MXFP8 copies, by weight
    for (const auto & m : members) {
        std::vector<char> src;
        if (m.kind <= 2) {  // MXFP8 elements (one byte each) or their scales (one per 32)
            auto & q = mx[m.w];
            if (!q.first) {
                axera_quantize_mxfp8(m.w, q.first, q.second);
            }
            const auto & b = m.kind == 1 ? q.first : q.second;
            src.assign((const char *) b->vir, (const char *) b->vir + m.bytes);
        } else if (m.kind == 3) {
            src.resize(m.bytes);
            axera_gen_to_bf16(m.w, ggml_nrows(m.w), (ggml_bf16_t *) src.data());
        } else {
            src.assign((const char *) m.w->data, (const char *) m.w->data + m.bytes);
        }
        char * dst = (char *) packs[m.pack]->vir + m.off;
        if (m.bw == 0) {
            memcpy(dst, src.data(), m.bytes);
        } else {  // [rows][cols] -> strip j = columns j * bw .. (j + 1) * bw of every row, one strip after another
            const size_t es = m.bytes / (size_t) (m.rows * m.cols);
            for (int64_t j = 0; j < m.cols / m.bw; ++j) {
                for (int64_t r = 0; r < m.rows; ++r) {
                    memcpy(dst + ((j * m.rows + r) * m.bw) * es, src.data() + (r * m.cols + j * m.bw) * es, m.bw * es);
                }
            }
        }
    }
    for (auto & p : packs) {
        p.second->flush(p.second->vir, p.second->size);
    }
    return packs;
}

// binds the step kernel (qwen3tts_cp_step<n_layer>_mx8_h<HT>_c<LMAX> and its .layout) for HT-wide embedding rows: its
// packs filled once from the predictor's weights, its K / V resident. False, nothing allocated, if there is no such
// kernel or it does not fit the model (axera_gen_layout_read, a pack too small, or the kernel's e / w_head / logits
// not of the model's sizes): the stacked kernel then runs the positions
static bool axera_gen_step_bind(axera_gen & g, const std::map<std::string, ggml_tensor *> & t, int n_layer,
                                int64_t HT, int positions, const ggml_tensor * proj_w, const ggml_tensor * proj_b,
                                const ggml_tensor * norm_w, const ggml_tensor * head_w, size_t logits_bytes) {
    if (g.cpx && g.cpx_bound == proj_w && g.cpx_norm == norm_w && g.E && g.E->size == (size_t) HT * sizeof(ggml_bf16_t)) {
        return true;  // bound to these weights already (the graph built again)
    }
    const std::string sn = "qwen3tts_cp_step" + std::to_string(n_layer) + "_mx8_h" + std::to_string(HT);
    const int64_t     sc = axera_weights_mx8() ? axera_kernels().ctx_for(sn, positions) : 0;
    if (sc == 0) {
        return false;
    }
    const std::string                name = sn + "_c" + std::to_string(sc);
    std::vector<axera_gen_member>    members;
    std::map<std::string, size_t>    psize;
    if (!axera_gen_layout_read(name, t, proj_w, proj_b, norm_w, members, psize)) {
        return false;
    }
    ggml_axera_kernel * k = axera_kernels().get(name);
    const int ie = k->input("e"), ih = k->input("w_head"), ics = k->input("cs"), ipn = k->input("pn"), im = k->input("mask");
    const int il = k->output("logits");
    if (ie < 0 || ih < 0 || ics < 0 || ipn < 0 || im < 0 || il < 0 ||
        k->info->pInputs[ie].nSize != HT * sizeof(ggml_bf16_t) ||
        k->info->pInputs[ih].nSize != (size_t) (head_w->ne[0] * head_w->ne[1]) * sizeof(ggml_bf16_t) ||
        k->info->pOutputs[il].nSize != logits_bytes) {
        GGML_LOG_WARN("axera gen: %s does not fit the model (e / w_head / cs / pn / mask / logits), not used\n",
                      name.c_str());
        return false;
    }
    if (!axera_gen_layout_inputs(name, k, psize, { "e", "w_head", "cs", "pn", "mask" })) {
        return false;
    }
    g.cpx_io   = axera_gen::layer();
    auto packs = axera_gen_layout_fill(members, psize, g.cpx_io);
    // the step kernel's own mask, RoPE rows and context (the stacked path's g.MASK / cp_* stay as they are)
    int lmax, bc;
    axera_q35_context(k, lmax, bc);
    g.cpx_LMAX = lmax, g.cpx_BC = bc;
    g.cpx_ROT  = (int) (k->info->pInputs[ics].nSize / sizeof(ggml_bf16_t));  // cos | sin: ROT / 2 each
    g.E        = axera_cmm_alloc(HT * sizeof(ggml_bf16_t));
    g.CS       = axera_cmm_alloc(g.cpx_ROT * sizeof(ggml_bf16_t));
    g.PN       = axera_cmm_alloc(2 * sizeof(int32_t));
    g.MASKX    = axera_cmm_alloc(std::max<size_t>(k->info->pInputs[im].nSize, lmax * sizeof(ggml_bf16_t)));
    g.cpx_io.in.assign(k->info->nInputSize, {});
    g.cpx_io.out.assign(k->info->nOutputSize, {});
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        if (n == "w_head") {
            continue;  // bound per call
        }
        ggml_backend_axera_buffer_context * b = n == "e" ? g.E.get() : n == "cs" ? g.CS.get() : n == "pn" ? g.PN.get() :
                                                n == "mask" ? g.MASKX.get() : packs[n];
        g.cpx_io.in[i] = axera_io(b, b->vir, size);
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const uint32_t size = k->info->pOutputs[i].nSize;
        g.cpx_io.own.push_back(axera_cmm_alloc(size));  // logits, or a layer's resident K / V
        auto * b = g.cpx_io.own.back().get();
        memset(b->vir, 0, size);
        b->flush(b->vir, size);
        if ((int) i == il) {
            g.cpx_lg = b;
        }
        g.cpx_io.out[i] = axera_io(b, b->vir, size);
    }
    g.cpx_e      = ie;
    g.cpx_head   = ih;
    g.cpx_logits = il;
    g.cpx        = k;
    g.cpx_bound  = proj_w;
    g.cpx_norm   = norm_w;
    return true;
}

// a whole frame of the code predictor in one call where its frame kernel exists and fits; GGML_AXERA_CP_FRAME=0 off
static bool axera_gen_frame_on() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_CP_FRAME");
        return !(e && strcmp(e, "0") == 0);
    }();
    return on;
}

// binds the frame kernel (qwen3tts_cp_frame<n_layer>_mx8_h<HT>_k<top_k>_c<LMAX>, q3tts_cp_frame.py, the step kernel's
// .layout) to the graph's frame: its packs as the step kernel's, the codebook heads and embedding tables as bf16, the
// 16 positions' RoPE rows, pos | nblk and mask rows. False, nothing allocated, if there is none or it does not fit
static bool axera_gen_frame_bind(axera_gen & g, const std::map<std::string, ggml_tensor *> & t, int n_layer,
                                 const axera_gen_analysis::frame_io & fr, const ggml_tensor * proj_w, const ggml_tensor * proj_b,
                                 const ggml_tensor * norm_w, float theta) {
    const int64_t HT = fr.h->ne[0], NP = fr.n_codes;
    if (g.cpf && g.cpf_bound == proj_w && g.cpf_topk == fr.top_k && g.F_E0 && g.F_E0->size == (size_t) HT * sizeof(ggml_bf16_t)) {
        return true;
    }
    const std::string sn = "qwen3tts_cp_frame" + std::to_string(n_layer) + "_mx8_h" + std::to_string(HT) + "_k" +
                           std::to_string(fr.top_k);
    const int64_t sc = axera_weights_mx8() ? axera_kernels().ctx_for(sn, NP) : 0;
    if (sc == 0) {
        return false;
    }
    const std::string             name = sn + "_c" + std::to_string(sc);
    std::vector<axera_gen_member> members;
    std::map<std::string, size_t> psize;
    if (!axera_gen_layout_read(name, t, proj_w, proj_b, norm_w, members, psize)) {
        return false;
    }
    ggml_axera_kernel * k = axera_kernels().get(name);
    const std::set<std::string> others = { "e0", "e1", "heads", "embd", "cs16", "pn16", "mask16", "rnd" };
    auto in_size = [&](const char * n) { const int i = k->input(n); return i < 0 ? (size_t) 0 : (size_t) k->info->pInputs[i].nSize; };
    const int    ic = k->output("codes"), il = k->output("lgs");
    const size_t V  = (size_t) fr.head_w->ne[1];
    if (ic < 0 || il < 0 || in_size("e0") != HT * sizeof(ggml_bf16_t) || in_size("e1") != HT * sizeof(ggml_bf16_t) ||
        in_size("heads") != (size_t) ggml_nelements(fr.head_w) * sizeof(ggml_bf16_t) ||
        in_size("embd") != (size_t) ggml_nelements(fr.embd_w) * sizeof(ggml_bf16_t) ||
        in_size("rnd") != (size_t) (NP - 1) * sizeof(float) || in_size("pn16") != (size_t) NP * 2 * sizeof(int32_t) ||
        k->info->pOutputs[il].nSize != (size_t) (NP - 1) * V * sizeof(float) ||
        k->info->pOutputs[ic].nSize % (NP * sizeof(int32_t)) != 0 || !axera_gen_layout_inputs(name, k, psize, others)) {
        GGML_LOG_WARN("axera gen: %s does not fit the model (its inputs / outputs), not used\n", name.c_str());
        return false;
    }
    g.cpf_io   = axera_gen::layer();
    const int im = k->input("mask16");  // [NP * LMAX / BC, BC] bf16
    if (im < 0 || k->info->pInputs[im].nShapeSize != 2 || k->info->pInputs[im].pShape[0] % NP != 0) {
        GGML_LOG_WARN("axera gen: %s does not fit the model (mask16), not used\n", name.c_str());
        return false;
    }
    const int bc   = (int) k->info->pInputs[im].pShape[1];
    auto      packs = axera_gen_layout_fill(members, psize, g.cpf_io);
    const size_t cs_bytes = in_size("cs16"), mask_bytes = in_size("mask16");
    const int    rot      = (int) (cs_bytes / NP / sizeof(ggml_bf16_t));  // per position cos | sin, ROT / 2 each
    g.F_E0 = axera_cmm_alloc(HT * sizeof(ggml_bf16_t));
    g.F_E1 = axera_cmm_alloc(HT * sizeof(ggml_bf16_t));
    g.F_RND = axera_cmm_alloc((NP - 1) * sizeof(float));
    g.F_CS  = axera_cmm_alloc(cs_bytes);
    g.F_PN  = axera_cmm_alloc(NP * 2 * sizeof(int32_t));
    g.F_MASK = axera_cmm_alloc(mask_bytes);
    auto * cs = (ggml_bf16_t *) g.F_CS->vir;
    auto * pn = (int32_t *) g.F_PN->vir;
    auto * mk = (ggml_bf16_t *) g.F_MASK->vir;
    const int64_t cells = (int64_t) (mask_bytes / NP / sizeof(ggml_bf16_t));
    for (int64_t p = 0; p < NP; ++p) {  // (as the step kernel's per call)
        for (int i = 0; i < rot / 2; ++i) {
            const float a                 = (float) p * powf(theta, -2.0f * i / rot);
            cs[p * rot + i]               = ggml_fp32_to_bf16(cosf(a));
            cs[p * rot + rot / 2 + i]     = ggml_fp32_to_bf16(sinf(a));
        }
        pn[2 * p]     = (int32_t) p;
        pn[2 * p + 1] = (int32_t) (p / bc + 1);
        for (int64_t c = 0; c < cells; ++c) {
            mk[p * cells + c] = ggml_fp32_to_bf16(c <= p ? 0.0f : -3e38f);
        }
    }
    g.F_CS->flush(g.F_CS->vir, cs_bytes);
    g.F_PN->flush(g.F_PN->vir, g.F_PN->size);
    g.F_MASK->flush(g.F_MASK->vir, mask_bytes);
    const auto heads = axera_gen_bf16(g, fr.head_w, ggml_nrows(fr.head_w), g.tmp[0]);
    const auto embd  = axera_gen_bf16(g, fr.embd_w, ggml_nrows(fr.embd_w), g.tmp[0]);
    g.cpf_io.in.assign(k->info->nInputSize, {});
    g.cpf_io.out.assign(k->info->nOutputSize, {});
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        if (n == "heads" || n == "embd") {
            const auto & w = n == "heads" ? heads : embd;
            g.cpf_io.in[i] = axera_io(w.first, w.second, size);
            continue;
        }
        ggml_backend_axera_buffer_context * b = n == "e0" ? g.F_E0.get() : n == "e1" ? g.F_E1.get() :
                                                n == "rnd" ? g.F_RND.get() : n == "cs16" ? g.F_CS.get() :
                                                n == "pn16" ? g.F_PN.get() : n == "mask16" ? g.F_MASK.get() : packs[n];
        g.cpf_io.in[i] = axera_io(b, b->vir, size);
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const uint32_t size = k->info->pOutputs[i].nSize;
        g.cpf_io.own.push_back(axera_cmm_alloc(size));  // codes, logits, or a layer's resident K / V
        auto * b = g.cpf_io.own.back().get();
        memset(b->vir, 0, size);
        b->flush(b->vir, size);
        if ((int) i == ic) {
            g.cpf_codes = b;
        }
        if ((int) i == il) {
            g.cpf_lgs = b;
        }
        g.cpf_io.out[i] = axera_io(b, b->vir, size);
    }
    g.cpf       = k;
    g.cpf_bound = proj_w;
    g.cpf_topk  = fr.top_k;
    return true;
}

// GGML_AXERA_CP_FRAME_CHECK=1: each sampled code against a host sampling (double precision) of the kernel's own
// logits and draw, mtmd's order (the K largest, the first two swapped); positions where they differ or the draw lies
// within 1e-5 of a cumulative sum logged with that margin (a different code is then rounding at the boundary)
static void axera_gen_frame_check(axera_gen & g, const axera_gen_analysis::frame_io & fr, const int32_t * codes) {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_CP_FRAME_CHECK");
        return e && strcmp(e, "1") == 0;
    }();
    static int64_t frame = 0;
    ++frame;
    if (!on) {
        return;
    }
    const int64_t V = fr.head_w->ne[1], K = fr.top_k;
    g.cpf_lgs->invalidate(g.cpf_lgs->vir, g.cpf_lgs->size);
    for (int64_t p = 1; p < fr.n_codes; ++p) {
        const float *        lg = (const float *) g.cpf_lgs->vir + (p - 1) * V;
        const float          r  = *(const float *) fr.rnd[p - 1]->data;
        std::vector<int32_t> id(V);
        for (int64_t i = 0; i < V; ++i) {
            id[i] = (int32_t) i;
        }
        std::partial_sort(id.begin(), id.begin() + K, id.end(),
                          [&](int32_t a, int32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
        if (K > 1) {
            std::swap(id[0], id[1]);
        }
        double mx = lg[id[0]], sum = 0.0, cdf = 0.0, margin = 1.0;
        for (int64_t k = 0; k < K; ++k) {
            mx = std::max(mx, (double) lg[id[k]]);
        }
        for (int64_t k = 0; k < K; ++k) {
            sum += std::exp(lg[id[k]] - mx);
        }
        int32_t pick = -1;
        for (int64_t k = 0; k < K; ++k) {
            cdf += std::exp(lg[id[k]] - mx) / sum;
            margin = std::min(margin, std::fabs(cdf - r));
            if (pick < 0 && cdf > r) {
                pick = id[k];
            }
        }
        if (pick != codes[p] || margin < 1e-5) {
            GGML_LOG_WARN("axera gen: frame check: frame %" PRId64 " position %" PRId64 ": kernel code %d, host %d, draw "
                          "%.9f, nearest cumulative sum %.3g away\n", frame, p, codes[p], pick, r, margin);
        }
    }
}

// the frame on the frame kernel: the talker's hidden state and code 0's embedding row as bf16, the draws; then the
// graph's outputs as its nodes would have computed them: the 16 codes, and the sum of their embeddings in f32. False
// (counted, with a warning; nothing written) if the cores sampled different codes: the graph then runs this frame on
// the step kernel
static bool axera_gen_frame_run(axera_gen & g, const axera_gen_analysis::frame_io & fr) {
    const int64_t HT = fr.h->ne[0], NP = fr.n_codes;
    const int32_t code0 = *(const int32_t *) fr.code0->data;
    GGML_ASSERT(code0 >= 0 && code0 < fr.out_embd_w->ne[1]);
    std::vector<float> e1(HT), row(HT), sum(HT);
    const auto *       tro = ggml_get_type_traits(fr.out_embd_w->type);
    const char *       r0  = (const char *) fr.out_embd_w->data + code0 * fr.out_embd_w->nb[1];
    if (fr.out_embd_w->type == GGML_TYPE_F32) {
        memcpy(e1.data(), r0, HT * sizeof(float));
    } else {
        tro->to_float(r0, e1.data(), HT);
    }
    ggml_fp32_to_bf16_row((const float *) fr.h->data, (ggml_bf16_t *) g.F_E0->vir, HT);
    ggml_fp32_to_bf16_row(e1.data(), (ggml_bf16_t *) g.F_E1->vir, HT);
    for (int64_t p = 0; p < NP - 1; ++p) {
        ((float *) g.F_RND->vir)[p] = *(const float *) fr.rnd[p]->data;
    }
    g.F_E0->flush(g.F_E0->vir, g.F_E0->size);
    g.F_E1->flush(g.F_E1->vir, g.F_E1->size);
    g.F_RND->flush(g.F_RND->vir, g.F_RND->size);
    {
        axera_timer tm("gen code predictor frame (npu)");
        if (axera_run_sync(g.cpf, g.cpf_io.in, g.cpf_io.out) != 0) {
            GGML_ABORT("axera gen: code predictor frame failed\n");
        }
    }
    g.cpf_codes->invalidate(g.cpf_codes->vir, g.cpf_codes->size);
    const auto * cr    = (const int32_t *) g.cpf_codes->vir;
    const size_t cores = g.cpf_codes->size / (NP * sizeof(int32_t));
    for (size_t c = 1; c < cores; ++c) {
        if (memcmp(cr, cr + c * NP, NP * sizeof(int32_t)) != 0) {
            static int n_diff = 0;
            GGML_LOG_WARN("axera gen: the frame kernel's cores sampled different codes (%d times), this frame on the step "
                          "kernel\n", ++n_diff);
            return false;
        }
    }
    std::vector<int32_t> codes(NP);
    codes[0] = code0;
    for (int64_t p = 1; p < NP; ++p) {
        if (cr[p] < 0 || cr[p] >= fr.embd_w->ne[1]) {
            static int n_bad = 0;
            GGML_LOG_WARN("axera gen: the frame kernel sampled code %d out of range (%d times), this frame on the step "
                          "kernel\n", cr[p], ++n_bad);
            return false;
        }
        codes[p] = cr[p];
    }
    axera_gen_frame_check(g, fr, codes.data());
    GGML_ASSERT(ggml_nbytes(fr.out_codes) == NP * sizeof(int32_t));
    memcpy(fr.out_codes->data, codes.data(), NP * sizeof(int32_t));
    // the next frame's embedding: code 0's row plus each code's row of its table, added in the graph's order
    const auto * tre = ggml_get_type_traits(fr.embd_w->type);
    sum = e1;
    for (int64_t gi = 1; gi < NP; ++gi) {
        const char * src = (const char *) fr.embd_w->data + (gi - 1) * fr.embd_w->nb[2] + codes[gi] * fr.embd_w->nb[1];
        if (fr.embd_w->type == GGML_TYPE_F32) {
            memcpy(row.data(), src, HT * sizeof(float));
        } else {
            tre->to_float(src, row.data(), HT);
        }
        for (int64_t i = 0; i < HT; ++i) {
            sum[i] += row[i];
        }
    }
    GGML_ASSERT(ggml_nbytes(fr.out_embd) == HT * sizeof(float));
    memcpy(fr.out_embd->data, sum.data(), HT * sizeof(float));
    for (const ggml_tensor * o : { fr.out_codes, fr.out_embd }) {
        if (ggml_backend_buffer_is_axera(o->buffer)) {
            ((ggml_backend_axera_buffer_context *) o->buffer->context)->flush(o->data, ggml_nbytes(o));
        }
    }
    return true;
}

// one predictor position on the step kernel: the position's mask cell and RoPE rows (position 0: a new frame, every
// cell masked), its embedding row as bf16, the step's head; its logits into the head node
static void axera_gen_step_run(ggml_backend_axera_context * ctx, axera_gen & g, const axera_gen_analysis & an,
                               const axera_gen_analysis::step & s) {
    auto * mask = (ggml_bf16_t *) g.MASKX->vir;
    if (s.pos == 0) {
        std::fill(mask, mask + g.MASKX->size / sizeof(ggml_bf16_t), ggml_fp32_to_bf16(-3e38f));
    }
    mask[s.pos] = ggml_fp32_to_bf16(0.0f);
    g.MASKX->flush(g.MASKX->vir, (size_t) g.cpx_BC * (s.pos / g.cpx_BC + 1) * sizeof(ggml_bf16_t));
    ((int32_t *) g.PN->vir)[0] = s.pos;
    ((int32_t *) g.PN->vir)[1] = s.pos / g.cpx_BC + 1;
    g.PN->flush(g.PN->vir, g.PN->size);
    const int half = g.cpx_ROT / 2;
    for (int i = 0; i < half; ++i) {
        const float a                           = (float) s.pos * powf(an.theta, -2.0f * i / g.cpx_ROT);
        ((ggml_bf16_t *) g.CS->vir)[i]          = ggml_fp32_to_bf16(cosf(a));
        ((ggml_bf16_t *) g.CS->vir)[half + i]   = ggml_fp32_to_bf16(sinf(a));
    }
    g.CS->flush(g.CS->vir, g.CS->size);
    const int64_t HT = s.e->ne[0];
    ggml_fp32_to_bf16_row((const float *) s.e->data, (ggml_bf16_t *) g.E->vir, HT);
    g.E->flush(g.E->vir, HT * sizeof(ggml_bf16_t));
    // the head: this step's (position 0's logits are not read: any head, the next position's)
    const ggml_tensor * hw = s.head_w;
    for (size_t i = 0; !hw && i < an.steps.size(); ++i) {
        hw = an.steps[i].head_w;
    }
    const auto w = axera_gen_bf16(g, hw, hw->ne[1], g.tmp[0]);
    g.cpx_io.in[g.cpx_head] = axera_io(w.first, w.second, (uint32_t) (hw->ne[0] * hw->ne[1] * sizeof(ggml_bf16_t)));
    if (axera_run_sync(g.cpx, g.cpx_io.in, g.cpx_io.out) != 0) {
        GGML_ABORT("axera gen: code predictor step at position %d failed\n", s.pos);
    }
    if (s.logits) {
        GGML_ASSERT(ggml_nbytes(s.logits) == g.cpx->info->pOutputs[g.cpx_logits].nSize);
        g.cpx_lg->invalidate(g.cpx_lg->vir, ggml_nbytes(s.logits));
        memcpy(s.logits->data, g.cpx_lg->vir, ggml_nbytes(s.logits));
        if (ggml_backend_buffer_is_axera(s.logits->buffer)) {
            ((ggml_backend_axera_buffer_context *) s.logits->buffer->context)->flush(s.logits->data, ggml_nbytes(s.logits));
        }
    }
    GGML_UNUSED(ctx);
}

// the epilogues of the graph's convolutions (MUL_MAT of a 1-D IM2COL): ggml_conv_1d's CONT of its permuted result (for
// one batch a copy in the same order), then through views the one ADD of a bias (an f32 weight viewed as [1, OC]),
// and if that ADD's only reader is an ADD of it and an f32 tensor of its shape, that one too (a residual)
static void axera_gen_conv_epis(struct ggml_cgraph * cgraph, std::map<const ggml_tensor *, axera_gen_conv_epi> & epi) {
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            users[n->src[s]].push_back(n);
        }
    }
    auto only = [&](const ggml_tensor * t) -> ggml_tensor * {
        auto it = users.find(t);
        return it != users.end() && it->second.size() == 1 ? it->second[0] : nullptr;
    };
    auto same = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
               ggml_nelements(a) == ggml_nelements(b) && a->ne[0] == b->ne[0];
    };
    auto in_cmm = [](const ggml_tensor * t) {
        return t->buffer && ggml_backend_buffer_is_axera(t->buffer);
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *       n  = cgraph->nodes[i];
        const ggml_tensor * s0 = n->op == GGML_OP_MUL_MAT ? n->src[0] : nullptr;  // the IM2COL, or a view of it
        if (!s0 || !(s0->op == GGML_OP_IM2COL || (s0->view_src && s0->view_src->op == GGML_OP_IM2COL)) ||
            n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n)) {
            continue;
        }
        // reshape [OL, 1, OC], permute to [OL, OC, 1], CONT
        const ggml_tensor * u = n;
        ggml_tensor *       c = only(u);
        while (c && (c->op == GGML_OP_RESHAPE || c->op == GGML_OP_VIEW || c->op == GGML_OP_PERMUTE)) {
            if (c->op == GGML_OP_PERMUTE) {
                const int32_t * ax = (const int32_t *) c->op_params;
                if (!(ax[0] == 0 && ax[1] == 2 && ax[2] == 1 && ax[3] == 3 && (u->ne[1] == 1 || u->ne[2] == 1))) {
                    c = nullptr;
                    break;
                }
            }
            u = c;
            c = only(u);
        }
        if (!c || c->op != GGML_OP_CONT || c->src[0] != u || !same(c, n) || !in_cmm(c)) {
            continue;
        }
        axera_gen_conv_epi e;
        e.cont                 = c;
        epi[n]                 = e;
        const ggml_tensor * ua = c;
        ggml_tensor *       a  = only(c);
        while (a && (a->op == GGML_OP_RESHAPE || a->op == GGML_OP_VIEW)) {
            ua = a;
            a  = only(a);
        }
        if (!a || a->op != GGML_OP_ADD || a->src[0] != ua || !same(a, n) || !in_cmm(a)) {
            continue;
        }
        const ggml_tensor * b = a->src[1];
        const ggml_tensor * w = b;
        while (w->view_src) {
            w = w->view_src;
        }
        if (b->ne[0] != 1 || b->ne[1] != n->ne[1] || ggml_nelements(b) != n->ne[1] || !axera_gen_is_weight(w) ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(b)) {
            continue;
        }
        e.add  = a;
        e.bias = b;
        ggml_tensor * r = only(a);
        if (r && r->op == GGML_OP_ADD && same(r, a) && in_cmm(r)) {
            const ggml_tensor * x = r->src[0] == a ? r->src[1] : r->src[0];
            if ((r->src[0] == a) != (r->src[1] == a) && same(x, a) && in_cmm(x)) {
                e.radd = r;
                e.res  = x;
            }
        }
        epi[n] = e;
    }
}

// the snakes whose output only a convolution reads (axera_gen_snake_fuse): by the snake's first node (MUL x * alpha;
// the snake's nodes as mtmd builds them: MUL, SIN, SQR, MUL beta, ADD x)
static void axera_gen_snake_fuses(struct ggml_cgraph * cgraph, std::map<const ggml_tensor *, axera_gen_snake_fuse> & sf) {
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            users[n->src[s]].push_back(n);
        }
    }
    auto readers = [&](const ggml_tensor * t) -> const std::vector<ggml_tensor *> & {
        static const std::vector<ggml_tensor *> none;
        auto it = users.find(t);
        return it == users.end() ? none : it->second;
    };
    auto snake_start = [](const ggml_tensor * h) -> const ggml_tensor * {  // h the snake's ADD
        if (h->op != GGML_OP_ADD || !h->src[1] || h->src[1]->op != GGML_OP_MUL) {
            return nullptr;
        }
        const ggml_tensor * sq = h->src[1]->src[0];
        if (!sq || sq->op != GGML_OP_SQR || !sq->src[0] || sq->src[0]->op != GGML_OP_SIN) {
            return nullptr;
        }
        const ggml_tensor * m = sq->src[0]->src[0];
        return m && m->op == GGML_OP_MUL && m->src[0] == h->src[0] ? m : nullptr;
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ic = cgraph->nodes[i];
        if (ic->op != GGML_OP_IM2COL || readers(ic).size() != 1) {
            continue;
        }
        const ggml_tensor * mm = readers(ic)[0];  // the MUL_MAT, through a view of the IM2COL
        if (mm->op != GGML_OP_MUL_MAT && (mm->op == GGML_OP_RESHAPE || mm->op == GGML_OP_VIEW) && readers(mm).size() == 1) {
            mm = readers(mm)[0];
        }
        if (mm->op != GGML_OP_MUL_MAT || (mm->src[0] != ic && mm->src[0]->view_src != ic) ||
            readers(ic->src[1]).size() < 1) {
            continue;
        }
        ggml_tensor *        in = ic->src[1];
        axera_gen_snake_fuse f;
        f.ic = ic;
        f.mm = mm;
        const ggml_tensor * h = in;
        if (in->op == GGML_OP_CONCAT && ((const int32_t *) in->op_params)[0] == 0 && in->src[1]) {
            // the CONCAT's readers: the IM2COL, and views of its tail read by CONTs only (the state)
            for (const ggml_tensor * r : readers(in)) {
                if (r == ic) {
                    continue;
                }
                bool state = r->op == GGML_OP_VIEW;
                for (const ggml_tensor * rr : readers(r)) {
                    state = state && rr->op == GGML_OP_CONT;
                }
                if (!state) {
                    h = nullptr;
                    break;
                }
            }
            f.concat = in;
            h        = h ? in->src[1] : nullptr;
            if (h && (readers(h).size() != 1 || readers(h)[0] != in)) {
                h = nullptr;
            }
        } else if (readers(in).size() != 1) {
            h = nullptr;
        }
        const ggml_tensor * st = h ? snake_start(h) : nullptr;
        // no other matrix product between the snake and its convolution (they write tmp[1] too)
        int is = -1, im = -1;
        for (int j = 0; st && j < cgraph->n_nodes; j++) {
            is = cgraph->nodes[j] == st ? j : is;
            im = cgraph->nodes[j] == mm ? j : im;
        }
        for (int j = is + 5; st && j < im; j++) {
            if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT) {
                st = nullptr;
            }
        }
        if (st && is >= 0 && im > is) {
            sf[st] = f;
        }
    }
}

// the frame's tensors in a code generation graph of n_codes positions: mtmd's inputs (inp_raw, inp_code0,
// inp_rand_<i>), the predictor's heads and embedding tables, the outputs (out_codes, the last node); its top_k (a
// TOP_K node, no top_p: no ARGSORT). False if any is missing or of another shape
static bool axera_gen_frame_find(struct ggml_cgraph * cgraph, const std::map<std::string, ggml_tensor *> & t,
                                 int n_codes, axera_gen_analysis::frame_io & fr) {
    fr         = axera_gen_analysis::frame_io();
    fr.n_codes = n_codes;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_ARGSORT) {
            return false;  // top_p < 1
        }
        if (n->op == GGML_OP_TOP_K) {
            if (fr.top_k && fr.top_k != (int) n->ne[0]) {
                return false;
            }
            fr.top_k = (int) n->ne[0];
        }
    }
    fr.h          = axera_graph_find(t, "inp_raw");
    fr.code0      = axera_graph_find(t, "inp_code0");
    fr.head_w     = axera_graph_find(t, "a.gen.code.head.weight");
    fr.embd_w     = axera_graph_find(t, "a.gen.code.embd.weight");
    fr.out_embd_w = axera_graph_find(t, "a.gen.code.out_embd.weight");
    fr.out_codes  = axera_graph_find(t, "out_codes");
    fr.out_embd   = cgraph->nodes[cgraph->n_nodes - 1];
    for (int i = 0; i < n_codes - 1; ++i) {
        fr.rnd.push_back(axera_graph_find(t, ("inp_rand_" + std::to_string(i)).c_str()));
    }
    if (fr.top_k <= 0 || !fr.h || !fr.code0 || !fr.head_w || !fr.embd_w || !fr.out_embd_w || !fr.out_codes ||
        std::find(fr.rnd.begin(), fr.rnd.end(), nullptr) != fr.rnd.end() || fr.h->type != GGML_TYPE_F32 ||
        fr.code0->type != GGML_TYPE_I32 || fr.out_codes->type != GGML_TYPE_I32 || fr.out_embd->type != GGML_TYPE_F32 ||
        ggml_nelements(fr.out_codes) != n_codes || ggml_nelements(fr.out_embd) != fr.h->ne[0] ||
        fr.head_w->ne[2] != n_codes - 1 || fr.embd_w->ne[2] != n_codes - 1 || fr.embd_w->ne[0] != fr.h->ne[0] ||
        fr.out_embd_w->ne[0] != fr.h->ne[0] || !ggml_is_contiguous(fr.head_w) || !ggml_is_contiguous(fr.embd_w) ||
        !ggml_is_contiguous(fr.out_embd) || !ggml_is_contiguous(fr.out_codes)) {
        return false;
    }
    return true;
}

static void axera_gen_analyze(ggml_backend_axera_context * ctx, axera_gen & g, struct ggml_cgraph * cgraph,
                              axera_gen_analysis & an) {
    std::set<const ggml_tensor *> &  skip  = an.skip;
    std::vector<axera_gen_cp_call> & calls = an.calls;
    float &                          theta = an.theta;
    axera_timer ta("gen graph analysis (host)");
    axera_gen_conv_epis(cgraph, an.epi);
    axera_gen_snake_fuses(cgraph, an.sfuse);
    if (axera_debug() && (!an.epi.empty() || !an.sfuse.empty())) {
        size_t res = 0;
        for (const auto & e : an.epi) {
            res += e.second.radd != nullptr;
        }
        fprintf(stderr, "axera gen: %zu convolution epilogues (%zu with a residual), %zu snakes into a conv input\n",
                an.epi.size(), res, an.sfuse.size());
    }
    std::vector<const ggml_tensor *> first;  // stacked: layer 0's output per position
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
        int positions = 0;
        for (const auto & c : calls) {
            positions = std::max(positions, c.pos + 1);
        }
        std::set<const ggml_tensor *> mids;
        const bool stack = !calls.empty() && axera_gen_cp_stackable(cgraph, calls, skip, n_layer, positions, mids);
        if (!calls.empty() &&
            !axera_gen_cp_bind(g, axera_graph_names(ctx, cgraph), n_layer, (int) calls[0].x->ne[0], positions, stack)) {
            if (axera_debug()) {
                fprintf(stderr, "axera gen: no code predictor layer kernel\n");
            }
            g.cp_off = true;  // no kernel for it: the layers as they are from now on
            calls.clear();
            skip.clear();
        }
        first.assign(positions, nullptr);
        if (g.cps && !calls.empty()) {  // one call per position: layer 0's input, the last layer's output
            std::vector<axera_gen_cp_call> st(positions);
            for (const auto & c : calls) {
                if (c.il == 0) {
                    st[c.pos].x = c.x, st[c.pos].il = 0, st[c.pos].pos = c.pos, first[c.pos] = c.out;
                }
                if (c.il == n_layer - 1) {
                    st[c.pos].out = c.out;
                }
            }
            calls = std::move(st);
            for (const ggml_tensor * m : mids) {  // (layer 0's outputs stay: the calls run there)
                if (std::find(first.begin(), first.end(), m) == first.end()) {
                    skip.insert(m);
                }
            }
            // the whole position (projection, layers, final norm, head) on the step kernel where it exists
            const ggml_tensor *pw = nullptr, *pb = nullptr, *nw = nullptr;
            std::set<const ggml_tensor *> sk = skip;
            auto steps = axera_gen_step_on() ? axera_gen_cp_steps(cgraph, calls, sk, &pw, &pb, &nw) :
                                               std::vector<axera_gen_analysis::step>();
            const axera_gen_analysis::step * hs = nullptr;  // a step with a head: its weight and logits sizes
            for (const auto & s : steps) {
                hs = hs ? hs : s.logits ? &s : nullptr;
            }
            if (hs && axera_gen_step_bind(g, axera_graph_names(ctx, cgraph), n_layer, steps[0].e->ne[0], positions, pw,
                                          pb, nw, hs->head_w, ggml_nbytes(hs->logits))) {
                an.steps = std::move(steps);
                skip     = std::move(sk);
                const auto & names = axera_graph_names(ctx, cgraph);
                an.frame.on      = axera_gen_frame_on() && (int) an.steps.size() == positions &&
                              axera_gen_frame_find(cgraph, names, positions, an.frame) &&
                              axera_gen_frame_bind(g, names, n_layer, an.frame, pw, pb, nw, theta);
            }
        }
        std::set<const ggml_tensor *> outs;
        for (const auto & c : calls) {
            outs.insert(c.out);
        }
        for (const auto & c : calls) {  // (a call without output runs right after the call that writes its input;
                                        // stacked: before the next position's)
            if (c.pos >= g.cp_LMAX || c.x->type != GGML_TYPE_F32 || ggml_nelements(c.x) != g.cp_H ||
                !ggml_is_contiguous(c.x) || (c.out && c.out->type != GGML_TYPE_F32) ||
                (!c.out && !g.cps && !outs.count(c.x))) {
                calls.clear();
                skip.clear();
                an.steps.clear();
                an.frame.on = false;
                break;
            }
        }
    }
    for (const auto & st : an.steps) {  // (the step kernel: at layer 0's output node, as the stacked calls)
        an.step_at[first[st.pos]] = &st;
    }
    for (const auto & c : calls) {
        if (g.cps && !an.steps.empty()) {
            continue;
        }
        if (g.cps) {
            an.stack_at[first[c.pos]] = &c;
            if (c.out) {
                an.stack_out[c.out] = &c;
            }
        } else {
            (c.out ? an.call_of[c.out] : an.after[c.x]) = &c;
        }
    }
    if (axera_debug() && g.cps) {
        fprintf(stderr, "axera gen: code predictor stacked, %zu calls (%zu with output), %zu step kernel calls\n",
                calls.size(), an.stack_out.size(), an.steps.size());
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
    key.reserve((size_t) cgraph->n_nodes * (3 + GGML_MAX_DIMS + GGML_MAX_SRC));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        key.push_back((uintptr_t) n);
        key.push_back((uintptr_t) n->op);
        key.push_back((uintptr_t) n->flags);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            key.push_back((uintptr_t) n->ne[d]);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            key.push_back((uintptr_t) n->src[s]);
        }
    }
    if (!g.analysis || key != g.key) {
        auto it = g.analyses.find(key);
        if (it != g.analyses.end()) {
            g.analysis = it->second;
        } else {
            auto an = std::make_shared<axera_gen_analysis>();
            axera_gen_analyze(ctx, g, cgraph, *an);
            g.analysis = an;
            if (g.analyses.size() >= 4) {
                g.analyses.clear();
            }
            g.analyses[key] = an;
        }
        g.key = std::move(key);
    }
    const axera_gen_analysis &            an    = *g.analysis;
    g.epi                                       = &an.epi;
    g.prebf16                                   = nullptr;
    if (an.frame.on && axera_gen_frame_run(g, an.frame)) {  // the whole code generation step on the frame kernel
        return GGML_STATUS_SUCCESS;
    }
    const std::set<const ggml_tensor *> & skip  = an.skip;
    const auto &                          call_of = an.call_of;
    const auto &                          after   = an.after;
    auto cp_run = [&](const axera_gen_cp_call & c) {
        axera_timer tm(g.cps ? "gen code predictor layers (npu)" : "gen code predictor layer (npu)");
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
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || skip.count(n) || g.done.count(n)) {
            continue;
        }
        auto c = call_of.find(n);
        if (c != call_of.end()) {
            cpu();
            cp_run(*c->second);
            continue;
        }
        auto sx = an.step_at.find(n);
        if (sx != an.step_at.end()) {  // a whole predictor position in one call, its logits into the head node
            cpu();
            axera_timer tm("gen code predictor step (npu)");
            axera_gen_step_run(ctx, g, an, *sx->second);
            continue;
        }
        auto sa = an.stack_at.find(n);
        if (sa != an.stack_at.end()) {  // a position's layers in one call, its result kept until its output node
            cpu();
            cp_run(*sa->second);
            continue;
        }
        auto so = an.stack_out.find(n);
        if (so != an.stack_out.end()) {
            g.Y->invalidate(g.Y->vir, g.cp_H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) g.Y->vir, (float *) n->data, g.cp_H);
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
        if (n->op == GGML_OP_MUL && !axera_gen_prof_nodes()) {
            auto sf = an.sfuse.find(n);
            if ((sf != an.sfuse.end() && axera_gen_snake_fused(g, cgraph, i, sf->second, cpu)) ||
                axera_gen_snake(cgraph, i, cpu)) {
                i += 4;
                continue;
            }
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
    g.done.clear();
    g.im2col.clear();
    return GGML_STATUS_SUCCESS;
}
