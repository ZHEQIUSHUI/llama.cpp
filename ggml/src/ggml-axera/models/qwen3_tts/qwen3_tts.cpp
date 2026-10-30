// Axera NPU backend: the Qwen3-TTS generation graphs of mtmd (neither the talker's, a llama.cpp model, nor an
// encoder's): the code predictor's frame, the codec decoder (code2wav), the speaker encoder
//
// The code predictor: one frame (its 16 codes, sampled with the graph's top-k) in one call of the frame kernel (role
// cp.frame.k<top_k>, q3tts_cp_frame.py: the predictor's layers and heads, the sampling on chip), its packs filled from
// the predictor's weights (params kinds cp, cp_out). A graph that samples otherwise (top-p) is not run.
// The other graphs node by node: a convolution (IM2COL + MUL_MAT) on its conv1d kernel (role gen.conv1d_*, the _b /
// _br variants computing the bias / residual ADDs after it), a matrix product on its GEMM (gen.mm_bf16_m<M>_n<N>_k<K>)
// where the manifest has that shape, SnakeBeta / CONCAT / transposes / COL2IM / depthwise convolutions on the host
// workers, the rest on the CPU backend.

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <set>

namespace axera {
namespace {

// a convolution's epilogue the conv1d kernel can compute (its _b / _br variant): the ADD of its bias (a [1, OC] view
// of an f32 weight) and, after it, the ADD of a residual [OW, OC] f32; their outputs written by the kernel
struct conv_epi {
    ggml_tensor *       cont = nullptr;  // ggml_conv_1d's CONT of its permuted result (a copy for one batch)
    ggml_tensor *       add = nullptr, *radd = nullptr;
    const ggml_tensor * bias = nullptr, *res = nullptr;
};

// a SnakeBeta whose output only a convolution reads (directly, or after the CONCAT of the convolution's left context,
// whose tail the graph also keeps as state): computed by the host straight into the conv1d kernel's bf16 input, and
// only that tail of the CONCAT in f32; the snake's own output and the CONCAT need not be written
struct snake_fuse {
    ggml_tensor *       concat = nullptr;  // nullptr: the IM2COL reads the snake's output
    const ggml_tensor * ic = nullptr, *mm = nullptr;
};

// the frame's tensors in a code generation graph: mtmd's inputs (inp_raw, inp_code0, inp_rand_<i>), the predictor's
// heads and embedding tables, the outputs (out_codes, the last node), its top_k
struct frame_io {
    int                              n_codes = 0, top_k = 0;
    const ggml_tensor *              h = nullptr, *code0 = nullptr, *head_w = nullptr, *embd_w = nullptr, *out_embd_w = nullptr;
    ggml_tensor *                    out_codes = nullptr, *out_embd = nullptr;
    std::vector<const ggml_tensor *> rnd;
};

// what a graph is: its epilogues and fusable snakes, or its frame
struct analysis {
    std::map<const ggml_tensor *, conv_epi>   epi;    // by MUL_MAT
    std::map<const ggml_tensor *, snake_fuse> sfuse;  // by the snake's first node
    frame_io                                  frame;
    bool                                      is_frame = false;
};

// ggml's MUL_MAT out [M][N] = src1 [M][K] . src0 [N][K]^T (out ne0 = N): src0 the weight or, for a convolution, the
// im2col rows
bool mm_shape(const ggml_tensor * n, int64_t * M, int64_t * N, int64_t * K) {
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

std::string mm_name(int64_t M, int64_t N, int64_t K) {
    char name[96];
    snprintf(name, sizeof(name), "mm_bf16_m%" PRId64 "_n%" PRId64 "_k%" PRId64, M, N, K);
    return name;
}

// a model weight (or a view of one): a tensor in a weights buffer, or a leaf that is not a graph input
bool is_weight(const ggml_tensor * t) {
    const ggml_tensor * s = t->view_src ? t->view_src : t;
    return (s->buffer && s->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
           (s->op == GGML_OP_NONE && !(s->flags & GGML_TENSOR_FLAG_INPUT) && s->data);
}

// the weight a tensor is made of by data movement only (reshape / view / transpose / permute / cont: e.g. a transposed
// convolution's kernel made contiguous on every run), nullptr if it reads anything else
const ggml_tensor * const_root(const ggml_tensor * t) {
    for (int depth = 0; t && depth < 8; ++depth) {
        if (is_weight(t)) {
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

// rows [0, rows) of a 2-D tensor of any type as bf16 into dst, on the host workers
void to_bf16(const ggml_tensor * t, int64_t rows, ggml_bf16_t * dst) {
    const int64_t K  = t->ne[0];
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
bool cont_bf16(const ggml_tensor * a, int64_t rows, ggml_bf16_t * dst) {
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

// x + sin(al x)^2 be, sin by range reduction and a polynomial (vectorizes)
inline float snake_val(float x, float al, float be) {
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
template <typename F> bool snake(ggml_cgraph * cgraph, int i, F before) {
    if (i + 4 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor **      n = cgraph->nodes + i;
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
            y[t] = snake_val(xr[t], al, be);
        }
    });
    return true;
}

// CONCAT along dim 0 of two f32 matrices (the causal convolutions' left context before their input), row by row on
// the host workers (the CPU's concat goes element by element); false if the node is not that
bool concat(ggml_tensor * n) {
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
bool transpose(ggml_tensor * n) {
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
bool col2im(ggml_tensor * n) {
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
bool dwconv(const ggml_tensor * ic, ggml_tensor * m) {
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
    const int     s0 = p[0], p0 = p[2], d0 = p[4];
    const int64_t IW = x->ne[0];
    axera_timer   tm("gen dwconv (host)");
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

// the conv1d kernel of a 1-D IM2COL's convolution (stride 1, no padding: mtmd's causal convolutions take their left
// context as input): conv1d_bf16_ic<IC>_oc<OC>_k<K>_d<D>_l<L> (x [IC, L] bf16, w [K, OC, IC] bf16 -> out [OC, L -
// (K - 1) D] f32, ggml's layout of the MUL_MAT after the IM2COL); empty if the convolution is not one
std::string conv_name(const ggml_tensor * ic, int64_t OC) {
    const int32_t * p = (const int32_t *) ic->op_params;
    if (p[0] != 1 || p[2] != 0) {
        return "";
    }
    char name[96];
    snprintf(name, sizeof(name), "conv1d_bf16_ic%" PRId64 "_oc%" PRId64 "_k%" PRId64 "_d%d_l%" PRId64, ic->src[1]->ne[1], OC,
             ic->src[0]->ne[0], p[4], ic->src[1]->ne[0]);
    return name;
}

// the epilogues of the graph's convolutions (MUL_MAT of a 1-D IM2COL): ggml_conv_1d's CONT of its permuted result (for
// one batch a copy in the same order), then through views the one ADD of a bias (an f32 weight viewed as [1, OC]),
// and if that ADD's only reader is an ADD of it and an f32 tensor of its shape, that one too (a residual)
void conv_epis(ggml_cgraph * cgraph, std::map<const ggml_tensor *, conv_epi> & epi) {
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
    auto in_cmm = [](const ggml_tensor * t) { return t->buffer && ggml_backend_buffer_is_axera(t->buffer); };
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
        conv_epi e;
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
        if (b->ne[0] != 1 || b->ne[1] != n->ne[1] || ggml_nelements(b) != n->ne[1] || !is_weight(w) ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(b)) {
            continue;
        }
        e.add           = a;
        e.bias          = b;
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

// the snakes whose output only a convolution reads (snake_fuse): by the snake's first node (MUL x * alpha; the snake's
// nodes as mtmd builds them: MUL, SIN, SQR, MUL beta, ADD x)
void snake_fuses(ggml_cgraph * cgraph, std::map<const ggml_tensor *, snake_fuse> & sf) {
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            users[n->src[s]].push_back(n);
        }
    }
    auto readers = [&](const ggml_tensor * t) -> const std::vector<ggml_tensor *> & {
        static const std::vector<ggml_tensor *> none;
        auto                                    it = users.find(t);
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
        if (mm->op != GGML_OP_MUL_MAT || (mm->src[0] != ic && mm->src[0]->view_src != ic) || readers(ic->src[1]).size() < 1) {
            continue;
        }
        ggml_tensor * in = ic->src[1];
        snake_fuse    f;
        f.ic                  = ic;
        f.mm                  = mm;
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

// a code generation graph's frame (frame_io) of n_codes positions; false if it is not one. mtmd builds the code
// generation and the code2wav sub-graphs into one graph and selects the one to run by the nodes' compute flags: a
// frame when out_codes is computed
bool frame_find(ggml_cgraph * cgraph, const graph_names & names, frame_io & fr) {
    fr                  = frame_io();
    const ggml_tensor * oc = names.find("out_codes");
    if (!oc || !(oc->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }
    fr.n_codes = (int) ggml_nelements(oc);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        if (n->op == GGML_OP_ARGSORT) {
            GGML_ABORT("axera qwen3_tts: the code predictor samples with top-p < 1; the frame kernel samples top-k only\n");
        }
        if (n->op == GGML_OP_TOP_K) {
            GGML_ASSERT(!fr.top_k || fr.top_k == (int) n->ne[0]);
            fr.top_k = (int) n->ne[0];
        }
    }
    fr.h          = names.find("inp_raw");
    fr.code0      = names.find("inp_code0");
    fr.head_w     = names.find("a.gen.code.head.weight");
    fr.embd_w     = names.find("a.gen.code.embd.weight");
    fr.out_embd_w = names.find("a.gen.code.out_embd.weight");
    fr.out_codes  = names.find("out_codes");
    fr.out_embd   = cgraph->nodes[cgraph->n_nodes - 1];
    for (int i = 0; i < fr.n_codes - 1; ++i) {
        fr.rnd.push_back(names.find("inp_rand_" + std::to_string(i)));
    }
    const int n_codes = fr.n_codes;
    if (fr.top_k <= 0 || !fr.h || !fr.code0 || !fr.head_w || !fr.embd_w || !fr.out_embd_w ||
        std::find(fr.rnd.begin(), fr.rnd.end(), nullptr) != fr.rnd.end() || fr.h->type != GGML_TYPE_F32 ||
        fr.code0->type != GGML_TYPE_I32 || fr.out_codes->type != GGML_TYPE_I32 || fr.out_embd->type != GGML_TYPE_F32 ||
        ggml_nelements(fr.out_embd) != fr.h->ne[0] || fr.head_w->ne[2] != n_codes - 1 || fr.embd_w->ne[2] != n_codes - 1 ||
        fr.embd_w->ne[0] != fr.h->ne[0] || fr.out_embd_w->ne[0] != fr.h->ne[0] || !ggml_is_contiguous(fr.head_w) ||
        !ggml_is_contiguous(fr.embd_w) || !ggml_is_contiguous(fr.out_embd) || !ggml_is_contiguous(fr.out_codes)) {
        GGML_ABORT("axera qwen3_tts: a code generation graph of an unexpected shape\n");
    }
    return true;
}

class qwen3_tts final : public model {
  public:
    explicit qwen3_tts(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    // bf16 copies of the weights the NPU reads (any GGUF type), by data and size (views of a 3-D weight are separate)
    std::map<std::pair<const void *, size_t>, axera_cmm> w;
    std::set<std::string>                                labels;  // profiling labels (axera_timer keeps the pointer)
    axera_cmm                                            tmp[2];  // per-call bf16 copies of non-weight operands
    std::set<const ggml_tensor *>                        done;    // nodes a kernel computed already (conv epilogues)
    const std::map<const ggml_tensor *, conv_epi> *      epi     = nullptr;  // the running graph's
    const ggml_tensor *                                  prebf16 = nullptr;  // the conv input tmp[1] holds already
    // 1-D IM2COL nodes not run on the CPU, by the NPU GEMM that reads them (which writes their rows as bf16 itself)
    std::map<const ggml_tensor *, const ggml_tensor *> im2col;
    // the frame kernel bound to the predictor's weights: IO, its inputs E0 / E1 / RND, its codes output
    struct {
        const kernel *                     k = nullptr;
        const ggml_tensor *                bound = nullptr;
        int                                top_k = 0;
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        std::vector<axera_cmm>             own;
        axera_cmm                          E0, E1, RND, CS, PN, MASK;
        ggml_backend_axera_buffer_context * codes = nullptr;  // (one of own)
    } fk;
    std::vector<uintptr_t>                                     key;  // the last graph's nodes, and its analysis
    std::shared_ptr<analysis>                                  an;
    std::map<std::vector<uintptr_t>, std::shared_ptr<analysis>> analyses;  // of the last few graphs

    bool have(const std::string & name) const { return man.has("gen." + name); }
    const char * label(const std::string & s) { return labels.insert(s).first->c_str(); }
    std::pair<const ggml_backend_axera_buffer_context *, void *> bf16(const ggml_tensor * a, int64_t rows, axera_cmm & t);
    std::pair<const ggml_backend_axera_buffer_context *, void *> im2col_bf16(const ggml_tensor * ic, axera_cmm & t);
    void conv(ggml_tensor * n, const ggml_tensor * ic, std::string name, const conv_epi * e);
    const ggml_tensor * im2col_user(ggml_cgraph * cgraph, int i) const;
    bool const_skip(ggml_cgraph * cgraph, int i);
    template <typename F> bool snake_fused(ggml_cgraph * cgraph, int i, const snake_fuse & f, F before);
    template <typename F> bool mm(ggml_tensor * n, F before);
    void frame_bind(ggml_cgraph * cgraph, const frame_io & fr);
    void frame_run(const frame_io & fr);
};

// the bf16 [rows][K] of an operand: the tensor itself (bf16 in CMM), its cached copy (a weight), or converted per call
// into t
std::pair<const ggml_backend_axera_buffer_context *, void *> qwen3_tts::bf16(const ggml_tensor * a, int64_t rows,
                                                                             axera_cmm & t) {
    const int64_t K     = a->ne[0];
    const size_t  bytes = (size_t) rows * K * sizeof(ggml_bf16_t);
    if (a->type == GGML_TYPE_BF16 && ggml_backend_buffer_is_axera(a->buffer)) {
        return { (const ggml_backend_axera_buffer_context *) a->buffer->context, a->data };
    }
    const ggml_tensor * root = is_weight(a) ? nullptr : const_root(a);
    if (root) {  // a constant made of a weight: cached by the weight and its own shape (its data is the graph's)
        auto & c = w[{ (const char *) root->data + a->ne[0], ggml_nbytes(a) }];
        if (!c) {
            c = axera_cmm_alloc(bytes);
            if (!cont_bf16(a, rows, (ggml_bf16_t *) c->vir)) {
                to_bf16(a, rows, (ggml_bf16_t *) c->vir);
            }
            c->flush(c->vir, bytes);
        }
        return { c.get(), c->vir };
    }
    if (is_weight(a)) {
        auto & c = w[{ a->data, ggml_nbytes(a) }];
        if (!c) {
            c = axera_cmm_alloc(bytes);
            to_bf16(a, rows, (ggml_bf16_t *) c->vir);
            c->flush(c->vir, bytes);
        }
        return { c.get(), c->vir };
    }
    if (!t || t->size < bytes) {
        t = axera_cmm_alloc_nc(bytes);  // written by the host only: no cache to clean before the NPU reads it
    }
    to_bf16(a, rows, (ggml_bf16_t *) t->vir);
    t->flush(t->vir, bytes);
    return { t.get(), t->vir };
}

// a 1-D IM2COL (ggml's rows [OW][IC][KW] of a convolution's input) written as bf16 rows into t, on the host workers:
// for the NPU GEMM that reads it, instead of the CPU's f16 rows converted once more
std::pair<const ggml_backend_axera_buffer_context *, void *> qwen3_tts::im2col_bf16(const ggml_tensor * ic, axera_cmm & t) {
    const int32_t *     p  = (const int32_t *) ic->op_params;
    const int           s0 = p[0], p0 = p[2], d0 = p[4];
    const ggml_tensor * x  = ic->src[1];
    const int64_t       KW = ic->src[0]->ne[0], IC = x->ne[1], IW = x->ne[0], OW = ic->ne[1], K = IC * KW;
    const size_t        bytes = (size_t) OW * K * sizeof(ggml_bf16_t);
    if (!t || t->size < bytes) {
        t = axera_cmm_alloc_nc(bytes);
    }
    auto *        dst = (ggml_bf16_t *) t->vir;
    const int64_t RB  = 64;  // output rows per task, built in cached memory and copied out whole (t is uncached)
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
    t->flush(t->vir, bytes);
    return { t.get(), t->vir };
}

// the MUL_MAT n after a 1-D IM2COL ic as one conv1d kernel call: the input converted to bf16, the kernel's taps
// repacked once ([OC][IC][K] -> [K][OC][IC]), the output written into n; or, with an epilogue e and the kernel's
// variant of it (<name>_br: bias and residual, <name>_b: bias), into the epilogue's last ADD, the ADDs marked done
void qwen3_tts::conv(ggml_tensor * n, const ggml_tensor * ic, std::string name, const conv_epi * e) {
    const bool br = e && e->radd && have(name + "_br");
    const bool bo = e && !br && have(name + "_b");
    name += br ? "_br" : bo ? "_b" : "";
    ggml_tensor *       dst  = br ? e->radd : bo ? e->add : e && e->cont ? e->cont : n;
    const ggml_tensor * x    = ic->src[1];
    const ggml_tensor * wt   = ic->src[0];
    const int64_t       KW = wt->ne[0], IC = x->ne[1], L = x->ne[0], OC = n->ne[1];
    auto *              dctx = (const ggml_backend_axera_buffer_context *) dst->buffer->context;
    axera_timer         tm(label("gen " + name + " (npu)"));
    auto &              wk = w[{ wt->data, ggml_nbytes(wt) + 1 }];  // (+1: not the GEMM's copy of the same weight)
    if (!wk) {
        GGML_ASSERT(ggml_is_contiguous(wt) && wt->ne[1] == IC && wt->ne[2] == OC);
        wk          = axera_cmm_alloc((size_t) KW * OC * IC * sizeof(ggml_bf16_t));
        auto *     d  = (ggml_bf16_t *) wk->vir;
        const auto tr = ggml_get_type_traits(wt->type);
        axera_host_pool().run(OC, [&](int64_t o) {  // row o: [IC][K] of any type
            std::vector<float> f(IC * KW);
            const char *       src = (const char *) wt->data + o * wt->nb[2];
            if (wt->type == GGML_TYPE_F32) {
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
    if (!tmp[1] || tmp[1]->size < xb) {
        tmp[1] = axera_cmm_alloc_nc(xb);  // written by the host only: no cache to clean before the NPU reads it
    }
    if (prebf16 != x) {
        axera_timer tx("gen conv: x (host)");
        to_bf16(x, IC, (ggml_bf16_t *) tmp[1]->vir);
    }
    prebf16              = nullptr;
    const kernel & k     = man.get("gen." + name);
    const int      ix = k.input("x"), iw = k.input("w"), io = k.output("out");
    const int      ib = k.find_input("b"), ir = k.find_input("res");
    const uint32_t n_in = 2 + (br || bo) + br;
    GGML_ASSERT(k.info->nInputSize == n_in && k.info->nOutputSize == 1 && (ib >= 0) == (br || bo) && (ir >= 0) == br);
    std::vector<AX_ENGINE_IO_BUFFER_T> in(n_in), out(1);
    in[ix]  = axera_io(tmp[1].get(), tmp[1]->vir, (uint32_t) xb);
    in[iw]  = axera_io(wk.get(), wk->vir, (uint32_t) wk->size);
    out[io] = axera_io(dctx, dst->data, (uint32_t) ggml_nbytes(dst));
    if (ib >= 0) {  // the bias, f32 (a CMM copy unless it is in CMM already)
        const ggml_tensor * b  = e->bias;
        const size_t        bs = ggml_nbytes(b);
        if (b->buffer && ggml_backend_buffer_is_axera(b->buffer)) {
            in[ib] = axera_io((const ggml_backend_axera_buffer_context *) b->buffer->context, b->data, (uint32_t) bs);
        } else {
            auto & c = w[{ b->data, bs + 2 }];  // (+2: not a bf16 copy of the same data)
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
    dctx->invalidate(dst->data, ggml_nbytes(dst));  // no dirty line of the output left to be written back over the NPU's
    {
        axera_timer tr("gen conv: run (npu)");
        k.run(in, out, name.c_str());
    }
    dctx->invalidate(dst->data, ggml_nbytes(dst));
    if (e && e->cont) {
        done.insert(e->cont);
    }
    if (br || bo) {
        done.insert(e->add);
    }
    if (br) {
        done.insert(e->radd);
    }
}

// the GEMM reading a 1-D IM2COL of node i within the next nodes, if a GEMM or conv1d kernel takes its shape: the
// IM2COL is then left to it
const ggml_tensor * qwen3_tts::im2col_user(ggml_cgraph * cgraph, int i) const {
    const ggml_tensor * ic = cgraph->nodes[i];
    const int32_t *     p  = (const int32_t *) ic->op_params;
    if (p[6] != 0 || ic->src[1]->type != GGML_TYPE_F32 || ic->src[1]->ne[2] * ic->src[1]->ne[3] != 1) {
        return nullptr;
    }
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 8); j++) {
        const ggml_tensor * m = cgraph->nodes[j];
        int64_t             M, N, K;
        if (m->op == GGML_OP_MUL_MAT && (m->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            (m->src[0] == ic || m->src[0]->view_src == ic) && mm_shape(m, &M, &N, &K) && N == ic->ne[1] && K == ic->ne[0]) {
            const std::string cn = conv_name(ic, M);
            return have(mm_name(M, N, K)) || (!cn.empty() && have(cn)) ? m : nullptr;
        }
    }
    return nullptr;
}

// a CONT of a weight (const_root) whose only readers are offloaded MUL_MATs, which have its bf16 copy or make it from
// the CONT's source view: nothing reads the CONT's own data, it is not run
bool qwen3_tts::const_skip(ggml_cgraph * cgraph, int i) {
    const ggml_tensor * n    = cgraph->nodes[i];
    const ggml_tensor * root = const_root(n);
    if (!root || (!w.count({ (const char *) root->data + n->ne[0], ggml_nbytes(n) }) &&
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
                if (m->op != GGML_OP_MUL_MAT || s != 0 || m->src[0] != n || !mm_shape(m, &M, &N, &K) ||
                    !have(mm_name(M, N, K))) {
                    return false;
                }
                ++readers;
            }
        }
    }
    return readers > 0;
}

// the snake at node i fused into its convolution's input (snake_fuse f): bf16 rows [IC][L] into tmp[1] (the CONCAT's
// left context, then the snake), the CONCAT's last pad steps of each row in f32 (the state the graph keeps); false if
// the nodes are not a snake or the convolution has no kernel
template <typename F> bool qwen3_tts::snake_fused(ggml_cgraph * cgraph, int i, const snake_fuse & f, F before) {
    ggml_tensor **      n  = cgraph->nodes + i;
    const ggml_tensor * x  = n[0]->src[0];
    const ggml_tensor * a  = n[0]->src[1];
    const ggml_tensor * b  = n[3]->src[1];
    const std::string   cn = conv_name(f.ic, f.mm->ne[1]);
    if (i + 4 >= cgraph->n_nodes || n[4]->op != GGML_OP_ADD || cn.empty() || !have(cn) || x->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
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
    if (!tmp[1] || tmp[1]->size < xb) {
        tmp[1] = axera_cmm_alloc_nc(xb);
    }
    auto *        dst = (ggml_bf16_t *) tmp[1]->vir;
    const int64_t TB = 4096, nt = (L + TB - 1) / TB;
    axera_timer   tm("gen snake + conv input (host)");
    axera_host_pool().run(C * nt, [&](int64_t job) {
        const int64_t c = job / nt, t0 = job % nt * TB, t1 = std::min(L, t0 + TB);
        const float   al = ((const float *) a->data)[c], be = ((const float *) b->data)[c];
        const float * xr   = (const float *) ((const char *) x->data + c * x->nb[1]);
        const float * lr   = left ? (const float *) ((const char *) left->data + c * left->nb[1]) : nullptr;
        float *       tail = f.concat ? (float *) ((char *) f.concat->data + c * f.concat->nb[1]) : nullptr;
        ggml_bf16_t   row[TB];
        float         v[TB];
        for (int64_t t = t0; t < t1; ++t) {
            v[t - t0] = t < P ? lr[t] : snake_val(xr[t - P], al, be);
        }
        ggml_fp32_to_bf16_row(v, row, t1 - t0);
        memcpy(dst + c * L + t0, row, (t1 - t0) * sizeof(ggml_bf16_t));
        for (int64_t t = std::max(t0, L - P); tail && t < t1; ++t) {  // the state's steps
            tail[t] = v[t - t0];
        }
    });
    if (f.concat) {
        done.insert(f.concat);
    }
    prebf16 = f.ic->src[1];
    return true;
}

// one MUL_MAT on the NPU if a kernel takes its shape: a convolution's conv1d kernel, else mm_bf16_m<M>_n<N>_k<K> (x
// [M, K], w [N, K] bf16 -> out [M, N] f32, written into the node). before() runs first (the CPU nodes ahead of it);
// false if no kernel takes it
template <typename F> bool qwen3_tts::mm(ggml_tensor * n, F before) {
    int64_t M, N, K;
    if (!mm_shape(n, &M, &N, &K)) {
        return false;
    }
    auto ic = im2col.find(n);
    if (ic != im2col.end()) {  // a convolution: its conv1d kernel if there is one
        const std::string cn = conv_name(ic->second, M);
        if (!cn.empty() && have(cn)) {
            before();
            const auto e = epi->find(n);
            conv(n, ic->second, cn, e != epi->end() ? &e->second : nullptr);
            return true;
        }
    }
    const std::string name = mm_name(M, N, K);
    if (!have(name)) {
        return false;
    }
    before();
    const ggml_tensor * a    = n->src[0];
    const ggml_tensor * b    = n->src[1];
    auto *              dctx = (const ggml_backend_axera_buffer_context *) n->buffer->context;
    char                lb[96];
    snprintf(lb, sizeof(lb), "gen mm (npu) m%" PRId64 " n%" PRId64 " k%" PRId64, M, N, K);
    axera_timer tm(label(lb));
    axera_timer tw(ic != im2col.end() ? "gen mm: im2col rows (host)" : "gen mm: w (host)");
    const auto  wb = ic != im2col.end() ? im2col_bf16(ic->second, tmp[0]) : bf16(a, N, tmp[0]);
    tw.stop();
    axera_timer tx("gen mm: x (host)");
    const auto  x = bf16(b, M, tmp[1]);
    tx.stop();
    axera_timer    tr("gen mm: run (npu)");
    const kernel & k = man.get("gen." + name);
    GGML_ASSERT(k.info->nInputSize == 2 && k.info->nOutputSize == 1);
    std::vector<AX_ENGINE_IO_BUFFER_T> in(2), out(1);
    in[k.input("x")]   = axera_io(x.first, x.second, (uint32_t) (M * K * sizeof(ggml_bf16_t)));
    in[k.input("w")]   = axera_io(wb.first, wb.second, (uint32_t) (N * K * sizeof(ggml_bf16_t)));
    out[k.output("out")] = axera_io(dctx, n->data, (uint32_t) ggml_nbytes(n));
    dctx->invalidate(n->data, ggml_nbytes(n));  // no dirty line of the output left to be written back over the NPU's
    k.run(in, out, name.c_str());
    dctx->invalidate(n->data, ggml_nbytes(n));
    return true;
}

// binds the frame kernel of the graph's top-k to the predictor's weights: its packs (params), the codebook heads and
// embedding tables as bf16, the positions' RoPE rows, pos | nblk and mask rows
void qwen3_tts::frame_bind(ggml_cgraph * cgraph, const frame_io & fr) {
    const ggml_tensor * w0 = names.at("a.gen.code.blk.0.ln1.weight");
    if (fk.k && fk.bound == w0 && fk.top_k == fr.top_k) {
        return;
    }
    const kernel & k  = man.get("cp.frame.k" + std::to_string(fr.top_k));
    const int64_t  HT = fr.h->ne[0], NP = fr.n_codes, V = fr.head_w->ne[1];
    GGML_ASSERT(k.in_bytes("e0") == HT * sizeof(ggml_bf16_t) && k.in_bytes("e1") == HT * sizeof(ggml_bf16_t) &&
                k.in_bytes("heads") == (size_t) ggml_nelements(fr.head_w) * sizeof(ggml_bf16_t) &&
                k.in_bytes("embd") == (size_t) ggml_nelements(fr.embd_w) * sizeof(ggml_bf16_t) &&
                k.in_bytes("rnd") == (size_t) (NP - 1) * sizeof(float) && k.in_bytes("pn16") == (size_t) NP * 2 * sizeof(int32_t) &&
                k.out_bytes("lgs") == (size_t) (NP - 1) * V * sizeof(float) && k.out_bytes("codes") % (NP * sizeof(int32_t)) == 0);
    const AX_ENGINE_IOMETA_T & mk = k.info->pInputs[k.input("mask16")];  // [NP * LMAX / BC, BC] bf16
    GGML_ASSERT(mk.nShapeSize == 2 && mk.pShape[0] % NP == 0);
    const int    bc       = (int) mk.pShape[1];
    const size_t cs_bytes = k.in_bytes("cs16"), mask_bytes = k.in_bytes("mask16");
    const int    rot      = (int) (cs_bytes / NP / sizeof(ggml_bf16_t));  // per position cos | sin, ROT / 2 each
    float        theta    = 0.0f;  // the predictor's RoPE base: a ROPE the frame computes (not code2wav's)
    for (int i = 0; i < cgraph->n_nodes && theta == 0.0f; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_ROPE && (t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            memcpy(&theta, (const int32_t *) t->op_params + 5, sizeof(float));
        }
    }
    GGML_ASSERT(theta > 0.0f);
    fk.E0   = axera_cmm_alloc(HT * sizeof(ggml_bf16_t));
    fk.E1   = axera_cmm_alloc(HT * sizeof(ggml_bf16_t));
    fk.RND  = axera_cmm_alloc((NP - 1) * sizeof(float));
    fk.CS   = axera_cmm_alloc(cs_bytes);
    fk.PN   = axera_cmm_alloc(NP * 2 * sizeof(int32_t));
    fk.MASK = axera_cmm_alloc(mask_bytes);
    auto *        cs    = (ggml_bf16_t *) fk.CS->vir;
    auto *        pn    = (int32_t *) fk.PN->vir;
    auto *        mask  = (ggml_bf16_t *) fk.MASK->vir;
    const int64_t cells = (int64_t) (mask_bytes / NP / sizeof(ggml_bf16_t));
    for (int64_t p = 0; p < NP; ++p) {
        for (int i = 0; i < rot / 2; ++i) {
            const float a             = (float) p * powf(theta, -2.0f * i / rot);
            cs[p * rot + i]           = ggml_fp32_to_bf16(cosf(a));
            cs[p * rot + rot / 2 + i] = ggml_fp32_to_bf16(sinf(a));
        }
        pn[2 * p]     = (int32_t) p;
        pn[2 * p + 1] = (int32_t) (p / bc + 1);
        for (int64_t c = 0; c < cells; ++c) {
            mask[p * cells + c] = ggml_fp32_to_bf16(c <= p ? 0.0f : -3e38f);
        }
    }
    fk.CS->flush();
    fk.PN->flush();
    fk.MASK->flush();
    const auto heads = bf16(fr.head_w, ggml_nrows(fr.head_w), tmp[0]);
    const auto embd  = bf16(fr.embd_w, ggml_nrows(fr.embd_w), tmp[0]);
    fk.in.assign(k.info->nInputSize, {});
    fk.out.assign(k.info->nOutputSize, {});
    for (const auto & d : k.desc->in) {
        const int i = k.input(d.name);
        if (d.name == "heads" || d.name == "embd") {
            const auto & b = d.name == "heads" ? heads : embd;
            fk.in[i]       = axera_io(b.first, b.second, d.bytes);
        } else if (k.desc->packs.count(d.name)) {
            fk.in[i] = axera_io(prm.pack(names.t, *k.desc, d.name, 0), d.bytes);
        } else {
            const axera_cmm * b = d.name == "e0" ? &fk.E0 : d.name == "e1" ? &fk.E1 : d.name == "rnd" ? &fk.RND :
                                  d.name == "cs16" ? &fk.CS : d.name == "pn16" ? &fk.PN : d.name == "mask16" ? &fk.MASK : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen3_tts: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
            }
            fk.in[i] = axera_io(*b, d.bytes);
        }
    }
    fk.own.clear();
    for (const auto & d : k.desc->out) {  // codes, logits, or a layer's resident K / V
        fk.own.push_back(axera_cmm_alloc(d.bytes));
        fk.out[k.output(d.name)] = axera_io(fk.own.back(), d.bytes);
        if (d.name == "codes") {
            fk.codes = fk.own.back().get();
        }
    }
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live = { &fk.in };
    prm.release(live);
    fk.k     = &k;
    fk.bound = w0;
    fk.top_k = fr.top_k;
}

// the frame on the frame kernel: the talker's hidden state and code 0's embedding row as bf16, the draws; then the
// graph's outputs as its nodes would have computed them: the 16 codes, and the sum of their embeddings in f32
void qwen3_tts::frame_run(const frame_io & fr) {
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
    ggml_fp32_to_bf16_row((const float *) fr.h->data, (ggml_bf16_t *) fk.E0->vir, HT);
    ggml_fp32_to_bf16_row(e1.data(), (ggml_bf16_t *) fk.E1->vir, HT);
    for (int64_t p = 0; p < NP - 1; ++p) {
        ((float *) fk.RND->vir)[p] = *(const float *) fr.rnd[p]->data;
    }
    fk.E0->flush();
    fk.E1->flush();
    fk.RND->flush();
    {
        axera_timer tm("gen code predictor frame (npu)");
        fk.k->run(fk.in, fk.out, "code predictor frame");
    }
    fk.codes->invalidate();
    const auto * cr    = (const int32_t *) fk.codes->vir;
    const size_t cores = fk.codes->size / (NP * sizeof(int32_t));
    for (size_t c = 1; c < cores; ++c) {  // every core sampled the same codes
        if (memcmp(cr, cr + c * NP, NP * sizeof(int32_t)) != 0) {
            GGML_ABORT("axera qwen3_tts: the frame kernel's cores sampled different codes\n");
        }
    }
    std::vector<int32_t> codes(NP);
    codes[0] = code0;
    for (int64_t p = 1; p < NP; ++p) {
        GGML_ASSERT(cr[p] >= 0 && cr[p] < fr.embd_w->ne[1]);
        codes[p] = cr[p];
    }
    GGML_ASSERT(ggml_nbytes(fr.out_codes) == NP * sizeof(int32_t));
    memcpy(fr.out_codes->data, codes.data(), NP * sizeof(int32_t));
    // the next frame's embedding: code 0's row plus each code's row of its table, added in the graph's order
    const auto * tre = ggml_get_type_traits(fr.embd_w->type);
    sum              = e1;
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
    for (const ggml_tensor * o : { (const ggml_tensor *) fr.out_codes, (const ggml_tensor *) fr.out_embd }) {
        if (ggml_backend_buffer_is_axera(o->buffer)) {
            ((ggml_backend_axera_buffer_context *) o->buffer->context)->flush(o->data, ggml_nbytes(o));
        }
    }
}

ggml_status qwen3_tts::compute(ggml_cgraph * cgraph) {
    names.of(cgraph);
    // analyzed once per graph (mtmd builds the same graph in the same memory for every frame; the code generation and
    // code2wav graphs alternate)
    std::vector<uintptr_t> k;
    k.reserve((size_t) cgraph->n_nodes * (3 + GGML_MAX_DIMS + GGML_MAX_SRC));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        k.push_back((uintptr_t) n);
        k.push_back((uintptr_t) n->op);
        k.push_back((uintptr_t) n->flags);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            k.push_back((uintptr_t) n->ne[d]);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            k.push_back((uintptr_t) n->src[s]);
        }
    }
    if (!an || k != key) {
        auto it = analyses.find(k);
        if (it != analyses.end()) {
            an = it->second;
        } else {
            axera_timer ta("gen graph analysis (host)");
            an           = std::make_shared<analysis>();
            an->is_frame = frame_find(cgraph, names, an->frame);
            if (!an->is_frame) {
                conv_epis(cgraph, an->epi);
                snake_fuses(cgraph, an->sfuse);
            }
            if (analyses.size() >= 4) {
                analyses.clear();
            }
            analyses[k] = an;
        }
        key = std::move(k);
    }
    if (an->is_frame) {  // the whole code generation step on the frame kernel
        frame_bind(cgraph, an->frame);
        frame_run(an->frame);
        return GGML_STATUS_SUCCESS;
    }
    epi     = &an->epi;
    prebf16 = nullptr;
    std::vector<ggml_tensor *> run;  // CPU nodes not yet computed
    auto cpu = [&] {
        cpu_compute(cgraph, run.data(), (int) run.size(), "gen (cpu)");
        run.clear();
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || done.count(n)) {
            continue;
        }
        if (n->op == GGML_OP_IM2COL) {
            if (const ggml_tensor * m = im2col_user(cgraph, i)) {
                im2col[m] = n;
                continue;
            }
            if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
                cgraph->nodes[i + 1]->src[0] == n && cgraph->nodes[i + 1]->src[1] == n->src[0]) {
                cpu();
                if (dwconv(n, cgraph->nodes[i + 1])) {
                    i += 1;
                    continue;
                }
            }
        }
        if ((n->op == GGML_OP_CONCAT || n->op == GGML_OP_CONT || n->op == GGML_OP_COL2IM_1D) &&
            ggml_nelements(n) >= 65536 && !const_root(n)) {  // (large ones: the host workers' start costs)
            cpu();
            if (concat(n) || transpose(n) || col2im(n)) {
                continue;
            }
        }
        if (n->op == GGML_OP_CONT && const_skip(cgraph, i)) {  // a constant the NPU already has as bf16
            continue;
        }
        if (n->op == GGML_OP_MUL) {
            auto sf = an->sfuse.find(n);
            if ((sf != an->sfuse.end() && snake_fused(cgraph, i, sf->second, cpu)) || snake(cgraph, i, cpu)) {
                i += 4;
                continue;
            }
        }
        if (n->op == GGML_OP_MUL_MAT && mm(n, cpu)) {
            continue;
        }
        run.push_back(n);
    }
    cpu();
    done.clear();
    im2col.clear();
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen3_tts(const manifest & m) {
    return std::make_shared<qwen3_tts>(m);
}

}  // namespace axera
