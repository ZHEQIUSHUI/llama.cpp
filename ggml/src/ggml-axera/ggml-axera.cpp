// Axera NPU backend.
//
// - Device memory is CMM allocated cached (AX_SYS_MemAllocCached). It is host-visible, so the CPU backend works on
//   it directly for every op this backend does not take; the NPU addresses it physically. Cache maintenance: the CPU
//   side flushes after writing (set_tensor, staging), the NPU side flushes its output range before a run (no dirty
//   line may be evicted over the result) and invalidates it afterwards.
// - Kernels are precompiled axmodels (AXDSL), looked up by name in GGML_AXERA_KERNEL_DIR (default ./axera_kernels).
//   An op is supported only if the kernel for its shape exists; everything else falls back to the CPU backend.
// - Ops: MUL_MAT with BF16 weights and F32 activations. Kernel "mul_mat_bf16_n<C>_k<K>.axmodel" (x [1, K] bf16,
//   w [C, K] bf16 -> out [1, C] f32) computes one activation row against C weight rows; N is covered by N / C calls
//   (C = N, or the largest kernel C that divides N, e.g. a vocabulary-sized output projection in chunks). Several
//   activation rows (prompt processing) run row by row, rereading the weights each time, until a batched kernel exists.
// - Single-core kernels only: the current NPU driver does not run tasks on several cores concurrently.
//
// GGML_AXERA_TEMPLATE=qwen35 runs whole Qwen3.5 graphs with per-layer kernels instead (see "graph template" below).
// GGML_AXERA_DEBUG=1 logs every missing kernel once.

#include "ggml-axera.h"
#include "mx.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <dirent.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t AXERA_ALIGNMENT = 128;

static bool axera_debug() {
    static const bool debug = [] {
        const char * e = std::getenv("GGML_AXERA_DEBUG");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return debug;
}

// AX_SYS / AX_ENGINE are process-wide; initialized once, never torn down while tensors may still live in CMM
static bool axera_runtime_init() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        if (AX_SYS_Init() != 0) {
            GGML_LOG_WARN("%s: AX_SYS_Init failed, Axera NPU unavailable\n", __func__);
            return;
        }
        AX_ENGINE_NPU_ATTR_T attr;
        memset(&attr, 0, sizeof(attr));
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        const int ret = AX_ENGINE_Init(&attr);
        if (ret != 0) {
            GGML_LOG_WARN("%s: AX_ENGINE_Init failed (0x%x), Axera NPU unavailable\n", __func__, ret);
            return;
        }
        ok = true;
    });
    return ok;
}

// CMM total/free from /proc/ax_proc/mem_cmm_info (sum over partitions)
static void axera_cmm_memory(size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    std::ifstream f("/proc/ax_proc/mem_cmm_info");
    std::string line;
    while (std::getline(f, line)) {
        size_t p = line.find("PARTITION:");
        if (p != std::string::npos) {
            size_t s = line.find("Size=", p);
            if (s != std::string::npos) {
                const size_t kb = std::strtoull(line.c_str() + s + 5, nullptr, 10);
                *total += kb * 1024;
                *free  += kb * 1024;
            }
            continue;
        }
        p = line.find("nbytes(");
        if (p != std::string::npos) {
            size_t c = line.find("Cur=", p);
            if (c != std::string::npos) {
                const size_t used = std::strtoull(line.c_str() + c + 4, nullptr, 10);
                *free -= std::min(*free, used);
            }
        }
    }
}

//
// kernels
//

struct ggml_axera_kernel {
    AX_ENGINE_HANDLE      handle = nullptr;
    AX_ENGINE_IO_INFO_T * info   = nullptr;

    ~ggml_axera_kernel() {
        if (handle) {
            AX_ENGINE_DestroyHandle(handle);
        }
    }

    int input(const char * name) const {
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            if (strcmp(info->pInputs[i].pName, name) == 0) {
                return (int) i;
            }
        }
        return -1;
    }

    int output(const char * name) const {
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            if (strcmp(info->pOutputs[i].pName, name) == 0) {
                return (int) i;
            }
        }
        return -1;
    }
};

struct ggml_axera_kernel_registry {
    std::mutex                                                mutex;
    std::string                                               dir;
    std::map<int64_t, std::set<int64_t>>                      mul_mat;      // bf16: K -> available weight-row counts C
    std::map<int64_t, std::set<int64_t>>                      mul_mat_mx8;  // MXFP8 weights (x, w, s -> out)
    std::set<std::string>                                     reported;
    std::map<std::string, std::unique_ptr<ggml_axera_kernel>> loaded;

    ggml_axera_kernel_registry() {
        const char * e = std::getenv("GGML_AXERA_KERNEL_DIR");
        dir = e ? e : "./axera_kernels";
        if (DIR * d = opendir(dir.c_str())) {
            while (dirent * f = readdir(d)) {
                long long c, k;
                char      tail[16];
                if (sscanf(f->d_name, "mul_mat_bf16_n%lld_k%lld%15s", &c, &k, tail) == 3 && strcmp(tail, ".axmodel") == 0) {
                    mul_mat[k].insert(c);
                }
                if (sscanf(f->d_name, "mul_mat_mx8_n%lld_k%lld%15s", &c, &k, tail) == 3 && strcmp(tail, ".axmodel") == 0) {
                    mul_mat_mx8[k].insert(c);
                }
            }
            closedir(d);
        }
    }

    std::string path(const std::string & name) const { return dir + "/" + name + ".axmodel"; }

    // weight rows per call for a [N, K] weight: N itself, else the largest available C dividing N; 0 if none
    int64_t mul_mat_chunk(int64_t n, int64_t k, bool mx8 = false) {
        std::lock_guard<std::mutex> lock(mutex);
        auto & kernels = mx8 ? mul_mat_mx8 : mul_mat;
        auto   it      = kernels.find(k);
        if (it != kernels.end()) {
            for (auto c = it->second.rbegin(); c != it->second.rend(); ++c) {
                if (*c <= n && n % *c == 0) {
                    return *c;
                }
            }
        }
        char key[64];
        snprintf(key, sizeof(key), "%s_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", n, k);
        if (axera_debug() && reported.insert(key).second) {
            GGML_LOG_INFO("%s: no kernel mul_mat_%s in %s\n", __func__, key, dir.c_str());
        }
        return 0;
    }

    ggml_axera_kernel * get(const std::string & name) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = loaded.find(name);
        if (it != loaded.end()) {
            return it->second.get();
        }
        std::ifstream f(path(name), std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto k = std::make_unique<ggml_axera_kernel>();
        int ret = data.empty() ? -1 : AX_ENGINE_CreateHandle(&k->handle, data.data(), data.size());
        if (ret == 0) {
            ret = AX_ENGINE_CreateContext(k->handle);
        }
        if (ret == 0) {
            ret = AX_ENGINE_GetIOInfo(k->handle, &k->info);
        }
        if (ret != 0) {
            GGML_ABORT("%s: failed to load kernel %s (0x%x)\n", __func__, path(name).c_str(), ret);
        }
        return (loaded[name] = std::move(k)).get();
    }
};

static ggml_axera_kernel_registry & axera_kernels() {
    static ggml_axera_kernel_registry registry;
    return registry;
}

static std::string axera_mul_mat_kernel_name(int64_t c, int64_t k, bool mx8 = false) {
    char name[96];
    snprintf(name, sizeof(name), "mul_mat_%s_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", c, k);
    return name;
}

//
// buffer (CMM)
//

struct ggml_backend_axera_buffer_context {
    AX_U64 phy  = 0;
    void * vir  = nullptr;
    size_t size = 0;

    ~ggml_backend_axera_buffer_context() {
        if (vir) {
            AX_SYS_MemFree(phy, vir);
        }
    }

    AX_U64 phy_of(const void * p) const { return phy + ((const char *) p - (const char *) vir); }

    void flush(void * p, size_t n) const {
        if (n) {
            AX_SYS_MflushCache(phy_of(p), p, (AX_U32) n);
        }
    }

    void invalidate(void * p, size_t n) const {
        if (n) {
            AX_SYS_MinvalidateCache(phy_of(p), p, (AX_U32) n);
        }
    }
};

static void ggml_backend_axera_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_axera_buffer_context *) buffer->context;
}

static void * ggml_backend_axera_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_axera_buffer_context *) buffer->context)->vir;
}

static void ggml_backend_axera_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                    size_t offset, size_t size) {
    auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
    memset((char *) tensor->data + offset, value, size);
    ctx->flush((char *) tensor->data + offset, size);
}

static void ggml_backend_axera_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                 size_t offset, size_t size) {
    auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
    memcpy((char *) tensor->data + offset, data, size);
    ctx->flush((char *) tensor->data + offset, size);
}

static void ggml_backend_axera_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                 size_t offset, size_t size) {
    // the range may hold NPU results not invalidated yet (template state read for a checkpoint) or CPU writes not
    // flushed yet (CPU backend ops on CMM): write back, then drop the cached lines, then read
    auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
    char * p   = (char *) tensor->data + offset;
    ctx->flush(p, size);
    ctx->invalidate(p, size);
    memcpy(data, p, size);
}

static bool ggml_backend_axera_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src,
                                                 ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
        memcpy(dst->data, src->data, ggml_nbytes(src));
        ctx->flush(dst->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_axera_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
    memset(ctx->vir, value, ctx->size);
    ctx->flush(ctx->vir, ctx->size);
}

static const struct ggml_backend_buffer_i ggml_backend_axera_buffer_i = {
    /* .free_buffer     = */ ggml_backend_axera_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_axera_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_axera_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_axera_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_axera_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_axera_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_axera_buffer_clear,
    /* .reset           = */ NULL,
};

// buffer type

static const char * ggml_backend_axera_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_axera_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ggml_backend_axera_buffer_context;
    ctx->size  = std::max<size_t>(size, AXERA_ALIGNMENT);
    if (AX_SYS_MemAllocCached(&ctx->phy, &ctx->vir, (AX_U32) ctx->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml") != 0) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
        ctx->vir = nullptr;
        delete ctx;
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_axera_buffer_i, ctx, size);
}

static size_t ggml_backend_axera_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return AXERA_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_axera_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return (size_t) 1 << 31;  // AX_SYS_MemAllocCached / cache maintenance take 32-bit sizes
    GGML_UNUSED(buft);
}

static bool ggml_backend_axera_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_axera(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft == ggml_backend_axera_buffer_type();
}

//
// backend
//

// CMM owned by the backend (staging, template state); same cache rules as the buffers above
static std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    if (AX_SYS_MemAllocCached(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
    memset(b->vir, 0, b->size);
    b->flush(b->vir, b->size);
    return b;
}

static AX_ENGINE_IO_BUFFER_T axera_io(const ggml_backend_axera_buffer_context * ctx, void * p, uint32_t size) {
    AX_ENGINE_IO_BUFFER_T b;
    memset(&b, 0, sizeof(b));
    b.phyAddr  = ctx->phy_of(p);
    b.pVirAddr = p;
    b.nSize    = size;
    return b;
}

//
// Qwen3.5 graph template (GGML_AXERA_TEMPLATE=qwen35)
//
// The backend claims the whole graph and runs it with the AXDSL layer kernels instead of op by op: per token, 24
// calls (gated delta net layer / gated attention layer, weights straight from the GGUF tensors in CMM), then the final
// norm on the CPU and the output projection with the GEMV kernels. The kernels' recurrent state and KV cache live in
// the backend, not in the llama.cpp memory: a single sequence with consecutive positions, restarted at position 0.
// Kernels in GGML_AXERA_KERNEL_DIR: qwen35_gdn.axmodel, qwen35_attn.axmodel, qwen35.txt ("LMAX BC ROT THETA"), and the
// mul_mat GEMV kernel for the output projection.
//

static bool axera_template_qwen35() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_TEMPLATE");
        return e != nullptr && strcmp(e, "qwen35") == 0;
    }();
    return on;
}

// GGML_AXERA_WEIGHTS=mxfp8: the template quantizes the bf16 GGUF weights to MXFP8 at load (about half the bytes per
// token; see docs/backend/AXERA.md for the accuracy cost)
static bool axera_weights_mx8() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_WEIGHTS");
        return e != nullptr && strcmp(e, "mxfp8") == 0;
    }();
    return on;
}

using axera_cmm = std::unique_ptr<ggml_backend_axera_buffer_context>;

// runs fn(row, float row) for every row of a 2-D weight of any GGUF type (f32, f16, bf16, quantized), on all cores
template <typename F> static void axera_for_rows_f32(const ggml_tensor * w, F fn) {
    GGML_ASSERT(ggml_is_contiguous(w));
    const int64_t K      = w->ne[0], N = ggml_nrows(w);
    const auto *  traits = ggml_get_type_traits(w->type);
    GGML_ASSERT(w->type == GGML_TYPE_F32 || traits->to_float != nullptr);
    const int                n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            std::vector<float> row(K);
            for (int64_t r = t; r < N; r += n_threads) {
                const void * src = (const char *) w->data + r * w->nb[1];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, K * sizeof(float));
                } else {
                    traits->to_float(src, row.data(), K);
                }
                fn(r, row.data());
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }
}

// [N, K] weight of any GGUF type -> MXFP8 elements [N, K] and scales [N, K / 32] in new CMM buffers
static void axera_quantize_mxfp8(const ggml_tensor * w, axera_cmm & q, axera_cmm & s) {
    GGML_ASSERT(w->ne[0] % AXERA_MX_GROUP == 0);
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    q = axera_cmm_alloc(N * K);
    s = axera_cmm_alloc(N * K / AXERA_MX_GROUP);
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        axera_quantize_row_mxfp8_f32(row, (uint8_t *) q->vir + r * K, (uint8_t *) s->vir + r * (K / AXERA_MX_GROUP), K);
    });
    q->flush(q->vir, q->size);
    s->flush(s->vir, s->size);
}

// [N, K] weight of another GGUF type (quantized, f16, f32) -> bf16 copy in a new CMM buffer
static axera_cmm axera_to_bf16(const ggml_tensor * w) {
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    axera_cmm     b = axera_cmm_alloc(N * K * sizeof(ggml_bf16_t));
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        ggml_fp32_to_bf16_row(row, (ggml_bf16_t *) b->vir + r * K, K);
    });
    b->flush(b->vir, b->size);
    return b;
}

struct axera_q35_layer {
    bool                               gdn = false;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    int                                si = -1, so = -1, ci = -1, co = -1;  // GDN state in/out, swapped every step
    bool                               in_place = false;  // state updated in the llama.cpp recurrent cache tensors
    struct region {
        ggml_backend_axera_buffer_context * b;
        void *                              p;
        size_t                              size;
    };
    std::vector<region>                cache_state;  // those cache ranges, cleared on restart
    struct prefill_io {  // prefill kernel IO: same weights / state / KV as the decode IO
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        int                                si = -1, so = -1, ci = -1, co = -1;
    };
    std::vector<prefill_io>            pre;  // one per prefill chunk size
    std::vector<axera_cmm>             own;                                // converted weights, state, KV cache
    std::map<std::string, std::pair<ggml_backend_axera_buffer_context *, ggml_backend_axera_buffer_context *>> mx;
    std::vector<ggml_backend_axera_buffer_context *> state;                // state buffers to clear on restart
};

struct axera_q35 {
    bool                  ready = false;
    const void *          bound = nullptr;  // data of blk.0.attn_norm.weight the layers were bound to
    int                   L = 0, H = 0, LMAX = 0, BC = 0, ROT = 0;
    double                THETA = 0;
    ggml_axera_kernel *   gdn = nullptr, *attn = nullptr;
    axera_cmm             A, B, POS, NBLK, MASK, COS, SIN;  // A/B: layer input/output ping-pong
    axera_cmm             lm_q, lm_s;                       // MXFP8 output projection
    axera_cmm             lm_bf;                            // bf16 copy of a non-bf16 output projection
    struct prefill {  // one prefill chunk size
        int                 P = 0;
        ggml_axera_kernel * gdn = nullptr, *attn = nullptr;
        axera_cmm           A, B, MASK, COS, SIN;
    };
    std::vector<prefill>  pre;  // largest chunk first
    const ggml_tensor *   lm = nullptr;
    std::map<const ggml_tensor *, axera_cmm> bf16_of;  // bf16 copies of non-bf16 GGUF weights (quantized GGUF)
    std::vector<axera_q35_layer> layers;
    std::vector<float>    inv_freq;
    int64_t               next_pos = 0;
    bool                  rollback = false;  // all GDN state lives in the llama.cpp cache: earlier positions resumable
};

struct ggml_backend_axera_context {
    axera_cmm staging;  // bf16 activation row
    axera_q35 q35;

    ggml_backend_axera_buffer_context * staging_for(size_t size) {
        if (!staging || staging->size < size) {
            staging = axera_cmm_alloc(size);
        }
        return staging.get();
    }
};

// out[0:N] (f32, inside a CMM tensor) = W[N, K] (bf16 at wdata in the CMM buffer wctx) . x[0:K], in N / C GEMV calls
static void axera_gemv_raw(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                           void * wdata, int64_t N, int64_t K, const float * x, ggml_tensor * dst, float * out) {
    const int64_t row = K * (int64_t) sizeof(ggml_bf16_t);
    const int64_t C   = axera_kernels().mul_mat_chunk(N, K);
    GGML_ASSERT(C > 0);

    ggml_axera_kernel * k  = axera_kernels().get(axera_mul_mat_kernel_name(C, K));
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    GGML_ASSERT(k->info->pInputs[ix].nSize == K * sizeof(ggml_bf16_t) &&
                k->info->pInputs[iw].nSize == C * row && k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
    auto * dctx = (ggml_backend_axera_buffer_context *) dst->buffer->context;
    ggml_fp32_to_bf16_row(x, xb, K);
    st->flush(xb, K * sizeof(ggml_bf16_t));
    dctx->flush(out, N * sizeof(float));

    AX_ENGINE_IO_BUFFER_T inputs[2], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, xb, K * sizeof(ggml_bf16_t));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw]    = axera_io(wctx, (char *) wdata + c0 * row, C * row);
        outputs[io]   = axera_io(dctx, out + c0, C * sizeof(float));
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x) for %s, n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, dst->name, N, K);
        }
    }
    dctx->invalidate(out, N * sizeof(float));
}

static void axera_gemv(ggml_backend_axera_context * ctx, const ggml_tensor * w, const float * x, ggml_tensor * dst,
                       float * out) {
    GGML_ASSERT(w->type == GGML_TYPE_BF16 && w->nb[1] == (size_t) w->ne[0] * sizeof(ggml_bf16_t));
    axera_gemv_raw(ctx, (ggml_backend_axera_buffer_context *) w->buffer->context, w->data, w->ne[1], w->ne[0], x, dst,
                   out);
}

// out[0:N] (f32, inside a CMM tensor) = MXFP8 W[N, K] (elements q, scales s) . x[0:K], in N / C GEMV calls
static void axera_gemv_mx8(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                           const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                           ggml_tensor * dst, float * out) {
    const int64_t C = axera_kernels().mul_mat_chunk(N, K, true);
    GGML_ASSERT(C > 0);
    ggml_axera_kernel * k  = axera_kernels().get(axera_mul_mat_kernel_name(C, K, true));
    const int           ix = k->input("x"), iw = k->input("w"), is = k->input("s"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && is >= 0 && io >= 0 && k->info->nInputSize == 3 && k->info->nOutputSize == 1);
    const int64_t gs = K / AXERA_MX_GROUP;
    GGML_ASSERT(k->info->pInputs[iw].nSize == C * K && k->info->pInputs[is].nSize == C * gs &&
                k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
    auto * dctx = (ggml_backend_axera_buffer_context *) dst->buffer->context;
    ggml_fp32_to_bf16_row(x, xb, K);
    st->flush(xb, K * sizeof(ggml_bf16_t));
    dctx->flush(out, N * sizeof(float));

    AX_ENGINE_IO_BUFFER_T inputs[3], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 3;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, xb, K * sizeof(ggml_bf16_t));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw]    = axera_io(q, (char *) q->vir + c0 * K, C * K);
        inputs[is]    = axera_io(sc, (char *) sc->vir + c0 * gs, C * gs);
        outputs[io]   = axera_io(dctx, out + c0, C * sizeof(float));
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, N, K);
        }
    }
    dctx->invalidate(out, N * sizeof(float));
}

static void ggml_backend_axera_mul_mat(ggml_backend_axera_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    // one activation row at a time: the weights are reread per row until a batched kernel exists
    for (int64_t m = 0; m < src1->ne[1]; ++m) {
        axera_gemv(ctx, src0, (const float *) ((const char *) src1->data + m * src1->nb[1]), dst,
                   (float *) ((char *) dst->data + m * dst->nb[1]));
    }
}

static float axera_rms_eps(const ggml_tensor * rms_norm) {
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    return eps;
}

// kernel input -> GGUF tensor of the layer, for the inputs taken as they are
static const std::map<std::string, const char *> axera_q35_direct = {
    { "ln1", "attn_norm.weight" },    { "ln2", "post_attention_norm.weight" }, { "w_qkv", "attn_qkv.weight" },
    { "w_z", "attn_gate.weight" },    { "w_b", "ssm_beta.weight" },            { "w_a", "ssm_alpha.weight" },
    { "dt_bias", "ssm_dt.bias" },     { "norm_w", "ssm_norm.weight" },         { "w_out", "ssm_out.weight" },
    { "w_gate", "ffn_gate.weight" },  { "w_up", "ffn_up.weight" },             { "w_down", "ffn_down.weight" },
    { "w_q", "attn_q.weight" },       { "w_k", "attn_k.weight" },              { "w_v", "attn_v.weight" },
    { "q_norm", "attn_q_norm.weight" }, { "k_norm", "attn_k_norm.weight" },    { "w_o", "attn_output.weight" },
};

// a bf16 view of a GGUF weight for a kernel input of `size` bytes: the tensor itself if it is bf16 in CMM, else a
// cached bf16 copy (quantized GGUF: Q8_0, Q4_K, ...; f16; f32)
static AX_ENGINE_IO_BUFFER_T axera_q35_weight_bf16(axera_q35 & m, const ggml_tensor * w, uint32_t size) {
    if (w->type == GGML_TYPE_BF16 && ggml_nbytes(w) == size && ggml_backend_buffer_is_axera(w->buffer)) {
        return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
    }
    if ((uint64_t) ggml_nelements(w) * sizeof(ggml_bf16_t) != size) {
        GGML_ABORT("qwen35 template: %s (%s, %" PRId64 " elements) does not fit a %u-byte bf16 kernel input\n", w->name,
                   ggml_type_name(w->type), ggml_nelements(w), size);
    }
    auto it = m.bf16_of.find(w);
    if (it == m.bf16_of.end()) {
        it = m.bf16_of.emplace(w, axera_to_bf16(w)).first;
    }
    return axera_io(it->second.get(), it->second->vir, size);
}

// binds one layer's kernel IO: GGUF tensors in CMM directly where the layout matches, converted copies otherwise
static void axera_q35_bind_layer(axera_q35 & m, int il, const std::map<std::string, ggml_tensor *> & t) {
    auto get = [&](const char * name) -> ggml_tensor * {
        char key[96];
        snprintf(key, sizeof(key), "blk.%d.%s", il, name);
        auto it = t.find(key);
        if (it == t.end()) {
            GGML_ABORT("qwen35 template: tensor %s not in the graph\n", key);
        }
        return it->second;
    };
    // llama.cpp recurrent cache of this layer (cache_r_l<il> / cache_s_l<il>), if one sequence row in CMM holds the state
    auto cache = [&](const char * which, uint32_t size) -> ggml_tensor * {
        auto it = t.find("cache_" + std::string(which) + "_l" + std::to_string(il));
        if (it == t.end() || !ggml_backend_buffer_is_axera(it->second->buffer) || it->second->ne[1] != 1 ||
            ggml_nbytes(it->second) < size) {
            return nullptr;
        }
        return it->second;
    };
    axera_q35_layer & ly = m.layers[il];
    ly.gdn               = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
    ggml_axera_kernel * k = ly.gdn ? m.gdn : m.attn;
    ly.in.assign(k->info->nInputSize, {});
    ly.out.assign(k->info->nOutputSize, {});
    ly.own.clear();
    ly.state.clear();
    ly.mx.clear();
    ly.in_place = false;
    ly.cache_state.clear();

    // layer input/output ping-pong: even layers read A and write B, odd layers the other way round
    auto * xin  = (il % 2 == 0 ? m.A : m.B).get();
    auto * xout = (il % 2 == 0 ? m.B : m.A).get();
    auto own = [&](uint32_t size) {
        ly.own.push_back(axera_cmm_alloc(size));
        return ly.own.back().get();
    };

    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        ggml_backend_axera_buffer_context * b = nullptr;
        void *                              p = nullptr;
        if (n == "x") {
            b = xin;
        } else if (n == "pos") {
            b = m.POS.get();
        } else if (n == "nblk") {
            b = m.NBLK.get();
        } else if (n == "mask") {
            b = m.MASK.get();
        } else if (n == "cos") {
            b = m.COS.get();
        } else if (n == "sin") {
            b = m.SIN.get();
        } else if ((n == "state_in" || n == "conv_in") && cache(n == "state_in" ? "s" : "r", size)) {
            // in place in the llama.cpp recurrent cache: its checkpoints save and restore exactly this state
            ggml_tensor * c = cache(n == "state_in" ? "s" : "r", size);
            b               = (ggml_backend_axera_buffer_context *) c->buffer->context;
            p               = c->data;
            ly.in_place     = true;
            ly.cache_state.push_back({ b, p, size });
        } else if (n == "state_in" || n == "conv_in") {
            b = own(size);
            ly.state.push_back(b);
            (n == "state_in" ? ly.si : ly.ci) = (int) i;
        } else if (n == "a_log") {  // GGUF ssm_a = -exp(A_log)
            const ggml_tensor * a = get("ssm_a");
            GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_nbytes(a) == size);
            b = own(size);
            for (int64_t j = 0; j < ggml_nelements(a); ++j) {
                ((float *) b->vir)[j] = logf(-((const float *) a->data)[j]);
            }
            b->flush(b->vir, size);
        } else if (n == "conv_w") {  // GGUF f32 [channels][taps] -> bf16, taps reversed (kernel tap 0 = current token)
            const ggml_tensor * c = get("ssm_conv1d.weight");
            const int64_t       taps = c->ne[0], ch = c->ne[1];
            GGML_ASSERT(c->type == GGML_TYPE_F32 && ch * taps * (int64_t) sizeof(ggml_bf16_t) == size);
            b = own(size);
            for (int64_t r = 0; r < ch; ++r) {
                for (int64_t j = 0; j < taps; ++j) {
                    ((ggml_bf16_t *) b->vir)[r * taps + j] =
                        ggml_fp32_to_bf16(((const float *) c->data)[r * taps + taps - 1 - j]);
                }
            }
            b->flush(b->vir, size);
        } else if ((n.rfind("w_", 0) == 0 || n.rfind("s_", 0) == 0) && k->input(("s_" + n.substr(2)).c_str()) >= 0) {
            // MXFP8 kernel: w_<x> elements and s_<x> scales, quantized from the bf16 GGUF tensor of w_<x>
            const std::string wn = "w_" + n.substr(2);
            auto              mx = ly.mx.find(wn);
            if (mx == ly.mx.end()) {
                auto d = axera_q35_direct.find(wn);
                if (d == axera_q35_direct.end()) {
                    GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", wn.c_str());
                }
                axera_cmm q, sc;
                axera_quantize_mxfp8(get(d->second), q, sc);
                mx = ly.mx.emplace(wn, std::make_pair(q.get(), sc.get())).first;
                ly.own.push_back(std::move(q));
                ly.own.push_back(std::move(sc));
            }
            b = n[0] == 'w' ? mx->second.first : mx->second.second;
            if (b->size < size) {
                GGML_ABORT("qwen35 template: %s is %zu bytes, kernel wants %u\n", n.c_str(), b->size, size);
            }
        } else {
            auto it = axera_q35_direct.find(n);
            if (it == axera_q35_direct.end()) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            ggml_tensor * w = get(it->second);
            if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) {  // quantized / f16 GGUF weight
                ly.in[i] = axera_q35_weight_bf16(m, w, size);
                continue;
            }
            if (ggml_nbytes(w) != size) {
                GGML_ABORT("qwen35 template: %s is %zu bytes, kernel input %s wants %u\n", w->name, ggml_nbytes(w),
                           n.c_str(), size);
            }
            if (ggml_backend_buffer_is_axera(w->buffer)) {
                b = (ggml_backend_axera_buffer_context *) w->buffer->context;
                p = w->data;
            } else {  // not in CMM (layer kept on the CPU): copy
                b = own(size);
                memcpy(b->vir, w->data, size);
                b->flush(b->vir, size);
            }
        }
        if (b->size < size && p == nullptr) {
            GGML_ABORT("qwen35 template: buffer for %s too small\n", n.c_str());
        }
        ly.in[i] = axera_io(b, p ? p : b->vir, size);
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const std::string n    = k->info->pOutputs[i].pName;
        const uint32_t    size = k->info->pOutputs[i].nSize;
        ggml_backend_axera_buffer_context * b;
        if (n == "out") {
            b = xout;
        } else if (ly.in_place && (n == "state_out" || n == "conv_out")) {
            const int j = k->input(n == "state_out" ? "state_in" : "conv_in");
            ly.out[i]   = ly.in[j];  // same buffer as the input
            continue;
        } else {  // state_out / conv_out, or the resident kc / vc
            b = own(size);
            if (n == "state_out" || n == "conv_out") {
                ly.state.push_back(b);
                (n == "state_out" ? ly.so : ly.co) = (int) i;
            }
        }
        ly.out[i] = axera_io(b, b->vir, size);
    }
}

// prefill IO of one layer for chunk size v: the decode IO entries of the same names (weights, recurrent state, KV
// cache), with P-row activations, the chunk mask and the chunk's RoPE tables
static void axera_q35_bind_prefill(axera_q35 & m, int il, size_t v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &        ly = m.layers[il];
    axera_q35::prefill &     pf = m.pre[v];
    ggml_axera_kernel *      k  = ly.gdn ? pf.gdn : pf.attn;
    ggml_axera_kernel *      kd = ly.gdn ? m.gdn : m.attn;
    auto *                   xin  = (il % 2 == 0 ? pf.A : pf.B).get();
    auto *                   xout = (il % 2 == 0 ? pf.B : pf.A).get();
    ly.pre.resize(m.pre.size());
    axera_q35_layer::prefill_io & io = ly.pre[v];
    io.in.assign(k->info->nInputSize, {});
    io.out.assign(k->info->nOutputSize, {});
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        if (n == "x" || n == "mask" || n == "cos" || n == "sin") {
            auto * b  = n == "x" ? xin : n == "mask" ? pf.MASK.get() : n == "cos" ? pf.COS.get() : pf.SIN.get();
            io.in[i]  = axera_io(b, b->vir, size);
            continue;
        }
        const int j = kd->input(n.c_str());
        if (j < 0 || kd->info->pInputs[j].nSize != size) {
            // a bf16 prefill weight next to MXFP8 decode kernels: the GGUF tensor itself
            auto d  = axera_q35_direct.find(n);
            auto it = d == axera_q35_direct.end() ? t.end() : t.find("blk." + std::to_string(il) + "." + d->second);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: prefill input %s matches neither the decode kernel nor a GGUF tensor\n",
                           n.c_str());
            }
            io.in[i] = axera_q35_weight_bf16(m, it->second, size);
            continue;
        }
        io.in[i] = ly.in[j];
        if (n == "state_in") {
            io.si = (int) i;
        } else if (n == "conv_in") {
            io.ci = (int) i;
        }
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const std::string n = k->info->pOutputs[i].pName;
        if (n == "out") {
            io.out[i] = axera_io(xout, xout->vir, k->info->pOutputs[i].nSize);
            continue;
        }
        const int j = kd->output(n.c_str());
        GGML_ASSERT(j >= 0 && kd->info->pOutputs[j].nSize == k->info->pOutputs[i].nSize);
        io.out[i] = ly.out[j];
        if (n == "state_out") {
            io.so = (int) i;
        } else if (n == "conv_out") {
            io.co = (int) i;
        }
    }
}

static void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                           const ggml_tensor * lm) {
    std::ifstream f(axera_kernels().dir + "/qwen35.txt");
    if (!(f >> m.LMAX >> m.BC >> m.ROT >> m.THETA)) {
        GGML_ABORT("qwen35 template: missing %s/qwen35.txt\n", axera_kernels().dir.c_str());
    }
    m.L    = L;
    m.H    = H;
    const bool mx8 = axera_weights_mx8();
    m.gdn  = axera_kernels().get(mx8 ? "qwen35_gdn_mx8" : "qwen35_gdn");
    m.attn = axera_kernels().get(mx8 ? "qwen35_attn_mx8" : "qwen35_attn");
    m.lm   = lm;
    m.lm_q.reset();
    m.lm_s.reset();
    m.lm_bf.reset();
    m.bf16_of.clear();
    if (mx8) {
        axera_quantize_mxfp8(lm, m.lm_q, m.lm_s);
    } else if (lm->type != GGML_TYPE_BF16) {
        m.lm_bf = axera_to_bf16(lm);
    }
    auto size_of = [](ggml_axera_kernel * k, const char * name) {
        const int i = k->input(name);
        GGML_ASSERT(i >= 0);
        return k->info->pInputs[i].nSize;
    };
    GGML_ASSERT(size_of(m.gdn, "x") == (uint32_t) H * sizeof(ggml_bf16_t));
    m.A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.B    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.POS  = axera_cmm_alloc(size_of(m.attn, "pos"));
    m.NBLK = axera_cmm_alloc(size_of(m.attn, "nblk"));
    m.MASK = axera_cmm_alloc(size_of(m.attn, "mask"));
    m.COS  = axera_cmm_alloc(size_of(m.attn, "cos"));
    m.SIN  = axera_cmm_alloc(size_of(m.attn, "sin"));
    GGML_ASSERT(m.MASK->size >= (size_t) m.LMAX * sizeof(ggml_bf16_t));
    m.inv_freq.resize(m.ROT / 2);
    for (int i = 0; i < m.ROT / 2; ++i) {
        m.inv_freq[i] = (float) (1.0 / std::pow(m.THETA, (double) (2 * i) / m.ROT));
    }
    m.layers.clear();
    m.layers.resize(L);
    m.rollback = true;
    for (int il = 0; il < L; ++il) {
        axera_q35_bind_layer(m, il, t);
        if (m.layers[il].gdn && !m.layers[il].in_place) {
            m.rollback = false;
        }
    }
    // prefill kernels (every chunk size found, largest first). Prefill is compute bound, so with MXFP8 decode it uses
    // the bf16 GGUF weights unless GGML_AXERA_PREFILL_WEIGHTS=mxfp8
    const char * pw     = std::getenv("GGML_AXERA_PREFILL_WEIGHTS");
    const bool   pre_mx = mx8 && pw != nullptr && strcmp(pw, "mxfp8") == 0;
    m.pre.clear();
    for (int P : { 64, 32, 16, 8, 4 }) {
        const std::string sfx = std::to_string(P) + (pre_mx ? "_mx8" : "");
        if (!std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx)).good() ||
            !std::ifstream(axera_kernels().path("qwen35_attn_prefill" + sfx)).good()) {
            continue;
        }
        axera_q35::prefill pf;
        pf.P    = P;
        pf.gdn  = axera_kernels().get("qwen35_gdn_prefill" + sfx);
        pf.attn = axera_kernels().get("qwen35_attn_prefill" + sfx);
        GGML_ASSERT(size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
        pf.A    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.B    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
        pf.COS  = axera_cmm_alloc(size_of(pf.attn, "cos"));
        pf.SIN  = axera_cmm_alloc(size_of(pf.attn, "sin"));
        m.pre.push_back(std::move(pf));
    }
    for (size_t v = 0; v < m.pre.size(); ++v) {
        for (int il = 0; il < L; ++il) {
            axera_q35_bind_prefill(m, il, v, t);
        }
    }
    std::string chunks;
    for (auto & pf : m.pre) {
        chunks += (chunks.empty() ? "" : "/") + std::to_string(pf.P);
    }
    if (!chunks.empty()) {
        chunks += pre_mx ? " (MXFP8)" : " (BF16)";
    }
    m.ready = true;
    GGML_LOG_INFO("%s: Axera qwen35 template: %d layers, context %d, %s weights, recurrent state %s, prefill chunks %s\n",
                  __func__, L, m.LMAX, mx8 ? "MXFP8" : "BF16",
                  m.rollback ? "in the llama.cpp cache (checkpoints work)" : "private", chunks.empty() ? "none" : chunks.c_str());
}

static void axera_q35_restart(axera_q35 & m) {
    for (auto & ly : m.layers) {
        for (auto * s : ly.state) {
            memset(s->vir, 0, s->size);
            s->flush(s->vir, s->size);
        }
        for (auto & c : ly.cache_state) {  // state kept in the llama.cpp cache tensors
            memset(c.p, 0, c.size);
            c.b->flush(c.p, c.size);
        }
    }
    auto * mask = (ggml_bf16_t *) m.MASK->vir;
    for (int j = 0; j < m.LMAX; ++j) {
        mask[j] = ggml_fp32_to_bf16(-3e38f);
    }
    m.MASK->flush(m.MASK->vir, m.MASK->size);
    m.next_pos = 0;
}

// the scheduler names its copies of another backend's tensors "<backend>#<name>#<copy>"
static std::string axera_base_name(const char * name) {
    const char * a = strchr(name, '#');
    const char * b = a ? strrchr(name, '#') : nullptr;
    return a && b > a ? std::string(a + 1, b) : std::string(name);
}

static enum ggml_status axera_q35_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
    std::map<std::string, ggml_tensor *> t;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        t.emplace(axera_base_name(node->name), node);
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            ggml_tensor * src = node->src[s];
            t.emplace(axera_base_name(src->name), src);
            for (ggml_tensor * v = src->view_src; v; v = v->view_src) {  // e.g. the cache tensors behind their views
                t.emplace(axera_base_name(v->name), v);
            }
        }
    }
    auto find = [&](const char * name) -> ggml_tensor * {
        auto it = t.find(name);
        return it == t.end() ? nullptr : it->second;
    };
    ggml_tensor * norm0  = find("attn_norm-0");
    ggml_tensor * logits = find("result_output");
    ggml_tensor * inp_pos = find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("qwen35 template: split of %d nodes (%s ... %s) lacks:%s%s%s\n", cgraph->n_nodes,
                       cgraph->n_nodes ? cgraph->nodes[0]->name : "", cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1]->name : "",
                       norm0 ? "" : " attn_norm-0", logits ? "" : " result_output", inp_pos ? "" : " inp_pos");
        GGML_ABORT("qwen35 template: the graph split does not hold the whole model; is the model Qwen3.5 and are all "
                   "layers offloaded (-ngl 99)?\n");
    }

    // layer 0 input: the source of the first RMS norm
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    x0 = x0->src[0];
    const int H = (int) x0->ne[0];
    const int T = (int) x0->ne[1];

    // final norm: the lm head input is [get_rows(] rms_norm(h) * output_norm [, inp_out_ids)]
    ggml_tensor * hnext = logits->src[1];
    while (hnext->op == GGML_OP_GET_ROWS || hnext->op == GGML_OP_VIEW || hnext->op == GGML_OP_RESHAPE) {
        hnext = hnext->src[0];
    }
    GGML_ASSERT(hnext->op == GGML_OP_MUL);
    ggml_tensor * rms = hnext->src[0]->op == GGML_OP_RMS_NORM ? hnext->src[0] : hnext->src[1];
    ggml_tensor * onw = hnext->src[0] == rms ? hnext->src[1] : hnext->src[0];
    GGML_ASSERT(rms->op == GGML_OP_RMS_NORM && onw->type == GGML_TYPE_F32 && onw->ne[0] == H);
    const float eps = axera_rms_eps(rms);
    ggml_tensor * lm = logits->src[0];
    GGML_ASSERT(logits->op == GGML_OP_MUL_MAT && lm->ne[0] == H && ggml_backend_buffer_is_axera(logits->buffer));

    int L = 0;
    while (t.count("blk." + std::to_string(L) + ".attn_norm.weight")) {
        ++L;
    }
    axera_q35 & m = ctx->q35;
    ggml_tensor * w0 = find("blk.0.attn_norm.weight");
    if (!m.ready || m.L != L || m.bound != w0->data || m.lm != lm) {
        axera_q35_init(m, L, H, t, lm);
        m.bound = w0->data;
        axera_q35_restart(m);
    }

    // the layers: chunks of P consecutive positions through the prefill kernels, single tokens through the decode ones
    const int32_t *    pos = (const int32_t *) inp_pos->data;
    std::vector<float> h((size_t) T * H);
    auto begin = [&](int64_t p) {  // accept position p as the next one; false fails the batch
        if (p == 0) {
            axera_q35_restart(m);
        } else if (m.rollback && p < m.next_pos) {
            // back to an earlier position: llama.cpp has restored the recurrent cache (a context checkpoint) to the
            // state after position p - 1, and dropped the later attention cells; hide them from the attention kernel
            auto * mask = (ggml_bf16_t *) m.MASK->vir;
            for (int64_t j = p; j < m.next_pos; ++j) {
                mask[j] = ggml_fp32_to_bf16(-3e38f);
            }
            m.next_pos = p;
        }
        if (p != m.next_pos || p >= m.LMAX) {
            // fail the batch, not the process: the next request starting at position 0 recovers
            GGML_LOG_ERROR("qwen35 template: position %" PRId64 " (expected %" PRId64 ", context %d). The template runs a "
                           "single sequence with consecutive positions; with llama-server use -np 1 --cache-ram 0%s\n",
                           p, m.next_pos, m.LMAX, m.rollback ? "" : " --ctx-checkpoints 0");
            m.next_pos = -1;  // only a restart at position 0 is accepted now
            return false;
        }
        return true;
    };
    auto rope_rows = [&](int64_t p0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) {
        auto * c = (ggml_bf16_t *) cb->vir;
        auto * s = (ggml_bf16_t *) sb->vir;
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < m.ROT / 2; ++i) {
                const float a             = (float) (p0 + r) * m.inv_freq[i];
                c[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::cos(a));
                s[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::sin(a));
            }
        }
        cb->flush(cb->vir, cb->size);
        sb->flush(sb->vir, sb->size);
    };
    auto run_layers = [&](int v, int64_t p) {  // v: prefill chunk size index, -1: decode
        const bool prefill = v >= 0;
        for (int il = 0; il < L; ++il) {
            axera_q35_layer & ly = m.layers[il];
            auto & in  = prefill ? ly.pre[v].in : ly.in;
            auto & out = prefill ? ly.pre[v].out : ly.out;
            if (prefill && ly.si >= 0) {  // private ping-pong state: take the current buffers
                in[ly.pre[v].si]  = ly.in[ly.si];
                in[ly.pre[v].ci]  = ly.in[ly.ci];
                out[ly.pre[v].so] = ly.out[ly.so];
                out[ly.pre[v].co] = ly.out[ly.co];
            }
            AX_ENGINE_IO_T io;
            memset(&io, 0, sizeof(io));
            io.pInputs     = in.data();
            io.nInputSize  = (uint32_t) in.size();
            io.pOutputs    = out.data();
            io.nOutputSize = (uint32_t) out.size();
            ggml_axera_kernel * k = ly.gdn ? (prefill ? m.pre[v].gdn : m.gdn) : (prefill ? m.pre[v].attn : m.attn);
            const int ret = AX_ENGINE_RunSync(k->handle, &io);
            if (ret != 0) {
                GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d, position %" PRId64 "%s\n", ret, il, p,
                           prefill ? " (prefill)" : "");
            }
            if (ly.si >= 0) {
                std::swap(ly.in[ly.si], ly.out[ly.so]);
                std::swap(ly.in[ly.ci], ly.out[ly.co]);
            }
        }
    };

    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        if (!begin(p)) {
            return GGML_STATUS_FAILED;
        }
        int v = -1;  // the largest prefill chunk that fits the remaining consecutive positions
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            const int P  = m.pre[c].P;
            bool      ok = it + P <= T && p + P <= m.LMAX;
            for (int j = 1; ok && j < P; ++j) {
                ok = pos[it + j] == p + j;
            }
            v = ok ? (int) c : -1;
        }
        if (v >= 0) {
            axera_q35::prefill & pf = m.pre[v];
            const int            P  = pf.P;
            for (int r = 0; r < P; ++r) {
                ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (it + r) * x0->nb[1]),
                                      (ggml_bf16_t *) pf.A->vir + (size_t) r * H, H);
            }
            pf.A->flush(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rope_rows(p, P, pf.COS.get(), pf.SIN.get());
            const int nblk = (int) ((p + P - 1) / m.BC + 1);
            *(int32_t *) m.POS->vir  = (int32_t) p;
            *(int32_t *) m.NBLK->vir = nblk;
            // chunk mask [LMAX / BC][R][BC], row r = head * P + token: positions up to p + token
            const int64_t R    = (int64_t) (pf.MASK->size / sizeof(ggml_bf16_t)) / m.LMAX;
            auto *        mask = (ggml_bf16_t *) pf.MASK->vir;
            const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f), ninf = ggml_fp32_to_bf16(-3e38f);
            for (int64_t blk = 0; blk < nblk; ++blk) {
                for (int64_t r = 0; r < R; ++r) {
                    const int64_t lim = p + r % P;
                    ggml_bf16_t * row = mask + (blk * R + r) * m.BC;
                    for (int c = 0; c < m.BC; ++c) {
                        row[c] = blk * m.BC + c <= lim ? zero : ninf;
                    }
                }
            }
            for (int r = 0; r < P; ++r) {
                ((ggml_bf16_t *) m.MASK->vir)[p + r] = zero;  // for the decode steps after the chunk
            }
            pf.MASK->flush(pf.MASK->vir, (size_t) nblk * R * m.BC * sizeof(ggml_bf16_t));
            m.MASK->flush(m.MASK->vir, m.MASK->size);
            m.POS->flush(m.POS->vir, m.POS->size);
            m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            run_layers(v, p);
            auto * fin = (L % 2 == 0 ? pf.A : pf.B).get();
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, (int64_t) P * H);
            m.next_pos = p + P;
            it += P;
            continue;
        }
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + it * x0->nb[1]), (ggml_bf16_t *) m.A->vir, H);
        m.A->flush(m.A->vir, H * sizeof(ggml_bf16_t));
        rope_rows(p, 1, m.COS.get(), m.SIN.get());
        *(int32_t *) m.POS->vir  = (int32_t) p;
        *(int32_t *) m.NBLK->vir = (int32_t) (p / m.BC + 1);
        ((ggml_bf16_t *) m.MASK->vir)[p] = ggml_fp32_to_bf16(0.0f);
        m.POS->flush(m.POS->vir, m.POS->size);
        m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
        m.MASK->flush(m.MASK->vir, m.MASK->size);
        run_layers(-1, p);
        auto * fin = (L % 2 == 0 ? m.A : m.B).get();  // where the last layer writes
        fin->invalidate(fin->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, H);
        m.next_pos = p + 1;
        ++it;
    }

    // final norm and output projection for the requested rows: the row ids are the GET_ROWS index that crops the
    // last layer or the final norm to the outputs (by structure: the scheduler may have renamed inp_out_ids)
    ggml_tensor *   rnorm   = find("result_norm");
    const int       n_out   = (int) logits->ne[1];
    ggml_tensor *   out_ids = nullptr;
    for (int i = 0; i < cgraph->n_nodes && n_out < T; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
            node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
            out_ids = node->src[1];
        }
    }
    if (n_out < T && !out_ids) {
        GGML_ABORT("qwen35 template: %d of %d rows requested but no output row index in the graph\n", n_out, T);
    }
    std::vector<float> hn(H);
    const float *   g = (const float *) onw->data;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        const float * x  = h.data() + (size_t) row * H;
        double        ss = 0;
        for (int i = 0; i < H; ++i) {
            ss += (double) x[i] * x[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
        for (int i = 0; i < H; ++i) {
            hn[i] = x[i] * scale * g[i];
        }
        if (rnorm && rnorm->type == GGML_TYPE_F32 && rnorm->ne[0] == H && rnorm->ne[1] == n_out) {
            memcpy((char *) rnorm->data + r * rnorm->nb[1], hn.data(), H * sizeof(float));
        }
        float * dst = (float *) ((char *) logits->data + r * logits->nb[1]);
        if (m.lm_q) {
            axera_gemv_mx8(ctx, m.lm_q.get(), m.lm_s.get(), lm->ne[1], H, hn.data(), logits, dst);
        } else if (m.lm_bf) {
            axera_gemv_raw(ctx, m.lm_bf.get(), m.lm_bf->vir, lm->ne[1], H, hn.data(), logits, dst);
        } else {
            axera_gemv(ctx, lm, hn.data(), logits, dst);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_axera_get_name(ggml_backend_t backend) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_axera_free(ggml_backend_t backend) {
    delete (ggml_backend_axera_context *) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_axera_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_axera_context *) backend->context;

    if (axera_template_qwen35()) {
        return axera_q35_compute(ctx, cgraph);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_axera_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i axera_backend_i = {
    /* .get_name                = */ ggml_backend_axera_get_name,
    /* .free                    = */ ggml_backend_axera_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_axera_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_axera_guid(void) {
    static ggml_guid guid = { 0x61, 0x78, 0x65, 0x72, 0x61, 0x2d, 0x6e, 0x70, 0x75, 0x2d, 0x38, 0x38, 0x36, 0x30, 0x00, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_axera_init(void) {
    if (!axera_runtime_init()) {
        return nullptr;
    }
    return new ggml_backend{
        /* .guid    = */ ggml_backend_axera_guid(),
        /* .iface   = */ axera_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_axera_reg(), 0),
        /* .context = */ new ggml_backend_axera_context,
    };
}

bool ggml_backend_is_axera(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_axera_guid());
}

//
// device
//

static const char * ggml_backend_axera_device_get_name(ggml_backend_dev_t dev) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_axera_device_get_description(ggml_backend_dev_t dev) {
    return "Axera NPU";
    GGML_UNUSED(dev);
}

static void ggml_backend_axera_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    axera_cmm_memory(free, total);
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_axera_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;  // device memory is host-visible CMM
    GGML_UNUSED(dev);
}

static void ggml_backend_axera_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_axera_device_get_name(dev);
    props->description = ggml_backend_axera_device_get_description(dev);
    props->type        = ggml_backend_axera_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_axera_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_axera_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_axera_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_axera_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_axera_buffer_type();
    GGML_UNUSED(dev);
}

static bool ggml_backend_axera_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (axera_template_qwen35()) {
        // the template runs the whole graph; weights must still be in CMM to be offloaded
        return true;
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            // bf16 weights resident in CMM, f32 activation rows (any count: llama.cpp probes weight placement with a
            // batch of rows, and the weights must be in CMM for decode)
            return ggml_backend_buffer_is_axera(src0->buffer) && src0->type == GGML_TYPE_BF16 &&
                   src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) && ggml_is_contiguous(op) && src0->ne[2] == 1 && src0->ne[3] == 1 &&
                   src1->ne[2] == 1 && src1->ne[3] == 1 && axera_kernels().mul_mat_chunk(src0->ne[1], src0->ne[0]) > 0;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_axera_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // NPU kernels need physical addresses: only CMM buffers. Activations from other host buffers are copied in by the
    // scheduler; the CPU backend in turn uses CMM buffers directly (they are host buffers).
    return buft == ggml_backend_axera_buffer_type();
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_axera_device_i = {
    /* .get_name             = */ ggml_backend_axera_device_get_name,
    /* .get_description      = */ ggml_backend_axera_device_get_description,
    /* .get_memory           = */ ggml_backend_axera_device_get_memory,
    /* .get_type             = */ ggml_backend_axera_device_get_type,
    /* .get_props            = */ ggml_backend_axera_device_get_props,
    /* .init_backend         = */ ggml_backend_axera_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_axera_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_axera_device_supports_op,
    /* .supports_buft        = */ ggml_backend_axera_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_axera_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_axera_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_axera_buffer_type_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ ggml_backend_axera_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_axera_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ ggml_backend_axera_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_axera_reg(), 0),
        /* .context = */ NULL,
    };
    return &buft;
}

//
// reg
//

static const char * ggml_backend_axera_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_axera_reg_get_device_count(ggml_backend_reg_t reg) {
    return axera_runtime_init() ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_axera_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_axera_device = {
        /* .iface   = */ ggml_backend_axera_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_axera_device;
    GGML_UNUSED(index);
}

static void * ggml_backend_axera_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_axera_reg_i = {
    /* .get_name         = */ ggml_backend_axera_reg_get_name,
    /* .get_device_count = */ ggml_backend_axera_reg_get_device_count,
    /* .get_device       = */ ggml_backend_axera_reg_get_device,
    /* .get_proc_address = */ ggml_backend_axera_get_proc_address,
};

ggml_backend_reg_t ggml_backend_axera_reg(void) {
    static struct ggml_backend_reg ggml_backend_axera_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_axera_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_axera_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_axera_reg)
