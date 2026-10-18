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
    std::map<std::string, std::set<int64_t>>                  ctx;  // <name>_c<LMAX>.axmodel: name -> contexts
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
                const std::string n  = f->d_name;
                const size_t      us = n.rfind("_c");
                if (us != std::string::npos && sscanf(n.c_str() + us, "_c%lld%15s", &c, tail) == 2 &&
                    strcmp(tail, ".axmodel") == 0) {
                    ctx[n.substr(0, us)].insert(c);
                }
            }
            closedir(d);
        }
    }

    std::string path(const std::string & name) const { return dir + "/" + name + ".axmodel"; }

    // the smallest context >= n among the <name>_c<LMAX> kernels, 0 if none
    int64_t ctx_for(const std::string & name, int64_t n) const {
        auto it = ctx.find(name);
        if (it == ctx.end()) {
            return 0;
        }
        auto c = it->second.lower_bound(n);
        return c == it->second.end() ? 0 : *c;
    }

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
// Kernels in GGML_AXERA_KERNEL_DIR: qwen35_gdn.axmodel, qwen35_attn.axmodel and the mul_mat GEMV kernel for the output
// projection. The attention kernels' mask input [LMAX / BC, BC] gives their context LMAX and KV block BC; the RoPE
// parameters come from the graph's ROPE nodes. On the llama.cpp KV cache the attention kernels are
// qwen35_attn*_kvc_c<LMAX> (several contexts may sit side by side); the smallest LMAX >= -c runs.
//

static bool axera_template_qwen35() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_TEMPLATE");
        return e != nullptr && strcmp(e, "qwen35") == 0;
    }();
    return on;
}

// GGML_AXERA_FLASH=1: split attention (flash decoding) even when a fused attention kernel holds the KV cache
static bool axera_flash_forced() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_FLASH");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return on;
}

// KV segment (cells) of the split attention kernels qwen35_attn_fd_seg<R>_c<S>, 0 if there are none
static int64_t axera_flash_seg() {
    return axera_kernels().ctx_for("qwen35_attn_fd_seg1", 0);
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
    struct region {  // a state input in the llama.cpp recurrent cache: one row per sequence
        ggml_backend_axera_buffer_context * b;
        char *                              base;    // row 0
        size_t                              size;    // bytes the kernel uses
        size_t                              stride;  // bytes per row
        int                                 in = -1, out = -1;  // decode IO indices (state_in/out or conv_in/out)
        bool                                conv = false;
    };
    std::vector<region>                cache_state;  // those cache ranges
    int64_t                            row = 0;      // the cache row (sequence) the IO points at
    struct kv_region {  // the llama.cpp KV cache of an attention layer (kc / vc output): one block per stream
        ggml_backend_axera_buffer_context * b;
        char *                              base;    // stream 0
        size_t                              size;    // the kernel's IO size (LMAX cells)
        size_t                              stride;  // bytes per stream
        size_t                              row = 0;  // bytes per cell
        int                                 out = -1;  // -1: split attention (host writes, segment IO)
        bool                                v = false;
    };
    std::vector<kv_region>             kv;
    int64_t                            stream = 0;  // the KV stream the IO points at
    struct prefill_io {  // prefill / batched decode kernel IO: same weights / state / KV as the decode IO
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        int                                si = -1, so = -1, ci = -1, co = -1, ko = -1, vo = -1;
        bool                               batch = false;  // state IO over P consecutive cache rows
    };
    std::vector<prefill_io>            pre;  // one per prefill chunk size
    struct fd_io {  // split attention IO of one run kind: head, segment and tail kernels
        std::vector<AX_ENGINE_IO_BUFFER_T> hin, hout, sin, sout, tin, tout;
        int                                kc = -1, vc = -1, mask = -1;  // segment inputs, rebound per call
    };
    std::vector<fd_io>                 fd;  // [0]: decode, [1 + v]: m.pre[v]
    std::vector<axera_cmm>             own;                                // converted weights, state, KV cache
    std::map<std::string, std::pair<ggml_backend_axera_buffer_context *, ggml_backend_axera_buffer_context *>> mx;
    std::vector<ggml_backend_axera_buffer_context *> state;                // state buffers to clear on restart
};

struct axera_q35 {
    bool                  ready = false;
    const void *          bound = nullptr;  // data of blk.0.attn_norm.weight the layers were bound to
    int                   L = 0, H = 0, LMAX = 0, BC = 0, ROT = 0;
    int64_t               N = 0;  // KV cells in use: the llama.cpp cache size (<= LMAX), else LMAX
    double                THETA = 0;
    float                 freq_scale = 1.0f;
    ggml_axera_kernel *   gdn = nullptr, *attn = nullptr;
    axera_cmm             A, B, POS, NBLK, MASK, COS, SIN;  // A/B: layer input/output ping-pong
    axera_cmm             lm_q, lm_s;                       // MXFP8 output projection
    axera_cmm             lm_bf;                            // bf16 copy of a non-bf16 output projection
    struct prefill {  // one prefill chunk size, or (batch) one batched decode size
        int                 P = 0;
        int64_t             R = 0;  // mask rows per KV block
        bool                batch = false;  // P tokens of P consecutive sequences (recurrent rows), one each
        ggml_axera_kernel * gdn = nullptr, *attn = nullptr;
        axera_cmm           A, B, MASK, COS, SIN, POS;  // POS: the batch's P cells
    };
    std::vector<prefill>  pre;  // prefill chunks, then batched decode sizes, largest first
    const ggml_tensor *   lm = nullptr;
    std::map<const ggml_tensor *, axera_cmm> bf16_of;  // bf16 copies of non-bf16 GGUF weights (quantized GGUF)
    std::vector<axera_q35_layer> layers;
    std::vector<float>    inv_freq;
    int64_t               next_pos = 0;
    bool                  rollback = false;  // all GDN state lives in the llama.cpp cache: earlier positions resumable
    bool                  kv_llama = false;  // attention kernels on the llama.cpp KV cache (cells, f16) and its KQ mask
    std::string           ctx_sfx;           // attention kernel name suffix: "_kvc_c<LMAX>" on the llama.cpp KV cache
    bool                  flash = false;     // attention as head / per-segment / tail kernels (flash decoding)
    int64_t               S     = 0;         // their KV segment, cells
    struct fd_kernels {  // split attention kernels for R rows
        ggml_axera_kernel *    head = nullptr, *seg = nullptr, *tail = nullptr;
        int                    hkv = 0, gr = 0, d = 0;  // segment outputs [hkv][gr][d]: KV head, row g * R + token
        axera_cmm              Q, GATE, K, V, M, Lsum, O, ATT;
        std::vector<axera_cmm> MASK;  // per segment [S / BC][R][BC], filled once per batch for all layers
        std::vector<int32_t>   nblk;  // per segment: blocks to scan, 0 if no token sees a cell of it
        std::vector<uint8_t>   seen;  // [segment][token]
    };
    std::map<int, fd_kernels> fd;  // by rows R
};

struct ggml_backend_axera_context {
    axera_cmm staging;    // bf16 activation row
    axera_cmm batch_out;  // f32 output rows of a batched GEMV call
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

// rows out[b][0:N] = W[N, K] (bf16 at wdata in the CMM buffer wctx) . x[b], b < Bn, through the batched GEMV
// mul_mat_bf16_b<Bn>_n<C>_k<K> (x [Bn, K] bf16 -> out [Bn, C] f32), N / C calls; false if that kernel is missing
static bool axera_gemv_batch(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx, void * wdata,
                             int64_t N, int64_t K, const float * x, int Bn, float * const * out) {
    const int64_t C = axera_kernels().mul_mat_chunk(N, K);
    char          name[96];
    snprintf(name, sizeof(name), "mul_mat_bf16_b%d_n%" PRId64 "_k%" PRId64, Bn, C, K);
    static std::map<std::string, bool> found;
    auto f = found.find(name);
    if (f == found.end()) {
        f = found.emplace(name, C > 0 && std::ifstream(axera_kernels().path(name)).good()).first;
    }
    if (!f->second) {
        return false;
    }
    ggml_axera_kernel * k  = axera_kernels().get(name);
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    const int64_t       row = K * (int64_t) sizeof(ggml_bf16_t);
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->pInputs[ix].nSize == Bn * row &&
                k->info->pInputs[iw].nSize == C * row && k->info->pOutputs[io].nSize == Bn * C * sizeof(float));
    auto * st = ctx->staging_for(Bn * row);
    ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) st->vir, Bn * K);
    st->flush(st->vir, Bn * row);
    if (!ctx->batch_out || ctx->batch_out->size < Bn * C * sizeof(float)) {
        ctx->batch_out = axera_cmm_alloc(Bn * C * sizeof(float));
    }
    auto * bo = ctx->batch_out.get();
    AX_ENGINE_IO_BUFFER_T inputs[2], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, st->vir, Bn * row);
    outputs[io]        = axera_io(bo, bo->vir, Bn * C * sizeof(float));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw]    = axera_io(wctx, (char *) wdata + c0 * row, C * row);
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 " rows=%d\n", __func__, ret, N, K, Bn);
        }
        bo->invalidate(bo->vir, Bn * C * sizeof(float));
        for (int b = 0; b < Bn; ++b) {
            memcpy(out[b] + c0, (const float *) bo->vir + (size_t) b * C, C * sizeof(float));
        }
    }
    return true;
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

static uint32_t axera_input_size(ggml_axera_kernel * k, const char * name) {
    const int i = k->input(name);
    GGML_ASSERT(i >= 0);
    return k->info->pInputs[i].nSize;
}

static uint32_t axera_output_size(ggml_axera_kernel * k, const char * name) {
    const int i = k->output(name);
    GGML_ASSERT(i >= 0);
    return k->info->pOutputs[i].nSize;
}

// kernel input n of `size` bytes from GGUF tensor w: in CMM as it is, a bf16 copy of a quantized / f16 weight, or a
// CMM copy (owned by the layer) of a tensor kept on the CPU
static AX_ENGINE_IO_BUFFER_T axera_q35_gguf_input(axera_q35 & m, axera_q35_layer & ly, const ggml_tensor * w,
                                                  const std::string & n, uint32_t size) {
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) {  // quantized / f16 GGUF weight
        return axera_q35_weight_bf16(m, w, size);
    }
    if (ggml_nbytes(w) != size) {
        GGML_ABORT("qwen35 template: %s is %zu bytes, kernel input %s wants %u\n", w->name, ggml_nbytes(w), n.c_str(),
                   size);
    }
    if (ggml_backend_buffer_is_axera(w->buffer)) {
        return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
    }
    ly.own.push_back(axera_cmm_alloc(size));  // not in CMM (layer kept on the CPU): copy
    auto * b = ly.own.back().get();
    memcpy(b->vir, w->data, size);
    b->flush(b->vir, size);
    return axera_io(b, b->vir, size);
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
        if (it == t.end() || !ggml_backend_buffer_is_axera(it->second->buffer) || it->second->nb[1] < size) {
            return nullptr;
        }
        return it->second;
    };
    axera_q35_layer & ly = m.layers[il];
    ly.gdn               = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
    ly.own.clear();
    ly.state.clear();
    ly.mx.clear();
    ly.in_place = false;
    ly.cache_state.clear();
    ly.row = 0;
    ly.kv.clear();
    ly.stream = 0;
    ly.fd.clear();
    if (!ly.gdn && m.flash) {  // split attention: axera_q35_bind_flash
        ly.in.clear();
        ly.out.clear();
        return;
    }
    ggml_axera_kernel * k = ly.gdn ? m.gdn : m.attn;
    ly.in.assign(k->info->nInputSize, {});
    ly.out.assign(k->info->nOutputSize, {});

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
            axera_q35_layer::region rg;
            rg.b      = b;
            rg.base   = (char *) c->data;
            rg.size   = size;
            rg.stride = c->nb[1];
            rg.in     = (int) i;
            rg.conv   = n == "conv_in";
            ly.cache_state.push_back(rg);
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
            ly.in[i] = axera_q35_gguf_input(m, ly, get(it->second), n, size);
            continue;
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
        } else if (m.kv_llama && (n == "kc" || n == "vc")) {
            // the llama.cpp KV cache of this layer: [cells][heads x dim] f16, the layout of the _kvc kernels
            // N cells per stream, allocated with room for LMAX after the last stream (the IO size must be the
            // kernel's); the kernels only touch the blocks below the runtime block count, all within the stream
            ggml_tensor * c = t.at("cache_" + std::string(n == "kc" ? "k" : "v") + "_l" + std::to_string(il));
            GGML_ASSERT(c->nb[1] * m.LMAX == size &&
                        ggml_backend_buft_get_alloc_size(c->buffer->buft, c) >= (c->ne[2] - 1) * c->nb[2] + size);
            axera_q35_layer::kv_region kr;
            kr.b      = (ggml_backend_axera_buffer_context *) c->buffer->context;
            kr.base   = (char *) c->data;
            kr.size   = size;
            kr.stride = c->nb[2];
            kr.out    = (int) i;
            kr.v      = n == "vc";
            ly.kv.push_back(kr);
            ly.out[i] = axera_io(kr.b, kr.base, size);
            continue;
        } else if (ly.in_place && (n == "state_out" || n == "conv_out")) {
            const int j = k->input(n == "state_out" ? "state_in" : "conv_in");
            ly.out[i]   = ly.in[j];  // same buffer as the input
            for (auto & rg : ly.cache_state) {
                if (rg.in == j) {
                    rg.out = (int) i;
                }
            }
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
    if (!ly.gdn && m.flash) {  // split attention: axera_q35_bind_flash
        ly.pre.resize(m.pre.size());
        ly.pre[v].batch = pf.batch;
        return;
    }
    ggml_axera_kernel *      k  = ly.gdn ? pf.gdn : pf.attn;
    ggml_axera_kernel *      kd = ly.gdn ? m.gdn : m.attn;
    auto *                   xin  = (il % 2 == 0 ? pf.A : pf.B).get();
    auto *                   xout = (il % 2 == 0 ? pf.B : pf.A).get();
    ly.pre.resize(m.pre.size());
    axera_q35_layer::prefill_io & io = ly.pre[v];
    io.in.assign(k->info->nInputSize, {});
    io.out.assign(k->info->nOutputSize, {});
    io.batch = pf.batch;
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        if (n == "x" || n == "mask" || n == "cos" || n == "sin" || (pf.batch && n == "pos")) {
            auto * b  = n == "x" ? xin : n == "mask" ? pf.MASK.get() : n == "cos" ? pf.COS.get() : n == "sin" ? pf.SIN.get() :
                                                                                                    pf.POS.get();
            io.in[i]  = axera_io(b, b->vir, size);
            continue;
        }
        if (pf.batch && (n == "state_in" || n == "conv_in")) {
            // P consecutive rows of the llama.cpp recurrent cache, from row 0 until axera_q35_set_batch_rows
            const int j = kd->input(n.c_str());
            for (auto & rg : ly.cache_state) {
                if (rg.in == j) {
                    GGML_ASSERT(size == pf.P * rg.stride);
                    io.in[i] = axera_io(rg.b, rg.base, size);
                }
            }
            (n == "state_in" ? io.si : io.ci) = (int) i;
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
        if (pf.batch && (n == "state_out" || n == "conv_out")) {  // in place
            const int si = n == "state_out" ? io.si : io.ci;
            GGML_ASSERT(si >= 0);
            io.out[i] = io.in[si];
            (n == "state_out" ? io.so : io.co) = (int) i;
            continue;
        }
        const int j = kd->output(n.c_str());
        GGML_ASSERT(j >= 0 && kd->info->pOutputs[j].nSize == k->info->pOutputs[i].nSize);
        io.out[i] = ly.out[j];
        if (n == "state_out") {
            io.so = (int) i;
        } else if (n == "conv_out") {
            io.co = (int) i;
        } else if (n == "kc") {
            io.ko = (int) i;
        } else if (n == "vc") {
            io.vo = (int) i;
        }
    }
}

// LMAX and BC of an attention kernel, from its mask input [LMAX / BC, BC]
static void axera_q35_context(ggml_axera_kernel * k, int & LMAX, int & BC) {
    const int i = k->input("mask");
    GGML_ASSERT(i >= 0);
    const AX_ENGINE_IOMETA_T & mt = k->info->pInputs[i];
    if (mt.nShapeSize != 2 || mt.nSize != (uint32_t) (mt.pShape[0] * mt.pShape[1]) * sizeof(ggml_bf16_t)) {
        GGML_ABORT("qwen35 template: attention kernel mask input is not [LMAX / BC, BC] bf16\n");
    }
    LMAX = mt.pShape[0] * mt.pShape[1];
    BC   = mt.pShape[1];
}

// the split attention kernels for R rows (qwen35_attn_fd_head<R>, _seg<R>_c<S>, _tail<R>) and their buffers
static bool axera_q35_load_flash(axera_q35 & m, int R) {
    if (m.fd.count(R)) {
        return true;
    }
    const std::string r  = std::to_string(R);
    const std::string hn = "qwen35_attn_fd_head" + r, sn = "qwen35_attn_fd_seg" + r + "_c" + std::to_string(m.S),
                      tn = "qwen35_attn_fd_tail" + r;
    for (const auto & n : { hn, sn, tn }) {
        if (!std::ifstream(axera_kernels().path(n)).good()) {
            return false;
        }
    }
    axera_q35::fd_kernels fk;
    fk.head = axera_kernels().get(hn);
    fk.seg  = axera_kernels().get(sn);
    fk.tail = axera_kernels().get(tn);
    const AX_ENGINE_IOMETA_T & o = fk.seg->info->pOutputs[fk.seg->output("o_out")];
    GGML_ASSERT(o.nShapeSize == 3);
    fk.hkv  = o.pShape[0];
    fk.gr   = o.pShape[1];
    fk.d    = o.pShape[2];
    fk.Q    = axera_cmm_alloc(axera_output_size(fk.head, "q_out"));
    fk.GATE = axera_cmm_alloc(axera_output_size(fk.head, "gate_out"));
    fk.K    = axera_cmm_alloc(axera_output_size(fk.head, "k_out"));
    fk.V    = axera_cmm_alloc(axera_output_size(fk.head, "v_out"));
    fk.M    = axera_cmm_alloc(axera_output_size(fk.seg, "m_out"));
    fk.Lsum = axera_cmm_alloc(axera_output_size(fk.seg, "l_out"));
    fk.O    = axera_cmm_alloc(axera_output_size(fk.seg, "o_out"));
    fk.ATT  = axera_cmm_alloc(axera_input_size(fk.tail, "attn"));
    m.fd.emplace(R, std::move(fk));
    return true;
}

// split attention IO of attention layer il for run kind v (-1: decode, else m.pre[v]): head and tail on the run's
// activations and RoPE tables and the GGUF weights; the segment kernel's KV and mask inputs are rebound per call
static void axera_q35_bind_flash(axera_q35 & m, int il, int v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &       ly = m.layers[il];
    const int               R  = v < 0 ? 1 : m.pre[v].P;
    axera_q35::fd_kernels & fk = m.fd.at(R);
    auto & A   = v < 0 ? m.A : m.pre[v].A;
    auto & B   = v < 0 ? m.B : m.pre[v].B;
    auto * xin = (il % 2 == 0 ? A : B).get(), *xout = (il % 2 == 0 ? B : A).get();
    ly.fd.resize(m.pre.size() + 1);
    axera_q35_layer::fd_io & io = ly.fd[v + 1];
    if (ly.kv.empty()) {  // the layer's KV cache: the host writes new cells, the segment kernel reads S cells
        for (const char * kv : { "k", "v" }) {
            const ggml_tensor * c = t.at("cache_" + std::string(kv) + "_l" + std::to_string(il));
            const int64_t       n = (m.N + m.S - 1) / m.S * m.S;
            GGML_ASSERT(ggml_backend_buft_get_alloc_size(c->buffer->buft, c) >= (c->ne[2] - 1) * c->nb[2] + n * c->nb[1]);
            axera_q35_layer::kv_region kr;
            kr.b      = (ggml_backend_axera_buffer_context *) c->buffer->context;
            kr.base   = (char *) c->data;
            kr.size   = m.S * c->nb[1];
            kr.stride = c->nb[2];
            kr.row    = c->nb[1];
            kr.v      = kv[0] == 'v';
            ly.kv.push_back(kr);
        }
    }
    using buf = std::pair<ggml_backend_axera_buffer_context *, char *>;
    auto bind = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                    const std::map<std::string, buf> & bufs) {
        in.assign(k->info->nInputSize, {});
        out.assign(k->info->nOutputSize, {});
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            const std::string n    = k->info->pInputs[i].pName;
            const uint32_t    size = k->info->pInputs[i].nSize;
            auto              b    = bufs.find(n);
            if (b != bufs.end()) {
                in[i] = axera_io(b->second.first, b->second.second, size);
                continue;
            }
            auto d  = axera_q35_direct.find(n);
            auto it = d == axera_q35_direct.end() ? t.end() : t.find("blk." + std::to_string(il) + "." + d->second);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            in[i] = axera_q35_gguf_input(m, ly, it->second, n, size);
        }
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            const auto & b = bufs.at(k->info->pOutputs[i].pName);
            out[i]         = axera_io(b.first, b.second, k->info->pOutputs[i].nSize);
        }
    };
    auto at   = [](const axera_cmm & c) { return buf(c.get(), (char *) c->vir); };
    auto & cs = v < 0 ? m.COS : m.pre[v].COS;
    auto & sn = v < 0 ? m.SIN : m.pre[v].SIN;
    bind(fk.head, io.hin, io.hout,
         { { "x", buf(xin, (char *) xin->vir) }, { "cos", at(cs) }, { "sin", at(sn) }, { "q_out", at(fk.Q) },
           { "gate_out", at(fk.GATE) }, { "k_out", at(fk.K) }, { "v_out", at(fk.V) } });
    const auto & kk = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
    const auto & vv = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
    bind(fk.seg, io.sin, io.sout,
         { { "qs", at(fk.Q) }, { "kc", buf(kk.b, kk.base) }, { "vc", buf(vv.b, vv.base) }, { "nblk", at(m.NBLK) },
           { "mask", at(fk.Q) }, { "m_out", at(fk.M) }, { "l_out", at(fk.Lsum) }, { "o_out", at(fk.O) } });
    io.kc   = fk.seg->input("kc");
    io.vc   = fk.seg->input("vc");
    io.mask = fk.seg->input("mask");
    bind(fk.tail, io.tin, io.tout,
         { { "x", buf(xin, (char *) xin->vir) }, { "attn", at(fk.ATT) }, { "gate", at(fk.GATE) },
           { "out", buf(xout, (char *) xout->vir) } });
}

static void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                           const ggml_tensor * lm, const ggml_tensor * rope) {
    m.L    = L;
    m.H    = H;
    const bool mx8 = axera_weights_mx8();
    // RoPE of the graph: rotated dims, base and linear position scale (YaRN is not implemented by the template)
    int32_t rp[16];
    memcpy(rp, rope->op_params, sizeof(rp));
    float base, scale, ext;
    memcpy(&base, rp + 5, sizeof(float));
    memcpy(&scale, rp + 6, sizeof(float));
    memcpy(&ext, rp + 7, sizeof(float));
    if (ext != 0.0f) {
        GGML_ABORT("qwen35 template: YaRN RoPE scaling is not supported\n");
    }
    m.ROT        = rp[1];
    m.THETA      = base;
    m.freq_scale = scale;
    // the llama.cpp KV cache is usable if every attention layer's cache is f16 cells of [kv heads x head dim] in CMM
    // (N cells, padded at allocation to the context LMAX of the smallest qwen35_attn*_kvc_c<LMAX> kernel that holds
    // them), V not transposed (flash-attention layout)
    const std::string attn_name = mx8 ? "qwen35_attn_mx8" : "qwen35_attn";
    {
        auto ki = t.find("attn_inp_k_idxs"), vi = t.find("attn_inp_v_idxs");
        m.kv_llama = ki != t.end() && vi != t.end() && ki->second->ne[0] == vi->second->ne[0];
    }
    m.N = 0;
    for (int il = 0; il < L && m.kv_llama; ++il) {
        if (t.count("blk." + std::to_string(il) + ".ssm_a")) {
            continue;
        }
        for (const char * kv : { "k", "v" }) {
            auto it = t.find("cache_" + std::string(kv) + "_l" + std::to_string(il));
            m.kv_llama = m.kv_llama && it != t.end() && it->second->type == GGML_TYPE_F16 &&
                         ggml_backend_buffer_is_axera(it->second->buffer) && ggml_is_contiguous(it->second) &&
                         it->second->ne[3] == 1 && (m.N == 0 || it->second->ne[1] == m.N);
            m.N = m.kv_llama ? it->second->ne[1] : m.N;
        }
    }
    // a KV cache no fused attention kernel holds (or GGML_AXERA_FLASH=1): split attention over S-cell segments
    const int64_t ctx = m.kv_llama ? axera_kernels().ctx_for(attn_name + "_kvc", m.N) : 0;
    m.fd.clear();
    m.S     = axera_flash_seg();
    m.flash = m.kv_llama && m.S > 0 && (ctx == 0 || axera_flash_forced()) && axera_q35_load_flash(m, 1);
    if (m.kv_llama && ctx == 0 && !m.flash) {
        GGML_LOG_WARN("%s: no %s_kvc_c<LMAX> kernel holds the %" PRId64 "-cell KV cache and no split attention "
                      "kernels qwen35_attn_fd_*; lower -c\n", __func__, attn_name.c_str(), m.N);
    }
    m.kv_llama = m.kv_llama && (ctx > 0 || m.flash);
    m.ctx_sfx  = m.kv_llama && !m.flash ? "_kvc_c" + std::to_string(ctx) : "";
    m.gdn  = axera_kernels().get(mx8 ? "qwen35_gdn_mx8" : "qwen35_gdn");
    m.attn = m.flash ? nullptr : axera_kernels().get(attn_name + m.ctx_sfx);
    if (m.flash) {  // segment mask [S / BC][R][BC]
        ggml_axera_kernel *        sk = m.fd.at(1).seg;
        const AX_ENGINE_IOMETA_T & mt = sk->info->pInputs[sk->input("mask")];
        GGML_ASSERT(mt.nShapeSize == 3 && (int64_t) mt.pShape[0] * mt.pShape[2] == m.S);
        m.LMAX = (int) m.S;
        m.BC   = mt.pShape[2];
    } else {
        axera_q35_context(m.attn, m.LMAX, m.BC);
    }
    if (m.kv_llama && !m.flash) {
        GGML_ASSERT(m.LMAX == ctx && m.N % m.BC == 0);
    } else if (!m.kv_llama) {
        m.N = m.LMAX;
    }
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
    auto size_of = axera_input_size;
    GGML_ASSERT(size_of(m.gdn, "x") == (uint32_t) H * sizeof(ggml_bf16_t));
    ggml_axera_kernel * rope_k = m.flash ? m.fd.at(1).head : m.attn;  // decode cos / sin [1, ROT / 2]
    m.A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.B    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.NBLK = axera_cmm_alloc(sizeof(int32_t));
    m.COS  = axera_cmm_alloc(size_of(rope_k, "cos"));
    m.SIN  = axera_cmm_alloc(size_of(rope_k, "sin"));
    m.POS.reset();
    m.MASK.reset();
    if (!m.flash) {
        m.POS  = axera_cmm_alloc(size_of(m.attn, "pos"));
        m.MASK = axera_cmm_alloc(size_of(m.attn, "mask"));
        GGML_ASSERT(m.MASK->size >= (size_t) m.LMAX * sizeof(ggml_bf16_t));
    }
    if (size_of(rope_k, "cos") != (uint32_t) (m.ROT / 2) * sizeof(ggml_bf16_t)) {
        GGML_ABORT("qwen35 template: the graph rotates %d dims, the attention kernel %zu\n", m.ROT,
                   2 * size_of(rope_k, "cos") / sizeof(ggml_bf16_t));
    }
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
        const std::string sfx  = std::to_string(P) + (pre_mx ? "_mx8" : "");
        const std::string asfx = sfx + m.ctx_sfx;
        if (!std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx)).good() ||
            !(m.flash ? axera_q35_load_flash(m, P) : std::ifstream(axera_kernels().path("qwen35_attn_prefill" + asfx)).good())) {
            continue;
        }
        axera_q35::prefill pf;
        pf.P    = P;
        pf.gdn  = axera_kernels().get("qwen35_gdn_prefill" + sfx);
        pf.attn = m.flash ? nullptr : axera_kernels().get("qwen35_attn_prefill" + asfx);
        GGML_ASSERT(size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
        ggml_axera_kernel * rk = m.flash ? m.fd.at(P).head : pf.attn;
        if (!m.flash) {
            // chunk mask [LMAX / BC][R][BC] (R = P rows, or G * P in older kernels)
            pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
            GGML_ASSERT(pf.R % P == 0 && (size_t) pf.R * m.LMAX * sizeof(ggml_bf16_t) == size_of(pf.attn, "mask"));
            pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
        }
        pf.A    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.B    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
        pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
        m.pre.push_back(std::move(pf));
    }
    // batched decode kernels (P = B tokens of B sequences, largest first): with the recurrent state and the KV cache
    // in the llama.cpp memory. bf16 kernels; next to MXFP8 decode kernels they use the bf16 GGUF tensors
    for (int Bn : { 8, 4, 2 }) {
        const std::string gn = "qwen35_gdn_batch" + std::to_string(Bn), an = "qwen35_attn_batch" + std::to_string(Bn);
        if (!m.kv_llama || !m.rollback || !std::ifstream(axera_kernels().path(gn)).good() ||
            !(m.flash ? axera_q35_load_flash(m, Bn) : std::ifstream(axera_kernels().path(an + m.ctx_sfx)).good())) {
            continue;
        }
        axera_q35::prefill pf;
        pf.P     = Bn;
        pf.batch = true;
        pf.gdn   = axera_kernels().get(gn);
        pf.attn  = m.flash ? nullptr : axera_kernels().get(an + m.ctx_sfx);
        GGML_ASSERT(size_of(pf.gdn, "x") == (uint32_t) Bn * H * sizeof(ggml_bf16_t));
        ggml_axera_kernel * rk = m.flash ? m.fd.at(Bn).head : pf.attn;
        if (!m.flash) {
            pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
            GGML_ASSERT(pf.R == Bn);
            pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
            pf.POS  = axera_cmm_alloc(size_of(pf.attn, "pos"));
        }
        pf.A    = axera_cmm_alloc(Bn * H * sizeof(ggml_bf16_t));
        pf.B    = axera_cmm_alloc(Bn * H * sizeof(ggml_bf16_t));
        pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
        pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
        m.pre.push_back(std::move(pf));
    }
    for (size_t v = 0; v < m.pre.size(); ++v) {
        for (int il = 0; il < L; ++il) {
            axera_q35_bind_prefill(m, il, v, t);
        }
    }
    for (int il = 0; il < L && m.flash; ++il) {
        for (int v = -1; v < (int) m.pre.size() && !m.layers[il].gdn; ++v) {
            axera_q35_bind_flash(m, il, v, t);
        }
    }
    std::string chunks, batches;
    for (auto & pf : m.pre) {
        auto & str = pf.batch ? batches : chunks;
        str += (str.empty() ? "" : "/") + std::to_string(pf.P);
    }
    if (!chunks.empty()) {
        chunks += pre_mx ? " (MXFP8)" : " (BF16)";
    }
    m.ready = true;
    GGML_LOG_INFO("%s: Axera qwen35 template: %d layers, context %" PRId64 " (kernels up to %d), %s weights, recurrent "
                  "state %s, KV cache %s, prefill chunks %s, batched decode %s\n", __func__, L, m.N, m.LMAX,
                  mx8 ? "MXFP8" : "BF16",
                  m.rollback ? "in the llama.cpp cache (checkpoints work)" : "private",
                  m.flash ? "the llama.cpp cache, attention split over segments (flash decoding)" :
                  m.kv_llama ? "the llama.cpp cache (prompt cache works)" : "private", chunks.empty() ? "none" : chunks.c_str(),
                  batches.empty() ? "none" : batches.c_str());
}

// point every delta-net layer's state IO (decode and prefill kernels) at recurrent-cache row `row` (a sequence)
static void axera_q35_set_row(axera_q35 & m, int64_t row) {
    for (auto & ly : m.layers) {
        if (ly.row == row) {
            continue;
        }
        for (auto & rg : ly.cache_state) {
            const AX_ENGINE_IO_BUFFER_T io = axera_io(rg.b, rg.base + row * rg.stride, (uint32_t) rg.size);
            ly.in[rg.in]                   = io;
            ly.out[rg.out]                 = io;
            for (auto & pre : ly.pre) {
                if (!pre.batch) {
                    pre.in[rg.conv ? pre.ci : pre.si]  = io;
                    pre.out[rg.conv ? pre.co : pre.so] = io;
                }
            }
        }
        ly.row = row;
    }
}

// point the batched decode IO v of every delta-net layer at the recurrent-cache rows row0 .. row0 + P - 1
static void axera_q35_set_batch_rows(axera_q35 & m, size_t v, int64_t row0) {
    for (auto & ly : m.layers) {
        auto & pre = ly.pre[v];
        for (auto & rg : ly.cache_state) {
            const AX_ENGINE_IO_BUFFER_T io =
                axera_io(rg.b, rg.base + row0 * rg.stride, (uint32_t) (m.pre[v].P * rg.stride));
            pre.in[rg.conv ? pre.ci : pre.si]  = io;
            pre.out[rg.conv ? pre.co : pre.so] = io;
        }
    }
}

// clear recurrent-cache row `row` (a new sequence) in every delta-net layer
static void axera_q35_zero_row(axera_q35 & m, int64_t row) {
    for (auto & ly : m.layers) {
        for (auto & rg : ly.cache_state) {
            char * p = rg.base + row * rg.stride;
            memset(p, 0, rg.size);
            rg.b->flush(p, rg.size);
        }
    }
}

// point every attention layer's KV IO (decode and prefill kernels) at KV cache stream `s` (llama.cpp's per-sequence
// streams without a unified KV cache)
static void axera_q35_set_stream(axera_q35 & m, int64_t s) {
    for (auto & ly : m.layers) {
        if (ly.stream == s) {
            continue;
        }
        for (auto & kr : ly.kv) {
            const AX_ENGINE_IO_BUFFER_T io = axera_io(kr.b, kr.base + s * kr.stride, (uint32_t) kr.size);
            ly.out[kr.out]                 = io;
            for (auto & pre : ly.pre) {
                pre.out[kr.v ? pre.vo : pre.ko] = io;
            }
        }
        ly.stream = s;
    }
}

static void axera_q35_restart(axera_q35 & m) {
    for (auto & ly : m.layers) {
        for (auto * s : ly.state) {
            memset(s->vir, 0, s->size);
            s->flush(s->vir, s->size);
        }
    }
    if (m.MASK) {
        auto * mask = (ggml_bf16_t *) m.MASK->vir;
        for (int j = 0; j < m.LMAX; ++j) {
            mask[j] = ggml_fp32_to_bf16(-3e38f);
        }
        m.MASK->flush(m.MASK->vir, m.MASK->size);
    }
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
        const ggml_tensor * rope = nullptr;
        for (int i = 0; i < cgraph->n_nodes && !rope; i++) {
            rope = cgraph->nodes[i]->op == GGML_OP_ROPE ? cgraph->nodes[i] : nullptr;
        }
        GGML_ASSERT(rope && "qwen35 template: no ROPE node in the graph");
        axera_q35_init(m, L, H, t, lm, rope);
        m.bound = w0->data;
        axera_q35_restart(m);
    }

    // the layers: chunks of P consecutive positions through the prefill kernels, single tokens through the decode ones
    const int32_t *    pos = (const int32_t *) inp_pos->data;
    std::vector<float> h((size_t) T * H);
    // with the llama.cpp KV cache: the cells of the new tokens and llama.cpp's KQ mask over the cells
    const ggml_tensor * k_idxs = m.kv_llama ? find("attn_inp_k_idxs") : nullptr;
    const ggml_tensor * kqm    = m.kv_llama ? find("attn_inp_kq_mask") : nullptr;
    if (m.kv_llama) {
        GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm &&
                    (kqm->ne[3] == 1 ? kqm->ne[1] >= T : kqm->ne[1] * kqm->ne[3] == T) &&
                    kqm->ne[0] <= m.N && (kqm->type == GGML_TYPE_F16 || kqm->type == GGML_TYPE_F32));
    }
    // k_idxs: stream * N + cell. The KQ mask is [n_kv, T / streams, 1, streams] (the ubatch's streams in order)
    auto stream_of = [&](int t) { return ((const int64_t *) k_idxs->data)[t] / m.N; };
    auto cell_of   = [&](int t) { return ((const int64_t *) k_idxs->data)[t] % m.N; };

    // sequences of the batch (llama.cpp splits recurrent batches into n_seqs equal runs of n_seq_tokens tokens) and
    // the recurrent-cache bookkeeping of this graph: rs_head (row of the first sequence: the lowest row the state
    // copies write), rs_z (row to clear, from the zeroing SCALE), s_copy (row i of the batch <- row s_copy[i])
    int64_t n_seqs = 1, rs_head = 0, rs_z = -1;
    bool    in_place = !m.layers.empty();
    for (auto & ly : m.layers) {
        in_place = in_place && (!ly.gdn || ly.in_place);
    }
    int g0 = 0;
    while (g0 < L && !m.layers[g0].gdn) {
        ++g0;
    }
    const ggml_tensor * s0 = g0 < L && in_place ? find(("cache_s_l" + std::to_string(g0)).c_str()) : nullptr;
    if (s0) {
        int64_t best = -1;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != s0 || ggml_nelements(node) == 0) {
                continue;
            }
            if (node->op == GGML_OP_CPY && (best < 0 || (int64_t) node->view_offs < best)) {
                best   = (int64_t) node->view_offs;
                n_seqs = (int64_t) (ggml_nbytes(node) / s0->nb[1]);
            } else if (node->op == GGML_OP_SCALE) {
                rs_z = (int64_t) (node->view_offs / s0->nb[1]);
            }
        }
        rs_head = best < 0 ? 0 : best / (int64_t) s0->nb[1];
        n_seqs  = std::max<int64_t>(1, n_seqs);
    }
    if (T % n_seqs != 0 || (n_seqs > 1 && !(m.kv_llama && in_place))) {
        GGML_LOG_ERROR("qwen35 template: %" PRId64 " sequences in a batch of %d need the recurrent state and KV cache in "
                       "the llama.cpp memory\n", n_seqs, T);
        return GGML_STATUS_FAILED;
    }
    const int64_t n_seq_tokens = T / n_seqs;
    if (s0) {
        if (rs_z >= 0) {
            axera_q35_zero_row(m, rs_z);
        }
        const ggml_tensor * sc = find("rs_s_copy");
        bool                identity = true;
        for (int64_t i = 0; sc && i < sc->ne[0]; ++i) {
            identity = identity && ((const int32_t *) sc->data)[i] == rs_head + i;
        }
        if (!identity) {  // move states between rows (sequence copies, cell reuse): row rs_head + i <- s_copy[i]
            for (auto & ly : m.layers) {
                for (auto & rg : ly.cache_state) {
                    std::vector<char> tmp((size_t) sc->ne[0] * rg.stride);
                    for (int64_t i = 0; i < sc->ne[0]; ++i) {
                        char * src = rg.base + (int64_t) ((const int32_t *) sc->data)[i] * rg.stride;
                        rg.b->flush(src, rg.stride);
                        rg.b->invalidate(src, rg.stride);
                        memcpy(tmp.data() + i * rg.stride, src, rg.stride);
                    }
                    for (int64_t i = 0; i < sc->ne[0]; ++i) {
                        char * dst = rg.base + (rs_head + i) * rg.stride;
                        memcpy(dst, tmp.data() + i * rg.stride, rg.stride);
                        rg.b->flush(dst, rg.stride);
                    }
                }
            }
        }
    }
    auto row_of = [&](int t) { return rs_head + t / n_seq_tokens; };
    // KQ mask row of token t -> bf16 over the first W cells (llama.cpp's n_kv rounded up to whole blocks, cells past
    // n_kv hidden); returns the number of blocks to scan. The kernels never read past those blocks.
    const int64_t W = m.kv_llama ? std::min<int64_t>(m.N, (kqm->ne[0] + m.BC - 1) / m.BC * m.BC) : 0;
    auto mask_row = [&](int t, ggml_bf16_t * dst) {
        const ggml_bf16_t ninf = ggml_fp32_to_bf16(-3e38f);  // finite, as the kernels' own masking: no inf - inf
        const char *      row  = (const char *) kqm->data + (kqm->ne[3] == 1 ? t * kqm->nb[1] :
                                                                 (t / kqm->ne[1]) * kqm->nb[3] + (t % kqm->ne[1]) * kqm->nb[1]);
        int64_t           last = -1;
        for (int64_t c = 0; c < W; ++c) {
            float v = -INFINITY;
            if (c < kqm->ne[0]) {
                v = kqm->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[c]) : ((const float *) row)[c];
            }
            if (std::isinf(v) && v < 0) {
                dst[c] = ninf;
            } else {
                dst[c] = ggml_fp32_to_bf16(v);
                last   = c;
            }
        }
        return (int) (last < 0 ? 1 : last / m.BC + 1);
    };
    auto begin = [&](int64_t p) {  // accept position p as the next one; false fails the batch
        if (m.kv_llama) {
            // llama.cpp owns the KV cells and, with the recurrent state in its cache, the state of every position it
            // continues from; only a new sequence needs its recurrent state cleared
            if (p == 0) {
                axera_q35_zero_row(m, m.layers.empty() ? 0 : m.layers[0].row);
            }
            m.next_pos = p;
            return true;
        }
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
    auto rope_rows = [&](int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) {
        auto * c = (ggml_bf16_t *) cb->vir;  // rows: the positions of tokens t0 .. t0 + n - 1
        auto * s = (ggml_bf16_t *) sb->vir;
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < m.ROT / 2; ++i) {
                const float a             = (float) pos[t0 + r] * m.freq_scale * m.inv_freq[i];
                c[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::cos(a));
                s[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::sin(a));
            }
        }
        cb->flush(cb->vir, cb->size);
        sb->flush(sb->vir, sb->size);
    };
    // split attention: the KQ mask rows of tokens t0 .. t0 + R - 1 cut into the segments' masks, and per segment the
    // blocks to scan (0: no token sees a cell of it); the same for every attention layer of the batch
    auto flash_prepare = [&](int R, int t0) {
        axera_q35::fd_kernels & fk   = m.fd.at(R);
        const int64_t           nseg = (W + m.S - 1) / m.S;
        const ggml_bf16_t       ninf = ggml_fp32_to_bf16(-3e38f);
        std::vector<ggml_bf16_t> rows((size_t) R * W);
        for (int r = 0; r < R; ++r) {
            mask_row(t0 + r, rows.data() + (size_t) r * W);
        }
        fk.nblk.assign(nseg, 0);
        fk.seen.assign(nseg * R, 0);
        if ((int64_t) fk.MASK.size() < nseg) {
            fk.MASK.resize(nseg);
        }
        for (int64_t g = 0; g < nseg; ++g) {
            const int64_t c0 = g * m.S, c1 = std::min<int64_t>(W, c0 + m.S);
            int64_t       last = -1;
            for (int r = 0; r < R; ++r) {
                for (int64_t c = c0; c < c1; ++c) {
                    if (rows[(size_t) r * W + c].bits != ninf.bits) {
                        fk.seen[g * R + r] = 1;
                        last               = std::max(last, c - c0);
                    }
                }
            }
            if (last < 0) {
                continue;
            }
            fk.nblk[g] = (int32_t) (last / m.BC + 1);
            if (!fk.MASK[g]) {
                fk.MASK[g] = axera_cmm_alloc(axera_input_size(fk.seg, "mask"));
            }
            auto * mk = (ggml_bf16_t *) fk.MASK[g]->vir;
            for (int64_t blk = 0; blk < fk.nblk[g]; ++blk) {
                for (int r = 0; r < R; ++r) {
                    for (int c = 0; c < m.BC; ++c) {
                        const int64_t col = c0 + blk * m.BC + c;
                        mk[(blk * R + r) * m.BC + c] = col < c1 ? rows[(size_t) r * W + col] : ninf;
                    }
                }
            }
            fk.MASK[g]->flush(mk, (size_t) fk.nblk[g] * R * m.BC * sizeof(ggml_bf16_t));
        }
    };
    // split attention layer: head kernel, the new K / V rows to their cells, one segment kernel call per segment in
    // use, merged on the CPU (m = max m_i, l = sum e^(m_i - m) l_i, o = sum e^(m_i - m) o_i / l), tail kernel
    auto flash_attn = [&](int il, int v, int t0, int64_t p) {
        axera_q35_layer &        ly = m.layers[il];
        const int                R  = v < 0 ? 1 : m.pre[v].P;
        axera_q35::fd_kernels &  fk = m.fd.at(R);
        axera_q35_layer::fd_io & io = ly.fd[v + 1];
        auto run = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                       std::vector<AX_ENGINE_IO_BUFFER_T> & out, const char * what) {
            AX_ENGINE_IO_T s;
            memset(&s, 0, sizeof(s));
            s.pInputs     = in.data();
            s.nInputSize  = (uint32_t) in.size();
            s.pOutputs    = out.data();
            s.nOutputSize = (uint32_t) out.size();
            const int ret = AX_ENGINE_RunSync(k->handle, &s);
            if (ret != 0) {
                GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d (%s), position %" PRId64 "\n", ret, il,
                           what, p);
            }
        };
        run(fk.head, io.hin, io.hout, "attention head");
        const int64_t st = stream_of(t0);
        fk.K->invalidate(fk.K->vir, fk.K->size);
        fk.V->invalidate(fk.V->vir, fk.V->size);
        for (auto & kr : ly.kv) {
            const char * src = (const char *) (kr.v ? fk.V : fk.K)->vir;
            for (int r = 0; r < R; ++r) {
                char * dst = kr.base + st * kr.stride + cell_of(t0 + r) * kr.row;
                memcpy(dst, src + r * kr.row, kr.row);
                kr.b->flush(dst, kr.row);
            }
        }
        const auto & kk  = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
        const auto & vv  = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
        const int    nr  = fk.hkv * fk.gr;  // rows: KV head hk, row gq * R + token
        std::vector<float>   am(nr, 0.0f), al(nr, 0.0f), ao((size_t) nr * fk.d, 0.0f);
        std::vector<uint8_t> started(nr, 0);
        for (size_t g = 0; g < fk.nblk.size(); ++g) {
            if (fk.nblk[g] == 0) {
                continue;
            }
            io.sin[io.kc]   = axera_io(kk.b, kk.base + st * kk.stride + g * m.S * kk.row, (uint32_t) kk.size);
            io.sin[io.vc]   = axera_io(vv.b, vv.base + st * vv.stride + g * m.S * vv.row, (uint32_t) vv.size);
            io.sin[io.mask] = axera_io(fk.MASK[g].get(), fk.MASK[g]->vir, (uint32_t) axera_input_size(fk.seg, "mask"));
            *(int32_t *) m.NBLK->vir = fk.nblk[g];
            m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            run(fk.seg, io.sin, io.sout, "attention segment");
            fk.M->invalidate(fk.M->vir, fk.M->size);
            fk.Lsum->invalidate(fk.Lsum->vir, fk.Lsum->size);
            fk.O->invalidate(fk.O->vir, fk.O->size);
            const auto * mi = (const ggml_bf16_t *) fk.M->vir;
            const auto * li = (const float *) fk.Lsum->vir;
            const auto * oi = (const ggml_bf16_t *) fk.O->vir;
            for (int j = 0; j < nr; ++j) {
                if (!fk.seen[g * R + (j % fk.gr) % R]) {
                    continue;
                }
                const float  mj = ggml_bf16_to_fp32(mi[j]);
                float *      o  = ao.data() + (size_t) j * fk.d;
                const auto * oj = oi + (size_t) j * fk.d;
                if (!started[j]) {
                    started[j] = 1;
                    am[j]      = mj;
                    al[j]      = li[j];
                    for (int i = 0; i < fk.d; ++i) {
                        o[i] = ggml_bf16_to_fp32(oj[i]);
                    }
                    continue;
                }
                const float mn = std::max(am[j], mj), a = std::exp(am[j] - mn), b = std::exp(mj - mn);
                am[j] = mn;
                al[j] = a * al[j] + b * li[j];
                for (int i = 0; i < fk.d; ++i) {
                    o[i] = a * o[i] + b * ggml_bf16_to_fp32(oj[i]);
                }
            }
        }
        // attention output [R][HQ][D]: query head hk * G + gq of token t is row gq * R + t of KV head hk
        const int G   = fk.gr / R;
        auto *    att = (ggml_bf16_t *) fk.ATT->vir;
        for (int hk = 0; hk < fk.hkv; ++hk) {
            for (int gq = 0; gq < G; ++gq) {
                for (int tt = 0; tt < R; ++tt) {
                    const int     j   = hk * fk.gr + gq * R + tt;
                    const float   inv = al[j] > 0.0f ? 1.0f / al[j] : 0.0f;
                    ggml_bf16_t * dst = att + ((size_t) tt * fk.hkv * G + hk * G + gq) * fk.d;
                    for (int i = 0; i < fk.d; ++i) {
                        dst[i] = ggml_fp32_to_bf16(ao[(size_t) j * fk.d + i] * inv);
                    }
                }
            }
        }
        fk.ATT->flush(fk.ATT->vir, fk.ATT->size);
        run(fk.tail, io.tin, io.tout, "attention tail");
    };
    auto run_layers = [&](int v, int t0, int64_t p) {  // v: prefill chunk size index, -1: decode
        const bool prefill = v >= 0;
        for (int il = 0; il < L; ++il) {
            axera_q35_layer & ly = m.layers[il];
            if (m.flash && !ly.gdn) {
                flash_attn(il, v, t0, p);
                continue;
            }
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
        if (s0) {
            axera_q35_set_row(m, row_of(it));
        }
        if (m.kv_llama && !m.flash) {
            axera_q35_set_stream(m, stream_of(it));
        }
        if (!begin(p)) {
            return GGML_STATUS_FAILED;
        }
        int v = -1;  // the largest prefill chunk that fits the remaining consecutive positions
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            const int P  = m.pre[c].P;
            bool      ok = !m.pre[c].batch && it + P <= T && (m.kv_llama || p + P <= m.LMAX) &&
                      it % n_seq_tokens + P <= n_seq_tokens;
            for (int j = 1; ok && j < P; ++j) {
                ok = pos[it + j] == p + j && (!m.kv_llama || (stream_of(it + j) == stream_of(it) &&
                                                              cell_of(it + j) == cell_of(it) + j));
            }
            ok = ok && (!m.kv_llama || cell_of(it) + P <= m.N);
            v = ok ? (int) c : -1;
        }
        // else one token of each of the next P sequences (rows rs_head + it ..) in one KV stream: batched decode
        for (size_t c = 0; c < m.pre.size() && v < 0 && n_seq_tokens == 1 && s0; ++c) {
            const int P  = m.pre[c].P;
            bool      ok = m.pre[c].batch && it + P <= T;
            for (int j = 1; ok && j < P; ++j) {
                ok = stream_of(it + j) == stream_of(it);
            }
            v = ok ? (int) c : -1;
        }
        if (v >= 0) {
            axera_q35::prefill & pf = m.pre[v];
            const int            P  = pf.P;
            if (pf.batch) {
                axera_q35_set_batch_rows(m, v, row_of(it));
                for (int r = 1; r < P; ++r) {  // token it's row was cleared by begin()
                    if (pos[it + r] == 0) {
                        axera_q35_zero_row(m, row_of(it + r));
                    }
                }
                for (int r = 0; r < P && !m.flash; ++r) {
                    ((int32_t *) pf.POS->vir)[r] = (int32_t) cell_of(it + r);
                }
                if (!m.flash) {
                    pf.POS->flush(pf.POS->vir, pf.POS->size);
                }
            }
            for (int r = 0; r < P; ++r) {
                ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (it + r) * x0->nb[1]),
                                      (ggml_bf16_t *) pf.A->vir + (size_t) r * H, H);
            }
            pf.A->flush(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rope_rows(it, P, pf.COS.get(), pf.SIN.get());
            if (m.flash) {
                flash_prepare(P, it);
            }
            int nblk = (int) ((p + P - 1) / m.BC + 1);
            // chunk mask [N / BC][R][BC], row r = token r % P (of head r / P in older kernels): positions up to
            // p + token, or with the llama.cpp KV cache that token's KQ mask row over the cells
            const int64_t R    = pf.R;
            auto *        mask = pf.MASK ? (ggml_bf16_t *) pf.MASK->vir : nullptr;  // none with split attention
            const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f), ninf = ggml_fp32_to_bf16(-3e38f);
            if (m.flash) {
                // the segments' masks: flash_prepare
            } else if (m.kv_llama) {
                std::vector<ggml_bf16_t> rows((size_t) P * W);
                nblk = 1;
                for (int r = 0; r < P; ++r) {
                    nblk = std::max(nblk, mask_row(it + r, rows.data() + (size_t) r * W));
                }
                for (int64_t blk = 0; blk < nblk; ++blk) {
                    for (int64_t r = 0; r < R; ++r) {
                        memcpy(mask + (blk * R + r) * m.BC, rows.data() + (size_t) (r % P) * W + blk * m.BC,
                               m.BC * sizeof(ggml_bf16_t));
                    }
                }
            } else {
                for (int64_t blk = 0; blk < nblk; ++blk) {
                    for (int64_t r = 0; r < R; ++r) {
                        const int64_t lim = p + r % P;
                        ggml_bf16_t * row = mask + (blk * R + r) * m.BC;
                        for (int c = 0; c < m.BC; ++c) {
                            row[c] = blk * m.BC + c <= lim ? zero : ninf;
                        }
                    }
                }
            }
            if (!m.flash) {
                *(int32_t *) m.POS->vir  = (int32_t) (m.kv_llama ? cell_of(it) : p);
                *(int32_t *) m.NBLK->vir = nblk;
                if (!m.kv_llama) {
                    for (int r = 0; r < P; ++r) {
                        ((ggml_bf16_t *) m.MASK->vir)[p + r] = zero;  // for the decode steps after the chunk
                    }
                    m.MASK->flush(m.MASK->vir, m.MASK->size);
                }
                pf.MASK->flush(pf.MASK->vir, (size_t) nblk * R * m.BC * sizeof(ggml_bf16_t));
                m.POS->flush(m.POS->vir, m.POS->size);
                m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            }
            run_layers(v, it, p);
            auto * fin = (L % 2 == 0 ? pf.A : pf.B).get();
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, (int64_t) P * H);
            m.next_pos = p + P;
            it += P;
            continue;
        }
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + it * x0->nb[1]), (ggml_bf16_t *) m.A->vir, H);
        m.A->flush(m.A->vir, H * sizeof(ggml_bf16_t));
        rope_rows(it, 1, m.COS.get(), m.SIN.get());
        if (m.flash) {
            flash_prepare(1, it);
        } else {
            if (m.kv_llama) {
                *(int32_t *) m.NBLK->vir = mask_row(it, (ggml_bf16_t *) m.MASK->vir);
                *(int32_t *) m.POS->vir  = (int32_t) cell_of(it);
            } else {
                *(int32_t *) m.POS->vir                = (int32_t) p;
                *(int32_t *) m.NBLK->vir               = (int32_t) (p / m.BC + 1);
                ((ggml_bf16_t *) m.MASK->vir)[p]       = ggml_fp32_to_bf16(0.0f);
            }
            m.POS->flush(m.POS->vir, m.POS->size);
            m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            m.MASK->flush(m.MASK->vir, (size_t) *(int32_t *) m.NBLK->vir * m.BC * sizeof(ggml_bf16_t));  // blocks read
        }
        run_layers(-1, it, p);
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
    std::vector<float> hn((size_t) n_out * H);
    const float *   g = (const float *) onw->data;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        const float * x  = h.data() + (size_t) row * H;
        float *       y  = hn.data() + (size_t) r * H;
        double        ss = 0;
        for (int i = 0; i < H; ++i) {
            ss += (double) x[i] * x[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
        for (int i = 0; i < H; ++i) {
            y[i] = x[i] * scale * g[i];
        }
        if (rnorm && rnorm->type == GGML_TYPE_F32 && rnorm->ne[0] == H && rnorm->ne[1] == n_out) {
            memcpy((char *) rnorm->data + r * rnorm->nb[1], y, H * sizeof(float));
        }
    }
    // several rows: the batched bf16 GEMV reads the output projection once per call (the bf16 GGUF tensor in CMM or
    // its bf16 copy; MXFP8 weights keep the MXFP8 GEMV, row by row, so that the logits do not depend on the batching)
    const ggml_backend_axera_buffer_context * lm_ctx =
        m.lm_q ? nullptr : m.lm_bf ? m.lm_bf.get() :
        lm->type == GGML_TYPE_BF16 && ggml_backend_buffer_is_axera(lm->buffer) ? (ggml_backend_axera_buffer_context *) lm->buffer->context :
                                                                                  nullptr;
    void * lm_data = m.lm_bf ? m.lm_bf->vir : lm->data;
    auto   dst_of  = [&](int r) { return (float *) ((char *) logits->data + r * logits->nb[1]); };
    for (int r = 0; r < n_out;) {
        int done = 0;
        for (int Bn : { 8, 4, 2 }) {
            if (done == 0 && lm_ctx && r + Bn <= n_out) {
                float * dsts[8];
                for (int b = 0; b < Bn; ++b) {
                    dsts[b] = dst_of(r + b);
                }
                done = axera_gemv_batch(ctx, lm_ctx, lm_data, lm->ne[1], H, hn.data() + (size_t) r * H, Bn, dsts) ? Bn : 0;
            }
        }
        if (done > 0) {
            r += done;
            continue;
        }
        const float * y = hn.data() + (size_t) r * H;
        if (m.lm_q) {
            axera_gemv_mx8(ctx, m.lm_q.get(), m.lm_s.get(), lm->ne[1], H, y, logits, dst_of(r));
        } else if (m.lm_bf) {
            axera_gemv_raw(ctx, m.lm_bf.get(), m.lm_bf->vir, lm->ne[1], H, y, logits, dst_of(r));
        } else {
            axera_gemv(ctx, lm, y, logits, dst_of(r));
        }
        ++r;
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

// the qwen35 template's attention kernels address the KV cache as one IO of their context LMAX: a llama.cpp KV cache
// tensor (cache_k_l* / cache_v_l*) of fewer cells gets room for the smallest kernel context that holds it, after its
// last stream
static size_t ggml_backend_axera_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    const size_t n = ggml_nbytes(t);
    if (!axera_template_qwen35() || t->type != GGML_TYPE_F16 || t->ne[3] != 1 ||
        (strncmp(t->name, "cache_k_l", 9) != 0 && strncmp(t->name, "cache_v_l", 9) != 0)) {
        return n;
    }
    const int64_t ctx = axera_kernels().ctx_for(axera_weights_mx8() ? "qwen35_attn_mx8_kvc" : "qwen35_attn_kvc", t->ne[1]);
    const int64_t S   = axera_flash_seg();  // split attention: whole segments
    const int64_t c   = S > 0 && (ctx == 0 || axera_flash_forced()) ? (t->ne[1] + S - 1) / S * S : ctx;
    return std::max(n, (size_t) (t->ne[2] - 1) * t->nb[2] + (size_t) c * t->nb[1]);  // ne[2]: per-sequence streams
    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_axera_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_axera_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_axera_buffer_type_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ ggml_backend_axera_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_axera_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_axera_buffer_type_get_alloc_size,
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
