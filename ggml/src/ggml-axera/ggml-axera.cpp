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
// - Multi-core builds (name_x<cores>, AXDSL kernels split over the cores without synchronization) replace the
//   single-core kernel of the same name with GGML_AXERA_CORES=4 (experimental, off by default: in llama.cpp the
//   chains still produce NaN / drifting results or time out (0x8006008a) now and then, while the kernels pass
//   stress tests on their own). They need all cores: other NPU users at the same time make the calls time out.
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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
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

// GGML_AXERA_PROF=1: wall time per part of the qwen35 template, printed at exit
struct axera_prof {
    const bool                         on = [] {
        const char * e = std::getenv("GGML_AXERA_PROF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    std::map<std::string, std::pair<double, int64_t>> t;  // ms, calls
    std::mutex                                        mu;  // timers on other threads (the MoE prefill slots calls)
    ~axera_prof() {
        for (const auto & [k, v] : t) {
            fprintf(stderr, "axera prof: %-22s %10.1f ms %8" PRId64 " calls %8.3f ms/call\n", k.c_str(), v.first, v.second,
                    v.first / std::max<int64_t>(v.second, 1));
        }
    }
};
static axera_prof g_axera_prof;

struct axera_timer {
    const char * what;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit axera_timer(const char * w) : what(g_axera_prof.on ? w : nullptr) {}
    ~axera_timer() { stop(); }
    void stop() {
        if (what) {
            std::lock_guard<std::mutex> lock(g_axera_prof.mu);
            auto &                      e = g_axera_prof.t[what];
            e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            e.second++;
            what = nullptr;
        }
    }
};

// persistent workers for the template's host loops (masks, merges): run(n, fn) calls fn(0 .. n - 1) on the workers
// and the caller, returning when all are done
class axera_pool {
  public:
    explicit axera_pool(int n) {
        for (int i = 0; i < n; ++i) {
            workers.emplace_back([this] { loop(); });
        }
    }

    ~axera_pool() {
        {
            std::lock_guard<std::mutex> l(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto & w : workers) {
            w.join();
        }
    }

    void run(int64_t n, const std::function<void(int64_t)> & fn) {
        if (n <= 1 || workers.empty()) {
            for (int64_t i = 0; i < n; ++i) {
                fn(i);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> l(mu);
            job     = &fn;
            total   = n;
            next    = 0;
            pending = workers.size();
            ++gen;
        }
        cv.notify_all();
        work();
        std::unique_lock<std::mutex> l(mu);
        done.wait(l, [&] { return pending == 0; });
        job = nullptr;
    }

  private:
    void work() {
        for (int64_t i; (i = next.fetch_add(1)) < total;) {
            (*job)(i);
        }
    }

    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
            }
            work();
            std::lock_guard<std::mutex> l(mu);
            if (--pending == 0) {
                done.notify_one();
            }
        }
    }

    std::vector<std::thread>                    workers;
    std::mutex                                  mu;
    std::condition_variable                     cv, done;
    const std::function<void(int64_t)> *        job = nullptr;
    std::atomic<int64_t>                        next{ 0 };
    int64_t                                     total   = 0;
    size_t                                      pending = 0;
    uint64_t                                    gen     = 0;
    bool                                        stop    = false;
};

static axera_pool & axera_host_pool() {
    static axera_pool pool(std::min(15, std::max(1, (int) std::thread::hardware_concurrency() - 1)));
    return pool;
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
    bool   nc   = false;  // non-cacheable: no cache maintenance (host writes only, e.g. gathered kernel inputs)

    ~ggml_backend_axera_buffer_context() {
        if (vir) {
            AX_SYS_MemFree(phy, vir);
        }
    }

    AX_U64 phy_of(const void * p) const { return phy + ((const char *) p - (const char *) vir); }

    void flush(void * p, size_t n) const {
        if (n && !nc) {
            AX_SYS_MflushCache(phy_of(p), p, (AX_U32) n);
        }
    }

    void invalidate(void * p, size_t n) const {
        if (n && !nc) {
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
    axera_timer tm("get_tensor");
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

// non-cacheable CMM for buffers the host only writes (no flushes; host reads from it are slow)
static std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc_nc(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    b->nc   = true;
    if (AX_SYS_MemAlloc(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera-nc") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
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
// GGML_AXERA_SPLIT_MIN: with a fused attention kernel, the KQ mask width (cells) from which a batch takes the split
// attention kernels instead when they exist for the context (multi-core segment scan; default 4096, 0: always)
static int64_t axera_split_min() {
    static const int64_t n = [] {
        const char * e = std::getenv("GGML_AXERA_SPLIT_MIN");
        return e ? std::max<int64_t>(0, std::atoll(e)) : (int64_t) 4096;
    }();
    return n;
}

// GGML_AXERA_CORES: NPU cores for the kernels that have multi-core builds (default 1: single-core kernels only; 4:
// the multi-core builds, experimental: inside llama.cpp they still give wrong results or time out now and then)
static int axera_cores() {
    static const int n = [] {
        const char * e = std::getenv("GGML_AXERA_CORES");
        return e ? std::max(1, std::atoi(e)) : 1;
    }();
    return n;
}

// name, or its multi-core build name_x<cores> when that exists (same IO, the work split over the cores)
static std::string axera_multi_core(const std::string & name) {
    static std::mutex                  mu;
    static std::map<std::string, bool> found;
    if (axera_cores() < 2) {
        return name;
    }
    const std::string           mc = name + "_x" + std::to_string(axera_cores());
    std::lock_guard<std::mutex> lock(mu);
    auto                        f = found.find(mc);
    if (f == found.end()) {
        f = found.emplace(mc, std::ifstream(axera_kernels().path(mc)).good()).first;
    }
    return f->second ? mc : name;
}

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
        // the chunk's snapshot kernel (prefill::gdn_rs): its IO, its state IO indices and the snapshot outputs
        // state_rs<j> / conv_rs<j> (j = 1 .. P - 1), pointed at the llama.cpp cache planes per call
        std::vector<AX_ENGINE_IO_BUFFER_T> rin, rout;
        int                                rsi = -1, rso = -1, rci = -1, rco = -1;
        std::vector<int>                   rs_state, rs_conv;  // [j]: output index
    };
    std::vector<prefill_io>            pre;  // one per prefill chunk size
    struct fd_io {  // split attention IO of one run kind: head, segment and tail kernels
        std::vector<AX_ENGINE_IO_BUFFER_T> hin, hout, sin, sout, tin, tout;
        int                                kc = -1, vc = -1, mask = -1;  // segment inputs, rebound per call
    };
    std::vector<fd_io>                 fd;  // [0]: decode, [1 + v]: m.pre[v]
    struct expert_w {  // routed experts of an MoE layer (gate or gate-up, up, down): MXFP4 repacked in place
        ggml_backend_axera_buffer_context * b = nullptr;
        char *                              base = nullptr;  // expert 0
        size_t                              stride = 0, data = 0, scale = 0;  // bytes per expert, its elements, scales
    };
    expert_w                           exps[3];
    bool                               exps_gu = false;  // exps[0] holds gate and up (m.ex[1]), exps[1] unused
    struct mc_io {  // multi-core prefill chain of this layer: [gdn | head, seg, tail] then mlp
        std::vector<AX_ENGINE_IO_BUFFER_T> ain, aout, sin, sout, tin, tout, min, mout;
        int si = -1, so = -1, ci = -1, co = -1;            // state IO of the gdn call (rebound to the current row)
        int dsi = -1, dci = -1;                            // the decode kernel's state inputs (ly.in)
        int kc = -1, vc = -1, mask = -1, mo = -1, lo = -1, oo = -1;  // segment IO, rebound per segment
    };
    std::map<int, mc_io>               mc;  // by chunk size P
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
    int                   rope_mode = 0;               // GGML_ROPE_TYPE_*: (I)MROPE takes position t / h / w / e by section
    int                   sections[4] = { 0, 0, 0, 0 };  // M-RoPE sections (rotary frequency pairs per position)
    ggml_axera_kernel *   gdn = nullptr, *attn = nullptr;
    axera_cmm             A, B, POS, NBLK, MASK, COS, SIN;  // A/B: layer input/output ping-pong
    std::shared_ptr<ggml_backend_axera_buffer_context> lm_q, lm_s;  // MXFP8 output projection (shared by contexts)
    axera_cmm             lm_bf;                            // bf16 copy of a non-bf16 output projection
    struct prefill {  // one prefill chunk size, or (batch) one batched decode size
        int                 P = 0;
        int64_t             R = 0;  // mask rows per KV block
        bool                batch = false;  // P tokens of P consecutive sequences (recurrent rows), one each
        bool                mc    = false;  // the multi-core chain m.mc[P] exists (gdn / attn may be null then)
        ggml_axera_kernel * gdn = nullptr, *attn = nullptr;
        axera_cmm           A, B, MASK, COS, SIN, POS;  // POS: the batch's P cells
        axera_cmm           BASE, H2, LOG;              // MoE: the layer kernel's base, h2 and router logits rows
        // speculative decoding (llama.cpp n_rs_seq > 0): the GDN kernel that also writes the recurrent state after
        // each of the P tokens (qwen35_gdn_prefill<P>[_mx8]_rs) into the cache's snapshot planes
        ggml_axera_kernel * gdn_rs = nullptr;
        // MoE: chunks of a group run layer by layer, the routed experts once per layer for the whole group (each
        // expert's weights read once for more tokens). sub > 0: a copy of the group's first chunk (own buffers and
        // split attention state m.fd[fdk]), never picked alone; group: the chunks of the group, this one first
        int                 sub = 0, fdk = 0;  // fdk: key of the split attention kernels in m.fd (0: P)
        std::vector<int>    group;
    };
    std::vector<prefill>  pre;  // prefill chunks, then batched decode sizes, largest first
    const ggml_tensor *   lm = nullptr;
    std::map<const ggml_tensor *, axera_cmm> bf16_of;  // bf16 copies of non-bf16 GGUF weights (quantized GGUF)
    std::vector<axera_q35_layer> layers;
    // MTP draft context (llama.cpp LLM_GRAPH_TYPE_DECODER_MTP): one attention layer, the model's NextN block after the
    // main layers; graph layer il is model layer il + off. Layer 0's input is eh_proj([enorm(embedding) | hnorm(h)])
    bool        mtp = false;
    int         off = 0;
    axera_cmm   eh_bf, EH;  // eh_proj in bf16, its output row
    std::vector<float>    inv_freq;
    int64_t               next_pos = 0;
    bool                  rollback = false;  // all GDN state lives in the llama.cpp cache: earlier positions resumable
    bool                  kv_llama = false;  // attention kernels on the llama.cpp KV cache (cells, f16) and its KQ mask
    std::string           ctx_sfx;           // attention kernel name suffix: "_kvc_c<LMAX>" on the llama.cpp KV cache
    bool                  flash = false;     // attention as head / per-segment / tail kernels (flash decoding)
    bool                  pre_mx = false;    // prefill kernels with MXFP8 weights
    int64_t               S     = 0;         // their KV segment, cells
    struct fd_kernels {  // split attention kernels for R rows
        ggml_axera_kernel *    head = nullptr, *seg = nullptr, *tail = nullptr;
        int                    hkv = 0, gr = 0, d = 0;  // segment outputs [hkv][gr][d]: KV head, row g * R + token
        int                    bc  = 0;  // cells per block of the segment kernel (its mask [S / bc][R][bc])
        int                    cores = 1;  // segment kernel cores, each scanning its own run of blocks
        axera_cmm              Q, GATE, K, V, M, Lsum, O, ATT, BLK;  // BLK: nblk [1], or [cores][start, stop)
        std::vector<axera_cmm> MASK;  // per segment [S / BC][R][BC], filled once per batch for all layers
        std::vector<int32_t>   nblk;  // per segment: blocks to scan, 0 if no token sees a cell of it
        std::vector<int32_t>   runs;  // per segment and core: the run of blocks [start, stop)
        struct part {                  // a partial result to merge: segment g, core c, the tokens that see a cell of it
            int64_t              g;
            int                  c;
            std::vector<uint8_t> seen;
        };
        std::vector<part> parts;
    };
    struct mc_kernels {  // multi-core prefill chain for P tokens: qwen35_{gdn,mlp,attn_head,attn_tail}_mc<P>_x<cores>
        ggml_axera_kernel * gdn = nullptr, *mlp = nullptr, *head = nullptr, *tail = nullptr;
        int                 np = 0;            // partial attention slots the tail merges (segments x segment cores)
        size_t              ms = 0, ls = 0, os = 0;  // bytes per slot of m / l / o
        axera_cmm           BASE1, PART0, PART1, ZERO, MALL, LALL, OALL;  // the residual base 0 is m.pre[v].A
    };
    std::map<int, mc_kernels> mc;  // by P
    std::map<int, fd_kernels> fd;  // by rows R
    // MoE (Qwen3.5-35B-A3B): per layer the layer kernel (qwen35moe_*: up to the shared expert and the router logits),
    // then the routed experts kernel on the top-K experts the host picks from the logits
    bool      moe      = false;
    int       n_expert = 0, top_k = 0;
    axera_cmm BASE, H2, LOGITS, RW;  // residual + gated shared expert, normed state, router logits, weights
    struct experts_kernel {          // qwen35moe_experts<K>[_gu]_mxfp4 and its IO (weights rebound per token)
        ggml_axera_kernel *                k = nullptr;
        axera_cmm                          PART;  // multi-core build: the cores' fp32 partial sums (out = their sum)
        std::vector<int>                   w[4], s[4];  // input indices of w_<kind><i> / s_<kind><i>, kinds g u d c
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out[2];  // out: by the layer parity
    };
    // [0]: gate and up weights apart (w_g / w_u), [1]: one gate-up weight per expert (w_c: gate rows then up rows,
    // one projection instead of two; axera_repack_mxfp4_gu), preferred when the kernel exists
    experts_kernel         ex[2];
    std::vector<axera_cmm> gu_own;  // gate-up experts of layers whose gate and up are not back to back
    // prefill: the routed experts of a chunk grouped by expert, qwen35moe_slots<S>_g<G>_gu_mxfp4: per call S slots,
    // each one expert (gate-up and down weights bound by address) on up to G token rows gathered by the host from h2
    struct slots_kernel {
        ggml_axera_kernel *                k = nullptr;
        int                                S = 0, G = 0, x = -1, rw = -1, out = -1;
        std::vector<int>                   w[2], s[2];  // input indices of w_<c|d><slot> / s_<c|d><slot>
        std::vector<AX_ENGINE_IO_BUFFER_T> in, outs;
    };
    slots_kernel slots;
    axera_cmm    SX, SRW, SOUT;  // all slots of a chunk's layer: rows [slots][G][H] bf16, weights f32, results f32
};

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

struct ggml_backend_axera_context {
    axera_cmm staging;    // bf16 activation row
    axera_cmm batch_out;  // f32 output rows of a batched GEMV call
    axera_q35 q35;
    axera_vit vit;
    // the qwen35 template's name -> tensor map of the last graph, kept while the graph keeps its nodes (llama.cpp
    // reuses the decode graph: rebuilding the map of a 35B-A3B graph takes ~10 ms on the A55)
    std::vector<uintptr_t>               graph_key;
    std::map<std::string, ggml_tensor *> graph_names;

    ggml_backend_axera_buffer_context * staging_for(size_t size) {
        if (!staging || staging->size < size) {
            staging = axera_cmm_alloc(size);
        }
        return staging.get();
    }
};

// out[0:N] (f32, inside a CMM tensor) = W[N, K] (bf16 at wdata in the CMM buffer wctx) . x[0:K], in N / C GEMV calls
static void axera_gemv_raw_to(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                              void * wdata, int64_t N, int64_t K, const float * x,
                              const ggml_backend_axera_buffer_context * dctx, float * out);

static void axera_gemv_raw(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                           void * wdata, int64_t N, int64_t K, const float * x, ggml_tensor * dst, float * out) {
    axera_gemv_raw_to(ctx, wctx, wdata, N, K, x, (const ggml_backend_axera_buffer_context *) dst->buffer->context, out);
}

// out[0:N] (f32, in the CMM buffer dctx) = W[N, K] (bf16 at wdata in wctx) . x[0:K], in N / C GEMV calls
static void axera_gemv_raw_to(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                              void * wdata, int64_t N, int64_t K, const float * x,
                              const ggml_backend_axera_buffer_context * dctx, float * out) {
    const int64_t row = K * (int64_t) sizeof(ggml_bf16_t);
    const int64_t C   = axera_kernels().mul_mat_chunk(N, K);
    GGML_ASSERT(C > 0);

    ggml_axera_kernel * k  = axera_kernels().get(axera_multi_core(axera_mul_mat_kernel_name(C, K)));
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    GGML_ASSERT(k->info->pInputs[ix].nSize == K * sizeof(ggml_bf16_t) &&
                k->info->pInputs[iw].nSize == C * row && k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
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
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, N, K);
        }
    }
    dctx->invalidate(out, N * sizeof(float));
}

// rows out[b][0:N] = W[N, K] . x[b], b < Bn, through the batched GEMV mul_mat_<bf16|mx8>_b<Bn>_n<C>_k<K> (x [Bn, K]
// bf16 -> out [Bn, C] f32), N / C calls. W: bf16 at wdata in the CMM buffer wctx, or (sctx) MXFP8 elements wctx and
// scales sctx. False if that kernel is missing
static bool axera_gemv_batch(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx, void * wdata,
                             const ggml_backend_axera_buffer_context * sctx, int64_t N, int64_t K, const float * x,
                             int Bn, float * const * out) {
    const bool    mx8 = sctx != nullptr;
    const int64_t C   = axera_kernels().mul_mat_chunk(N, K, mx8);
    char          name[96];
    snprintf(name, sizeof(name), "mul_mat_%s_b%d_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", Bn, C, K);
    static std::map<std::string, bool> found;
    auto f = found.find(name);
    if (f == found.end()) {
        f = found.emplace(name, C > 0 && std::ifstream(axera_kernels().path(name)).good()).first;
    }
    if (!f->second) {
        return false;
    }
    ggml_axera_kernel * k   = axera_kernels().get(name);
    const int           ix  = k->input("x"), iw = k->input("w"), is = k->input("s"), io = k->output("out");
    const int64_t       row = K * (int64_t) sizeof(ggml_bf16_t);
    const int64_t       wr  = mx8 ? K : row, sr = K / AXERA_MX_GROUP;  // weight / scale bytes per row
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && (is >= 0) == mx8 && k->info->pInputs[ix].nSize == Bn * row &&
                k->info->pInputs[iw].nSize == C * wr && (!mx8 || k->info->pInputs[is].nSize == C * sr) &&
                k->info->pOutputs[io].nSize == Bn * C * sizeof(float));
    auto * st = ctx->staging_for(Bn * row);
    ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) st->vir, Bn * K);
    st->flush(st->vir, Bn * row);
    if (!ctx->batch_out || ctx->batch_out->size < Bn * C * sizeof(float)) {
        ctx->batch_out = axera_cmm_alloc(Bn * C * sizeof(float));
    }
    auto * bo = ctx->batch_out.get();
    AX_ENGINE_IO_BUFFER_T inputs[3], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = mx8 ? 3 : 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, st->vir, Bn * row);
    outputs[io]        = axera_io(bo, bo->vir, Bn * C * sizeof(float));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw] = axera_io(wctx, (char *) wdata + c0 * wr, C * wr);
        if (mx8) {
            inputs[is] = axera_io(sctx, (char *) sctx->vir + c0 * sr, C * sr);
        }
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
    ggml_axera_kernel * k  = axera_kernels().get(axera_multi_core(axera_mul_mat_kernel_name(C, K, true)));
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

// MoE layers: the dense MLP inputs come from the shared expert, plus the router and the shared expert's gate
static const std::map<std::string, const char *> axera_q35_moe_direct = {
    { "w_gate", "ffn_gate_shexp.weight" }, { "w_up", "ffn_up_shexp.weight" }, { "w_down", "ffn_down_shexp.weight" },
    { "w_router", "ffn_gate_inp.weight" }, { "w_shg", "ffn_gate_inp_shexp.weight" },
};

// GGUF tensor (name within the layer) of kernel input n, nullptr if none
static const char * axera_q35_gguf_name(const axera_q35 & m, const std::string & n) {
    if (m.moe) {
        auto it = axera_q35_moe_direct.find(n);
        if (it != axera_q35_moe_direct.end()) {
            return it->second;
        }
    }
    auto it = axera_q35_direct.find(n);
    return it == axera_q35_direct.end() ? nullptr : it->second;
}

// ggml MXFP4 blocks (per 32 inputs: the E8M0 scale byte, then 16 bytes of E2M1 codes: low nibbles elements 0..15, high
// nibbles 16..31) -> the kernels' layout, in place per expert (dim 2): the codes [rows][K / 2], two consecutive elements
// per byte (the first in the low nibble), then the scales [rows][K / 32]. Once per tensor: the GGUF data is the
// kernels' afterwards (the template runs every MoE layer)
// ggml block_mxfp4 rows (e byte + 16 bytes of nibble pairs per 32) -> the kernels' layout: codes [rows][K / 2] (FP4
// pairs, low nibble first) then E8M0 scales [rows][K / 32]
static void axera_mxfp4_rows(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dq, uint8_t * ds) {
    const int64_t nb = K / 32;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = src + (r * nb + b) * 17;
            uint8_t *       o   = dq + r * K / 2 + b * 16;
            ds[r * nb + b]      = blk[0];
            for (int j = 0; j < 8; ++j) {  // elements 2j, 2j + 1 (low nibbles), 16 + 2j, 17 + 2j (high nibbles)
                o[j]     = (uint8_t) ((blk[1 + 2 * j] & 15) | ((blk[2 + 2 * j] & 15) << 4));
                o[8 + j] = (uint8_t) ((blk[1 + 2 * j] >> 4) | (blk[2 + 2 * j] & 0xF0));
            }
        }
    }
}

static void axera_check_mxfp4(const ggml_tensor * w) {
    GGML_ASSERT(w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w) && w->ne[0] % 32 == 0 && w->ne[3] == 1 &&
                ggml_backend_buffer_is_axera(w->buffer) && w->nb[2] == (size_t) (w->ne[1] * w->ne[0] / 32 * 17));
}

// MoE experts [K, rows, n_expert] in place, each expert's slot (stride bytes) in the kernels' layout
static void axera_repack_mxfp4(const ggml_tensor * w) {
    static std::set<const void *> done;
    if (!done.insert(w->data).second) {
        return;
    }
    axera_check_mxfp4(w);
    const int64_t K = w->ne[0], rows = w->ne[1];
    const size_t  stride = w->nb[2];
    axera_host_pool().run(w->ne[2], [&](int64_t e) {
        std::vector<uint8_t> tmp(stride);
        axera_mxfp4_rows((const uint8_t *) w->data + e * stride, rows, K, tmp.data(), tmp.data() + rows * K / 2);
        memcpy((uint8_t *) w->data + e * stride, tmp.data(), stride);
    });
    ((ggml_backend_axera_buffer_context *) w->buffer->context)->flush(w->data, ggml_nbytes(w));
}

// gate and up experts [K, rows, n] -> one gate-up tensor at dst (2 nbytes): expert e at 2 e strides, codes
// [2 rows][K / 2] (gate rows, then up), scales [2 rows][K / 32] (as a GGUF ffn_gate_up_exps repacked by
// axera_repack_mxfp4). In place when up follows gate (dst = gate: llama.cpp allocates them so unless a buffer boundary
// falls between), else into a buffer of its own
static bool axera_mxfp4_gu_adjacent(const ggml_tensor * g, const ggml_tensor * u) {
    return g->buffer == u->buffer && (const char *) u->data == (const char *) g->data + ggml_nbytes(g);
}

static void axera_repack_mxfp4_gu(const ggml_tensor * g, const ggml_tensor * u, uint8_t * dst,
                                  const ggml_backend_axera_buffer_context * db) {
    axera_check_mxfp4(g);
    axera_check_mxfp4(u);
    GGML_ASSERT(ggml_are_same_shape(g, u));
    const int64_t K = g->ne[0], rows = g->ne[1], n = g->ne[2];
    const size_t  stride = g->nb[2], half = (size_t) (rows * K / 2);
    std::vector<uint8_t> copy;  // in place: the sources, before they are overwritten
    const uint8_t *      gs = (const uint8_t *) g->data, *us = (const uint8_t *) u->data;
    if (dst == g->data) {
        GGML_ASSERT(axera_mxfp4_gu_adjacent(g, u));
        copy.assign(gs, gs + 2 * ggml_nbytes(g));
        gs = copy.data();
        us = copy.data() + ggml_nbytes(g);
    }
    axera_host_pool().run(n, [&](int64_t e) {
        uint8_t * dq = dst + 2 * e * stride;
        uint8_t * ds = dq + 2 * half;
        axera_mxfp4_rows(gs + e * stride, rows, K, dq, ds);
        axera_mxfp4_rows(us + e * stride, rows, K, dq + half, ds + rows * K / 32);
    });
    db->flush(dst, 2 * ggml_nbytes(g));
}

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
    if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) ||
        (w->type == GGML_TYPE_F32 && ggml_nbytes(w) == 2 * (size_t) size)) {  // quantized / f16 / f32 -> bf16 input
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

// MXFP8 kernel input w_<x> (elements) or s_<x> (scales), quantized once per layer from the GGUF tensor w of w_<x> and
// shared by the layer's decode, prefill and split attention kernels
static AX_ENGINE_IO_BUFFER_T axera_q35_mx_input(axera_q35_layer & ly, const ggml_tensor * w, const std::string & n,
                                                uint32_t size) {
    const std::string wn = "w_" + n.substr(2);
    auto              mx = ly.mx.find(wn);
    if (mx == ly.mx.end()) {
        axera_cmm q, sc;
        axera_quantize_mxfp8(w, q, sc);
        mx = ly.mx.emplace(wn, std::make_pair(q.get(), sc.get())).first;
        ly.own.push_back(std::move(q));
        ly.own.push_back(std::move(sc));
    }
    auto * b = n[0] == 'w' ? mx->second.first : mx->second.second;
    if (b->size < size) {
        GGML_ABORT("qwen35 template: %s is %zu bytes, kernel wants %u\n", n.c_str(), b->size, size);
    }
    return axera_io(b, b->vir, size);
}

// binds one layer's kernel IO: GGUF tensors in CMM directly where the layout matches, converted copies otherwise
static void axera_q35_bind_layer(axera_q35 & m, int il, const std::map<std::string, ggml_tensor *> & t) {
    auto get = [&](const char * name) -> ggml_tensor * {
        char key[96];
        snprintf(key, sizeof(key), "blk.%d.%s", il + m.off, name);
        auto it = t.find(key);
        if (it == t.end()) {
            GGML_ABORT("qwen35 template: tensor %s not in the graph\n", key);
        }
        return it->second;
    };
    // llama.cpp recurrent cache of this layer (cache_r_l<il> / cache_s_l<il>), if one sequence row in CMM holds the state
    auto cache = [&](const char * which, uint32_t size) -> ggml_tensor * {
        auto it = t.find("cache_" + std::string(which) + "_l" + std::to_string(il + m.off));
        if (it == t.end() || !ggml_backend_buffer_is_axera(it->second->buffer) || it->second->nb[1] < size) {
            return nullptr;
        }
        return it->second;
    };
    axera_q35_layer & ly = m.layers[il];
    ly.gdn               = t.count("blk." + std::to_string(il + m.off) + ".ssm_a") > 0;
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
            // MXFP8 kernel: w_<x> elements and s_<x> scales, quantized from the GGUF tensor of w_<x>
            const char * g = axera_q35_gguf_name(m, "w_" + n.substr(2));
            if (!g) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            ly.in[i] = axera_q35_mx_input(ly, get(g), n, size);
            continue;
        } else {
            const char * g = axera_q35_gguf_name(m, n);
            if (!g) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            ly.in[i] = axera_q35_gguf_input(m, ly, get(g), n, size);
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
            b = m.moe ? m.BASE.get() : xout;  // MoE: the experts kernel adds the routed experts and writes xout
        } else if (n == "h2_out" || n == "logits_out") {
            b = (n == "h2_out" ? m.H2 : m.LOGITS).get();
        } else if (m.kv_llama && (n == "kc" || n == "vc")) {
            // the llama.cpp KV cache of this layer: [cells][heads x dim] f16, the layout of the _kvc kernels
            // N cells per stream, allocated with room for LMAX after the last stream (the IO size must be the
            // kernel's); the kernels only touch the blocks below the runtime block count, all within the stream
            ggml_tensor * c = t.at("cache_" + std::string(n == "kc" ? "k" : "v") + "_l" + std::to_string(il + m.off));
            GGML_ASSERT(c->nb[1] * m.LMAX == size &&
                        ggml_backend_buft_get_alloc_size(c->buffer->buft, c) >= (c->ne[2] - 1) * c->nb[2] + size);
            axera_q35_layer::kv_region kr;
            kr.b      = (ggml_backend_axera_buffer_context *) c->buffer->context;
            kr.base   = (char *) c->data;
            kr.size   = size;
            kr.stride = c->nb[2];
            kr.row    = c->nb[1];
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
    if ((!ly.gdn && (m.flash || !pf.attn)) || !pf.gdn) {
        // split attention: axera_q35_bind_flash; multi-core only: axera_q35_bind_mc
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
            const char * g  = axera_q35_gguf_name(m, n);
            auto         it = !g ? t.end() : t.find("blk." + std::to_string(il + m.off) + "." + g);
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
        if (n == "out" || n == "h2_out" || n == "logits_out") {  // MoE: the experts add to the base, into xout
            auto * b  = n == "h2_out" ? pf.H2.get() : n == "logits_out" ? pf.LOG.get() : m.moe ? pf.BASE.get() : xout;
            io.out[i] = axera_io(b, b->vir, k->info->pOutputs[i].nSize);
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
    if (ly.gdn && pf.gdn_rs) {  // the snapshot kernel: the same inputs and outputs by name, plus the snapshots
        ggml_axera_kernel * kr = pf.gdn_rs;
        io.rin.assign(kr->info->nInputSize, {});
        io.rout.assign(kr->info->nOutputSize, {});
        io.rs_state.assign(pf.P, -1);
        io.rs_conv.assign(pf.P, -1);
        for (uint32_t i = 0; i < kr->info->nInputSize; ++i) {
            const std::string n = kr->info->pInputs[i].pName;
            const int         j = k->input(n.c_str());
            GGML_ASSERT(j >= 0 && k->info->pInputs[j].nSize == kr->info->pInputs[i].nSize);
            io.rin[i] = io.in[j];
            io.rsi    = n == "state_in" ? (int) i : io.rsi;
            io.rci    = n == "conv_in" ? (int) i : io.rci;
        }
        for (uint32_t i = 0; i < kr->info->nOutputSize; ++i) {
            const std::string n = kr->info->pOutputs[i].pName;
            if (n.rfind("state_rs", 0) == 0 || n.rfind("conv_rs", 0) == 0) {
                const bool st = n[0] == 's';
                const int  sl = atoi(n.c_str() + (st ? 8 : 7));
                GGML_ASSERT(sl >= 1 && sl < pf.P);
                (st ? io.rs_state : io.rs_conv)[sl] = (int) i;
                continue;
            }
            const int j = k->output(n.c_str());
            GGML_ASSERT(j >= 0 && k->info->pOutputs[j].nSize == kr->info->pOutputs[i].nSize);
            io.rout[i] = io.out[j];
            io.rso     = n == "state_out" ? (int) i : io.rso;
            io.rco     = n == "conv_out" ? (int) i : io.rco;
        }
        GGML_ASSERT(io.rsi >= 0 && io.rci >= 0 && io.rso >= 0 && io.rco >= 0);
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
static bool axera_q35_load_flash(axera_q35 & m, int R, bool seg_only = false) {
    if (m.fd.count(R)) {
        return seg_only || m.fd.at(R).head != nullptr;
    }
    const std::string r  = std::to_string(R);
    // MoE: the tail ends with the MoE's dense part (base, h2, router logits), qwen35moe_attn_fd_tail<R>
    std::string hn = "qwen35_attn_fd_head" + r, tn = (m.moe ? "qwen35moe" : "qwen35") + std::string("_attn_fd_tail") + r;
    std::string       sn = "qwen35_attn_fd_seg" + r + "_c" + std::to_string(m.S);
    // head / tail in the run's weight format where those kernels exist: MXFP8 for decode with MXFP8 weights, for
    // prefill rows with the MXFP8 prefill kernels
    if (axera_weights_mx8() && (R == 1 || m.pre_mx) && std::ifstream(axera_kernels().path(hn + "_mx8")).good() &&
        std::ifstream(axera_kernels().path(tn + "_mx8")).good()) {
        hn += "_mx8";
        tn += "_mx8";
    }
    if (axera_cores() > 1 && std::ifstream(axera_kernels().path(sn + "_x" + std::to_string(axera_cores()))).good()) {
        sn += "_x" + std::to_string(axera_cores());  // the multi-core segment kernel
    }
    const bool ht = std::ifstream(axera_kernels().path(hn)).good() && std::ifstream(axera_kernels().path(tn)).good();
    if (!std::ifstream(axera_kernels().path(sn)).good() || (!ht && !seg_only)) {
        return false;
    }
    axera_q35::fd_kernels fk;
    fk.head = ht ? axera_kernels().get(hn) : nullptr;
    fk.seg  = axera_kernels().get(sn);
    fk.tail = ht ? axera_kernels().get(tn) : nullptr;
    const AX_ENGINE_IOMETA_T & o = fk.seg->info->pOutputs[fk.seg->output("o_out")];
    GGML_ASSERT(o.nShapeSize == 3 || o.nShapeSize == 4);
    const int lead = o.nShapeSize - 3;  // [cores] [hkv][gr][d]
    fk.cores = lead ? o.pShape[0] : 1;
    fk.hkv   = o.pShape[lead];
    fk.gr    = o.pShape[lead + 1];
    fk.d     = o.pShape[lead + 2];
    GGML_ASSERT(axera_input_size(fk.seg, "nblk") == (size_t) (fk.cores > 1 ? 2 * fk.cores : 1) * sizeof(int32_t));
    fk.BLK  = axera_cmm_alloc(axera_input_size(fk.seg, "nblk"));
    const AX_ENGINE_IOMETA_T & mt = fk.seg->info->pInputs[fk.seg->input("mask")];
    GGML_ASSERT(mt.nShapeSize == 3 && mt.pShape[1] == R && (int64_t) mt.pShape[0] * mt.pShape[2] == m.S);
    fk.bc   = mt.pShape[2];
    // head outputs: q [R][HQ][D] bf16 (the segment input), gate likewise, new K / V rows [R][HKV * D] f16
    const size_t qsz = axera_input_size(fk.seg, "qs"), kvsz = (size_t) R * fk.hkv * fk.d * sizeof(ggml_fp16_t);
    GGML_ASSERT(!fk.head || (axera_output_size(fk.head, "q_out") == qsz && axera_output_size(fk.head, "k_out") == kvsz));
    fk.Q    = axera_cmm_alloc(qsz);
    fk.GATE = axera_cmm_alloc(qsz);
    fk.K    = axera_cmm_alloc(kvsz);
    fk.V    = axera_cmm_alloc(kvsz);
    fk.M    = axera_cmm_alloc(axera_output_size(fk.seg, "m_out"));
    fk.Lsum = axera_cmm_alloc(axera_output_size(fk.seg, "l_out"));
    fk.O    = axera_cmm_alloc(axera_output_size(fk.seg, "o_out"));
    fk.ATT  = fk.tail ? axera_cmm_alloc(axera_input_size(fk.tail, "attn")) : nullptr;
    m.fd.emplace(R, std::move(fk));
    return true;
}

// the split attention kernels and state of run kind v (-1: decode, else m.pre[v]): by rows, a group copy by fdk
static int axera_q35_fd_key(const axera_q35 & m, int v) {
    return v < 0 ? 1 : m.pre[v].fdk ? m.pre[v].fdk : m.pre[v].P;
}

// a copy of the split attention kernels for R rows under key: same kernels, buffers of its own
static void axera_q35_clone_flash(axera_q35 & m, int R, int key) {
    const axera_q35::fd_kernels & a = m.fd.at(R);
    axera_q35::fd_kernels         fk;
    fk.head  = a.head;
    fk.seg   = a.seg;
    fk.tail  = a.tail;
    fk.hkv   = a.hkv;
    fk.gr    = a.gr;
    fk.d     = a.d;
    fk.bc    = a.bc;
    fk.cores = a.cores;
    for (auto [dst, src] : { std::pair(&fk.Q, &a.Q), std::pair(&fk.GATE, &a.GATE), std::pair(&fk.K, &a.K),
                             std::pair(&fk.V, &a.V), std::pair(&fk.M, &a.M), std::pair(&fk.Lsum, &a.Lsum),
                             std::pair(&fk.O, &a.O), std::pair(&fk.ATT, &a.ATT), std::pair(&fk.BLK, &a.BLK) }) {
        *dst = *src ? axera_cmm_alloc((*src)->size) : nullptr;
    }
    m.fd[key] = std::move(fk);
}

// split attention IO of attention layer il for run kind v (-1: decode, else m.pre[v]): head and tail on the run's
// activations and RoPE tables and the GGUF weights; the segment kernel's KV and mask inputs are rebound per call
static void axera_q35_bind_flash(axera_q35 & m, int il, int v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &       ly = m.layers[il];
    axera_q35::fd_kernels & fk = m.fd.at(axera_q35_fd_key(m, v));
    auto & A   = v < 0 ? m.A : m.pre[v].A;
    auto & B   = v < 0 ? m.B : m.pre[v].B;
    auto * xin = (il % 2 == 0 ? A : B).get(), *xout = (il % 2 == 0 ? B : A).get();
    ly.fd.resize(m.pre.size() + 1);
    axera_q35_layer::fd_io & io = ly.fd[v + 1];
    if (ly.kv.empty()) {  // the layer's KV cache: the host writes new cells, the segment kernel reads S cells
        for (const char * kv : { "k", "v" }) {
            const ggml_tensor * c = t.at("cache_" + std::string(kv) + "_l" + std::to_string(il + m.off));
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
            const bool mx = (n.rfind("w_", 0) == 0 || n.rfind("s_", 0) == 0) && k->input(("s_" + n.substr(2)).c_str()) >= 0;
            const char * g  = axera_q35_gguf_name(m, mx ? "w_" + n.substr(2) : n);
            auto         it = !g ? t.end() : t.find("blk." + std::to_string(il + m.off) + "." + g);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            in[i] = mx ? axera_q35_mx_input(ly, it->second, n, size) : axera_q35_gguf_input(m, ly, it->second, n, size);
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
         { { "qs", at(fk.Q) }, { "kc", buf(kk.b, kk.base) }, { "vc", buf(vv.b, vv.base) }, { "nblk", at(fk.BLK) },
           { "mask", at(fk.Q) }, { "m_out", at(fk.M) }, { "l_out", at(fk.Lsum) }, { "o_out", at(fk.O) } });
    io.kc   = fk.seg->input("kc");
    io.vc   = fk.seg->input("vc");
    io.mask = fk.seg->input("mask");
    std::map<std::string, buf> tb = { { "x", buf(xin, (char *) xin->vir) }, { "attn", at(fk.ATT) },
                                      { "gate", at(fk.GATE) }, { "out", buf(xout, (char *) xout->vir) } };
    if (m.moe) {  // the experts add to the base, into xout
        tb["out"]        = at(v < 0 ? m.BASE : m.pre[v].BASE);
        tb["h2_out"]     = at(v < 0 ? m.H2 : m.pre[v].H2);
        tb["logits_out"] = at(v < 0 ? m.LOGITS : m.pre[v].LOG);
    }
    bind(fk.tail, io.tin, io.tout, tb);
}

// the multi-core prefill chain for P tokens (q35_mc.py kernels) and its buffers; the segment kernel of the split
// attention for P rows (any core count) must exist for the KV segment
static bool axera_q35_load_mc(axera_q35 & m, int P) {
    if (m.mc.count(P)) {
        return true;
    }
    const std::string sfx = std::to_string(P) + "_x" + std::to_string(axera_cores());
    std::vector<std::string> names;
    for (const char * k : { "qwen35_gdn_mc", "qwen35_mlp_mc", "qwen35_attn_head_mc", "qwen35_attn_tail_mc" }) {
        names.push_back(k + sfx);
        if (!std::ifstream(axera_kernels().path(names.back())).good()) {
            return false;
        }
    }
    if (axera_cores() < 2 || !m.kv_llama || !m.rollback || !axera_q35_load_flash(m, P, true)) {
        return false;
    }
    axera_q35::fd_kernels &  fk = m.fd.at(P);
    axera_q35::mc_kernels    mk;
    mk.gdn  = axera_kernels().get(names[0]);
    mk.mlp  = axera_kernels().get(names[1]);
    mk.head = axera_kernels().get(names[2]);
    mk.tail = axera_kernels().get(names[3]);
    const AX_ENGINE_IOMETA_T & ma = mk.tail->info->pInputs[mk.tail->input("m_all")];
    GGML_ASSERT(ma.nShapeSize == 4 && ma.pShape[1] == fk.hkv && ma.pShape[2] == fk.gr);
    mk.np = ma.pShape[0];
    mk.ms = axera_input_size(mk.tail, "m_all") / mk.np;
    mk.ls = axera_input_size(mk.tail, "l_all") / mk.np;
    mk.os = axera_input_size(mk.tail, "o_all") / mk.np;
    GGML_ASSERT(mk.ms * fk.cores == axera_output_size(fk.seg, "m_out") && mk.os * fk.cores == axera_output_size(fk.seg, "o_out"));
    GGML_ASSERT(axera_output_size(mk.head, "q_out") == fk.Q->size && axera_output_size(mk.head, "k_out") == fk.K->size);
    const size_t part = axera_input_size(mk.mlp, "part");
    mk.BASE1 = axera_cmm_alloc(axera_input_size(mk.mlp, "base"));
    mk.PART0 = axera_cmm_alloc(part);
    mk.PART1 = axera_cmm_alloc(part);
    mk.ZERO  = axera_cmm_alloc(part);
    memset(mk.ZERO->vir, 0, part);
    mk.ZERO->flush(mk.ZERO->vir, part);
    mk.MALL = axera_cmm_alloc(axera_input_size(mk.tail, "m_all"));
    mk.LALL = axera_cmm_alloc(axera_input_size(mk.tail, "l_all"));
    mk.OALL = axera_cmm_alloc(axera_input_size(mk.tail, "o_all"));
    memset(mk.LALL->vir, 0, mk.LALL->size);  // slots without data: m = -3e38 (set per batch), finite l / o
    memset(mk.OALL->vir, 0, mk.OALL->size);
    mk.LALL->flush(mk.LALL->vir, mk.LALL->size);
    mk.OALL->flush(mk.OALL->vir, mk.OALL->size);
    m.mc.emplace(P, std::move(mk));
    return true;
}

// the multi-core chain IO of layer il for chunk size v: the residual base / partials, the head's outputs into the
// split attention buffers, the tail reading the merge slots, the weights from the GGUF, the GDN state in place
static void axera_q35_bind_mc(axera_q35 & m, int il, size_t v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &       ly = m.layers[il];
    axera_q35::prefill &    pf = m.pre[v];
    axera_q35::mc_kernels & mk = m.mc.at(pf.P);
    axera_q35::fd_kernels & fk = m.fd.at(pf.P);
    axera_q35_layer::mc_io & io = ly.mc[pf.P];
    if (ly.kv.empty() && !ly.gdn) {  // as axera_q35_bind_flash
        for (const char * kv : { "k", "v" }) {
            const ggml_tensor * c = t.at("cache_" + std::string(kv) + "_l" + std::to_string(il + m.off));
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
    auto at   = [](const axera_cmm & c) { return buf(c.get(), (char *) c->vir); };
    ggml_axera_kernel * kd = ly.gdn ? m.gdn : nullptr;  // converted inputs (a_log, conv_w) and the state rows
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
            const int j = kd && (n == "a_log" || n == "conv_w" || n == "state_in" || n == "conv_in") ? kd->input(n.c_str()) : -1;
            if (j >= 0) {
                GGML_ASSERT(kd->info->pInputs[j].nSize == size);
                in[i] = ly.in[j];
                if (n == "state_in" || n == "conv_in") {
                    (n == "state_in" ? io.si : io.ci)   = (int) i;
                    (n == "state_in" ? io.dsi : io.dci) = j;
                }
                continue;
            }
            const char * g  = axera_q35_gguf_name(m, n);
            auto         it = !g ? t.end() : t.find("blk." + std::to_string(il + m.off) + "." + g);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            in[i] = axera_q35_gguf_input(m, ly, it->second, n, size);
        }
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            const std::string n = k->info->pOutputs[i].pName;
            if (n == "state_out" || n == "conv_out") {  // in place
                const int si = n == "state_out" ? io.si : io.ci;
                GGML_ASSERT(si >= 0);
                out[i]                             = in[si];
                (n == "state_out" ? io.so : io.co) = (int) i;
                continue;
            }
            const auto & b = bufs.at(n);
            out[i]         = axera_io(b.first, b.second, k->info->pOutputs[i].nSize);
        }
    };
    const buf base0 = at(pf.A), part0 = il == 0 ? at(mk.ZERO) : at(mk.PART0);
    if (ly.gdn) {
        bind(mk.gdn, io.ain, io.aout, { { "base", base0 }, { "part", part0 }, { "base_out", at(mk.BASE1) },
                                        { "part_out", at(mk.PART1) } });
    } else {
        bind(mk.head, io.ain, io.aout,
             { { "base", base0 }, { "part", part0 }, { "cos", at(pf.COS) }, { "sin", at(pf.SIN) },
               { "base_out", at(mk.BASE1) }, { "q_out", at(fk.Q) }, { "gate_out", at(fk.GATE) }, { "k_out", at(fk.K) },
               { "v_out", at(fk.V) } });
        const auto & kk = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
        const auto & vv = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
        bind(fk.seg, io.sin, io.sout,
             { { "qs", at(fk.Q) }, { "kc", buf(kk.b, kk.base) }, { "vc", buf(vv.b, vv.base) }, { "nblk", at(fk.BLK) },
               { "mask", at(fk.Q) }, { "m_out", at(mk.MALL) }, { "l_out", at(mk.LALL) }, { "o_out", at(mk.OALL) } });
        io.kc   = fk.seg->input("kc");
        io.vc   = fk.seg->input("vc");
        io.mask = fk.seg->input("mask");
        io.mo   = fk.seg->output("m_out");
        io.lo   = fk.seg->output("l_out");
        io.oo   = fk.seg->output("o_out");
        bind(mk.tail, io.tin, io.tout, { { "m_all", at(mk.MALL) }, { "l_all", at(mk.LALL) }, { "o_all", at(mk.OALL) },
                                         { "gate", at(fk.GATE) }, { "part_out", at(mk.PART1) } });
    }
    bind(mk.mlp, io.min, io.mout, { { "base", at(mk.BASE1) }, { "part", at(mk.PART1) }, { "base_out", base0 },
                                    { "part_out", at(mk.PART0) } });
}

static void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                           const ggml_tensor * lm, const ggml_tensor * rope) {
    m.L    = L;
    m.H    = H;
    const bool mx8 = axera_weights_mx8();
    m.moe          = t.count("blk." + std::to_string(m.off) + ".ffn_gate_inp.weight") > 0;  // Qwen3.5 MoE: qwen35moe_* kernels
    const std::string kp = m.moe ? "qwen35moe" : "qwen35";
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
    m.rope_mode  = rp[2];
    memcpy(m.sections, rp + 11, sizeof(m.sections));
    // the llama.cpp KV cache is usable if every attention layer's cache is f16 cells of [kv heads x head dim] in CMM
    // (N cells, padded at allocation to the context LMAX of the smallest qwen35_attn*_kvc_c<LMAX> kernel that holds
    // them), V not transposed (flash-attention layout)
    const std::string attn_name = kp + (mx8 ? "_attn_mx8" : "_attn");
    {
        auto ki = t.find("attn_inp_k_idxs"), vi = t.find("attn_inp_v_idxs");
        m.kv_llama = ki != t.end() && vi != t.end() && ki->second->ne[0] == vi->second->ne[0];
    }
    m.N = 0;
    for (int il = 0; il < L && m.kv_llama; ++il) {
        if (t.count("blk." + std::to_string(il + m.off) + ".ssm_a")) {
            continue;
        }
        for (const char * kv : { "k", "v" }) {
            auto it = t.find("cache_" + std::string(kv) + "_l" + std::to_string(il + m.off));
            m.kv_llama = m.kv_llama && it != t.end() && it->second->type == GGML_TYPE_F16 &&
                         ggml_backend_buffer_is_axera(it->second->buffer) && ggml_is_contiguous(it->second) &&
                         it->second->ne[3] == 1 && (m.N == 0 || it->second->ne[1] == m.N);
            m.N = m.kv_llama ? it->second->ne[1] : m.N;
        }
    }
    // a KV cache no fused attention kernel holds (or GGML_AXERA_FLASH=1): split attention over S-cell segments
    const int64_t ctx = m.kv_llama ? axera_kernels().ctx_for(attn_name + "_kvc", m.N) : 0;
    m.fd.clear();
    m.mc.clear();
    m.S     = axera_flash_seg();
    m.flash = m.kv_llama && m.S > 0 && (ctx == 0 || axera_flash_forced()) && axera_q35_load_flash(m, 1);
    if (m.kv_llama && ctx == 0 && !m.flash) {
        GGML_LOG_WARN("%s: no %s_kvc_c<LMAX> kernel holds the %" PRId64 "-cell KV cache and no split attention "
                      "kernels qwen35_attn_fd_*; lower -c\n", __func__, attn_name.c_str(), m.N);
    }
    m.kv_llama = m.kv_llama && (ctx > 0 || m.flash);
    m.ctx_sfx  = m.kv_llama && !m.flash ? "_kvc_c" + std::to_string(ctx) : "";
    if (m.kv_llama && !m.flash) {
        // fused kernels; split attention over one segment of the whole tier where its kernels exist (long batches)
        m.fd.clear();
        m.mc.clear();
        m.S = m.S > 0 && ctx % m.S == 0 ? std::min<int64_t>(ctx, m.S) : ctx;  // the split kernels' segment
        axera_q35_load_flash(m, 1);
    }
    m.gdn  = axera_kernels().get(kp + (mx8 ? "_gdn_mx8" : "_gdn"));
    m.attn = m.flash ? nullptr : axera_kernels().get(attn_name + m.ctx_sfx);
    if (m.flash) {  // the KQ mask is cut per segment kernel (fd_kernels::bc); whole 64-cell blocks here
        m.LMAX = (int) m.S;
        m.BC   = 64;
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
        // one MXFP8 copy per output projection tensor, shared with the MTP draft context of the same model
        static std::mutex mu;
        static std::map<const ggml_tensor *, std::pair<std::weak_ptr<ggml_backend_axera_buffer_context>,
                                                       std::weak_ptr<ggml_backend_axera_buffer_context>>> shared;
        std::lock_guard<std::mutex> lock(mu);
        auto & sh = shared[lm];
        m.lm_q    = sh.first.lock();
        m.lm_s    = sh.second.lock();
        if (!m.lm_q || !m.lm_s) {
            axera_cmm q, sc;
            axera_quantize_mxfp8(lm, q, sc);
            m.lm_q = std::move(q);
            m.lm_s = std::move(sc);
            sh     = { m.lm_q, m.lm_s };
        }
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
    if (m.moe) {  // the routed experts kernel and its IO: the layer kernels write BASE / H2 / LOGITS
        auto topk = t.find("ffn_moe_topk-" + std::to_string(m.off));
        GGML_ASSERT(topk != t.end() && "qwen35 template: no ffn_moe_topk node in the MoE graph");
        m.top_k    = (int) topk->second->ne[0];
        m.n_expert = (int) t.at("blk." + std::to_string(m.off) + ".ffn_gate_inp.weight")->ne[1];
        m.BASE     = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        m.H2       = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        m.LOGITS   = axera_cmm_alloc(m.n_expert * sizeof(float));
        m.RW       = axera_cmm_alloc(m.top_k * sizeof(float));
        for (int gu = 0; gu < 2; ++gu) {
            // the gate-up kernel's multi-core build (_x<cores>) writes the cores' partial sums, added on the host
            std::string name = "qwen35moe_experts" + std::to_string(m.top_k) + (gu ? "_gu" : "") + "_mxfp4";
            name             = gu ? axera_multi_core(name) : name;
            auto &            ek   = m.ex[gu];
            ek                     = {};
            if (gu && !std::ifstream(axera_kernels().path(name)).good()) {
                continue;  // optional
            }
            ek.k = axera_kernels().get(name);
            GGML_ASSERT(size_of(ek.k, "rw") == m.top_k * sizeof(float) && size_of(ek.k, "h2") == H * sizeof(ggml_bf16_t));
            ek.in.assign(ek.k->info->nInputSize, {});
            for (int g = 0; g < 4; ++g) {
                ek.w[g].assign(m.top_k, -1);
                ek.s[g].assign(m.top_k, -1);
            }
            for (uint32_t i = 0; i < ek.k->info->nInputSize; ++i) {
                const std::string n  = ek.k->info->pInputs[i].pName;
                const uint32_t    sz = ek.k->info->pInputs[i].nSize;
                if (n == "h2" || n == "base" || n == "rw") {
                    auto & c = n == "h2" ? m.H2 : n == "base" ? m.BASE : m.RW;
                    ek.in[i] = axera_io(c.get(), c->vir, sz);
                    continue;
                }
                const char * kinds = "gudc";  // w_<kind><expert> / s_<kind><expert>
                const char * kc    = n.size() > 3 ? strchr(kinds, n[2]) : nullptr;
                const int    e     = kc ? atoi(n.c_str() + 3) : -1;
                if ((n[0] != 'w' && n[0] != 's') || n[1] != '_' || !kc || e < 0 || e >= m.top_k) {
                    GGML_ABORT("qwen35 template: unexpected experts kernel input %s\n", n.c_str());
                }
                (n[0] == 'w' ? ek.w : ek.s)[kc - kinds][e] = (int) i;
            }
            const int po = ek.k->output("part_out");  // [cores, 1, H] f32
            if (po >= 0) {
                ek.PART = axera_cmm_alloc(ek.k->info->pOutputs[po].nSize);
                GGML_ASSERT(ek.PART->size >= (size_t) H * sizeof(float));
            }
            for (int par = 0; par < 2; ++par) {  // even layers write m.B, odd ones m.A
                auto & x = par == 0 ? m.B : m.A;
                ek.out[par].assign(ek.k->info->nOutputSize, {});
                if (po >= 0) {
                    ek.out[par][po] = axera_io(ek.PART.get(), ek.PART->vir, ek.k->info->pOutputs[po].nSize);
                } else {
                    ek.out[par][ek.k->output("out")] = axera_io(x.get(), x->vir, H * sizeof(ggml_bf16_t));
                }
            }
        }
        m.slots = {};
        std::string sn;  // the slots kernel with the most rows per slot (fewest slots: ~25 us per projection)
        for (const char * g : { "64", "32", "16", "8" }) {
            const std::string n = std::string("qwen35moe_slots32_g") + g + "_gu_mxfp4";
            if (sn.empty() && std::ifstream(axera_kernels().path(n)).good()) {
                sn = n;
            }
        }
        if (const char * e = std::getenv("GGML_AXERA_MOE_SLOTS")) {  // e.g. qwen35moe_slots32_g8_gu_mxfp4
            sn = e;
        }
        sn = sn.empty() ? sn : axera_multi_core(sn);
        if (m.ex[1].k && !sn.empty()) {
            auto & sk = m.slots;
            sk.k      = axera_kernels().get(sn);
            sk.x      = sk.k->input("x");
            sk.rw     = sk.k->input("rw");
            sk.out    = sk.k->output("out");
            GGML_ASSERT(sk.x >= 0 && sk.rw >= 0 && sk.out >= 0 && sk.k->info->pInputs[sk.x].nShapeSize == 3);
            sk.S = sk.k->info->pInputs[sk.x].pShape[0];
            sk.G = sk.k->info->pInputs[sk.x].pShape[1];
            GGML_ASSERT(size_of(sk.k, "x") == (uint32_t) (sk.S * sk.G * H) * sizeof(ggml_bf16_t) &&
                        size_of(sk.k, "rw") == (uint32_t) (sk.S * sk.G) * sizeof(float));
            sk.in.assign(sk.k->info->nInputSize, {});
            sk.outs.assign(sk.k->info->nOutputSize, {});
            for (int g = 0; g < 2; ++g) {
                sk.w[g].assign(sk.S, -1);
                sk.s[g].assign(sk.S, -1);
            }
            for (uint32_t i = 0; i < sk.k->info->nInputSize; ++i) {
                const std::string n = sk.k->info->pInputs[i].pName;
                if (n == "x" || n == "rw") {
                    continue;
                }
                const int e = n.size() > 3 ? atoi(n.c_str() + 3) : -1;
                if ((n[0] != 'w' && n[0] != 's') || n[1] != '_' || (n[2] != 'c' && n[2] != 'd') || e < 0 || e >= sk.S) {
                    GGML_ABORT("qwen35 template: unexpected slots kernel input %s\n", n.c_str());
                }
                (n[0] == 'w' ? sk.w : sk.s)[n[2] == 'd'][e] = (int) i;
            }
        }
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
    // each layer's routed experts, repacked for the kernels: gate-up when the gate-up experts kernel exists (a GGUF
    // ffn_gate_up_exps, gate and up in place when back to back, else into a buffer of their own), else apart
    static std::set<const void *> gu_done;  // gate tensors repacked in place
    m.gu_own.clear();
    for (int il = 0; il < L && m.moe; ++il) {
        auto &              ly = m.layers[il];
        const std::string   b  = "blk." + std::to_string(il + m.off) + ".";
        const auto          gu = t.find(b + "ffn_gate_up_exps.weight");
        const ggml_tensor * w[3];
        w[0] = gu != t.end() ? gu->second : t.at(b + "ffn_gate_exps.weight");
        w[1] = gu != t.end() ? nullptr : t.at(b + "ffn_up_exps.weight");
        w[2] = t.at(b + "ffn_down_exps.weight");
        for (const ggml_tensor * x : w) {
            if (x && (x->type != GGML_TYPE_MXFP4 || !ggml_backend_buffer_is_axera(x->buffer))) {
                GGML_ABORT("qwen35 template: %s is %s; the MoE kernels take MXFP4 experts in CMM (llama-quantize "
                           "MXFP4_MOE, -ngl 99)\n", x->name, ggml_type_name(x->type));
            }
        }
        if (gu != t.end() && !m.ex[1].k) {
            GGML_ABORT("qwen35 template: %s needs the kernel qwen35moe_experts%d_gu_mxfp4\n", w[0]->name, m.top_k);
        }
        ly.exps_gu = m.ex[1].k != nullptr;
        // the gate-up weight: buffer, start, bytes per expert
        ggml_backend_axera_buffer_context * gb = (ggml_backend_axera_buffer_context *) w[0]->buffer->context;
        uint8_t *                           gp = (uint8_t *) w[0]->data;
        if (gu != t.end()) {
            axera_repack_mxfp4(w[0]);
        } else if (ly.exps_gu && axera_mxfp4_gu_adjacent(w[0], w[1])) {
            if (gu_done.insert(w[0]->data).second) {
                axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
            }
        } else if (ly.exps_gu) {
            m.gu_own.push_back(axera_cmm_alloc(2 * ggml_nbytes(w[0])));
            gb = m.gu_own.back().get();
            gp = (uint8_t *) gb->vir;
            axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
        } else {
            axera_repack_mxfp4(w[0]);
            axera_repack_mxfp4(w[1]);
        }
        axera_repack_mxfp4(w[2]);
        const auto & ek = m.ex[ly.exps_gu];
        for (int g = 0; g < 3; ++g) {
            auto & e = ly.exps[g];
            e        = {};
            if (ly.exps_gu && g == 1) {
                continue;
            }
            const bool    two  = ly.exps_gu && g == 0 && gu == t.end();  // gate and up rows of two GGUF tensors
            const int64_t rows = w[g]->ne[1] * (two ? 2 : 1);
            e.b                = g == 0 ? gb : (ggml_backend_axera_buffer_context *) w[g]->buffer->context;
            e.base             = g == 0 ? (char *) gp : (char *) w[g]->data;
            e.data             = (size_t) (w[g]->ne[0] * rows / 2);
            e.scale            = (size_t) (w[g]->ne[0] / 32 * rows);
            e.stride           = e.data + e.scale;
            const int kind     = ly.exps_gu && g == 0 ? 3 : g;
            GGML_ASSERT(e.stride == (two ? 2 : 1) * w[g]->nb[2] && e.data == ek.k->info->pInputs[ek.w[kind][0]].nSize &&
                        e.scale == ek.k->info->pInputs[ek.s[kind][0]].nSize);
        }
    }
    if (!m.gu_own.empty()) {
        GGML_LOG_INFO("%s: %zu MoE layers with gate and up experts apart, repacked into %zu MB of their own\n", __func__,
                      m.gu_own.size(), m.gu_own.size() * m.gu_own[0]->size >> 20);
    }
    // prefill kernels (every chunk size found, largest first). With MXFP8 decode, prefill takes MXFP8 weights too when
    // the kernel directory has MXFP8 prefill kernels (built where prefill is bound by reading the weights, 9B: about
    // half the cycles), else the bf16 GGUF weights (0.8B: compute bound, MXFP8 is slower);
    // GGML_AXERA_PREFILL_WEIGHTS=mxfp8 / bf16 chooses
    const char * pw     = std::getenv("GGML_AXERA_PREFILL_WEIGHTS");
    bool         pre_mx = false;
    for (int P : { 128, 64, 32, 16 }) {
        pre_mx = pre_mx || std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + std::to_string(P) + "_mx8")).good();
    }
    pre_mx = mx8 && (pw != nullptr ? strcmp(pw, "mxfp8") == 0 : pre_mx);
    m.pre_mx = pre_mx;
    m.pre.clear();
    // P = 1: decode on the multi-core chain (GGML_AXERA_MC_DECODE=0: the single-core decode kernels)
    const char * mcd    = std::getenv("GGML_AXERA_MC_DECODE");
    const bool   mc_dec = mcd == nullptr || strcmp(mcd, "0") != 0;
    for (int P : { 256, 128, 64, 32, 16, 8, 4, 3, 2, 1 }) {
        if (m.moe) {  // MoE: qwen35moe_{gdn,attn}_prefill<P> (bf16 weights) with the slots kernel
            // attention: the fused kernel of the context tier, or split (flash) for long contexts
            // the GDN layer in MXFP8 where that kernel exists (prefill reads the weights: about 20% faster)
            std::string gn = "qwen35moe_gdn_prefill" + std::to_string(P);
            gn += mx8 && std::ifstream(axera_kernels().path(gn + "_mx8")).good() ? "_mx8" : "";
            const std::string an = "qwen35moe_attn_prefill" + std::to_string(P) + m.ctx_sfx;
            const bool fused = !m.flash && std::ifstream(axera_kernels().path(an)).good();
            const bool split = m.kv_llama && axera_q35_load_flash(m, P);  // wide masks, or no fused kernel
            if (P == 1 || !m.slots.k || !std::ifstream(axera_kernels().path(gn)).good() || !(fused || split)) {
                continue;
            }
            axera_q35::prefill pf;
            pf.P    = P;
            pf.gdn  = axera_kernels().get(gn);
            pf.attn = fused ? axera_kernels().get(an) : nullptr;
            GGML_ASSERT(size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
            if (pf.attn) {
                pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
                GGML_ASSERT(pf.R == P && (size_t) pf.R * m.LMAX * sizeof(ggml_bf16_t) == size_of(pf.attn, "mask"));
                pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
            }
            ggml_axera_kernel * rk = pf.attn ? pf.attn : m.fd.at(P).head;
            pf.A    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.B    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
            pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
            pf.BASE = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.H2   = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.LOG  = axera_cmm_alloc((size_t) P * m.n_expert * sizeof(float));
            // at most one partly filled slot per expert
            const int    G     = m.slots.G;
            const size_t slots = (size_t) GGML_PAD(std::min(P * m.top_k, m.n_expert + (P * m.top_k + G - 1) / G), m.slots.S);
            const size_t rows  = slots * G;
            if (!m.SX || m.SX->size < rows * H * sizeof(ggml_bf16_t)) {
                m.SX   = axera_cmm_alloc_nc(rows * H * sizeof(ggml_bf16_t));
                m.SRW  = axera_cmm_alloc_nc(rows * sizeof(float));
                m.SOUT = axera_cmm_alloc(rows * H * sizeof(float));
            }
            m.pre.push_back(std::move(pf));
            continue;
        }
        const std::string sfx  = std::to_string(P) + (pre_mx ? "_mx8" : "");
        const std::string asfx = sfx + m.ctx_sfx;
        // the multi-core chain (preferred), and / or the single-core layer kernels
        const bool mc  = !pre_mx && (P > 1 || mc_dec) && axera_q35_load_mc(m, P);
        const bool one = std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx)).good() &&
                         (m.flash ? axera_q35_load_flash(m, P) :
                                    std::ifstream(axera_kernels().path("qwen35_attn_prefill" + asfx)).good());
        if (!mc && !one) {
            continue;
        }
        if (one && m.kv_llama && !m.flash) {
            axera_q35_load_flash(m, P);  // optional
        }
        axera_q35::prefill pf;
        pf.P    = P;
        pf.mc   = mc;
        pf.gdn  = one ? axera_kernels().get("qwen35_gdn_prefill" + sfx) : nullptr;
        if (pf.gdn && std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx + "_rs")).good()) {
            pf.gdn_rs = axera_kernels().get("qwen35_gdn_prefill" + sfx + "_rs");
        }
        pf.attn = one && !m.flash ? axera_kernels().get("qwen35_attn_prefill" + asfx) : nullptr;
        GGML_ASSERT(!pf.gdn || size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
        ggml_axera_kernel * rk = mc ? m.mc.at(P).head : m.flash ? m.fd.at(P).head : pf.attn;
        if (pf.attn) {
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
    // in the llama.cpp memory. In the decode weight format (_mx8); bf16 kernels next to MXFP8 decode kernels use the
    // bf16 GGUF tensors
    for (int Bn : { 8, 4, 2 }) {
        const std::string b  = std::to_string(Bn);
        const std::string bs = mx8 && std::ifstream(axera_kernels().path("qwen35_gdn_batch" + b + "_mx8")).good() ? "_mx8" : "";
        const std::string gn = "qwen35_gdn_batch" + b + bs, an = "qwen35_attn_batch" + b + bs;
        if (m.moe || !m.kv_llama || !m.rollback || !std::ifstream(axera_kernels().path(gn)).good() ||
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
    // MoE groups: copies of the largest chunk when it is split (no fused attention kernel), GGML_AXERA_MOE_GROUP
    // chunks per group (default 8)
    if (m.moe && !m.pre.empty() && !m.pre[0].attn && m.pre[0].P > 1) {
        const char * ge = std::getenv("GGML_AXERA_MOE_GROUP");
        const int    Q  = std::max(1, ge ? atoi(ge) : 8);
        const int    P  = m.pre[0].P;
        m.pre[0].group  = { 0 };
        for (int c = 1; c < Q; ++c) {
            axera_q35::prefill pc;
            pc.P    = P;
            pc.gdn  = m.pre[0].gdn;
            pc.sub  = c;
            pc.fdk  = P + 1000 * c;
            pc.A    = axera_cmm_alloc(m.pre[0].A->size);
            pc.B    = axera_cmm_alloc(m.pre[0].B->size);
            pc.COS  = axera_cmm_alloc(m.pre[0].COS->size);
            pc.SIN  = axera_cmm_alloc(m.pre[0].SIN->size);
            pc.BASE = axera_cmm_alloc(m.pre[0].BASE->size);
            pc.H2   = axera_cmm_alloc(m.pre[0].H2->size);
            pc.LOG  = axera_cmm_alloc(m.pre[0].LOG->size);
            axera_q35_clone_flash(m, P, pc.fdk);
            m.pre[0].group.push_back((int) m.pre.size());
            m.pre.push_back(std::move(pc));
        }
        const int    NT    = P * Q, G = m.slots.G;
        const size_t slots = (size_t) GGML_PAD(std::min(NT * m.top_k, m.n_expert + (NT * m.top_k + G - 1) / G), m.slots.S);
        const size_t rows  = slots * G;
        if (m.SX->size < rows * H * sizeof(ggml_bf16_t)) {
            m.SX   = axera_cmm_alloc_nc(rows * H * sizeof(ggml_bf16_t));
            m.SRW  = axera_cmm_alloc_nc(rows * sizeof(float));
            m.SOUT = axera_cmm_alloc(rows * H * sizeof(float));
        }
    }
    for (size_t v = 0; v < m.pre.size(); ++v) {
        for (int il = 0; il < L; ++il) {
            axera_q35_bind_prefill(m, il, v, t);
            if (m.pre[v].mc) {
                axera_q35_bind_mc(m, il, v, t);
            }
        }
    }
    for (int il = 0; il < L && m.kv_llama; ++il) {
        for (int v = -1; v < (int) m.pre.size() && !m.layers[il].gdn; ++v) {
            const int R = v < 0 ? 1 : m.pre[v].P;
            if (m.fd.count(R) && m.fd.at(R).head) {
                axera_q35_bind_flash(m, il, v, t);
            }
        }
    }
    std::string split;
    for (const auto & [R, fk] : m.fd) {
        if (!fk.head) {
            continue;
        }
        split += (split.empty() ? "" : "/") + std::to_string(R) + (fk.cores > 1 ? "x" + std::to_string(fk.cores) : "");
    }
    std::string chunks, batches;
    for (auto & pf : m.pre) {
        if (pf.sub) {
            continue;
        }
        auto & str = pf.batch ? batches : chunks;
        str += (str.empty() ? "" : "/") + std::to_string(pf.P) + (pf.mc ? "x" + std::to_string(axera_cores()) : "");
    }
    if (!chunks.empty()) {
        chunks += pre_mx ? " (MXFP8)" : " (BF16)";
    }
    m.ready = true;
    GGML_LOG_INFO("%s: Axera qwen35 template: %d layers, context %" PRId64 " (kernels up to %d), %s weights, recurrent "
                  "state %s, KV cache %s, prefill chunks %s, batched decode %s, split attention rows %s%s\n", __func__, L,
                  m.N, m.LMAX,
                  mx8 ? "MXFP8" : "BF16",
                  m.rollback ? "in the llama.cpp cache (checkpoints work)" : "private",
                  m.flash ? "the llama.cpp cache, attention split over segments (flash decoding)" :
                  m.kv_llama ? "the llama.cpp cache (prompt cache works)" : "private", chunks.empty() ? "none" : chunks.c_str(),
                  batches.empty() ? "none" : batches.c_str(), split.empty() ? "none" : split.c_str(),
                  m.flash || split.empty() ? "" : (" from " + std::to_string(axera_split_min()) + " cells").c_str());
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
    axera_timer            tmap("graph name map");
    std::vector<uintptr_t> key;
    key.reserve((size_t) cgraph->n_nodes * (2 + GGML_MAX_SRC));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        key.push_back((uintptr_t) node);
        key.push_back((uintptr_t) node->op);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            key.push_back((uintptr_t) node->src[s]);
        }
    }
    std::map<std::string, ggml_tensor *> & t = ctx->graph_names;
    if (key != ctx->graph_key) {
        t.clear();
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
        ctx->graph_key = std::move(key);
    }
    auto find = [&](const char * name) -> ggml_tensor * {
        auto it = t.find(name);
        return it == t.end() ? nullptr : it->second;
    };
    tmap.stop();
    ggml_tensor * norm0  = find("attn_norm-0");
    ggml_tensor * logits = find("result_output");
    ggml_tensor * inp_pos = find("inp_pos");
    // MTP draft graph (llama.cpp LLM_GRAPH_TYPE_DECODER_MTP): its h input and the model layer of the NextN block
    const ggml_tensor * mtp_h  = find("mtp_h_input");
    int                 mtp_il = -1;
    for (const auto & kv : mtp_h ? t : std::map<std::string, ggml_tensor *>{}) {
        const std::string suffix = ".nextn.eh_proj.weight";
        if (kv.first.rfind("blk.", 0) == 0 && kv.first.size() > suffix.size() &&
            kv.first.compare(kv.first.size() - suffix.size(), suffix.size(), suffix) == 0) {
            mtp_il = atoi(kv.first.c_str() + 4);
        }
    }
    if (mtp_h && (mtp_il < 0 || mtp_h->type != GGML_TYPE_F32)) {
        GGML_ABORT("qwen35 template: MTP graph without a NextN block\n");
    }
    if ((!norm0 && !mtp_h) || !logits || !inp_pos) {
        GGML_LOG_ERROR("qwen35 template: split of %d nodes (%s ... %s) lacks:%s%s%s\n", cgraph->n_nodes,
                       cgraph->n_nodes ? cgraph->nodes[0]->name : "", cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1]->name : "",
                       norm0 ? "" : " attn_norm-0", logits ? "" : " result_output", inp_pos ? "" : " inp_pos");
        GGML_ABORT("qwen35 template: the graph split does not hold the whole model; is the model Qwen3.5 and are all "
                   "layers offloaded (-ngl 99)?\n");
    }

    // layer 0 input: the source of the first RMS norm (MTP: eh_proj of the embedding and h rows, computed below)
    ggml_tensor * x0 = mtp_h ? nullptr : norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(mtp_h || (x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0])));
    x0 = x0 ? x0->src[0] : nullptr;
    const int H = (int) (mtp_h ? mtp_h->ne[0] : x0->ne[0]);
    const int T = (int) (mtp_h ? mtp_h->ne[1] : x0->ne[1]);

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

    int L = mtp_h ? 1 : 0;
    while (!mtp_h && t.count("blk." + std::to_string(L) + ".attn_norm.weight")) {
        ++L;
    }
    axera_q35 &   m   = ctx->q35;
    const int     off = mtp_h ? mtp_il : 0;
    ggml_tensor * w0  = find(("blk." + std::to_string(off) + ".attn_norm.weight").c_str());
    if (!m.ready || m.L != L || m.bound != w0->data || m.lm != lm || m.off != off || m.mtp != (mtp_h != nullptr)) {
        m.mtp = mtp_h != nullptr;
        m.off = off;
        const ggml_tensor * rope = nullptr;
        for (int i = 0; i < cgraph->n_nodes && !rope; i++) {
            rope = cgraph->nodes[i]->op == GGML_OP_ROPE ? cgraph->nodes[i] : nullptr;
        }
        GGML_ASSERT(rope && "qwen35 template: no ROPE node in the graph");
        axera_q35_init(m, L, H, t, lm, rope);
        m.bound = w0->data;
        axera_q35_restart(m);
        m.eh_bf.reset();
        m.EH.reset();
        if (m.mtp) {
            m.eh_bf = axera_to_bf16(t.at("blk." + std::to_string(off) + ".nextn.eh_proj.weight"));
            m.EH    = axera_cmm_alloc(H * sizeof(float));
        }
    }

    // MTP: layer 0's input rows eh_proj([enorm(embedding of the token) | hnorm(h)]) (the norms on the host, the
    // projection [2H] -> [H] on the NPU)
    std::vector<float> x0m;
    if (m.mtp) {
        const std::string   nn = std::to_string(off);
        // the token rows: the embedding of token ids (get_rows), or embeddings given as input (warm-up, images)
        const ggml_tensor * te   = find(("mtp_tok_embd-" + nn).c_str());
        const bool          ids  = te && te->op == GGML_OP_GET_ROWS && te->src[1]->type == GGML_TYPE_I32;
        if (!te || (!ids && (te->type != GGML_TYPE_F32 || te->ne[0] != H || te->ne[1] != T))) {
            GGML_ABORT("qwen35 template: MTP input neither token ids nor embedding rows\n");
        }
        const ggml_tensor * ew  = ids ? te->src[0] : te;
        const ggml_tensor * enw = t.at("blk." + nn + ".nextn.enorm.weight");
        const ggml_tensor * hnw = t.at("blk." + nn + ".nextn.hnorm.weight");
        const ggml_tensor * en  = find(("mtp_enorm-" + nn).c_str());
        const ggml_tensor * hn  = find(("mtp_hnorm-" + nn).c_str());
        GGML_ASSERT(en && hn && enw->type == GGML_TYPE_F32 && hnw->type == GGML_TYPE_F32);
        auto rms_src = [](const ggml_tensor * mul) {  // the RMS_NORM operand of norm * weight
            return mul->src[0]->op == GGML_OP_RMS_NORM ? mul->src[0] : mul->src[1];
        };
        const float eps_e = axera_rms_eps(rms_src(en)), eps_h = axera_rms_eps(rms_src(hn));
        const auto * tt   = ggml_get_type_traits(ew->type);
        GGML_ASSERT((!ids || tt->to_float) && "qwen35 template: MTP embedding type without to_float");
        x0m.resize((size_t) T * H);
        std::vector<float> cat(2 * (size_t) H);
        auto normalize = [&](float * v, const float * w, float eps) {
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                ss += (double) v[i] * v[i];
            }
            const float sc = 1.0f / sqrtf((float) (ss / H) + eps);
            for (int i = 0; i < H; ++i) {
                v[i] = v[i] * sc * w[i];
            }
        };
        for (int r = 0; r < T; ++r) {
            if (ids) {
                const int32_t tok = ((const int32_t *) te->src[1]->data)[r];
                tt->to_float((const char *) ew->data + (size_t) tok * ew->nb[1], cat.data(), H);
            } else {
                memcpy(cat.data(), (const char *) te->data + (size_t) r * te->nb[1], H * sizeof(float));
            }
            normalize(cat.data(), (const float *) enw->data, eps_e);
            memcpy(cat.data() + H, (const char *) mtp_h->data + (size_t) r * mtp_h->nb[1], H * sizeof(float));
            normalize(cat.data() + H, (const float *) hnw->data, eps_h);
            axera_gemv_raw_to(ctx, m.eh_bf.get(), m.eh_bf->vir, H, 2 * H, cat.data(), m.EH.get(), (float *) m.EH->vir);
            memcpy(x0m.data() + (size_t) r * H, m.EH->vir, H * sizeof(float));
        }
    }
    auto x0_row = [&](int r) -> const float * {  // layer 0's input row of token r (f32)
        return m.mtp ? x0m.data() + (size_t) r * H : (const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]);
    };

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
    int64_t rs_written = 1, rs_plane = 0;  // recurrent-state snapshots this graph writes (speculative decoding)
    bool    rs_now     = false;            // this chunk runs the snapshot GDN kernels (a verification batch)
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
                // [state, n_seqs] (or with n_rs_seq: [state, n_seqs, snapshots], snapshot planes nb[2] apart)
                best       = (int64_t) node->view_offs;
                n_seqs     = node->ne[1];
                rs_written = node->ne[2];
                rs_plane   = node->ne[2] > 1 ? (int64_t) (node->nb[2] / s0->nb[1]) : 0;
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
    // span (optional): the first and last visible cell and the number of visible cells
    struct row_span {
        int64_t lo = -1, hi = -1, n = 0;
    };
    auto mask_row = [&](int t, ggml_bf16_t * dst, row_span * span = nullptr) {
        const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;  // finite, as the kernels' own masking: no inf - inf
        const char *   row  = (const char *) kqm->data + (kqm->ne[3] == 1 ? t * kqm->nb[1] :
                                                              (t / kqm->ne[1]) * kqm->nb[3] + (t % kqm->ne[1]) * kqm->nb[1]);
        const int64_t  n    = std::min<int64_t>(W, kqm->ne[0]);
        auto *         d    = (uint16_t *) dst;
        // fast path for the usual 0 / -inf masks (vectorized); any other value falls back to a conversion
        bool other = false;
        if (kqm->type == GGML_TYPE_F16) {
            const auto * s = (const uint16_t *) row;
            for (int64_t c = 0; c < n; ++c) {
                d[c] = s[c] == 0 ? 0 : ninf;
                other |= s[c] != 0 && s[c] != 0xFC00;
            }
        } else {
            const auto * s = (const uint32_t *) row;
            for (int64_t c = 0; c < n; ++c) {
                d[c] = s[c] == 0 ? 0 : ninf;
                other |= s[c] != 0 && s[c] != 0xFF800000u;
            }
        }
        if (other) {
            for (int64_t c = 0; c < n; ++c) {
                const float v = kqm->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[c]) :
                                                             ((const float *) row)[c];
                d[c] = std::isinf(v) && v < 0 ? ninf : ggml_fp32_to_bf16(v).bits;
            }
        }
        std::fill(d + n, d + W, ninf);
        int64_t last = n - 1;
        while (last >= 0 && d[last] == ninf) {
            --last;
        }
        if (span) {
            int64_t first = 0, cnt = 0;
            while (first <= last && d[first] == ninf) {
                ++first;
            }
            for (int64_t c = first; c <= last; ++c) {
                cnt += d[c] != ninf;
            }
            *span = { last < 0 ? -1 : first, last, cnt };
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
    // M-RoPE (positions [4][T]: t, h, w, e): rotary frequency i takes the position of its section, as ggml's
    // ggml_mrope_cache_init (interleaved t / h / w for IMROPE). Text tokens have t = h = w
    const bool mrope = (m.rope_mode & GGML_ROPE_TYPE_MROPE) && inp_pos->ne[0] == 4 * (int64_t) T;
    std::vector<int> sect(m.ROT / 2, 0);
    for (int i = 0; mrope && i < m.ROT / 2; ++i) {
        const int * sc = m.sections;
        const int   n  = sc[0] + sc[1] + sc[2] + sc[3], k = n > 0 ? i % n : 0;
        if (m.rope_mode == GGML_ROPE_TYPE_IMROPE) {
            sect[i] = k % 3 == 1 && k < 3 * sc[1] ? 1 : k % 3 == 2 && k < 3 * sc[2] ? 2 : k % 3 == 0 && k < 3 * sc[0] ? 0 : 3;
        } else {
            sect[i] = k < sc[0] ? 0 : k < sc[0] + sc[1] ? 1 : k < sc[0] + sc[1] + sc[2] ? 2 : 3;
        }
    }
    auto rope_rows = [&](int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) {
        auto * c = (ggml_bf16_t *) cb->vir;  // rows: the positions of tokens t0 .. t0 + n - 1
        auto * s = (ggml_bf16_t *) sb->vir;
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < m.ROT / 2; ++i) {
                const int32_t p           = pos[(mrope ? sect[i] * T : 0) + t0 + r];
                const float a             = (float) p * m.freq_scale * m.inv_freq[i];
                c[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::cos(a));
                s[r * (m.ROT / 2) + i]    = ggml_fp32_to_bf16(std::sin(a));
            }
        }
        cb->flush(cb->vir, cb->size);
        sb->flush(sb->vir, sb->size);
    };
    // split attention: the KQ mask rows of tokens t0 .. t0 + R - 1 cut into the segments' masks, and per segment the
    // blocks to scan (0: no token sees a cell of it); the same for every attention layer of the batch
    auto flash_prepare = [&](int key, int t0) {  // key: axera_q35_fd_key (R rows, or a group copy)
        axera_timer tm("fd prepare (cpu)");
        axera_q35::fd_kernels & fk   = m.fd.at(key);
        const int               R    = key % 1000;
        const int64_t           nseg = (W + m.S - 1) / m.S;
        const uint16_t          ninf = ggml_fp32_to_bf16(-3e38f).bits;
        std::vector<ggml_bf16_t> rows((size_t) R * W);
        std::vector<row_span>    span(R);
        {
            axera_timer tr("fd prepare: mask rows");
            axera_host_pool().run(R, [&](int64_t r) { mask_row(t0 + (int) r, rows.data() + (size_t) r * W, &span[r]); });
        }
        // does row r see a cell in [c0, c1)? Rows visible in one run (causal) answer from their span
        auto visible = [&](int r, int64_t c0, int64_t c1) {
            const row_span & sp = span[r];
            if (sp.lo < 0 || sp.hi < c0 || sp.lo >= c1) {
                return false;
            }
            if (sp.n == sp.hi - sp.lo + 1) {
                return true;
            }
            const auto * d = (const uint16_t *) rows.data() + (size_t) r * W;
            for (int64_t c = std::max(c0, sp.lo); c < std::min(c1, sp.hi + 1); ++c) {
                if (d[c] != ninf) {
                    return true;
                }
            }
            return false;
        };
        fk.nblk.assign(nseg, 0);
        fk.runs.assign(nseg * 2 * fk.cores, 0);
        fk.parts.clear();
        if ((int64_t) fk.MASK.size() < nseg) {
            fk.MASK.resize(nseg);
        }
        for (int64_t g = 0; g < nseg; ++g) {
            const int64_t c0 = g * m.S, c1 = std::min<int64_t>(W, c0 + m.S);
            int64_t       last = -1;  // the segment's last cell any row sees
            for (int r = 0; r < R; ++r) {
                const row_span & sp = span[r];
                if (sp.lo < 0 || sp.lo >= c1 || sp.hi < c0) {
                    continue;
                }
                const auto * d = (const uint16_t *) rows.data() + (size_t) r * W;
                int64_t      c = std::min(c1 - 1, sp.hi);
                while (c > c0 + last && d[c] == ninf) {
                    --c;
                }
                last = std::max(last, c - c0);
            }
            if (last < 0) {
                continue;
            }
            const int nb = (int) (last / fk.bc + 1);
            fk.nblk[g]   = nb;
            // the cores' runs of blocks; a core without blocks scans block 0 and its result is ignored
            static const int dbg_core = [] {  // GGML_AXERA_SEG_CORE=c: debugging, every block on core c
                const char * e = std::getenv("GGML_AXERA_SEG_CORE");
                return e ? std::atoi(e) : -1;
            }();
            const int per = dbg_core >= 0 ? nb : (nb + fk.cores - 1) / fk.cores;
            for (int c = 0; c < fk.cores; ++c) {
                const int b0 = dbg_core >= 0 ? (c == dbg_core ? 0 : nb) : std::min(nb, c * per);
                const int b1 = dbg_core >= 0 ? (c == dbg_core ? nb : nb) : std::min(nb, (c + 1) * per);
                int32_t * run = fk.runs.data() + (g * fk.cores + c) * 2;
                run[0]        = b1 > b0 ? b0 : 0;
                run[1]        = b1 > b0 ? b1 : 1;
                if (b1 <= b0) {
                    continue;
                }
                axera_q35::fd_kernels::part pt{ g, c, std::vector<uint8_t>(R, 0) };
                for (int r = 0; r < R; ++r) {
                    pt.seen[r] = visible(r, c0 + (int64_t) b0 * fk.bc, std::min<int64_t>(c1, c0 + (int64_t) b1 * fk.bc));
                }
                if (std::find(pt.seen.begin(), pt.seen.end(), 1) != pt.seen.end()) {
                    fk.parts.push_back(std::move(pt));
                }
            }
            if (!fk.MASK[g]) {
                fk.MASK[g] = axera_cmm_alloc(axera_input_size(fk.seg, "mask"));
            }
            axera_timer tc("fd prepare: segment masks");
            auto *      mk = (uint16_t *) fk.MASK[g]->vir;
            axera_host_pool().run(nb, [&](int64_t blk) {
                const int64_t col = c0 + blk * fk.bc, n = std::max<int64_t>(0, std::min<int64_t>(fk.bc, c1 - col));
                for (int r = 0; r < R; ++r) {
                    uint16_t * dst = mk + (blk * R + r) * fk.bc;
                    memcpy(dst, (const uint16_t *) rows.data() + (size_t) r * W + col, n * sizeof(uint16_t));
                    std::fill(dst + n, dst + fk.bc, ninf);
                }
            });
            fk.MASK[g]->flush(mk, (size_t) nb * R * fk.bc * sizeof(ggml_bf16_t));
        }
    };
    // the new K / V rows of tokens t0 .. t0 + R - 1 (head kernel outputs) to their cells of the llama.cpp cache
    auto write_kv = [&](axera_q35_layer & ly, axera_q35::fd_kernels & fk, int R, int t0) {
        axera_timer   tm("fd kv write (cpu)");
        const int64_t st = stream_of(t0);
        fk.K->invalidate(fk.K->vir, fk.K->size);
        fk.V->invalidate(fk.V->vir, fk.V->size);
        for (auto & kr : ly.kv) {
            const char * src = (const char *) (kr.v ? fk.V : fk.K)->vir;
            char *       run = nullptr;  // flush runs of consecutive cells at once
            size_t       len = 0;
            for (int r = 0; r < R; ++r) {
                char * dst = kr.base + st * kr.stride + cell_of(t0 + r) * kr.row;
                memcpy(dst, src + r * kr.row, kr.row);
                if (run && run + len == dst) {
                    len += kr.row;
                    continue;
                }
                if (run) {
                    kr.b->flush(run, len);
                }
                run = dst;
                len = kr.row;
            }
            kr.b->flush(run, len);
        }
    };
    // split attention layer: head kernel, the new K / V rows to their cells, one segment kernel call per segment in
    // use, merged on the CPU (m = max m_i, l = sum e^(m_i - m) l_i, o = sum e^(m_i - m) o_i / l), tail kernel
    auto flash_attn = [&](int il, int v, int t0, int64_t p) {
        axera_q35_layer &        ly = m.layers[il];
        const int                R  = v < 0 ? 1 : m.pre[v].P;
        axera_q35::fd_kernels &  fk = m.fd.at(axera_q35_fd_key(m, v));
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
        {
            axera_timer tm("fd head");
            run(fk.head, io.hin, io.hout, "attention head");
        }
        const int64_t st = stream_of(t0);
        write_kv(ly, fk, R, t0);
        const auto & kk  = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
        const auto & vv  = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
        const int    nr  = fk.hkv * fk.gr;  // rows: KV head hk, row gq * R + token
        std::vector<float>   am(nr, 0.0f), al(nr, 0.0f), ao((size_t) nr * fk.d, 0.0f);
        std::vector<uint8_t> started(nr, 0);
        size_t               pi = 0;  // fk.parts in segment order
        for (size_t g = 0; g < fk.nblk.size(); ++g) {
            if (fk.nblk[g] == 0) {
                continue;
            }
            io.sin[io.kc]   = axera_io(kk.b, kk.base + st * kk.stride + g * m.S * kk.row, (uint32_t) kk.size);
            io.sin[io.vc]   = axera_io(vv.b, vv.base + st * vv.stride + g * m.S * vv.row, (uint32_t) vv.size);
            io.sin[io.mask] = axera_io(fk.MASK[g].get(), fk.MASK[g]->vir, (uint32_t) axera_input_size(fk.seg, "mask"));
            if (fk.cores > 1) {
                memcpy(fk.BLK->vir, fk.runs.data() + g * fk.cores * 2, fk.cores * 2 * sizeof(int32_t));
            } else {
                *(int32_t *) fk.BLK->vir = fk.nblk[g];
            }
            fk.BLK->flush(fk.BLK->vir, fk.BLK->size);
            {
                axera_timer tm(R == 1 ? "fd seg (decode)" : "fd seg (prefill)");
                run(fk.seg, io.sin, io.sout, "attention segment");
            }
            axera_timer tm("fd merge (cpu)");
            fk.M->invalidate(fk.M->vir, fk.M->size);
            fk.Lsum->invalidate(fk.Lsum->vir, fk.Lsum->size);
            fk.O->invalidate(fk.O->vir, fk.O->size);
            for (; pi < fk.parts.size() && fk.parts[pi].g == (int64_t) g; ++pi) {
                const auto & pt = fk.parts[pi];
                const auto * mi = (const ggml_bf16_t *) fk.M->vir + (size_t) pt.c * nr;
                const auto * li = (const float *) fk.Lsum->vir + (size_t) pt.c * nr;
                const auto * oi = (const ggml_bf16_t *) fk.O->vir + (size_t) pt.c * nr * fk.d;
                axera_host_pool().run(nr, [&](int64_t j) {
                    if (!pt.seen[(j % fk.gr) % R]) {
                        return;
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
                        return;
                    }
                    const float mn = std::max(am[j], mj), a = std::exp(am[j] - mn), b = std::exp(mj - mn);
                    am[j] = mn;
                    al[j] = a * al[j] + b * li[j];
                    for (int i = 0; i < fk.d; ++i) {
                        o[i] = a * o[i] + b * ggml_bf16_to_fp32(oj[i]);
                    }
                });
            }
        }
        // attention output [R][HQ][D]: query head hk * G + gq of token t is row gq * R + t of KV head hk
        const int G   = fk.gr / R;
        auto *    att = (ggml_bf16_t *) fk.ATT->vir;
        axera_host_pool().run(fk.hkv * G, [&](int64_t hg) {
            const int hk = (int) hg / G, gq = (int) hg % G;
            {
                for (int tt = 0; tt < R; ++tt) {
                    const int     j   = hk * fk.gr + gq * R + tt;
                    const float   inv = al[j] > 0.0f ? 1.0f / al[j] : 0.0f;
                    ggml_bf16_t * dst = att + ((size_t) tt * fk.hkv * G + hk * G + gq) * fk.d;
                    for (int i = 0; i < fk.d; ++i) {
                        dst[i] = ggml_fp32_to_bf16(ao[(size_t) j * fk.d + i] * inv);
                    }
                }
            }
        });
        fk.ATT->flush(fk.ATT->vir, fk.ATT->size);
        {
            axera_timer tm("fd tail");
            run(fk.tail, io.tin, io.tout, "attention tail");
        }
    };
    // split attention for R rows: always with segments, else for wide masks when its kernels exist for the tier
    auto use_split = [&](int R) {
        return m.kv_llama && m.fd.count(R) && m.fd.at(R).head && (m.flash || W >= axera_split_min());
    };
    // a prefill chunk without a fused attention kernel (MoE P = 128): always split
    auto pre_split = [&](int v) { return use_split(m.pre[v].P) || (!m.pre[v].attn && !m.pre[v].mc); };
    // MoE layer il after its layer kernel: the top-K experts of the router logits (softmax, top-K, renormalized: as
    // transformers' Qwen3_5MoeTopKRouter; the softmax over the top-K alone gives the same weights), their weights
    // bound to the experts kernel, which adds them to the residual + shared expert (BASE) into the layer output
    std::vector<int> e_ids(m.n_expert);
    auto run_experts = [&](int il, int64_t p) {
        axera_q35_layer & ly = m.layers[il];
        axera_timer       tr("moe route (cpu)");
        m.LOGITS->invalidate(m.LOGITS->vir, m.LOGITS->size);
        const float * lg = (const float *) m.LOGITS->vir;
        std::iota(e_ids.begin(), e_ids.end(), 0);
        std::partial_sort(e_ids.begin(), e_ids.begin() + m.top_k, e_ids.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        auto * rw  = (float *) m.RW->vir;
        float  sum = 0.0f;
        for (int i = 0; i < m.top_k; ++i) {
            rw[i] = std::exp(lg[e_ids[i]] - lg[e_ids[0]]);
            sum += rw[i];
        }
        for (int i = 0; i < m.top_k; ++i) {
            rw[i] /= sum;
        }
        m.RW->flush(rw, m.top_k * sizeof(float));
        auto & ek = m.ex[ly.exps_gu];
        for (int i = 0; i < m.top_k; ++i) {
            for (int g = 0; g < 3; ++g) {
                if (ly.exps_gu && g == 1) {
                    continue;
                }
                const int    kind = ly.exps_gu && g == 0 ? 3 : g;
                const auto & e    = ly.exps[g];
                char *       w    = e.base + (size_t) e_ids[i] * e.stride;
                ek.in[ek.w[kind][i]] = axera_io(e.b, w, (uint32_t) e.data);
                ek.in[ek.s[kind][i]] = axera_io(e.b, w + e.data, (uint32_t) e.scale);
            }
        }
        auto & out = ek.out[il % 2];
        AX_ENGINE_IO_T io;
        memset(&io, 0, sizeof(io));
        io.pInputs     = ek.in.data();
        io.nInputSize  = (uint32_t) ek.in.size();
        io.pOutputs    = out.data();
        io.nOutputSize = (uint32_t) out.size();
        tr.stop();
        axera_timer tm("moe experts");
        const int ret = AX_ENGINE_RunSync(ek.k->handle, &io);
        if (ret != 0) {
            GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the experts of layer %d, position %" PRId64 "\n", ret,
                       il, p);
        }
        if (ek.PART) {  // multi-core: the layer output = the sum of the cores' rows
            const int cores = (int) (axera_output_size(ek.k, "part_out") / (H * sizeof(float)));
            ek.PART->invalidate(ek.PART->vir, ek.PART->size);
            const auto *       pr = (const float *) ek.PART->vir;
            std::vector<float> acc(pr, pr + H);
            for (int c = 1; c < cores; ++c) {
                for (int i = 0; i < H; ++i) {
                    acc[i] += pr[(size_t) c * H + i];
                }
            }
            auto * x = (il % 2 == 0 ? m.B : m.A).get();
            ggml_fp32_to_bf16_row(acc.data(), (ggml_bf16_t *) x->vir, H);
            x->flush(x->vir, H * sizeof(ggml_bf16_t));
        }
    };

    // MoE layer il of prefill chunks vs (a group, or one chunk) after their layer kernels: each token's top-K experts
    // (as run_experts), the picks grouped by expert into slots of up to G rows (h2 rows and routing weights gathered
    // here), S slots per call of the slots kernel, then per token base + its K results into the next layer's input
    auto run_experts_pre = [&](int il, const std::vector<int> & vs) {
        axera_q35_layer & ly = m.layers[il];
        auto &            sk = m.slots;
        const int         P = m.pre[vs[0]].P, NT = P * (int) vs.size(), K = m.top_k, NE = m.n_expert, S = sk.S, G = sk.G;
        GGML_ASSERT(ly.exps_gu);
        axera_timer tr("moe prefill route (cpu)");
        for (int v : vs) {
            const auto & pf = m.pre[v];
            pf.LOG->invalidate(pf.LOG->vir, pf.LOG->size);
            pf.H2->invalidate(pf.H2->vir, pf.H2->size);
            pf.BASE->invalidate(pf.BASE->vir, pf.BASE->size);
        }
        auto row_ptr = [&](const axera_cmm axera_q35::prefill::*buf, int t, size_t width) {  // token t of the group
            return (const char *) (m.pre[vs[t / P]].*buf)->vir + (size_t) (t % P) * width;
        };
        std::vector<int>   ids((size_t) NT * K);
        std::vector<float> wts((size_t) NT * K);
        axera_host_pool().run(NT, [&](int64_t r) {
            const float *    lg = (const float *) row_ptr(&axera_q35::prefill::LOG, (int) r, NE * sizeof(float));
            std::vector<int> o(NE);
            std::iota(o.begin(), o.end(), 0);
            std::partial_sort(o.begin(), o.begin() + K, o.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                ids[r * K + i] = o[i];
                wts[r * K + i] = std::exp(lg[o[i]] - lg[o[0]]);
                sum += wts[r * K + i];
            }
            for (int i = 0; i < K; ++i) {
                wts[r * K + i] /= sum;
            }
        });
        std::vector<std::vector<int>> by(NE);  // picks (token * K + i) per expert
        for (int pk = 0; pk < NT * K; ++pk) {
            by[ids[pk]].push_back(pk);
        }
        std::vector<int> se, sp;  // slot expert, slot rows' picks (-1: unused)
        for (int e = 0; e < NE; ++e) {
            for (size_t j = 0; j < by[e].size(); j += G) {
                se.push_back(e);
                for (int g = 0; g < G; ++g) {
                    sp.push_back(j + g < by[e].size() ? by[e][j + g] : -1);
                }
            }
        }
        const int calls = (int) (se.size() + S - 1) / S;
        GGML_ASSERT((size_t) calls * S * G * H * sizeof(ggml_bf16_t) <= m.SX->size);
        se.resize((size_t) calls * S, se[0]);  // padding slots: some expert on unused rows
        sp.resize((size_t) calls * S * G, -1);
        // a slot's picks fill its first rows; the other rows keep stale data (weight 0, results never read), so only
        // the used rows are written and kept coherent
        std::vector<int> loc((size_t) NT * K);  // pick -> slot row
        std::vector<int> used((size_t) calls * S, 0);
        auto *           sx = (ggml_bf16_t *) m.SX->vir;
        auto *           sw = (float *) m.SRW->vir;
        axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
            for (int g = 0; g < G; ++g) {
                const int64_t row = sl * G + g;
                const int     pk  = sp[row];
                sw[row]           = pk >= 0 ? wts[pk] : 0.0f;
                if (pk >= 0) {
                    memcpy(sx + row * H, row_ptr(&axera_q35::prefill::H2, pk / K, H * sizeof(ggml_bf16_t)),
                           H * sizeof(ggml_bf16_t));
                    loc[pk]  = (int) row;
                    used[sl] = g + 1;
                }
            }
            if (used[sl]) {
                m.SX->flush(sx + sl * G * H, (size_t) used[sl] * H * sizeof(ggml_bf16_t));
            }
        });
        m.SRW->flush(m.SRW->vir, (size_t) calls * S * G * sizeof(float));
        tr.stop();
        const auto & ec = ly.exps[0];  // gate-up
        const auto & ed = ly.exps[2];  // down
        for (int c = 0; c < calls; ++c) {
            const size_t r0 = (size_t) c * S * G;
            sk.in[sk.x]      = axera_io(m.SX.get(), sx + r0 * H, (uint32_t) (S * G * H * sizeof(ggml_bf16_t)));
            sk.in[sk.rw]     = axera_io(m.SRW.get(), sw + r0, (uint32_t) (S * G * sizeof(float)));
            sk.outs[sk.out]  = axera_io(m.SOUT.get(), (float *) m.SOUT->vir + r0 * H, (uint32_t) (S * G * H * sizeof(float)));
            for (int i = 0; i < S; ++i) {
                const int e = se[(size_t) c * S + i];
                char *    wc = ec.base + (size_t) e * ec.stride;
                char *    wd = ed.base + (size_t) e * ed.stride;
                sk.in[sk.w[0][i]] = axera_io(ec.b, wc, (uint32_t) ec.data);
                sk.in[sk.s[0][i]] = axera_io(ec.b, wc + ec.data, (uint32_t) ec.scale);
                sk.in[sk.w[1][i]] = axera_io(ed.b, wd, (uint32_t) ed.data);
                sk.in[sk.s[1][i]] = axera_io(ed.b, wd + ed.data, (uint32_t) ed.scale);
            }
            AX_ENGINE_IO_T io;
            memset(&io, 0, sizeof(io));
            io.pInputs     = sk.in.data();
            io.nInputSize  = (uint32_t) sk.in.size();
            io.pOutputs    = sk.outs.data();
            io.nOutputSize = (uint32_t) sk.outs.size();
            axera_timer tm("moe prefill slots");
            const int   ret = AX_ENGINE_RunSync(sk.k->handle, &io);
            if (ret != 0) {
                GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the prefill experts of layer %d\n", ret, il);
            }
        }
        axera_timer tc("moe prefill combine (cpu)");
        axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
            if (used[sl]) {
                m.SOUT->invalidate((float *) m.SOUT->vir + sl * G * H, (size_t) used[sl] * H * sizeof(float));
            }
        });
        const auto * so = (const float *) m.SOUT->vir;
        axera_host_pool().run(NT, [&](int64_t r) {
            const auto *       base = (const ggml_bf16_t *) row_ptr(&axera_q35::prefill::BASE, (int) r, H * sizeof(ggml_bf16_t));
            std::vector<float> acc(H);
            for (int i = 0; i < H; ++i) {
                acc[i] = ggml_bf16_to_fp32(base[i]);
            }
            for (int k = 0; k < K; ++k) {
                const float * o = so + (size_t) loc[r * K + k] * H;
                for (int i = 0; i < H; ++i) {
                    acc[i] += o[i];
                }
            }
            const auto & pf = m.pre[vs[r / P]];
            auto *       xo = (il % 2 == 0 ? pf.B : pf.A).get();
            ggml_fp32_to_bf16_row(acc.data(), (ggml_bf16_t *) xo->vir + (r % P) * H, H);
        });
        for (int v : vs) {
            auto * xo = (il % 2 == 0 ? m.pre[v].B : m.pre[v].A).get();
            xo->flush(xo->vir, (size_t) P * H * sizeof(ggml_bf16_t));
        }
    };

    // layer il's kernel(s) for run kind v (decode: -1) on tokens t0 .. (position p), without the MoE experts
    auto run_layer = [&](int il, int v, int t0, int64_t p) {
        const bool        prefill = v >= 0;
        const bool        split   = prefill ? pre_split(v) : use_split(1);
        axera_q35_layer & ly      = m.layers[il];
        if (split && !ly.gdn) {
            flash_attn(il, v, t0, p);
            return;
        }
        auto * in  = prefill ? &ly.pre[v].in : &ly.in;
        auto * out = prefill ? &ly.pre[v].out : &ly.out;
        if (prefill && ly.si >= 0) {  // private ping-pong state: take the current buffers
            (*in)[ly.pre[v].si]  = ly.in[ly.si];
            (*in)[ly.pre[v].ci]  = ly.in[ly.ci];
            (*out)[ly.pre[v].so] = ly.out[ly.so];
            (*out)[ly.pre[v].co] = ly.out[ly.co];
        }
        ggml_axera_kernel * k = ly.gdn ? (prefill ? m.pre[v].gdn : m.gdn) : (prefill ? m.pre[v].attn : m.attn);
        if (prefill && ly.gdn && rs_now) {
            // verification chunk: the same call plus the state after each earlier token into snapshot plane j (the
            // state after the chunk's token P - 1 - j), as llama.cpp's own graph writes them for a partial rollback
            auto &    pr = ly.pre[v];
            const int P  = m.pre[v].P;
            pr.rin[pr.rsi]  = (*in)[pr.si];
            pr.rin[pr.rci]  = (*in)[pr.ci];
            pr.rout[pr.rso] = (*out)[pr.so];
            pr.rout[pr.rco] = (*out)[pr.co];
            for (auto & rg : ly.cache_state) {
                for (int j = 1; j < P; ++j) {
                    pr.rout[(rg.conv ? pr.rs_conv : pr.rs_state)[j]] =
                        axera_io(rg.b, rg.base + (j * rs_plane + ly.row) * rg.stride, (uint32_t) rg.size);
                }
            }
            in  = &pr.rin;
            out = &pr.rout;
            k   = m.pre[v].gdn_rs;
        }
        AX_ENGINE_IO_T io;
        memset(&io, 0, sizeof(io));
        io.pInputs     = in->data();
        io.nInputSize  = (uint32_t) in->size();
        io.pOutputs    = out->data();
        io.nOutputSize = (uint32_t) out->size();
        int ret;
        {
            axera_timer tm(ly.gdn ? (prefill ? "gdn prefill" : "gdn decode") : (prefill ? "attn prefill" : "attn decode"));
            ret = AX_ENGINE_RunSync(k->handle, &io);
        }
        if (ret != 0) {
            GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d, position %" PRId64 "%s\n", ret, il, p,
                       prefill ? " (prefill)" : "");
        }
        if (ly.si >= 0) {
            std::swap(ly.in[ly.si], ly.out[ly.so]);
            std::swap(ly.in[ly.ci], ly.out[ly.co]);
        }
    };
    auto run_layers = [&](int v, int t0, int64_t p) {  // v: prefill chunk size index, -1: decode
        for (int il = 0; il < L; ++il) {
            run_layer(il, v, t0, p);
            if (m.moe && v >= 0) {
                run_experts_pre(il, { v });
            } else if (m.moe) {
                run_experts(il, p);
            }
        }
    };
    // a MoE group: chunks vs of P tokens from t0 (position p), layer by layer, the experts once per layer
    auto run_group = [&](const std::vector<int> & vs, int t0, int64_t p) {
        const int P = m.pre[vs[0]].P;
        for (int il = 0; il < L; ++il) {
            for (size_t c = 0; c < vs.size(); ++c) {
                run_layer(il, vs[c], t0 + (int) c * P, p + (int64_t) c * P);
            }
            run_experts_pre(il, vs);
        }
    };

    // multi-core prefill chain of chunk v (P tokens from t0): per layer gdn | head, segments, tail; then mlp. The
    // residual leaves the last layer as base (pf.A) + the partials (PART0)
    auto run_mc = [&](int v, int t0) {
        axera_q35::prefill &    pf = m.pre[v];
        const int               P  = pf.P;
        axera_q35::mc_kernels & mk = m.mc.at(P);
        axera_q35::fd_kernels & fk = m.fd.at(P);
        // GGML_AXERA_MC_CHECK=1: debugging, report the first kernel output holding a NaN / Inf (bf16, f16 or f32 by name)
        // (2: only invalidate the outputs, 3: only wait 300 us after each call)
        static const int check_mode = [] {
            const char * e = std::getenv("GGML_AXERA_MC_CHECK");
            return e ? std::atoi(e) : 0;
        }();
        static const bool check = check_mode == 1;
        static bool reported = false;
        auto run = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                       const char * what, int il) {
            AX_ENGINE_IO_T io;
            memset(&io, 0, sizeof(io));
            io.pInputs     = in.data();
            io.nInputSize  = (uint32_t) in.size();
            io.pOutputs    = out.data();
            io.nOutputSize = (uint32_t) out.size();
            {
                axera_timer tm(what);
                const int   ret = AX_ENGINE_RunSync(k->handle, &io);
                if (ret != 0) {
                    GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d (%s)\n", ret, il, what);
                }
            }
            if (check_mode == 2) {
                for (auto & b : out) {
                    AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
                }
            } else if (check_mode == 3) {  // GGML_AXERA_MC_WAIT=<kind>: only after those calls ("mc gdn", ...)
                static const char * kind = std::getenv("GGML_AXERA_MC_WAIT");
                static const int    us   = std::getenv("GGML_AXERA_MC_WAIT_US") ? std::atoi(std::getenv("GGML_AXERA_MC_WAIT_US")) : 300;
                if (!kind || strstr(what, kind)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(us));
                }
            }
            for (uint32_t i = 0; check && !reported && i < out.size(); ++i) {
                const AX_ENGINE_IOMETA_T & om = k->info->pOutputs[i];
                AX_SYS_MinvalidateCache(out[i].phyAddr, out[i].pVirAddr, out[i].nSize);
                const std::string n  = om.pName;
                const int         es = om.eDataType == AX_ENGINE_DT_FLOAT32 ? 4 : 2;
                const bool        f16 = om.eDataType == AX_ENGINE_DT_FLOAT16 || n == "k_out" || n == "v_out";
                int64_t           bad = -1;
                for (int64_t e = 0; e < (int64_t) out[i].nSize / es && bad < 0; ++e) {
                    float v;
                    if (es == 4) {
                        v = ((const float *) out[i].pVirAddr)[e];
                    } else if (f16) {
                        v = ggml_fp16_to_fp32(((const ggml_fp16_t *) out[i].pVirAddr)[e]);
                    } else {
                        v = ggml_bf16_to_fp32(((const ggml_bf16_t *) out[i].pVirAddr)[e]);
                    }
                    bad = std::isfinite(v) ? -1 : e;
                }
                if (bad >= 0) {
                    reported = true;
                    GGML_LOG_ERROR("qwen35 template: mc check: %s layer %d output %s element %" PRId64 " of %u is not "
                                   "finite (chunk at token %d, cell %" PRId64 ", W %" PRId64 ")\n", what, il, n.c_str(), bad,
                                   out[i].nSize / es, t0, cell_of(t0), W);
                }
            }
        };
        const size_t nseg = fk.nblk.size();
        // the slots hold `groups` segments (one slot per segment core). More segments in use: group 0 keeps the CPU
        // merge of the segments done so far, the other groups take the next ones in turns
        static const size_t max_groups = [] {  // GGML_AXERA_MC_GROUPS=n: debugging, use n segment groups of slots only
            const char * e = std::getenv("GGML_AXERA_MC_GROUPS");
            return e ? (size_t) std::max(2, std::atoi(e)) : SIZE_MAX;
        }();
        const size_t groups = std::min(max_groups, (size_t) mk.np / fk.cores);
        size_t       nused  = 0;
        for (size_t g = 0; g < nseg; ++g) {
            nused += fk.nblk[g] != 0;
        }
        const bool spill = nused > groups;
        if (spill && groups < 2) {
            GGML_ABORT("qwen35 template: %zu KV segments x %d cores need more than the %d merge slots of %s\n", nused,
                       fk.cores, mk.np, "qwen35_attn_tail_mc");
        }
        // slots without data this batch: m = -3e38 (weight 0 in the merge) and l = o = 0, as 0 * a stale NaN is NaN
        const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;
        auto clear_slot = [&](size_t slot) {
            auto * p = (uint16_t *) ((char *) mk.MALL->vir + slot * mk.ms);
            std::fill(p, p + mk.ms / sizeof(uint16_t), ninf);
            mk.MALL->flush(p, mk.ms);
            char * l = (char *) mk.LALL->vir + slot * mk.ls;
            char * o = (char *) mk.OALL->vir + slot * mk.os;
            memset(l, 0, mk.ls);
            memset(o, 0, mk.os);
            mk.LALL->flush(l, mk.ls);
            mk.OALL->flush(o, mk.os);
        };
        for (int i = 0; i < mk.np; ++i) {
            clear_slot(i);
        }
        // fold every slot (and slot 0 itself if acc) into slot 0, clear the others. Rows: KV head, g * P + token;
        // m bf16, l f32, o bf16 unnormalized (sum e^(s - m) v). Slot 0 gets m rounded to bf16 with l / o rescaled to it
        auto merge_slots = [&](bool acc) {
            axera_timer tm("mc attn merge (cpu, > slots)");
            const int   nr = fk.hkv * fk.gr, d = fk.d;
            mk.MALL->invalidate(mk.MALL->vir, mk.MALL->size);
            mk.LALL->invalidate(mk.LALL->vir, mk.LALL->size);
            mk.OALL->invalidate(mk.OALL->vir, mk.OALL->size);
            auto * M = (ggml_bf16_t *) mk.MALL->vir;
            auto * Ls = (float *) mk.LALL->vir;
            auto * O = (ggml_bf16_t *) mk.OALL->vir;
            axera_host_pool().run(nr, [&](int64_t j) {
                float mx = -3e38f;
                for (int sl = acc ? 0 : 1; sl < mk.np; ++sl) {
                    mx = std::max(mx, ggml_bf16_to_fp32(M[(size_t) sl * nr + j]));
                }
                std::vector<float> o(d, 0.0f);
                float              l = 0.0f;
                for (int sl = acc ? 0 : 1; sl < mk.np && mx > -1e37f; ++sl) {
                    const float w = std::exp(ggml_bf16_to_fp32(M[(size_t) sl * nr + j]) - mx);
                    if (w == 0.0f) {
                        continue;
                    }
                    l += w * Ls[(size_t) sl * nr + j];
                    const ggml_bf16_t * oi = O + ((size_t) sl * nr + j) * d;
                    for (int i = 0; i < d; ++i) {
                        o[i] += w * ggml_bf16_to_fp32(oi[i]);
                    }
                }
                ggml_bf16_t * o0 = O + (size_t) j * d;
                if (mx <= -1e37f) {  // no cell seen yet: an empty slot
                    M[j]  = ggml_fp32_to_bf16(-3e38f);
                    Ls[j] = 0.0f;
                    std::fill(o0, o0 + d, ggml_fp32_to_bf16(0.0f));
                    return;
                }
                const ggml_bf16_t mb = ggml_fp32_to_bf16(mx);
                const float       sc = std::exp(mx - ggml_bf16_to_fp32(mb));
                M[j]                 = mb;
                Ls[j]                = l * sc;
                for (int i = 0; i < d; ++i) {
                    o0[i] = ggml_fp32_to_bf16(o[i] * sc);
                }
            });
            mk.MALL->flush(mk.MALL->vir, mk.ms);
            mk.LALL->flush(mk.LALL->vir, mk.ls);
            mk.OALL->flush(mk.OALL->vir, mk.os);
            for (int i = 1; i < mk.np; ++i) {
                clear_slot(i);
            }
        };
        const int64_t st = stream_of(t0);
        for (int il = 0; il < L; ++il) {
            axera_q35_layer &        ly = m.layers[il];
            axera_q35_layer::mc_io & io = ly.mc.at(P);
            if (ly.gdn) {
                io.ain[io.si] = io.aout[io.so] = ly.in[io.dsi];  // the current recurrent-cache row
                io.ain[io.ci] = io.aout[io.co] = ly.in[io.dci];
                run(mk.gdn, io.ain, io.aout, "mc gdn", il);
                run(mk.mlp, io.min, io.mout, "mc mlp", il);
                continue;
            }
            run(mk.head, io.ain, io.aout, "mc attn head", il);
            write_kv(ly, fk, P, t0);
            const auto & kk = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
            const auto & vv = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
            size_t       k  = 0;      // segments in the slots since the last merge
            bool         acc = false;  // slot 0 holds merged segments
            for (size_t g = 0; g < nseg; ++g) {
                if (fk.nblk[g] == 0) {
                    continue;
                }
                if (spill && 1 + k == groups) {
                    merge_slots(acc);
                    acc = true;
                    k   = 0;
                }
                const size_t s0 = (spill ? 1 + k++ : k++) * fk.cores;  // the segment's first slot
                io.sin[io.kc]   = axera_io(kk.b, kk.base + st * kk.stride + g * m.S * kk.row, (uint32_t) kk.size);
                io.sin[io.vc]   = axera_io(vv.b, vv.base + st * vv.stride + g * m.S * vv.row, (uint32_t) vv.size);
                io.sin[io.mask] = axera_io(fk.MASK[g].get(), fk.MASK[g]->vir, (uint32_t) axera_input_size(fk.seg, "mask"));
                io.sout[io.mo]  = axera_io(mk.MALL.get(), (char *) mk.MALL->vir + s0 * mk.ms, (uint32_t) (fk.cores * mk.ms));
                io.sout[io.lo]  = axera_io(mk.LALL.get(), (char *) mk.LALL->vir + s0 * mk.ls, (uint32_t) (fk.cores * mk.ls));
                io.sout[io.oo]  = axera_io(mk.OALL.get(), (char *) mk.OALL->vir + s0 * mk.os, (uint32_t) (fk.cores * mk.os));
                if (fk.cores > 1) {
                    memcpy(fk.BLK->vir, fk.runs.data() + g * fk.cores * 2, fk.cores * 2 * sizeof(int32_t));
                } else {
                    *(int32_t *) fk.BLK->vir = fk.nblk[g];
                }
                fk.BLK->flush(fk.BLK->vir, fk.BLK->size);
                run(fk.seg, io.sin, io.sout, "mc attn seg", il);
                for (int c = 0; c < fk.cores; ++c) {  // a core given no block scanned block 0: no data
                    const int32_t * r = fk.runs.data() + (g * fk.cores + c) * 2;
                    bool            used = false;
                    for (const auto & pt : fk.parts) {
                        used = used || (pt.g == (int64_t) g && pt.c == c);
                    }
                    if (fk.cores > 1 && !used) {
                        clear_slot(s0 + c);
                    }
                    GGML_UNUSED(r);
                }
            }
            run(mk.tail, io.tin, io.tout, "mc attn tail", il);
            if (spill) {  // the next attention layer starts from empty slots
                for (int i = 0; i < mk.np; ++i) {
                    clear_slot(i);
                }
            }
            run(mk.mlp, io.min, io.mout, "mc mlp", il);
        }
    };
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
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
        auto chunk_fits = [&](size_t c) {
            const int P  = m.pre[c].P;
            bool      ok = !m.pre[c].batch && !m.pre[c].sub && it + P <= T && (m.kv_llama || p + P <= m.LMAX) &&
                      it % n_seq_tokens + P <= n_seq_tokens;
            // on the llama.cpp KV cache the cells must be consecutive, not the positions (M-RoPE: an image's tokens)
            for (int j = 1; ok && j < P; ++j) {
                ok = m.kv_llama ? stream_of(it + j) == stream_of(it) && cell_of(it + j) == cell_of(it) + j :
                                  pos[it + j] == p + j;
            }
            return ok && (!m.kv_llama || cell_of(it) + P <= m.N);
        };
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            v = m.pre[c].P > 1 && chunk_fits(c) ? (int) c : -1;
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
        // else one token on the multi-core chain (P = 1)
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            v = m.pre[c].P == 1 && chunk_fits(c) ? (int) c : -1;
        }
        // a MoE group: as many of its chunks as fit, consecutive
        int ng = 1;
        while (v >= 0 && ng < (int) m.pre[v].group.size()) {
            const int P = m.pre[v].P, n = ng + 1;
            bool      ok = it + n * P <= T && it % n_seq_tokens + n * P <= n_seq_tokens &&
                      (m.kv_llama ? cell_of(it) + n * P <= m.N : p + n * P <= m.LMAX);
            for (int j = ng * P; ok && j < n * P; ++j) {
                ok = m.kv_llama ? stream_of(it + j) == stream_of(it) && cell_of(it + j) == cell_of(it) + j :
                                  pos[it + j] == p + j;
            }
            if (!ok) {
                break;
            }
            ng = n;
        }
        if (ng > 1) {
            const int        P = m.pre[v].P;
            std::vector<int> vs(m.pre[v].group.begin(), m.pre[v].group.begin() + ng);
            for (int c = 0; c < ng; ++c) {
                axera_q35::prefill & pc = m.pre[vs[c]];
                const int            tc = it + c * P;
                for (int r = 0; r < P; ++r) {
                    ggml_fp32_to_bf16_row(x0_row(tc + r),
                                          (ggml_bf16_t *) pc.A->vir + (size_t) r * H, H);
                }
                pc.A->flush(pc.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                rope_rows(tc, P, pc.COS.get(), pc.SIN.get());
                flash_prepare(axera_q35_fd_key(m, vs[c]), tc);
            }
            run_group(vs, it, p);
            for (int c = 0; c < ng; ++c) {
                auto * fin = (L % 2 == 0 ? m.pre[vs[c]].A : m.pre[vs[c]].B).get();
                fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) (it + c * P) * H, (int64_t) P * H);
            }
            m.next_pos = p + (int64_t) ng * P;
            it += ng * P;
            continue;
        }
        if (v >= 0) {
            axera_q35::prefill & pf    = m.pre[v];
            const int            P     = pf.P;
            const bool           split = !pf.mc && pre_split(v);
            if (pf.batch) {
                axera_q35_set_batch_rows(m, v, row_of(it));
                for (int r = 1; r < P; ++r) {  // token it's row was cleared by begin()
                    if (pos[it + r] == 0) {
                        axera_q35_zero_row(m, row_of(it + r));
                    }
                }
                for (int r = 0; r < P && !split; ++r) {
                    ((int32_t *) pf.POS->vir)[r] = (int32_t) cell_of(it + r);
                }
                if (!split) {
                    pf.POS->flush(pf.POS->vir, pf.POS->size);
                }
            }
            for (int r = 0; r < P; ++r) {
                ggml_fp32_to_bf16_row(x0_row(it + r),
                                      (ggml_bf16_t *) pf.A->vir + (size_t) r * H, H);
            }
            pf.A->flush(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rope_rows(it, P, pf.COS.get(), pf.SIN.get());
            if (pf.mc) {
                flash_prepare(P, it);
                run_mc(v, it);
                auto * mk = &m.mc.at(P);
                pf.A->invalidate(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                mk->PART0->invalidate(mk->PART0->vir, mk->PART0->size);
                const int    C    = (int) (mk->PART0->size / ((size_t) P * H * sizeof(float)));
                const auto * base = (const ggml_bf16_t *) pf.A->vir;
                const auto * part = (const float *) mk->PART0->vir;
                axera_host_pool().run(P, [&](int64_t r) {
                    for (int i = 0; i < H; ++i) {
                        float x = ggml_bf16_to_fp32(base[r * H + i]);
                        for (int c = 0; c < C; ++c) {
                            x += part[((size_t) c * P + r) * H + i];
                        }
                        h[(size_t) (it + r) * H + i] = x;
                    }
                });
                m.next_pos = p + P;
                it += P;
                continue;
            }
            if (split) {
                flash_prepare(axera_q35_fd_key(m, v), it);
            }
            int nblk = (int) ((p + P - 1) / m.BC + 1);
            // chunk mask [N / BC][R][BC], row r = token r % P (of head r / P in older kernels): positions up to
            // p + token, or with the llama.cpp KV cache that token's KQ mask row over the cells
            const int64_t R    = pf.R;
            auto *        mask = pf.MASK ? (ggml_bf16_t *) pf.MASK->vir : nullptr;  // none with split attention
            const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f), ninf = ggml_fp32_to_bf16(-3e38f);
            if (split) {
                // the segments' masks: flash_prepare
            } else if (m.kv_llama) {
                axera_timer tm("mask (cpu)");
                std::vector<ggml_bf16_t> rows((size_t) P * W);
                std::vector<int>         nb(P, 1);
                axera_host_pool().run(P, [&](int64_t r) { nb[r] = mask_row(it + (int) r, rows.data() + (size_t) r * W); });
                nblk = *std::max_element(nb.begin(), nb.end());
                axera_host_pool().run(nblk, [&](int64_t blk) {
                    for (int64_t r = 0; r < R; ++r) {
                        memcpy(mask + (blk * R + r) * m.BC, rows.data() + (size_t) (r % P) * W + blk * m.BC,
                               m.BC * sizeof(ggml_bf16_t));
                    }
                });
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
            if (!split) {
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
            // speculative decoding: a verification batch (all T <= snapshot count tokens in this chunk) also writes the
            // recurrent-state snapshots; without the snapshot kernel a rollback would restore wrong states
            rs_now = rs_written > 1 && it == 0 && P == T && rs_written == T && pf.gdn_rs != nullptr;
            if (rs_written > 1 && rs_written == T && !rs_now) {
                GGML_LOG_ERROR("qwen35 template: a %d-token batch needs recurrent-state snapshots (speculative decoding, "
                               "n_rs_seq); no qwen35_gdn_prefill%d*_rs kernel runs it in one chunk\n", T, T);
                return GGML_STATUS_FAILED;
            }
            run_layers(v, it, p);
            rs_now = false;
            auto * fin = (L % 2 == 0 ? pf.A : pf.B).get();
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, (int64_t) P * H);
            m.next_pos = p + P;
            it += P;
            continue;
        }
        ggml_fp32_to_bf16_row(x0_row(it), (ggml_bf16_t *) m.A->vir, H);
        m.A->flush(m.A->vir, H * sizeof(ggml_bf16_t));
        rope_rows(it, 1, m.COS.get(), m.SIN.get());
        if (use_split(1)) {
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
        {
            axera_timer tly("layers (decode)");
            run_layers(-1, it, p);
        }
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
    // h_nextn (llama.cpp's MTP input, and in the MTP graph its output for the next draft step): the normed hidden
    // state of every token (the graph's T rows) or of the output rows
    ggml_tensor * hx = find("h_nextn");
    if (hx && hx->type == GGML_TYPE_F32 && hx->ne[0] == H && ggml_backend_buffer_is_axera(hx->buffer) &&
        (hx->ne[1] == n_out || hx->ne[1] == T)) {
        if (hx->ne[1] == n_out) {
            for (int r = 0; r < n_out; ++r) {
                memcpy((char *) hx->data + r * hx->nb[1], hn.data() + (size_t) r * H, H * sizeof(float));
            }
        } else {
            axera_host_pool().run(T, [&](int64_t r) {
                const float * x  = h.data() + (size_t) r * H;
                float *       y  = (float *) ((char *) hx->data + r * hx->nb[1]);
                double        ss = 0;
                for (int i = 0; i < H; ++i) {
                    ss += (double) x[i] * x[i];
                }
                const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
                for (int i = 0; i < H; ++i) {
                    y[i] = x[i] * scale * g[i];
                }
            });
        }
        ((ggml_backend_axera_buffer_context *) hx->buffer->context)->flush(hx->data, ggml_nbytes(hx));
    }
    // several rows: the batched GEMV of the weight format reads the output projection once per call (MXFP8, the bf16
    // GGUF tensor in CMM or its bf16 copy)
    const ggml_backend_axera_buffer_context * lm_ctx =
        m.lm_q ? m.lm_q.get() : m.lm_bf ? m.lm_bf.get() :
        lm->type == GGML_TYPE_BF16 && ggml_backend_buffer_is_axera(lm->buffer) ? (ggml_backend_axera_buffer_context *) lm->buffer->context :
                                                                                  nullptr;
    void * lm_data = m.lm_q ? m.lm_q->vir : m.lm_bf ? m.lm_bf->vir : lm->data;
    auto   dst_of  = [&](int r) { return (float *) ((char *) logits->data + r * logits->nb[1]); };
    axera_timer tl("lm_head");
    for (int r = 0; r < n_out;) {
        int done = 0;
        for (int Bn : { 8, 4, 2 }) {
            if (done == 0 && lm_ctx && r + Bn <= n_out) {
                float * dsts[8];
                for (int b = 0; b < Bn; ++b) {
                    dsts[b] = dst_of(r + b);
                }
                done = axera_gemv_batch(ctx, lm_ctx, lm_data, m.lm_q ? m.lm_s.get() : nullptr, lm->ne[1], H,
                                        hn.data() + (size_t) r * H, Bn, dsts) ? Bn : 0;
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

//
// Qwen3.5 vision encoder
//

// a graph is the vision encoder if it reads the first ViT layer's qkv weight
static bool axera_vit_graph(const struct ggml_cgraph * cgraph) {
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
static enum ggml_status axera_vit_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
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
    axera_vit &   v   = ctx->vit;
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
        static std::chrono::steady_clock::time_point last;  // profiling: host time between graphs
        if (g_axera_prof.on && last.time_since_epoch().count()) {
            auto & e = g_axera_prof.t["between graphs (host)"];
            e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last).count();
            e.second++;
        }
        enum ggml_status st;
        {
            axera_timer tm("graph_compute (all)");
            st = axera_vit_graph(cgraph) ? axera_vit_compute(ctx, cgraph) : axera_q35_compute(ctx, cgraph);
        }
        last = std::chrono::steady_clock::now();
        return st;
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
    // dense or MoE attention kernels (a kernel directory holds one model's; no graph yet to tell)
    const std::string kv  = axera_weights_mx8() ? "_attn_mx8_kvc" : "_attn_kvc";
    int64_t           ctx = axera_kernels().ctx_for("qwen35" + kv, t->ne[1]);
    ctx                   = ctx > 0 ? ctx : axera_kernels().ctx_for("qwen35moe" + kv, t->ne[1]);
    const int64_t S       = axera_flash_seg();  // split attention: whole segments
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
