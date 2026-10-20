// Axera NPU backend: internal declarations shared by the ggml-axera sources (model independent). The graph
// templates build on them: ggml-axera-attn.h (split attention on the llama.cpp KV cache, RoPE), ggml-axera-q35.h
// (Qwen3.5), the vision encoder in ggml-axera-vit.cpp
#pragma once

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
// GGML_AXERA_TEMPLATE=qwen35 runs whole Qwen3.5 graphs with per-layer kernels instead (ggml-axera-q35.h).
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

inline constexpr size_t AXERA_ALIGNMENT = 128;

struct axera_prof;
class axera_pool;
struct ggml_axera_kernel;
struct ggml_axera_kernel_registry;
struct ggml_backend_axera_buffer_context;
using axera_cmm = std::unique_ptr<ggml_backend_axera_buffer_context>;
struct ggml_backend_axera_context;
struct axera_q35;
struct axera_vit;

// functions of the backend sources (ggml-axera-*.cpp)
bool axera_debug();
axera_pool & axera_host_pool();
bool axera_runtime_init();
void axera_cmm_memory(size_t * free, size_t * total);
ggml_axera_kernel_registry & axera_kernels();
std::string axera_mul_mat_kernel_name(int64_t c, int64_t k, bool mx8 = false);
bool ggml_backend_buffer_is_axera(ggml_backend_buffer_t buffer);
std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc(size_t size);
std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc_nc(size_t size);
AX_ENGINE_IO_BUFFER_T axera_io(const ggml_backend_axera_buffer_context * ctx, void * p, uint32_t size);
bool axera_template_qwen35();
int64_t axera_split_min();
int axera_cores();
std::string axera_multi_core(const std::string & name);
bool axera_flash_forced();
int64_t axera_flash_seg();
bool axera_weights_mx8();
void axera_quantize_mxfp8(const ggml_tensor * w, axera_cmm & q, axera_cmm & s);
axera_cmm axera_to_bf16(const ggml_tensor * w);
void axera_gemv_raw(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                    void * wdata, int64_t N, int64_t K, const float * x, ggml_tensor * dst, float * out);
void axera_gemv_raw_to(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                       void * wdata, int64_t N, int64_t K, const float * x,
                       const ggml_backend_axera_buffer_context * dctx, float * out);
bool axera_gemv_batch(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx, void * wdata,
                      const ggml_backend_axera_buffer_context * sctx, int64_t N, int64_t K, const float * x,
                      int Bn, float * const * out);
void axera_gemv(ggml_backend_axera_context * ctx, const ggml_tensor * w, const float * x, ggml_tensor * dst,
                float * out);
void axera_gemv_mx8(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                    const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                    ggml_tensor * dst, float * out);
void ggml_backend_axera_mul_mat(ggml_backend_axera_context * ctx, ggml_tensor * dst);
float axera_rms_eps(const ggml_tensor * rms_norm);
void axera_mxfp4_rows(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dq, uint8_t * ds);
void axera_check_mxfp4(const ggml_tensor * w);
void axera_repack_mxfp4(const ggml_tensor * w);
bool axera_mxfp4_gu_adjacent(const ggml_tensor * g, const ggml_tensor * u);
void axera_repack_mxfp4_gu(const ggml_tensor * g, const ggml_tensor * u, uint8_t * dst,
                           const ggml_backend_axera_buffer_context * db);
uint32_t axera_input_size(ggml_axera_kernel * k, const char * name);
uint32_t axera_output_size(ggml_axera_kernel * k, const char * name);
std::string axera_base_name(const char * name);
int axera_run_sync(ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                   std::vector<AX_ENGINE_IO_BUFFER_T> & out);
// graph templates: the graph's name -> tensor map (cached in the context), the find by name, the output head
const std::map<std::string, ggml_tensor *> & axera_graph_names(ggml_backend_axera_context * ctx,
                                                               struct ggml_cgraph * cgraph);
ggml_tensor * axera_graph_find(const std::map<std::string, ggml_tensor *> & t, const char * name);
void axera_output_head(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph,
                       const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & h, int T,
                       const ggml_backend_axera_buffer_context * lm_q, const ggml_backend_axera_buffer_context * lm_s,
                       const ggml_backend_axera_buffer_context * lm_bf);
enum ggml_status axera_q35_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);
bool axera_vit_graph(const struct ggml_cgraph * cgraph);
enum ggml_status axera_vit_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);

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
inline axera_prof g_axera_prof;

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


struct ggml_backend_axera_context {
    axera_cmm staging;    // bf16 activation row
    axera_cmm batch_out;  // f32 output rows of a batched GEMV call
    // the graph templates' state (created by their first graph)
    std::shared_ptr<axera_q35> q35;
    std::shared_ptr<axera_vit> vit;
    // the graph templates' name -> tensor map of the last graph, kept while the graph keeps its nodes (llama.cpp
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

// runs fn(row, float row) for every row of a 2-D weight of any GGUF type (f32, f16, bf16, quantized), on all cores
template <typename F> void axera_for_rows_f32(const ggml_tensor * w, F fn) {
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
