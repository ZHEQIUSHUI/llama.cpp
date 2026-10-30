// Axera NPU backend core: the NPU runtime, CMM buffers, the host worker pool, profiling (model independent)
//
// - Device memory is CMM allocated cached (AX_SYS_MemAllocCached). It is host-visible, so the CPU backend works on
//   it directly for every op this backend does not take; the NPU addresses it physically. Cache maintenance: the CPU
//   side flushes after writing, the NPU side flushes its output range before a run (no dirty line may be evicted over
//   the result) and invalidates it afterwards.
// - GGML_AXERA_PROF=1: wall time per part, printed at exit. GGML_AXERA_TRACE=1: every graph's kind and time.
#pragma once

#include "ggml-impl.h"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

inline constexpr size_t AXERA_ALIGNMENT = 128;
// the largest single IO: the runtime's size field overflows at 2 GiB (packs are split well below it)
inline constexpr size_t AXERA_IO_MAX = (size_t) 2 << 30;

// the backend's buffers (CMM, or the mapped GGUF): ggml-axera-buffer.cpp
bool ggml_backend_buffer_is_axera(struct ggml_backend_buffer * buffer);

// AX_SYS / AX_ENGINE, once per process; false if the NPU is unavailable
bool axera_runtime_init();
// CMM total / free (/proc/ax_proc/mem_cmm_info)
void axera_cmm_memory(size_t * free, size_t * total);
// the largest sequence count of any llama.cpp context made in this process (0: none told the backend): a padded
// prefill chunk writes K / V into the cells after its batch's, which only one sequence leaves free
uint32_t axera_n_seq_max();

// a CMM range: a ggml buffer's (the model's tensors), or the backend's own (axera_cmm)
struct ggml_backend_axera_buffer_context {
    AX_U64 phy  = 0;
    void * vir  = nullptr;
    size_t size = 0;
    bool   nc   = false;  // non-cacheable: no cache maintenance (host writes only)
    bool   view = false;  // a range of another buffer (axera_cmm_view): not freed
    // a range the NPU wrote and the host invalidated in the current graph (axera_graph_epoch): read back without cache
    // maintenance (the logits, 1 MB per token)
    mutable const void * clean_p     = nullptr;
    mutable size_t       clean_n     = 0;
    mutable uint64_t     clean_epoch = 0;

    ~ggml_backend_axera_buffer_context() {
        if (vir && !view) {
            if (!nc && size) {  // no dirty line of ours may reach the memory after its next owner (the engine) writes it
                AX_SYS_MflushCache(phy, vir, (AX_U32) size);
                AX_SYS_MinvalidateCache(phy, vir, (AX_U32) size);
            }
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

    void flush() const { flush(vir, size); }
    void invalidate() const { invalidate(vir, size); }
};

using axera_cmm = std::unique_ptr<ggml_backend_axera_buffer_context>;

axera_cmm axera_cmm_alloc(size_t size);     // cached, zeroed
axera_cmm axera_cmm_alloc_nc(size_t size);  // non-cacheable (host writes only)
// bytes off .. off + size of b as a buffer of its own (b outlives it)
axera_cmm axera_cmm_view(const ggml_backend_axera_buffer_context * b, size_t off, size_t size);
// a kernel IO entry over p (within ctx)
AX_ENGINE_IO_BUFFER_T axera_io(const ggml_backend_axera_buffer_context * ctx, void * p, uint32_t size);
inline AX_ENGINE_IO_BUFFER_T axera_io(const axera_cmm & c, size_t size) {
    return axera_io(c.get(), c->vir, (uint32_t) size);
}
inline AX_ENGINE_IO_BUFFER_T axera_io(const axera_cmm & c) {
    return axera_io(c.get(), c->vir, (uint32_t) c->size);
}

extern std::atomic<uint64_t> axera_graph_epoch;  // graphs computed so far

// GGML_AXERA_PROF=1: wall time per part, printed at exit
struct axera_prof {
    const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_PROF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    std::map<std::string, std::pair<double, int64_t>> t;   // ms, calls
    std::mutex                                        mu;  // timers on other threads

    ~axera_prof() {
        for (const auto & [k, v] : t) {
            fprintf(stderr, "axera prof: %-34s %10.1f ms %8lld calls %8.3f ms/call\n", k.c_str(), v.first,
                    (long long) v.second, v.first / (double) std::max<int64_t>(v.second, 1));
        }
    }
};
inline axera_prof g_axera_prof;

struct axera_timer {
    const char *                          what;
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

// persistent workers for the host loops: run(n, fn) calls fn(0 .. n - 1) on the workers and the caller, returning
// when all are done
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

    std::vector<std::thread>             workers;
    std::mutex                           mu;
    std::condition_variable              cv, done;
    const std::function<void(int64_t)> * job = nullptr;
    std::atomic<int64_t>                 next{ 0 };
    int64_t                              total   = 0;
    size_t                               pending = 0;
    uint64_t                             gen     = 0;
    bool                                 stop    = false;
};

axera_pool & axera_host_pool();
