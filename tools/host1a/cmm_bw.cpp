// CPU <-> CMM bandwidth probe.  Build and run on the board; no llama.cpp runtime needed.
//
//   cmm_bw [csv]
//
// The probe separates cached CMM, non-cacheable CMM and AX_DMA transfers, and reports the
// median of several repetitions.  Cached writes include AX_SYS_MflushCache(); cached reads
// include AX_SYS_MinvalidateCache(), matching the host paths in ggml-axera.

#include <ax_dmadim_api.h>
#include <ax_sys_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

struct cmm {
    AX_U64 phy = 0;
    void * vir = nullptr;
    size_t size = 0;
    bool cached = false;
};

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static cmm alloc_cmm(size_t size, bool cached) {
    cmm b;
    b.size = size;
    b.cached = cached;
    int ret = cached ? AX_SYS_MemAllocCached(&b.phy, &b.vir, (AX_U32) size, 128, (const AX_S8 *) "host1a-bw")
                     : AX_SYS_MemAlloc(&b.phy, &b.vir, (AX_U32) size, 128, (const AX_S8 *) "host1a-bw-nc");
    if (ret != 0 || !b.vir) {
        std::fprintf(stderr, "cmm_bw: AX_SYS_MemAlloc%s(%zu) failed: 0x%x\n", cached ? "Cached" : "", size, ret);
        std::exit(2);
    }
    std::memset(b.vir, 0, size);
    if (cached) {
        AX_SYS_MflushCache(b.phy, b.vir, (AX_U32) size);
    }
    return b;
}

static void free_cmm(cmm & b) {
    if (b.vir) {
        AX_SYS_MemFree(b.phy, b.vir);
    }
    b = {};
}

static void flush(cmm & b, size_t off = 0, size_t n = 0) {
    if (b.cached) {
        AX_SYS_MflushCache(b.phy + off, (char *) b.vir + off, (AX_U32) (n ? n : b.size));
    }
}

static void invalidate(cmm & b, size_t off = 0, size_t n = 0) {
    if (b.cached) {
        AX_SYS_MinvalidateCache(b.phy + off, (char *) b.vir + off, (AX_U32) (n ? n : b.size));
    }
}

static int reps_for(size_t size) {
    const size_t target = (size_t) 768 << 20;
    return (int) std::max<size_t>(3, std::min<size_t>(64, target / std::max<size_t>(size, 1)));
}

static void report(const char * mode, size_t bytes, int threads, const std::vector<double> & ms) {
    const double med = median(ms);
    const double lo  = *std::min_element(ms.begin(), ms.end());
    const double hi  = *std::max_element(ms.begin(), ms.end());
    const double gbps = med > 0 ? (double) bytes / med / 1e6 : 0.0;  // bytes/ms = GB/s decimal
    std::printf("RESULT,%s,%zu,%d,%.3f,%.3f,%.3f,%.3f,%zu\n", mode, bytes, threads, med, lo, hi, gbps, ms.size());
}

static void run_cpu(const char * mode, size_t size, int threads, int reps,
                    void (*fn)(void *, void *, size_t), void * dst, void * src, cmm * dst_cmm, cmm * src_cmm,
                    bool flush_dst_after, bool inval_src_before) {
    std::vector<double> times;
    times.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        if (threads == 1) {
            if (inval_src_before && src_cmm) {
                invalidate(*src_cmm);
            }
            fn(dst, src, size);
            if (flush_dst_after && dst_cmm) {
                flush(*dst_cmm);
            }
        } else {
            std::vector<std::thread> th;
            th.reserve((size_t) threads);
            for (int i = 0; i < threads; ++i) {
                const size_t a = size * (size_t) i / (size_t) threads;
                const size_t z = size * (size_t) (i + 1) / (size_t) threads;
                th.emplace_back([=] {
                    if (inval_src_before && src_cmm) {
                        invalidate(*src_cmm, a, z - a);
                    }
                    fn((char *) dst + a, (char *) src + a, z - a);
                    if (flush_dst_after && dst_cmm) {
                        flush(*dst_cmm, a, z - a);
                    }
                });
            }
            for (auto & t : th) {
                t.join();
            }
        }
        times.push_back(ms_since(t0));
    }
    report(mode, size, threads, times);
}

static void cpu_memcpy(void * d, void * s, size_t n) {
    std::memcpy(d, s, n);
}

static void cpu_memset(void * d, void *, size_t n) {
    std::memset(d, 0x3c, n);
}

static int run_snapshot(size_t bytes, int reps) {
    cmm a = alloc_cmm(bytes, true);
    cmm b = alloc_cmm(bytes, true);
    std::vector<double> cpu_copy, cpu_flush, dma, cpu_out, cpu_in;

    bool dma_ok = true;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        std::memcpy(b.vir, a.vir, bytes);
        cpu_copy.push_back(ms_since(t0));

        t0 = clk::now();
        std::memcpy(b.vir, a.vir, bytes);
        flush(b);
        cpu_flush.push_back(ms_since(t0));

        t0 = clk::now();
        flush(a);
        int ret = AX_DMA_MemCopy(b.phy, a.phy, (AX_U64) bytes);
        invalidate(b);
        dma.push_back(ms_since(t0));
        if (ret != 0) {
            std::fprintf(stderr, "cmm_bw: snapshot AX_DMA_MemCopy(%zu) failed: 0x%x\n", bytes, ret);
            dma_ok = false;
            break;
        }
    }

    auto * host = (char *) std::aligned_alloc(4096, bytes);
    if (!host) {
        return 2;
    }
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        invalidate(a);
        std::memcpy(host, a.vir, bytes);
        cpu_out.push_back(ms_since(t0));

        t0 = clk::now();
        std::memcpy(a.vir, host, bytes);
        flush(a);
        cpu_in.push_back(ms_since(t0));
    }

    auto report_snapshot = [&](const char * mode, const std::vector<double> & v) {
        if (v.empty()) {
            return;
        }
        std::printf("SNAPSHOT,%s,%zu,%.3f,%.3f,%.3f,%.3f,%zu\n", mode, bytes, median(v),
                    *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()),
                    (double) bytes / median(v) / 1e6, v.size());
    };
    report_snapshot("cmm_cpu_copy", cpu_copy);
    report_snapshot("cmm_cpu_copy_flush", cpu_flush);
    if (dma_ok) {
        report_snapshot("cmm_dma_flush_inval", dma);
    }
    report_snapshot("cmm_to_sys_inval", cpu_out);
    report_snapshot("sys_to_cmm_flush", cpu_in);

    std::free(host);
    free_cmm(a);
    free_cmm(b);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    std::printf("KIND,MODE,BYTES,THREADS,MEDIAN_MS,MIN_MS,MAX_MS,GBPS,SAMPLES\n");
    std::fprintf(stderr, "cmm_bw: sizes 4KiB..256MiB, cached CMM includes flush/invalidate\n");
    if (AX_SYS_Init() != 0) {
        std::fprintf(stderr, "cmm_bw: AX_SYS_Init failed\n");
        return 2;
    }

    if (argc > 1 && std::string(argv[1]) == "snapshot") {
        const size_t mib = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 590;
        return run_snapshot(mib << 20, argc > 3 ? std::max(1, std::atoi(argv[3])) : 5);
    }

    static const size_t sizes[] = {
        4u << 10, 64u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20, 256u << 20,
    };
    const size_t max_size = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];
    auto * sys_a = (char *) std::aligned_alloc(128, max_size);
    auto * sys_b = (char *) std::aligned_alloc(128, max_size);
    if (!sys_a || !sys_b) {
        return 2;
    }
    std::memset(sys_a, 0x11, max_size);
    std::memset(sys_b, 0x22, max_size);
    cmm cached_a = alloc_cmm(max_size, true);
    cmm cached_b = alloc_cmm(max_size, true);
    cmm nc_a     = alloc_cmm(max_size, false);
    cmm nc_b     = alloc_cmm(max_size, false);

    // Validate the 1D DMA path with a small pattern before timing it.
    bool dma_ok = false;
    {
        const size_t n = 64u << 10;
        std::memset(cached_a.vir, 0x55, n);
        std::memset(cached_b.vir, 0xaa, n);
        flush(cached_a, 0, n);
        flush(cached_b, 0, n);
        int r = AX_DMA_MemCopy(cached_b.phy, cached_a.phy, (AX_U64) n);
        invalidate(cached_b, 0, n);
        dma_ok = r == 0 && ((const unsigned char *) cached_b.vir)[0] == 0x55;
        std::fprintf(stderr, "cmm_bw: AX_DMA_MemCopy 1D %s\n", dma_ok ? "ok" : "unavailable");
    }

    for (size_t size : sizes) {
        const int reps = reps_for(size);
        run_cpu("sys_memset", size, 1, reps, cpu_memset, sys_a, sys_b, nullptr, nullptr, false, false);
        run_cpu("sys_memcpy", size, 1, reps, cpu_memcpy, sys_a, sys_b, nullptr, nullptr, false, false);
        run_cpu("sys_to_cmm_cached", size, 1, reps, cpu_memcpy, cached_a.vir, sys_a, &cached_a, nullptr, false, false);
        run_cpu("sys_to_cmm_cached_flush", size, 1, reps, cpu_memcpy, cached_a.vir, sys_a, &cached_a, nullptr, true, false);
        run_cpu("cmm_cached_to_sys", size, 1, reps, cpu_memcpy, sys_a, cached_a.vir, nullptr, &cached_a, false, false);
        run_cpu("cmm_cached_to_sys_inval", size, 1, reps, cpu_memcpy, sys_a, cached_a.vir, nullptr, &cached_a, false, true);
        run_cpu("sys_to_cmm_nc", size, 1, reps, cpu_memcpy, nc_a.vir, sys_a, nullptr, nullptr, false, false);
        run_cpu("cmm_nc_to_sys", size, 1, reps, cpu_memcpy, sys_a, nc_a.vir, nullptr, nullptr, false, false);
        for (int threads : { 1, 2, 4, 8 }) {
            if ((size_t) threads > size / 4096) {
                continue;
            }
            run_cpu("sys_to_cmm_cached", size, threads, std::max(3, reps / std::max(1, threads)),
                    cpu_memcpy, cached_a.vir, sys_a, &cached_a, nullptr, false, false);
            run_cpu("sys_to_cmm_cached_flush", size, threads, std::max(3, reps / std::max(1, threads)),
                    cpu_memcpy, cached_a.vir, sys_a, &cached_a, nullptr, true, false);
            run_cpu("cmm_cached_to_sys", size, threads, std::max(3, reps / std::max(1, threads)),
                    cpu_memcpy, sys_a, cached_a.vir, nullptr, &cached_a, false, false);
            run_cpu("cmm_cached_to_sys_inval", size, threads, std::max(3, reps / std::max(1, threads)),
                    cpu_memcpy, sys_a, cached_a.vir, nullptr, &cached_a, false, true);
        }

        if (dma_ok) {
            std::vector<double> times;
            times.reserve((size_t) reps);
            for (int r = 0; r < reps; ++r) {
                flush(cached_a);
                flush(cached_b);
                const auto t0 = clk::now();
                const int ret = AX_DMA_MemCopy(cached_b.phy, cached_a.phy, (AX_U64) size);
                const double dt = ms_since(t0);
                invalidate(cached_b);
                if (ret != 0) {
                    std::fprintf(stderr, "cmm_bw: AX_DMA_MemCopy(%zu) failed: 0x%x\n", size, ret);
                    dma_ok = false;
                    break;
                }
                times.push_back(dt);
            }
            if (!times.empty()) {
                report("cmm_cached_dma_flush", size, 1, times);
            }
        }

        if (dma_ok) {
            std::vector<double> times;
            times.reserve((size_t) reps);
            for (int r = 0; r < reps; ++r) {
                const auto t0 = clk::now();
                const int ret = AX_DMA_MemCopyCoherent(cached_b.phy, cached_a.phy, (AX_U64) size);
                const double dt = ms_since(t0);
                if (ret != 0) {
                    std::fprintf(stderr, "cmm_bw: AX_DMA_MemCopyCoherent(%zu) failed: 0x%x\n", size, ret);
                    break;
                }
                times.push_back(dt);
            }
            if (!times.empty()) {
                report("cmm_cached_dma_coherent", size, 1, times);
            }
        }
    }

    if (!dma_ok) {
        std::printf("INFO,AX_DMA_MemCopy unavailable or rejected the 1D transfer\n");
    }

    free_cmm(cached_a);
    free_cmm(cached_b);
    free_cmm(nc_a);
    free_cmm(nc_b);
    std::free(sys_a);
    std::free(sys_b);
    return 0;
}
