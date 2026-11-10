// Compare disk -> host and disk -> cached CMM reads for prompt-cache restores.
//
//   disk_cmm FILE MODE BLOCK_BYTES [BYTES] [REPS]
//
// MODE: host, cmm, cmm_buffered, host_to_cmm, host_to_cmm_chunk
// The caller drops the page cache before each cold run, under lockrun.sh host1a.

#include <ax_sys_api.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

static size_t read_block(int fd, void * dst, size_t block, off_t off) {
    size_t done = 0;
    while (done < block) {
        const ssize_t n = pread(fd, (char *) dst + done, block - done, off + (off_t) done);
        if (n <= 0) {
            std::fprintf(stderr, "disk_cmm: pread: %s\n", n == 0 ? "short file" : std::strerror(errno));
            return 0;
        }
        done += (size_t) n;
    }
    return done;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s FILE MODE BLOCK_BYTES [BYTES] [REPS]\n", argv[0]);
        std::fprintf(stderr, "MODE: host, cmm, cmm_buffered, host_to_cmm, host_to_cmm_chunk\n");
        return 2;
    }
    const char * path = argv[1];
    const std::string mode = argv[2];
    const size_t block = std::strtoull(argv[3], nullptr, 10);
    if (block == 0 || block % 4096 != 0) {
        std::fprintf(stderr, "disk_cmm: block must be a nonzero multiple of 4096\n");
        return 2;
    }
    const int reps = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    const bool chained = mode == "host_to_cmm" || mode == "host_to_cmm_chunk";
    const bool to_cmm = mode == "cmm" || mode == "cmm_buffered";
    const bool direct = mode == "cmm";

    struct stat st {};
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        std::fprintf(stderr, "disk_cmm: stat %s: %s\n", path, std::strerror(errno));
        return 2;
    }
    size_t bytes = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : (size_t) st.st_size;
    bytes = std::min(bytes, (size_t) st.st_size);
    bytes = (bytes / block) * block;
    if (bytes == 0) {
        return 2;
    }

    if (AX_SYS_Init() != 0) {
        std::fprintf(stderr, "disk_cmm: AX_SYS_Init failed\n");
        return 2;
    }
    AX_U64 phy = 0;
    void * cmm = nullptr;
    if (chained || to_cmm) {
        if (AX_SYS_MemAllocCached(&phy, &cmm, (AX_U32) bytes, 4096, (const AX_S8 *) "host1a-disk") != 0) {
            std::fprintf(stderr, "disk_cmm: AX_SYS_MemAllocCached(%zu) failed\n", bytes);
            return 2;
        }
    }
    void * host = nullptr;
    const size_t host_bytes = mode == "host_to_cmm" ? bytes : block;
    if (posix_memalign(&host, 4096, host_bytes) != 0 || !host) {
        std::fprintf(stderr, "disk_cmm: posix_memalign failed\n");
        return 2;
    }

    std::vector<double> read_ms, copy_ms, flush_ms, total_ms;
    for (int r = 0; r < reps; ++r) {
        int flags = O_RDONLY | O_CLOEXEC;
        if (direct) {
            flags |= O_DIRECT;
        }
        const int fd = open(path, flags);
        if (fd < 0) {
            std::fprintf(stderr, "disk_cmm: open %s: %s\n", path, std::strerror(errno));
            return 2;
        }
        const auto t0 = clk::now();
        double copy_total = 0.0;
        for (size_t off = 0; off < bytes; off += block) {
            const size_t n = std::min(block, bytes - off);
            void * dst = to_cmm ? (char *) cmm + off : host;
            if (read_block(fd, dst, n, (off_t) off) != n) {
                close(fd);
                return 2;
            }
            if (mode == "host_to_cmm_chunk") {
                const auto tc = clk::now();
                std::memcpy((char *) cmm + off, host, n);
                copy_total += ms_since(tc);
            }
        }
        read_ms.push_back(ms_since(t0) - copy_total);
        if (mode == "host_to_cmm" || mode == "host_to_cmm_chunk") {
            if (mode == "host_to_cmm") {
                const auto t2 = clk::now();
                std::memcpy(cmm, host, bytes);
                copy_total = ms_since(t2);
                read_ms.back() = ms_since(t0) - copy_total;
                copy_ms.push_back(copy_total);
            } else {
                copy_ms.push_back(copy_total);
            }
        } else {
            copy_ms.push_back(0.0);
        }
        const auto t3 = clk::now();
        if (cmm) {
            AX_SYS_MflushCache(phy, cmm, (AX_U32) bytes);
        }
        flush_ms.push_back(ms_since(t3));
        total_ms.push_back(ms_since(t0));
        close(fd);
    }

    const double total = median(total_ms);
    std::printf("RESULT,disk_cmm,%s,%zu,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%zu\n", mode.c_str(), bytes, block,
                direct ? 1 : 0, median(read_ms), median(copy_ms), median(flush_ms), total,
                (double) bytes / total / 1e6, total_ms.size());
    std::free(host);
    if (cmm) {
        AX_SYS_MemFree(phy, cmm);
    }
    return 0;
}
