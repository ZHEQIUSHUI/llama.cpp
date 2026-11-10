// Raw sequential read probe for the board.  Each worker reads one contiguous segment of the first
// --bytes bytes with pread(); direct mode opens with O_DIRECT.  The caller drops the page cache
// before a run, under the host1a lock.
//
//   disk_bw FILE BLOCK_BYTES THREADS DIRECT(0|1) [BYTES] [REPS]

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
#include <thread>
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

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s FILE BLOCK_BYTES THREADS DIRECT(0|1) [BYTES] [REPS]\n", argv[0]);
        return 2;
    }
    const char * path  = argv[1];
    const size_t block = std::strtoull(argv[2], nullptr, 10);
    const int    nth   = std::max(1, std::atoi(argv[3]));
    const bool   direct = std::atoi(argv[4]) != 0;
    const int    reps  = argc > 6 ? std::max(1, std::atoi(argv[6])) : 1;

    struct stat st {};
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        std::fprintf(stderr, "disk_bw: stat %s: %s\n", path, std::strerror(errno));
        return 2;
    }
    size_t bytes = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : (size_t) 1 << 30;
    bytes = std::min(bytes, (size_t) st.st_size);
    bytes = (bytes / block) * block;
    if (bytes == 0) {
        return 2;
    }

    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        std::vector<std::thread> workers;
        std::vector<int>         errs((size_t) nth, 0);
        workers.reserve((size_t) nth);
        for (int t = 0; t < nth; ++t) {
            const size_t a = (bytes / block) * (size_t) t / (size_t) nth * block;
            const size_t z = (bytes / block) * (size_t) (t + 1) / (size_t) nth * block;
            workers.emplace_back([&, t, a, z] {
                int flags = O_RDONLY | O_CLOEXEC;
                if (direct) {
                    flags |= O_DIRECT;
                }
                const int fd = open(path, flags);
                if (fd < 0) {
                    errs[t] = errno;
                    return;
                }
                void * buf = nullptr;
                if (posix_memalign(&buf, 4096, block) != 0 || !buf) {
                    errs[t] = ENOMEM;
                    close(fd);
                    return;
                }
                for (size_t off = a; off < z; off += block) {
                    size_t done = 0;
                    while (done < block) {
                        const ssize_t n = pread(fd, (char *) buf + done, block - done, (off_t) (off + done));
                        if (n <= 0) {
                            errs[t] = n == 0 ? EIO : errno;
                            break;
                        }
                        done += (size_t) n;
                    }
                    if (done != block) {
                        break;
                    }
                }
                std::free(buf);
                close(fd);
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        for (int e : errs) {
            if (e) {
                std::fprintf(stderr, "disk_bw: pread: %s\n", std::strerror(e));
                return 2;
            }
        }
        times.push_back(ms_since(t0));
    }

    const double med = median(times);
    std::printf("RESULT,disk,%s,%zu,%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%zu\n", path, bytes, block, nth, direct ? 1 : 0,
                med, med, med, (double) bytes / med / 1e6, times.size());
    std::fprintf(stderr, "disk_bw: %s block=%zu threads=%d direct=%d bytes=%zu median=%.3f ms %.3f GB/s\n",
                 path, block, nth, direct ? 1 : 0, bytes, med, (double) bytes / med / 1e6);
    return 0;
}
