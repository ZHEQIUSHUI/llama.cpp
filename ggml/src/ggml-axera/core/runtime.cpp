// Axera NPU backend core: the NPU runtime, CMM buffers, the host worker pool

#include "runtime.h"

#include "ggml-axera.h"
#include "ggml-backend.h"

#include <algorithm>
#include <dirent.h>
#include <fcntl.h>

#include <fstream>
#include <dlfcn.h>
#include <iterator>

#include <sys/mman.h>
#include <unistd.h>

namespace {

using async_task_id_t         = uint64_t;
using async_output_attr_t     = uint32_t;
using async_context_t         = AX_ENGINE_CONTEXT_T;
constexpr async_output_attr_t ASYNC_OUT_UNCACHED = 2;
constexpr async_output_attr_t ASYNC_OUT_CACHED   = 3;
constexpr uint32_t            ASYNC_CAP          = 1U << 0;
constexpr int32_t             ASYNC_PARAM_INVALID = (int32_t) 0x80060084U;
constexpr int32_t             ASYNC_NOT_SUPPORT  = (int32_t) 0x80060088U;
constexpr int32_t             ASYNC_TASK_BUSY    = (int32_t) 0x80060089U;
constexpr int32_t             ASYNC_NOT_SUPPORTED = (int32_t) 0x800600a2U;

struct async_options {
    uint32_t              nSize;
    uint32_t              nOutputAttrCount;
    const async_output_attr_t * pOutputAttrs;
    uint64_t              u64Reserved[4];
};

using create_context_fn = int32_t (*)(AX_ENGINE_HANDLE, async_context_t *);
using callback_fn = void (*)(AX_ENGINE_HANDLE, async_context_t, async_task_id_t, int32_t, void *);
using run_fn = int32_t (*)(AX_ENGINE_HANDLE, async_context_t, const AX_ENGINE_IO_T *, const async_options *,
                           callback_fn, void *, async_task_id_t *);
using run_group_fn = int32_t (*)(AX_ENGINE_HANDLE, async_context_t, uint32_t, const AX_ENGINE_IO_T *,
                                 const async_options *, callback_fn, void *, async_task_id_t *);
using wait_fn      = int32_t (*)(AX_ENGINE_HANDLE, async_task_id_t, uint32_t, int32_t *);
using capability_fn = int32_t (*)(uint32_t *);

struct async_api {
    create_context_fn create_context = nullptr;
    run_fn            run            = nullptr;
    run_group_fn      run_group      = nullptr;
    wait_fn           wait           = nullptr;
    capability_fn     capability     = nullptr;
};

static async_api load_async_api(std::string & why) {
    async_api api;
    void * lib = dlopen("libax_engine.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        why = std::string("dlopen(libax_engine.so): ") + dlerror();
        return api;
    }
    api.create_context = reinterpret_cast<create_context_fn>(dlsym(lib, "AX_ENGINE_CreateAsyncContext"));
    api.run            = reinterpret_cast<run_fn>(dlsym(lib, "AX_ENGINE_RunAsync"));
    api.run_group      = reinterpret_cast<run_group_fn>(dlsym(lib, "AX_ENGINE_RunGroupIOAsync"));
    api.wait           = reinterpret_cast<wait_fn>(dlsym(lib, "AX_ENGINE_WaitAsync"));
    api.capability     = reinterpret_cast<capability_fn>(dlsym(lib, "AX_ENGINE_GetCapability"));
    if (!api.create_context || !api.run || !api.run_group || !api.wait || !api.capability) {
        why = "libax_engine.so lacks the Task-ID async API";
    }
    return api;
}

// V0.6.0 rejects a NULL callback with PARAM_INVALID even though polling via
// WaitAsync is documented. A no-op callback is still safe: WaitAsync owns the
// task result and the callback does not need to release anything.
static void async_noop(AX_ENGINE_HANDLE, async_context_t, async_task_id_t, int32_t, void *) {}

}  // namespace

std::atomic<uint64_t> axera_graph_epoch{ 0 };

static std::atomic<int64_t> g_axera_kv_cells{ 0 };

void axera_note_kv_cells(int64_t cells) {
    int64_t cur = g_axera_kv_cells.load();
    while (cells > cur && !g_axera_kv_cells.compare_exchange_weak(cur, cells)) {
    }
}

int64_t axera_kv_cells() {
    return g_axera_kv_cells.load();
}

void axera_thermal_guard() {
    static const int limit = [] {
        const char * e = getenv("GGML_AXERA_TEMP_MAX");
        return e ? atoi(e) : 100;
    }();
    if (limit <= 0) {
        return;
    }
    static const std::vector<std::string> sensors = [] {
        std::vector<std::string> v;
        if (DIR * d = opendir("/sys/class/thermal")) {
            while (const dirent * e = readdir(d)) {
                const std::string p = std::string("/sys/class/thermal/") + e->d_name;
                std::ifstream     t(p + "/type");
                std::string       type;
                if (t && std::getline(t, type) && type.compare(0, 11, "thermal_npu") == 0) {
                    v.push_back(p + "/temp");
                }
            }
            closedir(d);
        }
        return v;
    }();
    if (sensors.empty()) {
        return;
    }
    static std::mutex                                      mu;
    static std::chrono::steady_clock::time_point           last;
    static bool                                            hot = false;
    const auto                                             read = [] {
        int m = 0;
        for (const auto & p : sensors) {
            std::ifstream f(p);
            int           t = 0;
            if (f >> t) {
                m = std::max(m, t / 1000);
            }
        }
        return m;
    };
    std::lock_guard<std::mutex> lk(mu);
    const auto                  now = std::chrono::steady_clock::now();
    if (!hot && now - last < std::chrono::milliseconds(200)) {  // (a sensor read per kernel is too often)
        return;
    }
    last  = now;
    int t = read();
    if (t < limit) {
        return;
    }
    axera_timer tw("thermal pause");
    fprintf(stderr, "axera: NPU %d °C: pausing until %d °C (GGML_AXERA_TEMP_MAX=%d)\n", t, limit - 5, limit);
    hot = true;
    while (t > limit - 5) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        t = read();
    }
    hot  = false;
    last = std::chrono::steady_clock::now();
}

// AX_SYS / AX_ENGINE are process-wide; initialized once, never torn down while tensors may still live in CMM
bool axera_runtime_init() {
    static std::once_flag once;
    static bool           ok = false;
    std::call_once(once, [] {
        if (AX_SYS_Init() != 0) {
            GGML_LOG_WARN("%s: AX_SYS_Init failed, Axera NPU unavailable\n", __func__);
            return;
        }
        AX_ENGINE_NPU_ATTR_T attr;
        memset(&attr, 0, sizeof(attr));
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        const int ret  = AX_ENGINE_Init(&attr);
        if (ret != 0) {
            GGML_LOG_WARN("%s: AX_ENGINE_Init failed (0x%x), Axera NPU unavailable\n", __func__, ret);
            return;
        }
        ok = true;
    });
    return ok;
}

bool axera_keep_pages() {
    static const bool keep = [] {
        const char * e = std::getenv("GGML_AXERA_KEEP_PAGES");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return keep;
}

// the GGUF file mappings (/proc/self/maps): a tensor's address -> its file and file offset. The mappings outlive the
// model's tensors, and their fds are kept open for the process's fadvise calls
struct axera_file_mapping {
    uintptr_t   start;
    uintptr_t   end;
    off_t       off;
    std::string path;
};

static std::mutex                      g_file_mappings_mu;
static std::vector<axera_file_mapping> g_file_mappings;
static bool                            g_file_mappings_loaded = false;
static std::map<std::string, int>      g_file_fds;

struct axera_file_range {
    int   fd;
    off_t off;
    off_t end;
};

// fadvise is synchronous, and calling it per tensor during the load evicts pages that another bank reads again (the 27B
// load made ~2x the file's bytes and spent ~10 s in fadvise). A converted tensor queues its range instead; the model's
// init flushes the queue once, after every bank has read the file, coalescing the ranges into a few fadvise calls.
static std::mutex                      g_file_pages_mu;
static std::vector<axera_file_range>   g_file_pages_pending;

static void axera_read_file_mappings() {
    g_file_mappings.clear();
    FILE * f = std::fopen("/proc/self/maps", "r");
    if (!f) {
        return;
    }
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long lo = 0, hi = 0, off = 0, inode = 0;
        char          perms[5] = {}, dev[16] = {};
        int           used = 0;
        if (std::sscanf(line, "%lx-%lx %4s %lx %15s %lu %n", &lo, &hi, perms, &off, dev, &inode, &used) != 6 ||
            used <= 0 || perms[0] != 'r') {
            continue;
        }
        std::string path(line + used);
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) {
            path.pop_back();
        }
        if (path.empty() || path[0] == '[') {
            continue;
        }
        g_file_mappings.push_back({ (uintptr_t) lo, (uintptr_t) hi, (off_t) off, path });
    }
    std::fclose(f);
    std::sort(g_file_mappings.begin(), g_file_mappings.end(),
              [](const axera_file_mapping & a, const axera_file_mapping & b) { return a.start < b.start; });
    g_file_mappings_loaded = true;
}

static const axera_file_mapping * axera_find_file_mapping(uintptr_t p) {
    if (!g_file_mappings_loaded) {
        axera_read_file_mappings();
    }
    auto it = std::upper_bound(g_file_mappings.begin(), g_file_mappings.end(), p,
                               [](uintptr_t x, const axera_file_mapping & m) { return x < m.start; });
    auto found = [&](auto i) { return i != g_file_mappings.begin() && p < std::prev(i)->end ? &*std::prev(i) : nullptr; };
    if (const axera_file_mapping * m = found(it)) {
        return m;
    }
    // a file mapped since the last scan (the model and the mmproj are mapped at different times)
    axera_read_file_mappings();
    it = std::upper_bound(g_file_mappings.begin(), g_file_mappings.end(), p,
                          [](uintptr_t x, const axera_file_mapping & m) { return x < m.start; });
    return found(it);
}

static int axera_file_fd(const std::string & path) {
    auto it = g_file_fds.find(path);
    if (it != g_file_fds.end()) {
        return it->second;
    }
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    g_file_fds.emplace(path, fd);
    return fd;
}

void axera_drop_file_pages(const ggml_tensor * t) {
    if (axera_keep_pages() || !t || !t->data || ggml_nbytes(t) == 0) {
        return;
    }
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t        lo = (uintptr_t) t->data;
    {
        std::lock_guard<std::mutex> lock(g_file_mappings_mu);
        const axera_file_mapping * m = axera_find_file_mapping(lo);
        if (!m || lo >= m->end) {
            return;
        }
        off_t off = m->off + (off_t) (lo - m->start);
        off_t end = off + (off_t) ggml_nbytes(t);
        off      &= ~(off_t) (pg - 1);
        end       = (end + (off_t) pg - 1) & ~(off_t) (pg - 1);
        const int fd = axera_file_fd(m->path);
        if (end <= off || fd < 0) {
            return;
        }
        std::lock_guard<std::mutex> pages_lock(g_file_pages_mu);
        g_file_pages_pending.push_back({ fd, off, end });
    }
}

void axera_flush_file_pages() {
    std::vector<axera_file_range> ranges;
    {
        std::lock_guard<std::mutex> lock(g_file_pages_mu);
        ranges.swap(g_file_pages_pending);
    }
    if (ranges.empty()) {
        return;
    }
    std::sort(ranges.begin(), ranges.end(), [](const axera_file_range & a, const axera_file_range & b) {
        return a.fd != b.fd ? a.fd < b.fd : a.off < b.off;
    });
    size_t i = 0;
    while (i < ranges.size()) {
        const int   fd  = ranges[i].fd;
        off_t       off = ranges[i].off;
        off_t       end = ranges[i].end;
        while (++i < ranges.size() && ranges[i].fd == fd && ranges[i].off <= end) {
            end = std::max(end, ranges[i].end);
        }
        posix_fadvise(fd, off, end - off, POSIX_FADV_DONTNEED);
    }
}

void axera_drop_mapped(const ggml_tensor * t, bool force) {
    const ggml_tensor * s = t->view_src ? t->view_src : t;
    const bool token_table = strncmp(s->name, "token_embd", 10) == 0 ||  // (rows read on the CPU per token / frame)
                             strcmp(s->name, "a.gen.code.embd.weight") == 0 ||
                             strcmp(s->name, "a.gen.code.out_embd.weight") == 0;
    if (!s->buffer || strcmp(ggml_backend_buffer_name(s->buffer), GGML_AXERA_NAME "_Mapped") != 0 ||
        (token_table && (!force || axera_keep_pages()))) {
        return;
    }
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t        a  = ((uintptr_t) t->data + pg - 1) & ~(pg - 1);  // whole pages within the tensor
    const uintptr_t        b  = ((uintptr_t) t->data + ggml_nbytes(t)) & ~(pg - 1);
    if (b > a) {
        madvise((void *) a, b - a, MADV_DONTNEED);
    }
    axera_drop_file_pages(t);
}

// CMM total / free from /proc/ax_proc/mem_cmm_info (sum over partitions)
void axera_cmm_memory(size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    std::ifstream f("/proc/ax_proc/mem_cmm_info");
    std::string   line;
    while (std::getline(f, line)) {
        size_t p = line.find("PARTITION:");
        if (p != std::string::npos) {
            const size_t s = line.find("Size=", p);
            if (s != std::string::npos) {
                const size_t kb = std::strtoull(line.c_str() + s + 5, nullptr, 10);
                *total += kb * 1024;
                *free += kb * 1024;
            }
            continue;
        }
        p = line.find("nbytes(");
        if (p != std::string::npos) {
            const size_t c = line.find("Cur=", p);
            if (c != std::string::npos) {
                *free -= std::min(*free, (size_t) std::strtoull(line.c_str() + c + 4, nullptr, 10));
            }
        }
    }
}

axera_cmm axera_cmm_alloc(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    if (AX_SYS_MemAllocCached(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
    memset(b->vir, 0, b->size);
    b->flush(b->vir, b->size);
    return b;
}

axera_cmm axera_cmm_alloc_raw(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    if (AX_SYS_MemAllocCached(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
    return b;
}

axera_cmm axera_cmm_alloc_nc(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    b->nc   = true;
    if (AX_SYS_MemAlloc(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera-nc") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
    return b;
}

axera_cmm axera_cmm_view(const ggml_backend_axera_buffer_context * b, size_t off, size_t size) {
    GGML_ASSERT(off + size <= b->size);
    auto v  = std::make_unique<ggml_backend_axera_buffer_context>();
    v->phy  = b->phy + off;
    v->vir  = (char *) b->vir + off;
    v->size = size;
    v->nc   = b->nc;
    v->view = true;
    return v;
}

AX_ENGINE_IO_BUFFER_T axera_io(const ggml_backend_axera_buffer_context * ctx, void * p, uint32_t size) {
    AX_ENGINE_IO_BUFFER_T b;
    memset(&b, 0, sizeof(b));
    b.phyAddr  = ctx->phy_of(p);
    b.pVirAddr = p;
    b.nSize    = size;
    return b;
}

axera_pool & axera_host_pool() {
    static axera_pool pool([] {
        const char * e = std::getenv("AXERA_HOST_POOL_THREADS");
        const int    n = e ? std::atoi(e) : 0;
        return n > 0 ? n : std::min(15, std::max(1, (int) std::thread::hardware_concurrency() - 1));
    }());
    return pool;
}

struct axera_async_queue::impl {
    struct pending {
        std::vector<AX_ENGINE_IO_BUFFER_T>          in;
        std::vector<AX_ENGINE_IO_BUFFER_T>          out;
        std::vector<async_output_attr_t>            attrs;
        AX_ENGINE_IO_T                              io = {};
        async_options                               options = {};
        async_task_id_t                             id = 0;
        async_context_t                             ctx = nullptr;
    };

    const bool requested = [] {
        const char * e = std::getenv("GGML_AXERA_ASYNC");
        return e != nullptr && std::atoi(e) != 0;
    }();

    bool             api_tried = false;
    bool             api_ok    = false;
    bool             failed    = false;
    async_api        api;
    AX_ENGINE_HANDLE handle    = nullptr;
    uint32_t         groups    = 1;
    std::vector<async_context_t> contexts;
    std::vector<pending>         queued;
    std::string                  why;

    bool enabled() {
        if (!requested) {
            return false;
        }
        if (!api_tried) {
            api_tried = true;
            api       = load_async_api(why);
            if (!why.empty()) {
                GGML_LOG_WARN("%s: async disabled: %s\n", __func__, why.c_str());
                return false;
            }
            uint32_t cap = 0;
            const int32_t rc = api.capability(&cap);
            if (rc != 0 || !(cap & ASYNC_CAP)) {
                why = "runtime does not advertise AX_ENGINE_CAP_ASYNC";
                GGML_LOG_WARN("%s: async disabled: %s (rc 0x%08x, cap 0x%08x)\n", __func__, why.c_str(),
                              (unsigned) rc, cap);
                return false;
            }
            api_ok = true;
        }
        return api_ok && !failed;
    }

    bool reserve(AX_ENGINE_HANDLE h, size_t n) {
        if (!enabled()) {
            return false;
        }
        if (n > 64) {
            why = "more than 64 in-flight contexts requested";
            return false;
        }
        if (!handle) {
            handle = h;
            if (AX_ENGINE_GetGroupIOInfoCount(handle, &groups) != 0) {
                groups = 1;
            }
        } else if (handle != h) {
            why = "one async queue cannot serve two model handles";
            return false;
        }
        while (contexts.size() < n) {
            async_context_t ctx = nullptr;
            const int32_t rc = api.create_context(handle, &ctx);
            if (rc != 0 || !ctx) {
                why = "AX_ENGINE_CreateAsyncContext failed";
                GGML_LOG_ERROR("%s: %s for context %zu (rc 0x%08x)\n", __func__, why.c_str(), contexts.size(),
                               (unsigned) rc);
                return false;
            }
            contexts.push_back(ctx);
        }
        return true;
    }

    bool submit(AX_ENGINE_HANDLE h, const std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                const std::vector<AX_ENGINE_IO_BUFFER_T> & out, bool cached_outputs) {
        if (!reserve(h, queued.size() + 1)) {
            return false;
        }
        pending q;
        q.in  = in;
        q.out = out;
        q.attrs.assign(out.size(), cached_outputs ? ASYNC_OUT_CACHED : ASYNC_OUT_UNCACHED);
        q.io.pInputs       = q.in.data();
        q.io.nInputSize    = (uint32_t) q.in.size();
        q.io.pOutputs      = q.out.data();
        q.io.nOutputSize   = (uint32_t) q.out.size();
        q.io.u64Reserved[0] = 1;  // AX_ENGINE_IO_FLAG_ALLOW_INPLACE
        q.ctx              = contexts[queued.size()];

        q.options.nSize            = sizeof(q.options);
        q.options.nOutputAttrCount = (uint32_t) q.attrs.size();
        q.options.pOutputAttrs     = q.attrs.empty() ? nullptr : q.attrs.data();

        // The semantics probe showed that RunAsync accepts queued decode calls;
        // RunGroupIOAsync can reject the second in-flight call with
        // PARAM_INVALID even when the first accepted. Keep the probe's order and
        // use the group entry point only for a runtime that asks for it.
        int32_t rc = api.run(handle, q.ctx, &q.io, &q.options, async_noop, nullptr, &q.id);
        if (rc == ASYNC_NOT_SUPPORT || rc == ASYNC_NOT_SUPPORTED || rc == ASYNC_PARAM_INVALID) {
            rc = api.run_group(handle, q.ctx, 0, &q.io, &q.options, async_noop, nullptr, &q.id);
        }
        if (rc != 0 || !q.id) {
            why = "async submission failed";
            GGML_LOG_ERROR("%s: %s: ctx %zu rc 0x%08x id %llu (groups %u)\n", __func__, why.c_str(), queued.size(),
                           (unsigned) rc, (unsigned long long) q.id, groups);
            failed = true;
            return false;
        }
        queued.push_back(std::move(q));
        return true;
    }

    bool wait() {
        bool ok = true;
        for (pending & q : queued) {
            int32_t result = 0;
            const int32_t rc = api.wait(handle, q.id, 60000, &result);
            if (rc != 0 || result != 0) {
                why = "AX_ENGINE_WaitAsync failed";
                GGML_LOG_ERROR("%s: %s: id %llu rc 0x%08x result 0x%08x\n", __func__, why.c_str(),
                               (unsigned long long) q.id, (unsigned) rc, (unsigned) result);
                ok = false;
            }
        }
        queued.clear();
        return ok;
    }
};

axera_async_queue::axera_async_queue() : p(new impl) {}
axera_async_queue::~axera_async_queue() = default;

bool axera_async_queue::enabled() const {
    return const_cast<impl *>(p.get())->enabled();
}

bool axera_async_queue::reserve(AX_ENGINE_HANDLE handle, size_t n) {
    return p->reserve(handle, n);
}

bool axera_async_queue::submit(AX_ENGINE_HANDLE handle, const std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                               const std::vector<AX_ENGINE_IO_BUFFER_T> & out, bool cached_outputs) {
    return p->submit(handle, in, out, cached_outputs);
}

bool axera_async_queue::wait() {
    return p->wait();
}

const std::string & axera_async_queue::error() const {
    return p->why;
}
