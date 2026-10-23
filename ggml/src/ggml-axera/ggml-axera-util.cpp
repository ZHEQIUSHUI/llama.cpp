// Axera NPU backend: profiling, host thread pool, runtime, kernel registry, CMM, settings

#include "ggml-axera-impl.h"
bool axera_debug() {
    static const bool debug = [] {
        const char * e = std::getenv("GGML_AXERA_DEBUG");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return debug;
}
axera_pool & axera_host_pool() {
    static axera_pool pool(std::min(15, std::max(1, (int) std::thread::hardware_concurrency() - 1)));
    return pool;
}

// AX_SYS / AX_ENGINE are process-wide; initialized once, never torn down while tensors may still live in CMM
bool axera_runtime_init() {
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
void axera_cmm_memory(size_t * free, size_t * total) {
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
ggml_axera_kernel_registry & axera_kernels() {
    static ggml_axera_kernel_registry registry;
    return registry;
}
std::string axera_mul_mat_kernel_name(int64_t c, int64_t k, bool mx8) {
    char name[96];
    snprintf(name, sizeof(name), "mul_mat_%s_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", c, k);
    return name;
}

//
// backend
//

// CMM owned by the backend (staging, template state); same cache rules as the buffers above
std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc(size_t size) {
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
std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_alloc_nc(size_t size) {
    auto b  = std::make_unique<ggml_backend_axera_buffer_context>();
    b->size = GGML_PAD(std::max<size_t>(size, 1), AXERA_ALIGNMENT);
    b->nc   = true;
    if (AX_SYS_MemAlloc(&b->phy, &b->vir, (AX_U32) b->size, AXERA_ALIGNMENT, (const AX_S8 *) "ggml-axera-nc") != 0) {
        GGML_ABORT("%s: failed to allocate %zu bytes of CMM\n", __func__, size);
    }
    return b;
}
std::unique_ptr<ggml_backend_axera_buffer_context> axera_cmm_view(const ggml_backend_axera_buffer_context * b, size_t off,
                                                                  size_t size) {
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

// one kernel call; the AX_ENGINE_RunSync status
int axera_run_sync(ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                   std::vector<AX_ENGINE_IO_BUFFER_T> & out) {
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.pInputs     = in.data();
    io.nInputSize  = (uint32_t) in.size();
    io.pOutputs    = out.data();
    io.nOutputSize = (uint32_t) out.size();
    const int ret  = AX_ENGINE_RunSync(k->handle, &io);
    if (ret != 0) {  // the IO the kernel got, for the caller's abort (stderr: the log is asynchronous)
        for (int o = 0; o < 2; ++o) {
            const auto & bufs = o ? out : in;
            for (size_t i = 0; i < bufs.size(); ++i) {
                const AX_ENGINE_IOMETA_T & mt = o ? k->info->pOutputs[i] : k->info->pInputs[i];
                fprintf(stderr, "%s: RunSync 0x%x: %s %zu %s: %u bytes (bound %u at phy 0x%llx)\n", __func__, ret,
                               o ? "output" : "input", i, mt.pName, mt.nSize, bufs[i].nSize,
                               (unsigned long long) bufs[i].phyAddr);
            }
        }
    }
    return ret;
}

// GGML_AXERA_TEMPLATE=qwen35 (or qwen3): the graph template of Qwen3.5 and Qwen3 transformers (ggml-axera-q35.h)
bool axera_template_qwen35() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_TEMPLATE");
        return e != nullptr && (strcmp(e, "qwen35") == 0 || strcmp(e, "qwen3") == 0);
    }();
    return on;
}

// GGML_AXERA_FLASH=1: split attention (flash decoding) even when a fused attention kernel holds the KV cache
// GGML_AXERA_SPLIT_MIN: with a fused attention kernel, the KQ mask width (cells) from which a batch takes the split
// attention kernels instead when they exist for the context (multi-core segment scan; default 4096, 0: always)
std::atomic<uint64_t> axera_graph_epoch{ 0 };

bool axera_nonexp_nv4() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_NONEXP");
        return e != nullptr && strcmp(e, "nvfp4") == 0;
    }();
    return on;
}

int64_t axera_split_min() {
    static const int64_t n = [] {
        const char * e = std::getenv("GGML_AXERA_SPLIT_MIN");
        return e ? std::max<int64_t>(0, std::atoll(e)) : (int64_t) 4096;
    }();
    return n;
}

// GGML_AXERA_CORES: NPU cores for the kernels that have multi-core builds (default 1: single-core kernels only; 4:
// the multi-core builds, experimental: inside llama.cpp they still give wrong results or time out now and then)
int axera_cores() {
    static const int n = [] {
        const char * e = std::getenv("GGML_AXERA_CORES");
        return e ? std::max(1, std::atoi(e)) : 4;  // all cores; GGML_AXERA_CORES=1 for the single-core kernels
    }();
    return n;
}

bool axera_flash_forced() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_FLASH");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return on;
}

// KV segment (cells) of the split attention kernels qwen35_attn_fd_seg<R>_c<S> (Qwen3: qwen3_*), 0 if there are none
// the split attention kernels' segment for a KV cache of n cells: the largest S of the *_attn_fd_seg1_c<S> kernels
// within n (fewer segments: fewer calls per layer, and their partial results fit the multi-core tail's merge slots),
// else the smallest
int64_t axera_flash_seg(int64_t n) {
    for (const char * k : { "qwen35_attn_fd_seg1", "qwen3_attn_fd_seg1" }) {
        int64_t s = axera_kernels().ctx_at_most(k, n);
        if (s > 0 || (s = axera_kernels().ctx_for(k, 0)) > 0) {
            return s;
        }
    }
    return 0;
}

// GGML_AXERA_WEIGHTS=mxfp8: the template quantizes the bf16 GGUF weights to MXFP8 at load (about half the bytes per
// token; see docs/backend/AXERA.md for the accuracy cost)
bool axera_weights_mx8() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_AXERA_WEIGHTS");
        return e != nullptr && strcmp(e, "mxfp8") == 0;
    }();
    return on;
}
float axera_rms_eps(const ggml_tensor * rms_norm) {
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    return eps;
}
uint32_t axera_input_size(ggml_axera_kernel * k, const char * name) {
    const int i = k->input(name);
    GGML_ASSERT(i >= 0);
    return k->info->pInputs[i].nSize;
}
uint32_t axera_output_size(ggml_axera_kernel * k, const char * name) {
    const int i = k->output(name);
    GGML_ASSERT(i >= 0);
    return k->info->pOutputs[i].nSize;
}

// the scheduler names its copies of another backend's tensors "<backend>#<name>#<copy>"
std::string axera_base_name(const char * name) {
    const char * a = strchr(name, '#');
    const char * b = a ? strrchr(name, '#') : nullptr;
    return a && b > a ? std::string(a + 1, b) : std::string(name);
}

// nodes [i0, i1) of the graph on the CPU backend (their tensors are host memory: CMM or the CPU's), for the parts of a
// template's graph it does not run itself
void axera_cpu_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph, int i0, int i1, const char * what) {
    axera_cpu_compute_nodes(ctx, cgraph, cgraph->nodes + i0, i1 - i0, what);
}

// the n nodes at nodes (of cgraph, in its order) on the CPU backend
void axera_cpu_compute_nodes(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph, ggml_tensor ** nodes, int n,
                             const char * what) {
    if (n <= 0) {
        return;
    }
    // a slice of small tensors on one thread: waking the CPU backend's threads for each of its nodes costs more than
    // the nodes (the Qwen3-TTS code predictor's slices of one row)
    int64_t big = 0;  // the largest node's work: its elements (times K for a matrix product)
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) && !ggml_is_empty(t) && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE &&
            t->op != GGML_OP_PERMUTE && t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_NONE) {  // (views do no work)
            big = std::max(big, ggml_nelements(t) * (t->op == GGML_OP_MUL_MAT ? t->src[0]->ne[0] : 1));
        }
    }
    const int c = big <= 65536 ? 1 : 0;
    if (!ctx->cpu[c]) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "axera: no CPU backend");
        ctx->cpu[c]             = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_reg_t  reg = ggml_backend_dev_backend_reg(dev);
        const int           nt  = c ? 1 : (int) std::max(1u, std::thread::hardware_concurrency());
        auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_threads) {
            set_threads(ctx->cpu[c], nt);
        }
        // a threadpool of its own: without one the CPU backend starts its threads on every call (templates call it
        // once per slice of a graph, thousands of times per graph for mtmd's generation graphs)
        auto tp_new = (ggml_threadpool_t (*)(struct ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto tp_set = (void (*)(ggml_backend_t, ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        ctx->cpu_tp_free = (void (*)(ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        if (tp_new && tp_set && ctx->cpu_tp_free) {
            struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nt);
            tpp.poll                          = 0;  // no spinning between calls: the NPU and the template's host work run then
            ctx->cpu_tp[c]                    = tp_new(&tpp);
            tp_set(ctx->cpu[c], ctx->cpu_tp[c]);
        }
    }
    // (CPU) the slice of ggml_graph_view (not exported by ggml-base)
    struct ggml_cgraph view = *cgraph;
    view.size               = 0;
    view.n_nodes            = n;
    view.n_leafs            = 0;
    view.nodes              = nodes;
    view.grads              = nullptr;
    view.grad_accs          = nullptr;
    view.leafs              = nullptr;
    view.uid                = 0;
    axera_timer        tm(what);
    if (ggml_backend_graph_compute(ctx->cpu[c], &view) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("axera: %s failed on the CPU backend\n", what);
    }
}
