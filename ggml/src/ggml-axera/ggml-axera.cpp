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
// GGML_AXERA_DEBUG=1 logs every missing kernel once.

#include "ggml-axera.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <dirent.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
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
    std::map<int64_t, std::set<int64_t>>                      mul_mat;  // K -> available weight-row counts C
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
            }
            closedir(d);
        }
    }

    std::string path(const std::string & name) const { return dir + "/" + name + ".axmodel"; }

    // weight rows per call for a [N, K] weight: N itself, else the largest available C dividing N; 0 if none
    int64_t mul_mat_chunk(int64_t n, int64_t k) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = mul_mat.find(k);
        if (it != mul_mat.end()) {
            for (auto c = it->second.rbegin(); c != it->second.rend(); ++c) {
                if (*c <= n && n % *c == 0) {
                    return *c;
                }
            }
        }
        char key[64];
        snprintf(key, sizeof(key), "n%" PRId64 "_k%" PRId64, n, k);
        if (axera_debug() && reported.insert(key).second) {
            GGML_LOG_INFO("%s: no kernel mul_mat_bf16_%s in %s\n", __func__, key, dir.c_str());
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

static std::string axera_mul_mat_kernel_name(int64_t c, int64_t k) {
    char name[96];
    snprintf(name, sizeof(name), "mul_mat_bf16_n%" PRId64 "_k%" PRId64, c, k);
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
    // NPU outputs are invalidated right after each run, so the cached view is current
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
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

struct ggml_backend_axera_context {
    // bf16 activation staging in CMM
    ggml_backend_buffer_t staging = nullptr;

    ~ggml_backend_axera_context() {
        if (staging) {
            ggml_backend_buffer_free(staging);
        }
    }

    ggml_backend_axera_buffer_context * staging_for(size_t size) {
        if (!staging || ggml_backend_buffer_get_size(staging) < size) {
            if (staging) {
                ggml_backend_buffer_free(staging);
            }
            staging = ggml_backend_buft_alloc_buffer(ggml_backend_axera_buffer_type(), size);
            GGML_ASSERT(staging != nullptr);
        }
        return (ggml_backend_axera_buffer_context *) staging->context;
    }
};

static AX_ENGINE_IO_BUFFER_T axera_io(const ggml_backend_axera_buffer_context * ctx, void * p, uint32_t size) {
    AX_ENGINE_IO_BUFFER_T b;
    memset(&b, 0, sizeof(b));
    b.phyAddr  = ctx->phy_of(p);
    b.pVirAddr = p;
    b.nSize    = size;
    return b;
}

static void ggml_backend_axera_mul_mat(ggml_backend_axera_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t       K    = src0->ne[0];
    const int64_t       N    = src0->ne[1];
    const int64_t       M    = src1->ne[1];
    const int64_t       C    = axera_kernels().mul_mat_chunk(N, K);
    GGML_ASSERT(C > 0);

    ggml_axera_kernel * k  = axera_kernels().get(axera_mul_mat_kernel_name(C, K));
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    GGML_ASSERT(k->info->pInputs[ix].nSize == K * sizeof(ggml_bf16_t) &&
                k->info->pInputs[iw].nSize == C * src0->nb[1] && k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(GGML_PAD(K * sizeof(ggml_bf16_t), AXERA_ALIGNMENT));
    auto * xb   = (ggml_bf16_t *) st->vir;
    auto * wctx = (ggml_backend_axera_buffer_context *) src0->buffer->context;
    auto * dctx = (ggml_backend_axera_buffer_context *) dst->buffer->context;
    dctx->flush(dst->data, ggml_nbytes(dst));

    AX_ENGINE_IO_BUFFER_T inputs[2], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, xb, K * sizeof(ggml_bf16_t));

    for (int64_t m = 0; m < M; ++m) {
        // activation row f32 -> bf16 into staging
        ggml_fp32_to_bf16_row((const float *) ((const char *) src1->data + m * src1->nb[1]), xb, K);
        st->flush(xb, K * sizeof(ggml_bf16_t));
        for (int64_t c0 = 0; c0 < N; c0 += C) {
            inputs[iw]    = axera_io(wctx, (char *) src0->data + c0 * src0->nb[1], C * src0->nb[1]);
            outputs[io]   = axera_io(dctx, (char *) dst->data + m * dst->nb[1] + c0 * sizeof(float), C * sizeof(float));
            const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
            if (ret != 0) {
                GGML_ABORT("%s: RunSync failed (0x%x) for %s, n=%" PRId64 " k=%" PRId64 " row %" PRId64 "\n", __func__,
                           ret, dst->name, N, K, m);
            }
        }
    }
    dctx->invalidate(dst->data, ggml_nbytes(dst));
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
