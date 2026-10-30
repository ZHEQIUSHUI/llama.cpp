// Axera NPU backend core: the NPU runtime, CMM buffers, the host worker pool

#include "runtime.h"

#include "ggml-axera.h"
#include "ggml-backend.h"

#include <fstream>

#include <sys/mman.h>
#include <unistd.h>

std::atomic<uint64_t> axera_graph_epoch{ 0 };

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

void axera_drop_mapped(const ggml_tensor * t) {
    const ggml_tensor * s = t->view_src ? t->view_src : t;
    if (!s->buffer || strcmp(ggml_backend_buffer_name(s->buffer), GGML_AXERA_NAME "_Mapped") != 0 ||
        strncmp(s->name, "token_embd", 10) == 0) {  // (the token embedding: a row read on the CPU per token)
        return;
    }
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t        a  = ((uintptr_t) t->data + pg - 1) & ~(pg - 1);  // whole pages within the tensor
    const uintptr_t        b  = ((uintptr_t) t->data + ggml_nbytes(t)) & ~(pg - 1);
    if (b > a) {
        madvise((void *) a, b - a, MADV_DONTNEED);
    }
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
    static axera_pool pool(std::min(15, std::max(1, (int) std::thread::hardware_concurrency() - 1)));
    return pool;
}
