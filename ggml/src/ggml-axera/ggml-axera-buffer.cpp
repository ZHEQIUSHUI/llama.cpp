// Axera NPU backend: CMM buffers and buffer type

#include "ggml-axera-impl.h"
#include "core/manifest.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sys/mman.h>

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
    ctx->clean_n = 0;
    memcpy((char *) tensor->data + offset, data, size);
    ctx->flush((char *) tensor->data + offset, size);
}

static void ggml_backend_axera_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                 size_t offset, size_t size) {
    // the range may hold NPU results not invalidated yet (adapter state read for a checkpoint) or CPU writes not
    // flushed yet (CPU backend ops on CMM): write back, then drop the cached lines, then read; but not the logits the
    // NPU wrote this graph, already invalidated
    auto * ctx = (ggml_backend_axera_buffer_context *) buffer->context;
    char * p   = (char *) tensor->data + offset;
    axera_timer tm("get_tensor");
    if (!(ctx->clean_epoch == axera_graph_epoch && p >= (const char *) ctx->clean_p &&
          p + size <= (const char *) ctx->clean_p + ctx->clean_n)) {
        ctx->flush(p, size);
        ctx->invalidate(p, size);
    }
    if (size < (512u << 10)) {
        memcpy(data, p, size);
        return;
    }
    // (a verification's logits, 4 x 1 MB: one core copies at ~2 GB/s) in parts on the host workers
    const size_t part = (size / 8 + 63) & ~(size_t) 63;
    axera_host_pool().run(8, [&](int64_t i) {
        const size_t b = std::min(size, (size_t) i * part), e = std::min(size, b + part);
        memcpy((char *) data + b, p + b, e - b);
    });
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

// the llama.cpp KV / recurrent-state caches of an adapter that keeps the model's state in its own CMM pools (the
// manifests' models qwen38_27b, qwen35_whole and qwen35_moe_sgl): nothing reads or writes them, so they live in lazily-backed host
// memory (anonymous mmap: untouched pages cost nothing; clear() is a no-op, the pages are zero), not in CMM (16 GB at
// a 256k context)
struct ggml_backend_axera_lazy_context {
    void * base = nullptr;
    size_t size = 0;
};

static void ggml_backend_axera_lazy_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_axera_lazy_context *) buffer->context;
    if (ctx->base) {
        munmap(ctx->base, ctx->size);
    }
    delete ctx;
}

static void * ggml_backend_axera_lazy_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_axera_lazy_context *) buffer->context)->base;
}

static void ggml_backend_axera_lazy_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                  size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_lazy_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_lazy_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                               size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_lazy_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (value != 0) {  // (a fresh anonymous mapping is zero already)
        auto * ctx = (ggml_backend_axera_lazy_context *) buffer->context;
        memset(ctx->base, value, ctx->size);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_axera_lazy_buffer_i = {
    /* .free_buffer     = */ ggml_backend_axera_lazy_free_buffer,
    /* .get_base        = */ ggml_backend_axera_lazy_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_axera_lazy_memset_tensor,
    /* .set_tensor      = */ ggml_backend_axera_lazy_set_tensor,
    /* .get_tensor      = */ ggml_backend_axera_lazy_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_axera_lazy_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_axera_lazy_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_AXERA_NAME "_KV";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_axera_lazy_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ggml_backend_axera_lazy_context;
    ctx->size  = std::max<size_t>(size, 4096);
    ctx->base  = mmap(nullptr, ctx->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ctx->base == MAP_FAILED) {
        GGML_LOG_ERROR("%s: failed to map %zu bytes\n", __func__, size);
        delete ctx;
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_axera_lazy_buffer_i, ctx, size);
}

static size_t ggml_backend_axera_lazy_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
    GGML_UNUSED(buft);
}

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
bool ggml_backend_buffer_is_axera(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft == ggml_backend_axera_buffer_type();
}

// the adapters' attention kernels address a KV cache tensor (cache_k_l* / cache_v_l*) as one IO of the manifest's
// context: a cache of fewer cells gets room up to that context after its last stream
static size_t ggml_backend_axera_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    const size_t n  = ggml_nbytes(t);
    const bool   kv = t->type == GGML_TYPE_F16 && t->ne[3] == 1 &&
                    (strncmp(t->name, "cache_k_l", 9) == 0 || strncmp(t->name, "cache_v_l", 9) == 0);
    const axera::manifest * m = axera::kernels();
    if (kv) {
        axera_note_kv_cells(t->ne[1]);  // (the context llama.cpp runs with: the adapters size their pools to it)
    }
    return kv && m ? std::max(n, (size_t) (t->ne[2] - 1) * t->nb[2] + (size_t) m->context * t->nb[1]) : n;
    GGML_UNUSED(buft);
}
ggml_backend_buffer_type_t ggml_backend_axera_lazy_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_axera_lazy_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_axera_lazy_type_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ ggml_backend_axera_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_axera_lazy_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_axera_buffer_type_get_alloc_size,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ ggml_backend_axera_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_axera_reg(), 0),
        /* .context = */ NULL,
    };
    return &buft;
}

// the buffer type of the llama.cpp caches: lazy host memory for an adapter that does not use them, else CMM
ggml_backend_buffer_type_t ggml_backend_axera_kv_buffer_type(void) {
    const axera::manifest * m = axera::kernels();
    return m && (m->model == "qwen38_27b" || m->model == "qwen35_whole" || m->model == "qwen35_moe_sgl" ||
                    m->model == "hymt") ? ggml_backend_axera_lazy_buffer_type() :
                                                                             ggml_backend_axera_buffer_type();
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

// mapped buffers: the GGUF file mapping (llama.cpp mmap) wrapped as weights of this device, so the graph stays whole.
// Not CMM: the NPU cannot read them, and ggml_backend_buffer_is_axera is false for them, so the adapters take these
// weights the way they take weights kept on the CPU (converted, or copied into CMM once). The mapping is read only.

struct ggml_backend_axera_mapped_context {
    void * base;
    size_t size;
};

static void ggml_backend_axera_mapped_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_axera_mapped_context *) buffer->context;
}

static void * ggml_backend_axera_mapped_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_axera_mapped_context *) buffer->context)->base;
}

static void ggml_backend_axera_mapped_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                    size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_mapped_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                 size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_mapped_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                 size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_axera_mapped_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_axera_mapped_context *) buffer->context;
    memset(ctx->base, value, ctx->size);
}

static const struct ggml_backend_buffer_i ggml_backend_axera_mapped_i = {
    /* .free_buffer     = */ ggml_backend_axera_mapped_free_buffer,
    /* .get_base        = */ ggml_backend_axera_mapped_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_axera_mapped_memset_tensor,
    /* .set_tensor      = */ ggml_backend_axera_mapped_set_tensor,
    /* .get_tensor      = */ ggml_backend_axera_mapped_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_axera_mapped_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_axera_mapped_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_AXERA_NAME "_Mapped";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_axera_mapped_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_LOG_ERROR("%s: mapped buffers wrap a file mapping, they are not allocated (%zu bytes)\n", __func__, size);
    return nullptr;
    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_axera_mapped_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_axera_mapped_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_axera_mapped_type_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ ggml_backend_axera_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ ggml_backend_axera_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_axera_reg(), 0),
        /* .context = */ NULL,
    };
    return &buft;
}

ggml_backend_buffer_t axera_mapped_buffer(void * ptr, size_t size) {
    auto * ctx = new ggml_backend_axera_mapped_context{ ptr, size };
    return ggml_backend_buffer_init(ggml_backend_axera_mapped_buffer_type(), ggml_backend_axera_mapped_i, ctx, size);
}
