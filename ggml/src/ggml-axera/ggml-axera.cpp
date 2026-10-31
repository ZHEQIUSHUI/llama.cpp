// Axera NPU backend: backend, device and registry (the design: ggml-axera-impl.h)

#include "ggml-axera-impl.h"
#include "models/model.h"

#include <atomic>
#include <unordered_map>

static const char * ggml_backend_axera_get_name(ggml_backend_t backend) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_axera_free(ggml_backend_t backend) {
    delete (ggml_backend_axera_context *) backend->context;
    delete backend;
}

// the whole graph on the kernel directory's model adapter
static enum ggml_status ggml_backend_axera_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_axera_context *) backend->context;
    ++axera_graph_epoch;
    static std::chrono::steady_clock::time_point last;
    const auto                                   t0 = std::chrono::steady_clock::now();
    if (!ctx->model) {
        ctx->model = axera::make_model(*axera::kernels());
    }
    enum ggml_status st;
    {
        axera_timer tm("graph_compute");
        st = ctx->model->compute(cgraph);
    }
    static const bool trace = std::getenv("GGML_AXERA_TRACE") != nullptr;  // every graph: nodes, host gap, run
    if (trace) {
        const auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "axera trace: nodes %d gap %.3f ms run %.3f ms\n", cgraph->n_nodes,
                last.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(t0 - last).count() : -1.0,
                std::chrono::duration<double, std::milli>(now - t0).count());
    }
    last = std::chrono::steady_clock::now();
    return st;
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

static ggml_backend_buffer_t ggml_backend_axera_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr,
                                                                            size_t size, size_t max_tensor_size) {
    return axera_mapped_buffer(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
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
        /* .buffer_from_host_ptr  = */ true,  // the weights stay in the GGUF file mapping: the adapters convert what
        /* .events                = */ false,  // their kernels read into CMM
        /* .mmap_support          = */ true,
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

// the pooling inputs of an embeddings graph: the rows to pick (cls) or average (mean)
static bool axera_pooling_input(const ggml_tensor * t) {
    for (const char * n : { "inp_cls", "cls", "inp_mean", "mean" }) {
        if (strcmp(t->name, n) == 0) {
            return true;
        }
    }
    return false;
}

// a pooling op of an embeddings graph: after the final norm (result_norm), on the rows picked / averaged by the
// pooling inputs, up to the pooled output (and a rerank head): within 6 ops of a pooled result or a pooling input
static bool axera_pooling_walk(const ggml_tensor * op, int depth) {
    if (!op || depth > 6) {
        return false;
    }
    // (without pooling result_embd_pooled is the final norm itself, run by the template)
    const bool final_norm = op->op == GGML_OP_MUL && op->src[0] && op->src[0]->op == GGML_OP_RMS_NORM;
    if ((strcmp(op->name, "result_embd_pooled") == 0 && !final_norm) || strncmp(op->name, "result_norm (", 13) == 0) {
        return true;
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = op->src[s];
        if (src && ((src->op == GGML_OP_NONE && axera_pooling_input(src)) ||
                    (src->op != GGML_OP_NONE && axera_pooling_walk(src, depth + 1)))) {
            return true;
        }
    }
    return false;
}

// the walk's answer kept per node: walked from every node of a 1929-node graph it took 8 ms of the scheduler's split
// per batch. A node is walked again when its op or name changed (llama.cpp rebuilds the graph per batch, mostly at the
// same addresses with the same structure). Only the answer is kept, not distances: the walk stays six levels deep (a
// memoized walk over all sources did not end on mtmd's Qwen3-TTS generation graphs)
static bool axera_pooling_op(const ggml_tensor * op) {
    struct entry {
        ggml_op op;
        char    name[GGML_MAX_NAME];
        bool    pooling;
    };
    static std::unordered_map<const ggml_tensor *, entry> memo;  // the scheduler asks from one thread
    if (!op) {
        return false;
    }
    auto it = memo.find(op);
    if (it != memo.end() && it->second.op == op->op && strncmp(it->second.name, op->name, GGML_MAX_NAME) == 0) {
        return it->second.pooling;
    }
    entry & e = memo[op];
    e.op      = op->op;
    strncpy(e.name, op->name, GGML_MAX_NAME);
    e.pooling = axera_pooling_walk(op, 0);
    return e.pooling;
}

static bool ggml_backend_axera_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // the model adapter runs the whole graph up to the logits or, for pooled embeddings, the final norm: the pooling
    // stays on the CPU
    return !axera_pooling_op(op);
    GGML_UNUSED(dev);
}

static bool ggml_backend_axera_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // NPU kernels need physical addresses: CMM buffers, and the GGUF mapping whose weights the templates convert into
    // CMM. Activations from other host buffers are copied in by the
    // scheduler; the CPU backend in turn uses CMM buffers directly (they are host buffers).
    return buft == ggml_backend_axera_buffer_type() || buft == ggml_backend_axera_mapped_buffer_type();
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
    /* .buffer_from_host_ptr = */ ggml_backend_axera_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_axera_device_supports_op,
    /* .supports_buft        = */ ggml_backend_axera_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// reg
//

static const char * ggml_backend_axera_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_AXERA_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_axera_reg_get_device_count(ggml_backend_reg_t reg) {
    // the manifest's kernels loaded before the model's tensors; no manifest, no device
    if (!axera_runtime_init() || !axera::kernels()) {
        return 0;
    }
    return 1;
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

// the weights left on the CPU (the token embeddings: one row read per token) kept resident when the model is mmap'ed:
// the NPU weights take most of the board's memory, page reclaim evicts the embedding pages, and every token's row
// then faults in from storage (~2 ms per token on a 16 GB board with a 20 GB model)
static bool ggml_backend_axera_pin_host_weights(void) {
    return true;
}

// llama.cpp tells each context's sequence count: the largest is kept (a padded prefill chunk needs one sequence)
static std::atomic<uint32_t> g_axera_n_seq_max{ 0 };

static void ggml_backend_axera_set_n_seq_max(uint32_t n) {
    uint32_t cur = g_axera_n_seq_max.load();
    while (n > cur && !g_axera_n_seq_max.compare_exchange_weak(cur, n)) {
    }
}

uint32_t axera_n_seq_max() {
    return g_axera_n_seq_max.load();
}

static void * ggml_backend_axera_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_pin_host_weights") == 0) {
        return (void *) ggml_backend_axera_pin_host_weights;
    }
    if (strcmp(name, "ggml_backend_set_n_seq_max") == 0) {
        return (void *) ggml_backend_axera_set_n_seq_max;
    }
    return NULL;
    GGML_UNUSED(reg);
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

