// Axera NPU backend: backend, device and registry (the design: ggml-axera-impl.h)

#include "ggml-axera-impl.h"

#include <unordered_map>

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
    ++axera_graph_epoch;

    if (axera_template_qwen35()) {
        static std::chrono::steady_clock::time_point last;  // profiling: host time between graphs
        if (g_axera_prof.on && last.time_since_epoch().count()) {
            auto & e = g_axera_prof.t["between graphs (host)"];
            e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last).count();
            e.second++;
        }
        const auto t_in = std::chrono::steady_clock::now();
        g_axera_host.on_compute();
        enum ggml_status st;
        {
            axera_timer tm("graph_compute (all)");
            // which template: kept for the same graph (mtmd's and llama.cpp's graphs come again every token / frame,
            // and telling them apart reads every node's sources by name)
            auto & k = ctx->graph_kind;
            if (k.g != cgraph || k.n != cgraph->n_nodes || k.first != (cgraph->n_nodes ? cgraph->nodes[0] : nullptr) ||
                k.last != (cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1] : nullptr)) {
                k.g     = cgraph;
                k.n     = cgraph->n_nodes;
                k.first = cgraph->n_nodes ? cgraph->nodes[0] : nullptr;
                k.last  = cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1] : nullptr;
                k.kind  = axera_vit_graph(cgraph) ? 0 : axera_audio_graph(cgraph) ? 1 : axera_llama_graph(cgraph) ? 2 : 3;
            }
            if (g_axera_prof.on && last.time_since_epoch().count()) {  // the host time before a graph, by its kind
                static const char * before[4] = { "host before vit graph", "host before audio graph",
                                                  "host before llama graph", "host before gen graph" };
                auto & e = g_axera_prof.t[before[k.kind]];
                e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last).count();
                e.second++;
            }
            static const char * kinds[4] = { "graph_compute vit", "graph_compute audio", "graph_compute llama",
                                             "graph_compute gen" };
            axera_timer tk(kinds[k.kind]);
            st = k.kind == 0 ? axera_vit_compute(ctx, cgraph) :
                 k.kind == 1 ? axera_audio_compute(ctx, cgraph) :
                 k.kind == 2 ? axera_q35_compute(ctx, cgraph) :
                               axera_gen_compute(ctx, cgraph);
        }
        static const bool trace = std::getenv("GGML_AXERA_TRACE") != nullptr;  // every graph: kind, nodes, host gap, run
        if (trace) {
            const auto now = std::chrono::steady_clock::now();
            fprintf(stderr, "axera trace: kind %d nodes %d gap %.3f ms run %.3f ms\n", ctx->graph_kind.kind, cgraph->n_nodes,
                    last.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(t_in - last).count() : -1.0,
                    std::chrono::duration<double, std::milli>(now - t_in).count());
        }
        last             = std::chrono::steady_clock::now();
        g_axera_host.end = last;
        return st;
    }

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

// weights left in the GGUF file mapping (llama.cpp mmap, unless --no-mmap) instead of a CMM copy: with a template,
// which converts the weights it runs into CMM itself (MXFP8, bf16, copies) and so never reads the GGUF tensors on the
// NPU. Without one, the bf16 MUL_MAT kernels read the weights in CMM, so they are loaded there. GGML_AXERA_MMAP=0/1
// overrides.
static bool axera_mmap() {
    static const char * e  = getenv("GGML_AXERA_MMAP");
    static const bool   on = e ? atoi(e) != 0 : axera_template_qwen35();
    return on;
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
        /* .buffer_from_host_ptr  = */ axera_mmap(),
        /* .events                = */ false,
        /* .mmap_support          = */ axera_mmap(),
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
    if (op->op == GGML_OP_MUL_MAT && op->src[0] && strncmp(op->src[0]->name, "a.blk.", 6) == 0) {
        axera_audio_on_set(op->src[0]);  // (the audio encoders matrices to bf16 while mtmds warmup reserves its graph)
    }
    if (axera_template_qwen35()) {
        // the template runs the whole graph up to the logits or, for pooled embeddings, the final norm: the pooling
        // stays on the CPU. Weights must still be in CMM (or the mapped GGUF) to be offloaded
        return !axera_pooling_op(op);
    }
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

// the weights left on the CPU (the token embeddings: one row read per token) kept resident when the model is mmap'ed:
// the NPU weights take most of the board's memory, page reclaim evicts the embedding pages, and every token's row
// then faults in from storage (~2 ms per token on a 16 GB board with a 20 GB model). GGML_AXERA_PIN_HOST=0: off
static bool ggml_backend_axera_pin_host_weights(void) {
    const char * e = std::getenv("GGML_AXERA_PIN_HOST");
    return e == nullptr || strcmp(e, "0") != 0;
}

static void * ggml_backend_axera_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_pin_host_weights") == 0) {
        return (void *) ggml_backend_axera_pin_host_weights;
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

