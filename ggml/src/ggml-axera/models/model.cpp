// Axera NPU backend: the model adapters by manifest model name; a directory with ViT kernels (roles vit.*) also runs
// the mmproj vision encoder's graphs (models/qwen35_vit), one with audio encoder kernels (roles audio.*) the Qwen3-ASR
// audio encoder's (models/qwen3_asr), one with the code predictor's (roles cp.*) the Qwen3-TTS generation graphs
// (models/qwen3_tts)

#include "model.h"

#include <cstdlib>
#include <map>
#include <mutex>

#include <cstring>

namespace axera {
namespace {

// a graph is the vision encoder's if any of its tensors is one of the mmproj's v.* weights.  Clip graphs name the
// layer weights directly; the whole-model ViT adapters (q3vl30, lightnav) read them by name, so check both the node
// names and their sources and accept any layer / patch / position tensor rather than one exact spelling.
bool vit_graph(const ggml_cgraph * g) {
    auto vit_name = [](const char * n) {
        return strstr(n, "v.blk.") || strstr(n, "v.patch_embd") || strstr(n, "v.position_embd");
    };
    for (int i = 0; i < g->n_nodes; i++) {
        if (vit_name(g->nodes[i]->name)) {
            return true;
        }
        for (int s = 0; s < GGML_MAX_SRC && g->nodes[i]->src[s]; ++s) {
            if (vit_name(g->nodes[i]->src[s]->name)) {
                return true;
            }
        }
    }
    return false;
}

// a graph of the llama.cpp model (not one of mtmd's)
bool llm_graph(const ggml_cgraph * g) {
    for (int i = 0; i < g->n_nodes; i++) {
        const char * s = g->nodes[i]->name;
        if (strcmp(s, "attn_norm-0") == 0 || strcmp(s, "result_output") == 0 || strcmp(s, "result_norm") == 0) {
            return true;
        }
    }
    return false;
}

// the language model with its multimodal parts: each graph to the adapter of its kind (vision encoder, audio encoder,
// the language model, else (Qwen3-TTS) a generation graph)
struct composite final : model {
    std::shared_ptr<model> llm, vit, audio, gen;

    // The graphs seen lately and their adapters. llama.cpp reuses a context's graph, and with MTP two contexts take
    // turns every round, so one remembered graph would scan every node's sources again on each call.
    struct seen {
        const ggml_cgraph * g = nullptr;
        int                 n = 0;
        const ggml_tensor * first = nullptr, * last = nullptr;
        model *             m = nullptr;
    };
    seen   recent[4];
    size_t next = 0;

    model * select(const ggml_cgraph * g) {
        if (vit && (vit_graph(g) || minicpm_ocr_vit_graph(g))) {
            return vit.get();
        }
        if (audio && qwen3_asr_graph(g)) {
            return audio.get();
        }
        if (gen && !llm_graph(g)) {
            return gen.get();
        }
        // A package with a whole-model vision encoder and no audio / generation graphs has exactly two kinds of
        // graph: the language model's and the vision encoder's.  Clip may rename or fold the v.* weights, so the
        // name check above can miss it; the absence of the language model's result nodes is unambiguous.
        if (vit && !audio && !gen && !llm_graph(g)) {
            return vit.get();
        }
        return llm.get();
    }

    ggml_status compute(ggml_cgraph * g) override {
        const ggml_tensor * first = g->n_nodes > 0 ? g->nodes[0] : nullptr;
        const ggml_tensor * last  = g->n_nodes > 0 ? g->nodes[g->n_nodes - 1] : nullptr;
        for (const seen & e : recent) {
            if (e.g == g && e.n == g->n_nodes && e.first == first && e.last == last && e.m) {
                return e.m->compute(g);
            }
        }
        seen & e = recent[next++ % (sizeof(recent) / sizeof(recent[0]))];
        e        = {g, g->n_nodes, first, last, select(g)};
        return e.m->compute(g);
    }

    void prewarm(ggml_cgraph * g) override {
        select(g)->prewarm(g);
    }

    size_t state_size(uint32_t flags) const override {
        return llm ? llm->state_size(flags) : 0;
    }

    size_t state_write(void * dst, size_t size, uint32_t flags) override {
        return llm ? llm->state_write(dst, size, flags) : 0;
    }

    size_t state_read(const void * src, size_t size, uint32_t flags) override {
        return llm ? llm->state_read(src, size, flags) : 0;
    }

    int last_state_status() const override {
        return llm ? llm->last_state_status() : 0;
    }

    void state_flush() override {
        if (llm) {
            llm->state_flush();
        }
    }
};

// Every manifest model name: its language-model adapter, the vision encoder for a one-kernel ViT (role vit) and what the
// language-model adapter does in place of llama.cpp (model_caps; asked before the adapter is made, for the KV buffers).
using maker = std::shared_ptr<model> (*)(const manifest &);

std::shared_ptr<model> make_qwen38(const manifest & m) {
    // the 2B / 4B packages run the upstream fused kernels (a decode kernel without decode groups), 27B its own
    return m.has("decode") && !m.has("decode.group") ? make_qwen35_fused(m) : make_qwen38_27b(m);
}

struct adapter_row {
    const char * name;
    maker        llm;
    maker        vit;  // role "vit"; none: such a package is an error
    bool         owns_kv_cache, sched_io_only, causal_cells;
};

constexpr adapter_row adapters[] = {
    { "qwen38_27b",     make_qwen38,         make_qwen35_vit_full,      true,  true,  true  },
    { "qwen35_whole",   make_qwen35_whole,   make_qwen35_vit_full,      true,  false, false },
    // the Qwen3.5-0.8B design with the head merged into its kernels
    { "minicpm_ocr",    make_qwen35_whole,   make_minicpm_ocr_vit_full, false, false, false },
    { "qwen35_moe_sgl", make_qwen35_moe_sgl, make_qwen35_vit_full,      true,  true,  false },
    // the text tower is the published 35B-A3B NVFP4 stack
    { "minicpmv4_7",    make_qwen35_moe_sgl, make_minicpmv4_7_vit_full, true,  true,  false },
    { "hy_mt2",         make_moe_gqa,        nullptr,                   true,  true,  false },
    { "q3vl30",         make_moe_gqa,        make_q3vl30_vit_full,      true,  true,  false },
    { "lightnav",       make_lightnav,      make_lightnav_vit_full,    true,  true,  false },
    { "inkling2",       make_inkling2,       nullptr,                   true,  true,  true  },
    { "qwen3",          make_qwen3,          nullptr,                   false, false, false },
    { "cosyvoice3_llm", make_phase_dense,    nullptr,                   false, false, false },
    { "cpm5",           make_cpm5,           nullptr,                   true,  false, false },
    // the base LM is on the NPU only with its prefill kernel (else llama.cpp runs it on the CPU): see model_caps_of
    { "voxcpm2_base",   make_phase_dense,    nullptr,                   false, false, false },
    { "hymt",           make_hymt,           nullptr,                   true,  false, false },
};

const adapter_row & adapter_of(const manifest & m) {
    for (const adapter_row & r : adapters) {
        if (m.model == r.name) {
            return r;
        }
    }
    GGML_ABORT("axera: %s/manifest.json: no adapter for model %s\n", m.dir.c_str(), m.model.c_str());
}

}  // namespace

static std::shared_ptr<model> make_model_of(const manifest & m) {
    const adapter_row &    row = adapter_of(m);
    std::shared_ptr<model> llm = row.llm(m);
    bool vision = false, vision1 = false, audio = false, gen = false;
    for (const auto & [role, k] : m.kernels) {
        vision  = vision || role.rfind("vit.", 0) == 0;
        vision1 = vision1 || role == "vit";  // the whole vision encoder in one kernel
        audio  = audio || role.rfind("audio.", 0) == 0;
        gen    = gen || role.rfind("cp.", 0) == 0;
    }
    if (!vision && !vision1 && !audio && !gen) {
        return llm;
    }
    if (vision1 && !row.vit) {
        GGML_ABORT("axera: %s/manifest.json: model %s has no one-kernel vision encoder\n", m.dir.c_str(), row.name);
    }
    auto c   = std::make_shared<composite>();
    c->llm   = llm;
    c->vit   = vision1 ? row.vit(m) : vision ? make_qwen35_vit(m) : nullptr;
    c->audio = audio ? make_qwen3_asr(m) : nullptr;
    c->gen   = gen ? make_qwen3_tts(m) : nullptr;
    return c;
}

// one adapter per kernel directory in the process: every backend (llama.cpp's contexts of the model, e.g. a
// speculative draft context next to the target, mtmd's) shares its weights, state and kernels
std::shared_ptr<model> make_model(const manifest & m) {
    static std::mutex                                           mu;
    static std::map<const manifest *, std::weak_ptr<model>> made;
    std::lock_guard<std::mutex>                                 lock(mu);
    std::shared_ptr<model>                                      p = made[&m].lock();
    if (!p) {
        p        = make_model_of(m);
        made[&m] = p;
    }
    return p;
}

model_caps model_caps_of(const manifest & m) {
    const adapter_row & r = adapter_of(m);
    model_caps          c;
    c.owns_kv_cache = r.owns_kv_cache;
    c.sched_io_only = r.sched_io_only;
    c.causal_cells  = r.causal_cells;
    if (m.model == "voxcpm2_base" && m.has("prefill")) {
        c.owns_kv_cache = c.sched_io_only = true;
    }
    return c;
}

bool kq_mask_check() {
    static const bool on = [] {
        const char * e = getenv("GGML_AXERA_KQ_MASK_CHECK");
        return e && atoi(e) != 0;
    }();
    return on;
}

bool mmproj_host_weights(const manifest & m) {
    bool whole = false, other = false;
    for (const auto & [role, k] : m.kernels) {
        whole = whole || role == "vit";
        other = other || role.rfind("vit.", 0) == 0 || role.rfind("audio.", 0) == 0 || role.rfind("cp.", 0) == 0;
    }
    return whole && !other;
}

bool scheduler_llm_graph(const ggml_cgraph * g) {
    return llm_graph(g);
}

}  // namespace axera
