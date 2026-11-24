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

// a graph is the vision encoder's if it reads the first ViT layer's qkv weight
bool vit_graph(const ggml_cgraph * g) {
    for (int i = 0; i < g->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && g->nodes[i]->src[s]; ++s) {
            const char * n = g->nodes[i]->src[s]->name;
            const char * a = strchr(n, '#');  // the scheduler's copies "<backend>#<name>#<copy>"
            if (strncmp(a ? a + 1 : n, "v.blk.0.attn_qkv.weight", 23) == 0) {
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
        return vit && (vit_graph(g) || minicpm_ocr_vit_graph(g)) ? vit.get() :
               audio && qwen3_asr_graph(g)         ? audio.get() :
               gen && !llm_graph(g)                ? gen.get() :
                                                     llm.get();
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

}  // namespace

static std::shared_ptr<model> make_model_of(const manifest & m) {
    std::shared_ptr<model> llm;
    if (m.model == "qwen38_27b") {
        llm = make_qwen38_27b(m);
    } else if (m.model == "qwen35_whole") {
        llm = make_qwen35_whole(m);
    } else if (m.model == "minicpm_ocr") {
        llm = make_minicpm_ocr_text(m);
    } else if (m.model == "qwen35_moe_sgl") {
        llm = make_qwen35_moe_sgl(m);
    } else if (m.model == "minicpmv4_7") {
        llm = make_qwen35_moe_sgl(m);  // the text tower is the published 35B-A3B NVFP4 stack
    } else if (m.model == "hy_mt2") {
        llm = make_hy_mt2(m);
    } else if (m.model == "q3vl30") {
        llm = make_q3vl30(m);
    } else if (m.model == "qwen3") {
        llm = make_qwen3(m);
    } else if (m.model == "cosyvoice3_llm") {
        llm = make_cosyvoice3_llm(m);
    } else if (m.model == "cpm5") {
        llm = make_cpm5(m);
    } else if (m.model == "voxcpm2_base") {
        llm = make_voxcpm2_base(m);
    } else if (m.model == "hymt") {
        llm = make_hymt(m);
    } else {
        GGML_ABORT("axera: %s/manifest.json: no adapter for model %s\n", m.dir.c_str(), m.model.c_str());
    }
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
    auto c   = std::make_shared<composite>();
    c->llm   = llm;
    c->vit   = vision1 ? (m.model == "minicpm_ocr" ? make_minicpm_ocr_vit_full(m) :
                          m.model == "minicpmv4_7" ? make_minicpmv4_7_vit_full(m) : make_qwen35_vit_full(m)) :
               vision ? make_qwen35_vit(m) : nullptr;
    if (vision1 && m.model == "q3vl30") {
        c->vit = make_q3vl30_vit_full(m);
    }
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

// (next to the factory above: the same model names)
model_caps model_caps_of(const manifest & m) {
    model_caps c;
    const std::string & n = m.model;
    c.owns_kv_cache = n == "qwen38_27b" || n == "qwen35_whole" || n == "qwen35_moe_sgl" || n == "minicpmv4_7" ||
                      n == "cpm5" || n == "hymt" || n == "hy_mt2" || n == "q3vl30";
    c.sched_io_only = n == "qwen38_27b" || n == "qwen35_moe_sgl" || n == "minicpmv4_7" || n == "hy_mt2" ||
                      n == "q3vl30";
    c.causal_cells  = n == "qwen38_27b";
    // VoxCPM2's base LM: on the NPU only when the manifest carries its prefill kernel (else llama.cpp runs it on the CPU)
    if (n == "voxcpm2_base" && m.has("prefill")) {
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
