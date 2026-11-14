// Axera NPU backend: the model adapters by manifest model name; a directory with ViT kernels (roles vit.*) also runs
// the mmproj vision encoder's graphs (models/qwen35_vit), one with audio encoder kernels (roles audio.*) the Qwen3-ASR
// audio encoder's (models/qwen3_asr), one with the code predictor's (roles cp.*) the Qwen3-TTS generation graphs
// (models/qwen3_tts)

#include "model.h"

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
    const ggml_cgraph *    last = nullptr;  // the last graph and its kind (llama.cpp reuses the decode graph)
    int                    n = 0;
    model *                cur = nullptr;

    ggml_status compute(ggml_cgraph * g) override {
        if (g != last || g->n_nodes != n) {
            last = g;
            n    = g->n_nodes;
            cur  = vit && (vit_graph(g) || minicpm_ocr_vit_graph(g)) ? vit.get() :
                   audio && qwen3_asr_graph(g)         ? audio.get() :
                   gen && !llm_graph(g)                ? gen.get() :
                                                         llm.get();
        }
        return cur->compute(g);
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
};

}  // namespace

static std::shared_ptr<model> make_model_of(const manifest & m) {
    std::shared_ptr<model> llm;
    if (m.model == "qwen35_moe") {
        llm = make_qwen35_moe(m);
    } else if (m.model == "qwen35_dense") {
        llm = make_qwen35_dense(m);
    } else if (m.model == "qwen38_27b") {
        llm = make_qwen38_27b(m);
    } else if (m.model == "qwen35_whole") {
        llm = make_qwen35_whole(m);
    } else if (m.model == "minicpm_ocr") {
        llm = make_minicpm_ocr_text(m);
    } else if (m.model == "qwen35_35b_a3b") {
        llm = make_qwen35_35b_a3b(m);
    } else if (m.model == "qwen35_moe_sgl") {
        llm = make_qwen35_moe_sgl(m);
    } else if (m.model == "qwen3") {
        llm = make_qwen3(m);
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
    c->vit   = vision1 ? (m.model == "minicpm_ocr" ? make_minicpm_ocr_vit_full(m) : make_qwen35_vit_full(m)) :
               vision ? make_qwen35_vit(m) : nullptr;
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

}  // namespace axera
