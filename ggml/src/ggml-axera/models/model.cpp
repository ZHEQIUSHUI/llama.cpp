// Axera NPU backend: the model adapters by manifest model name; a directory with ViT kernels (roles vit.*) also runs
// the mmproj vision encoder's graphs (models/qwen35_vit)

#include "model.h"

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

struct with_vision final : model {
    std::shared_ptr<model> llm, vit;
    const ggml_cgraph *    last = nullptr;  // the last graph and its kind (llama.cpp reuses the decode graph)
    int                    n = 0;
    bool                   is_vit = false;

    ggml_status compute(ggml_cgraph * g) override {
        if (g != last || g->n_nodes != n) {
            last   = g;
            n      = g->n_nodes;
            is_vit = vit_graph(g);
        }
        return (is_vit ? vit : llm)->compute(g);
    }
};

}  // namespace

std::shared_ptr<model> make_model(const manifest & m) {
    std::shared_ptr<model> llm;
    if (m.model == "qwen35_moe") {
        llm = make_qwen35_moe(m);
    } else if (m.model == "qwen35_dense") {
        llm = make_qwen35_dense(m);
    } else {
        GGML_ABORT("axera: %s/manifest.json: no adapter for model %s\n", m.dir.c_str(), m.model.c_str());
    }
    bool vision = false;
    for (const auto & [role, k] : m.kernels) {
        vision = vision || role.rfind("vit.", 0) == 0;
    }
    if (!vision) {
        return llm;
    }
    auto v = std::make_shared<with_vision>();
    v->llm = llm;
    v->vit = make_qwen35_vit(m);
    return v;
}

}  // namespace axera
