// Axera NPU backend: the model adapters by manifest model name

#include "model.h"

namespace axera {

std::shared_ptr<model> make_model(const manifest & m) {
    if (m.model == "qwen35_moe") {
        return make_qwen35_moe(m);
    }
    if (m.model == "qwen35_dense") {
        return make_qwen35_dense(m);
    }
    GGML_ABORT("axera: %s/manifest.json: no adapter for model %s\n", m.dir.c_str(), m.model.c_str());
}

}  // namespace axera
