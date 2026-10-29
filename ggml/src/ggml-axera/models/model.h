// Axera NPU backend: the model adapters. A kernel directory's manifest names its model; that model's adapter runs
// every graph of the process with the directory's kernels (models/<model>/). Adapters depend on core/ and ops/ only,
// never on each other.
#pragma once

#include "../core/manifest.h"

#include "ggml-backend.h"

#include <memory>

namespace axera {

struct model {
    virtual ~model() = default;
    // the whole graph (llama.cpp's model graph: up to the logits); a graph the model does not run fails
    virtual ggml_status compute(ggml_cgraph * g) = 0;
};

// the adapter of the manifest's model; aborts for a model this backend has no adapter for
std::shared_ptr<model> make_model(const manifest & m);

std::shared_ptr<model> make_qwen35_moe(const manifest & m);
std::shared_ptr<model> make_qwen35_dense(const manifest & m);

}  // namespace axera
