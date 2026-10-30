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
std::shared_ptr<model> make_qwen38_27b(const manifest & m);  // Qwen3.8-27B: the whole-model prefill kernel
std::shared_ptr<model> make_qwen3(const manifest & m);  // Qwen3 text decoders (Embedding, ASR text, TTS talker)
std::shared_ptr<model> make_qwen35_vit(const manifest & m);  // the mmproj vision encoder (roles vit.*)
std::shared_ptr<model> make_qwen3_asr(const manifest & m);  // the Qwen3-ASR audio encoder (roles audio.*)
// the Qwen3-TTS generation graphs: code predictor frames, codec decoder, speaker encoder (roles cp.*, gen.*)
std::shared_ptr<model> make_qwen3_tts(const manifest & m);
// a graph is the Qwen3-ASR audio encoder's: it reads the conv front-end's output projection and the layers (the
// Qwen3-TTS speaker encoder has a conv_out too)
bool qwen3_asr_graph(const ggml_cgraph * g);
// a tensor the model loader set (or the scheduler asks about): an audio encoder matrix converted to bf16 now, so that
// the first request does not wait for it
void qwen3_asr_on_set(const ggml_tensor * t);

}  // namespace axera
