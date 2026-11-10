// Axera NPU backend: the model adapters. A kernel directory's manifest names its model; that model's adapter runs
// every graph of the process with the directory's kernels (models/<model>/). Adapters depend on core/ and ops/ only,
// never on each other.
#pragma once

#include "../core/manifest.h"

#include "ggml-backend.h"

#include <cstddef>
#include <memory>

namespace axera {

// llama.cpp's LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY. ggml does not include llama.h, so the value is repeated here.
constexpr uint32_t state_flags_partial_only = 1;

struct model {
    virtual ~model() = default;
    // llama.cpp's sequence state (context checkpoints / prompt cache). An adapter whose caches are private CMM pools
    // overrides these: the backend appends the blob to the state stream, so a checkpoint restore puts the state the
    // kernels actually read back. The default is no state, which is also the CPU-backend behaviour.
    virtual size_t state_size(uint32_t flags) const { (void) flags; return 0; }
    virtual size_t state_write(void * dst, size_t size, uint32_t flags) { (void) dst; (void) size; (void) flags; return 0; }
    virtual size_t state_read(const void * src, size_t size, uint32_t flags) { (void) src; (void) size; (void) flags; return 0; }
    // the whole graph (llama.cpp's model graph: up to the logits); a graph the model does not run fails
    virtual ggml_status compute(ggml_cgraph * g) = 0;
    // one-time host work for a graph that a later compute will run (llama.cpp can hand the next graph over before
    // its first token, e.g. a decode graph built while the prompt's call is still open: no output is produced here)
    virtual void prewarm(ggml_cgraph * g) { GGML_UNUSED(g); }
};

// the adapter of the manifest's model; aborts for a model this backend has no adapter for
std::shared_ptr<model> make_model(const manifest & m);

std::shared_ptr<model> make_qwen35_moe(const manifest & m);
std::shared_ptr<model> make_qwen35_dense(const manifest & m);
std::shared_ptr<model> make_qwen38_27b(const manifest & m);  // Qwen3.8-27B: the whole-model prefill kernel
std::shared_ptr<model> make_dense_blocks(const manifest & m);  // dense fused 4-layer blocks, one kernel call per block
std::shared_ptr<model> make_qwen35_whole(const manifest & m);  // Qwen3.5 dense: the whole-model prefill kernel
std::shared_ptr<model> make_qwen35_35b_a3b(const manifest & m);  // Qwen3.5-35B-A3B: one decode kernel, per-layer prefill
std::shared_ptr<model> make_qwen35_moe_sgl(const manifest & m);  // Qwen3.5/3.6-35B-A3B with the sgl NVFP4 kernels
std::shared_ptr<model> make_qwen3(const manifest & m);  // Qwen3 text decoders (Embedding, ASR text, TTS talker)
std::shared_ptr<model> make_qwen35_vit(const manifest & m);  // the mmproj vision encoder (roles vit.*)
std::shared_ptr<model> make_qwen35_vit_full(const manifest & m);  // the same in one call (role vit)
std::shared_ptr<model> make_qwen3_asr(const manifest & m);  // the Qwen3-ASR audio encoder (roles audio.*)
// the Qwen3-TTS generation graphs: code predictor frames, codec decoder, speaker encoder (roles cp.*, gen.*)
std::shared_ptr<model> make_qwen3_tts(const manifest & m);
// a graph is the Qwen3-ASR audio encoder's: it reads the conv front-end's output projection and the layers (the
// Qwen3-TTS speaker encoder has a conv_out too)
bool qwen3_asr_graph(const ggml_cgraph * g);

}  // namespace axera
