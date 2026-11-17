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
    // The state machine's result for the graph that just failed (axera::seq_status; 0: no miss). llama.cpp turns a
    // seq miss into a retryable decode result so the server can discard the stale prompt and replay from cell 0.
    virtual int last_state_status() const { return 0; }
    // the whole graph (llama.cpp's model graph: up to the logits); a graph the model does not run fails
    virtual ggml_status compute(ggml_cgraph * g) = 0;
    // one-time host work for a graph that a later compute will run (llama.cpp can hand the next graph over before
    // its first token, e.g. a decode graph built while the prompt's call is still open: no output is produced here)
    virtual void prewarm(ggml_cgraph * g) { GGML_UNUSED(g); }
};

// the adapter of the manifest's model; aborts for a model this backend has no adapter for
std::shared_ptr<model> make_model(const manifest & m);
// What llama.cpp may assume about the manifest's language-model adapter, known without making it (an adapter's
// constructor may allocate CMM):
//   owns_kv_cache  it keeps the KV cache in its own CMM pools, llama.cpp's cache tensors are unused (lazy host memory)
//   sched_io_only  of a language-model graph it reads the inputs and writes the outputs only; every other tensor the
//                  scheduler plans is unused and can live in lazy host memory instead of CMM
struct model_caps {
    bool owns_kv_cache = false;
    bool sched_io_only = false;
};
model_caps model_caps_of(const manifest & m);
// the manifest's vision adapter copies the mmproj weights its kernel reads into its own CMM (one kernel for the whole
// encoder): mtmd may then keep the mmproj as a file mapping. Adapters that bind weight tensors in place need CMM.
bool mmproj_host_weights(const manifest & m);
// a graph of the llama.cpp language model (not mtmd's vision, audio or generation graphs)
bool scheduler_llm_graph(const ggml_cgraph * g);

std::shared_ptr<model> make_qwen35_moe(const manifest & m);
std::shared_ptr<model> make_qwen35_dense(const manifest & m);
std::shared_ptr<model> make_qwen38_27b(const manifest & m);  // Qwen3.8-27B: the whole-model prefill kernel
std::shared_ptr<model> make_qwen35_whole(const manifest & m);  // Qwen3.5 dense: the whole-model prefill kernel
std::shared_ptr<model> make_qwen35_35b_a3b(const manifest & m);  // Qwen3.5-35B-A3B: one decode kernel, per-layer prefill
std::shared_ptr<model> make_qwen35_moe_sgl(const manifest & m);  // Qwen3.5/3.6-35B-A3B with the sgl NVFP4 kernels
std::shared_ptr<model> make_hy_mt2(const manifest & m);  // Hy-MT2-30B-A3B: one prefill page and one decode row
std::shared_ptr<model> make_qwen3(const manifest & m);  // Qwen3 text decoders (Embedding, ASR text, TTS talker)
std::shared_ptr<model> make_cpm5(const manifest & m);  // MiniCPM5-2B: plain 42-layer Llama, split-KV whole backbone
std::shared_ptr<model> make_hymt(const manifest & m);  // Hy-MT2-1.8B: HunYuan dense 32-layer Llama, split-KV whole backbone
std::shared_ptr<model> make_qwen35_vit(const manifest & m);  // the mmproj vision encoder (roles vit.*)
std::shared_ptr<model> make_qwen35_vit_full(const manifest & m);  // the same in one call (role vit)
std::shared_ptr<model> make_minicpm_ocr_vit_full(const manifest & m);  // MiniCPM-OCR SigLIP tower (role vit)
std::shared_ptr<model> make_minicpmv4_7_vit_full(const manifest & m);  // MiniCPM-V 4.7 SigLIP + window merger (role vit)
std::shared_ptr<model> make_minicpm_ocr_text(const manifest & m);  // MiniCPM-OCR text decoder (core/seqstate)
bool minicpm_ocr_vit_graph(const ggml_cgraph * g);
std::shared_ptr<model> make_qwen3_asr(const manifest & m);  // the Qwen3-ASR audio encoder (roles audio.*)
// the Qwen3-TTS generation graphs: code predictor frames, codec decoder, speaker encoder (roles cp.*, gen.*)
std::shared_ptr<model> make_qwen3_tts(const manifest & m);
// the VoxCPM2 BaseLM: one 256-row prefill and one decode AXModel, paged BF16 KV under core/seqstate
std::shared_ptr<model> make_voxcpm2_base(const manifest & m);
// a graph is the Qwen3-ASR audio encoder's: it reads the conv front-end's output projection and the layers (the
// Qwen3-TTS speaker encoder has a conv_out too)
bool qwen3_asr_graph(const ggml_cgraph * g);

}  // namespace axera
