// Axera NPU backend: VoxCPM2 acoustic stages (LocDiT CFM velocity, LocEnc patch encoder,
// AudioVAE decoder).  Each role is one four-core AXModel with the layer loop in the
// runtime; this header is the small host surface the llama.cpp VoxCPM2 runtime calls.
#pragma once

#include <map>
#include <string>
#include <vector>

struct ggml_tensor;

namespace axera {

struct vox_acoustic {
    virtual ~vox_acoustic() = default;
};

// Open the three acoustic roles from the kernel directory's manifest (GGML_AXERA_KERNEL_DIR)
// and bind every weight from the acoustic GGUF tensors (all prefixes merged by the caller).
// Returns nullptr and fills `err` when a role, a tensor or a conversion is missing.
vox_acoustic * vox_acoustic_open(const std::map<std::string, ggml_tensor *> & tensors,
                                 const std::vector<float> & rope_factors, std::string * err);
void           vox_acoustic_close(vox_acoustic * a);

// One CFG-blended Euler velocity for one four-frame patch.  Vectors are the runtime's
// float32 arrays in [64, 4] order; `vel` returns the same order.
bool vox_acoustic_dit(vox_acoustic * a, const float * x, const float * lm, const float * res,
                      const float * cond, float t, float * vel);
// The whole CFM Euler loop in one call.  k is the number of estimator
// evaluations (9 for the model's ten-step schedule); t_sched/dt_sched hold k
// f32 entries and x0 is the temperature-scaled noise.  `patch` returns the
// final latent in the runtime's float32 [64, 4] order.
bool vox_acoustic_dit_loop(vox_acoustic * a, const float * x0, const float * lm, const float * res,
                           const float * cond, const float * t_sched, const float * dt_sched,
                           int k, float * patch);
// FSQ + residual fusion + stop head for one decoded BaseLM row.  One four-core
// AXModel replaces the three host ggml graphs.  `lm` and `fusion` are 2048-wide
// BF16 model rows and `stop` returns the two fp32 stop-head logits.
// Batched prompt tail: ROWS prompt rows through FSQ and the residual fusion in
// one call.  `base` and `feat` are [16, 2048] f32 rows (zero padded), `mask`
// picks the feature rows; `lm` and `fusion` receive 16 rows each.
bool vox_acoustic_has_prefill_tail(vox_acoustic * a);
bool vox_acoustic_prefill_tail(vox_acoustic * a, const float * base, const float * feat,
                               const float * mask, float * lm, float * fusion);

// Sixteen latent frames (four patches) in one AudioVAE decoder call.  Same
// causal state as vox_acoustic_vae, so a stream either uses this role in
// whole groups of four patches or the single-patch one.
bool vox_acoustic_has_vae_batch(vox_acoustic * a);
// Number of latent frames (and 1920 samples per frame) one batched call consumes.
int  vox_acoustic_vae_batch_frames(vox_acoustic * a);
bool vox_acoustic_vae_batch(vox_acoustic * a, const float * latents, float * pcm);
void vox_acoustic_vae_batch_reset(vox_acoustic * a);

bool vox_acoustic_has_post(vox_acoustic * a);
bool vox_acoustic_post(vox_acoustic * a, const float * base, const float * feat, const float * prev_lm,
                       float * lm, float * fusion, float * stop);
// CLS + four patch frames -> 1024 hidden.  `x` is [64, 4].
bool vox_acoustic_locenc(vox_acoustic * a, const float * x, float * hidden);
// One four-frame latent patch -> 7680 samples at 48 kHz; the causal state carries across
// calls until vox_acoustic_vae_reset().
bool vox_acoustic_vae(vox_acoustic * a, const float * patch, float * pcm);
void vox_acoustic_vae_reset(vox_acoustic * a);

// AudioVAE reference encoder: one fixed 2560-sample (four latent frame) patch
// per call and one four-row latent patch.  The causal state carries across
// calls until vox_acoustic_vae_encode_reset().
bool vox_acoustic_has_vae_encoder(vox_acoustic * a);
bool vox_acoustic_vae_encode_patch(vox_acoustic * a, const float * samples, float * latent);
void vox_acoustic_vae_encode_reset(vox_acoustic * a);

// The eight-layer ResidualLM decoder, one token per call, with its own paged BF16 KV.
// The manifest may omit the acoustic.residual role; these calls then return false.
bool vox_acoustic_has_residual(vox_acoustic * a);
bool vox_acoustic_res_step(vox_acoustic * a, const float * x, float * out);
void vox_acoustic_res_reset(vox_acoustic * a);

}  // namespace axera
