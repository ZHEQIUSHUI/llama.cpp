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
