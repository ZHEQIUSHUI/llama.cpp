#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_AXERA_NAME "AXERA"

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_axera_init(void);

GGML_BACKEND_API bool ggml_backend_is_axera(ggml_backend_t backend);

// device memory (CMM): host-visible, usable by the CPU backend as well
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_axera_buffer_type(void);
// the llama.cpp KV / recurrent-state caches: lazily-backed host memory for the adapters that keep the state in their
// own pools (nothing reads them), else CMM; llama.cpp asks for it through the registry ("ggml_backend_axera_kv_buffer_type")
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_axera_kv_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_axera_lazy_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_axera_reg(void);

// the host sampler's side of the fused lm_head.topk kernel (common/sampling.cpp), fetched through the registry's
// proc address ("ggml_backend_axera_npu_sampling"); NULL when the manifest has no such kernel
typedef struct ggml_axera_npu_sampling {
    // the chain the host will run; returns the kernel's candidate count K, 0 when the chain is unsupported
    int (*configure)(int enabled, int32_t top_k, float top_p, float min_p, float temp_sumexp, float temp_minp,
                     int32_t penalty_last_n, float repeat, float freq, float present);
    void (*reset)(void);
    void (*accept)(int32_t token);
    // the adapter's next row of candidates (verification rows are taken in order): sorted by logit, the penalised
    // logits and their token ids, max and the full-vocabulary sum(exp((l - max) / temp)).
    // 1: candidates, 0: none (the full logits are already the path), -1: this row needs its full logits
    int (*take)(float * values, int32_t * ids, int32_t max_k, int32_t * n, float * vmax, float * sumexp);
    // compute the full logits row requested by take() == -1 (the adapter's regular head), for graph logits index idx,
    // into out[0..n); 1 on success
    int (*full_row)(int32_t idx, float * out, int32_t n);
    // the penalty state, for a sampler checkpoint: restore bumps the generation and republishes the restored rows
    void * (*save)(void);
    void   (*restore)(void * state);
    void   (*free_state)(void * state);
    // the configuration's id: a sampler that is freed disables the path only when it still owns it
    int64_t (*config_id)(void);
} ggml_axera_npu_sampling;

#ifdef  __cplusplus
}
#endif
