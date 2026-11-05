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

// The combined final-RMSNorm + lm-head + greedy-argmax head: common/sampling.cpp uses the NPU's (id, logit) for a
// pure greedy chain instead of reading the full logits. Fetched through the registry proc address
// "ggml_backend_axera_argmax"; NULL when the loaded manifest has no combined lm_head role.
typedef struct ggml_axera_argmax {
    // 1: enable for this pure greedy chain (ids: the model's suppress tokens), returns the head's row count; 0: off
    int  (*configure)(int enabled, const int32_t * ids, int32_t n);
    void (*reset)(void);
    // the published row for logits index idx (negative: from the end); 0: no fused row (read the full logits instead)
    int  (*take)(int idx, int32_t * id, float * logit);
} ggml_axera_argmax;

#ifdef  __cplusplus
}
#endif
