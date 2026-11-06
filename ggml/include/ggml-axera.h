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

// The last target graph's input-embedding rows in CMM, or NULL. Used by the MTP prompt
// catch-up to read the target's token embeddings in place instead of copying them back.
GGML_BACKEND_API const void * ggml_backend_axera_get_input_embed(ggml_backend_t backend, size_t * stride);

#ifdef  __cplusplus
}
#endif
