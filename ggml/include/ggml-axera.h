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

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_axera_reg(void);

#ifdef  __cplusplus
}
#endif
