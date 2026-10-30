// Axera NPU backend: what the backend, device and buffer sources share (ggml-axera.cpp, ggml-axera-buffer.cpp)
#pragma once

// Axera NPU backend.
//
// - The kernel directory (GGML_AXERA_KERNEL_DIR, default ./axera_kernels) holds one model's 4-core kernels and its
//   manifest.json (core/manifest.h): the model adapter it names (models/) runs every graph of the process with those
//   kernels. Without a manifest the backend has no device.
// - Device memory is CMM allocated cached (AX_SYS_MemAllocCached). It is host-visible, so the CPU backend works on it
//   directly for the graph nodes an adapter leaves to it; the NPU addresses it physically. Cache maintenance: the CPU
//   side flushes after writing, the NPU side flushes its output range before a run (no dirty line may be evicted over
//   the result) and invalidates it afterwards.
// - Weights stay in the GGUF file mapping (mapped buffers); the adapters convert what their kernels read into CMM.
//
// Settings: GGML_AXERA_KERNEL_DIR; GGML_AXERA_PROF=1 times per kernel role and host step (printed at exit);
// GGML_AXERA_TRACE=1 every graph's node count, host gap and run time.

#include "ggml-axera.h"
#include "core/runtime.h"
#include "models/model.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <memory>

bool ggml_backend_buffer_is_axera(ggml_backend_buffer_t buffer);
ggml_backend_buffer_type_t ggml_backend_axera_mapped_buffer_type(void);
ggml_backend_buffer_t axera_mapped_buffer(void * ptr, size_t size);

struct ggml_backend_axera_context {
    std::shared_ptr<axera::model> model;  // the kernel directory's model adapter (its manifest), made by the first graph
};
