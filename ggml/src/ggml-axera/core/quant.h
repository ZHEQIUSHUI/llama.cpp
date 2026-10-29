// Axera NPU backend core: weight conversions for the kernels (bf16, MXFP8, MXFP4, the MXFP4 expert layouts)
#pragma once

#include "runtime.h"
#include "../mx.h"

#include "ggml-backend-impl.h"

#include <thread>

// [N, K] weight of any GGUF type -> MXFP8 elements [N, K] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp8(const ggml_tensor * w, axera_cmm & q, axera_cmm & s);
// [N, K] weight of any GGUF type -> MXFP4 elements [N, K / 2] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp4(const ggml_tensor * w, axera_cmm & q, axera_cmm & s);
// [N, K] weight of another GGUF type (quantized, f16, f32) -> bf16 copy in a new CMM buffer
axera_cmm axera_to_bf16(const ggml_tensor * w);
// ggml block_mxfp4 rows -> the kernels' layout: codes [rows][K / 2] (FP4 pairs, low nibble first), scales [rows][K / 32]
void axera_mxfp4_rows(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dq, uint8_t * ds);
void axera_check_mxfp4(const ggml_tensor * w);
// MoE experts [K, rows, n_expert] in place, each expert's slot (stride bytes) in the kernels' layout (once per tensor)
void axera_repack_mxfp4(const ggml_tensor * w);
bool axera_mxfp4_gu_adjacent(const ggml_tensor * g, const ggml_tensor * u);
// gate and up experts -> one gate-up tensor at dst (codes gate rows then up rows, scales likewise)
void axera_repack_mxfp4_gu(const ggml_tensor * g, const ggml_tensor * u, uint8_t * dst,
                           const ggml_backend_axera_buffer_context * db);

// runs fn(row, float row) for every row of a 2-D weight of any GGUF type (f32, f16, bf16, quantized), on all cores
template <typename F> void axera_for_rows_f32(const ggml_tensor * w, F fn) {
    GGML_ASSERT(ggml_is_contiguous(w));
    const int64_t K      = w->ne[0], N = ggml_nrows(w);
    const auto *  traits = ggml_get_type_traits(w->type);
    GGML_ASSERT(w->type == GGML_TYPE_F32 || traits->to_float != nullptr);
    const int                n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            std::vector<float> row(K);
            for (int64_t r = t; r < N; r += n_threads) {
                const void * src = (const char *) w->data + r * w->nb[1];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, K * sizeof(float));
                } else {
                    traits->to_float(src, row.data(), K);
                }
                fn(r, row.data());
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }
}
