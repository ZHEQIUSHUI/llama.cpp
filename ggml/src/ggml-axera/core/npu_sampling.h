// Axera NPU backend: the host side of the fused lm_head.topk sampling kernel.
//
// common/sampling.cpp configures the chain it will run and feeds the penalty state through this interface; the
// adapter's output head runs the kernel when the chain is one the K candidates can serve exactly, and publishes the
// candidates back. The state is process wide: the adapter runs one sequence (-np 1).
#pragma once

#include "npu_correct.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace axera {

// one vocabulary row whose penalty state changed: mul_pos / mul_neg select llama.cpp's repeat penalty by the logit's
// sign (l > 0 ? l / repeat : l * repeat), bias is -(count * frequency + presence)
struct npu_penalty_row {
    int32_t token = 0;
    float   mul_pos = 1.0f;
    float   mul_neg = 1.0f;
    float   bias = 0.0f;
};

struct npu_candidates {
    std::vector<float>   values;
    std::vector<int32_t> ids;
    int32_t              n = 0;  // valid candidates
    float                vmax = 0.0f, sumexp = 0.0f;
    bool                 valid = false;
    bool                 full = false;  // the candidate list cannot serve this row: take() asks for its full logits
};

struct npu_sampling {
    mutable std::mutex mu;

    bool    enabled = false;  // the host chain is one the K candidates serve exactly
    int64_t generation = 0;   // bumps on configure / reset: the adapter initialises the arrays when it changes
    int64_t config_id  = 0;   // bumps on configure only: the sampler that owns the current configuration
    int32_t K = 0;            // the loaded kernel's candidate count per row (0: no lm_head.topk kernel)
    int32_t R = 0;            // the kernel's row count (1, or 4 for the MTP verification kernels)

    // the chain's options the adapter needs to check the one case the candidates alone cannot answer (top_k == 0)
    int32_t top_k = 0;
    float   top_p = 1.0f;
    float   min_p = 0.0f;
    float   temp_sumexp = 1.0f;  // the accumulated temperature at the top-p sampler
    float   temp_minp   = 1.0f;  // ... and at the min-p sampler

    // the penalty sampler's parameters and state (the host mirrors its ring buffer)
    int32_t                    penalty_last_n = 0;
    float                      repeat = 1.0f, freq = 0.0f, present = 0.0f;
    std::unordered_map<int32_t, int> counts;
    std::deque<int32_t>        ring;
    std::vector<npu_penalty_row> updates;  // the rows changed since the adapter last took them

    // the fused kernel's last result, one entry per verification row. The penalty arrays above hold the state up to
    // the verified prefix; `base_counts` / `base_ring` are that same snapshot. Rows after an accepted token simulate
    // the <= 3 extra accepts on the host and correct the candidates (or ask for that row's full logits).
    std::vector<npu_candidates> rows;
    bool                        valid = false;
    std::vector<npu_corrected_candidate> scratch;  // npu_take's corrected row (reused)
    std::unordered_map<int32_t, int> base_counts;
    std::deque<int32_t>              base_ring;
    bool                             base_valid = false;
    std::vector<int32_t>             row_extra;  // tokens accepted since the last publish, i.e. the next row index

    // the adapter's regular head, for a row whose correction is not exact from K candidates
    int (*full_row_cb)(void *, int32_t, float *, int32_t) = nullptr;
    void * full_row_ud = nullptr;
};

npu_sampling & npu_state();

// common/sampling.cpp: returns the kernel's K, 0 when there is no lm_head.topk kernel or the chain is unsupported
int  npu_configure(bool enabled, int32_t top_k, float top_p, float min_p, float temp_sumexp, float temp_minp,
                   int32_t penalty_last_n, float repeat, float freq, float present);
void npu_reset();
void npu_accept(int32_t token);
int  npu_take(float * values, int32_t * ids, int32_t max_k, int32_t * n, float * vmax, float * sumexp);
int  npu_full_row(int32_t idx, float * out, int32_t n);
void npu_set_full_row(int (*fn)(void *, int32_t, float *, int32_t), void * ud);
void * npu_save();
void   npu_restore(void * state);
void   npu_free_state(void * state);

struct npu_opts {
    bool    enabled = false;
    int32_t K = 0, top_k = 0;
    float   top_p = 1.0f, min_p = 0.0f, temp = 1.0f, temp_minp = 1.0f;
};
npu_opts npu_get_options();
int64_t  npu_config_id();

// the adapter: the rows to write and whether the arrays must be reset first; the kernel's result
void npu_take_updates(std::vector<npu_penalty_row> & rows, int64_t * generation);
void npu_publish(const float * values, const int32_t * ids, int32_t n, int32_t rows,
                 const float * vmax, const float * sumexp, const uint8_t * usable);
}  // namespace axera
