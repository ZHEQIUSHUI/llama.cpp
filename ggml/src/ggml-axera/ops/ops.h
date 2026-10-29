// Axera NPU backend ops: what the model adapters share around their kernels (model independent): the graph's tensors
// by name, the llama.cpp KV cache and KQ mask, RoPE tables, the output head, graph nodes on the CPU backend
#pragma once

#include "../core/manifest.h"
#include "../core/quant.h"

#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

namespace axera {

// the graph's tensors by name (the scheduler's copies "<backend>#<name>#<copy>" by <name>), rebuilt when the graph's
// nodes change (llama.cpp reuses the decode graph: rebuilding the map of a 35B graph takes ~10 ms on the A55)
struct graph_names {
    std::vector<uintptr_t>               key;
    std::map<std::string, ggml_tensor *> t;

    const std::map<std::string, ggml_tensor *> & of(const ggml_cgraph * g);
    ggml_tensor * find(const std::string & name) const;
    ggml_tensor * at(const std::string & name) const;  // aborts if absent
};

// the final norm rms_norm(h) * w (the MUL node): its weight and eps
const float * final_norm(const ggml_tensor * mul, int H, float * eps);

// a batch's view of the llama.cpp KV cache: the cells of its tokens (k_idxs: stream * N + cell) and its KQ mask
struct kv_view {
    const ggml_tensor * k_idxs = nullptr, *kqm = nullptr;
    int64_t             N = 0;   // cells per stream
    int                 BC = 0;  // KV block of the attention kernels' masks
    int64_t             W = 0;   // mask width: llama.cpp's n_kv rounded up to whole blocks (cells past n_kv hidden)

    kv_view() = default;
    kv_view(const ggml_tensor * k_idxs, const ggml_tensor * kqm, int64_t N, int BC);

    int64_t stream_of(int t) const { return ((const int64_t *) k_idxs->data)[t] / N; }
    int64_t cell_of(int t) const { return ((const int64_t *) k_idxs->data)[t] % N; }
    // KQ mask row of token t -> bf16 over the first W cells (-3e38 hidden); returns the blocks to scan
    int mask_row(int t, ggml_bf16_t * dst) const;
    // the mask [nblk][R][BC] of a chunk of P = R rows from token it with n real tokens (the padding rows the last
    // one's): from two mask rows when the tokens are one sequence's consecutive cells (llama.cpp's causal mask), else
    // row by row. Returns nblk
    int chunk_mask(const int32_t * pos, int it, int P, int n, ggml_bf16_t * mask) const;
};

// rotary tables of a batch: cos / sin rows [n][ROT / 2] (bf16) for its tokens (M-RoPE positions [4][T]: frequency i
// takes the position of its section, as ggml_mrope_cache_init)
struct rope {
    const int32_t *    pos = nullptr;
    int                T = 0, ROT = 0;
    float              freq_scale = 1.0f;
    std::vector<float> inv_freq;
    bool               mrope = false;
    std::vector<int>   sect;

    rope() = default;
    // from the graph's first ROPE node and its positions
    rope(const ggml_tensor * rope_node, const ggml_tensor * inp_pos, int T);
    // the rows of tokens t0 .. t0 + n - 1 into cb / sb (flushed)
    void rows(int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) const;
};

// the output head: the final norm of the output rows of h ([T][H] f32, the last layer's) and the MXFP4 output
// projection (role lm_head: x [K] bf16, w [N, K / 2], s [N, K / 32] -> out [N] f32) into the graph's logits; then the
// graph's nodes after the logits (llama.cpp's backend sampler chain) on the CPU
struct output_head {
    const kernel * k = nullptr;
    const ggml_tensor * w = nullptr;  // output.weight
    axera_cmm  q, s, x;               // the projection in MXFP4, the bf16 input row

    void bind(const kernel & lm, const ggml_tensor * output_weight);
    void run(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T);
};

// nodes of the graph on the CPU backend (their tensors are host memory: CMM or the CPU's)
void cpu_compute(ggml_cgraph * g, ggml_tensor ** nodes, int n, const char * what);

}  // namespace axera
