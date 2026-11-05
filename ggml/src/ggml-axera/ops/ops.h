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
    uint64_t                             uid = 0;  // ggml_cgraph.uid, when non-zero (the exact graph identity)
    std::vector<uintptr_t>               key;      // the node / op fallback when the graph has no uid
    std::map<std::string, ggml_tensor *> t;
    // the tensors and base names the map was built from, in graph order: a rebuilt graph with the same tensors and
    // names maps every name the same way, even when their shapes changed (a prompt after the warmup)
    std::vector<std::pair<const ggml_tensor *, uint64_t>> sig;
    // uid-keyed maps of recent graphs (llama.cpp alternates a few: a speculative target's and draft's)
    std::vector<std::pair<uint64_t, std::map<std::string, ggml_tensor *>>> recent_uid;
    // the maps of the last few other graphs (llama.cpp alternates a few: a speculative target's and draft's)
    std::vector<std::pair<std::vector<uintptr_t>, std::map<std::string, ggml_tensor *>>> recent;

    const std::map<std::string, ggml_tensor *> & of(const ggml_cgraph * g);
    ggml_tensor * find(const std::string & name) const;
    ggml_tensor * at(const std::string & name) const;  // aborts if absent
};

// the graph is llama.cpp's warmup: its state is not a sequence's
bool warmup_active();

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

// the output projection of a graph: result_output, or the codec head (Qwen3-TTS) whose logits llama.cpp pads to the
// text vocabulary with -inf in front (result_output = concat(pad, head)); nullptr if none (pooled embeddings)
ggml_tensor * graph_head(const graph_names & names);

// pooled embeddings (no output projection): the final norm of every row of h ([T][H] f32) into result_norm (or
// result_embd_pooled); the pooling after it stays on the CPU
void output_norm(const graph_names & names, const std::vector<float> & h, int T);

// the output head: the final norm of the output rows of h ([T][H] f32, the last layer's) and the output projection
// (role lm_head: x [K] bf16, w [N, K] MXFP8 or [N, K / 2] MXFP4, s [N, K / 32] -> out [N] f32) into the graph's
// logits (and its padded copy); then the graph's nodes after the logits (llama.cpp's backend sampler chain) on the CPU.
// With a whole-model kernel's head (bind_cpu): the rows it computed (normed: their final norm, last: the logits of
// the last row) as they are, the projection of any other row on the CPU (a small head: the TTS codec head's)
struct output_head {
    const kernel * k = nullptr;
    const ggml_tensor * w = nullptr;  // output.weight
    axera_cmm  q, s, x;               // the projection in the kernel's format, the bf16 input row(s)
    axera_cmm  yk, mk;                // a kernel of R rows per call (input m: the valid ones): its output, m
    axera_cmm  ffull;                 // the full-row fallback's output when the head kernel has no m (R == 1)
    axera_cmm  tx;                    // the fused topk kernel's input rows (tR, independent of the head's R)
    int        R   = 1;
    bool       mx4 = false;
    std::vector<float>                 yn;    // the normalized output rows
    std::vector<int>                   rows;  // their rows in the graph
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    std::vector<float> wf;            // the projection in f32 (the CPU's)
    // a draft head (speculative drafts): the rows of a draft vocabulary (ids ascending) gathered from q / s, one row
    // per call; the other logits -inf
    const kernel *       dk = nullptr;
    std::vector<int32_t> dids;
    axera_cmm            dq, ds, dy;
    bool                 draft = false;  // the next run's single output row through the draft head
    // the fused head + penalties + exact top-K (role lm_head.topk): the penalty arrays the host sampler maintains, the
    // candidates and the full-vocabulary max / sumexp it returns
    const kernel *       tk = nullptr;
    int                  tK = 0;
    int                  tR = 0;  // the kernel's rows per call (1, or 4 for a verification batch)
    axera_cmm            tbias, tmulp, tmuln, ttemp, tkm, tval, tids, tmax, tsum;
    int64_t              tgen = -1;  // the penalty state's generation tbias/tmulp/tmuln hold
    std::vector<float>   tv;
    std::vector<int32_t> ti;
    // the rows of the last fused run, for the sampler's per-row full-logits fallback
    ggml_tensor *      flogits = nullptr;
    void *             fdctx   = nullptr;
    std::vector<float> fy;
    std::vector<int>   frows;
    int                fn_out = 0;
    // the fused RMSNorm + projection head (role lm_head_fused): it consumes the layer kernel's bf16 hidden row
    // directly, so a decode saves the host final norm, the bf16 conversion and the hidden-row cache round trip
    const kernel *                             fk = nullptr;
    axera_cmm                                  fkw;  // the final norm weight, f32
    const float *                              fgw = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T>         fin, fout;
    // The combined output head (role lm_head): final RMSNorm + projection + greedy argmax. It always writes the full
    // logits for the fallback path, plus one id/logit per row. The greedy sampler reads only the latter.
    const kernel *                     ck = nullptr;
    int                                cR = 0;
    axera_cmm                          cnorm, cids, ctop, cx;
    const float *                      cgw = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> cin, cout;
    // the same combined head for an MTP verify's four rows (role lm_head.verify; input m = the valid rows)
    const kernel *                     c4k = nullptr;
    axera_cmm                          c4norm, c4x, c4m, c4ids, c4top;
    const float *                      c4gw = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> c4in, c4out;

    void bind(const kernel & lm, const ggml_tensor * output_weight);
    void bind_topk(const kernel & lm);  // after bind, when the manifest has lm_head.topk
    bool bind_fused(const kernel & lm, const ggml_tensor * output_weight);
    bool bind_combined(const kernel & lm, const ggml_tensor * output_weight);
    bool bind_combined4(const kernel & lm, const ggml_tensor * output_weight);
    bool combined_ready(ggml_cgraph * g, const graph_names & names) const;
    void run_combined(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T,
                      const std::vector<int> * compact_rows, const axera_cmm * xb);
    void run_combined4(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T);
    // ready for a one-row decode of this graph: the head's logits are the graph's and the graph's final-norm eps is
    // the one the kernel was built with
    bool fused_ready(ggml_cgraph * g, const graph_names & names) const;
    // the fused head's one row, from the layer kernel's bf16 hidden row
    void run_fused(ggml_cgraph * g, const graph_names & names, const axera_cmm & xb);
    void bind_draft(const kernel & dlm, const std::vector<int32_t> & ids);  // after bind (MXFP4)
    void bind_cpu(const ggml_tensor * output_weight);
    // compact_rows: h holds only the output rows, in the same order as the graph's output row indices
    void run(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T,
             const std::vector<char> * normed = nullptr, const float * last = nullptr,
             const std::vector<int> * compact_rows = nullptr);
    int full_row(int32_t idx, float * out, int32_t n);
    static int full_row_cb(void * ud, int32_t idx, float * out, int32_t n) {
        return static_cast<output_head *>(ud)->full_row(idx, out, n);
    }
    // the logits the caller wrote (a whole-model kernel's) into the graph's: the sampler nodes after them
    static void after(ggml_cgraph * g, const graph_names & names);
};

// a PAD node of f32 rows (zeros after each dimension) on the host; false if not that shape
bool pad_rows(ggml_tensor * n);

// ggml's TOP_K of a node (its F32 rows) on the host's workers
void top_k_rows(ggml_tensor * n);

// nodes of the graph on the CPU backend (their tensors are host memory: CMM or the CPU's)
void cpu_compute(ggml_cgraph * g, ggml_tensor ** nodes, int n, const char * what);

}  // namespace axera
