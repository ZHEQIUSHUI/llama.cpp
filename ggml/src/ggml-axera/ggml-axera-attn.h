// Axera NPU backend: attention on the llama.cpp KV cache for the graph templates (model independent)
//
// DEPRECATED: the split attention path (flash decoding) is kept only for the models not yet on the fused four-core
// kernels (the Qwen3 family, Qwen3-ASR, Qwen3-TTS, Qwen3.5-0.8B); remove it with their migration. Qwen3.5 2B / 9B / 27B
// and the 35B run fused at every context their kernel directories hold, and stop beyond it
// Split attention (flash decoding): a head kernel (the new Q / K / V rows), one call of a segment kernel per KV segment
// of S cells in use (partial m / l / o over the cells a token sees), merged on the CPU, and a tail kernel. The model's
// layer kernels around it, the kernels' names and their binding belong to the template.
#pragma once

#include "ggml-axera-impl.h"

struct axera_kv_region {  // the llama.cpp KV cache of an attention layer (kc / vc output): one block per stream
    ggml_backend_axera_buffer_context * b;
    char *                              base;    // stream 0
    size_t                              size;    // the kernel's IO size (LMAX cells)
    size_t                              stride;  // bytes per stream
    size_t                              row = 0;  // bytes per cell
    int                                 out = -1;  // -1: split attention (host writes, segment IO)
    bool                                v = false;
};

struct axera_fd_io {  // split attention IO of one run kind: head, segment and tail kernels
    std::vector<AX_ENGINE_IO_BUFFER_T> hin, hout, sin, sout, tin, tout;
    int                                kc = -1, vc = -1, mask = -1;  // segment inputs, rebound per call
};

struct axera_fd_kernels {  // split attention kernels for R rows
    ggml_axera_kernel *    head = nullptr, *seg = nullptr, *tail = nullptr;
    int                    hkv = 0, gr = 0, d = 0;  // segment outputs [hkv][gr][d]: KV head, row g * R + token
    int                    bc  = 0;  // cells per block of the segment kernel (its mask [S / bc][R][bc])
    int                    cores = 1;  // segment kernel cores, each scanning its own run of blocks
    axera_cmm              Q, GATE, K, V, M, Lsum, O, ATT, BLK;  // BLK: nblk [1], or [cores][start, stop)
    std::vector<axera_cmm> MASK;  // per segment [S / BC][R][BC], filled once per batch for all layers
    std::vector<int32_t>   nblk;  // per segment: blocks to scan, 0 if no token sees a cell of it
    std::vector<int32_t>   runs;  // per segment and core: the run of blocks [start, stop)
    struct part {                  // a partial result to merge: segment g, core c, the tokens that see a cell of it
        int64_t              g;
        int                  c;
        std::vector<uint8_t> seen;
    };
    std::vector<part> parts;
};

// a batch's view of the llama.cpp KV cache: the cells of its tokens (k_idxs: stream * N + cell) and its KQ mask
// [n_kv, T / streams, 1, streams] (the ubatch's streams in order)
struct axera_kv_batch {
    const ggml_tensor * k_idxs = nullptr, *kqm = nullptr;
    int64_t             N = 0;  // cells per stream
    int                 BC = 0;  // KV block of the fused attention kernels' masks
    int64_t             S = 0;   // split attention segment, cells
    int64_t             W = 0;   // mask width: llama.cpp's n_kv rounded up to whole blocks (cells past n_kv hidden)

    struct row_span {  // the first and last visible cell of a mask row and the number of visible cells
        int64_t lo = -1, hi = -1, n = 0;
    };

    axera_kv_batch() = default;
    axera_kv_batch(const ggml_tensor * k_idxs, const ggml_tensor * kqm, int64_t N, int BC, int64_t S);

    int64_t stream_of(int t) const { return ((const int64_t *) k_idxs->data)[t] / N; }
    int64_t cell_of(int t) const { return ((const int64_t *) k_idxs->data)[t] % N; }
    // KQ mask row of token t -> bf16 over the first W cells; returns the number of blocks to scan (the kernels never
    // read past them). span (optional): the row's visible cells
    int mask_row(int t, ggml_bf16_t * dst, row_span * span = nullptr) const;
};

// the KQ mask rows of tokens t0 .. t0 + R - 1 cut into the segments' masks, and per segment the blocks to scan (0: no
// token sees a cell of it); the same for every attention layer of the batch
void axera_flash_prepare(const axera_kv_batch & kb, axera_fd_kernels & fk, int R, int t0);
// the new K / V rows of tokens t0 .. t0 + R - 1 (head kernel outputs) to their cells of the llama.cpp cache
void axera_write_kv(const axera_kv_batch & kb, std::vector<axera_kv_region> & kv, axera_fd_kernels & fk, int R, int t0);
// split attention layer il for R rows from token t0 (position p, for messages)
void axera_flash_attn(const axera_kv_batch & kb, std::vector<axera_kv_region> & kv, axera_fd_kernels & fk,
                      axera_fd_io & io, int R, int t0, int il, int64_t p);

// rotary tables of a batch: cos / sin rows [n][ROT / 2] (bf16) for its tokens. M-RoPE (positions [4][T]: t, h, w, e):
// rotary frequency i takes the position of its section, as ggml's ggml_mrope_cache_init (interleaved t / h / w for
// IMROPE). Text tokens have t = h = w
struct axera_rope {
    const int32_t *  pos;
    int              T, ROT;
    float            freq_scale;
    const float *    inv_freq;
    bool             mrope;
    std::vector<int> sect;  // M-RoPE: the position (section) of each rotary frequency

    axera_rope(const ggml_tensor * inp_pos, int T, int ROT, float freq_scale, const std::vector<float> & inv_freq,
               int mode, const int sections[4]);
    void rows(int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) const;
};
