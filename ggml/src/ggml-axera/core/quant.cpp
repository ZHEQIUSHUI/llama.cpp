// Axera NPU backend core: weight conversions for the kernels

#include "quant.h"

#include <set>

// [N, K] weight of any GGUF type -> MXFP8 elements [N, K] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp8(const ggml_tensor * w, axera_cmm & q, axera_cmm & s) {
    GGML_ASSERT(w->ne[0] % AXERA_MX_GROUP == 0);
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    q = axera_cmm_alloc(N * K);
    s = axera_cmm_alloc(N * K / AXERA_MX_GROUP);
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        axera_quantize_row_mxfp8_f32(row, (uint8_t *) q->vir + r * K, (uint8_t *) s->vir + r * (K / AXERA_MX_GROUP), K);
    });
    q->flush(q->vir, q->size);
    s->flush(s->vir, s->size);
}

// [N, K] weight of any GGUF type -> MXFP4 (as ggml's and the 35B experts' kernels): per 32 inputs the scale
// 2^(floor(log2(amax)) - 2) (E8M0 byte e + 127) and E2M1 codes (nearest value, sign in bit 3), two per byte (the first in
// the low nibble): elements [N, K / 2] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp4(const ggml_tensor * w, axera_cmm & q, axera_cmm & s) {
    GGML_ASSERT(w->ne[0] % AXERA_MX_GROUP == 0);
    static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    q = axera_cmm_alloc(N * K / 2);
    s = axera_cmm_alloc(N * K / AXERA_MX_GROUP);
    if (w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w)) {  // already MXFP4: the same values, rows rearranged
        const int                n_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (int t = 0; t < n_threads; ++t) {
            pool.emplace_back([&, t] {
                const int64_t r0 = N * t / n_threads, r1 = N * (t + 1) / n_threads;
                axera_mxfp4_rows((const uint8_t *) w->data + r0 * w->nb[1], r1 - r0, K, (uint8_t *) q->vir + r0 * (K / 2),
                                 (uint8_t *) s->vir + r0 * (K / AXERA_MX_GROUP));
            });
        }
        for (auto & th : pool) {
            th.join();
        }
        return;
    }
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        uint8_t * qr = (uint8_t *) q->vir + r * (K / 2);
        uint8_t * sr = (uint8_t *) s->vir + r * (K / AXERA_MX_GROUP);
        for (int64_t g = 0; g < K / AXERA_MX_GROUP; ++g) {
            const float * v    = row + g * AXERA_MX_GROUP;
            float         amax = 0.0f;
            for (int i = 0; i < AXERA_MX_GROUP; ++i) {
                amax = std::max(amax, fabsf(v[i]));
            }
            const int   e  = amax > 0.0f ? (int) floorf(log2f(amax)) - 2 : 0;
            const float sc = ldexpf(1.0f, -e);
            sr[g]          = (uint8_t) std::min(254, std::max(0, e + 127));
            for (int i = 0; i < AXERA_MX_GROUP; i += 2) {
                uint8_t c[2];
                for (int h = 0; h < 2; ++h) {
                    const float x    = v[i + h] * sc, a = fabsf(x);
                    int         best = 0;
                    for (int k = 1; k < 8; ++k) {
                        if (fabsf(a - e2m1[k]) < fabsf(a - e2m1[best])) {
                            best = k;
                        }
                    }
                    c[h] = (uint8_t) (best | (x < 0.0f ? 8 : 0));
                }
                qr[(g * AXERA_MX_GROUP + i) / 2] = (uint8_t) (c[0] | (c[1] << 4));
            }
        }
    });
    q->flush(q->vir, q->size);
    s->flush(s->vir, s->size);
}

// [N, K] weight of another GGUF type (quantized, f16, f32) -> bf16 copy in a new CMM buffer
axera_cmm axera_to_bf16(const ggml_tensor * w) {
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    axera_cmm     b = axera_cmm_alloc(N * K * sizeof(ggml_bf16_t));
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        ggml_fp32_to_bf16_row(row, (ggml_bf16_t *) b->vir + r * K, K);
    });
    b->flush(b->vir, b->size);
    return b;
}
// ggml MXFP4 blocks (per 32 inputs: the E8M0 scale byte, then 16 bytes of E2M1 codes: low nibbles elements 0..15, high
// nibbles 16..31) -> the kernels' layout, in place per expert (dim 2): the codes [rows][K / 2], two consecutive elements
// per byte (the first in the low nibble), then the scales [rows][K / 32]. Once per tensor: the GGUF data is the
// kernels' afterwards (the template runs every MoE layer)
// ggml block_mxfp4 rows (e byte + 16 bytes of nibble pairs per 32) -> the kernels' layout: codes [rows][K / 2] (FP4
// pairs, low nibble first) then E8M0 scales [rows][K / 32]
void axera_mxfp4_rows(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dq, uint8_t * ds) {
    const int64_t nb = K / 32;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = src + (r * nb + b) * 17;
            uint8_t *       o   = dq + r * K / 2 + b * 16;
            ds[r * nb + b]      = blk[0];
            for (int j = 0; j < 8; ++j) {  // elements 2j, 2j + 1 (low nibbles), 16 + 2j, 17 + 2j (high nibbles)
                o[j]     = (uint8_t) ((blk[1 + 2 * j] & 15) | ((blk[2 + 2 * j] & 15) << 4));
                o[8 + j] = (uint8_t) ((blk[1 + 2 * j] >> 4) | (blk[2 + 2 * j] & 0xF0));
            }
        }
    }
}
void axera_check_mxfp4(const ggml_tensor * w) {
    GGML_ASSERT(w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w) && w->ne[0] % 32 == 0 && w->ne[3] == 1 &&
                ggml_backend_buffer_is_axera(w->buffer) && w->nb[2] == (size_t) (w->ne[1] * w->ne[0] / 32 * 17));
}

// MoE experts [K, rows, n_expert] in place, each expert's slot (stride bytes) in the kernels' layout
void axera_repack_mxfp4(const ggml_tensor * w) {
    static std::set<const void *> done;
    if (!done.insert(w->data).second) {
        return;
    }
    axera_check_mxfp4(w);
    const int64_t K = w->ne[0], rows = w->ne[1];
    const size_t  stride = w->nb[2];
    axera_host_pool().run(w->ne[2], [&](int64_t e) {
        std::vector<uint8_t> tmp(stride);
        axera_mxfp4_rows((const uint8_t *) w->data + e * stride, rows, K, tmp.data(), tmp.data() + rows * K / 2);
        memcpy((uint8_t *) w->data + e * stride, tmp.data(), stride);
    });
    ((ggml_backend_axera_buffer_context *) w->buffer->context)->flush(w->data, ggml_nbytes(w));
}

// gate and up experts [K, rows, n] -> one gate-up tensor at dst (2 nbytes): expert e at 2 e strides, codes
// [2 rows][K / 2] (gate rows, then up), scales [2 rows][K / 32] (as a GGUF ffn_gate_up_exps repacked by
// axera_repack_mxfp4). In place when up follows gate (dst = gate: llama.cpp allocates them so unless a buffer boundary
// falls between), else into a buffer of its own
bool axera_mxfp4_gu_adjacent(const ggml_tensor * g, const ggml_tensor * u) {
    return g->buffer == u->buffer && (const char *) u->data == (const char *) g->data + ggml_nbytes(g);
}
void axera_repack_mxfp4_gu(const ggml_tensor * g, const ggml_tensor * u, uint8_t * dst,
                           const ggml_backend_axera_buffer_context * db) {
    axera_check_mxfp4(g);
    axera_check_mxfp4(u);
    GGML_ASSERT(ggml_are_same_shape(g, u));
    const int64_t K = g->ne[0], rows = g->ne[1], n = g->ne[2];
    const size_t  stride = g->nb[2], half = (size_t) (rows * K / 2);
    std::vector<uint8_t> copy;  // in place: the sources, before they are overwritten
    const uint8_t *      gs = (const uint8_t *) g->data, *us = (const uint8_t *) u->data;
    if (dst == g->data) {
        GGML_ASSERT(axera_mxfp4_gu_adjacent(g, u));
        copy.assign(gs, gs + 2 * ggml_nbytes(g));
        gs = copy.data();
        us = copy.data() + ggml_nbytes(g);
    }
    axera_host_pool().run(n, [&](int64_t e) {
        uint8_t * dq = dst + 2 * e * stride;
        uint8_t * ds = dq + 2 * half;
        axera_mxfp4_rows(gs + e * stride, rows, K, dq, ds);
        axera_mxfp4_rows(us + e * stride, rows, K, dq + half, ds + rows * K / 32);
    });
    db->flush(dst, 2 * ggml_nbytes(g));
}
