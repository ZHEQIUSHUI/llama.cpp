// Axera NPU backend: weight conversions (bf16, MXFP8, MXFP4 repack) and the GEMV kernels

#include "ggml-axera-impl.h"


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
void axera_gemv_raw(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                    void * wdata, int64_t N, int64_t K, const float * x, ggml_tensor * dst, float * out) {
    axera_gemv_raw_to(ctx, wctx, wdata, N, K, x, (const ggml_backend_axera_buffer_context *) dst->buffer->context, out);
}

// out[0:N] (f32, in the CMM buffer dctx) = W[N, K] (bf16 at wdata in wctx) . x[0:K], in N / C GEMV calls
void axera_gemv_raw_to(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                       void * wdata, int64_t N, int64_t K, const float * x,
                       const ggml_backend_axera_buffer_context * dctx, float * out) {
    const int64_t row = K * (int64_t) sizeof(ggml_bf16_t);
    const int64_t C   = axera_kernels().mul_mat_chunk(N, K);
    GGML_ASSERT(C > 0);

    ggml_axera_kernel * k  = axera_kernels().get(axera_mul_mat_kernel_name(C, K));
    const int           ix = k->input("x"), iw = k->input("w"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && k->info->nInputSize == 2 && k->info->nOutputSize == 1);
    GGML_ASSERT(k->info->pInputs[ix].nSize == K * sizeof(ggml_bf16_t) &&
                k->info->pInputs[iw].nSize == C * row && k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
    ggml_fp32_to_bf16_row(x, xb, K);
    st->flush(xb, K * sizeof(ggml_bf16_t));
    dctx->flush(out, N * sizeof(float));

    AX_ENGINE_IO_BUFFER_T inputs[2], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, xb, K * sizeof(ggml_bf16_t));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw]    = axera_io(wctx, (char *) wdata + c0 * row, C * row);
        outputs[io]   = axera_io(dctx, out + c0, C * sizeof(float));
        axera_warn_one_core(k);
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, N, K);
        }
    }
    dctx->invalidate(out, N * sizeof(float));
    dctx->clean_p     = out;  // the host reads these logits back without more cache maintenance this graph
    dctx->clean_n     = N * sizeof(float);
    dctx->clean_epoch = axera_graph_epoch;
}

// out[0:N] = MXFP4 W[N, K] (elements q [N, K / 2], scales s [N, K / 32]) . x[0:K] in one call of mul_mat_mx4_n<N>_k<K>
void axera_gemv_mx4(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                    const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                    ggml_tensor * dst, float * out) {
    ggml_axera_kernel * k  = axera_kernels().get("mul_mat_mx4_n" + std::to_string(N) + "_k" + std::to_string(K));
    const int           ix = k->input("x"), iw = k->input("w"), is = k->input("s"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && is >= 0 && io >= 0 && k->info->pInputs[iw].nSize == N * K / 2 &&
                k->info->pInputs[is].nSize == N * K / AXERA_MX_GROUP && k->info->pOutputs[io].nSize == N * sizeof(float));
    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
    auto * dctx = (ggml_backend_axera_buffer_context *) dst->buffer->context;
    ggml_fp32_to_bf16_row(x, xb, K);
    st->flush(xb, K * sizeof(ggml_bf16_t));
    dctx->flush(out, N * sizeof(float));
    GGML_ASSERT(k->info->nInputSize == 3 && k->info->nOutputSize == 1);  // x, w, s -> out
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), o(k->info->nOutputSize);
    in[ix]    = axera_io(st, xb, K * sizeof(ggml_bf16_t));
    in[iw]    = axera_io(q, q->vir, (uint32_t) (N * K / 2));
    in[is]    = axera_io(sc, sc->vir, (uint32_t) (N * K / AXERA_MX_GROUP));
    o[io]     = axera_io(dctx, out, (uint32_t) (N * sizeof(float)));
    const int ret = axera_run_sync(k, in, o);
    if (ret != 0) {
        GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, N, K);
    }
    dctx->invalidate(out, N * sizeof(float));
    dctx->clean_p     = out;
    dctx->clean_n     = N * sizeof(float);
    dctx->clean_epoch = axera_graph_epoch;
}

// rows out[b][0:N] = W[N, K] . x[b], b < Bn, through the batched GEMV mul_mat_<bf16|mx8>_b<Bn>_n<C>_k<K> (x [Bn, K]
// bf16 -> out [Bn, C] f32), N / C calls. W: bf16 at wdata in the CMM buffer wctx, or (sctx) MXFP8 elements wctx and
// scales sctx. False if that kernel is missing
bool axera_gemv_batch(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx, void * wdata,
                      const ggml_backend_axera_buffer_context * sctx, int64_t N, int64_t K, const float * x,
                      int Bn, float * const * out) {
    const bool    mx8 = sctx != nullptr;
    const int64_t C   = axera_kernels().mul_mat_chunk(N, K, mx8);
    char          name[96];
    snprintf(name, sizeof(name), "mul_mat_%s_b%d_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", Bn, C, K);
    static std::map<std::string, bool> found;
    auto f = found.find(name);
    if (f == found.end()) {
        f = found.emplace(name, C > 0 && std::ifstream(axera_kernels().path(name)).good()).first;
    }
    if (!f->second) {
        return false;
    }
    ggml_axera_kernel * k   = axera_kernels().get(name);
    const int           ix  = k->input("x"), iw = k->input("w"), is = k->input("s"), io = k->output("out");
    const int64_t       row = K * (int64_t) sizeof(ggml_bf16_t);
    const int64_t       wr  = mx8 ? K : row, sr = K / AXERA_MX_GROUP;  // weight / scale bytes per row
    GGML_ASSERT(ix >= 0 && iw >= 0 && io >= 0 && (is >= 0) == mx8 && k->info->pInputs[ix].nSize == Bn * row &&
                k->info->pInputs[iw].nSize == C * wr && (!mx8 || k->info->pInputs[is].nSize == C * sr) &&
                k->info->pOutputs[io].nSize == Bn * C * sizeof(float));
    auto * st = ctx->staging_for(Bn * row);
    ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) st->vir, Bn * K);
    st->flush(st->vir, Bn * row);
    if (!ctx->batch_out || ctx->batch_out->size < Bn * C * sizeof(float)) {
        ctx->batch_out = axera_cmm_alloc(Bn * C * sizeof(float));
    }
    auto * bo = ctx->batch_out.get();
    AX_ENGINE_IO_BUFFER_T inputs[3], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = mx8 ? 3 : 2;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, st->vir, Bn * row);
    outputs[io]        = axera_io(bo, bo->vir, Bn * C * sizeof(float));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw] = axera_io(wctx, (char *) wdata + c0 * wr, C * wr);
        if (mx8) {
            inputs[is] = axera_io(sctx, (char *) sctx->vir + c0 * sr, C * sr);
        }
        axera_warn_one_core(k);
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 " rows=%d\n", __func__, ret, N, K, Bn);
        }
        bo->invalidate(bo->vir, Bn * C * sizeof(float));
        for (int b = 0; b < Bn; ++b) {
            memcpy(out[b] + c0, (const float *) bo->vir + (size_t) b * C, C * sizeof(float));
        }
    }
    return true;
}
void axera_gemv(ggml_backend_axera_context * ctx, const ggml_tensor * w, const float * x, ggml_tensor * dst,
                float * out) {
    GGML_ASSERT(w->type == GGML_TYPE_BF16 && w->nb[1] == (size_t) w->ne[0] * sizeof(ggml_bf16_t));
    axera_gemv_raw(ctx, (ggml_backend_axera_buffer_context *) w->buffer->context, w->data, w->ne[1], w->ne[0], x, dst,
                   out);
}

// out[0:N] (f32, inside a CMM tensor) = MXFP8 W[N, K] (elements q, scales s) . x[0:K], in N / C GEMV calls
void axera_gemv_mx8(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                    const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                    ggml_tensor * dst, float * out) {
    const int64_t C = axera_kernels().mul_mat_chunk(N, K, true);
    GGML_ASSERT(C > 0);
    ggml_axera_kernel * k  = axera_kernels().get(axera_mul_mat_kernel_name(C, K, true));
    const int           ix = k->input("x"), iw = k->input("w"), is = k->input("s"), io = k->output("out");
    GGML_ASSERT(ix >= 0 && iw >= 0 && is >= 0 && io >= 0 && k->info->nInputSize == 3 && k->info->nOutputSize == 1);
    const int64_t gs = K / AXERA_MX_GROUP;
    GGML_ASSERT(k->info->pInputs[iw].nSize == C * K && k->info->pInputs[is].nSize == C * gs &&
                k->info->pOutputs[io].nSize == C * sizeof(float));

    auto * st   = ctx->staging_for(K * sizeof(ggml_bf16_t));
    auto * xb   = (ggml_bf16_t *) st->vir;
    auto * dctx = (ggml_backend_axera_buffer_context *) dst->buffer->context;
    ggml_fp32_to_bf16_row(x, xb, K);
    st->flush(xb, K * sizeof(ggml_bf16_t));
    dctx->flush(out, N * sizeof(float));

    AX_ENGINE_IO_BUFFER_T inputs[3], outputs[1];
    AX_ENGINE_IO_T        io_set;
    memset(&io_set, 0, sizeof(io_set));
    io_set.pInputs     = inputs;
    io_set.nInputSize  = 3;
    io_set.pOutputs    = outputs;
    io_set.nOutputSize = 1;
    inputs[ix]         = axera_io(st, xb, K * sizeof(ggml_bf16_t));
    for (int64_t c0 = 0; c0 < N; c0 += C) {
        inputs[iw]    = axera_io(q, (char *) q->vir + c0 * K, C * K);
        inputs[is]    = axera_io(sc, (char *) sc->vir + c0 * gs, C * gs);
        outputs[io]   = axera_io(dctx, out + c0, C * sizeof(float));
        axera_warn_one_core(k);
        const int ret = AX_ENGINE_RunSync(k->handle, &io_set);
        if (ret != 0) {
            GGML_ABORT("%s: RunSync failed (0x%x), n=%" PRId64 " k=%" PRId64 "\n", __func__, ret, N, K);
        }
    }
    dctx->invalidate(out, N * sizeof(float));
    dctx->clean_p     = out;  // the host reads these logits back without more cache maintenance this graph
    dctx->clean_n     = N * sizeof(float);
    dctx->clean_epoch = axera_graph_epoch;
}

void ggml_backend_axera_mul_mat(ggml_backend_axera_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    // one activation row at a time: the weights are reread per row until a batched kernel exists
    for (int64_t m = 0; m < src1->ne[1]; ++m) {
        axera_gemv(ctx, src0, (const float *) ((const char *) src1->data + m * src1->nb[1]), dst,
                   (float *) ((char *) dst->data + m * dst->nb[1]));
    }
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
