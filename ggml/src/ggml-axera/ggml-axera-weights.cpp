// Axera NPU backend: the GEMV kernels (the weight conversions: core/quant.h)

#include "ggml-axera-impl.h"


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

