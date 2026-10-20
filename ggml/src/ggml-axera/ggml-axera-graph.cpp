// Axera NPU backend: what the graph templates share around their layers (model independent): the graph's tensors by
// name, and the output head (final norm, output projection, the normed hidden state llama.cpp's MTP reads)

#include "ggml-axera-impl.h"

const std::map<std::string, ggml_tensor *> & axera_graph_names(ggml_backend_axera_context * ctx,
                                                               struct ggml_cgraph * cgraph) {
    axera_timer            tmap("graph name map");
    std::vector<uintptr_t> key;
    key.reserve((size_t) cgraph->n_nodes * (2 + GGML_MAX_SRC));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        key.push_back((uintptr_t) node);
        key.push_back((uintptr_t) node->op);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            key.push_back((uintptr_t) node->src[s]);
        }
    }
    std::map<std::string, ggml_tensor *> & t = ctx->graph_names;
    if (key != ctx->graph_key) {
        t.clear();
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            t.emplace(axera_base_name(node->name), node);
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
                ggml_tensor * src = node->src[s];
                t.emplace(axera_base_name(src->name), src);
                for (ggml_tensor * v = src->view_src; v; v = v->view_src) {  // e.g. the cache tensors behind their views
                    t.emplace(axera_base_name(v->name), v);
                }
            }
        }
        ctx->graph_key = std::move(key);
    }
    return t;
}

ggml_tensor * axera_graph_find(const std::map<std::string, ggml_tensor *> & t, const char * name) {
    auto it = t.find(name);
    return it == t.end() ? nullptr : it->second;
}

// the final norm and the output projection for the requested rows of h (the last layer's output, [T][H] f32)
void axera_output_head(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph,
                       const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & h, int T,
                       const ggml_backend_axera_buffer_context * lm_q, const ggml_backend_axera_buffer_context * lm_s,
                       const ggml_backend_axera_buffer_context * lm_bf) {
    ggml_tensor * logits = axera_graph_find(t, "result_output");
    ggml_tensor * lm     = logits->src[0];
    const int     H      = (int) lm->ne[0];
    // final norm: the lm head input is [get_rows(] rms_norm(h) * output_norm [, inp_out_ids)]
    ggml_tensor * hnext = logits->src[1];
    while (hnext->op == GGML_OP_GET_ROWS || hnext->op == GGML_OP_VIEW || hnext->op == GGML_OP_RESHAPE) {
        hnext = hnext->src[0];
    }
    GGML_ASSERT(hnext->op == GGML_OP_MUL);
    ggml_tensor * rms = hnext->src[0]->op == GGML_OP_RMS_NORM ? hnext->src[0] : hnext->src[1];
    ggml_tensor * onw = hnext->src[0] == rms ? hnext->src[1] : hnext->src[0];
    GGML_ASSERT(rms->op == GGML_OP_RMS_NORM && onw->type == GGML_TYPE_F32 && onw->ne[0] == H);
    const float eps = axera_rms_eps(rms);

    // the row ids are the GET_ROWS index that crops the last layer or the final norm to the outputs (by structure:
    // the scheduler may have renamed inp_out_ids)
    ggml_tensor *   rnorm   = axera_graph_find(t, "result_norm");
    const int       n_out   = (int) logits->ne[1];
    ggml_tensor *   out_ids = nullptr;
    for (int i = 0; i < cgraph->n_nodes && n_out < T; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
            node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
            out_ids = node->src[1];
        }
    }
    if (n_out < T && !out_ids) {
        GGML_ABORT("axera template: %d of %d rows requested but no output row index in the graph\n", n_out, T);
    }
    std::vector<float> hn((size_t) n_out * H);
    const float *   g = (const float *) onw->data;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        const float * x  = h.data() + (size_t) row * H;
        float *       y  = hn.data() + (size_t) r * H;
        double        ss = 0;
        for (int i = 0; i < H; ++i) {
            ss += (double) x[i] * x[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
        for (int i = 0; i < H; ++i) {
            y[i] = x[i] * scale * g[i];
        }
        if (rnorm && rnorm->type == GGML_TYPE_F32 && rnorm->ne[0] == H && rnorm->ne[1] == n_out) {
            memcpy((char *) rnorm->data + r * rnorm->nb[1], y, H * sizeof(float));
        }
    }
    // h_nextn (llama.cpp's MTP input, and in the MTP graph its output for the next draft step): the normed hidden
    // state of every token (the graph's T rows) or of the output rows
    ggml_tensor * hx = axera_graph_find(t, "h_nextn");
    if (hx && hx->type == GGML_TYPE_F32 && hx->ne[0] == H && ggml_backend_buffer_is_axera(hx->buffer) &&
        (hx->ne[1] == n_out || hx->ne[1] == T)) {
        if (hx->ne[1] == n_out) {
            for (int r = 0; r < n_out; ++r) {
                memcpy((char *) hx->data + r * hx->nb[1], hn.data() + (size_t) r * H, H * sizeof(float));
            }
        } else {
            axera_host_pool().run(T, [&](int64_t r) {
                const float * x  = h.data() + (size_t) r * H;
                float *       y  = (float *) ((char *) hx->data + r * hx->nb[1]);
                double        ss = 0;
                for (int i = 0; i < H; ++i) {
                    ss += (double) x[i] * x[i];
                }
                const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
                for (int i = 0; i < H; ++i) {
                    y[i] = x[i] * scale * g[i];
                }
            });
        }
        ((ggml_backend_axera_buffer_context *) hx->buffer->context)->flush(hx->data, ggml_nbytes(hx));
    }
    // several rows: the batched GEMV of the weight format reads the output projection once per call (MXFP8, the bf16
    // GGUF tensor in CMM or its bf16 copy)
    const ggml_backend_axera_buffer_context * lm_ctx =
        lm_q ? lm_q : lm_bf ? lm_bf :
        lm->type == GGML_TYPE_BF16 && ggml_backend_buffer_is_axera(lm->buffer) ? (ggml_backend_axera_buffer_context *) lm->buffer->context :
                                                                                  nullptr;
    void * lm_data = lm_q ? lm_q->vir : lm_bf ? lm_bf->vir : lm->data;
    auto   dst_of  = [&](int r) { return (float *) ((char *) logits->data + r * logits->nb[1]); };
    axera_timer tl("lm_head");
    for (int r = 0; r < n_out;) {
        int done = 0;
        for (int Bn : { 8, 4, 2 }) {
            if (done == 0 && lm_ctx && r + Bn <= n_out) {
                float * dsts[8];
                for (int b = 0; b < Bn; ++b) {
                    dsts[b] = dst_of(r + b);
                }
                done = axera_gemv_batch(ctx, lm_ctx, lm_data, lm_q ? lm_s : nullptr, lm->ne[1], H,
                                        hn.data() + (size_t) r * H, Bn, dsts) ? Bn : 0;
            }
        }
        if (done > 0) {
            r += done;
            continue;
        }
        const float * y = hn.data() + (size_t) r * H;
        if (lm_q) {
            axera_gemv_mx8(ctx, lm_q, lm_s, lm->ne[1], H, y, logits, dst_of(r));
        } else if (lm_bf) {
            axera_gemv_raw(ctx, lm_bf, lm_bf->vir, lm->ne[1], H, y, logits, dst_of(r));
        } else {
            axera_gemv(ctx, lm, y, logits, dst_of(r));
        }
        ++r;
    }
}
