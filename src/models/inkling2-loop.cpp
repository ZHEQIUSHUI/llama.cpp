#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache-iswa.h"

#include <cmath>

// Inkling-2-Mini (inkling2_loop): 24-layer MoE trunk with the 12-layer block
// 6..17 applied three times, learned "block attention residual" messages
// between passes, short convolutions and a relative-position bias instead of
// RoPE. The graph follows transformers/models/inkling2_loop exactly.

namespace {

// unweighted RMS over ne0 (matches Inkling2LoopUnweightedRMSNorm)
static ggml_tensor * rms_now(ggml_context * ctx0, ggml_tensor * x, float eps) {
    return ggml_rms_norm(ctx0, x, eps);
}

static ggml_tensor * rms_w(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, float eps) {
    ggml_tensor * y = ggml_rms_norm(ctx0, x, eps);
    if (w) {
        y = ggml_mul(ctx0, y, w);
    }
    return y;
}

static ggml_tensor * l2_clip_impl(ggml_context * ctx0, ggml_tensor * x, float thr, float eps) {
    // y = x * thr / max(||x||_2 + eps, thr)   (== x * clamp(thr/||x||, max=1))
    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
    ggml_tensor * den  = ggml_clamp(ctx0, ggml_scale_bias(ctx0, norm, 1.0f, eps), thr, INFINITY);
    return ggml_div(ctx0, ggml_scale(ctx0, x, thr), den);
}

// custom graph input that fills the per-(query token, kv cell) distance index
// from the live cache positions. `sentinel` (== rel extent) selects the zero
// padding row of the relative-bias projection.
class llm_graph_input_inkling_rel : public llm_graph_input_i {
public:
    llm_graph_input_inkling_rel(const llama_kv_cache_iswa_context * mctx, int32_t extent)
        : mctx(mctx), extent(extent) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (base) {
            mctx->get_base()->set_input_pos_dist(base, ubatch, extent);
        }
        if (swa) {
            mctx->get_swa()->set_input_pos_dist(swa, ubatch, extent);
        }
    }

    ggml_tensor * base = nullptr; // I32 [n_kv_base, n_tokens]
    ggml_tensor * swa  = nullptr; // I32 [n_kv_swa,  n_tokens]

    const llama_kv_cache_iswa_context * mctx;
    const int32_t extent;
};

} // namespace

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key("inkling2_loop.loop.layer_start",      hparams.n_loop_start);
    ml.get_key("inkling2_loop.loop.layer_end",        hparams.n_loop_end);
    ml.get_key("inkling2_loop.loop.passes",           hparams.n_loop_passes);
    ml.get_key("inkling2_loop.loop.gate_rank",        hparams.n_loop_gate_rank);
    ml.get_key("inkling2_loop.attn_rel.dim",          hparams.n_rel_dim);
    ml.get_key("inkling2_loop.attn_rel.extent",       hparams.n_rel_extent);
    ml.get_key("inkling2_loop.attn_rel.clip",         hparams.f_rel_clip);
    ml.get_key("inkling2_loop.attn_qk.clip",          hparams.f_qk_clip);
    ml.get_key("inkling2_loop.log_scale.floor",       hparams.log_scale_floor);
    ml.get_key("inkling2_loop.log_scale.alpha",       hparams.f_log_scale_alpha);
    ml.get_key("inkling2_loop.mlp_gate_softcap",      hparams.f_mlp_gate_softcap);
    ml.get_key("inkling2_loop.mlp_up_softcap",        hparams.f_mlp_up_softcap);
    ml.get_key("inkling2_loop.sconv.kernel",          hparams.n_sconv_kernel);
    ml.get_key("inkling2_loop.attn_res.block",        hparams.n_attn_res_block);

    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,   hparams.expert_gating_func, false);

    // no dense feed_forward_length in this arch: every layer is MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.n_ff_arr[il] = hparams.n_ff_exp_arr[il];
    }

    // SITU == softcapped SwiGLU with beta = gate softcap and lb = up softcap
    hparams.situ_beta        = hparams.f_mlp_gate_softcap;
    hparams.situ_linear_beta = hparams.f_mlp_up_softcap;

    hparams.n_swa     = hparams.n_rel_extent;
    hparams.swa_type  = LLAMA_SWA_TYPE_STANDARD;

    // underlying (weight-bank) layer for each of the expanded cache slots
    const int loop_layers = int(hparams.n_loop_end - hparams.n_loop_start);
    auto slot_layer = [&](int slot) -> int {
        if (slot < int(hparams.n_loop_start) + loop_layers) {
            return slot; // pre-loop and first pass (0..17)
        }
        if (slot < int(hparams.n_loop_start) + loop_layers * int(hparams.n_loop_passes)) {
            return int(hparams.n_loop_start) + (slot - (int(hparams.n_loop_start) + loop_layers)) % loop_layers;
        }
        return int(hparams.n_loop_end) + (slot - (int(hparams.n_loop_start) + loop_layers * int(hparams.n_loop_passes)));
    };

    for (uint32_t s = 0; s < hparams.n_layer(); ++s) {
        const int u = slot_layer(int(s));
        const bool local = u != 5 && u != 11 && u != 17 && u != 23;
        hparams.is_swa_impl[s]  = local ? 1 : 0;
        hparams.is_recr_impl[s] = 1;
    }

    // four short-conv histories of 3 taps each
    hparams.n_embd_r_total = (hparams.n_sconv_kernel - 1) *
        (2 * hparams.n_embd_head_k() * hparams.n_head_kv() + 2 * hparams.n_embd);

    switch (hparams.n_embd) {
        case 2304: type = LLM_TYPE_UNKNOWN; break;
        default:   type = LLM_TYPE_UNKNOWN; break;
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_kv      = n_head_kv * n_embd_head_k;
    const int64_t n_qkv     = n_head * n_embd_head_k;
    const int64_t n_rel_dim = hparams.n_rel_dim;
    const int64_t n_rel_ext = hparams.n_rel_extent;
    const int64_t n_ff_exp  = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);
    inkling_embed_norm  = create_tensor(tn(LLM_TENSOR_INKLING_EMBED_NORM, "weight"), { n_embd }, 0);
    inkling_final_q     = create_tensor(tn(LLM_TENSOR_INKLING_FINAL_Q, "weight", -1), { n_embd }, 0);
    inkling_msg_input_q = create_tensor(tn(LLM_TENSOR_INKLING_MSG_INPUT_Q, "weight", -1),
                                        { n_embd, 4, int(hparams.n_loop_end - hparams.n_loop_start) }, 0);

    const int loop_start_b = int(hparams.n_loop_start);
    const int loop_end_b   = int(hparams.n_loop_end);
    const int loop_layers_b = loop_end_b - loop_start_b;
    const int n_pass_slots  = loop_start_b + loop_layers_b * int(hparams.n_loop_passes); // 42
    const int n_slots       = int(hparams.n_layer());

    // slots [loop_start+loop_layers, n_pass_slots) alias the loop weight bank and
    // are filled after the tensor pass; the post-loop slots [n_pass_slots, ...)
    // own the underlying blocks [loop_end, ...).
    for (int slot = 0; slot < n_slots; ++slot) {
        if (slot >= loop_start_b + loop_layers_b && slot < n_pass_slots) {
            continue; // aliased below
        }
        const int bid  = slot < loop_start_b + loop_layers_b ? slot
                        : loop_end_b + (slot - n_pass_slots);
        const bool full = !hparams.is_swa(slot);
        auto & layer = layers[slot];
        const int k = int(hparams.n_sconv_kernel);

        // full-attention layers use separate q/k/v input norms and have no attn_norm
        layer.attn_norm = full ? nullptr
                               : create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", bid), { n_embd }, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", bid), { n_embd }, 0);

        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), { n_embd, n_qkv }, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), { n_embd, n_kv  }, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), { n_embd, n_kv  }, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", bid), { n_qkv, n_embd }, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", bid), { n_embd, n_qkv }, 0);

        layer.inkling_attn_rel_linear = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_REL_LINEAR, "weight", bid),
                                                      { n_embd, n_head * n_rel_dim }, 0);
        layer.inkling_attn_rel_proj   = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_REL_PROJ, "weight", bid),
                                                      { n_rel_dim, n_rel_ext + 1 }, 0);
        layer.inkling_attn_rel_scale  = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_REL_SCALE, "weight", bid),
                                                      { n_rel_dim, n_head }, 0);
        layer.inkling_attn_q_scale    = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_Q_SCALE, "weight", bid),
                                                      { n_embd_head_k, n_head }, 0);
        layer.inkling_attn_sink_proj  = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_SINK_PROJ, "weight", bid),
                                                      { n_embd, n_head }, 0);

        layer.inkling_attn_conv = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_CONV, "weight", bid), { k, n_embd }, 0);
        layer.inkling_mlp_conv  = create_tensor(tn(LLM_TENSOR_INKLING_MLP_CONV,  "weight", bid), { k, n_embd }, 0);

        if (full) {
            layer.inkling_attn_q_norm = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_Q_NORM, "weight", bid), { n_embd }, 0);
            layer.inkling_attn_k_norm = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_K_NORM, "weight", bid), { n_embd }, 0);
            layer.inkling_attn_v_norm = create_tensor(tn(LLM_TENSOR_INKLING_ATTN_V_NORM, "weight", bid), { n_embd }, 0);
            layer.inkling_k_conv_static = create_tensor(tn(LLM_TENSOR_INKLING_K_CONV_STATIC, "weight", bid),
                                                        { k, n_kv }, 0);
            layer.inkling_v_conv_static = create_tensor(tn(LLM_TENSOR_INKLING_V_CONV_STATIC, "weight", bid),
                                                        { k, n_kv }, 0);
            layer.inkling_k_conv_proj = create_tensor(tn(LLM_TENSOR_INKLING_K_CONV_PROJ, "weight", bid),
                                                      { n_embd, n_head_kv * k }, 0);
            layer.inkling_v_conv_proj = create_tensor(tn(LLM_TENSOR_INKLING_V_CONV_PROJ, "weight", bid),
                                                      { n_embd, n_head_kv * k }, 0);
        } else {
            layer.inkling_k_conv = create_tensor(tn(LLM_TENSOR_INKLING_K_CONV, "weight", bid), { k, n_kv }, 0);
            layer.inkling_v_conv = create_tensor(tn(LLM_TENSOR_INKLING_V_CONV, "weight", bid), { k, n_kv }, 0);
        }

        if (bid > 0) {
            const int n_q = full ? 3 : 1;
            layer.inkling_attn_input_q = create_tensor(
                tn(LLM_TENSOR_INKLING_ATTN_INPUT_Q, "weight", bid),
                n_q == 1 ? std::initializer_list<int64_t>{ n_embd }
                         : std::initializer_list<int64_t>{ n_embd, n_q }, 0);
        }
        layer.inkling_mlp_input_q = create_tensor(tn(LLM_TENSOR_INKLING_MLP_INPUT_Q, "weight", bid), { n_embd }, 0);

        if (bid >= loop_start_b && bid < loop_end_b) {
            layer.inkling_msg_down   = create_tensor(tn(LLM_TENSOR_INKLING_MSG_DOWN,   "weight", bid),
                                                     { 2 * n_embd, 4 * int(hparams.n_loop_gate_rank) }, 0);
            layer.inkling_msg_forget = create_tensor(tn(LLM_TENSOR_INKLING_MSG_FORGET, "weight", bid),
                                                     { int(hparams.n_loop_gate_rank), 4 * n_embd }, 0);
            layer.inkling_msg_gain   = create_tensor(tn(LLM_TENSOR_INKLING_MSG_GAIN,   "weight", bid),
                                                     { int(hparams.n_loop_gate_rank), 4 * n_embd }, 0);
        }

        layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", bid), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "weight", bid), { n_expert }, 0);
        create_tensor_gate_up_exps(layer, bid, n_embd, n_ff_exp, n_expert, 0);
        layer.ffn_down_exps  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", bid),
                                             { n_ff_exp, n_embd, n_expert }, 0);
    }

    // expanded cache slots reuse the loop weight bank (no weight copies)
    for (int slot = loop_start_b + loop_layers_b; slot < n_pass_slots; ++slot) {
        const int src = loop_start_b + (slot - (loop_start_b + loop_layers_b)) % loop_layers_b;
        layers[slot] = layers[src];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------

ggml_tensor * llama_model_inkling2_loop::graph::l2_clip(ggml_tensor * x, float threshold, int il) {
    ggml_tensor * y = l2_clip_impl(ctx0, x, threshold, 1e-6f);
    cb(y, "l2_clip", il);
    return y;
}

ggml_tensor * llama_model_inkling2_loop::graph::log_scale(ggml_tensor * x, ggml_tensor * inp_pos) {
    const float invf = 1.0f / float(hparams.log_scale_floor);
    ggml_tensor * posf = ggml_cast(ctx0, inp_pos, GGML_TYPE_F32);
    ggml_tensor * arg  = ggml_scale_bias(ctx0, posf, invf, invf); // (pos+1)/floor
    arg = ggml_clamp(ctx0, arg, 1.0f, INFINITY);
    ggml_tensor * tau = ggml_scale_bias(ctx0, ggml_log(ctx0, arg), hparams.f_log_scale_alpha, 1.0f);
    tau = ggml_reshape_3d(ctx0, tau, 1, 1, inp_pos->ne[0]);
    return ggml_mul(ctx0, x, tau);
}

ggml_tensor * llama_model_inkling2_loop::graph::block_attn_res(
        ggml_tensor * blocks, ggml_tensor * partial, ggml_tensor * query, int n_q, int il) {
    const int64_t H = partial->ne[0];

    ggml_tensor * p3 = ggml_reshape_3d(ctx0, partial, H, partial->ne[1], 1);
    ggml_tensor * stacked = blocks ? ggml_concat(ctx0, blocks, p3, 2) : p3;
    const int64_t nb = stacked->ne[2];

    ggml_tensor * normed = rms_now(ctx0, stacked, hparams.f_norm_rms_eps);
    cb(normed, "bar_normed", il);

    ggml_tensor * q = query;
    if (q->ne[0] != H || ggml_n_dims(q) < 2) {
        q = ggml_reshape_2d(ctx0, q, H, n_q);
    }
    cb(q, "bar_q", il);

    ggml_tensor * scores = nullptr;
    for (int64_t n = 0; n < nb; ++n) {
        ggml_tensor * sn = ggml_view_3d(ctx0, normed, H, normed->ne[1], 1,
                                        normed->nb[1], normed->nb[2], n * normed->nb[2]);
        ggml_tensor * sc = ggml_mul_mat(ctx0, q, sn); // [n_q, n_tokens, 1]
        sc = ggml_scale(ctx0, sc, 1.0f / sqrtf(float(H)));
        scores = scores ? ggml_concat(ctx0, scores, sc, 2) : sc;
    }
    cb(scores, "bar_scores", il);

    // scores is [n_q, T, nb]; softmax runs over nb (the block axis)
    ggml_tensor * p = ggml_permute(ctx0, scores, 1, 2, 0, 3); // [nb, n_q, T, 1]
    p = ggml_cont(ctx0, p);
    p = ggml_soft_max(ctx0, p);
    p = ggml_permute(ctx0, p, 2, 0, 1, 3); // [n_q, T, nb, 1]
    p = ggml_cont(ctx0, p);
    cb(p, "bar_probs", il);

    ggml_tensor * out = nullptr;
    for (int64_t n = 0; n < nb; ++n) {
        ggml_tensor * pn = ggml_view_3d(ctx0, p, n_q, p->ne[1], 1, p->nb[1], p->nb[2], n * p->nb[2]);
        pn = ggml_reshape_3d(ctx0, pn, 1, n_q, p->ne[1]); // [1, n_q, T]
        pn = ggml_repeat_4d(ctx0, pn, H, n_q, p->ne[1], 1); // [H, n_q, T]
        ggml_tensor * sn = ggml_view_3d(ctx0, normed, H, normed->ne[1], 1,
                                        normed->nb[1], normed->nb[2], n * normed->nb[2]);
        sn = ggml_reshape_3d(ctx0, sn, H, 1, normed->ne[1]); // [H, 1, T]
        ggml_tensor * c = ggml_mul(ctx0, pn, sn);             // [H, n_q, T]
        out = out ? ggml_add(ctx0, out, c) : c;
    }
    // hand back [H, T, n_q] so each per-query slice is contiguous
    out = ggml_permute(ctx0, out, 0, 2, 1, 3); // [H, T, n_q, 1]
    out = ggml_cont(ctx0, out);
    cb(out, "bar_out", il);
    return out; // [H, T, n_q]
}

ggml_tensor * llama_model_inkling2_loop::graph::message_gate(
        ggml_tensor * hidden, ggml_tensor * message,
        ggml_tensor * down, ggml_tensor * forget, ggml_tensor * gain, int il) {
    ggml_tensor * hh = rms_now(ctx0, hidden,  hparams.f_norm_rms_eps);
    ggml_tensor * mm = rms_now(ctx0, message, hparams.f_norm_rms_eps);
    ggml_tensor * cat = ggml_concat(ctx0, hh, mm, 0);
    ggml_tensor * z  = build_lora_mm(down, cat);          // [rank*4, T]
    ggml_tensor * fg = ggml_tanh(ctx0, build_lora_mm(forget, z)); // [H*4, T]
    ggml_tensor * mg = ggml_tanh(ctx0, build_lora_mm(gain,   z));
    // the four controller branches are stacked along ne1; this layer's slice is
    // selected by the caller before the call (see build_layer), so here [H,T]
    ggml_tensor * keep = ggml_scale_bias(ctx0, fg, -1.0f, 1.0f);
    ggml_tensor * out  = ggml_add(ctx0, ggml_mul(ctx0, keep, hidden), ggml_mul(ctx0, mg, message));
    cb(out, "msg_gate", il);
    return out;
}

ggml_tensor * llama_model_inkling2_loop::graph::short_conv(
        ggml_tensor * pre, ggml_tensor * weight, ggml_tensor * static_w, ggml_tensor * wproj,
        ggml_tensor * dyn_input, int n_channels, int conv_idx, int slot, ggml_tensor * r_l,
        ggml_tensor * state_cur, int32_t head, float denom) {
    const int k  = int(hparams.n_sconv_kernel);
    const int st = k - 1;
    const int64_t nb = pre->ne[0];
    const int64_t T  = pre->ne[1];
    GGML_ASSERT(nb == n_channels);

    // state segment inside this slot's conv row:
    //   [k (st*nkv)] [v (st*nkv)] [post-attn (st*embd)] [post-mlp (st*embd)]
    const int64_t nkv_ch = int64_t(hparams.n_embd_head_k()) * hparams.n_head_kv();
    const int64_t seg_off =
        conv_idx == 0 ? 0 :
        conv_idx == 1 ? st * nkv_ch :
        conv_idx == 2 ? 2 * st * nkv_ch :
                        2 * st * nkv_ch + st * int64_t(hparams.n_embd);

    const int64_t row = hparams.n_embd_r();

    // [T, C] view of the pre-conv input (shared as the residual)
    ggml_tensor * xT = ggml_transpose(ctx0, pre);
    xT = ggml_cont_2d(ctx0, xT, T, n_channels);
    cb(xT, "conv_xT", slot);

    // read the previous st inputs for this sequence (stored [st, C])
    ggml_tensor * state = ggml_view_2d(ctx0, state_cur, st, n_channels, st * ggml_element_size(state_cur),
                                       seg_off * ggml_element_size(state_cur));
    cb(state, "conv_state", slot);

    ggml_tensor * cx3 = ggml_concat(ctx0, state, xT, 0); // [st+T, C]
    cx3 = ggml_reshape_3d(ctx0, cx3, st + T, n_channels, 1);
    cb(cx3, "conv_in", slot);

    ggml_tensor * out = nullptr;
    if (weight) {
        out = ggml_ssm_conv(ctx0, cx3, weight); // [C, T, 1]
        out = ggml_reshape_2d(ctx0, out, n_channels, T);
        out = ggml_add(ctx0, out, pre);
        out = ggml_scale(ctx0, out, 1.0f / denom);
    } else {
        // dynamic taps: static_w [k, C] + weight_proj(dyn_input) [n_head*k, T]
        ggml_tensor * dyn = build_lora_mm(wproj, dyn_input); // [n_head_kv*k, T]
        const int64_t n_hkv = hparams.n_head_kv();
        dyn = ggml_reshape_3d(ctx0, dyn, k, n_hkv, T);

        const int64_t per_head = n_channels / n_hkv;
        ggml_tensor * acc = nullptr;
        for (int tap = 0; tap < k; ++tap) {
            ggml_tensor * shifted = ggml_view_2d(ctx0, cx3, T, n_channels, cx3->nb[1], tap * cx3->nb[0]);

            // ggml_ssm_conv computes sum_i c[i] * x[t+i] with x incl. the st-token
            // state, so tap i corresponds to backward distance st-i. The reference
            // stores weight index 0 at distance 0, hence use (st - i).
            const int lag = st - tap;
            ggml_tensor * sw = ggml_view_2d(ctx0, static_w, 1, n_channels, static_w->nb[1],
                                            lag * static_w->nb[0]); // [1, C]

            ggml_tensor * dk = ggml_view_3d(ctx0, dyn, 1, n_hkv, T, dyn->nb[1], dyn->nb[2], lag * dyn->nb[0]);
            dk = ggml_cont(ctx0, dk);                             // [1, n_hkv, T]
            dk = ggml_reshape_2d(ctx0, dk, n_hkv, T);
            dk = ggml_reshape_3d(ctx0, dk, n_hkv, 1, T);
            dk = ggml_repeat_4d(ctx0, dk, n_hkv, per_head, T, 1); // [n_hkv, 128, T]
            dk = ggml_permute(ctx0, dk, 1, 0, 2, 3);              // [128, n_hkv, T]
            dk = ggml_cont(ctx0, dk);
            dk = ggml_reshape_2d(ctx0, dk, n_channels, T);        // [C, T]
            dk = ggml_transpose(ctx0, dk);                        // [T, C]
            dk = ggml_cont(ctx0, dk);                             // binary add needs contiguous dim0

            ggml_tensor * w = ggml_add(ctx0, dk, sw);             // [T, C]
            ggml_tensor * c = ggml_mul(ctx0, shifted, w);
            acc = acc ? ggml_add(ctx0, acc, c) : c;
        }
        acc = ggml_add(ctx0, acc, xT);
        acc = ggml_scale(ctx0, acc, 1.0f / denom);
        out = ggml_transpose(ctx0, acc);
        out = ggml_cont_2d(ctx0, out, n_channels, T);
    }
    cb(out, "conv_out", slot);

    // store the last st pre-conv inputs back into the cache
    ggml_tensor * last = ggml_view_2d(ctx0, cx3, st, n_channels, cx3->nb[1], T * cx3->nb[0]);
    ggml_tensor * dst  = ggml_view_2d(ctx0, r_l, st, n_channels, st * ggml_element_size(r_l),
                                      (head * row + seg_off) * ggml_element_size(r_l));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));
    return out;
}

ggml_tensor * llama_model_inkling2_loop::graph::rel_bias(
        ggml_tensor * q_hidden, ggml_tensor * proj, ggml_tensor * idx,
        bool log_scaling, ggml_tensor * inp_pos, int il, int64_t n_stream) {
    const int64_t T = q_hidden->ne[1];
    const int64_t d_rel = hparams.n_rel_dim;

    ggml_tensor * rs = build_lora_mm(model.layers[il].inkling_attn_rel_linear, q_hidden); // [n_head*d_rel, T]
    rs = ggml_reshape_3d(ctx0, rs, d_rel, hparams.n_head(), T);
    rs = ggml_mul(ctx0, rs, model.layers[il].inkling_attn_rel_scale); // [16, n_head, T]
    rs = l2_clip_impl(ctx0, rs, hparams.f_rel_clip, 1e-6f);
    if (log_scaling) {
        rs = log_scale(rs, inp_pos);
    }
    cb(rs, "rel_states", il);

    ggml_tensor * rs2 = ggml_reshape_2d(ctx0, rs, d_rel, hparams.n_head() * T);
    ggml_tensor * rel = ggml_mul_mat(ctx0, proj, rs2); // [extent, n_head*T]
    // ggml_get_rows(a,b) returns [a->ne0, b->ne0, b->ne1, b->ne2], so make the
    // gathered value dim0 a singleton and index per (key, head, token):
    //   a = [1, extent, n_head, T]  (ggml_permute maps src dim -> dst axis)
    //   b = [n_kv, n_head, T, 1]
    //   -> [1, n_kv, n_head, T] -> reshape -> [n_kv, n_head, T]
    rel = ggml_reshape_4d(ctx0, rel, proj->ne[1], hparams.n_head(), T, 1);
    rel = ggml_permute(ctx0, rel, 1, 2, 3, 0);         // [1, extent, n_head, T]
    cb(rel, "rel_logits", il);

    const int64_t n_kv = idx->ne[0];
    ggml_tensor * idx3 = ggml_reshape_3d(ctx0, idx, n_kv, T, 1);
    idx3 = ggml_repeat_4d(ctx0, idx3, n_kv, T, hparams.n_head(), 1); // [n_kv, T, n_head, 1]
    idx3 = ggml_permute(ctx0, idx3, 0, 2, 1, 3);                     // [n_kv, n_head, T, 1]

    ggml_tensor * bias = ggml_get_rows(ctx0, rel, idx3); // [1, n_kv, n_head, T]
    // kq from ggml_mul_mat(k, q) is [n_kv, T, n_head, n_stream]; reorder the
    // (key, head, token) gather to the same (key, token, head) layout.
    bias = ggml_permute(ctx0, bias, 3, 0, 2, 1);         // [n_kv, T, n_head, 1]
    if (n_stream > 1) {
        // the attention splits the token axis into n_stream contiguous chunks
        GGML_ASSERT(T % n_stream == 0);
        const int64_t tps = T / n_stream;
        bias = ggml_view_4d(ctx0, bias, n_kv, tps, hparams.n_head(), n_stream,
                            bias->nb[1], bias->nb[2], tps * bias->nb[1], 0);
        bias = ggml_cont(ctx0, bias);
    }
    cb(bias, "rel_bias", il);
    return bias;
}

ggml_tensor * llama_model_inkling2_loop::graph::build_layer(
        llm_graph_input_attn_kv_iswa * inp_attn,
        llm_graph_input_rs          * inp_rs,
                ggml_tensor * inp_pos,
                ggml_tensor * rel_idx_base,
                ggml_tensor * rel_idx_swa,
                ggml_tensor * cur,
                ggml_tensor *& blocks,
                ggml_tensor * messages,
                int slot,
                int loop_layer,
                bool is_swa_arch) {
    const auto & layer = model.layers[slot];
    const int64_t H = hparams.n_embd;
    const int64_t T = cur->ne[1];
    const bool full = !is_swa_arch;
    const int ls = int(hparams.n_loop_start);
    const int le = int(hparams.n_loop_end);
    const int nl = le - ls;
    const int ps = ls + nl * int(hparams.n_loop_passes);
    const int u_layer = slot < ls + nl ? slot : (slot < ps ? ls + (slot - (ls + nl)) % nl : le + (slot - ps));
    const bool has_attn_input = u_layer > 0;
    const bool boundary = (u_layer % int(hparams.n_attn_res_block)) == 0;

    ggml_tensor * block_residual = cur;

    ggml_tensor * q_msg = nullptr, * k_msg = nullptr, * v_msg = nullptr, * mlp_msg = nullptr;
    if (messages) {
        auto slice = [&](int j) {
            // messages is [H, T, 4]
            ggml_tensor * m = ggml_view_3d(ctx0, messages, H, T, 1, messages->nb[1], messages->nb[2],
                                           j * messages->nb[2]);
            return ggml_reshape_2d(ctx0, m, H, T);
        };
        q_msg = slice(0); k_msg = slice(1); v_msg = slice(2); mlp_msg = slice(3);
    }

    // controller slices (fused [4*rank, 2H] for down and [4*H, rank] for forget/gain)
    auto ctrl = [&](ggml_tensor * w, int j, int64_t ne0, int64_t ne1) {
        ggml_tensor * v = ggml_view_2d(ctx0, w, ne0, ne1, w->nb[1], j * ne1 * w->nb[1]);
        return v;
    };

    ggml_tensor * q_in = nullptr, * k_in = nullptr, * v_in = nullptr;

    ggml_tensor * ai = nullptr;
    if (has_attn_input) {
        ai = block_attn_res(blocks, cur, layer.inkling_attn_input_q, full ? 3 : 1, slot);
    }

    if (!full) {
        ggml_tensor * h = has_attn_input ? ggml_reshape_2d(ctx0, ai, H, T) : cur;
        if (messages) {
            ggml_tensor * qd = ctrl(layer.inkling_msg_down,   0, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * qf = ctrl(layer.inkling_msg_forget, 0, hparams.n_loop_gate_rank, H);
            ggml_tensor * qg = ctrl(layer.inkling_msg_gain,   0, hparams.n_loop_gate_rank, H);
            ggml_tensor * kd = ctrl(layer.inkling_msg_down,   1, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * kf = ctrl(layer.inkling_msg_forget, 1, hparams.n_loop_gate_rank, H);
            ggml_tensor * kg = ctrl(layer.inkling_msg_gain,   1, hparams.n_loop_gate_rank, H);
            ggml_tensor * vd = ctrl(layer.inkling_msg_down,   2, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * vf = ctrl(layer.inkling_msg_forget, 2, hparams.n_loop_gate_rank, H);
            ggml_tensor * vg = ctrl(layer.inkling_msg_gain,   2, hparams.n_loop_gate_rank, H);
            q_in = rms_w(ctx0, message_gate(h, q_msg, qd, qf, qg, slot), layer.attn_norm, hparams.f_norm_rms_eps);
            k_in = rms_w(ctx0, message_gate(h, k_msg, kd, kf, kg, slot), layer.attn_norm, hparams.f_norm_rms_eps);
            v_in = rms_w(ctx0, message_gate(h, v_msg, vd, vf, vg, slot), layer.attn_norm, hparams.f_norm_rms_eps);
        } else {
            q_in = k_in = v_in = rms_w(ctx0, h, layer.attn_norm, hparams.f_norm_rms_eps);
        }
    } else {
        ggml_tensor * qh, * kh, * vh;
        if (has_attn_input) {
            auto sl = [&](int j) {
                // ai is [H, T, 3]
                ggml_tensor * s = ggml_view_3d(ctx0, ai, H, T, 1, ai->nb[1], ai->nb[2], j * ai->nb[2]);
                return ggml_reshape_2d(ctx0, s, H, T);
            };
            qh = sl(0); kh = sl(1); vh = sl(2);
        } else {
            qh = kh = vh = cur;
        }
        if (messages) {
            ggml_tensor * qd = ctrl(layer.inkling_msg_down,   0, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * qf = ctrl(layer.inkling_msg_forget, 0, hparams.n_loop_gate_rank, H);
            ggml_tensor * qg = ctrl(layer.inkling_msg_gain,   0, hparams.n_loop_gate_rank, H);
            ggml_tensor * kd = ctrl(layer.inkling_msg_down,   1, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * kf = ctrl(layer.inkling_msg_forget, 1, hparams.n_loop_gate_rank, H);
            ggml_tensor * kg = ctrl(layer.inkling_msg_gain,   1, hparams.n_loop_gate_rank, H);
            ggml_tensor * vd = ctrl(layer.inkling_msg_down,   2, 2*H, hparams.n_loop_gate_rank);
            ggml_tensor * vf = ctrl(layer.inkling_msg_forget, 2, hparams.n_loop_gate_rank, H);
            ggml_tensor * vg = ctrl(layer.inkling_msg_gain,   2, hparams.n_loop_gate_rank, H);
            qh = message_gate(qh, q_msg, qd, qf, qg, slot);
            kh = message_gate(kh, k_msg, kd, kf, kg, slot);
            vh = message_gate(vh, v_msg, vd, vf, vg, slot);
        }
        q_in = rms_w(ctx0, qh, layer.inkling_attn_q_norm, hparams.f_norm_rms_eps);
        k_in = rms_w(ctx0, kh, layer.inkling_attn_k_norm, hparams.f_norm_rms_eps);
        v_in = rms_w(ctx0, vh, layer.inkling_attn_v_norm, hparams.f_norm_rms_eps);
    }

    // Q/K/V projections
    ggml_tensor * qlin = build_lora_mm(layer.wq, q_in); // [n_head*hd, T]
    qlin = ggml_reshape_3d(ctx0, qlin, hparams.n_embd_head_k(), hparams.n_head(), T);
    qlin = ggml_mul(ctx0, qlin, layer.inkling_attn_q_scale);
    qlin = l2_clip(qlin, hparams.f_qk_clip, slot);
    if (full) {
        qlin = log_scale(qlin, inp_pos);
    }
    cb(qlin, "Qcur", slot);

    const auto * rctx = inp_rs->mctx;
    const int32_t head = rctx->get_head();
    ggml_tensor * r_l = rctx->get_r_l(slot);
    // gather this sequence's state rows (also makes the rs_s_copy input live)
    ggml_tensor * rs_cur = build_rs(inp_rs, r_l, hparams.n_embd_r(), ubatch.n_seqs);

    ggml_tensor * kpre = build_lora_mm(layer.wk, k_in);
    const float kv_denom = full ? sqrtf(float(2 * int(hparams.n_sconv_kernel) + 1))
                                : sqrtf(float(    int(hparams.n_sconv_kernel) + 1));
    ggml_tensor * kcv  = short_conv(kpre, full ? nullptr : layer.inkling_k_conv,
                                    full ? layer.inkling_k_conv_static : nullptr,
                                    full ? layer.inkling_k_conv_proj : nullptr,
                                    full ? k_in : nullptr,
                                    hparams.n_embd_head_k() * hparams.n_head_kv(), 0, slot, r_l, rs_cur, head,
                                    kv_denom);
    ggml_tensor * Kcur = ggml_reshape_3d(ctx0, kcv, hparams.n_embd_head_k(), hparams.n_head_kv(), T);
    Kcur = l2_clip(Kcur, hparams.f_qk_clip, slot);
    cb(Kcur, "Kcur", slot);

    ggml_tensor * vpre = build_lora_mm(layer.wv, v_in);
    ggml_tensor * vcv  = short_conv(vpre, full ? nullptr : layer.inkling_v_conv,
                                    full ? layer.inkling_v_conv_static : nullptr,
                                    full ? layer.inkling_v_conv_proj : nullptr,
                                    full ? v_in : nullptr,
                                    hparams.n_embd_head_k() * hparams.n_head_kv(), 1, slot, r_l, rs_cur, head,
                                    kv_denom);
    ggml_tensor * Vcur = ggml_reshape_3d(ctx0, vcv, hparams.n_embd_head_k(), hparams.n_head_kv(), T);
    Vcur = rms_now(ctx0, Vcur, hparams.f_norm_rms_eps);
    cb(Vcur, "Vcur", slot);

    // attention with relative bias and per-token sink
    ggml_tensor * idx  = full ? rel_idx_base : rel_idx_swa;
    const int64_t n_stream = (full ? inp_attn->get_kq_mask() : inp_attn->get_kq_mask_swa())->ne[3];
    ggml_tensor * bias = rel_bias(q_in, layer.inkling_attn_rel_proj, idx, full, inp_pos, slot, n_stream);
    ggml_tensor * sinks = build_lora_mm(layer.inkling_attn_sink_proj, q_in); // [n_head, T]
    cb(sinks, "attn_sinks", slot);

    // fold the 1/sqrt(head_dim) attention scaling into q so that the relative
    // bias and sinks are added as unscaled logits (as in the reference, which
    // computes q@k * scaling + position_bias and appends raw sink logits)
    ggml_tensor * qscaled = ggml_scale(ctx0, qlin, 1.0f / sqrtf(float(hparams.n_embd_head_k())));
    ggml_tensor * attn = build_attn(inp_attn, nullptr, nullptr, nullptr,
                                    qscaled, Kcur, Vcur, bias, sinks, nullptr,
                                    1.0f, slot);
    cb(attn, "attn_raw", slot);

    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, q_in));
    ggml_tensor * gated = ggml_mul(ctx0, attn, gate);
    gated = build_lora_mm(layer.wo, gated);
    cb(gated, "attn_out", slot);

    ggml_tensor * post_attn = short_conv(gated, layer.inkling_attn_conv, nullptr, nullptr, nullptr,
                                         H, 2, slot, r_l, rs_cur, head, sqrtf(float(hparams.n_sconv_kernel + 1)));
    cb(post_attn, "post_attn_conv", slot);

    ggml_tensor * residual;
    if (boundary) {
        ggml_tensor * br = ggml_reshape_3d(ctx0, block_residual, H, T, 1);
        blocks = blocks ? ggml_concat(ctx0, blocks, br, 2) : br;
        residual = post_attn;
    } else {
        residual = ggml_add(ctx0, block_residual, post_attn);
    }

    // MLP input via block-attention residual, then message gate
    ggml_tensor * moe_in = block_attn_res(blocks, residual, layer.inkling_mlp_input_q, 1, slot);
    moe_in = ggml_reshape_2d(ctx0, moe_in, H, T);
    if (messages) {
        ggml_tensor * md = ctrl(layer.inkling_msg_down,   3, 2*H, hparams.n_loop_gate_rank);
        ggml_tensor * mf = ctrl(layer.inkling_msg_forget, 3, hparams.n_loop_gate_rank, H);
        ggml_tensor * mg = ctrl(layer.inkling_msg_gain,   3, hparams.n_loop_gate_rank, H);
        moe_in = message_gate(moe_in, mlp_msg, md, mf, mg, slot);
    }
    moe_in = rms_w(ctx0, moe_in, layer.ffn_norm, hparams.f_norm_rms_eps);
    cb(moe_in, "ffn_norm", slot);

    ggml_tensor * moe = build_moe_ffn(moe_in, layer.ffn_gate_inp,
                                      nullptr, nullptr,               // up_exps, gate_exps (fused below)
                                      layer.ffn_down_exps,
                                      layer.ffn_exp_probs_b,
                                      hparams.n_expert, hparams.n_expert_used(),
                                      LLM_FFN_SITU, true, hparams.expert_weights_scale,
                                      LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, slot,
                                      nullptr, layer.ffn_gate_up_exps);
    cb(moe, "ffn_moe_out", slot);

    ggml_tensor * post_mlp = short_conv(moe, layer.inkling_mlp_conv, nullptr, nullptr, nullptr,
                                       H, 3, slot, r_l, rs_cur, head, sqrtf(float(hparams.n_sconv_kernel + 1)));
    cb(post_mlp, "post_mlp_conv", slot);

    ggml_tensor * out = ggml_add(ctx0, residual, post_mlp);
    cb(out, "l_out", slot);
    return out;
}

llama_model_inkling2_loop::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cur = rms_w(ctx0, cur, model.inkling_embed_norm, hparams.f_norm_rms_eps);
    cb(cur, "embed_norm", -1);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_hybrid->get_attn();
    auto * inp_rs   = inp_hybrid->get_recr();

    // relative-distance index inputs, one per cache (full / sliding)
    auto * rel_inp = new llm_graph_input_inkling_rel(inp_attn->mctx, int32_t(hparams.n_rel_extent));
    {
        const int64_t n_kv_base = inp_attn->get_kq_mask()->ne[0];
        const int64_t n_kv_swa  = inp_attn->get_kq_mask_swa()->ne[0];
        rel_inp->base = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_base, inp_pos->ne[0]);
        rel_inp->swa  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_swa,  inp_pos->ne[0]);
        ggml_set_input(rel_inp->base);
        ggml_set_input(rel_inp->swa);
        ggml_set_name(rel_inp->base, "inkling_rel_base");
        ggml_set_name(rel_inp->swa,  "inkling_rel_swa");
    }
    ggml_tensor * rel_idx_base = rel_inp->base;
    ggml_tensor * rel_idx_swa  = rel_inp->swa;
    res->add_input(std::unique_ptr<llm_graph_input_i>(rel_inp));

    ggml_tensor * blocks = nullptr;
    const int loop_start = int(hparams.n_loop_start);
    const int loop_end   = int(hparams.n_loop_end);
    const int loop_layers = loop_end - loop_start;

    auto run_slots = [&](int first_slot, int n, int loop_base, ggml_tensor * messages, bool use_msg) {
        for (int i = 0; i < n; ++i) {
            const int slot = first_slot + i;
            const bool swa = hparams.is_swa(slot);
            ggml_tensor * m = use_msg ? messages : nullptr;
            cur = build_layer(inp_attn, inp_rs, inp_pos, rel_idx_base, rel_idx_swa,
                              cur, blocks, m, slot, loop_base >= 0 ? i : -1, swa);
        }
    };

    // pre-loop 0..5
    run_slots(0, loop_start, -1, nullptr, false);
    ggml_tensor * pre_hidden = cur;
    ggml_tensor * pre_blocks = blocks;

    // pass 1 over 6..17
    run_slots(loop_start, loop_layers, -1, nullptr, false);

    ggml_tensor * prev_hidden = cur;
    ggml_tensor * prev_blocks = nullptr;
    if (blocks) {
        // skip pre-loop blocks + the first block appended by pass 1: keep the last one
        const int64_t nb = blocks->ne[2];
        prev_blocks = ggml_view_3d(ctx0, blocks, hparams.n_embd, blocks->ne[1], 1,
                                   blocks->nb[1], blocks->nb[2], (nb - 1) * blocks->nb[2]);
        prev_blocks = ggml_cont(ctx0, prev_blocks);
    }

    // passes 2..3 restart from the pre-loop state and receive messages from the previous pass
    for (int pass = 1; pass < int(hparams.n_loop_passes); ++pass) {
        ggml_tensor * msg = nullptr;
        if (prev_blocks) {
            // message_input for all loop layers at once: [H, 4, T] per layer is
            // cheaper to build per layer, so keep prev state and build inside.
            msg = nullptr;
        }
        cur = pre_hidden;
        blocks = pre_blocks;
        for (int i = 0; i < loop_layers; ++i) {
            const int slot = loop_start + loop_layers * pass + i;
            ggml_tensor * q = ggml_view_3d(ctx0, model.inkling_msg_input_q, hparams.n_embd, 4, 1,
                                           model.inkling_msg_input_q->nb[1],
                                           model.inkling_msg_input_q->nb[2],
                                           i * model.inkling_msg_input_q->nb[2]);
            q = ggml_reshape_2d(ctx0, q, hparams.n_embd, 4);
            ggml_tensor * messages = block_attn_res(prev_blocks, prev_hidden, q, 4, slot);
            messages = ggml_cont(ctx0, messages); // [H,4,T]
            cur = build_layer(inp_attn, inp_rs, inp_pos, rel_idx_base, rel_idx_swa,
                              cur, blocks, messages, slot, i, hparams.is_swa(slot));
        }
        prev_hidden = cur;
        if (blocks) {
            const int64_t nb = blocks->ne[2];
            prev_blocks = ggml_view_3d(ctx0, blocks, hparams.n_embd, blocks->ne[1], 1,
                                       blocks->nb[1], blocks->nb[2], (nb - 1) * blocks->nb[2]);
            prev_blocks = ggml_cont(ctx0, prev_blocks);
        }
    }

    // post-loop 18..23 -> slots 42..47
    const int n_post = int(hparams.n_layer()) - (loop_start + loop_layers * int(hparams.n_loop_passes));
    for (int u = loop_end; u < loop_end + n_post; ++u) {
        const int slot = loop_start + loop_layers * int(hparams.n_loop_passes) + (u - loop_end);
        cur = build_layer(inp_attn, inp_rs, inp_pos, rel_idx_base, rel_idx_swa,
                          cur, blocks, nullptr, slot, -1, hparams.is_swa(slot));
    }

    // final cross-block attention residual
    cur = block_attn_res(blocks, cur, model.inkling_final_q, 1, -1); // [H, T, 1]
    cur = ggml_reshape_2d(ctx0, cur, hparams.n_embd, cur->ne[1]);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = rms_w(ctx0, cur, model.output_norm, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping)),
                         hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
