#include "models.h"

#include <string>

static int64_t div_ceil_signed(int64_t a, int64_t b) {
    return a / b + (a % b > 0 ? 1 : 0);
}

// x: [T, IC], w: [K, IC, OC] -> [T_out, OC]
// The Mimi encoder uses causal convolutions: all fixed padding goes on the left, plus
// the extra right padding that keeps the last partial frame, see MimiConv1d.forward().
ggml_tensor * clip_graph_qwen3tts_enc::conv1d_causal(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                                                     int stride, int dilation, bool replicate) const {
    const int64_t k_size  = (w->ne[0] - 1) * dilation + 1;
    const int64_t p_total = k_size - stride;
    const int64_t n_frames = div_ceil_signed(x->ne[0] - k_size + p_total, stride);
    const int64_t ideal_len = n_frames * stride + k_size - p_total;
    const int64_t p_extra   = ideal_len - x->ne[0];
    GGML_ASSERT(p_extra >= 0);

    if (replicate) {
        if (p_total > 0) {
            ggml_tensor * first = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0);
            ggml_tensor * left  = ggml_repeat_4d(ctx0, first, p_total, x->ne[1], 1, 1);
            x = ggml_concat(ctx0, left, x, 0);
        }
        if (p_extra > 0) {
            ggml_tensor * last = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1],
                    (size_t) (x->ne[0] - 1) * x->nb[0]);
            ggml_tensor * right = ggml_repeat_4d(ctx0, last, p_extra, x->ne[1], 1, 1);
            x = ggml_concat(ctx0, x, right, 0);
        }
    } else {
        x = ggml_pad_ext(ctx0, x, p_total, p_extra, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * y;
    if (w->type == GGML_TYPE_F32) {
        // ggml_conv_1d selects an F16 im2col buffer for non-BF16 kernels. That would
        // silently round the F32 encoder weights, so spell out the same convolution
        // with an F32 im2col when the kernel is F32.
        ggml_tensor * im2col = ggml_im2col(ctx0, w, x, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
        ggml_tensor * cols = ggml_reshape_2d(ctx0, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
        ggml_tensor * kernel = ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]);
        y = ggml_mul_mat(ctx0, cols, kernel);
        y = ggml_reshape_3d(ctx0, y, im2col->ne[1], im2col->ne[2], w->ne[2]);
        y = ggml_cont(ctx0, ggml_permute(ctx0, y, 0, 2, 1, 3));
    } else {
        y = ggml_conv_1d(ctx0, w, x, stride, 0, dilation);
    }
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

ggml_tensor * clip_graph_qwen3tts_enc::resnet_block(ggml_tensor * x, const clip_qwen3tts_enc::resnet & block) const {
    ggml_tensor * residual = x;
    x = ggml_elu(ctx0, x);
    x = conv1d_causal(x, block.res1.w, block.res1.b, 1, 1, false);
    x = ggml_elu(ctx0, x);
    x = conv1d_causal(x, block.res2.w, block.res2.b, 1, 1, false);
    return ggml_add(ctx0, residual, x);
}

// x: [T, 512], returns the RVQ indices for all codebooks as [n_q * T] I32.
ggml_tensor * clip_graph_qwen3tts_enc::quantize_rvq(ggml_tensor * x, ggml_tensor * in_proj,
                                                    const std::vector<ggml_tensor *> & codebooks) {
    // [T, 512] -> [512, T], then the 1x1 input projection is a plain matmul
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * w2 = ggml_reshape_2d(ctx0, in_proj, in_proj->ne[0] * in_proj->ne[1], in_proj->ne[2]);
    cur = ggml_mul_mat(ctx0, w2, cur); // [256, T]

    std::vector<ggml_tensor *> codes;
    codes.reserve(codebooks.size());
    for (ggml_tensor * codebook : codebooks) {
        ggml_tensor * cb_norm = ggml_cont(ctx0, ggml_transpose(ctx0,
                ggml_sum_rows(ctx0, ggml_sqr(ctx0, codebook))));               // [2048, 1]
        ggml_tensor * dot    = ggml_mul_mat(ctx0, codebook, cur);               // [2048, T]
        ggml_tensor * scores = ggml_sub(ctx0, ggml_scale(ctx0, dot, 2.0f), cb_norm);
        ggml_tensor * idx    = ggml_argmax(ctx0, scores);                       // [T]
        codes.push_back(idx);

        ggml_tensor * quant = ggml_get_rows(ctx0, codebook, idx);               // [256, T]
        cur = ggml_sub(ctx0, cur, quant);
    }

    ggml_tensor * out = codes[0];
    for (size_t q = 1; q < codes.size(); q++) {
        out = ggml_concat(ctx0, out, codes[q], 0);
    }
    return out;
}

ggml_cgraph * clip_graph_qwen3tts_enc::build() {
    const int64_t n_samples = img.nx();
    GGML_ASSERT(img.ny() == 1);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    // ---- SEANet encoder: stem, four (residual, /ratio) stages, final conv ----
    auto & enc = model.qwen3tts_enc;
    GGML_ASSERT(enc.seanet_conv.size() == 6);
    GGML_ASSERT(enc.seanet_res.size() == 4);

    ggml_tensor * cur = conv1d_causal(inp, enc.seanet_conv[0].w, enc.seanet_conv[0].b, 1, 1, false);
    static constexpr int RATIOS[4] = { 4, 5, 6, 8 };
    for (int s = 0; s < 4; s++) {
        cur = resnet_block(cur, enc.seanet_res[s]);
        cur = ggml_elu(ctx0, cur);
        cur = conv1d_causal(cur, enc.seanet_conv[s + 1].w, enc.seanet_conv[s + 1].b,
                            RATIOS[s], 1, false);
    }
    cur = ggml_elu(ctx0, cur);
    cur = conv1d_causal(cur, enc.seanet_conv[5].w, enc.seanet_conv[5].b, 1, 1, false); // [T, 512]
    cb(cur, "enc_seanet_out", -1);
    ggml_set_output(cur);

    // ---- causal transformer, sliding-window 250, RoPE theta 10000 ----
    const int64_t n_tok = cur->ne[0];
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * mask = build_inp_attn_mask(n_tok, n_tok);

    ggml_tensor * h = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [512, T]
    const int64_t n_head = 8; // speech_tokenizer/config.json: num_attention_heads
    const int64_t d_head = h->ne[0] / n_head;
    const float   kq_scale = 1.0f / sqrtf((float) d_head);

    for (size_t il = 0; il < enc.layers.size(); il++) {
        const auto & layer = enc.layers[il];
        ggml_tensor * residual = h;

        ggml_tensor * x = ggml_norm(ctx0, h, 1e-5f);
        x = ggml_mul(ctx0, x, layer.attn_norm_w);
        x = ggml_add(ctx0, x, layer.attn_norm_b);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q_w, x);
        ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k_w, x);
        ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v_w, x);
        q = ggml_reshape_4d(ctx0, q, d_head, n_head, n_tok, 1);
        k = ggml_reshape_4d(ctx0, k, d_head, n_head, n_tok, 1);
        v = ggml_reshape_4d(ctx0, v, d_head, n_head, n_tok, 1);

        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                          10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                          10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * attn = build_attn(layer.attn_o_w, nullptr, q, k, v, mask, kq_scale, (int) il);
        attn = ggml_mul(ctx0, attn, ggml_reshape_2d(ctx0, layer.attn_scale, attn->ne[0], 1));
        h = ggml_add(ctx0, residual, attn);

        residual = h;
        x = ggml_norm(ctx0, h, 1e-5f);
        x = ggml_mul(ctx0, x, layer.ffn_norm_w);
        x = ggml_add(ctx0, x, layer.ffn_norm_b);
        x = ggml_mul_mat(ctx0, layer.ffn_up_w, x);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_mul_mat(ctx0, layer.ffn_down_w, x);
        x = ggml_mul(ctx0, x, ggml_reshape_2d(ctx0, layer.ffn_scale, x->ne[0], 1));
        h = ggml_add(ctx0, residual, x);
    }
    cb(h, "enc_tfm_out", -1);
    ggml_set_output(h);

    // ---- downsample 12.5 Hz (k=4, s=2, replicate padding) ----
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [T, 512]
    cur = conv1d_causal(cur, enc.downsample_w, nullptr, 2, 1, true);
    cb(cur, "enc_down_out", -1);
    ggml_set_output(cur);

    // ---- split RVQ: semantic (1 codebook) and acoustic (15 used of 31) ----
    ggml_tensor * sem = quantize_rvq(cur, enc.sem_in_w, { enc.sem_cb_w }); // [T]
    ggml_tensor * aco = quantize_rvq(cur, enc.aco_in_w, enc.aco_cb_w);     // [15 * T]
    ggml_set_name(sem, "enc_sem");
    ggml_set_output(sem);
    ggml_set_name(aco, "enc_aco");
    ggml_set_output(aco);

    // The caller consumes a flat frame-major buffer (16 codes per frame). Build the
    // group-major RVQ stream as [T, 16], then transpose it into a contiguous [16, T]:
    // ggml's first dimension is contiguous, so this yields frame * 16 + codebook.
    ggml_tensor * all = ggml_concat(ctx0, sem, aco, 0); // [16 * T]
    ggml_set_name(all, "enc_all");
    ggml_set_output(all);
    all = ggml_reshape_2d(ctx0, all, cur->ne[0], 16); // [T, 16]
    ggml_tensor * out = ggml_cont(ctx0, ggml_transpose(ctx0, all)); // [16, T]
    ggml_set_name(out, "out_codes");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    return gf;
}
