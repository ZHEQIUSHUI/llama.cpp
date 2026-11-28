// Inkling-2-Mini vision tower: learned patch projection, class token, pre-LN, 23 layers of
// bidirectional attention with axial 2D RoPE (adjacent pairs), exact GELU MLP, LayerScale,
// then the 2 x 2 merger and the bias-free projector.  On the AXERA backend the whole graph
// runs as one kernel (role vit); this graph is also a correct CPU implementation.
#include "models.h"

ggml_cgraph * clip_graph_inkling2::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.pre_ln_w != nullptr);
    GGML_ASSERT(model.mm_0_w != nullptr && model.mm_2_w != nullptr);
    GGML_ASSERT(n_patches % 4 == 0);

    ggml_tensor * inp = build_inp();  // [n_embd, n_patches] patch projections, raster order

    // patch embeddings into the 2 x 2 merge-block order the tower and merger use
    ggml_tensor * merge_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(merge_idx, "merge_idx");
    ggml_set_input(merge_idx);
    inp = ggml_get_rows(ctx0, inp, merge_idx);
    cb(inp, "patch_merge_order", -1);

    // learned position embedding, bilinear to the image grid (align corners false), same order
    ggml_tensor * pos = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR);
    pos = ggml_get_rows(ctx0, pos, merge_idx);
    inp = ggml_add(ctx0, inp, pos);
    cb(inp, "pos_embed", -1);

    // class token in front of the patches
    ggml_tensor * cls = ggml_reshape_2d(ctx0, model.class_embedding, n_embd, 1);
    inp = ggml_concat(ctx0, cls, inp, 1);
    cb(inp, "class_token", -1);

    const int64_t n_pos = n_patches + 1;
    inp = ggml_reshape_3d(ctx0, inp, n_embd, n_pos, n_batch);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_w);
    ggml_set_input(pos_h);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, 10000.0f, false);
    };

    build_vit_opts opts;
    opts.skip_post_ln = true;  // the tower has only ln_pre
    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr, add_pos, opts);

    // merger: patch rows only, 2 x 2 groups -> 4 * n_embd, projector (no biases)
    cur = ggml_view_2d(ctx0, cur, n_embd, n_pos - 1, cur->nb[1], cur->nb[1]);
    cur = ggml_cont_2d(ctx0, cur, 4 * n_embd, (n_pos - 1) / 4);
    cb(cur, "merger_in", -1);
    cur = build_mm(model.mm_0_w, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cb(cur, "merger_gelu", -1);
    cur = build_mm(model.mm_2_w, cur);
    cb(cur, "projector_out", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
