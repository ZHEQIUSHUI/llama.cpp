// Axera NPU backend, Qwen3.5 graph template: running a graph

#include "ggml-axera-q35.h"

// DEPRECATED: the split attention path (flash decoding) is kept only for the models not yet on the fused four-core
// kernels (the Qwen3 family, Qwen3-ASR, Qwen3-TTS, Qwen3.5-0.8B); remove it with their migration. Qwen3.5 2B / 9B / 27B
// and the 35B run fused at every context their kernel directories hold, and stop beyond it
bool axera_q35_batch::use_split(int R) const {
    return m.kv_llama && m.fd.count(R) && m.fd.at(R).head && (m.flash || (!m.attn_all && kv.W >= axera_split_min()));
}

bool axera_q35_batch::pre_split(int v) const {
    return use_split(m.pre[v].P) || !m.pre[v].attn;
}

// the fused attention mask of a prefill chunk of P rows (R == P) holding n real tokens it .. it + n - 1 (n < P: a
// padded chunk, its padding rows the last real token's) from two KQ mask rows: when the tokens are one sequence's, in
// consecutive cells c0 .. c0 + n - 1 at consecutive positions, and the last token's row is the first's with exactly
// those cells visible, token r's row is the first's plus cells c0 + 1 .. c0 + r (llama.cpp's causal mask: the other
// cells are seen alike by all of them). Writes [nblk][R][BC] into mask, false (nothing written) when the chunk is not
// of that kind (GGML_AXERA_MASK_ROWS=1: never)
static bool axera_q35_causal_chunk_mask(const axera_kv_batch & kv, const int32_t * pos, int it, int P, int n, int64_t R,
                                        int BC, ggml_bf16_t * mask, int & nblk) {
    static const bool off = [] {
        const char * e = std::getenv("GGML_AXERA_MASK_ROWS");
        return e && strcmp(e, "1") == 0;
    }();
    if (off || !mask || R != P || n < 2 || n > P) {
        return false;
    }
    axera_timer   tm("mask (cpu, causal chunk)");
    const int64_t c0 = kv.cell_of(it);
    if (c0 + n > kv.W) {
        return false;
    }
    for (int r = 1; r < n; ++r) {
        if (kv.stream_of(it + r) != kv.stream_of(it) || kv.cell_of(it + r) != c0 + r || pos[it + r] != pos[it] + r) {
            return false;
        }
    }
    std::vector<ggml_bf16_t> first(kv.W), last(kv.W);
    kv.mask_row(it, first.data());
    const int nb = kv.mask_row(it + n - 1, last.data());
    const uint16_t z = ggml_fp32_to_bf16(0.0f).bits, ninf = ggml_fp32_to_bf16(-3e38f).bits;
    if (first[c0].bits != z || last[c0].bits != z) {
        return false;
    }
    for (int64_t c = c0 + 1; c < c0 + n; ++c) {
        if (first[c].bits != ninf || last[c].bits != z) {
            return false;
        }
    }
    if (memcmp(first.data(), last.data(), c0 * sizeof(ggml_bf16_t)) != 0 ||
        memcmp(first.data() + c0 + n, last.data() + c0 + n, (kv.W - c0 - n) * sizeof(ggml_bf16_t)) != 0) {
        return false;
    }
    nblk = nb;
    axera_host_pool().run(nblk, [&](int64_t blk) {
        const int64_t b0 = blk * BC;
        for (int64_t r = 0; r < R; ++r) {
            ggml_bf16_t * dst = mask + (blk * R + r) * BC;
            memcpy(dst, first.data() + b0, BC * sizeof(ggml_bf16_t));
            // cells c0 + 1 .. c0 + min(r, n - 1) of this block: visible to row r (the padding rows: the last token's)
            const int64_t lo = std::max<int64_t>(c0 + 1, b0), hi = std::min<int64_t>(c0 + std::min<int64_t>(r, n - 1) + 1, b0 + BC);
            for (int64_t c = lo; c < hi; ++c) {
                dst[c - b0].bits = z;
            }
        }
    });
    return true;
}

void axera_q35_batch::run_layer(int il, int v, int t0, int64_t p) {
    const bool        prefill = v >= 0;
    const bool        split   = prefill ? pre_split(v) : use_split(1);
    axera_q35_layer & ly      = m.layers[il];
    if (split && !ly.gdn) {
        const int key = axera_q35_fd_key(m, v);
        axera_flash_attn(kv, ly.kv, m.fd.at(key), ly.fd[v + 1], v < 0 ? 1 : m.pre[v].P, t0, il, p);
        return;
    }
    auto * in  = prefill ? &ly.pre[v].in : &ly.in;
    auto * out = prefill ? &ly.pre[v].out : &ly.out;
    if (prefill && ly.si >= 0) {  // private ping-pong state: take the current buffers
        (*in)[ly.pre[v].si]  = ly.in[ly.si];
        (*in)[ly.pre[v].ci]  = ly.in[ly.ci];
        (*out)[ly.pre[v].so] = ly.out[ly.so];
        (*out)[ly.pre[v].co] = ly.out[ly.co];
    }
    ggml_axera_kernel * k = ly.gdn ? (prefill ? m.pre[v].gdn : m.gdn) : (prefill ? m.pre[v].attn : m.attn);
    GGML_ASSERT((prefill || !ly.released) && "qwen35 template: a per-layer decode call after its packs were freed");
    if (prefill && ly.gdn && rs_now) {
        // verification chunk: the same call plus the state after each earlier token into snapshot plane j (the
        // state after the chunk's token P - 1 - j), as llama.cpp's own graph writes them for a partial rollback
        auto &    pr = ly.pre[v];
        const int P  = m.pre[v].P;
        pr.rin[pr.rsi]  = (*in)[pr.si];
        pr.rin[pr.rci]  = (*in)[pr.ci];
        pr.rout[pr.rso] = (*out)[pr.so];
        pr.rout[pr.rco] = (*out)[pr.co];
        for (auto & rg : ly.cache_state) {
            for (int j = 1; j < P; ++j) {
                pr.rout[(rg.conv ? pr.rs_conv : pr.rs_state)[j]] =
                    axera_io(rg.b, rg.base + (j * rs_plane + ly.row) * rg.stride, (uint32_t) rg.size);
            }
        }
        in  = &pr.rin;
        out = &pr.rout;
        k   = m.pre[v].gdn_rs;
    }
    // the packed kernel: its packs, the rest of its IO as bound for the unpacked one
    const axera_q35::packed & pk = prefill ? (ly.gdn ? m.pre[v].gdn_pk : m.pre[v].attn_pk) :
                                             (ly.gdn ? m.gdn_pk : m.attn_pk);
    auto &                    pin  = prefill ? ly.pre[v].pin : ly.pin;
    auto &                    pout = prefill ? ly.pre[v].pout : ly.pout;
    const bool                snap = k == (prefill ? m.pre[v].gdn_rs : nullptr) && k != nullptr;
    if (pk.k && !pin.empty() && !snap) {
        for (size_t i = 0; i < pk.in.size(); ++i) {
            if (pk.in[i] >= 0) {
                pin[i] = (*in)[pk.in[i]];
            }
        }
        for (size_t i = 0; i < pk.out.size(); ++i) {
            if (pk.out[i] >= 0) {
                pout[i] = (*out)[pk.out[i]];
            }
        }
        in  = &pin;
        out = &pout;
        k   = pk.k;
    }
    int ret;
    {
        axera_timer tm(ly.gdn ? (prefill ? "gdn prefill" : "gdn decode") : (prefill ? "attn prefill" : "attn decode"));
        ret = axera_run_sync(k, *in, *out);
    }
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d, position %" PRId64 "%s\n", ret, il, p,
                   prefill ? " (prefill)" : "");
    }
    if (ly.si >= 0) {
        std::swap(ly.in[ly.si], ly.out[ly.so]);
        std::swap(ly.in[ly.ci], ly.out[ly.co]);
    }
}

// the graph's logits tensor the whole-model kernel can write (a one-token batch whose graph wants only the logits: no
// normed rows, no MTP input), or nullptr
static ggml_tensor * axera_group_logits(const axera_q35 & m, const std::map<std::string, ggml_tensor *> & t, int T) {
    ggml_tensor * lg = m.grp.lm && T == 1 ? axera_graph_head(t) : nullptr;
    return lg && lg->ne[1] == 1 && lg->type == GGML_TYPE_F32 && ggml_backend_buffer_is_axera(lg->buffer) &&
                   ggml_nbytes(lg) == m.grp.LOGITS->size && !axera_graph_find(t, "h_nextn") &&
                   axera_graph_find(t, "result_output") == lg ? lg : nullptr;
}

void axera_q35_batch::run_layers(int v, int t0, int64_t p) {  // v: prefill chunk size index, -1: decode
    if (v < 0 && !m.moe && m.grp.k && !use_split(1)) {  // G layers per call
        axera_q35::group & g = m.grp;
        for (int gi = 0; gi < L / g.G; ++gi) {
            auto & in  = g.gin[gi];
            auto & out = g.gout[gi];
            for (size_t i = 0; i < g.in.size(); ++i) {
                if (g.in[i].layer >= 0) {
                    in[i] = m.layers[gi * g.G + g.in[i].layer].in[g.in[i].idx];
                }
            }
            for (size_t i = 0; i < g.out.size(); ++i) {
                if (g.out[i].layer >= 0) {
                    out[i] = m.layers[gi * g.G + g.out[i].layer].out[g.out[i].idx];
                } else {  // the logits: into the graph's tensor, or LOGITS to copy from
                    out[i] = g.dst ? axera_io((ggml_backend_axera_buffer_context *) g.dst->buffer->context, g.dst->data,
                                              (uint32_t) ggml_nbytes(g.dst)) :
                                     axera_io(g.LOGITS.get(), g.LOGITS->vir, (uint32_t) g.LOGITS->size);
                }
            }
            int ret;
            {
                axera_timer tm("group decode");
                ret = axera_run_sync(g.k, in, out);
            }
            if (ret != 0) {
                GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer group %d, position %" PRId64 "\n", ret, gi, p);
            }
        }
        g.ran = g.lm;
        return;
    }
    for (int il = 0; il < L; ++il) {
        if (v < 0 && m.moe && grouped(il)) {
            run_grouped(il, p);
            il += m.mgrp.G - 1;
            continue;
        }
        if (v < 0 && m.moe && whole(il)) {
            run_whole(il, p);
            continue;
        }
        if (v >= 0 || !m.moe || !merged(il)) {  // else it ran with layer il - 1's experts
            run_layer(il, v, t0, p);
        }
        if (m.moe && v >= 0) {
            run_experts_pre(il, { v });
        } else if (m.moe) {
            run_experts(il, p, merged(il + 1) ? il + 1 : -1);
        }
    }
}

void axera_q35_batch::run_group(const std::vector<int> & vs, int t0, int64_t p) {
    const int P = m.pre[vs[0]].P;
    for (int il = 0; il < L; ++il) {
        for (size_t c = 0; c < vs.size(); ++c) {
            run_layer(il, vs[c], t0 + (int) c * P, p + (int64_t) c * P);
        }
        run_experts_pre(il, vs);
    }
}

enum ggml_status axera_q35_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph) {
    const std::map<std::string, ggml_tensor *> & t = axera_graph_names(ctx, cgraph);
    auto find = [&](const char * name) { return axera_graph_find(t, name); };
    ggml_tensor * norm0  = find("attn_norm-0");
    ggml_tensor * logits = axera_graph_head(t);
    ggml_tensor * inp_pos = find("inp_pos");
    // pooled embeddings: no output projection, the graph split ends at the final norm (the pooling runs on the CPU)
    const bool    embd   = !logits && axera_graph_norm(t);
    // MTP draft graph (llama.cpp LLM_GRAPH_TYPE_DECODER_MTP): its h input and the model layer of the NextN block
    const ggml_tensor * mtp_h  = find("mtp_h_input");
    int                 mtp_il = -1;
    for (const auto & kv : mtp_h ? t : std::map<std::string, ggml_tensor *>{}) {
        const std::string suffix = ".nextn.eh_proj.weight";
        if (kv.first.rfind("blk.", 0) == 0 && kv.first.size() > suffix.size() &&
            kv.first.compare(kv.first.size() - suffix.size(), suffix.size(), suffix) == 0) {
            mtp_il = atoi(kv.first.c_str() + 4);
        }
    }
    if (mtp_h && (mtp_il < 0 || mtp_h->type != GGML_TYPE_F32)) {
        GGML_ABORT("qwen35 template: MTP graph without a NextN block\n");
    }
    if ((!norm0 && !mtp_h) || (!logits && !embd) || !inp_pos) {
        GGML_LOG_ERROR("qwen35 template: split of %d nodes (%s ... %s) lacks:%s%s%s\n", cgraph->n_nodes,
                       cgraph->n_nodes ? cgraph->nodes[0]->name : "", cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1]->name : "",
                       norm0 ? "" : " attn_norm-0", logits ? "" : " result_output", inp_pos ? "" : " inp_pos");
        GGML_ABORT("qwen35 template: the graph split does not hold the whole model; is the model Qwen3.5 and are all "
                   "layers offloaded (-ngl 99)?\n");
    }

    // layer 0 input: the source of the first RMS norm (MTP: eh_proj of the embedding and h rows, computed below)
    ggml_tensor * x0 = mtp_h ? nullptr : norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(mtp_h || (x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0])));
    // a layer-0 input computed in this split (llama.cpp's mixed input, the token embeddings set into rows of the given
    // embeddings: Qwen3-TTS): its nodes on the CPU
    if (x0 && x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < cgraph->n_nodes && cgraph->nodes[n0] != x0) {
            ++n0;
        }
        axera_cpu_compute(ctx, cgraph, 0, n0, "input (cpu)");
    }
    x0 = x0 ? x0->src[0] : nullptr;
    const int H = (int) (mtp_h ? mtp_h->ne[0] : x0->ne[0]);
    const int T = (int) (mtp_h ? mtp_h->ne[1] : x0->ne[1]);

    ggml_tensor * lm = logits ? logits->src[0] : nullptr;
    GGML_ASSERT(!logits || (logits->op == GGML_OP_MUL_MAT && lm->ne[0] == H && ggml_backend_buffer_is_axera(logits->buffer)));

    int L = mtp_h ? 1 : 0;
    while (!mtp_h && t.count("blk." + std::to_string(L) + ".attn_norm.weight")) {
        ++L;
    }
    if (!ctx->q35) {
        ctx->q35 = std::make_shared<axera_q35>();
    }
    axera_q35 &   m   = *ctx->q35;
    const int     off = mtp_h ? mtp_il : 0;
    ggml_tensor * w0  = find(("blk." + std::to_string(off) + ".attn_norm.weight").c_str());
    if (!m.ready || m.L != L || m.bound != w0->data || m.lm != lm || m.off != off || m.mtp != (mtp_h != nullptr)) {
        m.mtp = mtp_h != nullptr;
        m.off = off;
        const ggml_tensor * rope = nullptr;
        for (int i = 0; i < cgraph->n_nodes && !rope; i++) {
            rope = cgraph->nodes[i]->op == GGML_OP_ROPE ? cgraph->nodes[i] : nullptr;
        }
        GGML_ASSERT(rope && "qwen35 template: no ROPE node in the graph");
        axera_q35_init(m, L, H, t, lm, rope);
        m.bound = w0->data;
        axera_q35_restart(m);
        m.eh_bf.reset();
        m.EH.reset();
        if (m.mtp) {
            m.eh_bf = axera_to_bf16(t.at("blk." + std::to_string(off) + ".nextn.eh_proj.weight"));
            m.EH    = axera_cmm_alloc(H * sizeof(float));
        }
    }

    // MTP: layer 0's input rows eh_proj([enorm(embedding of the token) | hnorm(h)]) (the norms on the host, the
    // projection [2H] -> [H] on the NPU)
    std::vector<float> x0m;
    if (m.mtp) {
        const std::string   nn = std::to_string(off);
        // the token rows: the embedding of token ids (get_rows), or embeddings given as input (warm-up, images)
        const ggml_tensor * te   = find(("mtp_tok_embd-" + nn).c_str());
        const bool          ids  = te && te->op == GGML_OP_GET_ROWS && te->src[1]->type == GGML_TYPE_I32;
        if (!te || (!ids && (te->type != GGML_TYPE_F32 || te->ne[0] != H || te->ne[1] != T))) {
            GGML_ABORT("qwen35 template: MTP input neither token ids nor embedding rows\n");
        }
        const ggml_tensor * ew  = ids ? te->src[0] : te;
        const ggml_tensor * enw = t.at("blk." + nn + ".nextn.enorm.weight");
        const ggml_tensor * hnw = t.at("blk." + nn + ".nextn.hnorm.weight");
        const ggml_tensor * en  = find(("mtp_enorm-" + nn).c_str());
        const ggml_tensor * hn  = find(("mtp_hnorm-" + nn).c_str());
        GGML_ASSERT(en && hn && enw->type == GGML_TYPE_F32 && hnw->type == GGML_TYPE_F32);
        auto rms_src = [](const ggml_tensor * mul) {  // the RMS_NORM operand of norm * weight
            return mul->src[0]->op == GGML_OP_RMS_NORM ? mul->src[0] : mul->src[1];
        };
        const float eps_e = axera_rms_eps(rms_src(en)), eps_h = axera_rms_eps(rms_src(hn));
        const auto * tt   = ggml_get_type_traits(ew->type);
        GGML_ASSERT((!ids || tt->to_float) && "qwen35 template: MTP embedding type without to_float");
        x0m.resize((size_t) T * H);
        std::vector<float> cat(2 * (size_t) H);
        auto normalize = [&](float * v, const float * w, float eps) {
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                ss += (double) v[i] * v[i];
            }
            const float sc = 1.0f / sqrtf((float) (ss / H) + eps);
            for (int i = 0; i < H; ++i) {
                v[i] = v[i] * sc * w[i];
            }
        };
        for (int r = 0; r < T; ++r) {
            if (ids) {
                const int32_t tok = ((const int32_t *) te->src[1]->data)[r];
                tt->to_float((const char *) ew->data + (size_t) tok * ew->nb[1], cat.data(), H);
            } else {
                memcpy(cat.data(), (const char *) te->data + (size_t) r * te->nb[1], H * sizeof(float));
            }
            normalize(cat.data(), (const float *) enw->data, eps_e);
            memcpy(cat.data() + H, (const char *) mtp_h->data + (size_t) r * mtp_h->nb[1], H * sizeof(float));
            normalize(cat.data() + H, (const float *) hnw->data, eps_h);
            axera_gemv_raw_to(ctx, m.eh_bf.get(), m.eh_bf->vir, H, 2 * H, cat.data(), m.EH.get(), (float *) m.EH->vir);
            memcpy(x0m.data() + (size_t) r * H, m.EH->vir, H * sizeof(float));
        }
    }
    auto x0_row = [&](int r) -> const float * {  // layer 0's input row of token r (f32)
        return m.mtp ? x0m.data() + (size_t) r * H : (const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]);
    };

    // the layers: chunks of P consecutive positions through the prefill kernels, single tokens through the decode ones
    const int32_t *    pos = (const int32_t *) inp_pos->data;
    std::vector<float> h((size_t) T * H);
    // with the llama.cpp KV cache: the cells of the new tokens and llama.cpp's KQ mask over the cells
    const ggml_tensor * k_idxs = m.kv_llama ? find("attn_inp_k_idxs") : nullptr;
    const ggml_tensor * kqm    = m.kv_llama ? find("attn_inp_kq_mask") : nullptr;
    if (m.kv_llama) {
        GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm &&
                    (kqm->ne[3] == 1 ? kqm->ne[1] >= T : kqm->ne[1] * kqm->ne[3] == T) &&
                    kqm->ne[0] <= m.N && (kqm->type == GGML_TYPE_F16 || kqm->type == GGML_TYPE_F32));
    }
    axera_q35_batch b(ctx, m, L, H, T);
    if (m.kv_llama) {
        b.kv = axera_kv_batch(k_idxs, kqm, m.N, m.BC, m.S);
    }
    const axera_kv_batch & kv = b.kv;

    // sequences of the batch (llama.cpp splits recurrent batches into n_seqs equal runs of n_seq_tokens tokens) and
    // the recurrent-cache bookkeeping of this graph: rs_head (row of the first sequence: the lowest row the state
    // copies write), rs_z (row to clear, from the zeroing SCALE), s_copy (row i of the batch <- row s_copy[i])
    int64_t n_seqs = 1, rs_head = 0, rs_z = -1;
    int64_t rs_written = 1;  // recurrent-state snapshots this graph writes (speculative decoding)
    bool    in_place = !m.layers.empty();
    for (auto & ly : m.layers) {
        in_place = in_place && (!ly.gdn || ly.in_place);
    }
    int g0 = 0;
    while (g0 < L && !m.layers[g0].gdn) {
        ++g0;
    }
    const ggml_tensor * s0 = g0 < L && in_place ? find(("cache_s_l" + std::to_string(g0)).c_str()) : nullptr;
    if (s0) {
        int64_t best = -1;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != s0 || ggml_nelements(node) == 0) {
                continue;
            }
            if (node->op == GGML_OP_CPY && (best < 0 || (int64_t) node->view_offs < best)) {
                // [state, n_seqs] (or with n_rs_seq: [state, n_seqs, snapshots], snapshot planes nb[2] apart)
                best       = (int64_t) node->view_offs;
                n_seqs     = node->ne[1];
                rs_written = node->ne[2];
                b.rs_plane = node->ne[2] > 1 ? (int64_t) (node->nb[2] / s0->nb[1]) : 0;
            } else if (node->op == GGML_OP_SCALE) {
                rs_z = (int64_t) (node->view_offs / s0->nb[1]);
            }
        }
        rs_head = best < 0 ? 0 : best / (int64_t) s0->nb[1];
        n_seqs  = std::max<int64_t>(1, n_seqs);
    }
    if (T % n_seqs != 0 || (n_seqs > 1 && !(m.kv_llama && in_place))) {
        GGML_LOG_ERROR("qwen35 template: %" PRId64 " sequences in a batch of %d need the recurrent state and KV cache in "
                       "the llama.cpp memory\n", n_seqs, T);
        return GGML_STATUS_FAILED;
    }
    const int64_t n_seq_tokens = T / n_seqs;
    if (s0) {
        if (rs_z >= 0) {
            axera_q35_zero_row(m, rs_z);
        }
        const ggml_tensor * sc = find("rs_s_copy");
        bool                identity = true;
        for (int64_t i = 0; sc && i < sc->ne[0]; ++i) {
            identity = identity && ((const int32_t *) sc->data)[i] == rs_head + i;
        }
        if (!identity) {  // move states between rows (sequence copies, cell reuse): row rs_head + i <- s_copy[i]
            for (auto & ly : m.layers) {
                for (auto & rg : ly.cache_state) {
                    std::vector<char> tmp((size_t) sc->ne[0] * rg.stride);
                    for (int64_t i = 0; i < sc->ne[0]; ++i) {
                        char * src = rg.base + (int64_t) ((const int32_t *) sc->data)[i] * rg.stride;
                        rg.b->flush(src, rg.stride);
                        rg.b->invalidate(src, rg.stride);
                        memcpy(tmp.data() + i * rg.stride, src, rg.stride);
                    }
                    for (int64_t i = 0; i < sc->ne[0]; ++i) {
                        char * dst = rg.base + (rs_head + i) * rg.stride;
                        memcpy(dst, tmp.data() + i * rg.stride, rg.stride);
                        rg.b->flush(dst, rg.stride);
                    }
                }
            }
        }
    }
    auto row_of = [&](int t) { return rs_head + t / n_seq_tokens; };
    auto begin = [&](int64_t p) {  // accept position p as the next one; false fails the batch
        if (m.kv_llama) {
            // llama.cpp owns the KV cells and, with the recurrent state in its cache, the state of every position it
            // continues from; only a new sequence needs its recurrent state cleared
            if (p == 0) {
                axera_q35_zero_row(m, m.layers.empty() ? 0 : m.layers[0].row);
            }
            m.next_pos = p;
            return true;
        }
        if (p == 0) {
            axera_q35_restart(m);
        } else if (m.rollback && p < m.next_pos) {
            // back to an earlier position: llama.cpp has restored the recurrent cache (a context checkpoint) to the
            // state after position p - 1, and dropped the later attention cells; hide them from the attention kernel
            auto * mask = (ggml_bf16_t *) m.MASK->vir;
            for (int64_t j = p; j < m.next_pos; ++j) {
                mask[j] = ggml_fp32_to_bf16(-3e38f);
            }
            m.next_pos = p;
        }
        if (p != m.next_pos || p >= m.LMAX) {
            // fail the batch, not the process: the next request starting at position 0 recovers
            GGML_LOG_ERROR("qwen35 template: position %" PRId64 " (expected %" PRId64 ", context %d). The template runs a "
                           "single sequence with consecutive positions; with llama-server use -np 1 --cache-ram 0%s\n",
                           p, m.next_pos, m.LMAX, m.rollback ? "" : " --ctx-checkpoints 0");
            m.next_pos = -1;  // only a restart at position 0 is accepted now
            return false;
        }
        return true;
    };
    const axera_rope rope(inp_pos, T, m.ROT, m.freq_scale, m.inv_freq, m.rope_mode, m.sections);

    // GGML_AXERA_PAD_CHUNK=0: no padded chunks; GGML_AXERA_PAD_MIN: the fewest tokens one runs (default 2: a single
    // token takes the decode kernels)
    static const bool pad_env = [] {
        const char * e = std::getenv("GGML_AXERA_PAD_CHUNK");
        return e == nullptr || strcmp(e, "0") != 0;
    }();
    static const int pad_min = [] {
        const char * e = std::getenv("GGML_AXERA_PAD_MIN");
        return e ? std::max(1, atoi(e)) : 2;
    }();
    // GGML_AXERA_PAD_FILL=zero: padding rows of zeros instead (a check: the real rows' results do not depend on them)
    static const bool pad_zero = [] {
        const char * e = std::getenv("GGML_AXERA_PAD_FILL");
        return e && strcmp(e, "zero") == 0;
    }();
    // only with a single sequence (every llama.cpp context of the process made with n_seq_max 1): the cells after the
    // batch's then belong to no other sequence
    const bool pad_chunks = pad_env && !m.hybrid && !m.moe && axera_n_seq_max() == 1;

    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        if (s0) {
            axera_q35_set_row(m, row_of(it));
        }
        if (m.kv_llama && !m.flash) {
            axera_q35_set_stream(m, kv.stream_of(it));
        }
        if (!begin(p)) {
            return GGML_STATUS_FAILED;
        }
        int v = -1;  // the largest prefill chunk that fits the remaining consecutive positions
        auto chunk_fits = [&](size_t c) {
            const int P  = m.pre[c].P;
            bool      ok = !m.pre[c].batch && !m.pre[c].sub && it + P <= T && (m.kv_llama || p + P <= m.LMAX) &&
                      it % n_seq_tokens + P <= n_seq_tokens;
            // on the llama.cpp KV cache the cells must be consecutive, not the positions (M-RoPE: an image's tokens)
            for (int j = 1; ok && j < P; ++j) {
                ok = m.kv_llama ? kv.stream_of(it + j) == kv.stream_of(it) && kv.cell_of(it + j) == kv.cell_of(it) + j :
                                  pos[it + j] == p + j;
            }
            return ok && (!m.kv_llama || kv.cell_of(it) + P <= m.N);
        };
        // fewer consecutive tokens left than the largest chunk holds (a prompt's tail, or a short prompt): a pure
        // attention model (no recurrent state a padding row would advance) runs them as one chunk of the smallest
        // size that holds them, padded with copies of the last row. The padding rows' K / V go to the cells after the
        // batch's: cells no token sees (the last token's KQ mask hides them, or they lie past the mask) and the next
        // tokens overwrite before any reads them; else, or past the cache, the chunks that fit exactly as before
        int pad_n = 0;  // the real tokens of a padded chunk
        if (pad_chunks) {
            int run = 1;  // consecutive tokens from it (cells or positions), within its sequence
            while (it + run < T && it % n_seq_tokens + run < n_seq_tokens &&
                   (m.kv_llama ? kv.stream_of(it + run) == kv.stream_of(it) && kv.cell_of(it + run) == kv.cell_of(it) + run :
                                 pos[it + run] == p + run)) {
                ++run;
            }
            int best = -1;  // the smallest chunk that holds run, if run is below the largest one (chunks largest first)
            for (size_t c = 0; c < m.pre.size(); ++c) {
                const int P = m.pre[c].P;
                if (P <= 1 || m.pre[c].batch || m.pre[c].sub || m.pre[c].group.size() > 1 || b.pre_split((int) c)) {
                    continue;
                }
                if (P < run) {
                    break;
                }
                best = (int) c;
            }
            best = best >= 0 && m.pre[best].P == run ? -1 : best;  // a chunk fits exactly: as before
            if (best >= 0 && run >= pad_min) {
                const int     P   = m.pre[best].P;
                const int64_t c0  = m.kv_llama ? kv.cell_of(it) : p;
                bool          ok  = c0 + P <= (m.kv_llama ? m.N : m.LMAX);
                if (ok && m.kv_llama) {
                    std::vector<ggml_bf16_t> last(kv.W);
                    kv.mask_row(it + run - 1, last.data());
                    const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;
                    for (int64_t c = c0 + run; ok && c < std::min<int64_t>(c0 + P, kv.W); ++c) {
                        ok = last[c].bits == ninf;
                    }
                }
                if (ok) {
                    v     = best;
                    pad_n = run;
                }
            }
        }
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            v = m.pre[c].P > 1 && chunk_fits(c) ? (int) c : -1;
        }
        // else one token of each of the next P sequences (rows rs_head + it ..) in one KV stream: batched decode
        for (size_t c = 0; c < m.pre.size() && v < 0 && n_seq_tokens == 1 && s0; ++c) {
            const int P  = m.pre[c].P;
            bool      ok = m.pre[c].batch && it + P <= T;
            for (int j = 1; ok && j < P; ++j) {
                ok = kv.stream_of(it + j) == kv.stream_of(it);
            }
            v = ok ? (int) c : -1;
        }
        // a MoE group: as many of its chunks as fit, consecutive
        int ng = 1;
        while (v >= 0 && ng < (int) m.pre[v].group.size()) {
            const int P = m.pre[v].P, n = ng + 1;
            bool      ok = it + n * P <= T && it % n_seq_tokens + n * P <= n_seq_tokens &&
                      (m.kv_llama ? kv.cell_of(it) + n * P <= m.N : p + n * P <= m.LMAX);
            for (int j = ng * P; ok && j < n * P; ++j) {
                ok = m.kv_llama ? kv.stream_of(it + j) == kv.stream_of(it) && kv.cell_of(it + j) == kv.cell_of(it) + j :
                                  pos[it + j] == p + j;
            }
            if (!ok) {
                break;
            }
            ng = n;
        }
        // a fused attention chunk's inputs: its mask, the cell to write and the KV blocks (pf's own for the copies in a
        // MoE group, else the decode IO's)
        // n: the chunk's real tokens (a padded chunk: fewer than P; its padding rows take the last real one's mask)
        auto fused_inputs = [&](axera_q35::prefill & pf, int it, int64_t p, bool split, int n) {
            const int P = pf.P;
            int nblk = (int) ((p + P - 1) / m.BC + 1);
            // chunk mask [N / BC][R][BC], row r = token r % P (of head r / P in older kernels): positions up to
            // p + token, or with the llama.cpp KV cache that token's KQ mask row over the cells
            const int64_t R    = pf.R;
            auto *        mask = pf.MASK ? (ggml_bf16_t *) pf.MASK->vir : nullptr;  // none with split attention
            const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f), ninf = ggml_fp32_to_bf16(-3e38f);
            if (split) {
                // the segments' masks: flash_prepare
            } else if (m.kv_llama && axera_q35_causal_chunk_mask(kv, pos, it, P, n, R, m.BC, mask, nblk)) {
                // (one sequence's consecutive cells and positions: the chunk's rows from its first and last)
            } else if (m.kv_llama) {
                axera_timer tm("mask (cpu)");
                std::vector<ggml_bf16_t> rows((size_t) P * kv.W);
                std::vector<int>         nb(P, 1);
                axera_host_pool().run(P, [&](int64_t r) {
                    nb[r] = kv.mask_row(it + std::min((int) r, n - 1), rows.data() + (size_t) r * kv.W);
                });
                nblk = *std::max_element(nb.begin(), nb.end());
                axera_host_pool().run(nblk, [&](int64_t blk) {
                    for (int64_t r = 0; r < R; ++r) {
                        memcpy(mask + (blk * R + r) * m.BC, rows.data() + (size_t) (r % P) * kv.W + blk * m.BC,
                               m.BC * sizeof(ggml_bf16_t));
                    }
                });
            } else {
                for (int64_t blk = 0; blk < nblk; ++blk) {
                    for (int64_t r = 0; r < R; ++r) {
                        const int64_t lim = p + r % P;
                        ggml_bf16_t * row = mask + (blk * R + r) * m.BC;
                        for (int c = 0; c < m.BC; ++c) {
                            row[c] = blk * m.BC + c <= lim ? zero : ninf;
                        }
                    }
                }
            }
            if (!split) {
                auto & pos_b = pf.CPOS ? pf.CPOS : m.POS;
                auto & nblk_b = pf.CNBLK ? pf.CNBLK : m.NBLK;
                *(int32_t *) pos_b->vir  = (int32_t) (m.kv_llama ? kv.cell_of(it) : p);
                *(int32_t *) nblk_b->vir = nblk;
                if (!m.kv_llama) {
                    for (int r = 0; r < n; ++r) {
                        ((ggml_bf16_t *) m.MASK->vir)[p + r] = zero;  // for the decode steps after the chunk
                    }
                    m.MASK->flush(m.MASK->vir, m.MASK->size);
                }
                pf.MASK->flush(pf.MASK->vir, (size_t) nblk * R * m.BC * sizeof(ggml_bf16_t));
                pos_b->flush(pos_b->vir, pos_b->size);
                nblk_b->flush(nblk_b->vir, nblk_b->size);
            }
        };
        if (ng > 1) {
            const int        P = m.pre[v].P;
            std::vector<int> vs(m.pre[v].group.begin(), m.pre[v].group.begin() + ng);
            for (int c = 0; c < ng; ++c) {
                axera_q35::prefill & pc = m.pre[vs[c]];
                const int            tc = it + c * P;
                axera_host_pool().run(P, [&](int64_t r) {
                    ggml_fp32_to_bf16_row(x0_row(tc + (int) r), (ggml_bf16_t *) pc.A->vir + (size_t) r * H, H);
                });
                pc.A->flush(pc.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                rope.rows(tc, P, pc.COS.get(), pc.SIN.get());
                if (b.pre_split(vs[c])) {
                    const int key = axera_q35_fd_key(m, vs[c]);
                    axera_flash_prepare(kv, m.fd.at(key), key % 1000, tc);
                } else {
                    fused_inputs(pc, tc, p + (int64_t) c * P, false, P);
                }
            }
            b.run_group(vs, it, p);
            for (int c = 0; c < ng; ++c) {
                auto * fin = (L % 2 == 0 ? m.pre[vs[c]].A : m.pre[vs[c]].B).get();
                fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) (it + c * P) * H, (int64_t) P * H);
            }
            m.next_pos = p + (int64_t) ng * P;
            it += ng * P;
            continue;
        }
        if (v >= 0) {
            axera_q35::prefill & pf    = m.pre[v];
            const int            P     = pf.P;
            const int            n     = pad_n > 0 ? pad_n : P;  // real tokens
            const bool           split = b.pre_split(v);
            if (pf.batch) {
                axera_q35_set_batch_rows(m, v, row_of(it));
                for (int r = 1; r < P; ++r) {  // token it's row was cleared by begin()
                    if (pos[it + r] == 0) {
                        axera_q35_zero_row(m, row_of(it + r));
                    }
                }
                for (int r = 0; r < P && !split; ++r) {
                    ((int32_t *) pf.POS->vir)[r] = (int32_t) kv.cell_of(it + r);
                }
                if (!split) {
                    pf.POS->flush(pf.POS->vir, pf.POS->size);
                }
            }
            axera_host_pool().run(P, [&](int64_t r) {  // 0.5M elements for 9B P = 128: ~1 ms on one core
                auto * dst = (ggml_bf16_t *) pf.A->vir + (size_t) r * H;
                if (r >= n && pad_zero) {
                    memset(dst, 0, H * sizeof(ggml_bf16_t));
                } else {
                    ggml_fp32_to_bf16_row(x0_row(it + std::min((int) r, n - 1)), dst, H);
                }
            });
            pf.A->flush(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rope.rows(it, n, pf.COS.get(), pf.SIN.get());
            if (n < P) {  // the padding rows' rotary rows: the last real token's
                for (auto * cs : { pf.COS.get(), pf.SIN.get() }) {
                    auto * r0 = (ggml_bf16_t *) cs->vir;
                    for (int r = n; r < P; ++r) {
                        if (pad_zero) {
                            memset(r0 + (size_t) r * (m.ROT / 2), 0, (m.ROT / 2) * sizeof(ggml_bf16_t));
                        } else {
                            memcpy(r0 + (size_t) r * (m.ROT / 2), r0 + (size_t) (n - 1) * (m.ROT / 2), (m.ROT / 2) * sizeof(ggml_bf16_t));
                        }
                    }
                    cs->flush(cs->vir, cs->size);
                }
            }
            if (split) {
                const int key = axera_q35_fd_key(m, v);
                axera_flash_prepare(kv, m.fd.at(key), key % 1000, it);
            }
            fused_inputs(pf, it, p, split, n);
            // speculative decoding: a verification batch (all T <= snapshot count tokens in this chunk) also writes the
            // recurrent-state snapshots; without the snapshot kernel a rollback would restore wrong states
            b.rs_now = rs_written > 1 && it == 0 && P == T && rs_written == T && pf.gdn_rs != nullptr;
            if (rs_written > 1 && rs_written == T && !b.rs_now) {
                GGML_LOG_ERROR("qwen35 template: a %d-token batch needs recurrent-state snapshots (speculative decoding, "
                               "n_rs_seq); no qwen35_gdn_prefill%d*_rs kernel runs it in one chunk\n", T, T);
                return GGML_STATUS_FAILED;
            }
            b.run_layers(v, it, p);
            b.rs_now = false;
            auto * fin = (L % 2 == 0 ? pf.A : pf.B).get();
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            axera_host_pool().run(n, [&](int64_t r) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
            });
            m.next_pos = p + n;
            it += n;
            continue;
        }
        ggml_fp32_to_bf16_row(x0_row(it), (ggml_bf16_t *) m.A->vir, H);
        m.A->flush(m.A->vir, H * sizeof(ggml_bf16_t));
        rope.rows(it, 1, m.COS.get(), m.SIN.get());
        if (b.use_split(1)) {
            axera_flash_prepare(kv, m.fd.at(1), 1, it);
        } else {
            if (m.kv_llama) {
                *(int32_t *) m.NBLK->vir = kv.mask_row(it, (ggml_bf16_t *) m.MASK->vir);
                *(int32_t *) m.POS->vir  = (int32_t) kv.cell_of(it);
            } else {
                *(int32_t *) m.POS->vir                = (int32_t) p;
                *(int32_t *) m.NBLK->vir               = (int32_t) (p / m.BC + 1);
                ((ggml_bf16_t *) m.MASK->vir)[p]       = ggml_fp32_to_bf16(0.0f);
            }
            m.POS->flush(m.POS->vir, m.POS->size);
            m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            m.MASK->flush(m.MASK->vir, (size_t) *(int32_t *) m.NBLK->vir * m.BC * sizeof(ggml_bf16_t));  // blocks read
            // cs / pn of the packed kernels: cos | sin, pos | nblk
            memcpy(m.CS->vir, m.COS->vir, m.COS->size);
            memcpy((char *) m.CS->vir + m.COS->size, m.SIN->vir, m.SIN->size);
            m.CS->flush(m.CS->vir, m.CS->size);
            ((int32_t *) m.PN->vir)[0] = *(int32_t *) m.POS->vir;
            ((int32_t *) m.PN->vir)[1] = *(int32_t *) m.NBLK->vir;
            m.PN->flush(m.PN->vir, m.PN->size);
        }
        static const bool direct = [] {
            const char * e = std::getenv("GGML_AXERA_GROUP_DIRECT");  // 0: the logits through LOGITS, copied
            return e == nullptr || strcmp(e, "0") != 0;
        }();
        m.grp.dst = direct && !embd ? axera_group_logits(m, t, T) : nullptr;
        if (m.grp.dst) {  // no dirty lines left to be written back over the NPU's logits
            ((ggml_backend_axera_buffer_context *) m.grp.dst->buffer->context)->flush(m.grp.dst->data, ggml_nbytes(m.grp.dst));
        }
        {
            axera_timer tly("layers (decode)");
            b.run_layers(-1, it, p);
        }
        auto * fin = (L % 2 == 0 ? m.A : m.B).get();  // where the last layer writes
        fin->invalidate(fin->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, H);
        m.next_pos = p + 1;
        ++it;
    }
    if (embd) {
        axera_output_norm(t, h, T);
    } else {
        // the whole-model kernel computed the logits of this (single-token) batch: written in place (or copied) unless
        // the graph wants more than the logits (normed rows, MTP input)
        ggml_tensor * lg = m.grp.ran ? axera_group_logits(m, t, T) : nullptr;
        if (lg && lg == m.grp.dst) {  // written in place: drop the cached lines, get_tensor reads without maintenance
            axera_timer tl("lm_head (group)");
            auto * dctx = (ggml_backend_axera_buffer_context *) lg->buffer->context;
            dctx->invalidate(lg->data, ggml_nbytes(lg));
            dctx->clean_p     = lg->data;
            dctx->clean_n     = ggml_nbytes(lg);
            dctx->clean_epoch = axera_graph_epoch;
        } else if (lg) {
            axera_timer tl("lm_head (group)");
            m.grp.LOGITS->invalidate(m.grp.LOGITS->vir, m.grp.LOGITS->size);
            memcpy(lg->data, m.grp.LOGITS->vir, ggml_nbytes(lg));
            ((ggml_backend_axera_buffer_context *) lg->buffer->context)->flush(lg->data, ggml_nbytes(lg));
        } else {
            axera_output_head(ctx, cgraph, t, h, T, m.lm_q.get(), m.lm_s.get(), m.lm_bf.get());
        }
        m.grp.ran = false;
        m.grp.dst = nullptr;
        // nodes after the logits in this split: llama.cpp's backend sampler chain (-bs: top-k, argmax, ...) on the
        // logits, on the CPU backend (one pass over the vocabulary instead of the host sampler's copy and sorts)
        const ggml_tensor * head = axera_graph_head(t);
        int                 il   = -1;
        for (int i = 0; i < cgraph->n_nodes && head; i++) {
            il = cgraph->nodes[i] == head ? i : il;
        }
        const ggml_tensor * padded = axera_graph_find(t, "result_output");  // Qwen3-TTS: padded logits, done above
        if (il >= 0 && il + 1 < cgraph->n_nodes && (!padded || padded == head)) {
            axera_timer ts("sampler nodes (cpu)");
            axera_cpu_compute_nodes(ctx, cgraph, cgraph->nodes + il + 1, cgraph->n_nodes - il - 1, "sampler (cpu)");
        }
    }
    return GGML_STATUS_SUCCESS;
}
