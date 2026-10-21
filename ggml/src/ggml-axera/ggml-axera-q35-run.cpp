// Axera NPU backend, Qwen3.5 graph template: running a graph

#include "ggml-axera-q35.h"

bool axera_q35_batch::use_split(int R) const {
    return m.kv_llama && m.fd.count(R) && m.fd.at(R).head && (m.flash || kv.W >= axera_split_min());
}

bool axera_q35_batch::pre_split(int v) const {
    return use_split(m.pre[v].P) || (!m.pre[v].attn && !m.pre[v].mc);
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

void axera_q35_batch::run_layers(int v, int t0, int64_t p) {  // v: prefill chunk size index, -1: decode
    for (int il = 0; il < L; ++il) {
        run_layer(il, v, t0, p);
        if (m.moe && v >= 0) {
            run_experts_pre(il, { v });
        } else if (m.moe) {
            run_experts(il, p);
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
        // else one token on the multi-core chain (P = 1)
        for (size_t c = 0; c < m.pre.size() && v < 0; ++c) {
            v = m.pre[c].P == 1 && chunk_fits(c) ? (int) c : -1;
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
        if (ng > 1) {
            const int        P = m.pre[v].P;
            std::vector<int> vs(m.pre[v].group.begin(), m.pre[v].group.begin() + ng);
            for (int c = 0; c < ng; ++c) {
                axera_q35::prefill & pc = m.pre[vs[c]];
                const int            tc = it + c * P;
                for (int r = 0; r < P; ++r) {
                    ggml_fp32_to_bf16_row(x0_row(tc + r),
                                          (ggml_bf16_t *) pc.A->vir + (size_t) r * H, H);
                }
                pc.A->flush(pc.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                rope.rows(tc, P, pc.COS.get(), pc.SIN.get());
                {
                    const int key = axera_q35_fd_key(m, vs[c]);
                    axera_flash_prepare(kv, m.fd.at(key), key % 1000, tc);
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
            const bool           split = !pf.mc && b.pre_split(v);
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
            for (int r = 0; r < P; ++r) {
                ggml_fp32_to_bf16_row(x0_row(it + r),
                                      (ggml_bf16_t *) pf.A->vir + (size_t) r * H, H);
            }
            pf.A->flush(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rope.rows(it, P, pf.COS.get(), pf.SIN.get());
            if (pf.mc) {
                axera_flash_prepare(kv, m.fd.at(P), P, it);
                b.run_mc(v, it);
                auto * mk = &m.mc.at(P);
                pf.A->invalidate(pf.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                mk->PART0->invalidate(mk->PART0->vir, mk->PART0->size);
                const int    C    = (int) (mk->PART0->size / ((size_t) P * H * sizeof(float)));
                const auto * base = (const ggml_bf16_t *) pf.A->vir;
                const auto * part = (const float *) mk->PART0->vir;
                axera_host_pool().run(P, [&](int64_t r) {
                    for (int i = 0; i < H; ++i) {
                        float x = ggml_bf16_to_fp32(base[r * H + i]);
                        for (int c = 0; c < C; ++c) {
                            x += part[((size_t) c * P + r) * H + i];
                        }
                        h[(size_t) (it + r) * H + i] = x;
                    }
                });
                m.next_pos = p + P;
                it += P;
                continue;
            }
            if (split) {
                const int key = axera_q35_fd_key(m, v);
                axera_flash_prepare(kv, m.fd.at(key), key % 1000, it);
            }
            int nblk = (int) ((p + P - 1) / m.BC + 1);
            // chunk mask [N / BC][R][BC], row r = token r % P (of head r / P in older kernels): positions up to
            // p + token, or with the llama.cpp KV cache that token's KQ mask row over the cells
            const int64_t R    = pf.R;
            auto *        mask = pf.MASK ? (ggml_bf16_t *) pf.MASK->vir : nullptr;  // none with split attention
            const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f), ninf = ggml_fp32_to_bf16(-3e38f);
            if (split) {
                // the segments' masks: flash_prepare
            } else if (m.kv_llama) {
                axera_timer tm("mask (cpu)");
                std::vector<ggml_bf16_t> rows((size_t) P * kv.W);
                std::vector<int>         nb(P, 1);
                axera_host_pool().run(P, [&](int64_t r) { nb[r] = kv.mask_row(it + (int) r, rows.data() + (size_t) r * kv.W); });
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
                *(int32_t *) m.POS->vir  = (int32_t) (m.kv_llama ? kv.cell_of(it) : p);
                *(int32_t *) m.NBLK->vir = nblk;
                if (!m.kv_llama) {
                    for (int r = 0; r < P; ++r) {
                        ((ggml_bf16_t *) m.MASK->vir)[p + r] = zero;  // for the decode steps after the chunk
                    }
                    m.MASK->flush(m.MASK->vir, m.MASK->size);
                }
                pf.MASK->flush(pf.MASK->vir, (size_t) nblk * R * m.BC * sizeof(ggml_bf16_t));
                m.POS->flush(m.POS->vir, m.POS->size);
                m.NBLK->flush(m.NBLK->vir, m.NBLK->size);
            }
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
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) it * H, (int64_t) P * H);
            m.next_pos = p + P;
            it += P;
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
        axera_output_head(ctx, cgraph, t, h, T, m.lm_q.get(), m.lm_s.get(), m.lm_bf.get());
    }
    return GGML_STATUS_SUCCESS;
}
