// Axera NPU backend, Qwen3.5 graph template: the routed experts of the MoE layers (35B-A3B)

#include "ggml-axera-q35.h"

// MoE layer il after its layer kernel: the top-K experts of the router logits (softmax, top-K, renormalized: as
// transformers' Qwen3_5MoeTopKRouter; the softmax over the top-K alone gives the same weights), their weights
// bound to the experts kernel, which adds them to the residual + shared expert (BASE) into the layer output
bool axera_q35_batch::whole(int il) const {
    if (il >= L || !m.layers[il].exps[0].planar) {
        return false;
    }
    const auto & ly = m.layers[il];
    return m.full[ly.gdn ? 0 : 1].k && (ly.gdn || !use_split(1));
}

// a whole MoE layer in one call: the layer kernel's IO by name, the layer's planar expert tensors, out the next x
void axera_q35_batch::run_whole(int il, int64_t p) {
    axera_q35_layer & ly = m.layers[il];
    auto &            fk = m.full[ly.gdn ? 0 : 1];
    for (size_t i = 0; i < fk.in.size(); ++i) {
        if (fk.body[i] >= 0) {
            fk.in[i] = ly.in[fk.body[i]];
            continue;
        }
        if (fk.pack[i] >= 0) {
            const auto & pk = ly.packs[fk.pack[i]];
            fk.in[i]        = axera_io(pk.get(), pk->vir, (uint32_t) fk.pack_bytes[fk.pack[i]]);
            continue;
        }
        const auto & e = ly.exps[fk.plane[i] < 2 ? 0 : 2];  // gate-up, down
        fk.in[i] = fk.plane[i] % 2 == 0 ? axera_io(e.b, e.base, (uint32_t) (e.n * e.data)) :
                                          axera_io(e.b, e.s(0), (uint32_t) (e.n * e.scale));
    }
    auto & x = il % 2 == 0 ? m.B : m.A;  // as the experts kernel writes it
    for (size_t i = 0; i < fk.out.size(); ++i) {
        fk.out[i] = fk.bout[i] >= 0 ? ly.out[fk.bout[i]] : axera_io(x.get(), x->vir, (uint32_t) (m.H * sizeof(ggml_bf16_t)));
    }
    axera_timer tm(ly.gdn ? "moe layer gdn (whole)" : "moe layer attn (whole)");
    const int   ret = axera_run_sync(fk.k, fk.in, fk.out);
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the whole MoE layer %d, position %" PRId64 "\n", ret, il, p);
    }
    if (ly.si >= 0) {
        std::swap(ly.in[ly.si], ly.out[ly.so]);
        std::swap(ly.in[ly.ci], ly.out[ly.co]);
    }
}

bool axera_q35_batch::grouped(int il) const {
    const int G = m.mgrp.G;
    if (!m.mgrp.k || il % G != 0 || il + G > L) {
        return false;
    }
    for (int j = 0; j < G; ++j) {
        if (!whole(il + j)) {
            return false;
        }
    }
    return true;
}

// layers il .. il + G - 1 in one call: their IO by name with the layer index (layer kernels' buffers), the group's
// packs, out the x of the group's last layer (as its whole-layer kernel writes it)
void axera_q35_batch::run_grouped(int il, int64_t p) {
    auto & gk = m.mgrp;
    auto   io = [&](const axera_q35::moe_group_kernel::io & d, bool out) -> AX_ENGINE_IO_BUFFER_T {
        axera_q35_layer & ly = m.layers[il + d.j];
        switch (d.kind) {
            case 0:
                return ly.in[(ly.gdn ? m.gdn : m.attn)->input("x")];
            case 1: {
                const auto & pk = gk.packs[il / gk.G][d.idx];
                return axera_io(pk.get(), pk->vir, (uint32_t) gk.pack_bytes[d.idx]);
            }
            case 2: {
                const auto & e = ly.exps[d.idx < 2 ? 0 : 2];  // gate-up, down
                return d.idx % 2 == 0 ? axera_io(e.b, e.base, (uint32_t) (e.n * e.data)) :
                                        axera_io(e.b, e.s(0), (uint32_t) (e.n * e.scale));
            }
            case 3:
                return out ? ly.out[d.idx] : ly.in[d.idx];
            default: {
                auto & x = (il + gk.G - 1) % 2 == 0 ? m.B : m.A;
                return axera_io(x.get(), x->vir, (uint32_t) (m.H * sizeof(ggml_bf16_t)));
            }
        }
    };
    for (size_t i = 0; i < gk.in.size(); ++i) {
        gk.in[i] = io(gk.ins[i], false);
    }
    for (size_t i = 0; i < gk.out.size(); ++i) {
        gk.out[i] = io(gk.outs[i], true);
    }
    axera_timer tm("moe layer group");
    const int   ret = axera_run_sync(gk.k, gk.in, gk.out);
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the MoE layers %d .. %d, position %" PRId64 "\n", ret, il,
                   il + gk.G - 1, p);
    }
    for (int j = 0; j < gk.G; ++j) {
        axera_q35_layer & ly = m.layers[il + j];
        if (ly.si >= 0) {
            std::swap(ly.in[ly.si], ly.out[ly.so]);
            std::swap(ly.in[ly.ci], ly.out[ly.co]);
        }
    }
}

bool axera_q35_batch::merged(int il) const {
    if (il == 0 || il >= L || !m.layers[il - 1].exps_gu || whole(il) || whole(il - 1)) {
        return false;
    }
    const auto & ly = m.layers[il];
    return m.mx[ly.gdn ? 0 : 1].k && (ly.gdn || !use_split(1));
}

void axera_q35_batch::run_experts(int il, int64_t p, int next) {
    axera_q35_layer & ly = m.layers[il];
    axera_timer       tr("moe route (cpu)");
    m.LOGITS->invalidate(m.LOGITS->vir, m.LOGITS->size);
    const float * lg = (const float *) m.LOGITS->vir;
    std::iota(e_ids.begin(), e_ids.end(), 0);
    std::partial_sort(e_ids.begin(), e_ids.begin() + m.top_k, e_ids.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    auto * rw  = (float *) m.RW->vir;
    float  sum = 0.0f;
    for (int i = 0; i < m.top_k; ++i) {
        rw[i] = std::exp(lg[e_ids[i]] - lg[e_ids[0]]);
        sum += rw[i];
    }
    for (int i = 0; i < m.top_k; ++i) {
        rw[i] /= sum;
    }
    m.RW->flush(rw, m.top_k * sizeof(float));
    auto & ek = m.ex[ly.exps_gu];
    for (int i = 0; i < m.top_k; ++i) {
        for (int g = 0; g < 3; ++g) {
            if (ly.exps_gu && g == 1) {
                continue;
            }
            const int    kind = ly.exps_gu && g == 0 ? 3 : g;
            const auto & e    = ly.exps[g];
            ek.in[ek.w[kind][i]] = axera_io(e.b, e.w(e_ids[i]), (uint32_t) e.data);
            ek.in[ek.s[kind][i]] = axera_io(e.b, e.s(e_ids[i]), (uint32_t) e.scale);
        }
    }
    auto & out = ek.out[il % 2];
    tr.stop();
    if (next >= 0) {  // with layer next's body: its IO, the experts' IO for the rest
        axera_q35_layer & ln = m.layers[next];
        auto &            mk = m.mx[ln.gdn ? 0 : 1];
        for (size_t i = 0; i < mk.in.size(); ++i) {
            mk.in[i] = mk.body[i] >= 0 ? ln.in[mk.body[i]] : ek.in[mk.ein[i]];
        }
        for (size_t i = 0; i < mk.out.size(); ++i) {
            mk.out[i] = ln.out[mk.bout[i]];
        }
        axera_timer tm(ln.gdn ? "moe experts + gdn decode" : "moe experts + attn decode");
        const int   ret = axera_run_sync(mk.k, mk.in, mk.out);
        if (ret != 0) {
            GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the experts of layer %d with layer %d, position %"
                       PRId64 "\n", ret, il, next, p);
        }
        if (ln.si >= 0) {
            std::swap(ln.in[ln.si], ln.out[ln.so]);
            std::swap(ln.in[ln.ci], ln.out[ln.co]);
        }
        return;
    }
    axera_timer tm("moe experts");
    const int   ret = axera_run_sync(ek.k, ek.in, out);
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the experts of layer %d, position %" PRId64 "\n", ret,
                   il, p);
    }
}

// MoE layer il of prefill chunks vs (a group, or one chunk) after their layer kernels: each token's top-K experts
// (as run_experts), the picks grouped by expert into slots of up to G rows (h2 rows and routing weights gathered
// here), S slots per call of the slots kernel, then per token base + its K results into the next layer's input
// the group's experts with the rows read and added on the NPU: the host routes (top-K of the logits), lays the picks
// out in slots of up to G rows per expert and writes each slot's expert and tokens and each token's K slot rows; the
// slots kernel reads its rows out of the group's h2, the combine kernel adds them to the base into the next x
void axera_q35_batch::run_experts_px(int il, const std::vector<int> & vs) {
    axera_q35_layer & ly = m.layers[il];
    const int         NT0 = m.pre[vs[0]].P * (int) vs.size();
    // the slots kernel with the fewest rows per slot that still holds an expert's average picks (less padding)
    auto *            skp = &m.slots;
    for (auto it = m.slots_px.rbegin(); it != m.slots_px.rend(); ++it) {
        if ((int64_t) NT0 * m.top_k <= (int64_t) it->G * m.n_expert) {
            skp = &*it;
        }
    }
    auto &            sk = *skp;
    const int         P = m.pre[vs[0]].P, NT = P * (int) vs.size(), K = m.top_k, NE = m.n_expert, S = sk.S, G = sk.G;
    const size_t      es = sk.ob ? sizeof(ggml_bf16_t) : sizeof(float);
    axera_timer       tr("moe prefill route (cpu)");
    axera_timer       tk("moe prefill route: top-k (cpu)");
    m.GLOG->invalidate(m.GLOG->vir, (size_t) NT * NE * sizeof(float));
    const float *      lgs = (const float *) m.GLOG->vir;
    std::vector<int>   ids((size_t) NT * K);
    std::vector<float> wts((size_t) NT * K);
    // per token (a block of tokens per worker): its top-K by a running insertion into K slots (NE >> K), the weights
    // softmax-renormalized over the K (as transformers' Qwen3_5MoeTopKRouter)
    GGML_ASSERT(K <= 16);
    const int nb = std::max(1, NT / 64);
    axera_host_pool().run(nb, [&](int64_t b) {
        for (int64_t r = b * NT / nb; r < (b + 1) * NT / nb; ++r) {
            const float * lg = lgs + r * NE;
            int           top[16];
            float         val[16];
            int           n = 0;
            for (int e = 0; e < NE; ++e) {
                const float v = lg[e];
                if (n == K && v <= val[K - 1]) {
                    continue;
                }
                int j = n < K ? n++ : K - 1;
                for (; j > 0 && val[j - 1] < v; --j) {  // ties: the lower expert first, as a stable sort
                    val[j] = val[j - 1];
                    top[j] = top[j - 1];
                }
                val[j] = v;
                top[j] = e;
            }
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                ids[r * K + i] = top[i];
                wts[r * K + i] = std::exp(val[i] - val[0]);
                sum += wts[r * K + i];
            }
            for (int i = 0; i < K; ++i) {
                wts[r * K + i] /= sum;
            }
        }
    });
    tk.stop();
    axera_timer tl("moe prefill route: layout (cpu)");
    // picks grouped by expert (counting sort, picks in token order within an expert), then the slots: per expert
    // ceil(picks / G) of them; built in cached memory and copied to the non-cached TOK / SRW / LOC in bulk
    std::vector<int> cnt(NE + 1, 0), order((size_t) NT * K);
    for (int pk = 0; pk < NT * K; ++pk) {
        cnt[ids[pk] + 1]++;
    }
    std::vector<int> first(NE + 1, 0), slot0(NE + 1, 0);
    for (int e = 0; e < NE; ++e) {
        first[e + 1] = first[e] + cnt[e + 1];
        slot0[e + 1] = slot0[e] + (cnt[e + 1] + G - 1) / G;
    }
    {
        std::vector<int> fill(first.begin(), first.end() - 1);
        for (int pk = 0; pk < NT * K; ++pk) {
            order[fill[ids[pk]]++] = pk;
        }
    }
    const int ns    = slot0[NE];
    const int calls = (ns + S - 1) / S;
    GGML_ASSERT((size_t) calls * S * G <= (size_t) m.comb.R);
    // the combine kernel of the fewest tokens that holds the group's (the rows of a group's first chunks)
    auto * cbp = &m.comb;
    for (auto it = m.comb_small.rbegin(); it != m.comb_small.rend(); ++it) {
        if (NT <= it->T) {
            cbp = &*it;
        }
    }
    auto & cb = *cbp;
    std::vector<int32_t> tok_h((size_t) calls * S * (1 + G), 0), loc_h((size_t) K * cb.T, 0);
    std::vector<float>   sw_h((size_t) calls * S * G, 0.0f);  // padding slots / rows: expert 0, token 0, weight 0
    axera_host_pool().run(NE, [&](int64_t e) {
        for (int j = 0; j < cnt[e + 1]; ++j) {
            const int sl = slot0[e] + j / G, g = j % G, pk = order[first[e] + j];
            const size_t row = (size_t) sl * G + g;
            tok_h[(size_t) sl * (1 + G)]         = (int32_t) e;
            tok_h[(size_t) sl * (1 + G) + 1 + g] = pk / K;
            sw_h[row]                             = wts[pk];
            loc_h[(size_t) (pk % K) * cb.T + pk / K] = (int32_t) row;  // [K][T]
        }
    });
    memcpy(m.TOK->vir, tok_h.data(), tok_h.size() * sizeof(int32_t));
    memcpy(m.SRW->vir, sw_h.data(), sw_h.size() * sizeof(float));
    memcpy(m.LOC->vir, loc_h.data(), loc_h.size() * sizeof(int32_t));
    auto * tok = (int32_t *) m.TOK->vir;
    auto * sw  = (float *) m.SRW->vir;
    tl.stop();
    tr.stop();
    const auto & ec = ly.exps[0];
    const auto & ed = ly.exps[2];
    sk.in[sk.h2]       = axera_io(m.GH2.get(), m.GH2->vir, (uint32_t) m.GH2->size);
    sk.in[sk.plane[0]] = axera_io(ec.b, ec.base, (uint32_t) (ec.n * ec.data));
    sk.in[sk.plane[1]] = axera_io(ec.b, ec.s(0), (uint32_t) (ec.n * ec.scale));
    sk.in[sk.plane[2]] = axera_io(ed.b, ed.base, (uint32_t) (ed.n * ed.data));
    sk.in[sk.plane[3]] = axera_io(ed.b, ed.s(0), (uint32_t) (ed.n * ed.scale));
    for (int c = 0; c < calls; ++c) {
        const size_t r0 = (size_t) c * S * G;
        sk.in[sk.tok]   = axera_io(m.TOK.get(), tok + (size_t) c * S * (1 + G), (uint32_t) (S * (1 + G) * sizeof(int32_t)));
        sk.in[sk.rw]    = axera_io(m.SRW.get(), sw + r0, (uint32_t) (S * G * sizeof(float)));
        sk.outs[sk.out] = axera_io(m.SOUT.get(), (char *) m.SOUT->vir + r0 * H * es, (uint32_t) (S * G * H * es));
        axera_timer tm("moe prefill slots");
        const int   ret = axera_run_sync(sk.k, sk.in, sk.outs);
        if (ret != 0) {
            GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the prefill experts of layer %d\n", ret, il);
        }
    }
    auto & xo = il % 2 == 0 ? m.GB : m.GA;  // the group's next x (its first cb.T rows)
    const uint32_t xb = (uint32_t) ((size_t) cb.T * H * sizeof(ggml_bf16_t));
    cb.in.assign(cb.k->info->nInputSize, {});
    cb.out.assign(cb.k->info->nOutputSize, {});
    cb.in[cb.k->input("base")] = axera_io(m.GBASE.get(), m.GBASE->vir, xb);
    cb.in[cb.k->input("loc")]  = axera_io(m.LOC.get(), m.LOC->vir, (uint32_t) ((size_t) K * cb.T * sizeof(int32_t)));
    cb.in[cb.k->input("sout")] = axera_io(m.SOUT.get(), m.SOUT->vir, axera_input_size(cb.k, "sout"));
    cb.out[cb.k->output("out")] = axera_io(xo.get(), xo->vir, xb);
    axera_timer tc("moe prefill combine (npu)");
    const int   ret = axera_run_sync(cb.k, cb.in, cb.out);
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the prefill combine of layer %d\n", ret, il);
    }
}

void axera_q35_batch::run_experts_pre(int il, const std::vector<int> & vs) {
    axera_q35_layer & ly = m.layers[il];
    if (m.comb.k && ly.exps[0].planar) {  // a group's chunks (or its first) in order: their rows are contiguous
        bool ok = vs.size() <= m.pre[0].group.size();
        for (size_t c = 0; ok && c < vs.size(); ++c) {
            ok = vs[c] == m.pre[0].group[c];
        }
        if (ok) {
            run_experts_px(il, vs);
            return;
        }
    }
    auto &            sk = m.slots;
    const int         P = m.pre[vs[0]].P, NT = P * (int) vs.size(), K = m.top_k, NE = m.n_expert, S = sk.S, G = sk.G;
    const size_t      es = sk.ob ? sizeof(ggml_bf16_t) : sizeof(float);  // bytes per result element
    GGML_ASSERT(ly.exps_gu);
    axera_timer tr("moe prefill route (cpu)");
    for (int v : vs) {
        const auto & pf = m.pre[v];
        pf.LOG->invalidate(pf.LOG->vir, pf.LOG->size);
        pf.H2->invalidate(pf.H2->vir, pf.H2->size);
        pf.BASE->invalidate(pf.BASE->vir, pf.BASE->size);
    }
    auto row_ptr = [&](const axera_cmm axera_q35::prefill::*buf, int t, size_t width) {  // token t of the group
        return (const char *) (m.pre[vs[t / P]].*buf)->vir + (size_t) (t % P) * width;
    };
    std::vector<int>   ids((size_t) NT * K);
    std::vector<float> wts((size_t) NT * K);
    axera_host_pool().run(NT, [&](int64_t r) {
        const float *    lg = (const float *) row_ptr(&axera_q35::prefill::LOG, (int) r, NE * sizeof(float));
        std::vector<int> o(NE);
        std::iota(o.begin(), o.end(), 0);
        std::partial_sort(o.begin(), o.begin() + K, o.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            ids[r * K + i] = o[i];
            wts[r * K + i] = std::exp(lg[o[i]] - lg[o[0]]);
            sum += wts[r * K + i];
        }
        for (int i = 0; i < K; ++i) {
            wts[r * K + i] /= sum;
        }
    });
    axera_timer tg("moe prefill route: gather (cpu)");
    std::vector<std::vector<int>> by(NE);  // picks (token * K + i) per expert
    for (int pk = 0; pk < NT * K; ++pk) {
        by[ids[pk]].push_back(pk);
    }
    std::vector<int> se, sp;  // slot expert, slot rows' picks (-1: unused)
    for (int e = 0; e < NE; ++e) {
        for (size_t j = 0; j < by[e].size(); j += G) {
            se.push_back(e);
            for (int g = 0; g < G; ++g) {
                sp.push_back(j + g < by[e].size() ? by[e][j + g] : -1);
            }
        }
    }
    const int calls = (int) (se.size() + S - 1) / S;
    GGML_ASSERT((size_t) calls * S * G * H * sizeof(ggml_bf16_t) <= m.SX->size);
    se.resize((size_t) calls * S, se[0]);  // padding slots: some expert on unused rows
    sp.resize((size_t) calls * S * G, -1);
    // a slot's picks fill its first rows; the other rows keep stale data (weight 0, results never read), so only
    // the used rows are written and kept coherent
    std::vector<int> loc((size_t) NT * K);  // pick -> slot row
    std::vector<int> used((size_t) calls * S, 0);
    auto *           sx = (ggml_bf16_t *) m.SX->vir;
    auto *           sw = (float *) m.SRW->vir;
    const bool       px = sk.tok >= 0;  // the slots kernel reads rows of h2 by token: the chunks' h2 rows into SX
    auto *           tk = px ? (int32_t *) m.TOK->vir : nullptr;
    if (px) {
        GGML_ASSERT((size_t) calls * S * (1 + G) * sizeof(int32_t) <= m.TOK->size);
        for (int t = 0; t < NT; ++t) {
            memcpy(sx + (size_t) t * H, row_ptr(&axera_q35::prefill::H2, t, H * sizeof(ggml_bf16_t)),
                   H * sizeof(ggml_bf16_t));
        }
        m.SX->flush(sx, (size_t) NT * H * sizeof(ggml_bf16_t));
    }
    axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
        if (px) {
            tk[sl * (1 + G)] = se[sl];
        }
        for (int g = 0; g < G; ++g) {
            const int64_t row = sl * G + g;
            const int     pk  = sp[row];
            sw[row]           = pk >= 0 ? wts[pk] : 0.0f;
            if (px) {
                tk[sl * (1 + G) + 1 + g] = pk >= 0 ? pk / K : 0;
            }
            if (pk >= 0) {
                if (!px) {
                    memcpy(sx + row * H, row_ptr(&axera_q35::prefill::H2, pk / K, H * sizeof(ggml_bf16_t)),
                           H * sizeof(ggml_bf16_t));
                }
                loc[pk]  = (int) row;
                used[sl] = g + 1;
            }
        }
        if (used[sl] && !px) {
            m.SX->flush(sx + sl * G * H, (size_t) used[sl] * H * sizeof(ggml_bf16_t));
        }
    });
    m.SRW->flush(m.SRW->vir, (size_t) calls * S * G * sizeof(float));
    tg.stop();
    tr.stop();
    const auto & ec = ly.exps[0];  // gate-up
    const auto & ed = ly.exps[2];  // down
    for (int c = 0; c < calls; ++c) {
        const size_t r0 = (size_t) c * S * G;
        if (px) {
            sk.in[sk.h2]  = axera_io(m.SX.get(), sx, axera_input_size(sk.k, "h2"));
            sk.in[sk.tok] = axera_io(m.TOK.get(), tk + (size_t) c * S * (1 + G), (uint32_t) (S * (1 + G) * sizeof(int32_t)));
        } else {
            sk.in[sk.x] = axera_io(m.SX.get(), sx + r0 * H, (uint32_t) (S * G * H * sizeof(ggml_bf16_t)));
        }
        sk.in[sk.rw]     = axera_io(m.SRW.get(), sw + r0, (uint32_t) (S * G * sizeof(float)));
        sk.outs[sk.out]  = axera_io(m.SOUT.get(), (char *) m.SOUT->vir + r0 * H * es, (uint32_t) (S * G * H * es));
        if (sk.sid >= 0 || px) {  // planar: the layer's expert planes, each slot's expert by index
            for (int i = 0; i < S && !px; ++i) {
                ((int32_t *) sk.SID->vir)[i] = se[(size_t) c * S + i];
            }
            if (!px) {
                sk.SID->flush(sk.SID->vir, S * sizeof(int32_t));
            }
            sk.in[sk.plane[0]] = axera_io(ec.b, ec.base, (uint32_t) (ec.n * ec.data));
            sk.in[sk.plane[1]] = axera_io(ec.b, ec.s(0), (uint32_t) (ec.n * ec.scale));
            sk.in[sk.plane[2]] = axera_io(ed.b, ed.base, (uint32_t) (ed.n * ed.data));
            sk.in[sk.plane[3]] = axera_io(ed.b, ed.s(0), (uint32_t) (ed.n * ed.scale));
        }
        for (int i = 0; i < S && sk.sid < 0 && !px; ++i) {
            const int e = se[(size_t) c * S + i];
            sk.in[sk.w[0][i]] = axera_io(ec.b, ec.w(e), (uint32_t) ec.data);
            sk.in[sk.s[0][i]] = axera_io(ec.b, ec.s(e), (uint32_t) ec.scale);
            sk.in[sk.w[1][i]] = axera_io(ed.b, ed.w(e), (uint32_t) ed.data);
            sk.in[sk.s[1][i]] = axera_io(ed.b, ed.s(e), (uint32_t) ed.scale);
        }
        axera_timer tm("moe prefill slots");
        const int   ret = axera_run_sync(sk.k, sk.in, sk.outs);
        if (ret != 0) {
            GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the prefill experts of layer %d\n", ret, il);
        }
    }
    axera_timer tc("moe prefill combine (cpu)");
    axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
        if (used[sl]) {
            m.SOUT->invalidate((char *) m.SOUT->vir + sl * G * H * es, (size_t) used[sl] * H * es);
        }
    });
    tc.stop();
    axera_timer  ts("moe prefill combine: sum (cpu)");
    const auto * so = (const char *) m.SOUT->vir;
    axera_host_pool().run(NT, [&](int64_t r) {
        // bf16 -> f32 as a 16-bit shift (vectorized; ggml_bf16_to_fp32 per element is not)
        auto               f32 = [](uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; };
        const auto *       base = (const uint16_t *) row_ptr(&axera_q35::prefill::BASE, (int) r, H * sizeof(ggml_bf16_t));
        std::vector<float> acc(H);
        for (int i = 0; i < H; ++i) {
            acc[i] = f32(base[i]);
        }
        for (int k = 0; k < K; ++k) {
            const char * o = so + (size_t) loc[r * K + k] * H * es;
            if (sk.ob) {
                const auto * ob = (const uint16_t *) o;
                for (int i = 0; i < H; ++i) {
                    acc[i] += f32(ob[i]);
                }
            } else {
                for (int i = 0; i < H; ++i) {
                    acc[i] += ((const float *) o)[i];
                }
            }
        }
        const auto & pf = m.pre[vs[r / P]];
        auto *       xo = (il % 2 == 0 ? pf.B : pf.A).get();
        auto *     xr = (uint16_t *) xo->vir + (size_t) (r % P) * H;
        for (int i = 0; i < H; ++i) {  // round to nearest even, as ggml_fp32_to_bf16 (finite values), vectorized
            uint32_t u;
            memcpy(&u, &acc[i], 4);
            xr[i] = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        }
    });
    for (int v : vs) {
        auto * xo = (il % 2 == 0 ? m.pre[v].B : m.pre[v].A).get();
        xo->flush(xo->vir, (size_t) P * H * sizeof(ggml_bf16_t));
    }
}
