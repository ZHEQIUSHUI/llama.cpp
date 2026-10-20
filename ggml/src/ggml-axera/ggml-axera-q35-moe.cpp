// Axera NPU backend, Qwen3.5 graph template: the routed experts of the MoE layers (35B-A3B)

#include "ggml-axera-q35.h"

// MoE layer il after its layer kernel: the top-K experts of the router logits (softmax, top-K, renormalized: as
// transformers' Qwen3_5MoeTopKRouter; the softmax over the top-K alone gives the same weights), their weights
// bound to the experts kernel, which adds them to the residual + shared expert (BASE) into the layer output
void axera_q35_batch::run_experts(int il, int64_t p) {
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
            char *       w    = e.base + (size_t) e_ids[i] * e.stride;
            ek.in[ek.w[kind][i]] = axera_io(e.b, w, (uint32_t) e.data);
            ek.in[ek.s[kind][i]] = axera_io(e.b, w + e.data, (uint32_t) e.scale);
        }
    }
    auto & out = ek.out[il % 2];
    tr.stop();
    axera_timer tm("moe experts");
    const int   ret = axera_run_sync(ek.k, ek.in, out);
    if (ret != 0) {
        GGML_ABORT("qwen35 template: RunSync failed (0x%x) in the experts of layer %d, position %" PRId64 "\n", ret,
                   il, p);
    }
    if (ek.PART) {  // multi-core: the layer output = the sum of the cores' rows
        const int cores = (int) (axera_output_size(ek.k, "part_out") / (H * sizeof(float)));
        ek.PART->invalidate(ek.PART->vir, ek.PART->size);
        const auto *       pr = (const float *) ek.PART->vir;
        std::vector<float> acc(pr, pr + H);
        for (int c = 1; c < cores; ++c) {
            for (int i = 0; i < H; ++i) {
                acc[i] += pr[(size_t) c * H + i];
            }
        }
        auto * x = (il % 2 == 0 ? m.B : m.A).get();
        ggml_fp32_to_bf16_row(acc.data(), (ggml_bf16_t *) x->vir, H);
        x->flush(x->vir, H * sizeof(ggml_bf16_t));
    }
}

// MoE layer il of prefill chunks vs (a group, or one chunk) after their layer kernels: each token's top-K experts
// (as run_experts), the picks grouped by expert into slots of up to G rows (h2 rows and routing weights gathered
// here), S slots per call of the slots kernel, then per token base + its K results into the next layer's input
void axera_q35_batch::run_experts_pre(int il, const std::vector<int> & vs) {
    axera_q35_layer & ly = m.layers[il];
    auto &            sk = m.slots;
    const int         P = m.pre[vs[0]].P, NT = P * (int) vs.size(), K = m.top_k, NE = m.n_expert, S = sk.S, G = sk.G;
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
    axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
        for (int g = 0; g < G; ++g) {
            const int64_t row = sl * G + g;
            const int     pk  = sp[row];
            sw[row]           = pk >= 0 ? wts[pk] : 0.0f;
            if (pk >= 0) {
                memcpy(sx + row * H, row_ptr(&axera_q35::prefill::H2, pk / K, H * sizeof(ggml_bf16_t)),
                       H * sizeof(ggml_bf16_t));
                loc[pk]  = (int) row;
                used[sl] = g + 1;
            }
        }
        if (used[sl]) {
            m.SX->flush(sx + sl * G * H, (size_t) used[sl] * H * sizeof(ggml_bf16_t));
        }
    });
    m.SRW->flush(m.SRW->vir, (size_t) calls * S * G * sizeof(float));
    tr.stop();
    const auto & ec = ly.exps[0];  // gate-up
    const auto & ed = ly.exps[2];  // down
    for (int c = 0; c < calls; ++c) {
        const size_t r0 = (size_t) c * S * G;
        sk.in[sk.x]      = axera_io(m.SX.get(), sx + r0 * H, (uint32_t) (S * G * H * sizeof(ggml_bf16_t)));
        sk.in[sk.rw]     = axera_io(m.SRW.get(), sw + r0, (uint32_t) (S * G * sizeof(float)));
        sk.outs[sk.out]  = axera_io(m.SOUT.get(), (float *) m.SOUT->vir + r0 * H, (uint32_t) (S * G * H * sizeof(float)));
        for (int i = 0; i < S; ++i) {
            const int e = se[(size_t) c * S + i];
            char *    wc = ec.base + (size_t) e * ec.stride;
            char *    wd = ed.base + (size_t) e * ed.stride;
            sk.in[sk.w[0][i]] = axera_io(ec.b, wc, (uint32_t) ec.data);
            sk.in[sk.s[0][i]] = axera_io(ec.b, wc + ec.data, (uint32_t) ec.scale);
            sk.in[sk.w[1][i]] = axera_io(ed.b, wd, (uint32_t) ed.data);
            sk.in[sk.s[1][i]] = axera_io(ed.b, wd + ed.data, (uint32_t) ed.scale);
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
            m.SOUT->invalidate((float *) m.SOUT->vir + sl * G * H, (size_t) used[sl] * H * sizeof(float));
        }
    });
    const auto * so = (const float *) m.SOUT->vir;
    axera_host_pool().run(NT, [&](int64_t r) {
        const auto *       base = (const ggml_bf16_t *) row_ptr(&axera_q35::prefill::BASE, (int) r, H * sizeof(ggml_bf16_t));
        std::vector<float> acc(H);
        for (int i = 0; i < H; ++i) {
            acc[i] = ggml_bf16_to_fp32(base[i]);
        }
        for (int k = 0; k < K; ++k) {
            const float * o = so + (size_t) loc[r * K + k] * H;
            for (int i = 0; i < H; ++i) {
                acc[i] += o[i];
            }
        }
        const auto & pf = m.pre[vs[r / P]];
        auto *       xo = (il % 2 == 0 ? pf.B : pf.A).get();
        ggml_fp32_to_bf16_row(acc.data(), (ggml_bf16_t *) xo->vir + (r % P) * H, H);
    });
    for (int v : vs) {
        auto * xo = (il % 2 == 0 ? m.pre[v].B : m.pre[v].A).get();
        xo->flush(xo->vir, (size_t) P * H * sizeof(ggml_bf16_t));
    }
}
