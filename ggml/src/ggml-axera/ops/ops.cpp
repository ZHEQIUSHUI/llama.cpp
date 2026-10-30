// Axera NPU backend ops: the graph's tensors by name, the KV cache and KQ mask, RoPE, the output head, CPU nodes

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace axera {

static std::string base_name(const char * name) {  // "<backend>#<name>#<copy>" -> <name>
    const char * a = strchr(name, '#');
    const char * b = a ? strrchr(name, '#') : nullptr;
    return a && b > a ? std::string(a + 1, b) : std::string(name);
}

const std::map<std::string, ggml_tensor *> & graph_names::of(const ggml_cgraph * g) {
    axera_timer tm("graph name map");
    // the graph's identity: its node pointers and their ops
    std::vector<uintptr_t> k(2 * (size_t) g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        k[2 * i]     = (uintptr_t) g->nodes[i];
        k[2 * i + 1] = (uintptr_t) g->nodes[i]->op;
    }
    if (k == key) {
        return t;
    }
    // each tensor once (in the order met: the first of a name wins), sorted by name into the map
    std::unordered_set<const ggml_tensor *>             seen;
    std::vector<std::pair<std::string, ggml_tensor *>> named;
    seen.reserve(4 * (size_t) g->n_nodes);
    named.reserve(4 * (size_t) g->n_nodes);
    auto add = [&](ggml_tensor * x) {
        if (seen.insert(x).second) {
            named.emplace_back(base_name(x->name), x);
        }
    };
    for (int i = 0; i < g->n_nodes; i++) {
        ggml_tensor * node = g->nodes[i];
        add(node);
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            add(node->src[s]);
            for (ggml_tensor * v = node->src[s]->view_src; v; v = v->view_src) {  // the cache tensors behind views
                add(v);
            }
        }
    }
    std::stable_sort(named.begin(), named.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    t.clear();
    for (auto & n : named) {
        if (t.empty() || std::prev(t.end())->first != n.first) {
            t.emplace_hint(t.end(), std::move(n.first), n.second);
        }
    }
    key = std::move(k);
    return t;
}

ggml_tensor * graph_names::find(const std::string & name) const {
    auto it = t.find(name);
    return it == t.end() ? nullptr : it->second;
}

ggml_tensor * graph_names::at(const std::string & name) const {
    ggml_tensor * x = find(name);
    if (!x) {
        GGML_ABORT("axera: tensor %s not in the graph\n", name.c_str());
    }
    return x;
}

const float * final_norm(const ggml_tensor * mul, int H, float * eps) {
    GGML_ASSERT(mul->op == GGML_OP_MUL);
    const ggml_tensor * rms = mul->src[0]->op == GGML_OP_RMS_NORM ? mul->src[0] : mul->src[1];
    const ggml_tensor * onw = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    GGML_ASSERT(rms->op == GGML_OP_RMS_NORM && onw->type == GGML_TYPE_F32 && onw->ne[0] == H);
    memcpy(eps, rms->op_params, sizeof(float));
    return (const float *) onw->data;
}

//
// KV cache
//

kv_view::kv_view(const ggml_tensor * k_idxs, const ggml_tensor * kqm, int64_t N, int BC) :
    k_idxs(k_idxs), kqm(kqm), N(N), BC(BC), W(std::min<int64_t>(N, (kqm->ne[0] + BC - 1) / BC * BC)) {}

int kv_view::mask_row(int t, ggml_bf16_t * dst) const {
    const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;  // finite, as the kernels' own masking: no inf - inf
    const char *   row  = (const char *) kqm->data + (kqm->ne[3] == 1 ? t * kqm->nb[1] :
                                                          (t / kqm->ne[1]) * kqm->nb[3] + (t % kqm->ne[1]) * kqm->nb[1]);
    const int64_t  n    = std::min<int64_t>(W, kqm->ne[0]);
    auto *         d    = (uint16_t *) dst;
    // the usual 0 / -inf masks (vectorized); any other value through a conversion
    bool other = false;
    if (kqm->type == GGML_TYPE_F16) {
        const auto * s = (const uint16_t *) row;
        for (int64_t c = 0; c < n; ++c) {
            d[c] = s[c] == 0 ? 0 : ninf;
            other |= s[c] != 0 && s[c] != 0xFC00;
        }
    } else {
        const auto * s = (const uint32_t *) row;
        for (int64_t c = 0; c < n; ++c) {
            d[c] = s[c] == 0 ? 0 : ninf;
            other |= s[c] != 0 && s[c] != 0xFF800000u;
        }
    }
    if (other) {
        for (int64_t c = 0; c < n; ++c) {
            const float v = kqm->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[c]) :
                                                         ((const float *) row)[c];
            d[c] = std::isinf(v) && v < 0 ? ninf : ggml_fp32_to_bf16(v).bits;
        }
    }
    std::fill(d + n, d + W, ninf);
    int64_t last = n - 1;
    while (last >= 0 && d[last] == ninf) {
        --last;
    }
    return (int) (last < 0 ? 1 : last / BC + 1);
}

int kv_view::chunk_mask(const int32_t * pos, int it, int P, int n, ggml_bf16_t * mask) const {
    const int64_t  R    = P;
    const uint16_t z    = ggml_fp32_to_bf16(0.0f).bits, ninf = ggml_fp32_to_bf16(-3e38f).bits;
    // one sequence's consecutive cells and positions under llama.cpp's causal mask: token r's row is the first's
    // plus cells c0 + 1 .. c0 + r, the last token's the first's with all n cells visible
    const int64_t c0     = cell_of(it);
    bool          causal = n >= 2 && n <= P && c0 + n <= W;
    for (int r = 1; causal && r < n; ++r) {
        causal = stream_of(it + r) == stream_of(it) && cell_of(it + r) == c0 + r && pos[it + r] == pos[it] + r;
    }
    if (causal) {
        axera_timer              tm("mask (cpu, causal chunk)");
        std::vector<ggml_bf16_t> first(W), last(W);
        mask_row(it, first.data());
        const int nb = mask_row(it + n - 1, last.data());
        causal       = first[c0].bits == z && last[c0].bits == z;
        for (int64_t c = c0 + 1; causal && c < c0 + n; ++c) {
            causal = first[c].bits == ninf && last[c].bits == z;
        }
        causal = causal && memcmp(first.data(), last.data(), c0 * sizeof(ggml_bf16_t)) == 0 &&
                 memcmp(first.data() + c0 + n, last.data() + c0 + n, (W - c0 - n) * sizeof(ggml_bf16_t)) == 0;
        if (causal) {
            axera_host_pool().run(nb, [&](int64_t blk) {
                const int64_t b0 = blk * BC;
                for (int64_t r = 0; r < R; ++r) {
                    ggml_bf16_t * dst = mask + (blk * R + r) * BC;
                    memcpy(dst, first.data() + b0, BC * sizeof(ggml_bf16_t));
                    // cells c0 + 1 .. c0 + min(r, n - 1) of this block: visible to row r
                    const int64_t lo = std::max<int64_t>(c0 + 1, b0);
                    const int64_t hi = std::min<int64_t>(c0 + std::min<int64_t>(r, n - 1) + 1, b0 + BC);
                    for (int64_t c = lo; c < hi; ++c) {
                        dst[c - b0].bits = z;
                    }
                }
            });
            return nb;
        }
    }
    axera_timer              tm("mask (cpu)");
    std::vector<ggml_bf16_t> rows((size_t) P * W);
    std::vector<int>         nb(P, 1);
    axera_host_pool().run(P, [&](int64_t r) { nb[r] = mask_row(it + std::min((int) r, n - 1), rows.data() + (size_t) r * W); });
    const int nblk = *std::max_element(nb.begin(), nb.end());
    axera_host_pool().run(nblk, [&](int64_t blk) {
        for (int64_t r = 0; r < R; ++r) {
            memcpy(mask + (blk * R + r) * BC, rows.data() + (size_t) (r % P) * W + blk * BC, BC * sizeof(ggml_bf16_t));
        }
    });
    return nblk;
}

//
// RoPE
//

rope::rope(const ggml_tensor * rope_node, const ggml_tensor * inp_pos, int T) : pos((const int32_t *) inp_pos->data), T(T) {
    int32_t rp[16];
    memcpy(rp, rope_node->op_params, sizeof(rp));
    float base, ext;
    memcpy(&base, rp + 5, sizeof(float));
    memcpy(&freq_scale, rp + 6, sizeof(float));
    memcpy(&ext, rp + 7, sizeof(float));
    if (ext != 0.0f) {
        GGML_ABORT("axera: YaRN RoPE scaling is not supported\n");
    }
    ROT        = rp[1];
    const int mode = rp[2];
    int       sc[4];
    memcpy(sc, rp + 11, sizeof(sc));
    inv_freq.resize(ROT / 2);
    for (int i = 0; i < ROT / 2; ++i) {
        inv_freq[i] = (float) (1.0 / std::pow((double) base, (double) (2 * i) / ROT));
    }
    mrope = (mode & GGML_ROPE_TYPE_MROPE) && inp_pos->ne[0] == 4 * (int64_t) T;
    sect.assign(ROT / 2, 0);
    for (int i = 0; mrope && i < ROT / 2; ++i) {
        const int n = sc[0] + sc[1] + sc[2] + sc[3], k = n > 0 ? i % n : 0;
        if (mode == GGML_ROPE_TYPE_IMROPE) {
            sect[i] = k % 3 == 1 && k < 3 * sc[1] ? 1 : k % 3 == 2 && k < 3 * sc[2] ? 2 : k % 3 == 0 && k < 3 * sc[0] ? 0 : 3;
        } else {
            sect[i] = k < sc[0] ? 0 : k < sc[0] + sc[1] ? 1 : k < sc[0] + sc[1] + sc[2] ? 2 : 3;
        }
    }
}

void rope::rows(int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) const {
    auto * c = (ggml_bf16_t *) cb->vir;
    auto * s = (ggml_bf16_t *) sb->vir;
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < ROT / 2; ++i) {
            const int32_t p      = pos[(mrope ? sect[i] * T : 0) + t0 + r];
            const float   a      = (float) p * freq_scale * inv_freq[i];
            c[r * (ROT / 2) + i] = ggml_fp32_to_bf16(std::cos(a));
            s[r * (ROT / 2) + i] = ggml_fp32_to_bf16(std::sin(a));
        }
    }
    cb->flush(cb->vir, cb->size);
    sb->flush(sb->vir, sb->size);
}

//
// output head
//

ggml_tensor * graph_head(const graph_names & names) {
    ggml_tensor * r = names.find("result_output");
    if (r && r->op == GGML_OP_CONCAT && r->src[1] && r->src[1]->op == GGML_OP_MUL_MAT) {
        return r->src[1];
    }
    return r;
}

// the graph's normed rows: result_norm, named result_embd_pooled when the context keeps unpooled embeddings
static ggml_tensor * normed_rows(const graph_names & names) {
    ggml_tensor * rn = names.find("result_norm");
    if (!rn) {
        ggml_tensor * p = names.find("result_embd_pooled");
        rn              = p && p->op == GGML_OP_MUL && p->src[0]->op == GGML_OP_RMS_NORM ? p : nullptr;
    }
    return rn;
}

void output_norm(const graph_names & names, const std::vector<float> & h, int T) {
    ggml_tensor * rn = normed_rows(names);
    GGML_ASSERT(rn && rn->type == GGML_TYPE_F32 && rn->ne[1] == T && (size_t) T * rn->ne[0] <= h.size());
    const int     H = (int) rn->ne[0];
    float         eps;
    const float * gw = final_norm(rn, H, &eps);
    axera_host_pool().run(T, [&](int64_t r) {
        const float * x  = h.data() + (size_t) r * H;
        float *       y  = (float *) ((char *) rn->data + r * rn->nb[1]);
        double        ss = 0;
        for (int i = 0; i < H; ++i) {
            ss += (double) x[i] * x[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / H) + eps);
        for (int i = 0; i < H; ++i) {
            y[i] = x[i] * scale * gw[i];
        }
    });
    if (ggml_backend_buffer_is_axera(rn->buffer)) {
        ((ggml_backend_axera_buffer_context *) rn->buffer->context)->flush(rn->data, ggml_nbytes(rn));
    }
}

void output_head::bind(const kernel & lm, const ggml_tensor * output_weight) {
    k = &lm;
    if (w == output_weight) {
        return;
    }
    w                = output_weight;
    const int64_t N = w->ne[1], K = w->ne[0];
    mx4              = lm.in_bytes("w") == N * K / 2;
    GGML_ASSERT((mx4 || lm.in_bytes("w") == N * K) && lm.in_bytes("s") == N * K / AXERA_MX_GROUP &&
                lm.in_bytes("x") == K * sizeof(ggml_bf16_t) && lm.out_bytes("out") == N * sizeof(float));
    if (mx4) {
        axera_quantize_mxfp4(w, q, s);
    } else {
        axera_quantize_mxfp8(w, q, s);
    }
    x = axera_cmm_alloc(K * sizeof(ggml_bf16_t));
}

void output_head::run(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T) {
    ggml_tensor * logits = graph_head(names);
    GGML_ASSERT(logits && logits->op == GGML_OP_MUL_MAT && logits->src[0] == w && logits->type == GGML_TYPE_F32 &&
                ggml_backend_buffer_is_axera(logits->buffer));
    const int64_t N = w->ne[1], K = w->ne[0];
    // final norm: the head's input is [get_rows(] rms_norm(h) * output_norm [, inp_out_ids)]
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw    = final_norm(hn, (int) K, &eps);
    const int     n_out = (int) logits->ne[1];
    // the output rows: the GET_ROWS index that crops the last layer or the final norm to the outputs
    ggml_tensor * out_ids = nullptr;
    for (int i = 0; i < g->n_nodes && n_out < T; i++) {
        ggml_tensor * node = g->nodes[i];
        if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
            node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
            out_ids = node->src[1];
        }
    }
    if (n_out < T && !out_ids) {
        GGML_ABORT("axera: %d of %d rows requested but no output row index in the graph\n", n_out, T);
    }
    ggml_tensor *      rn = normed_rows(names);  // the normed rows, when the graph keeps them (embeddings: the
                                                 // TTS talker's hidden state for the code predictor)
    rn                    = rn && rn->type == GGML_TYPE_F32 && rn->ne[0] == K && rn->ne[1] == n_out ? rn : nullptr;
    std::vector<float> y((size_t) n_out * K);
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        const float * xr = h.data() + (size_t) row * K;
        float *       yr = y.data() + (size_t) r * K;
        double        ss = 0;
        for (int i = 0; i < K; ++i) {
            ss += (double) xr[i] * xr[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / K) + eps);
        for (int i = 0; i < K; ++i) {
            yr[i] = xr[i] * scale * gw[i];
        }
        if (rn) {
            memcpy((char *) rn->data + r * rn->nb[1], yr, K * sizeof(float));
        }
    }
    if (rn && ggml_backend_buffer_is_axera(rn->buffer)) {
        ((ggml_backend_axera_buffer_context *) rn->buffer->context)->flush(rn->data, ggml_nbytes(rn));
    }
    axera_timer tl("lm_head");
    auto *      dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), out(k->info->nOutputSize);
    in[k->input("x")] = axera_io(x);
    in[k->input("w")] = axera_io(q, mx4 ? N * K / 2 : N * K);
    in[k->input("s")] = axera_io(s, N * K / AXERA_MX_GROUP);
    for (int r = 0; r < n_out; ++r) {
        float * o = (float *) ((char *) logits->data + r * logits->nb[1]);
        ggml_fp32_to_bf16_row(y.data() + (size_t) r * K, (ggml_bf16_t *) x->vir, K);
        x->flush(x->vir, K * sizeof(ggml_bf16_t));
        dctx->flush(o, N * sizeof(float));
        out[k->output("out")] = axera_io(dctx, o, (uint32_t) (N * sizeof(float)));
        k->run(in, out, "output head");
        dctx->invalidate(o, N * sizeof(float));
        dctx->clean_p     = o;  // the host reads these logits back without more cache maintenance this graph
        dctx->clean_n     = N * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    tl.stop();
    ggml_tensor * padded = names.find("result_output");  // the codec head's logits padded in front with -inf
    if (padded && padded != logits) {
        const int64_t pad = padded->ne[0] - logits->ne[0];
        GGML_ASSERT(padded->type == GGML_TYPE_F32 && padded->ne[1] == n_out && pad >= 0);
        for (int r = 0; r < n_out; ++r) {
            float * d = (float *) ((char *) padded->data + r * padded->nb[1]);
            std::fill(d, d + pad, -INFINITY);
            memcpy(d + pad, (const char *) logits->data + r * logits->nb[1], logits->ne[0] * sizeof(float));
        }
    }
    after(g, names);
}

void output_head::after(ggml_cgraph * g, const graph_names & names) {
    // the nodes after the logits in this split: llama.cpp's backend sampler chain, on the CPU (not after the codec
    // head's logits: their padded copy is written above)
    const ggml_tensor * head   = graph_head(names);
    const ggml_tensor * padded = names.find("result_output");
    int                 il     = -1;
    for (int i = 0; i < g->n_nodes; i++) {
        il = g->nodes[i] == head ? i : il;
    }
    if (il >= 0 && il + 1 < g->n_nodes && padded == head) {
        cpu_compute(g, g->nodes + il + 1, g->n_nodes - il - 1, "sampler (cpu)");
    }
}

//
// CPU nodes
//

void cpu_compute(ggml_cgraph * g, ggml_tensor ** nodes, int n, const char * what) {
    if (n <= 0) {
        return;
    }
    static std::mutex          mu;
    static ggml_backend_t      cpu = nullptr;
    std::lock_guard<std::mutex> lock(mu);
    if (!cpu) {  // (through the registry: this backend does not link the CPU backend)
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "axera: no CPU backend");
        cpu         = ggml_backend_dev_init(dev, nullptr);
        auto set_nt = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                                         "ggml_backend_set_n_threads");
        if (set_nt) {
            set_nt(cpu, (int) std::max(1u, std::thread::hardware_concurrency()));
        }
    }
    ggml_cgraph view = *g;  // the slice (ggml_graph_view is not exported by ggml-base)
    view.size        = 0;
    view.n_nodes     = n;
    view.n_leafs     = 0;
    view.nodes       = nodes;
    view.grads       = nullptr;
    view.grad_accs   = nullptr;
    view.leafs       = nullptr;
    view.uid         = 0;
    axera_timer tm(what);
    if (ggml_backend_graph_compute(cpu, &view) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("axera: %s failed on the CPU backend\n", what);
    }
}

}  // namespace axera
