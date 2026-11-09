// Axera NPU backend ops: the graph's tensors by name, the KV cache and KQ mask, RoPE, the output head, CPU nodes

#include "ops.h"
#include "../core/argmax_state.h"

#include "../core/npu_sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace axera {

static std::string_view base_name(const char * name) {  // "<backend>#<name>#<copy>" -> <name>
    const char * a = strchr(name, '#');
    const char * b = a ? strrchr(name, '#') : nullptr;
    return a && b > a ? std::string_view(a + 1, b - a - 1) : std::string_view(name);
}

const std::map<std::string, ggml_tensor *> & graph_names::of(const ggml_cgraph * g) {
    axera_timer tm("graph name map");
    static const bool prof = getenv("LLAMA_GRAPH_PROF") != nullptr;
    const int64_t     prof_t0 = prof ? ggml_time_us() : 0;
    int64_t           scan_us = 0, cmp_us = 0, map_us = 0;
    static const bool stats = getenv("GGML_AXERA_GRAPH_STATS") != nullptr;
    enum { ST_UID, ST_RECENT, ST_KEY, ST_TOPO, ST_BUILD } st = ST_BUILD;
    // ggml_cgraph.uid is the exact identity llama.cpp assigns to a graph and reuses while the graph is unchanged. Fall
    // back to its node pointers and ops only when the caller supplied no uid (the uid is then 0 by contract).
    auto is_key = [&](const std::vector<uintptr_t> & k) {
        if (k.size() != 2 * (size_t) g->n_nodes) {
            return false;
        }
        for (int i = 0; i < g->n_nodes; ++i) {
            if (k[2 * i] != (uintptr_t) g->nodes[i] || k[2 * i + 1] != (uintptr_t) g->nodes[i]->op) {
                return false;
            }
        }
        return true;
    };
    auto build_key = [&]() {
        if (key.size() != 2 * (size_t) g->n_nodes) {
            key.resize(2 * (size_t) g->n_nodes);
        }
        for (int i = 0; i < g->n_nodes; i++) {
            key[2 * i]     = (uintptr_t) g->nodes[i];
            key[2 * i + 1] = (uintptr_t) g->nodes[i]->op;
        }
    };
    // The map's occurrence order. llama.cpp now gives every topology the same stable uid (a hash of ops and names),
    // so a new graph of that topology has the same walk order and only its tensor pointers change.
    auto walk = [&](std::vector<ggml_tensor *> & occ) {
        occ.clear();
        occ.reserve(5 * (size_t) g->n_nodes);
        for (int i = 0; i < g->n_nodes; i++) {
            ggml_tensor * node = g->nodes[i];
            occ.push_back(node);
            for (int src = 0; src < GGML_MAX_SRC && node->src[src]; ++src) {
                occ.push_back(node->src[src]);
                for (ggml_tensor * v = node->src[src]->view_src; v; v = v->view_src) {
                    occ.push_back(v);
                }
            }
        }
    };
    auto update_values = [&](named_map & m, const std::vector<ggml_tensor *> & occ) {
        if (m.slots.size() != occ.size()) {
            return false;
        }
        for (size_t i = 0; i < occ.size(); ++i) {
            if (m.slots[i]) {
                *m.slots[i] = occ[i];
            }
        }
        return true;
    };
    auto build_map = [&](const std::vector<ggml_tensor *> & occ) {
        named_map out;
        std::unordered_set<const ggml_tensor *>                  seen;
        std::vector<std::pair<std::string_view, ggml_tensor *>> named;
        seen.reserve(occ.size());
        named.reserve(occ.size());
        for (ggml_tensor * x : occ) {
            if (seen.insert(x).second) {
                named.emplace_back(base_name(x->name), x);
            }
        }
        std::stable_sort(named.begin(), named.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::unordered_map<const ggml_tensor *, ggml_tensor **> slot_of;
        slot_of.reserve(named.size());
        for (auto & n : named) {
            if (out.t.empty() || out.t.rbegin()->first.compare(n.first) != 0) {
                out.t.emplace_hint(out.t.end(), std::string(n.first), n.second);
                slot_of[n.second] = &out.t.rbegin()->second;
            } else {
                // The old map kept the first occurrence of a duplicate base name. Leave the other occurrences
                // without a slot so a topology hit does not overwrite it.
                slot_of[n.second] = nullptr;
            }
        }
        out.slots.resize(occ.size());
        for (size_t i = 0; i < occ.size(); ++i) {
            out.slots[i] = slot_of[occ[i]];
        }
        out.graph_uid = g->uid;
        return out;
    };
    // stable_uid survives the scheduler's per-call split of a reused graph; uid does not
    const uint64_t gid = g->stable_uid ? g->stable_uid : g->uid;
    // Keep the current graph's map before handing its slot to another graph. Swapping the maps in place would leave
    // the slot's key pointing at the other graph's map, so the next lookup of that key would return the wrong graph.
    auto remember_uid = [&]() {
        if (!uid || cur.t.empty()) {
            return;
        }
        for (auto & r : recent_uid) {
            if (r.first == uid) {
                r.second = std::move(cur);
                cur = {};
                return;
            }
        }
        if (recent_uid.size() >= 4) {
            recent_uid.erase(recent_uid.begin());
        }
        recent_uid.emplace_back(uid, std::move(cur));
        cur = {};
    };
    auto remember_key = [&]() {
        if (key.empty() || cur.t.empty()) {
            return;
        }
        for (auto & r : recent) {
            if (r.first == key) {
                r.second = std::move(cur);
                cur = {};
                return;
            }
        }
        if (recent.size() >= 4) {
            recent.erase(recent.begin());
        }
        recent.emplace_back(std::move(key), std::move(cur));
        key.clear();
        cur = {};
    };
    bool hit = false;
    std::vector<ggml_tensor *> occ;
    if (gid) {
        if (uid == gid) {
            if (cur.graph_uid == g->uid) {
                // The scheduler reused the graph: its tensor pointers did not change. A rebuilt graph of the same
                // topology gets a fresh cgraph.uid and pays the walk below to refresh them.
                key.clear();
                hit = true;
                st  = ST_UID;
            } else {
                const int64_t t0 = prof ? ggml_time_us() : 0;
                walk(occ);
                scan_us += (prof ? ggml_time_us() : 0) - t0;
                if (update_values(cur, occ)) {
                    cur.graph_uid = g->uid;
                    key.clear();
                    hit = true;
                    st  = ST_UID;
                }
            }
        } else {
            // Alternating graphs (a speculative target and its draft) come back with the same cgraph.uid as long as
            // llama.cpp reuses them; their tensor pointers have not changed then, so the expensive walk is not needed.
            auto it = std::find_if(recent_uid.begin(), recent_uid.end(),
                                   [&](const auto & r) { return r.first == gid; });
            if (it != recent_uid.end() && it->second.graph_uid == g->uid) {
                auto found = std::move(it->second);
                recent_uid.erase(it);
                remember_uid();
                cur = std::move(found);
                uid = gid;
                key.clear();
                hit = true;
                st  = ST_RECENT;
            } else {
                const int64_t t0 = prof ? ggml_time_us() : 0;
                walk(occ);
                scan_us += (prof ? ggml_time_us() : 0) - t0;
                if (it != recent_uid.end()) {
                    auto found = std::move(it->second);
                    recent_uid.erase(it);
                    remember_uid();
                    cur = std::move(found);
                    if (update_values(cur, occ)) {
                        cur.graph_uid = g->uid;
                        uid = gid;
                        key.clear();
                        hit = true;
                        st  = ST_RECENT;
                    }
                }
            }
        }
        if (!hit) {
            if (uid) {  // keep the current map for an alternation back to this graph
                remember_uid();
            }
            uid = gid;
            key.clear();
        }
    } else {
        hit = is_key(key);
        if (hit) {
            st = ST_KEY;
        } else {
            for (auto it = recent.begin(); it != recent.end(); ++it) {
                if (is_key(it->first)) {
                    auto found_key = std::move(it->first);
                    auto found     = std::move(it->second);
                    recent.erase(it);
                    remember_key();
                    key = std::move(found_key);
                    cur = std::move(found);
                    hit = true;
                    st  = ST_KEY;
                    break;
                }
            }
        }
        if (!hit) {
            if (!key.empty()) {
                remember_key();
            }
            uid = 0;
        }
    }
    if (!hit) {
        const int64_t t0 = prof ? ggml_time_us() : 0;
        if (occ.empty()) {
            walk(occ);
        }
        cur = build_map(occ);
        if (!gid) {
            build_key();
        }
        map_us += (prof ? ggml_time_us() : 0) - t0;
    }
    if (prof && g->n_nodes > 1000) {
        const int64_t total_us = ggml_time_us() - prof_t0;
        fprintf(stderr, "graph names: nodes %d state %d walk %.3f, compare %.3f, map %.3f, total %.3f ms\n",
                g->n_nodes, (int) st, scan_us / 1e3, cmp_us / 1e3, map_us / 1e3, total_us / 1e3);
    }
    if (stats) {
        static uint64_t n = 0, cnt[5] = {0, 0, 0, 0, 0};
        ++n; ++cnt[st];
        if (n % 8 == 0) {
            fprintf(stderr, "graph names: %llu calls, uid %llu/recent %llu/key %llu/topo %llu/build %llu "
                    "(uid %llu stable %llu)\n",
                    (unsigned long long) n, (unsigned long long) cnt[ST_UID], (unsigned long long) cnt[ST_RECENT],
                    (unsigned long long) cnt[ST_KEY], (unsigned long long) cnt[ST_TOPO],
                    (unsigned long long) cnt[ST_BUILD], (unsigned long long) g->uid,
                    (unsigned long long) g->stable_uid);
        }
    }
    return cur.t;
}

ggml_tensor * graph_names::find(const std::string & name) const {
    auto it = cur.t.find(name);
    return it == cur.t.end() ? nullptr : it->second;
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
    R                = lm.find_input("m") >= 0 ? (int) (lm.in_bytes("x") / (K * sizeof(ggml_bf16_t))) : 1;
    GGML_ASSERT((mx4 || lm.in_bytes("w") == N * K) && lm.in_bytes("s") == N * K / AXERA_MX_GROUP &&
                lm.in_bytes("x") == R * K * sizeof(ggml_bf16_t) && lm.out_bytes("out") == R * N * sizeof(float));
    if (R > 1) {  // rows R at a time (output to y, copied to the logits)
        yk = axera_cmm_alloc(R * N * sizeof(float));
        mk = axera_cmm_alloc(sizeof(int32_t));
    }
    if (mx4) {
        axera_quantize_mxfp4(w, q, s);
    } else {
        axera_quantize_mxfp8(w, q, s);
    }
    axera_drop_mapped(w, true);  // (a tied output head shares token_embd.weight: the mapped rows are not needed now)
    x = axera_cmm_alloc(R * K * sizeof(ggml_bf16_t));
}

void output_head::bind_topk(const kernel & lm) {
    GGML_ASSERT(k && w);
    if (tk == &lm) {
        return;
    }
    tk = &lm;
    const int64_t N = w->ne[1], K = w->ne[0];
    tR              = (int) (lm.in_bytes("x") / (K * sizeof(ggml_bf16_t)));
    GGML_ASSERT(tR == 1 || tR == 4);
    tK              = (int) (lm.out_bytes("values") / (sizeof(float) * tR));
    GGML_ASSERT(tK > 0 && (tR == 1 || lm.find_input("m") >= 0) && lm.find_input("bias") >= 0 && lm.find_input("mul_pos") >= 0 &&
                lm.find_input("mul_neg") >= 0 && lm.find_input("temp") >= 0 &&
                lm.in_bytes("x") == (uint32_t) (tR * K * sizeof(ggml_bf16_t)) &&
                lm.in_bytes("w") == (uint32_t) (mx4 ? N * K / 2 : N * K) &&
                lm.in_bytes("s") == N * K / AXERA_MX_GROUP && lm.in_bytes("bias") == N * sizeof(float) &&
                lm.in_bytes("mul_pos") == N * sizeof(float) && lm.in_bytes("mul_neg") == N * sizeof(float) &&
                lm.in_bytes("temp") == sizeof(float) && (tR == 1 || lm.in_bytes("m") == sizeof(int32_t)) &&
                lm.out_bytes("ids") == (uint32_t) (tK * tR * sizeof(int32_t)) &&
                lm.out_bytes("vmax") == (uint32_t) (tR * sizeof(float)) &&
                lm.out_bytes("sumexp") == (uint32_t) (tR * sizeof(float)));
    // the penalty state is the host sampler's: only the 1-2 rows that change per step are written (non-cacheable, so
    // the NPU sees them without a clean)
    tbias = axera_cmm_alloc_nc(N * sizeof(float));
    tmulp = axera_cmm_alloc_nc(N * sizeof(float));
    tmuln = axera_cmm_alloc_nc(N * sizeof(float));
    std::fill((float *) tmulp->vir, (float *) tmulp->vir + N, 1.0f);
    std::fill((float *) tmuln->vir, (float *) tmuln->vir + N, 1.0f);
    ttemp = axera_cmm_alloc_nc(sizeof(float));
    *(float *) ttemp->vir = 1.0f;
    tval  = axera_cmm_alloc((size_t) tR * tK * sizeof(float));
    tids  = axera_cmm_alloc((size_t) tR * tK * sizeof(int32_t));
    tmax  = axera_cmm_alloc((size_t) tR * sizeof(float));
    tsum  = axera_cmm_alloc((size_t) tR * sizeof(float));
    if (tR > 1) {
        tkm = axera_cmm_alloc(sizeof(int32_t));
    }
    tv.resize((size_t) tR * tK);
    ti.resize((size_t) tR * tK);
    tx = axera_cmm_alloc((size_t) tR * K * sizeof(ggml_bf16_t));
    tgen = -1;
}

bool output_head::bind_fused(const kernel & lm, const ggml_tensor * output_weight) {
    if (!k || w != output_weight || !mx4) {
        return false;
    }
    const int64_t N = w->ne[1], K = w->ne[0];
    if (lm.find_input("a") < 0 || lm.find_input("norm_weight") < 0 || lm.find_input("b_t") < 0 ||
        lm.find_input("b_scale") < 0 || lm.find_output("output") < 0) {
        return false;
    }
    if (lm.in_bytes("a") != (uint32_t) (K * sizeof(ggml_bf16_t)) || lm.in_bytes("norm_weight") != (uint32_t) (K * sizeof(float)) ||
        lm.in_bytes("b_t") != (uint32_t) (N * K / 2) || lm.in_bytes("b_scale") != (uint32_t) (N * K / AXERA_MX_GROUP) ||
        lm.out_bytes("output") != (uint32_t) (N * sizeof(float))) {
        return false;
    }
    fk = &lm;
    if (!fkw) {
        fkw = axera_cmm_alloc((size_t) K * sizeof(float));
    }
    fin.assign(fk->info->nInputSize, AX_ENGINE_IO_BUFFER_T{});
    fout.assign(fk->info->nOutputSize, AX_ENGINE_IO_BUFFER_T{});
    fin[fk->input("norm_weight")] = axera_io(fkw, (size_t) K * sizeof(float));
    fin[fk->input("b_t")]         = axera_io(q, mx4 ? (size_t) N * K / 2 : (size_t) N * K);
    fin[fk->input("b_scale")]     = axera_io(s, (size_t) N * K / AXERA_MX_GROUP);
    fgw                           = nullptr;
    return true;
}

bool output_head::bind_combined(const kernel & lm, const ggml_tensor * output_weight) {
    if (!k || w != output_weight) {
        return false;
    }
    const int64_t N = w->ne[1], K = w->ne[0];
    if (lm.find_input("x") < 0 || lm.find_input("norm_w") < 0 || lm.find_input("w") < 0 || lm.find_input("s") < 0 ||
        lm.find_output("out") < 0 || lm.find_output("ids") < 0 || lm.find_output("top_logits") < 0) {
        return false;
    }
    const int R = (int) (lm.in_bytes("x") / (K * sizeof(ggml_bf16_t)));
    if (R != 1 ||
        lm.in_bytes("norm_w") != (uint32_t) (K * sizeof(float)) ||
        lm.in_bytes("w") != (uint32_t) (mx4 ? N * K / 2 : N * K) ||
        lm.in_bytes("s") != (uint32_t) (N * K / AXERA_MX_GROUP) ||
        lm.in_bytes("x") != (uint32_t) (K * sizeof(ggml_bf16_t)) ||
        lm.out_bytes("out") != (uint32_t) (N * sizeof(float)) || lm.out_bytes("ids") != sizeof(int32_t) ||
        lm.out_bytes("top_logits") != sizeof(float)) {
        return false;
    }
    if (ck == &lm) {
        return true;
    }
    ck = &lm;
    cR = R;
    cnorm = axera_cmm_alloc((size_t) K * sizeof(float));
    cx    = axera_cmm_alloc((size_t) K * sizeof(ggml_bf16_t));
    cids  = axera_cmm_alloc(sizeof(int32_t));
    ctop  = axera_cmm_alloc(sizeof(float));
    cin.assign(ck->info->nInputSize, AX_ENGINE_IO_BUFFER_T{});
    cout.assign(ck->info->nOutputSize, AX_ENGINE_IO_BUFFER_T{});
    cin[ck->input("norm_w")] = axera_io(cnorm, (size_t) K * sizeof(float));
    cin[ck->input("w")]      = axera_io(q, mx4 ? (size_t) N * K / 2 : (size_t) N * K);
    cin[ck->input("s")]      = axera_io(s, (size_t) N * K / AXERA_MX_GROUP);
    cout[ck->output("ids")]  = axera_io(cids, sizeof(int32_t));
    cout[ck->output("top_logits")] = axera_io(ctop, sizeof(float));
    cgw = nullptr;
    return true;
}

bool output_head::combined_ready(ggml_cgraph * g, const graph_names & names) const {
    GGML_UNUSED(g);
    if (!ck) {
        return false;
    }
    ggml_tensor * logits = graph_head(names);
    if (!logits || logits->op != GGML_OP_MUL_MAT || logits->src[0] != w || logits->type != GGML_TYPE_F32 ||
        logits->ne[1] < 1) {
        return false;
    }
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw = final_norm(hn, (int) w->ne[0], &eps);
    return gw != nullptr && eps == 1e-6f;
}

bool output_head::topk_ready(ggml_cgraph * g, const graph_names & names, int T) const {
    GGML_UNUSED(g);
    if (!tk || draft) {
        return false;
    }
    ggml_tensor * logits = graph_head(names);
    if (!logits) {
        return false;
    }
    const int n_out = (int) logits->ne[1];
    if (n_out < 1 || n_out > tR || T != n_out) {
        return false;
    }
    if (names.find("result_output") && graph_head(names) != names.find("result_output")) {
        return false;
    }
    const npu_opts opts = npu_get_options();
    return opts.enabled && opts.K == tK;
}

bool output_head::bind_combined4(const kernel & lm, const ggml_tensor * output_weight) {
    if (!k || w != output_weight) {
        return false;
    }
    const int64_t N = w->ne[1], K = w->ne[0];
    if (lm.find_input("x") < 0 || lm.find_input("norm_w") < 0 || lm.find_input("w") < 0 ||
        lm.find_input("s") < 0 || lm.find_input("m") < 0 || lm.find_output("out") < 0 ||
        lm.find_output("ids") < 0 || lm.find_output("top_logits") < 0) {
        return false;
    }
    if (lm.in_bytes("x") != (uint32_t) (4 * K * sizeof(ggml_bf16_t)) ||
        lm.in_bytes("norm_w") != (uint32_t) (K * sizeof(float)) ||
        lm.in_bytes("w") != (uint32_t) (mx4 ? N * K / 2 : N * K) ||
        lm.in_bytes("s") != (uint32_t) (N * K / AXERA_MX_GROUP) || lm.in_bytes("m") != sizeof(int32_t) ||
        lm.out_bytes("out") != (uint32_t) (4 * N * sizeof(float)) || lm.out_bytes("ids") != 4 * sizeof(int32_t) ||
        lm.out_bytes("top_logits") != 4 * sizeof(float)) {
        return false;
    }
    if (c4k == &lm) {
        return true;
    }
    c4k   = &lm;
    c4norm = axera_cmm_alloc((size_t) K * sizeof(float));
    c4x    = axera_cmm_alloc((size_t) 4 * K * sizeof(ggml_bf16_t));
    c4m    = axera_cmm_alloc(sizeof(int32_t));
    c4ids  = axera_cmm_alloc(4 * sizeof(int32_t));
    c4top  = axera_cmm_alloc(4 * sizeof(float));
    c4in.assign(c4k->info->nInputSize, AX_ENGINE_IO_BUFFER_T{});
    c4out.assign(c4k->info->nOutputSize, AX_ENGINE_IO_BUFFER_T{});
    c4in[c4k->input("norm_w")] = axera_io(c4norm, (size_t) K * sizeof(float));
    c4in[c4k->input("w")]      = axera_io(q, mx4 ? (size_t) N * K / 2 : (size_t) N * K);
    c4in[c4k->input("s")]      = axera_io(s, (size_t) N * K / AXERA_MX_GROUP);
    c4in[c4k->input("m")]      = axera_io(c4m, sizeof(int32_t));
    c4out[c4k->output("ids")]  = axera_io(c4ids, 4 * sizeof(int32_t));
    c4out[c4k->output("top_logits")] = axera_io(c4top, 4 * sizeof(float));
    c4gw = nullptr;
    return true;
}

void output_head::run_combined(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T,
                               const std::vector<int> * compact_rows, const axera_cmm * xb, bool publish_argmax) {
    if (topk_ready(g, names, T)) {
        run(g, names, h, T, nullptr, nullptr, compact_rows);
        return;
    }
    ggml_tensor * logits = graph_head(names);
    GGML_ASSERT(logits && logits->op == GGML_OP_MUL_MAT && logits->src[0] == w && logits->type == GGML_TYPE_F32 &&
                ggml_backend_buffer_is_axera(logits->buffer));
    const int64_t N = w->ne[1], K = w->ne[0];
    const int     n_out = (int) logits->ne[1];
    if (c4k && n_out == 4 && T == 4 && logits->nb[1] == N * sizeof(float)) {
        run_combined4(g, names, h, T, publish_argmax);
        return;
    }
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw = final_norm(hn, (int) K, &eps);
    GGML_ASSERT(gw && eps == 1e-6f);
    if (gw != cgw) {
        memcpy(cnorm->vir, gw, (size_t) K * sizeof(float));
        cnorm->flush(cnorm->vir, (size_t) K * sizeof(float));
        cgw = gw;
    }
    ggml_tensor * out_ids = n_out < T ? names.find("out_ids") : nullptr;
    if (out_ids && (out_ids->type != GGML_TYPE_I32 || out_ids->ne[0] != n_out)) {
        out_ids = nullptr;
    }
    if (n_out < T && !out_ids) {
        for (int i = 0; i < g->n_nodes; i++) {
            ggml_tensor * node = g->nodes[i];
            if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
                node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
                out_ids = node->src[1];
            }
        }
    }
    GGML_ASSERT(n_out == T || out_ids);
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    std::vector<int32_t> pids((size_t) n_out);
    std::vector<float>   pvals((size_t) n_out);
    bool                 single = xb && T == 1 && n_out == 1 && (!compact_rows || (*compact_rows)[0] == 0);
    for (int r = 0; r < n_out; ++r) {
        const int     row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        const float * xr  = h.data() + (size_t) (compact_rows ? r : row) * K;
        if (single) {
            cin[ck->input("x")] = axera_io(*xb, (size_t) K * sizeof(ggml_bf16_t));
        } else {
            ggml_fp32_to_bf16_row(xr, (ggml_bf16_t *) cx->vir, K);
            cx->flush(cx->vir, (size_t) K * sizeof(ggml_bf16_t));
            cin[ck->input("x")] = axera_io(cx, (size_t) K * sizeof(ggml_bf16_t));
        }
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        // the output row is host-visible: flush a previous graph's CPU-dirty lines before the NPU writes it
        dctx->flush(o, (size_t) N * sizeof(float));
        cout[ck->output("out")] = axera_io(dctx, o, (size_t) N * sizeof(float));
        axera_timer tk("lm_head (combined)");
        ck->run(cin, cout, "output head (combined)");
        tk.stop();
        cids->invalidate(cids->vir, sizeof(int32_t));
        ctop->invalidate(ctop->vir, sizeof(float));
        memcpy(&pids[(size_t) r], cids->vir, sizeof(int32_t));
        memcpy(&pvals[(size_t) r], ctop->vir, sizeof(float));
    }
    const bool published = publish_argmax && axera::argmax_publish(pids.data(), pvals.data(), n_out, logits->data,
                                                                   (size_t) logits->nb[1]);
    // The sampler's fused-row stale-value check still reads the host-visible full row. Retire the NPU-written lines
    // even when the fused argmax was published, or that check can consume a previous graph's cached values.
    for (int r = 0; r < n_out; ++r) {
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        dctx->invalidate(o, (size_t) N * sizeof(float));
    }
    if (!published) {
        dctx->clean_p     = (float *) logits->data;
        dctx->clean_n     = (size_t) n_out * N * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    after(g, names);
}

void output_head::run_combined4(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T,
                                bool publish_argmax) {
    ggml_tensor * logits = graph_head(names);
    GGML_ASSERT(logits && logits->op == GGML_OP_MUL_MAT && logits->src[0] == w && logits->type == GGML_TYPE_F32 &&
                logits->ne[1] == 4 && T == 4 && ggml_backend_buffer_is_axera(logits->buffer));
    const int64_t N = w->ne[1], K = w->ne[0];
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw = final_norm(hn, (int) K, &eps);
    GGML_ASSERT(gw && eps == 1e-6f && h.size() >= (size_t) 4 * K);
    if (gw != c4gw) {
        memcpy(c4norm->vir, gw, (size_t) K * sizeof(float));
        c4norm->flush(c4norm->vir, (size_t) K * sizeof(float));
        c4gw = gw;
    }
    ggml_fp32_to_bf16_row(h.data(), (ggml_bf16_t *) c4x->vir, (int) (4 * K));
    c4x->flush(c4x->vir, (size_t) 4 * K * sizeof(ggml_bf16_t));
    *(int32_t *) c4m->vir = 4;
    c4m->flush(c4m->vir, sizeof(int32_t));
    c4in[c4k->input("x")] = axera_io(c4x, (size_t) 4 * K * sizeof(ggml_bf16_t));
    GGML_ASSERT(logits->nb[1] == N * sizeof(float));
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    c4out[c4k->output("out")] = axera_io(dctx, logits->data, (size_t) 4 * N * sizeof(float));
    {
        axera_timer tk("lm_head (combined m4)");
        c4k->run(c4in, c4out, "output head (combined m4)");
        tk.stop();
    }
    // the host samples the verification rows from these full logits, so invalidate them whether or not the fused
    // rows were also published (a published run skips the copy, not the host's read)
    dctx->invalidate(logits->data, (size_t) 4 * N * sizeof(float));
    c4ids->invalidate(c4ids->vir, 4 * sizeof(int32_t));
    c4top->invalidate(c4top->vir, 4 * sizeof(float));
    std::vector<int32_t> pids(4);
    std::vector<float>   pvals(4);
    memcpy(pids.data(), c4ids->vir, 4 * sizeof(int32_t));
    memcpy(pvals.data(), c4top->vir, 4 * sizeof(float));
    if (!publish_argmax || !axera::argmax_publish(pids.data(), pvals.data(), 4, logits->data,
                                                  (size_t) logits->nb[1])) {
        dctx->clean_p     = (float *) logits->data;
        dctx->clean_n     = (size_t) 4 * N * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    after(g, names);
}

bool output_head::fused_ready(ggml_cgraph * g, const graph_names & names) const {
    GGML_UNUSED(g);
    if (!fk) {
        return false;
    }
    ggml_tensor * logits = graph_head(names);
    if (!logits || logits->op != GGML_OP_MUL_MAT || logits->src[0] != w || logits->type != GGML_TYPE_F32 ||
        logits->ne[1] != 1) {
        return false;
    }
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw = final_norm(hn, (int) w->ne[0], &eps);
    // the kernel was built with the model's 1e-6; another context's eps keeps the host path
    return gw != nullptr && eps == 1e-6f;
}

void output_head::run_fused(ggml_cgraph * g, const graph_names & names, const axera_cmm & xb) {
    axera_timer ts("lm_head (fused): setup");
    ggml_tensor * logits = graph_head(names);
    const int64_t N = w->ne[1], K = w->ne[0];
    ggml_tensor * hn = logits->src[1];
    while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
        hn = hn->src[0];
    }
    float         eps;
    const float * gw = final_norm(hn, (int) K, &eps);
    if (gw != fgw) {
        memcpy(fkw->vir, gw, (size_t) K * sizeof(float));
        fkw->flush();
        fgw = gw;
    }
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    float * o   = (float *) logits->data;
    fin[fk->input("a")]         = axera_io(xb, (size_t) K * sizeof(ggml_bf16_t));
    fout[fk->output("output")] = axera_io(dctx, o, (size_t) N * sizeof(float));
    ts.stop();
    axera_timer tl("lm_head (fused)");
    fk->run(fin, fout, "output head (fused)");
    tl.stop();
    axera_timer ti("lm_head (fused): invalidate");
    dctx->invalidate(o, (size_t) N * sizeof(float));
    ti.stop();
    dctx->clean_p     = o;
    dctx->clean_n     = (size_t) N * sizeof(float);
    dctx->clean_epoch = axera_graph_epoch;
    axera_timer ta("lm_head (fused): after");
    after(g, names);
    ta.stop();
}

void output_head::bind_draft(const kernel & dlm, const std::vector<int32_t> & ids) {
    GGML_ASSERT(k);
    const int64_t N = w->ne[1], K = w->ne[0], NS = (int64_t) ids.size();
    const bool    argmax = dlm.find_output("ids") >= 0 || dlm.find_output("top_logits") >= 0;
    if (argmax) {
        GGML_ASSERT(dlm.find_output("ids") >= 0 && dlm.find_output("top_logits") >= 0 && dlm.find_output("out") < 0);
        GGML_ASSERT(dlm.out_bytes("ids") == sizeof(int32_t) && dlm.out_bytes("top_logits") == sizeof(float));
    } else {
        GGML_ASSERT(dlm.find_output("out") >= 0 && dlm.out_bytes("out") == NS * sizeof(float));
    }
    // the draft head's weight rows are the already-quantized output head's, in the same format: MXFP4
    // (K / 2 bytes a row) or MXFP8 (K bytes a row)
    const bool    dmx4 = dlm.in_bytes("w") == NS * K / 2;
    GGML_ASSERT(dmx4 == mx4 && (dmx4 || dlm.in_bytes("w") == NS * K) &&
                dlm.in_bytes("s") == NS * K / AXERA_MX_GROUP && dlm.in_bytes("x") == K * sizeof(ggml_bf16_t));
    dk = &dlm;
    if (dids == ids && dargmax == argmax && dqm4 == dmx4) {
        return;
    }
    dids    = ids;
    dargmax = argmax;
    dqm4    = dmx4;
    const int64_t rowb = dmx4 ? K / 2 : K;
    dq      = axera_cmm_alloc(NS * rowb);
    ds      = axera_cmm_alloc(NS * K / AXERA_MX_GROUP);
    if (!dargmax) {
        dy = axera_cmm_alloc(NS * sizeof(float));
    } else {
        did  = axera_cmm_alloc(sizeof(int32_t));
        dtop = axera_cmm_alloc(sizeof(float));
    }
    axera_host_pool().run(NS, [&](int64_t i) {
        const int64_t r = ids[i];
        GGML_ASSERT(r >= 0 && r < N && (i == 0 || ids[i - 1] < r));
        memcpy((uint8_t *) dq->vir + i * rowb, (const uint8_t *) q->vir + r * rowb, rowb);
        memcpy((uint8_t *) ds->vir + i * (K / AXERA_MX_GROUP), (const uint8_t *) s->vir + r * (K / AXERA_MX_GROUP), K / AXERA_MX_GROUP);
    });
    dq->flush();
    ds->flush();
}

void output_head::bind_verify(const kernel & lm) {
    GGML_ASSERT(k && mx4);
    GGML_ASSERT(lm.in_bytes("w") == k->in_bytes("w") && lm.in_bytes("s") == k->in_bytes("s"));
    kv = &lm;
    const int64_t K = w->ne[0];
    const int     Rv = lm.find_input("m") >= 0 ? (int) (lm.in_bytes("x") / (K * sizeof(ggml_bf16_t))) : 1;
    GGML_ASSERT(Rv > 1);
    if (!yk) {
        yk = axera_cmm_alloc((size_t) Rv * w->ne[1] * sizeof(float));
        mk = axera_cmm_alloc(sizeof(int32_t));
    }
    if (x->size < lm.in_bytes("x")) {
        x = axera_cmm_alloc(lm.in_bytes("x"));
    }
}

int32_t output_head::draft_argmax_row(const ggml_bf16_t * row, float * logit) {
    GGML_ASSERT(dk && dargmax && did && dtop);
    const size_t K = (size_t) w->ne[0];
    memcpy(x->vir, row, K * sizeof(ggml_bf16_t));
    x->flush(x->vir, K * sizeof(ggml_bf16_t));
    std::vector<AX_ENGINE_IO_BUFFER_T> din(dk->info->nInputSize), dout(dk->info->nOutputSize);
    din[dk->input("x")]            = axera_io(x, K * sizeof(ggml_bf16_t));
    din[dk->input("w")]            = axera_io(dq);
    din[dk->input("s")]            = axera_io(ds);
    dout[dk->output("ids")]        = axera_io(did, sizeof(int32_t));
    dout[dk->output("top_logits")] = axera_io(dtop, sizeof(float));
    axera_timer tl("lm_head (draft chain)");
    dk->run(din, dout, "draft head (argmax)");
    did->invalidate(did->vir, sizeof(int32_t));
    dtop->invalidate(dtop->vir, sizeof(float));
    int32_t lid = 0;
    memcpy(&lid, did->vir, sizeof(int32_t));
    memcpy(logit, dtop->vir, sizeof(float));
    GGML_ASSERT(lid >= 0 && lid < (int32_t) dids.size());
    return dids[(size_t) lid];
}

// a TOP_K of the draft head's single row: over its vocabulary's logits only (the others are -inf)
static const output_head * top_k_draft = nullptr;
static bool                dargmax_pub = false;  // the draft head just published its row (set around after())
static int32_t             dargmax_id  = -1;
// with a TOP_K in its sampler, a draft head writes no logits: the TOP_K writes those of its rows (here and in the
// padded copy, which is not made) and nothing reads the others
static float * draft_logits = nullptr;

void output_head::bind_cpu(const ggml_tensor * output_weight) {
    k = nullptr;
    if (w != output_weight) {
        w = output_weight;
        wf.clear();  // (converted at the first row the CPU projects)
    }
}

// the sampler found that a verification row cannot be served by the K candidates: project that row through the regular
// head into the caller's host buffer. The run that published the candidates kept its normed rows for this.
int output_head::full_row(int32_t idx, float * out, int32_t n) {
    if (!k || fn_out <= 0 || !out || n <= 0) {
        return 0;
    }
    const int64_t N = w->ne[1], K = w->ne[0];
    if (n < N) {
        return 0;
    }
    if (idx < 0) {  // llama.cpp's -1 is the last logits row
        idx = fn_out + idx;
    }
    int r = -1;
    for (int i = 0; i < fn_out; ++i) {
        if (frows[i] == idx) {
            r = i;
            break;
        }
    }
    if (r < 0) {
        r = idx;
    }
    if (r < 0 || r >= fn_out || (size_t) (r + 1) * K > fy.size()) {
        return 0;
    }
    if (ck) {  // the top-k path kept the pre-norm row, and the combined kernel writes the full fallback row
        const float * xr = fy.data() + (size_t) r * K;
        ggml_fp32_to_bf16_row(xr, (ggml_bf16_t *) cx->vir, K);
        cx->flush(cx->vir, K * sizeof(ggml_bf16_t));
        if (!ffull) {
            ffull = axera_cmm_alloc(N * sizeof(float));
        }
        cin[ck->input("x")] = axera_io(cx, (size_t) K * sizeof(ggml_bf16_t));
        cout[ck->output("out")] = axera_io(ffull, (size_t) N * sizeof(float));
        ck->run(cin, cout, "output head (combined fallback row)");
        ffull->invalidate(ffull->vir, (size_t) N * sizeof(float));
        memcpy(out, (const float *) ffull->vir, (size_t) N * sizeof(float));
        return 1;
    }
    ggml_fp32_to_bf16_row(fy.data() + (size_t) r * K, (ggml_bf16_t *) x->vir, K);
    x->flush(x->vir, K * sizeof(ggml_bf16_t));
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k->info->nInputSize), out_io(k->info->nOutputSize);
    in[k->input("x")] = axera_io(x);
    in[k->input("w")] = axera_io(q, N * K / 2);
    in[k->input("s")] = axera_io(s, N * K / AXERA_MX_GROUP);
    if (R > 1) {
        *(int32_t *) mk->vir = 1;
        mk->flush();
        in[k->input("m")] = axera_io(mk);
        out_io[k->output("out")] = axera_io(yk);
        k->run(in, out_io, "output head (fallback row)");
        yk->invalidate(yk->vir, N * sizeof(float));
        memcpy(out, (const float *) yk->vir, N * sizeof(float));
    } else {
        if (!ffull) {
            ffull = axera_cmm_alloc(N * sizeof(float));
        }
        out_io[k->output("out")] = axera_io(ffull);
        k->run(in, out_io, "output head (fallback row)");
        ffull->invalidate(ffull->vir, N * sizeof(float));
        memcpy(out, (const float *) ffull->vir, N * sizeof(float));
    }
    return 1;
}

// the projection in f32, once
static void head_f32(const ggml_tensor * w, std::vector<float> & wf) {
    const int64_t N = w->ne[1], K = w->ne[0];
    if (!wf.empty()) {
        return;
    }
    if (N * K > (int64_t) 16 << 20) {  // (a text vocabulary's: the lm_head kernel's)
        GGML_ABORT("axera: an output head of %lld x %lld on the CPU\n", (long long) N, (long long) K);
    }
    wf.resize((size_t) N * K);
    const auto * tr = ggml_get_type_traits(w->type);
    axera_host_pool().run(N, [&](int64_t r) {
        const char * src = (const char *) w->data + r * w->nb[1];
        if (w->type == GGML_TYPE_F32) {
            memcpy(wf.data() + r * K, src, K * sizeof(float));
        } else {
            tr->to_float(src, wf.data() + r * K, K);
        }
    });
}

void output_head::run(ggml_cgraph * g, const graph_names & names, const std::vector<float> & h, int T,
                      const std::vector<char> * normed, const float * last, const std::vector<int> * compact_rows) {
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
    if (ck && gw != cgw) {
        memcpy(cnorm->vir, gw, (size_t) K * sizeof(float));
        cnorm->flush(cnorm->vir, (size_t) K * sizeof(float));
        cgw = gw;
    }
    // the output rows: the GET_ROWS index that crops the last layer or the final norm to the outputs
    ggml_tensor * out_ids = nullptr;
    if (n_out < T) {
        out_ids = names.find("out_ids");
        if (!out_ids || out_ids->type != GGML_TYPE_I32 || out_ids->ne[0] != n_out) {
            out_ids = nullptr;
            for (int i = 0; i < g->n_nodes; i++) {
                ggml_tensor * node = g->nodes[i];
                if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
                    node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
                    out_ids = node->src[1];
                }
            }
        }
    }
    if (n_out < T && !out_ids) {
        GGML_ABORT("axera: %d of %d rows requested but no output row index in the graph\n", n_out, T);
    }
    ggml_tensor *      rn = normed_rows(names);  // the normed rows, when the graph keeps them (embeddings: the
                                                 // TTS talker's hidden state for the code predictor)
    rn                    = rn && rn->type == GGML_TYPE_F32 && rn->ne[0] == K && rn->ne[1] == n_out ? rn : nullptr;
    std::vector<float> & y = yn;
    y.resize((size_t) n_out * K);
    rows.resize(n_out);
    axera_timer thn("head: rows + norm");
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= 0 && row < T);
        rows[r]          = row;
        const float * xr = h.data() + (size_t) (compact_rows ? r : row) * K;
        float *       yr = y.data() + (size_t) r * K;
        if (normed && (*normed)[row]) {  // the kernel's final norm
            memcpy(yr, xr, K * sizeof(float));
            if (rn) {
                memcpy((char *) rn->data + r * rn->nb[1], yr, K * sizeof(float));
            }
            continue;
        }
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
    thn.stop();
    if (rn && ggml_backend_buffer_is_axera(rn->buffer)) {
        ((ggml_backend_axera_buffer_context *) rn->buffer->context)->flush(rn->data, ggml_nbytes(rn));
    }
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    // the fused head: the decode row(s) through lm_head.topk, which applies the host sampler's penalty state and keeps
    // the exact top-K, so common/sampling can run the chain on K candidates instead of the vocabulary. A verification
    // batch fills all of the kernel's rows in one call (m: the valid ones).
    // the output rows are the whole batch: a single decode row or the verification's draft + target rows. A prompt
    // batch has many input rows but only its last output row, and keeps the regular head's logits.
    if (tk && !draft && n_out >= 1 && n_out <= tR && T == n_out && !last &&
        (!names.find("result_output") || graph_head(names) == names.find("result_output"))) {
        const npu_opts opts = npu_get_options();
        if (opts.enabled && opts.K == tK) {
            for (int r = 0; r < tR; ++r) {
                if (r < n_out) {
                    ggml_fp32_to_bf16_row(y.data() + (size_t) r * K, (ggml_bf16_t *) tx->vir + (size_t) r * K, K);
                } else {
                    memset((ggml_bf16_t *) tx->vir + (size_t) r * K, 0, K * sizeof(ggml_bf16_t));
                }
            }
            tx->flush(tx->vir, (size_t) tR * K * sizeof(ggml_bf16_t));
            std::vector<npu_penalty_row> upd;
            int64_t                      gen = 0;
            npu_take_updates(upd, &gen);
            if (gen != tgen) {  // a new request (or a reset): the neutral state
                std::fill((float *) tbias->vir, (float *) tbias->vir + N, 0.0f);
                std::fill((float *) tmulp->vir, (float *) tmulp->vir + N, 1.0f);
                std::fill((float *) tmuln->vir, (float *) tmuln->vir + N, 1.0f);
                tgen = gen;
            }
            for (const auto & r : upd) {
                GGML_ASSERT(r.token >= 0 && r.token < N);
                memcpy((float *) tbias->vir + r.token, &r.bias,    sizeof(float));
                memcpy((float *) tmulp->vir + r.token, &r.mul_pos, sizeof(float));
                memcpy((float *) tmuln->vir + r.token, &r.mul_neg, sizeof(float));
            }
            *(float *) ttemp->vir = opts.temp;
            if (tR > 1) {
                *(int32_t *) tkm->vir = n_out;
                tkm->flush();
            }
            std::vector<AX_ENGINE_IO_BUFFER_T> tin(tk->info->nInputSize), tout(tk->info->nOutputSize);
            tin[tk->input("x")]       = axera_io(tx, (size_t) tR * K * sizeof(ggml_bf16_t));
            tin[tk->input("w")]       = axera_io(q, mx4 ? (size_t) N * K / 2 : (size_t) N * K);
            tin[tk->input("s")]       = axera_io(s, N * K / AXERA_MX_GROUP);
            tin[tk->input("bias")]    = axera_io(tbias, N * sizeof(float));
            tin[tk->input("mul_pos")] = axera_io(tmulp, N * sizeof(float));
            tin[tk->input("mul_neg")] = axera_io(tmuln, N * sizeof(float));
            tin[tk->input("temp")]    = axera_io(ttemp, sizeof(float));
            if (tR > 1) {
                tin[tk->input("m")] = axera_io(tkm, sizeof(int32_t));
            }
            tout[tk->output("values")] = axera_io(tval, (size_t) tR * tK * sizeof(float));
            tout[tk->output("ids")]    = axera_io(tids, (size_t) tR * tK * sizeof(int32_t));
            tout[tk->output("vmax")]   = axera_io(tmax, (size_t) tR * sizeof(float));
            tout[tk->output("sumexp")] = axera_io(tsum, (size_t) tR * sizeof(float));
            {
                axera_timer ttk("lm_head.topk");
                tk->run(tin, tout, "lm_head.topk");
            }
            tval->invalidate(tval->vir, (size_t) tR * tK * sizeof(float));
            tids->invalidate(tids->vir, (size_t) tR * tK * sizeof(int32_t));
            tmax->invalidate(tmax->vir, (size_t) tR * sizeof(float));
            tsum->invalidate(tsum->vir, (size_t) tR * sizeof(float));
            memcpy(tv.data(), tval->vir, (size_t) tR * tK * sizeof(float));
            memcpy(ti.data(), tids->vir, (size_t) tR * tK * sizeof(int32_t));
            std::vector<uint8_t> usable(tR, 0);
            for (int r = 0; r < n_out; ++r) {
                const float * vr = tv.data() + (size_t) r * tK;
                const float   vm = ((const float *) tmax->vir)[r];
                const float   zs = ((const float *) tsum->vir)[r];
                // with no top-k the chain's top-p normalises over the whole vocabulary: the K candidates are enough only
                // when the nucleus (and the min-p set) ends inside them. Otherwise this row needs its full logits.
                bool fits = true;
                if (opts.top_k == 0) {
                    if (opts.min_p > 0.0f && vr[tK - 1] >= vm + opts.temp_minp * logf(opts.min_p)) {
                        fits = false;
                    }
                    if (fits && opts.top_p < 1.0f) {
                        const double thr = (double) opts.top_p * (double) zs;
                        double       cum = 0.0;
                        int          last = -1;
                        for (int i = 0; i < tK; ++i) {
                            cum += (double) expf((vr[i] - vm) / opts.temp);
                            if (cum >= thr) {
                                last = i;
                                break;
                            }
                        }
                        // the host cuts with float arithmetic on the same values; keep two candidates of headroom so a
                        // rounding difference at the last row cannot silently drop a token the full nucleus keeps
                        fits = last >= 0 && last + 2 < tK;
                    } else if (fits && opts.top_p >= 1.0f && opts.min_p <= 0.0f) {
                        fits = false;  // the distribution would run over the vocabulary
                    }
                }
                usable[r] = fits;
            }
            npu_publish(tv.data(), ti.data(), tK, n_out, (const float *) tmax->vir, (const float *) tsum->vir,
                        usable.data());
            // the sampler's per-row fallback: the rows of this run, their normed inputs and the graph's logits
            flogits = logits;
            fdctx   = dctx;
            fy.resize((size_t) n_out * K);  // the pre-norm rows, for the combined full-row fallback
            for (int r = 0; r < n_out; ++r) {
                const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
                memcpy(fy.data() + (size_t) r * K, h.data() + (size_t) (compact_rows ? r : row) * K,
                       (size_t) K * sizeof(float));
            }
            frows   = std::move(rows);
            fn_out  = n_out;
            npu_set_full_row(&output_head::full_row_cb, this);
            after(g, names);
            return;
        }
    }
    if (draft && dk && n_out == 1 && !(last && rows[0] == T - 1)) {  // the draft head
        draft = false;
        axera_timer tl("lm_head (draft)");
        ggml_fp32_to_bf16_row(y.data(), (ggml_bf16_t *) x->vir, K);
        x->flush(x->vir, K * sizeof(ggml_bf16_t));
        std::vector<AX_ENGINE_IO_BUFFER_T> din(dk->info->nInputSize), dout(dk->info->nOutputSize);
        din[dk->input("x")]   = axera_io(x, K * sizeof(ggml_bf16_t));
        din[dk->input("w")]   = axera_io(dq);
        din[dk->input("s")]   = axera_io(ds);
        if (dargmax) {  // one (local id, logit): no 65536-float write and no host top-K
            dout[dk->output("ids")]        = axera_io(did, sizeof(int32_t));
            dout[dk->output("top_logits")] = axera_io(dtop, sizeof(float));
            dk->run(din, dout, "draft head (argmax)");
            did->invalidate(did->vir, sizeof(int32_t));
            dtop->invalidate(dtop->vir, sizeof(float));
            tl.stop();
            int32_t lid = 0;
            float   lv  = 0.0f;
            memcpy(&lid, did->vir, sizeof(int32_t));
            memcpy(&lv, dtop->vir, sizeof(float));
            GGML_ASSERT(lid >= 0 && lid < (int32_t) dids.size());
            const int32_t gid = dids[(size_t) lid];
            float *       o   = (float *) logits->data;
            // Nothing is written to the full logits row: filling it costs about 0.9 ms (uncached NPU memory), and the
            // draft sampler takes the publication directly.  The row still reaches the sampler as a device copy; mark
            // it clean so that copy skips cache maintenance, and after() fills the chain's top-k output instead.
            if (!axera::argmax_wanted()) {  // no fused sampler (A/B): the full-row greedy scan needs the whole row
                std::fill(o, o + N, -INFINITY);
                o[gid] = lv;
                dctx->flush(o, N * sizeof(float));
            }
            dargmax_id  = gid;
            dargmax_pub = true;
            // the sampler's fused row: keyed by this graph's logits row; only a greedy chain takes it
            axera::argmax_publish(&gid, &lv, 1, logits->data, (size_t) logits->nb[1]);
            dctx->clean_p     = o;
            dctx->clean_n     = N * sizeof(float);
            dctx->clean_epoch = axera_graph_epoch;
            after(g, names);
            dargmax_pub = false;
            return;
        }
        dout[dk->output("out")] = axera_io(dy);
        dk->run(din, dout, "draft head");
        dy->invalidate();
        tl.stop();
        axera_timer   tf("lm_head (draft): logits");
        float *       o  = (float *) logits->data;
        const float * yv = (const float *) dy->vir;
        bool          tk = false;
        for (int i = 0; i < g->n_nodes && !tk; i++) {
            tk = g->nodes[i]->op == GGML_OP_TOP_K && g->nodes[i]->ne[0] <= 64;
        }
        if (!tk) {
            std::fill(o, o + N, -INFINITY);
            for (size_t i = 0; i < dids.size(); ++i) {
                o[dids[i]] = yv[i];
            }
        }
        // (only the host reads them: no write-back, and none when they are read back this graph)
        dctx->clean_p     = o;
        dctx->clean_n     = N * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
        tf.stop();
        top_k_draft  = this;
        draft_logits = tk ? o : nullptr;
        after(g, names);
        top_k_draft  = nullptr;
        draft_logits = nullptr;
        return;
    }
    draft = false;
    const kernel * kk = kv && n_out > 1 ? kv : k;
    const int      Rk = kk && kk->find_input("m") >= 0 ? (int) (kk->in_bytes("x") / (K * sizeof(ggml_bf16_t))) : 1;
    axera_timer tl(kk ? "lm_head" : "lm_head (cpu)");
    std::vector<AX_ENGINE_IO_BUFFER_T> in(kk ? kk->info->nInputSize : 0), out(kk ? kk->info->nOutputSize : 0);
    if (kk) {
        in[kk->input("x")] = axera_io(x);
        in[kk->input("w")] = axera_io(q, mx4 ? N * K / 2 : N * K);
        in[kk->input("s")] = axera_io(s, N * K / AXERA_MX_GROUP);
    }
    std::vector<int> kr;  // the output rows of a kernel of R > 1 rows
    for (int r = 0; r < n_out; ++r) {
        float * o = (float *) ((char *) logits->data + r * logits->nb[1]);
        if ((last && rows[r] == T - 1) || !kk) {
            if (last && rows[r] == T - 1) {
                memcpy(o, last, N * sizeof(float));
            } else {
                head_f32(w, wf);
                const float * yr = y.data() + (size_t) r * K;
                axera_host_pool().run(N, [&](int64_t n) {
                    const float * wr = wf.data() + n * K;
                    float         a  = 0.0f;
                    for (int64_t i = 0; i < K; ++i) {
                        a += wr[i] * yr[i];
                    }
                    o[n] = a;
                });
            }
            dctx->flush(o, N * sizeof(float));
            continue;
        }
        if (Rk > 1) {  // (R rows per call, below)
            kr.push_back(r);
            continue;
        }
        ggml_fp32_to_bf16_row(y.data() + (size_t) r * K, (ggml_bf16_t *) x->vir, K);
        x->flush(x->vir, K * sizeof(ggml_bf16_t));
        dctx->flush(o, N * sizeof(float));
        out[kk->output("out")] = axera_io(dctx, o, (uint32_t) (N * sizeof(float)));
        kk->run(in, out, "output head");
        dctx->invalidate(o, N * sizeof(float));
        dctx->clean_p     = o;  // the host reads these logits back without more cache maintenance this graph
        dctx->clean_n     = N * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    for (size_t i0 = 0; i0 < kr.size(); i0 += Rk) {  // the kernel's rows R at a time, the weights read once per call
        const int nr = (int) std::min<size_t>(Rk, kr.size() - i0);
        for (int j = 0; j < nr; ++j) {
            ggml_fp32_to_bf16_row(y.data() + (size_t) kr[i0 + j] * K, (ggml_bf16_t *) x->vir + (size_t) j * K, K);
        }
        x->flush();
        *(int32_t *) mk->vir = nr;
        mk->flush();
        in[kk->input("m")] = axera_io(mk);
        // R consecutive logits rows: the kernel writes them in place (a verification batch), else through y
        const bool direct = nr == Rk && kr[i0 + Rk - 1] == kr[i0] + Rk - 1 && logits->nb[1] == N * sizeof(float);
        if (direct) {
            float * o = (float *) ((char *) logits->data + kr[i0] * logits->nb[1]);
            dctx->flush(o, (size_t) Rk * N * sizeof(float));
            out[kk->output("out")] = axera_io(dctx, o, (uint32_t) (Rk * N * sizeof(float)));
            kk->run(in, out, "output head");
            dctx->invalidate(o, (size_t) Rk * N * sizeof(float));
            dctx->clean_p     = o;
            dctx->clean_n     = (size_t) Rk * N * sizeof(float);
            dctx->clean_epoch = axera_graph_epoch;
            continue;
        }
        out[kk->output("out")] = axera_io(yk);
        kk->run(in, out, "output head");
        yk->invalidate(yk->vir, (size_t) nr * N * sizeof(float));
        for (int j = 0; j < nr; ++j) {
            float * o = (float *) ((char *) logits->data + kr[i0 + j] * logits->nb[1]);
            memcpy(o, (const float *) yk->vir + (size_t) j * N, N * sizeof(float));
            dctx->flush(o, N * sizeof(float));
        }
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
    if (g->n_nodes > 0 && g->nodes[g->n_nodes - 1] == head) {
        return;  // the graph ends at the logits: no backend sampler follows
    }
    int                 il     = -1;
    for (int i = 0; i < g->n_nodes; i++) {
        il = g->nodes[i] == head ? i : il;
    }
    if (il < 0 || il + 1 >= g->n_nodes || padded != head) {
        return;
    }
    // (a TOP_K over the vocabulary on the host's workers: the CPU backend's runs one row on one thread)
    int i0 = il + 1;
    for (int i = il + 1; i < g->n_nodes; i++) {
        ggml_tensor * n = g->nodes[i];
        if (dargmax_pub && n->op == GGML_OP_TOP_K && n->type == GGML_TYPE_I32 && n->ne[0] <= 64 &&
            (n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            // the argmax draft head already picked the token: skip its pad and top-k scan over 248320 logits.  Every
            // candidate slot is the same token, so the greedy chain's pick is that token whatever k the sampler asks.
            int32_t * o = (int32_t *) n->data;
            std::fill(o, o + n->ne[0], dargmax_id);
            if (ggml_backend_buffer_is_axera(n->buffer)) {
                ((ggml_backend_axera_buffer_context *) n->buffer->context)->flush(n->data, ggml_nbytes(n));
            }
            i0 = i + 1;
            continue;
        }
        if (n->op == GGML_OP_TOP_K && n->src[0]->type == GGML_TYPE_F32 && n->type == GGML_TYPE_I32 &&
            (n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            cpu_compute(g, g->nodes + i0, i - i0, "sampler (cpu)");
            axera_timer tk("sampler: top_k (host)");
            top_k_rows(n);
            i0 = i + 1;
        } else if (n->op == GGML_OP_PAD && i == i0 && (draft_logits || dargmax_pub)) {  // (the TOP_K writes its rows)
            i0 = i + 1;
        } else if (n->op == GGML_OP_PAD && i == i0 && pad_rows(n)) {  // (the sampler's padded logits: a copy)
            i0 = i + 1;
        }
    }
    cpu_compute(g, g->nodes + i0, g->n_nodes - i0, "sampler (cpu)");
}

// a PAD of f32 2-D rows at the end of each dimension (zero), on the host; false if not that
bool pad_rows(ggml_tensor * n) {
    const ggml_tensor * x = n->src[0];
    const int32_t *     p = (const int32_t *) n->op_params;  // lp0, rp0, lp1, rp1, ...
    if (x->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32 || x->ne[2] != 1 || x->ne[3] != 1 || p[0] != 0 || p[2] != 0 ||
        p[4] != 0 || p[6] != 0 || p[8] != 0 || x->nb[0] != sizeof(float) || n->nb[0] != sizeof(float) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }
    axera_timer tp("sampler: pad (host)");
    for (int64_t r = 0; r < n->ne[1]; ++r) {
        float * d = (float *) ((char *) n->data + r * n->nb[1]);
        if (r < x->ne[1]) {
            memcpy(d, (const char *) x->data + r * x->nb[1], x->ne[0] * sizeof(float));
            std::fill(d + x->ne[0], d + n->ne[0], 0.0f);
        } else {
            std::fill(d, d + n->ne[0], 0.0f);
        }
    }
    if (ggml_backend_buffer_is_axera(n->buffer) && !top_k_draft) {  // (a draft head's: only the host reads it)
        ((ggml_backend_axera_buffer_context *) n->buffer->context)->flush(n->data, ggml_nbytes(n));
    }
    return true;
}

// ggml's TOP_K: the k largest of each row, the first two swapped ("no particular order"), as the CPU backend
void top_k_rows(ggml_tensor * n) {
    const ggml_tensor * src = n->src[0];
    const int64_t       N = src->ne[0], K = n->ne[0];
    const int           P = (int) std::min<int64_t>(16, std::max<int64_t>(1, N / 4096));
    auto greater = [](const float * d) {
        return [d](int32_t a, int32_t b) { return d[a] > d[b] || (d[a] == d[b] && a < b); };
    };
    if (top_k_draft && ggml_nrows(src) == 1 && N >= (int64_t) top_k_draft->w->ne[1] && K <= (int64_t) top_k_draft->dids.size()) {
        // the draft head's row: its vocabulary's logits (in dy, ids ascending); the rest are -inf (and padding)
        // one pass per worker keeping its K best (sorted, ties: the lower index first), then the workers' merged
        const auto &  ids = top_k_draft->dids;
        const float * y   = (const float *) top_k_draft->dy->vir;
        const int64_t NS  = (int64_t) ids.size();
        const int     P   = (int) std::min<int64_t>(8, std::max<int64_t>(1, NS / 8192));
        auto better = [y](int32_t a, int32_t b) { return y[a] > y[b] || (y[a] == y[b] && a < b); };
        std::vector<int32_t> cand((size_t) P * K);
        std::vector<int>     cn(P, 0);
        axera_host_pool().run(P, [&](int64_t p) {
            int32_t *     best = cand.data() + p * K;
            int           nb   = 0;
            const int64_t a = NS * p / P, b = NS * (p + 1) / P;
            for (int64_t i = a; i < b; ++i) {
                if (nb == K && !better((int32_t) i, best[K - 1])) {
                    continue;
                }
                int j = nb < K ? nb++ : (int) K - 1;
                while (j > 0 && better((int32_t) i, best[j - 1])) {
                    best[j] = best[j - 1];
                    --j;
                }
                best[j] = (int32_t) i;
            }
            cn[p] = nb;
        });
        std::vector<int32_t> all;
        for (int p = 0; p < P; ++p) {
            all.insert(all.end(), cand.begin() + p * K, cand.begin() + p * K + cn[p]);
        }
        std::partial_sort(all.begin(), all.begin() + K, all.end(), better);
        int32_t * o = (int32_t *) n->data;
        for (int64_t j = 0; j < K; ++j) {
            o[j] = ids[all[j]];
            if (draft_logits) {  // the picked rows' logits, in the logits and their padded copy
                draft_logits[ids[all[j]]]          = y[all[j]];
                ((float *) src->data)[ids[all[j]]] = y[all[j]];
            }
        }
        if (K > 1) {
            std::swap(o[0], o[1]);
        }
        if (ggml_backend_buffer_is_axera(n->buffer)) {
            ((ggml_backend_axera_buffer_context *) n->buffer->context)->flush(n->data, ggml_nbytes(n));
        }
        return;
    }
    for (int64_t r = 0; r < ggml_nrows(src); ++r) {
        const float *        d = (const float *) ((const char *) src->data + r * src->nb[1]);
        std::vector<int32_t> cand((size_t) P * K);
        std::vector<int>     cn(P, 0);
        axera_host_pool().run(P, [&](int64_t p) {
            const int64_t        a = N * p / P, b = N * (p + 1) / P;
            std::vector<int32_t> idx((size_t) (b - a));
            for (int64_t j = a; j < b; ++j) {
                idx[(size_t) (j - a)] = (int32_t) j;
            }
            const int64_t m = std::min<int64_t>(K, b - a);
            std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), greater(d));
            std::copy(idx.begin(), idx.begin() + m, cand.begin() + p * K);
            cn[p] = (int) m;
        });
        std::vector<int32_t> all;
        for (int p = 0; p < P; ++p) {
            all.insert(all.end(), cand.begin() + p * K, cand.begin() + p * K + cn[p]);
        }
        std::partial_sort(all.begin(), all.begin() + K, all.end(), greater(d));
        int32_t * o = (int32_t *) ((char *) n->data + r * n->nb[1]);
        std::copy(all.begin(), all.begin() + K, o);
        if (K > 1) {
            std::swap(o[0], o[1]);
        }
    }
    if (ggml_backend_buffer_is_axera(n->buffer)) {
        ((ggml_backend_axera_buffer_context *) n->buffer->context)->flush(n->data, ggml_nbytes(n));
    }
}

//
// CPU nodes
//

void cpu_compute(ggml_cgraph * g, ggml_tensor ** nodes, int n, const char * what) {
    if (n <= 0) {
        return;
    }
    // a slice of small tensors on one thread: waking the CPU backend's threads for each of its nodes costs more than
    // the nodes (the Qwen3-TTS graphs' slices of one row)
    int64_t big = 0;  // the largest node's work: its elements (times K for a matrix product)
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) && !ggml_is_empty(t) && t->op != GGML_OP_VIEW &&
            t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE && t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_NONE) {
            big = std::max(big, ggml_nelements(t) * (t->op == GGML_OP_MUL_MAT ? t->src[0]->ne[0] : 1));
        }
    }
    const int                   c = big <= 65536 ? 1 : 0;
    static std::mutex           mu;
    static ggml_backend_t       cpu[2] = {};
    std::lock_guard<std::mutex> lock(mu);
    if (!cpu[c]) {  // (through the registry: this backend does not link the CPU backend)
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "axera: no CPU backend");
        cpu[c]                 = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const int          nt  = c ? 1 : (int) std::max(1u, std::thread::hardware_concurrency());
        auto set_nt = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_nt) {
            set_nt(cpu[c], nt);
        }
        // a threadpool of its own: without one the CPU backend starts its threads on every call (adapters call it once
        // per slice of a graph, thousands of times per graph for mtmd's generation graphs)
        auto tp_new = (ggml_threadpool_t (*)(struct ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto tp_set = (void (*)(ggml_backend_t, ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (tp_new && tp_set) {
            struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nt);
            tpp.poll                          = 0;  // no spinning between calls: the NPU and the adapter's host work run then
            tp_set(cpu[c], tp_new(&tpp));           // (kept for the process)
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
    if (ggml_backend_graph_compute(cpu[c], &view) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("axera: %s failed on the CPU backend\n", what);
    }
}

}  // namespace axera
