// Axera NPU backend: Qwen3.5 dense models, Qwen3.5-9B and Qwen3.8-27B (gated delta net and gated attention layers,
// dense MLP)
//
// One path, the manifest's kernels (ax-dsl-kernels tools/manifest, model qwen35_dense):
//   decode   decode.group: G consecutive layers per call (the layers' weights in packs), L / G calls per token; a
//            whole-model group with the output head (9B: a logits output) writes a single token's logits straight
//            into the graph's, else lm_head after the final norm
//   prefill  chunks of P consecutive cells, largest first: prefill128.*, 64, 16, 4. A prompt's tail of fewer tokens
//            than the largest chunk runs as one chunk of the smallest size that holds it, padded with copies of its
//            last row: the delta-net kernels take the valid rows (rv / cix: the padding leaves the state alone), the
//            padding's K / V go to cells after the batch's that no token sees. A single token through the decode group
// The recurrent state and the KV cache live in the llama.cpp memory (cache_s / cache_r, cache_k / cache_v in CMM),
// one sequence. Weights only from the GGUF (core/params).

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cmath>

namespace axera {
namespace {

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    struct dyn {
        bool out;
        int  idx, layer, what;  // what: 0 state, 1 conv, 2 kc, 3 vc
    };
    std::vector<dyn> d;
    int              logits = -1;  // the output index of a whole-model group's logits
};

struct chunk {  // one prefill size's buffers
    int            P = 0;
    const kernel * gdn = nullptr, *attn = nullptr;
    axera_cmm      A, B, MASK, COS, SIN, POS, NBLK, RV, CIX;
    int            cix_n = 0;
    std::vector<call> calls;  // per layer
};

class qwen35_dense final : public model {
  public:
    explicit qwen35_dense(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest &  man;
    params            prm;
    graph_names       names;
    const void *      bound = nullptr;
    int               L = 0, H = 0, G = 0, BC = 0, ROT = 0;
    int64_t           N = 0, LMAX = 0;
    std::vector<bool> gdn;
    struct cache {
        ggml_backend_axera_buffer_context * b[2]      = {};
        char *                              base[2]   = {};
        size_t                              stride[2] = {}, size[2] = {};
    };
    std::vector<cache> lc;
    int64_t            row = 0, stream = 0;

    std::vector<call> dec;
    axera_cmm         A, COS, SIN, CS, PN, MASK, LOGITS;
    output_head       head;
    std::vector<chunk> sizes;  // largest first

    void init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_);
    AX_ENGINE_IO_BUFFER_T cache_io(int il, int what) const;
    void set_place(int64_t r, int64_t s);
    void zero_row(int64_t r);
    call bind_layer(const std::map<std::string, ggml_tensor *> & t, const kernel & k, int il, chunk & c);
    ggml_tensor * group_logits(int T) const;
};

AX_ENGINE_IO_BUFFER_T qwen35_dense::cache_io(int il, int what) const {
    const cache & c = lc[il];
    const int     w = what % 2;
    const int64_t o = what < 2 ? row : stream;
    return axera_io(c.b[w], c.base[w] + o * c.stride[w], (uint32_t) c.size[w]);
}

void qwen35_dense::set_place(int64_t r, int64_t s) {
    if (r == row && s == stream) {
        return;
    }
    row    = r;
    stream = s;
    auto fix = [&](call & c) {
        for (const auto & d : c.d) {
            (d.out ? c.out : c.in)[d.idx] = cache_io(d.layer, d.what);
        }
    };
    for (auto & c : dec) {
        fix(c);
    }
    for (auto & ch : sizes) {
        for (auto & c : ch.calls) {
            fix(c);
        }
    }
}

void qwen35_dense::zero_row(int64_t r) {
    for (int il = 0; il < L; ++il) {
        for (int w = 0; w < 2 && gdn[il]; ++w) {
            char * p = lc[il].base[w] + r * lc[il].stride[w];
            memset(p, 0, lc[il].size[w]);
            lc[il].b[w]->flush(p, lc[il].size[w]);
        }
    }
}

// a prefill layer kernel's IO for chunk c of layer il: x / out the chunk's A / B by the layer's parity
call qwen35_dense::bind_layer(const std::map<std::string, ggml_tensor *> & t, const kernel & k, int il, chunk & c) {
    call cl;
    cl.k = &k;
    const auto & xi = il % 2 == 0 ? c.A : c.B;
    const auto & xo = il % 2 == 0 ? c.B : c.A;
    for (const auto & d : k.desc->in) {
        const std::string & n = d.name;
        if (!d.param.empty()) {
            cl.in.push_back(prm.get(t, il, d.kind, d.param));
        } else if (k.desc->packs.count(n)) {
            cl.in.push_back(axera_io(prm.pack(t, *k.desc, n, il), d.bytes));
        } else if (n == "x") {
            cl.in.push_back(axera_io(xi, d.bytes));
        } else if (n == "state_in" || n == "conv_in") {
            cl.d.push_back({ false, (int) cl.in.size(), il, n == "state_in" ? 0 : 1 });
            cl.in.push_back(cache_io(il, n == "state_in" ? 0 : 1));
        } else {
            const axera_cmm * b = n == "cos" ? &c.COS : n == "sin" ? &c.SIN : n == "pos" ? &c.POS : n == "nblk" ? &c.NBLK :
                                  n == "mask" ? &c.MASK : n == "rv" ? &c.RV : n == "cix" ? &c.CIX : nullptr;
            if (!b || !*b) {
                GGML_ABORT("axera qwen35_dense: %s: unexpected input %s\n", k.desc->role.c_str(), n.c_str());
            }
            cl.in.push_back(axera_io(*b, d.bytes));
        }
    }
    for (const auto & d : k.desc->out) {
        const std::string & n = d.name;
        if (n == "out") {
            cl.out.push_back(axera_io(xo, d.bytes));
            continue;
        }
        const int w = n == "state_out" ? 0 : n == "conv_out" ? 1 : n == "kc" ? 2 : n == "vc" ? 3 : -1;
        if (w < 0) {
            GGML_ABORT("axera qwen35_dense: %s: unexpected output %s\n", k.desc->role.c_str(), n.c_str());
        }
        cl.d.push_back({ true, (int) cl.out.size(), il, w });
        cl.out.push_back(cache_io(il, w));
    }
    return cl;
}

void qwen35_dense::init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_) {
    L = L_;
    H = H_;
    const kernel & dk = man.get("decode.group");
    gdn.assign(L, false);
    for (int il = 0; il < L; ++il) {
        gdn[il] = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
    }
    std::string pat;  // the group kernel's layers: st<j> outputs (delta net), kc<j> (attention)
    for (const auto & d : dk.desc->out) {
        if (d.name.rfind("st", 0) == 0 || d.name.rfind("kc", 0) == 0) {
            const int j = atoi(d.name.c_str() + 2);
            pat.resize(std::max<size_t>(pat.size(), j + 1), '?');
            pat[j] = d.name[0] == 's' ? 'g' : 'a';
        }
    }
    G = (int) pat.size();
    for (int il = 0; il < L; ++il) {
        if (L % G != 0 || (gdn[il] ? 'g' : 'a') != pat[il % G]) {
            GGML_ABORT("axera qwen35_dense: %d layers do not repeat the decode group's pattern %s (%s)\n", L, pat.c_str(),
                       dk.desc->file.c_str());
        }
    }
    const io_desc * kc = nullptr, *st = nullptr, *cv = nullptr;
    for (const auto & d : dk.desc->out) {
        kc = d.name == "kc" + std::to_string(pat.find('a')) ? &d : kc;
        st = d.name == "st0" ? &d : st;
        cv = d.name == "cv0" ? &d : cv;
    }
    GGML_ASSERT(kc && st && cv);
    lc.assign(L, {});
    N = 0;
    for (int il = 0; il < L; ++il) {
        for (int w = 0; w < 2; ++w) {
            const std::string cn = std::string("cache_") + (gdn[il] ? (w ? "r" : "s") : (w ? "v" : "k")) + "_l" + std::to_string(il);
            ggml_tensor *     c  = t.count(cn) ? t.at(cn) : nullptr;
            const uint32_t    sz = gdn[il] ? (w ? cv->bytes : st->bytes) : kc->bytes;
            if (!c || !ggml_backend_buffer_is_axera(c->buffer) || !ggml_is_contiguous(c)) {
                GGML_ABORT("axera qwen35_dense: %s not in CMM (all layers on the device: -ngl 99)\n", cn.c_str());
            }
            if (gdn[il]) {
                GGML_ASSERT(c->nb[1] >= sz);
                lc[il].stride[w] = c->nb[1];
            } else {
                GGML_ASSERT(c->type == GGML_TYPE_F16 && c->ne[3] == 1 && (N == 0 || c->ne[1] == N));
                N = c->ne[1];
                if (N > man.context || c->nb[1] * man.context != sz ||
                    ggml_backend_buft_get_alloc_size(c->buffer->buft, c) < (c->ne[2] - 1) * c->nb[2] + sz) {
                    GGML_ABORT("axera qwen35_dense: a KV cache of %lld cells (-c), the kernels hold %lld\n", (long long) N,
                               (long long) man.context);
                }
                lc[il].stride[w] = c->nb[2];
            }
            lc[il].b[w]    = (ggml_backend_axera_buffer_context *) c->buffer->context;
            lc[il].base[w] = (char *) c->data;
            lc[il].size[w] = sz;
        }
    }
    row = stream = 0;
    for (const auto & d : dk.desc->in) {
        if (d.name == "mask") {
            GGML_ASSERT(d.shape.size() == 2);
            LMAX = d.shape[0] * d.shape[1];
            BC   = (int) d.shape[1];
        }
    }
    GGML_ASSERT(LMAX == man.context && N % BC == 0);

    // decode: x / out ping-pong A (every group starts at an even layer and ends at an odd one); cs = cos | sin,
    // pn = pos | nblk
    const uint32_t csb = dk.in_bytes("cs");
    ROT  = (int) (csb / sizeof(ggml_bf16_t));  // [2][ROT / 2]: ROT / 2 cos then ROT / 2 sin
    A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    COS  = axera_cmm_alloc(csb / 2);
    SIN  = axera_cmm_alloc(csb / 2);
    CS   = axera_cmm_alloc(csb);
    PN   = axera_cmm_alloc(2 * sizeof(int32_t));
    MASK = axera_cmm_alloc(dk.in_bytes("mask"));
    LOGITS = {};
    GGML_ASSERT(G % 2 == 0 && dk.in_bytes("pn") == 2 * sizeof(int32_t));
    dec.assign(L / G, {});
    for (int gi = 0; gi < L / G; ++gi) {
        call & c = dec[gi];
        c.k      = &dk;
        for (const auto & d : dk.desc->in) {
            const std::string & n = d.name;
            if (dk.desc->packs.count(n)) {
                c.in.push_back(axera_io(prm.pack(t, *dk.desc, n, gi * G), d.bytes));
            } else {
                const axera_cmm * b = n == "x" ? &A : n == "cs" ? &CS : n == "pn" ? &PN : n == "mask" ? &MASK : nullptr;
                if (!b) {
                    GGML_ABORT("axera qwen35_dense: decode.group: unexpected input %s\n", n.c_str());
                }
                c.in.push_back(axera_io(*b, d.bytes));
            }
        }
        for (const auto & d : dk.desc->out) {
            const std::string & n = d.name;
            if (n == "out") {
                c.out.push_back(axera_io(A, d.bytes));
                continue;
            }
            if (n == "logits") {  // a whole-model group: by default into LOGITS, a single token's into the graph's
                GGML_ASSERT(G == L);
                LOGITS   = axera_cmm_alloc(d.bytes);
                c.logits = (int) c.out.size();
                c.out.push_back(axera_io(LOGITS, d.bytes));
                continue;
            }
            const int j = atoi(n.c_str() + 2), il = gi * G + j;
            const int w = n.rfind("st", 0) == 0 ? 0 : n.rfind("cv", 0) == 0 ? 1 : n.rfind("kc", 0) == 0 ? 2 : 3;
            c.d.push_back({ true, (int) c.out.size(), il, w });
            c.out.push_back(cache_io(il, w));
        }
    }

    // prefill sizes, largest first
    sizes.clear();
    for (int P : { 128, 64, 16, 4 }) {
        sizes.emplace_back();
        chunk & c = sizes.back();
        c.P       = P;
        c.gdn     = &man.get("prefill" + std::to_string(P) + ".gdn");
        c.attn    = &man.get("prefill" + std::to_string(P) + ".attn");
        const size_t xr = (size_t) P * H * sizeof(ggml_bf16_t);
        GGML_ASSERT(c.gdn->in_bytes("x") == xr && c.attn->in_bytes("mask") == (size_t) P * LMAX * sizeof(ggml_bf16_t) &&
                    c.gdn->in_bytes("rv") == P * sizeof(float));
        c.A     = axera_cmm_alloc(xr);
        c.B     = axera_cmm_alloc(xr);
        c.MASK  = axera_cmm_alloc(c.attn->in_bytes("mask"));
        c.COS   = axera_cmm_alloc(c.attn->in_bytes("cos"));
        c.SIN   = axera_cmm_alloc(c.attn->in_bytes("sin"));
        c.POS   = axera_cmm_alloc(sizeof(int32_t));
        c.NBLK  = axera_cmm_alloc(sizeof(int32_t));
        c.RV    = axera_cmm_alloc(c.gdn->in_bytes("rv"));
        c.CIX   = axera_cmm_alloc(c.gdn->in_bytes("cix"));
        c.cix_n = (int) (c.gdn->in_bytes("cix") / sizeof(int32_t));
    }
    for (auto & c : sizes) {
        for (int il = 0; il < L; ++il) {
            c.calls.push_back(bind_layer(t, gdn[il] ? *c.gdn : *c.attn, il, c));
        }
    }
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    for (auto & ch : sizes) {
        for (auto & c : ch.calls) {
            live.push_back(&c.in);
        }
    }
    prm.release(live);
    GGML_LOG_INFO("axera qwen35_dense: %d layers (%s per decode call), KV cache %lld cells (kernels %lld)\n", L,
                  pat.c_str(), (long long) N, (long long) LMAX);
}

// the graph's logits a whole-model group can write: a one-token batch whose graph wants only its logits
ggml_tensor * qwen35_dense::group_logits(int T) const {
    ggml_tensor * lg = LOGITS && T == 1 ? graph_head(names) : nullptr;
    return lg && lg->ne[1] == 1 && lg->type == GGML_TYPE_F32 && ggml_backend_buffer_is_axera(lg->buffer) &&
                   ggml_nbytes(lg) == LOGITS->size && names.find("result_output") == lg ? lg : nullptr;
}

ggml_status qwen35_dense::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_dense: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) {
            ++n0;
        }
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0           = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
        ++L_;
    }
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, L_, H_);
        bound = w0->data;
    }
    auto x0_row = [&](int r) { return (const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]); };

    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm &&
                (kqm->ne[3] == 1 ? kqm->ne[1] >= T : kqm->ne[1] * kqm->ne[3] == T) && kqm->ne[0] <= N &&
                (kqm->type == GGML_TYPE_F16 || kqm->type == GGML_TYPE_F32));
    const kv_view kv(k_idxs, kqm, N, BC);
    int64_t       n_seqs = 1, rs_head = 0, rs_z = -1;
    int           g0 = 0;
    while (!gdn[g0]) {
        ++g0;
    }
    const ggml_tensor * s0   = names.at("cache_s_l" + std::to_string(g0));
    int64_t             best = -1;
    for (int i = 0; i < g->n_nodes; i++) {
        const ggml_tensor * node = g->nodes[i];
        if (node->view_src != s0 || ggml_nelements(node) == 0) {
            continue;
        }
        if (node->op == GGML_OP_CPY && (best < 0 || (int64_t) node->view_offs < best)) {
            best   = (int64_t) node->view_offs;
            n_seqs = node->ne[1];
            if (node->ne[2] > 1) {
                GGML_LOG_ERROR("axera qwen35_dense: recurrent-state snapshots (speculative decoding) are not supported\n");
                return GGML_STATUS_FAILED;
            }
        } else if (node->op == GGML_OP_SCALE) {
            rs_z = (int64_t) (node->view_offs / s0->nb[1]);
        }
    }
    rs_head = best < 0 ? 0 : best / (int64_t) s0->nb[1];
    if (std::max<int64_t>(1, n_seqs) != 1) {
        GGML_LOG_ERROR("axera qwen35_dense: %lld sequences in a batch; one sequence only (-np 1)\n", (long long) n_seqs);
        return GGML_STATUS_FAILED;
    }
    if (rs_z >= 0) {
        zero_row(rs_z);
    }
    if (const ggml_tensor * sc = names.find("rs_s_copy")) {
        bool identity = true;
        for (int64_t i = 0; i < sc->ne[0]; ++i) {
            identity = identity && ((const int32_t *) sc->data)[i] == rs_head + i;
        }
        for (int il = 0; il < L && !identity; ++il) {
            for (int w = 0; w < 2 && gdn[il]; ++w) {
                cache &           c = lc[il];
                std::vector<char> tmp((size_t) sc->ne[0] * c.stride[w]);
                for (int64_t i = 0; i < sc->ne[0]; ++i) {
                    char * src = c.base[w] + (int64_t) ((const int32_t *) sc->data)[i] * c.stride[w];
                    c.b[w]->flush(src, c.stride[w]);
                    c.b[w]->invalidate(src, c.stride[w]);
                    memcpy(tmp.data() + i * c.stride[w], src, c.stride[w]);
                }
                for (int64_t i = 0; i < sc->ne[0]; ++i) {
                    char * dst = c.base[w] + (rs_head + i) * c.stride[w];
                    memcpy(dst, tmp.data() + i * c.stride[w], c.stride[w]);
                    c.b[w]->flush(dst, c.stride[w]);
                }
            }
        }
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera qwen35_dense: no ROPE node in the graph");
    const int32_t *    pos = (const int32_t *) inp_pos->data;
    const rope         rp(rope_node, inp_pos, T);
    GGML_ASSERT(rp.ROT == ROT);
    std::vector<float> h((size_t) T * H);
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const uint16_t     ninf = ggml_fp32_to_bf16(-3e38f).bits;
    bool               group_ran = false;  // the whole-model group computed the (single) token's logits
    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        set_place(rs_head, kv.stream_of(it));
        if (p == 0) {
            zero_row(row);
        }
        // consecutive cells of one stream from token it
        int run = 1;
        while (it + run < T && kv.stream_of(it + run) == kv.stream_of(it) && kv.cell_of(it + run) == kv.cell_of(it) + run) {
            ++run;
        }
        int v = -1, n = 0;  // the chunk size, its real tokens
        // fewer than the largest chunk: the smallest size that holds them, padded, if the contexts hold one sequence
        // (the cells after the batch's are free), its padding cells are hidden from the last token (and so from every
        // token) and lie within the cache
        int fit = -1;
        for (size_t s = 0; s < sizes.size() && sizes[s].P >= run; ++s) {
            fit = (int) s;
        }
        if (fit >= 0 && sizes[fit].P != run && run >= 2 && axera_n_seq_max() == 1) {
            const int     P  = sizes[fit].P;
            const int64_t c0 = kv.cell_of(it);
            bool          ok = c0 + P <= N;
            if (ok) {
                std::vector<ggml_bf16_t> last(kv.W);
                kv.mask_row(it + run - 1, last.data());
                for (int64_t c = c0 + run; ok && c < std::min<int64_t>(c0 + P, kv.W); ++c) {
                    ok = last[c].bits == ninf;
                }
            }
            if (ok) {
                v = fit;
                n = run;
            }
        }
        for (size_t s = 0; s < sizes.size() && v < 0; ++s) {  // else the largest chunk that fits exactly
            if (sizes[s].P <= run && kv.cell_of(it) + sizes[s].P <= N) {
                v = (int) s;
                n = sizes[s].P;
            }
        }
        if (v >= 0) {
            chunk &   c = sizes[v];
            const int P = c.P;
            axera_host_pool().run(P, [&](int64_t r) {
                ggml_fp32_to_bf16_row(x0_row(it + std::min((int) r, n - 1)), (ggml_bf16_t *) c.A->vir + (size_t) r * H, H);
            });
            c.A->flush(c.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            auto * rv  = (float *) c.RV->vir;  // the delta-net kernels' valid rows, the conv input rows after them
            auto * cix = (int32_t *) c.CIX->vir;
            for (int r = 0; r < P; ++r) {
                rv[r] = r < n ? 1.0f : 0.0f;
            }
            for (int j = 0; j < c.cix_n; ++j) {
                cix[j] = n + j;
            }
            c.RV->flush(c.RV->vir, (size_t) P * sizeof(float));
            c.CIX->flush(c.CIX->vir, (size_t) c.cix_n * sizeof(int32_t));
            rp.rows(it, n, c.COS.get(), c.SIN.get());
            if (n < P) {  // the padding rows' rotary rows: the last real token's
                for (auto * cs : { c.COS.get(), c.SIN.get() }) {
                    auto * r0 = (ggml_bf16_t *) cs->vir;
                    for (int r = n; r < P; ++r) {
                        memcpy(r0 + (size_t) r * (ROT / 2), r0 + (size_t) (n - 1) * (ROT / 2), (ROT / 2) * sizeof(ggml_bf16_t));
                    }
                    cs->flush();
                }
            }
            const int nblk           = kv.chunk_mask(pos, it, P, n, (ggml_bf16_t *) c.MASK->vir);
            *(int32_t *) c.POS->vir  = (int32_t) kv.cell_of(it);
            *(int32_t *) c.NBLK->vir = nblk;
            c.MASK->flush(c.MASK->vir, (size_t) nblk * P * BC * sizeof(ggml_bf16_t));
            c.POS->flush();
            c.NBLK->flush();
            for (int il = 0; il < L; ++il) {
                axera_timer tl(gdn[il] ? "gdn prefill" : "attn prefill");
                c.calls[il].k->run(c.calls[il].in, c.calls[il].out, "prefill layer");
            }
            const axera_cmm & fin = L % 2 == 0 ? c.A : c.B;
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            axera_host_pool().run(n, [&](int64_t r) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
            });
            it += n;
            continue;
        }
        // one token through the decode groups
        ggml_fp32_to_bf16_row(x0_row(it), (ggml_bf16_t *) A->vir, H);
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        rp.rows(it, 1, COS.get(), SIN.get());
        const int nblk = kv.mask_row(it, (ggml_bf16_t *) MASK->vir);
        MASK->flush(MASK->vir, (size_t) nblk * BC * sizeof(ggml_bf16_t));
        {
            const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);  // (the CMM buffers are larger: 128-byte multiples)
            memcpy(CS->vir, COS->vir, half);
            memcpy((char *) CS->vir + half, SIN->vir, half);
            CS->flush();
        }
        ((int32_t *) PN->vir)[0] = (int32_t) kv.cell_of(it);
        ((int32_t *) PN->vir)[1] = nblk;
        PN->flush();
        ggml_tensor * direct = group_logits(T);
        if (direct) {  // no dirty lines left to be written back over the kernel's logits
            auto * dctx = (ggml_backend_axera_buffer_context *) direct->buffer->context;
            dctx->flush(direct->data, ggml_nbytes(direct));
            dec.back().out[dec.back().logits] = axera_io(dctx, direct->data, (uint32_t) ggml_nbytes(direct));
        }
        {
            axera_timer tl("layers (decode)");
            for (auto & c : dec) {
                c.k->run(c.in, c.out, "decode group");
            }
        }
        if (direct) {
            dec.back().out[dec.back().logits] = axera_io(LOGITS, (uint32_t) LOGITS->size);
        }
        group_ran = direct != nullptr;
        A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, h.data() + (size_t) it * H, H);
        ++it;
    }
    if (group_ran) {  // written in place: drop the cached lines, get_tensor reads without maintenance
        axera_timer tl("lm_head (group)");
        auto *      dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
        dctx->invalidate(logits->data, ggml_nbytes(logits));
        dctx->clean_p     = logits->data;
        dctx->clean_n     = ggml_nbytes(logits);
        dctx->clean_epoch = axera_graph_epoch;
        output_head::after(g, names);
    } else {
        head.bind(man.get("lm_head"), logits->src[0]);
        head.run(g, names, h, T);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_dense(const manifest & m) {
    return std::make_shared<qwen35_dense>(m);
}

}  // namespace axera
