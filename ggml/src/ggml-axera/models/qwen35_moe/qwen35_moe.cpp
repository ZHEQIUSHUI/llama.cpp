// Axera NPU backend: Qwen3.5-35B-A3B (qwen35moe: gated delta net and gated attention layers, routed experts)
//
// One path, the manifest's kernels (ax-dsl-kernels tools/manifest, model qwen35_moe):
//   decode   decode.group: G consecutive whole MoE layers per call (layer, router, top-K and experts on the NPU),
//            L / G calls per token; then lm_head
//   prefill  chunks of P consecutive cells, largest first: prefill256.* (with the router's top-K), 128, 64, 16, 4;
//            per layer the chunk's layer kernel, then its routed experts: runs of up to 2048 / 256 chunks of 256 as one
//            group (experts grouped by expert into slots: moe.slots*, then moe.combine_t* adds them to each token's
//            base), other chunks through the slots kernel and a host combine; fewer tokens than the smallest chunk
//            through the decode group
// The recurrent state and the KV cache live in the llama.cpp memory (cache_s / cache_r, cache_k / cache_v in CMM),
// one sequence. Weights only from the GGUF (core/params).

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace axera {
namespace {

// a kernel call's IO, the entries that follow the recurrent-cache row or the KV stream
struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    struct dyn {
        bool out;
        int  idx, layer, what;  // what: 0 state, 1 conv, 2 kc, 3 vc
    };
    std::vector<dyn> d;
};

// one prefill chunk's own buffers (the group's chunks: views of the group's)
struct chunk {
    int       P = 0;
    axera_cmm A, B, H2, BASE, LOG, TKV, TKI, MASK, COS, SIN, POS, NBLK;
};

struct slots_k {  // moe.slots<S>_g<G>: S slots of G token rows, each one expert
    const kernel * k = nullptr;
    int            S = 0, G = 0;
};

struct combine_k {  // moe.combine_t<T>: T tokens' K slot rows added to their base
    const kernel * k = nullptr;
    int            T = 0, R = 0;
};

class qwen35_moe final : public model {
  public:
    explicit qwen35_moe(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;  // blk.0.attn_norm.weight's data the IO is bound to
    int              L = 0, H = 0, K = 0, NE = 0, G = 0, BC = 0;
    int64_t          N = 0, LMAX = 0;
    std::vector<bool> gdn;
    struct cache {  // a layer's llama.cpp cache: [0] state / k, [1] conv / v
        ggml_backend_axera_buffer_context * b[2]    = {};
        char *                              base[2] = {};
        size_t                              stride[2] = {}, size[2] = {};
    };
    std::vector<cache> lc;
    int64_t            row = 0, stream = 0;  // the cache row / KV stream the IO points at

    // decode
    std::vector<call> dec;  // one per group
    axera_cmm         A, COS, SIN, POS, NBLK, MASK;
    output_head       head;

    // prefill: per size (largest first) the layer kernels and, per chunk, the calls of each layer
    struct size_k {
        int            P = 0;
        const kernel * gdn = nullptr, *attn = nullptr;
        bool           tk  = false;  // the kernels output the router's top-K
    };
    std::vector<size_k>            sizes;
    std::vector<chunk>             grp;    // the chunks of a group (sizes[0])
    std::map<int, chunk>           single;  // P -> the chunk of the other sizes
    std::vector<std::vector<call>> grp_calls;               // [c][il]
    std::map<int, std::vector<call>> single_calls;          // P -> [il]
    axera_cmm                      GA, GB, GH2, GBASE, GTKV, GTKI;

    // routed experts of prefill
    slots_k                main, big;
    std::vector<slots_k>   rem, big_rem;  // fewer slots per call, fewest first
    std::vector<combine_k> combs;         // fewest tokens first
    axera_cmm              TOK, SRW, SOUT, LOC, SX;

    void init(ggml_cgraph * g, const std::map<std::string, ggml_tensor *> & t, int L_, int H_);
    AX_ENGINE_IO_BUFFER_T cache_io(int il, int what) const;
    void set_place(int64_t r, int64_t s);
    void zero_row(int64_t r);
    call bind_layer(const std::map<std::string, ggml_tensor *> & t, const kernel & k, int il, chunk & c);
    void run(call & c, const char * what);
    void experts_group(int il, int ng);
    void experts_single(int il, chunk & c);
};

AX_ENGINE_IO_BUFFER_T qwen35_moe::cache_io(int il, int what) const {
    const cache & c = lc[il];
    const int     w = what % 2;
    const int64_t o = what < 2 ? row : stream;
    return axera_io(c.b[w], c.base[w] + o * c.stride[w], (uint32_t) c.size[w]);
}

void qwen35_moe::set_place(int64_t r, int64_t s) {
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
    for (auto & cs : grp_calls) {
        for (auto & c : cs) {
            fix(c);
        }
    }
    for (auto & [P, cs] : single_calls) {
        for (auto & c : cs) {
            fix(c);
        }
    }
}

void qwen35_moe::zero_row(int64_t r) {  // a new sequence: its recurrent state cleared
    for (int il = 0; il < L; ++il) {
        for (int w = 0; w < 2 && gdn[il]; ++w) {
            char * p = lc[il].base[w] + r * lc[il].stride[w];
            memset(p, 0, lc[il].size[w]);
            lc[il].b[w]->flush(p, lc[il].size[w]);
        }
    }
}

void qwen35_moe::run(call & c, const char * what) {
    axera_timer tm(what);
    c.k->run(c.in, c.out, what);
}

// a prefill layer kernel's IO for chunk c of layer il: x the chunk's A / B by the layer's parity, the out the chunk's
// base (the experts add to it), the state and KV in the llama.cpp cache
call qwen35_moe::bind_layer(const std::map<std::string, ggml_tensor *> & t, const kernel & k, int il, chunk & c) {
    call cl;
    cl.k = &k;
    const auto & x = il % 2 == 0 ? c.A : c.B;
    for (const auto & d : k.desc->in) {
        const std::string & n = d.name;
        if (!d.param.empty()) {
            cl.in.push_back(prm.get(t, il, d.kind, d.param));
        } else if (k.desc->packs.count(n)) {
            cl.in.push_back(axera_io(prm.pack(t, *k.desc, n, il), d.bytes));
        } else if (n == "x") {
            cl.in.push_back(axera_io(x, d.bytes));
        } else if (n == "state_in" || n == "conv_in") {
            cl.d.push_back({ false, (int) cl.in.size(), il, n == "state_in" ? 0 : 1 });
            cl.in.push_back(cache_io(il, n == "state_in" ? 0 : 1));
        } else {
            const axera_cmm * b = n == "cos" ? &c.COS : n == "sin" ? &c.SIN : n == "pos" ? &c.POS : n == "nblk" ? &c.NBLK :
                                  n == "mask" ? &c.MASK : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen35_moe: %s: unexpected input %s\n", k.desc->role.c_str(), n.c_str());
            }
            cl.in.push_back(axera_io(*b, d.bytes));
        }
    }
    for (const auto & d : k.desc->out) {
        const std::string & n = d.name;
        if (n == "state_out" || n == "conv_out" || n == "kc" || n == "vc") {
            const int w = n == "state_out" ? 0 : n == "conv_out" ? 1 : n == "kc" ? 2 : 3;
            cl.d.push_back({ true, (int) cl.out.size(), il, w });
            cl.out.push_back(cache_io(il, w));
            continue;
        }
        const axera_cmm * b = n == "out" ? &c.BASE : n == "h2_out" ? &c.H2 : n == "logits_out" ? &c.LOG :
                              n == "topk_v" ? &c.TKV : n == "topk_i" ? &c.TKI : nullptr;
        if (!b || !*b) {
            GGML_ABORT("axera qwen35_moe: %s: unexpected output %s\n", k.desc->role.c_str(), n.c_str());
        }
        cl.out.push_back(axera_io(*b, d.bytes));
    }
    return cl;
}

void qwen35_moe::init(ggml_cgraph * g, const std::map<std::string, ggml_tensor *> & t, int L_, int H_) {
    GGML_UNUSED(g);
    L = L_;
    H = H_;
    const kernel & dk = man.get("decode.group");
    // the layers: delta-net (blk.<i>.ssm_a) or attention, the group's pattern repeated
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
            GGML_ABORT("axera qwen35_moe: %d layers do not repeat the decode group's pattern %s (%s)\n", L, pat.c_str(),
                       dk.desc->file.c_str());
        }
    }
    K  = (int) t.at("ffn_moe_topk-0")->ne[0];
    NE = (int) t.at("blk.0.ffn_gate_inp.weight")->ne[1];
    // the llama.cpp caches: recurrent state rows (cache_s / cache_r) and the KV cells (cache_k / cache_v, f16
    // [cells][kv heads x head dim], one block per stream, room for the kernels' context after the last)
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
                GGML_ABORT("axera qwen35_moe: %s not in CMM (all layers on the device: -ngl 99)\n", cn.c_str());
            }
            if (gdn[il]) {
                GGML_ASSERT(c->nb[1] >= sz);
                lc[il].stride[w] = c->nb[1];
            } else {
                GGML_ASSERT(c->type == GGML_TYPE_F16 && c->ne[3] == 1 && (N == 0 || c->ne[1] == N));
                N = c->ne[1];
                if (N > man.context || c->nb[1] * man.context != sz ||
                    ggml_backend_buft_get_alloc_size(c->buffer->buft, c) < (c->ne[2] - 1) * c->nb[2] + sz) {
                    GGML_ABORT("axera qwen35_moe: a KV cache of %lld cells (-c), the kernels hold %lld\n", (long long) N,
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
    // the attention mask [LMAX / BC][BC]
    for (const auto & d : dk.desc->in) {
        if (d.name == "mask") {
            GGML_ASSERT(d.shape.size() == 2);
            LMAX = d.shape[0] * d.shape[1];
            BC   = (int) d.shape[1];
        }
    }
    GGML_ASSERT(LMAX == man.context && N % BC == 0);

    // decode: x / out ping-pong A (every group starts at an even layer and ends at an odd one), per group its packs
    // and its layers' experts, state and KV
    A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    COS  = axera_cmm_alloc(dk.in_bytes("cos"));
    SIN  = axera_cmm_alloc(dk.in_bytes("sin"));
    POS  = axera_cmm_alloc(sizeof(int32_t));
    NBLK = axera_cmm_alloc(sizeof(int32_t));
    MASK = axera_cmm_alloc(dk.in_bytes("mask"));
    GGML_ASSERT(G % 2 == 0);
    dec.assign(L / G, {});
    for (int gi = 0; gi < L / G; ++gi) {
        call & c = dec[gi];
        c.k      = &dk;
        for (const auto & d : dk.desc->in) {
            const std::string & n = d.name;
            if (!d.param.empty()) {  // e_<plane><j>: layer j's expert planes
                c.in.push_back(prm.get(t, gi * G + atoi(n.c_str() + d.param.size()), d.kind, d.param));
            } else if (dk.desc->packs.count(n)) {
                c.in.push_back(axera_io(prm.pack(t, *dk.desc, n, gi * G), d.bytes));
            } else {
                const axera_cmm * b = n == "x" ? &A : n == "cos" ? &COS : n == "sin" ? &SIN : n == "pos" ? &POS :
                                      n == "nblk" ? &NBLK : n == "mask" ? &MASK : nullptr;
                if (!b) {
                    GGML_ABORT("axera qwen35_moe: decode.group: unexpected input %s\n", n.c_str());
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
            const int j = atoi(n.c_str() + 2), il = gi * G + j;
            const int w = n.rfind("st", 0) == 0 ? 0 : n.rfind("cv", 0) == 0 ? 1 : n.rfind("kc", 0) == 0 ? 2 : 3;
            c.d.push_back({ true, (int) c.out.size(), il, w });
            c.out.push_back(cache_io(il, w));
        }
    }
    head.bind(man.get("lm_head"), t.at("output.weight"));

    // prefill sizes, largest first
    sizes.clear();
    for (int P : { 256, 128, 64, 16, 4 }) {
        size_k s;
        s.P    = P;
        s.gdn  = &man.get("prefill" + std::to_string(P) + ".gdn");
        s.attn = &man.get("prefill" + std::to_string(P) + ".attn");
        bool tk_g = false, tk_a = false;
        for (const auto & d : s.gdn->desc->out) {
            tk_g = tk_g || d.name == "topk_i";
        }
        for (const auto & d : s.attn->desc->out) {
            tk_a = tk_a || d.name == "topk_i";
        }
        GGML_ASSERT(tk_g == tk_a);
        s.tk = tk_g;
        sizes.push_back(s);
    }
    // the routed experts' kernels
    auto slots_of = [&](const std::string & role) {
        slots_k s;
        s.k = &man.get(role);
        const io_desc * rw = nullptr;
        for (const auto & d : s.k->desc->in) {
            rw = d.name == "rw" ? &d : rw;
        }
        GGML_ASSERT(rw && rw->shape.size() == 2);
        s.S = (int) rw->shape[0];
        s.G = (int) rw->shape[1];
        GGML_ASSERT(s.k->out_bytes("out") == (size_t) s.S * s.G * H * sizeof(ggml_bf16_t));  // bf16 results
        return s;
    };
    main    = slots_of("moe.slots64_g64");
    rem     = { slots_of("moe.slots16_g64"), slots_of("moe.slots32_g64") };
    big     = slots_of("moe.slots32_g128");
    big_rem = { slots_of("moe.slots8_g128"), slots_of("moe.slots16_g128") };
    GGML_ASSERT(big.G == 2 * main.G);
    combs.clear();
    for (int T : { 512, 1024, 2048 }) {
        combine_k c;
        c.k = &man.get("moe.combine_t" + std::to_string(T));
        c.T = T;
        c.R = (int) (c.k->in_bytes("sout") / (H * sizeof(ggml_bf16_t)));
        combs.push_back(c);
    }

    // the group: 2048 tokens of chunks of the largest size, its chunks' rows in one buffer each
    const int    P0 = sizes[0].P, Q = combs.back().T / P0;
    const size_t xr = (size_t) P0 * H * sizeof(ggml_bf16_t), tr = (size_t) P0 * K * sizeof(float);
    GA    = axera_cmm_alloc(xr * Q);
    GB    = axera_cmm_alloc(xr * Q);
    GH2   = axera_cmm_alloc(xr * Q);
    GBASE = axera_cmm_alloc(xr * Q);
    GTKV  = axera_cmm_alloc(tr * Q);
    GTKI  = axera_cmm_alloc(tr * Q);
    GGML_ASSERT(sizes[0].tk && combs.back().k->in_bytes("base") == xr * Q);
    auto own_bufs = [&](chunk & c, const size_k & s) {
        c.P    = s.P;
        c.MASK = axera_cmm_alloc(s.attn->in_bytes("mask"));
        c.COS  = axera_cmm_alloc(s.attn->in_bytes("cos"));
        c.SIN  = axera_cmm_alloc(s.attn->in_bytes("sin"));
        c.POS  = axera_cmm_alloc(sizeof(int32_t));
        c.NBLK = axera_cmm_alloc(sizeof(int32_t));
        c.LOG  = axera_cmm_alloc((size_t) s.P * NE * sizeof(float));
        GGML_ASSERT(s.attn->in_bytes("mask") == (size_t) s.P * LMAX * sizeof(ggml_bf16_t));
    };
    grp.clear();
    grp.resize(Q);
    for (int c = 0; c < Q; ++c) {
        own_bufs(grp[c], sizes[0]);
        grp[c].A    = axera_cmm_view(GA.get(), c * xr, xr);
        grp[c].B    = axera_cmm_view(GB.get(), c * xr, xr);
        grp[c].H2   = axera_cmm_view(GH2.get(), c * xr, xr);
        grp[c].BASE = axera_cmm_view(GBASE.get(), c * xr, xr);
        grp[c].TKV  = axera_cmm_view(GTKV.get(), c * tr, tr);
        grp[c].TKI  = axera_cmm_view(GTKI.get(), c * tr, tr);
    }
    single.clear();
    for (size_t v = 1; v < sizes.size(); ++v) {
        chunk &      c  = single[sizes[v].P];
        const size_t cx = (size_t) sizes[v].P * H * sizeof(ggml_bf16_t);
        own_bufs(c, sizes[v]);
        c.A    = axera_cmm_alloc(cx);
        c.B    = axera_cmm_alloc(cx);
        c.H2   = axera_cmm_alloc(cx);
        c.BASE = axera_cmm_alloc(cx);
        GGML_ASSERT(!sizes[v].tk);
    }
    // slot rows: at most one partly filled slot per expert, for a whole group
    const int    NT    = P0 * Q, Gs = main.G;
    const size_t nsl   = (size_t) GGML_PAD(std::min(NT * K, NE + (NT * K + Gs - 1) / Gs), main.S);
    const size_t rows  = nsl * Gs;
    const int    R     = combs.back().R;
    const size_t ntok  = (size_t) R / Gs * (1 + Gs) + (size_t) 2 * big.S * (1 + big.G);
    GGML_ASSERT(rows <= (size_t) R);
    TOK  = axera_cmm_alloc(std::max(ntok, nsl * (1 + Gs)) * sizeof(int32_t));
    SRW  = axera_cmm_alloc(std::max((size_t) R, rows) * sizeof(float));
    SOUT = axera_cmm_alloc(std::max(combs.back().k->in_bytes("sout"), (uint32_t) (rows * H * sizeof(ggml_bf16_t))));
    LOC  = axera_cmm_alloc((size_t) NT * K * sizeof(int32_t));
    SX   = axera_cmm_alloc_nc(std::max(rows * H * sizeof(ggml_bf16_t), (size_t) main.k->in_bytes("h2")));
    GGML_ASSERT(combs.back().k->in_bytes("loc") == LOC->size);

    // every layer's calls of every chunk
    grp_calls.assign(Q, {});
    for (int c = 0; c < Q; ++c) {
        for (int il = 0; il < L; ++il) {
            grp_calls[c].push_back(bind_layer(t, gdn[il] ? *sizes[0].gdn : *sizes[0].attn, il, grp[c]));
        }
    }
    single_calls.clear();
    for (size_t v = 1; v < sizes.size(); ++v) {
        auto & cs = single_calls[sizes[v].P];
        for (int il = 0; il < L; ++il) {
            cs.push_back(bind_layer(t, gdn[il] ? *sizes[v].gdn : *sizes[v].attn, il, single[sizes[v].P]));
        }
    }
    // the converted parameters only the packs read: freed
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    for (auto & cs : grp_calls) {
        for (auto & c : cs) {
            live.push_back(&c.in);
        }
    }
    for (auto & [P, cs] : single_calls) {
        for (auto & c : cs) {
            live.push_back(&c.in);
        }
    }
    prm.release(live);
    GGML_LOG_INFO("axera qwen35_moe: %d layers (%s per decode call), %d experts top-%d, KV cache %lld cells (kernels %lld)\n",
                  L, pat.c_str(), NE, K, (long long) N, (long long) LMAX);
}

// the routed experts of layer il for the first ng chunks of the group: top-K from the layer kernels' outputs, picks
// grouped by expert into slots (the experts with more than G picks in slots of the larger kernel), the slots calls,
// then the combine kernel into the group's next x
void qwen35_moe::experts_group(int il, int ng) {
    const int    P = grp[0].P, NT = P * ng, G1 = main.G;
    const size_t es = sizeof(ggml_bf16_t);
    axera_timer  tr("moe prefill route (cpu)");
    std::vector<int>   ids((size_t) NT * K);
    std::vector<float> wts((size_t) NT * K);
    GTKV->invalidate(GTKV->vir, (size_t) NT * K * sizeof(float));
    GTKI->invalidate(GTKI->vir, (size_t) NT * K * sizeof(int32_t));
    const float *   tv  = (const float *) GTKV->vir;
    const int32_t * ti  = (const int32_t *) GTKI->vir;
    const int       nbk = std::max(1, NT / 64);
    axera_host_pool().run(nbk, [&](int64_t b) {  // the weights: softmax over the top-K (descending raw logits)
        for (int64_t r = b * NT / nbk; r < (b + 1) * NT / nbk; ++r) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                ids[r * K + i] = ti[r * K + i];
                wts[r * K + i] = std::exp(tv[r * K + i] - tv[r * K]);
                sum += wts[r * K + i];
            }
            for (int i = 0; i < K; ++i) {
                wts[r * K + i] /= sum;
            }
        }
    });
    // picks grouped by expert (counting sort, token order within an expert)
    std::vector<int> cnt(NE + 1, 0), order((size_t) NT * K);
    for (int pk = 0; pk < NT * K; ++pk) {
        cnt[ids[pk] + 1]++;
    }
    std::vector<int> first(NE + 1, 0);
    for (int e = 0; e < NE; ++e) {
        first[e + 1] = first[e] + cnt[e + 1];
    }
    {
        std::vector<int> fill(first.begin(), first.end() - 1);
        for (int pk = 0; pk < NT * K; ++pk) {
            order[fill[ids[pk]]++] = pk;
        }
    }
    // two regions of slots: experts with more than G picks in the big kernel's slots, the others in the main
    // kernel's; each region's slots as calls of its kernel, its last call through the smallest that holds the rest.
    // TOK / SRW / SOUT rows: region big first, then the other, each padded to its calls' slots
    struct region {
        const slots_k *              main;
        const std::vector<slots_k> * rem;
        int                          G = 0, ns = 0, cap = 0;
        size_t                       tok0 = 0, row0 = 0;
        std::vector<const slots_k *> calls;
    };
    region rg[2];
    rg[0].main = &big;
    rg[0].rem  = &big_rem;
    rg[0].G    = big.G;
    rg[1].main = &main;
    rg[1].rem  = &rem;
    rg[1].G    = G1;
    std::vector<int> isbig(NE, 0), slot0(NE, 0);
    auto plan = [&](region & r) {
        r.calls.clear();
        r.cap = 0;
        while (r.cap < r.ns) {
            const slots_k * kp = r.main;
            for (const auto & k : *r.rem) {
                if (r.ns - r.cap < r.main->S && k.S >= r.ns - r.cap) {
                    kp = &k;
                    break;
                }
            }
            r.calls.push_back(kp);
            r.cap += kp->S;
        }
    };
    const int R = combs.back().R;
    for (int pass = 0; pass < 2; ++pass) {  // pass 1: no big slots (pass 0's rows over the combine kernel's)
        rg[0].ns = rg[1].ns = 0;
        for (int e = 0; e < NE; ++e) {
            isbig[e] = pass == 0 && cnt[e + 1] > G1;
            auto & r = rg[isbig[e] ? 0 : 1];
            slot0[e] = r.ns;
            r.ns += (cnt[e + 1] + r.G - 1) / r.G;
        }
        plan(rg[0]);
        plan(rg[1]);
        rg[1].tok0 = (size_t) rg[0].cap * (1 + rg[0].G);
        rg[1].row0 = (size_t) rg[0].cap * rg[0].G;
        if (rg[1].row0 + (size_t) rg[1].cap * G1 <= (size_t) R &&
            rg[1].tok0 + (size_t) rg[1].cap * (1 + G1) <= TOK->size / sizeof(int32_t)) {
            break;
        }
        GGML_ASSERT(pass == 0 && "axera qwen35_moe: the group's expert slots exceed the combine kernel's rows");
    }
    const size_t ntok = rg[1].tok0 + (size_t) rg[1].cap * (1 + G1), nsw = rg[1].row0 + (size_t) rg[1].cap * G1;
    GGML_ASSERT(nsw * sizeof(float) <= SRW->size && nsw * H * es <= SOUT->size);
    const combine_k * cbp = &combs.back();  // the combine kernel of the fewest tokens that holds the group's
    for (auto it = combs.rbegin(); it != combs.rend(); ++it) {
        if (NT <= it->T) {
            cbp = &*it;
        }
    }
    const combine_k & cb   = *cbp;
    const size_t      nloc = (size_t) K * cb.T;
    auto *            tok  = (int32_t *) TOK->vir;
    auto *            sw   = (float *) SRW->vir;
    auto *            loc  = (int32_t *) LOC->vir;
    // padding slots / rows: expert 0, token 0, weight 0; tokens of chunks the group does not fill: row 0
    memset(tok, 0, ntok * sizeof(int32_t));
    memset(sw, 0, nsw * sizeof(float));
    if (NT < cb.T) {
        memset(loc, 0, nloc * sizeof(int32_t));
    }
    axera_host_pool().run(NE, [&](int64_t e) {
        const region & r = rg[isbig[e] ? 0 : 1];
        for (int j = 0; j < cnt[e + 1]; ++j) {
            const int    sl = slot0[e] + j / r.G, gg = j % r.G, pk = order[first[e] + j];
            const size_t rw = r.row0 + (size_t) sl * r.G + gg;
            int32_t *    tp = tok + r.tok0 + (size_t) sl * (1 + r.G);
            tp[0]           = (int32_t) e;
            tp[1 + gg]      = pk / K;
            sw[rw]          = wts[pk];
            loc[(size_t) (pk % K) * cb.T + pk / K] = (int32_t) rw;  // [K][T]
        }
    });
    TOK->flush(TOK->vir, ntok * sizeof(int32_t));
    SRW->flush(SRW->vir, nsw * sizeof(float));
    LOC->flush(LOC->vir, nloc * sizeof(int32_t));
    tr.stop();
    for (const region & r : rg) {
        int done = 0;
        for (const slots_k * kp : r.calls) {
            const kernel &                     k = *kp->k;
            const size_t                       r0 = r.row0 + (size_t) done * r.G;
            std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), out(k.info->nOutputSize);
            for (const auto & d : k.desc->in) {
                const int i = k.input(d.name);
                if (!d.param.empty()) {
                    in[i] = prm.get(names.t, il, d.kind, d.param);
                } else if (d.name == "h2") {
                    in[i] = axera_io(GH2);
                } else if (d.name == "tok") {
                    in[i] = axera_io(TOK.get(), tok + r.tok0 + (size_t) done * (1 + r.G),
                                     (uint32_t) (kp->S * (1 + r.G) * sizeof(int32_t)));
                } else if (d.name == "rw") {
                    in[i] = axera_io(SRW.get(), sw + r0, (uint32_t) (kp->S * r.G * sizeof(float)));
                } else {
                    GGML_ABORT("axera qwen35_moe: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
                }
            }
            out[k.output("out")] = axera_io(SOUT.get(), (char *) SOUT->vir + r0 * H * es, (uint32_t) (kp->S * r.G * H * es));
            axera_timer tm(&r == &rg[0] ? "moe prefill slots (big)" : "moe prefill slots");
            k.run(in, out, "moe prefill slots");
            done += kp->S;
        }
    }
    const axera_cmm & xo = il % 2 == 0 ? GB : GA;  // the group's next x (its first cb.T rows)
    const uint32_t    xb = (uint32_t) ((size_t) cb.T * H * sizeof(ggml_bf16_t));
    std::vector<AX_ENGINE_IO_BUFFER_T> in(cb.k->info->nInputSize), out(cb.k->info->nOutputSize);
    in[cb.k->input("base")]  = axera_io(GBASE, xb);
    in[cb.k->input("loc")]   = axera_io(LOC, (size_t) K * cb.T * sizeof(int32_t));
    in[cb.k->input("sout")]  = axera_io(SOUT, cb.k->in_bytes("sout"));
    out[cb.k->output("out")] = axera_io(xo, xb);
    axera_timer tc("moe prefill combine (npu)");
    cb.k->run(in, out, "moe prefill combine");
}

// the routed experts of layer il for one chunk of another size: top-K on the host from the router logits, slots of
// the main kernel, the results added to the base on the host
void qwen35_moe::experts_single(int il, chunk & c) {
    const int    P = c.P, NT = P, S = main.S, G1 = main.G;
    const size_t es = sizeof(ggml_bf16_t);
    axera_timer  tr("moe prefill route (cpu)");
    c.LOG->invalidate();
    c.H2->invalidate();
    c.BASE->invalidate();
    std::vector<int>   ids((size_t) NT * K);
    std::vector<float> wts((size_t) NT * K);
    axera_host_pool().run(NT, [&](int64_t r) {
        const float *    lg = (const float *) c.LOG->vir + (size_t) r * NE;
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
        for (size_t j = 0; j < by[e].size(); j += G1) {
            se.push_back(e);
            for (int g = 0; g < G1; ++g) {
                sp.push_back(j + g < by[e].size() ? by[e][j + g] : -1);
            }
        }
    }
    const int calls = (int) (se.size() + S - 1) / S;
    se.resize((size_t) calls * S, se[0]);  // padding slots: some expert on unused rows
    sp.resize((size_t) calls * S * G1, -1);
    GGML_ASSERT((size_t) calls * S * (1 + G1) * sizeof(int32_t) <= TOK->size &&
                (size_t) calls * S * G1 * H * es <= SOUT->size && (size_t) calls * S * G1 * sizeof(float) <= SRW->size);
    std::vector<int> loc((size_t) NT * K);  // pick -> slot row
    std::vector<int> used((size_t) calls * S, 0);
    auto *           sx = (ggml_bf16_t *) SX->vir;
    auto *           sw = (float *) SRW->vir;
    auto *           tk = (int32_t *) TOK->vir;
    memcpy(sx, c.H2->vir, (size_t) NT * H * sizeof(ggml_bf16_t));  // the slots kernel reads rows of h2 by token
    SX->flush(sx, (size_t) NT * H * sizeof(ggml_bf16_t));
    axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
        tk[sl * (1 + G1)] = se[sl];
        for (int g = 0; g < G1; ++g) {
            const int64_t rw = sl * G1 + g;
            const int     pk = sp[rw];
            sw[rw]           = pk >= 0 ? wts[pk] : 0.0f;
            tk[sl * (1 + G1) + 1 + g] = pk >= 0 ? pk / K : 0;
            if (pk >= 0) {
                loc[pk]  = (int) rw;
                used[sl] = g + 1;
            }
        }
    });
    SRW->flush(SRW->vir, (size_t) calls * S * G1 * sizeof(float));
    TOK->flush(TOK->vir, (size_t) calls * S * (1 + G1) * sizeof(int32_t));
    tr.stop();
    const kernel & k = *main.k;
    for (int cl = 0; cl < calls; ++cl) {
        const size_t                       r0 = (size_t) cl * S * G1;
        std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), out(k.info->nOutputSize);
        for (const auto & d : k.desc->in) {
            const int i = k.input(d.name);
            if (!d.param.empty()) {
                in[i] = prm.get(names.t, il, d.kind, d.param);
            } else if (d.name == "h2") {
                in[i] = axera_io(SX.get(), sx, k.in_bytes("h2"));
            } else if (d.name == "tok") {
                in[i] = axera_io(TOK.get(), tk + (size_t) cl * S * (1 + G1), (uint32_t) (S * (1 + G1) * sizeof(int32_t)));
            } else if (d.name == "rw") {
                in[i] = axera_io(SRW.get(), sw + r0, (uint32_t) (S * G1 * sizeof(float)));
            } else {
                GGML_ABORT("axera qwen35_moe: %s: unexpected input %s\n", k.desc->role.c_str(), d.name.c_str());
            }
        }
        out[k.output("out")] = axera_io(SOUT.get(), (char *) SOUT->vir + r0 * H * es, (uint32_t) (S * G1 * H * es));
        axera_timer tm("moe prefill slots");
        k.run(in, out, "moe prefill slots");
    }
    axera_timer tc("moe prefill combine (cpu)");
    axera_host_pool().run((int64_t) calls * S, [&](int64_t sl) {
        if (used[sl]) {
            SOUT->invalidate((char *) SOUT->vir + sl * G1 * H * es, (size_t) used[sl] * H * es);
        }
    });
    const auto *      so = (const char *) SOUT->vir;
    const axera_cmm & xo = il % 2 == 0 ? c.B : c.A;
    axera_host_pool().run(NT, [&](int64_t r) {
        auto f32 = [](uint16_t h) {  // bf16 -> f32 as a 16-bit shift (vectorized)
            const uint32_t u = (uint32_t) h << 16;
            float          f;
            memcpy(&f, &u, 4);
            return f;
        };
        const auto *       base = (const uint16_t *) c.BASE->vir + (size_t) r * H;
        std::vector<float> acc(H);
        for (int i = 0; i < H; ++i) {
            acc[i] = f32(base[i]);
        }
        for (int k2 = 0; k2 < K; ++k2) {
            const auto * ob = (const uint16_t *) (so + (size_t) loc[r * K + k2] * H * es);
            for (int i = 0; i < H; ++i) {
                acc[i] += f32(ob[i]);
            }
        }
        auto * xr = (uint16_t *) xo->vir + (size_t) r * H;
        for (int i = 0; i < H; ++i) {  // round to nearest even, as ggml_fp32_to_bf16 (finite values), vectorized
            uint32_t u;
            memcpy(&u, &acc[i], 4);
            xr[i] = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        }
    });
    xo->flush(xo->vir, (size_t) P * H * sizeof(ggml_bf16_t));
}

ggml_status qwen35_moe::compute(ggml_cgraph * g) {
    const auto & t      = names.of(g);
    ggml_tensor * norm0  = names.find("attn_norm-0");
    ggml_tensor * logits = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_moe: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    // layer 0's input: the source of the first RMS norm; nodes before it in this split (the embedding rows) on the CPU
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
    x0          = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
        ++L_;
    }
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(g, t, L_, H_);
        bound = w0->data;
    }
    auto x0_row = [&](int r) { return (const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]); };

    // the KV cells of the new tokens and llama.cpp's KQ mask
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm &&
                (kqm->ne[3] == 1 ? kqm->ne[1] >= T : kqm->ne[1] * kqm->ne[3] == T) && kqm->ne[0] <= N &&
                (kqm->type == GGML_TYPE_F16 || kqm->type == GGML_TYPE_F32));
    const kv_view kv(k_idxs, kqm, N, BC);
    // the recurrent-cache bookkeeping of this graph: rs_head (the row of the sequence, from the state copy into
    // cache_s of the first delta-net layer), rs_z (a row to clear), s_copy (row rs_head + i <- s_copy[i])
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
                GGML_LOG_ERROR("axera qwen35_moe: recurrent-state snapshots (speculative decoding) are not supported\n");
                return GGML_STATUS_FAILED;
            }
        } else if (node->op == GGML_OP_SCALE) {
            rs_z = (int64_t) (node->view_offs / s0->nb[1]);
        }
    }
    rs_head = best < 0 ? 0 : best / (int64_t) s0->nb[1];
    if (std::max<int64_t>(1, n_seqs) != 1) {
        GGML_LOG_ERROR("axera qwen35_moe: %lld sequences in a batch; one sequence only (-np 1)\n", (long long) n_seqs);
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
        for (int il = 0; il < L && !identity; ++il) {  // move states between rows (cell reuse)
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
    const int32_t *    pos = (const int32_t *) inp_pos->data;
    const ggml_tensor * rope_node = nullptr;  // the RoPE parameters: the graph's first ROPE node
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera qwen35_moe: no ROPE node in the graph");
    const rope         rp(rope_node, inp_pos, T);
    std::vector<float> h((size_t) T * H);
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    // consecutive cells of one stream from token it
    auto run_of = [&](int it, int n) {
        for (int j = 1; j < n; ++j) {
            if (it + j >= T || kv.stream_of(it + j) != kv.stream_of(it) || kv.cell_of(it + j) != kv.cell_of(it) + j) {
                return false;
            }
        }
        return kv.cell_of(it) + n <= N;
    };
    // a chunk's attention inputs: its mask, the cell it writes, the blocks to scan
    auto chunk_inputs = [&](chunk & c, int it) {
        const int nblk = kv.chunk_mask(pos, it, c.P, c.P, (ggml_bf16_t *) c.MASK->vir);
        *(int32_t *) c.POS->vir  = (int32_t) kv.cell_of(it);
        *(int32_t *) c.NBLK->vir = nblk;
        c.MASK->flush(c.MASK->vir, (size_t) nblk * c.P * BC * sizeof(ggml_bf16_t));
        c.POS->flush();
        c.NBLK->flush();
    };
    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        set_place(rs_head, kv.stream_of(it));
        if (p == 0) {  // a new sequence: llama.cpp owns the cells; its recurrent state starts at zero
            zero_row(row);
        }
        int v = -1;  // the largest chunk that fits the remaining consecutive cells
        for (size_t s = 0; s < sizes.size() && v < 0; ++s) {
            v = run_of(it, sizes[s].P) ? (int) s : -1;
        }
        if (v == 0) {  // a group: as many chunks of the largest size as fit
            const int P  = sizes[0].P;
            int       ng = 1;
            while (ng < (int) grp.size() && run_of(it, (ng + 1) * P)) {
                ++ng;
            }
            for (int c = 0; c < ng; ++c) {
                chunk &   ch = grp[c];
                const int tc = it + c * P;
                axera_host_pool().run(P, [&](int64_t r) {
                    ggml_fp32_to_bf16_row(x0_row(tc + (int) r), (ggml_bf16_t *) ch.A->vir + (size_t) r * H, H);
                });
                ch.A->flush(ch.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                rp.rows(tc, P, ch.COS.get(), ch.SIN.get());
                chunk_inputs(ch, tc);
            }
            for (int il = 0; il < L; ++il) {
                for (int c = 0; c < ng; ++c) {
                    run(grp_calls[c][il], gdn[il] ? "gdn prefill" : "attn prefill");
                }
                experts_group(il, ng);
            }
            for (int c = 0; c < ng; ++c) {
                const axera_cmm & fin = L % 2 == 0 ? grp[c].A : grp[c].B;
                fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir, h.data() + (size_t) (it + c * P) * H, (int64_t) P * H);
            }
            it += ng * P;
            continue;
        }
        if (v > 0) {
            const int P  = sizes[v].P;
            chunk &   ch = single.at(P);
            axera_host_pool().run(P, [&](int64_t r) {
                ggml_fp32_to_bf16_row(x0_row(it + (int) r), (ggml_bf16_t *) ch.A->vir + (size_t) r * H, H);
            });
            ch.A->flush(ch.A->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            rp.rows(it, P, ch.COS.get(), ch.SIN.get());
            chunk_inputs(ch, it);
            auto & cs = single_calls.at(P);
            for (int il = 0; il < L; ++il) {
                run(cs[il], gdn[il] ? "gdn prefill" : "attn prefill");
                experts_single(il, ch);
            }
            const axera_cmm & fin = L % 2 == 0 ? ch.A : ch.B;
            fin->invalidate(fin->vir, (size_t) P * H * sizeof(ggml_bf16_t));
            axera_host_pool().run(P, [&](int64_t r) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) fin->vir + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
            });
            it += P;
            continue;
        }
        // one token through the decode groups
        ggml_fp32_to_bf16_row(x0_row(it), (ggml_bf16_t *) A->vir, H);
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        rp.rows(it, 1, COS.get(), SIN.get());
        *(int32_t *) NBLK->vir = kv.mask_row(it, (ggml_bf16_t *) MASK->vir);
        *(int32_t *) POS->vir  = (int32_t) kv.cell_of(it);
        POS->flush();
        NBLK->flush();
        MASK->flush(MASK->vir, (size_t) *(int32_t *) NBLK->vir * BC * sizeof(ggml_bf16_t));
        {
            axera_timer tl("layers (decode)");
            for (auto & c : dec) {
                run(c, "moe layer group");
            }
        }
        A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, h.data() + (size_t) it * H, H);
        ++it;
    }
    head.run(g, names, h, T);
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen35_moe(const manifest & m) {
    return std::make_shared<qwen35_moe>(m);
}

}  // namespace axera
