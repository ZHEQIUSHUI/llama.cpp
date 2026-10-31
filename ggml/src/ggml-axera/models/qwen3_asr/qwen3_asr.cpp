// Axera NPU backend: the Qwen3-ASR audio encoder (mtmd's graph of the audio tower, projector qwen3a)
//
// The whole encoder in one call of the manifest's kernel audio.enc.n<n> (ax-dsl-kernels models/qwen3_asr, asr_kernel.py)
// for n chunks of 100 mel frames (the call's chunks padded to n = 1 / 2 / 4 / 8): the mel frames in, on chip the
// three 3 x 3 stride-2 convolutions with GELU, conv_out, the positions, all encoder layers (a runtime loop over weight
// banks [L, ...]), the post norm and the projector; out the projector rows (the first 13 per chunk valid).
// The weights: bf16 copies of the mmproj's tensors, the layers' stacked by layer (q / k / v joined), made by the first
// call of a model.

#include "../model.h"
#include "../../core/runtime.h"
#include "../../ops/ops.h"


namespace axera {
namespace {

// a tensor of any type as bf16 into dst, by rows on the host workers
void to_bf16(const ggml_tensor * p, ggml_bf16_t * dst) {
    const int64_t K = p->ne[0], rows = ggml_nelements(p) / K;
    const auto *  tr = ggml_get_type_traits(p->type);
    axera_host_pool().run(rows, [&](int64_t r) {
        std::vector<float> f(K);
        const char *       src = (const char *) p->data + r * ggml_row_size(p->type, K);
        if (p->type == GGML_TYPE_F32) {
            memcpy(f.data(), src, K * sizeof(float));
        } else {
            tr->to_float(src, f.data(), K);
        }
        ggml_fp32_to_bf16_row(f.data(), dst + r * K, K);
    });
}

std::string base_name(const char * n) {  // the scheduler's copies "<backend>#<name>#<copy>" by <name>
    const char * a = strchr(n, '#');
    if (!a) {
        return n;
    }
    const char * b = strchr(a + 1, '#');
    return b ? std::string(a + 1, b) : std::string(a + 1);
}

using io_map = std::map<std::string, AX_ENGINE_IO_BUFFER_T>;

// a kernel call with its IO by name (each at least the kernel's size)
void run(const kernel & k, const io_map & io, const char * what) {
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), out(k.info->nOutputSize);
    for (int o = 0; o < 2; ++o) {
        auto & bufs = o ? out : in;
        for (size_t i = 0; i < bufs.size(); ++i) {
            const AX_ENGINE_IOMETA_T & m  = o ? k.info->pOutputs[i] : k.info->pInputs[i];
            auto                       it = io.find(m.pName);
            if (it == io.end() || it->second.nSize < m.nSize) {
                GGML_ABORT("axera qwen3_asr: %s: IO %s of %u bytes, %u given\n", what, m.pName, m.nSize,
                           it == io.end() ? 0u : it->second.nSize);
            }
            bufs[i]       = it->second;
            bufs[i].nSize = m.nSize;
        }
    }
    axera_timer tm(what);
    k.run(in, out, what);
}

AX_ENGINE_IO_BUFFER_T cm_io(const axera_cmm & c, size_t bytes = 0) {
    return axera_io(c.get(), c->vir, (uint32_t) (bytes ? bytes : c->size));
}

class qwen3_asr final : public model {
  public:
    explicit qwen3_asr(const manifest & m) : man(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest &                 man;
    std::map<std::string, axera_cmm> w;    // the kernel's weight inputs by name, of `bound`'s model
    std::map<std::string, axera_cmm> buf;  // IO buffers by name
    const void *                     bound = nullptr;  // data of a.blk.0.attn_q.weight

    axera_cmm & buffer(const std::string & name, size_t bytes) {
        axera_cmm & b = buf[name];
        if (!b || b->size < bytes) {
            b = axera_cmm_alloc(bytes);
        }
        return b;
    }
    void weights(const std::map<std::string, ggml_tensor *> & t, const kernel & k);
};

// the kernel's weight inputs from the graph's tensors, bf16: the convolutions as [OC][K] rows (conv2d.1's 9 taps padded
// to 16; conv2d.2 / .3 tap-major [OC][kh][kw][IC]), the layers' tensors stacked [L, ...] (q / k / v rows joined)
void qwen3_asr::weights(const std::map<std::string, ggml_tensor *> & t, const kernel & k) {
    auto get = [&](const std::string & n) {
        auto it = t.find(n);
        if (it == t.end()) {
            GGML_ABORT("axera qwen3_asr: tensor %s not in the graph\n", n.c_str());
        }
        return it->second;
    };
    auto input = [&](const char * name) -> ggml_bf16_t * {
        axera_cmm & c = w[name];
        c             = axera_cmm_alloc(k.in_bytes(name));
        return (ggml_bf16_t *) c->vir;
    };
    auto conv = [&](const char * name, const char * src, int64_t K, bool taps) {
        const ggml_tensor * wt = get(src);
        const int64_t       OC = wt->ne[3], IC = wt->ne[2], KK = wt->ne[0] * wt->ne[1] * IC;
        GGML_ASSERT(k.in_bytes(name) == (size_t) OC * K * sizeof(ggml_bf16_t) && KK <= K);
        ggml_bf16_t * d = input(name);
        axera_host_pool().run(OC, [&](int64_t o) {
            std::vector<float> f(KK), g(K, 0.0f);
            const char *       s = (const char *) wt->data + o * wt->nb[3];
            if (wt->type == GGML_TYPE_F32) {
                memcpy(f.data(), s, KK * sizeof(float));
            } else {
                ggml_get_type_traits(wt->type)->to_float(s, f.data(), KK);
            }
            for (int64_t i = 0; i < KK; ++i) {  // [IC][3][3] (-> [3][3][IC])
                g[taps ? (i % 9) * IC + i / 9 : i] = f[i];
            }
            ggml_fp32_to_bf16_row(g.data(), d + o * K, K);
        });
    };
    auto copy = [&](ggml_bf16_t * d, const ggml_tensor * p) {
        to_bf16(p, d);
        return d + ggml_nelements(p);
    };
    auto tensor = [&](const char * name, const char * src) {
        const ggml_tensor * p = get(src);
        GGML_ASSERT(k.in_bytes(name) == (size_t) ggml_nelements(p) * sizeof(ggml_bf16_t));
        copy(input(name), p);
    };
    int L = 0;
    while (t.count("a.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    auto layers = [&](const char * name, const char * suffix) {
        ggml_bf16_t * d = input(name), *end = d + k.in_bytes(name) / sizeof(ggml_bf16_t);
        for (int il = 0; il < L; ++il) {
            const std::string b = "a.blk." + std::to_string(il) + ".";
            const std::string s = suffix;
            const size_t      q = s.find("attn_qkv");
            if (q == std::string::npos) {
                d = copy(d, get(b + s));
                continue;
            }
            for (const char * part : { "attn_q", "attn_k", "attn_v" }) {
                d = copy(d, get(b + part + s.substr(q + 8)));
            }
        }
        GGML_ASSERT(d == end);
    };
    using names = std::initializer_list<std::pair<const char *, const char *>>;
    axera_timer tm("audio weights (host)");
    conv("w1", "a.conv2d.1.weight", 16, false);
    conv("w2", "a.conv2d.2.weight", 9 * 480, true);
    conv("w3", "a.conv2d.3.weight", 9 * 480, true);
    for (auto [name, src] : names{ { "b1", "a.conv2d.1.bias" }, { "b2", "a.conv2d.2.bias" },
                              { "b3", "a.conv2d.3.bias" }, { "conv_out_w", "a.conv_out.weight" },
                              { "postw", "a.post_ln.weight" }, { "postb", "a.post_ln.bias" },
                              { "p1w", "mm.a.mlp.1.weight" }, { "p1b", "mm.a.mlp.1.bias" },
                              { "p2w", "mm.a.mlp.2.weight" }, { "p2b", "mm.a.mlp.2.bias" } }) {
        tensor(name, src);
    }
    for (auto [name, src] : names{ { "ln1w", "ln1.weight" }, { "ln1b", "ln1.bias" },
                              { "qkvw", "attn_qkv.weight" }, { "qkvb", "attn_qkv.bias" },
                              { "ow", "attn_out.weight" }, { "ob", "attn_out.bias" },
                              { "ln2w", "ln2.weight" }, { "ln2b", "ln2.bias" },
                              { "upw", "ffn_up.weight" }, { "upb", "ffn_up.bias" },
                              { "downw", "ffn_down.weight" }, { "downb", "ffn_down.bias" } }) {
        layers(name, src);
    }
    for (auto & [name, c] : w) {
        c->flush();
    }
}

ggml_status qwen3_asr::compute(ggml_cgraph * g) {
    std::map<std::string, ggml_tensor *> t;
    for (int i = 0; i < g->n_nodes; i++) {
        ggml_tensor * node = g->nodes[i];
        t.emplace(base_name(node->name), node);
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            t.emplace(base_name(node->src[s]->name), node->src[s]);
        }
    }
    GGML_ASSERT(t.count("inp_raw") && t.count("a.blk.0.attn_q.weight"));
    if (t.count("a.conv_out.bias")) {
        GGML_ABORT("axera qwen3_asr: a conv_out bias: the kernel has none\n");
    }
    const ggml_tensor * inp = t.at("inp_raw");
    const int64_t       NF = inp->ne[0], MEL = inp->ne[1], CH = 100, nch = NF / CH;
    GGML_ASSERT(inp->type == GGML_TYPE_F32 && ggml_is_contiguous(inp) && NF % CH == 0 && MEL == 128);
    int n = 0;  // the kernel's chunks: the fewest that hold the call's
    for (int c : { 1, 2, 4, 8 }) {
        n = n == 0 && c >= nch ? c : n;
    }
    if (n == 0) {
        GGML_ABORT("axera qwen3_asr: %lld chunks of audio in one call; the kernels take up to 8\n", (long long) nch);
    }
    const kernel & k = man.get("audio.enc.n" + std::to_string(n));
    if (bound != t.at("a.blk.0.attn_q.weight")->data) {
        w.clear();
        bound = t.at("a.blk.0.attn_q.weight")->data;
    }
    if (w.empty()) {
        weights(t, k);
    }
    // the mel frames time-major [n * 100][128] (the graph's [128][frames]), the padding chunks zero
    axera_cmm & X = buffer("mel", k.in_bytes("mel"));
    {
        axera_timer   tm("audio mel (host)");
        const float * mel = (const float *) inp->data;
        auto *        x   = (ggml_bf16_t *) X->vir;
        axera_host_pool().run(n * CH, [&](int64_t f) {
            float row[128] = {};
            for (int64_t h = 0; f < NF && h < MEL; ++h) {
                row[h] = mel[h * NF + f];
            }
            ggml_fp32_to_bf16_row(row, x + f * MEL, MEL);
        });
        X->flush(X->vir, k.in_bytes("mel"));
    }
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    GGML_ASSERT(out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && out->ne[1] == nch * 13 &&
                k.out_bytes("out") % (out->ne[0] * sizeof(float)) == 0);
    axera_cmm & O = buffer("out", k.out_bytes("out"));
    io_map      io{ { "mel", cm_io(X) }, { "out", cm_io(O) } };
    for (auto & [name, c] : w) {
        io[name] = cm_io(c);
    }
    run(k, io, "audio encoder (npu)");
    const size_t rb = ggml_nbytes(out);
    O->invalidate(O->vir, rb);
    memcpy(out->data, O->vir, rb);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, rb);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

bool qwen3_asr_graph(const ggml_cgraph * g) {
    bool conv_out = false, layers = false;
    for (int i = 0; i < g->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC && g->nodes[i]->src[s]; ++s) {
            const std::string n = base_name(g->nodes[i]->src[s]->name);
            conv_out            = conv_out || n == "a.conv_out.weight";
            layers              = layers || n == "a.blk.0.attn_q.weight";
        }
    }
    return conv_out && layers;
}

std::shared_ptr<model> make_qwen3_asr(const manifest & m) {
    return std::make_shared<qwen3_asr>(m);
}

}  // namespace axera
