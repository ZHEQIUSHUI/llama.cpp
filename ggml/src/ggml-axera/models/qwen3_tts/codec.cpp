// Axera NPU backend: the Qwen3-TTS codec decoder (code2wav: 16 codes per frame -> 1920 samples at 24 kHz) in one call
// of the manifest's role gen.codec per window of 72 frames (ax-dsl-kernels models/qwen3_tts/codec, decoder_kernel.py):
// the quantizer, the pre conv, the transformer, the upsamplers and the DAC blocks with their streaming state inside.
//
// mtmd decodes in batches within windows of 72 frames (a short first batch, a partial last one), with the position in
// state_in_tfm_pos. The kernel takes whole windows and the state at a window's start: a batch's frames are decoded as
// its window so far (the codes after them zero: the decoder is causal, they change none of the batch's samples), and
// the kernel's next state is kept only when a window is complete. The state is this file's (the graph's state tensors
// are not read; tfm_pos is written).
// The weights: three bf16 banks packed once per model from the GGUF (the codebooks and SnakeBeta parameters as the
// GGUF stores them, folded).

#include "codec.h"

#include "../../ops/ops.h"

#include <algorithm>

namespace axera {
namespace {

constexpr int WIN = 72, CODES = 16, HOP = 1920;  // frames per window, codes per frame, samples per frame

// all elements of a GGUF tensor as f32 (row order), converted by rows on the host workers
std::vector<float> f32_of(const ggml_tensor * w) {
    const int64_t      K = w->ne[0], rows = ggml_nelements(w) / K;
    std::vector<float> v((size_t) ggml_nelements(w));
    const auto *       tr = ggml_get_type_traits(w->type);
    axera_host_pool().run(rows, [&](int64_t r) {
        const char * src = (const char *) w->data + r * ggml_row_size(w->type, K);
        if (w->type == GGML_TYPE_F32) {
            memcpy(&v[(size_t) r * K], src, K * sizeof(float));
        } else {
            tr->to_float(src, &v[(size_t) r * K], K);
        }
    });
    return v;
}

}  // namespace

// the banks' tensors in order (ax-dsl-kernels models/qwen3_tts/codec/GGUF_LAYOUT.md); a bank's bf16 elements appended,
// each tensor's in parallel over its leading index
struct codec_packer {
    const graph_names &      names;
    std::vector<ggml_bf16_t> out;

    const ggml_tensor * T(const std::string & n) const { return names.at("a.gen.wav." + n); }
    ggml_bf16_t *       grow(size_t n) {
        out.resize(out.size() + n);
        return out.data() + out.size() - n;
    }
    void copy(const std::string & n) {
        const std::vector<float> g = f32_of(T(n));
        ggml_bf16_t *            d = grow(g.size());
        axera_host_pool().run((int64_t) (g.size() + 65535) / 65536, [&](int64_t c) {
            const size_t i0 = (size_t) c * 65536, m = std::min<size_t>(65536, g.size() - i0);
            ggml_fp32_to_bf16_row(g.data() + i0, d + i0, (int64_t) m);
        });
    }
    void ones(int64_t n) {
        ggml_bf16_t * d = grow(n);
        std::fill(d, d + n, ggml_fp32_to_bf16(1.0f));
    }
    // conv1d ne [K, IC, OC] -> [OC][K * IC]: bank[o, k * IC + i] = G[o, i, k]
    void conv(const std::string & n) {
        const ggml_tensor *      t = T(n);
        const int64_t            K = t->ne[0], IC = t->ne[1], OC = t->ne[2];
        const std::vector<float> g = f32_of(t);
        ggml_bf16_t *            d = grow(g.size());
        axera_host_pool().run(OC, [&](int64_t o) {
            for (int64_t k = 0; k < K; ++k) {
                for (int64_t i = 0; i < IC; ++i) {
                    d[(o * K + k) * IC + i] = ggml_fp32_to_bf16(g[k + K * (i + IC * o)]);
                }
            }
        });
    }
    // ConvTranspose1d ne [K, OC, IC], stride S, J = K / S -> [S][OC][J * IC]: bank[p, o, j * IC + i] = G[i, o, p + j S]
    void trans(const std::string & n, int64_t S) {
        const ggml_tensor *      t = T(n);
        const int64_t            K = t->ne[0], OC = t->ne[1], IC = t->ne[2], J = K / S;
        const std::vector<float> g = f32_of(t);
        GGML_ASSERT(K % S == 0);
        ggml_bf16_t * d = grow(g.size());
        axera_host_pool().run(S * OC, [&](int64_t po) {
            const int64_t p = po / OC, o = po % OC;
            for (int64_t j = 0; j < J; ++j) {
                for (int64_t i = 0; i < IC; ++i) {
                    d[po * J * IC + j * IC + i] = ggml_fp32_to_bf16(g[(p + j * S) + K * (o + OC * i)]);
                }
            }
        });
    }
    // depthwise conv ne [K, 1, C] -> [K][T][C]: every time row the tap's weights
    void dw(const std::string & n, int64_t TT) {
        const ggml_tensor *      t = T(n);
        const int64_t            K = t->ne[0], C = t->ne[2];
        const std::vector<float> g = f32_of(t);
        ggml_bf16_t *            d = grow((size_t) K * TT * C);
        axera_host_pool().run(K * TT, [&](int64_t kr) {
            for (int64_t c = 0; c < C; ++c) {
                d[kr * C + c] = ggml_fp32_to_bf16(g[kr / TT + K * c]);
            }
        });
    }
};

tts_codec::tts_codec(const manifest & m) : k(m.get("gen.codec")) {}

void tts_codec::pack(const graph_names & names) {
    axera_timer tm("codec weights (load)");
    {
        codec_packer p{ names, {} };
        p.copy("quant.first.codebook.weight");
        p.ones(T("quant.first.codebook.weight", names)->ne[1]);
        p.copy("quant.first.out_proj.weight");
        p.copy("quant.rest.codebook.weight");
        p.ones(T("quant.rest.codebook.weight", names)->ne[1] * T("quant.rest.codebook.weight", names)->ne[2]);
        p.copy("quant.rest.out_proj.weight");
        p.conv("pre_conv.weight");
        p.copy("pre_conv.bias");
        for (const char * n : { "tfm.in_proj.weight", "tfm.in_proj.bias", "tfm.out_proj.weight", "tfm.out_proj.bias",
                                "tfm.output_norm.weight" }) {
            p.copy(n);
        }
        int L = 0;
        while (names.find("a.gen.wav.tfm.blk." + std::to_string(L) + ".ln1.weight")) {
            ++L;
        }
        for (const char * s : { "ln1.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight",
                                "ls1.weight", "ln2.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                                "ls2.weight" }) {
            for (int l = 0; l < L; ++l) {
                p.copy("tfm.blk." + std::to_string(l) + "." + s);
            }
        }
        bank(QW, "quant_w", p.out);
    }
    {
        codec_packer p{ names, {} };
        for (int u = 0; u < 2; ++u) {
            const std::string b = "up.blk." + std::to_string(u) + ".";
            p.trans(b + "conv.weight", 2);
            p.copy(b + "conv.bias");
            p.dw(b + "dwconv.weight", (int64_t) 2 * WIN * (u + 1));
            for (const char * s : { "dwconv.bias", "norm.weight", "norm.bias", "pw1.weight", "pw1.bias", "pw2.weight",
                                    "pw2.bias", "gamma" }) {
                p.copy(b + s);
            }
        }
        bank(UW, "up_w", p.out);
    }
    {
        codec_packer p{ names, {} };
        p.conv("dac.entry.weight");
        p.copy("dac.entry.bias");
        for (int b = 0; b < 4; ++b) {
            const std::string     pre = "dac.blk." + std::to_string(b) + ".";
            const ggml_tensor *   tc  = T(pre + "conv.weight", names);
            p.copy(pre + "snake.alpha");
            p.copy(pre + "snake.beta");
            p.trans(pre + "conv.weight", tc->ne[0] / 2);  // kernel = 2 x stride
            p.copy(pre + "conv.bias");
            for (int r = 0; r < 3; ++r) {
                const std::string rp = pre + "res." + std::to_string(r) + ".";
                p.copy(rp + "act1.alpha");
                p.copy(rp + "act1.beta");
                p.conv(rp + "conv1.weight");
                p.copy(rp + "conv1.bias");
                p.copy(rp + "act2.alpha");
                p.copy(rp + "act2.beta");
                p.conv(rp + "conv2.weight");
                p.copy(rp + "conv2.bias");
            }
        }
        p.copy("dac.post_snake.alpha");
        p.copy("dac.post_snake.beta");
        p.conv("dac.post_conv.weight");
        p.copy("dac.post_conv.bias");
        bank(DW, "dac_w", p.out);
    }
    if (const char * d = getenv("GGML_AXERA_CODEC_DUMP")) {  // the banks as files, to compare with the packer's
        for (auto [c, n] : std::initializer_list<std::pair<axera_cmm *, const char *>>{ { &QW, "quant_w" }, { &UW, "up_w" }, { &DW, "dac_w" } }) {
            FILE * f = fopen((std::string(d) + "/" + n + ".bin").c_str(), "wb");
            if (f) {
                fwrite((*c)->vir, 1, k.in_bytes(n), f);
                fclose(f);
            }
        }
    }
}

const ggml_tensor * tts_codec::T(const std::string & n, const graph_names & names) {
    return names.at("a.gen.wav." + n);
}

void tts_codec::bank(axera_cmm & c, const char * name, const std::vector<ggml_bf16_t> & v) {
    if (v.size() * sizeof(ggml_bf16_t) != k.in_bytes(name)) {
        GGML_ABORT("axera qwen3_tts codec: %s: %zu bf16 packed, the kernel takes %u bytes\n", name, v.size(), k.in_bytes(name));
    }
    c = axera_cmm_alloc(v.size() * sizeof(ggml_bf16_t));
    memcpy(c->vir, v.data(), v.size() * sizeof(ggml_bf16_t));
    c->flush();
}

void tts_codec::reset() {
    for (auto & [n, s] : cur) {
        memset(s->vir, 0, s->size);
        s->flush();
    }
}

bool tts_codec::graph(const graph_names & names) {
    // (mtmd builds the code generation and code2wav sub-graphs into every graph and selects one: out_audio computed)
    const ggml_tensor * o = names.find("out_audio");
    return o && (o->flags & GGML_TENSOR_FLAG_COMPUTE) && names.find("inp_codes") && names.find("state_in_tfm_pos");
}

void tts_codec::run(const graph_names & names) {
    const ggml_tensor * codebook = names.at("a.gen.wav.quant.first.codebook.weight");
    if (bound != codebook->data) {
        pack(names);
        bound = codebook->data;
        // the state buffers (<x>_state / <x>_tail and their <x>_next), the codes, the position, the samples
        cur.clear();
        nxt.clear();
        for (int i = 0; i < (int) k.info->nInputSize; ++i) {
            const std::string n = k.info->pInputs[i].pName;
            if (n == "codes" || n == "pos" || n == "quant_w" || n == "up_w" || n == "dac_w" ||
                (n.size() > 5 && n.compare(n.size() - 5, 5, "_next") == 0)) {
                continue;
            }
            const std::string nn = n.size() > 6 && n.compare(n.size() - 6, 6, "_state") == 0 ? n.substr(0, n.size() - 6) + "_next" :
                                                                                                n + "_next";
            cur[n]  = axera_cmm_alloc(k.in_bytes(n));
            nxt[nn] = axera_cmm_alloc(k.in_bytes(nn));
            pair[n] = nn;
        }
        CODES_B = axera_cmm_alloc(k.in_bytes("codes"));
        POS     = axera_cmm_alloc(k.in_bytes("pos"));
        WAV     = axera_cmm_alloc(k.out_bytes("wav"));
        GGML_ASSERT(k.in_bytes("codes") == WIN * CODES * sizeof(int32_t) && k.out_bytes("wav") == WIN * HOP * sizeof(float));
    }
    const ggml_tensor * codes = names.at("inp_codes");
    ggml_tensor *       out   = names.at("out_audio");
    const int64_t       P = (int64_t) *(const float *) names.at("state_in_tfm_pos")->data;  // frames decoded before
    const int64_t       n = codes->ne[0], a = P % WIN, b = a + n;
    GGML_ASSERT(codes->type == GGML_TYPE_I32 && codes->ne[1] == CODES && b <= WIN && out->type == GGML_TYPE_F32 &&
                ggml_nelements(out) == n * HOP);
    if (P == 0) {  // a new utterance
        reset();
    }
    auto * cb = (int32_t *) CODES_B->vir;
    if (a == 0) {
        memset(cb, 0, WIN * CODES * sizeof(int32_t));
    }
    for (int64_t t = 0; t < n; ++t) {
        for (int g = 0; g < CODES; ++g) {
            cb[(a + t) * CODES + g] = ((const int32_t *) codes->data)[g * n + t];
        }
    }
    CODES_B->flush();
    *(int32_t *) POS->vir = (int32_t) (P - a);
    POS->flush();
    std::vector<AX_ENGINE_IO_BUFFER_T> in(k.info->nInputSize), outs(k.info->nOutputSize);
    for (int i = 0; i < (int) k.info->nInputSize; ++i) {
        const std::string n_ = k.info->pInputs[i].pName;
        const axera_cmm * c  = n_ == "codes" ? &CODES_B : n_ == "pos" ? &POS : n_ == "quant_w" ? &QW : n_ == "up_w" ? &UW :
                               n_ == "dac_w" ? &DW : cur.count(n_) ? &cur.at(n_) : nxt.count(n_) ? &nxt.at(n_) : nullptr;
        if (!c) {
            GGML_ABORT("axera qwen3_tts codec: unexpected input %s\n", n_.c_str());
        }
        in[i] = axera_io(*c, k.info->pInputs[i].nSize);
    }
    outs[k.output("wav")] = axera_io(WAV, k.out_bytes("wav"));
    {
        axera_timer tm("codec window (npu)");
        k.run(in, outs, "codec");
    }
    WAV->invalidate(WAV->vir, (size_t) b * HOP * sizeof(float));
    memcpy(out->data, (const float *) WAV->vir + a * HOP, (size_t) n * HOP * sizeof(float));
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, ggml_nbytes(out));
    }
    if (b == WIN) {  // a complete window: its end state is the next window's start
        for (auto & [s, nn] : pair) {
            std::swap(cur.at(s), nxt.at(nn));
        }
    }
    if (ggml_tensor * po = names.find("state_out_tfm_pos")) {
        *(float *) po->data = (float) (P + n);
        if (ggml_backend_buffer_is_axera(po->buffer)) {
            ((ggml_backend_axera_buffer_context *) po->buffer->context)->flush(po->data, ggml_nbytes(po));
        }
    }
}

}  // namespace axera
