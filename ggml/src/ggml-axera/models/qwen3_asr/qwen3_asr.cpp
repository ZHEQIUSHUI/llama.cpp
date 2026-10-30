// Axera NPU backend: the Qwen3-ASR audio encoder (mtmd's graph of the audio tower, projector qwen3a)
//
// One path, the manifest's kernels (roles audio.*), for a call of n chunks of 100 mel frames (n padded to 1 / 2 / 4 / 8):
//   front-end  three 3 x 3 stride-2 convolutions with bias and GELU: audio.fe1.n<n> (its im2col rows from the mel
//              frames by the host), audio.fe2.n<n> (its im2col on chip from fe1's output), audio.fe3.n<n> (its im2col
//              by the host); conv_out audio.conv_out.n<n> (bf16 GEMM), its bias and the per-chunk positions by the host
//   layers     audio.layer<NF> per layer (NF = 64 / 128 rows, the smallest that holds the 13 n rows; the padding rows
//              hidden by the key mask)
//   projector  the post norm and the two projector matrices, audio.proj<NF>: the graph's output rows
// The weights: bf16 copies of the mmproj's tensors (q / k / v joined), the encoder layers' made while the model loads.

#include "../model.h"
#include "../../core/runtime.h"
#include "../../ops/ops.h"

#include <mutex>
#include <set>

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

// the bf16 copies of the encoder's matrices made while the model loads (qwen3_asr_on_set), by their data, with a
// fingerprint of the source bytes they were made from
struct pre_copy {
    axera_cmm c;
    uint64_t  fp = 0;
};
std::mutex                              g_pre_mu;
std::map<const void *, pre_copy>        g_pre;
std::map<const void *, uint64_t>        g_seen;  // the fingerprint each source was last converted from
std::set<const void *>                  g_done;  // sources whose copy is in use: not looked at again

// FNV-1a over 64 slices of 256 bytes spread over a tensor's data: the scheduler can ask about a weight before the loader
// has read it (its buffer then holds whatever an earlier process left in that CMM: another model's data), so a copy is
// made again when the source no longer matches what it was made from
uint64_t fingerprint(const ggml_tensor * t) {
    const size_t    nb = ggml_nbytes(t), w = std::min<size_t>(256, nb);
    const uint8_t * p  = (const uint8_t *) t->data;
    uint64_t        h  = 1469598103934665603ull;
    for (int i = 0; i < (nb > w ? 64 : 1); ++i) {
        const size_t o = nb > w ? (nb - w) * i / 63 : 0;
        for (size_t j = 0; j < w; ++j) {
            h = (h ^ p[o + j]) * 1099511628211ull;
        }
    }
    return h;
}

// the load-time bf16 copy of p, taken out of the registry (nullptr if there is none or its source has changed since)
axera_cmm take_pre(const ggml_tensor * p) {
    std::lock_guard<std::mutex> lock(g_pre_mu);
    g_done.insert(p->data);
    auto it = g_pre.find(p->data);
    if (it == g_pre.end() || it->second.c->size != (size_t) ggml_nelements(p) * sizeof(ggml_bf16_t)) {
        return axera_cmm();
    }
    axera_cmm  c  = std::move(it->second.c);
    const bool ok = it->second.fp == fingerprint(p);
    g_pre.erase(it);
    if (!ok) {
        GGML_LOG_WARN("axera qwen3_asr: the load-time copy of %s was made before its data was loaded, converting it again\n",
                      p->name);
        return axera_cmm();
    }
    return c;
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
    std::map<std::string, axera_cmm> w;    // bf16 weights by name (q / k / v joined as attn_qkv), of `bound`'s model
    std::map<std::string, axera_cmm> buf;  // IO buffers by name
    const void *                     bound = nullptr;  // data of a.blk.0.attn_q.weight

    axera_cmm & buffer(const std::string & name, size_t bytes) {
        axera_cmm & b = buf[name];
        if (!b || b->size < bytes) {
            b = axera_cmm_alloc(bytes);
        }
        return b;
    }
    void front_end(const std::map<std::string, ggml_tensor *> & t, ggml_tensor * x0);
};

// the conv front-end of the call's chunks into the layers' input rows x0
void qwen3_asr::front_end(const std::map<std::string, ggml_tensor *> & t, ggml_tensor * x0) {
    auto get = [&](const char * n) {
        auto it = t.find(n);
        if (it == t.end()) {
            GGML_ABORT("axera qwen3_asr: tensor %s not in the graph\n", n);
        }
        return it->second;
    };
    const ggml_tensor * inp = get("inp_raw");
    const ggml_tensor * w1 = get("a.conv2d.1.weight"), *w2 = get("a.conv2d.2.weight"), *w3 = get("a.conv2d.3.weight");
    const ggml_tensor * b1 = get("a.conv2d.1.bias"), *b2 = get("a.conv2d.2.bias"), *b3 = get("a.conv2d.3.bias");
    const ggml_tensor * wo = get("a.conv_out.weight"), *pe = get("a.position_embd.weight");
    const ggml_tensor * bo = t.count("a.conv_out.bias") ? t.at("a.conv_out.bias") : nullptr;
    const int64_t       NF = inp->ne[0], MEL = inp->ne[1], CH = 100, H = x0->ne[0];
    const int64_t       C = w1->ne[3];  // 480
    GGML_ASSERT(inp->type == GGML_TYPE_F32 && ggml_is_contiguous(inp) && NF % CH == 0 && MEL == 128 && w1->ne[2] == 1 &&
                w2->ne[2] == C && w3->ne[2] == C && wo->ne[0] == 16 * C && wo->ne[1] == H && x0->ne[1] == NF / CH * 13 &&
                b1->type == GGML_TYPE_F32 && b2->type == GGML_TYPE_F32 && b3->type == GGML_TYPE_F32 &&
                (!bo || bo->type == GGML_TYPE_F32) && pe->type == GGML_TYPE_F32);
    const int64_t nch = NF / CH;
    int64_t       n   = 0;
    for (int64_t c : { 1, 2, 4, 8 }) {
        n = n == 0 && c >= nch ? c : n;
    }
    if (n == 0) {
        GGML_ABORT("axera qwen3_asr: %lld chunks of audio in one call; the kernels take up to 8\n", (long long) nch);
    }
    const std::string sfx = ".n" + std::to_string(n);
    const kernel &    k1 = man.get("audio.fe1" + sfx), &k2 = man.get("audio.fe2" + sfx), &k3 = man.get("audio.fe3" + sfx);
    const kernel &    ko = man.get("audio.conv_out" + sfx);
    const int64_t     R1 = 3200 * n, R2 = 800 * n, R3 = 208 * n, M = 13 * n, K1 = 16, K2 = 9 * C;
    // the kernels' weights as bf16 rows [OC][IC * 9] (conv2d.1's 9 taps padded to 16; conv2d.2's taps
    // [OC][kh][kw][IC]), once per model
    auto conv_w = [&](const ggml_tensor * wt, int64_t K, bool taps) -> axera_cmm & {
        axera_cmm & c = w[std::string(taps ? "fe:taps:" : "fe:") + base_name(wt->name)];
        if (!c) {
            const int64_t OC = wt->ne[3], KK = wt->ne[0] * wt->ne[1] * wt->ne[2], IC = wt->ne[2];
            c       = axera_cmm_alloc((size_t) OC * K * sizeof(ggml_bf16_t));
            auto * d = (ggml_bf16_t *) c->vir;
            axera_host_pool().run(OC, [&](int64_t o) {
                std::vector<float> f(KK), g(K, 0.0f);
                const char *       src = (const char *) wt->data + o * wt->nb[3];
                if (wt->type == GGML_TYPE_F32) {
                    memcpy(f.data(), src, KK * sizeof(float));
                } else {
                    ggml_get_type_traits(wt->type)->to_float(src, f.data(), KK);
                }
                for (int64_t i = 0; i < KK; ++i) {  // [IC][3][3] (-> [3][3][IC])
                    g[taps ? (i % 9) * IC + i / 9 : i] = f[i];
                }
                ggml_fp32_to_bf16_row(g.data(), d + o * K, K);
            });
            c->flush();
        }
        return c;
    };
    axera_cmm & W1 = conv_w(w1, K1, false);
    axera_cmm & W2 = conv_w(w2, K2, true);
    axera_cmm & W3 = conv_w(w3, K2, false);
    axera_cmm & WO = w["fe:a.conv_out.weight"];
    if (!WO) {
        WO = axera_cmm_alloc((size_t) H * 16 * C * sizeof(ggml_bf16_t));
        to_bf16(wo, (ggml_bf16_t *) WO->vir);
        WO->flush();
    }
    axera_cmm & X1 = buffer("x1", (size_t) R1 * K1 * 2), &O1 = buffer("o1", (size_t) C * R1 * 2);
    axera_cmm & O2 = buffer("o2", (size_t) C * R2 * 2);
    axera_cmm & X3 = buffer("x3", (size_t) R3 * K2 * 2), &O3 = buffer("o3", (size_t) C * R3 * 2);
    axera_cmm & XO = buffer("xo", (size_t) M * 16 * C * 2), &OO = buffer("oo", (size_t) M * H * 4);
    auto bias_io = [&](const ggml_tensor * b) -> AX_ENGINE_IO_BUFFER_T {
        axera_cmm & c = w["fe:" + base_name(b->name)];
        if (!c) {
            c = axera_cmm_alloc(ggml_nbytes(b));
            memcpy(c->vir, b->data, ggml_nbytes(b));
            c->flush();
        }
        return cm_io(c);
    };
    // conv2d.1: the mel rows of each chunk, one input channel; its columns parity-split for fe2's on-chip im2col
    {
        axera_timer   tm("audio fe im2col 1 (host)");
        const float * mel = (const float *) inp->data;
        auto *        d   = (ggml_bf16_t *) X1->vir;
        axera_host_pool().run(n * 64, [&](int64_t job) {
            const int64_t c = job / 64, oh = job % 64;
            for (int64_t ow = 0; ow < 50; ++ow) {
                float v[16] = {};
                for (int64_t kh = 0; kh < 3; ++kh) {
                    for (int64_t kw = 0; kw < 3; ++kw) {
                        const int64_t h = 2 * oh + kh - 1, f = 2 * ow + kw - 1;
                        v[kh * 3 + kw] = c < nch && h >= 0 && h < MEL && f >= 0 && f < CH ? mel[h * NF + c * CH + f] : 0.0f;
                    }
                }
                ggml_fp32_to_bf16_row(v, d + ((c * 64 + oh) * 50 + (ow % 2) * 25 + ow / 2) * K1, K1);
            }
        });
        X1->flush(X1->vir, (size_t) R1 * K1 * 2);
    }
    run(k1, { { "x", cm_io(X1) }, { "w", cm_io(W1) }, { "b", bias_io(b1) }, { "out", cm_io(O1) } }, "audio fe conv 1 (npu)");
    run(k2, { { "x", cm_io(O1) }, { "w", cm_io(W2) }, { "b", bias_io(b2) }, { "out", cm_io(O2) } }, "audio fe conv 2 (npu)");
    O2->invalidate(O2->vir, (size_t) C * R2 * 2);
    {  // conv2d.3's im2col: 3 x 3 stride-2 pad-1 of the bf16 map [C][n][32][25] (rows (n, oh, ow), columns (ic, kh, kw))
        axera_timer         tm("audio fe im2col 3 (host)");
        const int64_t       IH = 32, IW = 25, OH = 16, OW = 13;
        const ggml_bf16_t   zero = ggml_fp32_to_bf16(0.0f);
        const ggml_bf16_t * src  = (const ggml_bf16_t *) O2->vir;
        auto *              dst  = (ggml_bf16_t *) X3->vir;
        axera_host_pool().run(n * OH, [&](int64_t job) {
            const int64_t c = job / OH, oh = job % OH;
            ggml_bf16_t * row0 = dst + (c * OH + oh) * OW * K2;
            if (c >= nch) {  // a padding chunk
                std::fill(row0, row0 + OW * K2, zero);
                return;
            }
            for (int64_t ow = 0; ow < OW; ++ow) {
                ggml_bf16_t * d    = row0 + ow * K2;
                const int64_t w0   = 2 * ow - 1;
                const bool    edge = w0 < 0 || w0 + 2 >= IW;
                for (int64_t ic = 0; ic < C; ++ic) {
                    const ggml_bf16_t * s = src + (ic * n + c) * IH * IW;
                    for (int64_t kh = 0; kh < 3; ++kh, d += 3) {
                        const int64_t h = 2 * oh + kh - 1;
                        if (h < 0 || h >= IH) {
                            d[0] = d[1] = d[2] = zero;
                            continue;
                        }
                        const ggml_bf16_t * r = s + h * IW;
                        if (!edge) {
                            d[0] = r[w0];
                            d[1] = r[w0 + 1];
                            d[2] = r[w0 + 2];
                        } else {
                            for (int64_t kw = 0; kw < 3; ++kw) {
                                const int64_t x = w0 + kw;
                                d[kw]           = x >= 0 && x < IW ? r[x] : zero;
                            }
                        }
                    }
                }
                std::fill(d, row0 + (ow + 1) * K2, zero);
            }
        });
        X3->flush(X3->vir, (size_t) R3 * K2 * 2);
    }
    run(k3, { { "x", cm_io(X3) }, { "w", cm_io(W3) }, { "b", bias_io(b3) }, { "out", cm_io(O3) } }, "audio fe conv 3 (npu)");
    O3->invalidate(O3->vir, (size_t) C * R3 * 2);
    // conv_out's rows (n, ow), columns (oc, oh): the [OW, OH, OC, n] map permuted to [OH, OC, OW, n] as the graph does
    {
        axera_timer  tm("audio fe conv_out rows (host)");
        const auto * s = (const ggml_bf16_t *) O3->vir;
        auto *       d = (ggml_bf16_t *) XO->vir;
        axera_host_pool().run(M, [&](int64_t r) {
            const int64_t c = r / 13, ow = r % 13;
            for (int64_t oc = 0; oc < C; ++oc) {
                for (int64_t oh = 0; oh < 16; ++oh) {
                    d[r * 16 * C + oc * 16 + oh] = s[oc * R3 + (c * 16 + oh) * 13 + ow];
                }
            }
        });
        XO->flush(XO->vir, (size_t) M * 16 * C * 2);
    }
    run(ko, { { "x", cm_io(XO) }, { "w", cm_io(WO) }, { "out", cm_io(OO) } }, "audio fe conv_out (npu)");
    OO->invalidate(OO->vir, (size_t) M * H * 4);
    {
        axera_timer   tm("audio fe positions (host)");
        const auto *  y  = (const float *) OO->vir;
        auto *        x  = (float *) x0->data;
        const float * bb = bo ? (const float *) bo->data : nullptr;
        for (int64_t r = 0; r < nch * 13; ++r) {
            const float * p = (const float *) ((const char *) pe->data + (r % 13) * pe->nb[1]);
            for (int64_t i = 0; i < H; ++i) {
                x[r * H + i] = (y[r * H + i] + (bb ? bb[i] : 0.0f)) + p[i];
            }
        }
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
    auto get = [&](const std::string & name) -> ggml_tensor * {
        auto it = t.find(name);
        if (it == t.end()) {
            GGML_ABORT("axera qwen3_asr: tensor %s not in the graph\n", name.c_str());
        }
        return it->second;
    };
    int L = 0;
    while (t.count("a.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    // the layers' input: the source of the NORM under layer 0's ln1
    ggml_tensor * x0 = nullptr;
    for (int i = 0; i < g->n_nodes && !x0; i++) {
        ggml_tensor * mul = g->nodes[i];
        if (mul->op == GGML_OP_MUL && mul->src[1] && base_name(mul->src[1]->name) == "a.blk.0.ln1.weight") {
            GGML_ASSERT(mul->src[0]->op == GGML_OP_NORM);
            x0 = mul->src[0]->src[0];
        }
    }
    GGML_ASSERT(x0 && x0->type == GGML_TYPE_F32 && ggml_is_contiguous(x0));
    const int H = (int) x0->ne[0], n = (int) x0->ne[1];
    int       NF = 0;  // the layer kernel's rows: the fewest that hold the n rows
    for (int c : { 64, 128 }) {
        NF = NF == 0 && c >= n ? c : NF;
    }
    if (NF == 0) {
        GGML_ABORT("axera qwen3_asr: %d rows in one encoder call; the kernels take up to 128\n", n);
    }
    const kernel & kl = man.get("audio.layer" + std::to_string(NF));
    const kernel & kp = man.get("audio.proj" + std::to_string(NF));
    GGML_ASSERT(kl.in_bytes("x") == (size_t) NF * H * sizeof(ggml_bf16_t));
    if (bound != get("a.blk.0.attn_q.weight")->data) {
        w.clear();
        bound = get("a.blk.0.attn_q.weight")->data;
    }
    // a weight in bf16, converted once per model (q / k / v rows joined for attn_qkv)
    auto weight = [&](const std::string & key) -> AX_ENGINE_IO_BUFFER_T {
        axera_cmm & c = w[key];
        if (!c) {
            const size_t q = key.find(".attn_qkv.");
            std::vector<const ggml_tensor *> parts;
            if (q != std::string::npos) {
                for (const char * p : { ".attn_q.", ".attn_k.", ".attn_v." }) {
                    parts.push_back(get(key.substr(0, q) + p + key.substr(q + 10)));
                }
            } else {
                parts.push_back(get(key));
            }
            int64_t ne = 0;
            for (const ggml_tensor * p : parts) {
                ne += ggml_nelements(p);
            }
            if (parts.size() == 1) {
                c = take_pre(parts[0]);
            }
            if (!c) {
                c          = axera_cmm_alloc(ne * sizeof(ggml_bf16_t));
                auto * dst = (ggml_bf16_t *) c->vir;
                for (const ggml_tensor * p : parts) {
                    axera_cmm pre = take_pre(p);
                    if (pre) {
                        memcpy(dst, pre->vir, pre->size);
                    } else {
                        to_bf16(p, dst);
                    }
                    dst += ggml_nelements(p);
                }
            }
            c->flush();
        }
        return cm_io(c);
    };

    front_end(t, x0);

    const size_t xb    = (size_t) NF * H * sizeof(ggml_bf16_t);
    axera_cmm *  LX[2] = { &buffer("lx0", xb), &buffer("lx1", xb) };
    axera_cmm &  KM    = buffer("lmask", (size_t) NF * sizeof(ggml_bf16_t));
    {
        axera_timer tm("audio prepare (cpu)");
        auto *      x = (ggml_bf16_t *) (*LX[0])->vir;
        ggml_fp32_to_bf16_row((const float *) x0->data, x, (int64_t) n * H);
        std::fill(x + (size_t) n * H, x + (size_t) NF * H, ggml_fp32_to_bf16(0.0f));
        (*LX[0])->flush((*LX[0])->vir, xb);
        auto * km = (ggml_bf16_t *) KM->vir;
        for (int c = 0; c < NF; ++c) {
            km[c] = ggml_fp32_to_bf16(c < n ? 0.0f : -3e38f);
        }
        KM->flush();
    }
    int cur = 0;
    for (int il = 0; il < L; ++il) {
        const std::string b = "a.blk." + std::to_string(il) + ".";
        run(kl,
            { { "x", cm_io(*LX[cur]) }, { "ln1_w", weight(b + "ln1.weight") }, { "ln1_b", weight(b + "ln1.bias") },
              { "w_qkv", weight(b + "attn_qkv.weight") }, { "b_qkv", weight(b + "attn_qkv.bias") },
              { "w_o", weight(b + "attn_out.weight") }, { "b_o", weight(b + "attn_out.bias") },
              { "ln2_w", weight(b + "ln2.weight") }, { "ln2_b", weight(b + "ln2.bias") },
              { "w_up", weight(b + "ffn_up.weight") }, { "b_up", weight(b + "ffn_up.bias") },
              { "w_down", weight(b + "ffn_down.weight") }, { "b_down", weight(b + "ffn_down.bias") },
              { "kmask", cm_io(KM) }, { "out", cm_io(*LX[1 - cur]) } },
            "audio layer (npu)");
        cur = 1 - cur;
    }
    // the post norm and the projector from the last layer's rows, f32 rows [n][O]: the graph's output
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    GGML_ASSERT(out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && out->ne[1] == n &&
                kp.out_bytes("out") == (size_t) NF * out->ne[0] * sizeof(float));
    axera_cmm & PO = buffer("pout", kp.out_bytes("out"));
    run(kp,
        { { "x", cm_io(*LX[cur]) }, { "ln_w", weight("a.post_ln.weight") }, { "ln_b", weight("a.post_ln.bias") },
          { "w1", weight("mm.a.mlp.1.weight") }, { "b1", weight("mm.a.mlp.1.bias") },
          { "w2", weight("mm.a.mlp.2.weight") }, { "b2", weight("mm.a.mlp.2.bias") }, { "out", cm_io(PO) } },
        "audio projector (npu)");
    const size_t rb = (size_t) n * out->ne[0] * sizeof(float);
    PO->invalidate(PO->vir, rb);
    memcpy(out->data, PO->vir, rb);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, ggml_nbytes(out));
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

void qwen3_asr_on_set(const ggml_tensor * t) {
    static const char * names[] = { ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_out.weight",
                                    ".ffn_up.weight", ".ffn_down.weight" };
    const bool proj = strncmp(t->name, "mm.a.mlp.", 9) == 0;
    if (t->data == nullptr || t->buffer == nullptr || (strncmp(t->name, "a.blk.", 6) != 0 && !proj) ||
        ggml_n_dims(t) != 2 || !ggml_is_contiguous(t) ||
        (t->type != GGML_TYPE_F32 && !ggml_get_type_traits(t->type)->to_float)) {
        return;
    }
    const size_t n = strlen(t->name);
    bool         m = proj && n > 7 && strcmp(t->name + n - 7, ".weight") == 0;
    for (const char * s : names) {
        const size_t k = strlen(s);
        m              = m || (n > k && strcmp(t->name + n - k, s) == 0);
    }
    if (!m) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_pre_mu);
        if (g_done.count(t->data)) {
            return;
        }
    }
    const uint64_t fp = fingerprint(t);
    {
        std::lock_guard<std::mutex> lock(g_pre_mu);  // (converted again only if the source changed since)
        auto                        it = g_seen.find(t->data);
        if (it != g_seen.end() && it->second == fp) {
            return;
        }
        g_seen[t->data] = fp;
    }
    axera_timer tm("audio weights at load (host)");
    axera_cmm   c = axera_cmm_alloc(ggml_nelements(t) * sizeof(ggml_bf16_t));
    to_bf16(t, (ggml_bf16_t *) c->vir);
    c->flush();
    std::lock_guard<std::mutex> lock(g_pre_mu);
    g_pre[t->data] = { std::move(c), fp };
}

std::shared_ptr<model> make_qwen3_asr(const manifest & m) {
    return std::make_shared<qwen3_asr>(m);
}

}  // namespace axera
