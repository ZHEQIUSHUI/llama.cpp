// Axera NPU backend: Qwen3 text decoders (attention layers, dense MLP): Qwen3-Embedding, the Qwen3-ASR text model,
// the Qwen3-TTS talker
//
// One path, the manifest's kernels (ax-dsl-kernels models/<model>/text, qwen3_kernel.py), the whole model in one call:
//   prefill  role prefill: one call per page of 256 cells (a runtime loop over the layers, their weights in banks
//            [L / 4, ...] per phase), the last page of a batch padded (valid_tokens). Its history is whole pages: a batch
//            that starts inside a page restarts at the page's first cell with the page's earlier inputs (kept)
//   decode   role decode: one token through all layers (a decode group: the layers' weights in packs) on the same KV
//            cache; with the output head inside (ASR) it also gives a prompt's last token's logits
//   output   the logits (lm_head, the TTS codec head's padded to the text vocabulary), or for pooled embeddings the
//            final norm of every row (the pooling on the CPU)
// The prefill kernel ends with an RMS norm of unit weight, the decode group with the model's final norm when its head
// is inside, else of unit weight: the host's output head applies the model's final norm (the RMS norm of normed rows
// changes nothing).
// The KV cache is this adapter's: paged [L][pages][K, V][256][KV heads][128] bf16. One sequence at a time (pooled
// embeddings: the batch's sequences one after another, each from its first cell). Weights only from the GGUF.

#include "../model.h"
#include "../../core/params.h"
#include "../../core/mx.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"

#include <cmath>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;  // cells per page = prefill rows per call

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

class qwen3 final : public model {
  public:
    explicit qwen3(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, NBLK = 0, AX = 1;  // AX: position axes (1 RoPE, 3 M-RoPE)

    axera_cmm                        KV;
    std::map<std::string, axera_cmm> banks;
    call                             pre, dec;
    axera_cmm PX, POS, VALID, POUT, INDPTR, INDICES, LASTLEN, CLOC;
    axera_cmm DX, DOUT, COS, SIN, CS, PN, MASK, FNORM, HBUF, LOGITS;  // the decode group's (LOGITS: its head's)
    // the inputs of the current page's cells so far (from its first cell page_cell) and their positions [cells][4]
    int64_t                  page_cell = -1;
    std::vector<ggml_bf16_t> page_x;
    std::vector<int32_t>     page_p;
    output_head              head;

    void      init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    void      paging(int64_t hist, int64_t write_cell);
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    void      decode(int64_t c, int t, const ggml_bf16_t * x, const int32_t * p4, const rope & rp, float * h);
};

// a [rows] or [rows][cols] f32 view of any small GGUF tensor
std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) { memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float)); });
    }
    return v;
}

axera_cmm qwen3::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera qwen3: prefill: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const int         nb    = (int) d.shape[0];  // blocks (the cos / sin tables: one)
    GGML_ASSERT(d.bytes % nb == 0);
    axera_cmm    c  = axera_cmm_alloc(d.bytes);
    const size_t bb = d.bytes / nb;  // bytes per block
    auto         T  = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera qwen3: %s not in the graph\n", nm.c_str());
        }
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        const int il  = 4 * b + phase;
        uint8_t * dst = (uint8_t *) c->vir + (size_t) b * bb;
        auto      put = [&](const std::vector<float> & v, float scale = 1.0f) {
            GGML_ASSERT(v.size() * (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t)) == bb);
            for (size_t i = 0; i < v.size(); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v[i] * scale;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i] * scale);
                }
            }
        };
        auto fill = [&](float v) {  // a block's global scales / zero points: every element
            put(std::vector<float>(bb / (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t)), v));
        };
        auto mx8 = [&](const char * g, bool scales) {  // a weight [N, K] in MXFP8: elements [N][K] or scales [N][K / 32]
            const ggml_tensor * w = T(il, g);
            const int64_t       K = w->ne[0];
            GGML_ASSERT((size_t) ggml_nrows(w) * (scales ? K / AXERA_MX_GROUP : K) == bb);
            axera_for_rows_f32(w, [&](int64_t r, const float * row) {
                thread_local std::vector<uint8_t> q, s;
                q.resize(K);
                s.resize(K / AXERA_MX_GROUP);
                axera_quantize_row_mxfp8_f32(row, q.data(), s.data(), K);
                if (scales) {
                    memcpy(dst + r * (K / AXERA_MX_GROUP), s.data(), s.size());
                } else {
                    memcpy(dst + r * K, q.data(), q.size());
                }
            });
        };
        if (n == "cos" || n == "sin") {  // [positions][ROT / 2] f32
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT(d.dtype == 3 && (size_t) R == rp.inv_freq.size());
            for (int64_t q = 0; q < P; ++q) {
                for (int64_t i = 0; i < R; ++i) {
                    const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                    ((float *) dst)[q * R + i] = (float) (n == "cos" ? std::cos(th) : std::sin(th));
                }
            }
        } else if (n.rfind("w_scale2_", 0) == 0) {  // (w_scale2 alone: the MLP up projection's scales)
            fill(1.0f);
        } else if (n.rfind("w_zeropoint_", 0) == 0 || n.rfind("w_zp_scale", 0) == 0 || n == "input_permutation") {
            memset(dst, 0, bb);
        } else if (mlp) {
            static const std::map<std::string, const char *> W = { { "1", "ffn_gate.weight" }, { "2", "ffn_up.weight" },
                                                                    { "3", "ffn_down.weight" } };
            if (n == "w_norm") {
                put(f32_of(T(il, "ffn_norm.weight")));
            } else if (n.size() == 2 && W.count(n.substr(1))) {
                mx8(W.at(n.substr(1)), false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                mx8(W.at(n.substr(7)), true);
            } else {
                GGML_ABORT("axera qwen3: prefill: unexpected MLP input %s\n", d.name.c_str());
            }
        } else {
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (n == "w_norm_q") {  // the attention scale folded in
                put(f32_of(T(il, "attn_q_norm.weight")), 1.0f / std::sqrt((float) d.shape.back()));
            } else if (n == "w_norm_k") {
                put(f32_of(T(il, "attn_k_norm.weight")));
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                mx8(W.at(p), n != "w_" + p);
            } else {
                GGML_ABORT("axera qwen3: prefill: unexpected attention input %s\n", d.name.c_str());
            }
        }
    }
    c->flush();
    return c;
}

void qwen3::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    const kernel & pk = man.get("prefill");
    const kernel * dk = man.kernels.count("decode") ? &man.get("decode") : nullptr;  // (pooled embeddings: none)
    const io_desc * kv = nullptr;
    for (const auto & d : pk.desc->in) {
        kv = d.name == "b0_kv" ? &d : kv;
    }
    GGML_ASSERT(kv && kv->shape[0] == L && kv->shape[3] == PG);
    NBLK = (int) kv->shape[1];
    GGML_ASSERT((int64_t) NBLK * PG == man.context && (!dk || dk->in_bytes("kv") == kv->bytes));
    AX = (int) (pk.in_bytes("position_ids") / (PG * sizeof(int32_t)));
    GGML_ASSERT((AX == 1 || AX == 3) && pk.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                (!dk || dk->in_bytes("x") == (size_t) H * sizeof(ggml_bf16_t)));
    KV      = axera_cmm_alloc(kv->bytes);
    PX      = axera_cmm_alloc(pk.in_bytes("x"));
    POUT    = axera_cmm_alloc(pk.out_bytes("output"));
    POS     = axera_cmm_alloc(pk.in_bytes("position_ids"));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc((size_t) NBLK * sizeof(int32_t));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();
    page_cell = -1;
    page_x.clear();
    page_p.clear();

    const rope  rp(rope_node, names.find("inp_pos"), 1);
    axera_timer tb("init: banks");
    banks.clear();
    banks["final_norm"] = axera_cmm_alloc(pk.in_bytes("final_norm"));
    for (size_t i = 0; i < pk.in_bytes("final_norm") / sizeof(ggml_bf16_t); ++i) {
        ((ggml_bf16_t *) banks["final_norm"]->vir)[i] = ggml_fp32_to_bf16(1.0f);
    }
    banks["final_norm"]->flush();
    // prefill: the weight banks, the KV cache, one page's rows
    pre   = {};
    pre.k = &pk;
    for (const auto & d : pk.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &PX : n == "position_ids" ? &POS : n == "valid_tokens" ? &VALID :
                                n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                n == "cache_location" ? &CLOC : n == "b0_kv" ? &KV : nullptr;
        if (!b) {
            auto it = banks.find(n);
            b       = it != banks.end() ? &it->second : &(banks[n] = bank(t, d, rp));
        }
        GGML_ASSERT((*b)->size >= d.bytes);
        pre.in.push_back(axera_io(*b, d.bytes));
    }
    GGML_ASSERT(pk.desc->out.size() == 1 && pk.desc->out[0].name == "output");
    pre.out.push_back(axera_io(POUT, pk.desc->out[0].bytes));
    // decode: the group's packs from the GGUF, the same KV cache; cs = cos | sin, pn = cell | blocks | the K row of the
    // cell, the final norm the model's when the head is inside
    dec    = {};
    LOGITS = {};
    if (dk) {
        dec.k = dk;
        bool head_in = false;
        for (const auto & d : dk->desc->out) {
            head_in = head_in || d.name == "logits";
        }
        const uint32_t csb = dk->in_bytes("cs");
        DX    = axera_cmm_alloc(dk->in_bytes("x"));
        DOUT  = axera_cmm_alloc(dk->out_bytes("out"));
        COS   = axera_cmm_alloc(csb / 2);
        SIN   = axera_cmm_alloc(csb / 2);
        CS    = axera_cmm_alloc(csb);
        PN    = axera_cmm_alloc(3 * sizeof(int32_t));
        MASK  = axera_cmm_alloc(dk->in_bytes("mask"));
        HBUF  = axera_cmm_alloc(dk->in_bytes("hbuf"));
        FNORM = axera_cmm_alloc(dk->in_bytes("fnorm"));
        GGML_ASSERT(dk->in_bytes("pn") == 3 * sizeof(int32_t) && dk->in_bytes("mask") == (size_t) man.context * sizeof(ggml_bf16_t) &&
                    dk->in_bytes("fnorm") == (size_t) H * sizeof(ggml_bf16_t) && csb / sizeof(ggml_bf16_t) == 2 * rp.inv_freq.size());
        const std::vector<float> fn = head_in ? f32_of(t.at("output_norm.weight")) : std::vector<float>(H, 1.0f);
        ggml_fp32_to_bf16_row(fn.data(), (ggml_bf16_t *) FNORM->vir, H);
        FNORM->flush();
        for (const auto & d : dk->desc->in) {
            const std::string & n = d.name;
            if (dk->desc->packs.count(n)) {
                dec.in.push_back(axera_io(prm.pack(t, *dk->desc, n, 0), d.bytes));
                continue;
            }
            const axera_cmm * b = n == "x" ? &DX : n == "cs" ? &CS : n == "pn" ? &PN : n == "mask" ? &MASK : n == "kv" ? &KV :
                                  n == "fnorm" ? &FNORM : n == "hbuf" ? &HBUF : nullptr;
            if (!b) {
                GGML_ABORT("axera qwen3: decode: unexpected input %s\n", n.c_str());
            }
            dec.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : dk->desc->out) {
            if (d.name == "logits") {
                LOGITS = axera_cmm_alloc(d.bytes);
            }
            dec.out.push_back(axera_io(d.name == "logits" ? LOGITS : DOUT, d.bytes));
        }
        prm.release({ &dec.in });
    }
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen3: %d layers, banks %.2f GiB, KV cache %.2f GiB, %lld cells\n", L, bb / 1073741824.0,
                  KV->size / 1073741824.0, (long long) man.context);
}

// the history before the token(s) written from cell write_cell: hist cells in whole pages and a last one
void qwen3::paging(int64_t hist, int64_t write_cell) {
    const int64_t pages           = (hist + PG - 1) / PG;
    ((int32_t *) INDPTR->vir)[0]  = 0;
    ((int32_t *) INDPTR->vir)[1]  = (int32_t) pages;
    *(int32_t *) LASTLEN->vir     = (int32_t) (pages ? hist - (pages - 1) * PG : 0);
    ((int32_t *) CLOC->vir)[0]    = (int32_t) (write_cell / PG);
    ((int32_t *) CLOC->vir)[1]    = (int32_t) (write_cell % PG);
    for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
        (*b)->flush();
    }
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16, their positions p4 [n][4]; the hidden states of
// the rows from out0 on into h
void qwen3::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) PX->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        PX->flush();
        auto * pp = (int32_t *) POS->vir;
        for (int s = 0; s < AX; ++s) {
            for (int r = 0; r < PG; ++r) {
                pp[s * PG + r] = r < v ? p4[(size_t) (r0 + r) * 4 + s] : 0;
            }
        }
        POS->flush();
        paging(cell, cell);
        *(int32_t *) VALID->vir = v;
        VALID->flush();
        {
            axera_timer tl("prefill (page)");
            pre.k->run(pre.in, pre.out, "prefill");
        }
        POUT->invalidate(POUT->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
        // the inputs of a partial last page: a later batch restarts the page with them
        page_cell = v < PG ? cell : cell + PG;
        page_x.assign(x + (size_t) r0 * H, x + (size_t) (r0 + (v < PG ? v : 0)) * H);
        page_p.assign(p4 + (size_t) r0 * 4, p4 + (size_t) (r0 + (v < PG ? v : 0)) * 4);
    }
}

// one token at cell c > 0 (the batch's token t), its history the c cells before it
void qwen3::decode(int64_t c, int t, const ggml_bf16_t * x, const int32_t * p4, const rope & rp, float * h) {
    memcpy(DX->vir, x, H * sizeof(ggml_bf16_t));
    DX->flush();
    rp.rows(t, 1, COS.get(), SIN.get());
    const size_t half = CS->size / 2;
    memcpy(CS->vir, COS->vir, half);
    memcpy((char *) CS->vir + half, SIN->vir, half);
    CS->flush();
    ((int32_t *) PN->vir)[0] = (int32_t) c;
    ((int32_t *) PN->vir)[1] = (int32_t) (c / PG + 1);
    ((int32_t *) PN->vir)[2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
    PN->flush();
    auto *         m    = (ggml_bf16_t *) MASK->vir;  // causal over the flat cells of the populated pages
    const int64_t  w    = (c / PG + 1) * PG;
    const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;
    for (int64_t i = 0; i < w; ++i) {
        m[i].bits = i <= c ? 0 : ninf;
    }
    MASK->flush(MASK->vir, (size_t) w * sizeof(ggml_bf16_t));
    {
        axera_timer tl("layers (decode)");
        dec.k->run(dec.in, dec.out, "decode");
    }
    DOUT->invalidate(DOUT->vir, H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) DOUT->vir, h, H);
    if (c % PG == 0) {
        page_cell = c;
        page_x.clear();
        page_p.clear();
    }
    page_x.insert(page_x.end(), x, x + H);
    page_p.insert(page_p.end(), p4, p4 + 4);
}

ggml_status qwen3::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = graph_head(names);
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !inp_pos || (!logits && !names.find("result_norm") && !names.find("result_embd_pooled"))) {
        GGML_LOG_ERROR("axera qwen3: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {  // a layer-0 input computed in this graph (the TTS talker's): on the CPU
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
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera qwen3: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view   kv(k_idxs, kqm, man.context, PG);
    const int32_t * pos = (const int32_t *) inp_pos->data;
    const bool      mr  = inp_pos->ne[0] == 4 * (int64_t) T;  // M-RoPE: [4][T], else [T]
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
        }
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
    });
    std::vector<float> h((size_t) T * H);
    axera_timer        tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const rope         rp(rope_node, inp_pos, T);
    // the logits of the batch's last token only, and a decode group with the head: that token through the group
    const bool group_head = logits && LOGITS && logits->ne[1] == 1;
    if (logits && logits->ne[1] > 0 && !group_head && !man.kernels.count("lm_head")) {
        GGML_LOG_ERROR("axera qwen3: logits of %lld rows; the kernel directory has no lm_head\n", (long long) logits->ne[1]);
        return GGML_STATUS_FAILED;
    }
    // token t sees cell c (llama.cpp may keep several sequences in one stream, one after another: a sequence's cells
    // are this cache's from its first on)
    auto sees = [&](int t, int64_t c) {
        const char * row = (const char *) kqm->data + (kqm->ne[3] == 1 ? t * kqm->nb[1] :
                                                           (t / kqm->ne[1]) * kqm->nb[3] + (t % kqm->ne[1]) * kqm->nb[1]);
        const float  v   = kqm->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[c]) : ((const float *) row)[c];
        return v > -1e30f;
    };
    for (int it = 0; it < T;) {
        // one sequence's consecutive cells from token it
        int64_t first = kv.cell_of(it);
        while (first > 0 && sees(it, first - 1)) {
            --first;
        }
        int run = 1;
        while (it + run < T && kv.stream_of(it + run) == kv.stream_of(it) && kv.cell_of(it + run) == kv.cell_of(it) + run &&
               (first == kv.cell_of(it + run) || sees(it + run, first))) {
            ++run;
        }
        const int64_t c0 = kv.cell_of(it) - first;
        if (c0 > 0 && axera_n_seq_max() != 1) {
            GGML_LOG_ERROR("axera qwen3: a continued sequence needs one sequence only (-np 1)\n");
            return GGML_STATUS_FAILED;
        }
        if (run == 1 && c0 > 0 && dec.k) {  // one token: the decode kernel
            decode(c0, it, xb.data() + (size_t) it * H, pb.data() + (size_t) it * 4, rp, h.data() + (size_t) it * H);
            ++it;
            continue;
        }
        if (group_head && it + run == T && run > 1) {  // the last token through the group (its head)
            --run;
        }
        // from the run's page start: the page's earlier inputs before the run's
        const int64_t s0 = c0 - c0 % PG;
        const int     k  = (int) (c0 - s0);
        if (k > 0 && (page_cell != s0 || (int) (page_x.size() / H) < k)) {
            GGML_LOG_ERROR("axera qwen3: a batch from cell %lld, inside a page whose inputs are not kept\n", (long long) c0);
            return GGML_STATUS_FAILED;
        }
        page_x.resize((size_t) k * H);  // (a kept prefix, e.g. the server's prompt cache: the cells after it are gone)
        page_p.resize((size_t) k * 4);
        std::vector<ggml_bf16_t> xs(page_x.begin(), page_x.begin() + (size_t) k * H);
        std::vector<int32_t>     ps(page_p.begin(), page_p.begin() + (size_t) k * 4);
        xs.insert(xs.end(), xb.begin() + (size_t) it * H, xb.begin() + (size_t) (it + run) * H);
        ps.insert(ps.end(), pb.begin() + (size_t) it * 4, pb.begin() + (size_t) (it + run) * 4);
        std::vector<float> hr((size_t) run * H);
        prefill(s0, k + run, xs.data(), ps.data(), hr, k);
        memcpy(h.data() + (size_t) it * H, hr.data(), hr.size() * sizeof(float));
        it += run;
    }
    if (!logits) {
        output_norm(names, h, T);
    } else if (logits->ne[1] == 0) {  // a batch whose logits nobody reads (mtmd's prompt chunks before the last)
    } else if (group_head) {  // the group's logits of the last token
        GGML_ASSERT(logits->type == GGML_TYPE_F32 && ggml_nbytes(logits) == LOGITS->size);
        LOGITS->invalidate();
        memcpy(logits->data, LOGITS->vir, LOGITS->size);
        if (ggml_backend_buffer_is_axera(logits->buffer)) {
            ((ggml_backend_axera_buffer_context *) logits->buffer->context)->flush(logits->data, ggml_nbytes(logits));
        }
        output_head::after(g, names);
    } else {
        head.bind(man.get("lm_head"), logits->src[0]);  // output.weight, or the tied token_embd.weight
        head.run(g, names, h, T);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen3(const manifest & m) {
    return std::make_shared<qwen3>(m);
}

}  // namespace axera
