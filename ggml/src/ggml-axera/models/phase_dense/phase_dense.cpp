// Axera NPU backend: the dense TTS language models whose two whole-model kernels run their layers as a runtime loop
// over four phases, each phase's weights banked [layers / 4, ...] (one adapter, the per-model facts in phase_family):
//   voxcpm2_base    VoxCPM2 BaseLM (MiniCPM, 28 layers, H 2048, 16 Q / 2 KV heads of 128). The GGUF's Q/K rows are in
//                   llama.cpp's adjacent-pair RoPE order; the kernel rotates MiniCPM's half-split layout, so the rows map
//                   back. The RoPE table carries the GGUF's LongRoPE factors (llama.cpp's theta / factor).
//   cosyvoice3_llm  Fun-CosyVoice3 LLM (Qwen2, 24 layers, H 896, 14 Q / 2 KV heads of 64, Q/K/V bias, theta 1e6).
//                   Q is padded from 14 to 16 head slots so every core owns four heads, the output projection to the
//                   same 1024 input columns; the dummy slots are zero. The 6761-row speech head rides inside both
//                   kernels (the padded fp32 logits row and its top-k).
//
// The manifest's kernels:
//   prefill    role prefill: 256 cells through all layers, final RMSNorm included (an optional 16 / 32-row
//              prefill16 keeps a short prompt from padding to the page)
//   decode     role decode: one cell through the same layers and the same paged BF16 KV cache
// The KV cache is this adapter's: [layers][pages][K, V][256][KV heads][head_dim] bf16, one sequence at a time; the
// sequence state is the whole cache image (core/seqstate owns restart points, rewinds and misses).

#include "../model.h"
#include "../../core/quant.h"
#include "../../core/seqstate.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;

// what the kernels of one model were compiled for
struct phase_family {
    const char * model;         // the manifest's model name
    int          head_dim;
    bool         qk_pair_rope;  // GGUF q/k rows in llama.cpp's adjacent-pair RoPE order
    const int *  slot_head;     // the Q head of each kernel head slot (-1: a zero slot); nullptr: no padding
    int          slots;
    double       rope_theta;    // > 0: the kernel's own RoPE table; 0: the graph's RoPE (and its factors)
    int          speech_vocab;  // rows of the speech head inside the kernels (0: none)
};

constexpr int COSY_SLOT_HEAD[16] = { 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, -1 };

constexpr phase_family families[] = {
    { "voxcpm2_base", 128, true, nullptr, 0, 0.0, 0 },
    { "cosyvoice3_llm", 64, false, COSY_SLOT_HEAD, 16, 1.0e6, 6761 },
};

using call = npu_call;

class phase_dense final : public model, public seq_model {
  public:
    explicit phase_dense(const manifest & m) : man(m) {
        for (const auto & f : families) {
            if (m.model == f.model) fam = &f;
        }
        GGML_ASSERT(fam);
    }

    ggml_status compute(ggml_cgraph * g) override;
    int         last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override;
    void     seq_state_load(const uint8_t * src) override;
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override;

    size_t state_size(uint32_t flags) const override;
    size_t state_write(void * dst, size_t size, uint32_t flags) override;
    size_t state_read(const void * src, size_t size, uint32_t flags) override;

  private:
    const manifest &     man;
    const phase_family * fam = nullptr;
    graph_names          names;
    const void *         bound = nullptr;
    int                  L = 0, H = 0, NBLK = 0;
    std::vector<float>   rope_factors;

    axera_cmm                        KV;
    std::map<std::string, axera_cmm> banks;
    // one stage's call and its own buffers (the paging inputs and the KV are shared)
    struct stage {
        call      c;
        axera_cmm x, pos, valid, out, logits, topk_v, topk_i;
        int       rows = 0;
    };
    stage                    pre, pre16, dec;
    axera_cmm                INDPTR, INDICES, LASTLEN, CLOC;
    const axera_cmm *        last_logits = nullptr;
    size_t                   speech_pad = 0;  // the kernels' padded logits row

    const ggml_bf16_t *      batch_rows = nullptr;
    std::vector<ggml_bf16_t> src_rows;
    std::vector<float>       h_buf;
    float *                  out_h = nullptr;

    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;

    void      init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope * rp);
    void      bind(stage & s, const kernel & k, bool decode_stage);
    void      run_page(stage & s, int64_t cell, int v, const ggml_bf16_t * x, const int32_t * p4, int row0, int batch0,
                       const char * what);
    void      decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row);

    // the source row of a projection's destination row r (-1: a zero row)
    int64_t src_row(const std::string & p, int64_t r) const {
        const int64_t hd = fam->head_dim;
        if (fam->qk_pair_rope && (p == "q" || p == "k")) {  // half-split row -> adjacent pair (conversion/minicpm.py)
            const int64_t j = r % hd;
            return r / hd * hd + 2 * (j % (hd / 2)) + j / (hd / 2);
        }
        if (fam->slot_head && p == "q") {
            const int head = fam->slot_head[r / hd];
            return head < 0 ? -1 : head * hd + r % hd;
        }
        return r;
    }
};

static void row_f32(const ggml_tensor * w, int64_t r, float * dst) {
    const void * src = (const char *) w->data + r * w->nb[1];
    if (w->type == GGML_TYPE_F32) {
        memcpy(dst, src, w->ne[0] * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(w->type);
        GGML_ASSERT(traits->to_float);
        traits->to_float(src, dst, w->ne[0]);
    }
}

axera_cmm phase_dense::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope * rp) {
    auto bf16_bank = [&](const std::vector<float> & v) {
        axera_cmm c = axera_cmm_alloc_raw(d.bytes);
        GGML_ASSERT(v.size() * sizeof(ggml_bf16_t) <= d.bytes);
        std::vector<ggml_bf16_t> host(d.bytes / sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row(v.data(), host.data(), (int64_t) v.size());
        memcpy(c->vir, host.data(), d.bytes);
        c->flush();
        return c;
    };
    if (d.name == "final_norm") {
        auto v = f32_of(t.at("output_norm.weight"));
        GGML_ASSERT(v.size() * sizeof(ggml_bf16_t) == d.bytes);
        return bf16_bank(v);
    }
    if (d.name == "speech_head") {
        // The head GGUF stores the speech decoder in output.weight's first rows (the rest of the text vocabulary is
        // zero). The kernel pads it to four equal core slices; the padded rows are zero and masked to -inf in the graph.
        if (!t.count("output.weight")) {
            GGML_ABORT("axera %s: speech_head needs output.weight in the model\n", fam->model);
        }
        const ggml_tensor * w = t.at("output.weight");
        GGML_ASSERT(w->ne[0] == H && w->ne[1] >= fam->speech_vocab && d.dtype == 9 && d.bytes % (H * 2) == 0);
        std::vector<float> v((size_t) fam->speech_vocab * H);
        for (int64_t r = 0; r < fam->speech_vocab; ++r) {
            row_f32(w, r, v.data() + (size_t) r * H);
        }
        axera_cmm c = bf16_bank(v);
        axera_drop_mapped(w);
        return c;
    }

    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera %s: prefill: unexpected input %s\n", fam->model, d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(nb > 0 && d.bytes % nb == 0);
    axera_cmm c = axera_cmm_alloc_raw(d.bytes);
    // convert into ordinary memory first: a byte-wise store stream through the cached CMM mapping is much slower
    // than one contiguous copy
    std::vector<uint8_t> host(d.bytes);
    const size_t         bb         = d.bytes / nb;
    const bool           keep_pages = axera_keep_pages();
    const bool           f32        = d.dtype == 3;
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera %s: %s not in the graph\n", fam->model, nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    const int64_t hd = fam->head_dim;
    for (int b = 0; b < nb; ++b) {
        axera_mapped_drop_guard drop(read, !keep_pages);
        const int il  = 4 * b + phase;
        uint8_t * dst = host.data() + (size_t) b * bb;
        auto put = [&](const std::vector<float> & v) {
            GGML_ASSERT(v.size() * (f32 ? sizeof(float) : sizeof(ggml_bf16_t)) == bb);
            for (size_t i = 0; i < v.size(); ++i) {
                if (f32) {
                    ((float *) dst)[i] = v[i];
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i]);
                }
            }
        };
        auto fill = [&](float v) { put(std::vector<float>(bb / (f32 ? sizeof(float) : sizeof(ggml_bf16_t)), v)); };
        // projection p's weight in the kernel's layout: rows (head slots) and, for o, input columns mapped from the
        // GGUF; MXFP8 elements [N][K] or scales [N][K / 32], or BF16 [N][K]
        auto weight = [&](const char * g, const std::string & p, bool scales) {
            const ggml_tensor * w     = T(il, g);
            const bool          pad_o = fam->slot_head && p == "o";
            const int64_t       K     = pad_o ? fam->slots * hd : w->ne[0];
            const int64_t       N     = fam->slot_head && p == "q" ? fam->slots * hd : ggml_nrows(w);
            const bool          bf16  = d.dtype == 9;
            GGML_ASSERT(!(bf16 && scales));
            GGML_ASSERT((size_t) N * (bf16 ? K * sizeof(ggml_bf16_t) : scales ? K / AXERA_MX_GROUP : K) == bb);
            std::vector<float>   row(K), src(w->ne[0]);
            std::vector<uint8_t> q(K), s(K / AXERA_MX_GROUP);
            for (int64_t r = 0; r < N; ++r) {
                const int64_t sr = src_row(p, r);
                std::fill(row.begin(), row.end(), 0.0f);
                if (sr >= 0 && pad_o) {
                    row_f32(w, sr, src.data());
                    for (int slot = 0; slot < fam->slots; ++slot) {
                        if (fam->slot_head[slot] >= 0) {
                            memcpy(row.data() + slot * hd, src.data() + fam->slot_head[slot] * hd, hd * sizeof(float));
                        }
                    }
                } else if (sr >= 0) {
                    row_f32(w, sr, row.data());
                }
                if (bf16) {
                    ggml_fp32_to_bf16_row(row.data(), (ggml_bf16_t *) dst + r * K, K);
                    continue;
                }
                axera_quantize_row_mxfp8_f32(row.data(), q.data(), s.data(), K);
                if (scales) {
                    memcpy(dst + r * (K / AXERA_MX_GROUP), s.data(), s.size());
                } else {
                    memcpy(dst + r * K, q.data(), q.size());
                }
            }
        };
        // a projection bias (Q's mapped to the head slots)
        auto bias = [&](const char * g, const std::string & p) {
            const std::vector<float> v = f32_of(T(il, g));
            if (!(fam->slot_head && p == "q")) {
                put(v);
                return;
            }
            std::vector<float> o((size_t) fam->slots * hd, 0.0f);
            for (int64_t r = 0; r < (int64_t) o.size(); ++r) {
                const int64_t sr = src_row(p, r);
                o[(size_t) r] = sr < 0 ? 0.0f : v[(size_t) sr];
            }
            put(o);
        };
        if (n == "cos" || n == "sin") {
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT(f32 && (fam->rope_theta > 0 ? R == hd / 2 : rp && (size_t) R == rp->inv_freq.size()));
            for (int64_t p = 0; p < P; ++p) {
                for (int64_t i = 0; i < R; ++i) {
                    double th;
                    if (fam->rope_theta > 0) {
                        th = (double) p * (1.0 / std::pow(fam->rope_theta, (double) (2 * i) / (double) hd));
                    } else {
                        // llama.cpp applies the GGUF's per-dimension LongRoPE factors as theta / factor; the
                        // kernel's RoPE takes plain inv_freq, so bake them in
                        const double inv = (size_t) i < rope_factors.size()
                                               ? (double) rp->inv_freq[(size_t) i] / (double) rope_factors[(size_t) i]
                                               : (double) rp->inv_freq[(size_t) i];
                        th = (double) p * rp->freq_scale * inv;
                    }
                    ((float *) dst)[(size_t) p * R + i] = (float) (n == "cos" ? std::cos(th) : std::sin(th));
                }
            }
        } else if (n.rfind("w_scale2_", 0) == 0) {
            fill(1.0f);
        } else if (n.rfind("w_zp_scale", 0) == 0 || n == "input_permutation") {
            memset(dst, 0, bb);
        } else if (mlp) {
            static const std::map<std::string, const char *> W = { { "1", "ffn_gate.weight" }, { "2", "ffn_up.weight" },
                                                                    { "3", "ffn_down.weight" } };
            if (n == "w_norm") {
                put(f32_of(T(il, "ffn_norm.weight")));
            } else if (n.size() == 2 && W.count(n.substr(1))) {
                weight(W.at(n.substr(1)), "", false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                if (d.bytes <= sizeof(float)) {  // a BF16 weight's placeholder scale
                    fill(1.0f);
                } else {
                    weight(W.at(n.substr(7)), "", true);
                }
            } else {
                GGML_ABORT("axera %s: prefill: unexpected MLP input %s\n", fam->model, d.name.c_str());
            }
        } else {
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (n == "q_bias" || n == "k_bias" || n == "v_bias") {
                const std::string bp = n.substr(0, 1);
                bias(("attn_" + bp + ".bias").c_str(), bp);
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                if (n == "w_scale_" + p && d.bytes <= sizeof(float)) {
                    fill(1.0f);
                } else {
                    weight(W.at(p), p, n != "w_" + p);
                }
            } else {
                GGML_ABORT("axera %s: prefill: unexpected attention input %s\n", fam->model, d.name.c_str());
            }
        }
    }
    memcpy(c->vir, host.data(), host.size());
    c->flush();
    if (keep_pages) {
        for (const ggml_tensor * w : read) {
            axera_drop_mapped(w);
        }
    }
    return c;
}

// a stage's call: its own row buffers and outputs, the shared paging inputs, KV and banks
void phase_dense::bind(stage & s, const kernel & k, bool decode_stage) {
    s      = {};
    s.c.k  = &k;
    s.x    = axera_cmm_alloc(k.in_bytes("x"));
    s.rows = (int) (k.in_bytes("x") / (H * sizeof(ggml_bf16_t)));
    // decode reads the prefill's position / valid buffers (one value each)
    if (!decode_stage) {
        s.pos   = axera_cmm_alloc(k.in_bytes("position_ids"));
        s.valid = axera_cmm_alloc(sizeof(int32_t));
    }
    for (const auto & d : k.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &s.x : n == "position_ids" ? (decode_stage ? &pre.pos : &s.pos) :
                                n == "valid_tokens" ? (decode_stage ? &pre.valid : &s.valid) :
                                n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                n == "cache_location" ? &CLOC : n == "b0_kv" ? &KV : &banks.at(n);
        GGML_ASSERT(b->get() && (*b)->size >= d.bytes);
        s.c.in.push_back(axera_io(*b, d.bytes));
    }
    for (const auto & d : k.desc->out) {
        axera_cmm * b = d.name == "output" ? &s.out : d.name == "logits" ? &s.logits :
                        d.name == "topk_values" ? &s.topk_v : d.name == "topk_ids" ? &s.topk_i : nullptr;
        if (!b) {
            GGML_ABORT("axera %s: %s: unexpected output %s\n", fam->model, k.desc->role.c_str(), d.name.c_str());
        }
        *b = axera_cmm_alloc(d.bytes);
        s.c.out.push_back(axera_io(*b, d.bytes));
    }
    GGML_ASSERT(s.out);
}

void phase_dense::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");

    const io_desc * kv = nullptr;
    for (const auto & d : pk.desc->in) {
        kv = d.name == "b0_kv" ? &d : kv;
    }
    GGML_ASSERT(kv && kv->shape[0] == L && kv->shape[3] == PG);
    NBLK = (int) kv->shape[1];
    GGML_ASSERT((int64_t) NBLK * PG == man.context && dk.in_bytes("b0_kv") == kv->bytes);
    GGML_ASSERT(pk.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                dk.in_bytes("x") == (size_t) H * sizeof(ggml_bf16_t));

    KV      = axera_cmm_alloc(kv->bytes);
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc((size_t) NBLK * sizeof(int32_t));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    std::unique_ptr<rope> rp;
    rope_factors.clear();
    if (fam->rope_theta <= 0) {
        GGML_ASSERT(rope_node && "no ROPE node in the graph");
        rp = std::make_unique<rope>(rope_node, names.find("inp_pos"), 1);
        if (rope_node->src[2] != nullptr) {
            rope_factors = f32_of(rope_node->src[2]);
            if (rope_factors.size() != rp->inv_freq.size()) {
                GGML_LOG_WARN("axera %s: rope factors have %zu rows, expected %zu; ignoring them\n", fam->model,
                              rope_factors.size(), rp->inv_freq.size());
                rope_factors.clear();
            }
        }
    }
    axera_timer tb("init: banks");
    banks.clear();
    std::vector<const io_desc *> work;
    for (const auto & d : pk.desc->in) {
        const std::string & n = d.name;
        if (n == "x" || n == "position_ids" || n == "valid_tokens" || n == "indptr" || n == "indices" ||
            n == "last_page_len" || n == "cache_location" || n == "b0_kv") {
            continue;
        }
        work.push_back(&d);
    }
    // the banks touch disjoint tensors: convert them in parallel (each into host memory, then one copy per bank)
    std::vector<axera_cmm> made(work.size());
    axera_host_pool().run((int64_t) work.size(), [&](int64_t i) { made[(size_t) i] = bank(t, *work[(size_t) i], rp.get()); });
    for (size_t i = 0; i < work.size(); ++i) {
        banks[work[i]->name] = std::move(made[i]);
    }
    bind(pre, pk, false);
    bind(dec, dk, true);
    pre16 = {};
    if (man.has("prefill16")) {  // the same graph at 16 / 32 rows: a short prompt does not pad to the page
        const kernel & sk = man.get("prefill16");
        GGML_ASSERT(sk.in_bytes("b0_kv") == kv->bytes);
        bind(pre16, sk, false);
        GGML_ASSERT(pre16.rows == 16 || pre16.rows == 32);
    }
    speech_pad = 0;
    if (fam->speech_vocab) {
        GGML_ASSERT(pre.logits && dec.logits && pre.logits->size == dec.logits->size);
        speech_pad = dk.out_bytes("logits") / sizeof(float);
        GGML_ASSERT((int) speech_pad >= fam->speech_vocab);
    }

    seq = std::make_unique<seqstate>(*this, seq_config_env(1));
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera %s: %d layers, banks %.2f GiB, KV cache %.2f GiB, %lld cells%s\n", fam->model, L,
                  bb / 1073741824.0, KV->size / 1073741824.0, (long long) man.context,
                  rp ? (rope_factors.empty() ? ", no rope factors" : ", rope factors applied") : "");
}

// one call of a prefill stage: v rows of x from cell (a page's first), the hidden rows from batch row batch0 on to out_h
void phase_dense::run_page(stage & s, int64_t cell, int v, const ggml_bf16_t * x, const int32_t * p4, int row0,
                           int batch0, const char * what) {
    memcpy(s.x->vir, x, (size_t) v * H * sizeof(ggml_bf16_t));
    memset((ggml_bf16_t *) s.x->vir + (size_t) v * H, 0, (size_t) (s.rows - v) * H * sizeof(ggml_bf16_t));
    s.x->flush();
    int32_t * pp = (int32_t *) s.pos->vir;
    for (int r = 0; r < s.rows; ++r) {
        pp[r] = r < v ? p4[(size_t) r * 4] : 0;
    }
    s.pos->flush();
    llm_paging(INDPTR, LASTLEN, CLOC, cell, cell, PG);
    *(int32_t *) s.valid->vir = v;
    s.valid->flush();
    {
        axera_timer tl(what);
        s.c.k->run(s.c.in, s.c.out, s.c.k->desc->role.c_str());
    }
    s.out->invalidate(s.out->vir, (size_t) v * H * sizeof(ggml_bf16_t));
    for (axera_cmm * b : { &s.logits, &s.topk_v, &s.topk_i }) {
        if (*b) {
            (*b)->invalidate((*b)->vir, (*b)->size);
        }
    }
    if (s.logits) {
        last_logits = &s.logits;
    }
    for (int r = 0; r < v; ++r) {
        const int global = row0 + r;
        if (global >= batch0 && out_h) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) s.out->vir + (size_t) r * H, out_h + (size_t) (global - batch0) * H,
                                  H);
        }
    }
}

void phase_dense::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row) {
    memcpy(dec.x->vir, x, (size_t) H * sizeof(ggml_bf16_t));
    dec.x->flush();
    *(int32_t *) pre.pos->vir = p4[0];
    pre.pos->flush();
    llm_paging(INDPTR, LASTLEN, CLOC, c, c, PG);
    {
        axera_timer tl("decode");
        dec.c.k->run(dec.c.in, dec.c.out, "decode");
    }
    dec.out->invalidate(dec.out->vir, (size_t) H * sizeof(ggml_bf16_t));
    for (axera_cmm * b : { &dec.logits, &dec.topk_v, &dec.topk_i }) {
        if (*b) {
            (*b)->invalidate((*b)->vir, (*b)->size);
        }
    }
    if (dec.logits) {
        last_logits = &dec.logits;
    }
    if (out_h) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) dec.out->vir, out_h + (size_t) batch_row * H, H);
    }
}

// a graph that asks for more logits rows than the kernels' one: the speech head's rows of output.weight on the CPU
static void host_speech_logits(const ggml_tensor * w, const float * h, float * out, int64_t n) {
    const int64_t K = w->ne[0];
    axera_host_pool().run(n, [&](int64_t r) {
        thread_local std::vector<float> row;
        row.resize(K);
        row_f32(w, r, row.data());
        float acc = 0.0f;
        for (int64_t i = 0; i < K; ++i) {
            acc += row[i] * h[i];
        }
        out[r] = acc;
    });
}

ggml_status phase_dense::compute(ggml_cgraph * g) {
    last_seq_status = 0;
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = graph_head(names);
    ggml_tensor * inp_pos = names.find("inp_pos");
    ggml_tensor * result  = names.find("result_norm");
    if (!result) {
        result = names.find("result_embd_pooled");
    }
    if (!norm0 || !inp_pos || (!result && !logits)) {
        GGML_LOG_ERROR("axera %s: a graph of %d nodes that is not the whole model (-ngl 99)\n", fam->model, g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0);
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    const int L_ = llm_layers(t);
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L || H_ != H) {
        init(t, llm_rope_node(g), L_, H_);
        bound = w0->data;
    }

    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view        kv(k_idxs, kqm, man.context, PG);
    std::vector<int32_t> pb;
    llm_positions(inp_pos, T, pb);
    if (!llm_one_sequence(kv, T, fam->model)) {
        return GGML_STATUS_FAILED;
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                                  xb.data() + (size_t) r * H, H);
        });
    }
    h_buf.resize((size_t) T * H);
    batch_rows = xb.data();
    out_h      = h_buf.data();
    axera_timer      tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const seq_status sst = seq->feed(kv.cell_of(0), T, (const uint16_t *) xb.data(), pb.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) llm_seq_log(sst, kv.cell_of(0), *seq, fam->model);
        return GGML_STATUS_FAILED;
    }

    // the kernel's output is already the final RMSNorm(x) * output_norm: copy it to the graph's embedding result.
    // A context with embeddings asks for every row; plain generation for the last one.
    if (result) {
        GGML_ASSERT(result->type == GGML_TYPE_F32 && result->ne[0] == H && (result->ne[1] == T || result->ne[1] == 1));
        const int64_t nres = std::min<int64_t>(T, result->ne[1]);
        for (int64_t r = 0; r < nres; ++r) {
            const int64_t src = result->ne[1] == T ? r : T - nres + r;
            memcpy((char *) result->data + (size_t) r * result->nb[1], h_buf.data() + (size_t) src * H,
                   (size_t) H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(result->buffer)) {
            ((ggml_backend_axera_buffer_context *) result->buffer->context)->flush(result->data, ggml_nbytes(result));
        }
    }
    if (logits && logits->ne[1] > 0 && !fam->speech_vocab) {
        GGML_LOG_WARN("axera %s: output logits are not part of this model's kernels\n", fam->model);
    } else if (logits && logits->ne[1] > 0) {
        const int64_t n = fam->speech_vocab, n_out = logits->ne[1];
        GGML_ASSERT(logits->type == GGML_TYPE_F32 && logits->ne[0] >= n);
        GGML_ASSERT(last_logits && (*last_logits)->size >= speech_pad * sizeof(float));
        bool text_tail = false;
        if (n_out == 1) {
            // the kernel computed the head of the last valid row (prefill reduces it inside the graph)
            memcpy(logits->data, (*last_logits)->vir, (size_t) n * sizeof(float));
        } else {
            ggml_tensor * out_ids = names.find("out_ids");
            if (!out_ids) {
                for (int i = 0; i < g->n_nodes; i++) {
                    ggml_tensor * node = g->nodes[i];
                    if (node->op == GGML_OP_GET_ROWS && node->src[1] && node->src[1]->type == GGML_TYPE_I32 &&
                        node->src[1]->ne[0] == n_out && node->src[0]->ne[1] == T) {
                        out_ids = node->src[1];
                    }
                }
            }
            // An embeddings context asks for every hidden row but its caller samples only the last logits row,
            // which the kernel already reduced; the head on all T rows would cost hundreds of milliseconds.
            const bool last_row_only = result && result->ne[1] == T;
            text_tail                = !last_row_only;
            for (int64_t r = 0; r < n_out; ++r) {
                const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : (int) r;
                GGML_ASSERT(row >= 0 && row < T);
                float * dst = (float *) ((char *) logits->data + r * logits->nb[1]);
                if (!last_row_only) {
                    host_speech_logits(logits->src[0], h_buf.data() + (size_t) row * H, dst, n);
                } else if (r + 1 == n_out) {
                    memcpy(dst, (*last_logits)->vir, (size_t) n * sizeof(float));
                } else {
                    memset(dst, 0, (size_t) n * sizeof(float));
                }
            }
        }
        // The TTS runtime samples the speech rows only and never observes the text vocabulary: a prefill does not
        // write ~74 MiB of -inf into that tail, only a logits-only caller that asks for it gets it.
        for (int64_t r = 0; text_tail && r < n_out; ++r) {
            float * o = (float *) ((char *) logits->data + r * logits->nb[1]);
            std::fill(o + n, o + logits->ne[0], -INFINITY);
        }
        if (ggml_backend_buffer_is_axera(logits->buffer)) {
            ((ggml_backend_axera_buffer_context *) logits->buffer->context)->flush(logits->data, ggml_nbytes(logits));
        }
    }
    return GGML_STATUS_SUCCESS;
}

seq_caps phase_dense::seq_get_caps() const {
    seq_caps c;
    c.page           = PG;
    c.width          = H;
    c.image_bytes    = KV ? KV->size : 0;
    c.cells          = man.context;
    c.verify_rows    = 0;
    c.decode_at_zero = true;
    return c;
}

void phase_dense::seq_state_save(uint8_t * dst) {
    axera_timer tl("state save");
    KV->invalidate(KV->vir, KV->size);
    memcpy(dst, KV->vir, KV->size);
}

void phase_dense::seq_state_load(const uint8_t * src) {
    axera_timer tl("state load");
    // A new sequence (src == nullptr) starts at cell 0: the kernels read only cells they have written, so clearing
    // the image would only flush cold cache lines.
    if (src) {
        memcpy(KV->vir, src, KV->size);
        KV->flush();
    }
}

bool phase_dense::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    const bool short_prompt = pre16.c.k && n <= pre16.rows;
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int64_t cell = c0 + r0;
        if (snap && snap->cell == cell) {
            KV->invalidate(KV->vir, KV->size);
            memcpy(snap->dst, KV->vir, KV->size);
            snap->deferred = false;
        }
        run_page(short_prompt ? pre16 : pre, cell, std::min(PG, n - r0), xb + (size_t) r0 * H, p4 + (size_t) r0 * 4, r0,
                 batch0, short_prompt ? "prefill (short)" : "prefill (page)");
    }
    return true;
}

bool phase_dense::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    decode(c, (const ggml_bf16_t *) x, p4, batch_row);
    return true;
}

void phase_dense::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

size_t phase_dense::state_size(uint32_t flags) const {
    return seq ? seq->size((flags & state_flags_partial_only) != 0) : 0;
}

size_t phase_dense::state_write(void * dst, size_t size, uint32_t flags) {
    return seq ? seq->write((uint8_t *) dst, size, (flags & state_flags_partial_only) != 0) : 0;
}

size_t phase_dense::state_read(const void * src, size_t size, uint32_t flags) {
    (void) flags;
    return seq ? seq->read((const uint8_t *) src, size) : 0;
}

}  // namespace

std::shared_ptr<model> make_phase_dense(const manifest & m) {
    return std::make_shared<phase_dense>(m);
}

}  // namespace axera
