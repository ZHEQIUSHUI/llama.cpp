// Axera NPU backend: Fun-CosyVoice3 LLM (Qwen2 dense decoder, 24 layers, H=896, FFN=4864,
// 14 Q / 2 KV heads, head_dim 64, Q/K/V projection bias, theta 1e6).
//
// Two whole-model stages, each one AXModel on four cores:
//   prefill  role prefill: up to 256 cells through all 24 layers (a runtime loop over six four-layer
//            phase banks), final RMSNorm included
//   decode   role decode: one cell through the same layers and the same paged BF16 KV cache
// The KV cache is this adapter's: [layers][pages][K, V][256][2][64] bf16.  Q is padded from 14 to
// 16 head slots so every core owns four contiguous heads; the output projection is padded to the same
// 1024 input columns.  The dummy slots carry zero weights and zero bias and contribute exactly zero.
// One sequence at a time, with core/seqstate owning restart points, rewinds and misses.

#include "../model.h"
#include "../../core/params.h"
#include "../../core/quant.h"
#include "../../core/seqstate.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <regex>

namespace axera {
namespace {

constexpr int PG = 256;

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

class cosyvoice3_llm final : public model, public seq_model {
  public:
    explicit cosyvoice3_llm(const manifest & m) : man(m) {}

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
    const manifest & man;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, NBLK = 0;

    axera_cmm                        KV;
    std::map<std::string, axera_cmm> banks;
    call                             pre, dec;
    axera_cmm PX, POS, VALID, POUT, INDPTR, INDICES, LASTLEN, CLOC;
    axera_cmm DX, DOUT;
    // The speech head rides inside both stage AXModels: the padded fp32 logits row and its top-25.
    axera_cmm         PLOGITS, PTOPKV, PTOPKI, DLOGITS, DTOPKV, DTOPKI;
    const axera_cmm * last_logits = nullptr;

    const ggml_bf16_t *      batch_rows = nullptr;
    std::vector<ggml_bf16_t> src_rows;
    std::vector<float>       h_buf;
    float *                  out_h = nullptr;

    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;
    int                       dump_prefill = 0, dump_decode = 0;

    void      init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d);
    void      paging(int64_t hist, int64_t write_cell);
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, int batch0, seq_snap * snap);
    void      decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row);
};

// Diagnostic only: COSYVOICE3_DUMP_PREFIX writes the real kernel I/O as raw bf16 files, so a board run can be compared
// with the CPU reference using exactly the inputs the NPU saw.  No arithmetic depends on it.
void dump_bin(const char * prefix, const char * tag, const void * data, size_t bytes) {
    if (!prefix || !*prefix || !data || !bytes) {
        return;
    }
    const std::string path = std::string(prefix) + "." + tag + ".bin";
    FILE *            f    = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, 1, bytes, f);
        fclose(f);
    }
}

void dump_meta(const char * prefix, const char * tag, int valid, int cell, int pos) {
    if (!prefix || !*prefix) {
        return;
    }
    const std::string path = std::string(prefix) + "." + tag + ".txt";
    FILE *            f    = fopen(path.c_str(), "w");
    if (f) {
        fprintf(f, "valid %d\ncell %d\npos %d\nhidden 2048\n", valid, cell, pos);
        fclose(f);
    }
}

constexpr int HEAD_DIM   = 64;
constexpr int SLOT_HEADS = 16;
constexpr int SPEECH_VOCAB = 6761;
constexpr int SPEECH_PAD   = 6784;
constexpr int SPEECH_TOPK  = 25;
constexpr int SLOT_HEAD[SLOT_HEADS] = { 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, -1 };

axera_cmm cosyvoice3_llm::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d) {
    if (d.name == "final_norm") {
        auto      v = f32_of(t.at("output_norm.weight"));
        axera_cmm c = axera_cmm_alloc_raw(d.bytes);
        GGML_ASSERT(v.size() * sizeof(ggml_bf16_t) == d.bytes);
        std::vector<ggml_bf16_t> host(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            host[i] = ggml_fp32_to_bf16(v[i]);
        }
        memcpy(c->vir, host.data(), host.size() * sizeof(ggml_bf16_t));
        c->flush();
        return c;
    }

    if (d.name == "speech_head") {
        // The head GGUF stores the 6761-row speech decoder in output.weight's first rows (the rest of the Qwen2
        // vocabulary is zero).  The kernel pads it to four equal core slices; the padded rows are zero and are
        // masked to -inf inside the graph before the local top-k.
        if (!t.count("output.weight")) {
            GGML_ABORT("axera cosyvoice3_llm: speech_head needs output.weight in the model\n");
        }
        const ggml_tensor * w = t.at("output.weight");
        GGML_ASSERT(w->ne[0] == H && w->ne[1] >= SPEECH_VOCAB && d.dtype == 9 && d.bytes == (size_t) SPEECH_PAD * H * 2);
        axera_cmm c = axera_cmm_alloc_raw(d.bytes);
        std::memset(c->vir, 0, c->size);
        const auto *        traits = ggml_get_type_traits(w->type);
        std::vector<float>  row(H);
        for (int64_t r = 0; r < SPEECH_VOCAB; ++r) {
            const void * src = (const char *) w->data + r * w->nb[1];
            if (w->type == GGML_TYPE_F32) {
                memcpy(row.data(), src, H * sizeof(float));
            } else {
                GGML_ASSERT(traits->to_float);
                traits->to_float(src, row.data(), H);
            }
            for (int64_t i = 0; i < H; ++i) {
                ((ggml_bf16_t *) c->vir)[r * H + i] = ggml_fp32_to_bf16(row[i]);
            }
        }
        c->flush();
        axera_drop_mapped(w);
        return c;
    }

    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera cosyvoice3_llm: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(nb > 0 && d.bytes % nb == 0);
    axera_cmm             c = axera_cmm_alloc_raw(d.bytes);
    std::vector<uint8_t> host(d.bytes);
    const size_t         bb = d.bytes / nb;
    const bool            keep_pages = axera_keep_pages();
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera cosyvoice3_llm: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        axera_mapped_drop_guard drop(read, !keep_pages);
        const int il  = 4 * b + phase;
        uint8_t * dst = host.data() + (size_t) b * bb;
        auto put = [&](const std::vector<float> & v) {
            GGML_ASSERT(v.size() * (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t)) == bb);
            for (size_t i = 0; i < v.size(); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v[i];
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i]);
                }
            }
        };
        auto fill = [&](float v) {
            const size_t e = bb / (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t));
            for (size_t i = 0; i < e; ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v);
                }
            }
        };
        // mode 0: rows as stored; 1: Q rows padded 14 -> 16 head slots; 2: O columns padded to 1024.
        auto mx8 = [&](const char * g, bool scales, int mode) {
            const ggml_tensor * w = T(il, g);
            const int64_t       K = mode == 2 ? 1024 : w->ne[0];
            const int64_t       N = mode == 1 ? 1024 : ggml_nrows(w);
            GGML_ASSERT((size_t) N * (scales ? K / AXERA_MX_GROUP : K) == bb);
            thread_local std::vector<float>   row, row_src;
            thread_local std::vector<uint8_t> q, s;
            row.resize(K);
            q.resize(K);
            s.resize(K / AXERA_MX_GROUP);
            const auto * traits = ggml_get_type_traits(w->type);
            auto read_row = [&](int64_t r) {
                const int64_t KR = w->ne[0];
                row_src.resize(KR);
                const void * src = (const char *) w->data + r * w->nb[1];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(row_src.data(), src, KR * sizeof(float));
                } else {
                    GGML_ASSERT(traits->to_float);
                    traits->to_float(src, row_src.data(), KR);
                }
            };
            for (int64_t r = 0; r < N; ++r) {
                if (mode == 1) {
                    const int64_t slot = r / HEAD_DIM, j = r % HEAD_DIM;
                    const int64_t head = SLOT_HEAD[slot];
                    if (head < 0) {
                        std::fill(row.begin(), row.end(), 0.0f);
                    } else {
                        read_row(head * HEAD_DIM + j);
                        memcpy(row.data(), row_src.data(), (size_t) K * sizeof(float));
                    }
                } else if (mode == 2) {
                    std::fill(row.begin(), row.end(), 0.0f);
                    read_row(r);
                    for (int64_t slot = 0; slot < SLOT_HEADS; ++slot) {
                        const int64_t head = SLOT_HEAD[slot];
                        if (head >= 0) {
                            memcpy(row.data() + slot * HEAD_DIM, row_src.data() + head * HEAD_DIM,
                                   HEAD_DIM * sizeof(float));
                        }
                    }
                } else {
                    read_row(r);
                    memcpy(row.data(), row_src.data(), (size_t) K * sizeof(float));
                }
                axera_quantize_row_mxfp8_f32(row.data(), q.data(), s.data(), K);
                if (scales) {
                    memcpy(dst + r * (K / AXERA_MX_GROUP), s.data(), s.size());
                } else {
                    memcpy(dst + r * K, q.data(), q.size());
                }
            }
        };
        auto bias = [&](const char * g, int64_t rows, bool q_pad) {
            const ggml_tensor * w = T(il, g);
            std::vector<float>  v((size_t) rows);
            const int64_t       KR = w->ne[0];
            if (w->type == GGML_TYPE_F32) {
                memcpy(v.data(), w->data, KR * sizeof(float));
            } else {
                const auto * traits = ggml_get_type_traits(w->type);
                GGML_ASSERT(traits->to_float);
                traits->to_float(w->data, v.data(), KR);
            }
            if (q_pad) {
                std::vector<float> out((size_t) rows, 0.0f);
                GGML_ASSERT(rows == SLOT_HEADS * HEAD_DIM);
                for (int64_t slot = 0; slot < SLOT_HEADS; ++slot) {
                    const int64_t head = SLOT_HEAD[slot];
                    if (head >= 0) {
                        memcpy(out.data() + slot * HEAD_DIM, v.data() + head * HEAD_DIM,
                               HEAD_DIM * sizeof(float));
                    }
                }
                v.swap(out);
            }
            put(v);
        };
        if (n == "cos" || n == "sin") {
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT(d.dtype == 3 && R == HEAD_DIM / 2);
            const double theta = 1.0e6;
            for (int64_t p = 0; p < P; ++p) {
                for (int64_t i = 0; i < R; ++i) {
                    const double inv = 1.0 / std::pow(theta, (double) (2 * i) / (double) HEAD_DIM);
                    const double th  = (double) p * inv;
                    ((float *) dst)[p * R + i] = (float) (n == "cos" ? std::cos(th) : std::sin(th));
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
                mx8(W.at(n.substr(1)), false, 0);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                mx8(W.at(n.substr(7)), true, 0);
            } else {
                GGML_ABORT("axera cosyvoice3_llm: unexpected MLP input %s\n", d.name.c_str());
            }
        } else {
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (n == "q_bias") {
                bias("attn_q.bias", SLOT_HEADS * HEAD_DIM, true);
            } else if (n == "k_bias" || n == "v_bias") {
                bias(n == "k_bias" ? "attn_k.bias" : "attn_v.bias", 2 * HEAD_DIM, false);
            } else {
                const std::string p = n.substr(n.rfind('_') + 1);
                if (!W.count(p)) {
                    GGML_ABORT("axera cosyvoice3_llm: unexpected attention input %s\n", d.name.c_str());
                }
                if (n == "w_" + p) {
                    mx8(W.at(p), false, p == "q" ? 1 : p == "o" ? 2 : 0);
                } else if (n == "w_scale_" + p) {
                    mx8(W.at(p), true, p == "q" ? 1 : p == "o" ? 2 : 0);
                } else {
                    GGML_ABORT("axera cosyvoice3_llm: unexpected attention input %s\n", d.name.c_str());
                }
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

void cosyvoice3_llm::init(const std::map<std::string, ggml_tensor *> & t, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");

    const io_desc * kv = nullptr;
    for (const auto & d : pk.desc->in) {
        if (d.name == "b0_kv") {
            kv = &d;
        }
    }
    GGML_ASSERT(kv && kv->shape[0] == L && kv->shape[3] == PG);
    NBLK = (int) kv->shape[1];
    GGML_ASSERT((int64_t) NBLK * PG == man.context && dk.in_bytes("b0_kv") == kv->bytes);
    GGML_ASSERT(pk.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t) &&
                dk.in_bytes("x") == (size_t) H * sizeof(ggml_bf16_t));

    KV      = axera_cmm_alloc(kv->bytes);
    std::memset(KV->vir, 0, KV->size);
    KV->flush();
    PX      = axera_cmm_alloc(pk.in_bytes("x"));
    POUT    = axera_cmm_alloc(pk.out_bytes("output"));
    POS     = axera_cmm_alloc(pk.in_bytes("position_ids"));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc((size_t) NBLK * sizeof(int32_t));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    DX      = axera_cmm_alloc(dk.in_bytes("x"));
    DOUT    = axera_cmm_alloc(dk.out_bytes("output"));
    PLOGITS = axera_cmm_alloc(pk.out_bytes("logits"));
    PTOPKV  = axera_cmm_alloc(pk.out_bytes("topk_values"));
    PTOPKI  = axera_cmm_alloc(pk.out_bytes("topk_ids"));
    DLOGITS = axera_cmm_alloc(dk.out_bytes("logits"));
    DTOPKV  = axera_cmm_alloc(dk.out_bytes("topk_values"));
    DTOPKI  = axera_cmm_alloc(dk.out_bytes("topk_ids"));
    GGML_ASSERT(pk.out_bytes("logits") == (uint32_t) SPEECH_PAD * sizeof(float) &&
                dk.out_bytes("logits") == (uint32_t) SPEECH_PAD * sizeof(float));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

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
    // Banks touch disjoint tensors: convert them in parallel (each into host memory, then one copy per bank).
    std::vector<axera_cmm> made(work.size());
    axera_host_pool().run((int64_t) work.size(), [&](int64_t i) { made[(size_t) i] = bank(t, *work[(size_t) i]); });
    for (size_t i = 0; i < work.size(); ++i) {
        banks[work[i]->name] = std::move(made[i]);
    }
    pre = {};
    pre.k = &pk;
    for (const auto & d : pk.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &PX : n == "position_ids" ? &POS : n == "valid_tokens" ? &VALID :
                                n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                n == "cache_location" ? &CLOC : n == "b0_kv" ? &KV : &banks.at(n);
        GGML_ASSERT(b->get() && (*b)->size >= d.bytes);
        pre.in.push_back(axera_io(*b, d.bytes));
    }
    pre.out.clear();
    for (const auto & d : pk.desc->out) {
        axera_cmm * b = d.name == "output" ? &POUT : d.name == "logits" ? &PLOGITS :
                        d.name == "topk_values" ? &PTOPKV : d.name == "topk_ids" ? &PTOPKI : nullptr;
        if (!b) {
            GGML_ABORT("axera cosyvoice3_llm: prefill: unexpected output %s\n", d.name.c_str());
        }
        pre.out.push_back(axera_io(*b, d.bytes));
    }

    dec = {};
    dec.k = &dk;
    for (const auto & d : dk.desc->in) {
        const std::string & n = d.name;
        const axera_cmm *   b = n == "x" ? &DX : n == "position_ids" ? &POS : n == "valid_tokens" ? &VALID :
                                n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                n == "cache_location" ? &CLOC : n == "b0_kv" ? &KV : &banks.at(n);
        GGML_ASSERT(b->get() && (*b)->size >= d.bytes);
        dec.in.push_back(axera_io(*b, d.bytes));
    }
    dec.out.clear();
    for (const auto & d : dk.desc->out) {
        axera_cmm * b = d.name == "output" ? &DOUT : d.name == "logits" ? &DLOGITS :
                        d.name == "topk_values" ? &DTOPKV : d.name == "topk_ids" ? &DTOPKI : nullptr;
        if (!b) {
            GGML_ABORT("axera cosyvoice3_llm: decode: unexpected output %s\n", d.name.c_str());
        }
        dec.out.push_back(axera_io(*b, d.bytes));
    }

    seq_config cfg = seq_config_env(1);
    seq = std::make_unique<seqstate>(*this, cfg);

    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera cosyvoice3_llm: %d layers, banks %.2f GiB, KV cache %.2f GiB, %lld cells\n", L,
                  bb / 1073741824.0, KV->size / 1073741824.0, (long long) man.context);
}

void cosyvoice3_llm::paging(int64_t hist, int64_t write_cell) {
    const int64_t pages          = (hist + PG - 1) / PG;
    ((int32_t *) INDPTR->vir)[0] = 0;
    ((int32_t *) INDPTR->vir)[1] = (int32_t) pages;
    *(int32_t *) LASTLEN->vir    = (int32_t) (pages ? hist - (pages - 1) * PG : 0);
    ((int32_t *) CLOC->vir)[0]   = (int32_t) (write_cell / PG);
    ((int32_t *) CLOC->vir)[1]   = (int32_t) (write_cell % PG);
    INDPTR->flush();
    LASTLEN->flush();
    CLOC->flush();
}

void cosyvoice3_llm::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    GGML_ASSERT(c0 % PG == 0 && c0 + n <= man.context);
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (snap && snap->cell == cell) {
            KV->invalidate(KV->vir, KV->size);
            memcpy(snap->dst, KV->vir, KV->size);
            snap->deferred = false;
        }
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) PX->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        PX->flush();
        int32_t * pp = (int32_t *) POS->vir;
        for (int r = 0; r < PG; ++r) {
            pp[r] = r < v ? p4[(size_t) (r0 + r) * 4] : 0;
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
        PLOGITS->invalidate(PLOGITS->vir, PLOGITS->size);
        PTOPKV->invalidate(PTOPKV->vir, PTOPKV->size);
        PTOPKI->invalidate(PTOPKI->vir, PTOPKI->size);
        last_logits = &PLOGITS;
        {
            const char * dp = getenv("COSYVOICE3_DUMP_PREFIX");
            if (dp && dump_prefill == 0) {
                dump_bin(dp, "prefill_in", x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
                dump_bin(dp, "prefill_out", POUT->vir, (size_t) v * H * sizeof(ggml_bf16_t));
                dump_meta(dp, "prefill", v, (int) cell, 0);
                ++dump_prefill;
            }
        }
        for (int r = 0; r < v; ++r) {
            const int global = r0 + r;
            if (global >= batch0 && out_h) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT->vir + (size_t) r * H,
                                      out_h + (size_t) (global - batch0) * H, H);
            }
        }
    }
}

void cosyvoice3_llm::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row) {
    memcpy(DX->vir, x, (size_t) H * sizeof(ggml_bf16_t));
    DX->flush();
    *(int32_t *) POS->vir = p4[0];
    POS->flush();
    paging(c, c);
    {
        axera_timer tl("decode");
        dec.k->run(dec.in, dec.out, "decode");
    }
    DOUT->invalidate(DOUT->vir, (size_t) H * sizeof(ggml_bf16_t));
    DLOGITS->invalidate(DLOGITS->vir, DLOGITS->size);
    DTOPKV->invalidate(DTOPKV->vir, DTOPKV->size);
    DTOPKI->invalidate(DTOPKI->vir, DTOPKI->size);
    last_logits = &DLOGITS;
    {
        const char * dp = getenv("COSYVOICE3_DUMP_PREFIX");
        if (dp && dump_decode < 3) {
            const std::string tag = "decode" + std::to_string(dump_decode);
            dump_bin(dp, (tag + "_in").c_str(), DX->vir, (size_t) H * sizeof(ggml_bf16_t));
            dump_bin(dp, (tag + "_out").c_str(), DOUT->vir, (size_t) H * sizeof(ggml_bf16_t));
            dump_meta(dp, tag.c_str(), 1, (int) c, p4[0]);
            ++dump_decode;
        }
    }
    if (out_h) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) DOUT->vir, out_h + (size_t) batch_row * H, H);
    }
}

// Host fallback for a graph that asks for more than the one output row the stage AXModels produce (e.g. a caller
// requesting a whole prompt).  The speech head's first 6761 rows live in output.weight; the rest of the Qwen2
// vocabulary is not part of this model.
static void host_speech_logits(const ggml_tensor * w, const float * h, float * out, int64_t n) {
    const int64_t K      = w->ne[0];
    const auto *  traits = ggml_get_type_traits(w->type);
    GGML_ASSERT(w->type == GGML_TYPE_F32 || traits->to_float);
    axera_host_pool().run(n, [&](int64_t r) {
        thread_local std::vector<float> row;
        row.resize(K);
        const void * src = (const char *) w->data + r * w->nb[1];
        if (w->type == GGML_TYPE_F32) {
            memcpy(row.data(), src, K * sizeof(float));
        } else {
            traits->to_float(src, row.data(), K);
        }
        float acc = 0.0f;
        for (int64_t i = 0; i < K; ++i) {
            acc += row[i] * h[i];
        }
        out[r] = acc;
    });
}

ggml_status cosyvoice3_llm::compute(ggml_cgraph * g) {
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
        GGML_LOG_ERROR("axera cosyvoice3_llm: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    int n0 = 0;
    if (x0->src[0]->op != GGML_OP_NONE) {
        while (n0 < g->n_nodes && g->nodes[n0] != x0) {
            ++n0;
        }
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0 = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = llm_layers(t);
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L || H_ != H) {
        init(t, L_, H_);
        bound = w0->data;
    }

    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view   kv(k_idxs, kqm, man.context, PG);
    std::vector<int32_t> pb;
    llm_positions(inp_pos, T, pb);
    if (!llm_one_sequence(kv, T, "cosyvoice3_llm")) {
        return GGML_STATUS_FAILED;
    }

    const auto probe_t0 = std::chrono::steady_clock::now();
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                                  xb.data() + (size_t) r * H, H);
        });
    }
    h_buf.resize((size_t) T * H);
    batch_rows      = xb.data();
    out_h           = h_buf.data();
    last_seq_status = 0;
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const auto probe_t1 = std::chrono::steady_clock::now();
    const seq_status sst = seq->feed(kv.cell_of(0), T, (const uint16_t *) xb.data(), pb.data());
    const auto probe_t2 = std::chrono::steady_clock::now();
    if (sst != SEQ_OK) {
        last_seq_status = (int) llm_seq_log(sst, kv.cell_of(0), *seq, "cosyvoice3_llm");
        return GGML_STATUS_FAILED;
    }

    const auto probe_t3 = std::chrono::steady_clock::now();
    // The kernel's output is already final RMSNorm(x) * output_norm: copy it to the graph's embedding result.
    if (result) {
        // A context with embeddings asks for every row.  Plain text generation asks only for the last row,
        // so result_norm is one row and it is the batch's last output.
        GGML_ASSERT(result->type == GGML_TYPE_F32 && result->ne[0] == H);
        const int64_t nres = std::min<int64_t>(T, result->ne[1]);
        GGML_ASSERT(result->ne[1] == T || result->ne[1] == 1);
        for (int64_t r = 0; r < nres; ++r) {
            const int64_t src_row = result->ne[1] == T ? r : T - nres + r;
            memcpy((char *) result->data + (size_t) r * result->nb[1], h_buf.data() + (size_t) src_row * H,
                   (size_t) H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(result->buffer)) {
            ((ggml_backend_axera_buffer_context *) result->buffer->context)->flush(result->data, ggml_nbytes(result));
        }
    }
    if (logits && logits->ne[1] > 0) {
        GGML_ASSERT(logits->type == GGML_TYPE_F32 && logits->ne[0] >= SPEECH_VOCAB);
        const int64_t n_out = logits->ne[1];
        const int64_t n     = SPEECH_VOCAB;
        bool text_tail_consumed = false;
        GGML_ASSERT(last_logits && (*last_logits)->size >= (size_t) SPEECH_PAD * sizeof(float));
        if (n_out == 1) {
            text_tail_consumed = false;
            // The stage AXModel computed the head of the last valid row (prefill reduces it inside the graph).
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
            // An embeddings context asks for every hidden row but its caller samples only the last
            // logits row.  The stage AXModel already reduced the head of the last valid row, so
            // running the 6761-row head on all T rows here costs hundreds of milliseconds for a
            // prefill and is never observed.
            const bool last_row_only = result && result->ne[1] == T;
            text_tail_consumed = !last_row_only;
            const ggml_tensor * w = logits->src[0];
            for (int64_t r = 0; r < n_out; ++r) {
                const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : (int) r;
                GGML_ASSERT(row >= 0 && row < T);
                float * dst = (float *) ((char *) logits->data + r * logits->nb[1]);
                if (last_row_only) {
                    if (r + 1 == n_out) {
                        memcpy(dst, (*last_logits)->vir, (size_t) n * sizeof(float));
                    } else {
                        memset(dst, 0, (size_t) n * sizeof(float));
                    }
                } else {
                    host_speech_logits(w, h_buf.data() + (size_t) row * H, dst, n);
                }
            }
        }
        // The stage models expose a speech-only head.  The TTS runtime samples
        // the first 6761 values and never observes the Qwen2 text vocabulary,
        // so do not spend a prefill writing ~74 MiB of -inf into that tail.
        // A logits-only caller that explicitly asks for the text tail still
        // gets the finite-but-never-sampled values.
        if (text_tail_consumed) {
            for (int64_t r = 0; r < n_out; ++r) {
                float * o = (float *) ((char *) logits->data + r * logits->nb[1]);
                for (int64_t i = n; i < logits->ne[0]; ++i) {
                    o[i] = -INFINITY;
                }
            }
        }
        if (ggml_backend_buffer_is_axera(logits->buffer)) {
            ((ggml_backend_axera_buffer_context *) logits->buffer->context)->flush(logits->data, ggml_nbytes(logits));
        }
    }
        if (T > 1 && std::getenv("COSY_LLM_PROBE")) {
        const auto probe_t4 = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        fprintf(stderr, "[llm-probe] nodes=%d cpu_nodes=%d T=%d prep=%.3f feed=%.3f out=%.3f total=%.3f\n",
                g->n_nodes, n0, T, ms(probe_t0, probe_t1), ms(probe_t1, probe_t2),
                ms(probe_t3, probe_t4), ms(probe_t0, probe_t4));
    }
return GGML_STATUS_SUCCESS;
}

seq_caps cosyvoice3_llm::seq_get_caps() const {
    seq_caps c;
    c.page        = PG;
    c.width       = H;
    c.image_bytes = KV ? KV->size : 0;
    c.cells       = man.context;
    c.verify_rows = 0;
    c.decode_at_zero = true;
    return c;
}

void cosyvoice3_llm::seq_state_save(uint8_t * dst) {
    axera_timer tl("state save");
    KV->invalidate(KV->vir, KV->size);
    memcpy(dst, KV->vir, KV->size);
}

void cosyvoice3_llm::seq_state_load(const uint8_t * src) {
    axera_timer tl("state load");
    if (src) {
        memcpy(KV->vir, src, KV->size);
        KV->flush();
    }
    // A new sequence starts at cell 0.  Causal paging only reads cells that
    // the prefill/decode kernels have already written, so clearing the 48 MiB
    // image here would only flush cold cache lines and cost ~0.2 s per request.
}

bool cosyvoice3_llm::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    prefill(c0, n, xb, p4, batch0, snap);
    return true;
}

bool cosyvoice3_llm::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    decode(c, (const ggml_bf16_t *) x, p4, batch_row);
    return true;
}

void cosyvoice3_llm::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

size_t cosyvoice3_llm::state_size(uint32_t flags) const {
    return seq ? seq->size((flags & state_flags_partial_only) != 0) : 0;
}

size_t cosyvoice3_llm::state_write(void * dst, size_t size, uint32_t flags) {
    return seq ? seq->write((uint8_t *) dst, size, (flags & state_flags_partial_only) != 0) : 0;
}

size_t cosyvoice3_llm::state_read(const void * src, size_t size, uint32_t flags) {
    (void) flags;
    return seq ? seq->read((const uint8_t *) src, size) : 0;
}

}  // namespace

std::shared_ptr<model> make_cosyvoice3_llm(const manifest & m) {
    return std::make_shared<cosyvoice3_llm>(m);
}

}  // namespace axera
