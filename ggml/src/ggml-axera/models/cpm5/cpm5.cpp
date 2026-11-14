// Axera NPU backend: MiniCPM5-2B, a plain 42-layer Llama text decoder (GQA 16 / 2,
// head_dim 128, SwiGLU 6144, no Q/K RMSNorm, no vision, no MTP).
//
// One path, the manifest's kernels:
//   prefill  role prefill: the whole 42-layer backbone in one call per page of 256
//            cells.  The layers are unrolled at trace time (the 27B split-KV method)
//            and every layer owns one compact KV input b0_kv<j>; the weights are
//            42-row runtime banks.  The last page of a batch is padded and
//            valid_tokens masks its output rows.
//   decode   role decode: one token through the same 42 layers and banks.
//   lm_head  role lm_head: fused final RMSNorm + projection + greedy argmax.
//   lm_head.topk: fused projection + host penalty state + exact top-1024.
// The KV cache is this adapter's: 42 paged pools [pages][K, V][256][2][128] bf16.
// There is no recurrent state and no MTP; core/seqstate owns the restart points.

#include "../model.h"
#include "../../core/params.h"
#include "../../core/mx.h"
#include "../../core/quant.h"
#include "../../core/seqstate.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <vector>

namespace axera {
namespace {

constexpr int PG = 256;  // cells per page = prefill rows per call

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

// the adapter blob in llama.cpp's sequence state (context checkpoints / prompt cache):
// the KV pages before the current cell and core/seqstate's own restart-point blob.
struct cpm5_state_header {
    char     magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t partial;
    int64_t  state_cell;
    uint32_t nkv_pages;
    uint32_t nkv;
    uint64_t kv_page_b;
    uint64_t kv_n;
    uint64_t seq_n;
    uint32_t reserved;
};

class cpm5 final : public model, public seq_model {
  public:
    explicit cpm5(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size(uint32_t flags) const override;
    size_t       state_write(void * dst, size_t size, uint32_t flags) override;
    size_t       state_read(const void * src, size_t size, uint32_t flags) override;
    int          last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override { (void) dst; }  // KV-only: seqstate replays pages
    void     seq_state_load(const uint8_t * src) override { (void) src; }
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override;

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, NBLK = 0, AX = 1;
    int              kv_pages = 0;
    int64_t          kv_cells = 0;
    size_t           kv_b = 0, kv_alloc = 0;

    std::vector<axera_cmm>           KVC;  // one pool per layer (42 compact IOs)
    std::map<std::string, axera_cmm> banks;
    call                             pre, dec;
    axera_cmm                        PX, POUT, POS, VALID, INDPTR, INDICES, LASTLEN, CLOC;
    axera_cmm                        DX, DOUT, DPOS;
    const ggml_bf16_t *              batch_rows = nullptr;
    const kv_view *                  batch_kv   = nullptr;
    const rope *                     batch_rope = nullptr;
    float *                          out_h      = nullptr;
    std::vector<ggml_bf16_t>         src_rows;
    std::vector<float>               h_buf;
    std::unique_ptr<seqstate>        seq;
    int                              last_seq_status = 0;
    output_head                      head;

    void      init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    void      paging(int64_t hist, int64_t write_cell);
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    void      decode(int64_t c, int t, const ggml_bf16_t * x, float * h);
};

// a [rows] or [rows][cols] f32 view of any small GGUF tensor
std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) {
            memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float));
        });
    }
    return v;
}

axera_cmm cpm5::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp) {
    static const std::regex re("b0_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera cpm5: prefill: unexpected input %s\n", d.name.c_str());
    }
    const bool        mlp = mt[1] == "mlp";
    const std::string n   = mt[2];
    const int         nb  = (int) d.shape[0];  // 42 blocks; cos / sin are one shared table
    GGML_ASSERT(nb > 0 && d.bytes % nb == 0);
    axera_cmm         c  = axera_cmm_alloc(d.bytes);
    const size_t      bb = d.bytes / nb;
    const bool        keep_pages = axera_keep_pages();
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera cpm5: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    for (int b = 0; b < nb; ++b) {
        axera_mapped_drop_guard drop(read, !keep_pages);
        const int il  = b;
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
        auto fill = [&](float v) {
            put(std::vector<float>(bb / (d.dtype == 3 ? sizeof(float) : sizeof(ggml_bf16_t)), v));
        };
        auto mx8 = [&](const char * g, bool scales) {
            const ggml_tensor * w = T(il, g);
            const int64_t       K = w->ne[0];
            GGML_ASSERT((size_t) ggml_nrows(w) * (scales ? K / AXERA_MX_GROUP : K) == bb);
            // llama.cpp's GGUF stores q/k in the interleaved RoPE layout
            // (convert_hf_to_gguf's permute()); the kernels rotate the HF
            // half-split layout, so map every GGUF row back to its HF output
            // channel.  v/o are not permuted.
            int64_t n_head = 0;
            if (strcmp(g, "attn_q.weight") == 0) {
                n_head = 16;
            } else if (strcmp(g, "attn_k.weight") == 0) {
                n_head = 2;
            }
            const int64_t m = n_head ? w->ne[1] / n_head / 2 : 0;  // head_dim / 2
            auto          row_hf = [&](int64_t r) -> int64_t {
                if (!n_head) {
                    return r;
                }
                const int64_t h = r / (2 * m), rem = r % (2 * m);
                return (h * 2 + rem % 2) * m + rem / 2;
            };
            axera_for_rows_f32(w, [&](int64_t r, const float * row) {
                thread_local std::vector<uint8_t> q, s;
                q.resize(K);
                s.resize(K / AXERA_MX_GROUP);
                axera_quantize_row_mxfp8_f32(row, q.data(), s.data(), K);
                const int64_t rh = row_hf(r);
                if (scales) {
                    memcpy(dst + rh * (K / AXERA_MX_GROUP), s.data(), s.size());
                } else {
                    memcpy(dst + rh * K, q.data(), q.size());
                }
            });
        };
        if (n == "cos" || n == "sin") {
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT(d.dtype == 3 && (size_t) R == rp.inv_freq.size());
            for (int64_t q = 0; q < P; ++q) {
                for (int64_t i = 0; i < R; ++i) {
                    const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
                    ((float *) dst)[q * R + i] = (float) (n == "cos" ? std::cos(th) : std::sin(th));
                }
            }
        } else if (n.rfind("w_scale2_", 0) == 0) {
            fill(1.0f);
        } else if (n.rfind("w_zeropoint_", 0) == 0 || n.rfind("w_zp_scale", 0) == 0 || n == "input_permutation") {
            memset(dst, 0, bb);
        } else if (mlp) {
            static const std::map<std::string, const char *> W = {
                { "1", "ffn_gate.weight" }, { "2", "ffn_up.weight" }, { "3", "ffn_down.weight" },
            };
            if (n == "w_norm") {
                put(f32_of(T(il, "ffn_norm.weight")));
            } else if (n.size() == 2 && W.count(n.substr(1))) {
                mx8(W.at(n.substr(1)), false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                mx8(W.at(n.substr(7)), true);
            } else {
                GGML_ABORT("axera cpm5: prefill: unexpected MLP input %s\n", d.name.c_str());
            }
        } else {
            static const std::map<std::string, const char *> W = {
                { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                { "v", "attn_v.weight" }, { "o", "attn_output.weight" },
            };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                mx8(W.at(p), n != "w_" + p);
            } else {
                GGML_ABORT("axera cpm5: prefill: unexpected attention input %s\n", d.name.c_str());
            }
        }
    }
    c->flush();
    if (keep_pages) {
        for (const ggml_tensor * w : read) {
            axera_drop_mapped(w);
        }
    }
    return c;
}

void cpm5::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    L_ = 0;
    GGML_ASSERT(H_ == 2048 && L == 42);
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");

    std::vector<const io_desc *> kvd;
    for (const auto & d : pk.desc->in) {
        if (d.name.rfind("b0_kv", 0) == 0 && d.name.size() > 5 && isdigit((unsigned char) d.name[5])) {
            kvd.push_back(&d);
        }
    }
    std::sort(kvd.begin(), kvd.end(), [](const io_desc * a, const io_desc * b) {
        return atoi(a->name.c_str() + 5) < atoi(b->name.c_str() + 5);
    });
    GGML_ASSERT((int) kvd.size() == L);
    const io_desc & kv0 = *kvd[0];
    GGML_ASSERT(kv0.shape.size() == 5 && kv0.shape[2] == PG && kv0.shape[3] == 2 && kv0.shape[4] == 128);
    NBLK = (int) kv0.shape[0];  // [pages][K,V][256][2 heads][128]
    GGML_ASSERT((int64_t) NBLK * PG == man.context);
    kv_b     = kv0.bytes;
    kv_pages = NBLK;
    if (const int64_t cells = axera_kv_cells(); cells > 0) {
        kv_pages = (int) std::max<int64_t>(1, std::min<int64_t>(NBLK, (cells + PG - 1) / PG));
    }
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        kv_pages = std::max(1, std::min(NBLK, atoi(e)));
    }
    kv_cells = (int64_t) kv_pages * PG;
    kv_alloc = kv_b / NBLK * kv_pages;
    if (kv_pages < NBLK) {
        GGML_LOG_INFO("axera cpm5: the KV pools hold %d of the kernels' %d pages (%lld cells)\n",
                      kv_pages, NBLK, (long long) kv_cells);
    }
    KVC.resize(L);
    for (auto & k : KVC) {
        k = axera_cmm_alloc(kv_alloc);
    }

    AX = (int) (pk.in_bytes("position_ids") / (PG * sizeof(int32_t)));
    GGML_ASSERT(AX == 1 && pk.in_bytes("x") == (size_t) PG * H * sizeof(ggml_bf16_t));
    GGML_ASSERT(dk.in_bytes("x") == (size_t) H * sizeof(ggml_bf16_t) &&
                dk.in_bytes("position_ids") == sizeof(int32_t));

    PX      = axera_cmm_alloc(pk.in_bytes("x"));
    POUT    = axera_cmm_alloc(pk.out_bytes("output"));
    POS     = axera_cmm_alloc(pk.in_bytes("position_ids"));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    DX      = axera_cmm_alloc(dk.in_bytes("x"));
    DOUT    = axera_cmm_alloc(dk.out_bytes("output"));
    DPOS    = axera_cmm_alloc(sizeof(int32_t));
    GGML_ASSERT(pk.in_bytes("indices") == (size_t) NBLK * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    const rope rp(rope_node, names.find("inp_pos"), 1);
    GGML_ASSERT(rp.ROT == 128 && rp.inv_freq.size() == 64 && !rp.mrope && rp.freq_scale == 1.0f);

    banks.clear();
    for (const auto & d : pk.desc->in) {
        const std::string & n = d.name;
        if (n == "x" || n == "position_ids" || n == "indptr" || n == "indices" || n == "last_page_len" ||
            n == "cache_location" || n == "valid_tokens" || n.rfind("b0_kv", 0) == 0) {
            continue;
        }
        banks[n] = bank(t, d, rp);
    }

    auto kv_index = [&](const std::string & n) { return atoi(n.c_str() + 5); };
    auto bind_common = [&](const kernel & k, call & c) {
        c   = {};
        c.k = &k;
        for (const auto & d : k.desc->in) {
            const std::string & n = d.name;
            if (n.rfind("b0_kv", 0) == 0) {
                const int j = kv_index(n);
                GGML_ASSERT(j >= 0 && j < L);
                // The kernel's KV IO keeps its full manifest size; with a smaller -c the
                // CMM pool holds fewer pages and indices never address the rest (the
                // qwen38_27b / qwen35_whole split-KV layout).
                c.in.push_back(axera_io(KVC[j], (uint32_t) kv_b));
                continue;
            }
            const bool         prefill_call = c.k == &pk;
            const axera_cmm * b = n == "x" ? (prefill_call ? &PX : &DX) : n == "position_ids" ? (prefill_call ? &POS : &DPOS) :
                                  n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                  n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID : nullptr;
            if (!b) {
                auto it = banks.find(n);
                GGML_ASSERT(it != banks.end());
                b = &it->second;
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        c.out.push_back(axera_io(c.k == &pk ? POUT : DOUT, k.desc->out.at(0).bytes));
    };
    bind_common(pk, pre);
    bind_common(dk, dec);

    seq_config cfg;
    cfg.kept_max     = 1;
    cfg.verify_chain = 0;  // no MTP
    cfg.check        = [] {
        const char * e = getenv("GGML_AXERA_SEQ_CHECK");
        return e && atoi(e) != 0;
    }();
    cfg.force_miss = [] {
        const char * e = getenv("GGML_AXERA_SEQ_FORCE_MISS");
        return e ? std::max(0, atoi(e)) : 0;
    }();
    seq = std::make_unique<seqstate>(*this, cfg);

    size_t bsum = 0;
    for (const auto & [n, b] : banks) {
        bsum += b->size;
    }
    GGML_LOG_INFO("axera cpm5: %d layers, banks %.2f GiB, KV %.2f GiB (%d of %d pages, %lld cells)\n",
                  L, bsum / 1073741824.0, (double) kv_alloc * L / 1073741824.0, kv_pages, NBLK, (long long) kv_cells);
}

void cpm5::paging(int64_t hist, int64_t write_cell) {
    const int64_t pages          = (hist + PG - 1) / PG;
    ((int32_t *) INDPTR->vir)[0] = 0;
    ((int32_t *) INDPTR->vir)[1] = (int32_t) pages;
    *(int32_t *) LASTLEN->vir    = (int32_t) (pages ? hist - (pages - 1) * PG : 0);
    ((int32_t *) CLOC->vir)[0]   = (int32_t) (write_cell / PG);
    ((int32_t *) CLOC->vir)[1]   = (int32_t) (write_cell % PG);
    for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
        (*b)->flush();
    }
}

void cpm5::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0) {
    GGML_ASSERT(c0 % PG == 0 && c0 >= 0 && n >= 0 && c0 + n <= kv_cells);
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) PX->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        PX->flush();
        auto * pp = (int32_t *) POS->vir;
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
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT->vir + (size_t) r * H,
                                  h.data() + (size_t) (r0 + r - out0) * H, H);
        });
    }
}

void cpm5::decode(int64_t c, int t, const ggml_bf16_t * x, float * h) {
    (void) t;
    memcpy(DX->vir, x, H * sizeof(ggml_bf16_t));
    DX->flush(DX->vir, H * sizeof(ggml_bf16_t));
    *(int32_t *) DPOS->vir = (int32_t) c;
    DPOS->flush();
    paging(c, c);
    {
        axera_timer tl("layers (decode)");
        dec.k->run(dec.in, dec.out, "decode");
    }
    DOUT->invalidate(DOUT->vir, H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) DOUT->vir, h, H);
}

ggml_status cpm5::compute(ggml_cgraph * g) {
    last_seq_status = 0;
    const auto &  t       = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera cpm5: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
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
        axera_timer tc("input (cpu graph)");
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
    GGML_ASSERT(rope_node && "axera cpm5: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, man.context, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera cpm5: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera cpm5: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t *      pos = (const int32_t *) inp_pos->data;
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = pos[r];
        }
    }
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
    axera_timer          tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + r * x0->nb[1]),
                                  xb.data() + (size_t) r * H, H);
        });
    }
    batch_rows      = xb.data();
    batch_kv        = &kv;
    batch_rope      = nullptr;
    out_h           = h.data();
    last_seq_status = 0;
    const seq_status sst = seq->feed(kv.cell_of(0), T, (const uint16_t *) xb.data(), pb.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) sst;
        if (sst == SEQ_MISS) {
            GGML_LOG_WARN("axera cpm5: batch from cell %lld has no restart point (%s); the slot must replay\n",
                          (long long) kv.cell_of(0), seq->why());
        } else {
            GGML_LOG_ERROR("axera cpm5: sequence state rejected a batch from cell %lld: %s\n",
                           (long long) kv.cell_of(0), seq->why());
        }
        return GGML_STATUS_FAILED;
    }

    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    if (man.has("lm_head.topk")) {
        head.bind_topk(man.get("lm_head.topk"));
    }
    const bool combined = head.bind_combined(man.get("lm_head"), logits->src[0]) && head.combined_ready(g, names);
    if (combined) {
        head.run_combined(g, names, h, T, nullptr, nullptr);
    } else {
        head.run(g, names, h, T);
    }
    return GGML_STATUS_SUCCESS;
}

seq_caps cpm5::seq_get_caps() const {
    seq_caps c;
    c.page           = PG;
    c.width          = H;
    c.image_bytes    = 0;
    c.cells          = kv_cells;
    c.verify_rows    = 0;
    c.decode_at_zero = true;
    return c;
}

bool cpm5::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    (void) snap;  // KV needs no rewind: seqstate replays from the page's first cell
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    prefill(c0, n, xb, p4, h_buf, batch0);
    return true;
}

bool cpm5::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    (void) p4;
    const uint16_t * xr = x;
    if (!xr) {
        src_rows.resize(H);
        seq_rows(batch_row, 1, (uint16_t *) src_rows.data());
        xr = reinterpret_cast<const uint16_t *>(src_rows.data());
    }
    decode(c, 0, (const ggml_bf16_t *) xr, out_h);
    return true;
}

void cpm5::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

size_t cpm5::state_size(uint32_t flags) const {
    if (KVC.empty() || !seq) {
        return 0;
    }
    const bool   partial = flags & state_flags_partial_only;
    const size_t seq_n   = seq->size(partial);
    if (!seq_n) {
        return 0;
    }
    const int64_t state_cell = std::max<int64_t>(0, seq->cell());
    const size_t  nkv_pages  = partial ? 0 :
        std::min<size_t>((size_t) kv_pages, (size_t) ((state_cell + PG - 1) / PG));
    return sizeof(cpm5_state_header) + seq_n + (size_t) KVC.size() * nkv_pages * (kv_b / NBLK);
}

size_t cpm5::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    const size_t total = state_size(flags);
    if (!total || size < total) {
        return 0;
    }
    const bool   partial   = flags & state_flags_partial_only;
    const size_t seq_n     = seq->size(partial);
    const size_t nkv_pages = partial ? 0 :
        std::min<size_t>((size_t) kv_pages, (size_t) ((std::max<int64_t>(0, seq->cell()) + PG - 1) / PG));
    cpm5_state_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "CPM5S1", sizeof(h.magic) - 1);
    h.version    = 1;
    h.kind       = 1;
    h.partial    = partial;
    h.state_cell = seq->cell();
    h.nkv_pages  = partial ? 0 : (uint32_t) nkv_pages;
    h.nkv        = (uint32_t) KVC.size();
    h.kv_page_b  = kv_b / NBLK;
    h.kv_n       = partial ? 0 : (uint64_t) KVC.size() * nkv_pages * (kv_b / NBLK);
    h.seq_n      = seq_n;
    char * p     = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (!partial) {
        const size_t page_n = nkv_pages * (kv_b / NBLK);
        for (const axera_cmm & c : KVC) {
            c->invalidate(c->vir, page_n);
            memcpy(p, c->vir, page_n);
            p += page_n;
        }
    }
    if (seq->write((uint8_t *) p, seq_n, partial) != seq_n) {
        return 0;
    }
    p += seq_n;
    return (size_t) (p - (char *) dst);
}

size_t cpm5::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    if (size < sizeof(cpm5_state_header)) {
        return 0;
    }
    const bool partial = flags & state_flags_partial_only;
    cpm5_state_header h;
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, "CPM5S1", sizeof(h.magic) - 1) != 0 || h.version != 1 || h.kind != 1 ||
        h.partial != partial || !h.seq_n || h.nkv != KVC.size() || h.kv_page_b != kv_b / NBLK ||
        (partial ? h.nkv_pages != 0 || h.kv_n != 0 :
                   h.nkv_pages > (uint32_t) kv_pages ||
                       h.kv_n != (uint64_t) KVC.size() * h.nkv_pages * (kv_b / NBLK)) ||
        h.state_cell < -1 || h.state_cell > kv_cells) {
        return 0;
    }
    const size_t need = sizeof(h) + h.kv_n + h.seq_n;
    if (size < need) {
        return 0;
    }
    const char * p = (const char *) src + sizeof(h);
    if (!partial) {
        const size_t page_n = h.nkv_pages * (kv_b / NBLK);
        for (const axera_cmm & c : KVC) {
            memcpy(c->vir, p, page_n);
            c->flush(c->vir, page_n);
            p += page_n;
        }
    }
    if (seq->read((const uint8_t *) p, h.seq_n) != h.seq_n) {
        return 0;
    }
    p += h.seq_n;
    return (size_t) (p - (const char *) src);
}

}  // namespace

std::shared_ptr<model> make_cpm5(const manifest & m) {
    return std::make_shared<cpm5>(m);
}

}  // namespace axera
