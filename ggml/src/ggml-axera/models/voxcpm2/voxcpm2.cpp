// Axera NPU backend: VoxCPM2 BaseLM (MiniCPM dense decoder, 28 layers, H=2048, FFN=6144, 16 Q / 2 KV heads).
//
// The manifest has two whole-model stages, each one AXModel on four cores:
//   prefill  role prefill: 256 cells through all 28 layers (a runtime loop over seven four-layer phases whose weights
//            are banked [7, ...]), final RMSNorm included
//   decode   role decode: one cell through the same layers and the same paged BF16 KV cache, final RMSNorm included
// The KV cache is this adapter's: [layers][pages][K, V][256][2][128] bf16. The GGUF's Q/K rows are in llama.cpp's
// adjacent-pair RoPE order; the kernel uses MiniCPM's official half-split rotation, so the adapter undoes that row
// permutation while converting the weights to MXFP8. One sequence at a time, with core/seqstate owning restart points,
// rewinds and misses.

#include "../model.h"
#include "../../core/params.h"
#include "../../core/quant.h"
#include "../../core/seqstate.h"
#include "../../ops/ops.h"

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

class voxcpm2_base final : public model, public seq_model {
  public:
    explicit voxcpm2_base(const manifest & m) : man(m) {}

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

    const ggml_bf16_t *      batch_rows = nullptr;
    std::vector<ggml_bf16_t> src_rows;
    std::vector<float>       h_buf;
    float *                  out_h = nullptr;

    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;
    int                       dump_prefill = 0, dump_decode = 0;

    void      init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp);
    void      paging(int64_t hist, int64_t write_cell);
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, int batch0, seq_snap * snap);
    void      decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row);
};

// Diagnostic only: VOXCPM2_DUMP_PREFIX writes the real kernel I/O as raw bf16 files, so a board run can be compared
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

// row r of MiniCPM's half-split RoPE order -> the GGUF's adjacent-pair order (conversion/minicpm.py's inverse)
int64_t qk_source_row(int64_t r, int64_t rows, int heads) {
    const int64_t per = rows / heads;  // head dimension
    const int64_t h   = r / per;
    const int64_t j   = r % per;
    const int64_t half = j / (per / 2);
    const int64_t k    = j % (per / 2);
    return h * per + 2 * k + half;
}

axera_cmm voxcpm2_base::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const rope & rp) {
    if (d.name == "final_norm") {
        auto           v = f32_of(t.at("output_norm.weight"));
        axera_cmm      c = axera_cmm_alloc_raw(d.bytes);
        std::vector<ggml_bf16_t> host(v.size());
        GGML_ASSERT(v.size() * sizeof(ggml_bf16_t) == d.bytes);
        for (size_t i = 0; i < v.size(); ++i) {
            host[i] = ggml_fp32_to_bf16(v[i]);
        }
        memcpy(c->vir, host.data(), host.size() * sizeof(ggml_bf16_t));
        c->flush();
        return c;
    }

    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera voxcpm2: prefill: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const int         nb    = (int) d.shape[0];
    GGML_ASSERT(nb > 0 && d.bytes % nb == 0);
    axera_cmm    c  = axera_cmm_alloc_raw(d.bytes);
    // Convert into ordinary memory first: the CMM mapping is cached, but a byte-wise
    // store stream through it is much slower than one contiguous copy.  (#13)
    std::vector<uint8_t> host(d.bytes);
    const size_t bb = d.bytes / nb;
    const bool   keep_pages = axera_keep_pages();
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera voxcpm2: %s not in the graph\n", nm.c_str());
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
        auto mx8 = [&](const char * g, bool scales, int heads) {
            const ggml_tensor * w = T(il, g);
            const int64_t       K = w->ne[0], N = ggml_nrows(w);
            GGML_ASSERT((size_t) N * (scales ? K / AXERA_MX_GROUP : K) == bb);
            thread_local std::vector<float>   row;
            thread_local std::vector<uint8_t> q, s;
            row.resize(K);
            q.resize(K);
            s.resize(K / AXERA_MX_GROUP);
            const auto * traits = ggml_get_type_traits(w->type);
            for (int64_t r = 0; r < N; ++r) {
                const int64_t src_r = heads > 0 ? qk_source_row(r, N, heads) : r;
                const void *  src   = (const char *) w->data + src_r * w->nb[1];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, K * sizeof(float));
                } else {
                    GGML_ASSERT(traits->to_float);
                    traits->to_float(src, row.data(), K);
                }
                axera_quantize_row_mxfp8_f32(row.data(), q.data(), s.data(), K);
                if (scales) {
                    memcpy(dst + r * (K / AXERA_MX_GROUP), s.data(), s.size());
                } else {
                    memcpy(dst + r * K, q.data(), q.size());
                }
            }
        };
        if (n == "cos" || n == "sin") {
            const int64_t P = d.shape[1], R = d.shape[2];
            GGML_ASSERT(d.dtype == 3 && (size_t) R == rp.inv_freq.size());
            for (int64_t p = 0; p < P; ++p) {
                for (int64_t i = 0; i < R; ++i) {
                    const double th = (double) p * rp.freq_scale * rp.inv_freq[(size_t) i];
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
                mx8(W.at(n.substr(1)), false, 0);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                mx8(W.at(n.substr(7)), true, 0);
            } else {
                GGML_ABORT("axera voxcpm2: prefill: unexpected MLP input %s\n", d.name.c_str());
            }
        } else {
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                put(f32_of(T(il, "attn_norm.weight")));
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                const int heads = p == "q" ? 16 : p == "k" ? 2 : 0;
                mx8(W.at(p), n != "w_" + p, heads);
            } else {
                GGML_ABORT("axera voxcpm2: prefill: unexpected attention input %s\n", d.name.c_str());
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

void voxcpm2_base::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
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
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    const rope  rp(rope_node, names.find("inp_pos"), 1);
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
    axera_host_pool().run((int64_t) work.size(), [&](int64_t i) { made[(size_t) i] = bank(t, *work[(size_t) i], rp); });
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
    GGML_ASSERT(pk.desc->out.size() == 1 && pk.desc->out[0].name == "output");
    pre.out.push_back(axera_io(POUT, pk.desc->out[0].bytes));

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
    GGML_ASSERT(dk.desc->out.size() == 1 && dk.desc->out[0].name == "output");
    dec.out.push_back(axera_io(DOUT, dk.desc->out[0].bytes));

    seq_config cfg;
    cfg.kept_max = 1;
    cfg.check = [] {
        const char * e = getenv("GGML_AXERA_SEQ_CHECK");
        return e && atoi(e) != 0;
    }();
    cfg.force_miss = [] {
        const char * e = getenv("GGML_AXERA_SEQ_FORCE_MISS");
        return e ? std::max(0, atoi(e)) : 0;
    }();
    seq = std::make_unique<seqstate>(*this, cfg);

    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera voxcpm2_base: %d layers, banks %.2f GiB, KV cache %.2f GiB, %lld cells\n", L,
                  bb / 1073741824.0, KV->size / 1073741824.0, (long long) man.context);
}

void voxcpm2_base::paging(int64_t hist, int64_t write_cell) {
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

void voxcpm2_base::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
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
        {
            const char * dp = getenv("VOXCPM2_DUMP_PREFIX");
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

void voxcpm2_base::decode(int64_t c, const ggml_bf16_t * x, const int32_t * p4, int batch_row) {
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
    {
        const char * dp = getenv("VOXCPM2_DUMP_PREFIX");
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

ggml_status voxcpm2_base::compute(ggml_cgraph * g) {
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
        GGML_LOG_ERROR("axera voxcpm2_base: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
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
    x0 = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
        ++L_;
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; ++i) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera voxcpm2_base: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L || H_ != H) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }

    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view   kv(k_idxs, kqm, man.context, PG);
    const int32_t * pos = (const int32_t *) inp_pos->data;
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = pos[r];
        }
    }
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera voxcpm2_base: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera voxcpm2_base: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
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
    batch_rows      = xb.data();
    out_h           = h_buf.data();
    last_seq_status = 0;
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    const seq_status sst = seq->feed(kv.cell_of(0), T, (const uint16_t *) xb.data(), pb.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) sst;
        if (sst == SEQ_MISS) {
            GGML_LOG_WARN("axera voxcpm2_base: batch from cell %lld has no restart point (%s); the slot must replay\n",
                          (long long) kv.cell_of(0), seq->why());
        } else {
            GGML_LOG_ERROR("axera voxcpm2_base: sequence state rejected a batch from cell %lld: %s\n",
                           (long long) kv.cell_of(0), seq->why());
        }
        return GGML_STATUS_FAILED;
    }

    // The kernel's output is already final RMSNorm(x) * output_norm: copy it to the graph's embedding result.
    if (result) {
        GGML_ASSERT(result->type == GGML_TYPE_F32 && result->ne[0] == H && result->ne[1] == T);
        for (int r = 0; r < T; ++r) {
            memcpy((char *) result->data + (size_t) r * result->nb[1], h_buf.data() + (size_t) r * H,
                   (size_t) H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(result->buffer)) {
            ((ggml_backend_axera_buffer_context *) result->buffer->context)->flush(result->data, ggml_nbytes(result));
        }
    }
    if (logits) {
        GGML_LOG_WARN("axera voxcpm2_base: output logits are not part of the VoxCPM2 BaseLM TTS path\n");
    }
    return GGML_STATUS_SUCCESS;
}

seq_caps voxcpm2_base::seq_get_caps() const {
    seq_caps c;
    c.page        = PG;
    c.width       = H;
    c.image_bytes = KV ? KV->size : 0;
    c.cells       = man.context;
    c.verify_rows = 0;
    c.decode_at_zero = true;
    return c;
}

void voxcpm2_base::seq_state_save(uint8_t * dst) {
    axera_timer tl("state save");
    KV->invalidate(KV->vir, KV->size);
    memcpy(dst, KV->vir, KV->size);
}

void voxcpm2_base::seq_state_load(const uint8_t * src) {
    axera_timer tl("state load");
    if (src) {
        memcpy(KV->vir, src, KV->size);
        KV->flush();
    } else {
        memset(KV->vir, 0, KV->size);
        KV->flush();
    }
}

bool voxcpm2_base::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
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

bool voxcpm2_base::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    decode(c, (const ggml_bf16_t *) x, p4, batch_row);
    return true;
}

void voxcpm2_base::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

size_t voxcpm2_base::state_size(uint32_t flags) const {
    return seq ? seq->size((flags & state_flags_partial_only) != 0) : 0;
}

size_t voxcpm2_base::state_write(void * dst, size_t size, uint32_t flags) {
    return seq ? seq->write((uint8_t *) dst, size, (flags & state_flags_partial_only) != 0) : 0;
}

size_t voxcpm2_base::state_read(const void * src, size_t size, uint32_t flags) {
    (void) flags;
    return seq ? seq->read((const uint8_t *) src, size) : 0;
}

}  // namespace

std::shared_ptr<model> make_voxcpm2_base(const manifest & m) {
    return std::make_shared<voxcpm2_base>(m);
}

}  // namespace axera
