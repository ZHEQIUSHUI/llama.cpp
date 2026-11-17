// Axera NPU backend: Hy-MT2-30B-A3B (HYV3), a 48-layer GQA MoE translation model.
//
// The manifest has two merged kernels:
//   prefill  the whole model for one 256-row page, the 48 layers in a runtime loop over
//            contiguous weight banks, layer 0 dense and layers 1..47 MoE, plus the
//            final RMSNorm and NVFP4 output head for up to eight output rows.
//   decode   the same model for one token and one output row.
// The weights come from a bf16 GGUF quantized once to standard GGML NVFP4 blocks with
// one F32 .scale sidecar per quantization slice. Loading copies the codewords and the
// per-16-input UE4M3 scales straight into the kernels' pair-major banks; the sidecar is
// the kernels' FP32 scale2. No load-time requantization is done.
//
// KV is BF16, paged in 256-cell pages, [layers][513 pages][2][256][4][128]. The KV
// banks are kernel inputs (the kernels write them in place). seqstate owns page restart
// points and replay; this model has no recurrent state, so its state image is empty.

#include "../model.h"
#include "../../core/seqstate.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace axera {
namespace {

constexpr int PG = 256;
constexpr int H = 2048;
constexpr int HQ = 32;
constexpr int KVH = 4;
constexpr int HD = 128;
constexpr int E = 128;
constexpr int TOPK = 8;
constexpr int VOCAB = 120832;
constexpr int PAGES = 513;  // 131328 cells, enough for -c 131072
constexpr int HEAD_ROWS = 8;

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

struct group_info {
    int         first = 0;
    int         layers = 0;
    bool        dense = false;
    std::string pre;
};

struct state_header {
    char     magic[8];
    uint32_t version;
    uint32_t partial;
    uint32_t ngroups;
    uint32_t pad;
    int64_t  state_cell;
    uint32_t nkv_pages;
    uint32_t kv_page_b;
    uint64_t kv_n;
    uint64_t seq_n;
};

constexpr size_t state_magic_n = 8;

static std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) {
            memcpy(v.data() + (size_t) r * w->ne[0], row, (size_t) w->ne[0] * sizeof(float));
        });
    }
    return v;
}

class hy_mt2 final : public model, public seq_model {
  public:
    explicit hy_mt2(const manifest & m) : man(m) {}

    ggml_status compute(ggml_cgraph * g) override;
    size_t      state_size(uint32_t flags) const override;
    size_t      state_write(void * dst, size_t size, uint32_t flags) override;
    size_t      state_read(const void * src, size_t size, uint32_t flags) override;
    int         last_state_status() const override { return last_seq_status; }

    bool owns_kv_cache() const override { return true; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override { (void) dst; }
    void     seq_state_load(const uint8_t * src) override { (void) src; }
    bool seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void seq_rows(int batch_row0, int n, uint16_t * dst) override;

  private:
    const manifest & man;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H_ = 0, ROT = 0;
    int64_t          cells = 0;

    using bmap = std::map<std::string, axera_cmm>;
    bmap                                  gbank;   // head / RoPE tables
    bmap                                  dense;   // layer 0
    bmap                                  dense0;  // all-zero layer 0: an identity on calls 1..3
    std::vector<bmap>                     chunks;  // one 12-layer MoE bank per call
    std::map<std::string, axera_cmm>      io;
    std::map<std::string, uint32_t>       bank_n, bank_dtype, bank_slot;
    std::vector<group_info>               groups;
    int                                   per_chunk = 0, nchunks = 0;

    std::vector<call> pre, dec;
    std::vector<axera_cmm> route_cmm;  // one per call: that call's top-k expert ids
    FILE *                  route_file = nullptr;

    // batch input / output staging
    std::vector<uint16_t>    xbuf;
    std::vector<int32_t>     pbuf;
    const ggml_bf16_t *      batch_rows = nullptr;
    bool                     inited = false;
    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;

    // the output head computed by the last kernel call
    int head_first = 0, head_n = 0;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
              const ggml_tensor * inp_pos);
    void fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                   const ggml_tensor * inp_pos);
    void bind_calls(std::vector<call> & calls, const kernel & k, const char * tag, bool prefill);
    void run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                          bool need_head);
    void dump_hidden(const axera_cmm & b, int rows, int64_t cell, int call_index);
    void open_route_dump();
    void dump_routes(int64_t cell, int rows, int layer_stride);
    void run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4);
    void publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T);

    static const axera_cmm & at(const bmap & b, const std::string & name) {
        auto i = b.find(name);
        if (i == b.end()) {
            GGML_ABORT("axera hy_mt2: %s is not a bank of this call\n", name.c_str());
        }
        return i->second;
    }
    uint8_t * slot(const bmap & b, const std::string & name, int layer) const {
        auto i = b.find(name);
        auto s = bank_slot.find(name);
        if (i == b.end() || s == bank_slot.end()) {
            GGML_ABORT("axera hy_mt2: %s is not a layer bank\n", name.c_str());
        }
        return (uint8_t *) i->second->vir + (size_t) layer * s->second;
    }
};

static std::string tensor_name(int il, const char * suffix) {
    return "blk." + std::to_string(il) + "." + suffix;
}

static const ggml_tensor * need(const std::map<std::string, ggml_tensor *> & t, const std::string & n) {
    auto it = t.find(n);
    if (it == t.end()) {
        GGML_ABORT("axera hy_mt2: %s is not in the graph\n", n.c_str());
    }
    return it->second;
}

static const ggml_tensor * nv_scale(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * w) {
    std::string n = w->name;
    const std::string suffix = ".weight";
    if (n.size() >= suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
        n.resize(n.size() - suffix.size());
    }
    n += ".scale";
    auto it = t.find(n);
    if (it == t.end()) {
        GGML_ABORT("axera hy_mt2: NVFP4 tensor %s has no F32 .scale sidecar\n", w->name);
    }
    GGML_ASSERT(it->second->type == GGML_TYPE_F32 && ggml_nelements(it->second) >= 1);
    return it->second;
}

static bool io_name(const std::string & n) {
    return n == "x" || n == "position_ids" || n == "indptr" || n == "indices" || n == "last_page_len" ||
           n == "cache_location" || n == "head_x" || n == "head_rows" || n == "head_run" ||
           n == "head_logits" || n == "head_ids" ||
           n == "head_topk_v" || n == "head_topk_i" || n == "head_vmax" || n == "head_sumexp" ||
           n == "valid_tokens";
}

static bool starts_with(const std::string & n, const char * p) {
    const size_t k = strlen(p);
    return n.size() >= k && n.compare(0, k, p) == 0;
}

void hy_mt2::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                  const ggml_tensor * inp_pos) {
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");
    // The decode graph has no valid_tokens input (T=1); every other input name
    // is shared, so the two IO tables differ by exactly that one entry.
    GGML_ASSERT(pk.desc->in.size() == dk.desc->in.size() + 1);

    groups.clear();
    int first = 0;
    for (int gi = 0;; ++gi) {
        const std::string pre = "g" + std::to_string(gi) + "_b0_";
        auto it = std::find_if(pk.desc->in.begin(), pk.desc->in.end(), [&](const io_desc & d) {
            return d.name == pre + "attn_w_q";
        });
        if (it == pk.desc->in.end() || it->shape.empty()) {
            break;
        }
        const int n = (int) it->shape[0];
        const bool dense = std::any_of(pk.desc->in.begin(), pk.desc->in.end(), [&](const io_desc & d) {
            return d.name == pre + "mlp_w_norm";
        });
        groups.push_back({ first, n, dense, pre });
        first += n;
    }
    GGML_ASSERT(groups.size() == 2 && groups[0].layers == 1 && groups[0].dense && !groups[1].dense);
    per_chunk = groups[1].layers;
    nchunks = (L - 1 + per_chunk - 1) / per_chunk;
    GGML_ASSERT(per_chunk > 0 && per_chunk <= 15 && nchunks * per_chunk >= L - 1);
    chunks.resize((size_t) nchunks);

    for (const auto & d : pk.desc->in) {
        bank_n[d.name]     = d.bytes;
        bank_dtype[d.name] = d.dtype;
        // Only a g<group>_b<slot>_* bank has a leading layer axis.  Global banks such as
        // head_w / head_s / head_norm / cos / sin are one tensor at slot 0.
        const bool layer_bank = d.name.size() > 2 && d.name[0] == 'g' && std::isdigit((unsigned char) d.name[1]) &&
                                d.name[2] == '_';
        bank_slot[d.name] = layer_bank && d.shape.size() >= 1 && d.shape[0] > 0
                                ? (uint32_t) (d.bytes / d.shape[0])
                                : d.bytes;
        if (io_name(d.name)) {
            continue;
        }
        if (starts_with(d.name, "g0_")) {
            dense[d.name]  = axera_cmm_alloc(d.bytes);
            dense0[d.name] = axera_cmm_alloc(d.bytes);  // zeroed by the allocator: identity on later calls
        } else if (starts_with(d.name, "g1_")) {
            for (auto & c : chunks) {
                c[d.name] = axera_cmm_alloc(d.bytes);
            }
            if (d.name == groups[1].pre + "b0_kv") {
                GGML_ASSERT(d.shape.size() >= 1 && (int) d.shape[0] == per_chunk);
            }
        } else {
            gbank[d.name] = axera_cmm_alloc(d.bytes);
        }
    }
    GGML_ASSERT(gbank.count("cos") && gbank.count("sin"));

    fill_bank(t, rope_node, inp_pos);
    axera_flush_file_pages();

    auto alloc_io = [&](const io_desc & d) {
        if (!io.count(d.name)) {
            io[d.name] = axera_cmm_alloc(d.bytes);
        }
    };
    for (const auto & d : pk.desc->in) {
        if (io_name(d.name)) {
            alloc_io(d);
        }
    }
    for (const auto & d : dk.desc->in) {
        if (io_name(d.name) && d.name != "valid_tokens") {
            alloc_io(d);
        }
    }
    const uint32_t hidden_b = std::max<uint32_t>(pk.desc->out[0].bytes, dk.desc->out[0].bytes);
    io["output"]  = axera_cmm_alloc(hidden_b);
    io["hidden2"] = axera_cmm_alloc(hidden_b);
    for (const io_desc & d : pk.desc->out) {
        if (d.name == "route_ids") {
            for (int c = 0; c < nchunks; ++c) route_cmm.push_back(axera_cmm_alloc(d.bytes));
        }
    }
    open_route_dump();

    bind_calls(pre, pk, "prefill", true);
    bind_calls(dec, dk, "decode", false);

    cells = (int64_t) PAGES * PG;
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
    size_t bytes = 0;
    for (const auto & [n, b] : gbank) bytes += b->size;
    for (const auto & [n, b] : dense) bytes += b->size;
    for (const auto & c : chunks) {
        for (const auto & [n, b] : c) bytes += b->size;
    }
    GGML_LOG_INFO("axera hy_mt2: %d layers, %d bodies, %d calls/page, banks %.2f GiB, %lld cells\n", L,
                  (int) groups.size(), nchunks, bytes / 1073741824.0, (long long) cells);
    inited = true;
}

void hy_mt2::fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                       const ggml_tensor * inp_pos) {
    auto put_f32 = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v, float scale = 1.0f) {
        auto sl = bank_slot.find(n);
        // block_arguments pads a layer's scalar tensor to a 32-byte row when
        // several layers share one bank (12 x 8 floats for scale2).  Only the
        // first value is meaningful; the padding must be zero.
        GGML_ASSERT(sl != bank_slot.end() && v.size() * sizeof(float) <= sl->second);
        float * d = (float *) slot(bm, n, b);
        memset(d, 0, sl->second);
        for (size_t i = 0; i < v.size(); ++i) d[i] = v[i] * scale;
    };
    auto put_bf16 = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v, float scale = 1.0f) {
        auto sl = bank_slot.find(n);
        GGML_ASSERT(sl != bank_slot.end() && v.size() * sizeof(ggml_bf16_t) <= sl->second);
        memset(slot(bm, n, b), 0, sl->second);
        std::vector<float> tmp(v.size());
        for (size_t i = 0; i < v.size(); ++i) tmp[i] = v[i] * scale;
        ggml_fp32_to_bf16_row(tmp.data(), (ggml_bf16_t *) slot(bm, n, b), (int64_t) v.size());
    };
    auto fill_f32 = [&](bmap & bm, const std::string & n, int b, float value) {
        auto sl = bank_slot.find(n);
        GGML_ASSERT(sl != bank_slot.end() && sl->second % sizeof(float) == 0);
        float * d = (float *) slot(bm, n, b);
        std::fill(d, d + sl->second / sizeof(float), value);
    };
    auto fill_zero = [&](bmap & bm, const std::string & n, int b) {
        auto sl = bank_slot.find(n);
        GGML_ASSERT(sl != bank_slot.end());
        memset(slot(bm, n, b), 0, sl->second);
    };
    // a whole projection into one layer slot.  Projections are NVFP4 in the release GGUF
    // (codewords and E4M3 scales copied unchanged).  The untied output head stays Q8_0 in
    // the GGUF, exactly like the released 27B/35B packages; its 35B path quantizes the head
    // once while loading into the kernel's NVFP4 layout, so do the same here.
    auto put_nv_whole = [&](bmap & bm, const std::string & qn, const std::string & sn, const std::string & zn,
                            const ggml_tensor * w, int b) {
        const int64_t K = w->ne[0], N = ggml_nrows(w);
        GGML_ASSERT(w->ne[2] == 1 && K % 64 == 0);
        uint8_t * q = slot(bm, qn, b);
        uint8_t * s = slot(bm, sn, b);
        GGML_ASSERT((size_t) N * (K / 2) <= bank_slot.at(qn));
        GGML_ASSERT((size_t) N * (K / 16) <= bank_slot.at(sn));
        float s2;
        if (w->type == GGML_TYPE_NVFP4) {
            axera_copy_nvfp4_slice(w, 0, N, q, K / 2, s, K / 16);
            const ggml_tensor * ws = nv_scale(t, w);
            s2 = ((const float *) ws->data)[0];
        } else {
            s2 = axera_quantize_nvfp4_slice(w, 0, N, q, K / 2, s, K / 16);
        }
        fill_f32(bm, zn, b, s2);
    };
    auto put_nv_rows = [&](bmap & bm, const std::string & qn, const std::string & sn, const std::string & zn,
                           const ggml_tensor * w, int64_t r0, int64_t nr, int b, int64_t row0, int64_t s2i) {
        GGML_ASSERT(w->type == GGML_TYPE_NVFP4);
        const int64_t K = w->ne[0];
        uint8_t * q = slot(bm, qn, b) + (size_t) row0 * (K / 2);
        uint8_t * s = slot(bm, sn, b) + (size_t) row0 * (K / 16);
        axera_copy_nvfp4_slice(w, r0, nr, q, K / 2, s, K / 16);
        const ggml_tensor * ws = nv_scale(t, w);
        const float * v = (const float *) ws->data;
        ((float *) slot(bm, zn, b))[s2i] = v[ggml_nelements(ws) == 1 ? 0 : s2i];
    };
    auto drop = [&](const std::vector<const ggml_tensor *> & read) {
        if (!axera_keep_pages()) {
            for (const ggml_tensor * w : read) axera_drop_mapped(w);
        }
    };

    auto fill_layer = [&](const group_info & g, bmap & bm, int b, int il) {
        std::vector<const ggml_tensor *> read;
        auto W = [&](const char * s) -> const ggml_tensor * {
            const ggml_tensor * w = need(t, tensor_name(il, s));
            read.push_back(w);
            return w;
        };
        const std::string pa = g.pre + "attn_";
        put_bf16(bm, pa + "w_norm_attn", b, f32_of(W("attn_norm.weight")));
        put_f32(bm, pa + "w_norm_q", b, f32_of(W("attn_q_norm.weight")), 1.0f / std::sqrt((float) HD));
        put_bf16(bm, pa + "w_norm_k", b, f32_of(W("attn_k_norm.weight")));
        put_nv_whole(bm, pa + "w_q", pa + "w_scale_q", pa + "w_scale2_q", W("attn_q.weight"), b);
        put_nv_whole(bm, pa + "w_k", pa + "w_scale_k", pa + "w_scale2_k", W("attn_k.weight"), b);
        put_nv_whole(bm, pa + "w_v", pa + "w_scale_v", pa + "w_scale2_v", W("attn_v.weight"), b);
        put_nv_whole(bm, pa + "w_o", pa + "w_scale_o", pa + "w_scale2_o", W("attn_output.weight"), b);
        for (const char * c : { "q", "k", "v", "o" }) {
            fill_zero(bm, pa + "w_zeropoint_" + c, b);
        }
        if (g.dense) {
            const std::string pm = g.pre + "mlp_";
            put_f32(bm, pm + "w_norm", b, f32_of(W("ffn_norm.weight")));
            put_nv_whole(bm, pm + "w1", pm + "w_scale1", pm + "w_scale2_1", W("ffn_gate.weight"), b);
            put_nv_whole(bm, pm + "w2", pm + "w_scale2", pm + "w_scale2_2", W("ffn_up.weight"), b);
            put_nv_whole(bm, pm + "w3", pm + "w_scale3", pm + "w_scale2_3", W("ffn_down.weight"), b);
            fill_zero(bm, pm + "w_zp_scale1", b);
            fill_zero(bm, pm + "w_zp_scale2", b);
            fill_zero(bm, pm + "w_zp_scale3", b);
            fill_zero(bm, pm + "input_permutation", b);
        } else {
            const std::string pm = g.pre + "moe_";
            put_f32(bm, pm + "input_norm_w", b, f32_of(W("ffn_norm.weight")));
            put_f32(bm, pm + "correction_bias", b, f32_of(W("exp_probs_b")));
            put_bf16(bm, pm + "w_route", b, f32_of(W("ffn_gate_inp.weight")));
            fill_f32(bm, pm + "w_scale_route", b, 1.0f);
            fill_f32(bm, pm + "w_scale2_route", b, 1.0f);
            const ggml_tensor * wg = W("ffn_gate_exps.weight");
            const ggml_tensor * wu = W("ffn_up_exps.weight");
            const ggml_tensor * wd = W("ffn_down_exps.weight");
            GGML_ASSERT(wg->ne[1] == wu->ne[1] && wg->ne[2] == E && wu->ne[2] == E && wd->ne[2] == E);
            for (int e = 0; e < E; ++e) {
                put_nv_rows(bm, pm + "w_gate", pm + "w_scale_gate", pm + "w_scale2_gate", wg,
                            (int64_t) e * wg->ne[1], wg->ne[1], b, (int64_t) e * wg->ne[1], e);
                put_nv_rows(bm, pm + "w_up", pm + "w_scale_up", pm + "w_scale2_up", wu,
                            (int64_t) e * wu->ne[1], wu->ne[1], b, (int64_t) e * wu->ne[1], e);
                put_nv_rows(bm, pm + "w_down", pm + "w_scale_down", pm + "w_scale2_down", wd,
                            (int64_t) e * wd->ne[1], wd->ne[1], b, (int64_t) e * wd->ne[1], e);
            }
            const char * sn[3] = { "gate", "up", "down" };
            const char * wn[3] = { "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" };
            for (int j = 0; j < 3; ++j) {
                const std::string base = std::string("w_shared_expert_mlp_") + sn[j];
                put_nv_whole(bm, pm + base, pm + "w_scale_shared_expert_mlp_" + sn[j],
                             pm + "w_scale2_shared_expert_mlp_" + sn[j], W(wn[j]), b);
            }
        }
        drop(read);
    };

    const group_info & gd = groups[0];
    const group_info & gm = groups[1];
    fill_layer(gd, dense, 0, 0);
    for (int ci = 0; ci < nchunks; ++ci) {
        for (int b = 0; b < per_chunk; ++b) {
            const int il = 1 + ci * per_chunk + b;
            if (il < L) {
                fill_layer(gm, chunks[ci], b, il);
            }
        }
    }

    // final norm and NVFP4 lm_head
    {
        std::vector<const ggml_tensor *> read;
        const ggml_tensor * nw = need(t, "output_norm.weight");
        read.push_back(nw);
        put_f32(gbank, "head_norm", 0, f32_of(nw));
        const ggml_tensor * ow = need(t, "output.weight");
        read.push_back(ow);
        put_nv_whole(gbank, "head_w", "head_s", "head_s2", ow, 0);
        drop(read);
    }

    // RoPE tables are model-wide: one global pair instead of 48 copies.
    rope rp(rope_node, inp_pos, 0);
    std::vector<float> cs((size_t) PAGES * PG * (ROT / 2)), sn((size_t) PAGES * PG * (ROT / 2));
    for (int64_t q = 0; q < (int64_t) PAGES * PG; ++q) {
        for (int i = 0; i < ROT / 2; ++i) {
            const double th = (double) q * rp.freq_scale * rp.inv_freq[i];
            cs[(size_t) q * (ROT / 2) + i] = (float) std::cos(th);
            sn[(size_t) q * (ROT / 2) + i] = (float) std::sin(th);
        }
    }
    put_f32(gbank, "cos", 0, cs);
    put_f32(gbank, "sin", 0, sn);

    for (auto & [n, c] : gbank) c->flush();
    for (auto & [n, c] : dense) c->flush();
    for (auto & c : chunks) {
        for (auto & [n, b] : c) b->flush();
    }
}

seq_caps hy_mt2::seq_get_caps() const {
    return seq_caps{ PG, H_, 0, cells, 0, 0, true };
}

void hy_mt2::bind_calls(std::vector<call> & calls, const kernel & k, const char * tag, bool prefill) {
    (void) prefill;
    calls.assign((size_t) nchunks, {});
    for (int c = 0; c < nchunks; ++c) {
        call & cl = calls[c];
        cl.k      = &k;
        axera_cmm & src = c == 0 ? io.at("x") : (c % 2 ? io.at("output") : io.at("hidden2"));
        axera_cmm & dst = c % 2 == 0 ? io.at("output") : io.at("hidden2");
        cl.in.resize(k.desc->in.size());
        for (size_t i = 0; i < k.desc->in.size(); ++i) {
            const io_desc & d = k.desc->in[i];
            const axera_cmm * b = nullptr;
            if (d.name == "x") {
                b = &src;
            } else if (d.name == "output") {
                b = &dst;  // (an in-place graph input)
            } else if (io_name(d.name)) {
                auto it = io.find(d.name);
                if (it == io.end()) {
                    GGML_ABORT("axera hy_mt2: %s: no buffer for input %s\n", tag, d.name.c_str());
                }
                b = &it->second;
            } else if (starts_with(d.name, "g0_")) {
                b = &at(c == 0 ? dense : dense0, d.name);
            } else if (starts_with(d.name, "g1_")) {
                b = &at(chunks[c], d.name);
            } else if (d.name == "route_ids" && c < (int) route_cmm.size()) {
                b = &route_cmm[c];
            } else {
                b = &at(gbank, d.name);
            }
            cl.in[i] = axera_io(*b, d.bytes);
        }
        cl.out.resize(k.desc->out.size());
        for (size_t i = 0; i < k.desc->out.size(); ++i) {
            const io_desc & d = k.desc->out[i];
            if (d.name == "output") {
                cl.out[i] = axera_io(dst, d.bytes);
            } else if (d.name == "route_ids" && c < (int) route_cmm.size()) {
                cl.out[i] = axera_io(route_cmm[c], d.bytes);
            } else {
                GGML_ABORT("axera hy_mt2: %s: unexpected output %s\n", tag, d.name.c_str());
            }
        }
    }
}

void hy_mt2::open_route_dump() {
    const char * p = getenv("GGML_AXERA_HY_MT2_ROUTE_DUMP");
    if (!p || !*p) return;
    route_file = fopen(p, "wb");
    if (!route_file) {
        GGML_LOG_WARN("axera hy_mt2: cannot open route dump %s\n", p);
    }
}

void hy_mt2::dump_routes(int64_t cell, int rows, int layer_stride) {
    if (!route_file || route_cmm.empty()) return;
    const uint32_t magic = 0x52334854u;  // "TH3R"
    struct {
        uint32_t magic;
        int32_t  rows;
        int64_t  cell;
        int32_t  layers;
        int32_t  topk;
    } h = { magic, rows, cell, 47, 8 };
    fwrite(&h, sizeof(h), 1, route_file);
    for (int il = 1; il < 48; ++il) {
        const int ci = (il - 1) / per_chunk, b = (il - 1) % per_chunk;
        axera_cmm & r = route_cmm[ci];
        r->invalidate(r->vir, r->size);
        const int32_t * src = (const int32_t *) r->vir + (size_t) b * layer_stride;
        fwrite(src, sizeof(int32_t), (size_t) rows * 8, route_file);
    }
    fflush(route_file);
}

void hy_mt2::dump_hidden(const axera_cmm & b, int rows, int64_t cell, int call_index) {
    const char * p = getenv("GGML_AXERA_HY_MT2_HIDDEN_DUMP");
    if (!p || !*p) return;
    const size_t n = (size_t) rows * H * sizeof(uint16_t);
    b->invalidate(b->vir, n);
    FILE * f = fopen(p, "ab");
    if (!f) return;
    const uint32_t magic = 0x48334854u;  // "TH3H"
    const int32_t  h[4] = { rows, (int32_t) cell, call_index, 0 };
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(h, sizeof(h), 1, f);
    fwrite(b->vir, 1, n, f);
    fclose(f);
}

void hy_mt2::run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                              bool need_head) {
    GGML_ASSERT(cell % PG == 0 && rows > 0 && rows <= PG);
    auto xi = io.find("x"), oi = io.find("output");
    GGML_ASSERT(xi != io.end() && oi != io.end());
    std::vector<uint16_t> xb((size_t) PG * H, 0);
    memcpy(xb.data(), x, (size_t) rows * H * sizeof(uint16_t));
    memcpy(xi->second->vir, xb.data(), xb.size() * sizeof(uint16_t));
    auto pi = io.find("position_ids");
    int32_t * pos = (int32_t *) pi->second->vir;
    for (int r = 0; r < PG; ++r) pos[r] = r < rows ? p4[(size_t) r * 4] : 0;
    const int page = (int) (cell / PG);
    auto ind = io.find("indptr"), idx = io.find("indices"), last = io.find("last_page_len"), cl = io.find("cache_location");
    auto vt = io.find("valid_tokens");
    // prefill_attention_tile takes indptr as the *history* page range; the page
    // being written is cache_location, not part of indptr.
    ((int32_t *) ind->second->vir)[0] = 0;
    ((int32_t *) ind->second->vir)[1] = page;
    GGML_ASSERT((size_t) page * sizeof(int32_t) <= idx->second->size);
    int32_t * ix = (int32_t *) idx->second->vir;
    for (int i = 0; i <= page; ++i) ix[i] = i;
    *(int32_t *) last->second->vir = 0;  // the prefill graph does not read it
    ((int32_t *) cl->second->vir)[0] = page;
    ((int32_t *) cl->second->vir)[1] = 0;
    *(int32_t *) vt->second->vir = rows;
    for (auto * b : { &xi->second, &pi->second, &ind->second, &idx->second, &last->second, &cl->second, &vt->second }) {
        (*b)->flush();
    }
    // batch_cell0 is the cell of the caller's batch row 0.  A replayed page
    // prefix (a short -ub later in the page) is part of the kernel's rows but
    // its logits are not wanted; compute only the caller's tail.
    const int first = std::max(0, std::min(rows, (int) (batch_cell0 - cell)));
    const int want  = rows - first;
    GGML_ASSERT(want > 0 && want <= rows);
    const int take = std::min(HEAD_ROWS, want);
    const int hr0 = rows - take;
    auto hx = io.find("head_x"), hri = io.find("head_rows");
    memset(hx->second->vir, 0, hx->second->size);
    int32_t * hrb = (int32_t *) hri->second->vir;
    for (int r = 0; r < HEAD_ROWS; ++r) hrb[r] = r < take ? hr0 + r : hr0;
    hri->second->flush();

    const bool prof = getenv("GGML_AXERA_HY_MT2_PROF") != nullptr;
    const bool dump = getenv("GGML_AXERA_HY_MT2_HIDDEN_DUMP") != nullptr;
    auto head_run = io.find("head_run");
    GGML_ASSERT(head_run != io.end());
    for (int ci = 0; ci < nchunks; ++ci) {
        *(int32_t *) head_run->second->vir = need_head && ci + 1 == nchunks ? 1 : 0;
        head_run->second->flush();
        const int64_t t0   = prof ? ggml_time_us() : 0;
        pre[ci].k->run(pre[ci].in, pre[ci].out, "prefill");
        if (prof) {
            GGML_LOG_INFO("axera hy_mt2: prefill call %d cell %lld rows %d kernel %.3f ms\n",
                          ci, (long long) cell, rows, (ggml_time_us() - t0) / 1e3);
        }
        if (dump) {
            const axera_cmm & b = ci % 2 == 0 ? io.at("output") : io.at("hidden2");
            dump_hidden(b, rows, cell, ci);
        }
    }

    oi->second->invalidate(oi->second->vir, (size_t) rows * H * sizeof(uint16_t));
    dump_routes(cell, rows, PG * 8);
    head_first = (int) (cell + hr0 - batch_cell0);
    head_n = take;
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, (size_t) HEAD_ROWS * VOCAB * sizeof(float));
}

void hy_mt2::run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4) {
    auto xi = io.find("x"), oi = io.find("output");
    GGML_ASSERT(xi != io.end() && oi != io.end());
    memcpy(xi->second->vir, x, H * sizeof(uint16_t));
    xi->second->flush();
    auto pi = io.find("position_ids");
    *(int32_t *) pi->second->vir = p4[0];
    pi->second->flush();
    const int page = (int) (cell / PG), off = (int) (cell % PG);
    auto ind = io.find("indptr"), idx = io.find("indices"), last = io.find("last_page_len"), cl = io.find("cache_location");
    // decode_attention computes cached_len = (pages-1)*PG + last_page_len.
    // That is the number of tokens already in KV, i.e. the current cell itself;
    // the current token's K/V are passed separately and written at page/off.
    // Cell 0 has no history: one empty page keeps the kernel's
    // cached_len = (pages - 1) * PG + last_page_len at 0 instead of -PG.
    const int hist_pages = cell == 0 ? 1 : (int) ((cell + PG - 1) / PG);
    const int hist_last  = cell == 0 ? 0 : (int) (cell - (int64_t) (hist_pages - 1) * PG);
    ((int32_t *) ind->second->vir)[0] = 0;
    ((int32_t *) ind->second->vir)[1] = hist_pages;
    GGML_ASSERT((size_t) hist_pages * sizeof(int32_t) <= idx->second->size);
    int32_t * ix = (int32_t *) idx->second->vir;
    for (int i = 0; i <= page; ++i) ix[i] = i;
    *(int32_t *) last->second->vir = hist_last;
    ((int32_t *) cl->second->vir)[0] = page;
    ((int32_t *) cl->second->vir)[1] = off;
    for (auto * b : { &pi->second, &ind->second, &idx->second, &last->second, &cl->second }) (*b)->flush();
    auto hx = io.find("head_x"), hri = io.find("head_rows");
    memset(hx->second->vir, 0, hx->second->size);
    *(int32_t *) hri->second->vir = 0;
    hri->second->flush();
    const bool prof = getenv("GGML_AXERA_HY_MT2_PROF") != nullptr;
    auto head_run = io.find("head_run");
    GGML_ASSERT(head_run != io.end());
    for (int ci = 0; ci < (int) dec.size(); ++ci) {
        *(int32_t *) head_run->second->vir = ci + 1 == (int) dec.size() ? 1 : 0;
        head_run->second->flush();
        const int64_t t0 = prof ? ggml_time_us() : 0;
        dec[ci].k->run(dec[ci].in, dec[ci].out, "decode");
        if (prof) {
            GGML_LOG_INFO("axera hy_mt2: decode call %d cell %lld kernel %.3f ms\n",
                          ci, (long long) cell, (ggml_time_us() - t0) / 1e3);
        }
    }
    dump_routes(cell, 1, 8);
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, VOCAB * sizeof(float));
    head_first = 0;  // the decode batch is one row: its logits row is batch row 0
    head_n = 1;
}

bool hy_mt2::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const uint16_t * xr = x;
    if (!xr) {
        GGML_ASSERT(batch0 == 0);
        xbuf.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) xbuf.data());
        xr = xbuf.data();
    }
    for (int off = 0; off < n; off += PG) {
        const int rows = std::min(PG, n - off);
        run_prefill_page(c0 + off, rows, xr + (size_t) off * H, p4 + (size_t) off * 4, c0 + batch0,
                         off + PG >= n);
    }
    if (snap) snap->deferred = false;
    return true;
}

bool hy_mt2::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
    run_decode_row(c, x, p4);
    return true;
}

void hy_mt2::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) return;
    if (batch_rows) memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    else memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
}

void hy_mt2::publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T) {
    GGML_ASSERT(logits && logits->type == GGML_TYPE_F32 && logits->ne[0] == VOCAB);
    const int n_out = (int) logits->ne[1];
    if (n_out == 0) {
        return;  // a warm-up or an image-only batch asks for no logits
    }
    if (n_out > head_n) {
        GGML_ABORT("axera hy_mt2: %d requested logits rows but the page head computed %d (keep -ub <= %d)\n",
                   n_out, head_n, HEAD_ROWS);
    }
    ggml_tensor * out_ids = nullptr;
    if (n_out < T) {
        out_ids = names.find("out_ids");
        if (!out_ids || out_ids->type != GGML_TYPE_I32 || out_ids->ne[0] != n_out) {
            out_ids = nullptr;
            for (int i = 0; i < g->n_nodes; ++i) {
                ggml_tensor * n = g->nodes[i];
                if (n->op == GGML_OP_GET_ROWS && n->src[1] && n->src[1]->type == GGML_TYPE_I32 &&
                    n->src[1]->ne[0] == n_out && n->src[0]->ne[1] == T) {
                    out_ids = n->src[1];
                }
            }
        }
    }
    if (n_out < T && !out_ids) GGML_ABORT("axera hy_mt2: %d of %d output rows without row indices\n", n_out, T);
    const axera_cmm & hl = io.at("head_logits");
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        if (row < head_first || row >= head_first + head_n) {
            GGML_ABORT("axera hy_mt2: output row %d outside the head's %d..%d\n", row, head_first,
                       head_first + head_n - 1);
        }
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        memcpy(o, (const float *) hl->vir + (size_t) (row - head_first) * VOCAB, VOCAB * sizeof(float));
        dctx->flush(o, VOCAB * sizeof(float));
        dctx->clean_p = o;
        dctx->clean_n = VOCAB * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    output_head::after(g, names);
}

ggml_status hy_mt2::compute(ggml_cgraph * g) {
    const auto & t = names.of(g);
    ggml_tensor * norm0 = names.find("attn_norm-0");
    ggml_tensor * logits = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera hy_mt2: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) x0 = x0->src[0];
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) ++n0;
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0 = x0->src[0];
    const int Hh = (int) x0->ne[0], T = (int) x0->ne[1];
    int Ll = 0;
    while (t.count("blk." + std::to_string(Ll) + ".attn_norm.weight")) ++Ll;
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; ++i) rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    GGML_ASSERT(rope_node);
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || Ll != L || Hh != H_) {
        L = Ll; H_ = Hh;
        GGML_ASSERT(L == 48 && H_ == H && Hh == H);
        ROT = 0;
        // rope() below sets ROT through its parsed inv_freq; read it once from the graph's node.
        {
            rope rp(rope_node, inp_pos, 0);
            ROT = (int) rp.inv_freq.size() * 2;
        }
        init(t, rope_node, inp_pos);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, cells, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera hy_mt2: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera hy_mt2: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t * pos = (const int32_t *) inp_pos->data;
    const bool mr = inp_pos->ne[0] == 4 * (int64_t) T;
    pbuf.resize((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) pbuf[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
    });
    batch_rows = xb.data();
    last_seq_status = 0;
    const int64_t c0 = kv.cell_of(0);
    const seq_status sst = seq->feed(c0, T, (const uint16_t *) xb.data(), pbuf.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) sst;
        if (sst == SEQ_MISS) {
            GGML_LOG_WARN("axera hy_mt2: batch from cell %lld has no restart point (%s); the slot must replay\n",
                          (long long) c0, seq->why());
        } else {
            GGML_LOG_ERROR("axera hy_mt2: sequence state rejected a batch from cell %lld: %s\n",
                           (long long) c0, seq->why());
        }
        return GGML_STATUS_FAILED;
    }
    publish_logits(g, logits, T);
    return GGML_STATUS_SUCCESS;
}

size_t hy_mt2::state_size(uint32_t flags) const {
    if (!inited || !seq) return 0;
    const bool partial = flags & state_flags_partial_only;
    const size_t seq_n = seq->size(partial);
    if (!seq_n) return 0;
    size_t page_b = 0;
    if (!partial) {
        page_b = bank_n.at(groups[0].pre + "kv") / PAGES;  // one layer in the dense body
        GGML_ASSERT(bank_n.at(groups[1].pre + "kv") / per_chunk / PAGES == page_b);
    }
    const int64_t cell = std::max<int64_t>(0, seq->cell());
    const int nkv = partial ? 0 : (int) std::min<int64_t>(PAGES, (cell + PG - 1) / PG);
    return sizeof(state_header) + (uint64_t) L * nkv * page_b + seq_n;
}

size_t hy_mt2::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    const size_t total = state_size(flags);
    if (!total || size < total) return 0;
    const bool partial = flags & state_flags_partial_only;
    const size_t seq_n = seq->size(partial);
    const size_t page_b = bank_n.at(groups[0].pre + "kv") / PAGES;
    const int64_t cell = std::max<int64_t>(0, seq->cell());
    const int nkv = partial ? 0 : (int) std::min<int64_t>(PAGES, (cell + PG - 1) / PG);

    state_header h{};
    memcpy(h.magic, "HYMT2S1", state_magic_n - 1);
    h.version = 2;
    h.partial = partial;
    h.ngroups = (uint32_t) (1 + nchunks);
    h.state_cell = seq->cell();
    h.nkv_pages = (uint32_t) nkv;
    h.kv_page_b = (uint32_t) page_b;
    h.kv_n = (uint64_t) L * nkv * page_b;
    h.seq_n = seq_n;
    char * p = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (!partial) {
        const size_t n = (size_t) nkv * page_b;
        auto copy_out = [&](bmap & bm, const std::string & name, int layer) {
            axera_cmm & c = bm.at(name);
            char * src = (char *) c->vir + (size_t) layer * bank_slot.at(name);
            c->invalidate(src, n);
            memcpy(p, src, n);
            p += n;
        };
        const std::string dk = groups[0].pre + "kv", mk = groups[1].pre + "kv";
        copy_out(dense, dk, 0);
        int done = 1;
        for (int ci = 0; ci < nchunks; ++ci) {
            for (int b = 0; b < per_chunk && done < L; ++b, ++done) {
                copy_out(chunks[ci], mk, b);
            }
        }
        GGML_ASSERT(done == L);
    }
    if (seq->write((uint8_t *) p, seq_n, partial) != seq_n) return 0;
    p += seq_n;
    return (size_t) (p - (char *) dst);
}

size_t hy_mt2::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    if (size < sizeof(state_header)) return 0;
    const bool partial = flags & state_flags_partial_only;
    state_header h{};
    memcpy(&h, src, sizeof(h));
    const size_t page_b = bank_n.at(groups[0].pre + "kv") / PAGES;
    if (memcmp(h.magic, "HYMT2S1", state_magic_n - 1) != 0 || h.version != 2 || h.partial != partial ||
        h.ngroups != (uint32_t) (1 + nchunks) || !h.seq_n || h.state_cell < -1 || h.state_cell > cells ||
        h.nkv_pages > (uint32_t) PAGES) {
        return 0;
    }
    const uint64_t kv_n = partial ? 0 : (uint64_t) L * h.nkv_pages * page_b;
    if (h.kv_page_b != page_b || h.kv_n != kv_n || size < sizeof(h) + kv_n + h.seq_n) return 0;
    const char * p = (const char *) src + sizeof(h);
    if (!partial) {
        const size_t n = (size_t) h.nkv_pages * page_b;
        auto copy_in = [&](bmap & bm, const std::string & name, int layer) {
            axera_cmm & c = bm.at(name);
            char * d = (char *) c->vir + (size_t) layer * bank_slot.at(name);
            memcpy(d, p, n);
            c->flush(d, n);
            p += n;
        };
        const std::string dk = groups[0].pre + "kv", mk = groups[1].pre + "kv";
        copy_in(dense, dk, 0);
        int done = 1;
        for (int ci = 0; ci < nchunks; ++ci) {
            for (int b = 0; b < per_chunk && done < L; ++b, ++done) {
                copy_in(chunks[ci], mk, b);
            }
        }
        GGML_ASSERT(done == L);
    }
    if (seq->read((const uint8_t *) p, h.seq_n) != h.seq_n) return 0;
    p += h.seq_n;
    return (size_t) (p - (const char *) src);
}

}  // namespace

std::shared_ptr<model> make_hy_mt2(const manifest & m) { return std::make_shared<hy_mt2>(m); }

}  // namespace axera
