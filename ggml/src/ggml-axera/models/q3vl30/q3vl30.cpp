// Axera NPU backend: Qwen3-VL-30B-A3B, a 48-layer GQA MoE language tower with three deepstack planes.
//
// The manifest has two merged kernels:
//   prefill  one reusable six-layer body for a 4096-row workspace. The host launches
//            it eight times, rebinding the same body to layer groups 0..7, then the
//            final RMSNorm and NVFP4 output head for up to eight rows.
//   decode   all 48 layers statically expanded as eight six-layer groups for one token,
//            then the same final RMSNorm and output head.
// The weights come from a bf16 GGUF quantized once to standard GGML NVFP4 blocks with
// one F32 .scale sidecar per quantization slice. Loading copies the codewords and the
// per-16-input UE4M3 scales straight into the kernels' pair-major banks; the sidecar is
// the kernels' FP32 scale2. No load-time requantization is done.
//
// KV is BF16, paged in 256-cell pages, one [1040][2][256][4][128] input per layer.
// 1024 pages cover 262144 cells and 16 tail pages absorb the 4096-row prefill's writes
// past the last valid cell. The host allocates only ceil(-c / 256) + 16 pages per layer.
// seqstate owns page restart points and replay; this model has no recurrent state.

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
#include <sys/stat.h>
#include <vector>

namespace axera {
namespace {

constexpr int PG = 256;
constexpr int WP_MAX = 4096;
constexpr int H = 2048;
constexpr int HQ = 32;
constexpr int KVH = 4;
constexpr int HD = 128;
constexpr int E = 128;
constexpr int TOPK = 8;
constexpr int VOCAB = 151936;
constexpr int PAGES = 1040;  // 1024 pages of context plus 16 slots the 4096-row prefill writes past the last cell
constexpr int KV_HEADROOM = WP_MAX / PG;  // pages a 4096-row prefill call writes from its first page
constexpr int HEAD_ROWS = 8;
constexpr int LAYERS = 48;
constexpr int GROUP_LAYERS = 6;
constexpr int GROUPS = 8;
constexpr int IN = 4 * H;  // x plus the three deepstack planes

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

class q3vl30 final : public model, public seq_model {
  public:
    explicit q3vl30(const manifest & m) : man(m) {}

    ggml_status compute(ggml_cgraph * g) override;
    size_t      state_size(uint32_t flags) const override;
    size_t      state_write(void * dst, size_t size, uint32_t flags) override;
    size_t      state_read(const void * src, size_t size, uint32_t flags) override;
    int         last_state_status() const override { return last_seq_status; }

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
    int              L = 0, H_ = 0;
    int64_t          cells = 0;

    using bmap = std::map<std::string, axera_cmm>;
    bmap                                  gbank;   // head weights
    std::vector<bmap>                     chunks;  // one six-layer bank per group
    std::vector<axera_cmm>                KVL;     // one BF16 paged cache per layer
    int                                   kv_pages = 0, kv_total_pages = 0;
    size_t                                kv_page_b = 0;
    std::map<std::string, axera_cmm>      io;
    std::map<std::string, uint32_t>       bank_n, bank_dtype, bank_slot;
    std::vector<group_info>               groups;   // weight/decode bank groups
    std::vector<group_info>               pgroups;  // prefill graph groups
    int                                   nchunks = 1;   // weight banks
    int                                   pre_calls = 1; // prefill host launches
    int                                   wp = PG;  // prefill rows: 256 or 4096, from the manifest

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
    // A batch too small for the big prefill workspace is replayed one row at a
    // time through the decode kernel; its per-row logits are collected here.
    std::vector<float> small_logits;
    bool               small_head = false;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
              const ggml_tensor * inp_pos);
    void fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                   const ggml_tensor * inp_pos);
    void bind_calls(std::vector<call> & calls, const kernel & k, const char * tag, bool prefill);
    void run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                          bool need_head);
    void dump_x(const axera_cmm & b, int rows, int64_t cell, int call_index);
    void dump_hidden(const axera_cmm & b, int rows, int64_t cell, int call_index);
    void dump_debug_call(int ci, int64_t cell, int rows, int64_t batch_cell0, int page,
                         int first, int want, int take, int hr0, bool need_head);
    void dump_debug_output(int ci, int rows, int64_t cell);
    void open_route_dump();
    void dump_routes(int64_t cell, int rows, int layer_stride);
    void run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4);
    void publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T);

    static const axera_cmm & at(const bmap & b, const std::string & name) {
        auto i = b.find(name);
        if (i == b.end()) {
            GGML_ABORT("axera q3vl30: %s is not a bank of this call\n", name.c_str());
        }
        return i->second;
    }
    uint8_t * slot(const bmap & b, const std::string & name, int layer) const {
        auto i = b.find(name);
        auto s = bank_slot.find(name);
        if (i == b.end() || s == bank_slot.end()) {
            GGML_ABORT("axera q3vl30: %s is not a layer bank\n", name.c_str());
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
        GGML_ABORT("axera q3vl30: %s is not in the graph\n", n.c_str());
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
        GGML_ABORT("axera q3vl30: NVFP4 tensor %s has no F32 .scale sidecar\n", w->name);
    }
    GGML_ASSERT(it->second->type == GGML_TYPE_F32 && ggml_nelements(it->second) >= 1);
    return it->second;
}

static bool io_name(const std::string & n) {
    return n == "x" || n == "position_ids" || n == "indptr" || n == "indices" || n == "last_page_len" ||
           n == "cache_location" || n == "head_x" || n == "head_rows" || n == "head_run" ||
           n == "deepstack" || n == "ds_run" || n == "cos" || n == "sin" ||
           n == "head_logits" || n == "head_ids" ||
           n == "head_topk_v" || n == "head_topk_i" || n == "head_vmax" || n == "head_sumexp" ||
           n == "valid_tokens";
}

static bool starts_with(const std::string & n, const std::string & p) {
    return n.size() >= p.size() && n.compare(0, p.size(), p) == 0;
}

void q3vl30::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                  const ggml_tensor * inp_pos) {
    (void) rope_node;
    (void) inp_pos;
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");

    auto group_of = [](const std::string & n) {
        return n.size() >= 3 && n[0] == 'g' && std::isdigit((unsigned char) n[1]) && n[2] == '_'
                   ? n[1] - '0' : -1;
    };
    auto scan_groups = [&](const kernel & k, std::vector<group_info> & out) {
        int first = 0;
        for (int gi = 0; gi < GROUPS; ++gi) {
            const std::string pre = "g" + std::to_string(gi) + "_";
            auto it = std::find_if(k.desc->in.begin(), k.desc->in.end(), [&](const io_desc & d) {
                return d.name == pre + "b0_attn_w_q";
            });
            if (it == k.desc->in.end() || it->shape.empty()) {
                break;
            }
            const int n = (int) it->shape[0];
            out.push_back({ first, n, false, pre });
            first += n;
        }
    };
    scan_groups(pk, pgroups);
    scan_groups(dk, groups);
    GGML_ASSERT(pgroups.size() == 1 && pgroups[0].layers == GROUP_LAYERS && pgroups[0].first == 0);
    GGML_ASSERT(groups.size() == GROUPS);
    for (int gi = 0; gi < GROUPS; ++gi) {
        GGML_ASSERT(groups[gi].layers == GROUP_LAYERS && groups[gi].first == gi * GROUP_LAYERS);
    }
    nchunks = GROUPS;
    pre_calls = GROUPS;
    chunks.resize(GROUPS);

    std::vector<std::map<std::string, uint32_t>> cbytes((size_t) GROUPS);
    for (const auto & d : dk.desc->in) {
        bank_n[d.name]     = d.bytes;
        bank_dtype[d.name] = d.dtype;
        const bool layer_bank = group_of(d.name) >= 0;
        bank_slot[d.name] = layer_bank && !d.shape.empty() && d.shape[0] > 0
                                ? (uint32_t) (d.bytes / d.shape[0]) : d.bytes;
        if (io_name(d.name)) {
            continue;
        }
        const int gi = group_of(d.name);
        if (gi >= 0) {
            if (d.name.find("_kv") != std::string::npos) {
                continue;  // the KV pools are allocated from -c, not from the compiled extent
            }
            cbytes[(size_t) gi][d.name] = std::max(cbytes[(size_t) gi][d.name], d.bytes);
        } else {
            gbank[d.name] = axera_cmm_alloc_bank(d.bytes);
        }
    }
    // The prefill graph carries one reusable six-layer body, named g0_.  Its
    // banks have the same shapes as each decode group's.
    for (const auto & d : pk.desc->in) {
        if (io_name(d.name) || group_of(d.name) != 0 || d.name.find("_kv") != std::string::npos) {
            continue;
        }
        const std::string suffix = d.name.substr(3);
        for (int gi = 0; gi < GROUPS; ++gi) {
            const std::string mapped = "g" + std::to_string(gi) + "_" + suffix;
            auto it = cbytes[(size_t) gi].find(mapped);
            if (it != cbytes[(size_t) gi].end()) {
                it->second = std::max(it->second, d.bytes);
            }
        }
    }
    for (int gi = 0; gi < GROUPS; ++gi) {
        for (const auto & [name, bytes] : cbytes[(size_t) gi]) {
            chunks[(size_t) gi][name] = axera_cmm_alloc_bank(bytes);
        }
    }
    GGML_ASSERT(gbank.count("cos") == 0 && gbank.count("sin") == 0);
    GGML_ASSERT(gbank.count("head_w") && gbank.count("head_s") && gbank.count("head_norm"));

    auto alloc_io = [&](const io_desc & d) {
        auto it = io.find(d.name);
        if (it == io.end()) {
            io[d.name] = axera_cmm_alloc(d.bytes);
        } else if (it->second->size < d.bytes) {
            io[d.name] = axera_cmm_alloc(d.bytes);
        }
    };
    for (const auto * desc : { pk.desc, dk.desc }) {
        for (const auto & d : desc->in) {
            if (io_name(d.name)) {
                alloc_io(d);
            }
        }
        for (const auto & d : desc->out) {
            if (io_name(d.name) && (d.name == "output" || d.name == "head_logits")) {
                alloc_io(d);
            }
        }
    }
    const uint32_t hidden_b = (uint32_t) ((size_t) WP_MAX * H * sizeof(ggml_bf16_t));
    io["output"]  = axera_cmm_alloc(hidden_b);
    io["hidden2"] = axera_cmm_alloc(hidden_b);

    // One BF16 KV pool per layer.  The kernel was compiled for its page
    // capacity; llama.cpp's -c decides how many pages are allocated and how
    // many bytes are bound.
    const auto kv_in = std::find_if(dk.desc->in.begin(), dk.desc->in.end(), [](const io_desc & d) {
        return d.name == "g0_kv0";
    });
    GGML_ASSERT(kv_in != dk.desc->in.end());
    kv_page_b = (size_t) 2 * PG * KVH * HD * sizeof(ggml_bf16_t);
    const int NP = (int) (kv_in->bytes / kv_page_b);
    GGML_ASSERT(NP > 0 && (size_t) NP * kv_page_b == kv_in->bytes);
    GGML_ASSERT(NP > KV_HEADROOM);
    kv_total_pages = NP;
    // llama.cpp's -c decides how much of the kernel's page extent is allocated.
    // A 4096-row prefill call writes K/V for every physical row, including the
    // tail padding, so keep KV_HEADROOM pages after the context's last page.
    int64_t cells_want = (int64_t) (NP - KV_HEADROOM) * PG;
    if (const int64_t c = axera_kv_cells(); c > 0) {
        kv_pages  = (int) std::max<int64_t>(1, std::min<int64_t>(NP, (c + PG - 1) / PG + KV_HEADROOM));
        cells_want = c;
    } else {
        kv_pages = NP;
    }
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        kv_pages = std::max(1, std::min(NP, atoi(e)));
    }
    if (kv_pages <= KV_HEADROOM) {
        kv_pages = KV_HEADROOM + 1;
    }
    KVL.clear();
    for (int j = 0; j < LAYERS; ++j) {
        KVL.push_back(axera_cmm_alloc_bank((size_t) kv_pages * kv_page_b));
        KVL.back()->flush(KVL.back()->vir, KVL.back()->size);
    }
    cells = std::max<int64_t>(1, std::min<int64_t>(cells_want, (int64_t) (kv_pages - KV_HEADROOM) * PG));

    fill_bank(t, rope_node, inp_pos);
    axera_flush_file_pages();

    {
        auto it = io.find("x");
        GGML_ASSERT(it != io.end());
        wp = (int) (it->second->size / (H * sizeof(ggml_bf16_t)));
        GGML_ASSERT(wp == WP_MAX);
    }
    bind_calls(pre, pk, "prefill", true);
    bind_calls(dec, dk, "decode", false);

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
    for (const auto & c : chunks) {
        for (const auto & [n, b] : c) bytes += b->size;
    }
    GGML_LOG_INFO("axera q3vl30: %d layers, %d prefill calls, KV %d of %d pages, banks %.2f GiB, cells %lld\n",
                  LAYERS, pre_calls, kv_pages, NP, bytes / 1073741824.0, (long long) cells);
    inited = true;
}

void q3vl30::fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                       const ggml_tensor * inp_pos) {
    (void) rope_node;
    (void) inp_pos;
    auto put_f32 = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v, float scale = 1.0f) {
        auto sl = bank_slot.find(n);
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
    auto put_nv_whole = [&](bmap & bm, const std::string & qn, const std::string & sn, const std::string & zn,
                            const ggml_tensor * w, int b, float s2_scale = 1.0f) {
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
        fill_f32(bm, zn, b, s2 * s2_scale);
    };
    // One row of a GGUF NVFP4 tensor -> one row of the kernel's 16-value-group layout.
    auto copy_nv_row = [](const ggml_tensor * w, int64_t e, int64_t r, uint8_t * dq, uint8_t * ds) {
        const int64_t K = w->ne[0];
        const uint8_t * src = (const uint8_t *) w->data + (size_t) e * w->nb[2] + (size_t) r * w->nb[1];
        for (int64_t g = 0; g < K / 16; ++g) {
            const uint8_t * blk = src + (size_t) (g / 4) * 36;
            ds[g]               = blk[g % 4];
            const uint8_t * b   = blk + 4 + (size_t) (g % 4) * 8;
            uint8_t *       o   = dq + g * 8;
            for (int j = 0; j < 4; ++j) {
                o[j]     = (uint8_t) ((b[2 * j] & 0x0f) | ((b[2 * j + 1] & 0x0f) << 4));
                o[4 + j] = (uint8_t) ((b[2 * j] >> 4) | (b[2 * j + 1] & 0xf0));
            }
        }
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
            axera_page_in(w);
            read.push_back(w);
            return w;
        };
        // One bank holds all six layers of the group; the projection's input
        // name is fixed at the b0 label and block_layer selects the layer.
        const std::string pa = g.pre + "b0_attn_";
        put_f32(bm, pa + "w_norm_attn", b, f32_of(W("attn_norm.weight")));
        put_f32(bm, pa + "w_norm_q", b, f32_of(W("attn_q_norm.weight")), 1.0f / std::sqrt((float) HD));
        put_f32(bm, pa + "w_norm_k", b, f32_of(W("attn_k_norm.weight")));
        put_nv_whole(bm, pa + "w_q", pa + "w_scale_q", pa + "w_scale2_q", W("attn_q.weight"), b);
        put_nv_whole(bm, pa + "w_k", pa + "w_scale_k", pa + "w_scale2_k", W("attn_k.weight"), b);
        put_nv_whole(bm, pa + "w_v", pa + "w_scale_v", pa + "w_scale2_v", W("attn_v.weight"), b);
        put_nv_whole(bm, pa + "w_o", pa + "w_scale_o", pa + "w_scale2_o", W("attn_output.weight"), b);
        for (const char * c : { "q", "k", "v", "o" }) {
            fill_zero(bm, pa + "w_zeropoint_" + c, b);
        }

        const std::string pm = g.pre + "b0_moe_";
        put_f32(bm, pm + "input_norm_w", b, f32_of(W("ffn_norm.weight")));
        put_bf16(bm, pm + "w_route", b, f32_of(W("ffn_gate_inp.weight")));
        fill_f32(bm, pm + "w_scale_route", b, 1.0f);
        fill_f32(bm, pm + "w_scale2_route", b, 1.0f);
        const ggml_tensor * wg = W("ffn_gate_exps.weight");
        const ggml_tensor * wu = W("ffn_up_exps.weight");
        const ggml_tensor * wd = W("ffn_down_exps.weight");
        GGML_ASSERT(wg->type == GGML_TYPE_NVFP4 && wu->type == GGML_TYPE_NVFP4 && wd->type == GGML_TYPE_NVFP4);
        const int64_t I = wg->ne[1];
        GGML_ASSERT(wg->ne[2] == E && wu->ne[2] == E && wd->ne[2] == E && wu->ne[1] == I && wd->ne[1] == H);

        // gate and up share one fused [E, 2I, H] bank; one pool run per
        // projection (not per expert).
        const std::string qg = bank_slot.count(pm + "w_gate") ? pm + "w_gate" : pm + "w_up";
        const std::string sg = bank_slot.count(pm + "w_scale_gate") ? pm + "w_scale_gate" : pm + "w_scale_up";
        const std::string zg = bank_slot.count(pm + "w_scale2_gate") ? pm + "w_scale2_gate" : pm + "w_scale2_up";
        GGML_ASSERT(bank_slot.count(qg) && bank_slot.count(sg) && bank_slot.count(zg));
        uint8_t * qgate = slot(bm, qg, b);
        uint8_t * sgate = slot(bm, sg, b);
        const int64_t qrb = wg->ne[0] / 2, srb = wg->ne[0] / 16;
        axera_host_pool().run(E * I, [&](int64_t idx) {
            const int64_t e = idx / I, r = idx % I;
            copy_nv_row(wg, e, r, qgate + (size_t) ((e * 2) * I + r) * qrb,
                        sgate + (size_t) ((e * 2) * I + r) * srb);
        });
        axera_host_pool().run(E * I, [&](int64_t idx) {
            const int64_t e = idx / I, r = idx % I;
            copy_nv_row(wu, e, r, qgate + (size_t) ((e * 2 + 1) * I + r) * qrb,
                        sgate + (size_t) ((e * 2 + 1) * I + r) * srb);
        });
        float * s2 = (float *) slot(bm, zg, b);
        for (int64_t e = 0; e < E; ++e) {
            const ggml_tensor * gs = nv_scale(t, wg);
            const ggml_tensor * us = nv_scale(t, wu);
            s2[2 * e]     = ((const float *) gs->data)[ggml_nelements(gs) == 1 ? 0 : e];
            s2[2 * e + 1] = ((const float *) us->data)[ggml_nelements(us) == 1 ? 0 : e];
        }
        const std::string qd = pm + "w_down", sd = pm + "w_scale_down", zd = pm + "w_scale2_down";
        uint8_t * qdown = slot(bm, qd, b), * sdown = slot(bm, sd, b);
        const int64_t dqrb = wd->ne[0] / 2, dsrb = wd->ne[0] / 16;
        axera_host_pool().run(E * H, [&](int64_t idx) {
            const int64_t e = idx / H, r = idx % H;
            copy_nv_row(wd, e, r, qdown + (size_t) (e * H + r) * dqrb, sdown + (size_t) (e * H + r) * dsrb);
        });
        float * ds2 = (float *) slot(bm, zd, b);
        for (int64_t e = 0; e < E; ++e) {
            const ggml_tensor * wds = nv_scale(t, wd);
            ds2[e] = ((const float *) wds->data)[ggml_nelements(wds) == 1 ? 0 : e];
        }
        drop(read);
    };

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        bmap & bank = chunks[gi];
        for (int b = 0; b < groups[gi].layers; ++b) {
            fill_layer(groups[gi], bank, b, groups[gi].first + b);
        }
    }

    // final norm and NVFP4 lm_head
    {
        std::vector<const ggml_tensor *> read;
        const ggml_tensor * nw = need(t, "output_norm.weight");
        axera_page_in(nw);
        read.push_back(nw);
        put_f32(gbank, "head_norm", 0, f32_of(nw));
        const ggml_tensor * ow = need(t, "output.weight");
        axera_page_in(ow);
        read.push_back(ow);
        put_nv_whole(gbank, "head_w", "head_s", "head_s2", ow, 0);
        drop(read);
    }

    for (auto & [n, c] : gbank) c->flush();
    for (auto & c : chunks) {
        for (auto & [n, b] : c) b->flush();
    }
}

seq_caps q3vl30::seq_get_caps() const {
    return seq_caps{ PG, IN, 0, cells, 0, 0, true };
}

void q3vl30::bind_calls(std::vector<call> & calls, const kernel & k, const char * tag, bool prefill) {
    (void) tag;
    const int count = prefill ? pre_calls : 1;
    calls.assign((size_t) count, {});
    auto group_of = [](const std::string & n) {
        return n.size() >= 3 && n[0] == 'g' && std::isdigit((unsigned char) n[1]) && n[2] == '_'
                   ? n[1] - '0' : -1;
    };
    auto kv_index = [](const std::string & n) {
        const size_t p = n.find("_kv");
        if (p == std::string::npos) return -1;
        return std::atoi(n.c_str() + p + 3);
    };
    for (int c = 0; c < count; ++c) {
        call & cl = calls[c];
        cl.k      = &k;
        axera_cmm & src = !prefill || c == 0 ? io.at("x")
                          : (c % 2 ? io.at("output") : io.at("hidden2"));
        axera_cmm & dst = !prefill || c % 2 == 0 ? io.at("output") : io.at("hidden2");
        cl.in.resize(k.desc->in.size());
        for (size_t i = 0; i < k.desc->in.size(); ++i) {
            const io_desc & d = k.desc->in[i];
            const axera_cmm * b = nullptr;
            if (d.name == "x") {
                b = &src;
            } else if (d.name == "output") {
                b = &dst;
            } else if (io_name(d.name)) {
                auto it = io.find(d.name);
                if (it == io.end()) {
                    GGML_ABORT("axera q3vl30: %s: no buffer for input %s\n", tag, d.name.c_str());
                }
                b = &it->second;
            } else if (prefill) {
                GGML_ASSERT(group_of(d.name) == 0);
                const int ki = kv_index(d.name);
                if (ki >= 0) {
                    GGML_ASSERT(c * GROUP_LAYERS + ki < LAYERS);
                    cl.in[i] = axera_io(KVL.at((size_t) c * GROUP_LAYERS + ki).get(),
                                        KVL[(size_t) c * GROUP_LAYERS + ki]->vir, d.bytes);
                    continue;
                }
                const std::string mapped = "g" + std::to_string(c) + "_" + d.name.substr(3);
                b = &at(chunks[(size_t) c], mapped);
            } else {
                const int gi = group_of(d.name);
                if (gi >= 0) {
                    const int ki = kv_index(d.name);
                    if (ki >= 0) {
                        const int j = gi * GROUP_LAYERS + ki;
                        GGML_ASSERT(j < LAYERS);
                        cl.in[i] = axera_io(KVL[(size_t) j].get(), KVL[(size_t) j]->vir, d.bytes);
                        continue;
                    }
                    b = &at(chunks[(size_t) gi], d.name);
                } else {
                    b = &at(gbank, d.name);
                }
            }
            cl.in[i] = axera_io(*b, d.bytes);
        }
        cl.out.resize(k.desc->out.size());
        for (size_t i = 0; i < k.desc->out.size(); ++i) {
            const io_desc & d = k.desc->out[i];
            if (d.name == "output") {
                cl.out[i] = axera_io(dst, d.bytes);
            } else {
                auto it = io.find(d.name);
                if (it == io.end()) {
                    GGML_ABORT("axera q3vl30: %s: no buffer for output %s\n", tag, d.name.c_str());
                }
                cl.out[i] = axera_io(it->second, d.bytes);
            }
        }
    }
}

void q3vl30::open_route_dump() {
    const char * p = getenv("GGML_AXERA_HY_MT2_ROUTE_DUMP");
    if (!p || !*p) return;
    route_file = fopen(p, "wb");
    if (!route_file) {
        GGML_LOG_WARN("axera q3vl30: cannot open route dump %s\n", p);
    }
}

void q3vl30::dump_routes(int64_t cell, int rows, int layer_stride) {
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
        axera_cmm & r = route_cmm[0];
        r->invalidate(r->vir, r->size);
        const int32_t * src = (const int32_t *) r->vir + (size_t) (il - 1) * layer_stride;
        fwrite(src, sizeof(int32_t), (size_t) rows * 8, route_file);
    }
    fflush(route_file);
}

void q3vl30::dump_x(const axera_cmm & b, int rows, int64_t cell, int call_index) {
    const char * p = getenv("GGML_AXERA_HY_MT2_X_DUMP");
    if (!p || !*p) return;
    const size_t n = (size_t) rows * H * sizeof(uint16_t);
    b->invalidate(b->vir, n);
    FILE * f = fopen(p, "ab");
    if (!f) return;
    const uint32_t magic = 0x58334854u;  // "TH3X"
    const int32_t  h[4] = { rows, (int32_t) cell, call_index, 0 };
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(h, sizeof(h), 1, f);
    fwrite(b->vir, 1, n, f);
    fclose(f);
}

void q3vl30::dump_hidden(const axera_cmm & b, int rows, int64_t cell, int call_index) {
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

static uint64_t dump_hash(const void * p, size_t n) {
    const uint8_t * s = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool debug_call_wanted(int ci) {
    const char * p = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_CALLS");
    if (!p || !*p) return true;
    const std::string s = "," + std::string(p) + ",";
    return s.find("," + std::to_string(ci) + ",") != std::string::npos;
}

void q3vl30::dump_debug_call(int ci, int64_t cell, int rows, int64_t batch_cell0, int page,
                             int first, int want, int take, int hr0, bool need_head) {
    const char * base = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_PATH");
    if (!base || !*base || !debug_call_wanted(ci)) return;
    const char * rs = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_ROWS");
    if (rs && *rs && atoi(rs) != rows) return;
    const char * cs = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_CELL");
    if (cs && *cs && strtoll(cs, nullptr, 10) != cell) return;

    mkdir(base, 0777);
    static uint64_t serial = 0;
    const uint64_t id = serial++;
    char pdir[4096];
    snprintf(pdir, sizeof(pdir), "%s/page_%06lld_rows_%04d_seq_%05llu", base, (long long) cell, rows,
             (unsigned long long) id);
    mkdir(pdir, 0777);
    char dir[4096];
    snprintf(dir, sizeof(dir), "%s/call_%d", pdir, ci);
    mkdir(dir, 0777);

    char path[4096];
    snprintf(path, sizeof(path), "%s/meta.txt", dir);
    FILE * mf = fopen(path, "w");
    if (!mf) {
        fprintf(stderr, "axera q3vl30: debug dump fopen %s failed\n", path);
        return;
    }
    fprintf(mf, "time_us=%lld\ncell=%lld\nrows=%d\nbatch_cell0=%lld\npage=%d\ncall=%d\n"
                "first=%d\nwant=%d\ntake=%d\nhr0=%d\nneed_head=%d\n",
            (long long) ggml_time_us(), (long long) cell, rows, (long long) batch_cell0, page, ci,
            first, want, take, hr0, need_head ? 1 : 0);
    auto scalar = [&](const char * name) -> int32_t {
        auto it = io.find(name);
        return it == io.end() ? 0 : *(const int32_t *) it->second->vir;
    };
    fprintf(mf, "head_run=%d\nlayer0_run=%d\nbody_run=%d\nvalid_tokens=%d\nlast_page_len=%d\n"
                "indptr=[%d,%d]\ncache_location=[%d,%d]\n",
            scalar("head_run"), scalar("layer0_run"), scalar("body_run"), scalar("valid_tokens"),
            scalar("last_page_len"), ((const int32_t *) io.at("indptr")->vir)[0],
            ((const int32_t *) io.at("indptr")->vir)[1], ((const int32_t *) io.at("cache_location")->vir)[0],
            ((const int32_t *) io.at("cache_location")->vir)[1]);
    fprintf(mf, "indices=");
    const int32_t * ix = (const int32_t *) io.at("indices")->vir;
    for (int i = 0; i <= page + wp / PG; ++i) fprintf(mf, "%s%d", i ? "," : "", ix[i]);
    fprintf(mf, "\nhead_rows=");
    const int32_t * hr = (const int32_t *) io.at("head_rows")->vir;
    for (int i = 0; i < HEAD_ROWS; ++i) fprintf(mf, "%s%d", i ? "," : "", hr[i]);
    fprintf(mf, "\n");

    static const char * small_io[] = {
        "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
        "head_x", "head_rows", "head_run", "layer0_run", "body_run", "valid_tokens",
    };
    for (size_t i = 0; i < pre[ci].k->desc->in.size(); ++i) {
        const io_desc & d = pre[ci].k->desc->in[i];
        bool small = false;
        for (const char * n : small_io) small = small || d.name == n;
        if (small) {
            snprintf(path, sizeof(path), "%s/%s.bin", dir, d.name.c_str());
            FILE * f = fopen(path, "wb");
            GGML_ASSERT(f && pre[ci].in[i].pVirAddr);
            fwrite(pre[ci].in[i].pVirAddr, 1, d.bytes, f);
            fclose(f);
            fprintf(mf, "io %s bytes=%u hash=%016llx file=%s.bin\n", d.name.c_str(), d.bytes,
                    (unsigned long long) dump_hash(pre[ci].in[i].pVirAddr, d.bytes), d.name.c_str());
        }
    }

    const bool dump_kv = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_KV") == nullptr ||
                         atoi(getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_KV")) != 0;
    if (dump_kv) {
        std::vector<int> pages;
        // A 4096-row prefill call writes wp/PG sub-pages.  `page` is only the
        // first one (the replay's history end), so dumping 0..page saved just
        // page 0 and made a short-page replay depend on reconstructed pages.
        // Save every page the call's `indices` can touch, capped by the bank.
        const int last_page = std::min(PAGES - 1, page + wp / PG);
        for (int p = 0; p <= last_page; ++p) pages.push_back(p);
        for (int p : pages) {
            for (size_t i = 0; i < pre[ci].k->desc->in.size(); ++i) {
                const io_desc & d = pre[ci].k->desc->in[i];
                if (d.name.size() < 3 || d.name.compare(d.name.size() - 3, 3, "_kv") != 0) continue;
                // The KV input is [layers, PAGES, 2, 256, 4, 128], not a flat
                // page array.  The old bytes/PAGES slice was 12 pages of layer
                // 0 for g1, so layers 2..12 were never dumped.  A per-layer
                // page slice is 2*256*4*128*2 = 512 KiB.
                const size_t page_bytes = (size_t) 2 * 256 * 4 * 128 * sizeof(uint16_t);
                GGML_ASSERT(d.bytes % (page_bytes * (size_t) PAGES) == 0);
                const size_t layers = d.bytes / (page_bytes * (size_t) PAGES);
                for (size_t l = 0; l < layers; ++l) {
                    snprintf(path, sizeof(path), "%s/%s.layer_%02zu.page_%03d.bin", dir, d.name.c_str(), l, p);
                    const uint8_t * src = (const uint8_t *) pre[ci].in[i].pVirAddr +
                                          (((size_t) l * (size_t) PAGES + (size_t) p) * page_bytes);
                    FILE * f = fopen(path, "wb");
                    GGML_ASSERT(f);
                    fwrite(src, 1, page_bytes, f);
                    fclose(f);
                    fprintf(mf, "kv %s layer=%zu page=%d bytes=%zu hash=%016llx file=%s.layer_%02zu.page_%03d.bin\n",
                            d.name.c_str(), l, p, page_bytes, (unsigned long long) dump_hash(src, page_bytes),
                            d.name.c_str(), l, p);
                }
                continue;
            }
        }
    }
    for (size_t i = 0; i < pre[ci].k->desc->in.size(); ++i) {
        const io_desc & d = pre[ci].k->desc->in[i];
        bool small = false;
        for (const char * n : small_io) small = small || d.name == n;
        if (!small && !(dump_kv && d.name.size() >= 3 && d.name.compare(d.name.size() - 3, 3, "_kv") == 0)) {
            fprintf(mf, "weight %s bytes=%u\n", d.name.c_str(), d.bytes);
        }
    }
    fclose(mf);
    GGML_LOG_INFO("axera q3vl30: debug dump call cell %lld rows %d call %d -> %s\n",
                  (long long) cell, rows, ci, dir);
}

void q3vl30::dump_debug_output(int ci, int rows, int64_t cell) {
    const char * base = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_PATH");
    const char * out = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_OUT");
    if (!base || !*base || !out || atoi(out) == 0 || !debug_call_wanted(ci)) return;
    const char * rs = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_ROWS");
    if (rs && *rs && atoi(rs) != rows) return;
    const char * cs = getenv("GGML_AXERA_HY_MT2_DEBUG_DUMP_CELL");
    if (cs && *cs && strtoll(cs, nullptr, 10) != cell) return;
    static uint64_t serial = 0;  // shares the call order with dump_debug_call
    const uint64_t id = serial++;
    char dir[4096], path[4096];
    snprintf(dir, sizeof(dir), "%s/page_%06lld_rows_%04d_seq_%05llu/call_%d", base, (long long) cell, rows,
             (unsigned long long) id, ci);
    for (const auto & [name, cmm] : io) {
        if (name != "output" && name != "hidden2" && name != "head_logits") continue;
        const size_t n = name == "head_logits" ? (size_t) HEAD_ROWS * VOCAB * sizeof(float) : (size_t) rows * H * sizeof(uint16_t);
        if (n > cmm->size) continue;
        cmm->invalidate(cmm->vir, n);
        snprintf(path, sizeof(path), "%s/out_%s.bin", dir, name.c_str());
        FILE * f = fopen(path, "wb");
        if (!f) continue;
        fwrite(cmm->vir, 1, n, f);
        fclose(f);
    }
}

// llama.cpp's GGML_ROPE_TYPE_IMROPE cache for one token: 64 pairs, sections [24,20,20,0].
static void mrope_cache(float * cs, float * sn, const int32_t * p) {
    const float scale = std::pow(5000000.0f, -2.0f / (float) HD);
    float tt = (float) p[0], th = (float) p[1], tw = (float) p[2], te = (float) p[3];
    for (int i0 = 0; i0 < HD; i0 += 2) {
        const int sector = (i0 / 2) % 64;
        float theta;
        if (sector % 3 == 1 && sector < 60) {
            theta = th;
        } else if (sector % 3 == 2 && sector < 60) {
            theta = tw;
        } else if (sector % 3 == 0 && sector < 72) {
            theta = tt;
        } else {
            theta = te;
        }
        cs[i0 / 2] = std::cos(theta);
        sn[i0 / 2] = std::sin(theta);
        tt *= scale;
        th *= scale;
        tw *= scale;
        te *= scale;
    }
}

void q3vl30::run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                              bool need_head) {
    GGML_ASSERT(cell % PG == 0 && rows > 0 && rows <= wp);
    auto xi = io.find("x"), oi = io.find("output"), dsi = io.find("deepstack");
    GGML_ASSERT(xi != io.end() && oi != io.end() && dsi != io.end());
    uint16_t * xb = (uint16_t *) xi->second->vir;
    uint16_t * db = (uint16_t *) dsi->second->vir;
    axera_host_pool().run(wp, [&](int64_t r) {
        const int64_t s = r < rows ? r : r % rows;
        memcpy(xb + (size_t) r * H, x + (size_t) s * IN, (size_t) H * sizeof(uint16_t));
        for (int p = 0; p < 3; ++p) {
            memcpy(db + ((size_t) p * wp + r) * H, x + (size_t) s * IN + (size_t) (p + 1) * H,
                   (size_t) H * sizeof(uint16_t));
        }
    });
    xi->second->flush();
    dsi->second->flush();
    dump_x(xi->second, rows, cell, 0);

    auto pi = io.find("position_ids");
    int32_t * pos = (int32_t *) pi->second->vir;
    for (int r = 0; r < wp; ++r) pos[r] = r;
    pi->second->flush();
    auto ci = io.find("cos"), si = io.find("sin");
    GGML_ASSERT(ci != io.end() && si != io.end());
    float * cs = (float *) ci->second->vir;
    float * sn = (float *) si->second->vir;
    std::vector<int32_t> pp((size_t) wp * 4);
    for (int r = 0; r < rows; ++r) {
        memcpy(pp.data() + (size_t) r * 4, p4 + (size_t) r * 4, 4 * sizeof(int32_t));
    }
    for (int r = rows; r < wp; ++r) {
        pp[(size_t) r * 4 + 0] = (int32_t) (cell + r);
        pp[(size_t) r * 4 + 1] = 0;
        pp[(size_t) r * 4 + 2] = 0;
        pp[(size_t) r * 4 + 3] = 0;
    }
    axera_host_pool().run(wp, [&](int64_t r) {
        mrope_cache(cs + (size_t) r * (HD / 2), sn + (size_t) r * (HD / 2), pp.data() + (size_t) r * 4);
    });
    ci->second->flush();
    si->second->flush();

    const int page = (int) (cell / PG);
    auto ind = io.find("indptr"), idx = io.find("indices"), last = io.find("last_page_len"), cl = io.find("cache_location");
    auto vt = io.find("valid_tokens"), dsr = io.find("ds_run");
    ((int32_t *) ind->second->vir)[0] = 0;
    ((int32_t *) ind->second->vir)[1] = page;
    GGML_ASSERT((size_t) (page + wp / PG) * sizeof(int32_t) <= idx->second->size);
    int32_t * ix = (int32_t *) idx->second->vir;
    for (int i = 0; i <= page + wp / PG; ++i) ix[i] = i;
    *(int32_t *) last->second->vir = 0;
    ((int32_t *) cl->second->vir)[0] = page;
    ((int32_t *) cl->second->vir)[1] = 0;
    *(int32_t *) vt->second->vir = rows;
    if (dsr != io.end()) {
        *(uint16_t *) dsr->second->vir = 0x3f80;  // bf16 1.0
    }
    for (auto * b : { &xi->second, &pi->second, &ind->second, &idx->second, &last->second, &cl->second, &vt->second,
                      &ci->second, &si->second, &dsi->second }) {
        (*b)->flush();
    }
    if (dsr != io.end()) {
        dsr->second->flush();
    }

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

    const bool prof = getenv("GGML_AXERA_Q3VL30_PROF") != nullptr;
    const bool dump = getenv("GGML_AXERA_Q3VL30_HIDDEN_DUMP") != nullptr;
    for (int k = 0; k < (int) pre.size(); ++k) {
        const int64_t t0 = prof ? ggml_time_us() : 0;
        pre[k].k->run(pre[k].in, pre[k].out, "prefill");
        if (prof) {
            GGML_LOG_INFO("axera q3vl30: prefill call %d cell %lld rows %d kernel %.3f ms\n", k, (long long) cell,
                          rows, (ggml_time_us() - t0) / 1e3);
        }
        if (dump) {
            const axera_cmm & b = k % 2 == 0 ? io.at("output") : io.at("hidden2");
            dump_hidden(b, rows, cell, k);
        }
    }

    head_first = (int) (cell + hr0 - batch_cell0);
    head_n = take;
    small_head = false;
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, (size_t) HEAD_ROWS * VOCAB * sizeof(float));
}

void q3vl30::run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4) {
    auto xi = io.find("x"), oi = io.find("output");
    GGML_ASSERT(xi != io.end() && oi != io.end());
    memcpy(xi->second->vir, x, H * sizeof(uint16_t));
    xi->second->flush();
    auto pi = io.find("position_ids");
    *(int32_t *) pi->second->vir = 0;
    pi->second->flush();
    auto ci = io.find("cos"), si = io.find("sin");
    GGML_ASSERT(ci != io.end() && si != io.end());
    mrope_cache((float *) ci->second->vir, (float *) si->second->vir, p4);
    ci->second->flush();
    si->second->flush();

    const int page = (int) (cell / PG), off = (int) (cell % PG);
    auto ind = io.find("indptr"), idx = io.find("indices"), last = io.find("last_page_len"), cl = io.find("cache_location");
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
    for (auto * b : { &pi->second, &ind->second, &idx->second, &last->second, &cl->second, &ci->second, &si->second }) {
        (*b)->flush();
    }
    auto hx = io.find("head_x"), hri = io.find("head_rows");
    memset(hx->second->vir, 0, hx->second->size);
    *(int32_t *) hri->second->vir = 0;
    hri->second->flush();
    auto head_run = io.find("head_run");
    GGML_ASSERT(head_run != io.end());
    *(int32_t *) head_run->second->vir = 1;
    head_run->second->flush();

    GGML_ASSERT(dec.size() == 1);
    const bool prof = getenv("GGML_AXERA_Q3VL30_PROF") != nullptr;
    const int64_t t0 = prof ? ggml_time_us() : 0;
    dec[0].k->run(dec[0].in, dec[0].out, "decode");
    if (prof) {
        GGML_LOG_INFO("axera q3vl30: decode cell %lld kernel %.3f ms\n", (long long) cell,
                      (ggml_time_us() - t0) / 1e3);
    }
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, VOCAB * sizeof(float));
    head_first = 0;
    head_n = 1;
    small_head = false;
}

bool q3vl30::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const uint16_t * xr = x;
    if (!xr) {
        GGML_ASSERT(batch0 == 0);
        xbuf.resize((size_t) n * IN);
        seq_rows(0, n, (uint16_t *) xbuf.data());
        xr = xbuf.data();
    }
    for (int off = 0; off < n; off += wp) {
        const int rows = std::min(wp, n - off);
        run_prefill_page(c0 + off, rows, xr + (size_t) off * IN, p4 + (size_t) off * 4, c0 + batch0,
                         off + wp >= n);
    }
    if (snap) snap->deferred = false;
    return true;
}

bool q3vl30::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
    run_decode_row(c, x, p4);
    return true;
}

void q3vl30::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) return;
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * IN, (size_t) n * IN * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * IN * sizeof(uint16_t));
    }
}

void q3vl30::publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T) {
    GGML_ASSERT(logits && logits->type == GGML_TYPE_F32 && logits->ne[0] == VOCAB);
    const int n_out = (int) logits->ne[1];
    if (n_out == 0) {
        return;  // a warm-up or an image-only batch asks for no logits
    }
    if (n_out > head_n) {
        GGML_ABORT("axera q3vl30: %d requested logits rows but the page head computed %d (keep -ub <= %d)\n",
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
    if (n_out < T && !out_ids) GGML_ABORT("axera q3vl30: %d of %d output rows without row indices\n", n_out, T);
    const axera_cmm & hl = io.at("head_logits");
    const float *    head_src = small_head ? small_logits.data() : (const float *) hl->vir;
    auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        if (row < head_first || row >= head_first + head_n) {
            GGML_ABORT("axera q3vl30: output row %d outside the head's %d..%d\n", row, head_first,
                       head_first + head_n - 1);
        }
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        memcpy(o, head_src + (size_t) (row - head_first) * VOCAB, VOCAB * sizeof(float));
        dctx->flush(o, VOCAB * sizeof(float));
        dctx->clean_p = o;
        dctx->clean_n = VOCAB * sizeof(float);
        dctx->clean_epoch = axera_graph_epoch;
    }
    output_head::after(g, names);
}

ggml_status q3vl30::compute(ggml_cgraph * g) {
    const auto & t = names.of(g);
    ggml_tensor * norm0 = names.find("attn_norm-0");
    ggml_tensor * logits = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera q3vl30: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) x0 = x0->src[0];
    GGML_ASSERT(x0 && x0->src[0] && x0->src[0]->type == GGML_TYPE_F32);
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) ++n0;
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0 = x0->src[0];
    // Qwen3-VL's token input is a [H, T] view of the [H + 3 * H, T] embedding
    // tensor: the three deepstack planes follow the token row and llama.cpp
    // zeroes them for text tokens. Read all IN floats of each row.
    const int Hh = (int) x0->ne[0], T = (int) x0->ne[1];
    GGML_ASSERT(Hh == H && x0->nb[1] >= (size_t) IN * sizeof(float));
    int Ll = 0;
    while (t.count("blk." + std::to_string(Ll) + ".attn_norm.weight")) ++Ll;
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || Ll != L || Hh != H_) {
        L = Ll; H_ = Hh;
        GGML_ASSERT(L == 48 && H_ == H && Hh == H);
        init(t, nullptr, inp_pos);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, cells, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera q3vl30: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera q3vl30: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t * pos = (const int32_t *) inp_pos->data;
    const bool mr = inp_pos->ne[0] == 4 * (int64_t) T;
    pbuf.resize((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) pbuf[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
    }
    std::vector<ggml_bf16_t> xb((size_t) T * IN);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                              xb.data() + (size_t) r * IN, IN);
    });
    if (const char * ip = getenv("GGML_AXERA_HY_MT2_IDS_DUMP"); ip && *ip) {
        FILE * idf = fopen(ip, "wb");
        if (idf) {
            const uint32_t magic = 0x49334854u;  // "TH3I"
            const int32_t  h[4] = { T, 0, 0, 0 };
            fwrite(&magic, sizeof(magic), 1, idf);
            fwrite(h, sizeof(h), 1, idf);
            if (x0->op == GGML_OP_GET_ROWS && x0->src[1] && x0->src[1]->type == GGML_TYPE_I32) {
                fwrite(x0->src[1]->data, sizeof(int32_t), (size_t) T, idf);
            }
            fwrite((const char *) x0->data + (size_t) (T - 1) * x0->nb[1], sizeof(float), (size_t) H, idf);
            fwrite(xb.data() + (size_t) (T - 1) * IN, sizeof(uint16_t), (size_t) IN, idf);
            fclose(idf);
        }
    }
    batch_rows = xb.data();
    last_seq_status = 0;
    const int64_t c0 = kv.cell_of(0);
    const seq_status sst = seq->feed(c0, T, (const uint16_t *) xb.data(), pbuf.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) sst;
        if (sst == SEQ_MISS) {
            GGML_LOG_WARN("axera q3vl30: batch from cell %lld has no restart point (%s); the slot must replay\n",
                          (long long) c0, seq->why());
        } else {
            GGML_LOG_ERROR("axera q3vl30: sequence state rejected a batch from cell %lld: %s\n",
                           (long long) c0, seq->why());
        }
        return GGML_STATUS_FAILED;
    }
    publish_logits(g, logits, T);
    return GGML_STATUS_SUCCESS;
}

size_t q3vl30::state_size(uint32_t flags) const {
    if (!inited || !seq) return 0;
    const bool partial = flags & state_flags_partial_only;
    const size_t seq_n = seq->size(partial);
    if (!seq_n) return 0;
    const int64_t cell = std::max<int64_t>(0, seq->cell());
    const int nkv = partial ? 0 : (int) std::min<int64_t>(kv_pages, (cell + PG - 1) / PG);
    return sizeof(state_header) + (uint64_t) LAYERS * nkv * kv_page_b + seq_n;
}

size_t q3vl30::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    const size_t total = state_size(flags);
    if (!total || size < total) return 0;
    const bool partial = flags & state_flags_partial_only;
    const size_t seq_n = seq->size(partial);
    const int64_t cell = std::max<int64_t>(0, seq->cell());
    const int nkv = partial ? 0 : (int) std::min<int64_t>(kv_pages, (cell + PG - 1) / PG);

    state_header h{};
    memcpy(h.magic, "Q3VL30S1", state_magic_n - 1);
    h.version = 1;
    h.partial = partial;
    h.ngroups = (uint32_t) GROUPS;
    h.state_cell = seq->cell();
    h.nkv_pages = (uint32_t) nkv;
    h.kv_page_b = (uint32_t) kv_page_b;
    h.kv_n = (uint64_t) LAYERS * nkv * kv_page_b;
    h.seq_n = seq_n;
    char * p = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (!partial) {
        const size_t n = (size_t) nkv * kv_page_b;
        for (int j = 0; j < LAYERS; ++j) {
            axera_cmm & c = KVL[(size_t) j];
            c->invalidate(c->vir, n);
            memcpy(p, c->vir, n);
            p += n;
        }
    }
    if (seq->write((uint8_t *) p, seq_n, partial) != seq_n) return 0;
    p += seq_n;
    return (size_t) (p - (char *) dst);
}

size_t q3vl30::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    if (size < sizeof(state_header)) return 0;
    const bool partial = flags & state_flags_partial_only;
    state_header h{};
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, "Q3VL30S1", state_magic_n - 1) != 0 || h.version != 1 || h.partial != partial ||
        h.ngroups != (uint32_t) GROUPS || !h.seq_n || h.state_cell < -1 || h.state_cell > cells ||
        h.nkv_pages > (uint32_t) kv_pages) {
        return 0;
    }
    const uint64_t kv_n = partial ? 0 : (uint64_t) LAYERS * h.nkv_pages * kv_page_b;
    if (h.kv_page_b != kv_page_b || h.kv_n != kv_n || size < sizeof(h) + kv_n + h.seq_n) return 0;
    const char * p = (const char *) src + sizeof(h);
    if (!partial) {
        const size_t n = (size_t) h.nkv_pages * kv_page_b;
        for (int j = 0; j < LAYERS; ++j) {
            axera_cmm & c = KVL[(size_t) j];
            memcpy(c->vir, p, n);
            c->flush(c->vir, n);
            p += n;
        }
    }
    if (seq->read((const uint8_t *) p, h.seq_n) != h.seq_n) return 0;
    p += h.seq_n;
    return (size_t) (p - (const char *) src);
}

}  // namespace

std::shared_ptr<model> make_q3vl30(const manifest & m) { return std::make_shared<q3vl30>(m); }

}  // namespace axera
