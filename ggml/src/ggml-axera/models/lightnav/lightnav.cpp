// Axera NPU backend: LightNav-0 (Qwen3-VL-4B dense text tower) on 8-bit MXFP8 or NVFP4 group kernels
// (H 2560, 32 q / 8 kv heads of 128, dense FFN 9728, 36 layers in six groups of six).
//
// The manifest has two merged kernels:
//   decode   all 36 layers statically expanded as six weight groups g0_..g5_ of six layers for one token,
//            then the final RMSNorm and quantized output head.
//   prefill  a 256-row workspace whose graph holds one reusable 6-layer dense body.  The host launches it
//            once per decode group, rebinding the body to that group's banks; the last call computes the
//            head for the wanted rows.
// The weights come from a GGUF quantized once to MXFP8 (E4M3 codes, one E8M0 scale per 32 inputs) or to
// standard GGML NVFP4 blocks with one F32 .scale sidecar per quantization slice.  Loading converts any
// GGUF type into the descriptor's layout: MXFP8 when the kernel asks for E4M3 codes and E8M0 scales,
// otherwise NVFP4.  NVFP4 copies the codewords and the per-16-input UE4M3 scales and uses the sidecar as
// the kernels' FP32 scale2.
//
// What the manifest's descriptors decide:
//   KV      g<i>_kv<j>: one BF16 [pages][2][256][8][128] pool per layer, allocated from -c.
//   RoPE    a [rows][64] float32 cos / sin input: llama.cpp's IMROPE for each row's four positions.
//   planes  a deepstack input: three more H planes after each token row, added by the first call's
//           layers 1..3 (Qwen3-VL); later calls see zero planes.
//   norms   a norm bank the prefill graph takes in BF16 and decode in F32 gets a BF16 mirror per group.
// seqstate owns page restart points and replay; there is no recurrent state.

#include "../model.h"
#include "../../core/seqstate.h"
#include "../../core/quant.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace axera {
namespace {

constexpr int PG = 256;
constexpr int WP_MAX = 256;
constexpr int H = 2560;
constexpr int KVH = 8;
constexpr int HD = 128;
constexpr int E = 128;  // unused: the model is dense; kept for the shared fill helpers
constexpr int LAYERS = 36;
constexpr int PLANES = 3;  // Qwen3-VL's deepstack planes
constexpr int KV_HEADROOM = WP_MAX / PG;  // pages a 4096-row prefill call writes from its first page

// what the descriptors do not say
struct moe_family {
    const char * model;
    const char * magic;
    uint32_t     version;
    int          small_batch;  // a prefill batch below this replays rows through decode (0: never)
};

constexpr moe_family families[] = {
    { "lightnav", "LNAV0001", 1, 0 },
};

using call = npu_call;

struct group_info {
    int         first = 0;
    int         layers = 0;
    bool        dense = false;
    std::string pre;  // g<i>_
};

// g<digit>_...: the group of a layer bank; -1 otherwise
static int group_of(const std::string & n) {
    return n.size() >= 3 && n[0] == 'g' && std::isdigit((unsigned char) n[1]) && n[2] == '_' ? n[1] - '0' : -1;
}

// g<i>_kv<j>: a per-layer KV pool input; j or -1
static int pool_kv(const std::string & n) {
    return group_of(n) >= 0 && n.compare(3, 2, "kv") == 0 && n.size() > 5 && std::isdigit((unsigned char) n[5])
               ? std::atoi(n.c_str() + 5) : -1;
}

class lightnav final : public model, public seq_model {
  public:
    explicit lightnav(const manifest & m) : man(m) {
        for (const auto & f : families) {
            if (m.model == f.model) fam = &f;
        }
        GGML_ASSERT(fam);
        tag = std::string("axera ") + fam->model;
    }

    ggml_status compute(ggml_cgraph * g) override;
    size_t      state_size(uint32_t flags) const override;
    size_t      state_write(void * dst, size_t size, uint32_t flags) override;
    size_t      state_read(const void * src, size_t size, uint32_t flags) override;
    int         last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override { return seq_caps{ PG, IN, 0, cells, 0, 0, true }; }
    void     seq_state_save(uint8_t * dst) override { (void) dst; }
    void     seq_state_load(const uint8_t * src) override { (void) src; }
    bool seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void seq_rows(int batch_row0, int n, uint16_t * dst) override;

  private:
    const manifest &   man;
    const moe_family * fam = nullptr;
    std::string        tag;
    graph_names        names;
    const void *       bound = nullptr;
    int                L = 0;
    int64_t            cells = 0;

    // from the descriptors
    int  IN = H;            // a token row: x, then the deepstack planes
    int  VOCAB = 0;
    int  head_rows_n = 0;   // the prefill head's rows
    bool rope_table = false;
    bool kv_pools = false;

    using bmap = std::map<std::string, axera_cmm>;
    bmap                    gbank;   // head weights (and Hy-MT2's RoPE tables)
    std::vector<bmap>       chunks;  // one bank map per decode group
    bmap                    pre_bf16;  // BF16 mirrors of the norm banks the prefill graph takes in BF16
    std::vector<axera_cmm>  KVL;     // the per-layer KV pools
    std::vector<std::pair<const axera_cmm *, size_t>> kv_layer;  // each layer's KV: pool or bank, offset
    int                     kv_pages = 0;
    size_t                  kv_page_b = 0;
    bmap                    io;
    std::map<std::string, uint32_t> bank_n, bank_dtype, bank_slot;
    std::vector<group_info> groups;   // decode / weight groups
    std::vector<group_info> pgroups;  // prefill graph groups
    int                     pre_calls = 1;
    int                     body0 = 0;  // the first decode group the prefill body maps to
    int                     wp = WP_MAX;

    std::vector<call>      pre, dec;
    std::vector<axera_cmm> route_cmm;  // the kernels' top-k expert ids

    std::vector<uint16_t>     xbuf;
    std::vector<int32_t>      pbuf;
    const ggml_bf16_t *       batch_rows = nullptr;
    bool                      inited = false;
    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;

    // the output head computed by the last kernel call
    int head_first = 0, head_n = 0;
    // a small batch replayed through decode collects its per-row logits here
    std::vector<float> small_logits;
    bool               small_head = false;

    bool io_name(const std::string & n) const {
        return n == "x" || n == "position_ids" || n == "indptr" || n == "indices" || n == "last_page_len" ||
               n == "cache_location" || n == "head_x" || n == "head_rows" || n == "head_run" ||
               n == "layer0_run" || n == "body_run" || n == "deepstack" || n == "ds_run" ||
               (!rope_table && (n == "cos" || n == "sin")) || n == "head_logits" || n == "head_ids" ||
               n == "head_topk_v" || n == "head_topk_i" || n == "head_vmax" || n == "head_sumexp" ||
               n == "valid_tokens";
    }
    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
              const ggml_tensor * inp_pos);
    void fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                   const ggml_tensor * inp_pos);
    void bind_calls(std::vector<call> & calls, const kernel & k, bool prefill);
    void run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                          bool need_head);
    void run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4);
    void publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T);
    kv_state_format state_format() const;
    cmm_kv_pools    state_pools() const;

    const ggml_tensor * need(const std::map<std::string, ggml_tensor *> & t, const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            GGML_ABORT("%s: %s is not in the graph\n", tag.c_str(), n.c_str());
        }
        return it->second;
    }
    const ggml_tensor * nv_scale(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * w) const {
        std::string n = w->name;
        const std::string suffix = ".weight";
        if (n.size() >= suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            n.resize(n.size() - suffix.size());
        }
        n += ".scale";
        auto it = t.find(n);
        if (it == t.end()) {
            GGML_ABORT("%s: NVFP4 tensor %s has no F32 .scale sidecar\n", tag.c_str(), w->name);
        }
        GGML_ASSERT(it->second->type == GGML_TYPE_F32 && ggml_nelements(it->second) >= 1);
        return it->second;
    }
    const axera_cmm & at(const bmap & b, const std::string & name) const {
        auto i = b.find(name);
        if (i == b.end()) {
            GGML_ABORT("%s: %s is not a bank of this call\n", tag.c_str(), name.c_str());
        }
        return i->second;
    }
    uint8_t * slot(const bmap & b, const std::string & name, int layer) const {
        auto i = b.find(name);
        auto s = bank_slot.find(name);
        if (i == b.end() || s == bank_slot.end()) {
            GGML_ABORT("%s: %s is not a layer bank\n", tag.c_str(), name.c_str());
        }
        return (uint8_t *) i->second->vir + (size_t) layer * s->second;
    }
};

static const io_desc * find_io(const std::vector<io_desc> & v, const std::string & n) {
    auto it = std::find_if(v.begin(), v.end(), [&](const io_desc & d) { return d.name == n; });
    return it == v.end() ? nullptr : &*it;
}

void lightnav::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                   const ggml_tensor * inp_pos) {
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");

    const io_desc * hl = find_io(dk.desc->in, "head_logits");
    const io_desc * phr = find_io(pk.desc->in, "head_rows");
    const io_desc * cs = find_io(dk.desc->in, "cos");
    GGML_ASSERT(hl && hl->shape.size() == 2 && phr && cs && !cs->shape.empty());
    VOCAB       = (int) hl->shape[1];
    head_rows_n = (int) (phr->bytes / sizeof(int32_t));
    rope_table  = false;  // this kernel always takes a per-call cos / sin table in float32
    kv_pools    = find_io(dk.desc->in, "g0_kv0") != nullptr;
    IN          = find_io(pk.desc->in, "deepstack") ? (1 + PLANES) * H : H;

    auto scan_groups = [&](const kernel & k, std::vector<group_info> & out) {
        out.clear();
        int first = 0;
        for (int gi = 0; gi < 10; ++gi) {
            const std::string pre = "g" + std::to_string(gi) + "_";
            const io_desc * q = find_io(k.desc->in, pre + "b0_attn_w_q");
            if (!q || q->shape.empty()) {
                break;
            }
            const int n = (int) q->shape[0];
            out.push_back({ first, n, find_io(k.desc->in, pre + "b0_mlp_w_norm") != nullptr, pre });
            first += n;
        }
    };
    scan_groups(pk, pgroups);
    scan_groups(dk, groups);
    int grouped = 0;
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        GGML_ASSERT(groups[gi].layers > 0 && groups[gi].dense);
        grouped += groups[gi].layers;
    }
    GGML_ASSERT(grouped == LAYERS);
    // every group is dense: the prefill graph holds one 6-layer body and each of its calls binds to the
    // next decode group's banks
    body0 = 0;
    GGML_ASSERT((int) pgroups.size() == 1 && pgroups[0].dense);
    const group_info & body = pgroups.back();
    pre_calls = (int) groups.size();
    for (int gi = body0; gi < (int) groups.size(); ++gi) {
        GGML_ASSERT(groups[gi].layers == body.layers);
    }
    auto pre_group = [&](int c, const std::string & n) {  // the decode group call c binds a prefill bank to
        const int pg = group_of(n);
        return pg < 0 ? -1 : body0 + c;
    };

    // Decode's descriptor carries every weight bank; the prefill graph's banks map onto them.  A body
    // longer than its decode group (Hy-MT2's 11-layer last group) sizes the bank to the body.
    chunks.clear();
    chunks.resize(groups.size());
    std::vector<std::map<std::string, size_t>> cbytes(groups.size());
    for (const auto & d : dk.desc->in) {
        bank_n[d.name]     = d.bytes;
        bank_dtype[d.name] = d.dtype;
        const int gi = group_of(d.name);
        bank_slot[d.name] = gi >= 0 && !d.shape.empty() && d.shape[0] > 0 ? (uint32_t) (d.bytes / d.shape[0])
                                                                          : d.bytes;
        if (io_name(d.name) || d.name == "route_ids" || pool_kv(d.name) >= 0) {
            continue;  // the KV pools are allocated from -c, not from the compiled extent
        }
        if (gi >= 0) {
            cbytes[(size_t) gi][d.name] = std::max(cbytes[(size_t) gi][d.name], d.bytes);
        } else {
            gbank[d.name] = axera_cmm_alloc_bank(d.bytes);
        }
    }
    std::map<std::string, uint32_t> pre_dtype;  // by decode name
    for (int c = 0; c < pre_calls; ++c) {
        for (const auto & d : pk.desc->in) {
            const int gi = pre_group(c, d.name);
            if (gi < 0 || io_name(d.name) || pool_kv(d.name) >= 0) {
                continue;
            }
            const std::string mapped = groups[(size_t) gi].pre + d.name.substr(3);
            auto it = cbytes[(size_t) gi].find(mapped);
            if (it != cbytes[(size_t) gi].end()) {
                it->second = std::max(it->second, d.bytes);
                pre_dtype[mapped] = d.dtype;
            }
        }
    }
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        for (const auto & [name, bytes] : cbytes[gi]) {
            chunks[gi][name] = axera_cmm_alloc_bank(bytes);
        }
    }
    for (const auto & [name, dt] : pre_dtype) {
        if (dt == 9 && bank_dtype.at(name) == 3) {  // prefill BF16, decode F32
            pre_bf16[name] = axera_cmm_alloc_bank((size_t) bank_slot.at(name) / sizeof(float) *
                                                  sizeof(ggml_bf16_t) * body.layers);
        }
    }
    GGML_ASSERT(gbank.count("head_w") && gbank.count("head_s") && gbank.count("head_norm"));
    GGML_ASSERT(rope_table == (gbank.count("cos") && gbank.count("sin")));

    auto alloc_io = [&](const io_desc & d) {
        auto it = io.find(d.name);
        if (it == io.end() || it->second->size < d.bytes) {
            io[d.name] = axera_cmm_alloc(d.bytes);
        }
    };
    size_t route_bytes = 0, hidden_b = (size_t) WP_MAX * H * sizeof(ggml_bf16_t);
    for (const auto * desc : { pk.desc, dk.desc }) {
        for (const auto & d : desc->in) {
            if (io_name(d.name)) {
                alloc_io(d);
            }
        }
        for (const auto & d : desc->out) {
            if (d.name == "route_ids") {
                route_bytes = std::max(route_bytes, d.bytes);
            } else if (d.name == "output") {
                hidden_b = std::max(hidden_b, d.bytes);
            } else if (io_name(d.name)) {
                alloc_io(d);
            }
        }
    }
    io["output"]  = axera_cmm_alloc(hidden_b);
    io["hidden2"] = axera_cmm_alloc(hidden_b);
    if (route_bytes) {
        route_cmm.push_back(axera_cmm_alloc(route_bytes));
    }

    kv_page_b = (size_t) 2 * PG * KVH * HD * sizeof(ggml_bf16_t);
    kv_layer.clear();
    int NP = 0;
    if (kv_pools) {
        // one BF16 pool per layer: the kernel was compiled for its page extent, -c decides how many pages are
        // allocated.  A 4096-row prefill call writes K/V for every row including the tail padding, so keep
        // KV_HEADROOM pages after the context's last page.
        const io_desc * kv0 = find_io(dk.desc->in, "g0_kv0");
        NP = (int) (kv0->bytes / kv_page_b);
        GGML_ASSERT(NP > KV_HEADROOM && (size_t) NP * kv_page_b == kv0->bytes);
        int64_t cells_want = (int64_t) (NP - KV_HEADROOM) * PG;
        if (const int64_t c = axera_kv_cells(); c > 0) {
            kv_pages   = (int) std::max<int64_t>(1, std::min<int64_t>(NP, (c + PG - 1) / PG + KV_HEADROOM));
            cells_want = c;
        } else {
            kv_pages = NP;
        }
        if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
            kv_pages = std::max(1, std::min(NP, atoi(e)));
        }
        kv_pages = std::max(kv_pages, KV_HEADROOM + 1);
        KVL.clear();
        for (int j = 0; j < LAYERS; ++j) {
            KVL.push_back(axera_cmm_alloc_bank((size_t) kv_pages * kv_page_b));
            KVL.back()->flush(KVL.back()->vir, KVL.back()->size);
        }
        for (const axera_cmm & c : KVL) {
            kv_layer.push_back({ &c, 0 });
        }
        cells = std::max<int64_t>(1, std::min<int64_t>(cells_want, (int64_t) (kv_pages - KV_HEADROOM) * PG));
    } else {
        // the KV is each group's g<i>_b0_kv bank, one compiled page extent per layer slot
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const std::string kn = groups[gi].pre + "b0_kv";
            const size_t layer_b = bank_slot.at(kn);
            GGML_ASSERT(layer_b % kv_page_b == 0 && (NP == 0 || (int) (layer_b / kv_page_b) == NP));
            NP = (int) (layer_b / kv_page_b);
            for (int b = 0; b < groups[gi].layers; ++b) {
                kv_layer.push_back({ &chunks[gi].at(kn), (size_t) b * layer_b });
            }
        }
        kv_pages = NP;
        cells    = (int64_t) NP * PG;
    }
    GGML_ASSERT((int) kv_layer.size() == LAYERS);

    fill_bank(t, rope_node, inp_pos);
    axera_flush_file_pages();

    wp = (int) (io.at("x")->size / (H * sizeof(ggml_bf16_t)));
    GGML_ASSERT(wp == WP_MAX);
    bind_calls(pre, pk, true);
    bind_calls(dec, dk, false);

    seq = std::make_unique<seqstate>(*this, seq_config_env(1));
    size_t bytes = 0;
    for (const auto & [n, b] : gbank) bytes += b->size;
    for (const auto & c : chunks) {
        for (const auto & [n, b] : c) bytes += b->size;
    }
    GGML_LOG_INFO("%s: %d layers, %d groups, %d prefill calls, KV %d of %d pages, banks %.2f GiB, cells %lld\n",
                  tag.c_str(), LAYERS, (int) groups.size(), pre_calls, kv_pages, NP, bytes / 1073741824.0,
                  (long long) cells);
    inited = true;
}

void lightnav::fill_bank(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node,
                        const ggml_tensor * inp_pos) {
    auto put_f32 = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v, float scale = 1.0f) {
        // block_arguments pads a layer's small tensor to a 32-byte row when several layers share one bank;
        // only the values are meaningful, the padding must be zero
        auto sl = bank_slot.find(n);
        GGML_ASSERT(sl != bank_slot.end() && v.size() * sizeof(float) <= sl->second);
        float * d = (float *) slot(bm, n, b);
        memset(d, 0, sl->second);
        for (size_t i = 0; i < v.size(); ++i) d[i] = v[i] * scale;
    };
    auto put_bf16 = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v) {
        auto sl = bank_slot.find(n);
        GGML_ASSERT(sl != bank_slot.end() && v.size() * sizeof(ggml_bf16_t) <= sl->second);
        memset(slot(bm, n, b), 0, sl->second);
        ggml_fp32_to_bf16_row(v.data(), (ggml_bf16_t *) slot(bm, n, b), (int64_t) v.size());
    };
    // a norm weight in the dtype the decode descriptor declares, and its BF16 prefill mirror if any
    auto put_norm = [&](bmap & bm, const std::string & n, int b, const std::vector<float> & v) {
        if (bank_dtype.at(n) == 9) {
            put_bf16(bm, n, b, v);
        } else {
            put_f32(bm, n, b, v);
        }
        auto m = pre_bf16.find(n);
        if (m != pre_bf16.end()) {
            ggml_fp32_to_bf16_row(v.data(), (ggml_bf16_t *) m->second->vir + (size_t) b * v.size(),
                                  (int64_t) v.size());
        }
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
    // a whole projection into one layer slot, in the layout the kernel descriptor declares: MXFP8
    // (E4M3 + per-32 E8M0) or NVFP4 (codewords + per-16 E4M3 scales + one F32 global scale).  Any GGUF
    // source type is converted once at load; NVFP4 sources are copied without re-quantizing.
    auto put_nv_whole = [&](bmap & bm, const std::string & qn, const std::string & sn, const std::string & zn,
                            const ggml_tensor * w, int b, float s2_scale = 1.0f) {
        const int64_t K = w->ne[0], N = ggml_nrows(w);
        if (bank_dtype.at(qn) == 11 && bank_dtype.at(sn) == 1) {  // MXFP8: E4M3 codes + E8M0 / 32 inputs
            GGML_ASSERT(w->ne[2] == 1 && K % AXERA_MX_GROUP == 0);
            uint8_t * q = slot(bm, qn, b);
            uint8_t * s = slot(bm, sn, b);
            GGML_ASSERT((size_t) N * K <= bank_slot.at(qn));
            GGML_ASSERT((size_t) N * (K / AXERA_MX_GROUP) <= bank_slot.at(sn));
            axera_for_rows_f32(w, [&](int64_t r, const float * row) {
                axera_quantize_row_mxfp8_f32(row, q + r * K, s + r * (K / AXERA_MX_GROUP), K);
            });
            fill_f32(bm, zn, b, s2_scale);
            return;
        }
        GGML_ASSERT(w->ne[2] == 1 && K % 64 == 0);
        uint8_t * q = slot(bm, qn, b);
        uint8_t * s = slot(bm, sn, b);
        GGML_ASSERT((size_t) N * (K / 2) <= bank_slot.at(qn));
        GGML_ASSERT((size_t) N * (K / 16) <= bank_slot.at(sn));
        float s2;
        if (w->type == GGML_TYPE_NVFP4) {
            axera_copy_nvfp4_slice(w, 0, N, q, K / 2, s, K / 16);
            s2 = ((const float *) nv_scale(t, w)->data)[0];
        } else {
            s2 = axera_quantize_nvfp4_slice(w, 0, N, q, K / 2, s, K / 16);
        }
        fill_f32(bm, zn, b, s2 * s2_scale);
    };
    // one row of a GGUF NVFP4 expert tensor -> one row of the kernel's 16-value-group layout
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
    // the experts of one projection: rows [e][r] of w at row (e * stride + r0 + r) of the bank
    auto put_experts = [&](bmap & bm, const std::string & qn, const std::string & sn, const ggml_tensor * w, int b,
                           int64_t stride, int64_t r0) {
        GGML_ASSERT(w->type == GGML_TYPE_NVFP4 && w->ne[2] == E);
        const int64_t R = w->ne[1], qrb = w->ne[0] / 2, srb = w->ne[0] / 16;
        uint8_t * q = slot(bm, qn, b), * s = slot(bm, sn, b);
        axera_host_pool().run(E * R, [&](int64_t idx) {
            const int64_t e = idx / R, r = idx % R;
            copy_nv_row(w, e, r, q + (size_t) (e * stride + r0 + r) * qrb, s + (size_t) (e * stride + r0 + r) * srb);
        });
    };
    auto expert_s2 = [&](const ggml_tensor * w, int64_t e) {
        const ggml_tensor * ws = nv_scale(t, w);
        return ((const float *) ws->data)[ggml_nelements(ws) == 1 ? 0 : e];
    };
    auto has = [&](const std::string & n) { return bank_slot.count(n) != 0; };
    auto drop = [&](const std::vector<const ggml_tensor *> & read) {
        if (!axera_keep_pages()) {
            for (const ggml_tensor * w : read) axera_drop_mapped(w);
        }
    };

    auto fill_layer = [&](const group_info & g, bmap & bm, int b, int il) {
        std::vector<const ggml_tensor *> read;
        auto W = [&](const char * s) -> const ggml_tensor * {
            const ggml_tensor * w = need(t, "blk." + std::to_string(il) + "." + s);
            axera_page_in(w);
            read.push_back(w);
            return w;
        };
        // one bank holds the group's layers; the input names are fixed at the b0 label and the slot selects
        // the layer
        const std::string pa = g.pre + "b0_attn_";
        put_norm(bm, pa + "w_norm_attn", b, f32_of(W("attn_norm.weight")));
        put_f32(bm, pa + "w_norm_q", b, f32_of(W("attn_q_norm.weight")), 1.0f / std::sqrt((float) HD));
        put_norm(bm, pa + "w_norm_k", b, f32_of(W("attn_k_norm.weight")));
        put_nv_whole(bm, pa + "w_q", pa + "w_scale_q", pa + "w_scale2_q", W("attn_q.weight"), b);
        put_nv_whole(bm, pa + "w_k", pa + "w_scale_k", pa + "w_scale2_k", W("attn_k.weight"), b);
        put_nv_whole(bm, pa + "w_v", pa + "w_scale_v", pa + "w_scale2_v", W("attn_v.weight"), b);
        put_nv_whole(bm, pa + "w_o", pa + "w_scale_o", pa + "w_scale2_o", W("attn_output.weight"), b);
        for (const char * c : { "q", "k", "v", "o" }) {
            fill_zero(bm, pa + "w_zeropoint_" + c, b);
        }
        if (g.dense) {
            const std::string pm = g.pre + "b0_mlp_";
            put_norm(bm, pm + "w_norm", b, f32_of(W("ffn_norm.weight")));
            put_nv_whole(bm, pm + "w1", pm + "w_scale1", pm + "w_scale2_1", W("ffn_gate.weight"), b);
            put_nv_whole(bm, pm + "w2", pm + "w_scale2", pm + "w_scale2_2", W("ffn_up.weight"), b);
            put_nv_whole(bm, pm + "w3", pm + "w_scale3", pm + "w_scale2_3", W("ffn_down.weight"), b);
            for (const char * z : { "w_zp_scale1", "w_zp_scale2", "w_zp_scale3", "input_permutation" }) {
                fill_zero(bm, pm + z, b);
            }
            drop(read);
            return;
        }
        const std::string pm = g.pre + "b0_moe_";
        put_f32(bm, pm + "input_norm_w", b, f32_of(W("ffn_norm.weight")));
        if (has(pm + "correction_bias")) {
            put_f32(bm, pm + "correction_bias", b, f32_of(W("exp_probs_b")));
        }
        put_bf16(bm, pm + "w_route", b, f32_of(W("ffn_gate_inp.weight")));
        fill_f32(bm, pm + "w_scale_route", b, 1.0f);
        fill_f32(bm, pm + "w_scale2_route", b, 1.0f);
        const ggml_tensor * wg = W("ffn_gate_exps.weight");
        const ggml_tensor * wu = W("ffn_up_exps.weight");
        const ggml_tensor * wd = W("ffn_down_exps.weight");
        const int64_t I = wg->ne[1];
        GGML_ASSERT(wu->ne[1] == I && wd->ne[1] == H);
        // gate [0, I) and up [I, 2I) share one fused [E, 2I, H] bank and one [E, 2] scale2 row per expert
        const std::string qg = has(pm + "w_gate") ? pm + "w_gate" : pm + "w_up";
        const std::string sg = has(pm + "w_scale_gate") ? pm + "w_scale_gate" : pm + "w_scale_up";
        const std::string zg = has(pm + "w_scale2_gate") ? pm + "w_scale2_gate" : pm + "w_scale2_up";
        put_experts(bm, qg, sg, wg, b, 2 * I, 0);
        put_experts(bm, qg, sg, wu, b, 2 * I, I);
        put_experts(bm, pm + "w_down", pm + "w_scale_down", wd, b, H, 0);
        float * s2 = (float *) slot(bm, zg, b);
        float * ds2 = (float *) slot(bm, pm + "w_scale2_down", b);
        for (int64_t e = 0; e < E; ++e) {
            s2[2 * e]     = expert_s2(wg, e);
            s2[2 * e + 1] = expert_s2(wu, e);
            ds2[e]        = expert_s2(wd, e);
        }
        if (has(pm + "shared_expert_gate")) {
            // HYV3 has no shared-expert sigmoid: the kernel's gated branch with a zero gate (sigmoid(0) =
            // 0.5) and a 2x down scale2 is exactly the un-gated shared expert
            fill_zero(bm, pm + "shared_expert_gate", b);
            const char * sn[3] = { "gate", "up", "down" };
            const char * wn[3] = { "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" };
            for (int j = 0; j < 3; ++j) {
                const std::string base = std::string("w_shared_expert_mlp_") + sn[j];
                put_nv_whole(bm, pm + base, pm + "w_scale_shared_expert_mlp_" + sn[j],
                             pm + "w_scale2_shared_expert_mlp_" + sn[j], W(wn[j]), b, j == 2 ? 2.0f : 1.0f);
            }
        }
        drop(read);
    };

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        for (int b = 0; b < groups[gi].layers; ++b) {
            fill_layer(groups[gi], chunks[gi], b, groups[gi].first + b);
        }
    }

    // final norm and NVFP4 lm_head
    {
        std::vector<const ggml_tensor *> read;
        const ggml_tensor * nw = need(t, "output_norm.weight");
        const ggml_tensor * ow = need(t, "output.weight");
        axera_page_in(nw);
        axera_page_in(ow);
        read.push_back(nw);
        read.push_back(ow);
        put_f32(gbank, "head_norm", 0, f32_of(nw));
        put_nv_whole(gbank, "head_w", "head_s", "head_s2", ow, 0);
        drop(read);
    }


    for (auto & [n, c] : gbank) c->flush();
    for (auto & c : chunks) {
        for (auto & [n, b] : c) b->flush();
    }
    for (auto & [n, c] : pre_bf16) c->flush();
    if (std::getenv("GGML_AXERA_BANK_SUM")) {
        axera_bank_sum((std::string(fam->model) + "_head").c_str(), gbank);
        for (size_t gi = 0; gi < chunks.size(); ++gi) {
            axera_bank_sum((std::string(fam->model) + "_group" + std::to_string(gi)).c_str(), chunks[gi]);
        }
        axera_bank_sum((std::string(fam->model) + "_pre_bf16").c_str(), pre_bf16);
    }
}

void lightnav::bind_calls(std::vector<call> & calls, const kernel & k, bool prefill) {
    const char * what = prefill ? "prefill" : "decode";
    const int count = prefill ? pre_calls : 1;
    calls.assign((size_t) count, {});
    for (int c = 0; c < count; ++c) {
        call & cl = calls[c];
        cl.k      = &k;
        // prefill alternates the two hidden buffers across its calls; decode is one call over every group
        axera_cmm & src = !prefill || c == 0 ? io.at("x") : (c % 2 ? io.at("output") : io.at("hidden2"));
        axera_cmm & dst = !prefill || c % 2 == 0 ? io.at("output") : io.at("hidden2");
        cl.in.resize(k.desc->in.size());
        for (size_t i = 0; i < k.desc->in.size(); ++i) {
            const io_desc & d = k.desc->in[i];
            const axera_cmm * b = nullptr;
            const int pg = group_of(d.name);
            if (d.name == "x") {
                b = &src;
            } else if (d.name == "output") {
                b = &dst;  // an in-place graph input
            } else if (d.name == "route_ids" && !route_cmm.empty()) {
                b = &route_cmm[0];
            } else if (io_name(d.name)) {
                auto it = io.find(d.name);
                if (it == io.end()) {
                    GGML_ABORT("%s: %s: no buffer for input %s\n", tag.c_str(), what, d.name.c_str());
                }
                b = &it->second;
            } else if (pg < 0) {
                b = &at(gbank, d.name);
            } else {
                // lightnav's prefill graph is one dense body of six layers, so call c binds that body to
                // decode group c (groups 0..5 are the six layer banks of the 36-layer tower).
                const int gi = !prefill ? pg : body0 + c;
                GGML_ASSERT(gi < (int) groups.size());
                const std::string mapped = groups[(size_t) gi].pre + d.name.substr(3);
                if (const int j = pool_kv(d.name); j >= 0) {
                    const int layer = groups[(size_t) gi].first + j;
                    GGML_ASSERT(layer < LAYERS);
                    cl.in[i] = axera_io(KVL[(size_t) layer].get(), KVL[(size_t) layer]->vir, d.bytes);
                    continue;
                }
                auto m = prefill ? pre_bf16.find(mapped) : pre_bf16.end();
                b = m != pre_bf16.end() ? &m->second : &at(chunks[(size_t) gi], mapped);
            }
            cl.in[i] = axera_io(*b, d.bytes);
        }
        cl.out.resize(k.desc->out.size());
        for (size_t i = 0; i < k.desc->out.size(); ++i) {
            const io_desc & d = k.desc->out[i];
            if (d.name == "output") {
                cl.out[i] = axera_io(dst, d.bytes);
            } else if (d.name == "route_ids" && !route_cmm.empty()) {
                cl.out[i] = axera_io(route_cmm[0], d.bytes);
            } else {
                auto it = io.find(d.name);
                if (it == io.end()) {
                    GGML_ABORT("%s: %s: no buffer for output %s\n", tag.c_str(), what, d.name.c_str());
                }
                cl.out[i] = axera_io(it->second, d.bytes);
            }
        }
    }
}

// llama.cpp's GGML_ROPE_TYPE_IMROPE cache for one token: 64 pairs, sections [24,20,20,0]
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

void lightnav::run_prefill_page(int64_t cell, int rows, const uint16_t * x, const int32_t * p4, int64_t batch_cell0,
                               bool need_head) {
    GGML_ASSERT(cell % PG == 0 && rows > 0 && rows <= wp);
    axera_cmm & xi = io.at("x");
    auto dsi = io.find("deepstack");
    // The padding rows past `rows` are never seen by a valid query (attention is causal) or read by the head.
    // Cycling real rows through them keeps them out of the all-zero router bucket (every zero row would go
    // to the same eight experts).
    uint16_t * xb = (uint16_t *) xi->vir;
    uint16_t * db = dsi != io.end() ? (uint16_t *) dsi->second->vir : nullptr;
    axera_host_pool().run(wp, [&](int64_t r) {
        const int64_t s = r < rows ? r : r % rows;
        memcpy(xb + (size_t) r * H, x + (size_t) s * IN, (size_t) H * sizeof(uint16_t));
        for (int p = 0; db && p < PLANES; ++p) {
            memcpy(db + ((size_t) p * wp + r) * H, x + (size_t) s * IN + (size_t) (p + 1) * H,
                   (size_t) H * sizeof(uint16_t));
        }
    });

    // Padded rows continue the position sequence so their K/V land past the valid tail (masked for valid
    // queries and beyond last_page_len for the later decode), not over slot 0.
    int32_t * pos = (int32_t *) io.at("position_ids")->vir;
    for (int r = 0; r < wp; ++r) pos[r] = r;
    std::vector<int32_t> pp((size_t) wp * 4, 0);
    memcpy(pp.data(), p4, (size_t) rows * 4 * sizeof(int32_t));
    for (int r = rows; r < wp; ++r) pp[(size_t) r * 4] = (int32_t) (cell + r);
    float * cs = (float *) io.at("cos")->vir;
    float * sn = (float *) io.at("sin")->vir;
    axera_host_pool().run(wp, [&](int64_t r) {
        mrope_cache(cs + (size_t) r * (HD / 2), sn + (size_t) r * (HD / 2), pp.data() + (size_t) r * 4);
    });
    io.at("cos")->flush();
    io.at("sin")->flush();
    // the attention takes indptr as the history page range; the page being written is cache_location
    const int page = (int) (cell / PG);
    int32_t * ind = (int32_t *) io.at("indptr")->vir;
    ind[0] = 0;
    ind[1] = page;
    axera_cmm & idx = io.at("indices");
    GGML_ASSERT((size_t) (page + wp / PG) * sizeof(int32_t) <= idx->size);
    int32_t * ix = (int32_t *) idx->vir;
    for (int i = 0; i <= page + wp / PG; ++i) ix[i] = i;
    *(int32_t *) io.at("last_page_len")->vir = 0;  // the prefill graph does not read it
    int32_t * cl = (int32_t *) io.at("cache_location")->vir;
    cl[0] = page;
    cl[1] = 0;
    *(int32_t *) io.at("valid_tokens")->vir = rows;
    if (auto dsr = io.find("ds_run"); dsr != io.end()) {
        *(uint16_t *) dsr->second->vir = 0x3f80;  // bf16 1.0
        dsr->second->flush();
    }
    for (const char * n : { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                            "valid_tokens" }) {
        io.at(n)->flush();
    }
    if (db) {
        dsi->second->flush();
    }

    // batch_cell0 is the cell of the caller's batch row 0.  A replayed page prefix is part of the kernel's
    // rows but its logits are not wanted: the head takes the caller's last rows.
    const int first = std::max(0, std::min(rows, (int) (batch_cell0 - cell)));
    const int want  = rows - first;
    GGML_ASSERT(want > 0 && want <= rows);
    const int take = std::min(head_rows_n, want);
    const int hr0  = rows - take;
    axera_cmm & hx = io.at("head_x");
    memset(hx->vir, 0, hx->size);
    int32_t * hrb = (int32_t *) io.at("head_rows")->vir;
    for (int r = 0; r < head_rows_n; ++r) hrb[r] = r < take ? hr0 + r : hr0;
    io.at("head_rows")->flush();

    auto head_run = io.find("head_run"), layer0_run = io.find("layer0_run"), body_run = io.find("body_run");
    for (int ci = 0; ci < (int) pre.size(); ++ci) {
        // Hy-MT2's graph runs layer 0 on the first call only, and its 12-layer body runs the last group's
        // 11 layers
        if (layer0_run != io.end()) {
            *(int32_t *) layer0_run->second->vir = ci == 0 ? 1 : 0;
            layer0_run->second->flush();
        }
        if (body_run != io.end()) {
            *(int32_t *) body_run->second->vir = groups[(size_t) (body0 + ci)].layers;
            body_run->second->flush();
        }
        if (head_run != io.end()) {
            *(int32_t *) head_run->second->vir = need_head && ci + 1 == (int) pre.size() ? 1 : 0;
            head_run->second->flush();
        }
        pre[ci].k->run(pre[ci].in, pre[ci].out, "prefill");
        // the deepstack planes belong to the first call's layers 1..3 only
        if (ci == 0 && db && pre.size() > 1) {
            memset(dsi->second->vir, 0, dsi->second->size);
            dsi->second->flush();
        }
    }

    head_first = (int) (cell + hr0 - batch_cell0);
    head_n     = take;
    small_head = false;
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, (size_t) head_rows_n * VOCAB * sizeof(float));
}

void lightnav::run_decode_row(int64_t cell, const uint16_t * x, const int32_t * p4) {
    // the IO buffers are sized for a prefill page; a decode row writes a few bytes of each and cleans only those
    auto put = [&](const char * n, const void * v, size_t bytes) {
        axera_cmm & c = io.at(n);
        memcpy(c->vir, v, bytes);
        c->flush(c->vir, bytes);
    };
    put("x", x, H * sizeof(uint16_t));
    const int32_t p = 0;
    put("position_ids", &p, sizeof(p));
    {
        float cs[HD / 2], sn[HD / 2];
        mrope_cache(cs, sn, p4);
        put("cos", cs, sizeof(cs));
        put("sin", sn, sizeof(sn));
    }
    // decode_attention's cached_len = (pages - 1) * PG + last_page_len is the tokens already in KV, the
    // current cell itself; cell 0 has one empty page instead of -PG.  The current K/V go to page/off.
    const int page = (int) (cell / PG), off = (int) (cell % PG);
    const int hist_pages = cell == 0 ? 1 : (int) ((cell + PG - 1) / PG);
    const int32_t ind[2] = { 0, hist_pages };
    const int32_t hist_last = cell == 0 ? 0 : (int32_t) (cell - (int64_t) (hist_pages - 1) * PG);
    const int32_t loc[2] = { page, off };
    axera_cmm & idx = io.at("indices");
    GGML_ASSERT((size_t) (page + 1) * sizeof(int32_t) <= idx->size);
    int32_t * ix = (int32_t *) idx->vir;
    for (int i = 0; i <= page; ++i) ix[i] = i;
    idx->flush(idx->vir, (size_t) (page + 1) * sizeof(int32_t));
    put("indptr", ind, sizeof(ind));
    put("last_page_len", &hist_last, sizeof(hist_last));
    put("cache_location", loc, sizeof(loc));
    axera_cmm & hx = io.at("head_x");
    memset(hx->vir, 0, hx->size);
    const int32_t zero = 0, one = 1;
    put("head_rows", &zero, sizeof(zero));
    put("head_run", &one, sizeof(one));
    for (const char * n : { "layer0_run", "body_run" }) {
        if (io.count(n)) {
            put(n, &one, sizeof(one));
        }
    }
    GGML_ASSERT(dec.size() == 1);
    dec[0].k->run(dec[0].in, dec[0].out, "decode");
    axera_cmm & hl = io.at("head_logits");
    hl->invalidate(hl->vir, VOCAB * sizeof(float));
    head_first = 0;  // the decode batch is one row: its logits row is batch row 0
    head_n     = 1;
    small_head = false;
}

bool lightnav::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    const uint16_t * xr = x;
    if (!xr) {
        GGML_ASSERT(batch0 == 0);
        xbuf.resize((size_t) n * IN);
        seq_rows(0, n, xbuf.data());
        xr = xbuf.data();
    }
    if (snap) snap->deferred = false;
    if (n < fam->small_batch) {
        // A small batch (PPL -ub 4, a short chat turn) would pad to 4096 rows; replay it row by row through
        // decode.  The batch0 rows are the replay prefix seqstate already recorded: they rebuild the state,
        // the logits are the rows after them, renumbered from zero for publish_logits().
        const int b0 = std::max(0, std::min(n, batch0));
        const int T  = n - b0;
        GGML_ASSERT(T > 0);
        small_logits.resize((size_t) T * VOCAB);
        for (int i = 0; i < n; ++i) {
            run_decode_row(c0 + i, xr + (size_t) i * IN, p4 + (size_t) i * 4);
            if (i >= b0) {
                memcpy(small_logits.data() + (size_t) (i - b0) * VOCAB, io.at("head_logits")->vir,
                       (size_t) VOCAB * sizeof(float));
            }
        }
        head_first = 0;
        head_n     = T;
        small_head = true;
        return true;
    }
    for (int off = 0; off < n; off += wp) {
        const int rows = std::min(wp, n - off);
        run_prefill_page(c0 + off, rows, xr + (size_t) off * IN, p4 + (size_t) off * 4, c0 + batch0, off + wp >= n);
    }
    return true;
}

bool lightnav::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
    run_decode_row(c, x, p4);
    return true;
}

void lightnav::seq_rows(int batch_row0, int n, uint16_t * dst) {
    if (n <= 0) return;
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * IN, (size_t) n * IN * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * IN * sizeof(uint16_t));
    }
}

void lightnav::publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T) {
    GGML_ASSERT(logits && logits->type == GGML_TYPE_F32 && logits->ne[0] == VOCAB);
    const int n_out = (int) logits->ne[1];
    if (n_out == 0) {
        return;  // a warm-up or an image-only batch asks for no logits
    }
    if (n_out > head_n) {
        GGML_ABORT("%s: %d requested logits rows but the page head computed %d (keep -ub <= %d)\n", tag.c_str(),
                   n_out, head_n, head_rows_n);
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
    if (n_out < T && !out_ids) {
        GGML_ABORT("%s: %d of %d output rows without row indices\n", tag.c_str(), n_out, T);
    }
    const float * head_src = small_head ? small_logits.data() : (const float *) io.at("head_logits")->vir;
    // The scheduler may place result_output in the lazy host buffer instead of CMM; host memory needs no
    // cache maintenance, only a CMM output does.
    auto * dctx = ggml_backend_buffer_is_axera(logits->buffer)
                      ? (ggml_backend_axera_buffer_context *) logits->buffer->context : nullptr;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        if (row < head_first || row >= head_first + head_n) {
            GGML_ABORT("%s: output row %d outside the head's %d..%d\n", tag.c_str(), row, head_first,
                       head_first + head_n - 1);
        }
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        memcpy(o, head_src + (size_t) (row - head_first) * VOCAB, VOCAB * sizeof(float));
        if (dctx) {
            dctx->flush(o, VOCAB * sizeof(float));
            dctx->clean_p     = o;
            dctx->clean_n     = VOCAB * sizeof(float);
            dctx->clean_epoch = axera_graph_epoch;
        }
    }
    output_head::after(g, names);
}

ggml_status lightnav::compute(ggml_cgraph * g) {
    const auto & t = names.of(g);
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("%s: a graph of %d nodes that is not the whole model (-ngl 99)\n", tag.c_str(), g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    // Qwen3-VL's token input is a [H, T] view of the [H + 3 * H, T] embedding: the deepstack planes follow
    // the token row (zero for text tokens).  The row is read IN floats wide.
    ggml_tensor * x0 = llm_input(g, norm0, false);
    const int T = (int) x0->ne[1];
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data) {
        L = llm_layers(t);
        GGML_ASSERT(L == LAYERS && x0->ne[0] == H);
        init(t, llm_rope_node(g), inp_pos);
        bound = w0->data;
    }
    GGML_ASSERT(x0->ne[0] == H && x0->nb[1] >= (size_t) IN * sizeof(float));
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, cells, PG);
    if (!llm_one_sequence(kv, T, fam->model)) {
        return GGML_STATUS_FAILED;
    }
    llm_positions(inp_pos, T, pbuf);
    std::vector<ggml_bf16_t> xb((size_t) T * IN);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                              xb.data() + (size_t) r * IN, IN);
    });
    batch_rows      = xb.data();
    last_seq_status = 0;
    const int64_t    c0  = kv.cell_of(0);
    const seq_status sst = seq->feed(c0, T, (const uint16_t *) xb.data(), pbuf.data());
    if (sst != SEQ_OK) {
        last_seq_status = (int) llm_seq_log(sst, c0, *seq, fam->model);
        return GGML_STATUS_FAILED;
    }
    publish_logits(g, logits, T);
    return GGML_STATUS_SUCCESS;
}

// the sequence state: every layer's KV pages before the current cell and core/seqstate's blob (core/kv_state)
kv_state_format lightnav::state_format() const {
    kv_state_format f;
    f.magic   = fam->magic;
    f.version = fam->version;
    f.page_b  = kv_page_b;
    f.pages   = kv_pages;
    f.pg      = PG;
    f.cells   = cells;
    return f;
}

cmm_kv_pools lightnav::state_pools() const {
    cmm_kv_pools p;
    for (const auto & [c, off] : kv_layer) {
        p.p.push_back({ c, off });
    }
    return p;
}

size_t lightnav::state_size(uint32_t flags) const {
    return !inited ? 0 : kv_state_size(state_format(), state_pools(), *seq, flags & state_flags_partial_only);
}

size_t lightnav::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer  tw("state write");
    cmm_kv_pools p = state_pools();
    return !inited ? 0 : kv_state_write(state_format(), p, *seq, dst, size, flags & state_flags_partial_only);
}

size_t lightnav::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer  tr("state read");
    cmm_kv_pools p = state_pools();
    return !inited ? 0 : kv_state_read(state_format(), p, *seq, src, size, flags & state_flags_partial_only);
}

}  // namespace

std::shared_ptr<model> make_lightnav(const manifest & m) { return std::make_shared<lightnav>(m); }

}  // namespace axera
