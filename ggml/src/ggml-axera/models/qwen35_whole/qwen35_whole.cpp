// Axera NPU backend: the Qwen3.5 dense whole-model design (9B, 4B, 2B; 0.8B with MXFP8 weights from a Q8_0 GGUF). Every model of a manifest gets this
// adapter; the manifest names its kernels, and the kernel IO gives the model's layer counts and head sizes.
// MiniCPM-OCR-1.0's text decoder is the 0.8B design whose kernels also carry the output head (the last prefill call and
// the last decode group write the logits): the same adapter, dec_head / pre_head.
//
// One path, the manifest's kernels (ax-dsl-kernels models/qwen35_{2b,4b,9b}/whole):
//   prefill  role prefill: the whole model in one call per page of 256 cells (a runtime loop over the manifest's
//            blocks, the layers' weights in banks [blocks, ...] indexed by the block), the last page of a batch padded
//            (valid_tokens).
//            Its history is whole pages: a batch that starts inside a page restarts at the page's first cell from the
//            recurrent state saved there, with the page's earlier inputs (kept) before its own
//   decode   role decode.group: G consecutive layers per call (weights in packs), L / G calls per token, then lm_head
//   verify   role decode.verify, optional (speculative decoding): a batch of 2 - 4 tokens in one call
//   mtp      roles mtp.*, optional: a draft context's MTP block
// The state is this adapter's (CMM pools in the prefill kernel's layouts, the decode group bound to each layer's part):
// the KV cache paged [blocks][pages][K, V][256][kv heads x head dim] bf16, the delta-net states
// [layers][value heads][key dim][value dim] f32 and conv states q / k [layers][3][key heads x key dim], v
// [layers][3][value heads x value dim] bf16 (value heads in HF order); after a verification the current state of a
// delta-net layer is a snapshot's row (where[]). One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/params.h"
#include "../../core/seqstate.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"
#include "../../ops/gdn_bank.h"
#include "../../ops/gdn_mtp.h"
#include "../../ops/gdn_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <regex>
#include <set>

namespace axera {
namespace {

constexpr int PG = 256;  // cells per page = prefill rows per call = the decode kernels' KV block

using call = npu_call;

// the adapter blob in llama.cpp's sequence state (context checkpoints / prompt cache)
class qwen35_whole final : public model, public seq_model, gdn_state {
  public:
    explicit qwen35_whole(const manifest & m) : man(m), prm(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size(uint32_t flags) const override;
    size_t       state_write(void * dst, size_t size, uint32_t flags) override;
    size_t       state_read(const void * src, size_t size, uint32_t flags) override;
    kv_state_format state_format() const;
    cmm_kv_pools    state_pools() const;
    int          last_state_status() const override { return last_seq_status; }

    seq_caps seq_get_caps() const override;
    void     seq_state_save(uint8_t * dst) override { state_save(dst); }
    void     seq_state_load(const uint8_t * src) override {
        if (state_load(src)) {
            mtp.clear();
        }
    }
    bool     seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override;
    bool     seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override;
    void     seq_rows(int batch_row0, int n, uint16_t * dst) override;
    bool     seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) override;
    void     seq_verify_keep(int k) override { verify_keep(k); }
    void     seq_verify_peek(int k, uint8_t * dst) override { verify_peek(k, dst); }

  private:
    const manifest & man;
    params           prm;
    graph_names      names;
    const void *     bound = nullptr;
    bool             fp8 = false;  // the prefill banks' projections: MXFP8 (else MXFP4), from the kernel IO shapes
    int              KH = 16, VH = 0, REP = 1;  // delta-net key / value heads (from the prefill kernels' IO)
    std::vector<int> ord;  // layer -> its index among the delta-net layers / the attention layers

    // the state pools (the base's delta-net pools): the KV cache
    std::vector<axera_cmm> KVC;  // the KV cache: one buffer per prefill call's attention layers (a CMM allocation is < 4 GB)
    size_t    kv_b = 0;
    // the pages of the cache the pools hold (llama.cpp's -c through the buffer type, GGML_AXERA_KV_PAGES overrides;
    // default: the kernels' context): the kernels keep their declared IO size but touch only the pages up to the runtime
    // page count, so a smaller context needs only its pages (a single-block prefill kernel only: a call of several
    // layers has fixed layer strides in its KV IO)
    size_t    kv_alloc = 0;  // the bytes of one layer's pool

    gdn_mtp mtp;  // the MTP block (draft context graphs)
    std::vector<call>              pres;  // the prefill calls of a page (the kernel's blocks each)
    int                            ncall = 1;
    axera_cmm                      PX, POS, INDPTR, INDICES, LASTLEN, CLOC, VALID, POUT[2];
    output_head                    head;
    // A merged output head (the kernels have head_on / head inputs: MiniCPM-OCR's): the last prefill call and the
    // last decode group write the logits themselves; the calls then run one by one, their head flags set in between.
    bool                           dec_head = false;
    bool                           pre_head = false;
    ggml_tensor *                  pre_logits = nullptr;
    std::vector<int32_t>           pre_rows;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    void bind_head_calls(ggml_tensor * logits);
    const char * state_magic() const { return man.model == "minicpm_ocr" ? "AXMOCRS2" : "AXW35S2"; }
    gdn_mtp_env mtp_env();
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                      seq_snap * snap, ggml_tensor * logits, const std::vector<int32_t> & rows);
};

void qwen35_whole::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen35_whole: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        ord[il] = g ? 0 : na++;  // the delta-net rows are laid out below, once the prefill call size is known
        if (g) {
            ++ng;
        }
    }
    const kernel & dk = man.get("decode.group");
    const kernel & pk = man.get("prefill");
    G                 = 0;
    for (const auto & d : dk.desc->out) {
        if (d.name.rfind("st", 0) == 0 || d.name.rfind("kv", 0) == 0) {
            G = std::max(G, atoi(d.name.c_str() + 2) + 1);
        }
    }
    GGML_ASSERT(G % 4 == 0 && L % G == 0);
    // the pools: sizes from the prefill kernel's (all layers), checked against the decode group's (one layer)
    auto in_of = [&](const kernel & k, const std::string & n) -> const io_desc & {
        for (const auto & d : k.desc->in) {
            if (d.name == n) {
                return d;
            }
        }
        for (const auto & d : k.desc->out) {
            if (d.name == n) {
                return d;
            }
        }
        GGML_ABORT("axera qwen35_whole: %s has no IO %s\n", k.desc->file.c_str(), n.c_str());
    };
    const io_desc & pkv = in_of(pk, "b3_kv"), &pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in"), &pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const int pna = (int) pkv.shape[0], png = 3 * pna;  // the prefill kernel's layers per call
    GGML_ASSERT(pna > 0 && na % pna == 0);
    ncall = na / pna;
    GGML_ASSERT((pst.shape[0] == png || pst.shape[0] == ng || pst.shape[0] == L / 4 || pst.shape[0] == pna) &&
                pcq.shape[0] == pst.shape[0] && pcv.shape[0] == pst.shape[0]);
    // The common state bank (b0 [3 * pna]) is phase-major inside one call; calls follow one another. A call's
    // [3 * pna] rows are then one contiguous slice of the pool, which is what the prefill calls bind below. A build
    // with one bank per phase (b0 / b1 / b2, each [blocks]) keeps the phase-major pool instead.
    const bool common_state = pst.shape[0] == png || pst.shape[0] == ng;
    for (int il = 0; il < L; ++il) {
        if (t.count("blk." + std::to_string(il) + ".ssm_a") == 0) {
            continue;
        }
        const int b = il / 4, p = il % 4;
        ord[il] = common_state ? (b / pna) * png + p * pna + b % pna : p * (L / 4) + b;
    }
    state_group.assign(ng, -1);
    state_local.assign(ng, -1);
    for (int il = 0; il < L; ++il) {
        if (t.count("blk." + std::to_string(il) + ".ssm_a") == 0) {
            continue;
        }
        const int gi = (il / 4) / (G / 4), j = il - gi * G;
        state_group[ord[il]] = gi;
        state_local[ord[il]] = (j / 4) * 3 + j % 4;
    }
    // the delta-net head geometry: the state [layers, 1, VH, KD, VD], and conv1d's channels 2 * KH * KD + VH * VD
    GGML_ASSERT(pst.shape.size() == 5 && pcv.shape.size() >= 2);
    VH = (int) pst.shape[2];
    const int64_t KD = pst.shape[3], VD = pst.shape[4];
    const ggml_tensor * conv_w = t.at("blk.0.ssm_conv1d.weight");
    GGML_ASSERT(conv_w && conv_w->ne[1] == 2 * (int64_t) KH * KD + (int64_t) VH * VD);
    KH  = (int) ((conv_w->ne[1] - VH * VD) / (2 * KD));
    REP = VH / KH;
    GGML_ASSERT(KH > 0 && REP * KH == VH && conv_w->ne[1] == 2 * KH * KD + VH * VD);
    kv_b = pkv.bytes / pna;
    st_b = pst.bytes / pst.shape[0];
    cq_b = pcq.bytes / pcq.shape[0];
    cv_b = pcv.bytes / pcv.shape[0];
    GGML_ASSERT(in_of(dk, "kv3").bytes == kv_b && in_of(dk, "st0").bytes == st_b && in_of(dk, "cq0").bytes == cq_b &&
                in_of(dk, "ck0").bytes == cq_b && in_of(dk, "cv0").bytes == cv_b);
    NBLK = (int) pkv.shape[1];
    GGML_ASSERT(pkv.shape[3] == PG && (int64_t) NBLK * PG == man.context);
    kv_pages = llm_kv_pages(NBLK, PG, pna == 1);
    GGML_ASSERT(kv_pages == NBLK || pna == 1);
    kv_cells = (int64_t) kv_pages * PG;
    kv_alloc = kv_b / NBLK * kv_pages;
    if (kv_pages < NBLK) {
        GGML_LOG_INFO("axera qwen35_whole: the KV pools hold %d of the kernels' %d pages (%lld cells)\n", kv_pages, NBLK, (long long) kv_cells);
    }
    KVC.resize(ncall);
    for (int k = 0; k < ncall; ++k) {
        KVC[k] = axera_cmm_alloc(kv_alloc * pna);
    }
    ST = axera_cmm_alloc((size_t) ng * st_b);
    CQ = axera_cmm_alloc((size_t) ng * cq_b);
    CK = axera_cmm_alloc((size_t) ng * cq_b);
    CV = axera_cmm_alloc((size_t) ng * cv_b);
    state_image_b = ST->size + CQ->size + CK->size + CV->size;

    // decode: x / out A, cs = cos | sin, pn = cell | blocks | the K row of the cell (page * 2 * 256 + offset)
    const uint32_t csb = dk.in_bytes("cs");
    ROT  = (int) (csb / sizeof(ggml_bf16_t));
    A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    COS  = axera_cmm_alloc(csb / 2);
    SIN  = axera_cmm_alloc(csb / 2);
    CS   = axera_cmm_alloc(csb);
    PN   = axera_cmm_alloc(3 * sizeof(int32_t));
    MASK = axera_cmm_alloc(dk.in_bytes("mask"));
    GGML_ASSERT(dk.in_bytes("pn") == 3 * sizeof(int32_t) && dk.in_bytes("mask") == (size_t) man.context * sizeof(ggml_bf16_t));
    auto pool_io = [&](const std::string & n, int il) {
        if (n.rfind("kv", 0) == 0) {
            const int a = ord[il];
            return axera_io(KVC[a / pna].get(), (char *) KVC[a / pna]->vir + (size_t) (a % pna) * kv_b, (uint32_t) kv_b);
        }
        const axera_cmm & p = n[0] == 's' ? ST : n[1] == 'q' ? CQ : n[1] == 'k' ? CK : CV;
        const size_t      b = n[0] == 's' ? st_b : n[1] == 'v' ? cv_b : cq_b;
        return axera_io(p.get(), (char *) p->vir + ord[il] * b, (uint32_t) b);
    };
    // the prefill kernel's weight banks [blocks, ..] from the GGUF (first: decode and verification calls may read
    // their row-split weights from them, inputs k_<bank>: the blocks of the call's G layers)
    const rope rp(rope_node, names.find("inp_pos"), 1);
    banks.clear();
    {
        axera_timer tb("init: prefill banks");
        // The prefill bank IO distinguishes the two MX layouts without relying on the GGUF type (Q8_0 is converted at
        // load): an MXFP8 code row is K bytes, an MXFP4 code row K / 2, while both scale rows are K / 32. The first
        // MLP gate bank is present in every whole-model prefill, in either format.
        const io_desc * code = nullptr, * scale = nullptr;
        for (const auto & d : pk.desc->in) {
            if (d.name == "b0_mlp_w1") {
                code = &d;
            } else if (d.name == "b0_mlp_w_scale1") {
                scale = &d;
            }
        }
        GGML_ASSERT(code && scale && code->shape.size() >= 2 && scale->shape.size() >= 2);
        const int64_t code_k = code->shape.back(), scale_k = scale->shape.back() * 32;
        GGML_ASSERT(code_k == scale_k || code_k * 2 == scale_k);
        fp8 = code_k == scale_k;
        GGML_LOG_INFO("axera qwen35_whole: prefill banks are MXFP%d (%lld codes, %lld scales per row)\n",
                      fp8 ? 8 : 4, (long long) code_k, (long long) scale->shape.back());
        static const std::set<std::string> pools = { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                                                     "valid_tokens", "b3_kv", "head_on", "out_row", "nout", "norm_w",
                                                     "w_lm", "s_lm" };
        // b0 / b1 / b2: the recurrent state banks of the three delta-net phases (a build may use one common b0 for all)
        auto ends = [](const std::string & s, const char * suffix) {
            const size_t n = strlen(suffix);
            return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
        };
        auto state_pool = [&](const std::string & n) {
            return n.size() > 5 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                   (ends(n, "_attn_state_in") || ends(n, "_attn_q_conv_state_in") || ends(n, "_attn_k_conv_state_in") ||
                    ends(n, "_attn_v_conv_state_in"));
        };
        for (const auto & d : pk.desc->in) {
            if (!pools.count(d.name) && !state_pool(d.name)) {
                // the weights of all L / 4 blocks (a call binds its slice below); the cos / sin tables are the same
                // for every block, so the kernel's blocks of them are bound whole to every call
                const bool cs = ends(d.name, "_attn_cos") || ends(d.name, "_attn_sin");
                io_desc    dd = d;
                if (!cs && ncall > 1) {
                    dd.shape[0] *= ncall;
                    dd.bytes *= ncall;
                }
                banks[d.name] = gdn_prefill_bank({ "qwen35_whole", L, ROT, KH, VH, fp8 }, t, dd, rp);
            }
        }
        axera_bank_sum("qwen35_whole", banks);
    }
    bind_decode("qwen35_whole", man, prm, t, L, H, ng, ord, pool_io);

    // prefill: the weight banks (above), the pools, one page's inputs
    PX      = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POUT[0] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POUT[1] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    POS     = axera_cmm_alloc(pk.in_bytes("position_ids"));
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    GGML_ASSERT(pk.in_bytes("x") == PX->size && pk.in_bytes("position_ids") == 3 * PG * sizeof(int32_t) &&
                pk.in_bytes("indices") == (size_t) NBLK * sizeof(int32_t));
    for (int i = 0; i < NBLK; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();
    GGML_ASSERT(pk.out_bytes("output") == POUT[0]->size);
    pres.assign(ncall, {});
    auto ends = [](const std::string & s, const char * suffix) {
        const size_t z = strlen(suffix);
        return s.size() >= z && s.compare(s.size() - z, z, suffix) == 0;
    };
    for (int k = 0; k < ncall; ++k) {  // call k: blocks k * pna .. of the banks, its layers of the pools
        call & c = pres[k];
        c.k      = &pk;
        for (const auto & d : pk.desc->in) {
            const std::string & n = d.name;
            if (n == "head_on" || n == "out_row" || n == "nout" || n == "norm_w" || n == "w_lm" || n == "s_lm") {
                c.in.push_back(AX_ENGINE_IO_BUFFER_T{});  // a merged head's: bind_head_calls
                continue;
            }
            const axera_cmm *   b = n == "x" ? (k == 0 ? &PX : &POUT[(k - 1) & 1]) : n == "position_ids" ? &POS :
                                    n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                    n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID : nullptr;
            size_t off = 0;
            if (!b && n == "b3_kv") {
                b = &KVC[k];
            } else if (!b && n.size() > 5 && n[0] == 'b' && n[1] >= '0' && n[1] <= '2' &&
                       (ends(n, "_attn_state_in") || ends(n, "_attn_q_conv_state_in") || ends(n, "_attn_k_conv_state_in") ||
                        ends(n, "_attn_v_conv_state_in"))) {
                const int phase = n[1] - '0';
                const int kind  = n.rfind("_attn_state_in") != std::string::npos ? 0 :
                                  n.rfind("_attn_q_conv_state_in") != std::string::npos ? 1 :
                                  n.rfind("_attn_k_conv_state_in") != std::string::npos ? 2 : 3;
                b = &pool(kind);
                // A common bank (b0 [layers]) has the phase-major pool's own order: a call's slice follows its
                // predecessor's. A phase bank (bP [blocks]) is one contiguous phase of the pool, and the call's
                // blocks follow in that phase.
                if (d.shape[0] == png && ncall > 1) {
                    off = (size_t) k * d.bytes;
                } else if (d.shape[0] == ng) {
                    off = 0;
                } else if (d.shape[0] == L / 4) {
                    off = (size_t) phase * (L / 4) * part(kind);
                } else if (d.shape[0] == pna) {
                    off = (size_t) phase * (L / 4) * part(kind) + (size_t) k * d.bytes;
                } else {
                    GGML_ABORT("axera qwen35_whole: prefill state %s: shape0 %lld\n", n.c_str(), (long long) d.shape[0]);
                }
                if (d.bytes != (size_t) d.shape[0] * part(kind) || off + d.bytes > (*b)->size) {
                    GGML_ABORT("axera qwen35_whole: prefill state %s: shape0 %lld bytes %zu part %zu bank %zu off %zu\n", n.c_str(),
                               (long long) d.shape[0], (size_t) d.bytes, (size_t) part(kind), (*b)->size, off);
                }
            } else if (!b) {
                b   = &banks.at(n);
                // (cos / sin are bound whole; the weight banks hold every call's blocks)
                off = (*b)->size >= (size_t) ncall * d.bytes ? (size_t) k * d.bytes : 0;
            }
            GGML_ASSERT((*b)->size >= off + (n == "b3_kv" ? kv_alloc : d.bytes));
            c.in.push_back(axera_io(b->get(), (char *) (*b)->vir + off, (uint32_t) d.bytes));
        }
        for (const auto & d : pk.desc->out) {
            if (d.name == "output") {
                c.out.push_back(axera_io(POUT[k & 1], d.bytes));
            } else if (d.name == "logits" || d.name == "ids" || d.name == "top_logits") {
                c.out.push_back(AX_ENGINE_IO_BUFFER_T{});  // a merged head's: bind_head_calls
            } else {
                GGML_ABORT("axera qwen35_whole: prefill: unexpected output %s\n", d.name.c_str());
            }
        }
    }

    ver_rows = ver.empty() ? 0 : (int) (VX->size / (H * sizeof(ggml_bf16_t)));
    seq_config cfg = seq_config_env(1, env_int("GGML_AXERA_VERIFY_CHAIN", ver_rows));  // the pre-migration path only verified batches up to the kernel's rows
    seq = std::make_unique<seqstate>(*this, cfg);

    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen35_whole: %d layers (%d per decode call), prefill banks %.1f GiB, state %.2f GiB, %lld cells\n",
                  L, G, bb / 1073741824.0, (kv_b * na + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}

// the calls' state outputs to where the state is
// the current state into the pools (the prefill kernel's, the decode calls' default)
// the current state: the snapshot after row `row` of the last verification (set `set`)
seq_caps qwen35_whole::seq_get_caps() const {
    seq_caps c;
    c.page            = PG;
    c.width           = H;
    c.image_bytes     = state_image_b;
    c.cells           = kv_cells;
    c.verify_rows     = ver_rows;
    // The verify kernel writes its entry state in place, so a fully rejected batch has no zero-row image to keep.
    c.verify_min_keep = ver_rows > 0 ? 1 : 0;
    c.decode_at_zero  = true;  // the old path ran the decode group at cell 0 after zeroing the pools
    return c;
}

// src == nullptr: the zero state of a new sequence (and of the dummy warm-up batch's cleanup).
bool qwen35_whole::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    state_home();  // the prefill kernels read the pools
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    prefill(c0, n, xb, p4, h_buf, batch0, snap, pre_logits, pre_rows);
    return true;
}

bool qwen35_whole::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    (void) p4;  // the row's M-RoPE positions come from the graph's rope view (batch_rope)
    GGML_ASSERT(batch_row == 0);
    gdn_each each;
    if (dec_head) {  // the last group writes the logits: its head flags between the calls
        each = [&](size_t, bool last) {
            *(int32_t *) head.mgh->vir = last ? 1 : 0;
            *(int32_t *) head.mgt->vir = last && head.merged_topk() ? 1 : 0;
            head.mgh->flush(head.mgh->vir, sizeof(int32_t));
            head.mgt->flush(head.mgt->vir, sizeof(int32_t));
        };
    }
    decode(c, x, out_h, each);
    return true;
}

void qwen35_whole::seq_rows(int batch_row0, int n, uint16_t * dst) {
    rows(batch_row0, n, dst);
}

bool qwen35_whole::seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) {
    (void) p4;  // the rows' M-RoPE positions come from the graph's rope view (batch_rope)
    return verify_rows(c, n, x, batch_row0, man.context);
}

// llama.cpp's sequence state (prompt cache / context checkpoint): core/kv_state's blob of the KV pages before the
// current cell, core/seqstate's restart points (the recurrent state) and, its tail, the MTP cache (gdn_mtp)
kv_state_format qwen35_whole::state_format() const {
    kv_state_format f;
    f.magic   = state_magic();
    f.version = 4;
    f.pages   = kv_pages;
    f.pg      = PG;
    f.cells   = kv_cells;
    return f;
}

cmm_kv_pools qwen35_whole::state_pools() const {
    cmm_kv_pools p;
    p.add(KVC, kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0, kv_alloc, kv_b / NBLK);
    return p;
}

size_t qwen35_whole::state_size(uint32_t flags) const {
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache: size and bytes must agree
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    return kv_state_size(state_format(), state_pools(), *seq, flags & state_flags_partial_only, &mtp);
}

size_t qwen35_whole::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    return kv_state_write(state_format(), p, *seq, dst, size, flags & state_flags_partial_only, &mtp);
}

size_t qwen35_whole::state_read(const void * src, size_t size, uint32_t flags) {
    axera_timer tr("state read");
    mtp.wait();
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    const size_t n = kv_state_read(state_format(), p, *seq, src, size, flags & state_flags_partial_only, &mtp);
    if (n) {
        state_zero = false;
    }
    return n;
}

// prefill pages from cell c0 (a page's first): n rows x [n][H] bf16 and their M-RoPE positions p4 [n][4]; the hidden
// states of the rows from out0 on into h
void qwen35_whole::bind_head_calls(ggml_tensor * logits) {
    const kernel & dk = man.get("decode.group");
    for (auto & c : dec) {
        for (size_t i = 0; i < dk.desc->in.size(); ++i) {
            const std::string & n = dk.desc->in[i].name;
            const axera_cmm * b = n == "head" ? &head.mgh : n == "topk_on" ? &head.mgt : n == "norm_w" ? &head.cnorm :
                                  n == "w_lm" ? &head.q : n == "s_lm" ? &head.s : n == "bias" ? &head.tbias :
                                  n == "mul_pos" ? &head.tmulp : n == "mul_neg" ? &head.tmuln : n == "temp" ? &head.ttemp : nullptr;
            if (b) {
                c.in[i] = axera_io(*b, dk.desc->in[i].bytes);
            }
        }
        for (size_t i = 0; i < dk.desc->out.size(); ++i) {
            const std::string & n = dk.desc->out[i].name;
            if (n == "logits" && head.mdirect) {
                auto * dctx = (ggml_backend_axera_buffer_context *) logits->buffer->context;
                c.out[i] = axera_io(dctx, logits->data, dk.desc->out[i].bytes);
                continue;
            }
            const axera_cmm * b = n == "logits" ? &head.mgl : n == "ids" ? &head.cids : n == "top_logits" ? &head.ctop :
                                  n == "values" ? &head.tval : n == "kids" ? &head.tids : n == "vmax" ? &head.tmax :
                                  n == "sumexp" ? &head.tsum : nullptr;
            if (b) {
                c.out[i] = axera_io(*b, dk.desc->out[i].bytes);
            }
        }
    }
    const kernel & pk = man.get("prefill");
    for (auto & c : pres) {
        for (size_t i = 0; i < pk.desc->in.size(); ++i) {
            const std::string & n = pk.desc->in[i].name;
            const axera_cmm * b = n == "head_on" ? &head.mgh : n == "nout" ? &head.mgt : n == "out_row" ? &head.mpr :
                                  n == "norm_w" ? &head.cnorm : n == "w_lm" ? &head.q : n == "s_lm" ? &head.s : nullptr;
            if (b) {
                c.in[i] = axera_io(*b, pk.desc->in[i].bytes);
            }
        }
        for (size_t i = 0; i < pk.desc->out.size(); ++i) {
            const std::string & n = pk.desc->out[i].name;
            const axera_cmm * b = n == "logits" ? &head.mvl : n == "ids" ? &head.c4ids :
                                  n == "top_logits" ? &head.c4top : nullptr;
            if (b) {
                c.out[i] = axera_io(*b, pk.desc->out[i].bytes);
            }
        }
    }
}

void qwen35_whole::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                           seq_snap * snap, ggml_tensor * logits, const std::vector<int32_t> & rows) {
    GGML_ASSERT(c0 % PG == 0);
    GGML_ASSERT(c0 >= 0 && n >= 0 && c0 + n <= man.context);
    GGML_ASSERT(c0 + n <= kv_cells);  // (the pools' pages)
    for (int64_t r = 0; r < n; ++r) {
        for (int s = 0; s < 4; ++s) {
            GGML_ASSERT(p4[r * 4 + s] >= 0 && p4[r * 4 + s] < man.context);
        }
    }
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (v < PG && snap && snap->cell == cell) {  // the state before the run's last (partial) page
            seq_state_save(snap->dst);
        }
        memcpy(PX->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) PX->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        PX->flush();
        auto * pp = (int32_t *) POS->vir;
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < PG; ++r) {
                pp[s * PG + r] = r < v ? p4[(size_t) (r0 + r) * 4 + s] : 0;
            }
        }
        POS->flush();
        ((int32_t *) INDPTR->vir)[0]  = 0;
        ((int32_t *) INDPTR->vir)[1]  = (int32_t) (cell / PG);
        *(int32_t *) LASTLEN->vir     = PG;
        ((int32_t *) CLOC->vir)[0]    = (int32_t) (cell / PG);
        ((int32_t *) CLOC->vir)[1]    = 0;
        *(int32_t *) VALID->vir       = v;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        std::vector<std::pair<int, int32_t>> page_rows;  // a merged head: (logits row, page row) of this page
        if (pre_head && logits) {
            for (size_t j = 0; j < rows.size(); ++j) {
                const int64_t a = (int64_t) out0 + rows[j];
                if (a >= r0 && a < r0 + v) {
                    page_rows.push_back({ (int) j, (int32_t) (a - r0) });
                }
            }
            for (size_t j = 0; j < page_rows.size(); ++j) {
                ((int32_t *) head.mpr->vir)[j] = page_rows[j].second;
            }
            head.mpr->flush(head.mpr->vir, page_rows.size() * sizeof(int32_t));
        }
        {
            axera_timer tl("prefill (page)");
            if (pre_head) {  // the last call writes the logits: its head flags between the calls
                for (size_t k = 0; k < pres.size(); ++k) {
                    const bool last_call = k + 1 == pres.size();
                    *(int32_t *) head.mgh->vir = last_call && !page_rows.empty() ? 1 : 0;
                    *(int32_t *) head.mgt->vir = last_call ? (int32_t) page_rows.size() : 0;
                    head.mgh->flush(head.mgh->vir, sizeof(int32_t));
                    head.mgt->flush(head.mgt->vir, sizeof(int32_t));
                    pres[k].k->run(pres[k].in, pres[k].out, "prefill");
                }
            } else {
                axera_npu_batch nb;  // the page's dependent calls, waited for once
                for (auto & c : pres) {
                    nb.add(*c.k, c.in, c.out, "prefill");
                }
                nb.wait();
            }
            state_zero = false;
        }
        if (!page_rows.empty()) {
            const int64_t N = logits->ne[0];
            head.mvl->invalidate(head.mvl->vir, (size_t) page_rows.size() * N * sizeof(float));
            for (size_t j = 0; j < page_rows.size(); ++j) {
                memcpy((char *) logits->data + (size_t) page_rows[j].first * logits->nb[1],
                       (const char *) head.mvl->vir + j * N * sizeof(float), (size_t) N * sizeof(float));
            }
            if (ggml_backend_buffer_is_axera(logits->buffer)) {
                auto * dctx       = (ggml_backend_axera_buffer_context *) logits->buffer->context;
                dctx->clean_p     = logits->data;
                dctx->clean_n     = (size_t) logits->ne[1] * logits->nb[1];
                dctx->clean_epoch = axera_graph_epoch;
            }
        }
        axera_timer       to("prefill out rows");
        const axera_cmm & po = POUT[(ncall - 1) & 1];
        po->invalidate(po->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
    }
}

// T (2 - ver_rows) consecutive tokens from cell c0 > 0 through the verification groups, their hidden states into h at the
// batch rows: rows T..3 padded (the last row's positions and mask, later cells: KV rows nobody reads); the state after
// each row in the other snapshot set, the current state the last row's
ggml_status qwen35_whole::compute(ggml_cgraph * g) {
    const auto &  t       = names.of(g);
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return mtp.compute(mtp_env(), g);
    }
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen35_whole: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0);
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    // stop before a nextn / MTP block: its attn_norm is not a target-model layer
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight") &&
           !t.count("blk." + std::to_string(L_) + ".nextn.eh_proj.weight")) {
        ++L_;
    }
    const ggml_tensor * rope_node = llm_rope_node(g);
    GGML_ASSERT(rope_node && "axera qwen35_whole: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const bool merged = man.get("decode.group").find_input("head") >= 0 && man.get("prefill").find_input("head_on") >= 0;
    pre_logits = nullptr;
    pre_rows.clear();
    if (merged) {
        const ggml_tensor * ow = logits->src[0];
        dec_head = head.bind_merged(man.get("decode.group"), ow);
        pre_head = dec_head && head.bind_merged_prefill(man.get("prefill"), ow);
        GGML_ASSERT(dec_head && pre_head);
        head.bind_merged_logits(T == 1 ? logits : nullptr);
        bind_head_calls(logits);
        ggml_tensor * hn = logits->src[1];
        while (hn->op == GGML_OP_GET_ROWS || hn->op == GGML_OP_VIEW || hn->op == GGML_OP_RESHAPE) {
            hn = hn->src[0];
        }
        float         eps = 0.0f;
        const float * gw  = final_norm(hn, H, &eps);
        GGML_ASSERT(gw && eps == 1e-6f);
        head.set_merged_norm(gw, H);
        // the prefill head's rows for every graph: a one-row batch the sequence state rewinds runs through a page
        // replay (the prefill kernel), not the decode kernel
        pre_logits = logits;
        const int n_out = (int) logits->ne[1];
        if (n_out > 4) {
            GGML_ABORT("axera qwen35_whole: a prompt asking for %d logits rows; the merged prefill head covers 4\n", n_out);
        }
        llm_out_rows(g, names, n_out, T, pre_rows);
        if (T == 1) {
            head.prepare_merged_topk();
        }
    } else {
        dec_head = pre_head = false;
    }
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    int64_t     c0 = 0;
    if (feed(names, man.context, x0, inp_pos, rope_node, T, "qwen35_whole", c0) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    std::vector<float> & h = h_buf;
    if (dec_head) {  // the kernels wrote the logits (a prompt's in prefill); a decode publishes them
        if (T == 1 && strcmp(seq->path(), "decode") == 0) {  // the decode kernel's head wrote this row
            head.finish_merged_decode(g, names, h, T, logits);
        } else if (T == 1) {  // a page replay: the prefill head wrote the row's logits
            output_head::after(g, names);
        }
        return GGML_STATUS_SUCCESS;
    }
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    if (man.has("lm_head.topk")) {
        head.bind_topk(man.get("lm_head.topk"));
    }
    const bool combined = head.bind_combined(man.get("lm_head"), logits->src[0]) && head.combined_ready(g, names);
    // Only a four-row verify graph uses this head.  Binding it during a plain T=1 graph would allocate its CMM
    // buffers and perturb the address layout of dormant MTP requests (adaptive prose/sampled), which can flip
    // near-tie greedy tokens relative to the plain manifest.
    if (T == 4 && man.has("lm_head.verify")) {
        head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
    }
    if (combined) {
        head.run_combined(g, names, h, T, nullptr, nullptr);
    } else {
        head.run(g, names, h, T);
    }
    return GGML_STATUS_SUCCESS;
}

// the MTP block's kernels: its packs from blk.<il>.* (both kernels the same layout), its paged KV cache, the IO
// a draft context's MTP graph: token x_p (its embedding row) with the target's final-norm hidden of p - 1 through the
// MTP block at the draft context's cells; h_nextn (the block's shared head norm) and the logits of the output rows
}  // namespace

std::shared_ptr<model> make_qwen35_whole(const manifest & m) {
    return std::make_shared<qwen35_whole>(m);
}


gdn_mtp_env qwen35_whole::mtp_env() {
    gdn_mtp_env e;
    e.who          = "qwen35_whole";
    e.man          = &man;
    e.prm          = &prm;
    e.names        = &names;
    e.head         = &head;
    e.COS          = &COS;
    e.SIN          = &SIN;
    e.H            = H;
    e.ROT          = ROT;
    e.NBLK         = NBLK;
    e.kv_pages     = kv_pages;
    e.kv_cells     = kv_cells;
    e.target_bound = bound != nullptr;
    e.paged        = false;
    e.async        = false;
    for (auto & c : dec) {
        e.live.push_back(&c.in);
    }
    return e;
}

}  // namespace axera
