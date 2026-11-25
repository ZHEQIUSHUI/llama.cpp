// Axera NPU backend: Qwen3.8-27B (64 layers: 16 blocks of 3 gated delta net layers and one gated attention layer,
// dense MLP)
//
// One path, the manifest's kernels (ax-dsl-kernels models/qwen38_27b):
//   prefill  role prefill: the whole model in one call per page of 256 cells (a runtime loop over the 16 blocks, the
//            layers' weights in banks [16, ...] indexed by the block), the last page of a batch padded (valid_tokens).
//            Its history is whole pages: a batch that starts inside a page restarts at the page's first cell from the
//            recurrent state saved there, with the page's earlier inputs (kept) before its own
//   decode   role decode.group: 16 consecutive layers per call (weights in packs), 4 calls per token, then lm_head
//   verify   role decode.verify (speculative decoding: a batch of 2 - 4 tokens after the last cell): the decode
//            groups' layers for 4 rows per call, on the decode groups' packs, with the recurrent state after each row
//            (snapshots); a next batch that starts inside it (drafts rejected) continues from the snapshot there
//   mtp      roles mtp.t1 / mtp.prompt: the MTP block (blk.<L>.*, a draft context's graphs) on its own paged KV cache
// The state is this adapter's (CMM pools in the prefill kernel's layouts, the decode group bound to each layer's part):
// the KV cache paged [16][64 pages][K, V][256][4 x 256] bf16, the delta-net states [48][48][128][128] f32 and conv
// states q / k [48][3][2048], v [48][3][6144] bf16 (value heads in HF order); after a verification the current state of
// a delta-net layer is a snapshot's row (where[]). One sequence. Weights only from the GGUF.

#include "../model.h"
#include "../../core/seqstate.h"
#include "../../core/params.h"
#include "../../ops/ops.h"
#include "../../ops/llm.h"
#include "../../ops/gdn_bank.h"
#include "../../ops/gdn_mtp.h"
#include "../../ops/gdn_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>

namespace axera {
namespace {

constexpr int PG = 256;           // cells per page = prefill rows per call = the decode kernels' KV block
constexpr int KH = 16, VH = 48;   // key / value heads of a delta-net layer
constexpr int REP = VH / KH;

using call = npu_call;

class qwen38_27b final : public model, public seq_model, gdn_state {
  public:
    explicit qwen38_27b(const manifest & m) : man(m), prm(m) {}
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

    gdn_mtp     mtp;  // the MTP block (draft context graphs)
    output_head head;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    gdn_mtp_env mtp_env();
    void      prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                      seq_snap * snap);
};

void qwen38_27b::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L = L_;
    H = H_;
    GGML_ASSERT(L % 4 == 0);
    ord.assign(L, 0);
    int ng = 0, na = 0;
    for (int il = 0; il < L; ++il) {
        const bool g = t.count("blk." + std::to_string(il) + ".ssm_a") > 0;
        if (g != (il % 4 != 3)) {
            GGML_ABORT("axera qwen38_27b: layer %d is not the 3 delta-net + 1 attention pattern\n", il);
        }
        ord[il] = g ? ng++ : na++;
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
    auto in_of = [&](const kernel & k, const std::string & n) -> const io_desc & { return gdn_io(k, n, "qwen38_27b"); };
    // A split prefill kernel binds one KV pool per block (b3_kv<block>): every
    // block's K+V is then one IO, below the runtime's 2 GiB IO limit, and a
    // single call can run the whole model.
    auto split_kv_index = [](const std::string & n) -> int {
        if (n.rfind("b3_kv", 0) != 0 || n.size() <= 5) {
            return -1;
        }
        for (size_t i = 5; i < n.size(); ++i) {
            if (!isdigit((unsigned char) n[i])) {
                return -1;
            }
        }
        return atoi(n.c_str() + 5);
    };
    bool split_kv = false;
    for (const auto & d : pk.desc->in) {
        split_kv = split_kv || split_kv_index(d.name) >= 0;
    }
    const io_desc & pkv = in_of(pk, split_kv ? "b3_kv0" : "b3_kv"), &pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in"), &pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const io_desc * pkvs = nullptr;  // an int8 cache: its scales
    for (const auto & d : pk.desc->in) {
        if (d.name == "b3_attn_kv_scale") {
            pkvs = &d;
        }
    }
    const int png = (int) pst.shape[0];  // GDN layers the prefill banks cover (3 per block)
    const int pna = split_kv ? png / 3 : (int) pkv.shape[0];  // the prefill kernel's blocks per call
    GGML_ASSERT(pna > 0 && na % pna == 0 && png == 3 * pna && pcq.shape[0] == png && pcv.shape[0] == png);
    ncall = na / pna;
    const int kv_layers = split_kv ? 1 : pna;  // each b3_kv<block> IO is one layer, a b3_kv bank is pna layers
    kv_b  = pkv.bytes / kv_layers;
    kvs_b = pkvs ? pkvs->bytes / kv_layers : 0;
    st_b  = pst.bytes / png;
    cq_b  = pcq.bytes / png;
    cv_b  = pcv.bytes / png;
    GGML_ASSERT(in_of(dk, "kv3").bytes == kv_b && in_of(dk, "st0").bytes == st_b && in_of(dk, "cq0").bytes == cq_b &&
                in_of(dk, "ck0").bytes == cq_b && in_of(dk, "cv0").bytes == cv_b);
    NBLK = (int) pkv.shape[split_kv ? 0 : 1];
    GGML_ASSERT(pkv.shape[split_kv ? 2 : 3] == PG && (int64_t) NBLK * PG == man.context);
    // llama.cpp's -c sizes the pools unless a multi-layer prefill call needs every page
    kv_pages = llm_kv_pages(NBLK, PG, split_kv || pna == 1);
    GGML_ASSERT(kv_pages == NBLK || pna == 1 || split_kv);  // (a multi-layer bank needs fixed layer strides in its IO)
    kv_cells  = (int64_t) kv_pages * PG;
    kv_alloc  = kv_b / NBLK * kv_pages;
    kvs_alloc = kvs_b / NBLK * kv_pages;
    if (kv_pages < NBLK) {
        GGML_LOG_INFO("axera qwen38_27b: the KV pools hold %d of the kernels' %d pages (%lld cells)\n", kv_pages, NBLK, (long long) kv_cells);
    }
    const int npool = split_kv || pna == 1 ? na : ncall;  // one pool per attention layer, or one per prefill call
    alloc_pools(npool, split_kv ? 1 : pna, ng);  // (a split pool is one block's)
    for (const auto & d : dk.desc->out) {  // the decode kernels' cache IO matches the prefill's (kv<j>, kvs<j>)
        if (d.name.rfind("kvs", 0) == 0) {
            GGML_ASSERT(kvs_b && d.bytes == kvs_b);
        } else if (d.name.rfind("kv", 0) == 0) {
            GGML_ASSERT(d.bytes == kv_b);
        }
    }

    alloc_decode_io(dk, man.context);
    auto pool_io = [&](const std::string & n, int il) { return gdn_state::pool_io(n, il); };
    // the prefill kernel's weight banks [blocks, ..] from the GGUF (first: decode and verification calls may read
    // their row-split weights from them, inputs k_<bank>: the 4 blocks of the call's 16 layers)
    const rope rp(rope_node, names.find("inp_pos"), 1);
    banks.clear();
    {
        axera_timer tb("init: prefill banks");
        static const std::set<std::string> pools = { "x", "position_ids", "indptr", "indices", "last_page_len", "cache_location",
                                                     "valid_tokens", "b3_kv", "b3_attn_kv_scale", "b0_attn_state_in", "b0_attn_q_conv_state_in",
                                                     "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in" };
        for (const auto & d : pk.desc->in) {
            if (!pools.count(d.name) && split_kv_index(d.name) < 0) {
                // the weights of all L / 4 blocks (a call's slice of them bound below); the cos / sin tables are the
                // same for every block: the kernel's blocks of them, bound whole to every call
                const bool cs = d.name.size() > 4 && (d.name.compare(d.name.size() - 4, 4, "_cos") == 0 ||
                                                      d.name.compare(d.name.size() - 4, 4, "_sin") == 0);
                io_desc dd = d;
                if (!cs && ncall > 1) {
                    dd.shape[0] *= ncall;
                    dd.bytes *= ncall;
                }
                banks[d.name] = gdn_prefill_bank({ "qwen38_27b", L, ROT, KH, VH, false }, t, dd, rp);
            }
        }
        axera_bank_sum("qwen38_27b", banks);
    }
    bind_decode("qwen38_27b", man, prm, t, L, H, ng, ord, pool_io);

    // prefill: the weight banks (above), the pools, one page's inputs; three row variants share the banks and pools
    alloc_page_inputs(pk);
    auto bind_prefill = [&](int idx, const char * role) {
        if (!man.has(role)) {
            return;
        }
        gdn_page_set &     p = pfs[idx];
        const kernel &    k = man.get(role);
        const int         rows = (int) (k.in_bytes("x") / (H * sizeof(ggml_bf16_t)));
        GGML_ASSERT(rows >= 2 && rows <= PG);
        p.k                 = &k;
        p.rows              = rows;
        p.X                 = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.OUT[0]            = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.OUT[1]            = axera_cmm_alloc((size_t) rows * H * sizeof(ggml_bf16_t));
        p.POS               = axera_cmm_alloc(k.in_bytes("position_ids"));
        GGML_ASSERT(k.in_bytes("x") == p.X->size && k.in_bytes("position_ids") == 3 * (size_t) rows * sizeof(int32_t));
        GGML_ASSERT(k.desc->out.size() == 1 && k.desc->out[0].name == "output" && k.desc->out[0].bytes == p.OUT[0]->size);
        const io_desc & kv = in_of(k, split_kv ? "b3_kv0" : "b3_kv");
        const io_desc & st = in_of(k, "b0_attn_state_in");
        const int       pna_k = split_kv ? (int) st.shape[0] / 3 : (int) kv.shape[0];
        GGML_ASSERT(pna_k == pna && na % pna_k == 0);
        const int nc = na / pna_k;
        GGML_ASSERT(kv.bytes == kv_b && st.bytes == pst.bytes);
        p.calls.assign(nc, {});
        for (int c_i = 0; c_i < nc; ++c_i) {  // call c_i: blocks c_i * pna .. of the banks, its layers of the pools
            call & c = p.calls[c_i];
            c.k      = &k;
            for (const auto & d : k.desc->in) {
                const std::string & n = d.name;
                const int           ki = split_kv_index(n);
                const axera_cmm *   b = n == "x" ? (c_i == 0 ? &p.X : &p.OUT[(c_i - 1) & 1]) : n == "position_ids" ? &p.POS :
                                        n == "indptr" ? &INDPTR : n == "indices" ? &INDICES : n == "last_page_len" ? &LASTLEN :
                                        n == "cache_location" ? &CLOC : n == "valid_tokens" ? &VALID : nullptr;
                size_t off = 0;
                if (!b && (n == "b3_kv" || ki >= 0)) {
                    b = &KVC[ki >= 0 ? c_i * pna + ki : c_i];
                } else if (!b && n == "b3_attn_kv_scale") {
                    b = &KVSC[c_i];
                } else if (!b) {
                    b = n == "b0_attn_state_in" ? &ST : n == "b0_attn_q_conv_state_in" ? &CQ :
                        n == "b0_attn_k_conv_state_in" ? &CK : n == "b0_attn_v_conv_state_in" ? &CV : nullptr;
                    if (b) {
                        off = (size_t) c_i * d.bytes;  // (the pools: the prefill order of the layers, kind by kind)
                    } else {
                        b   = &banks.at(n);
                        off = (*b)->size >= (size_t) nc * d.bytes ? (size_t) c_i * d.bytes : 0;  // (cos / sin: whole)
                    }
                }
                const bool kvio = n == "b3_kv" || ki >= 0;
                GGML_ASSERT((*b)->size >= off + (kvio ? kv_alloc : n == "b3_attn_kv_scale" ? kvs_alloc : d.bytes));
                c.in.push_back(axera_io(b->get(), (char *) (*b)->vir + off, d.bytes));
            }
            c.out.push_back(axera_io(p.OUT[c_i & 1], k.desc->out[0].bytes));
        }
    };
    // the manifests carry the 256-row prefill and any tail variants (128 / 96 / 80 / 64 rows); the rows are the
    // kernel's x input, not a naming convention, and a missing role is simply not a candidate
    bind_prefill(0, "prefill");
    bind_prefill(1, "prefill128");
    bind_prefill(2, "prefill96");
    bind_prefill(3, "prefill80");
    bind_prefill(4, "prefill64");

    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
    for (auto & c : dec) {
        live.push_back(&c.in);
    }
    prm.release(live);
    ver_rows = man.has("decode.verify")
                   ? (int) (man.get("decode.verify").in_bytes("x") / (H * sizeof(ggml_bf16_t)))
                   : 0;
    // The 27B verify kernel materializes one 4-row snapshot per call; chaining several calls for a 5..32-row tail
    // changes the numerics (and the kernel count) of a path that used to be one prefill page, which flips greedy
    // continuations in long degraded chats. Keep the pre-migration path selection by default; the environment
    // variable still opts into a longer chain.
    seq_config cfg = seq_config_env(1, env_int("GGML_AXERA_VERIFY_CHAIN", ver_rows));
    seq = std::make_unique<seqstate>(*this, cfg);
    size_t bb = 0;
    for (const auto & [n, b] : banks) {
        bb += b->size;
    }
    GGML_LOG_INFO("axera qwen38_27b: %d layers (%d per decode call), prefill banks %.1f GiB, state %.2f GiB, %lld cells\n",
                  L, G, bb / 1073741824.0, ((kv_b + kvs_b) * na + ST->size + CQ->size + CK->size + CV->size) / 1073741824.0,
                  (long long) man.context);
}


seq_caps qwen38_27b::seq_get_caps() const {
    seq_caps c;
    c.page            = PG;
    c.width           = H;
    c.image_bytes     = state_image_b;
    c.cells           = kv_cells;
    c.verify_rows     = ver_rows;
    // The verify kernel updates its entry state in place, so a fully rejected batch has no saved entry image to keep
    // zero rows from. Rewind to the page's restart point instead (one image load, correctness over the rare case).
    c.verify_min_keep = ver_rows > 0 ? 1 : 0;
    c.decode_at_zero  = true;  // the decode group runs cell 0 (the old path did, after a zeroed state)
    return c;
}

// src == nullptr: the zero state of a new sequence (and of the dummy warm-up batch's cleanup).
bool qwen38_27b::seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) {
    state_home();  // the prefill kernels read the pools
    const ggml_bf16_t * xb = (const ggml_bf16_t *) x;
    if (!xb) {
        GGML_ASSERT(batch0 == 0);
        src_rows.resize((size_t) n * H);
        seq_rows(0, n, (uint16_t *) src_rows.data());
        xb = src_rows.data();
    }
    prefill(c0, n, xb, p4, h_buf, batch0, snap);
    return true;
}

bool qwen38_27b::seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) {
    GGML_ASSERT(batch_row == 0);
    (void) p4;  // the row's M-RoPE positions come from the graph's rope view (batch_rope)
    decode(c, x, out_h);
    return true;
}

void qwen38_27b::seq_rows(int batch_row0, int n, uint16_t * dst) {
    rows(batch_row0, n, dst);
}

bool qwen38_27b::seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int batch_row0) {
    (void) p4;  // the rows' M-RoPE positions come from the graph's rope view (batch_rope)
    return verify_rows(c, n, x, batch_row0, man.context);
}

// llama.cpp's sequence state (prompt cache / context checkpoint): core/kv_state's blob of the KV pages before the
// current cell, core/seqstate's restart points (the recurrent state) and, its tail, the MTP cache (gdn_mtp)
kv_state_format qwen38_27b::state_format() const {
    kv_state_format f;
    f.magic   = "AXW27S2";
    f.version = 4;
    f.pages   = kv_pages;
    f.pg      = PG;
    f.cells   = kv_cells;
    return f;
}

cmm_kv_pools qwen38_27b::state_pools() const {
    cmm_kv_pools p;
    const int pna = kv_alloc ? (int) (KVC[0]->size / kv_alloc) : 0;
    p.add(KVC, pna, kv_alloc, kv_b / NBLK);
    p.add(KVSC, pna, kvs_alloc, kvs_b / NBLK);  // the int8 caches' scales
    return p;
}

size_t qwen38_27b::state_size(uint32_t flags) const {
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache: size and bytes must agree
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    return kv_state_size(state_format(), state_pools(), *seq, flags & state_flags_partial_only, &mtp);
}

size_t qwen38_27b::state_write(void * dst, size_t size, uint32_t flags) {
    axera_timer tw("state write");
    mtp.wait();  // an asynchronous MTP catch-up extends the MTP cache
    if (KVC.empty() || !ST || !seq) {
        return 0;
    }
    cmm_kv_pools p = state_pools();
    return kv_state_write(state_format(), p, *seq, dst, size, flags & state_flags_partial_only, &mtp);
}

size_t qwen38_27b::state_read(const void * src, size_t size, uint32_t flags) {
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
void qwen38_27b::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                         seq_snap * snap) {
    gdn_prefill_opts o;
    o.pos_pad   = 1;     // the padding rows repeat the page's first row's positions
    o.seed_out0 = true;  // the three-kernel rotation starts with OUT[0] as the block input
    prefill_pages(c0, n, x, p4, h, out0, snap, o);
}

ggml_status qwen38_27b::compute(ggml_cgraph * g) {
    mtp.wait();
    const auto &  t       = names.of(g);
    if (names.find("mtp_h_input")) {  // a speculative draft context's MTP block
        return mtp.compute(mtp_env(), g);
    }
    ggml_tensor * norm0   = names.find("attn_norm-0");
    ggml_tensor * logits  = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera qwen38_27b: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0);
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = llm_layers(t);
    const ggml_tensor * rope_node = llm_rope_node(g);
    GGML_ASSERT(rope_node && "axera qwen38_27b: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    axera_timer tm_all(T > 1 ? "graph (prompt)" : "graph (decode)");
    int64_t     c0 = 0;
    if (feed(names, man.context, x0, inp_pos, rope_node, T, "qwen38_27b", c0) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    std::vector<float> & h = h_buf;
    // common.cpp marks its warm-up batch with llama_set_dummy_batch(): its state is not needed. Drop it here, while
    // the batch is still being timed as warm-up, so the prompt's own zero state can skip the memset.
    const bool warmup = c0 == 0 && names.find("inp_mixed_slots_dummy");
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    if (man.has("lm_head.topk")) {
        head.bind_topk(man.get("lm_head.topk"));
    }
    const bool combined = head.bind_combined(man.get("lm_head"), logits->src[0]) && head.combined_ready(g, names);
    if (man.has("lm_head.verify")) {
        head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
    }
    if (combined) {
        head.run_combined(g, names, h, T, nullptr, nullptr);
    } else {
        head.run(g, names, h, T);
    }
    if (warmup) {  // the warm-up sequence must not make the first real prompt clear the state
        seq->forget();
        seq_state_load(nullptr);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_qwen38_27b(const manifest & m) {
    return std::make_shared<qwen38_27b>(m);
}


gdn_mtp_env qwen38_27b::mtp_env() {
    gdn_mtp_env e;
    e.who          = "qwen38_27b";
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
    e.paged        = true;
    e.async        = true;
    for (auto & c : dec) {
        e.live.push_back(&c.in);
    }
    return e;
}

}  // namespace axera
