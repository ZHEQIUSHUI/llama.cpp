// Axera NPU backend: the recurrent state of the Qwen3.5 / Qwen3.8 hybrid adapters' gated delta-net layers: the CMM
// pools (states ST [layers][value heads][key dim][value dim] f32, conv states CQ / CK / CV bf16), the verification
// snapshots, where each layer's current state lives (its pool part, or a snapshot's row after a verification) and
// the decode / verify outputs bound there, and core/seqstate's restart-point images of it. One copy for the 27B and
// the 0.8B - 9B adapters (their base).
#pragma once

#include "llm.h"
#include "../core/params.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace axera {

// a prefill kernel variant: its calls of a page (its blocks each), its x / position inputs and the output rotation
struct gdn_page_set {
    const kernel *        k    = nullptr;
    int                   rows = 0;
    std::vector<npu_call> calls;
    axera_cmm             X, POS, OUT[4];
    int                   result = 0;  // the OUT[] buffer that holds the page's final rows
};

// how an adapter's prefill pages differ
struct gdn_prefill_opts {
    int  pos_pad   = 0;      // the padding rows' positions: 0 zeros, 1 the page's first row's, 2 its last valid row's
    bool seed_out0 = false;  // OUT[0] holds the first call's input too (a rotation that starts there)
    bool lastlen2  = false;  // last_page_len has two entries
    // a merged head: before a page's calls (rows r0 .., v valid) / before each call (then run one by one) / after them
    std::function<void(int r0, int v)> page_begin;
    std::function<void(size_t k, bool last)> each;
    std::function<void()> page_end;
};

struct gdn_state {
    int G = 0;  // the decode group's layers (3 / 4 of them delta-net layers)
    int L = 0, H = 0, ROT = 0, NBLK = 0;  // layers, hidden size, rotary dims, the kernels' KV blocks
    int     kv_pages = 0;  // the pages the KV caches hold
    int64_t kv_cells = 0;

    // the sequence (core/seqstate) and the graph batch it runs: its bf16 rows, KV view and rope; the hidden rows
    std::unique_ptr<seqstate> seq;
    int                       last_seq_status = 0;
    std::vector<float>        h_buf;  // the target graph's hidden rows
    std::vector<ggml_bf16_t>  src_rows;
    const ggml_bf16_t *       batch_rows = nullptr;
    const kv_view *           batch_kv   = nullptr;
    const rope *              batch_rope = nullptr;
    float *                   out_h      = nullptr;

    // the pools; one layer's part of each
    axera_cmm ST, CQ, CK, CV;
    size_t    st_b = 0, cq_b = 0, cv_b = 0;
    size_t    state_image_b = 0;  // one image of ST + CQ + CK + CV
    bool      state_zero    = true;  // the pools hold the zero state (CMM starts zero; a load sets it)

    // the current recurrent state of each delta-net layer: its pool part, or a verification snapshot's row (the
    // decode / verify calls' state outputs bound there); the snapshots of the verification groups, two sets
    struct place {
        const ggml_backend_axera_buffer_context * b = nullptr;
        char *                                    p = nullptr;
    };
    struct state_io {
        npu_call * c;
        size_t     i;
        int        kind, g;  // kind: 0 st, 1 cq, 2 ck, 3 cv; g: the delta-net layer
    };
    std::vector<place>                    where[4];
    std::vector<state_io>                 state_ios;
    std::vector<std::array<axera_cmm, 4>> snaps[2];
    // a delta-net layer -> its verification group / its row in the group's snapshot (empty: consecutive groups of
    // G / 4 * 3 layers)
    std::vector<int> state_group, state_local;
    // the decode groups (role decode.group, G layers each) and their verification calls (decode.verify, optional);
    // their x / out, cos | sin, cell | blocks | K row, mask inputs; the verification's 4-row ones
    std::vector<npu_call>            dec, ver;
    axera_cmm                        A, COS, SIN, CS, PN, MASK;
    axera_cmm                        VX, VCS, VPN, VMASK, VNT;
    std::map<std::string, axera_cmm> banks;
    // the prefill kernel's variants (rows descending; unused ones without a kernel) and their shared page inputs
    gdn_page_set pfs[5];
    axera_cmm    INDPTR, INDICES, LASTLEN, CLOC, VALID;  // the prefill kernel's weight banks (inputs k_<bank> of dec / ver)
    int              cur_set  = -1;  // the snapshot set holding the current state, -1: the pools
    int              ver_set  = -1;  // the snapshot set of the pending verification
    int              ver_rows = 0;

    size_t            part(int kind) const { return kind == 0 ? st_b : kind == 3 ? cv_b : cq_b; }
    const axera_cmm & pool(int kind) const { return kind == 0 ? ST : kind == 1 ? CQ : kind == 2 ? CK : CV; }

    // the decode / verify state outputs onto where[]
    void bind_state();
    // a snapshot's rows copied back into the pools (the current state there again)
    void state_home();
    // the current state: row `row` of snapshot set `set`
    void state_at(int set, int row);
    // seq_model's state image: save; load (nullptr: the zero state; returns true then)
    void state_save(uint8_t * dst);
    bool state_load(const uint8_t * src);
    // a verification's k accepted rows: the state after row k; a copy of it
    void verify_keep(int k);
    void verify_peek(int k, uint8_t * dst) const;

    // dec / ver bound (A, CS, PN, MASK allocated, banks filled, the pools and ord set): the groups' packs and banks,
    // their KV / state IO (pool_io(name, layer)), where[] at the pools, the snapshots and the state outputs
    using gdn_pool_io = std::function<AX_ENGINE_IO_BUFFER_T(const std::string & name, int il)>;
    void bind_decode(const char * who, const manifest & man, params & prm, const std::map<std::string, ggml_tensor *> & t,
                     int L, int H, int ng, const std::vector<int> & ord, const gdn_pool_io & pool_io);
    // one row x [H] bf16 at cell c through the decode groups (the rope / mask from the graph's views): its hidden
    // state into h. each(group, last): called before each group's call, which then run one by one (else batched)
    using gdn_each = std::function<void(size_t gi, bool last)>;
    void decode(int64_t c, const uint16_t * x, float * h, const gdn_each & each = nullptr);
    // a verification batch of T <= 4 rows xb from cell c0 (rows row0 .. of the graph's batch): the hidden rows into h,
    // the state after each row in the other snapshot set, the current state the last row's
    ggml_status verify(int64_t c0, int row0, int T, const ggml_bf16_t * xb, float * h, int64_t context);

    // the graph's batch (T rows x0, positions inp_pos, the cells of its KV inputs) through the sequence: the hidden
    // rows in h_buf, the MTP block's h_nextn input from them; c0 the batch's first cell
    ggml_status feed(const graph_names & names, int64_t context, const ggml_tensor * x0, const ggml_tensor * inp_pos,
                     const ggml_tensor * rope_node, int T, const char * who, int64_t & c0);
    // n rows x [n][H] bf16 (M-RoPE positions p4 [n][4]) from cell c0 (a page's first) through the prefill kernel,
    // page by page in the smallest variant holding each: the hidden rows from out0 on into h; snap: the state before
    // a last partial page
    void prefill_pages(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0,
                       seq_snap * snap, const gdn_prefill_opts & o);
    // seq_model's rows of the batch (zeros without one) and its verification
    void rows(int batch_row0, int n, uint16_t * dst) const;
    bool verify_rows(int64_t c, int n, const uint16_t * x, int batch_row0, int64_t context);

  private:
    int  group_of(size_t g) const { return state_group.empty() ? (int) g / (G / 4 * 3) : state_group[g]; }
    int  local_of(size_t g) const { return state_local.empty() ? (int) g % (G / 4 * 3) : state_local[g]; }
    char * snap_row(int set, size_t g, int kind, int row) const;
};

}  // namespace axera
