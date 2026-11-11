#pragma once

// The sequence state of a whole-model adapter: which cell the kernels' recurrent state is at, the restart points kept,
// and how a batch gets there (continue / rewind / miss). The adapters share this one implementation; a model only says
// how to save / load its recurrent state and how to run its kernels (seq_model).
//
// The model of the problem (the Qwen3.5-family hybrid layers):
//   - cells grow linearly; the attention KV is written per cell and needs no rewind (later cells are overwritten);
//   - the recurrent state (delta-net state + conv history) is updated in place by the kernels: only whole images of it
//     can be saved and restored;
//   - the prefill kernels start at a page's first cell only.
// So "the state before cell c" = the image at the first cell of c's page (a restart point) + that page's inputs up to c
// run again. This file owns the restart points and that rule; it touches no CMM and no kernel (host unit test:
// tests/test-seqstate.cpp runs it against a fake model).

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace axera {

enum seq_status {
    SEQ_OK = 0,
    SEQ_MISS,       // no restart point reaches the batch's first cell: the caller starts the sequence over from cell 0
    SEQ_BAD_BATCH,  // the batch does not follow the sequence (a gap, an empty batch, past the context)
    SEQ_FAILED,     // a kernel run failed
};

struct seq_caps {
    int     page        = 256;  // cells per page
    int     width       = 0;    // an input row's elements (bf16)
    size_t  image_bytes = 0;    // one image of the recurrent state
    int64_t cells       = 0;    // the context
    // verification kernel (rows per call; 0: none). After run_verify the state is pending: verify_keep(k) makes it
    // the state after the first k rows, k in [verify_min_keep, rows]
    int  verify_rows     = 0;
    int  verify_min_keep = 0;
    bool decode_at_zero  = true;  // the decode kernel runs cell 0 (else cell 0 always goes through the pages)
};

// a restart point a page run leaves behind: the state before `cell` (a page's first cell inside the run) into dst;
// a model that still holds that state in a spare set sets `deferred` instead of copying (seq_snap_fetch copies it
// later; it stays valid until the model's next seq_run_pages)
struct seq_snap {
    int64_t   cell     = -1;
    uint8_t * dst      = nullptr;
    bool      deferred = false;
};

// What a model provides. Rows are bf16 [n][width]; positions int32 [n][4]. x == nullptr: the rows of the current
// batch from the model's own source (e.g. token ids embedded on the NPU), batch rows from batch0.
// batch0: the index in these rows of the batch's row 0 (rows before it are a replay: no output wanted).
struct seq_model {
    virtual ~seq_model() = default;
    virtual seq_caps seq_get_caps() const = 0;
    // the whole recurrent state <-> a host image (the model folds anything pending first); src == nullptr: zero.
    // seq_state_load also cancels a pending verification.
    virtual void seq_state_save(uint8_t * dst)       = 0;
    virtual void seq_state_load(const uint8_t * src) = 0;
    // a new sequence starts (after seq_state_load(nullptr)): whatever else the model keeps per sequence
    virtual void seq_new_sequence() {}
    // n rows from the page's first cell c0; snap (optional): see seq_snap
    virtual bool seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) = 0;
    virtual bool seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) = 0;
    // the bf16 input rows [batch_row0, batch_row0 + n) of the current batch (for the x == nullptr runs)
    virtual void seq_rows(int batch_row0, int n, uint16_t * dst) = 0;
    virtual bool seq_snap_fetch(uint8_t * /*dst*/) { return false; }
    // verification (caps.verify_rows > 0)
    virtual bool seq_run_verify(int64_t /*c*/, int /*n*/, const uint16_t * /*x*/, const int32_t * /*p4*/, int /*batch_row0*/) {
        return false;
    }
    virtual void seq_verify_keep(int /*k*/) {}
    // the image of the state after the first k rows of the pending verification, which stays pending
    virtual void seq_verify_peek(int /*k*/, uint8_t * /*dst*/) {}
};

struct seq_config {
    int  kept_max     = 1;      // restart points kept besides the current page's (the pages where prompts ended)
    int  verify_chain = 0;      // a continuing batch of up to this many rows runs as chained verify calls (0: off)
    bool check        = false;  // check the invariants after every call (GGML_AXERA_SEQ_CHECK)
    int  force_miss   = 0;      // (tests) force the n-th batch that needs a restart point to report SEQ_MISS instead
};

class seqstate {
  public:
    seqstate(seq_model & m, const seq_config & cfg) : m(m), cfg(cfg), caps(m.seq_get_caps()) {}

    // a batch of T consecutive cells from c0
    seq_status feed(int64_t c0, int T, const uint16_t * x, const int32_t * p4);
    // drop everything (the model's state is unknown or about to be replaced)
    void forget();

    int64_t      cell() const { return state_cell; }       // the cell the state is before; -1: only restart points
    const char * path() const { return last_path; }        // how the last batch ran (logs, tests)
    const char * why() const { return last_why; }          // what the last failure was

    // the blob in a context checkpoint / saved slot. slim: what continuing from the current cell needs (the current
    // state and the current page's restart point); else the kept restart points too.
    size_t size(bool slim) const;
    size_t write(uint8_t * dst, size_t n, bool slim);
    size_t read(const uint8_t * src, size_t n);

    // the restart points (tests, logs)
    struct point {
        int64_t               cell   = -1;     // -1: none
        bool                  zero   = false;  // the zero state (cell 0): no image
        bool                  prompt = false;  // a prompt ended in this page: kept when the sequence moves on
        bool                  deferred = false;  // the image is still in the model (seq_snap_fetch)
        int                   rows   = 0;      // inputs recorded from `cell`
        std::vector<uint8_t>  image;
        std::vector<uint16_t> x;
        std::vector<int32_t>  p;
    };
    const point &             current() const { return cur; }
    const std::deque<point> & kept_points() const { return kept; }

  private:
    seq_model & m;
    seq_config  cfg;
    seq_caps    caps;

    int64_t           state_cell = 0;
    point             cur;
    std::deque<point> kept;  // newest first
    std::vector<point> spare;  // buffers to reuse
    int64_t           ver_c0 = -1;  // a pending verification: its first cell, rows
    int               ver_T  = 0;
    int               fallbacks = 0;  // batches that needed a restart point (GGML_AXERA_SEQ_FORCE_MISS)
    const char *      last_path = "";
    const char *      last_why  = "";
    std::vector<uint16_t> xs;  // scratch: a replay's rows
    std::vector<int32_t>  ps;
    std::vector<uint16_t> row_tmp;

    point  take();
    void   give(point && p);
    void   retire_cur();
    void    begin_page(int64_t c, int peek_k);
    void    record(const uint16_t * xr, const int32_t * pr, int batch_row0, int n);
    void    materialize(point & p);
    void    drop_after(int64_t c);
    void    truncate_to(int64_t c);
    point * find(int64_t s0, int k);
    seq_status run_decode(int64_t c0, const uint16_t * x, const int32_t * p4);
    seq_status run_verify(int64_t c0, int T, const uint16_t * x, const int32_t * p4);
    seq_status run_pages(int64_t s0, int k, int T, const uint16_t * x, const int32_t * p4, bool direct);
    seq_status fail(seq_status s, const char * why) {
        last_why = why;
        return s;
    }
    void check() const;
};

}  // namespace axera
