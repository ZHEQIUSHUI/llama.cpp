// Axera NPU backend: the MTP block of the Qwen3.5 / Qwen3.8 hybrid adapters (a speculative draft context's graphs:
// mtp_h_input, h_nextn): its kernels (roles mtp.t1, mtp.prompt, optional mtp.promptkv / mtp.m8), its paged KV cache and
// that cache's part of the adapter's sequence-state blob. One copy for the 27B and the 0.8B - 9B adapters.
#pragma once

#include "ops.h"
#include "llm.h"
#include "../core/kv_state.h"
#include "../core/params.h"

#include <map>
#include <string>
#include <thread>
#include <vector>

namespace axera {

// what the block takes from its adapter
struct gdn_mtp_env {
    const char *     who = "";
    const manifest * man = nullptr;
    params *         prm = nullptr;
    graph_names *    names = nullptr;
    output_head *    head = nullptr;  // the target's head (its lm_head kernels, combined for the MTP rows)
    axera_cmm *      COS = nullptr;   // the adapter's one-row rotary buffers (scratch)
    axera_cmm *      SIN = nullptr;
    int              H = 0, ROT = 0, NBLK = 0;
    int              kv_pages = 0;    // the pages the adapter's caches hold
    int64_t          kv_cells = 0;
    bool             target_bound = false;  // the target model's graph has run (its kernels are bound)
    bool             paged = false;  // the MTP cache holds kv_pages pages (else the kernel's whole context)
    bool             async = false;  // a catch-up batch nothing reads runs on its own thread
    // the adapter's kernel inputs that may point into converted parameters (kept by prm.release)
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
};

class gdn_mtp final : public kv_tail {
  public:
    ~gdn_mtp() { wait(); }
    // a draft context's MTP graph
    ggml_status compute(const gdn_mtp_env & e, ggml_cgraph * g);
    // an asynchronous catch-up, finished
    void wait() const {
        if (bg.joinable()) {
            bg.join();
        }
    }
    // the sequence is cleared: the cache's used pages zeroed
    void clear();
    // the sequence-state blob's tail (core/kv_state): the next cell and the cache's pages (partial: the page of the
    // last cell only)
    size_t tail_size(bool partial) const override;
    void   tail_write(uint8_t * dst, bool partial) const override;
    bool   tail_check(const uint8_t * src, size_t n, bool partial, int64_t cells) const override;
    void   tail_read(const uint8_t * src, size_t n) override;

  private:
    struct tail_header {
        uint64_t page0, pages, page_b;
        int64_t  cell;
    };
    tail_header blob_span(bool partial) const;
    void init(const gdn_mtp_env & e, const std::map<std::string, ggml_tensor *> & t, int il);

    npu_call  mtp1, mtpp, mtppkv, mtp8;  // mtp8: role mtp.m8 (optional), up to 8 consecutive rows per call
    axera_cmm ME, MH, MCS, MPN, MMASK, MOUT, MRAW, PE, PH, PCS, PPN, PMASK, PNBLK, POUT2, PRAW, MKV;
    axera_cmm PKE, PKH, PKCS;  // the compact prompt kernels use their own row width (not mtp.prompt's 256)
    axera_cmm PK, PV;          // the K/V-only prompt pass' plain row outputs (copied into the paged MKV)
    axera_cmm M8E, M8H, M8CS, M8PN, M8MASK, M8NT, M8OUT, M8RAW;
    size_t    kv_row = 0;
    bool      h_f32  = false;  // the K/V prompt kernel takes FP32 h rows (the EF32 build)
    bool      e_f32  = false;  // the K/V prompt kernel takes FP32 token-embedding rows
    const void *       bound_w = nullptr;
    size_t             page_b  = 0;  // bytes of one MTP KV page
    int64_t            cell    = 0;  // the next MTP KV cell
    std::vector<float> h_buf;        // the graph's hidden rows
    // a verification's MTP catch-up (nothing on the host reads its result: it only extends the MTP block's KV) runs
    // on its own thread while the host goes on (sampling, the next draft's setup); any next graph waits for it
    mutable std::thread bg;
};

}  // namespace axera
