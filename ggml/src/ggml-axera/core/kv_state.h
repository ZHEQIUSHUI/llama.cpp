#pragma once

// The sequence-state blob (llama.cpp's context checkpoints / saved slots) of an adapter whose only state besides
// core/seqstate's restart points is a paged KV cache: every pool's pages before the state's cell, then seqstate's own
// blob. One format for all of them:
//   header | pool 0 pages [0, nkv) | ... | pool n-1 pages [0, nkv) | seqstate blob
// A partial (slim) blob carries no KV pages. No CMM and no kernel here (host unit test: tests/test-kv-state.cpp).

#include "seqstate.h"

#include <cstddef>
#include <cstdint>

namespace axera {

// the pools a blob reads and writes: count() pools of pages, page 0 at pages(i)
struct kv_pools {
    virtual ~kv_pools() = default;
    virtual size_t    count() const      = 0;
    virtual uint8_t * pages(size_t i)    = 0;
    virtual void      to_host(size_t i, size_t n) { (void) i; (void) n; }  // before the host reads n bytes of pool i
    virtual void      to_npu(size_t i, size_t n) { (void) i; (void) n; }   // after the host wrote n bytes of pool i
};

// what a blob is checked against on read: a blob of another model, kernel set or context is refused (returns 0)
struct kv_state_format {
    const char * magic   = "";   // up to 7 characters: the adapter's
    uint32_t     version = 1;
    size_t       page_b  = 0;    // bytes of one pool's page
    int          pages   = 0;    // pages a pool holds
    int          pg      = 256;  // cells per page
    int64_t      cells   = 0;    // the context
};

size_t kv_state_size(const kv_state_format & f, const kv_pools & p, const seqstate & seq, bool partial);
size_t kv_state_write(const kv_state_format & f, kv_pools & p, seqstate & seq, void * dst, size_t size, bool partial);
size_t kv_state_read(const kv_state_format & f, kv_pools & p, seqstate & seq, const void * src, size_t size, bool partial);

}  // namespace axera
