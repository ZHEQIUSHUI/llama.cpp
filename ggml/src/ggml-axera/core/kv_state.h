#pragma once

// The sequence-state blob (llama.cpp's context checkpoints / saved slots) of an adapter whose state besides
// core/seqstate's restart points is paged KV caches (and optionally one more cache of its own, the tail). One format
// for all of them:
//   header | pool 0 pages [0, nkv) | ... | pool n-1 pages [0, nkv) | seqstate blob | tail
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
    virtual size_t    page_b(size_t i) const { (void) i; return 0; }  // bytes of pool i's page (0: the format's)
    virtual void      to_host(size_t i, size_t n) { (void) i; (void) n; }  // before the host reads n bytes of pool i
    virtual void      to_npu(size_t i, size_t n) { (void) i; (void) n; }   // after the host wrote n bytes of pool i
};

// the blob's last section: an adapter's own cache (written, checked, then read; check refuses a blob it can't take)
struct kv_tail {
    virtual ~kv_tail() = default;
    virtual size_t tail_size(bool partial) const                                            = 0;
    virtual void   tail_write(uint8_t * dst, bool partial) const                            = 0;
    virtual bool   tail_check(const uint8_t * src, size_t n, bool partial, int64_t cells) const = 0;
    virtual void   tail_read(const uint8_t * src, size_t n)                                 = 0;
};

// what a blob is checked against on read: a blob of another model, kernel set or context is refused (returns 0)
struct kv_state_format {
    const char * magic   = "";   // up to 7 characters: the adapter's
    uint32_t     version = 1;
    size_t       page_b  = 0;    // bytes of one pool's page (pools whose page_b() is 0)
    int          pages   = 0;    // pages a pool holds
    int          pg      = 256;  // cells per page
    int64_t      cells   = 0;    // the context
};

size_t kv_state_size(const kv_state_format & f, const kv_pools & p, const seqstate & seq, bool partial,
                     const kv_tail * tail = nullptr);
size_t kv_state_write(const kv_state_format & f, kv_pools & p, seqstate & seq, void * dst, size_t size, bool partial,
                      const kv_tail * tail = nullptr);
size_t kv_state_read(const kv_state_format & f, kv_pools & p, seqstate & seq, const void * src, size_t size, bool partial,
                     kv_tail * tail = nullptr);

}  // namespace axera
