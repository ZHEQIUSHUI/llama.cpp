#include "kv_state.h"

#include <algorithm>
#include <cstring>

namespace axera {
namespace {

constexpr uint32_t LAYOUT = 2;  // per-pool page sizes, the tail

struct header {
    char     magic[8];
    uint32_t version;
    uint32_t partial;
    uint32_t npools;
    uint32_t pg;
    int64_t  state_cell;
    uint64_t page_b;     // bytes of one page of every pool
    uint32_t nkv_pages;  // pages per pool in the blob
    uint32_t layout;     // LAYOUT (an older blob: 0)
    uint64_t seq_n;
    uint64_t tail_n;
};

// the pages before the state's cell (a partial blob: none)
int kv_pages_of(const kv_state_format & f, const seqstate & seq, bool partial) {
    if (partial) {
        return 0;
    }
    const int64_t cell = std::max<int64_t>(0, seq.cell());
    return (int) std::min<int64_t>(f.pages, (cell + f.pg - 1) / f.pg);
}

size_t page_of(const kv_state_format & f, const kv_pools & p, size_t i) {
    const size_t b = p.page_b(i);
    return b ? b : f.page_b;
}

size_t pages_b(const kv_state_format & f, const kv_pools & p) {
    size_t n = 0;
    for (size_t i = 0; i < p.count(); ++i) {
        n += page_of(f, p, i);
    }
    return n;
}

}  // namespace

size_t kv_state_size(const kv_state_format & f, const kv_pools & p, const seqstate & seq, bool partial,
                     const kv_tail * tail) {
    const size_t seq_n = seq.size(partial);
    if (!seq_n) {
        return 0;
    }
    return sizeof(header) + (size_t) kv_pages_of(f, seq, partial) * pages_b(f, p) + seq_n +
           (tail ? tail->tail_size(partial) : 0);
}

size_t kv_state_write(const kv_state_format & f, kv_pools & p, seqstate & seq, void * dst, size_t size, bool partial,
                      const kv_tail * tail) {
    const size_t total = kv_state_size(f, p, seq, partial, tail);
    if (!total || size < total) {
        return 0;
    }
    const int nkv = kv_pages_of(f, seq, partial);
    header    h;
    memset(&h, 0, sizeof(h));
    strncpy(h.magic, f.magic, sizeof(h.magic) - 1);
    h.version    = f.version;
    h.partial    = partial;
    h.npools     = (uint32_t) p.count();
    h.pg         = (uint32_t) f.pg;
    h.state_cell = seq.cell();
    h.page_b     = pages_b(f, p);
    h.nkv_pages  = (uint32_t) nkv;
    h.layout     = LAYOUT;
    h.seq_n      = seq.size(partial);
    h.tail_n     = tail ? tail->tail_size(partial) : 0;
    uint8_t * o  = (uint8_t *) dst;
    memcpy(o, &h, sizeof(h));
    o += sizeof(h);
    for (size_t i = 0; nkv && i < p.count(); ++i) {
        const size_t n = (size_t) nkv * page_of(f, p, i);
        p.to_host(i, n);
        memcpy(o, p.pages(i), n);
        o += n;
    }
    if (seq.write(o, h.seq_n, partial) != h.seq_n) {
        return 0;
    }
    o += h.seq_n;
    if (h.tail_n) {
        tail->tail_write(o, partial);
        o += h.tail_n;
    }
    return (size_t) (o - (uint8_t *) dst);
}

size_t kv_state_read(const kv_state_format & f, kv_pools & p, seqstate & seq, const void * src, size_t size, bool partial,
                     kv_tail * tail) {
    header h;
    if (size < sizeof(h)) {
        return 0;
    }
    memcpy(&h, src, sizeof(h));
    char magic[sizeof(h.magic)] = {};
    strncpy(magic, f.magic, sizeof(magic) - 1);
    if (memcmp(h.magic, magic, sizeof(magic)) != 0 || h.layout != LAYOUT || h.version != f.version || h.partial != (uint32_t) partial ||
        h.npools != p.count() || h.pg != (uint32_t) f.pg || h.page_b != pages_b(f, p) || !h.seq_n ||
        h.state_cell < -1 || h.state_cell > f.cells || h.nkv_pages > (uint32_t) f.pages ||
        (partial && h.nkv_pages != 0) || (h.tail_n && !tail)) {
        return 0;
    }
    const size_t kv_n = (size_t) h.nkv_pages * h.page_b;
    if (size < sizeof(h) + kv_n + h.seq_n + h.tail_n) {
        return 0;
    }
    const uint8_t * s = (const uint8_t *) src + sizeof(h);
    if (tail && !tail->tail_check(s + kv_n + h.seq_n, h.tail_n, partial, f.cells)) {
        return 0;
    }
    for (size_t i = 0; h.nkv_pages && i < p.count(); ++i) {
        const size_t n = (size_t) h.nkv_pages * page_of(f, p, i);
        memcpy(p.pages(i), s, n);
        p.to_npu(i, n);
        s += n;
    }
    if (seq.read(s, h.seq_n) != h.seq_n) {
        return 0;
    }
    s += h.seq_n;
    if (tail) {
        tail->tail_read(s, h.tail_n);
    }
    s += h.tail_n;
    return (size_t) (s - (const uint8_t *) src);
}

}  // namespace axera
