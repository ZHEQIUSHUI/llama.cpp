#include "kv_state.h"

#include <algorithm>
#include <cstring>

namespace axera {
namespace {

struct header {
    char     magic[8];
    uint32_t version;
    uint32_t partial;
    uint32_t npools;
    uint32_t pg;
    int64_t  state_cell;
    uint64_t page_b;
    uint32_t nkv_pages;  // pages per pool in the blob
    uint32_t reserved;
    uint64_t seq_n;
};

// the pages before the state's cell (a partial blob: none)
int kv_pages_of(const kv_state_format & f, const seqstate & seq, bool partial) {
    if (partial) {
        return 0;
    }
    const int64_t cell = std::max<int64_t>(0, seq.cell());
    return (int) std::min<int64_t>(f.pages, (cell + f.pg - 1) / f.pg);
}

}  // namespace

size_t kv_state_size(const kv_state_format & f, const kv_pools & p, const seqstate & seq, bool partial) {
    const size_t seq_n = seq.size(partial);
    if (!seq_n) {
        return 0;
    }
    return sizeof(header) + p.count() * (size_t) kv_pages_of(f, seq, partial) * f.page_b + seq_n;
}

size_t kv_state_write(const kv_state_format & f, kv_pools & p, seqstate & seq, void * dst, size_t size, bool partial) {
    const size_t total = kv_state_size(f, p, seq, partial);
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
    h.page_b     = f.page_b;
    h.nkv_pages  = (uint32_t) nkv;
    h.seq_n      = seq.size(partial);
    uint8_t * o  = (uint8_t *) dst;
    memcpy(o, &h, sizeof(h));
    o += sizeof(h);
    const size_t n = (size_t) nkv * f.page_b;
    for (size_t i = 0; n && i < p.count(); ++i) {
        p.to_host(i, n);
        memcpy(o, p.pages(i), n);
        o += n;
    }
    if (seq.write(o, h.seq_n, partial) != h.seq_n) {
        return 0;
    }
    o += h.seq_n;
    return (size_t) (o - (uint8_t *) dst);
}

size_t kv_state_read(const kv_state_format & f, kv_pools & p, seqstate & seq, const void * src, size_t size, bool partial) {
    header h;
    if (size < sizeof(h)) {
        return 0;
    }
    memcpy(&h, src, sizeof(h));
    char magic[sizeof(h.magic)] = {};
    strncpy(magic, f.magic, sizeof(magic) - 1);
    if (memcmp(h.magic, magic, sizeof(magic)) != 0 || h.version != f.version || h.partial != (uint32_t) partial ||
        h.npools != p.count() || h.pg != (uint32_t) f.pg || h.page_b != f.page_b || !h.seq_n ||
        h.state_cell < -1 || h.state_cell > f.cells || h.nkv_pages > (uint32_t) f.pages ||
        (partial && h.nkv_pages != 0)) {
        return 0;
    }
    const size_t n = (size_t) h.nkv_pages * f.page_b;
    if (size < sizeof(h) + p.count() * n + h.seq_n) {
        return 0;
    }
    const uint8_t * s = (const uint8_t *) src + sizeof(h);
    for (size_t i = 0; n && i < p.count(); ++i) {
        memcpy(p.pages(i), s, n);
        p.to_npu(i, n);
        s += n;
    }
    if (seq.read(s, h.seq_n) != h.seq_n) {
        return 0;
    }
    s += h.seq_n;
    return (size_t) (s - (const uint8_t *) src);
}

}  // namespace axera
