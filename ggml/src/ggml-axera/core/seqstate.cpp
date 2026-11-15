#include "seqstate.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace axera {

namespace {

struct blob_header {
    char     magic[8];
    uint32_t version;
    uint32_t page;
    uint32_t width;
    uint32_t flags;  // 1: slim, 2: the current state's image follows the header, 4: the first point is the current page's
    uint64_t image_bytes;
    int64_t  state_cell;
    uint32_t n_points;
    uint32_t reserved;
};

struct blob_point {
    int64_t  cell;
    uint32_t flags;  // 1: zero, 2: prompt
    uint32_t rows;
};

const char BLOB_MAGIC[8] = { 'A', 'X', 'S', 'E', 'Q', '0', '1', 0 };

}  // namespace

seqstate::point seqstate::take() {
    point p;
    if (!spare.empty()) {
        p = std::move(spare.back());
        spare.pop_back();
    }
    p.cell = -1;
    p.zero = p.prompt = p.deferred = false;
    p.rows = 0;
    p.x.clear();
    p.p.clear();
    return p;
}

void seqstate::give(point && p) {
    if (p.image.capacity() && spare.size() < 2) {
        spare.push_back(std::move(p));
    }
    p = point();
}

// the current page's restart point leaves: kept if a prompt ended in its page
void seqstate::retire_cur() {
    if (cur.cell >= 0 && cur.prompt && cfg.kept_max > 0) {
        kept.push_front(std::move(cur));
        while ((int) kept.size() > cfg.kept_max) {
            give(std::move(kept.back()));
            kept.pop_back();
        }
        cur = point();
    } else {
        give(std::move(cur));
    }
}

// a restart point at c (a page's first cell): the model's current state, or the state after peek_k rows of the pending
// verification
void seqstate::begin_page(int64_t c, int peek_k) {
    retire_cur();
    cur      = take();
    cur.cell = c;
    if (c == 0) {
        cur.zero = true;
        return;
    }
    cur.image.resize(caps.image_bytes);
    if (peek_k >= 0) {
        m.seq_verify_peek(peek_k, cur.image.data());
    } else {
        m.seq_state_save(cur.image.data());
    }
}

// n input rows after the current restart point's (xr == nullptr: the batch's rows from batch_row0, from the model)
void seqstate::record(const uint16_t * xr, const int32_t * pr, int batch_row0, int n) {
    const size_t W = (size_t) caps.width;
    const size_t o = cur.x.size();
    cur.x.resize(o + (size_t) n * W);
    if (xr) {
        memcpy(cur.x.data() + o, xr, (size_t) n * W * sizeof(uint16_t));
    } else {
        m.seq_rows(batch_row0, n, cur.x.data() + o);
    }
    cur.p.insert(cur.p.end(), pr, pr + (size_t) n * 4);
    cur.rows += n;
}

bool seqstate::materialize(point & p) {
    if (!p.deferred) {
        return true;
    }
    p.image.resize(caps.image_bytes);
    if (!m.seq_snap_fetch(p.image.data())) {  // (a model that defers must be able to fetch)
        fprintf(stderr, "axera seqstate: the deferred restart point of cell %lld is gone\n", (long long) p.cell);
        p.cell = -1;
        p.deferred = false;
        return false;
    }
    p.deferred = false;
    return true;
}

// the sequence is cut at c: the restart points of later pages describe cells that are gone
void seqstate::drop_after(int64_t c) {
    const int64_t s0 = c - c % caps.page;
    for (size_t i = 0; i < kept.size();) {
        if (kept[i].cell > s0) {
            give(std::move(kept[i]));
            kept.erase(kept.begin() + (long) i);
        } else {
            ++i;
        }
    }
    if (cur.cell > s0) {
        give(std::move(cur));
    }
}

// the restart point of the page from s0 that has the page's first k inputs, made the current one
seqstate::point * seqstate::find(int64_t s0, int k) {
    drop_after(s0);
    if (cur.cell == s0 && cur.rows >= k) {
        return &cur;
    }
    for (size_t i = 0; i < kept.size(); ++i) {
        if (kept[i].cell == s0 && kept[i].rows >= k) {
            point p = std::move(kept[i]);
            kept.erase(kept.begin() + (long) i);
            if (cur.cell == s0) {  // (the same page with fewer inputs: of no use beside p)
                give(std::move(cur));
            } else {
                retire_cur();  // (an earlier page's)
            }
            cur = std::move(p);
            return &cur;
        }
    }
    return nullptr;
}

// the records after a cut at c inside the sequence whose state the model already has (a verification's accepted rows)
void seqstate::truncate_to(int64_t c) {
    const int64_t s0 = c - c % caps.page;
    const int     k  = (int) (c - s0);
    if (point * p = find(s0, k)) {
        p->rows = k;
        p->x.resize((size_t) k * caps.width);
        p->p.resize((size_t) k * 4);
    } else if (cur.cell >= 0) {  // (the page of c has no restart point: nothing to rewind to inside it)
        retire_cur();
    }
}

void seqstate::forget() {
    give(std::move(cur));
    while (!kept.empty()) {
        give(std::move(kept.back()));
        kept.pop_back();
    }
    state_cell = -1;
    ver_c0     = -1;
    ver_T      = 0;
}

seq_status seqstate::feed(int64_t c0, int T, const uint16_t * x, const int32_t * p4) {
    last_why  = "";
    last_path = "";
    if (T <= 0 || c0 < 0 || c0 + T > caps.cells) {
        return fail(SEQ_BAD_BATCH, "an empty batch or cells outside the context");
    }
    const int PG = caps.page;
    if (ver_c0 >= 0) {  // the last verification: the rows up to c0 were accepted
        if (c0 > ver_c0 + ver_T) {
            return fail(SEQ_BAD_BATCH, "a batch past the last verification's rows");
        }
        if (c0 != 0 && c0 >= ver_c0 + caps.verify_min_keep) {
            m.seq_verify_keep((int) (c0 - ver_c0));
            state_cell = c0;
            if (c0 < ver_c0 + ver_T) {
                truncate_to(c0);
            }
        } else {  // before its rows: the state comes from a restart point (seq_state_load cancels the verification)
            state_cell = -1;
        }
        ver_c0 = -1;
        ver_T  = 0;
    }
    if (c0 == 0) {  // a new sequence
        forget();
        m.seq_state_load(nullptr);
        m.seq_new_sequence();
        state_cell = 0;
    }
    const int64_t s0 = c0 - c0 % PG;
    const int     k  = (int) (c0 - s0);
    if (state_cell >= 0 && c0 > state_cell) {
        return fail(SEQ_BAD_BATCH, "a batch that leaves a gap after the state's cell");
    }
    if (state_cell == c0) {
        if (T == 1 && (c0 > 0 || caps.decode_at_zero)) {
            return run_decode(c0, x, p4);
        }
        const int vr = caps.verify_rows;
        // one verify call writes all `vr` snapshot rows even when the batch is shorter; a chain's last call needs its
        // rows after the batch's tail. Keep the pre-migration condition for both adapters near the context's end.
        const int need = T <= vr ? vr : T + vr;
        if (vr > 0 && c0 > 0 && T > 1 && (T <= vr || T <= cfg.verify_chain) && c0 + need <= caps.cells) {
            return run_verify(c0, T, x, p4);
        }
        if (k == 0) {
            last_path = "pages";
            return run_pages(s0, 0, T, x, p4, true);
        }
    }
    if (cfg.force_miss > 0 && ++fallbacks == cfg.force_miss) {
        return fail(SEQ_MISS, "forced miss (GGML_AXERA_SEQ_FORCE_MISS)");
    }
    if (!find(s0, k)) {
        state_cell = state_cell == c0 ? state_cell : -1;  // (find dropped what the cut made stale)
        return fail(SEQ_MISS, "no restart point for the batch's page");
    }
    last_path = state_cell == c0 ? "replay" : "rewind";
    return run_pages(s0, k, T, x, p4, false);
}

seq_status seqstate::run_decode(int64_t c0, const uint16_t * x, const int32_t * p4) {
    last_path        = "decode";
    const int64_t s0 = c0 - c0 % caps.page;
    const int     k  = (int) (c0 - s0);
    if (k == 0 && !(cur.cell == c0 && cur.rows == 0)) {
        begin_page(c0, -1);
    }
    if (!m.seq_run_decode(c0, x, p4, 0)) {
        forget();
        return fail(SEQ_FAILED, "the decode kernel failed");
    }
    if (cur.cell == s0 && cur.rows == k) {
        record(x, p4, 0, 1);
    } else if (cur.cell >= 0) {
        retire_cur();
    }
    state_cell = c0 + 1;
    check();
    return SEQ_OK;
}

seq_status seqstate::run_verify(int64_t c0, int T, const uint16_t * x, const int32_t * p4) {
    last_path       = T <= caps.verify_rows ? "verify" : "verify chain";
    const int    PG = caps.page, vr = caps.verify_rows;
    const size_t W  = (size_t) caps.width;
    for (int r0 = 0; r0 < T; r0 += vr) {
        const int     n = std::min(vr, T - r0);
        const int64_t c = c0 + r0;
        if (r0 > 0) {  // (a chain accepts its own rows)
            m.seq_verify_keep(ver_T);
            ver_c0 = -1;
        }
        if (c % PG == 0 && !(cur.cell == c && cur.rows == 0)) {
            begin_page(c, -1);
        }
        if (!m.seq_run_verify(c, n, x ? x + (size_t) r0 * W : nullptr, p4 + (size_t) r0 * 4, r0)) {
            forget();
            return fail(SEQ_FAILED, "the verify kernel failed");
        }
        const uint16_t * xr = x ? x + (size_t) r0 * W : nullptr;
        if (!xr) {
            row_tmp.resize((size_t) n * W);
            m.seq_rows(r0, n, row_tmp.data());
            xr = row_tmp.data();
        }
        for (int r = 0; r < n; ++r) {
            const int64_t cr = c + r, sr = cr - cr % PG;
            if (r > 0 && cr == sr) {  // a page's first cell inside the call: the state before it
                begin_page(cr, r);
            }
            if (cur.cell == sr && cur.rows == (int) (cr - sr)) {
                record(xr + (size_t) r * W, p4 + (size_t) (r0 + r) * 4, 0, 1);
            } else if (cur.cell >= 0) {
                retire_cur();
            }
        }
        ver_c0     = c;
        ver_T      = n;
        state_cell = c + n;
    }
    check();
    return SEQ_OK;
}

// the batch's T rows after the first k inputs of the page from s0. direct: the state is at s0 already (k == 0);
// else the current restart point is the page's (find) and is loaded first
seq_status seqstate::run_pages(int64_t s0, int k, int T, const uint16_t * x, const int32_t * p4, bool direct) {
    const int    PG = caps.page;
    const size_t W  = (size_t) caps.width;
    const int    n  = k + T;
    const int64_t end = s0 + n, lastp = end - end % PG;
    const bool    partial = end != lastp;
    const uint16_t * rx = x;
    const int32_t *  rp = p4;
    // A deferred image lives in the model's spare set only until this run's first model call (seq_state_load
    // included), so take every image out before touching the model. In particular, a kept point may still be deferred
    // when cur is loaded for a rewind; materializing it afterwards would find the model's copy already gone.
    if (!materialize(cur)) {
        state_cell = -1;
        return fail(SEQ_MISS, "the current restart point's image is gone");
    }
    for (size_t i = 0; i < kept.size();) {
        if (materialize(kept[i])) {
            ++i;
        } else {
            give(std::move(kept[i]));
            kept.erase(kept.begin() + (long) i);
        }
    }
    if (direct) {
        if (!(cur.cell == s0 && cur.rows == 0) && partial && lastp == s0) {
            begin_page(s0, -1);  // (the run stays in this page: its restart point is the state before the run)
        }
    } else {
        cur.rows = k;
        cur.x.resize((size_t) k * W);
        cur.p.resize((size_t) k * 4);
        materialize(cur);
        if (cur.cell < 0) {
            state_cell = -1;
            return fail(SEQ_MISS, "the restart point's image is gone");
        }
        m.seq_state_load(cur.zero ? nullptr : cur.image.data());
        if (k > 0) {
            xs.resize((size_t) n * W);
            memcpy(xs.data(), cur.x.data(), (size_t) k * W * sizeof(uint16_t));
            if (x) {
                memcpy(xs.data() + (size_t) k * W, x, (size_t) T * W * sizeof(uint16_t));
            } else {
                m.seq_rows(0, T, xs.data() + (size_t) k * W);
            }
            ps.assign(cur.p.begin(), cur.p.end());
            ps.insert(ps.end(), p4, p4 + (size_t) T * 4);
            rx = xs.data();
            rp = ps.data();
        }
    }
    point    np;
    seq_snap snap;
    if (partial && lastp != s0) {
        np = take();
        np.image.resize(caps.image_bytes);
        snap.cell = lastp;
        snap.dst  = np.image.data();
    }
    if (!m.seq_run_pages(s0, n, rx, rp, k, snap.cell >= 0 ? &snap : nullptr)) {
        forget();
        return fail(SEQ_FAILED, "the prefill kernels failed");
    }
    if (cur.cell == s0) {  // the page of s0: its inputs after the first k, as far as the batch fills it
        const int fill = std::min(T, PG - k);
        record(x, p4, 0, fill);
    }
    if (partial && lastp != s0) {
        retire_cur();
        cur          = std::move(np);
        cur.cell     = lastp;
        cur.deferred = snap.deferred;
        const int b  = (int) (lastp - s0) - k;  // the batch row at lastp
        record(x ? x + (size_t) b * W : nullptr, p4 + (size_t) b * 4, b, (int) (end - lastp));
    } else if (!partial) {
        state_cell = end;
        begin_page(end, -1);  // a full last page: the next page starts here
    }
    cur.prompt = true;
    state_cell = end;
    check();
    return SEQ_OK;
}

size_t seqstate::size(bool slim) const {
    if (state_cell < 0 && cur.cell < 0 && kept.empty()) {
        return 0;
    }
    const bool live = state_cell >= 0 && !(cur.cell == state_cell && cur.rows == 0);
    size_t     n    = sizeof(blob_header) + (live ? caps.image_bytes : 0);
    const auto add  = [&](const point & p) {
        n += sizeof(blob_point) + (p.zero ? 0 : caps.image_bytes) + p.x.size() * sizeof(uint16_t) + p.p.size() * sizeof(int32_t);
    };
    if (cur.cell >= 0) {
        add(cur);
    }
    if (!slim) {
        for (const auto & p : kept) {
            add(p);
        }
    }
    return n;
}

size_t seqstate::write(uint8_t * dst, size_t n, bool slim) {
    const size_t total = size(slim);
    if (!total || n < total) {
        return 0;
    }
    const bool  live = state_cell >= 0 && !(cur.cell == state_cell && cur.rows == 0);
    blob_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, BLOB_MAGIC, sizeof(h.magic));
    h.version     = 1;
    h.page        = (uint32_t) caps.page;
    h.width       = (uint32_t) caps.width;
    h.flags       = (slim ? 1u : 0u) | (live ? 2u : 0u) | (cur.cell >= 0 ? 4u : 0u);
    h.image_bytes = caps.image_bytes;
    h.state_cell  = state_cell;
    h.n_points    = (cur.cell >= 0 ? 1u : 0u) + (slim ? 0u : (uint32_t) kept.size());
    uint8_t * o   = dst;
    memcpy(o, &h, sizeof(h));
    o += sizeof(h);
    if (live) {
        if (ver_c0 >= 0) {  // (a verification stays pending: the rows up to the state's cell, all of them)
            m.seq_verify_peek(ver_T, o);
        } else {
            m.seq_state_save(o);
        }
        o += caps.image_bytes;
    }
    const auto put = [&](const point & p) {
        blob_point b;
        b.cell  = p.cell;
        b.flags = (p.zero ? 1u : 0u) | (p.prompt ? 2u : 0u);
        b.rows  = (uint32_t) p.rows;
        memcpy(o, &b, sizeof(b));
        o += sizeof(b);
        if (!p.zero) {
            if (p.deferred) {  // straight from the model's spare set into the blob: one copy, not two
                if (!m.seq_snap_fetch(o)) {
                    return false;
                }
            } else {
                memcpy(o, p.image.data(), caps.image_bytes);
            }
            o += caps.image_bytes;
        }
        memcpy(o, p.x.data(), p.x.size() * sizeof(uint16_t));
        o += p.x.size() * sizeof(uint16_t);
        memcpy(o, p.p.data(), p.p.size() * sizeof(int32_t));
        o += p.p.size() * sizeof(int32_t);
        return true;
    };
    if (cur.cell >= 0) {
        if (!put(cur)) {
            return 0;
        }
    }
    if (!slim) {
        for (const auto & p : kept) {
            if (!put(p)) {
                return 0;
            }
        }
    }
    return (size_t) (o - dst);
}

size_t seqstate::read(const uint8_t * src, size_t n) {
    blob_header h;
    if (n < sizeof(h)) {
        return 0;
    }
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, BLOB_MAGIC, sizeof(h.magic)) != 0 || h.version != 1 || h.page != (uint32_t) caps.page ||
        h.width != (uint32_t) caps.width || h.image_bytes != caps.image_bytes || h.state_cell > caps.cells ||
        h.n_points > 1u + 64u) {
        return 0;
    }
    const bool   live = h.flags & 2u;
    const size_t W    = (size_t) caps.width;
    // the whole blob first: nothing changes unless it is sound
    size_t o = sizeof(h) + (live ? caps.image_bytes : 0);
    if (n < o || (live && h.state_cell < 0)) {
        return 0;
    }
    for (uint32_t i = 0; i < h.n_points; ++i) {
        blob_point b;
        if (n < o + sizeof(b)) {
            return 0;
        }
        memcpy(&b, src + o, sizeof(b));
        const bool zero = b.flags & 1u;
        if (b.cell < 0 || b.cell % caps.page != 0 || b.rows > (uint32_t) caps.page || zero != (b.cell == 0)) {
            return 0;
        }
        o += sizeof(b) + (zero ? 0 : caps.image_bytes) + (size_t) b.rows * (W * sizeof(uint16_t) + 4 * sizeof(int32_t));
        if (n < o) {
            return 0;
        }
    }
    forget();
    o = sizeof(h) + (live ? caps.image_bytes : 0);
    for (uint32_t i = 0; i < h.n_points; ++i) {
        blob_point b;
        memcpy(&b, src + o, sizeof(b));
        o += sizeof(b);
        point p  = take();
        p.cell   = b.cell;
        p.zero   = b.flags & 1u;
        p.prompt = b.flags & 2u;
        p.rows   = (int) b.rows;
        if (!p.zero) {
            p.image.assign(src + o, src + o + caps.image_bytes);
            o += caps.image_bytes;
        }
        p.x.resize((size_t) p.rows * W);
        memcpy(p.x.data(), src + o, p.x.size() * sizeof(uint16_t));
        o += p.x.size() * sizeof(uint16_t);
        p.p.resize((size_t) p.rows * 4);
        memcpy(p.p.data(), src + o, p.p.size() * sizeof(int32_t));
        o += p.p.size() * sizeof(int32_t);
        if (i == 0 && (h.flags & 4u)) {
            cur = std::move(p);
        } else if ((int) kept.size() < std::max(cfg.kept_max, 0)) {
            kept.push_back(std::move(p));
        }
    }
    if (live) {
        m.seq_state_load(src + sizeof(h));
        state_cell = h.state_cell;
    } else if (cur.cell >= 0 && cur.rows == 0 && cur.cell == h.state_cell) {
        m.seq_state_load(cur.zero ? nullptr : cur.image.data());
        state_cell = cur.cell;
    } else {
        state_cell = -1;  // only restart points: the next batch rewinds to one
    }
    check();
    return o;
}

void seqstate::check() const {
    if (!cfg.check) {
        return;
    }
    const auto bad = [&](const char * what) {
        fprintf(stderr, "axera seqstate: invariant broken after '%s': %s (state cell %lld, restart point %lld + %d rows)\n",
                last_path, what, (long long) state_cell, (long long) cur.cell, cur.rows);
        fprintf(stderr, "  cur cell=%lld rows=%d prompt=%d deferred=%d zero=%d\n", (long long) cur.cell, cur.rows,
                (int) cur.prompt, (int) cur.deferred, (int) cur.zero);
        for (size_t i = 0; i < kept.size(); ++i) {
            fprintf(stderr, "  kept[%zu] cell=%lld rows=%d prompt=%d deferred=%d zero=%d\n", i,
                    (long long) kept[i].cell, kept[i].rows, (int) kept[i].prompt, (int) kept[i].deferred,
                    (int) kept[i].zero);
        }
        abort();
    };
    int        deferred = 0;
    const auto point_ok = [&](const point & p) {
        if (p.cell % caps.page != 0 || p.rows < 0 || p.rows > caps.page || p.zero != (p.cell == 0) ||
            p.x.size() != (size_t) p.rows * caps.width || p.p.size() != (size_t) p.rows * 4 ||
            (!p.zero && p.image.size() != caps.image_bytes)) {
            bad("a malformed restart point");
        }
        deferred += p.deferred;
    };
    if (cur.cell >= 0) {
        point_ok(cur);
        if (state_cell >= 0 && cur.cell + cur.rows != state_cell) {
            bad("the current restart point does not reach the state's cell");
        }
    }
    for (size_t i = 0; i < kept.size(); ++i) {
        if (kept[i].cell < 0) {
            bad("an empty kept restart point");
        }
        point_ok(kept[i]);
        if (kept[i].cell == cur.cell) {
            bad("a kept restart point of the current page");
        }
        if (state_cell >= 0 && kept[i].cell > state_cell) {
            bad("a kept restart point past the state's cell");
        }
        for (size_t j = 0; j < i; ++j) {
            if (kept[j].cell == kept[i].cell) {
                bad("two kept restart points of one page");
            }
        }
    }
    if (deferred > 1) {
        bad("more than one deferred image");
    }
    if ((int) kept.size() > std::max(cfg.kept_max, 0)) {
        bad("too many kept restart points");
    }
}

}  // namespace axera
