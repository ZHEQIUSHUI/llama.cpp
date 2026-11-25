// Host test of core/kv_state: a KV-only fake model (pools of per-cell hashes, no recurrent state) under core/seqstate.
// A blob written after a prompt restores the pools and the sequence: the continuation after the restore must equal the
// one without it. Blobs of another format (magic, version, pool count, page size, context) and short buffers are
// refused, and a refused read leaves nothing half-done that a later valid read cannot repair. A tail (an adapter's own
// cache after seqstate's blob) round-trips, and a tail its reader refuses refuses the blob before any pool is written.
//
// Build and run (no SDK needed): ggml/src/ggml-axera/tests/run-host-tests.sh

#include "../core/kv_state.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>

using namespace axera;

namespace {

#define REQUIRE(c, ...)                                         \
    do {                                                        \
        if (!(c)) {                                             \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                       \
            fprintf(stderr, "\n");                              \
            abort();                                            \
        }                                                       \
    } while (0)

constexpr int W     = 4;   // row width
constexpr int PG    = 8;   // cells per page
constexpr int PAGES = 12;
constexpr int NPOOL = 3;
constexpr int CELL_B = 8;  // one cell's bytes in a pool

uint64_t mix(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
    a *= 0xff51afd7ed558ccdull;
    return a ^ (a >> 33);
}

// pool j's cell c: a hash of the row written there and of the cell before it in the same pool (attention's history)
struct fake final : seq_model, kv_pools {
    std::vector<std::vector<uint64_t>> pool = std::vector<std::vector<uint64_t>>(NPOOL, std::vector<uint64_t>((size_t) PAGES * PG));
    const uint16_t * bx = nullptr;
    int synced_host = 0, synced_npu = 0;

    seq_caps seq_get_caps() const override {
        seq_caps c;
        c.page = PG;
        c.width = W;
        c.image_bytes = 0;
        c.cells = (int64_t) PAGES * PG;
        return c;
    }
    void seq_state_save(uint8_t *) override {}
    void seq_state_load(const uint8_t *) override {}
    void put(int64_t c, const uint16_t * x) {
        for (int j = 0; j < NPOOL; ++j) {
            uint64_t h = mix(j + 1, c ? pool[j][c - 1] : 0);
            for (int i = 0; i < W; ++i) {
                h = mix(h, x[i]);
            }
            pool[j][c] = h;
        }
    }
    bool seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t *, int batch0, seq_snap *) override {
        std::vector<uint16_t> rows;
        if (!x) {
            rows.resize((size_t) n * W);
            seq_rows(0, n, rows.data());
            x = rows.data();
            (void) batch0;
        }
        for (int r = 0; r < n; ++r) {
            put(c0 + r, x + (size_t) r * W);
        }
        return true;
    }
    bool seq_run_decode(int64_t c, const uint16_t * x, const int32_t *, int batch_row) override {
        std::vector<uint16_t> row(W);
        if (!x) {
            seq_rows(batch_row, 1, row.data());
            x = row.data();
        }
        put(c, x);
        return true;
    }
    void seq_rows(int batch_row0, int n, uint16_t * dst) override {
        memcpy(dst, bx + (size_t) batch_row0 * W, (size_t) n * W * sizeof(uint16_t));
    }
    // kv_pools
    size_t    count() const override { return NPOOL; }
    uint8_t * pages(size_t i) override { return (uint8_t *) pool[i].data(); }
    void      to_host(size_t, size_t) override { ++synced_host; }
    void      to_npu(size_t, size_t) override { ++synced_npu; }
};

kv_state_format format() {
    kv_state_format f;
    f.magic  = "TESTKV1";
    f.version = 1;
    f.page_b = (size_t) PG * CELL_B;
    f.pages  = PAGES;
    f.pg     = PG;
    f.cells  = (int64_t) PAGES * PG;
    return f;
}

struct run {
    fake     m;
    seqstate s{ m, seq_config{} };
    std::vector<uint16_t> X;  // every row fed, by cell
    void feed(int64_t c0, int T, uint32_t seed) {
        std::vector<uint16_t> x((size_t) T * W);
        std::vector<int32_t>  p((size_t) T * 4);
        for (int r = 0; r < T; ++r) {
            for (int i = 0; i < W; ++i) {
                x[(size_t) r * W + i] = (uint16_t) mix(seed, (uint64_t) (c0 + r) * 31 + i);
            }
            for (int k = 0; k < 4; ++k) {
                p[(size_t) r * 4 + k] = (int32_t) (c0 + r);
            }
        }
        X.resize((size_t) (c0 + T) * W);
        memcpy(X.data() + (size_t) c0 * W, x.data(), x.size() * sizeof(uint16_t));
        m.bx = x.data();
        const seq_status st = s.feed(c0, T, x.data(), p.data());
        REQUIRE(st == SEQ_OK, "feed(%lld, %d): %d %s", (long long) c0, T, (int) st, s.why());
    }
};

bool same_pools(const fake & a, const fake & b, int64_t cells) {
    for (int j = 0; j < NPOOL; ++j) {
        if (memcmp(a.pool[j].data(), b.pool[j].data(), (size_t) cells * CELL_B) != 0) {
            return false;
        }
    }
    return true;
}

// a tail of n bytes, the first its length
struct tail final : kv_tail {
    std::vector<uint8_t> b;
    bool                 refuse = false;
    size_t tail_size(bool) const override { return b.size(); }
    void   tail_write(uint8_t * dst, bool) const override { memcpy(dst, b.data(), b.size()); }
    bool   tail_check(const uint8_t * src, size_t n, bool, int64_t) const override {
        return !refuse && (n == 0 || src[0] == (uint8_t) n);
    }
    void tail_read(const uint8_t * src, size_t n) override { b.assign(src, src + n); }
};

void test_tail(const kv_state_format & f) {
    run a;
    a.feed(0, PG + 3, 11);
    tail ta;
    ta.b = { 5, 1, 2, 3, 4 };
    const size_t n = kv_state_size(f, a.m, a.s, false, &ta);
    REQUIRE(n == kv_state_size(f, a.m, a.s, false) + 5, "the tail's bytes");
    std::vector<uint8_t> blob(n);
    REQUIRE(kv_state_write(f, a.m, a.s, blob.data(), n, false, &ta) == n, "write with a tail");

    run b;
    b.feed(0, 3, 99);
    const auto before = b.m.pool;
    tail tb;
    REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n, false) == 0, "a blob with a tail needs a reader with one");
    tb.refuse = true;
    REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n, false, &tb) == 0, "a refused tail refuses the blob");
    REQUIRE(b.m.pool == before && b.m.synced_npu == 0, "a refused tail: no pool written");
    tb.refuse = false;
    REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n, false, &tb) == n, "read with a tail");
    REQUIRE(tb.b == ta.b, "the tail round-trips");
    REQUIRE(same_pools(a.m, b.m, PG + 3), "the pools with a tail");

    // a blob without a tail into a reader with one: the tail's read sees 0 bytes
    const size_t m = kv_state_size(f, a.m, a.s, false);
    std::vector<uint8_t> plain(m);
    REQUIRE(kv_state_write(f, a.m, a.s, plain.data(), m, false) == m, "write without a tail");
    REQUIRE(kv_state_read(f, b.m, b.s, plain.data(), m, false, &tb) == m && tb.b.empty(), "an empty tail");
    printf("tail: OK\n");
}

}  // namespace

int main() {
    const kv_state_format f = format();
    for (int prompt : { 1, PG - 1, PG, PG + 3, 5 * PG, PAGES * PG - 2 }) {
        // the reference: prompt, then two decodes and a short continuation, uninterrupted
        run ref;
        ref.feed(0, prompt, 11);
        // the saved slot after the prompt
        run a;
        a.feed(0, prompt, 11);
        const size_t n = kv_state_size(f, a.m, a.s, false);
        REQUIRE(n > 0, "prompt %d: empty blob", prompt);
        std::vector<uint8_t> blob(n);
        REQUIRE(kv_state_write(f, a.m, a.s, blob.data(), n - 1, false) == 0, "a short buffer must be refused");
        REQUIRE(kv_state_write(f, a.m, a.s, blob.data(), n, false) == n, "prompt %d: write", prompt);
        REQUIRE(a.m.synced_host == NPOOL, "every pool synced to the host once (%d)", a.m.synced_host);

        // a fresh model restores it and continues
        run b;
        b.feed(0, 3, 99);  // some other sequence first
        for (const char * bad : { "OTHERS1", "TESTKV" }) {
            kv_state_format g = f;
            g.magic = bad;
            REQUIRE(kv_state_read(g, b.m, b.s, blob.data(), n, false) == 0, "magic %s must be refused", bad);
        }
        {
            kv_state_format g = f;
            g.version = 2;
            REQUIRE(kv_state_read(g, b.m, b.s, blob.data(), n, false) == 0, "another version must be refused");
            g = f;
            g.page_b += CELL_B;
            REQUIRE(kv_state_read(g, b.m, b.s, blob.data(), n, false) == 0, "another page size must be refused");
            g = f;
            g.cells = prompt - 1;
            if (prompt > 1) {
                REQUIRE(kv_state_read(g, b.m, b.s, blob.data(), n, false) == 0, "a smaller context must be refused");
            }
            REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n - 1, false) == 0, "a truncated blob must be refused");
            REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n, true) == 0, "a full blob read as partial must be refused");
        }
        REQUIRE(kv_state_read(f, b.m, b.s, blob.data(), n, false) == n, "prompt %d: read", prompt);
        REQUIRE(b.s.cell() == a.s.cell(), "cell %lld != %lld", (long long) b.s.cell(), (long long) a.s.cell());
        REQUIRE(same_pools(a.m, b.m, prompt), "prompt %d: restored pools differ", prompt);

        // the continuation is the reference's
        int64_t c = prompt;
        for (int k = 0; k < 2 && c < (int64_t) PAGES * PG; ++k, ++c) {
            ref.feed(c, 1, 77 + k);
            b.feed(c, 1, 77 + k);
        }
        const int more = (int) std::min<int64_t>(PG + 1, (int64_t) PAGES * PG - c);
        if (more > 0) {
            ref.feed(c, more, 5);
            b.feed(c, more, 5);
            c += more;
        }
        REQUIRE(same_pools(ref.m, b.m, c), "prompt %d: continuation after the restore differs", prompt);

        // a partial blob carries no KV pages and round-trips the sequence
        const size_t pn = kv_state_size(f, a.m, a.s, true);
        REQUIRE(pn > 0 && pn < n, "partial blob %zu (full %zu)", pn, n);
        std::vector<uint8_t> pb(pn);
        REQUIRE(kv_state_write(f, a.m, a.s, pb.data(), pn, true) == pn, "partial write");
        REQUIRE(kv_state_read(f, a.m, a.s, pb.data(), pn, false) == 0, "a partial blob read as full must be refused");
        REQUIRE(kv_state_read(f, a.m, a.s, pb.data(), pn, true) == pn, "partial read");
        printf("prompt %3d: blob %6zu bytes (partial %zu): restore + continue OK\n", prompt, n, pn);
    }
    test_tail(f);
    printf("test-kv-state: PASS\n");
    return 0;
}
