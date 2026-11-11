// Host test of core/seqstate against a fake model: the "recurrent state" is a hash chain over the input rows, the "KV"
// one hash per cell. Random continue / verify / rewind / new sequence / checkpoint traffic; after every batch the
// model's state, the KV cells and the batch's outputs must equal the ones of the true history run from cell 0.
//
// Build and run (no SDK needed): ggml/src/ggml-axera/tests/run-host-tests.sh

#include "../core/seqstate.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace axera;

namespace {

constexpr int W = 4;

#define REQUIRE(c, ...)                                         \
    do {                                                        \
        if (!(c)) {                                             \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                       \
            fprintf(stderr, "\n");                              \
            abort();                                            \
        }                                                       \
    } while (0)

uint64_t mix(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
    a *= 0xff51afd7ed558ccdull;
    return a ^ (a >> 33);
}

uint64_t row_hash(const uint16_t * x, const int32_t * p) {
    uint64_t h = 0x1234;
    for (int i = 0; i < W; ++i) {
        h = mix(h, x[i]);
    }
    for (int i = 0; i < 4; ++i) {
        h = mix(h, (uint32_t) p[i]);
    }
    return h;
}

struct image {
    uint64_t h;
    int64_t  cell;
};

struct fake final : seq_model {
    seq_caps caps;
    image    st{ 7, 0 };
    std::vector<uint64_t> kv;
    std::vector<uint64_t> out;       // the batch's outputs: the state hash after each row
    const uint16_t *      bx = nullptr;  // the batch's rows (the x == nullptr runs)
    bool     pending = false;
    int      vn      = 0;
    image    vst[9];
    bool     def_valid = false;
    image    def{};
    double   defer_p = 0;
    std::mt19937 * rng = nullptr;
    std::map<std::string, long> calls;

    seq_caps seq_get_caps() const override { return caps; }
    void     step(image & s, const uint16_t * x, const int32_t * p) {
        const uint64_t r = row_hash(x, p);
        kv[(size_t) s.cell] = r;
        s.h                 = mix(s.h, r);
        s.cell++;
    }
    void seq_state_save(uint8_t * dst) override {
        REQUIRE(!pending, "state_save with a verification pending");
        memcpy(dst, &st, sizeof(st));
        calls["save"]++;
    }
    void seq_state_load(const uint8_t * src) override {
        pending = false;
        if (src) {
            memcpy(&st, src, sizeof(st));
        } else {
            st = { 7, 0 };
        }
        calls["load"]++;
    }
    bool seq_run_pages(int64_t c0, int n, const uint16_t * x, const int32_t * p4, int batch0, seq_snap * snap) override {
        REQUIRE(!pending, "pages with a verification pending");
        REQUIRE(c0 % caps.page == 0, "pages from cell %lld, not a page start", (long long) c0);
        REQUIRE(st.cell == c0, "pages from cell %lld with the state at %lld", (long long) c0, (long long) st.cell);
        REQUIRE(c0 + n <= caps.cells, "pages past the context");
        REQUIRE(x || batch0 == 0, "model-sourced rows in a replay");
        if (!x) {
            x = bx;
        }
        def_valid = false;
        if (snap) {
            REQUIRE(snap->cell % caps.page == 0 && snap->cell > c0 && snap->cell < c0 + n, "a bad snapshot cell");
        }
        for (int i = 0; i < n; ++i) {
            if (snap && c0 + i == snap->cell) {
                if (std::uniform_real_distribution<>(0, 1)(*rng) < defer_p) {
                    def            = st;
                    def_valid      = true;
                    snap->deferred = true;
                } else {
                    memcpy(snap->dst, &st, sizeof(st));
                }
            }
            step(st, x + (size_t) i * W, p4 + (size_t) i * 4);
            if (i >= batch0) {
                out[(size_t) (i - batch0)] = st.h;
            }
        }
        calls["pages"]++;
        calls["page rows"] += n;
        return true;
    }
    bool seq_run_decode(int64_t c, const uint16_t * x, const int32_t * p4, int batch_row) override {
        REQUIRE(!pending, "decode with a verification pending");
        REQUIRE(st.cell == c, "decode at cell %lld with the state at %lld", (long long) c, (long long) st.cell);
        REQUIRE(c > 0 || caps.decode_at_zero, "decode at cell 0");
        step(st, x ? x : bx + (size_t) batch_row * W, p4);
        out[(size_t) batch_row] = st.h;
        calls["decode"]++;
        return true;
    }
    void seq_rows(int b0, int n, uint16_t * dst) override { memcpy(dst, bx + (size_t) b0 * W, (size_t) n * W * sizeof(uint16_t)); }
    bool seq_snap_fetch(uint8_t * dst) override {
        if (!def_valid) {
            return false;
        }
        memcpy(dst, &def, sizeof(def));
        return true;
    }
    bool seq_run_verify(int64_t c, int n, const uint16_t * x, const int32_t * p4, int b0) override {
        REQUIRE(!pending, "verify with a verification pending");
        REQUIRE(n >= 1 && n <= caps.verify_rows, "verify of %d rows", n);
        REQUIRE(st.cell == c && c > 0, "verify at cell %lld with the state at %lld", (long long) c, (long long) st.cell);
        vst[0] = st;
        for (int r = 0; r < n; ++r) {
            vst[r + 1] = vst[r];
            step(vst[r + 1], x ? x + (size_t) r * W : bx + (size_t) (b0 + r) * W, p4 + (size_t) r * 4);
            out[(size_t) (b0 + r)] = vst[r + 1].h;
        }
        vn      = n;
        pending = true;
        calls["verify"]++;
        return true;
    }
    void seq_verify_keep(int k) override {
        REQUIRE(pending && k >= caps.verify_min_keep && k <= vn, "verify_keep(%d) of %d", k, vn);
        st      = vst[k];
        pending = false;
    }
    void seq_verify_peek(int k, uint8_t * dst) override {
        REQUIRE(pending && k >= 0 && k <= vn, "verify_peek(%d) of %d", k, vn);
        memcpy(dst, &vst[k], sizeof(image));
    }
};

struct history {
    std::vector<uint16_t> x;
    std::vector<int32_t>  p;
    std::vector<uint64_t> h{ 7 };  // the state hash before each cell
    int64_t               len() const { return (int64_t) h.size() - 1; }
    void                  cut(int64_t c) {
        x.resize((size_t) c * W);
        p.resize((size_t) c * 4);
        h.resize((size_t) c + 1);
    }
    void add(const uint16_t * xr, const int32_t * pr) {
        x.insert(x.end(), xr, xr + W);
        p.insert(p.end(), pr, pr + 4);
        h.push_back(mix(h.back(), row_hash(xr, pr)));
    }
};

struct checkpoint {
    std::vector<uint8_t>  blob;
    std::vector<uint16_t> x;  // the history it was taken at
    bool                  slim;
};

struct runner {
    std::mt19937 rng;
    fake         M;
    seq_config   cfg;
    seqstate *   S = nullptr;
    history      H;
    double       null_p = 0;
    std::map<std::string, long> stat;
    std::vector<checkpoint>     cks;
    int64_t                     last_prompt_end = 0;
    int64_t                     last_verify_c0 = -1;
    int                         last_verify_T  = 0;

    int  rnd(int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); }
    bool coin(double p) { return std::uniform_real_distribution<>(0, 1)(rng) < p; }

    // is a miss of a batch from c0 legitimate: no restart point of its page with the page's inputs up to c0
    bool covered(int64_t c0) const {
        const int64_t s0 = c0 - c0 % M.caps.page;
        const int     k  = (int) (c0 - s0);
        if (S->current().cell == s0 && S->current().rows >= k) {
            return true;
        }
        for (const auto & p : S->kept_points()) {
            if (p.cell == s0 && p.rows >= k) {
                return true;
            }
        }
        return false;
    }

    void verify_all(int64_t c0, int T, const char * what) {
        for (int i = 0; i < T; ++i) {
            REQUIRE(M.out[(size_t) i] == H.h[(size_t) (c0 + i + 1)], "%s: output row %d of a batch from %lld (%d rows) via '%s'", what, i,
                    (long long) c0, T, S->path());
        }
        const image & st = M.pending ? M.vst[M.vn] : M.st;
        REQUIRE(S->cell() == c0 + T, "%s: seqstate at %lld, not %lld", what, (long long) S->cell(), (long long) (c0 + T));
        REQUIRE(st.cell == c0 + T && st.h == H.h[(size_t) (c0 + T)], "%s: the model's state after '%s' (cell %lld, want %lld)", what,
                S->path(), (long long) st.cell, (long long) (c0 + T));
        for (int64_t c = 0; c < c0 + T; ++c) {
            REQUIRE(M.kv[(size_t) c] == row_hash(H.x.data() + (size_t) c * W, H.p.data() + (size_t) c * 4), "%s: KV cell %lld after '%s'",
                    what, (long long) c, S->path());
        }
    }

    // a batch of T random rows from c0 (the server's view: everything from c0 on is replaced)
    void feed(int64_t c0, int T, const char * what) {
        if (c0 + T > M.caps.cells) {
            c0 = 0;
            T  = std::min<int64_t>(T, M.caps.cells);
        }
        std::vector<uint16_t> x((size_t) T * W);
        std::vector<int32_t>  p((size_t) T * 4);
        for (auto & v : x) {
            v = (uint16_t) rnd(0, 65535);
        }
        for (int i = 0; i < T; ++i) {
            p[(size_t) i * 4] = p[(size_t) i * 4 + 1] = p[(size_t) i * 4 + 2] = (int32_t) (c0 + i + rnd(0, 3));
            p[(size_t) i * 4 + 3] = 0;
        }
        const bool    hit_expected = c0 == 0 || (S->cell() == c0 && (c0 % M.caps.page == 0 || T == 1)) || covered(c0);
        const int64_t cell_before  = S->cell();
        H.cut(c0);
        for (int i = 0; i < T; ++i) {
            H.add(x.data() + (size_t) i * W, p.data() + (size_t) i * 4);
        }
        M.out.assign((size_t) T, 0);
        M.bx              = x.data();
        const bool   null = coin(null_p);
        seq_status   st   = S->feed(c0, T, null ? nullptr : x.data(), p.data());
        stat[std::string(what) + " / " + (st == SEQ_OK ? S->path() : st == SEQ_MISS ? "MISS" : "ERROR")]++;
        if (st == SEQ_MISS) {
            REQUIRE(!hit_expected, "%s: a miss from cell %lld (state at %lld) although a restart point covers it", what, (long long) c0,
                    (long long) cell_before);
            // what the server does: the whole prompt again from cell 0
            M.out.assign((size_t) (c0 + T), 0);
            M.bx = H.x.data();
            st   = S->feed(0, (int) (c0 + T), null ? nullptr : H.x.data(), H.p.data());
            REQUIRE(st == SEQ_OK, "%s: the fallback from cell 0 failed: %s", what, S->why());
            M.out.erase(M.out.begin(), M.out.begin() + c0);
        }
        REQUIRE(st == SEQ_OK, "%s: batch from %lld (%d rows, state at %lld): %s", what, (long long) c0, T, (long long) cell_before, S->why());
        verify_all(c0, T, what);
        const bool ver = !strcmp(S->path(), "verify");
        last_verify_c0 = ver ? c0 : -1;
        last_verify_T  = ver ? T : 0;
        if (T > 1) {
            last_prompt_end = c0 + T;
        }
    }

    void op() {
        const int     PG  = M.caps.page;
        const int64_t len = H.len();
        if (last_verify_c0 >= 0 && coin(0.5)) {  // the sampler accepted only some of the verified rows
            feed(last_verify_c0 + rnd(0, last_verify_T), rnd(1, 4), "partial accept");
            return;
        }
        const int r = rnd(0, 99);
        if (len == 0 || r < 3) {
            feed(0, coin(0.2) ? 1 : rnd(1, 20 * PG), "new sequence");
        } else if (r < 48) {
            for (int i = rnd(1, 3 * PG); i > 0 && H.len() < M.caps.cells; --i) {
                feed(H.len(), 1, "decode");
            }
        } else if (r < 58) {
            feed(len, rnd(2, 4), "continue 2-4");
        } else if (r < 68) {
            feed(len, rnd(5, 2 * PG), "continue short");
        } else if (r < 76) {
            feed(len, rnd(2 * PG, 12 * PG), "continue long");
        } else if (r < 93) {
            int64_t c0;
            const char * what;
            switch (rnd(0, 4)) {
                case 0:  c0 = std::max<int64_t>(0, last_prompt_end - rnd(0, PG)); what = "rewind near prompt end"; break;
                case 1:  c0 = last_prompt_end; what = "rewind to prompt end"; break;
                case 2:  c0 = std::max<int64_t>(0, len - rnd(0, PG)); what = "rewind near end"; break;
                case 3:  c0 = (len / PG) * PG - (coin(0.5) ? PG : 0); c0 = std::max<int64_t>(c0, 0); what = "rewind to page start"; break;
                default: c0 = rnd(0, (int) len); what = "rewind anywhere"; break;
            }
            c0 = std::min(c0, len);
            feed(c0, coin(0.3) ? 1 : rnd(1, 3 * PG), what);
        } else if (r < 97) {
            checkpoint ck;
            ck.slim = coin(0.7);
            ck.blob.resize(S->size(ck.slim));
            const size_t n = S->write(ck.blob.data(), ck.blob.size(), ck.slim);
            REQUIRE(n == ck.blob.size(), "checkpoint write %zu of %zu", n, ck.blob.size());
            if (n) {
                ck.x = H.x;
                cks.push_back(std::move(ck));
                if (cks.size() > 6) {
                    cks.erase(cks.begin());
                }
                stat["checkpoint write"]++;
            }
        } else {
            // the server restores a checkpoint whose history is a prefix of the current one
            for (size_t i = cks.size(); i-- > 0;) {
                const checkpoint & ck = cks[i];
                if (ck.x.size() <= H.x.size() && !memcmp(ck.x.data(), H.x.data(), ck.x.size() * sizeof(uint16_t))) {
                    const int64_t at = (int64_t) (ck.x.size() / W);
                    REQUIRE(S->read(ck.blob.data(), ck.blob.size()) == ck.blob.size(), "checkpoint read");
                    H.cut(at);
                    if (S->cell() >= 0) {
                        REQUIRE(S->cell() == at && M.st.cell == at && M.st.h == H.h[(size_t) at], "the state after a checkpoint read");
                    }
                    last_verify_c0  = -1;
                    last_prompt_end = std::min(last_prompt_end, at);
                    stat[ck.slim ? "checkpoint read (slim)" : "checkpoint read (full)"]++;
                    feed(std::max<int64_t>(0, at - (coin(0.5) ? 0 : rnd(0, PG))), rnd(1, 2 * PG), "after checkpoint");
                    break;
                }
            }
        }
    }
};

void fuzz(unsigned seed, int ops, std::map<std::string, long> & total) {
    runner R;
    R.rng.seed(seed);
    R.M.caps.page            = R.coin(0.5) ? 16 : 8;
    R.M.caps.width           = W;
    R.M.caps.image_bytes     = sizeof(image);
    R.M.caps.cells           = 1024;
    R.M.caps.verify_rows     = R.coin(0.7) ? 4 : 0;
    R.M.caps.verify_min_keep = R.rnd(0, 1);
    R.M.caps.decode_at_zero  = R.coin(0.5);
    R.M.kv.assign((size_t) R.M.caps.cells, 0);
    R.M.rng        = &R.rng;
    R.M.defer_p    = R.coin(0.5) ? 0.7 : 0;
    R.null_p       = R.coin(0.5) ? 0.5 : 0;
    R.cfg.kept_max = R.rnd(0, 3);
    R.cfg.verify_chain = R.coin(0.6) ? 12 : 0;
    R.cfg.check    = true;
    seqstate S(R.M, R.cfg);
    R.S = &S;
    for (int i = 0; i < ops; ++i) {
        R.op();
    }
    for (const auto & [k, v] : R.stat) {
        total[k] += v;
    }
    for (const auto & [k, v] : R.M.calls) {
        total["model: " + k] += v;
    }
}

// the cases behind past field failures, spelled out
void scenarios() {
    for (int defer = 0; defer < 2; ++defer) {
        runner R;
        R.rng.seed(1);
        R.M.caps = { 16, W, sizeof(image), 4096, 4, 0, true };
        R.M.kv.assign(4096, 0);
        R.M.rng     = &R.rng;
        R.M.defer_p = defer;
        R.cfg       = { 1, 12, true };
        seqstate S(R.M, R.cfg);
        R.S = &S;
        // a prompt, an answer that crosses pages, then the next turn continues: every page start of the answer must
        // have left a restart point (the 2026-11 "page-start state is not kept" failure)
        R.feed(0, 40, "prompt");
        for (int i = 0; i < 50; ++i) {
            R.feed(R.H.len(), 1, "answer");
        }
        R.feed(R.H.len(), 30, "next turn");
        REQUIRE(!strcmp(S.path(), "replay"), "the next turn ran as '%s'", S.path());
        // regenerate: back into the page where that prompt ended, after an answer that crossed pages
        const int64_t pe = R.H.len();
        for (int i = 0; i < 50; ++i) {
            R.feed(R.H.len(), 1, "answer");
        }
        R.feed(pe - 3, 5, "edit the prompt's end");
        REQUIRE(!strcmp(S.path(), "rewind"), "the edit ran as '%s'", S.path());
        // a short continuation runs on the verify kernel, not the pages
        R.feed(R.H.len(), 9, "short turn");
        REQUIRE(!strcmp(S.path(), "verify chain"), "the short turn ran as '%s'", S.path());
        // a prompt that ends exactly on a page start, then a regenerate from there
        R.feed(0, 48, "aligned prompt");
        for (int i = 0; i < 20; ++i) {
            R.feed(R.H.len(), 1, "answer");
        }
        R.feed(48, 1, "regenerate at the page start");
        REQUIRE(!strcmp(S.path(), "rewind"), "the regenerate ran as '%s'", S.path());
        // a slim checkpoint taken mid-page, the sequence moves on, the checkpoint comes back
        R.feed(0, 100, "prompt");
        std::vector<uint8_t> blob(S.size(true));
        REQUIRE(S.write(blob.data(), blob.size(), true) == blob.size(), "write");
        const std::vector<uint16_t> hx = R.H.x;
        for (int i = 0; i < 40; ++i) {
            R.feed(R.H.len(), 1, "answer");
        }
        REQUIRE(S.read(blob.data(), blob.size()) == blob.size(), "read");
        R.H.cut(100);
        R.feed(100, 20, "after the checkpoint");
        // a miss is reported as a miss
        R.feed(0, 100, "prompt");
        for (int i = 0; i < 60; ++i) {
            R.feed(R.H.len(), 1, "answer");
        }
        std::vector<uint16_t> x(W, 1);
        std::vector<int32_t>  p(4, 0);
        R.M.out.assign(1, 0);
        REQUIRE(S.feed(20, 1, x.data(), p.data()) == SEQ_MISS, "a rewind to a page nobody kept must miss");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 200;
    const int ops   = argc > 2 ? atoi(argv[2]) : 400;
    scenarios();
    std::map<std::string, long> total;
    for (int s = 0; s < seeds; ++s) {
        fuzz(1000 + s, ops, total);
    }
    for (const auto & [k, v] : total) {
        printf("%-44s %ld\n", k.c_str(), v);
    }
    printf("OK: %d seeds x %d ops\n", seeds, ops);
    return 0;
}
