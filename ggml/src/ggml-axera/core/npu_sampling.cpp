// Axera NPU backend: the host side of the fused lm_head.topk sampling kernel (see npu_sampling.h)

#include "npu_sampling.h"

#include "manifest.h"
#include "npu_correct.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace axera {

npu_sampling & npu_state() {
    static npu_sampling s;
    return s;
}

static npu_penalty_cfg penalty_cfg(const npu_sampling & s) {
    return { s.penalty_last_n, s.repeat, s.freq, s.present };
}

static bool penalty_active(const npu_sampling & s) {
    return npu_penalty_active(penalty_cfg(s));
}

static void penalty_params(const npu_sampling & s, int count, float * mul_pos, float * mul_neg, float * bias) {
    npu_penalty_params(penalty_cfg(s), count, mul_pos, mul_neg, bias);
}

static int kernel_rows_and_k(const kernel & k, int32_t * K, int32_t * R) {
    auto has_out = [&](const char * name) {
        for (const auto & d : k.desc->out) {
            if (d.name == name) {
                return true;
            }
        }
        return false;
    };
    if (k.find_input("x") < 0 || k.find_input("w") < 0 || k.find_input("s") < 0 ||
        !has_out("values") || !has_out("ids") || !has_out("vmax") || !has_out("sumexp")) {
        return 0;
    }
    int rows = 1;
    for (const auto & d : k.desc->in) {
        if (d.name == "x" && d.shape.size() == 2) {
            rows = (int) d.shape[0];
        }
    }
    const uint32_t vb = k.out_bytes("values");
    const uint32_t ib = k.out_bytes("ids");
    if (rows <= 0 || rows > 4 || vb % (sizeof(float) * rows) || ib % (sizeof(int32_t) * rows) ||
        vb / (sizeof(float) * rows) != ib / (sizeof(int32_t) * rows) ||
        k.out_bytes("vmax") != (uint32_t) rows * sizeof(float) ||
        k.out_bytes("sumexp") != (uint32_t) rows * sizeof(float)) {
        return 0;
    }
    *R = rows;
    *K = (int32_t) (vb / (sizeof(float) * rows));
    return *K > 0;
}

int npu_configure(bool enabled, int32_t top_k, float top_p, float min_p, float temp_sumexp, float temp_minp,
                  int32_t penalty_last_n, float repeat, float freq, float present) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.penalty_last_n = penalty_last_n;
    s.repeat         = repeat;
    s.freq           = freq;
    s.present        = present;
    s.top_k          = top_k;
    s.top_p          = top_p;
    s.min_p          = min_p;
    s.temp_sumexp    = temp_sumexp;
    s.temp_minp      = temp_minp;
    s.counts.clear();
    s.ring.clear();
    s.updates.clear();
    s.rows.clear();
    s.K = s.R = 0;
    s.valid = false;
    s.base_counts.clear();
    s.base_ring.clear();
    s.base_valid = false;
    s.row_extra.clear();
    s.full_row_cb = nullptr;
    s.full_row_ud = nullptr;
    ++s.generation;
    ++s.config_id;

    if (enabled) {
        const manifest * m = kernels();
        if (m && m->has("lm_head.topk")) {
            const kernel & k = m->get("lm_head.topk");
            int32_t K = 0, R = 0;
            if (k.find_input("bias") >= 0 && k.find_input("mul_pos") >= 0 && k.find_input("mul_neg") >= 0 &&
                k.find_input("temp") >= 0 && kernel_rows_and_k(k, &K, &R) &&
                (R == 1 || k.find_input("m") >= 0)) {
                s.K = K;
                s.R = R;
                s.rows.assign(R, npu_candidates{});
                for (auto & row : s.rows) {
                    row.values.assign(K, 0.0f);
                    row.ids.assign(K, 0);
                }
            }
        }
    }
    s.enabled = s.K > 0 && s.R > 0;
    return s.enabled ? s.K : 0;
}

int64_t npu_config_id() {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    return s.config_id;
}

npu_opts npu_get_options() {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    return { s.enabled, s.K, s.top_k, s.top_p, s.min_p, s.temp_sumexp, s.temp_minp };
}

static void clear_rows(npu_sampling & s) {
    for (auto & row : s.rows) {
        row.valid = false;
        row.full  = false;
    }
    s.valid = false;
}

void npu_reset() {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.counts.clear();
    s.ring.clear();
    s.updates.clear();
    clear_rows(s);
    s.base_counts.clear();
    s.base_ring.clear();
    s.base_valid = false;
    s.row_extra.clear();
    ++s.generation;  // the adapter clears the arrays to their neutral state
}

static void row_of(npu_sampling & s, int32_t token, std::vector<npu_penalty_row> & out) {
    const auto it = s.counts.find(token);
    npu_penalty_row r;
    r.token = token;
    if (it == s.counts.end() || it->second <= 0) {
        r.mul_pos = r.mul_neg = 1.0f;
        r.bias    = 0.0f;
    } else {
        const int c = it->second;
        r.mul_pos = 1.0f / s.repeat;
        r.mul_neg = s.repeat;
        r.bias    = -(float(c) * s.freq + s.present);
    }
    out.push_back(r);
}

void npu_accept(int32_t token) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.enabled) {
        return;
    }
    // The row index is the number of tokens accepted since the kernel published its candidates, whether or not the
    // penalty sampler changes anything. The correction below then sees those tokens in the same order.
    s.row_extra.push_back(token);
    if (!penalty_active(s)) {
        return;
    }
    ++s.counts[token];
    if ((int) s.ring.size() >= s.penalty_last_n) {
        const int32_t old = s.ring.front();
        s.ring.pop_front();
        auto it = s.counts.find(old);
        if (it != s.counts.end() && --it->second <= 0) {
            s.counts.erase(it);
        }
        row_of(s, old, s.updates);
    }
    s.ring.push_back(token);
    row_of(s, token, s.updates);
}

// The kernel applied the prefix's penalties to every row. Simulate the tokens accepted since it published and correct
// the candidates. Returns false when the K candidates cannot prove the corrected row exactly.
static bool correct_row(const npu_sampling & s, int row, std::vector<npu_corrected_candidate> & out) {
    if (!s.base_valid) {
        return false;
    }
    const npu_candidates & c = s.rows[row];
    return npu_correct_candidates(c.values.data(), c.ids.data(), c.n, s.base_counts, s.base_ring,
                                  s.row_extra, s.top_k, penalty_cfg(s), out);
}

int npu_take(float * values, int32_t * ids, int32_t max_k, int32_t * n, float * vmax, float * sumexp) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.valid) {
        return 0;
    }
    const int row = (int) s.row_extra.size();
    if (row < 0 || row >= s.R || !s.rows[row].valid) {
        return 0;
    }
    if (s.rows[row].full || !correct_row(s, row, s.scratch)) {
        s.rows[row].valid = false;
        s.rows[row].full  = false;
        s.valid = false;
        for (const auto & r : s.rows) {
            s.valid = s.valid || r.valid;
        }
        return -1;
    }
    const auto & corrected = s.scratch;
    const int32_t take = (int32_t) std::min<size_t>(corrected.size(), (size_t) max_k);
    for (int32_t i = 0; i < take; ++i) {
        values[i] = corrected[i].value;
        ids[i]    = corrected[i].id;
    }
    *n      = take;
    *vmax   = s.rows[row].vmax;
    *sumexp = s.rows[row].sumexp;
    s.rows[row].valid = false;
    s.valid = false;
    for (const auto & r : s.rows) {
        s.valid = s.valid || r.valid;
    }
    return 1;
}

int npu_full_row(int32_t idx, float * out, int32_t n) {
    npu_sampling & s = npu_state();
    int (*fn)(void *, int32_t, float *, int32_t) = nullptr;
    void * ud = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        fn  = s.full_row_cb;
        ud  = s.full_row_ud;
        if (!fn || (int) s.row_extra.size() >= s.R) {
            return 0;
        }
    }
    return fn(ud, idx, out, n) ? 1 : 0;
}

void npu_set_full_row(int (*fn)(void *, int32_t, float *, int32_t), void * ud) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.full_row_cb = fn;
    s.full_row_ud = ud;
}

struct npu_saved_state {
    std::unordered_map<int32_t, int> counts;
    std::deque<int32_t>              ring;
};

void * npu_save() {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.enabled) {
        return nullptr;
    }
    auto * p = new npu_saved_state;
    p->counts = s.counts;
    p->ring   = s.ring;
    return p;
}

void npu_restore(void * state) {
    auto * p = (npu_saved_state *) state;
    if (!p) {
        return;
    }
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.enabled) {
        return;
    }
    s.counts = p->counts;
    s.ring   = p->ring;
    s.updates.clear();
    clear_rows(s);
    s.base_counts.clear();
    s.base_ring.clear();
    s.base_valid = false;
    s.row_extra.clear();
    // The adapter sees the generation change and rewrites all arrays from neutral, then the rows below.
    ++s.generation;
    for (const auto & kv : s.counts) {
        row_of(s, kv.first, s.updates);
    }
}

void npu_free_state(void * state) {
    delete (npu_saved_state *) state;
}

void npu_take_updates(std::vector<npu_penalty_row> & rows, int64_t * generation) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    rows.swap(s.updates);
    s.updates.clear();
    *generation = s.generation;
}

void npu_publish(const float * values, const int32_t * ids, int32_t n, int32_t rows,
                 const float * vmax, const float * sumexp, const uint8_t * usable) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.enabled || n <= 0 || n > s.K || rows <= 0 || rows > s.R) {
        return;
    }
    for (int32_t r = 0; r < s.R; ++r) {
        s.rows[r].valid = false;
        s.rows[r].full  = false;
    }
    for (int32_t r = 0; r < rows; ++r) {
        const float *   v = values + (size_t) r * s.K;
        const int32_t * i = ids    + (size_t) r * s.K;
        memcpy(s.rows[r].values.data(), v, (size_t) n * sizeof(float));
        memcpy(s.rows[r].ids.data(),    i, (size_t) n * sizeof(int32_t));
        s.rows[r].n      = n;
        s.rows[r].vmax   = vmax[r];
        s.rows[r].sumexp = sumexp[r];
        s.rows[r].valid  = true;
        s.rows[r].full   = !usable || !usable[r];
    }
    s.base_counts = s.counts;
    s.base_ring   = s.ring;
    s.base_valid  = true;
    s.row_extra.clear();
    s.valid = true;
}

}  // namespace axera
