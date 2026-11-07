#include "argmax_state.h"
#include "../core/runtime.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <mutex>
#include <vector>

namespace axera {
namespace {

struct state {
    struct publication {
        const char *         base = nullptr;
        size_t               stride = 0;
        int32_t              n = 0;
        std::vector<int32_t> ids;
        std::vector<float>   logits;
        std::vector<uint8_t> used;
        uint64_t             epoch = 0;
    };
    mutable std::mutex   mu;
    bool                 enabled = false;
    int32_t              R = 0;
    std::vector<publication> pubs;
    std::vector<int32_t> suppress;
};

state & s() {
    static state v;
    return v;
}

}  // namespace

int argmax_configure(bool enabled, const int32_t * ids, int32_t n) {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    v.enabled = false;
    v.R = 0;
    v.pubs.clear();
    v.suppress.clear();
    if (ids && n > 0) {
        v.suppress.assign(ids, ids + n);
        std::sort(v.suppress.begin(), v.suppress.end());
        v.suppress.erase(std::unique(v.suppress.begin(), v.suppress.end()), v.suppress.end());
    }
    if (!enabled) {
        return 0;
    }
    // A sampler is configured before the adapter loads the manifest. Request the fast path now; the combined head
    // publishes rows once it runs, and a manifest without one simply never publishes (the caller then falls back to
    // the full logits). Four is the largest head the model code binds, including MTP verify.
    const int r = 4;
    v.enabled = true;
    v.R = r;
    return r;
}

void argmax_reset() {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    v.enabled = false;
    v.R = 0;
    v.pubs.clear();
    v.suppress.clear();
}

bool argmax_wanted() {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    return v.enabled;
}

bool argmax_publish(const int32_t * ids, const float * logits, int32_t rows, const void * rows_base, size_t row_stride) {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    if (!v.enabled || !ids || !logits || !rows_base || rows <= 0 || rows > v.R || row_stride < sizeof(float)) {
        return false;
    }
    auto same_rows = [&](const state::publication & p) {
        return p.base == (const char *) rows_base && p.stride == row_stride;
    };
    auto drop_rows = [&]() {
        v.pubs.erase(std::remove_if(v.pubs.begin(), v.pubs.end(), same_rows), v.pubs.end());
    };
    for (int32_t r = 0; r < rows; ++r) {
        if (std::binary_search(v.suppress.begin(), v.suppress.end(), ids[r])) {
            drop_rows();
            return false;
        }
    }
    auto it = std::find_if(v.pubs.begin(), v.pubs.end(), same_rows);
    if (it == v.pubs.end()) {
        if (v.pubs.size() >= 8) {
            v.pubs.erase(v.pubs.begin());
        }
        v.pubs.emplace_back();
        it = std::prev(v.pubs.end());
    }
    it->base   = (const char *) rows_base;
    it->stride = row_stride;
    it->n      = rows;
    it->ids.assign(ids, ids + rows);
    it->logits.assign(logits, logits + rows);
    it->used.assign((size_t) rows, 0);
    it->epoch = axera_graph_epoch;
    return true;
}

int argmax_published() {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    if (!v.enabled) {
        return 0;
    }
    for (auto it = v.pubs.rbegin(); it != v.pubs.rend(); ++it) {
        if (it->n > 0 && it->epoch == axera_graph_epoch) {
            return it->n;
        }
    }
    return 0;
}

int argmax_take(int idx, const void * row_logits, int32_t * id, float * logit) {
    (void) idx;
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    if (!v.enabled || !row_logits || !id || !logit) {
        return 0;
    }
    // The buffer is the graph's identity. In particular the MTP draft/catch-up graph runs after a target verify and
    // before host sampling, so the graph epoch is already newer there; matching the host row keeps the target's row.
    const float * row_values = (const float *) row_logits;
    for (auto it = v.pubs.rbegin(); it != v.pubs.rend(); ++it) {
        for (int32_t row = 0; row < it->n; ++row) {
            if ((const char *) row_logits != it->base + (size_t) row * it->stride || it->used[(size_t) row]) {
                continue;
            }
            // The same host buffer can be reused by a later graph while an older publication still names it.  Only
            // take the fused id when the current row still carries the value the kernel published with it; otherwise
            // fall back to the full logits (which are from the current graph).
            const float published = it->logits[(size_t) row];
            const float current   = row_values[it->ids[(size_t) row]];
            if (fabsf(current - published) > 1e-5f * std::max(1.0f, fabsf(published))) {
                continue;
            }
            it->used[(size_t) row] = 1;
            *id                   = it->ids[(size_t) row];
            *logit                = it->logits[(size_t) row];
            return 1;
        }
    }
    return 0;
}

}  // namespace axera
