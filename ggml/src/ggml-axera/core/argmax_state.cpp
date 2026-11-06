#include "argmax_state.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace axera {
namespace {

struct state {
    mutable std::mutex   mu;
    bool                 enabled = false;
    int32_t              R = 0;
    std::vector<int32_t> ids;
    std::vector<float>   logits;
    int32_t              n = 0;
    std::vector<uint8_t> used;
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
    v.ids.clear();
    v.logits.clear();
    v.n = 0;
    v.used.clear();
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
    v.ids.assign(r, 0);
    v.logits.assign(r, 0.0f);
    return r;
}

void argmax_reset() {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    v.enabled = false;
    v.R = 0;
    v.ids.clear();
    v.logits.clear();
    v.n = 0;
    v.used.clear();
    v.suppress.clear();
}

bool argmax_wanted() {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    return v.enabled;
}

bool argmax_publish(const int32_t * ids, const float * logits, int32_t rows) {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    if (!v.enabled || !ids || !logits || rows <= 0 || rows > v.R) {
        v.n = 0;
        v.used.clear();
        return false;
    }
    for (int32_t r = 0; r < rows; ++r) {
        if (std::binary_search(v.suppress.begin(), v.suppress.end(), ids[r])) {
            v.n = 0;
            return false;
        }
    }
    std::copy(ids, ids + rows, v.ids.begin());
    std::copy(logits, logits + rows, v.logits.begin());
    v.n = rows;
    v.used.assign((size_t) rows, 0);
    return true;
}

int argmax_take(int idx, int32_t * id, float * logit) {
    state & v = s();
    std::lock_guard<std::mutex> lk(v.mu);
    if (!v.enabled || !id || !logit || v.n <= 0) {
        return 0;
    }
    const int32_t row = idx < 0 ? v.n + idx : idx;
    if (row < 0 || row >= v.n || v.used[(size_t) row]) {
        return 0;
    }
    v.used[(size_t) row] = 1;
    *id = v.ids[(size_t) row];
    *logit = v.logits[(size_t) row];
    return 1;
}

}  // namespace axera
