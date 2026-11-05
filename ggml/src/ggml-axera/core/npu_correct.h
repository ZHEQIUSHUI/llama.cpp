// The host-side correction of a verification row's fused candidates.
//
// lm_head.topk applies the penalty arrays as they are at the verified prefix. Row r of a speculative verification
// starts after r accepted tokens, so those <= 3 tokens' penalty changes must be applied on the host. Everything here is
// pure: the backend and the offline chain checker run the same code.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace axera {

struct npu_penalty_cfg {
    int32_t last_n  = 0;
    float   repeat  = 1.0f;
    float   freq    = 0.0f;
    float   present = 0.0f;
};

inline bool npu_penalty_active(const npu_penalty_cfg & p) {
    return p.last_n > 0 && !(p.repeat == 1.0f && p.freq == 0.0f && p.present == 0.0f);
}

inline void npu_penalty_params(const npu_penalty_cfg & p, int count, float * mul_pos, float * mul_neg, float * bias) {
    if (!npu_penalty_active(p) || count <= 0) {
        *mul_pos = *mul_neg = 1.0f;
        *bias                = 0.0f;
        return;
    }
    *mul_pos = 1.0f / p.repeat;
    *mul_neg = p.repeat;
    *bias    = -(float(count) * p.freq + p.present);
}

inline void npu_accept_ring(const npu_penalty_cfg & p, int32_t token, std::unordered_map<int32_t, int> & counts,
                            std::deque<int32_t> & ring) {
    ++counts[token];
    if ((int) ring.size() >= p.last_n) {
        const int32_t old = ring.front();
        ring.pop_front();
        auto it = counts.find(old);
        if (it != counts.end() && --it->second <= 0) {
            counts.erase(it);
        }
    }
    ring.push_back(token);
}

// Recover the raw logit from the kernel's penalised value. The sign of raw decides the repeat factor; when the value
// falls in the interval where both branches are plausible, the row needs its full logits instead.
inline bool npu_raw_logit(float value, float mul_pos, float mul_neg, float bias, int count, float * raw) {
    if (count <= 0) {
        *raw = value;
        return true;
    }
    const float rp = (value - bias) / mul_pos;
    const float rn = (value - bias) / mul_neg;
    const bool  vp = rp > 0.0f;
    const bool  vn = rn <= 0.0f;
    if (vp && !vn) {
        *raw = rp;
        return true;
    }
    if (vn && !vp) {
        *raw = rn;
        return true;
    }
    return false;
}

inline float npu_corrected_absent_bound(const npu_penalty_cfg & p, float base_max, int old_count, int new_count) {
    float p0, n0, b0, p1, n1, b1;
    npu_penalty_params(p, old_count, &p0, &n0, &b0);
    npu_penalty_params(p, new_count, &p1, &n1, &b1);
    const float pos = ((base_max - b0) / p0) * p1 + b1;
    const float rn  = std::min(0.0f, (base_max - b0) / n0);
    const float neg = rn * n1 + b1;
    return std::max(pos, neg);
}

struct npu_corrected_candidate {
    float   value;
    int32_t id;
};

// Apply `extra` (the tokens accepted before this row) to the candidates the kernel computed with `base_counts` /
// `base_ring`. Returns false when the K candidates cannot prove the corrected row exactly.
inline bool npu_correct_candidates(const float * values, const int32_t * ids, int n,
                                   const std::unordered_map<int32_t, int> & base_counts,
                                   const std::deque<int32_t> & base_ring,
                                   const std::vector<int32_t> & extra, int32_t top_k,
                                   const npu_penalty_cfg & cfg, std::vector<npu_corrected_candidate> & out) {
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        out[i] = { values[i], ids[i] };
    }
    if (!npu_penalty_active(cfg) || extra.empty()) {
        return true;
    }

    // The extras are at most three accepts: their effect is a small delta over the base counts plus the evictions from
    // the front of the ring. Computing the delta avoids copying the (up to last_n-entry) count map per row.
    std::vector<std::pair<int32_t, int>> delta;
    auto add = [&](int32_t tok, int d) {
        for (auto & e : delta) {
            if (e.first == tok) {
                e.second += d;
                return;
            }
        }
        delta.emplace_back(tok, d);
    };
    std::deque<int32_t> ring = base_ring;
    for (int32_t token : extra) {
        add(token, 1);
        if ((int) ring.size() >= cfg.last_n) {
            add(ring.front(), -1);
            ring.pop_front();
        }
        ring.push_back(token);
    }
    std::vector<std::pair<int32_t, int>> changed;  // token -> new count
    for (const auto & e : delta) {
        if (e.second == 0) {
            continue;
        }
        const auto it  = base_counts.find(e.first);
        const int  old = it == base_counts.end() ? 0 : it->second;
        const int  now = old + e.second;
        if (now != old) {
            changed.emplace_back(e.first, now);
        }
    }
    if (changed.empty()) {
        return true;
    }
    // Without a top-k above it, the chain's distribution (and its top-p/min-p boundary) would need the full
    // vocabulary; the candidates alone cannot update vmax / sumexp exactly after a correction.
    if (top_k == 0) {
        return false;
    }

    // The candidate list is sorted by the kernel's value. Find the changed candidates with a linear scan (1024 entries,
    // at most a handful changed) and update their values in place.
    std::vector<int> cpos;
    cpos.reserve(changed.size());
    for (const auto & kv : changed) {
        int p = -1;
        for (int i = 0; i < n; ++i) {
            if (ids[i] == kv.first) {
                p = i;
                break;
            }
        }
        if (p < 0) {
            continue;
        }
        const auto it  = base_counts.find(kv.first);
        const int  old = it == base_counts.end() ? 0 : it->second;
        float p0, n0, b0, p1, n1, b1, raw;
        npu_penalty_params(cfg, old, &p0, &n0, &b0);
        npu_penalty_params(cfg, kv.second, &p1, &n1, &b1);
        if (!npu_raw_logit(values[p], p0, n0, b0, old, &raw)) {
            return false;
        }
        out[p].value = raw * (raw > 0.0f ? p1 : n1) + b1;
        cpos.push_back(p);
    }
    // Stable (original position) merge of the moved and the unchanged entries: this keeps the kernel's order wherever
    // the corrected values tie, exactly as std::stable_sort over the whole list would.
    if (!cpos.empty()) {
        std::sort(cpos.begin(), cpos.end());
        struct item {
            npu_corrected_candidate c;
            int                     orig;
        };
        std::vector<item> rest, moved;
        rest.reserve(n - cpos.size());
        moved.reserve(cpos.size());
        size_t k = 0;
        for (int i = 0; i < n; ++i) {
            if (k < cpos.size() && cpos[k] == i) {
                moved.push_back({ out[i], i });
                ++k;
            } else {
                rest.push_back({ out[i], i });
            }
        }
        std::stable_sort(moved.begin(), moved.end(), [](const item & a, const item & b) {
            return a.c.value > b.c.value || (a.c.value == b.c.value && a.orig < b.orig);
        });
        size_t r = 0, m = 0, w = 0;
        while (r < rest.size() || m < moved.size()) {
            bool take_rest = m >= moved.size();
            if (!take_rest && r < rest.size()) {
                take_rest = rest[r].c.value > moved[m].c.value ||
                            (rest[r].c.value == moved[m].c.value && rest[r].orig < moved[m].orig);
            }
            out[w++] = take_rest ? rest[r++].c : moved[m++].c;
        }
    }

    // top-k keeps M rows. No token outside the kernel's K can enter those M unless the corrected M-th value falls to
    // the unchanged K-th threshold, and an affected absent token is bounded by its most favorable penalty branch.
    const int M = std::min<int>(top_k, n);
    if (M <= 0) {
        return false;
    }
    const float threshold = out[M - 1].value;
    const float base_kth  = values[n - 1];
    if (threshold <= base_kth) {
        return false;
    }
    for (const auto & kv : changed) {
        bool present = false;
        for (int i = 0; i < n; ++i) {
            if (ids[i] == kv.first) {
                present = true;
                break;
            }
        }
        if (present) {
            continue;
        }
        const auto it  = base_counts.find(kv.first);
        const int  old = it == base_counts.end() ? 0 : it->second;
        if (npu_corrected_absent_bound(cfg, base_kth, old, kv.second) >= threshold) {
            return false;
        }
    }
    return true;
}

}  // namespace axera
