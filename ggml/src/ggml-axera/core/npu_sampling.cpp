// Axera NPU backend: the host side of the fused lm_head.topk sampling kernel (see npu_sampling.h)

#include "npu_sampling.h"

#include "manifest.h"

namespace axera {

npu_sampling & npu_state() {
    static npu_sampling s;
    return s;
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
    s.values.clear();
    s.ids.clear();
    s.n     = 0;
    s.valid = false;
    ++s.generation;
    ++s.config_id;

    s.K = 0;
    if (enabled) {
        const manifest * m = kernels();
        if (m && m->has("lm_head.topk")) {
            const kernel & k = m->get("lm_head.topk");
            if (k.find_input("bias") >= 0 && k.find_input("mul_pos") >= 0 && k.find_input("mul_neg") >= 0 &&
                k.find_input("temp") >= 0 && k.find_input("x") >= 0 && k.find_input("w") >= 0 && k.find_input("s") >= 0) {
                s.K = (int32_t) (k.out_bytes("values") / sizeof(float));
                s.values.resize(s.K);
                s.ids.resize(s.K);
            }
        }
    }
    s.enabled = s.K > 0;
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

void npu_reset() {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.counts.clear();
    s.ring.clear();
    s.updates.clear();
    s.valid = false;
    s.n     = 0;
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
    if (!s.enabled || s.penalty_last_n <= 0) {
        return;
    }
    // llama.cpp considers the penalty sampler disabled with these parameters: keep the arrays neutral
    if (s.repeat == 1.0f && s.freq == 0.0f && s.present == 0.0f) {
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

int npu_take(float * values, int32_t * ids, int32_t max_k, int32_t * n, float * vmax, float * sumexp) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.valid || s.n <= 0) {
        return 0;
    }
    const int32_t take = s.n < max_k ? s.n : max_k;
    for (int32_t i = 0; i < take; ++i) {
        values[i] = s.values[i];
        ids[i]    = s.ids[i];
    }
    *n      = take;
    *vmax   = s.vmax;
    *sumexp = s.sumexp;
    s.valid = false;
    return 1;
}

void npu_take_updates(std::vector<npu_penalty_row> & rows, int64_t * generation) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    rows.swap(s.updates);
    s.updates.clear();
    *generation = s.generation;
}

void npu_publish(const float * values, const int32_t * ids, int32_t n, float vmax, float sumexp) {
    npu_sampling & s = npu_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.enabled || n <= 0 || n > s.K) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        s.values[i] = values[i];
        s.ids[i]    = ids[i];
    }
    s.n      = n;
    s.vmax   = vmax;
    s.sumexp = sumexp;
    s.valid  = true;
}

}  // namespace axera
