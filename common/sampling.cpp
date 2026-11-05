#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-axera.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <map>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // greedy decoding with no sampler that can change the argmax (see common_sampler_init): the token of the largest
    // logit straight from the logits, without the candidate array over the vocabulary and the chain
    bool greedy_fast = false;
    // pure greedy chain and the combined lm_head wrote an NPU argmax: take its (id, logit), not the full row
    bool npu_argmax = false;
    // this sampler configured the backend's global argmax state; a clone only borrows it
    bool npu_argmax_own = false;
    // (greedy_fast) the logit-bias sampler's biases, added in the argmax: one per token (the sum of its biases), in
    // token order, the tokens in the vocabulary
    std::vector<llama_logit_bias> fast_bias;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    // the fused lm_head.topk path: the backend's K candidates (penalties already applied on the NPU) and the chain
    // over them; top-p uses the kernel's full-vocabulary sumexp when no top-k has narrowed the set
    const ggml_axera_npu_sampling * npu = nullptr;
    bool                 npu_sampling = false;
    int32_t              npu_K = 0;
    float                npu_temp = 1.0f;  // the accumulated temperature at the chain's top-p
    bool                 npu_exact_set = false;  // the rejection variant reads the whole target set: resolve the band
    int64_t              npu_cfg = 0;  // the backend configuration this sampler owns
    std::vector<float>   npu_v;
    std::vector<int32_t> npu_i;
    std::vector<float>   npu_full;  // a row's full logits, from the adapter's regular head (the fallback)
    void *               npu_saved = nullptr;  // a clone's penalty checkpoint (only the clone owns it)

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// the backend's fused lm_head.topk: is this chain one its K candidates serve exactly? (the sampler names carry a
// +/- prefix once the backend sampler support has been probed, so ignore it)
static const char * npu_strip(const char * n, bool * empty = nullptr) {
    if (empty) {
        *empty = false;
    }
    if (!n) {
        return nullptr;
    }
    if (n[0] == '?') {  // llama.cpp's no-op sampler for a disabled option
        if (empty) {
            *empty = true;
        }
        return n + 1;
    }
    return n[0] == '+' || n[0] == '-' ? n + 1 : n;
}

static const ggml_axera_npu_sampling * npu_api() {
    static const ggml_axera_npu_sampling * api = nullptr;
    if (!api) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_AXERA_NAME);
        if (reg) {
            api = (const ggml_axera_npu_sampling *) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_axera_npu_sampling");
        }
    }
    return api;
}

struct npu_chain_plan {
    bool  ok = false;
    float temp_sumexp = 1.0f;  // the accumulated temperature at the top-p sampler
    float temp_minp   = 1.0f;  // ... and at the min-p sampler
};

static npu_chain_plan npu_plan_chain(llama_sampler * chain, const common_params_sampling & params) {
    npu_chain_plan p;
    if (params.mirostat != 0 || params.n_probs != 0 || params.backend_sampling || params.dynatemp_range != 0.0f ||
        !params.logit_bias.empty() ||
        !std::isfinite(params.temp) || params.temp < 0.0f || params.penalty_last_n < 0 ||
        !std::isfinite(params.top_p) || !std::isfinite(params.min_p) || !std::isfinite(params.penalty_repeat)) {
        return p;
    }
    const int n = llama_sampler_chain_n(chain);
    int n_pen = 0, n_k = 0, n_p = 0, n_m = 0, n_t = 0;
    int i_pen = -1, i_p = -1, i_m = -1, i_t = -1;
    float tacc = 1.0f;
    bool seen_select = false;
    for (int i = 0; i < n; ++i) {
        bool         empty = false;
        const char * nm    = npu_strip(llama_sampler_name(llama_sampler_chain_get(chain, i)), &empty);
        if (!nm) {
            return p;
        }
        if (empty) {
            continue;  // a disabled option's no-op sampler
        }
        if (seen_select) {
            return p;  // dist / greedy must be last
        }
        if      (strcmp(nm, "penalties") == 0) { if (++n_pen > 1) return p; i_pen = i; }
        else if (strcmp(nm, "top-k")     == 0) { if (++n_k   > 1) return p; }
        else if (strcmp(nm, "top-p")     == 0) {
            if (++n_p > 1 || !(tacc > 0.0f) || (params.top_k > 0 && n_k == 0)) {
                return p;  // top-p before top-k would normalise over more than the K candidates
            }
            i_p = i;
            p.temp_sumexp = tacc;
        }
        else if (strcmp(nm, "min-p")     == 0) { if (++n_m   > 1) return p; i_m = i; p.temp_minp = tacc; }
        else if (strcmp(nm, "temp")      == 0 || strcmp(nm, "temp-ext") == 0) { if (++n_t > 1) return p; i_t = i; tacc *= params.temp; }
        else if (strcmp(nm, "dist")      == 0 || strcmp(nm, "greedy")   == 0) { if (seen_select) return p; seen_select = true; }
        else { return p; }  // DRY, XTC, typical, mirostat, grammar, logit bias, ...
    }
    if (!seen_select || (n_pen && i_pen != 0)) {
        return p;  // penalties must run on the whole vocabulary, before everything else
    }
    if (n_p && n_m && i_m < i_p) {
        return p;  // the host applies top-p before min-p, so min-p must come after it
    }
    if (n_t && n_p && i_t < i_p) {
        return p;  // the host's top-p divides by the kernel's untempered sumexp
    }
    if (params.top_k == 0) {
        // no top-k: the chain's distribution runs over the whole vocabulary unless top-p/min-p keep it inside K
        if (!(params.top_p < 1.0f || params.min_p > 0.0f)) {
            return p;
        }
    }
    p.ok = true;
    return p;
}

// top-p over the K (or top-k) candidates. With a top-k above it llama.cpp normalises over that set; without one it
// normalises over the whole vocabulary, whose sumexp the kernel returns.
static double npu_sumexp_eps() {
    static const double eps = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING_SUMEXP_EPS");
        return e && e[0] ? atof(e) : 2e-3;  // measured |hierarchical - sequential|/sequential: p50 5e-5, max 7.7e-4
    }();
    return eps;
}

// The kernel's sumexp is a hierarchical float sum; llama.cpp's top-p divides by a sequential sum over the full
// vocabulary in token order. The nucleus cut therefore lies between the cut at sumexp*(1 - eps) and the one at
// sumexp*(1 + eps). Returns the two cut indices (the number of retained candidates is index + 1), raised to min_keep.
static void npu_top_p_cuts(const llama_token_data_array * cur_p, float p, size_t min_keep, float sumexp, float temp,
                           int * lo, int * hi) {
    *lo = *hi = -1;
    if (p >= 1.0f || cur_p->size == 0 || !(sumexp > 0.0f)) {
        return;
    }
    const double eps   = npu_sumexp_eps();
    const float  max_l = cur_p->data[0].logit;
    const float  p_lo  = (float) ((double) p * (1.0 - eps));
    const float  p_hi  = (float) ((double) p * (1.0 + eps));
    float        c     = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        c += expf((cur_p->data[i].logit - max_l) / temp) / sumexp;
        if (*lo < 0 && c >= p_lo) {
            *lo = (int) i;
        }
        if (c >= p_hi) {
            *hi = (int) i;
            break;
        }
    }
    const int last = (int) cur_p->size - 1;
    if (*lo < 0) *lo = last;
    if (*hi < 0) *hi = last;
    const int mk = (int) min_keep - 1;
    *lo = std::min(last, std::max(*lo, mk));
    *hi = std::min(last, std::max(*hi, mk));
}

static void npu_top_p_apply(llama_token_data_array * cur_p, float p, size_t min_keep, bool local, float sumexp) {
    if (p >= 1.0f || cur_p->size == 0) {
        return;
    }
    const float max_l = cur_p->data[0].logit;  // the candidates are sorted
    if (!local) {
        // no top-k: the kernel's full-vocabulary sumexp is the denominator, so the tail after the nucleus is not
        // needed at all. Stop there (the same per-element float arithmetic as llama.cpp's softmax + cumulative sum).
        if (!(sumexp > 0.0f)) {
            return;
        }
        float  c        = 0.0f;
        size_t last_idx = cur_p->size;
        for (size_t i = 0; i < cur_p->size; ++i) {
            const float pi    = expf(cur_p->data[i].logit - max_l) / sumexp;
            cur_p->data[i].p  = pi;
            c                += pi;
            if (c >= p && i + 1 >= min_keep) {
                last_idx = i + 1;
                break;
            }
        }
        cur_p->size   = last_idx;
        cur_p->sorted = true;
        return;
    }
    float       cum   = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float q       = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p    = q;
        cum                += q;
    }
    if (cum > 0.0f) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= cum;
        }
    }
    float  c        = 0.0f;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        c += cur_p->data[i].p;
        if (c >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }
    cur_p->size   = last_idx;
    cur_p->sorted = true;
}

// LLAMA_NPU_SAMPLING_DEBUG=<file>: one block per sampling step, the candidates the chain ran on and its choice.
// fused=1 is the kernel's list (penalties already applied); fused=0 is the full-logit reference after the penalties
// sampler, sorted by logit (the first 24 rows are enough to see a value difference that can move a sampled draw).
static bool npu_dbg_on() {
    static const char * p = getenv("LLAMA_NPU_SAMPLING_DEBUG");
    return p && p[0];
}

static long long npu_dbg_step = 0;

static void npu_dbg_mark(const char * what, int fused, int K, int top_k, float top_p, float min_p, float temp) {
    if (!npu_dbg_on()) {
        return;
    }
    FILE * f = fopen(getenv("LLAMA_NPU_SAMPLING_DEBUG"), "a");
    if (!f) {
        return;
    }
    fprintf(f, "MARK %s fused=%d K=%d top_k=%d top_p=%.9g min_p=%.9g temp=%.9g\n",
            what, fused, K, top_k, top_p, min_p, temp);
    fclose(f);
}

static void npu_dbg_dump(int fused, int n, float vmax, float sumexp, const float * val, const int32_t * ids,
                         llama_token chosen, size_t size) {
    if (!npu_dbg_on()) {
        return;
    }
    FILE * f = fopen(getenv("LLAMA_NPU_SAMPLING_DEBUG"), "a");
    if (!f) {
        return;
    }
    fprintf(f, "STEP %lld fused=%d n=%d vmax=%.9g sumexp=%.9g chosen=%d size=%zu\n",
            npu_dbg_step++, fused, n, vmax, sumexp, chosen, size);
    const int m = n < 24 ? n : 24;
    for (int i = 0; i < m; ++i) {
        fprintf(f, " %d %.9g\n", ids[i], val[i]);
    }
    fclose(f);
}

static void npu_dbg_full(common_sampler * gsmpl, llama_token chosen) {
    if (!npu_dbg_on()) {
        return;
    }
    std::vector<llama_token_data> ref = gsmpl->cur;
    llama_token_data_array       rp  = { ref.data(), ref.size(), -1, false };
    const int                    nc  = llama_sampler_chain_n(gsmpl->chain);
    for (int i = 0; i < nc; ++i) {
        bool         empty = false;
        const char * nm    = npu_strip(llama_sampler_name(llama_sampler_chain_get(gsmpl->chain, i)), &empty);
        if (empty) {
            continue;
        }
        if (strcmp(nm, "penalties") == 0) {
            llama_sampler_apply(llama_sampler_chain_get(gsmpl->chain, i), &rp);
        }
        break;  // penalties are first, or absent
    }
    const int m = std::min<int>(24, (int) rp.size);
    std::partial_sort(rp.data, rp.data + m, rp.data + rp.size,
                      [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    std::vector<float>   val(m);
    std::vector<int32_t> ids(m);
    for (int i = 0; i < m; ++i) {
        val[i] = rp.data[i].logit;
        ids[i] = rp.data[i].id;
    }
    npu_dbg_dump(0, m, rp.data[0].logit, 0.0f, val.data(), ids.data(), chosen, rp.size);
}

// LLAMA_NPU_SAMPLING_PROF=1: the verification's host sampling time, one line per accepted batch (samples = the rows
// the batch sampled, i.e. the accepted drafts plus the target token)
static void npu_prof_spec(const char * kind, size_t samples, std::chrono::steady_clock::time_point t0, int64_t host_us) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING_PROF");
        return e && e[0] && strcmp(e, "0") != 0;
    }();
    if (!on) {
        return;
    }
    const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "npu spec sampling %s samples=%zu host_us=%lld wall_us=%.3f\n", kind, samples,
            (long long) host_us, wall);
}

// LLAMA_NPU_SAMPLING_TRACE_SPEC=1: the verification round's draft and every row's decision (for the paired gate)
static bool npu_trace_spec_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING_TRACE_SPEC");
        return e && e[0] && strcmp(e, "0") != 0;
    }();
    return on;
}

static void npu_trace_spec_round(const char * kind, const std::vector<int> & idxs, const llama_tokens & draft) {
    if (!npu_trace_spec_on()) {
        return;
    }
    fprintf(stderr, "spec trace %s round n_draft=%zu idxs=", kind, draft.size());
    for (size_t i = 0; i < idxs.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "", idxs[i]);
    }
    fprintf(stderr, " draft=");
    for (size_t i = 0; i < draft.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "", draft[i]);
    }
    fprintf(stderr, "\n");
}

static void npu_trace_spec_row(const char * kind, size_t i, int idx, llama_token draft, llama_token id, bool match) {
    if (!npu_trace_spec_on()) {
        return;
    }
    fprintf(stderr, "spec trace %s row i=%zu idx=%d draft=%d sampled=%d %s\n", kind, i, idx, (int) draft, (int) id,
            match ? "accept" : "reject");
}

// LLAMA_NPU_SAMPLING_CHECK=1: on a fused row, also run llama.cpp's own chain over that row's raw (full) logits with a
// clone of the chain (the same RNG state) and compare. The row comes from the adapter's regular head, so this is the
// in-process exactness gate on real rows, and it exercises the full_row fallback path.
static int64_t npu_fallback_rows = 0;
static int64_t npu_fb_ok = 0, npu_fb_bad = 0;
static int64_t npu_ok_k = 0, npu_bad_k = 0;    // top_k > 0: the candidates are the whole chain's support
static int64_t npu_ok_0 = 0, npu_bad_0 = 0;    // top_k == 0: the top-p cut uses the kernel's full-vocabulary sum
static int64_t npu_bad_0_cut = 0;              // ... and the mismatch was a different top-p cut (the fp32 sum order)

static bool npu_check_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING_CHECK");
        return e && e[0] && strcmp(e, "0") != 0;
    }();
    static const bool reg = [] {
        atexit([] {
            fprintf(stderr, "npu chain check: %lld verification rows fell back to the full logits; "
                            "fallback path %lld ok, %lld mismatches\n",
                    (long long) npu_fallback_rows, (long long) npu_fb_ok, (long long) npu_fb_bad);
            fprintf(stderr, "npu chain check: top-k rows %lld ok, %lld mismatches; "
                            "top-k=0 rows %lld ok, %lld mismatches (%lld a different top-p cut)\n",
                    (long long) npu_ok_k, (long long) npu_bad_k, (long long) npu_ok_0, (long long) npu_bad_0,
                    (long long) npu_bad_0_cut);
        });
        return true;
    }();
    (void) reg;
    return on;
}

// ref is a clone of the live chain taken BEFORE the live chain's draw, so the reference uses the same RNG state.
static void npu_chain_check(common_sampler * gsmpl, struct llama_context * ctx, int idx, llama_token npu_id,
                            llama_sampler * ref, float sumexp, bool local) {
    static int64_t n_ok = 0, n_bad = 0;
    static double max_p_diff = 0.0;
    static int64_t n_dist = 0;
    static const bool verbose = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING_CHECK_VERBOSE");
        return e && e[0] && strcmp(e, "0") != 0;
    }();
    static const bool reg = [] {
        atexit([] {
            fprintf(stderr, "npu chain check: %lld rows ok, %lld mismatches; distribution rows=%lld max|dp|=%.3g\n",
                    (long long) n_ok, (long long) n_bad, (long long) n_dist, max_p_diff);
        });
        return true;
    }();
    (void) reg;
    const int nv = (int) llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    gsmpl->npu_full.resize(nv);
    if (!gsmpl->npu->full_row(idx, gsmpl->npu_full.data(), nv)) {
        ++n_bad;
        fprintf(stderr, "npu chain check: row %d: no full-logit fallback\n", idx);
        return;
    }
    const float * lg = gsmpl->npu_full.data();
    std::vector<llama_token_data> full(nv);
    for (int i = 0; i < nv; ++i) {
        full[i] = llama_token_data{ (llama_token) i, lg[i], 0.0f };
    }
    llama_token_data_array fp = { full.data(), full.size(), -1, false };
    if (!ref) {
        ++n_bad;
        fprintf(stderr, "npu chain check: row %d: no chain clone\n", idx);
        return;
    }
    llama_sampler_apply(ref, &fp);
    const llama_token ref_id = fp.data[fp.selected].id;
    // llama.cpp's top-p softmax accumulates the full-vocabulary sum in token order (before its sort); the kernel's
    // sumexp uses a hierarchical order. Measure that difference when it matters (a top-k-less top-p boundary).
    double rel = 0.0;
    if (!local && (verbose || ref_id != npu_id)) {
        float seq = 0.0f, mx = lg[0];
        for (int i = 1; i < nv; ++i) {
            mx = std::max(mx, lg[i]);
        }
        for (int i = 0; i < nv; ++i) {
            seq += expf(lg[i] - mx);
        }
        rel = sumexp > 0.0f && seq > 0.0f ? fabs((double) sumexp - (double) seq) / (double) seq : 0.0;
    }
    // the target distribution the rejection variant reads: the same ids, and the same probabilities up to the float
    // chain (the fused and the full head project the same row with different accumulations)
    const llama_token_data_array & live = gsmpl->cur_p;
    if (live.size == fp.size) {
        bool same_ids = true;
        double dp = 0.0;
        for (size_t i = 0; i < fp.size; ++i) {
            same_ids = same_ids && live.data[i].id == fp.data[i].id;
            dp = std::max(dp, (double) fabsf(live.data[i].p - fp.data[i].p));
        }
        if (same_ids) {
            ++n_dist;
            max_p_diff = std::max(max_p_diff, dp);
        }
    }
    if (ref_id == npu_id) {
        ++n_ok;
        (local ? npu_ok_k : npu_ok_0)++;
        if (verbose) {
            fprintf(stderr, "npu chain check: row %d ok id=%d sumexp_rel=%.3g live.size=%zu ref.size=%zu\n", idx,
                    npu_id, rel, gsmpl->cur_p.size, fp.size);
        }
    } else {
        ++n_bad;
        if (local) {
            ++npu_bad_k;
        } else {
            ++npu_bad_0;
            if (gsmpl->cur_p.size != fp.size) {
                ++npu_bad_0_cut;
            }
        }
        fprintf(stderr, "npu chain check: row %d MISMATCH npu=%d full=%d sumexp_rel=%.3g live.size=%zu ref.size=%zu\n",
                idx, npu_id, ref_id, rel, gsmpl->cur_p.size, fp.size);
    }
}

// the host chain over the candidates: penalties were applied by the kernel; top-p without a top-k above it uses the
// full-vocabulary sumexp; everything else is the sampler itself (the same RNG for dist)
// force_cut >= 0: the top-p step keeps exactly that many candidates (index + 1) instead of computing its own cut; the
// chain's remaining samplers (min-p, dist) run on it. Used to sample the ambiguity band's two extremes with the same
// RNG state, which needs a clone, so the chain is a parameter.
static void npu_apply_chain_on(common_sampler * gsmpl, llama_sampler * chain, llama_token_data_array * cur_p,
                               float sumexp, int force_cut) {
    bool      top_k_applied = false;
    // With no top-k and top_p >= 1, llama.cpp's top-p sampler is a no-op and leaves the full-vocabulary
    // array in token order.  Its min-p sampler therefore takes the unsorted path and the dist sampler
    // draws in token-id order.  The kernel candidates arrive sorted by logit; mirror that order before
    // the draw so the same RNG state selects the same token.
    const bool minp_token_order = gsmpl->params.top_k == 0 && gsmpl->params.top_p >= 1.0f &&
                                  gsmpl->params.min_p > 0.0f;
    const int n             = llama_sampler_chain_n(chain);
    for (int i = 0; i < n; ++i) {
        llama_sampler * s     = llama_sampler_chain_get(chain, i);
        bool            empty = false;
        const char *    nm    = npu_strip(llama_sampler_name(s), &empty);
        if (empty || strcmp(nm, "penalties") == 0) {
            continue;
        }
        if (strcmp(nm, "top-k") == 0) {
            if (gsmpl->params.top_k > 0) {
                cur_p->size = std::min<size_t>(cur_p->size, (size_t) gsmpl->params.top_k);
                top_k_applied = true;
            }
            continue;
        }
        if (strcmp(nm, "top-p") == 0) {
            if (force_cut >= 0) {
                cur_p->size   = std::max<size_t>(std::min<size_t>((size_t) force_cut + 1, cur_p->size),
                                                 gsmpl->params.min_keep);
                cur_p->sorted = true;
            } else {
                npu_top_p_apply(cur_p, gsmpl->params.top_p, gsmpl->params.min_keep, top_k_applied, sumexp);
            }
            continue;
        }
        llama_sampler_apply(s, cur_p);
        if (minp_token_order && strcmp(nm, "min-p") == 0 &&
            cur_p->size >= gsmpl->params.min_keep) {
            std::sort(cur_p->data, cur_p->data + cur_p->size,
                      [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });
            cur_p->sorted = false;
        }
    }
}

static void npu_apply_chain(common_sampler * gsmpl, llama_token_data_array * cur_p, float sumexp) {
    npu_apply_chain_on(gsmpl, gsmpl->chain, cur_p, sumexp, -1);
}

static const ggml_axera_argmax * argmax_api() {
    static const ggml_axera_argmax * api = nullptr;
    if (!api) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_AXERA_NAME);
        if (reg) {
            api = (const ggml_axera_argmax *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_axera_argmax");
        }
    }
    return api;
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    std::vector<llama_logit_bias> fast_bias;  // (greedy_fast)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
            const int n_vocab = llama_vocab_n_tokens(vocab);
            std::map<llama_token, float> sum;
            for (const auto & b : merged) {
                if (b.token >= 0 && b.token < n_vocab) {
                    sum[b.token] += b.bias;
                }
            }
            for (const auto & [token, bias] : sum) {
                fast_bias.push_back({ token, bias });
            }
        }
    }

    bool greedy_chain = false;  // (greedy_fast) the chain ends in the greedy sampler and keeps the argmax
    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
            // every sampler before it keeps the largest logit, or does nothing with these parameters
            greedy_chain = greedy;
            for (const auto & cnstr : params.samplers) {
                switch (cnstr) {
                    case COMMON_SAMPLER_TYPE_TOP_K:
                    case COMMON_SAMPLER_TYPE_TOP_P:
                    case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    case COMMON_SAMPLER_TYPE_MIN_P:
                    case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    case COMMON_SAMPLER_TYPE_TEMPERATURE:
                        break;
                    case COMMON_SAMPLER_TYPE_DRY:
                        greedy_chain = greedy_chain && params.dry_multiplier == 0.0f;
                        break;
                    case COMMON_SAMPLER_TYPE_XTC:
                        greedy_chain = greedy_chain && params.xtc_probability <= 0.0f;
                        break;
                    case COMMON_SAMPLER_TYPE_PENALTIES:
                        greedy_chain = greedy_chain && (params.penalty_last_n == 0 || (params.penalty_repeat == 1.0f &&
                                       params.penalty_freq == 0.0f && params.penalty_present == 0.0f));
                        break;
                    default:  // infill, adaptive-p
                        greedy_chain = false;
                }
            }
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        /* .greedy_fast = */ false,
        /* .npu_argmax  = */ false,
        /* .npu_argmax_own = */ false,
        /* .fast_bias   = */ fast_bias,
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
        /* .npu     = */ nullptr,
        /* .npu_sampling = */ false,
        /* .npu_K   = */ 0,
        /* .npu_temp = */ 1.0f,
        /* .npu_exact_set = */ false,
        /* .npu_cfg = */ 0,
        /* .npu_v   = */ {},
        /* .npu_i   = */ {},
        /* .npu_full = */ {},
        /* .npu_saved = */ nullptr,
    };

    // LLAMA_SAMPLER_FAST=0: always the candidate array and the chain (a check)
    static const bool fast_env = [] {
        const char * e = getenv("LLAMA_SAMPLER_FAST");
        return e == nullptr || strcmp(e, "0") != 0;
    }();
    result->greedy_fast = fast_env && greedy_chain && !grmr && !rbudget && !params.backend_sampling;
    LOG_INF("%s: greedy fast path %s (greedy chain %d, %zu logit biases, grammar %d, reasoning budget %d, backend sampling %d)\n",
            __func__, result->greedy_fast ? "on" : "off", greedy_chain, fast_bias.size(), grmr != nullptr, rbudget != nullptr,
            params.backend_sampling);

    // the fused lm_head.topk: the backend applies the penalty state and keeps the exact top-K, and the chain above runs
    // on those candidates. Only when every sampler in it is covered exactly (see npu_plan_chain).
    // LLAMA_NPU_SAMPLING=0: the full-logit path (the gate's A/B run)
    static const bool npu_off = [] {
        const char * e = getenv("LLAMA_NPU_SAMPLING");
        return e && strcmp(e, "0") == 0;
    }();
    if (params.npu_sampling && !npu_off) if (const ggml_axera_npu_sampling * api = npu_api()) {
        // grammar, the reasoning budget and llama.cpp's own backend samplers read or write the full logits: always
        // reconfigure (so a previous request cannot leave the fused head on for a request that needs them). A pure
        // greedy chain keeps the existing argmax thread pool: it is exact and cheaper than the fused head.
        const bool eligible = params.npu_sampling && !params.backend_sampling && !grmr && !rbudget && !result->greedy_fast;
        const npu_chain_plan plan = eligible ? npu_plan_chain(chain, params) : npu_chain_plan{};
        const int K = api->configure(eligible && plan.ok, params.top_k, params.top_p, params.min_p,
                                     plan.temp_sumexp, plan.temp_minp, params.penalty_last_n,
                                     params.penalty_repeat, params.penalty_freq, params.penalty_present);
        // top_k > K cannot be served exactly: top-p would normalise over candidates the kernel did not return
        if (eligible && plan.ok && K > 0 && params.top_k <= K && (size_t) params.min_keep <= (size_t) K) {
            result->npu          = api;
            result->npu_sampling = true;
            result->npu_K        = K;
            result->npu_temp     = plan.temp_sumexp;
            result->npu_cfg      = api->config_id();
            result->npu_v.assign(K, 0.0f);
            result->npu_i.assign(K, 0);
            result->greedy_fast  = false;
            npu_dbg_mark("sampler", 1, K, params.top_k, params.top_p, params.min_p, plan.temp_sumexp);
            LOG_INF("%s: fused lm_head.topk sampling (K = %d)\n", __func__, K);
        } else {
            if (K > 0) {
                // the planner enabled the kernel but this request cannot consume its candidates: put the head back
                // on the full logits (top_k > K, min_keep > K)
                api->configure(0, 0, 0.0f, 0.0f, 1.0f, 1.0f, 0, 1.0f, 0.0f, 0.0f);
            }
            npu_dbg_mark("sampler", 0, 0, params.top_k, params.top_p, params.min_p, 1.0f);
        }
    }

    // The combined lm_head returns both the full logits and one (id, logit) per row. A pure greedy chain with no
    // caller-visible logits/probs and no grammar takes the NPU row; every other chain leaves this off and reads the
    // logits the same kernel wrote. GGML_AXERA_ARGMAX=0 is the A/B switch.
    static const bool argmax_off = [] {
        const char * e = getenv("GGML_AXERA_ARGMAX");
        return e && strcmp(e, "0") == 0;
    }();
    bool suppress_only = true;
    for (const auto & b : fast_bias) {
        suppress_only = suppress_only && std::isinf(b.bias) && b.bias < 0.0f;
    }
    const bool argmax_eligible = result->greedy_fast && suppress_only && params.n_probs == 0 &&
                                 !params.backend_sampling && !grmr && !rbudget && !argmax_off && !npu_off;
    if (const ggml_axera_argmax * api = argmax_api()) {
        std::vector<int32_t> ids;
        ids.reserve(fast_bias.size());
        for (const auto & b : fast_bias) {
            ids.push_back(b.token);
        }
        const int rows = api->configure(argmax_eligible, ids.empty() ? nullptr : ids.data(), (int32_t) ids.size());
        result->npu_argmax = rows > 0;
        result->npu_argmax_own = result->npu_argmax;
        if (result->npu_argmax) {
            LOG_INF("%s: fused lm_head greedy sampling (R = %d)\n", __func__, rows);
        }
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    // a request without sampling frees its sampler: the fused head must go back to writing the full logits (but a
    // sampler created after this one already owns the configuration, so only disable while this one still does)
    if (gsmpl->npu && gsmpl->npu_sampling && gsmpl->npu_cfg == gsmpl->npu->config_id()) {
        gsmpl->npu->configure(0, 0, 0.0f, 0.0f, 1.0f, 1.0f, 0, 1.0f, 0.0f, 0.0f);
    }
    if (gsmpl->npu && gsmpl->npu_saved) {
        gsmpl->npu->free_state(gsmpl->npu_saved);
        gsmpl->npu_saved = nullptr;
    }

    if (gsmpl->npu_argmax_own) {
        if (const ggml_axera_argmax * api = argmax_api()) {
            api->reset();
        }
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    if (gsmpl->npu_sampling) {
        gsmpl->npu->accept(token);  // the same ring buffer and counts, for the kernel's penalty arrays
    }

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();

    if (gsmpl->npu && gsmpl->npu_sampling) {
        gsmpl->npu->reset();  // the backend's penalty arrays go back to their neutral state before the prompt's tokens
    }
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .greedy_fast = */ gsmpl->greedy_fast,
        /* .npu_argmax  = */ gsmpl->npu_argmax,
        /* .npu_argmax_own = */ false,
        /* .fast_bias   = */ gsmpl->fast_bias,
        /* .rng     = */ gsmpl->rng,
        /* .npu     = */ gsmpl->npu,
        /* .npu_sampling = */ false,
        /* .npu_K   = */ 0,
        /* .npu_temp = */ gsmpl->npu_temp,
        /* .npu_exact_set = */ false,
        /* .npu_cfg = */ 0,
        /* .npu_v   = */ {},
        /* .npu_i   = */ {},
        /* .npu_full = */ {},
        /* .npu_saved = */ gsmpl->npu_sampling ? gsmpl->npu->save() : nullptr,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->greedy_fast = src->greedy_fast;
    dst->fast_bias   = src->fast_bias;
    dst->rng        = src->rng;
    // keep dst's backend binding: a clone is CPU-only (it may sample on a draft head), but copying a saved clone
    // back into the live sampler must not drop the fused head it was using
    dst->t_total_us = src->t_total_us;
    if (src->npu_saved && dst->npu) {
        dst->npu->restore(src->npu_saved);  // the penalty state the checkpoint was taken with
    }
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// the largest value of x[0 .. n) (NaN ignored)
static float max_of(const float * x, int n) {
    float m = -INFINITY;
    int   i = 0;
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(-INFINITY), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = vmaxnmq_f32(a0, vld1q_f32(x + i));
        a1 = vmaxnmq_f32(a1, vld1q_f32(x + i + 4));
        a2 = vmaxnmq_f32(a2, vld1q_f32(x + i + 8));
        a3 = vmaxnmq_f32(a3, vld1q_f32(x + i + 12));
    }
    m = vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(a0, a1), vmaxnmq_f32(a2, a3)));
#endif
    for (; i < n; ++i) {
        m = x[i] > m ? x[i] : m;
    }
    return m;
}

// the first index of m in x[b .. e)
static int first_of(const float * x, int b, int e, float m) {
    int i = b;
#if defined(__ARM_NEON)
    const float32x4_t mv = vdupq_n_f32(m);
    for (; i + 16 <= e; i += 16) {
        const uint32x4_t c0 = vceqq_f32(vld1q_f32(x + i), mv);
        const uint32x4_t c1 = vceqq_f32(vld1q_f32(x + i + 4), mv);
        const uint32x4_t c2 = vceqq_f32(vld1q_f32(x + i + 8), mv);
        const uint32x4_t c3 = vceqq_f32(vld1q_f32(x + i + 12), mv);
        if (vmaxvq_u32(vorrq_u32(vorrq_u32(c0, c1), vorrq_u32(c2, c3)))) {
            for (int j = i; j < i + 16; ++j) {
                if (x[j] == m) {
                    return j;
                }
            }
        }
    }
#endif
    for (; i < e; ++i) {
        if (x[i] == m) {
            return i;
        }
    }
    return -1;
}

// the greedy fast path's argmax over a vocabulary row (1 MB for 248k tokens): the maximum in parts on persistent
// workers, then only the winning part is scanned again for the first index
struct argmax_pool {
    static constexpr int     NW = 15; // workers; the caller takes part 0 (16 threads for a 1 MiB row)
    std::mutex               mu;
    std::condition_variable  cv;
    std::vector<std::thread> th;
    const float *            x     = nullptr;
    int                      n     = 0;
    int64_t                  gen   = 0;
    std::atomic<int>         left{ 0 };
    float                    part[NW + 1];

    argmax_pool() {
        for (int w = 1; w <= NW; ++w) {
            th.emplace_back([this, w] {
                int64_t seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return gen != seen; });
                    seen = gen;
                    lk.unlock();
                    part[w] = run(w);
                    left.fetch_sub(1, std::memory_order_release);
                }
            });
            th.back().detach();
        }
    }
    float run(int w) const {
        const int per = (n + NW) / (NW + 1);
        const int b = std::min(n, w * per), e = std::min(n, b + per);
        return max_of(x + b, e - b);
    }
    llama_token argmax(const float * v, int nv, float * best_v) {
        if (nv < 65536) {
            const float m = max_of(v, nv);
            *best_v = m;
            for (int j = 0; j < nv && m > -INFINITY; ++j) {
                if (v[j] == m) {
                    return j;
                }
            }
            return -1;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            x = v;
            n = nv;
            left.store(NW, std::memory_order_relaxed);
            ++gen;
        }
        cv.notify_all();
        part[0] = run(0);
        while (left.load(std::memory_order_acquire) > 0) {
        }
        const int per = (nv + NW) / (NW + 1);
        float     m   = -INFINITY;
        int       w0  = -1;
        for (int w = 0; w <= NW; ++w) {  // the first part holding the maximum
            if (part[w] > m) {
                m  = part[w];
                w0 = w;
            }
        }
        *best_v = m;
        if (w0 < 0) {
            return -1;
        }
        return first_of(v, w0 * per, std::min(nv, w0 * per + per), m);
    }
};

// GGML_AXERA_SAMPLE_DEBUG=1: the chosen id and the full row's two largest biased logits. The fused path reads the row
// here only for the diagnostic (it invalidates logits that the greedy path otherwise never touches).
static void debug_sample_row(const common_sampler * gsmpl, struct llama_context * ctx, int idx, llama_token id,
                             float logit, bool fused) {
    static const bool enabled = getenv("GGML_AXERA_SAMPLE_DEBUG") != nullptr;
    static int64_t     call = 0;
    if (!enabled) {
        return;
    }
    const float * logits  = llama_get_logits_ith(ctx, idx);
    const int     n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (!logits) {
        return;
    }
    llama_token i1 = -1, i2 = -1;
    float       v1 = -INFINITY, v2 = -INFINITY;
    for (llama_token i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        for (const auto & b : gsmpl->fast_bias) {
            if (b.token == i) {
                v += b.bias;
                break;
            }
        }
        if (v > v1) {
            v2 = v1; i2 = i1;
            v1 = v;  i1 = i;
        } else if (v > v2 && i != i1) {
            v2 = v;  i2 = i;
        }
    }
    fprintf(stderr, "SAMPLE_DEBUG call=%lld fused=%d id=%d logit=%.9g row1=%d %.9g row2=%d %.9g margin=%.9g\n",
            (long long) call++, fused ? 1 : 0, id, logit, i1, v1, i2, v2, v1 - v2);
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // the fused head's K candidates: penalties already applied on the NPU, sorted by logit
    bool have_full = false;
    // LLAMA_NPU_SAMPLING_CHECK: a fallback row's chain reference (taken before the live chain's draw)
    llama_sampler *    fb_ref    = nullptr;
    std::vector<float> fb_logits;
    if (gsmpl->npu_sampling) {
        const bool check = npu_check_on();
        static const int prof2 = [] {
            const char * e = getenv("LLAMA_NPU_SAMPLING_PROF2");
            return e ? atoi(e) : 0;
        }();
        const auto p0 = std::chrono::steady_clock::now();
        int32_t n = 0;
        float   vmax = 0.0f, sumexp = 0.0f;
        // with a top-k above the chain only the first top_k candidates can be selected, so there is no need to copy
        // (or build) the rest of K
        const int32_t take_k = gsmpl->params.top_k > 0 ? std::min(gsmpl->npu_K, gsmpl->params.top_k) : gsmpl->npu_K;
        const int take = gsmpl->npu->take(gsmpl->npu_v.data(), gsmpl->npu_i.data(), take_k, &n, &vmax, &sumexp);
        const auto p1 = std::chrono::steady_clock::now();
        bool need_full = false;  // the ambiguity trials found two different draws: take the full-logit row
        if (take == 1) {
            llama_sampler * ref = check ? llama_sampler_clone(gsmpl->chain) : nullptr;
            gsmpl->cur.resize(n);
            for (int32_t i = 0; i < n; ++i) {
                gsmpl->cur[i] = llama_token_data{ gsmpl->npu_i[i], gsmpl->npu_v[i], 0.0f };
            }
            cur_p = llama_token_data_array{ gsmpl->cur.data(), (size_t) n, 0, true };
            // Without a top-k the nucleus cut depends on the full-vocabulary sum, which the kernel sums in a different
            // order than llama.cpp does. When the cut is inside that error band, sample the band's two extreme cuts
            // with clones of the chain (the same RNG state): if they draw the same token the row is exact even though
            // the cut is not known, and a full-logit row is needed only when the draw actually differs.
            if (gsmpl->params.top_k == 0 && gsmpl->params.top_p < 1.0f) {
                int lo = -1, hi = -1;
                npu_top_p_cuts(&cur_p, gsmpl->params.top_p, gsmpl->params.min_keep, sumexp, gsmpl->npu_temp,
                               &lo, &hi);
                if (lo >= 0 && hi > lo && gsmpl->npu_exact_set) {
                    need_full = true;  // the target set itself must be exact here, not only the draw
                } else if (lo >= 0 && hi > lo) {
                    llama_token                   ids[2] = { LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL };
                    const int                     cuts[2] = { lo, hi };
                    std::vector<llama_token_data> tmp;
                    for (int t = 0; t < 2; ++t) {
                        llama_sampler * c = llama_sampler_clone(gsmpl->chain);
                        tmp.assign(cur_p.data, cur_p.data + cur_p.size);
                        llama_token_data_array a = { tmp.data(), tmp.size(), -1, true };
                        npu_apply_chain_on(gsmpl, c, &a, sumexp, cuts[t]);
                        GGML_ASSERT(a.selected >= 0 && (size_t) a.selected < a.size);
                        ids[t] = a.data[a.selected].id;
                        llama_sampler_free(c);
                    }
                    need_full = ids[0] != ids[1];
                    if (prof2 || need_full) {
                        fprintf(stderr, "npu row prof idx=%d ambiguous lo=%d hi=%d ids=%d/%d %s\n", idx, lo, hi,
                                (int) ids[0], (int) ids[1], need_full ? "fallback" : "same");
                    }
                }
            }
            const auto p2 = std::chrono::steady_clock::now();
            if (ref) {
                if (need_full) {
                    llama_sampler_free(ref);
                    ref = nullptr;
                }
            }
            if (!need_full) {
                npu_apply_chain(gsmpl, &cur_p, sumexp);
                const auto p3 = std::chrono::steady_clock::now();
                if (prof2) {
                    fprintf(stderr, "npu row prof idx=%d take=%.1f build=%.1f chain=%.1f us n=%d size=%zu\n", idx,
                            std::chrono::duration<double, std::micro>(p1 - p0).count(),
                            std::chrono::duration<double, std::micro>(p2 - p1).count(),
                            std::chrono::duration<double, std::micro>(p3 - p2).count(), n, cur_p.size);
                }
                GGML_ASSERT(cur_p.selected >= 0 && (size_t) cur_p.selected < cur_p.size);
                const llama_token npu_id = cur_p.data[cur_p.selected].id;
                npu_dbg_dump(1, n, vmax, sumexp, gsmpl->npu_v.data(), gsmpl->npu_i.data(), npu_id, cur_p.size);
                if (ref) {
                    const bool local = gsmpl->params.top_k > 0;
                    npu_chain_check(gsmpl, ctx, idx, npu_id, ref, sumexp, local);
                    llama_sampler_free(ref);
                }
                return npu_id;
            }
        }
        if (take < 0 || need_full) {
            ++npu_fallback_rows;
            if (prof2) {
                fprintf(stderr, "npu row prof idx=%d fallback take=%.1f us\n", idx,
                        std::chrono::duration<double, std::micro>(p1 - p0).count());
            }
            // this verification row cannot be proved from the K candidates: project its full logits through the
            // adapter's regular head and run the ordinary chain below over them
            const int nv = (int) llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
            gsmpl->npu_full.resize(nv);
            GGML_ASSERT(gsmpl->npu->full_row(idx, gsmpl->npu_full.data(), nv) &&
                        "no full-logit fallback for a fused-head row");
            gsmpl->cur.resize(nv);
            for (int i = 0; i < nv; ++i) {
                gsmpl->cur[i] = llama_token_data{ (llama_token) i, gsmpl->npu_full[i], 0.0f };
            }
            cur_p    = llama_token_data_array{ gsmpl->cur.data(), (size_t) nv, -1, false };
            have_full = true;
            if (check) {
                fb_ref    = llama_sampler_clone(gsmpl->chain);
                fb_logits = gsmpl->npu_full;
            }
            npu_dbg_mark("row-full", 0, 0, gsmpl->params.top_k, gsmpl->params.top_p, gsmpl->params.min_p,
                         gsmpl->params.temp);
        }
    }

    // The combined head already ran the final norm, projection and greedy reduction on the NPU. Consume its row
    // directly; a missing row (non-greedy graph, prompt batch, suppress-token winner) falls through to the full
    // logits, which the same kernel wrote.
    if (gsmpl->npu_argmax && llama_get_sampled_token_ith(ctx, idx) == LLAMA_TOKEN_NULL) {
        if (const ggml_axera_argmax * api = argmax_api()) {
            int32_t id_argmax = LLAMA_TOKEN_NULL;
            float   logit     = 0.0f;
            if (api->take(idx, &id_argmax, &logit)) {
                debug_sample_row(gsmpl, ctx, idx, (llama_token) id_argmax, logit, true);
                gsmpl->cur.resize(1);
                gsmpl->cur[0] = llama_token_data{ (llama_token) id_argmax, logit, 1.0f };
                cur_p         = { gsmpl->cur.data(), 1, 0, true };
                return (llama_token) id_argmax;
            }
        }
    }

    if (gsmpl->greedy_fast && llama_get_sampled_token_ith(ctx, idx) == LLAMA_TOKEN_NULL &&
        !llama_get_sampled_logits_ith(ctx, idx) && !llama_get_sampled_probs_ith(ctx, idx)) {
        const float * logits  = llama_get_logits_ith(ctx, idx);
        const int     n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        GGML_ASSERT(logits != nullptr);
        // the largest logit (the first of equal ones) after the logit biases, which add to their tokens' logits
        // (the biased tokens in token order, a plain argmax over the runs between them)
        llama_token best = -1;
        float       bv   = -INFINITY;
        llama_token i    = 0;
        if (gsmpl->fast_bias.empty()) {
            static argmax_pool & pool = *new argmax_pool;  // (never destroyed: its workers wait on its condition variable)
            static const bool prof = getenv("LLAMA_GRAPH_PROF") != nullptr;
            static int64_t prof_n = 0, prof_us = 0;
            const int64_t  prof_t0 = prof ? ggml_time_us() : 0;
            i    = n_vocab;
            best = pool.argmax(logits, n_vocab, &bv);
            if (prof) {
                prof_us += ggml_time_us() - prof_t0;
                if (++prof_n % 32 == 0) {
                    fprintf(stderr, "greedy argmax: %.3f ms/call (%lld)\n", prof_us / 1e3 / prof_n, (long long) prof_n);
                }
            }
        }
        auto        run  = [&](llama_token end) {
            for (; i < end; ++i) {
                if (logits[i] > bv) {
                    bv   = logits[i];
                    best = i;
                }
            }
        };
        for (const auto & b : gsmpl->fast_bias) {
            run(b.token);
            if (logits[i] + b.bias > bv) {
                bv   = logits[i] + b.bias;
                best = i;
            }
            ++i;
        }
        run(n_vocab);
        if (best < 0) {  // every logit -inf: the first token, as the greedy sampler over the array
            best = 0;
        }
        debug_sample_row(gsmpl, ctx, idx, best, bv, false);
        gsmpl->cur.resize(1);
        gsmpl->cur[0] = llama_token_data{ best, bv, 1.0f };
        cur_p         = { gsmpl->cur.data(), 1, 0, true };
        return best;
    }

    if (!have_full) {
        gsmpl->set_logits(ctx, idx);
    }
    npu_dbg_full(gsmpl, LLAMA_TOKEN_NULL);  // the full-logit reference, before the chain mutates it

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (fb_ref) {
        // the fallback row ran the ordinary chain over the regular head's logits; a clone with the same RNG state must
        // pick the same token from the same raw row
        std::vector<llama_token_data> full(fb_logits.size());
        for (size_t i = 0; i < fb_logits.size(); ++i) {
            full[i] = llama_token_data{ (llama_token) i, fb_logits[i], 0.0f };
        }
        llama_token_data_array fp = { full.data(), full.size(), -1, false };
        llama_sampler_apply(fb_ref, &fp);
        if (fp.data[fp.selected].id == id) {
            ++npu_fb_ok;
        } else {
            ++npu_fb_bad;
            fprintf(stderr, "npu chain check: fallback row %d MISMATCH live=%d chain=%d\n", idx, (int) id,
                    (int) fp.data[fp.selected].id);
        }
        llama_sampler_free(fb_ref);
        fb_ref = nullptr;
    }

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    const auto npa_t0 = std::chrono::steady_clock::now();
    const int64_t npa_us0 = gsmpl->t_total_us;
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    npu_trace_spec_round("match", idxs, draft);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        npu_trace_spec_row("match", i, idxs[i], draft[i], id, draft[i] == id);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        npu_trace_spec_row("match", i, idxs[i], -1, id, true);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    npu_prof_spec("match", result.size(), npa_t0, gsmpl->t_total_us - npa_us0);
    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    const auto npa_t0 = std::chrono::steady_clock::now();
    const int64_t npa_us0 = gsmpl->t_total_us;
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");
    npu_trace_spec_round("rejection", idxs, draft);
    gsmpl->npu_exact_set = true;

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            npu_trace_spec_row("rejection", i, idxs[i], draft[i], draft[i], true);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        npu_trace_spec_row("rejection", i, idxs[i], draft[i], id, false);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        npu_trace_spec_row("rejection", i, idxs[i], -1, id, true);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    gsmpl->npu_exact_set = false;
    npu_prof_spec("rejection", result.size(), npa_t0, gsmpl->t_total_us - npa_us0);
    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
