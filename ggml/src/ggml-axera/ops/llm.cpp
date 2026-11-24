#include "llm.h"

#include "../core/runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace axera {

ggml_tensor * llm_input(ggml_cgraph * g, ggml_tensor * norm0, bool contiguous) {
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0] && x0->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(!contiguous || ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) {
            ++n0;
        }
        axera_timer tc("input (cpu graph)");
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    return x0->src[0];
}

int llm_layers(const std::map<std::string, ggml_tensor *> & t) {
    int L = 0;
    while (t.count("blk." + std::to_string(L) + ".attn_norm.weight")) {
        ++L;
    }
    return L;
}

const ggml_tensor * llm_rope_node(const ggml_cgraph * g) {
    for (int i = 0; i < g->n_nodes; ++i) {
        if (g->nodes[i]->op == GGML_OP_ROPE) {
            return g->nodes[i];
        }
    }
    return nullptr;
}

bool llm_one_sequence(const kv_view & kv, int T, const char * who) {
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera %s: one sequence only (-np 1)\n", who);
        return false;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera %s: a batch of non-consecutive cells\n", who);
            return false;
        }
    }
    return true;
}

void llm_positions(const ggml_tensor * inp_pos, int T, std::vector<int32_t> & p4) {
    const int32_t * pos = (const int32_t *) inp_pos->data;
    const bool      mr  = inp_pos->ne[0] == 4 * (int64_t) T;  // M-RoPE: [4][T], else [T]
    p4.resize((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            p4[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
        }
    }
}

int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

seq_config seq_config_env(int kept_max, int verify_chain) {
    seq_config c;
    c.kept_max     = kept_max;
    c.verify_chain = verify_chain;
    c.check        = env_int("GGML_AXERA_SEQ_CHECK", 0) != 0;
    c.force_miss   = std::max(0, env_int("GGML_AXERA_SEQ_FORCE_MISS", 0));
    return c;
}

seq_status llm_seq_log(seq_status s, int64_t c0, const seqstate & seq, const char * who) {
    if (s == SEQ_MISS) {
        GGML_LOG_WARN("axera %s: batch from cell %lld has no restart point (%s); the slot must replay\n", who,
                      (long long) c0, seq.why());
    } else if (s != SEQ_OK) {
        GGML_LOG_ERROR("axera %s: sequence state rejected a batch from cell %lld: %s\n", who, (long long) c0,
                       seq.why());
    }
    return s;
}

int llm_kv_pages(int max_pages, int pg, bool sizable) {
    int p = max_pages;
    if (const int64_t cells = axera_kv_cells(); cells > 0 && sizable) {
        p = (int) std::max<int64_t>(1, std::min<int64_t>(max_pages, (cells + pg - 1) / pg));
    }
    if (const char * e = getenv("GGML_AXERA_KV_PAGES")) {
        p = std::max(1, std::min(max_pages, atoi(e)));
    }
    return p;
}

void llm_paging(const axera_cmm & indptr, const axera_cmm & last_len, const axera_cmm & cloc, int64_t hist,
                int64_t write_cell, int pg) {
    const int64_t pages           = (hist + pg - 1) / pg;
    ((int32_t *) indptr->vir)[0]  = 0;
    ((int32_t *) indptr->vir)[1]  = (int32_t) pages;
    *(int32_t *) last_len->vir    = (int32_t) (pages ? hist - (pages - 1) * pg : 0);
    ((int32_t *) cloc->vir)[0]    = (int32_t) (write_cell / pg);
    ((int32_t *) cloc->vir)[1]    = (int32_t) (write_cell % pg);
    for (const axera_cmm * b : { &indptr, &last_len, &cloc }) {
        (*b)->flush();
    }
}

std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) {
            memcpy(v.data() + (size_t) r * w->ne[0], row, (size_t) w->ne[0] * sizeof(float));
        });
    }
    return v;
}

}  // namespace axera
