// Axera NPU backend: Inkling-2-Mini (24 physical layers, 48 cache slots: the three loop passes over layers 6..17
// restart from the pre-loop hidden / block state and reuse the same weight banks).
//
// The manifest has two merged kernels:
//   prefill  the whole model for 256 rows (48 slot applications, block/message residuals, final norm and the
//            NVFP4 output head in one AXModel).
//   decode   the whole model for one row.
// KV: the kernel takes a linear, 2048-entry history per slot and appends the batch's own K/V inside the graph; the
// host commits the returned K/V at the write cursor.  Local slots mask everything older than 1024 tokens, global
// slots keep the whole 2048.  The rel index / attention mask inputs are built on the host per call.
// State: the KV history, the four short-convolution tails per slot and the write cursor live in the adapter's CMM;
// they are what state_write / state_read carry.
#include "../model.h"
#include "../../core/quant.h"
#include "../../ops/llm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace axera {
namespace {

constexpr int H     = 2304;
constexpr int NQ    = 24;
constexpr int NKV   = 8;
constexpr int HD    = 128;
constexpr int C     = 2048;  // KV history the kernels take
constexpr int WP    = 256;   // prefill rows
constexpr int LAYERS = 24;
constexpr int SLOTS = 48;
constexpr int KVCH  = NKV * HD;
constexpr int KW    = 4;     // short-convolution kernel size

constexpr int FULL_SET[4] = {5, 11, 17, 23};
constexpr int MODULES[SLOTS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                18, 19, 20, 21, 22, 23};
constexpr int SLOT_OF[SLOTS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                18, 19, 20, 21, 22, 23};

bool is_full(int u) {
    for (int f : FULL_SET) {
        if (f == u) return true;
    }
    return false;
}

const char * RUNTIME_IN[14] = {"x", "pos", "rel_idx_l", "rel_idx_g", "mask_l", "mask_g",
                               "keys_l", "values_l", "keys_g", "values_g",
                               "kconv", "vconv", "paconv", "pmconv"};
const char * RUNTIME_OUT[13] = {"apps_out", "blocks_out", "logits", "token_id", "final_hidden",
                                "new_keys_l", "new_values_l", "new_keys_g", "new_values_g",
                                "new_kconv", "new_vconv", "new_paconv", "new_pmconv"};

bool is_runtime_in(const std::string & n) {
    for (const char * s : RUNTIME_IN) {
        if (n == s) return true;
    }
    return false;
}

struct iobank {
    axera_cmm c;
    AX_ENGINE_IO_BUFFER_T buf() const { return axera_io(c, (size_t) c->size); }
};

class inkling2 final : public model {
  public:
    explicit inkling2(const manifest & m) : man(m) {}

    ggml_status compute(ggml_cgraph * g) override;
    size_t      state_size(uint32_t flags) const override;
    size_t      state_write(void * dst, size_t size, uint32_t flags) override;
    size_t      state_read(const void * src, size_t size, uint32_t flags) override;

  private:
    const manifest & man;
    std::string      tag = "axera inkling2";
    graph_names      names;
    const void *     bound = nullptr;
    bool             inited = false;
    int              VOCAB = 0;

    const kernel * pk = nullptr;
    const kernel * dk = nullptr;

    std::map<std::string, iobank> wbank;    // weight inputs, shared by both kernels
    std::map<std::string, iobank> pin, pout, din, dout;

    axera_cmm kvk_l, kvv_l, kvk_g, kvv_g;   // [40][8][C][128] / [8][8][C][128] bf16
    axera_cmm kc, vc, pca, pmc;             // prefill tails [48][3][1024] / [48][3][H] f32
    axera_cmm kc4, vc4, pca4, pmc4;         // decode tails [48][4][1024] / [48][4][H] f32
    int64_t   n_past = 0;
    std::vector<int32_t> pbuf;
    bool      reset = true;

    void init(const std::map<std::string, ggml_tensor *> & t);
    void fill_weights(const std::map<std::string, ggml_tensor *> & t);
    void fill_one(const std::string & name, const io_desc & d, const std::map<std::string, ggml_tensor *> & t);
    void clear_state();
    void run_call(const kernel & k, std::map<std::string, iobank> & in, std::map<std::string, iobank> & out,
                  const std::vector<std::string> & in_names, const std::vector<std::string> & out_names,
                  const char * what);
    bool run_prefill(const uint16_t * x, const int32_t * pos, int n);
    bool run_decode_row(const uint16_t * x, int32_t pos);
    void commit(bool pre, int n);
    void build_mask_prefill(std::vector<int32_t> & ridx, std::vector<float> & mask, const int32_t * pos, int n,
                            bool full);
    void build_mask_decode(std::vector<int32_t> & ridx, std::vector<float> & mask, int32_t pos, bool full);
    void publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T, int first_row);
};

static const io_desc * find_io(const std::vector<io_desc> & v, const std::string & n) {
    for (const io_desc & d : v) {
        if (d.name == n) return &d;
    }
    return nullptr;
}

static bool starts(const std::string & s, const char * p) {
    return s.rfind(p, 0) == 0;
}

static std::string layer_gguf(int u, const std::string & key) {
    static const std::map<std::string, std::string> m = {
        {"wq", "attn_q.weight"}, {"wk", "attn_k.weight"}, {"wv", "attn_v.weight"},
        {"wo", "attn_output.weight"}, {"attn_gate", "attn_gate.weight"},
        {"wup", "ffn_gate_up_exps.weight"}, {"wdn", "ffn_down_exps.weight"},
        {"attn_norm_w", "attn_norm.weight"}, {"ffn_norm_w", "ffn_norm.weight"},
        {"attn_q_norm_w", "attn_q_input_norm.weight"}, {"attn_k_norm_w", "attn_k_input_norm.weight"},
        {"attn_v_norm_w", "attn_v_input_norm.weight"},
        {"k_conv", "k_conv.weight"}, {"v_conv", "v_conv.weight"},
        {"k_conv_static", "k_conv_static.weight"}, {"v_conv_static", "v_conv_static.weight"},
        {"k_conv_proj", "k_conv_proj.weight"}, {"v_conv_proj", "v_conv_proj.weight"},
        {"attn_conv", "attn_conv.weight"}, {"mlp_conv", "mlp_conv.weight"},
        {"q_scale", "attn_q_scale.weight"}, {"rel_linear", "attn_rel_linear.weight"},
        {"rel_scale", "attn_rel_scale.weight"}, {"rel_proj", "attn_rel_proj.weight"},
        {"sink_proj", "attn_sink_proj.weight"}, {"attn_input_q", "attn_input_q.weight"},
        {"mlp_input_q", "mlp_input_q.weight"}, {"router_w", "ffn_gate_inp.weight"},
        {"router_b", "exp_probs_b.weight"}, {"msg_down", "msg_down.weight"},
        {"msg_forget", "msg_forget.weight"}, {"msg_gain", "msg_gain.weight"},
    };
    auto it = m.find(key);
    GGML_ASSERT(it != m.end());
    return "blk." + std::to_string(u) + "." + it->second;
}

// fill one weight input bank from the GGUF mapping
void inkling2::fill_one(const std::string & name, const io_desc & d, const std::map<std::string, ggml_tensor *> & t) {
    auto bank = wbank.find(name);
    GGML_ASSERT(bank != wbank.end() && bank->second.c->size >= d.bytes);
    uint8_t * dst = (uint8_t *) bank->second.c->vir;
    memset(dst, 0, bank->second.c->size);

    auto tensor = [&](const std::string & n) -> ggml_tensor * {
        auto it = t.find(n);
        if (it == t.end()) {
            GGML_ABORT("axera inkling2: %s not in the graph\n", n.c_str());
        }
        return it->second;
    };

    if (name == "message_input_q" || name == "attn_res_final_q" || name == "output_norm") {
        static const std::map<std::string, std::string> g = {
            {"message_input_q", "message_input_q.weight"},
            {"attn_res_final_q", "attn_res_final_q.weight"},
            {"output_norm", "output_norm.weight"}};
        ggml_tensor * w = tensor(g.at(name));
        GGML_ASSERT(ggml_nbytes(w) == d.bytes && w->type == GGML_TYPE_F32);
        memcpy(dst, w->data, d.bytes);
        return;
    }
    if (name == "head_w" || name == "head_s" || name == "head_s2") {
        ggml_tensor * w = tensor("output.weight");
        if (name == "head_s2") {
            ggml_tensor * s = tensor("output.weight.scale");
            GGML_ASSERT(s->type == GGML_TYPE_F32);
            memcpy(dst, s->data, sizeof(float));
            return;
        }
        const int64_t K = w->ne[0], N = ggml_nrows(w);
        GGML_ASSERT(w->type == GGML_TYPE_NVFP4);
        if (name == "head_w") {
            axera_copy_nvfp4_slice(w, 0, N, dst, K / 2, (uint8_t *) dst + (size_t) N * (K / 16), K / 16);
        } else {
            std::vector<uint8_t> q((size_t) N * (K / 2));
            axera_copy_nvfp4_slice(w, 0, N, q.data(), K / 2, dst, K / 16);
        }
        return;
    }

    // l<u>_<key> or l<u>_<key>_s / _s2
    GGML_ASSERT(starts(name, "l"));
    const size_t us = name.find('_');
    const int u = std::atoi(name.substr(1, us - 1).c_str());
    std::string key = name.substr(us + 1);
    bool scales = false, scale2 = false;
    if (key.size() > 3 && key.compare(key.size() - 3, 3, "_s2") == 0) {
        key.resize(key.size() - 3);
        scale2 = true;
    } else if (key.size() > 2 && key.compare(key.size() - 2, 2, "_s") == 0) {
        key.resize(key.size() - 2);
        scales = true;
    }
    ggml_tensor * w = tensor(layer_gguf(u, key));

    if (key == "wq" || key == "wk" || key == "wv" || key == "wo" || key == "attn_gate") {
        GGML_ASSERT(w->type == GGML_TYPE_NVFP4 && w->ne[2] == 1);
        const int64_t K = w->ne[0], N = w->ne[1];
        const std::string gname = layer_gguf(u, key);
        if (scale2) {
            ggml_tensor * s = tensor(gname.substr(0, gname.size() - 7) + ".scale");
            GGML_ASSERT(s->type == GGML_TYPE_F32 && ggml_nelements(s) == 1 && d.bytes == sizeof(float));
            memcpy(dst, s->data, sizeof(float));
        } else if (scales) {
            std::vector<uint8_t> q((size_t) N * (K / 2));
            axera_copy_nvfp4_slice(w, 0, N, q.data(), K / 2, dst, K / 16);
        } else {
            axera_copy_nvfp4_slice(w, 0, N, dst, K / 2, (uint8_t *) dst + (size_t) N * (K / 2), K / 16);
        }
        return;
    }
    if (key == "wup" || key == "wdn") {
        GGML_ASSERT(w->type == GGML_TYPE_NVFP4);
        const int64_t K = w->ne[0], N = ggml_nrows(w);
        if (scale2) {
            const std::string gname = key == "wup" ? "blk." + std::to_string(u) + ".ffn_up_exps.scale"
                                                   : "blk." + std::to_string(u) + ".ffn_down_exps.scale";
            ggml_tensor * s = tensor(gname);
            GGML_ASSERT(s->type == GGML_TYPE_F32 && ggml_nbytes(s) == d.bytes);
            memcpy(dst, s->data, d.bytes);
        } else if (scales) {
            std::vector<uint8_t> q((size_t) N * (K / 2));
            axera_copy_nvfp4_slice(w, 0, N, q.data(), K / 2, dst, K / 16);
        } else {
            axera_copy_nvfp4_slice(w, 0, N, dst, K / 2, (uint8_t *) dst + (size_t) N * (K / 16), K / 16);
        }
        return;
    }
    if (key == "router_w") {
        GGML_ASSERT(w->type == GGML_TYPE_F32 && (size_t) ggml_nelements(w) * 2 == d.bytes);
        const float * f = (const float *) w->data;
        ggml_fp32_to_bf16_row(f, (ggml_bf16_t *) dst, ggml_nelements(w));
        return;
    }
    if (key == "k_conv" || key == "v_conv" || key == "attn_conv" || key == "mlp_conv" ||
        key == "k_conv_static" || key == "v_conv_static") {
        // GGUF [taps, channels] -> the kernel's [channels, taps] transposed view
        const int64_t taps = w->ne[0], ch = w->ne[1];
        GGML_ASSERT(w->type == GGML_TYPE_F32 && ggml_nbytes(w) == d.bytes);
        const float * f = (const float *) w->data;
        float * o = (float *) dst;
        for (int64_t r = 0; r < ch; ++r) {
            for (int64_t j = 0; j < taps; ++j) {
                o[j * ch + r] = f[r * taps + j];
            }
        }
        return;
    }
    if (key == "attn_input_q" && u == 0) {
        return;  // layer 0 reads no block states: the input is a zero bank
    }
    // small f32 tensors copied as they are (mlp_input_q / attn_input_q may be reshaped by shape only)
    GGML_ASSERT(ggml_nbytes(w) == d.bytes && w->type == GGML_TYPE_F32);
    memcpy(dst, w->data, d.bytes);
}

void inkling2::fill_weights(const std::map<std::string, ggml_tensor *> & t) {
    std::map<std::string, const io_desc *> desc;
    for (const io_desc & d : pk->desc->in) {
        if (!is_runtime_in(d.name)) desc[d.name] = &d;
    }
    for (const io_desc & d : dk->desc->in) {
        if (!is_runtime_in(d.name)) desc[d.name] = &d;
    }
    for (const auto & [name, d] : desc) {
        fill_one(name, *d, t);
        wbank.at(name).c->flush();
    }
}

void inkling2::init(const std::map<std::string, ggml_tensor *> & t) {
    pk = &man.get("prefill");
    dk = &man.get("decode");

    const io_desc * lg = find_io(pk->desc->out, "logits");
    GGML_ASSERT(lg && lg->shape.size() == 2);
    VOCAB = (int) lg->shape[1];

    for (const io_desc & d : pk->desc->in) {
        if (is_runtime_in(d.name)) continue;
        if (!wbank.count(d.name)) {
            wbank[d.name] = {axera_cmm_alloc_bank(d.bytes)};
        } else {
            GGML_ASSERT(wbank.at(d.name).c->size == d.bytes);
        }
    }
    for (const io_desc & d : dk->desc->in) {
        if (is_runtime_in(d.name)) continue;
        GGML_ASSERT(wbank.count(d.name) && wbank.at(d.name).c->size == d.bytes);
    }

    auto alloc_io = [](std::map<std::string, iobank> & m, const io_desc & d) {
        if (!m.count(d.name)) m[d.name] = {axera_cmm_alloc(d.bytes)};
        GGML_ASSERT(m.at(d.name).c->size >= d.bytes);
    };
    auto persistent_in = [](const std::string & n) {
        return n == "keys_l" || n == "values_l" || n == "keys_g" || n == "values_g" ||
               n == "kconv" || n == "vconv" || n == "paconv" || n == "pmconv";
    };
    for (const io_desc & d : pk->desc->in) {
        if (is_runtime_in(d.name) && !persistent_in(d.name)) alloc_io(pin, d);
    }
    for (const io_desc & d : pk->desc->out) alloc_io(pout, d);
    for (const io_desc & d : dk->desc->in) {
        if (is_runtime_in(d.name) && !persistent_in(d.name)) alloc_io(din, d);
    }
    for (const io_desc & d : dk->desc->out) alloc_io(dout, d);

    // the KV history is shared by both kernels: [slots][heads][C][HD] bf16
    kvk_l = axera_cmm_alloc_bank((size_t) 40 * KVCH * C * sizeof(ggml_bf16_t));
    kvv_l = axera_cmm_alloc_bank((size_t) 40 * KVCH * C * sizeof(ggml_bf16_t));
    kvk_g = axera_cmm_alloc_bank((size_t) 8 * KVCH * C * sizeof(ggml_bf16_t));
    kvv_g = axera_cmm_alloc_bank((size_t) 8 * KVCH * C * sizeof(ggml_bf16_t));
    kc  = axera_cmm_alloc_bank((size_t) SLOTS * (KW - 1) * 1024 * sizeof(float));
    vc  = axera_cmm_alloc_bank((size_t) SLOTS * (KW - 1) * 1024 * sizeof(float));
    pca = axera_cmm_alloc_bank((size_t) SLOTS * (KW - 1) * H * sizeof(float));
    pmc = axera_cmm_alloc_bank((size_t) SLOTS * (KW - 1) * H * sizeof(float));
    kc4  = axera_cmm_alloc_bank((size_t) SLOTS * KW * 1024 * sizeof(float));
    vc4  = axera_cmm_alloc_bank((size_t) SLOTS * KW * 1024 * sizeof(float));
    pca4 = axera_cmm_alloc_bank((size_t) SLOTS * KW * H * sizeof(float));
    pmc4 = axera_cmm_alloc_bank((size_t) SLOTS * KW * H * sizeof(float));
    const io_desc * kl = find_io(pk->desc->in, "keys_l");
    GGML_ASSERT(kl && (size_t) 40 * KVCH * C * sizeof(ggml_bf16_t) == kl->bytes);

    n_past = 0;
    reset  = true;
    clear_state();
    fill_weights(t);
    GGML_LOG_INFO("%s: vocab %d, KV %d cells, banks %.2f GiB\n", tag.c_str(), VOCAB, C,
                  2.0 * (kvk_l->size + kvk_g->size) / 1073741824.0);
    inited = true;
}

void inkling2::clear_state() {
    memset(kvk_l->vir, 0, kvk_l->size);
    memset(kvv_l->vir, 0, kvv_l->size);
    memset(kvk_g->vir, 0, kvk_g->size);
    memset(kvv_g->vir, 0, kvv_g->size);
    memset(kc->vir, 0, kc->size);
    memset(vc->vir, 0, vc->size);
    memset(pca->vir, 0, pca->size);
    memset(pmc->vir, 0, pmc->size);
    memset(kc4->vir, 0, kc4->size);
    memset(vc4->vir, 0, vc4->size);
    memset(pca4->vir, 0, pca4->size);
    memset(pmc4->vir, 0, pmc4->size);
    kvk_l->flush(kvk_l->vir, kvk_l->size);
    kvv_l->flush(kvv_l->vir, kvv_l->size);
    kvk_g->flush(kvk_g->vir, kvk_g->size);
    kvv_g->flush(kvv_g->vir, kvv_g->size);
    kc->flush(kc->vir, kc->size);
    vc->flush(vc->vir, vc->size);
    pca->flush(pca->vir, pca->size);
    pmc->flush(pmc->vir, pmc->size);
    kc4->flush(kc4->vir, kc4->size);
    vc4->flush(vc4->vir, vc4->size);
    pca4->flush(pca4->vir, pca4->size);
    pmc4->flush(pmc4->vir, pmc4->size);
    n_past = 0;
}

void inkling2::run_call(const kernel & k, std::map<std::string, iobank> & in, std::map<std::string, iobank> & out,
                        const std::vector<std::string> & in_names, const std::vector<std::string> & out_names,
                        const char * what) {
    std::vector<AX_ENGINE_IO_BUFFER_T> iv, ov;
    iv.reserve(in_names.size());
    ov.reserve(out_names.size());
    for (const std::string & n : in_names) {
        const io_desc * d = find_io(k.desc->in, n);
        GGML_ASSERT(d);
        if (n == "keys_l") { iv.push_back(axera_io(kvk_l, d->bytes)); continue; }
        if (n == "values_l") { iv.push_back(axera_io(kvv_l, d->bytes)); continue; }
        if (n == "keys_g") { iv.push_back(axera_io(kvk_g, d->bytes)); continue; }
        if (n == "values_g") { iv.push_back(axera_io(kvv_g, d->bytes)); continue; }
        const bool pre = &k == pk;
        if (n == "kconv") { iv.push_back(axera_io(pre ? kc : kc4, d->bytes)); continue; }
        if (n == "vconv") { iv.push_back(axera_io(pre ? vc : vc4, d->bytes)); continue; }
        if (n == "paconv") { iv.push_back(axera_io(pre ? pca : pca4, d->bytes)); continue; }
        if (n == "pmconv") { iv.push_back(axera_io(pre ? pmc : pmc4, d->bytes)); continue; }
        auto it = in.find(n);
        if (it != in.end()) {
            iv.push_back(axera_io(it->second.c, (size_t) it->second.c->size));
        } else if (wbank.count(n)) {
            iv.push_back(axera_io(wbank.at(n).c, d->bytes));
        } else {
            GGML_ABORT("%s: no input %s\n", tag.c_str(), n.c_str());
        }
    }
    for (const std::string & n : out_names) {
        auto it = out.find(n);
        GGML_ASSERT(it != out.end());
        const io_desc * d = find_io(k.desc->out, n);
        GGML_ASSERT(d);
        ov.push_back(axera_io(it->second.c, d->bytes));
    }
    k.run(iv, ov, what);
}

static void collect_names(const std::vector<io_desc> & v, std::vector<std::string> & n) {
    n.clear();
    for (const io_desc & d : v) n.push_back(d.name);
}

void inkling2::build_mask_prefill(std::vector<int32_t> & ridx, std::vector<float> & mask, const int32_t * pos, int n,
                                  bool full) {
    const int W = C + n;
    ridx.assign((size_t) n * NQ * W, 1024);
    mask.assign((size_t) n * NQ * W, -INFINITY);
    const int64_t base = n_past;
    for (int i = 0; i < n; ++i) {
        const int64_t pt = base + i;
        int32_t * ri = ridx.data() + (size_t) i * NQ * W;
        float *   mk = mask.data() + (size_t) i * NQ * W;
        for (int64_t j = 0; j < base; ++j) {
            const int64_t dist = pt - j;
            if (dist < 0 || (!full && dist >= 1024)) continue;
            ri[j] = (int32_t) std::min<int64_t>(dist, 1024);
            mk[j] = 0.0f;
        }
        for (int t = 0; t <= i; ++t) {
            const int64_t dist = i - t;
            ri[C + t] = (int32_t) dist;
            mk[C + t] = 0.0f;
        }
        if (NQ > 1) {
            for (int q = 1; q < NQ; ++q) {
                memcpy(ri + (size_t) q * W, ri, sizeof(int32_t) * W);
                memcpy(mk + (size_t) q * W, mk, sizeof(float) * W);
            }
        }
    }
}

void inkling2::build_mask_decode(std::vector<int32_t> & ridx, std::vector<float> & mask, int32_t pos, bool full) {
    const int W = C + 1;
    ridx.assign(W, 1024);
    mask.assign(W, -INFINITY);
    for (int64_t j = 0; j < n_past; ++j) {
        const int64_t dist = pos - j;
        if (dist < 0 || (!full && dist >= 1024)) continue;
        ridx[j] = (int32_t) std::min<int64_t>(dist, 1024);
        mask[j] = 0.0f;
    }
    ridx[C] = 0;
    mask[C] = 0.0f;
}

void inkling2::commit(bool pre, int n) {
    auto copy_kv = [&](axera_cmm & bank, const std::string & name, int nslot, bool is_key) {
        (void) is_key;
        auto * src = (const ggml_bf16_t *) (pre ? pout : dout).at(name).c->vir;
        auto * dst = (ggml_bf16_t *) bank->vir;
        for (int s = 0; s < nslot; ++s) {
            for (int h = 0; h < NKV; ++h) {
                memcpy(dst + ((size_t) s * KVCH + h * C + n_past) * HD,
                       src + ((size_t) s * NKV + h) * n * HD, (size_t) n * HD * sizeof(ggml_bf16_t));
            }
        }
        bank->flush(bank->vir, bank->size);
    };
    copy_kv(kvk_l, "new_keys_l", 40, true);
    copy_kv(kvv_l, "new_values_l", 40, false);
    copy_kv(kvk_g, "new_keys_g", 8, true);
    copy_kv(kvv_g, "new_values_g", 8, false);

    if (pre) {
        // the prefill kernel's tails are the 3-row history itself
        auto move_state = [&](axera_cmm & dst, const std::string & name) {
            iobank & src = pout.at(name);
            GGML_ASSERT(src.c->size == dst->size);
            memcpy(dst->vir, src.c->vir, dst->size);
            dst->flush(dst->vir, dst->size);
        };
        move_state(kc, "new_kconv");
        move_state(vc, "new_vconv");
        move_state(pca, "new_paconv");
        move_state(pmc, "new_pmconv");
    } else {
        // the decode tail is [4][C] with row 0 stale; rows 1..3 are the history
        auto move_state = [&](axera_cmm & dst3, axera_cmm & src4, const std::string & name, size_t row_b) {
            iobank & src = dout.at(name);
            GGML_ASSERT(src.c->size == src4->size);
            for (int s = 0; s < SLOTS; ++s) {
                memcpy((uint8_t *) dst3->vir + (size_t) s * 3 * row_b,
                       (const uint8_t *) src.c->vir + (size_t) s * 4 * row_b + row_b, 3 * row_b);
            }
            dst3->flush(dst3->vir, dst3->size);
        };
        move_state(kc, kc4, "new_kconv", 1024 * sizeof(float));
        move_state(vc, vc4, "new_vconv", 1024 * sizeof(float));
        move_state(pca, pca4, "new_paconv", (size_t) H * sizeof(float));
        move_state(pmc, pmc4, "new_pmconv", (size_t) H * sizeof(float));
    }
    n_past += n;
}

bool inkling2::run_prefill(const uint16_t * x, const int32_t * pos, int n) {
    GGML_ASSERT(n == WP);
    memcpy(pin["x"].c->vir, x, (size_t) n * H * sizeof(ggml_bf16_t));
    pin["x"].c->flush(pin["x"].c->vir, (size_t) n * H * sizeof(ggml_bf16_t));
    memcpy(pin["pos"].c->vir, pos, (size_t) n * sizeof(int32_t));
    pin["pos"].c->flush(pin["pos"].c->vir, (size_t) n * sizeof(int32_t));

    std::vector<int32_t> rl, rg;
    std::vector<float>   ml, mg;
    build_mask_prefill(rl, ml, pos, n, false);
    build_mask_prefill(rg, mg, pos, n, true);
    GGML_ASSERT((size_t) rl.size() * sizeof(int32_t) == pin["rel_idx_l"].c->size);
    GGML_ASSERT((size_t) ml.size() * sizeof(float) == pin["mask_l"].c->size);
    memcpy(pin["rel_idx_l"].c->vir, rl.data(), rl.size() * sizeof(int32_t));
    memcpy(pin["rel_idx_g"].c->vir, rg.data(), rg.size() * sizeof(int32_t));
    memcpy(pin["mask_l"].c->vir, ml.data(), ml.size() * sizeof(float));
    memcpy(pin["mask_g"].c->vir, mg.data(), mg.size() * sizeof(float));
    for (const char * s : {"rel_idx_l", "rel_idx_g", "mask_l", "mask_g"}) {
        pin[s].c->flush(pin[s].c->vir, pin[s].c->size);
    }

    std::vector<std::string> inn, outn;
    collect_names(pk->desc->in, inn);
    collect_names(pk->desc->out, outn);
    run_call(*pk, pin, pout, inn, outn, "inkling2 prefill");
    commit(true, n);

    // logits of this call live in pout["logits"] for rows 0..n-1
    return true;
}

bool inkling2::run_decode_row(const uint16_t * x, int32_t pos) {
    memcpy(din["x"].c->vir, x, (size_t) H * sizeof(ggml_bf16_t));
    din["x"].c->flush(din["x"].c->vir, (size_t) H * sizeof(ggml_bf16_t));
    memcpy(din["pos"].c->vir, &pos, sizeof(int32_t));
    din["pos"].c->flush(din["pos"].c->vir, sizeof(int32_t));

    std::vector<int32_t> rl, rg;
    std::vector<float>   ml, mg;
    build_mask_decode(rl, ml, pos, false);
    build_mask_decode(rg, mg, pos, true);
    GGML_ASSERT((size_t) rl.size() * sizeof(int32_t) == din["rel_idx_l"].c->size);
    GGML_ASSERT((size_t) ml.size() * sizeof(float) == din["mask_l"].c->size);
    memcpy(din["rel_idx_l"].c->vir, rl.data(), rl.size() * sizeof(int32_t));
    memcpy(din["rel_idx_g"].c->vir, rg.data(), rg.size() * sizeof(int32_t));
    memcpy(din["mask_l"].c->vir, ml.data(), ml.size() * sizeof(float));
    memcpy(din["mask_g"].c->vir, mg.data(), mg.size() * sizeof(float));
    for (const char * s : {"rel_idx_l", "rel_idx_g", "mask_l", "mask_g"}) {
        din[s].c->flush(din[s].c->vir, din[s].c->size);
    }
    auto sync4 = [&](axera_cmm & dst4, axera_cmm & src3, size_t row_b) {
        for (int s = 0; s < SLOTS; ++s) {
            memcpy((uint8_t *) dst4->vir + (size_t) s * 4 * row_b + row_b,
                   (const uint8_t *) src3->vir + (size_t) s * 3 * row_b, 3 * row_b);
        }
        dst4->flush(dst4->vir, dst4->size);
    };
    sync4(kc4, kc, 1024 * sizeof(float));
    sync4(vc4, vc, 1024 * sizeof(float));
    sync4(pca4, pca, (size_t) H * sizeof(float));
    sync4(pmc4, pmc, (size_t) H * sizeof(float));
    std::vector<std::string> inn, outn;
    collect_names(dk->desc->in, inn);
    collect_names(dk->desc->out, outn);
    run_call(*dk, din, dout, inn, outn, "inkling2 decode");
    commit(false, 1);
    return true;
}

void inkling2::publish_logits(ggml_cgraph * g, ggml_tensor * logits, int T, int first_row) {
    GGML_ASSERT(logits && logits->type == GGML_TYPE_F32 && logits->ne[0] == VOCAB);
    const int n_out = (int) logits->ne[1];
    if (n_out == 0) return;
    const float * src = (const float *) pout.at("logits").c->vir;
    pout.at("logits").c->invalidate(pout.at("logits").c->vir, (size_t) T * VOCAB * sizeof(float));
    ggml_tensor * out_ids = names.find("out_ids");
    if (!out_ids || out_ids->type != GGML_TYPE_I32 || out_ids->ne[0] != n_out || n_out >= T) {
        out_ids = nullptr;
        for (int i = 0; i < g->n_nodes; ++i) {
            ggml_tensor * n = g->nodes[i];
            if (n->op == GGML_OP_GET_ROWS && n->src[1] && n->src[1]->type == GGML_TYPE_I32 &&
                n->src[1]->ne[0] == n_out && n->src[0]->ne[1] == T) {
                out_ids = n->src[1];
            }
        }
        if (n_out >= T) out_ids = nullptr;
    }
    GGML_ASSERT(out_ids || n_out == T);
    auto * dctx = ggml_backend_buffer_is_axera(logits->buffer)
                      ? (ggml_backend_axera_buffer_context *) logits->buffer->context : nullptr;
    for (int r = 0; r < n_out; ++r) {
        const int row = out_ids ? ((const int32_t *) out_ids->data)[r] : r;
        GGML_ASSERT(row >= first_row && row < first_row + T);
        float * o = (float *) ((char *) logits->data + (size_t) r * logits->nb[1]);
        memcpy(o, src + (size_t) (row - first_row) * VOCAB, VOCAB * sizeof(float));
        if (dctx) {
            dctx->flush(o, VOCAB * sizeof(float));
            dctx->clean_p     = o;
            dctx->clean_n     = VOCAB * sizeof(float);
            dctx->clean_epoch = axera_graph_epoch;
        }
    }
}

ggml_status inkling2::compute(ggml_cgraph * g) {
    const auto & t = names.of(g);
    ggml_tensor * norm0  = names.find("attn_norm-0");
    ggml_tensor * logits = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("%s: a graph of %d nodes that is not the whole model (-ngl 99)\n", tag.c_str(), g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = llm_input(g, norm0, true);
    const int T = (int) x0->ne[1];
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data) {
        init(t);
        bound = w0->data;
    }
    if (T < 1 || T > WP) {
        GGML_LOG_ERROR("%s: %d rows in one call (the prefill kernel takes at most %d, keep -ub <= %d)\n",
                       tag.c_str(), T, WP, WP);
        return GGML_STATUS_FAILED;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, C, WP);
    const int64_t c0 = kv.cell_of(0);
    if (c0 != n_past) {
        if (c0 == 0) {
            clear_state();
        } else {
            GGML_LOG_ERROR("%s: batch at cell %lld but the state is at %lld (no context shift support)\n",
                           tag.c_str(), (long long) c0, (long long) n_past);
            return GGML_STATUS_FAILED;
        }
    }
    llm_positions(inp_pos, T, pbuf);
    for (int i = 0; i < T; ++i) {
        GGML_ASSERT(pbuf[(size_t) i * 4] == (int32_t) (c0 + i));
    }

    std::vector<ggml_bf16_t> xb((size_t) T * H);
    axera_host_pool().run(T, [&](int64_t r) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]),
                              xb.data() + (size_t) r * H, H);
    });

    if (T == WP) {
        if (!run_prefill((const uint16_t *) xb.data(), pbuf.data(), T)) return GGML_STATUS_FAILED;
        publish_logits(g, logits, T, 0);
    } else {
        // a short batch replays through the decode kernel, row by row
        const int n_out = (int) logits->ne[1];
        for (int i = 0; i < T; ++i) {
            if (!run_decode_row((const uint16_t *) (xb.data() + (size_t) i * H), pbuf[(size_t) i * 4])) {
                return GGML_STATUS_FAILED;
            }
            const bool wanted = n_out > 0 && i >= T - n_out;
            if (wanted) {
                const float * src = (const float *) dout.at("logits").c->vir;
                dout.at("logits").c->invalidate(dout.at("logits").c->vir, (size_t) VOCAB * sizeof(float));
                auto * dctx = ggml_backend_buffer_is_axera(logits->buffer)
                                  ? (ggml_backend_axera_buffer_context *) logits->buffer->context : nullptr;
                float * o = (float *) ((char *) logits->data + (size_t) (i - (T - n_out)) * logits->nb[1]);
                memcpy(o, src, (size_t) VOCAB * sizeof(float));
                if (dctx) {
                    dctx->flush(o, VOCAB * sizeof(float));
                    dctx->clean_p     = o;
                    dctx->clean_n     = VOCAB * sizeof(float);
                    dctx->clean_epoch = axera_graph_epoch;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

size_t inkling2::state_size(uint32_t flags) const {
    (void) flags;
    if (!inited) return 0;
    return sizeof(int64_t) + kvk_l->size + kvv_l->size + kvk_g->size + kvv_g->size + kc->size + vc->size +
           pca->size + pmc->size + kc4->size + vc4->size + pca4->size + pmc4->size;
}

size_t inkling2::state_write(void * dst, size_t size, uint32_t flags) {
    (void) flags;
    if (!inited) return 0;
    const size_t need = state_size(0);
    if (size < need) return 0;
    uint8_t * d = (uint8_t *) dst;
    memcpy(d, &n_past, sizeof(int64_t));
    d += sizeof(int64_t);
    auto put = [&](axera_cmm & c) {
        c->invalidate(c->vir, c->size);
        memcpy(d, c->vir, c->size);
        d += c->size;
    };
    put(kvk_l); put(kvv_l); put(kvk_g); put(kvv_g); put(kc); put(vc); put(pca); put(pmc);
    put(kc4); put(vc4); put(pca4); put(pmc4);
    return need;
}

size_t inkling2::state_read(const void * src, size_t size, uint32_t flags) {
    (void) flags;
    if (!inited) return 0;
    const size_t need = state_size(0);
    if (size < need) return 0;
    const uint8_t * s = (const uint8_t *) src;
    memcpy(&n_past, s, sizeof(int64_t));
    s += sizeof(int64_t);
    auto get = [&](axera_cmm & c) {
        memcpy(c->vir, s, c->size);
        c->flush(c->vir, c->size);
        s += c->size;
    };
    get(kvk_l); get(kvv_l); get(kvk_g); get(kvv_g); get(kc); get(vc); get(pca); get(pmc);
    get(kc4); get(vc4); get(pca4); get(pmc4);
    return need;
}

}  // namespace

std::shared_ptr<model> make_inkling2(const manifest & m) {
    return std::make_shared<inkling2>(m);
}

}  // namespace axera
