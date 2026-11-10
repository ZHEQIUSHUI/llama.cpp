// Axera NPU backend: dense Qwen3.8 (3 gated-delta-net layers + 1 gated attention layer per block, dense MLP).
//
// The manifest's kernels are the upstream fused block functions, split by IO size only: one axmodel holds one
// 4-layer block (all of its weights and its attention layer's KV are separate kernel inputs, each below the runtime's
// 2 GiB IO limit). One call runs one block; the host chains L / 4 calls per page (prefill, 256 rows) or per token
// (decode, 1 row).
//
// Layout: prefill role, x [256, H], the b0..b3 bank inputs (one layer each), b3_kv [pages, 2, 256, 4, 256] bf16 and
// the three delta-net layers' state / conv-state banks; decode role, the same banks and state at x [1, H]. The kernel
// loops its four layers internally; the adapter restarts a partial page from the state saved at the page's first cell
// (whole-page history only), keeping the page's inputs so a batch that starts inside it can replay them.

#include "../model.h"
#include "../../core/params.h"
#include "../../ops/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <regex>
#include <set>

namespace axera {
namespace {

constexpr int PG = 256;           // cells per page = prefill rows per call
constexpr int KH = 16, VH = 48;   // key / value heads of a delta-net layer
constexpr int REP = VH / KH;
constexpr int ROT = 64;           // 256 * 0.25 rotary channels

struct call {
    const kernel *                     k = nullptr;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
};

struct state_header_dense {
    char     magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t has_target;
    int64_t  state_cell;
    int64_t  snap_cell;
    int64_t  ksnap_cell;
    uint32_t snap_prompt;
    uint32_t ncall;
    uint32_t nkv_pages;
    uint64_t kv_page_b;
    uint64_t kv_n;
    uint64_t st_b;
    uint64_t cq_b;
    uint64_t ck_b;
    uint64_t cv_b;
    uint64_t snap_n;
    uint64_t ksnap_n;
    uint64_t page_x_n;
    uint64_t page_p_n;
    uint64_t kpage_x_n;
    uint64_t kpage_p_n;
    uint32_t partial;
    uint32_t reserved;
};

inline int tiled(int g) {
    return (g % REP) * KH + g / REP;
}

// GGUF MXFP4 rows of an [N, K] weight in the kernel layout: packed codes [K/2] then a scale row [K/32]
struct mx4 {
    const ggml_tensor * w;
    axera_cmm           q, s;
    int64_t             N = 0, K = 0;
    explicit mx4(const ggml_tensor * w) : w(w), N(w->ne[1]), K(w->ne[0]) {
        if (w->type != GGML_TYPE_MXFP4) {
            axera_quantize_mxfp4(w, q, s);
        }
    }
    void row(int64_t r, bool scales, uint8_t * dst) const {
        if (q) {
            memcpy(dst, scales ? (const uint8_t *) s->vir + r * (K / 32) : (const uint8_t *) q->vir + r * (K / 2),
                   scales ? K / 32 : K / 2);
            return;
        }
        thread_local std::vector<uint8_t> other;
        other.resize(scales ? K / 2 : K / 32);
        const uint8_t * src = (const uint8_t *) w->data + r * w->nb[1];
        axera_mxfp4_rows(src, 1, K, scales ? other.data() : dst, scales ? dst : other.data());
    }
};

std::vector<float> f32_of(const ggml_tensor * w) {
    std::vector<float> v((size_t) ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(v.data(), w->data, v.size() * sizeof(float));
    } else {
        axera_for_rows_f32(w, [&](int64_t r, const float * row) { memcpy(&v[(size_t) r * w->ne[0]], row, w->ne[0] * sizeof(float)); });
    }
    return v;
}

class dense_blocks final : public model {
  public:
    explicit dense_blocks(const manifest & m) : man(m) {}
    ggml_status compute(ggml_cgraph * g) override;
    size_t       state_size(uint32_t flags) const override;
    size_t       state_write(void * dst, size_t size, uint32_t flags) override;
    size_t       state_read(const void * src, size_t size, uint32_t flags) override;

  private:
    const manifest & man;
    graph_names      names;
    const void *     bound = nullptr;
    int              L = 0, H = 0, NB = 0;  // layers, hidden, blocks (kernel calls per row batch)
    int              kv_pages = 0;
    int64_t          kv_cells = 0;

    std::vector<axera_cmm> KVC;          // one 1-layer KV pool per block
    size_t                 kv_b = 0;     // bytes of one block's KV pool
    axera_cmm              ST, CQ, CK, CV;
    size_t                 st_b = 0, cq_b = 0, cv_b = 0;
    axera_cmm              INDPTR, INDICES, LASTLEN, CLOC, VALID;
    axera_cmm              A, OUT[2], X, POS;
    std::map<std::string, axera_cmm> banks;  // one buffer per bank name: NB block copies
    std::vector<call>                pf, dec;

    std::vector<uint8_t>     snap;
    int64_t                  snap_cell = -1;
    std::vector<ggml_bf16_t> page_x;
    std::vector<int32_t>     page_p;
    std::vector<uint8_t>     ksnap;
    int64_t                  ksnap_cell  = -1;
    bool                     snap_prompt = false;
    std::vector<ggml_bf16_t> kpage_x;
    std::vector<int32_t>     kpage_p;
    std::vector<float>       h_buf;
    output_head              head;

    void init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_);
    axera_cmm bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d, const ggml_tensor * rope_node);
    void zero_state();
    void save_state(int64_t cell);
    void load_state();
    void swap_kept();
    bool restart_point(int64_t s0, int k);
    void prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h, int out0);
    int64_t state_cell() const { return snap_cell < 0 ? 0 : snap_cell + (int64_t) (page_x.size() / std::max(1, H)); }

    const io_desc & in_of(const kernel & k, const std::string & n) const {
        for (const auto & d : k.desc->in) {
            if (d.name == n) {
                return d;
            }
        }
        GGML_ABORT("axera dense_blocks: %s has no IO %s\n", k.desc->file.c_str(), n.c_str());
    }
    int state_phase(const std::string & n) const {
        if (n.size() < 5 || n[0] != 'b' || n[1] < '0' || n[1] > '2') {
            return -1;
        }
        if (n.find("_state_in") != std::string::npos || n.find("_conv_state_in") != std::string::npos) {
            return n[1] - '0';
        }
        return -1;
    }
};

// Fill one bank name for all NB blocks from the GGUF. A block's four layers are 4*b + phase.
axera_cmm dense_blocks::bank(const std::map<std::string, ggml_tensor *> & t, const io_desc & d,
                             const ggml_tensor * rope_node) {
    static const std::regex re("b([0-3])_(attn|mlp)_(.+)");
    std::smatch             mt;
    if (!std::regex_match(d.name, mt, re)) {
        GGML_ABORT("axera dense_blocks: unexpected input %s\n", d.name.c_str());
    }
    const int         phase = std::stoi(mt[1]);
    const bool        mlp   = mt[2] == "mlp";
    const std::string n     = mt[3];
    const size_t      bb    = d.bytes;  // one block's bytes (the kernels hold one block)
    axera_cmm         c     = axera_cmm_alloc(bb * (size_t) NB);
    std::vector<const ggml_tensor *> read;
    auto T = [&](int il, const char * g) -> ggml_tensor * {
        const std::string nm = "blk." + std::to_string(il) + "." + g;
        if (!t.count(nm)) {
            GGML_ABORT("axera dense_blocks: %s not in the graph\n", nm.c_str());
        }
        read.push_back(t.at(nm));
        return t.at(nm);
    };
    for (int b = 0; b < NB; ++b) {
        const int il    = 4 * b + phase;
        uint8_t * dst   = (uint8_t *) c->vir + (size_t) b * bb;
        auto put_f32 = [&](const std::vector<float> & v, float scale = 1.0f) {
            GGML_ASSERT(v.size() * sizeof(float) == bb || v.size() * sizeof(ggml_bf16_t) == bb);
            for (size_t i = 0; i < v.size(); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v[i] * scale;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v[i] * scale);
                }
            }
        };
        auto fill = [&](float v) {
            for (size_t i = 0; i < bb / (d.dtype == 3 ? 4 : 2); ++i) {
                if (d.dtype == 3) {
                    ((float *) dst)[i] = v;
                } else {
                    ((ggml_bf16_t *) dst)[i] = ggml_fp32_to_bf16(v);
                }
            }
        };
        auto rows = [&](const mx4 & w, int64_t r0, int64_t nr, bool scales, int64_t vrow = -1, int64_t hr = 0) {
            const size_t rb = scales ? w.K / 32 : w.K / 2;
            GGML_ASSERT((size_t) nr * rb == bb);
            axera_host_pool().run(nr, [&](int64_t r) {
                int64_t src = r0 + r;
                if (vrow >= 0 && src >= vrow) {
                    const int64_t j = src - vrow;
                    src             = vrow + tiled((int) (j / hr)) * hr + j % hr;
                }
                w.row(src, scales, dst + r * rb);
            });
        };
        auto cols = [&](const mx4 & w, bool scales) {
            const size_t rb = scales ? w.K / 32 : w.K / 2, cb = rb / VH;
            GGML_ASSERT((size_t) w.N * rb == bb);
            axera_host_pool().run(w.N, [&](int64_t r) {
                thread_local std::vector<uint8_t> src;
                src.resize(rb);
                w.row(r, scales, src.data());
                for (int g = 0; g < VH; ++g) {
                    memcpy(dst + r * rb + g * cb, src.data() + tiled(g) * cb, cb);
                }
            });
        };
        auto vec = [&](const char * g, bool heads, float scale = 1.0f) {
            std::vector<float> v = f32_of(T(il, g));
            if (heads) {
                std::vector<float> u(v.size());
                for (int h = 0; h < VH; ++h) {
                    u[h] = v[tiled(h)];
                }
                v = u;
            }
            put_f32(v, scale);
        };
        if (mlp) {
            static const std::map<std::string, const char *> W = { { "1", "ffn_gate.weight" }, { "2", "ffn_up.weight" },
                                                                    { "3", "ffn_down.weight" } };
            if (n == "w_norm") {
                vec("post_attention_norm.weight", false);
            } else if (n == "input_permutation") {
                memset(dst, 0, bb);
            } else if (n.rfind("w_scale2_", 0) == 0) {
                fill(1.0f);
            } else if (n.rfind("w_zp_scale", 0) == 0) {
                fill(0.0f);
            } else if (n.size() == 2 && W.count(n.substr(1))) {
                const mx4 w(T(il, W.at(n.substr(1))));
                rows(w, 0, w.N, false);
            } else if (n.rfind("w_scale", 0) == 0 && W.count(n.substr(7))) {
                const mx4 w(T(il, W.at(n.substr(7))));
                rows(w, 0, w.N, true);
            } else {
                GGML_ABORT("axera dense_blocks: unexpected MLP input %s\n", d.name.c_str());
            }
            continue;
        }
        if (phase == 3) {
            static const std::map<std::string, const char *> W = { { "q", "attn_q.weight" }, { "k", "attn_k.weight" },
                                                                    { "v", "attn_v.weight" }, { "o", "attn_output.weight" } };
            const std::string p = n.substr(n.rfind('_') + 1);
            if (n == "w_norm_attn") {
                vec("attn_norm.weight", false);
            } else if (n == "w_norm_q") {
                vec("attn_q_norm.weight", false, 1.0f / std::sqrt((float) d.shape.back()));
            } else if (n == "w_norm_k") {
                vec("attn_k_norm.weight", false);
            } else if (n == "cos_sin_cache") {
                // [max_position, rotary_dim] bf16: the first half is cos, the second sin. Text and image share the
                // same frequency table; the kernel gathers each token's M-RoPE sections from it.
                const int64_t P = d.shape[1];
                GGML_ASSERT(d.dtype == 9 && d.shape[2] == ROT && bb == (size_t) P * ROT * 2);
                std::vector<float> inv(ROT / 2);
                for (int i = 0; i < ROT / 2; ++i) {
                    inv[i] = 1.0f / std::pow(10000000.0f, (float) (2 * i) / ROT);
                }
                for (int64_t q = 0; q < P; ++q) {
                    for (int i = 0; i < ROT / 2; ++i) {
                        const float th = (float) q * inv[i];
                        ((ggml_bf16_t *) dst)[q * ROT + i]       = ggml_fp32_to_bf16(std::cos(th));
                        ((ggml_bf16_t *) dst)[q * ROT + ROT / 2 + i] = ggml_fp32_to_bf16(std::sin(th));
                    }
                }
            } else if (n.rfind("w_scale2_", 0) == 0) {
                fill(1.0f);
            } else if (n.rfind("w_zeropoint_", 0) == 0) {
                fill(0.0f);
            } else if (W.count(p) && (n == "w_" + p || n == "w_scale_" + p)) {
                const mx4 w(T(il, W.at(p)));
                rows(w, 0, w.N, n != "w_" + p);
            } else {
                GGML_ABORT("axera dense_blocks: unexpected attention input %s\n", d.name.c_str());
            }
            continue;
        }
        if (n == "input_norm_w") {
            vec("attn_norm.weight", false);
        } else if (n == "o_norm_w") {
            vec("ssm_norm.weight", false);
        } else if (n == "a_log") {
            std::vector<float> a = f32_of(T(il, "ssm_a")), u(VH);
            for (int h = 0; h < VH; ++h) {
                u[h] = logf(-a[tiled(h)]);
            }
            put_f32(u);
        } else if (n == "dt_bias") {
            vec("ssm_dt.bias", true);
        } else if (n == "w_q_conv" || n == "w_k_conv" || n == "w_v_conv") {
            const ggml_tensor * w    = T(il, "ssm_conv1d.weight");
            const std::vector<float> v = f32_of(w);
            const int64_t tp = w->ne[0], C = d.shape[1], v0 = w->ne[1] - C;
            const int64_t c0 = n[2] == 'q' ? 0 : n[2] == 'k' ? C : v0;
            std::vector<float> u((size_t) C * tp);
            for (int64_t r = 0; r < C; ++r) {
                const int64_t hr  = C / VH;
                const int64_t src = n[2] == 'v' ? v0 + tiled((int) (r / hr)) * hr + r % hr : c0 + r;
                memcpy(&u[(size_t) r * tp], &v[(size_t) src * tp], tp * sizeof(float));
            }
            put_f32(u);
        } else if (n.rfind("w_", 0) == 0 && n.size() >= 3) {
            const char        p      = n[2];
            const std::string suffix = n.substr(3);
            if (suffix == "_scale2") {
                fill(1.0f);
                continue;
            }
            const bool scales = suffix == "_scale";
            GGML_ASSERT(suffix.empty() || scales);
            if (p == 'q' || p == 'k' || p == 'v') {
                const mx4     w(T(il, "attn_qkv.weight"));
                const int64_t nr = d.shape[1];
                if (p == 'v') {
                    rows(w, w.N - nr, nr, scales, w.N - nr, nr / VH);
                } else {
                    rows(w, p == 'q' ? 0 : nr, nr, scales);
                }
            } else if (p == 'z') {
                const mx4 w(T(il, "attn_gate.weight"));
                rows(w, 0, w.N, scales, 0, w.N / VH);
            } else if (p == 'a' || p == 'b') {
                const mx4 w(T(il, p == 'a' ? "ssm_alpha.weight" : "ssm_beta.weight"));
                rows(w, 0, w.N, scales, 0, 1);
            } else if (p == 'o') {
                const mx4 w(T(il, "ssm_out.weight"));
                cols(w, scales);
            } else {
                GGML_ABORT("axera dense_blocks: unexpected delta-net input %s\n", d.name.c_str());
            }
        } else {
            GGML_ABORT("axera dense_blocks: unexpected delta-net input %s\n", d.name.c_str());
        }
    }
    c->flush();
    for (const ggml_tensor * w : read) {
        axera_drop_mapped(w);
    }
    return c;
}

void dense_blocks::init(const std::map<std::string, ggml_tensor *> & t, const ggml_tensor * rope_node, int L_, int H_) {
    L  = L_;
    H  = H_;
    NB = L / 4;
    const kernel & pk = man.get("prefill");
    const kernel & dk = man.get("decode");
    const io_desc & pst = in_of(pk, "b0_attn_state_in");
    const io_desc & pcq = in_of(pk, "b0_attn_q_conv_state_in");
    const io_desc & pcv = in_of(pk, "b0_attn_v_conv_state_in");
    const io_desc & pkv = in_of(pk, "b3_kv");
    st_b = pst.bytes;
    cq_b = pcq.bytes;
    cv_b = pcv.bytes;
    kv_b = pkv.bytes;
    GGML_ASSERT(pkv.shape[0] == 1 && pkv.shape[2] == PG && (int64_t) man.context == (int64_t) pkv.shape[1] * PG);
    kv_pages = (int) pkv.shape[1];
    kv_cells = man.context;
    GGML_ASSERT(in_of(dk, "b3_kv").bytes == kv_b && in_of(dk, "b0_attn_state_in").bytes == st_b);

    axera_timer tl("init: dense_blocks");
    KVC.resize(NB);
    for (int b = 0; b < NB; ++b) {
        KVC[b] = axera_cmm_alloc(kv_b);
    }
    ST = axera_cmm_alloc(st_b * (size_t) (NB * 3));
    CQ = axera_cmm_alloc(cq_b * (size_t) (NB * 3));
    CK = axera_cmm_alloc(cq_b * (size_t) (NB * 3));
    CV = axera_cmm_alloc(cv_b * (size_t) (NB * 3));
    snap.assign(ST->size + CQ->size + CK->size + CV->size, 0);

    // the weight banks: every block's copy of each input, except the runtime buffers and the shared RoPE table
    static const std::set<std::string> runtime = { "x", "position_ids", "indptr", "indices", "last_page_len",
                                                   "cache_location", "valid_tokens", "b3_kv", "b0_attn_state_in",
                                                   "b1_attn_state_in", "b2_attn_state_in", "b0_attn_q_conv_state_in",
                                                   "b0_attn_k_conv_state_in", "b0_attn_v_conv_state_in",
                                                   "b1_attn_q_conv_state_in", "b1_attn_k_conv_state_in",
                                                   "b1_attn_v_conv_state_in", "b2_attn_q_conv_state_in",
                                                   "b2_attn_k_conv_state_in", "b2_attn_v_conv_state_in" };
    for (const auto & d : pk.desc->in) {
        if (runtime.count(d.name)) {
            continue;
        }
        const bool shared = d.name == "b3_attn_cos_sin_cache";
        axera_cmm  c      = bank(t, d, rope_node);
        if (shared) {
            axera_cmm one = axera_cmm_alloc(d.bytes);
            memcpy(one->vir, c->vir, d.bytes);
            one->flush();
            c = std::move(one);
        }
        banks[d.name] = std::move(c);
        GGML_ASSERT(c->size == (size_t) (shared ? d.bytes : d.bytes * NB));
    }

    // the four phase banks above are one layer each: the state/conv-state pools hold every block's three layers
    INDPTR  = axera_cmm_alloc(2 * sizeof(int32_t));
    INDICES = axera_cmm_alloc(pk.in_bytes("indices"));
    LASTLEN = axera_cmm_alloc(sizeof(int32_t));
    CLOC    = axera_cmm_alloc(2 * sizeof(int32_t));
    VALID   = axera_cmm_alloc(sizeof(int32_t));
    GGML_ASSERT(pk.in_bytes("indices") == (size_t) kv_pages * sizeof(int32_t));
    for (int i = 0; i < kv_pages; ++i) {
        ((int32_t *) INDICES->vir)[i] = i;
    }
    INDICES->flush();

    X    = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    OUT[0] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    OUT[1] = axera_cmm_alloc((size_t) PG * H * sizeof(ggml_bf16_t));
    A    = axera_cmm_alloc((size_t) H * sizeof(ggml_bf16_t));

    auto bind = [&](const kernel & k, std::vector<call> & calls, bool decode) {
        calls.assign(NB, {});
        for (int b = 0; b < NB; ++b) {
            call & c = calls[b];
            c.k      = &k;
            for (const auto & d : k.desc->in) {
                const std::string & n = d.name;
                if (n == "x") {
                    c.in.push_back(axera_io(decode ? A : (b == 0 ? X : OUT[(b - 1) & 1]), d.bytes));
                } else if (n == "position_ids") {
                    c.in.push_back(axera_io(POS, d.bytes));
                } else if (n == "indptr") {
                    c.in.push_back(axera_io(INDPTR, d.bytes));
                } else if (n == "indices") {
                    c.in.push_back(axera_io(INDICES, d.bytes));
                } else if (n == "last_page_len") {
                    c.in.push_back(axera_io(LASTLEN, d.bytes));
                } else if (n == "cache_location") {
                    c.in.push_back(axera_io(CLOC, d.bytes));
                } else if (n == "valid_tokens") {
                    c.in.push_back(axera_io(VALID, d.bytes));
                } else if (n == "b3_kv") {
                    GGML_ASSERT(d.bytes == kv_b);
                    c.in.push_back(axera_io(KVC[b], (uint32_t) kv_b));
                } else if (state_phase(n) >= 0) {
                    const int p = state_phase(n);
                    const axera_cmm * pool = n.find("q_conv_state_in") != std::string::npos ? &CQ :
                                             n.find("k_conv_state_in") != std::string::npos ? &CK :
                                             n.find("v_conv_state_in") != std::string::npos ? &CV : &ST;
                    const size_t part = pool == &ST ? st_b : pool == &CV ? cv_b : cq_b;
                    const size_t off  = (size_t) (3 * b + p) * part;
                    GGML_ASSERT(off + part <= (*pool)->size);
                    c.in.push_back(axera_io(pool->get(), (char *) (*pool)->vir + off, d.bytes));
                } else {
                    auto it = banks.find(n);
                    if (it == banks.end()) {
                        GGML_ABORT("axera dense_blocks: %s: unexpected input %s\n", k.desc->role.c_str(), n.c_str());
                    }
                    const size_t off = it->second->size == (size_t) d.bytes ? 0 : (size_t) b * d.bytes;
                    c.in.push_back(axera_io(it->second.get(), (char *) it->second->vir + off, d.bytes));
                }
            }
            for (const auto & d : k.desc->out) {
                GGML_ASSERT(d.name == "output");
                c.out.push_back(axera_io(decode ? A : OUT[b & 1], d.bytes));
            }
        }
    };
    const io_desc & ppos = in_of(pk, "position_ids");
    POS = axera_cmm_alloc(ppos.bytes);
    GGML_ASSERT(ppos.bytes == 3 * (size_t) PG * sizeof(int32_t));
    bind(pk, pf, false);
    bind(dk, dec, true);
    GGML_LOG_INFO("axera dense_blocks: %d layers, %d blocks, banks %.1f GiB, state %.2f GiB, %lld cells\n", L, NB,
                  [&] {
                      size_t s = 0;
                      for (auto & [n, b] : banks) s += b->size;
                      return s / 1073741824.0;
                  }(),
                  (ST->size + CQ->size + CK->size + CV->size + KVC[0]->size * NB) / 1073741824.0, (long long) kv_cells);
    snap_cell = -1;
    page_x.clear();
    page_p.clear();
    ksnap_cell = -1;
    snap_prompt = false;
}

void dense_blocks::zero_state() {
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memset((*p)->vir, 0, (*p)->size);
        (*p)->flush();
    }
    snap_cell  = -1;
    ksnap_cell = -1;
    snap_prompt = false;
    page_x.clear();
    page_p.clear();
    snap.clear();
    snap.resize(ST->size + CQ->size + CK->size + CV->size, 0);
}

void dense_blocks::swap_kept() {
    std::swap(snap, ksnap);
    std::swap(snap_cell, ksnap_cell);
    std::swap(page_x, kpage_x);
    std::swap(page_p, kpage_p);
}

bool dense_blocks::restart_point(int64_t s0, int k) {
    if (snap_cell == s0 && (int) (page_x.size() / H) >= k) {
        return true;
    }
    if (ksnap_cell == s0 && (int) (kpage_x.size() / H) >= k) {
        swap_kept();
        snap_prompt = true;
        return true;
    }
    return false;
}

void dense_blocks::save_state(int64_t cell) {
    GGML_ASSERT(cell % PG == 0);
    axera_timer tl("state save");
    if (snap_prompt && snap_cell >= 0) {
        swap_kept();
    }
    snap_prompt = false;
    snap.resize(ST->size + CQ->size + CK->size + CV->size);
    size_t o = 0;
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        (*p)->invalidate((*p)->vir, (*p)->size);
        memcpy(snap.data() + o, (*p)->vir, (*p)->size);
        o += (*p)->size;
    }
    snap_cell = cell;
    page_x.clear();
    page_p.clear();
}

void dense_blocks::load_state() {
    axera_timer tl("state load");
    size_t o = 0;
    for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
        memcpy((*p)->vir, snap.data() + o, (*p)->size);
        (*p)->flush();
        o += (*p)->size;
    }
}

void dense_blocks::prefill(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h,
                           int out0) {
    GGML_ASSERT(c0 % PG == 0);
    GGML_ASSERT(c0 + n <= kv_cells);
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (v < PG) {
            save_state(cell);
        }
        memcpy(X->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) X->vir + (size_t) v * H, 0, (size_t) (PG - v) * H * sizeof(ggml_bf16_t));
        X->flush();
        auto * pp = (int32_t *) POS->vir;
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < PG; ++r) {
                const int rr = r < v ? r : std::max(0, v - 1);
                pp[s * PG + r] = p4[(size_t) (r0 + rr) * 4 + std::min(s, 2)];
            }
        }
        POS->flush();
        ((int32_t *) INDPTR->vir)[0] = 0;
        ((int32_t *) INDPTR->vir)[1] = (int32_t) (cell / PG);
        *(int32_t *) LASTLEN->vir    = PG;
        ((int32_t *) CLOC->vir)[0]   = (int32_t) (cell / PG);
        ((int32_t *) CLOC->vir)[1]   = 0;
        *(int32_t *) VALID->vir      = v;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        {
            axera_timer tl("prefill (page)");
            for (auto & c : pf) {
                c.k->run(c.in, c.out, "dense prefill block");
            }
        }
        const axera_cmm & po = OUT[(pf.size() - 1) & 1];
        po->invalidate(po->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
        if (v < PG) {
            page_x.assign(x + (size_t) r0 * H, x + (size_t) (r0 + v) * H);
            page_p.assign(p4 + (size_t) r0 * 4, p4 + (size_t) (r0 + v) * 4);
        } else if (r0 + PG == n) {
            save_state(cell + PG);
        }
    }
}

ggml_status dense_blocks::compute(ggml_cgraph * g) {
    const auto & t       = names.of(g);
    ggml_tensor * norm0  = names.find("attn_norm-0");
    ggml_tensor * logits = names.find("result_output");
    ggml_tensor * inp_pos = names.find("inp_pos");
    if (!norm0 || !logits || !inp_pos) {
        GGML_LOG_ERROR("axera dense_blocks: a graph of %d nodes that is not the whole model (-ngl 99)\n", g->n_nodes);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * x0 = norm0;
    while (x0 && x0->op != GGML_OP_RMS_NORM) {
        x0 = x0->src[0];
    }
    GGML_ASSERT(x0 && x0->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(x0->src[0]));
    if (x0->src[0]->op != GGML_OP_NONE) {
        int n0 = 0;
        while (n0 < g->n_nodes && g->nodes[n0] != x0) {
            ++n0;
        }
        axera_timer tc("input (cpu graph)");
        cpu_compute(g, g->nodes, n0, "input (cpu)");
    }
    x0 = x0->src[0];
    const int H_ = (int) x0->ne[0], T = (int) x0->ne[1];
    int       L_ = 0;
    while (t.count("blk." + std::to_string(L_) + ".attn_norm.weight")) {
        ++L_;
    }
    const ggml_tensor * rope_node = nullptr;
    for (int i = 0; i < g->n_nodes && !rope_node; i++) {
        rope_node = g->nodes[i]->op == GGML_OP_ROPE ? g->nodes[i] : nullptr;
    }
    GGML_ASSERT(rope_node && "axera dense_blocks: no ROPE node in the graph");
    const ggml_tensor * w0 = t.at("blk.0.attn_norm.weight");
    if (bound != w0->data || L_ != L) {
        init(t, rope_node, L_, H_);
        bound = w0->data;
    }
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    const kv_view kv(k_idxs, kqm, man.context, PG);
    if (axera_n_seq_max() != 1) {
        GGML_LOG_ERROR("axera dense_blocks: one sequence only (-np 1)\n");
        return GGML_STATUS_FAILED;
    }
    for (int i = 1; i < T; ++i) {
        if (kv.stream_of(i) != 0 || kv.cell_of(i) != kv.cell_of(0) + i) {
            GGML_LOG_ERROR("axera dense_blocks: a batch of non-consecutive cells\n");
            return GGML_STATUS_FAILED;
        }
    }
    const int32_t * pos = (const int32_t *) inp_pos->data;
    const bool      mr  = inp_pos->ne[0] == 4 * (int64_t) T;
    std::vector<int32_t> pb((size_t) T * 4);
    for (int r = 0; r < T; ++r) {
        for (int s = 0; s < 4; ++s) {
            pb[(size_t) r * 4 + s] = mr ? pos[(size_t) s * T + r] : pos[r];
        }
    }
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
    const int64_t        c0 = kv.cell_of(0);
    if (c0 == 0) {
        zero_state();
    }
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
        });
    }
    const bool warmup = c0 == 0 && names.find("inp_mixed_slots_dummy");
    if (T > 1) {
        const int64_t s0 = c0 - c0 % PG;
        const int     k  = (int) (c0 - s0);
        std::vector<ggml_bf16_t> xs;
        std::vector<int32_t>     ps;
        if (k > 0) {
            if (!restart_point(s0, k)) {
                GGML_LOG_ERROR("axera dense_blocks: a prompt from cell %lld, inside a page whose state is not kept\n",
                               (long long) c0);
                return GGML_STATUS_FAILED;
            }
            page_x.resize((size_t) k * H);
            page_p.resize((size_t) k * 4);
            load_state();
            xs = page_x;
            ps = page_p;
        }
        xs.insert(xs.end(), xb.begin(), xb.end());
        ps.insert(ps.end(), pb.begin(), pb.end());
        axera_timer tp("prefill (all pages)");
        prefill(s0, k + T, xs.data(), ps.data(), h, k);
        snap_prompt = true;
    } else {
        if (c0 % PG == 0) {
            save_state(c0);
        }
        memcpy(A->vir, xb.data(), H * sizeof(ggml_bf16_t));
        A->flush(A->vir, H * sizeof(ggml_bf16_t));
        auto * pp = (int32_t *) POS->vir;
        for (int s = 0; s < 3; ++s) {
            pp[s] = pb[s];
        }
        POS->flush();
        ((int32_t *) INDPTR->vir)[0] = 0;
        ((int32_t *) INDPTR->vir)[1] = (int32_t) (c0 / PG + 1);
        *(int32_t *) LASTLEN->vir    = (int32_t) (c0 % PG);
        ((int32_t *) CLOC->vir)[0]   = (int32_t) (c0 / PG);
        ((int32_t *) CLOC->vir)[1]   = (int32_t) (c0 % PG);
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC }) {
            (*b)->flush();
        }
        {
            axera_timer tl("layers (decode)");
            for (auto & c : dec) {
                c.k->run(c.in, c.out, "dense decode block");
            }
        }
        A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, h.data(), H);
        page_x.insert(page_x.end(), xb.begin(), xb.end());
        page_p.insert(page_p.end(), pb.begin(), pb.end());
    }
    axera_timer th("output head");
    head.bind(man.get("lm_head"), logits->src[0]);
    if (man.has("lm_head.topk")) {
        head.bind_topk(man.get("lm_head.topk"));
    }
    const bool combined = head.bind_combined(man.get("lm_head"), logits->src[0]) && head.combined_ready(g, names);
    if (combined) {
        head.run_combined(g, names, h, T, nullptr, nullptr);
    } else {
        head.run(g, names, h, T);
    }
    if (warmup) {
        zero_state();
    }
    return GGML_STATUS_SUCCESS;
}

size_t dense_blocks::state_size(uint32_t flags) const {
    const bool partial = flags & state_flags_partial_only;
    const int64_t cur_cell = state_cell();
    const bool has_target = cur_cell > 0 || snap_cell >= 0 || !page_x.empty();
    if (!has_target) {
        return 0;
    }
    const size_t nkv_pages = std::min<size_t>((size_t) kv_pages, (size_t) ((cur_cell + PG - 1) / PG));
    size_t n = sizeof(state_header_dense);
    if (!partial) {
        n += (size_t) NB * nkv_pages * (kv_b / kv_pages);
    }
    n += ST->size + CQ->size + CK->size + CV->size;
    n += snap.size() + ksnap.size() + page_x.size() * sizeof(ggml_bf16_t) + page_p.size() * sizeof(int32_t) +
         kpage_x.size() * sizeof(ggml_bf16_t) + kpage_p.size() * sizeof(int32_t);
    return n;
}

size_t dense_blocks::state_write(void * dst, size_t size, uint32_t flags) {
    const size_t total = state_size(flags);
    if (!total || size < total) {
        return 0;
    }
    const bool    partial  = flags & state_flags_partial_only;
    const int64_t cur_cell = state_cell();
    const size_t  nkv_pages = std::min<size_t>((size_t) kv_pages, (size_t) ((cur_cell + PG - 1) / PG));
    state_header_dense h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "AXDBLK1", sizeof(h.magic) - 1);
    h.version    = 1;
    h.kind       = 3;
    h.has_target = 1;
    h.state_cell = cur_cell;
    h.snap_cell  = snap_cell;
    h.ksnap_cell = ksnap_cell;
    h.snap_prompt = snap_prompt;
    h.ncall      = (uint32_t) NB;
    h.nkv_pages  = partial ? 0 : (uint32_t) nkv_pages;
    h.kv_page_b  = kv_b / kv_pages;
    h.kv_n       = partial ? 0 : (uint64_t) NB * nkv_pages * (kv_b / kv_pages);
    h.st_b = ST->size;
    h.cq_b = CQ->size;
    h.ck_b = CK->size;
    h.cv_b = CV->size;
    h.snap_n = snap.size();
    h.ksnap_n = ksnap.size();
    h.page_x_n = page_x.size() * sizeof(ggml_bf16_t);
    h.page_p_n = page_p.size() * sizeof(int32_t);
    h.kpage_x_n = kpage_x.size() * sizeof(ggml_bf16_t);
    h.kpage_p_n = kpage_p.size() * sizeof(int32_t);
    h.partial = partial;
    char * p = (char *) dst;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    if (!partial) {
        const size_t page_n = nkv_pages * (kv_b / kv_pages);
        for (const axera_cmm & c : KVC) {
            c->invalidate(c->vir, page_n);
            memcpy(p, c->vir, page_n);
            p += page_n;
        }
    }
    for (const axera_cmm * b : { &ST, &CQ, &CK, &CV }) {
        (*b)->invalidate((*b)->vir, (*b)->size);
        memcpy(p, (*b)->vir, (*b)->size);
        p += (*b)->size;
    }
    auto put = [&](const void * src, size_t n) {
        if (n) {
            memcpy(p, src, n);
            p += n;
        }
    };
    put(snap.data(), h.snap_n);
    put(ksnap.data(), h.ksnap_n);
    put(page_x.data(), h.page_x_n);
    put(page_p.data(), h.page_p_n);
    put(kpage_x.data(), h.kpage_x_n);
    put(kpage_p.data(), h.kpage_p_n);
    return (size_t) (p - (char *) dst);
}

size_t dense_blocks::state_read(const void * src, size_t size, uint32_t flags) {
    if (size < sizeof(state_header_dense)) {
        return 0;
    }
    const bool partial = flags & state_flags_partial_only;
    state_header_dense h;
    memcpy(&h, src, sizeof(h));
    if (memcmp(h.magic, "AXDBLK1", sizeof(h.magic) - 1) != 0 || h.version != 1 || h.kind != 3 || h.partial != partial) {
        return 0;
    }
    if (!h.has_target || h.ncall != (uint32_t) NB || h.kv_page_b != kv_b / kv_pages || h.st_b != ST->size ||
        h.cq_b != CQ->size || h.ck_b != CK->size || h.cv_b != CV->size || h.page_x_n % (H * sizeof(ggml_bf16_t)) ||
        h.page_p_n % (4 * sizeof(int32_t)) || h.kpage_x_n % (H * sizeof(ggml_bf16_t)) ||
        h.kpage_p_n % (4 * sizeof(int32_t)) || h.state_cell < 0 || h.state_cell > kv_cells ||
        (partial ? h.nkv_pages != 0 || h.kv_n != 0 : h.nkv_pages > (uint32_t) kv_pages ||
                                                      h.kv_n != (uint64_t) NB * h.nkv_pages * h.kv_page_b)) {
        return 0;
    }
    size_t need = sizeof(h) + h.kv_n + h.st_b + h.cq_b + h.ck_b + h.cv_b + h.snap_n + h.ksnap_n + h.page_x_n +
                  h.page_p_n + h.kpage_x_n + h.kpage_p_n;
    if (size < need) {
        return 0;
    }
    const char * p = (const char *) src + sizeof(h);
    if (!partial) {
        const size_t page_n = h.nkv_pages * h.kv_page_b;
        for (const axera_cmm & c : KVC) {
            memcpy(c->vir, p, page_n);
            c->flush(c->vir, page_n);
            p += page_n;
        }
    }
    for (const axera_cmm * b : { &ST, &CQ, &CK, &CV }) {
        memcpy((*b)->vir, p, (*b)->size);
        (*b)->flush((*b)->vir, (*b)->size);
        p += (*b)->size;
    }
    auto get = [&](void * dst, size_t n) {
        if (n) {
            memcpy(dst, p, n);
            p += n;
        }
    };
    snap.resize(h.snap_n);
    ksnap.resize(h.ksnap_n);
    page_x.resize(h.page_x_n / sizeof(ggml_bf16_t));
    page_p.resize(h.page_p_n / sizeof(int32_t));
    kpage_x.resize(h.kpage_x_n / sizeof(ggml_bf16_t));
    kpage_p.resize(h.kpage_p_n / sizeof(int32_t));
    get(snap.data(), h.snap_n);
    get(ksnap.data(), h.ksnap_n);
    get(page_x.data(), h.page_x_n);
    get(page_p.data(), h.page_p_n);
    get(kpage_x.data(), h.kpage_x_n);
    get(kpage_p.data(), h.kpage_p_n);
    snap_cell   = h.snap_cell;
    ksnap_cell  = h.ksnap_cell;
    snap_prompt = h.snap_prompt;
    return (size_t) (p - (const char *) src);
}

}  // namespace

std::shared_ptr<model> make_dense_blocks(const manifest & m) {
    return std::make_shared<dense_blocks>(m);
}

}  // namespace axera
