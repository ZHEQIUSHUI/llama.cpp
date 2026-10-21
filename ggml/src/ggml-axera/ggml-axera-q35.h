// Axera NPU backend: the Qwen3.5 graph template (GGML_AXERA_TEMPLATE=qwen35; 0.8B .. 27B dense, 35B-A3B MoE, MTP), which
// also runs Qwen3 transformers (attention layers only, qwen3_* kernels: Qwen3, the text models of Qwen3-VL / Qwen3-ASR,
// the Qwen3-TTS talker, Qwen3-Embedding / Reranker, whose pooling after the final norm stays on the CPU)
#pragma once
//
//
// The backend claims the whole graph and runs it with the AXDSL layer kernels instead of op by op: per token, 24
// calls (gated delta net layer / gated attention layer, weights straight from the GGUF tensors in CMM), then the final
// norm on the CPU and the output projection with the GEMV kernels. The kernels' recurrent state and KV cache live in
// the backend, not in the llama.cpp memory: a single sequence with consecutive positions, restarted at position 0.
// Kernels in GGML_AXERA_KERNEL_DIR: qwen35_gdn.axmodel, qwen35_attn.axmodel and the mul_mat GEMV kernel for the output
// projection. The attention kernels' mask input [LMAX / BC, BC] gives their context LMAX and KV block BC; the RoPE
// parameters come from the graph's ROPE nodes. On the llama.cpp KV cache the attention kernels are
// qwen35_attn*_kvc_c<LMAX> (several contexts may sit side by side); the smallest LMAX >= -c runs.
//

// Sources: ggml-axera-q35-bind.cpp (kernels, IO binding, initialization), ggml-axera-q35-run.cpp (a graph: batch
// bookkeeping, chunks, layers), ggml-axera-q35-moe.cpp (routed experts), ggml-axera-q35-mc.cpp (multi-core prefill
// chain). The split attention, RoPE and output head are the backend's (ggml-axera-attn.h, ggml-axera-graph.cpp).

#include "ggml-axera-attn.h"

struct axera_q35_layer {
    bool                               gdn = false;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    int                                si = -1, so = -1, ci = -1, co = -1;  // GDN state in/out, swapped every step
    bool                               in_place = false;  // state updated in the llama.cpp recurrent cache tensors
    struct region {  // a state input in the llama.cpp recurrent cache: one row per sequence
        ggml_backend_axera_buffer_context * b;
        char *                              base;    // row 0
        size_t                              size;    // bytes the kernel uses
        size_t                              stride;  // bytes per row
        int                                 in = -1, out = -1;  // decode IO indices (state_in/out or conv_in/out)
        bool                                conv = false;
    };
    std::vector<region>                cache_state;  // those cache ranges
    int64_t                            row = 0;      // the cache row (sequence) the IO points at
    using kv_region = axera_kv_region;
    std::vector<kv_region>             kv;
    int64_t                            stream = 0;  // the KV stream the IO points at
    struct prefill_io {  // prefill / batched decode kernel IO: same weights / state / KV as the decode IO
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        int                                si = -1, so = -1, ci = -1, co = -1, ko = -1, vo = -1;
        bool                               batch = false;  // state IO over P consecutive cache rows
        // the chunk's snapshot kernel (prefill::gdn_rs): its IO, its state IO indices and the snapshot outputs
        // state_rs<j> / conv_rs<j> (j = 1 .. P - 1), pointed at the llama.cpp cache planes per call
        std::vector<AX_ENGINE_IO_BUFFER_T> rin, rout;
        int                                rsi = -1, rso = -1, rci = -1, rco = -1;
        std::vector<int>                   rs_state, rs_conv;  // [j]: output index
    };
    std::vector<prefill_io>            pre;  // one per prefill chunk size
    using fd_io = axera_fd_io;
    std::vector<fd_io>                 fd;  // [0]: decode, [1 + v]: m.pre[v]
    struct expert_w {  // routed experts of an MoE layer (gate or gate-up, up, down): MXFP4 repacked in place
        ggml_backend_axera_buffer_context * b = nullptr;
        char *                              base = nullptr;  // expert 0
        size_t                              stride = 0, data = 0, scale = 0;  // bytes per expert, its elements, scales
    };
    expert_w                           exps[3];
    bool                               exps_gu = false;  // exps[0] holds gate and up (m.ex[1]), exps[1] unused
    struct mc_io {  // multi-core prefill chain of this layer: [gdn | head, seg, tail] then mlp
        std::vector<AX_ENGINE_IO_BUFFER_T> ain, aout, sin, sout, tin, tout, min, mout;
        int si = -1, so = -1, ci = -1, co = -1;            // state IO of the gdn call (rebound to the current row)
        int dsi = -1, dci = -1;                            // the decode kernel's state inputs (ly.in)
        int kc = -1, vc = -1, mask = -1, mo = -1, lo = -1, oo = -1;  // segment IO, rebound per segment
    };
    std::map<int, mc_io>               mc;  // by chunk size P
    std::vector<axera_cmm>             own;                                // converted weights, state, KV cache
    std::map<std::string, std::pair<ggml_backend_axera_buffer_context *, ggml_backend_axera_buffer_context *>> mx;
    std::vector<ggml_backend_axera_buffer_context *> state;                // state buffers to clear on restart
};

struct axera_q35 {
    bool                  ready = false;
    const void *          bound = nullptr;  // data of blk.0.attn_norm.weight the layers were bound to
    int                   L = 0, H = 0, LMAX = 0, BC = 0, ROT = 0;
    // Qwen3.5 (gated delta net layers between gated attention layers: qwen35_* kernels) or a Qwen3 transformer
    // (attention layers only, no output gate: qwen3_* kernels; also the text model of Qwen3-VL / Qwen3-ASR)
    bool                  hybrid = true;
    std::string           ap     = "qwen35";  // kernel name prefix of the attention / dense layers
    int64_t               N = 0;  // KV cells in use: the llama.cpp cache size (<= LMAX), else LMAX
    double                THETA = 0;
    float                 freq_scale = 1.0f;
    int                   rope_mode = 0;               // GGML_ROPE_TYPE_*: (I)MROPE takes position t / h / w / e by section
    int                   sections[4] = { 0, 0, 0, 0 };  // M-RoPE sections (rotary frequency pairs per position)
    ggml_axera_kernel *   gdn = nullptr, *attn = nullptr;
    axera_cmm             A, B, POS, NBLK, MASK, COS, SIN;  // A/B: layer input/output ping-pong
    std::shared_ptr<ggml_backend_axera_buffer_context> lm_q, lm_s;  // MXFP8 output projection (shared by contexts)
    axera_cmm             lm_bf;                            // bf16 copy of a non-bf16 output projection
    struct prefill {  // one prefill chunk size, or (batch) one batched decode size
        int                 P = 0;
        int64_t             R = 0;  // mask rows per KV block
        bool                batch = false;  // P tokens of P consecutive sequences (recurrent rows), one each
        bool                mc    = false;  // the multi-core chain m.mc[P] exists (gdn / attn may be null then)
        ggml_axera_kernel * gdn = nullptr, *attn = nullptr;
        axera_cmm           A, B, MASK, COS, SIN, POS;  // POS: the batch's P cells
        axera_cmm           BASE, H2, LOG;              // MoE: the layer kernel's base, h2 and router logits rows
        // speculative decoding (llama.cpp n_rs_seq > 0): the GDN kernel that also writes the recurrent state after
        // each of the P tokens (qwen35_gdn_prefill<P>[_mx8]_rs) into the cache's snapshot planes
        ggml_axera_kernel * gdn_rs = nullptr;
        // MoE: chunks of a group run layer by layer, the routed experts once per layer for the whole group (each
        // expert's weights read once for more tokens). sub > 0: a copy of the group's first chunk (own buffers and
        // split attention state m.fd[fdk]), never picked alone; group: the chunks of the group, this one first
        int                 sub = 0, fdk = 0;  // fdk: key of the split attention kernels in m.fd (0: P)
        std::vector<int>    group;
    };
    std::vector<prefill>  pre;  // prefill chunks, then batched decode sizes, largest first
    const ggml_tensor *   lm = nullptr;
    std::map<const ggml_tensor *, axera_cmm> bf16_of;  // bf16 copies of non-bf16 GGUF weights (quantized GGUF)
    std::vector<axera_q35_layer> layers;
    // MTP draft context (llama.cpp LLM_GRAPH_TYPE_DECODER_MTP): one attention layer, the model's NextN block after the
    // main layers; graph layer il is model layer il + off. Layer 0's input is eh_proj([enorm(embedding) | hnorm(h)])
    bool        mtp = false;
    int         off = 0;
    axera_cmm   eh_bf, EH;  // eh_proj in bf16, its output row
    std::vector<float>    inv_freq;
    int64_t               next_pos = 0;
    bool                  rollback = false;  // all GDN state lives in the llama.cpp cache: earlier positions resumable
    bool                  kv_llama = false;  // attention kernels on the llama.cpp KV cache (cells, f16) and its KQ mask
    std::string           ctx_sfx;           // attention kernel name suffix: "_kvc_c<LMAX>" on the llama.cpp KV cache
    bool                  flash = false;     // attention as head / per-segment / tail kernels (flash decoding)
    bool                  pre_mx = false;    // prefill kernels with MXFP8 weights
    int64_t               mc_dec_min = 0;    // decode on the multi-core chain from this KV width (cells) on
    int64_t               S     = 0;         // their KV segment, cells
    using fd_kernels = axera_fd_kernels;
    struct mc_kernels {  // multi-core chain for P tokens: <ap>_{gdn,mlp,attn_head,attn_tail}_mc<P>[_mx8] (no gdn: Qwen3)
        ggml_axera_kernel * gdn = nullptr, *mlp = nullptr, *head = nullptr, *tail = nullptr;
        int                 np = 0;            // partial attention slots the tail merges (segments x segment cores)
        size_t              ms = 0, ls = 0, os = 0;  // bytes per slot of m / l / o
        axera_cmm           BASE1, PART0, PART1, ZERO, MALL, LALL, OALL;  // the residual base 0 is m.pre[v].A
    };
    std::map<int, mc_kernels> mc;  // by P
    std::map<int, fd_kernels> fd;  // by rows R
    // MoE (Qwen3.5-35B-A3B): per layer the layer kernel (qwen35moe_*: up to the shared expert and the router logits),
    // then the routed experts kernel on the top-K experts the host picks from the logits
    bool      moe      = false;
    int       n_expert = 0, top_k = 0;
    axera_cmm BASE, H2, LOGITS, RW;  // residual + gated shared expert, normed state, router logits, weights
    struct experts_kernel {          // qwen35moe_experts<K>[_gu]_mxfp4 and its IO (weights rebound per token)
        ggml_axera_kernel *                k = nullptr;
        axera_cmm                          PART;  // multi-core build: the cores' fp32 partial sums (out = their sum)
        std::vector<int>                   w[4], s[4];  // input indices of w_<kind><i> / s_<kind><i>, kinds g u d c
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out[2];  // out: by the layer parity
    };
    // [0]: gate and up weights apart (w_g / w_u), [1]: one gate-up weight per expert (w_c: gate rows then up rows,
    // one projection instead of two; axera_repack_mxfp4_gu), preferred when the kernel exists
    experts_kernel         ex[2];
    std::vector<axera_cmm> gu_own;  // gate-up experts of layers whose gate and up are not back to back
    // prefill: the routed experts of a chunk grouped by expert, qwen35moe_slots<S>_g<G>_gu_mxfp4: per call S slots,
    // each one expert (gate-up and down weights bound by address) on up to G token rows gathered by the host from h2
    struct slots_kernel {
        ggml_axera_kernel *                k = nullptr;
        int                                S = 0, G = 0, x = -1, rw = -1, out = -1;
        std::vector<int>                   w[2], s[2];  // input indices of w_<c|d><slot> / s_<c|d><slot>
        std::vector<AX_ENGINE_IO_BUFFER_T> in, outs;
    };
    slots_kernel slots;
    axera_cmm    SX, SRW, SOUT;  // all slots of a chunk's layer: rows [slots][G][H] bf16, weights f32, results f32
};

// one graph on the template: what its layers share (ggml-axera-q35-run.cpp, -moe.cpp, -mc.cpp)
struct axera_q35_batch {
    ggml_backend_axera_context * ctx;
    axera_q35 &                  m;
    int                          L, H, T;
    axera_kv_batch               kv;  // with the llama.cpp KV cache
    int64_t                      rs_plane = 0;      // rows between recurrent-state snapshot planes
    bool                         rs_now   = false;  // this chunk runs the snapshot GDN kernels (a verification batch)
    std::vector<int>             e_ids;             // MoE decode: experts by router logit

    axera_q35_batch(ggml_backend_axera_context * ctx, axera_q35 & m, int L, int H, int T) :
        ctx(ctx), m(m), L(L), H(H), T(T), e_ids(m.n_expert) {}

    // split attention for R rows: always with segments, else for wide masks when its kernels exist for the tier
    bool use_split(int R) const;
    // a prefill chunk without a fused attention kernel (MoE P = 128): always split
    bool pre_split(int v) const;
    // layer il's kernel(s) for run kind v (decode: -1) on tokens t0 .. (position p), without the MoE experts
    void run_layer(int il, int v, int t0, int64_t p);
    void run_layers(int v, int t0, int64_t p);
    // a MoE group: chunks vs of P tokens from t0 (position p), layer by layer, the experts once per layer
    void run_group(const std::vector<int> & vs, int t0, int64_t p);
    // ggml-axera-q35-moe.cpp
    void run_experts(int il, int64_t p);
    void run_experts_pre(int il, const std::vector<int> & vs);
    // ggml-axera-q35-mc.cpp
    void run_mc(int v, int t0);
};

const char * axera_q35_gguf_name(const axera_q35 & m, const std::string & n);
AX_ENGINE_IO_BUFFER_T axera_q35_weight_bf16(axera_q35 & m, const ggml_tensor * w, uint32_t size);
AX_ENGINE_IO_BUFFER_T axera_q35_gguf_input(axera_q35 & m, axera_q35_layer & ly, const ggml_tensor * w,
                                           const std::string & n, uint32_t size);
AX_ENGINE_IO_BUFFER_T axera_q35_mx_input(axera_q35_layer & ly, const ggml_tensor * w, const std::string & n,
                                         uint32_t size);
void axera_q35_bind_layer(axera_q35 & m, int il, const std::map<std::string, ggml_tensor *> & t);
void axera_q35_bind_prefill(axera_q35 & m, int il, size_t v, const std::map<std::string, ggml_tensor *> & t);
void axera_q35_context(ggml_axera_kernel * k, int & LMAX, int & BC);
bool axera_q35_load_flash(axera_q35 & m, int R, bool seg_only = false);
int axera_q35_fd_key(const axera_q35 & m, int v);
void axera_q35_clone_flash(axera_q35 & m, int R, int key);
void axera_q35_bind_flash(axera_q35 & m, int il, int v, const std::map<std::string, ggml_tensor *> & t);
bool axera_q35_load_mc(axera_q35 & m, int P);
void axera_q35_bind_mc(axera_q35 & m, int il, size_t v, const std::map<std::string, ggml_tensor *> & t);
void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                    const ggml_tensor * lm, const ggml_tensor * rope);
void axera_q35_set_row(axera_q35 & m, int64_t row);
void axera_q35_set_batch_rows(axera_q35 & m, size_t v, int64_t row0);
void axera_q35_zero_row(axera_q35 & m, int64_t row);
void axera_q35_set_stream(axera_q35 & m, int64_t s);
void axera_q35_restart(axera_q35 & m);