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
// bookkeeping, chunks, layers), ggml-axera-q35-moe.cpp (routed experts). The split attention, RoPE and output head are the backend's (ggml-axera-attn.h, ggml-axera-graph.cpp).

#include "ggml-axera-attn.h"

#include <array>

// a packed variant of a layer kernel (<name>_pk, its <file>.layout: the layer's weights in the packs pw / ps / pf / pb,
// the other IO by name): per input the unpacked kernel's input index or -1 - pack, per output the unpacked output
struct axera_q35_pk_kernel {
    ggml_axera_kernel * k = nullptr;
    std::vector<int>    in, out;
    size_t              pack_bytes[4] = {};
    struct member {
        int         pack, body;  // pack 0 .. 3, the unpacked kernel's input
        size_t      off, bytes;
    };
    std::vector<member> members;
    std::string         sig;  // the layout: kernels of the same layout share a layer's packs
};

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
        std::vector<AX_ENGINE_IO_BUFFER_T> pin, pout;  // the packed kernel's IO (its packs bound; the rest per call)
    };
    std::map<std::string, std::array<axera_cmm, 4>> pk_packs;  // packed prefill kernels' packs, by layout
    std::vector<prefill_io>            pre;  // one per prefill chunk size
    using fd_io = axera_fd_io;
    std::vector<fd_io>                 fd;  // [0]: decode, [1 + v]: m.pre[v]
    struct expert_w {  // routed experts of an MoE layer (gate or gate-up, up, down): MXFP4 repacked in place
        ggml_backend_axera_buffer_context * b = nullptr;
        char *                              base = nullptr;  // expert 0
        size_t                              stride = 0, data = 0, scale = 0;  // bytes per expert, its elements, scales
        int64_t                             n = 0;       // experts
        bool                                planar = false;  // all experts' codes, then all their scales
        char * w(int64_t e) const { return planar ? base + e * data : base + e * stride; }
        char * s(int64_t e) const { return planar ? base + n * data + e * scale : base + e * stride + data; }
    };
    expert_w                           exps[3];
    bool                               exps_gu = false;  // exps[0] holds gate and up (m.ex[1]), exps[1] unused
    std::vector<axera_cmm>             own;                                // converted weights, state, KV cache
    axera_cmm                          packs[4];  // a packed whole-layer kernel's pw / ps / pf / pb inputs
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
        ggml_axera_kernel * gdn = nullptr, *attn = nullptr;
        axera_q35_pk_kernel gdn_pk, attn_pk;  // their packed variants (<name>_pk), if built
        axera_cmm           A, B, MASK, COS, SIN, POS;  // POS: the batch's P cells
        axera_cmm           CPOS, CNBLK;  // a MoE group's chunk copy with fused attention: its own pos / nblk inputs
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
    bool                  attn_all = false;  // the fused attention kernels run on all cores: no split below LMAX
    bool                  pre_mx = false;    // prefill kernels with MXFP8 weights
    int64_t               S     = 0;         // their KV segment, cells
    using fd_kernels = axera_fd_kernels;
    std::map<int, fd_kernels> fd;  // by rows R
    // MoE (Qwen3.5-35B-A3B): per layer the layer kernel (qwen35moe_*: up to the shared expert and the router logits),
    // then the routed experts kernel on the top-K experts the host picks from the logits
    bool      moe      = false;
    int       n_expert = 0, top_k = 0;
    axera_cmm BASE, H2, LOGITS, RW;  // residual + gated shared expert, normed state, router logits, weights
    struct experts_kernel {          // qwen35moe_experts<K>[_gu]_mxfp4 and its IO (weights rebound per token)
        ggml_axera_kernel *                k = nullptr;
        std::vector<int>                   w[4], s[4];  // input indices of w_<kind><i> / s_<kind><i>, kinds g u d c
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out[2];  // out: by the layer parity
    };
    // [0]: gate and up weights apart (w_g / w_u), [1]: one gate-up weight per expert (w_c: gate rows then up rows,
    // one projection instead of two; axera_repack_mxfp4_gu), preferred when the kernel exists
    experts_kernel         ex[2];
    // decode: layer i's routed experts and layer i + 1's layer kernel in one call (qwen35moe_experts<K>_gu_gdn[_mx8],
    // _gu_attn[_mx8]_kvc_c<LMAX>): its inputs those of the gate-up experts kernel and of the layer kernel but x
    struct merged_kernel {
        ggml_axera_kernel * k = nullptr;
        std::vector<int>    body, ein;  // per input: the layer kernel's input index, else (-1) the experts kernel's
        std::vector<int>    bout;       // per output: the layer kernel's output index
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    };
    merged_kernel          mx[2];  // [0]: next layer a GDN layer, [1]: an attention layer
    // decode: a whole MoE layer in one call (qwen35moe_layer_gdn_mx8, _layer_attn_mx8_kvc_c<LMAX>): the layer, the
    // router, top-K and the experts on the NPU, which picks them out of the layer's planar expert tensors (e_gw / e_gs
    // gate-up codes / scales, e_dw / e_ds down); inputs and outputs those of the layer kernel by name, out the next x
    struct full_kernel {
        ggml_axera_kernel * k = nullptr;
        std::vector<int>    body, plane;  // per input: the layer kernel's input index, else (-1) the expert plane 0 .. 3
        // packed build (_pk, <file>.layout): per input its pack 0 .. 3 (pw ps pf pb, else -1); the members, copied
        // from the layer kernel's bound inputs once: pack, byte offset, bytes, layer kernel input index
        std::vector<int>    pack;
        size_t              pack_bytes[4] = {};
        struct member {
            int    pack, body;
            size_t off, bytes;
            size_t rows = 0, row_bytes = 0;                    // column blocks: source rows, bytes per row
            std::vector<std::pair<size_t, size_t>> blocks;     // byte ranges of a row, stored block after block
        };
        std::vector<member> members;
        std::vector<int>    bout;         // per output: the layer kernel's output index (-1: out, the next layer's x)
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    };
    full_kernel            full[2];  // [0]: a GDN layer, [1]: an attention layer
    // decode: G consecutive whole MoE layers in one call (qwen35moe_group<G>_<kinds>_mx8_kvc_c<LMAX>, kinds g / a per
    // layer): x, the packs of all G layers (members "L<j>.<name>" in the .layout), per layer j its expert planes
    // e_gw<j> .. e_ds<j> and its state st<j> / cv<j> (GDN) or kc<j> / vc<j> (attention), in place; cos .. mask shared
    struct moe_group_kernel {
        ggml_axera_kernel * k = nullptr;
        int                 G = 0;
        size_t              pack_bytes[4] = {};
        struct member : full_kernel::member {
            int         j = 0;  // its layer in the group
            std::string name;   // input name of the layer kernel
        };
        std::vector<member> members;
        struct io {
            int         kind = 0;  // 0 x, 1 pack, 2 expert plane, 3 layer j's kernel input / output n, 4 out
            int         j = 0, idx = 0;  // the layer, the pack / plane index
            std::string n;
        };
        std::vector<io>                    ins, outs;
        std::vector<std::array<axera_cmm, 4>> packs;  // per group (layers g * G ..)
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    };
    moe_group_kernel       mgrp;
    bool                   planar = false;  // the layers' experts in planes (whole layers, the planar slots kernel)
    std::vector<axera_cmm> gu_own;  // gate-up experts of layers whose gate and up are not back to back
    // prefill: the routed experts of a chunk grouped by expert, qwen35moe_slots<S>_g<G>_gu_mxfp4: per call S slots,
    // each one expert (gate-up and down weights bound by address) on up to G token rows gathered by the host from h2
    struct slots_kernel {
        ggml_axera_kernel *                k = nullptr;
        int                                S = 0, G = 0, x = -1, rw = -1, out = -1;
        bool                               ob = false;  // out in bf16 (else f32)
        std::vector<int>                   w[2], s[2];  // input indices of w_<c|d><slot> / s_<c|d><slot>
        int                                tok = -1, h2 = -1;  // _px build: [S][expert, G tokens] rows of h2
        int                                sid = -1, plane[4] = { -1, -1, -1, -1 };  // planar build (_pl): slot
                                                                                      // experts, e_gw e_gs e_dw e_ds
        axera_cmm                          SID;
        std::vector<AX_ENGINE_IO_BUFFER_T> in, outs;
    };
    slots_kernel slots;
    // px builds with fewer rows per slot (qwen35moe_slots<S>_g<G>_px..., G < slots.G), smallest G first: a group whose
    // experts average fewer picks takes one (less padding computed); run_experts_px picks by the average
    std::vector<slots_kernel> slots_px;
    // MoE groups on the NPU (slots kernel _px, qwen35moe_combine<K>_t<T>_r<R>): the group's chunks' A / B / h2 / base /
    // logits rows are views of one buffer each, the slots kernel reads its rows out of the group's h2 by token, the
    // combine kernel adds each token's K slot rows to its base into the group's next x
    struct combine_kernel {
        ggml_axera_kernel *                k = nullptr;
        int                                T = 0, R = 0;
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
    };
    combine_kernel comb;
    std::vector<combine_kernel> comb_small;  // the same for fewer tokens (T < comb.T), smallest first: a group that
                                             // fills only its first chunks adds only their rows
    axera_cmm      GA, GB, GH2, GBASE, GLOG, TOK, LOC;
    axera_cmm    SX, SRW, SOUT;  // all slots of a chunk's layer: rows [slots][G][H] bf16, weights f32, results f32 / bf16
};

// one graph on the template: what its layers share (ggml-axera-q35-run.cpp, -moe.cpp)
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
    void run_experts(int il, int64_t p, int next = -1);  // next >= 0: with layer next in one call (m.mx)
    bool merged(int il) const;  // layer il's body runs with layer il - 1's experts
    bool whole(int il) const;   // layer il runs as one call (m.full)
    void run_whole(int il, int64_t p);
    bool grouped(int il) const;  // layers il .. il + G - 1 run as one call (m.mgrp)
    void run_grouped(int il, int64_t p);
    void run_experts_pre(int il, const std::vector<int> & vs);
    void run_experts_px(int il, const std::vector<int> & vs);  // a group of contiguous chunks, routing on the host only
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
void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                    const ggml_tensor * lm, const ggml_tensor * rope);
void axera_q35_set_row(axera_q35 & m, int64_t row);
void axera_q35_set_batch_rows(axera_q35 & m, size_t v, int64_t row0);
void axera_q35_zero_row(axera_q35 & m, int64_t row);
void axera_q35_set_stream(axera_q35 & m, int64_t s);
void axera_q35_restart(axera_q35 & m);