// Axera NPU backend: the LightNav-0 / Qwen3-VL-4B vision tower (mtmd clip graph, projector qwen3vl) in one call of the
// manifest's role vit: patch embedding, 24 layers, the main 2 x 2 merger and the mergers after layers 5 / 11 / 17, for
// one two-frame tubelet padded to N rows. The shipped kernel is vit_512 (448 real patch rows); the real patch count
// hides the padding through the key mask. The output row is [main, ds5, ds11, ds17] x 2560 float32.
//
// The host orders the patches in 2 x 2 merge order (as clip_graph_qwen3vl), resizes the learned position table
// (bilinear, align corners), makes the 2-D RoPE tables and the key mask: those depend on the image's size only and are
// kept for the next image of the same size. A video comes as pairs of frames (the tower's temporal merge): what goes
// into the kernel's patch rows and patch weight for a still or a pair is core/patch_embed's business. The layers'
// weights from the mmproj GGUF are stacked [layers, ...] bf16 in CMM once. If the manifest carries the NVFP4 scale
// inputs, the adapter quantizes the projections to the kernel's bank layout (codes, E4M3 group scales, one FP32
// scale2 a layer) instead; the shipped mmproj and ViT kernel are both BF16. This adapter is the only reader of the
// mmproj.
//
// The manifest may carry the same graph for several patch counts: role vit and roles vit_<N>. Each image runs on the
// smallest kernel that holds it; the kernels share the weights' CMM (the same inputs at every N) and keep their own
// tables, scratch and outputs.

#include "../model.h"
#include "../../ops/ops.h"
#include "../../core/patch_embed.h"
#include "ggml-axera.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>
#include <thread>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace axera {
namespace {

static std::string_view graph_base_name(const char * name) {
    const char * a = strchr(name, '#');
    const char * b = a ? strrchr(name, '#') : nullptr;
    return a && b > a ? std::string_view(a + 1, b - a - 1) : std::string_view(name);
}

// Normalize one 16x16 patch from both frames in kernel order: all rows of c0/f0, then c0/f1, c1/f0, ...
// Qwen2-VL uses mean = std = 0.5, i.e. x / 127.5 - 1.  Keeping the two IEEE operations (rather than a fused fma)
// preserves the exact values the R6 accuracy gate was measured with.
static inline void store_norm_patch(const uint8_t * row0, const uint8_t * row1, size_t stride, float * dst) {
#if defined(__ARM_NEON)
    const float32x4_t q255 = vdupq_n_f32(255.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t two  = vdupq_n_f32(2.0f);
    for (int c = 0; c < 3; ++c) {
        for (int f = 0; f < 2; ++f) {
            const uint8_t * src = f ? row1 : row0;
            for (int ky = 0; ky < 16; ++ky) {
                uint8x16x3_t rgb = vld3q_u8(src + ky * stride);
                uint8x16_t v = rgb.val[c];
                uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
                uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
                float32x4_t x0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                float32x4_t x1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
                float32x4_t x2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                float32x4_t x3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
                x0 = vmulq_f32(vsubq_f32(vdivq_f32(x0, q255), half), two);
                x1 = vmulq_f32(vsubq_f32(vdivq_f32(x1, q255), half), two);
                x2 = vmulq_f32(vsubq_f32(vdivq_f32(x2, q255), half), two);
                x3 = vmulq_f32(vsubq_f32(vdivq_f32(x3, q255), half), two);
                vst1q_f32(dst, x0);
                vst1q_f32(dst + 4, x1);
                vst1q_f32(dst + 8, x2);
                vst1q_f32(dst + 12, x3);
                dst += 16;
            }
        }
    }
#else
    for (int c = 0; c < 3; ++c) {
        for (int f = 0; f < 2; ++f) {
            const uint8_t * s = f ? row1 : row0;
            for (int ky = 0; ky < 16; ++ky) {
                for (int x = 0; x < 16; ++x) {
                    *dst++ = ((float) s[ky * stride + x * 3 + c] / 255.0f - 0.5f) / 0.5f;
                }
            }
        }
    }
#endif
}

// The clip graph is rebuilt for every tubelet. Its full name map costs ~3.7 ms, but this adapter needs four tensors.
// Scan the node sources once per name and stop at the first match; no persistent pointer is kept.
static ggml_tensor * graph_tensor(const ggml_cgraph * g, const char * want) {
    const std::string_view w(want);
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_tensor * node = g->nodes[i];
        if (graph_base_name(node->name) == w) {
            return node;
        }
        for (int src = 0; src < GGML_MAX_SRC && node->src[src]; ++src) {
            if (graph_base_name(node->src[src]->name) == w) {
                return node->src[src];
            }
            for (ggml_tensor * v = node->src[src]->view_src; v; v = v->view_src) {
                if (graph_base_name(v->name) == w) {
                    return v;
                }
            }
        }
    }
    return nullptr;
}

// A pulsar2-compiled vision tower (one two-frame tubelet, role vit). The axmodel carries the whole tower and
// the main merger plus deepstack 5/11/17; its IO is pixel F32 [N, 2 * 3 * PS * PS] and feat F32 [N / 4, 4 * H].
// The host only packs HF's patch rows (channel, temporal, patch rows, patch columns) from mtmd's normalized
// frames [W, H, 3, 2]; no weight conversion or separate patch projection is involved.
class lightnav_vit_pulsar final : public model {
  public:
    explicit lightnav_vit_pulsar(const manifest & m) :
        man(m), k(m.get("vit")) {
        GGML_ASSERT(k.find_input("pixel") >= 0 && k.find_output("feat") >= 0 &&
                    "axera lightnav_vit_pulsar: expected pixel -> feat IO");
        pixel_ = axera_cmm_alloc(k.in_bytes("pixel"));
        feat_  = axera_cmm_alloc(k.out_bytes("feat"));
        in_.push_back(axera_io(pixel_));
        out_.push_back(axera_io(feat_));
        const AX_ENGINE_IOMETA_T & pm = k.info->pInputs[k.input("pixel")];
        GGML_ASSERT(pm.nShapeSize == 2 && pm.pShape[1] == 1536 && "axera lightnav_vit_pulsar: pixel row width");
        NP = (int) pm.pShape[0];
        GGML_ASSERT(k.out_bytes("feat") == (size_t) (NP / 4) * 10240 * sizeof(float));
    }

    ggml_status compute(ggml_cgraph * g) override {
        ggml_tensor * raw = graph_tensor(g, "inp_raw");  // [W, H, 3, 2] f32, normalized
        ggml_tensor * pe  = graph_tensor(g, "v.position_embd.weight");
        ggml_tensor * pwe = graph_tensor(g, "v.patch_embd.weight");
        ggml_tensor * out = g->nodes[g->n_nodes - 1];
        GGML_ASSERT(raw && pe && pwe && "axera lightnav_vit_pulsar: expected tensors in the clip graph");
        const int     PS  = (int) pwe->ne[0];
        const int     W   = (int) raw->ne[0], HI = (int) raw->ne[1];
        const int     gw  = W / PS, gh = HI / PS, n = gw * gh;
        GGML_ASSERT(raw->type == GGML_TYPE_F32 && raw->ne[3] == 2 && PS == 16 && gw % 2 == 0 && gh % 2 == 0);
        GGML_ASSERT(pe->ne[0] == 1024 && n == NP && "axera lightnav_vit_pulsar: the kernel holds one 448-patch tubelet");
        GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == 10240 && out->ne[1] == n / 4);

        {
            axera_timer tm("vit pulsar pack (cpu, frame pairs)");
            const float * img = (const float *) raw->data;
            const size_t  fsz = (size_t) W * HI * 3;
            auto *        px  = (float *) pixel_->vir;
            int32_t n_raw = 0;
            const ggml_axera_raw_frame * raw_frames = ggml_backend_axera_raw_frames(&n_raw);
            bool raw_hwc = n_raw == 2;
            for (int i = 0; i < n_raw && raw_hwc; ++i) {
                raw_hwc = raw_frames[i].data && raw_frames[i].width == W && raw_frames[i].height == HI;
            }
            // the same 2 x 2 merge order as the DSL adapter's tables(): block row, block column, then the four patches
            axera_host_pool().run(n, [&](int64_t i) {
                const int64_t blk = i / 4;
                const int     py  = (int) ((blk / (gw / 2)) * 2 + (i % 4) / 2);
                const int     pxp = (int) ((blk % (gw / 2)) * 2 + (i % 4) % 2);
                float *       d  = px + i * 1536;
                if (raw_hwc) {
                    const uint8_t * row0 = raw_frames[0].data + ((size_t) (py * PS) * W + pxp * PS) * 3;
                    const uint8_t * row1 = raw_frames[1].data + ((size_t) (py * PS) * W + pxp * PS) * 3;
                    store_norm_patch(row0, row1, (size_t) W * 3, d);
                } else {
                    for (int c = 0; c < 3; ++c) {
                        for (int f = 0; f < 2; ++f) {
                            const float * plane = img + f * fsz + (size_t) c * W * HI;
                            for (int ky = 0; ky < PS; ++ky) {
                                memcpy(d, plane + (size_t) (py * PS + ky) * W + pxp * PS, PS * sizeof(float));
                                d += PS;
                            }
                        }
                    }
                }
            });
            pixel_->flush();
        }
        const size_t bytes = (size_t) (n / 4) * 10240 * sizeof(float);
        std::vector<AX_ENGINE_IO_BUFFER_T> out_call;
        const bool direct_out = ggml_backend_buffer_is_axera(out->buffer) && out->data && ggml_nbytes(out) == bytes;
        if (direct_out) {
            out_call.push_back(axera_io((ggml_backend_axera_buffer_context *) out->buffer->context, out->data, bytes));
        } else {
            out_call = out_;
        }
        {
            axera_timer tm(direct_out ? "vit pulsar (one call, direct CMM output)" :
                                        "vit pulsar (one call, buffered output)");
            k.run(in_, out_call, "vit");
        }
        if (!direct_out) {
            axera_timer tm("vit pulsar output readback");
            feat_->invalidate(feat_->vir, bytes);
            memcpy(out->data, feat_->vir, bytes);
            if (ggml_backend_buffer_is_axera(out->buffer)) {
                ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    void prewarm(ggml_cgraph * g) override {
        // the axmodel's weights are loaded with its handle; the pixel / feat buffers are already allocated
        GGML_UNUSED(g);
    }

  private:
    const manifest &                  man;
    const kernel &                    k;
    axera_cmm                         pixel_, feat_;
    std::vector<AX_ENGINE_IO_BUFFER_T> in_, out_;
    int                               NP = 0;
};

// The same pulsar2 tower compiled with a U8 RGB input and in-kernel normalization / Qwen2-VL patch reordering:
// frames [2, H, W, 3] -> feat [N / 4, 4 * H].  mtmd retains the source HWC bytes for this path, so no normalized
// F32 image is materialized or uploaded.  The graph's `inp_raw` stays an unused placeholder.
class lightnav_vit_pulsar_u8 final : public model {
  public:
    explicit lightnav_vit_pulsar_u8(const manifest & m) :
        man(m), k(m.get("vit")) {
        GGML_ASSERT(k.find_input("frames") >= 0 && k.find_output("feat") >= 0 &&
                    "axera lightnav_vit_pulsar_u8: expected frames -> feat IO");
        const AX_ENGINE_IOMETA_T & fm = k.info->pInputs[k.input("frames")];
        GGML_ASSERT(fm.eDataType == AX_ENGINE_DT_UINT8 && fm.nShapeSize == 4 && fm.pShape[0] == 2 &&
                    fm.pShape[1] == 256 && fm.pShape[2] == 448 && fm.pShape[3] == 3 &&
                    "axera lightnav_vit_pulsar_u8: expected frames U8 [2,256,448,3]");
        frames_ = axera_cmm_alloc(k.in_bytes("frames"));
        feat_   = axera_cmm_alloc(k.out_bytes("feat"));
        in_.push_back(axera_io(frames_));
        out_.push_back(axera_io(feat_));
        FH = fm.pShape[1];
        FW = fm.pShape[2];
        NP = (int) ((int64_t) FW / 16 * (FH / 16));
    }

    ggml_status compute(ggml_cgraph * g) override {
        ggml_tensor * raw = graph_tensor(g, "inp_raw");  // [W, H, 3, 2] f32 in the reference graph
        ggml_tensor * pe  = graph_tensor(g, "v.position_embd.weight");
        ggml_tensor * pwe = graph_tensor(g, "v.patch_embd.weight");
        ggml_tensor * out = g->nodes[g->n_nodes - 1];
        GGML_ASSERT(raw && pe && pwe && "axera lightnav_vit_pulsar_u8: expected tensors in the clip graph");
        const int PS = (int) pwe->ne[0];
        const int W  = (int) raw->ne[0], HI = (int) raw->ne[1];
        const int gw = W / PS, gh = HI / PS, n = gw * gh;
        GGML_ASSERT(raw->type == GGML_TYPE_F32 && raw->ne[3] == 2 && PS == 16 && gw % 2 == 0 && gh % 2 == 0);
        GGML_ASSERT(pe->ne[0] == 1024 && n == NP && W == FW && HI == FH &&
                    "axera lightnav_vit_pulsar_u8: the kernel holds one 448x256 two-frame tubelet");

        auto *       dst = (uint8_t *) frames_->vir;
        const size_t frame_bytes = (size_t) FW * FH * 3;
        int32_t n_raw = 0;
        const ggml_axera_raw_frame * raw_frames = ggml_backend_axera_raw_frames(&n_raw);
        bool packed = n_raw == 2;
        for (int i = 0; i < n_raw && packed; ++i) {
            packed = raw_frames[i].data && raw_frames[i].width == FW && raw_frames[i].height == FH;
        }
        {
            axera_timer tm(packed ? "vit pulsar u8 pack (cpu, raw frames)" : "vit pulsar u8 pack (cpu, fallback)");
            if (packed) {
                memcpy(dst, raw_frames[0].data, frame_bytes);
                memcpy(dst + frame_bytes, raw_frames[1].data, frame_bytes);
            } else {
                const int64_t fsz = (int64_t) W * HI;
                const float * img = (const float *) raw->data;
                for (int f = 0; f < 2; ++f) {
                    for (int c = 0; c < 3; ++c) {
                        const float * plane = img + f * 3 * fsz + (size_t) c * fsz;
                        for (int y = 0; y < HI; ++y) {
                            uint8_t * row = dst + f * frame_bytes + ((size_t) y * W) * 3 + c;
                            const float * src = plane + (size_t) y * W;
                            for (int x = 0; x < W; ++x) {
                                row[x * 3] = (uint8_t) std::max(0, std::min(255, (int) lrintf(src[x] * 127.5f + 127.5f)));
                            }
                        }
                    }
                }
            }
            frames_->flush();
        }

        const size_t bytes = (size_t) (n / 4) * 10240 * sizeof(float);
        std::vector<AX_ENGINE_IO_BUFFER_T> out_call;
        const bool direct_out = ggml_backend_buffer_is_axera(out->buffer) && out->data && ggml_nbytes(out) == bytes;
        if (direct_out) {
            out_call.push_back(axera_io((ggml_backend_axera_buffer_context *) out->buffer->context, out->data, bytes));
        } else {
            out_call = out_;
        }
        {
            axera_timer tm(direct_out ? "vit pulsar u8 (one call, direct CMM output)" :
                                        "vit pulsar u8 (one call, buffered output)");
            k.run(in_, out_call, "vit");
        }
        if (!direct_out) {
            // the CPU fallback: one copy when the graph's output is not CMM
            axera_timer tm("vit pulsar u8 output readback");
            feat_->invalidate(feat_->vir, bytes);
            memcpy(out->data, feat_->vir, bytes);
            if (ggml_backend_buffer_is_axera(out->buffer)) {
                ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    void prewarm(ggml_cgraph * g) override {
        GGML_UNUSED(g);
    }

  private:
    const manifest &                  man;
    const kernel &                    k;
    axera_cmm                         frames_, feat_;
    std::vector<AX_ENGINE_IO_BUFFER_T> in_, out_;
    int                               FW = 0, FH = 0, NP = 0;
};

class lightnav_vit_full final : public model {
  public:
    explicit lightnav_vit_full(const manifest & m);
    ~lightnav_vit_full() override {
        if (pairs_bg.joinable()) {
            pairs_bg.join();
        }
    }
    ggml_status compute(ggml_cgraph * g) override;
    void        prewarm(ggml_cgraph * g) override;

  private:
    // one kernel of the vision graph at its patch count
    struct sized {
        const kernel *                   k = nullptr;
        std::string                      role;
        int                              N = 0, QB = 0, BC = 0;
        std::map<std::string, axera_cmm> buf;              // its own IO: tables, scratch, outputs
        int                              gh_ = -1, gw_ = -1;  // the image size of the tables in buf
        std::vector<int>                 py, px;           // the patches in merge order
    };
    const manifest &                  man;
    std::vector<sized>                ks;           // by patch count, smallest first
    const kernel &                    k;            // the weights' sizes: ks[0]'s (every kernel's are the same)
    graph_names                       names;
    int                               HD = 0;
    std::map<std::string, axera_cmm>  wbuf;         // the weights' IO by name, shared by the kernels
    const void *                      bound = nullptr;
    patch_embed                       pemb;         // the patch projection: the kernel's px rows and w_patch
    int                               w_frames = 0; // the frames per call w_patch is filled for
    std::thread                       pairs_bg;     // builds the frame pairs' weight while the model loads
    bool                              wbf16 = false;  // BF16 banks (the accuracy variant): no NVFP4 codes/scales

    static bool is_weight(const std::string & n);
    // a weight's buffer (shared)
    axera_cmm & io(const char * n) {
        GGML_ASSERT(is_weight(n));
        auto & c = wbuf[n];
        if (!c) {
            c = axera_cmm_alloc(k.in_bytes(n));
        }
        return c;
    }
    // any IO of kernel v: a weight's shared buffer or v's own
    axera_cmm & io(sized & v, const char * n, bool out = false) {
        if (!out && is_weight(n)) {
            return io(n);
        }
        auto & c = v.buf[n];
        if (!c) {
            c = axera_cmm_alloc(out ? v.k->out_bytes(n) : v.k->in_bytes(n));
        }
        return c;
    }
    void weights(int L, int H);
    void weights_bf16(int L);
    void patch_weight(int frames);
    void tables(sized & v, int gh, int gw, int n, const ggml_tensor * pe, int H);
};

// the kernel inputs filled from the mmproj once (weights()) or per frame count (patch_weight())
bool lightnav_vit_full::is_weight(const std::string & n) {
    static const char * w[] = {
        "ln1_w", "ln1_b", "w_qkv", "w_qkv_s", "w_qkv_s2", "b_qkv", "w_o", "w_o_s", "w_o_s2", "b_o", "ln2_w", "ln2_b",
        "w_up", "w_up_s", "w_up_s2", "b_up", "w_down", "w_down_s", "w_down_s2", "b_down", "b_patch", "post_w", "post_b",
        "w0", "w0_s", "w0_s2", "b0", "w2", "w2_s", "w2_s2", "b2", "ds_post_w", "ds_post_b", "ds_w0", "ds_w0_s",
        "ds_w0_s2", "ds_b0", "ds_w2", "ds_w2_s", "ds_w2_s2", "ds_b2", "w_patch", "w_patch_s", "w_patch_s2",
    };
    for (const char * x : w) {
        if (n == x) {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> vit_roles(const manifest & m) {
    std::vector<std::string> r = { "vit" };
    for (const auto & [role, kd] : m.kernels) {
        if (role.rfind("vit_", 0) == 0) {
            r.push_back(role);
        }
    }
    return r;
}

lightnav_vit_full::lightnav_vit_full(const manifest & m) :
    man(m),
    ks([&] {
        std::vector<sized> v;
        for (const std::string & role : vit_roles(m)) {
            sized s;
            s.k    = &m.get(role);
            s.role = role;
            const AX_ENGINE_IOMETA_T & km = s.k->info->pInputs[s.k->input("kmask")];  // [N / BC, QB, BC]
            const AX_ENGINE_IOMETA_T & cm = s.k->info->pInputs[s.k->input("cos")];    // [N, HD / 2]
            GGML_ASSERT(km.nShapeSize == 3 && cm.nShapeSize == 2);
            s.N  = (int) cm.pShape[0];
            s.QB = (int) km.pShape[1];
            s.BC = (int) km.pShape[2];
            GGML_ASSERT(km.pShape[0] * s.BC == s.N);
            v.push_back(std::move(s));
        }
        std::sort(v.begin(), v.end(), [](const sized & a, const sized & b) { return a.N < b.N; });
        return v;
    }()),
    k(*ks[0].k) {
    HD = 2 * (int) ks[0].k->info->pInputs[ks[0].k->input("cos")].pShape[1];
    wbf16 = ks[0].k->find_input("w_qkv_s") < 0;
    for (const sized & v : ks) {
        GGML_ASSERT((v.k->find_input("w_qkv_s") < 0) == wbf16 && "axera lightnav_vit_full: mixed weight precisions");
        GGML_ASSERT(2 * (int) v.k->info->pInputs[v.k->input("cos")].pShape[1] == HD);
        GGML_ASSERT(v.k->in_bytes("px") / v.N == k.in_bytes("px") / ks[0].N && "axera lightnav_vit_full: the vit kernels' patch rows differ");
        for (uint32_t i = 0; i < v.k->info->nInputSize; ++i) {
            const char * n = v.k->info->pInputs[i].pName;
            if (is_weight(n) && v.k->in_bytes(n) != k.in_bytes(n)) {
                GGML_ABORT("axera lightnav_vit_full: %s: weight %s differs from %s's\n", v.role.c_str(), n, ks[0].role.c_str());
            }
        }
        if (ks.size() > 1) {
            GGML_LOG_INFO("axera lightnav_vit_full: role %s: %d patches\n", v.role.c_str(), v.N);
        }
    }
}

// bf16 rows of a weight of any type into dst (ne elements)
void to_bf16(const ggml_tensor * w, ggml_bf16_t * dst) {
    const int64_t ne = ggml_nelements(w);
    if (w->type == GGML_TYPE_BF16) {
        memcpy(dst, w->data, ne * sizeof(ggml_bf16_t));
    } else if (w->type == GGML_TYPE_F32) {
        ggml_fp32_to_bf16_row((const float *) w->data, dst, ne);
    } else {
        std::vector<float> f(ne);
        ggml_get_type_traits(w->type)->to_float(w->data, f.data(), ne);
        ggml_fp32_to_bf16_row(f.data(), dst, ne);
    }
}

void lightnav_vit_full::weights(int L, int H) {
    axera_timer tm("vit weights (load)");
    if (wbf16) {
        weights_bf16(L);
    } else {
    // the projections: NVFP4 codes, E4M3 group scales and one FP32 scale2 a layer, stacked [layers, ...]
    static const char * nv_q[] = { "w_qkv", "w_o", "w_up", "w_down" };
    static const char * nv_t[] = { "attn_qkv", "attn_out", "ffn_up", "ffn_down" };
    static const char * nv_b[] = { "b_qkv", "b_o", "b_up", "b_down" };
    for (int j = 0; j < 4; ++j) {
        axera_cmm &        cq    = io(nv_q[j]);
        axera_cmm &        cs    = io((std::string(nv_q[j]) + "_s").c_str());
        axera_cmm &        cz    = io((std::string(nv_q[j]) + "_s2").c_str());
        axera_cmm &        cb    = io(nv_b[j]);
        const std::string  sn    = std::string(nv_q[j]) + "_s", zn = sn + "2";
        const size_t       q_log = k.in_bytes(nv_q[j]), s_log = k.in_bytes(sn.c_str());
        const size_t       b_log = k.in_bytes(nv_b[j]), z_log = k.in_bytes(zn.c_str());
        const size_t       q_per = q_log / L, s_per = s_log / L, b_per = b_log / L;
        std::vector<float> z(L);
        if (q_per * L != q_log || s_per * L != s_log || b_per * L != b_log || z_log != L * sizeof(float)) {
            GGML_ABORT("axera lightnav_vit_full: weights L=%d H=%d %s q=%zu s=%zu b=%zu z=%zu\n", L, H, nv_q[j], q_log,
                       s_log, b_log, z_log);
        }
        axera_host_pool().run(L, [&](int64_t l) {
            const std::string   p = "v.blk." + std::to_string(l) + ".";
            const ggml_tensor * w = names.at(p + nv_t[j] + ".weight");
            const int64_t       K = w->ne[0], N = ggml_nrows(w);
            GGML_ASSERT((size_t) N * (K / 2) == q_per && (size_t) N * (K / 16) == s_per);
            axera_page_in(w);
            z[l] = axera_quantize_nvfp4_slice(w, 0, N, (uint8_t *) cq->vir + (size_t) l * q_per, K / 2,
                                              (uint8_t *) cs->vir + (size_t) l * s_per, K / 16);
            axera_drop_mapped(w);  // (an mmproj mapped by mtmd: the copy above is the only one kept)
            const ggml_tensor * b = names.at(p + nv_t[j] + ".bias");
            GGML_ASSERT((size_t) ggml_nelements(b) * sizeof(ggml_bf16_t) == b_per);
            axera_page_in(b);
            to_bf16(b, (ggml_bf16_t *) ((char *) cb->vir + (size_t) l * b_per));
            axera_drop_mapped(b);
        });
        cq->flush();
        cs->flush();
        cb->flush();
        std::copy(z.begin(), z.end(), (float *) cz->vir);
        cz->flush();
    }
    // the layer norms, stacked bf16
    static const std::pair<const char *, const char *> per_layer[] = {
        { "ln1_w", "ln1.weight" }, { "ln1_b", "ln1.bias" }, { "ln2_w", "ln2.weight" }, { "ln2_b", "ln2.bias" },
    };
    for (const auto & [in, t] : per_layer) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / L;
        GGML_ASSERT(per * L == k.in_bytes(in));
        axera_host_pool().run(L, [&](int64_t l) {
            const ggml_tensor * w = names.at("v.blk." + std::to_string(l) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + l * per));
            axera_drop_mapped(w);
        });
        c->flush();
    }
    // the patch projection's bias and the post-layer norm, in bf16
    static const std::pair<const char *, const char *> whole[] = {
        { "b_patch", "v.patch_embd.bias" }, { "post_w", "v.post_ln.weight" }, { "post_b", "v.post_ln.bias" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        axera_page_in(w);
        to_bf16(w, (ggml_bf16_t *) io(in)->vir);
        io(in)->flush();
        axera_drop_mapped(w);
    }
    // the main merger (one tensor): NVFP4 weight and bf16 bias
    static const std::pair<const char *, const char *> merger[] = { { "w0", "mm.0" }, { "w2", "mm.2" } };
    for (const auto & [q, t] : merger) {
        axera_cmm &        cq = io(q);
        axera_cmm &        cs = io((std::string(q) + "_s").c_str());
        axera_cmm &        cz = io((std::string(q) + "_s2").c_str());
        const ggml_tensor * w = names.at(t + std::string(".weight"));
        const int64_t       K = w->ne[0], N = ggml_nrows(w);
        const std::string sn = std::string(q) + "_s", zn = sn + "2";
        GGML_ASSERT((size_t) N * (K / 2) == k.in_bytes(q) && (size_t) N * (K / 16) == k.in_bytes(sn.c_str()) &&
                    k.in_bytes(zn.c_str()) == sizeof(float));
        axera_page_in(w);
        *(float *) cz->vir = axera_quantize_nvfp4_slice(w, 0, N, (uint8_t *) cq->vir, K / 2, (uint8_t *) cs->vir, K / 16);
        axera_drop_mapped(w);
        cq->flush();
        cs->flush();
        cz->flush();
        const ggml_tensor * b  = names.at(t + std::string(".bias"));
        const std::string   bn = std::string("b") + std::string(q).substr(1);
        GGML_ASSERT((size_t) ggml_nelements(b) * sizeof(ggml_bf16_t) == k.in_bytes(bn.c_str()));
        axera_page_in(b);
        to_bf16(b, (ggml_bf16_t *) io(bn.c_str())->vir);
        io(bn.c_str())->flush();
        axera_drop_mapped(b);
    }
    // Qwen3-VL adds the merged features of vision layers 5 / 11 / 17 to the
    // language model's first three layers. The kernel stacks the three
    // mergers' tensors over that order.
    static const int ds_layer[3] = { 5, 11, 17 };
    static const std::pair<const char *, const char *> ds[] = {
        { "ds_post_w", "norm.weight" }, { "ds_post_b", "norm.bias" },
        { "ds_b0", "fc1.bias" },        { "ds_b2", "fc2.bias" },
    };
    for (const auto & [in, t] : ds) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / 3;
        GGML_ASSERT(c->size >= 3 * per);
        axera_host_pool().run(3, [&](int64_t j) {
            const ggml_tensor * w = names.at("v.deepstack." + std::to_string(ds_layer[j]) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + j * per));
            axera_drop_mapped(w);
        });
        c->flush();
    }
    static const std::pair<const char *, const char *> ds_nv[] = { { "ds_w0", "fc1.weight" }, { "ds_w2", "fc2.weight" } };
    for (const auto & [q, t] : ds_nv) {
        axera_cmm &  cq = io(q);
        axera_cmm &  cs = io((std::string(q) + "_s").c_str());
        axera_cmm &  cz = io((std::string(q) + "_s2").c_str());
        const std::string sn = std::string(q) + "_s", zn = sn + "2";
        const size_t      q_per = k.in_bytes(q) / 3, s_per = k.in_bytes(sn.c_str()) / 3;
        GGML_ASSERT(q_per * 3 == k.in_bytes(q) && s_per * 3 == k.in_bytes(sn.c_str()) && k.in_bytes(zn.c_str()) == 3 * sizeof(float));
        float           z[3];
        axera_host_pool().run(3, [&](int64_t j) {
            const ggml_tensor * w = names.at("v.deepstack." + std::to_string(ds_layer[j]) + "." + t);
            const int64_t       K = w->ne[0], N = ggml_nrows(w);
            GGML_ASSERT((size_t) N * (K / 2) == q_per && (size_t) N * (K / 16) == s_per);
            axera_page_in(w);
            z[j] = axera_quantize_nvfp4_slice(w, 0, N, (uint8_t *) cq->vir + (size_t) j * q_per, K / 2,
                                              (uint8_t *) cs->vir + (size_t) j * s_per, K / 16);
            axera_drop_mapped(w);
        });
        cq->flush();
        cs->flush();
        std::copy(z, z + 3, (float *) cz->vir);
        cz->flush();
    }
    }
    axera_flush_file_pages();
    axera_keep_mapped(names.at("v.position_embd.weight"));  // (tables() reads it per image size)
    // the patch projection [H][3][PS][PS] x 2 frames: w_patch itself is filled per frame count (patch_weight)
    const ggml_tensor * w0 = names.at("v.patch_embd.weight"), *w1 = names.at("v.patch_embd.weight.1");
    const int           CP = (int) (ggml_nelements(w0) / H);
    const int           PW = (int) (k.in_bytes("px") / sizeof(ggml_bf16_t) / ks[0].N);
    GGML_ASSERT(w0->type == GGML_TYPE_F32 && w1->type == GGML_TYPE_F32 && ggml_nelements(w0) == ggml_nelements(w1) &&
                ((wbf16 && (size_t) H * PW * sizeof(ggml_bf16_t) == k.in_bytes("w_patch")) ||
                 (!wbf16 && (size_t) H * PW / 2 == k.in_bytes("w_patch") &&
                  (size_t) H * PW / 16 == k.in_bytes("w_patch_s") && k.in_bytes("w_patch_s2") == sizeof(float))));
    axera_page_in(w0);
    axera_page_in(w1);
    if (pairs_bg.joinable()) {
        pairs_bg.join();
    }
    const bool ok = pemb.init((const float *) w0->data, (const float *) w1->data, H, CP, PW);
    GGML_ASSERT(ok && "axera lightnav_vit_full: the kernel's px row is neither one nor two frames wide");
    w_frames = 0;
    // a video's first frame pair would otherwise wait seconds for the pairs' weight: build it now, on threads of its
    // own (the host pool is the loader's and the prompts')
    pairs_bg = std::thread([this] { pemb.prepare_pairs(patch_embed::threads((int) std::max(1u, std::thread::hardware_concurrency()))); });
}

// BF16 weight banks: the same IO contract as the NVFP4 kernel, filled with one BF16 copy of each mmproj tensor
void lightnav_vit_full::weights_bf16(int L) {
    static const char * pw[] = { "w_qkv", "w_o", "w_up", "w_down" };
    static const char * pt[] = { "attn_qkv", "attn_out", "ffn_up", "ffn_down" };
    static const char * pb[] = { "b_qkv", "b_o", "b_up", "b_down" };
    for (int j = 0; j < 4; ++j) {
        axera_cmm &  cq = io(pw[j]), &cb = io(pb[j]);
        const size_t per = k.in_bytes(pw[j]) / L, bper = k.in_bytes(pb[j]) / L;
        GGML_ASSERT(per * L == k.in_bytes(pw[j]) && bper * L == k.in_bytes(pb[j]));
        axera_host_pool().run(L, [&](int64_t l) {
            const std::string   p = "v.blk." + std::to_string(l) + ".";
            const ggml_tensor * w = names.at(p + pt[j] + ".weight");
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) cq->vir + (size_t) l * per));
            axera_drop_mapped(w);
            const ggml_tensor * b = names.at(p + pt[j] + ".bias");
            GGML_ASSERT((size_t) ggml_nelements(b) * sizeof(ggml_bf16_t) == bper);
            axera_page_in(b);
            to_bf16(b, (ggml_bf16_t *) ((char *) cb->vir + (size_t) l * bper));
            axera_drop_mapped(b);
        });
        cq->flush();
        cb->flush();
    }
    static const std::pair<const char *, const char *> per_layer[] = {
        { "ln1_w", "ln1.weight" }, { "ln1_b", "ln1.bias" }, { "ln2_w", "ln2.weight" }, { "ln2_b", "ln2.bias" },
    };
    for (const auto & [in, t] : per_layer) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / L;
        GGML_ASSERT(per * L == k.in_bytes(in));
        axera_host_pool().run(L, [&](int64_t l) {
            const ggml_tensor * w = names.at("v.blk." + std::to_string(l) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + l * per));
            axera_drop_mapped(w);
        });
        c->flush();
    }
    static const std::pair<const char *, const char *> whole[] = {
        { "b_patch", "v.patch_embd.bias" }, { "post_w", "v.post_ln.weight" }, { "post_b", "v.post_ln.bias" },
    };
    for (const auto & [in, t] : whole) {
        const ggml_tensor * w = names.at(t);
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(in));
        axera_page_in(w);
        to_bf16(w, (ggml_bf16_t *) io(in)->vir);
        io(in)->flush();
        axera_drop_mapped(w);
    }
    static const std::pair<const char *, const char *> merger[] = { { "w0", "mm.0" }, { "w2", "mm.2" } };
    for (const auto & [q, t] : merger) {
        axera_cmm &         c = io(q);
        const ggml_tensor * w = names.at(t + std::string(".weight"));
        GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == k.in_bytes(q));
        axera_page_in(w);
        to_bf16(w, (ggml_bf16_t *) c->vir);
        c->flush();
        axera_drop_mapped(w);
        const ggml_tensor * b  = names.at(t + std::string(".bias"));
        const std::string   bn = std::string("b") + std::string(q).substr(1);
        GGML_ASSERT((size_t) ggml_nelements(b) * sizeof(ggml_bf16_t) == k.in_bytes(bn.c_str()));
        axera_page_in(b);
        to_bf16(b, (ggml_bf16_t *) io(bn.c_str())->vir);
        io(bn.c_str())->flush();
        axera_drop_mapped(b);
    }
    static const int ds_layer[3] = { 5, 11, 17 };
    static const std::pair<const char *, const char *> ds[] = {
        { "ds_post_w", "norm.weight" }, { "ds_post_b", "norm.bias" },
        { "ds_b0", "fc1.bias" },        { "ds_b2", "fc2.bias" },
    };
    for (const auto & [in, t] : ds) {
        axera_cmm &  c   = io(in);
        const size_t per = k.in_bytes(in) / 3;
        GGML_ASSERT(c->size >= 3 * per);
        axera_host_pool().run(3, [&](int64_t j) {
            const ggml_tensor * w = names.at("v.deepstack." + std::to_string(ds_layer[j]) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + j * per));
            axera_drop_mapped(w);
        });
        c->flush();
    }
    static const std::pair<const char *, const char *> ds_w[] = { { "ds_w0", "fc1.weight" }, { "ds_w2", "fc2.weight" } };
    for (const auto & [q, t] : ds_w) {
        axera_cmm &  c   = io(q);
        const size_t per = k.in_bytes(q) / 3;
        GGML_ASSERT(per * 3 == k.in_bytes(q));
        axera_host_pool().run(3, [&](int64_t j) {
            const ggml_tensor * w = names.at("v.deepstack." + std::to_string(ds_layer[j]) + "." + t);
            GGML_ASSERT((size_t) ggml_nelements(w) * sizeof(ggml_bf16_t) == per);
            axera_page_in(w);
            to_bf16(w, (ggml_bf16_t *) ((char *) c->vir + j * per));
            axera_drop_mapped(w);
        });
        c->flush();
    }
}

void lightnav_vit_full::patch_weight(int frames) {
    if (frames == w_frames) {
        return;
    }
    axera_timer                tm(frames == 2 ? "vit patch weight (frame pairs)" : "vit patch weight (stills)");
    const std::vector<float> & w = frames == 2 ? pemb.weight_pairs([](int64_t n, const std::function<void(int64_t)> & fn) {
        axera_host_pool().run(n, fn);
    }) : pemb.weight_still();
    // the kernel's patch projection: rows H, PW-wide pixel rows
    const int64_t H = names.at("v.patch_embd.bias")->ne[0];
    const int64_t K = wbf16 ? (int64_t) io("w_patch")->size / sizeof(ggml_bf16_t) / H
                            : (int64_t) io("w_patch")->size * 2 / H;
    GGML_ASSERT((size_t) w.size() == (size_t) H * K);
    if (wbf16) {
        GGML_ASSERT((size_t) H * K * sizeof(ggml_bf16_t) == k.in_bytes("w_patch"));
        ggml_fp32_to_bf16_row(w.data(), (ggml_bf16_t *) io("w_patch")->vir, (int64_t) H * K);
        io("w_patch")->flush();
        w_frames = frames;
        return;
    }
    GGML_ASSERT(k.in_bytes("w_patch_s") == (size_t) H * K / 16);
    ggml_tensor t{};
    t.type  = GGML_TYPE_F32;
    t.ne[0] = K;
    t.ne[1] = H;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float) * K;
    t.nb[2] = t.nb[3] = t.nb[1] * H;
    t.data  = (void *) w.data();
    *(float *) io("w_patch_s2")->vir = axera_quantize_nvfp4_slice(&t, 0, H, (uint8_t *) io("w_patch")->vir, K / 2,
                                                                  (uint8_t *) io("w_patch_s")->vir, K / 16);
    io("w_patch")->flush();
    io("w_patch_s")->flush();
    io("w_patch_s2")->flush();
    w_frames = frames;
    if (frames == 2) {
        GGML_LOG_INFO("axera lightnav_vit_full: video frame pairs through %d-wide patch rows (basis residual %.1e)\n", pemb.width(),
                      pemb.pair_residual());
    }
}

// the tables of an image of gh x gw patches (n real): merge order, positions, RoPE, key mask
void lightnav_vit_full::tables(sized & v, int gh, int gw, int n, const ggml_tensor * pe, int H) {
    axera_timer tm("vit tables (cpu)");
    const int N = v.N, QB = v.QB, BC = v.BC;
    std::vector<int> & py = v.py, & px = v.px;
    py.assign(N, 0);
    px.assign(N, 0);
    for (int y = 0, i = 0; y < gh; y += 2) {
        for (int x = 0; x < gw; x += 2) {
            for (int d = 0; d < 4; ++d, ++i) {
                py[i] = y + d / 2;
                px[i] = x + d % 2;
            }
        }
    }
    // learned positions [side x side][H] (x fastest), bilinear with align corners to gh x gw
    const int     side = (int) std::lround(std::sqrt((double) pe->ne[1]));
    const float * tab  = (const float *) pe->data;
    auto *        pos  = (ggml_bf16_t *) io(v, "pos")->vir;
    GGML_ASSERT(pe->type == GGML_TYPE_F32 && side * side == pe->ne[1]);
    axera_host_pool().run(N, [&](int64_t i) {
        ggml_bf16_t * d = pos + i * H;
        if (i >= n) {
            std::fill(d, d + H, ggml_fp32_to_bf16(0.0f));
            return;
        }
        const float   sy = gh > 1 ? (float) py[i] * (side - 1) / (gh - 1) : 0.0f;
        const float   sx = gw > 1 ? (float) px[i] * (side - 1) / (gw - 1) : 0.0f;
        const int     y0 = (int) sy, x0 = (int) sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
        const float   fy = sy - y0, fx = sx - x0;
        const float * a = tab + ((int64_t) y0 * side + x0) * H, *b = tab + ((int64_t) y0 * side + x1) * H;
        const float * c = tab + ((int64_t) y1 * side + x0) * H, *e = tab + ((int64_t) y1 * side + x1) * H;
        for (int j = 0; j < H; ++j) {
            d[j] = ggml_fp32_to_bf16((1 - fy) * ((1 - fx) * a[j] + fx * b[j]) + fy * ((1 - fx) * c[j] + fx * e[j]));
        }
    });
    io(v, "pos")->flush();
    // RoPE: frequency j of HD / 4 for the row, then HD / 4 for the column, 10000^(-4 j / HD)
    std::vector<float> inv(HD / 4);
    for (int q = 0; q < HD / 4; ++q) {
        inv[q] = (float) std::pow(10000.0, -4.0 * q / HD);
    }
    auto * cs = (ggml_bf16_t *) io(v, "cos")->vir, *sn = (ggml_bf16_t *) io(v, "sin")->vir;
    axera_host_pool().run(N, [&](int64_t i) {
        for (int j = 0; j < HD / 2; ++j) {
            const float a = (float) (j < HD / 4 ? py[i] : px[i]) * inv[j % (HD / 4)];
            cs[i * (HD / 2) + j] = ggml_fp32_to_bf16(std::cos(a));
            sn[i * (HD / 2) + j] = ggml_fp32_to_bf16(std::sin(a));
        }
    });
    io(v, "cos")->flush();
    io(v, "sin")->flush();
    // keys: the n patches in nblk blocks, the padding of the last hidden ([N / BC][QB][BC], the key row repeated)
    *(int32_t *) io(v, "nblk")->vir = (n + BC - 1) / BC;
    io(v, "nblk")->flush();
    auto * km = (ggml_bf16_t *) io(v, "kmask")->vir;
    axera_host_pool().run(N / BC, [&](int64_t b) {
        std::vector<ggml_bf16_t> row(BC);
        for (int c = 0; c < BC; ++c) {
            row[c] = ggml_fp32_to_bf16(b * BC + c < n ? 0.0f : -3e38f);
        }
        for (int r = 0; r < QB; ++r) {
            memcpy(km + ((size_t) b * QB + r) * BC, row.data(), BC * sizeof(ggml_bf16_t));
        }
    });
    io(v, "kmask")->flush();
}

// mtmd opened the model: load the weights now (seconds of file reads and conversion), not inside the first image
void lightnav_vit_full::prewarm(ggml_cgraph * g) {
    names.of(g);
    if (!names.find("v.position_embd.weight") || !names.find("v.patch_embd.weight") || !names.find("v.blk.0.ln1.weight")) {
        return;
    }
    const int H = (int) names.at("v.position_embd.weight")->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0];
    int       L = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    const ggml_tensor * w0 = names.at("v.blk.0.ln1.weight");
    if (!w0->data || bound == w0->data) {
        return;
    }
    GGML_ASSERT(k.in_bytes("px") == (size_t) ks[0].N * 3 * PS * PS * sizeof(ggml_bf16_t) ||
                k.in_bytes("px") == (size_t) ks[0].N * 2 * 3 * PS * PS * sizeof(ggml_bf16_t));
    weights(L, H);
    bound = w0->data;
    for (sized & s : ks) {
        s.gh_ = s.gw_ = -1;
    }
}

ggml_status lightnav_vit_full::compute(ggml_cgraph * g) {
    names.of(g);
    ggml_tensor * raw = names.at("inp_raw");  // [W, H, 3, frames] f32, normalized
    ggml_tensor * pe  = names.at("v.position_embd.weight");
    ggml_tensor * out = g->nodes[g->n_nodes - 1];
    const int     H = (int) pe->ne[0], PS = (int) names.at("v.patch_embd.weight")->ne[0], CP = 3 * PS * PS;
    const int     gw = (int) raw->ne[0] / PS, gh = (int) raw->ne[1] / PS, n = gw * gh;
    int           L  = 0;
    while (names.find("v.blk." + std::to_string(L) + ".ln1.weight")) {
        ++L;
    }
    const int frames = (int) raw->ne[3];  // 1: a still image, 2: a pair of video frames
    const int NMAX   = ks.back().N;
    if (frames < 1 || frames > 2 || gw % 2 || gh % 2 || n > NMAX) {
        GGML_LOG_ERROR("axera lightnav_vit_full: %d frame(s) of %d x %d patches: one image or a pair of video frames, even patch "
                       "counts, at most %d patches (mtmd --image-max-tokens)\n", frames, gw, gh, NMAX);
        return GGML_STATUS_FAILED;
    }
    sized & v = *std::find_if(ks.begin(), ks.end(), [&](const sized & s) { return n <= s.N; });
    const kernel & kv = *v.k;
    const int      N  = v.N;
    const std::vector<int> & py = v.py, & px = v.px;
    const ggml_tensor * w0 = names.at("v.blk.0.ln1.weight");
    if (bound != w0->data) {
        weights(L, H);
        bound = w0->data;
        for (sized & s : ks) {
            s.gh_ = s.gw_ = -1;
        }
    }
    if (gh != v.gh_ || gw != v.gw_) {
        tables(v, gh, gw, n, pe, H);
        v.gh_ = gh;
        v.gw_ = gw;
    }
    patch_weight(frames);
    {
        axera_timer   tm(frames == 2 ? "vit patches (cpu, frame pairs)" : "vit patches (cpu)");
        const float * img = (const float *) raw->data;
        const int64_t W = raw->ne[0], HI = raw->ne[1];
        const int     PW = pemb.width();
        const bool    plain = frames == 1 && PW == CP;  // the pixels are the row
        auto *        pxb = (ggml_bf16_t *) io(v, "px")->vir;
        axera_host_pool().run(N, [&](int64_t i) {
            ggml_bf16_t * d = pxb + i * PW;
            if (i >= n) {
                std::fill(d, d + PW, ggml_fp32_to_bf16(0.0f));
                return;
            }
            if (plain) {
                for (int c = 0; c < 3; ++c) {
                    for (int ky = 0; ky < PS; ++ky) {
                        const float * row = img + c * W * HI + (int64_t) (py[i] * PS + ky) * W + px[i] * PS;
                        ggml_fp32_to_bf16_row(row, d + (c * PS + ky) * PS, PS);
                    }
                }
                return;
            }
            std::vector<float> p((size_t) 2 * CP + PW);  // the patch in both frames, its row
            for (int f = 0; f < frames; ++f) {
                for (int c = 0; c < 3; ++c) {
                    for (int ky = 0; ky < PS; ++ky) {
                        const float * row = img + (f * 3 + c) * W * HI + (int64_t) (py[i] * PS + ky) * W + px[i] * PS;
                        memcpy(p.data() + (size_t) f * CP + (c * PS + ky) * PS, row, PS * sizeof(float));
                    }
                }
            }
            pemb.row(p.data(), frames == 2 ? p.data() + CP : nullptr, p.data() + 2 * CP);
            ggml_fp32_to_bf16_row(p.data() + 2 * CP, d, PW);
        });
        io(v, "px")->flush();
    }
    std::vector<AX_ENGINE_IO_BUFFER_T> in(kv.info->nInputSize), outs(kv.info->nOutputSize);
    for (uint32_t i = 0; i < kv.info->nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = kv.info->pInputs[i];
        in[i]                        = axera_io(io(v, m.pName), m.nSize);
    }
    for (uint32_t i = 0; i < kv.info->nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T & m = kv.info->pOutputs[i];
        outs[i]                      = axera_io(io(v, m.pName, true), m.nSize);
    }
    {
        axera_timer tm("vit (one call)");
        kv.run(in, outs, v.role.c_str());
    }
    axera_cmm &  o     = io(v, "out", true);
    const int    OUTD  = (int) (kv.out_bytes("out") / sizeof(float) / (N / 4));
    const int    OUTM  = (int) names.at("mm.2.weight")->ne[1];
    GGML_ASSERT(OUTD == 4 * OUTM && "lightnav ViT kernel must carry the main plane and three deepstack planes");
    const size_t bytes = (size_t) (n / 4) * OUTD * sizeof(float);
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->ne[0] == OUTD && out->ne[1] == n / 4);
    o->invalidate(o->vir, bytes);
    memcpy(out->data, o->vir, bytes);
    if (ggml_backend_buffer_is_axera(out->buffer)) {
        ((ggml_backend_axera_buffer_context *) out->buffer->context)->flush(out->data, bytes);
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

std::shared_ptr<model> make_lightnav_vit_full(const manifest & m) {
    if (m.get("vit").find_input("pixel") >= 0) {
        return std::make_shared<lightnav_vit_pulsar>(m);
    }
    if (m.get("vit").find_input("frames") >= 0) {
        return std::make_shared<lightnav_vit_pulsar_u8>(m);
    }
    return std::make_shared<lightnav_vit_full>(m);
}

}  // namespace axera
