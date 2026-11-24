// Axera NPU backend: the prefill weight banks of the Qwen3.5 / Qwen3.8 hybrid whole-model kernels (blocks of three
// gated delta-net layers and one gated attention layer, dense MLP): every bank input b<phase>_<attn|mlp>_<name>
// [blocks, ...] of a prefill kernel, block b holding layer 4 * b + phase, from the GGUF. One copy for the 27B and the
// 0.8B - 9B adapters.
#pragma once

#include "ops.h"

#include <map>
#include <string>

namespace axera {

// rows of an [N, K] weight of the GGUF in the kernels' MX layout: MXFP8 codes [K] or MXFP4 codes [K / 2], and scales
// [K / 32]. A matching MXFP4 GGUF tensor is converted row by row as read; every other type (a community GGUF: Q4_K,
// Q8_0, ...) is quantized once, in the format the prefill kernel's bank IO selected.
struct mx_weight {
    const ggml_tensor * w;
    axera_cmm           q, s;
    bool                fp8 = false;
    int64_t             N = 0, K = 0;
    mx_weight(const ggml_tensor * w, bool fp8 = false) : w(w), fp8(fp8), N(w->ne[1]), K(w->ne[0]) {
        if (fp8) {
            axera_quantize_mxfp8(w, q, s);
        } else if (w->type != GGML_TYPE_MXFP4) {
            axera_quantize_mxfp4(w, q, s);
        }
    }
    // row r: its codes or its scales into dst
    void row(int64_t r, bool scales, uint8_t * dst) const {
        const size_t code = fp8 ? K : K / 2;
        if (q) {
            memcpy(dst, scales ? (const uint8_t *) s->vir + r * (K / 32) : (const uint8_t *) q->vir + r * code,
                   scales ? K / 32 : code);
            return;
        }
        GGML_ASSERT(!fp8 && w->type == GGML_TYPE_MXFP4);
        thread_local std::vector<uint8_t> other;
        other.resize(scales ? K / 2 : K / 32);
        const uint8_t * src = (const uint8_t *) w->data + r * w->nb[1];
        axera_mxfp4_rows(src, 1, K, scales ? other.data() : dst, scales ? dst : other.data());
    }
};

// the model sizes a bank depends on
struct gdn_bank_shape {
    const char * who = "";          // the adapter (messages)
    int          L   = 0;           // layers
    int          ROT = 0;           // rotary dims of the attention layers
    int          KH = 16, VH = 48;  // delta-net key / value heads
    bool         fp8 = false;       // MXFP8 projections (else MXFP4)
    // GGUF's tiled value head of HF (grouped) head g
    int tiled(int g) const {
        const int rep = VH / KH;
        return (g % rep) * KH + g / rep;
    }
};

// the bank of prefill input d: a CMM buffer [d.bytes], filled and flushed
axera_cmm gdn_prefill_bank(const gdn_bank_shape & s, const std::map<std::string, ggml_tensor *> & t, const io_desc & d,
                           const rope & rp);

}  // namespace axera
