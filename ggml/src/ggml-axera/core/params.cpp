// Axera NPU backend core: layer parameters from the GGUF tensors, and the packs of packed kernels

#include "params.h"

#include "ggml-axera.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace axera {

static ggml_tensor * tensor_of(const std::map<std::string, ggml_tensor *> & t, int il, const std::string & name) {
    const std::string key = "blk." + std::to_string(il) + "." + name;
    auto              it  = t.find(key);
    if (it == t.end()) {
        GGML_ABORT("axera: tensor %s not in the graph\n", key.c_str());
    }
    return it->second;
}

// a whole-model tensor (parameter kind "out": the output head's): the first of "a|b|..." in the graph (output.weight,
// or the tied token_embd.weight); nullptr if none (a graph without the output head: pooled embeddings)
static ggml_tensor * model_tensor(const std::map<std::string, ggml_tensor *> & t, const std::string & names) {
    size_t a = 0;
    while (a <= names.size()) {
        const size_t b  = std::min(names.find('|', a), names.size());
        auto         it = t.find(names.substr(a, b - a));
        if (it != t.end()) {
            return it->second;
        }
        a = b + 1;
    }
    return nullptr;
}

// experts in the GGUF file mapping (read only) get a CMM copy, once per tensor (shared by the model's contexts): a
// layer's gate and up back to back in one buffer, down in another
static void experts_to_cmm(const ggml_tensor * w[3]) {
    static std::mutex                                 mu;
    static std::map<const ggml_tensor *, ggml_tensor> copy;
    std::lock_guard<std::mutex>                       lock(mu);
    for (auto [g0, g1] : { std::pair<int, int>{ 0, 2 }, std::pair<int, int>{ 2, 3 } }) {
        if (ggml_backend_buffer_is_axera(w[g0]->buffer)) {
            continue;
        }
        if (!copy.count(w[g0])) {
            size_t bytes = 0;
            for (int g = g0; g < g1; ++g) {
                bytes += ggml_nbytes(w[g]);
            }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_axera_buffer_type(), bytes);
            if (!buf) {
                GGML_ABORT("axera: no CMM for a copy of %s (%zu bytes)\n", w[g0]->name, bytes);
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            char * p = (char *) ggml_backend_buffer_get_base(buf);
            for (int g = g0; g < g1; ++g) {
                ggml_tensor c = *w[g];
                c.buffer      = buf;
                c.data        = p;
                c.view_src    = nullptr;
                ggml_backend_tensor_set(&c, w[g]->data, 0, ggml_nbytes(w[g]));
                axera_drop_mapped(w[g]);
                copy.emplace(w[g], c);
                p += ggml_nbytes(w[g]);
            }
        }
        for (int g = g0; g < g1; ++g) {
            w[g] = &copy.at(w[g]);
        }
    }
}

// experts [n][codes | scales] -> planar, in place: all n experts' codes, then all their scales; once per tensor
static void mxfp4_planar(ggml_backend_axera_buffer_context * b, char * base, int64_t n, size_t data, size_t scale) {
    static std::mutex             mu;
    static std::set<const void *> done;
    std::lock_guard<std::mutex>   lock(mu);
    if (!done.insert(base).second) {
        return;
    }
    const size_t      stride = data + scale;
    std::vector<char> tmp((size_t) n * stride);
    memcpy(tmp.data(), base, tmp.size());
    axera_host_pool().run(n, [&](int64_t e) {
        memcpy(base + e * data, tmp.data() + e * stride, data);
        memcpy(base + n * data + e * scale, tmp.data() + e * stride + data, scale);
    });
    b->flush(base, tmp.size());
}

const params::experts & params::experts_of(const std::map<std::string, ggml_tensor *> & t, int il) {
    auto it = exps.find(il);
    if (it != exps.end()) {
        return it->second;
    }
    // gate and up (GGUF ffn_gate_exps, ffn_up_exps) as one gate-up weight per expert, down
    const ggml_tensor * w[3] = { tensor_of(t, il, "ffn_gate_exps.weight"), tensor_of(t, il, "ffn_up_exps.weight"),
                                 tensor_of(t, il, "ffn_down_exps.weight") };
    experts_to_cmm(w);
    for (const ggml_tensor * x : w) {
        if (x->type != GGML_TYPE_MXFP4 || !ggml_backend_buffer_is_axera(x->buffer)) {
            GGML_ABORT("axera: %s is %s; the MoE kernels take MXFP4 experts (llama-quantize MXFP4_MOE)\n", x->name,
                       ggml_type_name(x->type));
        }
    }
    static std::mutex             mu;
    static std::set<const void *> gu_done;  // gate tensors repacked in place (shared by the model's contexts)
    auto *                        gb = (ggml_backend_axera_buffer_context *) w[0]->buffer->context;
    auto *                        gp = (uint8_t *) w[0]->data;
    if (axera_mxfp4_gu_adjacent(w[0], w[1])) {
        std::lock_guard<std::mutex> lock(mu);
        if (gu_done.insert(w[0]->data).second) {
            axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
        }
    } else {
        exp_own.push_back(axera_cmm_alloc(2 * ggml_nbytes(w[0])));
        gb = exp_own.back().get();
        gp = (uint8_t *) gb->vir;
        axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
    }
    axera_repack_mxfp4(w[2]);
    experts e;
    e.n = w[0]->ne[2];
    for (int g = 0; g < 2; ++g) {
        const ggml_tensor * x    = g == 0 ? w[0] : w[2];
        const int64_t       rows = x->ne[1] * (g == 0 ? 2 : 1);
        e.b[g]                   = g == 0 ? gb : (ggml_backend_axera_buffer_context *) x->buffer->context;
        e.base[g]                = g == 0 ? (char *) gp : (char *) x->data;
        e.data[g]                = (size_t) (x->ne[0] * rows / 2);
        e.scale[g]               = (size_t) (x->ne[0] / 32 * rows);
        GGML_ASSERT(e.data[g] + e.scale[g] == (g == 0 ? 2 : 1) * x->nb[2]);
        GGML_ASSERT((size_t) e.n * e.data[g] < AXERA_IO_MAX && (size_t) e.n * e.scale[g] < AXERA_IO_MAX);
        mxfp4_planar(e.b[g], e.base[g], e.n, e.data[g], e.scale[g]);
    }
    return exps[il] = e;
}

// value heads from GGUF's tiled order (head r * KH + k, llama.cpp's converter) to HF's grouped one (k * REP + r, REP =
// VH / KH), the order of kernels that give each core whole key-head groups
static void vheads_grouped(const param_desc & p, const std::string & name, const uint8_t * src, uint8_t * dst) {
    int        r0 = 0, VH = 0, KH = 0;
    const bool rows = sscanf(p.vperm.c_str(), "rows:%d:%d:%d", &r0, &VH, &KH) == 3;
    if ((!rows && sscanf(p.vperm.c_str(), "cols:%d:%d", &VH, &KH) != 2) || KH <= 0 || VH % KH != 0 || p.shape.empty()) {
        GGML_ABORT("axera: parameter %s: bad vperm %s\n", name.c_str(), p.vperm.c_str());
    }
    const int     REP = VH / KH;
    const int64_t N   = p.shape[0];
    const size_t  rb  = p.bytes / N;
    GGML_ASSERT((size_t) N * rb == p.bytes);
    auto from = [&](int g) { return (g % REP) * KH + g / REP; };  // the tiled head of grouped head g
    if (rows) {
        const int64_t hr = (N - r0) / VH;
        GGML_ASSERT(r0 >= 0 && r0 + hr * VH == N);
        memcpy(dst, src, (size_t) r0 * rb);
        for (int g = 0; g < VH; ++g) {
            memcpy(dst + (size_t) (r0 + g * hr) * rb, src + (size_t) (r0 + from(g) * hr) * rb, (size_t) hr * rb);
        }
        return;
    }
    const size_t cb = rb / VH;
    GGML_ASSERT(cb * VH == rb);
    for (int64_t r = 0; r < N; ++r) {
        for (int g = 0; g < VH; ++g) {
            memcpy(dst + r * rb + g * cb, src + r * rb + from(g) * cb, cb);
        }
    }
}

AX_ENGINE_IO_BUFFER_T params::get(const std::map<std::string, ggml_tensor *> & t, int il, const std::string & kind,
                                  const std::string & name) {
    const param_desc &          p  = m.param(kind, name);
    const AX_ENGINE_IO_BUFFER_T io = convert(t, il, kind, name, p);
    if (p.vperm.empty() || !io.pVirAddr) {
        return io;
    }
    auto & c = own[{ il, kind + "/" + name + "/vperm" }];
    if (!c) {
        c = axera_cmm_alloc(p.bytes);
        vheads_grouped(p, name, (const uint8_t *) io.pVirAddr, (uint8_t *) c->vir);
        c->flush();
    }
    return axera_io(c, p.bytes);
}

AX_ENGINE_IO_BUFFER_T params::convert(const std::map<std::string, ggml_tensor *> & t, int il, const std::string & kind,
                                      const std::string & name, const param_desc & p) {
    const uint32_t size = p.bytes;
    if (p.convert == "mxfp4_planar_codes" || p.convert == "mxfp4_planar_scales") {
        const experts & e = experts_of(t, il);
        const int       g = name == "e_gw" || name == "e_gs" ? 0 : 1;
        const bool      codes = p.convert == "mxfp4_planar_codes";
        GGML_ASSERT((codes ? e.n * e.data[g] : e.n * e.scale[g]) == size);
        return axera_io(e.b[g], codes ? e.base[g] : e.base[g] + e.n * e.data[g], size);
    }
    // the tensor: blk.<il>.<gguf>, or a name of its own ("{L}" the layer: a.gen.code.blk.{L}.*; whole-model kinds out /
    // cp_out)
    ggml_tensor * w = nullptr;
    if (p.gguf.find("{L}") != std::string::npos) {
        std::string n = p.gguf;
        n.replace(n.find("{L}"), 3, std::to_string(il));
        w = model_tensor(t, n);
        if (!w) {
            GGML_ABORT("axera: tensor %s not in the graph\n", n.c_str());
        }
    } else {
        w = kind == "out" || kind == "cp_out" ? model_tensor(t, p.gguf) : tensor_of(t, il, p.gguf);
    }
    if (!w && kind == "out") {  // the output head's weights where the graph has no output head: nothing (pack members
        return {};              // stay zero)
    }
    if (!w) {
        GGML_ABORT("axera: none of %s in the graph\n", p.gguf.c_str());
    }
    auto bf16_of = [&]() -> AX_ENGINE_IO_BUFFER_T {  // the tensor as bf16: itself if bf16 in CMM, else a copy
        if (w->type == GGML_TYPE_BF16 && ggml_nbytes(w) == size && ggml_backend_buffer_is_axera(w->buffer)) {
            return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
        }
        if ((uint64_t) ggml_nelements(w) * sizeof(ggml_bf16_t) != size) {
            GGML_ABORT("axera: %s (%s, %lld elements) does not fit the %u-byte bf16 parameter %s\n", w->name,
                       ggml_type_name(w->type), (long long) ggml_nelements(w), size, name.c_str());
        }
        auto it = bf16.find(w);
        if (it == bf16.end()) {
            it = bf16.emplace(w, axera_to_bf16(w)).first;
            axera_drop_mapped(w);
        }
        return axera_io(it->second, size);
    };
    auto owned = [&](const std::function<void(void *)> & fill) -> AX_ENGINE_IO_BUFFER_T {
        auto & c = own[{ il, kind + "/" + name }];
        if (!c) {
            c = axera_cmm_alloc(size);
            fill(c->vir);
            c->flush();
        }
        return axera_io(c, size);
    };
    if (p.convert == "tensor") {
        // as it is (f32 / bf16 of the kernel's size), a bf16 copy of a quantized / f16 / larger f32 tensor
        if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) ||
            (w->type == GGML_TYPE_F32 && ggml_nbytes(w) == 2 * (size_t) size)) {
            return bf16_of();
        }
        if (ggml_nbytes(w) != size) {
            GGML_ABORT("axera: %s is %zu bytes, parameter %s %u\n", w->name, ggml_nbytes(w), name.c_str(), size);
        }
        if (ggml_backend_buffer_is_axera(w->buffer)) {
            return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
        }
        return owned([&](void * d) {  // not in CMM (the GGUF mapping)
            memcpy(d, w->data, size);
            axera_drop_mapped(w);
        });
    }
    if (p.convert == "bf16") {
        return bf16_of();
    }
    if (p.convert == "mxfp8" || p.convert == "mxfp8_scales" || p.convert == "mxfp4" || p.convert == "mxfp4_scales") {
        const int bits = p.convert.compare(0, 5, "mxfp4") == 0 ? 4 : 8;
        auto &    q    = mx[{ w, bits }];
        if (!q.q) {
            if (bits == 4) {
                axera_quantize_mxfp4(w, q.q, q.s);
            } else {
                axera_quantize_mxfp8(w, q.q, q.s);
            }
            axera_drop_mapped(w);
        }
        const axera_cmm & b = p.convert.size() > 5 ? q.s : q.q;
        if (b->size < size) {
            GGML_ABORT("axera: %s in MXFP%d is %zu bytes, parameter %s %u\n", w->name, bits, b->size, name.c_str(), size);
        }
        return axera_io(b, size);
    }
    if (p.convert == "log_neg") {  // GGUF ssm_a = -exp(A_log)
        GGML_ASSERT(w->type == GGML_TYPE_F32 && ggml_nbytes(w) == size);
        return owned([&](void * d) {
            for (int64_t j = 0; j < ggml_nelements(w); ++j) {
                ((float *) d)[j] = logf(-((const float *) w->data)[j]);
            }
        });
    }
    if (p.convert == "bf16_taps_reversed") {  // f32 [channels][taps] -> bf16, kernel tap 0 = the current token
        const int64_t taps = w->ne[0], ch = w->ne[1];
        GGML_ASSERT(w->type == GGML_TYPE_F32 && ch * taps * (int64_t) sizeof(ggml_bf16_t) == size);
        return owned([&](void * d) {
            for (int64_t r = 0; r < ch; ++r) {
                for (int64_t j = 0; j < taps; ++j) {
                    ((ggml_bf16_t *) d)[r * taps + j] = ggml_fp32_to_bf16(((const float *) w->data)[r * taps + taps - 1 - j]);
                }
            }
        });
    }
    GGML_ABORT("axera: parameter %s: unknown conversion %s\n", name.c_str(), p.convert.c_str());
}

const axera_cmm & params::pack(const std::map<std::string, ggml_tensor *> & t, const kernel_desc & kd,
                               const std::string & pack, int il0) {
    std::string sig = std::to_string(il0) + " " + pack + ":";
    for (const auto & mb : kd.members) {
        if (mb.pack == pack) {
            sig += mb.kind + "/" + mb.param + "@" + std::to_string(mb.layer) + " " + std::to_string(mb.offset) + " " +
                   std::to_string(mb.bytes) + " " + std::to_string(mb.strip_w) + " " + std::to_string(mb.ranges.size()) + ";";
        }
    }
    const size_t bytes = kd.packs.at(pack);
    auto         it    = packs.find(sig);
    if (it != packs.end() && it->second->size >= bytes) {
        return it->second;
    }
    axera_cmm b = axera_cmm_alloc(bytes);  // zeroed
    for (const auto & mb : kd.members) {
        if (mb.pack != pack) {
            continue;
        }
        const AX_ENGINE_IO_BUFFER_T src = get(t, il0 + std::max(mb.layer, 0), mb.kind, mb.param);
        if (!src.pVirAddr && mb.kind == "out") {  // (a whole-model group's logits are then not read)
            static std::once_flag once;
            std::call_once(once, [] { GGML_LOG_INFO("axera: no output head in the graph: the group's head members left zero\n"); });
            continue;
        }
        GGML_ASSERT(src.pVirAddr && src.nSize == mb.bytes && mb.offset + mb.bytes <= b->size);
        char *       dst = (char *) b->vir + mb.offset;
        const char * sp  = (const char *) src.pVirAddr;
        if (mb.strip_w > 0) {  // [rows][cols] -> strip j = columns j * w .. (j + 1) * w of every row, one after another
            const size_t es = mb.bytes / (size_t) (mb.strip_rows * mb.strip_cols);
            for (int64_t j = 0; j < mb.strip_cols / mb.strip_w; ++j) {
                for (int64_t r = 0; r < mb.strip_rows; ++r) {
                    memcpy(dst + ((j * mb.strip_rows + r) * mb.strip_w) * es, sp + (r * mb.strip_cols + j * mb.strip_w) * es,
                           mb.strip_w * es);
                }
            }
        } else if (!mb.ranges.empty()) {  // column blocks: each range of every row in turn
            for (const auto & [x, y] : mb.ranges) {
                for (size_t r = 0; r < mb.blk_rows; ++r, dst += y - x) {
                    memcpy(dst, sp + r * mb.blk_row_bytes + x, y - x);
                }
            }
        } else {
            memcpy(dst, sp, mb.bytes);
        }
    }
    b->flush();
    return packs[sig] = std::move(b);
}

void params::release(const std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> & live) {
    std::vector<const void *> used;
    for (const auto * io : live) {
        for (const auto & b : *io) {
            used.push_back(b.pVirAddr);
        }
    }
    std::sort(used.begin(), used.end());
    auto in_use = [&](const axera_cmm & b) {
        auto it = std::lower_bound(used.begin(), used.end(), (const void *) b->vir);
        return it != used.end() && *it < (const char *) b->vir + b->size;
    };
    size_t freed = 0;
    for (auto it = mx.begin(); it != mx.end();) {  // elements and scales go together
        if (in_use(it->second.q) || in_use(it->second.s)) {
            ++it;
            continue;
        }
        freed += it->second.q->size + it->second.s->size;
        it = mx.erase(it);
    }
    auto drop = [&](auto & map) {
        for (auto it = map.begin(); it != map.end();) {
            if (in_use(it->second)) {
                ++it;
                continue;
            }
            freed += it->second->size;
            it = map.erase(it);
        }
    };
    drop(bf16);
    drop(own);
    size_t pk = 0, kept = 0, ex = 0;  // what stays: the packs, the conversions kernels read directly, expert copies
    for (const auto & [sig, b] : packs) {
        pk += b->size;
    }
    for (const auto & [k, q] : mx) {
        kept += q.q->size + q.s->size;
    }
    for (const auto * map : { &bf16 }) {
        for (const auto & [k, b] : *map) {
            kept += b->size;
        }
    }
    for (const auto & [k, b] : own) {
        kept += b->size;
    }
    for (const auto & b : exp_own) {
        ex += b->size;
    }
    size_t cmm_free, cmm_total;
    axera_cmm_memory(&cmm_free, &cmm_total);
    GGML_LOG_INFO("axera: parameters: %.1f MiB freed (read only through packs); kept: packs %.1f MiB, direct %.1f MiB, "
                  "expert copies %.1f MiB; CMM in use %.1f of %.1f MiB\n", freed / 1048576.0, pk / 1048576.0,
                  kept / 1048576.0, ex / 1048576.0, (cmm_total - cmm_free) / 1048576.0, cmm_total / 1048576.0);
}

}  // namespace axera
