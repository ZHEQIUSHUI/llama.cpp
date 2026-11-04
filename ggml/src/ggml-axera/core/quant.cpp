// Axera NPU backend core: weight conversions for the kernels

#include "quant.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <set>

// AXERA_SGL35_SPLIT=1: time a read-only page-in pass separately from the dequantize + NVFP4 encode pass. The normal
// loader (and every release run) reads the mapped GGUF rows inside the conversion pass; this mode is measurement only.
namespace {
bool axera_split_profile() {
    static const bool on = [] {
        const char * e = std::getenv("AXERA_SGL35_SPLIT");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

std::atomic<long long> axera_split_read_us{ 0 };
std::atomic<long long> axera_split_conv_us{ 0 };
std::atomic<long long> axera_split_bytes{ 0 };
std::atomic<long long> axera_split_slices{ 0 };

void axera_split_touch(const ggml_tensor * w, int64_t r0, int64_t nr) {
    if (!axera_split_profile()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const size_t  stride2 = w->nb[2] ? w->nb[2] : (size_t) w->ne[1] * w->nb[1];
    axera_host_pool().run(nr, [&](int64_t r) {
        const int64_t rr = r0 + r, d1 = rr % w->ne[1], d2 = rr / w->ne[1];
        const uint8_t * p = (const uint8_t *) w->data + (size_t) d2 * stride2 + (size_t) d1 * w->nb[1];
        volatile uint8_t sink = 0;
        for (size_t o = 0; o < w->nb[1]; o += 4096) {
            sink = (uint8_t) (sink ^ p[o]);
        }
        (void) sink;
    });
    axera_split_read_us.fetch_add(ggml_time_us() - t0);
    axera_split_bytes.fetch_add((long long) nr * (long long) w->nb[1]);
    axera_split_slices.fetch_add(1);
}

struct axera_split_report {
    ~axera_split_report() {
        if (!axera_split_profile()) {
            return;
        }
        const double rs = axera_split_read_us.load() / 1e6, cs = axera_split_conv_us.load() / 1e6;
        const double mb = axera_split_bytes.load() / (1024.0 * 1024.0);
        std::fprintf(stderr, "axera split: read %.1f s (%.1f MiB/s), convert %.1f s, total %.1f s, "
                             "%lld slices, %.1f MiB\n", rs, rs > 0 ? mb / rs : 0.0, cs, rs + cs,
                     (long long) axera_split_slices.load(), mb);
    }
};
axera_split_report axera_split_report_instance;
}  // namespace

// [N, K] weight of any GGUF type -> MXFP8 elements [N, K] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp8(const ggml_tensor * w, axera_cmm & q, axera_cmm & s) {
    GGML_ASSERT(w->ne[0] % AXERA_MX_GROUP == 0);
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    q = axera_cmm_alloc(N * K);
    s = axera_cmm_alloc(N * K / AXERA_MX_GROUP);
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        axera_quantize_row_mxfp8_f32(row, (uint8_t *) q->vir + r * K, (uint8_t *) s->vir + r * (K / AXERA_MX_GROUP), K);
    });
    q->flush(q->vir, q->size);
    s->flush(s->vir, s->size);
}

// [N, K] weight of any GGUF type -> MXFP4 (as ggml's and the 35B experts' kernels): per 32 inputs the scale
// 2^(floor(log2(amax)) - 2) (E8M0 byte e + 127) and E2M1 codes (nearest value, sign in bit 3), two per byte (the first in
// the low nibble): elements [N, K / 2] and scales [N, K / 32] in new CMM buffers
void axera_quantize_mxfp4(const ggml_tensor * w, axera_cmm & q, axera_cmm & s) {
    GGML_ASSERT(w->ne[0] % AXERA_MX_GROUP == 0);
    static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    q = axera_cmm_alloc(N * K / 2);
    s = axera_cmm_alloc(N * K / AXERA_MX_GROUP);
    if (w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w)) {  // already MXFP4: the same values, rows rearranged
        const int                n_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (int t = 0; t < n_threads; ++t) {
            pool.emplace_back([&, t] {
                const int64_t r0 = N * t / n_threads, r1 = N * (t + 1) / n_threads;
                axera_mxfp4_rows((const uint8_t *) w->data + r0 * w->nb[1], r1 - r0, K, (uint8_t *) q->vir + r0 * (K / 2),
                                 (uint8_t *) s->vir + r0 * (K / AXERA_MX_GROUP));
            });
        }
        for (auto & th : pool) {
            th.join();
        }
        return;
    }
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        uint8_t * qr = (uint8_t *) q->vir + r * (K / 2);
        uint8_t * sr = (uint8_t *) s->vir + r * (K / AXERA_MX_GROUP);
        for (int64_t g = 0; g < K / AXERA_MX_GROUP; ++g) {
            const float * v    = row + g * AXERA_MX_GROUP;
            float         amax = 0.0f;
            for (int i = 0; i < AXERA_MX_GROUP; ++i) {
                amax = std::max(amax, fabsf(v[i]));
            }
            const int   e  = amax > 0.0f ? (int) floorf(log2f(amax)) - 2 : 0;
            const float sc = ldexpf(1.0f, -e);
            sr[g]          = (uint8_t) std::min(254, std::max(0, e + 127));
            for (int i = 0; i < AXERA_MX_GROUP; i += 2) {
                uint8_t c[2];
                for (int h = 0; h < 2; ++h) {
                    const float x    = v[i + h] * sc, a = fabsf(x);
                    int         best = 0;
                    for (int k = 1; k < 8; ++k) {
                        if (fabsf(a - e2m1[k]) < fabsf(a - e2m1[best])) {
                            best = k;
                        }
                    }
                    c[h] = (uint8_t) (best | (x < 0.0f ? 8 : 0));
                }
                qr[(g * AXERA_MX_GROUP + i) / 2] = (uint8_t) (c[0] | (c[1] << 4));
            }
        }
    });
    q->flush(q->vir, q->size);
    s->flush(s->vir, s->size);
}

static float axera_e4m3_to_f32(uint8_t b) {
    const float sign = b & 0x80 ? -1.0f : 1.0f;
    const int   e    = (b >> 3) & 15, m = b & 7;
    if (e == 0) {
        return sign * ldexpf((float) m, -9);
    }
    return sign * ldexpf(1.0f + (float) m / 8.0f, e - 7);
}

static uint8_t axera_f32_to_e2m1(float x) {
    static const float v[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float       a     = fabsf(x);
    int               best  = 0;
    for (int i = 1; i < 8; ++i) {
        const float di = fabsf(a - v[i]), db = fabsf(a - v[best]);
        if (di < db || (di == db && (i & 1) == 0)) {
            best = i;
        }
    }
    return (uint8_t) (best | (x < 0.0f ? 8 : 0));
}

// one row (K floats) of a K x N x ... weight -> NVFP4, in place into the caller's row buffers: the E2M1 pairs
// (low nibble = even K) and the E4M3 scale of each 16-input block, with the caller's global tensor scale
static void axera_nvfp4_row(const float * v, int64_t K, float scale2, uint8_t * qr, uint8_t * sr) {
    for (int64_t g = 0; g < K / 16; ++g) {
        const float * b  = v + g * 16;
        float         bm = 0.0f;
        for (int i = 0; i < 16; ++i) {
            bm = std::fmax(bm, std::fabs(b[i]));
        }
        const uint8_t sb = axera_f32_to_e4m3(bm / (6.0f * scale2));
        sr[g]            = sb;
        const float ds   = axera_e4m3_to_f32(sb) * scale2;
        for (int i = 0; i < 16; i += 2) {
            const uint8_t a = axera_f32_to_e2m1(ds ? b[i] / ds : 0.0f);
            const uint8_t c = axera_f32_to_e2m1(ds ? b[i + 1] / ds : 0.0f);
            qr[(g * 16 + i) / 2] = (uint8_t) (a | (c << 4));
        }
    }
}

// rows r0 .. r0 + nr of a weight of any GGUF type (logical row-major [r][K]) -> float row r
template <typename F> static void axera_nvfp4_rows_f32(const ggml_tensor * w, int64_t r0, int64_t nr, F fn) {
    const int64_t K      = w->ne[0];
    const auto *  traits = ggml_get_type_traits(w->type);
    GGML_ASSERT(w->type == GGML_TYPE_F32 || traits->to_float != nullptr);
    GGML_ASSERT(w->nb[2] == 0 || w->nb[2] == (size_t) (w->ne[1] * w->nb[1]));
    axera_host_pool().run(nr, [&](int64_t r) {
        thread_local std::vector<float> row;
        row.resize(K);
        const int64_t rr = r0 + r, d1 = rr % w->ne[1], d2 = rr / w->ne[1];
        const void *  src = (const char *) w->data + (size_t) d2 * w->nb[2] + (size_t) d1 * w->nb[1];
        if (w->type == GGML_TYPE_F32) {
            memcpy(row.data(), src, K * sizeof(float));
        } else {
            traits->to_float(src, row.data(), K);
        }
        fn(r, row.data());
    });
}

// the global scale of rows r0 .. r0 + nr: amax / (448 * 6), or 1 for an all-zero slice
float axera_nvfp4_amax_rows(const ggml_tensor * w, int64_t r0, int64_t nr) {
    GGML_ASSERT(w->ne[0] % 16 == 0 && r0 >= 0 && nr > 0 && r0 + nr <= ggml_nrows(w));
    std::vector<float> mx(nr, 0.0f);
    axera_nvfp4_rows_f32(w, r0, nr, [&](int64_t r, const float * v) {
        float m = 0.0f;
        for (int64_t i = 0, K = w->ne[0]; i < K; ++i) {
            m = std::fmax(m, std::fabs(v[i]));
        }
        mx[r] = m;
    });
    float amax = 0.0f;
    for (float v : mx) {
        amax = std::fmax(amax, v);
    }
    return amax ? amax / (448.0f * 6.0f) : 1.0f;
}

// rows r0 .. r0 + nr of a weight -> NVFP4 at q + r * q_row_bytes / s + r * s_row_bytes, with a known global scale
void axera_quantize_nvfp4_into(const ggml_tensor * w, int64_t r0, int64_t nr, float scale2,
                               uint8_t * q, int64_t q_row_bytes, uint8_t * s, int64_t s_row_bytes) {
    const int64_t K = w->ne[0];
    GGML_ASSERT(K % 16 == 0 && r0 >= 0 && nr > 0 && r0 + nr <= ggml_nrows(w));
    GGML_ASSERT(q_row_bytes == K / 2 && s_row_bytes == K / 16);
    axera_nvfp4_rows_f32(w, r0, nr, [&](int64_t r, const float * v) {
        axera_nvfp4_row(v, K, scale2, q + (size_t) r * q_row_bytes, s + (size_t) r * s_row_bytes);
    });
}

// rows r0 .. r0 + nr -> NVFP4, decoding each row once into a small host buffer first (a slice at a time)
float axera_quantize_nvfp4_slice(const ggml_tensor * w, int64_t r0, int64_t nr, uint8_t * q, int64_t q_row_bytes,
                                 uint8_t * s, int64_t s_row_bytes) {
    const int64_t K = w->ne[0];
    GGML_ASSERT(K % 16 == 0 && r0 >= 0 && nr > 0 && r0 + nr <= ggml_nrows(w));
    GGML_ASSERT(q_row_bytes == K / 2 && s_row_bytes == K / 16);
    axera_split_touch(w, r0, nr);
    const int64_t      t0 = ggml_time_us();
    std::vector<float> rows((size_t) nr * K);
    axera_nvfp4_rows_f32(w, r0, nr, [&](int64_t r, const float * v) {
        memcpy(rows.data() + (size_t) r * K, v, K * sizeof(float));
    });
    float amax = 0.0f;
    for (float v : rows) {
        amax = std::fmax(amax, std::fabs(v));
    }
    const float scale2 = amax ? amax / (448.0f * 6.0f) : 1.0f;
    axera_host_pool().run(nr, [&](int64_t r) {
        axera_nvfp4_row(rows.data() + (size_t) r * K, K, scale2, q + (size_t) r * q_row_bytes,
                        s + (size_t) r * s_row_bytes);
    });
    axera_split_conv_us.fetch_add(ggml_time_us() - t0);
    return scale2;
}

// the same, with destination row r read from local source row row_src[r] and destination 16-input block g from
// source block blk_src[g] (either may be null). GGUF stores GDN value heads tiled; this undoes that permutation.
float axera_quantize_nvfp4_slice_perm(const ggml_tensor * w, int64_t r0, int64_t nr, const std::vector<int64_t> * row_src,
                                      const std::vector<int64_t> * blk_src, uint8_t * q, int64_t q_row_bytes,
                                      uint8_t * s, int64_t s_row_bytes) {
    const int64_t K = w->ne[0];
    GGML_ASSERT(K % 16 == 0 && r0 >= 0 && nr > 0 && r0 + nr <= ggml_nrows(w));
    GGML_ASSERT(q_row_bytes == K / 2 && s_row_bytes == K / 16);
    GGML_ASSERT(!row_src || (int64_t) row_src->size() == nr);
    GGML_ASSERT(!blk_src || (int64_t) blk_src->size() == K / 16);
    axera_split_touch(w, r0, nr);
    const int64_t      t0 = ggml_time_us();
    std::vector<float>   rows((size_t) nr * K);
    std::vector<int64_t> dst_of;
    if (row_src) {
        dst_of.assign(nr, -1);
        for (int64_t d = 0; d < nr; ++d) {
            GGML_ASSERT((*row_src)[d] >= 0 && (*row_src)[d] < nr && dst_of[(*row_src)[d]] < 0);
            dst_of[(*row_src)[d]] = d;
        }
    }
    axera_nvfp4_rows_f32(w, r0, nr, [&](int64_t r, const float * v) {
        memcpy(rows.data() + (size_t) (row_src ? dst_of[r] : r) * K, v, K * sizeof(float));
    });
    float amax = 0.0f;
    for (float v : rows) {
        amax = std::fmax(amax, std::fabs(v));
    }
    const float scale2 = amax ? amax / (448.0f * 6.0f) : 1.0f;
    axera_host_pool().run(nr, [&](int64_t r) {
        const float * v  = rows.data() + (size_t) r * K;
        uint8_t *     qr = q + (size_t) r * q_row_bytes;
        uint8_t *     sr = s + (size_t) r * s_row_bytes;
        for (int64_t g = 0; g < K / 16; ++g) {
            const float * b  = v + (blk_src ? (*blk_src)[g] : g) * 16;
            float         bm = 0.0f;
            for (int i = 0; i < 16; ++i) {
                bm = std::fmax(bm, std::fabs(b[i]));
            }
            const uint8_t sb = axera_f32_to_e4m3(bm / (6.0f * scale2));
            sr[g]            = sb;
            const float ds   = axera_e4m3_to_f32(sb) * scale2;
            for (int i = 0; i < 16; i += 2) {
                const uint8_t a = axera_f32_to_e2m1(ds ? b[i] / ds : 0.0f);
                const uint8_t c = axera_f32_to_e2m1(ds ? b[i + 1] / ds : 0.0f);
                qr[(g * 16 + i) / 2] = (uint8_t) (a | (c << 4));
            }
        }
    });
    axera_split_conv_us.fetch_add(ggml_time_us() - t0);
    return scale2;
}

axera_nv4 axera_quantize_nvfp4_rows(const ggml_tensor * w, int64_t r0, int64_t nr) {
    GGML_ASSERT(w->ne[0] % 16 == 0 && r0 >= 0 && nr >= 0 && r0 + nr <= ggml_nrows(w));
    const int64_t K = w->ne[0];
    axera_nv4     out;
    out.q = axera_cmm_alloc((size_t) nr * K / 2);
    out.s = axera_cmm_alloc((size_t) nr * K / 16);
    if (nr) {
        out.scale2 = axera_quantize_nvfp4_slice(w, r0, nr, (uint8_t *) out.q->vir, K / 2, (uint8_t *) out.s->vir,
                                                K / 16);
    }
    out.q->flush();
    out.s->flush();
    return out;
}

// [N, K] weight of another GGUF type (quantized, f16, f32) -> bf16 copy in a new CMM buffer
axera_cmm axera_to_bf16(const ggml_tensor * w) {
    const int64_t K = w->ne[0], N = ggml_nrows(w);
    axera_cmm     b = axera_cmm_alloc(N * K * sizeof(ggml_bf16_t));
    axera_for_rows_f32(w, [&](int64_t r, const float * row) {
        ggml_fp32_to_bf16_row(row, (ggml_bf16_t *) b->vir + r * K, K);
    });
    b->flush(b->vir, b->size);
    return b;
}
// ggml MXFP4 blocks (per 32 inputs: the E8M0 scale byte, then 16 bytes of E2M1 codes: low nibbles elements 0..15, high
// nibbles 16..31) -> the kernels' layout, in place per expert (dim 2): the codes [rows][K / 2], two consecutive elements
// per byte (the first in the low nibble), then the scales [rows][K / 32]. Once per tensor: the GGUF data is the
// kernels' afterwards (the template runs every MoE layer)
// ggml block_mxfp4 rows (e byte + 16 bytes of nibble pairs per 32) -> the kernels' layout: codes [rows][K / 2] (FP4
// pairs, low nibble first) then E8M0 scales [rows][K / 32]
void axera_mxfp4_rows(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dq, uint8_t * ds) {
    const int64_t nb = K / 32;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = src + (r * nb + b) * 17;
            uint8_t *       o   = dq + r * K / 2 + b * 16;
            ds[r * nb + b]      = blk[0];
            for (int j = 0; j < 8; ++j) {  // elements 2j, 2j + 1 (low nibbles), 16 + 2j, 17 + 2j (high nibbles)
                o[j]     = (uint8_t) ((blk[1 + 2 * j] & 15) | ((blk[2 + 2 * j] & 15) << 4));
                o[8 + j] = (uint8_t) ((blk[1 + 2 * j] >> 4) | (blk[2 + 2 * j] & 0xF0));
            }
        }
    }
}
void axera_check_mxfp4(const ggml_tensor * w) {
    GGML_ASSERT(w->type == GGML_TYPE_MXFP4 && ggml_is_contiguous(w) && w->ne[0] % 32 == 0 && w->ne[3] == 1 &&
                ggml_backend_buffer_is_axera(w->buffer) && w->nb[2] == (size_t) (w->ne[1] * w->ne[0] / 32 * 17));
}

// MoE experts [K, rows, n_expert] in place, each expert's slot (stride bytes) in the kernels' layout
void axera_repack_mxfp4(const ggml_tensor * w) {
    static std::set<const void *> done;
    if (!done.insert(w->data).second) {
        return;
    }
    axera_check_mxfp4(w);
    const int64_t K = w->ne[0], rows = w->ne[1];
    const size_t  stride = w->nb[2];
    axera_host_pool().run(w->ne[2], [&](int64_t e) {
        std::vector<uint8_t> tmp(stride);
        axera_mxfp4_rows((const uint8_t *) w->data + e * stride, rows, K, tmp.data(), tmp.data() + rows * K / 2);
        memcpy((uint8_t *) w->data + e * stride, tmp.data(), stride);
    });
    ((ggml_backend_axera_buffer_context *) w->buffer->context)->flush(w->data, ggml_nbytes(w));
}

// gate and up experts [K, rows, n] -> one gate-up tensor at dst (2 nbytes): expert e at 2 e strides, codes
// [2 rows][K / 2] (gate rows, then up), scales [2 rows][K / 32] (as a GGUF ffn_gate_up_exps repacked by
// axera_repack_mxfp4). In place when up follows gate (dst = gate: llama.cpp allocates them so unless a buffer boundary
// falls between), else into a buffer of its own
bool axera_mxfp4_gu_adjacent(const ggml_tensor * g, const ggml_tensor * u) {
    return g->buffer == u->buffer && (const char *) u->data == (const char *) g->data + ggml_nbytes(g);
}
void axera_repack_mxfp4_gu(const ggml_tensor * g, const ggml_tensor * u, uint8_t * dst,
                           const ggml_backend_axera_buffer_context * db) {
    axera_check_mxfp4(g);
    axera_check_mxfp4(u);
    GGML_ASSERT(ggml_are_same_shape(g, u));
    const int64_t K = g->ne[0], rows = g->ne[1], n = g->ne[2];
    const size_t  stride = g->nb[2], half = (size_t) (rows * K / 2);
    std::vector<uint8_t> copy;  // in place: the sources, before they are overwritten
    const uint8_t *      gs = (const uint8_t *) g->data, *us = (const uint8_t *) u->data;
    if (dst == g->data) {
        GGML_ASSERT(axera_mxfp4_gu_adjacent(g, u));
        copy.assign(gs, gs + 2 * ggml_nbytes(g));
        gs = copy.data();
        us = copy.data() + ggml_nbytes(g);
    }
    axera_host_pool().run(n, [&](int64_t e) {
        uint8_t * dq = dst + 2 * e * stride;
        uint8_t * ds = dq + 2 * half;
        axera_mxfp4_rows(gs + e * stride, rows, K, dq, ds);
        axera_mxfp4_rows(us + e * stride, rows, K, dq + half, ds + rows * K / 32);
    });
    db->flush(dst, 2 * ggml_nbytes(g));
}
