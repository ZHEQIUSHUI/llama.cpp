#pragma once

// MXFP8 (OCP MX, E4M3 elements with one E8M0 scale per 32 consecutive inputs) quantization of BF16 weight rows.
// Scale exponent e = ceil(log2(amax / 448)), so no element saturates; elements are rounded to nearest even.

#include <cmath>
#include <cstdint>
#include <cstring>

#define AXERA_MX_GROUP 32

static inline float axera_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float    f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// |x| <= 448 (finite); returns the E4M3FN bit pattern, round to nearest even
static inline uint8_t axera_f32_to_e4m3(float x) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    float         a    = std::fabs(x);
    if (a < std::ldexp(1.0f, -6)) {  // subnormal: steps of 2^-9, 8 steps reach the smallest normal
        const int q = (int) std::nearbyint(std::ldexp(a, 9));
        return sign | (uint8_t) q;  // q == 8 is exponent 1, mantissa 0: the same bits
    }
    int         e;
    const float m = std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    int         q = (int) std::nearbyint((m * 2.0f - 1.0f) * 8.0f);
    int         E = e - 1 + 7;
    if (q == 8) {
        q = 0;
        ++E;
    }
    if (E > 15 || (E == 15 && q > 6)) {  // cannot happen with the scale rule; saturate to 448
        E = 15;
        q = 6;
    }
    return sign | (uint8_t) (E << 3) | (uint8_t) q;
}

// one row of K float values -> K e4m3 bytes and K / 32 e8m0 scales
static inline void axera_quantize_row_mxfp8_f32(const float * src, uint8_t * q, uint8_t * scale, int64_t K) {
    for (int64_t g = 0; g < K / AXERA_MX_GROUP; ++g) {
        const float * s    = src + g * AXERA_MX_GROUP;
        float         amax = 0.0f;
        for (int i = 0; i < AXERA_MX_GROUP; ++i) {
            amax = std::fmax(amax, std::fabs(s[i]));
        }
        amax  = std::fmax(amax, std::ldexp(1.0f, -126));
        int e = (int) std::ceil(std::log2(amax / 448.0f));
        e     = e < -127 ? -127 : (e > 127 ? 127 : e);
        scale[g]        = (uint8_t) (e + 127);
        const float inv = std::ldexp(1.0f, -e);
        for (int i = 0; i < AXERA_MX_GROUP; ++i) {
            q[g * AXERA_MX_GROUP + i] = axera_f32_to_e4m3(s[i] * inv);
        }
    }
}
// E4M3FN bit pattern -> float (finite patterns)
static inline float axera_e4m3_to_f32(uint8_t b) {
    const int   E = (b >> 3) & 15, q = b & 7;
    const float v = E == 0 ? std::ldexp((float) q, -9) : std::ldexp(1.0f + q / 8.0f, E - 7);
    return b & 0x80 ? -v : v;
}

// NVFP4 (E2M1 elements, one E4M3 scale per 16 inputs, an FP32 tensor scale s2): a block's scale is its amax / 6 / s2
// in E4M3, its elements x / (scale * s2) rounded to the nearest E2M1 value (ties: the smaller), two per byte with the
// first in the low nibble; one row of K floats -> K / 2 bytes and K / 16 scales
#define AXERA_NV_GROUP 16
static inline void axera_quantize_row_nvfp4_f32(const float * src, uint8_t * q, uint8_t * scale, int64_t K, float s2) {
    static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int64_t g = 0; g < K / AXERA_NV_GROUP; ++g) {
        const float * v    = src + g * AXERA_NV_GROUP;
        float         amax = 0.0f;
        for (int i = 0; i < AXERA_NV_GROUP; ++i) {
            amax = std::fmax(amax, std::fabs(v[i]));
        }
        const uint8_t sb = axera_f32_to_e4m3(std::fmin(amax / 6.0f / s2, 448.0f));
        const float   d  = axera_e4m3_to_f32(sb) * s2;
        scale[g]         = sb;
        for (int i = 0; i < AXERA_NV_GROUP; i += 2) {
            uint8_t c[2];
            for (int h = 0; h < 2; ++h) {
                const float x = d > 0.0f ? v[i + h] / d : 0.0f, a = std::fabs(x);
                int         best = 0;
                for (int k = 1; k < 8; ++k) {
                    if (std::fabs(a - e2m1[k]) < std::fabs(a - e2m1[best])) {
                        best = k;
                    }
                }
                c[h] = (uint8_t) (best | (x < 0.0f ? 8 : 0));
            }
            q[(g * AXERA_NV_GROUP + i) / 2] = (uint8_t) (c[0] | (c[1] << 4));
        }
    }
}
