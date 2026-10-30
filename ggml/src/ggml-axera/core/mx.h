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
