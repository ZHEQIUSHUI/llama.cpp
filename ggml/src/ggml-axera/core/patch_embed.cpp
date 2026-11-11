#include "patch_embed.h"

#include <cmath>
#include <cstring>

namespace axera {

namespace {

double dot(const double * a, const double * b, int n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int    i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return s0 + s1 + s2 + s3;
}

float dotf(const float * a, const float * b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    int   i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// orthonormal columns: q [n][m] (column j at q + j * m), classical Gram-Schmidt run twice per column
void orthonormalize(std::vector<double> & q, int n, int m, const patch_embed::parallel & par) {
    std::vector<double> r(n);
    const int           chunks = 8, step = (m + chunks - 1) / chunks;
    for (int j = 0; j < n; ++j) {
        double * y = q.data() + (size_t) j * m;
        for (int pass = 0; pass < 2 && j > 0; ++pass) {
            par(j, [&](int64_t i) { r[i] = dot(q.data() + (size_t) i * m, y, m); });
            par(chunks, [&](int64_t c) {
                const int h0 = (int) c * step, h1 = h0 + step < m ? h0 + step : m;
                for (int i = 0; i < j; ++i) {
                    const double * qi = q.data() + (size_t) i * m;
                    const double   ri = r[i];
                    for (int h = h0; h < h1; ++h) {
                        y[h] -= ri * qi[h];
                    }
                }
            });
        }
        const double nrm = std::sqrt(dot(y, y, m));
        const double inv = nrm > 0 ? 1.0 / nrm : 0.0;
        for (int h = 0; h < m; ++h) {
            y[h] *= inv;
        }
    }
}

}  // namespace

bool patch_embed::init(const float * w0, const float * w1, int H_, int CP_, int width, parallel par_) {
    if (H_ <= 0 || CP_ <= 0 || (width != CP_ && width != 2 * CP_)) {
        return false;
    }
    H   = H_;
    CP  = CP_;
    PW  = width;
    par = std::move(par_);
    W0.assign(w0, w0 + (size_t) H * CP);
    W1.assign(w1, w1 + (size_t) H * CP);
    frames_cur = 0;
    basis      = false;
    resid      = 0;
    Wk.clear();
    A.clear();
    return true;
}

// Q: the CP leading directions of [W0 W1]'s column space (a randomized range finder with one power iteration: the
// spectrum drops by orders of magnitude after CP, one iteration is at the limit already), and A = Q^T [W0 W1]
void patch_embed::build_basis() {
    basis = true;
    A.assign((size_t) CP * 2 * CP, 0.0f);
    Wk.assign((size_t) H * CP, 0.0f);
    if (H <= CP) {  // e itself fits a px row: W = [I 0], px = [e 0]
        for (int h = 0; h < H; ++h) {
            memcpy(A.data() + (size_t) h * 2 * CP, W0.data() + (size_t) h * CP, CP * sizeof(float));
            memcpy(A.data() + (size_t) h * 2 * CP + CP, W1.data() + (size_t) h * CP, CP * sizeof(float));
            Wk[(size_t) h * CP + h] = 1.0f;
        }
        resid = 0;
        return;
    }
    // Y = [W0 W1] Omega, Omega [2 CP][CP] standard normal (a fixed seed: the same basis every run), stored by column
    std::vector<double> om((size_t) 2 * CP * CP);
    {
        uint64_t   s    = 0x9e3779b97f4a7c15ull;
        const auto next = [&]() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            return (double) ((s >> 11) + 1) / 9007199254740993.0;  // (0, 1)
        };
        for (size_t i = 0; i + 1 < om.size(); i += 2) {
            const double a = std::sqrt(-2.0 * std::log(next())), b = 6.283185307179586 * next();
            om[i]     = a * std::cos(b);
            om[i + 1] = a * std::sin(b);
        }
    }
    std::vector<double> q((size_t) CP * H);  // column j at q + j * H
    par(CP, [&](int64_t j) {
        double *       y  = q.data() + (size_t) j * H;
        const double * oj = om.data() + (size_t) j * 2 * CP;
        for (int h = 0; h < H; ++h) {
            const float * a = W0.data() + (size_t) h * CP, *b = W1.data() + (size_t) h * CP;
            double        s = 0;
            for (int m = 0; m < CP; ++m) {
                s += a[m] * oj[m] + b[m] * oj[CP + m];
            }
            y[h] = s;
        }
    });
    om.clear();
    om.shrink_to_fit();
    orthonormalize(q, CP, H, par);
    // one power iteration: q <- orth([W0 W1] [W0 W1]^T q)
    {
        std::vector<double> z((size_t) CP * H);
        par(CP, [&](int64_t j) {
            const double *      y = q.data() + (size_t) j * H;
            std::vector<double> t0(CP, 0.0), t1(CP, 0.0);
            for (int h = 0; h < H; ++h) {
                const float * a = W0.data() + (size_t) h * CP, *b = W1.data() + (size_t) h * CP;
                const double  v = y[h];
                for (int m = 0; m < CP; ++m) {
                    t0[m] += v * a[m];
                    t1[m] += v * b[m];
                }
            }
            double * o = z.data() + (size_t) j * H;
            for (int h = 0; h < H; ++h) {
                const float * a = W0.data() + (size_t) h * CP, *b = W1.data() + (size_t) h * CP;
                double        s = 0;
                for (int m = 0; m < CP; ++m) {
                    s += a[m] * t0[m] + b[m] * t1[m];
                }
                o[h] = s;
            }
        });
        q.swap(z);
    }
    orthonormalize(q, CP, H, par);
    std::vector<double> kept(CP, 0.0);
    par(CP, [&](int64_t j) {
        const double *      y = q.data() + (size_t) j * H;
        std::vector<double> t((size_t) 2 * CP, 0.0);
        for (int h = 0; h < H; ++h) {
            const float * a = W0.data() + (size_t) h * CP, *b = W1.data() + (size_t) h * CP;
            const double  v = y[h];
            for (int m = 0; m < CP; ++m) {
                t[m]      += v * a[m];
                t[CP + m] += v * b[m];
            }
        }
        float * o = A.data() + (size_t) j * 2 * CP;
        double  e = 0;
        for (int m = 0; m < 2 * CP; ++m) {
            o[m] = (float) t[m];
            e   += t[m] * t[m];
        }
        kept[j] = e;
        for (int h = 0; h < H; ++h) {
            Wk[(size_t) h * CP + j] = (float) y[h];
        }
    });
    double all = 0, in = 0;
    for (size_t i = 0; i < W0.size(); ++i) {
        all += (double) W0[i] * W0[i] + (double) W1[i] * W1[i];
    }
    for (double e : kept) {
        in += e;
    }
    resid = all > 0 ? (all - in) / all : 0;
    resid = resid < 0 ? 0 : resid;
}

const std::vector<float> & patch_embed::weight(int frames) {
    if (frames == frames_cur) {
        return Wk;
    }
    frames_cur = frames;
    if (PW == 2 * CP) {  // room for both frames: the weights as they are
        Wk.resize((size_t) H * PW);
        for (int h = 0; h < H; ++h) {
            memcpy(Wk.data() + (size_t) h * PW, W0.data() + (size_t) h * CP, CP * sizeof(float));
            memcpy(Wk.data() + (size_t) h * PW + CP, W1.data() + (size_t) h * CP, CP * sizeof(float));
        }
    } else if (frames == 1) {
        Wk.resize((size_t) H * CP);
        for (size_t i = 0; i < Wk.size(); ++i) {
            Wk[i] = W0[i] + W1[i];
        }
        basis = false;  // (Wk is not the basis any more)
    } else {
        build_basis();
    }
    return Wk;
}

void patch_embed::row(const float * p0, const float * p1, float * dst) const {
    if (PW == 2 * CP) {
        memcpy(dst, p0, CP * sizeof(float));
        memcpy(dst + CP, p1 ? p1 : p0, CP * sizeof(float));
    } else if (!p1) {
        memcpy(dst, p0, CP * sizeof(float));
    } else {
        for (int j = 0; j < CP; ++j) {
            const float * a = A.data() + (size_t) j * 2 * CP;
            dst[j]          = dotf(a, p0, CP) + dotf(a + CP, p1, CP);
        }
    }
}

}  // namespace axera
