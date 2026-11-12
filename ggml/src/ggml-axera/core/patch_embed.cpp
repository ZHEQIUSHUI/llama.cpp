#include "patch_embed.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <thread>

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

// orthonormal columns: q [n][m] (column j at q + j * m), n <= m, by Cholesky QR done twice (the second pass restores
// the orthogonality the first loses to the columns' conditioning): a few large parallel loops, not one per column
void orthonormalize(std::vector<double> & q, int n, int m, const patch_embed::parallel & par) {
    std::vector<double> g((size_t) n * n), l((size_t) n * n, 0.0), row((size_t) m * n);
    for (int pass = 0; pass < 2; ++pass) {
        par(n, [&](int64_t i) {  // the Gram matrix (upper triangle)
            const double * qi = q.data() + (size_t) i * m;
            for (int j = (int) i; j < n; ++j) {
                g[(size_t) i * n + j] = dot(qi, q.data() + (size_t) j * m, m);
            }
        });
        // g = L L^T, L lower triangular by rows (every inner loop runs along two rows)
        for (int i = 0; i < n; ++i) {
            double * li = l.data() + (size_t) i * n;
            for (int j = 0; j <= i; ++j) {
                const double * lj = l.data() + (size_t) j * n;
                double         s  = g[(size_t) j * n + i] - dot(li, lj, j);
                if (j == i) {
                    li[i] = std::sqrt(s > 1e-30 ? s : 1e-30);
                } else {
                    li[j] = s / lj[j];
                }
            }
        }
        // Q = Y L^-T, row by row of Y (x L^T = y: x_j = (y_j - sum_k<j x_k L[j][k]) / L[j][j])
        par(m, [&](int64_t h) {
            double * x = row.data() + (size_t) h * n;
            for (int j = 0; j < n; ++j) {
                const double * lj = l.data() + (size_t) j * n;
                x[j]              = (q[(size_t) j * m + h] - dot(x, lj, j)) / lj[j];
            }
        });
        par(n, [&](int64_t j) {
            double * qj = q.data() + (size_t) j * m;
            for (int h = 0; h < m; ++h) {
                qj[h] = row[(size_t) h * n + j];
            }
        });
    }
}

}  // namespace

patch_embed::parallel patch_embed::threads(int n) {
    return [n](int64_t count, const std::function<void(int64_t)> & fn) {
        const int t = (int) std::min<int64_t>(n, count);
        if (t <= 1) {
            serial(count, fn);
            return;
        }
        std::vector<std::thread> th;
        for (int w = 0; w < t; ++w) {
            th.emplace_back([&, w] {
                for (int64_t i = count * w / t; i < count * (w + 1) / t; ++i) {
                    fn(i);
                }
            });
        }
        for (auto & x : th) {
            x.join();
        }
    };
}

void patch_embed::serial(int64_t n, const std::function<void(int64_t)> & fn) {
    for (int64_t i = 0; i < n; ++i) {
        fn(i);
    }
}

bool patch_embed::init(const float * w0, const float * w1, int H_, int CP_, int width) {
    if (H_ <= 0 || CP_ <= 0 || (width != CP_ && width != 2 * CP_)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu);
    H  = H_;
    CP = CP_;
    PW = width;
    W0.assign(w0, w0 + (size_t) H * CP);
    W1.assign(w1, w1 + (size_t) H * CP);
    Ws.resize((size_t) H * PW);
    if (PW == 2 * CP) {  // room for both frames: the weights as they are, a still is both frames
        for (int h = 0; h < H; ++h) {
            memcpy(Ws.data() + (size_t) h * PW, W0.data() + (size_t) h * CP, CP * sizeof(float));
            memcpy(Ws.data() + (size_t) h * PW + CP, W1.data() + (size_t) h * CP, CP * sizeof(float));
        }
        Wp    = Ws;
        pairs = true;
    } else {
        for (size_t i = 0; i < Ws.size(); ++i) {
            Ws[i] = W0[i] + W1[i];
        }
        Wp.clear();
        A.clear();
        pairs = false;
    }
    resid = 0;
    return true;
}

const std::vector<float> & patch_embed::weight_pairs(const parallel & par) {
    std::lock_guard<std::mutex> lock(mu);
    if (!pairs) {
        build_basis(par);
        pairs = true;
    }
    return Wp;
}

// Q: the CP leading directions of [W0 W1]'s column space (a randomized range finder with one power iteration: the
// spectrum drops by orders of magnitude after CP, one iteration is at the limit already), and A = Q^T [W0 W1]
void patch_embed::build_basis(const parallel & par) {
    A.assign((size_t) CP * 2 * CP, 0.0f);
    Wp.assign((size_t) H * CP, 0.0f);
    if (H <= CP) {  // e itself fits a px row: W = [I 0], px = [e 0]
        for (int h = 0; h < H; ++h) {
            memcpy(A.data() + (size_t) h * 2 * CP, W0.data() + (size_t) h * CP, CP * sizeof(float));
            memcpy(A.data() + (size_t) h * 2 * CP + CP, W1.data() + (size_t) h * CP, CP * sizeof(float));
            Wp[(size_t) h * CP + h] = 1.0f;
        }
        resid = 0;
        return;
    }
    // The products with the weights run in float (their rows are float; a dot product of 2 CP terms keeps ~1e-6, below
    // the kernels' bf16 inputs), the orthonormalization in double. Y = [W0 W1] Omega, Omega [2 CP][CP] standard normal
    // (a fixed seed: the same basis every run), stored by column
    std::vector<float> om((size_t) 2 * CP * CP);
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
            om[i]     = (float) (a * std::cos(b));
            om[i + 1] = (float) (a * std::sin(b));
        }
    }
    std::vector<double> q((size_t) CP * H);  // column j at q + j * H
    par(CP, [&](int64_t j) {
        double *      y  = q.data() + (size_t) j * H;
        const float * oj = om.data() + (size_t) j * 2 * CP;
        for (int h = 0; h < H; ++h) {
            y[h] = (double) dotf(W0.data() + (size_t) h * CP, oj, CP) + dotf(W1.data() + (size_t) h * CP, oj + CP, CP);
        }
    });
    om.clear();
    om.shrink_to_fit();
    orthonormalize(q, CP, H, par);
    // t = [W0 W1]^T y for a column y of q: [2 CP] floats
    const auto times_wt = [&](const double * y, float * t) {
        std::fill(t, t + 2 * CP, 0.0f);
        for (int h = 0; h < H; ++h) {
            const float * a = W0.data() + (size_t) h * CP, *b = W1.data() + (size_t) h * CP;
            const float   v = (float) y[h];
            for (int m = 0; m < CP; ++m) {
                t[m] += v * a[m];
            }
            for (int m = 0; m < CP; ++m) {
                t[CP + m] += v * b[m];
            }
        }
    };
    // one power iteration: q <- orth([W0 W1] [W0 W1]^T q)
    {
        std::vector<double> z((size_t) CP * H);
        par(CP, [&](int64_t j) {
            std::vector<float> t((size_t) 2 * CP);
            times_wt(q.data() + (size_t) j * H, t.data());
            double * o = z.data() + (size_t) j * H;
            for (int h = 0; h < H; ++h) {
                o[h] = (double) dotf(W0.data() + (size_t) h * CP, t.data(), CP) + dotf(W1.data() + (size_t) h * CP, t.data() + CP, CP);
            }
        });
        q.swap(z);
    }
    orthonormalize(q, CP, H, par);
    std::vector<double> kept(CP, 0.0);
    par(CP, [&](int64_t j) {
        const double * y = q.data() + (size_t) j * H;
        float *        o = A.data() + (size_t) j * 2 * CP;
        times_wt(y, o);
        double e = 0;
        for (int m = 0; m < 2 * CP; ++m) {
            e += (double) o[m] * o[m];
        }
        kept[j] = e;
        for (int h = 0; h < H; ++h) {
            Wp[(size_t) h * CP + j] = (float) y[h];
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
