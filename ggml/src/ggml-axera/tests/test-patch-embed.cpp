// Host test of core/patch_embed: whatever the kernel's px row width, px W^T must be the two-frame projection
// W0 p0 + W1 p1 (exactly, or within the stated residual when the tower is wider than a px row).

#include "../core/patch_embed.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace axera;

namespace {

void serial(int64_t n, const std::function<void(int64_t)> & fn) {
    for (int64_t i = 0; i < n; ++i) {
        fn(i);
    }
}

// the relative error of px W^T against W0 p0 + W1 p1 over random patches
double check(const std::vector<float> & w0, const std::vector<float> & w1, int H, int CP, int width, int frames, std::mt19937 & rng) {
    patch_embed pe;
    if (!pe.init(w0.data(), w1.data(), H, CP, width, serial)) {
        fprintf(stderr, "FAIL init\n");
        exit(1);
    }
    const std::vector<float> & W = pe.weight(frames);
    std::normal_distribution<float> nd;
    double num = 0, den = 0;
    for (int t = 0; t < 64; ++t) {
        std::vector<float> p0(CP), p1(CP), px(width);
        for (int i = 0; i < CP; ++i) {
            p0[i] = nd(rng);
            p1[i] = frames == 2 ? nd(rng) : p0[i];
        }
        pe.row(p0.data(), frames == 2 ? p1.data() : nullptr, px.data());
        for (int h = 0; h < H; ++h) {
            double want = 0, got = 0;
            for (int m = 0; m < CP; ++m) {
                want += (double) w0[(size_t) h * CP + m] * p0[m] + (double) w1[(size_t) h * CP + m] * p1[m];
            }
            for (int m = 0; m < width; ++m) {
                got += (double) W[(size_t) h * width + m] * px[m];
            }
            num += (got - want) * (got - want);
            den += want * want;
        }
    }
    printf("H %4d CP %3d px width %3d frames %d: rel err %.2e, basis residual %.2e\n", H, CP, width, frames, std::sqrt(num / den),
           pe.pair_residual());
    return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1) {  // the cost at a real tower's size: test-patch-embed <H> <CP>
        const int          H = atoi(argv[1]), CP = argc > 2 ? atoi(argv[2]) : 768;
        std::mt19937       rng(1);
        std::normal_distribution<float> nd;
        std::vector<float> w0((size_t) H * CP), w1((size_t) H * CP);
        for (auto & v : w0) {
            v = nd(rng);
        }
        for (size_t i = 0; i < w1.size(); ++i) {
            w1[i] = w0[i] + 0.1f * nd(rng);
        }
        return check(w0, w1, H, CP, CP, 2, rng) < 1 ? 0 : 1;
    }
    std::mt19937                    rng(7);
    std::normal_distribution<float> nd;
    const int                       CP = 48;
    bool                            ok = true;
    for (int H : { 32, 48, 72 }) {
        // weights like a trained tower's: both frames' columns in one CP-dimensional space, plus a small tail
        std::vector<float> b((size_t) H * CP), w0((size_t) H * CP), w1((size_t) H * CP);
        for (auto & v : b) {
            v = nd(rng);
        }
        for (int f = 0; f < 2; ++f) {
            std::vector<float> mix((size_t) CP * CP);
            for (auto & v : mix) {
                v = nd(rng) / std::sqrt((float) CP);
            }
            std::vector<float> & w = f ? w1 : w0;
            for (int h = 0; h < H; ++h) {
                for (int m = 0; m < CP; ++m) {
                    double s = 0;
                    for (int c = 0; c < CP; ++c) {
                        s += b[(size_t) h * CP + c] * mix[(size_t) c * CP + m];
                    }
                    w[(size_t) h * CP + m] = (float) s + 1e-3f * nd(rng);
                }
            }
        }
        ok &= check(w0, w1, H, CP, CP, 1, rng) < 1e-5;
        ok &= check(w0, w1, H, CP, 2 * CP, 1, rng) < 1e-5;
        ok &= check(w0, w1, H, CP, 2 * CP, 2, rng) < 1e-5;
        ok &= check(w0, w1, H, CP, CP, 2, rng) < (H <= CP ? 1e-5 : 5e-3);
    }
    printf(ok ? "OK\n" : "FAIL\n");
    return ok ? 0 : 1;
}
