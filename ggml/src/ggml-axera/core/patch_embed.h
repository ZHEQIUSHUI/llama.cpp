#pragma once

// The patch projection of a vision tower with a temporal merge of two frames (Qwen-VL: a Conv3D of depth 2):
//     e = W0 p0 + W1 p1      (p0, p1: one patch's pixels in the two frames; a still image is both frames)
//
// The vision kernels take the pixel rows `px` and the projection weight as inputs and compute px W^T. What the host
// puts in those two inputs depends on the kernel's px row width, not on the kernel otherwise:
//   - width == 2 * CP: the kernel has room for both frames: W = [W0 W1], px = [p0 p1] (exact);
//   - width == CP (one frame): a still is W = W0 + W1, px = p0 (exact). A frame pair cannot be one pixel row, but the
//     kernel does not care what the row means: the host gives it the coordinates of e in an orthonormal basis Q of
//     the column space of [W0 W1], W = Q and px = Q^T (W0 p0 + W1 p1). Exact when the tower's width H <= CP; else
//     [W0 W1] has rank up to H > CP and Q keeps its CP leading directions (the trained weights leave ~1e-5 of
//     their energy outside: the rounding of the bf16 inputs is larger).
// This file is plain host math (no SDK): tests/test-patch-embed.cpp.

#include <cstdint>
#include <functional>
#include <vector>

namespace axera {

class patch_embed {
  public:
    // run fn(i) for i in [0, n), possibly on several threads
    using parallel = std::function<void(int64_t, const std::function<void(int64_t)> &)>;

    // w0, w1: [H][CP] (a patch's pixels in the weights' own order); width: the kernel's px row
    bool init(const float * w0, const float * w1, int H, int CP, int width, parallel par);

    int width() const { return PW; }
    // the kernel's weight [H][width] for calls of `frames` (1 or 2) frames; the same object until frames changes
    const std::vector<float> & weight(int frames);
    // one patch's px row [width] (p1 == nullptr: a still); weight(frames) of the same frame count must be current
    void row(const float * p0, const float * p1, float * dst) const;
    // the share of [W0 W1]'s energy outside the frame pairs' basis (0: exact)
    double pair_residual() const { return resid; }

  private:
    int      H = 0, CP = 0, PW = 0;
    int      frames_cur = 0;
    parallel par;
    std::vector<float> W0, W1;  // [H][CP]
    std::vector<float> Wk;      // the weight of frames_cur
    std::vector<float> A;       // one-frame kernels, pairs: [CP][2 * CP], px = A [p0 p1]
    bool               basis = false;
    double             resid = 0;

    void build_basis();
};

}  // namespace axera
