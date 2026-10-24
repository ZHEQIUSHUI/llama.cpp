// Axera NPU backend: attention on the llama.cpp KV cache for the graph templates (model independent)

#include "ggml-axera-attn.h"

axera_kv_batch::axera_kv_batch(const ggml_tensor * k_idxs, const ggml_tensor * kqm, int64_t N, int BC, int64_t S) :
    k_idxs(k_idxs), kqm(kqm), N(N), BC(BC), S(S), W(std::min<int64_t>(N, (kqm->ne[0] + BC - 1) / BC * BC)) {}

int axera_kv_batch::mask_row(int t, ggml_bf16_t * dst, row_span * span) const {
    const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;  // finite, as the kernels' own masking: no inf - inf
    const char *   row  = (const char *) kqm->data + (kqm->ne[3] == 1 ? t * kqm->nb[1] :
                                                          (t / kqm->ne[1]) * kqm->nb[3] + (t % kqm->ne[1]) * kqm->nb[1]);
    const int64_t  n    = std::min<int64_t>(W, kqm->ne[0]);
    auto *         d    = (uint16_t *) dst;
    // fast path for the usual 0 / -inf masks (vectorized); any other value falls back to a conversion
    bool other = false;
    if (kqm->type == GGML_TYPE_F16) {
        const auto * s = (const uint16_t *) row;
        for (int64_t c = 0; c < n; ++c) {
            d[c] = s[c] == 0 ? 0 : ninf;
            other |= s[c] != 0 && s[c] != 0xFC00;
        }
    } else {
        const auto * s = (const uint32_t *) row;
        for (int64_t c = 0; c < n; ++c) {
            d[c] = s[c] == 0 ? 0 : ninf;
            other |= s[c] != 0 && s[c] != 0xFF800000u;
        }
    }
    if (other) {
        for (int64_t c = 0; c < n; ++c) {
            const float v = kqm->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[c]) :
                                                         ((const float *) row)[c];
            d[c] = std::isinf(v) && v < 0 ? ninf : ggml_fp32_to_bf16(v).bits;
        }
    }
    std::fill(d + n, d + W, ninf);
    int64_t last = n - 1;
    while (last >= 0 && d[last] == ninf) {
        --last;
    }
    if (span) {
        int64_t first = 0, cnt = 0;
        while (first <= last && d[first] == ninf) {
            ++first;
        }
        for (int64_t c = first; c <= last; ++c) {
            cnt += d[c] != ninf;
        }
        *span = { last < 0 ? -1 : first, last, cnt };
    }
    return (int) (last < 0 ? 1 : last / BC + 1);
}

void axera_flash_prepare(const axera_kv_batch & kb, axera_fd_kernels & fk, int R, int t0) {
    using row_span = axera_kv_batch::row_span;
    axera_timer              tm("fd prepare (cpu)");
    const int64_t            W    = kb.W;
    const int64_t            nseg = (W + kb.S - 1) / kb.S;
    const uint16_t           ninf = ggml_fp32_to_bf16(-3e38f).bits;
    std::vector<ggml_bf16_t> rows((size_t) R * W);
    std::vector<row_span>    span(R);
    {
        axera_timer tr("fd prepare: mask rows");
        axera_host_pool().run(R, [&](int64_t r) { kb.mask_row(t0 + (int) r, rows.data() + (size_t) r * W, &span[r]); });
    }
    // does row r see a cell in [c0, c1)? Rows visible in one run (causal) answer from their span
    auto visible = [&](int r, int64_t c0, int64_t c1) {
        const row_span & sp = span[r];
        if (sp.lo < 0 || sp.hi < c0 || sp.lo >= c1) {
            return false;
        }
        if (sp.n == sp.hi - sp.lo + 1) {
            return true;
        }
        const auto * d = (const uint16_t *) rows.data() + (size_t) r * W;
        for (int64_t c = std::max(c0, sp.lo); c < std::min(c1, sp.hi + 1); ++c) {
            if (d[c] != ninf) {
                return true;
            }
        }
        return false;
    };
    fk.nblk.assign(nseg, 0);
    fk.runs.assign(nseg * 2 * fk.cores, 0);
    fk.parts.clear();
    if ((int64_t) fk.MASK.size() < nseg) {
        fk.MASK.resize(nseg);
    }
    for (int64_t g = 0; g < nseg; ++g) {
        const int64_t c0 = g * kb.S, c1 = std::min<int64_t>(W, c0 + kb.S);
        int64_t       last = -1;  // the segment's last cell any row sees
        for (int r = 0; r < R; ++r) {
            const row_span & sp = span[r];
            if (sp.lo < 0 || sp.lo >= c1 || sp.hi < c0) {
                continue;
            }
            const auto * d = (const uint16_t *) rows.data() + (size_t) r * W;
            int64_t      c = std::min(c1 - 1, sp.hi);
            while (c > c0 + last && d[c] == ninf) {
                --c;
            }
            last = std::max(last, c - c0);
        }
        if (last < 0) {
            continue;
        }
        const int nb = (int) (last / fk.bc + 1);
        fk.nblk[g]   = nb;
        // the cores' runs of blocks; a core without blocks scans block 0 and its result is ignored
        static const int dbg_core = [] {  // GGML_AXERA_SEG_CORE=c: debugging, every block on core c
            const char * e = std::getenv("GGML_AXERA_SEG_CORE");
            return e ? std::atoi(e) : -1;
        }();
        const int per = dbg_core >= 0 ? nb : (nb + fk.cores - 1) / fk.cores;
        for (int c = 0; c < fk.cores; ++c) {
            const int b0 = dbg_core >= 0 ? (c == dbg_core ? 0 : nb) : std::min(nb, c * per);
            const int b1 = dbg_core >= 0 ? (c == dbg_core ? nb : nb) : std::min(nb, (c + 1) * per);
            int32_t * run = fk.runs.data() + (g * fk.cores + c) * 2;
            run[0]        = b1 > b0 ? b0 : 0;
            run[1]        = b1 > b0 ? b1 : 1;
            if (b1 <= b0) {
                continue;
            }
            axera_fd_kernels::part pt{ g, c, std::vector<uint8_t>(R, 0) };
            for (int r = 0; r < R; ++r) {
                pt.seen[r] = visible(r, c0 + (int64_t) b0 * fk.bc, std::min<int64_t>(c1, c0 + (int64_t) b1 * fk.bc));
            }
            if (std::find(pt.seen.begin(), pt.seen.end(), 1) != pt.seen.end()) {
                fk.parts.push_back(std::move(pt));
            }
        }
        if (!fk.MASK[g]) {
            fk.MASK[g] = axera_cmm_alloc(axera_input_size(fk.seg, "mask"));
        }
        axera_timer tc("fd prepare: segment masks");
        auto *      mk = (uint16_t *) fk.MASK[g]->vir;
        axera_host_pool().run(nb, [&](int64_t blk) {
            const int64_t col = c0 + blk * fk.bc, n = std::max<int64_t>(0, std::min<int64_t>(fk.bc, c1 - col));
            for (int r = 0; r < R; ++r) {
                uint16_t * dst = mk + (blk * R + r) * fk.bc;
                memcpy(dst, (const uint16_t *) rows.data() + (size_t) r * W + col, n * sizeof(uint16_t));
                std::fill(dst + n, dst + fk.bc, ninf);
            }
        });
        fk.MASK[g]->flush(mk, (size_t) nb * R * fk.bc * sizeof(ggml_bf16_t));
    }
}

void axera_write_kv(const axera_kv_batch & kb, std::vector<axera_kv_region> & kv, axera_fd_kernels & fk, int R, int t0) {
    axera_timer   tm("fd kv write (cpu)");
    const int64_t st = kb.stream_of(t0);
    fk.K->invalidate(fk.K->vir, fk.K->size);
    fk.V->invalidate(fk.V->vir, fk.V->size);
    for (auto & kr : kv) {
        const char * src = (const char *) (kr.v ? fk.V : fk.K)->vir;
        char *       run = nullptr;  // flush runs of consecutive cells at once
        size_t       len = 0;
        for (int r = 0; r < R; ++r) {
            char * dst = kr.base + st * kr.stride + kb.cell_of(t0 + r) * kr.row;
            memcpy(dst, src + r * kr.row, kr.row);
            if (run && run + len == dst) {
                len += kr.row;
                continue;
            }
            if (run) {
                kr.b->flush(run, len);
            }
            run = dst;
            len = kr.row;
        }
        kr.b->flush(run, len);
    }
}

// head kernel, the new K / V rows to their cells, one segment kernel call per segment in use, merged on the CPU
// (m = max m_i, l = sum e^(m_i - m) l_i, o = sum e^(m_i - m) o_i / l), tail kernel
void axera_flash_attn(const axera_kv_batch & kb, std::vector<axera_kv_region> & kv, axera_fd_kernels & fk,
                      axera_fd_io & io, int R, int t0, int il, int64_t p) {
    auto run = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                   std::vector<AX_ENGINE_IO_BUFFER_T> & out, const char * what) {
        const int ret = axera_run_sync(k, in, out);
        if (ret != 0) {
            GGML_ABORT("axera template: RunSync failed (0x%x) at layer %d (%s), position %" PRId64 "\n", ret, il, what,
                       p);
        }
    };
    {
        axera_timer tm("fd head");
        run(fk.head, io.hin, io.hout, "attention head");
    }
    const int64_t st = kb.stream_of(t0);
    axera_write_kv(kb, kv, fk, R, t0);
    const auto & kk  = kv[0].v ? kv[1] : kv[0];
    const auto & vv  = kv[0].v ? kv[0] : kv[1];
    const int    nr  = fk.hkv * fk.gr;  // rows: KV head hk, row gq * R + token
    std::vector<float>   am(nr, 0.0f), al(nr, 0.0f), ao((size_t) nr * fk.d, 0.0f);
    std::vector<uint8_t> started(nr, 0);
    size_t               pi = 0;  // fk.parts in segment order
    for (size_t g = 0; g < fk.nblk.size(); ++g) {
        if (fk.nblk[g] == 0) {
            continue;
        }
        io.sin[io.kc]   = axera_io(kk.b, kk.base + st * kk.stride + g * kb.S * kk.row, (uint32_t) kk.size);
        io.sin[io.vc]   = axera_io(vv.b, vv.base + st * vv.stride + g * kb.S * vv.row, (uint32_t) vv.size);
        io.sin[io.mask] = axera_io(fk.MASK[g].get(), fk.MASK[g]->vir, (uint32_t) axera_input_size(fk.seg, "mask"));
        if (fk.cores > 1) {
            memcpy(fk.BLK->vir, fk.runs.data() + g * fk.cores * 2, fk.cores * 2 * sizeof(int32_t));
        } else {
            *(int32_t *) fk.BLK->vir = fk.nblk[g];
        }
        fk.BLK->flush(fk.BLK->vir, fk.BLK->size);
        {
            axera_timer tm(R == 1 ? "fd seg (decode)" : "fd seg (prefill)");
            run(fk.seg, io.sin, io.sout, "attention segment");
        }
        axera_timer tm("fd merge (cpu)");
        fk.M->invalidate(fk.M->vir, fk.M->size);
        fk.Lsum->invalidate(fk.Lsum->vir, fk.Lsum->size);
        fk.O->invalidate(fk.O->vir, fk.O->size);
        for (; pi < fk.parts.size() && fk.parts[pi].g == (int64_t) g; ++pi) {
            const auto & pt = fk.parts[pi];
            const auto * mi = (const ggml_bf16_t *) fk.M->vir + (size_t) pt.c * nr;
            const auto * li = (const float *) fk.Lsum->vir + (size_t) pt.c * nr;
            const auto * oi = (const ggml_bf16_t *) fk.O->vir + (size_t) pt.c * nr * fk.d;
            axera_host_pool().run(nr, [&](int64_t j) {
                if (!pt.seen[(j % fk.gr) % R]) {
                    return;
                }
                const float  mj = ggml_bf16_to_fp32(mi[j]);
                float *      o  = ao.data() + (size_t) j * fk.d;
                const auto * oj = oi + (size_t) j * fk.d;
                if (!started[j]) {
                    started[j] = 1;
                    am[j]      = mj;
                    al[j]      = li[j];
                    for (int i = 0; i < fk.d; ++i) {
                        o[i] = ggml_bf16_to_fp32(oj[i]);
                    }
                    return;
                }
                const float mn = std::max(am[j], mj), a = std::exp(am[j] - mn), b = std::exp(mj - mn);
                am[j] = mn;
                al[j] = a * al[j] + b * li[j];
                for (int i = 0; i < fk.d; ++i) {
                    o[i] = a * o[i] + b * ggml_bf16_to_fp32(oj[i]);
                }
            });
        }
    }
    // attention output [R][HQ][D]: query head hk * G + gq of token t is row gq * R + t of KV head hk
    const int G   = fk.gr / R;
    auto *    att = (ggml_bf16_t *) fk.ATT->vir;
    axera_host_pool().run(fk.hkv * G, [&](int64_t hg) {
        const int hk = (int) hg / G, gq = (int) hg % G;
        for (int tt = 0; tt < R; ++tt) {
            const int     j   = hk * fk.gr + gq * R + tt;
            const float   inv = al[j] > 0.0f ? 1.0f / al[j] : 0.0f;
            ggml_bf16_t * dst = att + ((size_t) tt * fk.hkv * G + hk * G + gq) * fk.d;
            for (int i = 0; i < fk.d; ++i) {
                dst[i] = ggml_fp32_to_bf16(ao[(size_t) j * fk.d + i] * inv);
            }
        }
    });
    fk.ATT->flush(fk.ATT->vir, fk.ATT->size);
    {
        axera_timer tm("fd tail");
        run(fk.tail, io.tin, io.tout, "attention tail");
    }
}

axera_rope::axera_rope(const ggml_tensor * inp_pos, int T, int ROT, float freq_scale, const std::vector<float> & inv_freq,
                       int mode, const int sections[4]) :
    pos((const int32_t *) inp_pos->data),
    T(T),
    ROT(ROT),
    freq_scale(freq_scale),
    inv_freq(inv_freq.data()),
    mrope((mode & GGML_ROPE_TYPE_MROPE) && inp_pos->ne[0] == 4 * (int64_t) T),
    sect(ROT / 2, 0) {
    for (int i = 0; mrope && i < ROT / 2; ++i) {
        const int * sc = sections;
        const int   n  = sc[0] + sc[1] + sc[2] + sc[3], k = n > 0 ? i % n : 0;
        if (mode == GGML_ROPE_TYPE_IMROPE) {
            sect[i] = k % 3 == 1 && k < 3 * sc[1] ? 1 : k % 3 == 2 && k < 3 * sc[2] ? 2 : k % 3 == 0 && k < 3 * sc[0] ? 0 : 3;
        } else {
            sect[i] = k < sc[0] ? 0 : k < sc[0] + sc[1] ? 1 : k < sc[0] + sc[1] + sc[2] ? 2 : 3;
        }
    }
}

void axera_rope::rows(int t0, int n, ggml_backend_axera_buffer_context * cb, ggml_backend_axera_buffer_context * sb) const {
    rows_to(t0, n, (ggml_bf16_t *) cb->vir, (ggml_bf16_t *) sb->vir);
    cb->flush(cb->vir, cb->size);
    sb->flush(sb->vir, sb->size);
}

void axera_rope::rows_to(int t0, int n, ggml_bf16_t * c, ggml_bf16_t * s) const {  // the positions of tokens t0 .. t0 + n - 1
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < ROT / 2; ++i) {
            const int32_t p        = pos[(mrope ? sect[i] * T : 0) + t0 + r];
            const float   a        = (float) p * freq_scale * inv_freq[i];
            c[r * (ROT / 2) + i]   = ggml_fp32_to_bf16(std::cos(a));
            s[r * (ROT / 2) + i]   = ggml_fp32_to_bf16(std::sin(a));
        }
    }
}
