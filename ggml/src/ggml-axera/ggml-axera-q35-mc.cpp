// Axera NPU backend, Qwen3.5 graph template: the multi-core prefill chain

#include "ggml-axera-q35.h"

// multi-core prefill chain of chunk v (P tokens from t0): per layer gdn | head, segments, tail; then mlp. The
// residual leaves the last layer as base (pf.A) + the partials (PART0)
void axera_q35_batch::run_mc(int v, int t0, int il0, int il1) {
    axera_q35::prefill &    pf  = m.pre[v];
    const int               P   = pf.P;
    const int               key = axera_q35_fd_key(m, v);  // a MoE group's chunk: its own split attention buffers
    axera_q35::mc_kernels & mk  = m.mc.at(P);
    axera_q35::fd_kernels & fk  = m.fd.at(key);
    // GGML_AXERA_MC_CHECK=1: debugging, report the first kernel output holding a NaN / Inf (bf16, f16 or f32 by name)
    // (2: only invalidate the outputs, 3: only wait 300 us after each call)
    static const int check_mode = [] {
        const char * e = std::getenv("GGML_AXERA_MC_CHECK");
        return e ? std::atoi(e) : 0;
    }();
    static const bool check = check_mode == 1;
    static bool reported = false;
    auto run = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                   const char * what, int il) {
        {
            axera_timer tm(what);
            const int   ret = axera_run_sync(k, in, out);
            if (ret != 0) {
                GGML_ABORT("qwen35 template: RunSync failed (0x%x) at layer %d (%s)\n", ret, il, what);
            }
        }
        if (check_mode == 2) {
            for (auto & b : out) {
                AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
            }
        } else if (check_mode == 3) {  // GGML_AXERA_MC_WAIT=<kind>: only after those calls ("mc gdn", ...)
            static const char * kind = std::getenv("GGML_AXERA_MC_WAIT");
            static const int    us   = std::getenv("GGML_AXERA_MC_WAIT_US") ? std::atoi(std::getenv("GGML_AXERA_MC_WAIT_US")) : 300;
            if (!kind || strstr(what, kind)) {
                std::this_thread::sleep_for(std::chrono::microseconds(us));
            }
        }
        for (uint32_t i = 0; check && !reported && i < out.size(); ++i) {
            const AX_ENGINE_IOMETA_T & om = k->info->pOutputs[i];
            AX_SYS_MinvalidateCache(out[i].phyAddr, out[i].pVirAddr, out[i].nSize);
            const std::string n  = om.pName;
            const int         es = om.eDataType == AX_ENGINE_DT_FLOAT32 ? 4 : 2;
            const bool        f16 = om.eDataType == AX_ENGINE_DT_FLOAT16 || n == "k_out" || n == "v_out";
            int64_t           bad = -1;
            for (int64_t e = 0; e < (int64_t) out[i].nSize / es && bad < 0; ++e) {
                float v;
                if (es == 4) {
                    v = ((const float *) out[i].pVirAddr)[e];
                } else if (f16) {
                    v = ggml_fp16_to_fp32(((const ggml_fp16_t *) out[i].pVirAddr)[e]);
                } else {
                    v = ggml_bf16_to_fp32(((const ggml_bf16_t *) out[i].pVirAddr)[e]);
                }
                bad = std::isfinite(v) ? -1 : e;
            }
            if (bad >= 0) {
                reported = true;
                GGML_LOG_ERROR("qwen35 template: mc check: %s layer %d output %s element %" PRId64 " of %u is not "
                               "finite (chunk at token %d, cell %" PRId64 ", W %" PRId64 ")\n", what, il, n.c_str(), bad,
                               out[i].nSize / es, t0, kv.cell_of(t0), kv.W);
            }
        }
    };
    const size_t nseg = fk.nblk.size();
    // the slots hold `groups` segments (one slot per segment core). More segments in use: group 0 keeps the CPU
    // merge of the segments done so far, the other groups take the next ones in turns
    static const size_t max_groups = [] {  // GGML_AXERA_MC_GROUPS=n: debugging, use n segment groups of slots only
        const char * e = std::getenv("GGML_AXERA_MC_GROUPS");
        return e ? (size_t) std::max(2, std::atoi(e)) : SIZE_MAX;
    }();
    const size_t groups = std::min(max_groups, (size_t) mk.np / fk.cores);
    size_t       nused  = 0;
    for (size_t g = 0; g < nseg; ++g) {
        nused += fk.nblk[g] != 0;
    }
    const bool spill = nused > groups;
    if (spill && groups < 2) {
        GGML_ABORT("qwen35 template: %zu KV segments x %d cores need more than the %d merge slots of %s\n", nused,
                   fk.cores, mk.np, "qwen35_attn_tail_mc");
    }
    // slots without data this batch: m = -3e38 (weight 0 in the merge) and l = o = 0, as 0 * a stale NaN is NaN
    const uint16_t ninf = ggml_fp32_to_bf16(-3e38f).bits;
    auto clear_slot = [&](size_t slot) {
        auto * p = (uint16_t *) ((char *) mk.MALL->vir + slot * mk.ms);
        std::fill(p, p + mk.ms / sizeof(uint16_t), ninf);
        mk.MALL->flush(p, mk.ms);
        char * l = (char *) mk.LALL->vir + slot * mk.ls;
        char * o = (char *) mk.OALL->vir + slot * mk.os;
        memset(l, 0, mk.ls);
        memset(o, 0, mk.os);
        mk.LALL->flush(l, mk.ls);
        mk.OALL->flush(o, mk.os);
    };
    for (int i = 0; i < mk.np; ++i) {
        clear_slot(i);
    }
    // fold every slot (and slot 0 itself if acc) into slot 0, clear the others. Rows: KV head, g * P + token;
    // m bf16, l f32, o bf16 unnormalized (sum e^(s - m) v). Slot 0 gets m rounded to bf16 with l / o rescaled to it
    auto merge_slots = [&](bool acc) {
        axera_timer tm("mc attn merge (cpu, > slots)");
        const int   nr = fk.hkv * fk.gr, d = fk.d;
        mk.MALL->invalidate(mk.MALL->vir, mk.MALL->size);
        mk.LALL->invalidate(mk.LALL->vir, mk.LALL->size);
        mk.OALL->invalidate(mk.OALL->vir, mk.OALL->size);
        auto * M = (ggml_bf16_t *) mk.MALL->vir;
        auto * Ls = (float *) mk.LALL->vir;
        auto * O = (ggml_bf16_t *) mk.OALL->vir;
        axera_host_pool().run(nr, [&](int64_t j) {
            float mx = -3e38f;
            for (int sl = acc ? 0 : 1; sl < mk.np; ++sl) {
                mx = std::max(mx, ggml_bf16_to_fp32(M[(size_t) sl * nr + j]));
            }
            std::vector<float> o(d, 0.0f);
            float              l = 0.0f;
            for (int sl = acc ? 0 : 1; sl < mk.np && mx > -1e37f; ++sl) {
                const float w = std::exp(ggml_bf16_to_fp32(M[(size_t) sl * nr + j]) - mx);
                if (w == 0.0f) {
                    continue;
                }
                l += w * Ls[(size_t) sl * nr + j];
                const ggml_bf16_t * oi = O + ((size_t) sl * nr + j) * d;
                for (int i = 0; i < d; ++i) {
                    o[i] += w * ggml_bf16_to_fp32(oi[i]);
                }
            }
            ggml_bf16_t * o0 = O + (size_t) j * d;
            if (mx <= -1e37f) {  // no cell seen yet: an empty slot
                M[j]  = ggml_fp32_to_bf16(-3e38f);
                Ls[j] = 0.0f;
                std::fill(o0, o0 + d, ggml_fp32_to_bf16(0.0f));
                return;
            }
            const ggml_bf16_t mb = ggml_fp32_to_bf16(mx);
            const float       sc = std::exp(mx - ggml_bf16_to_fp32(mb));
            M[j]                 = mb;
            Ls[j]                = l * sc;
            for (int i = 0; i < d; ++i) {
                o0[i] = ggml_fp32_to_bf16(o[i] * sc);
            }
        });
        mk.MALL->flush(mk.MALL->vir, mk.ms);
        mk.LALL->flush(mk.LALL->vir, mk.ls);
        mk.OALL->flush(mk.OALL->vir, mk.os);
        for (int i = 1; i < mk.np; ++i) {
            clear_slot(i);
        }
    };
    const int64_t st = kv.stream_of(t0);
    for (int il = il0; il < (il1 < 0 ? L : il1); ++il) {
        axera_q35_layer &        ly = m.layers[il];
        axera_q35_layer::mc_io & io = ly.mc.at(key);
        if (ly.gdn) {
            io.ain[io.si] = io.aout[io.so] = ly.in[io.dsi];  // the current recurrent-cache row
            io.ain[io.ci] = io.aout[io.co] = ly.in[io.dci];
            run(mk.gdn, io.ain, io.aout, "mc gdn", il);
            run(mk.mlp, io.min, io.mout, "mc mlp", il);
            continue;
        }
        run(mk.head, io.ain, io.aout, "mc attn head", il);
        axera_write_kv(kv, ly.kv, fk, P, t0);
        const auto & kk = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
        const auto & vv = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
        size_t       k  = 0;      // segments in the slots since the last merge
        bool         acc = false;  // slot 0 holds merged segments
        for (size_t g = 0; g < nseg; ++g) {
            if (fk.nblk[g] == 0) {
                continue;
            }
            if (spill && 1 + k == groups) {
                merge_slots(acc);
                acc = true;
                k   = 0;
            }
            const size_t s0 = (spill ? 1 + k++ : k++) * fk.cores;  // the segment's first slot
            io.sin[io.kc]   = axera_io(kk.b, kk.base + st * kk.stride + g * m.S * kk.row, (uint32_t) kk.size);
            io.sin[io.vc]   = axera_io(vv.b, vv.base + st * vv.stride + g * m.S * vv.row, (uint32_t) vv.size);
            io.sin[io.mask] = axera_io(fk.MASK[g].get(), fk.MASK[g]->vir, (uint32_t) axera_input_size(fk.seg, "mask"));
            io.sout[io.mo]  = axera_io(mk.MALL.get(), (char *) mk.MALL->vir + s0 * mk.ms, (uint32_t) (fk.cores * mk.ms));
            io.sout[io.lo]  = axera_io(mk.LALL.get(), (char *) mk.LALL->vir + s0 * mk.ls, (uint32_t) (fk.cores * mk.ls));
            io.sout[io.oo]  = axera_io(mk.OALL.get(), (char *) mk.OALL->vir + s0 * mk.os, (uint32_t) (fk.cores * mk.os));
            if (fk.cores > 1) {
                memcpy(fk.BLK->vir, fk.runs.data() + g * fk.cores * 2, fk.cores * 2 * sizeof(int32_t));
            } else {
                *(int32_t *) fk.BLK->vir = fk.nblk[g];
            }
            fk.BLK->flush(fk.BLK->vir, fk.BLK->size);
            run(fk.seg, io.sin, io.sout, "mc attn seg", il);
            for (int c = 0; c < fk.cores; ++c) {  // a core given no block scanned block 0: no data
                const int32_t * r = fk.runs.data() + (g * fk.cores + c) * 2;
                bool            used = false;
                for (const auto & pt : fk.parts) {
                    used = used || (pt.g == (int64_t) g && pt.c == c);
                }
                if (fk.cores > 1 && !used) {
                    clear_slot(s0 + c);
                }
                GGML_UNUSED(r);
            }
        }
        run(mk.tail, io.tin, io.tout, "mc attn tail", il);
        if (spill) {  // the next attention layer starts from empty slots
            for (int i = 0; i < mk.np; ++i) {
                clear_slot(i);
            }
        }
        run(mk.mlp, io.min, io.mout, "mc mlp", il);
    }
}
