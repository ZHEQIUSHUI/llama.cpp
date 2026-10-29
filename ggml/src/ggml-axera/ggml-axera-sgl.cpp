// Axera NPU backend: the whole text model in the fused kernels of sglang_axera_opt (sgl-kernel-axera
// models/qwen38_27b: Qwen3.8-27B), GGML_AXERA_SGL=<dir> with the qwen35 template
//
// One kernel call runs all layers (a runtime loop over blocks of three GDN + MLP and one gated attention + MLP layers)
// on 256 rows (prefill) or 1 row (decode). Weights, recurrent state and the paged KV cache (pages of 256 cells) are
// the kernels' inputs; <dir> holds them as the offline packer wrote them (pack_sgl27.py: <name>.bin of every weight
// bank, manifest.txt "<name> <bytes> <state>"), and the kernels sgl27_r<rows>_v<valid>_h<history>_npu4.axmodel in
// GGML_AXERA_KERNEL_DIR (loaded with the others at initialization). The state banks (GDN state pools, KV pages) are
// allocated here and cleared at position 0. A prompt runs in chunks of 256 positions from a page start (the chunk
// kernels with valid < 256 take a shorter last chunk), the rest token by token through the decode kernel.

#include "ggml-axera-q35.h"

#include <algorithm>
#include <fstream>
#include <tuple>
#include <sstream>

struct axera_sgl {
    struct variant {
        ggml_axera_kernel *                k = nullptr;
        int                                rows = 0, valid = 0;
        bool                               hist = false;
        std::vector<AX_ENGINE_IO_BUFFER_T> in, out;
        axera_cmm                          X, OUT, POS, INDPTR, INDICES, LPL, CLOC;
    };
    std::vector<variant>             vars;
    std::map<std::string, axera_cmm> banks;
    std::vector<std::string>         state;  // banks cleared at position 0
    int                              H = 0, pages = 0;  // pages: the page table's entries (the context / 256)
    int64_t                          pages_kv = 0;      // pages of the KV bank
    int64_t                          next_pos = -1;
};

bool axera_sgl_on() {
    static const bool on = std::getenv("GGML_AXERA_SGL") != nullptr;
    return on;
}

std::shared_ptr<axera_sgl> axera_sgl_load(int H) {
    const std::string dir = std::getenv("GGML_AXERA_SGL");
    auto              s   = std::make_shared<axera_sgl>();
    s->H                  = H;
    // the largest banks first, while the CMM has the holes for them (a bank is one IO: contiguous)
    std::ifstream man(dir + "/manifest.txt");
    std::vector<std::tuple<size_t, std::string, int>> list;
    std::string name;
    size_t      bytes = 0, total = 0;
    int         st    = 0;
    while (man >> name >> bytes >> st) {
        list.emplace_back(bytes, name, st);
    }
    size_t cfree = 0, ctotal = 0;
    axera_cmm_memory(&cfree, &ctotal);
    GGML_LOG_WARN("axera sgl: loading %zu banks, CMM free %.2f of %.2f GiB\n", list.size(), cfree / 1073741824.0,
                  ctotal / 1073741824.0);
    std::sort(list.begin(), list.end(), [](const auto & a, const auto & b) { return std::get<0>(a) > std::get<0>(b); });
    for (const auto & [n_bytes, n_name, n_state] : list) {
        if (getenv("GGML_AXERA_SGL_TRACE")) {
            axera_cmm_memory(&cfree, &ctotal);
            GGML_LOG_WARN("axera sgl: %s %zu (loaded %.2f GiB, CMM free %.2f GiB)\n", n_name.c_str(), n_bytes,
                          total / 1073741824.0, cfree / 1073741824.0);
        }
        axera_cmm b = axera_cmm_alloc(n_bytes);
        if (n_state) {
            memset(b->vir, 0, n_bytes);
            s->state.push_back(n_name);
        } else {
            FILE * f = fopen((dir + "/" + n_name + ".bin").c_str(), "rb");
            if (!f || fread(b->vir, 1, n_bytes, f) != n_bytes) {
                GGML_ABORT("axera sgl: cannot read %s/%s.bin (%zu bytes)\n", dir.c_str(), n_name.c_str(), n_bytes);
            }
            fclose(f);
        }
        b->flush(b->vir, n_bytes);
        total += n_bytes;
        s->banks[n_name] = std::move(b);
    }
    if (s->banks.count("b3_kv")) {
        s->pages_kv = (int64_t) (s->banks.at("b3_kv")->size / (16 * 2 * 256 * 1024 * sizeof(ggml_bf16_t)));
    }
    if (s->banks.empty()) {
        GGML_ABORT("axera sgl: no banks in %s/manifest.txt\n", dir.c_str());
    }
    for (const auto & f : axera_kernels().files) {
        int rows, valid, hist, len = 0;
        if (sscanf(f.first.c_str(), "sgl27_r%d_v%d_h%d%n", &rows, &valid, &hist, &len) != 3 || (size_t) len != f.first.size()) {
            continue;
        }
        axera_sgl::variant v;
        v.k = axera_kernels().get(f.first);
        if (!v.k) {
            GGML_ABORT("axera sgl: cannot load %s\n", f.first.c_str());
        }
        v.rows  = rows;
        v.valid = valid;
        v.hist  = hist != 0;
        const AX_ENGINE_IO_INFO_T * io = v.k->info;
        auto small = [&](axera_cmm & b, uint32_t n) {
            b = axera_cmm_alloc(n);
            memset(b->vir, 0, n);
            return axera_io(b.get(), b->vir, n);
        };
        for (uint32_t i = 0; i < io->nInputSize; ++i) {
            const AX_ENGINE_IOMETA_T & mt = io->pInputs[i];
            const std::string          nm = mt.pName;
            auto                       it = s->banks.find(nm);
            if (it != s->banks.end()) {
                GGML_ASSERT(it->second->size == mt.nSize && "axera sgl: bank size differs from the kernel input");
                v.in.push_back(axera_io(it->second.get(), it->second->vir, mt.nSize));
            } else if (nm == "position_ids") {
                v.in.push_back(small(v.POS, mt.nSize));
            } else if (nm == "indptr") {
                v.in.push_back(small(v.INDPTR, mt.nSize));
            } else if (nm == "indices") {
                v.in.push_back(small(v.INDICES, mt.nSize));
                for (uint32_t j = 0; j < mt.nSize / 4; ++j) {
                    ((int32_t *) v.INDICES->vir)[j] = (int32_t) j;
                }
                v.INDICES->flush(v.INDICES->vir, mt.nSize);
                s->pages = (int) (mt.nSize / 4);
            } else if (nm == "last_page_len") {
                v.in.push_back(small(v.LPL, mt.nSize));
            } else if (nm == "cache_location") {
                v.in.push_back(small(v.CLOC, mt.nSize));
            } else {
                GGML_ASSERT(!v.X && mt.nSize == (uint32_t) rows * H * 2 && "axera sgl: unknown kernel input");
                v.in.push_back(small(v.X, mt.nSize));
            }
        }
        for (uint32_t i = 0; i < io->nOutputSize; ++i) {
            const AX_ENGINE_IOMETA_T & mt = io->pOutputs[i];
            auto                       it = s->banks.find(mt.pName);
            if (it != s->banks.end()) {
                v.out.push_back(axera_io(it->second.get(), it->second->vir, mt.nSize));
            } else {
                GGML_ASSERT(!v.OUT && mt.nSize == (uint32_t) rows * H * 2 && "axera sgl: unknown kernel output");
                v.out.push_back(small(v.OUT, mt.nSize));
            }
        }
        GGML_ASSERT(v.X && v.OUT && v.POS && v.INDPTR && v.LPL && v.CLOC);
        s->vars.push_back(std::move(v));
    }
    GGML_LOG_WARN("axera sgl: %zu banks (%.2f GiB) from %s, %zu kernels, %d pages\n", s->banks.size(),
                  total / 1073741824.0, dir.c_str(), s->vars.size(), s->pages);
    return s;
}

static void axera_sgl_restart(axera_sgl & s) {
    for (const auto & n : s.state) {
        auto & b = s.banks.at(n);
        if (n.find("state") != std::string::npos) {  // the KV pages need no clearing: read up to the valid length
            memset(b->vir, 0, b->size);
            b->flush(b->vir, b->size);
        }
    }
    s.next_pos = 0;
}

// the layers of T tokens at positions pos (consecutive, a single sequence): layer 0 input rows x0(r) -> h [T, H].
// Returns the tokens run (-1: failed); prefill_only: the chunk kernels only, stopping at a rest shorter than them all
int axera_sgl_layers(std::shared_ptr<axera_sgl> & sp, const int32_t * pos, int T, int H,
                     const std::function<const float *(int)> & x0, std::vector<float> & h, bool prefill_only) {
    if (!sp) {
        sp = axera_sgl_load(H);
    }
    axera_sgl & s = *sp;
    GGML_ASSERT(s.H == H);
    for (int it = 0; it < T;) {
        const int64_t p = pos[it];
        if (p == 0) {
            axera_sgl_restart(s);
        }
        if (p != s.next_pos || p >= (int64_t) s.pages * 256) {
            GGML_LOG_ERROR("axera sgl: position %" PRId64 " (expected %" PRId64 ", context %d)\n", p, s.next_pos,
                           s.pages * 256);
            s.next_pos = -1;
            return -1;
        }
        // the largest chunk kernel that fits from a page start, with history after the first page (a kernel built with
        // history fails without: inf)
        axera_sgl::variant * v = nullptr;
        for (auto & c : s.vars) {
            const bool hist_ok = c.hist == (p > 0);
            if (p % 256 == 0 && c.rows > 1 && hist_ok && c.valid <= T - it && p + c.valid <= (int64_t) s.pages * 256 &&
                (!v || c.valid > v->valid)) {
                v = &c;
            }
        }
        if (!v && prefill_only) {
            return it;
        }
        if (!v) {
            for (auto & c : s.vars) {
                v = c.rows == 1 ? &c : v;
            }
        }
        if (!v) {
            GGML_LOG_ERROR("axera sgl: no kernel for position %" PRId64 " (no decode kernel sgl27_r1_v1_h1)\n", p);
            return -1;
        }
        const int n = v->rows == 1 ? 1 : v->valid;
        axera_timer tm(v->rows == 1 ? "sgl decode" : "sgl prefill chunk");
        auto * X = (ggml_bf16_t *) v->X->vir;
        axera_host_pool().run(n, [&](int64_t r) { ggml_fp32_to_bf16_row(x0(it + (int) r), X + (size_t) r * H, H); });
        v->X->flush(v->X->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        auto * ps = (int32_t *) v->POS->vir;
        for (int j = 0; j < n; ++j) {
            ps[j] = (int32_t) (p + j);
        }
        // the history before p (sgl_kernel_axera batch_metadata.prepare_attention_metadata): its pages, the cells of
        // its last page and the page table (identity, unused entries 0); the new rows go to page p / 256, cell p % 256
        const int32_t hist = (int32_t) ((p + 255) / 256);
        ((int32_t *) v->INDPTR->vir)[0] = 0;
        ((int32_t *) v->INDPTR->vir)[1] = hist;
        ((int32_t *) v->LPL->vir)[0]    = p > 0 ? (int32_t) (p - (int64_t) (hist - 1) * 256) : 0;
        for (int j = 0; j < s.pages; ++j) {
            ((int32_t *) v->INDICES->vir)[j] = j < hist ? j : 0;
        }
        ((int32_t *) v->CLOC->vir)[0] = (int32_t) (p / 256);
        ((int32_t *) v->CLOC->vir)[1] = (int32_t) (p % 256);
        for (auto * b : { v->POS.get(), v->INDPTR.get(), v->LPL.get(), v->CLOC.get(), v->INDICES.get() }) {
            b->flush(b->vir, b->size);
        }
        const int ret = axera_run_sync(v->k, v->in, v->out);
        if (ret != 0) {
            GGML_ABORT("axera sgl: RunSync 0x%x (rows %d valid %d, position %" PRId64 ")\n", ret, v->rows, v->valid, p);
        }
        v->OUT->invalidate(v->OUT->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        const auto * O = (const ggml_bf16_t *) v->OUT->vir;
        axera_host_pool().run(n, [&](int64_t r) {
            ggml_bf16_to_fp32_row(O + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
        });
        s.next_pos = p + n;
        it += n;
    }
    return T;
}

// after the whole-model kernels ran positions 0 .. n-1: their recurrent state and KV cache into the template's
// (the llama.cpp recurrent cache rows and KV cache cells), so that its own kernels continue at position n.
// GDN value heads: the kernels' HF order (key head kh, its value head vo: kh * 3 + vo), the template's GGUF order
// (vo * 16 + kh); the conv state q | k | v in one row per step. KV: pages [page][K, V][256 cells][heads x dim] bf16 into
// [cell][heads x dim] f16 at the cells of the batch's tokens.
void axera_sgl_to_q35(axera_sgl & s, axera_q35 & m, int64_t n, const std::function<int64_t(int64_t)> & cell_of) {
    axera_timer tm("sgl state -> template");
    auto bank = [&](const char * name) -> const char * {
        auto & b = s.banks.at(name);
        b->invalidate(b->vir, b->size);
        return (const char *) b->vir;
    };
    const char * st = bank("b0_attn_state_in");
    const char * qc = bank("b0_attn_q_conv_state_in");
    const char * kc = bank("b0_attn_k_conv_state_in");
    const char * vc = bank("b0_attn_v_conv_state_in");
    const char * kv = bank("b3_kv");
    const int    NK = 16, NV = 48, DK = 128, DV = 128, W = 3, KVW = 1024;
    const size_t head = (size_t) DK * DV * sizeof(float);
    int64_t      lo = cell_of(0), hi = lo;  // the cells written (a single sequence's: consecutive)
    for (int64_t t = 1; t < n; ++t) {
        lo = std::min(lo, cell_of(t));
        hi = std::max(hi, cell_of(t));
    }
    // one job per GDN layer (state and conv rows) and per attention layer's K or V cache, in parallel
    std::vector<std::function<void()>> jobs;
    int                                g = 0, a = 0;
    for (auto & ly : m.layers) {
        if (ly.gdn) {
            GGML_ASSERT(ly.in_place && "axera sgl: the GDN state in the llama.cpp recurrent cache");
            for (auto & rg : ly.cache_state) {
                jobs.push_back([&, g, rg, row = ly.row] {
                    char * dst = rg.base + row * rg.stride;
                    if (!rg.conv) {
                        GGML_ASSERT(rg.size == NV * head);
                        for (int ha = 0; ha < NV; ++ha) {
                            const int hb = (ha % NK) * (NV / NK) + ha / NK;
                            memcpy(dst + ha * head, st + ((size_t) g * NV + hb) * head, head);
                        }
                    } else {
                        const int C = 2 * NK * DK + NV * DV;
                        GGML_ASSERT(rg.size == (size_t) W * C * sizeof(ggml_bf16_t));
                        auto * d = (ggml_bf16_t *) dst;
                        for (int r = 0; r < W; ++r) {
                            const size_t i = (size_t) g * W + r;
                            memcpy(d + r * C, qc + i * NK * DK * 2, NK * DK * 2);
                            memcpy(d + r * C + NK * DK, kc + i * NK * DK * 2, NK * DK * 2);
                            for (int ha = 0; ha < NV; ++ha) {
                                const int hb = (ha % NK) * (NV / NK) + ha / NK;
                                memcpy(d + r * C + 2 * NK * DK + ha * DV, vc + (i * NV * DV + hb * DV) * 2, DV * 2);
                            }
                        }
                    }
                    rg.b->flush(dst, rg.size);
                });
            }
            ++g;
            continue;
        }
        for (auto & kr : ly.kv) {
            GGML_ASSERT(kr.row == KVW * sizeof(ggml_fp16_t));
            jobs.push_back([&, a, kr, stream = ly.stream] {
                char *       base = kr.base + stream * kr.stride;
                const auto * src  = (const ggml_bf16_t *) kv + (size_t) a * s.pages_kv * 2 * 256 * KVW;
                for (int64_t t = 0; t < n; ++t) {
                    const ggml_bf16_t * row = src + (((t / 256) * 2 + (kr.v ? 1 : 0)) * 256 + t % 256) * (size_t) KVW;
                    auto *              out = (ggml_fp16_t *) (base + cell_of(t) * kr.row);
                    for (int i = 0; i < KVW; ++i) {
                        out[i] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(row[i]));
                    }
                }
                kr.b->flush(base + lo * kr.row, (size_t) (hi - lo + 1) * kr.row);
            });
        }
        ++a;
    }
    GGML_ASSERT(g == 48 && a == 16);
    axera_host_pool().run((int64_t) jobs.size(), [&](int64_t j) { jobs[j](); });
}
