// Axera NPU backend: the whole text model in the fused kernels of sglang_axera_opt (sgl-kernel-axera models/qwen38_27b
// prefill for Qwen3.8-27B, models/qwen36_35b_a3b decode for Qwen3.5-35B-A3B), GGML_AXERA_SGL=<dir> with the qwen35
// template
//
// One kernel call runs all layers (a runtime loop over blocks of three GDN and one gated attention layer, each with its
// MLP / MoE) on 256 rows (prefill) or 1 row (decode). Weights, recurrent state and the paged KV cache (pages of 256
// cells) are the kernels' inputs; <dir> holds the weight banks as the offline packers wrote them (pack_sgl27.py /
// pack_sgl35.py: <name>.bin of every bank, manifest.txt "<name> <bytes> <state>"), the kernels sgl<model>_r<rows>_v<valid>
// _h<history>_npu4.axmodel are in GGML_AXERA_KERNEL_DIR (loaded with the others at initialization). The state banks (GDN
// state pools, KV pages) are allocated here and cleared at position 0.
// The template (ggml-axera-q35-run.cpp) runs a prompt's whole 256-position chunks through the chunk kernels (27B), and
// single tokens through the decode kernel (35B); the state moves between the kernels' banks and the template's caches
// (axera_sgl_sync) whenever the other side runs next. The 27B decode kernel and the chunk kernels of fewer valid rows
// are not used (the former is slower than the template's, the latter compute NaN). (sgl-kernel-axera flash_attn.py
// project_token's NVFP4 path drops the banks offset, every block then reading block 0's attention Q/K/V weights: the
// decode kernels are built with it fixed, `if not banks and ...`.)

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
        axera_cmm                          X, OUT, POS, INDPTR, INDICES, LPL, CLOC, ATT;  // ATT: decode workspace
    };
    std::vector<variant>             vars;
    std::map<std::string, axera_cmm> banks;
    std::map<std::string, size_t>    bytes;  // a bank's size (its CMM rounded up)
    std::vector<std::string>         state;  // banks cleared at position 0
    int                              H = 0, pages = 0;  // pages: the page table's entries (the context / 256)
    int64_t                          pages_kv = 0;      // pages of the KV bank
    int64_t                          next_pos = -1;
    // which side holds the current recurrent state (the kernels' banks or the template's caches) and the positions
    // whose KV each side holds (0 .. *_upto - 1)
    bool                             cur = false;
    int64_t                          sgl_upto = 0, tpl_upto = 0;
    int                              NK = 0, NV = 0, KVW = 0;  // set at the first state move, from the template's caches
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
        s->bytes[n_name] = n_bytes;
    }
    if (s->banks.empty()) {
        GGML_ABORT("axera sgl: no banks in %s/manifest.txt\n", dir.c_str());
    }
    for (const auto & f : axera_kernels().files) {
        int rows, valid, hist, len = 0;
        int model = 0;
        if (sscanf(f.first.c_str(), "sgl%d_r%d_v%d_h%d%n", &model, &rows, &valid, &hist, &len) != 4 || (size_t) len != f.first.size()) {
            continue;
        }
        if (rows > 1 && valid != rows) {  // NaN (sglang runs a partial chunk through its per-layer kernels)
            GGML_LOG_WARN("axera sgl: %s skipped: a chunk of fewer valid rows computes NaN\n", f.first.c_str());
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
                GGML_ASSERT(s->bytes.at(nm) == mt.nSize && "axera sgl: bank size differs from the kernel input");
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
            } else if (nm == "attention") {
                v.in.push_back(small(v.ATT, mt.nSize));
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
    s.cur      = false;
    s.sgl_upto = s.tpl_upto = 0;
}

// the page metadata of n rows at position p (sgl_kernel_axera batch_metadata.prepare_attention_metadata): the history
// before p (its pages, the cells of its last page, the page table: identity, unused entries 0), the new rows' page
// p / 256 and cell p % 256
static void axera_sgl_meta(const axera_sgl & s, axera_sgl::variant & v, int64_t p, int n) {
    auto * ps = (int32_t *) v.POS->vir;
    for (int j = 0; j < n; ++j) {
        ps[j] = (int32_t) (p + j);
    }
    const int32_t hist = (int32_t) ((p + 255) / 256);
    ((int32_t *) v.INDPTR->vir)[0] = 0;
    ((int32_t *) v.INDPTR->vir)[1] = hist;
    ((int32_t *) v.LPL->vir)[0]    = p > 0 ? (int32_t) (p - (int64_t) (hist - 1) * 256) : 0;
    for (int j = 0; j < s.pages; ++j) {
        ((int32_t *) v.INDICES->vir)[j] = j < hist ? j : 0;
    }
    ((int32_t *) v.CLOC->vir)[0] = (int32_t) (p / 256);
    ((int32_t *) v.CLOC->vir)[1] = (int32_t) (p % 256);
    for (auto * b : { v.POS.get(), v.INDPTR.get(), v.LPL.get(), v.CLOC.get(), v.INDICES.get() }) {
        b->flush(b->vir, b->size);
    }
}

static void axera_sgl_call(axera_sgl::variant & v, int64_t p) {
    const int ret = axera_run_sync(v.k, v.in, v.out);
    if (ret != 0) {
        GGML_ABORT("axera sgl: RunSync 0x%x (rows %d valid %d, position %" PRId64 ")\n", ret, v.rows, v.valid, p);
    }
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
    static const bool unaligned = [] {
        const char * e = std::getenv("GGML_AXERA_SGL_UNALIGNED");
        return e == nullptr || strcmp(e, "0") != 0;
    }();
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
            // also from inside a page, within it (the last chunk after a shorter one; GGML_AXERA_SGL_UNALIGNED=0: off)
            const bool start_ok = p % 256 == 0 || (unaligned && p % 256 + c.valid <= 256);
            if (start_ok && c.rows > 1 && hist_ok && c.valid <= T - it && p + c.valid <= (int64_t) s.pages * 256 &&
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
            GGML_LOG_ERROR("axera sgl: no kernel for position %" PRId64 "\n", p);
            return -1;
        }
        const int n = v->rows == 1 ? 1 : v->valid;
        axera_timer tm(v->rows == 1 ? "sgl decode" : "sgl prefill chunk");
        auto * X = (ggml_bf16_t *) v->X->vir;
        axera_host_pool().run(n, [&](int64_t r) { ggml_fp32_to_bf16_row(x0(it + (int) r), X + (size_t) r * H, H); });
        v->X->flush(v->X->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        axera_sgl_meta(s, *v, p, n);
        axera_sgl_call(*v, p);
        v->OUT->invalidate(v->OUT->vir, (size_t) n * H * sizeof(ggml_bf16_t));
        const auto * O = (const ggml_bf16_t *) v->OUT->vir;
        axera_host_pool().run(n, [&](int64_t r) {
            ggml_bf16_to_fp32_row(O + (size_t) r * H, h.data() + (size_t) (it + r) * H, H);
        });
        s.next_pos = p + n;
        s.sgl_upto = s.next_pos;
        s.cur      = true;
        it += n;
    }
    return T;
}

int64_t axera_sgl_next(const axera_sgl * s) {
    return s ? s->next_pos : -1;
}

bool axera_sgl_has_decode(const axera_sgl * s) {
    return s && std::any_of(s->vars.begin(), s->vars.end(), [](const axera_sgl::variant & v) { return v.rows == 1; });
}

// the recurrent state and the KV of positions p0 .. p0 + n - 1 (cells cell_of(i) of the template's cache, i = 0 .. n - 1)
// from the kernels' banks into the template's caches (to_tpl) or back. GDN value heads: the kernels' HF order (key head
// kh, its value heads vo: kh * R + vo, R = NV / NK), the template's GGUF order (vo * NK + kh); its conv state q | k | v
// in one row per step, the kernels' three pools. KV: pages [page][K, V][256 cells][heads x dim] bf16 against
// [cell][heads x dim] f16.
static void axera_sgl_move(axera_sgl & s, axera_q35 & m, int64_t p0, int64_t n, const std::function<int64_t(int64_t)> & cell_of,
                           bool to_tpl) {
    axera_timer tm(to_tpl ? "sgl state -> template" : "sgl state <- template");
    const int DK = 128, DV = 128, W = 3;
    int       nG = 0, nA = 0;
    for (const auto & ly : m.layers) {
        (ly.gdn ? nG : nA)++;
    }
    auto bank = [&](const char * name) -> char * {
        auto & b = s.banks.at(name);
        if (to_tpl) {
            b->invalidate(b->vir, b->size);
        }
        return (char *) b->vir;
    };
    char * st = bank("b0_attn_state_in");
    char * qc = bank("b0_attn_q_conv_state_in");
    char * kc = bank("b0_attn_k_conv_state_in");
    char * vc = bank("b0_attn_v_conv_state_in");
    char * kv = bank("b3_kv");
    if (!s.NV) {
        s.NV       = (int) (s.bytes.at("b0_attn_state_in") / ((size_t) nG * DK * DV * sizeof(float)));
        s.NK       = (int) (s.bytes.at("b0_attn_q_conv_state_in") / ((size_t) nG * W * DK * sizeof(ggml_bf16_t)));
        for (const auto & ly : m.layers) {
            if (!ly.gdn && !s.KVW) {
                s.KVW = (int) (ly.kv.at(0).row / sizeof(ggml_fp16_t));
            }
        }
        s.pages_kv = (int64_t) (s.bytes.at("b3_kv") / ((size_t) nA * 2 * 256 * s.KVW * sizeof(ggml_bf16_t)));
        GGML_LOG_INFO("axera sgl: %d GDN layers (%d key / %d value heads), %d attention layers (KV row %d), %" PRId64
                      " KV pages\n", nG, s.NK, s.NV, nA, s.KVW, s.pages_kv);
    }
    const int    NK = s.NK, NV = s.NV, R = NV / NK, KVW = s.KVW;
    const size_t head = (size_t) DK * DV * sizeof(float);
    int64_t      lo = n ? cell_of(0) : 0, hi = lo;  // the template's cells (a single sequence's: consecutive)
    for (int64_t t = 1; t < n; ++t) {
        lo = std::min(lo, cell_of(t));
        hi = std::max(hi, cell_of(t));
    }
    // one job per GDN layer's state / conv rows and per attention layer's K or V cache, in parallel
    std::vector<std::function<void()>> jobs;
    int                                g = 0, a = 0;
    for (auto & ly : m.layers) {
        if (ly.gdn) {
            GGML_ASSERT(ly.in_place && "axera sgl: the GDN state in the llama.cpp recurrent cache");
            for (auto & rg : ly.cache_state) {
                jobs.push_back([&, g, rg, row = ly.row] {
                    char * tp = rg.base + row * rg.stride;
                    if (!to_tpl) {
                        rg.b->invalidate(tp, rg.size);
                    }
                    if (!rg.conv) {
                        GGML_ASSERT(rg.size == NV * head);
                        for (int ha = 0; ha < NV; ++ha) {
                            char * sp = st + ((size_t) g * NV + (ha % NK) * R + ha / NK) * head;
                            memcpy(to_tpl ? tp + ha * head : sp, to_tpl ? sp : tp + ha * head, head);
                        }
                    } else {
                        const int C = 2 * NK * DK + NV * DV;
                        GGML_ASSERT(rg.size == (size_t) W * C * sizeof(ggml_bf16_t));
                        auto * d  = (ggml_bf16_t *) tp;
                        auto   cp = [&](ggml_bf16_t * t, char * k, size_t bytes) {
                            memcpy(to_tpl ? (void *) t : (void *) k, to_tpl ? (void *) k : (void *) t, bytes);
                        };
                        for (int r = 0; r < W; ++r) {
                            const size_t i = (size_t) g * W + r;
                            cp(d + r * C, qc + i * NK * DK * 2, NK * DK * 2);
                            cp(d + r * C + NK * DK, kc + i * NK * DK * 2, NK * DK * 2);
                            for (int ha = 0; ha < NV; ++ha) {
                                const int hb = (ha % NK) * R + ha / NK;
                                cp(d + r * C + 2 * NK * DK + ha * DV, vc + (i * NV * DV + hb * DV) * 2, DV * 2);
                            }
                        }
                    }
                    if (to_tpl) {
                        rg.b->flush(tp, rg.size);
                    }
                });
            }
            ++g;
            continue;
        }
        for (auto & kr : ly.kv) {
            GGML_ASSERT(kr.row == KVW * sizeof(ggml_fp16_t));
            jobs.push_back([&, a, kr, stream = ly.stream] {
                char * base = kr.base + stream * kr.stride;
                auto * pg   = (ggml_bf16_t *) kv + (size_t) a * s.pages_kv * 2 * 256 * KVW;
                if (!to_tpl && n) {
                    kr.b->invalidate(base + lo * kr.row, (size_t) (hi - lo + 1) * kr.row);
                }
                for (int64_t t = 0; t < n; ++t) {
                    const int64_t q    = p0 + t;
                    ggml_bf16_t * row  = pg + (((q / 256) * 2 + (kr.v ? 1 : 0)) * 256 + q % 256) * (size_t) KVW;
                    auto *        cell = (ggml_fp16_t *) (base + cell_of(t) * kr.row);
                    for (int i = 0; i < KVW; ++i) {
                        if (to_tpl) {
                            cell[i] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(row[i]));
                        } else {
                            row[i] = ggml_fp32_to_bf16(ggml_fp16_to_fp32(cell[i]));
                        }
                    }
                }
                if (to_tpl && n) {
                    kr.b->flush(base + lo * kr.row, (size_t) (hi - lo + 1) * kr.row);
                }
            });
        }
        ++a;
    }
    axera_host_pool().run((int64_t) jobs.size(), [&](int64_t j) { jobs[j](); });
    if (!to_tpl) {
        for (const char * name : { "b0_attn_state_in", "b0_attn_q_conv_state_in", "b0_attn_k_conv_state_in",
                                   "b0_attn_v_conv_state_in", "b3_kv" }) {
            auto & b = s.banks.at(name);
            b->flush(b->vir, b->size);
        }
    }
}

// a prompt's chunks just ran here (positions p0 .. p0 + n - 1, batch tokens 0 .. n - 1 at cells cell_of): the state and
// their KV into the template's caches, which run next
void axera_sgl_to_q35(axera_sgl & s, axera_q35 & m, int64_t p0, int64_t n, const std::function<int64_t(int64_t)> & cell_of) {
    axera_sgl_move(s, m, p0, n, cell_of, true);
    s.cur      = false;
    s.tpl_upto = p0 + n;
}

// before a batch at position p runs on the template's kernels (sgl = false) or here (sgl = true): the state and the KV
// the side lacks moved to it (a single sequence at the cells of its positions: llama.cpp's for one sequence from 0)
void axera_sgl_sync(axera_sgl & s, axera_q35 & m, int64_t p, bool sgl) {
    auto ident = [](int64_t base) { return [base](int64_t t) { return base + t; }; };
    if (sgl && !s.cur) {
        axera_sgl_move(s, m, s.sgl_upto, p - s.sgl_upto, ident(s.sgl_upto), false);
        s.sgl_upto = p;
        s.next_pos = p;
        s.cur      = true;
    } else if (!sgl && s.cur) {
        axera_sgl_move(s, m, s.tpl_upto, p - s.tpl_upto, ident(s.tpl_upto), true);
        s.tpl_upto = p;
        s.cur      = false;
    }
}

// the template's kernels ran positions up to p - 1
void axera_sgl_template_ran(axera_sgl * s, int64_t p) {
    if (s) {
        s->tpl_upto = p;
        s->cur      = false;
    }
}

// one token at position p through the decode kernel: layer 0 input row x -> out [H]
bool axera_sgl_decode(axera_sgl & s, axera_q35 & m, int64_t p, const float * x, float * out) {
    if (p == 0) {
        axera_sgl_restart(s);
    }
    axera_sgl_sync(s, m, p, true);
    axera_sgl::variant * v = nullptr;
    for (auto & c : s.vars) {
        v = c.rows == 1 ? &c : v;
    }
    if (!v || p >= (int64_t) s.pages * 256 || p >= s.pages_kv * 256) {
        GGML_LOG_ERROR("axera sgl: no decode at position %" PRId64 " (context %d)\n", p, s.pages * 256);
        return false;
    }
    axera_timer tm("sgl decode");
    ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) v->X->vir, s.H);
    v->X->flush(v->X->vir, (size_t) s.H * sizeof(ggml_bf16_t));
    axera_sgl_meta(s, *v, p, 1);
    axera_sgl_call(*v, p);
    v->OUT->invalidate(v->OUT->vir, (size_t) s.H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) v->OUT->vir, out, s.H);
    s.next_pos = s.sgl_upto = p + 1;
    s.cur                   = true;
    return true;
}
