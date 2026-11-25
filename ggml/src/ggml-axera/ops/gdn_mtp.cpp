#include "gdn_mtp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace axera {
namespace {
constexpr int PG = 256;  // cells per page
}  // namespace

void gdn_mtp::init(const gdn_mtp_env & e, const std::map<std::string, ggml_tensor *> & t, int il) {
    const manifest & man = *e.man;
    params &         prm = *e.prm;
    const int        H = e.H, ROT = e.ROT, NBLK = e.NBLK;
    axera_timer    tb("init: mtp");
    const kernel & k1 = man.get("mtp.t1");
    const kernel & kp = man.get("mtp.prompt");
    GGML_ASSERT(k1.desc->packs == kp.desc->packs && k1.out_bytes("kv") == kp.out_bytes("kv"));
    // paged like the layers' caches (the kernel touches only the runtime pages), or the kernel's whole IO
    MKV = axera_cmm_alloc(e.paged ? k1.out_bytes("kv") / NBLK * e.kv_pages : k1.out_bytes("kv"));
    page_b = k1.out_bytes("kv") / NBLK;
    kv_row = (size_t) k1.out_bytes("kv") / NBLK / (2 * PG);   // one K (or V) row: HKV * D bf16
    auto bind = [&](npu_call & c, const kernel & k, std::initializer_list<std::pair<const char *, axera_cmm *>> io) {
        c   = {};
        c.k = &k;
        auto of = [&](const std::string & n) -> axera_cmm * {
            for (const auto & [name, b] : io) {
                if (n == name) {
                    return b;
                }
            }
            return nullptr;
        };
        for (const auto & d : k.desc->in) {
            if (k.desc->packs.count(d.name)) {
                c.in.push_back(axera_io(prm.pack(t, *k.desc, d.name, il), d.bytes));
                continue;
            }
            axera_cmm * b = of(d.name);
            if (!b) {
                GGML_ABORT("axera %s: %s: unexpected input %s\n", e.who, k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(d.bytes);
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : k.desc->out) {
            axera_cmm * b = d.name == std::string("kv") ? &MKV : of(d.name);
            if (!b) {
                GGML_ABORT("axera %s: %s: unexpected output %s\n", e.who, k.desc->role.c_str(), d.name.c_str());
            }
            if (!*b) {
                *b = axera_cmm_alloc(d.bytes);
            }
            c.out.push_back(d.name == std::string("kv") ? axera_io(b->get(), (*b)->vir, (uint32_t) d.bytes) : axera_io(*b, d.bytes));
        }
    };
    bind(mtp1, k1, { { "e", &ME }, { "h", &MH }, { "cs", &MCS }, { "pn", &MPN }, { "mask", &MMASK }, { "hnext", &MOUT },
                     { "hraw", &MRAW } });
    bind(mtpp, kp, { { "e", &PE }, { "h", &PH }, { "cs", &PCS }, { "pn", &PPN }, { "mask", &PMASK }, { "nblk", &PNBLK },
                     { "hnext", &POUT2 }, { "hraw", &PRAW } });
    GGML_ASSERT(MCS->size == (size_t) ROT * sizeof(ggml_bf16_t) && MMASK->size == (size_t) man.context * sizeof(ggml_bf16_t) &&
                PE->size % ((size_t) H * sizeof(ggml_bf16_t)) == 0);
    std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live = { &mtp1.in, &mtpp.in };
    if (man.has("mtp.promptkv")) {  // the K/V-only prompt pass (the last row uses mtp1's full block)
        const kernel & kpk = man.get("mtp.promptkv");
        // its compact pack has only the eh_proj / K / V weights; its K/V rows are copied into MKV by the host
        bind(mtppkv, kpk, { { "e", &PKE }, { "h", &PKH }, { "cs", &PKCS }, { "kout", &PK }, { "vout", &PV } });
        const io_desc * ed = nullptr;
        for (const auto & d : kpk.desc->in) {
            if (d.name == "e") {
                ed = &d;
            }
        }
        GGML_ASSERT(ed && ed->shape.size() >= 2);
        const size_t P = (size_t) ed->shape[1];
        h_f32 = PKH->size == P * H * sizeof(float);
        e_f32 = PKE->size == P * H * sizeof(float);
        live.push_back(&mtppkv.in);
    }
    if (man.has("mtp.m8")) {
        const kernel & k8 = man.get("mtp.m8");
        GGML_ASSERT(k8.desc->packs == k1.desc->packs && k8.out_bytes("kv") == k1.out_bytes("kv"));
        bind(mtp8, k8, { { "e", &M8E }, { "h", &M8H }, { "cs", &M8CS }, { "pn", &M8PN }, { "mask", &M8MASK }, { "ntok", &M8NT },
                         { "hnext", &M8OUT }, { "hraw", &M8RAW } });
        GGML_ASSERT(M8MASK->size == M8E->size / ((size_t) H * sizeof(ggml_bf16_t)) * MMASK->size);
        live.push_back(&mtp8.in);
    }
    for (const auto * l : e.live) {
        live.push_back(l);
    }
    prm.release(live);
}

ggml_status gdn_mtp::compute(const gdn_mtp_env & e, ggml_cgraph * g) {
    wait();
    const manifest & man   = *e.man;
    graph_names &    names = *e.names;
    output_head &    head  = *e.head;
    axera_cmm &      COS   = *e.COS;
    axera_cmm &      SIN   = *e.SIN;
    const int        H = e.H, ROT = e.ROT;
    const int64_t    kv_cells = e.kv_cells;
    const auto &  t     = names.cur.t;
    ggml_tensor * hin   = names.find("mtp_h_input");
    ggml_tensor * hn    = names.find("h_nextn");
    ggml_tensor * pos   = names.find("inp_pos");
    ggml_tensor * kidx  = names.find("attn_inp_k_idxs");
    ggml_tensor * kqm   = names.find("attn_inp_kq_mask");
    const ggml_tensor * embd = nullptr;
    int           il    = -1;
    for (int i = 0; i < g->n_nodes && !embd; i++) {  // the token rows: the input of the embedding norm mtp_enorm-<il>
        const char * n = g->nodes[i]->name;
        if (strncmp(n, "mtp_enorm-", 10) == 0) {
            il                    = atoi(n + 10);
            const ggml_tensor * m = g->nodes[i];
            const ggml_tensor * r = m->src[0]->op == GGML_OP_RMS_NORM ? m->src[0] : m->src[1];
            embd                  = r->op == GGML_OP_RMS_NORM ? r->src[0] : nullptr;
            if (embd && embd->op != GGML_OP_NONE) {  // (computed in this graph: the nodes before the norm on the CPU)
                int n0 = 0;
                while (n0 < g->n_nodes && g->nodes[n0] != r) {
                    ++n0;
                }
                cpu_compute(g, g->nodes, n0, "mtp input (cpu)");
            }
        }
    }
    if (!embd || !hn || !pos || !kidx || !kqm || il < 0 || !e.target_bound) {
        GGML_LOG_ERROR("axera %s: an MTP graph before the target model's or of an unexpected shape (embd %d h_nextn %d "
                       "inp_pos %d k_idxs %d kq_mask %d layer %d target %d)\n", e.who, !!embd, !!hn, !!pos, !!kidx, !!kqm, il, (int) e.target_bound);
        return GGML_STATUS_FAILED;
    }
    const int T = (int) hin->ne[1];
    GGML_ASSERT(hin->type == GGML_TYPE_F32 && hin->ne[0] == H && embd->type == GGML_TYPE_F32 && embd->ne[0] == H &&
                embd->ne[1] == T && hn->ne[1] == T);
    const ggml_tensor * w0 = t.at("blk." + std::to_string(il) + ".nextn.eh_proj.weight");
    if (bound_w != w0->data) {
        init(e, t, il);
        bound_w = w0->data;
    }
    const ggml_tensor * rope_node = llm_rope_node(g);
    const rope    rp(rope_node, pos, T);
    const kv_view kv(kidx, kqm, man.context, PG);
    cell = 0;
    for (int r = 0; r < T; ++r) {
        cell = std::max(cell, kv.cell_of(r) + 1);
    }
    auto row_bf16 = [&](const ggml_tensor * x, int r, ggml_bf16_t * d) {
        ggml_fp32_to_bf16_row((const float *) ((const char *) x->data + (size_t) r * x->nb[1]), d, H);
    };
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
    std::vector<float> hr((size_t) T * H);  // the pre-head-norm rows, for the combined head
    const size_t       half = (ROT / 2) * sizeof(ggml_bf16_t);
    axera_timer        tm(T == 1 ? "mtp (draft)" : T <= 8 ? "mtp (catch-up rows)" : "mtp (prompt)");
    const int R8 = mtp8.k ? (int) (M8E->size / (H * sizeof(ggml_bf16_t))) : 0;
    if (T >= 2 && T <= R8) {  // a verification's accepted rows in one call (each row sees the rows before it)
        const size_t mrow = MMASK->size;
        auto *       pn   = (int32_t *) M8PN->vir;
        for (int r = 0; r < R8; ++r) {
            const int rr = std::min(r, T - 1);  // rows from T on: the last row's (the kernel leaves their KV alone)
            if (r < T) {
                row_bf16(embd, r, (ggml_bf16_t *) M8E->vir + (size_t) r * H);
                row_bf16(hin, r, (ggml_bf16_t *) M8H->vir + (size_t) r * H);
            }
            rp.rows(rr, 1, COS.get(), SIN.get());
            memcpy((char *) M8CS->vir + r * 2 * half, COS->vir, half);
            memcpy((char *) M8CS->vir + r * 2 * half + half, SIN->vir, half);
            auto * mk = (ggml_bf16_t *) ((char *) M8MASK->vir + r * mrow);
            std::fill(mk, mk + man.context, ggml_fp32_to_bf16(-3e38f));
            const int64_t c    = kv.cell_of(rr);
            const int     nblk = kv.mask_row(rr, mk);
            pn[3 * r]          = (int32_t) c;
            pn[3 * r + 1]      = std::max(nblk, (int) (c / PG + 1));
            pn[3 * r + 2]      = (int32_t) ((c / PG) * 2 * PG + c % PG);
        }
        for (int r = 0; r < 3 * R8; ++r) {  // (an index out of the kernel's range hangs the NPU: stop here instead)
            GGML_ASSERT(pn[r] >= 0 && pn[r] < 2 * man.context);
        }
        *(int32_t *) M8NT->vir = T;
        for (const axera_cmm * b : { &M8E, &M8H, &M8CS, &M8PN, &M8MASK, &M8NT }) {
            (*b)->flush();
        }
        // (GGML_AXERA_MTP_SYNC=1: in line) asynchronous when this graph has no output rows: neither its logits nor
        // its h_nextn rows are read
        static const bool  sync = getenv("GGML_AXERA_MTP_SYNC") && atoi(getenv("GGML_AXERA_MTP_SYNC")) != 0;
        const ggml_tensor * lg   = names.find("result_output");
        if (e.async && !sync && !(lg && lg->ne[1] > 0 && (lg->flags & GGML_TENSOR_FLAG_COMPUTE))) {
            tm.stop();
            bg = std::thread([this] {
                axera_timer tb("mtp (catch-up rows, async)");
                mtp8.k->run(mtp8.in, mtp8.out, "mtp m8");
            });
            return GGML_STATUS_SUCCESS;
        }
        mtp8.k->run(mtp8.in, mtp8.out, "mtp m8");
        M8OUT->invalidate();
        M8RAW->invalidate();
        for (int r = 0; r < T; ++r) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) M8OUT->vir + (size_t) r * H, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) M8RAW->vir + (size_t) r * H, hr.data() + (size_t) r * H, H);
        }
    } else if (T <= 8) {  // a draft step, or a verification's accepted rows: row by row
        for (int r = 0; r < T; ++r) {
            row_bf16(embd, r, (ggml_bf16_t *) ME->vir);
            row_bf16(hin, r, (ggml_bf16_t *) MH->vir);
            rp.rows(r, 1, COS.get(), SIN.get());
            memcpy(MCS->vir, COS->vir, half);
            memcpy((char *) MCS->vir + half, SIN->vir, half);
            const int64_t c    = kv.cell_of(r);
            const int     nblk = kv.mask_row(r, (ggml_bf16_t *) MMASK->vir);
            auto *        pn   = (int32_t *) MPN->vir;
            pn[0]              = (int32_t) c;
            pn[1]              = std::max(nblk, (int) (c / PG + 1));
            pn[2]              = (int32_t) ((c / PG) * 2 * PG + c % PG);
            for (const axera_cmm * b : { &ME, &MH, &MCS, &MPN, &MMASK }) {
                (*b)->flush();
            }
            mtp1.k->run(mtp1.in, mtp1.out, "mtp t1");
            MOUT->invalidate();
            MRAW->invalidate();
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MOUT->vir, h.data() + (size_t) r * H, H);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MRAW->vir, hr.data() + (size_t) r * H, H);
        }
    } else {
        const bool kvonly = mtppkv.k != nullptr;  // all but the last row need only their K/V
        axera_cmm & pv_e  = kvonly ? PKE : PE;
        axera_cmm & pv_h  = kvonly ? PKH : PH;
        axera_cmm & pv_cs = kvonly ? PKCS : PCS;
        const size_t e_row = e_f32 ? sizeof(float) : sizeof(ggml_bf16_t);
        const int  P      = (int) (pv_e->size / (H * e_row));  // rows per call
        const int  n_rows = kvonly ? T - 1 : T;
        auto run_t1 = [&](int r) {
            row_bf16(embd, r, (ggml_bf16_t *) ME->vir);
            row_bf16(hin, r, (ggml_bf16_t *) MH->vir);
            rp.rows(r, 1, COS.get(), SIN.get());
            memcpy(MCS->vir, COS->vir, half);
            memcpy((char *) MCS->vir + half, SIN->vir, half);
            const int64_t c    = kv.cell_of(r);
            const int     nblk = kv.mask_row(r, (ggml_bf16_t *) MMASK->vir);
            auto *        pn   = (int32_t *) MPN->vir;
            pn[0]              = (int32_t) c;
            pn[1]              = std::max(nblk, (int) (c / PG + 1));
            pn[2]              = (int32_t) ((c / PG) * 2 * PG + c % PG);
            for (const axera_cmm * b : { &ME, &MH, &MCS, &MPN, &MMASK }) {
                (*b)->flush();
            }
            mtp1.k->run(mtp1.in, mtp1.out, "mtp t1");
            MOUT->invalidate();
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) MOUT->vir, h.data() + (size_t) r * H, H);
        };
        for (int it = 0; it < n_rows; it += P) {
            const int n = std::min(P, n_rows - it);
            if (n < P) {  // (a full chunk overwrites every row)
                memset((char *) pv_e->vir + (size_t) n * H * e_row, 0, (size_t) (P - n) * H * e_row);
                memset((char *) pv_h->vir + (size_t) n * H * (h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)), 0,
                       (size_t) (P - n) * H * (h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)));
            }
            // mtp.promptkv has no pn input. Keep its cells in host memory instead of writing three
            // uncached CMM words per row and reading them back again for the K/V scatter.
            std::vector<int32_t> pn_host;
            if (kvonly) {
                pn_host.resize((size_t) 3 * P);
            }
            auto * pn = kvonly ? pn_host.data() : (int32_t *) PPN->vir;
            const int nmax = kvonly ? 0 : kv.chunk_mask((const int32_t *) pos->data, it, P, n, (ggml_bf16_t *) PMASK->vir);
            axera_host_pool().run((int64_t) 2 * n, [&](int64_t j) {
                const int rr = (int) (j % n);
                if (j < n) {
                    if (e_f32) {
                        memcpy((char *) pv_e->vir + (size_t) rr * H * sizeof(float),
                               (const char *) embd->data + (size_t) (it + rr) * embd->nb[1], H * sizeof(float));
                    } else {
                        row_bf16(embd, it + rr, (ggml_bf16_t *) pv_e->vir + (size_t) rr * H);
                    }
                } else {
                    if (h_f32) {
                        memcpy((char *) pv_h->vir + (size_t) rr * H * sizeof(float),
                               (const char *) hin->data + (size_t) (it + rr) * hin->nb[1], H * sizeof(float));
                    } else {
                        row_bf16(hin, it + rr, (ggml_bf16_t *) pv_h->vir + (size_t) rr * H);
                    }
                }
            });
            for (int r = 0; r < P; ++r) {
                const int     rr = std::min(r, n - 1);  // padded rows: the last row's positions and mask, later cells
                const int64_t c  = kv.cell_of(it + rr) + (r - rr);
                GGML_ASSERT(c < kv_cells);
                rp.rows(it + rr, 1, COS.get(), SIN.get());
                memcpy((char *) pv_cs->vir + r * 2 * half, COS->vir, half);
                memcpy((char *) pv_cs->vir + r * 2 * half + half, SIN->vir, half);
                pn[3 * r]     = (int32_t) c;
                pn[3 * r + 1] = (int32_t) (c / PG + 1);
                pn[3 * r + 2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
            }
            pv_e->flush(pv_e->vir, (size_t) n * H * e_row);
            pv_h->flush(pv_h->vir, (size_t) n * H * (h_f32 ? sizeof(float) : sizeof(ggml_bf16_t)));
            pv_cs->flush(pv_cs->vir, (size_t) n * 2 * half);
            if (!kvonly) {
                PPN->flush();
            }
            if (kvonly) {
                mtppkv.k->run(mtppkv.in, mtppkv.out, "mtp prompt kv");
                PK->invalidate();
                PV->invalidate();
                int64_t pg0 = -1, pg1 = -1;
                for (int r = 0; r < n; ++r) {  // each row's K/V into its paged cell (K page then V page)
                    const int64_t cell = pn[3 * r];
                    const int64_t page = cell / PG, cl = cell % PG;
                    memcpy((char *) MKV->vir + (size_t) ((page * 2 + 0) * PG + cl) * kv_row,
                           (char *) PK->vir + (size_t) r * kv_row, kv_row);
                    memcpy((char *) MKV->vir + (size_t) ((page * 2 + 1) * PG + cl) * kv_row,
                           (char *) PV->vir + (size_t) r * kv_row, kv_row);
                    pg0 = pg0 < 0 ? page : std::min(pg0, page);
                    pg1 = std::max(pg1, page);
                }
                if (pg1 >= 0) {
                    MKV->flush((char *) MKV->vir + (size_t) (pg0 * 2) * PG * kv_row,
                               (size_t) ((pg1 - pg0 + 1) * 2) * PG * kv_row);
                }
            } else {
                *(int32_t *) PNBLK->vir = nmax;
                PNBLK->flush();
                PMASK->flush(PMASK->vir, (size_t) nmax * P * PG * sizeof(ggml_bf16_t));
                mtpp.k->run(mtpp.in, mtpp.out, "mtp prompt");
                POUT2->invalidate();
                for (int r = 0; r < n; ++r) {
                    ggml_bf16_to_fp32_row((const ggml_bf16_t *) POUT2->vir + (size_t) r * H,
                                          h.data() + (size_t) (it + r) * H, H);
                }
            }
        }
        if (kvonly) {
            // the last prompt row's full block: its h_nextn seeds the first draft step, and mtp1 writes its KV cell too
            run_t1(T - 1);
        }
    }
    if (hn->flags & GGML_TENSOR_FLAG_COMPUTE) {
        for (int r = 0; r < T; ++r) {
            memcpy((char *) hn->data + r * hn->nb[1], h.data() + (size_t) r * H, H * sizeof(float));
        }
        if (ggml_backend_buffer_is_axera(hn->buffer)) {
            ((ggml_backend_axera_buffer_context *) hn->buffer->context)->flush(hn->data, ggml_nbytes(hn));
        }
    }
    ggml_tensor * logits = names.find("result_output");
    if (logits && logits->ne[1] > 0 && (logits->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        head.bind(man.get("lm_head"), logits->src[0]);
        if (man.has("lm_head.topk")) {
            head.bind_topk(man.get("lm_head.topk"));
        }
        if (T == 1 && man.has("lm_head.draft")) {  // a draft step: the draft vocabulary's head (draft_vocab.i32)
            if (!head.dk) {
                const std::string f  = man.dir + "/draft_vocab.i32";
                FILE *            fp = fopen(f.c_str(), "rb");
                GGML_ASSERT(fp);
                // the vocabulary is the file's, not the kernel's output shape: an argmax draft head has no full row
                fseek(fp, 0, SEEK_END);
                const long fsz = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                GGML_ASSERT(fsz > 0 && fsz % (long) sizeof(int32_t) == 0);
                std::vector<int32_t> ids((size_t) fsz / sizeof(int32_t));
                GGML_ASSERT(fread(ids.data(), sizeof(int32_t), ids.size(), fp) == ids.size());
                fclose(fp);
                axera_timer tb("init: draft head");
                head.bind_draft(man.get("lm_head.draft"), ids);
            }
            head.draft = true;
            const std::vector<char> normed(T, 1);  // (the draft kernel projects the MTP shared-head norm)
            head.run(g, names, h, T, &normed);
        } else {
            if (!head.bind_combined(man.get("lm_head"), logits->src[0])) {
                GGML_LOG_ERROR("axera %s: the MTP head is not a combined lm_head\n", e.who);
                return GGML_STATUS_FAILED;
            }
            if (T == 4 && man.has("lm_head.verify")) {
                head.bind_combined4(man.get("lm_head.verify"), logits->src[0]);
            }
            // the draft context's catch-up rows are not host-sampled; publishing them can replace the target verify
            // rows when both graphs reuse the same host-visible logits buffer
            head.run_combined(g, names, hr, T, nullptr, nullptr, false);
        }
    }
    return GGML_STATUS_SUCCESS;
}

void gdn_mtp::clear() {
    wait();
    if (MKV && page_b && cell > 0) {
        const size_t pages = std::min<size_t>(MKV->size / page_b, (size_t) ((cell + PG - 1) / PG));
        if (pages) {
            memset(MKV->vir, 0, pages * page_b);
            MKV->flush(MKV->vir, pages * page_b);
        }
    }
    cell = 0;
}

gdn_mtp::tail_header gdn_mtp::blob_span(bool partial) const {
    wait();  // the MTP catch-up extends MKV and cell; size and bytes must be consistent
    tail_header th{ 0, 0, page_b, cell };
    if (MKV && page_b && cell > 0) {
        if (partial) {
            th.pages = 1;
            th.page0 = (uint64_t) ((cell - 1) / PG);
        } else {
            th.pages = std::min<uint64_t>(MKV->size / page_b, (uint64_t) ((cell + PG - 1) / PG));
        }
    }
    return th;
}

size_t gdn_mtp::tail_size(bool partial) const {
    return MKV && page_b ? sizeof(tail_header) + blob_span(partial).pages * page_b : 0;
}

void gdn_mtp::tail_write(uint8_t * dst, bool partial) const {
    const tail_header th = blob_span(partial);
    memcpy(dst, &th, sizeof(th));
    if (th.pages) {
        char * src = (char *) MKV->vir + th.page0 * page_b;
        MKV->invalidate(src, th.pages * page_b);
        memcpy(dst + sizeof(th), src, th.pages * page_b);
    }
}

bool gdn_mtp::tail_check(const uint8_t * src, size_t n, bool partial, int64_t cells) const {
    if (!n) {
        return true;  // a blob of a model without the cache: the cache starts empty
    }
    if (!MKV || !page_b || n < sizeof(tail_header)) {
        return false;
    }
    tail_header th;
    memcpy(&th, src, sizeof(th));
    const uint64_t np = MKV->size / page_b;
    return th.page_b == page_b && n == sizeof(th) + th.pages * page_b && th.cell >= 0 && th.cell <= cells &&
           (th.pages == 0 ? th.cell == 0 : th.pages <= np && th.page0 <= np - th.pages && (!partial || th.pages == 1));
}

void gdn_mtp::tail_read(const uint8_t * src, size_t n) {
    wait();
    if (!n) {
        cell = 0;
        return;
    }
    tail_header th;
    memcpy(&th, src, sizeof(th));
    if (th.pages) {
        char * dst = (char *) MKV->vir + th.page0 * page_b;
        memcpy(dst, src + sizeof(th), th.pages * page_b);
        MKV->flush(dst, th.pages * page_b);
    }
    cell = th.cell;
}

}  // namespace axera
