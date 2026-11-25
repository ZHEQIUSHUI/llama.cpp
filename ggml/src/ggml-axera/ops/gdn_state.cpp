#include "gdn_state.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstring>

namespace {
constexpr int PG = 256;  // cells per page
}  // namespace

namespace axera {

char * gdn_state::snap_row(int set, size_t g, int kind, int row) const {
    const int         per = G / 4 * 3;  // delta-net layers per group
    const axera_cmm & sn  = snaps[set][group_of(g)][kind];
    return (char *) sn->vir + ((size_t) row * per + local_of(g)) * part(kind);
}

void gdn_state::bind_state() {
    for (const auto & s : state_ios) {
        const place & w = where[s.kind][s.g];
        s.c->out[s.i]   = axera_io(w.b, w.p, (uint32_t) part(s.kind));
    }
}

void gdn_state::state_home() {
    if (cur_set < 0) {
        return;
    }
    axera_timer tl("state home");
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            char * dst = (char *) pool(k)->vir + g * part(k);
            where[k][g].b->invalidate(where[k][g].p, part(k));
            memcpy(dst, where[k][g].p, part(k));
            where[k][g] = { pool(k).get(), dst };
        }
        pool(k)->flush();
    }
    cur_set    = -1;
    state_zero = false;
    bind_state();
}

void gdn_state::state_at(int set, int row) {
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            GGML_ASSERT(k > 0 || state_group.empty() || (state_group[g] >= 0 && state_local[g] >= 0));
            where[k][g] = { snaps[set][group_of(g)][k].get(), snap_row(set, g, k, row) };
        }
    }
    cur_set = set;
    bind_state();
}

void gdn_state::state_save(uint8_t * dst) {
    axera_timer tl("state save");
    size_t      o = 0;
    if (cur_set < 0) {  // the pools: one contiguous copy per kind
        for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
            (*p)->invalidate((*p)->vir, (*p)->size);
            memcpy(dst + o, (*p)->vir, (*p)->size);
            o += (*p)->size;
        }
    } else {  // a verification snapshot: its rows are not contiguous in the pools' order
        for (int k = 0; k < 4; ++k) {
            for (const place & w : where[k]) {
                w.b->invalidate(w.p, part(k));
                memcpy(dst + o, w.p, part(k));
                o += part(k);
            }
        }
    }
    GGML_ASSERT(o == state_image_b);
    state_zero = false;
}

bool gdn_state::state_load(const uint8_t * src) {
    axera_timer tl("state load");
    const bool  had_snapshot = cur_set >= 0;
    ver_set                  = -1;
    if (!src) {
        if (!state_zero || had_snapshot) {
            axera_timer tz("state zero");
            for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
                memset((*p)->vir, 0, (*p)->size);
                (*p)->flush();
            }
        }
        state_zero = true;
    } else {
        size_t o = 0;
        for (const axera_cmm * p : { &ST, &CQ, &CK, &CV }) {
            memcpy((*p)->vir, src + o, (*p)->size);
            (*p)->flush();
            o += (*p)->size;
        }
        GGML_ASSERT(o == state_image_b);
        state_zero = false;
    }
    for (int k = 0; k < 4; ++k) {
        for (size_t g = 0; g < where[k].size(); ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set = -1;
    bind_state();
    return !src;
}

void gdn_state::verify_keep(int k) {
    GGML_ASSERT(ver_set >= 0 && k > 0 && k <= ver_rows);
    state_at(ver_set, k - 1);
}

void gdn_state::verify_peek(int k, uint8_t * dst) const {
    GGML_ASSERT(ver_set >= 0 && k > 0 && k <= ver_rows);
    size_t o = 0;
    for (int kind = 0; kind < 4; ++kind) {
        for (size_t g = 0; g < where[kind].size(); ++g) {
            const axera_cmm & sn = snaps[ver_set][group_of(g)][kind];
            char *            p  = snap_row(ver_set, g, kind, k - 1);
            sn->invalidate(p, part(kind));
            memcpy(dst + o, p, part(kind));
            o += part(kind);
        }
    }
    GGML_ASSERT(o == state_image_b);
}

void gdn_state::bind_decode(const char * who, const manifest & man, params & prm,
                            const std::map<std::string, ggml_tensor *> & t, int L, int H, int ng,
                            const std::vector<int> & ord, const gdn_pool_io & pool_io) {
    const kernel & dk  = man.get("decode.group");
    const uint32_t csb = dk.in_bytes("cs");
    auto bank_io = [&](const kernel & k, const io_desc & d, int gi) -> AX_ENGINE_IO_BUFFER_T {
        auto it = banks.find(d.name.substr(2));
        if (d.name.rfind("k_", 0) != 0 || it == banks.end()) {
            GGML_ABORT("axera %s: %s: unexpected input %s\n", who, k.desc->role.c_str(), d.name.c_str());
        }
        const size_t bb = it->second->size / (L / 4);  // a block's bytes
        GGML_ASSERT(d.bytes == (size_t) (G / 4) * bb);
        return axera_io(it->second.get(), (char *) it->second->vir + (size_t) gi * d.bytes, d.bytes);
    };
    dec.assign(L / G, {});
    for (int gi = 0; gi < L / G; ++gi) {
        npu_call & c = dec[gi];
        c.k      = &dk;
        for (const auto & d : dk.desc->in) {
            if (d.name.rfind("k_", 0) == 0) {
                c.in.push_back(bank_io(dk, d, gi));
                continue;
            }
            if (dk.desc->packs.count(d.name)) {
                c.in.push_back(axera_io(prm.pack(t, *dk.desc, d.name, gi * G), d.bytes));
                continue;
            }
            if (d.name == "head" || d.name == "topk_on" || d.name == "norm_w" || d.name == "w_lm" ||
                d.name == "s_lm" || d.name == "bias" || d.name == "mul_pos" || d.name == "mul_neg" ||
                d.name == "temp") {  // a merged head's: bind_head_calls
                c.in.push_back(AX_ENGINE_IO_BUFFER_T{});
                continue;
            }
            const axera_cmm * b = d.name == "x" ? &A : d.name == "cs" ? &CS : d.name == "pn" ? &PN : d.name == "mask" ? &MASK : nullptr;
            if (!b) {
                GGML_ABORT("axera %s: decode.group: unexpected input %s\n", who, d.name.c_str());
            }
            c.in.push_back(axera_io(*b, d.bytes));
        }
        for (const auto & d : dk.desc->out) {
            if (d.name == "logits" || d.name == "ids" || d.name == "top_logits" || d.name == "values" ||
                d.name == "kids" || d.name == "vmax" || d.name == "sumexp") {  // a merged head's: bind_head_calls
                c.out.push_back(AX_ENGINE_IO_BUFFER_T{});
                continue;
            }
            c.out.push_back(d.name == "out" ? axera_io(A, d.bytes) : pool_io(d.name, gi * G + atoi(d.name.c_str() + strspn(d.name.c_str(), "abcdefghijklmnopqrstuvwxyz"))));
        }
        // this group's converted parameters, read only through its packs now: freed before the next group's
        // (the load's peak: the packs so far and one group's conversions, not every layer's)
        std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> live;
        for (int gj = 0; gj <= gi; ++gj) {
            live.push_back(&dec[gj].in);
        }
        prm.release(live);
    }
    for (int k = 0; k < 4; ++k) {  // the current state: the pools
        where[k].assign(ng, {});
        for (int g = 0; g < ng; ++g) {
            where[k][g] = { pool(k).get(), (char *) pool(k)->vir + g * part(k) };
        }
    }
    cur_set = -1;
    ver_set = -1;
    // a call's state outputs st<j> / cq<j> / ck<j> / cv<j> (layer j of the group from layer l0): bound to where[]
    auto state_outs = [&](npu_call & c, int l0) {
        for (size_t i = 0; i < c.k->desc->out.size(); ++i) {
            const std::string & n = c.k->desc->out[i].name;
            const int kind = n.rfind("st", 0) == 0 ? 0 : n.rfind("cq", 0) == 0 ? 1 : n.rfind("ck", 0) == 0 ? 2 :
                             n.rfind("cv", 0) == 0 ? 3 : -1;
            if (kind >= 0 && isdigit((unsigned char) n[2])) {
                state_ios.push_back({ &c, i, kind, ord[l0 + atoi(n.c_str() + 2)] });
            }
        }
    };
    state_ios.clear();
    for (int gi = 0; gi < L / G; ++gi) {
        state_outs(dec[gi], gi * G);
    }
    // verification: the decode groups' packs, x / out in place, 4 rows' cs / pn / mask, the snapshots
    ver.clear();
    if (man.has("decode.verify")) {
        const kernel & vk = man.get("decode.verify");
        GGML_ASSERT(vk.in_bytes("x") == 4 * H * sizeof(ggml_bf16_t) && vk.in_bytes("cs") == 4 * csb &&
                    vk.in_bytes("pn") == 4 * 3 * sizeof(int32_t) && vk.in_bytes("mask") == 4 * dk.in_bytes("mask"));
        VX    = axera_cmm_alloc(vk.in_bytes("x"));
        VCS   = axera_cmm_alloc(vk.in_bytes("cs"));
        VPN   = axera_cmm_alloc(vk.in_bytes("pn"));
        VMASK = axera_cmm_alloc(vk.in_bytes("mask"));
        VNT   = axera_cmm_alloc(sizeof(int32_t));
        static const char * SN[4] = { "ssnap", "cqsnap", "cksnap", "cvsnap" };
        for (auto & set : snaps) {
            set.clear();
            set.resize(L / G);
            for (auto & a : set) {
                for (int k = 0; k < 4; ++k) {
                    a[k] = axera_cmm_alloc(vk.out_bytes(SN[k]));
                    GGML_ASSERT(a[k]->size == 4 * (size_t) (G / 4 * 3) * part(k));
                }
            }
        }
        ver.assign(L / G, {});
        for (int gi = 0; gi < L / G; ++gi) {
            npu_call & c = ver[gi];
            c.k      = &vk;
            for (const auto & d : vk.desc->in) {
                if (vk.desc->packs.count(d.name)) {  // the decode group's (the same layout)
                    GGML_ASSERT(dk.desc->packs.count(d.name) && dk.desc->packs.at(d.name) == vk.desc->packs.at(d.name));
                    size_t j = 0;
                    while (dk.desc->in[j].name != d.name) {
                        ++j;
                    }
                    c.in.push_back(dec[gi].in[j]);
                    continue;
                }
                if (d.name.rfind("k_", 0) == 0) {
                    c.in.push_back(bank_io(vk, d, gi));
                    continue;
                }
                const axera_cmm * b = d.name == "x" ? &VX : d.name == "cs" ? &VCS : d.name == "pn" ? &VPN :
                                      d.name == "mask" ? &VMASK : d.name == "ntok" ? &VNT : nullptr;
                if (!b) {
                    GGML_ABORT("axera %s: decode.verify: unexpected input %s\n", who, d.name.c_str());
                }
                c.in.push_back(axera_io(*b, d.bytes));
            }
            for (const auto & d : vk.desc->out) {
                int k = -1;
                for (int j = 0; j < 4; ++j) {
                    k = d.name == SN[j] ? j : k;
                }
                c.out.push_back(d.name == "out" ? axera_io(VX, d.bytes) : k >= 0 ? axera_io(snaps[0][gi][k], d.bytes) :
                                                   pool_io(d.name, gi * G + atoi(d.name.c_str() + strspn(d.name.c_str(), "abcdefghijklmnopqrstuvwxyz"))));
            }
            state_outs(c, gi * G);
        }
    }
}

void gdn_state::decode(int64_t c, const uint16_t * x, float * h, const gdn_each & each) {
    memcpy(A->vir, x, H * sizeof(ggml_bf16_t));
    A->flush(A->vir, H * sizeof(ggml_bf16_t));
    batch_rope->rows(0, 1, COS.get(), SIN.get());
    const int nblk = batch_kv->mask_row(0, (ggml_bf16_t *) MASK->vir);
    GGML_ASSERT(nblk > 0 && nblk <= NBLK);
    MASK->flush(MASK->vir, (size_t) nblk * PG * sizeof(ggml_bf16_t));
    const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
    memcpy(CS->vir, COS->vir, half);
    memcpy((char *) CS->vir + half, SIN->vir, half);
    CS->flush();
    int32_t * pn = (int32_t *) PN->vir;
    pn[0] = (int32_t) c;
    pn[1] = nblk;
    pn[2] = (int32_t) ((c / PG) * 2 * PG + c % PG);
    GGML_ASSERT(pn[2] >= 0 && pn[2] < kv_pages * 2 * PG);
    PN->flush();
    {
        axera_timer tl("layers (decode)");
        if (each) {
            for (size_t gi = 0; gi < dec.size(); ++gi) {
                each(gi, gi + 1 == dec.size());
                dec[gi].k->run(dec[gi].in, dec[gi].out, "decode group");
            }
        } else {
            axera_npu_batch nb;  // the groups' dependent calls, waited for once
            for (auto & d : dec) {
                nb.add(*d.k, d.in, d.out, "decode group");
            }
            nb.wait();
        }
    }
    state_zero = false;
    A->invalidate(A->vir, H * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) A->vir, h, H);
}

ggml_status gdn_state::verify(int64_t c0, int row0, int T, const ggml_bf16_t * xb, float * h, int64_t context) {
    GGML_ASSERT(batch_kv && batch_rope && T >= 1 && T <= ver_rows);
    memset(VX->vir, 0, VX->size);
    memcpy(VX->vir, xb, (size_t) T * H * sizeof(ggml_bf16_t));
    VX->flush();
    const size_t half = (ROT / 2) * sizeof(ggml_bf16_t);
    auto *       cs   = (char *) VCS->vir;
    auto *       pn   = (int32_t *) VPN->vir;
    for (int r = 0; r < 4; ++r) {
        const int     rr = std::min(r, T - 1);
        const int64_t c  = c0 + r;
        batch_rope->rows(row0 + rr, 1, COS.get(), SIN.get());
        memcpy(cs + r * 2 * half, COS->vir, half);
        memcpy(cs + r * 2 * half + half, SIN->vir, half);
        const int nblk = batch_kv->mask_row(row0 + rr, (ggml_bf16_t *) VMASK->vir + (size_t) r * context);
        pn[3 * r]      = (int32_t) c;
        pn[3 * r + 1]  = std::max(nblk, (int) (c / PG + 1));
        pn[3 * r + 2]  = (int32_t) ((c / PG) * 2 * PG + c % PG);
    }
    *(int32_t *) VNT->vir = T;
    for (const axera_cmm * b : { &VCS, &VPN, &VMASK, &VNT }) {
        (*b)->flush();
    }
    const int set = cur_set == 0 ? 1 : 0;
    static const char * SN[4] = { "ssnap", "cqsnap", "cksnap", "cvsnap" };
    {
        axera_timer tl("layers (verify)");
        axera_npu_batch nb;  // the groups' dependent calls, waited for once
        for (size_t gi = 0; gi < ver.size(); ++gi) {
            npu_call & c = ver[gi];
            for (int k = 0; k < 4; ++k) {
                c.out[c.k->output(SN[k])] = axera_io(snaps[set][gi][k], snaps[set][gi][k]->size);
            }
            nb.add(*c.k, c.in, c.out, "verify group");
        }
        nb.wait();
    }
    VX->invalidate(VX->vir, (size_t) T * H * sizeof(ggml_bf16_t));
    for (int r = 0; r < T; ++r) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) VX->vir + (size_t) r * H, h + (size_t) (row0 + r) * H, H);
    }
    state_at(set, T - 1);
    ver_set = set;
    return GGML_STATUS_SUCCESS;
}

ggml_status gdn_state::feed(const graph_names & names, int64_t context, const ggml_tensor * x0,
                            const ggml_tensor * inp_pos, const ggml_tensor * rope_node, int T, const char * who,
                            int64_t & c0) {
    const ggml_tensor * k_idxs = names.find("attn_inp_k_idxs");
    const ggml_tensor * kqm    = names.find("attn_inp_kq_mask");
    GGML_ASSERT(k_idxs && k_idxs->type == GGML_TYPE_I64 && k_idxs->ne[0] == T && kqm);
    axera_timer   tkv("kv view");
    const kv_view kv(k_idxs, kqm, context, PG);
    tkv.stop();
    if (!llm_one_sequence(kv, T, who)) {
        return GGML_STATUS_FAILED;
    }
    std::vector<int32_t> pb;
    llm_positions(inp_pos, T, pb);
    const rope rp(rope_node, inp_pos, T);
    GGML_ASSERT(rp.ROT == ROT);
    h_buf.resize((size_t) T * H);
    std::vector<float> & h = h_buf;
    c0 = kv.cell_of(0);
    std::vector<ggml_bf16_t> xb((size_t) T * H);
    {
        axera_timer tin("input rows");
        axera_host_pool().run(T, [&](int64_t r) {
            ggml_fp32_to_bf16_row((const float *) ((const char *) x0->data + (size_t) r * x0->nb[1]), xb.data() + (size_t) r * H, H);
        });
    }
    batch_rows      = xb.data();
    batch_kv        = &kv;
    batch_rope      = &rp;
    out_h           = h.data();
    last_seq_status = 0;
    const seq_status sst = seq->feed(c0, T, (const uint16_t *) xb.data(), pb.data());
    batch_rows = nullptr;  // the batch's rows, view and rope end here
    batch_kv   = nullptr;
    batch_rope = nullptr;
    if (sst != SEQ_OK) {
        last_seq_status = (int) llm_seq_log(sst, c0, *seq, who);
        return GGML_STATUS_FAILED;
    }
    ggml_tensor * hn = names.find("h_nextn");  // the MTP block's input: every row's final norm
    if (hn && (hn->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        axera_timer tn("h_nextn");
        GGML_ASSERT(hn->type == GGML_TYPE_F32 && hn->ne[0] == H && hn->ne[1] == T);
        float         eps;
        const float * w = final_norm(hn, H, &eps);
        axera_host_pool().run(T, [&](int64_t r) {
            const float * x = h.data() + (size_t) r * H;
            float *       y = (float *) ((char *) hn->data + r * hn->nb[1]);
            double        ss = 0;
            for (int i = 0; i < H; ++i) {
                ss += (double) x[i] * x[i];
            }
            const float sc = 1.0f / sqrtf((float) (ss / H) + eps);
            for (int i = 0; i < H; ++i) {
                y[i] = x[i] * sc * w[i];
            }
        });
        if (ggml_backend_buffer_is_axera(hn->buffer)) {
            ((ggml_backend_axera_buffer_context *) hn->buffer->context)->flush(hn->data, ggml_nbytes(hn));
        }
    }
    return GGML_STATUS_SUCCESS;
}

void gdn_state::prefill_pages(int64_t c0, int n, const ggml_bf16_t * x, const int32_t * p4, std::vector<float> & h,
                              int out0, seq_snap * snap, const gdn_prefill_opts & o) {
    GGML_ASSERT(c0 % PG == 0);
    GGML_ASSERT(c0 + n <= kv_cells);  // (the pools' pages)
    for (int r0 = 0; r0 < n; r0 += PG) {
        const int     v    = std::min(PG, n - r0);
        const int64_t cell = c0 + r0;
        if (v < PG && snap && snap->cell == cell) {  // the state before the run's last (partial) page
            state_save(snap->dst);
        }
        // the smallest row variant that holds this page's valid rows (the tails save the padding work)
        gdn_page_set * P = nullptr;
        for (int i = 4; i >= 0; --i) {
            if (pfs[i].k && v <= pfs[i].rows) {
                P = &pfs[i];
                break;
            }
        }
        GGML_ASSERT(P && v <= P->rows);
        memcpy(P->X->vir, x + (size_t) r0 * H, (size_t) v * H * sizeof(ggml_bf16_t));
        memset((ggml_bf16_t *) P->X->vir + (size_t) v * H, 0, (size_t) (P->rows - v) * H * sizeof(ggml_bf16_t));
        P->X->flush();
        if (o.seed_out0) {
            memcpy(P->OUT[0]->vir, P->X->vir, P->X->size);
            P->OUT[0]->flush();
        }
        auto * pp = (int32_t *) P->POS->vir;
        for (int s = 0; s < 3; ++s) {
            for (int r = 0; r < P->rows; ++r) {
                pp[s * P->rows + r] = r < v           ? p4[(size_t) (r0 + r) * 4 + s] :
                                      o.pos_pad == 0 ? 0 :
                                      o.pos_pad == 1 ? p4[(size_t) r0 * 4 + s] :
                                                       p4[(size_t) (r0 + std::max(0, v - 1)) * 4 + s];
            }
        }
        P->POS->flush();
        ((int32_t *) INDPTR->vir)[0] = 0;
        ((int32_t *) INDPTR->vir)[1] = (int32_t) (cell / PG);
        *(int32_t *) LASTLEN->vir    = PG;
        if (o.lastlen2) {
            ((int32_t *) LASTLEN->vir)[1] = PG;
        }
        ((int32_t *) CLOC->vir)[0] = (int32_t) (cell / PG);
        ((int32_t *) CLOC->vir)[1] = 0;
        *(int32_t *) VALID->vir    = v;
        for (const axera_cmm * b : { &INDPTR, &LASTLEN, &CLOC, &VALID }) {
            (*b)->flush();
        }
        if (o.page_begin) {
            o.page_begin(r0, v);
        }
        {
            axera_timer tl("prefill (page)");
            if (o.each) {
                for (size_t k = 0; k < P->calls.size(); ++k) {
                    o.each(k, k + 1 == P->calls.size());
                    P->calls[k].k->run(P->calls[k].in, P->calls[k].out, "prefill");
                }
            } else {
                axera_npu_batch nb;  // the page's dependent calls, waited for once
                for (auto & cl : P->calls) {
                    nb.add(*cl.k, cl.in, cl.out, "prefill");
                }
                nb.wait();
            }
            state_zero = false;
        }
        if (o.page_end) {
            o.page_end();
        }
        axera_timer       to("prefill out rows");
        const axera_cmm & po = P->OUT[P->result];
        po->invalidate(po->vir, (size_t) v * H * sizeof(ggml_bf16_t));
        axera_host_pool().run(v - std::max(0, out0 - r0), [&](int64_t i) {
            const int r = std::max(0, out0 - r0) + (int) i;
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) po->vir + (size_t) r * H, h.data() + (size_t) (r0 + r - out0) * H, H);
        });
    }
}

void gdn_state::rows(int batch_row0, int n, uint16_t * dst) const {
    if (n <= 0) {
        return;
    }
    if (batch_rows) {
        memcpy(dst, batch_rows + (size_t) batch_row0 * H, (size_t) n * H * sizeof(uint16_t));
    } else {
        memset(dst, 0, (size_t) n * H * sizeof(uint16_t));
    }
}

bool gdn_state::verify_rows(int64_t c, int n, const uint16_t * x, int batch_row0, int64_t context) {
    const uint16_t * xr = x;
    if (!xr) {
        src_rows.resize((size_t) n * H);
        rows(batch_row0, n, (uint16_t *) src_rows.data());
        xr = (const uint16_t *) src_rows.data();
    }
    return verify(c, batch_row0, n, (const ggml_bf16_t *) xr, out_h, context) == GGML_STATUS_SUCCESS;
}

}  // namespace axera
