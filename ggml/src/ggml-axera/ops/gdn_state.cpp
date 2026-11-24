#include "gdn_state.h"

#include <cstring>

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

}  // namespace axera
