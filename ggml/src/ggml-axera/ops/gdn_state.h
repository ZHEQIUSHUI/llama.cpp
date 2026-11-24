// Axera NPU backend: the recurrent state of the Qwen3.5 / Qwen3.8 hybrid adapters' gated delta-net layers: the CMM
// pools (states ST [layers][value heads][key dim][value dim] f32, conv states CQ / CK / CV bf16), the verification
// snapshots, where each layer's current state lives (its pool part, or a snapshot's row after a verification) and
// the decode / verify outputs bound there, and core/seqstate's restart-point images of it. One copy for the 27B and
// the 0.8B - 9B adapters (their base).
#pragma once

#include "llm.h"

#include <array>
#include <cstdint>
#include <vector>

namespace axera {

struct gdn_state {
    int G = 0;  // the decode group's layers (3 / 4 of them delta-net layers)

    // the pools; one layer's part of each
    axera_cmm ST, CQ, CK, CV;
    size_t    st_b = 0, cq_b = 0, cv_b = 0;
    size_t    state_image_b = 0;  // one image of ST + CQ + CK + CV
    bool      state_zero    = true;  // the pools hold the zero state (CMM starts zero; a load sets it)

    // the current recurrent state of each delta-net layer: its pool part, or a verification snapshot's row (the
    // decode / verify calls' state outputs bound there); the snapshots of the verification groups, two sets
    struct place {
        const ggml_backend_axera_buffer_context * b = nullptr;
        char *                                    p = nullptr;
    };
    struct state_io {
        npu_call * c;
        size_t     i;
        int        kind, g;  // kind: 0 st, 1 cq, 2 ck, 3 cv; g: the delta-net layer
    };
    std::vector<place>                    where[4];
    std::vector<state_io>                 state_ios;
    std::vector<std::array<axera_cmm, 4>> snaps[2];
    // a delta-net layer -> its verification group / its row in the group's snapshot (empty: consecutive groups of
    // G / 4 * 3 layers)
    std::vector<int> state_group, state_local;
    int              cur_set  = -1;  // the snapshot set holding the current state, -1: the pools
    int              ver_set  = -1;  // the snapshot set of the pending verification
    int              ver_rows = 0;

    size_t            part(int kind) const { return kind == 0 ? st_b : kind == 3 ? cv_b : cq_b; }
    const axera_cmm & pool(int kind) const { return kind == 0 ? ST : kind == 1 ? CQ : kind == 2 ? CK : CV; }

    // the decode / verify state outputs onto where[]
    void bind_state();
    // a snapshot's rows copied back into the pools (the current state there again)
    void state_home();
    // the current state: row `row` of snapshot set `set`
    void state_at(int set, int row);
    // seq_model's state image: save; load (nullptr: the zero state; returns true then)
    void state_save(uint8_t * dst);
    bool state_load(const uint8_t * src);
    // a verification's k accepted rows: the state after row k; a copy of it
    void verify_keep(int k);
    void verify_peek(int k, uint8_t * dst) const;

  private:
    int  group_of(size_t g) const { return state_group.empty() ? (int) g / (G / 4 * 3) : state_group[g]; }
    int  local_of(size_t g) const { return state_local.empty() ? (int) g % (G / 4 * 3) : state_local[g]; }
    char * snap_row(int set, size_t g, int kind, int row) const;
};

}  // namespace axera
