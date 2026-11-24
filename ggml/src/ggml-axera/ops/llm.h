// Axera NPU backend: what every whole-model language adapter does around its kernels with llama.cpp's graph (model
// independent): layer 0's input rows, the layer count and RoPE node, the batch checks, the positions, the sequence
// state's configuration and its result. One copy here instead of one per adapter.
#pragma once

#include "ops.h"
#include "../core/seqstate.h"

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace axera {

// Layer 0's input rows ([H][T] f32): from attn_norm-0 back to its RMS_NORM, that node's source. A layer-0 input the
// graph computes itself (the TTS talker's sum of embeddings) runs on the CPU first. contiguous: the rows must be
// contiguous (false for a view of wider rows, Qwen3-VL's token row in front of its deepstack planes).
ggml_tensor * llm_input(ggml_cgraph * g, ggml_tensor * norm0, bool contiguous = true);

// the layers of the graph's model: blk.<i>.attn_norm.weight from 0
int llm_layers(const std::map<std::string, ggml_tensor *> & t);

// the graph's first ROPE node; nullptr if none
const ggml_tensor * llm_rope_node(const ggml_cgraph * g);

// A whole-model adapter runs one sequence (-np 1) whose batch fills consecutive cells of stream 0: false (logged,
// `who` names the adapter) if this batch is not that.
bool llm_one_sequence(const kv_view & kv, int T, const char * who);

// llama.cpp's positions of T tokens ([T], or M-RoPE's [4][T]) as the kernels' [T][4]
void llm_positions(const ggml_tensor * inp_pos, int T, std::vector<int32_t> & p4);

// the GGML_AXERA_SEQ_CHECK / GGML_AXERA_SEQ_FORCE_MISS switches into a sequence state's configuration
seq_config seq_config_env(int kept_max, int verify_chain = 0);
// an integer environment switch (default if unset)
int env_int(const char * name, int def);

// what seqstate's feed returned for the batch from cell c0, logged: a miss is a warning (the slot replays from cell
// 0), anything else an error. Returns s.
seq_status llm_seq_log(seq_status s, int64_t c0, const seqstate & seq, const char * who);

// The pages of the KV cache an adapter's pools hold: by default (and at most) the kernels' max_pages; llama.cpp's -c
// (its own cache's cells) when the kernels can run on fewer pages (sizable: a pool with a fixed page stride);
// GGML_AXERA_KV_PAGES overrides both.
int llm_kv_pages(int max_pages, int pg, bool sizable = true);

// The paged attention kernels' page table for the token(s) written from cell write_cell after hist cells of history:
// indptr {0, pages}, last_page_len (the cells of the last history page), cache_location {page, offset}; flushed.
void llm_paging(const axera_cmm & indptr, const axera_cmm & last_len, const axera_cmm & cloc, int64_t hist,
                int64_t write_cell, int pg);

// The batch rows llama.cpp wants the logits of (its n_out logits rows of a T-row batch): the graph's out_ids, or rows
// 0 .. n_out - 1 when it wants every row.
void llm_out_rows(const ggml_cgraph * g, const graph_names & names, int n_out, int T, std::vector<int32_t> & rows);

// a [rows] or [rows][cols] f32 copy of any small GGUF tensor
std::vector<float> f32_of(const ggml_tensor * w);

}  // namespace axera
