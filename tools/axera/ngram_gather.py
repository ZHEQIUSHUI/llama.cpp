#!/usr/bin/env python3
"""Host-side n-gram gather prototype for the Q8_0 PLE table.

Reads the constants straight from the GGUF metadata and gathers 16 rows per
token from the mmap'd ``per_layer_token_embd.weight`` tensor, dequantising
Q8_0 to BF16.  Reports cold (uniform-random rows) and hot (prompt rows) per
token latency; this is the host half of the unified-memory plan in DESIGN.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import gguf

MASK = (1 << 64) - 1


def shift_right_ignore_eos(tokens, shift, eos):
    n = len(tokens)
    positions = np.arange(n, dtype=np.int64)
    eos_positions = np.where(tokens == eos, positions, -1)
    prev_incl = np.maximum.accumulate(eos_positions)
    previous_eos = np.concatenate((np.array([-1], dtype=np.int64), prev_incl[:-1]))
    segment_start = previous_eos + 1
    pos_in_segment = positions - segment_start
    source = positions - shift
    gather = np.clip(source, 0, None)
    shifted = tokens[gather]
    valid = (pos_in_segment >= shift) & (source >= 0)
    return np.where(valid, shifted, np.int64(eos))


def ngram_rows(tokens, mult, offsets, sizes, eos):
    # Full prefill: two EOS of context, then the tokens, slice back to T.
    hist = np.concatenate((np.full(2, eos, dtype=np.int64), tokens))
    shifted = [shift_right_ignore_eos(hist, s, eos) for s in range(len(mult))]
    # 16 heads: 8 from bigram, 8 from trigram.
    rows = np.empty((len(hist), offsets.size), dtype=np.int64)
    for ngram in range(2, len(mult) + 1):
        start = (ngram - 2) * (offsets.size // (len(mult) - 1))
        stop = start + (offsets.size // (len(mult) - 1))
        mixed = (shifted[0].astype(np.uint64) * np.uint64(mult[0])) & MASK
        for p in range(1, ngram):
            mixed = mixed ^ ((shifted[p].astype(np.uint64) * np.uint64(mult[p])) & MASK)
        for h in range(start, stop):
            rows[:, h] = offsets[h] + (mixed % np.uint64(sizes[h]))
    return rows[-len(tokens):]


def dequant_q8(raw_blocks, count):
    # raw_blocks: (count, 5, 34) uint8 -> (count, 160) float32
    d = raw_blocks[:, :, :2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    q = raw_blocks[:, :, 2:].view(np.int8).astype(np.float32)
    return (d[:, :, None] * q).reshape(count, 160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=Path("gguf/q38fn-nvfp4-q8.gguf"))
    ap.add_argument("--tokens", type=Path, default=Path("output/ref_full/tokens.json"))
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--hot", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path("logs/ngram_gather.json"))
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.gguf))
    f = reader.fields
    mult = [int(x) for x in f["qwen4exp.ple.layer_multipliers"].contents()]
    offsets = np.array(f["qwen4exp.ple.head_offsets"].contents(), dtype=np.int64)
    sizes = np.array(f["qwen4exp.ple.head_vocab_sizes"].contents(), dtype=np.int64)
    eos_value = f["qwen4exp.ple.eos_token_id"].contents()
    eos = int(eos_value[0] if hasattr(eos_value, "__getitem__") else eos_value)
    tensor = next(t for t in reader.tensors if t.name.startswith("per_layer_token_embd"))
    total_rows = int(tensor.shape[1])
    row_elems = int(tensor.shape[0])
    row_bytes = row_elems // 32 * 34
    data = tensor.data.view(np.uint8).reshape(-1)
    block_view = data.reshape(total_rows, row_elems // 32, 34)

    prompt = np.array(json.loads(args.tokens.read_text()), dtype=np.int64)
    rows = ngram_rows(prompt, mult, offsets, sizes, eos)
    assert rows.min() >= 0 and rows.max() < total_rows

    def gather(row_ids):
        blocks = block_view[row_ids]           # (n,) or (n,k) -> memmap copy
        if blocks.ndim == 2:
            blocks = blocks[None]
        flat = blocks.reshape(-1, row_elems // 32, 34)
        return dequant_q8(flat, flat.shape[0])

    # hot: repeat the prompt's rows.
    hot_rows = rows.reshape(-1)
    warm = gather(hot_rows)                    # fault the rows in once
    t0 = time.perf_counter()
    for i in range(args.hot // len(hot_rows) + 1):
        out = gather(hot_rows)
    hot_s = (time.perf_counter() - t0) / (args.hot // len(hot_rows) + 1)
    hot_per_token = hot_s / len(prompt)

    # cold: uniform random rows across the whole table.
    rng = np.random.default_rng(0)
    cold_ids = rng.integers(0, total_rows, size=(args.random // 16, 16), dtype=np.int64)
    t0 = time.perf_counter()
    for ids in cold_ids:
        gather(ids)
    cold_s = time.perf_counter() - t0
    cold_per_token = cold_s / len(cold_ids)

    out_json = {
        "total_rows": total_rows,
        "row_elems": row_elems,
        "row_bytes": row_bytes,
        "table_bytes": total_rows * row_bytes,
        "prompt_tokens": len(prompt),
        "hot_ms_per_token": hot_per_token * 1e3,
        "cold_ms_per_token": cold_per_token * 1e3,
        "cold_rows_per_s": (args.random // 16) / cold_s,
        "sample_rows": rows[:3].tolist(),
        "sample_bytes": float(np.asarray(warm).nbytes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_json, indent=1) + "\n")
    print(json.dumps(out_json, indent=1))


if __name__ == "__main__":
    main()
