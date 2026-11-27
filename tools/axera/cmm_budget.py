#!/usr/bin/env python3
"""Compute the Qwen3.8-Flash-Next AX8860 CMM / host memory budget.

Categorises the real tensor sizes of the NVFP4/Q8 GGUF and adds the per-context
KV/indexer/GDN state.  This is the budget behind DESIGN.md; no estimated kernel
performance is involved.
"""
import argparse
import json
import math
from pathlib import Path

import gguf

GIB = 1 << 30
MIB = 1 << 20
CONFIG = Path("/data/junda/hf/Qwen3.8-Flash-Next/config.json")


def is_expert(name: str) -> bool:
    return "ffn_gate_up_exps" in name or "ffn_down_exps" in name or "exps_s" in name


def is_ngram(name: str) -> bool:
    return name.startswith("per_layer_token_embd")


def is_vision(name: str) -> bool:
    return name.startswith(("v.", "vision.", "mmproj."))


def state_bytes(context: int, qsa_layers: int, mtp_qsa: int) -> dict:
    pages = math.ceil(context / 256)
    tokens = pages * 256
    kv = (qsa_layers + mtp_qsa) * pages * 2 * 256 * 2 * 256 * 2
    raw = (qsa_layers + mtp_qsa) * (tokens + 4) * 128 * 2
    pooled = (qsa_layers + mtp_qsa) * (tokens // 4 + 2) * 128 * 2
    gdn = 36 * 48 * 128 * 128 * 4
    gdn_conv = 36 * 10240 * 3 * 4
    ple = 10240 * 9 * 4 + 2 * 4
    return {
        "pages": pages,
        "qsa_kv": kv,
        "indexer": raw + pooled,
        "gdn_recurrent": gdn,
        "gdn_conv": gdn_conv,
        "ple": ple,
        "total": kv + raw + pooled + gdn + gdn_conv + ple,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=Path("gguf/q38fn-nvfp4-q8.gguf"))
    ap.add_argument("--out", type=Path, default=Path("logs/cmm_budget.json"))
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    text = cfg["text_config"]
    qsa_layers = sum(1 for t in text["layer_types"] if t == "full_attention")
    mtp_qsa = text.get("mtp_num_hidden_layers", 1)

    reader = gguf.GGUFReader(str(args.gguf))
    groups = {"experts_nvfp4": 0, "ngram_q8": 0, "vision": 0, "text_dense": 0}
    for t in reader.tensors:
        n = t.name
        if is_ngram(n):
            groups["ngram_q8"] += t.n_bytes
        elif is_vision(n):
            groups["vision"] += t.n_bytes
        elif is_expert(n):
            groups["experts_nvfp4"] += t.n_bytes
        else:
            groups["text_dense"] += t.n_bytes
    assert groups["ngram_q8"] > 0 and groups["experts_nvfp4"] > 0

    # Host ngram alternatives.  Q8_0: 34 B per 32 elements + 2 B d.
    rows = text.get("make_ngram_vocab_size_divisible_by", 128)
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    head_dim = text["ple_embed_dim"] // heads
    # padded row count read from the tensor itself
    ngram_tensor = next(t for t in reader.tensors if is_ngram(t.name))
    nrow = int(ngram_tensor.shape[1])
    row_elems = int(ngram_tensor.shape[0])
    q8 = nrow * (row_elems // 32 * 34)
    bf16 = nrow * row_elems * 2
    # Q4_0: 32 values -> 2 B d + 16 B qs = 18 B/block
    q4_0 = nrow * (row_elems // 32 * 18)

    contexts = [32768, 131072, 262144]
    tables = []
    for c in contexts:
        st = state_bytes(c, qsa_layers, mtp_qsa)
        wb = groups["experts_nvfp4"] + groups["text_dense"]
        workspace = 8 * GIB
        cmm = wb + st["total"] + workspace
        tables.append(
            {
                "context": c,
                "weights": wb,
                "state": st["total"],
                "kv": st["qsa_kv"],
                "indexer": st["indexer"],
                "gdn": st["gdn_recurrent"] + st["gdn_conv"],
                "workspace": workspace,
                "cmm_total": cmm,
                "linux_headroom_128g": 128 * GIB - cmm,
            }
        )

    out = {
        "weights": groups,
        "ngram_rows": nrow,
        "ngram_row_elems": row_elems,
        "ngram_q8": q8,
        "ngram_q4_0": q4_0,
        "ngram_bf16": bf16,
        "contexts": tables,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")

    print("weights (GiB):", {k: f"{v / GIB:.3f}" for k, v in groups.items()})
    print(f"ngram rows={nrow} row_elems={row_elems}")
    print(f"ngram Q8_0={q8 / GIB:.3f} Q4_0={q4_0 / GIB:.3f} BF16={bf16 / GIB:.3f} GiB")
    print(f"{'ctx':>8} {'weights':>8} {'state':>7} {'KV':>7} {'idx':>6} {'GDN':>6} {'workspc':>8} {'CMM':>8} {'headroom':>9}")
    for t in tables:
        print(
            f"{t['context']:>8} {t['weights']/GIB:>8.2f} {t['state']/GIB:>7.2f} "
            f"{t['kv']/GIB:>7.2f} {t['indexer']/GIB:>6.2f} {t['gdn']/GIB:>6.3f} "
            f"{t['workspace']/GIB:>8.2f} {t['cmm_total']/GIB:>8.2f} {t['linux_headroom_128g']/GIB:>9.2f}"
        )


if __name__ == "__main__":
    main()
