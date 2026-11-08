#!/usr/bin/env python3
"""Compare the weight error of the NVFP4 GGUF sources against the bf16 source.

For every tensor the Axera adapter quantizes, this reports the relative RMSE of
three reconstructions versus bf16:

  bf16->MXFP4     the old MXFP4_MOE GGUF as the old adapter reads it
  bf16->NVFP4     the new NVFP4 GGUF (blocks + F32 .scale sidecar)
  MX->NVFP4       the old MXFP4 values quantized again to NVFP4 in memory

The split is the adapter's: attn_qkv is q|k|v, every routed expert is one
slice, and each other tensor is one slice. MTP routed experts stay MXFP4 and
are reported separately.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

import gguf
from gguf.constants import GGMLQuantizationType as QT
from sgl_kernel_axera.quantization import quantize_nvfp4

NV = QT.NVFP4
MX = QT.MXFP4
BF16 = QT.BF16
QKV_ROWS = (2048, 2048, 4096)
NV_WHOLE = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "nextn.eh_proj.weight",
)
NV_EXPERT = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
E2M1 = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

RBF = RMX = RNV = None
RBF_MAP = RMX_MAP = RNV_MAP = None
S2 = {}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u = np.asarray(raw).view("<u2").astype(np.uint32)
    return (u << 16).view(np.float32)


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    e = (b >> 3).astype(np.int32) & 15
    m = (b & 7).astype(np.float32)
    normal = (1.0 + m / 8.0) * np.exp2((e.astype(np.float32) - 7.0))
    sub = m * np.float32(2.0 ** -9)
    return np.where(e == 0, sub, normal).astype(np.float32)


def nv_roundtrip(x: np.ndarray) -> np.ndarray:
    """[N, K] F32 -> the exact values the kernel's NVFP4 code decodes to."""
    import torch
    q, sc, gs = quantize_nvfp4(torch.from_numpy(np.ascontiguousarray(x)))
    q = q.view(torch.uint8).numpy()
    sc = sc.view(torch.uint8).numpy()
    gs = np.float32(gs.item())
    n, k = x.shape
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = q & 0x0f
    codes[:, 1::2] = q >> 4
    groups = codes.reshape(n, k // 16, 16)
    groups[sc == 0] = 0
    return (E2M1[groups] * (e4m3_to_f32(sc) * gs)[:, :, None]).reshape(n, k)


def stats(name: str, kind: str, x: np.ndarray, mx: np.ndarray | None, nv: np.ndarray | None) -> dict:
    x64 = x.astype(np.float64)
    sq = float(np.sum(x64 * x64))
    out = {"kind": kind, "count": int(x.size), "sum_sq": sq}
    if mx is not None:
        out["mx_err2"] = float(np.sum((mx.astype(np.float64) - x64) ** 2))
    if nv is not None:
        out["nv_err2"] = float(np.sum((nv.astype(np.float64) - x64) ** 2))
    return out


def task_slice(name: str, kind: str, r0: int, r1: int, s2i: int) -> dict:
    bt = RBF_MAP[name]
    mt = RMX_MAP[name]
    nt = RNV_MAP[name]
    x = bf16_to_f32(bt.data[r0:r1])
    m = gguf.quants.dequantize(np.asarray(mt.data[r0:r1]), mt.tensor_type).astype(np.float32)
    d = gguf.quants.dequantize(np.asarray(nt.data[r0:r1]), NV).astype(np.float32)
    d *= np.float32(S2[name][s2i])
    return stats(name, kind, x, m, d)


def task(name: str, kind: str, a: int, b: int, c: int) -> dict:
    if kind == "expert":
        return task_expert(name, kind, a, b)
    return task_slice(name, kind, a, b, c)


def task_expert(name: str, kind: str, e0: int, e1: int) -> dict:
    bt = RBF_MAP[name]
    mt = RMX_MAP[name]
    nt = RNV_MAP[name]
    acc = {}
    for e in range(e0, e1):
        x = bf16_to_f32(bt.data[e])
        m = gguf.quants.dequantize(np.asarray(mt.data[e]), mt.tensor_type).astype(np.float32)
        d = gguf.quants.dequantize(np.asarray(nt.data[e]), NV).astype(np.float32)
        d *= np.float32(S2[name][e])
        st = stats(name, kind, x, m, d)
        for k, v in st.items():
            if k == "kind":
                continue
            acc[k] = acc.get(k, 0) + v
    acc["kind"] = kind
    return acc


def mx_task(name: str, kind: str, r0: int, r1: int, s2i: int) -> float:
    mt = RMX_MAP[name]
    m = gguf.quants.dequantize(np.asarray(mt.data[r0:r1]), mt.tensor_type).astype(np.float32)
    x = bf16_to_f32(RBF_MAP[name].data[r0:r1]).astype(np.float64)
    return float(np.sum((nv_roundtrip(m).astype(np.float64) - x) ** 2))


def mx_task_expert(name: str, kind: str, e0: int, e1: int) -> float:
    bt, mt = RBF_MAP[name], RMX_MAP[name]
    acc = 0.0
    for e in range(e0, e1):
        m = gguf.quants.dequantize(np.asarray(mt.data[e]), mt.tensor_type).astype(np.float32)
        x = bf16_to_f32(bt.data[e]).astype(np.float64)
        acc += float(np.sum((nv_roundtrip(m).astype(np.float64) - x) ** 2))
    return acc


def mx_task_dispatch(name: str, kind: str, a: int, b: int, c: int) -> float:
    if kind == "expert":
        return mx_task_expert(name, kind, a, b)
    return mx_task(name, kind, a, b, c)


def worker(init):
    global RBF, RMX, RNV, RBF_MAP, RMX_MAP, RNV_MAP, S2
    RBF, RMX, RNV = init
    RBF_MAP = {t.name: t for t in RBF.tensors}
    RMX_MAP = {t.name: t for t in RMX.tensors}
    RNV_MAP = {t.name: t for t in RNV.tensors}


def classify(name: str):
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    il, rest = int(m.group(1)), m.group(2)
    if il > 40:
        return None
    if il == 40:
        return ("mtp" if rest in NV_EXPERT else None)
    if rest == "attn_qkv.weight":
        return "qkv"
    if rest in NV_EXPERT:
        return "expert"
    if rest in NV_WHOLE:
        return "whole"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--mxfp4", required=True)
    ap.add_argument("--nvfp4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()

    global RBF, RMX, RNV, RBF_MAP, RMX_MAP, RNV_MAP, S2
    RBF, RMX, RNV = gguf.GGUFReader(a.bf16), gguf.GGUFReader(a.mxfp4), gguf.GGUFReader(a.nvfp4)
    RBF_MAP = {t.name: t for t in RBF.tensors}
    RMX_MAP = {t.name: t for t in RMX.tensors}
    RNV_MAP = {t.name: t for t in RNV.tensors}

    # read the F32 sidecars once
    names = {t.name for t in RNV.tensors}
    for t in RNV.tensors:
        if t.tensor_type == QT.F32 and t.name.endswith(".scale"):
            S2[t.name[:-len(".scale")] + ".weight"] = np.asarray(t.data, dtype=np.float32).reshape(-1)

    tasks = []
    mtp = []
    for t in RBF.tensors:
        kind = classify(t.name)
        if kind is None:
            continue
        if kind == "mtp":
            mtp.append(t.name)
            continue
        ne = [int(x) for x in t.shape]
        if kind == "qkv":
            row = 0
            for i, nr in enumerate(QKV_ROWS):
                tasks.append((t.name, kind, row, row + nr, row))
                row += nr
        elif kind == "expert":
            E = ne[2]
            chunk = max(1, (E + 31) // 32)
            for e in range(0, E, chunk):
                tasks.append((t.name, kind, e, min(E, e + chunk), 0))
        else:
            tasks.append((t.name, kind, 0, ne[1], 0))

    t0 = time.time()
    print(f"tasks {len(tasks)}, MTP tensors {len(mtp)}, jobs {a.jobs}", flush=True)
    results = []
    with mp.Pool(a.jobs, initializer=worker, initargs=((RBF, RMX, RNV),)) as pool:
        it = pool.starmap(task, tasks)
        for i, r in enumerate(it):
            results.append((tasks[i][0], r))
            if i % 20 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    print("pool done", time.time() - t0, flush=True)

    agg = {}
    for name, r in results:
        if name not in agg:
            agg[name] = {"kind": r["kind"], "count": 0, "sum_sq": 0.0, "mx_err2": 0.0, "nv_err2": 0.0}
        for k in ("count", "sum_sq", "mx_err2", "nv_err2"):
            agg[name][k] += r[k]
    out = {}
    for name, v in agg.items():
        den = max(v["sum_sq"], 1e-30)
        out[name] = {
            "kind": v["kind"], "count": v["count"],
            "rel_rmse_mxfp4": float(np.sqrt(v["mx_err2"] / den)),
            "rel_rmse_nvfp4": float(np.sqrt(v["nv_err2"] / den)),
            "rel_rmse_mx_to_nv": None,
        }

    # MTP routed experts: new MXFP4 vs old MXFP4 vs bf16
    for name in mtp:
        bt, mt, nt = RBF_MAP[name], RMX_MAP[name], RNV_MAP[name]
        same = bytes(np.asarray(mt.data)) == bytes(np.asarray(nt.data))
        sq = eo = en = 0.0
        for e in range(int(bt.shape[2])):
            x = bf16_to_f32(bt.data[e]).astype(np.float64)
            mo = gguf.quants.dequantize(np.asarray(mt.data[e]), mt.tensor_type).astype(np.float64)
            nn = gguf.quants.dequantize(np.asarray(nt.data[e]), nt.tensor_type).astype(np.float64)
            sq += float(np.sum(x * x))
            eo += float(np.sum((mo - x) ** 2))
            en += float(np.sum((nn - x) ** 2))
        out[name] = {"kind": "mtp_mxfp4", "count": int(np.prod([int(x) for x in bt.shape])),
                     "rel_rmse_mxfp4": float(np.sqrt(eo / sq)),
                     "rel_rmse_nvfp4": float(np.sqrt(en / sq)),
                     "new_bytes_equal_old": same}

    # MXFP4 -> NVFP4 needs a second pass over the MX values; it is the expensive column
    print("MX->NV pass ...", flush=True)

    with mp.Pool(a.jobs, initializer=worker, initargs=((RBF, RMX, RNV),)) as pool:
        it = pool.starmap(mx_task_dispatch, tasks)
    mx_err = {}
    for i, (name, kind, r0, r1, s2i) in enumerate(tasks):
        mx_err[name] = mx_err.get(name, 0.0) + it[i]
        if i % 20 == 0:
            print(f"mx {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    for name, v in out.items():
        if name in mx_err:
            v["rel_rmse_mx_to_nv"] = float(np.sqrt(mx_err[name] / max(agg[name]["sum_sq"], 1e-30)))

    # summary
    def wavg(key):
        num = den = 0.0
        for name, v in out.items():
            if v["kind"] in ("whole", "qkv", "expert"):
                den += v["count"]
                num += v["count"] * v[key] ** 2
        return float(np.sqrt(num / max(den, 1)))

    rep = {
        "source": {"bf16": a.bf16, "mxfp4": a.mxfp4, "nvfp4": a.nvfp4},
        "summary": {
            "params": int(sum(v["count"] for v in out.values() if v["kind"] in ("whole", "qkv", "expert"))),
            "rel_rmse_mxfp4": wavg("rel_rmse_mxfp4"),
            "rel_rmse_nvfp4": wavg("rel_rmse_nvfp4"),
            "rel_rmse_mx_to_nv": wavg("rel_rmse_mx_to_nv"),
        },
        "tensors": out,
    }
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep["summary"], indent=1), flush=True)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
