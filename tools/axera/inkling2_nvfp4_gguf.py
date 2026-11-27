#!/usr/bin/env python3
"""Rewrite the Inkling-2-Mini bf16 GGUF with GGML_NVFP4 tensors.

The MoE expert tensors of layers 0..19 are stored as modelopt NVFP4 in the HF
checkpoint: E2M1 pairs (even element in the low nibble), one E4M3 block scale
per 16 values and a per-expert global factor amax/2688.  They are mapped
losslessly into the GGML NVFP4 block layout (nibble regrouping only).  Layers
20..23 store MXFP8 experts; those are dequantized and quantized once with the
AXERA NVFP4 quantizer.  With --all-fc every other two-dimensional matmul weight
is quantized from bf16 the same way.

Each quantized tensor gets a standard F32 ".scale" sidecar holding one global
scale per slice (one per expert for the routed experts).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
import torch

import gguf
from gguf.constants import GGMLQuantizationType

from inkling2_nvfp4_pack import E4M3, direct_pack, interleave_row_order, pack_rows  # noqa: E402

NV = GGMLQuantizationType.NVFP4
BF16 = GGMLQuantizationType.BF16
F32 = GGMLQuantizationType.F32

DENSE_FC = {
    "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate",
}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u = np.asarray(raw).view("<u2")
    return (u.astype(np.uint32) << 16).view(np.float32)


def quant_codes(x: np.ndarray):
    """[rows, K] f32 -> (pair-packed uint8 [rows,K/2], E4M3 scales [rows,K/16], global float)."""
    from sgl_kernel_axera.quantization import quantize_nvfp4
    q, sc, gs = quantize_nvfp4(torch.from_numpy(np.ascontiguousarray(x)))
    q = q.view(torch.uint8).numpy()
    sc = sc.view(torch.uint8).numpy()
    return q, sc, float(gs.item())


def ggml_pack_from_f32(x: np.ndarray) -> tuple[np.ndarray, float]:
    q, sc, gs = quant_codes(x)
    rows, k = x.shape
    codes = np.empty((rows, k), dtype=np.uint8)
    codes[:, 0::2] = q & 0x0F
    codes[:, 1::2] = q >> 4
    return pack_rows(codes, sc), gs


def ggml_pack_mxfp8(w8: np.ndarray, sc8: np.ndarray) -> tuple[np.ndarray, float]:
    """w8 [rows,K] F8_E4M3, sc8 [rows,K/32] E8M0 -> GGML NVFP4 rows."""
    x = E4M3[w8] * np.repeat(np.exp2(sc8.astype(np.float32) - 127.0), 32, axis=1)
    return ggml_pack_from_f32(x)


class safetensor_reader:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n

    def has(self, name: str) -> bool:
        return name in self.hdr

    def read(self, name: str, dtype) -> np.ndarray:
        a, b = self.hdr[name]["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            buf = f.read(b - a)
        return np.frombuffer(buf, dtype=dtype).copy()

    def shape(self, name: str):
        return self.hdr[name]["shape"]


def expert_layout(sr: safetensor_reader, bid: int):
    p = f"model.llm.layers.{bid}.mlp.experts."
    return {
        "nvfp4": sr.has(p + "w13_weight_amax"),
        "w13": sr.shape(p + "w13_weight"),
        "w2": sr.shape(p + "w2_weight"),
    }


def expert_pack_direct(sr, bid, which, experts):
    p = f"model.llm.layers.{bid}.mlp.experts."
    shp = tuple(sr.shape(p + f"{which}_weight"))
    raw = sr.read(p + f"{which}_weight", np.uint8).reshape(shp)
    wsc = sr.read(p + f"{which}_weight_scale", np.uint8).reshape(tuple(sr.shape(p + f"{which}_weight_scale")))
    amax = sr.read(p + f"{which}_weight_amax", "<f4")
    den = float(sr.read(p + f"{which}_weight_outer_scale_denominator", "<f4")[0])
    ne, rows, nbytes = shp
    K = nbytes * 2
    order = interleave_row_order(rows) if which == "w13" else None
    out = np.empty((ne, rows, K // 64 * 36), dtype=np.uint8)
    for e in range(ne):
        out[e] = direct_pack(raw[e], wsc[e], order)
    return out, (amax / den).astype(np.float32)


def expert_pack_mxfp8(sr, bid, which, experts):
    p = f"model.llm.layers.{bid}.mlp.experts."
    shp = tuple(sr.shape(p + f"{which}_weight"))
    w8 = sr.read(p + f"{which}_weight", np.uint8).reshape(shp)
    sc8 = sr.read(p + f"{which}_weight_scale", np.uint8).reshape(tuple(sr.shape(p + f"{which}_weight_scale")))
    ne, rows, K = shp
    order = interleave_row_order(rows) if which == "w13" else None
    out = np.empty((ne, rows, K // 64 * 36), dtype=np.uint8)
    gs = np.empty(ne, dtype=np.float32)
    for e in range(ne):
        packed, g = ggml_pack_mxfp8(w8[e], sc8[e])
        out[e] = packed[order] if order is not None else packed
        gs[e] = g
    return out, gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("hf")
    ap.add_argument("dst")
    ap.add_argument("--all-fc", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()

    hf_dir = os.path.abspath(a.hf)
    hf_st = os.path.join(hf_dir, "model-00001-of-00001.safetensors")
    if not os.path.exists(hf_st):
        hf_st = a.hf
    sr = safetensor_reader(hf_st)

    r = gguf.GGUFReader(a.src)
    w = gguf.GGUFWriter(a.dst, r.fields["general.architecture"].contents(), use_temp_file=False)
    w.data_alignment = int(r.alignment)
    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key == "general.file_type":
            continue
        val = f.contents()
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(key, val, gguf.GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(key, val, f.types[0])
    w.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_NVFP4))
    w.add_bool("axera.nvfp4_scale2", True)

    report = {"src": a.src, "hf": hf_st, "tensors": {}}
    t0 = time.time()
    n_tot = len(r.tensors)
    for i, t in enumerate(r.tensors):
        name = t.name
        m = re.fullmatch(r"blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.weight", name)
        kind = None
        bid = None
        if m:
            bid, base = int(m.group(1)), m.group(2)
            which = "w13" if base == "ffn_gate_up_exps" else "w2"
            kind = "expert"
        elif a.all_fc:
            m2 = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.weight", name)
            if m2 and m2.group(2) in DENSE_FC:
                bid, which, kind = int(m2.group(1)), m2.group(2), "dense"
            elif name == "output.weight":
                kind, which = "output", "output"

        if kind is None:
            data = np.asarray(t.data)
            if data.dtype == np.uint8 and t.tensor_type == BF16:
                # a uint16 view keeps the (rows, K) shape so the writer does not
                # treat the byte shape as a quantized 64-element block shape
                w.add_tensor(name, data.view(np.uint16), raw_dtype=t.tensor_type)
            elif data.dtype == np.uint8:
                w.add_tensor(name, data, raw_shape=t.shape, raw_dtype=t.tensor_type)
            else:
                w.add_tensor(name, data, raw_dtype=t.tensor_type)
            continue

        if kind == "expert":
            if expert_layout(sr, bid)["nvfp4"]:
                packed, gs = expert_pack_direct(sr, bid, which, 128)
                src_kind = "direct"
            else:
                packed, gs = expert_pack_mxfp8(sr, bid, which, 128)
                src_kind = "mxfp8->nvfp4"
            scale_name = f"blk.{bid}." + ("ffn_up_exps.scale" if which == "w13" else "ffn_down_exps.scale")
        else:
            x = bf16_to_f32(np.asarray(t.data))
            if x.ndim != 2:
                raise ValueError(name)
            packed, g = ggml_pack_from_f32(x)
            gs = np.asarray([g], dtype=np.float32)
            scale_name = name[:-len(".weight")] + ".scale"
            src_kind = "bf16->nvfp4"

        w.add_tensor(name, packed, raw_dtype=NV)
        w.add_tensor(scale_name, gs.astype(np.float32), raw_dtype=F32)
        report["tensors"][name] = {"kind": src_kind, "n_scale": int(gs.size),
                                   "scale2_first": float(gs[0]), "scale2_last": float(gs[-1])}
        print(f"{i + 1}/{n_tot} {name} {src_kind} {time.time() - t0:.1f}s", flush=True)

    print("writing gguf ...", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
