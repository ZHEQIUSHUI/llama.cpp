#!/usr/bin/env python3
"""Convert the Qwen3-VL-30B-A3B bf16 GGUF to the AX8860 NVFP4 release GGUF.

Projection and routed-expert matrices use the same E2M1 / E4M3 / FP32-scale2
quantizer as the released AX8860 models.  Every quantized tensor has a standard
F32 ``.scale`` sidecar; routed experts carry one scale2 per expert.  The
embedding and untied output head stay Q8_0.  Norms, the router and the Q/K norm
vectors stay F32.  The text GGUF carries no ``visual.*`` tensors: the mmproj
stays bf16 and is converted separately.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import gguf
from gguf.constants import GGMLQuantizationType, GGUFValueType
from sgl_kernel_axera.quantization import quantize_nvfp4

NV = GGMLQuantizationType.NVFP4
Q8 = GGMLQuantizationType.Q8_0
F32 = GGMLQuantizationType.F32
BF16 = GGMLQuantizationType.BF16

NV_WHOLE = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)
NV_EXPERT = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

E2M1 = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def bf16_bytes_to_f32(raw: np.ndarray) -> np.ndarray:
    """Reader raw bytes (last dimension K * 2) -> F32 (..., K)."""
    u = np.asarray(raw).view("<u2")
    return (u.astype(np.uint32) << 16).view(np.float32)


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    e = (b >> 3).astype(np.int32) & 15
    m = (b & 7).astype(np.float32)
    normal = (1.0 + m / 8.0) * np.exp2((e.astype(np.float32) - 7.0))
    sub = m * np.float32(2.0 ** -9)
    return np.where(e == 0, sub, normal).astype(np.float32)


def quant_slice(x: np.ndarray) -> tuple[np.ndarray, np.float32, float]:
    """[N, K] F32 -> GGML NVFP4 bytes and scale2; also returns relative RMSE."""
    n, k = x.shape
    if n == 0 or k % 64:
        raise ValueError(f"NVFP4 slice shape must be [N, K % 64 == 0], got {x.shape}")
    q, sc, gs = quantize_nvfp4(torch.from_numpy(x))
    q = q.view(torch.uint8).numpy()
    sc = sc.view(torch.uint8).numpy()
    gs = np.float32(gs.item())

    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = q & 0x0f
    codes[:, 1::2] = q >> 4
    groups = codes.reshape(n, k // 16, 16)
    groups[sc == 0] = 0
    qs = (groups[:, :, :8] | (groups[:, :, 8:] << 4)).reshape(n, k // 2)
    out = np.concatenate((sc.reshape(n, k // 64, 4), qs.reshape(n, k // 64, 32)), axis=2)
    out = out.reshape(n, k // 64 * 36)

    pred = (E2M1[groups] * (e4m3_to_f32(sc) * gs)[:, :, None]).reshape(n, k)
    rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - x.astype(np.float64)) ** 2)) /
                 max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-30))
    return out, gs, rmse


def q8_slice(x: np.ndarray) -> np.ndarray:
    return gguf.quants.quantize(x, Q8)


def classify(name: str) -> str:
    if name == "token_embd.weight":
        return "q8"
    if name == "output.weight":
        # The AX8860 lm_head is natively NVFP4.  Keeping the head quantized in
        # the file means CPU and NPU see identical weights (no load-time Q8 to
        # NVFP4 double quantization) and the CPU reference is the real one.
        return "nv_whole"
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return "f32"
    rest = m.group(2)
    if rest in NV_EXPERT:
        return "nv_expert"
    if rest in NV_WHOLE:
        return "nv_whole"
    return "f32"


@dataclass
class report:
    tensors: dict[str, dict] = field(default_factory=dict)

    def add(self, name: str, kind: str, slices: list[float], scales: list[float]) -> None:
        self.tensors[name] = {
            "kind": kind,
            "slices": len(slices),
            "rel_rmse_mean": float(np.mean(slices)) if slices else 0.0,
            "rel_rmse_max": float(np.max(slices)) if slices else 0.0,
            "scale2": scales,
        }


def add_scale(w: gguf.GGUFWriter, name: str, value: np.ndarray) -> None:
    w.add_tensor(name, value.astype(np.float32, copy=False), raw_dtype=F32)


def convert(src: str, dst: str, report_path: str | None) -> None:
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(dst, arch, use_temp_file=False)
    w.data_alignment = int(r.alignment)

    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key == "general.file_type":
            continue
        val = f.contents()
        if f.types[0] == GGUFValueType.ARRAY:
            w.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(key, val, f.types[0])
    w.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_NVFP4))
    w.add_bool("axera.nvfp4_scale2", True)

    rep = report()
    t0 = time.time()
    total = len(r.tensors)
    for i, t in enumerate(r.tensors):
        name = t.name
        kind = classify(name)
        dt = t.tensor_type
        extra: tuple[np.ndarray, str] | None = None

        if kind == "f32":
            if dt == F32:
                w.add_tensor(name, np.asarray(t.data))
            elif dt == BF16:
                w.add_tensor(name, bf16_bytes_to_f32(t.data).astype(np.float32))
            else:
                w.add_tensor(name, np.asarray(t.data), raw_dtype=dt)
        elif kind == "q8":
            x = bf16_bytes_to_f32(t.data)
            w.add_tensor(name, q8_slice(x), raw_dtype=Q8)
        elif kind == "nv_whole":
            x = bf16_bytes_to_f32(t.data)
            q, s2, rmse = quant_slice(x)
            w.add_tensor(name, q, raw_dtype=NV)
            extra = (np.asarray([s2], dtype=np.float32), name[:-len(".weight")] + ".scale")
            rep.add(name, kind, [rmse], [float(s2)])
        elif kind == "nv_expert":
            x = bf16_bytes_to_f32(t.data)
            if x.ndim != 3:
                raise ValueError(f"{name}: expected [experts, rows, columns], got {x.shape}")
            out = np.empty((x.shape[0], x.shape[1], x.shape[2] // 64 * 36), dtype=np.uint8)
            s2 = np.empty(x.shape[0], dtype=np.float32)
            rms = []
            for e in range(x.shape[0]):
                out[e], s2[e], rmse = quant_slice(np.ascontiguousarray(x[e]))
                rms.append(rmse)
            w.add_tensor(name, out, raw_dtype=NV)
            extra = (s2, name[:-len(".weight")] + ".scale")
            rep.add(name, kind, rms, s2.tolist())
        else:
            raise AssertionError(kind)

        if extra is not None:
            add_scale(w, extra[1], extra[0])
        if i % 10 == 0 or i + 1 == total:
            print(f"{i + 1}/{total} {name} {kind} {time.time() - t0:.1f}s", flush=True)

    print("writing GGUF header and tensors ...", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    if report_path:
        with open(report_path, "w") as f:
            json.dump({"source": src, "tensors": rep.tensors}, f, indent=1)
    print(f"wrote {dst} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--report")
    a = ap.parse_args()
    convert(a.src, a.dst, a.report)


if __name__ == "__main__":
    main()
