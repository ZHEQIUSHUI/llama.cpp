#!/usr/bin/env python3
"""Convert a Qwen4Exp BF16 GGUF to experts NVFP4 and embeddings Q8_0."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

import gguf
from gguf.constants import GGMLQuantizationType as QT, GGUFValueType
from sgl_kernel_axera.quantization import quantize_nvfp4


NV = QT.NVFP4
Q8 = QT.Q8_0
F32 = QT.F32
BF16 = QT.BF16

E2M1 = np.asarray(
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


class LazyChunkedTensor:
    """A uint8 tensor produced chunk-by-chunk at write time.

    Keeps the Q8_0 embedding tables out of RAM during conversion; only the
    gguf writer's ``tofile`` materialises one chunk at a time.
    """

    def __init__(self, chunks, shape, dtype):
        self._chunks = chunks
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def tofile(self, fp):
        written = 0
        for make in self._chunks:
            arr = np.ascontiguousarray(make(), dtype=self.dtype)
            arr.tofile(fp)
            written += arr.nbytes
        if written != self.nbytes:
            raise ValueError(f"lazy tensor wrote {written} bytes, expected {self.nbytes}")


def bf16_bytes_to_f32(raw: np.ndarray) -> np.ndarray:
    u = np.asarray(raw).view("<u2")
    return (u.astype(np.uint32) << 16).view(np.float32)


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    e = (b >> 3).astype(np.int32) & 15
    m = (b & 7).astype(np.float32)
    normal = (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7.0)
    sub = m * np.float32(2.0**-9)
    return np.where(e == 0, sub, normal).astype(np.float32)


def quant_slice(x: np.ndarray) -> tuple[np.ndarray, np.float32, float]:
    """[N,K] F32 -> GGML NVFP4 blocks, global scale and relative RMSE."""
    n, k = x.shape
    if n == 0 or k % 64:
        raise ValueError(f"NVFP4 slice must be [N, K % 64 == 0], got {x.shape}")
    q, sc, gs = quantize_nvfp4(torch.from_numpy(x))
    q = q.view(torch.uint8).numpy()
    sc = sc.view(torch.uint8).numpy()
    gs = np.float32(gs.item())

    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = q & 0x0F
    codes[:, 1::2] = q >> 4
    groups = codes.reshape(n, k // 16, 16)
    groups[sc == 0] = 0
    qs = (groups[:, :, :8] | (groups[:, :, 8:] << 4)).reshape(n, k // 2)
    out = np.concatenate((sc.reshape(n, k // 64, 4), qs.reshape(n, k // 64, 32)), axis=2)
    out = out.reshape(n, k // 64 * 36)

    pred = (E2M1[groups] * (e4m3_to_f32(sc) * gs)[:, :, None]).reshape(n, k)
    rmse = float(
        np.sqrt(np.mean((pred.astype(np.float64) - x.astype(np.float64)) ** 2))
        / max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-30)
    )
    return out, gs, rmse


def is_expert(name: str) -> bool:
    return name.endswith((".ffn_gate_up_exps.weight", ".ffn_down_exps.weight"))


def is_embedding(name: str) -> bool:
    return name in ("token_embd.weight", "per_layer_token_embd.weight")


def scale_name(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"scale sidecar needs a .weight tensor, got {name}")
    return name[:-len(".weight")] + ".scale"


@dataclass
class Report:
    tensors: dict[str, dict] = field(default_factory=dict)

    def expert(self, name: str, rmse: list[float], scales: np.ndarray) -> None:
        self.tensors[name] = {
            "kind": "nvfp4_expert",
            "experts": len(rmse),
            "rel_rmse_mean": float(np.mean(rmse)),
            "rel_rmse_max": float(np.max(rmse)),
            "scale2_min": float(scales.min()),
            "scale2_max": float(scales.max()),
        }

    def embedding(self, name: str, rows: int, source: str) -> None:
        self.tensors[name] = {
            "kind": "q8_0",
            "rows": rows,
            "source": source,
        }


def q8_chunks(data: np.ndarray, rows: int, chunk_rows: int) -> list[Callable[[], np.ndarray]]:
    chunks: list[Callable[[], np.ndarray]] = []
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)

        def load(start=start, stop=stop) -> np.ndarray:
            x = bf16_bytes_to_f32(data[start:stop])
            return gguf.quants.quantize(x, Q8)

        chunks.append(load)
    return chunks


def add_embedding(w: gguf.GGUFWriter, t: gguf.ReaderTensor, chunk_rows: int) -> None:
    rows = int(t.data.shape[0])
    width = int(t.data.shape[1]) // 2
    chunks = q8_chunks(t.data, rows, chunk_rows)
    out = LazyChunkedTensor(
        chunks,
        shape=(rows, width // 32 * 34),
        dtype=np.uint8,
    )
    w.add_tensor(t.name, out, raw_dtype=Q8)


def add_expert(w: gguf.GGUFWriter, t: gguf.ReaderTensor, rep: Report) -> None:
    x = bf16_bytes_to_f32(t.data)
    if x.ndim != 3:
        raise ValueError(f"{t.name}: expected 3D expert tensor, got {x.shape}")
    out = np.empty((x.shape[0], x.shape[1], x.shape[2] // 64 * 36), dtype=np.uint8)
    scales = np.empty(x.shape[0], dtype=np.float32)
    rmses: list[float] = []
    for e in range(x.shape[0]):
        out[e], scales[e], rmse = quant_slice(np.ascontiguousarray(x[e]))
        rmses.append(rmse)
    w.add_tensor(t.name, out, raw_dtype=NV)
    w.add_tensor(scale_name(t.name), scales, raw_dtype=F32)
    rep.expert(t.name, rmses, scales)


def copy_tensor(w: gguf.GGUFWriter, t: gguf.ReaderTensor) -> None:
    w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)


def convert(src: str, dst: str, report_path: str | None, chunk_rows: int) -> None:
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(dst, arch, use_temp_file=False)
    w.data_alignment = int(r.alignment)

    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "general.file_type":
            continue
        val = f.contents()
        if f.types[0] == GGUFValueType.ARRAY:
            w.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(key, val, f.types[0])
    w.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_NVFP4))
    w.add_bool("axera.nvfp4_scale2", True)

    rep = Report()
    started = time.time()
    total = len(r.tensors)
    for i, t in enumerate(r.tensors):
        if is_expert(t.name):
            if t.tensor_type != BF16:
                raise ValueError(f"{t.name}: expected BF16 source, got {t.tensor_type.name}")
            add_expert(w, t, rep)
            kind = "nvfp4"
        elif is_embedding(t.name):
            if t.tensor_type != BF16:
                raise ValueError(f"{t.name}: expected BF16 source, got {t.tensor_type.name}")
            add_embedding(w, t, chunk_rows)
            rep.embedding(t.name, int(t.data.shape[0]), "bf16")
            kind = "q8_0"
        else:
            copy_tensor(w, t)
            kind = t.tensor_type.name
        if i % 10 == 0 or i + 1 == total:
            print(f"{i + 1}/{total} {t.name} {kind} {time.time() - started:.1f}s", flush=True)

    print("writing GGUF header and tensors ...", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    if report_path:
        with open(report_path, "w") as f:
            json.dump(
                {
                    "source": src,
                    "chunk_rows": chunk_rows,
                    "elapsed_s": time.time() - started,
                    "tensors": rep.tensors,
                },
                f,
                indent=1,
            )
    print(f"wrote {dst} in {time.time() - started:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--report")
    ap.add_argument("--chunk-rows", type=int, default=4_000_000)
    a = ap.parse_args()
    if a.chunk_rows <= 0:
        ap.error("--chunk-rows must be positive")
    convert(a.src, a.dst, a.report, a.chunk_rows)


if __name__ == "__main__":
    main()
