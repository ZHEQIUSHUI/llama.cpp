"""Lossless modelopt-NVFP4 / MXFP8 -> GGML NVFP4 expert packing for Inkling-2-Mini."""
from __future__ import annotations

import json
import struct
import numpy as np

E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

# E4M3FN decode table (finite values; index 127/255 are NaN -> clamped to 0)
def _e4m3_table():
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        e = (b >> 3) & 15
        m = b & 7
        s = -1.0 if (b >> 7) else 1.0
        if e == 0:
            v = m * 2.0 ** -9
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[b] = s * v
    return out

E4M3 = _e4m3_table()


def pack_rows(codes: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    """codes [rows, K] 0..15 (element order), scales [rows, K//16] E4M3 bytes -> [rows, K//64*36]."""
    rows, K = codes.shape
    assert K % 64 == 0 and scales_u8.shape == (rows, K // 16)
    g = codes.reshape(rows, K // 16, 16)
    qs = (g[:, :, :8] | (g[:, :, 8:] << 4)).reshape(rows, K // 64, 32)
    sc = scales_u8.reshape(rows, K // 64, 4)
    return np.concatenate([sc, qs], axis=2).reshape(rows, K // 64 * 36).copy()


def unpack_rows(packed: np.ndarray, rows: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_rows: -> codes [rows,K], scales [rows,K//16]."""
    p = packed.reshape(rows, K // 64, 36)
    sc = p[:, :, :4].reshape(rows, K // 16)
    gb = p[:, :, 4:].reshape(rows, K // 16, 8)
    codes = np.empty((rows, K // 16, 16), dtype=np.uint8)
    codes[:, :, :8] = gb & 0xF
    codes[:, :, 8:] = gb >> 4
    return codes.reshape(rows, K), sc


def dequant_ggml(packed: np.ndarray, rows: int, K: int, global_scale: np.ndarray) -> np.ndarray:
    codes, sc = unpack_rows(packed, rows, K)
    scaled = E4M3[sc] * np.float32(global_scale)
    return E2M1[codes] * np.repeat(scaled, 16, axis=1)


def direct_pack(raw_u8: np.ndarray, scales_u8: np.ndarray, row_order: np.ndarray | None = None) -> np.ndarray:
    """modelopt packed pairs (even=low nibble) + E4M3 per 16 -> GGML rows [rows, K//64*36]."""
    rows, nbytes = raw_u8.shape
    K = nbytes * 2
    b = raw_u8.reshape(rows, nbytes // 8, 8)
    nb = np.empty_like(b)
    nb[:, :, 0::2] = (b[:, :, :4] & 0x0F) | ((b[:, :, 4:] & 0x0F) << 4)
    nb[:, :, 1::2] = (b[:, :, :4] >> 4) | (b[:, :, 4:] & 0xF0)
    codes = np.empty((rows, K // 16, 16), dtype=np.uint8)
    codes[:, :, :8] = nb & 0xF
    codes[:, :, 8:] = nb >> 4
    out = pack_rows(codes.reshape(rows, K), scales_u8)
    if row_order is not None:
        out = out[row_order]
    return out


def interleave_row_order(rows: int) -> np.ndarray:
    """modelopt w13 stores [gate0, up0, gate1, up1, ...]; GGUF wants [gate..., up...]."""
    return np.concatenate([np.arange(0, rows, 2), np.arange(1, rows, 2)]).astype(np.int64)
