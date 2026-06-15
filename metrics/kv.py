"""
KV-cache capacity, compression, and model-dimension utilities.

KV-cache maths
--------------
``parse_kv_bits(dtype)``
    Parse a vLLM cache-dtype string (e.g. ``"fp8_e4m3"``, ``"k8v8"``,
    ``"auto"``) into a ``(k_bits, v_bits)`` tuple.  Returns ``None`` for
    unrecognised strings.

``nominal_ratio(dtype)``
    Compute the theoretical compression ratio vs. fp16: ``(16+16) / (k+v)``.
    Returns ``None`` when the dtype is not parseable.

``fp16_bytes_per_token(layers, kv_heads, head_dim)``
    Size in bytes of one token's KV contribution in fp16:
    ``2 (K+V) × layers × kv_heads × head_dim × 2 bytes``.

``compute_kv(...)``
    Derive a ``KvInfo`` from the cache-config labels emitted by vLLM:
    capacity in tokens, used tokens, compression ratio (achieved when
    ``kv_cache_memory_bytes`` is available, otherwise nominal from dtype),
    fp16-equivalent token count, and the memory a full context would consume.

Model dimensions
----------------
``dims_from_config(config)``
    Extract ``{"layers", "kv_heads", "head_dim"}`` from a HuggingFace
    ``config.json`` dict.  Returns ``None`` on any missing or invalid field.

``load_model_dims(root, max_model_len)``
    Load model dimensions from ``<root>/config.json`` when the path is locally
    accessible.  Always returns a ``ModelDims`` (dims may be ``None`` if the
    file is absent or unreadable).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

_FP16 = {"auto", "fp16", "float16", "bf16", "bfloat16", "half"}


def parse_kv_bits(dtype: str | None) -> tuple[int, int] | None:
    if not dtype:
        return None
    d = dtype.lower()
    if d in _FP16:
        return (16, 16)
    if "fp8" in d or "int8" in d or "e4m3" in d or "e5m2" in d:
        return (8, 8)
    m = re.search(r"k(\d+)v(\d+)", d)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def nominal_ratio(dtype: str | None) -> float | None:
    bits = parse_kv_bits(dtype)
    if not bits:
        return None
    bk, bv = bits
    return (16 + 16) / (bk + bv)


def fp16_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> int:
    return 2 * layers * kv_heads * head_dim * 2


@dataclass
class KvInfo:
    dtype: str | None
    capacity_tokens: int | None
    used_tokens: int | None
    ratio: float | None
    ratio_kind: str
    fp16_equiv_tokens: int | None
    fp16_full_ctx_gb: float | None


def compute_kv(
    *,
    cache_dtype: str | None,
    num_gpu_blocks: int | None,
    block_size: int | None,
    kv_usage: float,
    kv_cache_memory_bytes: int | None,
    dims: dict[str, int] | None,
    max_model_len: int | None,
) -> KvInfo:
    capacity = None
    if num_gpu_blocks and block_size:
        capacity = num_gpu_blocks * block_size
    used = round(capacity * kv_usage) if capacity is not None else None

    bpt = None
    if dims and all(k in dims for k in ("layers", "kv_heads", "head_dim")):
        bpt = fp16_bytes_per_token(dims["layers"], dims["kv_heads"], dims["head_dim"])

    ratio: float | None = None
    kind = "none"
    if capacity and bpt and kv_cache_memory_bytes:
        ratio = (capacity * bpt) / kv_cache_memory_bytes
        kind = "achieved"
    else:
        nom = nominal_ratio(cache_dtype)
        if nom is not None:
            ratio, kind = nom, "nominal"

    fp16_equiv = round(capacity / ratio) if (capacity and ratio) else None
    fp16_full_ctx_gb = (max_model_len * bpt / 1e9) if (max_model_len and bpt) else None

    return KvInfo(
        dtype=cache_dtype,
        capacity_tokens=capacity,
        used_tokens=used,
        ratio=ratio,
        ratio_kind=kind,
        fp16_equiv_tokens=fp16_equiv,
        fp16_full_ctx_gb=fp16_full_ctx_gb,
    )


@dataclass
class ModelDims:
    dims: dict[str, int] | None
    max_model_len: int | None


def dims_from_config(config: dict) -> dict[str, int] | None:
    try:
        layers = int(config["num_hidden_layers"])
        n_attn = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads") or n_attn)
        head_dim = config.get("head_dim")
        head_dim = int(head_dim) if head_dim else int(config["hidden_size"]) // n_attn
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    return {"layers": layers, "kv_heads": kv_heads, "head_dim": head_dim}


def load_model_dims(root: str | None, max_model_len: int | None) -> ModelDims:
    dims = None
    if root and os.path.isdir(root):
        cfg = os.path.join(root, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg) as f:
                    dims = dims_from_config(json.load(f))
            except (OSError, json.JSONDecodeError):
                dims = None
    return ModelDims(dims=dims, max_model_len=max_model_len)
