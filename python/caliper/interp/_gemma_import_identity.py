"""Local file and model gain evidence for the pinned Gemma Scope import."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

import torch

from ._dictionary import _GEMMA_IDENTITY, _hash_admission, _state_hash, model_identity
from ._memory import autorelease_scope, family_memory_limit


_WIDTH, _LATENTS = 2304, 16384
_L0 = (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)
_SOURCE = {"repo": "google/gemma-scope-2b-pt-transcoders",
           "revision": "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"}
_SELECTION = {"repo": "mntss/gemma-scope-transcoders",
              "revision": "9250a2d4860ce5ed5c96c14d5882b7d8162809a3", "file": "config.yaml"}
_CHUNK_BYTES = 1024 * 1024
_CHANGED = "Gemma import binding differs from the current files or verified model gain"


def _require(condition):
    if not condition:
        raise ValueError(_CHANGED)


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _file_digest(path, memory):
    """Hash a bounded regular file without a tensor allocation."""
    maximum = 8 * (2 * _WIDTH * _LATENTS + 2 * _LATENTS + _WIDTH) + 65_536
    path_before = path.stat()
    _require(stat.S_ISREG(path_before.st_mode))
    _hash_admission(memory, force=True)
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum)
        # CPython 3.12 on Windows uses creation time for stat() ctime and
        # change time for fstat() ctime. Compare ctime only within each API.
        _require(all(getattr(before, name) == getattr(path_before, name)
                     for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")))
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            _hash_admission(memory)
            chunk = source.read(min(remaining, _CHUNK_BYTES))
            _require(bool(chunk))
            digest.update(chunk)
            remaining -= len(chunk)
            del chunk
        _require(not source.read(1) and _stamp(before) == _stamp(os.fstat(source.fileno()))
                 and _stamp(path_before) == _stamp(path.stat()))
    return digest.hexdigest()


def verified_import_binding(folder, metadata, *, model, memory=None):
    """Bind import records to local bytes and the exact verified float32 gain.

    The verified pinned identity requires this check independently of import metadata.
    A pinned identity claim or import declaration also requires this check.
    This evidence checks local consistency. It does not authenticate remote files.
    """
    family = getattr(getattr(model, "config", None), "model_type", None)
    if family != "gemma2" and metadata.get("model_hash") != _GEMMA_IDENTITY and "import_source" not in metadata:
        return None
    _require(family == "gemma2")
    limit = None if memory is None else family_memory_limit(model, memory=memory)
    identity = model_identity(model, memory=memory, memory_limit_gb=limit)
    if identity != _GEMMA_IDENTITY and metadata.get("model_hash") != _GEMMA_IDENTITY and "import_source" not in metadata:
        return None
    _require(identity == metadata.get("model_hash") == _GEMMA_IDENTITY)
    blocks = model.model.layers
    _require(len(blocks) == len(_L0) and metadata.get("layers") == len(_L0)
             and metadata.get("width") == _WIDTH and metadata.get("latents") == _LATENTS
             and metadata.get("activation") == "jumprelu" and metadata.get("k") is None)
    imported = metadata.get("import_source")
    _require(isinstance(imported, dict))
    _require(imported.get("kind") == "gemma_scope" and imported.get("license") == "CC-BY-4.0"
             and imported.get("source") == _SOURCE and imported.get("selection") == _SELECTION
             and imported.get("original_input") == "pure_rms_normalized_pre_mlp_before_learned_gain"
             and imported.get("input_conversion") == "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]"
             and imported.get("output") == "after_post_feedforward_layernorm"
             and imported.get("output_conversion") == "none")
    rows = imported.get("layers")
    _require(isinstance(rows, list) and len(rows) == len(blocks))
    evidence = []
    for i, (row, block, l0) in enumerate(zip(rows, blocks, _L0)):
        _require(isinstance(row, dict))
        filename = f"layer_{i:02d}.safetensors"
        _require(all(type(row.get(key)) is int and row[key] == value for key, value in
                     (("layer", i), ("width", _WIDTH), ("latents", _LATENTS), ("average_l0", l0)))
                 and row.get("file") == filename
                 and row.get("source_file") == f"layer_{i}/width_16k/average_l0_{l0}/params.npz"
                 and isinstance(row.get("source_sha256"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is not None)
        weight = block.pre_feedforward_layernorm.weight.detach()
        _require(weight.dtype == torch.float32 and weight.shape == (_WIDTH,))
        _require(weight.device.type == "cpu" or memory is not None)
        _hash_admission(memory, force=True)
        with autorelease_scope(weight.device):
            # Match the converter operation and dtype exactly.
            gain = 1 + weight.to(device="cpu")
            _require(bool(torch.isfinite(gain).all()) and not bool((gain == 0).any()))
            gain_hash = _state_hash({"gain": gain}, memory=memory)
            del gain
        digest = _file_digest(Path(folder) / filename, memory)
        _require(row.get("sha256") == digest and row.get("input_gain_hash") == gain_hash)
        evidence.append({"layer": i, "file": filename, "sha256": digest, "input_gain_hash": gain_hash})
    return {"version": 1, "model_hash": identity, "layers": evidence}
