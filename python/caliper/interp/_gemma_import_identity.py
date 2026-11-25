"""Local file and model gain evidence for the pinned Gemma Scope import."""
from __future__ import annotations

import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace

import torch

from ._dictionary import _GEMMA_IDENTITY, _admit, _hash_admission, _hf_identity, _state_hash, model_identity
from ._memory import MemoryGuard, autorelease_scope, family_memory_limit


_WIDTH, _LATENTS = 2304, 16384
_L0 = (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)
_SOURCE = {"repo": "google/gemma-scope-2b-pt-transcoders",
           "revision": "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"}
_SELECTION = {"repo": "mntss/gemma-scope-transcoders",
              "revision": "9250a2d4860ce5ed5c96c14d5882b7d8162809a3", "file": "config.yaml"}
_CHUNK_BYTES = 1024 * 1024
_CHANGED = "Gemma import binding differs from the current files or verified model gain"
_REFERENCE_PROFILE = "circuit_tracer_82996f14"
_REFERENCE_INPUT = {"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original", "dtype": "bfloat16"}
_REFERENCE_INPUT_CONVERSION = "(W_enc.T / reference_gain).to(bfloat16)"
_REFERENCE_OUTPUT_CONVERSION = "direct source cast to bfloat16"
_REFERENCE_CONVERSION = {
    "version": 1, "profile": _REFERENCE_PROFILE, "model_hash": _GEMMA_IDENTITY,
    "gain": "(1 + source_norm_weight.to(float32)).to(bfloat16)",
    "division_dtype": "promote_types(source_dtype, float32)",
    "tensors": {
        "W_enc_original": "W_enc.T.to(bfloat16).contiguous()",
        "W_enc": _REFERENCE_INPUT_CONVERSION,
        **{name: f"{name}.to(bfloat16).contiguous()" for name in ("b_enc", "W_dec", "b_dec", "threshold")},
    },
}


def _snapshot_identity(path):
    return _hf_identity(SimpleNamespace(_name_or_path=str(path), _commit_hash=None))


def _reference_gain(snapshot, index, weight, memory):
    """Read one source norm vector and bind its cast to verified model bytes."""
    _require(snapshot is not None and _snapshot_identity(snapshot) == _GEMMA_IDENTITY)
    _require(weight.dtype in (torch.float32, torch.bfloat16) and weight.shape == (_WIDTH,))
    _admit(memory, 128 * 1024 * 1024, "scope_gain_header_admission")
    files = list(islice(Path(snapshot).glob("*.safetensors"), 65))
    _require(0 < len(files) <= 64)
    key = f"model.layers.{index}.pre_feedforward_layernorm.weight"
    found = None
    for path in files:
        path_before = path.stat()
        _require(stat.S_ISREG(path_before.st_mode))
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            _require(stat.S_ISREG(before.st_mode) and all(
                getattr(before, name) == getattr(path_before, name)
                for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")))
            prefix = source.read(8)
            length = int.from_bytes(prefix, "little")
            _require(len(prefix) == 8 and 2 <= length <= min(4 * 1024 * 1024, before.st_size - 8))
            entries = json.loads(source.read(length))
            _require(isinstance(entries, dict) and len(entries) <= 4096)
            if key in entries:
                _require(found is None)
                entry = entries[key]
                _require(isinstance(entry, dict))
                dtype = {"F16": torch.float16, "BF16": torch.bfloat16,
                         "F32": torch.float32, "F64": torch.float64}.get(entry.get("dtype"))
                offsets = entry.get("data_offsets")
                _require(dtype is not None and entry.get("shape") == [_WIDTH]
                         and isinstance(offsets, list) and len(offsets) == 2
                         and all(type(value) is int for value in offsets))
                _require(0 <= offsets[0] < offsets[1] <= before.st_size - length - 8
                         and offsets[1] - offsets[0] == _WIDTH * dtype.itemsize)
                _admit(memory, 64 * _WIDTH + 4 * _CHUNK_BYTES, "scope_gain_admission")
                source.seek(8 + length + offsets[0])
                raw = source.read(offsets[1] - offsets[0])
                _require(len(raw) == offsets[1] - offsets[0])
                found = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
            _require(_stamp(before) == _stamp(os.fstat(source.fileno()))
                     and _stamp(path_before) == _stamp(path.stat()))
    _require(found is not None)
    with autorelease_scope(weight.device):
        actual = weight.detach().to(device="cpu").contiguous()
        expected = found.to(dtype=actual.dtype)
        _require(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
    gain_float = 1 + found.to(torch.float32)
    gain = gain_float.to(torch.bfloat16)
    _require(bool(torch.isfinite(gain).all()) and not bool((gain == 0).any()))
    return gain, {
        "source_gain_dtype": str(found.dtype).removeprefix("torch."),
        "source_gain_hash": _state_hash({"weight": found}, memory=memory),
        "float32_gain_hash": _state_hash({"gain": gain_float}, memory=memory),
        "input_gain_hash": _state_hash({"gain": gain}, memory=memory),
    }


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
