"""Local file and model gain evidence for the pinned Gemma Scope import."""
from __future__ import annotations

import hashlib
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from ._dictionary import (
    _GEMMA_IDENTITY, _VERIFIED_MODELS, _admit, _finite, _hash_admission, _hf_identity, _memory_limit,
    _positive, _state_hash, model_identity,
)
from ._memory import BudgetMemory, MemoryGuard, autorelease_scope, family_memory_limit


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


def _unique_object(pairs):
    result = dict(pairs)
    _require(len(result) == len(pairs))
    return result


def _reference_gain(snapshot, index, weight, memory):
    """Read one source norm vector and bind its cast to verified model bytes."""
    _require(snapshot is not None and _snapshot_identity(snapshot) == _GEMMA_IDENTITY)
    _require(weight.dtype in (torch.float32, torch.bfloat16) and weight.shape == (_WIDTH,))
    _admit(memory, 64 * (4 * 1024 * 1024 + 1) + 4096, "scope_gain_header_admission")
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
            entries = json.loads(source.read(length), object_pairs_hook=_unique_object)
            _require(isinstance(entries, dict) and len(entries) <= 4096)
            if key in entries:
                _require(found is None)
                entry = entries[key]
                _require(isinstance(entry, dict))
                dtype = {"F16": torch.float16, "BF16": torch.bfloat16,
                         "F32": torch.float32, "F64": torch.float64}.get(entry.get("dtype"))
                offsets = entry.get("data_offsets")
                # The pinned checkpoint stores these norm vectors in float32.
                _require(dtype == torch.float32 and entry.get("shape") == [_WIDTH]
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
    _require(torch.equal(gain.view(torch.uint8), (1 + actual.float()).to(torch.bfloat16).view(torch.uint8)))
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


def _source_arrays(path):
    """Read admitted NPZ arrays after bounded ZIP and NPY header checks."""
    shapes = {"W_enc": (_WIDTH, _LATENTS), "W_dec": (_LATENTS, _WIDTH),
              "b_enc": (_LATENTS,), "b_dec": (_WIDTH,), "threshold": (_LATENTS,)}
    with path.open("rb") as source:
        # Refuse a large central directory before ZipFile allocates its table.
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - 65_557))
        tail = source.read(65_557)
        offset = tail.rfind(b"PK\x05\x06")
        _require(offset >= 0 and offset + 22 <= len(tail))
        _require(offset < 20 or tail[offset - 20:offset - 16] != b"PK\x06\x07")
        _, disk, directory_disk, disk_count, count, length, start, comment = struct.unpack_from(
            "<4s4H2LH", tail, offset)
        _require(disk == directory_disk == 0 and disk_count == count == len(shapes)
                 and length <= 65_536 and start + length == size - len(tail) + offset
                 and offset + 22 + comment == len(tail))
        source.seek(0)
        arrays = {}
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            _require(len(entries) == len(shapes)
                     and {entry.filename for entry in entries} == {name + ".npy" for name in shapes})
            for entry in entries:
                name = entry.filename[:-4]
                maximum = math.prod(shapes[name]) * 8 + 10_000
                _require(not entry.flag_bits & 1 and entry.file_size <= maximum
                         and entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED))
                with archive.open(entry) as member:
                    raw = member.read(maximum + 1)
                _require(len(raw) == entry.file_size and len(raw) <= maximum)
                value = io.BytesIO(raw)
                prefix = value.read(8)
                _require(prefix[:6] == b"\x93NUMPY" and prefix[6:] in (b"\x01\x00", b"\x02\x00"))
                count = 2 if prefix[6] == 1 else 4
                length = int.from_bytes(value.read(count), "little")
                _require(0 < length <= 10_000)
                value.seek(8)
                reader = (np.lib.format.read_array_header_1_0 if count == 2
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(value, max_header_size=10_000)
                _require(shape == shapes[name] and dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")))
                _require(value.tell() + math.prod(shape) * dtype.itemsize == len(raw))
                value.seek(0)
                arrays[name] = np.load(value, allow_pickle=False, max_header_size=10_000)
        _require(len({value.dtype for value in arrays.values()}) == 1)
    return arrays


def _reference_layer(path, source_path, row, gain, memory):
    """Check every stored tensor against an independent original-source formula."""
    shapes = {"W_enc": (_LATENTS, _WIDTH), "W_enc_original": (_LATENTS, _WIDTH),
              "W_dec": (_LATENTS, _WIDTH), "b_enc": (_LATENTS,),
              "b_dec": (_WIDTH,), "threshold": (_LATENTS,)}
    source_elements = 2 * _WIDTH * _LATENTS + 2 * _LATENTS + _WIDTH
    # Source reads/copies, float64 arithmetic, cast results, and tensor byte buffers.
    required = 3 * source_elements * 8 + 5 * _WIDTH * _LATENTS * 8 + 4 * _CHUNK_BYTES
    _admit(memory, required, "scope_reference_layer_admission")
    source_before, before = _stamp(source_path.stat()), _stamp(path.stat())
    source_hash = _file_digest(source_path, memory)
    output_hash = _file_digest(path, memory)
    _require(row.get("source_sha256") == source_hash and row.get("sha256") == output_hash)
    arrays = _source_arrays(source_path)
    _require(row.get("original_dtype") == str(arrays["W_enc"].dtype))
    hashes = row.get("tensor_hashes")
    _require(isinstance(hashes, dict) and set(hashes) == set(shapes))
    with path.open("rb") as output:
        prefix = output.read(8)
        length = int.from_bytes(prefix, "little")
        _require(len(prefix) == 8 and 2 <= length <= min(65_536, before[2] - 8))
        entries = json.loads(output.read(length), object_pairs_hook=_unique_object)
        _require(isinstance(entries, dict) and set(entries) == set(shapes))
        spans = []
        for name, shape in shapes.items():
            entry = entries[name]
            _require(isinstance(entry, dict) and entry.get("dtype") == "BF16" and entry.get("shape") == list(shape))
            offsets = entry.get("data_offsets")
            _require(isinstance(offsets, list) and len(offsets) == 2
                     and all(type(value) is int for value in offsets))
            _require(0 <= offsets[0] < offsets[1] <= before[2] - length - 8
                     and offsets[1] - offsets[0] == math.prod(shape) * 2)
            spans.append(offsets)
        end = 0
        for start, stop in sorted(spans):
            _require(start == end)
            end = stop
        _require(end == before[2] - length - 8)
        checked = {}
        for name, shape in shapes.items():
            memory.check("scope_reference_tensor")
            offsets = entries[name]["data_offsets"]
            output.seek(8 + length + offsets[0])
            raw = output.read(offsets[1] - offsets[0])
            _require(len(raw) == offsets[1] - offsets[0])
            actual = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
            source_name = "W_enc" if name == "W_enc_original" else name
            original = torch.from_numpy(np.array(arrays[source_name], copy=True, order="C"))
            _require(_finite(original))
            if source_name == "W_enc":
                original = original.T
            if name == "W_enc":
                working = torch.float64 if original.dtype == torch.float64 else torch.float32
                original = original.to(working) / gain.to(working)[None, :]
            expected = original.to(torch.bfloat16).contiguous()
            _require(_finite(expected) and (name != "threshold" or _positive(expected)))
            # Byte equality also distinguishes signed zeros. No tolerance applies.
            _require(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            digest = _state_hash({name: actual}, memory=memory)
            _require(hashes.get(name) == digest)
            checked[name] = digest
            del raw, actual, original, expected
    del arrays
    _require(_stamp(source_path.stat()) == source_before and _stamp(path.stat()) == before
             and _file_digest(source_path, memory) == source_hash and _file_digest(path, memory) == output_hash)
    return {"source_sha256": source_hash, "sha256": output_hash,
            "original_dtype": row["original_dtype"], "tensor_hashes": checked}


def verified_import_binding(folder, metadata, *, model, memory=None, profile="caliper",
                            source=None, model_snapshot=None):
    """Bind import records to local bytes and exact verified model gain evidence.

    The verified pinned identity requires this check independently of import metadata.
    A pinned identity claim or import declaration also requires this check.
    This evidence checks local consistency. It does not authenticate remote files.
    Reference qualification requires explicit pinned source and model snapshots.
    It checks all tensor bytes against original-source formulas without inversion.
    """
    _require(profile in ("caliper", _REFERENCE_PROFILE))
    reference = profile == _REFERENCE_PROFILE
    if reference:
        _require(metadata.get("reference_input") == _REFERENCE_INPUT)
        _require(source is not None and _snapshot_identity(source) == f"hf:{_SOURCE['repo']}@{_SOURCE['revision']}")
        _require(model_snapshot is not None and _snapshot_identity(model_snapshot) == _GEMMA_IDENTITY)
    else:
        _require("reference_input" not in metadata)
    family = getattr(getattr(model, "config", None), "model_type", None)
    if not reference and family != "gemma2" and metadata.get("model_hash") != _GEMMA_IDENTITY and "import_source" not in metadata:
        return None
    _require(family == "gemma2")
    if memory is None:
        _require(next(model.parameters()).device.type == "cpu")
        memory = MemoryGuard(40, next(model.parameters()).device)
    limit = family_memory_limit(model, memory=memory)
    _memory_limit(limit, _GEMMA_IDENTITY, "import qualification")
    memory = BudgetMemory(memory, limit)
    _admit(memory, 4 * _CHUNK_BYTES, "scope_import_admission")
    identity = model_identity(model, memory=memory, memory_limit_gb=limit)
    if not reference and identity != _GEMMA_IDENTITY and metadata.get("model_hash") != _GEMMA_IDENTITY and "import_source" not in metadata:
        return None
    _require(identity == metadata.get("model_hash") == _GEMMA_IDENTITY)
    verified = _VERIFIED_MODELS.get(model)
    if verified is not None:
        memory = BudgetMemory(memory, limit, extra=verified.memory)
        limit = memory.limit / 1e9
    if reference:
        _require(next(model.parameters()).dtype == torch.bfloat16)
    blocks = model.model.layers
    _require(len(blocks) == len(_L0) and metadata.get("layers") == len(_L0)
             and metadata.get("width") == _WIDTH and metadata.get("latents") == _LATENTS
             and metadata.get("activation") == "jumprelu" and metadata.get("k") is None)
    imported = metadata.get("import_source")
    _require(isinstance(imported, dict))
    _require(imported.get("kind") == "gemma_scope" and imported.get("license") == "CC-BY-4.0"
             and imported.get("source") == _SOURCE and imported.get("selection") == _SELECTION
             and imported.get("original_input") == "pure_rms_normalized_pre_mlp_before_learned_gain"
             and imported.get("input_conversion") == (_REFERENCE_INPUT_CONVERSION if reference else
                 "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]")
             and imported.get("output") == "after_post_feedforward_layernorm"
             and imported.get("output_conversion") == (_REFERENCE_OUTPUT_CONVERSION if reference else "none"))
    if reference:
        _require(imported.get("reference_conversion") == _REFERENCE_CONVERSION
                 and type(imported["reference_conversion"]["version"]) is int
                 and imported.get("reference") == {"repo": "decoderesearch/circuit-tracer",
                     "revision": "82996f144b3371ae858ea5713590d004c218f0c2"}
                 and imported.get("source_verification") == "trusted_local_snapshot_and_file_sha256"
                 and metadata.get("model_id") == "google/gemma-2-2b"
                 and metadata.get("revision") == _GEMMA_IDENTITY.rsplit("@", 1)[1])
    else:
        _require("reference_conversion" not in imported)
    rows = imported.get("layers")
    _require(isinstance(rows, list) and len(rows) == len(blocks))
    evidence = []
    stamps = {}
    if reference:
        checkpoint_files = set(islice(Path(model_snapshot).glob("*.safetensors"), 65))
        _require(0 < len(checkpoint_files) <= 64)
        stamps.update((path, _stamp(path.stat())) for path in checkpoint_files)
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
        if reference:
            output_path, source_path = Path(folder) / filename, Path(source) / row["source_file"]
            stamps.update((path, _stamp(path.stat())) for path in (output_path, source_path))
            gain, gain_evidence = _reference_gain(model_snapshot, i, weight, memory)
            _require(all(row.get(key) == value for key, value in gain_evidence.items()))
            layer_evidence = _reference_layer(output_path, source_path, row, gain, memory)
            evidence.append({"layer": i, "file": filename, "source_file": row["source_file"],
                             **gain_evidence, **layer_evidence})
            del gain
            continue
        _require(weight.dtype == torch.float32 and weight.shape == (_WIDTH,))
        _require(weight.device.type == "cpu" or memory is not None)
        _admit(memory, 16 * _WIDTH + 4 * _CHUNK_BYTES, "scope_import_gain_admission")
        with autorelease_scope(weight.device):
            # Match the converter operation and dtype exactly.
            gain = 1 + weight.to(device="cpu")
            _require(bool(torch.isfinite(gain).all()) and not bool((gain == 0).any()))
            gain_hash = _state_hash({"gain": gain}, memory=memory)
            del gain
        digest = _file_digest(Path(folder) / filename, memory)
        _require(row.get("sha256") == digest and row.get("input_gain_hash") == gain_hash)
        evidence.append({"layer": i, "file": filename, "sha256": digest, "input_gain_hash": gain_hash})
    if reference:
        _require(model_identity(model, memory=memory, memory_limit_gb=limit) == identity)
        _require(set(islice(Path(model_snapshot).glob("*.safetensors"), 65)) == checkpoint_files
                 and all(_stamp(path.stat()) == before for path, before in stamps.items()))
        return {"version": 2, "model_hash": identity, "profile": profile,
                "reference_input": dict(_REFERENCE_INPUT), "source": dict(_SOURCE), "layers": evidence}
    return {"version": 1, "model_hash": identity, "layers": evidence}
