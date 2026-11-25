"""Convert local pinned Gemma Scope transcoders without a download or training."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
from types import SimpleNamespace
import uuid
import zipfile

import numpy as np
import torch
from safetensors.torch import save_file

from caliper.interp._dictionary import (
    _GEMMA_IDENTITY, _admit, _finite, _hf_identity, _manifest, _memory_limit,
    _positive, _state_hash, model_identity,
)
from caliper.interp._memory import MemoryGuard
from caliper.interp._gemma_import_identity import (
    _REFERENCE_CONVERSION, _REFERENCE_INPUT, _REFERENCE_INPUT_CONVERSION,
    _REFERENCE_OUTPUT_CONVERSION, _REFERENCE_PROFILE, _reference_gain,
)


MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
SOURCE_ID = "google/gemma-scope-2b-pt-transcoders"
SOURCE_REVISION = "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"
SELECTION_ID = "mntss/gemma-scope-transcoders"
SELECTION_REVISION = "9250a2d4860ce5ed5c96c14d5882b7d8162809a3"
CIRCUIT_REVISION = "82996f144b3371ae858ea5713590d004c218f0c2"
L0 = (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)
SOURCE_PATHS = tuple(f"layer_{i}/width_16k/average_l0_{l0}/params.npz" for i, l0 in enumerate(L0))
WIDTH, LATENTS = 2304, 16384
_NAMES = ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")
_HEADER_BYTES = 65_536
_CHUNK_BYTES = 1024 * 1024


def _shapes(width, latents):
    if type(width) is not int or type(latents) is not int or width < 1 or latents < 1:
        raise ValueError("conversion requires positive integer dimensions")
    return {"W_enc": (width, latents), "b_enc": (latents,),
            "W_dec": (latents, width), "b_dec": (width,), "threshold": (latents,)}


def _array_type(dtype):
    return dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))


def _reference_profile(profile):
    if profile not in ("caliper", _REFERENCE_PROFILE):
        raise ValueError("unsupported Gemma Scope conversion profile")
    return profile == _REFERENCE_PROFILE


def convert_arrays(arrays, gain, *, width, latents, profile="caliper"):
    """Convert one layer in memory, with explicit dimensions for synthetic tests.

    The caller admits array storage before this pure operation. The encoder uses
    feature rows and divides each input column by the learned RMSNorm gain.
    The reference profile also preserves direct source encoder rows. Its output
    tensors use bfloat16. The default retains source values and dtype elsewhere.
    """
    reference = _reference_profile(profile)
    shapes = _shapes(width, latents)
    if set(arrays) != set(_NAMES):
        raise ValueError("Gemma Scope arrays have unexpected names")
    for name, shape in shapes.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or not _array_type(value.dtype)):
            raise ValueError(f"invalid Gemma Scope {name} shape or dtype")
    if len({value.dtype for value in arrays.values()}) != 1:
        raise ValueError("Gemma Scope arrays must share one floating dtype")
    tensors = {name: torch.from_numpy(np.array(value, copy=True, order="C")) for name, value in arrays.items()}
    if any(not _finite(value) for value in tensors.values()):
        raise ValueError("Gemma Scope arrays must be finite")
    if not _positive(tensors["threshold"]):
        raise ValueError("Gemma Scope thresholds must be positive")
    if (not isinstance(gain, torch.Tensor) or gain.device.type != "cpu"
            or gain.shape != (width,) or not gain.is_floating_point()
            or not _finite(gain) or bool((gain == 0).any())):
        raise ValueError("Gemma Scope conversion requires finite nonzero CPU gain")
    if reference and gain.dtype != torch.bfloat16:
        raise ValueError("reference conversion requires the rounded bfloat16 gain")
    # The division uses at least float32 precision, then returns to the source dtype.
    dtype = tensors["W_enc"].dtype
    working = torch.promote_types(torch.float32, torch.promote_types(dtype, gain.dtype))
    encoder = tensors["W_enc"].T.to(working) / gain.detach().to(working).unsqueeze(0)
    if reference:
        # Preserve source bytes before any gain division. Inversion loses bits.
        tensors["W_enc_original"] = tensors["W_enc"].T.to(torch.bfloat16).contiguous()
        dtype = torch.bfloat16
    tensors["W_enc"] = encoder.to(dtype).contiguous()
    if reference:
        tensors = {name: value.to(dtype).contiguous() for name, value in tensors.items()}
    if any(not _finite(value) for value in tensors.values()):
        raise ValueError("Gemma Scope encoder conversion overflows its dtype")
    if not _positive(tensors["threshold"]):
        raise ValueError("Gemma Scope thresholds must remain positive after conversion")
    return tensors


def _central_directory(source):
    """Bound ZIP metadata before ZipFile can allocate its entry table."""
    source.seek(0, os.SEEK_END)
    size = source.tell()
    source.seek(max(0, size - 65_557))
    tail = source.read(65_557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("Gemma Scope requires an NPZ ZIP directory")
    if offset >= 20 and tail[offset - 20:offset - 16] == b"PK\x06\x07":
        raise ValueError("Gemma Scope does not require a ZIP64 directory")
    _, disk, directory_disk, disk_count, count, length, start, comment = struct.unpack_from("<4s4H2LH", tail, offset)
    if (disk or directory_disk or disk_count != len(_NAMES) or count != len(_NAMES)
            or length > _HEADER_BYTES or start + length != size - len(tail) + offset
            or offset + 22 + comment != len(tail)):
        raise ValueError("Gemma Scope NPZ directory exceeds the fixed bounds")
    source.seek(0)


def _npy_header(source):
    # NumPy checks max_header_size after its read. Bound the length first.
    prefix = source.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (b"\x01\x00", b"\x02\x00"):
        raise ValueError("unsupported Gemma Scope NPY version")
    count = 2 if prefix[6] == 1 else 4
    raw_length = source.read(count)
    if len(raw_length) != count:
        raise ValueError("truncated Gemma Scope NPY header")
    length = int.from_bytes(raw_length, "little")
    if length > 10_000:
        raise ValueError("Gemma Scope NPY header exceeds the byte limit")
    raw = source.read(length)
    if len(raw) != length:
        raise ValueError("truncated Gemma Scope NPY header")
    data = io.BytesIO(raw_length + raw)
    reader = (np.lib.format.read_array_header_1_0 if count == 2
              else np.lib.format.read_array_header_2_0)
    return reader(data, max_header_size=10_000)


def _headers(source, shapes, memory):
    _admit(memory, 4 * _HEADER_BYTES, "scope_header_admission")
    _central_directory(source)
    headers = {}
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) != len(_NAMES) or {entry.filename for entry in entries} != {name + ".npy" for name in _NAMES}:
            raise ValueError("Gemma Scope NPZ has unexpected or duplicate members")
        for entry in entries:
            name = entry.filename[:-4]
            if (entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or entry.file_size > math.prod(shapes[name]) * 8 + 10_000):
                raise ValueError("Gemma Scope NPZ member exceeds its array bounds")
            with archive.open(entry) as member:
                shape, order, dtype = _npy_header(member)
                if shape != shapes[name] or not _array_type(dtype):
                    raise ValueError(f"invalid Gemma Scope {name} shape or dtype")
                size = math.prod(shape) * dtype.itemsize
                if member.tell() + size != entry.file_size:
                    raise ValueError("Gemma Scope NPY size differs from its header")
                headers[name] = (shape, order, dtype, size)
    if len({header[2] for header in headers.values()}) != 1:
        raise ValueError("Gemma Scope arrays must share one floating dtype")
    source.seek(0)
    return headers


def _hash_file(source, memory):
    _admit(memory, 2 * _CHUNK_BYTES, "scope_file_hash")
    source.seek(0)
    digest = hashlib.sha256()
    while chunk := source.read(_CHUNK_BYTES):
        memory.check("scope_file_hash", force=False)
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _arrays(source, headers):
    # Recheck the directory before allocation. Read each member into bounded,
    # immutable bytes before NumPy parses it, which closes the header race.
    _central_directory(source)
    arrays = {}
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) != len(_NAMES) or {entry.filename for entry in entries} != {name + ".npy" for name in _NAMES}:
            raise ValueError("Gemma Scope NPZ members changed after admission")
        for name, (shape, order, dtype, size) in headers.items():
            with archive.open(name + ".npy") as member:
                raw = member.read(size + 10_001)
            if len(raw) > size + 10_000:
                raise ValueError("Gemma Scope NPY grew after admission")
            data = io.BytesIO(raw)
            current = _npy_header(data)
            if current != (shape, order, dtype) or data.tell() + size != len(raw):
                raise ValueError("Gemma Scope NPY header changed after admission")
            data.seek(0)
            arrays[name] = np.load(data, allow_pickle=False, max_header_size=10_000)
    return arrays


def _stamp(source):
    info = os.fstat(source.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def load_scope_layer(path, gain, *, width, latents, memory, profile="caliper", evidence=None):
    """Admit and convert a local NPZ layer with no pickle or retained file map."""
    reference = _reference_profile(profile)
    path = Path(path)
    shapes = _shapes(width, latents)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Gemma Scope requires regular NPZ files")
    with path.open("rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("Gemma Scope requires regular NPZ files")
        before = _stamp(source)
        maximum = sum(math.prod(shape) * 8 for shape in shapes.values()) + _HEADER_BYTES
        if before[2] > maximum:
            raise ValueError("Gemma Scope NPZ exceeds the layer byte limit")
        headers = _headers(source, shapes, memory)
        raw_bytes = sum(header[3] for header in headers.values())
        # Source arrays, copies, transpose/division, validation, and serialization.
        # Eight-byte arithmetic also covers a float64 gain on float16 arrays.
        working_bytes = sum(math.prod(shape) * 8 for shape in shapes.values())
        companion_bytes = 2 * width * latents if reference else 0
        _admit(memory, 3 * raw_bytes + 3 * working_bytes + 3 * companion_bytes + 4 * _CHUNK_BYTES,
               "scope_layer_admission")
        original_hash = _hash_file(source, memory)
        arrays = _arrays(source, headers)
        converted = convert_arrays(arrays, gain, width=width, latents=latents, profile=profile)
        del arrays
        if _stamp(source) != before or _hash_file(source, memory) != original_hash:
            raise ValueError("Gemma Scope source changed during conversion")
        if evidence is not None:
            evidence["original_dtype"] = str(headers["W_enc"][2])
    return converted, original_hash


def _snapshot_identity(path):
    return _hf_identity(SimpleNamespace(_name_or_path=str(path), _commit_hash=None))


def _model_admission(model, memory_limit_gb):
    identity = model_identity(model, memory_limit_gb=memory_limit_gb)
    _memory_limit(memory_limit_gb, identity, "conversion")
    if identity != _GEMMA_IDENTITY:
        raise ValueError("Gemma Scope requires the verified pinned Gemma model")
    config = model.config
    if (config.model_type, config.num_hidden_layers, config.hidden_size) != ("gemma2", len(L0), WIDTH):
        raise ValueError("Gemma Scope model dimensions or family differ")
    blocks = model.model.layers
    if len(blocks) != len(L0):
        raise ValueError("Gemma Scope model layer count differs")
    return blocks


def _evaluation_source(record):
    """Keep the local held-out corpus separate from publisher training data."""
    if record is None:
        return None
    from .text_stream import DATASET_ID, DATASET_CONFIG, DATASET_REVISION, partition_metadata
    expected = {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                "dataset_revision": DATASET_REVISION, "split": "train",
                "partitions": partition_metadata()}
    if (not isinstance(record, dict) or set(record) != {*expected, "source_id"}
            or any(record.get(k) != value for k, value in expected.items())
            or not isinstance(record.get("source_id"), str) or not record["source_id"]):
        raise ValueError("conversion requires the pinned evaluation source and partitions")
    encoded = json.dumps(record, allow_nan=False)
    if len(encoded.encode("utf-8")) > 65_536:
        raise ValueError("evaluation source exceeds the metadata byte limit")
    return json.loads(encoded)


def _metadata(rows, evaluation_source=None, *, profile="caliper"):
    reference = _reference_profile(profile)
    metadata = _manifest({
        "format_version": 1, "model_id": MODEL_ID, "model_hash": _GEMMA_IDENTITY,
        "revision": MODEL_REVISION, "layers": len(L0), "width": WIDTH, "latents": LATENTS,
        "activation": "jumprelu", "k": None,
        "read_points": [f"model.layers.{i}.mlp_in" for i in range(len(L0))],
        "write_points": [f"model.layers.{i}.mlp_out" for i in range(len(L0))],
        "training": {"origin": "imported", "dataset_id": "unknown", "dataset_config": "unknown",
                     "dataset_revision": "unknown", "split": "unknown", "partitions": {}, "tokens": 0,
                     "elapsed_seconds": 0, "peak_memory_bytes": 0, "publisher_tokens": None,
                     "local_training": False, "publisher_training_disjointness": "unknown"},
        "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None, "evaluated_tokens": 0},
        "import_source": {
            "kind": "gemma_scope", "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source": {"repo": SOURCE_ID, "revision": SOURCE_REVISION},
            "selection": {"repo": SELECTION_ID, "revision": SELECTION_REVISION, "file": "config.yaml"},
            "reference": {"repo": "decoderesearch/circuit-tracer", "revision": CIRCUIT_REVISION},
            "source_verification": "trusted_local_snapshot_and_file_sha256",
            "original_input": "pure_rms_normalized_pre_mlp_before_learned_gain",
            "input_conversion": "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]",
            "output": "after_post_feedforward_layernorm", "output_conversion": "none",
            "layers": rows,
        },
    })
    if reference:
        metadata["reference_input"] = dict(_REFERENCE_INPUT)
        metadata["import_source"].update(
            input_conversion=_REFERENCE_INPUT_CONVERSION,
            output_conversion=_REFERENCE_OUTPUT_CONVERSION,
            reference_conversion=json.loads(json.dumps(_REFERENCE_CONVERSION)))
    if evaluation_source is not None:
        metadata["evaluation_source"] = _evaluation_source(evaluation_source)
    return metadata


def convert_gemma_scope(source, output, *, model, memory_limit_gb=40, evaluation_source=None,
                        profile="caliper", model_snapshot=None):
    """Publish a complete dictionary from the selected local snapshot atomically.

    The model must pass the established HF verification before this call. Source
    paths identify the pinned trusted local snapshot. Hashes record local bytes,
    and do not authenticate those bytes against a remote service.
    """
    reference = _reference_profile(profile)
    source, output = Path(source), Path(output)
    if reference and (model_snapshot is None or _snapshot_identity(model_snapshot) != _GEMMA_IDENTITY):
        raise ValueError("reference conversion requires the pinned local Gemma model snapshot")
    evaluation_source = _evaluation_source(evaluation_source)
    if _snapshot_identity(source) != f"hf:{SOURCE_ID}@{SOURCE_REVISION}":
        raise ValueError("Gemma Scope source snapshot revision differs")
    blocks = _model_admission(model, memory_limit_gb)
    memory = MemoryGuard(memory_limit_gb, next(model.parameters()).device)
    memory.check("scope_conversion_start")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A sibling lock serializes publishers without exposing an incomplete output.
    lock = output.with_name("." + output.name + ".conversion-lock")
    lock.mkdir()
    staging = lock / ("dictionary-" + uuid.uuid4().hex)
    try:
        if os.path.lexists(output):
            raise FileExistsError("Gemma Scope output already exists")
        staging.mkdir()
        rows = []
        for index, relative in enumerate(SOURCE_PATHS):
            _admit(memory, 8 * WIDTH + 4 * _CHUNK_BYTES, "scope_gain_admission")
            weight = blocks[index].pre_feedforward_layernorm.weight.detach()
            if weight.dtype != torch.float32 or weight.shape != (WIDTH,):
                raise ValueError("Gemma Scope conversion requires float32 model norm weights")
            extra = {}
            if reference:
                gain, extra = _reference_gain(model_snapshot, index, weight, memory)
            else:
                gain = 1 + weight.to(device="cpu")
            tensors, original_hash = load_scope_layer(
                source / relative, gain, width=WIDTH, latents=LATENTS, memory=memory,
                profile=profile, evidence=extra if reference else None)
            destination = staging / f"layer_{index:02d}.safetensors"
            _admit(memory, 2 * sum(value.numel() * value.element_size() for value in tensors.values())
                   + 4 * _CHUNK_BYTES, "scope_save_admission")
            if reference:
                extra["tensor_hashes"] = {name: _state_hash({name: value}, memory=memory)
                                          for name, value in tensors.items()}
            save_file(tensors, destination)
            del tensors
            # Windows requires a writable descriptor for fsync.
            with destination.open("r+b") as saved:
                os.fsync(saved.fileno())
                converted_hash = _hash_file(saved, memory)
            rows.append({"layer": index, "width": WIDTH, "latents": LATENTS, "average_l0": L0[index],
                         "source_file": relative, "source_sha256": original_hash,
                         "file": destination.name, "sha256": converted_hash,
                         "input_gain_hash": _state_hash({"gain": gain}, memory=memory), **extra})
            del gain
            memory.check("scope_layer_saved")
        _admit(memory, 64 * _HEADER_BYTES, "scope_metadata_admission")
        metadata = _metadata(rows, evaluation_source, profile=profile)
        with (staging / "dictionary.json").open("w", encoding="utf-8") as target:
            json.dump(metadata, target, sort_keys=True, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        # Full verified digests detect model changes, including unversioned aliases.
        _model_admission(model, memory_limit_gb)
        if os.path.lexists(output):
            raise FileExistsError("Gemma Scope output already exists")
        os.rename(staging, output)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--memory-limit-gb", type=float, default=40)
    parser.add_argument("--profile", choices=("caliper", _REFERENCE_PROFILE), default="caliper",
                        help="Explicit original-coordinate bfloat16 reference artifact")
    parser.add_argument("--evaluation-source", type=Path,
                        help="Pinned local evaluation and index source JSON, separate from publisher training")
    args = parser.parse_args(argv)
    evaluation_source = None
    if args.evaluation_source is not None:
        if args.evaluation_source.stat().st_size > 65_536:
            raise ValueError("evaluation source exceeds the metadata byte limit")
        evaluation_source = _evaluation_source(json.loads(args.evaluation_source.read_text(encoding="utf-8")))
    # CPU conversion avoids a second device copy of the complete checkpoint.
    _memory_limit(args.memory_limit_gb, _GEMMA_IDENTITY, "conversion")
    if _snapshot_identity(args.model_snapshot) != _GEMMA_IDENTITY:
        raise ValueError("conversion requires the pinned local Gemma model snapshot")
    if _snapshot_identity(args.source) != f"hf:{SOURCE_ID}@{SOURCE_REVISION}":
        raise ValueError("Gemma Scope source snapshot revision differs")
    from caliper.interp._admission import _load_gemma_model

    memory = MemoryGuard(args.memory_limit_gb, "cpu")
    model = _load_gemma_model(args.model_snapshot, device="cpu", memory=memory)
    convert_gemma_scope(args.source, args.output, model=model, memory_limit_gb=args.memory_limit_gb,
                        evaluation_source=evaluation_source, profile=args.profile, model_snapshot=args.model_snapshot)


if __name__ == "__main__":
    main()
