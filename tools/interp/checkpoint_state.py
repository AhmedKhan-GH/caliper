"""Atomic safe-tensor checkpoints with one parameter copy at a time."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from caliper.interp._dictionary import _zero, fallback_metadata
from .memory_limits import autorelease_scope


def _save_tensors(values, path):
    save_file(values, path)
    # Windows file flush requires a handle with write access.
    with Path(path).open("r+b") as source:
        os.fsync(source.fileno())


def write_json(path, value):
    """Replace a JSON file only after the complete new file reaches disk."""
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def dictionary_fingerprint(folder, metadata):
    """Hash stable metadata and every layer file, with no mutable quality report."""
    digest = hashlib.sha256(json.dumps({k: v for k, v in metadata.items() if k != "quality"},
                                     sort_keys=True, allow_nan=False).encode())
    for i in range(metadata["layers"]):
        with (Path(folder) / f"layer_{i:02d}.safetensors").open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(root, trainer, stream, progress, *, elapsed=None):
    """Save Adam, weights, RNG, feature activity, and the next stream position.

    Each weight and its two Adam moments form one small file. The writer never
    holds a second complete Adam state. An atomic pointer selects the generation.
    The prior complete generation remains valid if a write fails.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = "checkpoint-" + uuid.uuid4().hex
    folder = root / generation
    folder.mkdir()
    groups = []
    for i, (layer, optimizer) in enumerate(zip(trainer.layers, trainer.optimizers)):
        groups.append([{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
                      if optimizer is not None else [])
        for name, parameter in layer.named_parameters():
            with autorelease_scope(trainer.device):
                values = {"weight": parameter.detach().cpu().contiguous()}
                if optimizer is None and not _zero(parameter):
                    raise ValueError("inactive checkpoint weights must remain zero")
                for key, value in (optimizer.state.get(parameter, {}) if optimizer is not None else {}).items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError("Adam checkpoint state must contain tensors")
                    values[key] = value.detach().cpu().contiguous()
                _save_tensors(values, folder / f"layer_{i:02d}_{name}.safetensors")
                del values
        with autorelease_scope(trainer.device):
            _save_tensors({"last_active": trainer.last_active[i].cpu()}, folder / f"activity_{i:02d}.safetensors")
    numpy_state = np.random.get_state()
    rng = {"torch_cpu": torch.get_rng_state(), "numpy": torch.from_numpy(numpy_state[1].copy())}
    if trainer.device.type == "mps":
        torch.mps.synchronize()
        rng["torch_mps"] = torch.mps.get_rng_state()
    _save_tensors(rng, folder / "rng.safetensors")
    record = {"version": 1, "settings": trainer.settings, "metadata": trainer.metadata,
              "progress": dict(progress), "stream": stream.state_dict(),
              "initialized": trainer.initialized, "optimizer_groups": groups,
              "python_rng": random.getstate(),
              "numpy_rng": [numpy_state[0], numpy_state[2], numpy_state[3], numpy_state[4]]}
    if elapsed is not None:
        record["progress"]["elapsed_seconds"] = elapsed()
        progress["elapsed_seconds"] = record["progress"]["elapsed_seconds"]
    write_json(folder / "state.json", record)
    previous = json.loads((root / "latest.json").read_text())["generation"] if (root / "latest.json").exists() else None
    write_json(root / "latest.json", {"generation": generation})
    if previous and previous != generation and Path(previous).name == previous and previous.startswith("checkpoint-"):
        shutil.rmtree(root / previous)
    return folder


def checkpoint_record(root):
    root = Path(root)
    name = json.loads((root / "latest.json").read_text())["generation"]
    if Path(name).name != name or not name.startswith("checkpoint-"):
        raise ValueError("invalid checkpoint generation")
    folder = root / name
    record = json.loads((folder / "state.json").read_text())
    if record.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    runtime = root.parent / "runtime.json"
    if runtime.exists():
        saved = json.loads(runtime.read_text())
        if saved["generation"] == folder.name:
            record["progress"]["elapsed_seconds"] = max(record["progress"]["elapsed_seconds"], saved["elapsed_seconds"])
            record["progress"]["checkpoint_seconds"] = saved.get("checkpoint_seconds", 0)
    return folder, record


def validate_checkpoint(record, settings, metadata, source_id):
    """Refuse changed fallback contracts before model or optimizer allocation."""
    if record["settings"] != settings or record["metadata"]["model_hash"] != metadata["model_hash"]:
        raise ValueError("checkpoint settings or model identity differ")
    fallback = fallback_metadata(metadata)
    saved_fallback = fallback_metadata(record["metadata"])
    if fallback != saved_fallback:
        raise ValueError("checkpoint fallback metadata differs")
    if fallback and (settings.get("fallback") != fallback
                     or record["stream"]["source_id"] != source_id
                     or source_id != fallback["source_id"]):
        raise ValueError("checkpoint fallback settings or text source differ")
    if fallback:
        progress = record["progress"]
        pending, next_layer = progress["pending_ids"], progress["next_layer"]
        valid_next = {0, *(index + 1 for index in fallback["active_layers"])}
        if (type(next_layer) is not int or next_layer not in valid_next
                or (pending is None and next_layer != 0)
                or type(progress["tokens"]) is not int or not 0 <= progress["tokens"] <= fallback["token_budget"]
                or (pending is not None and (not isinstance(pending, list) or not pending
                    or len(pending) > settings["batch_tokens"]
                    or progress["tokens"] + len(pending) > fallback["token_budget"]
                    or any(type(token) is not int or token < 0 for token in pending)))
                or len(record["initialized"]) != metadata["layers"]
                or any(record["initialized"][i] is not True for i in fallback["active_layers"])):
            raise ValueError("invalid fallback checkpoint progress or upper initialization")


def restore_checkpoint(root, trainer, stream):
    """Restore into existing device weights without a whole-state CPU copy."""
    folder, record = checkpoint_record(root)
    validate_checkpoint(record, trainer.settings, trainer.metadata, stream.source_id)
    for i, (layer, optimizer) in enumerate(zip(trainer.layers, trainer.optimizers)):
        groups = optimizer.param_groups if optimizer is not None else []
        if len(record["optimizer_groups"][i]) != len(groups):
            raise ValueError("checkpoint optimizer groups differ")
        for group, saved in zip(groups, record["optimizer_groups"][i]):
            group.update(saved)
            group["betas"] = tuple(group["betas"])
        for name, parameter in layer.named_parameters():
            with autorelease_scope(trainer.device), safe_open(folder / f"layer_{i:02d}_{name}.safetensors", framework="pt") as source:
                weight = source.get_tensor("weight")
                if weight.shape != parameter.shape or not torch.isfinite(weight).all():
                    raise ValueError("invalid checkpoint weight")
                if optimizer is None and (not _zero(weight) or set(source.keys()) != {"weight"}):
                    raise ValueError("inactive checkpoint layer has weights or Adam state")
                with torch.no_grad():
                    parameter.copy_(weight)
                del weight
                state = {}
                if set(source.keys()) not in ({"weight"}, {"weight", "step", "exp_avg", "exp_avg_sq"}):
                    raise ValueError("invalid Adam checkpoint keys")
                for key in set(source.keys()) - {"weight"}:
                    value = source.get_tensor(key)
                    if (key != "step" and value.shape != parameter.shape) or not torch.isfinite(value).all():
                        raise ValueError("invalid Adam checkpoint tensor")
                    state[key] = value.to("cpu" if key == "step" else parameter.device)
                if optimizer is not None:
                    optimizer.state[parameter] = state
        with autorelease_scope(trainer.device), safe_open(folder / f"activity_{i:02d}.safetensors", framework="pt") as source:
            activity = source.get_tensor("last_active")
            if optimizer is None and not _zero(activity):
                raise ValueError("inactive checkpoint layer has feature activity")
            trainer.last_active[i].copy_(activity)
    trainer.initialized = record["initialized"]
    trainer.metadata = record["metadata"]
    stream.load_state_dict(record["stream"])
    random.setstate(_tuples(record["python_rng"]))
    with safe_open(folder / "rng.safetensors", framework="pt") as source:
        torch.set_rng_state(source.get_tensor("torch_cpu"))
        algorithm, position, has_gauss, cached = record["numpy_rng"]
        np.random.set_state((algorithm, source.get_tensor("numpy").numpy(), position, has_gauss, cached))
        if trainer.device.type == "mps":
            torch.mps.set_rng_state(source.get_tensor("torch_mps"))
    return record["progress"]


def export_dictionary(root, trainer, progress):
    """Invalidate the prior manifest, then publish it after all layers succeed."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # A partial export must not retain quality metrics for the prior weights.
    (root / "dictionary.json").unlink(missing_ok=True)
    for i, layer in enumerate(trainer.layers):
        if trainer.optimizers[i] is None and any(not _zero(p) for p in layer.parameters()):
            raise ValueError("inactive export weights must remain zero")
        path = root / f"layer_{i:02d}.safetensors"
        with autorelease_scope(trainer.device):
            values = {name: parameter.detach().cpu().contiguous() for name, parameter in layer.named_parameters()}
            _save_tensors(values, path.with_suffix(".new"))
            del values
        os.replace(path.with_suffix(".new"), path)
    trainer.metadata["training"].update(tokens=progress["tokens"], elapsed_seconds=progress["elapsed_seconds"])
    trainer.metadata["quality"] = {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                                   "evaluated_tokens": 0, "status": "not_evaluated"}
    write_json(root / "dictionary.json", trainer.metadata)
