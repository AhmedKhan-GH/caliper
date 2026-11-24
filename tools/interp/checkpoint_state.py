"""Atomic safe-tensor checkpoints with one parameter copy at a time."""
from __future__ import annotations

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
from .memory_limits import autorelease_scope


def _save_tensors(values, path):
    save_file(values, path)
    with Path(path).open("rb") as source:
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
        groups.append([{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups])
        for name, parameter in layer.named_parameters():
            with autorelease_scope(trainer.device):
                values = {"weight": parameter.detach().cpu().contiguous()}
                for key, value in optimizer.state.get(parameter, {}).items():
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
    return folder, record


def restore_checkpoint(root, trainer, stream):
    """Restore into existing device weights without a whole-state CPU copy."""
    folder, record = checkpoint_record(root)
    if record["settings"] != trainer.settings or record["metadata"]["model_hash"] != trainer.metadata["model_hash"]:
        raise ValueError("checkpoint settings or model identity differ")
    for i, (layer, optimizer) in enumerate(zip(trainer.layers, trainer.optimizers)):
        if len(record["optimizer_groups"][i]) != len(optimizer.param_groups):
            raise ValueError("checkpoint optimizer groups differ")
        for group, saved in zip(optimizer.param_groups, record["optimizer_groups"][i]):
            group.update(saved)
            group["betas"] = tuple(group["betas"])
        for name, parameter in layer.named_parameters():
            with autorelease_scope(trainer.device), safe_open(folder / f"layer_{i:02d}_{name}.safetensors", framework="pt") as source:
                weight = source.get_tensor("weight")
                if weight.shape != parameter.shape or not torch.isfinite(weight).all():
                    raise ValueError("invalid checkpoint weight")
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
                optimizer.state[parameter] = state
        with safe_open(folder / f"activity_{i:02d}.safetensors", framework="pt") as source:
            trainer.last_active[i].copy_(source.get_tensor("last_active"))
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
    runtime = Path(root).parent / "runtime.json"
    if runtime.exists():
        saved = json.loads(runtime.read_text())
        if saved["generation"] == folder.name:
            record["progress"]["elapsed_seconds"] = max(record["progress"]["elapsed_seconds"], saved["elapsed_seconds"])
    return record["progress"]


def export_dictionary(root, trainer, progress):
    """Write one layer at a time and publish the manifest last."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(trainer.layers):
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
