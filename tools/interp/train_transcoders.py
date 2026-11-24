"""Online TopK transcoders with serial layer updates and bounded checkpoints.

Run with ``python -m tools.interp.train_transcoders --help``. Real model jobs
require the shared heavy-job wrapper. The model and tokenizer use local files.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from caliper._interpret_protocol import bounded
from caliper.interp import capture, points
from caliper.interp._dictionary import (
    TopKTranscoder, _finite, _layer_header, _manifest, fallback_metadata, model_identity,
)
from .checkpoint_state import (
    checkpoint_record, dictionary_fingerprint, export_dictionary, restore_checkpoint,
    save_checkpoint, validate_checkpoint, write_json,
)
from .memory_limits import CleanStop, MemoryGuard, StopControl, autorelease_scope
from .model_load import MODEL_ID, MODEL_REVISION, hidden_forward, load_text_model
from .stop_records import clean_stop_record, stop_reason
from .text_stream import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, TextStream, fineweb_documents, partition_metadata


def make_manifest(model, latents, k):
    rows = points(model)
    return {"format_version": 1, "model_id": MODEL_ID, "model_hash": model_identity(model),
            "revision": MODEL_REVISION, "layers": model.config.num_hidden_layers,
            "width": model.config.hidden_size, "latents": latents, "activation": "topk", "k": k,
            "read_points": [p["point"] for p in rows if p.get("probe") == "mlp_in"],
            "write_points": [p["point"] for p in rows if p.get("probe") == "mlp_out"],
            "training": {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                         "dataset_revision": DATASET_REVISION, "split": "train",
                         "partitions": partition_metadata(), "tokens": 0, "elapsed_seconds": 0,
                         "peak_memory_bytes": 0},
            "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                        "evaluated_tokens": 0, "status": "not_evaluated"}}


def trainer_settings(metadata, *, learning_rate=3e-4, microbatch_tokens=256,
                     auxiliary_weight=1 / 32, dead_tokens=100_000, context=256, batch_tokens=4096,
                     upper_half_from=None):
    """Describe checkpoint compatibility before any model or optimizer allocation."""
    settings = {"learning_rate": learning_rate, "microbatch_tokens": microbatch_tokens,
                "auxiliary_weight": auxiliary_weight, "dead_tokens": dead_tokens,
                "context": context, "batch_tokens": batch_tokens,
                "decoder_normalization": "initial_directions_only",
                "layers": metadata["layers"], "width": metadata["width"],
                "latents": metadata["latents"], "k": metadata["k"], "dtype": "float32"}
    fallback = fallback_metadata(metadata)
    if fallback:
        settings.update(fallback=copy.deepcopy(fallback), model_hash=metadata["model_hash"],
                        source_id=metadata["training"]["source_id"],
                        upper_half_from=str(Path(upper_half_from).resolve()) if upper_half_from is not None else None)
    return settings


class Trainer:
    """Complete dictionary layers with Adam only for active layers."""

    def __init__(self, metadata, *, device="cpu", learning_rate=3e-4, microbatch_tokens=256,
                 auxiliary_weight=1 / 32, dead_tokens=100_000, context=256, batch_tokens=4096,
                 upper_half_from=None):
        self.metadata = metadata
        self.fallback_source = copy.deepcopy(metadata.get("fallback_source"))
        self.device = torch.device(device)
        self.settings = trainer_settings(metadata, learning_rate=learning_rate,
                                         microbatch_tokens=microbatch_tokens, auxiliary_weight=auxiliary_weight,
                                         dead_tokens=dead_tokens, context=context, batch_tokens=batch_tokens,
                                         upper_half_from=upper_half_from)
        fallback = fallback_metadata(metadata)
        self.active_layers = tuple(fallback["active_layers"] if fallback else range(metadata["layers"]))
        if (not 0 < learning_rate < 1 or microbatch_tokens < 1 or dead_tokens < 1
                or not 0 <= auxiliary_weight <= 1 or context < 2 or batch_tokens < context
                or not 1 <= metadata["k"] <= metadata["latents"]):
            raise ValueError("invalid trainer settings")
        self.layers, self.optimizers, self.last_active = [], [], []
        self.initialized = [False] * metadata["layers"]
        self.metadata["training"]["optimizer"] = {
            "name": "Adam", **{k: v for k, v in self.settings.items() if k != "upper_half_from"},
            "decay_start": 0.8, "final_lr_ratio": 0.1,
        }
        width, latents = metadata["width"], metadata["latents"]
        for index in range(metadata["layers"]):
            # Separate random encoder and decoder directions. No teacher weights enter here.
            with autorelease_scope(self.device):
                active = index in self.active_layers
                if fallback:
                    encoder = torch.zeros(latents, width, device=self.device)
                    decoder = torch.zeros_like(encoder)
                else:
                    encoder = torch.randn(latents, width, device=self.device) / math.sqrt(width)
                    decoder = 0.1 * F.normalize(torch.randn(latents, width, device=self.device), dim=-1)
                layer = TopKTranscoder(encoder, torch.zeros(latents, device=self.device), decoder,
                                      torch.zeros(width, device=self.device), metadata["k"], trainable=active)
                self.layers.append(layer)
                self.optimizers.append(torch.optim.Adam(layer.parameters(), lr=learning_rate, foreach=False)
                                       if active else None)
                self.last_active.append(torch.zeros(latents, dtype=torch.int64, device=self.device))

    def initialize_upper(self, source_folder):
        """Copy approved source parameters one at a time into fresh upper modules."""
        if fallback_metadata(self.metadata) is None:
            raise ValueError("upper initialization requires fallback metadata")
        if any(optimizer.state for optimizer in self.optimizers if optimizer is not None):
            raise ValueError("upper initialization requires fresh Adam state")
        for index in self.active_layers:
            layer = self.layers[index]
            shapes = {name: tuple(parameter.shape) for name, parameter in layer.named_parameters()}
            with autorelease_scope(self.device), safe_open(
                Path(source_folder) / f"layer_{index:02d}.safetensors", framework="pt", device="cpu",
            ) as source:
                _layer_header(source, index, shapes, torch.float32)
                for name, parameter in layer.named_parameters():
                    weight = source.get_tensor(name)
                    if not _finite(weight):
                        raise ValueError("invalid warm source weight")
                    with torch.no_grad():
                        parameter.copy_(weight)
                    del weight
                    if not _finite(parameter):
                        raise ValueError("warm source weight overflows float32")
            self.initialized[index] = True

    def step_layer(self, index, x, y, tokens_before, *, health=None):
        """Accumulate microbatch gradients, then update one layer once."""
        if index not in self.active_layers:
            raise ValueError("cannot update an inactive lower layer")
        layer, optimizer = self.layers[index], self.optimizers[index]
        x, y = x.detach().reshape(-1, x.shape[-1]), y.detach().reshape(-1, y.shape[-1])
        if x.shape != y.shape or not x.numel():
            raise ValueError("MLP input and target shapes differ")
        if not self.initialized[index]:
            with torch.no_grad():
                layer.b_dec.copy_(y.mean(0))
            self.initialized[index] = True
        optimizer.zero_grad(set_to_none=True)
        variance = y.var(dim=0, unbiased=False).mean().clamp_min(1e-8)
        dead = tokens_before - self.last_active[index] >= self.settings["dead_tokens"]
        use_aux = self.settings["auxiliary_weight"] > 0 and bool(dead.any())
        active = torch.zeros_like(dead)
        mse = torch.zeros((), device=self.device)
        microbatch = self.settings["microbatch_tokens"]
        for start in range(0, len(x), microbatch):
            if health:
                health()
            with autorelease_scope(self.device):
                inputs, targets = x[start:start + microbatch], y[start:start + microbatch]
                pre = F.linear(inputs, layer.W_enc, layer.b_enc)
                values, indices = pre.relu().topk(layer.k, dim=-1)
                features = torch.zeros_like(pre).scatter(-1, indices, values)
                prediction = layer.decode(features)
                residual = targets - prediction
                loss = residual.square().mean() / variance
                mse += residual.detach().square().sum() / y.numel()
                active |= (features.detach() > 0).any(0)
                if use_aux:
                    aux_values, aux_indices = pre.masked_fill(~dead, -torch.inf).topk(min(2 * layer.k, pre.shape[-1]), dim=-1)
                    aux = torch.zeros_like(pre).scatter(-1, aux_indices, aux_values.relu())
                    aux_prediction = aux @ layer.W_dec
                    loss = loss + self.settings["auxiliary_weight"] * (residual.detach() - aux_prediction).square().mean() / variance
                (loss * (len(inputs) / len(x))).backward()
        # Decoder directions start normalized. Their magnitudes remain trainable:
        # TopK selection uses encoder scores, so rescaling encoders changes gates.
        with autorelease_scope(self.device):
            norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            with torch.no_grad():
                self.last_active[index].masked_fill_(active, tokens_before + len(x))
            optimizer.zero_grad(set_to_none=True)
            return {"fvu": float((mse / variance).item()), "gradient_norm": float(norm.item())}

    def set_learning_rate(self, fraction, *, decay_start=0.8, final_ratio=0.1):
        """Linearly reduce Adam's rate during the last part of the token budget."""
        decay = min(1.0, max(0.0, (fraction - decay_start) / (1 - decay_start)))
        rate = self.settings["learning_rate"] * (1 - decay * (1 - final_ratio))
        for optimizer in self.optimizers:
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = rate


def initial_progress():
    return {"tokens": 0, "steps": 0, "next_layer": 0, "pending_ids": None, "elapsed_seconds": 0.0}


def _save_reserve(progress):
    """Retain the admission floor and twice the observed final write costs."""
    costs = [progress.get(key, 0.0) for key in ("checkpoint_seconds", "export_seconds")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("invalid checkpoint or export reserve")
    return max(60.0, 2 * sum(costs))


def _validate_token_budget(metadata, token_budget, *, resume=False):
    """Keep the all-layer learning-rate schedule fixed across resume."""
    saved = metadata["training"].get("token_budget")
    if saved is None and not resume:
        return
    if type(saved) is not int or saved != token_budget:
        raise ValueError("resume token budget differs from the saved learning-rate schedule")


def train_online(model, trainer, stream, output, *, token_budget, progress=None, memory=None,
                 stop=None, save_seconds=240, max_wall_seconds=36_000, run_seconds=None,
                 started=None, clock=time.monotonic):
    """Train one model batch at a time and save partial layer progress on stop."""
    if not 0 < save_seconds <= 240 or not 0 < max_wall_seconds <= 36_000:
        raise ValueError("save interval is at most 240 seconds and total wall time is at most ten hours")
    if token_budget < 1 or (run_seconds is not None and run_seconds <= 0):
        raise ValueError("token and runtime budgets must be positive")
    fallback = fallback_metadata(trainer.metadata)
    if trainer.metadata.get("fallback_source") != trainer.fallback_source:
        raise ValueError("admitted fallback source changed across resume")
    prior_wall = 0.0
    if fallback:
        if (fallback != trainer.settings["fallback"] or token_budget != fallback["token_budget"]
                or max_wall_seconds != fallback["aggregate_wall_limit_seconds"]):
            raise ValueError("fallback token or aggregate time contract changed")
        if not all(trainer.initialized[i] for i in trainer.active_layers):
            raise ValueError("fallback requires warm upper weights or a restored checkpoint")
        prior_wall = fallback["prior_wall_seconds"]
    else:
        _validate_token_budget(trainer.metadata, token_budget, resume=progress is not None)
    output = Path(output)
    progress = initial_progress() if progress is None else progress
    start = clock() if started is None else started
    elapsed_before = progress["elapsed_seconds"]
    if fallback and (type(elapsed_before) not in (int, float)
                     or not math.isfinite(elapsed_before) or elapsed_before < 0):
        raise ValueError("invalid fallback elapsed time")
    elapsed = lambda: elapsed_before + clock() - start
    invocation_limit = min(max_wall_seconds - prior_wall - elapsed_before, run_seconds or max_wall_seconds)
    last_save = clock()
    checkpoint_seconds = progress.get("checkpoint_seconds", 0.0)
    if invocation_limit - (clock() - start) <= _save_reserve(progress):
        raise ValueError("wall time budget has no checkpoint reserve")
    last_metrics = {}
    model.requires_grad_(False).eval()
    export_stop = None

    def check_memory(stage, *, force=False):
        nonlocal export_stop
        if memory:
            try:
                memory.check(stage, force=force)
            except CleanStop as error:
                export_stop = stop_reason(error)
                raise

    def health():
        if stop:
            stop.check()
        if clock() - start >= invocation_limit - _save_reserve(progress):
            raise CleanStop("wall time reserve")
        if clock() - last_save >= 270:
            raise CleanStop("checkpoint deadline")
        check_memory("train")

    def save():
        nonlocal last_save, checkpoint_seconds
        for optimizer in trainer.optimizers:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        if memory:
            trainer.metadata["training"]["peak_memory_bytes"] = max(
                trainer.metadata["training"]["peak_memory_bytes"], memory.peak)
        before = clock()
        save_checkpoint(output / "checkpoints", trainer, stream, progress, elapsed=elapsed)
        completed = clock()
        duration = completed - before
        checkpoint_seconds = max(checkpoint_seconds, duration)
        progress["checkpoint_seconds"] = checkpoint_seconds
        last_save = completed
        print(json.dumps({"stage": "checkpoint", "tokens": progress["tokens"],
                          "next_layer": progress["next_layer"], "save_seconds": duration,
                          "max_save_seconds": checkpoint_seconds,
                          "elapsed_seconds": elapsed()}), flush=True)

    reason = "token_budget"
    trainer.metadata["training"]["token_budget"] = token_budget
    captured = None
    # An initial complete checkpoint also covers an unsupported model operator.
    if not (output / "checkpoints" / "latest.json").exists():
        save()
    try:
        while progress["tokens"] < token_budget:
            health()
            if progress["pending_ids"] is None:
                requested = min(trainer.settings["batch_tokens"], token_budget - progress["tokens"])
                batch = stream.next_tokens(requested, health=health)
                if not len(batch):
                    raise CleanStop("text source exhausted")
                progress["pending_ids"] = batch.input_ids.tolist()
            trainer.set_learning_rate(progress["tokens"] / token_budget)
            count = len(progress["pending_ids"])
            context = trainer.settings["context"]
            ids = torch.tensor(progress["pending_ids"], dtype=torch.int64, device=trainer.device)
            padded = F.pad(ids, (0, (-len(ids)) % context)).reshape(-1, context)
            requested = [trainer.metadata[field][i] for field in ("read_points", "write_points")
                         for i in trainer.active_layers]
            with autorelease_scope(trainer.device), capture(model, requested) as captured, torch.no_grad():
                hidden_forward(model, padded)
            check_memory("after_forward")
            for index in trainer.active_layers:
                if index < progress["next_layer"]:
                    continue
                health()
                read, write = trainer.metadata["read_points"][index], trainer.metadata["write_points"][index]
                x = captured[read].reshape(-1, trainer.metadata["width"])[:count]
                y = captured[write].reshape(-1, trainer.metadata["width"])[:count]
                metrics = trainer.step_layer(index, x, y, progress["tokens"], health=health)
                progress["next_layer"] = index + 1
                last_metrics[str(index)] = metrics
                del x, y
                captured.pop(read)
                captured.pop(write)
                if clock() - last_save >= save_seconds:
                    health()
                    save()
            captured.close(clear=True)
            progress["tokens"] += count
            progress["steps"] += 1
            progress["next_layer"] = 0
            progress["pending_ids"] = None
            rate = progress["tokens"] / max(elapsed(), 1e-9)
            print(json.dumps({"stage": "train_step", "tokens": progress["tokens"], "step": progress["steps"],
                              "elapsed_seconds": elapsed(), "tokens_per_second": rate,
                              "projected_seconds": token_budget / max(rate, 1e-9), "layers": last_metrics}), flush=True)
    except CleanStop as error:
        reason = stop_reason(error)
    finally:
        if captured is not None:
            captured.close(clear=True)
    # Other exceptions retain the prior complete checkpoint and propagate.
    trainer.metadata["training"]["stop_reason"] = reason
    trainer.metadata["training"]["last_batch_metrics"] = last_metrics
    trainer.metadata["training"]["partial_batch_tokens"] = len(progress["pending_ids"] or [])
    trainer.metadata["training"]["next_layer"] = progress["next_layer"]
    if fallback:
        trainer.metadata["training"]["status"] = (
            "fallback_complete" if progress["tokens"] == token_budget else "fallback_incomplete"
        )
    save()
    progress["elapsed_seconds"] = elapsed()
    if export_stop is None:
        try:
            check_memory("before_dictionary_export", force=True)
        except CleanStop:
            pass
    # The resumable checkpoint is essential. A dictionary copy can wait for recovery.
    if export_stop is None:
        before_export = clock()
        export_dictionary(output, trainer, progress)
        trainer.metadata["training"]["elapsed_seconds"] = elapsed()
        write_json(output / "dictionary.json", trainer.metadata)
        progress["export_seconds"] = max(progress.get("export_seconds", 0.0), clock() - before_export)
    progress["dictionary_export"] = "complete" if export_stop is None else "skipped"
    progress["elapsed_seconds"] = elapsed()
    generation = json.loads((output / "checkpoints" / "latest.json").read_text())["generation"]
    write_json(output / "runtime.json", {"generation": generation, "elapsed_seconds": elapsed(),
                                         "checkpoint_seconds": checkpoint_seconds,
                                         "export_seconds": progress.get("export_seconds", 0.0),
                                         "dictionary_export": progress["dictionary_export"],
                                         "export_stop_reason": export_stop})
    print(json.dumps({"stage": "train_end", "reason": reason, "tokens": progress["tokens"],
                      "dictionary_export": progress["dictionary_export"], "export_stop_reason": export_stop,
                      "status": "complete" if progress["tokens"] == token_budget else "interrupted",
                      **({"status": trainer.metadata["training"]["status"],
                          "aggregate_elapsed_seconds": prior_wall + elapsed()} if fallback else {}),
                      "elapsed_seconds": elapsed()}), flush=True)
    return progress


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("mps", "cpu"), default="mps")
    result.add_argument("--latents", type=int, default=16_384)
    result.add_argument("--k", type=int, default=32)
    result.add_argument("--tokens", type=int, default=40_000_000,
                        help="Total tokens for the learning-rate schedule. Resume requires the saved value.")
    result.add_argument("--context", type=int, default=256)
    result.add_argument("--batch-tokens", type=int, default=4096)
    result.add_argument("--microbatch-tokens", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--auxiliary-weight", type=float, default=1 / 32)
    result.add_argument("--dead-tokens", type=int, default=100_000)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--save-seconds", type=float, default=240)
    result.add_argument("--max-wall-seconds", type=float, default=36_000)
    result.add_argument("--run-seconds", type=float)
    result.add_argument("--memory-gb", type=float, default=24)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--upper-half-from", type=Path)
    result.add_argument("--prior-wall-seconds", type=float)
    result.add_argument("--dataset-cache", type=Path)
    result.add_argument("--local-dataset-manifest", type=Path)
    return result


_MODEL_FIELDS = ("model_id", "model_hash", "revision", "layers", "width", "latents", "activation", "k",
                 "read_points", "write_points")
_TEXT_FIELDS = ("dataset_id", "dataset_config", "dataset_revision", "split", "partitions", "source_id")


def _startup_deadline(args, progress, started):
    """Reserve final writes within the aggregate and invocation time limits."""
    elapsed = progress.get("elapsed_seconds", 0)
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed time or checkpoint reserve")
    reserve = _save_reserve(progress)
    deadline = started + min(args.max_wall_seconds - (args.prior_wall_seconds or 0) - elapsed,
                             args.run_seconds or args.max_wall_seconds) - reserve
    if time.monotonic() >= deadline:
        raise ValueError("aggregate wall time has no model-load and checkpoint reserve")
    return deadline


def _source_quality_state(folder, metadata, identity):
    """Bind failed quality to the exact dictionary in the completed evaluation."""
    try:
        state = json.loads((folder / "evaluation" / "state.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("fallback requires completed source evaluation state") from error
    stream, settings = state.get("stream", {}), state.get("settings", {})
    if (state.get("version") != 1 or state.get("count") != 200_000
            or "pending_ids" not in state or state["pending_ids"] is not None
            or state.get("identity") != identity or settings.get("tokens") != 200_000
            or settings.get("scope", "all_layers") != "all_layers"
            or settings.get("active_layers", list(range(metadata["layers"]))) != list(range(metadata["layers"]))
            or stream.get("partition") != "quality"
            or stream.get("source_id") != metadata["training"]["source_id"]
            or stream.get("partitions") != metadata["training"]["partitions"]
            or metadata["quality"].get("dictionary_hash", identity) != identity):
        raise ValueError("source held-out evaluation identity or completed cursor differs")


def prepare_upper_half(args, *, source_id, record=None):
    """Validate a completed source and bind the fallback before model allocation."""
    source_folder, output = args.upper_half_from.resolve(), args.output.resolve()
    if source_folder.is_relative_to(output) or output.is_relative_to(source_folder):
        raise ValueError("fallback output and source folders must be separate")
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("fallback output is not empty; use --resume or a new output folder")
    source = _manifest(json.loads((source_folder / "dictionary.json").read_text()))
    source_training, quality = source["training"], source["quality"]
    if fallback_metadata(source) is not None or "fallback_source" in source or source["layers"] < 2:
        raise ValueError("fallback requires an all-layer source with at least two layers")
    if (source["model_id"] != MODEL_ID or source["revision"] != MODEL_REVISION
            or source["latents"] != args.latents or source["k"] != args.k):
        raise ValueError("fallback source model or dimensions differ")
    expected_text = {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                     "dataset_revision": DATASET_REVISION, "split": "train",
                     "partitions": partition_metadata(), "source_id": source_id}
    if any(source_training.get(key) != value for key, value in expected_text.items()):
        raise ValueError("fallback text source differs")
    if (quality.get("status") != "failed" or quality["evaluated_tokens"] != 200_000
            or quality.get("partition") != "quality" or quality.get("stop_reason") != "complete"
            or quality.get("all_mlps_replaced") != source["layers"]
            or any(quality[key] is None for key in ("fvu", "dead_fraction", "cross_entropy_delta"))
            or not (quality["fvu"] > 0.40 or quality["dead_fraction"] > 0.15
                    or quality["cross_entropy_delta"] > 0.5)):
        raise ValueError("fallback requires completed failed held-out quality on 200000 tokens")
    generation, source_record = checkpoint_record(source_folder / "checkpoints")
    progress = source_record["progress"]
    budget = source_training.get("token_budget")
    if (type(budget) is not int or budget < 1 or source_training["tokens"] != budget
            or progress["tokens"] != budget or progress["pending_ids"] is not None
            or progress["next_layer"] != 0 or source_training.get("stop_reason") != "token_budget"
            or source_training.get("partial_batch_tokens") != 0 or source_training.get("next_layer") != 0):
        raise ValueError("fallback source has not completed its full token budget")
    saved_metadata = _manifest(source_record["metadata"])
    if (any(source[key] != saved_metadata[key] for key in _MODEL_FIELDS)
            or any(source_training[key] != saved_metadata["training"].get(key) for key in _TEXT_FIELDS)
            or source_record["stream"]["source_id"] != source_id
            or fallback_metadata(saved_metadata) is not None or "fallback_source" in saved_metadata):
        raise ValueError("fallback source checkpoint identity differs")
    elapsed_values = [source_training["elapsed_seconds"], progress["elapsed_seconds"],
                      saved_metadata["training"]["elapsed_seconds"]]
    runtime = source_folder / "runtime.json"
    if runtime.exists():
        saved_runtime = json.loads(runtime.read_text())
        if saved_runtime["generation"] != generation.name:
            raise ValueError("fallback source runtime generation differs")
        elapsed_values.append(saved_runtime["elapsed_seconds"])
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in elapsed_values):
        raise ValueError("invalid fallback source elapsed time")
    source_elapsed = max(elapsed_values)
    if args.prior_wall_seconds < source_elapsed:
        raise ValueError("prior wall seconds are below the source elapsed time")
    shapes = {"W_enc": (source["latents"], source["width"]), "b_enc": (source["latents"],),
              "W_dec": (source["latents"], source["width"]), "b_dec": (source["width"],)}
    for i in range(source["layers"]):
        with safe_open(source_folder / f"layer_{i:02d}.safetensors", framework="pt", device="cpu") as tensors:
            _layer_header(tensors, i, shapes, torch.float32)
    source_hash = dictionary_fingerprint(source_folder, source)
    _source_quality_state(source_folder, source, source_hash)
    metadata = copy.deepcopy(source)
    metadata["fallback_source"] = {
        "metadata": copy.deepcopy(source),
        "identity": {"model_hash": source["model_hash"], "dictionary_fingerprint": source_hash},
        "evaluation": {
            "version": 1, "count": 200_000, "pending_ids": None, "identity": source_hash,
            "settings": {"tokens": 200_000, "scope": "all_layers", "active_layers": list(range(source["layers"]))},
            "stream": {"partition": "quality", "source_id": source_training["source_id"],
                       "partitions": copy.deepcopy(source_training["partitions"])},
        },
    }
    metadata["training"] = {**expected_text, "tokens": 0, "elapsed_seconds": 0,
                            "peak_memory_bytes": 0, "seed": args.seed, "fallback": {
        "version": 1, "mode": "upper_half", "active_layers": list(range(source["layers"] // 2, source["layers"])),
        "zero_layers": list(range(source["layers"] // 2)),
        "source_dictionary_hash": source_hash,
        "source_checkpoint_generation": generation.name, "source_training_tokens": budget,
        "source_elapsed_seconds": source_elapsed, "prior_wall_seconds": args.prior_wall_seconds,
        "aggregate_wall_limit_seconds": args.max_wall_seconds, "token_budget": args.tokens,
        "warm_start": "upper_weights_fresh_adam", "source_id": source_id,
        "lower_behavior": "zero_dictionary_original_mlp",
    }}
    metadata["quality"] = {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                           "evaluated_tokens": 0, "status": "not_evaluated"}
    _manifest(metadata)
    # Include the service metadata and its source projection at response depth.
    if not bounded({"result": {"metadata": metadata, "fallback_source": metadata["fallback_source"]}}):
        raise ValueError("admitted fallback source exceeds response limits")
    if record is not None:
        if record["metadata"].get("fallback_source") != metadata["fallback_source"]:
            raise ValueError("checkpoint admitted fallback source differs")
        validate_checkpoint(record, trainer_settings(metadata, **_trainer_arguments(args)), metadata, source_id)
    return metadata


def _trainer_arguments(args):
    return {key: getattr(args, key) for key in ("learning_rate", "microbatch_tokens", "auxiliary_weight",
            "dead_tokens", "context", "batch_tokens", "upper_half_from")}


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.monotonic()
    if (not 1 <= args.latents <= 16_384 or not 1 <= args.k <= args.latents
            or not 2 <= args.context <= 256 or not args.context <= args.batch_tokens <= 4096
            or not 1 <= args.microbatch_tokens <= 256):
        raise ValueError("requested dimensions exceed the bounded trainer allocations")
    if (not 0 < args.save_seconds <= 240 or not 0 < args.max_wall_seconds <= 36_000
            or args.tokens < 1 or (args.run_seconds is not None and not 0 < args.run_seconds <= 36_000)):
        raise ValueError("invalid token, checkpoint, or wall-time budget")
    if (not 0 < args.learning_rate < 1 or not 0 <= args.auxiliary_weight <= 1
            or args.dead_tokens < 1 or not 0 < args.memory_gb <= 24):
        raise ValueError("invalid trainer settings or memory limit")
    if (args.upper_half_from is None) != (args.prior_wall_seconds is None):
        raise ValueError("--upper-half-from and --prior-wall-seconds are required together")
    fallback = args.upper_half_from is not None
    if fallback and (not math.isfinite(args.prior_wall_seconds) or args.prior_wall_seconds < 0):
        raise ValueError("prior wall seconds must be finite and nonnegative")
    if not args.resume and args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("output is not empty; use --resume or a new output folder")
    try:
        with StopControl() as stop, ExitStack() as resources:
            stop.check()
            return _run(args, started, stop, resources)
    except CleanStop as error:
        print(json.dumps(clean_stop_record(error)), flush=True)
        return 2


def _run(args, started, stop, resources):
    """Admit startup before any checkpoint can replace a complete generation."""
    stop.check()
    record = checkpoint_record(args.output / "checkpoints")[1] if args.resume else None
    stop.check()
    fallback = args.upper_half_from is not None
    if record is not None and bool(fallback_metadata(record["metadata"])) != fallback:
        raise ValueError("resume requires the original all-layer or upper-half mode")
    deadline = _startup_deadline(args, record["progress"] if record else {}, started)
    if record is not None and not fallback:
        _validate_token_budget(record["metadata"], args.tokens, resume=True)
        requested = {**record["metadata"], "latents": args.latents, "k": args.k}
        validate_checkpoint(record, trainer_settings(requested, **_trainer_arguments(args)),
                            requested, record["stream"]["source_id"])
    memory = MemoryGuard(args.memory_gb, args.device)

    def health(stage="source admission", *, force=False):
        stop.check()
        if time.monotonic() >= deadline:
            raise CleanStop("wall time reserve")
        memory.check(stage, force=force)
        stop.check()
        if time.monotonic() >= deadline:
            raise CleanStop("wall time reserve")

    health("start", force=True)
    stream, prepared = None, None
    if fallback:
        # Source identity is established before the model, with no token consumption.
        stream = _training_stream(args, None, resources, health=health, memory=memory)
        health()
        prepared = prepare_upper_half(args, source_id=stream.source_id, record=record)
        health()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    health("before model")
    model, tokenizer = load_text_model(device=args.device, memory=memory)
    health("after model", force=True)
    metadata = make_manifest(model, args.latents, args.k)
    if prepared is not None:
        if any(metadata[key] != prepared[key] for key in _MODEL_FIELDS):
            raise ValueError("fallback source differs from the loaded model")
        metadata = prepared
    metadata["training"]["seed"] = args.seed
    health("before dictionary")
    trainer = Trainer(metadata, device=args.device, **_trainer_arguments(args))
    health("after_dictionary_init", force=True)
    if fallback and not args.resume:
        trainer.initialize_upper(args.upper_half_from)
        health("after warm start", force=True)
    if stream is None:
        stream = _training_stream(args, tokenizer, resources, health=health, memory=memory)
    else:
        stream.tokenizer = tokenizer
    health()
    trainer.metadata["training"]["source_id"] = stream.source_id
    progress = restore_checkpoint(args.output / "checkpoints", trainer, stream) if args.resume else None
    if trainer.metadata.get("fallback_source") != trainer.fallback_source:
        raise ValueError("admitted fallback source changed across resume")
    health("after_resume", force=True)
    result = train_online(model, trainer, stream, args.output, token_budget=args.tokens, progress=progress,
                          memory=memory, stop=stop, save_seconds=args.save_seconds,
                          max_wall_seconds=args.max_wall_seconds, run_seconds=args.run_seconds, started=started)
    return 0 if result["tokens"] == args.tokens and result["dictionary_export"] == "complete" else 2


def _training_stream(args, tokenizer, resources, *, health=None, memory=None):
    source = fineweb_documents(cache_dir=args.dataset_cache, local_manifest=args.local_dataset_manifest,
                               health=health, memory=memory)
    if hasattr(source, "close"):
        resources.callback(source.close)
    return resources.enter_context(TextStream(source, tokenizer, "train"))


if __name__ == "__main__":
    raise SystemExit(main())
