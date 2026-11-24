"""Held-out FVU, dead features, and cross-entropy for the active dictionary scope."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from caliper.interp import capture, load_dictionary
from caliper.interp._dictionary import fallback_metadata
from .checkpoint_state import _save_tensors, dictionary_fingerprint, write_json
from .evaluation_binding import EvaluationBinding
from .memory_limits import CleanStop, MemoryGuard, StopControl, autorelease_scope
from .model_load import hidden_forward, load_text_model
from .stop_records import clean_stop_record, stop_reason
from .text_stream import TextStream, fineweb_documents, partition_metadata


@contextmanager
def replace_all_mlps(model, dictionary):
    """Replace every active MLP for one forward and restore it on every exit."""
    originals = []
    try:
        if len(dictionary.layers) != model.config.num_hidden_layers:
            raise ValueError("cross-entropy requires a replacement for every MLP")
        fallback = fallback_metadata(dictionary.metadata)
        active = fallback["active_layers"] if fallback else range(len(dictionary.layers))
        for i in active:
            point = dictionary.metadata["read_points"][i]
            block = model.get_submodule(point.rsplit(".", 1)[0])
            originals.append((block, block.mlp))
            block.mlp = dictionary.layers[i]
        yield
    finally:
        for block, original in originals:
            block.mlp = original


@torch.no_grad()
def cross_entropy_sum(model, hidden, input_ids, count, *, chunk_tokens=64, health=None):
    """Project a bounded token chunk. Exclude padding and context boundaries."""
    width = hidden.shape[-1]
    context = input_ids.shape[-1]
    positions = torch.arange(max(count - 1, 0), device=hidden.device)
    positions = positions[(positions + 1) % context != 0]
    head = model.get_output_embeddings()
    states = hidden.reshape(-1, width)
    ids = input_ids.reshape(-1)
    total = 0.0
    for start in range(0, len(positions), chunk_tokens):
        if health:
            health()
        selected = positions[start:start + chunk_tokens]
        logits = head(states[selected]).float()
        loss = F.cross_entropy(logits, ids[selected + 1], reduction="sum")
        total += float(loss.item())
    return total, len(positions)


class QualityStats:
    """Merge stable per-batch variance statistics and feature activity counts."""

    def __init__(self, layers, width, latents, *, active_layers=None):
        self.scope = "all_layers" if active_layers is None else "upper_half"
        self.active_layers = tuple(range(layers) if active_layers is None else active_layers)
        self.count = 0
        self.mean = torch.zeros(layers, width, dtype=torch.float64)
        self.variance_sum = torch.zeros(layers, dtype=torch.float64)
        self.error_sum = torch.zeros(layers, dtype=torch.float64)
        self.hits = torch.zeros(layers, latents, dtype=torch.int64)
        self.original_ce = 0.0
        self.replaced_ce = 0.0
        self.ce_tokens = 0

    def merge(self, other):
        if self.scope != other.scope or self.active_layers != other.active_layers:
            raise ValueError("quality scopes differ")
        total = self.count + other.count
        if other.count == 0:
            return
        delta = other.mean - self.mean
        self.variance_sum += other.variance_sum + delta.square().sum(-1) * (self.count * other.count / total)
        self.mean += delta * (other.count / total)
        self.error_sum += other.error_sum
        self.hits += other.hits
        self.count = total
        self.original_ce += other.original_ce
        self.replaced_ce += other.replaced_ce
        self.ce_tokens += other.ce_tokens

    def metrics(self):
        active = list(self.active_layers)
        if not self.count or not self.ce_tokens or (self.variance_sum[active] <= 0).any():
            raise ValueError("quality needs nonzero target variance and next-token targets")
        fvu = self.error_sum[active] / self.variance_sum[active]
        dead = (self.hits[active] == 0).double().mean(-1)
        delta = (self.replaced_ce - self.original_ce) / self.ce_tokens
        values = {"fvu": float(fvu.mean()), "dead_fraction": float(dead.mean()),
                  "cross_entropy_delta": delta, "evaluated_tokens": self.count,
                  "cross_entropy_tokens": self.ce_tokens,
                  "original_cross_entropy": self.original_ce / self.ce_tokens,
                  "replaced_cross_entropy": self.replaced_ce / self.ce_tokens,
                  "per_layer_fvu": fvu.tolist(), "per_layer_dead_fraction": dead.tolist(),
                  "thresholds": {"fvu": 0.40, "dead_fraction": 0.15, "cross_entropy_delta": 0.5,
                                 "evaluated_tokens": 200_000}}
        finite = all(torch.isfinite(x).all().item() for x in (fvu, dead)) and math.isfinite(delta)
        if not finite:
            raise ValueError("quality metrics are nonfinite")
        passed = values["fvu"] <= 0.40 and values["dead_fraction"] <= 0.15 and delta <= 0.5
        values["status"] = ("passed" if passed else "failed") if self.count == 200_000 else "partial"
        if self.scope == "upper_half":
            values["status"] = "fallback_" + values["status"]
            values.update(self.scope_fields())
            for key, tensor in (("per_layer_fvu", fvu), ("per_layer_dead_fraction", dead)):
                values[key] = [None] * len(self.hits)
                for index, value in zip(active, tensor.tolist()):
                    values[key][index] = value
        return values

    def scope_fields(self):
        return {"scope": self.scope, "active_layers": list(self.active_layers),
                "per_layer_status": ["active" if i in self.active_layers else "error_only"
                                     for i in range(len(self.hits))]}


def quality_stats(metadata):
    fallback = fallback_metadata(metadata)
    return QualityStats(metadata["layers"], metadata["width"], metadata["latents"],
                        active_layers=fallback["active_layers"] if fallback else None)


@torch.no_grad()
def evaluate_batch(model, dictionary, ids, count, *, chunk_tokens=64, health=None):
    """Commit no statistics until both complete model forwards succeed."""
    with autorelease_scope(ids.device):
        return _evaluate_batch(model, dictionary, ids, count, chunk_tokens=chunk_tokens, health=health)


def _evaluate_batch(model, dictionary, ids, count, *, chunk_tokens, health):
    metadata = dictionary.metadata
    result = quality_stats(metadata)
    requested = [metadata[field][i] for field in ("read_points", "write_points") for i in result.active_layers]
    captured = None
    try:
        with capture(model, requested) as captured:
            original = hidden_forward(model, ids)
        result.original_ce, result.ce_tokens = cross_entropy_sum(model, original, ids, count,
                                                                chunk_tokens=chunk_tokens, health=health)
        del original
        for i in result.active_layers:
            layer = dictionary.layers[i]
            if health:
                health()
            x = captured[metadata["read_points"][i]].reshape(-1, metadata["width"])[:count]
            y = captured[metadata["write_points"][i]].reshape(-1, metadata["width"])[:count]
            mean = y.mean(0)
            result.mean[i] = mean.cpu().double()
            result.variance_sum[i] = (y - mean).square().sum().cpu().double()
            # Chunk sparse features as well as vocabulary logits.
            for start in range(0, count, chunk_tokens):
                a = layer.encode(x[start:start + chunk_tokens])
                residual = y[start:start + chunk_tokens] - layer.decode(a)
                result.error_sum[i] += residual.square().sum().cpu().double()
                result.hits[i] += (a > 0).sum(0).cpu()
            del x, y
            captured.pop(metadata["read_points"][i])
            captured.pop(metadata["write_points"][i])
        captured.close(clear=True)
        with replace_all_mlps(model, dictionary):
            replaced = hidden_forward(model, ids)
        result.replaced_ce, replaced_count = cross_entropy_sum(model, replaced, ids, count,
                                                              chunk_tokens=chunk_tokens, health=health)
        if replaced_count != result.ce_tokens:
            raise ValueError("cross-entropy target counts differ")
        result.count = count
        return result
    finally:
        if captured is not None:
            captured.close(clear=True)


def save_evaluation(folder, stats, stream, pending, identity, settings, elapsed, binding):
    """Atomically publish small statistics and an exact quality cursor."""
    stream_state = stream.state_dict()
    binding.verify()
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    # Two alternating files prevent a partial update from corrupting the old state.
    prior = json.loads((folder / "state.json").read_text()) if (folder / "state.json").exists() else {}
    name = "stats-b.safetensors" if prior.get("tensors") == "stats-a.safetensors" else "stats-a.safetensors"
    _save_tensors({"mean": stats.mean, "variance_sum": stats.variance_sum,
                   "error_sum": stats.error_sum, "hits": stats.hits}, folder / name)
    binding.verify()
    write_json(folder / "state.json", {"version": 1, "identity": identity, "settings": settings,
               "tensor_binding": binding.evidence, "model_binding": binding.model_evidence,
               "tensors": name, "stream": stream_state, "pending_ids": pending,
               "count": stats.count, "original_ce": stats.original_ce,
               "replaced_ce": stats.replaced_ce, "ce_tokens": stats.ce_tokens, "elapsed_seconds": elapsed})


def restore_evaluation(folder, stats, stream, identity, settings, binding):
    folder = Path(folder)
    state = json.loads((folder / "state.json").read_text())
    if (state.get("version") != 1 or state["identity"] != identity or state["settings"] != settings
            or state.get("tensor_binding") != binding.evidence):
        raise ValueError("evaluation dictionary or settings changed across resume")
    if state.get("model_binding") != binding.model_evidence:
        raise ValueError("evaluation model identity mismatch: cursor binding differs")
    binding.verify()
    if state["tensors"] not in ("stats-a.safetensors", "stats-b.safetensors"):
        raise ValueError("invalid evaluation tensor path")
    tensors = load_file(folder / state["tensors"])
    for name in ("mean", "variance_sum", "error_sum", "hits"):
        getattr(stats, name).copy_(tensors[name])
    for name in ("count", "original_ce", "replaced_ce", "ce_tokens"):
        setattr(stats, name, state[name])
    stream.load_state_dict(state["stream"])
    return state["pending_ids"], state["elapsed_seconds"]


def evaluate(model, dictionary, stream, folder, *, tokens=200_000, context=256,
             batch_tokens=512, chunk_tokens=64, memory=None, stop=None, resume=False):
    if (not 2 <= tokens <= 200_000 or not 2 <= context <= 256
            or not context <= batch_tokens <= 4096 or not 1 <= chunk_tokens <= 128):
        raise ValueError("invalid evaluation dimensions")
    if stream.partition != "quality":
        raise ValueError("quality evaluation requires the quality partition")
    metadata = dictionary.metadata
    trained_source = metadata["training"].get("source_id")
    if trained_source is not None and trained_source != stream.source_id:
        raise ValueError("quality source differs from the training source")
    if metadata["training"]["dataset_id"] != "fixture" and metadata["training"]["partitions"] != partition_metadata():
        raise ValueError("dictionary text partitions differ from this evaluator")
    folder = Path(folder)
    identity = dictionary_fingerprint(folder, metadata, memory=memory)
    binding = EvaluationBinding(folder, dictionary, identity, model=model, memory=memory)
    state_folder = folder / "evaluation"
    stats = quality_stats(metadata)
    settings = {"tokens": tokens, "context": context, "batch_tokens": batch_tokens, "chunk_tokens": chunk_tokens}
    if stats.scope == "upper_half":
        settings.update(scope=stats.scope, active_layers=list(stats.active_layers))
    pending, elapsed_before = None, 0.0
    if not resume and (state_folder / "state.json").exists():
        raise ValueError("evaluation state exists; use --resume")
    started = time.monotonic()
    last_save = started
    device = dictionary.layers[0].W_enc.device

    def health():
        if stop:
            stop.check()
        if memory:
            memory.check("evaluation", force=False)

    reason = "complete"
    ready = False
    # A rejected or interrupted restore must preserve the existing quality report.
    if resume:
        health()
        pending, elapsed_before = restore_evaluation(state_folder, stats, stream, identity, settings, binding)
        ready = True
    # Admission ends after artifact binding. Earlier startup stops preserve quality.
    # A stop in this block records partial quality for the admitted artifact.
    try:
        if not ready:
            health()
            ready = True
        while stats.count < tokens:
            health()
            binding.check()
            if pending is None:
                batch = stream.next_tokens(min(batch_tokens, tokens - stats.count), health=health)
                if not len(batch):
                    raise CleanStop("quality source exhausted")
                pending = batch.input_ids.tolist()
            count = len(pending)
            ids = torch.tensor(pending, dtype=torch.int64, device=device)
            ids = F.pad(ids, (0, (-count) % context)).reshape(-1, context)
            binding.check()
            result = evaluate_batch(model, dictionary, ids, count, chunk_tokens=chunk_tokens, health=health)
            binding.check()
            stats.merge(result)
            pending = None
            if time.monotonic() - last_save >= 240:
                save_evaluation(state_folder, stats, stream, pending, identity, settings,
                                elapsed_before + time.monotonic() - started, binding)
                last_save = time.monotonic()
            print(json.dumps({"stage": "quality_batch", "tokens": stats.count, "target": tokens}), flush=True)
    except CleanStop as error:
        reason = stop_reason(error)
    if stats.count >= 2 and stats.ce_tokens:
        quality = stats.metrics()
    else:
        quality = {"status": "fallback_partial" if stats.scope == "upper_half" else "partial",
                   "evaluated_tokens": stats.count, "cross_entropy_tokens": stats.ce_tokens,
                   "fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                   "per_layer_fvu": [None] * len(dictionary.layers),
                   "per_layer_dead_fraction": [None] * len(dictionary.layers),
                   **stats.scope_fields()}
    quality.update(elapsed_seconds=elapsed_before + time.monotonic() - started, stop_reason=reason,
                   dictionary_hash=identity, model_binding=binding.model_evidence, partition="quality")
    if stats.scope == "upper_half":
        quality.update(replaced_mlps=len(stats.active_layers))
    else:
        quality.update(all_mlps_replaced=len(dictionary.layers))
    if memory:
        quality["peak_memory_bytes"] = memory.peak
    if ready:
        save_evaluation(state_folder, stats, stream, pending, identity, settings,
                        elapsed_before + time.monotonic() - started, binding)
    binding.verify()
    metadata["quality"] = quality
    write_json(folder / "dictionary.json", metadata)
    print(json.dumps({"stage": "quality_end", **quality}), flush=True)
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--tokens", type=int, default=200_000)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--memory-gb", type=float, default=12)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--local-dataset-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 2 <= args.tokens <= 200_000 or not 2 <= args.context <= 256
            or not args.context <= args.batch_tokens <= 4096 or not 1 <= args.chunk_tokens <= 128):
        raise ValueError("requested dimensions exceed the bounded evaluator allocations")
    if not 0 < args.memory_gb <= 12:
        raise ValueError("the evaluator memory limit is at most 12 GB")
    try:
        with StopControl() as stop, ExitStack() as resources:
            stop.check()
            memory = MemoryGuard(args.memory_gb, args.device)
            memory.check("start")
            stop.check()
            model, tokenizer = load_text_model(device=args.device, memory=memory)
            stop.check()
            dictionary = load_dictionary(args.dictionary, model=model, device=args.device, dtype=torch.float32,
                                         memory_limit_gb=args.memory_gb)
            stop.check()
            memory.check("after_dictionary_load")
            stop.check()
            source = fineweb_documents(cache_dir=args.dataset_cache, local_manifest=args.local_dataset_manifest)
            if hasattr(source, "close"):
                resources.callback(source.close)
            stream = resources.enter_context(TextStream(source, tokenizer, "quality"))
            stop.check()
            result = evaluate(model, dictionary, stream, args.dictionary, tokens=args.tokens,
                              context=args.context, batch_tokens=args.batch_tokens, chunk_tokens=args.chunk_tokens,
                              memory=memory, stop=stop, resume=args.resume)
    except CleanStop as error:
        print(json.dumps(clean_stop_record(error)), flush=True)
        return 2
    return 0 if result["status"] in ("passed", "fallback_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
