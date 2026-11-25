"""Held-out FVU, dead features, and cross-entropy for the active dictionary scope."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from caliper.interp import capture, load_dictionary
from caliper.interp._context import context_policy, prepare_contexts, source_segments
from caliper.interp._dictionary import fallback_metadata, _canonical_device
from caliper.interp._memory import BudgetMemory
from .checkpoint_state import _save_tensors, dictionary_fingerprint, write_json
from .evaluation_binding import EvaluationBinding
from .memory_limits import CleanStop, MemoryGuard, StopControl, autorelease_scope
from .model_load import GEMMA_ID, MODEL_ID, hidden_forward, load_text_model
from .stop_records import clean_stop_record, stop_reason
from .text_stream import TextStream, evaluation_source_metadata, fineweb_documents, partition_metadata


_QUALITY_THRESHOLDS = {"fvu": 0.40, "dead_fraction": 0.15, "cross_entropy_delta": 0.5,
                       "evaluated_tokens": 200_000}


@contextmanager
def replace_all_mlps(model, dictionary, *, content_mask=None):
    """Replace active MLPs and restore them on exit. Gemma masks preserve unselected writes."""
    if content_mask is not None:
        if getattr(model.config, "model_type", None) != "gemma2":
            raise ValueError("legacy replacement content masks require Gemma")
        if not isinstance(content_mask, torch.Tensor) or content_mask.dtype != torch.bool or content_mask.ndim != 2:
            raise ValueError("invalid replacement content mask")
    originals = []
    handles, reads = [], {}
    try:
        if len(dictionary.layers) != model.config.num_hidden_layers:
            raise ValueError("cross-entropy requires a replacement for every MLP")
        fallback = fallback_metadata(dictionary.metadata)
        active = fallback["active_layers"] if fallback else range(len(dictionary.layers))
        if getattr(model.config, "model_type", None) == "gemma2":
            from caliper.interp._points import _architecture, _input
            probes = _architecture(model)
            for i in active:
                read = probes[dictionary.metadata["read_points"][i]]
                write = probes[dictionary.metadata["write_points"][i]]
                if read.event != "input" or write.event != "output":
                    raise ValueError("Gemma replacement requires semantic input and output probes")

                def remember(module, args, kwargs, output, index=i):
                    reads[index] = _input(args, kwargs)

                def replace(module, args, output, index=i):
                    # The dictionary decoder already writes after the post norm.
                    value = dictionary.layers[index](reads.pop(index))
                    if value.shape != output.shape:
                        raise ValueError("Gemma replacement shape differs from the semantic write")
                    value = value.to(dtype=output.dtype, device=output.device)
                    if content_mask is not None:
                        if content_mask.shape != output.shape[:-1] or content_mask.device != output.device:
                            raise ValueError("replacement content mask differs from the model context")
                        value = torch.where(content_mask[..., None], value, output)
                    return value

                handles.append(model.get_submodule(read.module).register_forward_hook(remember, with_kwargs=True))
                handles.append(model.get_submodule(write.module).register_forward_hook(replace))
            yield
            return
        for i in active:
            point = dictionary.metadata["read_points"][i]
            block = model.get_submodule(point.rsplit(".", 1)[0])
            originals.append((block, block.mlp))
            block.mlp = dictionary.layers[i]
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        reads.clear()
        for block, original in originals:
            block.mlp = original


@torch.no_grad()
def cross_entropy_sum(model, hidden, input_ids, count, *, chunk_tokens=64, health=None,
                      prediction_mask=None):
    """Sum target loss in head chunks. An explicit mask selects predictors within each row."""
    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 128:
        raise ValueError("invalid cross-entropy chunk size")
    width = hidden.shape[-1]
    context = input_ids.shape[-1]
    if prediction_mask is None:
        positions = torch.arange(max(count - 1, 0), device=hidden.device)
        positions = positions[(positions + 1) % context != 0]
    else:
        if (not isinstance(prediction_mask, torch.Tensor) or prediction_mask.dtype != torch.bool
                or prediction_mask.shape != input_ids[:, :-1].shape
                or prediction_mask.device != hidden.device or input_ids.device != hidden.device):
            raise ValueError("invalid cross-entropy prediction mask")
        positions = torch.arange(input_ids.numel(), device=hidden.device).reshape_as(input_ids)
        positions = positions[:, :-1][prediction_mask]
    head = model.get_output_embeddings()
    states = hidden.reshape(-1, width)
    ids = input_ids.reshape(-1)
    total = 0.0
    for start in range(0, len(positions), chunk_tokens):
        if health:
            health()
        with autorelease_scope(hidden.device):
            selected = positions[start:start + chunk_tokens]
            logits = head(states[selected]).float()
            if getattr(model.config, "model_type", None) == "gemma2":
                cap = model.config.final_logit_softcapping
                if cap is not None:
                    if isinstance(cap, bool) or not math.isfinite(cap) or cap <= 0:
                        raise ValueError("invalid Gemma final logit softcap")
                    logits = (logits / cap).tanh() * cap
            loss = F.cross_entropy(logits, ids[selected + 1], reduction="sum")
            total += float(loss.item())
            del logits, loss, selected
    return total, len(positions)


class QualityStats:
    """Merge stable per-batch variance statistics and feature activity counts."""

    def __init__(self, layers, width, latents, *, active_layers=None, thresholds=None):
        self.thresholds = dict(_QUALITY_THRESHOLDS if thresholds is None else thresholds)
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
        if (self.scope != other.scope or self.active_layers != other.active_layers
                or self.thresholds != other.thresholds):
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
        if not self.count or not self.ce_tokens:
            raise ValueError("quality needs observations and next-token targets")
        variance, error = self.variance_sum[active], self.error_sum[active]
        if (variance < 0).any() or not all(torch.isfinite(x).all().item() for x in (variance, error)):
            raise ValueError("quality variance or error is invalid")
        fvu = [float(e / v) if v > 0 else None for e, v in zip(error, variance)]
        mean_fvu = sum(fvu) / len(fvu) if all(v is not None for v in fvu) else None
        dead = (self.hits[active] == 0).double().mean(-1)
        delta = (self.replaced_ce - self.original_ce) / self.ce_tokens
        values = {"fvu": mean_fvu, "dead_fraction": float(dead.mean()),
                  "cross_entropy_delta": delta, "evaluated_tokens": self.count,
                  "cross_entropy_tokens": self.ce_tokens,
                  "original_cross_entropy": self.original_ce / self.ce_tokens,
                  "replaced_cross_entropy": self.replaced_ce / self.ce_tokens,
                  "per_layer_fvu": fvu, "per_layer_dead_fraction": dead.tolist(),
                  "thresholds": dict(self.thresholds)}
        finite = (all(math.isfinite(v) for v in fvu if v is not None)
                  and torch.isfinite(dead).all().item() and math.isfinite(delta))
        if not finite:
            raise ValueError("quality metrics are nonfinite")
        passed = (mean_fvu is not None and mean_fvu <= self.thresholds["fvu"]
                  and values["dead_fraction"] <= self.thresholds["dead_fraction"]
                  and delta <= self.thresholds["cross_entropy_delta"])
        complete = self.count == self.thresholds["evaluated_tokens"] and mean_fvu is not None
        values["status"] = ("passed" if passed else "failed") if complete else "partial"
        if self.scope == "upper_half":
            values["status"] = "fallback_" + values["status"]
            values.update(self.scope_fields())
            for key, per_layer in (("per_layer_fvu", fvu), ("per_layer_dead_fraction", dead.tolist())):
                values[key] = [None] * len(self.hits)
                for index, value in zip(active, per_layer):
                    values[key][index] = value
        return values

    def scope_fields(self):
        return {"scope": self.scope, "active_layers": list(self.active_layers),
                "per_layer_status": ["active" if i in self.active_layers else "error_only"
                                     for i in range(len(self.hits))]}


def quality_stats(metadata):
    fallback = fallback_metadata(metadata)
    thresholds = dict(_QUALITY_THRESHOLDS)
    if metadata.get("model_id") == GEMMA_ID:
        if fallback:
            raise ValueError("Gemma quality requires every layer")
        thresholds["fvu"] = 0.30
    return QualityStats(metadata["layers"], metadata["width"], metadata["latents"],
                        active_layers=fallback["active_layers"] if fallback else None, thresholds=thresholds)


@torch.no_grad()
def evaluate_batch(model, dictionary, ids, count, *, chunk_tokens=64, health=None, check_binding=None,
                   prepared=None):
    """Commit no statistics until both complete model forwards succeed."""
    with autorelease_scope(ids.device):
        return _evaluate_batch(model, dictionary, ids, count, chunk_tokens=chunk_tokens,
                               health=health, check_binding=check_binding, prepared=prepared)


def _evaluate_batch(model, dictionary, ids, count, *, chunk_tokens, health, check_binding, prepared):
    metadata = dictionary.metadata
    result = quality_stats(metadata)
    requested = [metadata[field][i] for field in ("read_points", "write_points") for i in result.active_layers]
    captured = None
    options = {} if prepared is None else {"attention_mask": prepared.attention_mask}
    ce_options = {} if prepared is None else {"prediction_mask": prepared.prediction_mask}
    if prepared is not None and (prepared.input_ids is not ids or int(prepared.content_mask.sum()) != count):
        raise ValueError("prepared context differs from evaluation content")
    try:
        with capture(model, requested) as captured:
            original = hidden_forward(model, ids, **options)
        if check_binding:
            check_binding()
        result.original_ce, result.ce_tokens = cross_entropy_sum(model, original, ids, count,
                                                                chunk_tokens=chunk_tokens, health=health, **ce_options)
        del original
        for i in result.active_layers:
            layer = dictionary.layers[i]
            if health:
                health()
            x, y = (captured[metadata[field][i]] for field in ("read_points", "write_points"))
            if prepared is None:
                x, y = (value.reshape(-1, metadata["width"])[:count] for value in (x, y))
            else:
                x, y = x[prepared.content_mask], y[prepared.content_mask]
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
        if check_binding:
            check_binding()
        with replace_all_mlps(model, dictionary, content_mask=None if prepared is None else prepared.content_mask):
            replaced = hidden_forward(model, ids, **options)
        # Check only after the original MLP modules are restored.
        if check_binding:
            check_binding()
        result.replaced_ce, replaced_count = cross_entropy_sum(model, replaced, ids, count,
                                                              chunk_tokens=chunk_tokens, health=health, **ce_options)
        if replaced_count != result.ce_tokens:
            raise ValueError("cross-entropy target counts differ")
        result.count = count
        return result
    finally:
        if captured is not None:
            captured.close(clear=True)


def prepare_evaluation(folder, stats, stream, pending, identity, settings, elapsed, binding):
    """Write statistics and a staged cursor without replacing published evidence."""
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
    staged = folder / "state.pending.json"
    write_json(staged, {"version": 1, "identity": identity, "settings": settings,
                       "tensor_binding": binding.evidence, "model_binding": binding.model_evidence,
                       "tensors": name, "stream": stream_state, "pending_ids": pending,
                       "count": stats.count, "original_ce": stats.original_ce,
                       "replaced_ce": stats.replaced_ce, "ce_tokens": stats.ce_tokens,
                       "elapsed_seconds": elapsed,
                       **({"content_segments": stats.content_segments} if hasattr(stats, "content_segments") else {})})
    return staged


def save_evaluation(folder, stats, stream, pending, identity, settings, elapsed, binding):
    """Atomically publish small statistics and an exact quality cursor."""
    staged = prepare_evaluation(folder, stats, stream, pending, identity, settings, elapsed, binding)
    staged.replace(Path(folder) / "state.json")


def restore_evaluation(folder, stats, stream, identity, settings, binding):
    folder = Path(folder)
    state = json.loads((folder / "state.json").read_text())
    if (not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1
            or state.get("identity") != identity or not _same_json(state.get("settings"), settings)
            or not _same_json(state.get("tensor_binding"), binding.evidence)):
        raise ValueError("evaluation dictionary or settings changed across resume")
    if not _same_json(state.get("model_binding"), binding.model_evidence):
        raise ValueError("evaluation model identity mismatch: cursor binding differs")
    _validate_cursor_scalars(state, settings)
    vocab = binding.model.config.vocab_size
    if "context_policy" in settings:
        capacity = settings["context_policy"]["content_context"]
        _validate_segments(state.get("content_segments"), state["count"], capacity)
        _validate_pending(state.get("pending_ids"), settings["tokens"] - state["count"], vocab_size=vocab)
        if state["ce_tokens"] != state["count"]:
            raise ValueError("invalid legacy Gemma target count")
    elif state.get("pending_ids") is not None:
        _validate_ids(state["pending_ids"], vocab)
        if len(state["pending_ids"]) > min(settings["batch_tokens"], settings["tokens"] - state["count"]):
            raise ValueError("invalid legacy pending token count")
    _validate_stream_cursor(state, stream, settings, vocab, state["count"])
    binding.verify()
    _validate_tensor_headers(folder, state, stats, ("mean", "variance_sum", "error_sum", "hits"))
    stream.load_state_dict(state["stream"])
    tensors = load_file(folder / state["tensors"])
    _validate_stat_tensors(tensors, state["count"], state["ce_tokens"])
    for name in ("mean", "variance_sum", "error_sum", "hits"):
        getattr(stats, name).copy_(tensors[name])
    for name in ("count", "original_ce", "replaced_ce", "ce_tokens"):
        setattr(stats, name, state[name])
    if "context_policy" in settings:
        stats.content_segments = state["content_segments"]
    return state["pending_ids"], state["elapsed_seconds"]


def _same_json(left, right):
    """Compare checkpoint settings without boolean or numeric type aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], v) for k, v in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _finite_nonnegative(value):
    return type(value) in (int, float) and 0 <= value <= float.fromhex("0x1.fffffffffffffp+1023")


def _validate_cursor_scalars(state, settings, *, research=False):
    counters = ("count", "ce_tokens", "source_tokens") if research else ("count", "ce_tokens")
    if any(type(state.get(key)) is not int or not 0 <= state[key] <= settings["tokens"] for key in counters):
        raise ValueError("invalid evaluation cursor counts")
    if state["ce_tokens"] > state["count"] or (research and state["count"] > state["source_tokens"]):
        raise ValueError("invalid evaluation observation counts")
    if any(not _finite_nonnegative(state.get(key)) for key in ("original_ce", "replaced_ce", "elapsed_seconds")):
        raise ValueError("invalid evaluation loss or elapsed time")
    if not state["ce_tokens"] and (state["original_ce"] or state["replaced_ce"]):
        raise ValueError("evaluation loss requires target observations")


def _validate_ids(ids, vocab_size):
    if (not isinstance(ids, list) or any(type(value) is not int or not 0 <= value < vocab_size for value in ids)):
        raise ValueError("invalid evaluation source token IDs")


def _validate_segments(segments, count, capacity):
    if (not isinstance(segments, list) or any(
            not isinstance(row, list) or len(row) != 3
            or any(type(value) is not int or not 0 <= value < 2**63 for value in row)
            or not 1 <= row[2] <= capacity or row[1] + row[2] >= 2**63 for row in segments)
            or sum(row[2] for row in segments) != count):
        raise ValueError("invalid evaluation content segmentation")
    previous = None
    for doc, position, length in segments:
        _validate_next_coordinate(previous, doc, position)
        if previous is not None and position and previous[2] != capacity:
            raise ValueError("incomplete evaluation context before document continuation")
        previous = (doc, position + length, length)


def _validate_next_coordinate(previous, doc, position):
    if position != 0 and (previous is None or doc != previous[0] or position != previous[1]):
        raise ValueError("evaluation source positions are not contiguous")


def _validate_stream_cursor(state, stream, settings, vocab_size, committed):
    """Validate source metadata and coordinates before statistics or source restoration."""
    source = state.get("stream")
    expected = {"version": 1, "source_id": stream.source_id, "partition": stream.partition,
                "partitions": partition_metadata()}
    if not isinstance(source, dict) or any(not _same_json(source.get(k), v) for k, v in expected.items()):
        raise ValueError("invalid evaluation stream identity")
    for key in ("rows", "emitted", "skipped_long", "offset", "document_id"):
        if type(source.get(key)) is not int or not 0 <= source[key] < 2**63:
            raise ValueError("invalid evaluation stream cursor")
    _validate_ids(source.get("pending"), vocab_size)
    if (source["offset"] > len(source["pending"]) or source["skipped_long"] > source["rows"]
            or (hasattr(stream.documents, "__len__") and source["rows"] > len(stream.documents))):
        raise ValueError("invalid evaluation stream source bounds")
    pending = state.get("pending_ids")
    pending_count = len(pending["ids"]) if isinstance(pending, dict) else len(pending or [])
    if source["emitted"] != committed + pending_count:
        raise ValueError("evaluation stream emitted count differs from committed and pending tokens")
    buffer = source.get("buffer", [[], [], []])
    if not isinstance(buffer, list) or len(buffer) != 3:
        raise ValueError("invalid evaluation stream buffer")
    buffered = dict(zip(("ids", "documents", "positions"), buffer), exhausted=False)
    _validate_pending(buffered, settings["tokens"] - source["emitted"], vocab_size=vocab_size)
    segments = state.get("content_segments")
    if segments is not None:
        previous = None if not segments else (segments[-1][0], segments[-1][1] + segments[-1][2])
        for block in (pending, buffered):
            for doc, position in zip(block["documents"], block["positions"]):
                _validate_next_coordinate(previous, doc, position)
                previous = (doc, position + 1)
        if source["pending"] and previous != (source["document_id"], source["offset"]):
            raise ValueError("evaluation source offset differs from emitted coordinates")
    nested = source.get("source_state")
    if nested is not None:
        if not isinstance(nested, dict) or not hasattr(stream.documents, "load_state_dict"):
            raise ValueError("invalid evaluation source cursor")
        if "version" in nested and type(nested["version"]) is not int:
            raise ValueError("invalid evaluation source cursor version")


def _validate_tensor_headers(folder, state, stats, names):
    """Bound every saved tensor before any tensor payload is loaded."""
    if state.get("tensors") not in ("stats-a.safetensors", "stats-b.safetensors"):
        raise ValueError("invalid evaluation tensor path")
    from safetensors import safe_open
    with safe_open(folder / state["tensors"], framework="pt", device="cpu") as source:
        if set(source.keys()) != set(names):
            raise ValueError("invalid evaluation statistics tensors")
        for name in names:
            stored = source.get_slice(name)
            if (stored.get_shape() != list(getattr(stats, name).shape)
                    or stored.get_dtype() != ("I64" if name == "hits" else "F64")):
                raise ValueError("invalid evaluation statistics shape or dtype")


def _validate_stat_tensors(tensors, count, ce_tokens):
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError("evaluation statistics contain nonfinite values")
        if name != "mean" and (value < 0).any():
            raise ValueError("invalid negative evaluation statistics")
        if not count and value.any():
            raise ValueError("evaluation statistics require observations")
    if ((tensors["hits"] > count).any()
            or (not ce_tokens and "isolated_ce" in tensors and tensors["isolated_ce"].any())):
        raise ValueError("invalid evaluation statistics counts")


def _validate_pending(pending, remaining, *, vocab_size=None):
    if (not isinstance(pending, dict) or set(pending) != {"ids", "documents", "positions", "exhausted"}
            or type(pending["exhausted"]) is not bool
            or any(not isinstance(pending[name], list) for name in ("ids", "documents", "positions"))
            or not len(pending["ids"]) == len(pending["documents"]) == len(pending["positions"])
            or not 0 <= len(pending["ids"]) <= min(4096, remaining)
            or any(type(value) is not int or not 0 <= value < 2**63 for name in ("ids", "documents", "positions")
                   for value in pending[name])):
        raise ValueError("invalid evaluation pending content")
    if vocab_size is not None:
        _validate_ids(pending["ids"], vocab_size)
    list(source_segments(pending["documents"], pending["positions"], 4096, final=True))


def _segmentation_record(segments):
    """Bind exact cursor spans without an unbounded list in dictionary metadata."""
    digest = hashlib.sha256()
    for row in segments:
        digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode("ascii"))
    return {"contexts": len(segments), "sha256": digest.hexdigest(),
            "encoding": "sha256_ascii_json_rows_with_newlines",
            "fields": ["document_id", "source_position", "content_length"],
            "record": "evaluation/state.json:content_segments"}


def _gemma_batch(stream, pending, remaining, policy, batch_tokens, *, device, health, prepare=None,
                 vocab_size=None, prior_segments=None):
    """Retain source coordinates and complete contexts across source batch boundaries."""
    _validate_pending(pending, remaining, vocab_size=vocab_size)
    row_limit = batch_tokens // policy["model_context"]
    capacity = policy["content_context"]
    while True:
        final = pending["exhausted"] or len(pending["ids"]) == remaining
        spans = list(source_segments(pending["documents"], pending["positions"], capacity, final=final))
        if len(spans) >= row_limit or final:
            spans = spans[:row_limit]
            if not spans:
                raise CleanStop("quality source exhausted")
            rows = [pending["ids"][start:end] for start, end in spans]
            segments = [[pending["documents"][start], pending["positions"][start], end - start]
                        for start, end in spans]
            if prior_segments is not None:
                last = prior_segments[-1] if prior_segments else None
                previous = None if last is None else (last[0], last[1] + last[2])
                _validate_next_coordinate(previous, segments[0][0], segments[0][1])
            return (prepare or prepare_contexts)(rows, policy, device=device, pad_to=policy["model_context"]), segments
        requested = min(row_limit * capacity, remaining) - len(pending["ids"])
        batch = stream.next_tokens(requested, health=health)
        for field, value in (("ids", batch.input_ids), ("documents", batch.document_ids), ("positions", batch.positions)):
            if value.ndim != 1 or value.dtype != torch.int64 or value.device.type != "cpu":
                raise ValueError("evaluation source requires CPU int64 token coordinates")
            pending[field].extend(value.tolist())
        pending["exhausted"] = len(batch) < requested
        _validate_pending(pending, remaining, vocab_size=vocab_size)


def evaluate(model, dictionary, stream, folder, *, tokens=200_000, context=None,
             batch_tokens=None, chunk_tokens=64, memory=None, stop=None, resume=False, protocol="legacy"):
    context = (1024 if protocol == "research_v1" else 256) if context is None else context
    batch_tokens = (max(1024, context) if protocol == "research_v1" else 512) if batch_tokens is None else batch_tokens
    if protocol == "research_v1":
        return evaluate_research(model, dictionary, stream, folder, tokens=tokens, context=context,
                                 batch_tokens=batch_tokens, chunk_tokens=chunk_tokens,
                                 memory=memory, stop=stop, resume=resume)
    if protocol != "legacy":
        raise ValueError("unknown evaluation protocol")
    if (any(type(x) is not int for x in (tokens, context, batch_tokens, chunk_tokens))
            or not 2 <= tokens <= 200_000 or not 2 <= context <= 256
            or not context <= batch_tokens <= 4096 or not 1 <= chunk_tokens <= 128):
        raise ValueError("invalid evaluation dimensions")
    if stream.partition != "quality":
        raise ValueError("quality evaluation requires the quality partition")
    metadata = dictionary.metadata
    source_metadata = evaluation_source_metadata(metadata)
    trained_source = source_metadata.get("source_id")
    if trained_source is not None and trained_source != stream.source_id:
        label = "evaluation" if "import_source" in metadata else "training"
        raise ValueError(f"quality source differs from the {label} source")
    if source_metadata["dataset_id"] != "fixture" and source_metadata["partitions"] != partition_metadata():
        raise ValueError("dictionary text partitions differ from this evaluator")
    folder = Path(folder)
    identity = dictionary_fingerprint(folder, metadata, memory=memory)
    binding = EvaluationBinding(folder, dictionary, identity, model=model, memory=memory)
    state_folder = folder / "evaluation"
    stats = quality_stats(metadata)
    policy = context_policy(model, context, tokenizer=getattr(stream, "tokenizer", None))
    settings = {"tokens": tokens, "context": context, "batch_tokens": batch_tokens, "chunk_tokens": chunk_tokens}
    if policy is not None:
        settings["context_policy"] = policy
        stats.content_segments = []
    if "import_source" in metadata:
        settings["evaluation_source"] = source_metadata
    if stats.scope == "upper_half":
        settings.update(scope=stats.scope, active_layers=list(stats.active_layers))
    pending, elapsed_before = None, 0.0
    if policy is not None:
        pending = {"ids": [], "documents": [], "positions": [], "exhausted": False}
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
            prepared, segments = None, None
            if policy is not None:
                prepared, segments = _gemma_batch(stream, pending, tokens - stats.count, policy, batch_tokens,
                                                   device=device, health=health, vocab_size=model.config.vocab_size,
                                                   prior_segments=stats.content_segments)
                ids = prepared.input_ids
                count = sum(row[2] for row in segments)
            elif pending is None:
                batch = stream.next_tokens(min(batch_tokens, tokens - stats.count), health=health)
                if not len(batch):
                    raise CleanStop("quality source exhausted")
                pending = batch.input_ids.tolist()
            if policy is None:
                count = len(pending)
                ids = torch.tensor(pending, dtype=torch.int64, device=device)
                ids = F.pad(ids, (0, (-count) % context)).reshape(-1, context)
            binding.check()
            batch_options = {} if prepared is None else {"prepared": prepared}
            result = evaluate_batch(model, dictionary, ids, count, chunk_tokens=chunk_tokens,
                                    health=health, check_binding=binding.check, **batch_options)
            binding.check()
            stats.merge(result)
            if policy is None:
                pending = None
            else:
                stats.content_segments.extend(segments)
                for field in ("ids", "documents", "positions"):
                    del pending[field][:count]
            del ids, prepared, result, batch_options
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
                   "original_cross_entropy": None, "replaced_cross_entropy": None,
                   "per_layer_fvu": [None] * len(dictionary.layers),
                   "per_layer_dead_fraction": [None] * len(dictionary.layers),
                   "thresholds": dict(stats.thresholds)}
    quality.update(elapsed_seconds=elapsed_before + time.monotonic() - started, stop_reason=reason,
                   dictionary_hash=identity, tensor_binding=binding.evidence,
                   model_binding=binding.model_evidence, partition="quality")
    if policy is not None:
        quality.update(context_policy=policy, content_segmentation=_segmentation_record(stats.content_segments))
    if "import_source" in metadata:
        quality.update(evaluation_source=source_metadata, publisher_training_overlap="unknown")
    if stats.scope == "upper_half":
        quality.update(stats.scope_fields(), replaced_mlps=len(stats.active_layers))
    else:
        quality.update(all_mlps_replaced=len(dictionary.layers))
    if memory:
        quality["peak_memory_bytes"] = memory.peak
    if ready:
        staged = prepare_evaluation(state_folder, stats, stream, pending, identity, settings,
                                     elapsed_before + time.monotonic() - started, binding)
    # Keep the prior cursor and quality until every final memory and identity check passes.
    binding.verify()
    if ready:
        staged.replace(state_folder / "state.json")
    metadata["quality"] = quality
    write_json(folder / "dictionary.json", metadata)
    print(json.dumps({"stage": "quality_end", **quality}), flush=True)
    return quality


@torch.no_grad()
def evaluate_research_batch(model, dictionary, prepared, source_count, *, chunk_tokens, health, check_binding):
    """Measure one atomic batch with one baseline and serial replacement passes."""
    from .evaluation_protocol import ResearchStats, research_mlps

    result = ResearchStats(quality_stats(dictionary.metadata))
    result.source_tokens = source_count
    result.count = int(prepared.content_mask.sum())
    ids = prepared.input_ids
    metadata = dictionary.metadata
    requested = [metadata[field][i] for field in ("read_points", "write_points") for i in result.active_layers]
    captured = None
    try:
        with autorelease_scope(ids.device):
            with capture(model, requested) as captured:
                original = hidden_forward(model, ids, attention_mask=prepared.attention_mask)
            check_binding()
            result.original_ce, result.ce_tokens = cross_entropy_sum(
                model, original, ids, source_count, chunk_tokens=chunk_tokens,
                health=health, prediction_mask=prepared.prediction_mask)
            del original
            if not math.isfinite(result.original_ce):
                raise ValueError("research baseline cross-entropy is nonfinite")
            # Reduce CPU float64 moments in chunks, including between-context mean shifts.
            positions = prepared.content_mask.reshape(-1).nonzero().flatten()
            for i in result.active_layers:
                x, y = (captured[metadata[field][i]].reshape(-1, metadata["width"])
                        for field in ("read_points", "write_points"))
                observed = 0
                for start in range(0, result.count, chunk_tokens):
                    health()
                    with autorelease_scope(ids.device):
                        selected = positions[start:start + chunk_tokens]
                        features = dictionary.layers[i].encode(x[selected])
                        decoded = dictionary.layers[i].decode(features)
                        target = y[selected].cpu().double()
                        reconstructed = decoded.cpu().double()
                        if not all(torch.isfinite(t).all() for t in (features, target, reconstructed)):
                            raise ValueError("research observations contain nonfinite values")
                        n = len(selected)
                        mean = target.mean(0)
                        delta = mean - result.mean[i]
                        result.variance_sum[i] += ((target - mean).square().sum()
                                                  + delta.square().sum() * (observed * n / (observed + n)))
                        result.mean[i] += delta * (n / (observed + n))
                        result.error_sum[i] += (target - reconstructed).square().sum()
                        result.hits[i] += (features > 0).sum(0).cpu()
                        observed += n
                        del selected, features, decoded, target, reconstructed, mean, delta
                del x, y
                captured.pop(metadata["read_points"][i])
                captured.pop(metadata["write_points"][i])
            captured.close(clear=True)
            del positions
        # Every pass releases hidden states and autorelease objects before the next pass.
        for index in [*result.active_layers, None]:
            layers = result.active_layers if index is None else (index,)
            health()
            check_binding()
            with autorelease_scope(ids.device):
                with research_mlps(model, dictionary, layers, prepared.content_mask,
                                   chunk_tokens=chunk_tokens, health=health):
                    replaced = hidden_forward(model, ids, attention_mask=prepared.attention_mask)
                check_binding()
                ce, targets = cross_entropy_sum(model, replaced, ids, source_count, chunk_tokens=chunk_tokens,
                                                health=health, prediction_mask=prepared.prediction_mask)
                del replaced
            if targets != result.ce_tokens or not math.isfinite(ce):
                raise ValueError("research cross-entropy targets differ or loss is nonfinite")
            # The final pass remains distinct even for a one-layer active scope.
            if index is None:
                result.replaced_ce = ce
            else:
                result.isolated_ce[index] = ce
        result.validate()
        return result
    finally:
        if captured is not None:
            captured.close(clear=True)


def save_research(folder, stats, stream, pending, identity, settings, elapsed, binding, *, quality=None):
    """Publish committed statistics with a replayable pending batch."""
    stats.validate()
    binding.verify()
    folder.mkdir(exist_ok=True, parents=True)
    prior = json.loads((folder / "state.json").read_text()) if (folder / "state.json").exists() else {}
    name = "stats-b.safetensors" if prior.get("tensors") == "stats-a.safetensors" else "stats-a.safetensors"
    _save_tensors({key: getattr(stats, key) for key in stats.tensor_names}, folder / name)
    state = {"version": 2, "identity": identity, "settings": settings,
             "tensor_binding": binding.evidence, "model_binding": binding.model_evidence,
             "tensors": name, "stream": stream.state_dict(), "pending_ids": pending,
             "content_segments": stats.content_segments, "elapsed_seconds": elapsed,
             "metric_observations": stats.count, "cross_entropy_tokens": stats.ce_tokens,
             **{key: getattr(stats, key) for key in stats.scalar_names}}
    staged = folder / "state.pending.json"
    write_json(staged, state)
    binding.verify()
    staged.replace(folder / "state.json")
    if quality is not None:
        # Bind report counters to the committed cursor. A report can lag after a crash.
        quality["checkpoint_sha256"] = hashlib.sha256((folder / "state.json").read_bytes()).hexdigest()
        write_json(folder / "quality.json", quality)


def restore_research(folder, stats, stream, identity, settings, binding):
    """Reject protocol and identity changes before statistics payload loads."""
    state = json.loads((folder / "state.json").read_text())
    if (not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 2
            or not _same_json(state.get("settings"), settings) or state.get("identity") != identity
            or not _same_json(state.get("tensor_binding"), binding.evidence)
            or not _same_json(state.get("model_binding"), binding.model_evidence)):
        raise ValueError("research evaluation protocol, source, model, or dictionary settings changed across resume")
    _validate_cursor_scalars(state, settings, research=True)
    if (not _same_json(state.get("metric_observations"), state["count"])
            or not _same_json(state.get("cross_entropy_tokens"), state["ce_tokens"])):
        raise ValueError("research observation counters differ")
    segments = state.get("content_segments")
    capacity = settings["context_policy"]["content_context"]
    _validate_segments(segments, state["source_tokens"], capacity)
    vocab = binding.model.config.vocab_size
    _validate_pending(state.get("pending_ids"), settings["tokens"] - state["source_tokens"], vocab_size=vocab)
    if settings["context_policy"]["prefix_ids"] and state["ce_tokens"] != state["count"]:
        raise ValueError("research BOS target count differs from metric observations")
    _validate_stream_cursor(state, stream, settings, vocab, state["source_tokens"])
    binding.verify()
    _validate_tensor_headers(folder, state, stats, stats.tensor_names)
    stream.load_state_dict(state["stream"])
    tensors = load_file(folder / state["tensors"])
    _validate_stat_tensors(tensors, state["count"], state["ce_tokens"])
    for name in stats.tensor_names:
        getattr(stats, name).copy_(tensors[name])
    for name in stats.scalar_names:
        setattr(stats, name, state[name])
    stats.content_segments = segments
    stats.validate()
    return state["pending_ids"], state["elapsed_seconds"]


def evaluate_research(model, dictionary, stream, folder, *, tokens, context, batch_tokens,
                      chunk_tokens, memory, stop, resume):
    """Run the explicit research protocol without a change to legacy publications."""
    from .evaluation_protocol import (
        CHECKPOINT_SECONDS, ResearchStats, admit_workspace, protocol_settings,
        research_contexts, research_policy, workspace_bytes,
    )
    from .text_stream import DATASET_ID, DATASET_CONFIG, DATASET_REVISION

    if (any(type(x) is not int for x in (tokens, context, batch_tokens, chunk_tokens))
            or not 2 <= tokens <= 200_000 or not 2 <= context <= 1024
            or not context <= batch_tokens <= 4096 or not 1 <= chunk_tokens <= 128):
        raise ValueError("invalid research evaluation dimensions")
    if stream.partition != "quality":
        raise ValueError("research evaluation requires the quality partition")
    metadata = dictionary.metadata
    source = evaluation_source_metadata(metadata)
    if source.get("source_id") is not None and source["source_id"] != stream.source_id:
        raise ValueError("research evaluation source differs from the dictionary source")
    if source["dataset_id"] != "fixture" and any(source.get(key) != value for key, value in {
            "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION, "split": "train", "partitions": partition_metadata()}.items()):
        raise ValueError("research evaluation requires the pinned FineWeb source")
    policy = research_policy(model, context, stream.tokenizer)
    device = dictionary.layers[0].W_enc.device
    ceiling = 40 if metadata.get("model_id") == GEMMA_ID else 12
    memory = MemoryGuard(ceiling, device) if memory is None else BudgetMemory(memory, ceiling)
    if _canonical_device(memory.device) != _canonical_device(device):
        raise ValueError("research evaluation requires a guard for its device")
    required = workspace_bytes(model, metadata, batch_tokens, context, chunk_tokens)
    if stop:
        stop.check()
    # This check precedes context, capture, feature, and head allocations.
    admit_workspace(memory, required)
    folder = Path(folder)
    identity = dictionary_fingerprint(folder, metadata, memory=memory)
    binding = EvaluationBinding(folder, dictionary, identity, model=model, memory=memory)
    state_folder = folder / "evaluation-research-v1"
    settings = {"protocol": protocol_settings(policy), "context_policy": policy,
                "tokens": tokens, "context": context, "batch_tokens": batch_tokens, "chunk_tokens": chunk_tokens,
                "source": {**{k: source[k] for k in ("dataset_id", "dataset_config", "dataset_revision", "split")},
                           "source_id": stream.source_id, "partition": "quality", "partitions": partition_metadata()},
                "dictionary_hash": identity, "tensor_binding": binding.evidence, "model_binding": binding.model_evidence}
    stats = ResearchStats(quality_stats(metadata))
    settings.update(scope=stats.scope, active_layers=list(stats.active_layers))
    pending = {"ids": [], "documents": [], "positions": [], "exhausted": False}
    elapsed_before = 0.0
    if resume:
        pending, elapsed_before = restore_research(state_folder, stats, stream, identity, settings, binding)
    elif (state_folder / "state.json").exists():
        raise ValueError("research evaluation state exists; use --resume")
    started = last_save = time.monotonic()

    def health():
        nonlocal last_save
        if stop:
            stop.check()
        memory.check("research_evaluation", force=False)
        if stream.source_id != settings["source"]["source_id"] or research_policy(model, context, stream.tokenizer) != policy:
            raise ValueError("research source or context policy changed during evaluation")
        if time.monotonic() - last_save >= CHECKPOINT_SECONDS:
            save_research(state_folder, stats, stream, pending, identity, settings,
                          elapsed_before + time.monotonic() - started, binding)
            last_save = time.monotonic()

    reason = "complete"
    handles = []
    try:
        # Long sweeps can checkpoint the prior committed batch between model layers.
        # Hooks change outputs, never module identities or model parameters.
        for point in metadata["read_points"]:
            module = model.get_submodule(point.rsplit(".", 1)[0])
            handles.append(module.register_forward_pre_hook(lambda *args: health()))
        while stats.source_tokens < tokens:
            health()
            binding.check()
            admit_workspace(memory, required)
            prepared, segments = _gemma_batch(stream, pending, tokens - stats.source_tokens,
                                               policy, batch_tokens, device=device, health=health,
                                               prepare=research_contexts, vocab_size=model.config.vocab_size,
                                               prior_segments=stats.content_segments)
            count = sum(row[2] for row in segments)
            try:
                result = evaluate_research_batch(model, dictionary, prepared, count, chunk_tokens=chunk_tokens,
                                                 health=health, check_binding=binding.check)
                binding.check()
                result.content_segments = segments
                stats.merge(result)
                for field in ("ids", "documents", "positions"):
                    del pending[field][:count]
                del result
            finally:
                del prepared
            print(json.dumps({"stage": "research_quality_batch", "source_tokens": stats.source_tokens,
                              "metric_observations": stats.count, "cross_entropy_tokens": stats.ce_tokens,
                              "target": tokens}), flush=True)
    except CleanStop as error:
        reason = stop_reason(error)
    finally:
        for handle in reversed(handles):
            handle.remove()
    quality = stats.metrics()
    quality.update(settings=settings, context_policy=policy,
                   content_segmentation={**_segmentation_record(stats.content_segments),
                                         "record": "evaluation-research-v1/state.json:content_segments"},
                   elapsed_seconds=elapsed_before + time.monotonic() - started, stop_reason=reason,
                   dictionary_hash=identity, tensor_binding=binding.evidence, model_binding=binding.model_evidence,
                   partition="quality", evaluation_source=settings["source"], publisher_training_overlap="unknown",
                   peak_memory_bytes=memory.peak, workspace_reserve_bytes=required,
                   replaced_mlps=len(stats.active_layers))
    if stats.scope == "all_layers":
        quality["all_mlps_replaced"] = len(dictionary.layers)
    save_research(state_folder, stats, stream, pending, identity, settings, quality["elapsed_seconds"],
                  binding, quality=quality)
    print(json.dumps({"stage": "research_quality_end", **quality}), flush=True)
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--model", choices=(MODEL_ID, GEMMA_ID), default=MODEL_ID)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--tokens", type=int, default=200_000)
    parser.add_argument("--context", type=int)
    parser.add_argument("--batch-tokens", type=int)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--local-dataset-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol", choices=("legacy", "research_v1"), default="legacy")
    args = parser.parse_args(argv)
    if args.context is None:
        args.context = 1024 if args.protocol == "research_v1" else 256
    if args.batch_tokens is None:
        args.batch_tokens = 1024 if args.protocol == "research_v1" else 512
    if (not 2 <= args.tokens <= 200_000 or not 2 <= args.context <= (1024 if args.protocol == "research_v1" else 256)
            or not args.context <= args.batch_tokens <= 4096 or not 1 <= args.chunk_tokens <= 128):
        raise ValueError("requested dimensions exceed the bounded evaluator allocations")
    ceiling = 40 if args.model == GEMMA_ID else 12
    args.memory_gb = ceiling if args.memory_gb is None else args.memory_gb
    if not 0 < args.memory_gb <= ceiling:
        raise ValueError(f"the evaluator memory limit is at most {ceiling} GB")
    try:
        with StopControl() as stop, ExitStack() as resources:
            stop.check()
            memory = MemoryGuard(args.memory_gb, args.device)
            memory.check("start")
            stop.check()
            selection = {"model_id": args.model} if args.model != MODEL_ID else {}
            model, tokenizer = load_text_model(device=args.device, memory=memory, **selection)
            stop.check()
            dictionary = load_dictionary(args.dictionary, model=model, device=args.device, dtype=torch.float32,
                                         memory_limit_gb=args.memory_gb, memory=memory)
            stop.check()
            memory.check("after_dictionary_load")
            stop.check()

            def source_health():
                stop.check()
                memory.check("evaluation_source", force=False)

            source = fineweb_documents(cache_dir=args.dataset_cache, local_manifest=args.local_dataset_manifest,
                                       health=source_health, memory=memory)
            if hasattr(source, "close"):
                resources.callback(source.close)
            stream = resources.enter_context(TextStream(source, tokenizer, "quality"))
            stop.check()
            result = evaluate(model, dictionary, stream, args.dictionary, tokens=args.tokens,
                              context=args.context, batch_tokens=args.batch_tokens, chunk_tokens=args.chunk_tokens,
                              memory=memory, stop=stop, resume=args.resume, protocol=args.protocol)
    except CleanStop as error:
        print(json.dumps(clean_stop_record(error)), flush=True)
        return 2
    return 0 if result["status"] in ("passed", "fallback_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
