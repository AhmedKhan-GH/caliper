"""Versioned research contexts, bounded interventions, and dataset statistics."""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from caliper.interp._context import ContextBatch
from caliper.interp._memory import allocation_bytes, autorelease_scope
from caliper.interp._points import _architecture, _input
from .memory_limits import CleanStop


PROTOCOL = "research_v1"
CHECKPOINT_SECONDS = 120


def research_policy(model, context, tokenizer):
    """Describe actual special tokens without a synthetic BOS substitute."""
    if type(context) is not int or not 2 <= context <= 1024:
        raise ValueError("research context must have two to 1024 model positions")
    config = model.config
    vocab = config.vocab_size
    if type(vocab) is not int or vocab < 1:
        raise ValueError("research context requires a valid vocabulary size")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if not hasattr(tokenizer, name):
            raise ValueError("research tokenizer must declare its special token IDs")
        value = getattr(tokenizer, name)
        if value is not None and (type(value) is not int or not 0 <= value < vocab):
            raise ValueError("research tokenizer requires valid special token IDs")
    bos = getattr(config, "bos_token_id", None)
    if bos is not None and (type(bos) is not int or not 0 <= bos < vocab):
        raise ValueError("research model requires a valid BOS ID")
    token_bos = getattr(tokenizer, "bos_token_id", None)
    if bos != token_bos:
        raise ValueError("research tokenizer BOS differs from the model")
    add_bos = getattr(tokenizer, "add_bos_token", False)
    if type(add_bos) is not bool or (bos is None and add_bos):
        raise ValueError("research tokenizer requests an undeclared BOS")
    pad = getattr(config, "pad_token_id", None)
    if pad is not None and (type(pad) is not int or not 0 <= pad < vocab):
        raise ValueError("research model requires a valid padding ID")
    token_pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None and token_pad != pad:
        raise ValueError("research tokenizer padding differs from the model")
    pad = token_pad if pad is None else pad
    eos = getattr(config, "eos_token_id", None)
    eos = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if any(type(value) is not int or not 0 <= value < vocab for value in eos):
        raise ValueError("research model requires valid EOS IDs")
    token_eos = getattr(tokenizer, "eos_token_id", None)
    if eos and token_eos not in eos:
        raise ValueError("research tokenizer EOS differs from the model")
    if not eos and token_eos is not None:
        eos = [token_eos]
    special = sorted(set([pad, *eos, *([] if bos is None else [bos])]), key=lambda x: -1 if x is None else x)
    if any(type(value) is not int or not 0 <= value < config.vocab_size for value in special):
        raise ValueError("research context requires valid special token IDs and padding")
    return {"version": 1, "family": config.model_type,
            "bos_policy": "native_no_bos" if bos is None else "genuine_bos_prefix",
            "prefix_ids": [] if bos is None else [bos], "pad_id": pad, "eos_ids": eos,
            "excluded_ids": special, "model_context": context,
            "content_context": context - (bos is not None),
            "segmentation": "contiguous_document_positions",
            "observations": "source_positions_excluding_bos_eos_pad",
            "replacement": "metric_positions_only_preserve_special_outputs",
            "cross_entropy": "attended_predictor_and_next_nonspecial_source_target_in_same_row",
            "first_token_loss": "no_predictor" if bos is None else "bos_predicts_first_nonspecial_target"}


def research_contexts(rows, policy, *, device, pad_to=None):
    """Keep source tokens intact and mask genuine special positions."""
    prefix = policy["prefix_ids"]
    lengths = [len(row) for row in rows]
    if not lengths or min(lengths) < 1 or max(lengths) > policy["content_context"]:
        raise ValueError("invalid research source context lengths")
    required = max(lengths) + len(prefix)
    width = required if pad_to is None else pad_to
    if type(width) is not int or not required <= width <= policy["model_context"]:
        raise ValueError("invalid research padded context")
    with autorelease_scope(device):
        ids = torch.full((len(rows), width), policy["pad_id"], dtype=torch.int64, device=device)
        attention = torch.zeros_like(ids)
        observed = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            end = len(prefix) + len(row)
            ids[index, :end] = torch.tensor(prefix + list(row), dtype=torch.int64, device=device)
            attention[index, :end] = 1
            observed[index, len(prefix):end] = True
        for special in policy["excluded_ids"]:
            observed &= ids != special
        return ContextBatch(ids, attention, observed)


def protocol_settings(policy):
    """Return the exact aggregation and restart contract for checkpoint identity."""
    return {"name": PROTOCOL, "version": 1, "context_policy": policy,
            "fvu": "sum_squared_error_over_sum_squared_deviation_from_dataset_mean_vector_per_layer",
            "activity": "positive_features_per_nonspecial_source_position",
            "loss": "sum_nats_over_identical_targets_baseline_isolated_and_all_active",
            "statistics_dtype": "cpu_float64", "source_budget": 200_000,
            "restart": "atomic_batch_replay_no_partial_layer_commit",
            "checkpoint_seconds": CHECKPOINT_SECONDS,
            "paper_reference": {"section": "4.1", "pages": [5, 6], "sequences": 2048,
                                "model_context": 1024,
                                "url": "https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf"},
            "sample_difference": "held_out_fineweb_edu_200000_source_tokens_with_partial_document_contexts"}


def workspace_bytes(model, metadata, batch_tokens, context, chunk_tokens):
    """Reserve captures, attention, MLP workspace, bounded features, and head chunks."""
    config = model.config
    width, layers, latents = (metadata[k] for k in ("width", "layers", "latents"))
    rows = batch_tokens // context
    positions = rows * context
    heads = getattr(config, "num_attention_heads", 1)
    intermediate = getattr(config, "intermediate_size", 4 * width)
    # Eight bytes cover CPU float64 fixtures and temporary device/CPU copies.
    captures = 8 * positions * width * (2 * layers + 12)
    attention = 8 * rows * heads * context * context * 4
    mlp = 8 * positions * intermediate * 6
    chunks = 8 * chunk_tokens * (config.vocab_size * 4 + latents * 4 + width * 8)
    stats = 32 * layers * (width + latents)
    return captures + attention + mlp + chunks + stats + 16 * 1024 * 1024


def admit_workspace(memory, required):
    """Refuse before allocation under the process limit and 8 GB host reserve."""
    import psutil

    memory.check("research_evaluation_admission")
    available = min(memory.limit - allocation_bytes(memory), psutil.virtual_memory().available - 8_000_000_000)
    if required > available:
        raise CleanStop("research evaluation workspace exceeds available memory headroom")


@contextmanager
def research_mlps(model, dictionary, active_layers, mask, *, chunk_tokens, health=None):
    """Replace semantic writes in bounded chunks and restore all hooks on exit."""
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2
            or type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 128):
        raise ValueError("invalid research replacement mask or chunk size")
    handles, reads = [], {}
    probes = _architecture(model)
    try:
        for index in active_layers:
            read = probes[dictionary.metadata["read_points"][index]]
            write = probes[dictionary.metadata["write_points"][index]]
            if read.event != "input" or write.event != "output":
                raise ValueError("research replacement requires semantic MLP input and output")

            def remember(module, args, kwargs, index=index):
                reads[index] = _input(args, kwargs)

            def replace(module, args, output, index=index):
                x = reads.pop(index)
                if (output.shape != x.shape or mask.shape != output.shape[:-1]
                        or mask.device != output.device):
                    raise ValueError("research replacement shape differs from context")
                value = output.clone()
                positions = mask.reshape(-1).nonzero().flatten()
                flat_x, flat_value = x.reshape(-1, x.shape[-1]), value.reshape(-1, value.shape[-1])
                layer = dictionary.layers[index]
                for start in range(0, len(positions), chunk_tokens):
                    if health:
                        health()
                    with autorelease_scope(output.device):
                        selected = positions[start:start + chunk_tokens]
                        features = layer.encode(flat_x[selected])
                        decoded = layer.decode(features)
                        if not torch.isfinite(features).all() or not torch.isfinite(decoded).all():
                            raise ValueError("research replacement contains nonfinite values")
                        flat_value[selected] = decoded.to(dtype=output.dtype, device=output.device)
                        del features, decoded, selected
                return value

            handles.append(model.get_submodule(read.module).register_forward_pre_hook(remember, with_kwargs=True))
            handles.append(model.get_submodule(write.module).register_forward_hook(replace))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        reads.clear()


class ResearchStats:
    """Accumulate independent layer moments and loss sums without partial batches."""

    tensor_names = ("mean", "variance_sum", "error_sum", "hits", "isolated_ce")
    scalar_names = ("count", "source_tokens", "original_ce", "replaced_ce", "ce_tokens")

    def __init__(self, legacy):
        self.__dict__.update(legacy.__dict__)
        self.source_tokens = 0
        self.isolated_ce = torch.zeros(len(self.hits), dtype=torch.float64)
        self.content_segments = []

    def validate(self):
        for name in self.tensor_names:
            value = getattr(self, name)
            if not torch.isfinite(value).all():
                raise ValueError("research statistics contain nonfinite values")
        if any(not math.isfinite(getattr(self, name)) for name in self.scalar_names):
            raise ValueError("research loss or counts contain nonfinite values")
        if (any(type(getattr(self, name)) is not int or getattr(self, name) < 0
                for name in ("count", "source_tokens", "ce_tokens"))
                or self.count > self.source_tokens or self.ce_tokens > self.count
                or (self.hits < 0).any() or (self.hits > self.count).any()
                or (self.variance_sum < 0).any() or (self.error_sum < 0).any()
                or (self.isolated_ce < 0).any() or min(self.original_ce, self.replaced_ce) < 0):
            raise ValueError("invalid research statistics")

    def merge(self, other):
        if self.active_layers != other.active_layers or self.thresholds != other.thresholds:
            raise ValueError("research scopes differ")
        other.validate()
        total = self.count + other.count
        if other.count:
            delta = other.mean - self.mean
            self.variance_sum += other.variance_sum + delta.square().sum(-1) * (self.count * other.count / total)
            self.mean += delta * (other.count / total)
        self.count = total
        self.source_tokens += other.source_tokens
        self.error_sum += other.error_sum
        self.hits += other.hits
        self.original_ce += other.original_ce
        self.replaced_ce += other.replaced_ce
        self.ce_tokens += other.ce_tokens
        self.isolated_ce += other.isolated_ce
        self.content_segments.extend(other.content_segments)
        self.validate()

    def metrics(self):
        self.validate()
        layers = len(self.hits)
        fvu, dead, l0, isolated, delta = ([None] * layers for _ in range(5))
        baseline = self.original_ce / self.ce_tokens if self.ce_tokens else None
        replaced = self.replaced_ce / self.ce_tokens if self.ce_tokens else None
        for i in self.active_layers:
            if self.count:
                variance = self.variance_sum[i].item()
                fvu[i] = self.error_sum[i].item() / variance if variance else None
                dead[i] = (self.hits[i] == 0).double().mean().item()
                l0[i] = self.hits[i].sum().item() / self.count
            if self.ce_tokens:
                isolated[i] = self.isolated_ce[i].item() / self.ce_tokens
                delta[i] = isolated[i] - baseline
        active_fvu = [fvu[i] for i in self.active_layers]
        mean_fvu = sum(active_fvu) / len(active_fvu) if all(x is not None for x in active_fvu) else None
        mean_dead = sum(dead[i] for i in self.active_layers) / len(self.active_layers) if self.count else None
        all_delta = replaced - baseline if self.ce_tokens else None
        values = [baseline, replaced, mean_fvu, mean_dead, all_delta, *fvu, *dead, *l0, *isolated, *delta]
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("research metrics contain nonfinite values")
        complete = self.source_tokens == self.thresholds["evaluated_tokens"]
        measurable = mean_fvu is not None and all_delta is not None
        passed = measurable and (mean_fvu <= self.thresholds["fvu"] and
                                  mean_dead <= self.thresholds["dead_fraction"] and
                                  all_delta <= self.thresholds["cross_entropy_delta"])
        status = ("passed" if passed else "failed") if complete else "partial"
        if self.scope == "upper_half":
            status = "fallback_" + status
        return {"protocol": PROTOCOL, "protocol_version": 1, "status": status,
                "source_budget_complete": complete, "source_tokens": self.source_tokens,
                "evaluated_tokens": self.source_tokens, "metric_observations": self.count,
                "cross_entropy_tokens": self.ce_tokens, "original_cross_entropy": baseline,
                "replaced_cross_entropy": replaced, "cross_entropy_delta": all_delta,
                "per_layer_cross_entropy": isolated, "per_layer_cross_entropy_delta": delta,
                "fvu": mean_fvu, "per_layer_fvu": fvu, "per_layer_l0": l0,
                "dead_fraction": mean_dead, "per_layer_dead_fraction": dead,
                "thresholds": dict(self.thresholds), "scope": self.scope,
                "active_layers": list(self.active_layers),
                "per_layer_status": ["active" if i in self.active_layers else "error_only" for i in range(layers)]}
