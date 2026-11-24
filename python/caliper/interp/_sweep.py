"""Activation patch recovery from bounded batches of real model forwards."""
from contextlib import ExitStack
import math

import torch

from ._memory import autorelease_scope
from ._output import forward_logits, prompt_inputs
from ._points import _architecture, _input, capture


def patch_sweep(model, tokenizer, interventions, *, clean_prompt, corrupted_prompt, metric, guard=None):
    """Patch each residual input cell, with at most 32 cells per batch.

    The metric is the positive token logit minus the negative token logit.
    Every forward suspends existing interventions. The clean cache is local.
    """
    if (type(metric) is not dict or set(metric) != {"positive_token", "negative_token"}
            or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in metric.values())
            or metric["positive_token"] == metric["negative_token"]):
        raise ValueError("metric requires two distinct vocabulary token IDs")
    clean = prompt_inputs(model, tokenizer, clean_prompt)
    corrupted = prompt_inputs(model, tokenizer, corrupted_prompt)
    if clean["input_ids"].shape != corrupted["input_ids"].shape:
        raise ValueError("clean and corrupted prompts must have equal token counts")
    probes = sorted((p for p in _architecture(model).values() if p.point.endswith(".resid_pre")),
                    key=lambda p: p.layer)
    modules = dict(model.named_modules())
    positions = clean["input_ids"].shape[1]
    positive, negative = metric["positive_token"], metric["negative_token"]

    def score(logits):
        return logits[:, -1, positive] - logits[:, -1, negative]

    cache = {}
    try:
        with interventions.suspend(), torch.no_grad(), autorelease_scope(next(model.parameters()).device):
            with capture(model, [p.point for p in probes]) as captured:
                clean_value = float(score(forward_logits(model, clean, guard=guard))[0])
                cache = {p.point: captured[p.point].clone() for p in probes}
            captured.close(clear=True)
            corrupted_value = float(score(forward_logits(model, corrupted, guard=guard))[0])
            denominator = clean_value - corrupted_value
            if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
                raise ValueError("patch recovery is undefined because the clean metric equals the corrupted metric")
            recovered = []
            cells = [(probe, position) for probe in probes for position in range(positions)]
            for start in range(0, len(cells), 32):
                batch = cells[start:start + 32]
                with autorelease_scope(next(model.parameters()).device), ExitStack() as hooks:
                    for probe in probes:
                        selected = [(row, position) for row, (target, position) in enumerate(batch) if target == probe]
                        if not selected:
                            continue

                        def patch(module, args, kwargs, *, selected=selected, source=cache[probe.point]):
                            value = _input(args, kwargs).clone()
                            rows, columns = zip(*selected)
                            value[list(rows), list(columns)] = source[0, list(columns)]
                            if args:
                                return (value,) + args[1:], kwargs
                            key = "hidden_states" if "hidden_states" in kwargs else "x"
                            return args, dict(kwargs, **{key: value})

                        handle = modules[probe.module].register_forward_pre_hook(patch, with_kwargs=True)
                        hooks.callback(handle.remove)
                    inputs = {k: v.repeat(len(batch), 1) for k, v in corrupted.items()}
                    recovered.append((score(forward_logits(model, inputs, guard=guard)) - corrupted_value) / denominator)
            grid = torch.cat(recovered).reshape(len(probes), positions).to(torch.float32)
            if not torch.isfinite(grid).all():
                raise ValueError("patch recovery contains a nonfinite value")
            from . import publish
            publish("patch.grid", grid)
            return {"clean_metric": clean_value, "corrupted_metric": corrupted_value,
                    "normalizer": denominator, "recovery": grid.tolist(),
                    "layers": len(probes), "positions": positions, "binding": "live:patch.grid"}
    finally:
        cache.clear()
