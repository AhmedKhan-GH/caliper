"""Absolute incoming normalization and backward influence propagation."""
from __future__ import annotations

import torch


def normalized_incoming(weights):
    """Normalize each target column, with zero columns left at zero."""
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("incoming weights must be a floating point matrix")
    if not torch.isfinite(weights).all():
        raise ValueError("incoming weights must be finite")
    absolute = weights.abs()
    total = absolute.sum(dim=0, keepdim=True)
    return absolute / total.clamp_min(torch.finfo(weights.dtype).tiny)


def influence(weights, targets, seeds, *, steps=32):
    """Return source influence for an N by M direct-edge matrix.

    Column j contains edges into node ``targets[j]``. Seeds hold logit
    probabilities. The recurrence is ``v = seeds + P @ v[targets]``.
    Thus influence travels from a target to its sources, including mixed signs.
    No square adjacency matrix or inverse is allocated.
    """
    if type(steps) is not int or steps != 32:
        raise ValueError("influence requires 32 power steps")
    if seeds.ndim != 1 or seeds.shape[0] != weights.shape[0]:
        raise ValueError("seeds must contain one value per source node")
    if targets.ndim != 1 or targets.shape[0] != weights.shape[1] or targets.dtype != torch.long:
        raise ValueError("targets must contain one integer node index per column")
    if targets.device != weights.device or seeds.device != weights.device:
        raise ValueError("influence tensors must share one device")
    if targets.numel() and (targets.min() < 0 or targets.max() >= len(seeds)):
        raise ValueError("target index is out of bounds")
    if targets.unique().numel() != targets.numel():
        raise ValueError("target columns must be unique")
    if not torch.isfinite(seeds).all() or (seeds < 0).any():
        raise ValueError("logit seeds must be finite and nonnegative")
    normalized = normalized_incoming(weights)
    value = seeds.clone()
    for _ in range(steps):
        value = seeds + normalized @ value[targets]
    return value, normalized
