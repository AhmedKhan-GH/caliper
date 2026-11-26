"""Validate traced direct feature effects against real feature ablations."""
from __future__ import annotations

import math

import torch


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(predicted: list[float], measured: list[float]) -> float | None:
    """Return tie-corrected rank correlation, or None for zero variance."""
    if len(predicted) < 2:
        return None
    x, y = _ranks(predicted), _ranks(measured)
    mean = (len(x) - 1) / 2.0
    x = [v - mean for v in x]
    y = [v - mean for v in y]
    denom = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(a * b for a, b in zip(x, y)) / denom if denom else None


def _demeaned_logit_objective(model, logits: torch.Tensor, *, position: int, token_id: int) -> float:
    """Compute demeaned logit before output softcap when softcap is active."""
    row = logits[0, position].float()
    softcap = float(getattr(getattr(model, "cfg", None), "output_logits_soft_cap", 0.0) or 0.0)
    if softcap > 0.0:
        clamped = (row / softcap).clamp(-0.999999, 0.999999)
        row = softcap * torch.atanh(clamped)
    return float((row[token_id] - row.mean()).item())


def validate_trace(backend, interventions, graph, *, prompt: str, guard=None) -> dict:
    """Zero-clamp up to ten influential active features and correlate with ``-effect``."""
    if backend.dictionary is None:
        raise ValueError("validation requires a loaded dictionary")
    ids = backend.tokenize(prompt)
    if ids[0].cpu().tolist() != [t["token_id"] for t in graph.tokens]:
        raise ValueError("validation prompt does not match the trace")
    targets = [n for n in graph.nodes if n["kind"] == "logit"]
    if not targets:
        raise ValueError("validation requires a traced logit target")
    top = max(targets, key=lambda n: (n["probability"], -n["token_id"]))
    position, token_id = int(top["position"]), int(top["token_id"])

    features = sorted(
        (n for n in graph.nodes if n["kind"] == "feature" and n["activation"] != 0),
        key=lambda n: (-n["influence"], n["layer"], n["position"], n["feature"]),
    )[:10]

    specs = interventions.specifications() if interventions is not None else []
    feature_specs = [s for s in specs if s.get("op") in ("clamp", "scale")]
    base_tuples = backend.specs_to_tuples(ids[0], feature_specs) if feature_specs else []

    base_logits, _ = backend.intervene_logits(ids[0], base_tuples, freeze_attention=True, guard=guard)
    baseline = _demeaned_logit_objective(backend.model, base_logits, position=position, token_id=token_id)

    measurements = []
    for node in features:
        layer, pos, feat = int(node["layer"]), int(node["position"]), int(node["feature"])
        clamped_logits, _ = backend.intervene_logits(
            ids[0], [*base_tuples, (layer, pos, feat, 0.0)], freeze_attention=True, guard=guard
        )
        measured = _demeaned_logit_objective(backend.model, clamped_logits, position=position, token_id=token_id) - baseline
        predicted = -float(node["effect"])
        if not math.isfinite(predicted) or not math.isfinite(measured):
            raise ValueError("validation produced a nonfinite measurement")
        sign_match = ((predicted > 0) - (predicted < 0)) == ((measured > 0) - (measured < 0))
        measurements.append({
            "layer": layer,
            "position": pos,
            "feature": feat,
            "predicted": predicted,
            "measured": measured,
            "sign_match": bool(sign_match),
        })

    correlation = spearman([r["predicted"] for r in measurements], [r["measured"] for r in measurements])
    signs = sum(1 for row in measurements if row["sign_match"])
    reasons = []
    if len(measurements) < 10:
        reasons.append("insufficient_active_features")
    if correlation is None:
        reasons.append("zero_variance")
    elif correlation < 0.6:
        reasons.append("spearman_below_0.6")
    if signs < 8:
        reasons.append("fewer_than_eight_correct_signs")

    return {
        "trace_id": graph.trace_id,
        "token": token_id,
        "position": position,
        "baseline_centered_logit": baseline,
        "count": len(measurements),
        "required": 10,
        "spearman": correlation,
        "sign_agreement": signs / len(measurements) if measurements else None,
        "correct_signs": signs,
        "passed": not reasons,
        "reasons": reasons,
        "measurements": measurements,
    }
