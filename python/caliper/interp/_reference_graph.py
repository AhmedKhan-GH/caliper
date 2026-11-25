"""Independent graph arithmetic for the circuit_tracer_82996f14 profile."""
from __future__ import annotations

import math

import torch

from ._memory import autorelease_scope


PROFILE = "circuit_tracer_82996f14"


def fraction(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number from zero to one")
    return float(value)


def propagate(weights, targets, seeds, *, partial=False, check=lambda stage: None):
    """Propagate through source rows, without the initial logit seed.

    Float32 CPU storage matches the reference graph workspace. The target
    vector maps each compact column to its source row. No inverse is needed.
    """
    if weights.dtype != torch.float32 or weights.device.type != "cpu":
        raise ValueError("reference influence requires float32 CPU weights")
    if (weights.ndim != 2 or seeds.shape != (len(weights),)
            or targets.shape != (weights.shape[1],) or targets.dtype != torch.long
            or seeds.dtype != weights.dtype or seeds.device != weights.device
            or targets.device != weights.device):
        raise ValueError("reference influence dimensions or coordinates differ")
    if (not torch.isfinite(weights).all() or not torch.isfinite(seeds).all()
            or (seeds < 0).any() or targets.unique().numel() != targets.numel()
            or (targets.numel() and (targets.min() < 0 or targets.max() >= len(weights)))):
        raise ValueError("reference influence requires finite weights and unique valid targets")
    # Target-major contiguous storage preserves the reference reduction order.
    normalized = weights.T.contiguous().abs()
    normalized /= normalized.sum(1, keepdim=True).clamp_min(1e-8 if partial else 1e-10)
    product = seeds
    total = torch.zeros_like(seeds)
    for _ in range(128 if partial else 1001):
        check("reference influence")
        product = product[targets] @ normalized
        if not product.any():
            return total, normalized.T
        total += product
    raise RuntimeError("reference influence did not converge")


def cutoff(scores, threshold):
    """Return the inclusive score cutoff, including threshold ties."""
    ordered = torch.sort(scores.flatten(), descending=True).values
    if not len(ordered):
        raise ValueError("reference threshold requires nonempty scores")
    cumulative = ordered.cumsum(0) / ordered.sum()
    index = int(torch.searchsorted(cumulative, threshold))
    nonzero = int((ordered > 0).sum())
    return ordered[min(index, max(nonzero - 1, 0))]


def prune(nodes, weights, targets, seeds, *, node_threshold, edge_threshold,
          max_nodes, max_edges, guard, check):
    """Apply node, reduced-edge, support, and serializer filters in order.

    Nodes use reference order: selected features, errors, tokens, then logits.
    The square CPU workspace includes absent target rows. This also preserves
    the pinned zero-cutoff behavior for graphs with no positive influence.
    """
    count = len(nodes)
    # Matrix copies, score sort values and indices, masks, and Python export.
    guard.reserve("reference pruning workspace", count * count * 40 + count * 2048)
    adjacency = torch.zeros((count, count), dtype=torch.float32)
    adjacency[targets] = weights.T
    all_targets = torch.arange(count)
    scores, normalized = propagate(adjacency.T, all_targets, seeds, check=check)
    del normalized
    keep = scores >= cutoff(scores, node_threshold)
    feature = torch.tensor([n["kind"] == "feature" for n in nodes])
    error = torch.tensor([n["kind"] == "error" for n in nodes])
    fixed = ~(feature | error)
    keep[fixed] = True
    reduced = adjacency.clone()
    reduced[~keep] = 0
    reduced[:, ~keep] = 0
    reduced_scores, reduced_normalized = propagate(reduced.T, all_targets, seeds, check=check)
    edge_scores = reduced_normalized.T * (reduced_scores + seeds)[:, None]
    edge_mask = edge_scores >= cutoff(edge_scores.flatten(), edge_threshold)
    # Match the pinned sequence, including its first support test before masks.
    while True:
        check("reference support pruning")
        previous = keep.clone()
        keep[feature | error] &= edge_mask[:, feature | error].any(0)
        keep[feature] &= edge_mask[feature].any(1)
        if torch.equal(previous, keep):
            break
        edge_mask[~keep] = False
        edge_mask[:, ~keep] = False
    # The serializer only exports edges whose endpoints survived node pruning.
    edge_mask[~keep] = False
    edge_mask[:, ~keep] = False
    connected = edge_mask.any(0) | edge_mask.any(1)
    keep &= connected | fixed
    node_count, edge_count = int(keep.sum()), int(edge_mask.sum())
    if node_count > max_nodes or edge_count > max_edges:
        raise ValueError(f"complete reference graph exceeds resource ceilings: {node_count} nodes, {edge_count} edges")
    guard.reserve("reference graph records", node_count * 2048 + edge_count * 512)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative = torch.zeros_like(scores)
    # Empty influence has no finite cumulative rank. Keep the absence explicit.
    has_influence = float(sorted_scores.sum()) > 0
    if has_influence:
        cumulative[sorted_indices] = sorted_scores.cumsum(0) / sorted_scores.sum()
    order = keep.nonzero().flatten().tolist()
    row = {old: new for new, old in enumerate(order)}
    top = next(i for i, n in enumerate(nodes) if n["kind"] == "logit")
    kept_nodes = [dict(nodes[i], index=row[i], influence=float(scores[i]),
                       cumulative_influence=float(cumulative[i]) if has_influence else None,
                       effect=float(adjacency[top, i]))
                  for i in order]
    endpoints = edge_mask.nonzero().tolist()
    raw_weights = adjacency[edge_mask].tolist()
    edges = [{"source": row[source], "target": row[target], "weight": weight}
             for (target, source), weight in zip(endpoints, raw_weights)]
    total = float(scores.sum())
    omitted = max(0., 1 - float(scores[keep].sum()) / total) if total else 0.
    edge_total = float(edge_scores.sum())
    edge_omitted = max(0., 1 - float(edge_scores[edge_mask].sum()) / edge_total) if edge_total else 0.
    source_total = float(scores.sum())
    feature_share = float(scores[feature].sum()) / source_total if source_total else 0.
    error_share = float(scores[error].sum()) / source_total if source_total else 0.
    return kept_nodes, edges, omitted, edge_omitted, feature_share, error_share


def build(effects, *, max_targets, max_nodes, max_edges, batch_size, update_interval,
          node_threshold, edge_threshold, check):
    """Select reference targets with logical queues and bounded physical chunks."""
    path, guard = effects.path, effects.path.guard
    # Retained decoder rows must count toward every physical batch admission.
    effects.prepare_decoders()
    # The feature order matches sparse COO coordinates: layer, position, feature.
    features = [i for i in effects.feature_nodes if effects.nodes[i]["position"] != 0]
    errors = [i for i, n in enumerate(effects.nodes) if n["kind"] == "error"]
    tokens = [i for i, n in enumerate(effects.nodes) if n["kind"] == "token"]
    order = features + errors + tokens + effects.logit_nodes
    inverse = {node: i for i, node in enumerate(order)}
    budget = len(effects.logit_nodes) + min(max_targets, len(features))
    guard.reserve("reference edge workspace", len(order) * (budget * 12 + 64 * 8 + 64))
    weights = torch.zeros((len(order), budget), dtype=torch.float32)
    seeds = torch.zeros(len(order), dtype=torch.float32)
    seeds[-len(effects.logit_nodes):] = torch.tensor(
        [effects.nodes[i]["probability"] for i in effects.logit_nodes])
    targets, diagnostics, physical_batches, logical_queues = [], [], [], []
    selected = set()
    source_rows = torch.tensor(order, device=path.device)
    bos_errors = [inverse[i] for i in errors if effects.nodes[i]["position"] == 0]

    def evaluate(queue):
        logical_queues.append([effects.nodes[i]["id"] for i in queue])
        # Logical batches remain separate even if admission splits them further.
        for logical_start in range(0, len(queue), batch_size):
            logical = queue[logical_start:logical_start + batch_size]
            offset = 0
            while offset < len(logical):
                check("reference batch admission")
                size = guard.batch_size(min(64, len(logical) - offset), layers=path.layers,
                    tokens=path.tokens, width=path.width, mixer_width=path.mixer_width,
                    itemsize=path.embedding.element_size())
                batch = logical[offset:offset + size]
                with autorelease_scope(path.device):
                    incoming, accounting = effects.backward(batch)
                    projected = incoming[source_rows].float().cpu()
                    projected[bos_errors] = 0
                    begin = len(targets)
                    weights[:, begin:begin + len(batch)] = projected
                    del incoming, projected
                targets.extend(inverse[i] for i in batch)
                diagnostics.extend(accounting)
                physical_batches.append(len(batch))
                offset += len(batch)
                check("reference batch complete")

    evaluate(effects.logit_nodes)
    requested = min(max_targets, len(features))
    while len(selected) < requested:
        if requested == len(features):
            pending = list(range(len(features)))
        else:
            estimated, _ = propagate(weights[:, :len(targets)], torch.tensor(targets), seeds,
                                     partial=True, check=check)
            ranked = torch.argsort(estimated[:len(features)], descending=True).tolist()
            pending = [i for i in ranked if i not in selected][:min(
                update_interval * batch_size, requested - len(selected))]
        evaluate([features[i] for i in pending])
        selected.update(pending)
    effects.release_decoders()
    # Remove unselected feature columns before final influence and pruning.
    retained = sorted(selected) + list(range(len(features), len(order)))
    mapping = {old: new for new, old in enumerate(retained)}
    projected = weights[retained]
    all_incoming = weights.sum(0)
    selected_incoming = projected.sum(0)
    excluded_incoming = (all_incoming - selected_incoming).tolist()
    selected_incoming = selected_incoming.tolist()
    for column, diagnostic in enumerate(diagnostics):
        check("reference projection diagnostics")
        diagnostic["reference_projection"] = {
            "selected_source_incoming": selected_incoming[column],
            "unselected_feature_incoming": excluded_incoming[column],
            "completeness_scope": "projection_only"}
    projected_targets = torch.tensor([mapping[i] for i in targets])
    projected_nodes = [effects.nodes[order[i]] for i in retained]
    result = prune(projected_nodes, projected, projected_targets, seeds[retained],
                   node_threshold=node_threshold, edge_threshold=edge_threshold,
                   max_nodes=max_nodes, max_edges=max_edges, guard=guard, check=check)
    return result, diagnostics, [order[i] for i in targets], physical_batches, logical_queues
