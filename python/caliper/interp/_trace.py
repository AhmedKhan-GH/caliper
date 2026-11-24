"""Target-batched direct effects through prompt-local frozen value paths."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json

import torch

from ._frozen import FrozenPath, MemoryGuard
from ._memory import autorelease_scope
from ._graph import Graph, bounded_integer
from ._influence import influence


@dataclass
class _Target:
    node: int
    layer: int
    position: int
    direction: torch.Tensor
    expected: torch.Tensor
    encoder_bias: torch.Tensor
    unembedding_bias: torch.Tensor


class _DirectEffects:
    """Independent source writes and bounded vector-Jacobian products."""

    def __init__(self, path, dictionary, token_ids, tokenizer):
        self.path, self.dictionary = path, dictionary
        self.nodes, self.source_layers = [], []
        self.tokens = [{"position": p, "token_id": token, "text": tokenizer.decode([token])}
                       for p, token in enumerate(token_ids)]
        for token in self.tokens:
            self.nodes.append({"id": f"token:{token['position']}:{token['token_id']}", "kind": "token",
                               "layer": -1, **token, "activation": 1., "label": token["text"]})
        for layer, a in enumerate(path.features):
            with autorelease_scope(path.device):
                weights = dictionary.layers[layer]
                # Fixed k columns permit a batched decoder product without B by N by width expansion.
                indices = a[0].topk(weights.k, dim=-1).indices.sort(dim=-1).values
                activations = a[0].gather(-1, indices)
                active = activations > 0
                decoders = weights.W_dec.detach()[indices].to(path.dtype)
                positions = torch.arange(path.tokens, device=path.device)[:, None].expand_as(indices)
                start = len(self.nodes)
                for position, feature, activation in zip(positions[active].tolist(), indices[active].tolist(),
                                                          activations[active].tolist()):
                    self.nodes.append({"id": f"feature:{layer}:{position}:{feature}", "kind": "feature",
                                       "layer": layer, "position": position, "feature": feature,
                                       "activation": activation, "label": f"L{layer} F{feature}"})
                end = len(self.nodes)
                for position in range(path.tokens):
                    self.nodes.append({"id": f"error:{layer}:{position}", "kind": "error", "layer": layer,
                                       "position": position, "activation": 1., "label": f"L{layer} error"})
                self.source_layers.append((slice(start, end), slice(end, len(self.nodes)),
                                           decoders, activations, active))
        probabilities = path.logits[0].softmax(-1)
        order = torch.argsort(probabilities, descending=True, stable=True)[:10]
        count = min(10, int((probabilities[order].cumsum(0) < .95).sum()) + 1, len(order))
        self.logit_nodes = []
        for token, probability in zip(order[:count].tolist(), probabilities[order[:count]].tolist()):
            self.logit_nodes.append(len(self.nodes))
            self.nodes.append({"id": f"logit:{path.tokens - 1}:{token}", "kind": "logit", "layer": path.layers,
                               "position": path.tokens - 1, "token_id": token,
                               "activation": float(path.logits[0, token] - path.logits[0].mean()),
                               "probability": probability, "label": tokenizer.decode([token])})
        self.feature_nodes = [i for i, n in enumerate(self.nodes) if n["kind"] == "feature"]
        path.guard.check("source preparation")

    def target(self, index):
        node = self.nodes[index]
        path = self.path
        zero = path.embedding.new_zeros(())
        if node["kind"] == "logit":
            token = node["token_id"]
            bias = path.unembedding_bias
            return _Target(index, path.layers, path.tokens - 1,
                           path.unembedding[token] - path.unembedding_mean,
                           path.logits[0, token] - path.logits[0].mean(), zero,
                           zero if bias is None else bias[token] - bias.mean())
        if node["kind"] != "feature":
            raise ValueError("only features and logits are trace targets")
        layer, position, feature = node["layer"], node["position"], node["feature"]
        weights = self.dictionary.layers[layer]
        direction = weights.W_enc[feature].detach().to(path.dtype)
        bias = weights.b_enc[feature].detach().to(path.dtype)
        expected = path.inputs[layer][0, position] @ direction + bias
        return _Target(index, layer, position, direction, expected, bias, zero)

    def backward(self, indices):
        """Return all direct incoming edges and explicit bias accounting.

        Each batch row has one target and its own independent write leaves.
        A single backward computes all rows. Later feature activations cannot
        enter this path because ``FrozenPath.run`` never evaluates an encoder.
        """
        bounded_integer(len(indices), "target batch", 1, 64)
        path = self.path
        targets = [self.target(index) for index in indices]
        batch = len(targets)
        with torch.enable_grad():
            zero = path.embedding.new_zeros(batch, path.tokens, path.width)
            leaves = [zero.detach().requires_grad_(True) for _ in range(path.layers + 1)]
            reads, final = path.run(leaves[0], leaves[1:], bias_masks=(0., 0., 0.))
            objective = torch.stack([((final if target.layer == path.layers else reads[target.layer])
                                      [row, target.position] * target.direction).sum()
                                     for row, target in enumerate(targets)]).sum()
            path.guard.check("target forward")
            gradients = torch.autograd.grad(objective, leaves, allow_unused=True)
        path.guard.check("target backward")
        weights = path.embedding.new_zeros(len(self.nodes), batch)
        token_gradient = gradients[0]
        if token_gradient is not None:
            weights[:path.tokens] = (token_gradient * path.embedding).sum(-1).T
        for layer, (feature_slice, error_slice, decoders, activations, active) in enumerate(self.source_layers):
            gradient = gradients[layer + 1]
            if gradient is None:
                continue
            products = torch.bmm(gradient.transpose(0, 1), decoders.transpose(1, 2)).transpose(0, 1)
            products = products * activations
            weights[feature_slice] = products[:, active].T
            weights[error_slice] = (gradient * path.errors[layer]).sum(-1).T
        expected = torch.stack([target.expected for target in targets])
        affine = torch.stack([(path.bias_final if target.layer == path.layers else path.bias_reads[target.layer])
                              [:, target.position] @ target.direction for target in targets])
        encoder = torch.stack([target.encoder_bias for target in targets])
        unembedding = torch.stack([target.unembedding_bias for target in targets])
        incoming = weights.sum(0)
        residual = (incoming + affine.sum(-1) + encoder + unembedding - expected).abs()
        # One bounded readback per target batch avoids thousands of MPS scalar synchronizations.
        accounting = torch.cat((expected[:, None], incoming[:, None], affine, encoder[:, None],
                                unembedding[:, None], residual[:, None]), dim=1).tolist()
        diagnostics = []
        for target, row in zip(targets, accounting):
            biases = dict(zip(("decoder", "normalization", "attention", "encoder", "unembedding"), row[2:7]))
            diagnostics.append({"id": self.nodes[target.node]["id"], "expected": row[0],
                                "incoming": row[1], "biases": biases, "absolute_residual": row[7],
                                "relative_residual": row[7] / max(1., abs(row[0]))})
        path.guard.check("direct edge products")
        return weights, diagnostics


def _versions(module):
    return [(name, id(value), value._version, str(value.device), str(value.dtype))
            for name, value in list(module.named_parameters()) + list(module.named_buffers())]


def _prune(effects, weights, target_indices, scores, normalized, *, max_nodes, max_edges):
    nodes = effects.nodes
    logits = effects.logit_nodes
    device = weights.device
    rank = torch.argsort(scores, descending=True, stable=True).tolist()
    score_values = scores.tolist()
    top_effects = weights[:, 0].tolist()
    total = float(scores.sum())
    selected = set(logits[:max_nodes])
    retained = sum(score_values[i] for i in selected)
    for index in rank:
        if retained >= .8 * total or len(selected) >= max_nodes:
            break
        if index not in selected and score_values[index] > 0:
            selected.add(index)
            retained += score_values[index]
    # A real token edge can preserve a supported path without a synthetic link.
    required_edge = None
    if max_nodes >= 2 and max_edges and logits[0] in selected:
        token_edges = weights[:effects.path.tokens, 0].abs()
        token = int(torch.argmax(token_edges))
        if token_edges[token] > 0 and score_values[token] > 0:
            if token not in selected:
                if len(selected) == max_nodes:
                    removable = [i for i in selected if i != logits[0]]
                    selected.remove(min(removable, key=lambda i: (score_values[i], -i)))
                selected.add(token)
            required_edge = (token, logits[0])
    order = sorted(selected)
    selected_tensor = torch.tensor(order, device=device, dtype=torch.long)
    column_list = [j for j, i in enumerate(target_indices) if i in selected]
    column_tensor = torch.tensor(column_list, device=device, dtype=torch.long)
    subset = weights[selected_tensor][:, column_tensor]
    edge_scores = normalized[selected_tensor][:, column_tensor] * scores[torch.tensor(target_indices, device=device)][column_tensor]
    valid = subset != 0
    coordinates = valid.nonzero()
    values = subset[valid]
    importance = edge_scores[valid]
    edge_rank = torch.argsort(importance, descending=True, stable=True)
    edge_total = float(importance.sum())
    if edge_total > 0 and max_edges:
        ranked = importance[edge_rank]
        keep = min(max_edges, int((ranked.cumsum(0) < .98 * edge_total).sum()) + 1, len(edge_rank))
        chosen = edge_rank[:keep].tolist()
    else:
        chosen = []
    coordinate_list = coordinates.tolist()
    if required_edge is not None:
        required = next((i for i, (s, t) in enumerate(coordinate_list)
                         if (order[s], target_indices[column_list[t]]) == required_edge), None)
        if required is not None and required not in chosen:
            if len(chosen) == max_edges:
                chosen.pop()
            chosen.append(required)
    # Resolve a supported multihop path within the retained nodes when no direct token edge exists.
    if required_edge is None and max_edges:
        incoming = {}
        for i in edge_rank.tolist():
            s, t = coordinate_list[i]
            incoming.setdefault(target_indices[column_list[t]], []).append((order[s], i))
        def token_path(target, seen):
            if nodes[target]["kind"] == "token":
                return []
            if target in seen:
                return None
            for source, edge in incoming.get(target, []):
                found = token_path(source, seen | {target})
                if found is not None:
                    return found + [edge]
            return None
        path = token_path(logits[0], set()) if logits[0] in selected else None
        if path is not None and len(path) <= max_edges:
            chosen = path + [i for i in chosen if i not in path][:max_edges - len(path)]
    row = {old: new for new, old in enumerate(order)}
    display_nodes = [dict(nodes[i], index=row[i], influence=score_values[i], effect=top_effects[i]) for i in order]
    value_list = values.tolist()
    display_edges = []
    for index in chosen:
        source, target = coordinate_list[index]
        display_edges.append({"source": source, "target": row[target_indices[column_list[target]]],
                              "weight": value_list[index]})
    display_edges.sort(key=lambda e: (e["target"], e["source"]))
    nonlogit_total = float(scores[:logits[0]].sum())
    kept = sum(score_values[i] for i in order if i < logits[0])
    omitted = max(0., 1. - kept / nonlogit_total) if nonlogit_total else 0.
    global_edge_total = float((normalized.sum(0) * scores[torch.tensor(target_indices, device=device)]).sum())
    retained_edges = float(importance[chosen].sum()) if chosen else 0.
    edge_omitted = max(0., 1. - retained_edges / global_edge_total) if global_edge_total else 0.
    return display_nodes, display_edges, omitted, edge_omitted


class Tracer:
    """Trace the fixture or Qwen3.5 with an eight-entry state-keyed graph cache."""

    def __init__(self, model, tokenizer, dictionary):
        if dictionary is None:
            raise ValueError("trace requires a dictionary")
        self.model, self.tokenizer, self.dictionary = model, tokenizer, dictionary
        self.target_batch_size = 64
        self._cache = OrderedDict()
        self._forward_cache = None
        self._prompt = None
        self.guard = MemoryGuard(next(model.parameters()).device)

    def _service(self):
        from .._controls import CONTROLS
        service = getattr(CONTROLS, "intervention_service", None)
        return service if service is not None and service.model is self.model else None

    def _intervention_state(self):
        service = self._service()
        if service is not None:
            from ._intervene import selected_rows
            return {"specifications": service.specifications(), "rows": selected_rows(self.model)}
        from .._controls import CONTROLS
        with CONTROLS.lock:
            watched = CONTROLS._iv_model
            if watched is None or watched.model is not self.model:
                return []
            return [dict(iv.entry(), error=iv.error, clear=iv.clear) for iv in CONTROLS._ivs.values()]

    def _new_prompt(self, prompt):
        if prompt != self._prompt:
            self._forward_cache = None
            service = self._service()
            if service is not None:
                service.invalidate_source_cache()
            self._prompt = prompt

    @torch.inference_mode(False)
    def trace(self, prompt, max_targets=512, max_nodes=512, max_edges=8192):
        """Publish and return one graph without truncation or model downloads."""
        with autorelease_scope(next(self.model.parameters()).device):
            if self.model.training:
                raise ValueError("trace requires an evaluation model")
            device = next(self.model.parameters()).device
            if device != self.guard.device:
                self.guard = MemoryGuard(device)
                self._forward_cache = None
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string")
            bounded_integer(max_targets, "max_targets", 0, 512)
            bounded_integer(max_nodes, "max_nodes", 1, 512)
            bounded_integer(max_edges, "max_edges", 0, 8192)
            bounded_integer(self.target_batch_size, "target_batch_size", 1, 64)
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            ids = encoded["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("tokenizer must return one integer token sequence")
            bounded_integer(ids.shape[1], "prompt tokens including special tokens", 1, 32)
            if ids.min() < 0 or ids.max() >= self.model.config.vocab_size:
                raise ValueError("tokenizer returned an out-of-range token")
            self._new_prompt(prompt)
            identity = {"model": id(self.model), "model_weights": _versions(self.model),
                        "dictionary": id(self.dictionary), "dictionary_weights": _versions(self.dictionary),
                        "dictionary_metadata": self.dictionary.metadata,
                        "tokenizer": id(self.tokenizer), "prompt": prompt, "tokens": ids.tolist(),
                        "interventions": self._intervention_state()}
            state_key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            cache_key = (state_key, max_targets, max_nodes, max_edges)
            if cache_key in self._cache:
                graph = self._cache.pop(cache_key)
                self._cache[cache_key] = graph
                self.guard.check("cached publication")
                return graph.publish()
            if self._forward_cache is not None and self._forward_cache[0] == state_key:
                path = self._forward_cache[1]
            else:
                self._forward_cache = None
                path = FrozenPath(self.model, ids.to(self.guard.device), self.dictionary, guard=self.guard)
                if self._intervention_state() != identity["interventions"]:
                    raise RuntimeError("intervention state changed during the trace forward")
                self._forward_cache = (state_key, path)
            effects = _DirectEffects(path, self.dictionary, ids[0].tolist(), self.tokenizer)
            target_indices = list(effects.logit_nodes)
            budget = len(target_indices) + min(max_targets, len(effects.feature_nodes))
            weights = path.embedding.new_zeros(len(effects.nodes), budget)
            seeds = path.embedding.new_zeros(len(effects.nodes))
            for index in effects.logit_nodes:
                seeds[index] = effects.nodes[index]["probability"]
            diagnostics = []
            visited = torch.zeros(len(effects.nodes), dtype=torch.bool, device=path.device)
            feature_mask = visited.clone()
            feature_mask[effects.feature_nodes] = True
            completed = 0
            while completed < budget:
                with autorelease_scope(path.device):
                    size = self.guard.batch_size(self.target_batch_size, layers=path.layers, tokens=path.tokens,
                                                 width=path.width, mixer_width=path.mixer_width)
                    if completed < len(effects.logit_nodes):
                        batch = target_indices[completed:min(completed + size, len(effects.logit_nodes))]
                    else:
                        targets = torch.tensor(target_indices, dtype=torch.long, device=path.device)
                        estimated, _ = influence(weights[:, :completed], targets, seeds)
                        available = feature_mask & ~visited
                        ranked = torch.argsort(estimated.masked_fill(~available, -1), descending=True, stable=True)
                        count = min(size, budget - completed)
                        batch = ranked[:count].tolist()
                        target_indices.extend(batch)
                    incoming, accounting = effects.backward(batch)
                    weights[:, completed:completed + len(batch)] = incoming
                    diagnostics.extend(accounting)
                    visited[batch] = True
                    completed += len(batch)
                    self.guard.check("target batch complete")
            targets = torch.tensor(target_indices, dtype=torch.long, device=path.device)
            scores, normalized = influence(weights, targets, seeds)
            top_seed = torch.zeros_like(seeds)
            top_seed[effects.logit_nodes[0]] = 1
            top_scores, _ = influence(weights, targets, top_seed)
            source_total = float(top_scores[:effects.logit_nodes[0]].sum())
            feature_share = float(top_scores[feature_mask].sum()) / source_total if source_total else 0.
            error_mask = torch.tensor([n["kind"] == "error" for n in effects.nodes], device=path.device)
            error_share = float(top_scores[error_mask].sum()) / source_total if source_total else 0.
            display_nodes, display_edges, omitted, edge_omitted = _prune(
                effects, weights, target_indices, scores, normalized, max_nodes=max_nodes, max_edges=max_edges)
            completeness = {"max_absolute_residual": max(d["absolute_residual"] for d in diagnostics),
                            "max_relative_residual": max(d["relative_residual"] for d in diagnostics),
                            "forward_residual": path.forward_residual,
                            "direct_forward_residual": path.direct_forward_residual,
                            "replacement_forward_residual": path.replacement_forward_residual,
                            "target_count": len(diagnostics),
                            "feature_targets": len(target_indices) - len(effects.logit_nodes),
                            "logit_targets": len(effects.logit_nodes), "logit_probability_mass": float(seeds.sum()),
                            "unvisited_feature_influence": float(scores[feature_mask & ~visited].sum()),
                            "omitted_edge_influence": edge_omitted, "targets": diagnostics}
            tolerance = 1e-9 if path.dtype == torch.float64 else 2e-2
            if completeness["max_absolute_residual"] > tolerance:
                raise ValueError(f"direct-effect completeness residual exceeds {tolerance}")
            graph = Graph.build(nodes=display_nodes, edges=display_edges, tokens=effects.tokens,
                                layers=path.layers, completeness=completeness, omitted_influence=omitted,
                                error_share=error_share, feature_share=feature_share, trace_id=state_key,
                                device=path.device)
            self.guard.check("graph publication")
            if self._intervention_state() != identity["interventions"]:
                self._forward_cache = None
                raise RuntimeError("intervention state changed during the trace")
            graph.publish()
            self._cache[cache_key] = graph
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            return graph
