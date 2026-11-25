"""Target-batched direct effects through prompt-local frozen value paths."""
from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time

import torch

from ._dictionary import _HASH_CHUNK_BYTES, _canonical_device, _state_hash, weights_hash
from ._families import model_family
from ._forward_identity import arithmetic_settings
from . import _qwen_family
from ._frozen import (FrozenPath, MemoryGuard, _validate_dictionary, _validate_ablation_points,
                      _validate_intervention_positions, _finite_max, _require_finite, _no_check,
                      _DictionaryAdmission, _ENCODER_BLOCK_ELEMENTS)
from ._memory import BudgetMemory, CleanStop, allocation_bytes, autorelease_scope, family_memory_limit
from ._graph import Graph, bounded_integer
from ._influence import influence
from ._labels import token_label
from ._feature_labels import label_graph_nodes


class TraceTimeout(TimeoutError):
    """One trace exceeds its request budget without a process resource stop."""

    code = "interpret.trace_timeout"


class _Deadline:
    def __init__(self):
        self.expires = time.monotonic() + 20

    def __call__(self, stage):
        if time.monotonic() >= self.expires:
            raise TraceTimeout(f"Trace exceeded the 20 second request budget at {stage}.")


class _DeadlineMemory:
    """Check the deadline at each bounded identity hash chunk."""

    def __init__(self, memory, check):
        self.memory, self.deadline = memory, check

    def __getattr__(self, name):
        return getattr(self.memory, name)

    def check(self, stage, **kwargs):
        self.deadline(stage)
        self.memory.check(stage, **kwargs)
        self.deadline(stage)


class _SuppliedTraceGuard(MemoryGuard):
    """Apply supplied lower limits to trace reserves and full identity checks."""

    def __init__(self, device, *, memory_limit_gb, memory):
        if _canonical_device(memory.device) != _canonical_device(device):
            raise ValueError("trace requires a memory guard for its model device")
        memory.check("trace supplied guard")
        super().__init__(device, memory_limit_gb=memory_limit_gb)
        self._guard = BudgetMemory(self._guard, memory_limit_gb, extra=memory)
        self.check("trace supplied budget")

    @property
    def limit(self):
        return min(self._limit, getattr(getattr(self, "_guard", None), "limit", self._limit))

    @limit.setter
    def limit(self, value):
        self._limit = value

    def check(self, stage):
        return max(super().check(stage), allocation_bytes(self._guard))


@dataclass
class _Target:
    node: int
    layer: int
    position: int
    direction: torch.Tensor
    expected: torch.Tensor
    encoder_bias: torch.Tensor
    unembedding_bias: torch.Tensor
    intervention_bias: torch.Tensor


class _DirectEffects:
    """Independent source writes and bounded vector-Jacobian products."""

    def __init__(self, path, dictionary, token_ids, tokenizer, *, check=_no_check):
        self.path, self.dictionary = path, dictionary
        self.check = check
        self.nodes, self.source_layers = [], []
        self.feature_columns = {}
        count = path.tokens * (path.layers + 1) + 10
        count += sum(int((output.activations != 0).sum()) for output in path.feature_outputs)
        # Python node dictionaries and their lookup tables also use process memory.
        # Refuse an oversized active set explicitly. Never truncate JumpReLU sources.
        path.guard.reserve("source nodes", count * 2048)
        self.tokens = [{"position": p, "token_id": token, "text": token_label(tokenizer, token)}
                       for p, token in enumerate(token_ids)]
        for token in self.tokens:
            self.nodes.append({"id": f"token:{token['position']}:{token['token_id']}", "kind": "token",
                               "layer": -1, **token, "activation": 1., "label": token["text"]})
        for layer, output in enumerate(path.feature_outputs):
            with autorelease_scope(path.device):
                check(f"source layer {layer}")
                # Active columns and explicit edits retain signed and constant writes.
                indices = output.indices
                activations = output.activations
                active = activations != 0
                positions = torch.arange(path.tokens, device=path.device)[:, None].expand_as(indices)
                start = len(self.nodes)
                columns = torch.arange(indices.shape[1], device=path.device)[None].expand_as(indices)
                for position, feature, activation, column in zip(
                        positions[active].tolist(), indices[active].tolist(),
                        activations[active].tolist(), columns[active].tolist()):
                    self.feature_columns[len(self.nodes)] = column
                    self.nodes.append({"id": f"feature:{layer}:{position}:{feature}", "kind": "feature",
                                       "layer": layer, "position": position, "feature": feature,
                                       "activation": activation, "label": f"L{layer} F{feature}"})
                end = len(self.nodes)
                for position in range(path.tokens):
                    self.nodes.append({"id": f"error:{layer}:{position}", "kind": "error", "layer": layer,
                                       "position": position, "activation": 1., "label": f"L{layer} error"})
                self.source_layers.append((slice(start, end), slice(end, len(self.nodes)),
                                           indices, activations, active))
        probabilities = path.logits[0].softmax(-1)
        order = torch.argsort(probabilities, descending=True, stable=True)[:10]
        count = min(10, int((probabilities[order].cumsum(0) < .95).sum()) + 1, len(order))
        self.logit_nodes = []
        for token, probability in zip(order[:count].tolist(), probabilities[order[:count]].tolist()):
            self.logit_nodes.append(len(self.nodes))
            self.nodes.append({"id": f"logit:{path.tokens - 1}:{token}", "kind": "logit", "layer": path.layers,
                               "position": path.tokens - 1, "token_id": token,
                               "activation": float(path.logits[0, token] - path.logits[0].mean()),
                               "probability": probability, "label": token_label(tokenizer, token)})
        self.feature_nodes = [i for i, n in enumerate(self.nodes) if n["kind"] == "feature"]
        path.guard.check("source preparation")
        check("source preparation")

    def target(self, index):
        node = self.nodes[index]
        path = self.path
        zero = path.embedding.new_zeros(())
        if node["kind"] == "logit":
            token = node["token_id"]
            direction, bias = path.logit_target(token)
            return _Target(index, path.layers, path.tokens - 1, direction,
                           path.logits[0, token] - path.logits[0].mean(), zero, bias, zero)
        if node["kind"] != "feature":
            raise ValueError("only features and logits are trace targets")
        layer, position, feature = node["layer"], node["position"], node["feature"]
        weights = self.dictionary.layers[layer]
        output = path.feature_outputs[layer]
        column = self.feature_columns[index]
        coefficient = output.coefficients[position, column]
        direction = coefficient * weights.W_enc[feature].detach().to(path.dtype)
        bias = coefficient * weights.b_enc[feature].detach().to(path.dtype)
        constant = output.constants[position, column]
        expected = output.activations[position, column]
        return _Target(index, layer, position, direction, expected, bias, zero, constant)

    def backward(self, indices):
        """Return all direct incoming edges and explicit bias accounting.

        Each batch row has one target and its own independent write leaves.
        A single backward computes all rows. Later feature activations cannot
        enter this path because ``FrozenPath.run`` never evaluates an encoder.
        """
        bounded_integer(len(indices), "target batch", 1, 64)
        self.check("target batch admission")
        path = self.path
        targets = [self.target(index) for index in indices]
        batch = len(targets)
        with torch.enable_grad():
            zero = path.embedding.new_zeros(batch, path.tokens, path.width)
            leaves = [zero.detach().requires_grad_(True) for _ in range(path.layers + 1)]
            reads, final = path.run(leaves[0], leaves[1:], bias_masks=(0., 0., 0., 0.), check=self.check)
            objective = torch.stack([((final if target.layer == path.layers else reads[target.layer])
                                      [row, target.position] * target.direction).sum()
                                     for row, target in enumerate(targets)]).sum()
            path.guard.check("target forward")
            self.check("target forward")
            gradients = torch.autograd.grad(objective, leaves, allow_unused=True)
        path.guard.check("target backward")
        self.check("target backward")
        path.guard.reserve("direct edge matrix", len(self.nodes) * batch * path.embedding.element_size())
        weights = path.embedding.new_zeros(len(self.nodes), batch)
        token_gradient = gradients[0]
        if token_gradient is not None:
            weights[:path.tokens] = (token_gradient * path.embedding).sum(-1).T
        for layer, (feature_slice, error_slice, indices, activations, active) in enumerate(self.source_layers):
            self.check(f"edge layer {layer}")
            gradient = gradients[layer + 1]
            if gradient is None:
                continue
            # JumpReLU can activate every column. Copy only bounded decoder
            # blocks instead of retaining token-by-feature-by-width tensors.
            rows = self.dictionary.layers[layer].W_dec.detach()
            step = max(1, _ENCODER_BLOCK_ELEMENTS // path.width)
            node = feature_slice.start
            for position in range(path.tokens):
                selected = indices[position, active[position]]
                values = activations[position, active[position]]
                for start in range(0, len(selected), step):
                    with autorelease_scope(path.device):
                        self.check("direct decoder block")
                        decoder = rows[selected[start:start + step]].to(path.dtype)
                        products = (gradient[:, position] @ decoder.T) * values[start:start + step]
                        weights[node:node + decoder.shape[0]] = products.T
                        node += decoder.shape[0]
            weights[error_slice] = (gradient * path.errors[layer]).sum(-1).T
        expected = torch.stack([target.expected for target in targets])
        affine = torch.stack([(path.bias_final if target.layer == path.layers else path.bias_reads[target.layer])
                              [:, target.position] @ target.direction for target in targets])
        affine[:, 3] += torch.stack([target.intervention_bias for target in targets])
        encoder = torch.stack([target.encoder_bias for target in targets])
        unembedding = torch.stack([target.unembedding_bias for target in targets])
        incoming = weights.sum(0)
        residual = (incoming + affine.sum(-1) + encoder + unembedding - expected).abs()
        _require_finite(residual, "direct-effect completeness residual", self.check)
        # One bounded readback per target batch avoids thousands of MPS scalar synchronizations.
        accounting = torch.cat((expected[:, None], incoming[:, None], affine, encoder[:, None],
                                unembedding[:, None], residual[:, None]), dim=1).tolist()
        diagnostics = []
        for target, row in zip(targets, accounting):
            biases = dict(zip(("decoder", "normalization", "attention", "intervention", "encoder", "unembedding"), row[2:8]))
            diagnostics.append({"id": self.nodes[target.node]["id"], "expected": row[0],
                                "incoming": row[1], "biases": biases, "absolute_residual": row[8],
                                "relative_residual": row[8] / max(1., abs(row[0]))})
        path.guard.check("direct edge products")
        self.check("direct edge products")
        return weights, diagnostics


def _versions(module):
    return [(name, id(value), value._version, str(value.device), str(value.dtype))
            for name, value in list(module.named_parameters()) + list(module.named_buffers())]


def _forward_settings(model, memory):
    """Snapshot settings read by the fixture and Qwen text forward paths.

    Bind runtime arithmetic separately from checkpoint and dictionary hashes.
    Capture temporarily selects eager attention. The configured backend does not
    affect the frozen path. Recorded outputs such as ``last_scale`` are not inputs.
    """
    fields = (
        "training", "eps", "variance_epsilon", "frozen_scale", "frozen_pattern",
        # Fixture attention and TopK, then Qwen full attention.
        "num_heads", "n_heads", "n_head", "head_dim", "in_features", "k",
        "num_key_value_groups", "scaling", "attention_dropout",
        # Qwen block routing, DeltaNet dimensions, and both SiLU gates.
        *_qwen_family.ROUTING_FIELDS, *_qwen_family.DELTA_HEAD_FIELDS,
        *_qwen_family.DELTA_DIM_FIELDS, *_qwen_family.DELTA_CONV_FIELDS, "activation",
        # Rotary factors and the embedding's optional weight normalization.
        "rope_type", "attention_scaling", *_qwen_family.ROTARY_FIELDS, "max_seq_len_cached",
        "original_max_seq_len", "long_inv_freq", "max_norm", "norm_type",
        "padding_idx", "scale_grad_by_freq", "sparse", "inplace",
        # Gemma attention routing, softcaps, and scaled embedding.
        "layer_type", "sliding_window", "attn_logit_softcapping", "scalar_embed_scale", "embed_scale",
    )
    config_fields = ("model_type", "num_hidden_layers", "hidden_size", "vocab_size",
                     "layer_types", "is_causal", "rope_parameters", "max_position_embeddings",
                     "head_dim", "num_attention_heads", "num_key_value_heads",
                     "sliding_window", "attn_logit_softcapping", "final_logit_softcapping",
                     "query_pre_attn_scalar", "use_bidirectional_attention", "hidden_activation")
    settings = arithmetic_settings(model)

    def record(name, value):
        if isinstance(value, torch.Tensor):
            # Fixture overrides are plain tensors. Read bytes to catch alias writes.
            value = {"tensor": _state_hash({name: value}, memory=memory), "device": str(value.device)}
        settings[name] = json.dumps(value, sort_keys=True, allow_nan=False)

    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        record(prefix + "type", type(module).__module__ + "." + type(module).__qualname__)
        for field in fields:
            if hasattr(module, field):
                record(prefix + field, getattr(module, field))
        if hasattr(module, "act_fn"):
            # The Qwen MLP calls this object, not config.hidden_act.
            record(prefix + "act_fn", id(module.act_fn))
        config = getattr(module, "config", None)
        for field in config_fields:
            if hasattr(config, field):
                record(prefix + "config." + field, getattr(config, field))
    return settings


def _token_path(nodes, incoming, target):
    """Find a shortest token path with at most one visit per reachable node."""
    pending = deque([target])
    next_step = {target: None}
    while pending:
        node = pending.popleft()
        if nodes[node]["kind"] == "token":
            path = []
            while next_step[node] is not None:
                node, edge = next_step[node]
                path.append(edge)
            return path
        for source, edge in incoming.get(node, ()):
            if source not in next_step:
                next_step[source] = (node, edge)
                pending.append(source)
    return None


def _supported_token_path(effects, weights, target_indices):
    """Find a shortest path before display caps remove its intermediate features.

    Only visited features can have incoming edges. Thus the CPU search holds at
    most 544 sources and 513 targets, independent of the full source count.
    Edge magnitude and then source index resolve ties. Each node enters once.
    """
    top_logit = effects.logit_nodes[0]
    columns = [column for column, target in enumerate(target_indices)
               if target == top_logit or effects.nodes[target]["kind"] == "feature"]
    sources = sorted(set(range(effects.path.tokens)) | {
        target for target in target_indices if effects.nodes[target]["kind"] == "feature"})
    rows = torch.tensor(sources, device=weights.device, dtype=torch.long)
    selected_columns = torch.tensor(columns, device=weights.device, dtype=torch.long)
    compact = weights[rows][:, selected_columns].abs().T.tolist()
    incoming = {}
    for column, values in zip(columns, compact):
        target = target_indices[column]
        ranked = sorted((i for i, value in enumerate(values) if value > 0),
                        key=lambda i: (-values[i], sources[i]))
        incoming[target] = [(sources[i], (sources[i], target)) for i in ranked]
    return _token_path(effects.nodes, incoming, top_logit)


def _prune(effects, weights, target_indices, scores, normalized, *, max_nodes, max_edges):
    """Reserve a supported path, then apply influence thresholds within caps.

    If caps cannot hold the shortest path, retain the top logit and ranked
    sources without a connectivity guarantee. The result reports that limit.
    An unavailable path means no path exists through the visited targets.
    Secondary logits never consume the last source slot when influence exists.
    """
    nodes = effects.nodes
    logits = effects.logit_nodes
    device = weights.device
    rank = torch.argsort(scores, descending=True, stable=True).tolist()
    score_values = scores.tolist()
    top_effects = weights[:, 0].tolist()
    # Terminal logit seeds weight source influence but do not count as sources.
    total = float(scores[:logits[0]].sum())
    path = _supported_token_path(effects, weights, target_indices)
    path_nodes = {index for edge in path for index in edge} if path is not None else set()
    fits = path is not None and len(path_nodes) <= max_nodes and len(path) <= max_edges
    connectivity = {"status": "retained" if fits else "unavailable" if path is None else "capped",
                    "minimum_nodes": len(path_nodes) if path is not None else None,
                    "minimum_edges": len(path) if path is not None else None}
    selected = {logits[0]}
    if fits:
        selected.update(path_nodes)
    elif max_nodes > 1:
        source = next((i for i in rank if i < logits[0] and score_values[i] > 0), None)
        if source is not None:
            selected.add(source)
    for index in logits[1:]:
        if len(selected) >= max_nodes:
            break
        selected.add(index)
    retained = sum(score_values[i] for i in selected if i < logits[0])
    for index in rank:
        if retained >= .8 * total or len(selected) >= max_nodes:
            break
        if index < logits[0] and index not in selected and score_values[index] > 0:
            selected.add(index)
            retained += score_values[index]
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
    if fits:
        required_pairs = set(path)
        required = [i for i, (s, t) in enumerate(coordinate_list)
                    if (order[s], target_indices[column_list[t]]) in required_pairs]
        required_indices = set(required)
        chosen = required + [i for i in chosen if i not in required_indices][:max_edges - len(required)]
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
    return display_nodes, display_edges, omitted, edge_omitted, connectivity


class Tracer:
    """Trace a supported decoder with an eight-entry state-keyed graph cache.

    ``controls`` supplies the model's intervention owner. Standalone callers use
    the global owner when this keyword is absent.
    """

    def __init__(self, model, tokenizer, dictionary, *, controls=None, index=None, memory_limit_gb=None, memory=None):
        self._initialize(model, tokenizer, dictionary, controls=controls, index=index,
                         memory_limit_gb=memory_limit_gb, memory=memory, eager_admission=True)

    @classmethod
    def _for_service(cls, model, tokenizer, dictionary, **kwargs):
        """Defer the constructor scan to the service's first full request.

        Each request validates after tokenizer callbacks and before cache access
        or a forward. No validation result survives from construction.
        """
        tracer = cls.__new__(cls)
        tracer._initialize(model, tokenizer, dictionary, eager_admission=False, **kwargs)
        return tracer

    def _initialize(self, model, tokenizer, dictionary, *, eager_admission, controls=None, index=None,
                    memory_limit_gb=None, memory=None):
        if dictionary is None:
            raise ValueError("trace requires a dictionary")
        self.memory_limit_gb = family_memory_limit(model, memory_limit_gb, memory)
        self._memory = memory
        self.guard = self._new_guard(next(model.parameters()).device)
        if eager_admission:
            _validate_dictionary(model, dictionary, self._dictionary_check, memory=self.guard)
        if controls is None:
            from .._controls import CONTROLS
            controls = CONTROLS
        self.model, self.tokenizer, self.dictionary = model, tokenizer, dictionary
        self.controls = controls
        self.bind_index(index)
        self.target_batch_size = 64
        self._cache = OrderedDict()
        self._forward_cache = None
        self._prompt = None
        self._settings = _forward_settings(model, self.guard._guard)

    def _new_guard(self, device):
        if self._memory is None:
            return MemoryGuard(device, memory_limit_gb=self.memory_limit_gb)
        return _SuppliedTraceGuard(device, memory_limit_gb=self.memory_limit_gb, memory=self._memory)

    def _dictionary_check(self, stage):
        if stage == "dictionary admission":
            self.guard.reserve(stage, 4 * _HASH_CHUNK_BYTES)
        elif stage == "model_identity":
            current = self.guard._check_interval(stage)
            if current + 4 * _HASH_CHUNK_BYTES > self.guard.limit:
                raise CleanStop("model identity allocation exceeds the trace memory limit")
        elif stage.startswith("finite block"):
            self.guard._guard.check(stage, force=False)
        else:
            self.guard.check(stage)

    def bind_index(self, index):
        """Bind heuristic labels to the same immutable trace resources."""
        if index is not None:
            for name in ("model", "tokenizer", "dictionary"):
                if getattr(index, name, None) is not getattr(self, name):
                    raise ValueError("trace feature index must share the same " + name)
            index.label_identity  # Refuse closed or invalid indexes before trace work.
        self.index = index

    def _validate_forward_settings(self, check=_no_check):
        try:
            check("forward settings")
            current = _forward_settings(self.model, _DeadlineMemory(self.guard._guard, check))
            check("forward settings complete")
            if current != self._settings:
                changed = next(name for name in sorted(current.keys() | self._settings.keys())
                               if current.get(name) != self._settings.get(name))
                raise ValueError(f"trace model forward setting changed after admission: {changed}")
        except (TypeError, ValueError):
            self._cache.clear()
            self._forward_cache = None
            raise

    def _service(self):
        from ._intervene import _ACTIVE_REPLAYS
        for replay in reversed(_ACTIVE_REPLAYS.get()):
            if replay.model is self.model:
                return replay
        service = getattr(self.controls, "intervention_service", None)
        if service is None:
            owner = getattr(self.controls, "_iv_model", None)
            service = owner if callable(getattr(owner, "specifications", None)) else None
        return service if service is not None and service.model is self.model else None

    def _intervention_state(self, check=_no_check):
        from ._intervene import _ACTIVE_REPLAYS, _MUTED_HOOKS, _source_input_identity, selected_rows
        muted = _MUTED_HOOKS.get()
        service = self._service()
        if service is not None:
            if service in _ACTIVE_REPLAYS.get():
                specifications = [deepcopy(spec) for spec, hook in service._effects
                                  if hook not in muted and not getattr(hook, "removed", False)]
            else:
                with self.controls.lock:
                    specifications = [deepcopy(iv.applied_spec) for iv in self.controls._ivs.values()
                                      if iv.state == "active" and iv.applied_spec is not None
                                      and iv.hook is not None and iv.hook not in muted
                                      and not getattr(iv.hook, "removed", False)]
            feature_effect = any(spec.get("op", spec.get("kind")) in ("scale", "clamp") for spec in specifications)
            patch_effect = any(spec.get("op", spec.get("kind")) == "patch" for spec in specifications)
            bound = service.dictionary if feature_effect else None
            rows = selected_rows(self.model)
            if feature_effect and (rows is None or 0 in rows) and bound is not self.dictionary:
                raise ValueError("feature interventions require the same dictionary instance as the trace")
            source_inputs = []
            if patch_effect:
                executor = service.executor
                memory = _DeadlineMemory(BudgetMemory(self.guard._guard, self.guard.limit / 1e9,
                                                     extra=executor.resource_guard), check)
                prompts = {}
                for spec in specifications:
                    if spec.get("op", spec.get("kind")) == "patch":
                        prompts.setdefault(spec["source_prompt"], spec)
                # Tokenization runs outside the controls lock. Its current result
                # must enter both cache keys before either cache can return.
                for prompt, spec in sorted(prompts.items()):
                    memory.check("trace patch source inputs")
                    _, digest = _source_input_identity(executor._source_inputs(spec), memory)
                    memory.check("trace patch source inputs complete")
                    source_inputs.append((prompt, digest))
            return {"specifications": specifications, "rows": rows,
                    "dictionary": None if bound is None else (id(bound), _versions(bound)),
                    "tokenizer": id(service.executor.tokenizer) if patch_effect else None,
                    "source_inputs": source_inputs}
        with self.controls.lock:
            watched = self.controls._iv_model
            if watched is None or watched.model is not self.model:
                return []
            return [deepcopy(iv.applied_spec) for iv in self.controls._ivs.values()
                    if iv.state == "active" and iv.hook is not None and iv.applied_spec is not None
                    and iv.hook not in muted and not getattr(iv.hook, "removed", False)]

    def _new_prompt(self, prompt):
        if prompt != self._prompt:
            self._forward_cache = None
            service = self._service()
            if service is not None:
                service.executor.invalidate_cache()
            self._prompt = prompt

    @torch.inference_mode(False)
    def trace(self, prompt, max_targets=512, max_nodes=512, max_edges=8192):
        """Publish and return one graph without truncation or model downloads."""
        from ._publish import _publication_scope
        with _publication_scope(self.model):
            return self._trace(prompt, max_targets, max_nodes, max_edges)

    def _trace(self, prompt, max_targets, max_nodes, max_edges):
        check = _Deadline()
        label_index = self.index
        label_key = None if label_index is None else label_index.label_identity
        check("trace admission")

        state = self._intervention_state(check)
        specifications = state["specifications"] if isinstance(state, dict) else state
        if (specifications and isinstance(state, dict) and state["rows"] is not None
                and any(row != 0 for row in state["rows"])):
            raise ValueError("trace intervention rows must fit its single prompt")
        if isinstance(state, dict) and state["rows"] is not None and 0 not in state["rows"]:
            specifications = []
        _validate_ablation_points(self.model, specifications)
        with autorelease_scope(next(self.model.parameters()).device), ExitStack() as admission_scope:
            if self.model.training:
                raise ValueError("trace requires an evaluation model")
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string")
            bounded_integer(max_targets, "max_targets", 0, 512)
            bounded_integer(max_nodes, "max_nodes", 1, 512)
            bounded_integer(max_edges, "max_edges", 0, 8192)
            bounded_integer(self.target_batch_size, "target_batch_size", 1, 64)
            self.guard.check("tokenization admission")
            check("tokenization admission")
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            check("tokenization")
            ids = encoded["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("tokenizer must return one integer token sequence")
            bounded_integer(ids.shape[1], "prompt tokens including special tokens", 1, 32)
            if ids.min() < 0 or ids.max() >= self.model.config.vocab_size:
                raise ValueError("tokenizer returned an out-of-range token")
            _validate_intervention_positions(ids.shape[1], specifications)
            device = next(self.model.parameters()).device
            if device != self.guard.device:
                self.guard = self._new_guard(device)
                self._forward_cache = None
            try:
                def dictionary_check(stage):
                    check(stage)
                    self._dictionary_check(stage)
                # Prompt and patch-source tokenizers can change bytes through aliases.
                # Share validation only after those callbacks, within this request.
                admission = admission_scope.enter_context(_DictionaryAdmission(
                    self.model, self.dictionary, dictionary_check, memory=self.guard))
            except (TypeError, ValueError):
                self._cache.clear()
                self._forward_cache = None
                raise
            self._new_prompt(prompt)
            content = weights_hash(self.dictionary, memory=_DeadlineMemory(
                self.guard._guard, lambda stage: check("dictionary content " + stage)))
            check("dictionary content complete")
            identity = {"model": id(self.model), "model_weights": _versions(self.model),
                        "dictionary": id(self.dictionary), "dictionary_weights": _versions(self.dictionary),
                        "dictionary_content": content,
                        "dictionary_metadata": self.dictionary.metadata,
                        "tokenizer": id(self.tokenizer), "prompt": prompt, "tokens": ids.tolist(),
                        "interventions": state}
            identity.update(arithmetic_settings(self.model))
            state_key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            cache_key = (state_key, max_targets, max_nodes, max_edges, self.target_batch_size, label_key)
            self._validate_forward_settings(check)
            if cache_key in self._cache:
                graph = self._cache[cache_key]
                self.guard.check("cached publication")
                check("cached publication")
                graph.publish()
                self._cache.move_to_end(cache_key)
                return graph
            if self._forward_cache is not None and self._forward_cache[0] == state_key:
                path = self._forward_cache[1]
            else:
                self._forward_cache = None
                def check_forward(stage):
                    check(stage)
                    # A failed live hook invalidates the recording before frozen arithmetic.
                    if stage == "record complete" and self._intervention_state(check) != identity["interventions"]:
                        raise RuntimeError("intervention state changed during the trace forward")
                path = FrozenPath(self.model, ids.to(self.guard.device), self.dictionary,
                                  guard=self.guard, interventions=specifications, check=check_forward,
                                  _admission=admission)
                self._validate_forward_settings(check)
                if self._intervention_state(check) != identity["interventions"]:
                    raise RuntimeError("intervention state changed during the trace forward")
            check("forward complete")
            effects = _DirectEffects(path, self.dictionary, ids[0].tolist(), self.tokenizer, check=check)
            target_indices = list(effects.logit_nodes)
            budget = len(target_indices) + min(max_targets, len(effects.feature_nodes))
            # Influence retains normalized edges and work tensors beside the matrix.
            path.guard.reserve("trace edge matrices", len(effects.nodes) *
                               (4 * budget + 4 * self.target_batch_size + 32) * path.embedding.element_size())
            weights = path.embedding.new_zeros(len(effects.nodes), budget)
            seeds = path.embedding.new_zeros(len(effects.nodes))
            for index in effects.logit_nodes:
                seeds[index] = effects.nodes[index]["probability"]
            diagnostics, batch_sizes = [], []
            visited = torch.zeros(len(effects.nodes), dtype=torch.bool, device=path.device)
            feature_mask = visited.clone()
            feature_mask[effects.feature_nodes] = True
            completed = 0
            while completed < budget:
                with autorelease_scope(path.device):
                    check("target selection")
                    size = self.guard.batch_size(self.target_batch_size, layers=path.layers, tokens=path.tokens,
                                                 width=path.width, mixer_width=path.mixer_width,
                                                 itemsize=path.embedding.element_size())
                    if completed < len(effects.logit_nodes):
                        batch = target_indices[completed:min(completed + size, len(effects.logit_nodes))]
                    else:
                        targets = torch.tensor(target_indices, dtype=torch.long, device=path.device)
                        estimated, _ = influence(weights[:, :completed], targets, seeds)
                        check("target influence")
                        available = feature_mask & ~visited
                        ranked = torch.argsort(estimated.masked_fill(~available, -1), descending=True, stable=True)
                        count = min(size, budget - completed)
                        batch = ranked[:count].tolist()
                        target_indices.extend(batch)
                    incoming, accounting = effects.backward(batch)
                    for field in ("absolute_residual", "relative_residual"):
                        _finite_max((row[field] for row in accounting), f"direct-effect {field}")
                    batch_sizes.append(len(batch))
                    weights[:, completed:completed + len(batch)] = incoming
                    diagnostics.extend(accounting)
                    visited[batch] = True
                    completed += len(batch)
                    self.guard.check("target batch complete")
                    check("target batch complete")
            targets = torch.tensor(target_indices, dtype=torch.long, device=path.device)
            scores, normalized = influence(weights, targets, seeds)
            check("graph influence")
            top_seed = torch.zeros_like(seeds)
            top_seed[effects.logit_nodes[0]] = 1
            top_scores, _ = influence(weights, targets, top_seed)
            check("top logit influence")
            source_total = float(top_scores[:effects.logit_nodes[0]].sum())
            feature_share = float(top_scores[feature_mask].sum()) / source_total if source_total else 0.
            error_mask = torch.tensor([n["kind"] == "error" for n in effects.nodes], device=path.device)
            error_share = float(top_scores[error_mask].sum()) / source_total if source_total else 0.
            display_nodes, display_edges, omitted, edge_omitted, connectivity = _prune(
                effects, weights, target_indices, scores, normalized, max_nodes=max_nodes, max_edges=max_edges)
            check("graph pruning")
            completeness = {"max_absolute_residual": _finite_max(
                                (d["absolute_residual"] for d in diagnostics), "direct-effect absolute residual"),
                            "max_relative_residual": _finite_max(
                                (d["relative_residual"] for d in diagnostics), "direct-effect relative residual"),
                            "forward_residual": path.forward_residual,
                            "direct_forward_residual": path.direct_forward_residual,
                            "replacement_forward_residual": path.replacement_forward_residual,
                            "target_count": len(diagnostics),
                            "feature_targets": len(target_indices) - len(effects.logit_nodes),
                            "logit_targets": len(effects.logit_nodes), "logit_probability_mass": float(seeds.sum()),
                            "unvisited_feature_influence": float(scores[feature_mask & ~visited].sum()),
                            "omitted_edge_influence": edge_omitted, "token_path": connectivity,
                            "targets": diagnostics}
            tolerance = 1e-9 if path.dtype == torch.float64 else 2e-2
            if completeness["max_absolute_residual"] > tolerance:
                raise ValueError(f"direct-effect completeness residual exceeds {tolerance}")
            display_nodes = label_graph_nodes(display_nodes, label_index, check=check)
            graph_data = dict(nodes=display_nodes, edges=display_edges, tokens=effects.tokens,
                              layers=path.layers, completeness=completeness, omitted_influence=omitted,
                              error_share=error_share, feature_share=feature_share,
                              logits=[effects.nodes[index] for index in effects.logit_nodes])
            # Resource admission can change batch boundaries and thus visited targets.
            # Hash the request and the actual result without changing target selection.
            trace_identity = {"format": "caliper.trace.v2", "request": cache_key,
                              "targets": [effects.nodes[index]["id"] for index in target_indices],
                              "batches": batch_sizes, "graph": graph_data}
            trace_id = hashlib.sha256(json.dumps(trace_identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            graph = Graph.build(**graph_data, trace_id=trace_id, device=path.device)
            check("graph construction")
            self.guard.check("graph publication")
            if self._intervention_state(check) != identity["interventions"]:
                self._forward_cache = None
                raise RuntimeError("intervention state changed during the trace")
            self._validate_forward_settings(check)
            # Admit publication and both caches together. No timeout follows this boundary.
            check("graph publication")
            graph.publish()
            self._forward_cache = (state_key, path)
            self._cache[cache_key] = graph
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            return graph
