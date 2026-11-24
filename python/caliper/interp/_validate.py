"""Compare graph effects with centered logits after real feature ablations."""
import math
from contextlib import contextmanager

import torch

from ._frozen import _trace_features
from ._dictionary import _canonical_device
from ._memory import BudgetMemory, MemoryGuard, autorelease_scope, family_memory_limit
from ._output import forward_logits, prompt_inputs
from ._points import _architecture, capture
from ._intervene import InterventionError, InterventionService, selected_rows


def _validation_guard(model, dictionary, interventions, guard, memory_limit_gb=None):
    owner = getattr(interventions, "executor", None)
    if (guard is None and owner is not None and owner.model is model and owner.dictionary is dictionary
            and owner.resource_guard is not None):
        guard = owner.resource_guard
    limit = family_memory_limit(model, memory_limit_gb, guard)
    device = next(model.parameters()).device
    if any(_canonical_device(value.device) != _canonical_device(device)
           for value in dictionary.state_dict().values()):
        raise ValueError("validation dictionary and model must share one device")
    if guard is not None:
        if _canonical_device(guard.device) != _canonical_device(device):
            raise ValueError("validation requires a memory guard for its model device")
        return BudgetMemory(guard, limit)
    if memory_limit_gb is None and device.type == "cpu":
        return None
    return MemoryGuard(limit, device)


@contextmanager
def _validation_executor(model, tokenizer, dictionary, interventions, guard):
    """Retain exact content evidence in a private executor for this call."""
    owner = getattr(interventions, "executor", None)
    shared = (owner is not None and owner.model is model and owner.dictionary is dictionary
              and owner._default_dictionary_id)
    temporary = InterventionService(model)
    try:
        temporary.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=guard,
                            dictionary_id=owner.dictionary_id if shared else None)
        if shared:
            # The identifier is valid only with full content checks. Borrow the
            # owner's immutable evidence, never its hooks or patch source cache.
            temporary._default_dictionary_id = True
            temporary._dictionary_snapshot = owner._dictionary_snapshot
        if temporary._dictionary_snapshot is None:
            temporary._prepare_dictionary_identity()
        if not temporary._dictionary_matches():
            raise InterventionError("dictionary content changed. Reconfigure before use",
                                    "intervene.unknown_dictionary")
        yield temporary
    finally:
        temporary.close()


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(predicted, measured):
    """Return tie-corrected rank correlation, or None for zero variance."""
    if len(predicted) < 2:
        return None
    x, y = _ranks(predicted), _ranks(measured)
    mean = (len(x) - 1) / 2
    x, y = [v - mean for v in x], [v - mean for v in y]
    denominator = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(a * b for a, b in zip(x, y)) / denominator if denominator else None


def validate_trace(model, tokenizer, dictionary, interventions, graph, *, prompt, guard=None, memory_limit_gb=None):
    """Zero-clamp up to ten influential features at their traced positions.

    Temporary output hooks leave the intervention lifecycle and caches intact.
    Ordered output resets, scale effects, and clamp effects define the live value.
    An independent secondary measurement retains the original demo selection.
    It never contributes to the ten-feature score.
    """
    rows = selected_rows(model)
    if rows is not None and 0 not in rows:
        raise ValueError("validation requires interventions on row zero")
    guard = _validation_guard(model, dictionary, interventions, guard, memory_limit_gb)
    if guard is not None:
        guard.check("validation admission")
    inputs = prompt_inputs(model, tokenizer, prompt)
    if inputs["input_ids"][0].tolist() != [t["token_id"] for t in graph.tokens]:
        raise ValueError("validation prompt does not match the trace")
    targets = [n for n in graph.nodes if n["kind"] == "logit"]
    if not targets:
        raise ValueError("validation requires a traced logit target")
    top = max(targets, key=lambda n: (n["probability"], -n["token_id"]))
    position, token_id = top["position"], top["token_id"]
    features = sorted((n for n in graph.nodes if n["kind"] == "feature" and n["activation"] != 0),
                      key=lambda n: (-n["influence"], n["layer"], n["position"], n["feature"]))[:10]
    layers = getattr(graph, "layers", model.config.num_hidden_layers)
    secondary = max(((n, index) for index, n in enumerate(graph.nodes)
                     if n["kind"] == "feature" and n["layer"] < layers - 1),
                    key=lambda item: (item[0]["influence"], -item[0].get("index", item[1])), default=None)
    secondary = secondary[0] if secondary is not None else None
    architecture = _architecture(model)
    probes = {p.layer: p for p in architecture.values() if p.point.endswith(".mlp_out")}
    specifications = interventions.specifications()
    if (any(s["op"] in ("scale", "clamp") for s in specifications)
            and interventions.dictionary is not dictionary):
        raise ValueError("feature interventions require the same dictionary instance as validation")
    modules = dict(model.named_modules())

    def write_module(point):
        probe = architecture.get(point)
        if probe is not None:
            return modules[probe.module] if probe.event == "output" else None
        # Legacy module points write to the module output. Head points differ.
        try:
            return model.get_submodule(point)
        except AttributeError:
            return None

    effective_points = [(spec, write_module(spec["point"])) for spec in specifications]

    def centered(logits):
        row = logits[0, -1 if position == inputs["input_ids"].shape[1] - 1 else position]
        return float(row[token_id] - row.mean())

    read_points = {layer: probe.point.rsplit(".", 1)[0] + ".mlp_in" for layer, probe in probes.items()}
    captured = capture(model, list(read_points.values()))
    try:
        with _validation_executor(model, tokenizer, dictionary, interventions, guard) as temporary, \
                autorelease_scope(next(model.parameters()).device):
            with captured:
                baseline = centered(forward_logits(model, inputs, last_token=position == inputs["input_ids"].shape[1] - 1, guard=guard))
            def measure(node):
                with autorelease_scope(next(model.parameters()).device):
                    layer, feature, selected = node["layer"], node["feature"], node["position"]
                    probe = probes[layer]
                    related = [s for s, module in effective_points if module is modules[probe.module]
                               and selected in s.get("positions", range(inputs["input_ids"].shape[1]))]
                    weights = dictionary.layers[layer]

                    x = captured[read_points[layer]][0, selected].to(device=weights.W_enc.device)
                    with torch.no_grad():
                        activation = float(_trace_features(weights, x, guard)[feature])
                    coefficient, constant = 1., 0.
                    for spec in related:
                        if spec["op"] in ("patch", "ablate"):
                            coefficient, constant = 0., 0.
                        elif spec.get("feature") == feature:
                            if spec["op"] == "clamp":
                                coefficient -= 1
                                constant += spec["value"]
                            elif spec["op"] == "scale":
                                coefficient += spec["multiplier"] - 1
                    effective = coefficient * activation + constant
                    precision = torch.finfo(x.dtype).eps
                    if not math.isclose(float(node["activation"]), effective, rel_tol=64 * precision, abs_tol=64 * precision):
                        raise ValueError("validation feature differs from its current output contribution; retrace first")
                    hook = None
                    try:
                        hook = temporary.install({"kind": "clamp", "point": probe.point, "feature": feature,
                                                  "positions": [selected], "value": activation - effective})
                        measured = centered(forward_logits(model, inputs, last_token=position == inputs["input_ids"].shape[1] - 1, guard=guard)) - baseline
                        if hook.error is not None:
                            raise ValueError("validation clamp failed: " + hook.error)
                    finally:
                        if hook is not None:
                            hook()
                    predicted = -float(node["effect"])
                    if not math.isfinite(predicted) or not math.isfinite(measured):
                        raise ValueError("validation produced a nonfinite measurement")
                    return {"layer": layer, "position": selected, "feature": feature,
                            "predicted": predicted, "measured": measured,
                            "sign_match": (predicted > 0) - (predicted < 0) == (measured > 0) - (measured < 0)}
            measurements = [measure(node) for node in features]
            correlation = spearman([r["predicted"] for r in measurements], [r["measured"] for r in measurements])
            signs = sum(row["sign_match"] for row in measurements)
            reasons = []
            if len(measurements) < 10:
                reasons.append("insufficient_active_features")
            if correlation is None:
                reasons.append("zero_variance")
            elif correlation < .6:
                reasons.append("spearman_below_0.6")
            if signs < 8:
                reasons.append("fewer_than_eight_correct_signs")
            secondary_measurement = None
            if secondary is not None:
                secondary_measurement = next((row for row in measurements
                    if all(row[key] == secondary[key] for key in ("layer", "position", "feature"))), None)
                if secondary_measurement is None:
                    secondary_measurement = measure(secondary)
            return {"trace_id": graph.trace_id, "token": token_id, "position": position,
                    "baseline_centered_logit": baseline, "count": len(measurements), "required": 10,
                    "spearman": correlation, "sign_agreement": signs / len(measurements) if measurements else None,
                    "correct_signs": signs, "passed": not reasons, "reasons": reasons, "measurements": measurements,
                    "secondary_measurement": secondary_measurement}
    finally:
        captured.close(clear=True)
