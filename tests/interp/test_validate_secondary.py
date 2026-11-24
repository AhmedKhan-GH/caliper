"""Independent secondary evidence preserves the original ten-feature score."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from caliper import _interpret_protocol as protocol
from caliper.interp import _validate
from caliper.interp._intervene import InterventionService
from caliper.interp._memory import CleanStop
from .test_intervene import explicit_forward
from .test_validate import validation_case


@pytest.fixture
def case():
    model, tokenizer, dictionary, controls, service = validation_case()
    ids = tokenizer("AB")["input_ids"]
    with torch.no_grad():
        logits, activations = explicit_forward(model, ids)
    token = int(logits[0, -1].argmax())
    direction = model.lm_head.weight[token] - model.lm_head.weight.mean(0)
    gradient = (model.norm.last_scale[0, -1] * model.norm.weight * direction).detach()
    nodes = [{"kind": "logit", "position": 2, "token_id": token, "probability": 1., "index": 0}]
    for position in (2, 1):
        for feature in activations[-1][0, position].nonzero().flatten().tolist():
            activation = float(activations[-1][0, position, feature])
            effect = float(activation * dictionary.layers[-1].W_dec[feature] @ gradient) if position == 2 else 0.
            nodes.append(dict(kind="feature", index=len(nodes), layer=3, position=position, feature=feature,
                              activation=activation, influence=100. - len(nodes), effect=effect))
            if len(nodes) == 11:
                break
        if len(nodes) == 11:
            break
    assert len(nodes) == 11
    # Equal influence and reversed row order distinguish the node-index tie rule.
    for layer, index in ((0, 21), (1, 20)):
        feature = int(activations[layer][0, 2].argmax())
        nodes.append(dict(kind="feature", index=index, layer=layer, position=2, feature=feature,
                          activation=float(activations[layer][0, 2, feature]), influence=1., effect=.25))
    graph = SimpleNamespace(nodes=nodes, tokens=[{"token_id": t} for t in ids[0].tolist()],
                            layers=4, trace_id="secondary-fixture")
    try:
        yield model, tokenizer, dictionary, controls, service, graph
    finally:
        service.remove()


def validate(case, graph=None, **kwargs):
    model, tokenizer, dictionary, _, service, original = case
    return _validate.validate_trace(model, tokenizer, dictionary, service, graph or original,
                                    prompt="AB", **kwargs)


def instrument_forwards(monkeypatch):
    forward = _validate.forward_logits
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return forward(*args, **kwargs)
    monkeypatch.setattr(_validate, "forward_logits", observed)
    return calls


def test_secondary_outside_ten_keeps_every_scored_result(case, monkeypatch):
    model, tokenizer, _, _, service, graph = case
    scored_only = deepcopy(graph)
    scored_only.nodes = scored_only.nodes[:11]
    previous = validate(case, scored_only)
    assert previous["secondary_measurement"] is None
    calls = instrument_forwards(monkeypatch)
    state = service.state_dict()
    result = validate(case)
    assert len(calls) == 12  # One baseline, ten scored rows, one independent row.
    assert service.state_dict() == state
    assert {k: v for k, v in result.items() if k != "secondary_measurement"} == {
        k: v for k, v in previous.items() if k != "secondary_measurement"}
    row = result["secondary_measurement"]
    node = graph.nodes[-1]
    assert all(row[key] == node[key] for key in ("layer", "position", "feature"))
    assert row not in result["measurements"] and result["count"] == 10
    assert set(row) == {"layer", "position", "feature", "predicted", "measured", "sign_match"}
    assert row["predicted"] == -node["effect"]
    def ablate(layer, activations):
        if layer == node["layer"]:
            activations = activations.clone()
            activations[0, node["position"], node["feature"]] = 0
        return activations
    with torch.no_grad():
        edited, _ = explicit_forward(model, tokenizer("AB")["input_ids"], edit=ablate)
    logits = edited[0, result["position"]]
    actual = float(logits[result["token"]] - logits.mean()) - result["baseline_centered_logit"]
    assert row["measured"] == pytest.approx(actual, abs=1e-10)
    assert row["sign_match"] == ((row["predicted"] > 0) - (row["predicted"] < 0)
                                == (actual > 0) - (actual < 0))
    assert protocol.bounded(result)
    assert len(json.dumps(result, allow_nan=False).encode()) < protocol.MAX_BYTES


def test_secondary_reuses_an_existing_scored_measurement(case, monkeypatch):
    graph = case[-1]
    graph.nodes[-1]["influence"] = 200.
    calls = instrument_forwards(monkeypatch)
    result = validate(case)
    assert len(calls) == 11
    assert result["secondary_measurement"] is result["measurements"][0]
    assert result["count"] == 10


def test_no_eligible_feature_adds_no_forward(case, monkeypatch):
    graph = case[-1]
    graph.nodes = graph.nodes[:11]
    calls = instrument_forwards(monkeypatch)
    result = validate(case)
    assert result["secondary_measurement"] is None and len(calls) == 11


def test_zero_activation_is_still_eligible_for_secondary_evidence(case, monkeypatch):
    model, tokenizer, _, _, _, graph = case
    with torch.no_grad():
        _, activations = explicit_forward(model, tokenizer("AB")["input_ids"])
    node = graph.nodes[-1]
    node.update(feature=int((activations[1][0, 2] == 0).nonzero()[0]), activation=0., effect=0.)
    calls = instrument_forwards(monkeypatch)
    result = validate(case)
    row = result["secondary_measurement"]
    assert row["feature"] == node["feature"] and row["measured"] == row["predicted"] == 0
    assert row["sign_match"] and len(calls) == 12
    assert result["count"] == 10 and all(r["layer"] == 3 for r in result["measurements"])


@pytest.mark.parametrize("failure", [None, "forward", "guard"])
def test_secondary_uses_owned_executor_guard_and_cleanup(case, monkeypatch, failure):
    model, _, _, controls, service, _ = case
    service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "AB"})
    service.set({"kind": "scale", "point": "layers.2.mlp_out", "feature": 0,
                 "multiplier": 1., "from_step": 20})
    controls.step()
    state, cache = service.state_dict(), dict(service.executor._cache)
    hooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    stages = []
    def check(stage):
        stages.append(stage)
        if failure == "guard" and stages.count("inference forward start") == 12:
            raise CleanStop("secondary guard refusal")
    guard = SimpleNamespace(limit=20_000_000, peak=1, device=torch.device("cpu"), check=check)
    installed, executors = [], []
    install = InterventionService.install
    def owned(executor, spec, **kwargs):
        assert executor is not service.executor and executor.resource_guard is guard
        executors.append(executor)
        hook = install(executor, spec, **kwargs)
        installed.append(hook)
        return hook
    monkeypatch.setattr(InterventionService, "install", owned)
    forward = _validate.forward_logits
    def fail_secondary(*args, **kwargs):
        if failure == "forward" and len(installed) == 11:
            raise RuntimeError("secondary forward failure")
        return forward(*args, **kwargs)
    monkeypatch.setattr(_validate, "forward_logits", fail_secondary)
    if failure:
        with pytest.raises(CleanStop if failure == "guard" else RuntimeError, match="secondary"):
            validate(case, guard=guard)
    else:
        assert validate(case, guard=guard)["secondary_measurement"] is not None
    assert len(installed) == 11 and all(hook.removed for hook in installed)
    assert all(executor is executors[0] for executor in executors)
    # The test retains hook references, thus the executor's WeakSet can retain removed hooks.
    assert all(hook.removed for hook in executors[0]._hooks) and not executors[0]._cache
    assert executors[0]._dictionary_snapshot is None and executors[0]._cache_identity is None
    assert hooks == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    assert service.state_dict() == state and controls.n == 1
    assert service.executor._cache.keys() == cache.keys()
    assert all(service.executor._cache[key] is value for key, value in cache.items())
