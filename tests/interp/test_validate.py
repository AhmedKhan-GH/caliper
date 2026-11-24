"""Centered-logit measurements and restoration of pre-existing effects."""
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp._validate import spearman, validate_trace
from .fixture import make_fixture
from .test_dictionary import fixture_manifest
from .test_intervene import dictionary_for, explicit_forward


def validation_case():
    model, tokenizer = make_fixture()
    dictionary = dictionary_for(model)
    dictionary.metadata = fixture_manifest(model)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    return model, tokenizer, dictionary, controls, service


def frozen_graph(model, tokenizer, prompt):
    ids = tokenizer(prompt)["input_ids"]
    logits, activations = explicit_forward(model, ids)
    position = ids.shape[1] - 1
    token = int(logits[0, position].argmax())
    direction = model.lm_head.weight[token] - model.lm_head.weight.mean(0)
    gradient = model.norm.last_scale[0, position] * model.norm.weight * direction
    # Last-layer finite changes have an exact independent frozen-norm oracle.
    active = activations[-1][0, position].nonzero().flatten().tolist()
    nodes = [{"kind": "logit", "position": position, "token_id": token, "probability": 1.}]
    for rank, feature in enumerate(active):
        a = float(activations[-1][0, position, feature].detach())
        nodes.append({"kind": "feature", "layer": 3, "position": position, "feature": feature,
                      "activation": a, "influence": len(active) - rank,
                      "effect": float((a * model.layers[-1].mlp.W_dec[feature] @ gradient).detach())})
    return SimpleNamespace(nodes=nodes, trace_id="fixture", tokens=[{"token_id": t} for t in ids[0].tolist()])


def test_validation_matches_centered_prediction_at_selected_position():
    model, tokenizer, dictionary, controls, service = validation_case()
    graph = frozen_graph(model, tokenizer, "AB")
    try:
        with model.frozen_attention(norms=True):
            result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert result["count"] == 8
        assert result["correct_signs"] == 8 and result["sign_agreement"] == 1
        assert result["spearman"] == pytest.approx(1)
        assert result["reasons"] == ["insufficient_active_features"]
        assert not result["passed"]
        for row in result["measurements"]:
            assert row["position"] == 2
            assert row["measured"] == pytest.approx(row["predicted"], abs=1e-6)
        assert service.state_dict()["interventions"] == []
        assert controls.n == 0
    finally:
        service.remove()


def test_validation_preserves_active_pending_and_ended_state_on_failure(monkeypatch):
    model, tokenizer, dictionary, controls, service = validation_case()
    graph = frozen_graph(model, tokenizer, "AB")
    service.set({"kind": "add", "point": "layers.0.resid_pre", "vector": [0.] * 64, "coefficient": 0.})
    service.set({"kind": "scale", "point": "layers.1.mlp_out", "feature": 0, "multiplier": 1., "from_step": 20})
    service.set({"kind": "ablate", "point": "layers.2.mlp", "until_step": 2})
    controls.step()
    before = service.state_dict()
    assert {r["state"] for r in before["interventions"]} == {"active", "pending", "ended"}
    counts = {name: (len(m._forward_hooks), len(m._forward_pre_hooks)) for name, m in model.named_modules()}
    original = model.forward
    call = 0
    def fail(*a, **kw):
        nonlocal call
        call += 1
        if call == 2:
            raise RuntimeError("fixture validation failure")
        return original(*a, **kw)
    monkeypatch.setattr(model, "forward", fail)
    try:
        with pytest.raises(RuntimeError, match="fixture validation failure"):
            validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert before == service.state_dict()
        assert counts == {name: (len(m._forward_hooks), len(m._forward_pre_hooks)) for name, m in model.named_modules()}
        assert controls.n == 1
    finally:
        service.remove()


def test_validation_uses_ten_features_and_reports_failed_criteria():
    model, tokenizer, dictionary, controls, service = validation_case()
    from caliper.interp import Tracer
    try:
        graph = Tracer(model, tokenizer, dictionary).trace("AB", max_targets=16)
        result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        expected_count = min(10, sum(n["kind"] == "feature" and n["activation"] > 0 for n in graph.nodes))
        assert result["count"] == expected_count
        assert len(result["measurements"]) == expected_count
        assert all(row["predicted"] == -next(n["effect"] for n in graph.nodes if
                   n["kind"] == "feature" and (n["layer"], n["position"], n["feature"]) ==
                   (row["layer"], row["position"], row["feature"])) for row in result["measurements"])
        assert result["passed"] == (expected_count == 10 and result["correct_signs"] >= 8
                                     and result["spearman"] is not None and result["spearman"] >= .6)
        with pytest.raises(ValueError, match="does not match"):
            validate_trace(model, tokenizer, dictionary, service, graph, prompt="CD")
    finally:
        service.remove()


def test_spearman_handles_ties_and_zero_variance():
    assert spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1], [2, 3]) is None
    assert spearman([], []) is None


def test_validation_measures_ten_distinct_feature_positions():
    model, tokenizer, dictionary, controls, service = validation_case()
    graph = frozen_graph(model, tokenizer, "AB")
    _, activations = explicit_forward(model, tokenizer("AB")["input_ids"])
    for feature in activations[-1][0, 1].nonzero().flatten().tolist()[:2]:
        graph.nodes.append({"kind": "feature", "layer": 3, "position": 1, "feature": feature,
                            "activation": float(activations[-1][0, 1, feature].detach()), "influence": .01, "effect": 0.})
    try:
        with model.frozen_attention(norms=True):
            result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert result["count"] == 10 and result["passed"]
        assert result["correct_signs"] == 10
        assert result["spearman"] == pytest.approx(1)
    finally:
        service.remove()


def test_trace_snapshot_restores_current_lifecycle_on_every_exit():
    model, tokenizer, dictionary, controls, service = validation_case()
    traced = service.state_dict()
    service.set({"kind": "clamp", "point": "layers.3.mlp_out", "feature": 7, "value": 0.})
    service.set({"kind": "scale", "point": "layers.1.mlp_out", "feature": 0, "multiplier": 1., "from_step": 20})
    service.set({"kind": "ablate", "point": "layers.2.mlp", "until_step": 2})
    controls.step()
    saved = service.state_dict()
    try:
        with pytest.raises(RuntimeError, match="snapshot exit"):
            with service.temporary_state(traced) as replay:
                assert replay.specifications() == []
                assert service.specifications()[0]["kind"] == "clamp"
                assert controls.n == 1
                raise RuntimeError("snapshot exit")
        assert service.state_dict() == saved
        assert service.specifications()[0]["kind"] == "clamp"
        assert controls.n == 1
    finally:
        service.remove()
