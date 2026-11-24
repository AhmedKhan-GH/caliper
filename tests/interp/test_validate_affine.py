"""Nonlinear validation deltas for existing signed and constant output effects."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import Tracer
from caliper.interp._trace import _DirectEffects
from caliper.interp._validate import validate_trace
from .test_output_affine import setup, feature_at, specs_for


@pytest.mark.parametrize("case", ["zero", "constant", "negative", "revived", "patch_first"])
def test_validation_removes_the_effective_write_from_actual_nonlinear_output(case):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specifications, coefficient, constant = specs_for(case, "layers.1.mlp_out", feature, 12)
        for spec in specifications:
            service.set(spec)
        owner.step()
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        traced = tracer.trace("AB", max_targets=32)
        path = tracer._forward_cache[1]
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        logit = effects.logit_nodes[0]
        weights, _ = effects.backward([logit])
        feature_id = f"feature:1:1:{feature}"
        selected = [i for i in effects.feature_nodes if effects.nodes[i]["id"] == feature_id]
        original = float(dictionary.encode(1, path.inputs[1])[0, 1, feature])
        effective = coefficient * original + constant
        if case == "zero":
            assert not selected
            assert not any(node["id"] == feature_id for node in traced.nodes)
        else:
            assert len(selected) == 1
            assert effects.nodes[selected[0]]["activation"] == pytest.approx(effective)
        # Select one real unpruned source for a bounded measurement. Display pruning stays unchanged.
        graph = SimpleNamespace(trace_id=traced.trace_id, tokens=effects.tokens,
            nodes=[effects.nodes[logit]] + [dict(effects.nodes[i], influence=1., effect=float(weights[i, 0])) for i in selected])
        state = service.state_dict()
        cache = dict(service.executor._cache)
        result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert service.state_dict() == state
        assert all(service.executor._cache[key] is value for key, value in cache.items())
        assert result["count"] == len(selected)
        if case == "zero":
            assert result["measurements"] == []
            return

        token = effects.nodes[logit]["token_id"]
        def centered(logits):
            return float(logits[0, -1, token] - logits[0, -1].mean())
        with torch.no_grad():
            baseline = centered(model(input_ids=ids).logits)
            def remove_effective(module, args, output):
                changed = output.clone()
                changed[:, 1] -= effective * dictionary.layers[1].W_dec[feature]
                return changed
            handle = model.layers[1].mlp.register_forward_hook(remove_effective)
            try:
                actual = centered(model(input_ids=ids).logits) - baseline
            finally:
                handle.remove()
        measurement = result["measurements"][0]
        assert result["baseline_centered_logit"] == pytest.approx(baseline, abs=1e-9)
        assert measurement["measured"] == pytest.approx(actual, abs=1e-9)
        assert measurement["predicted"] == pytest.approx(-float(weights[selected[0], 0]), abs=1e-9)
        assert abs(actual) > 1e-6


def test_validation_uses_applied_clamp_before_pending_replacement():
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        spec = {"kind": "clamp", "point": "layers.1.mlp_out", "feature": feature, "positions": [1], "value": .4}
        service.set(spec)
        owner.step()
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        baseline = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        service.set(dict(spec, value=2.))
        pending = service.state_dict()
        actual = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert actual == baseline
        assert service.state_dict() == pending
