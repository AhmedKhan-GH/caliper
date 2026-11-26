"""Independent CPU oracles for ordered output intervention derivatives."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from caliper._controls import Controls
from caliper.interp import Tracer, capture
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@contextmanager
def setup():
    model = TinyLM(layers=3, width=12, heads=3, latents=16, k=3)
    tokenizer = TinyTokenizer()
    dictionary = exact_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    try:
        yield model, tokenizer, dictionary, owner, service
    finally:
        service.remove()


@contextmanager
def fixed_patch_sources(model, tokenizer, service):
    """Keep original clean tensors fixed while the oracle changes target math.

    A clean prompt must not run under the target's frozen norm denominators or
    temporary oracle hooks. Capture it independently from the unmodified model.
    """
    specs = [spec for spec in service.specifications() if spec.get("op", spec.get("kind")) == "patch"]
    saved = [(module, name, getattr(module, name)) for module in model.modules()
             for name in ("last_scale", "last_pattern") if hasattr(module, name)]
    sources = {}
    try:
        for prompt in sorted({spec["source_prompt"] for spec in specs}):
            points = sorted({spec["point"] for spec in specs if spec["source_prompt"] == prompt})
            with service.suspend(), capture(model, points) as values, torch.no_grad():
                model(**tokenizer(prompt), use_cache=False)
                sources.update({(prompt, point): values[point].detach().clone() for point in points})
        for spec in specs:
            torch.testing.assert_close(service.executor._source(spec), sources[spec["source_prompt"], spec["point"]],
                                       rtol=0, atol=0)
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
    with patch.object(service.executor, "_source", lambda spec: sources[spec["source_prompt"], spec["point"]]):
        yield


def feature_at(model, dictionary, ids, layer=1, *, inactive=False):
    blocks = model.layers
    reads = []
    handle = blocks[layer].mlp.register_forward_pre_hook(lambda m, a: reads.append(a[0].detach()))
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    values = dictionary.encode(layer, reads[0])[0, 1]
    return int((values == 0).nonzero()[0]) if inactive else int(values.argmax())


def specs_for(case, point, feature, width):
    clamp = {"kind": "clamp", "point": point, "feature": feature, "positions": [1], "value": -.4}
    scale = {"kind": "scale", "point": point, "feature": feature, "positions": [1], "multiplier": -2.}
    add = {"kind": "add", "point": point, "positions": [1], "vector": [.03] * width, "coefficient": 2.}
    patch = {"kind": "patch", "point": point, "positions": [1], "source_prompt": "CD", "source_position": 2}
    ablate = {"kind": "ablate", "point": point}
    # Positioned edits retain the last delta at each group coordinate.
    return {
        "zero": ([dict(clamp, value=0)], 0., 0.),
        "constant": ([clamp], 0., -.4),
        "negative": ([scale], -2., 0.),
        "revived": ([ablate, add, clamp], -1., -.4),
        "reset": ([clamp, scale, add, ablate], 0., 0.),
        "patch_first": ([patch, add, scale, clamp], -1., -.4),
        "patch_last": ([scale, clamp, add, patch], 0., 0.),
    }[case]


@pytest.mark.parametrize("case", ["zero", "constant", "negative", "revived", "reset", "patch_first", "patch_last"])
def test_tiny_output_affine_matches_live_frozen_oracle(case):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specs, coefficient, constant = specs_for(case, "layers.1.mlp_out", feature, 12)
        for spec in specs:
            service.set(spec)
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        original = dictionary.encode(1, path.inputs[1])[0, 1, feature]
        assert original > 0
        torch.testing.assert_close(path.features[1][0, 1, feature], coefficient * original + constant)
        assert all(error.count_nonzero() == 0 for error in path.errors)
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        node_id = f"feature:1:1:{feature}"
        matches = [i for i in effects.feature_nodes if effects.nodes[i]["id"] == node_id]
        if coefficient == constant == 0:
            assert not matches
        else:
            assert len(matches) == 1
            target = effects.target(matches[0])
            torch.testing.assert_close(target.direction, coefficient * dictionary.layers[1].W_enc[feature])
            incoming, diagnostics = effects.backward(matches)
            assert diagnostics[0]["absolute_residual"] < 1e-9
            if coefficient == 0:
                assert incoming.count_nonzero() == 0
                assert diagnostics[0]["biases"]["intervention"] == pytest.approx(constant)

        direction = torch.linspace(-.3, .4, path.embedding.numel(), dtype=path.dtype).reshape_as(path.embedding)
        with fixed_patch_sources(model, tokenizer, service), model.frozen_attention(norms=True), torch.no_grad():
            # Two small perturbations exercise actual live output hooks and the nonlinear TinyLM.
            for sign in (-1, 1):
                embedding = path.embedding + sign * 1e-6 * direction
                expected = model(inputs_embeds=embedding).logits[:, -1]
                _, final = path.replacement(embedding, dictionary)
                actual = model.lm_head(final[:, -1])
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("inactive", [False, True])
def test_constant_clamp_keeps_a_signed_node_without_an_incoming_derivative(inactive):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids, inactive=inactive)
        service.set({"kind": "clamp", "point": "layers.1.mlp_out", "feature": feature,
                     "positions": [1], "value": -.4})
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
        incoming, diagnostics = effects.backward([target])
        assert incoming.count_nonzero() == 0
        assert effects.nodes[target]["activation"] == pytest.approx(-.4)
        torch.testing.assert_close(path.features[1][0, 1, feature], path.embedding.new_tensor(-.4), atol=0, rtol=0)
        assert diagnostics[0]["biases"]["intervention"] == pytest.approx(-.4)
        assert diagnostics[0]["absolute_residual"] < 1e-9
        # A constant feature still writes its decoder vector into later targets.
        outgoing, _ = effects.backward(effects.logit_nodes)
        assert outgoing[target].abs().max() > 1e-6
        assert path.features[1].count_nonzero() > 0
        assert path.errors[1].count_nonzero() == 0


def test_fractional_scale_keeps_the_model_coefficient_precision():
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        service.set({"kind": "scale", "point": "layers.1.mlp_out", "feature": feature,
                     "positions": [1], "multiplier": .1})
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
        torch.testing.assert_close(effects.target(target).direction, .1 * dictionary.layers[1].W_enc[feature],
                                   atol=1e-15, rtol=0)


@pytest.mark.parametrize("legacy_owner", [False, True])
def test_pending_replacement_rows_and_cold_source_cache_keep_applied_edges(legacy_owner, monkeypatch):
    with setup() as (model, tokenizer, dictionary, owner, service):
        if legacy_owner:
            monkeypatch.delattr(Controls, "intervention_service", raising=False)
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        tracer._new_prompt("AB")
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specs, _, _ = specs_for("patch_first", "layers.1.mlp_out", feature, 12)
        # Input edits remain active with output edits and share the clean source prompt.
        specs.insert(0, {"kind": "patch", "point": "layers.1.mlp_in", "positions": [0], "source_prompt": "CD"})
        specs.insert(1, {"kind": "add", "point": "layers.1.resid_pre", "positions": [2],
                         "vector": [.01] * 12, "coefficient": 1.})
        for spec in specs:
            service.set(spec)
        owner.step()
        service.invalidate_source_cache()
        calls = []
        handle = model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            cold = tracer.trace("AB", max_targets=48)
            assert len(calls) == 2  # Target forward and one shared clean source forward.
            sources = dict(service.executor._cache)
            assert len(sources) == 2
            assert tracer.trace("AB", max_targets=48) is cold
            assert len(calls) == 2
            tracer._forward_cache = None
            tracer._cache.clear()
            warm = tracer.trace("AB", max_targets=48)
            assert len(calls) == 3
            assert all(service.executor._cache[key] is value for key, value in sources.items())
            assert warm.nodes == cold.nodes and warm.edges == cold.edges
            service.set(dict(specs[-1], value=2.))
            assert tracer.trace("AB", max_targets=48) is warm
            assert len(calls) == 3
            with service.rows([]):
                excluded = tracer.trace("AB", max_targets=48)
            with service.suspend():
                suspended = tracer.trace("AB", max_targets=48)
            assert excluded.nodes == suspended.nodes and excluded.edges == suspended.edges
            with service.rows([0]):
                selected = tracer.trace("AB", max_targets=48)
            assert selected.nodes == warm.nodes and selected.edges == warm.edges
            owner.step()
            changed = tracer.trace("AB", max_targets=48)
            assert changed.trace_id != warm.trace_id
            assert changed.completeness["max_absolute_residual"] < 1e-9
        finally:
            handle.remove()
