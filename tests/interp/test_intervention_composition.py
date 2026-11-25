"""Bounded CPU checks for live, replayed, and frozen feature composition."""
import json
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from caliper.interp._validate import validate_trace
from .fixture import make_fixture
from .test_intervene import explicit_forward
from .test_trace import exact_dictionary


@pytest.fixture
def composition(request):
    model, tokenizer = make_fixture(layers=2, width=8, heads=2, latents=8, k=3,
                                    dtype=getattr(request, "param", torch.float64))
    dictionary = exact_dictionary(model)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    ids = tokenizer("AB")["input_ids"]
    _, activations = explicit_forward(model, ids)
    feature = int(activations[0][0, 2].argmax())
    try:
        yield model, tokenizer, dictionary, controls, service, ids, activations[0], feature
    finally:
        service.remove()


def observe(model, ids):
    outputs = []
    handle = model.layers[0].mlp.register_forward_hook(
        lambda module, args, output: outputs.append(output.detach().clone()))
    try:
        logits = model(ids).logits.detach()
    finally:
        handle.remove()
    return outputs[-1], logits


def assert_frozen(composition, expected):
    model, tokenizer, dictionary, controls, service, ids, _, _ = composition
    path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
    tolerance = 1e-9 if path.dtype == torch.float64 else 2e-5
    output_tolerance = 1e-12 if path.dtype == torch.float64 else 2e-5
    torch.testing.assert_close(path.outputs[0], expected[0], rtol=0, atol=output_tolerance)
    torch.testing.assert_close(path.logits, expected[1][:, -1], rtol=0, atol=output_tolerance)
    assert path.forward_residual < tolerance
    graph = Tracer(model, tokenizer, dictionary, controls=controls).trace("AB", max_targets=16)
    assert graph.completeness["forward_residual"] < tolerance
    assert graph.completeness["max_absolute_residual"] < tolerance
    assert all(abs(row["incoming"] + sum(row["biases"].values()) - row["expected"]) < tolerance
               for row in graph.completeness["targets"])
    return path


def assert_validation(composition, path):
    model, tokenizer, dictionary, _, service, ids, _, feature = composition
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
    target = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0
                  and effects.nodes[i]["position"] == 2 and effects.nodes[i]["feature"] == feature)
    top = effects.logit_nodes[0]
    weights, _ = effects.backward([top])
    node = dict(effects.nodes[target], influence=1., effect=float(weights[target, 0]))
    graph = SimpleNamespace(trace_id="composition", tokens=effects.tokens,
                            nodes=[effects.nodes[top], node])
    result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
    assert result["count"] == 1

    def remove_contribution(module, args, output):
        changed = output.clone()
        changed[:, 2] -= node["activation"] * dictionary.layers[0].W_dec[feature]
        return changed

    handle = model.layers[0].mlp.register_forward_hook(remove_contribution)
    try:
        logits = model(ids).logits[0, -1].detach()
    finally:
        handle.remove()
    measured = float(logits[result["token"]] - logits.mean()) - result["baseline_centered_logit"]
    tolerance = 1e-9 if path.dtype == torch.float64 else 2e-5
    assert result["measurements"][0]["measured"] == pytest.approx(measured, abs=tolerance)


@pytest.mark.parametrize("mode", ["snapshot", "specifications", "restore"])
@pytest.mark.parametrize("removal", ["clear", "expire"])
def test_group_keeps_hook_position_after_first_member_removal(composition, mode, removal):
    model, _, _, controls, service, ids, _, feature = composition
    point = "layers.0.mlp_out"
    anchor = (feature + 1) % 8
    scale = dict(kind="scale", point=point, feature=anchor, positions=[2], multiplier=1.)
    if removal == "expire":
        scale["until_step"] = 3
    service.set(scale)
    service.set(dict(kind="patch", point=point, positions=[2], source_prompt="CD"))
    service.set(dict(kind="clamp", point=point, feature=feature, positions=[2], value=-2.))
    controls.step()
    before = observe(model, ids)
    if removal == "clear":
        assert service.clear_feature(point, anchor, kind="scale", positions=[2])
    controls.step()
    expected = observe(model, ids)
    for actual, original in zip(expected, before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(expected[0][:, 2], service.executor._cache["CD", point][:, 2], rtol=0, atol=0)
    saved = json.loads(json.dumps(service.state_dict()))
    if mode == "restore":
        service.load_state_dict(saved, restore_active=True)
        actual = observe(model, ids)
    else:
        state = saved if mode == "snapshot" else service.specifications()
        with service.temporary_state(state):
            actual = observe(model, ids)
    for value, original in zip(actual, expected):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    path = assert_frozen(composition, expected)
    assert torch.count_nonzero(path.features[0][:, 2]) == 0
    tracer = Tracer(model, composition[1], composition[2], controls=controls)
    assert tracer._intervention_state()["specifications"] == service.specifications()
    # A later boundary retains the group's position when its member changes.
    service.set(dict(kind="clamp", point=point, feature=feature, positions=[2], value=-3.))
    controls.step()
    for value, original in zip(observe(model, ids), expected):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("overlap", ["scale_clamp", "clamp_positions"])
@pytest.mark.parametrize("composition", [torch.float64, torch.float32], indirect=True)
@pytest.mark.parametrize("reverse", [False, True])
def test_frozen_group_uses_last_edit_at_each_coordinate(composition, overlap, reverse):
    model, _, _, controls, service, ids, original, feature = composition
    point = "layers.0.mlp_out"
    if overlap == "scale_clamp":
        first = dict(kind="scale", point=point, feature=feature, positions=[2], multiplier=2.)
    else:
        first = dict(kind="clamp", point=point, feature=feature, positions=[1, 2], value=3.)
    second = dict(kind="clamp", point=point, feature=feature, positions=[2], value=-1.)
    baseline = observe(model, ids)[0]
    for spec in ([second, first] if reverse else [first, second]):
        service.set(spec)
    controls.step()
    expected = observe(model, ids)
    effective = original.clone()
    if overlap == "clamp_positions":
        effective[:, 1, feature] = 3.
    effective[:, 2, feature] = -1.
    if reverse:
        effective[:, 2, feature] = original[:, 2, feature] * 2 if overlap == "scale_clamp" else 3.
    manual = baseline + (effective - original) @ service.dictionary.layers[0].W_dec
    tolerance = 1e-12 if original.dtype == torch.float64 else 2e-5
    torch.testing.assert_close(expected[0], manual, rtol=0, atol=tolerance)
    saved = json.loads(json.dumps(service.state_dict()))
    with service.temporary_state(saved):
        replay = observe(model, ids)
    service.load_state_dict(saved, restore_active=True)
    restored = observe(model, ids)
    for actual in (replay, restored):
        for value, live in zip(actual, expected):
            torch.testing.assert_close(value, live, rtol=0, atol=0)
    path = assert_frozen(composition, expected)
    torch.testing.assert_close(path.features[0], effective, rtol=0, atol=tolerance)
    assert torch.count_nonzero(path.output_offsets[0]) == 0
    assert_validation(composition, path)
    # Independent input changes test the affine coefficients beyond one forward.
    direction = torch.linspace(-.3, .4, path.embedding.numel(), dtype=path.dtype).reshape_as(path.embedding)
    epsilon = 1e-6 if path.dtype == torch.float64 else 1e-3
    model(ids)
    with model.frozen_attention(norms=True), torch.no_grad():
        embedding = path.embedding + epsilon * direction
        live = model(inputs_embeds=embedding).logits[:, -1]
        _, final = path.replacement(embedding, service.dictionary)
        torch.testing.assert_close(model.lm_head(final[:, -1]), live, rtol=0, atol=tolerance)


@pytest.mark.parametrize("positioned", [(False, False), (True, False), (False, True)])
def test_ungrouped_feature_hooks_retain_independent_deltas(composition, positioned):
    model, _, _, controls, service, ids, original, feature = composition
    first = dict(kind="scale", point="layers.0.mlp_out", feature=feature, multiplier=2.)
    second = dict(kind="clamp", point="layers.0.mlp_out", feature=feature, value=-1.)
    if positioned[0]:
        first["positions"] = [1, 2]
    if positioned[1]:
        second["positions"] = [2]
    baseline = observe(model, ids)[0]
    service.set(first)
    service.set(second)
    controls.step()
    expected = observe(model, ids)
    effective = original.clone()
    scaled = first.get("positions", [0, 1, 2])
    clamped = second.get("positions", [0, 1, 2])
    effective[:, scaled, feature] += original[:, scaled, feature]
    effective[:, clamped, feature] += -1. - original[:, clamped, feature]
    manual = baseline + (effective - original) @ service.dictionary.layers[0].W_dec
    torch.testing.assert_close(expected[0], manual, rtol=0, atol=1e-12)
    path = assert_frozen(composition, expected)
    torch.testing.assert_close(path.features[0], effective, rtol=0, atol=1e-12)
    assert_validation(composition, path)


def test_recreated_group_gets_a_new_hook_position(composition):
    model, _, _, controls, service, ids, original, feature = composition
    point = "layers.0.mlp_out"
    spec = dict(kind="clamp", point=point, feature=feature, positions=[2], value=-1.)
    service.set(spec)
    service.set(dict(kind="patch", point=point, positions=[2], source_prompt="CD"))
    controls.step()
    patched = observe(model, ids)[0]
    assert service.clear_feature(point, feature, positions=[2])
    controls.step()
    assert not service.executor._feature_groups
    service.set(spec)
    controls.step()
    expected = observe(model, ids)
    manual = patched.clone()
    manual[:, 2] += (-1. - original[:, 2, feature, None]) * service.dictionary.layers[0].W_dec[feature]
    torch.testing.assert_close(expected[0], manual, rtol=0, atol=1e-12)
    assert [s["op"] for s in service.specifications()] == ["patch", "clamp"]
    saved = service.state_dict()
    with service.temporary_state(saved):
        replay = observe(model, ids)
    service.load_state_dict(saved, restore_active=True)
    for actual in (replay, observe(model, ids)):
        for value, live in zip(actual, expected):
            torch.testing.assert_close(value, live, rtol=0, atol=0)
    assert_frozen(composition, expected)


def test_patch_before_overlapping_group_retains_only_the_final_delta(composition):
    model, _, _, controls, service, ids, original, feature = composition
    point = "layers.0.mlp_out"
    service.set(dict(kind="patch", point=point, positions=[2], source_prompt="CD"))
    service.set(dict(kind="scale", point=point, feature=feature, positions=[2], multiplier=2.))
    service.set(dict(kind="clamp", point=point, feature=feature, positions=[2], value=-1.))
    controls.step()
    expected = observe(model, ids)
    delta = (-1. - original[:, 2, feature, None]) * service.dictionary.layers[0].W_dec[feature]
    patched = service.executor._cache["CD", point][:, 2]
    torch.testing.assert_close(expected[0][:, 2], patched + delta, rtol=0, atol=1e-12)
    path = assert_frozen(composition, expected)
    effective = original.clone()
    effective[:, 2] = 0
    effective[:, 2, feature] = -1. - original[:, 2, feature]
    torch.testing.assert_close(path.features[0], effective, rtol=0, atol=1e-12)
    assert_validation(composition, path)
