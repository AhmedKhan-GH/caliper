"""Bounded CPU checks for live, replayed, and frozen feature composition."""
import json

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from .fixture import make_fixture
from .test_intervene import explicit_forward
from .test_trace import exact_dictionary


@pytest.fixture
def composition():
    model, tokenizer = make_fixture(layers=2, width=8, heads=2, latents=8, k=3)
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
    torch.testing.assert_close(path.outputs[0], expected[0], rtol=0, atol=1e-12)
    torch.testing.assert_close(path.logits, expected[1][:, -1], rtol=0, atol=1e-12)
    assert path.forward_residual < 1e-9
    graph = Tracer(model, tokenizer, dictionary, controls=controls).trace("AB", max_targets=16)
    assert graph.completeness["forward_residual"] < 1e-9
    assert graph.completeness["max_absolute_residual"] < 1e-9
    assert all(abs(row["incoming"] + sum(row["biases"].values()) - row["expected"]) < 1e-9
               for row in graph.completeness["targets"])
    return path


@pytest.mark.parametrize("mode", ["snapshot", "specifications", "restore"])
def test_group_keeps_hook_position_after_first_member_removal(composition, mode):
    model, _, _, controls, service, ids, _, feature = composition
    point = "layers.0.mlp_out"
    anchor = (feature + 1) % 8
    service.set(dict(kind="scale", point=point, feature=anchor, positions=[2], multiplier=1.))
    service.set(dict(kind="patch", point=point, positions=[2], source_prompt="CD"))
    service.set(dict(kind="clamp", point=point, feature=feature, positions=[2], value=-2.))
    controls.step()
    before = observe(model, ids)
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
    assert_frozen(composition, expected)


@pytest.mark.parametrize("overlap", ["scale_clamp", "clamp_positions"])
def test_frozen_group_uses_last_edit_at_each_coordinate(composition, overlap):
    model, _, _, controls, service, ids, original, feature = composition
    point = "layers.0.mlp_out"
    if overlap == "scale_clamp":
        first = dict(kind="scale", point=point, feature=feature, positions=[2], multiplier=2.)
    else:
        first = dict(kind="clamp", point=point, feature=feature, positions=[1, 2], value=3.)
    second = dict(kind="clamp", point=point, feature=feature, positions=[2], value=-1.)
    baseline = observe(model, ids)[0]
    service.set(first)
    service.set(second)
    controls.step()
    expected = observe(model, ids)
    effective = original.clone()
    if overlap == "clamp_positions":
        effective[:, 1, feature] = 3.
    effective[:, 2, feature] = -1.
    manual = baseline + (effective - original) @ service.dictionary.layers[0].W_dec
    torch.testing.assert_close(expected[0], manual, rtol=0, atol=1e-12)
    saved = json.loads(json.dumps(service.state_dict()))
    with service.temporary_state(saved):
        replay = observe(model, ids)
    service.load_state_dict(saved, restore_active=True)
    restored = observe(model, ids)
    for actual in (replay, restored):
        for value, live in zip(actual, expected):
            torch.testing.assert_close(value, live, rtol=0, atol=0)
    path = assert_frozen(composition, expected)
    torch.testing.assert_close(path.features[0], effective, rtol=0, atol=1e-12)
