"""CPU admission checks for feature targets in upper-half dictionaries."""
import pytest
import torch

from caliper.interp._dictionary import fallback_metadata
from caliper.interp._intervene import InterventionError, InterventionService
from .test_intervene import controlled, dictionary_for, explicit_forward
from .test_intervention_replay import hook_objects, snapshot


def upper_dictionary(model):
    dictionary = dictionary_for(model)
    count = len(dictionary.layers)
    dictionary.metadata.update(layers=count, training={
        "source_id": "synthetic-text-v1",
        "fallback": {
            "version": 1, "mode": "upper_half",
            "active_layers": list(range(count // 2, count)),
            "zero_layers": list(range(count // 2)),
            "source_dictionary_hash": "a" * 64,
            "source_checkpoint_generation": "checkpoint-" + "b" * 32,
            "source_training_tokens": 16, "source_elapsed_seconds": 100.0,
            "prior_wall_seconds": 120.0, "aggregate_wall_limit_seconds": 36000.0,
            "token_budget": 16, "warm_start": "upper_weights_fresh_adam",
            "source_id": "synthetic-text-v1", "lower_behavior": "zero_dictionary_original_mlp",
        },
    })
    fallback = fallback_metadata(dictionary.metadata)
    with torch.no_grad():
        for i in fallback["zero_layers"]:
            for parameter in dictionary.layers[i].parameters():
                parameter.zero_()
    return dictionary


@pytest.mark.parametrize("kind", ["scale", "clamp"])
@pytest.mark.parametrize("route", ["install", "set", "restore", "restore_active", "replay"])
def test_excluded_feature_refuses_before_hooks_or_forwards(controlled, monkeypatch, kind, route):
    model, tokenizer, ctl, service = controlled
    service.configure(dictionary=upper_dictionary(model), tokenizer=tokenizer, dictionary_id="upper")
    value = {"multiplier": 2} if kind == "scale" else {"value": 1}
    spec = {"kind": kind, "point": "layers.0.mlp_out", "feature": 0, **value}
    service.set(dict(spec, point="layers.2.mlp_out"))
    ctl.step()
    before, hooks = service.state_dict(), hook_objects(model)
    state = snapshot({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "abc"}, spec)

    def forbidden(*args, **kwargs):
        pytest.fail("excluded features must refuse before hook installation or a forward")

    monkeypatch.setattr(model, "forward", forbidden)
    if route != "install":
        monkeypatch.setattr(InterventionService, "install", forbidden)
    with pytest.raises(InterventionError, match="excludes this feature layer") as error:
        if route == "install":
            service.executor.install(spec)
        elif route == "set":
            service.set(spec)
        elif route.startswith("restore"):
            service.load_state_dict(state, restore_active=route == "restore_active")
        else:
            with service.temporary_state(state):
                pytest.fail("excluded features must refuse before replay")
    assert error.value.code == "intervene.unknown_feature"
    assert service.state_dict() == before and hook_objects(model) == hooks


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("kind", ["scale", "clamp"])
def test_admitted_feature_retains_exact_effect(controlled, upper, kind):
    model, tokenizer, ctl, service = controlled
    dictionary = upper_dictionary(model) if upper else dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    ids = tokenizer("abc")["input_ids"]
    layer, position = (2 if upper else 0), 2
    baseline, activations = explicit_forward(model, ids)
    feature = int(activations[layer][0, position].argmax())

    def edit(index, active):
        result = active.clone()
        if index == layer:
            result[0, position, feature] *= -1 if kind == "scale" else 0
        return result

    expected, _ = explicit_forward(model, ids, edit=edit)
    assert not torch.equal(expected, baseline)
    value = {"multiplier": -1} if kind == "scale" else {"value": 0}
    service.set({"kind": kind, "point": f"layers.{layer}.mlp_out", "feature": feature,
                 "positions": [position], **value})
    ctl.step()
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=1e-12)
