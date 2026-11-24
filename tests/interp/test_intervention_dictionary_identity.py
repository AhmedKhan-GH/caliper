"""Generic restore contracts for default dictionary identifiers."""
from copy import deepcopy

import pytest
import torch

from caliper.interp._intervene import InterventionError
from .test_intervene import add_spec, controlled, dictionary_for
from .test_intervention_replay import hook_objects


def feature_spec():
    return {"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0, "value": 1}


@pytest.mark.parametrize("change", ["weights", "topk", "metadata"])
@pytest.mark.parametrize("restore_active", [False, True])
def test_default_identifier_refuses_different_dictionary_atomically(controlled, monkeypatch, change, restore_active):
    model, tokenizer, ctl, service = controlled
    original = dictionary_for(model)
    service.configure(dictionary=original, tokenizer=tokenizer)
    service.set(feature_spec())
    ctl.step()
    state = service.state_dict()
    identifier = state["interventions"][0]["specification"]["dictionary"]
    assert identifier != original.metadata["model_hash"]
    different = dictionary_for(model)
    if change == "weights":
        different.layers[1].W_dec.data[0, 0].add_(0.125)
    elif change == "topk":
        different.layers[1].k -= 1
    else:
        different.metadata["revision"] = "different-dictionary"
    assert original.metadata["model_hash"] == different.metadata["model_hash"]
    service.configure(dictionary=different, tokenizer=tokenizer)
    service.set(add_spec())
    ctl.step()
    before, hooks = service.state_dict(), hook_objects(model)

    def forbidden(*args, **kwargs):
        pytest.fail("dictionary mismatch must refuse before hook installation")

    monkeypatch.setattr(service.executor, "install", forbidden)
    with pytest.raises(InterventionError, match="unknown or absent dictionary") as error:
        service.load_state_dict(state, restore_active=restore_active)
    assert error.value.code == "intervene.unknown_dictionary"
    assert service.state_dict() == before and hook_objects(model) == hooks


def test_default_identifier_restores_identical_content_and_ignores_quality(controlled):
    model, tokenizer, ctl, service = controlled
    original = dictionary_for(model)
    service.configure(dictionary=original, tokenizer=tokenizer)
    service.set(feature_spec())
    ctl.step()
    state = service.state_dict()
    expected = model(**tokenizer("abc")).logits
    identical = dictionary_for(model)
    identical.metadata["quality"] = {"fvu": 0.0}
    service.configure(dictionary=identical, tokenizer=tokenizer)
    service.load_state_dict(deepcopy(state), restore_active=True)
    assert service.state_dict() == state
    assert state["version"] == 1
    torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)
    with service.temporary_state(state):
        torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["set", "restore", "replay"])
def test_default_identity_detects_alias_change_after_configuration(controlled, route):
    model, tokenizer, ctl, service = controlled
    dictionary = dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    service.set(feature_spec())
    ctl.step()
    state, hooks = service.state_dict(), hook_objects(model)
    dictionary.layers[1].W_dec.data[0, 0].add_(0.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        if route == "set":
            service.set(feature_spec())
        elif route == "restore":
            service.load_state_dict(state, restore_active=True)
        else:
            with service.temporary_state(state):
                pytest.fail("changed dictionary must refuse before replay")
    assert service.state_dict() == state and hook_objects(model) == hooks


def test_explicit_dictionary_identifier_remains_a_caller_alias(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(feature_spec())
    ctl.step()
    state = service.state_dict()
    assert state["interventions"][0]["specification"]["dictionary"] == "fixture"
    dictionary = dictionary_for(model)
    dictionary.layers[1].W_dec.data[0, 0].add_(0.125)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, dictionary_id="fixture")
    service.load_state_dict(state, restore_active=True)
    assert service.state_dict() == state


def test_legacy_model_identifier_requires_explicit_alias(controlled):
    model, tokenizer, ctl, service = controlled
    dictionary = dictionary_for(model)
    legacy_id = dictionary.metadata["model_hash"]
    service.configure(dictionary=dictionary, tokenizer=tokenizer, dictionary_id=legacy_id)
    service.set(feature_spec())
    ctl.step()
    state = service.state_dict()
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    with pytest.raises(InterventionError, match="unknown or absent dictionary"):
        service.load_state_dict(state, restore_active=True)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, dictionary_id=legacy_id)
    service.load_state_dict(state, restore_active=True)
    assert service.state_dict() == state


def test_default_identifier_guard_refusal_preserves_configuration(controlled):
    from caliper.interp._memory import CleanStop
    from .test_intervention_identity import IdentityGuard
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before, hooks = service.state_dict(), hook_objects(model)
    dictionary = service.dictionary
    guard = IdentityGuard(limit=1)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, resource_guard=guard)
    assert guard.stages == ["model_identity"]
    assert service.dictionary is dictionary and service.executor.dictionary_id == "fixture"
    assert service.state_dict() == before and hook_objects(model) == hooks
