"""Exact feature effects on a synthetic dictionary."""
import pytest
import torch

from .test_intervene import controlled, dictionary_for, explicit_forward


@pytest.mark.parametrize("kind", ["scale", "clamp"])
def test_admitted_feature_retains_exact_effect(controlled, kind):
    model, tokenizer, ctl, service = controlled
    dictionary = dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    ids = tokenizer("abc")["input_ids"]
    layer, position = 0, 2
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
