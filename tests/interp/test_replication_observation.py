"""Actual post-intervention feature observations against an independent formula."""
import pytest
import torch

from caliper.interp._replication_native import observe_features
from caliper.interp._output import forward_logits
from .test_replication_coordinates import original_case, oracle


def hooks(model):
    return [(tuple(module._forward_pre_hooks), tuple(module._forward_hooks)) for module in model.modules()]


def test_observation_uses_the_actual_changed_pre_gain_forward(original_case):
    model, tokenizer, service, runtime, captured = original_case
    ids = tokenizer("AB")["input_ids"]
    _, members = runtime.case(26)
    clamps = {(row["layer"], row["feature"], row["position"]): row["value"] for row in members}
    expected_logits, values, _, _ = oracle(model, ids, service.dictionary, clamps=clamps, patterns=runtime.patterns)
    node = (1, int(values[1][0, 1].argmax()), 1)
    runtime.run(26)
    before = hooks(model)
    reservations = []
    with observe_features(model, service.dictionary, [node], input_shape=ids.shape, guard=None,
                          reserve=reservations.append) as observed:
        assert observed == {}
        actual = forward_logits(model, tokenizer("AB"))
    assert reservations and reservations[0] > 4 * 1024 * 1024
    assert observed[node].dtype == torch.bfloat16 and observed[node].shape == (1,)
    torch.testing.assert_close(observed[node], values[1][0, 1, node[1]].reshape(1), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected_logits, rtol=0, atol=0)
    assert all(not row for row in captured)
    assert hooks(model) == before


def test_observation_reserve_refuses_before_new_hooks(original_case):
    model, tokenizer, service, runtime, _ = original_case
    before = hooks(model)
    def refuse(amount):
        raise RuntimeError("fixture observation reserve refusal")
    with pytest.raises(RuntimeError, match="observation reserve refusal"):
        with observe_features(model, service.dictionary, [(1, 0, 1)],
                              input_shape=tokenizer("AB")["input_ids"].shape, guard=None, reserve=refuse):
            pytest.fail("observation passed refused admission")
    assert hooks(model) == before


@pytest.mark.parametrize("failure", ["missing_forward", "forward_error"])
def test_failed_observation_removes_its_hooks(original_case, failure):
    model, tokenizer, service, runtime, _ = original_case
    before = hooks(model)
    expected = "no completed forward" if failure == "missing_forward" else "fixture forward refusal"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        with observe_features(model, service.dictionary, [(1, 0, 1)],
                              input_shape=tokenizer("AB")["input_ids"].shape, guard=None, reserve=lambda amount: None):
            if failure == "forward_error":
                raise RuntimeError("fixture forward refusal")
    assert hooks(model) == before
