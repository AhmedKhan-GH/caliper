"""Bounded CPU reproductions for intervention audit corrections."""
from copy import deepcopy

import pytest
import torch

from caliper import _wire as wire
from caliper._controls import Controls
from caliper.interp._intervene import InterventionError
from tests.interp.test_intervene import add_spec, controlled
from tests.interp.fixture import make_fixture


@pytest.mark.parametrize("spec", [
    add_spec(coefficient=2**53 + 1),
    add_spec(vector=[2**53 + 1] * 64),
    add_spec(positions=(1,)),
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "A\0B"},
    {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 2**53 + 1},
])
def test_local_requests_obey_wire_metadata_before_conversion(controlled, spec):
    _, _, ctl, service = controlled
    assert not wire._intervene_request({"set": spec})
    before = service.state_dict()
    with pytest.raises(InterventionError):
        service.set(spec)
    assert service.state_dict() == before
    wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, ctl.controls_meta()))


def test_adapter_only_ablation_point_preserves_legacy_execution():
    class Model:
        active = False

        def __call__(self):
            return 0 if self.active else 7

    class Adapter:
        def points(self, model):
            return [{"point": "adapter.output", "kind": "module", "module": "adapter.output",
                     "head": None, "type": "Fixture", "shape": None}]

        def ablate(self, model, point):
            assert point == "adapter.output"
            model.active = True

            def remove():
                model.active = False

            return remove

    model, ctl = Model(), Controls()
    service = ctl.interventions_on(model, adapter=Adapter())
    try:
        service.set({"op": "ablate", "point": "adapter.output"})
        assert model() == 7
        ctl.step()
        assert model() == 0
        saved = service.state_dict()
        service.load_state_dict(saved)
        assert model() == 7
        ctl.step()
        assert model() == 0
        service.clear("adapter.output")
        assert model() == 0
        ctl.step()
        assert model() == 7
        with pytest.raises(InterventionError, match="unknown intervention point"):
            service.set({"op": "ablate", "point": "other.output"})
    finally:
        service.remove()


def test_adapter_alias_uses_adapter_in_replay(controlled):
    from caliper._ablate import ablate
    model, tokenizer, ctl, _ = controlled

    class Adapter:
        def points(self, model):
            return [{"point": "adapter.output", "kind": "module", "module": "layers.0.mlp",
                     "head": None, "type": "Fixture", "shape": None}]

        def ablate(self, model, point):
            assert point == "adapter.output"
            return ablate(model, "layers.0.mlp")

    inputs = tokenizer("AB")
    baseline = model(**inputs).logits
    service = ctl.interventions_on(model, adapter=Adapter())
    try:
        service.set({"kind": "ablate", "point": "adapter.output"})
        ctl.step()
        active = model(**inputs).logits
        assert not torch.equal(active, baseline)
        with service.temporary_state(service.specifications()):
            torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)
        service.clear()
        ctl.step()
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
    finally:
        service.remove()


def test_retired_dictionary_records_restore_and_replay(controlled):
    model, tokenizer, ctl, service = controlled
    service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 4})
    service.set({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AB"})
    ctl.step()
    service.configure(dictionary=service.dictionary, dictionary_id="new")
    state = service.state_dict()
    baseline = model(**tokenizer("AB")).logits
    assert all(record["state"] == "ended" for record in state["interventions"])
    service.load_state_dict(state)
    assert service.state_dict() == state
    ctl.step()
    with service.temporary_state(state) as replay:
        assert not replay.specifications()
        torch.testing.assert_close(model(**tokenizer("AB")).logits, baseline, rtol=0, atol=0)
    assert service.state_dict() == state
    wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, ctl.controls_meta()))


@pytest.mark.parametrize("operation", ["restore", "replay"])
@pytest.mark.parametrize("changes", [
    {"specification": add_spec(coefficient=2**53 + 1)},
    {"specification": add_spec(positions=[32])},
    {"error": "bad\0text"}, {"error": "é" * 513}, {"error_code": "x" * 64},
])
def test_ended_records_still_require_valid_structure(controlled, operation, changes):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before = service.state_dict()
    state = deepcopy(before)
    state["interventions"][0].update(state="ended", **changes)
    with pytest.raises(InterventionError):
        if operation == "restore":
            service.load_state_dict(state)
        else:
            with service.temporary_state(state):
                pytest.fail("invalid ended record reached replay")
    assert service.state_dict() == before


@pytest.mark.parametrize("late_start", [False, True])
def test_overlap_order_matches_replay_restore_and_direct_oracle(controlled, late_start):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("AB")
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out"}
    service.set(dict(ablation, from_step=4) if late_start else ablation)
    service.set(add_spec())
    ctl.step()
    if late_start:
        ctl.step(3)
    else:
        service.set(dict(ablation, until_step=40))
        ctl.step()
    # Ablation precedes addition in the saved order. This independent hook replaces
    # their combined MLP output with the expected constant tensor.
    with service.suspend():
        handle = model.layers[0].mlp.register_forward_hook(lambda m, a, y: torch.full_like(y, 0.1))
        try:
            expected = model(**inputs).logits
        finally:
            handle.remove()
    actual = model(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with service.temporary_state(service.specifications()):
        torch.testing.assert_close(model(**inputs).logits, actual, rtol=0, atol=0)
    other, _ = make_fixture()
    fork = Controls()
    restored = fork.interventions_on(other)
    try:
        fork.n = ctl.n
        restored.load_state_dict(service.state_dict())
        fork.step()
        torch.testing.assert_close(other(**inputs).logits, actual, rtol=0, atol=0)
    finally:
        restored.remove()


def test_cancelled_replacement_preserves_hooks_output_and_cache_identity(controlled):
    import json
    model, tokenizer, ctl, service = controlled
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out"}
    service.set(ablation)
    service.set(add_spec())
    ctl.step()
    before = model(**tokenizer("AB")).logits
    hooks = [iv.hook for iv in ctl._ivs.values()]
    key = json.dumps(service.specifications(), sort_keys=True)
    service.set(dict(ablation, until_step=40))
    service.set(ablation)
    ctl.step()
    assert key == json.dumps(service.specifications(), sort_keys=True)
    torch.testing.assert_close(model(**tokenizer("AB")).logits, before, rtol=0, atol=0)
    assert hooks == [iv.hook for iv in ctl._ivs.values()]


@pytest.mark.parametrize("queued", ["replacement", "clear", "future"])
def test_script_restore_preserves_applied_output_and_queued_boundary(controlled, queued):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("AB")
    service.set({"kind": "ablate", "point": "layers.0.mlp_out"})
    service.set(add_spec())
    service.set(add_spec(point="layers.2.mlp_out", from_step=100))
    ctl.step(5)
    before = model(**inputs).logits
    if queued == "clear":
        service.clear("layers.0.mlp_out")
    else:
        service.set(add_spec(coefficient=2, **({"from_step": 100} if queued == "future" else {})))
    state = service.state_dict()
    other, _ = make_fixture()
    fork = Controls()
    restored = fork.interventions_on(other)
    fork.n = ctl.n - 1
    try:
        restored.load_state_dict(state, restore_active=True)
        assert fork.n == 4 and restored.state_dict() == state
        torch.testing.assert_close(other(**inputs).logits, before, rtol=0, atol=0)
        with service.temporary_state(state):
            torch.testing.assert_close(model(**inputs).logits, before, rtol=0, atol=0)
        ctl.step(5)
        fork.step()
        assert fork.n == ctl.n == 5
        after = model(**inputs).logits
        assert not torch.equal(before, after)
        torch.testing.assert_close(other(**inputs).logits, after, rtol=0, atol=0)
        assert restored.state_dict() == service.state_dict()
    finally:
        restored.remove()


def test_script_restore_accepts_older_checkpoint_records(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    expected = model(**tokenizer("AB")).logits
    state = service.state_dict()
    for record in state["interventions"]:
        record.pop("applied_specification", None)
    service.load_state_dict(state, restore_active=True)
    torch.testing.assert_close(model(**tokenizer("AB")).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("applied", [None, add_spec(point="layers.1.mlp_out"), add_spec(coefficient=2**53 + 1)])
def test_invalid_applied_snapshot_refuses_before_live_replacement(controlled, applied):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before = service.state_dict()
    state = deepcopy(before)
    state["interventions"][0]["applied_specification"] = applied
    with pytest.raises(InterventionError):
        service.load_state_dict(state, restore_active=True)
    assert service.state_dict() == before
