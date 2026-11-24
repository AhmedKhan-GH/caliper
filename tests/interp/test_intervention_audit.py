"""Bounded CPU reproductions for intervention audit corrections."""
from copy import deepcopy

import pytest
import torch

from caliper import _wire as wire
from caliper.interp._intervene import InterventionError
from tests.interp.test_intervene import add_spec, controlled


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
