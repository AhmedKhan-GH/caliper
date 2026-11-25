"""M8 wire schemas, real lifecycle parity, and absent-handler refusals."""
import json
from pathlib import Path

import pytest

from caliper import _wire as wire
from caliper._intervention_protocol import specification
from caliper._interpret_protocol import request
from caliper.interp._service import InterpretationService
from .test_intervene import controlled
from .test_service import actual

CASES = json.loads((Path(__file__).parents[1] / "fixtures/wire_v3/m8_interventions.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_codec_cases(case):
    value = case["specification"]
    assert specification(value) == case["accepted"]
    message = wire.Control(1, "intervene", intervene={"set": value})
    if case["accepted"]:
        assert wire.decode(wire.encode(message)).intervene == {"set": value}
    else:
        with pytest.raises(wire.MessageError):
            wire.decode(wire.encode(message))


@pytest.mark.parametrize("fields", [
    {"vector": [1.] * 16385}, {"vector": [float("nan")]}, {"coefficient": float("inf")},
    {"point": "\ud800"}, {"point": "é" * 128}, {"point": "x\0"},
])
def test_intervention_bounds(fields):
    value = {"point": "layers.0.mlp_out", "kind": "add", "vector": [1.], "coefficient": 1., **fields}
    assert not specification(value)
    assert not wire._intervene_request({"set": value})


def test_real_lifecycle_entries_decode_without_lost_fields(controlled):
    _, _, controls, service = controlled
    requests = [
        {"point": "layers.0.mlp_out", "kind": "add", "vector": [.1] * 64, "coefficient": .5},
        {"point": "layers.1.mlp_out", "kind": "scale", "feature": 0, "multiplier": 2., "dictionary": "fixture"},
        {"point": "layers.2.mlp_out", "kind": "clamp", "feature": 0, "value": 0., "positions": [1]},
        {"point": "layers.3.mlp_out", "kind": "patch", "source_prompt": "AB", "source_position": 1}]
    for value in requests:
        decoded = wire.decode(wire.encode(wire.Control(1, "intervene", intervene={"set": value})))
        service.set(decoded.intervene["set"])
    for boundary in (False, True):
        if boundary:
            controls.step()
        metadata = controls.controls_meta()
        decoded = wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, metadata))
        assert decoded.interventions == metadata["interventions"]
        for entry, value in zip(decoded.interventions["active"], requests):
            for key, expected in value.items():
                assert entry[key] == expected
            assert entry["state"] == ("active" if boundary else "pending")


def test_m8_methods_refuse_absent_execution_and_malformed_metrics():
    service = InterpretationService()
    replies = []
    sweep = {"clean_prompt": "AB", "corrupted_prompt": "AC",
             "metric": {"positive_token": 1, "negative_token": 2}}
    for method, params in (("patch.sweep", sweep), ("trace.validate", {})):
        assert service.submit({"version": 1, "method": method, "params": params}, lambda *args: replies.append(args))[0] == "queued"
        service.step()
        assert replies[-1][0] == "error" and replies[-1][2]["code"] == "interpret.unsupported"
    for params in ({}, {**sweep, "clean_prompt": "é" * 2049},
                   {**sweep, "metric": {"positive_token": True, "negative_token": 1}},
                   {**sweep, "metric": {"positive_token": 1, "negative_token": 1}},
                   {**sweep, "metric": {"positive_token": float("nan"), "negative_token": 1}}):
        assert not request({"version": 1, "method": "patch.sweep", "params": params})
    assert not request({"version": 1, "method": "trace.validate", "params": {"prompt": "A"}})


def test_real_feature_queries_suspend_active_interventions(actual):
    import torch
    from caliper._controls import Controls
    from caliper.interp._service import register_service, unregister_service
    controls = Controls()
    handle = controls.interventions_on(actual.model)
    handle.configure(dictionary=actual.dictionary, tokenizer=actual.tokenizer, dictionary_id="fixture")
    register_service(actual, controls=controls)
    feature = int(actual.index.counts[1].argmax())
    expected = actual.index.examples(1, feature)
    ids = actual.tokenizer("AB")["input_ids"]
    baseline = actual.model(ids).logits.detach()
    try:
        handle.set({"kind": "add", "point": "layers.0.mlp_out", "vector": [.5] * 64, "coefficient": 1.})
        controls.step()
        changed = actual.model(ids).logits.detach()
        assert not torch.equal(changed, baseline)
        actual.index.clear_cache()
        replies = []
        actual.submit({"version": 1, "method": "feature.examples", "params": {"layer": 1, "feature": feature}},
                      lambda *args: replies.append(args))
        controls.step()
        assert replies[-1][0] == "ok", replies[-1]
        assert replies[-1][1]["examples"] == expected
        torch.testing.assert_close(actual.model(ids).logits, changed)
    finally:
        unregister_service(actual, controls=controls)
        handle.remove()


def test_missing_index_and_model_are_structured_refusals(actual):
    for service, method, params, code in (
        (InterpretationService(dictionary=actual.dictionary), "feature.show", {"layer": 0, "feature": 0}, "interpret.missing_index"),
        (InterpretationService(), "trace.run", {"prompt": "AB"}, "interpret.missing_model"),
    ):
        replies = []
        assert service.submit({"version": 1, "method": method, "params": params}, lambda *args: replies.append(args))[0] == "queued"
        service.step()
        assert replies[-1][0] == "error" and replies[-1][2]["code"] == code


TRACE_CASES = json.loads((Path(__file__).parents[1] / "fixtures/wire_v3/replication_trace.json").read_text())


@pytest.mark.parametrize("case", TRACE_CASES, ids=lambda case: case["name"])
def test_shared_replication_trace_cases(case):
    value = case["request"]
    assert request(value) == case["accepted"]
    message = wire.Control(1, "interpret", interpretation=value)
    if case["accepted"]:
        assert wire.decode(wire.encode(message)).interpretation == value
    else:
        with pytest.raises(wire.MessageError):
            wire.encode(message)
        with pytest.raises(wire.MessageError):
            wire.decode(wire.encode_frame(wire.MsgType.CONTROL, message.to_meta()))
