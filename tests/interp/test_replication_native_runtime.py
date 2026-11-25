"""Native action exports bind actual forwards and roll back with the transaction."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest
import torch

from caliper.interp import _replication_actions as actions
from tests.interp.test_replication_coordinates import original_case, oracle
from tests.interp.test_replication_groups import replicated_hosted


@pytest.mark.parametrize("trace_first", [False, True])
def test_complete_hosted_status_fits_protocol_for_every_published_case(replicated_hosted, trace_first):
    from caliper import _interpret_protocol as protocol, _wire
    from tests.interp.test_service import call

    run, route = replicated_hosted
    if trace_first:
        queued = call(route, "trace.run", {"prompt": run.prompt.value, "max_targets": 4,
                                           "max_nodes": 64, "max_edges": 128}, request_id=90)
        assert queued["status"] == "queued"
        run.boundary()
        replies = [reply for reply in route.transport.replies if reply.request_id == 90]
        assert len(replies) == 1 and replies[0].status == "ok"
    for sequence, cell in enumerate(actions.CASES, 1):
        action = f"published_dallas_v1.cell{cell}"
        queued = route.on_control(None, _wire.Control(sequence, "run", name=action).to_meta())
        assert queued["status"] == "queued"
        run.boundary()
        replies = [reply for reply in route.transport.replies if reply.request_id == sequence]
        assert len(replies) == 1 and replies[0].status == "ok"
        status, result = run.service.submit({"version": 1, "method": "trace.status", "params": {}}, None)
        assert status == "ok"
        assert result["state"] == ("done" if trace_first else "idle")
        assert protocol.MAX_DEPTH == 8 and protocol.bounded(result), cell
        reply = call(route, "trace.status", request_id=100 + sequence)
        assert reply["status"] == "ok" and protocol.bounded(reply["result"])
        inference = reply["result"]["inference"]
        record = inference["replication"]["result"]
        assert record == run.replication.last_result
        assert record["case_id"] == inference["comparison"]["case_id"] == action
        assert inference["replication"]["active_cell"] == cell
        assert record["sequence"] == sequence
        assert set(record["native_evidence"]["tensors"]) == {"logits", "probabilities"}
        if cell == 29:
            observation = record["observations"]["Say Victoria"]
            assert observation["node"] == dict(zip(("layer", "feature", "position"), actions.GROUPS["Say Victoria"][0]))
            native = observation["native_evidence"]
            saved = torch.load(run.replication.output_directory / native["file"], weights_only=True)
            assert set(saved) == {"activation"}
            assert native["tensor"] == "activation" and native["shape"] == [1]
            assert native["dtype"] == str(saved["activation"].dtype)


def installed_case(runtime, model, tokenizer, service, monkeypatch, nonzero):
    cases = deepcopy(actions.CASES)
    cases[29] = deepcopy(cases[26])
    monkeypatch.setattr(actions, "CASES", cases)
    _, members = runtime.case(29)
    targets = {(item["layer"], item["feature"], item["position"]): item["value"] for item in members}
    expected, features, _, _ = oracle(model, tokenizer("AB")["input_ids"], service.dictionary,
                                      clamps=targets, patterns=runtime.patterns)
    values = features[1][0, 1]
    candidates = (values != 0 if nonzero else values == 0).nonzero().flatten()
    assert candidates.numel() > 0
    node = (1, int(candidates[0]), 1)
    assert node not in targets
    monkeypatch.setattr(actions, "GROUPS", {**actions.GROUPS, "Say Victoria": [node]})
    return expected, values[node[1]].reshape(1), node


@pytest.mark.parametrize("nonzero", [False, True])
def test_case_retains_native_outputs_and_actual_observation(original_case, monkeypatch, nonzero):
    model, tokenizer, service, runtime, captures = original_case
    expected, activation, node = installed_case(runtime, model, tokenizer, service, monkeypatch, nonzero)
    result = runtime.run(29)
    path = runtime.output_directory / result["native_evidence"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["native_evidence"]["sha256"]
    native = torch.load(path, weights_only=True, map_location="cpu")
    assert set(native) == {"logits", "probabilities"}
    assert all(value.dtype == torch.bfloat16 for value in native.values())
    torch.testing.assert_close(native["logits"][1], expected[0, -1], rtol=0, atol=0)
    torch.testing.assert_close(native["probabilities"], native["logits"].softmax(-1), rtol=0, atol=0)
    with np.load(runtime.output_directory / result["evidence"]["file"], allow_pickle=False) as arrays:
        for key, value in native.items():
            np.testing.assert_array_equal(value.float().numpy(), arrays[key])
    observed = result["observations"]["Say Victoria"]
    assert observed["node"] == dict(zip(("layer", "feature", "position"), node))
    descriptor = observed["native_evidence"]
    observation_path = runtime.output_directory / descriptor["file"]
    assert descriptor == {"file": descriptor["file"], "sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
                          "tensor": "activation", "dtype": "torch.bfloat16", "shape": [1]}
    native_observation = torch.load(observation_path, weights_only=True, map_location="cpu")
    assert set(native_observation) == {"activation"}
    assert native_observation["activation"].dtype == torch.bfloat16
    torch.testing.assert_close(native_observation["activation"], activation, rtol=0, atol=0)
    assert all(not values for values in captures)


def test_late_failure_removes_all_new_native_exports(original_case, monkeypatch):
    model, tokenizer, service, runtime, _ = original_case
    installed_case(runtime, model, tokenizer, service, monkeypatch, True)
    prior = runtime.run(10)
    files = set(runtime.directory.iterdir())
    state = deepcopy(service.state_dict())
    def fail_after_export(pair):
        assert any(path.suffix == ".pt" for path in set(runtime.directory.iterdir()) - files)
        raise ValueError("injected publication refusal")
    with pytest.raises(ValueError, match="injected publication refusal"):
        runtime.run(29, prepare=fail_after_export)
    assert runtime.last_result == prior
    assert service.state_dict() == state
    assert set(runtime.directory.iterdir()) == files
    assert runtime.last_attempt["status"] == "error"
