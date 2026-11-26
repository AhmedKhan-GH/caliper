"""Offline checks for real interpretation objects and deferred wire replies."""
import threading
from types import SimpleNamespace

import pytest
import torch

from caliper import _controls, _wire
from caliper import _interpret_protocol as protocol
from caliper.interp._backend import make_tiny_backend
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._memory import CleanStop
from caliper.interp._service import InterpretationService, Refused, register_service, unregister_service


class Transport:
    def __init__(self):
        self.replies = []
        self.allowed = True

    def may_control(self, conn):
        return self.allowed

    def reply(self, conn, metadata):
        self.replies.append(_wire.decode(_wire.encode_frame(_wire.MsgType.CONTROL_REPLY, metadata)))

    def broadcast(self, *args, **kwargs):
        pass

    def controllers(self):
        return 1


def call(route, method, params=None, request_id=1):
    message = _wire.Control(request_id, "interpret", interpretation={"version": 1, "method": method, "params": params or {}})
    return route.on_control(None, message.to_meta())


@pytest.fixture
def actual(tmp_path, request):
    scale = float(getattr(request, "param", 1.0))
    backend = make_tiny_backend()
    if scale != 1.0:
        with torch.no_grad():
            for layer in backend.dictionary.covered_layers:
                backend.dictionary.raw[layer].W_dec.mul_(scale)
    build_index(
        tmp_path,
        [("fixture", [2, 37, 38, 39, 40] * 12)],
        model=backend.model,
        tokenizer=backend.tokenizer,
        dictionary=backend.dictionary,
        provenance={"partition": "fixture"},
        context_length=17,
    )
    index = FeatureIndex(
        tmp_path, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary
    )
    service = InterpretationService(
        model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary, index=index
    )
    yield service
    service.close()
    index.close()


def test_real_service_operations(actual, monkeypatch):
    published = {}
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    register_service(actual, controls=controls)
    layer, feature = 0, int(actual.index.counts[0].argmax())
    requests = [
        ("feature.dictionary", {}),
        ("feature.show", {"layer": layer, "feature": feature}),
        ("feature.examples", {"layer": layer, "feature": feature}),
        ("trace.run", {"prompt": "AB", "max_targets": 8, "max_nodes": 64, "max_edges": 128}),
    ]
    for number, (method, params) in enumerate(requests, 1):
        assert call(route, method, params, number)["status"] == "queued"
        assert actual.step(number) == 1
        reply = transport.replies[-1]
        assert reply.status == "ok", reply.error
        assert reply.result
    assert call(route, "trace.status")["result"]["state"] == "done"
    assert call(route, "trace.read", {"node_limit": 1, "edge_limit": 1})["status"] == "queued"
    actual.step(5)
    result = transport.replies[-1].result
    assert len(result["nodes"]) == 1
    assert result["counts"]["nodes"] > 1
    assert set(published) == {
        "trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
        "examples.tokens", "examples.activations", "examples.example_ids",
        "examples.valid_tokens", "examples.label_text", "trace.label_text",
        "trace.probs", "trace.logit_tokens", "trace.logit_label_text",
    }
    assert published["trace.label_text"].shape == (544, 16)
    assert published["trace.logit_label_text"].shape == (10, 16)
    assert published["examples.tokens"].shape == (16, 25)
    assert published["examples.label_text"].shape == (400, 16)
    unregister_service(actual, controls=controls)
    assert controls.interpretation_service is None


def test_missing_resources_and_read_permission():
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    assert call(route, "feature.dictionary")["error"]["code"] == "interpret.unsupported"
    service = register_service(controls=controls)
    transport.allowed = False
    assert call(route, "trace.run", {"prompt": "A"})["error"]["code"] == _wire.ERR_FORBIDDEN
    assert call(route, "feature.dictionary")["status"] == "queued"
    service.step(1)
    assert transport.replies[-1].error["code"] == "interpret.missing_dictionary"
    assert call(route, "trace.read")["status"] == "queued"
    service.step(2)
    assert transport.replies[-1].error["code"] == "interpret.missing_trace"
    unregister_service(service, controls=controls)


@pytest.mark.parametrize("lazy", [False, True])
def test_index_registration_binds_resources(actual, lazy):
    service = InterpretationService(**({"index_factory": lambda: actual.index} if lazy else {"index": actual.index}))
    replies = []
    service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, lambda *args: replies.append(args))
    service.step()
    assert replies[0][0] == "ok", replies[0][2]
    assert replies[0][1]["identity"]["index_hash"] == actual.index.metadata["index_hash"]
    assert service.model is actual.model and service.tokenizer is actual.tokenizer
    assert service.dictionary is actual.dictionary
    service.close()


def test_resource_registration_refuses_mixed_models(actual):
    other = make_tiny_backend(seed=99).model
    with pytest.raises(ValueError, match="share one model"):
        InterpretationService(index=actual.index, model=other)


def test_ordinary_trace_error_remains_valid_on_wire_and_in_status():
    def fail():
        error = ValueError("bad\0" + "é" * 1024 + "\ud800")
        error.code = "invalid\0code"
        raise error

    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    service = register_service(InterpretationService(tracer_factory=fail), controls=controls)
    assert call(route, "trace.run", {"prompt": "A"})["status"] == "queued"
    service.step()
    error = transport.replies[-1].error
    assert error["code"] == "interpret.refused"
    assert "\0" not in error["message"] and len(error["message"].encode("utf-8")) <= 1024
    status = call(route, "trace.status")
    assert protocol.bounded(status["result"]) and status["result"]["error"] == error
    unregister_service(service, controls=controls)


@pytest.mark.parametrize("change", [
    {"version": True}, {"version": 2}, {"method": "unknown"}, {"extra": 1}, {"params": []},
    {"params": {"prompt": "x", "max_nodes": True}}, {"params": {"prompt": "x", "max_targets": 5001}},
    {"params": {"prompt": "x", "max_edges": float("nan")}}, {"params": {"prompt": "\ud800"}},
    {"params": {"prompt": "é" * 2049}}, {"params": {"prompt": ""}}, {"params": {"prompt": "x\0"}},
])
def test_request_refusal(change):
    metadata = {"version": 1, "method": "trace.run", "params": {"prompt": "x"}, **change}
    assert not protocol.request(metadata)
    with pytest.raises(_wire.MessageError):
        _wire.Control.from_frame(_wire.Frame(_wire.MsgType.CONTROL, {"request_id": 1, "op": "interpret", "interpretation": metadata}))


def test_resource_stop_terminates_queue_and_cancels_actions():
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    failure = CleanStop("tracer admission")
    service = register_service(
        InterpretationService(tracer_factory=lambda: (_ for _ in ()).throw(failure)),
        controls=controls,
    )
    action = controls.action("later", lambda: pytest.fail("later action ran"))
    try:
        call(route, "trace.run", {"prompt": "AB"}, 1)
        route.on_control(None, _wire.Control(2, "run", name="later").to_meta())
        with pytest.raises(CleanStop):
            controls.checkpoint()
        assert [(r.request_id, r.status) for r in transport.replies] == [(1, "error"), (2, "error")]
        assert not controls.queue and not action.queued
    finally:
        unregister_service(service, controls=controls)
        action.remove()
        route.detach()
