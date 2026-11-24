"""Offline checks for real interpretation objects and deferred wire replies."""
import json
import threading

import pytest

from caliper import _controls, _wire
from caliper import _interpret_protocol as protocol
from caliper.interp._service import InterpretationService, register_service, unregister_service
from .fixture import TinyTokenizer
from .test_dictionary import write_fixture


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
def actual(tmp_path):
    from caliper.interp import load_dictionary
    from caliper.interp._index import FeatureIndex, build_index
    model, _ = write_fixture(tmp_path)
    tokenizer = TinyTokenizer()
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, [("fixture", [65, 66, 67, 68] * 12)], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=17)
    index = FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)
    service = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary, index=index)
    yield service
    service.close()


def test_real_service_all_m7_operations(actual, monkeypatch):
    import caliper.interp
    published = {}
    monkeypatch.setattr(caliper.interp, "publish", lambda name, tensor: published.update({name: tensor}))
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    register_service(actual, controls=controls)
    layer, feature = 0, int(actual.index.counts[0].argmax())
    requests = [("feature.dictionary", {}), ("feature.show", {"layer": layer, "feature": feature}),
                ("feature.examples", {"layer": layer, "feature": feature}),
                ("trace.run", {"prompt": "AB", "max_targets": 4, "max_nodes": 64, "max_edges": 128})]
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
    assert len(result["nodes"]) == 1 and len(result["edges"]) == 1
    assert result["counts"]["nodes"] > 1
    assert set(published) == {"trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
                               "examples.tokens", "examples.activations", "examples.example_ids",
                               "examples.valid_tokens", "examples.label_text"}
    assert published["examples.tokens"].shape == (16, 25)
    assert published["examples.label_text"].shape == (400, 16)
    assert published["examples.example_ids"].tolist() == list(range(16))
    assert actual.index.examples(layer, feature)[0]["token_ids"][16] in published["examples.tokens"][0].tolist()
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


@pytest.mark.parametrize("change", [
    {"version": True}, {"version": 2}, {"method": "unknown"}, {"extra": 1}, {"params": []},
    {"params": {"prompt": "x", "max_nodes": True}}, {"params": {"prompt": "x", "max_targets": 513}},
    {"params": {"prompt": "x", "max_edges": float("nan")}}, {"params": {"prompt": "\ud800"}},
    {"params": {"prompt": "é" * 2049}}, {"params": {"prompt": ""}}, {"params": {"prompt": "x\0"}},
])
def test_request_refusal(change):
    metadata = {"version": 1, "method": "trace.run", "params": {"prompt": "x"}, **change}
    assert not protocol.request(metadata)
    with pytest.raises(_wire.MessageError):
        _wire.Control.from_frame(_wire.Frame(_wire.MsgType.CONTROL, {"request_id": 1, "op": "interpret", "interpretation": metadata}))


@pytest.mark.parametrize("result", [{"x": float("inf")}, {"x": "x" * 4097}, {"x": [0] * 513},
                                    {"x": 2**53}, {"x": "\ud800"}, {str(i): i for i in range(65)}])
def test_reply_bounds(result):
    assert not protocol.bounded(result)
    with pytest.raises(_wire.MessageError):
        _wire.ControlReply.from_frame(_wire.Frame(_wire.MsgType.CONTROL_REPLY,
            {"request_id": 1, "op": "interpret", "status": "ok", "result": result}))


def test_old_controls_and_extension_separation():
    for metadata in ({"request_id": 1, "op": "pause"}, {"request_id": 2, "op": "set", "name": "lr", "value": .1},
                     {"request_id": 3, "op": "intervene", "intervene": {"set": {"point": "layer", "op": "ablate"}}}):
        assert _wire.Control.from_frame(_wire.Frame(_wire.MsgType.CONTROL, metadata)).to_meta() == metadata
    for metadata in ({"request_id": 1, "op": "pause", "interpretation": {}},):
        with pytest.raises(_wire.MessageError):
            _wire.Control.from_frame(_wire.Frame(_wire.MsgType.CONTROL, metadata))
    with pytest.raises(_wire.MessageError):
        _wire.ControlReply.from_frame(_wire.Frame(_wire.MsgType.CONTROL_REPLY,
                                      {"request_id": 1, "op": "pause", "status": "ok", "result": {}}))


def test_lazy_factory_runs_only_at_boundary_and_status_does_not_wait():
    entered, release = threading.Event(), threading.Event()
    thread_ids = []

    def factory():
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        raise ValueError("fixture refusal")

    service = InterpretationService(tracer_factory=factory)
    replies = []
    assert service.submit({"version": 1, "method": "trace.run", "params": {"prompt": "x"}},
                          lambda *args: replies.append(args)) == ("queued", None)
    assert not thread_ids
    thread = threading.Thread(target=service.step)
    thread.start()
    assert entered.wait(5)
    assert service.submit({"version": 1, "method": "trace.status", "params": {}}, None)[1]["state"] == "running"
    release.set()
    thread.join(5)
    assert not thread.is_alive() and replies[0][0] == "error"
