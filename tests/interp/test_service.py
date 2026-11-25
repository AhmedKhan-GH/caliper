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
def actual(tmp_path, request):
    from caliper.interp import load_dictionary
    from caliper.interp._index import FeatureIndex, build_index
    from .fixture import TinyLM
    import torch
    model = TinyLM()
    with torch.no_grad():
        for block in model.layers:
            block.mlp.W_dec.mul_(getattr(request, "param", 1.0))
    model, _ = write_fixture(tmp_path, model)
    tokenizer = TinyTokenizer()
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, [("fixture", [65, 66, 67, 68] * 12)], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=17)
    index = FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)
    service = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary, index=index)
    yield service
    service.close()
    index.close()


def test_real_service_all_m7_operations(actual, monkeypatch):
    published = {}
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
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
                               "examples.valid_tokens", "examples.label_text", "trace.label_text",
                               "trace.probs", "trace.logit_tokens", "trace.logit_label_text"}
    assert published["trace.label_text"].shape == (544, 16)
    assert published["trace.logit_label_text"].shape == (10, 16)
    logits = [node for node in actual.graph.nodes if node["kind"] == "logit"][:10]
    assert published["trace.logit_tokens"][:len(logits)].tolist() == [n["token_id"] for n in logits]
    assert published["trace.probs"][0].tolist() == published["trace.probs"][1].tolist()
    assert published["trace.probs"][0, :len(logits)].tolist() == pytest.approx([n["probability"] for n in logits])
    assert len(actual.graph.tensors) == 5
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


@pytest.mark.parametrize("lazy", [False, True])
def test_index_registration_binds_resources_and_qualifies_dictionary(actual, lazy):
    service = InterpretationService(**({"index_factory": lambda: actual.index} if lazy else {"index": actual.index}))
    replies = []
    service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, lambda *args: replies.append(args))
    service.step()
    assert replies[0][0] == "ok", replies[0][2]
    assert replies[0][1]["identity"]["index_hash"] == actual.index.metadata["index_hash"]
    assert service.model is actual.model and service.tokenizer is actual.tokenizer
    assert service.dictionary is actual.dictionary
    service.close()


def test_resource_only_examples_publish_on_model_device(actual, monkeypatch):
    import torch
    service = InterpretationService(index=actual.index)
    published = {}
    # The meta device tests placement without an accelerator allocation.
    monkeypatch.setattr(actual.model, "parameters", lambda: iter((torch.empty(1, device="meta"),)))
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
    service._publish_examples([])
    assert len(published) == 5
    assert all(value.device.type == "meta" for value in published.values())
    service.close()


def test_resource_registration_refuses_mixed_models(actual):
    from .fixture import TinyLM
    with pytest.raises(ValueError, match="share one model"):
        InterpretationService(index=actual.index, model=TinyLM())


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


def test_close_disconnected_waiter_does_not_block_detachment():
    controls = _controls.Controls()
    service = register_service(controls=controls)
    replies = []

    def disconnected(*args):
        raise ConnectionError("fixture connection closed")

    request = {"version": 1, "method": "feature.dictionary", "params": {}}
    service.submit(request, disconnected)
    service.submit(request, lambda *args: replies.append(args))
    unregister_service(service, controls=controls)
    assert service.closed and not service.pending
    assert controls.interpretation_service is None
    assert replies == [("error", None, {"code": "interpret.closed", "message": "The interpretation service closed."})]


def test_status_read_keeps_idle_controls_fast_and_work_wakes_them():
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    controls.state = _controls.RUN_RUNNING
    controls._report_due = False
    controls._update_busy()
    assert not controls.busy
    service = register_service(controls=controls)
    assert not controls.busy
    assert call(route, "trace.status")["status"] == "ok"
    assert not controls.busy
    assert call(route, "feature.dictionary")["status"] == "queued"
    assert controls.busy
    unregister_service(service, controls=controls)
    assert not controls.busy


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


@pytest.mark.parametrize("method", ["feature.show", "feature.examples"])
def test_legacy_ablation_refuses_before_index_access(actual, monkeypatch, method):
    import torch
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    handle = controls.interventions_on(actual.model)
    # M7 has no public suspension property. Keep its actual hook lifecycle.
    monkeypatch.delattr(_controls.Controls, "intervention_service", raising=False)
    register_service(actual, controls=controls)
    index, factories = actual.index, []
    feature = int(index.counts[1].argmax())
    expected = index.examples(1, feature)
    index.clear_cache()
    actual.index = None
    actual.factories["index"] = lambda: factories.append("index") or index
    ids = actual.tokenizer("AB")["input_ids"]
    baseline = actual.model(ids).logits.detach()
    try:
        route.on_control(None, {"request_id": 1, "op": "intervene",
            "intervene": {"set": {"point": "layers.0.mlp_out", "op": "ablate"}}})
        controls.step()
        assert not torch.equal(actual.model(ids).logits, baseline)
        assert call(route, method, {"layer": 1, "feature": feature})["status"] == "queued"
        controls.step()
        assert transport.replies[-1].error["code"] == "interpret.baseline_required"
        assert not factories
        assert actual.index is None
        route.on_control(None, {"request_id": 2, "op": "intervene", "intervene": {"clear_all": True}})
        controls.step()
        assert call(route, "feature.examples", {"layer": 1, "feature": feature})["status"] == "queued"
        controls.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert transport.replies[-1].result["examples"] == expected
        assert factories == ["index"]
    finally:
        unregister_service(actual, controls=controls)
        handle.remove()


@pytest.mark.parametrize("method", ["feature.examples", "trace.run"])
@pytest.mark.parametrize("replace", [False, True])
def test_service_close_waits_for_real_work_and_publication(actual, monkeypatch, method, replace):
    from caliper.interp import _publish
    controls = _controls.Controls()
    register_service(actual, controls=controls)
    index = actual.index
    name = "index" if method == "feature.examples" else "tracer"
    resource = actual._resource(name)
    setattr(actual, name, None)
    entered, release, waiting, finished = (threading.Event() for _ in range(4))
    publications, replies, failures = [], [], []
    replacement = InterpretationService()

    def factory():
        entered.set()
        assert release.wait(5)
        assert not index._closed
        return resource

    publish_many = _publish._publish_many

    def publish(values):
        assert controls.interpretation_service is actual
        assert not finished.is_set()
        publications.extend(values)
        publish_many(values)

    wait = actual.idle.wait

    def close_wait(timeout=None):
        waiting.set()
        return wait(timeout)

    def close():
        try:
            if replace:
                register_service(replacement, controls=controls)
            else:
                unregister_service(actual, controls=controls)
            index.close()
            finished.set()
        except BaseException as exc:
            failures.append(exc)

    actual.factories[name] = factory
    monkeypatch.setattr(actual.idle, "wait", close_wait)
    monkeypatch.setattr(_publish, "_publish_many", publish)
    params = {"layer": 1, "feature": int(index.counts[1].argmax())} if name == "index" else {
        "prompt": "AB", "max_targets": 4, "max_nodes": 64, "max_edges": 128}
    actual.submit({"version": 1, "method": method, "params": params}, lambda *args: replies.append(args))
    worker = threading.Thread(target=actual.step)
    closer = threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        actual.submit({"version": 1, "method": "feature.dictionary", "params": {}},
                      lambda *args: replies.append(args))
        closer.start()
        assert waiting.wait(5)
        assert not finished.is_set()
        assert controls.interpretation_service is actual
        assert not index._closed
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not worker.is_alive() and not closer.is_alive()
    assert not failures
    assert finished.is_set() and index._closed
    assert controls.interpretation_service is (replacement if replace else None)
    assert replies[0][0] == "ok", replies
    assert replies[1][2]["code"] == "interpret.closed"
    assert publications
    before = list(publications)
    assert actual.step() == 0
    assert publications == before
    if replace:
        unregister_service(replacement, controls=controls)


def test_service_refuses_recursive_close_without_detachment():
    from caliper.interp._service import Refused
    controls = _controls.Controls()
    service = register_service(controls=controls)
    replies = []

    def factory():
        unregister_service(service, controls=controls)

    service.factories["dictionary"] = factory
    service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, lambda *args: replies.append(args))
    service.step()
    assert replies[0][2]["code"] == "interpret.busy"
    assert controls.interpretation_service is service
    assert not service.closed
    unregister_service(service, controls=controls)
    with pytest.raises(Refused, match="closed"):
        register_service(service, controls=controls)
    assert controls.interpretation_service is None


@pytest.mark.parametrize("legacy_owner", [True, False])
def test_default_tracer_uses_registered_owner(actual, monkeypatch, legacy_owner):
    import torch
    if legacy_owner:
        monkeypatch.delattr(_controls.Controls, "intervention_service", raising=False)
    owner, transport = _controls.Controls(), Transport()
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    route = owner.attach(transport)
    watched = owner.interventions_on(actual.model)
    register_service(actual, controls=owner)
    params = {"prompt": "AB", "max_targets": 0}
    try:
        assert call(route, "trace.run", params)["status"] == "queued"
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert actual.tracer.controls is owner
        baseline = actual.graph
        logits = actual.tracer._forward_cache[1].logits.clone()
        route.on_control(None, {"request_id": 2, "op": "intervene",
            "intervene": {"set": {"point": "layers.0.mlp", "op": "ablate"}}})
        owner.step()
        assert call(route, "trace.run", params)["status"] == "queued"
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert actual.graph is not baseline
        assert not torch.equal(actual.tracer._forward_cache[1].logits, logits)
        route.on_control(None, {"request_id": 3, "op": "intervene", "intervene": {"clear_all": True}})
        owner.step()
        assert call(route, "trace.run", params)["status"] == "queued"
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert actual.graph is baseline
    finally:
        unregister_service(actual, controls=owner)
        watched.remove()


@pytest.mark.parametrize("replace", [False, True])
def test_recursive_registration_refuses_before_concurrent_close_lock(monkeypatch, replace):
    controls = _controls.Controls()
    service = register_service(controls=controls)
    assert register_service(service, controls=controls) is service
    entered, waiting = threading.Event(), threading.Event()
    replies, failures = [], []
    replacement = InterpretationService()

    class BoundedLifecycleLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            # A failing regression must end both threads without a deadlock.
            assert self.lock.acquire(timeout=2), "Recursive registration waited for the lifecycle lock."

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(controls, "interpretation_registration_lock", BoundedLifecycleLock())
    wait = service.idle.wait

    def close_wait(timeout=None):
        waiting.set()
        return wait(timeout)

    def factory():
        entered.set()
        assert waiting.wait(5)
        register_service(service, controls=controls)

    def close():
        try:
            if replace:
                register_service(replacement, controls=controls)
            else:
                unregister_service(service, controls=controls)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(service.idle, "wait", close_wait)
    service.factories["dictionary"] = factory
    service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, lambda *args: replies.append(args))
    worker = threading.Thread(target=service.step)
    closer = threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
    finally:
        worker.join(7)
        if closer.ident is not None:
            closer.join(7)
    assert not worker.is_alive() and not closer.is_alive()
    assert not failures
    assert replies[0][2]["code"] == "interpret.busy"
    assert service.closed
    assert controls.interpretation_service is (replacement if replace else None)
    if replace:
        assert register_service(replacement, controls=controls) is replacement
        unregister_service(replacement, controls=controls)


@pytest.mark.parametrize("method", ["trace.run", "feature.examples"])
def test_resource_stop_terminates_queue_and_reaches_controls_owner(method):
    from types import SimpleNamespace
    from caliper.interp._memory import CleanStop
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    calls = []
    failure = CleanStop("independent admission reserve")

    def stop(**kwargs):
        calls.append(method)
        raise failure

    service = InterpretationService(dictionary=SimpleNamespace(metadata={}),
        tracer=SimpleNamespace(trace=stop, bind_index=lambda index: None), index=SimpleNamespace(examples=stop))
    register_service(service, controls=controls)
    params = {"prompt": "AB"} if method == "trace.run" else {"layer": 0, "feature": 1}
    if method == "feature.examples":
        service.index.examples = lambda *args, **kwargs: stop(**kwargs)
    try:
        assert call(route, method, params, 1)["status"] == "queued"
        assert call(route, "feature.dictionary", request_id=2)["status"] == "queued"
        with pytest.raises(CleanStop) as caught:
            controls.checkpoint()
        assert caught.value is failure
        assert calls == [method]
        terminal = {reply.request_id: reply for reply in transport.replies if reply.status == "error"}
        assert set(terminal) == {1, 2}
        assert all(reply.error["code"] == "interpret.stopped" for reply in terminal.values())
        assert call(route, "trace.status")["result"]["state"] == "error"
        assert call(route, "feature.dictionary")["error"]["code"] == "interpret.stopped"
        with pytest.raises(CleanStop) as repeated:
            controls.checkpoint()
        assert repeated.value is failure and calls == [method]
    finally:
        unregister_service(service, controls=controls)
        route.detach()
    assert service.step() == 0


def test_resource_stop_survives_failed_terminal_delivery():
    from caliper.interp._memory import CleanStop
    failure = CleanStop("resource stop")
    service = InterpretationService(tracer_factory=lambda: (_ for _ in ()).throw(failure))
    replies = []

    def disconnected(*args):
        raise OSError("disconnected")

    service.submit({"version": 1, "method": "trace.run", "params": {"prompt": "A"}}, disconnected)
    service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, disconnected)
    service.submit({"version": 1, "method": "trace.read", "params": {}}, lambda *args: replies.append(args))
    with pytest.raises(CleanStop) as caught:
        service.step(3)
    assert caught.value is failure and replies[0][2]["code"] == "interpret.stopped"
    assert not service.running and not service.queue
    service.close()


@pytest.mark.parametrize("stops", [True, False])
def test_action_stop_replies_once_and_prevents_later_work(stops):
    from caliper.interp._memory import CleanStop
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    service = register_service(controls=controls)
    failure = CleanStop("action admission\0\ud800" + "😀" * 400) if stops else ValueError("ordinary failure")
    calls = []

    def first():
        calls.append("first")
        raise failure

    first_action = controls.action("first", first)
    second_action = controls.action("second", lambda: calls.append("second"))
    for rid, name in ((1, "first"), (2, "first"), (3, "second")):
        assert route.on_control(None, _wire.Control(rid, "run", name=name).to_meta())["status"] == "queued"
    try:
        if stops:
            with pytest.raises(CleanStop) as caught:
                controls.checkpoint()
            assert caught.value is failure and service.failure is failure
            assert calls == ["first"]
            with pytest.raises(CleanStop):
                controls.checkpoint()
            assert route.on_control(None, _wire.Control(4, "run", name="second").to_meta())["error"]["code"] == "interpret.stopped"
        else:
            with pytest.warns(RuntimeWarning, match="ordinary failure"):
                controls.checkpoint()
            assert calls == ["first", "second"] and service.failure is None
        replies = transport.replies
        assert [reply.request_id for reply in replies] == [1, 2, 3]
        assert [reply.status for reply in replies] == ["error", "error", "error" if stops else "ok"]
        if stops:
            assert all("\0" not in reply.error["message"] and
                       len(reply.error["message"].encode("utf-8")) <= 1024 for reply in replies)
        assert not first_action.running and not second_action.queued and not controls.queue
    finally:
        unregister_service(service, controls=controls)
        first_action.remove()
        second_action.remove()
        route.detach()


def test_service_stop_cancels_actions_before_their_callbacks():
    from caliper.interp._memory import CleanStop
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    failure = CleanStop("tracer admission")
    service = register_service(InterpretationService(
        tracer_factory=lambda: (_ for _ in ()).throw(failure)), controls=controls)
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


def test_ordinary_controls_actions_need_no_torch_installation():
    from pathlib import Path
    import subprocess
    import sys
    source = Path(__file__).resolve().parents[2] / "python"
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
from caliper._controls import Controls
assert "torch" not in sys.modules
controls = Controls()
action = controls.action("plain", lambda: "done")
assert controls._run_action(action) == (True, "done")
controls.checkpoint()
assert "torch" not in sys.modules
action.remove()
'''
    result = subprocess.run([sys.executable, "-S", "-c", script, str(source)],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_trace_options_reach_tracer_and_result_metadata():
    from types import SimpleNamespace
    from unittest.mock import Mock
    params = {"prompt": "Fact: the capital of the state containing Dallas is",
              "profile": "circuit_tracer_82996f14", "max_targets": 5000, "max_nodes": 8192,
              "max_edges": 262144, "max_n_logits": 10, "desired_logit_prob": 0.95,
              "node_threshold": 0.7, "edge_threshold": 0.9, "batch_size": 512, "update_interval": 4}
    summary = {"nodes": [], "edges": [], "tokens": [], "settings": params,
               "counts": {"nodes": 855, "edges": 34801}}
    graph = SimpleNamespace(read=Mock(return_value=summary))
    tracer = SimpleNamespace(trace=Mock(return_value=graph), bind_index=Mock())
    service = InterpretationService(tracer=tracer)
    replies = []
    try:
        assert service.submit({"version": 1, "method": "trace.run", "params": params},
                              lambda *args: replies.append(args))[0] == "queued"
        service.step(7)
        tracer.trace.assert_called_once_with(**params)
        graph.read.assert_called_once_with(node_limit=0, edge_limit=0)
        assert replies[-1][0] == "ok", replies
        assert replies[-1][1]["settings"] == params
        assert replies[-1][1]["counts"] == summary["counts"]
        page = {"node_offset": 768, "edge_offset": 32768, "node_limit": 256, "edge_limit": 256}
        assert service.submit({"version": 1, "method": "trace.read", "params": page},
                              lambda *args: replies.append(args))[0] == "queued"
        service.step(8)
        graph.read.assert_called_with(**page)
        assert replies[-1][0] == "ok"
    finally:
        service.close()
