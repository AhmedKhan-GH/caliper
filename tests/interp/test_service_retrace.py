"""Request completion, interruption, and ownership checks for inline traces."""
import threading
from types import SimpleNamespace

import pytest

from caliper.interp._memory import CleanStop
from caliper.interp._service import InterpretationService, Refused


def enqueue(service, method="trace.run", reply=None):
    replies = []
    service.submit({"version": 1, "method": method,
                    "params": {"prompt": "AB"} if method == "trace.run" else {}},
                   reply if reply is not None else lambda *args: replies.append(args))
    return replies


@pytest.fixture
def service():
    graph = SimpleNamespace(read=lambda **kwargs: {"nodes": [], "edges": [], "tokens": []})
    result = InterpretationService(tracer=SimpleNamespace(trace=lambda **kwargs: graph, bind_index=lambda index: None))
    yield result
    result.close()


def execute(service, source):
    if source == "automatic":
        return service._retrace({"prompt": "AB"}, 3)
    replies = enqueue(service)
    service.step(3)
    return replies[0]


@pytest.mark.parametrize("source", ["queued", "automatic"])
@pytest.mark.parametrize("stage", ["admission", "trace", "complete"])
def test_trace_failure_releases_status_and_allows_recovery(service, monkeypatch, source, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("ordinary trace failure")

    guard = service._guard_resources

    def guarded(current):
        if current == "service request " + stage:
            fail()
        return guard(current)

    with monkeypatch.context() as patch:
        if stage == "trace":
            patch.setattr(service.tracer, "trace", fail)
        else:
            patch.setattr(service, "_guard_resources", guarded)
        response = execute(service, source)
    assert response == ("error", None, {"code": "interpret.refused", "message": "ordinary trace failure"})
    assert service.status == {"state": "error", "seconds": 0., "step": 3, "error": response[2]}
    assert not service.running and service.running_thread is None
    assert service._request_stack is None and not service._borrowed and service.failure is None
    replies = enqueue(service)
    service.step(4)
    assert replies[0][0] == "ok" and service.status["state"] == "done"


@pytest.mark.parametrize("source", ["queued", "automatic"])
@pytest.mark.parametrize("failure_type", [CleanStop, KeyboardInterrupt, SystemExit])
def test_trace_stop_and_interrupt_propagate_with_released_ownership(service, monkeypatch, source, failure_type):
    failure = failure_type("trace stopped")
    pending = []

    def fail(**params):
        pending.extend([enqueue(service, "trace.read")])
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(service.tracer, "trace", fail)
        with pytest.raises(failure_type) as caught:
            execute(service, source)
    assert caught.value is failure
    assert not service.running and service.running_thread is None
    assert service._request_stack is None and not service._borrowed
    assert service.status["state"] == "error"
    if failure_type is CleanStop:
        assert service.failure is failure and not service.queue
        assert pending[0][0][2]["code"] == "interpret.stopped"
        with pytest.raises(Refused, match="stopped"):
            enqueue(service)
    else:
        assert service.failure is None and len(service.queue) == 1 and not pending[0]
        replies = enqueue(service)
        service.step(4)
        assert len(pending[0]) == 1 and replies[0][0] == "ok"


@pytest.mark.parametrize("source", ["queued", "automatic"])
def test_trace_owns_status_until_completion_guard_returns(service, monkeypatch, source):
    guard = service._guard_resources
    observations = []

    def checked(stage):
        if stage == "service request complete":
            observations.append(service.status["state"])
            with pytest.raises(Refused, match="trace already"):
                enqueue(service)
        return guard(stage)

    monkeypatch.setattr(service, "_guard_resources", checked)
    assert execute(service, source)[0] == "ok"
    assert observations == ["running"] and not service.queue


def test_automatic_trace_retains_pending_request_without_execution(service, monkeypatch):
    replies = enqueue(service)
    pending = service.status
    monkeypatch.setattr(service.tracer, "trace", lambda **kw: pytest.fail("pending request was bypassed"))
    assert service._retrace({"prompt": "AC"}, 3) is None
    assert service.status is pending and len(service.queue) == 1 and not replies


def test_reply_callback_can_queue_next_trace_without_status_overwrite(service):
    replies = []

    def completed(*args):
        replies.append(args)
        enqueue(service)

    enqueue(service, reply=completed)
    enqueue(service, "feature.dictionary")
    assert service.step(3) == 2
    assert replies[0][0] == "ok"
    assert service.status == {"state": "pending", "seconds": 0., "step": None}
    assert len(service.queue) == 1
    assert service.step(4) == 1 and service.status["state"] == "done"


def test_close_waits_for_automatic_trace_and_recursive_work_refuses(service, monkeypatch):
    entered, release, waiting, finished = (threading.Event() for _ in range(4))
    errors, results = [], []
    trace, wait = service.tracer.trace, service.idle.wait

    def traced(**params):
        assert service.running_thread == threading.get_ident()
        assert service.step() == 0
        assert service._retrace(params, 3) is None
        with pytest.raises(Refused, match="own service"):
            service.close()
        entered.set()
        assert release.wait(5)
        return trace(**params)

    def wait_for_request(*args):
        waiting.set()
        return wait(*args)

    def run():
        try:
            results.append(service._retrace({"prompt": "AB"}, 3))
        except BaseException as error:
            errors.append(error)

    def close():
        service.close()
        finished.set()

    monkeypatch.setattr(service.tracer, "trace", traced)
    monkeypatch.setattr(service.idle, "wait", wait_for_request)
    worker, closer = threading.Thread(target=run), threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert waiting.wait(5) and not finished.is_set()
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not worker.is_alive() and not closer.is_alive() and not errors
    assert results[0][0] == "ok" and finished.is_set()
