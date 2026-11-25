"""CPU regressions for automatic trace failures and queued request ownership."""
from types import SimpleNamespace

import pytest

from caliper import _controls, _interpret_protocol as protocol, _wire
from caliper.interp._trace import TraceTimeout
from .test_hosted import hosted_run, _trace_at_boundary
from .test_hosted_trace_admission import case as gemma_case
from .test_service import call


@pytest.fixture(autouse=True)
def in_process_transport(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))


def queue_action(route, name, request_id=10):
    assert route.on_control(None, _wire.Control(request_id, "run", name=name).to_meta())["status"] == "queued"


def assert_released(run):
    assert not run.service.running and run.service.running_thread is None
    assert not run.service.boundary_running and run.service.boundary_thread is None
    assert run.service._request_stack is None and not run.service._borrowed


@pytest.mark.parametrize("kind", ["ablate", "add"])
def test_unsupported_live_edit_reports_error_and_recovers_after_clear(hosted_run, kind):
    import torch
    from caliper.interp import capture
    from caliper.interp._output import internal_forward
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    with internal_forward(), torch.no_grad(), capture(run.model, ["norm"]):
        run.model(**run.tokenizer("AB"))
    run.retrace.set(True)
    spec = {"point": "norm", "kind": kind}
    if kind == "add":
        spec.update(vector=[.01] * 64, coefficient=1.)
    run.interventions.set(spec)
    run.boundary()
    assert run.interventions.specifications()[0]["point"] == "norm"
    status = call(route, "trace.status")["result"]
    assert status["state"] == "error" and status["error"]["code"] == "interpret.refused"
    assert "norm" in status["error"]["message"] and protocol.bounded(status)
    assert run.service.graph is graph and run.service.failure is None
    assert run.status.value == status["error"]["message"]
    assert_released(run)
    run.interventions.clear()
    run.boundary()
    assert call(route, "trace.run", {"prompt": "AB", "max_targets": 0})["status"] == "queued"
    run.boundary()
    assert route.transport.replies[-1].status == "ok"
    assert run.service.status["state"] == "done"


@pytest.mark.parametrize("failure_type", [ValueError, NotImplementedError, RuntimeError, TraceTimeout])
def test_automatic_failure_preserves_actions_pause_and_bounded_status(hosted_run, monkeypatch, failure_type):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    tracer = run.service.tracer
    events = []
    failure = failure_type("trace\0\ud800" + "é" * 1024)
    if failure_type is not TraceTimeout:
        failure.code = "bad\0code"

    def fail(**kwargs):
        events.append("trace")
        raise failure

    run.actions.append(run.controls.action("after_trace", lambda: events.append("action")))
    run.retrace.set(True)
    run.prompt.set("AC")
    queue_action(route, "after_trace")
    run.controls.state = _controls.RUN_PAUSING
    with monkeypatch.context() as patch:
        patch.setattr(tracer, "trace", fail)
        patch.setattr(run.controls, "_wait", lambda: events.append("pause"))
        run.boundary()
    run.controls.state = _controls.RUN_RUNNING
    assert events == ["trace", "action", "pause"]
    assert route.transport.replies[-1].status == "ok"
    status = call(route, "trace.status")["result"]
    assert status["state"] == "error"
    assert status["error"]["code"] == ("interpret.trace_timeout" if failure_type is TraceTimeout else "interpret.refused")
    assert "\0" not in status["error"]["message"] and len(status["error"]["message"].encode()) <= 1024
    assert protocol.bounded(status) and run.service.graph is graph
    assert run.service.failure is None and not run.controls.queue
    assert_released(run)
    # An unchanged failed input does not trigger another automatic attempt.
    run.boundary()
    assert run.service.status["state"] == "error"
    assert call(route, "trace.run", {"prompt": "AC", "max_targets": 0})["status"] == "queued"
    run.boundary()
    assert route.transport.replies[-1].status == "ok"


@pytest.mark.parametrize("source", ["boundary", "apply_clamp", "reset"])
def test_late_trace_retains_queue_status_and_prompt_through_output(hosted_run, monkeypatch, source):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    if source == "reset":
        run.clamp.set(2.)
        run.boundary()
        _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.retrace.set(True)
    run.boundary()
    compare, trace = run.output.compare, run.service.tracer.trace
    arrivals, traces = [], []

    def compared(prompt):
        result = compare(prompt)
        if not arrivals:
            arrivals.append(call(route, "trace.run", {"prompt": "AD", "max_targets": 0}, 20))
        return result

    def traced(**params):
        traces.append(params)
        return trace(**params)

    monkeypatch.setattr(run.output, "compare", compared)
    monkeypatch.setattr(run.service.tracer, "trace", traced)
    if source == "boundary":
        run.prompt.set("AC")
    else:
        queue_action(route, source)
    run.boundary()
    assert arrivals[0]["status"] == "queued" and not traces
    assert run.service.status == {"state": "pending", "seconds": 0., "step": None}
    assert len(run.service.queue) == 1 and run.service.queue[0][1]["prompt"] == "AD"
    assert call(route, "trace.run", {"prompt": "AE"}, 21)["error"]["code"] == "interpret.busy"
    if source != "boundary":
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(10, "ok")]
    run.boundary()
    assert traces == [{"prompt": "AD", "max_targets": 0}]
    assert run.prompt.value == run.traced_prompt == "AD"
    assert [(reply.request_id, reply.status) for reply in route.transport.replies if reply.request_id == 20] == [(20, "ok")]
    assert run.service.status["state"] == "done" and not run.service.queue
    assert_released(run)


@pytest.mark.parametrize("kind", ["ablate", "add"])
def test_gemma_raw_mlp_retrace_refuses_and_next_request_recovers(gemma_case, kind):
    import torch
    from caliper.interp import capture
    from .test_service_trace_admission import run as trace_request
    case = gemma_case
    graph = trace_request(case.service)
    spec = {"point": "model.layers.0.mlp", "kind": kind}
    with torch.no_grad(), capture(case.model, [spec["point"]]):
        case.model(**case.tokenizer("AB"), use_cache=False)
    if kind == "add":
        spec.update(vector=[.01] * 16, coefficient=1.)
    case.owner.interventions.set(spec)
    case.owner.controls.step()
    assert case.owner.interventions.specifications()[0]["point"] == spec["point"]
    response = case.service._retrace({"prompt": "AB", "max_targets": 0}, 3)
    assert response[0] == "error" and response[2]["code"] == "interpret.refused"
    message = "ablation point" if kind == "ablate" else "no exact affine intervention path"
    assert message in response[2]["message"]
    assert case.service.status["state"] == "error" and protocol.bounded(case.service.status)
    assert case.service.graph is graph and case.service.failure is None
    assert not case.service.running and not case.service._borrowed
    assert case.service._request_stack is None
    case.owner.interventions.clear()
    case.owner.controls.step()
    assert trace_request(case.service) is graph


def test_real_deadline_failure_is_bounded_and_next_trace_recovers(hosted_run, monkeypatch):
    from caliper.interp import _trace
    run, route = hosted_run
    _trace_at_boundary(run, prompt="AB", max_targets=0)
    clock = SimpleNamespace(now=100.)
    deadline = _trace._Deadline.__call__
    stages = []

    def expire(owner, stage):
        if stage == "dictionary admission":
            clock.now += 20
            stages.append(stage)
        deadline(owner, stage)

    run.retrace.set(True)
    run.prompt.set("AC")
    with monkeypatch.context() as patch:
        patch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
        patch.setattr(_trace._Deadline, "__call__", expire)
        run.boundary()
    assert stages == ["dictionary admission"]
    assert run.service.status["state"] == "error"
    assert run.service.status["error"]["code"] == "interpret.trace_timeout"
    assert run.service.failure is None
    assert_released(run)
    assert call(route, "trace.run", {"prompt": "AC", "max_targets": 0})["status"] == "queued"
    run.boundary()
    assert route.transport.replies[-1].status == "ok"


def prepare_action_retrace(run, action):
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    if action == "reset":
        run.clamp.set(2.)
        run.boundary()
        _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.retrace.set(True)
    run.boundary()


@pytest.mark.parametrize("action", ["apply_clamp", "reset"])
def test_clamp_action_completes_when_its_automatic_trace_fails(hosted_run, monkeypatch, action):
    run, route = hosted_run
    prepare_action_retrace(run, action)
    events = []

    def fail(**params):
        events.append("trace")
        raise RuntimeError("automatic trace failed")

    run.actions.append(run.controls.action("later", lambda: events.append("later")))
    monkeypatch.setattr(run.service.tracer, "trace", fail)
    queue_action(route, action, 10)
    queue_action(route, "later", 11)
    run.boundary()
    assert events == ["trace", "later"]
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(10, "ok"), (11, "ok")]
    assert bool(run.interventions.specifications()) == (action == "apply_clamp")
    assert run.service.status["state"] == "error" and run.status.value == "automatic trace failed"
    assert run.service.failure is None and not run.controls.queue
    assert_released(run)


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_automatic_interrupt_reaches_hosted_owner_and_preserves_queued_work(hosted_run, monkeypatch, failure_type):
    run, route = hosted_run
    _trace_at_boundary(run, prompt="AB", max_targets=0)
    failure = failure_type("automatic trace interrupted")
    replies, completed = [], []

    def interrupt(**params):
        run.service.submit({"version": 1, "method": "trace.read", "params": {}}, lambda *args: replies.append(args))
        raise failure

    run.actions.append(run.controls.action("later", lambda: completed.append(1)))
    queue_action(route, "later")
    run.prompt.set("AC")
    run.retrace.set(True)
    with monkeypatch.context() as patch:
        patch.setattr(run.service.tracer, "trace", interrupt)
        with pytest.raises(failure_type) as caught:
            run.boundary()
    assert caught.value is failure and not replies and not completed
    assert run.service.failure is run._failure is None
    assert run.service.status["state"] == "error"
    assert len(run.service.queue) == 1 and run.controls.actions["later"].queued
    assert_released(run)
    assert call(route, "trace.run", {"prompt": "AC", "max_targets": 0})["status"] == "queued"
    run.boundary()
    assert replies[0][0] == "ok" and completed == [1]
    assert route.transport.replies[-1].status == "ok"


@pytest.mark.parametrize("source", ["boundary", "apply_clamp", "reset"])
def test_automatic_resource_stop_terminates_actions_and_checkpoints_once(hosted_run, monkeypatch, source):
    from caliper.interp._memory import CleanStop
    run, route = hosted_run
    prepare_action_retrace(run, source)
    failure = CleanStop("automatic trace resource stop")
    replies, checkpoints = [], []

    def stop(**params):
        run.service.submit({"version": 1, "method": "trace.read", "params": {}}, lambda *args: replies.append(args))
        raise failure

    monkeypatch.setattr(run.service.tracer, "trace", stop)
    monkeypatch.setattr(run.managed, "checkpoint", lambda *args: checkpoints.append(1))
    run.actions.append(run.controls.action("later", lambda: pytest.fail("action followed a resource stop")))
    if source == "boundary":
        run.prompt.set("AC")
    else:
        queue_action(route, source, 10)
    queue_action(route, "later", 11)
    with pytest.raises(CleanStop) as caught:
        run.boundary()
    assert caught.value is run.service.failure is run._failure is failure
    assert replies[0][2]["code"] == "interpret.stopped" and len(replies) == 1
    expected = [11] if source == "boundary" else [10, 11]
    assert sorted((reply.request_id, reply.status) for reply in route.transport.replies) == [(rid, "error") for rid in expected]
    assert not run.service.queue and not run.controls.queue and checkpoints == [1]
    assert_released(run)
    with pytest.raises(CleanStop) as repeated:
        run.boundary()
    assert repeated.value is failure and checkpoints == [1]
