"""CPU regressions for automatic trace failures and queued request ownership."""
from types import SimpleNamespace

import pytest

from caliper import _controls, _interpret_protocol as protocol, _wire
from caliper.interp._trace import TraceTimeout
from .test_hosted import hosted_run, _trace_at_boundary
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
