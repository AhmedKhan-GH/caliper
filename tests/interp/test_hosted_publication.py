"""CPU checks for scheduled publication while hosted inference is idle."""
from types import SimpleNamespace

import pytest
import torch

from caliper import _programs as P, _wire
from caliper.interp import publish
from caliper.interp import _publish
from .test_hosted import hosted_run, _trace_at_boundary
from .test_publication import registry


@pytest.fixture
def scheduled_program(registry, monkeypatch):
    program = P._Program("bars", {"inputs": [{"var": "probs", "source": "live:output.probs"}],
                                 "output": {"kind": "tensor", "shape": [2, 10]},
                                 "limits": {"every": 3}}, "", None, "")
    program.state, program.handle, program.warm = "compiled", 1, 2
    P._PROGS[program.id] = program
    P._CAD.n = 4
    P._STATE.update(step=None, **{"pass": None, "pass_budget": True, "dense": True})
    runs, publications = [], []
    runtime = SimpleNamespace(
        run=lambda handles, step, *args: runs.append(step) or {"ran": True},
        status=lambda handle: {"state": "running", "last_pass_ms": .1},
        remove=lambda handle: None, set_params=lambda *args: None)
    P._STATE["rt"] = runtime
    monkeypatch.setattr(P, "_marks", lambda: None)
    monkeypatch.setattr(P, "_torch_adapter", lambda: SimpleNamespace(framework_done=lambda: None))
    monkeypatch.setattr(P, "_refresh", lambda p: None)
    monkeypatch.setattr(P, "_outputs", lambda *args: {})
    monkeypatch.setattr(P, "_publish_pass", lambda batch: publications.extend(p.last_step for p, _ in batch))
    monkeypatch.setattr(P, "_OUTBOX", {})
    P._set_active()
    return program, runs, publications


def test_pending_publication_keeps_warmup_cadence_and_declared_period(scheduled_program):
    program, runs, publications = scheduled_program
    tensor = torch.ones(2, 10)
    publish("output.probs", tensor)
    assert P.inference_pending()
    P.at_step(1)
    P.at_step(2)
    assert not runs and P.inference_pending()
    P.at_step(3)
    assert runs == publications == [3] and not P.inference_pending()
    # The same storage can hold new values, thus pointer identity is insufficient.
    tensor.mul_(2)
    publish("output.probs", tensor)
    assert P.inference_pending()
    P.at_step(4)
    P.at_step(5)
    P.at_step(6)
    assert runs == [3] and P.inference_pending()
    P.at_step(7)
    assert runs == publications == [3, 7] and not P.inference_pending()
    assert program.every_min == 3


def test_program_parameter_edit_requires_a_pass(scheduled_program):
    program, runs, _ = scheduled_program
    publish("output.probs", torch.ones(2, 10))
    P.at_step(1)
    P.at_step(2)
    P.at_step(3)
    assert not P.inference_pending()
    P.set_params(program.id, {"scale": 2.})
    assert P.inference_pending()
    P.at_step(4)
    P.at_step(5)
    P.at_step(6)
    P.at_step(7)
    assert runs == [3, 7] and not P.inference_pending()
    P.set_params(program.id, {"scale": 2.})
    assert not P.inference_pending()


def test_missing_inputs_wait_and_runtime_refusals_remain_terminal(scheduled_program, monkeypatch):
    program, runs, _ = scheduled_program
    program.inputs = [("probs", "live:trace.probs")]
    assert not P.inference_pending() and not runs
    publish("trace.probs", torch.ones(2, 10))
    assert P.inference_pending()
    def refuse(p):
        raise P._Refused("program.import_refused", "fixture refusal")
    monkeypatch.setattr(P, "_refresh", refuse)
    P.at_step(1)
    P.at_step(2)
    P.at_step(3)
    assert program.state == "refused" and not runs and not P.inference_pending()


def test_publication_preserves_native_stop(scheduled_program):
    program, runs, publications = scheduled_program
    publish("output.probs", torch.ones(2, 10))
    P._STATE["rt"].status = lambda handle: {"state": "stopped", "stop_code": "program.over_time"}
    P.at_step(1)
    P.at_step(2)
    P.at_step(3)
    assert runs == [3] and not publications
    assert program.error["code"] == "program.over_time" and not P.inference_pending()


def test_publication_outbox_retains_pending_work(registry, monkeypatch):
    monkeypatch.setattr(P, "_OUTBOX", {1: object()})
    assert P.inference_pending()
    P._OUTBOX.clear()
    assert not P.inference_pending()


def test_hosted_slider_drains_publication_without_another_forward(hosted_run, monkeypatch):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    initial_step = run.controls.n
    initial = _publish._resolve("output.probs").clone()
    publication, forwards, waits = [], [], []
    hook = run.model.register_forward_pre_hook(lambda *args: forwards.append(run.controls.n))
    def pending():
        return not publication
    def at_step(step):
        # Simulate a cadence that needs two further boundaries after the edit.
        if step == initial_step + 3:
            publication.append((_publish._resolve("output.probs").clone(), step))
    class Finished(Exception):
        pass
    def wait_for(predicate, **kwargs):
        waits.append(run.controls.n)
        assert publication and not predicate()
        raise Finished
    monkeypatch.setattr(P, "inference_pending", pending)
    monkeypatch.setattr(P, "at_step", at_step)
    monkeypatch.setattr(P, "ACTIVE", True)
    monkeypatch.setattr(run.controls.cond, "wait_for", wait_for)
    route.on_control(None, _wire.Control(1, "set", name="clamp", value=0.).to_meta())
    try:
        with pytest.raises(Finished):
            run.run()
    finally:
        hook.remove()
    assert forwards == [initial_step + 1]
    assert waits == [initial_step + 3]
    assert len(publication) == 1 and not torch.equal(initial, publication[0][0])
    evidence = run.inference_status()
    assert evidence["comparison"]["completed_step"] == initial_step + 1
    assert evidence["comparison"]["interventions_hash"] == evidence["interventions_hash"]


def test_busy_viewer_cannot_consume_a_new_input_version(scheduled_program, monkeypatch):
    program, _, _ = scheduled_program
    publish("output.probs", torch.ones(2, 10))
    def skipped(runtime, p):
        p.skipped += 1
        return []
    monkeypatch.setattr(P, "_outputs", skipped)
    P.at_step(1)
    P.at_step(2)
    P.at_step(3)
    assert P.inference_pending() and program.input_version == -1
    monkeypatch.setattr(P, "_outputs", lambda *args: {})
    P.at_step(4)
    P.at_step(5)
    P.at_step(6)
    P.at_step(7)
    assert not P.inference_pending()


def test_new_or_delayed_viewer_retains_publication_work(scheduled_program, monkeypatch):
    program, _, _ = scheduled_program
    publish("output.probs", torch.ones(2, 10))
    P.at_step(1)
    P.at_step(2)
    P.at_step(3)
    assert not P.inference_pending()
    program.kind = "piece"
    link = SimpleNamespace()
    monkeypatch.setattr(P, "_controls_ports", lambda: [object()])
    monkeypatch.setattr(P, "_resource_links", lambda ctl: [link])
    assert P.inference_pending()
    program.relayed.add(link)
    assert not P.inference_pending()
    program.behind.add(link)
    assert P.inference_pending()
    monkeypatch.setattr(P, "_resource_links", lambda ctl: [])
    assert not P.inference_pending()
