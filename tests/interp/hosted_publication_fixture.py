"""Bounded CPU publication fixture with discrete control events."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _controls, _programs as P, _wire
from caliper.interp import _hosted, _publish, publish
from caliper.interp._service import InterpretationService, register_service, unregister_service


@pytest.fixture
def publication_run(monkeypatch):
    for name in ("_PROGS", "_DRAWN", "_IMPORTS", "_LIVE", "_ACTS", "_ACT_HOOKS", "_OUTBOX"):
        monkeypatch.setattr(P, name, {})
    monkeypatch.setattr(P, "_CAD", P._Cadence())
    monkeypatch.setattr(P, "_VIEWS", {"ring": None})
    monkeypatch.setattr(P, "_STATE", {**P._STATE, "step": None, "pass": None, "marks": False,
                                    "pass_budget": True, "dense": True, "model": None,
                                    "pub_pass": None, "view_s": deque(maxlen=P.VIEW_SAMPLES)})
    monkeypatch.setattr(P, "listeners", [])
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    monkeypatch.setattr(_publish, "_VERSION", 0)
    controls = _controls.Controls()
    controls.state = _controls.RUN_RUNNING
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    run = _hosted.HostedInference.__new__(_hosted.HostedInference)
    run.controls, run.service = controls, InterpretationService()
    run.service.owner = run
    register_service(run.service, controls=controls)
    run.guard = run.managed = run._failure = None
    run.status = SimpleNamespace(set=lambda text: pytest.fail(text))
    shown, replies, statuses = [], [], []
    monkeypatch.setattr(P, "_clock", lambda: float(controls.n))
    monkeypatch.setattr(P, "_controls_ports", lambda: [])
    monkeypatch.setattr(P, "_refresh", lambda p: None)
    monkeypatch.setattr(P, "_torch_adapter", lambda: SimpleNamespace(framework_done=lambda: None))
    monkeypatch.setattr(P, "_outputs", lambda *args: float(_publish._resolve("output.probs").item()))
    monkeypatch.setattr(P, "_publish_pass", lambda batch: shown.extend(value for _, value in batch))
    program = P._Program("bars", {
        "inputs": [{"var": "probs", "source": "live:output.probs"}],
        "output": {"kind": "tensor", "shape": [1]}, "limits": {"every": 1},
    }, "", None, "")
    program.state, program.handle = "compiled", 1
    P._PROGS[program.id] = program
    monkeypatch.setattr(P, "ACTIVE", True)

    def execute(handles, step, time_s, every):
        assert step <= 64, "finite publication boundary limit"
        return {"ran": True}

    runtime = SimpleNamespace(run=execute, remove=lambda handle: None,
                              status=lambda handle: {"state": "running"})
    P._STATE["rt"] = runtime
    controls.action("compare", lambda: publish("output.probs", torch.tensor([float(len(statuses))])))
    route = _controls.Route(controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))

    def enqueue():
        reply = route.on_control(None, _wire.Control(1, "run", name="compare").to_meta())
        assert reply["status"] == "queued"

    class Finished(BaseException):
        pass

    def wait_for(predicate, **kwargs):
        if predicate():
            return True
        statuses.append(program.item())
        if len(statuses) == 8:
            raise Finished
        controls.cond.release()
        try:
            assert not P.inference_pending()
            enqueue()
        finally:
            controls.cond.acquire()
        return True

    monkeypatch.setattr(controls.cond, "wait_for", wait_for)

    boundary = run.boundary

    def bounded_boundary():
        if controls.n >= 64:
            pytest.fail(f"publication did not complete: state={program.state}, warm={program.warm}, "
                        f"runs={program.runs}, next_pass={P._STATE['pass']}, error={program.error}")
        boundary()

    monkeypatch.setattr(run, "boundary", bounded_boundary)

    def drive():
        enqueue()
        with pytest.raises(Finished):
            run.run()

    try:
        yield SimpleNamespace(run=run, program=program, runtime=runtime, drive=drive,
                              shown=shown, replies=replies, statuses=statuses)
    finally:
        unregister_service(run.service, controls=controls)
