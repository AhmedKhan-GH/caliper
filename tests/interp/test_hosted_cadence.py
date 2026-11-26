"""CPU publication cadence across discrete hosted actions."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _adapter, _controls, _programs as P, _wire
from caliper.interp import _hosted, _publish, publish


@pytest.fixture
def action_run(monkeypatch):
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
    run = _hosted.HostedInference.__new__(_hosted.HostedInference)
    run.controls, run.service = controls, SimpleNamespace(pending=False)
    run.guard = run.managed = run._failure = None
    run.status = SimpleNamespace(set=lambda text: pytest.fail(text))
    clock = SimpleNamespace(now=0., idle_s=0., action_s=0., pass_s=.00002, export_s=0.,
                            scheduler_s=.0005, publish_s=.0001, waves=8, action_finished=0.)
    shown, replies, statuses, uniforms, delays = [], [], [], [], []

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(P, "_clock", lambda: clock.now)
    monkeypatch.setattr(P, "_controls_ports", lambda: [])
    monkeypatch.setattr(P, "_refresh", lambda p: None)
    monkeypatch.setattr(P, "_torch_adapter", lambda: SimpleNamespace(framework_done=lambda: None))
    marks = P._marks
    boundaries = []

    def scheduler():
        if len(boundaries) >= 20000:
            raise AssertionError("finite publication boundary limit")
        boundaries.append(controls.n)
        advance(clock.scheduler_s)                  # controlled routine work, not a measurement
        return marks()

    monkeypatch.setattr(P, "_marks", scheduler)

    def outputs(*args):
        advance(clock.export_s)
        return float(_publish._resolve("output.probs").item())

    def publish_pass(batch):
        advance(clock.publish_s)
        shown.extend(value for _, value in batch)
        if batch:
            delays.append(clock.now - clock.action_finished)

    monkeypatch.setattr(P, "_outputs", outputs)
    monkeypatch.setattr(P, "_publish_pass", publish_pass)
    program = P._Program("bars", {
        "inputs": [{"var": "probs", "source": "live:output.probs"}],
        "output": {"kind": "tensor", "shape": [1]}, "limits": {"every": 1},
    }, "", None, "")
    program.state, program.handle = "compiled", 1
    P._PROGS[program.id] = program
    monkeypatch.setattr(P, "ACTIVE", True)

    def execute(handles, step, time_s, every):
        uniforms.append((step, time_s, every))
        advance(clock.pass_s)
        return {"ran": True}

    runtime = SimpleNamespace(run=execute, remove=lambda handle: None,
                              status=lambda handle: {"state": "running", "last_pass_ms": .02})
    P._STATE["rt"] = runtime

    def action():
        advance(clock.action_s)
        publish("output.probs", torch.tensor([float(len(statuses))]))
        clock.action_finished = clock.now

    controls.action("compare", action)
    route = _controls.Route(controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))

    def enqueue():
        reply = route.on_control(None, _wire.Control(1, "run", name="compare").to_meta())
        assert reply["status"] == "queued"

    class Finished(Exception):
        pass

    def wait_for(predicate, **kwargs):
        # Condition.wait_for returns at once for an action that already waits.
        if predicate():
            return True
        statuses.append(program.item())
        if len(statuses) == clock.waves:
            raise Finished
        controls.cond.release()
        try:
            assert not P.inference_pending()
            advance(clock.idle_s)
            enqueue()
        finally:
            controls.cond.acquire()
        return True

    monkeypatch.setattr(controls.cond, "wait_for", wait_for)

    def drive():
        enqueue()
        with pytest.raises(Finished):
            run.run()

    return SimpleNamespace(run=run, clock=clock, program=program, runtime=runtime, drive=drive,
                           shown=shown, replies=replies, statuses=statuses, uniforms=uniforms, delays=delays)


@pytest.mark.parametrize("every", [1, 7])
@pytest.mark.parametrize("marks", [False, True])
@pytest.mark.parametrize("idle_s,action_s", [(0., 0.), (.55, 0.), (0., .55), (.55, .55)])
def test_idle_and_action_work_do_not_become_pass_cost(action_run, every, marks, idle_s, action_s):
    case = action_run
    case.program.every_min = every
    case.clock.idle_s, case.clock.action_s = idle_s, action_s
    if marks:
        # CPU events exercise the queue adapter and its dense-to-sparse transition.
        P._STATE["marks"] = _adapter.StepMarks(
            lambda: SimpleNamespace(record=lambda: None, query=lambda: True))
    case.drive()
    assert case.shown == list(range(8))
    assert all(reply["status"] == "ok" for reply in case.replies)
    steps = [item["last_step"] for item in case.statuses]
    assert steps[0] >= P.WARMUP_STEPS
    assert all(b - a >= max(every, P.N_MIN) for a, b in zip(steps, steps[1:]))
    assert P._CAD.cost_s() == pytest.approx(.00002)
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.n == P.N_MIN
    assert len(P._CAD.cost) == 7
    # The frame uniform and pass rate include all elapsed action work.
    assert case.uniforms[-1][1] >= 8 * action_s + 7 * idle_s
    assert case.program.every_min == every and case.program.time_ms == P.TIME_MS_DEFAULT
    assert case.program.memory_mb == P.MEMORY_MB_DEFAULT


def test_hosted_pass_and_export_cost_still_slow_publication(action_run):
    case = action_run
    case.clock.idle_s = case.clock.action_s = .55
    case.clock.pass_s, case.clock.export_s = .0004, .0006
    case.drive()
    assert case.shown == list(range(8))
    assert P._CAD.cost_s() == pytest.approx(.001)
    assert P._CAD.n >= 42
    assert P._CAD.share() <= P.SHARE
    assert len(P._CAD.cost) == 7


def test_hosted_clock_counts_framework_waits(action_run, monkeypatch):
    case = action_run
    case.clock.idle_s = case.clock.action_s = .55

    class Event:
        def record(self):
            self.at = case.clock.now + .0003

        def query(self):
            return case.clock.now >= self.at

    def sleep(seconds):
        case.clock.now += max(seconds, .0003)

    P._STATE["marks"] = _adapter.StepMarks(Event)
    monkeypatch.setattr(_adapter, "time", SimpleNamespace(sleep=sleep))
    case.drive()
    assert case.shown == list(range(8))
    assert P._CAD.cost_s() == pytest.approx(.00032)
    assert P._CAD.n > P.N_MIN
    assert len(P._CAD.cost) == 7


@pytest.mark.parametrize("sender_s", [0., .00001])
def test_hosted_clock_counts_snapshots_before_checkpoint(action_run, monkeypatch, sender_s):
    case = action_run
    case.clock.idle_s = case.clock.action_s = .55
    ring = SimpleNamespace(gate_step=None, gate_t=0., gate_open=False, gate_cost=0., publish_s=0., pass_t=0.,
                           step_tick=lambda *args: None, wants_pass=lambda now: False,
                           pass_rate=lambda: None, asked_interval=lambda: None)

    def open_pass(step):
        ring.gate_open, ring.gate_cost = True, 0.
        ring.gate_step, ring.gate_t = step, case.clock.now

    def close_pass(step, now):
        ring.gate_open, ring.pass_t = False, now
        return int(ring.gate_cost > 0.)

    ring.open_pass, ring.close_pass = open_pass, close_pass
    P._VIEWS["ring"] = ring
    step = caliper.step

    def snapshot_step():
        if ring.gate_open:
            case.clock.now += .00003
            ring.gate_cost += .00003
            # The sender can publish the previous snapshot while the host is idle.
            ring.publish_s += sender_s
        step()

    monkeypatch.setattr(caliper, "step", snapshot_step)
    case.drive()
    assert case.shown == list(range(8))
    assert P._CAD.cost_s() == pytest.approx(.00005 + sender_s)
    assert P._CAD.target == P.VIEWS_SHARE
    assert len(P._CAD.cost) == 7


@pytest.mark.parametrize("failed_boundary", [False, True])
def test_normal_steps_keep_wall_clock_feedback(action_run, monkeypatch, failed_boundary):
    case = action_run
    case.clock.idle_s = .55
    if failed_boundary:
        def fail():
            raise RuntimeError("boundary failure")
        with monkeypatch.context() as patch:
            patch.setattr(caliper, "step", fail)
            with pytest.raises(RuntimeError, match="boundary failure"):
                case.run.boundary()
    # The same gaps remain visible to the ordinary training entry point.
    from contextlib import nullcontext
    monkeypatch.setattr(P, "inference_boundary", lambda **kwargs: nullcontext())
    monkeypatch.setattr(case.run, "boundary", caliper.step)
    case.drive()
    assert P._CAD.cost_s() == pytest.approx(.55002)
    assert P._CAD.n > P.N_MIN
