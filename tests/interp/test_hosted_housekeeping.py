"""Nonzero idle scans remain separate from useful work and sender CPU."""
import threading
from types import SimpleNamespace

import pytest

from caliper import _programs as P, _net_ring as N, _net_programs as NP
from caliper import _net_controls as NC, _wire
from .test_hosted_cadence import action_run
from .test_hosted_sender_overlap import sender_cpu


def idle_injection(case, monkeypatch, *, scan_s, count, idle_wall):
    original = case.run.controls.cond.wait_for
    pending = P.inference_pending
    state = SimpleNamespace(left=count, ticks=0, scans=0, idle_scan_s=0.)

    def scanned():
        result = pending()
        case.clock.now += scan_s
        state.scans += 1
        if not result:
            state.idle_scan_s += scan_s
        return result

    def wait_for(predicate, **kwargs):
        if not predicate() and state.left:
            state.left -= 1
            state.ticks += 1
            case.clock.now += idle_wall
            return False
        state.left = count
        return original(predicate, **kwargs)

    monkeypatch.setattr(P, 'inference_pending', scanned)
    monkeypatch.setattr(case.run.controls.cond, 'wait_for', wait_for)
    return state


@pytest.mark.parametrize("idle_wall", [1., 100.])
@pytest.mark.parametrize("count,scan_s,waves", [(3, .0004, 4), (30, .01, 18)])
def test_nonzero_idle_scans_do_not_accumulate_into_pass_cost(
        action_run, monkeypatch, idle_wall, count, scan_s, waves):
    case = action_run
    case.clock.scheduler_s, case.clock.publish_s = .0099, .0001
    case.clock.waves = waves
    idle = idle_injection(case, monkeypatch, scan_s=scan_s, count=count, idle_wall=idle_wall)
    case.drive()
    assert idle.ticks == count * waves
    assert case.shown == list(range(waves))
    assert case.program.state == "running" and case.program.error is None
    # An admitted wake includes its second readiness scan. Empty polls add no cost.
    assert P._CAD.step_s() == pytest.approx(.01 + scan_s)
    assert P._CAD.cost_s() == pytest.approx(case.clock.pass_s + scan_s)
    assert not P._INFERENCE_ACTIVE


@pytest.mark.parametrize('frame_cpu', [0., .00001])
def test_periodic_status_cpu(action_run, monkeypatch, sender_cpu, frame_cpu):
    case = action_run
    case.clock.waves = 4
    controls = case.run.controls
    ring = N.NetRing(engine=object())
    P.views(ring)
    # Build only the synchronous port. No server, socket, or timer thread starts.
    port = NC.ControlsPort.__new__(NC.ControlsPort)
    link = SimpleNamespace(attached=True, closed=False,
                           caps={_wire.CAP_PROGRAMS, _wire.CAP_CONTROLS})
    port.server = SimpleNamespace(ring=ring, producer=SimpleNamespace(links=[link]))
    port._lock = threading.Lock()
    port._controls = port._held = None
    port._scheduled = port._after_data = False
    port._capacity_waits = {}
    port._capacity_closed = port._capacity_notify = False
    port._selections, port._replies, port._posts = {}, [], []
    port._unsent, port._unsent_all = {}, {}
    port._kick = lambda: None
    sent = []

    def send(target, frame):
        assert target is link
        sent.append(len(frame))
        case.clock.now += frame_cpu
        sender_cpu.value += frame_cpu

    port._send = send
    programs = NP.ProgramsPort(port)
    monkeypatch.setattr(programs, '_watched', lambda: True)
    monkeypatch.setattr(programs, 'items', lambda: [case.program.item()])
    monkeypatch.setattr(NP, 'time', SimpleNamespace(monotonic=lambda: case.clock.now))
    monkeypatch.setattr(controls, '_arm_report', lambda: None)
    controls.routes.append(SimpleNamespace(_broadcast=port.broadcast))
    idle = idle_injection(case, monkeypatch, scan_s=.0004, count=3, idle_wall=0.)
    wait_for = controls.cond.wait_for

    def periodic_wait(predicate, **kwargs):
        before = idle.ticks
        result = wait_for(predicate, **kwargs)
        if idle.ticks != before:
            # Match Condition.wait_for: a timer runs while the condition is unlocked.
            controls.cond.release()
            try:
                for _ in range(_wire.PROGRAM_STATE_RATE_HZ):
                    case.clock.now += 1. / _wire.PROGRAM_STATE_RATE_HZ
                    programs._tick()
                    port.flush()
                controls._report()
            finally:
                controls.cond.acquire()
        return result

    monkeypatch.setattr(controls.cond, 'wait_for', periodic_wait)
    checkpoint = controls.checkpoint

    def service_checkpoint():
        checkpoint()
        port.flush()

    monkeypatch.setattr(controls, 'checkpoint', service_checkpoint)
    case.drive()
    assert programs.counters['states_out'] == idle.ticks * _wire.PROGRAM_STATE_RATE_HZ == 24
    assert controls.counters['controls_out'] >= 12
    assert len(sent) == 40
    assert ring.publish_s == pytest.approx(len(sent) * frame_cpu)
    assert P._CAD.inference_account.sample() == pytest.approx(ring.publish_s)
    assert case.program.state == 'running'
    # Completed cycles and pending credit together own all sender CPU.
    charged = sum(value - .00042 for value in P._CAD.cost) + P._CAD.extra
    assert charged == pytest.approx(ring.publish_s, abs=1e-12)
    assert P._CAD.step_s() == pytest.approx(.001)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_poll_retains_mandatory_cost_and_restores_nested_scopes(
        action_run, sender_cpu, accepted, failure):
    case = action_run
    ring = N.NetRing(engine=object())
    P.views(ring)
    ring.open_pass(1)

    def snapshot(seconds):
        case.clock.now += seconds
        ring.gate_cost += seconds
        ring.gate_taken += 1

    with P.inference_boundary():
        case.clock.now += .0002
        try:
            with P.inference_poll() as poll:
                case.clock.now += .0004
                snapshot(.003)
                with ring.sender_cost():
                    sender_cpu.value += .00001
                    case.clock.now += .00001
                with P.inference_boundary():
                    case.clock.now += .0002
                with P.inference_excluded():
                    case.clock.now += .7
                    snapshot(.002)
                P.at_step(1)
                if accepted:
                    poll.accept()
                if failure:
                    raise ValueError("poll failure")
        except ValueError:
            pass
        case.clock.now += .0003
    assert P._CAD.inference_s == pytest.approx(.0061 + (.0006 if accepted else 0.))
    assert P._CAD.extra == pytest.approx(.00001)
    assert not P._INFERENCE_ACTIVE and P._INFERENCE.clock is None


def test_idle_clock_operations_have_nonzero_cost_without_a_step(action_run, monkeypatch):
    case = action_run
    P._PROGS.clear()
    P._set_active()
    monkeypatch.setattr(case.run, "boundary", lambda: pytest.fail("unexpected active boundary"))
    waits = []

    def clock():
        case.clock.now += .00001
        return case.clock.now

    class Finished(BaseException):
        pass

    def wait_for(predicate, **kwargs):
        assert not predicate()
        waits.append(case.run.controls.n)
        if len(waits) == 4:
            raise Finished
        case.clock.now += 1.
        return False

    monkeypatch.setattr(P, "_clock", clock)
    monkeypatch.setattr(case.run.controls.cond, "wait_for", wait_for)
    with pytest.raises(Finished):
        case.run.run()
    assert waits == [0] * 4
    assert case.clock.now > 3.
    assert P._CAD.inference_s == 0.
    assert not P._INFERENCE_ACTIVE and P._INFERENCE.clock is None


def test_idle_clean_stop_retains_failure_and_single_checkpoint(action_run, monkeypatch):
    from caliper.interp._memory import CleanStop

    case = action_run
    P._PROGS.clear()
    P._set_active()
    failure = CleanStop("idle resource stop")
    stopped, saved = [], []
    case.run.guard = SimpleNamespace(failure=failure)
    case.run.service.stop = lambda exc, step: stopped.append((exc, step))
    case.run.model = object()
    case.run._stop_checkpointed = False

    def checkpoint(*args):
        saved.append(case.run.controls.n)
        case.clock.now += .004

    case.run.managed = SimpleNamespace(checkpoint=checkpoint)
    original = case.run._check_stop

    def check_stop():
        case.clock.now += .0003
        original()

    monkeypatch.setattr(case.run, "_check_stop", check_stop)
    monkeypatch.setattr(case.run.controls.cond, "wait_for", lambda *args, **kwargs: False)
    with pytest.raises(CleanStop) as caught:
        case.run.run()
    assert caught.value is failure and case.run._failure is failure
    assert stopped == [(failure, 0)] and saved == [0]
    assert P._CAD.inference_s == 0.
    with pytest.raises(CleanStop) as caught_again:
        case.run.boundary()
    assert caught_again.value is failure and saved == [0]
    assert not P._INFERENCE_ACTIVE and P._INFERENCE.clock is None
