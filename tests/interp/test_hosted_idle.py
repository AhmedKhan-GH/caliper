"""CPU checks for idle hosted checkpoints and later active work."""
from types import SimpleNamespace

import pytest

import caliper
from caliper import _controls, _programs as P, _wire
from caliper.interp import _hosted
from caliper.interp._service import InterpretationService
from .test_hosted_cadence import action_run


def idle_timeouts(case, monkeypatch, *, count=3):
    """Return empty timeouts before the fixture queues each later action."""
    controls = case.run.controls
    original = controls.cond.wait_for
    timeouts = []
    left = count

    def wait_for(predicate, **kwargs):
        nonlocal left
        if not predicate() and left:
            assert not P.inference_pending()
            left -= 1
            timeouts.append(controls.n)
            case.clock.now += kwargs["timeout"]
            return False
        left = count
        return original(predicate, **kwargs)

    monkeypatch.setattr(controls.cond, "wait_for", wait_for)
    return timeouts


@pytest.mark.parametrize("routine", ["checkpoint", "service"])
def test_idle_checkpoints_do_not_enter_later_pass_cost(action_run, monkeypatch, routine):
    case = action_run
    case.clock.waves = 4
    controls = case.run.controls
    checkpoint = controls.checkpoint
    calls = []

    def routine_checkpoint():
        calls.append(controls.n)
        if routine == "checkpoint":
            case.clock.now += .004
        checkpoint()

    if routine == "service":
        service = _hosted.HostedService.__new__(_hosted.HostedService)
        InterpretationService.__init__(service)
        service.owner = case.run
        service.boundary_running, service.boundary_thread = False, None
        controls.interpretation_service = case.run.service = service

        def routine_work():
            case.clock.now += .002

        monkeypatch.setattr(case.run, "update_controls", routine_work)
        monkeypatch.setattr(case.run, "update_output", routine_work)
    monkeypatch.setattr(controls, "checkpoint", routine_checkpoint)
    timeouts = idle_timeouts(case, monkeypatch)
    case.drive()
    assert len(timeouts) == 12
    assert len(calls) == controls.n + len(timeouts)
    assert case.shown == list(range(4))
    assert len(P._CAD.cost) == 3
    assert P._CAD.step_s() == pytest.approx(.0046)
    assert P._CAD.cost_s() == pytest.approx(.00002)
    assert P._CAD.n == P.N_MIN
    assert P._CAD.inference_s == pytest.approx(case.clock.now - 12. - 12 * .004)


@pytest.mark.parametrize("arrival", ["checkpoint", "callback"])
def test_action_queued_after_empty_timeout_reaches_next_publication(action_run, monkeypatch, arrival):
    case = action_run
    controls = case.run.controls
    original = controls.checkpoint
    handled, replies = [], []
    route = _controls.Route(controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))

    def callback():
        case.clock.now += .7
        handled.append(controls.n)

    controls.action("late", callback)

    def enqueue():
        reply = route.on_control(None, _wire.Control(2, "run", name="late").to_meta())
        assert reply["status"] == "queued"

    compare = controls.actions["compare"]
    compare_callback = compare.fn

    def enqueue_from_callback():
        compare.fn = compare_callback
        compare_callback()
        enqueue()

    injected = False

    def checkpoint():
        nonlocal injected
        idle = not controls.queue and not P.inference_pending()
        if idle and not injected:
            injected = True
            if arrival == "checkpoint":
                enqueue()
            else:
                # An action can queue another action after admission captures the queue.
                compare.fn = enqueue_from_callback
                reply = route.on_control(None, _wire.Control(3, "run", name="compare").to_meta())
                assert reply["status"] == "queued"
        original()

    monkeypatch.setattr(controls, "checkpoint", checkpoint)
    timeouts = idle_timeouts(case, monkeypatch, count=1)
    case.drive()
    assert injected and timeouts
    assert handled == [timeouts[0] + (arrival == "callback")]
    assert all(reply["status"] == "ok" for reply in replies)
    assert not controls.queue
    assert case.shown[-1] == 7 and len(case.shown) >= 8
    assert not P.inference_pending()
    assert getattr(P._INFERENCE, "clock", None) is None


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_idle_checkpoint_exception_restores_later_boundary(action_run, monkeypatch, failure):
    case = action_run
    controls = case.run.controls
    original = controls.checkpoint
    messages = []
    case.run.status = SimpleNamespace(set=messages.append)
    injected = False

    def checkpoint():
        nonlocal injected
        if not injected and not controls.queue and not P.inference_pending():
            injected = True
            case.clock.now += .004
            raise failure("idle checkpoint failure")
        original()

    monkeypatch.setattr(controls, "checkpoint", checkpoint)
    timeouts = idle_timeouts(case, monkeypatch, count=1)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            case.drive()
    else:
        case.drive()
        assert messages == ["RuntimeError: idle checkpoint failure"]
        assert case.shown == list(range(8))
    assert injected and timeouts
    assert getattr(P._INFERENCE, "clock", None) is None
    before = P._CAD.inference_s
    case.run.boundary()
    assert P._CAD.inference_s - before == pytest.approx(.0006)


def test_idle_checkpoint_retains_snapshot_and_reentrant_step_cost(action_run, monkeypatch):
    from caliper._net_ring import NetRing

    case = action_run
    controls = case.run.controls
    original = controls.checkpoint
    ring = NetRing(engine=object())
    P.views(ring)
    injected = False
    charged = []

    def checkpoint():
        nonlocal injected
        if not injected and not controls.queue and not P.inference_pending():
            injected = True
            before = P._CAD.inference_s
            case.clock.now += .004
            ring.open_pass(controls.n + 1)
            case.clock.now += .003
            ring.gate_cost += .003
            ring.gate_taken += 1
            # A callback can enter a real step after the idle decision.
            caliper.step()
            charged.append(P._CAD.inference_s - before)
        original()

    monkeypatch.setattr(controls, "checkpoint", checkpoint)
    idle_timeouts(case, monkeypatch, count=1)
    case.drive()
    assert injected and charged == [pytest.approx(.0036)]
    assert case.shown == list(range(8))
