"""CPU checks for idle control delivery and boundary recovery."""
from types import SimpleNamespace

import pytest

from caliper import _controls, _programs as P, _wire
from .hosted_publication_fixture import publication_run


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
            return False
        left = count
        return original(predicate, **kwargs)

    monkeypatch.setattr(controls.cond, "wait_for", wait_for)
    return timeouts


@pytest.mark.parametrize("arrival", ["checkpoint", "callback"])
def test_action_queued_after_empty_timeout_reaches_next_publication(publication_run, monkeypatch, arrival):
    case = publication_run
    controls = case.run.controls
    original = controls.checkpoint
    handled, replies = [], []
    route = _controls.Route(controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))

    def callback():
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


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_idle_checkpoint_exception_restores_later_boundary(publication_run, monkeypatch, failure):
    case = publication_run
    controls = case.run.controls
    original = controls.checkpoint
    messages = []
    case.run.status = SimpleNamespace(set=messages.append)
    injected = False

    def checkpoint():
        nonlocal injected
        if not injected and not controls.queue and not P.inference_pending():
            injected = True
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
    before = controls.n
    case.run.boundary()
    assert controls.n == before + 1


def test_idle_clean_stop_retains_failure_and_single_checkpoint(publication_run, monkeypatch):
    from caliper.interp._memory import CleanStop

    case = publication_run
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

    case.run.managed = SimpleNamespace(checkpoint=checkpoint)
    monkeypatch.setattr(case.run.controls.cond, "wait_for", lambda *args, **kwargs: False)
    with pytest.raises(CleanStop) as caught:
        case.run.run()
    assert caught.value is failure and case.run._failure is failure
    assert stopped == [(failure, 0)] and saved == [0]
    with pytest.raises(CleanStop) as caught_again:
        case.run.boundary()
    assert caught_again.value is failure and saved == [0]
