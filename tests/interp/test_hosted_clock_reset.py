"""Hosted clock ownership across inactive subscription and install transitions."""
from contextlib import nullcontext
import threading

import pytest

from caliper import _net_ring as N, _programs as P
from .test_hosted_cadence import action_run
from .test_hosted_sender_overlap import sender_cpu


def inactive():
    P._PROGS.clear()
    P._DRAWN.clear()
    P._OUTBOX.clear()
    P._set_active()
    assert not P.ACTIVE and P._VIEWS["ring"] is None


def on_subscription_thread(call):
    errors = []

    def invoke():
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(5.)
    assert not worker.is_alive()
    assert not errors


@pytest.mark.parametrize("excluded", [False, True])
def test_inactive_first_subscription_retains_sender_account(action_run, sender_cpu, excluded):
    case = action_run
    inactive()
    ring = N.NetRing(engine=object())
    with P.inference_boundary():
        case.clock.now += .0002
        on_subscription_thread(lambda: P.views(ring))
        account = ring._sender.account
        assert account is P._CAD.inference_account
        assert not P._CAD.inference
        with P.inference_excluded() if excluded else nullcontext():
            with ring.sender_cost():
                case.clock.now += .00001
                sender_cpu.value += .00001
        case.clock.now += .0003
    assert P._CAD.inference_s == pytest.approx(.0005)
    assert P._CAD.extra == pytest.approx(.00001)
    assert account is P._CAD.inference_account
    assert not P._INFERENCE_ACTIVE


@pytest.mark.parametrize("transition", ["views", "install", "piece"])
@pytest.mark.parametrize("excluded", [False, True])
@pytest.mark.parametrize("sender_s", [0., .00007])
def test_inactive_reconnect_preserves_clock_and_history(
        action_run, monkeypatch, sender_cpu, transition, excluded, sender_s):
    case = action_run
    inactive()
    first, second = N.NetRing(engine=object()), N.NetRing(engine=object())
    P.views(first)
    with P.inference_boundary():
        case.clock.now += .0002
        with first.sender_cost():
            sender_cpu.value += sender_s
            case.clock.now += sender_s
        clock = P._INFERENCE.clock
        clock.flush()
        P._CAD.base.append(.001)
        P._CAD.cost.append(.00002)
        P._CAD.n = 3
        P._CAD.done[1] = .0002
        account = P._CAD.inference_account
        P.views_closed(first)
        assert not P.ACTIVE
        if transition == "views":
            attach = lambda: P.views(second)
        elif transition == "install":
            monkeypatch.setattr(P, "_prepare", lambda *args: None)
            monkeypatch.setattr(P, "_check", lambda *args: None)
            monkeypatch.setattr(P, "_compile", lambda p: None)
            attach = lambda: P.install("reset-probe", {}, "")
        else:
            # The lifecycle reset precedes declaration validation.
            attach = lambda: P.install_piece("reset-probe", piece={}, manifest={})
        with P.inference_excluded() if excluded else nullcontext():
            on_subscription_thread(attach)
            assert P._CAD.inference_account is account
            assert P._CAD.inference_accounted == pytest.approx(sender_s)
            case.clock.now += .0003
        clock.flush()
        assert P._CAD.inference_s == pytest.approx(.0002 if excluded else .0005)
        assert P._CAD.extra == pytest.approx(sender_s)
        assert list(P._CAD.base) == [.001]
        assert list(P._CAD.cost) == [.00002]
        assert P._CAD.n == 3 and not P._CAD.done


def test_subscription_between_hosted_boundaries_retains_sender_cost(action_run, sender_cpu):
    case = action_run
    inactive()
    first, second = N.NetRing(engine=object()), N.NetRing(engine=object())
    P.views(first)
    with P.inference_boundary():
        P.at_step(1)
        case.clock.now += .0002
    assert P._CAD.inference and not P._INFERENCE_ACTIVE
    with first.sender_cost():
        sender_cpu.value += .00007
        case.clock.now += .00007
    P.views_closed(first)
    before = P._CAD.inference_s
    on_subscription_thread(lambda: P.views(second))
    with P.inference_boundary():
        case.clock.now += .0003
    assert P._CAD.inference_s - before == pytest.approx(.0003)
    assert P._CAD.extra == pytest.approx(.00007)
