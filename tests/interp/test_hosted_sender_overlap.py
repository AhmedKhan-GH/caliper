"""Controlled sender intervals for hosted cadence cost ownership."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import caliper
from caliper import _programs as P, _net_ring as N
from .test_hosted_boundary_clock import snapshot_ring
from .test_hosted_cadence import action_run


@pytest.fixture
def sender_cpu(monkeypatch):
    cpu = SimpleNamespace(value=0.)
    monkeypatch.setattr(N, "_ThreadCPU", lambda *args: SimpleNamespace(
        read=lambda: cpu.value, close=lambda: None))
    monkeypatch.setattr(N.time, "thread_time", lambda: cpu.value)
    return cpu


@pytest.mark.parametrize("inside,outside,other", [
    pytest.param(.00001, 0., 0., id="complete-overlap"),
    pytest.param(.00001, .00002, 0., id="partial-overlap"),
    pytest.param(.00001, .00002, .00003, id="overlap-and-separate-work"),
    pytest.param(0., .00001, .00001, id="separate-intervals"),
    pytest.param(0., .00004, .00001, id="separate-sender-exceeds-wall-cost"),
    pytest.param(0., .00001, .00004, id="separate-wall-exceeds-sender-cost"),
])
def test_sender_and_boundary_intervals_have_one_owner(
        action_run, monkeypatch, sender_cpu, inside, outside, other):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    original = P._publish_pass

    def publish(batch):
        original(batch)
        if batch:
            # One pump crosses the boundary. Its CPU total arrives only at its end.
            with ring.sender_cost():
                with P.inference_excluded():
                    case.clock.now += outside
                    sender_cpu.value += outside
                case.clock.now += inside
                sender_cpu.value += inside
            case.clock.now += other

    monkeypatch.setattr(P, "_publish_pass", publish)
    case.drive()
    assert case.shown == list(range(8))
    assert all(reply["status"] == "ok" for reply in case.replies)
    assert case.program.state == "running"
    assert not ring.passes                      # No snapshot owns these charges.
    assert len(P._CAD.cost) == 7
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.cost_s() == pytest.approx(case.clock.pass_s + inside + outside + other)


@pytest.mark.parametrize("change", ["replace", "close", "unsubscribe"])
def test_sender_cost_before_ring_transition_remains_separate(action_run, change):
    case = action_run
    first, second = snapshot_ring(case.clock), snapshot_ring(case.clock)
    first.publish_s, second.publish_s = 2., 8.
    P.views(first)
    with P.inference_boundary():
        P.at_step(1)
    # This work belongs to the old ring and precedes any new measured interval.
    first.publish_s += .00003
    case.clock.now += .00003
    if change == "replace":
        P.views(second)
        second.publish_s += .00002
        case.clock.now += .00002
        P.views_closed(second)
    elif change == "close":
        P.views_closed(first)
    else:
        P.views(None)
    assert P._CAD.extra == pytest.approx(.00005 if change == "replace" else .00003)
    assert P._CAD.inference_sender is None


@pytest.mark.parametrize("excluded", [False, True], ids=["measured", "excluded"])
def test_sender_cost_from_prior_cycle_survives_new_intervals(action_run, excluded):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    case.run.controls.actions["compare"].fn()
    for step in range(1, 13):
        with P.inference_boundary():
            P.at_step(step)
    assert len(P._CAD.cost) == 1
    assert P._CAD.cost[-1] == pytest.approx(case.clock.pass_s)
    ring.publish_s += .00007
    case.clock.now += .00007
    with P._LOCK:
        P._inference_sender(ring)
    # Existing credit must not cancel later, unrelated measured work.
    with P.inference_boundary():
        with P.inference_excluded() if excluded else nullcontext():
            case.clock.now += .00011
    assert P._CAD.extra == pytest.approx(.00007)
    for step in range(13, 16):
        with P.inference_boundary():
            P.at_step(step)
    assert len(P._CAD.cost) == 2
    assert P._CAD.extra == 0.
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.cost[-1] == pytest.approx(case.clock.pass_s + .00007 + (0. if excluded else .00011))


@pytest.mark.parametrize("before", [0., .000005, .00001])
@pytest.mark.parametrize("dispatch", ["ordinary", "data-view"])
def test_snapshot_start_splits_cpu_before_the_dispatch_lock(
        action_run, monkeypatch, sender_cpu, before, dispatch):
    case = action_run
    monkeypatch.setattr(N.time, "perf_counter", lambda: case.clock.now)
    ring = snapshot_ring(case.clock)
    P.views(ring)
    ring.open_pass(1)

    class Group:
        resolved = None
        stages = []
        last = 0.

    group = Group()
    view = SimpleNamespace(groups={"group": group}, seen=0., last=0.)
    tick = SimpleNamespace(served=set(), pending=[], mps=False)

    def apply(call, resolved):
        case.clock.now = .00005
        return object()

    monkeypatch.setattr(ring, "due", lambda *args: True)
    monkeypatch.setattr(ring, "_register", lambda *args: view)
    monkeypatch.setattr(ring, "_gated", lambda now: False)
    monkeypatch.setattr(ring, "_group_due", lambda *args: True)
    monkeypatch.setattr(ring, "_open_tick", lambda now: tick)
    monkeypatch.setattr(ring, "_complete", lambda *args: False)
    monkeypatch.setattr(ring, "_stage", lambda *args: SimpleNamespace(mps=False))
    monkeypatch.setattr(ring, "engine", SimpleNamespace(apply=apply))
    monkeypatch.setattr(caliper, "_routes", [SimpleNamespace(
        ring=ring, transport=SimpleNamespace(attached=lambda: True), sync=False)])
    monkeypatch.setattr(caliper, "_after", lambda *args: None)

    class DispatchLock:
        def __enter__(self):
            case.clock.now = .00004
            sender_cpu.value = .00001

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(caliper, "LOCK", DispatchLock())
    with P.inference_boundary(), P.inference_excluded():
        with ring.sender_cost():
            case.clock.now, sender_cpu.value = .00002, before
            if dispatch == "ordinary":
                caliper._dispatch(1, "view", None, lambda *args: SimpleNamespace(kind=1),
                                  None, None, None, None, None)
            else:
                monkeypatch.setattr(ring, "wants", lambda *args: True)
                monkeypatch.setattr(caliper._dataview, "DataCall", lambda *args: None)
                monkeypatch.setattr(caliper._dataview, "net_plan", lambda *args: SimpleNamespace(kind=1))
                caliper._dv_dispatch(SimpleNamespace(name="view", rate=None, options={}), "source")
    assert ring.gate_cost == pytest.approx(.00003)
    assert ring.gate_sender_s == pytest.approx(.00001 - before)
    assert P._CAD.inference_s + P._CAD.extra == pytest.approx(.00003 + before)
    assert ring.gate_taken == 1
    assert not P._CAD.inference_account.active


@pytest.mark.parametrize("change", ["replace", "close", "unsubscribe"])
@pytest.mark.parametrize("cross_thread", [False, True])
def test_active_sender_survives_subscription_change_and_late_counter(
        action_run, sender_cpu, change, cross_thread):
    case = action_run
    ring = snapshot_ring(case.clock)
    replacement = snapshot_ring(case.clock) if change == "replace" else None
    P.views(ring)
    with P.inference_boundary():
        P.at_step(1)
        with ring.sender_cost():
            case.clock.now += .00001
            sender_cpu.value += .00001
            def transition():
                if change == "close":
                    P.views_closed(ring)
                else:
                    P.views(replacement)
            if cross_thread:
                import threading
                worker = threading.Thread(target=transition)
                worker.start()
                worker.join(5.)
                assert not worker.is_alive()
            else:
                transition()
            assert ring.publish_s == 0.              # The pump has not reported its counter yet.
            case.clock.now += .00002
            sender_cpu.value += .00002
        case.clock.now += .00004                    # Separate wall work must remain.
    assert P._CAD.inference_s == pytest.approx(.0006 + .00004)
    assert P._CAD.extra == pytest.approx(.00003)
    assert not P._CAD.inference_account.active
    # A retired ring's next pump has no subscriber cost owner.
    with ring.sender_cost():
        sender_cpu.value += .00009
    with P.inference_boundary():
        pass
    assert P._CAD.extra == pytest.approx(.00003)


@pytest.mark.parametrize("send_step", [10, 11])
def test_outbox_and_sender_keep_separate_and_overlapping_costs(
        action_run, monkeypatch, sender_cpu, send_step):
    from collections import deque
    from caliper.interp import publish
    import torch

    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    publish("output.probs", torch.ones(1))
    link = SimpleNamespace(closed=False, attached=True)

    def post(*args):
        with ring.sender_cost():
            case.clock.now += .00001
            sender_cpu.value += .00001
        case.clock.now += .00002

    monkeypatch.setattr(P, "_post", post)
    monkeypatch.setattr(P, "_backlog", lambda *args: 0)
    for step in range(1, 13):
        if step == send_step:
            P._OUTBOX[id(link)] = [link, None, case.program, deque([(4, [b"part"])])]
        with P.inference_boundary():
            P.at_step(step)
    assert not P._OUTBOX
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.cost_s() == pytest.approx(.00005)


def test_sender_account_has_no_completed_interval_history(sender_cpu):
    ring = N.NetRing(engine=object())
    account = N._SenderAccount()
    ring._sender.watch(account)
    with ring.sender_cost():
        sender_cpu.value += .00001
        assert account.sample() == pytest.approx(.00001)
        assert len(account.active) == 1
    assert account.sample() == pytest.approx(.00001)
    assert not account.active
    assert ring._sender.reader is ring._sender.thread is ring._sender.current is None
    with pytest.raises(ValueError), ring.sender_cost():
        sender_cpu.value += .00002
        raise ValueError("pump failure")
    assert account.sample() == pytest.approx(.00003)
    assert ring.publish_s == pytest.approx(.00003)
    assert not account.active
    assert ring.sender_untracked() == pytest.approx(0.)


def test_live_cpu_samples_precede_counter_publication():
    import threading

    ring = N.NetRing(engine=object())
    account = N._SenderAccount()
    ring._sender.watch(account)
    entered, resume, worked, finish = (threading.Event() for _ in range(4))
    failures = []

    def sender():
        try:
            with ring.sender_cost():
                entered.set()
                assert resume.wait(5.)
                sum(range(10000))
                worked.set()
                assert finish.wait(5.)
        except BaseException as error:
            failures.append(error)
            entered.set()
            worked.set()

    worker = threading.Thread(target=sender)
    worker.start()
    try:
        assert entered.wait(5.)
        assert not failures
        first = account.sample()
        resume.set()
        assert worked.wait(5.)
        second = account.sample()
        assert second >= first >= 0.
        assert ring.publish_s == 0.
        assert len(account.active) == 1
    finally:
        resume.set()
        finish.set()
        worker.join(5.)
    assert not worker.is_alive()
    assert not failures
    assert account.sample() >= second
    assert not account.active
    assert ring.publish_s > 0.


def test_remote_clock_and_thread_time_use_the_same_cpu_origin():
    import threading
    import time

    reader = N._ThreadCPU(threading.get_ident(), threading.get_native_id())
    try:
        before = time.thread_time()
        sample = reader.read()
        after = time.thread_time()
    finally:
        reader.close()
    # The macOS query reports microseconds. These are clock consistency bounds.
    assert before - 1e-6 <= sample <= after + 1e-6


@pytest.mark.parametrize("step", [10, 11, 12])
def test_measured_sender_does_not_enter_either_base_half(action_run, sender_cpu, step):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    case.run.controls.actions["compare"].fn()
    for number in range(1, 14):
        with P.inference_boundary():
            if number == step:
                with ring.sender_cost():
                    sender_cpu.value += .00001
                    case.clock.now += .00001
            P.at_step(number)
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.cost[0] == pytest.approx(.00003)


def test_active_pump_crosses_cycles_without_a_late_duplicate(action_run, sender_cpu):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    case.run.controls.actions["compare"].fn()
    for step in range(1, 13):
        with P.inference_boundary():
            P.at_step(step)
    assert len(P._CAD.cost) == 1
    with ring.sender_cost():
        sender_cpu.value += .00007
        case.clock.now += .00007
        for step in range(13, 17):
            with P.inference_boundary():
                P.at_step(step)
        assert P._CAD.cost[-1] == pytest.approx(.00009)
        assert ring.publish_s == 0.
        sender_cpu.value += .00009
        case.clock.now += .00009
    for step in range(17, 21):
        with P.inference_boundary():
            P.at_step(step)
    assert len(P._CAD.cost) == 3
    assert P._CAD.cost[-1] == pytest.approx(.00011)
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.extra == 0.
    assert not P._CAD.inference_account.active


@pytest.mark.parametrize("excluded", [False, True])
def test_snapshot_floor_removes_only_its_sender_overlap(
        action_run, monkeypatch, sender_cpu, excluded):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    monkeypatch.setattr(N.time, "perf_counter", lambda: case.clock.now)
    original = caliper.step

    def step():
        if ring.gate_open:
            with P.inference_excluded() if excluded else nullcontext():
                stamp, start = P._sender_stamp(), case.clock.now
                with ring.sender_cost():
                    sender_cpu.value += .00001
                    case.clock.now += .00001
                case.clock.now += .00002
                ring._snapshot_cost(start, stamp)
                ring.gate_taken += 1
        original()

    monkeypatch.setattr(caliper, "step", step)
    case.drive()
    assert case.shown == list(range(8))
    assert len(P._CAD.cost) == 7
    assert P._CAD.cost_s() == pytest.approx(.00005)
    assert P._CAD.step_s() == pytest.approx(.0006)


@pytest.mark.parametrize("source", ["tick", "series"])
def test_snapshot_endpoints_keep_the_original_wall_cost(
        action_run, monkeypatch, sender_cpu, source):
    import torch

    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    monkeypatch.setattr(N.time, "perf_counter", lambda: case.clock.now)
    ring.open_pass(1)

    def copy(*args):
        with ring.sender_cost():
            sender_cpu.value += .00001
            case.clock.now += .00001
        case.clock.now += .00002

    with P.inference_boundary(), P.inference_excluded():
        if source == "tick":
            ring.tick = object()
            monkeypatch.setattr(ring, "_close_tick", copy)
            ring.close_pass(1, case.clock.now)
            # close_pass retains the costs for the checkpoint's final sample.
            ring.gate_open = True
        else:
            ring.tap("series", 1.)
            ring.series["series"].watched = True
            monkeypatch.setattr(ring, "_tick_series", copy)
            ring.tap("series", torch.ones(1))
    assert ring.gate_cost == pytest.approx(.00003)
    assert ring.gate_sender_s == pytest.approx(.00001)
    assert P._CAD.inference_s + P._CAD.extra == pytest.approx(.00003)


def test_subscription_before_first_checkpoint_owns_sender_cpu(action_run, sender_cpu):
    import threading

    case = action_run
    ring = snapshot_ring(case.clock)
    with P.inference_boundary():
        worker = threading.Thread(target=lambda: P.views(ring))
        worker.start()
        worker.join(5.)
        assert not worker.is_alive()
        assert not P._CAD.inference
        with ring.sender_cost():
            sender_cpu.value += .00001
            case.clock.now += .00001
    assert P._CAD.inference_s == pytest.approx(0.)
    assert P._CAD.extra == pytest.approx(.00001)
    assert not P._INFERENCE_ACTIVE


@pytest.mark.parametrize("native", [False, True])
def test_both_sender_loops_publish_cpu_metadata_without_wait_cost(sender_cpu, monkeypatch, native):
    import asyncio
    from caliper import _net, _net_native

    ring = N.NetRing(engine=object())
    account = N._SenderAccount()
    ring._sender.watch(account)
    server = SimpleNamespace(ring=ring, closed=False, counters={"pump_errors": 0})
    calls = []

    def pump():
        calls.append(1)
        if native and len(calls) == 1:
            return False                         # The original startup call precedes subscription.
        sender_cpu.value += .00002
        server.closed = True
        return True

    def control(wait):
        sender_cpu.value += .001                  # Outside the measured pump in both senders.
        return []

    async def sleep(delay):
        control(delay)
        raise asyncio.CancelledError

    server.producer = SimpleNamespace(
        pump_once=pump, wait_ms=lambda: 0, controls=SimpleNamespace(flush=lambda: None))
    if native:
        server.nio = SimpleNamespace(control=control)
        _net_native.NativeServer._run(server)
    else:
        monkeypatch.setattr(_net.asyncio, "sleep", sleep)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_net.NetServer._pump(server))
    assert account.sample() == pytest.approx(.00002)
    assert ring.publish_s == pytest.approx(.00002)
    assert not account.active
