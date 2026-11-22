"""The shared cadence of the work that viewers ask for (caliper._programs with caliper._net_ring).

The snapshots of the network route and the GPU programs share one controller: the checkpoint
before a step decides whether that step is a pass, the view calls of a pass step take their
snapshots, and its checkpoint closes them with one event and runs the programs that are due with
one wait. The tests run a training loop on a fake clock, as test_programs.py does, with the
snapshots of a pass as CPU and GPU time of the training thread; a second part checks the gate of
the real ring without a viewer.
"""
from __future__ import annotations

import collections
import math

import pytest

from caliper import _adapter, _net_ring
from caliper import _programs as P


class Sim:
    """A training loop on a fake clock, as torch's MPS queue runs it (test_programs.Sim): the
    thread queues a step's GPU work in ``cpu`` seconds, the GPU runs each step in ``gpu`` seconds
    once a mark gives it the work, and a program pass waits for the queue, then runs ``pass_s``.
    As Metal does, a commit waits while ``inflight`` command buffers have not completed, which
    bounds how far the thread runs ahead of the GPU."""

    INFLIGHT = 16

    def __init__(self, cpu, gpu, pass_s=0.1e-3):
        self.flight = collections.deque()   # completion times of the committed buffers
        self.t = 0.0
        self.gpu_free = 0.0
        self.queued = 0.0
        self.cpu, self.gpu, self.pass_s = cpu, gpu, pass_s
        self.ran = 0
        self.records = 0
        self.waits = 0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 1e-6)

    def event(self):
        return SimEvent(self)

    def run(self, handles, step, time_s, every):
        self.ran += 1
        self.gpu_free = max(self.gpu_free, self.t) + self.pass_s / 4
        self.t += self.pass_s
        return {"ran": 1}

    def end(self):
        """The wall time of the run so far: the later of the thread and the GPU."""
        return max(self.t, self.gpu_free)


class SimEvent:
    def __init__(self, sim):
        self.sim, self.at = sim, math.inf

    def record(self):
        s = self.sim
        s.records += 1
        while s.flight and s.flight[0] <= s.t:
            s.flight.popleft()
        if len(s.flight) >= s.INFLIGHT:          # Metal's limit: the commit waits
            s.t = s.flight.popleft()
        s.gpu_free = max(s.gpu_free, s.t) + s.queued
        s.queued = 0.0
        self.at = s.gpu_free
        s.flight.append(self.at)

    def query(self):
        return self.sim.t >= self.at


class SimRuntime:
    def __init__(self, sim):
        self.sim = sim

    def run(self, handles, step, time_s, every):
        return self.sim.run(handles, step, time_s, every)

    def status(self, h):
        return {"state": "running", "last_pass_ms": 0.05}

    def remove(self, h):
        pass

    def stats(self):
        return {}


class SimRing:
    """The part of caliper._net_ring.NetRing that the cadence drives, with the views' snapshots
    as time on the fake clock: a pass step's view calls cost ``snap_s`` on the thread and
    ``snap_gpu`` on the GPU, and each view asks for ``rate`` snapshots a second."""

    def __init__(self, sim, snap_s, snap_gpu=0.0, rate=60.0):
        self.sim = sim
        self.snap_s, self.snap_gpu = snap_s, snap_gpu
        self.calls = 30                            # the view calls of a pass
        self.publish_s = 0.0                       # the sender thread's time (none here)
        self.interval = 1.0 / rate
        self.gate_open = False
        self.registered = True
        self.cadence: dict = {}
        self.pass_t = -math.inf
        self.last = -math.inf
        self.taken = 0
        self.steps: list = []                     # the steps whose view calls took snapshots
        self.passes = collections.deque(maxlen=_net_ring.PASS_RATE_SAMPLES + 1)

    def step_tick(self, step, now):
        pass

    def wants_pass(self, now):
        return now - self.last >= _net_ring.PASS_SLACK * self.interval

    def open_pass(self, step):
        self.gate_open = True
        self.gate_step = step
        self.gate_cost = 0.0

    def hooks(self, step):
        """The view calls of one step: snapshots only while the gate is open, after the look at
        the marks that the ring's first call of a pass makes."""
        if self.gate_open and self.sim.t - self.last >= self.interval / 2:
            self.sim.t += self.snap_s
            self.gate_cost += self.snap_s
            self.sim.queued += self.snap_gpu
            self.last = self.sim.t
            self.taken += 1
            self.steps.append(step)

    def close_pass(self, step, now):
        taken, self.taken = self.taken, 0
        self.gate_open = False
        if taken:
            self.pass_t = now
            self.passes.append((now, step))
        return taken

    def pass_rate(self):
        return _net_ring.NetRing.pass_rate(self)

    def asked_interval(self):
        return self.interval

    def release_gate(self):
        self.gate_open = False


@pytest.fixture
def loop(monkeypatch):
    """The cadence over the fake clock of a Sim, with a SimRing and optionally one program."""
    made = {}

    def make(cpu, gpu, snap_s=0.0, snap_gpu=0.0, rate=60.0, program=False, pass_s=0.1e-3, sleep=0.0):
        s = Sim(cpu, gpu, pass_s)
        s.pause = sleep
        monkeypatch.setattr(P, "_clock", s.clock)
        monkeypatch.setattr(_adapter, "time", type("T", (), {"sleep": staticmethod(s.sleep)}))
        monkeypatch.setitem(P._STATE, "rt", SimRuntime(s))
        monkeypatch.setitem(P._STATE, "marks", _adapter.StepMarks(s.event, size=64))
        for k, v in (("step", None), ("dense", True), ("drained", 0), ("short", 0), ("long", 0),
                     ("switches", 0), ("pass", None), ("pass_budget", True), ("floor", 0),
                     ("view_s", collections.deque(maxlen=P.VIEW_SAMPLES))):
            monkeypatch.setitem(P._STATE, k, v)
        monkeypatch.setattr(P, "_outputs", lambda rt, p: {})
        monkeypatch.setattr(P, "_publish", lambda p, outs: None)
        P._CAD.reset()
        ring = None
        if snap_s or snap_gpu:
            ring = SimRing(s, snap_s, snap_gpu, rate)
            P.views(ring)
        if program:
            p = P._Program("sim", {"output": {"kind": "pixels", "width": 1, "height": 1}}, "", None, "")
            p.handle, p.state = 1, "compiled"
            P._PROGS["sim"] = p
            P._set_active()
            made["p"] = p
        made["ring"] = ring

        def train(steps, first=0):
            for k in range(first, first + steps):
                if ring is not None:
                    ring.hooks(k)             # the view calls come at the start of the forward
                s.t += s.cpu                  # forward and backward of step k, optimizer of k - 1
                s.queued += s.gpu
                if P.ACTIVE:
                    P.at_step(k)
                s.t += s.pause                # a loop that also waits on the CPU
        s.train = train
        s.ring = ring
        return s

    yield make
    P._PROGS.pop("sim", None)
    P.views(None)
    P._set_active()
    P._CAD.reset()


def base_time(cpu, gpu, steps):
    """The wall time of ``steps`` steps with nothing attached and a mark at each step: the
    slower side sets the pace."""
    return steps * max(cpu, gpu)


def test_costly_snapshots_keep_to_the_budget(loop):
    # The proof's model: a step of 3 ms on the GPU, and a pass of 34 views that costs the training
    # thread 7 ms. Every step's snapshots, as the ticks at 60 Hz took them, cost 40 % of the time.
    s = loop(cpu=1.0e-3, gpu=3.0e-3, snap_s=7.0e-3, snap_gpu=0.3e-3)
    s.train(6000)
    b, d = P._CAD.step_s(), P._CAD.cost_s()
    assert b == pytest.approx(3.0e-3, rel=0.05)
    n = P._CAD.n
    assert n >= math.ceil(d / (P.VIEWS_SHARE * b))   # attached views keep to the smaller share
    assert base_time(1.0e-3, 3.0e-3, 6000) / s.end() >= 0.95
    st = P.views_stats()
    assert st["every"] == n and st["limit"] == "budget"
    assert st["share"] <= P.VIEWS_SHARE * 1.05 and st["budget"] == P.VIEWS_SHARE
    # The views still update a few times a second.
    assert st["passes_per_s"] is not None and st["passes_per_s"] >= 2


@pytest.mark.parametrize("sleep", [0.0, 10e-3])
def test_the_pass_is_measured_at_its_own_cost(loop, sleep):
    # The view calls come at the start of a step, long before its checkpoint: the marks that the
    # pass's checkpoint finds move back by the pass's own work, so that D is that work and not
    # also the rest of the step. A loop that sleeps 10 ms a step, as the proof's script does,
    # keeps the same D.
    s = loop(cpu=3.0e-3, gpu=1.0e-3, snap_s=2.0e-3, sleep=sleep)
    s.train(4000)
    assert P._CAD.cost_s() == pytest.approx(2.0e-3, rel=0.15)
    assert P._CAD.step_s() == pytest.approx(3.0e-3 + sleep, rel=0.05)


def test_cheap_snapshots_follow_their_own_rate(loop):
    # A pass that costs 20 µs: the budget allows a pass every 3 steps, and the views' 60 a second
    # set the pace instead.
    s = loop(cpu=0.3e-3, gpu=0.4e-3, snap_s=20e-6)
    s.train(5000)
    st = P.views_stats()
    assert st["limit"] == "rate"
    assert st["passes_per_s"] == pytest.approx(60, rel=0.25)
    assert base_time(0.3e-3, 0.4e-3, 5000) / s.end() >= 0.98


def test_views_and_programs_share_one_pass_and_one_wait(loop):
    s = loop(cpu=0.3e-3, gpu=0.42e-3, snap_s=1.0e-3, program=True)
    s.train(6000)
    steps = set(s.ring.steps)
    p = P._PROGS["sim"]
    assert p.runs > 10
    # Each program pass ran at a step whose view calls also took their snapshots.
    passes = sorted(s.ring.steps)
    assert p.last_step in steps
    # The program runs at most once in every N steps of the shared cadence.
    assert P.stats()["every"] == P._CAD.n
    assert base_time(0.3e-3, 0.42e-3, 6000) / s.end() >= 0.95
    assert all(b - a >= P.N_MIN for a, b in zip(passes, passes[1:]))


def test_the_floor_keeps_cheap_views_updating_beside_a_costly_program(loop):
    # A program pass of 50 ms on steps of 1 ms: the cadence allows it once in about 1.25 s. The
    # views' own work is 0.2 ms a pass, so they keep a pass of their own every FLOOR_S, which the
    # cadence does not count, and the program keeps to N.
    s = loop(cpu=0.5e-3, gpu=1.0e-3, snap_s=0.2e-3, program=True, pass_s=50e-3)
    s.train(20000)
    st = P.views_stats()
    assert st["floor_passes"] > 5
    floors = [x for x in s.ring.steps if x not in P._CAD.passes]
    assert floors
    n, b = P._CAD.n, P._CAD.step_s()
    assert n * b > P.FLOOR_S
    gaps = [(b2 - a2) * b for a2, b2 in zip(s.ring.steps[-6:], s.ring.steps[-5:])]
    assert max(gaps) <= P.FLOOR_S * 1.6
    assert base_time(0.5e-3, 1.0e-3, 20000) / s.end() >= 0.94


def test_costly_views_get_no_floor(loop):
    # A pass of 60 ms of the views' own work: a pass every FLOOR_S would cost 12 %, thus only
    # the budget sets the pace.
    s = loop(cpu=0.5e-3, gpu=1.0e-3, snap_s=60e-3)
    s.train(20000)
    st = P.views_stats()
    assert st["floor_passes"] == 0 and st["limit"] == "budget"
    # The stall also empties the GPU's queue, which the step after it fills again.
    assert 60e-3 <= P._CAD.cost_s() <= 80e-3
    t0, k0 = s.end(), 20000
    s.train(20000, first=k0)
    assert 20000 * 1.0e-3 / (s.end() - t0) >= 0.95



def test_the_last_single_step_is_a_pass_whatever_its_views_cost(loop, monkeypatch):
    # Costly views get no floor, thus the run that single-steps once paused at a step that was
    # no pass and showed an older one: the step after which the run pauses is a pass.
    from caliper import _controls as C
    s = loop(cpu=0.5e-3, gpu=1.0e-3, snap_s=60e-3)
    s.train(20000)
    assert P.views_stats()["floor_passes"] == 0
    k = 20000
    while P._CAD.due(k + 1):                       # a step that the cadence makes no pass
        s.train(1, first=k)
        k += 1
    monkeypatch.setattr(C.CONTROLS, "state", C.RUN_STEPPING)
    monkeypatch.setattr(C.CONTROLS, "steps_left", 1)
    s.train(1, first=k)
    assert P._STATE["pass"] == k + 1
    monkeypatch.setattr(C.CONTROLS, "state", C.RUN_RUNNING)
    monkeypatch.setattr(C.CONTROLS, "steps_left", 0)
    s.train(1, first=k + 1)
    assert s.ring.steps[-1] == k + 1

def test_a_pass_that_took_nothing_does_not_count(loop):
    s = loop(cpu=0.3e-3, gpu=0.4e-3, snap_s=1.0e-3, rate=1000.0)
    s.train(200)
    s.ring.snap_s = 0.0
    s.ring.hooks = lambda step: None           # the views stopped calling
    s.train(200, first=200)
    # A pass decided for a step whose calls then took nothing counts as a step without a pass;
    # only the pass decided for the next step can still stand.
    assert {x for x in P._CAD.passes if 200 <= x < 400} == set()
    assert P._STATE["pass"] in (None, 400)


# ---- the gate of the real ring ------------------------------------------------------------------
class _Link:
    pass


def _subscribed_ring():
    import torch
    from caliper._net_dataview import Call
    from caliper._convert import plan_show
    r = _net_ring.NetRing()
    plan = plan_show(torch.ones(4, 4), 0, None, None, False)
    call = Call(plan.kind, "v", plan, (torch.ones(4, 4), 0, None, None, False))
    r._register(call, 0.0)
    v = r.views["v"]
    g = _net_ring.Group("v", r.engine.resolve(v.entry(), None), "k")
    g.subs[(_Link(), 1)] = None
    v.groups["k"] = g
    return r, plan


def test_the_gate_holds_snapshots_back_between_passes(monkeypatch):
    seen = []
    monkeypatch.setattr(P, "views", lambda ring: seen.append(ring))
    r, plan = _subscribed_ring()
    r.changed()
    assert seen == [r] and r.registered
    import time
    now = time.perf_counter()
    r.views["v"].offer_t = now                    # the offer is fresh: no refresh is due
    r.views["v"].seen = now                       # and the view is called at each step
    r.step_tick(1, now)
    assert r.next_due > now                       # attached() is False until the gate opens
    assert not r.due("v", r.default_interval, now)
    r.open_pass(2)
    assert r.next_due == 0.0
    assert r.due("v", r.default_interval, time.perf_counter())
    import torch
    t = torch.ones(4, 4)
    assert r.take_call("v", plan, time.perf_counter(), (t, 0, None, None, False))
    assert r.close_pass(2, time.perf_counter()) == 1
    assert not r.gate_open and r.pending
    r.step_tick(2, time.perf_counter())
    assert not r.due("v", r.default_interval, time.perf_counter())


def test_a_closed_ring_leaves_the_cadence():
    # A server that closes takes its ring out of the shared cadence, so that later steps neither
    # keep the programs at the share of attached views nor gate the scene colors; a ring that
    # empties after another took its place leaves the newer one registered.
    r, _ = _subscribed_ring()
    r2, _ = _subscribed_ring()
    try:
        r.changed()
        assert P._VIEWS["ring"] is r and P.ACTIVE
        r2.changed()
        assert P._VIEWS["ring"] is r2 and not r.gate_open
        r.views["v"].groups.clear()
        r.changed()                                  # the old ring empties: the new one stays
        assert P._VIEWS["ring"] is r2 and P.ACTIVE
        r2.close()
        assert P._VIEWS["ring"] is None and not r2.registered
        r2.changed()                                 # a link that leaves after the close
        assert P._VIEWS["ring"] is None
    finally:
        P.views(None)


def test_the_ring_follows_the_clock_again_when_the_checkpoints_stop(monkeypatch):
    monkeypatch.setattr(P, "views", lambda ring: None)
    r, plan = _subscribed_ring()
    r.changed()
    r.step_tick(1, 0.0)                           # a checkpoint long ago
    import time
    assert r.due("v", r.default_interval, time.perf_counter())
    assert not r.step_mode


def test_a_snapshot_carries_the_cadence_of_its_pass(monkeypatch):
    import time

    import numpy as np
    from caliper import _wire as w
    monkeypatch.setattr(P, "views", lambda ring: None)
    r, plan = _subscribed_ring()
    r.changed()
    cadence = {"every": 59, "passes_per_s": 5.6, "step_ms": 3.0, "pass_ms": 7.0, "share": 0.039,
               "budget": P.SHARE, "limit": "budget"}
    r.cadence = cadence
    r.views["v"].offer_t = time.perf_counter()
    r.step_tick(1, time.perf_counter())
    r.open_pass(2)
    import torch
    assert r.take_call("v", plan, time.perf_counter(), (torch.ones(4, 4), 0, None, None, False))
    r.close_pass(2, time.perf_counter())
    assert r.pending[-1].cadence == cadence
    # The key of the 3.2 draft: present for a peer with "programs", left out otherwise.
    data = np.ones((4, 4), np.float32)
    snap = w.Snapshot(1, "v", 1, 0, int(w.Kind.HEATMAP), {}, data, cadence=cadence)
    assert w.decode(w.encode(snap)).cadence == cadence
    assert w.decode(w.encode(w.Snapshot(1, "v", 1, 0, int(w.Kind.HEATMAP), {}, data))).cadence is None


def test_the_scene_colors_copy_nothing_from_the_host_after_the_first_time(monkeypatch):
    # On MPS a copy from the host waits for torch's whole queue: the colors of the scene model
    # once did it at each publish, which cost a third of the throughput of a small transformer.
    import types

    import torch
    from caliper._watch_run import RunHooks
    hooks = types.SimpleNamespace(_torch=torch, _lut={},
                                  means=[torch.tensor(0.5), None, torch.tensor(2.0)])
    first = RunHooks._colors(hooks)[0]
    made = []
    real = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: made.append(k.get("device")) or real(*a, **k))
    again = RunHooks._colors(hooks)[0]
    assert made == [] and torch.equal(first, again)
    assert again.shape == (3, 3)


def test_a_watched_float_series_wakes_the_sender_at_most_at_the_session_rate(monkeypatch):
    import time
    monkeypatch.setattr(P, "views", lambda ring: None)
    wakes = []
    r = _net_ring.NetRing(wake=lambda: wakes.append(1))
    r.tap("lr", 0.1)
    r.series["lr"].watched = True
    wakes.clear()
    for _ in range(50):                           # on the clock: each value wakes it
        r.tap("lr", 0.1)
    assert len(wakes) == 50
    wakes.clear()
    r.step_tick(1, time.perf_counter())          # under the pass gate
    t0 = time.perf_counter()
    for _ in range(200):
        r.tap("lr", 0.1)
    elapsed = time.perf_counter() - t0
    assert len(wakes) <= 1 + elapsed / r.tick_interval
    assert len(r.series["lr"].q) == 251          # every value is kept for the sender


# ---- the real ring on the fake clock --------------------------------------------------------------
class _RingClock:
    """The clock of caliper._net_ring as the fake clock of a Sim."""

    def __init__(self, s):
        import time
        self.perf_counter = s.clock
        self.monotonic_ns, self.time_ns, self.sleep = time.monotonic_ns, time.time_ns, time.sleep


def _ring_on_clock(monkeypatch, s, names):
    """A real ring on the fake clock of ``s``, with a subscribed view of each name in ``names``,
    and the view calls of one step as caliper._dispatch makes them: NetTransport.attached(),
    then due() and take_call(). The sender is not there, thus each snapshot's staging buffers
    are free again at once. Returns the ring and a function that makes the calls of one step
    and returns the snapshots they took as (name, gate open) pairs."""
    import torch
    from caliper._convert import plan_show
    from caliper._net_dataview import Call
    monkeypatch.setattr(_net_ring, "time", _RingClock(s))
    r = _net_ring.NetRing()
    t = torch.ones(4, 4)
    plan = plan_show(t, 0, None, None, False)
    args = (t, 0, None, None, False)
    for name in names:
        r._register(Call(plan.kind, name, plan, args), s.t)
        v = r.views[name]
        v.seen = s.t                              # as take_call() leaves a view it registers
        g = _net_ring.Group(name, r.engine.resolve(v.entry(), None), "k")
        g.subs[(_Link(), 1)] = None
        v.groups["k"] = g
    r.changed()

    def calls(names_now):
        took = []
        for name in names_now:
            now = s.t
            if now >= r.next_due and r.due(name, r.default_interval, now):   # attached(), due()
                if r.take_call(name, plan, now, args):
                    took.append((name, r.gate_open))
        for p in r.pending:
            p.stage.busy = False
        r.pending.clear()
        return took
    return r, calls


def _train_views(s, r, calls, steps, gap, every):
    """Train ``steps`` steps on the ring of _ring_on_clock: view "rare" is called every ``gap``
    steps, and view "every", when ``every``, at each step. Returns the snapshots as (step, name,
    gate open) and the held times of the checkpoints that closed the gate."""
    took, holds = [], []
    for k in range(steps):
        names = (["every"] if every else []) + (["rare"] if k % gap == 0 else [])
        took += [(k, name, gate) for name, gate in calls(names)]
        s.t += s.cpu
        s.queued += s.gpu
        P.at_step(k)
        if not r.gate_open:
            holds.append(r.held_t)
    return took, holds


def test_a_view_called_less_than_once_a_second_keeps_updating(loop, monkeypatch):
    # A subscribed view called every 2 s while the script calls caliper.step(): its calls between
    # passes once returned after attached(), so the passes never saw it again, and it showed its
    # first snapshot for good. A call held back while its group is overdue by more than FLOOR_S
    # now takes its snapshot.
    s = loop(cpu=1.0e-3, gpu=1.0e-3)
    r, calls = _ring_on_clock(monkeypatch, s, ["rare"])
    gap = 2000                                    # steps of about 1 ms
    took, _ = _train_views(s, r, calls, 12 * gap, gap, every=False)
    assert [k for k, _, _ in took] == list(range(0, 12 * gap, gap))
    assert r.counters["held"] == 11               # the first call came before the checkpoints


def test_a_view_called_at_each_step_keeps_to_the_passes(loop, monkeypatch):
    # Beside the view called every 2 s, a view called at each step takes its snapshots only in
    # the passes, and without the rare view no checkpoint lets a call through the closed gate.
    s = loop(cpu=1.0e-3, gpu=1.0e-3)
    r, calls = _ring_on_clock(monkeypatch, s, ["every", "rare"])
    gap = 2000
    took, _ = _train_views(s, r, calls, 12 * gap, gap, every=True)
    assert [k for k, name, _ in took if name == "rare"] == list(range(0, 12 * gap, gap))
    every = [(k, gate) for k, name, gate in took if name == "every"]
    assert len(every) > 100 and all(gate for k, gate in every if k > 0)
    s2 = loop(cpu=1.0e-3, gpu=1.0e-3)
    r2, calls2 = _ring_on_clock(monkeypatch, s2, ["every"])
    took2, holds = _train_views(s2, r2, calls2, 6 * gap, gap, every=True)
    assert holds and all(h == math.inf for h in holds)
    assert r2.counters["held"] == 0 and all(gate for k, _, gate in took2 if k > 0)
