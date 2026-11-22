"""The facade of GPU programs (caliper._programs) over the producer runtime (caliper._producer).

The cadence formula and the refusals of a declaration run anywhere. The passes need an MPS device
and the native module, and they publish through stand-ins of ``caliper.scene`` and
``caliper.show``, thus no server and no viewer runs. A test that starts ``caliper.watch(model,
optimizer)`` runs in its own process with the shared-memory session off.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import warnings

import pytest
import torch

import caliper
from caliper import _programs as P

MPS = torch.backends.mps.is_available()
try:
    import caliper._producer  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False
needs_device = pytest.mark.skipif(not (MPS and NATIVE), reason="needs MPS and caliper._producer")

# One point for each row of w: x the row, y the row's norm, z the step of the frame uniform.
ROW_NORMS = """struct CaliperFrame { step: u32, pass_index: u32, max_count: u32, width: u32,
                     height: u32, time_s: f32, every: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(4) var<uniform> caliper: CaliperFrame;
@compute @workgroup_size(64)
fn norms(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= ROWSu) { return; }
    var s = 0.0;
    for (var c = 0u; c < COLSu; c++) { let v = w[row * COLSu + c]; s += v * v; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = f32(row);
    positions[i * 3u + 1u] = sqrt(s);
    positions[i * 3u + 2u] = f32(caliper.step);
    values[i] = sqrt(s);
}
"""

PIXELS = """@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 8u || id.y >= 4u) { return; }
    let v = clamp(abs(w[id.y * 8u + id.x]), 0.0, 1.0);
    pixels[id.y * 8u + id.x] = pack4x8unorm(vec4<f32>(v, 0.0, 1.0 - v, 1.0));
}
"""


def _row_norms(rows, cols):
    return ROW_NORMS.replace("ROWS", str(rows)).replace("COLS", str(cols))


def _geometry(rows, every=None, time_ms=None, writes=False, source="live:fc.weight"):
    decl = {"inputs": [{"var": "w", "source": source}],
            "output": {"kind": "geometry", "panel": "p", "part": "points", "max_count": rows,
                       "arrays": ["positions", "values"]},
            "limits": {}}
    if every:
        decl["limits"]["every"] = every
    if time_ms:
        decl["limits"]["time_ms"] = time_ms
    src = _row_norms(rows, 8)
    if writes:
        src = src.replace("var<storage, read> w", "var<storage, read_write> w")
    return decl, src


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
        self.out = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.out(torch.relu(self.fc(x)))


class FakeScene:
    space = "3d"

    def __init__(self, calls):
        self.calls = calls

    def points(self, path, positions, **kw):
        self.calls.append(("points", path, positions.clone(), {k: v for k, v in kw.items()}))
        return True

    def lines(self, path, positions, **kw):
        self.calls.append(("lines", path, positions.clone(), kw))
        return True

    def clear(self, path="", **kw):
        self.calls.append(("clear", path))
        return True


@pytest.fixture
def run(monkeypatch):
    """A model on MPS attached to the facade, and stand-ins of the publish path."""
    calls, shown = [], []
    monkeypatch.setattr(P, "SHARE", 1e6)             # N = 1: a pass on each due step
    monkeypatch.setattr(P, "N_MIN", 1)
    monkeypatch.setattr(caliper, "scene", lambda name="scene", **kw: FakeScene(calls))
    monkeypatch.setattr(caliper, "show", lambda t, name="tensor", **kw: shown.append((name, t.clone())))
    monkeypatch.setattr(caliper, "remove", lambda name: shown.append(("removed", name)))
    torch.manual_seed(0)
    net = Net().to("mps") if MPS else Net()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    P.attach(net, opt)
    yield net, opt, calls, shown
    P.detach()
    rt = P._STATE["rt"]
    if rt is not None:
        rt.close()
    P._STATE["rt"] = None
    P.listeners.clear()
    P._CAD.reset()


def _train(net, opt, steps, first=0):
    x = torch.randn(16, 8, device=next(net.parameters()).device)
    for i in range(first, first + steps):
        loss = net(x).square().mean()
        opt.zero_grad()
        loss.backward()
        caliper.step(i)            # the checkpoint of watch(model, optimizer): before the update
        opt.step()


class Sim:
    """A training loop on a fake clock, as torch's MPS queue runs it.

    The training thread queues the GPU work of a step in ``cpu`` seconds and goes on without a
    wait. The GPU gets the queued work only at a mark (a commit of the queue) and runs each step in
    ``gpu`` seconds, in order. A pass waits for the queue and then runs ``pass_s`` seconds, of which
    ``pass_gpu`` on the GPU. ``passes`` can give the time of single passes, for an outlier.
    """

    def __init__(self, cpu, gpu, pass_s, pass_gpu=None, item=False):
        self.item = item        # the loop waits for the GPU at each step, as loss.item() does
        self.records = 0
        self.t = 0.0            # the training thread's clock
        self.gpu_free = 0.0     # when the GPU finishes the work it has
        self.queued = 0.0       # GPU work queued since the last mark
        self.cpu, self.gpu, self.pass_s = cpu, gpu, pass_s
        self.pass_gpu = pass_s / 4 if pass_gpu is None else pass_gpu
        self.passes = {}        # pass number -> its time, else pass_s
        self.ran = 0
        self.sleeps = 0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps += 1
        self.t += max(s, 1e-6)

    def event(self):
        return SimEvent(self)

    def train(self, steps, first=0):
        for k in range(first, first + steps):
            self.t += self.cpu              # forward and backward of step k, optimizer of k - 1
            self.queued += self.gpu
            if self.item:                   # torch commits the queue and waits for it
                self.gpu_free = max(self.gpu_free, self.t) + self.queued
                self.queued = 0.0
                self.t = self.gpu_free
            P.at_step(k)

    def run(self, handles, step, time_s, every):
        p = self.passes.get(self.ran, self.pass_s)
        self.ran += 1
        self.gpu_free = max(self.gpu_free, self.t) + self.pass_gpu
        self.t += p
        return {"ran": 1}


class SimEvent:
    def __init__(self, sim):
        self.sim, self.at = sim, float("inf")

    def record(self):
        s = self.sim
        s.records += 1
        s.gpu_free = max(s.gpu_free, s.t) + s.queued
        s.queued = 0.0
        self.at = s.gpu_free

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

    def release(self, h):
        pass

    def stats(self):
        return {}


@pytest.fixture
def sim(monkeypatch):
    """The facade with one program over the fake clock of a Sim: no device and no native module."""
    from caliper import _adapter

    made = {}

    def make(cpu, gpu, pass_s, **kw):
        s = Sim(cpu, gpu, pass_s, **kw)
        monkeypatch.setattr(P, "_clock", s.clock)
        monkeypatch.setattr(_adapter, "time", type("T", (), {"sleep": staticmethod(s.sleep)}))
        monkeypatch.setitem(P._STATE, "rt", SimRuntime(s))
        monkeypatch.setitem(P._STATE, "marks", _adapter.StepMarks(s.event, size=64))
        monkeypatch.setitem(P._STATE, "step", None)
        for k, v in (("dense", True), ("drained", 0), ("short", 0), ("long", 0), ("switches", 0)):
            monkeypatch.setitem(P._STATE, k, v)
        monkeypatch.setattr(P, "_outputs", lambda rt, p: {})
        monkeypatch.setattr(P, "_publish", lambda p, outs: None)
        P._CAD.reset()
        p = P._Program("sim", {"output": {"kind": "pixels", "width": 1, "height": 1}}, "", None, "")
        p.handle, p.state = 1, "compiled"
        P._PROGS["sim"] = p
        P._set_active()
        made["p"] = p
        return s

    yield make
    P._PROGS.pop("sim", None)
    P._set_active()
    P._CAD.reset()
    P.listeners.clear()


def _bounds(cost_ms, late_ms, step_ms):
    """The range of N for a true overhead ``cost_ms`` that the facade may see late by ``late_ms``."""
    low = math.ceil(cost_ms / (0.04 * step_ms))
    return low, math.ceil((cost_ms + late_ms) / (0.04 * step_ms)) * (1 + P.CHANGE) + 1


def test_the_cadence_converges_on_a_gpu_bound_run(sim):
    # The GPU is the slower side: b is the GPU's step, and a pass costs its own time plus the time
    # that the training thread needs to queue the next step, while the GPU idles. The middle mark of
    # a cycle is seen up to one step of the training thread late, which makes D larger by up to
    # twice that step: at most 0.60 ms here.
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=0.10e-3)
    s.train(3000)
    st = P.stats()
    assert st["step_ms"] == pytest.approx(0.42, rel=0.03)
    assert 0.40 <= st["cost_ms"] <= 0.40 + 0.60
    low, high = _bounds(0.40, 0.60, 0.42)            # 24 and 75
    assert low <= st["every"] <= high
    assert st["overhead_share"] <= 0.04 * 1.05
    item = P.status()[0]
    assert item["every"] == st["every"] and item["state"] == "running"
    assert item["step_ms"] == st["step_ms"] and item["last_cost_ms"] == st["cost_ms"]


def test_the_cadence_converges_on_a_cpu_bound_run(sim):
    # The training thread is the slower side: b is its step, and a pass costs the wait for the
    # step's GPU work and the pass, which the thread would otherwise have spent queuing.
    s = sim(cpu=1.0e-3, gpu=0.5e-3, pass_s=0.2e-3)
    s.train(2000)
    st = P.stats()
    assert st["step_ms"] == pytest.approx(1.0, rel=0.03)
    assert st["cost_ms"] == pytest.approx(0.5 + 0.2, rel=0.15)
    low, high = _bounds(0.7, 0.0, 1.0)               # 18 and 23.5
    assert low <= st["every"] <= high


def test_a_loop_that_waits_at_each_step_gets_sparse_marks(sim):
    # As with loss.item() at each step: torch's queue is empty at each checkpoint, a mark there
    # only adds a command buffer, and the facade times the steps with its own clock.
    s = sim(cpu=0.9e-3, gpu=0.4e-3, pass_s=0.5e-3, item=True)
    s.train(400)
    assert not P._STATE["dense"]
    records = s.records
    s.train(1600, first=400)
    st = P.stats()
    assert st["step_ms"] == pytest.approx(1.3, rel=0.03)
    assert st["cost_ms"] == pytest.approx(0.5, rel=0.15)
    low, high = _bounds(0.5, 0.0, 1.3)               # 10 and 13.5
    assert low <= st["every"] <= high
    passes = 1600 / st["every"]
    assert s.records - records <= 1600 / P.SPARSE + 2 * passes + 2


def test_a_step_without_a_pass_never_waits(sim):
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=0.10e-3)
    s.train(200)
    passes = []
    for k in range(200, 600):
        before, ran = s.sleeps, s.ran
        s.train(1, first=k)
        if s.ran == ran:
            assert s.sleeps == before, k              # no wait on a step without a pass
        else:
            passes.append(k)
    assert len(passes) >= 5
    assert all(b - a >= P.N_MIN for a, b in zip(passes, passes[1:]))


def test_the_cadence_follows_a_step_time_that_changes(sim):
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=0.10e-3)
    s.train(3000)
    before = P.stats()["every"]
    s.gpu = 0.84e-3                                  # the step time doubles
    s.train(4000, first=3000)
    st = P.stats()
    assert st["step_ms"] == pytest.approx(0.84, rel=0.03)
    low, high = _bounds(0.40, 0.60, 0.84)            # 12 and 38
    assert low <= st["every"] <= high and st["every"] < before


def test_one_slow_pass_does_not_move_the_cadence(sim):
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=0.10e-3)
    s.train(3000)
    before = P.stats()
    s.passes[s.ran] = 50e-3                          # the next pass takes 50 ms once
    runs = s.ran
    k = 3000
    while s.ran < runs + 3:                          # that pass and the two after it
        s.train(1, first=k)
        k += 1
    st = P.stats()
    assert st["every"] == before["every"]
    assert st["cost_ms"] == pytest.approx(before["cost_ms"], rel=0.2)


def test_the_cadence_moves_by_bounded_factors(sim):
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=2.0e-3)
    seen = []
    k = 0
    while len(seen) < 6:
        ran = s.ran
        s.train(1, first=k)
        k += 1
        if s.ran > ran and P._CAD.n is not None and (not seen or seen[-1] != P._CAD.n):
            seen.append(P._CAD.n)
    assert all(b <= math.ceil(a * P.GROW) for a, b in zip(seen, seen[1:])), seen


def test_a_pass_over_the_budget_at_the_ceiling_stops_the_programs(sim, monkeypatch):
    monkeypatch.setattr(P, "CEILING_S", 0.01)        # at most 24 steps of 0.42 ms between passes
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=2.0e-3)
    seen = []
    P.listeners.append(seen.append)
    s.train(3000)
    item = P.status()[0]
    assert item["state"] == "stopped" and item["error"]["code"] == "program.over_budget"
    assert re.search(r"more than the ceiling of 2[4-7] steps", item["error"]["message"])
    assert seen and seen[-1]["state"] == "stopped"
    assert not P.ACTIVE                              # training goes on without the program
    runs = s.ran
    s.train(100, first=3000)
    assert s.ran == runs


def test_a_declared_every_above_the_need_is_not_over_budget(sim, monkeypatch):
    monkeypatch.setattr(P, "CEILING_S", 0.01)
    s = sim(cpu=0.30e-3, gpu=0.42e-3, pass_s=2.0e-3)
    P._PROGS["sim"].every_min = 400                  # passes rarer than the 137 steps the cost needs
    s.train(6000)
    assert P.status()[0]["state"] == "running"


def test_step_marks_report_each_completion_in_order():
    from caliper._adapter import StepMarks

    class Ev:
        def __init__(self):
            self.done = False

        def record(self):
            self.done = False

        def query(self):
            return self.done

    m = StepMarks(Ev, size=4)
    m.record(0, 1.0)
    m.record(1, 2.0)
    assert m.poll(2.5) == [] and m.pending() == 2
    m._ring[0].done = True
    assert m.poll(3.0) == [(0, 2.5, 3.0)]            # pending at 2.5, complete at 3.0
    m.record(2, 3.5)
    m.record(3, 4.0)
    m.record(4, 4.5)                                 # the ring is full: mark 1 is dropped
    assert m.pending() == 3
    for e in m._ring:
        e.done = True
    assert [s for s, _, _ in m.poll(5.0)] == [2, 3, 4]
    m.clear()
    assert m.pending() == 0


def test_a_declaration_is_refused_before_any_pass(monkeypatch):
    monkeypatch.setitem(P._STATE, "model", Net())
    decl, src = _geometry(4)
    item = P.install("g", decl, src, scene=None)
    assert item["state"] == "refused" and item["error"]["code"] == "program.bad_target"
    item = P.install("g", decl, src, scene="s")       # a CPU tensor cannot be imported in place
    assert item["state"] == "refused" and item["error"]["code"] == "program.import_refused"
    decl, src = _geometry(4, source="live:nothing.here")
    item = P.install("g", decl, src, scene="s")
    assert item["state"] == "refused" and item["error"]["code"] == "program.unknown_input"
    decl["limits"]["time_ms"] = 60
    assert P.install("g", decl, src, scene="s")["error"]["code"] == "program.over_limits"
    assert set(item) == {"id", "kind", "state", "every", "every_min", "last_pass_ms", "last_cost_ms",
                         "step_ms", "overhead_share", "last_step", "runs", "memory_bytes",
                         "memory_limit_bytes", "error", "sha256"}
    P.remove("g")
    assert P.status() == [] and not P.ACTIVE


def test_the_runtime_loads_only_at_the_first_program():
    code = ("import sys, os, torch, caliper\n"
            "net = torch.nn.Linear(4, 2); opt = torch.optim.SGD(net.parameters(), lr=0.1)\n"
            "h = caliper.watch(net, opt)\n"
            "a = 'caliper._producer' in sys.modules or caliper._programs.loaded()\n"
            "h.remove(); caliper.close()\n"
            "s = caliper._server\n"
            "s is not None and s.close()\n"
            "print(a)\n")
    env = dict(os.environ, CALIPER_SESSION="0", CALIPER_NETIO="0", CALIPER_AGENT_OFFLINE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


@needs_device
def test_a_geometry_program_reads_the_weight_in_place_and_publishes_its_points(run):
    net, opt, calls, _ = run
    seen = []
    P.listeners.append(seen.append)
    decl, src = _geometry(4)
    item = P.install("norms", decl, src, scene="s", sha256="abc")
    assert item["state"] == "compiled", item["error"]
    assert P.loaded() and P.ACTIVE
    assert item["memory_bytes"] == 4 * 12 + 4 * 4 + 4 + 4 * 8 * 4
    assert item["memory_limit_bytes"] == 64 << 20 and item["sha256"] == "abc"
    _train(net, opt, P.WARMUP_STEPS)                 # T is measured; no pass yet
    assert P.status()[0]["runs"] == 0 and not calls
    w = net.fc.weight.detach().clone()               # the weight the pass reads: before the update
    _train(net, opt, 1, first=P.WARMUP_STEPS)
    st = P.status()[0]
    assert st["state"] == "running" and st["runs"] == 1 and st["last_step"] == P.WARMUP_STEPS
    assert st["every"] >= 1 and st["step_ms"] > 0      # D needs the step after a pass
    assert st["last_pass_ms"] is not None and st["last_pass_ms"] >= 0
    kind, path, pos, kw = calls[-1]
    assert kind == "points" and path == "programs/norms"
    assert pos.shape == (4, 3) and pos.device.type == "mps"
    pos = pos.cpu()
    rows = pos[:, 0].long()
    assert sorted(rows.tolist()) == [0, 1, 2, 3]
    torch.testing.assert_close(pos[:, 1], w.norm(dim=1).cpu()[rows], rtol=1e-5, atol=1e-6)
    assert torch.all(pos[:, 2] == P.WARMUP_STEPS)
    torch.testing.assert_close(kw["values"].cpu(), pos[:, 1])
    assert [s["state"] for s in seen] == ["compiled", "running"]
    # The optimizer changed the weight in place: the same import reads the new values.
    _train(net, opt, 4, first=P.WARMUP_STEPS + 1)
    st = P.status()[0]
    assert st["runs"] == 5 and st["every"] == 1
    stats = P.stats()
    assert stats["imports_in_place"] == 1 and stats["bytes_copied"] == 0
    assert stats["output_bytes_copied"] == 0 and stats["outputs_exported"] >= 2 * st["runs"]
    w = net.fc.weight.detach().cpu()
    last = calls[-1][2].cpu()
    assert float(last[0, 2]) == st["last_step"]
    P.remove("norms")
    assert ("clear", "programs/norms") == calls[-1]
    assert P.status() == [] and not P.ACTIVE and P.stats()["imports"] == 0


@needs_device
def test_the_cadence_measures_a_real_run(monkeypatch):
    """Without stand-ins for the cadence, a run on MPS measures b and D and sets N from them."""
    monkeypatch.setattr(caliper, "scene", lambda name="scene", **kw: FakeScene([]))
    net = Net().to("mps")
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    P.attach(net, opt)
    try:
        decl, src = _geometry(4)
        assert P.install("norms", decl, src, scene="s")["state"] == "compiled"
        _train(net, opt, 400)
        st = P.status()[0]
        assert st["state"] == "running" and st["runs"] >= 3
        assert st["step_ms"] > 0 and st["last_cost_ms"] is not None
        assert P.N_MIN <= st["every"] <= P._CAD.ceiling
        assert st["overhead_share"] == pytest.approx(
            st["last_cost_ms"] / (P._CAD.n * st["step_ms"]))
        assert P._STATE["marks"].pending() < P._STATE["marks"].SIZE
    finally:
        P.detach()
        if P._STATE["rt"] is not None:
            P._STATE["rt"].close()
            P._STATE["rt"] = None
        P._CAD.reset()


@needs_device
def test_an_input_declared_read_write_is_refused_and_never_runs(run):
    net, opt, calls, _ = run
    decl, src = _geometry(4, writes=True)
    item = P.install("bad", decl, src, scene="s")
    assert item["state"] == "refused" and item["error"]["code"] == "program.writes_input"
    _train(net, opt, P.WARMUP_STEPS + 3)
    assert P.status()[0]["runs"] == 0 and not calls and not P.ACTIVE


@needs_device
def test_a_pixels_program_reaches_the_live_view(run):
    net, opt, _, shown = run
    decl = {"inputs": [{"var": "w", "source": "live:fc.weight"}],
            "output": {"kind": "pixels", "width": 8, "height": 4, "format": "rgba8unorm"}}
    assert P.install("img", decl, PIXELS)["state"] == "compiled"
    _train(net, opt, P.WARMUP_STEPS)
    w = net.fc.weight.detach().cpu().flatten()       # the weight the pass reads: before the update
    _train(net, opt, 1, first=P.WARMUP_STEPS)
    name, img = shown[-1]
    assert name == "program.img" and img.shape == (4, 8, 4) and img.dtype == torch.uint8
    v = w.abs().clamp(0, 1).view(4, 8)
    torch.testing.assert_close(img[..., 0].cpu().float(), (v * 255).round(), atol=1, rtol=0)
    assert torch.all(img[..., 3] == 255)
    P.remove("img")
    assert shown[-1] == ("removed", "program.img")


@needs_device
def test_the_declared_every_is_a_floor(run):
    net, opt, _, _ = run
    decl, src = _geometry(4, every=5)
    P.install("slow", decl, src, scene="s")
    steps = []
    for i in range(P.WARMUP_STEPS + 30):
        before = P.status()[0]["runs"]
        _train(net, opt, 1, first=i)
        if P.status()[0]["runs"] > before:
            steps.append(i)
    assert len(steps) >= 3
    assert all(b - a >= 5 for a, b in zip(steps, steps[1:]))
    assert P.status()[0]["every_min"] == 5 and P.status()[0]["every"] >= 5


@needs_device
def test_a_program_over_its_time_limit_stops_and_training_continues(run):
    net, opt, _, _ = run
    seen = []
    P.listeners.append(seen.append)
    decl, src = _geometry(4, time_ms=0.01)           # the sandbox's least limit
    loop = "c < 8u; c++) { let v = w[row * 8u + c];"
    assert loop in src
    src = src.replace(loop, "c < 200000u; c++) { let v = w[row * 8u + c % 8u];")   # every pass is over it
    P.install("slow", decl, src, scene="s")
    for i in range(P.WARMUP_STEPS + 10):
        _train(net, opt, 1, first=i)
        if P.status()[0]["state"] == "stopped":
            break
    st = P.status()[0]
    assert st["state"] == "stopped" and st["runs"] == 3
    assert st["error"]["code"] == "program.over_time"
    assert seen[-1]["state"] == "stopped" and not P.ACTIVE
    _train(net, opt, 3, first=100)                   # no pass, no error
    assert P.status()[0]["runs"] == 3


@needs_device
def test_a_gradient_input_waits_for_the_first_backward_pass(run):
    net, opt, calls, _ = run
    decl, src = _geometry(4, source="live:fc.weight.grad")
    item = P.install("g", decl, src, scene="s")
    assert item["state"] == "waiting"
    _train(net, opt, P.WARMUP_STEPS + 1)
    st = P.status()[0]
    assert st["state"] == "running" and st["runs"] == 1 and calls


LINES = """@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> colors: array<u32>;
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(64)
fn segments(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= 4u) { return; }
    let i = atomicAdd(&count, 2u);
    for (var k = 0u; k < 2u; k++) {
        positions[(i + k) * 3u] = f32(row);
        positions[(i + k) * 3u + 1u] = w[row * 8u + k];
        positions[(i + k) * 3u + 2u] = 0.0;
        colors[i + k] = pack4x8unorm(vec4<f32>(1.0, 0.5, 0.0, 1.0));
    }
}
"""


@needs_device
def test_outputs_pass_the_argument_checks_of_the_real_scene_calls(monkeypatch):
    """The real caliper.scene checks the outputs' shapes and types, attached or not."""
    from caliper import _scene
    monkeypatch.setattr(P, "SHARE", 1e6)
    monkeypatch.setattr(P, "N_MIN", 1)
    net = Net().to("mps")
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    P.attach(net, opt)
    try:
        decl, src = _geometry(4)
        P.install("pts", decl, src, scene="progtest")
        decl = {"inputs": [{"var": "w", "source": "live:fc.weight"}],
                "output": {"kind": "geometry", "panel": "p", "part": "lines", "max_count": 8,
                           "arrays": ["positions", "colors"]}}
        P.install("seg", decl, LINES, scene="progtest")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _train(net, opt, P.WARMUP_STEPS + 2)
        assert not [w for w in caught if "program" in str(w.message)], [str(w.message) for w in caught]
        assert [s["runs"] for s in P.status()] == [2, 2]
        assert not any(k[0] == "publish" for k in P._WARNED)
    finally:
        P.detach()
        if P._STATE["rt"] is not None:
            P._STATE["rt"].close()
            P._STATE["rt"] = None
        P._CAD.reset()
        _scene.forget("progtest")


@needs_device
def test_programs_that_share_a_moving_gradient_both_keep_running(run):
    net, opt, calls, _ = run
    decl, src = _geometry(4, source="live:fc.weight.grad")
    assert P.install("a", decl, src, scene="s")["state"] == "waiting"
    assert P.install("b", decl, src, scene="s")["state"] == "waiting"
    _train(net, opt, P.WARMUP_STEPS + 4)             # zero_grad sets the gradient to None each step
    st = P.status()
    assert [s["state"] for s in st] == ["running", "running"], st
    assert [s["runs"] for s in st] == [4, 4]          # steps 8 to 11
    assert P.stats()["imports"] == 1                 # the older imports were released


class _TimesOut:
    """The real runtime, whose passes time out as a pass that never ends does."""

    def __init__(self, rt):
        self._rt = rt

    def run(self, *args, **kw):
        raise RuntimeError("program.timeout", "The pass did not finish in time.", 0, 0, 3)

    def __getattr__(self, name):
        return getattr(self._rt, name)


@needs_device
def test_activation_hooks_leave_the_model_when_the_last_program_stops(run):
    net, opt, _, _ = run
    decl, src = _geometry(4, source="live:fc.act")
    P.install("acts", decl, src, scene="s")
    assert set(P._ACT_HOOKS) == {"fc"} and len(net.fc._forward_hooks) == 1
    _train(net, opt, P.WARMUP_STEPS + 2)
    assert P.status()[0]["state"] == "running"
    P._STATE["rt"] = _TimesOut(P._STATE["rt"])
    _train(net, opt, 1, first=50)
    st = P.status()[0]
    assert st["state"] == "stopped" and st["error"]["code"] == "program.timeout"
    # The program stays installed, stopped; nothing reads the module's output any more.
    assert not P._ACT_HOOKS and not P._ACTS and not net.fc._forward_hooks


def test_activation_hooks_stay_while_another_program_needs_them(monkeypatch):
    net = Net()
    P.attach(net, None)
    try:
        for pid in ("a", "b"):
            p = P._Program(pid, {"inputs": [{"var": "w", "source": "live:fc.act"}],
                                 "output": {"kind": "pixels", "width": 1, "height": 1}}, "", None, "")
            with pytest.raises(P._Waiting):
                P._resolve("fc.act")                 # registers the hook, as a compile does
            p.state = "waiting"
            P._PROGS[pid] = p
        with P._LOCK:
            P._set_active()
        assert len(net.fc._forward_hooks) == 1
        with P._LOCK:
            P._PROGS["a"].state = "refused"
            P._set_active()
        assert len(net.fc._forward_hooks) == 1       # b still reads fc's output
        with P._LOCK:
            P._PROGS["b"].state = "stopped"
            P._set_active()
        assert not net.fc._forward_hooks and not P._ACT_HOOKS
    finally:
        P._PROGS.clear()
        P.detach()


SPIN = """@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> spin: array<f32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(1)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    var s = w[0];
    let n = u32(spin[0]);
    for (var k = 0u; k < n; k++) { s = s * 0.999999 + w[k % 8u]; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = s;
    positions[i * 3u + 1u] = 0.0;
    positions[i * 3u + 2u] = 0.0;
}
"""


@needs_device
def test_training_keeps_its_step_rate_after_a_pass_times_out(run):
    """A pass that runs far longer than the timeout: the step with the timeout returns after the
    timeout while the GPU still runs the pass, and the next steps keep the rate of the steps
    before it."""
    import time
    from caliper import _producer
    net, opt, _, _ = run

    def timed(first, steps):
        out = []
        for i in range(first, first + steps):
            t = time.perf_counter()
            _train(net, opt, 1, first=i)
            out.append(time.perf_counter() - t)
        return out

    timed(0, 3)                                      # MPS warms up
    baseline = timed(3, 5)                           # steps without a program
    net.register_buffer("spin", torch.tensor([1e5, 0.0, 0.0, 0.0], device="mps"))
    decl = {"inputs": [{"var": "w", "source": "live:fc.weight"}, {"var": "spin", "source": "live:spin"}],
            "output": {"kind": "geometry", "panel": "p", "part": "points", "max_count": 1,
                       "arrays": ["positions"]},
            "limits": {"time_ms": 50}, "workgroups": [1, 1, 1]}
    assert P.install("spin", decl, SPIN, scene="s")["state"] == "compiled"
    _train(net, opt, P.WARMUP_STEPS + 1, first=10)   # one pass of 100,000 iterations
    st = P.status()[0]
    assert st["state"] == "running" and st["last_pass_ms"]
    target_ms = 2000.0                               # 4 times the timeout of max(100, 10 x 50) ms
    net.spin[0] = min(2e9, 1e5 * target_ms / st["last_pass_ms"])
    torch.mps.synchronize()

    before = timed(20, 1)                            # the long pass, with the timeout
    reaping = _producer.reaping()
    st = P.status()[0]
    after = timed(21, 10)
    assert st["state"] == "stopped" and st["error"]["code"] == "program.timeout", st
    assert "cannot be stopped" in st["error"]["message"]
    assert reaping == 1, "the step returned only after the GPU ended the pass"
    assert before[0] < 1.0, before                   # the timeout of 0.5 s, not the pass of 2 s
    assert sorted(after)[5] < 3 * sorted(baseline)[2] + 0.02, (after, baseline)
    deadline = time.monotonic() + 30
    while _producer.reaping() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _producer.reaping() == 0
    P._reap()
    assert not P._ABANDONED
