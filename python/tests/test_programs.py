"""The facade of GPU programs (caliper._programs) over the producer runtime (caliper._producer).

The cadence formula and the refusals of a declaration run anywhere. The passes need an MPS device
and the native module, and they publish through stand-ins of ``caliper.scene`` and
``caliper.show``, thus no server and no viewer runs. A test that starts ``caliper.watch(model,
optimizer)`` runs in its own process with the shared-memory session off.
"""
from __future__ import annotations

import os
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


def test_the_cadence_follows_the_formula():
    c = P._Cadence()
    c.drained(0.0)
    now = 0.0
    for _ in range(10):
        now += 0.001                   # the CPU queues a step in 1 ms
        c.tick(now)
    # The first sync point waits 90 ms for the queued steps: T = 100 ms / 10 steps = 10 ms.
    # The pass takes 0.9 ms, and the GPU then idles for one queued step: C = 0.9 + 1 = 1.9 ms.
    c.update(now, now + 0.090, now + 0.0909)
    assert c.step_s() == pytest.approx(0.010)
    assert c.last_cost == pytest.approx(0.0019)
    assert c.n == 5                    # ceil(1.9 / (0.04 * 10))
    assert c.period(1) == 5 and c.period(8) == 10 and c.period(5) == 5
    assert c.period(10000) == 10000


def test_the_cadence_changes_n_only_beyond_a_quarter():
    c = P._Cadence()
    c.drained(0.0)
    now = 0.0

    def pass_step(steps, pass_s):
        nonlocal now
        for _ in range(steps):
            now += 0.010               # a loop that waits for the GPU at each step: T = 10 ms
            c.tick(now)
        c.update(now, now, now + pass_s)
        now += pass_s

    pass_step(5, 0.0019)
    assert c.n == 5
    pass_step(5, 0.0023)               # the median C is 2.1 ms: ceil(5.25) = 6, within 25 %
    assert c.n == 5
    for _ in range(8):
        pass_step(5, 0.0064)           # ceil(6.4 / 0.4) = 16: more than 25 % above 5
    assert c.n == 16


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
                         "step_ms", "last_step", "runs", "memory_bytes", "memory_limit_bytes",
                         "error", "sha256"}
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
    assert st["every"] >= 1 and st["step_ms"] > 0 and st["last_cost_ms"] > 0
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
    decl, src = _geometry(4, time_ms=1e-6)           # every pass is over the limit
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
