"""caliper.watch(model): hooks are registered, produce views, and come off.

caliper.watch(model, optimizer): the server, the names it publishes (the parameter lr, the series
loss and lr, the scene model), the optimizer step as the checkpoint (a new lr at the next step, a
pause before the weights change), lr under a learning rate scheduler, and the cost of the hooks.
The tests of the optimizer need no native core; the bounded timing runs only with
CALIPER_WATCH_TIMING=1 (it is a benchmark: run it in a slot on a quiet machine)."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper
from conftest import live_views_or_skip, mps_available

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


@pytest.fixture(scope="module")
def canvas():
    caliper.offscreen(256, 256)
    yield
    caliper.close()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.fc1 = nn.Linear(6 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def _step(net, device):
    x = torch.randn(4, 1, 16, 16, device=device)
    loss = net(x).square().mean()
    loss.backward()


def _hook_count(net):
    return sum(len(m._forward_hooks) for m in net.modules())


@pytest.mark.usefixtures("canvas")
@pytest.mark.parametrize("device", DEVICES)
def test_watch_registers_produces_and_removes(device):
    live_views_or_skip()
    torch.manual_seed(0)
    net = Net().to(device)
    h = caliper.watch(net, activations=True, gradients=True)
    try:
        assert h.modules == ["conv1", "fc1", "fc2"]
        assert set(h.views) == {"conv1.act", "fc1.act", "fc2.act",
                                "conv1.weight.grad", "fc1.weight.grad", "fc2.weight.grad"}
        assert _hook_count(net) == 3 and len(h) == 6
        _step(net, device)
        conv = caliper.view_info("conv1.act")              # (4,6,16,16) -> item 0 mosaic 3x2
        assert (conv["width"], conv["height"]) == (48, 32) and conv["updates"] == 1
        assert (caliper.view_info("fc1.act")["width"], caliper.view_info("fc1.act")["height"]) == (16, 4)
        g = caliper.view_info("conv1.weight.grad")         # (6,1,3,3) -> (6, 9)
        assert (g["width"], g["height"]) == (9, 6)
        assert caliper.view_info("fc1.weight.grad")["imported"] is True
    finally:
        h.remove()
    assert _hook_count(net) == 0 and not h.active
    before = caliper.view_info("conv1.act")["updates"]
    _step(net, device)
    assert caliper.view_info("conv1.act")["updates"] == before
    for v in h.views:
        caliper.remove(v)


@pytest.mark.usefixtures("canvas")
def test_watch_include_and_every():
    live_views_or_skip()
    net = Net()
    with caliper.watch(net, include=["conv*"], every=2) as h:
        assert h.modules == ["conv1"] and h.views == ["conv1.act"]
        for _ in range(4):
            net(torch.randn(2, 1, 16, 16))
        assert caliper.view_info("conv1.act")["updates"] == 2     # every 2nd forward
    assert _hook_count(net) == 0
    pick = caliper.watch(net, include=lambda name, m: isinstance(m, nn.Linear), activations=True)
    assert pick.modules == ["fc1", "fc2"]
    pick.remove()
    caliper.remove("conv1.act")


@pytest.mark.usefixtures("canvas")
def test_watch_is_free_when_nothing_is_attached():
    net = Net()
    caliper.close()                       # no transport at all: the hooks call a no-op
    try:
        import os
        os.environ["CALIPER_SESSION"] = "0"
        caliper._auto_tried = False
        with caliper.watch(net, gradients=True):
            _step(net, "cpu")
        assert caliper._native_core() is None
    finally:
        os.environ.pop("CALIPER_SESSION", None)
        caliper.offscreen(256, 256)


# ---------------------------------------------------------------------------
# caliper.watch(model, optimizer)
# ---------------------------------------------------------------------------
import ast                                                     # noqa: E402
import asyncio                                                 # noqa: E402
import os                                                      # noqa: E402
import pathlib                                                 # noqa: E402
import statistics                                              # noqa: E402
import threading                                               # noqa: E402
import time                                                    # noqa: E402

from caliper import _controls as C                             # noqa: E402
from caliper import _scene                                     # noqa: E402
from caliper import _watch_run as R                            # noqa: E402
from caliper import _wire as w                                 # noqa: E402

try:
    from caliper import _netio  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False


@pytest.fixture
def run_env(monkeypatch):
    """No route open, a fresh Controls, the server and the scene closed after the test."""
    caliper.close()
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1" if NATIVE else "0")
    c = C.Controls()
    monkeypatch.setattr(C, "CONTROLS", c)
    monkeypatch.setattr(caliper, "_CTL", c)
    monkeypatch.setattr(C, "WAIT_POLL_S", 0.01)
    monkeypatch.setattr(caliper, "_auto_tried", False)
    yield c
    with c.lock:
        c.state = C.RUN_RUNNING
        c.cond.notify_all()
    s = caliper._server
    if s is not None:
        s.close()
    caliper.close()
    _scene.forget(R.SCENE_NAME)


class Mlp(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.fc1 = nn.Linear(8, width)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(width, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _train_step(net, opt, x=None):
    x = torch.randn(16, 8) if x is None else x
    loss = net(x).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.detach()


def _used_lr(opt):
    """The rates that each step used: a pre-hook registered after watch()'s runs after it."""
    used = []
    opt.register_step_pre_hook(
        lambda o, a, k: used.append([float(g["lr"]) for g in o.param_groups]))
    return used


def test_watch_without_optimizer_starts_nothing(run_env):
    net = Mlp()
    orig = torch.autograd.backward
    with caliper.watch(net, activations=False) as h:
        assert caliper._server is None and "lr" not in run_env.params
        assert torch.autograd.backward is orig and h._run is None and h.modules == []
    assert R.SCENE_NAME not in _scene.SCENES


def test_watch_with_optimizer_publishes_the_names(run_env, monkeypatch):
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    series = []
    real = caliper.series
    monkeypatch.setattr(caliper, "series", lambda v, name, rate=None, **kw: (
        series.append((name, v)), real(v, name, rate=rate, **kw))[1])
    h = caliper.watch(net, opt)
    try:
        assert caliper._server is not None and caliper._auto_tried
        assert caliper._session_route is None
        lr = run_env.params["lr"]
        assert (lr.value, lr.min, lr.max, lr.log) == (0.05, 0.05 / 1000, 0.05 * 10, True)
        assert h.modules == ["fc1", "act", "fc2"] and R.SCENE_NAME in _scene.SCENES
        assert len(opt._optimizer_step_pre_hooks) == 1 and len(opt._optimizer_step_post_hooks) == 1
        losses = [_train_step(net, opt) for _ in range(3)]
        assert [n for n, _ in series] == ["loss", "lr"] * 3
        assert all(torch.equal(v, l) for (n, v), l in zip(series[0::2], losses))
        assert [v for n, v in series[1::2]] == [0.05] * 3
        assert run_env.n == 3                         # caliper.step() once per optimizer step
    finally:
        h.remove()
    assert torch.autograd.backward.__name__ == "backward"
    assert not getattr(torch.autograd.backward, "_caliper_loss_tap", False)
    assert "lr" not in run_env.params and not opt._optimizer_step_pre_hooks


def test_the_loss_of_a_closure(run_env, monkeypatch):
    net = Mlp()
    opt = torch.optim.LBFGS(net.parameters(), lr=0.1, max_iter=2)
    got = []
    monkeypatch.setattr(caliper, "series", lambda v, name, rate=None, **kw: got.append((name, v)))
    x = torch.randn(16, 8)

    def closure():
        opt.zero_grad()
        loss = net(x).square().mean()
        loss.backward()
        return loss
    with caliper.watch(net, opt):
        opt.step(closure)
    assert [n for n, _ in got] == ["lr", "loss"] and got[1][1].ndim == 0


def test_the_scene_has_one_box_per_module(run_env, monkeypatch):
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    calls = []
    monkeypatch.setattr(_scene.Scene, "boxes", lambda self, path, c, hs, **kw: calls.append(
        (path, c, hs, kw)) or True)
    watched = [False]
    monkeypatch.setattr(_scene.Scene, "_watched", lambda self: watched[0])
    with caliper.watch(net, opt) as h:
        path, c, hs, kw = calls[0]                    # the layout, grey, lists the scene
        assert path == R.ENTITY and kw["labels"] == h.modules == ["fc1", "act", "fc2"]
        assert c[:, 0].tolist() == [0, 0, 0] and c[:, 1].tolist() == [0, 1.25, 2.5]
        assert hs[0, 0] > hs[2, 0] > hs[1, 0] == R._SIDE_MIN / 2   # 288 > 66 > 0 parameters
        for _ in range(3):
            _train_step(net, opt)
        assert len(calls) == 1 and h._run.means == [None] * 3     # nobody watches: no work
        watched[0] = True
        h._run._next = 0.0
        _train_step(net, opt)                         # arms the colors
        _train_step(net, opt)                         # records them, then sends
        assert len(calls) == 2 and calls[1][3]["colors"].shape == (3, 3)
        assert h._run.means == [None] * 3


def test_lr_applies_at_the_next_step_and_a_pause_holds_before_the_weights_change(run_env):
    from test_net_controls import Viewer, run, until
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    h = caliper.watch(net, opt)
    server = caliper._server
    used = _used_lr(opt)
    after = []                                        # the weights after each step
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            _train_step(net, opt)
            after.append(net.fc1.weight.detach().clone())
            time.sleep(0.002)
    t = threading.Thread(target=loop, daemon=True)

    async def go():
        async with Viewer(server) as v:
            t.start()
            await v.state(lambda m: m.run["state"] == "running")
            r = await v.control("set", name="lr", value=0.01)
            assert r.status == "ok"
            await v.state(lambda m: m.params[0]["value"] == 0.01 and "pending" not in m.params[0])
            n = len(used)
            await until(lambda: len(used) > n)
            assert used[n][0] == 0.01 and opt.param_groups[0]["lr"] == 0.01
            assert (await v.control("pause")).status == "ok"
            await v.state(lambda m: m.run["state"] == "paused")
            k = len(after)
            await asyncio.sleep(0.1)
            assert len(after) == k and torch.equal(net.fc1.weight, after[-1])
            assert net.fc1.weight.grad is not None    # backward ran, the step did not
            assert (await v.control("resume")).status == "ok"
            await until(lambda: len(after) > k + 3)
    try:
        run(go())
    finally:
        stop.set()
        caliper._CTL._release()
        t.join(5)
        h.remove()


@pytest.mark.parametrize("make", [
    lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda e: 1.0 / (1 + e)),
    lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5),
    lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=20),
], ids=["lambda", "step", "onecycle"])
def test_a_scheduler_neither_undoes_nor_compounds_lr(run_env, make):
    net = Mlp()
    opt = torch.optim.SGD([{"params": net.fc1.parameters(), "lr": 0.04},
                           {"params": net.fc2.parameters(), "lr": 0.02}], lr=0.04)
    ref_net = Mlp()
    ref_opt = torch.optim.SGD([{"params": ref_net.fc1.parameters(), "lr": 0.04},
                               {"params": ref_net.fc2.parameters(), "lr": 0.02}], lr=0.04)
    sched, ref_sched = make(opt), make(ref_opt)
    h = caliper.watch(net, opt, scene=False)
    used = _used_lr(opt)
    lr = run_env.params["lr"]
    try:
        for i in range(12):
            ref = [g["lr"] for g in ref_opt.param_groups]     # the schedule at this step
            if i == 4:
                lr.set(ref[0] / 4)                            # a quarter of this step's rate
            _train_step(net, opt)
            sched.step()
            f = 0.25 if i >= 4 else 1.0
            assert used[i] == pytest.approx([r * f for r in ref]), i
            ref_opt.step()
            ref_sched.step()
            assert sched.get_last_lr() == pytest.approx(ref_sched.get_last_lr()), i
    finally:
        h.remove()


def test_lr_without_a_scheduler_stays_in_the_groups(run_env):
    net = Mlp()
    opt = torch.optim.SGD([{"params": net.fc1.parameters(), "lr": 0.04},
                           {"params": net.fc2.parameters(), "lr": 0.02}])
    with caliper.watch(net, opt, scene=False):
        used = _used_lr(opt)
        lr = run_env.params["lr"]
        _train_step(net, opt)
        lr.set(0.01)
        _train_step(net, opt)
        assert used[-1] == [0.01, 0.005] and [g["lr"] for g in opt.param_groups] == [0.01, 0.005]
        opt.param_groups[0]["lr"] = 0.002              # the script changes group 0 only
        lr.set(0.01)                                   # the same value again still applies
        _train_step(net, opt)
        assert used[-1] == [0.01, 0.025]               # both groups move by the factor 5


def test_an_error_in_the_hook_never_stops_the_step(run_env, monkeypatch):
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    calls = []

    def broken(v, name, rate=None, **kw):
        calls.append(name)
        raise ValueError("no series")
    monkeypatch.setattr(caliper, "series", broken)
    h = caliper.watch(net, opt)
    monkeypatch.setattr(h._run, "_scene_tick", lambda: (_ for _ in ()).throw(KeyError("x")))
    try:
        with pytest.warns(RuntimeWarning, match="step hook raised ValueError") as rec:
            for _ in range(3):
                before = net.fc1.weight.detach().clone()
                _train_step(net, opt)
                assert not torch.equal(before, net.fc1.weight)     # the step ran
        assert len([r for r in rec if "step hook" in str(r.message)]) == 1
        assert "lr" in calls and run_env.n == 3
    finally:
        h.remove()


def test_a_step_that_raised_leaves_no_scaled_rates(run_env):
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.04)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    h = caliper.watch(net, opt, scene=False)
    raised = []

    def fail_once(o, a, k):
        if not raised:
            raised.append(1)
            raise RuntimeError("the step fails")
    opt.register_step_pre_hook(fail_once)     # runs after watch()'s pre-hook
    used = _used_lr(opt)
    try:
        run_env.params["lr"].set(0.01)
        with pytest.raises(RuntimeError, match="the step fails"):
            _train_step(net, opt)                  # the rates stay scaled, no post-hook ran
        _train_step(net, opt)
        sched.step()
        assert used == [pytest.approx([0.01])]     # scaled once, not twice
        assert sched.get_last_lr() == pytest.approx([0.04])
    finally:
        h.remove()


def test_watch_run_code_never_reads_back():
    """The hooks of every step read nothing back from the device (as caliper._watch)."""
    from test_dataview import _bad_calls
    src = pathlib.Path(R.__file__).read_text()
    assert _bad_calls(ast.parse(src)) == []


def test_the_checkpoint_stays_on_the_fast_path(run_env, monkeypatch):
    net = Mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    seen = []
    monkeypatch.setattr(C.Controls, "checkpoint",
                        lambda self, real=C.Controls.checkpoint: (seen.append(1), real(self)))
    with caliper.watch(net, opt):
        for _ in range(5):
            _train_step(net, opt)
    assert len(seen) == 1 and run_env.busy is False   # only the first step takes the slow path


VIEWER = """
import sys
from websockets.sync.client import connect
from caliper import _wire as w
with connect(sys.argv[1], max_size=None, compression=None) as ws:
    ws.send(w.encode(w.Hello("viewer", [w.CAP_CONTROLS, w.CAP_SCENES],
                             {"name": "timing", "version": "0"})))
    w.decode(ws.recv())
    for sub, view in enumerate(sys.argv[2:], 1):
        ws.send(w.encode(w.Subscribe(sub, view, 100 + sub)))
    print("subscribed", flush=True)
    try:
        while True:
            ws.recv()
    except Exception:
        pass
"""


def _untimed(net, opt, x, y):
    for _ in range(50):
        loss = (net(x) - y).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@pytest.mark.skipif(os.environ.get("CALIPER_WATCH_TIMING") != "1",
                    reason="a bounded timing: CALIPER_WATCH_TIMING=1 in a slot on a quiet machine")
def test_the_throughput_cost_of_the_hook(run_env):
    """Training with watch(model, optimizer) and a viewer of loss, lr and the scene in another
    process runs at 95 % or more of its speed without the hook: the median of five runs of each,
    interleaved, each well under 60 seconds."""
    import subprocess
    import sys
    torch.set_num_threads(4)
    steps = int(os.environ.get("CALIPER_WATCH_STEPS", "3000"))
    x, y = torch.randn(256, 64), torch.randn(256, 10)

    def make():
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, 10))
        return net, torch.optim.Adam(net.parameters(), lr=1e-3)

    def train(net, opt):
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = (net(x) - y).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return time.perf_counter() - t0

    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(caliper.__file__).parent.parent))
    train(*make())                                    # warm up
    plain, watched = [], []
    for _ in range(5):
        plain.append(train(*make()))
        net, opt = make()
        h = caliper.watch(net, opt)
        viewer = subprocess.Popen([sys.executable, "-c", VIEWER, caliper._server.stream_url,
                                   "loss", "lr", R.SCENE_NAME], env=env, stdout=subprocess.PIPE,
                                  text=True)
        try:
            assert viewer.stdout.readline().strip() == "subscribed"
            end = time.monotonic() + 10
            while not h._run.scene._watched():        # untimed steps until the viewer watches
                assert time.monotonic() < end
                _untimed(net, opt, x, y)
            watched.append(train(net, opt))
        finally:
            h.remove()
            caliper._server.close()
            viewer.wait(10)
            _scene.forget(R.SCENE_NAME)
        assert max(plain[-1], watched[-1]) < 60
    p, q = statistics.median(plain), statistics.median(watched)
    print(f"\nwatch throughput: {steps} steps, plain {p:.3f} s, watched {q:.3f} s, "
          f"speed {p / q:.3f}, load {os.getloadavg()[0]:.2f}; runs plain "
          f"{[round(v, 3) for v in plain]} watched {[round(v, 3) for v in watched]}")
    assert p / q >= 0.95
