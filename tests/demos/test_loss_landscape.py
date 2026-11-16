"""Demo 0, the loss landscape (demos/loss_landscape): the acceptance rows of
design/demos.md that need no pixels.

The notebook's own code cells run in this process on the CPU, with fixed
seeds and deterministic algorithms, and a test viewer on a real loopback
connection to the notebook's ``caliper.serve()`` decodes every scene batch:

* the path's plane, 300 steps with ``every`` at 100, so that refits come at
  steps 40, 100 and 200 (a point every 10th step, a snapshot every 20): the
  scene's entities and the frame at its root, nothing drawn before step 40,
  the mesh's shape, the centre vertex at half the log10 of the loss computed
  separately, the path's last point at the current point whenever the path
  is sent (each 10th step and each refit), orthonormal directions, the first
  eleven points of the path after the refit at step 200 against the
  snapshots' projections in float64, and ``plane share``, one value per
  training step: NaN before the first plane, then the last refit's share;
* random directions, 100 steps with ``every`` at 50: the filter-normalised
  rows, zero biases and normalisation entries, the centre vertex, a path of
  one point every 10th step that is never redrawn, and ``plane share`` NaN
  at every step;
* the controls: ``span``, ``directions`` and ``every`` set while paused take
  effect at the next surface, and ``surface now`` computes one while paused;
* two fixtures of ``Landscape`` alone: snapshots along a known curve whose
  second component outgrows its first, for the order and sign of the matched
  directions, and the thinning rule with room for 8 snapshots and for 256.

    PYTHONPATH=python .venv/bin/python -m pytest tests/demos

``test_capture``, the capture row, runs where CALIPER_TEST_CALIPER_EXE names the
Caliper app and Metal draws it."""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

import caliper  # noqa: E402
from caliper import _controls as C  # noqa: E402
from caliper import _wire as w  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demos" / "loss_landscape"
NOTEBOOK = DEMO / "loss_landscape.ipynb"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
from landscape import Landscape  # noqa: E402

F = torch.nn.functional
GRID, CENTRE = 41, 41 * 41 // 2              # vertex 840 is the current weights
# The plane share at step 200 on the first verified run (CPU, the seeds of
# the notebook, torch 2.15); a later run must reach it within 0.05.
PINNED_SHARE_200 = 0.953


# ---------------------------------------------------------------------------
# the notebook, run in this process
# ---------------------------------------------------------------------------
@pytest.fixture
def notebook(monkeypatch):
    """A function that runs the notebook's code cells, all but the training
    run, the random-directions cell and the close, on the CPU, and returns
    their namespace; closes the server and removes what it registered."""
    made = []
    monkeypatch.setenv("CALIPER_SESSION", "0")
    ctl = C.Controls()                                  # a fresh run state for each test
    monkeypatch.setattr(C, "CONTROLS", ctl)
    monkeypatch.setattr(caliper, "_CTL", ctl)
    monkeypatch.setattr(C, "WAIT_POLL_S", 0.01)
    real = torch.backends.mps.is_available
    monkeypatch.setattr(torch.backends.mps, "is_available", functools.wraps(real)(lambda: False))
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)

    def run(steps):
        monkeypatch.setenv("CALIPER_NOTEBOOK_STEPS", str(steps))
        ns = {"__name__": "loss_landscape"}
        made.append(ns)
        for cell in json.loads(NOTEBOOK.read_text())["cells"]:
            if cell["cell_type"] != "code" or cell["id"] in ("run", "random", "close"):
                continue
            if cell["id"] == "widgets" and importlib.util.find_spec("anywidget") is None:
                continue
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec"), ns)
        assert ns["device"].type == "cpu"
        return ns

    yield run
    with ctl.lock:                                      # never leave a loop paused
        ctl.state = C.RUN_RUNNING
        ctl.cond.notify_all()
    for ns in made:
        for name in ("every", "span", "directions", "surface_now"):
            if name in ns:
                ns[name].remove()
        caliper.remove("landscape")
        caliper.remove("plane share")
        ns["server"].close()
    torch.use_deterministic_algorithms(deterministic)


def plain_loss(ns) -> float:
    """The landscape batch's loss at the model's current weights, by a plain
    forward pass, computed here rather than by Landscape."""
    with torch.no_grad():
        logits = ns["model"](ns["eval_x"][:128])[0]
        return F.cross_entropy(logits, ns["eval_y"][:128]).item()


def loss_at(ns, flat) -> float:
    """The landscape batch's loss with the perturbed weights set to ``flat``
    and the others as the model holds them, by functional_call, computed here
    rather than by Landscape."""
    land = ns["land"]
    params = dict(zip(land.names, land.split(flat)))
    with torch.no_grad():
        logits = torch.func.functional_call(ns["model"], params, (ns["eval_x"][:128],))[0]
        return F.cross_entropy(logits, ns["eval_y"][:128]).item()


def far_corner(ns) -> float:
    """log10 of the loss at vertex GRID - 1 of the surface just computed: the
    first row (b at -1) and the last column (a at +1), theta + h d1 - h d2,
    where theta is the weights the surface was computed at and h its
    half-width. The grid's centre and vertex 0 lie on its diagonal, so they
    cannot tell a surface from its mirror image; this vertex can."""
    land = ns["land"]
    theta, d, h = land.flat(ns["model"]), land.directions, land.half_width
    return math.log10(loss_at(ns, theta + h * d[0] - h * d[1]))


def clip(log_loss: float) -> float:
    return min(max(log_loss, -4.0), 2.0)


class Recorder:
    """Wraps the notebook's show_surface, show_path and land.point to record,
    at each surface, the step, the separately computed loss, the loss at the
    surface's vertex GRID - 1 and the plane, and the steps at which anything
    was drawn."""

    def __init__(self, ns):
        self.ns = ns
        self.surfaces: list[dict] = []
        self.losses: dict[int, float] = {}               # snapshot steps -> plain loss
        self.thetas: dict = {}                           # point steps -> the weights
        self.drawn: list[int] = []
        land = ns["land"]
        point, surface, path = land.point, ns["show_surface"], ns["show_path"]

        def recording_point(model, step):
            p = point(model, step)
            if step % land.snapshot_every == 0:
                self.losses[step] = plain_loss(ns)
            if p is not None:
                self.thetas[step] = land.flat(model).clone()
            return p

        def recording_surface():
            entry = {"step": land.step, "loss": plain_loss(ns)}
            surface()
            entry.update(far_corner=far_corner(ns), mode=land.mode, span=land.span, half_width=float(land.half_width),
                         centre=land.centre.clone(), directions=land.directions.clone(),
                         share=None if land.share is None else float(land.share),
                         snapshots=len(land.snapshot_steps), path_length=land.path_length)
            self.surfaces.append(entry)

        def recording_path(rate=10):
            self.drawn.append(land.step)
            return path(rate)

        land.point = recording_point
        ns["show_surface"] = recording_surface
        ns["show_path"] = recording_path


# ---------------------------------------------------------------------------
# the test viewer
# ---------------------------------------------------------------------------
class Viewer:
    """A viewer on a thread of its own, on a real loopback connection: it
    announces scenes and controls, subscribes to the scene ``landscape`` and,
    once it is listed, the series ``plane share``, and keeps every message
    decodes."""

    def __init__(self, server):
        self.url = server.stream_url
        self.msgs: list = []
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.subscribed: set = set()
        self.rid = 1000

    def __enter__(self):
        self.thread.start()
        self.call(self._open())
        self.wait(lambda: any(isinstance(m, w.SceneClear) and m.reset for m in self.all()),
                  what="the scene's state")
        return self

    def __exit__(self, *exc):
        self.call(self.ws.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)

    def call(self, coro, timeout=30.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _open(self):
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None,
                                close_timeout=1)
        await self.ws.send(w.encode(w.Hello("viewer", [w.CAP_SCENES, w.CAP_CONTROLS],
                                            {"name": "test", "version": "0"})))
        self.reader = asyncio.ensure_future(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = w.decode(raw)
            with self.lock:
                self.msgs.append(m)
            if isinstance(m, w.Views):
                names = {v["name"] for v in m.views}
                for sub, view in ((1, "landscape"), (2, "plane share")):
                    if view in names and view not in self.subscribed:
                        self.subscribed.add(view)
                        await self.ws.send(w.encode(w.Subscribe(sub, view, 100 + sub, None, None)))

    def all(self) -> list:
        with self.lock:
            return list(self.msgs)

    def wait(self, pred, timeout=60.0, what="a condition"):
        end = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > end:
                raise TimeoutError(f"the viewer waited {timeout} s for {what}")
            time.sleep(0.01)

    def updates(self, path, part) -> list:
        return [m for m in self.all() if isinstance(m, w.SceneUpdate) and m.sub == 1
                and m.path == path and m.part == part]

    def series(self) -> list[float]:
        out = []
        for m in self.all():
            if isinstance(m, w.Series) and m.sub == 2:
                vals = m.values.array().tolist()
                out[m.start:m.start + len(vals)] = vals
        return out

    def control(self, op, done=False, **kw):
        """Send a CONTROL and return its reply; with ``done``, the second reply
        of an action, after it ran."""
        self.rid += 1
        rid = self.rid
        self.call(self.ws.send(w.encode(w.Control(rid, op, **kw))))
        replies = lambda: [m for m in self.all()  # noqa: E731
                           if isinstance(m, w.ControlReply) and m.request_id == rid]
        self.wait(lambda: len(replies()) >= (2 if done else 1), what=f"the reply to {op}")
        return replies()[-1]

    def params(self) -> dict:
        """The values of the three landscape parameters in the newest CONTROLS
        with nothing pending."""
        states = [m for m in self.all() if isinstance(m, w.Controls)]
        if not states:
            return {}
        return {p["name"]: p["value"] for p in states[-1].params
                if p["name"] in ("span", "directions", "every") and "pending" not in p}

    def run_state(self):
        states = [m for m in self.all() if isinstance(m, w.Controls)]
        return states[-1].run if states else None


def arr(m, name):
    return m.arrays[name].array()


def check_mesh(m):
    pos, idx, vals = arr(m, "positions"), arr(m, "indices"), arr(m, "values")
    assert pos.shape == (GRID * GRID, 3) and vals.shape == (GRID * GRID,)
    assert idx.shape == (2 * (GRID - 1) ** 2, 3) and idx.dtype == np.int32
    assert idx.min() >= 0 and idx.max() < GRID * GRID
    assert (m.params["vmin"], m.params["vmax"]) == (-4.0, 2.0) and m.params["colormap"] == 0
    np.testing.assert_allclose(pos[:, 2], 0.5 * vals, rtol=0, atol=1e-6)
    return pos, vals


def path_pairs(v):
    """(path, current) batches logged one after the other, both received."""
    current = {m.seq: m for m in v.updates("current", "points")}
    return [(p, current[p.seq + 1]) for p in v.updates("path", "lines") if p.seq + 1 in current]


def check_path_ends_at_current(v):
    pairs = path_pairs(v)
    assert len(pairs) >= 3
    for p, c in pairs:
        pts = arr(p, "positions")
        assert arr(p, "strip_lengths").tolist() == [len(pts)]
        assert p.params["color"] == [235, 235, 235, 255] and c.params["color"] == [255, 80, 80, 255]
        np.testing.assert_array_equal(pts[-1], arr(c, "positions")[0])     # exactly


# ---------------------------------------------------------------------------
# the path's plane
# ---------------------------------------------------------------------------
def test_path_plane(notebook):
    ns = notebook(300)
    land = ns["land"]
    ns["every"].set(100)
    rec = Recorder(ns)
    with Viewer(ns["server"]) as v:
        v.wait(lambda: any(isinstance(m, w.SceneUpdate) and m.state for m in v.all()),
               what="the static frame")
        state = [m for m in v.all() if isinstance(m, w.SceneUpdate) and m.state]
        assert [(m.path, m.part, m.static) for m in state] == [("", "lines", True)]
        assert arr(state[0], "positions").shape == (2 * (12 + 18), 3)       # box edges, floor grid
        ns["train"](300)
        v.wait(lambda: any(len(arr(m, "positions")) == 20 for m in v.updates("path", "lines")),
               what="the last path")
        v.wait(lambda: len(v.series()) == 300, what="a plane share for each step")

    # refits at 40, 100 and 200; nothing drawn before step 40, then only at
    # points (every 10th step) and refits, and once more after the last step
    assert [s["step"] for s in rec.surfaces] == [40, 100, 200]
    assert land.refits == [(40, "path"), (100, "path"), (200, "path")] and land.method == "vmap"
    assert min(rec.drawn) >= 40 and rec.drawn[-1] == 299
    assert all(step % 10 == 0 for step in rec.drawn[:-1])
    received = {(m.path, m.part) for m in v.all() if isinstance(m, w.SceneUpdate) and m.sub == 1}
    assert received == {("", "lines"), ("surface", "mesh"), ("path", "lines"),
                        ("current", "points")}

    # the mesh, its centre vertex at the current weights
    meshes = v.updates("surface", "mesh")
    assert len(meshes) == 3
    for m, s in zip(meshes, rec.surfaces):
        pos, _ = check_mesh(m)
        assert abs(pos[CENTRE, 2] - 0.5 * math.log10(s["loss"])) < 1e-4
        assert pos[CENTRE, 0] == 0.0 and pos[CENTRE, 1] == 0.0               # the plane's origin
        np.testing.assert_array_equal(pos[:GRID, 0], land.offsets.numpy())    # x, y from -1 to 1
        np.testing.assert_array_equal(pos[::GRID, 1], land.offsets.numpy())   # y down the rows
        # the orientation: vertex GRID - 1, at (1, -1), is theta + h d1 - h d2
        assert abs(pos[GRID - 1, 2] - 0.5 * clip(s["far_corner"])) < 1e-4, (pos[GRID - 1, 2], s["far_corner"])

    check_path_ends_at_current(v)

    # orthonormal directions
    for s in rec.surfaces:
        d = s["directions"].double()
        np.testing.assert_allclose((d @ d.T).numpy(), np.eye(2), rtol=0, atol=1e-5)

    # after the refit at step 200: one point per kept snapshot, then one every
    # 10th step (210 to 290)
    last = rec.surfaces[-1]
    assert last["snapshots"] == 11 and land.snapshot_steps[:11] == list(range(0, 201, 20))
    final = arr(v.updates("path", "lines")[-1], "positions")
    assert len(final) == 11 + 9
    snaps = land.snapshots[:11].double()
    theta_r, plane = last["centre"].double(), last["directions"].double()
    coords = (snaps - theta_r) @ plane.T                                   # float64, on the CPU
    half = max(1.5 * 1.0 * coords.abs().max().item(), 1e-3 * theta_r.norm().item())
    np.testing.assert_allclose(final[:11, :2], (coords / half).numpy(), rtol=0, atol=1e-4)
    heights = [0.5 * math.log10(rec.losses[s]) for s in range(0, 201, 20)]
    np.testing.assert_allclose(final[:11, 2], heights, rtol=0, atol=1e-4)
    assert abs(last["half_width"] - half) <= 1e-5 * half

    # the plane holds nearly as much of the path as the exact top two components
    m64 = snaps - theta_r
    top = torch.linalg.eigvalsh(m64 @ m64.T)[-2:].sum() / (m64 * m64).sum()
    assert abs(last["share"] - top.item()) < 1e-3

    # plane share: one value per training step, so that its index is the step:
    # NaN (a gap) before the first plane, then from each refit's step on that
    # refit's share, between 0 and 1
    shares = np.array(v.series())
    assert len(shares) == 300
    assert np.isnan(shares[:40]).all()
    bounds = [s["step"] for s in rec.surfaces] + [300]
    for s, a, b in zip(rec.surfaces, bounds, bounds[1:]):
        np.testing.assert_allclose(shares[a:b], s["share"], rtol=0, atol=1e-6)
        assert 0.0 <= s["share"] <= 1.0
    assert shares[200] >= PINNED_SHARE_200 - 0.05, shares[200]


# ---------------------------------------------------------------------------
# random directions
# ---------------------------------------------------------------------------
def test_random_directions(notebook):
    ns = notebook(100)
    land, model = ns["land"], ns["model"]
    step0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    ns["every"].set(50)
    ns["directions"].set("random")
    rec = Recorder(ns)
    with Viewer(ns["server"]) as v:
        ns["train"](100)
        v.wait(lambda: any(len(arr(m, "positions")) == 10 for m in v.updates("path", "lines")),
               what="the last path")
        v.wait(lambda: len(v.series()) == 100, what="a plane share for each step")

    assert [(s["step"], s["mode"]) for s in rec.surfaces] == [(0, "random"), (50, "random")]
    # each row of each direction has the norm of the same row of the weights at step 0
    for k in range(2):
        parts = land.as_parameters(land.random_directions[k], model)
        for name, d in parts.items():
            if d.ndim < 2:
                assert not d.any(), name                     # biases and LayerNorm entries
                continue
            want = step0[name].reshape(d.shape[0], -1).double().norm(dim=1)
            got = d.reshape(d.shape[0], -1).double().norm(dim=1)
            np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=0)

    meshes = v.updates("surface", "mesh")
    assert len(meshes) == 2
    paths = v.updates("path", "lines")
    for m, s in zip(meshes, rec.surfaces):
        pos, _ = check_mesh(m)
        assert abs(pos[CENTRE, 2] - 0.5 * math.log10(s["loss"])) < 1e-4
        # the surface lies around the current point's coordinates
        after = next(p for p in paths if p.seq > m.seq)
        np.testing.assert_allclose(pos[CENTRE, :2], arr(after, "positions")[-1, :2], atol=1e-6)
        np.testing.assert_allclose(pos[:GRID, 0] - pos[CENTRE, 0], land.offsets.numpy(),
                                   atol=1e-6)

    check_path_ends_at_current(v)
    # one point every 10th step (0 to 90), never redrawn
    for a, b in zip(paths, paths[1:]):
        pa, pb = arr(a, "positions"), arr(b, "positions")
        assert len(pb) >= len(pa)
        np.testing.assert_array_equal(pb[:len(pa)], pa)
    final = arr(paths[-1], "positions")
    assert len(final) == 10
    # no plane, so no plane share: a gap at every step
    assert np.isnan(np.array(v.series())).all()
    # the path barely moves in random directions: its last point, at step 90,
    # against the coordinates of that step's weights on the anchor by least
    # squares, in float64
    theta = rec.thetas[90].double() - land.anchor.double()
    d = land.random_directions.double()
    coef = torch.linalg.solve(d @ d.T, d @ theta)
    np.testing.assert_allclose(final[-1, :2], coef.numpy() / 1.0, rtol=0, atol=1e-4)
    assert np.abs(final[:, :2]).max() < 0.05


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------
def test_controls(notebook):
    ns = notebook(160)
    land, model = ns["land"], ns["model"]
    ns["every"].set(1000)                                # one surface, at step 40
    rec = Recorder(ns)
    errors = []

    def train():
        try:
            ns["train"](160)
        except BaseException as e:                       # noqa: BLE001
            errors.append(e)

    with Viewer(ns["server"]) as v:
        loop = threading.Thread(target=train, daemon=True)
        loop.start()
        v.wait(lambda: len(v.updates("surface", "mesh")) == 1, what="the first surface")
        assert v.control("pause").status == "ok"
        v.wait(lambda: (v.run_state() or {}).get("state") == "paused", what="the pause")
        paused_at = land.step
        for name, value in (("span", 2.0), ("directions", "random"), ("every", 20)):
            assert v.control("set", name=name, value=value).status == "ok"
        want = {"span": 2.0, "directions": "random", "every": 20}
        v.wait(lambda: v.params() == want, what="the values taken while paused")
        assert len(v.updates("surface", "mesh")) == 1 and land.mode == "path" and land.span == 1.0
        reply = v.control("run", done=True, name="surface now")
        assert reply.status == "ok" and reply.message == f"refitted at step {paused_at}"
        v.wait(lambda: len(v.updates("surface", "mesh")) == 2, what="the surface while paused")
        assert land.step == paused_at and v.run_state()["state"] == "paused"
        now = rec.surfaces[1]
        assert (now["step"], now["mode"], now["span"]) == (paused_at, "random", 2.0)
        # its corner: theta - 2 d1 - 2 d2, at the coordinates of the current point
        pos, vals = check_mesh(v.updates("surface", "mesh")[1])
        theta = land.flat(model)
        corner = theta - 2.0 * land.random_directions[0] - 2.0 * land.random_directions[1]
        params = {n: t for n, t in zip(land.names, land.split(corner))}
        with torch.no_grad():
            logits = torch.func.functional_call(model, params, (ns["eval_x"][:128],))[0]
            want = math.log10(F.cross_entropy(logits, ns["eval_y"][:128]).item())
        assert abs(vals[0] - min(max(want, -4.0), 2.0)) < 1e-4
        d = land.random_directions.double()
        coef = torch.linalg.solve(d @ d.T, d @ (theta.double() - land.anchor.double()))
        np.testing.assert_allclose(pos[0, :2], coef.numpy() / 2.0 - 1.0, rtol=0, atol=1e-4)
        # the orientation: vertex GRID - 1 is theta + 2 d1 - 2 d2, one span right and one down
        corner = theta + 2.0 * land.random_directions[0] - 2.0 * land.random_directions[1]
        want = math.log10(loss_at(ns, corner))
        assert abs(vals[GRID - 1] - clip(want)) < 1e-4, (vals[GRID - 1], want)
        assert abs(now["far_corner"] - want) < 1e-6
        np.testing.assert_allclose(pos[GRID - 1, :2], coef.numpy() / 2.0 + [1.0, -1.0], rtol=0, atol=1e-4)

        assert v.control("resume").status == "ok"
        v.wait(lambda: len(v.updates("surface", "mesh")) == 3, what="the next surface")
        assert v.control("set", name="every", value=1000).status == "ok"
        nxt = rec.surfaces[2]
        assert nxt["step"] > paused_at and nxt["step"] % 20 == 0
        assert (nxt["mode"], nxt["span"]) == ("random", 2.0)
        loop.join(120)
        assert not loop.is_alive() and not errors, errors
    assert land.step == 159


# ---------------------------------------------------------------------------
# fixtures of Landscape alone
# ---------------------------------------------------------------------------
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(6, 8)
        self.b = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))


def toy():
    torch.manual_seed(1)
    model = Toy()
    x, y = torch.randn(16, 6), torch.randint(0, 3, (16,))
    return model, x, y


def set_weights(land, model, flat):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in zip(land.names, land.split(flat)):
            params[n].copy_(t)


def test_matched_directions_keep_order_and_sign():
    """Snapshots along a known curve in the plane of e1 and e2, around a base
    point that is the current weights at both refits: the e1 component is the
    same in both halves of the run and the e2 component grows four times, so
    that the second component outgrows the first between the refits at steps
    125 and 250 while the components stay e1 and e2 (the patterns are
    orthogonal, so the Gram matrix is diagonal in them)."""
    model, x, y = toy()
    land = Landscape(model, x, y, grid=5, point_every=1, snapshot_every=25)
    base = land.flat(model).clone()
    g = torch.Generator().manual_seed(2)
    e = torch.linalg.qr(torch.randn(land.size, 2, generator=g, dtype=torch.float64))[0].T.float()
    s1, s2 = [1, -1, -1, 1], [1, 1, -1, -1]
    curve = [(0.0, 0.0)] + [(1.0 * a, 0.5 * b) for a, b in zip(s1, s2)] + [(0.0, 0.0)] + \
        [(1.0 * a, 2.0 * b) for a, b in zip(s1, s2)] + [(0.0, 0.0)]       # at steps 0, 25, .., 250
    planes, eig = [], []
    for step in range(251):
        a, b = curve[step // 25] if step % 25 == 0 else (0.0, 0.0)
        set_weights(land, model, base + a * e[0] + b * e[1])
        land.point(model, step)
        if step in (125, 250):
            land.refit(model, 1.0, "path")
            planes.append(land.directions.double())
            eig.append(land.eigenvalues.double())
    first, second = planes
    np.testing.assert_allclose((first @ e.double().T).abs().numpy(), np.eye(2), atol=1e-5)
    assert eig[0][0] > eig[0][1]                                  # 4 against 1
    assert eig[1][1] > eig[1][0]                                  # 8 against 17: outgrown
    np.testing.assert_allclose(torch.diagonal(second @ first.T).numpy(), [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(eig[1].numpy(), [8.0, 17.0], rtol=1e-4)


def test_thinning_rule():
    model, x, y = toy()
    land = Landscape(model, x, y, grid=5, keep=8, point_every=1, snapshot_every=25)
    theta0 = land.flat(model).clone()
    kept = {}
    for step in range(601):
        set_weights(land, model, theta0 + 0.01 * step)
        land.point(model, step)
        kept[step] = list(land.snapshot_steps)
    assert kept[175] == list(range(0, 176, 25))
    assert kept[200] == [0, 50, 100, 150, 200]
    assert kept[350] == list(range(0, 351, 50))
    assert kept[400] == [0, 100, 200, 300, 400]
    assert kept[600] == list(range(0, 601, 100))
    for k, step in enumerate(land.snapshot_steps):              # each row the weights of its step
        np.testing.assert_allclose(land.snapshots[k].numpy(), (theta0 + 0.01 * step).numpy(),
                                   rtol=0, atol=1e-6)

    land = Landscape(model, x, y, grid=5)              # the defaults: keep 256, every 20 steps
    for step in range(8000):
        land.point(model, step)
        if step == 5119:
            assert len(land.snapshot_steps) == 256
        if step == 5120:
            assert len(land.snapshot_steps) == 129 and land.snapshot_steps[-2:] == [5080, 5120]
    assert len(land.snapshot_steps) == 200
    assert land.snapshot_steps == list(range(0, 5120, 40)) + list(range(5120, 8000, 40))
    with pytest.raises(ValueError):
        Landscape(model, x, y, grid=5, point_every=10, snapshot_every=25)


# ---------------------------------------------------------------------------
# the capture row: the hidden app on Metal, against a reviewed reference
# ---------------------------------------------------------------------------
EXE = os.environ.get("CALIPER_TEST_CALIPER_EXE", "")
REFERENCE = Path(__file__).resolve().parent / "reference" / "loss_landscape_app_step299.png"
WINDOW = (1280, 800)
MARKER = (255, 80, 80)


class Where:
    """The scene view's data area and camera, as the app's UI driver reports
    them (``where landscape``)."""

    def __init__(self, line: str):
        f = line.split()
        self.area = [float(v) for v in f[f.index("area") + 1:f.index("area") + 5]]
        t = f.index("target")
        self.target = np.array([float(v) for v in f[t + 1:t + 4]])
        self.distance = float(f[f.index("distance") + 1])
        self.azimuth = float(f[f.index("azimuth") + 1])
        self.elevation = float(f[f.index("elevation") + 1])
        self.fovy = float(f[f.index("fovy") + 1])
        p = f.index("target_px")
        self.target_px = (int(f[p + 1]), int(f[p + 2]))
        # The scale bar's band in points, which the view draws over its picture, if any.
        s = f.index("scale_bar") if "scale_bar" in f else -1
        self.scale_bar = [float(v) for v in f[s + 1:s + 5]] if s >= 0 else None

    def pixel(self, p, k: float):
        """The frame pixel (x, y) at which world point p is drawn (up is +z)."""
        u, a = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
        b = np.cross(a, u)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        eye = self.target + ((a * math.cos(self.azimuth) + b * math.sin(self.azimuth)) * ce +
                             u * se) * self.distance
        fwd = (self.target - eye) / np.linalg.norm(self.target - eye)
        side = np.cross(fwd, u)
        side /= np.linalg.norm(side)
        up = np.cross(side, fwd)
        d = np.asarray(p, float) - eye
        xv, yv, zv = d @ side, d @ up, -(d @ fwd)
        f = 1.0 / math.tan(self.fovy / 2)
        nx = f / (self.target_px[0] / self.target_px[1]) * xv / -zv
        ny = f * yv / -zv
        ax, ay, aw, ah = self.area
        return (ax + (nx * 0.5 + 0.5) * aw) * k, (ay + (0.5 - ny * 0.5) * ah) * k


def parity(cur, ref, ignore=None):
    """The parity rules of the scene rows (design/web-geometry.md): away from
    edges a pixel of a flat colour (the path, the marker, the frame, the
    background) equals the reference exactly and a lit pixel of the surface
    within 2 levels; a pixel within one pixel of an edge in the reference may
    take the colour of either side, within 2 levels. Pixels where ``ignore``
    is true are left out."""
    if cur.shape != ref.shape:
        return False, f"size {cur.shape} against the reference's {ref.shape}"
    c, r = cur.astype(np.int32), ref.astype(np.int32)
    h, w = r.shape[:2]
    pad = np.pad(r, ((1, 1), (1, 1), (0, 0)), mode="edge")
    near3 = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])
    edge = (near3.max(0) - near3.min(0)).max(-1) > 2
    diff = np.abs(c - r).max(-1)
    flat = np.zeros((h, w), bool)
    for colour in ((235, 235, 235), MARKER, (120, 120, 120)):
        flat |= (r == np.array(colour)).all(-1)
    either = np.abs(near3 - c[None]).max(-1).min(0)
    bad = (~edge & ((diff > 2) | (flat & (diff > 0)))) | (edge & (either > 2))
    if ignore is not None:
        bad &= ~ignore
    return not bad.any(), f"{int(bad.sum())} of {h * w} pixels outside the parity rules, " \
                          f"largest difference {int(diff.max())}"


@pytest.mark.skipif(sys.platform != "darwin" or not EXE or not os.path.exists(EXE),
                    reason="needs the Caliper app (CALIPER_TEST_CALIPER_EXE) and Metal")
def test_capture(notebook, tmp_path):
    """The scene after a surface forced at step 299, in the path's plane: the
    notebook's cells on the CPU with fixed seeds and deterministic algorithms
    for 300 steps with ``every`` at 100, the hidden app attached from the start
    (so its camera fits the static frame), and the frame from Caliper API. The
    marker is at the current point's projected pixel, and the scene view's
    area matches the reference under the parity rules. Without a reference the
    test writes the capture for review and skips: CALIPER_DEMO0_WRITE_REFERENCE=1
    writes it as the reference, which is committed once a person has reviewed
    it. CALIPER_SCENE_CAPTURE_DIR keeps the frame and the area."""
    Image = pytest.importorskip("PIL.Image")
    ns = notebook(300)
    land = ns["land"]
    ns["every"].set(100)
    data = tmp_path / "app"
    env = dict(os.environ, CALIPER_DATA_DIR=str(data / "data"),
               CALIPER_USER_APPLETS_DIR=str(data / "user"),
               CALIPER_WINDOW_SIZE=f"{WINDOW[0]}x{WINDOW[1]}",
               CALIPER_TEST_UI="; ".join(["where landscape", "wait 15"] * 4000))
    for d in ("data", "user"):
        (data / d).mkdir(parents=True)
    app = subprocess.Popen([EXE, "--hidden", "--url", ns["server"].stream_url, "--api",
                            "--api-port", "0"], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
    lines: list[str] = []
    threading.Thread(target=lambda: [lines.append(x.rstrip()) for x in app.stderr],
                     daemon=True).start()

    def wait(pred, what, timeout=60.0):
        end = time.monotonic() + timeout
        while not pred():
            assert app.poll() is None, "\n".join(lines[-30:])
            assert time.monotonic() < end, f"waited {timeout} s for {what}"
            time.sleep(0.02)

    def wheres():
        return [x for x in lines if x.startswith("[ui] where landscape ")]

    def capture():
        api = json.loads((data / "data" / "api.json").read_text())
        url = f"http://127.0.0.1:{api['port']}/v0/capture.png?token={api['token']}"
        with urllib.request.urlopen(url, timeout=20) as r:
            return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))

    try:
        wait(lambda: any(x.startswith("[renderer] ") for x in lines), "the renderer")
        renderer = next(x for x in lines if x.startswith("[renderer] "))
        if "metal" not in renderer:
            pytest.skip(f"the app has no Metal renderer here ({renderer})")
        # The app has subscribed once its scene view has drawn the scene's
        # state (the UI driver reports where only after an update). The
        # connections' stats are no sign of it: the native sender keeps a
        # scene's subscriptions in Python (caliper._net_scene), not there.
        wait(lambda: len(wheres()) > 0, "the app's subscription")
        wait(lambda: (data / "data" / "api.json").exists(), "Caliper API")
        ns["train"](300)
        ns["show_surface"]()                                  # a surface forced at step 299
        assert land.refits[-1] == (299, "path")
        # The app has drawn the final state once two captures, each after a new
        # frame of the UI driver, agree.
        prev, n = None, len(wheres())
        for _ in range(40):
            wait(lambda: len(wheres()) > n, "a frame of the app")
            n = len(wheres())
            img = capture()
            if prev is not None and np.array_equal(img, prev):
                break
            prev = img
        where = Where(wheres()[-1])
        k = img.shape[1] / WINDOW[0]
        ax, ay, aw, ah = (int(round(v * k)) for v in where.area)
        area = img[ay:ay + ah, ax:ax + aw]
        keep_dir = os.environ.get("CALIPER_SCENE_CAPTURE_DIR")
        if keep_dir:
            Image.fromarray(img).save(Path(keep_dir) / "loss-landscape-app-step299-frame.png")
            Image.fromarray(area).save(Path(keep_dir) / "loss-landscape-app-step299.png")
        # The marker, unlit and flat, around the current point's projected pixel
        # (within 4 points, for the rounding of the projection: from the default
        # camera, azimuth 0.95 and elevation 0.42 radians, no edge of the box
        # crosses the marker).
        x, y = (int(v) for v in where.pixel(land.current[0].tolist(), k))
        r = int(4 * k)
        patch = img[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(int)
        assert np.abs(patch - np.array(MARKER)).max(1).min() <= 1, "no marker near its point"
        if os.environ.get("CALIPER_DEMO0_WRITE_REFERENCE") == "1":
            REFERENCE.parent.mkdir(exist_ok=True)
            Image.fromarray(area).save(REFERENCE)
        if not REFERENCE.exists():
            out = tmp_path / REFERENCE.name
            Image.fromarray(area).save(out)
            pytest.skip(f"no reference yet: review {out} and commit it as "
                        f"{REFERENCE.relative_to(REPO)} (CALIPER_DEMO0_WRITE_REFERENCE=1)")
        # The scale bar is an overlay of the view layer, drawn over the picture
        # since the reference was taken, and checked by its own tests: its band
        # is left out of the comparison.
        ignore = np.zeros(area.shape[:2], bool)
        if where.scale_bar:
            x0, y0, x1, y1 = where.scale_bar
            ignore[max(0, int((y0 - where.area[1]) * k) - 1):int(math.ceil((y1 - where.area[1]) * k)) + 1,
                   max(0, int((x0 - where.area[0]) * k) - 1):int(math.ceil((x1 - where.area[0]) * k)) + 1] = True
        ok, report = parity(area, np.asarray(Image.open(REFERENCE).convert("RGB")), ignore)
        assert ok, report
    finally:
        if app.poll() is None:
            app.terminate()
            try:
                app.wait(20)
            except subprocess.TimeoutExpired:
                app.kill()


# ---------------------------------------------------------------------------
# the browser row: the viewer page in headless Chromium, against a reviewed
# reference
# ---------------------------------------------------------------------------
WEB_VIEWS = REPO / "python" / "caliper" / "_web" / "caliper-views.wasm"
PLAYWRIGHT = REPO / "web" / "wasm" / "test" / "node_modules" / "playwright-core"
WEB_ADAPTER = os.environ.get("CALIPER_DEMO0_WEB_ADAPTER", "swiftshader")
WEB_REFERENCE = Path(__file__).resolve().parent / "reference" / f"loss_landscape_web_{WEB_ADAPTER}_step299.png"


@pytest.mark.skipif(not WEB_VIEWS.exists() or shutil.which("node") is None or not PLAYWRIGHT.is_dir(),
                    reason="needs the WebAssembly views in python/caliper/_web (scripts/build_wasm.py), "
                           "Node and `npm ci` in web/wasm/test")
def test_capture_browser(notebook, tmp_path):
    """The same scene on the viewer page of ``caliper.serve()``, drawn by Caliper's
    WebAssembly views in headless Chromium (tests/demos/viewer_page.mjs), on
    SwiftShader unless CALIPER_DEMO0_WEB_ADAPTER is "metal": the notebook's cells
    on the CPU with fixed seeds and deterministic algorithms for 300 steps with
    ``every`` at 100, the page open from the start (so its camera fits the
    static frame), a surface forced at step 299, and the scene's drawing area
    once its module has received every message the server sent on its
    connection and drawn 30 more frames. The path's, the marker's and the
    frame's colours appear exactly, and the area matches the reference under
    the parity rules, leaving out the scale bar's band, which the module
    publishes. Without a reference the test writes the capture for review
    and skips: CALIPER_DEMO0_WRITE_WEB_REFERENCE=1 writes it as the reference,
    which is committed once a person has reviewed it. CALIPER_SCENE_CAPTURE_DIR
    keeps the area and the page's report of it (its geometry and scale bar)."""
    Image = pytest.importorskip("PIL.Image")
    ns = notebook(300)
    land, server = ns["land"], ns["server"]
    ns["every"].set(100)
    page = subprocess.Popen(["node", str(Path(__file__).resolve().parent / "viewer_page.mjs"), server.url,
                             WEB_ADAPTER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    lines: list[str] = []
    threading.Thread(target=lambda: [lines.append(x.rstrip()) for x in page.stdout], daemon=True).start()

    def wait(pred, what, timeout=120.0):
        end = time.monotonic() + timeout
        while not pred():
            assert page.poll() is None or pred(), "\n".join(lines[-20:])
            assert time.monotonic() < end, f"waited {timeout} s for {what}"
            time.sleep(0.02)

    # A link's id in the server's stats: the native sender's connection, or the
    # Python sender's link. Scene frames are counted on the link either way
    # (caliper._net_scene), but only the Python sender's stats list them.
    def stats_id(ln):
        return getattr(ln, "conn", ln.id)

    def frames_out(lid):
        return next(c for c in server.stats()["connections"] if c["id"] == lid)["frames_out"]

    def sent_when_idle(lid):
        """The frames sent on connection ``lid`` once nothing waits to go on it:
        no scene batch in flight from the device, and on the connection no
        control message and no scene batch queued, nor with the native sender
        a frame handed to it and not yet sent, at two reads in a row with the
        same count."""
        from caliper import _net_scene
        route = _net_scene.route_of(server.ring)
        link = next(ln for ln in server.producer.links if stats_id(ln) == lid)

        def idle():
            try:
                return not route.busy() and not link.control and (link.scenes is None or all(
                    not q.state and not q.pending for q in list(link.scenes.subs.values()))) and (
                    not hasattr(link, "conn") or link.p.nio.queued(link.conn) == 0)
            except RuntimeError:                                 # changed while read
                return False
        prev = None
        while True:
            n = frames_out(lid)
            if idle():
                if n == prev:
                    return n
                prev = n
            else:
                prev = None
            time.sleep(0.05)

    try:
        wait(lambda: any(x == "READY" or x.split(" ")[0] in ("SKIP", "FAIL") for x in lines), "the page")
        first = next(x for x in lines if x == "READY" or x.split(" ")[0] in ("SKIP", "FAIL"))
        if first.startswith("SKIP"):
            pytest.skip(first[5:])
        assert first == "READY", first
        links = [ln for ln in list(server.producer.links) if ln.stats.get("scenes_sent", 0) > 0]
        scene = [stats_id(ln) for ln in links]
        assert len(scene) == 1, server.stats()["connections"]   # the scene's module, one connection
        ns["train"](300)
        ns["show_surface"]()                                      # a surface forced at step 299
        assert land.refits[-1] == (299, "path")
        n = sent_when_idle(scene[0])                              # the run is idle now
        scenes_at = links[0].stats["scenes_sent"]
        png = tmp_path / "browser-step299.png"
        page.stdin.write(f"shot {n} {png}\n")
        page.stdin.flush()
        wait(lambda: any(x.startswith("SHOT ") for x in lines), "the capture")
        shot = json.loads(next(x for x in lines if x.startswith("SHOT "))[5:])
        assert "error" not in shot or not shot["error"], shot
        # No scene message was sent after the count. The module PINGs after 10 s
        # without a message (CALIPER_NET_PING_INTERVAL_MS), which a slow capture
        # can reach, and the PONG it gets is a frame of the connection too.
        assert links[0].stats["scenes_sent"] == scenes_at, (scenes_at, links[0].stats["scenes_sent"])
        assert shot["drawer"] == "wasm" and not shot["gfxErrors"], shot
        assert shot["info"].startswith("4 entities"), shot["info"]
        area = np.asarray(Image.open(png).convert("RGB"))
        keep_dir = os.environ.get("CALIPER_SCENE_CAPTURE_DIR")
        if keep_dir:
            Image.fromarray(area).save(Path(keep_dir) / f"loss-landscape-web-{WEB_ADAPTER}-step299.png")
            (Path(keep_dir) / f"loss-landscape-web-{WEB_ADAPTER}-step299.json").write_text(json.dumps(shot, indent=1))
        # the flat colours of the path, the marker and the frame, exactly
        flat = area.reshape(-1, 3)
        for colour in ((235, 235, 235), MARKER, (120, 120, 120)):
            assert (flat == np.array(colour)).all(-1).any(), f"no pixel of {colour}"
        if os.environ.get("CALIPER_DEMO0_WRITE_WEB_REFERENCE") == "1":
            WEB_REFERENCE.parent.mkdir(exist_ok=True)
            Image.fromarray(area).save(WEB_REFERENCE)
        if not WEB_REFERENCE.exists():
            out = tmp_path / WEB_REFERENCE.name
            Image.fromarray(area).save(out)
            pytest.skip(f"no reference yet: review {out} and commit it as "
                        f"{WEB_REFERENCE.relative_to(REPO)} (CALIPER_DEMO0_WRITE_WEB_REFERENCE=1)")
        # The scale bar, as in the app's row: an overlay drawn over the picture
        # since the reference was taken, its band (which the module publishes in
        # the canvas's CSS pixels, the canvas filling the capture) left out.
        assert shot["geometry"] is not None and shot["cssW"] > 0, shot
        ignore = np.zeros(area.shape[:2], bool)
        if shot["geometry"].get("scaleBar"):
            x0, y0, x1, y1 = shot["geometry"]["scaleBar"]
            ax, ay, aw, ah = shot["geometry"]["area"]
            # a corner band inside the picture, never a large part of it
            assert ax <= x0 < x1 <= ax + aw and ay <= y0 < y1 <= ay + ah, shot["geometry"]
            assert (x1 - x0) * (y1 - y0) < 0.05 * aw * ah, shot["geometry"]
            k = area.shape[1] / shot["cssW"]
            ignore[max(0, int(y0 * k) - 1):int(math.ceil(y1 * k)) + 1,
                   max(0, int(x0 * k) - 1):int(math.ceil(x1 * k)) + 1] = True
        ok, report = parity(area, np.asarray(Image.open(WEB_REFERENCE).convert("RGB")), ignore)
        assert ok, report
    finally:
        if page.poll() is None:
            page.stdin.close()
            try:
                page.wait(20)
            except subprocess.TimeoutExpired:
                page.kill()
