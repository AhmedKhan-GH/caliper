"""Demo 0, the loss landscape (demos/loss_landscape): the acceptance rows of
design/demos.md that need no pixels.

The notebook's own code cells run in this process on the CPU, with fixed
seeds and deterministic algorithms, and a test viewer on a real loopback
connection to the notebook's ``caliper.serve()`` decodes every scene batch:

* the path's plane, 300 steps with ``every`` at 100, so that refits come at
  steps 50, 100 and 200: the scene's entities and the frame at its root,
  nothing drawn before step 50, the mesh's shape, the centre vertex at half
  the log10 of the loss computed separately, the path's last point at the
  current point, orthonormal directions, the first nine points of the path
  after the refit at step 200 against the snapshots' projections in float64,
  and ``plane share``;
* random directions, 100 steps with ``every`` at 50: the filter-normalised
  rows, zero biases and normalisation entries, the centre vertex, a path of
  one point per step that is never redrawn;
* the controls: ``span``, ``directions`` and ``every`` set while paused take
  effect at the next surface, and ``surface now`` computes one while paused;
* two fixtures of ``Landscape`` alone: snapshots along a known curve whose
  second component outgrows its first, for the order and sign of the matched
  directions, and the thinning rule with room for 8 snapshots and for 256.

    PYTHONPATH=python .venv/bin/python -m pytest tests/demos

The capture row of the acceptance waits for the scene view (see
``test_capture``)."""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import math
import sys
import threading
import time
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


class Recorder:
    """Wraps the notebook's show_surface, show_path and land.point to record,
    at each surface, the step, the separately computed loss and the plane, and
    the steps at which anything was drawn."""

    def __init__(self, ns):
        self.ns = ns
        self.surfaces: list[dict] = []
        self.losses: dict[int, float] = {}               # snapshot steps -> plain loss
        self.drawn: list[int] = []
        land = ns["land"]
        point, surface, path = land.point, ns["show_surface"], ns["show_path"]

        def recording_point(model, step):
            p = point(model, step)
            if step % land.snapshot_every == 0:
                self.losses[step] = plain_loss(ns)
            return p

        def recording_surface():
            entry = {"step": land.step, "loss": plain_loss(ns)}
            surface()
            entry.update(mode=land.mode, span=land.span, half_width=float(land.half_width),
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
    once it is listed, the series ``plane share``, and keeps every message it
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
        v.wait(lambda: any(len(arr(m, "positions")) == 108 for m in v.updates("path", "lines")),
               what="the last path")
        v.wait(lambda: len(v.series()) == 3, what="three plane shares")

    # refits at 50, 100 and 200; nothing drawn before step 50
    assert [s["step"] for s in rec.surfaces] == [50, 100, 200]
    assert land.refits == [(50, "path"), (100, "path"), (200, "path")] and land.method == "vmap"
    assert min(rec.drawn) >= 50
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

    check_path_ends_at_current(v)

    # orthonormal directions
    for s in rec.surfaces:
        d = s["directions"].double()
        np.testing.assert_allclose((d @ d.T).numpy(), np.eye(2), rtol=0, atol=1e-5)

    # after the refit at step 200: one point per kept snapshot, then one per step
    last = rec.surfaces[-1]
    assert last["snapshots"] == 9 and land.snapshot_steps[:9] == list(range(0, 201, 25))
    final = arr(v.updates("path", "lines")[-1], "positions")
    assert len(final) == 9 + (299 - 200)
    snaps = land.snapshots[:9].double()
    theta_r, plane = last["centre"].double(), last["directions"].double()
    coords = (snaps - theta_r) @ plane.T                                   # float64, on the CPU
    half = max(1.5 * 1.0 * coords.abs().max().item(), 1e-3 * theta_r.norm().item())
    np.testing.assert_allclose(final[:9, :2], (coords / half).numpy(), rtol=0, atol=1e-4)
    heights = [0.5 * math.log10(rec.losses[s]) for s in range(0, 201, 25)]
    np.testing.assert_allclose(final[:9, 2], heights, rtol=0, atol=1e-4)
    assert abs(last["half_width"] - half) <= 1e-5 * half

    # the plane holds nearly as much of the path as the exact top two components
    m64 = snaps - theta_r
    top = torch.linalg.eigvalsh(m64 @ m64.T)[-2:].sum() / (m64 * m64).sum()
    assert abs(last["share"] - top.item()) < 1e-3

    # plane share: one value per refit, between 0 and 1
    shares = v.series()
    np.testing.assert_allclose(shares, [s["share"] for s in rec.surfaces], rtol=0, atol=1e-6)
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert shares[2] >= PINNED_SHARE_200 - 0.05, shares


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
        v.wait(lambda: any(len(arr(m, "positions")) == 100 for m in v.updates("path", "lines")),
               what="the last path")

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
    # one point per step, never redrawn
    for a, b in zip(paths, paths[1:]):
        pa, pb = arr(a, "positions"), arr(b, "positions")
        assert len(pb) >= len(pa)
        np.testing.assert_array_equal(pb[:len(pa)], pa)
    final = arr(paths[-1], "positions")
    assert len(final) == 100
    # the path barely moves in random directions: its coordinates against the
    # anchor by least squares, in float64
    theta = land.flat(model).double() - land.anchor.double()
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
    ns["every"].set(1000)                                # one surface, at step 50
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
        time.sleep(0.3)
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
    land = Landscape(model, x, y, grid=5)
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
    land = Landscape(model, x, y, grid=5, keep=8, snapshot_every=25)
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

    land = Landscape(model, x, y, grid=5)                       # keep 256, every 25
    for step in range(8000):
        land.point(model, step)
        if step == 6399:
            assert len(land.snapshot_steps) == 256
        if step == 6400:
            assert len(land.snapshot_steps) == 129 and land.snapshot_steps[-2:] == [6350, 6400]
    assert len(land.snapshot_steps) == 160
    assert land.snapshot_steps == list(range(0, 6400, 50)) + list(range(6400, 8000, 50))


# ---------------------------------------------------------------------------
# the capture row
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="the scene view that draws a scene in the browser is not on this branch "
                         "yet (stream view of sprint 4); the capture of the scene after step 299 "
                         "in the path's plane, 512 x 512 from the default camera, runs once it is")
def test_capture():
    pass
