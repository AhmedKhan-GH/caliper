"""caliper.scene on the network route (caliper/_scene.py, _net_scene.py): the
scenes subset of wire protocol 3.2 (draft), over a real loopback connection
with a websockets viewer, on both senders (the native one when
caliper._netio is built) and on the CPU and, where there is one, MPS:

* scene.mesh, scene.lines and scene.points reach a viewer with the right
  arrays, cast and bounded on the device;
* a late subscriber receives the static items and the current state, opened
  by SCENE_CLEAR with reset;
* a rate limit drops a batch and never a static one;
* every refusal raises, attached or not, and the arguments of later
  increments raise NotImplementedError;
* a viewer without the capability "scenes" sees no scene;
* the calls of demo 0 (the loss landscape) run in their order;
* the cost of a call with nothing attached stays small."""
from __future__ import annotations

import asyncio
import math
import os
import statistics
import time
import warnings

import numpy as np
import pytest
import torch

import caliper
from caliper import _scene, _wire as w

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

MPS = torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="no MPS device"))]
try:
    from caliper import _netio  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False
MODES = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not NATIVE, reason="caliper._netio is not built (python scripts/build_netio.py)"))]


@pytest.fixture(autouse=True)
def _forget_scenes():
    yield
    for name in list(_scene.SCENES):
        caliper.remove(name)


@pytest.fixture(params=MODES)
def server(request, monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1" if request.param == "native" else "0")
    s = caliper.serve(port=0)
    assert (type(s).__name__ == "NativeServer") == (request.param == "native")
    yield s
    s.close()


def run(coro, timeout=30.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Viewer:
    """A viewer that announces "scenes" (or not) and decodes everything."""

    def __init__(self, server, caps=(w.CAP_SCENES,)):
        self.url = server.stream_url
        self.caps = list(caps)
        self.inbox: list = []
        self.views: dict = {}

    async def __aenter__(self):
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None,
                                close_timeout=1).__aenter__()
        await self.ws.send(w.encode(w.Hello("viewer", self.caps)))
        self.welcome = await self.expect(w.Welcome)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def recv(self, timeout):
        m = w.decode(await asyncio.wait_for(self.ws.recv(), timeout))
        if isinstance(m, w.Views):
            self.views = {v["name"]: v for v in m.views}
        return m

    async def expect(self, cls, pred=lambda m: True, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            for i, m in enumerate(self.inbox):
                if isinstance(m, cls) and pred(m):
                    return self.inbox.pop(i)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {cls.__name__}; got {[type(m).__name__ for m in self.inbox]}")
            try:
                self.inbox.append(await self.recv(left))
            except asyncio.TimeoutError:
                pass

    async def quiet(self, seconds=0.3) -> list:
        """Everything that arrives within ``seconds``."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            try:
                self.inbox.append(await self.recv(left))
            except asyncio.TimeoutError:
                break
        out, self.inbox = self.inbox, []
        return out

    async def subscribe(self, sub, view, max_rate=None, data_view=None):
        if view not in self.views:
            await self.expect(w.Views, lambda m: any(v["name"] == view for v in m.views))
        await self.send(w.Subscribe(sub, view, 100 + sub, max_rate, data_view))
        return await self.expect(w.ViewReply, lambda m: m.request_id == 100 + sub)

    async def send(self, msg):
        await self.ws.send(w.encode(msg))

    async def update(self, path, part=None, timeout=10.0):
        return await self.expect(w.SceneUpdate, lambda m: m.path == path and
                                 (part is None or m.part == part), timeout)

    async def state(self, sub):
        """The scene's state for subscription ``sub``: the reset, then the updates."""
        clear = await self.expect(w.SceneClear, lambda m: m.sub == sub)
        assert clear.reset and clear.state and clear.path == ""
        out = []
        for m in await self.quiet(0.5):
            if isinstance(m, w.SceneUpdate) and m.sub == sub and m.state:
                out.append(m)
            else:
                self.inbox.append(m)
        return out


def arr(m, name):
    return m.arrays[name].array()


# ---------------------------------------------------------------------------
# the three parts reach a viewer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_mesh_lines_and_points_reach_a_viewer(server, device):
    sc = caliper.scene("landscape", up="+z", units="")
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(1681, 3, generator=g).to(device)
    tri = torch.randint(0, 1681, (3200, 3), generator=g).to(device)              # i64: cast to i32
    vals = torch.rand(1681, generator=g).to(device).half()                       # f16: cast to f32
    path = torch.rand(7, 3, generator=g, dtype=torch.float64).to("cpu")           # f64 on the host
    pts = torch.rand(5, 3, generator=g).to(device)
    rgb = torch.rand(5, 3, generator=g).to(device)                               # floats in [0, 1]
    radii = torch.full((5,), 0.25).to(device)

    async def body():
        async with Viewer(server) as v:
            r = await v.subscribe(1, "landscape")
            assert r.data_view == w.SCENE_DATA_VIEW
            assert v.views["landscape"]["kind"] == w.Kind.SCENE
            assert w.scene_view_info(v.views["landscape"]) == {"space": "3d", "up": "+z", "units": ""}
            await v.state(1)
            assert sc.mesh("surface", pos, tri, values=vals, cmap="viridis", vmin=-4.0, vmax=2.0)
            assert sc.lines("path", path, strip_lengths=[7], colors=(235, 235, 235), rate=10)
            assert sc.points("cloud", pts, colors=rgb, radii=radii)
            m = await v.update("surface", "mesh")
            np.testing.assert_array_equal(arr(m, "positions"), pos.cpu().numpy())
            assert arr(m, "indices").dtype == np.int32
            np.testing.assert_array_equal(arr(m, "indices"), tri.cpu().int().numpy())
            np.testing.assert_array_equal(arr(m, "values"), vals.float().cpu().numpy())
            assert (m.params["colormap"], m.params["flags"], m.params["vmin"], m.params["vmax"]) == \
                (0, 0, -4.0, 2.0)
            p = pos.cpu().numpy()
            want = [float(vals.float().min()), float(vals.float().max())]
            for k in range(3):
                want += [float(p[:, k].min()), float(p[:, k].max())]
            np.testing.assert_allclose(m.params["stats"], want, rtol=0, atol=1e-6)
            assert m.view == "landscape" and not m.static and not m.state and m.axes is None

            ln = await v.update("path", "lines")
            np.testing.assert_array_equal(arr(ln, "positions"), path.float().numpy())
            assert arr(ln, "strip_lengths").tolist() == [7]
            assert ln.params["mode"] == "strips" and ln.params["color"] == [235, 235, 235, 255]
            assert ln.params["width"] == 2.0

            pt = await v.update("cloud", "points")
            want_rgb = (rgb.cpu().clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
            np.testing.assert_array_equal(arr(pt, "colors")[:, :3], want_rgb)
            assert (arr(pt, "colors")[:, 3] == 255).all()
            np.testing.assert_array_equal(arr(pt, "radii"), radii.cpu().numpy())
            assert pt.seq > ln.seq > m.seq
    run(body())


def test_segments_strips_per_strip_colours_and_auto_range(server):
    sc = caliper.scene("lines")
    seg = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=torch.float32)
    strips = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    colours = torch.tensor([[255, 0, 0, 255], [0, 0, 255, 128]], dtype=torch.uint8)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "lines")
            await v.state(1)
            sc.lines("box", seg, width=1.5, values=torch.tensor([0.0, 1.0, 2.0, 3.0]))
            sc.lines("tracks", strips, strip_lengths=torch.tensor([3, 2]), colors=colours)
            a = await v.update("box")
            assert a.params["mode"] == "segments" and a.params["width"] == 1.5
            assert a.params["flags"] == w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX
            assert a.params["stats"][:2] == [0.0, 3.0]
            b = await v.update("tracks")
            assert b.params["color_per"] == "strip" and arr(b, "colors").tolist() == colours.tolist()
            assert arr(b, "strip_lengths").tolist() == [3, 2]
    run(body())


def test_2d_scene_mesh(server):
    sc = caliper.scene("flat", space="2d")
    assert sc.up == "-y"
    tri = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])

    async def body():
        async with Viewer(server) as v:
            assert w.scene_view_info((await v.expect(w.Views)).views[0])["space"] == "2d"
            await v.subscribe(1, "flat")
            await v.state(1)
            sc.mesh("shape", tri, colors=[(10, 20, 30)] * 3)
            m = await v.update("shape")
            assert arr(m, "positions").shape == (3, 2)
            assert m.params["stats"] == [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0]
            assert arr(m, "colors").tolist() == [[10, 20, 30, 255]] * 3
    run(body())


def test_points_over_the_budget_are_sampled_by_stride(server):
    sc = caliper.scene("big")
    n = 150_000
    pos = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "big")
            await v.state(1)
            sc.points("cloud", pos, values=torch.arange(n, dtype=torch.float32))
            m = await v.update("cloud", timeout=20)
            size = w.DEFAULT_MAX_POINTS
            assert m.axes == [{"dim": "point", "start": 0, "step": 1, "count": n, "size": size}]
            rows = np.arange(size, dtype=np.int64) * n // size
            np.testing.assert_array_equal(arr(m, "values"), rows.astype(np.float32))
            np.testing.assert_array_equal(arr(m, "positions"), pos.numpy()[rows])
    run(body())


# ---------------------------------------------------------------------------
# a late subscriber, static items, the rate limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_late_subscriber_gets_static_items_and_the_current_state(server, device):
    sc = caliper.scene("world")
    frame = torch.tensor([[-1, -1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]],
                         dtype=torch.float32).to(device)
    assert sc.lines("", frame, static=True, width=1.0) is True          # nothing attached
    assert sc.mesh("surface", torch.zeros(3, 3).to(device)) is False      # not retained

    async def body():
        async with Viewer(server) as a:
            await a.subscribe(1, "world")
            st = await a.state(1)
            assert [(m.path, m.part, m.static) for m in st] == [("", "lines", True)]
            np.testing.assert_array_equal(arr(st[0], "positions"), frame.cpu().numpy())
            surf = torch.rand(6, 3).to(device)
            assert sc.mesh("surface", surf, values=torch.arange(6.0).to(device))
            assert sc.points("current", torch.ones(1, 3).to(device), colors=(255, 80, 80))
            await a.update("surface")
            await a.update("current")
            async with Viewer(server) as b:
                await b.subscribe(7, "world")
                st = await b.state(7)
                assert [(m.path, m.part) for m in st] == [("", "lines"), ("current", "points"),
                                                          ("surface", "mesh")]
                assert all(m.state for m in st) and [m.static for m in st] == [True, False, False]
                np.testing.assert_array_equal(arr(st[2], "positions"), surf.cpu().numpy())
                assert st[1].params["color"] == [255, 80, 80, 255]
    run(body())


def test_static_items_logged_before_serve_are_listed_and_sent(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "0")
    sc = caliper.scene("setup")
    assert sc.points("anchor", torch.zeros(2, 3), static=True) is True
    s = caliper.serve(port=0)
    try:
        async def body():
            async with Viewer(s) as v:
                await v.subscribe(1, "setup")
                st = await v.state(1)
                assert [(m.path, m.static) for m in st] == [("anchor", True)]
        run(body())
    finally:
        s.close()


def test_a_newer_static_item_replaces_the_older(server):
    sc = caliper.scene("setup")
    sc.lines("", torch.zeros(2, 3), static=True)
    sc.lines("", torch.ones(4, 3), static=True)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "setup")
            st = await v.state(1)
            assert len(st) == 1 and arr(st[0], "positions").shape == (4, 3)
    run(body())


def test_rate_limit_drops_a_batch_and_never_a_static_one(server):
    sc = caliper.scene("rates", rate=2)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "rates")
            await v.state(1)
            got = [sc.points("p", torch.full((1, 3), float(i))) for i in range(20)]
            assert got[0] is True and not any(got[1:])                   # 2 per second
            assert all(sc.points("q", torch.zeros(1, 3), static=True) for _ in range(5))
            assert sc.points("r", torch.zeros(1, 3), rate=0) and sc.points("r", torch.zeros(1, 3), rate=0)
            msgs = await v.quiet(0.8)
            ups = [m for m in msgs if isinstance(m, w.SceneUpdate)]
            assert [arr(m, "positions")[0, 0] for m in ups if m.path == "p"] == [0.0]
            assert len([m for m in ups if m.path == "q"]) >= 1            # coalesced per part
            assert _scene.SCENES["rates"].counters["rate_limited"] == 19
    run(body())


def test_unwatched_batches_cost_no_snapshot(server):
    sc = caliper.scene("idle")
    assert sc.mesh("m", torch.zeros(3, 3)) is False
    assert _scene.SCENES["idle"].counters.get("taken", 0) == 0


def test_max_rate_applies_to_each_entity_part(server):
    sc = caliper.scene("pace", rate=0)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "pace", max_rate=4)
            await v.state(1)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                sc.points("a", torch.zeros(1, 3))
                sc.points("b", torch.zeros(1, 3))
                await asyncio.sleep(0.01)
            msgs = await v.quiet(0.4)
            for p in ("a", "b"):
                n = len([m for m in msgs if isinstance(m, w.SceneUpdate) and m.path == p])
                assert 2 <= n <= 7, (p, n)                     # 4 per second each, not 100
    run(body())


# ---------------------------------------------------------------------------
# capabilities, removal, the widget
# ---------------------------------------------------------------------------
def test_a_viewer_without_scenes_sees_none(server):
    caliper.scene("hidden")
    caliper.series(1.0, name="loss")

    async def body():
        async with Viewer(server, caps=()) as v:
            assert w.CAP_SCENES in v.welcome.capabilities
            views = await v.expect(w.Views, lambda m: any(x["name"] == "loss" for x in m.views))
            assert "hidden" not in {x["name"] for x in views.views}
            v.views["hidden"] = {}
            r = await v.subscribe(1, "hidden")
            assert r.error["code"] == w.ERR_UNKNOWN_VIEW
    run(body())


def test_remove_ends_the_scene(server):
    sc = caliper.scene("gone")

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "gone")
            await v.state(1)
            assert caliper.remove("gone") is False
            await v.expect(w.Views, lambda m: all(x["name"] != "gone" for x in m.views))
            assert "gone" not in _scene.SCENES
            assert sc.points("p", torch.zeros(1, 3)) is False       # a removed scene logs nowhere
    run(body())


def test_widget_takes_the_scene_kind(server):
    pytest.importorskip("anywidget")
    caliper.scene("landscape")
    wd = caliper.widget("landscape", "scene", height=480)
    assert wd.kind == "scene" and wd.view == "landscape"


def test_warns_once_without_a_network_route(monkeypatch):
    class Fake:
        session = "fake"

        def attached(self):
            return True

    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setattr(_scene, "_warned", False)
    route = caliper._attach_transport(Fake())
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sc = caliper.scene("nowhere")
            assert sc.points("p", torch.zeros(1, 3)) is False
            assert sc.points("p", torch.zeros(1, 3)) is False
        assert len([r for r in rec if "scenes reach only caliper.serve()" in str(r.message)]) == 1
    finally:
        caliper._detach_transport(route)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_scene_refusals():
    for kw in ({"space": "4d"}, {"up": "z"}, {"space": "2d", "up": "+z"}, {"units": "x" * 16},
               {"rate": -1}, {"rate": math.nan}):
        with pytest.raises(ValueError):
            caliper.scene("s", **kw)
    for name in ("", "a##b", "x" * 64, 3):
        with pytest.raises(ValueError):
            caliper.scene(name)
    with pytest.raises(TypeError):
        caliper.scene("s", rate="fast")
    caliper.scene("s")
    with pytest.raises(ValueError, match="keeps its space"):
        caliper.scene("s", space="2d")
    with pytest.raises(NotImplementedError, match="increment 3"):
        caliper.scene("s", data_view={})
    with pytest.raises(NotImplementedError, match="increment 3"):
        caliper.scene("s", view_camera={"azimuth": 30})
    assert caliper.scene("s", up="+z").up == "+z"                     # a second call updates up


def test_call_refusals():
    sc = caliper.scene("s")
    p3 = torch.zeros(4, 3)
    bad = [
        (TypeError, lambda: sc.points(3, p3)),
        (ValueError, lambda: sc.points("a//b", p3)),
        (ValueError, lambda: sc.points("a/..", p3)),
        (ValueError, lambda: sc.points("p", torch.zeros(4, 2))),                    # 2-D in 3-D
        (ValueError, lambda: sc.points("p", torch.zeros(4))),
        (ValueError, lambda: sc.points("p", torch.zeros(4, 3, dtype=torch.int64))),
        (TypeError, lambda: sc.points("p", "positions")),
        (ValueError, lambda: sc.points("p", p3, colors=torch.zeros(4, 3), values=torch.zeros(4))),
        (ValueError, lambda: sc.points("p", p3, colors=torch.zeros(3, 3))),
        (ValueError, lambda: sc.points("p", p3, colors=torch.zeros(4, 3, dtype=torch.int32))),
        (ValueError, lambda: sc.points("p", p3, colors=(300, 0, 0))),
        (ValueError, lambda: sc.points("p", p3, colors=(0.5, 0.5))),
        (ValueError, lambda: sc.points("p", p3, values=torch.zeros(5))),
        (ValueError, lambda: sc.points("p", p3, radii=0)),
        (ValueError, lambda: sc.points("p", p3, radii=torch.zeros(3))),
        (ValueError, lambda: sc.points("p", p3, cmap="jet")),
        (ValueError, lambda: sc.points("p", p3, vmin=math.inf)),
        (TypeError, lambda: sc.points("p", p3, static=1)),
        (ValueError, lambda: sc.points("p", p3, rate=-1)),
        (ValueError, lambda: sc.lines("l", torch.zeros(3, 3))),                     # odd segments
        (ValueError, lambda: sc.lines("l", p3, strip_lengths=[3])),                 # sum
        (ValueError, lambda: sc.lines("l", p3, strip_lengths=[4, 0])),
        (ValueError, lambda: sc.lines("l", p3, strip_lengths=[1.5, 2.5])),
        (ValueError, lambda: sc.lines("l", p3, strip_lengths=torch.ones(2, 2))),
        (ValueError, lambda: sc.lines("l", p3, width=0)),
        (ValueError, lambda: sc.lines("l", p3, color_per="strip")),
        (ValueError, lambda: sc.lines("l", p3, color_per="face")),
        (ValueError, lambda: sc.lines("l", p3, strip_lengths=[2, 2], colors=torch.zeros(3, 4))),
        (ValueError, lambda: sc.lines("l", torch.zeros(2, 3), strip_lengths=[1, 1],
                                      values=torch.zeros(2))),                      # V == S
        (ValueError, lambda: sc.mesh("m", p3)),                                     # 4 % 3
        (ValueError, lambda: sc.mesh("m", p3, torch.zeros(2, 3))),                  # float indices
        (ValueError, lambda: sc.mesh("m", p3, torch.zeros(2, 2, dtype=torch.int64))),
        (ValueError, lambda: sc.mesh("m", p3[:3], values=torch.zeros(4))),
        (ValueError, lambda: sc.points("p", torch.zeros(1, 3).expand(w.MAX_BATCH_POINTS + 1, 3))),
        (ValueError, lambda: sc.mesh("m", torch.zeros(1, 3).expand(w.MAX_BATCH_POINTS, 3),
                                     torch.zeros(1, 3, dtype=torch.int32).expand(
                                         w.MAX_BATCH_POINTS, 3))),                  # > 64 MiB
    ]
    for exc, call in bad:
        with pytest.raises(exc):
            call()
    later = [
        (lambda: sc.points("p", p3, class_ids=torch.zeros(4)), "increment 3"),
        (lambda: sc.points("p", p3, labels=["a"] * 4), "increment 3"),
        (lambda: sc.points("p", p3, items="test"), "increment 5"),
        (lambda: sc.lines("l", p3, indices=torch.zeros(2, 2)), "increment 3"),
        (lambda: sc.lines("l", p3, color_per="edge"), "increment 3"),
        (lambda: sc.mesh("m", p3[:3], normals=torch.zeros(3, 3)), "increment 3"),
        (lambda: sc.mesh("m", p3[:3], uvs=torch.zeros(3, 2)), "increment 3"),
        (lambda: sc.mesh("m", p3[:3], texture=torch.zeros(2, 2, 3)), "increment 3"),
        (lambda: sc.arrows("a", p3, p3), "increment 3"),
        (lambda: sc.boxes("b"), "increment 3"),
        (lambda: sc.image("i", torch.zeros(2, 2, 3)), "increment 3"),
        (lambda: sc.camera("c", width=2, height=2), "increment 3"),
        (lambda: sc.transform("t"), "increment 3"),
        (lambda: sc.classes({}), "increment 3"),
        (lambda: sc.clear(""), "increment 3"),
    ]
    for call, inc in later:
        with pytest.raises(NotImplementedError, match=inc):
            call()


def test_entity_limit():
    sc = caliper.scene("many")
    pos = torch.zeros(1, 3)
    for i in range(w.MAX_ENTITIES):
        sc.points(f"e{i}", pos)
    sc.lines("e0", torch.zeros(2, 3))                    # another part of a known entity is fine
    with pytest.raises(ValueError, match="1024 entities"):
        sc.points("one-more", pos)


# ---------------------------------------------------------------------------
# demo 0: the calls of design/demos.md, in their order
# ---------------------------------------------------------------------------
class TinyLandscape:
    """A stand-in for demo 0's Landscape with the shapes it logs: a 41 x 41
    grid, 3,200 triangles, the frame of 12 box edges and 9 floor lines each
    way, a path strip and the current point."""

    def __init__(self, grid=41):
        self.grid = grid
        a = torch.linspace(-1, 1, grid)
        x, y = torch.meshgrid(a, a, indexing="xy")
        self.xy = torch.stack([x.reshape(-1), y.reshape(-1)], 1)
        i = torch.arange(grid - 1)
        r, c = torch.meshgrid(i, i, indexing="ij")
        v0 = (r * grid + c).reshape(-1)
        self.triangles = torch.cat([torch.stack([v0, v0 + 1, v0 + grid], 1),
                                    torch.stack([v0 + 1, v0 + grid + 1, v0 + grid], 1)]).int()
        corners = torch.tensor([[x, y, z] for z in (-2.0, 1.0) for y in (-1.0, 1.0)
                                for x in (-1.0, 1.0)])
        edges = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        lines = [corners[list(e)] for e in edges]
        for t in torch.linspace(-1, 1, 9):
            lines += [torch.tensor([[t, -1.0, -2.0], [t, 1.0, -2.0]]),
                      torch.tensor([[-1.0, t, -2.0], [1.0, t, -2.0]])]
        self.frame = torch.cat(lines)
        self.points = []
        self.ready = False

    def point(self, step):
        p = torch.tensor([math.cos(step / 10) * 0.5, math.sin(step / 10) * 0.5, -0.5 + step / 1e3])
        self.points.append(p)
        return p

    def due(self, step, every):
        return step >= 3 and step % every == 0

    def refit(self, step):
        self.ready = True
        z = 0.5 * torch.log10(1.0 + self.xy.pow(2).sum(1) + step / 100)
        values = z.clamp(-4, 2)
        return torch.cat([self.xy, z[:, None]], 1), values, torch.tensor(0.5)

    @property
    def path(self):
        return torch.stack(self.points)

    @property
    def path_length(self):
        return len(self.points)


def test_the_demo_sequence_runs(server):
    scene = caliper.scene("landscape", up="+z", units="")
    every = caliper.param("every", 5, min=1, max=5000, help="steps between refits")
    span = caliper.param("span", 1.0, min=0.05, max=4.0, log=True, help="the box's half-width")
    directions = caliper.param("directions", "path", choices=["path", "random"], help="the plane")
    land = TinyLandscape()
    assert scene.lines("", land.frame, static=True)       # the box and the floor grid, once
    asked = []

    @caliper.action("surface now", help="refits now")
    def surface_now():
        asked.append(True)

    async def body():
        async with Viewer(server, caps=(w.CAP_SCENES, w.CAP_CONTROLS)) as v:
            await v.subscribe(1, "landscape")
            st = await v.state(1)
            assert [(m.path, m.part, m.static) for m in st] == [("", "lines", True)]
            assert arr(st[0], "positions").shape == (12 * 2 + 18 * 2, 3)
            for step in range(12):
                caliper.step()
                p = land.point(step)
                if land.due(step, every.value):
                    positions, values, share = land.refit(step)
                    assert scene.mesh("surface", positions, land.triangles, values=values,
                                      cmap="viridis", vmin=-4.0, vmax=2.0)
                    caliper.series(share, name="plane share")
                if land.ready:
                    scene.lines("path", land.path, strip_lengths=[land.path_length],
                                colors=(235, 235, 235), rate=10)
                    scene.points("current", p[None], colors=(255, 80, 80), rate=10)
                await asyncio.sleep(0.02)
            m = await v.update("surface", "mesh")
            assert arr(m, "positions").shape == (1681, 3) and arr(m, "indices").shape == (3200, 3)
            assert int(arr(m, "indices").max()) < 1681
            assert (m.params["vmin"], m.params["vmax"]) == (-4.0, 2.0)
            ln = await v.update("path", "lines")
            assert ln.params["mode"] == "strips" and ln.params["color"] == [235, 235, 235, 255]
            assert arr(ln, "strip_lengths").tolist() == [arr(ln, "positions").shape[0]]
            cur = await v.update("current", "points")
            assert arr(cur, "positions").shape == (1, 3)
            assert set(_scene.SCENES["landscape"].batches) == {
                ("", "lines"), ("surface", "mesh"), ("path", "lines"), ("current", "points")}
    run(body())
    pytest.importorskip("anywidget")
    assert caliper.widget("landscape", "scene", height=480).kind == "scene"
    every.remove()
    span.remove()
    directions.remove()
    surface_now.remove()


# ---------------------------------------------------------------------------
# cost with nothing attached, and no device wait on the training thread
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_a_call_with_nothing_attached_is_cheap(server, device):
    sc = caliper.scene("cost")
    pos = torch.rand(1681, 3, device=device)
    tri = torch.randint(0, 1681, (3200, 3), device=device, dtype=torch.int32)
    vals = torch.rand(1681, device=device)
    pc = time.perf_counter_ns
    meds = {}
    for name, fn in [("mesh", lambda: sc.mesh("surface", pos, tri, values=vals, vmin=-4.0, vmax=2.0)),
                     ("points", lambda: sc.points("current", pos[:1], colors=(255, 80, 80))),
                     ("lines", lambda: sc.lines("path", pos[:40], strip_lengths=[40]))]:
        ts = []
        for _ in range(2000):
            a = pc()
            assert fn() is False
            ts.append(pc() - a)
        meds[name] = statistics.median(ts) / 1e3
    print("scene calls, nothing attached, median us:", meds)
    assert all(m < 50.0 for m in meds.values()), meds
    assert _scene.SCENES["cost"].counters.get("taken", 0) == 0


def test_the_training_thread_never_waits_for_the_device():
    """caliper/_scene.py, which runs on the training thread, calls no
    .item(), .cpu(), .numpy(), .tolist() or synchronize() (the AST check of
    test_net.py for the ring and the data view engine)."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(_scene.__file__).read_text())
    bad = [(n.lineno, n.func.attr) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr in {"item", "cpu", "numpy", "tolist", "synchronize"}]
    assert bad == []
