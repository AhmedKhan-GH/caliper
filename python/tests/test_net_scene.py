"""caliper.scene on the network route (caliper/_scene.py, _net_scene.py): the
scenes subset of wire protocol 3.2 (draft), over a real loopback connection
with a websockets viewer, on both senders (the native one when
caliper._netio is built) and on the CPU and, where there is one, MPS:

* scene.mesh, scene.lines and scene.points reach a viewer with the right
  arrays, cast and bounded on the device;
* a late subscriber receives the static items and the current state, opened
  by SCENE_CLEAR with reset;
* a rate limit drops a batch and never a static one, and a viewer's
  max_rate holds each entity part to one batch per interval, the newest
  waiting;
* every refusal raises, attached or not, and the arguments that are not
  built yet (data_view, items, blend and the instance arrays) raise
  NotImplementedError;
* a viewer without the capability "scenes" sees no scene;
* the calls of demo 0 (the loss landscape) run in their order;
* a call with nothing attached takes no snapshot."""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
import warnings

import numpy as np
import pytest
import torch

import caliper
from caliper import _net_scene, _scene, _wire as w

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

    async def state(self, sub, n):
        """The scene's state for subscription ``sub``: the reset, then its ``n`` updates,
        waited for as events. A subscription's state goes before any other batch of it, so a
        test that must know no more state came logs a batch and waits for that."""
        clear = await self.expect(w.SceneClear, lambda m: m.sub == sub)
        assert clear.reset and clear.state and clear.path == ""
        out = [await self.expect(w.SceneUpdate, lambda m: m.sub == sub and m.state) for _ in range(n)]
        assert not [m for m in self.inbox if isinstance(m, w.SceneUpdate) and m.sub == sub and m.state]
        return out


async def until(pred, timeout=10.0):
    """Wait for published state: ``pred`` read every 2 ms until it holds."""
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise TimeoutError("the state the test waits for was not published")
        await asyncio.sleep(0.002)


class Clock:
    """The clock caliper/_scene.py reads for a batch's rate limit, set by the test, so that
    the limit is checked on a known sequence of calls and times rather than on how fast the
    machine makes the calls. Every other function of the time module is the real one."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


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
            assert w.scene_view_info(v.views["landscape"]) == {"space": "3d", "up": "+z", "units": "",
                                                                  "view_camera": None}
            await v.state(1, 0)
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
            await v.state(1, 0)
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
            await v.state(1, 0)
            sc.mesh("shape", tri, colors=[(10, 20, 30)] * 3)
            m = await v.update("shape")
            assert arr(m, "positions").shape == (3, 2)
            assert m.params["stats"] == [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0]
            assert arr(m, "colors").tolist() == [[10, 20, 30, 255]] * 3
    run(body())


def test_points_and_segments_over_the_budget_are_sampled_by_stride(server):
    sc = caliper.scene("big")
    n = 150_000
    pos = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "big")
            await v.state(1, 0)
            sc.points("cloud", pos, values=torch.arange(n, dtype=torch.float32))
            m = await v.update("cloud", timeout=20)
            size = w.DEFAULT_MAX_POINTS
            assert m.axes == [{"dim": "point", "start": 0, "step": 1, "count": n, "size": size}]
            rows = np.arange(size, dtype=np.int64) * n // size
            np.testing.assert_array_equal(arr(m, "values"), rows.astype(np.float32))
            np.testing.assert_array_equal(arr(m, "positions"), pos.numpy()[rows])
            seg = torch.arange(2 * n * 3, dtype=torch.float32).reshape(2 * n, 3)
            sc.lines("edges", seg)                                     # n segments
            e = await v.update("edges", timeout=20)
            size = w.SCENE_DEFAULT_MAX_ITEMS
            assert e.axes == [{"dim": "segment", "start": 0, "step": 1, "count": n, "size": size}]
            keep = np.arange(size, dtype=np.int64) * n // size
            pairs = np.stack([2 * keep, 2 * keep + 1], 1).reshape(-1)
            np.testing.assert_array_equal(arr(e, "positions"), seg.numpy()[pairs])
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
            st = await a.state(1, 1)
            assert [(m.path, m.part, m.static) for m in st] == [("", "lines", True)]
            np.testing.assert_array_equal(arr(st[0], "positions"), frame.cpu().numpy())
            surf = torch.rand(6, 3).to(device)
            assert sc.mesh("surface", surf, values=torch.arange(6.0).to(device))
            assert sc.points("current", torch.ones(1, 3).to(device), colors=(255, 80, 80))
            await a.update("surface")
            await a.update("current")
            async with Viewer(server) as b:
                await b.subscribe(7, "world")
                st = await b.state(7, 3)
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
                st = await v.state(1, 1)
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
            st = await v.state(1, 1)
            assert len(st) == 1 and arr(st[0], "positions").shape == (4, 3)
    run(body())


def test_rate_limit_drops_a_batch_and_never_a_static_one(server, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(_scene, "time", clock)
    sc = caliper.scene("rates", rate=2)                                   # 0.5 s per batch

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "rates")
            await v.state(1, 0)
            got = [sc.points("p", torch.full((1, 3), float(i))) for i in range(20)]
            assert got[0] is True and not any(got[1:])                   # 20 calls at one time
            assert all(sc.points("q", torch.zeros(1, 3), static=True) for _ in range(5))
            assert sc.points("r", torch.zeros(1, 3), rate=0) and sc.points("r", torch.zeros(1, 3), rate=0)
            counters = _scene.SCENES["rates"].counters
            assert counters["rate_limited"] == 19
            assert arr(await v.update("p"), "positions")[0, 0] == 0.0     # the first one is sent
            await v.update("q")
            await v.update("r")
            clock.now += 0.25
            assert sc.points("p", torch.full((1, 3), 20.0)) is False      # within the interval
            clock.now += 0.25
            assert sc.points("p", torch.full((1, 3), 21.0)) is True       # the next interval
            assert arr(await v.update("p"), "positions")[0, 0] == 21.0
            assert counters["rate_limited"] == 20 and counters["taken"] == 1 + 5 + 2 + 1
    run(body())


def test_unwatched_batches_cost_no_snapshot(server):
    sc = caliper.scene("idle")
    assert sc.mesh("m", torch.zeros(3, 3)) is False
    assert _scene.SCENES["idle"].counters.get("taken", 0) == 0


def test_max_rate_applies_to_each_entity_part(server):
    """A subscription's max_rate holds each entity part to one batch per 1/max_rate, and the
    batches logged meanwhile wait as one, the newest. Nothing here depends on how fast the
    machine is: the slow subscription's interval (1,000 s) is far longer than the test, so
    after its first batch of each part the 39 that follow can only wait as one; the paced one
    (4 a second) receives each part in order, the newest last, and every batch it does not
    receive was replaced by a newer one."""
    sc = caliper.scene("pace", rate=0)
    route = _net_scene.route_of(server.ring)

    def value(m):
        return float(arr(m, "positions")[0, 0])

    async def body():
        async with Viewer(server) as paced, Viewer(server) as slow:
            await paced.subscribe(1, "pace", max_rate=4)
            await slow.subscribe(2, "pace", max_rate=0.001)
            await paced.state(1, 0)
            await slow.state(2, 0)
            for p in ("a", "b"):
                assert sc.points(p, torch.zeros(1, 3))
            for v in (paced, slow):                                   # the first goes at once
                for p in ("a", "b"):
                    assert value(await v.update(p)) == 0.0
            for i in range(1, 40):
                for p in ("a", "b"):
                    assert sc.points(p, torch.full((1, 3), float(i)))
            got = {}
            for p in ("a", "b"):
                got[p] = [0.0]
                while got[p][-1] != 39.0:
                    got[p].append(value(await paced.update(p)))
                assert all(x < y for x, y in zip(got[p], got[p][1:])), got[p]   # newer each time
            links = {sub: (link, q) for (link, sub), q in route.subs["pace"].items()}
            link, q = links[2]

            def newest_waits():
                return all(k in q.pending and float(q.pending[k].arrays["positions"][0, 0]) == 39.0
                           for k in (("a", w.PART_POINTS), ("b", w.PART_POINTS)))

            await until(newest_waits)                         # the slow one's 39 wait as one
            assert link.stats["scene_superseded"] == 2 * 38
            assert links[1][0].stats["scene_superseded"] == 80 - len(got["a"]) - len(got["b"])
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
            await v.state(1, 0)
            assert sc.points("p", torch.zeros(1, 3), static=True)
            assert caliper.remove("gone") is False
            await v.expect(w.Views, lambda m: all(x["name"] != "gone" for x in m.views))
            assert "gone" not in _scene.SCENES
            # A call on its object makes it again, empty, as a view call makes its view.
            assert sc.points("q", torch.zeros(1, 3)) is False       # nobody subscribes yet
            await v.expect(w.Views, lambda m: any(x["name"] == "gone" for x in m.views))
            await v.subscribe(2, "gone")
            assert await v.state(2, 0) == []
            assert sc.points("q", torch.ones(1, 3))                 # watched now
            assert (await v.update("q")).sub == 2
            # A subscription's state goes before its other batches: none came before q, so the
            # static item went with the scene.
            assert not [m for m in v.inbox if isinstance(m, w.SceneUpdate) and m.sub == 2]
            caliper.scene("other")
            caliper.remove("other")
            again = caliper.scene("other")
            with pytest.raises(ValueError, match="made again"):
                stale = _scene.Scene(_scene.SceneState("other", "3d", "+y", "m", None))
                stale.points("p", torch.zeros(1, 3))
            assert again.points("p", torch.zeros(1, 3)) is False
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
    with pytest.raises(NotImplementedError, match=re.escape("caliper.scene(data_view=...) is "
                                                            "not built yet")):
        caliper.scene("s", data_view={})
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
    ]
    for exc, call in bad:
        with pytest.raises(exc):
            call()
    later = [
        (lambda: sc.points("p", p3, items="test"), "items="),
        (lambda: sc.points("p", p3, blend="additive"), "blend="),
        (lambda: sc.lines("l", p3, blend="additive"), "blend="),
        (lambda: sc.boxes("b", p3, p3, items="test"), "items="),
        (lambda: sc.mesh("m", p3[:3], instance_poses=torch.zeros(2, 4, 4)), "instance_poses="),
        (lambda: sc.mesh("m", p3[:3], instance_colors=torch.zeros(2, 4)), "instance_colors="),
        (lambda: sc.mesh("m", p3[:3], instance_values=torch.zeros(2)), "instance_values="),
    ]
    for call, what in later:
        with pytest.raises(NotImplementedError, match="^" + re.escape(what + " is not built yet (")):
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
            st = await v.state(1, 1)
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
def test_a_call_with_nothing_attached_is_cheap(server, device, monkeypatch):
    """With nothing attached a call runs its checks and returns False: no snapshot, nothing
    queued, no device work. What that costs in time is a benchmark row, not a test:
    python/bench/bench_call_cost.py --scenes, outside CI (docs/development/performance.md,
    "Python Calls")."""
    sc = caliper.scene("cost")
    pos = torch.rand(1681, 3, device=device)
    tri = torch.randint(0, 1681, (3200, 3), device=device, dtype=torch.int32)
    vals = torch.rand(1681, device=device)

    def no_snapshot(*a, **kw):
        raise AssertionError("a call with nothing attached took a snapshot")

    monkeypatch.setattr(_scene, "_snapshot", no_snapshot)
    route = _net_scene.route_of(server.ring)
    queued = route.counters["queued"]
    calls = [lambda: sc.mesh("surface", pos, tri, values=vals, vmin=-4.0, vmax=2.0),
             lambda: sc.points("current", pos[:1], colors=(255, 80, 80)),
             lambda: sc.lines("path", pos[:40], strip_lengths=[40])]
    for _ in range(100):
        for fn in calls:
            assert fn() is False
    counters = _scene.SCENES["cost"].counters
    assert counters["unwatched"] == 300 and counters.get("taken", 0) == 0
    assert route.counters["queued"] == queued


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


def test_a_viewer_without_scenes_gets_unknown_view_whatever_its_data_view(server):
    """The capability goes before the data view: a 3.1 viewer that names a scene with a data
    view the scene refuses still gets unknown_view, so nothing tells it that the view exists."""
    caliper.scene("hidden")
    caliper.series(1.0, name="loss")

    async def body():
        async with Viewer(server, caps=()) as v:
            await v.expect(w.Views, lambda m: any(x["name"] == "loss" for x in m.views))
            v.views["hidden"] = {}
            r = await v.subscribe(1, "hidden", data_view={"entities": {"include": ["a/**"], "exclude": []}})
            assert r.error["code"] == w.ERR_UNKNOWN_VIEW
    run(body())


def test_a_hello_without_capabilities_is_a_3_1_viewer(server):
    """A HELLO that omits the optional key "capabilities" completes the handshake, and its
    peer is a 3.1 viewer: no scene in VIEWS, unknown_view for a SUBSCRIBE to one."""
    caliper.scene("hidden")
    caliper.series(1.0, name="loss")

    async def body():
        async with connect(server.stream_url, max_size=None, compression=None, ping_interval=None,
                           close_timeout=1) as ws:
            await ws.send(w.encode_frame(w.MsgType.HELLO, {"role": w.ROLE_VIEWER}))
            welcome = w.decode(await asyncio.wait_for(ws.recv(), 10))
            assert isinstance(welcome, w.Welcome) and welcome.role == w.ROLE_PRODUCER
            names: set = set()
            while "loss" not in names:
                m = w.decode(await asyncio.wait_for(ws.recv(), 10))
                if isinstance(m, w.Views):
                    names = {x["name"] for x in m.views}
            assert "hidden" not in names
            await ws.send(w.encode(w.Subscribe(1, "hidden", 101, None, None)))
            while True:
                m = w.decode(await asyncio.wait_for(ws.recv(), 10))
                if isinstance(m, w.ViewReply) and m.request_id == 101:
                    break
            assert m.error["code"] == w.ERR_UNKNOWN_VIEW
    run(body())


def test_strip_lengths_that_do_not_add_up_drop_the_batch_when_finished(monkeypatch):
    """Strip lengths in a device tensor are checked on the network thread once their copy is
    done (_net_scene.finish): lengths that do not add up to the vertices, or a length below 1,
    drop the batch with a warning instead of sending a frame that every receiver refuses. The
    batches here are made from CPU tensors past the call's own check, as an MPS call makes them."""
    monkeypatch.setenv("CALIPER_SESSION", "0")
    sc = caliper.scene("strips")
    pos = torch.zeros(5, 3)

    def batch(lengths):
        spec = _scene._Spec(w.PART_LINES, pos, None, None, 0, None, None,
                            {"mode": w.LINES_STRIPS, "width": 2.0},
                            extra={"strip_lengths": torch.tensor(lengths, dtype=torch.int32)})
        return _scene._snapshot(sc._s, "path", spec, False)

    for lengths in ([2, 2], [3, 3], [5, 0], []):
        b = batch(lengths)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert _net_scene.finish(b) is False, lengths
        assert b.arrays is None
        assert any("strip lengths" in str(r.message) for r in rec), lengths
    b = batch([2, 3])
    assert _net_scene.finish(b) is True
    assert list(b.arrays["strip_lengths"]) == [2, 3]


@pytest.mark.skipif(not MPS, reason="no MPS device")
def test_device_strip_lengths_that_do_not_add_up_never_reach_a_viewer(server):
    sc = caliper.scene("strips", up="+z")
    pos = torch.zeros(5, 3, device="mps")

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "strips")
            await v.state(1, 0)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                assert sc.lines("bad", pos, strip_lengths=torch.tensor([2, 2], device="mps")) is True
                assert sc.lines("good", pos, strip_lengths=torch.tensor([2, 3], device="mps")) is True
                m = await v.update("good")
            assert list(arr(m, "strip_lengths")) == [2, 3]
            assert not [x for x in v.inbox if isinstance(x, w.SceneUpdate) and x.path == "bad"]
            assert any("strip lengths" in str(r.message) for r in rec)
    run(body())


def test_native_pump_waits_until_a_rate_limited_scene_frame_is_due():
    """The native sender's pump polls every POLL_MS while copies are in flight or a scene frame
    waits for room in its link (pump_native's 4 ms), and otherwise waits until the next
    rate-limited scene frame is due, at most IDLE_MS: a batch that waits 1,000 s for its
    max_rate does not wake the pump, and take the GIL, every 4 ms."""
    import types
    from caliper import _net_native as nn
    sc = types.SimpleNamespace(pending=[], native_wait=None)
    p = nn._NativeProducer.__new__(nn._NativeProducer)
    p.ring = types.SimpleNamespace(pending=[], series={}, scenes=sc)
    p.controls = None
    cases = ((None, False, nn.IDLE_MS), (1000.0, False, nn.IDLE_MS), (0.05, False, 50),
             (0.0101, False, 11), (0.004, True, nn.POLL_MS), (0.0, True, nn.POLL_MS))
    for wait, busy, _ in cases:
        sc.native_wait = wait
        assert p.busy() is busy, wait
    for wait, _, ms in cases:
        sc.native_wait = wait
        assert p.wait_ms() == ms, wait
    sc.native_wait = 1000.0
    sc.pending = [object()]                          # a copy in flight
    assert p.busy() is True and p.wait_ms() == nn.POLL_MS


# ---------------------------------------------------------------------------
# the native sender: scene frames up to the protocol's frame limit
# ---------------------------------------------------------------------------
class RawViewer:
    """A viewer on a bare socket with a small receive buffer, so that it can stop reading and
    make the sender queue frames for it. It reads the WebSocket frames itself."""

    def __init__(self, server, rcvbuf=64 << 10):
        import base64
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        s.settimeout(20)
        s.connect(("127.0.0.1", server.port))
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET /v3/stream?token={server.token} HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += s.recv(1)
        assert b" 101 " in head.split(b"\r\n")[0]
        self.s = s
        self.close_code = None

    def send(self, msg) -> None:
        import struct
        payload = w.encode(msg)
        n = len(payload)
        hdr = bytes((0x82, 0x80 | n)) if n < 126 else bytes((0x82, 0x80 | 126)) + struct.pack("!H", n)
        mask = os.urandom(4)
        m = np.frombuffer(mask * (n // 4 + 1), np.uint8)[:n]
        self.s.sendall(hdr + mask + (np.frombuffer(payload, np.uint8) ^ m).tobytes())

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            b = self.s.recv(min(n - len(out), 1 << 20))
            if not b:
                raise ConnectionError("the server closed the socket")
            out += b
        return bytes(out)

    def next(self):
        """The next wire message, or None once the server closed the connection (close_code)."""
        import struct
        b0, b1 = self._read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self._read(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self._read(8))[0]
        payload = self._read(n)
        op = b0 & 0x0F
        if op == 0x8:
            self.close_code = struct.unpack("!H", payload[:2])[0] if n >= 2 else 1005
            return None
        if op in (0x9, 0xA):
            return self.next()
        assert op == 0x2 and b0 & 0x80, f"opcode {op}"
        return w.decode(payload)

    def expect(self, pred, what: str):
        while True:
            m = self.next()
            assert m is not None, f"closed with {self.close_code} waiting for {what}"
            if pred(m):
                return m

    def close(self) -> None:
        self.s.close()


def _native_scene_frames(server, big_mib: float, waiting_mib: float) -> None:
    """One mesh frame of about ``big_mib`` MiB reaches a viewer that reads; then, with the viewer
    not reading, three of about ``waiting_mib`` MiB each, the first in the socket and two waiting
    in the native sender's queue, reach it once it reads again."""
    def mesh_rows(mib):
        return max(3, int(mib * (1 << 20) / 12) // 3 * 3)  # (V, 3) f32 positions, V a multiple of 3

    sc = caliper.scene("frames", up="+z")
    v = RawViewer(server)
    try:
        v.send(w.Hello("viewer", [w.CAP_SCENES]))
        v.expect(lambda m: isinstance(m, w.Views) and any(x["name"] == "frames" for x in m.views), "VIEWS")
        v.send(w.Subscribe(1, "frames", 101))
        assert v.expect(lambda m: isinstance(m, w.ViewReply), "the reply").error is None
        v.expect(lambda m: isinstance(m, w.SceneClear) and m.reset, "the state")
        n = mesh_rows(big_mib)
        assert sc.mesh("big", torch.zeros(n, 3)) is True
        m = v.expect(lambda m: isinstance(m, w.SceneUpdate) and m.path == "big", "the big mesh")
        assert arr(m, "positions").shape == (n, 3)
        # The viewer stops reading while three frames go out.
        links = list(server.producer.links)
        sent = sum(ln.stats["scenes_sent"] for ln in links)
        n = mesh_rows(waiting_mib)
        for path in ("a", "b", "c"):
            assert sc.mesh(path, torch.full((n, 3), 0.5)) is True
        end = time.monotonic() + 20
        while sum(ln.stats["scenes_sent"] for ln in links) < sent + 3 and time.monotonic() < end:
            time.sleep(0.005)
        got = []
        while len(got) < 3:
            m = v.expect(lambda m: isinstance(m, w.SceneUpdate), "the waiting meshes")
            got.append(m.path)
            assert arr(m, "positions").shape == (n, 3)
        assert got == ["a", "b", "c"]
    finally:
        v.close()


@pytest.mark.skipif(not NATIVE, reason="caliper._netio is not built (python scripts/build_netio.py)")
def test_native_sender_sends_scene_frames_past_its_control_cap(monkeypatch):
    """The native sender's scene frames are bounded by the protocol's frame limit and its own
    backlog of two, not by the 16 MiB cap on a connection's control replies: a mesh frame over
    16 MiB, and two waiting frames over 8 MiB each, reach the viewer."""
    from caliper import _netio
    assert _netio.MAX_SCENE_BYTES == _net_scene.NATIVE_BACKLOG * w.MAX_FRAME_BYTES
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1")
    server = caliper.serve(port=0)
    try:
        assert type(server).__name__ == "NativeServer"
        _native_scene_frames(server, 17, 10)
    finally:
        server.close()


def test_pump_native_counts_only_the_frames_a_connection_took():
    """send() returns False for a connection that is closing (its queue overflowed, or it
    went): pump_native counts that frame as refused, not as sent, and gives the link nothing
    more this round."""
    import collections
    import types

    class Nio:
        def __init__(self):
            self.calls = 0

        def queued(self, conn):
            return 0

        def send(self, conn, frame, scene=False):
            self.calls += 1
            return False

    q = _net_scene.SubQueue(1, "s", 0.0)
    for _ in range(3):
        q.state.append(w.SceneClear(1, "s", 0, 0, "", reset=True, state=True))
    scenes = _net_scene.LinkScenes()
    scenes.subs[1] = q
    nio = Nio()
    link = types.SimpleNamespace(scenes=scenes, conn=7, p=types.SimpleNamespace(nio=nio),
                                 stats=collections.Counter())
    _net_scene.route_of(types.SimpleNamespace()).pump_native([link], time.monotonic())
    assert nio.calls == 1
    assert link.stats["scenes_sent"] == 0
    assert link.stats["scenes_refused"] == 1
