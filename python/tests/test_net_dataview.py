"""Data views over the network route: caliper.serve() with caliper._dataview's
engine, tested with a websockets client and the codec caliper/_wire.py. The
offer in VIEWS, one head of a 4-D attention tensor, reductions and tiles,
two subscribers with different data views, bad requests that never raise on
the training thread, watch()'s groups switching source, budgets, offer() and
update(), and the default data view set from Python. MPS cases skip without
an MPS device."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper
from caliper import _dataview as dv
from caliper import _wire as w
from test_net import Client, run  # noqa: F401  (the viewer client of test_net.py)

MPS = torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="no MPS device"))]
ATTN_DIMS = ("layer", "sequence", "head", "query", "key")
ATTN_DEFAULT = {"index": {"layer": 1, "sequence": 0, "head": 0}}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    s = caliper.serve(port=0)
    yield s
    s.close()
    dv.REGISTRY.clear()


def views_entry(c, name, pred=lambda v: bool(v["sources"]), pump=None):
    """The entry of view ``name`` in the first VIEWS message where ``pred`` holds."""
    async def go():
        m = await c.expect(w.Views, pump=pump, pred=lambda m: any(
            v["name"] == name and pred(v) for v in m.views))
        return next(v for v in m.views if v["name"] == name)
    return go()


def arr(s):
    return s.data.array()


@pytest.mark.parametrize("device", DEVICES)
def test_offer_one_head_reduce_and_tile(server, device):
    attn = torch.randn(2, 3, 4, 24, 40, device=device)
    show = lambda: caliper.show(attn, "attention", rate=0, dims=ATTN_DIMS,  # noqa: E731
                                data_view=ATTN_DEFAULT)
    ref = attn.cpu().numpy()

    async def go():
        async with Client(server) as c:
            show()
            view = await views_entry(c, "attention")
            assert view["kind"] == w.Kind.HEATMAP
            assert view["sources"] == [{"name": "attention", "dtype": "f32", "dims": [
                {"name": n, "size": s} for n, s in zip(ATTN_DIMS, attn.shape)]}]
            assert view["default"] == {"source": "attention", "index": ATTN_DEFAULT["index"],
                                       "reduce": {}, "budget": {"max_extent": 512, "pool": "mean"}}
            # {} is the view's default
            r = await c.subscribe(1, "attention", {}, pump=show)
            assert r.data_view == view["default"]
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 1)
            assert np.array_equal(arr(s), ref[1, 0, 0])
            # one head
            await c.send(w.ViewRequest(2, 1, "attention", {"index": {"layer": 0, "sequence": 1,
                                                                     "head": -1}}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 2, pump=show)
            assert rr.data_view["index"] == {"layer": 0, "sequence": 1, "head": 3}
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data_view == rr.data_view)
            assert np.array_equal(arr(s), ref[0, 1, 3])
            assert s.axes == [{"dim": "query", "start": 0, "step": 1, "count": 24, "size": 24},
                              {"dim": "key", "start": 0, "step": 1, "count": 40, "size": 40}]
            lo, hi = s.stats
            assert lo == pytest.approx(ref[0, 1, 3].min()) and hi == pytest.approx(ref[0, 1, 3].max())
            # a reduction over the heads
            await c.send(w.ViewRequest(3, 1, "attention", {"index": {"layer": 1, "sequence": 0},
                                                           "reduce": {"head": "max"}}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 3, pump=show)
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data_view == rr.data_view)
            assert np.allclose(arr(s), ref[1, 0].max(0))
            # the heads tiled as a mosaic
            await c.send(w.ViewRequest(4, 1, "attention", {"index": {"layer": 0, "sequence": 0},
                                                           "tile": "head"}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 4, pump=show)
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data_view == rr.data_view)
            assert s.tile == {"dim": "head", "rows": 2, "cols": 2}
            a = arr(s)
            assert a.shape == (48, 80) and np.array_equal(a[24:, :40], ref[0, 0, 2])
            assert s.stats == [pytest.approx(ref[0, 0].min()), pytest.approx(ref[0, 0].max())]
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_two_subscribers_see_different_heads(server, device):
    attn = torch.randn(2, 1, 4, 16, 16, device=device)
    show = lambda: caliper.show(attn, "attention", rate=0, dims=ATTN_DIMS,  # noqa: E731
                                data_view={"index": {"layer": 0, "sequence": 0, "head": 0}})
    ref = attn.cpu().numpy()

    async def go():
        async with Client(server) as a, Client(server) as b:
            show()
            await views_entry(a, "attention")
            await views_entry(b, "attention")
            await a.subscribe(1, "attention", {"index": {"layer": 1, "sequence": 0, "head": 1}},
                              pump=show)
            await b.subscribe(1, "attention", {"index": {"layer": 0, "sequence": 0, "head": 3}},
                              pump=show)
            sa = await a.expect(w.Snapshot, pump=show)
            sb = await b.expect(w.Snapshot, pump=show)
            assert np.array_equal(arr(sa), ref[1, 0, 1]) and np.array_equal(arr(sb), ref[0, 0, 3])
            st = server.stats()["views"]["attention"]
            assert st["groups"] == 2
    run(go())


def test_bad_requests_never_raise_on_the_training_thread(server):
    x = torch.randn(4, 8, 8)
    calls = []

    def show():
        calls.append(caliper.show(x, "act", rate=0, dims=("batch", "y", "x")))

    async def go():
        async with Client(server) as c:
            show()
            await views_entry(c, "act")
            for i, req in enumerate(({"index": {"nope": 0}}, {"index": {"batch": 4}},
                                     {"index": {}}, {"reduce": {"batch": "median"}},
                                     {"tile": "y"}, {"source": "other"},
                                     {"budget": {"max_extent": 0}})):
                await c.send(w.Subscribe(10 + i, "act", 10 + i, None, req))
                r = await c.expect(w.ViewReply, pred=lambda m, i=i: m.request_id == 10 + i, pump=show)
                assert r.error["code"] == w.ERR_BAD_DATA_VIEW, req
                assert len(r.error["message"]) < 300
            r = await c.subscribe(1, "act", {"index": {"batch": 3}}, pump=show)
            s = await c.expect(w.Snapshot, pump=show)
            assert np.array_equal(arr(s), x[3].numpy())
            # the source shrinks: the subscription no longer fits and gets nothing, quietly
            nonlocal_x = x[:2].clone()
            for _ in range(20):
                caliper.show(nonlocal_x, "act", rate=0, dims=("batch", "y", "x"))
            assert caliper.flush(timeout=5.0) == 0
            c.inbox.clear()
            for _ in range(5):
                caliper.show(nonlocal_x, "act", rate=0, dims=("batch", "y", "x"))
            # and back: snapshots again
            s = await c.expect(w.Snapshot, pump=show)
            assert np.array_equal(arr(s), x[3].numpy())
    run(go())
    assert calls and all(r in (True, False) for r in calls)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.fc1 = nn.Linear(6 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


@pytest.mark.parametrize("device", DEVICES)
def test_watch_groups_switch_source(server, device, monkeypatch):
    torch.manual_seed(0)
    net = Net().to(device)
    x = torch.randn(4, 1, 16, 16, device=device)
    applied = []
    real = dv._apply

    def spy(t, p, pre=False):
        applied.append(p.resolved["source"])
        return real(t, p, pre)
    monkeypatch.setattr(dv, "_apply", spy)

    def step():
        net(x).square().mean().backward()

    h = caliper.watch(net, gradients=True)
    try:
        async def go():
            async with Client(server) as c:
                step()
                acts = await views_entry(c, "activations", lambda v: len(v["sources"]) == 3,
                                         pump=step)
                assert [s["name"] for s in acts["sources"]] == ["conv1", "fc1", "fc2"]
                assert acts["default"]["index"] == {"batch": 0, "channel": 0}
                await c.subscribe(1, "activations", {"source": "fc1", "index": {}}, pump=step)
                s = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 1)
                with torch.no_grad():
                    f1 = net.fc1(F.max_pool2d(F.relu(net.conv1(x)), 2).flatten(1))
                assert np.allclose(arr(s), f1.cpu().numpy(), atol=1e-6)
                # nothing computed the sources nobody subscribed to
                time.sleep(0.03)                        # past the subscription's 60 Hz rate
                applied.clear()
                step()
                assert set(applied) == {"fc1"}
                # switch the source, and add a second subscriber on another weight
                await c.send(w.ViewRequest(2, 1, "activations",
                                           {"source": "conv1", "index": {"batch": 1, "channel": 2}}))
                rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 2, pump=step)
                s = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.data_view == rr.data_view)
                with torch.no_grad():
                    c1 = net.conv1(x)
                assert np.allclose(arr(s), c1[1, 2].cpu().numpy(), atol=1e-6)
                await c.subscribe(3, "weights", {"source": "fc2.weight"}, pump=step)
                await c.subscribe(4, "weights", {"source": "conv1.weight"}, pump=step)
                s3 = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 3)
                s4 = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 4)
                assert np.array_equal(arr(s3), net.fc2.weight.detach().cpu().numpy())
                assert np.array_equal(arr(s4), net.conv1.weight.detach().reshape(6, 9).cpu().numpy())
                await c.subscribe(5, "gradients", {"source": "fc1.weight"}, pump=step)
                s5 = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 5)
                assert s5.data.array().shape == (16, 384)
        run(go())
    finally:
        h.remove()


@pytest.mark.parametrize("device", DEVICES)
def test_budgets_and_zoom(server, device):
    big = torch.randn(1000, 777, device=device)
    z = torch.randn(3, 5000, 2, device=device)
    labels = (torch.arange(5000, device=device) % 5).float()
    plain = torch.randn(300, 400, device=device)

    def step():
        caliper.show(big, "big", rate=0, dims=("out", "in"))
        caliper.points(z, labels, "emb", classes=5, rate=0, dims=("layer", "point"))
        caliper.show(plain, "plain", rate=0)

    async def go():
        async with Client(server) as c:
            step()
            await views_entry(c, "emb")
            await c.subscribe(1, "big", {"budget": {"max_extent": 100, "pool": "max"}}, pump=step)
            s = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 1)
            ref = F.adaptive_max_pool2d(big.cpu()[None, None], (100, 100))[0, 0].numpy()
            assert np.array_equal(arr(s), ref)
            assert s.axes[1] == {"dim": "in", "start": 0, "step": 1, "count": 777, "size": 100}
            await c.subscribe(2, "emb", {"index": {"layer": 2}, "budget": {"max_points": 1000}},
                              pump=step)
            s = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 2)
            rows = np.arange(1000) * 5000 // 1000
            assert np.array_equal(arr(s), z[2].cpu().numpy()[rows])
            assert np.array_equal(s.aux.array(), (rows % 5).astype(np.float32))
            assert s.points_dims == 2 and s.label_count == 5
            # a view without dims: its payload is the source "data", and a range zooms
            await c.subscribe(3, "plain", {"index": {"row": [100, 200], "col": [0, 400, 4]}},
                              pump=step)
            s = await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 3)
            assert np.array_equal(arr(s), plain.cpu().numpy()[100:200, ::4])
    run(go())


def test_offer_and_update_fetch_only_what_is_subscribed(server):
    calls = {"a": 0, "b": 0}
    ta, tb = torch.randn(4, 6), torch.randn(5, 7)

    def make(name, t):
        def get():
            calls[name] += 1
            return t
        return get

    caliper.offer("probe", {"a": make("a", ta), "b": make("b", tb)}, dims=("row", "col"))

    async def go():
        async with Client(server) as c:
            caliper.update()
            await views_entry(c, "probe")
            await c.subscribe(1, "probe", {"source": "b"}, pump=caliper.update)
            s = await c.expect(w.Snapshot, pump=caliper.update)
            assert np.array_equal(arr(s), tb.numpy())
            before = dict(calls)
            for _ in range(5):
                time.sleep(0.03)                   # past the subscription's 60 Hz rate
                caliper.update()
            assert calls["a"] == before["a"] and calls["b"] > before["b"]
    run(go())


def test_set_data_view_is_the_default_of_new_subscriptions(server):
    attn = torch.randn(2, 1, 4, 8, 8)
    show = lambda: caliper.show(attn, "attention", rate=0, dims=ATTN_DIMS,  # noqa: E731
                                data_view=ATTN_DEFAULT)

    async def go():
        async with Client(server) as c:
            show()
            await views_entry(c, "attention")
            await c.subscribe(1, "attention", {}, pump=show)
            rv = caliper.set_data_view("attention", {"index": {"layer": 0, "sequence": 0,
                                                               "head": 2}})
            m = await c.expect(w.Views, pump=show, pred=lambda m: any(
                v.get("default", {}).get("index", {}).get("head") == 2 for v in m.views))
            assert next(v for v in m.views if v["name"] == "attention")["default"] == rv
            r2 = await c.subscribe(2, "attention", {}, pump=show)
            assert r2.data_view == rv
            s1 = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 1)
            s2 = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 2)
            assert np.array_equal(arr(s1), attn[1, 0, 0].numpy())      # keeps its own
            assert np.array_equal(arr(s2), attn[0, 0, 2].numpy())
    run(go())
