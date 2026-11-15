"""Data views (caliper._dataview): resolution by the wire protocol's rules,
including hostile requests; the payload layout; application on the CPU and on
MPS against numpy implementations of the same rules; the registry under
concurrent requests; show/points/histogram with dims, offer/update and
watch's grouped views through a fake transport; and the no-readback rule,
checked in the source and with a busy MPS queue. The timings of apply on MPS
print as a table."""
import ast
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper
from caliper import _dataview as dv
from caliper._dataview import DataViewError, Offer, SourceSpec
from conftest import mps_available, requires_mps

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])
PKG = Path(caliper.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    dv.REGISTRY.clear()


class FakeTransport:
    """A Transport over plain torch tensors that records what it publishes."""

    def __init__(self):
        self.published, self.removed = [], []

    def attached(self):
        return True

    def alloc(self, name, spec, slot, role, shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=spec.device)

    def reading_slot(self, name):
        return -1

    def publish(self, name, slot, meta):
        self.published.append({
            "name": name, "meta": meta, "payload": slot.payload.clone(),
            "aux": None if slot.aux is None else slot.aux.clone(),
            "data_view": getattr(slot, "data_view", None), "axes": getattr(slot, "axes", None),
            "tile": getattr(slot, "tile", None)})

    def remove(self, name):
        self.removed.append(name)

    def last(self, name):
        assert caliper.flush(timeout=5.0) == 0
        got = [p for p in self.published if p["name"] == name]
        return got[-1] if got else None


@pytest.fixture
def fake():
    tr = FakeTransport()
    route = caliper._attach_transport(tr, rate=0)       # every call snapshots
    yield tr
    caliper._detach_transport(route)


def offer(kind="heatmap", default=None, **sources):
    """An Offer from name=(dims, sizes) pairs."""
    specs = tuple(SourceSpec(n, tuple(d), tuple(s), "f32") for n, (d, s) in sources.items())
    return Offer(kind, specs, default, "v")


ATTN = offer(attention=(("layer", "head", "query", "key"), (12, 8, 64, 48)),
             default={"index": {"layer": 0, "head": 0}})


def npy(t):
    return t.detach().to("cpu").numpy()


# ---------------------------------------------------------------------------
# numpy implementations of the protocol's rules (the references)
# ---------------------------------------------------------------------------
def np_windows(count, size):
    return [(i * count // size, -((-(i + 1) * count) // size)) for i in range(size)]


def np_pool(a, axis, size, op):
    count = a.shape[axis]
    if size >= count:
        return a
    f = {"mean": np.mean, "max": np.max, "min": np.min}[op]
    return np.concatenate([f(np.take(a, range(s, e), axis=axis), axis=axis, keepdims=True)
                           for s, e in np_windows(count, size)], axis=axis)


def np_mosaic(x):
    c, h, w = x.shape
    cols = math.isqrt(c - 1) + 1
    rows = -(-c // cols)
    out = np.full((rows * h, cols * w), np.nan, dtype=x.dtype)
    for t in range(c):
        r, k = divmod(t, cols)
        out[r * h:(r + 1) * h, k * w:(k + 1) * w] = x[t]
    return out


def np_reference(x, off, rv):
    """Index, reduce one dim at a time in source order, tile, pool or sample."""
    kind = off.kind
    src = off.source(rv["source"])
    names = list(src.dims)
    x = np.asarray(x, dtype=np.float64)
    for d in src.dims:
        v = rv["index"].get(d)
        if v is None:
            continue
        ax = names.index(d)
        if isinstance(v, int):
            x = np.take(x, v, axis=ax)
            names.remove(d)
        else:
            x = np.take(x, range(*v), axis=ax)
    for d in src.dims:
        op = rv["reduce"].get(d)
        if op is None:
            continue
        ax = names.index(d)
        x = (np.sqrt((x ** 2).sum(axis=ax)) if op == "norm" else
             {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min}[op](x, axis=ax))
        names.remove(d)
    lay = dv.layout(off, rv)
    if kind == "heatmap":
        op = rv["budget"]["pool"]
        if rv.get("tile") is not None:
            t, r, c = lay["axes"]
            x = np.moveaxis(x, [names.index(t["dim"]), names.index(r["dim"]),
                                names.index(c["dim"])], [0, 1, 2])
            x = np_pool(np_pool(x, 1, r["size"], op), 2, c["size"], op)
            return np_mosaic(x)
        r, c = lay["axes"]
        return np_pool(np_pool(x, 0, r["size"], op), 1, c["size"], op)
    if kind == "image":                               # x: (H, W, C)
        r, c = lay["axes"]
        return np_pool(np_pool(x, 0, r["size"], "mean"), 1, c["size"], "mean")
    if kind == "points":
        (a,) = lay["axes"]
        return x[[i * a["count"] // a["size"] for i in range(a["size"])]]
    return x


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_members_come_from_the_default_and_replace_it_whole():
    assert dv.resolve(ATTN, {}) == {"source": "attention", "index": {"layer": 0, "head": 0},
                                    "reduce": {}, "budget": {"max_extent": 512, "pool": "mean"}}
    # index replaces the default's index as a whole
    rv = dv.resolve(ATTN, {"index": {"layer": -1, "head": [1, None, 2]}, "reduce": {"head": "max"}})
    assert rv["index"] == {"layer": 11, "head": [1, 8, 2]} and rv["reduce"] == {"head": "max"}
    # {"index": {}} is the default without its indexing: four kept dims do not fit
    with pytest.raises(DataViewError, match="keeps 2 dims"):
        dv.resolve(ATTN, {"index": {}})
    # ranges: negative and null bounds, clamped, written [start, stop, step]
    rv = dv.resolve(ATTN, {"index": {"layer": 3, "head": 0, "query": [-10, None],
                                     "key": [-1000, 1000, 5]}})
    assert rv["index"]["query"] == [54, 64, 1] and rv["index"]["key"] == [0, 48, 5]
    # budget merges key by key; unknown members and budget keys are ignored
    o = offer(w=(("out", "in"), (10, 10)), default={"budget": {"max_extent": 64, "pool": "max"}})
    assert dv.resolve(o, {"budget": {"max_extent": 32}, "colour": "red"})["budget"] == \
        {"max_extent": 32, "pool": "max"}
    assert dv.resolve(o, {"budget": {"other": 1}})["budget"] == {"max_extent": 64, "pool": "max"}
    # the source defaults to the first; JSON text is accepted
    assert dv.resolve(offer(a=(("x", "y"), (2, 3)), b=(("u", "v"), (4, 5))), '{}')["source"] == "a"
    assert dv.resolve(offer(a=(("x", "y"), (2, 3)), b=(("u", "v"), (4, 5))),
                      b'{"source": "b"}')["source"] == "b"


def test_kept_dims_must_fit_the_kind():
    src = dict(s=(("a", "b", "c"), (4, 5, 6)))
    assert dv.resolve(offer("heatmap", **src), {"index": {"a": 1}})["index"] == {"a": 1}
    assert dv.resolve(offer("heatmap", **src), {"tile": "a"})["tile"] == "a"
    with pytest.raises(DataViewError, match="tiled heatmap keeps 3"):
        dv.resolve(offer("heatmap", **src), {"tile": "a", "index": {"b": 0}})
    with pytest.raises(DataViewError, match="only a heatmap can tile"):
        dv.resolve(offer("image", **src), {"tile": "a"})
    with pytest.raises(DataViewError, match="is not kept"):
        dv.resolve(offer("heatmap", **src), {"tile": "a", "index": {"a": 0}})
    assert dv.resolve(offer("image", **src), {"reduce": {"c": "mean"}})["budget"] == {"max_extent": 512}
    rv = dv.resolve(offer("points", **src), {"index": {"a": 0, "b": 1}})
    assert rv["budget"] == {"max_points": 100_000}
    with pytest.raises(DataViewError, match="points keep 1 dim"):
        dv.resolve(offer("points", **src), {"index": {"a": 0}})
    assert dv.resolve(offer("histogram", **src), {})["budget"] == {}
    s = Offer("series", (), None)
    assert dv.resolve(s, {}) == {"budget": {"max_backfill": 4096}}
    assert dv.resolve(s, {"budget": {"max_backfill": 10}}, retain=8000) == {"budget": {"max_backfill": 10}}
    with pytest.raises(DataViewError, match="a series takes no index"):
        dv.resolve(s, {"index": {"a": 1}})


def test_budget_limits():
    o = offer(w=(("out", "in"), (10, 10)))
    assert dv.resolve(o, {"budget": {"max_extent": 10 ** 12}})["budget"]["max_extent"] == 4096
    p = offer("points", w=(("n",), (10,)))
    assert dv.resolve(p, {"budget": {"max_points": 10 ** 12}})["budget"]["max_points"] == 1_000_000
    for bad in (0, -5, 2.0, "512", True, None and 1):
        if bad is None:
            continue
        with pytest.raises(DataViewError, match="positive integer"):
            dv.resolve(o, {"budget": {"max_extent": bad}})
    with pytest.raises(DataViewError, match="pool must be"):
        dv.resolve(o, {"budget": {"pool": "median"}})
    assert dv.resolve(o, {"budget": {"max_points": -1}})["budget"]["max_extent"] == 512   # not a heatmap key


def _deep(n):
    d = {}
    cur = d
    for _ in range(n):
        cur["a"] = {}
        cur = cur["a"]
    return d


HOSTILE = [
    ("not json", "valid JSON"),
    ("[" * 50_000, "valid JSON"),
    ('{"a":' * 10_000 + "1" + "}" * 10_000, "valid JSON"),        # nested 10,000 deep
    (b"\xff\xfe\x00", "UTF-8"),
    ('{"source": "' + "x" * 70_000 + '"}', "limited"),
    ('1' * 5000, "valid JSON"),
    ([], "JSON object"),
    (42, "JSON object"),
    ({"source": 5}, "no source"),
    ({"source": ""}, "no source"),
    ({"source": "x" * 10_000}, "no source"),
    ({"source": ["attention"]}, "no source"),
    ({"index": {"nope": 1}}, "no dim"),
    ({"index": {"layer": 12}}, "out of range"),
    ({"index": {"layer": -13}}, "out of range"),
    ({"index": {"layer": 10 ** 400, "head": 0}}, "out of range"),
    ({"index": {"layer": True, "head": 0}}, "must be an integer"),
    ({"index": {"layer": 1.0, "head": 0}}, "must be an integer"),
    ({"index": {"layer": "1", "head": 0}}, "must be an integer"),
    ({"index": {"layer": [0], "head": 0}}, "must be an integer"),
    ({"index": {"layer": [0, 1, 2, 3], "head": 0}}, "must be an integer"),
    ({"index": {"layer": [[0], 1], "head": 0}}, "must be an integer"),
    ({"index": {"layer": [0, 5, 0], "head": 0}}, "step"),
    ({"index": {"layer": [0, 5, -1], "head": 0}}, "step"),
    ({"index": {"layer": [5, 2], "head": 0}}, "selects nothing"),
    ({"index": {"layer": [100, None], "head": 0}}, "selects nothing"),
    ({"index": {"layer": _deep(5000), "head": 0}}, "must be an integer"),
    ({"index": _deep(5000)}, "no dim"),
    ({"index": {f"d{i}": 0 for i in range(10_000)}}, "no dim"),
    ({"index": "layer"}, "must be an object"),
    ({"reduce": {"head": "median"}, "index": {"layer": 0}}, "must be mean"),
    ({"reduce": {"head": ["mean"]}, "index": {"layer": 0}}, "must be mean"),
    ({"reduce": {"layer": "mean"}}, "cannot be reduced"),
    ({"reduce": "mean"}, "must be an object"),
    ({"tile": ["layer"]}, "is not kept"),
    ({"tile": "nope"}, "is not kept"),
    ({"budget": [512]}, "budget must be an object"),
    ({"budget": {"max_extent": float("nan")}}, "positive integer"),
    ({"budget": {"max_extent": float("inf")}}, "positive integer"),
    ({"budget": {"max_extent": 0}}, "positive integer"),
    ({"budget": {"pool": {"op": "max"}}}, "pool must be"),
    ({"budget": _deep(5000)}, None),          # unknown budget keys are ignored: resolves
]


@pytest.mark.parametrize("request_, match", HOSTILE, ids=range(len(HOSTILE)))
def test_hostile_requests_are_refused_with_a_short_message(request_, match):
    if match is None:
        assert dv.resolve(ATTN, request_)["budget"] == {"max_extent": 512, "pool": "mean"}
        return
    with pytest.raises(DataViewError) as ei:
        dv.resolve(ATTN, request_)
    msg = str(ei.value)
    assert match in msg and len(msg) < 300 and "\n" not in msg, msg


def test_huge_mosaic_is_refused():
    o = offer(s=(("cell", "y", "x"), (20_000_000, 2, 2)))
    with pytest.raises(DataViewError, match="exceeds 4096"):
        dv.resolve(o, {"tile": "cell"})


def test_resolve_agrees_with_the_protocol_reference():
    """When the wire module is present (sprint 2 integration): random views and
    requests resolve identically, or are refused by both."""
    wire = pytest.importorskip("caliper._wire")
    rng = random.Random(3)
    dims = ["a", "b", "c", "d"]
    agree = 0
    for _ in range(3000):
        kind = rng.choice([1, 1, 2, 3, 3, 5])
        view = {"name": "v", "kind": kind, "sources": [
            {"name": f"s{k}", "dims": [{"name": n, "size": rng.choice([1, 3, 64, 600])}
                                       for n in rng.sample(dims, rng.randint(1, 4))]}
            for k in range(rng.randint(1, 3))]}
        names = dims + ["zz"]
        req = {"source": rng.choice(["s0", "s1", None]),
               "index": {d: rng.choice([0, -1, 2, [0, None, 2], [1, 3], "x"])
                         for d in rng.sample(names, rng.randint(0, 3))},
               "reduce": {d: rng.choice(["mean", "sum", "norm", "bad"])
                          for d in rng.sample(names, rng.randint(0, 2))},
               "budget": {"max_extent": rng.choice([1, 64, 5000]), "max_points": rng.choice([1, 50])}}
        if rng.random() < 0.3:
            req["tile"] = rng.choice(names)
        try:
            r1 = wire.resolve_data_view(view, req)
        except wire.MessageError:
            r1 = None
        try:
            r2 = dv.resolve(view, req)
        except DataViewError:
            r2 = None
        assert r1 == r2, (view, req, r1, r2)
        if r1 is not None:
            assert wire.payload_layout(view, r1) == dv.layout(view, r2)
            agree += 1
    assert agree > 100


# ---------------------------------------------------------------------------
# layout and application
# ---------------------------------------------------------------------------
def test_layout_follows_the_window_rule():
    o = offer(w=(("out", "in"), (1000, 777)))
    lay = dv.layout(o, dv.resolve(o, {"budget": {"max_extent": 100}}))
    assert lay["extent"] == [100, 100] and lay["tile"] is None
    assert lay["axes"][0] == {"dim": "out", "start": 0, "step": 1, "count": 1000, "size": 100}
    o = offer(s=(("batch", "channel", "h", "w"), (64, 20, 30, 40)))
    lay = dv.layout(o, dv.resolve(o, {"index": {"batch": 3}, "tile": "channel",
                                      "budget": {"max_extent": 64}}))
    assert lay["tile"] == {"dim": "channel", "rows": 4, "cols": 5}
    assert [a["size"] for a in lay["axes"]] == [20, 16, 12] and lay["extent"] == [64, 60]


def _apply_cases():
    """(name, kind, dims, shape, request, dtype)"""
    return [
        ("index", "heatmap", ("l", "h", "q", "k"), (3, 4, 20, 30), {"index": {"l": 2, "h": -1}}, "f32"),
        ("ranges", "heatmap", ("l", "h", "q", "k"), (3, 4, 20, 30),
         {"index": {"l": 1, "h": 0, "q": [2, 17, 3], "k": [-10, None]}}, "f32"),
        ("mean", "heatmap", ("l", "h", "q", "k"), (3, 4, 20, 30),
         {"index": {"l": 0}, "reduce": {"h": "mean"}}, "f32"),
        ("sum-max-order", "heatmap", ("l", "h", "q", "k"), (3, 4, 20, 30),
         {"index": {"h": [1, 3]}, "reduce": {"l": "max", "h": "sum"}}, "f32"),
        ("min-norm", "heatmap", ("a", "b", "c", "d"), (6, 5, 3, 3),
         {"reduce": {"c": "min", "d": "norm"}}, "f32"),
        ("norm2", "heatmap", ("o", "i", "kh", "kw"), (6, 5, 3, 3),
         {"reduce": {"kh": "norm", "kw": "norm"}}, "f32"),
        ("pool-mean-uneven", "heatmap", ("r", "c"), (1000, 777), {"budget": {"max_extent": 100}}, "f32"),
        ("pool-max-even", "heatmap", ("r", "c"), (1024, 96), {"budget": {"max_extent": 64, "pool": "max"}}, "f32"),
        ("pool-min-uneven", "heatmap", ("r", "c"), (513, 700), {"budget": {"max_extent": 100, "pool": "min"}}, "f32"),
        ("pool-int", "heatmap", ("r", "c"), (50, 70), {"budget": {"max_extent": 16}}, "i64"),
        ("pool-half", "heatmap", ("r", "c"), (300, 200), {"budget": {"max_extent": 64}}, "f16"),
        ("tile", "heatmap", ("b", "ch", "y", "x"), (4, 6, 12, 10),
         {"index": {"b": 1}, "tile": "ch"}, "f32"),
        ("tile-pooled-middle", "heatmap", ("y", "ch", "x"), (90, 10, 70),
         {"tile": "ch", "budget": {"max_extent": 64}}, "f32"),
        ("tile-reduced", "heatmap", ("b", "ch", "y", "x"), (8, 7, 9, 9),
         {"reduce": {"b": "mean"}, "tile": "ch"}, "f32"),
        ("histogram", "histogram", ("b", "c", "y"), (16, 8, 4), {"reduce": {"b": "mean"}}, "f32"),
        ("points", "points", ("layer", "point"), (3, 5000), {"index": {"layer": 2}}, "f32"),
        ("points-sampled", "points", ("layer", "point"), (3, 5000),
         {"index": {"layer": 1, "point": [10, None, 3]}, "budget": {"max_points": 333}}, "f32"),
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", _apply_cases(), ids=lambda c: c[0])
def test_apply_matches_numpy(case, device):
    name, kind, dims, shape, req, dt = case
    g = torch.Generator().manual_seed(7)
    if kind == "points":
        x = torch.randn(*shape, 2, generator=g)
    else:
        x = torch.randn(*shape, generator=g)
    if dt == "i64":
        x = (x * 50).to(torch.int64)
    elif dt == "f16":
        x = x.half()
    if name == "pool-mean-uneven":
        x[5, 5] = float("nan")
    off = offer(kind, s=(dims, shape))
    rv = dv.resolve(off, req)
    p = dv.compile_view(off, rv)
    out = dv.apply(x.to(device), p)
    assert out.device.type == device
    ref = np_reference(npy(x.float()) if dt == "f16" else npy(x), off, rv)
    assert tuple(out.shape) == ref.shape
    assert list(out.shape[:2]) == p.extent or kind in ("points", "histogram")
    np.testing.assert_allclose(npy(out).astype(np.float64), ref, rtol=2e-3 if dt == "f16" else 1e-5,
                               atol=1e-3 if dt == "f16" else 1e-5, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout_", ["hwc", "chw"])
def test_image_pools_by_mean_and_keeps_uint8(layout_, device):
    g = torch.Generator().manual_seed(1)
    hwc = torch.randint(0, 256, (2, 40, 30, 3), generator=g, dtype=torch.uint8)
    x = hwc if layout_ == "hwc" else hwc.permute(0, 3, 1, 2).contiguous()
    off = offer("image", s=(("batch", "height", "width"), (2, 40, 30)))
    rv = dv.resolve(off, {"index": {"batch": 1}, "budget": {"max_extent": 16}})
    out = dv.apply(x.to(device), dv.compile_view(off, rv))
    assert out.dtype == torch.uint8 and tuple(out.shape) == (16, 16, 3)
    ref = np_pool(np_pool(npy(hwc[1]).astype(np.float64), 0, 16, "mean"), 1, 16, "mean")
    assert np.abs(npy(out).astype(np.float64) - ref).max() <= 0.5 + 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_points_sample_is_a_stable_stride_and_maps_back(device):
    z = torch.randn(4, 5000, 3).to(device)
    labels = torch.arange(5000, dtype=torch.float32).to(device)
    off = offer("points", s=(("layer", "point"), (4, 5000)))
    rv = dv.resolve(off, {"index": {"layer": 3, "point": [100, None, 2]},
                          "budget": {"max_points": 1000}})
    p = dv.compile_view(off, rv)
    a, b = dv.apply(z, p), dv.apply(z, p)
    rows = dv.source_rows(p)
    assert rows.shape == (1000,) and np.all(np.diff(rows) > 0) and rows[0] == 100
    assert np.array_equal(rows, 100 + 2 * ((np.arange(1000) * 2450) // 1000))
    assert torch.equal(a, b) and torch.equal(a.cpu(), z[3, rows].cpu())
    assert torch.equal(dv.take_rows(labels, p).cpu(), torch.from_numpy(rows).float())
    assert np.array_equal(dv.source_rows(p, [0, 7, 999]), rows[[0, 7, 999]])
    with pytest.raises(DataViewError, match="out of range"):
        dv.source_rows(p, [1000])
    with pytest.raises(DataViewError, match="values for a point dim"):
        dv.take_rows(labels[:10], p)


def test_apply_refuses_a_tensor_of_another_shape():
    p = dv.compile_view(ATTN, dv.resolve(ATTN, {}))
    with pytest.raises(DataViewError, match="does not fit"):
        dv.apply(torch.zeros(12, 8, 64, 47), p)
    img = offer("image", s=(("height", "width"), (8, 8)))
    with pytest.raises(DataViewError, match="does not fit"):
        dv.apply(torch.zeros(8, 8), dv.compile_view(img, dv.resolve(img, {})))


# ---------------------------------------------------------------------------
# the calls with dims, the registry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_show_with_dims_and_a_default(fake, device):
    attn = torch.randn(2, 3, 4, 24, 600).to(device)
    dims = ("layer", "sequence", "head", "query", "key")
    default = {"index": {"layer": 1, "sequence": 0, "head": 0}}
    assert caliper.show(attn, "attention", cmap="magma", dims=dims, data_view=default) is True
    got = fake.last("attention")
    assert torch.allclose(got["payload"].cpu(), F.adaptive_avg_pool1d(attn[1, 0, 0].cpu()[None], 512)[0])
    assert got["data_view"] == caliper.data_view("attention") == {
        "source": "attention", "index": {"layer": 1, "sequence": 0, "head": 0}, "reduce": {},
        "budget": {"max_extent": 512, "pool": "mean"}}
    assert [a["size"] for a in got["axes"]] == [24, 512] and got["tile"] is None
    assert got["meta"].colormap == 2 - 1                       # magma
    view = dv.REGISTRY.offer("attention").to_json()
    assert view["kind"] == 1 and view["sources"][0]["dims"][4] == {"name": "key", "size": 600}

    rv = caliper.set_data_view("attention", {"index": {"layer": 0, "sequence": 1},
                                             "reduce": {"head": "max"}})
    assert rv["reduce"] == {"head": "max"}
    caliper.show(attn, "attention", cmap="magma", dims=dims, data_view=default)
    want = F.adaptive_avg_pool1d(attn[0, 1].amax(0).cpu()[None], 512)[0]
    assert torch.allclose(fake.last("attention")["payload"].cpu(), want, atol=1e-6)

    with pytest.raises(DataViewError, match="out of range"):
        caliper.set_data_view("attention", {"index": {"layer": 7}})
    assert caliper.data_view("attention") == rv                # refused: kept the one it had
    assert caliper.set_data_view("attention", {}) == rv          # {} changes nothing
    assert caliper.set_data_view("attention", None)["index"] == default["index"]


def test_show_without_dims_is_unchanged(fake):
    x = torch.randn(6, 5, 5)
    caliper.show(x, "plain")
    got = fake.last("plain")
    assert tuple(got["payload"].shape) == (10, 15)                # today's channel mosaic
    assert got["data_view"] is None and caliper.data_view("plain") is None
    assert dv.REGISTRY.get("plain") is None


@pytest.mark.parametrize("device", DEVICES)
def test_show_an_image_source(fake, device):
    imgs = torch.rand(5, 3, 20, 30).to(device)
    caliper.show(imgs, "inputs", dims=("batch", "height", "width"), data_view={"index": {"batch": 2}})
    got = fake.last("inputs")
    assert dv.REGISTRY.get("inputs").kind == "image"
    want = (imgs[2].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).to(torch.uint8)
    assert tuple(got["payload"].shape) == (20, 30, 4)
    assert torch.equal(got["payload"][..., :3].cpu(), want.cpu())
    with pytest.raises(ValueError, match="dims"):
        caliper.show(torch.rand(5, 7, 20, 30), "notimg", dims=("batch", "height", "width"))


@pytest.mark.parametrize("device", DEVICES)
def test_points_with_dims_sample_points_and_labels(fake, device):
    z = torch.randn(3, 2000, 2).to(device)
    labels = (torch.arange(2000) % 7).to(device)
    assert caliper.set_data_view("emb", {"index": {"layer": 1}, "budget": {"max_points": 500}}) is None
    caliper.points(z, labels, "emb", classes=7, dims=("layer", "point"))
    got = fake.last("emb")
    rows = dv.REGISTRY.rows("emb")
    assert rows.shape == (500,) and got["data_view"]["budget"] == {"max_points": 500}
    assert torch.equal(got["payload"][:, :2].cpu(), z[1, rows].cpu())
    assert torch.equal(got["aux"].cpu(), labels[rows].float().cpu())


def test_histogram_with_dims(fake):
    x = torch.randn(4, 100)
    caliper.histogram(x, "h", bins=10, dims=("layer", "value"), data_view={"index": {"layer": 2}})
    got = fake.last("h")
    assert got["payload"].sum().item() == 100 and got["data_view"]["index"] == {"layer": 2}


def test_a_shape_change_falls_back_to_the_default_with_a_warning(fake):
    dims = ("batch", "y", "x")
    caliper.show(torch.zeros(4, 8, 8), "act", dims=dims)
    caliper.set_data_view("act", {"index": {"batch": 3}})
    x4 = torch.arange(4.0)[:, None, None].expand(4, 8, 8)
    caliper.show(x4, "act", dims=dims)
    assert fake.last("act")["payload"][0, 0].item() == 3.0
    with pytest.warns(RuntimeWarning, match="does not fit"):
        caliper.show(x4[:2], "act", dims=dims)
    assert fake.last("act")["payload"][0, 0].item() == 0.0       # the default, item 0
    caliper.show(x4, "act", dims=dims)
    assert fake.last("act")["payload"][0, 0].item() == 3.0       # the request comes back


def test_registry_under_concurrent_requests():
    reg = dv.Registry()
    e = reg.register("attn", "heatmap", [dv.Source("attn", ("layer", "head", "q", "k"))],
                     default={"index": {"layer": 0, "head": 0}})
    x = (torch.arange(4.0)[:, None, None, None] * 100
         + torch.arange(8.0)[None, :, None, None]).expand(4, 8, 16, 16).contiguous()
    reg.prepare(e, x)                                   # the source has data: requests resolve
    stop = threading.Event()
    errors, accepted = [], [0]

    def requester(seed):
        rng = random.Random(seed)
        while not stop.is_set():
            req = {"index": {"layer": rng.randint(-1, 5), "head": rng.randint(0, 9)}}
            try:
                reg.request("attn", req)
                accepted[0] += 1
            except DataViewError:
                pass
            except Exception as err:                     # pragma: no cover
                errors.append(err)

    threads = [threading.Thread(target=requester, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(3000):
            out, p = reg.prepare(e, x)
            v = out[0, 0].item()
            assert (out == v).all()
            assert v == p.resolved["index"]["layer"] * 100 + p.resolved["index"]["head"]
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert errors == [] and accepted[0] > 100
    reg.request("attn", {"index": {"layer": 2, "head": 5}})
    assert reg.prepare(e, x)[0][0, 0].item() == 205


def test_pending_requests_are_bounded():
    for i in range(dv.PENDING_LIMIT):
        assert caliper.set_data_view(f"later{i}", {}) is None
    with pytest.raises(DataViewError, match="unknown view"):
        caliper.set_data_view("one-too-many", {})
    caliper.set_data_view("later0", None)                       # None drops it


def test_offer_and_update(fake):
    calls = {"a": 0, "b": 0}
    ta, tb = torch.randn(4, 6), torch.randn(5, 7)

    def make(name, t):
        def get():
            calls[name] += 1
            return t
        return get

    rv = caliper.offer("probe", {"a": make("a", ta), "b": make("b", tb)}, dims=("row", "col"))
    assert calls == {"a": 1, "b": 1} and rv["source"] == "a"     # each callable ran once
    assert caliper.update() is True
    assert calls == {"a": 2, "b": 1}                             # the other source does no work
    assert torch.equal(fake.last("probe")["payload"], ta)
    caliper.set_data_view("probe", {"source": "b"})
    caliper.update("probe")
    assert calls == {"a": 2, "b": 2} and torch.equal(fake.last("probe")["payload"], tb)

    w = torch.zeros(3, 3)
    caliper.offer("w", w, rate=1e-3)                             # a reference, read each snapshot
    caliper.update("w")
    w.add_(1)
    caliper.update("w")                                          # not due: rate 0.001 Hz
    assert fake.last("w")["payload"].sum().item() == 0
    caliper.offer("w2", w)
    caliper.update("w2")
    assert fake.last("w2")["payload"].sum().item() == 9

    caliper.offer("pts", {"p": torch.randn(50, 2)}, "points", dims=("point",),
                  labels=lambda: torch.arange(50), classes=50, data_view={"budget": {"max_points": 10}})
    caliper.update("pts")
    assert torch.equal(fake.last("pts")["aux"], torch.arange(0, 50, 5).float())
    caliper.offer("none", lambda: None)
    assert caliper.update("none") is False and fake.last("none") is None
    with pytest.raises(TypeError, match="unexpected option"):
        caliper.offer("x", ta, "heatmap", bins=3)
    with pytest.raises(ValueError, match="no sources"):
        caliper.offer("x", ta, "series")
    with pytest.raises(ValueError, match="was offered"):
        caliper.update("never")
    caliper.remove("probe")
    assert caliper.data_view("probe") is None


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.fc1 = nn.Linear(6 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def _step(net, x):
    loss = net(x).square().mean()
    loss.backward()


def _hook_count(net):
    return sum(len(m._forward_hooks) for m in net.modules())


@pytest.mark.parametrize("device", DEVICES)
def test_watch_groups_beside_the_existing_views(fake, device):
    torch.manual_seed(0)
    net = Net().to(device)
    x = torch.randn(4, 1, 16, 16, device=device)
    h = caliper.watch(net, activations=True, gradients=True)
    try:
        assert h.modules == ["conv1", "fc1", "fc2"]               # as before
        assert set(h.views) == {"conv1.act", "fc1.act", "fc2.act", "conv1.weight.grad",
                                "fc1.weight.grad", "fc2.weight.grad"}
        assert _hook_count(net) == 3 and len(h) == 6
        assert h.groups == ["weights", "activations", "gradients"]
        _step(net, x)
        assert caliper.flush(timeout=5.0) == 0
        names = {p["name"] for p in fake.published}
        assert set(h.views) | set(h.groups) <= names
        with torch.no_grad():
            c1 = net.conv1(x)
        assert torch.allclose(fake.last("activations")["payload"], c1[0, 0])     # batch 0, channel 0
        assert torch.equal(fake.last("weights")["payload"], net.conv1.weight.detach().reshape(6, 9))
        assert torch.equal(fake.last("gradients")["payload"], net.conv1.weight.grad.reshape(6, 9))
        acts = dv.REGISTRY.offer("activations").to_json()
        assert [s["name"] for s in acts["sources"]] == ["conv1", "fc1", "fc2"]
        assert [d["name"] for d in acts["sources"][0]["dims"]] == ["batch", "channel", "height", "width"]
        assert [d["name"] for d in acts["sources"][1]["dims"]] == ["batch", "feature"]

        caliper.set_data_view("weights", {"source": "fc2.weight"})   # same dims: the source alone
        with pytest.raises(DataViewError, match="no dim"):
            caliper.set_data_view("activations", {"source": "fc1"})  # the default's index names channel
        caliper.set_data_view("activations", {"source": "fc1", "index": {}})
        caliper.set_data_view("gradients", {"source": "fc1.weight"})
        _step(net, x)
        with torch.no_grad():
            f1 = net.fc1(F.max_pool2d(F.relu(net.conv1(x)), 2).flatten(1))
        assert torch.allclose(fake.last("activations")["payload"], f1, atol=1e-6)
        assert torch.equal(fake.last("weights")["payload"], net.fc2.weight.detach())
        assert torch.equal(fake.last("gradients")["payload"], net.fc1.weight.grad)
    finally:
        h.remove()
    assert _hook_count(net) == 0 and dv.REGISTRY.get("weights") is None
    h2 = caliper.watch(net, groups="enc")
    assert h2.groups == ["enc.weights", "enc.activations"]
    h2.remove()
    h3 = caliper.watch(net, groups=False)
    assert h3.groups == [] and dv.REGISTRY.names() == []
    h3.remove()


class _Ops(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_watch_sources_nobody_views_do_no_work(fake, monkeypatch):
    monkeypatch.setattr(caliper, "show", lambda *a, **k: False)   # leave the per-module views out
    net = Net()
    x = torch.randn(2, 1, 16, 16)
    h = caliper.watch(net, gradients=True)
    prepared = []
    real = dv.DVWrite.__call__

    def spy(self, slot):                                     # where device work starts
        prepared.append((self.call.entry.name, self.call.source))
        return real(self, slot)
    monkeypatch.setattr(dv.DVWrite, "__call__", spy)
    try:
        _step(net, x)
        assert sorted(prepared) == [("activations", "conv1"), ("gradients", "conv1.weight"),
                                    ("weights", "conv1.weight")]
        hook = next(iter(net.fc1._forward_hooks.values()))
        inp, out = torch.randn(2, 384), torch.randn(2, 16)
        with _Ops() as rec:
            hook(net.fc1, (inp,), out)                        # not the selected source
        assert rec.ops == []
        hook2 = next(iter(net.conv1._forward_hooks.values()))
        out2 = torch.randn(2, 6, 16, 16)
        with _Ops() as rec:
            hook2(net.conv1, (x,), out2)
        assert rec.ops                                        # the selected source does work
        caliper.close()                                       # nothing attached at all
        prepared.clear()
        observed = []
        monkeypatch.setattr(dv.REGISTRY, "observe", lambda *a: observed.append(a))
        _step(net, x)
        assert prepared == [] and observed == []
    finally:
        h.remove()


# ---------------------------------------------------------------------------
# no readback
# ---------------------------------------------------------------------------
FORBIDDEN = {"item", "cpu", "numpy", "tolist", "synchronize", "histc", "nonzero", "unique",
             "masked_select", "bincount", "equal", "allclose"}


def _bad_calls(tree):
    return [(n.lineno, n.func.attr) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in FORBIDDEN]


def test_dataview_code_never_reads_back():
    """The data view code calls nothing that reads a value back from the device
    or synchronizes, and it never copies to a device (no upload that waits)."""
    tree = ast.parse((PKG / "_dataview.py").read_text())
    assert _bad_calls(tree) == []
    to_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "to"]
    assert all(isinstance(n.args[0], ast.Name) and n.args[0].id == "dtype0" for n in to_calls)
    init = ast.parse((PKG / "__init__.py").read_text())
    for fn in ast.walk(init):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("_dv_"):
            assert _bad_calls(fn) == [], fn.name
    assert _bad_calls(ast.parse((PKG / "_watch.py").read_text())) == []


@requires_mps
def test_apply_never_waits_for_the_device(fake):
    """With a long queue of work on MPS, every data view call returns while the
    queue is still busy: nothing in it waited for the device."""
    dev = torch.device("mps")
    a = torch.randn(4096, 4096, device=dev)
    cases = [(ATTN, {"index": {"layer": 3, "head": 1}}, torch.randn(12, 8, 64, 48, device=dev))]
    for req in ({"reduce": {"h": "mean"}, "index": {"l": 0}}, {"reduce": {"l": "norm", "h": "max"}},
                {"index": {"l": 1, "h": 2}, "budget": {"max_extent": 7, "pool": "min"}},
                {"index": {"l": 1}, "tile": "h", "budget": {"max_extent": 33}}):
        cases.append((offer(s=(("l", "h", "q", "k"), (3, 5, 100, 90))), req,
                      torch.randn(3, 5, 100, 90, device=dev)))
    img = offer("image", s=(("height", "width"), (101, 77)))
    cases.append((img, {"budget": {"max_extent": 20}},
                  torch.randint(0, 255, (101, 77, 3), dtype=torch.uint8, device=dev)))
    pts = offer("points", s=(("point",), (100_000,)))
    cases.append((pts, {"budget": {"max_points": 999}}, torch.randn(100_000, 3, device=dev)))
    progs = [(dv.compile_view(o, dv.resolve(o, r)), t) for o, r, t in cases]
    labels = torch.arange(100_000, device=dev)

    def run_all():
        for p, t in progs:
            dv.apply(t, p)
        dv.take_rows(labels, progs[-1][0])
        caliper.show(cases[0][2], "busy", dims=("layer", "head", "query", "key"))
        caliper.points(cases[-1][2], labels, "busy-points", dims=("point",))

    run_all()                                       # compile the graphs first
    torch.mps.synchronize()
    with dv._INDEX_LOCK:
        dv._INDEX_CACHE.clear()                     # the index tensors are made in the window too
    x = a
    for _ in range(60):
        x = x @ a
        x = x / x.abs().amax()
    ev = torch.mps.Event()
    ev.record()
    t0 = time.perf_counter()
    run_all()
    dt = time.perf_counter() - t0
    busy = not ev.query()
    torch.mps.synchronize()
    assert busy, f"the queue drained during the data view calls ({dt * 1e3:.1f} ms)"


@requires_mps
def test_points_with_cpu_labels_never_wait_for_the_device():
    """plan_points uploads CPU labels for MPS points without waiting behind the
    queue, and they arrive as they were at the call even if the caller then
    reuses its buffer (a non-blocking copy reads host memory when it runs)."""
    from caliper._convert import plan_points
    from caliper._ring import RingSlot
    dev = torch.device("mps")
    z = torch.randn(1000, 2, device=dev)
    labels = torch.arange(1000, dtype=torch.float32)
    plan_points(z, labels)                              # warm up
    a = torch.randn(4096, 4096, device=dev)
    torch.mps.synchronize()
    x = a
    for _ in range(40):
        x = x @ a
        x = x / x.abs().amax()
    ev = torch.mps.Event()
    ev.record()
    plan = plan_points(z, labels)
    busy = not ev.query()
    labels.fill_(-1.0)                                  # the caller reuses its buffer at once
    stats = torch.zeros(8, device=dev)
    slot = RingSlot(0, torch.zeros(1000, 3, device=dev), stats, torch.zeros(1000, device=dev), None,
                    stats[0], stats[1], stats[2:8:2], stats[3:8:2])
    plan.write(slot)
    torch.mps.synchronize()
    assert busy, "plan_points waited for the device"
    assert torch.equal(slot.aux.cpu(), torch.arange(1000, dtype=torch.float32))


def test_import_does_not_import_torch():
    code = "import sys, caliper, caliper._dataview; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------
@requires_mps
def test_apply_timings_on_mps(capsys):
    """Queued cost of apply on the training thread (no sync in the timed region;
    the queue is drained between runs, outside it), and the device time."""
    dev = torch.device("mps")
    cases = []
    attn = torch.randn(12, 12, 512, 512, device=dev)
    o = offer(a=(("layer", "head", "query", "key"), tuple(attn.shape)))
    cases.append(("one head of (12, 12, 512, 512)", attn, o, {"index": {"layer": 3, "head": 5}}))
    act = torch.randn(64, 256, 14, 14, device=dev)
    o = offer(a=(("batch", "channel", "height", "width"), tuple(act.shape)))
    cases.append(("batch mean of (64, 256, 14, 14), channel mosaic", act, o,
                  {"reduce": {"batch": "mean"}, "tile": "channel"}))
    oh = offer("histogram", a=(("batch", "channel", "height", "width"), tuple(act.shape)))
    cases.append(("batch mean of (64, 256, 14, 14), whole", act, oh, {"reduce": {"batch": "mean"}}))
    w = torch.randn(4096, 4096, device=dev)
    o = offer(w=(("out", "in"), (4096, 4096)))
    cases.append(("(4096, 4096) pooled to 512, mean", w, o, {}))
    cases.append(("(4096, 4096) pooled to 512, max", w, o, {"budget": {"pool": "max"}}))
    w3 = torch.randn(3000, 3000, device=dev)
    o = offer(w=(("out", "in"), (3000, 3000)))
    cases.append(("(3000, 3000) pooled to 512, uneven windows", w3, o, {}))
    z = torch.randn(1_000_000, 2, device=dev)
    o = offer("points", z=(("point",), (1_000_000,)))
    cases.append(("1,000,000 points sampled to 100,000", z, o, {}))
    rows = []
    for label, t, off, req in cases:
        p = dv.compile_view(off, dv.resolve(off, req))
        with dv._INDEX_LOCK:
            dv._INDEX_CACHE.clear()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        dv.apply(t, p)
        first = time.perf_counter() - t0
        queued, total = [], []
        for _ in range(40):
            torch.mps.synchronize()
            a0 = time.perf_counter()
            dv.apply(t, p)
            a1 = time.perf_counter()
            torch.mps.synchronize()
            queued.append(a1 - a0)
            total.append(time.perf_counter() - a0)
        rows.append((label, first * 1e3, statistics.median(queued) * 1e3,
                     sorted(queued)[int(0.9 * len(queued))] * 1e3, statistics.median(total) * 1e3))
    with capsys.disabled():
        print("\napply on MPS, ms: first call | queued median | queued p90 | to completion median")
        for r in rows:
            print(f"  {r[0]:52s} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f}")
    assert all(r[2] < 50 for r in rows)
