"""The shared SnapshotRing and the routing, against fake transports (no core):
near-free unattached calls, rate limiting, slot choice, and the payload/stats
layout of every kind (byte-exact against torch references)."""
import statistics
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _ring
from caliper._convert import SERIES_CAPACITY, plan_points, plan_show
from conftest import mps_available

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


class FakeEvent:
    def __init__(self):
        self.done = False

    def record(self):
        self.done = False

    def query(self):
        return self.done


class FakeTransport:
    """Transport protocol over plain torch tensors; records what it receives."""

    def __init__(self, attached=True, reading=()):
        self.att = attached
        self.reading = set(reading)
        self.allocs, self.published, self.removed = [], [], []
        self.reading_calls = 0

    def attached(self):
        return self.att

    def alloc(self, name, spec, slot, role, shape, dtype):
        self.allocs.append((name, slot, role, tuple(shape), spec.slots))
        return torch.zeros(shape, dtype=dtype, device=spec.device)

    def reading_slot(self, name):
        self.reading_calls += 1
        return min(self.reading) if self.reading else -1

    def reading_slots(self, name):
        self.reading_calls += 1
        return list(self.reading)

    def publish(self, name, slot, meta):
        # device clones, in stream order: the content at publish time
        self.published.append((name, slot.index, meta, slot.payload.clone(), slot.stats.clone(),
                               None if slot.aux is None else slot.aux.clone()))

    def remove(self, name):
        self.removed.append(name)


@pytest.fixture
def fake():
    tr = FakeTransport()
    route = caliper._attach_transport(tr, rate=0)     # every call snapshots unless rate=
    yield tr
    caliper._detach_transport(route)


def _flush():
    assert caliper.flush(timeout=5.0) == 0


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def test_unattached_is_near_free():
    tr = FakeTransport(attached=False)
    route = caliper._attach_transport(tr)
    try:
        dev = "mps" if mps_available() else "cpu"
        x = torch.rand(2048, 2048, device=dev)
        z = torch.rand(1000, 2, device=dev)
        pc = time.perf_counter_ns
        meds = {}
        for name, fn in [("show", lambda: caliper.show(x, "x")),
                         ("points", lambda: caliper.points(z, None, "p")),
                         ("series", lambda: caliper.series(0.5, "s")),
                         ("histogram", lambda: caliper.histogram(x, "h"))]:
            ts = []
            for _ in range(3000):
                a = pc()
                r = fn()
                ts.append(pc() - a)
                assert r is False
            meds[name] = statistics.median(ts) / 1e3
        print("unattached median us:", meds)
        assert all(m < 5.0 for m in meds.values()), meds
        assert tr.allocs == [] and tr.published == []
    finally:
        caliper._detach_transport(route)


def test_every_attached_transport_receives_the_offer():
    a, b, off = FakeTransport(), FakeTransport(), FakeTransport(attached=False)
    routes = [caliper._attach_transport(t, rate=0) for t in (a, b, off)]
    try:
        assert caliper.show(torch.ones(3, 4), "both") is True
        assert [p[0] for p in a.published] == ["both"] and [p[0] for p in b.published] == ["both"]
        assert off.allocs == [] and off.published == []
    finally:
        for r in routes:
            caliper._detach_transport(r)


def test_rate_limit(fake):
    x = torch.rand(8, 8)
    taken = 0
    t_end = time.perf_counter() + 0.5
    calls = 0
    while time.perf_counter() < t_end:
        taken += bool(caliper.show(x, "rl", rate=20))
        calls += 1
    assert calls > 1000
    assert 8 <= taken <= 12, taken                  # 20 Hz over 0.5 s (first call immediate)
    n0 = len(fake.published)
    for _ in range(50):
        assert caliper.show(x, "rl0", rate=0) is True   # rate=0: every call
    assert len(fake.published) == n0 + 50


# ---------------------------------------------------------------------------
# slot choice
# ---------------------------------------------------------------------------
def test_slot_choice_skips_published_reading_and_pending():
    tr = FakeTransport()
    ring = _ring.SnapshotRing(tr, rate=0)
    x = torch.rand(4, 4)
    plan = plan_show(x, 0, 0.0, 1.0)
    now = time.perf_counter()
    assert ring.take("v", plan, now)
    v = ring._views["v"]
    evs = [FakeEvent() for _ in v.slots]
    for s, e in zip(v.slots, evs):
        s.event = e                                   # make every snapshot pend
    evs[0].done = True                                # slot 0's first snapshot: CPU, published
    assert v.published == 0
    assert ring.take("v", plan, now) and v.pending[-1][0] == 1      # not the published slot
    tr.reading = {2}
    assert not ring.take("v", plan, now)              # 0 published, 1 pending, 2 read
    assert ring.counters["ring_full"] == 1
    tr.reading = set()
    assert ring.take("v", plan, now) and v.pending[-1][0] == 2
    evs[1].done = evs[2].done = True
    ring.poll()                                       # both complete: the newest wins
    assert v.published == 2 and ring.counters["superseded"] == 1
    assert [p[1] for p in tr.published] == [0, 2]
    assert [p[2].seq for p in tr.published] == [1, 2]
    # least recently written free slot: 0 (written first) before 1
    assert ring.take("v", plan, now) and v.pending[-1][0] == 0


def test_view_recreated_on_layout_change(fake):
    caliper.show(torch.rand(4, 4), "re")
    n = len(fake.allocs)
    caliper.show(torch.rand(4, 4), "re")
    assert len(fake.allocs) == n                      # same layout: same slots
    caliper.show(torch.rand(5, 4), "re")
    assert fake.removed == ["re"] and len(fake.allocs) == 2 * n


# ---------------------------------------------------------------------------
# kinds: payload / stats layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_heatmap_payload_and_autoscale_stats(fake, device):
    big = torch.randn(70, 33, device=device)
    s = big[5:45]                                     # nonzero storage offset
    assert caliper.show(s, "hm")
    _flush()
    name, idx, meta, payload, stats, _ = fake.published[-1]
    assert meta.kind == _ring.KIND_HEATMAP
    assert meta.flags & _ring.FLAG_AUTO_VMIN and meta.flags & _ring.FLAG_AUTO_VMAX
    assert torch.equal(payload.cpu(), s.cpu())
    lo, hi = torch.aminmax(s)
    assert stats[0].item() == lo.item() and stats[1].item() == hi.item()


@pytest.mark.parametrize("device", DEVICES)
def test_mosaic_and_rgb_payloads(fake, device):
    x = torch.randn(5, 3, 6, device=device)
    caliper.show(x, "mo")
    _flush()
    payload, stats = fake.published[-1][3].cpu(), fake.published[-1][4].cpu()
    assert payload.shape == (6, 18)                   # 3 cols x 2 rows of 3x6 tiles
    assert torch.equal(payload[3:6, 0:6], x[3].cpu()) and torch.isnan(payload[3:, 12:]).all()
    assert stats[0].item() == x.min().item() and stats[1].item() == x.max().item()
    rgb = torch.randint(0, 256, (6, 7, 3), dtype=torch.uint8, device=device)
    caliper.show(rgb, "rgb")
    _flush()
    p = fake.published[-1][3].cpu()
    assert p.shape == (6, 7, 4) and torch.equal(p[..., :3], rgb.cpu()) and (p[..., 3] == 255).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dims", [2, 3])
def test_points_payload_labels_and_bbox(fake, device, dims):
    g = torch.Generator().manual_seed(dims)
    z = torch.randn(257, dims, generator=g).to(device)
    y = torch.randint(0, 5, (257,), generator=g).to(device)       # int64 labels
    assert caliper.points(z, labels=y, name="pts", classes=5)
    _flush()
    _, _, meta, payload, stats, aux = fake.published[-1]
    assert meta.kind == _ring.KIND_POINTS and meta.points_dims == dims and meta.label_count == 5
    assert meta.stats_count == 8 and not meta.flags & _ring.FLAG_AUTO_VMIN
    p = payload.cpu()
    assert p.shape == (257, 3) and torch.equal(p[:, :dims], z.cpu())
    if dims == 2:
        assert (p[:, 2] == 0).all()
    assert aux.dtype == torch.float32 and torch.equal(aux.cpu(), y.cpu().float())
    st = stats.cpu()
    for a in range(dims):
        assert st[2 + 2 * a].item() == z[:, a].min().item()
        assert st[3 + 2 * a].item() == z[:, a].max().item()
    if dims == 2:
        assert st[6].item() == 0 and st[7].item() == 0
    # continuous labels: colormapped over their autoscaled range
    w = torch.rand(257, generator=g).to(device)
    caliper.points(z, labels=w, name="ptc")
    _flush()
    _, _, meta, _, stats, aux = fake.published[-1]
    assert meta.label_count == 0 and meta.flags & _ring.FLAG_AUTO_VMIN
    assert stats[0].item() == w.min().item() and stats[1].item() == w.max().item()


def test_points_rejects_bad_input(fake):
    with pytest.raises(ValueError):
        caliper.points(torch.zeros(4, 4), name="bad")
    with pytest.raises(ValueError):
        caliper.points(torch.zeros(4, 2), labels=torch.zeros(3), name="bad")
    with pytest.raises(ValueError):
        plan_points(torch.zeros(4, 2), None, classes=3)


@pytest.mark.parametrize("device", DEVICES)
def test_histogram_counts_match_histc(fake, device):
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(300, 301, generator=g) * 3 + 1).to(device)
    assert caliper.histogram(x, "hist", bins=50)
    _flush()
    _, _, meta, payload, stats, _ = fake.published[-1]
    lo, hi = x.min().item(), x.max().item()
    ref = torch.histc(x.cpu(), 50, lo, hi)
    assert torch.equal(payload.cpu(), ref), (payload.cpu() - ref).abs().max()
    assert stats[0].item() == lo and stats[1].item() == hi
    assert meta.flags == _ring.FLAG_AUTO_VMIN | _ring.FLAG_AUTO_VMAX
    # constant data: histc's [v - 1, v + 1] range, everything in the middle bin
    c = torch.full((100,), 2.5, device=device)
    caliper.histogram(c, "histc", bins=8)
    _flush()
    _, _, _, payload, stats, _ = fake.published[-1]
    assert stats[0].item() == 1.5 and stats[1].item() == 3.5
    assert torch.equal(payload.cpu(), torch.histc(c.cpu(), 8, 0, 0))


@pytest.mark.parametrize("device", DEVICES)
def test_series_appends_every_call_and_wraps(fake, device):
    n = SERIES_CAPACITY + 904
    vals = torch.arange(n, dtype=torch.float32, device=device)
    for i in range(n):
        caliper.series(vals[i] if device == "mps" else float(i), "loss", rate=1)   # 1 Hz publishes
    pubs = [p for p in fake.published if p[0] == "loss"]
    assert len(pubs) <= 2                              # rate limited
    _flush()                                           # the final snapshot
    _, idx, meta, payload, _, _ = [p for p in fake.published if p[0] == "loss"][-1]
    assert meta.kind == _ring.KIND_SERIES and meta.series_total == n and idx == 0
    ring_vals = payload.cpu().numpy()
    i = np.arange(n - SERIES_CAPACITY, n)
    assert np.array_equal(ring_vals[i % SERIES_CAPACITY], i.astype(np.float32))
    assert fake.allocs[0][4] == 1                      # one slot (ViewSpec.slots == 1)


def test_series_never_consults_reading_slot():
    tr = FakeTransport(reading={0})
    route = caliper._attach_transport(tr, rate=0)
    try:
        for i in range(10):
            assert caliper.series(float(i), "s1")
        assert tr.reading_calls == 0 and len(tr.published) == 10
        assert tr.published[-1][2].series_total == 10
    finally:
        caliper._detach_transport(route)


@pytest.mark.skipif(not mps_available(), reason="no MPS device")
def test_series_cpu_then_device_values_keeps_history(fake):
    for i in range(10):
        caliper.series(float(i), "mix")
    for i in range(10, 20):
        caliper.series(torch.tensor(float(i), device="mps"), "mix")
    _flush()
    _, _, meta, payload, _, _ = fake.published[-1]
    assert payload.device.type == "mps" and meta.series_total == 20
    assert torch.equal(payload.cpu()[:20], torch.arange(20, dtype=torch.float32))
