"""The window transport with its render thread (caliper.window(offscreen=True):
the real render thread on an offscreen canvas): the producer never blocks, MPS
snapshots never touch the host, the slot rule never shows a torn frame."""
import os
import subprocess
import time

import numpy as np
import pytest
import torch

import caliper
from conftest import live_views_or_skip, reference_heatmap, requires_mps


@pytest.fixture(scope="module")
def win():
    w = caliper.window(512, 512, offscreen=True)
    yield w
    caliper.close()


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True).stdout
    return int(out.strip()) / 1024.0


def wait_drawn(w, name=None, timeout=3.0):
    """Flush, then wait until the render thread drew everything published."""
    caliper.flush()
    t_end = time.perf_counter() + timeout
    while time.perf_counter() < t_end:
        st = w.core.render_stats()
        if st["queued"] == 0 and (name is None or not w.transport.claimed_slots(name)
                                  or w.ring.view_kind(name) == 3):
            time.sleep(0.05)
            return
        time.sleep(0.01)


@requires_mps
def test_calls_never_block_on_queued_gpu_work(win, monkeypatch):
    live_views_or_skip()
    x = torch.rand(2048, 2048, device="mps")
    z = torch.randn(4096, 3, device="mps")
    y = torch.randint(0, 8, (4096,), device="mps")
    loss = torch.tensor(0.25, device="mps")
    calls = [("nb_x", lambda: caliper.show(x, "nb_x", vmin=0, vmax=1, rate=0)),
             ("nb_xa", lambda: caliper.show(x, "nb_xa", rate=0)),
             ("nb_p", lambda: caliper.points(z, y, "nb_p", classes=8, rate=0)),
             ("nb_h", lambda: caliper.histogram(x, "nb_h", rate=0)),
             ("nb_s", lambda: caliper.series(loss, "nb_s", rate=0))]
    for _, fn in calls:                    # create the views, complete their first snapshots
        fn()
    torch.mps.synchronize()
    wait_drawn(win)
    a = torch.randn(4096, 4096, device="mps")
    torch.mps.synchronize()

    def boom(*args, **kw):
        raise AssertionError("host sync / readback on the snapshot path")

    timings = {}
    with monkeypatch.context() as m:
        m.setattr(torch.mps, "synchronize", boom)
        for attr in ("item", "cpu", "numpy", "tolist"):
            m.setattr(torch.Tensor, attr, boom)
        for name, fn in calls:
            for _ in range(4):
                b = a @ a                  # tens of ms of GPU work queued ahead
            t0 = time.perf_counter()
            took = fn()
            timings[name] = ((time.perf_counter() - t0) * 1e6, took)
            timings[name] += (bool(win.ring._views[name].pending),)
    del b
    print("call us while GPU busy (us, took, still pending):", timings)
    for name, (us, took, pending) in timings.items():
        assert took is True, name
        assert pending, f"{name}: snapshot completed at once (was GPU work really queued?)"
        assert us < 200.0, (name, us)
    torch.mps.synchronize()
    wait_drawn(win)
    st = win.stats()
    assert st["errors"] == 0, st["last_error"]
    ref = reference_heatmap(x.cpu().numpy(), "viridis", 0.0, 1.0)
    assert np.array_equal(caliper.view_readback("nb_x"), ref)


@requires_mps
def test_no_host_copy_for_mps(win):
    x = torch.rand(1024, 1024, device="mps")
    for auto in (False, True):
        name = f"hc{int(auto)}"
        kw = {} if auto else dict(vmin=0.0, vmax=1.0)
        for _ in range(10):
            caliper.show(x, name, rate=0, **kw)
            time.sleep(0.002)
        torch.mps.synchronize()
        wait_drawn(win)
        m0, r0 = torch.mps.current_allocated_memory(), rss_mb()
        taken0 = win.ring.counters["taken"]
        for _ in range(200):
            x.mul_(1.0)
            caliper.show(x, name, rate=0, **kw)
            time.sleep(0.001)
        torch.mps.synchronize()
        wait_drawn(win)
        m1, r1 = torch.mps.current_allocated_memory(), rss_mb()
        taken = win.ring.counters["taken"] - taken0
        info = caliper.view_info(name)
        print(f"{name}: {taken} snapshots, d device mem {(m1 - m0) / 2**20:+.2f} MB, "
              f"d RSS {r1 - r0:+.2f} MB, {info['path']}")
        assert taken >= 50
        assert m1 == m0
        assert r1 - r0 < 4.0                 # one 4 MB host copy per call would be 800 MB
        assert info["path"] == "compute-imported" and info["imported"] is True


@requires_mps
def test_no_torn_or_stale_frames(win):
    """A's detector: every snapshot is a uniform tensor of value k shown over
    [0, 255], so a readback that holds two values is torn and one whose value
    was never published was rewritten under the reader."""
    lut = {int(c): i for i, c in enumerate(caliper.colormap_lut("viridis"))}
    ring = win.ring
    cur_k = [0]
    slot_k, published = {}, set()
    real_pick, real_publish = ring._pick, win.transport.publish

    def pick(name, v):
        idx = real_pick(name, v)
        if name == "T" and idx is not None:
            slot_k[idx] = cur_k[0]
        return idx

    def publish(name, slot, meta):
        if name == "T":
            published.add(slot_k.get(slot.index))
        real_publish(name, slot, meta)

    ring._pick = pick
    win.transport.publish = publish
    try:
        W = torch.zeros(1024, 1024, device="mps")
        checks = torn = wrong = snaps = it = 0
        t_end = time.perf_counter() + 1.5
        next_check = time.perf_counter() + 0.05
        while time.perf_counter() < t_end:
            cur_k[0] = (cur_k[0] + 7) % 256
            W.fill_(float(cur_k[0]))
            snaps += bool(caliper.show(W, "T", vmin=0, vmax=255, rate=0))
            it += 1
            if it % 8 == 0:
                torch.mps.synchronize()      # back-pressure, like a loop's periodic .item()
            if time.perf_counter() >= next_check and published:
                try:
                    rb = caliper.view_readback("T")
                except RuntimeError:     # published, not yet applied by the render thread
                    continue
                vals = np.unique(rb.view(np.uint32).ravel())
                checks += 1
                if len(vals) > 1:
                    torn += 1
                elif lut.get(int(vals[0])) not in published:
                    wrong += 1
                next_check = time.perf_counter() + 0.01
    finally:
        ring._pick = real_pick
        win.transport.publish = real_publish
    print(f"{it} iterations, {snaps} snapshots, {len(published)} values published, "
          f"{checks} readbacks: {torn} torn, {wrong} never published")
    assert checks >= 20 and snaps >= 20
    assert torn == 0 and wrong == 0


def test_default_rate_is_60hz(win):
    x = torch.rand(64, 64)
    taken = 0
    t_end = time.perf_counter() + 0.5
    while time.perf_counter() < t_end:
        taken += bool(caliper.show(x, "rate60"))
    assert 20 <= taken <= 32, taken


@requires_mps
def test_claims_follow_the_kind_rules(win):
    live_views_or_skip()
    z = torch.randn(256, 2, device="mps")
    caliper.points(z, name="cl_p", rate=0)
    caliper.show(torch.rand(32, 32, device="mps"), "cl_h", rate=0)
    torch.mps.synchronize()
    wait_drawn(win)
    time.sleep(0.1)
    v = win.ring._views["cl_p"]
    assert win.transport.claimed_slots("cl_p") == [v.published]   # points: current stays claimed
    assert win.transport.claimed_slots("cl_h") == []              # heatmap: free once read_done


def test_stats_report_frames_and_latency(win):
    caliper.show(torch.rand(16, 16), "st", rate=0)
    wait_drawn(win)
    st = win.stats()
    for k in ("frames", "published", "superseded", "drawn", "errors", "max_frame_gap_ms",
              "latency_ms", "deferred_takes", "ring"):
        assert k in st, k
    assert st["frames"] > 0 and st["published"] >= 1 and st["latency_ms"]
    assert caliper.stats()["window"]["ring"]["taken"] >= 1


def test_run_without_a_real_window_just_runs(win):
    assert caliper.run(lambda a, b=0: a + b, 2, b=3) == 5
