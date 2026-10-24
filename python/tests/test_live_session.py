"""caliper.live.SessionTransport: the live session other processes attach to.

Without a viewer: the session file, attached() (near-free), alloc() aliasing
the shared segment, publish/reading_slot, close() cleanup, a SIGKILLed script
reclaimed by the lister. With a viewer: a headless caliper-viewer process
(offscreen canvas, --dump-dir) attached to the same session: its heartbeat
makes attached() True and a published heatmap appears byte-exact in its
texture (caliper_core_view_readback in the viewer), CPU and MPS; killing it
turns attached() False without blocking the script.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from caliper import _ring, live
from conftest import mps_available, reference_heatmap

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A private data root: session files go to <root>/live."""
    monkeypatch.setenv("CALIPER_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def tr(root):
    t = live.SessionTransport.open(script_name="unit_session.py")
    yield t
    t.close()


def heatmap_spec(h, w, device):
    return _ring.ViewSpec(kind=_ring.KIND_HEATMAP, shape=(h, w), dtype=torch.float32,
                          device=torch.device(device))


def host_floats(seg, offset_key, slot, n):
    info = seg.info()
    import ctypes
    addr = info["host_ptr"] + info[offset_key][slot]
    return np.ctypeslib.as_array((ctypes.c_float * n).from_address(addr)).copy()


def alloc_slots(t, name, spec, n=3):
    slots = []
    for i in range(n):
        p = t.alloc(name, spec, i, _ring.ROLE_PAYLOAD, spec.shape, spec.dtype)
        s = t.alloc(name, spec, i, _ring.ROLE_STATS, (_ring.STATS_FLOATS,), torch.float32)
        slots.append(_ring.Slot(i, p, s))
    return slots


def meta(slot, **kw):
    return _ring.SnapshotMeta(kind=_ring.KIND_HEATMAP, slot=slot, seq=0, stamp_ns=time.monotonic_ns(), **kw)


# ---------------------------------------------------------------------------
# without a viewer
# ---------------------------------------------------------------------------
def test_open_writes_session_file(root, tr):
    f = Path(tr.session_file)
    assert f == root / "live" / f"{os.getpid()}.json"
    assert (f.stat().st_mode & 0o777) == 0o600
    d = json.loads(f.read_text())
    assert d["protocol"] == live._ext().PROTOCOL_VERSION
    assert d["pid"] == os.getpid()
    assert d["session"] == tr.session
    assert d["script"] == "unit_session.py"
    assert d["start_unix_ms"] > 0
    assert d["views"] == []
    assert d["segments"] == [f"/{tr.session}-ctl"]
    spec = heatmap_spec(4, 4, "cpu")
    tr.alloc("w", spec, 0, _ring.ROLE_PAYLOAD, (4, 4), torch.float32)
    d = json.loads(f.read_text())
    assert d["views"] == ["w"] and len(d["segments"]) == 2
    [s] = [s for s in live.list_sessions() if s["session"] == tr.session]
    assert s["pid"] == os.getpid() and s["alive"] and s["compatible"]
    assert s["script"] == "unit_session.py" and s["view_count"] == 1 and s["viewer_pid"] == 0
    assert live.data_root() == str(root)


def test_attached_false_and_near_free(tr):
    assert tr.attached() is False
    n = 10_000
    ts = []
    f = tr.attached
    clock = time.perf_counter_ns
    for _ in range(n):
        a = clock()
        f()
        ts.append(clock() - a)
    ts.sort()
    med_us = ts[n // 2] / 1e3
    t0 = clock()
    for _ in range(n):
        f()
    loop_us = (clock() - t0) / n / 1e3
    print(f"attached(): median {med_us:.3f} us per call (timed singly), {loop_us:.3f} us in a loop")
    assert med_us < 5.0
    assert loop_us < 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_alloc_aliases_the_segment(tr, device):
    spec = heatmap_spec(8, 16, device)
    slots = alloc_slots(tr, "w", spec)
    seg = tr.segment("w")
    assert seg.slot_count == 3
    for s in slots:
        assert s.payload.device.type == device and s.payload.shape == (8, 16)
        assert s.stats.device.type == device and s.stats.shape == (_ring.STATS_FLOATS,)
    x = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    slots[1].payload.copy_(x.to(device))
    torch.aminmax(slots[1].payload, out=(slots[1].stats[0], slots[1].stats[1]))
    if device == "mps":
        torch.mps.synchronize()
    np.testing.assert_array_equal(host_floats(seg, "slot_offsets", 1, 128), x.numpy().ravel())
    np.testing.assert_array_equal(host_floats(seg, "stats_offsets", 1, 2), [0.0, 127.0])
    assert not host_floats(seg, "slot_offsets", 0, 128).any()   # other slots untouched
    # Points: (N,3) payload + (N,) aux per slot, in the same segment.
    pspec = _ring.ViewSpec(kind=_ring.KIND_POINTS, shape=(32, 3), dtype=torch.float32,
                           device=torch.device(device), aux_shape=(32,))
    aux = tr.alloc("pts", pspec, 2, _ring.ROLE_AUX, (32,), torch.float32)
    aux.copy_(torch.arange(32, dtype=torch.float32).to(device) % 4)
    if device == "mps":
        torch.mps.synchronize()
    np.testing.assert_array_equal(host_floats(tr.segment("pts"), "aux_offsets", 2, 32),
                                  np.arange(32) % 4)
    # A region larger than the view's is refused.
    with pytest.raises(ValueError):
        tr.alloc("w", spec, 0, _ring.ROLE_PAYLOAD, (9, 16), torch.float32)
    # A changed spec recreates the segment.
    old = seg.info()["segment"]
    tr.alloc("w", heatmap_spec(4, 4, device), 0, _ring.ROLE_PAYLOAD, (4, 4), torch.float32)
    assert tr.segment("w").info()["segment"] != old
    assert not live._ext().shm_exists(old)


def test_publish_and_reading_slot(tr):
    spec = heatmap_spec(4, 4, "cpu")
    slots = alloc_slots(tr, "w", spec)
    seg = tr.segment("w")
    assert tr.reading_slot("w") == -1 and tr.claimed_slots("w") == ()
    assert tr.reading_slot("nope") == -1
    tr.publish("w", slots[2], meta(2, colormap=2, vmin=-1.0, vmax=3.0,
                                   flags=_ring.FLAG_AUTO_VMIN | _ring.FLAG_RAW, stats_count=2))
    assert seg.published() == (2, 1)
    m = seg.slot_meta(2)
    ext = live._ext()
    assert m["seq"] == 1 and m["colormap"] == 2 and m["vmin"] == -1.0 and m["vmax"] == 3.0
    assert m["flags"] == ext.SNAP_AUTO_VMIN | ext.SNAP_RAW
    # A viewer's claims (as live_core.cpp leaves them).
    seg._set_claims(2, -1)
    assert tr.reading_slot("w") == 2 and tr.claimed_slots("w") == (2,)
    seg._set_claims(2, 0)          # points: current 2 (== published), superseded 0
    assert tr.reading_slot("w") == 0 and set(tr.claimed_slots("w")) == {0, 2}
    tr.publish("w", slots[1], meta(1))
    assert tr.reading_slot("w") == 2   # current one first once the client published past it
    assert set(tr.claimed_slots("w")) == {0, 2}
    seg._set_claims(-1, -1)
    # Series: one slot, series_total carried.
    sspec = _ring.ViewSpec(kind=_ring.KIND_SERIES, shape=(64,), dtype=torch.float32,
                           device=torch.device("cpu"), slots=1)
    sp = tr.alloc("loss", sspec, 0, _ring.ROLE_PAYLOAD, (64,), torch.float32)
    sp[:5] = torch.arange(5.0)
    tr.publish("loss", 0, _ring.SnapshotMeta(kind=_ring.KIND_SERIES, slot=0, seq=0,
                                              stamp_ns=time.monotonic_ns(), series_total=5, stats_count=0))
    assert tr.segment("loss").slot_count == 1
    assert tr.segment("loss").slot_meta(0)["series_total"] == 5
    st = tr.stats()
    assert st["views"]["w"]["published"] == 2 and st["views"]["loss"]["seq"] == 1
    tr.count_drop("w")
    assert tr.stats()["views"]["w"]["dropped_client"] == 1


def test_close_removes_file_and_unlinks(root):
    t = live.SessionTransport.open()
    f = t.session_file
    alloc_slots(t, "w", heatmap_spec(4, 4, "cpu"))
    names = [f"/{t.session}-ctl", t.segment("w").info()["segment"]]
    assert all(live._ext().shm_exists(n) for n in names)
    t.close()
    t.close()   # idempotent
    assert not os.path.exists(f)
    assert not any(live._ext().shm_exists(n) for n in names)
    assert t.attached() is False and t.session_file == ""
    assert [s for s in live.list_sessions() if s["session"] == t.session] == []


def test_killed_script_is_reclaimed_by_the_lister(root):
    code = textwrap.dedent("""
        import sys, time, torch
        from caliper import live, _ring
        t = live.SessionTransport.open(script_name="doomed.py")
        spec = _ring.ViewSpec(kind=_ring.KIND_HEATMAP, shape=(4, 4), dtype=torch.float32,
                              device=torch.device("cpu"))
        t.alloc("w", spec, 0, "payload", (4, 4), torch.float32)
        print(t.session, t.session_file, t.segment("w").info()["segment"], flush=True)
        time.sleep(60)
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        token, path, seg = p.stdout.readline().split()
        assert [s["session"] for s in live.list_sessions()] == [token]
        p.send_signal(signal.SIGKILL)
        p.wait(10)
    finally:
        if p.poll() is None:
            p.kill()
    assert os.path.exists(path) and live._ext().shm_exists(seg)   # leaked by the kill
    assert live.list_sessions() == []                             # the lister reclaims
    assert not os.path.exists(path)
    assert not live._ext().shm_exists(seg) and not live._ext().shm_exists(f"/{token}-ctl")


# ---------------------------------------------------------------------------
# with a viewer process (headless caliper-viewer)
# ---------------------------------------------------------------------------
def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.01)
    return None


def dumped(d: Path, name: str, seq: int):
    j = d / f"{name}.json"
    if not j.exists():
        return None
    info = json.loads(j.read_text())
    if info["seq"] < seq:
        return None
    px = np.fromfile(d / info["file"], dtype=np.uint8)
    return info, px.reshape(info["height"], info["width"], 4)


@pytest.mark.parametrize("device", DEVICES)
def test_viewer_attaches_and_draws_byte_exact(tr, tmp_path, device):
    dump = tmp_path / "dump"
    dump.mkdir()
    log = tmp_path / "viewer.log"
    v = tr.launch_viewer(args=("--headless", "--dump-dir", str(dump), "--size", "64x64",
                               "--stats", "--exit-after", "60"), log=str(log))
    try:
        if not tr.wait_attached(15):
            pytest.skip(f"the headless viewer did not attach (no GPU?): {log.read_text()[-600:]}")
        assert tr.attached() is True and tr.viewer_alive()
        assert tr.stats()["viewer_pid"] == v.pid
        H, W = 48, 64
        slots = alloc_slots(tr, "ramp", heatmap_spec(H, W, device))
        ramp = (torch.arange(W, dtype=torch.float32) / (W - 1)).repeat(H, 1)

        # 1: fixed range, written on the producer's device.
        slots[0].payload.copy_(ramp.to(device))
        if device == "mps":
            torch.mps.synchronize()   # the ring polls an Event instead; the test just waits
        tr.publish("ramp", slots[0], meta(0, colormap=0, vmin=0.0, vmax=1.0))
        got = wait_for(lambda: dumped(dump, "ramp", 1))
        assert got is not None, log.read_text()[-800:]
        info, px = got
        assert info["width"] == W and info["height"] == H
        if device == "mps":
            assert info["path"] == "compute-imported" and info["imported"] == 1
        assert np.array_equal(px, reference_heatmap(ramp.numpy(), "viridis", 0.0, 1.0))
        assert tr.reading_slot("ramp") == 0

        # 2: autoscaled: the range comes from the slot's stats (GPU aminmax).
        torch.manual_seed(0)
        x = torch.randn(H, W) * 3 + 1
        s = slots[1]
        s.payload.copy_(x.to(device))
        torch.aminmax(s.payload, out=(s.stats[0], s.stats[1]))
        if device == "mps":
            torch.mps.synchronize()
        tr.publish("ramp", s, meta(1, colormap=2, flags=_ring.FLAG_AUTO_VMIN | _ring.FLAG_AUTO_VMAX))
        got = wait_for(lambda: dumped(dump, "ramp", 2))
        assert got is not None
        lo, hi = float(x.min()), float(x.max())
        assert np.array_equal(got[1], reference_heatmap(x.numpy(), "rdbu", lo, hi))
        assert wait_for(lambda: tr.reading_slot("ramp") == 1)

        # The viewer dies: nothing blocks, attached() turns False, publishing
        # still works (snapshots just stop being drawn).
        v.send_signal(signal.SIGKILL)
        v.wait(10)
        t0 = time.monotonic()
        assert wait_for(lambda: not tr.attached(), timeout=3.0)
        gone_s = time.monotonic() - t0
        print(f"[{device}] attached() False {gone_s:.2f} s after the viewer was SIGKILLed")
        assert gone_s < 1.5
        tr.publish("ramp", slots[2], meta(2, vmin=0.0, vmax=1.0))
        assert tr.viewer_alive() is False and tr.wait_viewer(0.1)
    finally:
        if v.poll() is None:
            v.kill()
            v.wait(5)


def test_viewer_sees_the_session_end_and_keeps_its_frames(tr, tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    log = tmp_path / "viewer.log"
    v = tr.launch_viewer(args=("--headless", "--dump-dir", str(dump), "--size", "32x32",
                               "--stats", "--exit-after", "60"), log=str(log))
    try:
        if not tr.wait_attached(15):
            pytest.skip(f"the headless viewer did not attach (no GPU?): {log.read_text()[-600:]}")
        slots = alloc_slots(tr, "w", heatmap_spec(8, 8, "cpu"))
        slots[0].payload.fill_(0.5)
        tr.publish("w", slots[0], meta(0, vmin=0.0, vmax=1.0))
        assert wait_for(lambda: dumped(dump, "w", 1))
        t0 = time.monotonic()
        tr.close()                       # the script ends cleanly; never waits on the viewer
        assert time.monotonic() - t0 < 0.5
        assert v.poll() is None          # the viewer stays up
        assert wait_for(lambda: "client=exited" in log.read_text(), timeout=6.0), log.read_text()[-800:]
        line = [ln for ln in log.read_text().splitlines() if "client=exited" in ln][-1]
        assert " views=1" in line        # it keeps what it received
    finally:
        v.terminate()
        v.wait(5)
