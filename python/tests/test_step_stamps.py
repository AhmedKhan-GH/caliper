"""The tests of the step stamps that the script puts on the snapshots of watched views.

See design/instrument-interface.md, section Step History.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from caliper import _controls, _net, _net_ring, _wire
from caliper._net_dataview import Reduced

FIXTURE = (Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "wire_v3" / "v3_2"
           / "snapshot_stamped.json")
INT64_MIN = -(1 << 63)


def _snap(stamp_steps=None) -> _net.SnapData:
    red = Reduced(data=None, vmin=0.0, vmax=1.0)
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    return _net.SnapData("attn", 1, 5, 81234567894123, _wire.Kind.HEATMAP,
                         {"source": "layer0"}, data, None, [0.0, 3.0], red, stamp_steps)


def _frame(snap, stamped):
    parts, n = _net.Link._snapshot_frame(1, snap, stamped)
    buf = b"".join(bytes(p) for p in parts)
    assert len(buf) == n
    return buf


def test_stamped_frame_decodes_with_step_and_wall():
    buf = _frame(_snap((1200, 1763200000123456789)), True)
    msg = _wire.decode(buf)
    st = msg.stamp.array()
    assert st.dtype == np.int64 and st.tolist() == [1200, 1763200000123456789]
    plain = _wire.decode_frame(_frame(_snap(), False))
    assert _wire.decode_frame(buf).minor == plain.minor


def test_unstamped_frame_is_unchanged():
    snap = _snap((1200, 1763200000123456789))
    before = _frame(_snap(), False)
    assert _frame(snap, False) == before
    assert _frame(snap, True) != before
    assert _frame(_snap(), True) == before        # no stamp data: nothing to add
    assert _wire.decode(before).stamp is None


def test_missing_step_is_absent():
    msg = _wire.decode(_frame(_snap((None, 77)), True))
    assert msg.stamp.array().tolist() == [INT64_MIN, 77]
    assert INT64_MIN == _wire.STAMP_ABSENT


def _producer():
    ring = SimpleNamespace(lock=threading.Lock(), views={})
    return SimpleNamespace(ring=ring, controls=None, identity={"id": "s1"})


def test_views_timelines_only_with_the_capability():
    p = _producer()
    plain = _net._Producer.views_msg(p, SimpleNamespace(caps=frozenset()))
    assert plain.timelines is None
    assert "timelines" not in _wire.decode(_wire.encode(plain)).to_meta()
    msg = _net._Producer.views_msg(p, SimpleNamespace(caps=frozenset({_wire.CAP_TIMELINES})))
    assert msg.timelines == [{"name": "step", "kind": "sequence"},
                             {"name": "time", "kind": "time", "epoch": True}]
    back = _wire.decode(_wire.encode(msg))
    assert back.timelines == msg.timelines
    assert _net._Producer.views_msg(p).timelines is None
    assert _wire.CAP_TIMELINES in _net._Producer.capabilities(p)


def test_layout_matches_the_fixture():
    want = json.loads(FIXTURE.read_text())["meta"]["sections"]
    f = _wire.decode_frame(_frame(_snap((1200, 1763200000123456789)), True))
    got = [{"name": s["name"], "dtype": s["dtype"], "shape": s["shape"]}
           for s in f.meta["sections"]]
    assert got == want
    assert got[1]["shape"] == [2]


def test_run_step_follows_the_controls_step():
    old = _controls.CONTROLS.n
    try:
        _controls.CONTROLS.n = 42
        assert _net_ring._run_step() == 42
        _controls.CONTROLS.n = -1
        assert _net_ring._run_step() is None
        _controls.CONTROLS.n = _wire._SAFE_INT + 1
        assert _net_ring._run_step() is None
    finally:
        _controls.CONTROLS.n = old
