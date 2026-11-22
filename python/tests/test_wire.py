"""Wire protocol v3 (caliper/_wire.py): round trips, the golden frames in
tests/fixtures/wire_v3 (3.0) and wire_v3/v3_1 (what 3.1 adds), rejection of bad
frames, data view resolution, the controls and linked selection of 3.1, and
agreement between include/caliper/live_net.h, _wire.py and the reference page.

Needs numpy and pytest only: _wire imports no torch and no native extension, so
``pytest --noconftest python/tests/test_wire.py`` runs it without them."""
from __future__ import annotations

import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "wire_v3"
HEADER_H = ROOT / "include" / "caliper" / "live_net.h"
# The wire protocol reference is split across these pages of the docs tree.
PAGES = [ROOT / "docs" / "reference" / name for name in (
    "wire-protocol.md", "wire-frames.md", "session-messages.md", "data-messages.md",
    "data-view-messages.md", "run-control-messages.md", "linked-selection-messages.md")]
PAGE_32 = ROOT / "docs" / "reference" / "wire-protocol-3-2.md"   # the 3.2 draft's built subset


def _pages():
    return "\n".join(p.read_text() for p in PAGES if p.exists())


def _section(page, heading):
    m = re.search(r"^## " + re.escape(heading) + r"$", page, re.M | re.I)
    assert m, heading
    rest = page[m.end():]
    end = re.search(r"^## ", rest, re.M)
    return page[m.start():m.end() + (end.start() if end else len(rest))]

sys.path.insert(0, str(ROOT / "python"))
from caliper import _wire as w  # noqa: E402


def _generator():
    spec = importlib.util.spec_from_file_location("wire_v3_generate", FIXTURES / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _generator()
VALID_30 = sorted(p.stem for p in FIXTURES.glob("*.bin"))
VALID_31 = sorted(f"v3_1/{p.stem}" for p in (FIXTURES / "v3_1").glob("*.bin"))
VALID = VALID_30 + VALID_31
VALID_32 = sorted(f"v3_2/{p.stem}" for p in (FIXTURES / "v3_2").glob("*.bin"))   # test_wire_32.py
TYPES_32 = {0x70, 0x71, 0x72, 0x73, 0x74, 0x78, 0x79, 0x7A, 0x7B, 0x80, 0x81, 0x82}
LATER_MINOR = ("ping_future", "unknown_type")
TYPES_30 = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x10, 0x11, 0x20, 0x21, 0x22, 0x23, 0x24, 0x30,
            0x31, 0x40}
EXPECT = json.loads((FIXTURES / "invalid" / "expect.json").read_text())


# ---------------------------------------------------------------------------
# the codec stands alone
# ---------------------------------------------------------------------------
def test_imports_no_torch_or_native():
    code = ("import sys; import caliper._wire; "
            "bad = [m for m in ('torch', 'caliper._native', 'caliper._live') if m in sys.modules]; "
            "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "python")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# golden frames
# ---------------------------------------------------------------------------
def test_generator_reproduces_every_file():
    files = GEN.render()
    on_disk = {str(p.relative_to(FIXTURES)) for p in FIXTURES.rglob("*")
               if p.suffix in (".bin", ".json")}
    assert on_disk == set(files)
    for rel, data in files.items():
        assert (FIXTURES / rel).read_bytes() == data, rel


def test_one_golden_frame_per_message_type():
    covered = {w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).type for n in VALID + VALID_32}
    assert {int(t) for t in w.MsgType} <= covered


def test_golden_frames_keep_their_minor():
    """The 3.0 frames stay what a 3.0 sender wrote (minor 0), so implementations
    that only know 3.0 keep passing; the frames 3.1 adds carry minor 1."""
    for n in VALID_30:
        want = 9 if n in LATER_MINOR else 0
        assert w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).minor == want, n
    assert len(VALID_31) >= 15
    for n in VALID_31:
        assert w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).minor == w.MINOR == 1, n
    for n in sorted(p.name for p in (FIXTURES / "invalid").glob("*.bin")):
        buf = (FIXTURES / "invalid" / n).read_bytes()
        if len(buf) >= 8:
            assert struct.unpack_from("<H", buf, 6)[0] == 0, n


def test_31_frames_are_well_formed_for_30():
    """A 3.0 receiver accepts every 3.1 frame: the frame rules did not change,
    the new types are new ids, and the 3.0 types only gained optional keys."""
    new = {int(t) for t in w.MsgType} - TYPES_30 - TYPES_32
    assert new == {0x50, 0x51, 0x52, 0x60}
    for n in VALID_31:
        f = w.decode_frame((FIXTURES / f"{n}.bin").read_bytes())
        assert f.type in TYPES_30 | new and f.flags == 0, n


@pytest.mark.parametrize("name", VALID)
def test_golden_frame_small(name):
    assert (FIXTURES / f"{name}.bin").stat().st_size <= 600


@pytest.mark.parametrize("name", VALID)
def test_golden_frame_matches_description(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    desc = json.loads((FIXTURES / f"{name}.json").read_text())
    h = desc["header"]
    assert struct.unpack_from("<IHHHHIIHH", buf, 0) == (
        h["magic"], h["major"], h["minor"], h["type"], h["flags"], h["frame_len"],
        h["meta_len"], h["section_count"], h["reserved"])
    assert h["frame_len"] == len(buf)
    f = w.decode_frame(buf)
    assert f.meta == desc["meta"]
    mo = desc["meta_offset"]
    assert json.loads(buf[mo:mo + h["meta_len"]]) == desc["meta"]
    assert len(f.sections) == len(desc["sections"]) == len(desc["section_lengths"])
    for s, d in zip(f.sections, desc["sections"]):
        assert d["offset"] % 8 == 0
        assert (s.name, s.dtype, list(s.shape), s.data.nbytes) == (
            d["name"], d["dtype"], d["shape"], d["length"])
        raw = buf[d["offset"]:d["offset"] + d["length"]]
        assert bytes(s.data) == raw
        assert s.array().reshape(-1).tolist() == d["values"]
    # padding is zero in what we send
    end = mo + h["meta_len"]
    for d in desc["sections"]:
        assert buf[end:d["offset"]] == bytes(d["offset"] - end)
        end = d["offset"] + d["length"]


@pytest.mark.parametrize("name", VALID)
def test_golden_frame_round_trips(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    msg = w.decode(buf)
    if name in LATER_MINOR:
        return                                  # sent by a later minor; covered below
    assert w.encode(msg, minor=w.decode_frame(buf).minor) == buf


def test_later_minor_is_accepted():
    msg = w.decode((FIXTURES / "ping_future.bin").read_bytes())
    assert isinstance(msg, w.Ping) and msg.nonce == 17
    f = w.decode_frame((FIXTURES / "ping_future.bin").read_bytes())
    assert (f.minor, f.flags) == (9, 0x8001)
    u = w.decode((FIXTURES / "unknown_type.bin").read_bytes())
    assert isinstance(u, w.Unknown) and u.type == 0x0777 and u.meta == {"future": True}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_invalid_frame_rejected(name):
    buf = (FIXTURES / "invalid" / name).read_bytes()
    with pytest.raises(w.WireError) as e:
        w.decode_frame(buf)
    assert (e.value.reason, e.value.close_code) == (EXPECT[name]["reason"],
                                                    EXPECT[name]["close_code"])


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------
PING = w.encode(w.Ping(1))


def _patch(buf, fmt, off, value):
    b = bytearray(buf)
    struct.pack_into(fmt, b, off, value)
    return bytes(b)


def _reject(buf, reason, code):
    with pytest.raises(w.WireError) as e:
        w.decode_frame(buf)
    assert (e.value.reason, e.value.close_code) == (reason, code)


def test_bad_magic():
    _reject(b"CALX" + PING[4:], "bad_magic", 4000)
    _reject(b"\0" * 24, "bad_magic", 4000)


@pytest.mark.parametrize("major", [0, 1, 2, 4, 0xFFFF])
def test_wrong_major(major):
    _reject(_patch(PING, "<H", 4, major), "unsupported_major", 4001)
    _reject(_patch(PING, "<H", 4, major)[:8], "unsupported_major", 4001)


def test_oversize_lengths():
    _reject(_patch(PING, "<I", 12, w.MAX_FRAME_BYTES + 1), "too_big", 1009)
    _reject(_patch(PING, "<I", 12, 0xFFFFFFFF), "too_big", 1009)
    _reject(_patch(PING, "<I", 16, w.MAX_META_BYTES + 1), "too_big", 1009)
    _reject(_patch(PING, "<H", 20, w.MAX_SECTIONS + 1), "too_big", 1009)


def test_oversize_frame_not_encoded():
    big = np.zeros(w.MAX_FRAME_BYTES // 4, dtype=np.float32)
    with pytest.raises(w.WireError) as e:
        w.encode(w.Series(1, "s", 0, big.size, 0, big))
    assert e.value.close_code == 1009
    with pytest.raises(w.WireError):
        w.encode_frame(w.MsgType.PING, {"pad": "x" * w.MAX_META_BYTES})


@pytest.mark.parametrize("name", ["snapshot_points", "series", "views"])
def test_every_truncation_rejected(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    for n in range(len(buf)):
        _reject(buf[:n], "truncated", 4000)


@pytest.mark.parametrize("depth", [17, 1000, 100_000])
def test_deep_metadata_rejected_without_recursion(depth):
    """A hostile peer's deeply nested metadata is malformed (4000), never a
    RecursionError in the receiver."""
    for opener, closer in ((b"[", b"]"), (b'{"a":', b"}")):
        meta = b'{"x":' + opener * depth + closer * depth + b"}"
        _reject(bytes(GEN._raw(w.MsgType.PING, meta)), "malformed", 4000)
    unclosed = b'{"x":' + b"[" * 200_000 + b"}"
    _reject(bytes(GEN._raw(w.MsgType.PING, unclosed)), "malformed", 4000)


def test_depth_ignores_brackets_in_strings():
    meta = {"nonce": 3, "note": "[[[[[[[[[[[[[[[[[[[[{{{{{{{{{{{{{{{{{{ \\\" ]]"}
    assert w.decode(w.encode_frame(w.MsgType.PING, meta)) == w.Ping(3)
    nested = {"a": [[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]}                 # depth 16: allowed
    assert w.decode_frame(w.encode_frame(0x7777, nested)).meta == nested
    assert w.json_depth(b'{"a":[[["]]]]]]"]]]}') == 4


def test_trailing_bytes_and_layout():
    _reject(PING + b"\0", "malformed", 4000)
    _reject(_patch(PING, "<I", 16, len(PING) - 24 + 1), "malformed", 4000)
    series = (FIXTURES / "series.bin").read_bytes()
    _reject(_patch(series, "<I", 24, 8), "malformed", 4000)      # section shorter than its slot


def test_reserved_flags_minor_ignored():
    buf = _patch(_patch(_patch(PING, "<H", 22, 0xBEEF), "<H", 10, 0xFFFF), "<H", 6, 0xFFFF)
    assert w.decode(buf) == w.Ping(1)


def test_section_alignment():
    for meta in ({}, {"a": "x"}, {"a": "xy"}, {"a": "xyz" * 5}):
        secs = [w.Section.from_array("a", np.arange(3, dtype=np.uint8)),
                w.Section.from_array("b", np.arange(5, dtype=np.float32)),
                w.Section.from_array("c", np.arange(2, dtype=np.int64))]
        buf = w.encode_frame(0x7777, meta, secs)
        f = w.decode_frame(buf)
        starts = []
        pos = 24 + 4 * 3 + struct.unpack_from("<I", buf, 16)[0]
        for s in f.sections:
            pos = (pos + 7) & ~7
            starts.append(pos)
            assert bytes(s.data) == buf[pos:pos + s.data.nbytes]
            pos += s.data.nbytes
        assert all(x % 8 == 0 for x in starts) and pos == len(buf)
        assert f.section("c").array().tolist() == [0, 1]


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------
def _frame(type_, meta, sections=()):
    return w.encode_frame(type_, meta, sections)


@pytest.mark.parametrize("type_, meta", [
    (w.MsgType.HELLO, {}),
    (w.MsgType.HELLO, {"role": "hub"}),
    (w.MsgType.HELLO, {"role": "producer"}),
    (w.MsgType.SUBSCRIBE, {"sub": 1, "view": "x"}),
    (w.MsgType.SUBSCRIBE, {"sub": -1, "view": "x", "request_id": 1}),
    (w.MsgType.SUBSCRIBE, {"sub": 1, "view": "x", "request_id": 1, "max_rate": 0}),
    (w.MsgType.SUBSCRIBE, {"sub": True, "view": "x", "request_id": 1}),
    (w.MsgType.ATTACH, {}),
    (w.MsgType.VIEW_REPLY, {"request_id": 1, "sub": 1, "view": "x"}),
    (w.MsgType.VIEW_REPLY, {"request_id": 1, "sub": 1, "view": "x", "data_view": {},
                            "error": {"code": "limit"}}),
    (w.MsgType.SNAPSHOT, {"sub": 1, "view": "x", "seq": 1, "stamp_ns": 1, "kind": 1,
                          "data_view": {}}),
    (w.MsgType.EVENT, {"view": "x", "type": "select", "detail": {"indices": [-1]}}),
    (w.MsgType.VIEWS, {"views": [{"id": 1, "name": "", "kind": 1, "sources": []}]}),
])
def test_invalid_message_is_bad_request(type_, meta):
    with pytest.raises(w.MessageError) as e:
        w.decode(_frame(type_, meta))
    assert e.value.code == w.ERR_BAD_REQUEST


def test_snapshot_payload_checked():
    dv = {"source": "s", "index": {}, "reduce": {}, "budget": {}}
    ok = w.Snapshot(1, "img", 1, 1, w.Kind.IMAGE, dv, np.zeros((2, 3, 4), np.uint8))
    assert w.decode(w.encode(ok)).data.array().shape == (2, 3, 4)
    for kind, arr in [(w.Kind.IMAGE, np.zeros((2, 3, 3), np.uint8)),
                      (w.Kind.HEATMAP, np.zeros((2, 3), np.uint8)),
                      (w.Kind.POINTS, np.zeros((4, 4), np.float32)),
                      (w.Kind.HISTOGRAM, np.zeros((2, 2), np.float32))]:
        with pytest.raises(w.MessageError):
            w.decode(w.encode(w.Snapshot(1, "v", 1, 1, kind, dv, arr)))
    bad_aux = w.Snapshot(1, "p", 1, 1, w.Kind.POINTS, dv, np.zeros((4, 2), np.float32),
                         aux=np.zeros(3, np.float32))
    with pytest.raises(w.MessageError):
        w.decode(w.encode(bad_aux))


def test_stats_non_finite_are_null():
    dv = {"source": "s", "index": {}, "reduce": {}, "budget": {}}
    s = w.Snapshot(1, "h", 1, 1, w.Kind.HEATMAP, dv, np.zeros((1, 1), np.float32),
                   stats=[math.nan, math.inf, -2.0])
    buf = w.encode(s)
    assert w.decode_frame(buf).meta["stats"] == [None, None, -2.0]
    got = w.decode(buf).stats
    assert math.isnan(got[0]) and math.isnan(got[1]) and got[2] == -2.0


def test_series_bounds():
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.Series(1, "s", 5, 6, 0, [1.0, 2.0])))   # start + M > total
    assert w.decode(w.encode(w.Series(1, "s", 5, 7, 0, [1.0, 2.0]))).values.array().tolist() \
        == [1.0, 2.0]


def test_unknown_metadata_keys_ignored():
    assert w.decode(_frame(w.MsgType.UNSUBSCRIBE, {"sub": 4, "later": {"x": 1}})) == \
        w.Unsubscribe(4)


def test_metadata_is_canonical_utf8():
    buf = w.encode(w.Bye("normal", "fertig, schöne Grüße"))
    f = w.decode_frame(buf)
    text = buf[24:24 + struct.unpack_from("<I", buf, 16)[0]].decode("utf-8")
    assert text == '{"message":"fertig, schöne Grüße","reason":"normal"}'
    assert f.meta["message"] == "fertig, schöne Grüße"


# ---------------------------------------------------------------------------
# 3.1: controls and linked selection
# ---------------------------------------------------------------------------
def _decoded(name):
    return w.decode((FIXTURES / f"v3_1/{name}.bin").read_bytes())


def test_31_golden_messages():
    c = _decoded("controls")
    assert c.run == {"state": "paused", "step": 1200}
    assert [p["name"] for p in c.params] == ["lr", "augment", "optimizer"]
    assert c.actions == [GEN.SAVE]
    assert _decoded("controls_none") == w.Controls({"state": "none"}, [], [])
    assert _decoded("control_step") == w.Control(12, "step", count=5)
    assert _decoded("control_set") == w.Control(13, "set", "lr", 0.5)
    assert _decoded("control_reply_set").value == 0.1          # clamped to max
    assert _decoded("control_reply_queued").status == "queued"
    assert _decoded("control_reply_error").error["code"] == w.ERR_FORBIDDEN
    assert _decoded("welcome").control is True
    assert _decoded("welcome_hub_read_only").control is False
    assert w.decode((FIXTURES / "welcome.bin").read_bytes()).control is None   # a 3.0 server
    sel = _decoded("selection")
    assert (sel.space, sel.items, sel.view, sel.truncated) == ("test", [0, 500], "embedding", False)
    assert sel.details["values"][1] == [2, 1.5]
    assert _decoded("selection_clear") == w.Selection("step", [])
    ev = _decoded("event_select_items")
    assert (ev.detail["space"], ev.detail["items"]) == ("test", [0, 500])
    v = _decoded("views_items")
    assert v.views[0]["sources"][0]["dims"][0]["items"] == "test" and v.views[2]["items"] == "step"


@pytest.mark.parametrize("type_, meta", [
    (w.MsgType.CONTROL, {"op": "pause"}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "jump"}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "set", "value": 1}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "set", "name": "lr"}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "set", "name": "lr", "value": [1]}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "set", "name": "lr", "value": {"x": 1}}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "run"}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "step", "count": 0}),
    (w.MsgType.CONTROL, {"request_id": 1, "op": "step", "count": 1.5}),
    (w.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "pause", "status": "error"}),
    (w.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "pause", "status": "ok",
                               "error": {"code": "forbidden"}}),
    (w.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "set", "status": "queued"}),
    (w.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "run", "status": "done"}),
    (w.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "run", "status": "ok",
                               "message": "x" * (w.TEXT_MAX + 1)}),
    (w.MsgType.CONTROLS, {"params": [], "actions": []}),
    (w.MsgType.CONTROLS, {"run": {"state": "sleeping"}, "params": [], "actions": []}),
    (w.MsgType.CONTROLS, {"run": {"state": "none"}, "actions": []}),
    (w.MsgType.CONTROLS, {"run": {"state": "none"}, "params": [GEN.LR, GEN.LR], "actions": []}),
    (w.MsgType.CONTROLS, {"run": {"state": "none"}, "params": [], "actions": [{"name": "a"}]}),
    (w.MsgType.SELECTION, {"space": "test", "items": [3, 1]}),
    (w.MsgType.SELECTION, {"space": "test", "items": [1, 1]}),
    (w.MsgType.SELECTION, {"space": "", "items": []}),
    (w.MsgType.SELECTION, {"space": "test", "items": list(range(w.MAX_EVENT_INDICES + 1))}),
    (w.MsgType.SELECTION, {"space": "test", "items": [1],
                           "details": {"fields": ["a"], "items": [1], "values": [[1, 2]]}}),
    (w.MsgType.SELECTION, {"space": "test", "items": [1],
                           "details": {"fields": ["a"], "items": [1],
                                       "values": [["x" * (w.DETAIL_TEXT_MAX + 1)]]}}),
    (w.MsgType.EVENT, {"view": "x", "type": "select", "detail": {"space": "test"}}),
    (w.MsgType.EVENT, {"view": "x", "type": "select", "detail": {"items": [1]}}),
    (w.MsgType.WELCOME, {"role": "hub", "control": 1}),
    (w.MsgType.VIEWS, {"views": [{"id": 1, "name": "s", "kind": 4, "sources": [], "items": ""}]}),
])
def test_invalid_31_message_is_bad_request(type_, meta):
    with pytest.raises(w.MessageError) as e:
        w.decode(_frame(type_, meta))
    assert e.value.code == w.ERR_BAD_REQUEST


@pytest.mark.parametrize("param", [
    {"name": "p", "type": "number", "value": 1.0, "min": 2.0, "max": 1.0},
    {"name": "p", "type": "number", "value": 1.0, "log": True},
    {"name": "p", "type": "number", "value": 1.0, "min": 0.0, "max": 1.0, "log": True},
    {"name": "p", "type": "integer", "value": 1.5},
    {"name": "p", "type": "integer", "value": 1, "min": 0.5},
    {"name": "p", "type": "boolean", "value": 1},
    {"name": "p", "type": "choice", "value": "c", "choices": ["a", "b"]},
    {"name": "p", "type": "choice", "value": "a", "choices": []},
    {"name": "p", "type": "choice", "value": "a", "choices": ["a", "a"]},
    {"name": "p", "type": "text", "value": "a"},
    {"name": "p", "type": "number", "value": 1.0, "pending": True},
    {"name": "p", "type": "number"},
])
def test_bad_parameter_entry(param):
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.Controls({"state": "running"}, [param], [])))


def test_parameter_values_validated_and_clamped():
    lr = GEN.LR
    assert w.param_value(lr, 0.5) == 0.1 and w.param_value(lr, 0) == 1e-05
    assert w.param_value(lr, 3e-3) == 3e-3 and isinstance(w.param_value(lr, 1), float)
    batch = {"name": "batch", "type": "integer", "value": 64, "min": 8, "max": 512}
    assert w.param_value(batch, 64.0) == 64 and isinstance(w.param_value(batch, 64.0), int)
    assert w.param_value(batch, 4) == 8 and w.param_value(batch, 10 ** 6) == 512
    free = {"name": "seed", "type": "integer", "value": 0}
    assert w.param_value(free, -7) == -7
    assert w.param_value(GEN.AUGMENT, False) is False
    assert w.param_value(GEN.OPTIMIZER, "sgd") == "sgd"
    for p, bad in [(lr, True), (lr, "0.1"), (batch, 64.5), (batch, False), (free, 2.0 ** 60),
                   (GEN.AUGMENT, 0), (GEN.AUGMENT, "true"), (GEN.OPTIMIZER, "SGD"),
                   (GEN.OPTIMIZER, 1)]:
        with pytest.raises(w.MessageError) as e:
            w.param_value(p, bad)
        assert e.value.code == w.ERR_BAD_VALUE, (p["name"], bad)


def test_selection_items():
    assert w.selection_items([500, 0, 500, 7]) == ([0, 7, 500], False)
    ids, cut = w.selection_items(range(w.MAX_EVENT_INDICES + 10, -1, -1))
    assert cut and ids == list(range(w.MAX_EVENT_INDICES))
    assert w.selection_items([]) == ([], False)


def test_selection_truncated_written_only_when_true():
    assert "truncated" not in w.decode_frame(w.encode(w.Selection("s", [1]))).meta
    assert w.decode(w.encode(w.Selection("s", [1], truncated=True))).truncated is True


def test_control_drops_fields_of_other_ops():
    """A receiver keeps only the fields of the op: 'count' of a set, 'name' of a pause."""
    m = w.decode(_frame(w.MsgType.CONTROL, {"request_id": 2, "op": "pause", "name": "lr",
                                            "count": 3, "args": {"x": 1}}))
    assert m == w.Control(2, "pause")


# ---------------------------------------------------------------------------
# data views
# ---------------------------------------------------------------------------
ATTN = GEN.ATTN
ACT = {"id": 4, "name": "act", "kind": 1, "sources": [
    {"name": "out", "dims": [{"name": "batch", "size": 8}, {"name": "channel", "size": 10},
                             {"name": "h", "size": 300}, {"name": "w", "size": 200}]}],
       "default": {"index": {"batch": 0}, "tile": "channel"}}


def test_default_and_merge():
    assert w.resolve_data_view(ATTN, None) == w.resolve_data_view(ATTN, {}) == {
        "source": "layer0", "index": {"head": 0}, "reduce": {},
        "budget": {"max_extent": 512, "pool": "mean"}}
    r = w.resolve_data_view(ATTN, {"source": "layer1", "budget": {"pool": "max"}})
    assert r["source"] == "layer1" and r["index"] == {"head": 0}
    assert r["budget"] == {"max_extent": 512, "pool": "max"}
    r = w.resolve_data_view(ATTN, {"index": {}, "reduce": {"head": "norm"}})
    assert r["index"] == {} and r["reduce"] == {"head": "norm"}


def test_index_normalised():
    r = w.resolve_data_view(ATTN, {"index": {"head": -1, "key": [None, -1]}})
    assert r["index"] == {"head": 3, "key": [0, 2, 1]}
    r = w.resolve_data_view(ATTN, {"index": {"head": 0, "query": [-100, 100, 2]}})
    assert r["index"]["query"] == [0, 2, 2]


@pytest.mark.parametrize("dv", [
    {"source": "nope"},
    {"index": {"head": 4}},
    {"index": {"head": -5}},
    {"index": {"chan": 0}},
    {"index": {"head": 0, "key": [2, 2]}},
    {"index": {"head": 0, "key": [0, 3, 0]}},
    {"index": {"head": 1.5}},
    {"index": {"head": True}},
    {"index": {"head": 0}, "reduce": {"head": "mean"}},
    {"index": {}, "reduce": {"head": "median"}},
    {"index": {}},                                     # 3 kept dims for a heatmap
    {"index": {"head": 0}, "reduce": {"key": "max"}},  # 1 kept dim
    {"index": {"head": 0}, "tile": "query"},           # 2 kept, tile needs 3
    {"budget": {"max_extent": 0}},
    {"budget": {"pool": "median"}},
    [],
])
def test_bad_data_view(dv):
    with pytest.raises(w.MessageError) as e:
        w.resolve_data_view(ATTN, dv)
    assert e.value.code == w.ERR_BAD_DATA_VIEW


def test_budget_clamped():
    r = w.resolve_data_view(ATTN, {"budget": {"max_extent": 10 ** 9}})
    assert r["budget"]["max_extent"] == w.LIMIT_MAX_EXTENT
    r = w.resolve_data_view(GEN.EMBEDDING, {"budget": {"max_points": 10 ** 9}})
    assert r["budget"] == {"max_points": w.LIMIT_MAX_POINTS}


def test_series_resolution():
    assert w.resolve_data_view(GEN.LOSS, None, retain=10000) == {
        "budget": {"max_backfill": 10000}}
    assert w.resolve_data_view(GEN.LOSS, {"budget": {"max_backfill": 0}}) == {
        "budget": {"max_backfill": 0}}
    with pytest.raises(w.MessageError):
        w.resolve_data_view(GEN.LOSS, {"index": {"step": [0, 5]}})


def test_kinds_need_their_dims():
    img = {"id": 5, "name": "img", "kind": 2, "sources": [
        {"name": "rgb", "dims": [{"name": "h", "size": 2000}, {"name": "w", "size": 100}]}]}
    r = w.resolve_data_view(img, {"budget": {"pool": "max"}})
    assert r["budget"] == {"max_extent": 512}
    assert w.payload_layout(img, r)["extent"] == [512, 100]
    with pytest.raises(w.MessageError):
        w.resolve_data_view(img, {"tile": "h"})
    hist = {"id": 6, "name": "hist", "kind": 5, "sources": ACT["sources"]}
    r = w.resolve_data_view(hist, {"index": {"batch": [0, 4]}, "reduce": {"channel": "mean"}})
    assert r["budget"] == {} and w.payload_layout(hist, r)["axes"] == []


def test_mosaic_layout():
    r = w.resolve_data_view(ACT, None)
    lay = w.payload_layout(ACT, r)
    assert lay["tile"] == {"dim": "channel", "rows": 3, "cols": 4}   # ceil(sqrt(10)) = 4
    t, rows, cols = lay["axes"]
    assert (t["dim"], t["size"], rows["size"], cols["size"]) == ("channel", 10, 170, 128)
    assert lay["extent"] == [3 * 170, 4 * 128]
    assert max(lay["extent"]) <= 512


def test_points_stride_sampling_axis():
    r = w.resolve_data_view(GEN.EMBEDDING, {"index": {"point": [100, 1000, 3]}})
    (a,) = w.payload_layout(GEN.EMBEDDING, r)["axes"]
    assert (a["start"], a["step"], a["count"]) == (100, 3, 300)
    assert a["size"] == 4 == r["budget"]["max_points"]
    first = [a["start"] + (i * a["count"] // a["size"]) * a["step"] for i in range(a["size"])]
    assert first == [100, 325, 550, 775]


def test_pooled_axis_cells_cover_the_range():
    r = w.resolve_data_view(ATTN, {"index": {"head": 0}, "budget": {"max_extent": 2}})
    _, key = w.payload_layout(ATTN, r)["axes"]
    assert (key["count"], key["size"]) == (3, 2)
    cells = [(i * key["count"] // key["size"], -(-(i + 1) * key["count"] // key["size"]))
             for i in range(key["size"])]
    assert cells == [(0, 2), (1, 3)]                  # torch adaptive pooling windows


# ---------------------------------------------------------------------------
# the header, the codec and the page agree
# ---------------------------------------------------------------------------
def _c_value(text: str):
    text = text.strip()
    if text.startswith('"'):
        return json.loads(text)
    text = re.sub(r"(?<=[0-9a-fA-F])(u|LL)\b", "", text)
    if not re.fullmatch(r"[0-9a-fA-Fx()<< +.-]+", text):
        raise ValueError(text)
    return eval(text, {"__builtins__": {}})   # digits, parentheses, shifts and signs only


def _header_constants() -> dict:
    src = HEADER_H.read_text()
    out = {}
    for m in re.finditer(r"^#define (CALIPER_NET_\w+)\s+(.+?)\s*(?:/\*.*)?$", src, re.M):
        out[m.group(1)] = _c_value(m.group(2))
    for m in re.finditer(r"^\s+(CALIPER_NET_\w+)\s*=\s*(\w+)", src, re.M):
        out[m.group(1)] = int(m.group(2), 0)
    return out


def _expected(name: str):
    """The _wire value a header constant must equal."""
    short = name[len("CALIPER_NET_"):]
    for prefix, enum in (("MSG_", w.MsgType), ("KIND_", w.Kind), ("DTYPE_", w.DType),
                         ("ENCODING_", w.Encoding)):
        if short.startswith(prefix):
            return int(enum[short[len(prefix):]])
    if short.startswith("REDUCE_"):
        v = short[len("REDUCE_"):].lower()
        assert v in w.REDUCE_OPS
        return v
    if short.startswith("POOL_"):
        v = short[len("POOL_"):].lower()
        assert v in w.POOL_OPS
        return v
    alias = {"PROTOCOL_MAJOR": "MAJOR", "PROTOCOL_MINOR": "MINOR"}
    return getattr(w, alias.get(short, short))


def test_header_matches_codec():
    consts = _header_constants()
    assert len(consts) > 100
    for name, value in consts.items():
        assert _expected(name) == value, name
    assert struct.calcsize("<IHHHHIIHH") == consts["CALIPER_NET_HEADER_BYTES"]


def test_page_tables_match_header():
    page = _pages()
    both = page + PAGE_32.read_text()
    consts = _header_constants()
    rows = re.findall(r"^\| `0x([0-9a-f]{4})` \| `(\w+)` \|", both, re.M)
    assert {int(i, 16): n for i, n in rows} == {int(t): t.name.lower() for t in w.MsgType}
    codes = dict(re.findall(r"^\| (\d{4}) \| `(CALIPER_NET_CLOSE_\w+)` \|", page, re.M))
    assert codes and all(consts[n] == int(c) for c, n in codes.items())
    assert set(codes.values()) == {n for n in consts if n.startswith("CALIPER_NET_CLOSE_")}
    limits = re.findall(r"^\| `(CALIPER_NET_\w+)` \| ([\d,]+)", both, re.M)
    assert len(limits) >= 25
    for n, v in limits:
        assert consts[n] == int(v.replace(",", "")), n
    errs = set(re.findall(r"^\| `([a-z_]+)` \| ", _section(page, "Error codes"), re.M))
    p32 = PAGE_32.read_text()                       # the 3.2 draft's codes, in its own section
    errs |= set(re.findall(r"^\| `([a-z_]+)` \| ", _section(p32, "Error codes and check order"), re.M))
    assert errs == {v for n, v in consts.items() if n.startswith("CALIPER_NET_ERR_")}
    dtypes = re.findall(r"^\| (\d) \| `(\w+)` \| (\d) \|", both, re.M)   # u16: the 3.2 page
    assert {int(i): int(s) for i, _, s in dtypes} == {int(k): v for k, v in w.DTYPE_SIZE.items()}
    assert {int(i): n.upper() for i, n, _ in dtypes} == {int(d): d.name for d in w.DType}


def test_page_frame_layout_matches_struct():
    page = _pages()
    rows = re.findall(r"^\| (\d+) \| (\d) \| (u\d+) \| `(\w+)` \|", page, re.M)
    names = ["magic", "major", "minor", "type", "flags", "frame_len", "meta_len",
             "section_count", "reserved"]
    assert [r[3] for r in rows] == names
    off = 0
    for (o, size, typ, _), code in zip(rows, "IHHHHIIHH"):
        assert int(o) == off and int(size) == struct.calcsize(code) == int(typ[1:]) // 8
        off += int(size)
    assert off == w.HEADER_BYTES


def test_page_worked_example_is_the_golden_frame():
    page = _pages()
    part = _section(page, "Worked example")
    block = part[part.index("```text") + len("```text"):]
    block = block[:block.index("```")]
    data = bytearray()
    for line in block.strip().splitlines():
        off, *hexes = line.split()
        assert int(off, 16) == len(data)
        data += bytes(int(h, 16) for h in hexes)
    assert bytes(data) == (FIXTURES / "series.bin").read_bytes()
    meta = part[part.index("```json") + len("```json"):]
    meta = meta[:meta.index("```")].strip()
    assert meta.encode() == data[28:28 + struct.unpack_from("<I", data, 16)[0]]


def test_page_embeds_the_header():
    assert '--8<-- "include/caliper/live_net.h"' in _pages()
