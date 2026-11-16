"""The protocol 3.2 draft's scenes subset in caliper/_wire.py: the golden
frames of tests/fixtures/wire_v3/v3_2 decode as described and encode back to
the same bytes, every frame of v3_2/invalid is refused as its expect.json says,
the limits, entity paths, the VIEWS key "scene" and the default scene data
view, and agreement with the 3.2 block of include/caliper/live_net.h and the
reference page docs/references/wire-protocol-3-2.md.

Needs numpy and pytest only: ``pytest --noconftest python/tests/test_wire_32.py``."""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "wire_v3"
V32 = FIXTURES / "v3_2"
HEADER_H = ROOT / "include" / "caliper" / "live_net.h"
PAGE = ROOT / "docs" / "references" / "wire-protocol-3-2.md"

sys.path.insert(0, str(ROOT / "python"))
from caliper import _wire as w  # noqa: E402

VALID = sorted(f"v3_2/{p.stem}" for p in V32.glob("*.bin"))
DECODE_ONLY = {"v3_2/scene_update_points_stamped", "v3_2/scene_update_mesh_normals"}
EXPECT = json.loads((V32 / "invalid" / "expect.json").read_text())


def test_the_frames_are_there():
    assert len(VALID) >= 16
    types = {w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).type for n in VALID}
    assert {int(w.MsgType.SCENE_UPDATE), int(w.MsgType.SCENE_CLEAR)} <= types
    parts = {w.decode((FIXTURES / f"{n}.bin").read_bytes()).part for n in VALID
             if n.startswith("v3_2/scene_update")}
    assert parts == {"points", "lines", "mesh"}
    assert len(EXPECT) >= 15
    assert {"error"} <= set().union(*EXPECT.values()) and {"reason"} <= set().union(*EXPECT.values())


@pytest.mark.parametrize("name", VALID)
def test_frame_matches_description(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    desc = json.loads((FIXTURES / f"{name}.json").read_text())
    h = desc["header"]
    assert struct.unpack_from("<IHHHHIIHH", buf, 0) == (
        h["magic"], h["major"], h["minor"], h["type"], h["flags"], h["frame_len"],
        h["meta_len"], h["section_count"], h["reserved"])
    assert h["minor"] == w.MINOR_3_2 == 2 and h["flags"] == 0 and len(buf) <= 600
    f = w.decode_frame(buf)
    assert f.meta == desc["meta"]
    for s, d in zip(f.sections, desc["sections"]):
        assert (s.name, s.dtype, list(s.shape), s.data.nbytes) == (
            d["name"], d["dtype"], d["shape"], d["length"])
        assert bytes(s.data) == buf[d["offset"]:d["offset"] + d["length"]]
        assert s.array().reshape(-1).tolist() == d["values"]
    try:
        tname = w.MsgType(f.type).name.lower()
    except ValueError:
        tname = None
    assert tname == desc["type_name"]


@pytest.mark.parametrize("name", sorted(set(VALID) - DECODE_ONLY))
def test_frame_round_trips(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    assert w.encode(w.decode(buf), minor=w.MINOR_3_2) == buf


def test_decode_only_frames_ignore_what_is_not_built():
    m = w.decode((V32 / "scene_update_points_stamped.bin").read_bytes())
    assert isinstance(m, w.SceneUpdate) and set(m.arrays) == {"positions", "colors"}
    again = w.decode_frame(w.encode(m, minor=w.MINOR_3_2))
    assert [s.name for s in again.sections] == ["positions", "colors"]
    m = w.decode((V32 / "scene_update_mesh_normals.bin").read_bytes())
    assert set(m.arrays) == {"positions", "indices"}


def test_golden_values():
    m = w.decode((V32 / "scene_update_mesh.bin").read_bytes())
    assert (m.part, m.path, m.view, m.sub, m.seq) == ("mesh", "surface", "landscape", 2, 39)
    assert m.arrays["indices"].array().tolist() == [[0, 1, 2], [1, 3, 2]]
    assert m.arrays["values"].array().tolist() == [-1.5, -0.5, 0.25, 1.0]
    assert (m.params["vmin"], m.params["vmax"], m.params["colormap"]) == (-4.0, 2.0, 0)
    s = w.decode((V32 / "scene_update_lines_strips.bin").read_bytes())
    assert s.params["mode"] == "strips" and s.params["color_per"] == "strip"
    assert s.arrays["strip_lengths"].array().tolist() == [3, 2]
    assert s.arrays["colors"].array().shape == (2, 4)
    seg = w.decode((V32 / "scene_update_lines_segments.bin").read_bytes())
    assert seg.path == "" and seg.static and seg.params["width"] == 1.0
    tri = w.decode((V32 / "scene_update_mesh_colors.bin").read_bytes())
    assert tri.arrays["positions"].array().shape == (3, 2) and tri.static and tri.state
    c = w.decode((V32 / "scene_clear_reset.bin").read_bytes())
    assert isinstance(c, w.SceneClear) and c.reset and c.state and c.path == ""
    c = w.decode((V32 / "scene_clear.bin").read_bytes())
    assert c.recursive and c.parts == ["points", "mesh"] and not c.reset
    v = w.decode((V32 / "views_scene.bin").read_bytes())
    scene = next(e for e in v.views if e["kind"] == w.Kind.SCENE)
    assert w.scene_view_info(scene) == {"space": "3d", "up": "+z", "units": ""}
    r = w.decode((V32 / "view_reply_scene.bin").read_bytes())
    assert r.data_view == w.SCENE_DATA_VIEW
    hello = w.decode((V32 / "hello.bin").read_bytes())
    assert w.CAP_SCENES in hello.capabilities


def test_31_receivers_find_the_frames_well_formed():
    for n in VALID:
        f = w.decode_frame((FIXTURES / f"{n}.bin").read_bytes())
        assert f.type in (0x01, 0x02, 0x20, 0x24, 0x70, 0x72)


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_invalid_frame_refused(name):
    buf = (V32 / "invalid" / name).read_bytes()
    want = EXPECT[name]
    if "reason" in want:
        with pytest.raises(w.WireError) as e:
            w.decode(buf)
        assert (e.value.reason, e.value.close_code) == (want["reason"], want["close_code"])
    else:
        with pytest.raises(w.MessageError) as e:
            w.decode(buf)
        assert e.value.code == want["error"], e.value.message


def test_every_truncation_of_a_scene_frame_is_refused():
    buf = (V32 / "scene_update_mesh.bin").read_bytes()
    for n in range(len(buf)):
        with pytest.raises(w.WireError):
            w.decode(buf[:n])


def _update(part, sections, **keys):
    meta = {"part": part, "path": "p", "seq": 1, "stamp_ns": 1, "sub": 2, "view": "s", **keys}
    return w.encode_frame(w.MsgType.SCENE_UPDATE, meta,
                          [w.Section.from_array(n, a) for n, a in sections.items()])


def _refused(buf, code=w.ERR_BAD_REQUEST, match=None):
    with pytest.raises(w.MessageError) as e:
        w.decode(buf)
    assert e.value.code == code, e.value.message
    if match:
        assert match in e.value.message


def test_limits():
    big = np.zeros((w.MAX_BATCH_POINTS + 1, 2), np.float32)
    _refused(_update("points", {"positions": big}), match="more than 4194304")
    ok = np.zeros((w.MAX_BATCH_POINTS, 2), np.float32)
    assert w.decode(_update("points", {"positions": ok})).arrays["positions"].shape[0] == ok.shape[0]
    n = w.MAX_BATCH_ITEMS + 1
    _refused(_update("lines", {"positions": np.zeros((n, 2), np.float32),
                               "strip_lengths": np.ones(n, np.int32)}, mode="strips"),
             match="strips, more than")
    tris = np.zeros((w.MAX_BATCH_POINTS + 1, 3), np.int32)
    _refused(_update("mesh", {"positions": np.zeros((3, 3), np.float32), "indices": tris}),
             match="triangles, more than")


def test_message_rules():
    p3 = np.zeros((4, 3), np.float32)
    rgba = np.zeros((4, 4), np.uint8)
    _refused(_update("points", {"positions": p3, "values": np.zeros(3, np.float32)}))   # rows
    _refused(_update("points", {"positions": p3, "radii": np.zeros((4, 1), np.float32)}))
    _refused(_update("points", {"positions": p3}, color=[1, 2, 3]))
    _refused(_update("points", {"positions": p3}, radius=0))
    _refused(_update("lines", {"positions": p3}, mode="segments", width=-1))
    _refused(_update("lines", {"positions": p3}, mode="segments", color_per="strip"))
    _refused(_update("lines", {"positions": p3, "strip_lengths": np.array([4], np.int32)},
                     mode="segments"))
    _refused(_update("lines", {"positions": p3, "strip_lengths": np.array([2, 2], np.int32),
                               "colors": rgba}, mode="strips", color_per="strip"))  # 4 rows, 2 strips
    _refused(_update("lines", {"positions": p3, "strip_lengths": np.array([4, 0], np.int32)},
                     mode="strips"))
    _refused(_update("mesh", {"positions": p3}))                                      # 4 % 3
    _refused(_update("mesh", {"positions": p3[:3], "indices": np.zeros((2, 2), np.int32)}))
    _refused(_update("points", {"positions": p3.astype(np.float16)}))
    # per-strip colours, and values per vertex, are fine
    ok = w.decode(_update("lines", {"positions": p3, "strip_lengths": np.array([3, 1], np.int32),
                                    "colors": rgba[:2]}, mode="strips", color_per="strip"))
    assert ok.arrays["colors"].shape == (2, 4)
    ok = w.decode(_update("lines", {"positions": p3, "values": np.zeros(4, np.float32)},
                          mode="segments", vmin=0, vmax=1))
    assert ok.params["vmin"] == 0
    # unknown keys and sections are ignored; the root may hold primitives
    ok = w.decode(_update("points", {"positions": p3, "class_ids": np.zeros(4, np.int32)},
                          labels=["a"], future=1))
    assert set(ok.arrays) == {"positions"} and "labels" not in ok.params


def test_entity_paths():
    good = ["", "a", "world/cloud", "a.b-c_d/E9", "/".join(["s"] * 16), "x" * 63,
            "/".join(["y" * 63] * 4)[:255]]
    bad = ["/", "a/", "/a", "a//b", ".", "a/..", "a/./b", "a b", "é", "x" * 64,
           "/".join(["s"] * 17), "/".join(["y" * 63] * 5), None, 3]
    for p in good:
        assert w.entity_path_ok(p), p
    for p in bad:
        assert not w.entity_path_ok(p), p


def test_scene_view_info():
    assert w.scene_view_info({"kind": 6}) == {"space": "3d", "up": "+y", "units": "m"}
    assert w.scene_view_info({"kind": 6, "scene": {"space": "2d"}})["up"] == "-y"
    assert w.scene_view_info({"kind": 6, "scene": {"view_camera": {"azimuth": 30}}})["up"] == "+y"
    for sc in ({"space": "4d"}, {"space": "2d", "up": "+z"}, {"up": "z"}, {"units": "x" * 16},
               {"units": 3}, "3d"):
        with pytest.raises(w.MessageError):
            w.scene_view_info({"kind": 6, "scene": sc})


def test_default_scene_data_view_only():
    view = {"id": 1, "name": "s", "kind": int(w.Kind.SCENE), "sources": []}
    for req in (None, {}, w.SCENE_DATA_VIEW):
        assert w.resolve_data_view(view, req) == w.SCENE_DATA_VIEW
    assert w.resolve_data_view(view, None) is not w.SCENE_DATA_VIEW       # a copy
    for req in ({"budget": {"max_points": 10}}, {"entities": {"include": ["a/**"]}},
                {"source": "data"}, [1]):
        with pytest.raises(w.MessageError) as e:
            w.resolve_data_view(view, req)
        assert e.value.code == w.ERR_BAD_DATA_VIEW


def test_header_block_is_separate():
    """Every 3.2 constant of live_net.h sits in the one block after its 3.2
    marker, and the codec agrees with each (test_wire.py checks the values)."""
    src = HEADER_H.read_text()
    head, block = src.split("PROTOCOL 3.2 DRAFT: THE SCENES SUBSET")
    for name in ("MINOR_3_2", "CAP_SCENES", "KIND_SCENE", "MSG_SCENE_UPDATE", "MSG_SCENE_CLEAR",
                 "PART_POINTS", "MAX_BATCH_POINTS", "SCENE_DEFAULT_MAX_VERTICES"):
        assert f"CALIPER_NET_{name}" in block and f"CALIPER_NET_{name}" not in head
    assert "#define CALIPER_NET_PROTOCOL_MINOR 1u" in head


def test_page_example_is_the_golden_frame():
    page = PAGE.read_text()
    part = page[page.index("## Golden frames"):]
    block = part[part.index("```text") + len("```text"):]
    block = block[:block.index("```")]
    data = bytearray()
    for line in block.strip().splitlines():
        off, *hexes = line.split()
        assert int(off, 16) == len(data)
        data += bytes(int(h, 16) for h in hexes)
    assert bytes(data) == (V32 / "scene_update_mesh.bin").read_bytes()
    meta = part[part.index("```json") + len("```json"):]
    meta = meta[:meta.index("```")].strip()
    n = struct.unpack_from("<H", data, 20)[0]
    mo = 24 + 4 * n
    assert meta.encode() == data[mo:mo + struct.unpack_from("<I", data, 16)[0]]
    rows = dict(re.findall(r"^\| `0x([0-9a-f]{4})` \| `(\w+)` \|", page, re.M))
    assert rows == {"0070": "scene_update", "0072": "scene_clear"}
