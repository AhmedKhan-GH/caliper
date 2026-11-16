"""The protocol 3.2 draft's scenes in caliper/_wire.py: the golden frames of
tests/fixtures/wire_v3/v3_2 decode as described and encode back to the same
bytes, every frame of v3_2/invalid is refused as its expect.json says, every
part and its rules, the stamps, entity transforms and class tables, the
limits, entity paths, the VIEWS key "scene" with its starting camera and the
default scene data view, and agreement with the 3.2 block of
include/caliper/live_net.h and the reference page
docs/references/wire-protocol-3-2.md.

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
DECODE_ONLY = {"v3_2/scene_update_points_future"}
EXPECT = json.loads((V32 / "invalid" / "expect.json").read_text())


def test_the_frames_are_there():
    assert len(VALID) >= 36
    types = {w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).type for n in VALID}
    assert {0x70, 0x71, 0x72, 0x73} <= types
    msgs = [w.decode((FIXTURES / f"{n}.bin").read_bytes()) for n in VALID]
    parts = {m.part for m in msgs if isinstance(m, w.SceneUpdate)}
    assert parts == set(w.PARTS)
    forms = {m.params["form"] for m in msgs if isinstance(m, w.SceneUpdate) and m.part == "image"}
    assert forms == set(w.FORMS)
    modes = {m.params["mode"] for m in msgs if isinstance(m, w.SceneUpdate) and m.part == "lines"}
    assert modes == {"segments", "strips", "edges"}
    stamped = {type(m).__name__ for m in msgs if getattr(m, "stamp", None) is not None
               or getattr(m, "stamps", None) is not None}
    assert stamped == {"SceneUpdate", "SceneTransforms", "SceneClear", "SceneClasses"}
    assert any(s.dtype == w.DType.U16 for n in VALID
               for s in w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).sections)
    assert len(EXPECT) >= 55
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
    m = w.decode((V32 / "scene_update_points_future.bin").read_bytes())
    assert isinstance(m, w.SceneUpdate) and set(m.arrays) == {"positions"}
    again = w.decode_frame(w.encode(m, minor=w.MINOR_3_2))
    assert [s.name for s in again.sections] == ["positions"]


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
    assert w.scene_view_info(scene) == {"space": "3d", "up": "+z", "units": "",
                                        "view_camera": None}
    r = w.decode((V32 / "view_reply_scene.bin").read_bytes())
    assert r.data_view == w.SCENE_DATA_VIEW
    hello = w.decode((V32 / "hello.bin").read_bytes())
    assert w.CAP_SCENES in hello.capabilities


def _d(name):
    return w.decode((V32 / f"{name}.bin").read_bytes())


def test_golden_values_of_the_other_parts():
    st = _d("scene_update_points_stamped")
    assert st.stamp.array().tolist() == [1200, 1763200000123456789]
    assert _d("scene_update_points").stamp is None
    lab = _d("scene_update_points_labels")
    assert lab.params["labels"] == ["a", "b", "c"]
    assert lab.arrays["class_ids"].array().tolist() == [0, 1, 1]
    e = _d("scene_update_lines_edges")
    assert (e.params["mode"], e.params["color_per"]) == ("edges", "edge")
    assert e.arrays["indices"].array().tolist() == [[0, 1], [1, 2], [2, 0]]
    assert e.arrays["values"].array().shape == (3,)
    a = _d("scene_update_arrows")
    assert a.arrays["vectors"].array().tolist() == [[0, 0, 1], [0, 0.5, 0]]
    b = _d("scene_update_boxes")
    assert b.static and b.stamp is None and b.arrays["rotations"].array().tolist() == [[0, 0, 0, 1]]
    b2 = _d("scene_update_boxes_2d")
    assert b2.params == {"labels": ["car"], "fill": True}
    assert b2.arrays["half_sizes"].array().shape == (2, 2)
    t = _d("scene_update_mesh_textured")
    assert t.arrays["texture"].array().shape == (2, 2, 4) and t.arrays["uvs"].array().shape == (4, 2)
    n = _d("scene_update_mesh_normals")
    assert set(n.arrays) == {"positions", "indices", "normals"}
    dep = _d("scene_update_image_depth")
    assert dep.arrays["pixels"].dtype == w.DType.U16
    assert dep.arrays["pixels"].array().tolist() == [[500, 1000, 0], [1500, 2000, 2500]]
    assert (dep.params["form"], dep.params["scale"], dep.params["cloud"]) == ("depth", 0.001, True)
    enc = _d("scene_update_image_encoded")
    assert bytes(enc.arrays["pixels"].data)[:8] == b"\x89PNG\r\n\x1a\n"
    assert enc.params["format"] == "png"
    seg = _d("scene_update_image_segmentation_u16")
    assert seg.arrays["pixels"].array().tolist() == [[0, 300], [65535, 7]]
    cam = _d("scene_update_camera")
    assert cam.arrays == {} and cam.params == {"fx": 525.0, "fy": 525.0, "cx": 319.5,
                                               "cy": 239.5, "width": 640, "height": 480,
                                               "convention": "rdf"}
    assert _d("scene_update_camera_rub").params["plane_distance"] == 0.25
    tr = _d("scene_transforms")
    assert isinstance(tr, w.SceneTransforms) and tr.paths == ["arm/link0", "arm/link0/link1"]
    m = tr.matrices.array().reshape(2, 4, 4)
    assert m[1].T[:3, 3].tolist() == [0, 0, 0.5]                     # column-major
    assert tr.stamp.array().tolist() == [1200, 1763200000123456789] and tr.stamps is None
    ts = _d("scene_transforms_stamps")
    assert ts.stamps.array().tolist() == [[1200, 1763200000123456789], [1201, 1763200000128456789]]
    assert _d("scene_transforms_static").static and _d("scene_transforms_static").state
    cl = _d("scene_classes")
    assert isinstance(cl, w.SceneClasses) and [c["id"] for c in cl.classes] == [0, 1, 65535]
    assert _d("scene_classes_stamped").classes == []
    c = _d("scene_clear_stamped")
    assert c.parts == ["transform", "boxes"] and c.stamp is not None


def test_31_receivers_find_the_frames_well_formed():
    for n in VALID:
        f = w.decode_frame((FIXTURES / f"{n}.bin").read_bytes())
        assert f.type in (0x01, 0x02, 0x20, 0x24, 0x70, 0x71, 0x72, 0x73)


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
    pairs = np.zeros((w.MAX_BATCH_POINTS + 1, 2), np.int32)
    _refused(_update("lines", {"positions": np.zeros((2, 3), np.float32), "indices": pairs},
                     mode="edges"), match="edges, more than")
    items = np.zeros((w.MAX_BATCH_ITEMS + 1, 2), np.float32)
    _refused(_update("arrows", {"origins": items, "vectors": items}), match="arrows, more than")
    _refused(_update("boxes", {"centers": items, "half_sizes": items}), match="boxes, more than")
    _refused(_update("points", {"positions": np.zeros((2, 2), np.float32)},
                     labels=["l"] * (w.MAX_LABELS + 1)))
    side = np.zeros((1, w.MAX_IMAGE_SIDE), np.uint8)
    assert w.decode(_update("image", {"pixels": side}, form="segmentation"))
    _refused(_update("image", {"pixels": np.zeros((1, w.MAX_IMAGE_SIDE + 1), np.uint8)},
                     form="segmentation"))
    _refused(_update("mesh", {"positions": np.zeros((3, 2), np.float32),
                              "uvs": np.zeros((3, 2), np.float32),
                              "texture": np.zeros((1, w.MAX_IMAGE_SIDE + 1, 4), np.uint8)}))


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
    _refused(_update("lines", {"positions": p3, "strip_lengths": np.array([4], np.int32)},
                     mode="strips", color_per="edge"))
    _refused(_update("lines", {"positions": p3, "indices": np.zeros((1, 2), np.int32)},
                     mode="strips"))
    _refused(_update("lines", {"positions": p3}, mode="loops"))
    _refused(_update("mesh", {"positions": p3}))                                      # 4 % 3
    _refused(_update("mesh", {"positions": p3[:3], "indices": np.zeros((2, 2), np.int32)}))
    _refused(_update("points", {"positions": p3.astype(np.float16)}))
    _refused(_update("points", {"positions": p3}, labels=[]))
    _refused(_update("points", {"positions": p3}, labels="a"))
    _refused(_update("boxes", {"centers": p3, "half_sizes": p3}, fill=1))
    _refused(_update("image", {"pixels": np.zeros((2, 2), np.uint8)}, form="color", opacity=-0.5))
    _refused(_update("image", {"pixels": np.zeros((2, 2), np.uint8)}, form="encoded",
                     format="gif"))
    _refused(_update("image", {"pixels": np.zeros((2, 2), np.float32)}, form="segmentation"))
    _refused(_update("image", {"pixels": np.zeros((4,), np.uint8)}, form="color"))
    _refused(_update("image", {"pixels": np.zeros((2, 2), np.uint16)}, form="color"))
    _refused(_update("image", {"pixels": np.zeros(2, np.uint8)}, form="depth", scale=0))
    _refused(_update("camera", {}, fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=8193, height=2,
                     convention="rdf"))
    _refused(_update("camera", {}, fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2,
                     convention="flu"))
    # per-strip colours, values per vertex, per-edge and per-vertex colours of edges are fine
    ok = w.decode(_update("lines", {"positions": p3, "strip_lengths": np.array([3, 1], np.int32),
                                    "colors": rgba[:2]}, mode="strips", color_per="strip"))
    assert ok.arrays["colors"].shape == (2, 4)
    ok = w.decode(_update("lines", {"positions": p3, "values": np.zeros(4, np.float32)},
                          mode="segments", vmin=0, vmax=1))
    assert ok.params["vmin"] == 0
    edges = np.array([[0, 1], [2, 3], [3, 0]], np.int32)
    ok = w.decode(_update("lines", {"positions": p3, "indices": edges, "colors": rgba},
                          mode="edges"))
    assert ok.arrays["colors"].shape == (4, 4) and ok.arrays["indices"].shape == (3, 2)
    ok = w.decode(_update("lines", {"positions": p3, "indices": edges,
                                    "colors": rgba[:3]}, mode="edges", color_per="edge"))
    assert ok.arrays["colors"].shape == (3, 4)
    # labels: one for all or one per item; a camera holds no array
    assert w.decode(_update("points", {"positions": p3}, labels=["x"])).params["labels"] == ["x"]
    assert len(w.decode(_update("boxes", {"centers": p3, "half_sizes": p3},
                                labels=list("abcd"))).params["labels"]) == 4
    cam = w.decode(_update("camera", {"positions": p3}, fx=1.5, fy=2, cx=0, cy=0, width=2,
                           height=3, convention="rub"))
    assert cam.arrays == {} and cam.params["width"] == 2
    # unknown keys and sections are ignored; the root may hold primitives
    ok = w.decode(_update("points", {"positions": p3, "normals": np.zeros((4, 3), np.float32)},
                          future=1, fill=True))
    assert set(ok.arrays) == {"positions"} and ok.params == {}
    # the colour pixels may be (H, W); an empty batch is well formed
    assert w.decode(_update("image", {"pixels": np.zeros((2, 3), np.uint8)}, form="color"))
    assert w.decode(_update("points", {"positions": np.zeros((0, 3), np.float32)}))


def test_stamps():
    p3 = np.zeros((2, 3), np.float32)
    t = np.array([5, 7], np.int64)
    m = w.decode(_update("points", {"positions": p3, "stamp": t}))
    assert m.stamp.array().tolist() == [5, 7]
    assert w.decode(_update("points", {"positions": p3,
                                       "stamp": np.zeros(w.MAX_TIMELINES, np.int64)}))
    absent = np.array([1, w.STAMP_ABSENT], np.int64)
    assert w.decode(_update("points", {"positions": p3, "stamp": absent})).stamp.array()[1] == \
        -(1 << 63)
    for bad in (np.zeros(0, np.int64), np.zeros(9, np.int64), np.zeros(2, np.int32)):
        _refused(_update("points", {"positions": p3, "stamp": bad}))
    _refused(_update("points", {"positions": p3, "stamp": t}, static=True))
    # a scene update takes one stamp; "stamps" is not one of its sections
    m = w.decode(_update("points", {"positions": p3, "stamps": np.zeros((2, 2), np.int64)}))
    assert m.stamp is None and set(m.arrays) == {"positions"}
    # the stamp is the last section written
    msg = w.SceneUpdate(1, "s", 1, 1, "p", "points", {"positions": p3}, stamp=t)
    assert [x.name for x in w.decode_frame(w.encode(msg)).sections] == ["positions", "stamp"]


def test_transforms_and_classes():
    mats = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (3, 1))
    msg = w.SceneTransforms(1, "s", 2, 3, ["a", "a/b", "c"], mats, cursor=4)
    back = w.decode(w.encode(msg))
    assert back.paths == ["a", "a/b", "c"] and back.cursor == 4
    assert back.matrices.array().tolist() == mats.tolist()
    big = np.zeros((w.MAX_TRANSFORMS + 1, 16), np.float32)
    many = [f"e{i}" for i in range(w.MAX_TRANSFORMS + 1)]
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.SceneTransforms(1, "s", 2, 3, many, big)))
    ok = w.SceneTransforms(1, "s", 2, 3, many[:-1], big[:-1])
    assert len(w.decode(w.encode(ok)).paths) == w.MAX_TRANSFORMS
    table = [{"id": i, "label": f"c{i}", "color": [i % 256, 0, 0, 255]}
             for i in range(w.MAX_CLASSES)]
    assert len(w.decode(w.encode(w.SceneClasses(1, "s", 2, 3, table))).classes) == w.MAX_CLASSES
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.SceneClasses(1, "s", 2, 3, table + [
            {"id": 65535, "label": "x", "color": [0, 0, 0, 0]}])))
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.SceneClasses(1, "s", 2, 3, [{"id": 1, "label": "x" * 64,
                                                          "color": [0, 0, 0, 0]}])))


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
    base = {"space": "3d", "up": "+y", "units": "m", "view_camera": None}
    assert w.scene_view_info({"kind": 6}) == base
    assert w.scene_view_info({"kind": 6, "scene": {"space": "2d"}})["up"] == "-y"
    assert w.scene_view_info({"kind": 6, "scene": {"view_camera": {"azimuth": 30}}}) == dict(
        base, view_camera={"azimuth": 30})
    for sc in ({"space": "4d"}, {"space": "2d", "up": "+z"}, {"up": "z"}, {"units": "x" * 16},
               {"units": 3}, "3d"):
        with pytest.raises(w.MessageError):
            w.scene_view_info({"kind": 6, "scene": sc})
    good = [{}, {"target": [0, 1, 2.5]}, {"distance": 3}, {"elevation": -89},
            {"elevation": 89, "azimuth": -720.5, "target": [0, 0, 0], "distance": 0.5},
            "rig/left"]
    for vc in good:
        assert w.scene_view_info({"kind": 6, "scene": {"view_camera": vc}})["view_camera"] == vc
    # a view_camera a viewer cannot read is ignored, never refused, and a 2-D scene takes none
    unreadable = [{"target": [0, 1]}, {"distance": 0}, {"elevation": 90}, {"azimuth": "x"},
                  {"fov": 45}, "", "a/../b", 3, [1, 2, 3], {"target": [0, 0, float("nan")]}]
    for vc in unreadable:
        assert w.scene_view_info({"kind": 6, "scene": {"view_camera": vc}})["view_camera"] is None
    assert w.scene_view_info({"kind": 6, "scene": {"space": "2d", "view_camera": {}}}
                             )["view_camera"] is None
    views = w.decode((V32 / "views_scene_camera.bin").read_bytes())
    infos = [w.scene_view_info(e) for e in views.views]
    assert infos[0]["view_camera"] == {"target": [0.0, 0.0, 0.5], "azimuth": 30.0,
                                       "elevation": 25.0}
    assert infos[1] == {"space": "2d", "up": "-y", "units": "px", "view_camera": None}


def test_scene_view_info_null_keys_are_absent():
    """A null key means the same as an absent one: the golden VIEWS whose scene keys are null."""
    v = w.decode((V32 / "views_scene_nulls.bin").read_bytes())
    assert v.views[0]["scene"] == {"space": None, "up": None, "units": None}
    assert w.scene_view_info(v.views[0]) == {"space": "3d", "up": "+y", "units": "m",
                                             "view_camera": None}
    assert w.scene_view_info({"kind": 6, "scene": {"space": "2d", "units": None}}) == \
        {"space": "2d", "up": "-y", "units": "m", "view_camera": None}


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
                 "MSG_SCENE_TRANSFORMS", "MSG_SCENE_CLASSES", "DTYPE_U16", "PART_POINTS",
                 "PART_CAMERA", "FORM_DEPTH", "MAX_BATCH_POINTS", "MAX_TIMELINES",
                 "STAMP_ABSENT", "SCENE_DEFAULT_MAX_VERTICES"):
        assert f"#define CALIPER_NET_{name} " in block and f"#define CALIPER_NET_{name} " not in head
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
    assert rows == {"0070": "scene_update", "0071": "scene_transforms", "0072": "scene_clear",
                    "0073": "scene_classes"}
