"""The protocol 3.2 draft's scenes in caliper/_wire.py: the golden frames of
tests/fixtures/wire_v3/v3_2 decode as described and encode back to the same
bytes, every frame of v3_2/invalid is refused as its expect.json says, every
part and its rules, the stamps, entity transforms and class tables, the
limits, entity paths, the VIEWS key "scene" with its starting camera and the
default scene data view, and agreement with the 3.2 block of
include/caliper/live_net.h and the reference page
docs/reference/wire-protocol-3-2.md.

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
PAGE = ROOT / "docs" / "reference" / "wire-protocol-3-2.md"

sys.path.insert(0, str(ROOT / "python"))
from caliper import _wire as w  # noqa: E402

VALID = sorted(f"v3_2/{p.stem}" for p in V32.glob("*.bin"))
DECODE_ONLY = {"v3_2/scene_update_points_future"}
EXPECT = json.loads((V32 / "invalid" / "expect.json").read_text())


def test_the_frames_are_there():
    assert len(VALID) >= 60
    types = {w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).type for n in VALID}
    assert {0x70, 0x71, 0x72, 0x73, 0x74, 0x78, 0x79, 0x7A, 0x7B, 0x80, 0x81, 0x82} <= types
    msgs = [w.decode((FIXTURES / f"{n}.bin").read_bytes()) for n in VALID]
    parts = {m.part for m in msgs if isinstance(m, w.SceneUpdate)}
    assert parts == set(w.PARTS)
    forms = {m.params["form"] for m in msgs if isinstance(m, w.SceneUpdate) and m.part == "image"}
    assert forms == set(w.FORMS)
    modes = {m.params["mode"] for m in msgs if isinstance(m, w.SceneUpdate) and m.part == "lines"}
    assert modes == {"segments", "strips", "edges"}
    stamped = {type(m).__name__ for m in msgs if getattr(m, "stamp", None) is not None
               or getattr(m, "stamps", None) is not None}
    assert stamped == {"SceneUpdate", "SceneTransforms", "SceneClear", "SceneClasses", "Snapshot",
                       "Series", "Controls", "Text"}
    assert any(s.dtype == w.DType.U16 for n in VALID
               for s in w.decode_frame((FIXTURES / f"{n}.bin").read_bytes()).sections)
    assert len(EXPECT) >= 90
    assert {"error"} <= set().union(*EXPECT.values()) and {"reason"} <= set().union(*EXPECT.values())


@pytest.mark.parametrize("name", VALID)
def test_frame_matches_description(name):
    buf = (FIXTURES / f"{name}.bin").read_bytes()
    desc = json.loads((FIXTURES / f"{name}.json").read_text())
    h = desc["header"]
    assert struct.unpack_from("<IHHHHIIHH", buf, 0) == (
        h["magic"], h["major"], h["minor"], h["type"], h["flags"], h["frame_len"],
        h["meta_len"], h["section_count"], h["reserved"])
    # An instance of a piece carries its manifest and its WGSL, thus a larger frame.
    most = 2048 if "_piece" in name else 9000 if name.endswith("_texture_wide") else 600
    assert h["minor"] == w.MINOR_3_2 == 2 and h["flags"] == 0 and len(buf) <= most
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
        assert f.type in (0x01, 0x02, 0x10, 0x20, 0x24, 0x30, 0x31, 0x50, 0x51, 0x52, 0x70, 0x71,
                          0x72, 0x73, 0x74, 0x78, 0x79, 0x7A, 0x7B, 0x80, 0x81, 0x82)


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
    part = page[page.index("## Golden Frames"):]
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
                    "0073": "scene_classes", "0074": "text", "0078": "recordings",
                    "0079": "recording_open", "007a": "cursor", "007b": "time_state",
                    "0080": "program", "0081": "program_state", "0082": "resource"}



# ---------------------------------------------------------------------------
# timelines, text logs, bar charts, split updates and recordings
# ---------------------------------------------------------------------------
def test_golden_values_of_timelines_text_bars_and_recordings():
    v = _d("views_timelines")
    assert [t["name"] for t in v.timelines] == ["step", "time", "episode"]
    assert v.views[0]["recorded"] == [{"budget": {"max_backfill": 4096}}]
    assert w.decode((FIXTURES / "views.bin").read_bytes()).timelines is None
    se = _d("sessions_recording")
    assert se.sessions[0]["recording"] == {"id": "train-20261115-120314", "seekable": True,
                                           "complete": True}
    sr = _d("series_stamps")
    assert sr.stamps.array().tolist() == [[1200, 1763200000123456789],
                                          [1201, 1763200000128456789],
                                          [1202, 1763200000133456789]]
    assert _d("series_stamp").stamp.array().tolist() == [1200, 1763200000123456789]
    dec = _d("series_decimated")
    assert dec.indices.array().tolist() == [0, 2048, 4095] and dec.decimated == 1024
    assert dec.cursor == 2
    sn = _d("snapshot_stamped")
    assert sn.stamp is not None and not sn.static
    assert _d("snapshot_static").static and _d("snapshot_static").stamp is None
    em = _d("snapshot_empty")
    assert em.empty and em.data is None and em.cursor == 3
    b = _d("snapshot_bars")
    assert b.kind == w.Kind.BARS and b.labels == ["cat", "dog", "fish"] and b.horizontal
    assert b.colors.array().shape == (3, 4) and b.data.array().tolist() == [0.125, 0.75, -0.25]
    assert _d("controls_stamped").stamp.array().tolist() == [1200, 1763200000123456789]
    t = _d("text")
    assert t.lines() == ["epoch 1 starts", "loss rose\nto 2.5", "pas de données é"]
    assert t.levels.array().tolist() == [1, 2, 3] and t.stamps.array().shape == (3, 2)
    ts = _d("text_stamp")
    assert (ts.start, ts.total, ts.dropped, ts.cursor) == (10, 12, 10, 1)
    assert ts.lines() == ["a", ""]
    r = _d("recordings")
    assert r.recordings[0]["id"] == "train-20261115-120314"
    assert _d("recordings_none").recordings == []
    assert _d("recording_open").recording == "train-20261115-120314"
    c = _d("cursor_play")
    assert (c.op, c.timeline, c.speed, c.pace, c.loop) == ("play", "step", 2.0, "even",
                                                          ["1000", "5000"])
    assert _d("cursor_step").count == -5
    assert _d("cursor_follow").timeline is None
    p = _d("time_state_paused")
    assert (p.request_id, p.cursor, p.generation) == (2, "5000", 2)
    f = _d("time_state_following")
    assert f.cursor is None and f.live
    assert b'"cursor":null' in (V32 / "time_state_following.bin").read_bytes()


def test_31_encodings_are_unchanged():
    """A 3.0 or 3.1 message without the 3.2 keys encodes to the bytes it always did."""
    for p in sorted(FIXTURES.glob("*.bin")) + sorted((FIXTURES / "v3_1").glob("*.bin")):
        if p.stem in ("ping_future", "unknown_type"):       # frames of a later minor
            continue
        buf = p.read_bytes()
        m = w.decode(buf)
        if isinstance(m, w.Unknown):
            continue
        assert w.encode(m, minor=w.decode_frame(buf).minor) == buf, p.name


def test_stamp_text():
    for ok in ("0", "-1", "1763200000123456789", "9223372036854775807",
               "-9223372036854775808", "007"):
        assert w.stamp_text_ok(ok), ok
    for bad in ("", "-", "+1", "1.5", "1e3", " 1", "9223372036854775808",
                "-9223372036854775809", "1" * 21, 12, None, "١٢"):
        assert not w.stamp_text_ok(bad), bad


def test_cursor_keeps_only_its_ops_fields():
    meta = {"request_id": 1, "op": "pause", "timeline": "step", "value": "5", "count": 3,
            "speed": 2, "pace": "even", "loop": None}
    c = w.decode(w.encode_frame(w.MsgType.CURSOR, meta))
    assert (c.timeline, c.value, c.count, c.speed) == (None, None, None, None)
    assert w.encode(c) == w.encode_frame(w.MsgType.CURSOR, {"request_id": 1, "op": "pause"})
    c = w.decode(w.encode_frame(w.MsgType.CURSOR, dict(meta, op="seek")))
    assert (c.timeline, c.value, c.count) == ("step", "5", None)
    for op, extra, code in (("step", {"count": 1001}, w.ERR_BAD_CURSOR),
                            ("step", {"count": 1.5}, w.ERR_BAD_CURSOR),
                            ("play", {"speed": 0.05}, w.ERR_BAD_CURSOR),
                            ("play", {"pace": "fast"}, w.ERR_BAD_CURSOR),
                            ("play", {"timeline": "bad name"}, w.ERR_BAD_CURSOR),
                            ("rewind", {}, w.ERR_BAD_REQUEST)):
        m = {"request_id": 1, "op": op, "timeline": "step", **extra}
        _refused(w.encode_frame(w.MsgType.CURSOR, m), code)
    ok = w.decode(w.encode_frame(w.MsgType.CURSOR, {"request_id": 1, "op": "play",
                                                    "timeline": "t", "speed": 1 / 16}))
    assert ok.speed == 0.0625 and ok.loop is None


def test_text_rules():
    t = w.Text.from_lines(1, "log", 0, 2, 5, ["x" * w.TEXT_LINE_MAX, "é" * 3], [0, 3])
    back = w.decode(w.encode(t))
    assert back.lines() == ["x" * 4096, "ééé"]
    many = w.Text.from_lines(1, "log", 0, w.TEXT_MAX_LINES, 5, [""] * w.TEXT_MAX_LINES,
                             [1] * w.TEXT_MAX_LINES)
    assert len(w.decode(w.encode(many)).lines()) == w.TEXT_MAX_LINES
    over = w.Text.from_lines(1, "log", 0, w.TEXT_MAX_LINES + 1, 5,
                             [""] * (w.TEXT_MAX_LINES + 1), [1] * (w.TEXT_MAX_LINES + 1))
    _refused(w.encode(over))
    _refused(w.encode(w.Text.from_lines(1, "log", 0, 1, 5, ["a"], [1],
                                        stamps=np.zeros((2, 2), np.int64))))


def test_bars_and_text_data_views():
    log = {"id": 1, "name": "log", "kind": int(w.Kind.TEXT), "sources": []}
    assert w.resolve_data_view(log, None) == {"budget": {"max_backfill": w.SERIES_RETAIN_MIN}}
    assert w.resolve_data_view(log, {"budget": {"max_backfill": 10}}) == {
        "budget": {"max_backfill": 10}}
    bars = {"id": 2, "name": "b", "kind": int(w.Kind.BARS), "sources": [
        {"name": "s", "dtype": "f32", "dims": [{"name": "layer", "size": 4},
                                               {"name": "class", "size": 5000}]}]}
    with pytest.raises(w.MessageError):
        w.resolve_data_view(bars, None)                      # two kept dims
    r = w.resolve_data_view(bars, {"index": {"layer": 0, "class": [0, 4096]}})
    assert w.payload_layout(bars, r)["extent"] == [4096]
    with pytest.raises(w.MessageError) as e:
        w.resolve_data_view(bars, {"index": {"layer": 0}})   # 5000 bars
    assert e.value.code == w.ERR_BAD_DATA_VIEW


def _pieces():
    return [w.decode((V32 / f"scene_update_split_{k}.bin").read_bytes()) for k in range(5)]


def test_split_update_assembles_to_the_whole():
    from importlib import util
    spec = util.spec_from_file_location("gen_v3_2", FIXTURES / "gen_v3_2.py")
    gen = util.module_from_spec(spec)
    sys.path.insert(0, str(FIXTURES))
    spec.loader.exec_module(gen)
    whole = gen.SPLIT
    pieces = _pieces()
    assert [p.split["piece"] for p in pieces] == [0, 1, 2, 3, 4]
    assert pieces[0].params == {"radius": 0.25} and all(p.params == {} for p in pieces[1:])
    assert all(p.stamp is not None for p in pieces)
    su = w.SplitUpdates()
    got = [su.add(p) for p in pieces]
    assert got[:4] == [None] * 4 and not su.open
    u = got[4]
    assert isinstance(u, w.SceneUpdate) and u.split is None
    assert u.arrays["positions"].array().tolist() == whole.arrays["positions"].tolist()
    assert u.arrays["colors"].array().tolist() == whole.arrays["colors"].tolist()
    assert (u.path, u.seq, u.params, u.stamp.array().tolist()) == (
        "world/split", 37, {"radius": 0.25}, [1200, 1763200000123456789])
    assert w.encode(u) == w.encode(whole)


def test_split_update_rules():
    pieces = _pieces()
    # a piece without its piece 0 is dropped and counted
    su = w.SplitUpdates()
    assert su.add(pieces[1]) is None and su.dropped == 1 and not su.open
    # a piece twice
    su = w.SplitUpdates()
    su.add(pieces[0])
    su.add(pieces[1])
    with pytest.raises(w.MessageError):
        su.add(pieces[1])
    assert not su.open
    # a newer update of the same entity part discards the incomplete one
    su = w.SplitUpdates()
    su.add(pieces[0])
    newer = w.SceneUpdate(**{**pieces[0].__dict__, "split": dict(pieces[0].split, update=5)})
    su.add(newer)
    assert list(su.open) == [5] and su.discarded == 1
    assert su.add(pieces[1]) is None and su.dropped == 1
    # at most two incomplete updates; the oldest goes
    su = w.SplitUpdates()
    for k, path in enumerate(["a", "b", "c"]):
        su.add(w.SceneUpdate(**{**pieces[0].__dict__, "path": path,
                                "split": dict(pieces[0].split, update=10 + k)}))
    assert list(su.open) == [11, 12]
    # a piece that differs from its piece 0, or whose rows overlap
    for bad in (dict(pieces[1].__dict__, seq=38),
                dict(pieces[1].__dict__, split=dict(pieces[1].split, first={"positions": 4}))):
        su = w.SplitUpdates()
        su.add(pieces[0])
        with pytest.raises(w.MessageError):
            su.add(w.SceneUpdate(**bad))
    # the last piece arrives with rows missing
    su = w.SplitUpdates()
    su.add(w.SceneUpdate(**dict(pieces[0].__dict__, split=dict(pieces[0].split, pieces=2))))
    with pytest.raises(w.MessageError):
        su.add(w.SceneUpdate(**dict(pieces[4].__dict__, split=dict(pieces[4].split, piece=1,
                                                                     pieces=2))))
    # the whole update is checked as an unsplit one: colours of the wrong row count
    su = w.SplitUpdates()
    p0 = w.SceneUpdate(**dict(pieces[0].__dict__, split=dict(
        pieces[0].split, pieces=2, rows={"positions": 20, "colors": 2})))
    rest = w.SceneUpdate(2, "world", 37, pieces[0].stamp_ns, "world/split", "points",
                         {"positions": np.zeros((14, 3), np.float32),
                          "colors": np.zeros((2, 4), np.uint8)}, {}, stamp=pieces[0].stamp,
                         split={"update": 4, "piece": 1, "pieces": 2,
                                "first": {"positions": 6, "colors": 0}})
    su.add(w.decode(w.encode(p0)))
    with pytest.raises(w.MessageError):
        su.add(w.decode(w.encode(rest)))


def test_split_update_of_a_large_mesh():
    """A mesh over one frame travels in pieces and comes back whole (small
    frames stand in for 64 MiB, so that the test allocates a few MB)."""
    n = 30_000
    pos = np.random.default_rng(1).random((n, 3), dtype=np.float32)
    tri = (np.arange(n * 3, dtype=np.int32) % n).reshape(n, 3)
    u = w.SceneUpdate(1, "s", 3, 4, "m", "mesh", {"positions": pos, "indices": tri},
                      {"color": [1, 2, 3, 255]})
    pieces = w.split_update(u, 9, max_frame=64 * 1024)
    assert 2 <= len(pieces) <= w.MAX_PIECES
    assert all(len(w.encode(p)) <= 64 * 1024 for p in pieces)
    su = w.SplitUpdates()
    out = [su.add(w.decode(w.encode(p))) for p in pieces]
    assert all(x is None for x in out[:-1])
    assert out[-1].arrays["positions"].array().tolist() == pos.tolist()
    assert out[-1].arrays["indices"].array().tolist() == tri.tolist()
    with pytest.raises(ValueError):
        w.split_update(u, 9, max_frame=16 * 1024)       # more than 16 pieces
    assert w.split_update(u, 9) == [u]                  # fits one frame


# ---------------------------------------------------------------------------
# GPU programs and interventions
# ---------------------------------------------------------------------------
def test_golden_values_of_programs():
    import hashlib
    g = _d("program_set_geometry")
    assert (g.op, g.id, g.scene, g.request_id) == ("set", "rows", "world", 11)
    assert hashlib.sha256(g.wgsl.encode("utf-8")).hexdigest() == g.sha256
    assert "\u00b5" in g.wgsl and g.declaration["output"]["max_count"] == 4096
    f = w.decode_frame((V32 / "program_set_geometry.bin").read_bytes())
    assert [s.name for s in f.sections] == ["wgsl"] and f.sections[0].dtype == w.DType.U8
    p = _d("program_set_pixels")
    assert p.scene is None and p.declaration["output"]["width"] == 64
    r = _d("program_remove")
    assert (r.op, r.declaration, r.wgsl, r.sha256) == ("remove", None, None, None)
    assert w.decode_frame((V32 / "program_remove.bin").read_bytes()).sections == []
    st = _d("program_state_reply")
    assert (st.request_id, st.step, st.step_ms) == (12, 1200, 31.25)
    assert [x["state"] for x in st.programs] == ["running", "refused"]
    assert st.programs[1]["error"]["code"] == "program.unknown_input"
    wt = _d("program_state_waiting")
    assert wt.request_id is None and wt.step is None and len(wt.programs) == 1


def test_golden_values_of_interventions():
    s = _d("control_intervene_set")
    assert s.intervene == {"set": {"point": "blocks.0.attn#3", "op": "ablate",
                                   "from_step": 1201, "until_step": None}}
    assert _d("control_intervene_clear").intervene == {"clear": "blocks.0.attn#3"}
    assert _d("control_intervene_clear_all").intervene == {"clear_all": True}
    assert _d("control_reply_intervene").op == "intervene"
    c = _d("controls_interventions")
    assert [p["kind"] for p in c.interventions["points"]] == ["module", "head"]
    assert c.interventions["active"][0]["state"] == "active"
    assert w.decode((FIXTURES / "v3_1" / "controls.bin").read_bytes()).interventions is None


def test_program_codec_rules():
    src = "fn f() {}"
    sha = "0" * 64
    decl = {"output": {"kind": "pixels", "width": 4, "height": 4}}
    ok = w.Program(1, "set", "img", decl, src, sha)
    assert w.decode(w.encode(ok)) == ok
    # An id of exactly 55 bytes passes, and program.<id> then fits a view name.
    long_id = "p" * w.PROGRAM_ID_MAX
    assert w.decode(w.encode(w.Program(1, "remove", long_id))).id == long_id
    assert len(f"program.{long_id}".encode()) <= w.NAME_MAX
    # A remove ignores a declaration and a section it does not need.
    rm = w.encode_frame(w.MsgType.PROGRAM, {"request_id": 2, "op": "remove", "id": "img",
                                            "declaration": 5})
    assert w.decode(rm).declaration is None
    # A geometry output needs the scene; a pixels output does not.
    geo = {"output": {"kind": "geometry", "part": "points", "max_count": 1,
                      "arrays": ["positions"]}}
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.Program(1, "set", "g", geo, src, sha)))
    assert w.decode(w.encode(w.Program(1, "set", "g", geo, src, sha, "s"))).scene == "s"
    # A wgsl section of exactly the limit passes.
    big = "a" * w.PROGRAM_SOURCE_MAX
    assert len(w.decode(w.encode(w.Program(1, "set", "img", decl, big, sha))).wgsl) == len(big)


def test_golden_values_of_pieces_and_resources():
    p = _d("program_set_piece")
    assert (p.op, p.id, p.request_id, p.panel) == ("set", "heads", 31, "bars")
    assert p.piece == {"name": "head_bars", "version": 2, "hash": p.piece["hash"]}
    assert p.piece["hash"].startswith("sha256:") and p.manifest["resources"][0]["id"] == "heads"
    assert p.bind == {"attn": "live:blocks.0.attn"} and p.params == {"gain": 2.0}
    assert [x["part"] for x in p.segments] == ["lib/colormap@1", "common.wgsl"]
    assert p.declaration is None and p.wgsl is None and p.sha256 is None
    assert "mean_heads" in p.derive and "\u00b5" in p.draw and "colormap__viridis" in p.common
    f = w.decode_frame((V32 / "program_set_piece.bin").read_bytes())
    assert [s.name for s in f.sections] == ["common", "derive", "draw"]
    relay = _d("program_set_piece_draw")
    assert (relay.request_id, relay.derive) == (0, None) and relay.draw == p.draw
    assert _d("program_remove_piece").request_id == 0
    assert _d("program_state_piece").programs[0]["kind"] == "piece"
    b = _d("resource_buffer")
    assert (b.program, b.resource, b.step, b.count, b.stride, b.width) == ("heads", "heads", 1200, 4, 4, None)
    assert np.frombuffer(b.data, np.float32).tolist() == [0.5, 0.25, 0.125, 0.0625]
    st = _d("resource_struct")
    assert (st.count, st.stride, len(st.data)) == (2, 24, 48)
    assert np.frombuffer(st.data, np.uint32)[:4].tolist() == [3, 1, 1, 2]
    t = _d("resource_texture")
    assert (t.width, t.height, t.count, t.stride, t.step) == (2, 2, 4, 4, 1210)
    assert np.frombuffer(t.data, np.uint32)[0] == 0xff0000ff
    e = _d("resource_empty")
    assert (e.count, e.stride, len(e.data)) == (0, 24, 0)


def test_resource_codec_rules():
    data = np.arange(12, dtype=np.float32)
    r = w.Resource("arcs", "arcs", 7, 2, 24, data)
    buf = w.encode(r)
    back = w.decode(buf)
    assert (back.program, back.resource, back.step, back.count, back.stride) == ("arcs", "arcs", 7, 2, 24)
    assert isinstance(back.data, memoryview) and bytes(back.data) == data.tobytes()   # a view, no copy
    # The section's length must equal count * stride.
    with pytest.raises(w.MessageError, match="count \\* stride"):
        w.decode(w.encode(w.Resource("arcs", "arcs", 7, 3, 24, data)))
    # A count past the byte limit is refused before any product is taken.
    frame = w.encode_frame(w.MsgType.RESOURCE, {"program": "a", "resource": "r", "step": 0,
                                                "count": 2**32 - 1, "stride": 256},
                           [w.Section.from_array("resource", np.zeros(256, np.uint8))])
    with pytest.raises(w.MessageError, match="at most 67043328 bytes"):
        w.decode(frame)
    # The limit leaves room for the header and the metadata in one frame.
    assert w.RESOURCE_MAX_BYTES + 65536 == w.MAX_FRAME_BYTES


def _golden(name):
    return w.decode((V32 / f"{name}.bin").read_bytes())


def test_resource_formats_and_narrow_elements_decode():
    n = _golden("resource_narrow_u16")
    assert (n.element, n.elements, n.count, n.stride, n.width, n.format) == ("u16", 5, 3, 4, None, None)
    assert np.frombuffer(n.data, np.uint16).tolist() == [1, 2, 3, 4, 5, 0]
    t = _golden("resource_texture_r8")
    assert (t.format, t.width, t.height, t.count, t.stride, t.element) == ("r8unorm", 3, 3, 3, 4, None)
    assert bytes(t.data)[:9] == bytes(range(1, 10))


def test_resource_part_with_another_format_is_refused():
    import dataclasses
    p0, p1, _ = (_golden(f"resource_split_{k}") for k in range(3))
    sr = w.SplitResources()
    assert sr.add(p0) is None
    with pytest.raises(w.MessageError, match="differs from its part 0"):
        sr.add(dataclasses.replace(p1, format="r8unorm"))


WHOLE_SEGMENTS = np.array([[k, -k] for k in range(10)], np.float32).tobytes()


def test_resource_in_parts_is_put_together_whole():
    parts = [_golden(f"resource_split_{k}") for k in range(3)]
    assert [p.split for p in parts] == [{"version": 5, "part": k, "parts": 3, "first": 4 * k} for k in range(3)]
    sr = w.SplitResources()
    assert sr.add(parts[0]) is None and sr.add(parts[1]) is None
    whole = sr.add(parts[2])
    assert (whole.program, whole.resource, whole.step, whole.count, whole.stride, whole.split) == \
        ("lines", "segments", 1300, 10, 8, None)
    assert bytes(whole.data) == WHOLE_SEGMENTS and not sr.open
    # The whole encodes as a version in one frame, which a fresh receiver takes at once.
    again = w.decode(w.encode(whole))
    assert w.SplitResources().add(again).count == 10
    # resource_parts cuts the same parts, as views without a copy.
    cut = w.resource_parts(w.Resource("lines", "segments", 1300, 10, 8, WHOLE_SEGMENTS), 5, 32)
    assert [w.encode(p, minor=2) for p in cut] == [(V32 / f"resource_split_{k}.bin").read_bytes() for k in range(3)]


def test_resource_parts_that_a_receiver_drops_or_refuses():
    p0, p1, p2 = (_golden(f"resource_split_{k}") for k in range(3))
    # A part of a version whose part 0 never came is dropped and counted.
    sr = w.SplitResources()
    assert sr.add(_golden("resource_split_orphan")) is None and sr.dropped == 1 and not sr.open
    sr.add(p0)
    assert sr.add(_golden("resource_split_orphan")) is None and sr.dropped == 2 and len(sr.open) == 1
    # Parts out of order: the version is refused and discarded.
    with pytest.raises(w.MessageError, match="out of order"):
        sr.add(p2)
    assert not sr.open and sr.discarded == 1
    # A part again is out of order too.
    sr.add(p0)
    sr.add(p1)
    with pytest.raises(w.MessageError, match="out of order"):
        sr.add(p1)
    # A total that disagrees with the parts: the last part ends before the count.
    sr.add(p0)
    sr.add(p1)
    with pytest.raises(w.MessageError, match="end before its count"):
        sr.add(_golden("resource_split_short"))
    # A part whose keys differ from part 0.
    sr.add(p0)
    with pytest.raises(w.MessageError, match="differs from its part 0"):
        sr.add(_golden("resource_split_differs"))
    assert not sr.open
    # A version that never completes: the next part after RESOURCE_PART_WAIT further frames is dropped.
    sr = w.SplitResources()
    sr.add(p0)
    other = _golden("resource_buffer")
    for _ in range(w.RESOURCE_PART_WAIT - 1):
        assert sr.add(other) is not None
    assert sr.add(p1) is None and len(sr.open) == 1          # the 256th frame after part 0: in time
    for _ in range(w.RESOURCE_PART_WAIT):
        sr.add(other)
    assert len(sr.open) == 1
    sr.add(other)                                              # the 257th: expired
    assert not sr.open and sr.expired == 1
    assert sr.add(p2) is None and sr.dropped == 1
    # Part 0 of a newer version, or a version in one frame, of the same resource discards an incomplete one.
    sr = w.SplitResources()
    sr.add(p0)
    sr.add(p0)
    assert len(sr.open) == 1 and sr.discarded == 1
    sr.add(w.Resource("lines", "segments", 1400, 1, 8, bytes(8)))
    assert not sr.open and sr.discarded == 2
    # At most MAX_INCOMPLETE_UPDATES versions are held; the oldest goes first.
    for name in ("a", "b", "c"):
        sr.add(w.Resource("lines", name, 1300, 10, 8, p0.data,
                          split={"version": 5, "part": 0, "parts": 3, "first": 0}))
    assert [k[1] for k in sr.open] == ["b", "c"] and w.MAX_INCOMPLETE_UPDATES == 2


def test_resource_in_parts_bounds_every_size_first():
    # The bound of a version in parts holds before any product: a count near 2^32 is refused.
    frame = w.encode_frame(w.MsgType.RESOURCE, {"program": "a", "resource": "r", "step": 0, "count": 2**32 - 1,
                                                "stride": 256, "split": {"version": 0, "part": 0, "parts": 2,
                                                                         "first": 0}},
                           [w.Section.from_array("resource", np.zeros(256, np.uint8))])
    with pytest.raises(w.MessageError, match="in parts holds at most 134217728 bytes"):
        w.decode(frame)
    assert w.RESOURCE_VERSION_MAX_BYTES == 128 << 20 and w.RESOURCE_PART_WAIT == 256
    # The sender's cut: one frame when it fits, parts of whole elements, and refusals past the bounds.
    big = w.Resource("p", "r", 0, 9, 8, bytes(72))
    assert w.resource_parts(big, 0, 72) == [big]
    assert [len(p.data) for p in w.resource_parts(big, 0, 20)] == [16, 16, 16, 16, 8]
    with pytest.raises(ValueError, match="more than 16"):
        w.resource_parts(w.Resource("p", "r", 0, 17, 8, bytes(136)), 0, 8)
    with pytest.raises(ValueError, match="holds no element"):
        w.resource_parts(big, 0, 4)
    with pytest.raises(ValueError, match="more than 134217728"):
        w.resource_parts(w.Resource("p", "r", 0, (128 << 20) // 4 + 1, 4, bytearray((128 << 20) + 4)), 0, 8 << 20)


def test_instance_codec_rules():
    src = "@fragment fn f() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
    pin = {"name": "dots", "version": 1, "hash": "sha256:" + "0" * 64}
    inst = w.Program(4, "set", "dots", piece=pin, manifest={"name": "dots"}, draw=src)
    assert w.decode(w.encode(inst)) == inst
    # A draw half of exactly the limit passes; one byte more is refused.
    big = "a" * w.PROGRAM_SOURCE_MAX
    assert len(w.decode(w.encode(w.Program(4, "set", "dots", piece=pin, manifest={},
                                             derive=big))).derive) == len(big)
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.Program(4, "set", "dots", piece=pin, manifest={}, derive=big + "a")))
    common = "b" * w.PIECE_COMMON_MAX
    ok = w.Program(4, "set", "dots", piece=pin, manifest={}, common=common, draw=src)
    assert len(w.decode(w.encode(ok)).common) == w.PIECE_COMMON_MAX
    with pytest.raises(w.MessageError):
        w.decode(w.encode(w.Program(4, "set", "dots", piece=pin, manifest={}, common=common + "b",
                                    draw=src)))
    # A params op of an instance keeps the params form.
    assert w.decode(w.encode(w.Program(5, "params", "dots", params={"gain": 1}))).piece is None


# The piece form of SCENE_UPDATE (capability "scene_pieces") and the ids "scene.<n>".
PIECE_PIN = {"name": "points", "version": 1, "hash": "sha256:" + "ab" * 32}
PIECE_KEYS = {"program": "scene.1", "piece": PIECE_PIN, "resources": {"points": 3},
              "arrays": ["positions", "values"]}


def _piece_update(part="points", sections=None, **change):
    keys = {k: v for k, v in {**PIECE_KEYS, **change}.items() if v is not None}
    return _update(part, sections or {}, **keys)


def test_scene_pieces_capability_and_prefix():
    assert w.CAP_SCENE_PIECES == "scene_pieces" and w.SCENE_PROGRAM_PREFIX == "scene."


def test_piece_form_golden_frames():
    m = _d("scene_update_points_piece")
    assert m.program == "scene.1" and m.arrays == {} and m.split is None
    assert m.piece == {"name": "points", "version": 1} and m.resources == {"points": 3}
    assert m.names == ["positions", "values"] and m.stamp is not None
    r = _d("resource_scene_points")
    assert (r.program, r.resource, r.step, r.count, r.stride, len(r.data)) == ("scene.1", "points", 7, 3, 32, 96)
    rows = np.frombuffer(r.data, dtype=[("p", "<f4", 4), ("bits", "<u4"), ("index", "<u4"), ("pad", "V8")])
    assert rows["p"][:, 3].tolist() == [1.0, 1.0, 1.0] and rows["index"].tolist() == [0, 1, 2]


def test_piece_form_round_trips():
    for part, piece, names in [("points", "points", ["positions"]), ("lines", "lines", ["positions", "colors"]),
                               ("arrows", "arrows", ["origins", "vectors"]), ("boxes", "boxes", ["centers"]),
                               ("mesh", "mesh", ["positions", "indices", "uvs"])]:
        kw = {"mode": "segments"} if part == "lines" else {}
        buf = _piece_update(part, piece=dict(PIECE_PIN, name=piece), arrays=names, **kw)
        m = w.decode(buf)
        assert m.names == names and m.arrays == {} and m.piece["name"] == piece
        assert w.encode(m) == buf
    for form, piece in [("color", "image"), ("depth", "depth"), ("segmentation", "segmentation")]:
        buf = _piece_update("image", piece=dict(PIECE_PIN, name=piece), arrays=["pixels"], form=form)
        assert w.encode(w.decode(buf)) == buf


def test_piece_form_refusals():
    _refused(_piece_update(program="scene.0123456789"), match="program")
    _refused(_piece_update(piece=None), match="piece")
    _refused(_piece_update(piece=dict(PIECE_PIN, name="lines")), match="library piece")
    _refused(_piece_update(piece=dict(PIECE_PIN, hash="sha256:xyz")), match="pin")
    _refused(_piece_update(piece=dict(PIECE_PIN, hash="sha256:" + "AB" * 32)), match="pin")
    assert w.decode(_piece_update(piece={"name": "points", "version": 1})).piece == {"name": "points", "version": 1}
    _refused(_piece_update("camera", piece=dict(PIECE_PIN, name="camera"), arrays=["x"]))
    _refused(_piece_update(resources=None), match="resources")
    _refused(_piece_update(resources={}), match="resources")
    _refused(_piece_update(resources={f"r{k}": 1 for k in range(9)}), match="resources")
    _refused(_piece_update(resources={"__x": 1}), match="resources")
    _refused(_piece_update(resources={"points": -1}), match="resources")
    _refused(_piece_update(resources={"points": 2**32}), match="resources")
    _refused(_piece_update(arrays=None), match="arrays")
    _refused(_piece_update(arrays=[]), match="arrays")
    _refused(_piece_update(arrays=["values"]), match="arrays")
    _refused(_piece_update(arrays=["positions", "positions"]), match="arrays")
    _refused(_piece_update(arrays=["values", "positions"]), match="arrays")
    _refused(_piece_update(arrays=["positions", "nope"]), match="arrays")
    _refused(_piece_update(arrays=["positions", 3]), match="arrays")
    _refused(_piece_update(split={"update": 1, "piece": 0, "pieces": 2}), match="split")
    _refused(_piece_update(sections={"positions": np.zeros((3, 3), np.float32)}), match="array section")
    _refused(_piece_update(sections={"colors": np.zeros((3, 4), np.uint8)}), match="array section")
    # An image takes the piece of its form.
    _refused(_piece_update("image", piece=dict(PIECE_PIN, name="depth"), arrays=["pixels"], form="color"))
    _refused(_piece_update("image", piece=dict(PIECE_PIN, name="image"), arrays=["pixels"], form="encoded"))
    # The part's keys are still checked.
    _refused(_piece_update(colormap="x"))
    # The stamp is read as usual.
    ok = _piece_update(sections={"stamp": np.array([1, 2], np.int64)})
    assert w.decode(ok).stamp is not None


def test_scene_program_ids():
    def res(program):
        return w.encode_frame(w.MsgType.RESOURCE, {"program": program, "resource": "points", "step": 1,
                                                   "count": 1, "stride": 4},
                              [w.Section.from_array("resource", np.zeros(4, np.uint8))])
    for ok in ("scene.0", "scene.1", "scene.123456789"):
        assert w.decode(res(ok)).program == ok and w.scene_program_id(ok)
    for bad in ("scene.", "scene.x", "scene.0123456789", "scene.1a", "scene.-1", "scene.1.2"):
        _refused(res(bad), match="program")
        assert not w.scene_program_id(bad)
    prog = w.Program(0, w.PROGRAM_REMOVE, "scene.7")
    assert w.decode(w.encode(prog)).id == "scene.7"
    _refused(w.encode_frame(w.MsgType.PROGRAM, {"request_id": 0, "op": "remove", "id": "scene.x"}))
