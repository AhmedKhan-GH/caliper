"""The golden frames of the protocol 3.2 draft's scenes subset, for generate.py.

  v3_2/<name>.bin, .json       the frames of a 3.2 sender (minor 2): HELLO and
                               WELCOME with the capability "scenes", VIEWS with
                               a scene, VIEWS with a scene whose keys are
                               null (the defaults), the resolved default scene
                               data view,
                               SCENE_UPDATE of each part built (points, lines
                               as segments and strips, mesh) and SCENE_CLEAR.
                               Every implementation decodes them and encodes
                               them back to the same bytes, except DECODE_ONLY
  DECODE_ONLY                  frames with sections this subset ignores (a
                               stamp, normals): they decode, and the ignored
                               sections do not come back on encoding
  v3_2/invalid/<name>.bin      a 3.2 frame every decoder refuses
  v3_2/invalid/expect.json     per file: {"reason", "close_code"} for a frame
                               that is not well formed, or {"error"}, the
                               ERROR code that answers a well-formed frame
                               whose message is invalid

The ids, the kind and the capability are proposals of the 3.2 draft; see the
3.2 block of include/caliper/live_net.h.
"""
from __future__ import annotations

import struct

import numpy as np

from caliper import _wire as w

SESSION = {"id": "clp-4242-a1b2c3", "script": "train.py", "pid": 4242,
           "start_unix_ms": 1763200000000}
PRODUCER_SOFTWARE = {"name": "caliper-python", "version": "0.1.0"}
VIEWER_SOFTWARE = {"name": "caliper-web", "version": "0.1.0"}
STAMP = 81234567890123
M = w.MINOR_3_2

LANDSCAPE = {"id": 5, "name": "landscape", "kind": int(w.Kind.SCENE), "sources": [],
             "scene": {"space": "3d", "up": "+z", "units": ""}}
PLANE_SHARE = {"id": 6, "name": "plane share", "kind": int(w.Kind.SERIES), "sources": []}
# A scene whose optional keys are null, which means the same as absent: 3-D, +y up, metres.
NULLS = {"id": 7, "name": "defaults", "kind": int(w.Kind.SCENE), "sources": [],
         "scene": {"space": None, "up": None, "units": None}}

DECODE_ONLY = ("v3_2/scene_update_points_stamped", "v3_2/scene_update_mesh_normals")


def _f32(rows) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _stats(values, positions) -> list:
    """[values min, values max, x min, x max, y min, y max, z min, z max]."""
    p = np.asarray(positions, dtype=np.float32)
    out = [float(np.min(values)), float(np.max(values))] if values is not None else [0.0, 0.0]
    for k in range(3):
        out += [float(p[:, k].min()), float(p[:, k].max())] if k < p.shape[1] else [0.0, 0.0]
    return out


# The example scene: the draft's three points, the demo's marker, frame, path
# and surface.
CLOUD = _f32([[0, 0, 0], [1, 0, 0.5], [0, 2, 0.25]])
RGB = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
CURRENT = _f32([[0.0, 0.0, -0.25]])
MARKERS = _f32([[0.25, -0.5, 0.0], [-0.75, 0.5, -0.5]])
MARKER_VALUES = _f32([0.125, 0.75])
MARKER_RADII = _f32([0.05, 0.1])
FRAME = _f32([[-1, -1, -2], [1, -1, -2], [-1, -1, -2], [-1, 1, -2]])      # two edges of the box
PATH = _f32([[-0.5, -0.5, -0.1], [-0.25, -0.25, -0.15], [0, -0.1, -0.2], [0, 0, -0.25]])
TWO_STRIPS = _f32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]])
GRID = _f32([[-1, -1, -0.75], [1, -1, -0.25], [-1, 1, 0.125], [1, 1, 0.5]])
GRID_VALUES = _f32([-1.5, -0.5, 0.25, 1.0])
GRID_TRIANGLES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
FLAT = _f32([[0, 0], [1, 0], [0, 1]])                                       # a 2-D triangle


def _update(sub, view, seq, stamp, path, part, arrays, params, **kw):
    return w.SceneUpdate(sub, view, seq, stamp, path, part, arrays, params, **kw)


def valid_vectors() -> dict[str, bytes]:
    msgs = {
        "hello": w.Hello(w.ROLE_VIEWER, [w.CAP_CONTROLS, w.CAP_SCENES, w.CAP_SELECTION],
                         VIEWER_SOFTWARE),
        "welcome": w.Welcome(w.ROLE_PRODUCER, [w.CAP_CONTROLS, w.CAP_EVENTS, w.CAP_SCENES,
                                               w.CAP_SELECTION],
                             PRODUCER_SOFTWARE, SESSION, control=True),
        "views_scene": w.Views([LANDSCAPE, PLANE_SHARE], SESSION["id"]),
        "views_scene_nulls": w.Views([NULLS], SESSION["id"]),
        "view_reply_scene": w.ViewReply(9, 2, "landscape",
                                        data_view=w.resolve_data_view(LANDSCAPE, {})),
        # The draft's example frame without its stamp (timelines are not built).
        "scene_update_points": _update(2, "world", 12, STAMP + 1377, "world/cloud", w.PART_POINTS,
                                       {"positions": CLOUD, "colors": RGB}, {}),
        "scene_update_points_color": _update(
            2, "landscape", 41, STAMP + 3000, "current", w.PART_POINTS, {"positions": CURRENT},
            {"color": [255, 80, 80, 255], "stats": _stats(None, CURRENT)}),
        "scene_update_points_values": _update(
            2, "landscape", 7, STAMP + 3100, "markers", w.PART_POINTS,
            {"positions": MARKERS, "values": MARKER_VALUES, "radii": MARKER_RADII},
            {"colormap": w.COLORMAP_MAGMA, "flags": w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX,
             "vmin": 0.0, "vmax": 1.0, "stats": _stats(MARKER_VALUES, MARKERS)}, static=True),
        "scene_update_lines_segments": _update(
            2, "landscape", 1, STAMP + 3200, "", w.PART_LINES, {"positions": FRAME},
            {"mode": w.LINES_SEGMENTS, "color": [160, 160, 160, 255], "width": 1.0,
             "stats": _stats(None, FRAME)}, static=True),
        "scene_update_lines_strip": _update(
            2, "landscape", 40, STAMP + 3300, "path", w.PART_LINES,
            {"positions": PATH, "strip_lengths": np.array([4], dtype=np.int32)},
            {"mode": w.LINES_STRIPS, "color": [235, 235, 235, 255], "width": 2.0,
             "stats": _stats(None, PATH)}),
        "scene_update_lines_strips": _update(
            2, "world", 14, STAMP + 3400, "world/tracks", w.PART_LINES,
            {"positions": TWO_STRIPS, "strip_lengths": np.array([3, 2], dtype=np.int32),
             "colors": np.array([[255, 160, 0, 255], [0, 160, 255, 255]], dtype=np.uint8)},
            {"mode": w.LINES_STRIPS, "color_per": w.COLOR_PER_STRIP, "width": 3.0}),
        "scene_update_mesh": _update(
            2, "landscape", 39, STAMP + 3500, "surface", w.PART_MESH,
            {"positions": GRID, "indices": GRID_TRIANGLES, "values": GRID_VALUES},
            {"colormap": w.COLORMAP_VIRIDIS, "flags": 0, "vmin": -4.0, "vmax": 2.0,
             "stats": _stats(GRID_VALUES, GRID)}),
        "scene_update_mesh_colors": _update(
            3, "flat", 2, STAMP + 3600, "shapes/tri", w.PART_MESH,
            {"positions": FLAT, "colors": RGB}, {}, static=True, state=True),
        "scene_clear_reset": w.SceneClear(2, "landscape", 41, STAMP + 3700, "", reset=True,
                                          state=True),
        "scene_clear": w.SceneClear(2, "world", 15, STAMP + 3800, "world/cloud", recursive=True,
                                    parts=[w.PART_POINTS, w.PART_MESH]),
    }
    out = {f"v3_2/{name}": w.encode(m, minor=M) for name, m in msgs.items()}
    # Decode only: the draft's example frame itself, with its stamp section
    # (step 1,200 at 1,763,200,000,123,456,789 ns), and a mesh with normals.
    stamp = np.array([1200, 1763200000123456789], dtype=np.int64)
    out["v3_2/scene_update_points_stamped"] = w.encode_frame(
        w.MsgType.SCENE_UPDATE,
        {"part": "points", "path": "world/cloud", "seq": 12, "stamp_ns": STAMP + 1377, "sub": 2,
         "view": "world"},
        [w.Section.from_array("positions", CLOUD), w.Section.from_array("colors", RGB),
         w.Section.from_array("stamp", stamp)], minor=M)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))
    out["v3_2/scene_update_mesh_normals"] = w.encode_frame(
        w.MsgType.SCENE_UPDATE,
        {"part": "mesh", "path": "surface", "seq": 39, "stamp_ns": STAMP + 3500, "sub": 2,
         "view": "landscape"},
        [w.Section.from_array("positions", GRID), w.Section.from_array("indices", GRID_TRIANGLES),
         w.Section.from_array("normals", normals)], minor=M)
    return out


def _raw(type_: int, meta: bytes, sections=()) -> bytes:
    """A frame from literal metadata bytes, laid out by the rules (no checks)."""
    pos = w.HEADER_BYTES + 4 * len(sections) + len(meta)
    offs = []
    for s in sections:
        pos = (pos + 7) & ~7
        offs.append(pos)
        pos += len(s)
    out = bytearray(pos)
    w.HEADER.pack_into(out, 0, w.MAGIC, w.MAJOR, M, type_, 0, pos, len(meta), len(sections), 0)
    struct.pack_into(f"<{len(sections)}I", out, w.HEADER_BYTES, *[len(s) for s in sections])
    mo = w.HEADER_BYTES + 4 * len(sections)
    out[mo:mo + len(meta)] = meta
    for s, o in zip(sections, offs):
        out[o:o + len(s)] = s
    return bytes(out)


def _frame(part: str, path: str, sections: dict, **keys) -> bytes:
    """A SCENE_UPDATE frame of literal sections and keys (no message checks)."""
    meta = {"part": part, "path": path, "seq": 1, "stamp_ns": STAMP, "sub": 2, "view": "s", **keys}
    return w.encode_frame(w.MsgType.SCENE_UPDATE, meta,
                          [w.Section.from_array(n, a) for n, a in sections.items()], minor=M)


def invalid_vectors(valid: dict[str, bytes]) -> dict[str, tuple[bytes, dict]]:
    P = w.CLOSE_PROTOCOL_ERROR
    bad = {"error": w.ERR_BAD_REQUEST}
    unsupported = {"error": w.ERR_UNSUPPORTED}
    mesh = valid["v3_2/scene_update_mesh"]
    pos_meta = (b'{"part":"points","path":"p","sections":[{"dtype":0,"name":"positions",'
                b'"shape":[3,3]}],"seq":1,"stamp_ns":1,"sub":2,"view":"s"}')
    deep = "/".join(f"s{i}" for i in range(17))
    return {
        "truncated_mesh": (mesh[:-4], {"reason": "truncated", "close_code": P}),
        "positions_length_mismatch": (_raw(w.MsgType.SCENE_UPDATE, pos_meta, [bytes(32)]),
                                      {"reason": "malformed", "close_code": P}),
        "indices_i64": (_frame("mesh", "surface", {"positions": GRID,
                                                   "indices": GRID_TRIANGLES.astype(np.int64)}),
                        bad),
        "strip_lengths_f32": (_frame("lines", "path", {"positions": PATH,
                                                       "strip_lengths": _f32([4])},
                                     mode="strips"), bad),
        "strip_lengths_sum": (_frame("lines", "path", {"positions": TWO_STRIPS,
                                                       "strip_lengths": np.array([3, 3], np.int32)},
                                     mode="strips"), bad),
        "colors_rgb": (_frame("points", "p", {"positions": CLOUD, "colors": RGB[:, :3].copy()}),
                       bad),
        "colors_and_values": (_frame("points", "p", {"positions": CLOUD, "colors": RGB,
                                                     "values": _f32([1, 2, 3])}), bad),
        "positions_missing": (_frame("points", "p", {"colors": RGB}), bad),
        "positions_4d": (_frame("points", "p", {"positions": np.zeros((3, 4), np.float32)}), bad),
        "segments_odd": (_frame("lines", "", {"positions": CLOUD}, mode="segments"), bad),
        "mode_missing": (_frame("lines", "", {"positions": FRAME}), bad),
        "part_unknown": (_frame("blob", "p", {"positions": CLOUD}), bad),
        "part_arrows": (_frame("arrows", "p", {"origins": CLOUD, "vectors": CLOUD}), unsupported),
        "lines_edges": (_frame("lines", "g", {"positions": CLOUD,
                                              "indices": np.array([[0, 1], [1, 2]], np.int32)},
                               mode="edges"), unsupported),
        "split_piece": (_frame("points", "p", {"positions": CLOUD},
                               split={"update": 1, "piece": 0, "pieces": 2, "first": [0],
                                      "rows": [6]}), unsupported),
        "path_dotdot": (_frame("points", "world/..", {"positions": CLOUD}), bad),
        "path_too_deep": (_frame("points", deep, {"positions": CLOUD}), bad),
        "clear_reset_path": (w.encode_frame(w.MsgType.SCENE_CLEAR,
                                            {"path": "world", "reset": True, "seq": 1,
                                             "stamp_ns": STAMP, "sub": 2, "view": "s"}, minor=M),
                             bad),
    }
