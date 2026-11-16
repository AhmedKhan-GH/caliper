"""The golden frames of the protocol 3.2 draft's scenes subset, for generate.py.

  v3_2/<name>.bin, .json       the frames of a 3.2 sender (minor 2): HELLO and
                               WELCOME with the capability "scenes", VIEWS with
                               scenes, VIEWS with a scene whose keys are null
                               (the defaults), the resolved default scene
                               data view,
                               SCENE_UPDATE of every part and form,
                               SCENE_TRANSFORMS, SCENE_CLEAR and SCENE_CLASSES,
                               with and without their stamps. Every
                               implementation decodes them and encodes them
                               back to the same bytes, except DECODE_ONLY
  DECODE_ONLY                  frames with a section no part has: it decodes,
                               and the section does not come back on encoding
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

DECODE_ONLY = ("v3_2/scene_update_points_future",)


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


def _update(sub, view, seq, stamp_ns, path, part, arrays, params, **kw):
    return w.SceneUpdate(sub, view, seq, stamp_ns, path, part, arrays, params, **kw)


# The draft's stamp: step 1,200 at 1,763,200,000,123,456,789 ns.
STAMP_T = np.array([1200, 1763200000123456789], dtype=np.int64)
WORLD = {"id": 7, "name": "world", "kind": int(w.Kind.SCENE), "sources": [],
         "scene": {"space": "3d", "up": "+z", "units": "m",
                   "view_camera": {"target": [0.0, 0.0, 0.5], "azimuth": 30.0, "elevation": 25.0}}}
OVERLAY = {"id": 8, "name": "overlay", "kind": int(w.Kind.SCENE), "sources": [],
           "scene": {"space": "2d", "up": "-y", "units": "px"}}
# A 1 x 1 PNG of one orange pixel, the bytes of an encoded image.
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489000000"
                    "0d4944415478da63f8dfc0f01f000680027fcf1209f10000000049454e44ae426082")


def _matrix(tx, ty, tz) -> list:
    """A translation as a column-major 4 x 4 matrix."""
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (tx, ty, tz)
    return m.T.reshape(-1).tolist()


# Timelines, text logs, bar charts and recordings.
TIMELINES = [{"name": "step", "kind": "sequence"}, {"name": "time", "kind": "time", "epoch": True},
             {"name": "episode", "kind": "sequence"}]
STEP_TIME = np.array([[1200, 1763200000123456789], [1201, 1763200000128456789],
                      [1202, 1763200000133456789]], np.int64)
RECORDING = {"id": "train-20261115-120314", "title": "train.py", "script": "train.py",
             "bytes": 1048576, "items": 20000, "created_unix_ms": 1763208194000,
             "ended_unix_ms": None, "complete": False,
             "timelines": [{"name": "step", "kind": "sequence", "min": "0", "max": "19999"},
                           {"name": "time", "kind": "time", "epoch": True,
                            "min": "1763208194000000000", "max": "1763208794000000000"}]}
LOG = {"id": 9, "name": "log", "kind": int(w.Kind.TEXT), "sources": []}
SCORES = {"id": 10, "name": "scores", "kind": int(w.Kind.BARS),
          "sources": [{"name": "scores", "dtype": "f32", "dims": [{"name": "class", "size": 3}]}]}
LOSS_RECORDED = {"id": 3, "name": "loss", "kind": int(w.Kind.SERIES), "sources": [],
                 "recorded": [{"budget": {"max_backfill": 4096}}]}


SPLIT = w.SceneUpdate(2, "world", 37, STAMP + 3900, "world/split", w.PART_POINTS,
                      {"positions": np.arange(60, dtype=np.float32).reshape(20, 3) / 8,
                       "colors": (np.arange(80) * 3 % 256).astype(np.uint8).reshape(20, 4)},
                      {"radius": 0.25}, stamp=STAMP_T)


def _timeline_vectors() -> dict:
    loss = {"sub": 3, "view": "loss", "stamp_ns": 81234567891123}
    bars_dv = w.resolve_data_view(SCORES, None)
    return {
        "views_timelines": w.Views([LOSS_RECORDED, LOG, SCORES], SESSION["id"], TIMELINES),
        "sessions_recording": w.Sessions([dict(SESSION, state="ended", views=3, recording={
            "id": "train-20261115-120314", "seekable": True, "complete": True})]),
        # The draft's example frame: series.bin of 3.0 with a stamp per value.
        "series_stamps": w.Series(start=0, total=3, values=_f32([2.5, 1.25, 0.75]),
                                  stamps=STEP_TIME, **loss),
        "series_stamp": w.Series(start=3, total=5, values=_f32([0.5, 0.25]), stamp=STAMP_T,
                                 **dict(loss, stamp_ns=81234567891999)),
        "series_decimated": w.Series(start=0, total=4096, values=_f32([3, 2, 1]),
                                     indices=np.array([0, 2048, 4095], np.int64), decimated=1024,
                                     cursor=2, stamps=STEP_TIME, **loss),
        "snapshot_stamped": w.Snapshot(1, "attn", 5, STAMP + 4000, int(w.Kind.HEATMAP),
                                       {"source": "layer0"}, _f32([[0, 1], [2, 3]]),
                                       stats=[0.0, 3.0], stamp=STAMP_T),
        "snapshot_static": w.Snapshot(1, "attn", 6, STAMP + 4100, int(w.Kind.HEATMAP),
                                      {"source": "layer0"}, _f32([[1]]), stats=[1.0, 1.0],
                                      static=True),
        "snapshot_empty": w.Snapshot(1, "attn", 7, STAMP + 4200, int(w.Kind.HEATMAP),
                                     {"source": "layer0"}, None, empty=True, cursor=3),
        "snapshot_bars": w.Snapshot(4, "scores", 8, STAMP + 4300, int(w.Kind.BARS), bars_dv,
                                    _f32([0.125, 0.75, -0.25]),
                                    colors=np.array([[255, 0, 0, 255], [0, 128, 0, 255],
                                                     [0, 0, 255, 255]], np.uint8),
                                    stats=[-0.25, 0.75], labels=["cat", "dog", "fish"],
                                    horizontal=True, stamp=STAMP_T),
        "controls_stamped": w.Controls({"state": "running", "step": 1200, "steps_left": None},
                                       [], [], stamp=STAMP_T),
        "text": w.Text.from_lines(5, "log", 0, 3, STAMP + 4400,
                                  ["epoch 1 starts", "loss rose\nto 2.5", "pas de données é"],
                                  [w.LEVEL_INFO, w.LEVEL_WARNING, w.LEVEL_ERROR], stamps=STEP_TIME),
        "text_stamp": w.Text.from_lines(5, "log", 10, 12, STAMP + 4500, ["a", ""],
                                        [w.LEVEL_DEBUG, w.LEVEL_INFO], dropped=10, stamp=STAMP_T,
                                        cursor=1),
        "recordings": w.Recordings([RECORDING]),
        "recordings_none": w.Recordings([]),
        "recording_open": w.RecordingOpen(1, "train-20261115-120314"),
        "cursor_seek": w.Cursor(2, w.CURSOR_SEEK, timeline="step", value="5000"),
        "cursor_play": w.Cursor(3, w.CURSOR_PLAY, timeline="step", speed=2.0, pace=w.PACE_EVEN,
                                loop=["1000", "5000"]),
        "cursor_step": w.Cursor(4, w.CURSOR_STEP, timeline="time", count=-5),
        "cursor_pause": w.Cursor(5, w.CURSOR_PAUSE),
        "cursor_follow": w.Cursor(6, w.CURSOR_FOLLOW),
        "time_state_paused": w.TimeState("train-20261115-120314", 2, w.TIME_PAUSED, "step",
                                         "5000", {"step": ["0", "19999"],
                                                  "time": ["1763208194000000000",
                                                           "1763208794000000000"]},
                                         False, request_id=2),
        "time_state_following": w.TimeState("clp-4242-a1b2c3", 1, w.TIME_FOLLOWING, "time", None,
                                            {"time": ["-5", "1763208794000000000"]}, True,
                                            speed=1.0, pace=w.PACE_RECORDED),
    }


def valid_vectors() -> dict[str, bytes]:
    u8, i32, u16 = np.uint8, np.int32, np.uint16
    msgs = {
        "hello": w.Hello(w.ROLE_VIEWER, [w.CAP_CONTROLS, w.CAP_SCENES, w.CAP_SELECTION],
                         VIEWER_SOFTWARE),
        "welcome": w.Welcome(w.ROLE_PRODUCER, [w.CAP_CONTROLS, w.CAP_EVENTS, w.CAP_SCENES,
                                               w.CAP_SELECTION],
                             PRODUCER_SOFTWARE, SESSION, control=True),
        "views_scene": w.Views([LANDSCAPE, PLANE_SHARE], SESSION["id"]),
        "views_scene_nulls": w.Views([NULLS], SESSION["id"]),
        "views_scene_camera": w.Views([WORLD, OVERLAY], SESSION["id"]),
        "view_reply_scene": w.ViewReply(9, 2, "landscape",
                                        data_view=w.resolve_data_view(LANDSCAPE, {})),
        # The draft's example frame, with and without its stamp.
        "scene_update_points": _update(2, "world", 12, STAMP + 1377, "world/cloud", w.PART_POINTS,
                                       {"positions": CLOUD, "colors": RGB}, {}),
        "scene_update_points_stamped": _update(2, "world", 12, STAMP + 1377, "world/cloud",
                                               w.PART_POINTS, {"positions": CLOUD, "colors": RGB},
                                               {}, stamp=STAMP_T),
        "scene_update_points_color": _update(
            2, "landscape", 41, STAMP + 3000, "current", w.PART_POINTS, {"positions": CURRENT},
            {"color": [255, 80, 80, 255], "stats": _stats(None, CURRENT)}),
        "scene_update_points_values": _update(
            2, "landscape", 7, STAMP + 3100, "markers", w.PART_POINTS,
            {"positions": MARKERS, "values": MARKER_VALUES, "radii": MARKER_RADII},
            {"colormap": w.COLORMAP_MAGMA, "flags": w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX,
             "vmin": 0.0, "vmax": 1.0, "stats": _stats(MARKER_VALUES, MARKERS)}, static=True),
        "scene_update_points_labels": _update(
            2, "world", 20, STAMP + 3150, "graph/nodes", w.PART_POINTS,
            {"positions": CLOUD, "class_ids": np.array([0, 1, 1], i32)},
            {"labels": ["a", "b", "c"]}, stamp=STAMP_T),
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
        # Index-pair edges: a triangle of three edges, one value each.
        "scene_update_lines_edges": _update(
            2, "world", 21, STAMP + 3450, "graph/edges", w.PART_LINES,
            {"positions": CLOUD, "indices": np.array([[0, 1], [1, 2], [2, 0]], i32),
             "values": _f32([0.25, 0.5, 1.0])},
            {"mode": w.LINES_EDGES, "color_per": w.COLOR_PER_EDGE, "width": 1.5,
             "colormap": w.COLORMAP_MAGMA, "flags": 0, "vmin": 0.0, "vmax": 1.0}, stamp=STAMP_T),
        "scene_update_arrows": _update(
            2, "world", 22, STAMP + 3460, "field", w.PART_ARROWS,
            {"origins": _f32([[0, 0, 0], [1, 0, 0]]), "vectors": _f32([[0, 0, 1], [0, 0.5, 0]]),
             "colors": RGB[:2]}, {"width": 2.0}, stamp=STAMP_T),
        "scene_update_boxes": _update(
            2, "world", 23, STAMP + 3470, "arm/link0", w.PART_BOXES,
            {"centers": _f32([[0, 0, 0.25]]), "half_sizes": _f32([[0.0625, 0.0625, 0.25]]),
             "rotations": _f32([[0, 0, 0, 1]]), "class_ids": np.array([1], i32)},
            {"labels": ["link"], "width": 2.0}, static=True),
        "scene_update_boxes_2d": _update(
            3, "overlay", 4, STAMP + 3480, "rig/left/cars", w.PART_BOXES,
            {"centers": _f32([[10, 20], [40.5, 8]]), "half_sizes": _f32([[4, 3], [2.5, 2]]),
             "colors": RGB[1:]}, {"labels": ["car"], "fill": True}, stamp=STAMP_T),
        "scene_update_mesh": _update(
            2, "landscape", 39, STAMP + 3500, "surface", w.PART_MESH,
            {"positions": GRID, "indices": GRID_TRIANGLES, "values": GRID_VALUES},
            {"colormap": w.COLORMAP_VIRIDIS, "flags": 0, "vmin": -4.0, "vmax": 2.0,
             "stats": _stats(GRID_VALUES, GRID)}),
        "scene_update_mesh_normals": _update(
            2, "landscape", 39, STAMP + 3500, "surface", w.PART_MESH,
            {"positions": GRID, "indices": GRID_TRIANGLES,
             "normals": np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))}, {}),
        "scene_update_mesh_textured": _update(
            2, "world", 24, STAMP + 3550, "floor", w.PART_MESH,
            {"positions": GRID, "indices": GRID_TRIANGLES,
             "uvs": _f32([[0, 0], [1, 0], [0, 1], [1, 1]]),
             "texture": np.array([[[255, 0, 0, 255], [0, 255, 0, 255]],
                                  [[0, 0, 255, 255], [255, 255, 255, 255]]], u8)},
            {}, static=True),
        "scene_update_mesh_colors": _update(
            3, "flat", 2, STAMP + 3600, "shapes/tri", w.PART_MESH,
            {"positions": FLAT, "colors": RGB}, {}, static=True, state=True),
        "scene_update_image_color": _update(
            2, "world", 25, STAMP + 3650, "rig/left/rgb", w.PART_IMAGE,
            {"pixels": np.arange(24, dtype=u8).reshape(2, 3, 4) * 10},
            {"form": w.FORM_COLOR, "opacity": 1.0}, stamp=STAMP_T),
        "scene_update_image_depth": _update(
            2, "world", 26, STAMP + 3660, "rig/left/depth", w.PART_IMAGE,
            {"pixels": np.array([[500, 1000, 0], [1500, 2000, 2500]], u16)},
            {"form": w.FORM_DEPTH, "scale": 0.001, "cloud": True, "colormap": w.COLORMAP_VIRIDIS,
             "flags": w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX, "vmin": 0.0, "vmax": 1.0,
             "stats": [0.5, 2.5]}, stamp=STAMP_T),
        "scene_update_image_depth_f32": _update(
            3, "overlay", 5, STAMP + 3670, "elevation", w.PART_IMAGE,
            {"pixels": _f32([[-1, 0.5], [0.25, 1]])},
            {"form": w.FORM_DEPTH, "colormap": w.COLORMAP_RDBU, "flags": 0, "vmin": -1.0,
             "vmax": 1.0}, static=True),
        "scene_update_image_segmentation": _update(
            2, "world", 27, STAMP + 3680, "rig/left/classes", w.PART_IMAGE,
            {"pixels": np.array([[0, 1, 1], [2, 2, 0]], u8)},
            {"form": w.FORM_SEGMENTATION, "opacity": 0.5}, stamp=STAMP_T),
        "scene_update_image_segmentation_u16": _update(
            3, "overlay", 6, STAMP + 3690, "labels", w.PART_IMAGE,
            {"pixels": np.array([[0, 300], [65535, 7]], u16)},
            {"form": w.FORM_SEGMENTATION, "opacity": 0.75}),
        "scene_update_image_encoded": _update(
            2, "world", 28, STAMP + 3700, "rig/right/rgb", w.PART_IMAGE,
            {"pixels": np.frombuffer(PNG, u8)},
            {"form": w.FORM_ENCODED, "format": w.FORMAT_PNG, "opacity": 1.0}, stamp=STAMP_T),
        "scene_update_camera": _update(
            2, "world", 29, STAMP + 3710, "rig/left", w.PART_CAMERA, {},
            {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480,
             "convention": w.CONVENTION_RDF}, static=True),
        "scene_update_camera_rub": _update(
            2, "world", 30, STAMP + 3720, "sim/eye", w.PART_CAMERA, {},
            {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240,
             "convention": w.CONVENTION_RUB, "plane_distance": 0.25}, stamp=STAMP_T),
        "scene_transforms": w.SceneTransforms(
            2, "world", 31, STAMP + 3730, ["arm/link0", "arm/link0/link1"],
            _f32([_matrix(0, 0, 0), _matrix(0, 0, 0.5)]), stamp=STAMP_T),
        "scene_transforms_stamps": w.SceneTransforms(
            2, "world", 32, STAMP + 3740, ["a", "b"], _f32([_matrix(1, 0, 0), _matrix(0, 2, 0)]),
            stamps=np.array([[1200, 1763200000123456789], [1201, 1763200000128456789]], np.int64)),
        "scene_transforms_static": w.SceneTransforms(
            2, "world", 33, STAMP + 3750, ["rig/left", "rig/right"],
            _f32([_matrix(-0.0625, -2.5, 1.25), _matrix(0.0625, -2.5, 1.25)]), static=True,
            state=True),
        "scene_classes": w.SceneClasses(
            2, "world", 34, STAMP + 3760,
            [{"id": 0, "label": "floor", "color": [128, 128, 128, 255]},
             {"id": 1, "label": "obstacle", "color": [230, 80, 60, 255]},
             {"id": 65535, "label": "unknown", "color": [0, 0, 0, 0]}], static=True),
        "scene_classes_stamped": w.SceneClasses(2, "world", 35, STAMP + 3770, [], stamp=STAMP_T),
        "scene_clear_reset": w.SceneClear(2, "landscape", 41, STAMP + 3700, "", reset=True,
                                          state=True),
        "scene_clear": w.SceneClear(2, "world", 15, STAMP + 3800, "world/cloud", recursive=True,
                                    parts=[w.PART_POINTS, w.PART_MESH]),
        "scene_clear_stamped": w.SceneClear(2, "world", 36, STAMP + 3810, "arm", recursive=True,
                                            parts=[w.PART_TRANSFORM, w.PART_BOXES],
                                            stamp=STAMP_T),
    }
    msgs.update(_timeline_vectors())
    out = {f"v3_2/{name}": w.encode(m, minor=M) for name, m in msgs.items()}
    # A split update: 20 coloured points in five pieces of at most 600 bytes, with a stamp.
    for k, piece in enumerate(w.split_update(SPLIT, update=4, max_frame=600)):
        out[f"v3_2/scene_update_split_{k}"] = w.encode(piece, minor=M)
    # Decode only: a section that no part has, which decoders ignore.
    out["v3_2/scene_update_points_future"] = w.encode_frame(
        w.MsgType.SCENE_UPDATE,
        {"part": "points", "path": "world/cloud", "seq": 12, "stamp_ns": STAMP + 1377, "sub": 2,
         "view": "world"},
        [w.Section.from_array("positions", CLOUD), w.Section.from_array("future", _f32([1, 2, 3]))],
        minor=M)
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


def _message(type_: int, **meta) -> bytes:
    """A frame of another scene message from literal keys and sections."""
    sections = meta.pop("sections", {})
    base = {"seq": 1, "stamp_ns": STAMP, "sub": 2, "view": "s", **meta}
    return w.encode_frame(type_, base, [w.Section.from_array(n, a) for n, a in sections.items()],
                          minor=M)


def invalid_vectors(valid: dict[str, bytes]) -> dict[str, tuple[bytes, dict]]:
    P = w.CLOSE_PROTOCOL_ERROR
    bad = {"error": w.ERR_BAD_REQUEST}
    mesh = valid["v3_2/scene_update_mesh"]
    pos_meta = (b'{"part":"points","path":"p","sections":[{"dtype":0,"name":"positions",'
                b'"shape":[3,3]}],"seq":1,"stamp_ns":1,"sub":2,"view":"s"}')
    u16_meta = (b'{"form":"depth","part":"image","path":"d","sections":[{"dtype":6,"name":"pixels",'
                b'"shape":[2,2]}],"seq":1,"stamp_ns":1,"sub":2,"view":"s"}')
    deep = "/".join(f"s{i}" for i in range(17))
    u8, i32 = np.uint8, np.int32
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480,
              "convention": "rdf"}
    mats = _f32([_matrix(0, 0, 0), _matrix(1, 0, 0)])
    T, K = w.MsgType.SCENE_TRANSFORMS, w.MsgType.SCENE_CLASSES
    return {
        "truncated_mesh": (mesh[:-4], {"reason": "truncated", "close_code": P}),
        "positions_length_mismatch": (_raw(w.MsgType.SCENE_UPDATE, pos_meta, [bytes(32)]),
                                      {"reason": "malformed", "close_code": P}),
        "u16_length_mismatch": (_raw(w.MsgType.SCENE_UPDATE, u16_meta, [bytes(6)]),
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
        "split_piece": (_frame("points", "p", {"positions": CLOUD},
                               split={"update": 1, "piece": 0, "pieces": 2, "first": [0],
                                      "rows": [6]}), bad),
        "path_dotdot": (_frame("points", "world/..", {"positions": CLOUD}), bad),
        "path_too_deep": (_frame("points", deep, {"positions": CLOUD}), bad),
        "clear_reset_path": (w.encode_frame(w.MsgType.SCENE_CLEAR,
                                            {"path": "world", "reset": True, "seq": 1,
                                             "stamp_ns": STAMP, "sub": 2, "view": "s"}, minor=M),
                             bad),
        # stamps
        "stamp_static": (_frame("points", "p", {"positions": CLOUD, "stamp": STAMP_T},
                                static=True), bad),
        "stamp_f32": (_frame("points", "p", {"positions": CLOUD, "stamp": _f32([1, 2])}), bad),
        "stamp_too_long": (_frame("points", "p", {"positions": CLOUD,
                                                  "stamp": np.zeros(9, np.int64)}), bad),
        "stamp_rank_2": (_frame("points", "p", {"positions": CLOUD,
                                                "stamp": np.zeros((1, 2), np.int64)}), bad),
        "clear_stamp_static": (_message(w.MsgType.SCENE_CLEAR, path="a", static=True,
                                        sections={"stamp": STAMP_T}), bad),
        # points and lines
        "labels_count": (_frame("points", "p", {"positions": CLOUD}, labels=["a", "b"]), bad),
        "labels_long": (_frame("points", "p", {"positions": CLOUD}, labels=["x" * 64]), bad),
        "class_ids_rows": (_frame("points", "p", {"positions": CLOUD,
                                                  "class_ids": np.array([1, 2], i32)}), bad),
        "edges_indices_missing": (_frame("lines", "g", {"positions": CLOUD}, mode="edges"), bad),
        "edges_2e": (_frame("lines", "g", {"positions": CLOUD,
                                           "indices": np.array([[0, 1, 2], [1, 2, 0]], i32)},
                            mode="edges"), bad),
        "edges_color_per_strip": (_frame("lines", "g", {"positions": CLOUD,
                                                        "indices": np.array([[0, 1]], i32)},
                                         mode="edges", color_per="strip"), bad),
        "edges_strip_lengths": (_frame("lines", "g", {"positions": CLOUD,
                                                      "strip_lengths": np.array([3], i32),
                                                      "indices": np.array([[0, 1]], i32)},
                                       mode="edges"), bad),
        "edge_values_rows": (_frame("lines", "g", {"positions": CLOUD,
                                                   "indices": np.array([[0, 1]], i32),
                                                   "values": _f32([1, 2])},
                                    mode="edges", color_per="edge"), bad),
        # arrows and boxes
        "arrows_vectors_missing": (_frame("arrows", "p", {"origins": CLOUD}), bad),
        "arrows_vectors_rows": (_frame("arrows", "p", {"origins": CLOUD, "vectors": CLOUD[:2]}),
                                bad),
        "boxes_rotations_2d": (_frame("boxes", "b", {"centers": FLAT, "half_sizes": FLAT,
                                                     "rotations": np.zeros((3, 4), np.float32)}),
                               bad),
        "boxes_half_sizes_dims": (_frame("boxes", "b", {"centers": CLOUD, "half_sizes": FLAT}),
                                  bad),
        # meshes
        "mesh_uvs_without_texture": (_frame("mesh", "m", {"positions": FLAT,
                                                          "uvs": np.zeros((3, 2), np.float32)}),
                                     bad),
        "mesh_uvs_and_colors": (_frame("mesh", "m", {"positions": FLAT, "colors": RGB,
                                                     "uvs": np.zeros((3, 2), np.float32),
                                                     "texture": np.zeros((1, 1, 4), u8)}), bad),
        "mesh_texture_rgb": (_frame("mesh", "m", {"positions": FLAT,
                                                  "uvs": np.zeros((3, 2), np.float32),
                                                  "texture": np.zeros((1, 1, 3), u8)}), bad),
        "mesh_normals_2d": (_frame("mesh", "m", {"positions": FLAT,
                                                 "normals": np.zeros((3, 2), np.float32)}), bad),
        # images
        "image_form_missing": (_frame("image", "i", {"pixels": np.zeros((2, 2), u8)}), bad),
        "image_form_unknown": (_frame("image", "i", {"pixels": np.zeros((2, 2), u8)},
                                      form="video"), bad),
        "image_pixels_missing": (_frame("image", "i", {}, form="color"), bad),
        "image_depth_u8": (_frame("image", "i", {"pixels": np.zeros((2, 2), u8)}, form="depth"),
                           bad),
        "image_color_rgb": (_frame("image", "i", {"pixels": np.zeros((1, 2, 3), u8)},
                                   form="color"), bad),
        "image_side": (_frame("image", "i", {"pixels": np.zeros((8193, 1), u8)}, form="color"),
                       bad),
        "image_encoded_format_missing": (_frame("image", "i", {"pixels": np.frombuffer(PNG, u8)},
                                                form="encoded"), bad),
        "image_opacity_range": (_frame("image", "i", {"pixels": np.zeros((2, 2), u8)},
                                       form="segmentation", opacity=1.5), bad),
        # cameras
        "camera_fx_missing": (_frame("camera", "c", {}, **{k: v for k, v in camera.items()
                                                          if k != "fx"}), bad),
        "camera_convention_missing": (_frame("camera", "c", {}, **{
            k: v for k, v in camera.items() if k != "convention"}), bad),
        "camera_width_fraction": (_frame("camera", "c", {}, **dict(camera, width=640.5)), bad),
        "camera_height_zero": (_frame("camera", "c", {}, **dict(camera, height=0)), bad),
        # entity transforms
        "transforms_root": (_message(T, paths=["", "a"], sections={"matrices": mats}), bad),
        "transforms_repeat": (_message(T, paths=["a", "a"], sections={"matrices": mats}), bad),
        "transforms_rows": (_message(T, paths=["a"], sections={"matrices": mats}), bad),
        "transforms_matrices_missing": (_message(T, paths=["a"]), bad),
        "transforms_stamps_rows": (_message(T, paths=["a", "b"], sections={
            "matrices": mats, "stamps": np.zeros((1, 2), np.int64)}), bad),
        "transforms_stamp_and_stamps": (_message(T, paths=["a", "b"], sections={
            "matrices": mats, "stamp": STAMP_T, "stamps": np.zeros((2, 2), np.int64)}), bad),
        # class tables
        "classes_order": (_message(K, classes=[{"id": 2, "label": "b", "color": [0, 0, 0, 255]},
                                               {"id": 1, "label": "a", "color": [0, 0, 0, 255]}]),
                          bad),
        "classes_id_range": (_message(K, classes=[{"id": 65536, "label": "x",
                                                   "color": [0, 0, 0, 255]}]), bad),
        "classes_color_missing": (_message(K, classes=[{"id": 1, "label": "x"}]), bad),
        **_timeline_invalid(valid),
    }


def _meta_frame(type_, meta: dict, sections: dict | None = None) -> bytes:
    return w.encode_frame(type_, meta, [w.Section.from_array(n, a)
                                        for n, a in (sections or {}).items()], minor=M)


def _timeline_invalid(valid: dict) -> dict:
    bad = {"error": w.ERR_BAD_REQUEST}
    cursor_bad = {"error": w.ERR_BAD_CURSOR}
    S, R, X, C, TS, V, RO = (w.MsgType.SNAPSHOT, w.MsgType.SERIES, w.MsgType.TEXT,
                             w.MsgType.CURSOR, w.MsgType.TIME_STATE, w.MsgType.VIEWS,
                             w.MsgType.RECORDING_OPEN)
    snap = {"sub": 1, "view": "v", "seq": 1, "stamp_ns": 1, "kind": 1, "data_view": {}}
    bars = dict(snap, kind=8)
    series = {"sub": 3, "view": "loss", "start": 0, "total": 3, "stamp_ns": 1}
    text = {"sub": 5, "view": "log", "start": 0, "total": 2, "stamp_ns": 1}
    v3 = _f32([1, 2, 3])

    def lines(raw: bytes, offs, levels):
        return {"text": np.frombuffer(raw, np.uint8), "offsets": np.array(offs, np.int32),
                "levels": np.array(levels, np.uint8)}
    split = {"part": "points", "path": "p", "seq": 1, "stamp_ns": 1, "sub": 2, "view": "s"}
    pos = {"positions": CLOUD}
    return {
        "snapshot_stamp_static": (_meta_frame(S, dict(snap, static=True), {
            "data": _f32([[1]]), "stamp": STAMP_T}), bad),
        "snapshot_empty_with_data": (_meta_frame(S, dict(snap, empty=True),
                                                 {"data": _f32([[1]])}), bad),
        "bars_too_many": (_meta_frame(S, bars, {"data": np.zeros(4097, np.float32)}), bad),
        "bars_labels_count": (_meta_frame(S, dict(bars, labels=["a", "b"]), {"data": v3}), bad),
        "bars_colors_rows": (_meta_frame(S, bars, {"data": v3, "colors": RGB[:2]}), bad),
        "bars_rank_2": (_meta_frame(S, bars, {"data": _f32([[1, 2]])}), bad),
        "series_stamps_rows": (_meta_frame(R, series, {"values": v3,
                                                       "stamps": STEP_TIME[:2]}), bad),
        "series_indices_alone": (_meta_frame(R, series, {
            "values": v3, "indices": np.array([0, 1, 2], np.int64)}), bad),
        "series_decimated_zero": (_meta_frame(R, dict(series, decimated=0), {
            "values": v3, "indices": np.array([0, 1, 2], np.int64)}), bad),
        "controls_stamp_f32": (_meta_frame(w.MsgType.CONTROLS, {
            "run": {"state": "none", "step": None, "steps_left": None}, "params": [],
            "actions": []}, {"stamp": _f32([1, 2])}), bad),
        "text_offsets_end": (_meta_frame(X, text, lines(b"ab", [0, 1, 1], [1, 1])), bad),
        "text_offsets_decrease": (_meta_frame(X, text, lines(b"ab", [0, 2, 1, 2], [1, 1, 1])
                                              ), bad),
        "text_line_long": (_meta_frame(X, dict(text, total=1),
                                       lines(b"x" * 4097, [0, 4097], [1])), bad),
        "text_level": (_meta_frame(X, text, lines(b"ab", [0, 1, 2], [1, 4])), bad),
        "text_utf8": (_meta_frame(X, text, lines(b"a\xc3", [0, 1, 2], [1, 1])), bad),
        "text_total": (_meta_frame(X, dict(text, total=1), lines(b"ab", [0, 1, 2], [1, 1])), bad),
        "text_levels_missing": (_meta_frame(X, text, {"text": np.frombuffer(b"ab", np.uint8),
                                                      "offsets": np.array([0, 1, 2], np.int32)}),
                                bad),
        "views_timelines_order": (_meta_frame(V, {"session": None, "views": [], "timelines": [
            TIMELINES[1], TIMELINES[0]]}), bad),
        "views_timelines_repeat": (_meta_frame(V, {"session": None, "views": [], "timelines": [
            TIMELINES[0], TIMELINES[1], TIMELINES[0]]}), bad),
        "views_recorded_many": (_meta_frame(V, {"session": None, "views": [
            dict(LOSS_RECORDED, recorded=[{}] * 65)]}), bad),
        "recordings_id": (_meta_frame(w.MsgType.RECORDINGS, {"recordings": [
            dict(RECORDING, id="runs/train")]}), bad),
        "recording_open_id": (_meta_frame(RO, {"request_id": 1, "recording": ""}), bad),
        "cursor_op": (_meta_frame(C, {"request_id": 1, "op": "rewind"}), bad),
        "cursor_request_id": (_meta_frame(C, {"op": "pause"}), bad),
        "cursor_seek_text": (_meta_frame(C, {"request_id": 1, "op": "seek", "timeline": "step",
                                             "value": "12a"}), cursor_bad),
        "cursor_seek_range": (_meta_frame(C, {"request_id": 1, "op": "seek", "timeline": "step",
                                              "value": "9223372036854775808"}), cursor_bad),
        "cursor_seek_timeline": (_meta_frame(C, {"request_id": 1, "op": "seek",
                                                 "value": "5"}), cursor_bad),
        "cursor_step_zero": (_meta_frame(C, {"request_id": 1, "op": "step", "timeline": "step",
                                             "count": 0}), cursor_bad),
        "cursor_speed": (_meta_frame(C, {"request_id": 1, "op": "play", "timeline": "step",
                                         "speed": 128}), cursor_bad),
        "cursor_loop_order": (_meta_frame(C, {"request_id": 1, "op": "play", "timeline": "step",
                                              "loop": ["9", "1"]}), cursor_bad),
        "time_state_cursor_following": (_meta_frame(TS, {
            "session": "s", "generation": 1, "state": "following", "timeline": "step",
            "cursor": "5", "ranges": {}, "live": True}), bad),
        "time_state_ranges": (_meta_frame(TS, {
            "session": "s", "generation": 1, "state": "paused", "timeline": "step",
            "cursor": "5", "ranges": {"step": ["1"]}, "live": False}), bad),
        "split_rows_missing": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {"positions": 0}}), bad),
        "split_first_names": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {}, "rows": {"positions": 6}}), bad),
        "split_one_piece": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 1, "first": {"positions": 0},
            "rows": {"positions": 3}}), bad),
        "split_rows_later": (_frame("points", "p", pos, split={
            "update": 1, "piece": 1, "pieces": 2, "first": {"positions": 3},
            "rows": {"positions": 6}}), bad),
        "split_rows_pass": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {"positions": 2},
            "rows": {"positions": 4}}), bad),
    }
