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

import hashlib
import struct

import numpy as np

from caliper import _wire as w
# Per-shape encoders exist only in the test tree.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/tests"))
import legacy_scene_encoder
legacy_scene_encoder.install()

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
    msgs["scene_update_lines_unused"] = _update(
        2, "world", 40, STAMP + 3900, "unused", w.PART_LINES,
        {"positions": CLOUD, "indices": np.array([[0, 1]], i32)}, {"mode": w.LINES_EDGES})
    msgs["scene_update_lines_singletons"] = _update(
        2, "world", 41, STAMP + 3901, "singleton", w.PART_LINES,
        {"positions": CLOUD, "strip_lengths": np.array([1, 1, 1], i32)}, {"mode": w.LINES_STRIPS})
    msgs.update(_timeline_vectors())
    msgs.update(_program_vectors())
    msgs.update(_resource_vectors())
    msgs.update(_scene_piece_vectors())
    msgs.update(_scene_bundles(msgs))
    out = {f"v3_2/{name}": w.encode(m, minor=M) for name, m in msgs.items()}
    # A split update: 20 coloured points in five pieces of at most 600 bytes, with a stamp.
    for k, piece in enumerate(w.split_update(SPLIT, update=4, max_frame=600)):
        out[f"v3_2/scene_update_split_{k}"] = w.encode(piece, minor=M)
    # A version in parts: 10 segments of 8 bytes in three parts of at most 32 bytes, and the parts that
    # a receiver drops or refuses in a sequence (_resource_split_vectors).
    for k, part in enumerate(w.resource_parts(RESOURCE_SPLIT, version=5, part_bytes=32)):
        out[f"v3_2/resource_split_{k}"] = w.encode(part, minor=M)
    out.update({f"v3_2/{name}": w.encode(m, minor=M) for name, m in _resource_split_vectors().items()})
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
        **_program_invalid(),
        **_resource_invalid(),
        **_scene_piece_invalid(),
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
        # A first row that a 64-bit sum wraps back below the rows: 2^64 - 1 is -1 as a signed
        # number, and 2^64 - 3 plus the 3 rows of the piece is 0.
        "split_first_max": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {"positions": 2**64 - 1},
            "rows": {"positions": 6}}), bad),
        "split_first_wraps": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {"positions": 2**64 - 3},
            "rows": {"positions": 6}}), bad),
        "split_first_rows": (_frame("points", "p", pos, split={
            "update": 1, "piece": 0, "pieces": 2, "first": {"positions": 6},
            "rows": {"positions": 6}}), bad),
    }


# ---------------------------------------------------------------------------
# GPU programs and interventions
# ---------------------------------------------------------------------------
WGSL = "@compute @workgroup_size(64) fn main() {} // \u00b5\n"
GEOMETRY = {"entry": "main", "inputs": [{"var": "w", "source": "live:fc1.weight"}],
            "output": {"kind": "geometry", "entity": "programs/rows", "part": "lines",
                       "max_count": 4096, "arrays": ["positions", "values"]},
            "limits": {"every": 5}, "workgroups": [64, 1, 1]}
PIXELS = {"inputs": [{"var": "w", "source": "live:fc1.weight"}],
          "output": {"kind": "pixels", "width": 64, "height": 32, "format": "rgba8unorm"}}


TENSOR = {"inputs": [{"var": "w", "source": "live:fc1.weight"}],
          "output": {"kind": "tensor", "shape": [64, 32]}}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _program_vectors() -> dict:
    running = {"id": "rows", "kind": "geometry", "state": "running", "every": 12, "every_min": 5,
               "last_pass_ms": 0.25, "last_cost_ms": 11.71875, "step_ms": 31.25,
               "overhead_share": 0.03125, "passes_per_s": 2.5, "last_step": 1200,
               "runs": 7, "memory_bytes": 1310720, "memory_limit_bytes": 67108864,
               "error": None, "sha256": _sha(WGSL)}
    refused = {"id": "image", "kind": "pixels", "state": "refused", "every_min": 1,
               "error": {"code": "program.unknown_input", "message": "No tensor fc9.weight.",
                         "line": 0, "column": 0}}
    return {
        "program_set_geometry": w.Program(11, w.PROGRAM_SET, "rows", GEOMETRY, WGSL, _sha(WGSL),
                                          "world"),
        "program_set_pixels": w.Program(12, w.PROGRAM_SET, "image", PIXELS, WGSL, _sha(WGSL)),
        "program_remove": w.Program(13, w.PROGRAM_REMOVE, "rows"),
        "program_set_tensor": w.Program(14, w.PROGRAM_SET, "norms", TENSOR, WGSL, _sha(WGSL)),
        "program_set_operator": w.Program(15, w.PROGRAM_SET, "operator.row-norms.12", TENSOR, WGSL,
                                          _sha(WGSL)),
        "program_params": w.Program(16, w.PROGRAM_PARAMS, "rows",
                                    params={"gain": 1.5, "_bias": -2, "Scale2": 0.0}),
        "program_params_operator": w.Program(17, w.PROGRAM_PARAMS, "operator.row-norms.12",
                                             params={"gain": 0.25}),
        "program_state_operator": w.ProgramState([
            {"id": "operator.row-norms.12", "kind": "operator", "state": "running", "runs": 3},
            {"id": "norms", "kind": "tensor", "state": "compiled"}]),
        "program_state_reply": w.ProgramState([running, refused], 12, 1200, 31.25),
        "program_state_waiting": w.ProgramState([{"id": "rows", "kind": "geometry",
                                                  "state": "compiled"}]),
        "control_intervene_set": w.Control(21, w.OP_INTERVENE, intervene={"set": {
            "point": "blocks.0.attn#3", "op": w.INTERVENE_ABLATE, "from_step": 1201,
            "until_step": None}}),
        "control_intervene_clear": w.Control(22, w.OP_INTERVENE,
                                             intervene={"clear": "blocks.0.attn#3"}),
        "control_intervene_clear_all": w.Control(23, w.OP_INTERVENE,
                                                 intervene={"clear_all": True}),
        "control_reply_intervene": w.ControlReply(21, w.OP_INTERVENE, w.STATUS_ERROR, error={
            "code": "intervene.unknown_point", "message": "The model has no point blocks.9."}),
        "controls_interventions": w.Controls(
            {"state": w.RUN_RUNNING, "step": 1200, "steps_left": 0}, [], [],
            interventions={
                "points": [{"point": "blocks.0.attn", "kind": "module", "module": "blocks.0.attn",
                            "head": None, "type": "Attention", "shape": [8, 64, 128]},
                           {"point": "blocks.0.attn#3", "kind": "head",
                            "module": "blocks.0.attn", "head": 3, "type": "Attention",
                            "shape": None}],
                "active": [{"point": "blocks.0.attn#3", "op": "ablate", "from_step": 1201,
                            "until_step": None, "since_step": 1201, "state": "active"}]}),
    }


def _program_invalid() -> dict:
    bad = {"error": w.ERR_BAD_REQUEST}
    PG, PS, C, CS = (w.MsgType.PROGRAM, w.MsgType.PROGRAM_STATE, w.MsgType.CONTROL,
                     w.MsgType.CONTROLS)
    src = np.frombuffer(WGSL.encode("utf-8"), np.uint8)
    base = {"request_id": 1, "op": "set", "id": "rows", "declaration": GEOMETRY,
            "scene": "world", "sha256": _sha(WGSL)}

    def prog(sections=None, **change):
        meta = {k: v for k, v in dict(base, **change).items() if v is not None}
        return _meta_frame(PG, meta, {"wgsl": src} if sections is None else sections)

    def decl(**change):
        return dict(GEOMETRY, **change)

    def out(**change):
        return decl(output=dict(GEOMETRY["output"], **change))

    item = {"id": "rows", "kind": "geometry", "state": "running"}

    def tensor(**change):
        return decl(output=dict({"kind": "tensor", "shape": [64, 32]}, **change))

    def params(v, **change):
        meta = dict({"request_id": 1, "op": "params", "id": "rows", "params": v}, **change)
        return _meta_frame(PG, {k: x for k, x in meta.items() if x is not None})

    def state(*items, **meta):
        return _meta_frame(PS, dict({"programs": list(items)}, **meta))

    run = {"state": "running", "step": 1, "steps_left": 0}
    point = {"point": "a", "kind": "module", "module": "a"}
    active = {"point": "a", "op": "ablate", "state": "active"}

    def controls(**iv):
        return _meta_frame(CS, {"run": run, "params": [], "actions": [],
                                "interventions": dict({"points": [point], "active": []}, **iv)})

    def intervene(v):
        return _meta_frame(C, {"request_id": 1, "op": "intervene", "intervene": v})
    return {
        "program_op": (prog(op="install"), bad),
        "program_request_id": (prog(request_id=None), bad),
        "program_id_long": (prog(id="r" * 56), bad),
        "program_id_upper": (prog(id="Rows"), bad),
        "program_declaration_missing": (prog(declaration=None), bad),
        "program_output_kind": (prog(declaration=out(kind="mesh")), bad),
        "program_max_count": (prog(declaration=out(max_count=1000001)), bad),
        "program_arrays_positions": (prog(declaration=out(arrays=["colors"])), bad),
        "program_arrays_repeat": (prog(declaration=out(arrays=["positions", "positions"])), bad),
        "program_entity": (prog(declaration=out(entity="a/../b")), bad),
        "program_pixels_side": (prog(declaration=dict(PIXELS, output=dict(PIXELS["output"],
                                                                          width=2049))), bad),
        "program_inputs_many": (prog(declaration=decl(inputs=[{"var": f"v{i}",
                                                               "source": "live:x"}
                                                              for i in range(9)])), bad),
        "program_input_source": (prog(declaration=decl(inputs=[{"var": "w",
                                                                "source": "x" * 256}])), bad),
        "program_time_ms": (prog(declaration=decl(limits={"time_ms": 0})), bad),
        "program_every": (prog(declaration=decl(limits={"every": 10001})), bad),
        "program_workgroups": (prog(declaration=decl(workgroups=[64, 1])), bad),
        "program_scene_missing": (prog(scene=None), bad),
        "program_sha256": (prog(sha256=_sha(WGSL).upper()), bad),
        "program_wgsl_missing": (prog(sections={}), bad),
        "program_wgsl_dtype": (prog(sections={"wgsl": src[:64].view(np.int32)}), bad),
        "program_wgsl_empty": (prog(sections={"wgsl": np.zeros(0, np.uint8)}), bad),
        "program_wgsl_big": (prog(sections={"wgsl": np.full(65537, 0x20, np.uint8)}), bad),
        "program_wgsl_utf8": (prog(sections={"wgsl": np.array([0x66, 0x6e, 0xff], np.uint8)}),
                              bad),
        "program_tensor_shape_empty": (prog(declaration=tensor(shape=[])), bad),
        "program_tensor_shape_axes": (prog(declaration=tensor(shape=[1, 1, 1, 1, 1])), bad),
        "program_tensor_shape_zero": (prog(declaration=tensor(shape=[4, 0])), bad),
        "program_tensor_shape_big": (prog(declaration=tensor(shape=[1001, 1000])), bad),
        "program_tensor_shape_float": (prog(declaration=tensor(shape=[4.5])), bad),
        "program_tensor_shape_missing": (prog(declaration=decl(output={"kind": "tensor"})), bad),
        "program_tensor_extra_key": (prog(declaration=tensor(width=4)), bad),
        "program_id_operator_bare": (prog(id="operator.row"), bad),
        "program_id_operator_digits": (prog(id="operator.row.123456"), bad),
        "program_id_operator_nondigit": (prog(id="operator.row.1a"), bad),
        "program_id_operator_spec": (prog(id="operator.Row.1"), bad),
        "program_id_operator_empty_spec": (prog(id="operator..1"), bad),
        "program_params_missing": (params(None), bad),
        "program_params_empty": (params({}), bad),
        "program_params_string": (params({"gain": "1"}), bad),
        "program_params_bool": (params({"gain": True}), bad),
        "program_params_name_digit": (params({"1gain": 1}), bad),
        "program_params_name_reserved": (params({"__gain": 1}), bad),
        "program_params_name_dash": (params({"a-b": 1}), bad),
        "program_params_name_long": (params({"a" * 64: 1}), bad),
        "program_params_many": (params({f"p{i}": 1 for i in range(17)}), bad),
        "program_params_list": (params([1.0]), bad),
        "program_params_declaration": (params({"gain": 1}, declaration=PIXELS), bad),
        "program_params_section": (_meta_frame(PG, {"request_id": 1, "op": "params", "id": "rows",
                                                    "params": {"gain": 1}},
                                               sections={"wgsl": np.frombuffer(WGSL.encode(), np.uint8)}),
                                   bad),
        "program_state_programs": (_meta_frame(PS, {"request_id": 1}), bad),
        "program_state_many": (state(*[dict(item, id=f"p{i}") for i in range(17)]), bad),
        "program_state_repeat": (state(item, item), bad),
        "program_state_state": (state(dict(item, state="paused")), bad),
        "program_state_kind": (state(dict(item, kind="material")), bad),
        "program_state_kind_operator_id": (state(dict(item, id="operator.row")), bad),
        "program_state_runs": (state(dict(item, runs=-1)), bad),
        "program_state_runs_exact": (state(dict(item, runs=2**53)), bad),
        "program_state_every": (state(dict(item, every=0)), bad),
        "program_state_share": (state(dict(item, overhead_share=-0.5)), bad),
        "program_state_rate": (state(dict(item, passes_per_s="2")), bad),
        "program_state_error": (state(dict(item, error={"message": "x"})), bad),
        "program_state_sha256": (state(dict(item, sha256="a" * 63)), bad),
        "program_state_step": (state(item, step=-1), bad),
        "control_intervene_missing": (_meta_frame(C, {"request_id": 1, "op": "intervene"}), bad),
        "control_intervene_two": (intervene({"clear": "a", "clear_all": True}), bad),
        "control_intervene_clear_all": (intervene({"clear_all": False}), bad),
        "control_intervene_point": (intervene({"clear": "p" * 256}), bad),
        "control_intervene_step": (intervene({"set": {"point": "a", "op": "ablate",
                                                      "from_step": -1}}), bad),
        "control_intervene_op": (intervene({"set": {"point": "a"}}), bad),
        "controls_interventions_kind": (controls(points=[dict(point, kind="layer")]), bad),
        "controls_interventions_active": (controls(active=[active] * 65), bad),
        "controls_interventions_shape": (controls(points=[dict(point, shape=[1] * 9)]), bad),
        "controls_interventions_state": (controls(active=[dict(active, state="on")]), bad),
    }


# ---------------------------------------------------------------------------
# Programmable graphics: instances of pieces and their resources
# ---------------------------------------------------------------------------
PIECE_COMMON = ("fn colormap__viridis(t: f32) -> vec3<f32> { return vec3<f32>(t, t * t, 1.0 - t); }\n"
                "fn bar_height(v: f32) -> f32 { return v * params.gain; }\n")
PIECE_DERIVE = ("@compute @workgroup_size(4) fn mean_heads(@builtin(global_invocation_id) id: vec3<u32>) "
                "{ heads[id.x] = f32(id.x); }\n")
PIECE_DRAW = ("@vertex fn bar_vertex(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> "
              "{ return vec4<f32>(0.0, bar_height(heads[v / 6u]), 0.0, 1.0); }\n"
              "@fragment fn bar_fragment() -> @location(0) vec4<f32> "
              "{ return vec4<f32>(colormap__viridis(0.5), 1.0); } // \u00b5\n")
PIECE_MANIFEST = {
    "name": "head_bars", "version": 2, "description": "One bar for the mean attention of each head.",
    "imports": ["colormap@1"],
    "inputs": [{"name": "attn", "kind": "tensor", "rank_min": 3, "rank_max": 4}],
    "params": [{"name": "gain", "default": 1.0, "min": 0.0, "max": 4.0}],
    "resources": [{"id": "heads", "kind": "buffer", "element": "f32", "count": 4}],
    "derive": {"passes": [{"entry": "mean_heads", "workgroups": [1, 1, 1], "writes": ["heads"]}]},
    "draw": {"camera": "pan_zoom", "passes": [{"kind": "render", "vertex": "bar_vertex",
                                              "fragment": "bar_fragment", "instances": "heads",
                                              "vertices": 6}]},
}
PIECE_PIN = {"name": "head_bars", "version": 2, "hash": "sha256:" + _sha("head_bars@2")}
PIECE_SEGMENTS = [{"part": "lib/colormap@1", "first_line": 1, "lines": 1},
                  {"part": "common.wgsl", "first_line": 2, "lines": 1}]


def _instance(request_id: int, derive: bool) -> "w.Program":
    return w.Program(request_id, w.PROGRAM_SET, "heads", params={"gain": 2.0}, piece=PIECE_PIN,
                     manifest=PIECE_MANIFEST, bind={"attn": "live:blocks.0.attn"}, panel="bars",
                     segments=PIECE_SEGMENTS, common=PIECE_COMMON,
                     derive=PIECE_DERIVE if derive else None, draw=PIECE_DRAW)


# The piece form of SCENE_UPDATE: three points whose arrays travel as the resource "points" of the scene
# program "scene.1", an instance of the library piece "points".
SCENE_PIECE_VALUES = _f32([0.25, 0.5, 1.0])
SCENE_PIECE_PIN = {"name": "points", "version": 1}
SCENE_PIECE_KEYS = {"program": "scene.1", "piece": SCENE_PIECE_PIN, "resources": {"points": 3},
                    "arrays": ["positions", "values"], "colormap": w.COLORMAP_MAGMA,
                    "flags": w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX, "vmin": 0.0, "vmax": 1.0,
                    "stats": _stats(SCENE_PIECE_VALUES, CLOUD)}


def _scene_piece_vectors() -> dict:
    # struct Point { position: vec4<f32>, bits: u32, index: u32 } and 8 bytes of padding: 32 bytes.
    rows = np.zeros(3, dtype=[("position", "<f4", 4), ("bits", "<u4"), ("index", "<u4"), ("pad", "V8")])
    rows["position"][:, :3] = CLOUD
    rows["position"][:, 3] = 1.0
    rows["bits"] = SCENE_PIECE_VALUES.view(np.uint32)
    rows["index"] = np.arange(3)
    return {
        "scene_update_points_piece": _update(
            2, "world", 14, STAMP + 1400, "world/cloud", w.PART_POINTS, {},
            {k: SCENE_PIECE_KEYS[k] for k in ("colormap", "flags", "vmin", "vmax", "stats")},
            stamp=STAMP_T, program="scene.1", piece=SCENE_PIECE_PIN, resources={"points": 3},
            names=["positions", "values"]),
        "resource_scene_points": w.Resource("scene.1", "points", 7, 3, 32, rows.tobytes()),
    }


def _scene_piece_invalid() -> dict:
    bad = {"error": w.ERR_BAD_REQUEST}
    key = SCENE_PIECE_KEYS
    stamp = {"stamp": STAMP_T}
    RS = w.MsgType.RESOURCE
    return {
        "scene_update_piece_section": (_frame("points", "p", {"positions": CLOUD, **stamp}, **key), bad),
        "scene_update_piece_wrong": (_frame("points", "p", stamp, **dict(
            key, piece=dict(SCENE_PIECE_PIN, name="lines"))), bad),
        "scene_update_piece_arrays": (_frame("points", "p", stamp, **dict(key, arrays=["values"])), bad),
        "scene_update_piece_split": (_frame("points", "p", stamp, **dict(
            key, split={"update": 1, "piece": 0, "pieces": 2, "rows": {"positions": 3},
                        "first": {"positions": 0}})), bad),
        "resource_scene_id": (_meta_frame(RS, {"program": "scene.0123456789", "resource": "points",
                                               "step": 7, "count": 3, "stride": 32},
                                          {"resource": np.zeros(96, np.uint8)}), bad),
    }


def _resource_vectors() -> dict:
    heads = np.array([0.5, 0.25, 0.125, 0.0625], np.float32)
    # Two elements of struct Arc { from_token: u32, to_token: u32, layer: u32, head: u32,
    # weight: f32, score: f32 }, 24 bytes each.
    arcs = np.zeros(2, dtype=[("from_token", "<u4"), ("to_token", "<u4"), ("layer", "<u4"),
                              ("head", "<u4"), ("weight", "<f4"), ("score", "<f4")])
    arcs[0] = (3, 1, 1, 2, 0.75, 1.5)
    arcs[1] = (5, 2, 1, 2, 0.5, 1.25)
    # A 2 by 2 rgba8unorm texture, packed as one u32 a texel, in row order.
    texels = np.array([0xff0000ff, 0xff00ff00, 0xffff0000, 0x80808080], np.uint32)
    return {
        "program_set_piece": _instance(31, True),
        "program_set_piece_draw": _instance(0, False),
        "program_remove_piece": w.Program(0, w.PROGRAM_REMOVE, "heads"),
        "program_state_piece": w.ProgramState([{"id": "heads", "kind": "piece", "state": "running",
                                                "runs": 4, "last_step": 1200}], step=1200),
        "resource_buffer": w.Resource("heads", "heads", 1200, 4, 4, heads),
        "resource_struct": w.Resource("arcs", "arcs", 1200, 2, 24, arcs.tobytes()),
        "resource_texture": w.Resource("arcs", "glow", 1210, 4, 4, texels, width=2, height=2),
        "resource_empty": w.Resource("arcs", "arcs", 1220, 0, 24, b""),
        # A 3 by 3 r8unorm texture: nine texels, four to a word, three words.
        "resource_texture_r8": w.Resource("arcs", "mask", 1230, 3, 4, bytes(range(1, 10)) + bytes(3),
                                          width=3, height=3, format=w.TEXTURE_R8),
        "resource_texture_r32": w.Resource("arcs", "ids", 1240, 4, 4,
                                           np.array([1, 2, 3, 4000000000], np.uint32),
                                           width=2, height=2, format=w.TEXTURE_R32UINT),
        "resource_texture_wide": w.Resource("arcs", "strip", 1250, 2048, 4,
                                            (np.arange(8192) % 251).astype(np.uint8),
                                            width=8192, height=1, format=w.TEXTURE_R8),
        # Seven u8 elements in two words, five u16 elements in three.
        "resource_narrow_u8": w.Resource("heads", "bytes", 1260, 2, 4, bytes(range(1, 8)) + bytes(1),
                                         element=w.ELEMENT_U8, elements=7),
        "resource_narrow_u16": w.Resource("heads", "halves", 1270, 3, 4,
                                          np.array([1, 2, 3, 4, 5, 0], np.uint16),
                                          element=w.ELEMENT_U16, elements=5),
    }


# 10 segments of two f32 each, (k, -k), 80 bytes: the version that resource_split_0 to _2 carry.
RESOURCE_SPLIT = w.Resource("lines", "segments", 1300, 10, 8,
                            np.array([[k, -k] for k in range(10)], np.float32).tobytes())


def _resource_split_vectors() -> dict:
    """Parts that are valid frames on their own, for the sequences that the receivers' tests play:
    a part 1 of a version whose part 0 never came (dropped), a last part that ends before the count
    (refused after parts 0 and 1), and a part 1 whose step differs from part 0 (refused)."""
    data = memoryview(RESOURCE_SPLIT.data)

    def part(k, first, n, step=1300, version=5):
        return w.Resource("lines", "segments", step, 10, 8, data[first * 8:(first + n) * 8],
                          split={"version": version, "part": k, "parts": 3, "first": first})
    return {
        "resource_split_orphan": part(1, 4, 4, version=6),
        "resource_split_short": part(2, 8, 1),
        "resource_split_differs": part(1, 4, 4, step=1301),
    }


def _resource_invalid() -> dict:
    bad = {"error": w.ERR_BAD_REQUEST}
    PG, RS = w.MsgType.PROGRAM, w.MsgType.RESOURCE
    u8 = lambda text: np.frombuffer(text.encode("utf-8"), np.uint8)  # noqa: E731
    halves = {"common": u8(PIECE_COMMON), "derive": u8(PIECE_DERIVE), "draw": u8(PIECE_DRAW)}
    base = {"request_id": 1, "op": "set", "id": "heads", "piece": PIECE_PIN,
            "manifest": PIECE_MANIFEST, "bind": {"attn": "live:blocks.0.attn"},
            "params": {"gain": 2.0}, "panel": "bars", "segments": PIECE_SEGMENTS}

    def inst(sections=None, **change):
        meta = {k: v for k, v in dict(base, **change).items() if v is not None}
        return _meta_frame(PG, meta, halves if sections is None else sections)

    def pin(**change):
        return dict(PIECE_PIN, **change)

    def seg(**change):
        return [dict(PIECE_SEGMENTS[0], **change)]

    res = {"program": "heads", "resource": "heads", "step": 1200, "count": 4, "stride": 4}
    data = np.arange(16, dtype=np.uint8)

    def resource(section=None, **change):
        meta = {k: v for k, v in dict(res, **change).items() if v is not None}
        return _meta_frame(RS, meta, {"resource": data} if section is None else section)

    texture = {"width": 2, "height": 2}
    return {
        "program_piece_declaration": (inst(declaration=GEOMETRY), bad),
        "program_piece_sha256": (inst(sha256=_sha(WGSL)), bad),
        "program_piece_wgsl": (inst(sections=dict(halves, wgsl=u8(WGSL))), bad),
        "program_piece_pin_list": (inst(piece=["head_bars", 2]), bad),
        "program_piece_name": (inst(piece=pin(name="Head_bars")), bad),
        "program_piece_name_long": (inst(piece=pin(name="h" * 33)), bad),
        "program_piece_name_digit": (inst(piece=pin(name="2heads")), bad),
        "program_piece_version": (inst(piece=pin(version=0)), bad),
        "program_piece_version_big": (inst(piece=pin(version=1000001)), bad),
        "program_piece_hash_prefix": (inst(piece=pin(hash=_sha("head_bars@2"))), bad),
        "program_piece_hash_upper": (inst(piece=pin(hash="sha256:" + _sha("x").upper())), bad),
        "program_piece_manifest_missing": (inst(manifest=None), bad),
        "program_piece_manifest_list": (inst(manifest=[PIECE_MANIFEST]), bad),
        "program_piece_bind_kind": (inst(bind={"attn": "tensor:blocks.0.attn"}), bad),
        "program_piece_bind_tensor_long": (inst(bind={"attn": "live:" + "t" * 256}), bad),
        "program_piece_bind_program": (inst(bind={"arcs": "program:arcs"}), bad),
        "program_piece_bind_resource": (inst(bind={"arcs": "program:arcs/__x"}), bad),
        "program_piece_bind_name": (inst(bind={"1attn": "live:a"}), bad),
        "program_piece_bind_many": (inst(bind={f"i{k}": "live:a" for k in range(9)}), bad),
        "program_piece_params": (inst(params={"gain": "2"}), bad),
        "program_piece_panel": (inst(panel=""), bad),
        "program_piece_segments_many": (inst(segments=PIECE_SEGMENTS * 9), bad),
        "program_piece_segment_line": (inst(segments=seg(first_line=0)), bad),
        "program_piece_segment_part": (inst(segments=seg(part=7)), bad),
        "program_piece_halves": (inst(sections={"common": halves["common"]}), bad),
        "program_piece_draw_empty": (inst(sections={"draw": np.zeros(0, np.uint8)}), bad),
        "program_piece_draw_big": (inst(sections={"draw": np.full(65537, 0x20, np.uint8)}), bad),
        "program_piece_derive_utf8": (inst(sections={"derive": np.array([0x66, 0xff], np.uint8)}),
                                      bad),
        "program_piece_draw_dtype": (inst(sections={"draw": u8(PIECE_DRAW)[:64].view(np.int32)}),
                                     bad),
        "program_piece_common_utf8": (inst(sections={"common": np.array([0xc3], np.uint8),
                                                     "draw": halves["draw"]}), bad),
        "resource_program": (resource(program="Heads"), bad),
        "resource_program_missing": (resource(program=None), bad),
        "resource_name": (resource(resource="__heads"), bad),
        "resource_name_digit": (resource(resource="1heads"), bad),
        "resource_step": (resource(step=-1), bad),
        "resource_step_float": (resource(step=1.5), bad),
        "resource_step_big": (resource(step=2**53), bad),
        "resource_count_missing": (resource(count=None), bad),
        "resource_count_u32": (resource(count=2**32), bad),
        "resource_stride_zero": (resource(stride=0), bad),
        "resource_stride_odd": (resource(stride=6), bad),
        "resource_stride_big": (resource(stride=260), bad),
        "resource_width_alone": (resource(width=4), bad),
        "resource_texture_stride": (resource(count=2, stride=8, **texture), bad),
        "resource_texture_count": (resource(count=3, **texture,
                                            section={"resource": data[:12]}), bad),
        "resource_texture_side": (resource(width=8193, height=1, count=8193,
                                           section={"resource": np.zeros(32772, np.uint8)}), bad),
        "resource_texture_side_big": (resource(width=8193, height=1, count=2049, format="r8unorm",
                                               section={"resource": np.zeros(8196, np.uint8)}), bad),
        "resource_format_unknown": (resource(format="bgra8unorm", **texture), bad),
        "resource_format_alone": (resource(format="r8unorm"), bad),
        "resource_format_count": (resource(count=9, format="r8unorm", width=3, height=3,
                                           section={"resource": np.zeros(36, np.uint8)}), bad),
        "resource_element_unknown": (resource(element="u32", elements=4), bad),
        "resource_element_texture": (resource(element="u8", elements=4, **texture), bad),
        "resource_element_stride": (resource(count=1, stride=8, element="u8", elements=4,
                                             section={"resource": data[:8]}), bad),
        "resource_elements_missing": (resource(element="u8"), bad),
        "resource_elements_alone": (resource(elements=4), bad),
        "resource_elements_count": (resource(count=2, element="u16", elements=5,
                                             section={"resource": data[:8]}), bad),
        # count * stride is 2^32 + 256, which wraps to the section's 256 bytes in 32 bits.
        "resource_count_wrap": (resource(count=16777217, stride=256,
                                         section={"resource": np.zeros(256, np.uint8)}), bad),
        "resource_over_max": (resource(count=w.RESOURCE_MAX_BYTES // 4 + 1), bad),
        "resource_section_missing": (resource(section={}), bad),
        "resource_section_short": (resource(section={"resource": data[:15]}), bad),
        "resource_section_dtype": (resource(section={"resource": data.view(np.float32)}), bad),
        "resource_section_2d": (resource(section={"resource": data.reshape(4, 4)}), bad),
        **_resource_split_invalid(resource, bad),
    }


def _resource_split_invalid(resource, bad) -> dict:
    """Parts of a version in parts that every decoder refuses on their own: the keys of split, the
    bound of a version in parts before any product, and a section of whole elements within the count."""
    sp = {"version": 5, "part": 1, "parts": 3, "first": 4}
    data = np.arange(16, dtype=np.uint8)            # 4 elements of 4 bytes

    def part(section=None, count=10, **change):
        split = {k: v for k, v in dict(sp, **change).items() if v is not None}
        return resource(count=count, split=split, section=section)
    return {
        "resource_split_list": (resource(count=10, split=[5, 1, 3, 4]), bad),
        "resource_split_version": (part(version=-1), bad),
        "resource_split_version_missing": (part(version=None), bad),
        "resource_split_parts_one": (part(parts=1, part=0, first=0), bad),
        "resource_split_parts_many": (part(parts=33), bad),
        "resource_split_part_past": (part(part=3), bad),
        "resource_split_first_missing": (part(first=None), bad),
        "resource_split_first_u32": (part(first=2**32), bad),
        "resource_split_part0_first": (part(part=0), bad),
        "resource_split_first_past": (part(first=10), bad),
        # first near 2^32: first + elements wraps in 32 bits back below the count.
        "resource_split_first_wraps": (part(first=2**32 - 2), bad),
        # count * stride is 2^40 less 256, which wraps in 32 bits; refused by the bound of a version in parts.
        "resource_split_count_wrap": (part(count=2**32 - 1, stride=256,
                                           section={"resource": np.zeros(256, np.uint8)}), bad),
        "resource_split_over_max": (part(count=w.RESOURCE_VERSION_MAX_BYTES // 4 + 1), bad),
        "resource_split_section_empty": (part(section={"resource": np.zeros(0, np.uint8)}), bad),
        "resource_split_section_partial": (part(section={"resource": data[:14]}), bad),
        "resource_split_section_past": (part(first=8, section={"resource": np.zeros(12, np.uint8)}), bad),
    }


def _scene_bundles(messages):
    """Each scene part in the same layout that the Python network sender writes."""
    from types import SimpleNamespace
    from dataclasses import replace
    from caliper import _scene_pieces
    out = {}
    for index, (name, message) in enumerate(list(messages.items()), 100):
        if not isinstance(message, w.SceneUpdate) or not message.arrays or message.program or message.part == "camera":
            continue
        if message.params.get("form") == "encoded":
            continue
        batch = SimpleNamespace(part=message.part, arrays=message.arrays, params=message.params, packed=None)
        piece = _scene_pieces.route(batch)
        names, resources = _scene_pieces.pack(batch, piece)
        program = f"scene.{index}"
        counts = {}
        for rid, count, stride, data, width, height in resources:
            out[name.replace("scene_update_", "resource_scene_", 1) + "_bundle_" + rid] = _scene_pieces.resource_message(piece, program, rid, 7, count, stride, data, width, height)
            counts[rid] = count
        params = dict(message.params, scale=1.0) if piece == "depth" else message.params
        out[name + "_bundle"] = replace(message, arrays={}, params=params, program=program, piece=_scene_pieces.PINS[piece],
                                         resources=counts, names=names)
    return out
