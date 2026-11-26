"""Wire protocol 3.2 scene messages, part decoders, scene view info, and split update assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import sys as _sys
from typing import Any, ClassVar

import numpy as np

if __package__:
    from . import _wire_frame as _f, _wire_programs as _p
else:
    _prefix = __name__.rpartition(".")[0]
    _f = _sys.modules[f"{_prefix}._wire_frame"]
    _p = _sys.modules[f"{_prefix}._wire_programs"]

DEFAULT_MAX_POINTS = _f.DEFAULT_MAX_POINTS
DTYPE_SIZE = _f.DTYPE_SIZE
DType = _f.DType
ERR_BAD_DATA_VIEW = _f.ERR_BAD_DATA_VIEW
Encoding = _f.Encoding
Frame = _f.Frame
MAX_STATS = _f.MAX_STATS
MessageError = _f.MessageError
MsgType = _f.MsgType
NAME_MAX = _f.NAME_MAX
PART_ARROWS = _f.PART_ARROWS
PART_BOXES = _f.PART_BOXES
PART_CAMERA = _f.PART_CAMERA
PART_IMAGE = _f.PART_IMAGE
PART_LINES = _f.PART_LINES
PART_MESH = _f.PART_MESH
PART_POINTS = _f.PART_POINTS
PART_TRANSFORM = _f.PART_TRANSFORM
Section = _f.Section
_Msg = _f._Msg
_ascending = _f._ascending
_bad = _f._bad
_bool = _f._bool
_finite = _f._finite
_is_int = _f._is_int
_is_num = _f._is_num
_json_float = _f._json_float
_name = _f._name
_obj = _f._obj
_opt = _f._opt
_part_section = _f._part_section
_req = _f._req
_section = _f._section
_stamp = _f._stamp
_stat = _f._stat
_str = _f._str
_u32 = _f._u32
_uint = _f._uint
entity_path_ok = _f.entity_path_ok

MAX_INCOMPLETE_UPDATES = _p.MAX_INCOMPLETE_UPDATES
MAX_RESOURCES = _p.MAX_RESOURCES
_library_pin = _p._library_pin
_param_name = _p._param_name
scene_program_id = _p.scene_program_id

# ===========================================================================
# PROTOCOL 3.2 DRAFT: THE SCENES SUBSET
# ===========================================================================
CAP_SCENES = "scenes"
LINES_SEGMENTS, LINES_STRIPS, LINES_EDGES = "segments", "strips", "edges"
COLOR_PER_VERTEX, COLOR_PER_STRIP, COLOR_PER_EDGE = "vertex", "strip", "edge"
FORM_COLOR, FORM_DEPTH, FORM_SEGMENTATION, FORM_ENCODED = "color", "depth", "segmentation", "encoded"
FORMAT_JPEG, FORMAT_PNG = "jpeg", "png"
CONVENTION_RDF, CONVENTION_RUB = "rdf", "rub"
SPACE_3D, SPACE_2D = "3d", "2d"
UNITS_MAX = 15
MAX_ENTITIES = 1024
MAX_BATCHES = 2048
MAX_BATCH_POINTS = 4_194_304
MAX_BATCH_ITEMS = 1_048_576
MAX_LABELS = 4096
MAX_CLASSES = 4096
MAX_CLASS_ID = 65535
MAX_IMAGE_SIDE = 8192
MAX_ENCODED_BYTES = 16_777_216
MAX_TRANSFORMS = 1024
SCENE_DEFAULT_MAX_ITEMS = 100_000
SCENE_LIMIT_MAX_ITEMS = 1_048_576
SCENE_DEFAULT_MAX_VERTICES = 1_000_000
SCENE_LIMIT_MAX_VERTICES = 4_194_304
SCENE_DEFAULT_MAX_EXTENT = 1024
SCENE_LIMIT_MAX_EXTENT = 8192
MAX_PIECES = 16
MAX_UPDATE_BYTES = 536_870_912

PARTS = (PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES, PART_MESH, PART_IMAGE, PART_CAMERA)
FORMS = (FORM_COLOR, FORM_DEPTH, FORM_SEGMENTATION, FORM_ENCODED)
UP_AXES_3D = ("+x", "-x", "+y", "-y", "+z", "-z")
UP_AXES_2D = ("-y", "+y")
# The library piece of each part, and of each form of an image.
SCENE_PIECES = {PART_POINTS: ("points",), PART_LINES: ("lines",), PART_ARROWS: ("arrows",),
                PART_BOXES: ("boxes",), PART_MESH: ("mesh",),
                PART_IMAGE: ("image", "depth", "segmentation")}
FORM_PIECE = {FORM_COLOR: "image", FORM_DEPTH: "depth", FORM_SEGMENTATION: "segmentation"}
# The array sections of each part, in the order a sender writes them; the
# section "stamp" follows them.
PART_SECTIONS = {
    PART_POINTS: ("positions", "colors", "values", "class_ids", "radii"),
    PART_LINES: ("positions", "strip_lengths", "indices", "colors", "values"),
    PART_ARROWS: ("origins", "vectors", "colors", "values"),
    PART_BOXES: ("centers", "half_sizes", "rotations", "colors", "class_ids"),
    PART_MESH: ("positions", "indices", "normals", "colors", "values", "uvs", "texture"),
    PART_IMAGE: ("pixels",),
    PART_CAMERA: (),
}
# The keys of each part (SCENE_UPDATE metadata beside the common ones), in the
# order their types are checked.
PART_KEYS = {
    PART_POINTS: ("color", "radius", "labels", "colormap", "flags", "vmin", "vmax", "stats"),
    PART_LINES: ("mode", "color_per", "color", "width", "colormap", "flags", "vmin", "vmax",
                 "stats"),
    PART_ARROWS: ("color", "width", "colormap", "flags", "vmin", "vmax", "stats"),
    PART_BOXES: ("color", "labels", "fill", "width", "stats"),
    PART_MESH: ("color", "colormap", "flags", "vmin", "vmax", "stats"),
    PART_IMAGE: ("form", "opacity", "scale", "cloud", "colormap", "flags", "vmin", "vmax",
                 "format", "stats"),
    PART_CAMERA: ("fx", "fy", "cx", "cy", "width", "height", "convention", "plane_distance"),
}
# The keys a sender writes as floats (a JSON number with a fraction or ".0").
_FLOAT_KEYS = ("vmin", "vmax", "radius", "width", "opacity", "scale", "fx", "fy", "cx", "cy",
               "plane_distance")
CAMERA_REQUIRED = ("fx", "fy", "cx", "cy", "width", "height", "convention")
CLEAR_PARTS = PARTS + (PART_TRANSFORM,)
SCENE_DATA_VIEW = {"entities": {"include": ["**"], "exclude": []},
                   "budget": {"max_points": DEFAULT_MAX_POINTS,
                              "max_items": SCENE_DEFAULT_MAX_ITEMS,
                              "max_vertices": SCENE_DEFAULT_MAX_VERTICES,
                              "max_extent": SCENE_DEFAULT_MAX_EXTENT}}
VIEW_CAMERA_KEYS = ("target", "distance", "azimuth", "elevation")


def _rgba(v) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(_is_int(x) and 0 <= x <= 255 for x in v)


def _positive(v) -> bool:
    return _finite(v) and v > 0


def _stats_list(v) -> bool:
    return isinstance(v, list) and len(v) <= MAX_STATS and all(_stat(x) for x in v)


def _labels(v) -> bool:
    return (isinstance(v, list) and 1 <= len(v) <= MAX_LABELS
            and all(isinstance(x, str) and len(x.encode("utf-8")) <= NAME_MAX for x in v))


def _side(v) -> bool:
    return _is_int(v) and 1 <= v <= MAX_IMAGE_SIDE


_KEY_CHECKS = {
    "color": (_rgba, "[r, g, b, a] of integers from 0 to 255"),
    "radius": (_positive, "a positive number"),
    "width": (_positive, "a positive number"),
    "labels": (_labels, f"a list of 1 to {MAX_LABELS} strings of at most {NAME_MAX} bytes"),
    "colormap": (_is_int, "an integer"),
    "flags": (_uint, "an unsigned integer"),
    "vmin": (_is_num, "a number"),
    "vmax": (_is_num, "a number"),
    "stats": (_stats_list, f"at most {MAX_STATS} numbers or nulls"),
    "mode": (lambda v: v in (LINES_SEGMENTS, LINES_STRIPS, LINES_EDGES), "segments, strips or edges"),
    "color_per": (lambda v: v in (COLOR_PER_VERTEX, COLOR_PER_STRIP, COLOR_PER_EDGE),
                  "vertex, strip or edge"),
    "fill": (_bool, "a boolean"),
    "form": (lambda v: v in FORMS, "color, depth, segmentation or encoded"),
    "opacity": (lambda v: _finite(v) and 0 <= v <= 1, "a number from 0 to 1"),
    "scale": (_positive, "a positive number"),
    "cloud": (_bool, "a boolean"),
    "format": (lambda v: v in (FORMAT_JPEG, FORMAT_PNG), "jpeg or png"),
    "fx": (_positive, "a positive number"),
    "fy": (_positive, "a positive number"),
    "cx": (_finite, "a finite number"),
    "cy": (_finite, "a finite number"),
    "height": (_side, f"an integer from 1 to {MAX_IMAGE_SIDE}"),
    "convention": (lambda v: v in (CONVENTION_RDF, CONVENTION_RUB), "rdf or rub"),
    "plane_distance": (_positive, "a positive number"),
}
_CAMERA_WIDTH = (_side, f"an integer from 1 to {MAX_IMAGE_SIDE}")


def _required(f: Frame, part: str, name: str) -> None:
    if f.section(name) is None:
        _bad(f"a {part} part needs a {name!r} section")


def _rows_section(f: Frame, part: str, name: str, limit: int, what: str, dims=None):
    """A required f32 (N, D) section of a part, D 2 or 3 (or ``dims``); (section, N, D)."""
    _required(f, part, name)
    s = _part_section(f, name, DType.F32, (None, dims))
    n, d = s.shape
    if dims is None and d not in (2, 3):
        _bad(f"{name!r} is (N, 2) or (N, 3), got {list(s.shape)}")
    if n > limit:
        _bad(f"{n} {what}, more than {limit} in a batch")
    return s, n, d


def _colours(f: Frame, rows: int, arrays: dict) -> None:
    c = _part_section(f, "colors", DType.U8, (rows, 4))
    v = _part_section(f, "values", DType.F32, (rows,))
    if c is not None and v is not None:
        _bad("'colors' and 'values' together")
    if c is not None:
        arrays["colors"] = c
    if v is not None:
        arrays["values"] = v


def _label_count(params: dict, n: int) -> None:
    labels = params.get("labels")
    if labels is not None and len(labels) not in (1, n):
        _bad(f"{len(labels)} labels for {n} items: give one label for all or one per item")


def _class_ids(f: Frame, n: int, arrays: dict) -> None:
    s = _part_section(f, "class_ids", DType.I32, (n,))
    if s is not None:
        arrays["class_ids"] = s


def _decode_points(f: Frame, params: dict) -> dict:
    pos, n, _ = _rows_section(f, PART_POINTS, "positions", MAX_BATCH_POINTS, "positions")
    arrays = {"positions": pos}
    _colours(f, n, arrays)
    _class_ids(f, n, arrays)
    r = _part_section(f, "radii", DType.F32, (n,))
    if r is not None:
        arrays["radii"] = r
    _label_count(params, n)
    return arrays


def _decode_lines(f: Frame, params: dict) -> dict:
    mode = params.get("mode")
    if mode is None:
        _bad("a lines part needs 'mode'")
    color_per = params.get("color_per")
    pos, v, _ = _rows_section(f, PART_LINES, "positions", MAX_BATCH_POINTS, "positions")
    arrays = {"positions": pos}
    if mode != LINES_EDGES and f.section("indices") is not None:
        _bad(f"lines of mode {mode!r} take no 'indices'")
    rows = v
    if mode == LINES_SEGMENTS:
        if v % 2:
            _bad(f"segments need an even number of vertices, got {v}")
        if f.section("strip_lengths") is not None:
            _bad("segments take no 'strip_lengths'")
        if color_per not in (None, COLOR_PER_VERTEX):
            _bad(f"segments take color_per 'vertex', got {color_per!r}")
    elif mode == LINES_STRIPS:
        sl = _part_section(f, "strip_lengths", DType.I32, (None,))
        if sl is None:
            _bad("strips need a 'strip_lengths' section")
        lens = sl.array()
        if lens.shape[0] > MAX_BATCH_ITEMS:
            _bad(f"{lens.shape[0]} strips, more than {MAX_BATCH_ITEMS} in a batch")
        if lens.size and int(lens.min()) < 1:
            _bad("every strip length is at least 1")
        total = int(lens.sum(dtype=np.int64))
        if total != v:
            _bad(f"the strip lengths add up to {total}, not to the {v} vertices")
        arrays["strip_lengths"] = sl
        if color_per == COLOR_PER_EDGE:
            _bad("strips take color_per 'vertex' or 'strip'")
        if color_per == COLOR_PER_STRIP:
            rows = lens.shape[0]
    else:
        if f.section("strip_lengths") is not None:
            _bad("edges take no 'strip_lengths'")
        idx = _part_section(f, "indices", DType.I32, (None, 2))
        if idx is None:
            _bad("edges need an 'indices' section, (E, 2) i32")
        if idx.shape[0] > MAX_BATCH_POINTS:
            _bad(f"{idx.shape[0]} edges, more than {MAX_BATCH_POINTS} in a batch")
        arrays["indices"] = idx
        if color_per == COLOR_PER_STRIP:
            _bad("edges take color_per 'vertex' or 'edge'")
        if color_per == COLOR_PER_EDGE:
            rows = idx.shape[0]
    _colours(f, rows, arrays)
    return arrays


def _decode_arrows(f: Frame, params: dict) -> dict:
    o, n, d = _rows_section(f, PART_ARROWS, "origins", MAX_BATCH_ITEMS, "arrows")
    _required(f, PART_ARROWS, "vectors")
    vec = _part_section(f, "vectors", DType.F32, (n, d))
    arrays = {"origins": o, "vectors": vec}
    _colours(f, n, arrays)
    return arrays


def _decode_boxes(f: Frame, params: dict) -> dict:
    c, n, d = _rows_section(f, PART_BOXES, "centers", MAX_BATCH_ITEMS, "boxes")
    _required(f, PART_BOXES, "half_sizes")
    h = _part_section(f, "half_sizes", DType.F32, (n, d))
    arrays = {"centers": c, "half_sizes": h}
    r = _part_section(f, "rotations", DType.F32, (n, 4))
    if r is not None:
        if d != 3:
            _bad("'rotations' are for 3-D boxes only")
        arrays["rotations"] = r
    col = _part_section(f, "colors", DType.U8, (n, 4))
    if col is not None:
        arrays["colors"] = col
    _class_ids(f, n, arrays)
    _label_count(params, n)
    return arrays


def _decode_mesh(f: Frame, params: dict) -> dict:
    pos, v, _ = _rows_section(f, PART_MESH, "positions", MAX_BATCH_POINTS, "positions")
    arrays = {"positions": pos}
    idx = _part_section(f, "indices", DType.I32, (None, 3))
    if idx is not None:
        if idx.shape[0] > MAX_BATCH_POINTS:
            _bad(f"{idx.shape[0]} triangles, more than {MAX_BATCH_POINTS} in a batch")
        arrays["indices"] = idx
    elif v % 3:
        _bad(f"a mesh without 'indices' needs a multiple of 3 vertices, got {v}")
    nrm = _part_section(f, "normals", DType.F32, (v, 3))
    if nrm is not None:
        arrays["normals"] = nrm
    _colours(f, v, arrays)
    uv = _part_section(f, "uvs", DType.F32, (v, 2))
    tex = _part_section(f, "texture", DType.U8, (None, None, 4))
    if uv is not None and ("colors" in arrays or "values" in arrays):
        _bad("'uvs' and 'colors' or 'values' together")
    if (uv is None) != (tex is None):
        _bad("'uvs' and 'texture' go together")
    if tex is not None:
        if not (1 <= tex.shape[0] <= MAX_IMAGE_SIDE and 1 <= tex.shape[1] <= MAX_IMAGE_SIDE):
            _bad(f"a texture is 1 to {MAX_IMAGE_SIDE} pixels on a side, got {list(tex.shape)}")
        arrays["uvs"] = uv
        arrays["texture"] = tex
    return arrays


# The pixels each image form takes: (dtype, rank) pairs.
_FORM_PIXELS = {
    FORM_COLOR: ((DType.U8, 3), (DType.U8, 2)),
    FORM_DEPTH: ((DType.F32, 2), (DType.U16, 2)),
    FORM_SEGMENTATION: ((DType.U16, 2), (DType.U8, 2)),
    FORM_ENCODED: ((DType.U8, 1),),
}


def _decode_image(f: Frame, params: dict) -> dict:
    form = params.get("form")
    if form is None:
        _bad("an image part needs 'form'")
    _required(f, PART_IMAGE, "pixels")
    p = f.section("pixels")
    if p.encoding != Encoding.RAW or (p.dtype, len(p.shape)) not in _FORM_PIXELS[form]:
        _bad(f"the pixels of a {form} image cannot be dtype {p.dtype} shape {list(p.shape)}")
    if form == FORM_COLOR and len(p.shape) == 3 and p.shape[2] != 4:
        _bad(f"colour pixels are (H, W, 4) or (H, W), got {list(p.shape)}")
    if form == FORM_ENCODED:
        if p.shape[0] > MAX_ENCODED_BYTES:
            _bad(f"an encoded frame of {p.shape[0]} bytes, more than {MAX_ENCODED_BYTES}")
        if params.get("format") is None:
            _bad("an encoded image needs 'format'")
    elif p.shape[0] > MAX_IMAGE_SIDE or p.shape[1] > MAX_IMAGE_SIDE:
        _bad(f"an image is at most {MAX_IMAGE_SIDE} pixels on a side, got {list(p.shape)}")
    return {"pixels": p}


def _decode_camera(f: Frame, params: dict) -> dict:
    for key in CAMERA_REQUIRED:
        if key not in params:
            _bad(f"a camera part needs {key!r}")
    return {}


_DECODERS = {PART_POINTS: _decode_points, PART_LINES: _decode_lines,
             PART_ARROWS: _decode_arrows, PART_BOXES: _decode_boxes, PART_MESH: _decode_mesh,
             PART_IMAGE: _decode_image, PART_CAMERA: _decode_camera}


def _part_keys(m: dict, part: str) -> dict:
    out = {}
    for key in PART_KEYS[part]:
        v = m.get(key)
        if v is None:
            continue
        check, what = _CAMERA_WIDTH if (part == PART_CAMERA and key == "width") else \
            _KEY_CHECKS[key]
        if not check(v):
            _bad(f"{key!r} must be {what}, got {v!r}")
        out[key] = v
    return out


def _common(m: dict) -> tuple:
    """The keys every scene message starts with: sub, view, seq, stamp_ns."""
    return (_req(m, "sub", _u32, "a uint32"), _req(m, "view", _name, "a view name"),
            _req(m, "seq", _uint, "an unsigned integer"),
            _req(m, "stamp_ns", _uint, "an unsigned integer"))


def _flags(m: dict) -> tuple:
    """static, state and cursor, in that order."""
    return (_opt(m, "static", _bool, "a boolean", False),
            _opt(m, "state", _bool, "a boolean", False),
            _opt(m, "cursor", _u32, "a uint32"))


def _write_flags(meta: dict, msg) -> None:
    if msg.static:
        meta["static"] = True
    if msg.state:
        meta["state"] = True
    if msg.cursor is not None:
        meta["cursor"] = msg.cursor


@dataclass
class SceneUpdate(_Msg):
    """3.2 draft: one part of one entity of a scene. ``arrays`` maps section
    names to arrays (numpy, or :class:`Section` when decoded), written in the
    part's order (PART_SECTIONS); ``params`` holds the part's keys
    (PART_KEYS), ``stats`` entries may be None (a non-finite value). ``stamp``
    is the i64 (T,) stamp section; ``cursor`` the generation of the
    TIME_STATE a recorded message answers; ``split`` the key of a piece of a
    split update, whose ``arrays`` then hold the piece's rows, checked against
    the part only once :class:`SplitUpdates` has put the update together, and
    whose ``params`` are the part's keys in piece 0 and empty in the others.

    The piece form (capability "scene_pieces") has ``program``, the scene program
    (``scene.<n>``) whose RESOURCE versions came just before it and hold the batch's
    arrays as the resources of the library piece ``piece`` (the pin: name, version and
    hash); ``resources`` maps each resource id to the count of that version, and
    ``names`` lists the part's arrays that the batch held (the key "arrays" on the
    wire). It carries no array section and no ``split``, thus ``arrays`` is empty."""
    TYPE: ClassVar[MsgType] = MsgType.SCENE_UPDATE
    sub: int
    view: str
    seq: int
    stamp_ns: int
    path: str
    part: str
    arrays: dict
    params: dict = field(default_factory=dict)
    static: bool = False
    state: bool = False
    axes: list | None = None
    items: str | None = None
    stamp: Any = None
    cursor: int | None = None
    split: dict | None = None
    program: str | None = None
    piece: dict | None = None
    resources: dict | None = None
    names: list | None = None

    def to_meta(self):
        meta = {"sub": self.sub, "view": self.view, "seq": self.seq, "stamp_ns": self.stamp_ns,
                "path": self.path, "part": self.part}
        if self.program is not None:
            meta.update(program=self.program, piece=dict(self.piece),
                        resources={k: int(v) for k, v in self.resources.items()},
                        arrays=list(self.names))
        camera = self.part == PART_CAMERA
        for key, v in self.params.items():
            if v is None:
                continue
            if key == "stats":
                v = [_json_float(x) for x in v]
            elif camera and key in ("width", "height"):
                v = int(v)
            elif key in _FLOAT_KEYS:
                v = float(v)
            elif key == "color":
                v = [int(x) for x in v]
            elif key == "labels":
                v = list(v)
            meta[key] = v
        _write_flags(meta, self)
        if self.axes is not None:
            meta["axes"] = list(self.axes)
        if self.items is not None:
            meta["items"] = self.items
        if self.split is not None:
            meta["split"] = self.split
        return meta

    def to_sections(self):
        if self.arrays or self.split is not None:
            raise ValueError("per-shape scene updates are read-only recording data")
        return [] if self.stamp is None else [_section("stamp", self.stamp)]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        sub, view, seq, stamp_ns = _common(m)
        path = _req(m, "path", entity_path_ok, "an entity path")
        part = _req(m, "part", _str, "a string")
        if part not in PARTS:
            _bad(f"unknown part {part!r}")
        static, state, cursor = _flags(m)
        axes = _opt(m, "axes", lambda v: isinstance(v, list), "a list")
        items = _opt(m, "items", _name, "an item space name")
        split = _opt(m, "split", _obj, "an object")
        program = piece = resources = names = None
        if "program" in m:
            program, piece, resources, names = _piece_form(m, part)
            if split is not None:
                _bad("an update in the piece form is never split")
            params = _part_keys(m, part)
            if part == PART_IMAGE and FORM_PIECE.get(params.get("form")) != piece["name"]:
                _bad(f"an image of the form {params.get('form')!r} has no piece {piece['name']!r}")
            for name in PART_SECTIONS[part]:
                if f.section(name) is not None:
                    _bad(f"an update in the piece form carries no array section, got {name!r}")
            arrays = {}
        elif split is not None:
            params, arrays = _piece(f, m, part, split)
        else:
            params = _part_keys(m, part)
            arrays = _DECODERS[part](f, params)
        stamp, _ = _stamp(f, static)
        if "stats" in params:
            params["stats"] = [math.nan if x is None else float(x) for x in params["stats"]]
        return cls(sub, view, seq, stamp_ns, path, part, arrays, params, static, state, axes,
                   items, stamp, cursor, split, program, piece, resources, names)


def _piece_form(m: dict, part: str) -> tuple:
    """The keys of the piece form of SCENE_UPDATE, checked in this order: ``program``,
    ``piece``, ``resources``, ``arrays``."""
    program = _req(m, "program", scene_program_id, "a scene program id scene.<n> of 1 to 9 digits")
    piece = _req(m, "piece", _library_pin, "a library pin {name, version} with an optional hash")
    if piece["name"] not in SCENE_PIECES.get(part, ()):
        _bad(f"the part {part!r} has no library piece {piece['name']!r}")
    resources = _req(m, "resources", lambda v: _obj(v) and 1 <= len(v) <= MAX_RESOURCES and all(
        _param_name(k) and _u32(n) for k, n in v.items()),
        f"an object of 1 to {MAX_RESOURCES} resource ids, each with a uint32 count")
    table = PART_SECTIONS[part]
    names = _req(m, "arrays", lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
                 "a list of array names")
    if not names or names[0] != table[0] or any(x not in table for x in names) or \
            [table.index(x) for x in names] != sorted({table.index(x) for x in names}):
        _bad(f"'arrays' must list the arrays of {part!r} in their order from {table[0]!r}, "
             f"each once, got {names!r}")
    return program, dict(piece), dict(resources), list(names)


@dataclass
class SceneTransforms(_Msg):
    """3.2 draft: the entity transforms of up to MAX_TRANSFORMS entities of a
    scene, each a 4 x 4 matrix stored column-major (row k of ``matrices``,
    (K, 16) f32, belongs to ``paths[k]``), with one ``stamp`` (T,) for all or
    ``stamps`` (K, T), one each."""
    TYPE: ClassVar[MsgType] = MsgType.SCENE_TRANSFORMS
    sub: int
    view: str
    seq: int
    stamp_ns: int
    paths: list
    matrices: Any
    static: bool = False
    state: bool = False
    stamp: Any = None
    stamps: Any = None
    cursor: int | None = None

    def to_meta(self):
        meta = {"sub": self.sub, "view": self.view, "seq": self.seq, "stamp_ns": self.stamp_ns,
                "paths": list(self.paths)}
        _write_flags(meta, self)
        return meta

    def to_sections(self):
        out = [_section("matrices", np.asarray(self.matrices, dtype=np.float32)
                        if not isinstance(self.matrices, Section) else self.matrices)]
        if self.stamps is not None:
            out.append(_section("stamps", self.stamps))
        elif self.stamp is not None:
            out.append(_section("stamp", self.stamp))
        return out

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        sub, view, seq, stamp_ns = _common(m)
        static, state, cursor = _flags(m)
        paths = _req(m, "paths", lambda v: isinstance(v, list) and 1 <= len(v) <= MAX_TRANSFORMS
                     and all(isinstance(p, str) and p != "" and entity_path_ok(p) for p in v),
                     f"a list of 1 to {MAX_TRANSFORMS} entity paths other than the root")
        if len(set(paths)) != len(paths):
            _bad("'paths' names an entity twice")
        if f.section("matrices") is None:
            _bad("a scene_transforms needs a 'matrices' section")
        mat = _part_section(f, "matrices", DType.F32, (len(paths), 16))
        stamp, stamps = _stamp(f, static, len(paths))
        return cls(sub, view, seq, stamp_ns, paths, mat, static, state, stamp, stamps, cursor)


@dataclass
class SceneClear(_Msg):
    """3.2 draft: clear entity ``path`` (its subtree too with ``recursive``),
    only ``parts`` when given; ``reset`` empties the whole scene, static items
    included, and opens a scene's state for a new subscription (``state``)."""
    TYPE: ClassVar[MsgType] = MsgType.SCENE_CLEAR
    sub: int
    view: str
    seq: int
    stamp_ns: int
    path: str = ""
    recursive: bool = False
    parts: list | None = None
    reset: bool = False
    static: bool = False
    state: bool = False
    stamp: Any = None
    cursor: int | None = None

    def to_meta(self):
        meta = {"sub": self.sub, "view": self.view, "seq": self.seq, "stamp_ns": self.stamp_ns,
                "path": self.path}
        for key in ("recursive", "reset"):
            if getattr(self, key):
                meta[key] = True
        _write_flags(meta, self)
        if self.parts is not None:
            meta["parts"] = list(self.parts)
        return meta

    def to_sections(self):
        return [] if self.stamp is None else [_section("stamp", self.stamp)]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        sub, view, seq, stamp_ns = _common(m)
        path = _req(m, "path", entity_path_ok, "an entity path")
        recursive = _opt(m, "recursive", _bool, "a boolean", False)
        parts = _opt(m, "parts", lambda v: isinstance(v, list) and len(v) > 0
                     and all(x in CLEAR_PARTS for x in v), "a list of parts")
        reset = _opt(m, "reset", _bool, "a boolean", False)
        static, state, cursor = _flags(m)
        if reset and path != "":
            _bad("a reset clears the root: its 'path' is \"\"")
        stamp, _ = _stamp(f, static)
        return cls(sub, view, seq, stamp_ns, path, recursive, parts, reset, static, state, stamp,
                   cursor)


def _class_entry(e) -> bool:
    return (_obj(e) and _is_int(e.get("id")) and 0 <= e["id"] <= MAX_CLASS_ID
            and isinstance(e.get("label"), str) and len(e["label"].encode("utf-8")) <= NAME_MAX
            and _rgba(e.get("color")))


@dataclass
class SceneClasses(_Msg):
    """3.2 draft: a scene's class table, which replaces the previous one
    whole: ``classes`` is a list of ``{"id", "label", "color"}`` in ascending
    id order."""
    TYPE: ClassVar[MsgType] = MsgType.SCENE_CLASSES
    sub: int
    view: str
    seq: int
    stamp_ns: int
    classes: list
    static: bool = False
    state: bool = False
    stamp: Any = None
    cursor: int | None = None

    def to_meta(self):
        meta = {"sub": self.sub, "view": self.view, "seq": self.seq, "stamp_ns": self.stamp_ns,
                "classes": list(self.classes)}
        _write_flags(meta, self)
        return meta

    def to_sections(self):
        return [] if self.stamp is None else [_section("stamp", self.stamp)]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        sub, view, seq, stamp_ns = _common(m)
        static, state, cursor = _flags(m)
        classes = _req(m, "classes", lambda v: isinstance(v, list) and len(v) <= MAX_CLASSES
                       and all(_class_entry(e) for e in v),
                       f"a list of at most {MAX_CLASSES} {{id, label, color}} entries")
        if not _ascending([e["id"] for e in classes]):
            _bad("'classes' must be in ascending id order without repeats")
        stamp, _ = _stamp(f, static)
        return cls(sub, view, seq, stamp_ns, classes, static, state, stamp, cursor)


def _view_camera(sc: dict, space: str):
    """A VIEWS scene's "view_camera" as a viewer reads it: the dict of the
    orbit keys or the entity path of a camera, or None for none, for a 2-D
    scene, and for a value it cannot read (it then opens on the fit)."""
    vc = sc.get("view_camera")
    if vc is None or space != SPACE_3D:
        return None
    if isinstance(vc, str):
        return vc if vc != "" and entity_path_ok(vc) else None
    if not _obj(vc) or not set(vc) <= set(VIEW_CAMERA_KEYS):
        return None
    t = vc.get("target")
    if t is not None and not (isinstance(t, list) and len(t) == 3 and all(_finite(x) for x in t)):
        return None
    if vc.get("distance") is not None and not _positive(vc["distance"]):
        return None
    if vc.get("azimuth") is not None and not _finite(vc["azimuth"]):
        return None
    e = vc.get("elevation")
    if e is not None and not (_finite(e) and -89 <= e <= 89):
        return None
    return {k: v for k, v in vc.items() if v is not None}


def scene_view_info(entry: dict) -> dict:
    """The key "scene" of a VIEWS entry of kind scene, checked, with its
    defaults filled in: {"space", "up", "units", "view_camera"}, the last None
    when absent or unreadable. Raises :class:`MessageError` (bad_request)."""
    sc = entry.get("scene")
    if sc is None:
        sc = {}
    if not _obj(sc):
        _bad(f"'scene' must be an object, got {sc!r}")
    space = sc.get("space")                     # a null key means the same as an absent one
    if space is None:
        space = SPACE_3D
    if space not in (SPACE_3D, SPACE_2D):
        _bad(f"scene space {space!r}; expected '3d' or '2d'")
    up = sc.get("up")
    axes = UP_AXES_3D if space == SPACE_3D else UP_AXES_2D
    if up is None:
        up = "+y" if space == SPACE_3D else "-y"
    if up not in axes:
        _bad(f"scene up {up!r}; a {space} scene takes one of {list(axes)}")
    units = sc.get("units")
    if units is None:
        units = "m"
    if not isinstance(units, str) or len(units.encode("utf-8")) > UNITS_MAX:
        _bad(f"scene units must be a string of at most {UNITS_MAX} bytes, got {units!r}")
    return {"space": space, "up": up, "units": units, "view_camera": _view_camera(sc, space)}


def resolve_scene_data_view(view: dict, request: dict | None) -> dict:
    """The resolved data view of a scene subscription. Only the default data
    view is built in this subset: a request that is absent, {}, or the
    resolved default itself. Raises :class:`MessageError` (bad_data_view)."""
    if request is not None and not _obj(request):
        raise MessageError(ERR_BAD_DATA_VIEW, "a data view is a JSON object")
    if view.get("default") not in (None, {}, SCENE_DATA_VIEW):
        raise MessageError(ERR_BAD_DATA_VIEW, "a scene's own default data view is not built yet")
    if request not in (None, {}, SCENE_DATA_VIEW):
        raise MessageError(ERR_BAD_DATA_VIEW,
                           "only the default data view of a scene is built yet: send {} or none; "
                           "entity filters and budgets come in a later increment")
    return json.loads(json.dumps(SCENE_DATA_VIEW))


# ---------------------------------------------------------------------------
# 3.2 draft: split updates
# ---------------------------------------------------------------------------
def _piece(f: Frame, m: dict, part: str, split: dict):
    """The checks of a piece of a split update, in this order: the keys of
    ``split`` (update, pieces, piece, first, then rows in piece 0 only), the
    part's keys in piece 0, and that ``first`` names exactly the piece's array
    sections, each a raw array of at least one dimension, whose rows in piece
    0 stay within ``rows``. The arrays are checked against the part once the
    whole update is put together (:class:`SplitUpdates`)."""
    def need(key, ok, what):
        v = split.get(key)
        if v is None or not ok(v):
            _bad(f"'split.{key}' must be {what}, got {v!r}")
        return v
    rows_map = lambda v: _obj(v) and all(_uint(x) for x in v.values())        # noqa: E731
    need("update", _uint, "an unsigned integer")
    pieces = need("pieces", lambda v: _is_int(v) and 2 <= v <= MAX_PIECES,
                  f"an integer from 2 to {MAX_PIECES}")
    piece = need("piece", lambda v: _is_int(v) and 0 <= v < pieces, "a piece number below 'pieces'")
    first = need("first", rows_map, "an object of row numbers")
    rows = None
    if piece == 0:
        rows = need("rows", rows_map, "an object of row counts")
    elif split.get("rows") is not None:
        _bad("only piece 0 carries 'split.rows'")
    params = _part_keys(m, part) if piece == 0 else {}
    arrays = {}
    for s in f.sections:
        if s.name == "stamp" or s.name not in PART_SECTIONS[part]:
            continue                               # the stamp, or a section of no array: ignored
        if s.encoding != Encoding.RAW or len(s.shape) < 1:
            _bad(f"a piece of {s.name!r} is a raw array of at least one dimension")
        arrays[s.name] = s
    if set(first) != set(arrays):
        _bad(f"'split.first' names {sorted(first)}, the piece holds {sorted(arrays)}")
    if rows is not None:
        for name in rows:
            if name not in PART_SECTIONS[part]:
                _bad(f"'split.rows' names {name!r}, not an array of a {part} part")
        for name, s in arrays.items():
            if name not in rows or first[name] + s.shape[0] > rows[name]:
                _bad(f"the rows of {name!r} in piece 0 pass 'split.rows'")
    return params, arrays


class _Assembly:
    """One split update being put together."""

    def __init__(self, p0: SceneUpdate):
        self.p0 = p0
        self.key = (p0.path, p0.part, p0.seq, p0.split["pieces"],
                    None if p0.stamp is None else bytes(p0.stamp.data))
        self.rows = dict(p0.split["rows"])
        self.seen: set = set()
        self.arrays: dict = {}                    # name -> [dtype, trailing shape, buffer, ranges]
        self.nbytes = 0

    def add(self, p: SceneUpdate) -> None:
        sp = p.split
        key = (p.path, p.part, p.seq, sp["pieces"], None if p.stamp is None else bytes(p.stamp.data))
        if key != self.key:
            _bad("a piece's path, part, seq, pieces or stamp differ from its update's piece 0")
        if sp["piece"] in self.seen:
            _bad(f"piece {sp['piece']} of update {sp['update']} arrived twice")
        self.seen.add(sp["piece"])
        for name, s in p.arrays.items():
            if name not in self.rows:
                _bad(f"piece {sp['piece']} holds {name!r}, which 'split.rows' does not name")
            item = DTYPE_SIZE.get(s.dtype)
            if item is None:
                _bad(f"a piece of {name!r} has the unknown dtype {s.dtype}")
            trailing = tuple(s.shape[1:])
            rowbytes = math.prod(trailing) * item
            a = self.arrays.get(name)
            if a is None:
                total = self.rows[name] * rowbytes
                if self.nbytes + total > MAX_UPDATE_BYTES:
                    _bad(f"a split update of more than {MAX_UPDATE_BYTES} bytes")
                self.nbytes += total
                a = self.arrays[name] = [s.dtype, trailing, bytearray(total), []]
            elif (a[0], a[1]) != (s.dtype, trailing):
                _bad(f"the pieces of {name!r} differ in dtype or shape")
            first, n = sp["first"][name], s.shape[0]
            if first + n > self.rows[name]:
                _bad(f"rows {first} to {first + n} of {name!r} pass its {self.rows[name]} rows")
            if any(first < f0 + n0 and f0 < first + n for f0, n0 in a[3]):
                _bad(f"two pieces hold the same rows of {name!r}")
            a[3].append((first, n))
            a[2][first * rowbytes:(first + n) * rowbytes] = s.data

    def done(self) -> bool:
        if len(self.seen) < self.key[3]:
            return False
        for name, rows in self.rows.items():
            a = self.arrays.get(name)
            got = sum(n for _, n in a[3]) if a is not None else 0
            if got != rows:
                _bad(f"the pieces ended before every row of {name!r} arrived")
        return True

    def whole(self) -> SceneUpdate:
        p0 = self.p0
        meta = p0.to_meta()
        meta.pop("split")
        sections = []
        for name in PART_SECTIONS[p0.part]:
            a = self.arrays.get(name)
            if a is not None:
                sections.append(Section(name, a[0], (self.rows[name],) + a[1],
                                        memoryview(a[2]).cast("B") if a[2] else memoryview(b"")))
        if p0.stamp is not None:
            sections.append(Section("stamp", DType.I64, tuple(p0.stamp.shape),
                                    memoryview(bytes(p0.stamp.data))))
        return SceneUpdate.from_frame(Frame(MsgType.SCENE_UPDATE, meta, sections))


class SplitUpdates:
    """A receiver's split updates of one subscription. :meth:`add` takes each
    piece and returns the whole update, checked as an unsplit one, once every
    piece and every row has arrived, else None. A newer update of the same
    entity part discards an incomplete one, at most MAX_INCOMPLETE_UPDATES are
    held (the oldest goes first), and a piece of an update it does not hold,
    whose piece 0 never arrived or was discarded, is dropped and counted. A
    piece that contradicts its update raises :class:`MessageError`
    (bad_request), and the update is discarded."""

    def __init__(self):
        self.open: dict = {}                      # update -> _Assembly, oldest first
        self.dropped = 0
        self.discarded = 0

    def add(self, piece: SceneUpdate) -> SceneUpdate | None:
        uid, num = piece.split["update"], piece.split["piece"]
        if num == 0:
            for k in [k for k, a in self.open.items()
                      if k == uid or (a.p0.path, a.p0.part) == (piece.path, piece.part)]:
                del self.open[k]
                self.discarded += 1
            self.open[uid] = _Assembly(piece)
            while len(self.open) > MAX_INCOMPLETE_UPDATES:
                del self.open[next(iter(self.open))]
                self.discarded += 1
        a = self.open.get(uid)
        if a is None:
            self.dropped += 1
            return None
        try:
            a.add(piece)
            if not a.done():
                return None
            del self.open[uid]
            return a.whole()
        except MessageError:
            self.open.pop(uid, None)
            self.discarded += 1
            raise
