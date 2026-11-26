"""Wire protocol 3.2 GPU programs, interventions, and published resources."""
from __future__ import annotations

from dataclasses import dataclass, field
import sys as _sys
from typing import Any, ClassVar

import numpy as np

if __package__:
    from . import _interpret_protocol, _intervention_protocol, _wire_frame as _f
else:
    _prefix = __name__.rpartition(".")[0]
    _interpret_protocol = _sys.modules[f"{_prefix}._interpret_protocol"]
    _intervention_protocol = _sys.modules[f"{_prefix}._intervention_protocol"]
    _f = _sys.modules[f"{_prefix}._wire_frame"]

DType = _f.DType
ERR_BAD_REQUEST = _f.ERR_BAD_REQUEST
Encoding = _f.Encoding
MAX_DIMS = _f.MAX_DIMS
MessageError = _f.MessageError
MsgType = _f.MsgType
PART_LINES = _f.PART_LINES
PART_POINTS = _f.PART_POINTS
Section = _f.Section
_Msg = _f._Msg
_bad = _f._bad
_bounded_text = _f._bounded_text
_drop_none = _f._drop_none
_finite = _f._finite
_int_in = _f._int_in
_list_of = _f._list_of
_name = _f._name
_nonneg = _f._nonneg
_obj = _f._obj
_opt = _f._opt
_req = _f._req
_safe_uint = _f._safe_uint
_text = _f._text
_u32 = _f._u32
entity_path_ok = _f.entity_path_ok

# ---------------------------------------------------------------------------
# 3.2 draft: GPU programs and interventions (design/gpu-programs.md, Wire Path)
# ---------------------------------------------------------------------------
CAP_PROGRAMS = "programs"
CAP_INTERVENTIONS = "interventions"
PROGRAM_SET, PROGRAM_REMOVE, PROGRAM_PARAMS = "set", "remove", "params"
SECTION_WGSL = "wgsl"
INTERVENE_ABLATE = "ablate"
INTERVENE_OPS = ("ablate", "add", "scale", "clamp", "patch")
MAX_PROGRAMS = 16
PROGRAM_ID_MAX = 55
PROGRAM_SOURCE_MAX = 65536
PROGRAM_STATE_RATE_HZ = 2
POINT_MAX = 255
MAX_POINTS_LISTED = 4096
MAX_INTERVENTIONS = 64
PROGRAM_MAX_INPUTS = 8
PROGRAM_MAX_COUNT = 1000000
PROGRAM_SIDE_MAX = 2048
PROGRAM_TIME_MS_MAX = 50
PROGRAM_MEMORY_MB_MAX = 512
PROGRAM_EVERY_MAX = 10000
PROGRAM_WORKGROUPS_MAX = 65535
TENSOR_NAME_MAX = 255
PROGRAM_PARAMS_MAX = 16
PROGRAM_PARAM_NAME_MAX = 63
PROGRAM_BUILTIN_MAX = 32
TENSOR_AXES_MAX = 4
TENSOR_ELEMENTS_MAX = 1000000
# 3.2 draft, programmable graphics (design/programmable-graphics.md, Resource Message):
# the capability "resources", RESOURCE and the instance form of PROGRAM.
CAP_RESOURCES = "resources"
# The piece form of SCENE_UPDATE: the arrays of an entity part travel as the RESOURCE versions of
# a scene program, an instance of the part's library piece, and the update names them.
CAP_SCENE_PIECES = "scene_pieces"
SCENE_PROGRAM_PREFIX = "scene."
SECTION_DERIVE, SECTION_DRAW, SECTION_COMMON, SECTION_RESOURCE = "derive", "draw", "common", "resource"
MAX_RESOURCES = 8
RESOURCE_MAX_BYTES = 67043328      # 64 MiB less 64 KiB: the section fits one frame with its metadata
RESOURCE_VERSION_MAX_BYTES = 134217728   # buffer: 128 MiB, WebGPU's default largest storage binding
RESOURCE_TEXTURE_VERSION_MAX_BYTES = 268435456  # texture: 8192 by 8192 four-byte texels
RESOURCE_MAX_PARTS = 32            # texture versions use at most 32 parts of 8 MiB
RESOURCE_PART_WAIT = 256           # RESOURCE frames a receiver waits for the next part of a version
RESOURCE_STRIDE_MAX = 256
RESOURCE_SIDE_MAX = 8192           # a texture's width and height: WebGPU's default largest 2-D texture
TEXTURE_RGBA8, TEXTURE_R8, TEXTURE_R32UINT = "rgba8unorm", "r8unorm", "r32uint"   # "format"
ELEMENT_U8, ELEMENT_U16 = "u8", "u16"   # "element": four or two elements a word
PIECE_NAME_MAX = 32
PIECE_VERSION_MAX = 1000000
PIECE_COMMON_MAX = 589824          # 8 lib modules and common.wgsl of 64 KiB each
PIECE_SEGMENTS_MAX = 16
STATUS_PIECE = "piece"
OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR = "geometry", "pixels", "tensor"
STATUS_OPERATOR = "operator"
PROGRAM_STATES = ("waiting", "review", "compiled", "running", "refused", "stopped")
PROGRAM_ARRAYS = ("positions", "colors", "values", "radii")
INTERVENTION_STATES = ("pending", "active", "ended")
MAX_INCOMPLETE_UPDATES = 2
_ID_TAIL = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
_DIGITS = frozenset("0123456789")
_PARAM_HEAD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_PARAM_TAIL = _PARAM_HEAD | _DIGITS
_HEX = frozenset("0123456789abcdef")
_PIECE_HEAD = frozenset("abcdefghijklmnopqrstuvwxyz")
_PIECE_TAIL = _PIECE_HEAD | _DIGITS | {"_"}


def _spec_id(v: str) -> bool:
    return bool(v) and "a" <= v[0] <= "z" and all(c in _ID_TAIL for c in v)


def _program_id(v) -> bool:
    """A program id: 1 to 55 bytes of [a-z0-9_-], the first a letter, the id of an
    operator, ``operator.<id>.<n>``: a spec id, a dot and 1 to 5 digits, or the id of a
    scene program, ``scene.<n>`` with 1 to 9 digits (:func:`scene_program_id`)."""
    if not (isinstance(v, str) and 0 < len(v) <= PROGRAM_ID_MAX and v.isascii()):
        return False
    if v.startswith(SCENE_PROGRAM_PREFIX):
        return scene_program_id(v)
    if v.startswith("operator."):
        spec, dot, n = v[len("operator."):].rpartition(".")
        return bool(dot) and _spec_id(spec) and 1 <= len(n) <= 5 and all(c in _DIGITS for c in n)
    return _spec_id(v)


def scene_program_id(v) -> bool:
    """The id of a scene program, ``scene.<n>`` with n of 1 to 9 digits: the program whose
    RESOURCE versions hold the arrays of one entity part of a scene in the piece form of
    SCENE_UPDATE. No spec program takes such an id, since spec ids hold no dot."""
    if not (isinstance(v, str) and v.startswith(SCENE_PROGRAM_PREFIX)):
        return False
    n = v[len(SCENE_PROGRAM_PREFIX):]
    return 1 <= len(n) <= 9 and all(c in _DIGITS for c in n)


def _param_name(v) -> bool:
    """A WGSL field name of 1 to 63 bytes, [A-Za-z_][A-Za-z0-9_]*, not starting with "__"."""
    return (isinstance(v, str) and 0 < len(v) <= PROGRAM_PARAM_NAME_MAX and v.isascii()
            and v[0] in _PARAM_HEAD and all(c in _PARAM_TAIL for c in v)
            and not v.startswith("__"))


def _params_ok(v) -> bool:
    """Program parameters: 1 to 16 names of a WGSL field, each with a finite number."""
    return (isinstance(v, dict) and 1 <= len(v) <= PROGRAM_PARAMS_MAX
            and all(_param_name(k) and _finite(x) for k, x in v.items()))


def _tensor_shape_ok(v) -> bool:
    """The shape of a tensor output: 1 to 4 integers from 1, at most 1,000,000 elements."""
    if not _list_of(_int_in(1, TENSOR_ELEMENTS_MAX), TENSOR_AXES_MAX, 1)(v):
        return False
    n = 1
    for x in v:
        n *= x
    return n <= TENSOR_ELEMENTS_MAX


def _sha256_hex(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in _HEX for c in v)


def _input_entry(v) -> bool:
    return (_obj(v) and _name(v.get("var")) and _bounded_text(TENSOR_NAME_MAX)(v.get("source")))


def _arrays_ok(v) -> bool:
    return (isinstance(v, list) and 1 <= len(v) <= len(PROGRAM_ARRAYS)
            and all(isinstance(x, str) and x in PROGRAM_ARRAYS for x in v)
            and len(set(v)) == len(v) and "positions" in v)


def _check_declaration(d: dict) -> None:
    """The checks of a PROGRAM set's ``declaration``, in this order: ``output``
    with ``kind`` and the keys of its kind, then ``inputs``, ``limits``,
    ``workgroups`` and ``entry``. Every count that sizes a buffer is bounded
    here; the sandbox checks the rest before any run."""
    out = _req(d, "output", _obj, "an object")
    kind = _req(out, "kind", lambda v: v in (OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR),
                "geometry, pixels or tensor")
    if kind == OUTPUT_TENSOR:
        _req(out, "shape", _tensor_shape_ok, f"1 to {TENSOR_AXES_MAX} integers from 1 with at most "
             f"{TENSOR_ELEMENTS_MAX} elements")
        if set(out) - {"kind", "shape"}:
            _bad("a tensor output holds only 'kind' and 'shape'")
    elif kind == OUTPUT_GEOMETRY:
        _req(out, "part", lambda v: v in (PART_POINTS, PART_LINES), "points or lines")
        _req(out, "max_count", _int_in(1, PROGRAM_MAX_COUNT), f"an integer from 1 to {PROGRAM_MAX_COUNT}")
        _req(out, "arrays", _arrays_ok, "unique names of positions, colors, values and radii, "
             "with positions")
        _opt(out, "entity", entity_path_ok, "an entity path")
    else:
        side = _int_in(1, PROGRAM_SIDE_MAX)
        _req(out, "width", side, f"an integer from 1 to {PROGRAM_SIDE_MAX}")
        _req(out, "height", side, f"an integer from 1 to {PROGRAM_SIDE_MAX}")
        _opt(out, "format", lambda v: v == "rgba8unorm", "rgba8unorm")
    _opt(d, "inputs", _list_of(_input_entry, PROGRAM_MAX_INPUTS),
         f"at most {PROGRAM_MAX_INPUTS} inputs {{var, source}}")
    lim = _opt(d, "limits", _obj, "an object")
    if lim is not None:
        _opt(lim, "time_ms", lambda v: _finite(v) and 0 < v <= PROGRAM_TIME_MS_MAX,
             f"a number above 0 and at most {PROGRAM_TIME_MS_MAX}")
        _opt(lim, "memory_mb", _int_in(1, PROGRAM_MEMORY_MB_MAX),
             f"an integer from 1 to {PROGRAM_MEMORY_MB_MAX}")
        _opt(lim, "every", _int_in(1, PROGRAM_EVERY_MAX), f"an integer from 1 to {PROGRAM_EVERY_MAX}")
    _opt(d, "workgroups", _list_of(_int_in(1, PROGRAM_WORKGROUPS_MAX), 3, 3),
         f"three integers from 1 to {PROGRAM_WORKGROUPS_MAX}")
    _opt(d, "entry", _name, "an entry point name")
    _opt(d, "params", _params_ok, f"1 to {PROGRAM_PARAMS_MAX} names of a WGSL field with finite numbers")
    _opt(d, "builtin", _bounded_text(PROGRAM_BUILTIN_MAX),
         f"a built-in name of 1 to {PROGRAM_BUILTIN_MAX} bytes")


@dataclass
class Program(_Msg):
    """Viewer to producer, under the capability "programs": install or replace
    (``op`` "set"), remove (``op`` "remove") or change the parameters of (``op`` "params") one GPU
    program. A params op carries only ``params``, 1 to 16 names of the program's uniform with
    finite numbers, which the program reads at its next pass, with no compile. A set carries
    the declaration without its source, the scene of a geometry output's panel,
    the SHA-256 of the source and the section ``wgsl`` (the source as UTF-8).
    The producer runs the source only on the GPU, after the sandbox checks, and
    answers with PROGRAM_STATE (``design/gpu-programs.md``, Wire Path).

    Under the capability "resources", a set may instead install an instance of a
    piece (``design/programmable-graphics.md``, Resource Message): ``piece`` (the
    pin, ``{name, version, hash}``), ``manifest``, and optionally ``bind``, ``params``,
    ``panel`` and ``segments``, with the sections ``common``, ``derive`` and ``draw``
    in place of ``wgsl``. A producer also sends this form, with ``request_id`` 0 and
    with all three sections, to each viewer that announced "resources"."""
    TYPE: ClassVar[MsgType] = MsgType.PROGRAM
    request_id: int
    op: str                                      # "set" or "remove"
    id: str
    declaration: dict | None = None              # set
    wgsl: str | None = None                      # set: the section "wgsl"
    sha256: str | None = None                    # set: of wgsl, in lowercase hex
    scene: str | None = None                     # set: a geometry output's scene
    params: dict | None = None                   # params, or an instance's: the names and their numbers
    piece: dict | None = None                    # instance: {name, version, hash}
    manifest: dict | None = None                 # instance: the piece's manifest
    bind: dict | None = None                     # instance: input name to live:<tensor> or program:<id>/<resource>
    panel: str | None = None                     # instance: the scene panel of the draw half
    segments: list | None = None                 # instance: [{part, first_line, lines}] of ``common``
    common: str | None = None                    # instance: the section "common"
    derive: str | None = None                    # instance: the section "derive"
    draw: str | None = None                      # instance: the section "draw"

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op, "id": self.id,
                           "declaration": self.declaration, "scene": self.scene,
                           "sha256": self.sha256, "params": self.params, "piece": self.piece,
                           "manifest": self.manifest, "bind": self.bind, "panel": self.panel,
                           "segments": self.segments})

    def to_sections(self):
        texts = ((SECTION_WGSL, self.wgsl), (SECTION_COMMON, self.common),
                 (SECTION_DERIVE, self.derive), (SECTION_DRAW, self.draw))
        return [Section.from_array(name, np.frombuffer(text.encode("utf-8"), dtype=np.uint8))
                for name, text in texts if text is not None]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        rid = _req(m, "request_id", _u32, "a uint32")
        op = _req(m, "op", lambda v: v in (PROGRAM_SET, PROGRAM_REMOVE, PROGRAM_PARAMS),
                  "set, remove or params")
        pid = _req(m, "id", _program_id, f"a program id of at most {PROGRAM_ID_MAX} bytes")
        if op == PROGRAM_REMOVE:
            return cls(rid, op, pid)
        if op == PROGRAM_PARAMS:
            # Values alone: a params op never declares or carries a program.
            if "declaration" in m:
                _bad("a params op carries no 'declaration'")
            if f.sections:
                _bad("a params op carries no section")
            ps = _req(m, "params", _params_ok, f"1 to {PROGRAM_PARAMS_MAX} names of a WGSL field "
                      "with finite numbers")
            return cls(rid, op, pid, params=ps)
        if "piece" in m:
            return _program_instance(cls, f, rid, pid)
        d = _req(m, "declaration", _obj, "an object")
        _check_declaration(d)
        geometry = d["output"]["kind"] == OUTPUT_GEOMETRY
        scene = (_req if geometry else _opt)(m, "scene", _name, "a scene name")
        sha = _req(m, "sha256", _sha256_hex, "64 lowercase hex digits")
        w = f.section(SECTION_WGSL)
        if w is None or w.encoding != Encoding.RAW or w.dtype != DType.U8 or len(w.shape) != 1:
            _bad("a program set needs a (B,) u8 'wgsl' section")
        if not 1 <= w.shape[0] <= PROGRAM_SOURCE_MAX:
            _bad(f"'wgsl' holds 1 to {PROGRAM_SOURCE_MAX} bytes, got {w.shape[0]}")
        try:
            src = bytes(w.data).decode("utf-8")
        except UnicodeDecodeError:
            _bad("'wgsl' is not UTF-8")
        return cls(rid, op, pid, d, src, sha, scene)


def _program_error(v) -> bool:
    return (_obj(v) and _name(v.get("code")) and _text(v.get("message"))
            and all(v.get(k) is None or _u32(v[k]) for k in ("line", "column")))


_STATUS_KEYS = (("every", _int_in(1, PROGRAM_EVERY_MAX)), ("every_min", _int_in(1, PROGRAM_EVERY_MAX)),
                ("last_pass_ms", _nonneg), ("last_cost_ms", _nonneg), ("step_ms", _nonneg),
                ("overhead_share", _nonneg), ("passes_per_s", _nonneg),
                ("last_step", _safe_uint), ("runs", _safe_uint), ("memory_bytes", _safe_uint),
                ("memory_limit_bytes", _safe_uint), ("error", _program_error),
                ("warning", _program_error))


def _status_item(v) -> bool:
    """A status item of PROGRAM_STATE: ``id``, ``kind`` and ``state``, and the
    optional keys of ``_STATUS_KEYS`` and ``sha256``, null or absent when
    unknown. ``host`` is the app's and is ignored here."""
    if not (_obj(v) and _program_id(v.get("id"))
            and v.get("kind") in (OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR, STATUS_OPERATOR,
                                  STATUS_PIECE)
            and v.get("state") in PROGRAM_STATES):
        return False
    if not all(v.get(k) is None or check(v[k]) for k, check in _STATUS_KEYS):
        return False
    sha = v.get("sha256")
    return sha is None or sha == "" or _sha256_hex(sha)


def _status_items(v) -> bool:
    return (_list_of(_status_item, MAX_PROGRAMS)(v)
            and len({x["id"] for x in v}) == len(v))


@dataclass
class ProgramState(_Msg):
    """Producer to viewer, under the capability "programs": the status item of
    each program, in the order of installation. It answers each PROGRAM with
    its ``request_id``, and goes out without one after a change of a state and
    at most ``PROGRAM_STATE_RATE_HZ`` times a second while programs run.
    ``step`` is the script's step and ``step_ms`` the base step time of the cadence."""
    TYPE: ClassVar[MsgType] = MsgType.PROGRAM_STATE
    programs: list = field(default_factory=list)
    request_id: int | None = None
    step: int | None = None
    step_ms: float | None = None

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "step": self.step,
                           "step_ms": self.step_ms, "programs": list(self.programs)})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        rid = _opt(m, "request_id", _u32, "a uint32")
        step = _opt(m, "step", _safe_uint, "an unsigned integer")
        step_ms = _opt(m, "step_ms", _nonneg, "a number of at least 0")
        items = _req(m, "programs", _status_items,
                     f"at most {MAX_PROGRAMS} status items with unique program ids")
        return cls(items, rid, step, step_ms)


def _point(v) -> bool:
    return _bounded_text(POINT_MAX)(v)


def _opt_step(v) -> bool:
    return v is None or _safe_uint(v)


def _intervene_request(v) -> bool:
    if (not _obj(v) or set(v) - {"set", "clear", "clear_all"}
            or sum(v.get(k) is not None for k in ("set", "clear", "clear_all")) != 1):
        return False
    if v.get("set") is not None:
        return _intervention_protocol.specification(v["set"])
    if v.get("clear") is not None:
        return _interpret_protocol.text(v["clear"], POINT_MAX, False)
    return v["clear_all"] is True


def _point_entry(v) -> bool:
    shape = v.get("shape") if _obj(v) else None
    return (_obj(v) and _point(v.get("point")) and v.get("kind") in ("module", "head")
            and _point(v.get("module")) and (v.get("head") is None or _u32(v["head"]))
            and (v.get("type") is None or _name(v["type"]))
            and (shape is None or _list_of(_safe_uint, MAX_DIMS)(shape)))


def _active_entry(v) -> bool:
    return _intervention_protocol.specification(v, lifecycle=True)


def _interventions(v) -> bool:
    """CONTROLS' ``interventions``: ``points`` (at most ``MAX_POINTS_LISTED``)
    and ``active`` (at most ``MAX_INTERVENTIONS``)."""
    return (_obj(v) and _list_of(_point_entry, MAX_POINTS_LISTED)(v.get("points"))
            and _list_of(_active_entry, MAX_INTERVENTIONS)(v.get("active")))


def _piece_pin(v) -> bool:
    """The pin of a piece: ``name`` ([a-z][a-z0-9_]*, at most 32 bytes), ``version`` (1 to
    1,000,000), ``hash`` ("sha256:" and 64 lowercase hex digits) and optional ``from``
    (``library``, ``workspace`` or ``inline``)."""
    if not _obj(v):
        return False
    name, h, fr = v.get("name"), v.get("hash"), v.get("from")
    return (isinstance(name, str) and 0 < len(name) <= PIECE_NAME_MAX and name.isascii()
            and name[0] in _PIECE_HEAD and all(c in _PIECE_TAIL for c in name)
            and _int_in(1, PIECE_VERSION_MAX)(v.get("version"))
            and isinstance(h, str) and h.startswith("sha256:") and _sha256_hex(h[len("sha256:"):])
            and (fr is None or fr in ("library", "workspace", "inline")))


def _library_pin(v) -> bool:
    """The pin of a library piece in the piece form of SCENE_UPDATE: ``name`` and ``version``
    as in :func:`_piece_pin`, and ``hash`` optional. The viewer draws its own library piece of
    that name and version, whose version names the layout of its resources."""
    if not _obj(v):
        return False
    return _piece_pin(dict(v, hash=v.get("hash", "sha256:" + "0" * 64)))


def _binding(v) -> bool:
    """One value of ``bind``: ``live:<tensor>`` (a tensor name of 1 to 255 bytes) or
    ``program:<id>/<resource>`` (a program id and a resource's WGSL name)."""
    if not isinstance(v, str):
        return False
    if v.startswith("live:"):
        return _bounded_text(TENSOR_NAME_MAX)(v[len("live:"):])
    if v.startswith("program:"):
        pid, slash, res = v[len("program:"):].partition("/")
        return bool(slash) and _program_id(pid) and _param_name(res)
    return False


def _bind_ok(v) -> bool:
    return (_obj(v) and len(v) <= PROGRAM_MAX_INPUTS
            and all(_param_name(k) and _binding(x) for k, x in v.items()))


def _segment(v) -> bool:
    return (_obj(v) and _name(v.get("part")) and _u32(v.get("first_line")) and v["first_line"] >= 1
            and _u32(v.get("lines")))


def _text_section(f, name: str, most: int) -> str | None:
    """An optional (B,) u8 section of 1 to ``most`` bytes of UTF-8, as text."""
    s = f.section(name)
    if s is None:
        return None
    if s.encoding != Encoding.RAW or s.dtype != DType.U8 or len(s.shape) != 1:
        _bad(f"{name!r} must be a (B,) u8 section")
    if not 1 <= s.shape[0] <= most:
        _bad(f"{name!r} holds 1 to {most} bytes, got {s.shape[0]}")
    try:
        return bytes(s.data).decode("utf-8")
    except UnicodeDecodeError:
        _bad(f"{name!r} is not UTF-8")


def _program_instance(cls, f, rid: int, pid: str):
    """The set of an instance of a piece, in this order: no key or section of the
    one-pass form, ``piece``, ``manifest``, ``bind``, ``params``, ``panel``,
    ``segments``, then the sections ``common``, ``derive`` and ``draw``, of which
    ``derive`` or ``draw`` is present."""
    m = f.meta
    if any(k in m for k in ("declaration", "scene", "sha256")) or f.section(SECTION_WGSL):
        _bad("an instance of a piece carries no 'declaration', 'scene', 'sha256' or 'wgsl'")
    piece = _req(m, "piece", _piece_pin, "a pin {name, version, hash}")
    manifest = _req(m, "manifest", _obj, "an object")
    bind = _opt(m, "bind", _bind_ok, f"at most {PROGRAM_MAX_INPUTS} inputs bound to live:<tensor> "
                "or program:<id>/<resource>")
    ps = _opt(m, "params", _params_ok, f"1 to {PROGRAM_PARAMS_MAX} names of a WGSL field with "
              "finite numbers")
    panel = _opt(m, "panel", _name, "a panel id")
    segments = _opt(m, "segments", _list_of(_segment, PIECE_SEGMENTS_MAX),
                    f"at most {PIECE_SEGMENTS_MAX} segments {{part, first_line, lines}}")
    common = _text_section(f, SECTION_COMMON, PIECE_COMMON_MAX)
    derive = _text_section(f, SECTION_DERIVE, PROGRAM_SOURCE_MAX)
    draw = _text_section(f, SECTION_DRAW, PROGRAM_SOURCE_MAX)
    if derive is None and draw is None:
        _bad("an instance of a piece carries the section 'derive' or 'draw'")
    return cls(rid, PROGRAM_SET, pid, params=ps, piece=piece, manifest=manifest, bind=bind,
               panel=panel, segments=segments, common=common, derive=derive, draw=draw)


def _stride_ok(v) -> bool:
    return _int_in(4, RESOURCE_STRIDE_MAX)(v) and v % 4 == 0


@dataclass
class Resource(_Msg):
    """Producer to viewer, under the capability "resources": one version of one
    published resource of a program, the valid elements that a derive pass wrote
    (``design/programmable-graphics.md``, Resource Message). ``step`` is the run's
    step at the pass, ``count`` the valid elements and ``stride`` the bytes of one;
    a texture also has ``width`` and ``height`` (1 to RESOURCE_SIDE_MAX each) and
    optionally ``format`` (rgba8unorm when absent), with a stride of 4 and ``count``
    the 32-bit words of its texels: width * height, or (width * height + 3) // 4
    for r8unorm. A buffer of narrow elements has ``element`` ("u8" or "u16") and
    ``elements``, with a stride of 4 and ``count`` (elements * bytes + 3) // 4. The section ``resource`` holds exactly count * stride bytes,
    and ``data`` is a view of them. A part of a version in parts (:func:`resource_parts`)
    also has ``split`` ({version, part, parts, first}), and its section holds whole
    elements from element ``first`` on."""
    TYPE: ClassVar[MsgType] = MsgType.RESOURCE
    program: str
    resource: str
    step: int
    count: int
    stride: int
    data: Any = b""
    width: int | None = None
    height: int | None = None
    split: dict | None = None
    format: str | None = None
    element: str | None = None
    elements: int | None = None

    def to_meta(self):
        return _drop_none({"program": self.program, "resource": self.resource, "step": self.step,
                           "count": self.count, "stride": self.stride, "width": self.width,
                           "height": self.height, "format": self.format, "element": self.element,
                           "elements": self.elements, "split": self.split})

    def to_sections(self):
        mv = memoryview(self.data)
        mv = mv.cast("B") if mv.nbytes else memoryview(b"")
        return [Section(SECTION_RESOURCE, int(DType.U8), (mv.nbytes,), mv)]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        program = _req(m, "program", _program_id, f"a program id of at most {PROGRAM_ID_MAX} bytes")
        resource = _req(m, "resource", _param_name, "a WGSL name of 1 to 63 bytes")
        step = _req(m, "step", _safe_uint, "an unsigned integer")
        count = _req(m, "count", _u32, "a uint32")
        stride = _req(m, "stride", _stride_ok, f"a multiple of 4 from 4 to {RESOURCE_STRIDE_MAX}")
        side = _int_in(1, RESOURCE_SIDE_MAX)
        width = _opt(m, "width", side, f"an integer from 1 to {RESOURCE_SIDE_MAX}")
        height = _opt(m, "height", side, f"an integer from 1 to {RESOURCE_SIDE_MAX}")
        if (width is None) != (height is None):
            _bad("a texture holds both 'width' and 'height'")
        fmt = _opt(m, "format", lambda v: v in (TEXTURE_RGBA8, TEXTURE_R8, TEXTURE_R32UINT),
                   "rgba8unorm, r8unorm or r32uint")
        if fmt is not None and width is None:
            _bad("'format' belongs to a texture, with 'width' and 'height'")
        if width is not None:
            texels = width * height
            words = (texels + 3) // 4 if fmt == TEXTURE_R8 else texels
            if stride != 4 or count != words:
                _bad("a texture has a stride of 4 and the words of its texels as its count")
        element = _opt(m, "element", lambda v: v in (ELEMENT_U8, ELEMENT_U16), "u8 or u16")
        elements = _opt(m, "elements", _u32, "a uint32")
        if (element is None) != (elements is None):
            _bad("a buffer of narrow elements holds both 'element' and 'elements'")
        if element is not None:
            if width is not None:
                _bad("a texture has no 'element'")
            nbytes = 1 if element == ELEMENT_U8 else 2
            if stride != 4 or count != (elements * nbytes + 3) // 4:
                _bad("a buffer of narrow elements has a stride of 4 and the words of its elements as its count")
        split = _opt(m, "split", _obj, "an object")
        if split is not None:
            def need(key, ok, what):
                v = split.get(key)
                if v is None or not ok(v):
                    _bad(f"'split.{key}' must be {what}, got {v!r}")
                return v
            version = need("version", _safe_uint, "an unsigned integer")
            parts = need("parts", _int_in(2, RESOURCE_MAX_PARTS), f"an integer from 2 to {RESOURCE_MAX_PARTS}")
            part = need("part", _int_in(0, parts - 1), "a part number below 'parts'")
            first = need("first", _u32, "a uint32")
            split = {"version": version, "part": part, "parts": parts, "first": first}
        limit = (RESOURCE_MAX_BYTES if split is None else
                 RESOURCE_TEXTURE_VERSION_MAX_BYTES if width is not None else RESOURCE_VERSION_MAX_BYTES)
        if count > limit // stride:
            _bad(f"{'a resource' if split is None else 'a resource in parts'} holds at most {limit} bytes")
        s = f.section(SECTION_RESOURCE)
        if s is None or s.encoding != Encoding.RAW or s.dtype != DType.U8 or len(s.shape) != 1:
            _bad("a resource needs a (B,) u8 'resource' section")
        if split is not None:
            b = s.shape[0]
            if split["part"] == 0 and split["first"] != 0:
                _bad("part 0 of a resource starts at element 0")
            if split["first"] >= count:
                _bad("'split.first' must lie below the count")
            if b <= 0 or b % stride or b // stride > count - split["first"]:
                _bad("a part of a resource holds whole elements from 'split.first' up to the count")
        elif s.shape[0] != count * stride:
            _bad(f"'resource' holds count * stride = {count * stride} bytes, got {s.shape[0]}")
        return cls(program, resource, step, count, stride, s.data, width, height, split, fmt, element, elements)


def resource_parts(r: Resource, version: int, part_bytes: int) -> list:
    """The parts of the version ``r`` (counter ``version`` of the sender's versions
    in parts), each with at most ``part_bytes`` bytes of whole elements, as views
    of ``r.data`` without a copy; ``[r]`` when its section holds at most
    ``part_bytes`` bytes. Raises ValueError for a version over
    the texture or buffer byte limit, or one that needs more than RESOURCE_MAX_PARTS parts."""
    mv = memoryview(r.data)
    mv = mv.cast("B") if mv.nbytes else memoryview(b"")
    if r.count * r.stride != mv.nbytes:
        raise ValueError(f"a version of {r.count} elements of {r.stride} bytes holds {mv.nbytes} bytes")
    limit = RESOURCE_TEXTURE_VERSION_MAX_BYTES if r.width is not None else RESOURCE_VERSION_MAX_BYTES
    if mv.nbytes > limit:
        raise ValueError(f"a version of {mv.nbytes} bytes, more than {limit}")
    if mv.nbytes <= part_bytes:
        return [r]
    per = part_bytes // r.stride
    if per < 1:
        raise ValueError(f"a part of {part_bytes} bytes holds no element of {r.stride} bytes")
    parts = -(-r.count // per)
    if parts > RESOURCE_MAX_PARTS:
        raise ValueError(f"a version of {mv.nbytes} bytes needs {parts} parts of {part_bytes} bytes, "
                         f"more than {RESOURCE_MAX_PARTS}")
    out = []
    for k in range(parts):
        a, b = k * per, min(r.count, (k + 1) * per)
        out.append(Resource(r.program, r.resource, r.step, r.count, r.stride, mv[a * r.stride:b * r.stride],
                            r.width, r.height, {"version": version, "part": k, "parts": parts, "first": a},
                            r.format, r.element, r.elements))
    return out


class SplitResources:
    """A receiver's versions in parts of one connection. :meth:`add` takes every
    RESOURCE of the connection in order and returns the whole version, else
    None. A version in one frame comes back at once. Part 0 allocates the
    version's count * stride bytes, which the decoder bounded by
    the texture or buffer byte limit, and each part copies its elements in; the last
    part returns the whole. A part whose version is not held is dropped and
    counted. A part out of order, one whose keys differ from part 0, or a last
    part that does not end at the count raises :class:`MessageError` and
    discards the version. Part 0 of a newer version, or a version in one frame,
    of the same resource discards an incomplete one; at most
    MAX_INCOMPLETE_UPDATES are held, the oldest going first; and one whose next
    part has not come after RESOURCE_PART_WAIT further RESOURCE frames expires."""

    def __init__(self):
        self.open: dict = {}                      # (program, resource) -> [head, next_part, next_first, last, bytes]
        self.frames = 0
        self.dropped = 0
        self.discarded = 0
        self.expired = 0

    def add(self, r: Resource) -> Resource | None:
        self.frames += 1
        for k in [k for k, a in self.open.items() if self.frames - a[3] > RESOURCE_PART_WAIT]:
            del self.open[k]
            self.expired += 1
        key = (r.program, r.resource)
        sp = r.split
        if (sp is None or sp["part"] == 0) and key in self.open:
            del self.open[key]
            self.discarded += 1
        if sp is None:
            return r
        data = memoryview(r.data).cast("B") if memoryview(r.data).nbytes else memoryview(b"")
        n = data.nbytes // r.stride
        if sp["part"] == 0:
            while len(self.open) >= MAX_INCOMPLETE_UPDATES:
                del self.open[next(iter(self.open))]
                self.discarded += 1
            buf = bytearray(r.count * r.stride)
            buf[:data.nbytes] = data
            self.open[key] = [r, 1, n, self.frames, buf]
            return None
        a = self.open.get(key)
        if a is None or a[0].split["version"] != sp["version"]:
            self.dropped += 1
            return None
        h, name = a[0], f"'{r.program}/{r.resource}'"

        def refuse(why):
            del self.open[key]
            self.discarded += 1
            raise MessageError(ERR_BAD_REQUEST, why)
        if (r.step, r.count, r.stride, r.width, r.height, r.format, r.element, r.elements, sp["parts"]) != \
                (h.step, h.count, h.stride, h.width, h.height, h.format, h.element, h.elements, h.split["parts"]):
            refuse(f"a part of the resource {name} differs from its part 0")
        if sp["part"] != a[1] or sp["first"] != a[2]:
            refuse(f"a part of the resource {name} comes out of order")
        end, last = a[2] + n, sp["part"] + 1 == sp["parts"]
        if last != (end == h.count):
            refuse(f"the parts of the resource {name} end before its count" if last else
                   f"a part of the resource {name} before the last reaches its count")
        a[4][a[2] * h.stride:end * h.stride] = data
        a[1], a[2], a[3] = a[1] + 1, end, self.frames
        if not last:
            return None
        del self.open[key]
        return Resource(h.program, h.resource, h.step, h.count, h.stride, memoryview(a[4]), h.width, h.height, None, h.format, h.element, h.elements)
