"""Wire protocol v3 framing, core constants, enums, errors, and shared field validators."""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import json
import math
import re
import struct
from typing import Any, ClassVar

import numpy as np

# --- version, endpoint (caliper/live_net.h) ---------------------------------
MAJOR = 3
MINOR = 1
MINOR_3_2 = 2
MAGIC = 0x574C4143                 # bytes b"CALW" read as a little-endian uint32
PATH = "/v3/stream"
TOKEN_PARAM = "token"

HEADER = struct.Struct("<IHHHHIIHH")   # magic major minor type flags frame_len meta_len count reserved
HEADER_BYTES = 24
SECTION_ALIGN = 8
FRAME_FLAGS_NONE = 0
assert HEADER.size == HEADER_BYTES


class MsgType(enum.IntEnum):
    HELLO = 0x0001
    WELCOME = 0x0002
    BYE = 0x0003
    ERROR = 0x0004
    PING = 0x0005
    PONG = 0x0006
    SESSIONS = 0x0010
    ATTACH = 0x0011
    VIEWS = 0x0020
    SUBSCRIBE = 0x0021
    UNSUBSCRIBE = 0x0022
    VIEW_REQUEST = 0x0023
    VIEW_REPLY = 0x0024
    SNAPSHOT = 0x0030
    SERIES = 0x0031
    EVENT = 0x0040
    CONTROLS = 0x0050              # 3.1, capability "controls"
    CONTROL = 0x0051
    CONTROL_REPLY = 0x0052
    SELECTION = 0x0060             # 3.1, capability "selection"
    SCENE_UPDATE = 0x0070          # 3.2 draft, capability "scenes" (see the 3.2 block)
    SCENE_TRANSFORMS = 0x0071      # 3.2 draft
    SCENE_CLEAR = 0x0072           # 3.2 draft
    SCENE_CLASSES = 0x0073         # 3.2 draft
    TEXT = 0x0074                  # 3.2 draft, capability "text"
    RECORDINGS = 0x0078            # 3.2 draft, capability "recordings"
    RECORDING_OPEN = 0x0079
    CURSOR = 0x007A
    TIME_STATE = 0x007B
    PROGRAM = 0x0080               # 3.2 draft, capability "programs"
    PROGRAM_STATE = 0x0081
    RESOURCE = 0x0082              # 3.2 draft, capability "resources"


class Kind(enum.IntEnum):          # CaliperLiveKind
    HEATMAP = 1
    IMAGE = 2
    POINTS = 3
    SERIES = 4
    HISTOGRAM = 5
    SCENE = 6                      # 3.2 draft, capability "scenes" (see the 3.2 block)
    TEXT = 7                       # 3.2 draft, capability "text"
    BARS = 8                       # 3.2 draft, capability "bars"


class DType(enum.IntEnum):         # CaliperDType
    F32 = 0
    F16 = 1
    BF16 = 2
    I64 = 3
    I32 = 4
    U8 = 5
    U16 = 6                        # 3.2 draft (see the 3.2 block)


class Encoding(enum.IntEnum):
    RAW = 0
    ARROW_IPC = 1                  # reserved


DTYPE_SIZE = {DType.F32: 4, DType.F16: 2, DType.BF16: 2, DType.I64: 8, DType.I32: 4,
              DType.U8: 1, DType.U16: 2}
# numpy views of a section; bf16 has no numpy type and is viewed as raw uint16.
_NP_DTYPE = {DType.F32: "<f4", DType.F16: "<f2", DType.BF16: "<u2", DType.I64: "<i8",
             DType.I32: "<i4", DType.U8: "u1", DType.U16: "<u2"}
_FROM_NP = {np.dtype("float32"): DType.F32, np.dtype("float16"): DType.F16,
            np.dtype("int64"): DType.I64, np.dtype("int32"): DType.I32,
            np.dtype("uint8"): DType.U8, np.dtype("uint16"): DType.U16}
_DTYPE_NAME = {DType.F32: "f32", DType.F16: "f16", DType.BF16: "bf16", DType.I64: "i64",
               DType.I32: "i32", DType.U8: "u8", DType.U16: "u16"}

COLORMAP_VIRIDIS, COLORMAP_MAGMA, COLORMAP_RDBU = 0, 1, 2
COLORMAP_GREYS, COLORMAP_CIVIDIS, COLORMAP_INFERNO, COLORMAP_PLASMA = 3, 4, 5, 6
COLORMAP_TURBO, COLORMAP_TWILIGHT, COLORMAP_COOLWARM, COLORMAP_BATLOW = 7, 8, 9, 10
COLORMAP_VIK, COLORMAP_ROMA, COLORMAP_OSLO, COLORMAP_LAJOLLA, COLORMAP_BERLIN = 11, 12, 13, 14, 15
SNAP_AUTO_VMIN = 0x1
SNAP_AUTO_VMAX = 0x2
SNAP_RAW = 0x4

CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_TOO_BIG = 1009
CLOSE_PROTOCOL_ERROR = 4000
CLOSE_UNSUPPORTED_MAJOR = 4001
CLOSE_ROLE_REFUSED = 4002
CLOSE_HANDSHAKE_TIMEOUT = 4003
CLOSE_IDLE_TIMEOUT = 4004
CLOSE_SERVER_FULL = 4005

ROLE_VIEWER, ROLE_PRODUCER, ROLE_HUB = "viewer", "producer", "hub"
SESSION_LIVE, SESSION_ENDED = "live", "ended"
BYE_NORMAL, BYE_SHUTDOWN, BYE_REPLACED, BYE_ERROR = "normal", "shutdown", "replaced", "error"
ERR_BAD_REQUEST = "bad_request"
ERR_UNKNOWN_SESSION = "unknown_session"
ERR_UNKNOWN_VIEW = "unknown_view"
ERR_UNKNOWN_SUBSCRIPTION = "unknown_subscription"
ERR_BAD_DATA_VIEW = "bad_data_view"
ERR_NOT_ATTACHED = "not_attached"
ERR_UNSUPPORTED = "unsupported"
ERR_LIMIT = "limit"
ERR_INTERNAL = "internal"
ERR_FORBIDDEN = "forbidden"                # 3.1: CONTROL_REPLY errors
ERR_UNKNOWN_CONTROL = "unknown_control"
ERR_BAD_VALUE = "bad_value"
ERR_TIMEOUT = "timeout"
ERR_ACTION_FAILED = "action_failed"
EVENT_SELECT, EVENT_PICK, EVENT_HOVER = "select", "pick", "hover"
REDUCE_OPS = ("mean", "sum", "max", "min", "norm")
POOL_OPS = ("mean", "max", "min")
CAP_EVENTS = "events"
CAP_ARROW_IPC = "arrow_ipc"
CAP_CONTROLS = "controls"                  # 3.1
CAP_SELECTION = "selection"                # 3.1

PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES = "points", "lines", "arrows", "boxes"
PART_MESH, PART_IMAGE, PART_CAMERA = "mesh", "image", "camera"
PART_TRANSFORM = "transform"

MAX_FRAME_BYTES = 64 << 20
MAX_META_BYTES = 1 << 20
MAX_SECTIONS = 8
MAX_JSON_DEPTH = 16
NAME_MAX = 63
SESSION_ID_MAX = 63
TOKEN_MIN = 22
TOKEN_MAX = 128
MAX_VIEWS = 256
MAX_SOURCES = 64
MAX_DIMS = 8
MAX_SUBSCRIPTIONS = 256
MAX_STATS = 64
DEFAULT_MAX_EXTENT = 512
LIMIT_MAX_EXTENT = 4096
DEFAULT_MAX_POINTS = 100_000
LIMIT_MAX_POINTS = 1_000_000
MAX_BINS = 65536
SERIES_RETAIN_MIN = 4096
SERIES_MAX_VALUES = 262_144
MAX_EVENT_INDICES = 32768
DEFAULT_RATE_HZ = 60
LIMIT_RATE_HZ = 240
MAX_QUEUED_EVENTS = 256
HANDSHAKE_TIMEOUT_MS = 10_000
PING_INTERVAL_MS = 10_000
IDLE_TIMEOUT_MS = 30_000
TEXT_MAX = 1024
PATH_MAX = 255
PATH_SEGMENTS_MAX = 16
MAX_TIMELINES = 8
STAMP_ABSENT = -(1 << 63)            # a stamp value: absent on that timeline

_U32 = (1 << 32) - 1
_INT_EXACT = 1 << 53
_SAFE_INT = (1 << 53) - 1
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_.\-]{1,63}")


# ===========================================================================
# errors
# ===========================================================================
class WireError(Exception):
    """A frame that is not well formed. ``reason`` is one of ``truncated``,
    ``bad_magic``, ``unsupported_major``, ``too_big``, ``malformed``;
    ``close_code`` is the WebSocket close code the receiver closes with."""

    def __init__(self, reason: str, message: str, close_code: int = CLOSE_PROTOCOL_ERROR):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.close_code = close_code


class MessageError(Exception):
    """A well-formed frame whose message is invalid: the receiver answers with
    an ERROR message of this ``code`` and keeps the connection."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ===========================================================================
# frames
# ===========================================================================
def _align(n: int) -> int:
    return (n + SECTION_ALIGN - 1) & ~(SECTION_ALIGN - 1)


def canonical_json(obj) -> bytes:
    """The canonical metadata encoding: compact, keys sorted, raw UTF-8, no NaN."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


@dataclass
class Section:
    """One payload section: a raw little-endian array."""
    name: str
    dtype: int
    shape: tuple
    data: Any                       # bytes-like (a memoryview of the frame when decoded)
    encoding: int = Encoding.RAW

    @classmethod
    def from_array(cls, name: str, arr) -> "Section":
        a = np.asarray(arr)
        dt = _FROM_NP.get(np.dtype(a.dtype.name))
        if dt is None:
            raise TypeError(f"section {name!r}: no wire dtype for numpy {a.dtype}")
        a = np.ascontiguousarray(a, dtype=_NP_DTYPE[dt])
        data = memoryview(a).cast("B") if a.size else memoryview(b"")   # an empty array: no bytes
        return cls(name, int(dt), tuple(int(s) for s in a.shape), data)

    def descriptor(self) -> dict:
        d = {"name": self.name, "dtype": int(self.dtype), "shape": list(self.shape)}
        if self.encoding != Encoding.RAW:
            d["encoding"] = int(self.encoding)
        return d

    def array(self) -> np.ndarray:
        """A numpy view (no copy) of a RAW section."""
        if self.encoding != Encoding.RAW or self.dtype not in _NP_DTYPE:
            raise MessageError(ERR_UNSUPPORTED, f"section {self.name!r}: encoding "
                               f"{self.encoding}, dtype {self.dtype}")
        return np.frombuffer(self.data, dtype=_NP_DTYPE[DType(self.dtype)]).reshape(self.shape)


@dataclass
class Frame:
    type: int
    meta: dict
    sections: list = field(default_factory=list)
    flags: int = 0
    major: int = MAJOR
    minor: int = MINOR

    def section(self, name: str) -> Section | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None


def frame_parts(type_: int, meta: dict, sections=(), *, flags: int = 0, minor: int = MINOR):
    """One frame as a list of buffers that concatenate to its bytes, without
    copying the sections: the header, section table and metadata in one bytes
    object, then for each section its zero padding (if any) and its data.
    Returns ``(parts, frame_len)``. A sender can hand the list to a socket as
    it is (``sendmsg``); :func:`encode_frame` joins it."""
    sections = list(sections)
    meta = dict(meta)
    if sections:
        meta["sections"] = [s.descriptor() for s in sections]
    else:
        meta.pop("sections", None)
    mbytes = canonical_json(meta)
    if len(mbytes) > MAX_META_BYTES:
        raise WireError("too_big", f"metadata of {len(mbytes)} bytes", CLOSE_TOO_BIG)
    if len(sections) > MAX_SECTIONS:
        raise WireError("too_big", f"{len(sections)} sections", CLOSE_TOO_BIG)
    lens = [memoryview(s.data).nbytes for s in sections]
    pos = HEADER_BYTES + 4 * len(sections) + len(mbytes)
    offsets = []
    for n in lens:
        pos = _align(pos)
        offsets.append(pos)
        pos += n
    frame_len = pos
    if frame_len > MAX_FRAME_BYTES:
        raise WireError("too_big", f"frame of {frame_len} bytes", CLOSE_TOO_BIG)
    head = HEADER.pack(MAGIC, MAJOR, minor, int(type_), flags, frame_len, len(mbytes),
                       len(sections), 0) + struct.pack(f"<{len(lens)}I", *lens) + mbytes
    parts = [head]
    end = len(head)
    for s, off, n in zip(sections, offsets, lens):
        if off > end:
            parts.append(bytes(off - end))
        if n:
            parts.append(memoryview(s.data).cast("B"))
        end = off + n
    return parts, frame_len


def encode_frame(type_: int, meta: dict, sections=(), *, flags: int = 0,
                 minor: int = MINOR, as_bytearray: bool = False):
    """Encode one frame. ``meta["sections"]`` is written from ``sections``.
    ``as_bytearray=True`` returns a bytearray."""
    parts, _ = frame_parts(type_, meta, sections, flags=flags, minor=minor)
    out = b"".join(parts)
    return bytearray(out) if as_bytearray else out


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")


_JSON_STRING = re.compile(rb'"(?:[^"\\]|\\.)*"', re.S)
_OPEN = np.zeros(256, np.int8)
_OPEN[[ord("{"), ord("[")]] = 1
_OPEN[[ord("}"), ord("]")]] = -1


def json_depth(text: bytes) -> int:
    """The deepest nesting of objects and arrays in JSON text, found without
    parsing it (strings are skipped), so that a hostile peer cannot make the
    parser recurse. Linear time; about a millisecond per MiB."""
    if b"[" not in text and b"{" not in text:
        return 0
    bare = _JSON_STRING.sub(b"", text) if b'"' in text else text
    steps = _OPEN[np.frombuffer(bare, np.uint8)]
    return int(np.cumsum(steps, dtype=np.int64).max(initial=0))


def _depth(v, d=1) -> int:
    if isinstance(v, dict):
        return max([d] + [_depth(x, d + 1) for x in v.values()])
    if isinstance(v, list):
        return max([d] + [_depth(x, d + 1) for x in v])
    return d - 1


def decode_frame(buf) -> Frame:
    """Decode one frame, applying the checks of live_net.h in their order.
    Sections are zero-copy memoryviews of ``buf``. Raises :class:`WireError`."""
    mv = memoryview(buf).cast("B")
    n = mv.nbytes
    if n < 8:
        raise WireError("truncated", f"{n} bytes, fewer than 8")
    magic, major = struct.unpack_from("<IH", mv, 0)
    if magic != MAGIC:
        raise WireError("bad_magic", f"magic 0x{magic:08x}")
    if major != MAJOR:
        raise WireError("unsupported_major", f"major {major}", CLOSE_UNSUPPORTED_MAJOR)
    if n < HEADER_BYTES:
        raise WireError("truncated", f"{n} bytes, fewer than {HEADER_BYTES}")
    (_, _, minor, type_, flags, frame_len, meta_len, count, _) = HEADER.unpack_from(mv, 0)
    if frame_len > MAX_FRAME_BYTES or meta_len > MAX_META_BYTES or count > MAX_SECTIONS:
        raise WireError("too_big", f"frame_len {frame_len}, meta_len {meta_len}, "
                        f"{count} sections", CLOSE_TOO_BIG)
    if n != frame_len:
        raise WireError("truncated" if n < frame_len else "malformed",
                        f"message of {n} bytes, frame_len {frame_len}")
    table_end = HEADER_BYTES + 4 * count
    if table_end > frame_len:
        raise WireError("malformed", "section table past the end of the frame")
    lens = struct.unpack_from(f"<{count}I", mv, HEADER_BYTES)
    meta_end = table_end + meta_len
    pos = meta_end
    offsets = []
    for ln in lens:
        pos = _align(pos)
        offsets.append(pos)
        pos += ln
    if pos != frame_len:
        raise WireError("malformed", f"layout ends at {pos}, frame_len {frame_len}")
    raw = bytes(mv[table_end:meta_end])
    if json_depth(raw) > MAX_JSON_DEPTH:
        raise WireError("malformed", f"metadata nested deeper than {MAX_JSON_DEPTH}")
    try:
        meta = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise WireError("malformed", f"metadata: {type(e).__name__}: {e}") from None
    if not isinstance(meta, dict):
        raise WireError("malformed", "metadata is not a JSON object")
    if _depth(meta) > MAX_JSON_DEPTH:
        raise WireError("malformed", f"metadata nested deeper than {MAX_JSON_DEPTH}")
    sections = []
    descs = meta.get("sections")
    if count:
        if not isinstance(descs, list) or len(descs) != count:
            raise WireError("malformed", f"'sections' does not describe {count} sections")
        names = set()
        for d, off, ln in zip(descs, offsets, lens):
            if not isinstance(d, dict):
                raise WireError("malformed", "section descriptor is not an object")
            name, dt, shape = d.get("name"), d.get("dtype"), d.get("shape")
            enc = d.get("encoding", 0)
            if (not isinstance(name, str) or not _is_int(dt) or not _is_int(enc)
                    or not isinstance(shape, list)
                    or not all(_is_int(x) and x >= 0 for x in shape)):
                raise WireError("malformed", f"section descriptor {d!r}")
            if name in names:
                raise WireError("malformed", f"section name {name!r} repeated")
            names.add(name)
            if enc == Encoding.RAW and dt in DTYPE_SIZE:
                want = math.prod(shape) * DTYPE_SIZE[DType(dt)]
                if want != ln:
                    raise WireError("malformed", f"section {name!r}: {ln} bytes, shape "
                                    f"{shape} needs {want}")
            sections.append(Section(name, dt, tuple(shape), mv[off:off + ln], enc))
    elif descs not in (None, []):
        raise WireError("malformed", "'sections' given for a frame without sections")
    return Frame(type_, meta, sections, flags, major, minor)


# ===========================================================================
# field checks
# ===========================================================================
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bad(msg: str):
    raise MessageError(ERR_BAD_REQUEST, msg)


def _unsupported(msg: str):
    raise MessageError(ERR_UNSUPPORTED, msg)


def _req(meta: dict, key: str, check, what: str):
    if key not in meta:
        _bad(f"missing {key!r}")
    v = meta[key]
    if not check(v):
        _bad(f"{key!r} must be {what}, got {v!r}")
    return v


def _opt(meta: dict, key: str, check, what: str, default=None):
    v = meta.get(key)
    if v is None:
        return default
    if not check(v):
        _bad(f"{key!r} must be {what}, got {v!r}")
    return v


def _u32(v) -> bool:
    return _is_int(v) and 0 <= v <= _U32


def _uint(v) -> bool:
    return _is_int(v) and v >= 0


def _name(v) -> bool:
    return isinstance(v, str) and 0 < len(v.encode("utf-8")) <= NAME_MAX


def _str(v) -> bool:
    return isinstance(v, str)


def _obj(v) -> bool:
    return isinstance(v, dict)


def _strs(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _session_id(v) -> bool:
    return (isinstance(v, str) and 0 < len(v) <= SESSION_ID_MAX
            and all(c.isascii() and (c.isalnum() or c in "._-") for c in v))


def _identity(v) -> bool:
    """A session identity: id, script, pid, start_unix_ms."""
    return (_obj(v) and _session_id(v.get("id")) and _str(v.get("script"))
            and _uint(v.get("pid")) and _uint(v.get("start_unix_ms")))


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _stat(v) -> bool:
    return v is None or _is_num(v)


def _json_float(x):
    """A stats entry: a finite float, or None for NaN and the infinities."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _text(v) -> bool:
    return isinstance(v, str) and len(v.encode("utf-8")) <= TEXT_MAX


def _finite(v) -> bool:
    return _is_num(v) and math.isfinite(v)


def _bool(v) -> bool:
    return isinstance(v, bool)


def _ascending(v) -> bool:
    return all(a < b for a, b in zip(v, v[1:]))


def _safe_uint(v) -> bool:
    """An unsigned integer that every codec holds exactly: 0 to 2^53 - 1."""
    return _is_int(v) and 0 <= v <= _SAFE_INT


def _bounded_text(limit: int):
    return lambda v: isinstance(v, str) and 0 < len(v.encode("utf-8")) <= limit


def _int_in(lo: int, hi: int):
    return lambda v: _is_int(v) and lo <= v <= hi


def _nonneg(v) -> bool:
    return _finite(v) and v >= 0


def _list_of(check, most: int, least: int = 0):
    return lambda v: isinstance(v, list) and least <= len(v) <= most and all(check(x) for x in v)


def _section(name: str, a) -> Section:
    return a if isinstance(a, Section) else Section.from_array(name, a)


def entity_path_ok(v) -> bool:
    """An entity path: 0 to 16 segments of [A-Za-z0-9_.-]{1,63} joined by
    '/', at most 255 bytes, never '.' or '..'. The empty path is the root."""
    if not isinstance(v, str):
        return False
    if v == "":
        return True
    if len(v) > PATH_MAX or not v.isascii():
        return False
    segs = v.split("/")
    return len(segs) <= PATH_SEGMENTS_MAX and all(
        _PATH_SEGMENT.fullmatch(x) is not None and x not in (".", "..") for x in segs)


def _part_section(f: Frame, name: str, dtype: int, shape: tuple):
    """The section ``name`` of a scene part, or None; ``shape`` holds the
    rows it must have (None: any) and its rank."""
    s = f.section(name)
    if s is None:
        return None
    want = "(" + ", ".join("_" if x is None else str(x) for x in shape) + ")"
    if (s.encoding != Encoding.RAW or s.dtype != dtype or len(s.shape) != len(shape)
            or any(x is not None and x != y for x, y in zip(shape, s.shape))):
        _bad(f"{name!r} must be {_DTYPE_NAME[DType(dtype)]} {want}, got dtype {s.dtype} "
             f"shape {list(s.shape)}")
    return s


def _stamp(f: Frame, static: bool, rows: int | None = None):
    """The stamp sections of a message: ``stamp`` i64 (T,), and where ``rows``
    is given ``stamps`` i64 (rows, T), T from 1 to MAX_TIMELINES. A static
    item carries neither. Returns (stamp, stamps), each a Section or None."""
    s = f.section("stamp")
    ss = f.section("stamps") if rows is not None else None
    if s is not None and ss is not None:
        _bad("'stamp' and 'stamps' together")
    if static and (s is not None or ss is not None):
        _bad("a static item carries no stamp")
    if s is not None and (s.encoding != Encoding.RAW or s.dtype != DType.I64 or len(s.shape) != 1
                          or not 1 <= s.shape[0] <= MAX_TIMELINES):
        _bad(f"'stamp' must be i64 (T,) with T from 1 to {MAX_TIMELINES}, got dtype {s.dtype} "
             f"shape {list(s.shape)}")
    if ss is not None and (ss.encoding != Encoding.RAW or ss.dtype != DType.I64
                           or len(ss.shape) != 2 or ss.shape[0] != rows
                           or not 1 <= ss.shape[1] <= MAX_TIMELINES):
        _bad(f"'stamps' must be i64 ({rows}, T) with T from 1 to {MAX_TIMELINES}, got dtype "
             f"{ss.dtype} shape {list(ss.shape)}")
    return s, ss


class _Msg:
    TYPE: ClassVar[MsgType]

    def to_meta(self) -> dict:
        raise NotImplementedError

    def to_sections(self) -> list:
        return []

    @classmethod
    def from_frame(cls, f: Frame):
        raise NotImplementedError


@dataclass
class Unknown:
    """A message type this version does not know: receivers ignore it."""
    type: int
    meta: dict
    sections: list
