"""Wire protocol v3: frames and messages, encoded and decoded in pure Python.

The protocol is defined by ``include/caliper/live_net.h`` and
``docs/reference/wire-protocol.md``; this module follows them and mirrors
their constants (``python/tests/test_wire.py`` checks that the three agree).
It has no sockets and imports no torch: :func:`encode` turns a message object
into the bytes of one WebSocket message, :func:`decode` turns those bytes back
into a message object, and :func:`resolve_data_view` and :func:`payload_layout`
apply the data view rules a producer needs. :func:`param_value` and
:func:`selection_items` apply the rules of 3.1's controls and linked selection.
numpy is used only for payload sections (a decoded section is a zero-copy view
of the received buffer).

A frame is a 24-byte little-endian header, ``section_count`` uint32 section
lengths, a UTF-8 JSON metadata object, then the payload sections, each starting
at an offset that is a multiple of 8 from the start of the frame.
"""
from __future__ import annotations

import enum
import json
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "MAJOR", "MINOR", "MAGIC", "MsgType", "Kind", "DType", "Encoding", "WireError",
    "MessageError", "Section", "Frame", "encode_frame", "decode_frame", "canonical_json",
    "encode", "decode", "Hello", "Welcome", "Bye", "Error", "Ping", "Pong", "Sessions",
    "Attach", "Views", "Subscribe", "Unsubscribe", "ViewRequest", "ViewReply", "Snapshot",
    "Series", "Event", "Unknown", "resolve_data_view", "payload_layout",
    "Controls", "Control", "ControlReply", "Selection", "param_value", "selection_items",
]

# --- version, endpoint (caliper/live_net.h) ---------------------------------
MAJOR = 3
MINOR = 1
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

COLORMAP_VIRIDIS, COLORMAP_MAGMA, COLORMAP_RDBU = 0, 1, 2
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
RUN_NONE, RUN_RUNNING, RUN_PAUSING, RUN_PAUSED, RUN_STEPPING = (
    "none", "running", "pausing", "paused", "stepping")
RUN_STATES = (RUN_NONE, RUN_RUNNING, RUN_PAUSING, RUN_PAUSED, RUN_STEPPING)
OP_PAUSE, OP_RESUME, OP_STEP, OP_SET, OP_RUN = "pause", "resume", "step", "set", "run"
OP_INTERVENE = "intervene"                 # 3.2 draft, capability "interventions"
OPS = (OP_PAUSE, OP_RESUME, OP_STEP, OP_SET, OP_RUN, OP_INTERVENE)
PARAM_NUMBER, PARAM_INTEGER, PARAM_BOOLEAN, PARAM_CHOICE = "number", "integer", "boolean", "choice"
PARAM_TYPES = (PARAM_NUMBER, PARAM_INTEGER, PARAM_BOOLEAN, PARAM_CHOICE)
STATUS_OK, STATUS_QUEUED, STATUS_ERROR = "ok", "queued", "error"
STATUSES = (STATUS_OK, STATUS_QUEUED, STATUS_ERROR)

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
MAX_PARAMS = 64                            # 3.1
MAX_ACTIONS = 64
MAX_CHOICES = 64
TEXT_MAX = 1024
CONTROL_RATE_HZ = 30
ACTION_WAIT_MS = 600_000
MAX_ITEM_SPACES = 16
MAX_DETAIL_ITEMS = 32
MAX_DETAIL_FIELDS = 16
DETAIL_TEXT_MAX = 256
DETAILS_TIMEOUT_MS = 2000
REPLY_WAIT_MS = 60_000
RELEASE_GRACE_MS = 10_000
ORDER_WAIT_MS = 2000

_U32 = (1 << 32) - 1


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


# ===========================================================================
# messages
# ===========================================================================
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
class Hello(_Msg):
    """Client to server, the first frame of a connection."""
    TYPE: ClassVar[MsgType] = MsgType.HELLO
    role: str                                    # "viewer" or "producer"
    capabilities: list = field(default_factory=list)
    software: dict | None = None                 # {"name", "version"}, informational
    session: dict | None = None                  # producer: {"id", "script", "pid", "start_unix_ms"}

    def to_meta(self):
        return _drop_none({"role": self.role, "capabilities": list(self.capabilities),
                           "software": self.software, "session": self.session})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        role = _req(m, "role", lambda v: v in (ROLE_VIEWER, ROLE_PRODUCER), "viewer or producer")
        session = _opt(m, "session", _identity, "a session identity")
        if role == ROLE_PRODUCER and session is None:
            _bad("a producer's hello needs 'session'")
        return cls(role, _opt(m, "capabilities", _strs, "a list of strings", []),
                   _opt(m, "software", _obj, "an object"), session)


@dataclass
class Welcome(_Msg):
    """Server to client, the answer to HELLO."""
    TYPE: ClassVar[MsgType] = MsgType.WELCOME
    role: str                                    # "producer" or "hub"
    capabilities: list = field(default_factory=list)
    software: dict | None = None
    session: dict | None = None                  # a producer's own identity, or the hub's
                                                 # assignment to a connecting producer
    control: bool | None = None                  # 3.1: whether CONTROL from this client is
                                                 # accepted (to a producer: forwarded)

    def to_meta(self):
        return _drop_none({"role": self.role, "capabilities": list(self.capabilities),
                           "software": self.software, "session": self.session,
                           "control": self.control})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "role", lambda v: v in (ROLE_PRODUCER, ROLE_HUB), "producer or hub"),
                   _opt(m, "capabilities", _strs, "a list of strings", []),
                   _opt(m, "software", _obj, "an object"),
                   _opt(m, "session", _identity, "a session identity"),
                   _opt(m, "control", lambda v: isinstance(v, bool), "a boolean"))


@dataclass
class Bye(_Msg):
    TYPE: ClassVar[MsgType] = MsgType.BYE
    reason: str
    message: str = ""

    def to_meta(self):
        return {"reason": self.reason, "message": self.message}

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "reason", _str, "a string"),
                   _opt(f.meta, "message", _str, "a string", ""))


@dataclass
class Error(_Msg):
    TYPE: ClassVar[MsgType] = MsgType.ERROR
    code: str
    message: str = ""
    fatal: bool = False
    in_reply_to: int | None = None               # message type the error answers
    request_id: int | None = None
    sub: int | None = None
    view: str | None = None
    session: str | None = None

    def to_meta(self):
        return _drop_none({"code": self.code, "message": self.message, "fatal": self.fatal,
                           "in_reply_to": self.in_reply_to, "request_id": self.request_id,
                           "sub": self.sub, "view": self.view, "session": self.session})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "code", _str, "a string"), _opt(m, "message", _str, "a string", ""),
                   _opt(m, "fatal", lambda v: isinstance(v, bool), "a boolean", False),
                   _opt(m, "in_reply_to", _u32, "a message type"),
                   _opt(m, "request_id", _u32, "a uint32"), _opt(m, "sub", _u32, "a uint32"),
                   _opt(m, "view", _str, "a string"), _opt(m, "session", _str, "a string"))


@dataclass
class Ping(_Msg):
    TYPE: ClassVar[MsgType] = MsgType.PING
    nonce: int

    def to_meta(self):
        return {"nonce": self.nonce}

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "nonce", _u32, "a uint32"))


@dataclass
class Pong(Ping):
    TYPE: ClassVar[MsgType] = MsgType.PONG


@dataclass
class Sessions(_Msg):
    """Server to viewer: every session the server holds."""
    TYPE: ClassVar[MsgType] = MsgType.SESSIONS
    sessions: list                               # identities + "state", "views"

    def to_meta(self):
        return {"sessions": list(self.sessions)}

    @classmethod
    def from_frame(cls, f):
        def ok(v):
            return isinstance(v, list) and all(
                _identity(s) and s.get("state") in (SESSION_LIVE, SESSION_ENDED)
                and _uint(s.get("views"))
                and (s.get("recording") is None or _recording_ref(s["recording"]))   # 3.2 draft
                for s in v)
        return cls(_req(f.meta, "sessions", ok, "a list of sessions"))


@dataclass
class Attach(_Msg):
    """Viewer to server: attach to session ``session`` (None detaches)."""
    TYPE: ClassVar[MsgType] = MsgType.ATTACH
    session: str | None

    def to_meta(self):
        return {"session": self.session}

    @classmethod
    def from_frame(cls, f):
        if "session" not in f.meta:
            _bad("missing 'session'")
        return cls(_opt(f.meta, "session", _session_id, "a session id or null"))


def _view_entry(v) -> bool:
    if not (_obj(v) and _uint(v.get("id")) and _name(v.get("name"))
            and _is_int(v.get("kind")) and isinstance(v.get("sources"), list)):
        return False
    for s in v["sources"]:
        if not (_obj(s) and _name(s.get("name")) and isinstance(s.get("dims"), list)):
            return False
        for d in s["dims"]:
            if not (_obj(d) and _name(d.get("name")) and _uint(d.get("size"))
                    and (d.get("items") is None or _name(d["items"]))):
                return False
    return ((v.get("default") is None or _obj(v.get("default")))
            and (v.get("items") is None or _name(v["items"]))
            and (v.get("recorded") is None or _recorded(v["recorded"])))     # 3.2 draft


@dataclass
class Views(_Msg):
    """Every view of a session, resent whenever the list changes."""
    TYPE: ClassVar[MsgType] = MsgType.VIEWS
    views: list
    session: str | None = None                   # set on viewer connections
    timelines: list | None = None                # 3.2 draft, capability "timelines"

    def to_meta(self):
        meta = {"session": self.session, "views": list(self.views)}
        if self.timelines is not None:
            meta["timelines"] = list(self.timelines)
        return meta

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "views", lambda v: isinstance(v, list)
                        and all(_view_entry(x) for x in v), "a list of views"),
                   _opt(f.meta, "session", _session_id, "a session id"),
                   _opt(f.meta, "timelines", _timelines, "2 to 8 timelines, step and time first"))


def _rate(v) -> bool:
    return _is_num(v) and v > 0 and math.isfinite(v)


@dataclass
class Subscribe(_Msg):
    TYPE: ClassVar[MsgType] = MsgType.SUBSCRIBE
    sub: int
    view: str
    request_id: int
    max_rate: float | None = None                # default 60, clamped to 240
    data_view: dict | None = None                # None: the view's default

    def to_meta(self):
        return _drop_none({"sub": self.sub, "view": self.view, "request_id": self.request_id,
                           "max_rate": self.max_rate, "data_view": self.data_view})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "sub", _u32, "a uint32"), _req(m, "view", _name, "a view name"),
                   _req(m, "request_id", _u32, "a uint32"),
                   _opt(m, "max_rate", _rate, "a positive number"),
                   _opt(m, "data_view", _obj, "an object"))


@dataclass
class Unsubscribe(_Msg):
    TYPE: ClassVar[MsgType] = MsgType.UNSUBSCRIBE
    sub: int

    def to_meta(self):
        return {"sub": self.sub}

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "sub", _u32, "a uint32"))


@dataclass
class ViewRequest(_Msg):
    """A new data view for an open subscription."""
    TYPE: ClassVar[MsgType] = MsgType.VIEW_REQUEST
    request_id: int
    sub: int
    view: str
    data_view: dict

    def to_meta(self):
        return {"request_id": self.request_id, "sub": self.sub, "view": self.view,
                "data_view": self.data_view}

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "request_id", _u32, "a uint32"), _req(m, "sub", _u32, "a uint32"),
                   _req(m, "view", _name, "a view name"),
                   _req(m, "data_view", _obj, "an object"))


@dataclass
class ViewReply(_Msg):
    """The answer to SUBSCRIBE and VIEW_REQUEST: the resolved data view, or an
    error ({"code", "message"})."""
    TYPE: ClassVar[MsgType] = MsgType.VIEW_REPLY
    request_id: int
    sub: int
    view: str
    data_view: dict | None = None
    error: dict | None = None

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "sub": self.sub, "view": self.view,
                           "data_view": self.data_view, "error": self.error})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        dv = _opt(m, "data_view", _obj, "an object")
        err = _opt(m, "error", lambda v: _obj(v) and _str(v.get("code")), "{code, message}")
        if (dv is None) == (err is None):
            _bad("a view_reply carries exactly one of 'data_view' and 'error'")
        return cls(_req(m, "request_id", _u32, "a uint32"), _req(m, "sub", _u32, "a uint32"),
                   _req(m, "view", _name, "a view name"), dv, err)


def _stat(v) -> bool:
    return v is None or _is_num(v)


@dataclass
class Snapshot(_Msg):
    """One reduced snapshot of a view under one subscription.

    ``data`` is the payload (heatmap (H, W) f32, image (H, W, 4) u8, points
    (N, D) f32, histogram (B,) f32); ``aux`` the points' (N,) f32 labels or
    values. ``stats`` entries may be None (a non-finite value)."""
    TYPE: ClassVar[MsgType] = MsgType.SNAPSHOT
    sub: int
    view: str
    seq: int
    stamp_ns: int
    kind: int
    data_view: dict
    data: Any
    aux: Any = None
    colormap: int = 0
    flags: int = 0
    vmin: float = 0.0
    vmax: float = 1.0
    stats: list = field(default_factory=list)
    axes: list = field(default_factory=list)
    tile: dict | None = None
    points_dims: int | None = None
    label_count: int | None = None
    # 3.2 draft (see the 3.2 block): a bar chart's colours, labels and direction,
    # a static snapshot, its stamp, and the cursor and emptiness of a recorded one.
    colors: Any = None
    labels: list | None = None
    horizontal: bool = False
    static: bool = False
    stamp: Any = None
    cursor: int | None = None
    empty: bool = False
    # 3.2 draft, capability "programs": the shared cadence of the views' snapshots
    cadence: dict | None = None

    def to_meta(self):
        return _drop_none({"sub": self.sub, "view": self.view, "seq": self.seq,
                           "stamp_ns": self.stamp_ns, "kind": int(self.kind),
                           "data_view": self.data_view, "colormap": self.colormap,
                           "flags": self.flags, "vmin": float(self.vmin),
                           "vmax": float(self.vmax), "stats": [_json_float(x) for x in self.stats],
                           "axes": list(self.axes), "tile": self.tile,
                           "points_dims": self.points_dims, "label_count": self.label_count,
                           "labels": None if self.labels is None else list(self.labels),
                           "horizontal": True if self.horizontal else None,
                           "static": True if self.static else None, "cursor": self.cursor,
                           "empty": True if self.empty else None, "cadence": self.cadence})

    def to_sections(self):
        s = []
        if self.data is not None:
            s.append(_section("data", self.data))
        for name in ("aux", "colors", "stamp"):
            if getattr(self, name) is not None:
                s.append(_section(name, getattr(self, name)))
        return s

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        kind = _req(m, "kind", _is_int, "a kind id")
        empty = _opt(m, "empty", _bool, "a boolean", False)                  # 3.2 draft
        data = f.section("data")
        if empty:
            if f.sections:
                _bad("an empty snapshot carries no section")
        elif data is None:
            _bad("a snapshot needs a 'data' section")
        aux = f.section("aux")
        stats = _opt(m, "stats", lambda v: isinstance(v, list) and len(v) <= MAX_STATS
                     and all(_stat(x) for x in v), "a list of numbers or nulls", [])
        snap = cls(_req(m, "sub", _u32, "a uint32"), _req(m, "view", _name, "a view name"),
                   _req(m, "seq", _uint, "an unsigned integer"),
                   _req(m, "stamp_ns", _uint, "an unsigned integer"), kind,
                   _req(m, "data_view", _obj, "an object"), data, aux,
                   _opt(m, "colormap", _is_int, "an integer", 0),
                   _opt(m, "flags", _uint, "an unsigned integer", 0),
                   _opt(m, "vmin", _is_num, "a number", 0.0),
                   _opt(m, "vmax", _is_num, "a number", 1.0),
                   [math.nan if x is None else float(x) for x in stats],
                   _opt(m, "axes", lambda v: isinstance(v, list), "a list", []),
                   _opt(m, "tile", _obj, "an object"),
                   _opt(m, "points_dims", lambda v: v in (2, 3), "2 or 3"),
                   _opt(m, "label_count", _uint, "an unsigned integer"))
        snap.empty = empty
        if not empty:
            _check_payload(snap)
        _snapshot_32(f, snap)                                                 # 3.2 draft
        return snap


def _json_float(x):
    """A stats entry: a finite float, or None for NaN and the infinities."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _check_payload(s: Snapshot) -> None:
    d = s.data
    want = {Kind.HEATMAP: (DType.F32, 2), Kind.IMAGE: (DType.U8, 3),
            Kind.POINTS: (DType.F32, 2), Kind.HISTOGRAM: (DType.F32, 1),
            Kind.BARS: (DType.F32, 1)}.get(s.kind)
    if want is None:
        return                                   # an unknown kind: the caller ignores it
    if d.dtype != want[0] or len(d.shape) != want[1]:
        _bad(f"kind {s.kind}: 'data' is dtype {d.dtype} shape {list(d.shape)}")
    if s.kind == Kind.IMAGE and d.shape[2] != 4:
        _bad("an image's 'data' is (H, W, 4)")
    if s.kind == Kind.BARS and not 1 <= d.shape[0] <= MAX_BARS:               # 3.2 draft
        _bad(f"a bar chart has 1 to {MAX_BARS} bars, got {d.shape[0]}")
    if s.kind == Kind.POINTS:
        if d.shape[1] not in (2, 3) or (s.points_dims not in (None, d.shape[1])):
            _bad("points 'data' is (N, points_dims) with points_dims 2 or 3")
        s.points_dims = d.shape[1]
        if s.aux is not None and (s.aux.dtype != DType.F32 or s.aux.shape != (d.shape[0],)):
            _bad("points 'aux' is (N,) f32")


@dataclass
class Series(_Msg):
    """Series values ``start .. start + len(values)`` (absolute append indices);
    ``total`` is the number appended so far."""
    TYPE: ClassVar[MsgType] = MsgType.SERIES
    sub: int
    view: str
    start: int
    total: int
    stamp_ns: int
    values: Any
    # 3.2 draft (see the 3.2 block): the values' stamps, and for a recorded
    # series that was decimated the absolute index of each value and the bucket.
    stamp: Any = None
    stamps: Any = None
    indices: Any = None
    decimated: int | None = None
    cursor: int | None = None

    def to_meta(self):
        return _drop_none({"sub": self.sub, "view": self.view, "start": self.start,
                           "total": self.total, "stamp_ns": self.stamp_ns,
                           "decimated": self.decimated, "cursor": self.cursor})

    def to_sections(self):
        v = self.values
        out = [v if isinstance(v, Section) else
               Section.from_array("values", np.asarray(v, dtype=np.float32))]
        for name in ("stamps", "stamp", "indices"):
            if getattr(self, name) is not None and not (name == "stamp" and self.stamps is not None):
                out.append(_section(name, getattr(self, name)))
        return out

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        v = f.section("values")
        if v is None or v.dtype != DType.F32 or len(v.shape) != 1:
            _bad("a series needs a (M,) f32 'values' section")
        s = cls(_req(m, "sub", _u32, "a uint32"), _req(m, "view", _name, "a view name"),
                _req(m, "start", _uint, "an unsigned integer"),
                _req(m, "total", _uint, "an unsigned integer"),
                _req(m, "stamp_ns", _uint, "an unsigned integer"), v)
        sparse = f.section("indices") is not None                           # 3.2 draft
        if not 1 <= v.shape[0] <= SERIES_MAX_VALUES or (
                not sparse and s.start + v.shape[0] > s.total):
            _bad(f"series: {v.shape[0]} values from {s.start} with total {s.total}")
        _series_32(f, s)                                                      # 3.2 draft
        return s


@dataclass
class Event(_Msg):
    """Viewer to producer: a selection, pick or hover that Python code reads.
    Never a command."""
    TYPE: ClassVar[MsgType] = MsgType.EVENT
    view: str
    type: str                                    # "select", "pick", "hover"
    detail: dict
    sub: int | None = None
    seq: int | None = None
    data_view: dict | None = None

    def to_meta(self):
        return _drop_none({"view": self.view, "type": self.type, "detail": self.detail,
                           "sub": self.sub, "seq": self.seq, "data_view": self.data_view})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        detail = _req(m, "detail", _obj, "an object")
        for key in ("indices", "row_ids", "items"):
            lst = detail.get(key)
            if lst is not None and not (isinstance(lst, list) and len(lst) <= MAX_EVENT_INDICES
                                        and all(_uint(x) for x in lst)):
                _bad(f"detail.{key} must be at most {MAX_EVENT_INDICES} unsigned integers")
        if (detail.get("space") is None) != (detail.get("items") is None) or (
                detail.get("space") is not None and not _name(detail["space"])):
            _bad("detail.space (an item space name) and detail.items go together")
        return cls(_req(m, "view", _name, "a view name"), _req(m, "type", _str, "a string"),
                   detail, _opt(m, "sub", _u32, "a uint32"), _opt(m, "seq", _uint, "an integer"),
                   _opt(m, "data_view", _obj, "an object"))


# ---------------------------------------------------------------------------
# 3.1: controls (capability "controls") and linked selection ("selection")
# ---------------------------------------------------------------------------
def _text(v) -> bool:
    return isinstance(v, str) and len(v.encode("utf-8")) <= TEXT_MAX


def _finite(v) -> bool:
    return _is_num(v) and math.isfinite(v)


def _bool(v) -> bool:
    return isinstance(v, bool)


_INT_EXACT = 1 << 53


def _fits_type(p: dict, v) -> bool:
    """Whether ``v`` is a value of parameter ``p``'s type (its range aside)."""
    t = p["type"]
    if t == PARAM_NUMBER:
        return _finite(v)
    if t == PARAM_INTEGER:
        return _is_int(v) and abs(v) <= _INT_EXACT
    if t == PARAM_BOOLEAN:
        return _bool(v)
    return isinstance(v, str) and v in p["choices"]


def _param_entry(p) -> bool:
    if not (_obj(p) and _name(p.get("name")) and p.get("type") in PARAM_TYPES):
        return False
    t = p["type"]
    if t in (PARAM_NUMBER, PARAM_INTEGER):
        bound = _finite if t == PARAM_NUMBER else (lambda x: _is_int(x) and abs(x) <= _INT_EXACT)
        lo, hi = p.get("min"), p.get("max")
        if any(x is not None and not bound(x) for x in (lo, hi)):
            return False
        if lo is not None and hi is not None and lo > hi:
            return False
        log = p.get("log")
        if log is not None and not _bool(log):
            return False
        if log and not (lo is not None and lo > 0 and hi is not None):
            return False
    elif t == PARAM_CHOICE:
        ch = p.get("choices")
        if not (isinstance(ch, list) and 1 <= len(ch) <= MAX_CHOICES
                and all(_name(c) for c in ch) and len(set(ch)) == len(ch)):
            return False
    if p.get("help") is not None and not _text(p["help"]):
        return False
    if "value" not in p or not _fits_type(p, p["value"]):
        return False
    return p.get("pending") is None or _fits_type(p, p["pending"])


def _action_entry(a) -> bool:
    return (_obj(a) and _name(a.get("name")) and _bool(a.get("running"))
            and _bool(a.get("queued")) and (a.get("help") is None or _text(a["help"])))


def _run_obj(r) -> bool:
    return (_obj(r) and r.get("state") in RUN_STATES
            and (r.get("step") is None or _uint(r["step"]))
            and (r.get("steps_left") is None or _uint(r["steps_left"])))


def _unique_names(entries) -> bool:
    names = [e["name"] for e in entries]
    return len(set(names)) == len(names)


@dataclass
class Controls(_Msg):
    """Producer to viewer: the run state and every parameter and action the
    script registered. Sent after VIEWS, and again whole whenever any of it
    changes; a newer one replaces an unsent older one.

    ``run`` is ``{"state", "step", "steps_left"}``; a parameter is ``{"name",
    "type", "value", "pending", "min", "max", "log", "choices", "help"}``
    (``pending``: accepted, applied at the next checkpoint); an action is
    ``{"name", "running", "queued", "help"}``."""
    TYPE: ClassVar[MsgType] = MsgType.CONTROLS
    run: dict
    params: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    stamp: Any = None                            # 3.2 draft: when the state or a value changed
    interventions: dict | None = None            # 3.2 draft: {"points", "active"}

    def to_meta(self):
        meta = {"run": dict(self.run), "params": list(self.params),
                "actions": list(self.actions)}
        if self.interventions is not None:
            meta["interventions"] = self.interventions
        return meta

    def to_sections(self):
        return [] if self.stamp is None else [_section("stamp", self.stamp)]

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        c = cls(_req(m, "run", _run_obj, "a run state"),
                   _req(m, "params", lambda v: isinstance(v, list) and len(v) <= MAX_PARAMS
                        and all(_param_entry(p) for p in v) and _unique_names(v),
                        f"at most {MAX_PARAMS} parameters with unique names"),
                   _req(m, "actions", lambda v: isinstance(v, list) and len(v) <= MAX_ACTIONS
                        and all(_action_entry(a) for a in v) and _unique_names(v),
                        f"at most {MAX_ACTIONS} actions with unique names"))
        c.stamp, _ = _stamp(f, False)                                         # 3.2 draft
        c.interventions = _opt(m, "interventions", _interventions,                # 3.2 draft
                               "{points, active} of intervention points and interventions")
        return c


def _set_value(v) -> bool:
    return _finite(v) or _bool(v) or _name(v)


@dataclass
class Control(_Msg):
    """Viewer to producer: pause, resume, ``count`` steps, set parameter
    ``name`` to ``value``, or run action ``name``. Never code: a producer acts
    only on the parameters and actions its script registered. The 3.2 draft
    adds ``intervene`` (capability "interventions"), whose key ``intervene`` is
    ``{"set": {"point", "op", "from_step", "until_step"}}``, ``{"clear":
    point}`` or ``{"clear_all": true}``."""
    TYPE: ClassVar[MsgType] = MsgType.CONTROL
    request_id: int
    op: str                                      # "pause", "resume", "step", "set", "run", "intervene"
    name: str | None = None                      # set, run
    value: Any = None                            # set
    count: int | None = None                     # step: default 1
    intervene: dict | None = None                # intervene

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op, "name": self.name,
                           "value": self.value, "count": self.count,
                           "intervene": self.intervene})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        rid = _req(m, "request_id", _u32, "a uint32")
        op = _req(m, "op", lambda v: v in OPS, "pause, resume, step, set, run or intervene")
        name = _opt(m, "name", _name, "a parameter or action name")
        if op in (OP_SET, OP_RUN) and name is None:
            _bad(f"'{op}' needs 'name'")
        value = None
        if op == OP_SET:
            value = _req(m, "value", _set_value, "a number, a boolean or a choice")
        count = None
        if op == OP_STEP:
            count = _opt(m, "count", lambda v: _u32(v) and v >= 1, "a positive uint32")
        intervene = None
        if op == OP_INTERVENE:                                                 # 3.2 draft
            intervene = _req(m, "intervene", _intervene_request,
                             "exactly one of {set: {point, op, from_step, until_step}}, "
                             "{clear: point} and {clear_all: true}")
        return cls(rid, op, name if op in (OP_SET, OP_RUN) else None, value, count, intervene)


def _reply_error(v) -> bool:
    return _obj(v) and _str(v.get("code")) and (v.get("message") is None or _text(v["message"]))


@dataclass
class ControlReply(_Msg):
    """Producer to the viewer that sent a CONTROL: ``status`` "ok", "error"
    (with ``error`` = {"code", "message"}) or, for ``run``, first "queued" and
    later one more reply, "ok" or "error". ``value`` is the value a ``set``
    was accepted with, after clamping; ``message`` an action's result text."""
    TYPE: ClassVar[MsgType] = MsgType.CONTROL_REPLY
    request_id: int
    op: str
    status: str
    name: str | None = None
    value: Any = None
    message: str | None = None
    error: dict | None = None

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op, "status": self.status,
                           "name": self.name, "value": self.value, "message": self.message,
                           "error": self.error})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        status = _req(m, "status", lambda v: v in STATUSES, "ok, queued or error")
        op = _req(m, "op", lambda v: v in OPS, "a control op")
        err = _opt(m, "error", _reply_error, "{code, message}")
        if (err is None) == (status == STATUS_ERROR):
            _bad("a control_reply carries 'error' exactly when its status is 'error'")
        if status == STATUS_QUEUED and op != OP_RUN:
            _bad("only 'run' is queued")
        return cls(_req(m, "request_id", _u32, "a uint32"), op, status,
                   _opt(m, "name", _name, "a parameter or action name"),
                   _opt(m, "value", _set_value, "a number, a boolean or a choice"),
                   _opt(m, "message", _text, f"a string of at most {TEXT_MAX} bytes"), err)


def _ascending(v) -> bool:
    return all(a < b for a, b in zip(v, v[1:]))


def _cell(v) -> bool:
    return (v is None or _bool(v) or _finite(v)
            or (isinstance(v, str) and len(v.encode("utf-8")) <= DETAIL_TEXT_MAX))


def _details(d) -> bool:
    if not _obj(d):
        return False
    fields, items, values = d.get("fields"), d.get("items"), d.get("values")
    return (isinstance(fields, list) and len(fields) <= MAX_DETAIL_FIELDS
            and all(_name(x) for x in fields) and len(set(fields)) == len(fields)
            and isinstance(items, list) and len(items) <= MAX_DETAIL_ITEMS
            and all(_uint(x) for x in items)
            and isinstance(values, list) and len(values) == len(items)
            and all(isinstance(row, list) and len(row) == len(fields)
                    and all(_cell(c) for c in row) for row in values))


@dataclass
class Selection(_Msg):
    """Producer to viewer: the current selection of item space ``space``,
    ``items`` ascending without repeats (empty: nothing selected). ``view``
    names the view a viewer selected in (None: the script set it). ``details``
    is ``{"fields", "items", "values"}``, the script's fields of the first
    selected items; ``details_error`` says why they are missing."""
    TYPE: ClassVar[MsgType] = MsgType.SELECTION
    space: str
    items: list
    view: str | None = None
    truncated: bool = False
    details: dict | None = None
    details_error: str | None = None

    def to_meta(self):
        return _drop_none({"space": self.space, "items": list(self.items), "view": self.view,
                           "truncated": True if self.truncated else None,
                           "details": self.details, "details_error": self.details_error})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "space", _name, "an item space name"),
                   _req(m, "items", lambda v: isinstance(v, list) and len(v) <= MAX_EVENT_INDICES
                        and all(_uint(x) for x in v) and _ascending(v),
                        f"at most {MAX_EVENT_INDICES} ascending unsigned integers"),
                   _opt(m, "view", _name, "a view name"),
                   _opt(m, "truncated", _bool, "a boolean", False),
                   _opt(m, "details", _details, "{fields, items, values} within the limits"),
                   _opt(m, "details_error", _text, f"a string of at most {TEXT_MAX} bytes"))


def param_value(param: dict, value):
    """The value a producer accepts for ``param`` (a parameter entry of
    CONTROLS) from a CONTROL ``set``: a number or integer clamped to ``min``
    and ``max``, a boolean, or one of the choices. An integer takes a whole
    number (``64.0`` is 64). Raises :class:`MessageError` with ``bad_value``
    for a value of another type, a fraction for an integer, or an unknown
    choice."""
    t = param["type"]
    if t == PARAM_BOOLEAN or t == PARAM_CHOICE:
        if not _fits_type(param, value):
            raise MessageError(ERR_BAD_VALUE, f"{param['name']!r} takes "
                               + ("true or false" if t == PARAM_BOOLEAN else
                                  f"one of {param['choices']}") + f", not {value!r}")
        return value
    if t == PARAM_NUMBER:
        if not _finite(value):
            raise MessageError(ERR_BAD_VALUE, f"{param['name']!r} takes a number, not {value!r}")
        v = float(value)
    else:
        if not (_finite(value) and float(value).is_integer() and abs(value) <= _INT_EXACT):
            raise MessageError(ERR_BAD_VALUE, f"{param['name']!r} takes a whole number, "
                               f"not {value!r}")
        v = int(value)
    lo, hi = param.get("min"), param.get("max")
    if lo is not None and v < lo:
        v = type(v)(lo)
    if hi is not None and v > hi:
        v = type(v)(hi)
    return v


def selection_items(items) -> tuple[list, bool]:
    """Item ids as SELECTION carries them, from a select event's ``items``:
    ascending, without repeats, cut to the first MAX_EVENT_INDICES; and
    whether they were cut."""
    ids = sorted({int(i) for i in items})
    return ids[:MAX_EVENT_INDICES], len(ids) > MAX_EVENT_INDICES


@dataclass
class Unknown:
    """A message type this version does not know: receivers ignore it."""
    type: int
    meta: dict
    sections: list


MESSAGES = {cls.TYPE: cls for cls in (Hello, Welcome, Bye, Error, Ping, Pong, Sessions, Attach,
                                      Views, Subscribe, Unsubscribe, ViewRequest, ViewReply,
                                      Snapshot, Series, Event, Controls, Control, ControlReply,
                                      Selection)}


def encode(msg, *, minor: int = MINOR, as_bytearray: bool = False):
    """The frame bytes of a message object (a bytearray with ``as_bytearray``)."""
    return encode_frame(msg.TYPE, msg.to_meta(), msg.to_sections(), minor=minor,
                        as_bytearray=as_bytearray)


def encode_parts(msg, *, minor: int = MINOR):
    """A message object as :func:`frame_parts`: ``(buffers, frame_len)``."""
    return frame_parts(msg.TYPE, msg.to_meta(), msg.to_sections(), minor=minor)


def decode(buf):
    """A message object (or :class:`Unknown`) from one frame's bytes. Raises
    :class:`WireError` for a frame that is not well formed and
    :class:`MessageError` for an invalid message."""
    f = decode_frame(buf)
    cls = MESSAGES.get(f.type)
    if cls is None:
        return Unknown(f.type, f.meta, f.sections)
    return cls.from_frame(f)


# ===========================================================================
# data views
# ===========================================================================
def _dv_bad(msg: str):
    raise MessageError(ERR_BAD_DATA_VIEW, msg)


# A SNAPSHOT is one frame of at most MAX_FRAME_BYTES, and its head may take
# the fixed header, a section table of MAX_SECTIONS entries, metadata of up to
# MAX_META_BYTES and each section's alignment padding; this is what is left
# for its payload. Only a heatmap or an image (4 bytes a cell: float32, or
# RGBA8) near LIMIT_MAX_EXTENT on both sides can exceed it: points at
# LIMIT_MAX_POINTS with labels take 16 MB and a histogram at MAX_BINS 256 KiB.
MAX_PAYLOAD_BYTES = (MAX_FRAME_BYTES - MAX_META_BYTES - HEADER_BYTES
                     - MAX_SECTIONS * (4 + SECTION_ALIGN - 1))


def payload_frame_error(kind, max_extent: int, extent_for) -> str | None:
    """The data view rule that follows from MAX_FRAME_BYTES, shared by this
    reference resolver and Caliper LIB's (caliper._dataview): None when the
    payload of a data view fits one frame, else the refusal's message, which
    names the largest ``max_extent`` that would fit. ``kind`` is a Kind or its
    name; ``extent_for(m)`` returns the payload's extent with ``max_extent``
    ``m`` (and may raise when that budget breaks another rule)."""
    if kind not in (Kind.HEATMAP, Kind.IMAGE, "heatmap", "image"):
        return None

    def fits(m: int) -> bool:
        try:
            return 4 * math.prod(extent_for(m)) <= MAX_PAYLOAD_BYTES
        except Exception:
            return False
    extent = extent_for(max_extent)
    if 4 * math.prod(extent) <= MAX_PAYLOAD_BYTES:
        return None
    lo, hi = 1, max_extent
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
    return (f"a {extent[0]} by {extent[1]} payload is too large to send in one frame; "
            f"ask for a smaller max_extent, at most {lo} here")


def _merge(request: dict | None, default: dict | None) -> dict:
    """Members the request leaves out come from the view's default data view;
    budget members are merged one by one."""
    request = request or {}
    default = default or {}
    out = {}
    for key in ("source", "index", "reduce", "tile"):
        if key in request:
            out[key] = request[key]
        elif key in default:
            out[key] = default[key]
    budget = dict(default.get("budget") or {})
    rb = request.get("budget")
    if rb is not None:
        if not _obj(rb):
            _dv_bad("'budget' must be an object")
        budget.update(rb)
    out["budget"] = budget
    return out


def _budget_int(budget: dict, key: str, default: int, limit: int) -> int:
    v = budget.get(key)
    if v is None:
        return default
    if not _is_int(v) or v < 1:
        _dv_bad(f"budget.{key} must be a positive integer, got {v!r}")
    return min(v, limit)


def resolve_data_view(view: dict, request: dict | None, *,
                      retain: int = SERIES_RETAIN_MIN) -> dict:
    """The resolved form of ``request`` for the view entry ``view`` (as in a
    VIEWS message): the request's shape with every default filled in, indices
    normalised and budgets clamped. ``retain`` is the producer's series
    retention. Raises :class:`MessageError` (``bad_data_view``)."""
    if view.get("kind") == Kind.SCENE:          # 3.2 draft
        return resolve_scene_data_view(view, request)
    if request is not None and not _obj(request):
        _dv_bad("a data view is a JSON object")
    dv = _merge(request, view.get("default"))
    kind = view.get("kind")
    budget = dv["budget"]
    if kind in (Kind.SERIES, Kind.TEXT):           # 3.2 draft: a text log's is a series'
        for key in ("index", "reduce", "tile"):
            if dv.get(key):
                _dv_bad(f"a series takes no '{key}'")
        mb = budget.get("max_backfill")
        if mb is not None and (not _is_int(mb) or mb < 0):
            _dv_bad(f"budget.max_backfill must be an integer >= 0, got {mb!r}")
        return {"budget": {"max_backfill": retain if mb is None else min(mb, retain)}}
    sources = view.get("sources") or []
    if not sources:
        _dv_bad(f"view {view.get('name')!r} offers no source")
    name = dv.get("source")
    if name is None:
        src = sources[0]
    else:
        src = next((s for s in sources if s.get("name") == name), None)
        if src is None:
            _dv_bad(f"no source {name!r}")
    dims = {d["name"]: d["size"] for d in src["dims"]}
    order = [d["name"] for d in src["dims"]]

    index = dv.get("index") or {}
    if not _obj(index):
        _dv_bad("'index' must be an object")
    rindex = {}
    for dim, sel in index.items():
        if dim not in dims:
            _dv_bad(f"source {src['name']!r} has no dim {dim!r}")
        size = dims[dim]
        if _is_int(sel):
            i = sel + size if sel < 0 else sel
            if not 0 <= i < size:
                _dv_bad(f"index {sel} out of range for {dim!r} of size {size}")
            rindex[dim] = i
        elif isinstance(sel, list) and len(sel) in (2, 3) and all(
                x is None or _is_int(x) for x in sel):
            start, stop = sel[0], sel[1]
            step = sel[2] if len(sel) == 3 and sel[2] is not None else 1
            if step < 1:
                _dv_bad(f"step {step} for {dim!r}; steps are >= 1")
            start = 0 if start is None else (start + size if start < 0 else start)
            stop = size if stop is None else (stop + size if stop < 0 else stop)
            start, stop = min(max(start, 0), size), min(max(stop, 0), size)
            if stop <= start:
                _dv_bad(f"empty range {sel} for {dim!r} of size {size}")
            rindex[dim] = [start, stop, step]
        else:
            _dv_bad(f"index for {dim!r} must be an integer or [start, stop(, step)], got {sel!r}")

    reduce = dv.get("reduce") or {}
    if not _obj(reduce):
        _dv_bad("'reduce' must be an object")
    for dim, op in reduce.items():
        if dim not in dims:
            _dv_bad(f"source {src['name']!r} has no dim {dim!r}")
        if _is_int(rindex.get(dim)):
            _dv_bad(f"dim {dim!r} is indexed and cannot be reduced")
        if op not in REDUCE_OPS:
            _dv_bad(f"reduce op {op!r}; expected one of {list(REDUCE_OPS)}")

    kept = [d for d in order if not _is_int(rindex.get(d)) and d not in reduce]
    tile = dv.get("tile")
    if tile is not None:
        if kind != Kind.HEATMAP:
            _dv_bad("only a heatmap can tile")
        if tile not in kept:
            _dv_bad(f"tile dim {tile!r} is not kept")
    need = {Kind.HEATMAP: 3 if tile is not None else 2, Kind.IMAGE: 2, Kind.POINTS: 1,
            Kind.BARS: 1}.get(kind)
    if need is not None and len(kept) != need:
        _dv_bad(f"kind {kind} needs {need} kept dims, the data view keeps {kept}")

    if kind in (Kind.HEATMAP, Kind.IMAGE):
        rbudget = {"max_extent": _budget_int(budget, "max_extent", DEFAULT_MAX_EXTENT,
                                             LIMIT_MAX_EXTENT)}
        if kind == Kind.HEATMAP:
            pool = budget.get("pool") or "mean"
            if pool not in POOL_OPS:
                _dv_bad(f"budget.pool {pool!r}; expected one of {list(POOL_OPS)}")
            rbudget["pool"] = pool
    elif kind == Kind.POINTS:
        rbudget = {"max_points": _budget_int(budget, "max_points", DEFAULT_MAX_POINTS,
                                             LIMIT_MAX_POINTS)}
    else:
        rbudget = {}
    out = {"source": src["name"], "index": rindex, "reduce": dict(reduce), "budget": rbudget}
    if tile is not None:
        out["tile"] = tile
    payload_layout(view, out)                    # the result must fit the hard limits
    if "max_extent" in rbudget:
        err = payload_frame_error(kind, rbudget["max_extent"], lambda m: payload_layout(
            view, dict(out, budget=dict(rbudget, max_extent=m)))["extent"])
        if err is not None:
            _dv_bad(err)
    return out


def payload_layout(view: dict, resolved: dict) -> dict:
    """How a resolved data view's payload maps onto the source.

    Returns ``{"extent": [...], "axes": [...], "tile": {...} or None}``:
    ``extent`` is the payload size along each kept axis (heatmap rows and
    columns, image height and width, the point count; a histogram has none),
    and each axis is ``{"dim", "start", "step", "count", "size"}``. Payload
    index i of an axis stands for the selected positions
    [floor(i * count / size), ceil((i + 1) * count / size)); position p is
    source index start + p * step. A pooled axis aggregates those positions,
    a sampled axis (points) takes the first."""
    kind = view.get("kind")
    if kind in (Kind.SERIES, Kind.HISTOGRAM, Kind.TEXT):
        return {"extent": [], "axes": [], "tile": None}
    src = next(s for s in view["sources"] if s["name"] == resolved["source"])
    index, reduce, tile = resolved["index"], resolved["reduce"], resolved.get("tile")
    axes = {}
    for d in src["dims"]:
        name, size = d["name"], d["size"]
        sel = index.get(name)
        if _is_int(sel) or name in reduce:
            continue
        start, stop, step = sel if sel is not None else (0, size, 1)
        axes[name] = {"dim": name, "start": start, "step": step,
                      "count": len(range(start, stop, step))}
    budget = resolved["budget"]
    tinfo = None
    if kind == Kind.POINTS:
        (a,) = axes.values()
        a["size"] = min(a["count"], budget["max_points"])
        order = [a]
    elif kind == Kind.BARS:                                                    # 3.2 draft
        (a,) = axes.values()
        if not 1 <= a["count"] <= MAX_BARS:
            _dv_bad(f"a bar chart shows 1 to {MAX_BARS} bars, the data view selects {a['count']}")
        a["size"] = a["count"]
        order = [a]
    else:
        ext = budget["max_extent"]
        if tile is not None:
            t = axes.pop(tile)
            t["size"] = t["count"]
            cols = math.isqrt(t["count"] - 1) + 1 if t["count"] > 0 else 1
            rows = -(-t["count"] // cols)
            r, c = axes.values()
            r["size"] = min(r["count"], max(1, ext // rows))
            c["size"] = min(c["count"], max(1, ext // cols))
            tinfo = {"dim": tile, "rows": rows, "cols": cols}
            order = [t, r, c]
            extent = [rows * r["size"], cols * c["size"]]
            if max(extent) > LIMIT_MAX_EXTENT:
                _dv_bad(f"a mosaic of {extent} exceeds {LIMIT_MAX_EXTENT}")
            return {"extent": extent, "axes": order, "tile": tinfo}
        r, c = axes.values()
        r["size"] = min(r["count"], ext)
        c["size"] = min(c["count"], ext)
        order = [r, c]
    return {"extent": [a["size"] for a in order], "axes": order, "tile": tinfo}


# ===========================================================================
# PROTOCOL 3.2 DRAFT: THE SCENES SUBSET
#
# Provisional proposals of the 3.2 draft: the minor 2,
# the capability "scenes", the kind scene (6), the scene messages (0x0070 to
# 0x0073), the dtype u16 (6) and the limits below; each is one line, here and
# in the 3.2 block of include/caliper/live_net.h. The MsgType, Kind and DType
# members are the only lines outside this block.
# docs/reference/wire-protocol-3-2.md states what is built. Senders keep
# writing MINOR (1) until 3.2 is confirmed; the 3.2 golden frames carry
# MINOR_3_2.
#
# The checks of each message run in the order written here, and the C++ codec
# (src/host/net/wire.cpp) and the TypeScript one (web/src/protocol/codec.ts)
# run them in the same order, so all three refuse a frame with the same code.
# Where the draft is silent this block takes the smallest reading, and the
# reference page lists each such reading under "Readings of the draft".
# ===========================================================================
MINOR_3_2 = 2
CAP_SCENES = "scenes"
PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES = "points", "lines", "arrows", "boxes"
PART_MESH, PART_IMAGE, PART_CAMERA = "mesh", "image", "camera"
PART_TRANSFORM = "transform"
LINES_SEGMENTS, LINES_STRIPS, LINES_EDGES = "segments", "strips", "edges"
COLOR_PER_VERTEX, COLOR_PER_STRIP, COLOR_PER_EDGE = "vertex", "strip", "edge"
FORM_COLOR, FORM_DEPTH, FORM_SEGMENTATION, FORM_ENCODED = "color", "depth", "segmentation", "encoded"
FORMAT_JPEG, FORMAT_PNG = "jpeg", "png"
CONVENTION_RDF, CONVENTION_RUB = "rdf", "rub"
SPACE_3D, SPACE_2D = "3d", "2d"
PATH_MAX = 255
PATH_SEGMENTS_MAX = 16
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
MAX_TIMELINES = 8
STAMP_ABSENT = -(1 << 63)            # a stamp value: absent on that timeline
SCENE_DEFAULT_MAX_ITEMS = 100_000
SCENE_LIMIT_MAX_ITEMS = 1_048_576
SCENE_DEFAULT_MAX_VERTICES = 1_000_000
SCENE_LIMIT_MAX_VERTICES = 4_194_304
SCENE_DEFAULT_MAX_EXTENT = 1024
SCENE_LIMIT_MAX_EXTENT = 8192

PARTS = (PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES, PART_MESH, PART_IMAGE, PART_CAMERA)
FORMS = (FORM_COLOR, FORM_DEPTH, FORM_SEGMENTATION, FORM_ENCODED)
UP_AXES_3D = ("+x", "-x", "+y", "-y", "+z", "-z")
UP_AXES_2D = ("-y", "+y")
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

_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_.\-]{1,63}")


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


def _unsupported(msg: str):
    raise MessageError(ERR_UNSUPPORTED, msg)


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
_DTYPE_NAME = {DType.F32: "f32", DType.F16: "f16", DType.BF16: "bf16", DType.I64: "i64",
               DType.I32: "i32", DType.U8: "u8", DType.U16: "u16"}


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


def _section(name: str, a) -> Section:
    return a if isinstance(a, Section) else Section.from_array(name, a)


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
    whose ``params`` are the part's keys in piece 0 and empty in the others."""
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

    def to_meta(self):
        meta = {"sub": self.sub, "view": self.view, "seq": self.seq, "stamp_ns": self.stamp_ns,
                "path": self.path, "part": self.part}
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
        out = [_section(name, self.arrays[name]) for name in PART_SECTIONS.get(self.part, ())
               if self.arrays.get(name) is not None]
        if self.stamp is not None:
            out.append(_section("stamp", self.stamp))
        return out

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
        if split is not None:
            params, arrays = _piece(f, m, part, split)
        else:
            params = _part_keys(m, part)
            arrays = _DECODERS[part](f, params)
        stamp, _ = _stamp(f, static)
        if "stats" in params:
            params["stats"] = [math.nan if x is None else float(x) for x in params["stats"]]
        return cls(sub, view, seq, stamp_ns, path, part, arrays, params, static, state, axes,
                   items, stamp, cursor, split)


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


MESSAGES[SceneUpdate.TYPE] = SceneUpdate
MESSAGES[SceneTransforms.TYPE] = SceneTransforms
MESSAGES[SceneClear.TYPE] = SceneClear
MESSAGES[SceneClasses.TYPE] = SceneClasses
__all__ += ["SceneUpdate", "SceneTransforms", "SceneClear", "SceneClasses", "entity_path_ok",
            "scene_view_info", "resolve_scene_data_view"]


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
        _dv_bad("a data view is a JSON object")
    if view.get("default") not in (None, {}, SCENE_DATA_VIEW):
        _dv_bad("a scene's own default data view is not built yet")
    if request not in (None, {}, SCENE_DATA_VIEW):
        _dv_bad("only the default data view of a scene is built yet: send {} or none; "
                "entity filters and budgets come in a later increment")
    return json.loads(json.dumps(SCENE_DATA_VIEW))


# ---------------------------------------------------------------------------
# 3.2 draft: timelines and stamps on the 3.0 and 3.1 messages, text logs, bar
# charts, split updates and the recording messages
# ---------------------------------------------------------------------------
CAP_TIMELINES = "timelines"
CAP_TEXT = "text"
CAP_BARS = "bars"
CAP_RECORDINGS = "recordings"
ERR_UNKNOWN_RECORDING = "unknown_recording"
ERR_BAD_RECORDING = "bad_recording"
ERR_NOT_SEEKABLE = "not_seekable"
ERR_BAD_CURSOR = "bad_cursor"
TIMELINE_SEQUENCE, TIMELINE_TIME = "sequence", "time"
LEVEL_DEBUG, LEVEL_INFO, LEVEL_WARNING, LEVEL_ERROR = 0, 1, 2, 3
CURSOR_SEEK, CURSOR_PLAY, CURSOR_PAUSE, CURSOR_STEP, CURSOR_FOLLOW = (
    "seek", "play", "pause", "step", "follow")
CURSOR_OPS = (CURSOR_SEEK, CURSOR_PLAY, CURSOR_PAUSE, CURSOR_STEP, CURSOR_FOLLOW)
PACE_RECORDED, PACE_EVEN = "recorded", "even"
TIME_FOLLOWING, TIME_PAUSED, TIME_PLAYING = "following", "paused", "playing"
STAMP_TEXT_MAX = 20
MAX_PIECES = 16
MAX_UPDATE_BYTES = 536_870_912
MAX_INCOMPLETE_UPDATES = 2
TEXT_LINE_MAX = 4096
TEXT_MAX_LINES = 65536
TEXT_RETAIN_MIN = 4096
MAX_BARS = 4096
MAX_RECORDINGS = 1024
MAX_RECORDED = 64
MAX_CURSOR_STEP = 1000
CURSOR_RATE_HZ = 30
TIME_STATE_RATE_HZ = 30
MIN_SPEED = 0.0625
MAX_SPEED = 64
RECORDING_LINGER_MS = 600_000

_TIMELINE_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,63}")
_STAMP_TEXT = re.compile(r"-?[0-9]+")


def _timeline_name(v) -> bool:
    return isinstance(v, str) and _TIMELINE_NAME.fullmatch(v) is not None


def stamp_text_ok(v) -> bool:
    """Stamp text: decimal digits with an optional leading minus, at most 20
    characters, a value of a signed 64-bit integer."""
    return (isinstance(v, str) and 0 < len(v) <= STAMP_TEXT_MAX and v.isascii()
            and _STAMP_TEXT.fullmatch(v) is not None and -(1 << 63) <= int(v) < (1 << 63))


def _timelines(v) -> bool:
    """The VIEWS key "timelines": 2 to 8 {name, kind, epoch?}, step and time first."""
    if not (isinstance(v, list) and 2 <= len(v) <= MAX_TIMELINES):
        return False
    for t in v:
        if not (_obj(t) and _timeline_name(t.get("name"))
                and t.get("kind") in (TIMELINE_SEQUENCE, TIMELINE_TIME)
                and (t.get("epoch") is None or _bool(t["epoch"]))):
            return False
    names = [t["name"] for t in v]
    return (len(set(names)) == len(names) and (v[0]["name"], v[0]["kind"]) == ("step", "sequence")
            and (v[1]["name"], v[1]["kind"]) == ("time", "time"))


def _recorded(v) -> bool:
    return isinstance(v, list) and len(v) <= MAX_RECORDED and all(_obj(x) for x in v)


def _recording_ref(v) -> bool:
    """A session's key "recording": {id, seekable, complete}."""
    return (_obj(v) and _session_id(v.get("id")) and _bool(v.get("seekable"))
            and _bool(v.get("complete")))


def _snapshot_32(f: Frame, snap) -> None:
    """The 3.2 additions to SNAPSHOT, checked after the 3.1 ones: a bar
    chart's colours, labels and direction, then static, cursor and stamp."""
    m = f.meta
    if snap.kind == Kind.BARS and not snap.empty:
        b = snap.data.shape[0]
        snap.colors = _part_section(f, "colors", DType.U8, (b, 4))
        snap.labels = _opt(m, "labels", lambda v: isinstance(v, list) and len(v) == b and all(
            isinstance(x, str) and len(x.encode("utf-8")) <= NAME_MAX for x in v),
            f"{b} labels of at most {NAME_MAX} bytes")
        snap.horizontal = _opt(m, "horizontal", _bool, "a boolean", False)
    snap.static = _opt(m, "static", _bool, "a boolean", False)
    snap.cursor = _opt(m, "cursor", _u32, "a uint32")
    snap.cadence = _opt(m, "cadence", _obj, "an object")
    snap.stamp, _ = _stamp(f, snap.static)


def _series_32(f: Frame, s) -> None:
    """The 3.2 additions to SERIES: indices and decimated (together), cursor,
    then the stamps."""
    m = f.meta
    n = s.values.shape[0]
    s.indices = _part_section(f, "indices", DType.I64, (n,))
    s.decimated = _opt(m, "decimated", lambda v: _is_int(v) and v >= 1, "an integer >= 1")
    if (s.indices is None) != (s.decimated is None):
        _bad("'indices' and 'decimated' go together")
    s.cursor = _opt(m, "cursor", _u32, "a uint32")
    s.stamp, s.stamps = _stamp(f, False, n)


@dataclass
class Text(_Msg):
    """3.2 draft: new lines of a text log, appended as a series' values are:
    lines ``start .. start + M`` of ``total``; line i is bytes ``offsets[i]``
    to ``offsets[i + 1]`` of ``text`` (UTF-8), at level ``levels[i]`` (0 debug,
    1 info, 2 warning, 3 error). :meth:`from_lines` builds one."""
    TYPE: ClassVar[MsgType] = MsgType.TEXT
    sub: int
    view: str
    start: int
    total: int
    stamp_ns: int
    text: Any
    offsets: Any
    levels: Any
    dropped: int | None = None
    stamp: Any = None
    stamps: Any = None
    cursor: int | None = None

    @classmethod
    def from_lines(cls, sub, view, start, total, stamp_ns, lines, levels, **kw) -> "Text":
        data = [x.encode("utf-8") for x in lines]
        offsets = np.zeros(len(data) + 1, np.int32)
        offsets[1:] = np.cumsum([len(x) for x in data])
        return cls(sub, view, start, total, stamp_ns, np.frombuffer(b"".join(data), np.uint8),
                   offsets, np.asarray(levels, np.uint8), **kw)

    def lines(self) -> list:
        t = bytes(self.text.data if isinstance(self.text, Section) else np.asarray(self.text))
        o = (self.offsets.array() if isinstance(self.offsets, Section)
             else np.asarray(self.offsets)).tolist()
        return [t[a:b].decode("utf-8") for a, b in zip(o, o[1:])]

    def to_meta(self):
        return _drop_none({"sub": self.sub, "view": self.view, "start": self.start,
                           "total": self.total, "stamp_ns": self.stamp_ns,
                           "dropped": self.dropped, "cursor": self.cursor})

    def to_sections(self):
        out = [_section("text", self.text), _section("offsets", self.offsets),
               _section("levels", self.levels)]
        if self.stamps is not None:
            out.append(_section("stamps", self.stamps))
        elif self.stamp is not None:
            out.append(_section("stamp", self.stamp))
        return out

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        t = f.section("text")
        if t is None or t.encoding != Encoding.RAW or t.dtype != DType.U8 or len(t.shape) != 1:
            _bad("a text message needs a (L,) u8 'text' section")
        o = f.section("offsets")
        if (o is None or o.encoding != Encoding.RAW or o.dtype != DType.I32 or len(o.shape) != 1
                or o.shape[0] < 2):
            _bad("a text message needs an (M + 1,) i32 'offsets' section, M >= 1")
        n = o.shape[0] - 1
        if n > TEXT_MAX_LINES:
            _bad(f"{n} lines, more than {TEXT_MAX_LINES} in one message")
        lv = _part_section(f, "levels", DType.U8, (n,))
        if lv is None:
            _bad("a text message needs an (M,) u8 'levels' section")
        msg = cls(_req(m, "sub", _u32, "a uint32"), _req(m, "view", _name, "a view name"),
                  _req(m, "start", _uint, "an unsigned integer"),
                  _req(m, "total", _uint, "an unsigned integer"),
                  _req(m, "stamp_ns", _uint, "an unsigned integer"), t, o, lv,
                  _opt(m, "dropped", _uint, "an unsigned integer"))
        msg.cursor = _opt(m, "cursor", _u32, "a uint32")
        if msg.start + n > msg.total:
            _bad(f"text: {n} lines from {msg.start} with total {msg.total}")
        offs = o.array().astype(np.int64)
        if offs[0] != 0 or offs[-1] != t.shape[0]:
            _bad(f"'offsets' must start at 0 and end at the {t.shape[0]} bytes of 'text'")
        steps = np.diff(offs)
        if steps.size and int(steps.min()) < 0:
            _bad("'offsets' must not decrease")
        if steps.size and int(steps.max()) > TEXT_LINE_MAX:
            _bad(f"a line is at most {TEXT_LINE_MAX} bytes")
        if lv.array().size and int(lv.array().max()) > LEVEL_ERROR:
            _bad("a level is 0 (debug), 1 (info), 2 (warning) or 3 (error)")
        raw = bytes(t.data)
        for a, b in zip(offs.tolist(), offs[1:].tolist()):
            try:
                raw[a:b].decode("utf-8")
            except UnicodeDecodeError:
                _bad(f"line {a}..{b} of 'text' is not UTF-8")
        msg.stamp, msg.stamps = _stamp(f, False, n)
        return msg


def _recording_entry(e) -> bool:
    if not (_obj(e) and _session_id(e.get("id")) and _str(e.get("title")) and _str(e.get("script"))
            and _uint(e.get("bytes")) and _uint(e.get("items")) and _bool(e.get("complete"))):
        return False
    for key in ("created_unix_ms", "ended_unix_ms"):
        if e.get(key) is not None and not _uint(e[key]):
            return False
    tl = e.get("timelines")
    return isinstance(tl, list) and len(tl) <= MAX_TIMELINES and all(
        _obj(t) and _timeline_name(t.get("name"))
        and t.get("kind") in (TIMELINE_SEQUENCE, TIMELINE_TIME)
        and (t.get("epoch") is None or _bool(t["epoch"]))
        and stamp_text_ok(t.get("min")) and stamp_text_ok(t.get("max")) for t in tl)


@dataclass
class Recordings(_Msg):
    """3.2 draft, server to viewer: the recordings the server can open."""
    TYPE: ClassVar[MsgType] = MsgType.RECORDINGS
    recordings: list

    def to_meta(self):
        return {"recordings": list(self.recordings)}

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "recordings", lambda v: isinstance(v, list)
                        and len(v) <= MAX_RECORDINGS and all(_recording_entry(e) for e in v),
                        f"a list of at most {MAX_RECORDINGS} recordings"))


@dataclass
class RecordingOpen(_Msg):
    """3.2 draft, viewer to server: open a recording as an ended session."""
    TYPE: ClassVar[MsgType] = MsgType.RECORDING_OPEN
    request_id: int
    recording: str

    def to_meta(self):
        return {"request_id": self.request_id, "recording": self.recording}

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "request_id", _u32, "a uint32"),
                   _req(m, "recording", _session_id, "a recording id"))


def _cursor_bad(msg: str):
    raise MessageError(ERR_BAD_CURSOR, msg)


def _speed(v) -> bool:
    return _finite(v) and MIN_SPEED <= v <= MAX_SPEED


def _loop(v) -> bool:
    return (isinstance(v, list) and len(v) == 2 and all(stamp_text_ok(x) for x in v)
            and int(v[0]) <= int(v[1]))


@dataclass
class Cursor(_Msg):
    """3.2 draft, viewer to server: move this viewer's cursor. Only the
    fields of the op are kept: seek takes timeline and value, play timeline,
    speed, pace and loop, step timeline and count. A request_id or op that is
    missing or wrong is bad_request; any other field, bad_cursor."""
    TYPE: ClassVar[MsgType] = MsgType.CURSOR
    request_id: int
    op: str
    timeline: str | None = None
    value: str | None = None
    count: int | None = None
    speed: float | None = None
    pace: str | None = None
    loop: list | None = None

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op,
                           "timeline": self.timeline, "value": self.value, "count": self.count,
                           "speed": None if self.speed is None else float(self.speed),
                           "pace": self.pace, "loop": self.loop})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        c = cls(_req(m, "request_id", _u32, "a uint32"),
                _req(m, "op", lambda v: v in CURSOR_OPS, "seek, play, pause, step or follow"))

        def get(key, ok, what, need):
            v = m.get(key)
            if v is None:
                if need:
                    _cursor_bad(f"'{c.op}' needs {key!r}")
                return None
            if not ok(v):
                _cursor_bad(f"{key!r} must be {what}, got {v!r}")
            return v
        if c.op in (CURSOR_SEEK, CURSOR_PLAY, CURSOR_STEP):
            c.timeline = get("timeline", _timeline_name, "a timeline name", True)
        if c.op == CURSOR_SEEK:
            c.value = get("value", stamp_text_ok, "stamp text", True)
        if c.op == CURSOR_STEP:
            c.count = get("count", lambda v: _is_int(v) and v != 0 and abs(v) <= MAX_CURSOR_STEP,
                          f"an integer from -{MAX_CURSOR_STEP} to {MAX_CURSOR_STEP}, not 0", True)
        if c.op == CURSOR_PLAY:
            c.speed = get("speed", _speed, f"a number from {MIN_SPEED} to {MAX_SPEED}", False)
            c.pace = get("pace", lambda v: v in (PACE_RECORDED, PACE_EVEN), "recorded or even",
                         False)
            c.loop = get("loop", _loop, "[start, end] as stamp text", False)
        return c


def _ranges(v) -> bool:
    return _obj(v) and all(_timeline_name(k) and _loop(r) for k, r in v.items())


@dataclass
class TimeState(_Msg):
    """3.2 draft, server to viewer: the state of the viewer's cursor.
    ``cursor`` is stamp text, None while following (written as null)."""
    TYPE: ClassVar[MsgType] = MsgType.TIME_STATE
    session: str
    generation: int
    state: str
    timeline: str
    cursor: str | None
    ranges: dict
    live: bool
    request_id: int | None = None
    speed: float | None = None
    pace: str | None = None
    loop: list | None = None

    def to_meta(self):
        meta = _drop_none({"request_id": self.request_id, "session": self.session,
                           "generation": self.generation, "state": self.state,
                           "timeline": self.timeline, "ranges": dict(self.ranges),
                           "live": self.live,
                           "speed": None if self.speed is None else float(self.speed),
                           "pace": self.pace, "loop": self.loop})
        meta["cursor"] = self.cursor
        return meta

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        request_id = _opt(m, "request_id", _u32, "a uint32")
        t = cls(_req(m, "session", _session_id, "a session id"),
                _req(m, "generation", _u32, "a uint32"),
                _req(m, "state", lambda v: v in (TIME_FOLLOWING, TIME_PAUSED, TIME_PLAYING),
                     "following, paused or playing"),
                _req(m, "timeline", _timeline_name, "a timeline name"),
                _opt(m, "cursor", stamp_text_ok, "stamp text or null"),
                {}, False, request_id,
                _opt(m, "speed", _speed, f"a number from {MIN_SPEED} to {MAX_SPEED}"),
                _opt(m, "pace", lambda v: v in (PACE_RECORDED, PACE_EVEN), "recorded or even"),
                _opt(m, "loop", _loop, "[start, end] as stamp text"))
        t.ranges = _req(m, "ranges", _ranges, "an object of [min, max] stamp text per timeline")
        t.live = _req(m, "live", _bool, "a boolean")
        if (t.state == TIME_FOLLOWING) != (t.cursor is None):
            _bad("the cursor is null exactly while following")
        return t


for _cls in (Text, Recordings, RecordingOpen, Cursor, TimeState):
    MESSAGES[_cls.TYPE] = _cls
__all__ += ["Text", "Recordings", "RecordingOpen", "Cursor", "TimeState", "stamp_text_ok"]


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


def split_update(u: SceneUpdate, update: int, max_frame: int = MAX_FRAME_BYTES) -> list:
    """The pieces of scene update ``u`` (split counter ``update``), each frame
    within ``max_frame`` bytes, or ``[u]`` when it fits one frame. Rows are cut
    along the first axis of each array in the part's order; piece 0 carries
    the part's keys and ``rows``, every piece the stamp. Raises ValueError
    for an update over MAX_UPDATE_BYTES or one that needs more than MAX_PIECES
    pieces."""
    try:
        _, size = frame_parts(u.TYPE, u.to_meta(), u.to_sections())
    except WireError:                                # over MAX_FRAME_BYTES: it does not fit
        size = max_frame + 1
    if size <= max_frame:
        return [u]
    arrays = []
    for name in PART_SECTIONS[u.part]:
        a = u.arrays.get(name)
        if a is not None:
            arrays.append((name, a.array() if isinstance(a, Section) else np.asarray(a)))
    total = sum(a.nbytes for _, a in arrays)
    if total > MAX_UPDATE_BYTES:
        raise ValueError(f"a scene update of {total} bytes, more than {MAX_UPDATE_BYTES}")
    # A bound on every piece's head: the header, the section table, padding,
    # and metadata with the part's keys, the largest "split" key and the
    # descriptors of the whole arrays.
    rows = {name: int(a.shape[0]) for name, a in arrays}
    probe = u.to_meta()
    probe["split"] = {"update": update, "piece": MAX_PIECES - 1, "pieces": MAX_PIECES,
                      "first": rows, "rows": rows}
    sections = [Section.from_array(name, a[:0]) for name, a in arrays]
    probe["sections"] = [dict(x.descriptor(), shape=[int(n) for n in a.shape])
                         for x, (_, a) in zip(sections, arrays)]
    nsec = len(arrays) + (u.stamp is not None)
    stamp_bytes = 0 if u.stamp is None else 8 * MAX_TIMELINES + SECTION_ALIGN
    head = HEADER_BYTES + 4 * nsec + len(canonical_json(probe)) + 64 + SECTION_ALIGN * nsec
    room = max_frame - head - stamp_bytes
    if room <= 0:
        raise ValueError(f"a frame of {max_frame} bytes leaves no room for rows")
    pieces, cur, used = [], {}, 0
    for name, a in arrays:
        rowbytes = max(1, a.nbytes // a.shape[0]) if a.shape[0] else 1
        r = 0
        while r < a.shape[0]:
            take = min(a.shape[0] - r, max(0, (room - used) // rowbytes))
            if take == 0:
                if not cur:
                    raise ValueError(f"one row of {name!r} needs more than {room} bytes")
                pieces.append(cur)
                cur, used = {}, 0
                continue
            cur[name] = (r, a[r:r + take])
            used += take * rowbytes
            r += take
    if cur:
        pieces.append(cur)
    if not 2 <= len(pieces) <= MAX_PIECES:
        raise ValueError(f"the update needs {len(pieces)} pieces; a split update has 2 to "
                         f"{MAX_PIECES}")
    out = []
    for k, cut in enumerate(pieces):
        sp = {"update": update, "piece": k, "pieces": len(pieces),
              "first": {name: int(r) for name, (r, _) in cut.items()}}
        if k == 0:
            sp["rows"] = rows
        out.append(SceneUpdate(u.sub, u.view, u.seq, u.stamp_ns, u.path, u.part,
                               {name: a for name, (_, a) in cut.items()},
                               dict(u.params) if k == 0 else {}, u.static, u.state, u.axes,
                               u.items, u.stamp, u.cursor, sp))
        if frame_parts(u.TYPE, out[-1].to_meta(), out[-1].to_sections())[1] > max_frame:
            raise ValueError(f"piece {k} would pass {max_frame} bytes")
    return out


__all__ += ["SplitUpdates", "split_update"]


# ---------------------------------------------------------------------------
# 3.2 draft: GPU programs and interventions (design/gpu-programs.md, Wire Path)
# ---------------------------------------------------------------------------
CAP_PROGRAMS = "programs"
CAP_INTERVENTIONS = "interventions"
PROGRAM_SET, PROGRAM_REMOVE, PROGRAM_PARAMS = "set", "remove", "params"
SECTION_WGSL = "wgsl"
INTERVENE_ABLATE = "ablate"
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
OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR = "geometry", "pixels", "tensor"
STATUS_OPERATOR = "operator"
PROGRAM_STATES = ("waiting", "review", "compiled", "running", "refused", "stopped")
PROGRAM_ARRAYS = ("positions", "colors", "values", "radii")
INTERVENTION_STATES = ("pending", "active", "ended")
_SAFE_INT = (1 << 53) - 1
_ID_TAIL = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
_DIGITS = frozenset("0123456789")
_PARAM_HEAD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_PARAM_TAIL = _PARAM_HEAD | _DIGITS
_HEX = frozenset("0123456789abcdef")


def _safe_uint(v) -> bool:
    """An unsigned integer that every codec holds exactly: 0 to 2^53 - 1."""
    return _is_int(v) and 0 <= v <= _SAFE_INT


def _spec_id(v: str) -> bool:
    return bool(v) and "a" <= v[0] <= "z" and all(c in _ID_TAIL for c in v)


def _program_id(v) -> bool:
    """A program id: 1 to 55 bytes of [a-z0-9_-], the first a letter, or the id of an
    operator, ``operator.<id>.<n>``: a spec id, a dot and 1 to 5 digits."""
    if not (isinstance(v, str) and 0 < len(v) <= PROGRAM_ID_MAX and v.isascii()):
        return False
    if v.startswith("operator."):
        spec, dot, n = v[len("operator."):].rpartition(".")
        return bool(dot) and _spec_id(spec) and 1 <= len(n) <= 5 and all(c in _DIGITS for c in n)
    return _spec_id(v)


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


def _bounded_text(limit: int):
    return lambda v: isinstance(v, str) and 0 < len(v.encode("utf-8")) <= limit


def _int_in(lo: int, hi: int):
    return lambda v: _is_int(v) and lo <= v <= hi


def _nonneg(v) -> bool:
    return _finite(v) and v >= 0


def _list_of(check, most: int, least: int = 0):
    return lambda v: isinstance(v, list) and least <= len(v) <= most and all(check(x) for x in v)


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
    answers with PROGRAM_STATE (``design/gpu-programs.md``, Wire Path)."""
    TYPE: ClassVar[MsgType] = MsgType.PROGRAM
    request_id: int
    op: str                                      # "set" or "remove"
    id: str
    declaration: dict | None = None              # set
    wgsl: str | None = None                      # set: the section "wgsl"
    sha256: str | None = None                    # set: of wgsl, in lowercase hex
    scene: str | None = None                     # set: a geometry output's scene
    params: dict | None = None                   # params: the names and their numbers

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op, "id": self.id,
                           "declaration": self.declaration, "scene": self.scene,
                           "sha256": self.sha256, "params": self.params})

    def to_sections(self):
        if self.wgsl is None:
            return []
        return [Section.from_array(SECTION_WGSL,
                                   np.frombuffer(self.wgsl.encode("utf-8"), dtype=np.uint8))]

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
                ("memory_limit_bytes", _safe_uint), ("error", _program_error))


def _status_item(v) -> bool:
    """A status item of PROGRAM_STATE: ``id``, ``kind`` and ``state``, and the
    optional keys of ``_STATUS_KEYS`` and ``sha256``, null or absent when
    unknown. ``host`` is the app's and is ignored here."""
    if not (_obj(v) and _program_id(v.get("id"))
            and v.get("kind") in (OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR, STATUS_OPERATOR)
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
    """CONTROL's ``intervene``: exactly one of ``set``, ``clear`` and ``clear_all``."""
    if not _obj(v) or sum(v.get(k) is not None for k in ("set", "clear", "clear_all")) != 1:
        return False
    if v.get("set") is not None:
        st = v["set"]
        return (_obj(st) and _point(st.get("point")) and _name(st.get("op"))
                and _opt_step(st.get("from_step")) and _opt_step(st.get("until_step")))
    if v.get("clear") is not None:
        return _point(v["clear"])
    return v["clear_all"] is True


def _point_entry(v) -> bool:
    shape = v.get("shape") if _obj(v) else None
    return (_obj(v) and _point(v.get("point")) and v.get("kind") in ("module", "head")
            and _point(v.get("module")) and (v.get("head") is None or _u32(v["head"]))
            and (v.get("type") is None or _name(v["type"]))
            and (shape is None or _list_of(_safe_uint, MAX_DIMS)(shape)))


def _active_entry(v) -> bool:
    return (_obj(v) and _point(v.get("point")) and _name(v.get("op"))
            and all(_opt_step(v.get(k)) for k in ("from_step", "until_step", "since_step"))
            and v.get("state") in INTERVENTION_STATES)


def _interventions(v) -> bool:
    """CONTROLS' ``interventions``: ``points`` (at most ``MAX_POINTS_LISTED``)
    and ``active`` (at most ``MAX_INTERVENTIONS``)."""
    return (_obj(v) and _list_of(_point_entry, MAX_POINTS_LISTED)(v.get("points"))
            and _list_of(_active_entry, MAX_INTERVENTIONS)(v.get("active")))


for _cls in (Program, ProgramState):
    MESSAGES[_cls.TYPE] = _cls
__all__ += ["Program", "ProgramState"]
