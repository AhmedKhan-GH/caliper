"""Wire protocol v3: frames and messages, encoded and decoded in pure Python.

The protocol is defined by ``include/caliper/live_net.h`` and
``docs/references/wire-protocol.md``; this module follows them and mirrors
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


class Kind(enum.IntEnum):          # CaliperLiveKind
    HEATMAP = 1
    IMAGE = 2
    POINTS = 3
    SERIES = 4
    HISTOGRAM = 5


class DType(enum.IntEnum):         # CaliperDType
    F32 = 0
    F16 = 1
    BF16 = 2
    I64 = 3
    I32 = 4
    U8 = 5


class Encoding(enum.IntEnum):
    RAW = 0
    ARROW_IPC = 1                  # reserved


DTYPE_SIZE = {DType.F32: 4, DType.F16: 2, DType.BF16: 2, DType.I64: 8, DType.I32: 4,
              DType.U8: 1}
# numpy views of a section; bf16 has no numpy type and is viewed as raw uint16.
_NP_DTYPE = {DType.F32: "<f4", DType.F16: "<f2", DType.BF16: "<u2", DType.I64: "<i8",
             DType.I32: "<i4", DType.U8: "u1"}
_FROM_NP = {np.dtype("float32"): DType.F32, np.dtype("float16"): DType.F16,
            np.dtype("int64"): DType.I64, np.dtype("int32"): DType.I32,
            np.dtype("uint8"): DType.U8}

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
OPS = (OP_PAUSE, OP_RESUME, OP_STEP, OP_SET, OP_RUN)
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
        return cls(name, int(dt), tuple(int(s) for s in a.shape), memoryview(a).cast("B"))

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
                and _uint(s.get("views")) for s in v)
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
            and (v.get("items") is None or _name(v["items"])))


@dataclass
class Views(_Msg):
    """Every view of a session, resent whenever the list changes."""
    TYPE: ClassVar[MsgType] = MsgType.VIEWS
    views: list
    session: str | None = None                   # set on viewer connections

    def to_meta(self):
        return {"session": self.session, "views": list(self.views)}

    @classmethod
    def from_frame(cls, f):
        return cls(_req(f.meta, "views", lambda v: isinstance(v, list)
                        and all(_view_entry(x) for x in v), "a list of views"),
                   _opt(f.meta, "session", _session_id, "a session id"))


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

    def to_meta(self):
        return _drop_none({"sub": self.sub, "view": self.view, "seq": self.seq,
                           "stamp_ns": self.stamp_ns, "kind": int(self.kind),
                           "data_view": self.data_view, "colormap": self.colormap,
                           "flags": self.flags, "vmin": float(self.vmin),
                           "vmax": float(self.vmax), "stats": [_json_float(x) for x in self.stats],
                           "axes": list(self.axes), "tile": self.tile,
                           "points_dims": self.points_dims, "label_count": self.label_count})

    def to_sections(self):
        s = [self.data if isinstance(self.data, Section) else Section.from_array("data", self.data)]
        if self.aux is not None:
            s.append(self.aux if isinstance(self.aux, Section) else Section.from_array("aux", self.aux))
        return s

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        kind = _req(m, "kind", _is_int, "a kind id")
        data = f.section("data")
        if data is None:
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
        _check_payload(snap)
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
            Kind.POINTS: (DType.F32, 2), Kind.HISTOGRAM: (DType.F32, 1)}.get(s.kind)
    if want is None:
        return                                   # an unknown kind: the caller ignores it
    if d.dtype != want[0] or len(d.shape) != want[1]:
        _bad(f"kind {s.kind}: 'data' is dtype {d.dtype} shape {list(d.shape)}")
    if s.kind == Kind.IMAGE and d.shape[2] != 4:
        _bad("an image's 'data' is (H, W, 4)")
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

    def to_meta(self):
        return {"sub": self.sub, "view": self.view, "start": self.start, "total": self.total,
                "stamp_ns": self.stamp_ns}

    def to_sections(self):
        v = self.values
        return [v if isinstance(v, Section) else
                Section.from_array("values", np.asarray(v, dtype=np.float32))]

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
        if not 1 <= v.shape[0] <= SERIES_MAX_VALUES or s.start + v.shape[0] > s.total:
            _bad(f"series: {v.shape[0]} values from {s.start} with total {s.total}")
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

    def to_meta(self):
        return {"run": dict(self.run), "params": list(self.params),
                "actions": list(self.actions)}

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        return cls(_req(m, "run", _run_obj, "a run state"),
                   _req(m, "params", lambda v: isinstance(v, list) and len(v) <= MAX_PARAMS
                        and all(_param_entry(p) for p in v) and _unique_names(v),
                        f"at most {MAX_PARAMS} parameters with unique names"),
                   _req(m, "actions", lambda v: isinstance(v, list) and len(v) <= MAX_ACTIONS
                        and all(_action_entry(a) for a in v) and _unique_names(v),
                        f"at most {MAX_ACTIONS} actions with unique names"))


def _set_value(v) -> bool:
    return _finite(v) or _bool(v) or _name(v)


@dataclass
class Control(_Msg):
    """Viewer to producer: pause, resume, ``count`` steps, set parameter
    ``name`` to ``value``, or run action ``name``. Never code: a producer acts
    only on the parameters and actions its script registered."""
    TYPE: ClassVar[MsgType] = MsgType.CONTROL
    request_id: int
    op: str                                      # "pause", "resume", "step", "set", "run"
    name: str | None = None                      # set, run
    value: Any = None                            # set
    count: int | None = None                     # step: default 1

    def to_meta(self):
        return _drop_none({"request_id": self.request_id, "op": self.op, "name": self.name,
                           "value": self.value, "count": self.count})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        rid = _req(m, "request_id", _u32, "a uint32")
        op = _req(m, "op", lambda v: v in OPS, "pause, resume, step, set or run")
        name = _opt(m, "name", _name, "a parameter or action name")
        if op in (OP_SET, OP_RUN) and name is None:
            _bad(f"'{op}' needs 'name'")
        value = None
        if op == OP_SET:
            value = _req(m, "value", _set_value, "a number, a boolean or a choice")
        count = None
        if op == OP_STEP:
            count = _opt(m, "count", lambda v: _u32(v) and v >= 1, "a positive uint32")
        return cls(rid, op, name if op in (OP_SET, OP_RUN) else None, value, count)


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
    if request is not None and not _obj(request):
        _dv_bad("a data view is a JSON object")
    dv = _merge(request, view.get("default"))
    kind = view.get("kind")
    budget = dv["budget"]
    if kind == Kind.SERIES:
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
    need = {Kind.HEATMAP: 3 if tile is not None else 2, Kind.IMAGE: 2, Kind.POINTS: 1}.get(kind)
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
    if kind in (Kind.SERIES, Kind.HISTOGRAM):
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
