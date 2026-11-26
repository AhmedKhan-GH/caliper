"""Wire protocol 3.2 timelines, text logs, bar-chart additions, and recording messages."""
from __future__ import annotations

from dataclasses import dataclass
import re
import sys as _sys
from typing import Any, ClassVar

import numpy as np

if __package__:
    from . import _wire_frame as _f
else:
    _prefix = __name__.rpartition(".")[0]
    _f = _sys.modules[f"{_prefix}._wire_frame"]

DType = _f.DType
Encoding = _f.Encoding
Frame = _f.Frame
Kind = _f.Kind
MAX_TIMELINES = _f.MAX_TIMELINES
MessageError = _f.MessageError
MsgType = _f.MsgType
NAME_MAX = _f.NAME_MAX
Section = _f.Section
_Msg = _f._Msg
_bad = _f._bad
_bool = _f._bool
_drop_none = _f._drop_none
_finite = _f._finite
_is_int = _f._is_int
_name = _f._name
_obj = _f._obj
_opt = _f._opt
_part_section = _f._part_section
_req = _f._req
_section = _f._section
_session_id = _f._session_id
_stamp = _f._stamp
_str = _f._str
_u32 = _f._u32
_uint = _f._uint

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
