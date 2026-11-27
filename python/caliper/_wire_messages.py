"""Wire protocol 3.0 session, view, subscription, snapshot, series, and event messages."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys as _sys
from typing import Any, ClassVar

import numpy as np

if __package__:
    from . import _wire_frame as _f, _wire_recordings as _r
else:
    _prefix = __name__.rpartition(".")[0]
    _f = _sys.modules[f"{_prefix}._wire_frame"]
    _r = _sys.modules[f"{_prefix}._wire_recordings"]

DType = _f.DType
Kind = _f.Kind
MAX_EVENT_INDICES = _f.MAX_EVENT_INDICES
MAX_STATS = _f.MAX_STATS
MsgType = _f.MsgType
ROLE_HUB = _f.ROLE_HUB
ROLE_PRODUCER = _f.ROLE_PRODUCER
ROLE_VIEWER = _f.ROLE_VIEWER
SERIES_MAX_VALUES = _f.SERIES_MAX_VALUES
SESSION_ENDED = _f.SESSION_ENDED
SESSION_LIVE = _f.SESSION_LIVE
Section = _f.Section
_Msg = _f._Msg
_bad = _f._bad
_bool = _f._bool
_drop_none = _f._drop_none
_identity = _f._identity
_is_int = _f._is_int
_is_num = _f._is_num
_json_float = _f._json_float
_name = _f._name
_obj = _f._obj
_opt = _f._opt
_req = _f._req
_section = _f._section
_session_id = _f._session_id
_stat = _f._stat
_str = _f._str
_strs = _f._strs
_u32 = _f._u32
_uint = _f._uint

MAX_BARS = _r.MAX_BARS
_recorded = _r._recorded
_recording_ref = _r._recording_ref
_series_32 = _r._series_32
_snapshot_32 = _r._snapshot_32
_timelines = _r._timelines


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
