"""Wire protocol 3.1 controls, linked selection, and interpretation control messages."""
from __future__ import annotations

from dataclasses import dataclass, field
import sys as _sys
from typing import Any, ClassVar

if __package__:
    from . import _interpret_protocol, _wire_frame as _f, _wire_programs as _p
else:
    _prefix = __name__.rpartition(".")[0]
    _interpret_protocol = _sys.modules[f"{_prefix}._interpret_protocol"]
    _f = _sys.modules[f"{_prefix}._wire_frame"]
    _p = _sys.modules[f"{_prefix}._wire_programs"]

ERR_BAD_VALUE = _f.ERR_BAD_VALUE
MAX_EVENT_INDICES = _f.MAX_EVENT_INDICES
MessageError = _f.MessageError
MsgType = _f.MsgType
TEXT_MAX = _f.TEXT_MAX
_INT_EXACT = _f._INT_EXACT
_Msg = _f._Msg
_ascending = _f._ascending
_bad = _f._bad
_bool = _f._bool
_drop_none = _f._drop_none
_finite = _f._finite
_is_int = _f._is_int
_name = _f._name
_obj = _f._obj
_opt = _f._opt
_req = _f._req
_section = _f._section
_stamp = _f._stamp
_str = _f._str
_text = _f._text
_u32 = _f._u32
_uint = _f._uint

_intervene_request = _p._intervene_request
_interventions = _p._interventions

# ---------------------------------------------------------------------------
# 3.1: controls (capability "controls") and linked selection ("selection")
# ---------------------------------------------------------------------------
RUN_NONE, RUN_RUNNING, RUN_PAUSING, RUN_PAUSED, RUN_STEPPING = (
    "none", "running", "pausing", "paused", "stepping")
RUN_STATES = (RUN_NONE, RUN_RUNNING, RUN_PAUSING, RUN_PAUSED, RUN_STEPPING)
OP_PAUSE, OP_RESUME, OP_STEP, OP_SET, OP_RUN = "pause", "resume", "step", "set", "run"
OP_INTERVENE = "intervene"                 # 3.2 draft, capability "interventions"
OP_INTERPRET = "interpret"
CAP_INTERPRETATION = "interpretation"
CAP_TEXT_PARAMS = "text_params"
PARAM_TEXT = "text"
INTERPRET_MAX_BYTES = 262144
INTERPRET_STRING_MAX = 4096
OPS = (OP_PAUSE, OP_RESUME, OP_STEP, OP_SET, OP_RUN, OP_INTERVENE, OP_INTERPRET)
PARAM_NUMBER, PARAM_INTEGER, PARAM_BOOLEAN, PARAM_CHOICE = "number", "integer", "boolean", "choice"
PARAM_TYPES = (PARAM_NUMBER, PARAM_INTEGER, PARAM_BOOLEAN, PARAM_CHOICE, PARAM_TEXT)
STATUS_OK, STATUS_QUEUED, STATUS_ERROR = "ok", "queued", "error"
STATUSES = (STATUS_OK, STATUS_QUEUED, STATUS_ERROR)

MAX_PARAMS = 64                            # 3.1
MAX_ACTIONS = 64
MAX_CHOICES = 64
PARAM_TEXT_MAX = 4096
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


def _fits_type(p: dict, v) -> bool:
    """Whether ``v`` is a value of parameter ``p``'s type (its range aside)."""
    t = p["type"]
    if t == PARAM_NUMBER:
        return _finite(v)
    if t == PARAM_INTEGER:
        return _is_int(v) and abs(v) <= _INT_EXACT
    if t == PARAM_BOOLEAN:
        return _bool(v)
    if t == PARAM_TEXT:
        return _interpret_protocol.text(v, PARAM_TEXT_MAX)
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
        c.interventions = _opt(m, "interventions", _interventions,            # 3.2 draft
                               "{points, active} of intervention points and interventions")
        return c


def _set_value(v) -> bool:
    return _finite(v) or _bool(v) or _interpret_protocol.text(v, PARAM_TEXT_MAX)


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
    interpretation: dict | None = None

    def to_meta(self):
        if self.op == OP_SET and not _set_value(self.value):
            _bad("invalid set value")
        if self.op == OP_INTERPRET and not _interpret_protocol.request(self.interpretation):
            _bad("invalid interpretation request")
        if self.op != OP_INTERPRET and self.interpretation is not None:
            _bad("interpretation requires op interpret")
        return _drop_none({"request_id": self.request_id, "op": self.op, "name": self.name,
                           "value": self.value, "count": self.count,
                           "intervene": self.intervene, "interpretation": self.interpretation})

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
            value = _req(m, "value", _set_value, "a finite number, a boolean, or bounded text")
        count = None
        if op == OP_STEP:
            count = _opt(m, "count", lambda v: _u32(v) and v >= 1, "a positive uint32")
        intervene = None
        if op == OP_INTERVENE:                                                 # 3.2 draft
            intervene = _req(m, "intervene", _intervene_request,
                             "exactly one of {set: {point, op, from_step, until_step}}, "
                             "{clear: point} and {clear_all: true}")
        interpretation = None
        if op == OP_INTERPRET:
            interpretation = _req(m, "interpretation", _interpret_protocol.request,
                                  "a version 1 interpretation request")
        elif "interpretation" in m:
            _bad("interpretation requires op interpret")
        return cls(rid, op, name if op in (OP_SET, OP_RUN) else None, value, count, intervene, interpretation)


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
    result: dict | None = None

    def to_meta(self):
        if self.value is not None and not _set_value(self.value):
            _bad("invalid control reply value")
        if self.result is not None and (self.op != OP_INTERPRET or self.status != STATUS_OK):
            _bad("result requires a successful interpretation reply")
        if self.op == OP_INTERPRET and self.status == STATUS_OK and not _interpret_protocol.bounded(self.result):
            _bad("invalid interpretation result")
        return _drop_none({"request_id": self.request_id, "op": self.op, "status": self.status,
                           "name": self.name, "value": self.value, "message": self.message,
                           "error": self.error, "result": self.result})

    @classmethod
    def from_frame(cls, f):
        m = f.meta
        status = _req(m, "status", lambda v: v in STATUSES, "ok, queued or error")
        op = _req(m, "op", lambda v: v in OPS, "a control op")
        err = _opt(m, "error", _reply_error, "{code, message}")
        if (err is None) == (status == STATUS_ERROR):
            _bad("a control_reply carries 'error' exactly when its status is 'error'")
        if status == STATUS_QUEUED and op not in (OP_RUN, OP_INTERPRET):
            _bad("only 'run' is queued")
        result = _opt(m, "result", _interpret_protocol.bounded, "bounded structured result")
        if "result" in m and (op != OP_INTERPRET or status != STATUS_OK):
            _bad("result requires a successful interpretation reply")
        if op == OP_INTERPRET and status == STATUS_OK and result is None:
            _bad("a successful interpretation reply requires result")
        return cls(_req(m, "request_id", _u32, "a uint32"), op, status,
                   _opt(m, "name", _name, "a parameter or action name"),
                   _opt(m, "value", _set_value, "a finite number, a boolean, or bounded text"),
                   _opt(m, "message", _text, f"a string of at most {TEXT_MAX} bytes"), err, result)


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
    if t in (PARAM_BOOLEAN, PARAM_CHOICE, PARAM_TEXT):
        if not _fits_type(param, value):
            raise MessageError(ERR_BAD_VALUE, f"{param['name']!r} takes "
                               + ("true or false" if t == PARAM_BOOLEAN else
                                  "UTF-8 text" if t == PARAM_TEXT else f"one of {param['choices']}") + f", not {value!r}")
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
