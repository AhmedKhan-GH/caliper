"""Controls of the running script and linked selection (wire protocol 3.1).

A viewer can pause, resume and single-step the training loop, change the
parameters the script registered, run the actions it registered, and select
items that every view of the session then highlights
(docs/reference/wire-protocol.md, "Controls" and "Linked selection"). This
module holds all of that state for the process, one :class:`Controls` object,
``CONTROLS``, behind the public calls of caliper/__init__.py:

    caliper.step()                 the checkpoint, once per training step
    caliper.pause()                pause at the next checkpoint
    caliper.param(name, value)     a Param: .value, .set(v), .remove()
    caliper.action(name, fn)       an Action: called by a viewer at a checkpoint
    caliper.items(space, details)  an ItemSpace: .selection, .select(ids)
    items= on show, points, series and offer: which axis indexes which space

It imports neither torch nor numpy, nor caliper._wire (which imports numpy)
until a transport attaches or a value is validated, and it works without any
transport: parameters keep their values, ``step()`` returns at once, and an
action can be called from Python.

THREADS. The script calls ``step()`` on its training thread, and requests
arrive on the transport's thread. A request only changes small state under
``Controls.lock``. All the script's code runs where the protocol says: an
action at a checkpoint on the thread that calls ``step()``; a details
function on the thread "caliper.details", never on the training thread or the
transport's thread, so it must only read data that the training loop does not
change in place (a dataset's labels, arrays the script computed before). The
thread "caliper.controls" only keeps time (action and details time-outs).

THE COST OF A CHECKPOINT. ``caliper.step()`` reads one flag, ``busy``, which is
true only while something waits for a checkpoint (a parameter value, an
action, a pause, a step count, the first call). Otherwise it returns after
counting the step.

THE SEAM. A transport (the route of ``caliper.serve()``, later of
``caliper.connect()``) attaches once and gets a :class:`Route`:

    route = caliper._controls.attach(transport)

and the transport provides these members, called from any thread. None of
them may block or call back into this module:

    transport.broadcast(type, meta)  send a CONTROLS or a SELECTION message
                                     (type is caliper._wire.MsgType) with this
                                     metadata to every peer of the route that
                                     announced the capability for it ("controls"
                                     or "selection"); it may coalesce them as the
                                     protocol allows. The CONTROLS that announces
                                     a pause comes with after_data=True: it goes
                                     to each peer only after the data that the
                                     route took before it (the protocol's order)
    transport.reply(conn, meta)      send a CONTROL_REPLY with this metadata to
                                     the connection conn (an opaque key the
                                     transport chose)
    transport.may_control(conn)      bool: whether CONTROL from conn is accepted
                                     (WELCOME's "control" for that connection)
    transport.controllers()          int: the open connections that may control
                                     (to a hub: 1 while the hub forwards controls)
    transport.views_changed()        optional: the item spaces of a view changed,
                                     so send VIEWS again

The route gives the transport, callable from any thread:

    route.on_control(conn, meta) -> dict
        The metadata of a CONTROL from conn; returns the metadata of the
        CONTROL_REPLY to send to conn now. An accepted "run" is answered
        "queued", and its final reply comes later through transport.reply.
        Raises caliper._wire.MessageError (bad_request) for a malformed one,
        which the transport answers with ERROR as for any malformed message.
    route.on_select(conn, meta, axes=None) -> None
        The metadata of every select EVENT (which still also goes to Python's
        event handlers as before). ``axes``: the payload axes of the snapshot
        the viewer was drawing, when the transport knows them; they map a
        points selection that has ``indices`` but no ``row_ids``.
    route.snapshot_state() -> list of (type, meta)
        What a viewer receives right after the VIEWS of an attachment: one
        CONTROLS, then one SELECTION per item space with a selection. Send
        each only to a peer with its capability.
    route.annotate_view(entry) -> dict
        A VIEWS entry with the "items" keys of its item spaces, for a peer
        that announced "selection" (the entry itself when it has none).
    route.detach()
        The transport stops: a paused run is released if no route remains.

A transport may also append a function to ``CONTROLS.wait_hooks``: it runs on
the training thread, in the checkpoint, right before a pause starts to wait
and before the pause is announced (the network route closes its open tick and
moves its gathered series values there, and waits for their copies, so that
the data of the step the run paused at goes out before the run state).

``CAPABILITIES`` are the strings a transport announces for this module.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import collections
import heapq
import itertools
import json
import logging
import math
import numbers
import threading
import time
import warnings
from typing import Any, Callable

__all__ = ["CONTROLS", "Controls", "Param", "Action", "ItemSpace", "Route", "attach",
           "CAPABILITIES", "Interventions"]

# The strings and limits of wire protocol 3.1 (caliper/live_net.h), mirrored so
# that importing this module loads neither caliper._wire nor numpy.
# python/tests/test_controls.py checks that they agree with _wire.
CAP_CONTROLS, CAP_SELECTION = "controls", "selection"
CAPABILITIES = (CAP_CONTROLS, CAP_SELECTION)
RUN_NONE, RUN_RUNNING, RUN_PAUSING, RUN_PAUSED, RUN_STEPPING = (
    "none", "running", "pausing", "paused", "stepping")
NAME_MAX = 63
MAX_PARAMS = 64
MAX_ACTIONS = 64
MAX_META_BYTES = 1 << 20
MAX_CHOICES = 64
TEXT_MAX = 1024
PARAM_TEXT_MAX = 4096
MAX_ITEM_SPACES = 16
MAX_SELECTION = 32768                  # CALIPER_NET_MAX_EVENT_INDICES
MAX_DETAIL_ITEMS = 32
MAX_DETAIL_FIELDS = 16
DETAIL_TEXT_MAX = 256

# Times, in seconds (module attributes, so tests can shorten them).
ACTION_WAIT_S = 600.0                  # CALIPER_NET_ACTION_WAIT_MS
DETAILS_TIMEOUT_S = 2.0                # CALIPER_NET_DETAILS_TIMEOUT_MS
RELEASE_GRACE_S = 10.0                 # a paused run with no connection that may
                                       # control resumes after this long (a page
                                       # reload reconnects well within it)
WAIT_POLL_S = 0.1                      # a waiting checkpoint wakes this often, so
                                       # Ctrl-C and departed controllers are seen
STEPS_REPORT_S = 0.1                   # while stepping, steps_left goes out at most
                                       # this often
RUN_REPORT_S = 1.0                     # while running, the step goes out at most this
                                       # often, at the first checkpoint after it
LINGER_CHECK_S = 0.25                  # while an intervention acts or waits, the connections
                                       # that may control are counted at most this often

# The interventions (design/gpu-programs.md, Interventions): CONTROL op "intervene" and the
# CONTROLS key "interventions" of capability "interventions" (3.2 draft).
OP_INTERVENE = "intervene"
INTERVENE_OPS = ("ablate", "add", "scale", "clamp", "patch")
INTERVENE_RESERVED = ("steer",)
MAX_POINTS_LISTED = 4096               # CALIPER_NET_MAX_POINTS_LISTED
MAX_INTERVENTIONS = 64                 # CALIPER_NET_MAX_INTERVENTIONS
POINT_MAX = 255                        # CALIPER_NET_POINT_MAX
ERR_UNKNOWN_POINT = "intervene.unknown_point"
ERR_OP_RESERVED = "intervene.op_reserved"
ERR_BAD_RANGE = "intervene.bad_range"
ERR_NOT_SUPPORTED = "intervene.not_supported"
IV_PENDING, IV_ACTIVE, IV_ENDED = "pending", "active", "ended"

_ITEM_KINDS = ("heatmap", "image", "points", "series")
_INT_EXACT = 1 << 53
_NO = object()                          # no pending value

_log = logging.getLogger("caliper.controls")
_log.addHandler(logging.NullHandler())
_log.propagate = False

_wire_mod = None


def _w():
    """caliper._wire, imported on first use (it imports numpy)."""
    global _wire_mod
    if _wire_mod is None:
        from . import _wire
        _wire_mod = _wire
    return _wire_mod


# ===========================================================================
# small checks
# ===========================================================================
def _name_ok(v) -> bool:
    return isinstance(v, str) and 0 < len(v.encode("utf-8")) <= NAME_MAX


def _check_name(v, what: str) -> str:
    if not _name_ok(v):
        raise ValueError(f"invalid {what} name {v!r} (a str of 1 to {NAME_MAX} UTF-8 bytes)")
    return v


def _cut(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` UTF-8 bytes, on a character boundary."""
    b = text.encode("utf-8")
    if len(b) <= limit:
        return text
    return b[:limit].decode("utf-8", "ignore")


def _check_help(help) -> str | None:
    if help is None:
        return None
    if not isinstance(help, str) or len(help.encode("utf-8")) > TEXT_MAX:
        raise ValueError(f"help must be a str of at most {TEXT_MAX} UTF-8 bytes")
    return help


def _check_controls_budget(meta):
    """Check a detached prospective message with space for automatic state changes."""
    from ._intervention_protocol import MAX_BYTES

    def size(value):
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8"))

    # Steps and action flags can change without a new admission request.
    run = meta["run"]
    meta["run"] = {"state": RUN_STEPPING, "step": max(_INT_EXACT - 1, run.get("step", 0)),
                   "steps_left": max(_INT_EXACT - 1, run.get("steps_left", 0))}
    for action in meta["actions"]:
        action.update(running=False, queued=False)
    for entry in meta.get("interventions", {}).get("active", []):
        if entry["state"] != IV_ENDED:
            # One UTF-8 byte can require six JSON bytes. Reserve the full error
            # bounds, even when every live hook fails at the same boundary.
            entry.update(state=IV_PENDING,
                         since_step=max(_INT_EXACT - 1, run.get("step", 0) + 1,
                                        entry.get("since_step") or 0),
                         error="\x01" * TEXT_MAX, error_code="\x01" * NAME_MAX)
        if size(entry) > MAX_BYTES:
            raise ValueError("intervention metadata exceeds the wire limit with lifecycle headroom")
    if size(meta) > MAX_META_BYTES:
        raise ValueError("CONTROLS metadata exceeds the wire limit with lifecycle headroom")


def _scalar(v):
    """A numpy or torch scalar as a Python number (a torch tensor on a GPU is
    read back once; registration only)."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    item = getattr(v, "item", None)
    if item is not None:
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return v
    return v


# ===========================================================================
# parameters
# ===========================================================================
def _run_value(p) -> None:
    """Give ``p`` the value that the model runner set for it (caliper._managed), checked and
    clamped like a value from a viewer."""
    from . import _managed
    v = _managed.value(p.name)
    if v is _managed.NONE:
        return
    w = _w()
    try:
        p._value = w.param_value(p.entry(), _scalar(v))
    except w.MessageError as e:
        warnings.warn(f"caliper: the managed run's value of parameter {p.name!r} does not "
                      f"apply: {e.message}", RuntimeWarning, stacklevel=4)


class Param:
    """A parameter that viewers can change while the script runs, made by [`param`][caliper.param].

    The script reads `value` in its loop. A viewer changes it through the stream.

    Attributes:
        name (str): The name of the parameter.
        type (str): `"boolean"`, `"integer"`, `"number"`, `"choice"`, or `"text"`.
        min (int | float | None): The lower end of the range, or `None`.
        max (int | float | None): The upper end of the range, or `None`.
        log (bool): Whether a viewer draws the slider on a log scale.
        choices (tuple[str, ...] | None): The choices of a `"choice"` parameter, or `None`.
        help (str | None): The tooltip, or `None`.
    """
    __slots__ = ("name", "type", "min", "max", "log", "choices", "help", "_value", "_pending",
                 "_ctl", "removed", "_version")

    def __init__(self, ctl, name, type_, value, lo, hi, log, choices, help):
        self.name, self.type = name, type_
        self.min, self.max, self.log, self.choices, self.help = lo, hi, log, choices, help
        self._value = value
        self._pending = _NO
        self._ctl = ctl
        self.removed = False
        self._version = 0               # counts the values set, also a value equal to the last

    @property
    def value(self):
        """The current value, to read in the loop, from any thread.

        It changes only in [`step`][caliper.step], or at once before the first `step()` of the
        script.
        """
        return self._value

    @property
    def pending(self):
        """A value from a viewer that `value` takes at the next `step()`, or `None`."""
        p = self._pending
        return None if p is _NO else p

    def set(self, value):
        """Set the value from the script, at once and for every viewer.

        The value is checked and clamped like a value from a viewer.

        Args:
            value (bool | int | float | str): The new value, of the type of the parameter.

        Returns:
            bool | int | float | str: The value set.

        Raises:
            ValueError: When the value has another type or is not one of the choices.
        """
        return self._ctl._script_set(self, value)

    def remove(self) -> bool:
        """Take the parameter away from the viewers.

        `value` keeps its last value.

        Returns:
            bool: `True` if the parameter was registered.
        """
        return self._ctl._remove_param(self)

    def entry(self) -> dict:
        """The parameter object of a CONTROLS message."""
        e = {"name": self.name, "type": self.type, "value": self._value}
        if self._pending is not _NO:
            e["pending"] = self._pending
        if self.min is not None:
            e["min"] = self.min
        if self.max is not None:
            e["max"] = self.max
        if self.log:
            e["log"] = True
        if self.choices is not None:
            e["choices"] = list(self.choices)
        if self.help is not None:
            e["help"] = self.help
        return e

    def __repr__(self) -> str:
        p = "" if self._pending is _NO else f", pending {self._pending!r}"
        return f"<caliper.Param {self.name!r} {self.type} = {self._value!r}{p}>"


def _param_spec(name, value, lo, hi, log, choices, help):
    """(type, value, min, max, log, choices, help) of a registration, checked."""
    _check_name(name, "parameter")
    help = _check_help(help)
    value, lo, hi = _scalar(value), _scalar(lo), _scalar(hi)
    if choices is not None:
        choices = list(choices)
        if not 1 <= len(choices) <= MAX_CHOICES or not all(_name_ok(c) for c in choices) \
                or len(set(choices)) != len(choices):
            raise ValueError(f"choices must be 1 to {MAX_CHOICES} distinct str of 1 to "
                             f"{NAME_MAX} UTF-8 bytes")
        if value not in choices:
            raise ValueError(f"parameter {name!r}: {value!r} is not one of {choices}")
        if lo is not None or hi is not None or log:
            raise ValueError(f"parameter {name!r}: a choice takes no min, max or log")
        return "choice", value, None, None, False, tuple(choices), help
    if isinstance(value, bool):
        if lo is not None or hi is not None or log:
            raise ValueError(f"parameter {name!r}: a boolean takes no min, max or log")
        return "boolean", value, None, None, False, None, help
    if isinstance(value, str):
        from ._interpret_protocol import text
        if not text(value, PARAM_TEXT_MAX):
            raise ValueError(f"parameter {name!r}: text must be valid UTF-8 without NUL within {PARAM_TEXT_MAX} bytes")
        if lo is not None or hi is not None or log:
            raise ValueError(f"parameter {name!r}: text takes no min, max or log")
        return "text", value, None, None, False, None, help
    if isinstance(value, numbers.Integral):
        type_, conv = "integer", int
        ok = (lambda x: isinstance(x, numbers.Integral) and not isinstance(x, bool)
              and abs(int(x)) <= _INT_EXACT)
    elif isinstance(value, numbers.Real):
        type_, conv = "number", float
        ok = (lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool)
              and math.isfinite(float(x)))
    else:
        raise TypeError(f"parameter {name!r}: the value must be a bool, int, float or a str "
                        f"with choices, not {type(value).__name__}")
    for what, x in (("value", value), ("min", lo), ("max", hi)):
        if x is not None and not ok(x):
            raise ValueError(f"parameter {name!r}: {what} {x!r} is not a"
                             f"{' whole number' if type_ == 'integer' else ' finite number'}"
                             f"{f' of magnitude at most 2**53' if type_ == 'integer' else ''}")
    value = conv(value)
    lo = None if lo is None else conv(lo)
    hi = None if hi is None else conv(hi)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"parameter {name!r}: min {lo} is above max {hi}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"parameter {name!r}: value {value} is outside [{lo}, {hi}]")
    log = bool(log)
    if log and not (lo is not None and lo > 0 and hi is not None):
        raise ValueError(f"parameter {name!r}: log=True needs min > 0 and max")
    return type_, value, lo, hi, log, None, help


# ===========================================================================
# actions
# ===========================================================================
class Action:
    """A function that viewers can run, made by [`action`][caliper.action].

    A request of a viewer runs `fn()` at the next `caliper.step()`, on the thread that calls it.

    Attributes:
        name (str): The name of the action.
        fn (Callable[[], object]): The registered function.
        help (str | None): The tooltip, or `None`.
    """
    __slots__ = ("name", "fn", "help", "queued", "running", "waiters", "run_id", "_ctl",
                 "removed", "__wrapped__")

    def __init__(self, ctl, name, fn, help):
        self.name, self.fn, self.help = name, fn, help
        self.queued = self.running = False
        self.waiters: list = []                  # (route, conn, request_id) of the queued run
        self.run_id = 0
        self._ctl = ctl
        self.removed = False
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        """Run `fn` at once, on the thread of the caller, and return what it returns."""
        return self.fn(*args, **kwargs)

    def remove(self) -> bool:
        """Take the action away from the viewers.

        A run that waits is answered `unknown_control`.

        Returns:
            bool: `True` if the action was registered.
        """
        return self._ctl._remove_action(self)

    def entry(self) -> dict:
        """The action object of a CONTROLS message."""
        e = {"name": self.name, "running": self.running, "queued": self.queued}
        if self.help is not None:
            e["help"] = self.help
        return e

    def __repr__(self) -> str:
        state = "running" if self.running else ("queued" if self.queued else "idle")
        return f"<caliper.Action {self.name!r} {state}>"


# ===========================================================================
# item spaces, selections and details
# ===========================================================================
def _column(seq) -> list:
    """A static details column as a host list (a tensor or array is copied
    once, at registration)."""
    tolist = getattr(seq, "tolist", None)
    if tolist is not None and not isinstance(seq, (list, tuple)):
        detach = getattr(seq, "detach", None)
        if detach is not None:
            seq = detach()
            tolist = seq.tolist
        out = tolist()
        return out if isinstance(out, list) else [out]
    return list(seq)


def _cell(v):
    """One details value: a number, a str of at most 256 bytes, a bool or None."""
    v = _scalar(v)
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, numbers.Integral):
        return int(v) if abs(int(v)) <= _INT_EXACT else float(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return f if math.isfinite(f) else None
    return _cut(v if isinstance(v, str) else str(v), DETAIL_TEXT_MAX)


def _fields(names) -> list:
    names = [n for n in names]
    if len(names) > MAX_DETAIL_FIELDS:
        raise ValueError(f"details have at most {MAX_DETAIL_FIELDS} fields, not {len(names)}")
    for n in names:
        _check_name(n, "details field")
    return names


class ItemSpace:
    """A set of items numbered from 0 that several views index, made by [`items`][caliper.items].

    An example is the samples of an evaluation set. A selection in one view highlights the same
    items in every view that names the space.

    Attributes:
        name (str): The name of the item space.
    """
    __slots__ = ("name", "_ctl", "_table", "_fields", "_fn", "_selection", "_truncated",
                 "_view", "_version", "_details", "_details_error", "_settled")

    def __init__(self, ctl, name):
        self.name = name
        self._ctl = ctl
        self._table = None                       # {field: list} (static details)
        self._fields: list = []
        self._fn = None                          # details function
        self._selection: list = []
        self._truncated = False
        self._view = None
        self._version = 0
        self._details = None                     # {"fields", "items", "values"} or None
        self._details_error = None
        self._settled = True                     # the details of this version are final

    @property
    def selection(self) -> list:
        """The list of selected item ids, ascending.

        A viewer or `select` set it. It is current even while a notebook cell runs.
        """
        return list(self._selection)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def select(self, items) -> list:
        """Set the selection from the script, for every viewer.

        An empty sequence clears it.

        Args:
            items (Iterable[int]): The item ids to select.

        Returns:
            list[int]: The item ids as viewers get them: ascending, without repeats, at most 32,768.

        Raises:
            ValueError: When an id is negative.
        """
        ids = [int(i) for i in items]
        if any(i < 0 for i in ids):
            raise ValueError("item ids are non-negative integers")
        self._ctl._set_selection(self.name, ids, None, False)
        return self.selection

    def clear(self) -> None:
        """Clear the selection for every viewer."""
        self.select(())

    @staticmethod
    def _details_source(details) -> tuple:
        """(table, fields, function) of a details argument; a table is copied
        to host lists here, outside the controls' lock."""
        if details is None:
            return None, [], None
        if callable(details):
            return None, [], details
        if not (hasattr(details, "items") and callable(details.items)):
            raise TypeError("details must be a mapping {field: sequence} or a function")
        pairs = list(details.items())
        fields = _fields([k for k, _ in pairs])
        return {k: _column(v) for k, v in pairs}, fields, None

    def _lookup(self, ids) -> dict:
        """The details of ``ids`` from the static table."""
        values = []
        for i in ids:
            row = []
            for f in self._fields:
                col = self._table[f]
                row.append(_cell(col[i]) if 0 <= i < len(col) else None)
            values.append(row)
        return {"fields": list(self._fields), "items": list(ids), "values": values}

    def __repr__(self) -> str:
        n = len(self._selection)
        return f"<caliper.ItemSpace {self.name!r}, {n} selected>"


def _details_from(out, ids) -> dict:
    """The details object from what a details function returned."""
    if isinstance(out, dict) or (hasattr(out, "items") and callable(out.items)
                                 and not isinstance(out, (list, tuple))):
        pairs = list(out.items())
        fields = _fields([k for k, _ in pairs])
        cols = [_column(v) for _, v in pairs]
        values = [[_cell(c[r]) if r < len(c) else None for c in cols] for r in range(len(ids))]
        return {"fields": fields, "items": list(ids), "values": values}
    rows = list(out)
    fields: list = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    fields = _fields(fields)
    values = [[_cell(rows[r].get(f)) if r < len(rows) else None for f in fields]
              for r in range(len(ids))]
    return {"fields": fields, "items": list(ids), "values": values}


_FIRST, _LAST = 0, -1                   # mapping keys: a source's first or last dim


def _view_dims(kind: str, items) -> dict:
    """The item spaces of a view call's ``items``, as {key: space}: a str
    names the space of the view's item axis, which is the first dim of each
    source for a heatmap or image (its rows when no data view lays it out
    otherwise), the last dim for points (the point dim) and the index of a
    series (key None); a dict maps dim names to spaces. The first and last
    dims are found in the view's VIEWS entry (:meth:`Controls.annotate_view`)."""
    if kind not in _ITEM_KINDS:
        raise ValueError(f"items= is for heatmaps, images, points and series, not {kind!r}")
    if isinstance(items, str):
        _check_name(items, "item space")
        if kind == "series":
            return {None: items}
        return {(_LAST if kind == "points" else _FIRST): items}
    if isinstance(items, dict):
        if kind == "series":
            raise ValueError("a series has no dims: items= takes the space's name as a str")
        out = {}
        for d, s in items.items():
            _check_name(d, "dim")
            out[d] = _check_name(s, "item space")
        return out
    raise TypeError("items= takes an item space name (str) or a dict {dim: space}")


# ===========================================================================
# time-outs and details functions off the training thread
# ===========================================================================
class _Timers:
    """One daemon thread, "caliper.controls", that calls functions at given
    times (monotonic seconds). Started on first use."""

    def __init__(self):
        self._heap: list = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._thread = None

    def at(self, when: float, fn, *args) -> None:
        with self._cond:
            heapq.heappush(self._heap, (when, next(self._seq), fn, args))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caliper.controls",
                                                daemon=True)
                self._thread.start()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._heap or self._heap[0][0] > time.monotonic():
                    wait = None if not self._heap else self._heap[0][0] - time.monotonic()
                    self._cond.wait(wait)
                _, _, fn, args = heapq.heappop(self._heap)
            try:
                fn(*args)
            except Exception:
                _log.debug("a timer failed", exc_info=True)


class _DetailsWorker:
    """One daemon thread, "caliper.details", that runs the script's details
    functions, the newest request of each item space first replacing an
    older one that has not started."""

    def __init__(self):
        self._jobs: collections.OrderedDict = collections.OrderedDict()
        self._cond = threading.Condition()
        self._thread = None

    def submit(self, key, fn) -> None:
        with self._cond:
            self._jobs.pop(key, None)
            self._jobs[key] = fn
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caliper.details",
                                                daemon=True)
                self._thread.start()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs:
                    self._cond.wait()
                _, fn = self._jobs.popitem(last=False)
            try:
                fn()
            except Exception:
                _log.debug("a details job failed", exc_info=True)


# ===========================================================================
# the seam
# ===========================================================================
# ===========================================================================
# interventions
# ===========================================================================
class _Intervention:
    """One intervention of the watched model, keyed by its point and its op."""

    __slots__ = ("point", "op", "from_step", "until_step", "since_step", "state", "hook",
                 "clear", "error", "error_code", "spec", "applied_spec", "replace", "report")

    def __init__(self, point: str, op: str, from_step, until_step, spec=None):
        self.point, self.op = point, op
        self.from_step, self.until_step = from_step, until_step
        self.since_step = None
        self.state = IV_PENDING
        self.hook = None                  # the adapter's remover while the hook is on the model
        self.clear = False                # intervene.clear asked for it; gone at the next step
        self.error = None                 # why the script ended it, or None
        self.error_code = None
        self.spec = spec or {"point": point, "op": op, "kind": op,
                             "from_step": from_step, "until_step": until_step}
        self.applied_spec = None
        self.replace = False
        self.report = False

    def entry(self) -> dict:
        from copy import deepcopy
        entry = dict(deepcopy(self.spec), from_step=self.from_step, until_step=self.until_step,
                     since_step=self.since_step, state=self.state)
        if self.op == "ablate":
            entry.pop("kind", None)  # Preserve the original ablation CONTROLS entry.
        if self.error is not None:
            entry["error"] = self.error
        if self.error_code is not None:
            entry["error_code"] = self.error_code
        return entry


class Interventions:
    """The interventions of the model of ``caliper.watch(model)``, as
    :meth:`Controls.interventions_on` starts them.

    ``remove()`` takes every hook of the interventions off the model and stops announcing them,
    as a torch hook handle does, so a ``WatchHandle`` can hold it among its hooks.
    """

    def __init__(self, ctl: "Controls", model, adapter):
        self._ctl = ctl
        self.model = model
        self.adapter = adapter
        self.points: list = []            # the CONTROLS items, at most MAX_POINTS_LISTED
        self.names: frozenset = frozenset()
        self._shapes_off = None           # removes the shape hooks at the next step
        from .interp._intervene import InterventionService
        self.executor = InterventionService(model, adapter=adapter)
        self._operation_owner = None
        self._operation_depth = 0
        self._readers = 0
        self._retired = None

    @contextmanager
    def _operation(self, kind):
        """Reserve executor mutation without a wait, with the controls lock held.

        Only nested replay on the same thread shares ownership. Requests can still
        validate and queue during hook preparation. Configuration requires no readers.
        """
        from .interp._intervene import InterventionError
        if self._ctl._iv_model is not self:
            raise InterventionError("the intervention service is no longer attached")
        owner = (kind, threading.get_ident())
        if ((self._operation_owner is not None
             and not (kind == "replay" and self._operation_owner == owner))
                or (self._ctl._iv_attaching and kind != "watch")
                or (kind in ("configure", "watch") and self._readers)):
            raise InterventionError("an intervention lifecycle operation is in progress", ERR_NOT_SUPPORTED)
        self._operation_owner = owner
        self._operation_depth += 1
        try:
            yield
        finally:
            self._operation_depth -= 1
            if not self._operation_depth:
                self._operation_owner = None
                self._ctl._finish_retired(self)

    @property
    def dictionary(self):
        return self.executor.dictionary

    @property
    def suspended(self):
        return self.executor.suspended

    def suspend(self):
        return self.executor.suspend()

    def rows(self, indices):
        return self.executor.rows(indices)

    def invalidate_source_cache(self):
        """Release clean patch sources after an external prompt or model change."""
        with self._ctl.lock, self._operation("cache"):
            with self._ctl._outside_lock():
                self.executor.invalidate_cache()

    def configure(self, *, dictionary=None, tokenizer=None, dictionary_id=None, resource_guard=None):
        """Bind resources and end prior interventions after validation succeeds."""
        with self._ctl.lock, self._operation("configure"):
            records = list(self._ctl._ivs.values())
            with self._ctl._outside_lock():
                self.executor.configure(dictionary=dictionary, tokenizer=tokenizer,
                                        dictionary_id=dictionary_id, resource_guard=resource_guard)
                for iv in records:
                    if iv.hook is not None:
                        iv.hook()
            if self._ctl._iv_model is not self:
                raise ValueError("the intervention service is no longer attached")
            for iv in records:
                iv.hook, iv.applied_spec = None, None
                if iv.state != IV_ENDED:
                    iv.state = IV_ENDED
                    iv.error = "intervention resources changed"
                    iv.error_code = "intervene.configuration_changed"
            self._ctl._iv_revision += 1
            self._ctl._iv_due = True
            self._ctl._update_busy()
        self._ctl._publish()
        return self

    def set(self, specification):
        """Queue one specification for the next step boundary, or raise a refusal."""
        self._request({"set": specification})

    def clear(self, point=None):
        """Queue removal at the next step boundary."""
        self._request({"clear_all": True} if point is None else {"clear": point})

    def clear_feature(self, point, feature, *, kind="clamp"):
        """Queue removal only if the latest specification matches this feature.

        Return True for a new clear. Preserve replacements for other features and
        all other records. The current hook remains until the next real boundary.
        """
        from .interp._intervene import InterventionError
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            if self._operation_owner is not None and self._operation_owner[0] in ("configure", "watch"):
                raise InterventionError("an intervention lifecycle operation is in progress", ERR_NOT_SUPPORTED)
            if not isinstance(point, str) or point not in self.names:
                raise InterventionError("unknown intervention point", "intervene.unknown_point")
            if kind not in ("clamp", "scale") or type(feature) is not int or feature < 0:
                raise InterventionError("clear_feature requires a feature index and clamp or scale kind")
            iv = self._ctl._ivs.get((point, kind))
            if iv is None or iv.state == IV_ENDED or iv.clear or iv.spec.get("feature") != feature:
                return False
            iv.clear = True
            self._ctl._iv_revision += 1
            self._ctl._iv_due = True
            self._ctl.counters["interventions_in"] += 1
            self._ctl._update_busy()
            self._ctl.cond.notify_all()
        self._ctl._publish()
        return True

    def _request(self, request):
        from .interp._intervene import InterventionError
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            refused = self._ctl._intervene(request)
            self._ctl._update_busy()
        if refused is not None:
            raise InterventionError(refused[1], refused[0])

    def specifications(self, *, active_only=True):
        """Return independent JSON specifications for the current effects.

        With active_only=False, include queued specifications and pending schedules.
        Ended records remain available through state_dict().
        """
        from copy import deepcopy
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                return []
            if active_only:
                return [deepcopy(iv.applied_spec) for iv in self._ctl._ivs.values()
                        if iv.state == IV_ACTIVE and iv.hook is not None and iv.applied_spec is not None]
            return [deepcopy(iv.spec) for iv in self._ctl._ivs.values()
                    if iv.state != IV_ENDED and not iv.clear]

    def state_dict(self):
        """Return complete JSON state. The run checkpoint owns the step counter."""
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise ValueError("the intervention service is no longer attached")
            return self._state_dict()

    def _state_dict(self):
        """Export records while the controls lock is held."""
        from copy import deepcopy
        return {"version": 1, "interventions": [
            {"specification": deepcopy(iv.spec), "state": iv.state,
             "applied_specification": deepcopy(iv.applied_spec),
             "since_step": iv.since_step, "clear": iv.clear,
             "error": iv.error, "error_code": iv.error_code}
            for iv in self._ctl._ivs.values()]}

    @contextmanager
    def temporary_state(self, state):
        """Replay active snapshot effects on the run thread without lifecycle changes.

        The caller suppresses managed forward callbacks for this scope. Requests may
        still queue through Controls.lock and take effect at the next real boundary.
        Snapshot specifications must describe the effects at the original trace.
        Temporary hook failures raise InterventionError at exit and leave live records intact.

        Accept a version 1 lifecycle object or an active list from specifications().
        Lifecycle snapshots retain both applied effects and queued replacements.
        """
        from .interp._intervene import InterventionError, _InterventionReplay
        if isinstance(state, list):
            if len(state) > MAX_INTERVENTIONS:
                raise InterventionError("invalid intervention records")
            state = {"version": 1, "interventions": [
                {"specification": spec, "state": IV_ACTIVE, "since_step": 0,
                 "clear": False, "error": None, "error_code": None} for spec in state]}
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != 1 or set(state) != {"version", "interventions"}):
            raise InterventionError("unsupported intervention state")
        records = state["interventions"]
        if not isinstance(records, list) or len(records) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention records")
        with self._ctl.lock, self._operation("replay"):
            hooks = [iv.hook for iv in self._ctl._ivs.values() if iv.hook is not None]
            with self._ctl._outside_lock():
                replay = _InterventionReplay(self.executor)
                # Validate every record before any hook or patch-source forward.
                specifications, keys = [], set()
                for record in records:
                    spec, applied = _record_specifications(record, replay.executor)
                    key = spec["point"], spec["op"]
                    if key in keys:
                        raise InterventionError("duplicate intervention record")
                    keys.add(key)
                    if record["state"] == IV_ACTIVE:
                        # A queued clear retains its effect until the next boundary.
                        spec = applied
                        spec.pop("from_step", None)
                        spec.pop("until_step", None)
                        specifications.append(spec)
                with replay._scope(hooks, specifications):
                    yield replay

    def load_state_dict(self, state, *, restore_active=False):
        """Restore queued records, with optional applied hooks before a script forward.

        The default resumes hooks at a boundary. restore_active installs only the saved
        applied effects and leaves queued requests and the public step counter unchanged.
        """
        from .interp._intervene import InterventionError
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != 1 or set(state) != {"version", "interventions"}):
            raise InterventionError("unsupported intervention state")
        records = state["interventions"]
        if not isinstance(records, list) or len(records) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention records")
        with self._ctl.lock, self._operation("restore"):
            self._restore_records(records, restore_active=restore_active)

    def apply_overrides(self, specifications):
        """Apply explicit fork overrides without a boundary for unrelated saved work."""
        from copy import deepcopy
        from .interp._intervene import InterventionError
        if not isinstance(specifications, list) or len(specifications) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention overrides")
        with self._ctl.lock, self._operation("restore"):
            if not specifications:
                return
            revision = self._ctl._iv_revision
            records = {(record["specification"]["point"], record["specification"]["op"]): record
                       for record in self._state_dict()["interventions"]}
            keys = set()
            step = self._ctl._next_step()
            for request in specifications:
                with self._ctl._outside_lock():
                    spec = self.executor.validate(request)
                key = spec["point"], spec["op"]
                if key in keys:
                    raise InterventionError("duplicate intervention override")
                keys.add(key)
                if spec.get("until_step") is not None and spec["until_step"] <= step:
                    raise InterventionError("override until_step is in the past", "intervene.bad_range")
                active = spec.get("from_step") is None or spec["from_step"] <= step
                records[key] = {"specification": spec, "state": IV_ACTIVE if active else IV_PENDING,
                                "applied_specification": deepcopy(spec) if active else None,
                                "since_step": step if active else None, "clear": False,
                                "error": None, "error_code": None}
            while len(records) > MAX_INTERVENTIONS:
                old = next((key for key, record in records.items() if record["state"] == IV_ENDED), None)
                if old is None:
                    raise InterventionError("too many live intervention records", _w().ERR_LIMIT)
                del records[old]
            self._check_revision(revision)
            self._restore_records(list(records.values()), restore_active=True, evict_ended=True)
        self._ctl._publish()

    def _restore_records(self, records, *, restore_active, evict_ended=False):
        """Prepare a complete replacement, then commit if its input still owns the state."""
        from .interp._intervene import InterventionError
        revision = self._ctl._iv_revision
        restored = {}
        with self._ctl._outside_lock():
            for record in records:
                spec, applied = _record_specifications(record, self.executor)
                key = spec["point"], spec["op"]
                if key in restored:
                    raise InterventionError("duplicate intervention record")
                iv = _Intervention(*key, spec.get("from_step"), spec.get("until_step"), spec)
                iv.state, iv.since_step = record["state"], record["since_step"]
                iv.clear, iv.error, iv.error_code = record["clear"], record["error"], record["error_code"]
                iv.applied_spec = applied
                iv.replace = applied is not None and spec != applied
                restored[key] = iv
        self._check_revision(revision)
        try:
            restored = self._ctl._admit_interventions(restored, evict_ended=evict_ended)
        except ValueError as error:
            raise InterventionError(str(error), _w().ERR_LIMIT) from None
        cache, identity = self.executor._cache.copy(), self.executor._cache_identity
        retiring = [iv.hook for iv in self._ctl._ivs.values()]
        try:
            if restore_active:
                with self._ctl._outside_lock(), self.executor._retire_sources(retiring):
                    for iv in restored.values():
                        if iv.applied_spec is not None:
                            self._ctl._install_intervention(self, iv, iv.applied_spec)
            self._check_revision(revision)
            # Other controls can consume metadata space during preparation.
            try:
                restored = self._ctl._admit_interventions(restored, evict_ended=evict_ended)
            except ValueError as error:
                raise InterventionError(str(error), _w().ERR_LIMIT) from None
        except BaseException:
            with self._ctl._outside_lock():
                for iv in restored.values():
                    self._ctl._unhook(iv, retain_source=True)
                self.executor._cache.clear()
                self.executor._cache.update(cache)
                self.executor._cache_identity = identity
            raise
        old = list(self._ctl._ivs.values())
        self._ctl._ivs = restored
        self._ctl._iv_revision += 1
        self._ctl._iv_due = True
        self._ctl._update_busy()
        with self._ctl._outside_lock():
            for iv in old:
                self._ctl._unhook(iv, retain_source=True)
            self.executor._discard_unused_sources()

    def _check_revision(self, revision):
        from .interp._intervene import InterventionError
        if self._ctl._iv_model is not self or self._ctl._iv_revision != revision:
            raise InterventionError("intervention state changed during preparation", ERR_NOT_SUPPORTED)

    def remove(self) -> None:
        self._ctl._interventions_off(self)


def _record_specifications(record, executor):
    """Validate lifecycle structure before resources that an effect still needs."""
    from . import _intervention_protocol as metadata
    from .interp._intervene import InterventionError
    fields = {
        "specification", "state", "since_step", "clear", "error", "error_code"
    }
    if not isinstance(record, dict) or set(record) not in (fields, fields | {"applied_specification"}):
        raise InterventionError("invalid intervention record")
    state, since = record["state"], record["since_step"]
    if (state not in (IV_PENDING, IV_ACTIVE, IV_ENDED) or type(record["clear"]) is not bool
            or (state == IV_ACTIVE and since is None)
            or (state == IV_PENDING and since is not None)):
        raise InterventionError("invalid intervention lifecycle state")
    spec = executor.validate(record["specification"], resources=False)
    entry = dict(spec, state=state, since_step=since)
    entry.update({key: record[key] for key in ("error", "error_code") if record[key] is not None})
    if not metadata.specification(entry, lifecycle=True):
        raise InterventionError("invalid intervention lifecycle metadata")
    if state != IV_ENDED:
        spec = executor.validate(spec)
    # Older version 1 checkpoints had only the requested specification.
    applied = record.get("applied_specification", spec if state == IV_ACTIVE else None)
    if state == IV_ACTIVE:
        applied = executor.validate(applied)
        if (applied["point"], applied["op"]) != (spec["point"], spec["op"]):
            raise InterventionError("applied intervention must match the record point and kind")
    elif applied is not None:
        raise InterventionError("inactive intervention cannot have an applied specification")
    return spec, applied


def _step_ok(v) -> bool:
    return v is None or (isinstance(v, numbers.Integral) and not isinstance(v, bool)
                         and 0 <= v < _INT_EXACT)


class Route:
    """A transport attached to the controls; see the module docstring."""

    def __init__(self, ctl: "Controls", transport):
        self.ctl = ctl
        self.transport = transport
        self.attached = True

    def on_control(self, conn, meta, *, acknowledge=None):
        return self.ctl._on_control(self, conn, meta, acknowledge=acknowledge)

    def on_select(self, conn, meta, axes=None) -> None:
        try:
            self.ctl._on_select(meta, axes)
        except Exception:
            _log.debug("a select event failed", exc_info=True)

    def snapshot_state(self) -> list:
        return self.ctl.snapshot_state()

    def annotate_view(self, entry: dict) -> dict:
        return self.ctl.annotate_view(entry)

    def detach(self) -> None:
        self.ctl._detach(self)

    def _reply(self, conn, meta) -> None:
        if self.attached:
            try:
                self.transport.reply(conn, meta)
            except Exception:
                _log.debug("a reply failed", exc_info=True)

    def _broadcast(self, type_, meta, after_data: bool = False) -> None:
        if self.attached:
            try:
                if after_data:
                    self.transport.broadcast(type_, meta, after_data=True)
                else:
                    self.transport.broadcast(type_, meta)
            except Exception:
                _log.debug("a broadcast failed", exc_info=True)

    def _controllers(self) -> int:
        fn = getattr(self.transport, "controllers", None)
        if fn is None:
            return 1
        try:
            return int(fn())
        except Exception:
            return 0


# ===========================================================================
# the state
# ===========================================================================
class Controls:
    """The controls and selections of this process (``CONTROLS``)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._send = threading.RLock()           # orders broadcasts (taken before lock)
        self.busy = True                         # step() takes the slow path
        self.n = 0                               # the script's step number
        self.state = RUN_NONE
        self.steps_left = 0
        self._report_due = False                 # the next checkpoint sends the step
        self._report_armed = False               # a timer of RUN_REPORT_S is on its way
        self._steps_reported = 0.0
        self.params: dict[str, Param] = {}
        self.actions: dict[str, Action] = {}
        self.interpretation_service = None
        self.interpretation_registration_lock = threading.Lock()
        self.queue: collections.deque = collections.deque()
        self.spaces: dict[str, ItemSpace] = {}
        self.view_items: dict[str, dict] = {}    # view -> {dim, _FIRST, _LAST or None: space}
        self._view_args: dict[str, tuple] = {}
        self._resolved: dict[str, dict] = {}     # view -> {dim or None: space}, from VIEWS
        self.routes: list[Route] = []
        self._npending = 0
        self.wait_hooks: list = []               # called on the training thread before a pause
                                                 # waits (a route sends its last snapshots)
        self._timers = _Timers()
        self._details_worker = _DetailsWorker()
        self.counters = collections.Counter()
        self._iv_model: Interventions | None = None   # the watched model of the interventions
        self._ivs: dict = {}                     # (point, op) -> _Intervention, oldest first
        self._iv_due = False                     # the next checkpoint has intervention work
        self._iv_revision = 0                    # cancels preparation after a lifecycle edit
        self._iv_retiring = set()                # detached executors with work still in flight
        self._iv_attaching = False
        self._alone_since = None                 # no connection that may control since then,
                                                 # while an intervention acts or waits
        self._linger_next = 0.0                  # the next count of those connections

    @contextmanager
    def _outside_lock(self):
        """Release the caller's controls lock for executor work, then reacquire it."""
        self.lock.release()
        try:
            yield
        finally:
            self.lock.acquire()

    # -- the checkpoint (the training thread) ----------------------------------
    def step(self, step=None) -> None:
        """caliper.step(); see there."""
        if self._iv_model is not None and self._iv_model.executor.preparing_source:
            return  # A clean patch-source forward is internal to the current boundary.
        prospective = self.n + 1 if step is None else step
        if (isinstance(prospective, numbers.Integral) and not isinstance(prospective, bool)
                and prospective >= _INT_EXACT):
            # Ordinary counters fit the headroom reserved by every admission.
            # Larger counters retain support, but must fit before any state changes.
            with self.lock:
                meta = self._controls_meta()
                meta["run"]["step"] = int(prospective)
                _check_controls_budget(meta)
                self.n = prospective
        else:
            self.n = prospective
        if self.busy:
            self.checkpoint()

    def checkpoint(self) -> None:
        """The slow path of step(): something waits for this checkpoint."""
        publish = False
        with self.lock:
            if self.state == RUN_NONE:
                self.state = RUN_RUNNING
                publish = True
            elif self.state == RUN_STEPPING:
                self.steps_left -= 1
                if self.steps_left <= 0:
                    self.steps_left = 0
                    self.state = RUN_PAUSING
                elif time.monotonic() - self._steps_reported >= STEPS_REPORT_S:
                    publish = True
            if self._report_due:
                self._report_due = False
                publish = True
            self._update_busy()
        if publish:
            self._publish()
        self._service()
        with self.lock:
            pause = self.state == RUN_PAUSING
        if pause:
            self._wait()
        with self.lock:
            self._update_busy()

    def _update_busy(self) -> None:
        self.busy = (self.state in (RUN_NONE, RUN_PAUSING, RUN_STEPPING)
                     or self._npending > 0 or bool(self.queue) or self._report_due
                     or self._iv_due or (self.interpretation_service is not None
                                        and self.interpretation_service.pending))

    def _arm_report(self) -> None:
        """While the run runs, have the first checkpoint after RUN_REPORT_S send
        CONTROLS with its step (lock held)."""
        if self.state == RUN_RUNNING and not self._report_armed:
            self._report_armed = True
            self._timers.at(time.monotonic() + RUN_REPORT_S, self._report)

    def _report(self) -> None:
        with self.lock:
            self._report_armed = False
            if self.state == RUN_RUNNING and self.routes:
                self._report_due = True
                self.busy = True

    def _apply_pending(self) -> bool:
        """Set every pending value (lock held). True if one was set."""
        if not self._npending:
            return False
        for p in self.params.values():
            if p._pending is not _NO:
                p._value, p._pending = p._pending, _NO
                p._version += 1
        self._npending = 0
        return True

    def _resource_stop(self, error) -> bool:
        service = self.interpretation_service
        return service is not None and isinstance(error, getattr(service, "stop_exception", ()))

    def _raise_resource_stop(self) -> None:
        failure = getattr(self.interpretation_service, "failure", None)
        if failure is not None:
            raise failure

    def _service(self) -> None:
        """End queued actions when an interpretation resource stop reaches the owner."""
        try:
            self._raise_resource_stop()
            self._service_work()
        except Exception as error:
            if self._resource_stop(error):
                self.interpretation_service.stop(error, self.n)
                self._terminate_actions(error)
            raise

    @staticmethod
    def _stop_message(error):
        text = str(error).replace("\0", "").encode("utf-8", errors="replace").decode("utf-8")
        return _cut(text, TEXT_MAX)

    def _terminate_actions(self, error):
        with self.lock:
            pending = []
            while self.queue:
                action = self.queue.popleft()
                pending.append((action, action.waiters))
                action.queued, action.waiters = False, []
            self._update_busy()
        self._publish()
        for action, waiters in pending:
            self._action_replies(action, waiters, False, self._stop_message(error))

    def _service_work(self) -> None:
        """Set pending values and interventions, then run the queued actions in order."""
        controllers = self._controllers_due()
        with self.lock:
            changed = self._apply_pending()
            changed = self._apply_interventions(controllers) or changed
        if changed:
            self._publish()
        self._raise_resource_stop()
        if self.interpretation_service is not None:
            self.interpretation_service.step(self.n)
        while True:
            self._raise_resource_stop()
            with self.lock:
                if not self.queue:
                    return
                a = self.queue.popleft()
                a.queued, a.running = False, True
                waiters, a.waiters = a.waiters, []
                self._update_busy()
            self._publish()
            try:
                ok, text = self._run_action(a)
                self._raise_resource_stop()
            except BaseException as error:
                if self._resource_stop(error):
                    self.interpretation_service.stop(error, self.n)
                    self._action_replies(a, waiters, False, self._stop_message(error))
                raise
            finally:
                with self.lock:
                    a.running = False
                self._publish()
            self._action_replies(a, waiters, ok, text)

    def _action_replies(self, action, waiters, ok, text):
        w = _w()
        for route, conn, rid in waiters:
            meta = w.ControlReply(rid, w.OP_RUN, w.STATUS_OK if ok else w.STATUS_ERROR, action.name,
                message=(text or None) if ok else None,
                error=None if ok else {"code": w.ERR_ACTION_FAILED, "message": text}).to_meta()
            route._reply(conn, meta)

    def _run_action(self, a: Action):
        self.counters["actions_run"] += 1
        try:
            r = a.fn()
        except Exception as e:
            if self._resource_stop(e):
                raise
            text = _cut(f"{type(e).__name__}: {e}", TEXT_MAX)
            warnings.warn(f"caliper: action {a.name!r} raised {text}", RuntimeWarning,
                          stacklevel=2)
            return False, text
        return True, "" if r is None else _cut(str(r), TEXT_MAX)

    def _wait(self) -> None:
        """Hold the training thread in this checkpoint while the run is
        paused, setting values and running actions as they arrive."""
        with self.lock:
            if self.state != RUN_PAUSING:
                return
            self.state = RUN_PAUSED
        for hook in list(self.wait_hooks):
            try:
                hook()
            except Exception:
                _log.debug("a wait hook failed", exc_info=True)
        self._publish(after_data=True)            # "paused" follows the step's data
        absent_since = None
        try:
            while True:
                self._service()
                with self.lock:
                    if self.state != RUN_PAUSED:
                        return
                    routes = list(self.routes)
                count = sum(r._controllers() for r in routes) if routes else 0
                now = time.monotonic()
                if count > 0:
                    absent_since = None
                elif not routes or (absent_since is not None
                                    and now - absent_since >= RELEASE_GRACE_S):
                    self._release()
                    return
                elif absent_since is None:
                    absent_since = now
                with self.lock:
                    if self.state != RUN_PAUSED:
                        return
                    if not self.queue and not self._npending:
                        self.cond.wait(WAIT_POLL_S)
        except BaseException:
            self._release()                       # Ctrl-C: no wait at the next step()
            raise

    def _release(self) -> None:
        with self.lock:
            if self.state not in (RUN_PAUSED, RUN_PAUSING, RUN_STEPPING):
                return
            self.state = RUN_RUNNING
            self.steps_left = 0
            self._update_busy()
            self.cond.notify_all()
        self._publish()

    # -- the script's calls ------------------------------------------------------
    def pause(self) -> bool:
        """caliper.pause(); see there."""
        with self.lock:
            routes = list(self.routes)
        if not routes:
            warnings.warn("caliper.pause(): no stream server that carries controls runs, so "
                          "no viewer could resume the run; it goes on", RuntimeWarning,
                          stacklevel=3)
            return False
        if sum(r._controllers() for r in routes) == 0:
            warnings.warn(f"caliper.pause(): no connection may control the run now; it "
                          f"resumes after {RELEASE_GRACE_S:g} s unless one connects",
                          RuntimeWarning, stacklevel=3)
        with self.lock:
            if self.state in (RUN_NONE, RUN_RUNNING, RUN_STEPPING):
                self.state = RUN_PAUSING
                self.steps_left = 0
                self._update_busy()
                changed = True
            else:
                changed = False
        if changed:
            self._publish()
        return True

    def param(self, name, value, *, min=None, max=None, log=False, choices=None,
              help=None) -> Param:
        """caliper.param(); see there."""
        spec = _param_spec(name, value, min, max, log, choices, help)
        with self.lock:
            old = self.params.get(name)
            if old is None and len(self.params) >= MAX_PARAMS:
                raise ValueError(f"at most {MAX_PARAMS} parameters")
            p = Param(self, name, *spec)
            _run_value(p)
            self._check_metadata_budget(params={**self.params, name: p})
            if old is not None:
                old.removed = True
                if old._pending is not _NO:
                    self._npending -= 1
            self.params[name] = p
            self._update_busy()
        self._publish()
        return p

    def _script_set(self, p: Param, value):
        w = _w()
        try:
            v = w.param_value(p.entry(), _scalar(value))
        except w.MessageError as e:
            raise ValueError(e.message) from None
        with self.lock:
            if self.params.get(p.name) is p:
                entry = p.entry()
                entry["value"] = v
                entry.pop("pending", None)
                self._check_param_budget(p, entry)
            p._value = v
            p._version += 1
            if p._pending is not _NO:
                p._pending = _NO
                self._npending -= 1
            self._update_busy()
        if not p.removed:
            self._publish()
        return v

    def _remove_param(self, p: Param) -> bool:
        with self.lock:
            if self.params.get(p.name) is not p:
                return False
            del self.params[p.name]
            p.removed = True
            if p._pending is not _NO:
                p._value, p._pending = p._pending, _NO
                p._version += 1
                self._npending -= 1
            self._update_busy()
        self._publish()
        return True

    def action(self, name, fn, *, help=None) -> Action:
        """caliper.action(); see there."""
        _check_name(name, "action")
        if not callable(fn):
            raise TypeError("caliper.action() takes a callable")
        help = _check_help(help)
        with self.lock:
            old = self.actions.get(name)
            if old is None and len(self.actions) >= MAX_ACTIONS:
                raise ValueError(f"at most {MAX_ACTIONS} actions")
            a = Action(self, name, fn, help)
            self._check_metadata_budget(actions={**self.actions, name: a})
            if old is not None:
                # a run of the old one that waits becomes a run of the new one
                old.removed = True
                if old.queued:
                    a.queued, a.waiters, a.run_id = True, old.waiters, old.run_id
                    self.queue = collections.deque(a if x is old else x for x in self.queue)
                    old.queued, old.waiters = False, []
            self.actions[name] = a
        self._publish()
        return a

    def _remove_action(self, a: Action) -> bool:
        with self.lock:
            if self.actions.get(a.name) is not a:
                return False
            del self.actions[a.name]
            a.removed = True
            waiters = a.waiters if a.queued else []
            if a.queued:
                self.queue.remove(a)
                a.queued, a.waiters = False, []
            self._update_busy()
        self._publish()
        self._answer(waiters, a.name, "unknown_control", f"the action {a.name!r} was removed")
        return True

    def items(self, space, details=None) -> ItemSpace:
        """caliper.items(); see there."""
        _check_name(space, "item space")
        source = ItemSpace._details_source(details)
        with self.lock:
            sp = self.spaces.get(space)
            if sp is None:
                if len(self.spaces) >= MAX_ITEM_SPACES:
                    raise ValueError(f"at most {MAX_ITEM_SPACES} item spaces")
                sp = self.spaces[space] = ItemSpace(self, space)
            sp._table, sp._fields, sp._fn = source
        return sp

    # -- item spaces of views ------------------------------------------------------
    def view_items_call(self, name: str, kind: str, items) -> None:
        """A view call's ``items=``: remember which of its axes index which
        item space (cheap when nothing changed)."""
        args = (kind, items)
        if self._view_args.get(name) == args:
            return
        mapping = _view_dims(kind, items)
        with self.lock:
            spaces = set(self.spaces) | {s for m in self.view_items.values() for s in m.values()}
            if len(spaces | set(mapping.values())) > MAX_ITEM_SPACES:
                raise ValueError(f"a session uses at most {MAX_ITEM_SPACES} item spaces")
            changed = self.view_items.get(name) != mapping
            self.view_items[name] = mapping
            self._view_args[name] = args
            self._resolved.pop(name, None)
            routes = list(self.routes)
        if changed:
            for r in routes:
                fn = getattr(r.transport, "views_changed", None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        _log.debug("views_changed failed", exc_info=True)

    def forget_view(self, name: str) -> None:
        with self.lock:
            self.view_items.pop(name, None)
            self._view_args.pop(name, None)
            self._resolved.pop(name, None)

    def annotate_view(self, entry: dict) -> dict:
        name = entry.get("name")
        m = self.view_items.get(name)
        if not m:
            return entry
        out = dict(entry)
        resolved = {}
        if None in m and entry.get("kind") == 4:
            out["items"] = resolved[None] = m[None]
        sources = []
        for s in entry.get("sources") or []:
            dims = list(s.get("dims", []))
            new = []
            for i, d in enumerate(dims):
                space = m.get(d.get("name"))
                if space is None and i == 0:
                    space = m.get(_FIRST)
                if space is None and i == len(dims) - 1:
                    space = m.get(_LAST)
                if space is not None:
                    resolved[d["name"]] = space
                    d = dict(d, items=space)
                new.append(d)
            sources.append(dict(s, dims=new))
        out["sources"] = sources
        with self.lock:
            if self.view_items.get(name) is m:
                self._resolved[name] = resolved
        return out

    def _dims_of(self, view: str) -> dict:
        """{dim name or None: space} of a view: as its last VIEWS entry
        resolved it, or with the dim names of a view without dims (``row``
        and ``col``, ``point``)."""
        r = self._resolved.get(view)
        if r is not None:
            return r
        m = self.view_items.get(view) or {}
        out = {}
        for k, space in m.items():
            if k == _FIRST:
                out.setdefault("row", space)
            elif k == _LAST:
                out.setdefault("point", space)
            else:
                out[k] = space
        return out

    # -- requests (the transport's thread) -------------------------------------------
    def _on_control(self, route: Route, conn, meta, *, acknowledge=None):
        w = _w()
        if not isinstance(meta, dict):
            raise w.MessageError(w.ERR_BAD_REQUEST, "a control's metadata is an object")
        msg = w.Control.from_frame(w.Frame(w.MsgType.CONTROL, meta))
        rid, op, name = msg.request_id, msg.op, msg.name
        self.counters["controls_in"] += 1

        def error(code, text):
            return w.ControlReply(rid, op, w.STATUS_ERROR, name,
                                  error={"code": code, "message": text}).to_meta()

        if op == "interpret":
            from . import _interpret_protocol as protocol
            from .interp._service import Refused
            if msg.interpretation["method"] not in protocol.READ_METHODS:
                if not route.transport.may_control(conn):
                    return error(w.ERR_FORBIDDEN, "This connection may not change the run.")
            service = self.interpretation_service
            if service is None:
                return error("interpret.unsupported", "The run has no interpretation service.")

            def completed(status, result, refused):
                route._reply(conn, w.ControlReply(rid, op, status, result=result, error=refused).to_meta())

            try:
                queued = (lambda: acknowledge(w.ControlReply(rid, op, w.STATUS_QUEUED).to_meta())) \
                    if acknowledge is not None else None
                status, result = service.submit(msg.interpretation, completed, acknowledge=queued)
            except Refused as exc:
                return error(exc.code, str(exc))
            with self.lock:
                self._update_busy()
                if status == w.STATUS_QUEUED:
                    self.cond.notify_all()
            if status == w.STATUS_QUEUED and acknowledge is not None:
                return None
            return w.ControlReply(rid, op, status, result=result).to_meta()

        try:
            allowed = bool(route.transport.may_control(conn))
        except Exception:
            allowed = False
        if not allowed:
            return error(w.ERR_FORBIDDEN, "this connection may not control the run")
        reply = None
        timer = None
        with self.lock:
            if getattr(self.interpretation_service, "failure", None) is not None:
                return error("interpret.stopped", "The interpretation run stopped.")
            if op in (w.OP_PAUSE, w.OP_STEP) and self.state == RUN_NONE:
                return error(w.ERR_UNSUPPORTED, "the script has called no checkpoint "
                                                "(caliper.step()) yet")
            if op == w.OP_PAUSE:
                if self.state in (RUN_RUNNING, RUN_STEPPING):
                    self.state = RUN_PAUSING
                    self.steps_left = 0
            elif op == w.OP_RESUME:
                if self.state in (RUN_PAUSING, RUN_PAUSED, RUN_STEPPING):
                    self.state = RUN_RUNNING
                    self.steps_left = 0
            elif op == w.OP_STEP:
                self.state = RUN_STEPPING
                self.steps_left = msg.count or 1
                self._steps_reported = time.monotonic()
            elif op == w.OP_SET:
                p = self.params.get(name)
                if p is None:
                    return error(w.ERR_UNKNOWN_CONTROL, f"no parameter {name!r}")
                try:
                    v = w.param_value(p.entry(), msg.value)
                except w.MessageError as e:
                    return error(e.code, e.message)
                entry = p.entry()
                if self.state == RUN_NONE:
                    entry["value"] = v
                    entry.pop("pending", None)
                else:
                    entry["pending"] = v
                try:
                    self._check_param_budget(p, entry)
                except ValueError as e:
                    return error(w.ERR_LIMIT, str(e))
                if self.state == RUN_NONE:
                    p._value = v
                    p._version += 1
                    if p._pending is not _NO:
                        p._pending = _NO
                        self._npending -= 1
                else:
                    if p._pending is _NO:
                        self._npending += 1
                    p._pending = v
                reply = w.ControlReply(rid, op, w.STATUS_OK, name, v).to_meta()
            elif op == w.OP_RUN:
                a = self.actions.get(name)
                if a is None:
                    return error(w.ERR_UNKNOWN_CONTROL, f"no action {name!r}")
                if not a.queued:
                    a.queued = True
                    a.run_id += 1
                    self.queue.append(a)
                    timer = (a, a.run_id)
                a.waiters.append((route, conn, rid))
                reply = w.ControlReply(rid, op, w.STATUS_QUEUED, name).to_meta()
            elif op == OP_INTERVENE:
                refused = self._intervene(getattr(msg, "intervene", None))
                if refused is not None:
                    return error(*refused)
            self._update_busy()
            self.cond.notify_all()
        if timer is not None:
            self._timers.at(time.monotonic() + ACTION_WAIT_S, self._expire, *timer)
        self._publish()
        return reply or w.ControlReply(rid, op, w.STATUS_OK, name).to_meta()

    def _expire(self, a: Action, run_id: int) -> None:
        with self.lock:
            if not a.queued or a.run_id != run_id:
                return
            self.queue.remove(a)
            a.queued = False
            waiters, a.waiters = a.waiters, []
            self._update_busy()
        self._publish()
        self._answer(waiters, a.name, "timeout",
                     f"no checkpoint ran the action within {ACTION_WAIT_S:g} s")

    def _answer(self, waiters, name, code, text) -> None:
        if not waiters:
            return
        w = _w()
        for route, conn, rid in waiters:
            route._reply(conn, w.ControlReply(rid, w.OP_RUN, w.STATUS_ERROR, name, error={
                "code": code, "message": text}).to_meta())

    # -- selections ----------------------------------------------------------------
    def _on_select(self, meta, axes=None) -> None:
        if not isinstance(meta, dict) or meta.get("type") != "select":
            return
        view, detail = meta.get("view"), meta.get("detail")
        m = self._dims_of(view) if isinstance(view, str) else None
        if not m or not isinstance(detail, dict):
            return
        self.counters["selects_in"] += 1
        for space, ids, cut in _event_items(m, detail, axes):
            self._set_selection(space, ids, view, cut)

    def _set_selection(self, space: str, ids, view, cut: bool) -> None:
        ids, cut2 = _normalise(ids)
        submit = None
        with self.lock:
            sp = self.spaces.get(space)
            if sp is None:
                if len(self.spaces) >= MAX_ITEM_SPACES:
                    return
                sp = self.spaces[space] = ItemSpace(self, space)
            sp._selection, sp._truncated, sp._view = ids, cut or cut2, view
            sp._version += 1
            sp._details, sp._details_error, sp._settled = None, None, True
            if ids and sp._table is not None:
                sp._details = sp._lookup(ids[:MAX_DETAIL_ITEMS])
            elif ids and sp._fn is not None:
                sp._settled = False
                submit = (sp, sp._version)
        self._publish_selection(sp)
        if submit is not None:
            sp_, version = submit
            self._details_worker.submit(sp_.name, lambda: self._run_details(sp_, version))
            self._timers.at(time.monotonic() + DETAILS_TIMEOUT_S, self._details_late, sp_,
                            version)

    def _run_details(self, sp: ItemSpace, version: int) -> None:
        with self.lock:
            if sp._version != version or sp._settled:
                return
            ids, fn = sp._selection[:MAX_DETAIL_ITEMS], sp._fn
        details = error = None
        try:
            details = _details_from(fn(list(ids)), ids)
        except Exception as e:
            error = _cut(f"{type(e).__name__}: {e}", TEXT_MAX)
        with self.lock:
            if sp._version != version or sp._settled:
                return
            sp._settled = True
            sp._details, sp._details_error = details, error
        self._publish_selection(sp)

    def _details_late(self, sp: ItemSpace, version: int) -> None:
        with self.lock:
            if sp._version != version or sp._settled:
                return
            sp._settled = True
            sp._details_error = f"timeout: the details function took more than " \
                                f"{DETAILS_TIMEOUT_S:g} s"
        self._publish_selection(sp)

    # -- interventions (design/gpu-programs.md, Interventions) -----------------------
    @property
    def intervention_service(self):
        """The intervention handle for the current watch, or None."""
        return self._iv_model

    def interventions_on(self, model, handle=None, adapter=None) -> Interventions:
        """Take interventions on ``model`` from now on (the training thread).

        ``caliper.watch(model)`` calls this with or without an optimizer. CONTROLS holds the key
        ``interventions`` with the points of ``model`` and its interventions, and a CONTROL of op
        ``intervene`` sets or clears one, applied at the next checkpoint through the hooks of
        ``adapter`` (``caliper._adapter.TORCH`` by default). A later call replaces the model and
        takes every hook of the earlier one off. The output shapes of the points are recorded at
        the next forward pass.

        Args:
            model (torch.nn.Module): The watched model.
            handle (WatchHandle | None): When given, its ``remove()`` also stops the
                interventions.
            adapter: The framework adapter, with ``points(model)`` and ``ablate(model, point)``.

        Returns:
            Interventions: Its ``remove()`` takes every hook off and stops the interventions.
        """
        if adapter is None:
            from ._adapter import TORCH as adapter
        from . import _ablate
        m = Interventions(self, model, adapter)
        pts = list(adapter.points(model))
        m.points = pts[:MAX_POINTS_LISTED]
        m.names = frozenset(p["point"] for p in pts)
        m.executor._ablation_points = m.names
        with self.lock:
            from .interp._intervene import InterventionError
            if self._iv_attaching or self._iv_retiring:
                raise InterventionError("an intervention lifecycle operation is in progress", ERR_NOT_SUPPORTED)
            self._check_metadata_budget(points=m.points, ivs={})
            old = self._iv_model
            with old._operation("watch") if old is not None else nullcontext():
                self._iv_attaching = True
                try:
                    try:
                        with self._outside_lock():
                            m._shapes_off = _ablate.record_shapes(model)
                    except Exception:
                        _log.debug("the shape hooks failed", exc_info=True)
                    if self._iv_model is not old:
                        raise InterventionError("intervention state changed during preparation", ERR_NOT_SUPPORTED)
                    self._check_metadata_budget(points=m.points, ivs={})
                    ivs = list(self._ivs.values())
                    self._ivs = {}
                    self._iv_model = m
                    self._iv_revision += 1
                    self._iv_due = m._shapes_off is not None
                    self._update_busy()
                    if old is not None:
                        with self._outside_lock():
                            self._close_interventions(old, ivs)
                except BaseException:
                    with self._outside_lock():
                        self._close_interventions(m, ())
                    raise
                finally:
                    self._iv_attaching = False
        if handle is not None and hasattr(handle, "_handles"):
            handle._handles.append(m)
        self._publish()
        return m

    def _interventions_off(self, m: Interventions) -> None:
        """Take every hook of ``m`` off its model and forget its interventions."""
        with self.lock:
            if self._iv_model is not m:
                return
            self._iv_model = None
            ivs, self._ivs = list(self._ivs.values()), {}
            self._iv_revision += 1
            self._iv_due = False
            self._update_busy()
            m._retired = ivs
            self._iv_retiring.add(m)
            self._finish_retired(m)
        self._publish()

    def _finish_retired(self, m):
        """Close detached resources only after their last executor call returns."""
        if m._retired is None or m._operation_owner is not None or m._readers:
            return
        ivs, m._retired = m._retired, None
        try:
            with self._outside_lock():
                self._close_interventions(m, ivs)
        finally:
            self._iv_retiring.discard(m)

    def _close_interventions(self, m, ivs):
        shapes_off, m._shapes_off = m._shapes_off, None
        for iv in ivs:
            self._unhook(iv)
        m.executor.close()
        m.executor.dictionary = m.executor.tokenizer = None
        if shapes_off is not None:
            shapes_off()

    @staticmethod
    def _unhook(iv: _Intervention, *, retain_source=False) -> None:
        hook, iv.hook = iv.hook, None
        iv.applied_spec = None
        if hook is not None:
            try:
                from .interp._intervene import Intervention
                if retain_source and isinstance(hook, Intervention):
                    hook._remove(retain_source=True)
                else:
                    hook()
            except Exception:                     # a hook handle's remove() does not fail
                _log.debug("removing the hook of %s failed", iv.point, exc_info=True)

    def _install_intervention(self, service, iv, spec, *, owner=None):
        """Install one effect with the same error callback on restore and at a boundary."""
        from copy import deepcopy
        callback = lambda message, iv=iv if owner is None else owner: self._intervention_failed(iv, message)
        if iv.op == "ablate":
            iv.hook = service.adapter.ablate(service.model, iv.point)
            if hasattr(iv.hook, "on_error"):
                iv.hook.on_error = callback
        else:
            iv.hook = service.executor.install(spec, on_error=callback)
        iv.applied_spec = deepcopy(spec)

    def _intervene(self, req):
        """Take the ``intervene`` key of a CONTROL (lock held). Returns ``(code, text)`` of a
        refusal, or None."""
        m = self._iv_model
        if m is None:
            return (ERR_NOT_SUPPORTED, "the script takes no interventions: it watches no model "
                                       "with caliper.watch(model)")
        if (not isinstance(req, dict) or set(req) - {"set", "clear", "clear_all"}
                or sum(req.get(k) is not None for k in ("set", "clear", "clear_all")) != 1):
            return (_w().ERR_BAD_REQUEST, "intervene holds exactly one of set, clear or clear_all")
        if m._operation_owner is not None and m._operation_owner[0] in ("configure", "watch"):
            return (ERR_NOT_SUPPORTED, "an intervention lifecycle operation is in progress")
        if req.get("set") is not None:
            from .interp._intervene import InterventionError
            m._readers += 1
            try:
                with self._outside_lock():
                    spec = m.executor.validate(req["set"])
            except InterventionError as error:
                return error.code, str(error)
            finally:
                m._readers -= 1
                self._finish_retired(m)
            if self._iv_model is not m:
                return (ERR_NOT_SUPPORTED, "the intervention service is no longer attached")
            if getattr(self.interpretation_service, "failure", None) is not None:
                return ("interpret.stopped", "The interpretation run stopped.")
            nxt = self._next_step()
            point, op = spec["point"], spec["op"]
            frm, until = spec.get("from_step"), spec.get("until_step")
            if until is not None and until <= nxt:
                return (ERR_BAD_RANGE, f"until_step {until} is in the past: the next step is {nxt}")
            key = (point, op)
            iv = self._ivs.get(key)
            candidate = _Intervention(point, op, frm, until, spec)
            if iv is not None and iv.state != IV_ENDED:
                candidate.state, candidate.since_step = iv.state, iv.since_step
            try:
                prospective = self._admit_interventions({**self._ivs, key: candidate}, evict_ended=True)
            except ValueError as error:
                return _w().ERR_LIMIT, str(error)
            if iv is not None:
                # Keep the record identity: an installed hook's callback owns it.
                iv.replace = iv.applied_spec is not None and spec != iv.applied_spec
                iv.spec = spec
                iv.from_step, iv.until_step = frm, until
                iv.clear, iv.error, iv.error_code = False, None, None
                if iv.state == IV_ENDED:
                    iv.state, iv.since_step = IV_PENDING, None
                prospective[key] = iv
            self._ivs.clear()
            self._ivs.update(prospective)
        elif req.get("clear") is not None:
            point = req["clear"]
            if not isinstance(point, str) or point not in m.names:
                return (ERR_UNKNOWN_POINT, "the model has no such point")
            for iv in self._ivs.values():
                if iv.point == point:
                    iv.clear = True
        elif req.get("clear_all") is True:
            for iv in self._ivs.values():
                iv.clear = True
        else:
            return (_w().ERR_BAD_REQUEST, "intervene holds set, clear or clear_all")
        self._iv_due = True
        self._iv_revision += 1
        self.counters["interventions_in"] += 1
        return None

    def _next_step(self) -> int:
        """The step that the next forward pass belongs to: one past the latest checkpoint."""
        n = self.n
        return int(n) + 1 if isinstance(n, numbers.Integral) and not isinstance(n, bool) else 1

    def _controllers_due(self):
        """The number of connections that may control, when a count is due: while an intervention
        acts or waits, at most every ``LINGER_CHECK_S``. None otherwise. The routes are counted
        outside the lock, as the pause's wait counts them."""
        now = time.monotonic()
        with self.lock:
            if not any(iv.state != IV_ENDED and not iv.clear for iv in self._ivs.values()):
                self._alone_since = None             # a later intervention waits a whole linger
                return None
            if now < self._linger_next:
                return None
            self._linger_next = now + LINGER_CHECK_S
            routes = list(self.routes)
        return sum(r._controllers() for r in routes) if routes else 0

    def _linger(self, controllers: int, records=None) -> bool:
        """End every intervention once no connection that may control stayed connected for the
        linger time that also removes the GPU programs (``caliper._net_programs.LINGER_S``):
        nobody could see or clear it any more (lock held). True if one ended."""
        from ._net_programs import LINGER_S
        now = time.monotonic()
        if controllers > 0:
            self._alone_since = None
            return False
        if self._alone_since is None:
            self._alone_since = now
        if now - self._alone_since < LINGER_S:
            return False
        if records is None:
            self._alone_since = None
        from copy import copy
        ended, removed = [], []
        for iv in (self._ivs if records is None else records).values():
            if iv.state == IV_ENDED or iv.clear:
                continue
            removed.append(copy(iv))
            iv.hook, iv.applied_spec = None, None
            iv.state = IV_ENDED
            iv.error = self._stop_message(
                f"ended: no connection that may control the run stayed for {LINGER_S:g} s")
            iv.error_code = "intervene.no_controller"
            ended.append(iv.point)
        if ended:
            if records is None:
                self._iv_revision += 1
                with self._outside_lock():
                    for iv in removed:
                        self._unhook(iv)
                    self._report_linger(ended)
        return bool(ended)

    def _report_linger(self, ended):
        from ._net_programs import LINGER_S
        self.counters["interventions_lingered"] += len(ended)
        warnings.warn(f"caliper: no connection that may control the run stayed for {LINGER_S:g} s, "
                      f"so the interventions on {', '.join(ended)} ended", RuntimeWarning,
                      stacklevel=2)

    def _intervention_failed(self, iv, message):
        """End a failed hook now and publish its error at the next boundary."""
        from copy import copy
        with self.lock:
            if self._ivs.get((iv.point, iv.op)) is not iv:
                return
            removed = copy(iv)
            iv.hook, iv.applied_spec = None, None
            iv.state, iv.error = IV_ENDED, self._stop_message(message)
            iv.error_code, iv.report = "intervene.hook_failed", True
            self._iv_revision += 1
            self._iv_due = True
            self._update_busy()
            with self._outside_lock():
                self._unhook(removed)

    def _apply_interventions(self, controllers=None) -> bool:
        """Prepare boundary effects outside the lock and commit one unchanged revision.

        Due lifecycle work completes first. Concurrent requests cancel speculative
        additions and remain queued. A failed preparation removes only its new hooks.
        """
        from copy import copy
        if not self._iv_due:
            return False
        m = self._iv_model
        if m is None:
            self._iv_due = False
            return False
        if m._operation_owner is not None or self._iv_attaching:
            return False
        with m._operation("boundary"):
            completed, traffic = self._complete_due_interventions(m, self._next_step())
            if traffic or self._iv_model is not m:
                return completed
            revision = self._iv_revision
            originals = dict(self._ivs)
            records = {key: copy(iv) for key, iv in originals.items()}
            previous = {key: copy(iv) for key, iv in originals.items()}
            changed = controllers is not None and self._linger(controllers, records)
            lingered = [iv.point for key, iv in records.items()
                        if changed and iv.state == IV_ENDED and previous[key].state != IV_ENDED]
            step = self._next_step()
            cache, identity = m.executor._cache.copy(), m.executor._cache_identity
            committed = False
            reports = []
            points_changed = False
            try:
                if m._shapes_off is not None and self.state != RUN_NONE:
                    shapes_off, m._shapes_off = m._shapes_off, None
                    try:
                        with self._outside_lock():
                            shapes_off()
                            points = list(m.adapter.points(m.model))[:MAX_POINTS_LISTED]
                        # Shapes describe the model, independent of queued effects.
                        # Admit them against the latest metadata even if requests changed.
                        if self._iv_model is m:
                            self._check_metadata_budget(points=points)
                            m.points = points
                            points_changed = True
                    except ValueError as error:
                        warnings.warn(f"caliper: point metadata update refused: {error}",
                                      RuntimeWarning, stacklevel=2)
                    except Exception:
                        _log.debug("the shapes of the points failed", exc_info=True)
                    changed = True
                with (self._outside_lock(), m.executor._retire_sources(
                        self._boundary_retirements(records, step))):
                    prepared, reports = self._prepare_interventions(m, records, originals, step)
                    if prepared is None:
                        for message in reports:
                            warnings.warn(message, RuntimeWarning, stacklevel=2)
                        return points_changed or completed
                    changed = prepared or changed
                if self._iv_model is not m or self._iv_revision != revision:
                    return points_changed or completed
                self._check_metadata_budget(ivs=records)
                for key, iv in originals.items():
                    if key not in records:
                        del self._ivs[key]
                    else:
                        prepared = records[key]
                        for field in _Intervention.__slots__:
                            setattr(iv, field, getattr(prepared, field))
                self._iv_revision += 1
                self._iv_due = m._shapes_off is not None or any(
                    iv.clear or iv.state != IV_ENDED for iv in self._ivs.values())
                if lingered:
                    self._alone_since = None
                committed = True
                return changed or completed
            finally:
                with self._outside_lock():
                    if committed:
                        for key, iv in previous.items():
                            if key not in records or iv.hook is not records[key].hook:
                                self._unhook(iv, retain_source=True)
                        m.executor._discard_unused_sources()
                        if lingered:
                            self._report_linger(lingered)
                        for message in reports:
                            warnings.warn(message, RuntimeWarning, stacklevel=2)
                    else:
                        for key, iv in records.items():
                            if iv.hook is not previous[key].hook:
                                self._unhook(iv, retain_source=True)
                        m.executor._cache.clear()
                        m.executor._cache.update(cache)
                        m.executor._cache_identity = identity

    def _complete_due_interventions(self, m, step):
        """Complete expirations, clears, and scheduled starts in one bounded pass.

        Source callbacks may queue requests. Only lifecycle fields from this pass
        commit, so later requests retain their specification and clear flag.
        """
        from copy import copy
        from .interp._intervene import InterventionError

        removed = []
        changed = False
        for key, iv in list(self._ivs.items()):
            expired = iv.until_step is not None and iv.until_step <= step
            applied_until = (iv.applied_spec or {}).get("until_step")
            applied_expired = applied_until is not None and applied_until <= step
            if not iv.clear and not (iv.state != IV_ENDED and (expired or applied_expired)):
                continue
            removed.append(copy(iv))
            if iv.clear:
                del self._ivs[key]
            else:
                iv.hook, iv.applied_spec, iv.replace = None, None, False
                iv.state = IV_ENDED if expired else IV_PENDING
                if not expired:
                    iv.since_step = None
            changed = True
        if changed:
            self._iv_revision += 1
            with self._outside_lock():
                for iv in removed:
                    self._unhook(iv)
        if self._iv_model is not m:
            return changed, True

        scheduled = {key for key, iv in self._ivs.items()
                     if (iv.state != IV_ENDED and not iv.clear and iv.hook is None
                         and iv.from_step is not None and iv.from_step <= step)}
        if not scheduled:
            return changed, False
        revision = self._iv_revision
        originals = dict(self._ivs)
        previous = {key: copy(iv) for key, iv in originals.items()
                    if iv.hook is not None or key in scheduled}
        records = {key: copy(iv) for key, iv in previous.items()}
        for iv in records.values():
            if iv.hook is not None:
                # Reorder existing effects with their applied specification. A
                # replacement request belongs to the later speculative phase.
                iv.spec = iv.applied_spec
                iv.from_step, iv.until_step = iv.spec.get("from_step"), iv.spec.get("until_step")
                iv.replace, iv.clear = False, False
        cache, identity = m.executor._cache.copy(), m.executor._cache_identity
        committed = False
        reports = []
        try:
            with (self._outside_lock(), m.executor._retire_sources(
                    self._boundary_retirements(records, step))):
                prepared, reports = self._prepare_interventions(m, records, originals, step)
            if self._iv_model is not m:
                return changed, True
            if prepared is None:
                raise InterventionError("scheduled intervention preparation failed", ERR_NOT_SUPPORTED)
            prospective = {key: copy(iv) for key, iv in self._ivs.items()}
            for key, prepared in records.items():
                current = prospective.get(key)
                if (self._ivs.get(key) is not originals[key]
                        or current.hook is not previous[key].hook):
                    raise InterventionError("scheduled intervention ownership changed", ERR_NOT_SUPPORTED)
                current.hook, current.applied_spec = prepared.hook, prepared.applied_spec
                if key in scheduled or prepared.state != previous[key].state:
                    current.state, current.since_step = prepared.state, prepared.since_step
                    current.error, current.error_code = prepared.error, prepared.error_code
                    if prepared.state == IV_ENDED and current.spec != previous[key].spec:
                        # A newer request survives a failure of the captured schedule.
                        current.state, current.since_step = IV_PENDING, None
                        current.error, current.error_code = None, None
                current.replace = current.applied_spec is not None and current.spec != current.applied_spec
            self._check_metadata_budget(ivs=prospective)
            traffic = self._iv_revision != revision
            for key in records:
                iv = originals[key]
                for field in _Intervention.__slots__:
                    setattr(iv, field, getattr(prospective[key], field))
            self._iv_revision += 1
            committed = True
            return True, traffic
        finally:
            with self._outside_lock():
                if committed:
                    for key, iv in previous.items():
                        if iv.hook is not records[key].hook:
                            self._unhook(iv, retain_source=True)
                    m.executor._discard_unused_sources()
                    for message in reports:
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                else:
                    for key, iv in records.items():
                        if iv.hook is not previous[key].hook:
                            self._unhook(iv, retain_source=True)
                    m.executor._cache.clear()
                    m.executor._cache.update(cache)
                    m.executor._cache_identity = identity

    @staticmethod
    def _boundary_retirements(records, step):
        """Identify old hooks that a successful boundary removes or replaces."""
        reinstall = False
        retired = []
        for iv in records.values():
            ended = (iv.state == IV_ENDED or iv.clear
                     or (iv.until_step is not None and iv.until_step <= step)
                     or (iv.hook is not None and getattr(iv.hook, "errors", 0)))
            on = not ended and (iv.from_step is None or iv.from_step <= step)
            reinstall = reinstall or (on and (iv.hook is None or iv.replace))
            if iv.hook is not None and (not on or iv.replace or reinstall):
                retired.append(iv.hook)
        return retired

    def _prepare_interventions(self, m, records, originals, s):
        """Build the next hook sequence without changes to the published records."""
        from .interp._memory import CleanStop

        changed = False
        reinstall = False
        reports = []
        for key in list(records):
            iv = records[key]
            if iv.report:
                iv.report = False
                changed = True
                reports.append(f"caliper: the intervention at {iv.point!r} failed and ended: {iv.error}")
            if iv.replace:
                iv.hook, iv.applied_spec = None, None
                iv.replace = False
                changed = True
            if iv.clear:
                del records[key]
                changed = True
                continue
            if iv.state == IV_ENDED:
                continue
            failed = iv.hook is not None and getattr(iv.hook, "errors", 0)
            if failed:
                iv.error = self._stop_message(getattr(iv.hook, "error", None) or "the hook failed")
                iv.error_code = "intervene.hook_failed"
            ended = failed or (iv.until_step is not None and iv.until_step <= s)
            on = not ended and (iv.from_step is None or iv.from_step <= s)
            if on and (iv.hook is None or reinstall):
                # Keep registration order equal to record order, including a late
                # activation or replacement before an existing overlapping effect.
                iv.hook, iv.applied_spec = None, None
                try:
                    self._install_intervention(m, iv, iv.spec, owner=originals[key])
                    reinstall = True
                except Exception as e:
                    if isinstance(e, CleanStop) or self._resource_stop(e):
                        raise
                    iv.error = self._stop_message(f"{type(e).__name__}: {e}")
                    iv.error_code = "intervene.hook_failed"
                    ended, on = True, False
                    reports.append(f"caliper: the intervention at {iv.point!r} failed: {iv.error}")
                    if originals[key].hook is not None:
                        # A replacement failure rolls back the complete hook sequence.
                        return None, reports
            if not on:
                iv.hook, iv.applied_spec = None, None
            state = IV_ACTIVE if on else IV_ENDED if ended else IV_PENDING
            if state != iv.state:
                iv.state = state
                iv.since_step = s if on else iv.since_step if ended else None
                changed = True
                if failed:
                    reports.append(f"caliper: the intervention at {iv.point!r} failed and ended: {iv.error}")
        return changed, reports

    # -- messages ------------------------------------------------------------------
    def _run_obj(self) -> dict:
        r = {"state": self.state}
        s = self.n if self.state != RUN_NONE else None     # the step of the latest checkpoint
        if isinstance(s, numbers.Integral) and not isinstance(s, bool) and s >= 0:
            r["step"] = int(s)
        if self.state == RUN_STEPPING:
            r["steps_left"] = self.steps_left
        return r

    def controls_meta(self) -> dict:
        """The metadata of the CONTROLS message of the current state."""
        with self.lock:
            return self._controls_meta()

    def _controls_meta(self) -> dict:
        meta = {"run": self._run_obj(), "params": [p.entry() for p in self.params.values()],
                "actions": [a.entry() for a in self.actions.values()]}
        if self._iv_model is not None:   # a training or inference watch
            meta["interventions"] = self._interventions_obj()
        return meta

    def _check_metadata_budget(self, *, params=None, actions=None, ivs=None, points=None):
        """Check prospective collections without state changes, with the lock held."""
        meta = self._controls_meta()
        if params is not None:
            meta["params"] = [p.entry() for p in params.values()]
        if actions is not None:
            meta["actions"] = [a.entry() for a in actions.values()]
        if points is not None:
            meta.setdefault("interventions", {"active": []})["points"] = points
        if ivs is not None:
            meta["interventions"]["active"] = [iv.entry() for iv in ivs.values()]
        _check_controls_budget(meta)

    def _check_param_budget(self, param, entry):
        meta = self._controls_meta()
        meta["params"] = [entry if p is param else p.entry() for p in self.params.values()]
        _check_controls_budget(meta)

    def _admit_interventions(self, ivs, *, evict_ended=False):
        """Return a bounded collection. Evict history only after complete admission."""
        ivs = dict(ivs)
        while True:
            try:
                if len(ivs) > MAX_INTERVENTIONS:
                    raise ValueError(f"the script holds {MAX_INTERVENTIONS} interventions")
                self._check_metadata_budget(ivs=ivs)
                return ivs
            except ValueError:
                old = next((key for key, iv in ivs.items() if iv.state == IV_ENDED), None)
                if not evict_ended or old is None:
                    raise
                del ivs[old]

    def _interventions_obj(self) -> dict:
        return {"points": list(self._iv_model.points),
                "active": [iv.entry() for iv in self._ivs.values()]}

    def _selection_meta(self, sp: ItemSpace) -> dict:
        w = _w()
        return w.Selection(sp.name, list(sp._selection), sp._view, sp._truncated, sp._details,
                           sp._details_error).to_meta()

    def snapshot_state(self) -> list:
        w = _w()
        with self.lock:
            out = [(w.MsgType.CONTROLS, self._controls_meta())]
            out += [(w.MsgType.SELECTION, self._selection_meta(sp))
                    for sp in self.spaces.values() if sp._selection]
        return out

    def _publish(self, after_data: bool = False) -> None:
        if not self.routes:
            return
        w = _w()
        with self._send:
            with self.lock:
                meta = self._controls_meta()
                routes = list(self.routes)
                if self.state == RUN_STEPPING:
                    self._steps_reported = time.monotonic()
                self._arm_report()
            self.counters["controls_out"] += 1
            for r in routes:
                r._broadcast(w.MsgType.CONTROLS, meta, after_data)

    def _publish_selection(self, sp: ItemSpace) -> None:
        if not self.routes:
            return
        w = _w()
        with self._send:
            with self.lock:
                meta = self._selection_meta(sp)
                routes = list(self.routes)
            self.counters["selections_out"] += 1
            for r in routes:
                r._broadcast(w.MsgType.SELECTION, meta)

    # -- routes ----------------------------------------------------------------------
    def attach(self, transport) -> Route:
        r = Route(self, transport)
        with self.lock:
            self.routes.append(r)
            self._arm_report()
        return r

    def _detach(self, route: Route) -> None:
        with self.lock:
            route.attached = False
            if route in self.routes:
                self.routes.remove(route)
            self.cond.notify_all()                # a waiting checkpoint looks at the routes

    def summary(self) -> dict:
        """The state for stats() and tests: run, params, actions, selections."""
        with self.lock:
            return {"run": self._run_obj(),
                    "params": {n: p._value for n, p in self.params.items()},
                    "pending": {n: p._pending for n, p in self.params.items()
                                if p._pending is not _NO},
                    "actions": {n: a.entry() for n, a in self.actions.items()},
                    "selections": {n: list(s._selection) for n, s in self.spaces.items()},
                    "interventions": [dict(iv.entry(), error=iv.error)
                                      for iv in self._ivs.values()],
                    "routes": len(self.routes), **dict(self.counters)}


def _normalise(ids):
    """Item ids ascending without repeats, cut to MAX_SELECTION; and whether cut."""
    out = sorted({int(i) for i in ids if isinstance(i, numbers.Integral)
                  and not isinstance(i, bool) and i >= 0})
    return out[:MAX_SELECTION], len(out) > MAX_SELECTION


def _ids_list(v):
    """A list of item ids from an event (at most MAX_SELECTION + 1 looked at)."""
    if not isinstance(v, list):
        return None, False
    cut = len(v) > MAX_SELECTION
    return v[:MAX_SELECTION], cut


def _range_ids(r):
    """The ids of a half-open [start, stop] range from an event, bounded."""
    if not (isinstance(r, list) and len(r) == 2 and all(
            isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in r)):
        return None, False
    a, b = max(int(r[0]), 0), int(r[1])
    if b <= a:
        return [], False
    return range(a, min(b, a + MAX_SELECTION)), b - a > MAX_SELECTION


def _axis_rows(axis: dict, positions) -> list:
    """Source indices of payload positions along a sampled axis (the point
    dim): position i is source index start + floor(i * count / size) * step."""
    start, step, count, size = (int(axis["start"]), int(axis["step"]), int(axis["count"]),
                                int(axis["size"]))
    return [start + (int(i) * count // size) * step for i in positions
            if isinstance(i, numbers.Integral) and 0 <= i < size]


def _event_items(m: dict, detail: dict, axes=None):
    """The (space, ids, truncated) a select event's detail selects in the
    item spaces of its view (mapping ``m``: {dim or None: space})."""
    spaces = set(m.values())
    if not detail:                                        # {} clears the view's spaces
        return [(s, [], False) for s in sorted(spaces)]
    space = detail.get("space")
    if space is not None:                                 # a 3.1 viewer names it
        ids, cut = _ids_list(detail.get("items"))
        if space not in spaces or ids is None:
            return []
        return [(space, ids, cut or detail.get("truncated") is True)]
    # Otherwise from the kind's own keys: points by row_ids (or their indices in
    # the sent sample, through the point axis), heatmaps by rows or columns,
    # series by steps.
    cut = detail.get("truncated") is True
    if "row_ids" in detail or "indices" in detail:
        dims = [k for k in m if k is not None]
        dim = axes[-1]["dim"] if axes else (dims[0] if len(dims) == 1 else "point")
        space = m.get(dim)
        if space is None:
            return []
        ids, c = _ids_list(detail.get("row_ids"))
        if ids is None:
            pos, c = _ids_list(detail.get("indices"))
            if pos is None or not axes:
                return []
            ids = _axis_rows(axes[-1], pos)
        return [(space, ids, cut or c)]
    if "steps" in detail:
        space = m.get(None)
        ids, c = _range_ids(detail.get("steps"))
        return [] if space is None or ids is None else [(space, ids, cut or c)]
    if "rows" in detail or "cols" in detail:
        row_dim = axes[-2]["dim"] if axes and len(axes) >= 2 else "row"
        col_dim = axes[-1]["dim"] if axes else "col"
        for dim, key in ((row_dim, "rows"), (col_dim, "cols")):
            if dim in m:
                ids, c = _range_ids(detail.get(key))
                if ids is not None:
                    return [(m[dim], ids, cut or c)]
    return []


CONTROLS = Controls()


def attach(transport) -> Route:
    """Attach a transport to the controls of this process; see the module
    docstring for what it provides and what it gets."""
    return CONTROLS.attach(transport)
