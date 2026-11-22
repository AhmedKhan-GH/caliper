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

import collections
import heapq
import itertools
import logging
import math
import numbers
import threading
import time
import warnings
from typing import Any, Callable

__all__ = ["CONTROLS", "Controls", "Param", "Action", "ItemSpace", "Route", "attach",
           "CAPABILITIES"]

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
MAX_CHOICES = 64
TEXT_MAX = 1024
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
        type (str): `"boolean"`, `"integer"`, `"number"` or `"choice"`.
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
        raise TypeError(f"parameter {name!r}: a str value needs choices=[...]")
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
class Route:
    """A transport attached to the controls; see the module docstring."""

    def __init__(self, ctl: "Controls", transport):
        self.ctl = ctl
        self.transport = transport
        self.attached = True

    def on_control(self, conn, meta) -> dict:
        return self.ctl._on_control(self, conn, meta)

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

    # -- the checkpoint (the training thread) ----------------------------------
    def step(self, step=None) -> None:
        """caliper.step(); see there."""
        if step is None:
            self.n += 1
        else:
            self.n = step
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
                     or self._npending > 0 or bool(self.queue) or self._report_due)

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

    def _service(self) -> None:
        """Set pending values, then run the queued actions in order."""
        with self.lock:
            changed = self._apply_pending()
        if changed:
            self._publish()
        while True:
            with self.lock:
                if not self.queue:
                    return
                a = self.queue.popleft()
                a.queued, a.running = False, True
                waiters, a.waiters = a.waiters, []
                self._update_busy()
            self._publish()
            ok, text = self._run_action(a)
            with self.lock:
                a.running = False
            self._publish()
            w = _w()
            for route, conn, rid in waiters:
                if ok:
                    meta = w.ControlReply(rid, w.OP_RUN, w.STATUS_OK, a.name,
                                          message=text or None).to_meta()
                else:
                    meta = w.ControlReply(rid, w.OP_RUN, w.STATUS_ERROR, a.name, error={
                        "code": w.ERR_ACTION_FAILED, "message": text}).to_meta()
                route._reply(conn, meta)

    def _run_action(self, a: Action):
        self.counters["actions_run"] += 1
        try:
            r = a.fn()
        except Exception as e:
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
    def _on_control(self, route: Route, conn, meta) -> dict:
        w = _w()
        if not isinstance(meta, dict):
            raise w.MessageError(w.ERR_BAD_REQUEST, "a control's metadata is an object")
        msg = w.Control.from_frame(w.Frame(w.MsgType.CONTROL, meta))
        rid, op, name = msg.request_id, msg.op, msg.name
        self.counters["controls_in"] += 1

        def error(code, text):
            return w.ControlReply(rid, op, w.STATUS_ERROR, name,
                                  error={"code": code, "message": text}).to_meta()

        try:
            allowed = bool(route.transport.may_control(conn))
        except Exception:
            allowed = False
        if not allowed:
            return error(w.ERR_FORBIDDEN, "this connection may not control the run")
        reply = None
        timer = None
        with self.lock:
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
        w = _w()
        return w.Controls(self._run_obj(), [p.entry() for p in self.params.values()],
                          [a.entry() for a in self.actions.values()]).to_meta()

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
