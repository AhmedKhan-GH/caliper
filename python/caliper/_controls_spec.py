"""Control constants, validation helpers, Param, Action, and Route types."""
from __future__ import annotations

import json
import logging
import math
import numbers
import warnings

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
_LIFECYCLE_HEADROOM = 1280
_NO = object()
_FIRST, _LAST = 0, -1

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


def _name_ok(v) -> bool:
    return isinstance(v, str) and 0 < len(v.encode("utf-8")) <= NAME_MAX


def _check_name(v, what: str) -> str:
    if not _name_ok(v):
        raise ValueError(f"invalid {what} name {v!r} (a str of 1 to {NAME_MAX} UTF-8 bytes)")
    return v


def _cut(text: str, limit: int) -> str:
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


def intervention_key(spec):
    """Keep legacy keys and distinguish each positioned feature record."""
    key = (spec["point"], spec.get("op", spec.get("kind")))
    if key[1] in ("clamp", "scale") and "positions" in spec:
        return (*key, spec["feature"], tuple(sorted(spec["positions"])))
    return key


def _json_bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _check_controls_budget(meta, *, ivs=None) -> None:
    """Check direct UTF-8 JSON size against wire metadata limits."""
    from ._intervention_protocol import MAX_BYTES

    entries = meta.get("interventions", {}).get("active", [])
    live_count = 0
    for entry in entries:
        extra = _LIFECYCLE_HEADROOM if entry.get("state") != IV_ENDED else 0
        if extra:
            live_count += 1
        if _json_bytes(entry) + extra > MAX_BYTES:
            raise ValueError("intervention metadata exceeds the wire limit")
    reserve = 128 + live_count * _LIFECYCLE_HEADROOM
    if _json_bytes(meta) + reserve > MAX_META_BYTES:
        raise ValueError("CONTROLS metadata exceeds the wire limit")


def _scalar(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    item = getattr(v, "item", None)
    if item is not None:
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return v
    return v


def _step_ok(v) -> bool:
    return v is None or (isinstance(v, numbers.Integral) and not isinstance(v, bool)
                         and 0 <= v < _INT_EXACT)


# ===========================================================================
# parameters
# ===========================================================================
def _run_value(p) -> None:
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
        self._version = 0

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
    _check_name(name, "parameter")
    help = _check_help(help)
    value, lo, hi = _scalar(value), _scalar(lo), _scalar(hi)
    if choices is not None:
        choices = list(choices)
        if (not 1 <= len(choices) <= MAX_CHOICES or not all(_name_ok(c) for c in choices)
                or len(set(choices)) != len(choices)):
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
        self.waiters: list = []
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


class Route:
    """A transport attached to the controls."""

    def __init__(self, ctl, transport):
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
