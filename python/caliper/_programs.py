"""The facade of GPU programs in the script's process (``design/gpu-programs.md``).

A program is WGSL compute that reads tensors of the script in place and writes one output. The
wire path of Caliper LIB calls :func:`install`, :func:`remove` and :func:`status` when a PROGRAM
message arrives, and sends the status items back in PROGRAM_STATE. ``caliper.step()`` calls
:func:`at_step` after the framework's work of the step is complete. The facade loads the native
module ``caliper._producer`` on the first :func:`install` only, so that a script that only
streams tensors loads nothing new.

The facade keeps the cadence (``design/gpu-programs.md``, Cadence) by feedback on the run's wall
clock. At each step it puts a mark on torch's queue (:class:`caliper._adapter.StepMarks`), which
also gives the queued work to the GPU, and it reads the marks that completed without a wait. From
their completion times it measures the base step time ``b`` over the steps without a pass and the
overhead ``D`` that a pass adds to the steps around it, and it runs the programs every
``N = ceil(D / (0.04 * b))`` steps, moved by bounded factors, at least ``N_MIN`` and at most the
steps of ``CEILING_S`` seconds. When even that ceiling costs more than the share, every program
stops with ``program.over_budget`` and training goes on. It gives each geometry output to the
publish path of ``caliper.scene`` and each pixels output to ``caliper.show`` as the live view
``program.<id>``. A refused or stopped program is an error of its panel, never of the script:
no function here raises for it.

THREADS. The wire path may call :func:`install`, :func:`remove` and :func:`status` from any
thread. One lock orders them with :func:`at_step` on the training thread, thus the runtime is
never used from two threads at once. :data:`listeners` are called after the lock is released.

INPUTS. An input's source is ``live:<name>``: a parameter or buffer of the watched model by its
name in ``model.named_parameters()``, a parameter's gradient as ``<parameter>.grad``, or a
module's last output as ``<module>.act``. A gradient or an output that does not exist yet keeps
the program in the state ``waiting`` until it does. When the tensor of an input moves to new
storage (a gradient set to ``None`` and made again, each new output of a module), the facade
imports it again and compiles the program again at the next pass, because an import binds one
storage.
"""
from __future__ import annotations

import math
import sys
import threading
import time
import warnings
from collections import deque

__all__ = ["install", "remove", "status", "at_step", "loaded"]

ACTIVE = False
"""Whether a program may run; ``caliper.step()`` calls :func:`at_step` only while it is true."""

listeners: list = []
"""Callables that get a status item after each change of a program's state; the wire path's
PROGRAM_STATE uses them. An error in a listener is warned once and never stops the step."""

SHARE = 0.04            # the runtime's share of the step time (design/producer-runtime.md, Budget)
WARMUP_STEPS = 8        # steps after an install before the program's first pass
N_MIN = 3               # a cycle keeps two steps without a pass, which the measurement needs
BASE_SAMPLES = 9        # stretches without a pass in the median of the base step time b
COST_SAMPLES = 8        # passes in the median of the overhead D
CHANGE = 0.25           # N moves down only when it is more than this share above the wanted value
GROW, SHRINK = 2.0, 1.5 # the largest factors of one change of N, up and down
CEILING_S = 10.0        # N is at most the steps of this many seconds of training
OVER_UPDATES = 3        # updates over budget at the ceiling before the programs stop
EVERY_MAX = 10000
TIME_MS_DEFAULT, TIME_MS_MAX = 4.0, 50.0
MEMORY_MB_DEFAULT, MEMORY_MB_MAX = 64, 512
INPUTS_MAX = 8
_ARRAYS = ("positions", "colors", "values", "radii")

_LOCK = threading.RLock()
_PROGS: dict = {}       # id -> _Program, in the order of installation
_IMPORTS: dict = {}     # input name -> its newest _Import
_LIVE: dict = {}        # handle -> each _Import not released yet (an older one may still be bound)
_ACTS: dict = {}        # module name -> its last output, for the inputs <module>.act
_ACT_HOOKS: dict = {}   # module name -> the forward hook's handle
_STATE = {"rt": None, "mod": None, "model": None, "optimizer": None, "marks": None, "step": None}
_clock = time.perf_counter
_WARNED: set = set()


def _median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class _Cadence:
    """The cadence N by feedback on the measured overhead of a pass (``design/gpu-programs.md``).

    The facade names each step with a pass by :meth:`passed` before its mark can complete, and gives
    each completion by :meth:`completed`, in the order of the steps. A completion time is the time
    of the first look that saw the mark complete: never early, and late by at most the time between
    two looks, which is a step of the training thread, or a few microseconds while a pass waits.
    The controller then measures:

    - the base step time ``b``: over each stretch of marks without a pass, from the first mark after
      a pass to the mark before the next pass, the time per step; ``b`` is the median of the last
      ``BASE_SAMPLES`` stretches, thus it follows a step time that changes;
    - the overhead ``D`` of a pass at step ``k``: the time from the mark of ``k - 1`` to the mark of
      ``k + 1`` minus ``2 * b``. Both ends are marks of steps without a pass, thus a wait that only
      drains torch's queued work costs nothing here, and the GPU's idle time while the training
      thread waits, runs the pass and queues the next step counts. When the training thread waits
      at each step anyway, both ends are late by the same time; else only the end after the pass can
      be late, thus ``D`` can come out larger than it is, never smaller. ``D`` is the median of the
      last ``COST_SAMPLES`` passes, which ignores an outlier.

    After each new ``D`` it wants ``ceil(D / (SHARE * b))`` steps between passes, between
    ``N_MIN`` and the ceiling. N moves up as soon as it is below that value, by at most a factor
    ``GROW``, and down only when it is more than ``CHANGE`` above it, by at most a factor
    ``SHRINK``. Thus N rests between the wanted value and ``1 + CHANGE`` times it.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.done: dict = {}        # step -> completion time of its mark, for the last few marks
        self.passes: set = set()    # the recent steps with a pass
        self.start = None           # (step, time) of the first mark after the last pass
        self.base = deque(maxlen=BASE_SAMPLES)
        self.cost = deque(maxlen=COST_SAMPLES)
        self.n = None               # N, None before the first measured overhead
        self.want = None            # ceil(D / (SHARE * b)) of the last update
        self.ceiling = EVERY_MAX
        self.over = 0               # updates in a row that wanted more than the ceiling
        self.last_pass = None       # the step of the last pass

    def restart(self) -> None:
        """Forget the marks and passes of the current windows, but keep ``b``, ``D`` and N: the
        script numbered its steps again."""
        self.done = {}
        self.passes = set()
        self.start = None
        self.last_pass = None

    def step_s(self):
        """The base step time ``b`` in seconds, or None before the first stretch."""
        return _median(self.base) if self.base else None

    def cost_s(self):
        """The overhead ``D`` of a pass in seconds, or None before the first pass is measured."""
        return _median(self.cost) if self.cost else None

    def share(self):
        """The measured share ``D / (N * b)`` of the training time, or None."""
        b, d = self.step_s(), self.cost_s()
        return None if b is None or d is None or b <= 0 else d / ((self.n or N_MIN) * b)

    def due(self, step: int) -> bool:
        """Whether ``step`` may run a pass: N steps after the last one."""
        return self.last_pass is None or step - self.last_pass >= (self.n or N_MIN)

    def passed(self, step: int) -> None:
        """Step ``step`` runs a pass."""
        self.passes.add(step)
        self.last_pass = step

    def period(self, every_min: int) -> int:
        """The steps between two passes of a program: max(N, every), rounded up to a multiple of N."""
        n = self.n or N_MIN
        return min(n * math.ceil(max(n, every_min) / n), EVERY_MAX)

    def completed(self, step: int, t: float) -> None:
        """The mark of ``step`` completed at ``t``."""
        self.done[step] = t
        prev = self.done.get(step - 1)
        if step in self.passes:
            s = self.start
            if s is not None and prev is not None and step - 1 not in self.passes and step - 1 > s[0]:
                self.base.append((prev - s[1]) / (step - 1 - s[0]))
            self.start = None
        elif self.start is None or step - 1 in self.passes:
            self.start = (step, t)
        if step - 1 in self.passes and step not in self.passes and step - 2 not in self.passes:
            before, b = self.done.get(step - 2), self.step_s()
            if before is not None and b is not None:
                self.cost.append(max(0.0, t - before - 2.0 * b))
                self._update()
        for k in [k for k in self.done if k < step - 2]:
            del self.done[k]
        self.passes = {k for k in self.passes if k >= step - 2}

    def _update(self) -> None:
        b, d = self.step_s(), self.cost_s()
        if b is None or d is None:
            return
        self.ceiling = max(N_MIN, min(EVERY_MAX, math.ceil(CEILING_S / b))) if b > 0 else N_MIN
        self.want = math.ceil(d / (SHARE * b)) if b > 0 else EVERY_MAX
        n = self.n or N_MIN
        want = min(max(self.want, N_MIN), self.ceiling)
        if want > n:
            n = min(want, math.ceil(n * GROW))
        elif want * (1 + CHANGE) < n:
            n = max(want, math.floor(n / SHRINK))
        self.n = n
        self.over = self.over + 1 if self.want > self.ceiling and n >= self.ceiling else 0

    def over_budget(self, every_min: int) -> bool:
        """Whether the programs cannot run inside the share on this run: the wanted N stayed above
        the ceiling for ``OVER_UPDATES`` updates, and the declared ``every_min`` does not keep the
        passes rare enough either."""
        return (self.over >= OVER_UPDATES and self.want is not None
                and self.want > max(self.ceiling, every_min))


_CAD = _Cadence()


class _Import:
    """One tensor of the script imported in place: its handle and the storage it binds."""
    __slots__ = ("name", "handle", "key")

    def __init__(self, name, handle, key):
        self.name, self.handle, self.key = name, handle, key


class _Program:
    __slots__ = ("id", "decl", "wgsl", "sha256", "scene", "kind", "inputs", "output", "every_min",
                 "time_ms", "memory_mb", "handle", "state", "error", "runs", "last_step",
                 "last_pass_ms", "memory_bytes", "warm", "every", "t0", "bound")

    def __init__(self, pid, decl, wgsl, scene, sha256):
        self.id = pid
        self.decl = decl
        self.wgsl = wgsl
        self.sha256 = sha256
        self.scene = scene
        out = decl.get("output") or {}
        self.output = out
        self.kind = out.get("kind", "geometry")
        self.inputs = [(str(i.get("var", "")), str(i.get("source", ""))) for i in decl.get("inputs") or []]
        lim = decl.get("limits") or {}
        self.every_min = int(lim.get("every") or 1)
        self.time_ms = float(lim.get("time_ms") or TIME_MS_DEFAULT)
        self.memory_mb = int(lim.get("memory_mb") or MEMORY_MB_DEFAULT)
        self.handle = None
        self.state = "waiting"
        self.error = None
        self.runs = 0
        self.last_step = None
        self.last_pass_ms = None
        self.memory_bytes = 0
        self.warm = WARMUP_STEPS
        self.every = None
        self.t0 = _clock()
        self.bound = {}         # var -> the _Import it was compiled with

    def item(self) -> dict:
        b, d, share = _CAD.step_s(), _CAD.cost_s(), _CAD.share()
        return {
            "id": self.id, "kind": self.kind, "state": self.state, "every": self.every,
            "every_min": self.every_min, "last_pass_ms": self.last_pass_ms,
            "last_cost_ms": None if d is None else d * 1e3,
            "step_ms": None if b is None else b * 1e3,
            "overhead_share": None if share is None else share,
            "last_step": self.last_step, "runs": self.runs, "memory_bytes": self.memory_bytes,
            "memory_limit_bytes": self.memory_mb << 20, "error": self.error,
            "sha256": self.sha256,
        }


class _Refused(Exception):
    def __init__(self, code, message, line=None, column=None):
        super().__init__(message)
        self.error = {"code": code, "message": message, "line": line, "column": column}


class _Waiting(Exception):
    """An input that exists but has no tensor yet: a gradient before the first backward pass, or
    a module's output before its first forward pass."""


# ---- the watched model --------------------------------------------------------------------------
def attach(model, optimizer, handle=None) -> None:
    """Give the facade the model and the optimizer of ``caliper.watch(model, optimizer)``.

    Loads no native module. With ``handle`` (a WatchHandle), its ``remove()`` detaches again.
    """
    with _LOCK:
        _STATE["model"] = model
        _STATE["optimizer"] = optimizer
    if handle is not None and hasattr(handle, "_handles"):
        handle._handles.append(_Detach(model))


class _Detach:
    """A hook handle whose ``remove()`` detaches the facade from its model."""

    def __init__(self, model):
        self._model = model

    def remove(self) -> None:
        detach(self._model)


def detach(model=None) -> None:
    """Forget the watched model (only ``model``, when given), and remove every program."""
    with _LOCK:
        if model is not None and _STATE["model"] is not model:
            return
        ids = list(_PROGS)
    for pid in ids:
        remove(pid)
    with _LOCK:
        _drop_act_hooks()
        _STATE["model"] = None
        _STATE["optimizer"] = None


def _drop_act_hooks() -> None:
    for h in _ACT_HOOKS.values():
        h.remove()
    _ACT_HOOKS.clear()
    _ACTS.clear()


def _torch_adapter():
    from ._adapter import TORCH
    return TORCH


def _resolve(name: str):
    """The tensor of an input name, or None for no such name; raises _Waiting for a gradient or an
    output that does not exist yet."""
    model = _STATE["model"]
    if model is None:
        return None
    if name.endswith(".grad"):
        p = dict(model.named_parameters()).get(name[:-5])
        if p is None:
            return None
        if p.grad is None:
            raise _Waiting(name)
        return p.grad
    if name.endswith(".act"):
        mname = name[:-4]
        mods = dict(model.named_modules())
        if mname not in mods:
            return None
        if mname not in _ACT_HOOKS:
            _ACT_HOOKS[mname] = mods[mname].register_forward_hook(
                lambda _m, _i, out, mname=mname: _keep_act(mname, out))
        t = _ACTS.get(mname)
        if t is None:
            raise _Waiting(name)
        return t
    t = dict(model.named_parameters()).get(name)
    if t is None:
        t = dict(model.named_buffers()).get(name)
    return t


def _keep_act(mname, out) -> None:
    if isinstance(out, (tuple, list)):
        out = next((x for x in out if hasattr(x, "detach")), None)
    if out is not None:
        _ACTS[mname] = out.detach()


def _storage_key(t):
    return (t.untyped_storage().data_ptr(), t.storage_offset(), t.numel(), t.dtype, str(t.device))


# ---- the native runtime -------------------------------------------------------------------------
def _runtime():
    if _STATE["rt"] is None:
        if _STATE["mod"] is None:
            try:
                from . import _producer
            except ImportError as e:
                raise _Refused("program.unavailable",
                               f"The producer runtime caliper._producer cannot load here: {e}") from e
            _STATE["mod"] = _producer
        try:
            _STATE["rt"] = _STATE["mod"].Runtime()
        except RuntimeError as e:
            raise _refusal(e) from e
    return _STATE["rt"]


def _refusal(e) -> _Refused:
    args = getattr(e, "args", ())
    if len(args) >= 4 and isinstance(args[0], str):
        return _Refused(args[0], args[1], args[2] or None, args[3] or None)
    return _Refused("program.unavailable", str(e))


def _import(name: str, t) -> _Import:
    """The import of tensor ``t`` under ``name``, made again when its storage changed."""
    key = _storage_key(t)
    im = _IMPORTS.get(name)
    if im is not None and im.key == key:
        return im
    if str(t.device).split(":")[0] != "mps":
        raise _Refused("program.import_refused",
                       f"The tensor {name} is on {t.device}; the producer runtime reads MPS tensors in "
                       "place only.")
    rt = _runtime()
    try:
        cap = _torch_adapter().export(t)
    except ValueError as e:
        raise _Refused("program.import_refused", f"The tensor {name}: {e}.") from e
    except RuntimeError as e:
        raise _Refused("program.unavailable", f"The tensor {name}: {e}.") from e
    try:
        h = rt.bind(name, cap)
    except RuntimeError as e:
        raise _refusal(e) from e
    im = _IMPORTS[name] = _LIVE[h] = _Import(name, h, key)
    return im


def _release_unused() -> None:
    """Release each import that no program binds. An older import of a name stays while a
    program still binds it, thus programs that share an input compile again one by one."""
    used = {im.handle for p in _PROGS.values() for im in p.bound.values()}
    rt = _STATE["rt"]
    for h in [h for h in _LIVE if h not in used]:
        im = _LIVE.pop(h)
        if _IMPORTS.get(im.name) is im:
            del _IMPORTS[im.name]
        if rt is not None:
            rt.release(h)


def _check(p: _Program) -> None:
    """The facade's part of the declaration: what the native call needs to take its arguments."""
    out = p.output
    if p.kind not in ("geometry", "pixels"):
        raise _Refused("program.output_mismatch", "The output is of kind geometry or pixels.")
    if len(p.inputs) > INPUTS_MAX:
        raise _Refused("program.over_limits", f"A program has at most {INPUTS_MAX} inputs.")
    if not (1 <= p.every_min <= EVERY_MAX):
        raise _Refused("program.over_limits", f"limits.every is between 1 and {EVERY_MAX}.")
    if not (0 < p.time_ms <= TIME_MS_MAX):
        raise _Refused("program.over_limits", f"limits.time_ms is above 0 and at most {TIME_MS_MAX:g}.")
    if not (1 <= p.memory_mb <= MEMORY_MB_MAX):
        raise _Refused("program.over_limits", f"limits.memory_mb is between 1 and {MEMORY_MB_MAX}.")
    if p.kind == "geometry":
        if not p.scene:
            raise _Refused("program.bad_target", "A geometry output needs the scene of its panel.")
        arrays = list(out.get("arrays") or [])
        if "positions" not in arrays or any(a not in _ARRAYS for a in arrays):
            raise _Refused("program.output_mismatch",
                           "A geometry output declares positions and any of colors, values and radii.")
    for var, src in p.inputs:
        if not src.startswith("live:"):
            raise _Refused("program.unknown_input", f"The input {var} has the source {src!r}, "
                           "which is not live:<name>.")


def _compile(p: _Program) -> None:
    """Import the inputs and compile the program; raises _Refused or _Waiting."""
    bound = {}
    for var, src in p.inputs:
        name = src[len("live:"):]
        t = _resolve(name)
        if t is None:
            raise _Refused("program.unknown_input", f"The script has no tensor named {name}.")
        bound[var] = _import(name, t)
    rt = _runtime()
    out = p.output
    wg = p.decl.get("workgroups") or (0, 0, 0)
    old = p.handle
    try:
        h = rt.program(p.id, p.wgsl, [(v, im.handle) for v, im in bound.items()], p.kind,
                       entry=p.decl.get("entry"), part=str(out.get("part", "points")),
                       max_count=int(out.get("max_count") or 0), arrays=list(out.get("arrays") or []),
                       width=int(out.get("width") or 0), height=int(out.get("height") or 0),
                       workgroups=tuple(int(x) for x in wg), time_ms=p.time_ms,
                       memory_bytes=p.memory_mb << 20)
    except RuntimeError as e:
        raise _refusal(e) from e
    if old is not None:
        rt.remove(old)
    p.handle = h
    p.bound = bound
    st = rt.status(h)
    p.memory_bytes = int(st.get("memory_bytes") or 0)
    if p.state != "running":
        p.state = "compiled"
    p.error = None


def _set_active() -> None:
    global ACTIVE
    ACTIVE = any(p.state in ("waiting", "compiled", "running") for p in _PROGS.values())


def _notify(items) -> None:
    for item in items:
        for cb in list(listeners):
            try:
                cb(item)
            except Exception as e:  # a listener never stops the step
                _warn_once(("listener", id(cb)), f"caliper: a program state listener failed: {e!r}")


def _warn_once(key, text) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(text, RuntimeWarning, stacklevel=3)


# ---- the facade ---------------------------------------------------------------------------------
def install(program_id: str, declaration: dict, wgsl: str, *, scene: str | None = None,
            sha256: str = "") -> dict:
    """Install a program, or replace the program with the same id.

    Args:
        program_id: The program's id.
        declaration: The program declaration of the PROGRAM message, without ``wgsl``
            (``inputs``, ``output``, ``limits``, ``entry``, ``workgroups``).
        wgsl: The WGSL source.
        scene: The scene name of a geometry output's panel, or ``None`` for a pixels output.
        sha256: The SHA-256 of ``wgsl`` that the app sent, in hex.

    Returns:
        dict: The program's status item (the keys of ``program_status`` without ``host``), with
        the state ``compiled`` or ``refused`` and, for a refusal, the ``error``. The state is
        ``waiting`` while an input is a gradient or a module output that does not exist yet.
    """
    _remove(program_id)
    with _LOCK:
        if not ACTIVE:
            _CAD.reset()                             # the measurement starts again from this install
            if _STATE["marks"]:
                _STATE["marks"].clear()
        p = _Program(str(program_id), dict(declaration or {}), str(wgsl), scene, str(sha256 or ""))
        _PROGS[p.id] = p
        try:
            _check(p)
            _compile(p)
        except _Waiting:
            p.state = "waiting"
        except _Refused as r:
            p.state = "refused"
            p.error = r.error
        _set_active()
        item = p.item()
    _notify([item])
    return item


def _remove(program_id: str):
    with _LOCK:
        p = _PROGS.pop(str(program_id), None)
        if p is None:
            return None
        rt = _STATE["rt"]
        if rt is not None and p.handle is not None:
            rt.remove(p.handle)
        p.handle = None
        p.bound = {}
        _release_unused()
        _set_active()
        if not _PROGS:
            _drop_act_hooks()
    return p


def remove(program_id: str) -> None:
    """Stop and remove a program, and clear its output; an unknown id does nothing."""
    p = _remove(program_id)
    if p is None or p.runs == 0:
        return
    try:
        pkg = sys.modules[__package__]
        if p.kind == "pixels":
            pkg.remove(f"program.{p.id}")
        elif p.scene:
            pkg.scene(p.scene).clear(_entity(p))
    except Exception as e:  # clearing an output never stops the script
        _warn_once(("clear", p.id), f"caliper: clearing the output of program {p.id} failed: {e!r}")


def status() -> list:
    """Return one status item for each installed program, in the order of installation."""
    with _LOCK:
        return [p.item() for p in _PROGS.values()]


def loaded() -> bool:
    """Whether the native module ``caliper._producer`` is loaded in this process."""
    return "caliper._producer" in sys.modules


def stats() -> dict:
    """The counters of the runtime's routes (imports in place, bytes copied, outputs given), and
    the cadence; empty before the first program."""
    with _LOCK:
        rt = _STATE["rt"]
        out = dict(rt.stats()) if rt is not None else {}
        b, d = _CAD.step_s(), _CAD.cost_s()
        out.update({"every": _CAD.n, "step_ms": None if b is None else b * 1e3,
                    "cost_ms": None if d is None else d * 1e3, "overhead_share": _CAD.share(),
                    "wanted_every": _CAD.want, "ceiling": _CAD.ceiling})
        return out


def _stop_all(code: str, message: str) -> list:
    """A timeout or a lost device: every program stops, the runtime closes, training continues."""
    changed = []
    for p in _PROGS.values():
        if p.state in ("waiting", "compiled", "running"):
            p.state = "stopped"
            p.error = {"code": code, "message": message, "line": None, "column": None}
            changed.append(p)
        p.handle = None
        p.bound = {}
    _IMPORTS.clear()
    _LIVE.clear()
    rt = _STATE["rt"]
    _STATE["rt"] = None
    if rt is not None:
        rt.close()
    return changed


_STOP_TEXT = {
    "program.over_time": "The program went over limits.time_ms on 3 passes in a row.",
    "program.import_refused": "An input of the program was released.",
    "program.layer_refused": "The graphics layer refused the pass.",
}


def at_step(step: int) -> None:
    """Run the programs that are due at ``step`` and publish their outputs.

    ``caliper.step()`` calls it on the training thread after the framework's work of the step is
    queued, before ``optimizer.step()``. Without an installed program it returns at once and touches
    no device. With one, it puts the step's mark on torch's queue and reads the marks that completed,
    without a wait; only a step with a pass waits.
    """
    if not ACTIVE:
        return
    try:
        _at_step(int(step))
    except Exception as e:  # a program never stops the training loop
        _warn_once(("step",), f"caliper: the GPU programs failed at step {step}: {e!r}")


def _marks():
    """The step marks of torch's queue, made at the first step that needs them; None off MPS."""
    m = _STATE["marks"]
    if m is None:
        m = _STATE["marks"] = _torch_adapter().step_marks() or False
    return m or None


def _at_step(step: int) -> None:
    changed = []
    published = []
    with _LOCK:
        now = _clock()
        marks = _marks()
        last = _STATE["step"]
        _STATE["step"] = step
        if last is not None and step <= last:        # the script numbered its steps again
            _CAD.restart()
            if marks is not None:
                marks.clear()
        if marks is not None:
            marks.record(step, now)                  # also gives torch's queued work to the GPU
        due = []
        for p in _PROGS.values():
            if p.state not in ("waiting", "compiled", "running"):
                continue
            if p.warm > 0:
                p.warm -= 1
                continue
            if p.last_step is None or step < p.last_step or step - p.last_step >= p.every_min:
                due.append(p)
        if due and not _CAD.due(step):
            due = []
        if due:
            _CAD.passed(step)
        done = marks.poll(now) if marks is not None else [(step, now, now)]
        if due:
            if marks is not None:                    # sync point 1: torch's work is complete
                done += marks.wait(step, _clock)
            else:
                _torch_adapter().framework_done()
            ran = []
            try:
                for p in due:
                    before = p.state
                    try:
                        _refresh(p)
                    except _Waiting:
                        continue
                    except _Refused as r:
                        p.state, p.error = "refused", r.error
                        changed.append(p)
                        continue
                    if before != p.state:
                        changed.append(p)
                    every = _CAD.period(p.every_min)
                    res = _STATE["rt"].run([p.handle], int(step), _clock() - p.t0, every)
                    if res.get("ran"):
                        p.runs += 1                  # a compile again starts a new native count
                        ran.append(p)
            except RuntimeError as e:                # a timeout or a lost device
                r = _refusal(e)
                changed.extend(_stop_all(r.error["code"], r.error["message"]))
                ran = []
            rt = _STATE["rt"]
            for p in ran:
                st = rt.status(p.handle)
                p.last_step = int(step)
                p.last_pass_ms = st.get("last_pass_ms")
                p.every = _CAD.period(p.every_min)
                if st.get("state") == "stopped":
                    code = st.get("stop_code") or "program.over_time"
                    p.state = "stopped"
                    p.error = {"code": code, "message": _STOP_TEXT.get(code, code), "line": None,
                               "column": None}
                    changed.append(p)
                else:
                    if p.state != "running":
                        p.state = "running"
                        changed.append(p)
                    published.append((p, _outputs(rt, p)))
        for k, _lo, hi in done:
            _CAD.completed(k, hi)
        live = [p.every_min for p in _PROGS.values() if p.state in ("waiting", "compiled", "running")]
        if live and _CAD.over_budget(min(live)):
            changed.extend(_stop_over_budget())
            published = []
        _set_active()
        items = [p.item() for p in dict.fromkeys(changed)]
    for p, outs in published:
        _publish(p, outs)
    _notify(items)


def _stop_over_budget() -> list:
    """Every program stops with ``program.over_budget``: a pass even at the ceiling of N costs more
    than the share of the training time. The runtime stays open, and training goes on."""
    b, d = _CAD.step_s(), _CAD.cost_s()
    message = (f"A pass adds {d * 1e3:.2f} ms to a step of {b * 1e3:.3f} ms, thus the programs need "
               f"at least {_CAD.want} steps between passes to stay under {SHARE:.0%} of the training "
               f"time, more than the ceiling of {_CAD.ceiling} steps ({_CAD.ceiling * b:.3g} s) on "
               "this run. The programs stopped, and training goes on.")
    changed = []
    rt = _STATE["rt"]
    for p in _PROGS.values():
        if p.state in ("waiting", "compiled", "running"):
            p.state = "stopped"
            p.error = {"code": "program.over_budget", "message": message, "line": None, "column": None}
            if rt is not None and p.handle is not None:
                rt.remove(p.handle)
            p.handle = None
            p.bound = {}
            changed.append(p)
    _release_unused()
    return changed


def _refresh(p: _Program) -> None:
    """Compile a waiting program, or compile it again when an input's storage changed."""
    if p.handle is None:
        _compile(p)
        return
    for var, src in p.inputs:
        im = p.bound.get(var)
        if im is None:
            continue
        t = _resolve(src[len("live:"):])
        if t is None:
            raise _Refused("program.unknown_input", f"The script no longer has the tensor {im.name}.")
        if _storage_key(t) != im.key:
            _compile(p)
            _release_unused()
            return


def _outputs(rt, p: _Program) -> dict:
    """The outputs of the last pass as tensors over the runtime's buffers, without a copy."""
    import torch
    names = ["pixels"] if p.kind == "pixels" else list(p.output.get("arrays") or [])
    out = {}
    for n in names:
        cap, count = rt.output(p.handle, n)
        out[n] = torch.from_dlpack(cap)
        out["count"] = count
    return out


def _entity(p: _Program) -> str:
    return p.output.get("entity") or f"programs/{p.id}"


def _publish(p: _Program, outs: dict) -> None:
    """Give the outputs to the publish path of caliper.scene or caliper.show."""
    pkg = sys.modules[__package__]
    try:
        if p.kind == "pixels":
            pkg.show(outs["pixels"], f"program.{p.id}")
            return
        sc = pkg.scene(p.scene)
        count = int(outs.get("count", 0))
        part = p.output.get("part", "points")
        if part == "lines":
            count -= count % 2
        pos = outs["positions"][:count]
        if sc.space == "2d":
            pos = pos[:, :2]
        style = p.output.get("style") or {}
        kw = {"cmap": style.get("colormap") or "viridis", "vmin": style.get("vmin"),
              "vmax": style.get("vmax")}
        if "colors" in outs:
            kw["colors"] = outs["colors"][:count]
        elif "values" in outs:
            kw["values"] = outs["values"][:count]
        if part == "lines":
            if "colors" in kw or "values" in kw:
                kw["color_per"] = "vertex"
            sc.lines(_entity(p), pos, **kw)
        else:
            if "radii" in outs:
                kw["radii"] = outs["radii"][:count]
            sc.points(_entity(p), pos, **kw)
    except Exception as e:  # the publish path never stops the step
        _warn_once(("publish", p.id), f"caliper: publishing the output of program {p.id} failed: {e!r}")
