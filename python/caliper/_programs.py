"""The facade of GPU programs in the script's process (``design/gpu-programs.md``).

A program is WGSL compute that reads tensors of the script in place and writes one output. The
wire path of Caliper LIB calls :func:`install`, :func:`remove` and :func:`status` when a PROGRAM
message arrives, and sends the status items back in PROGRAM_STATE. ``caliper.step()`` calls
:func:`at_step` after the framework's work of the step is complete. The facade loads the native
module ``caliper._producer`` on the first :func:`install` only, so that a script that only
streams tensors loads nothing new.

The facade keeps the cadence (``design/gpu-programs.md``, Cadence): it measures the step time
``T`` and the added time ``C`` of a pass step, and runs the programs every
``N = max(ceil(C / (0.04 * T)), every, 1)`` steps. It gives each geometry output to the publish
path of ``caliper.scene`` and each pixels output to ``caliper.show`` as the live view
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
WARMUP_STEPS = 8        # steps that measure T after an install, before the first pass
T_WINDOW = 32           # steps without a pass in the median of T
C_WINDOW = 8            # pass steps in the median of C
CHANGE = 0.25           # N changes only when the new value differs by more than this share
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
_STATE = {"rt": None, "mod": None, "model": None, "optimizer": None, "t0": time.perf_counter()}
_WARNED: set = set()


def _median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class _Cadence:
    """The step time ``T``, the added time ``C`` of a pass step and the rate ``N``.

    Torch's MPS queue lets the training thread run ahead of the GPU, thus the time between two
    ``caliper.step()`` calls is often only the time to queue a step, and the first sync point then
    waits for every queued step. The facade therefore measures ``T`` as the time from the end of
    one pass, when torch's queue is empty, to the end of the next first sync point, when it is
    empty again, divided by the steps in between. ``C`` is the pass (sync points 2 and 3) plus the
    idle time of the GPU after the first sync point, which is at most the wait of that sync point
    and at most the median time to queue a step.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.cpu = deque(maxlen=T_WINDOW)   # times between two steps without a pass
        self.t = deque(maxlen=C_WINDOW)     # T samples: step times between two empty queues
        self.c = deque(maxlen=C_WINDOW)     # C samples: added times of pass steps
        self.last = None        # perf_counter() of the previous at_step
        self.passed = False     # whether the previous at_step ran a pass
        self.n = None           # N, None before the first pass
        self.last_cost = None
        self.mark = None        # perf_counter() when torch's queue was last empty
        self.since = 0          # steps since then

    def tick(self, now: float) -> None:
        """Count a step, and record its time unless a pass ran in it."""
        if self.last is not None and not self.passed:
            self.cpu.append(now - self.last)
        self.last = now
        self.passed = False
        self.since += 1

    def drained(self, now: float) -> None:
        """Torch's queue is empty now: the start of the next T sample."""
        self.mark = now
        self.since = 0

    def step_s(self):
        return _median(self.t) if self.t else None

    def update(self, t0: float, t1: float, t2: float) -> None:
        """Record a pass step (sync point 1 from ``t0`` to ``t1``, the pass from ``t1`` to
        ``t2``) and compute N again."""
        self.passed = True
        if self.mark is not None and self.since > 0:
            self.t.append((t1 - self.mark) / self.since)
        wait = t1 - t0
        cost = (t2 - t1) + (min(wait, _median(self.cpu)) if self.cpu else wait)
        self.c.append(cost)
        self.last_cost = cost
        self.drained(t2)
        T = self.step_s()
        budget = math.ceil(_median(self.c) / (SHARE * T)) if T and T > 0 else 1
        budget = min(max(budget, 1), EVERY_MAX)
        if self.n is None or abs(budget - self.n) > CHANGE * self.n:
            self.n = budget

    def period(self, every_min: int) -> int:
        """The steps between two passes of a program: max(N, every), rounded up to a multiple of N."""
        n = self.n or 1
        return min(n * math.ceil(max(n, every_min) / n), EVERY_MAX)


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
        self.t0 = time.perf_counter()
        self.bound = {}         # var -> the _Import it was compiled with

    def item(self) -> dict:
        T = _CAD.step_s()
        return {
            "id": self.id, "kind": self.kind, "state": self.state, "every": self.every,
            "every_min": self.every_min, "last_pass_ms": self.last_pass_ms,
            "last_cost_ms": None if _CAD.last_cost is None else _CAD.last_cost * 1e3,
            "step_ms": None if T is None else T * 1e3, "last_step": self.last_step,
            "runs": self.runs, "memory_bytes": self.memory_bytes,
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
            _CAD.reset()                             # T starts again from this install
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
        T = _CAD.step_s()
        out.update({"every": _CAD.n, "step_ms": None if T is None else T * 1e3,
                    "cost_ms": None if _CAD.last_cost is None else _CAD.last_cost * 1e3})
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
    complete. Without an installed program it returns at once and touches no device.
    """
    if not ACTIVE:
        return
    try:
        _at_step(int(step))
    except Exception as e:  # a program never stops the training loop
        _warn_once(("step",), f"caliper: the GPU programs failed at step {step}: {e!r}")


def _at_step(step: int) -> None:
    changed = []
    published = []
    with _LOCK:
        _CAD.tick(time.perf_counter())
        if _CAD.mark is None:                        # one sync after an install: T starts here
            _torch_adapter().framework_done()
            _CAD.drained(time.perf_counter())
        due = []
        for p in _PROGS.values():
            if p.state not in ("waiting", "compiled", "running"):
                continue
            if p.warm > 0:
                p.warm -= 1
                continue
            if p.last_step is None or step - p.last_step >= _CAD.period(p.every_min):
                due.append(p)
        if not due:
            return
        t0 = time.perf_counter()
        _torch_adapter().framework_done()            # sync point 1: torch's work is complete
        t1 = time.perf_counter()
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
                res = _STATE["rt"].run([p.handle], int(step), time.perf_counter() - p.t0, every)
                if res.get("ran"):
                    p.runs += 1                      # a compile again starts a new native count
                    ran.append(p)
        except RuntimeError as e:                    # a timeout or a lost device
            r = _refusal(e)
            changed.extend(_stop_all(r.error["code"], r.error["message"]))
            ran = []
        _CAD.update(t0, t1, time.perf_counter())     # sync point 1, then the pass to sync point 3
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
        _set_active()
        items = [p.item() for p in dict.fromkeys(changed)]
    for p, outs in published:
        _publish(p, outs)
    _notify(items)


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
