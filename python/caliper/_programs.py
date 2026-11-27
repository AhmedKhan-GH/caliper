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
their completion times it measures the base step time ``b`` over the second half of each cycle
between two passes and the overhead ``D`` that a pass adds to the first half, and it runs the
programs every
``N = ceil(D / (0.04 * b))`` steps, moved by bounded factors, at least ``N_MIN`` and at most the
steps of ``CEILING_S`` seconds. In training runs, when even that ceiling costs more than the share,
programs other than built-in library pieces carry a recoverable ``program.over_budget`` warning
while they keep running; built-in library pieces get no hard stop or budget warning, and
interpretability runs skip the training-overhead cadence and time only program work. A real hang
still stops every program with ``program.timeout``. It gives each geometry output to the
publish path of ``caliper.scene`` and each pixels output to ``caliper.show`` as the live view
``program.<id>``. A tensor output, the f32 array ``out`` that the program writes, goes to
``caliper.show`` as the same live view, in the declared shape, and :func:`set_params` gives new
values to the uniform ``params`` of a program, which its next pass reads without a compile. A
program of a built-in operator reports the kind ``operator``. A refused or stopped program is an
error of its panel. Installation raises only when output admission has no free slot.

PIECES. :func:`install_piece` installs an instance of a piece (``design/programmable-graphics.md``):
its derive half runs several compute passes in their order in one submit at a pass of the same
cadence, under the same limits, and writes named, typed resources that keep their contents between
runs. After each run the facade copies each published resource that a pass of the run writes (and
each static one after the first run) and sends it as one RESOURCE message to each viewer that
announced ``resources``, through the controls port of either sender. It also relays the draw half,
PROGRAM with the sections ``common``, ``derive`` and ``draw``, to each such viewer after the
install and when the viewer attaches, before its first resource. A piece without a derive half is only relayed.

VIEWS. The same cadence gates the snapshots of the network route (caliper._net_ring) while a
viewer subscribes, so that everything viewers ask of the run (snapshots, series chunks, scene
colors, operators and programs) keeps to one budget. The checkpoint before a step decides whether
that step is a pass: N steps after the last one and something due, a program by its ``every`` or
a view by its own rate. The view calls of a pass step take their snapshots, and its checkpoint
closes them with one event and runs the programs with one wait. The overhead ``D`` of a pass is
at least the training thread's own time in its view calls plus the sender thread's time until the
next pass, and while views are attached the passes keep to ``VIEWS_SHARE`` instead of ``SHARE``,
because the viewer's own work on the snapshots costs the run about as much again. While the views' own work in a pass is
cheap enough that a pass every ``FLOOR_S`` stays inside the share by itself, they get such a pass
of their own when the cadence is slower (the floor), which the cadence does not count.
:func:`views_stats` gives the cadence of the views for their status.

THREADS. The wire path may call :func:`install`, :func:`remove` and :func:`status` from any
thread. One lock orders them with :func:`at_step` on the training thread, thus the runtime is
never used from two threads at once. :data:`listeners` are called after the lock is released.
:func:`install` compiles the program for the device before it takes the lock
(``caliper._producer.prepare``, which uses no runtime and lets the GIL go), thus the compile under
the lock takes the checks and the binder only, and the training thread never waits for the Metal
compiler.

INPUTS. An input's source is ``live:<name>``. It can name a tensor from ``caliper.interp.publish``.
It can also name a parameter or buffer of the watched model, a parameter's gradient as ``<parameter>.grad``, or a
module's last output as ``<module>.act``. A gradient or an output that does not exist yet keeps
the program in the state ``waiting`` until it does. When the tensor of an input moves to new
storage (a gradient set to ``None`` and made again, each new output of a module), the facade
imports it again at the next pass, because an import binds one storage. With the same element
count and type it binds the new import in the program (``Runtime.rebind``) without a compile;
otherwise it compiles the program again.
"""
from __future__ import annotations

from collections import deque
import math
import sys
import threading
import time
import warnings

from . import _controls
from ._programs_cadence import (
    _Cadence,
    _clear_over_budget_warnings,
    _every_ok,
    _floor_due,
    _last_single_step,
    _mark,
    _marks,
    _median,
    _update_over_budget,
    _views_status,
)
from ._programs_outputs import (
    _check_output_capacity,
    _drain_outputs,
    _entity,
    _output_bytes,
    _outputs,
    _publish,
    _publish_output,
    _publish_pass,
    _remove_output,
    _retire_output,
)
from ._programs_pieces import (
    _VERSIONS,
    _all_busy,
    _backlog,
    _check_piece,
    _chunks,
    _compile_piece,
    _controls_ports,
    _drop_outbox,
    _offer,
    _post,
    _pump,
    _relay,
    _relay_frame,
    _relay_removal,
    _relay_to,
    _resource_frames,
    _resource_links,
    _send_resources,
    _size,
    install_piece,
)
from ._programs_step import (
    _at_step,
    _inference_work,
    _is_builtin_library_piece,
    _is_interpretability_run,
    _rebind,
    _refresh,
    at_step,
    inference_blocked,
    inference_pending,
)
from ._programs_types import (
    PIECE_COUNT_MAX,
    PIECE_FIELDS_MAX,
    PIECE_PASSES_MAX,
    PIECE_RESOURCES_MAX,
    PIECE_SIDE_MAX,
    RESOURCE_TYPE,
    TEXTURE_FORMATS,
    WORKGROUPS_MAX,
    _ARRAYS,
    _DTYPES,
    _STOP_TEXT,
    _TENSOR_DTYPES,
    _Detach,
    _Import,
    _LinkSet,
    _OutputRemoval,
    _Program,
    _Refused,
    _Waiting,
    _tensor_dtype_ok,
)

__all__ = ["install", "install_piece", "remove", "set_params", "status", "at_step", "loaded"]

ACTIVE = False
"""Whether a program may run; ``caliper.step()`` calls :func:`at_step` only while it is true."""

listeners: list = []
"""Callables that get a status item after each change of a program's state; the wire path's
PROGRAM_STATE uses them. An error in a listener is warned once and never stops the step."""

SHARE = 0.04            # the runtime's share of the step time (design/producer-runtime.md, Budget)
VIEWS_SHARE = 0.02      # the share while views are attached: the sender's and the viewer's own work
                        # on the snapshots costs the run about as much again as the measured passes
WARMUP_STEPS = 8        # steps after an install before the program's first pass
N_MIN = 3               # a cycle keeps two steps without a pass, which the measurement needs
BASE_SAMPLES = 9        # stretches without a pass in the median of the base step time b
COST_SAMPLES = 8        # cycles in the trimmed mean of the overhead D
CHANGE = 0.25           # N moves down only when it is more than this share above the wanted value
GROW, SHRINK = 2.0, 1.5 # the largest factors of one change of N, up and down
CEILING_S = 10.0        # N is at most the steps of this many seconds of training
OVER_UPDATES = 3        # updates over budget at the ceiling before the warning fires
DRAINED_STEPS = 16      # steps in a row with torch's queue empty before the marks become sparse
SPARSE = 8              # the steps between two marks while the loop waits for the GPU by itself
SHORT_WAITS = 2         # passes in a row whose first sync point found torch's queue empty
WAIT_EMPTY = 2e-4       # a first sync point shorter than this found torch's queue empty
RATE_PASSES = 8         # the last passes of a program over which its status gives passes a second
FLOOR_S = 0.5           # the views take a pass at least this often while their own work is cheap
VIEW_SAMPLES = 8        # passes in the median of the views' own time in a pass
EVERY_MAX = 10000
TIME_MS_DEFAULT, TIME_MS_MAX = 4.0, 50.0
MEMORY_MB_DEFAULT, MEMORY_MB_MAX = 64, 512
INPUTS_MAX = 8
PARAMS_MAX = 16
TENSOR_AXES_MAX, TENSOR_ELEMENTS_MAX = 4, 1000000

# Published piece resource limits and chunking budgets (see _programs_pieces.py).
PUBLISHED_BYTES_MAX = 512 << 20
PUBLISHED_RESOURCE_MAX = 128 << 20
PUBLISHED_TEXTURE_MAX = 256 << 20
PART_BYTES = 8 << 20
RESOURCE_BACKLOG_BYTES = PART_BYTES + (64 << 10)

_LOCK = threading.RLock()
_WATCH_EPOCH = 0  # publication admission, protected by _LOCK
_OUTPUT_EVENTS = deque()  # callbacks in program state order, protected by _LOCK
_OUTPUT_DRAINING = False
_OUTPUT_PENDING_MAX = 16  # the wire's MAX_PROGRAMS
_OUTPUT_CONTROL_MAX = 16  # installed programs plus pending removals reserve control slots
_OUTPUT_BYTES_MAX = 64 << 20  # the default program memory budget
_PROGS: dict = {}       # id -> _Program, in the order of installation
_DRAWN: dict = {}       # id -> _Program of a piece without a derive half, only relayed
_IMPORTS: dict = {}     # input name -> its newest _Import
_LIVE: dict = {}        # handle -> each _Import not released yet (an older one may still be bound)
_ACTS: dict = {}        # module name -> its last output, for the inputs <module>.act
_ACT_HOOKS: dict = {}   # module name -> the forward hook's handle
_ABANDONED: list = []   # runtimes given up after a timeout, kept until the GPU ends their last pass
_STATE = {"rt": None, "mod": None, "model": None, "optimizer": None, "marks": None, "step": None,
          "dense": True, "drained": 0, "short": 0, "long": 0, "switches": 0, "pass": None,
          "pass_budget": True, "floor": 0, "view_s": deque(maxlen=VIEW_SAMPLES), "interp": False}
_VIEWS = {"ring": None}  # the network route's ring while it has snapshots to take
_OUTBOX: dict = {}       # id(link) -> [link, ctl, program, deque of chunks], under _OUTBOX_LOCK
_OUTBOX_LOCK = threading.Lock()
_clock = time.perf_counter
_WARNED: set = set()
_CAD = _Cadence()


# ---- the watched model --------------------------------------------------------------------------
def attach(model, optimizer, handle=None) -> None:
    """Give the facade the model and the optimizer of ``caliper.watch(model, optimizer)``.

    Loads no native module. With ``handle`` (a WatchHandle), its ``remove()`` detaches again.
    """
    global _WATCH_EPOCH
    conflict = None
    with _LOCK:
        from .interp import _publish
        # A watch replacement must not expose tensors from the previous model.
        if _STATE["model"] is not None and _STATE["model"] is not model:
            _detach_locked()
        for name in _publish._TENSORS:
            if _publish._model_name(model, name):
                conflict = name
                break
        if conflict is None:
            if _STATE["model"] is not model:
                _WATCH_EPOCH += 1
            _STATE["model"] = model
            _STATE["optimizer"] = optimizer
            if handle is not None and hasattr(handle, "_handles"):
                handle._handles.append(_Detach(model))
    _drain_outputs()
    if conflict is not None:
        if handle is not None:
            handle.remove()
        raise ValueError(f"publication {conflict!r} conflicts with a model tensor")


def detach(model=None) -> None:
    """Forget the watched model (only ``model``, when given), and remove every program."""
    with _LOCK:
        if model is not None and _STATE["model"] is not model:
            return
        _detach_locked()
    _drain_outputs()


def _detach_locked():
    """Remove watch-owned state under _LOCK and defer external output callbacks."""
    global _WATCH_EPOCH
    _WATCH_EPOCH += 1
    for pid in list(_PROGS) + list(_DRAWN):
        _remove(pid, clear_output=True)
    _drop_act_hooks()
    from .interp import _publish
    _publish._clear()
    _release_unused()
    _STATE["model"] = None
    _STATE["optimizer"] = None


def _drop_act_hooks() -> None:
    for h in _ACT_HOOKS.values():
        h.remove()
    _ACT_HOOKS.clear()
    _ACTS.clear()
    for p in _PROGS.values():
        p.act_dirty = False


def _torch_adapter():
    from ._adapter import TORCH
    return TORCH


def _resolve(name: str):
    """Return a known tensor or None for an unknown name.

    Gradients, module outputs, and fixed interpretation names can await their first value.
    """
    from .interp._publish import INTERPRETATION_TENSORS, _resolve as published
    tensor = published(name)
    if tensor is not None:
        return tensor
    model = _STATE["model"]
    if model is None:
        if name in INTERPRETATION_TENSORS:
            raise _Waiting(name)
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
                lambda _m, _i, out, mname=mname, epoch=_WATCH_EPOCH: _keep_act(mname, out, epoch))
        t = _ACTS.get(mname)
        if t is None:
            raise _Waiting(name)
        return t
    t = dict(model.named_parameters()).get(name)
    if t is None:
        t = dict(model.named_buffers()).get(name)
    if t is None and name in INTERPRETATION_TENSORS:
        raise _Waiting(name)
    return t


def _keep_act(mname, out, epoch) -> None:
    from ._intervention_protocol import is_internal_forward
    if is_internal_forward():
        return
    if isinstance(out, (tuple, list)):
        out = next((x for x in out if hasattr(x, "detach")), None)
    if out is not None:
        with _LOCK:
            if epoch != _WATCH_EPOCH or mname not in _ACT_HOOKS:
                return
            _ACTS[mname] = out.detach()
            source = f"live:{mname}.act"
            for p in _PROGS.values():
                if p.state in ("waiting", "compiled", "running") and any(
                        src == source for _, src in p.inputs):
                    p.act_dirty = True


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
    if p.kind not in ("geometry", "pixels", "tensor"):
        raise _Refused("program.output_mismatch", "The output is of kind geometry, pixels or tensor.")
    if len(p.params) > PARAMS_MAX:
        raise _Refused("program.over_limits", f"A program has at most {PARAMS_MAX} params.")
    if p.kind == "tensor":
        shape = p.shape
        ok = (isinstance(shape, (list, tuple)) and 1 <= len(shape) <= TENSOR_AXES_MAX
              and all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in shape))
        if not ok or math.prod(shape) > TENSOR_ELEMENTS_MAX:
            raise _Refused("program.output_mismatch",
                           f"A tensor output has a shape of 1 to {TENSOR_AXES_MAX} axes from 1 "
                           f"with at most {TENSOR_ELEMENTS_MAX} elements.")
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


def _resolve_inputs(inputs):
    """Check every input name before a known but absent tensor delays compilation."""
    tensors, waiting = {}, None
    for var, src in inputs:
        name = src[len("live:"):]
        try:
            tensor = _resolve(name)
        except _Waiting as exc:
            waiting = exc
            continue
        if tensor is None:
            raise _Refused("program.unknown_input", f"The script has no tensor named {name}.")
        tensors[var] = tensor
    if waiting is not None:
        raise waiting
    return tensors


def _compile(p: _Program) -> None:
    """Import the inputs and compile the program; raises _Refused or _Waiting."""
    if p.kind == "piece":
        _compile_piece(p)
        return
    tensors = _resolve_inputs(p.inputs)
    bound = {}
    for var, src in p.inputs:
        name = src[len("live:"):]
        bound[var] = _import(name, tensors[var])
    rt = _runtime()
    out = p.output
    wg = p.decl.get("workgroups") or (0, 0, 0)
    old = p.handle
    extra = {}
    if p.kind == "tensor":
        extra["tensor_shape"] = [int(n) for n in p.shape]
    if p.params:
        extra["params"] = [(k, float(v)) for k, v in p.params.items()]
    try:
        h = rt.program(p.id, p.wgsl, [(v, im.handle) for v, im in bound.items()], p.kind,
                       entry=p.decl.get("entry"), part=str(out.get("part", "points")),
                       max_count=int(out.get("max_count") or 0), arrays=list(out.get("arrays") or []),
                       width=int(out.get("width") or 0), height=int(out.get("height") or 0),
                       workgroups=tuple(int(x) for x in wg), time_ms=p.time_ms,
                       memory_bytes=p.memory_mb << 20, **extra)
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
    ACTIVE = (any(p.state in ("waiting", "compiled", "running") for p in _PROGS.values())
              or _VIEWS["ring"] is not None or bool(_OUTBOX))
    _prune_act_hooks()


def _prune_act_hooks() -> None:
    """Remove the forward hook of each module whose output no program that may still run reads.

    Every change of a program's state ends in :func:`_set_active`, thus the hooks leave the model
    when the last program that needs them stops for any reason: a removal, a refusal, a stop over
    time or over budget, a timeout or a lost device.
    """
    for p in _PROGS.values():
        if p.state not in ("waiting", "compiled", "running"):
            p.act_dirty = False
    if not _ACT_HOOKS:
        return
    needed = {src[len("live:"):-len(".act")] for p in _PROGS.values()
              if p.state in ("waiting", "compiled", "running")
              for _var, src in p.inputs if src.startswith("live:") and src.endswith(".act")}
    for mname in [m for m in _ACT_HOOKS if m not in needed]:
        _ACT_HOOKS.pop(mname).remove()
        _ACTS.pop(mname, None)


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

    Raises:
        RuntimeError: Installed programs and pending output removals occupy all 16 admission slots.
    """
    _prepare(str(wgsl), (declaration or {}).get("entry"))
    with _LOCK:
        _check_output_capacity(program_id)
        _remove(program_id)
        if _ABANDONED:
            _reap()
        if not ACTIVE:
            _CAD.reset()                             # the measurement starts again from this install
            _STATE.update(dense=True, drained=0, short=0, long=0, switches=0,
                          interp=_is_interpretability_run())
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


def _prepare(wgsl: str, entry) -> None:
    """Compile the program for the device before the lock: the compile under the lock then takes
    the compiled library and pipeline. A failure here is reported again by that compile."""
    try:
        from . import _producer
        prepare = getattr(_producer, "prepare", None)
        if prepare is not None:
            prepare(wgsl, None if entry is None else str(entry))
    except Exception:  # the module cannot load: the compile under the lock says why
        pass


def set_params(program_id: str, params: dict) -> None:
    """Give new values to the uniform ``params`` of a program; its next pass reads them.

    The values are stored by name over the earlier ones, and a program that is compiled gets them
    at once in the runtime. Nothing compiles again and the SHA-256 stays. A name that the WGSL
    struct lacks leaves the old values and shows as the error ``program.params_mismatch`` of the
    program. An unknown id does nothing.

    Args:
        program_id: The program's id.
        params: Names of the uniform's fields and their numbers.
    """
    item = None
    with _LOCK:
        p = _PROGS.get(str(program_id))
        if p is None:
            return
        new = {str(k): float(v) for k, v in dict(params or {}).items()}
        old = dict(p.params)
        p.params.update(new)
        rt = _STATE["rt"]
        if p.handle is not None and rt is not None and new:
            try:
                rt.set_params(p.handle, list(new.items()))
            except RuntimeError as e:
                p.params = old
                p.error = _refusal(e).error
                item = p.item()
        if p.params != old:
            p.params_dirty = True
    if item is not None:
        _notify([item])


def _remove(program_id: str, *, clear_output=False):
    with _LOCK:
        p = _PROGS.pop(str(program_id), None)
        if p is None:
            p = _DRAWN.pop(str(program_id), None)
            if p is not None:
                _retire_output(p, clear_output)
            return p
        rt = _STATE["rt"]
        if rt is not None and p.handle is not None:
            rt.remove(p.handle)
        p.handle = None
        p.bound = {}
        p.act_dirty = False
        _release_unused()
        if not _PROGS:
            _drop_act_hooks()
        _retire_output(p, clear_output)
        _set_active()
    return p


def remove(program_id: str) -> None:
    """Stop and remove a program, and clear its output; an unknown id does nothing."""
    _remove(program_id, clear_output=True)
    _drain_outputs()


def status() -> list:
    """Return one status item for each installed program, in the order of installation."""
    with _LOCK:
        return [p.item() for p in _PROGS.values()] + [p.item() for p in _DRAWN.values()]


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
                    "wanted_every": _CAD.want, "ceiling": _CAD.ceiling,
                    "sparse_marks": not _STATE["dense"], "mark_switches": _STATE["switches"]})
        return out


def _stop_all(code: str, message: str) -> list:
    """A timeout or a lost device: every program stops, and the training thread goes on at once.

    Caliper cannot stop a pass that runs on the GPU. It stops waiting for it, and the GPU stays
    busy until the driver ends the work. Thus the runtime is given up without a wait
    (``Runtime.abandon()``): its GPU objects wait in the layer's reaper, and the facade keeps the
    runtime, with the capsules of its imports, in ``_ABANDONED`` until the GPU ended the last pass
    (:func:`_reap`) or the process exits. Nothing touches its imports again.
    """
    changed = []
    for p in _PROGS.values():
        if p.state in ("waiting", "compiled", "running"):
            p.state = "stopped"
            p.error = {"code": code, "message": message, "line": None, "column": None}
            p.warning = None
            changed.append(p)
        p.handle = None
        p.bound = {}
    _IMPORTS.clear()
    _LIVE.clear()
    rt = _STATE["rt"]
    _STATE["rt"] = None
    if rt is not None:
        abandon = getattr(rt, "abandon", None)
        if abandon is None:
            rt.close()
        else:
            abandon()
            _ABANDONED.append(rt)
    return changed


def _reap() -> None:
    """Let the runtimes given up after a timeout go once the GPU ended their last passes."""
    mod = _STATE["mod"]
    try:
        done = mod is None or mod.reaping() == 0
    except Exception:                                # an older module: keep them until exit
        done = False
    if done:
        _ABANDONED.clear()


def views(ring) -> None:
    """The network route's ring has snapshots or series values to take (``ring``), or none
    (``None``): from then on the shared cadence gates its snapshots at the checkpoint. Called on
    the thread that changed a subscription."""
    with _LOCK:
        old = _VIEWS["ring"]
        if old is not None and old is not ring:
            old.release_gate()
        if ring is not None and old is None and not ACTIVE:
            _CAD.reset()                             # the measurement starts again
            _STATE.update(dense=True, drained=0, short=0, long=0, switches=0, floor=0,
                          interp=_is_interpretability_run())
            _STATE["pass"] = None
            if _STATE["marks"]:
                _STATE["marks"].clear()
        _VIEWS["ring"] = ring
        _set_active()


def views_closed(ring) -> None:
    """The ring ``ring`` has nothing to take any more, or its server closed: the shared cadence
    stops gating it, unless another ring has taken its place already."""
    with _LOCK:
        if _VIEWS["ring"] is not ring:
            return
        ring.release_gate()
        _VIEWS["ring"] = None
        _set_active()


def pass_next() -> bool:
    """Whether the next step may do the work that viewers ask for: always while the shared
    cadence is off, else when the checkpoint made the next step a pass."""
    return not ACTIVE or _STATE["pass"] is not None


def views_stats() -> dict:
    """The cadence of the views' snapshots and the passes that only the floor allowed."""
    with _LOCK:
        ring = _VIEWS["ring"]
        out = dict(ring.cadence) if ring is not None else {}
        out["floor_passes"] = _STATE["floor"]
        return out
