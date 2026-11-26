"""Adaptive cadence feedback, step marking, and over-budget tracking for GPU programs."""
from __future__ import annotations

from collections import deque
import math

from . import _controls
from ._programs_types import _Program


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

    Each cycle runs from the mark before one pass, at step ``k - 1``, to the mark before the next
    pass, at step ``k' - 1``. Its middle mark ``m`` is at ``max(k + 1, (k + k') // 2)``. Both
    halves end at marks of steps without a pass, thus a wait that only drains torch's queued work
    costs nothing here. The controller measures per cycle:

    - the base step time ``b``: the time per step of the second half, from ``m`` to ``k' - 1``,
      far from both passes; ``b`` is the median of the last ``BASE_SAMPLES`` cycles, thus it
      follows a step time that changes;
    - the overhead ``D`` of the pass at ``k``: the time of the first half, from ``k - 1`` to ``m``,
      minus ``m - k + 1`` steps at the time per step of the same cycle's second half, thus a step
      time that changes moves only the one cycle across the change. It counts the GPU's idle time while the training thread waits,
      runs the pass, publishes and queues the next step, and the slower steps after a pass while
      torch's queue fills again or the publish path works in its own thread. ``D`` is the mean of
      the last ``COST_SAMPLES`` cycles without the largest and the smallest. Thus it ignores a single
      slow pass but counts costs that come back often, which a median can miss.

    When the training thread waits at each step anyway, all marks are late by the same time, and
    the lateness cancels. Else a mark in the middle of a cycle can be late, which makes ``D``
    larger and ``b`` smaller, thus ``D`` can come out larger than it is, but not smaller.

    After each new ``D`` it wants ``ceil(D / (SHARE * b))`` steps between passes, between
    ``N_MIN`` and the ceiling. N moves up as soon as it is below that value, by at most a factor
    ``GROW``, and down only when it is more than ``CHANGE`` above it, by at most a factor
    ``SHRINK``. Thus N rests between the wanted value and ``1 + CHANGE`` times it.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        from . import _programs as _mod

        self.done: dict = {}        # step -> completion time of its mark, from the last cycle's start
        self.passes: set = set()    # the recent steps with a pass
        self.k = None               # the step of the last pass whose mark completed
        self.base = deque(maxlen=_mod.BASE_SAMPLES)
        self.cost = deque(maxlen=_mod.COST_SAMPLES)
        self.n = None               # N, None before the first measured overhead
        self.want = None            # ceil(D / (target * b)) of the last update
        self.target = _mod.SHARE    # the share the passes keep to: VIEWS_SHARE while views attach
        self.ceiling = _mod.EVERY_MAX
        self.over = 0               # updates in a row that wanted more than the ceiling
        self.last_pass = None       # the step of the last pass
        self.prev_pass = None       # the pass before it, for unpass()
        self.view_cost: dict = {}   # pass step -> the training thread's time in its view calls
        self.extra = 0.0            # the training thread's time in the parts sent since the last cycle

    def restart(self) -> None:
        """Forget the marks and passes of the current cycle, but keep ``b``, ``D``, N and the step
        of the last pass: the script numbered its steps again, or the clock of the completions
        changed."""
        self.done = {}
        self.passes = set()
        self.k = None
        self.view_cost = {}

    def reset_mode(self) -> None:
        """Clear cadence measurements when switching between training and interpretability runs,
        keeping ``last_pass`` and ``prev_pass``."""
        from . import _programs as _mod

        self.done = {}
        self.passes = set()
        self.k = None
        self.base.clear()
        self.cost.clear()
        self.n = None
        self.want = None
        self.target = _mod.SHARE
        self.ceiling = _mod.EVERY_MAX
        self.over = 0
        self.view_cost = {}
        self.extra = 0.0

    def record_pass_cost(self, cost_s: float) -> None:
        """Record direct program-pass wall time for an interpretability run without touching
        training-step timing or cadence throttling."""
        self.cost.append(max(0.0, float(cost_s)))

    def step_s(self):
        """The base step time ``b`` in seconds, or None before the first stretch."""
        return _median(self.base) if self.base else None

    def cost_s(self):
        """The overhead ``D`` of a pass in seconds, or None before the first pass is measured: the
        mean of the last cycles without the largest and the smallest once there are four."""
        c = sorted(self.cost)
        if len(c) >= 4:
            c = c[1:-1]
        return sum(c) / len(c) if c else None

    def share(self):
        """The measured share ``D / (N * b)`` of the training time, or None."""
        from . import _programs as _mod

        b, d = self.step_s(), self.cost_s()
        return None if b is None or d is None or b <= 0 else d / ((self.n or _mod.N_MIN) * b)

    def due(self, step: int) -> bool:
        """Whether ``step`` may run a pass: N steps after the last one, or before it when the
        script numbered its steps again."""
        from . import _programs as _mod

        last = self.last_pass
        return last is None or step < last or step - last >= (self.n or _mod.N_MIN)

    def passed(self, step: int) -> None:
        """Step ``step`` runs a pass."""
        self.passes.add(step)
        self.prev_pass = self.last_pass
        self.last_pass = step

    def unpass(self, step: int) -> None:
        """The pass of ``step`` took nothing: it counts as a step without a pass."""
        if self.last_pass == step:
            self.passes.discard(step)
            self.last_pass = self.prev_pass

    def repass(self, old: int, new: int) -> None:
        """The pass decided for step ``old`` runs at step ``new``, the number of the next
        checkpoint, because the script's step numbers jumped past ``old``."""
        if old in self.passes and self.last_pass == old:
            self.passes.discard(old)
            self.passes.add(new)
            self.last_pass = new

    def period(self, every_min: int) -> int:
        """The steps between two passes of a program: max(N, every), rounded up to a multiple of N."""
        from . import _programs as _mod

        n = self.n or _mod.N_MIN
        return min(n * math.ceil(max(n, every_min) / n), _mod.EVERY_MAX)

    def completed(self, step: int, t: float) -> None:
        """The mark of ``step`` completed at ``t``."""
        from . import _programs as _mod

        done = self.done
        done[step] = t
        if step not in self.passes:
            return
        end = done.get(step - 1)                    # the mark before this pass closes a cycle
        k = self.k
        if end is not None and step - 1 not in self.passes:
            if k is None:                           # the first cycle: no pass at its start
                first = min(done)
                if step - 1 > first and end > done[first]:
                    self.base.append((end - done[first]) / (step - 1 - first))
            else:
                m = max(k + 1, (k + step) // 2)
                mid, start = done.get(m), done.get(k - 1)
                b = None
                if mid is not None and step - 1 > m and end > mid:
                    b = (end - mid) / (step - 1 - m)
                    self.base.append(b)
                if mid is not None and start is not None and self.base:
                    b = self.step_s() if b is None else b
                    d = max(0.0, mid - start - (m - k + 1) * b)
                    # never less than the training thread's own time in the pass's view calls; the
                    # parts of large versions that later steps sent belong to the passes too
                    self.cost.append(max(d, self.view_cost.pop(k, 0.0)) + self.extra)
                    self.extra = 0.0
                    self._update()
            if not self.base:
                # The cycle was too short to time a step without a pass, for example when a long
                # pass stalls the thread and the marks of the steps after it complete together:
                # longer cycles until one can be timed.
                self.n = min(_mod.EVERY_MAX, math.ceil((self.n or _mod.N_MIN) * _mod.GROW))
        self.k = step
        self.done = {s: v for s, v in done.items() if s >= step - 1}
        self.passes = {s for s in self.passes if s >= step - 1}
        self.view_cost = {s: v for s, v in self.view_cost.items() if s >= step}

    def _update(self) -> None:
        from . import _programs as _mod

        b, d = self.step_s(), self.cost_s()
        if b is None or d is None:
            return
        self.ceiling = max(_mod.N_MIN, min(_mod.EVERY_MAX, math.ceil(_mod.CEILING_S / b))) if b > 0 else _mod.N_MIN
        self.want = math.ceil(d / (self.target * b)) if b > 0 else _mod.EVERY_MAX
        n = self.n or _mod.N_MIN
        want = min(max(self.want, _mod.N_MIN), self.ceiling)
        if want > n:
            n = min(want, math.ceil(n * _mod.GROW))
        elif want * (1 + _mod.CHANGE) < n:
            n = max(want, math.floor(n / _mod.SHRINK))
        self.n = n
        self.over = self.over + 1 if self.want > self.ceiling and n >= self.ceiling else 0

    def over_budget(self, every_min: int) -> bool:
        """Whether the programs cannot run inside the share on this run: the wanted N stayed above
        the ceiling for ``OVER_UPDATES`` updates, and the declared ``every_min`` does not keep the
        passes rare enough either."""
        from . import _programs as _mod

        return (self.over >= _mod.OVER_UPDATES and self.want is not None
                and self.want > max(self.ceiling, every_min))


def _marks():
    """The step marks of torch's queue, made at the first step that needs them; None off MPS."""
    from . import _programs as _mod

    m = _mod._STATE["marks"]
    if m is None:
        m = _mod._STATE["marks"] = _mod._torch_adapter().step_marks() or False
    return m or None


def _mark(marks, step: int, now: float, is_pass: bool) -> list:
    """Put the mark of ``step`` on torch's queue when it is needed, and return the marks that
    completed since the last step.

    A mark at each step gives torch's queued work to the GPU at once, which keeps the GPU busy
    while the training thread queues the next step. A loop that waits for the GPU by itself, for
    example with ``loss.item()``, gains nothing from it, and there the extra command buffer costs
    about 4 % of a short step. Such a loop shows two signs: the previous mark has completed before
    each step, for ``DRAINED_STEPS`` steps in a row, and the first sync point of the last
    ``SHORT_WAITS`` passes waited less than ``WAIT_EMPTY``, thus torch's queue was empty at the
    step. Then the facade marks only every ``SPARSE`` steps, the step before a pass and the pass,
    and times the steps with its own clock. A sparse mark that is still pending at the next one, or
    ``SHORT_WAITS`` long waits at passes in a row, show that the loop no longer waits by itself, and
    each step is marked again.
    """
    from . import _programs as _mod

    if marks is None:
        return [(step, now, now)]
    done = marks.poll(now)
    if _mod._STATE["dense"]:
        _mod._STATE["drained"] = _mod._STATE["drained"] + 1 if marks.pending() == 0 else 0
        if _mod._STATE["drained"] >= _mod.DRAINED_STEPS and _mod._STATE["short"] >= _mod.SHORT_WAITS:
            _mod._STATE["dense"] = False
            _mod._STATE["switches"] += 1
            _mod._CAD.restart()
    elif (step % _mod.SPARSE == 0 and marks.pending() > 0) or _mod._STATE["long"] >= _mod.SHORT_WAITS:
        _mod._STATE["dense"], _mod._STATE["drained"] = True, 0
        _mod._STATE["switches"] += 1
        _mod._CAD.restart()
        done = []
    if _mod._STATE["dense"] or is_pass or step % _mod.SPARSE == 0 or _mod._CAD.due(step + 1):
        marks.record(step, now)                  # also gives torch's queued work to the GPU
    return done


def _every_ok(p: _Program, step: int) -> bool:
    """Whether ``every`` lets program ``p`` run at ``step``."""
    return p.last_step is None or step < p.last_step or step - p.last_step >= p.every_min


def _floor_due(ring, now: float) -> bool:
    """The views wait longer than FLOOR_S for a pass, and their own work in a pass is cheap
    enough that a pass every FLOOR_S keeps inside the share by itself."""
    from . import _programs as _mod

    if now - ring.pass_t < _mod.FLOOR_S:
        return False
    v = _mod._STATE["view_s"]
    return not v or _median(v) <= _mod._CAD.target * _mod.FLOOR_S


def _last_single_step() -> bool:
    """The run single-steps and pauses after the next step: that step is a pass whatever its
    views cost, so that the paused run shows its own step and not an older one."""
    c = _controls.CONTROLS
    return c.state == _controls.RUN_STEPPING and c.steps_left == 1


def _views_status(ring, floor: bool) -> dict:
    """The cadence of the views' snapshots, for their status: the steps between two passes,
    the passes a second, the measured times and share, and what sets the pace: ``budget`` when
    the passes come less often than the views ask, ``floor`` for a pass of the floor, else
    ``rate``, the views' own rate."""
    from . import _programs as _mod

    cad = _mod._CAD
    b, d, share = cad.step_s(), cad.cost_s(), cad.share()
    n = cad.n or _mod.N_MIN
    rate = ring.pass_rate()
    asked = ring.asked_interval()
    if floor:
        limit = "floor"
    elif b is not None and asked is not None and n * b > asked * 1.1:
        limit = "budget"
    else:
        limit = "rate"
    return {"every": n, "passes_per_s": None if rate is None else round(rate, 2),
            "step_ms": None if b is None else round(b * 1e3, 3),
            "pass_ms": None if d is None else round(d * 1e3, 3),
            "share": None if share is None else round(share, 4), "budget": cad.target, "limit": limit}


def _clear_over_budget_warnings() -> list:
    """Clear active ``program.over_budget`` warnings on running programs."""
    from . import _programs as _mod

    changed = []
    for p in _mod._PROGS.values():
        if p.warning is not None and p.warning.get("code") == "program.over_budget":
            p.warning = None
            changed.append(p)
    return changed


def _update_over_budget() -> list:
    """Set or clear recoverable ``program.over_budget`` warnings in a training run.

    Built-in library pieces get no hard stop and no budget warning. Agent-written pieces and
    other programs carry a warning while keeping running; a hard stop happens only on a real
    hang (``program.timeout``).
    """
    from . import _programs as _mod

    cad = _mod._CAD
    warnable = [p for p in _mod._PROGS.values()
                if p.state in ("waiting", "compiled", "running") and not _mod._is_builtin_library_piece(p)]
    if not warnable or not cad.over_budget(min(p.every_min for p in warnable)):
        return _clear_over_budget_warnings()
    b, d = cad.step_s(), cad.cost_s()
    if b is None or d is None or b <= 0:
        return _clear_over_budget_warnings()
    message = (f"A pass adds {d * 1e3:.2f} ms to a step of {b * 1e3:.3f} ms, thus the programs need "
               f"at least {cad.want} steps between passes to stay under {cad.target:.0%} of the "
               f"training time, more than the ceiling of {cad.ceiling} steps ({cad.ceiling * b:.3g} s) "
               "on this run. The programs keep running.")
    warning = {"code": "program.over_budget", "message": message, "line": None, "column": None}
    changed = []
    for p in _mod._PROGS.values():
        if p.state not in ("waiting", "compiled", "running"):
            continue
        if _mod._is_builtin_library_piece(p):
            if p.warning is not None and p.warning.get("code") == "program.over_budget":
                p.warning = None
                changed.append(p)
            continue
        was_warned = p.warning is not None and p.warning.get("code") == "program.over_budget"
        p.warning = dict(warning)
        if not was_warned:
            changed.append(p)
    return changed
