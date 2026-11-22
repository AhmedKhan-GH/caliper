"""The run side of caliper.watch(model, optimizer): the checkpoint, ``lr``, ``loss`` and ``model``.

A step pre-hook of the optimizer is the checkpoint of the run. In order, it publishes the loss of
the step as the series ``loss``, calls caliper.step() (pending parameter values apply, queued
actions run, a pause waits there, before the weights change), writes the parameter ``lr`` into the
parameter groups, publishes the rate that the step uses as the series ``lr``, and publishes the
scene ``model`` when a viewer watches it. A step post-hook puts the scheduler's rates back (see
below) and publishes the loss of a closure that ran inside ``optimizer.step(closure)``.

THE LOSS is the last 0-d tensor that ``backward()`` ran on. A wrapper of
``torch.autograd.backward`` (which ``Tensor.backward`` calls) keeps it, detached, while a watch
with an optimizer is on; the wrapper goes when the last such watch is removed. The step post-hook
does not receive the value that ``optimizer.step(closure)`` returns, and wrapping
``optimizer.step`` would break the check of a learning rate scheduler, so the loss of a closure is
the 0-d tensor that its own ``backward()`` ran on.

LEARNING RATE AND A SCHEDULER. The parameter ``lr`` is the rate of parameter group 0 at the step
where a new value applies; the other groups move by the same factor. Without a scheduler, the hook
writes the new rates into the groups, where they stay. A torch learning rate scheduler (it puts
``initial_lr`` into the groups) sets the rates again at each of its steps, from its base rates or
from the rates it set last. So with one attached, the hook keeps the factor between the new value
and the scheduler's rate of group 0, multiplies each group's scheduled rate by it before every
step and puts the scheduled rates back after the step. The scheduler thus never undoes the change
and never compounds it, closed-form schedules (LambdaLR, OneCycleLR, CyclicLR) included: the whole
schedule from that step on is scaled. ``optimizer.param_groups`` and ``scheduler.get_last_lr()``
show the scheduler's rates between steps, and the series ``lr`` shows the rate each step used.

COST. With nothing waiting, a step costs the checkpoint's flag, two series values, a few dict
reads and one clock read. The scene's colors (the mean absolute output of each module) are
computed in one forward pass at most ``SCENE_RATE_HZ`` times a second, and only while a viewer
watches the scene. Nothing here reads a value back from the device.
"""
from __future__ import annotations

import functools
import math
import threading
import time
import warnings

__all__ = ["RunHooks"]

SCENE_NAME = "model"
"""The name of the scene of the watched modules."""
ENTITY = "modules"
"""The entity of the scene that holds one box per watched module."""
SCENE_RATE_HZ = 10.0
"""How often the scene's colors are computed and sent while a viewer watches it."""
LR_DOWN = 1000.0
"""The range of ``lr`` goes down to the start value divided by this."""
LR_UP = 10.0
"""The range of ``lr`` goes up to the start value multiplied by this."""
_PITCH = 1.25              # distance between two boxes along +y
_SIDE_MIN, _SIDE_MAX = 0.25, 1.0
_GREY = (0.55, 0.55, 0.6)
_VIRIDIS = ((0.267, 0.005, 0.329), (0.229, 0.322, 0.546), (0.128, 0.567, 0.551),
            (0.369, 0.789, 0.383), (0.993, 0.906, 0.144))


# ---------------------------------------------------------------------------
# the loss tap: the last 0-d tensor that backward() ran on, shared by every watch
# ---------------------------------------------------------------------------
_tap_lock = threading.Lock()
_tap_users = 0
_tap_orig = None
_loss = [0, None]          # [number of 0-d backward calls, the last one detached]


def _tap_on() -> None:
    global _tap_users, _tap_orig
    import torch
    with _tap_lock:
        _tap_users += 1
        if _tap_orig is not None:
            return
        orig = torch.autograd.backward
        Tensor = torch.Tensor

        @functools.wraps(orig)
        def backward(tensors, *args, **kwargs):
            if type(tensors) is Tensor and tensors.ndim == 0:
                _loss[1] = tensors.detach()
                _loss[0] += 1
            return orig(tensors, *args, **kwargs)
        backward._caliper_loss_tap = True
        torch.autograd.backward = backward
        _tap_orig = orig


def _tap_off() -> None:
    global _tap_users, _tap_orig
    import torch
    with _tap_lock:
        _tap_users -= 1
        if _tap_users > 0 or _tap_orig is None:
            return
        if getattr(torch.autograd.backward, "_caliper_loss_tap", False):
            torch.autograd.backward = _tap_orig
        _tap_orig = None                 # a wrapper another library put on top stays; ours no-ops
        _loss[1] = None


# ---------------------------------------------------------------------------
# learning rates of the parameter groups (a float, or a tensor as torch allows)
# ---------------------------------------------------------------------------
def _rate(group) -> float:
    return float(group["lr"])


def _set(group, value: float) -> None:
    lr = group["lr"]
    if isinstance(lr, float) or not hasattr(lr, "fill_"):
        group["lr"] = value
    else:
        lr.fill_(value)


def _positive(x: float) -> bool:
    return x > 0.0 and math.isfinite(x)


class RunHooks:
    """The hooks of one ``watch(model, optimizer)``: the checkpoint, ``lr``, ``loss``, ``model``.

    Attributes:
        lr (Param | None): The parameter ``lr``, or ``None`` when group 0 has no positive rate.
        factor (float): The factor that multiplies the scheduler's rates (1 without a scheduler).
        armed (bool): Whether the next forward pass records the scene's colors.
    """

    def __init__(self, pkg, handle, optimizer, scene: bool, rate=None):
        import torch
        if not (hasattr(optimizer, "register_step_pre_hook")
                and hasattr(optimizer, "param_groups")):
            raise TypeError("caliper.watch expects a torch.optim.Optimizer as optimizer")
        self.pkg = pkg
        self.opt = optimizer
        self.rate = rate
        self.factor = 1.0
        self._saved = None               # the scheduler's rates while the step runs
        self._written = None
        self._loss_seen = _loss[0]
        self.armed = False
        self.means: list = []
        self.scene = None
        self._next = 0.0
        self._interval = 1.0 / SCENE_RATE_HZ
        self._geom: dict = {}
        self._torch = torch
        with pkg.LOCK:
            pkg._auto_tried = True      # the server carries the views, not the shared memory
        pkg.serve()
        groups = optimizer.param_groups
        start = _rate(groups[0]) if groups else float("nan")
        self.start = start
        self.lr = None
        if _positive(start):
            self.lr = pkg.param("lr", start, min=start / LR_DOWN, max=start * LR_UP, log=True,
                                help="The learning rate of parameter group 0. A new value applies "
                                     "at the next step and moves every group by the same factor.")
            self._seen = self.lr._version
        else:
            warnings.warn(f"caliper.watch: parameter group 0 has the learning rate {start!r}, so "
                          "the parameter 'lr' is not registered", RuntimeWarning, stacklevel=4)
        self.scene = pkg.scene(SCENE_NAME) if scene else None
        handle._handles.append(optimizer.register_step_pre_hook(self._pre))
        handle._handles.append(optimizer.register_step_post_hook(self._post))
        _tap_on()
        self._tapped = True

    # -- the scene -------------------------------------------------------------
    def modules(self, labels, mods) -> None:
        """Lay out the boxes of the modules ``labels`` (the order of ``WatchHandle.modules``)."""
        n = len(labels)
        self.means = [None] * n
        if self.scene is None or not n:
            self.scene = None
            return
        counts = [sum(p.numel() for p in m.parameters()) for m in mods]
        top = max(counts) or 1
        sides = [_SIDE_MIN + (_SIDE_MAX - _SIDE_MIN) * math.sqrt(c / top) for c in counts]
        self._centers = [(0.0, _PITCH * i, 0.0) for i in range(n)]
        self._halves = [(s / 2, s / 2, s / 2) for s in sides]
        self._labels = [_cut(x) for x in labels] if n <= 4096 else None
        self._publish(None)              # lists the scene for every viewer

    def record(self, i: int, out) -> None:
        """Keep the mean absolute value of output ``out`` of module ``i`` (a forward hook)."""
        torch = self._torch
        t = out
        if isinstance(t, (tuple, list)):
            t = next((x for x in t if isinstance(x, torch.Tensor)), None)
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
            self.means[i] = t.detach().abs().mean()

    def _geometry(self, device):
        g = self._geom.get(device)
        if g is None:
            torch = self._torch
            g = (torch.tensor(self._centers, dtype=torch.float32, device=device),
                 torch.tensor(self._halves, dtype=torch.float32, device=device))
            self._geom[device] = g
        return g

    def _colors(self):
        torch = self._torch
        got = [m for m in self.means if m is not None]
        if not got:
            return None, None
        dev = got[0].device
        zero = torch.zeros((), device=dev)
        v = torch.stack([zero if m is None else m.float().to(dev) for m in self.means])
        v = v / v.max().clamp_min(1e-30)
        lut = torch.tensor(_VIRIDIS, dtype=torch.float32, device=dev)
        pos = v.clamp(0, 1) * (len(_VIRIDIS) - 1)
        i = pos.floor().clamp(max=len(_VIRIDIS) - 2).long()
        f = (pos - i).unsqueeze(1)
        return lut[i] * (1 - f) + lut[i + 1] * f, dev

    def _publish(self, colors, device=None) -> bool:
        c, h = self._geometry(device if colors is not None else "cpu")
        return self.scene.boxes(ENTITY, c, h, colors=_GREY if colors is None else colors,
                                labels=self._labels, fill=True, rate=0)

    def _scene_tick(self) -> None:
        if self.armed:                   # the forward pass of this step recorded the colors
            self.armed = False
            colors, dev = self._colors()
            self._publish(colors, dev)
            self.means = [None] * len(self.means)
        now = time.perf_counter()
        if now >= self._next:
            self._next = now + self._interval
            self.armed = self.scene._watched()

    # -- the learning rate -----------------------------------------------------
    def _set_lr(self, groups, v: float) -> None:
        s0 = _rate(groups[0])
        if "initial_lr" in groups[0]:    # a scheduler sets the rates: scale its schedule
            self.factor = v / (s0 if _positive(s0) else self.start)
            return
        self.factor = 1.0
        if _positive(s0):
            k = v / s0
            for g in groups:
                _set(g, _rate(g) * k)
        else:
            for g in groups:
                _set(g, v)

    def _apply_lr(self, groups):
        p = self.lr
        if p is not None and p._version != self._seen:
            self._seen = p._version
            self._set_lr(groups, float(p.value))
        if self.factor != 1.0 and "initial_lr" in groups[0]:
            saved = [g["lr"] for g in groups]
            saved_f = [_rate(g) for g in groups]
            for g, s in zip(groups, saved_f):
                _set(g, s * self.factor)
            self._saved = (saved, saved_f)
            self._written = [g["lr"] for g in groups]
        return groups[0]["lr"]

    def _restore(self) -> None:
        saved, self._saved = self._saved, None
        if saved is None:
            return
        groups = self.opt.param_groups
        for g, s, sf, wr in zip(groups, saved[0], saved[1], self._written):
            if g["lr"] is wr:            # nobody else changed it during the step
                if isinstance(s, float):
                    g["lr"] = s
                else:
                    _set(g, sf)
        self._written = None

    # -- the hooks -------------------------------------------------------------
    def _loss_tick(self) -> None:
        n, t = _loss
        if n != self._loss_seen and t is not None:
            self._loss_seen = n
            self.pkg.series(t, "loss", rate=self.rate)

    def _pre(self, opt, args, kwargs):
        self._loss_tick()
        self.pkg.step()
        groups = opt.param_groups
        if groups:
            self.pkg.series(self._apply_lr(groups), "lr", rate=self.rate)
        if self.scene is not None:
            self._scene_tick()

    def _post(self, opt, args, kwargs):
        if self._saved is not None:
            self._restore()
        self._loss_tick()                # the loss of a closure that ran inside step()

    def close(self) -> None:
        """Put the scheduler's rates back if a step is under way, remove ``lr`` and the loss tap."""
        self._restore()
        if self.lr is not None:
            self.lr.remove()
        if self._tapped:
            self._tapped = False
            _tap_off()


def _cut(label: str) -> str:
    b = label.encode("utf-8")
    return label if len(b) <= 63 else b[:63].decode("utf-8", "ignore")
