"""The data view seam of the network route (``caliper.serve()``).

The network route asks one *engine* three things about a view, and nothing
else about data views:

* ``offer(call)``: what the view offers, ``{"sources": [...], "default": {...}}``,
  the members of its entry in a VIEWS message (docs/references/wire-protocol.md,
  "What a view offers"). Called on the training thread whenever
  ``signature(call)`` changes, and at most once a second otherwise.
* ``resolve(view, request)``: the resolved data view for a SUBSCRIBE or
  VIEW_REQUEST, or ``caliper._wire.MessageError`` with code ``bad_data_view``.
  ``view`` is the view's VIEWS entry (``id``, ``name``, ``kind``, ``sources``,
  ``default``). Called on the network thread; it must not touch a device.
* ``apply(call, resolved)``: queue on the call's device the work that turns the
  call into the payload of ``resolved``, and return a :class:`Reduced` of
  device tensors. Called on the training thread, once per subscription group
  that is due. It must never wait for the device: no ``.item()``, ``.cpu()``,
  ``.numpy()``, ``.tolist()`` or ``synchronize()``. The route copies the
  result into pinned host memory without blocking and reads it once the GPU
  has finished.

plus ``signature(call)``, a cheap hashable that changes when the offer may
have changed. A :class:`Call` carries the view call: its kind, the view name,
the ``caliper._convert`` plan (the normalised payload writer of protocol v2)
and the raw call arguments. Series never reach an engine: their values travel
as they are.

:class:`DefaultEngine` is the engine of this module: the default data view of
each kind with the budget reductions (heatmaps and images pooled to
``max_extent``, points stride-sampled to ``max_points`` with their labels),
computed from the plan's payload. It refuses any other request with
``bad_data_view``. ``make_engine()`` is the one place that picks the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from . import _wire
from ._ring import (FLAG_AUTO_VMAX, FLAG_AUTO_VMIN, KIND_HEATMAP, KIND_HISTOGRAM, KIND_IMAGE,
                    KIND_POINTS, STATS_FLOATS, RingSlot)

__all__ = ["Call", "Reduced", "Engine", "DefaultEngine", "make_engine"]


@dataclass(eq=False)
class Call:
    """One view call as the network route sees it."""
    kind: int                   # CaliperLiveKind of the plan
    name: str                   # the view name
    plan: Any                   # caliper._convert.Plan: kind, shape, dtype, device, write(slot), ...
    args: tuple | None = None   # the call's arguments as _dispatch got them, or None:
                                #   show       (t, cmap, vmin, vmax, raw)
                                #   points     (z, labels, classes, cmap, (vmin, vmax))
                                #   histogram  (t, bins, None, None, None)
    cache: dict = field(default_factory=dict)   # per-call scratch an engine may use

    @property
    def device(self):
        return self.plan.device


@dataclass(eq=False)
class Reduced:
    """The payload of one snapshot under one resolved data view, still on the
    call's device (or CPU tensors for a CPU call)."""
    data: Any                    # heatmap (H, W) f32, image (H, W, 4) u8, points (N, D) f32,
                                 # histogram (B,) f32; contiguous
    aux: Any = None              # points: (N,) f32 labels or values
    stats: Any = None            # 1-D f32 tensor of stats_count entries, or None
    stats_count: int = 0
    colormap: int = 0
    flags: int = 0               # CALIPER_NET_SNAP_* (== caliper._ring FLAG_* >> 1, see _snap_flags)
    vmin: float = 0.0
    vmax: float = 1.0
    axes: list = field(default_factory=list)
    tile: dict | None = None
    points_dims: int | None = None
    label_count: int | None = None


class Engine(Protocol):
    def signature(self, call: Call): ...
    def offer(self, call: Call) -> dict: ...
    def resolve(self, view: dict, request: dict | None) -> dict: ...
    def apply(self, call: Call, resolved: dict) -> Reduced: ...


def snap_flags(flags: int) -> int:
    """caliper._ring FLAG_* (live_views.h CALIPER_LIVE_APPLY_*) -> the wire's
    CALIPER_NET_SNAP_* (live.h CALIPER_LIVE_SNAP_*)."""
    from ._ring import FLAG_AUTO_VMAX, FLAG_AUTO_VMIN, FLAG_RAW
    out = 0
    if flags & FLAG_AUTO_VMIN:
        out |= _wire.SNAP_AUTO_VMIN
    if flags & FLAG_AUTO_VMAX:
        out |= _wire.SNAP_AUTO_VMAX
    if flags & FLAG_RAW:
        out |= _wire.SNAP_RAW
    return out


_SOURCE = "data"
_DTYPE_NAME = {"torch.float32": "f32", "torch.uint8": "u8"}


def _axis(dim: str, count: int, size: int) -> dict:
    return {"dim": dim, "start": 0, "step": 1, "count": int(count), "size": int(size)}


_POOL_INDEX: dict = {}


def _pool_index(n: int, m: int, device):
    """Gather indices for pooling n positions into m cells by the protocol's
    rule: cell i covers [floor(i n / m), ceil((i + 1) n / m)) (torch's adaptive
    pooling). Returns (idx (m * k,), valid (m, k), counts (m,), k), cached on
    the device; short cells repeat their last position where valid is False."""
    key = (n, m, device)
    have = _POOL_INDEX.get(key)
    if have is not None:
        return have
    import torch
    i = torch.arange(m, dtype=torch.int64)
    start = i * n // m
    stop = -((-(i + 1) * n) // m)
    counts = stop - start
    k = int(counts.max())
    j = torch.arange(k, dtype=torch.int64)
    idx = torch.minimum(start[:, None] + j, stop[:, None] - 1)
    valid = j[None, :] < counts[:, None]
    have = (idx.reshape(-1).to(device), valid.to(device),
            counts.to(torch.float32).to(device), k)
    _POOL_INDEX[key] = have
    return have


def pool_axis(x, dim: int, m: int, op: str = "mean"):
    """Pool dimension ``dim`` of ``x`` to ``m`` cells (``mean``, ``max`` or
    ``min``) by the protocol's cell rule, on x's device. A NaN stays inside its
    own cell. (torch's adaptive pooling refuses sizes that do not divide on MPS.)"""
    import torch
    n = x.shape[dim]
    if m >= n:
        return x
    if n % m == 0:
        g = x.unflatten(dim, (m, n // m))
        return g.mean(dim + 1) if op == "mean" else (g.amax(dim + 1) if op == "max"
                                                     else g.amin(dim + 1))
    idx, valid, counts, k = _pool_index(n, m, x.device)
    g = x.index_select(dim, idx).unflatten(dim, (m, k))
    if op == "max":
        return g.amax(dim + 1)
    if op == "min":
        return g.amin(dim + 1)
    shape = [1] * g.dim()
    shape[dim], shape[dim + 1] = m, k
    g = torch.where(valid.view(shape), g, torch.zeros((), dtype=g.dtype, device=g.device))
    cshape = [1] * (g.dim() - 1)
    cshape[dim] = m
    return g.sum(dim + 1) / counts.view(cshape)


class DefaultEngine:
    """Default data views and budgets, computed from the plan's payload.

    The plan is written once per call into a scratch slot on the call's device
    (reused across calls: every write is ordered on the device's stream), then
    each subscription group pools or samples it to its budget."""

    def __init__(self):
        self._scratch: dict = {}        # view name -> (plan key, RingSlot)
        self._index: dict = {}          # (device, N, n) -> stride sample rows, on the device
        self._stats: dict = {}          # view name -> (key, f32[8]) sample stats for points

    def forget(self, name: str) -> None:
        """View ``name`` was removed: drop its scratch buffers."""
        self._scratch.pop(name, None)
        self._stats.pop(name, None)

    # -- what a view offers ----------------------------------------------------
    def signature(self, call: Call):
        return (call.kind, call.plan.key)

    def offer(self, call: Call) -> dict:
        p = call.plan
        if call.kind in (KIND_HEATMAP, KIND_IMAGE):
            dims = [{"name": "row", "size": int(p.shape[0])}, {"name": "col", "size": int(p.shape[1])}]
        elif call.kind == KIND_POINTS:
            dims = [{"name": "point", "size": int(p.shape[0])}]
        else:
            dims = []
        return {"sources": [{"name": _SOURCE, "dims": dims,
                             "dtype": _DTYPE_NAME.get(str(p.dtype), "f32")}],
                "default": {}}

    # -- resolving -------------------------------------------------------------
    def resolve(self, view: dict, request: dict | None) -> dict:
        r = _wire.resolve_data_view(view, request)
        if r.get("index") or r.get("reduce") or r.get("tile") is not None:
            raise _wire.MessageError(
                _wire.ERR_BAD_DATA_VIEW,
                "this producer offers each view's default data view with a budget only; "
                "'index', 'reduce' and 'tile' are not available")
        return r

    # -- computing ---------------------------------------------------------------
    def _slot(self, call: Call) -> RingSlot:
        """The plan written into the view's scratch slot (once per call)."""
        slot = call.cache.get("slot")
        if slot is not None:
            return slot
        import torch
        p = call.plan
        have = self._scratch.get(call.name)
        if have is None or have[0] != p.key:
            dev = p.device
            payload = torch.empty(p.shape, dtype=p.dtype, device=dev)
            stats = torch.zeros(STATS_FLOATS, dtype=torch.float32, device=dev)
            aux = torch.empty(p.aux_shape, dtype=torch.float32, device=dev) if p.aux_shape else None
            slot = RingSlot(0, payload, stats, aux, None, stats[0], stats[1], stats[2:8:2], stats[3:8:2])
            if p.init is not None:
                p.init(slot)
            self._scratch[call.name] = (p.key, slot)
        else:
            slot = have[1]
        p.write(slot)
        call.cache["slot"] = slot
        return slot

    def apply(self, call: Call, resolved: dict) -> Reduced:
        p = call.plan
        slot = self._slot(call)
        budget = resolved.get("budget", {})
        common = dict(colormap=p.colormap, flags=snap_flags(p.flags), vmin=p.vmin, vmax=p.vmax)
        if call.kind == KIND_HEATMAP:
            return self._heatmap(slot, budget, common)
        if call.kind == KIND_IMAGE:
            return self._image(slot, budget, common)
        if call.kind == KIND_POINTS:
            return self._points(call, slot, budget, common)
        if call.kind == KIND_HISTOGRAM:
            return Reduced(slot.payload, stats=slot.stats[:2], stats_count=2, **common)
        raise _wire.MessageError(_wire.ERR_UNSUPPORTED, f"kind {call.kind} has no data view")

    @staticmethod
    def _heatmap(slot, budget, common) -> Reduced:
        x = slot.payload
        h, w = x.shape
        ext = budget.get("max_extent", _wire.DEFAULT_MAX_EXTENT)
        oh, ow = min(h, ext), min(w, ext)
        if (oh, ow) != (h, w):
            pool = budget.get("pool", "mean")
            x = pool_axis(pool_axis(x, 0, oh, pool), 1, ow, pool).contiguous()
        # stats: the plan's range over the data before pooling (protocol v2's
        # rule; a mosaic's NaN padding is left out of it)
        return Reduced(x, stats=slot.stats[:2], stats_count=2,
                       axes=[_axis("row", h, oh), _axis("col", w, ow)], **common)

    @staticmethod
    def _image(slot, budget, common) -> Reduced:
        import torch
        x = slot.payload
        h, w, _ = x.shape
        ext = budget.get("max_extent", _wire.DEFAULT_MAX_EXTENT)
        oh, ow = min(h, ext), min(w, ext)
        if (oh, ow) != (h, w):
            y = pool_axis(pool_axis(x.float(), 0, oh), 1, ow)
            x = y.add_(0.5).clamp_(0.0, 255.0).to(torch.uint8).contiguous()
        return Reduced(x, axes=[_axis("row", h, oh), _axis("col", w, ow)], **common)

    def _points(self, call, slot, budget, common) -> Reduced:
        import torch
        p = call.plan
        n_all = p.shape[0]
        d = p.points_dims
        n = min(n_all, budget.get("max_points", _wire.DEFAULT_MAX_POINTS))
        pts, aux = slot.payload, slot.aux
        if n == n_all:
            data = pts[:, :d] if d == 2 else pts
            if d == 2:
                data = data.contiguous()
            stats = slot.stats                          # computed over every point
        else:
            key = (pts.device, n_all, n)
            idx = self._index.get(key)
            if idx is None:                             # stride sample: row floor(i * N / n)
                idx = torch.arange(n, dtype=torch.int64, device=pts.device).mul_(n_all).floor_divide_(n)
                self._index[key] = idx
            data = pts.index_select(0, idx)
            if d == 2:
                data = data[:, :2].contiguous()
            if aux is not None:
                aux = aux.index_select(0, idx)
            have = self._stats.get(call.name)
            if have is None or have[0] != p.key:
                have = (p.key, torch.zeros(STATS_FLOATS, dtype=torch.float32, device=pts.device))
                self._stats[call.name] = have
            stats = have[1]
            torch.aminmax(data, dim=0, out=(stats[2:2 + 2 * d:2], stats[3:3 + 2 * d:2]))
            if aux is not None and p.label_count == 0 and p.flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX):
                torch.aminmax(aux, out=(stats[0], stats[1]))
        return Reduced(data, aux=aux, stats=stats, stats_count=STATS_FLOATS,
                       axes=[_axis("point", n_all, n)], points_dims=d,
                       label_count=p.label_count, **common)


def make_engine() -> Engine:
    """The engine a new network route uses."""
    return DefaultEngine()
