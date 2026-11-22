"""The data view seam of the network route (``caliper.serve()``).

The network route asks one *engine* three things about a view, and nothing
else about data views:

* ``offer(call)``: what the view offers, ``{"sources": [...], "default": {...}}``,
  the members of its entry in a VIEWS message (docs/reference/wire-protocol.md,
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
have changed, and two optional members: ``serves(call, resolved)``, whether
the call carries what a subscription's data view reads (a view with several
sources gets one call per source), and ``forget(name)``. A :class:`Call`
carries the view call: its kind, the view name, the ``caliper._convert`` plan
(the normalised payload writer of protocol v2) and the raw call arguments.
Series never reach an engine: their values travel as they are.

:class:`DataViewEngine` is the engine (``make_engine()`` is the one place that
picks it). It computes every data view with caliper._dataview, on the call's
device, so a subscription's snapshot is the same payload the window and the
session draw for the same data view:

* a view with named dims (``dims=`` or ``data_view=`` on a call,
  ``caliper.offer``, ``caliper.watch``'s grouped views) offers the sources
  and dims of caliper._dataview's registry, and its call carries the source
  tensor (a :class:`caliper._dataview.DataCall` in the plan's ``write``);
* any other view offers one source, ``data``, over the payload of its call
  (heatmap and image dims ``row`` and ``col``, points ``point``, a histogram
  none), so a viewer can still range it, which is how it zooms, and pool or
  sample it to a budget.

Each subscription group's data view is resolved and checked again when the
source's sizes change; one that no longer fits gets no snapshots until it
fits again, and nothing raises on the training thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import json
from collections import OrderedDict

from . import _dataview as dv
from . import _wire
from ._ring import (FLAG_AUTO_VMAX, FLAG_AUTO_VMIN, KIND_HEATMAP, KIND_HISTOGRAM, KIND_IMAGE,
                    KIND_POINTS, STATS_FLOATS, RingSlot)

__all__ = ["Call", "Reduced", "Engine", "DataViewEngine", "make_engine"]


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
                                 # histogram (B,) f32
    aux: Any = None              # points: (N,) labels or values (cast to f32 when staged)
    stats: Any = None            # 1-D f32 tensor of stats_count entries; None for a heatmap or
                                 # points whose stats are those of the payload as sent: the
                                 # transport computes them on the host, with no device work
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
_PLAIN_DIMS = {KIND_HEATMAP: ("row", "col"), KIND_IMAGE: ("row", "col"), KIND_POINTS: ("point",),
               KIND_HISTOGRAM: ()}
_PROGRAMS = 256                 # compiled data views kept by an engine


def _bad(e: Exception):
    return _wire.MessageError(_wire.ERR_BAD_DATA_VIEW, str(e))


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


class DataViewEngine:
    """Data views for the network route, computed by caliper._dataview (see
    the module docstring). Everything on the call's device is queued work."""

    def __init__(self):
        self._scratch: dict = {}           # (view name, plan key) -> RingSlot
        self._programs: OrderedDict = OrderedDict()   # (name, signature, resolved) -> Program
        self._stats: dict = {}             # (view name, key) -> f32[8] stats of a point sample
        self._canon: dict = {}             # id(resolved) -> (resolved, its canonical JSON)
        self._last = None                  # (call, resolved, program) of the last lookup

    def forget(self, name: str) -> None:
        """View ``name`` was removed: drop its scratch buffers and programs."""
        for d in (self._scratch, self._stats):
            for k in [k for k in d if k[0] == name]:
                del d[k]
        for k in [k for k in self._programs if k[0] == name]:
            del self._programs[k]

    # -- what a view offers ----------------------------------------------------------------
    @staticmethod
    def _dv(call: Call):
        w = call.plan.write
        return w.call if isinstance(w, dv.DVWrite) else None

    def signature(self, call: Call):
        c = self._dv(call)
        if c is not None:
            return ("dv", id(c.entry), dv.REGISTRY.generation)
        return (call.kind, call.plan.key)

    @staticmethod
    def view_signature(name: str):
        """The signature of a view with data views without a call (None for any
        other view): it changes when the registry does, so the route refreshes
        the offer as soon as a source of the view gets or changes its shape."""
        e = dv.REGISTRY.get(name)
        return None if e is None else ("dv", id(e), dv.REGISTRY.generation)

    @staticmethod
    def _plain(call: Call) -> dv.Offer:
        p = call.plan
        dims = _PLAIN_DIMS.get(call.kind, ())
        dtype = "u8" if call.kind == KIND_IMAGE else "f32"
        spec = dv.SourceSpec(_SOURCE, dims, tuple(int(n) for n in p.shape[:len(dims)]), dtype)
        return dv.Offer(dv.KIND_NAMES[call.kind], (spec,), None, call.name)

    def offer(self, call: Call) -> dict:
        c = self._dv(call)
        if c is None:
            o = self._plain(call).to_json()
            return {"sources": o["sources"], "default": {}}
        o = dv.REGISTRY.offer(c.entry.name)
        if o is None:
            return {"sources": [], "default": {}}
        o = o.to_json()
        return {"sources": o["sources"], "default": o.get("default") or {}}

    # -- resolving (the network thread) -------------------------------------------------
    def resolve(self, view: dict, request: dict | None) -> dict:
        try:
            return dv.resolve(view, request)
        except dv.DataViewError as e:
            raise _bad(e) from None

    # -- computing (the training thread) ------------------------------------------------
    def _program(self, call: Call, resolved: dict):
        """The compiled data view of a subscription group for this call's
        source sizes, or None when it does not fit them (any more)."""
        last = self._last
        if last is not None and last[0] is call and last[1] is resolved:
            return last[2]                 # serves() then apply() for the same group
        p = self._lookup(call, resolved)
        self._last = (call, resolved, p)
        return p

    def _lookup(self, call: Call, resolved: dict):
        e = self._canon.get(id(resolved))
        if e is None or e[0] is not resolved:   # the entry holds resolved: its id stays its own
            if len(self._canon) >= _PROGRAMS:
                self._canon.clear()
            e = self._canon[id(resolved)] = (resolved, _canonical(resolved))
        key = (call.name, self.signature(call), e[1])
        p = self._programs.get(key)
        if p is not None:
            self._programs.move_to_end(key)
            return p if p is not False else None
        c = self._dv(call)
        off = (dv.Offer(c.entry.kind, c.entry.specs()) if c is not None else self._plain(call))
        try:
            again = dv._resolve(off, dv._explicit(resolved), off.kind)
            p = dv.compile_view(off, resolved) if again == resolved else False
        except dv.DataViewError:
            p = False
        self._programs[key] = p
        while len(self._programs) > _PROGRAMS:
            self._programs.popitem(last=False)
        return p if p is not False else None

    def serves(self, call: Call, resolved: dict) -> bool:
        """Whether this call carries the source ``resolved`` reads, with a
        tensor that fits it. A view with several sources gets a call per
        source; a group of another source waits for its own call."""
        c = self._dv(call)
        if c is not None:
            if resolved.get("source") != c.source:
                return False
            x = c.tensor()
            if x is None:
                return False
            p = self._program(call, resolved)
            if p is None:
                return False
            import torch
            return dv.dim_sizes(p.kind, x.shape, x.dtype == torch.uint8) == p.sizes
        return self._program(call, resolved) is not None

    def _scratch_slot(self, name: str, plan) -> RingSlot:
        import torch
        key = (name, plan.key)
        slot = self._scratch.get(key)
        if slot is None:
            dev = plan.device
            payload = torch.empty(plan.shape, dtype=plan.dtype, device=dev)
            stats = torch.zeros(STATS_FLOATS, dtype=torch.float32, device=dev)
            aux = torch.empty(plan.aux_shape, dtype=torch.float32, device=dev) if plan.aux_shape else None
            slot = RingSlot(0, payload, stats, aux, None, stats[0], stats[1], stats[2:8:2], stats[3:8:2])
            if plan.init is not None:
                plan.init(slot)
            self._scratch[key] = slot
        return slot

    def apply(self, call: Call, resolved: dict) -> Reduced:
        p = self._program(call, resolved)
        self._last = None                  # do not keep the call's tensors
        if p is None:                      # NetRing asks serves() first; this is a guard
            raise _bad(dv.DataViewError("the data view does not fit the view's data any more"))
        c = self._dv(call)
        if c is not None:
            y, pre = dv._apply(c.x, p, pre=True)
            plan = dv.payload_plan(p, y, c)
            slot = self._scratch_slot(call.name, plan)
            dv.finish(slot, p, pre, plan)
            return self._reduced(plan, slot, p, slot.payload, slot.aux, slot.stats)
        fast = self._fast(call, p)
        if fast is not None:
            return fast
        return self._apply_plain(call, p)

    def _fast(self, call: Call, p) -> Reduced | None:
        """A view without dims whose plan would only copy the call's tensor (a
        float32 2-D heatmap, float32 (N, 2|3) points): the data view is applied
        to that tensor itself, with no scratch copy. The stats are those of
        _apply_plain: a heatmap reports the range of the whole tensor (one
        aminmax when the data view pools, zooms or tiles it; the transport
        computes the same range on the host when the payload is the whole
        tensor), points those of the points sent (on the host)."""
        import torch
        plan = call.plan
        common = dict(colormap=plan.colormap, flags=snap_flags(plan.flags), vmin=plan.vmin,
                      vmax=plan.vmax, axes=p.axes, tile=p.tile)
        if call.kind == KIND_HEATMAP:
            src = _plain_source(call)
            if src is None:
                return None
            y = dv._apply(src, p)
            stats = None
            whole = y.shape == src.shape and y.stride() == src.stride() and y.data_ptr() == src.data_ptr()
            if not whole and plan.flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX):
                key = (call.name, "range")
                r = self._stats.get(key)
                if r is None or r[0].device != src.device:
                    stats = torch.zeros(2, dtype=torch.float32, device=src.device)
                    r = self._stats[key] = (stats, (stats[0], stats[1]))
                stats = r[0]
                torch.aminmax(src, out=r[1])
            if not y.is_contiguous():            # a strided zoom: MPS copies to the host need it
                y = y.contiguous()
            return Reduced(y, stats=stats, stats_count=2, **common)
        if call.kind == KIND_POINTS:
            fast = _points_source(call)
            if fast is None:
                return None
            z, lab, _, d = fast
            y = dv._apply(z, p)
            if not y.is_contiguous():
                y = y.contiguous()
            if lab is not None and (p.sample is not None or p.row_slice is not None):
                lab = dv.take_rows(lab, p)
            if lab is not None and not lab.is_contiguous():
                lab = lab.contiguous()
            return Reduced(y, aux=lab, stats=None, stats_count=STATS_FLOATS, points_dims=d,
                           label_count=plan.label_count, **common)
        return None

    def _reduced(self, plan, slot, p, data, aux, stats) -> Reduced:
        common = dict(colormap=plan.colormap, flags=snap_flags(plan.flags), vmin=plan.vmin,
                      vmax=plan.vmax, axes=p.axes, tile=p.tile)
        if plan.kind == KIND_HEATMAP or plan.kind == KIND_HISTOGRAM:
            return Reduced(data, stats=stats[:2], stats_count=2, **common)
        if plan.kind == KIND_IMAGE:
            return Reduced(data, **common)
        d = plan.points_dims
        if data.shape[1] != d:
            data = data[:, :d].contiguous()
        return Reduced(data, aux=aux, stats=stats, stats_count=STATS_FLOATS, points_dims=d,
                       label_count=plan.label_count, **common)

    def _base(self, call: Call) -> RingSlot:
        """The plan of a view without dims written once per call into its
        scratch slot: its payload is the view's source."""
        slot = call.cache.get("slot")
        if slot is None:
            slot = self._scratch_slot(call.name, call.plan)
            call.plan.write(slot)
            call.cache["slot"] = slot
        return slot

    def _apply_plain(self, call: Call, p) -> Reduced:
        import torch
        plan = call.plan
        slot = self._base(call)
        if call.kind == KIND_HISTOGRAM:
            return self._reduced(plan, slot, p, slot.payload, None, slot.stats)
        if call.kind != KIND_POINTS:
            # a heatmap reports the range of its whole payload (protocol v2's rule)
            y = dv._apply(slot.payload, p)
            return self._reduced(plan, slot, p, y.contiguous(), None, slot.stats)
        d = plan.points_dims
        src = slot.payload if d == 3 else slot.payload[:, :d]
        y = dv._apply(src, p)
        aux = slot.aux
        if p.sample is None and p.row_slice is None:
            return self._reduced(plan, slot, p, y.contiguous(), aux, slot.stats)
        if aux is not None:
            aux = dv.take_rows(aux, p)
        key = (call.name, plan.key)
        stats = self._stats.get(key)
        if stats is None:
            stats = self._stats[key] = torch.zeros(STATS_FLOATS, dtype=torch.float32,
                                                   device=y.device)
        torch.aminmax(y, dim=0, out=(stats[2:2 + 2 * d:2], stats[3:3 + 2 * d:2]))
        if aux is not None and plan.label_count == 0 and plan.flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX):
            torch.aminmax(aux, out=(stats[0], stats[1]))
        return self._reduced(plan, slot, p, y.contiguous(), aux, stats)


def _plain_source(call: Call):
    """The call's own tensor when the plan would only copy it: a 2-D float32
    heatmap on the plan's device (no cast, no mosaic). None otherwise."""
    if call.args is None or call.plan.key[3] != "plain":
        return None
    t = call.args[0]
    import torch
    if (isinstance(t, torch.Tensor) and t.ndim == 2 and t.dtype == torch.float32
            and t.device == call.plan.device):
        return t.detach()
    return None


def _points_source(call: Call):
    """(z, labels, N, D) straight from the call when they need no cast or move:
    float32 (N, 2|3) points with labels (if any) of N values on the same device."""
    if call.args is None:
        return None
    import torch
    z, labels = call.args[0], call.args[1]
    p = call.plan
    if not (isinstance(z, torch.Tensor) and z.ndim == 2 and z.shape[1] in (2, 3)
            and z.dtype == torch.float32 and z.device == p.device):
        return None
    lab = None
    if labels is not None:
        if not (isinstance(labels, torch.Tensor) and labels.device == z.device
                and labels.numel() == z.shape[0]):
            return None
        lab = labels.detach().reshape(-1)
    return z.detach(), lab, int(z.shape[0]), int(z.shape[1])


def make_engine() -> Engine:
    """The engine a new network route uses."""
    return DataViewEngine()
