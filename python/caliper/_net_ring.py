"""The training-thread side of the network route (``caliper.serve()``).

:class:`NetRing` takes the place of :class:`caliper._ring.SnapshotRing` in the
network route: ``_dispatch`` calls its ``due`` and ``take_call``, ``series``
calls its ``tap``, and ``flush``/``forget``/``close`` work as for the other
routes. Everything here runs on the thread that made the view call and only
QUEUES device work:

* a snapshot of a subscribed view is reduced to each subscription group's data
  view by the engine (caliper/_net_dataview.py) and copied with
  ``copy_(non_blocking=True)`` into pinned host memory. The network thread reads
  the pinned memory only once a ``torch.mps.Event`` recorded after the copy has
  completed, so nothing waits for the GPU. (Reading a device tensor with
  ``.cpu()`` on another thread instead drains torch's queue and stalls
  training: measured at 64 to 70 % of the commit-matched control.)
* THE TICK. Recording an event commits torch's command buffer, and commits
  beyond the commit-matched control's 60 a second cost the loop by themselves
  (events at 300 Hz: 3 %). So the views take their snapshots together, in a
  tick at the session's rate (60 Hz): the first call after the tick is due
  opens it, every due subscription group and watched series called in the same
  step joins it, and ONE event recorded when it closes covers all of their
  copies, which go to the network thread in one batch. A tick closes as soon as
  every due group that was called in the last second has its snapshot (right
  after the step's last view call, so the last step of a run is not held back),
  when a group already served in it is called again (the next step began), at
  the first call after it has been open for TICK_MAX_AGE_S (the calls stopped),
  or at flush and close. A tick has no shorter age limit, so that the view calls
  of one step share it however far apart in the step they are made.
* a series value is retained whether or not anyone subscribes (the protocol's
  backfill rule): floats go to the network thread as they are, tensors are
  gathered (detached) and moved in chunks the same way as snapshots.

There is no ``.item()``, ``.cpu()``, ``.numpy()``, ``.tolist()`` or
``synchronize()`` in this module (python/tests/test_net.py checks it).

The state shared with the network thread (views, groups, the queue of pending
snapshots and series chunks) is guarded by ``NetRing.lock``. A view call with
no viewer connected never gets here: the route's ``attached()`` is False and
``caliper.show`` returns after that check.
"""
from __future__ import annotations

import collections
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import _wire
from ._net_dataview import Call, Reduced, make_engine
from ._ring import DEFAULT_RATE_HZ, KIND_SERIES, rate_interval

OFFER_REFRESH_S = 1.0          # an unsubscribed view's offer is rechecked this often
STAGE_SETS = 3                 # pinned staging sets per group: in flight, pending, free
SERIES_CHUNK = 64              # tensor values gathered before a chunk is moved to the host
SERIES_IDLE_FLUSH_S = 1.0      # chunk cadence of a series nobody watches
SERIES_CAPACITY = 4096         # values kept of one huge tensor call (as caliper.series)
TICK_MAX_AGE_S = 1.0           # a tick still open after this closes at the next call


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def canonical(dv: dict) -> str:
    return json.dumps(dv, sort_keys=True, separators=(",", ":"))


@dataclass(eq=False)
class Stage:
    """Pinned host buffers one snapshot is copied into."""
    key: tuple
    data: Any
    aux: Any
    stats: Any
    busy: bool = False


@dataclass(eq=False)
class Group:
    """Every subscription to one view under one resolved data view; the
    snapshot is computed once per group."""
    view: str
    resolved: dict
    key: str
    subs: dict = field(default_factory=dict)   # (link, sub) -> max_rate
    interval: float = 1.0 / DEFAULT_RATE_HZ
    last: float = -math.inf
    stages: list = field(default_factory=list)

    def update_rate(self) -> None:
        rate = max(self.subs.values()) if self.subs else DEFAULT_RATE_HZ
        self.interval = 1.0 / rate


@dataclass(eq=False)
class ViewState:
    name: str
    id: int
    kind: int
    sig: Any = None
    offer: dict = field(default_factory=dict)
    offer_t: float = -math.inf
    groups: dict = field(default_factory=dict)  # key -> Group
    last: float = -math.inf
    seq: int = 0
    seen: float = -math.inf                     # perf_counter of its last call through due()
    iv: float = 0.0                             # the call's own interval (rate=)

    def entry(self) -> dict:
        e = {"id": self.id, "name": self.name, "kind": int(self.kind),
             "sources": self.offer.get("sources", [])}
        if self.offer.get("default"):
            e["default"] = self.offer["default"]
        return e


@dataclass(eq=False)
class Pending:
    """A snapshot queued on the device, waiting for its event."""
    group: Group
    view: str
    view_id: int
    seq: int
    stamp_ns: int
    stage: Stage
    red: Reduced
    kind: int
    event: Any = None
    mps: bool = False                           # its copies are on torch's MPS stream

    def ready(self) -> bool:
        return self.event is None or self.event.query()


@dataclass(eq=False)
class Chunk:
    """Series values on their way to the host."""
    buf: Any                  # pinned f32 tensor holding n values
    n: int
    stamp_ns: int
    event: Any = None
    mps: bool = False

    def ready(self) -> bool:
        return self.event is None or self.event.query()


@dataclass(eq=False)
class SeriesState:
    name: str
    view: ViewState
    q: collections.deque = field(default_factory=collections.deque)  # floats (v, stamp) and Chunks
    refs: list = field(default_factory=list)     # training thread: values not yet in a chunk
    nref: int = 0
    thread: int = 0                              # the thread that gathered refs (device work
                                                 # happens only there: torch's MPS stream is
                                                 # not safe from two threads)
    flush_t: float = -math.inf
    watched: bool = False                        # some subscription (read on the training thread)
    pool: list = field(default_factory=list)     # pinned chunk buffers the pump has finished with
    held: bool = False                           # a chunk of it waits in the open tick
    scalars: bool = False                        # refs are all 0-d tensors
    seen: float = -math.inf                      # perf_counter of its last tensor value


@dataclass(eq=False)
class Tick:
    """The snapshots and series chunks taken together, sharing one event."""
    t0: float
    thread: int
    served: set = field(default_factory=set)     # Groups and SeriesStates served in it
    pending: list = field(default_factory=list)  # Pendings, no event yet
    chunks: list = field(default_factory=list)   # (SeriesState, Chunk), no event yet
    mps: bool = False


class NetRing:
    """The ring of the network route. See the module docstring."""

    def __init__(self, engine=None, wake=None):
        self.engine = engine or make_engine()
        self.lock = threading.Lock()
        self.default_interval = rate_interval(DEFAULT_RATE_HZ)
        self.counters = collections.Counter()
        self.views: dict[str, ViewState] = {}
        self.series: dict[str, SeriesState] = {}
        self.pending: list[Pending] = []
        self.views_version = 0
        # The earliest perf_counter at which any recently called view may have
        # work: until then NetTransport.attached() is False and a view call
        # returns after that check, as with no viewer. 0 forces calls through
        # (a new view, a subscription change).
        self.next_due = 0.0
        self.tick: Tick | None = None
        self.tick_interval = rate_interval(DEFAULT_RATE_HZ)
        self.tick_next = 0.0                    # perf_counter when the next tick may open
        self.tick_views: set = set()            # the views the last tick served (see due)
        self._next_id = 1
        self._wake = wake or (lambda: None)
        self.closed = False

    # -- the registry (training thread writes, network thread reads) -----------
    def _view(self, name: str, kind: int, now: float) -> ViewState:
        v = self.views.get(name)
        if v is None or v.kind != kind:
            if v is not None:
                self.series.pop(name, None)
            v = ViewState(name, self._next_id, kind)
            self._next_id += 1
            with self.lock:
                self.views[name] = v
                self.views_version += 1
            self.next_due = 0.0
            self._wake()
        return v

    def _register(self, call: Call, now: float) -> ViewState:
        v = self._view(call.name, call.kind, now)
        sig = self.engine.signature(call)
        if sig != v.sig or now - v.offer_t >= OFFER_REFRESH_S:
            v.offer_t = now
            if sig != v.sig:
                offer = self.engine.offer(call)
                with self.lock:
                    v.sig = sig
                    if offer != v.offer:
                        v.offer = offer
                        self.views_version += 1
                self._wake()
        return v

    # -- the ring interface (caliper._dispatch, under caliper's module lock) ----
    def wants(self, name: str, source: str) -> bool:
        """Whether a call carrying source ``source`` of view ``name`` (a view
        with several sources, caliper._dataview) has a use here: a subscription
        selects that source, or the view's offer is to be registered or
        refreshed (no device work)."""
        v = self.views.get(name)
        if v is None or time.perf_counter() - v.offer_t >= OFFER_REFRESH_S or self._stale(v):
            return True
        return any(g.resolved.get("source") == source for g in list(v.groups.values()))

    def _stale(self, v: ViewState) -> bool:
        """The engine says the view's offer changed (a source got its shape)."""
        sig = getattr(self.engine, "view_signature", None)
        if sig is None:
            return False
        s = sig(v.name)
        return s is not None and s != v.sig

    def due(self, name: str, interval: float, now: float) -> bool:
        v = self.views.get(name)
        if v is not None:
            v.seen = now
            v.iv = interval
        if v is None or now - v.offer_t >= OFFER_REFRESH_S or self._stale(v):
            self.next_due = 0.0
            return True                     # register or refresh the offer; no device work
        if not v.groups:
            self.counters["unsubscribed"] += 1
            self._idle_due(now)
            return False
        t = self.tick
        if t is not None and now - t.t0 > TICK_MAX_AGE_S:
            self._close_tick()              # its step's calls stopped
            t = None
        if t is None:
            if now < self.tick_next:
                self.counters["skipped"] += 1
                if name in self.tick_views:     # the next step began: calls wait for the tick
                    self.next_due = self.tick_next
                return False
            if not any(self._group_due(g, v, now, False) for g in list(v.groups.values())):
                self.counters["rate_limited"] += 1
                return False                # a slower view: not due for this tick
        return True                         # take_call opens or joins the tick

    def _idle_due(self, now: float) -> None:
        """next_due while this view has no subscription: the next tick if some
        view has one, else the next offer refresh of a view called lately."""
        if self.tick is not None:
            return
        if any(v.groups for v in list(self.views.values())):
            return                          # the ticks set next_due
        nd = math.inf
        for v in list(self.views.values()):
            if now - v.seen < OFFER_REFRESH_S:
                nd = min(nd, v.offer_t + OFFER_REFRESH_S)
        self.next_due = 0.0 if nd == math.inf else nd

    def changed(self) -> None:
        """A subscription opened or closed (network thread): recheck at once."""
        self.next_due = 0.0

    def _group_due(self, g: Group, v: ViewState, now: float, joining: bool) -> bool:
        """The group may take a snapshot: its subscription's rate and the
        call's own rate allow it. A group joining an open tick, or one at the
        session's rate, has half a tick of slack, so that it keeps to the
        ticks; a slower group that would open a tick itself waits its whole
        interval, so that it never exceeds its rate."""
        iv = max(g.interval, v.iv)
        tick = self.tick_interval
        slack = tick / 2 if joining or iv <= tick * 1.01 else 0.001
        return now - g.last >= iv - slack

    def _complete(self, t: Tick, now: float) -> bool:
        """Every due group of a view called lately, and every watched series
        with values gathered, has been served in tick ``t``."""
        for v in list(self.views.values()):
            if v.kind == KIND_SERIES or not v.groups or now - v.seen > OFFER_REFRESH_S:
                continue                    # a view not called for a second does not hold ticks
            for g in list(v.groups.values()):
                if g not in t.served and self._group_due(g, v, now, True):
                    return False
        for st in list(self.series.values()):
            if st.watched and st.refs and st not in t.served and now - st.seen <= OFFER_REFRESH_S:
                return False
        return True

    def _open_tick(self, now: float) -> Tick:
        t = self.tick = Tick(now, threading.get_ident())
        self.next_due = 0.0                 # calls pass while it is open
        return t

    def _close_tick(self) -> None:
        """One event for everything the tick queued, then one batch to the
        network thread. The event is recorded on the calling thread, which is
        one of the threads that queue torch work (the training thread, or
        autograd's worker thread on MPS, where backward hooks run); never on
        one of caliper's own threads (pump, network, events), which do no
        device work: there the tick stays open for the next view call."""
        t = self.tick
        if t is None or threading.current_thread().name.startswith("caliper."):
            return
        self.tick = None
        self.tick_next = max(t.t0 + self.tick_interval, time.perf_counter() + 1e-4)
        # The calls after the tick in its step still reach due(), which keeps
        # their views counted as called (v.seen), so that a view called after
        # the tick completed is waited for by the next one; the first call of
        # a view the tick served (the next step) makes calls wait.
        self.tick_views = {p.view for p in t.pending}
        self.next_due = 0.0 if self.tick_views else self.tick_next
        ev = None
        if t.mps:
            torch = _torch()
            ev = torch.mps.Event()
            ev.record()
        for p in t.pending:
            if p.mps:
                p.event = ev
        for st, c in t.chunks:
            if c.mps:
                c.event = ev
            st.q.append(c)
            st.held = False
        if t.pending:
            with self.lock:
                self.pending.extend(t.pending)
        self.counters["ticks"] += 1
        self._wake()

    def take(self, name: str, plan, now: float) -> bool:
        return self.take_call(name, plan, now, None)

    def take_call(self, name: str, plan, now: float, args) -> bool:
        """Reduce ``plan`` for every due group of view ``name`` and queue the
        copies to pinned memory. Returns True if a snapshot was queued."""
        call = Call(plan.kind, name, plan, args)
        v = self._register(call, now)
        v.seen = now
        if not v.groups:
            return False
        with self.lock:
            groups = list(v.groups.values())
        t = self.tick
        stamp = time.monotonic_ns()
        taken = 0
        serves = getattr(self.engine, "serves", None)
        for g in groups:
            if serves is not None and not serves(call, g.resolved):
                continue                    # another source's group, or a data view that no longer fits
            if t is not None and g in t.served:
                self._close_tick()          # served in this tick: the next step began
                return bool(taken)
            if not self._group_due(g, v, now, t is not None):
                self.counters["rate_limited"] += 1
                continue
            stage_free = any(not s.busy for s in g.stages) or len(g.stages) < STAGE_SETS
            if not stage_free:
                self.counters["ring_full"] += 1
                continue
            if t is None:
                if now < self.tick_next:
                    return False
                t = self._open_tick(now)
            red = self.engine.apply(call, g.resolved)
            p = self._stage(g, v, red, stamp)
            if p is None:
                self.counters["ring_full"] += 1
                continue
            g.last = now
            t.served.add(g)
            t.pending.append(p)
            t.mps = t.mps or p.mps
            taken += 1
        if taken:
            v.last = now
            self.counters["taken"] += taken
        if t is not None and self._complete(t, now):
            self._close_tick()
        return bool(taken)

    def _stage(self, g: Group, v: ViewState, red: Reduced, stamp: int) -> Pending | None:
        import torch
        data, aux = red.data, red.aux
        key = (tuple(data.shape), data.dtype, None if aux is None else tuple(aux.shape))
        stage = None
        for s in g.stages:
            if not s.busy and s.key == key:
                stage = s
                break
        if stage is None:
            g.stages = [s for s in g.stages if s.busy]
            if len(g.stages) >= STAGE_SETS:
                return None
            pin = data.device.type != "cpu"
            stage = Stage(key, torch.empty(data.shape, dtype=data.dtype, pin_memory=pin),
                          None if aux is None else torch.empty(aux.shape, dtype=torch.float32,
                                                               pin_memory=pin),
                          torch.zeros(_wire.MAX_STATS, dtype=torch.float32, pin_memory=pin))
            g.stages.append(stage)
        nb = data.device.type != "cpu"
        stage.data.copy_(data, non_blocking=nb)
        if aux is not None:
            stage.aux.copy_(aux, non_blocking=nb)
        if red.stats is not None and red.stats_count:
            stage.stats[:red.stats_count].copy_(red.stats[:red.stats_count], non_blocking=nb)
        stage.busy = True
        v.seq += 1
        return Pending(g, v.name, v.id, v.seq, stamp, stage, red, v.kind, None,
                       data.device.type == "mps")

    def append(self, name, sp, now, interval) -> bool:
        return False                         # series come through tap()

    def poll(self) -> int:
        return 0                             # the network thread publishes

    def has_pending(self) -> bool:
        return False

    # -- series (caliper.series, before its route loop) -------------------------
    def tap(self, name: str, value) -> bool:
        """Retain one series call: a number or a tensor of values. Returns True
        while a viewer watches the series."""
        s = self.series.get(name)
        if s is None:
            v = self._view(name, KIND_SERIES, time.perf_counter())
            with self.lock:
                v.offer = {"sources": []}
                v.sig = KIND_SERIES
                s = self.series[name] = SeriesState(name, v)
        cls = value.__class__
        if (cls is float or cls is int) and not s.refs and not s.held:
            s.q.append((float(value), time.monotonic_ns()))
            if s.watched:
                self._wake()
            return s.watched
        torch = _TORCH or _torch()
        scalar = False
        if cls is torch.Tensor and value.ndim == 0:   # the common case: loss.detach()
            t = value.detach() if value.requires_grad else value
            n = 1
            scalar = True
        elif cls is float or cls is int or cls is bool:
            t = float(value)
            n = 1
        else:
            t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            t = t.detach()                       # reshaped when the chunk is made
            n = t.numel()
            if n == 0:
                raise ValueError("series: empty value")
            if n > SERIES_CAPACITY:
                t = t.reshape(-1)[-SERIES_CAPACITY:]
                n = SERIES_CAPACITY
        if not s.refs:
            s.thread = threading.get_ident()
            s.scalars = scalar
        elif not scalar:
            s.scalars = False
        s.refs.append(t)
        s.nref += n
        now = time.perf_counter()
        s.seen = now
        if s.watched:
            if self.tick is None and now < self.tick_next and s.nref < SERIES_CAPACITY:
                return True                  # the common step: until the next tick
            self._tick_series(s, now)
        elif s.nref >= SERIES_CHUNK or now - s.flush_t >= SERIES_IDLE_FLUSH_S:
            self._flush_series(s, now)
        return s.watched

    def _tick_series(self, s: SeriesState, now: float) -> None:
        """A watched series moves its gathered values in the tick, with the
        snapshots; past SERIES_CAPACITY values it moves them on its own."""
        t = self.tick
        if t is not None and (now - t.t0 > TICK_MAX_AGE_S or s in t.served):
            self._close_tick()              # its step's calls stopped, or the next step began
            t = None
        if t is None:
            if now < self.tick_next:
                if s.nref >= SERIES_CAPACITY:
                    self._flush_series(s, now)
                return
            t = self._open_tick(now)
        self._flush_series(s, now, t)
        t.served.add(s)
        if self._complete(t, now):
            self._close_tick()

    def _flush_series(self, s: SeriesState, now: float, tick: Tick | None = None) -> None:
        import torch
        refs, s.refs, n = s.refs, [], s.nref
        s.nref = 0
        s.flush_t = now
        if not refs:
            return
        if s.scalars:                        # the common case: loss.detach() at every step
            try:
                chunk = torch.stack(refs)
            except RuntimeError:             # on several devices
                chunk = None
            if chunk is not None:
                self._move(s, chunk, n, chunk.device, tick)
                return
        dev = next((r.device for r in refs if isinstance(r, torch.Tensor)), torch.device("cpu"))
        parts = []
        floats = []
        for r in refs:
            if isinstance(r, torch.Tensor):
                if floats:
                    parts.append(torch.tensor(floats, dtype=torch.float32).to(dev))
                    floats = []
                r = r.reshape(-1)
                parts.append(r.to(dev) if r.device != dev else r)
            else:
                floats.append(r)
        if floats:
            parts.append(torch.tensor(floats, dtype=torch.float32).to(dev))
        self._move(s, parts[0] if len(parts) == 1 else torch.cat(parts), n, dev, tick)

    def _move(self, s: SeriesState, chunk, n: int, dev, tick: Tick | None = None) -> None:
        """Queue chunk's n values to pinned host memory (a buffer the pump has
        finished with when there is one): into ``tick``, whose event covers it,
        or with an event of its own."""
        import torch
        pin = dev.type != "cpu"
        buf = None
        while s.pool:
            b = s.pool.pop()
            if b.numel() >= n:
                buf = b
                break
        if buf is None:
            buf = torch.empty(max(n, SERIES_CHUNK), dtype=torch.float32, pin_memory=pin)
        buf[:n].copy_(chunk, non_blocking=pin)                # casts to float32 if needed
        mps = dev.type == "mps"
        self.counters["series_chunks"] += 1
        c = Chunk(buf, n, time.monotonic_ns(), None, mps)
        if tick is not None:
            tick.chunks.append((s, c))
            tick.mps = tick.mps or mps
            s.held = True
            return
        if mps:
            c.event = torch.mps.Event()
            c.event.record()
        s.q.append(c)
        if s.watched:
            self._wake()

    # -- lifetime --------------------------------------------------------------------
    def flush(self, name: str | None = None, timeout: float = 5.0) -> int:
        """Move gathered series values to the host and wait (polling) until the
        queued snapshots and chunks have completed on the device. Returns how
        many are still in flight. Not for the training loop."""
        now = time.perf_counter()
        me = threading.get_ident()
        if self.tick is not None:
            self._close_tick()
        for s in list(self.series.values()):
            if (name is None or s.name == name) and s.refs and s.thread == me:
                self._flush_series(s, now)       # on another thread they wait for the next call
        self._wake()
        deadline = time.perf_counter() + timeout
        while True:
            with self.lock:
                left = sum(1 for p in self.pending if not p.ready())
            left += sum(1 for s in self.series.values() for c in list(s.q)
                        if isinstance(c, Chunk) and not c.ready())
            if not left or time.perf_counter() >= deadline:
                return left
            time.sleep(0.001)

    def forget(self, name: str) -> None:
        if self.tick is not None:
            self._close_tick()
        with self.lock:
            if self.views.pop(name, None) is not None:
                self.views_version += 1
            self.series.pop(name, None)
        self.engine_forget(name)
        self._wake()

    def engine_forget(self, name: str) -> None:
        forget = getattr(self.engine, "forget", None)
        if forget is not None:
            forget(name)

    def close(self) -> None:
        self.closed = True

    def views_list(self) -> list:
        return list(self.views)

    def view_kind(self, name: str) -> int:
        v = self.views.get(name)
        return v.kind if v is not None else 0
