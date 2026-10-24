"""The snapshot ring shared by both transports (contract section).

A view's snapshots live in a small ring of slots. A slot holds the payload (the
tensor the viewer draws), an optional aux buffer (per-point labels) and a stats
buffer (GPU-computed vmin/vmax, bbox, histogram range). All three are allocated
by the TRANSPORT as CPU-readable shared memory (shm segment slots for a live
session; caliper-made shared MTLBuffers for the in-process window) and handed to
torch through DLPack, so torch writes them on its own stream and nothing is ever
copied to the host.

The ring (SnapshotRing, below the contract) owns slot choice, the per-view rate
limit, torch.mps.Event recording and the oldest-first query/publish polling; it
knows nothing about where a completed snapshot goes. A Transport knows nothing
about tensors beyond allocating slots and receiving completed snapshots.

The names and field meanings below mirror caliper/live_views.h
(CaliperLiveSnapshot); the transport maps them onto its own carrier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# Kinds (values shared with caliper/live.h CaliperLiveKind).
KIND_HEATMAP = 1
KIND_IMAGE = 2
KIND_POINTS = 3
KIND_SERIES = 4
KIND_HISTOGRAM = 5

# Flags (values shared with caliper/live_views.h CALIPER_LIVE_APPLY_*).
FLAG_RAW = 1 << 0        # heatmap/image: draw 1:1 at the canvas origin (tests)
FLAG_AUTO_VMIN = 1 << 1  # vmin = stats[0]
FLAG_AUTO_VMAX = 1 << 2  # vmax = stats[1]

# Roles of the buffers a transport allocates for one slot.
ROLE_PAYLOAD = "payload"
ROLE_AUX = "aux"
ROLE_STATS = "stats"


@dataclass
class Slot:
    index: int
    payload: Any              # torch.Tensor, contiguous, on the source device
    stats: Any                # torch.Tensor (STATS_FLOATS,) f32, CPU-readable
    aux: Any = None           # torch.Tensor (N,) f32 or None (points labels)
    event: Any = None         # torch.mps.Event for MPS slots, None for CPU


STATS_FLOATS = 8              # vmin, vmax, then kind-specific (points bbox: 6)


@dataclass
class SnapshotMeta:
    """What the transport forwards with a completed slot (the C struct's fields)."""
    kind: int
    slot: int
    seq: int
    stamp_ns: int             # time.monotonic_ns() when the snapshot was taken
    colormap: int = 0
    vmin: float = 0.0
    vmax: float = 1.0
    flags: int = 0            # FLAG_*
    series_total: int = 0     # SERIES: values appended so far
    points_dims: int = 0      # POINTS: 2 or 3
    label_count: int = 0      # POINTS: categorical classes in aux, 0 = continuous
    stats_count: int = 2      # floats of stats that are meaningful


@dataclass
class ViewSpec:
    """A view's slot layout, fixed for the view's lifetime (a change recreates it)."""
    kind: int
    shape: tuple              # payload shape
    dtype: Any                # torch dtype of the payload
    device: Any               # torch.device the producer writes from
    aux_shape: tuple | None = None   # (N,) for points with labels
    slots: int = 3            # SERIES uses 1 (append-only ring, never claimed)


class Transport(Protocol):
    """Where slots live and where completed snapshots go."""

    def attached(self) -> bool:
        """True while a consumer exists (a viewer attached to the session; the
        in-process window open). When False, show() must do no GPU work."""

    def alloc(self, name: str, spec: ViewSpec, slot: int, role: str, shape: tuple, dtype):
        """A contiguous torch tensor over CPU-readable shared memory the consumer
        can map, on spec.device (MPS tensors via a kDLMetal DLPack capsule)."""

    def reading_slot(self, name: str) -> int:
        """The slot the consumer may still be reading (claimed), or -1."""

    def publish(self, name: str, slot: Slot, meta: SnapshotMeta) -> None:
        """Hand over a completed snapshot. Must not block on the GPU."""

    def remove(self, name: str) -> None:
        """The view is gone; release its slots once the consumer is done."""


# ===========================================================================
# Implementation (below the contract): the ring both transports share
# ===========================================================================
import collections
import math
import threading
import time

DEFAULT_RATE_HZ = 60.0

# caliper's module lock: serialises every ring, the publisher thread and the
# transports' teardown (show() may run on any thread).
LOCK = threading.RLock()


def rate_interval(rate) -> float:
    """Seconds between snapshots for a ``rate`` in Hz (0 = every call)."""
    if rate is None:
        raise ValueError("rate_interval(None)")
    rate = float(rate)
    if rate < 0 or math.isnan(rate):
        raise ValueError(f"rate must be >= 0 Hz, got {rate!r}")
    return 1.0 / rate if rate > 0 else 0.0


@dataclass(eq=False)
class RingSlot(Slot):
    """A Slot plus cached views of its stats buffer (the aminmax(out=) targets),
    so a snapshot does not build them per call."""
    lo: Any = None            # stats[0]
    hi: Any = None            # stats[1]
    axis_min: Any = None      # stats[2:8:2]: xmin, ymin, zmin
    axis_max: Any = None      # stats[3:8:2]: xmax, ymax, zmax


@dataclass(eq=False)
class _View:
    spec: ViewSpec
    key: tuple
    slots: list
    written: list
    published: int = -1                         # slot of the last publish
    pending: collections.deque = field(default_factory=collections.deque)  # (slot, meta, event)
    last: float = -math.inf                     # perf_counter of the last snapshot
    seq: int = 0                                # publishes so far (SnapshotMeta.seq)
    total: int = 0                              # SERIES: values appended
    queued_total: int = 0                       # SERIES: series_total of the last snapshot


class SnapshotRing:
    """The snapshot slots of every view one transport carries.

    ``take(name, plan, now)`` writes a :class:`caliper._convert.Plan` into a free
    slot on the tensor's own stream and records a ``torch.mps.Event`` (which also
    commits torch's command buffer); ``poll()`` publishes completed snapshots,
    oldest first, newest completed wins. Nothing here waits for the GPU.

    Slot choice: a slot that is neither the view's last published one (the
    consumer may pick it up next), nor one the consumer reads, nor pending; the
    least recently written one. Claimed slots come from the optional
    ``transport.claimed_slots(name)`` (every claim: a points view can hold two
    for about a frame) when the transport has it, else ``reading_slot(name)``.
    When none is free the snapshot is skipped (``ring_full``, also reported to
    the optional ``transport.count_drop(name)``): the GPU is behind and more
    copies would only slow the loop further.

    A 1-slot view (SERIES) is append-only: ``append()`` writes every call at
    index ``total % capacity`` and snapshots (records an event, publishes
    ``series_total``) at most at the rate, one pending snapshot at a time; its
    slot is never claimed.

    Not thread-safe: the owner serialises every call (caliper's module lock).
    """

    def __init__(self, transport: Transport, rate: float = DEFAULT_RATE_HZ, slots: int = 3,
                 on_pending=None):
        if slots < 3:
            raise ValueError("a ring needs at least 3 slots (published, reading, writing)")
        self.transport = transport
        self.nslots = slots
        self.default_interval = rate_interval(rate)
        self.on_pending = on_pending           # called when a snapshot is left pending
        self._claimed = (getattr(transport, "claimed_slots", None)
                         or getattr(transport, "reading_slots", None))
        self._count_drop = getattr(transport, "count_drop", None)
        self._views: dict[str, _View] = {}
        self._pending: set[str] = set()
        self._retiring: list[_View] = []
        self._wseq = 0
        self.counters = collections.Counter()

    # -- rate --------------------------------------------------------------------
    def interval(self, rate) -> float:
        return self.default_interval if rate is None else rate_interval(rate)

    def due(self, name: str, interval: float, now: float) -> bool:
        v = self._views.get(name)
        if v is None or now - v.last >= interval:
            return True
        self.counters["skipped"] += 1
        return False

    # -- snapshots -------------------------------------------------------------------
    def take(self, name: str, plan, now: float) -> bool:
        """Snapshot ``plan`` into view ``name``. Returns True if a snapshot was
        taken (published now for CPU slots, once its event completed for MPS)."""
        stamp = time.monotonic_ns()
        v = self._views.get(name)
        if v is None or v.key != plan.key:
            v = self._create(name, plan.kind, plan.key, plan.shape, plan.dtype, plan.device,
                             plan.aux_shape, self.nslots, plan.init)
        idx = self._pick(name, v)
        if idx is None:
            self.counters["ring_full"] += 1
            if self._count_drop is not None:
                self._count_drop(name)
            if v.pending:
                self._poll_view(name, v)
            return False
        slot = v.slots[idx]
        plan.write(slot)
        ev = slot.event
        if ev is not None:
            ev.record()
        meta = SnapshotMeta(plan.kind, idx, 0, stamp, plan.colormap, plan.vmin, plan.vmax,
                            plan.flags, 0, plan.points_dims, plan.label_count, plan.stats_count)
        self._wseq += 1
        v.written[idx] = self._wseq
        v.pending.append((idx, meta, ev))
        v.last = now
        self._pending.add(name)
        self.counters["taken"] += 1
        self._poll_view(name, v)
        if v.pending and self.on_pending is not None:
            self.on_pending()
        return True

    def append(self, name: str, sp, now: float, interval: float) -> bool:
        """Append ``sp`` (a caliper._convert.SeriesPlan) to series ``name``; take
        a snapshot when due. Returns True if a snapshot was taken."""
        import torch
        v = self._views.get(name)
        if v is None or v.spec.kind != KIND_SERIES:
            v = self._create_series(name, sp.device)
        elif v.spec.device.type == "cpu" and sp.device.type != "cpu":
            v = self._upgrade_series(name, v, sp.device)
        slot = v.slots[0]
        cap = v.spec.shape[0]
        sp.append(slot.payload, v.total % cap)
        v.total += sp.count
        self.counters["appended"] += sp.count
        if v.pending:
            self._poll_view(name, v)
        if v.pending or now - v.last < interval:
            return False
        self._snapshot_series(name, v)
        v.last = now
        self._poll_view(name, v)
        if v.pending and self.on_pending is not None:
            self.on_pending()
        return True

    def _snapshot_series(self, name: str, v: _View) -> None:
        slot = v.slots[0]
        ev = slot.event
        if ev is not None:
            ev.record()
        meta = SnapshotMeta(KIND_SERIES, 0, 0, time.monotonic_ns(), series_total=v.total,
                            stats_count=0)
        v.queued_total = v.total
        v.pending.append((0, meta, ev))
        self._pending.add(name)
        self.counters["taken"] += 1

    # -- publishing ------------------------------------------------------------------
    def poll(self) -> int:
        """Publish the newest completed snapshot of each view. Never waits.
        Returns how many snapshots are still pending."""
        left = 0
        if self._pending:
            for name in list(self._pending):
                v = self._views.get(name)
                if v is None:
                    self._pending.discard(name)
                    continue
                self._poll_view(name, v)
                left += len(v.pending)
        if self._retiring:
            self._retiring = [v for v in self._retiring if self._drain_retired(v)]
            left += len(self._retiring)
        return left

    def has_pending(self) -> bool:
        return bool(self._pending or self._retiring)

    def flush(self, name: str | None = None, timeout: float = 5.0) -> int:
        """Publish what is pending, polling the events until they completed (or
        ``timeout`` s): for spin()/close() and the synchronous offscreen mode, not
        the training loop. Series values appended since the last snapshot get a
        final snapshot. Returns how many snapshots are still pending."""
        names = [name] if name is not None else list(self._views)
        for n in names:
            v = self._views.get(n)
            if v is not None and v.spec.kind == KIND_SERIES and v.total != v.queued_total \
                    and not v.pending:
                self._snapshot_series(n, v)
        deadline = time.perf_counter() + timeout
        while True:
            if name is None:
                left = self.poll()
            else:
                v = self._views.get(name)
                left = 0
                if v is not None and v.pending:
                    self._poll_view(name, v)
                    left = len(v.pending)
            if not left or time.perf_counter() >= deadline:
                return left
            time.sleep(50e-6)

    def _poll_view(self, name: str, v: _View) -> None:
        done = None
        pend = v.pending
        while pend:
            idx, meta, ev = pend[0]
            if ev is not None and not ev.query():
                break
            pend.popleft()
            if done is not None:
                self.counters["superseded"] += 1      # a newer one completed too
            done = (idx, meta)
        if done is not None:
            idx, meta = done
            v.seq += 1
            meta.seq = v.seq
            v.published = idx
            self.counters["published"] += 1
            self.transport.publish(name, v.slots[idx], meta)
        if not pend:
            self._pending.discard(name)

    # -- views -------------------------------------------------------------------------
    def forget(self, name: str) -> None:
        """Drop view ``name``; the transport is told once (``remove``)."""
        v = self._views.pop(name, None)
        self._pending.discard(name)
        if v is not None:
            self._retire(name, v)

    def close(self) -> None:
        for name in list(self._views):
            self.forget(name)

    def views(self) -> list[str]:
        return list(self._views)

    def view_kind(self, name: str) -> int:
        v = self._views.get(name)
        return v.spec.kind if v is not None else 0

    def _retire(self, name: str, v: _View) -> None:
        # Slots with a pending event stay referenced until it completed: torch
        # may encode with unretained references to their buffers.
        self.transport.remove(name)
        if self._drain_retired(v):
            self._retiring.append(v)

    @staticmethod
    def _drain_retired(v: _View) -> bool:
        while v.pending:
            _, _, ev = v.pending[0]
            if ev is not None and not ev.query():
                return True
            v.pending.popleft()
        return False

    def _create(self, name, kind, key, shape, dtype, device, aux_shape, nslots, init) -> _View:
        """(Re)allocate a view's slots through the transport."""
        import torch
        old = self._views.pop(name, None)
        if old is not None:
            self._pending.discard(name)
            self._retire(name, old)
            self.counters["recreated"] += 1
        spec = ViewSpec(kind, tuple(shape), dtype, device, aux_shape, slots=nslots)
        tr = self.transport
        slots = []
        for i in range(nslots):
            payload = tr.alloc(name, spec, i, ROLE_PAYLOAD, spec.shape, dtype)
            stats = tr.alloc(name, spec, i, ROLE_STATS, (STATS_FLOATS,), torch.float32)
            aux = tr.alloc(name, spec, i, ROLE_AUX, aux_shape, torch.float32) if aux_shape else None
            ev = torch.mps.Event() if device.type == "mps" else None
            s = RingSlot(i, payload, stats, aux, ev, stats[0], stats[1], stats[2:8:2], stats[3:8:2])
            if init is not None:
                init(s)
            slots.append(s)
        v = _View(spec, key, slots, [0] * nslots)
        self._views[name] = v
        self.counters["allocs"] += 1
        return v

    def _create_series(self, name: str, device) -> _View:
        import torch
        from ._convert import SERIES_CAPACITY
        shape = (SERIES_CAPACITY,)
        return self._create(name, KIND_SERIES, (KIND_SERIES, shape, device), shape,
                            torch.float32, device, None, 1, None)

    def _upgrade_series(self, name: str, v: _View, device) -> _View:
        """A CPU series that receives device values moves to the device (one
        host-to-device copy of its history) instead of reading values back."""
        old_payload, total, seq, last = v.slots[0].payload, v.total, v.seq, v.last
        nv = self._create_series(name, device)
        nv.slots[0].payload.copy_(old_payload)
        nv.total = nv.queued_total = total
        nv.seq, nv.last = seq, last
        return nv

    def _pick(self, name: str, v: _View):
        busy = {v.published}
        rs = self._claimed
        if rs is not None:
            busy.update(rs(name))
        else:
            busy.add(self.transport.reading_slot(name))
        for idx, _, _ in v.pending:
            busy.add(idx)
        best = None
        written = v.written
        for i in range(len(v.slots)):
            if i not in busy and (best is None or written[i] < written[best]):
                best = i
        return best


class Publisher:
    """A daemon thread that publishes completed snapshots while the producer is
    busy elsewhere (a long step, a sleep, the end of an epoch), so a snapshot
    does not wait for the next show() call. It wakes only while snapshots are
    pending and polls every ``interval`` s under the owner's lock."""

    def __init__(self, lock, interval: float = 0.002):
        self.lock = lock
        self.interval = interval
        self.rings: list[SnapshotRing] = []
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = False

    def kick(self) -> None:
        if not self._wake.is_set():
            self._wake.set()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caliper.publish",
                                                daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait()
            if self._stop:
                break
            with self.lock:
                left = 0
                for r in self.rings:
                    try:
                        left += r.poll()
                    except Exception:      # a closed transport; its owner reports it
                        pass
                if not left:
                    self._wake.clear()
            if left:
                time.sleep(self.interval)

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
