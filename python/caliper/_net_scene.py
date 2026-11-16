"""Scenes on the network route: the network thread's side of caliper.scene.

Each NetRing (one per ``caliper.serve()`` producer) has one :class:`SceneRoute`
(``route_of(ring)``), which holds

* the scene views in the ring's registry, so that VIEWS lists them (kind 6
  with the key "scene"), to a peer that announced the capability "scenes";
* the subscriptions to each scene, which the training thread reads to decide
  whether a call takes a snapshot (``watched``);
* the batches whose device copies are still in flight (``queue``), which the
  pump finishes once their event has completed (``pump``), keeping the newest
  stamped batch of each entity part for a later subscription, and offers to
  every subscriber of their scene;
* per subscription a :class:`SubQueue`: the scene's state for a new
  subscription (SCENE_CLEAR with reset, then every batch held, each with
  ``state`` true), sent first and in order, then at most one pending batch
  per entity part, which a newer one replaces, each entity part at the
  subscription's ``max_rate``.

The Python sender asks a link's :class:`LinkScenes` for its next frame
between series and snapshots; the native sender's links get their frames
from ``pump_native`` through ``send()``, one frame per link at a time.
Nothing here touches a device: the copies were queued on the training thread.
"""
from __future__ import annotations

import collections
import math
import threading
import time

from . import _wire
from ._net_ring import ViewState
from ._scene import KIND_SCENE, PART_ORDER, SCENES, Batch, SceneState

__all__ = ["route_of", "SceneRoute", "LinkScenes", "SubQueue"]

_routes_lock = threading.Lock()
NATIVE_BACKLOG = 2                       # scene frames a native link may have waiting at once


def route_of(ring) -> "SceneRoute":
    """The SceneRoute of a NetRing, made on first use."""
    sr = ring.__dict__.get("scenes")
    if sr is None:
        with _routes_lock:
            sr = ring.__dict__.get("scenes")
            if sr is None:
                sr = ring.__dict__["scenes"] = SceneRoute(ring)
    return sr


class SceneViewState(ViewState):
    """A scene in the ring's registry: its VIEWS entry reads the scene."""

    def __init__(self, name: str, view_id: int, scene: SceneState):
        super().__init__(name, view_id, KIND_SCENE)
        self.scene = scene
        self.version = scene.version
        self.offer = {"sources": []}

    def entry(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": KIND_SCENE, "sources": [],
                "scene": self.scene.info()}


def finish(b: Batch) -> None:
    """Read a completed batch out of host memory (network thread)."""
    if b.arrays is not None:
        return
    vrange, bounds, dims = b.stats
    stats = vrange.tolist() if vrange is not None else [0.0, 0.0]
    if bounds is not None:
        stats += bounds.tolist()
        if dims == 2:
            stats += [0.0, 0.0]
    else:
        stats += [math.nan] * 6
    b.params["stats"] = stats
    b.arrays = {name: t.numpy() for name, t in b.tensors.items()}


def _key(path: str, part: str):
    return path.encode("utf-8"), PART_ORDER[part]


class SubQueue:
    """What one subscription to a scene still has to send."""
    __slots__ = ("sub", "view", "interval", "state", "pending", "last")

    def __init__(self, sub: int, view: str, interval: float):
        self.sub = sub
        self.view = view
        self.interval = interval
        self.state: collections.deque = collections.deque()   # sent first, in order, never dropped
        self.pending: dict = {}           # (path, part) -> Batch, the newest wins
        self.last: dict = {}              # (path, part) -> monotonic time of its last send

    def offer(self, b: Batch) -> bool:
        replaced = (b.path, b.part) in self.pending
        self.pending[(b.path, b.part)] = b
        return replaced

    def next_item(self, now: float):
        """(message, None) for the next frame, else (None, seconds until one is due)."""
        if self.state:
            return self.state.popleft(), None
        wait = None
        for key, b in self.pending.items():
            due = self.last.get(key, -math.inf) + self.interval - now
            if due <= 0:
                del self.pending[key]
                self.last[key] = now
                return self.update(b, False), None
            wait = due if wait is None else min(wait, due)
        return None, wait

    def update(self, b: Batch, state: bool) -> _wire.SceneUpdate:
        return _wire.SceneUpdate(self.sub, self.view, b.seq, b.stamp_ns, b.path, b.part,
                                 b.arrays, b.params, static=b.static, state=state, axes=b.axes)


class LinkScenes:
    """The scene subscriptions of one link (network thread)."""
    __slots__ = ("subs", "turn")

    def __init__(self):
        self.subs: dict[int, SubQueue] = {}
        self.turn = 0

    def next_frame(self, now: float):
        """(frame, None), or (None, seconds until a frame is due, or None)."""
        if not self.subs:
            return None, None
        qs = list(self.subs.values())
        self.turn = (self.turn + 1) % len(qs)
        wait = None
        for q in qs[self.turn:] + qs[:self.turn]:      # subscriptions take turns
            msg, w = q.next_item(now)
            if msg is not None:
                return _wire.encode_parts(msg), None
            if w is not None:
                wait = w if wait is None else min(wait, w)
        return None, wait


class SceneRoute:
    """The scenes of one network route. See the module docstring."""

    def __init__(self, ring):
        self.ring = ring
        self.lock = threading.Lock()
        self.watch: dict[str, int] = {}               # scene -> open subscriptions
        self.subs: dict[str, dict] = {}               # scene -> {(link, sub): SubQueue}
        self.pending: list[Batch] = []                # copies in flight (under lock)
        self.held: dict[str, dict] = {}               # scene -> {(path, part): newest stamped Batch}
        self.counters = collections.Counter()
        self.native_wait = None                       # native sender: a scene frame is waiting

    # -- the training thread ---------------------------------------------------------
    def ensure(self, st: SceneState) -> None:
        """List the scene in this route's VIEWS, or announce a change of it."""
        ring = self.ring
        v = ring.views.get(st.name)
        if v is not None and getattr(v, "scene", None) is st:
            if v.version != st.version:
                v.version = st.version
                with ring.lock:
                    ring.views_version += 1
                ring._wake()
            return
        with ring.lock:
            v = SceneViewState(st.name, ring._next_id, st)
            ring._next_id += 1
            ring.views[st.name] = v
            ring.views_version += 1
        ring._wake()

    def watched(self, name: str) -> bool:
        return name in self.watch

    def queue(self, b: Batch) -> None:
        with self.lock:
            self.pending.append(b)
        self.counters["queued"] += 1
        self.ring._wake()

    # -- the network thread ----------------------------------------------------------
    def busy(self) -> bool:
        return bool(self.pending)

    def pump(self) -> bool:
        """Finish the batches whose copies completed and offer them. Returns
        True while copies are still in flight."""
        if not self.pending:
            self._drop_gone()
            return False
        with self.lock:
            done, left = [], []
            for b in self.pending:
                (done if b.ready() else left).append(b)
            self.pending = left
        for b in done:
            finish(b)
            st = SCENES.get(b.scene)
            if st is None:
                continue                              # removed meanwhile
            if not b.static:
                h = self.held.setdefault(b.scene, {})
                old = h.get((b.path, b.part))
                if old is None or old.seq < b.seq:
                    h[(b.path, b.part)] = b
            elif st.static.get((b.path, b.part)) is not b:
                continue                              # a newer static batch replaced it
            for (link, _sub), q in list(self.subs.get(b.scene, {}).items()):
                if q.offer(b):
                    link.stats["scene_superseded"] += 1
                link.wake.set()
            self.counters["published"] += 1
        self._drop_gone()
        return bool(left)

    def _drop_gone(self) -> None:
        for name in [n for n in self.held if n not in SCENES]:
            del self.held[name]

    def join(self, link, s) -> None:
        """A subscription to a scene opened: queue the scene's state for it."""
        st = SCENES.get(s.view)
        if st is None:
            return
        q = SubQueue(s.sub, s.view, s.interval)
        q.state.append(_wire.SceneClear(s.sub, s.view, st.seq, time.monotonic_ns(), "",
                                        reset=True, state=True))
        items = dict(st.static)
        for key, b in self.held.get(s.view, {}).items():
            if key not in items or items[key].seq < b.seq:
                items[key] = b
        late = []
        for key in sorted(items, key=lambda k: _key(*k)):
            b = items[key]
            if b.arrays is None and b.ready():
                finish(b)
            if b.arrays is not None:
                q.state.append(q.update(b, True))
            else:
                late.append(b)                         # the pump offers it once its copy is done
        if late:
            with self.lock:
                known = set(map(id, self.pending))
                self.pending.extend(b for b in late if id(b) not in known)
        scenes = link.scenes
        if scenes is None:
            scenes = link.scenes = LinkScenes()
        scenes.subs[s.sub] = q
        self.subs.setdefault(s.view, {})[(link, s.sub)] = q
        self.watch[s.view] = self.watch.get(s.view, 0) + 1
        self.counters["joined"] += 1
        link.wake.set()
        self.ring._wake()

    def leave(self, link, s) -> None:
        subs = self.subs.get(s.view)
        if subs is not None and subs.pop((link, s.sub), None) is not None:
            n = self.watch.get(s.view, 1) - 1
            if n > 0:
                self.watch[s.view] = n
            else:
                self.watch.pop(s.view, None)
            if not subs:
                del self.subs[s.view]
        if link.scenes is not None:
            link.scenes.subs.pop(s.sub, None)

    # -- the native sender ---------------------------------------------------------------
    def pump_native(self, links, now: float) -> float | None:
        """Hand each native link its next scene frames through ``send()``,
        at most NATIVE_BACKLOG waiting at once. Returns the seconds until a
        rate-limited frame is due, or None."""
        wait = None
        for link in links:
            sc = link.scenes
            if sc is None or not sc.subs:
                continue
            room = NATIVE_BACKLOG - link.p.nio.queued(link.conn)
            while room > 0:
                frame, w = sc.next_frame(now)
                if frame is None:
                    if w is not None:
                        wait = w if wait is None else min(wait, w)
                    break
                parts, _ = frame
                link.p.nio.send(link.conn, b"".join(bytes(p) for p in parts))
                link.stats["scenes_sent"] += 1
                room -= 1
            else:
                wait = 0.004 if wait is None else min(wait, 0.004)
        return wait
