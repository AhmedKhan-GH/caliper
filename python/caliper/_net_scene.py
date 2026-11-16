"""Scenes on the network route: the network thread's side of caliper.scene.

Each NetRing (one per ``caliper.serve()`` producer) has one :class:`SceneRoute`
(``route_of(ring)``), which holds

* the scene views in the ring's registry, so that VIEWS lists them (kind 6
  with the key "scene", and its "view_camera" when the script set one), to a
  peer that announced the capability "scenes";
* the subscriptions to each scene, which the training thread reads to decide
  whether a call takes a snapshot (``watched``);
* the items whose device copies are still in flight (``queue``): batches,
  cameras, entity transforms, class tables and clears. The pump finishes them
  in the order of the calls of each scene (``pump``), keeps the newest stamped
  batch of each entity part for a later subscription, applies a clear to what
  it keeps, and offers each to every subscriber of its scene;
* per subscription a :class:`SubQueue`: the scene's state for a new
  subscription (SCENE_CLEAR with reset, the class table, the entity
  transforms, then every camera and batch held, each with ``state`` true),
  and the clears, sent first and in order; then the newest class table, the
  pending entity transforms merged into SCENE_TRANSFORMS (the newest per path,
  the static ones apart), and at most one pending batch per entity part,
  which a newer one replaces, each entity part at the subscription's
  ``max_rate`` (cameras unlimited). A clear drops the pending batches and
  transforms it clears. A batch over one frame travels as a split update, its
  pieces sent one after another before anything else of the subscription.

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
import warnings

from . import _wire
from ._net_ring import ViewState
from ._scene import (KIND_SCENE, PART_ORDER, SCENES, Batch, Classes, Clear, SceneState,
                     Transform)

__all__ = ["route_of", "SceneRoute", "LinkScenes", "SubQueue"]

_routes_lock = threading.Lock()
NATIVE_BACKLOG = 2                       # scene frames a native link may have waiting at once
                                         # (caliper._netio.MAX_SCENE_BYTES is this many frames)
FRAME_ROOM = _wire.MAX_FRAME_BYTES - 64 * 1024   # array bytes a batch sends whole, at most
PIECE_BYTES = _wire.MAX_FRAME_BYTES              # the frame of a piece of a split update, at most


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


def finish(item) -> bool:
    """Read a completed item out of host memory (network thread). False for a
    batch that every receiver would refuse: strip lengths from a device tensor,
    which the call could not read, that do not add up to the vertices or hold a
    length below 1. Such a batch is dropped with a warning (once per entity, by
    the default warning filter)."""
    if isinstance(item, Transform):
        if item.values is None:
            m = item.matrix
            item.values = m.numpy().astype("<f4") if hasattr(m, "numpy") else m
        return True
    if not isinstance(item, Batch):
        return True
    b = item
    if b.arrays is not None:
        return True
    if b.refused:
        return False
    arrays = {name: t.numpy() for name, t in b.tensors.items()}
    for name, dt in b.cast.items():
        arrays[name] = arrays[name].astype(dt)
    sl = arrays.get("strip_lengths")
    if sl is not None and ((sl.size and int(sl.min()) < 1) or
                           int(sl.sum(dtype="int64")) != arrays["positions"].shape[0]):
        b.refused = True
        warnings.warn(f"caliper: scene {b.scene!r}: the strip lengths of the lines at "
                      f"{b.path or 'the root'!r} do not add up to their vertices, or one is "
                      "below 1; such a batch is dropped", stacklevel=2)
        return False
    if b.stats is not None:
        vrange, bounds, dims = b.stats
        if dims == 0:                                 # a depth image: its range in scene units
            b.params["stats"] = vrange.tolist() if vrange is not None else [math.nan, math.nan]
        else:
            stats = vrange.tolist() if vrange is not None else [0.0, 0.0]
            if bounds is not None:
                stats += bounds.tolist()
                if dims == 2:
                    stats += [0.0, 0.0]
            else:
                stats += [math.nan] * 6
            b.params["stats"] = stats
    b.arrays = arrays
    return True


def _key(path: str, part: str):
    return path.encode("utf-8"), PART_ORDER[part]


def _transforms(sub: int, view: str, items, static: bool, state: bool) -> list:
    """SCENE_TRANSFORMS of finished transforms, at most MAX_TRANSFORMS each."""
    import numpy as np
    items = sorted(items, key=lambda t: t.path.encode("utf-8"))
    out = []
    for k in range(0, len(items), _wire.MAX_TRANSFORMS):
        chunk = items[k:k + _wire.MAX_TRANSFORMS]
        out.append(_wire.SceneTransforms(
            sub, view, max(t.seq for t in chunk), max(t.stamp_ns for t in chunk),
            [t.path for t in chunk], np.stack([t.values for t in chunk]).astype("<f4"),
            static=static, state=state))
    return out


class SubQueue:
    """What one subscription to a scene still has to send."""
    __slots__ = ("sub", "view", "interval", "state", "pending", "last", "transforms", "classes",
                 "pieces", "splits")

    def __init__(self, sub: int, view: str, interval: float):
        self.sub = sub
        self.view = view
        self.interval = interval
        self.state: collections.deque = collections.deque()   # sent first, in order, never dropped
        self.pending: dict = {}           # (path, part) -> Batch, the newest wins
        self.last: dict = {}              # (path, part) -> monotonic time of its last send
        self.transforms: dict = {}        # path -> Transform, the newest wins
        self.classes: Classes | None = None
        self.pieces: collections.deque = collections.deque()  # the rest of a split update
        self.splits = 0                   # this subscription's split updates so far

    def offer(self, item) -> bool:
        """Queue an item; True when it replaced a pending one."""
        if isinstance(item, Clear):
            for key in [k for k, b in self.pending.items() if item.removes(k[0], b.static)]:
                del self.pending[key]
            for path in [p for p, t in self.transforms.items() if item.removes(p, t.static)]:
                del self.transforms[path]
            self.state.append(self.clear(item, False))
            return False
        if isinstance(item, Transform):
            replaced = item.path in self.transforms
            self.transforms[item.path] = item
            return replaced
        if isinstance(item, Classes):
            replaced = self.classes is not None
            self.classes = item
            return replaced
        replaced = (item.path, item.part) in self.pending
        self.pending[(item.path, item.part)] = item
        return replaced

    def next_item(self, now: float):
        """(message, None) for the next frame, else (None, seconds until one is due)."""
        if self.pieces:
            return self.pieces.popleft(), None
        if self.state:
            return self.state.popleft(), None
        if self.classes is not None:
            k, self.classes = self.classes, None
            return self.table(k, False), None
        if self.transforms:
            static = any(t.static for t in self.transforms.values())
            group = [t for t in self.transforms.values() if t.static == static]
            group = sorted(group, key=lambda t: t.path.encode("utf-8"))[:_wire.MAX_TRANSFORMS]
            for t in group:
                del self.transforms[t.path]
            return _transforms(self.sub, self.view, group, static, False)[0], None
        wait = None
        for key, b in self.pending.items():
            interval = 0.0 if b.part == _wire.PART_CAMERA else self.interval
            due = self.last.get(key, -math.inf) + interval - now
            if due <= 0:
                del self.pending[key]
                self.last[key] = now
                return self.whole_or_pieces(self.update(b, False)), None
            wait = due if wait is None else min(wait, due)
        return None, wait

    def whole_or_pieces(self, msg: _wire.SceneUpdate):
        """The update, or its first piece when its arrays pass one frame (the
        other pieces wait in ``pieces``)."""
        size = sum(getattr(a, "nbytes", 0) for a in msg.arrays.values())
        if size <= FRAME_ROOM:
            return msg
        self.splits += 1
        pieces = _wire.split_update(msg, self.splits, max_frame=PIECE_BYTES)
        self.pieces.extend(pieces[1:])
        return pieces[0]

    def update(self, b: Batch, state: bool) -> _wire.SceneUpdate:
        return _wire.SceneUpdate(self.sub, self.view, b.seq, b.stamp_ns, b.path, b.part,
                                 b.arrays, b.params, static=b.static, state=state, axes=b.axes)

    def table(self, k: Classes, state: bool) -> _wire.SceneClasses:
        return _wire.SceneClasses(self.sub, self.view, k.seq, k.stamp_ns, k.classes,
                                  static=k.static, state=state)

    def clear(self, c: Clear, state: bool) -> _wire.SceneClear:
        return _wire.SceneClear(self.sub, self.view, c.seq, c.stamp_ns, c.path,
                                recursive=c.recursive, static=c.static, state=state)


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
        self.pending: list = []                       # items in flight, in call order (under lock)
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

    def queue(self, item) -> None:
        with self.lock:
            self.pending.append(item)
        self.counters["queued"] += 1
        self.ring._wake()

    # -- the network thread ----------------------------------------------------------
    def busy(self) -> bool:
        return bool(self.pending)

    def pump(self) -> bool:
        """Finish the items whose copies completed and offer them, in the order
        of the calls of each scene: an item waits for the earlier ones of its
        scene. Returns True while copies are still in flight."""
        if not self.pending:
            self._drop_gone()
            return False
        with self.lock:
            done, left, blocked = [], [], set()
            for item in self.pending:
                if item.scene in blocked or not item.ready():
                    blocked.add(item.scene)
                    left.append(item)
                else:
                    done.append(item)
            self.pending = left
        for item in done:
            if not finish(item):
                self.counters["refused"] += 1
                continue
            st = SCENES.get(item.scene)
            if st is None:
                continue                              # removed meanwhile
            if isinstance(item, Clear):
                h = self.held.get(item.scene, {})
                for key in [k for k in h if item.removes(k[0], False)]:
                    del h[key]
            elif isinstance(item, Batch):
                key = (item.path, item.part)
                if item.part == _wire.PART_CAMERA:
                    if st.cameras.get(item.path) is not item:
                        continue                      # a newer camera replaced it, or a clear
                elif not item.static:
                    h = self.held.setdefault(item.scene, {})
                    old = h.get(key)
                    if old is None or old.seq < item.seq:
                        h[key] = item
                elif st.static.get(key) is not item:
                    continue                          # a newer static batch replaced it, or a clear
            for (link, _sub), q in list(self.subs.get(item.scene, {}).items()):
                if q.offer(item):
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
        late = []
        if st.classes is not None:
            q.state.append(q.table(st.classes, True))
        transforms = list(st.transforms.values())
        ready = [t for t in transforms if t.ready()]
        late += [t for t in transforms if not t.ready()]
        for t in ready:
            finish(t)
        for static in (True, False):
            group = [t for t in ready if t.static == static]
            if group:
                q.state.extend(_transforms(s.sub, s.view, group, static, True))
        items = dict(st.static)
        for key, b in self.held.get(s.view, {}).items():
            if key not in items or items[key].seq < b.seq:
                items[key] = b
        for path, b in list(st.cameras.items()):
            items[(path, _wire.PART_CAMERA)] = b
        for key in sorted(items, key=lambda k: _key(*k)):
            b = items[key]
            if b.arrays is None and b.ready() and not finish(b):
                continue                               # refused: no receiver would take it
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
                if not link.p.nio.send(link.conn, b"".join(bytes(p) for p in parts)):
                    link.stats["scenes_refused"] += 1  # the connection is closing: nothing more
                    break
                link.stats["scenes_sent"] += 1
                room -= 1
            else:
                wait = 0.004 if wait is None else min(wait, 0.004)
        return wait
