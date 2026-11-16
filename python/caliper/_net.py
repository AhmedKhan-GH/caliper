"""The network route: ``caliper.serve()`` and wire protocol v3 in the producer.

``caliper.serve(port=0)`` starts a WebSocket server on 127.0.0.1 inside the
training process, on its own thread with its own asyncio loop. Viewers (the
browser page this server also serves, the notebook widget, the native app)
connect to ``ws://127.0.0.1:<port>/v3/stream?token=<token>`` and speak the
protocol of include/caliper/live_net.h and docs/references/wire-protocol.md,
through the codec caliper/_wire.py.

The pieces, from the training loop outwards:

* :class:`caliper._net_ring.NetRing` (training thread): reduces each snapshot
  of a subscribed view on its device, queues a non-blocking copy into pinned
  host memory and records a ``torch.mps.Event``. It never waits for the GPU.
* :class:`NetTransport`: the route's ``Transport`` (caliper/_ring.py).
  ``attached()`` is True while a viewer is attached; with none, a view call
  returns after that check, as with the other routes.
* The pump (network thread) publishes what completed: it reads the pinned
  memory only after the event, hands snapshots to the links of the group's
  subscribers (latest wins) and appends series values to the retained history.
* :class:`Link` (network thread): the protocol of one connection after the
  handshake: subscriptions, VIEW_REPLY, EVENT, PING, and the order in which
  frames leave (control first, then series, then snapshots, each subscription
  at its own rate). It knows nothing about sockets: ``handle(frame)`` takes
  bytes in and ``next_frame()`` gives bytes out, so an outbound producer
  connection to a hub can reuse it.
* :class:`_Listener`: the websockets server: the upgrade checks (path, Host,
  Origin, token), the viewer page and its files over plain HTTP, HELLO and
  WELCOME, and one reader and one writer task per connection.
* :class:`NetServer`: what ``caliper.serve()`` returns.
"""
from __future__ import annotations

import asyncio
import collections
import hmac
import itertools
import json
import logging
import os
import queue
import re
import secrets
import struct
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import _wire
from ._net_ring import Group, NetRing, canonical
from ._ring import KIND_HEATMAP, KIND_POINTS, KIND_SERIES, SnapshotMeta, Slot

__all__ = ["serve", "current_server", "NetServer", "NetTransport", "Link"]

SERIES_RETAIN = 1 << 20            # values of each series the producer keeps (>= SERIES_RETAIN_MIN)
MAX_CONNECTIONS = 64
WRITE_LIMIT = 1 << 16              # the socket buffer a send waits to drain below
POLL_S = 0.004                     # the pump's event poll while snapshots are in flight
                                   # (they spend ~15 ms in torch's queue; every wake of
                                   # this thread costs the training loop GIL time)
SCRIPT_MAX = 127

_log = logging.getLogger("caliper.net")
_log.addHandler(logging.NullHandler())
_log.propagate = False


class _RedactToken(logging.Filter):
    """The websockets server logs request paths, which carry the session token:
    a handler attached to "caliper.net" sees them with the token as ``...``."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        if record.exc_info and record.exc_info[1] is not None:
            text = logging.Formatter().formatException(record.exc_info)
            if redact(text) != text:
                record.msg = f"{record.msg}\n{redact(text)}"
                record.exc_info = record.exc_text = None
        return True


_log.addFilter(_RedactToken())

_ORIGIN_LOOPBACK = re.compile(r"(?i)^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$")
_ORIGIN_VSCODE = re.compile(r"(?i)^vscode-webview://[^/\s?#@]+$")
_NOT_BUILT = (b"<!doctype html><meta charset=utf-8><title>Caliper</title>"
              b"<p>The Caliper viewer assets are not built.</p>\n")


# The control frames waiting to be sent on one connection (as caliper._netio).
MAX_CONTROL_FRAMES = 65536
MAX_CONTROL_BYTES = 16 << 20
CLOSE_OVERFLOW = 1008                # as the hub closes an overflowing send queue


class _Fatal(Exception):
    """Close the connection with ``code`` (after an optional fatal ERROR)."""

    def __init__(self, code: int, reason: str, error: str | None = None):
        super().__init__(reason)
        self.code, self.reason, self.error = code, reason, error


def redact(text: str, token: str | None = None) -> str:
    """``text`` (a link, or a message that may quote one) with the value of
    every ``token=`` query parameter, and ``token`` itself when given, written
    as ``...``: a repr, an error or a log line never repeats the session
    token, which only the link that serve() prints carries."""
    out = re.sub(r"(?<=[?&]" + _wire.TOKEN_PARAM + r"=)[^&#\s'\"]*", "...", str(text))
    if token:
        out = out.replace(token, "...")
    return out


def origin_ok(origin: str | None) -> bool:
    """The protocol's Origin rule (docs/references/wire-protocol.md)."""
    if origin is None:
        return True
    return (origin == "null" or bool(_ORIGIN_LOOPBACK.match(origin))
            or bool(_ORIGIN_VSCODE.match(origin)))


def host_ok(host: str | None, port: int) -> bool:
    """The protocol's Host rule: 127.0.0.1:<port> or localhost:<port>."""
    if not host:
        return False
    h = host.lower()
    ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if port == 80:
        ok |= {"127.0.0.1", "localhost"}
    return h in ok


def _script_name() -> str:
    a = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if not a or a == "-c":
        return "python"
    return os.path.basename(a)[:SCRIPT_MAX] or "python"


# ===========================================================================
# what a viewer receives
# ===========================================================================
@dataclass(eq=False)
class SnapData:
    """One completed snapshot, copied out of pinned memory (shared by every
    link of its group; each link encodes its own frame with its ``sub``)."""
    view: str
    view_id: int
    seq: int
    stamp_ns: int
    kind: int
    resolved: dict
    data: Any                 # numpy
    aux: Any                  # numpy or None
    stats: list
    red: Any                  # the Reduced (colormap, flags, range, axes, points fields)


class _History:
    """The retained values of one series (network thread only)."""

    def __init__(self):
        import numpy as np
        self._np = np
        self.buf = np.empty(4096, dtype=np.float32)
        self.total = 0                       # values appended so far
        self.stamp_ns = 0

    def first(self) -> int:
        return max(0, self.total - self.buf.shape[0])

    def append(self, vals, stamp_ns: int) -> None:
        np = self._np
        n = len(vals)
        if n == 0:
            return
        cap = self.buf.shape[0]
        if self.total + n > cap and cap < SERIES_RETAIN:
            new = cap
            while new < self.total + n and new < SERIES_RETAIN:
                new *= 2
            nb = np.empty(new, dtype=np.float32)
            nb[:self.total] = self.buf[:self.total]   # not wrapped yet: cap < SERIES_RETAIN
            self.buf = nb
            cap = new
        if n > cap:
            vals = vals[-cap:]
            self.total += n - cap
            n = cap
        i = self.total % cap
        k = min(n, cap - i)
        self.buf[i:i + k] = vals[:k]
        if k < n:
            self.buf[:n - k] = vals[k:]
        self.total += n
        self.stamp_ns = stamp_ns

    def values(self, start: int, stop: int):
        """Values with absolute indices [start, stop), start >= first()."""
        np = self._np
        cap = self.buf.shape[0]
        a, b = start % cap, stop - start
        if a + b <= cap:
            return self.buf[a:a + b]
        return np.concatenate((self.buf[a:], self.buf[:b - (cap - a)]))


@dataclass(eq=False)
class Sub:
    sub: int
    view: str
    view_id: int
    kind: int
    resolved: dict
    interval: float
    max_rate: float
    group: Group | None = None
    last_sent: float = -float("inf")
    snap: SnapData | None = None             # the pending snapshot (latest wins)
    series: Any = None                       # SeriesState for a series subscription
    next: int = 0                            # series: the next absolute index to send


class _Wake:
    """``set()`` and ``await wait(timeout)`` like an asyncio.Event, without the
    task that ``asyncio.wait_for`` creates on every wait (each costs the
    training loop GIL time). Network thread only."""
    __slots__ = ("_flag", "_fut")

    def __init__(self):
        self._flag = False
        self._fut = None

    def set(self) -> None:
        self._flag = True
        f = self._fut
        if f is not None and not f.done():
            f.set_result(None)

    def clear(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    async def wait(self, timeout: float | None = None) -> None:
        if self._flag:
            return
        loop = asyncio.get_running_loop()
        f = self._fut = loop.create_future()
        h = loop.call_later(max(timeout, 0.0), _resolve, f) if timeout is not None else None
        try:
            await f
        finally:
            self._fut = None
            if h is not None:
                h.cancel()


def _resolve(f) -> None:
    if not f.done():
        f.set_result(None)


# ===========================================================================
# one connection's protocol (no sockets)
# ===========================================================================
_VIEWS = object()                            # control-queue marker: send the current view list
_link_ids = itertools.count(1)


class Link:
    """The producer's side of one connection after the handshake.

    ``handle(frame)`` processes one received frame and may raise
    :class:`_Fatal`; ``next_frame(now)`` returns ``(frame, None)`` for the next
    frame to send, either bytes or (for snapshots and series) the pair
    ``(buffers, frame_len)`` of ``_wire.encode_parts``, or ``(None, wait)``
    with the seconds until a rate-limited frame is due (None: nothing pending). ``wake`` is set when something new
    may be sendable. Network thread only."""
    caps: frozenset = frozenset()            # 3.1: "controls", "selection" if the peer announced them
    may_control = False                      # 3.1: a CONTROL from it is accepted
    scenes = None                            # 3.2 draft: its scene subscriptions (caliper._net_scene)

    def __init__(self, producer: "_Producer", peer: str = ""):
        self.p = producer
        self.id = next(_link_ids)
        self.peer = peer
        self.control: collections.deque = collections.deque()
        self.control_bytes = 0                   # the bytes of the frames in control
        self.overflow = False                    # past the limits below: closed with 1008
        self.pong: int | None = None             # the PONG to send, to the latest PING only
        self.subs: dict[int, Sub] = {}
        self.attached = False
        self.follows_default = True
        self.wake = _Wake()
        self.closed = False
        self.stats = collections.Counter()
        self.software: dict | None = None

    # -- queueing -------------------------------------------------------------
    def queue(self, msg) -> None:
        self.push(_wire.encode(msg))

    def push(self, frame) -> bool:
        """Queue an encoded control frame (from any thread). A peer that keeps
        sending while it reads nothing fills the queue: past MAX_CONTROL_FRAMES
        frames or MAX_CONTROL_BYTES bytes it is dropped and the connection is
        closed with 1008, as the hub closes an overflowing send queue."""
        if self.overflow:
            return False
        if len(self.control) >= MAX_CONTROL_FRAMES or self.control_bytes + len(frame) > MAX_CONTROL_BYTES:
            self.overflow = True
            self.control.clear()
            self.control_bytes = 0
            self.p.counters["overflow_closed"] += 1
            self.wake.set()
            return False
        self.control.append(frame)
        self.control_bytes += len(frame)
        self.wake.set()
        return True

    def queue_views(self) -> None:
        if _VIEWS not in self.control and not self.overflow:
            self.control.append(_VIEWS)
        self.wake.set()

    def start_upstream(self) -> None:
        """After a hub's WELCOME on an outbound producer connection (not built
        yet): the hub is this session's one viewer, so the link is attached and
        sends VIEWS; everything else is the same as for a viewer."""
        self.follows_default = False
        if not self.attached:
            self.attached = True
            self.p.attached_changed(+1)
        self.queue_views()

    def start(self) -> None:
        """After WELCOME: the session list and the server's default attachment."""
        self.queue(self.p.sessions_msg())
        if self.follows_default:
            self.attach(self.p.identity["id"])

    def attach(self, session: str | None) -> None:
        if session is not None and session != self.p.identity["id"]:
            self.queue(_wire.Error(_wire.ERR_UNKNOWN_SESSION, f"no session {session!r}",
                                   in_reply_to=int(_wire.MsgType.ATTACH), session=session))
            return
        if self.attached and session is None:
            self.drop_all()
        was = self.attached
        self.attached = session is not None
        if self.attached != was:
            self.p.attached_changed(+1 if self.attached else -1)
        self.queue_views()
        if self.attached and self.p.controls is not None:
            self.p.controls.on_attach(self)

    # -- incoming -------------------------------------------------------------
    def handle(self, frame) -> None:
        self.stats["frames_in"] += 1
        try:
            msg = _wire.decode(frame)
        except _wire.WireError as e:
            raise _Fatal(e.close_code, e.reason, str(e)) from None
        except _wire.MessageError as e:
            self.queue(_wire.Error(e.code, e.message))
            return
        t = type(msg)
        if t is _wire.Ping:
            self.pong = msg.nonce                # a PING flood leaves one PONG, to the latest
            self.wake.set()
        elif t is _wire.Pong:
            pass
        elif t is _wire.Subscribe:
            self.subscribe(msg)
        elif t is _wire.Unsubscribe:
            self.drop(msg.sub)
        elif t is _wire.ViewRequest:
            self.view_request(msg)
        elif t is _wire.Event:
            self.p.event(self, msg)
        elif t is _wire.Attach:
            self.follows_default = False
            self.attach(msg.session)
        elif t is _wire.Control and self.p.controls is not None:
            self.p.controls.on_control(self, msg)
        elif t is _wire.Bye:
            raise _Fatal(_wire.CLOSE_NORMAL, "bye")
        elif t is _wire.Error:
            self.stats["errors_in"] += 1
            if msg.fatal:
                raise _Fatal(_wire.CLOSE_NORMAL, f"peer error {msg.code}")
        elif t is _wire.Unknown:
            self.stats["unknown_in"] += 1
        else:                                # a known message in the wrong direction
            raise _Fatal(_wire.CLOSE_PROTOCOL_ERROR, f"unexpected {t.__name__.lower()}",
                         f"a producer does not accept {t.__name__.lower()}")

    def _reply_error(self, msg, code: str, text: str) -> None:
        self.queue(_wire.ViewReply(msg.request_id, msg.sub, msg.view,
                                   error={"code": code, "message": text}))

    def subscribe(self, msg: _wire.Subscribe) -> None:
        if not self.attached:
            self._reply_error(msg, _wire.ERR_NOT_ATTACHED, "attach to a session first")
            return
        if msg.sub not in self.subs and len(self.subs) >= _wire.MAX_SUBSCRIPTIONS:
            self._reply_error(msg, _wire.ERR_LIMIT,
                              f"at most {_wire.MAX_SUBSCRIPTIONS} subscriptions per connection")
            return
        self.drop(msg.sub)
        try:
            v, resolved = self.p.resolve(msg.view, msg.data_view, self.caps)
        except _wire.MessageError as e:
            self._reply_error(msg, e.code, e.message)
            return
        rate = min(float(msg.max_rate or _wire.DEFAULT_RATE_HZ), float(_wire.LIMIT_RATE_HZ))
        s = Sub(msg.sub, v.name, v.id, v.kind, resolved, 1.0 / rate, rate)
        self.subs[msg.sub] = s
        self.queue(_wire.ViewReply(msg.request_id, msg.sub, v.name, data_view=resolved))
        self.p.join(self, s)

    def view_request(self, msg: _wire.ViewRequest) -> None:
        s = self.subs.get(msg.sub)
        if s is None:
            self._reply_error(msg, _wire.ERR_UNKNOWN_SUBSCRIPTION, f"no subscription {msg.sub}")
            return
        if msg.view != s.view:
            self._reply_error(msg, _wire.ERR_BAD_REQUEST,
                              f"subscription {msg.sub} is to view {s.view!r}")
            return
        try:
            v, resolved = self.p.resolve(s.view, msg.data_view)
        except _wire.MessageError as e:
            self._reply_error(msg, e.code, e.message)
            return
        self.p.leave(self, s)
        s.resolved = resolved
        s.view_id = v.id
        self.queue(_wire.ViewReply(msg.request_id, msg.sub, s.view, data_view=resolved))
        self.p.join(self, s)

    def drop(self, sub: int) -> None:
        s = self.subs.pop(sub, None)
        if s is not None:
            self.p.leave(self, s)

    def drop_all(self) -> None:
        for sub in list(self.subs):
            self.drop(sub)

    # -- from the pump ----------------------------------------------------------
    def offer_snapshot(self, sub: int, snap: SnapData) -> None:
        s = self.subs.get(sub)
        if s is None:
            return
        if s.snap is not None:
            self.stats["superseded"] += 1
        s.snap = snap
        self.wake.set()

    def views_changed(self, views: dict) -> None:
        """The view list changed: end subscriptions whose view went away or was
        re-created, and send the new list."""
        for sub, s in list(self.subs.items()):
            v = views.get(s.view)
            if v is None or v.id != s.view_id:
                self.drop(sub)
        if self.attached:
            self.queue_views()

    # -- outgoing ---------------------------------------------------------------
    def next_frame(self, now: float):
        if self.overflow:
            raise _Fatal(CLOSE_OVERFLOW, "control queue full")
        if self.pong is not None:
            nonce, self.pong = self.pong, None
            return _wire.encode(_wire.Pong(nonce)), None
        if self.control:
            item = self.control.popleft()
            if item is _VIEWS:
                msg = self.p.views_msg(self) if self.attached else _wire.Views([], None)
                return _wire.encode(msg), None
            self.control_bytes -= len(item)
            return item, None
        wait = None
        # series before snapshots; every subscription at its own rate
        for s in self.subs.values():
            if s.series is None:
                continue
            h = self.p.history(s.series)
            if h is None or s.next >= h.total:
                continue
            due = s.last_sent + s.interval - now
            if due > 0:
                wait = due if wait is None else min(wait, due)
                continue
            start = max(s.next, h.first())
            stop = min(h.total, start + _wire.SERIES_MAX_VALUES)
            vals = h.values(start, stop)
            s.next = stop
            s.last_sent = now
            self.stats["series_sent"] += 1
            self.stats["series_values"] += stop - start
            return _wire.encode_parts(_wire.Series(s.sub, s.view, start, h.total, h.stamp_ns,
                                                   vals)), None
        if self.scenes is not None:          # 3.2 draft: scene messages after series
            frame, due = self.scenes.next_frame(now)
            if frame is not None:
                self.stats["scenes_sent"] += 1
                return frame, None
            if due is not None:
                wait = due if wait is None else min(wait, due)
        best = None
        for s in self.subs.values():
            if s.snap is None:
                continue
            due = s.last_sent + s.interval - now
            if due > 0:
                wait = due if wait is None else min(wait, due)
                continue
            if best is None or s.last_sent < best.last_sent:
                best = s
        if best is not None:
            snap, best.snap = best.snap, None
            best.last_sent = now
            try:
                frame = self._snapshot_frame(best.sub, snap)
            except _wire.WireError:              # over the frame limit: counted, not sent
                self.p.counters["encode_failed"] += 1
                return self.next_frame(now)
            self.stats["snapshots_sent"] += 1
            return frame, None
        return None, wait

    @staticmethod
    def _snapshot_frame(sub: int, snap: SnapData):
        r = snap.red
        return _wire.encode_parts(_wire.Snapshot(
            sub, snap.view, snap.seq, snap.stamp_ns, snap.kind, snap.resolved, snap.data,
            aux=snap.aux, colormap=r.colormap, flags=r.flags, vmin=r.vmin, vmax=r.vmax,
            stats=snap.stats, axes=r.axes, tile=r.tile, points_dims=r.points_dims,
            label_count=r.label_count))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.drop_all()
        if self.attached:
            self.attached = False
            self.p.attached_changed(-1)


# ===========================================================================
# the session: views, groups, publishing (network thread)
# ===========================================================================
class _Producer:
    controls = None                            # 3.1: the server's caliper._net_controls.ControlsPort

    def __init__(self, ring: NetRing, identity: dict, transport: "NetTransport"):
        self.ring = ring
        self.identity = identity
        self.transport = transport
        self.links: set[Link] = set()
        self.views_version = -1
        self.hist: dict = {}                       # SeriesState -> _History
        self.events: collections.deque = collections.deque(maxlen=_wire.MAX_QUEUED_EVENTS)
        self.handlers: list[Callable] = []
        self._event_q: queue.SimpleQueue | None = None
        self.counters = collections.Counter()

    # -- identity and lists -----------------------------------------------------
    def sessions_msg(self) -> _wire.Sessions:
        with self.ring.lock:
            n = len(self.ring.views)
        return _wire.Sessions([dict(self.identity, state=_wire.SESSION_LIVE, views=n,
                                    capabilities=self.capabilities())])

    def capabilities(self) -> list:
        from ._net_controls import capabilities
        return sorted(capabilities(self.controls is not None) + [_wire.CAP_SCENES])

    def views_msg(self, link: Link | None = None) -> _wire.Views:
        with self.ring.lock:
            entries = [v.entry() for v in self.ring.views.values()][:_wire.MAX_VIEWS]
        if link is not None and self.controls is not None and _wire.CAP_SELECTION in link.caps:
            entries = self.controls.annotate(entries)
        if link is None or _wire.CAP_SCENES not in link.caps:          # 3.2 draft
            entries = [e for e in entries if e["kind"] != _wire.Kind.SCENE]
        return _wire.Views(entries, self.identity["id"])

    def attached_changed(self, delta: int) -> None:
        self.transport.n_attached = max(0, self.transport.n_attached + delta)
        self.ring.changed()

    # -- subscriptions ------------------------------------------------------------
    def resolve(self, name: str, request: dict | None, caps=None):
        """The view and its resolved data view. ``caps``, the capabilities of the
        peer that subscribes: a scene is unknown to a peer without "scenes",
        whatever data view it names (3.2 draft)."""
        with self.ring.lock:
            v = self.ring.views.get(name)
            entry = v.entry() if v is not None else None
        if v is None or (caps is not None and v.kind == _wire.Kind.SCENE and _wire.CAP_SCENES not in caps):
            raise _wire.MessageError(_wire.ERR_UNKNOWN_VIEW, f"no view {name!r}")
        try:
            if v.kind == KIND_SERIES:
                return v, _wire.resolve_data_view(entry, request, retain=SERIES_RETAIN)
            if v.kind == _wire.Kind.SCENE:                          # 3.2 draft
                return v, _wire.resolve_scene_data_view(entry, request)
            return v, self.ring.engine.resolve(entry, request)
        except _wire.MessageError:
            raise
        except Exception as e:                     # an engine bug: say so, keep the connection
            raise _wire.MessageError(_wire.ERR_INTERNAL, f"{type(e).__name__}: {e}") from None

    def join(self, link: Link, s: Sub) -> None:
        if s.kind == _wire.Kind.SCENE:                              # 3.2 draft
            from ._net_scene import route_of
            route_of(self.ring).join(link, s)
            return
        if s.kind == KIND_SERIES:
            st = self.ring.series.get(s.view)
            if st is None:
                return
            h = self.history(st)
            back = s.resolved["budget"]["max_backfill"]
            s.series = st
            s.next = max(h.first(), h.total - back)
            st.watched = True
            link.wake.set()
            return
        with self.ring.lock:
            v = self.ring.views.get(s.view)
            if v is None or v.id != s.view_id:
                return
            key = canonical(s.resolved)
            g = v.groups.get(key)
            if g is None:
                g = v.groups[key] = Group(s.view, s.resolved, key)
            g.subs[(link, s.sub)] = s.max_rate
            g.update_rate()
            s.group = g
        self.ring.changed()

    def leave(self, link: Link, s: Sub) -> None:
        s.snap = None
        if s.kind == _wire.Kind.SCENE:                              # 3.2 draft
            from ._net_scene import route_of
            route_of(self.ring).leave(link, s)
            return
        if s.series is not None:
            st = s.series
            s.series = None
            st.watched = any(x.series is st for ln in self.links for x in ln.subs.values())
            return
        g = s.group
        if g is None:
            return
        s.group = None
        with self.ring.lock:
            g.subs.pop((link, s.sub), None)
            if g.subs:
                g.update_rate()
            else:
                v = self.ring.views.get(g.view)
                if v is not None and v.groups.get(g.key) is g:
                    del v.groups[g.key]
        self.ring.changed()

    def history(self, st) -> _History | None:
        h = self.hist.get(st)
        if h is None and st is not None:
            h = self.hist[st] = _History()
        return h

    # -- events -------------------------------------------------------------------
    def event(self, link: Link, msg: _wire.Event) -> None:
        ev = {"view": msg.view, "type": msg.type, "detail": msg.detail, "sub": msg.sub,
              "seq": msg.seq, "data_view": msg.data_view, "connection": link.id,
              "time": time.time()}
        self.counters["events"] += 1
        self.events.append(ev)
        q = self._event_q
        if q is not None:
            q.put(ev)
        if msg.type == _wire.EVENT_SELECT and self.controls is not None:
            self.controls.on_select(link, msg)

    # -- the pump -----------------------------------------------------------------
    def pump_once(self) -> bool:
        """Publish what completed. Returns True while work is still in flight."""
        ring = self.ring
        if ring.views_version != self.views_version:
            with ring.lock:
                self.views_version = ring.views_version
                views = dict(ring.views)
                live = set(ring.series.values())
            for st in [st for st in self.hist if st not in live]:
                del self.hist[st]
            for link in list(self.links):
                link.views_changed(views)
        busy = False
        with ring.lock:
            pend = ring.pending
            ready = [p for p in pend if p.ready()]
            if ready:
                ring.pending = [p for p in pend if p not in ready]
            busy = len(ring.pending) > 0
        for p in ready:
            self.transport.publish(p.view, Slot(0, p.stage.data, p.stage.stats, p.stage.aux,
                                                p.event), _NetMeta(p))
        scenes = ring.__dict__.get("scenes")                        # 3.2 draft
        if scenes is not None and scenes.pump():
            busy = True
        for st in list(ring.series.values()):
            q = st.q
            got = False
            floats, stamp = [], 0
            while q:
                item = q[0]
                if isinstance(item, tuple):
                    floats.append(item[0])
                    stamp = item[1]
                else:
                    if floats:
                        self.history(st).append(floats, stamp)
                        floats = []
                    if not item.ready():
                        busy = True
                        break
                    self.history(st).append(item.buf.numpy()[:item.n], item.stamp_ns)
                q.popleft()
                got = True
            if floats:
                self.history(st).append(floats, stamp)
            if got and st.watched:
                for link in self.links:
                    for s in link.subs.values():
                        if s.series is st:
                            link.wake.set()
                            break
        return busy

    def deliver(self, p) -> None:
        """A completed snapshot: copy it out of pinned memory and offer it to
        every subscriber of its group."""
        stage, red = p.stage, p.red
        data = stage.data.numpy().copy()
        aux = stage.aux.numpy().copy() if stage.aux is not None else None
        if red.stats is None:
            stats = host_stats(p.kind, data, aux)
        else:
            stats = stage.stats[:red.stats_count].numpy().tolist() if red.stats_count else []
        stage.busy = False
        self.counters["published"] += 1
        snap = SnapData(p.view, p.view_id, p.seq, p.stamp_ns, p.kind, p.group.resolved,
                        data, aux, stats, red)
        with self.ring.lock:
            subs = list(p.group.subs)
        for link, sub in subs:
            link.offer_snapshot(sub, snap)


def host_stats(kind: int, data, aux) -> list:
    """The stats of a snapshot from its payload as sent (NaN left out, NaN when
    nothing is left), as caliper._netio computes them: heatmap [min, max];
    points [aux min, aux max, x min, x max, y min, y max, z min, z max]."""
    import numpy as np

    def rng(a):
        a = a[~np.isnan(a)] if a.size else a
        return [float(a.min()), float(a.max())] if a.size else [float("nan")] * 2
    if kind == KIND_HEATMAP:
        return rng(data.reshape(-1))
    if kind == KIND_POINTS:
        out = rng(aux) if aux is not None else [0.0, 0.0]
        for k in range(3):
            out += rng(data[:, k]) if k < data.shape[1] else [0.0, 0.0]
        return out
    return []


class _NetMeta(SnapshotMeta):
    """SnapshotMeta plus the pending entry the network route publishes."""

    def __init__(self, pending):
        super().__init__(kind=0, slot=0, seq=pending.seq, stamp_ns=pending.stamp_ns)
        self.pending = pending


# ===========================================================================
# the Transport of the route (caliper/_ring.py protocol)
# ===========================================================================
class NetTransport:
    """Where the network route's slots live (pinned host memory) and where
    completed snapshots go (the links). ``attached()`` is True while a viewer
    is attached and some view may be due now: an attribute read and a clock
    read, so a call between due times costs what it costs with no viewer."""

    def __init__(self):
        self.n_attached = 0
        self.producer: _Producer | None = None
        self.server: "NetServer | None" = None
        self.ring = None

    def attached(self) -> bool:
        return self.n_attached > 0 and time.perf_counter() >= self.ring.next_due

    def alloc(self, name, spec, slot, role, shape, dtype):
        import torch
        return torch.empty(shape, dtype=dtype, pin_memory=spec.device.type != "cpu")

    def reading_slot(self, name: str) -> int:
        return -1

    def publish(self, name: str, slot: Slot, meta: SnapshotMeta) -> None:
        self.producer.deliver(meta.pending)

    def remove(self, name: str) -> None:
        if self.server is not None:
            self.server.ring.forget(name)

    def stats(self) -> dict:
        return self.server.stats() if self.server is not None else {}

    def close(self) -> None:
        if self.server is not None:
            self.server.close()


# ===========================================================================
# the listener
# ===========================================================================
def _response(status: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    h = Headers()
    h["Content-Type"] = ctype
    h["Content-Length"] = str(len(body))
    h["Cache-Control"] = "no-store"
    h["X-Content-Type-Options"] = "nosniff"
    h["Connection"] = "close"
    return Response(status, HTTPStatus(status).phrase, h, body)


def _static(path: str, origin: str | None = None):
    """The viewer page and its files, or None for a 404. Public code with no
    data, so no token; a cross-origin request from an origin the protocol
    accepts (a notebook page fetching the widget's modules) gets that origin
    back in Access-Control-Allow-Origin, never a wildcard."""
    r = _static_file(path)
    if r is not None and origin is not None and origin_ok(origin):
        r.headers["Access-Control-Allow-Origin"] = origin
        r.headers["Vary"] = "Origin"
    return r


def _static_file(path: str):
    try:
        from . import _assets
    except ImportError:
        _assets = None
    if _assets is None or not _assets.asset_names():
        return _response(200, _NOT_BUILT, "text/html; charset=utf-8") if path in ("/", "/index.html") else None
    try:
        name = _assets.name_for_url_path(path)
        body = _assets.read_asset(name)
    except FileNotFoundError:
        if path in ("/", "/index.html"):
            return _response(200, _NOT_BUILT, "text/html; charset=utf-8")
        return None
    except ValueError:
        return None
    return _response(200, body, _assets.content_type(name))


def _ws_header(n: int) -> bytes:
    """RFC 6455 header of an unmasked, unfragmented binary frame of n bytes."""
    if n < 126:
        return bytes((0x82, n))
    if n < 65536:
        return struct.pack("!BBH", 0x82, 126, n)
    return struct.pack("!BBQ", 0x82, 127, n)


async def _send_parts(ws, parts: list, n: int) -> None:
    """Send one binary message given as buffers without copying them.

    websockets copies a message twice while it frames it; for megabyte
    snapshots that is most of the network thread's time, and it holds the GIL
    the training loop needs. A server frame is unmasked, so this writes the
    2 to 10 header bytes and the buffers to the connection's transport in one
    call (asyncio sends a list with sendmsg, no copy), then waits for the write
    buffer with websockets' own flow control. Nothing else can write in between:
    the call does not yield. Falls back to ``ws.send`` when the connection is
    not open or the attributes are missing."""
    from websockets.protocol import State
    transport = getattr(ws, "transport", None)
    proto = getattr(ws, "protocol", None)
    if transport is None or proto is None or proto.state is not State.OPEN \
            or transport.is_closing():
        await ws.send(b"".join(parts))
        return
    transport.writelines([_ws_header(n), *parts])
    await ws.drain()


class _Listener:
    """The websockets server on 127.0.0.1 and the connections it accepts."""

    def __init__(self, server: "NetServer"):
        self.s = server
        self.ws_server = None
        self.port = 0
        self.conns: set = set()

    async def start(self, port: int) -> None:
        from websockets.asyncio.server import serve
        self.ws_server = await serve(
            self.handler, "127.0.0.1", port, process_request=self.process_request,
            compression=None, ping_interval=None, ping_timeout=None,
            max_size=_wire.MAX_FRAME_BYTES, max_queue=16, write_limit=WRITE_LIMIT,
            server_header=None, open_timeout=_wire.HANDSHAKE_TIMEOUT_MS / 1000,
            close_timeout=2, logger=_log)
        self.port = self.ws_server.sockets[0].getsockname()[1]

    def process_request(self, connection, request):
        s = self.s
        s.counters["http_requests"] += 1
        if not host_ok(request.headers.get("Host"), self.port):
            s.counters["refused_host"] += 1
            return _response(403, b"Forbidden: Host\n")
        if getattr(request, "method", "GET") != "GET":
            r = _response(405, b"Method Not Allowed\n")
            r.headers["Allow"] = "GET"
            return r
        path, _, query = request.path.partition("?")
        if path == _wire.PATH:
            if not origin_ok(request.headers.get("Origin")):
                s.counters["refused_origin"] += 1
                return _response(403, b"Forbidden: Origin\n")
            token = parse_qs(query, keep_blank_values=True).get(_wire.TOKEN_PARAM, [""])[0]
            if not hmac.compare_digest(token.encode("utf-8"), s.token.encode("utf-8")):
                s.counters["refused_token"] += 1
                return _response(403, b"Forbidden: token\n")
            if len(self.conns) >= MAX_CONNECTIONS:
                s.counters["refused_full"] += 1
            return None
        r = _static(path, request.headers.get("Origin"))
        if r is None:
            return _response(404, b"Not Found\n")
        s.counters["http_files"] += 1
        return r

    async def handler(self, ws) -> None:
        s = self.s
        self.conns.add(ws)
        link = None
        try:
            if len(self.conns) > MAX_CONNECTIONS:
                await ws.close(_wire.CLOSE_SERVER_FULL, "server full")
                return
            try:
                first = await asyncio.wait_for(ws.recv(), _wire.HANDSHAKE_TIMEOUT_MS / 1000)
            except asyncio.TimeoutError:
                await ws.close(_wire.CLOSE_HANDSHAKE_TIMEOUT, "no hello")
                return
            if isinstance(first, str):
                await ws.close(_wire.CLOSE_UNSUPPORTED_DATA, "binary frames only")
                return
            try:
                hello = _wire.decode(first)
            except _wire.WireError as e:
                await ws.close(e.close_code, e.reason)
                return
            except _wire.MessageError as e:
                await self._fatal(ws, e.code, e.message, _wire.CLOSE_PROTOCOL_ERROR)
                return
            if isinstance(hello, _wire.Bye):
                await ws.close(_wire.CLOSE_NORMAL)
                return
            if not isinstance(hello, _wire.Hello):
                await self._fatal(ws, _wire.ERR_BAD_REQUEST, "the first message is hello",
                                  _wire.CLOSE_PROTOCOL_ERROR)
                return
            if hello.role != _wire.ROLE_VIEWER:
                await self._fatal(ws, _wire.ERR_UNSUPPORTED,
                                  "caliper.serve() accepts viewers only; producers connect "
                                  "to a hub", _wire.CLOSE_ROLE_REFUSED)
                return
            link = Link(s.producer, peer=str(ws.remote_address))
            link.software = hello.software
            ctl = s.producer.controls
            control = None
            if ctl is not None:
                from ._net_controls import peer_caps
                link.caps = peer_caps(hello.capabilities)
                if _wire.CAP_CONTROLS in link.caps:
                    control = link.may_control = ctl.allow
            if _wire.CAP_SCENES in hello.capabilities:            # 3.2 draft
                link.caps = link.caps | {_wire.CAP_SCENES}
            await ws.send(_wire.encode(_wire.Welcome(
                _wire.ROLE_PRODUCER, s.producer.capabilities(), s.software, s.identity,
                control=control)))
            s.producer.links.add(link)
            s.counters["connections"] += 1
            link.start()
            await self.run(ws, link)
        except Exception as e:                    # a closed socket mid-handshake, a bug
            from websockets.exceptions import ConnectionClosed
            if not isinstance(e, (ConnectionClosed, asyncio.CancelledError)):
                s.counters["handler_errors"] += 1
                _log.debug("connection failed", exc_info=True)
        finally:
            self.conns.discard(ws)
            if link is not None:
                s.producer.links.discard(link)
                link.close()

    async def _fatal(self, ws, code: str, text: str, close_code: int) -> None:
        try:
            await ws.send(_wire.encode(_wire.Error(code, text, True)))
        finally:
            await ws.close(close_code, text[:120])

    async def run(self, ws, link: Link) -> None:
        reader = asyncio.ensure_future(self._reader(ws, link))
        writer = asyncio.ensure_future(self._writer(ws, link))
        done, pending = await asyncio.wait({reader, writer}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if isinstance(exc, _Fatal):
                if exc.error and exc.code != _wire.CLOSE_NORMAL:
                    try:
                        await asyncio.wait_for(
                            ws.send(_wire.encode(_wire.Error(_wire.ERR_BAD_REQUEST, exc.error, True))), 1.0)
                    except Exception:
                        pass
                try:                                 # a peer that reads nothing cannot take the close
                    await asyncio.wait_for(ws.close(exc.code, exc.reason[:120]), 3.0)
                except asyncio.TimeoutError:
                    ws.transport.abort()
            elif exc is not None:
                raise exc
        await ws.close()

    async def _reader(self, ws, link: Link) -> None:
        from websockets.exceptions import ConnectionClosed
        idle = _wire.PING_INTERVAL_MS / 1000
        limit = _wire.IDLE_TIMEOUT_MS / 1000
        last = time.monotonic()
        pinged = False
        nonce = itertools.count(1)
        while True:
            wait = (last + (limit if pinged else idle)) - time.monotonic()
            try:
                msg = await asyncio.wait_for(ws.recv(), max(wait, 0.0))
            except asyncio.TimeoutError:
                if pinged:
                    raise _Fatal(_wire.CLOSE_IDLE_TIMEOUT, "idle") from None
                link.queue(_wire.Ping(next(nonce) & 0xFFFFFFFF))
                pinged = True
                continue
            except ConnectionClosed:
                return
            last = time.monotonic()
            pinged = False
            if isinstance(msg, str):
                raise _Fatal(_wire.CLOSE_UNSUPPORTED_DATA, "binary frames only")
            link.handle(msg)
            if link.overflow:
                raise _Fatal(CLOSE_OVERFLOW, "control queue full")

    async def _writer(self, ws, link: Link) -> None:
        mono = time.monotonic
        while True:
            frame, wait = link.next_frame(mono())
            if frame is None:
                link.wake.clear()
                frame, wait = link.next_frame(mono())
                if frame is None:
                    await link.wake.wait(wait)
                    continue
            if type(frame) is tuple:
                parts, n = frame
                await _send_parts(ws, parts, n)
            else:
                await ws.send(frame)
                n = len(frame)
            link.stats["frames_out"] += 1
            link.stats["bytes_out"] += n

    async def shutdown(self) -> None:
        bye = _wire.encode(_wire.Bye(_wire.BYE_SHUTDOWN, "the producer is closing"))
        conns = list(self.conns)

        async def close_one(ws):
            try:
                await asyncio.wait_for(ws.send(bye), 1.0)
            except Exception:
                pass
            try:
                await asyncio.wait_for(ws.close(_wire.CLOSE_GOING_AWAY, "shutdown"), 2.0)
            except Exception:
                pass
        await asyncio.gather(*(close_one(ws) for ws in conns), return_exceptions=True)
        if self.ws_server is not None:
            self.ws_server.close()
            try:
                await asyncio.wait_for(self.ws_server.wait_closed(), 3.0)
            except asyncio.TimeoutError:
                pass


# ===========================================================================
# the server object
# ===========================================================================
class NetServer:
    """What ``caliper.serve()`` returns: the running stream server.

    ``url`` is the viewer page (``http://127.0.0.1:<port>/?token=<token>``),
    ``stream_url`` the WebSocket endpoint, ``port``, ``token`` and ``session``
    (the session id) identify it, and ``closed`` says whether it still runs.
    ``on_event(fn)`` registers a function that receives every viewer event as a
    dict (on a thread of its own, never the training thread); ``events()``
    returns and clears the events received since the last call (at most 256).
    ``close()`` ends every connection with BYE and stops the server; it also
    runs at ``caliper.close()`` and at interpreter exit."""

    def __init__(self, port: int = 0, *, engine=None, control: bool = True):
        port = int(port)
        if not 0 <= port <= 65535:
            raise ValueError(f"port must be in 0..65535, got {port}")
        self.token = secrets.token_urlsafe(24)
        self.counters = collections.Counter()
        self.closed = False
        pid = os.getpid()
        self.identity = {"id": f"clp-{pid}-{secrets.token_hex(3)}", "script": _script_name(),
                         "pid": pid, "start_unix_ms": int(time.time() * 1000)}
        from . import __version__
        self.software = {"name": "caliper-python", "version": __version__}
        self.transport = NetTransport()
        self.transport.server = self
        self._loop = asyncio.new_event_loop()
        self._pump_wake: _Wake | None = None
        self._pump_idle = True
        self.ring = NetRing(engine, wake=self._wake)
        self.transport.ring = self.ring
        self.producer = _Producer(self.ring, self.identity, self.transport)
        self.transport.producer = self.producer
        self._listener = _Listener(self)
        self._thread = threading.Thread(target=self._run, name="caliper.net", daemon=True)
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._pump_task = None
        self._events_thread: threading.Thread | None = None
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._start(port), self._loop)
        try:
            fut.result(15)
        except BaseException:
            self._stop_loop()
            raise
        self.port = self._listener.port
        self.url = f"http://127.0.0.1:{self.port}/?token={self.token}"
        self.stream_url = f"ws://127.0.0.1:{self.port}{_wire.PATH}?{_wire.TOKEN_PARAM}={self.token}"
        from ._net_controls import ControlsPort
        self.producer.controls = ControlsPort(self, control, native=False)

    @property
    def control(self) -> bool:
        """Whether viewers may control the script (``serve(control=...)``)."""
        c = self.producer.controls
        return c is not None and c.allow

    @property
    def session(self) -> str:
        return self.identity["id"]

    # -- the network thread -------------------------------------------------------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Finish every task here, on this loop: a websockets task left
            # pending (a stalled viewer at shutdown) would otherwise be
            # collected on another thread, whose loop would get its async
            # generators to close.
            try:
                left = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for t in left:
                    t.cancel()
                if left:
                    self._loop.run_until_complete(asyncio.wait(left, timeout=2.0))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _start(self, port: int) -> None:
        self._pump_wake = _Wake()
        await self._listener.start(port)
        self._pump_task = asyncio.ensure_future(self._pump())

    def _wake(self) -> None:
        """From the training thread: something new for the pump."""
        if self._pump_idle and not self.closed:
            self._pump_idle = False
            try:
                self._loop.call_soon_threadsafe(self._pump_wake.set)
            except RuntimeError:                      # the loop has closed
                pass

    async def _pump(self) -> None:
        p = self.producer
        while True:
            try:
                busy = p.pump_once()
            except Exception:
                self.counters["pump_errors"] += 1
                _log.debug("pump failed", exc_info=True)
                busy = False
            if busy or any(st.watched and st.q for st in list(self.ring.series.values())):
                await asyncio.sleep(POLL_S)
                continue
            self._pump_idle = True
            self._pump_wake.clear()
            sc = self.ring.__dict__.get("scenes")
            if self.ring.pending or (sc is not None and sc.pending):   # raced with a wake
                continue
            await self._pump_wake.wait(0.25)
            self._pump_idle = False

    # -- events --------------------------------------------------------------------
    def on_event(self, fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
        """Call ``fn(event)`` for every viewer event from now on (usable as a
        decorator). ``event`` is a dict: ``view``, ``type`` ("select", "pick",
        "hover"), ``detail``, ``sub``, ``seq``, ``data_view``, ``connection``,
        ``time``. ``fn`` runs on the thread "caliper.events"; an exception in it
        is reported as a warning and does not stop later events."""
        if not callable(fn):
            raise TypeError("on_event() takes a callable")
        p = self.producer
        p.handlers.append(fn)
        if self._events_thread is None:
            p._event_q = queue.SimpleQueue()
            self._events_thread = threading.Thread(target=self._dispatch_events,
                                                   name="caliper.events", daemon=True)
            self._events_thread.start()
        return fn

    def remove_event_handler(self, fn) -> bool:
        try:
            self.producer.handlers.remove(fn)
            return True
        except ValueError:
            return False

    def _dispatch_events(self) -> None:
        q = self.producer._event_q
        while True:
            ev = q.get()
            if ev is None:
                return
            for fn in list(self.producer.handlers):
                try:
                    fn(ev)
                except Exception as e:
                    warnings.warn(f"caliper: event handler {fn!r} raised {type(e).__name__}: {e}",
                                  RuntimeWarning, stacklevel=1)

    def events(self) -> list[dict]:
        """The viewer events received since the last call, oldest first, and
        clears them. Never blocks."""
        out = []
        d = self.producer.events
        while True:
            try:
                out.append(d.popleft())
            except IndexError:
                return out

    # -- inspection ----------------------------------------------------------------
    def stats(self) -> dict:
        p = self.producer
        links = []
        for ln in list(p.links):
            links.append({"id": ln.id, "peer": ln.peer, "attached": ln.attached,
                          "subscriptions": len(ln.subs), "control_frames": len(ln.control),
                          "control_bytes": ln.control_bytes, **dict(ln.stats)})
        with self.ring.lock:
            views = {n: {"id": v.id, "kind": v.kind, "groups": len(v.groups), "seq": v.seq}
                     for n, v in self.ring.views.items()}
        return {"url": self.url, "port": self.port, "session": self.session,
                "closed": self.closed, "attached": self.transport.n_attached,
                "connections": links, "views": views, "ring": dict(self.ring.counters),
                "server": dict(self.counters), "producer": dict(p.counters),
                "series": {st.name: self.producer.hist[st].total
                           for st in list(self.producer.hist)}}

    # -- lifetime -------------------------------------------------------------------
    def close(self) -> None:
        """Deliver what is pending (at most about a second), end every
        connection with BYE and stop the server. Idempotent."""
        if self.closed:
            return
        if self._loop.is_running() and threading.current_thread() is not self._thread:
            try:
                # gathered series values (moved only if this thread gathered
                # them) and queued snapshots
                self.ring.flush(timeout=1.0)
                asyncio.run_coroutine_threadsafe(self._drain(1.0), self._loop).result(3)
            except Exception:
                pass
        self.closed = True
        self.ring.close()
        self.transport.n_attached = 0
        _unroute(self)
        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                fut.result(10)
            except Exception:
                pass
        self._stop_loop()
        q = self.producer._event_q
        if q is not None:
            q.put(None)

    async def _drain(self, timeout: float) -> None:
        """Publish what completed and let the writers send it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            busy = self.producer.pump_once()
            links = list(self.producer.links)
            if not busy and not self.ring.pending and all(
                    not ln.control and all(s.snap is None and (s.series is None or s.next >= (
                        self.producer.history(s.series).total)) for s in ln.subs.values())
                    for ln in links):
                return
            await asyncio.sleep(0.005)

    async def _shutdown(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
        await self._listener.shutdown()

    def _stop_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(5)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else "running"
        return f"<caliper.NetServer {redact(self.url)} {state}>"


class NetRoute:
    """The route caliper/__init__.py keeps for the server (as ``_Route``).
    Series reach it through ``caliper._series_tap``, not its ring."""
    sync = False
    taps_series = True

    def __init__(self, server: NetServer):
        self.server = server
        self.transport = server.transport
        self.ring = server.ring

    @property
    def closed(self) -> bool:
        return self.server.closed

    def close(self) -> None:
        self.server.close()

    def __repr__(self) -> str:
        return "serve"


# ===========================================================================
# caliper.serve()
# ===========================================================================
def current_server() -> NetServer | None:
    """The running server, or None."""
    import caliper
    s = getattr(caliper, "_server", None)
    return s if s is not None and not s.closed else None


_said_python_sender = False


def _make_server(port: int, control: bool = True) -> NetServer:
    """The native sender (caliper._netio) when it is built, else the Python
    sender, which says once what that costs. ``CALIPER_NETIO=0`` forces the
    Python sender."""
    global _said_python_sender
    if os.environ.get("CALIPER_NETIO", "1") != "0":
        try:
            from . import _netio  # noqa: F401
        except ImportError:
            pass
        else:
            from ._net_native import NativeServer
            return NativeServer(port, control=control)
        if not _said_python_sender:
            _said_python_sender = True
            print("caliper: the native network sender (caliper._netio) is not built, so serve() "
                  "sends from Python, which costs a training loop about 5 to 9 % of its steps "
                  "while a viewer watches; build it with `python scripts/build_netio.py`.",
                  file=sys.stderr, flush=True)
    return NetServer(port, control=control)


def _add_route(s) -> None:
    """Make ``s`` this process's network route (under caliper.LOCK)."""
    import caliper
    global _producer
    route = NetRoute(s)
    s._route = route
    caliper._extra_routes.append(route)
    caliper._set_routes()
    caliper._series_tap = s.ring.tap
    sc = sys.modules.get("caliper._scene")      # 3.2 draft: list the scenes made before
    if sc is not None:
        sc.attach_ring(s.ring)
    _producer = s


_producer = None                     # the route's server: serve()'s, or connect()'s without a listener


def serve(port: int = 0, control: bool = True) -> NetServer:
    """Start (or return) the stream server of this process. See caliper.serve."""
    import caliper
    with caliper.LOCK:
        s = current_server()
        if s is not None:
            return s
        p = _producer
        if p is not None and not p.closed and not getattr(p, "listening", True):
            p.listen(port, control=control)   # connect() came first: one producer, now listening too
            s = p
        else:
            s = _make_server(port, control)
            _add_route(s)
        caliper._server = s
    print(s.url, flush=True)
    from ._assets import note_missing_views
    note_missing_views()
    return s


# ---------------------------------------------------------------------------
# caliper.connect(): the producer's connection out to a `caliper serve` hub
# ---------------------------------------------------------------------------
def hub_file() -> str:
    """Where `caliper serve` writes its hub file: $CALIPER_HUB_FILE, else
    hub.json in Caliper's app-data folder (the hub's own rule)."""
    p = os.environ.get("CALIPER_HUB_FILE")
    if p:
        return p
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support", "Caliper")
    elif os.name == "nt":
        base = os.path.join(os.environ.get("APPDATA") or os.getcwd(), "Caliper")
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = os.path.join(xdg, "caliper") if xdg else os.path.join(home, ".local", "share", "caliper")
    return os.path.join(base, "hub.json")


def hub_stream_url(url: str | None = None) -> str:
    """The hub's stream URL with its token: ``url`` (a ws:// stream URL, or the
    http:// viewer link the hub prints), else $CALIPER_HUB_URL, else the hub file."""
    from urllib.parse import parse_qs, urlsplit
    src = url or os.environ.get("CALIPER_HUB_URL")
    if not src:
        path = hub_file()
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"caliper.connect(): no hub runs (no hub file {path}); start one with "
                               "`caliper serve`, or pass its URL") from None
        except (OSError, ValueError) as e:
            raise RuntimeError(f"caliper.connect(): cannot read the hub file {path}: {e}") from None
        token, port = info.get("token"), info.get("port")
        if not isinstance(token, str) or not isinstance(port, int):
            raise RuntimeError(f"caliper.connect(): the hub file {path} has no port and token")
        base = info.get("stream_url") or f"ws://127.0.0.1:{port}{_wire.PATH}"
        return f"{base.split('?', 1)[0]}?{_wire.TOKEN_PARAM}={token}"
    u = urlsplit(src)
    token = parse_qs(u.query).get(_wire.TOKEN_PARAM, [""])[0]
    if u.scheme not in ("ws", "http") or not u.hostname or not token:
        raise ValueError(f"caliper.connect(): expected ws://127.0.0.1:<port>/v3/stream?token=<token> "
                         f"or the hub's http:// link, got {redact(src)!r}")
    if u.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("caliper.connect(): the hub must be on this machine (127.0.0.1)")
    port = u.port or 80
    return f"ws://127.0.0.1:{port}{_wire.PATH}?{_wire.TOKEN_PARAM}={token}"


def connect(url: str | None = None):
    """Connect this process to a `caliper serve` hub. See caliper.connect."""
    import caliper
    ws_url = hub_stream_url(url)
    try:
        from . import _netio  # noqa: F401
    except ImportError:
        raise RuntimeError("caliper.connect() needs the native sender (caliper._netio); build it "
                           "with `python scripts/build_netio.py`") from None
    with caliper.LOCK:
        p = _producer if _producer is not None and not _producer.closed else None
        if p is None:
            from ._net_native import NativeServer
            p = NativeServer(None)
            _add_route(p)
        elif not hasattr(p, "connect_hub"):
            raise RuntimeError("caliper.connect() needs the native sender, and caliper.serve() runs "
                               "on the Python sender (CALIPER_NETIO=0)")
        return p.connect_hub(ws_url)


def _unroute(server: NetServer) -> None:
    import caliper
    global _producer
    ctl = server.producer.controls
    if ctl is not None:
        ctl.detach()                     # a paused run with no other route goes on
    with caliper.LOCK:
        if _producer is server:
            _producer = None
        route = getattr(server, "_route", None)
        if route is not None and route in caliper._extra_routes:
            caliper._extra_routes.remove(route)
            caliper._set_routes()
        if getattr(caliper, "_server", None) is server:
            caliper._server = None
            caliper._series_tap = None
