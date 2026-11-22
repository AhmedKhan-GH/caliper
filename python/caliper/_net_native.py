"""caliper.serve() on the native sender (caliper._netio).

The pure-Python sender (caliper/_net.py) frames, encodes and sends on an
asyncio thread, and every wake-up of that thread holds the GIL the training
loop needs. Here a native thread (python/src/_netio.cpp) does the listening,
the upgrade checks, the viewer's files, HELLO and WELCOME, SESSIONS, ATTACH,
VIEWS, PING and the timeouts, latest-wins snapshots and coalesced series, the
encoding and the sends, all without the GIL. Python keeps three jobs:

* the training thread reduces snapshots and queues their copies into pinned
  memory (caliper/_net_ring.py, unchanged);
* the pump thread ("caliper.net") waits in ``Server.control()`` with the GIL
  released; it wakes for a control message, a wake from the training thread,
  every ``POLL_MS`` while snapshots are in flight, or when a scene frame that
  waits for its rate is due (``wait_ms``), polls their MPS events
  (``torch.mps.Event.query`` needs the GIL), and hands each completed one to
  the native side with one ``publish()`` call, which copies the bytes with the
  GIL released before it returns: from then on the native side owns them, so
  the pinned memory is reused at once;
* SUBSCRIBE, UNSUBSCRIBE, VIEW_REQUEST, EVENT and (wire protocol 3.1)
  CONTROL, which need the data view engine or Python code, arrive as frames
  through a queue the native side bounds (256 per connection) and are answered
  by the same :class:`Link` logic as in the Python sender. The controls'
  messages (caliper/_net_controls.py) go out with ``send()`` from the pump
  thread.
"""
from __future__ import annotations

import collections
import json
import math
import os
import threading
import time

from . import _wire
from ._net import (MAX_CONNECTIONS, SERIES_RETAIN, Link, NetServer, NetTransport, _Producer,
                   _script_name)
from ._net_ring import NetRing
from ._ring import KIND_SERIES

__all__ = ["NativeServer"]

POLL_MS = 4             # the pump's event poll while snapshots or chunks are in flight
IDLE_MS = 250           # otherwise: retained series of unwatched views drain this often
_DTYPE = {"torch.float32": _wire.DType.F32, "torch.uint8": _wire.DType.U8,
          "float32": _wire.DType.F32, "uint8": _wire.DType.U8}


def _asset_table() -> list:
    """(name, content type, file path) of every file of the viewer page."""
    try:
        from . import _assets
    except ImportError:
        return []
    out = []
    for n in _assets.asset_names():
        try:
            out.append((n, _assets.content_type(n), str(_assets.asset_path(n))))
        except (ValueError, FileNotFoundError):
            pass
    return out


class _NativeLink(Link):
    """A connection of the native sender: Python sees only the messages it
    must answer; frames go out through the native side."""

    def __init__(self, producer: "_NativeProducer", conn: int):
        super().__init__(producer, peer=f"conn {conn}")
        self.conn = conn

    def queue(self, msg) -> None:
        self.p.nio.send(self.conn, _wire.encode(msg))

    def queue_views(self) -> None:
        pass                                      # the native side sends VIEWS


class _NativeProducer(_Producer):
    def __init__(self, ring: NetRing, identity: dict, transport: NetTransport, nio):
        super().__init__(ring, identity, transport)
        self.nio = nio
        self.by_conn: dict[int, _NativeLink] = {}
        self._gids = 0
        self._series_seen: set = set()

    # -- subscriptions: Python's groups (for the training thread) and the native side's
    def join(self, link: _NativeLink, s) -> None:
        if s.kind == KIND_SERIES:
            st = self.ring.series.get(s.view)
            if st is None:
                return
            s.series = st
            st.watched = True
            self.nio.sub_open(link.conn, s.sub, s.view, int(s.kind), -1, "{}", s.max_rate,
                              int(s.resolved["budget"]["max_backfill"]))
            self.ring.changed()
            return
        super().join(link, s)
        g = s.group
        if g is None:
            return
        gid = g.__dict__.get("gid")
        if gid is None:
            self._gids += 1
            gid = g.__dict__["gid"] = self._gids
        self.nio.sub_open(link.conn, s.sub, s.view, int(s.kind), gid,
                          json.dumps(s.resolved, sort_keys=True), s.max_rate, 0)

    def leave(self, link: _NativeLink, s) -> None:
        self.nio.sub_close(link.conn, s.sub)
        super().leave(link, s)

    # -- the pump ---------------------------------------------------------------------
    def busy(self) -> bool:
        """Work the pump polls for every POLL_MS: copies or chunks in flight, a held
        CONTROLS, series values to hand over, or a scene frame waiting for room in its
        link (pump_native's wait of 4 ms). A scene frame that waits for its rate is not
        busy work: the pump sleeps until it is due (wait_ms)."""
        if self.ring.pending or (self.controls is not None and self.controls.holding()):
            return True
        sc = self.ring.__dict__.get("scenes")                       # 3.2 draft
        if sc is not None and (sc.pending or (sc.native_wait is not None and
                                              sc.native_wait * 1000.0 <= POLL_MS)):
            return True
        for st in list(self.ring.series.values()):
            if st.q:
                return True
        return False

    def wait_ms(self) -> int:
        """How long the pump waits for a control message or a wake: POLL_MS while
        busy, else until the next rate-limited scene frame is due, at most IDLE_MS."""
        if self.busy():
            return POLL_MS
        sc = self.ring.__dict__.get("scenes")
        due = sc.native_wait if sc is not None else None
        if due is None:
            return IDLE_MS
        return max(POLL_MS, min(IDLE_MS, math.ceil(due * 1000.0)))

    def pump_once(self) -> bool:
        ring = self.ring
        if ring.views_version != self.views_version:
            with ring.lock:
                self.views_version = ring.views_version
                views = dict(ring.views)
                entries = [v.entry() for v in ring.views.values()][:_wire.MAX_VIEWS]
                live = set(ring.series)
            items = ""
            if self.controls is not None and self.controls.route.ctl.view_items:
                items = json.dumps(self.controls.annotate(entries), sort_keys=True)
            self.nio.set_views(json.dumps(entries, sort_keys=True), items)
            for name in self._series_seen - live:
                self.nio.series_drop(name)
            self._series_seen = live
            for link in list(self.links):
                link.views_changed(views)
            sf = self.stream_file
            if sf is not None:
                sf.views(views)
        with ring.lock:
            pend = ring.pending
            ready = [p for p in pend if p.ready()]
            if ready:
                ring.pending = [p for p in pend if p not in ready]
            busy = len(ring.pending) > 0
        for p in ready:
            self.deliver(p)
        sc = ring.__dict__.get("scenes")                            # 3.2 draft: scene messages
        if sc is not None:
            if sc.pump():
                busy = True
            sc.native_wait = sc.pump_native(list(self.links), time.monotonic())
        for st in list(ring.series.values()):
            q = st.q
            floats, stamp = [], 0
            while q:
                item = q[0]
                if isinstance(item, tuple):
                    floats.append(item[0])
                    stamp = item[1]
                else:
                    if floats:
                        self._append(st.name, floats, stamp)
                        floats = []
                    if not item.ready():
                        busy = True
                        break
                    self.nio.series_append(st.name, item.buf.data_ptr(), item.n, item.stamp_ns)
                    if len(st.pool) < 4:
                        st.pool.append(item.buf)              # copied: the training thread may reuse it
                q.popleft()
            if floats:
                self._append(st.name, floats, stamp)
        return busy

    def _append(self, name: str, floats: list, stamp: int) -> None:
        import array
        a = array.array("f", floats)
        self.nio.series_append(name, a.buffer_info()[0], len(a), stamp)

    def deliver(self, p) -> None:
        """Hand a completed snapshot to the native side, which copies it before
        it returns; the pinned stage is free again at once."""
        stage, red, g = p.stage, p.red, p.group
        gid = g.__dict__.get("gid")
        if gid is not None:
            data = stage.data
            aux = stage.aux
            axes = g.__dict__.get("axes_json")
            if axes is None or g.__dict__.get("axes") != red.axes:
                axes = json.dumps(red.axes, sort_keys=True)
                g.__dict__["axes"], g.__dict__["axes_json"] = red.axes, axes
            self.nio.publish(
                gid, p.view, p.seq, p.stamp_ns, int(p.kind), axes,
                json.dumps(red.tile, sort_keys=True) if red.tile is not None else "",
                int(red.colormap), int(red.flags), float(red.vmin), float(red.vmax),
                stage.stats.data_ptr() if red.stats is not None else 0, int(red.stats_count),
                int(red.points_dims or 0), int(red.label_count if red.label_count is not None else -1),
                int(_DTYPE[str(data.dtype)]), list(data.shape), data.data_ptr(),
                data.numel() * data.element_size(),
                aux.data_ptr() if aux is not None else 0, aux.numel() if aux is not None else 0,
                json.dumps(p.cadence, sort_keys=True) if p.cadence else "")
        stage.busy = False
        self.counters["published"] += 1


class NativeServer(NetServer):
    """caliper.serve() on caliper._netio. The same public object as
    :class:`caliper._net.NetServer` (url, port, token, stream_url, session,
    closed, stats(), on_event(), events(), close())."""

    def __init__(self, port: int | None = 0, *, engine=None, control: bool = True):
        """``port=None`` starts a producer without a listener (caliper.connect()
        before caliper.serve()); :meth:`listen` adds one later. ``control``:
        whether viewers may control the script."""
        from . import _netio
        import secrets
        if port is not None:
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
        self.software = {"name": "caliper-python", "version": __version__, "sender": "native"}
        self.transport = NetTransport()
        self.transport.server = self
        self._idle = True
        self.ring = NetRing(engine, wake=self._wake)
        self.transport.ring = self.ring
        self.nio = _netio.Server(-1 if port is None else port, self.token,
                                 _wire.canonical_json(self.identity).decode(),
                                 _wire.canonical_json(self.software).decode(), _asset_table(),
                                 MAX_CONNECTIONS, SERIES_RETAIN, int(_wire.HANDSHAKE_TIMEOUT_MS),
                                 int(_wire.PING_INTERVAL_MS), int(_wire.IDLE_TIMEOUT_MS))
        self.producer = _NativeProducer(self.ring, self.identity, self.transport, self.nio)
        self.transport.producer = self.producer
        from ._net_controls import ControlsPort
        self.nio.enable_controls(bool(control))
        self.producer.controls = ControlsPort(self, control, native=True)
        self._events_thread = None
        self.port = 0
        self.url = self.stream_url = None
        self.hub = None
        if port is not None:
            self._set_urls()
        self._thread = threading.Thread(target=self._run, name="caliper.net", daemon=True)
        self._thread.start()

    def _set_urls(self) -> None:
        self.port = self.nio.port
        self.url = f"http://127.0.0.1:{self.port}/?token={self.token}"
        self.stream_url = f"ws://127.0.0.1:{self.port}{_wire.PATH}?{_wire.TOKEN_PARAM}={self.token}"

    @property
    def listening(self) -> bool:
        return self.nio.listening

    def listen(self, port: int = 0, control: bool = True) -> None:
        """Start listening for viewers (a producer that connect() started)."""
        self.producer.controls.allow = bool(control)
        self.nio.enable_controls(bool(control))
        if not self.nio.listening:
            self.nio.listen(int(port))
        self._set_urls()

    # -- the connection to a hub ---------------------------------------------------------
    def connect_hub(self, url: str) -> "HubConnection":
        """Connect out to a hub at ``url`` (ws://127.0.0.1:<port>/v3/stream?token=<token>),
        reconnecting with back-off on a loss; never blocks."""
        self.nio.connect_upstream(url, _wire.canonical_json(self.identity).decode())
        self.hub = HubConnection(self, url)
        return self.hub

    # -- the pump thread -------------------------------------------------------------------
    def _wake(self) -> None:
        """From the training thread: something new for the pump."""
        if self._idle and not self.closed:
            self._idle = False
            self.nio.wake()

    def _run(self) -> None:
        p = self.producer
        nio = self.nio
        p.pump_once()                                 # the view list before any viewer
        while not self.closed:
            try:
                wait = p.wait_ms()
                self._idle = wait > POLL_MS               # a training thread's wake ends the wait
                if self._idle and p.busy():
                    # queued between the check and the flag, by a call whose wake
                    # saw the pump busy: poll instead of waiting
                    self._idle = False
                    wait = POLL_MS
                msgs = nio.control(wait)
                self._idle = False
                for kind, conn, frame in msgs:
                    self._on_control(kind, conn, frame)
                p.pump_once()
                p.controls.flush()
            except Exception:                         # a bug: count it, keep serving
                self.counters["pump_errors"] += 1
                if self.closed:
                    return

    def _on_control(self, kind: str, conn: int, frame: bytes) -> None:
        p = self.producer
        if kind == "opened":
            link = _NativeLink(p, conn)
            if frame:                                 # what the peer announced (3.1)
                info = json.loads(frame)
                link.caps = frozenset(c for c in (_wire.CAP_CONTROLS, _wire.CAP_SELECTION,
                                                  _wire.CAP_SCENES, _wire.CAP_PROGRAMS,
                                                  _wire.CAP_INTERVENTIONS) if info.get(c))
                link.may_control = bool(info.get("control"))
            p.by_conn[conn] = link
            p.links.add(link)
            self.counters["connections"] += 1
            return
        link = p.by_conn.get(conn)
        if link is None:
            return
        if kind == "attached":
            if not link.attached:
                link.attached = True
                p.attached_changed(+1)
                p.controls.on_attach(link)
        elif kind == "detached":
            link.drop_all()
            if link.attached:
                link.attached = False
                p.attached_changed(-1)
        elif kind == "closed":
            p.by_conn.pop(conn, None)
            p.links.discard(link)
            link.close()
        elif kind == "msg":
            try:
                link.handle(frame)
            except Exception:                         # validated natively; nothing to close here
                self.counters["control_errors"] += 1

    # -- inspection ---------------------------------------------------------------------
    def stats(self) -> dict:
        n = self.nio.stats()
        with self.ring.lock:
            views = {name: {"id": v.id, "kind": v.kind, "groups": len(v.groups), "seq": v.seq}
                     for name, v in self.ring.views.items()}
        return {"url": self.url, "port": self.port, "session": self.session, "sender": "native",
                "closed": self.closed, "attached": self.transport.n_attached,
                "connections": n["connections"], "views": views, "ring": dict(self.ring.counters),
                "server": {**dict(self.counters), "http_requests": n["http_requests"],
                           "refused": n["refused"]},
                "producer": {**dict(self.producer.counters), "native_published": n["published"],
                             "overflow_closed": n["overflow_closed"], "encode_failed": n["encode_failed"]},
                "series": n["series"]}

    # -- lifetime ------------------------------------------------------------------------
    def close(self) -> None:
        """Deliver what is pending (at most about a second), end every
        connection with BYE and stop the server. Idempotent."""
        if self.closed:
            return
        if self._stream_file is not None:
            self._stream_file.remove()          # the app stops listing it before it goes
        from ._net import _unroute
        if threading.current_thread() is not self._thread:
            try:
                self.ring.flush(timeout=1.0)
                end = time.monotonic() + 1.0
                while self.producer.busy() and time.monotonic() < end:
                    time.sleep(0.002)
            except Exception:
                pass
        self.closed = True
        self.ring.close()
        self.transport.n_attached = 0
        _unroute(self)
        self.nio.close()                              # BYE, close 1001, joins the native threads
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(5)
        q = self.producer._event_q
        if q is not None:
            q.put(None)

    def __repr__(self) -> str:
        state = "closed" if self.closed else "running"
        from ._net import redact
        return f"<caliper.NetServer (native sender) {redact(self.url)} {state}>"


class HubConnection:
    """The connection of this process to a hub, returned by [`connect`][caliper.connect].

    Attributes:
        url (str): The stream URL of the hub, without the token.
    """

    def __init__(self, producer: NativeServer, url: str):
        from urllib.parse import parse_qs, urlsplit
        self._p = producer
        self.url = url.split("?", 1)[0]
        self._token = parse_qs(urlsplit(url).query).get(_wire.TOKEN_PARAM, [""])[0]

    def _state(self) -> dict:
        from ._net import redact
        d = self._p.nio.upstream()
        if d.get("error"):                           # a native error may quote the link
            d["error"] = redact(d["error"], self._token)
        return d

    @property
    def state(self) -> str:
        """The state of the connection.

        One of `connecting`, `connected`, `waiting` (to try again), `replaced` (another connection
        of this session took over) or `closed`.
        """
        return self._state()["state"]

    @property
    def connected(self) -> bool:
        """`True` while the connection is open."""
        return self._state()["connected"]

    @property
    def session(self) -> str:
        """The session id that the hub assigned.

        Until the hub answers, it is the proposed id `clp-<pid>-<6 hex>`.
        """
        return self._state()["session"] or self._p.identity["id"]

    def stats(self) -> dict:
        """Return the state of the connection as a dict.

        Returns:
            dict: `state`, `connected`, `session`, `error` (the last failure, or empty), `attempts`
                and `connects`.
        """
        d = self._state()
        d.pop("url", None)
        return d

    def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait, with polls, until the connection is open.

        Use it in tests and scripts, not in the training loop.

        Args:
            timeout (float): The longest wait, in seconds.

        Returns:
            bool: `True` if the connection is open.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.connected:
                return True
            time.sleep(0.01)
        return self.connected

    def close(self) -> None:
        """End the connection to the hub.

        A running `serve()` keeps serving. `caliper.close()` ends the connection as well.
        """
        self._p.nio.disconnect_upstream()
        if self._p.hub is self:
            self._p.hub = None
        if not self._p.listening:                    # nothing else to serve
            self._p.close()

    def __repr__(self) -> str:
        return f"<caliper.HubConnection {self.url} {self.state} session={self.session}>"
