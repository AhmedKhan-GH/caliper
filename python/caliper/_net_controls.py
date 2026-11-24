"""The controls of a running script on the network route (wire protocol 3.1).

caliper._controls holds the run state, the parameters, the actions and the
selections, and knows no transport. :class:`ControlsPort` is the transport it
attaches for one stream server (``caliper.serve()``, ``caliper.connect()``),
on either sender: the pure-Python one (caliper/_net.py), whose connections
are :class:`caliper._net.Link` objects on the asyncio thread, and the native
one (caliper/_net_native.py), whose connections are ``_NativeLink`` objects
that the pump thread answers for ``caliper._netio``.

What it does for the server:

* announces the capabilities ``controls`` and ``selection`` and tells a viewer
  that announced ``controls`` whether it may control (``serve(control=...)``;
  for the hub connection, the hub's WELCOME says);
* hands every CONTROL to ``route.on_control`` and sends its CONTROL_REPLY, and
  answers a CONTROL from a peer that did not announce ``controls`` with ERROR
  ``bad_request``;
* hands every select EVENT to ``route.on_select``, with the payload axes of
  the subscription it names, as well as to Python's event handlers as before;
* sends a late viewer CONTROLS and the current SELECTIONs right after the
  VIEWS of its attachment, and gives a peer with ``selection`` the VIEWS with
  item spaces;
* counts the bytes of the frames posted for each connection until the
  server's thread queues them (``unsent``), which the programs add to the
  connection's backlog before they post more;
* sends CONTROLS, SELECTION and the later replies of actions from the
  server's own thread, coalesced: of the CONTROLS (and of each item space's
  SELECTION) made between two sends, only the newest goes out;
* at a pause, closes the network ring's open tick and moves its gathered
  series values on the training thread, waiting for their copies, and holds
  the CONTROLS that announces the pause until no connection has a snapshot or
  series values left to send (at most ORDER_WAIT_S): on every connection the
  run state then follows the data of the step at which the run paused. A
  newer CONTROLS while one is held replaces it, and the hold ends when it is
  sent, so the run state never overtakes that data and later changes are not
  held back.

Nothing here runs on the training thread per snapshot or per step: a change
of the controls stores a message and wakes the server's thread.
"""
from __future__ import annotations

import threading
import time

from . import _controls, _net_programs, _wire

__all__ = ["ControlsPort", "capabilities"]

_CAPS = frozenset((_wire.CAP_CONTROLS, _wire.CAP_SELECTION, _wire.CAP_PROGRAMS,
                   _wire.CAP_INTERVENTIONS, _wire.CAP_INTERPRETATION, _wire.CAP_TEXT_PARAMS))
ORDER_WAIT_S = _wire.ORDER_WAIT_MS / 1000    # the longest a pause's run state waits for data
COPY_WAIT_S = 1.0                            # the longest a pause waits for the device's copies
RECHECK_S = 0.002                            # how often a held run state looks again


def capabilities(controls: bool) -> list:
    """The capabilities a producer announces: with controls, also ``programs`` and
    ``interventions`` while a ``caliper.watch(model, optimizer)`` is on."""
    if not controls:
        return [_wire.CAP_EVENTS]
    return sorted([_wire.CAP_CONTROLS, _wire.CAP_EVENTS, _wire.CAP_SELECTION, _wire.CAP_TEXT_PARAMS]
                  + _net_programs.capabilities()
                  + ([_wire.CAP_INTERPRETATION] if _controls.CONTROLS.interpretation_service is not None else []))


def peer_caps(announced) -> frozenset:
    """The 3.1 capabilities, ``programs`` and ``interventions`` among those a peer announced."""
    return _CAPS & frozenset(announced or ())


def _for_peer(meta: dict, link) -> dict:
    """A CONTROLS meta for ``link``: without the key ``interventions`` when the peer did not
    announce the capability (any other meta unchanged)."""
    out = meta
    if "interventions" in meta and _wire.CAP_INTERVENTIONS not in link.caps:
        out = {k: v for k, v in out.items() if k != "interventions"}
    if "params" in out and _wire.CAP_TEXT_PARAMS not in link.caps:
        out = {**out, "params": [p for p in out["params"] if p["type"] != _wire.PARAM_TEXT]}
    return out



class ControlsPort:
    """The transport of caliper._controls for one server (see the module
    docstring). ``allow``: whether viewers may control. ``native``: the
    server is caliper._net_native.NativeServer."""

    def __init__(self, server, allow: bool, native: bool):
        self.server = server
        self.allow = bool(allow)
        self.native = native
        self._lock = threading.Lock()
        self._controls = None                     # the newest CONTROLS not sent yet
        self._selections: dict = {}               # space -> the newest SELECTION not sent yet
        self._replies: list = []                  # (link, meta) of later replies
        self._scheduled = False                   # the Python sender: a flush is on its way
        self._after_data = False                  # the stored CONTROLS announces a pause
        self._held = None                         # the CONTROLS that waits for data to go out
        self._held_until = 0.0
        self._recheck = False                     # the Python sender: a look again is on its way
        self._posts: list = []                    # (link or None, frame, capability) to send
        self._unsent: dict = {}                   # link -> the bytes of its posts not queued yet
        self._unsent_all: dict = {}               # capability -> the same, of posts to every peer
        self.route = _controls.attach(self)
        self.programs = _net_programs.ProgramsPort(self)
        _controls.CONTROLS.wait_hooks.append(self._before_wait)

    # -- what caliper._controls calls (any thread) ---------------------------------------
    def broadcast(self, type_, meta, after_data: bool = False) -> None:
        with self._lock:
            if int(type_) == _wire.MsgType.CONTROLS:
                self._controls = meta
                self._after_data = self._after_data or after_data
            else:
                self._selections[meta["space"]] = meta
        self._kick()

    def reply(self, conn, meta) -> None:
        with self._lock:
            self._replies.append((conn, meta))
        self._kick()

    def post(self, link, frame: bytes, cap: str) -> None:
        """Send ``frame`` to ``link``, or to every attached peer that announced ``cap`` when
        ``link`` is None, on the server's thread (from any thread)."""
        with self._lock:
            self._posts.append((link, frame, cap))
            self._count(link, cap, len(frame))
        self._kick()

    def unsent(self, link) -> int:
        """The bytes of the frames posted for ``link`` that the server's thread has not queued
        on it yet: a sender that bounds the queue of a peer adds them to its backlog, so that
        several posts before one flush are bounded together (from any thread)."""
        with self._lock:
            n = self._unsent.get(link, 0)
            for cap, b in self._unsent_all.items():
                if cap in getattr(link, "caps", ()):
                    n += b
            return n

    def may_control(self, conn) -> bool:
        return bool(getattr(conn, "may_control", False)) and not getattr(conn, "closed", True)

    def controllers(self) -> int:
        return sum(1 for ln in list(self.server.producer.links)
                   if getattr(ln, "may_control", False) and not ln.closed)

    def views_changed(self) -> None:
        self.server.producer.views_version = -1
        self.server._wake()

    # -- what the server calls (its own thread) ---------------------------------------------
    def on_control(self, link, msg: _wire.Control) -> None:
        """A CONTROL from ``link``."""
        if _wire.CAP_CONTROLS not in link.caps:
            link.queue(_wire.Error(_wire.ERR_BAD_REQUEST,
                                   "a CONTROL needs the capability controls in HELLO",
                                   in_reply_to=int(_wire.MsgType.CONTROL),
                                   request_id=msg.request_id))
            return
        if msg.op == _wire.OP_INTERPRET and _wire.CAP_INTERPRETATION not in link.caps:
            link.queue(_wire.ControlReply(msg.request_id, msg.op, _wire.STATUS_ERROR,
                                          error={"code": "interpret.unsupported",
                                                 "message": "The peer did not announce interpretation."}))
            return
        try:
            reply = self.route.on_control(link, msg.to_meta())
        except _wire.MessageError as e:
            link.queue(_wire.Error(e.code, e.message, in_reply_to=int(_wire.MsgType.CONTROL),
                                   request_id=msg.request_id))
            return
        self._send(link, _wire.encode_frame(_wire.MsgType.CONTROL_REPLY, reply))

    def on_select(self, link, msg: _wire.Event) -> None:
        """A select EVENT from ``link`` (it also goes to Python's handlers)."""
        axes = None
        d = msg.detail
        if msg.sub is not None and "space" not in d and ("indices" in d or "rows" in d
                                                          or "cols" in d):
            axes = self._axes(link, msg.sub)
        self.route.on_select(link, {"view": msg.view, "type": msg.type, "detail": d}, axes)

    def on_attach(self, link) -> None:
        """``link`` was attached: its state follows the VIEWS of the attachment."""
        for type_, meta in self.route.snapshot_state():
            cap = _wire.CAP_CONTROLS if type_ == _wire.MsgType.CONTROLS else _wire.CAP_SELECTION
            if cap in link.caps:
                frame = _wire.encode_frame(type_, _for_peer(meta, link))
                if self.native:
                    self.server.nio.send_after_views(link.conn, frame)
                else:
                    self._send(link, frame)           # the VIEWS marker is already queued
        self.programs.on_attach(link)

    def annotate(self, entries: list) -> list:
        """VIEWS entries with their item spaces, for a peer with ``selection``."""
        return [self.route.annotate_view(e) for e in entries]

    def holding(self) -> bool:
        """A pause's run state waits for data to go out (the server's thread
        then looks again soon)."""
        return self._held is not None

    def pending(self) -> bool:
        """Something that the controls stored is not handed to the connections yet."""
        with self._lock:
            return (self._controls is not None or self._held is not None or bool(self._replies)
                    or bool(self._selections) or bool(self._posts))

    def flush(self, final: bool = False) -> None:
        """Send what the controls stored: CONTROLS and SELECTION to the
        attached peers that announced the capability, then the later replies.
        A CONTROLS that announces a pause, and any newer one, waits until no
        connection has data left to send (see the module docstring), except
        in the ``final`` flush of the server's close, which sends it at once."""
        with self._lock:
            self._scheduled = False
            controls, self._controls = self._controls, None
            selections, self._selections = self._selections, {}
            replies, self._replies = self._replies, []
            posts, self._posts = self._posts, []
            if controls is not None and (self._after_data or self._held is not None):
                if self._held is None:
                    self._held_until = time.monotonic() + ORDER_WAIT_S
                self._held, controls = controls, None
                self._after_data = False
        if self._held is not None:
            controls = self._release(final)
        if controls is None and not selections and not replies and not posts:
            return
        links = [ln for ln in list(self.server.producer.links) if ln.attached and not ln.closed]
        if controls is not None:
            frames = {}                               # with or without "interventions"
            for ln in links:
                if _wire.CAP_CONTROLS in ln.caps:
                    meta = _for_peer(controls, ln)
                    key = (_wire.CAP_INTERVENTIONS in ln.caps, _wire.CAP_TEXT_PARAMS in ln.caps)
                    if key not in frames:
                        frames[key] = _wire.encode_frame(_wire.MsgType.CONTROLS, meta)
                    self._send(ln, frames[key])
        for meta in selections.values():
            frame = _wire.encode_frame(_wire.MsgType.SELECTION, meta)
            for ln in links:
                if _wire.CAP_SELECTION in ln.caps:
                    self._send(ln, frame)
        for ln, meta in replies:
            if not ln.closed:
                self._send(ln, _wire.encode_frame(_wire.MsgType.CONTROL_REPLY, meta))
        try:
            for ln, frame, cap in posts:
                for target in ([ln] if ln is not None else links):
                    if not target.closed and cap in target.caps:
                        self._send(target, frame)
        finally:
            if posts:                                 # now in the links' queues, or dropped
                with self._lock:
                    for ln, frame, cap in posts:
                        self._count(ln, cap, -len(frame))

    def _release(self, final: bool = False):
        """The held CONTROLS once the data sent before it has gone out (or the
        wait is over, or the server closes), else None and a look again soon.
        The server's thread."""
        self.server.producer.pump_once()          # the paused step's data, published first
        if not final and self._data_pending() and time.monotonic() < self._held_until:
            if not self.native and not self._recheck:
                self._recheck = True
                self.server._loop.call_later(RECHECK_S, self._look_again)
            return None
        held, self._held, self._held_until = self._held, None, 0.0
        return held

    def _look_again(self) -> None:
        self._recheck = False
        self.flush()

    def _data_pending(self) -> bool:
        """A connection has a snapshot or series values it has not sent yet."""
        if self.native:
            return self.server.nio.data_pending()
        p = self.server.producer
        for ln in list(p.links):
            if ln.closed:
                continue
            for s in list(ln.subs.values()):
                if s.snap is not None:
                    return True
                if s.series is not None and s.next < p.history(s.series).total:
                    return True
        return False

    def detach(self) -> None:
        self.programs.close()
        self.route.detach()
        try:
            _controls.CONTROLS.wait_hooks.remove(self._before_wait)
        except ValueError:
            pass

    # -- inside ---------------------------------------------------------------------------
    def _count(self, link, cap: str, n: int) -> None:
        """Adds ``n`` to the bytes not queued yet for ``link``, or for every peer with ``cap``
        when ``link`` is None (under the lock)."""
        table, key = (self._unsent_all, cap) if link is None else (self._unsent, link)
        left = table.get(key, 0) + n
        if left > 0:
            table[key] = left
        else:
            table.pop(key, None)

    def _send(self, link, frame: bytes) -> None:
        if self.native:
            self.server.nio.send(link.conn, frame)
        else:
            link.push(frame)

    def _kick(self) -> None:
        """Make the server's thread flush soon (from any thread)."""
        if self.native:
            self.server.nio.wake()
            return
        with self._lock:
            if self._scheduled:
                return
            self._scheduled = True
        try:
            self.server._loop.call_soon_threadsafe(self.flush)
        except RuntimeError:                       # the loop has closed
            pass

    def _axes(self, link, sub: int):
        """The payload axes of subscription ``sub`` of ``link``, or None."""
        s = link.subs.get(sub)
        if s is None:
            return None
        ring = self.server.ring
        with ring.lock:
            v = ring.views.get(s.view)
            entry = v.entry() if v is not None else None
        if entry is None:
            return None
        try:
            return _wire.payload_layout(entry, s.resolved)["axes"]
        except Exception:
            return None

    def _before_wait(self) -> None:
        """On the training thread, before a pause waits and is announced: the
        open tick and the gathered series values go out, and their copies
        complete, so that the server's thread can send them first."""
        if not self.server.closed:
            self.server.ring.flush(timeout=COPY_WAIT_S)
