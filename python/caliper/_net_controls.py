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
* sends CONTROLS, SELECTION and the later replies of actions from the
  server's own thread, coalesced: of the CONTROLS (and of each item space's
  SELECTION) made between two sends, only the newest goes out;
* closes the network ring's open tick on the training thread right before a
  pause waits, so that the viewers show the step at which it paused.

Nothing here runs on the training thread per snapshot or per step: a change
of the controls stores a message and wakes the server's thread.
"""
from __future__ import annotations

import threading

from . import _controls, _wire

__all__ = ["ControlsPort", "capabilities"]

_CAPS = frozenset((_wire.CAP_CONTROLS, _wire.CAP_SELECTION))


def capabilities(controls: bool) -> list:
    """The capabilities a producer announces."""
    if not controls:
        return [_wire.CAP_EVENTS]
    return [_wire.CAP_CONTROLS, _wire.CAP_EVENTS, _wire.CAP_SELECTION]


def peer_caps(announced) -> frozenset:
    """The 3.1 capabilities among those a peer announced."""
    return _CAPS & frozenset(announced or ())


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
        self.route = _controls.attach(self)
        _controls.CONTROLS.wait_hooks.append(self._before_wait)

    # -- what caliper._controls calls (any thread) ---------------------------------------
    def broadcast(self, type_, meta) -> None:
        with self._lock:
            if int(type_) == _wire.MsgType.CONTROLS:
                self._controls = meta
            else:
                self._selections[meta["space"]] = meta
        self._kick()

    def reply(self, conn, meta) -> None:
        with self._lock:
            self._replies.append((conn, meta))
        self._kick()

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
                frame = _wire.encode_frame(type_, meta)
                if self.native:
                    self.server.nio.send_after_views(link.conn, frame)
                else:
                    self._send(link, frame)           # the VIEWS marker is already queued

    def annotate(self, entries: list) -> list:
        """VIEWS entries with their item spaces, for a peer with ``selection``."""
        return [self.route.annotate_view(e) for e in entries]

    def flush(self) -> None:
        """Send what the controls stored: CONTROLS and SELECTION to the
        attached peers that announced the capability, then the later replies."""
        with self._lock:
            self._scheduled = False
            controls, self._controls = self._controls, None
            selections, self._selections = self._selections, {}
            replies, self._replies = self._replies, []
        if controls is None and not selections and not replies:
            return
        links = [ln for ln in list(self.server.producer.links) if ln.attached and not ln.closed]
        if controls is not None:
            frame = _wire.encode_frame(_wire.MsgType.CONTROLS, controls)
            for ln in links:
                if _wire.CAP_CONTROLS in ln.caps:
                    self._send(ln, frame)
        for meta in selections.values():
            frame = _wire.encode_frame(_wire.MsgType.SELECTION, meta)
            for ln in links:
                if _wire.CAP_SELECTION in ln.caps:
                    self._send(ln, frame)
        for ln, meta in replies:
            if not ln.closed:
                self._send(ln, _wire.encode_frame(_wire.MsgType.CONTROL_REPLY, meta))

    def detach(self) -> None:
        self.route.detach()
        try:
            _controls.CONTROLS.wait_hooks.remove(self._before_wait)
        except ValueError:
            pass

    # -- inside ---------------------------------------------------------------------------
    def _send(self, link, frame: bytes) -> None:
        if self.native:
            self.server.nio.send(link.conn, frame)
        else:
            link.control.append(frame)
            link.wake.set()

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
        """On the training thread, before a pause waits: the open tick goes out."""
        if not self.server.closed:
            self.server.ring.flush(timeout=0)
