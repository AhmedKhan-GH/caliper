"""The wire path of GPU programs in the script (``design/gpu-programs.md``, Wire Path).

A viewer that announced the capability ``programs`` sends PROGRAM to install, replace or remove
one program. This module checks the message, gives it to the facade ``caliper._programs`` and
answers with PROGRAM_STATE, the status item of each program. The script announces ``programs``
and ``interventions`` only while a ``caliper.watch(model, optimizer)`` is on (:func:`enable`).

THREADS. A PROGRAM arrives on the server's thread (the asyncio loop of the Python sender, the pump
thread of the native sender). The checks run there, and the install or the removal runs in order
on one daemon thread, ``caliper.programs``, so that a compile never holds the network. The answers
go back through the controls port's queue (``ControlsPort.post``) and leave on the server's
thread. The training thread never waits on this module.

STATE. The thread also looks at the facade's status every ``1 / PROGRAM_STATE_RATE_HZ`` seconds
while a program is installed. It sends an unrequested PROGRAM_STATE to each attached viewer with
``programs`` when a state changed, and at that rate while a program runs. When no viewer with
``programs`` stays connected for ``LINGER_S`` seconds, it removes every program, because nobody
can see their outputs and the app sends each program again when it attaches.

FAILURE STAYS LOCAL. A facade that cannot load the runtime, or raises for any reason, makes a
refused status item with the code ``program.unavailable``; nothing here raises into the server.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
import weakref

from . import _wire

__all__ = ["enable", "disable", "enabled", "capabilities", "ProgramsPort", "LINGER_S"]

LINGER_S = 5.0                     # programs outlive the last viewer with ``programs`` this long
UNAVAILABLE = "program.unavailable"
_PERIOD_S = 1.0 / _wire.PROGRAM_STATE_RATE_HZ

_lock = threading.Lock()
_watches = 0                       # the watch(model, optimizer) calls that are on
_ports: "weakref.WeakSet[ProgramsPort]" = weakref.WeakSet()


def _native_caps(on: bool) -> None:
    """Tell the native sender, when it is loaded, whether to announce both capabilities."""
    try:
        from . import _netio
    except ImportError:
        return
    setter = getattr(_netio, "set_watch_capabilities", None)
    if setter is not None:
        setter(bool(on))


def enable() -> None:
    """A ``caliper.watch(model, optimizer)`` started: announce ``programs`` and
    ``interventions`` to the peers that connect from now on."""
    global _watches
    with _lock:
        _watches += 1
        first = _watches == 1
    if first:
        _native_caps(True)


def disable() -> None:
    """A ``caliper.watch(model, optimizer)`` ended; the last one stops the announcement."""
    global _watches
    with _lock:
        if _watches == 0:
            return
        _watches -= 1
        last = _watches == 0
    if last:
        _native_caps(False)


def enabled() -> bool:
    """Whether the script announces ``programs`` and ``interventions``."""
    return _watches > 0


def capabilities() -> list:
    """The capabilities of this module that a producer announces now."""
    return [_wire.CAP_INTERVENTIONS, _wire.CAP_PROGRAMS] if enabled() else []


def _facade():
    from . import _programs
    return _programs


def _refused(program_id: str, kind: str, code: str, message: str, sha256: str = "") -> dict:
    return {"id": program_id, "kind": kind, "state": "refused", "every": None,
            "every_min": None, "last_pass_ms": None, "last_cost_ms": None, "step_ms": None,
            "overhead_share": None, "passes_per_s": None,
            "last_step": None, "runs": 0, "memory_bytes": 0, "memory_limit_bytes": 0,
            "error": {"code": code, "message": message[:_wire.TEXT_MAX], "line": 0,
                      "column": 0},
            "warning": None,
            "sha256": sha256}


def _step() -> int | None:
    try:
        from . import _controls
        n = int(_controls.CONTROLS.n)
    except Exception:
        return None
    return n if 0 <= n <= _wire._SAFE_INT else None


class ProgramsPort:
    """The wire path of programs for one server, beside its controls port.

    Args:
        controls: The server's ``caliper._net_controls.ControlsPort``, whose queue and thread
            carry the answers.
    """

    def __init__(self, controls):
        self.controls = controls
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._local: dict = {}                    # id -> a status item refused here
        self._installed: set = set()              # ids given to the facade
        self._last_sent = None                    # the states of the last unrequested message
        self._next_tick = 0.0
        self._alone_since = None                  # no viewer with programs since then
        self.closed = False                       # close() ran; no new job starts the thread
        self.counters = {"programs_in": 0, "states_out": 0, "refused": 0}
        _ports.add(self)

    # -- the server's thread ------------------------------------------------------------
    def on_program(self, link, msg: _wire.Program) -> None:
        """A PROGRAM from ``link``: refuse it at once, or queue it for the program thread."""
        self.counters["programs_in"] += 1
        code, text = None, None
        if _wire.CAP_PROGRAMS not in link.caps or not enabled():
            code, text = _wire.ERR_UNSUPPORTED, ("a PROGRAM needs the capability programs in "
                                                 "HELLO and caliper.watch(model, optimizer)")
        elif not self.controls.may_control(link):
            code, text = _wire.ERR_FORBIDDEN, "this server does not allow control"
        elif msg.op == _wire.PROGRAM_SET and msg.piece is None and (
                hashlib.sha256(msg.wgsl.encode("utf-8")).hexdigest() != msg.sha256):
            code, text = _wire.ERR_BAD_REQUEST, "'sha256' is not the SHA-256 of 'wgsl'"
        if code is not None:
            self.counters["refused"] += 1
            link.queue(_wire.Error(code, text, in_reply_to=int(_wire.MsgType.PROGRAM),
                                   request_id=msg.request_id))
            return
        self._jobs.put((link, msg))
        self._start()

    def on_attach(self, link) -> None:
        """``link`` was attached: it learns which programs run, so that the app can remove the
        ones its spec no longer holds."""
        if _wire.CAP_PROGRAMS in link.caps and enabled():
            self._jobs.put((link, None))
            self._start()

    # -- the program thread -------------------------------------------------------------
    def _start(self) -> None:
        if not self.closed and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._run, name="caliper.programs",
                                            daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                link, msg = self._jobs.get(timeout=_PERIOD_S if self._installed else 5.0)
            except queue.Empty:
                if self._installed or self._local:
                    self._tick()
                continue
            if link is None:                         # close(): the last job
                self._remove_all()
                return
            try:
                if msg is not None:
                    self._apply(msg)
                self._answer(link, None if msg is None else msg.request_id)
            except Exception:                        # a bug: never into the server
                self.counters["refused"] += 1
            self._tick()

    def _apply(self, msg: _wire.Program) -> None:
        self._local.pop(msg.id, None)
        if msg.op == _wire.PROGRAM_REMOVE:
            self._installed.discard(msg.id)
            try:
                _facade().remove(msg.id)
            except Exception:
                pass
            return
        if msg.op == _wire.PROGRAM_PARAMS:
            self._set_params(msg)
            return
        if msg.piece is not None:
            try:
                _facade().install_piece(msg.id, piece=msg.piece, manifest=msg.manifest, bind=msg.bind,
                                        params=msg.params, panel=msg.panel, segments=msg.segments,
                                        common=msg.common or "", derive=msg.derive or "", draw=msg.draw or "")
            except Exception as e:
                self._installed.discard(msg.id)
                pin = str((msg.piece or {}).get("hash") or "").removeprefix("sha256:")
                if not _wire._sha256_hex(pin):
                    pin = ""
                self._local[msg.id] = _refused(msg.id, _wire.STATUS_PIECE, UNAVAILABLE,
                                               f"The script cannot run pieces: {e}", pin)
                return
            self._installed.add(msg.id)
            return
        decl = msg.declaration
        kind = _wire.STATUS_OPERATOR if decl.get("builtin") else decl["output"]["kind"]
        try:
            _facade().install(msg.id, decl, msg.wgsl, scene=msg.scene, sha256=msg.sha256)
        except Exception as e:                       # NotImplementedError until the runtime is built
            self._installed.discard(msg.id)
            self._local[msg.id] = _refused(msg.id, kind, UNAVAILABLE,
                                           f"The script cannot run GPU programs: {e}",
                                           msg.sha256)
            return
        self._installed.add(msg.id)

    def _set_params(self, msg: _wire.Program) -> None:
        """A params op: new values for the uniform of an installed program, read at its next
        pass. The facade ignores an id that no program holds, as for a removal, and the answer then
        shows the programs without it."""
        try:
            _facade().set_params(msg.id, msg.params)
        except Exception:
            pass                                     # a bad name shows in the status, not here

    def items(self) -> list:
        """The status items to send: the facade's, then the ones refused here, at most
        ``MAX_PROGRAMS``, each valid on the wire."""
        try:
            items = [dict(x) for x in _facade().status()] if self._installed else []
        except Exception:
            items = []
        seen = {x.get("id") for x in items}
        items += [x for k, x in self._local.items() if k not in seen]
        out, ids = [], set()
        for x in items:
            x.pop("host", None)
            if _wire._status_item(x) and x["id"] not in ids and len(out) < _wire.MAX_PROGRAMS:
                ids.add(x["id"])
                out.append(x)
        return out

    def _state(self, request_id) -> tuple:
        items = self.items()
        step_ms = next((x["step_ms"] for x in items if x.get("step_ms") is not None), None)
        meta = _wire.ProgramState(items, request_id, _step(), step_ms).to_meta()
        return meta, items

    def _answer(self, link, request_id) -> None:
        meta, items = self._state(request_id)
        self.controls.post(link, _wire.encode_frame(_wire.MsgType.PROGRAM_STATE, meta),
                           _wire.CAP_PROGRAMS)
        self.counters["states_out"] += 1
        self._last_sent = [(x["id"], x["state"], (x.get("warning") or {}).get("code")) for x in items]

    def _tick(self) -> None:
        """At most ``PROGRAM_STATE_RATE_HZ`` times a second: an unrequested PROGRAM_STATE on a
        change of a state or while a program runs, and the removal of every program once no
        viewer with ``programs`` stayed connected for ``LINGER_S``."""
        now = time.monotonic()
        if now < self._next_tick:
            return
        self._next_tick = now + _PERIOD_S
        if not self._watched():
            if self._alone_since is None:
                self._alone_since = now
            elif now - self._alone_since >= LINGER_S:
                self._remove_all()
            return
        self._alone_since = None
        meta, items = self._state(None)
        states = [(x["id"], x["state"], (x.get("warning") or {}).get("code")) for x in items]
        if states != self._last_sent or any(s == "running" for _, s, _ in states):
            self.controls.post(None, _wire.encode_frame(_wire.MsgType.PROGRAM_STATE, meta),
                               _wire.CAP_PROGRAMS)
            self.counters["states_out"] += 1
            self._last_sent = states

    def _watched(self) -> bool:
        for port in list(_ports):
            links = list(getattr(port.controls.server.producer, "links", ()))
            if any(_wire.CAP_PROGRAMS in ln.caps and not ln.closed for ln in links):
                return True
        return False

    def _remove_all(self) -> None:
        for pid in list(self._installed):
            try:
                _facade().remove(pid)
            except Exception:
                pass
        self._installed.clear()
        self._local.clear()
        self._last_sent = None

    def close(self) -> None:
        """Stop the program thread after the jobs before it, and remove the programs of this
        server."""
        self.closed = True
        if self._thread is not None and self._thread.is_alive():
            self._jobs.put((None, None))
