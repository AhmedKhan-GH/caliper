"""Optional local request channel for a live session. See docs/live-requests.md.

Socket I/O runs on a reader thread; handlers run serially on serve()'s caller
(the model's thread). Stop and disconnect set a cancellation event immediately.
No model code ever runs on the socket thread. No files contain tensor data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import socket
import stat
import threading

MAX_MESSAGE = 4 * 1024 * 1024


def socket_path(session: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,20}", session):
        raise ValueError("invalid session token")
    return f"/tmp/caliper-{os.getuid()}/{session}.sock"


def encode(message: dict) -> bytes:
    data = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("message too large")
    return data


class _Peer:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.cancel = threading.Event()

    def send(self, message):
        self.send_frame(encode(message))

    def send_frame(self, frame):
        with self.lock:
            try:
                self.sock.sendall(frame)
            except OSError:
                # sendall may already have written part of a JSON frame. It is
                # unsafe to append another reply to that incomplete frame.
                self.close()
                raise

    def close(self):
        self.cancel.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def _reject_constant(value):
    raise ValueError(f"invalid JSON number: {value}")


class RequestServer:
    """One controller, one outstanding operation; stop is always accepted.

    handler(request, emit, cancelled) returns a JSON object. emit(dict) sends
    progress; the return sends the final result. Exceptions become error replies.
    Disconnect cancels the active operation. Requests are never replayed.
    """
    def __init__(self, session: str, capabilities=()):
        self.path = socket_path(session)
        directory = Path(self.path).parent
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("request socket directory must be owned by this user with mode 0700")
        self.capabilities = list(capabilities)
        if any(not isinstance(capability, str) for capability in self.capabilities):
            raise TypeError("capabilities must be strings")
        self.closed = threading.Event()
        self.jobs = queue.Queue(maxsize=1)
        self.state_lock = threading.Lock()
        self.busy = False
        self.peer = None
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            self.listener.bind(self.path)  # never unlink another server's socket
            bound = True
            os.chmod(self.path, 0o600)
            self.listener.listen(1)
            self.listener.settimeout(0.2)
            self.thread = threading.Thread(target=self._listen, name="caliper requests", daemon=True)
            self.thread.start()
        except BaseException:
            self.listener.close()
            if bound:
                Path(self.path).unlink(missing_ok=True)
            raise

    def _listen(self):
        while not self.closed.is_set():
            try:
                sock, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # Allow a large reply to drain across several app frames, while a
            # controller that stops reading still cannot retain the worker.
            sock.settimeout(2.0)
            peer = _Peer(sock)
            self.peer = peer
            try:
                self._read(peer)
            except (OSError, ValueError, UnicodeError, RecursionError):
                pass
            finally:
                peer.close()
                self.peer = None

    def _read(self, peer):
        buffer = b""
        while not self.closed.is_set():
            try:
                data = peer.sock.recv(65536)
            except socket.timeout:
                continue
            if not data:
                return
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if len(line) >= MAX_MESSAGE:
                    raise ValueError("message too large")
                request = json.loads(line, parse_constant=_reject_constant)
                if (not isinstance(request, dict) or type(request.get("id")) is not int
                        or not 0 <= request["id"] <= 2**53 or not isinstance(request.get("op"), str)):
                    raise ValueError("expected id and op")
                rid, op = request["id"], request["op"]
                if op == "hello":
                    peer.send(dict(id=rid, done=True, result=dict(protocol=1, capabilities=self.capabilities)))
                elif op == "stop":
                    peer.cancel.set()
                    peer.send(dict(id=rid, done=True, result=dict(stopping=True)))
                else:
                    with self.state_lock:
                        if self.busy:
                            peer.send(dict(id=rid, done=True, error="busy; stop or wait for the current operation"))
                        else:
                            self.busy = True
                            peer.cancel.clear()
                            self.jobs.put_nowait((peer, request))
            if len(buffer) >= MAX_MESSAGE:
                raise ValueError("message too large")

    def serve(self, handler):
        while not self.closed.is_set():
            try:
                peer, request = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            rid = request["id"]
            def emit(event):
                if peer.cancel.is_set():
                    return
                peer.send(dict(id=rid, done=False, event=event))
            try:
                if peer.cancel.is_set():
                    result = dict(cancelled=True)
                else:
                    result = handler(request, emit, peer.cancel)
                frame = encode(dict(id=rid, done=True, result=result))
            except Exception as exc:
                frame = encode(dict(id=rid, done=True, error=str(exc)[:2048]))
            # Publish idle before the final reply, under the same lock used by
            # admission: a controller may send its next request immediately.
            with self.state_lock:
                self.busy = False
                try:
                    peer.send_frame(frame)
                except OSError:
                    peer.cancel.set()

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        peer = self.peer
        if peer:
            peer.close()
        self.listener.close()
        self.thread.join(timeout=1)
        Path(self.path).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
