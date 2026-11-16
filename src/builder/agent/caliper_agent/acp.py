"""JSON-RPC 2.0 over stdio, the agent side of ACP version 1's transport.

Each message is one line of UTF-8 JSON. A reader thread reads the client's
lines (so that a blocking read never stalls the event loop, on any platform)
and hands them to the loop; the loop dispatches requests to the agent as tasks,
notifications at once, and responses to the requests the agent sent (the
permission requests). Writes take a lock, so a line is never interleaved.

Only this module writes the protocol stream. Every line is scrubbed before it
is written, as a last guard against a secret reaching the client. Malformed
lines are dropped and reported on stderr; a line longer than 8 MiB is refused.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
from typing import IO, Any, Awaitable, Callable

from .keyguard import scrub

log = logging.getLogger("caliper_agent.acp")

MAX_LINE_BYTES = 8 << 20
SHUTDOWN_GRACE_S = 2.0

# JSON-RPC and ACP error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTH_REQUIRED = -32000


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class PeerClosed(Exception):
    """The client closed the connection."""


class Connection:
    """One JSON-RPC peer over a pair of byte streams."""

    def __init__(self, reader: IO[bytes], writer: IO[bytes]) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbox: asyncio.Queue | None = None
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    # ---- writing --------------------------------------------------------------

    def _write(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        line = scrub(line)
        data = line.encode("utf-8") + b"\n"
        with self._write_lock:
            if self._closed:
                return
            try:
                view = memoryview(data)
                while view:
                    written = self._writer.write(view)
                    if written is None:  # a non-blocking stream that is full: try again
                        continue
                    view = view[written:]
                self._writer.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._closed = True

    def notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def respond(self, rpc_id: Any, result: Any) -> None:
        self._write({"jsonrpc": "2.0", "id": rpc_id, "result": result})

    def respond_error(self, rpc_id: Any, code: int, message: str, data: Any = None) -> None:
        error: dict = {"code": code, "message": scrub(message)}
        if data is not None:
            error["data"] = data
        self._write({"jsonrpc": "2.0", "id": rpc_id, "error": error})

    async def request(self, method: str, params: dict) -> Any:
        """Sends a request to the client and waits for its result.

        Raises RpcError when the client answers with an error, PeerClosed when
        the connection ends first, and CancelledError when the caller is
        cancelled (the answer, if it comes, is then dropped).
        """
        if self._loop is None:
            raise PeerClosed()
        rpc_id = next(self._ids)
        future = self._loop.create_future()
        self._pending[rpc_id] = future
        try:
            self._write({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(rpc_id, None)

    # ---- reading --------------------------------------------------------------

    def _read_lines(self) -> None:
        """The reader thread: hands each line to the loop, then None at the end."""
        assert self._loop is not None and self._inbox is not None
        try:
            while True:
                line = self._reader.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES and not line.endswith(b"\n"):
                    # Skip the rest of an oversized line.
                    while True:
                        rest = self._reader.readline(MAX_LINE_BYTES)
                        if not rest or rest.endswith(b"\n"):
                            break
                    self._loop.call_soon_threadsafe(self._inbox.put_nowait, b"\x00oversize")
                    continue
                self._loop.call_soon_threadsafe(self._inbox.put_nowait, line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self._loop.call_soon_threadsafe(self._inbox.put_nowait, None)
            except RuntimeError:
                pass  # the loop is gone

    async def serve(self, on_request: Callable[[str, Any], Awaitable[Any]],
                    on_notification: Callable[[str, Any], None]) -> None:
        """Runs until the client closes its end. Requests run as tasks."""
        self._loop = asyncio.get_running_loop()
        self._inbox = asyncio.Queue()
        reader = threading.Thread(target=self._read_lines, name="acp-reader", daemon=True)
        reader.start()
        try:
            while True:
                line = await self._inbox.get()
                if line is None:
                    break
                if line == b"\x00oversize":
                    log.warning("dropped a message longer than %d bytes", MAX_LINE_BYTES)
                    continue
                self._dispatch(line, on_request, on_notification)
        finally:
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(PeerClosed())
            # The client closed its end: short requests still get their answer,
            # and a turn that runs on is cancelled.
            if self._tasks:
                await asyncio.wait(list(self._tasks), timeout=SHUTDOWN_GRACE_S)
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            with self._write_lock:
                self._closed = True

    def _dispatch(self, line: bytes, on_request, on_notification) -> None:
        text = line.strip()
        if not text:
            return
        try:
            message = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            log.warning("dropped a line that is not JSON")
            return
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            log.warning("dropped a message that is not JSON-RPC 2.0")
            return
        method = message.get("method")
        has_id = "id" in message
        if isinstance(method, str):
            params = message.get("params")
            if has_id:
                rpc_id = message["id"]
                task = asyncio.ensure_future(self._run_request(rpc_id, method, params, on_request))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            else:
                try:
                    on_notification(method, params)
                except Exception:
                    log.exception("a notification handler failed")
            return
        if has_id and ("result" in message or "error" in message):
            rpc_id = message["id"]
            future = self._pending.get(rpc_id) if isinstance(rpc_id, int) else None
            if future is None or future.done():
                return  # an answer to a request nobody waits for any more
            if "error" in message:
                err = message.get("error") if isinstance(message.get("error"), dict) else {}
                future.set_exception(RpcError(int(err.get("code", INTERNAL_ERROR)) if isinstance(err.get("code"), int)
                                              else INTERNAL_ERROR, str(err.get("message", "error")), err.get("data")))
            else:
                future.set_result(message.get("result"))
            return
        log.warning("dropped a message that is neither a request, a notification nor a response")

    async def _run_request(self, rpc_id: Any, method: str, params: Any, on_request) -> None:
        try:
            result = await on_request(method, params)
        except RpcError as exc:
            self.respond_error(rpc_id, exc.code, exc.message, exc.data)
            return
        except asyncio.CancelledError:
            self.respond_error(rpc_id, INTERNAL_ERROR, "The agent is shutting down.")
            raise
        except Exception as exc:
            log.error("the request %s failed: %s", method, type(exc).__name__)
            self.respond_error(rpc_id, INTERNAL_ERROR, f"The agent failed while handling {method}.")
            return
        self.respond(rpc_id, result)
