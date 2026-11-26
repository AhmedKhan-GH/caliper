"""MCP transport and event-driven waits for the Circuit Trace demo."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


class Failure(RuntimeError):
    """Raised when an MCP call, child process, or readiness barrier fails."""


def attempt(errors: list[str], label: str, operation):
    """Run one cleanup operation without skipping remaining cleanup steps."""
    try:
        return operation()
    except Exception as exc:
        errors.append(f"{label}: {exc}")
        return None


def wait_file(path: str | Path, *, process=None, timeout: float = 120.0) -> dict:
    """Wait until ``path`` contains a complete JSON object."""
    target = Path(path)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if process is not None and process.poll() is not None:
            raise Failure("The owned process ended before writing its API file.")
        if time.monotonic() >= deadline:
            raise Failure(f"Timed out waiting for {target.name}.")
        time.sleep(0.05)


def stop(process, *, group: bool = False) -> None:
    """Terminate an owned child process cleanly."""
    if process is None or process.poll() is not None:
        return
    if group and os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


class MCP:
    """JSON-RPC stdio client for ``caliper-mcp``."""

    def __init__(self, executable: Path, api_file: Path, out: Path, env: dict[str, str]):
        self.log = self.journal = self.event_journal = self.process = self.reader = None
        self.messages: queue.Queue = queue.Queue()
        self.counter = 0
        try:
            self.log = (out / "mcp.log").open("w", encoding="utf-8")
            self.journal = (out / "commands.jsonl").open("w", encoding="utf-8")
            self.event_journal = (out / "events.jsonl").open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                [str(executable), "--api-file", str(api_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                encoding="utf-8",
                env=env,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.request("server/discover", {})
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                raise Failure(f"MCP startup failed: {exc}; {cleanup}") from exc
            raise

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except BaseException as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(None)

    def request(self, method: str, params: dict, timeout: float = 90.0) -> dict:
        self.counter += 1
        meta = {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "circuit-trace", "version": "1"},
        }
        payload = {"jsonrpc": "2.0", "id": self.counter, "method": method, "params": {"_meta": meta, **params}}
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise Failure(f"MCP {method} timed out.") from exc
            if message is None or isinstance(message, BaseException):
                raise Failure("The MCP process closed its response stream.")
            if message.get("id") != self.counter:
                continue
            if "error" in message:
                raise Failure(f"MCP {method}: {message['error']}")
            return message["result"]

    def call(self, name: str, /, **args) -> dict:
        result = self.request("tools/call", {"name": name.replace(".", "_"), "arguments": args})
        logged = {k: v for k, v in result.items() if k != "content"}
        if "structuredContent" in logged:
            logged["structuredContent"] = {k: v for k, v in logged["structuredContent"].items() if k != "png_base64"}
        self.journal.write(json.dumps({"command": name, "args": args, "reply": logged}) + "\n")
        self.journal.flush()
        if result.get("isError"):
            raise Failure(f"{name}: {result.get('content')}")
        return result

    def body(self, name: str, /, **args) -> dict:
        return self.call(name, **args).get("structuredContent", {})

    def close(self) -> None:
        errors: list[str] = []
        if self.process is not None:
            if self.process.stdin is not None:
                attempt(errors, "MCP input close", self.process.stdin.close)
            attempt(errors, "MCP process stop", lambda: stop(self.process))
        if self.reader is not None:
            attempt(errors, "MCP reader join", lambda: self.reader.join(10))
        for name, handle in (
            ("MCP output", getattr(self.process, "stdout", None)),
            ("MCP log", self.log),
            ("command journal", self.journal),
            ("event journal", self.event_journal),
        ):
            if handle is not None:
                attempt(errors, f"{name} close", handle.close)
        if errors:
            raise Failure("; ".join(errors))


class Events:
    """Cursor-based event consumer and deferred-request waiter."""

    def __init__(self, mcp: MCP):
        self.mcp = mcp
        self.cursor = 0
        self.pending: list[dict] = []
        self.history: list[dict] = []

    def wait(self, timeout: float = 1.0) -> list[dict]:
        result = self.mcp.body("events.wait", cursor=self.cursor, timeout_s=timeout, limit=256)
        self.cursor = result["cursor"]
        if result["lost"]:
            raise Failure("The event bridge lost events.")
        for event in result["events"]:
            self.mcp.event_journal.write(json.dumps(event) + "\n")
        self.mcp.event_journal.flush()
        self.history.extend(result["events"])
        self.history = self.history[-4096:]
        self.pending.extend(result["events"])
        return result["events"]

    def until(self, description: str, read, *, timeout: float = 120.0):
        deadline = time.monotonic() + timeout
        while True:
            value = read()
            if value:
                return value
            left = deadline - time.monotonic()
            if left <= 0:
                raise Failure(f"Timed out: {description}.")
            self.wait(min(0.05, left))

    def complete(self, name: str, /, **args):
        reply = self.mcp.body(name, **args)
        if not reply.get("pending"):
            return reply["result"]
        request_id = reply["request_id"]

        def poll():
            value = self.mcp.body(name, request_id=request_id)
            return (value["result"],) if not value["pending"] else None

        timeout = 600.0 if name == "trace.run" else 180.0
        return self.until(name, poll, timeout=timeout)[0]
