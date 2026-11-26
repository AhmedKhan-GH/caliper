"""MCP transport and event-driven waits for the circuit demo."""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path


class Failure(RuntimeError):
    pass


def attempt(errors, label, operation):
    """Run one cleanup action without suppressing the remaining actions."""
    try:
        return operation()
    except Exception as exc:
        errors.append(f"{label}: {exc}")


def wait_file(path, *, process=None, timeout=120):
    """Wait for a complete JSON file through filesystem notifications."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    path = Path(path)
    changed = threading.Event()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if not event.is_directory:
                changed.set()

    observer = Observer()
    observer.schedule(Handler(), str(path.parent), recursive=False)
    observer.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            changed.clear()
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if process is not None and process.poll() is not None:
                raise Failure("The owned process ended before its ready file.")
            left = deadline - time.monotonic()
            if left <= 0 or not changed.wait(left):
                raise Failure(f"No complete {path.name} file before the deadline.")
    finally:
        observer.stop()
        observer.join()


def stop(process):
    """Close only a process that this driver owns."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_owned(command, *, stdout, timeout, env=None):
    """Reap the command and its owned children after an interrupted wait."""
    import psutil
    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=subprocess.STDOUT)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            errors = []
            try:
                owner = psutil.Process(process.pid)
                children = owner.children(recursive=True)
                for child in reversed(children):
                    if child.is_running():
                        attempt(errors, "comparison child stop", child.terminate)
                _, alive = psutil.wait_procs(children, timeout=10)
                for child in alive:
                    attempt(errors, "comparison child kill", child.kill)
            finally:
                attempt(errors, "comparison process stop", lambda: stop(process))
            if errors:
                raise Failure("; ".join(errors))


class MCP:
    def __init__(self, executable, api_file, out, env):
        self.log = self.journal = self.event_journal = self.process = self.reader = None
        self.messages = queue.Queue()
        self.counter = 0
        try:
            self.log = (out / "mcp.log").open("w", encoding="utf-8")
            self.journal = (out / "commands.jsonl").open("w", encoding="utf-8")
            self.event_journal = (out / "events.jsonl").open("w", encoding="utf-8")
            self.process = subprocess.Popen([str(executable), "--api-file", str(api_file)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, encoding="utf-8", env=env)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.request("server/discover", {})
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                raise Failure(f"MCP startup failed: {exc}; {cleanup}") from exc
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except BaseException as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(None)

    def request(self, method, params, timeout=90):
        self.counter += 1
        meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "circuit-trace", "version": "1"}}
        params = {"_meta": meta, **params}
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.counter,
                                            "method": method, "params": params}) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise Failure(f"MCP {method} timed out.") from exc
            if message is None or isinstance(message, BaseException):
                raise Failure("The MCP process closed its response stream.")
            if message.get("id") != self.counter:
                continue
            if "error" in message:
                raise Failure(f"MCP {method}: {message['error']}")
            return message["result"]

    def call(self, name, /, **args):
        result = self.request("tools/call", {"name": name.replace(".", "_"), "arguments": args})
        # Images remain separate artifacts. The command log contains no API token.
        logged = {k: v for k, v in result.items() if k != "content"}
        if "structuredContent" in logged:
            logged["structuredContent"] = {k: v for k, v in logged["structuredContent"].items()
                                            if k != "png_base64"}
        self.journal.write(json.dumps({"command": name, "args": args, "reply": logged}) + "\n")
        self.journal.flush()
        if result.get("isError"):
            raise Failure(f"{name}: {result.get('content')}")
        return result

    def body(self, name, /, **args):
        return self.call(name, **args).get("structuredContent", {})

    def close(self):
        errors = []
        if self.process is not None:
            if self.process.stdin is not None:
                attempt(errors, "MCP input close", self.process.stdin.close)
            attempt(errors, "MCP process stop", lambda: stop(self.process))
        if self.reader is not None:
            attempt(errors, "MCP reader join", lambda: self.reader.join(10))
        for name, file in (("MCP output", getattr(self.process, "stdout", None)),
                           ("MCP log", self.log), ("command journal", self.journal),
                           ("event journal", self.event_journal)):
            if file is not None:
                attempt(errors, name + " close", file.close)
        if errors:
            raise Failure("; ".join(errors))


class Events:
    def __init__(self, mcp):
        self.mcp, self.cursor = mcp, 0
        self.pending = []
        self.history = []

    def wait(self, timeout=1):
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
        if len(self.pending) > 4096:
            raise Failure("The event bridge exceeded its bounded pending queue.")
        return result["events"]

    def until(self, description, read, *, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            value = read()
            if value:
                return value
            left = deadline - time.monotonic()
            if left <= 0:
                raise Failure(f"Timed out: {description}.")
            # Resource and interpretation completion do not always emit an event.
            # A short event timeout bounds the next authoritative state read.
            self.wait(min(.05, left))

    def complete(self, name, /, **args):
        reply = self.mcp.body(name, **args)
        if not reply.get("pending"):
            return reply["result"]
        request_id = reply["request_id"]
        def read():
            value = self.mcp.body(name, request_id=request_id)
            return (value["result"],) if not value["pending"] else None
        options = {"timeout": 600} if name == "trace.run" else {}
        return self.until(name, read, **options)[0]
