"""Shared fixtures: the sidecar as a child process, a fake Caliper API and scripts.

No test calls a real model or needs a key: the sidecar runs with a fake key and
a script (caliper_agent/scripted.py), answered either by Pydantic AI's
FunctionModel ("function" route, the default) or as Claude API events under
the real Anthropic model (the "anthropic" route, caliper_agent/stub_transport.py).
Every file a test writes is under pytest's tmp_path.
"""

from __future__ import annotations

import difflib
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent
COMMAND_TABLE = AGENT_DIR.parent / "commands" / "command_table.json"
FAKE_KEY = "sk-ant-api03-TESTONLY-" + "Zx9" * 20 + "-fakeAA"
FAKE_TOKEN = "fakeSessionToken_0123456789abcdef"

sys.path.insert(0, str(AGENT_DIR))


def tiny_png(width: int = 2, height: int = 2) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ---- a fake Caliper API --------------------------------------------------------------


class FakeCaliper:
    """Caliper API on loopback, with a spec the panel commands change."""

    def __init__(self, folder: Path, *, command_layer: bool = True, capture_status: int = 200) -> None:
        self.calls: list[dict] = []
        self.command_layer = command_layer
        self.capture_status = capture_status
        self.spec_text = 'caliper_spec = 0\n\n[applet]\nid = "demo"\ntitle = "Demo"\n'
        self.handlers: dict[str, Callable[[dict], dict]] = {}
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, doc: Any) -> None:
                self._reply(status, json.dumps(doc).encode())

            def _authorised(self) -> bool:
                if self.headers.get("Authorization") != f"Bearer {FAKE_TOKEN}":
                    self._json(401, {"error": "The session token is missing or wrong."})
                    return False
                return True

            def do_GET(self):
                if not self._authorised():
                    return
                fake._record(self, "GET", None)
                if self.path == "/v0/info":
                    self._json(200, {"abi_epoch": 2, "host": "0.6.0", "libtorch": None, "sdk": "0.2.0",
                                     "services": ["caliper.ui.v1"], "ui_stack": None})
                elif self.path == "/v0/applets":
                    self._json(200, {"applets": []})
                elif self.path == "/v0/capture.png":
                    if fake.capture_status == 200:
                        self._reply(200, tiny_png(4, 3), "image/png")
                    else:
                        self._json(fake.capture_status, {"error": "The window's frame cannot be read back."})
                else:
                    self._json(404, {"error": "No such endpoint."})

            def do_POST(self):
                if not self._authorised():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                args = json.loads(body) if body else None
                fake._record(self, "POST", args)
                if self.path.startswith("/v0/applets/") and self.path.endswith("/launch"):
                    applet = self.path[len("/v0/applets/"):-len("/launch")]
                    self._json(200, {"already_open": False, "id": applet, "status": "ACTIVE", "version": "0.1.0"})
                    return
                if self.path.startswith("/v0/commands/"):
                    if not fake.command_layer:
                        self._json(404, {"error": "No such endpoint."})
                        return
                    name = self.path[len("/v0/commands/"):]
                    self._json(200, fake.run(name, args or {}))
                    return
                self._json(404, {"error": "No such endpoint."})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.api_file = folder / "api.json"
        self.api_file.write_text(json.dumps({"version": 1, "pid": os.getpid(), "port": self.port,
                                             "token": FAKE_TOKEN, "url": f"http://127.0.0.1:{self.port}/v0",
                                             "stream_url": f"ws://127.0.0.1:{self.port}/v3/stream"}))

    def _record(self, handler, method: str, args: Any) -> None:
        with self._lock:
            self.calls.append({"method": method, "path": handler.path, "args": args,
                               "origin": handler.headers.get("X-Caliper-Origin"),
                               "group": handler.headers.get("X-Caliper-Group"),
                               "preview": handler.headers.get("X-Caliper-Preview"),
                               "host": handler.headers.get("Host")})

    def _edit(self, args: dict, new_text: str, result: dict) -> dict:
        """A spec edit as the session answers it: the diff, and no change on a dry run."""
        diff = "".join(difflib.unified_diff(self.spec_text.splitlines(keepends=True),
                                            new_text.splitlines(keepends=True), "before", "after"))
        applied = not args.get("dry_run", False) and diff != ""
        if applied:
            self.spec_text = new_text
        return {"ok": True, "result": {**result, "warnings": [], "diff": diff, "applied": applied}}

    def run(self, name: str, args: dict) -> dict:
        if name in self.handlers:
            return self.handlers[name](args)
        if name == "spec.get":
            return {"ok": True, "result": {"id": "demo", "path": "applets/demo/demo.spec.toml", "dirty": True,
                                           "text": self.spec_text, "json": {}}}
        if name == "panel.create":
            if f'id = "{args.get("id")}"' in self.spec_text:
                return {"ok": False, "error": {"code": "panel.exists", "message": "The panel exists.",
                                               "where": "args.id"}}
            new_text = self.spec_text + f'\n[[panel]]\nid = "{args.get("id")}"\nprimitive = "{args.get("primitive")}"\n'
            return self._edit(args, new_text, {"id": args.get("id")})
        if name == "spec.set_text":
            return self._edit(args, args["text"], {"id": "demo"})
        if name == "panel.render":
            import base64
            return {"ok": True, "result": {"png_base64": base64.b64encode(tiny_png(8, 5)).decode(),
                                           "width": 8, "height": 5}}
        if name == "catalog.list":
            return {"ok": True, "result": {"entries": [{"id": "table:run42_metrics", "kind": "table"}],
                                           "truncated": False, "warnings": []}}
        return {"ok": True, "result": {}}

    def commands(self, name: str | None = None, *, dry: bool | None = False) -> list[dict]:
        """The command calls received; by default without the dry runs (dry=None: all)."""
        with self._lock:
            out = [c for c in self.calls if c["path"].startswith("/v0/commands/")]
        out = [c for c in out if name is None or c["path"] == f"/v0/commands/{name}"]
        if dry is None:
            return out
        return [c for c in out if bool((c["args"] or {}).get("dry_run")) == dry]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def caliper(tmp_path):
    fake = FakeCaliper(tmp_path)
    yield fake
    fake.close()


# ---- scripts ------------------------------------------------------------------------------


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def thinking(t: str, sig: str = "sig-1") -> dict:
    return {"type": "thinking", "thinking": t, "signature": sig}


def tool_use(call_id: str, name: str, inp: Any = None, raw: str | None = None) -> dict:
    block = {"type": "tool_use", "id": call_id, "name": name, "input": inp if inp is not None else {}}
    if raw is not None:
        block["raw_input"] = raw
    return block


def message(*content: dict, stop: str = "end_turn", usage: dict | None = None, **extra) -> dict:
    spec = {"content": list(content), "stop_reason": stop,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5}}
    spec.update(extra)
    return {"message": spec}


def widget_meta(target: dict | None = None, preview: str = "widget-1", scope: list | None = None) -> dict:
    """`session/new`'s `_meta` for a widget session."""
    widget: dict = {"target": target or {"kind": "panel", "panel": "curves"}, "preview": preview}
    if scope is not None:
        widget["scope"] = scope
    return {"caliper": {"widget": widget}}


def write_script(folder: Path, steps: list[dict], route: str = "function", **extra) -> tuple[Path, Path]:
    """A script for the sidecar's test switch. `route` "function" answers with
    Pydantic AI's FunctionModel; "anthropic" answers as the Claude API's wire
    format, under the real Anthropic model (caliper_agent/scripted.py)."""
    record = folder / "requests.jsonl"
    script = folder / "script.json"
    script.write_text(json.dumps({"steps": steps, "record": str(record), "backoff_scale": 0, "route": route,
                                  **extra}))
    return script, record


def read_requests(record: Path) -> list[dict]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines() if line.strip()]


# ---- the sidecar as a child process -------------------------------------------------------


class AgentProcess:
    """Runs `python -P -m caliper_agent` and speaks ACP to it, as a client would."""

    def __init__(self, *, script: Path | None, api_file: Path | None, key: str | None = FAKE_KEY,
                 extra_args: list[str] | None = None, cwd: Path | None = None) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_") and k != "CALIPER_AGENT_TEST_SCRIPT"}
        env["PYTHONPATH"] = str(AGENT_DIR)
        if key is not None:
            env["ANTHROPIC_API_KEY"] = key
        if script is not None:
            env["CALIPER_AGENT_TEST_SCRIPT"] = str(script)
        argv = [sys.executable, "-P", "-m", "caliper_agent", "--command-table", str(COMMAND_TABLE)]
        if api_file is not None:
            argv += ["--api-file", str(api_file)]
        argv += extra_args or []
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, cwd=str(cwd) if cwd else None)
        self.lines: list[str] = []
        self.stderr_chunks: list[bytes] = []
        self.inbox: queue.Queue = queue.Queue()
        self.next_id = 1
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read_out(self) -> None:
        for raw in self.proc.stdout:
            line = raw.decode("utf-8")
            self.lines.append(line)
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                self.inbox.put({"unparsed": line})
        self.inbox.put(None)

    def _read_err(self) -> None:
        for raw in self.proc.stderr:
            self.stderr_chunks.append(raw)

    def send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict, *, timeout: float = 30.0,
                on_permission: Callable[[dict], dict] | None = None,
                on_update: Callable[[dict], None] | None = None) -> tuple[dict, list[dict]]:
        """Sends a request; returns its response and the updates seen meanwhile.

        Permission requests are answered with `on_permission(params)`, or
        refused when there is none.
        """
        rpc_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
        updates: list[dict] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no answer to {method} within {timeout} s")
            try:
                msg = self.inbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is None:
                raise EOFError(f"the agent closed its output during {method}; stderr: {self.stderr()}")
            if msg.get("id") == rpc_id and ("result" in msg or "error" in msg):
                return msg, updates
            if msg.get("method") == "session/update":
                updates.append(msg["params"]["update"])
                if on_update:
                    on_update(msg["params"]["update"])
            elif msg.get("method") == "session/request_permission":
                updates.append({"sessionUpdate": "_permission_request", **msg["params"]})
                answer = on_permission(msg["params"]) if on_permission else {"outcome": {"outcome": "cancelled"}}
                if answer is not None:
                    self.send({"jsonrpc": "2.0", "id": msg["id"], "result": answer})

    def start_session(self, cwd: str = "/tmp/caliper-test-workspace", meta: dict | None = None) -> str:
        response, _ = self.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        assert response["result"]["protocolVersion"] == 1
        params: dict = {"cwd": cwd, "mcpServers": []}
        if meta is not None:
            params["_meta"] = meta
        response, _ = self.request("session/new", params)
        assert "result" in response, response
        return response["result"]["sessionId"]

    def prompt(self, session_id: str, prompt: str | list, **kwargs) -> tuple[dict, list[dict]]:
        blocks = [{"type": "text", "text": prompt}] if isinstance(prompt, str) else prompt
        return self.request("session/prompt", {"sessionId": session_id, "prompt": blocks}, **kwargs)

    def stderr(self) -> str:
        return b"".join(self.stderr_chunks).decode("utf-8", "replace")

    def close(self) -> int:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            code = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        return code

    def all_output(self) -> str:
        return "".join(self.lines) + self.stderr()


def allow(params: dict) -> dict:
    option = next(o for o in params["options"] if o["kind"] == "allow_once")
    return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}


def deny(params: dict) -> dict:
    option = next(o for o in params["options"] if o["kind"] == "reject_once")
    return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}


@pytest.fixture
def run_agent(tmp_path, caliper):
    """Starts the sidecar with a script; closes it after the test."""
    started: list[AgentProcess] = []

    def start(steps: list[dict] | None, route: str = "function", script_extra: dict | None = None,
              **kwargs) -> tuple[AgentProcess, Path]:
        folder = tmp_path / f"agent{len(started) + 1}"
        folder.mkdir()
        script, record = (write_script(folder, steps, route, **(script_extra or {})) if steps is not None
                          else (None, folder / "none.jsonl"))
        kwargs.setdefault("api_file", caliper.api_file)
        agent = AgentProcess(script=script, **kwargs)
        started.append(agent)
        return agent, record

    yield start
    for agent in started:
        agent.close()
        # Whatever the test did, the key never reaches the client or stderr.
        assert FAKE_KEY not in agent.all_output()
        assert FAKE_TOKEN not in agent.all_output()


# ---- what the function route recorded --------------------------------------------------


def parts(messages: list[dict], kind: str) -> list[dict]:
    """Every part of a kind (`user-prompt`, `tool-return`, `tool-call`, `text`, `thinking`)."""
    return [p for m in messages for p in m["parts"] if p.get("part_kind") == kind]


def tool_results(messages: list[dict]) -> dict[str, dict]:
    return {p["tool_call_id"]: p for p in parts(messages, "tool-return")}


def prompts(messages: list[dict]) -> list:
    return [p["content"] for p in parts(messages, "user-prompt")]


def message_text(updates: list[dict]) -> str:
    return "".join(u["content"]["text"] for u in updates if u.get("sessionUpdate") == "agent_message_chunk")


def statuses(updates: list[dict]) -> list[dict]:
    return [u["_meta"]["caliper"] | {"text": u["content"]["text"]}
            for u in updates if u.get("sessionUpdate") == "agent_thought_chunk" and "_meta" in u]
