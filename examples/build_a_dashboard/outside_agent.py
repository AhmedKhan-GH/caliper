"""An outside agent that drives the running Caliper app through caliper-mcp.

It speaks MCP over standard input and output to caliper-mcp, exactly as an
agent such as Claude Code or Codex does once caliper-mcp is in its MCP
configuration: it starts the server, lists its tools, reads the catalog and the
open spec, and makes one edit to the open dashboard. caliper-mcp reads the
running app's API file, whose token runs commands as the origin `mcp`, so the
app holds the edit on an approval card until the person decides, and the call
answers once they have.

Two ways to choose the calls:

- scripted (the default): no model. The edit adds a series panel of the
  learning rate by step beside the first panel, bound to the open spec's first
  source that has the columns step and lr; without one it gives the first panel
  a new title.
- with --model: a model that Ollama serves on this machine chooses the calls
  from the tool list, with the task in --task, through Ollama's own chat API
  with a context of --num-ctx tokens. Only a server on loopback is used, and no
  key is sent.

    python3 outside_agent.py --mcp <build>/caliper-mcp [--api-file <the app's api.json>]
    python3 outside_agent.py --mcp <build>/caliper-mcp --model example-model

caliper-mcp finds the API file itself (api.json in the app-data folder, or in
CALIPER_DATA_DIR) unless --api-file names it. It prints what it does, one line
per call, and exits with 0 when the edit was approved and landed, 1 when it
was denied or failed, and 2 for bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PROTOCOL = "2025-11-25"
CALL_TIMEOUT_S = 140.0   # a call may wait up to 120 s for the person's approval
TASK = ("Add a series panel of the learning rate by step to the open dashboard, beside its first panel. "
        "Look at the open spec and the catalog first, and use the source the spec already has.")


class McpClient:
    """One caliper-mcp child, spoken to over stdio: one JSON-RPC message per line."""

    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.next_id = 1
        self.stderr: list[str] = []
        # Standard output is read by a thread into a queue, so that a wait for an answer ends at its
        # deadline even when caliper-mcp writes nothing; None marks its end.
        self.messages: queue.Queue = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()
        threading.Thread(target=self._read, daemon=True).start()

    def _drain(self) -> None:
        for raw in self.proc.stderr:
            self.stderr.append(raw.decode("utf-8", "replace").rstrip())

    def _read(self) -> None:
        for raw in self.proc.stdout:
            self.messages.put(raw)
        self.messages.put(None)

    def _send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        rpc_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                line = self.messages.get(timeout=left)
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError("caliper-mcp ended: " + " | ".join(self.stderr[-3:]))
            try:
                message = json.loads(line)
            except ValueError:
                continue                            # not a JSON-RPC message
            if message.get("id") == rpc_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message["result"]
        raise TimeoutError(f"{method}: no answer within {timeout:.0f} s")

    def initialize(self) -> dict:
        result = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                             "clientInfo": {"name": "outside-agent-example", "version": "0.1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def call(self, tool: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": tool, "arguments": arguments}, timeout=CALL_TIMEOUT_S)

    def close(self) -> int:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait()


def say(text: str) -> None:
    print(text, flush=True)


def result_text(result: dict, limit: int = 160) -> str:
    """A tool result in one line: its error, or its structured content cut short."""
    if result.get("isError"):
        for block in result.get("content", []):
            if block.get("type") == "text":
                return "error " + block["text"][:limit]
        return "error"
    content = result.get("structuredContent")
    text = json.dumps(content) if content is not None else json.dumps(result.get("content"))
    return text if len(text) <= limit else text[:limit] + "..."


# ---- scripted ---------------------------------------------------------------------------------


def scripted(mcp: McpClient) -> bool:
    spec = mcp.call("spec_get", {})
    if spec.get("isError"):
        say(f"spec_get: {result_text(spec)}")
        return False
    doc = spec["structuredContent"]
    panels = doc.get("json", {}).get("panel", [])
    sources = doc.get("json", {}).get("source", [])
    say(f"spec_get: {doc.get('id')} with {len(panels)} panels ({', '.join(p['id'] for p in panels)}) "
        f"and {len(sources)} sources")
    if not panels:
        say("the open spec has no panel to put the new one beside")
        return False
    # The open spec's first source over a table with the columns step and lr.
    chosen = None
    for source in sources:
        entry = source.get("catalog")
        if not isinstance(entry, str):
            continue
        described = mcp.call("catalog_describe", {"id": entry})
        entry_doc = (described.get("structuredContent") or {}).get("entry") or {}
        columns = [c.get("name") for c in (entry_doc.get("columns") or [])]
        say(f"catalog_describe {entry}: columns {', '.join(map(str, columns)) or 'none'}")
        if "step" in columns and "lr" in columns:
            chosen = source["id"]
            break
    target = panels[0]["id"]
    if chosen:
        tool, args = "panel_create", {"id": "learning-rate", "primitive": "series", "title": "Learning rate",
                                      "bind": {"x": f"{chosen}.step", "y": [f"{chosen}.lr"]},
                                      "place": {"target": target, "where": "beside"}}
    else:
        tool, args = "panel_patch", {"id": target, "patch": {"title": "Edited by an outside agent"}}
    say(f"{tool} {json.dumps(args)}: waiting for the person's approval in the app")
    started = time.monotonic()
    result = mcp.call(tool, args)
    say(f"{tool} answered after {time.monotonic() - started:.1f} s: {result_text(result)}")
    if result.get("isError"):
        return False
    after = mcp.call("spec_get", {})
    ids = [p["id"] for p in after["structuredContent"].get("json", {}).get("panel", [])]
    say(f"spec_get: panels now {', '.join(ids)}")
    return True


# ---- with a local model -----------------------------------------------------------------------


def chat(ollama: str, model: str, messages: list, tools: list, num_ctx: int) -> dict:
    """One request to Ollama's own chat API, which loads the model with `num_ctx`
    (its OpenAI-compatible endpoint ignores it) and unloads it after 60 s."""
    body = {"model": model, "messages": messages, "tools": tools, "stream": False, "keep_alive": "60s",
            "options": {"num_ctx": num_ctx}}
    request = urllib.request.Request(ollama.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # loopback: never through a proxy
    with opener.open(request, timeout=600) as response:
        return json.loads(response.read())


def with_model(mcp: McpClient, tools: list, ollama: str, model: str, task: str, max_requests: int,
               num_ctx: int) -> bool:
    host = urlsplit(ollama).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        say(f"--ollama must be a server on this machine, not {host}")
        return False
    functions = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", "")[:1024],
                                                   "parameters": t["inputSchema"]}} for t in tools]
    messages = [{"role": "system", "content": (
        "You are an agent outside Caliper, an app that shows machine learning data in panels, and you drive the "
        "running app through the tools of its MCP server: each is one command of its command layer. The person "
        "approves each change to the open spec in the app, so do not ask for permission. Never invent a catalog "
        "id, a source, a column or a panel id; take them from the tools. When the change is made, answer in one "
        "sentence.")},
                {"role": "user", "content": task}]
    edited = False
    for n in range(max_requests):
        started = time.monotonic()
        answer = chat(ollama, model, messages, functions, num_ctx)
        reply = answer.get("message") or {}
        calls = reply.get("tool_calls") or []
        say(f"model request {n + 1}: {answer.get('prompt_eval_count', '?')} prompt tokens, {len(calls)} calls, "
            f"{time.monotonic() - started:.1f} s")
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
        if not calls:
            say(f"model: {(reply.get('content') or '').strip()[:400]}")
            return edited
        for c in calls:
            name = c.get("function", {}).get("name", "")
            args = c.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                result = {"isError": True, "content": [{"type": "text", "text": "The arguments are not a JSON object."}]}
            else:
                result = mcp.call(name, args)
            say(f"  {name} {json.dumps(args)[:200]}: {result_text(result)}")
            if not result.get("isError") and name.split("_", 1)[0] in ("panel", "source", "layout", "spec") \
                    and name not in ("spec_get", "spec_validate"):
                edited = True
            text = next((b["text"] for b in result.get("content", []) if b.get("type") == "text"), "")
            messages.append({"role": "tool", "tool_name": name, "content": text[:8000]})
    say(f"stopped after {max_requests} model requests")
    return edited


# ---- main -------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mcp", default=shutil.which("caliper-mcp") or "",
                        help="the caliper-mcp executable (default: the one on PATH)")
    parser.add_argument("--api-file", help="the running app's API file (default: caliper-mcp's own lookup)")
    parser.add_argument("--model", help="a model that Ollama serves on this machine chooses the calls")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama's address (default: %(default)s)")
    parser.add_argument("--task", default=TASK, help="what the model is asked to do")
    parser.add_argument("--max-requests", type=int, default=12, help="the most model requests (default: 12)")
    parser.add_argument("--num-ctx", type=int, default=16384,
                        help="the context Ollama loads the model with, in tokens (default: %(default)s)")
    args = parser.parse_args()
    if not args.mcp or not Path(args.mcp).is_file():
        print("outside_agent.py: give the caliper-mcp executable with --mcp", file=sys.stderr)
        return 2
    argv = [args.mcp] + (["--api-file", args.api_file] if args.api_file else [])
    mcp = McpClient(argv)
    try:
        server = mcp.initialize()
        say(f"initialize: {server['serverInfo']['name']} {server['serverInfo'].get('version', '')}, "
            f"protocol {server['protocolVersion']}")
        tools = mcp.request("tools/list", {})["tools"]
        say(f"tools/list: {len(tools)} tools: {', '.join(t['name'] for t in tools)}")
        listed = mcp.call("catalog_list", {})
        entries = (listed.get("structuredContent") or {}).get("entries", [])
        say(f"catalog_list: {len(entries)} entries: {', '.join(e['id'] for e in entries[:12])}"
            + (" ..." if len(entries) > 12 else "") if not listed.get("isError") else f"catalog_list: {result_text(listed)}")
        if args.model:
            ok = with_model(mcp, tools, args.ollama, args.model, args.task, args.max_requests, args.num_ctx)
        else:
            ok = scripted(mcp)
    finally:
        code = mcp.close()
        if code != 0:
            print(f"caliper-mcp exited with {code}: " + " | ".join(mcp.stderr[-3:]), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
