"""A scripted stand-in for the Claude API, for tests only.

When the environment variable CALIPER_AGENT_TEST_SCRIPT names a JSON file, the
sidecar sends its Claude API requests to this transport instead of the
network. Nothing leaves the process: each request is answered with the next
step of the script, written as the streamed events the Messages API sends.
The switch can only make the sidecar talk to a file; it cannot send the key or
any data anywhere.

The script:

    {"steps": [<step>, ...],
     "record": "<optional path of a JSON Lines file of the requests seen>",
     "backoff_scale": <optional factor for retry delays, 0 in tests>}

A step is one of:

    {"message": {"content": [<block>, ...], "stop_reason": "end_turn",
                 "usage": {...}, "stop_details": {...}, "model": "...",
                 "iterations": [...]},
     "hang_after": <optional number of events after which the stream stalls>}
    {"error": {"status": 429, "headers": {...}, "body": {...}}}
    {"connect_error": "<message>"}
    {"stream_error": {"after": <events>, "body": {...}}}

A block is a text, thinking, tool_use, fallback or compaction block in the
API's response shape; a tool_use block may give "raw_input", the exact JSON
text to stream instead of its "input".

A request to any host other than the Claude API's is refused and recorded.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

import httpx2

from .keyguard import scrub

API_HOST = "api.anthropic.com"
SCRIPT_ENV = "CALIPER_AGENT_TEST_SCRIPT"
MAX_SCRIPT_BYTES = 4 << 20


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")


def _chunks(text: str, size: int = 24) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def render_events(spec: dict, request_model: str, index: int) -> list[bytes]:
    """The SSE events of one scripted message."""
    model = spec.get("model", request_model)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    usage.update(spec.get("usage", {}))
    start_usage = dict(usage)
    start_usage["output_tokens"] = 1
    events = [_sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_stub_{index}", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": start_usage}})]
    for i, block in enumerate(spec.get("content", [])):
        kind = block.get("type")
        if kind == "text":
            events.append(_sse("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {"type": "text", "text": ""}}))
            for piece in _chunks(block.get("text", "")):
                events.append(_sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                           "delta": {"type": "text_delta", "text": piece}}))
        elif kind == "thinking":
            events.append(_sse("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {"type": "thinking", "thinking": "",
                                                                         "signature": ""}}))
            for piece in _chunks(block.get("thinking", "")):
                events.append(_sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                           "delta": {"type": "thinking_delta", "thinking": piece}}))
            events.append(_sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                       "delta": {"type": "signature_delta",
                                                                 "signature": block.get("signature", "sig")}}))
        elif kind == "tool_use":
            events.append(_sse("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {"type": "tool_use", "id": block["id"],
                                                                         "name": block["name"], "input": {}}}))
            raw = block["raw_input"] if "raw_input" in block else json.dumps(block.get("input", {}))
            for piece in _chunks(raw, 16):
                events.append(_sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                           "delta": {"type": "input_json_delta",
                                                                     "partial_json": piece}}))
        else:
            # fallback, compaction and any other block arrive whole at their start.
            events.append(_sse("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": block}))
        events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    delta_usage = dict(usage)
    if "iterations" in spec:
        delta_usage["iterations"] = spec["iterations"]
    events.append(_sse("message_delta", {"type": "message_delta",
                                         "delta": {"stop_reason": spec.get("stop_reason", "end_turn"),
                                                   "stop_sequence": None,
                                                   "stop_details": spec.get("stop_details")},
                                         "usage": delta_usage}))
    events.append(_sse("message_stop", {"type": "message_stop"}))
    return events


class _EventStream(httpx2.AsyncByteStream):
    def __init__(self, events: list[bytes], hang_after: int | None) -> None:
        self._events = events
        self._hang_after = hang_after

    async def __aiter__(self):
        for n, event in enumerate(self._events):
            if self._hang_after is not None and n >= self._hang_after:
                await asyncio.sleep(3600)
            yield event
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        pass


class StubTransport(httpx2.AsyncBaseTransport):
    """Answers each Claude API request with the next scripted step."""

    def __init__(self, script: dict) -> None:
        self._steps = list(script.get("steps", []))
        self._record = script.get("record")
        self._lock = threading.Lock()
        self.requests: list[dict] = []
        self.hosts: list[str] = []
        self.backoff_scale = float(script.get("backoff_scale", 1.0))

    def _log(self, entry: dict) -> None:
        self.requests.append(entry)
        if self._record:
            with self._lock, open(self._record, "a", encoding="utf-8") as f:
                f.write(scrub(json.dumps(entry, sort_keys=True)) + "\n")

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        host = request.url.host
        self.hosts.append(host)
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = None
        headers = {k.lower(): v for k, v in request.headers.items()
                   if k.lower() not in ("authorization", "x-api-key")}
        entry = {"host": host, "method": request.method, "path": request.url.path, "headers": headers,
                 "has_credential": bool(request.headers.get("authorization") or request.headers.get("x-api-key")),
                 "body": body}
        self._log(entry)
        if host != API_HOST:
            raise httpx2.ConnectError(f"The stub transport refuses the host {host}.", request=request)
        if not self._steps:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error",
                                                                          "message": "The stub script has no more steps."}})
        step = self._steps.pop(0)
        index = len(self.requests)
        if "connect_error" in step:
            raise httpx2.ConnectError(str(step["connect_error"]), request=request)
        if "error" in step:
            err = step["error"]
            return httpx2.Response(int(err.get("status", 500)), headers=err.get("headers", {}),
                                   json=err.get("body", {"type": "error", "error": {"type": "api_error",
                                                                                     "message": "stub error"}}))
        model = body.get("model", "") if isinstance(body, dict) else ""
        if "stream_error" in step:
            spec = step["stream_error"]
            events = render_events({"content": [{"type": "text", "text": "partial"}], "stop_reason": "end_turn"},
                                   model, index)
            events = events[: int(spec.get("after", 1))]
            events.append(_sse("error", spec.get("body", {"type": "error", "error": {"type": "overloaded_error",
                                                                                     "message": "Overloaded"}})))
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                   stream=_EventStream(events, None))
        events = render_events(step["message"], model, index)
        return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": f"req_stub_{index}"},
                               stream=_EventStream(events, step.get("hang_after")))


def load_script(path: str) -> dict:
    p = Path(path)
    if p.stat().st_size > MAX_SCRIPT_BYTES:
        raise ValueError("The test script is larger than 4 MiB.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
        raise ValueError("The test script has no list of steps.")
    return doc


def transport_from_environment() -> StubTransport | None:
    path = os.environ.get(SCRIPT_ENV, "")
    if not path:
        return None
    return StubTransport(load_script(path))
