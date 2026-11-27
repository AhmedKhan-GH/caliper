"""A fake OpenAI-compatible server on loopback, for the provider, probe and conformance tests.

It speaks the chat completions API (streamed, as Server-Sent Events) and
`GET /v1/models`, with the differences real servers have, each a switch:

- **Capabilities.** `refuse_tools` answers a request with tools with a 400 that
  names tools; `tool_mode="text"` makes the model answer the probe's tool
  request in text instead. `refuse_images` answers an image with a 400 that
  names images; `vision_mode="blind"` makes the model accept an image and fail
  to say what it shows, and `"see"` (the default) names the colour of the
  probe's PNG, which the server decodes.
- **Quirks.** `refuse_strict` refuses tools that carry `strict`;
  `refuse_max_completion_tokens` refuses that field; `args_style` streams a
  call's arguments with its name (`"whole"`), after it in one chunk
  (`"split"`, the default) or in three-character pieces (`"pieces"`);
  `reasoning` puts a thought in the `reasoning` or `reasoning_content` field,
  or in `<think>` tags (`"think_tags"`).
- **The list of models.** `models`, a list of entries (with any extra fields,
  such as a context length), or None for a server without the endpoint;
  `lmstudio` answers LM Studio's `GET /api/v0/models/<id>` with
  `max_context_length`.
- `require_key` answers 401 to a request without that bearer key.
- **Ollama's own chat API.** `POST /api/chat` answers as Ollama does, in lines
  of JSON (or one object without streaming), with the same model behaviour:
  thinking in `thinking`, tool calls with their arguments as objects (without
  ids when `native_ids` is False), a 400 `{"error": ...}` for a refusal, and
  `done_reason` and the token counts in the last line. Its requests are kept
  in `requests` as sent, and `native_requests` too. `show`, Ollama's
  `model_info`, answers `POST /api/show`; `prompt_tokens` gives the
  `prompt_eval_count` of each reply in turn.

Replies to other requests are taken in order from `replies`, or made by
`responder(body)` when it is given.
"""

from __future__ import annotations

import base64
import json
import re
import socketserver
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class _LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]

PROBE_TOOL = "record_number"

# Ollama's model metadata (POST /api/show) of a hybrid 27B model: 65 layers, one in four with attention,
# 4 key and value heads of 256: 17 x 4 x 512 x 2 bytes, 68 KiB of context cache a token.
HYBRID_27B_INFO = {"general.architecture": "qwen35", "qwen35.block_count": 65, "qwen35.full_attention_interval": 4,
                   "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
                   "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
                   "qwen35.context_length": 262144}

_COLOURS = {"red": (220, 30, 30), "green": (30, 170, 40), "blue": (30, 60, 220), "yellow": (240, 215, 20)}


def png_colour(data_url: str) -> str | None:
    """The colour name of the probe's solid PNG (an RGB image, filter 0 on each row)."""
    m = re.match(r"data:image/png;base64,(.*)", data_url)
    if not m:
        return None
    raw = base64.b64decode(m.group(1))
    i, idat = 8, b""
    while i < len(raw):
        length = int.from_bytes(raw[i:i + 4], "big")
        kind = raw[i + 4:i + 8]
        if kind == b"IDAT":
            idat += raw[i + 8:i + 8 + length]
        i += 12 + length
    pixels = zlib.decompress(idat)
    rgb = tuple(pixels[1:4])
    return next((name for name, value in _COLOURS.items() if value == rgb), None)


class FakeOpenAI:
    def __init__(self, replies: list[dict] | None = None, *, refuse_tools: bool = False, refuse_images: bool = False,
                 tool_mode: str = "call", vision_mode: str = "see", refuse_strict: bool = False,
                 refuse_max_completion_tokens: bool = False, args_style: str = "split", reasoning: str | None = None,
                 models: list[dict] | None = None, lmstudio: int | None = None, require_key: str | None = None,
                 responder: Callable[[dict], dict] | None = None, record: str | None = None,
                 native_ids: bool = True, show: dict | None = None,
                 prompt_tokens: list[int] | None = None, ps_context: int | None = None,
                 native_error: tuple[int, str] | None = None) -> None:
        self.replies = list(replies or [])
        self.refuse_tools = refuse_tools
        self.refuse_images = refuse_images
        self.tool_mode = tool_mode
        self.vision_mode = vision_mode
        self.refuse_strict = refuse_strict
        self.refuse_max_completion_tokens = refuse_max_completion_tokens
        self.args_style = args_style
        self.reasoning = reasoning
        self.models = models
        self.lmstudio = lmstudio
        self.require_key = require_key
        self.responder = responder
        self.record = record  # a file that gets one JSON line per request: method, path, Authorization
        self.native_ids = native_ids
        self.show = show   # Ollama's model_info for POST /api/show; None answers 404
        self.prompt_tokens = list(prompt_tokens or [])   # prompt_eval_count of each native reply, else 11
        self.ps_context = ps_context   # the context_length GET /api/ps gives for each listed model; None: none listed
        self.native_error = native_error   # (status, message) for every /api/chat request, as Ollama refuses one
        self.requests: list[dict] = []
        self.native_requests: list[dict] = []
        self.headers: list[dict] = []
        self.paths: list[str] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status: int, doc: Any) -> None:
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _record(self, method: str) -> None:
                if fake.record:
                    line = json.dumps({"method": method, "path": self.path,
                                       "authorization": self.headers.get("Authorization")})
                    with fake._lock, open(fake.record, "a", encoding="utf-8") as out:
                        out.write(line + "\n")

            def _authorised(self) -> bool:
                if fake.require_key and self.headers.get("Authorization") != f"Bearer {fake.require_key}":
                    self._json(401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error"}})
                    return False
                return True

            def do_GET(self):
                with fake._lock:
                    fake.paths.append(self.path)
                self._record("GET")
                if not self._authorised():
                    return
                if self.path == "/v1/models" and fake.models is not None:
                    self._json(200, {"object": "list", "data": [{"object": "model", "owned_by": "fake", **m}
                                                                for m in fake.models]})
                elif self.path == "/api/ps":
                    loaded = [{"name": m["id"], "model": m["id"], "context_length": fake.ps_context}
                              for m in (fake.models or [])] if fake.ps_context else []
                    self._json(200, {"models": loaded})
                elif self.path.startswith("/api/v0/models/") and fake.lmstudio is not None:
                    self._json(200, {"id": self.path.rsplit("/", 1)[-1], "object": "model", "type": "llm",
                                     "max_context_length": fake.lmstudio})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length))
                with fake._lock:
                    fake.paths.append(self.path)
                    if self.path != "/api/show":   # the chat requests only
                        fake.requests.append(body)
                        fake.headers.append({k.lower(): v for k, v in self.headers.items()})
                self._record("POST")
                if not self._authorised():
                    return
                if self.path == "/api/show":
                    if fake.show is None:
                        self._json(404, {"error": "not found"})
                    else:
                        self._json(200, {"model_info": fake.show, "capabilities": ["completion", "tools"]})
                    return
                if self.path == "/api/chat":
                    with fake._lock:
                        fake.native_requests.append(body)
                    if fake.native_error is not None:
                        self._json(fake.native_error[0], {"error": fake.native_error[1]})
                        return
                    view = fake.openai_view(body)
                    error = fake.refusal(view)
                    if error is not None:
                        self._json(400, {"error": error})
                        return
                    lines = fake.native_lines(fake.reply(view), body.get("model", ""))
                    if not body.get("stream", True):
                        self._json(200, fake.native_whole(lines))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for line in lines:
                        self.wfile.write(json.dumps(line).encode() + b"\n")
                    return
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "not found"}})
                    return
                error = fake.refusal(body)
                if error is not None:
                    self._json(400, {"error": {"message": error, "type": "invalid_request_error"}})
                    return
                reply = fake.reply(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in fake.chunks(reply, body.get("model", "")):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = _LoopbackHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # ---- what the model does ---------------------------------------------------------------------

    @staticmethod
    def _tool_names(body: dict) -> list[str]:
        return [t.get("function", {}).get("name", "") for t in body.get("tools") or []]

    @staticmethod
    def _images(body: dict) -> list[str]:
        urls = []
        for m in body.get("messages", []):
            if isinstance(m.get("content"), list):
                urls += [p["image_url"]["url"] for p in m["content"] if p.get("type") == "image_url"]
        return urls

    def refusal(self, body: dict) -> str | None:
        if self.refuse_max_completion_tokens and "max_completion_tokens" in body:
            return "Unrecognized request argument supplied: max_completion_tokens"
        if body.get("tools") and (self.refuse_tools or (self.tool_mode == "refuse" and PROBE_TOOL in self._tool_names(body))):
            return "fake-model does not support tools"
        if self.refuse_strict and any(t.get("function", {}).get("strict") for t in body.get("tools") or []):
            return "Invalid schema: strict is not supported by this server"
        if (self.refuse_images or self.vision_mode == "refuse") and self._images(body):
            return "image input is not supported by this model"
        return None

    def reply(self, body: dict) -> dict:
        if PROBE_TOOL in self._tool_names(body):
            if self.tool_mode == "text":
                return {"text": "The number is 7."}
            if self.tool_mode == "cutoff":
                return {"text": "Let me think about which tool", "finish": "length"}
            return {"tool_calls": [{"id": "call_probe", "name": PROBE_TOOL, "arguments": json.dumps({"number": 7})}]}
        images = self._images(body)
        if images and not body.get("tools"):
            if self.vision_mode == "blind":
                return {"text": "I am not able to see images."}
            return {"text": f"{png_colour(images[0]) or 'grey'}"}
        if self.responder is not None:
            return self.responder(body)
        with self._lock:
            return self.replies.pop(0) if self.replies else {"text": "(no more replies)"}

    def chunks(self, reply: dict, model: str) -> list[dict]:
        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        out = [chunk({"role": "assistant"})]
        thought = reply.get("thought", "Thinking about it." if self.reasoning else None)
        text = reply.get("text", "")
        if thought and self.reasoning in ("reasoning", "reasoning_content"):
            out.append(chunk({"content": None, self.reasoning: thought}))
        elif thought and self.reasoning == "think_tags":
            # Each tag is one token, so it arrives as a chunk of its own.
            out += [chunk({"content": "<think>"}), chunk({"content": thought}), chunk({"content": "</think>"})]
        if text:
            out += [chunk({"content": text[i:i + 8]}) for i in range(0, len(text), 8)]
        for n, call in enumerate(reply.get("tool_calls", [])):
            args = call["arguments"]
            head = {"index": n, "id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": ""}}
            if self.args_style == "whole":
                head["function"]["arguments"] = args
                out.append(chunk({"tool_calls": [head]}))
            elif self.args_style == "pieces":
                out.append(chunk({"tool_calls": [head]}))
                out += [chunk({"tool_calls": [{"index": n, "function": {"arguments": args[i:i + 3]}}]})
                        for i in range(0, len(args), 3)]
            else:
                out.append(chunk({"tool_calls": [head]}))
                out.append(chunk({"tool_calls": [{"index": n, "function": {"arguments": args}}]}))
        # Some servers say "stop" even after tool calls; the sidecar runs them anyway.
        out.append(chunk({}, reply.get("finish", "tool_calls" if reply.get("tool_calls") else "stop")))
        out.append({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})
        return out

    # ---- Ollama's own chat API ----------------------------------------------------------------------

    @staticmethod
    def openai_view(body: dict) -> dict:
        """A native request as the chat completions API would carry it, for refusal() and reply()."""
        messages = []
        for m in body.get("messages", []):
            if m.get("images"):
                content = [{"type": "text", "text": m.get("content", "")}] + [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + i}} for i in m["images"]]
                messages.append({**m, "content": content})
            else:
                messages.append(m)
        return {"model": body.get("model"), "messages": messages, "tools": body.get("tools")}

    def native_lines(self, reply: dict, model: str) -> list[dict]:
        def line(message: dict) -> dict:
            return {"model": model, "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "", **message}, "done": False}
        out = []
        thought = reply.get("thought", "Thinking about it." if self.reasoning else None)
        if thought:
            out.append(line({"thinking": thought}))
        text = reply.get("text", "")
        out += [line({"content": text[i:i + 8]}) for i in range(0, len(text), 8)]
        for n, call in enumerate(reply.get("tool_calls", [])):
            try:
                args = json.loads(call["arguments"])
            except ValueError:
                args = {"_unparsed": call["arguments"]}
            c = {"function": {"index": n, "name": call["name"], "arguments": args}}
            if self.native_ids:
                c["id"] = call["id"]
            out.append(line({"tool_calls": [c]}))
        out.append({"model": model, "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": ""}, "done": True,
                    "done_reason": "length" if reply.get("finish") == "length" else "stop",
                    "prompt_eval_count": self.prompt_tokens.pop(0) if self.prompt_tokens else 11, "eval_count": 3})
        return out

    @staticmethod
    def native_whole(lines: list[dict]) -> dict:
        message = {"role": "assistant", "content": "", "thinking": "", "tool_calls": []}
        for line in lines:
            m = line.get("message") or {}
            message["content"] += m.get("content", "")
            message["thinking"] += m.get("thinking", "")
            message["tool_calls"] += m.get("tool_calls", [])
        return {**lines[-1], "message": {k: v for k, v in message.items() if v}}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
