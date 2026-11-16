"""Ollama's own chat API under the OpenAI client, so that every request carries the context length and keep_alive.

Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) takes no context
length: it ignores `options.num_ctx`, so the server loads the model at its
default, which for a model of 262,144 tokens of context grew Ollama's server
from 20 GB to 34 GB in a test on a 64 GB Mac. Ollama's own chat API
(`/api/chat`) takes `options.num_ctx` and `keep_alive` in every request. The
Ollama preset therefore keeps the OpenAI client and Pydantic AI's OpenAI chat
model, and puts `OllamaNativeTransport` under the client: it sends each chat
request to `/api/chat` instead, translated, and translates the reply back into
the chat completions API's chunks, so the rest of the sidecar sees no
difference. Every other request (`GET /v1/models`) passes through unchanged.

What is translated:

- **The request.** Messages (system, developer as system, user, assistant and
  tool), their text parts, inline images as the message's `images`, the
  assistant's tool calls with their arguments as objects, the thinking sent
  back in `reasoning` or `reasoning_content` as `thinking`, and a tool result's
  call id and tool name; the tools; `max_tokens` or `max_completion_tokens` as
  `options.num_predict`; temperature, top_p, seed, stop and the penalties as
  options; `reasoning_effort` as `think`; a JSON schema response format as
  `format`. Every request gets `options.num_ctx` and `keep_alive`.
- **The reply.** Ollama streams lines of JSON; each becomes a chunk of the
  chat completions API: `thinking` as `reasoning` (the field Ollama's own
  OpenAI-compatible endpoint uses), text as `content`, each tool call with an
  index and an id (one is made when Ollama sends none), then the finish reason
  (`tool_calls` after a call, `length` when the output limit was reached, else
  `stop`) and the usage (`prompt_eval_count`, `eval_count`). A reply without
  streaming becomes one chat completion.
- **Errors.** Ollama's `{"error": "..."}` keeps its status and becomes the chat
  completions API's error object, so the OpenAI client raises the same
  exceptions and the provider's checks of the message (tools, images) still
  apply; an error in the middle of a stream becomes an error event.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import httpx2

DEFAULT_KEEP_ALIVE = "60s"
MAX_ERROR_CHARS = 500


def native_chat_url(base_url: str) -> str:
    """`/api/chat` of the server whose OpenAI-compatible base URL is `base_url` (`http://host:11434/v1`)."""
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return f"{parts.scheme}://{parts.netloc}{path}/api/chat"


def native_context(preset: Any, value: Any, source: str) -> int | None:
    """The context length a preset with Ollama's own chat API loads the model with: the person's (an override
    in the settings, or --context-length), else the preset's; None for any other preset."""
    if preset is None or getattr(preset, "chat_api", "openai") != "ollama":
        return None
    if source in ("override", "option") and isinstance(value, int) and value > 0:
        return value
    return preset.context_length


# ---- the request ---------------------------------------------------------------------------------


def _text_and_images(content: Any) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[str] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                images.append(url.split(";base64,", 1)[1])
            else:
                texts.append("[An image that is not inline was left out.]")
    return "\n\n".join(texts), images


def native_messages(messages: list) -> list[dict]:
    out: list[dict] = []
    names: dict[str, str] = {}   # tool call id -> tool name, for the results
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            continue
        role = "system" if m.get("role") == "developer" else m.get("role")
        text, images = _text_and_images(m.get("content"))
        n: dict = {"role": role, "content": text}
        if images:
            n["images"] = images
        thinking = m.get("reasoning") or m.get("reasoning_content")
        if isinstance(thinking, str) and thinking:
            n["thinking"] = thinking
        calls = []
        for c in m.get("tool_calls") or []:
            f = c.get("function") or {}
            try:
                args = json.loads(f.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            if isinstance(c.get("id"), str):
                names[c["id"]] = f.get("name", "")
            calls.append({"id": c.get("id"), "function": {"name": f.get("name", ""), "arguments": args}})
        if calls:
            n["tool_calls"] = calls
        if role == "tool":
            call_id = m.get("tool_call_id")
            if isinstance(call_id, str):
                n["tool_call_id"] = call_id
                if call_id in names:
                    n["tool_name"] = names[call_id]
        out.append(n)
    return out


def native_request(body: dict, *, num_ctx: int, keep_alive: str) -> dict:
    """The `/api/chat` request for a chat completions request."""
    options: dict = {"num_ctx": num_ctx}
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if isinstance(limit, int) and not isinstance(limit, bool):
        options["num_predict"] = limit
    for name in ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty"):
        if body.get(name) is not None:
            options[name] = body[name]
    stop = body.get("stop")
    if stop:
        options["stop"] = [stop] if isinstance(stop, str) else list(stop)
    out: dict = {"model": body.get("model"), "messages": native_messages(body.get("messages")),
                 "stream": bool(body.get("stream")), "options": options, "keep_alive": keep_alive}
    tools = [{"type": "function", "function": {k: v for k, v in (t.get("function") or {}).items()
                                               if k in ("name", "description", "parameters")}}
             for t in body.get("tools") or [] if isinstance(t, dict) and t.get("type") == "function"]
    if tools:
        out["tools"] = tools
    effort = body.get("reasoning_effort")
    if isinstance(effort, str):
        out["think"] = False if effort == "none" else effort if effort in ("low", "medium", "high") else True
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        out["format"] = (fmt.get("json_schema") or {}).get("schema")
    elif isinstance(fmt, dict) and fmt.get("type") == "json_object":
        out["format"] = "json"
    return out


# ---- the reply ------------------------------------------------------------------------------------


class _Reply:
    """Turns Ollama's reply objects into chat completion chunks, keeping the count of tool calls."""

    def __init__(self, model: str) -> None:
        self.id = "chatcmpl-" + uuid.uuid4().hex[:12]
        self.model = model
        self.created = int(time.time())
        self.calls = 0

    def chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def deltas(self, message: dict) -> list[dict]:
        out = []
        if isinstance(message.get("thinking"), str) and message["thinking"]:
            out.append({"reasoning": message["thinking"]})
        if isinstance(message.get("content"), str) and message["content"]:
            out.append({"content": message["content"]})
        for c in message.get("tool_calls") or []:
            f = c.get("function") or {}
            call_id = c.get("id") if isinstance(c.get("id"), str) and c.get("id") else "call_" + uuid.uuid4().hex[:12]
            args = f.get("arguments")
            out.append({"tool_calls": [{"index": self.calls, "id": call_id, "type": "function",
                                        "function": {"name": f.get("name", ""),
                                                     "arguments": args if isinstance(args, str)
                                                     else json.dumps(args if args is not None else {})}}]})
            self.calls += 1
        return out

    def finish(self, done: dict | None) -> str:
        if self.calls:
            return "tool_calls"
        return "length" if (done or {}).get("done_reason") == "length" else "stop"

    @staticmethod
    def usage(done: dict | None) -> dict:
        prompt = int((done or {}).get("prompt_eval_count") or 0)
        completion = int((done or {}).get("eval_count") or 0)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _error_body(message: Any) -> dict:
    return {"error": {"message": str(message)[:MAX_ERROR_CHARS], "type": "invalid_request_error", "code": None}}


def _event(doc: dict) -> bytes:
    return b"data: " + json.dumps(doc).encode() + b"\n\n"


class _SseStream(httpx2.AsyncByteStream):
    """The chat completions API's event stream, made from Ollama's streamed reply."""

    def __init__(self, upstream: httpx2.Response, model: str) -> None:
        self._upstream = upstream
        self._reply = _Reply(model)

    async def _lines(self) -> AsyncIterator[str]:
        buffer = b""
        async for piece in self._upstream.aiter_raw():
            buffer += piece
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8", "replace")
        if buffer.strip():
            yield buffer.decode("utf-8", "replace")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        r = self._reply
        yield _event(r.chunk({"role": "assistant"}))
        done = None
        async for line in self._lines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            if not isinstance(doc, dict):
                continue
            if "error" in doc:
                yield _event(_error_body(doc["error"]))
                return
            for delta in r.deltas(doc.get("message") or {}):
                yield _event(r.chunk(delta))
            if doc.get("done"):
                done = doc
                break
        if done is None:
            yield _event(_error_body("The model server's reply ended before it was done."))
            return
        yield _event(r.chunk({}, r.finish(done)))
        yield _event({"id": r.id, "object": "chat.completion.chunk", "created": r.created, "model": r.model,
                      "choices": [], "usage": r.usage(done)})
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        await self._upstream.aclose()


def _completion(doc: dict, model: str) -> dict:
    r = _Reply(model)
    message = doc.get("message") or {}
    out: dict = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("thinking"):
        out["reasoning"] = message["thinking"]
    calls = [d["tool_calls"][0] for d in r.deltas({"tool_calls": message.get("tool_calls")}) if "tool_calls" in d]
    if calls:
        out["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
    return {"id": r.id, "object": "chat.completion", "created": r.created, "model": model,
            "choices": [{"index": 0, "message": out, "finish_reason": r.finish(doc)}], "usage": r.usage(doc)}


# ---- the transport --------------------------------------------------------------------------------


class OllamaNativeTransport(httpx2.AsyncBaseTransport):
    """Sends the OpenAI client's chat requests to Ollama's `/api/chat`, with `num_ctx` and `keep_alive`."""

    def __init__(self, base_url: str, *, num_ctx: int, keep_alive: str = DEFAULT_KEEP_ALIVE,
                 inner: httpx2.AsyncBaseTransport | None = None) -> None:
        self.chat_url = native_chat_url(base_url)
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self._inner = inner if inner is not None else httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return await self._inner.handle_async_request(request)
        try:
            body = json.loads(await request.aread())
        except ValueError:
            return httpx2.Response(400, json=_error_body("The request is not JSON."), request=request)
        native = native_request(body, num_ctx=self.num_ctx, keep_alive=self.keep_alive)
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("content-length", "host", "accept-encoding", "accept")}
        headers["content-type"] = "application/json"
        upstream_request = httpx2.Request("POST", self.chat_url, headers=headers,
                                          content=json.dumps(native).encode(), extensions=request.extensions)
        upstream = await self._inner.handle_async_request(upstream_request)
        model = str(body.get("model") or "")
        if upstream.status_code != 200:
            raw = await upstream.aread()
            await upstream.aclose()
            try:
                message = json.loads(raw).get("error", raw.decode("utf-8", "replace"))
            except (ValueError, AttributeError):
                message = raw.decode("utf-8", "replace") or f"HTTP {upstream.status_code}"
            return httpx2.Response(upstream.status_code, json=_error_body(message), request=request)
        if native["stream"]:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                   stream=_SseStream(upstream, model), request=request)
        raw = await upstream.aread()
        await upstream.aclose()
        try:
            doc = json.loads(raw)
        except ValueError:
            return httpx2.Response(502, json=_error_body("The model server's reply is not JSON."), request=request)
        if "error" in doc:
            return httpx2.Response(500, json=_error_body(doc["error"]), request=request)
        return httpx2.Response(200, json=_completion(doc, model), request=request)

    async def aclose(self) -> None:
        await self._inner.aclose()
