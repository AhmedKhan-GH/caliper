"""The capability probe: what a model can do, learned from its server with a few cheap requests.

The widget agent's settings run it when the person picks a model
(design/widget-agent.md, Capabilities and degradations), through the
sidecar's command line, `caliper-agent probe` (cli.py), which prints one JSON
object. What it learns is cached (capabilities.py) and read by the sidecar at
start, so a model without tool calling is refused before any session, a model
without vision gets no images, and the app sizes the context bundle to the
model's context length.

**An OpenAI-compatible server** gets at most four requests, each bounded in
time and in output tokens:

1. `GET <base>/models`, where the server offers it, for the list of models
   and, where the entry of the chosen model carries one, its context length
   (`context_length`, `context_window`, `max_context_length` or
   `max_model_len`, whichever the server sends).
2. A streamed chat request that offers one tool, `record_number`, and asks the
   model to call it with the number 7. Tool calling works when the streamed
   call assembles to that tool with that number. A 400 that names tools, or a
   reply without the call, means no tool calling. A 400 that names `strict`
   is retried once without it, and a 400 that names `max_completion_tokens`
   is retried once with `max_tokens`; either is recorded as a quirk that the
   provider then follows.
3. A streamed chat request with a 16 by 16 PNG of one solid colour, chosen at
   random from four, that the model must name. Vision works when the reply
   names that colour. A 400 that names images means no vision.
4. For the LM Studio preset only, its own `GET /api/v0/models/<model>`, whose
   `max_context_length` LM Studio documents.
5. For the Ollama preset and any server on loopback, Ollama's own
   `POST /api/show`, which reads the model's metadata without loading it: the
   number of attention layers (every one, or one in `full_attention_interval`
   for a hybrid model), the key and value heads and their sizes give the bytes
   of context cache one token takes (`kv_bytes_per_token`, kept with the
   quirks), from which the app's settings estimate the memory of the context
   cache. Its `context_length` is the most the model takes, not the context
   the server runs it with, so it is never taken as the context length. For a
   server on loopback other than the Ollama preset, after the test calls have
   loaded the model, Ollama's `GET /api/ps` states the context it runs the
   model with, and that is the context length; otherwise it stays unknown. A
   server that is not Ollama answers 404, and nothing is learned.

For the Ollama preset the test calls go to Ollama's own `/api/chat`
(ollama_native.py) with the context length the agent will use, so the probe
loads the model as the agent will, and that context length is the answer.

The probe also records where the server put the model's reasoning, if it sent
any: the `reasoning` field, the `reasoning_content` field, or `<think>` tags in
the text. Pydantic AI reads all three as thoughts, so this is shown, not used.

**The Claude API** gets no test call, since every model it offers calls tools:
the probe reads the Models API (`GET /v1/models` and `GET /v1/models/<id>`),
whose answer gives the context length (`max_input_tokens`) and image input
(`capabilities.image_input.supported`) at no cost in tokens.

A failure that says nothing about the model (a connection error, a timeout, a
401, a 5xx) leaves that capability unknown, with a scrubbed reason; the sidecar
then assumes the capability until the server says otherwise, as before.
Nothing the probe returns carries a key: every message passes through the
scrubber, and the key reaches the SDK client as an explicit argument only.
"""

from __future__ import annotations

import base64
import json
import random
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import httpx2

from .keyguard import Secret, scrub
from .offline import OfflineError, check_provider, is_loopback_host, trust_env

PROBE_VERSION = 1
PROBE_TIMEOUT_S = 60.0       # one test call, including the model's thinking
LIST_TIMEOUT_S = 10.0        # the list of models and the server's own model entry
CONNECT_TIMEOUT_S = 10.0
PROBE_MAX_TOKENS = 1024      # enough for a short thought and one tool call
MAX_LISTED_MODELS = 1000
MAX_ID_CHARS = 256
MAX_ERROR_CHARS = 300
MAX_STREAM_CHARS = 64 << 10  # what the probe reads of one streamed reply

PROBE_TOOL = "record_number"
PROBE_NUMBER = 7
PROBE_TOOL_PROMPT = (f"Call the tool {PROBE_TOOL} with the number {PROBE_NUMBER}. "
                     "Do not answer in text.")
PROBE_VISION_PROMPT = "What colour is this image? Answer with one lowercase English word."
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30), "green": (30, 170, 40), "blue": (30, 60, 220), "yellow": (240, 215, 20)}

# The fields of a model entry in `GET /models` that some servers use for the
# context length. Only LM Studio's `max_context_length` (in its own REST API)
# was read in a server's documentation; the others are read when a server
# sends them.
CONTEXT_FIELDS = ("context_length", "context_window", "max_context_length", "max_model_len")


def solid_png(rgb: tuple[int, int, int], size: int = 16) -> bytes:
    """A PNG of one solid colour."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    row = b"\x00" + bytes(rgb) * size
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


def now_iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


@dataclass
class Capability:
    """One answer: the value (None when unknown), where it came from, and why."""

    value: Any = None
    source: str = "unknown"      # probe, server, known, override, option, default, unknown
    detail: str = ""

    def as_dict(self) -> dict:
        out: dict = {"value": self.value, "source": self.source}
        if self.detail:
            out["detail"] = self.detail
        return out

    @staticmethod
    def from_dict(doc: Any) -> "Capability":
        if not isinstance(doc, dict):
            return Capability()
        return Capability(doc.get("value"), str(doc.get("source", "unknown"))[:32],
                          str(doc.get("detail", ""))[:MAX_ERROR_CHARS])


@dataclass
class ProbeResult:
    provider: str                 # "claude" or "openai-compatible"
    base_url: str
    model: str
    probed_at: float = field(default_factory=time.time)
    models: list[str] | None = None
    tools: Capability = field(default_factory=Capability)
    vision: Capability = field(default_factory=Capability)
    context_length: Capability = field(default_factory=Capability)
    quirks: dict = field(default_factory=dict)       # strict, max_tokens_field, reasoning_field, streamed_arguments
    errors: list[dict] = field(default_factory=list)

    def error(self, capability: str, code: str, message: str) -> None:
        self.errors.append({"capability": capability, "code": code, "message": scrub(message)[:MAX_ERROR_CHARS]})

    def capabilities(self) -> dict:
        return {"tools": self.tools.as_dict(), "vision": self.vision.as_dict(),
                "context_length": self.context_length.as_dict()}

    def as_dict(self) -> dict:
        return {"version": PROBE_VERSION, "provider": self.provider, "base_url": self.base_url, "model": self.model,
                "probed_at": now_iso(self.probed_at), "models": self.models, "capabilities": self.capabilities(),
                "quirks": dict(self.quirks), "errors": list(self.errors)}


# ---- helpers -----------------------------------------------------------------------------------------


def _status_and_message(exc: BaseException) -> tuple[int | None, str]:
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message = err["message"]
        elif isinstance(err, str):
            message = err
    return (status if isinstance(status, int) else None), scrub(str(message))[:MAX_ERROR_CHARS]


def _failure_code(exc: BaseException) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if "Timeout" in name:
        return "probe.timeout"
    if "Connection" in name:
        return "probe.unreachable"
    if status == 401:
        return "probe.key_rejected"
    if status == 403:
        return "probe.forbidden"
    if status == 404:
        return "probe.model_unavailable"
    if isinstance(status, int) and status >= 500:
        return "probe.server_error"
    return "probe.failed"


def _names_tools(message: str) -> bool:
    text = message.lower()
    return "tool" in text or "function call" in text or "function_call" in text


def _names_images(message: str) -> bool:
    text = message.lower()
    return "image" in text or "vision" in text or "multimodal" in text


def _context_from_entry(entry: Any) -> int | None:
    if not isinstance(entry, dict):
        return None
    for name in CONTEXT_FIELDS:
        value = entry.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 100_000_000:
            return value
    return None


def _openai_client(base_url: str, key: Secret | None, transport: Any = None, native: dict | None = None):
    from openai import AsyncOpenAI
    if native is not None:
        # Ollama: the test calls go to /api/chat with the preset's context length,
        # so the probe loads the model as the agent will (ollama_native.py).
        from .ollama_native import DEFAULT_KEEP_ALIVE, OllamaNativeTransport
        transport = OllamaNativeTransport(base_url, num_ctx=native["num_ctx"],
                                          keep_alive=native.get("keep_alive", DEFAULT_KEEP_ALIVE), inner=transport)
    return AsyncOpenAI(base_url=base_url, api_key=key.reveal() if key is not None else "not-needed", max_retries=0,
                       timeout=httpx2.Timeout(PROBE_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                       http_client=(httpx2.AsyncClient(transport=transport, trust_env=trust_env())
                                    if transport is not None or not trust_env() else None))


@dataclass
class _Reply:
    text: str = ""
    calls: dict = field(default_factory=dict)        # index -> {"name", "arguments", "chunks"}
    reasoning_field: str | None = None
    finish: str | None = None


async def _stream_chat(client: Any, model: str, messages: list, *, tools: list | None, token_field: str) -> _Reply:
    """One streamed chat request, assembled; arguments may arrive whole or in pieces."""
    kwargs: dict = {"model": model, "messages": messages, "stream": True, token_field: PROBE_MAX_TOKENS}
    if tools is not None:
        kwargs["tools"] = tools
    reply = _Reply()
    stream = await client.chat.completions.create(**kwargs)
    read = 0
    async for chunk in stream:
        for choice in chunk.choices or []:
            delta = choice.delta
            if choice.finish_reason:
                reply.finish = choice.finish_reason
            if delta is None:
                continue
            if delta.content:
                reply.text += delta.content
                read += len(delta.content)
            extra = getattr(delta, "model_extra", None) or {}
            for name in ("reasoning", "reasoning_content"):
                value = extra.get(name)
                if isinstance(value, str) and value:
                    reply.reasoning_field = reply.reasoning_field or name
                    read += len(value)
            for call in delta.tool_calls or []:
                slot = reply.calls.setdefault(call.index, {"name": "", "arguments": "", "chunks": 0})
                if call.function is not None:
                    if call.function.name:
                        slot["name"] = call.function.name
                    if call.function.arguments:
                        slot["arguments"] += call.function.arguments
                        slot["chunks"] += 1
                        read += len(call.function.arguments)
        if read > MAX_STREAM_CHARS:
            break
    if reply.reasoning_field is None and "<think>" in reply.text:
        reply.reasoning_field = "think_tags"
    return reply


def _visible_text(text: str) -> str:
    """The reply without a `<think>` block."""
    while "<think>" in text:
        start = text.index("<think>")
        end = text.find("</think>", start)
        text = text[:start] + (text[end + len("</think>"):] if end >= 0 else "")
    return text


def _called_with_number(reply: _Reply) -> bool:
    for call in reply.calls.values():
        if call["name"] != PROBE_TOOL:
            continue
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            continue
        value = args.get("number") if isinstance(args, dict) else None
        if value == PROBE_NUMBER or value == str(PROBE_NUMBER):
            return True
    return False


# ---- an OpenAI-compatible server ---------------------------------------------------------------------


def _tool_definition(strict: bool) -> dict:
    function: dict = {"name": PROBE_TOOL, "description": "Records one number.",
                      "parameters": {"type": "object", "properties": {"number": {"type": "integer"}},
                                     "required": ["number"], "additionalProperties": False}}
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


async def _list_openai_models(client: Any, result: ProbeResult) -> dict | None:
    """Fills `result.models`; returns the chosen model's entry as the server sent it."""
    chosen = None
    models: list[str] = []
    try:
        async for m in client.models.list(timeout=LIST_TIMEOUT_S):
            mid = getattr(m, "id", None)
            if isinstance(mid, str) and mid and len(mid) <= MAX_ID_CHARS:
                models.append(mid)
                if mid == result.model:
                    chosen = {**(getattr(m, "model_extra", None) or {}), "id": mid}
            if len(models) >= MAX_LISTED_MODELS:
                break
    except Exception as exc:  # the list is optional: many servers have none
        status, message = _status_and_message(exc)
        result.error("models", _failure_code(exc), message)
        return None
    result.models = models
    return chosen


async def _lmstudio_context(base_url: str, model: str, transport: Any, result: ProbeResult) -> int | None:
    parts = urlsplit(base_url)
    url = f"{parts.scheme}://{parts.netloc}/api/v0/models/{quote(model, safe='')}"
    try:
        async with httpx2.AsyncClient(transport=transport, timeout=LIST_TIMEOUT_S, trust_env=False) as http:
            response = await http.get(url)
        if response.status_code != 200 or len(response.content) > (1 << 20):
            return None
        return _context_from_entry(response.json())
    except Exception as exc:
        result.error("context_length", _failure_code(exc), _status_and_message(exc)[1])
        return None


def _kv_bytes_per_token(info: dict) -> int | None:
    """The bytes of context cache (keys and values at 16 bits) one token takes, from Ollama's model metadata."""
    arch = info.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None

    def number(name: str) -> int | None:
        value = info.get(f"{arch}.{name}")
        if isinstance(value, list):   # per layer in some models: the largest
            value = max((v for v in value if isinstance(v, int)), default=None)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    layers, kv_heads = number("block_count"), number("attention.head_count_kv")
    key, value = number("attention.key_length"), number("attention.value_length")
    if key is None:
        heads, width = number("attention.head_count"), number("embedding_length")
        key = value = width // heads if heads and width else None
    if not layers or not kv_heads or not key:
        return None
    interval = number("full_attention_interval") or 1
    attention_layers = -(-layers // interval)
    return attention_layers * kv_heads * (key + (value or key)) * 2


async def _ollama_context_in_force(base_url: str, model: str, transport: Any) -> int | None:
    """The context length Ollama runs a loaded model with (`GET /api/ps`), or None: another server answers 404."""
    parts = urlsplit(base_url.rstrip("/"))
    root = parts.path[: -len("/v1")] if parts.path.endswith("/v1") else parts.path
    try:
        async with httpx2.AsyncClient(transport=transport, timeout=LIST_TIMEOUT_S, trust_env=False) as http:
            response = await http.get(f"{parts.scheme}://{parts.netloc}{root}/api/ps")
        if response.status_code != 200 or len(response.content) > (1 << 20):
            return None
        loaded = response.json().get("models")
    except Exception:
        return None
    for m in loaded if isinstance(loaded, list) else []:
        if isinstance(m, dict) and model in (m.get("name"), m.get("model")):
            value = m.get("context_length")
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 100_000_000:
                return value
    return None


async def _ollama_show(base_url: str, model: str, transport: Any, result: ProbeResult) -> int | None:
    """Ollama's model metadata: fills kv_bytes_per_token, and returns the model's context length."""
    parts = urlsplit(base_url.rstrip("/"))
    root = parts.path[: -len("/v1")] if parts.path.endswith("/v1") else parts.path
    try:
        async with httpx2.AsyncClient(transport=transport, timeout=LIST_TIMEOUT_S, trust_env=False) as http:
            response = await http.post(f"{parts.scheme}://{parts.netloc}{root}/api/show", json={"model": model})
        if response.status_code != 200 or len(response.content) > (16 << 20):
            return None
        info = response.json().get("model_info")
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    per_token = _kv_bytes_per_token(info)
    if per_token is not None:
        result.quirks["kv_bytes_per_token"] = per_token
    arch = info.get("general.architecture")
    context = info.get(f"{arch}.context_length") if isinstance(arch, str) else None
    return context if isinstance(context, int) and not isinstance(context, bool) and 0 < context <= 100_000_000 \
        else None


async def _probe_tools(client: Any, result: ProbeResult) -> None:
    messages = [{"role": "user", "content": PROBE_TOOL_PROMPT}]
    strict, token_field = True, "max_completion_tokens"
    answered = False
    for _attempt in range(3):
        try:
            reply = await _stream_chat(client, result.model, messages, tools=[_tool_definition(strict)],
                                       token_field=token_field)
        except Exception as exc:
            status, message = _status_and_message(exc)
            lowered = message.lower()
            if status == 400 and strict and "strict" in lowered:
                strict = False
                continue
            if status == 400 and token_field == "max_completion_tokens" and "max_completion_tokens" in lowered:
                token_field = "max_tokens"
                continue
            if status in (400, 422) and _names_tools(message):
                result.tools = Capability(False, "probe", "The server refused a request with a tool: " + message)
            else:
                result.error("tools", _failure_code(exc), message)
            break
        else:
            answered = True
            if reply.reasoning_field:
                result.quirks["reasoning_field"] = reply.reasoning_field
            if _called_with_number(reply):
                result.tools = Capability(True, "probe", f"The model called {PROBE_TOOL} as asked.")
                chunks = max(c["chunks"] for c in reply.calls.values())
                result.quirks["streamed_arguments"] = "whole" if chunks <= 1 else "pieces"
            elif reply.calls:
                result.tools = Capability(False, "probe", "The model called a tool, but not as asked.")
            elif reply.finish == "length":
                # A thinking model can spend the output limit before it calls the
                # tool; that says nothing about tool calling.
                result.error("tools", "probe.cut_off", "The reply reached the probe's output limit before it "
                                                       "called the tool.")
            else:
                result.tools = Capability(False, "probe", "The model answered in text instead of calling the tool.")
            break
    if answered or result.tools.value is False:
        # What the server took; unknown when no request got through.
        result.quirks["strict"] = strict
        result.quirks["max_tokens_field"] = token_field


async def _probe_vision(client: Any, result: ProbeResult, colour: str) -> None:
    png = base64.b64encode(solid_png(COLOURS[colour])).decode("ascii")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROBE_VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]
    token_field = result.quirks.get("max_tokens_field", "max_completion_tokens")
    try:
        reply = await _stream_chat(client, result.model, messages, tools=None, token_field=token_field)
    except Exception as exc:
        status, message = _status_and_message(exc)
        if status in (400, 422) and _names_images(message):
            result.vision = Capability(False, "probe", "The server refused a request with an image: " + message)
        else:
            result.error("vision", _failure_code(exc), message)
        return
    if reply.reasoning_field and "reasoning_field" not in result.quirks:
        result.quirks["reasoning_field"] = reply.reasoning_field
    words = "".join(c if c.isalpha() else " " for c in _visible_text(reply.text).lower()).split()
    if colour not in words and reply.finish == "length":
        result.error("vision", "probe.cut_off", "The reply reached the probe's output limit before it named a colour.")
        return
    if colour in words:
        result.vision = Capability(True, "probe", f"The model named the colour of a {colour} image.")
    else:
        result.vision = Capability(False, "probe", f"The model did not name the colour of a {colour} image.")


async def probe_openai_compatible(base_url: str, model: str, key: Secret | None, *, preset: str | None = None,
                                  transport: Any = None, vision: bool = True, list_only: bool = False,
                                  colour: str | None = None, native: dict | None = None) -> ProbeResult:
    """Probes one model of an OpenAI-compatible server; with `native` ({"num_ctx", "keep_alive"}), an Ollama
    server through its own chat API, with that context length."""
    result = ProbeResult("openai-compatible", base_url, model)
    try:
        check_provider("openai-compatible", base_url=base_url, transport=transport)
    except OfflineError as exc:
        result.error("models", exc.code, str(exc))
        return result
    client = _openai_client(base_url, key, transport, native)
    try:
        entry = await _list_openai_models(client, result)
        if list_only:
            return result
        context = _context_from_entry(entry)
        if context is None and preset == "lmstudio":
            context = await _lmstudio_context(base_url, model, transport, result)
        local = native is not None or is_loopback_host(urlsplit(base_url).hostname)
        if local:
            # The cache a token takes. The metadata's context length is the most
            # the model takes, not what the server runs it with, so it is not used.
            await _ollama_show(base_url, model, transport, result)
        if native is not None:
            result.context_length = Capability(native["num_ctx"], "preset",
                                               "The context length the preset loads the model with.")
        elif context is not None:
            result.context_length = Capability(context, "server", "Reported by the server's list of models.")
        await _probe_tools(client, result)
        if vision:
            await _probe_vision(client, result, colour or random.choice(sorted(COLOURS)))
        else:
            result.vision = Capability(None, "unknown", "The vision probe was not run.")
        if local and native is None and context is None:
            # The test calls loaded the model: Ollama's /api/ps states the context it runs it with.
            in_force = await _ollama_context_in_force(base_url, model, transport)
            if in_force is not None:
                result.context_length = Capability(in_force, "server",
                                                   "The context the server runs the model with (Ollama's /api/ps).")
        return result
    finally:
        await client.close()


# ---- the Claude API ----------------------------------------------------------------------------------


async def probe_claude(model: str, key: Secret | None, *, transport: Any = None,
                       list_only: bool = False) -> ProbeResult:
    """Reads the Claude API's Models API: no test call, no tokens."""
    from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
    import anthropic

    from .claude import API_BASE_URL

    result = ProbeResult("claude", API_BASE_URL, model)
    try:
        check_provider("claude", transport=transport)
    except OfflineError as exc:
        result.error("models", exc.code, str(exc))
        return result
    if key is None:
        result.error("models", "probe.no_key", "No Claude API key is set.")
        return result
    client = AsyncAnthropic(api_key=key.reveal(), base_url=API_BASE_URL, max_retries=0,
                            timeout=anthropic.Timeout(LIST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                            open_telemetry=False,
                            http_client=(DefaultAsyncHttpxClient(transport=transport, trust_env=trust_env())
                                         if transport is not None or not trust_env() else None))
    try:
        models: list[str] = []
        try:
            async for m in client.models.list(limit=100):
                if isinstance(m.id, str) and len(m.id) <= MAX_ID_CHARS:
                    models.append(m.id)
                if len(models) >= MAX_LISTED_MODELS:
                    break
            result.models = models
        except Exception as exc:
            result.error("models", _failure_code(exc), _status_and_message(exc)[1])
        if list_only:
            return result
        result.tools = Capability(True, "known", "Every model the Claude API offers calls tools.")
        try:
            info = await client.models.retrieve(model)
        except Exception as exc:
            code = _failure_code(exc)
            result.error("vision", code, _status_and_message(exc)[1])
            if code == "probe.model_unavailable":
                result.tools = Capability(None, "unknown", "The Claude API does not offer this model to this key.")
            return result
        if isinstance(info.max_input_tokens, int) and info.max_input_tokens > 0:
            result.context_length = Capability(info.max_input_tokens, "server", "From the Models API.")
        caps = info.capabilities
        if caps is not None and caps.image_input is not None:
            result.vision = Capability(bool(caps.image_input.supported), "server", "From the Models API.")
        return result
    finally:
        await client.close()
