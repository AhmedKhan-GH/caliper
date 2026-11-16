"""The Claude API provider: every Claude-specific detail of the sidecar lives here.

One request per call of `ClaudeProvider.request()`, through the official
`anthropic` SDK (2.x), always streamed:

- `client.beta.messages.stream(...)`, iterated for the pane and closed with
  `get_final_message()`; the beta namespace because server-side refusal
  fallbacks and compaction are betas.
- Adaptive thinking with summarized display, so the pane can show the model's
  reasoning as thoughts; `output_config.effort` set explicitly on every request.
- Tools from the command table, `strict` where the schema allows it and with
  `eager_input_streaming`; `tool_choice` auto (forced tool use is refused by
  these models). No assistant prefill.
- Prompt caching: the tools and the system prompt come first and never change
  within a conversation; a breakpoint on the system block caches both, and the
  request's top-level `cache_control` caches the conversation up to its last
  block. Nothing volatile (no time, no ids) is placed before a breakpoint.
- Server-side refusal fallbacks (`fallbacks: "default"`, beta
  `server-side-fallback-2026-07-01`) on every model that has them; Claude
  Haiku 5.5 has none, so its requests carry neither.
- Server-side compaction (beta `compact-2026-01-12`, the `compact_20260112`
  edit at its default trigger) for long sessions, unless turned off.
- The SDK's own retries are off (`max_retries=0`): this module retries
  rate limits, overloads, server errors and broken connections itself, so that
  it can tell the pane what it is waiting for. A request is retried only while
  nothing of it has reached the pane.
- The full response content is appended to the history, except that after a
  mid-reply fallback the blocks of the declining model before the last
  `fallback` block are left out as the API requires; a final refusal is not
  appended at all.

The client gets the key as an explicit argument, so the SDK never looks for
another credential (a profile, a token, federation), and the base URL is fixed
to the Claude API's, so no environment variable can send a request elsewhere.
"""

from __future__ import annotations

import asyncio
import base64
import email.utils
import json
import random
import time
from typing import Any

import anthropic
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from .keyguard import Secret, describe_exception, scrub
from .provider import AgentError, EventSink, ModelInfo, RequestResult, Settings, ToolCall, ToolOutcome
from .tools import ToolSpec

API_BASE_URL = "https://api.anthropic.com"
API_HOST = "api.anthropic.com"

MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("claude-opus-5-5", "Claude Opus 5.5", "The default: strong at building and checking visuals."),
    ModelInfo("claude-sonnet-5-5", "Claude Sonnet 5.5", "Faster and cheaper, for everyday edits."),
    ModelInfo("claude-haiku-5-5", "Claude Haiku 5.5", "The fastest and cheapest, for small changes."),
    ModelInfo("claude-fable-5-2", "Claude Fable 5.2", "The most capable, and the most expensive."),
)
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "high"

# Models without server-side refusal fallbacks: the parameter is left out.
NO_SERVER_FALLBACK = frozenset({"claude-haiku-5-5"})

MAX_TOKENS = 64000
BETA_FALLBACK = "server-side-fallback-2026-07-01"
BETA_COMPACTION = "compact-2026-01-12"

MAX_ATTEMPTS = 5
MAX_RETRY_AFTER_S = 60.0
MAX_BACKOFF_S = 30.0
MAX_ERROR_TEXT = 500

_STOPS = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "context_full",
    "refusal": "refusal",
    "pause_turn": "pause",
    "compaction": "pause",
}

_RETRYABLE_STREAM_ERRORS = frozenset({"overloaded_error", "api_error", "rate_limit_error", "timeout_error"})


def model_ids() -> tuple[str, ...]:
    return tuple(m.id for m in MODELS)


def api_tool(spec: ToolSpec, strict: bool) -> dict:
    tool = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "eager_input_streaming": True,
    }
    if strict and spec.strict:
        tool["strict"] = True
    return tool


def echo_content(blocks: list[dict]) -> list[dict]:
    """The content of a response as it goes back into the history.

    Normally every block, unchanged. After a mid-reply fallback, the blocks of
    the declining model before the last `fallback` block are left out, except
    text, the fallback blocks themselves, compaction blocks and server-tool
    blocks that have their result; the `fallback` block keeps its place.
    """
    last = -1
    for i, block in enumerate(blocks):
        if block.get("type") == "fallback":
            last = i
    if last < 0:
        return list(blocks)
    paired = set()
    for block in blocks[:last]:
        t = str(block.get("type", ""))
        if t.endswith("_tool_result") and isinstance(block.get("tool_use_id"), str):
            paired.add(block["tool_use_id"])
    out = []
    for i, block in enumerate(blocks):
        t = block.get("type")
        if i < last:
            if t in ("text", "fallback", "compaction"):
                pass
            elif t == "server_tool_use" and block.get("id") in paired:
                pass
            elif str(t).endswith("_tool_result") and block.get("tool_use_id") in paired:
                pass
            else:
                continue
        out.append(block)
    return out


class ClaudeConversation:
    """The history of one session, append-only.

    Messages are appended and never edited, which keeps the prompt cache warm and
    keeps every thinking block valid for the models that check the history
    before them. The one removal is `rollback_turn()`, after a refusal: ACP's
    `refusal` stop reason means that the prompt and everything after it are left
    out of the next request, so the history is cut back to what it was before
    the prompt. What remains is a prefix of what was sent before, so every
    thinking block in it still sees the same history and the cache still holds.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.strict = True
        self._prompt_index: int | None = None

    def add_user_prompt(self, text: str) -> None:
        self._prompt_index = len(self.messages)
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def rollback_turn(self) -> bool:
        if self._prompt_index is None or self._prompt_index >= len(self.messages):
            return False
        del self.messages[self._prompt_index:]
        self._prompt_index = None
        return True

    def add_assistant(self, content: list[dict]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, outcomes: list[ToolOutcome]) -> None:
        if not outcomes:
            return
        blocks = []
        for o in outcomes:
            content: list[dict] = [{"type": "text", "text": o.text or ("Error." if o.is_error else "Done.")}]
            if o.png is not None and not o.is_error:
                content.append({"type": "image",
                                "source": {"type": "base64", "media_type": "image/png",
                                           "data": base64.b64encode(o.png).decode("ascii")}})
            block = {"type": "tool_result", "tool_use_id": o.call_id, "content": content}
            if o.is_error:
                block["is_error"] = True
            blocks.append(block)
        self.messages.append({"role": "user", "content": blocks})

    @property
    def has_assistant_turn(self) -> bool:
        return any(m["role"] == "assistant" for m in self.messages)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, IndexError):
        return None


def _api_message(exc: Exception) -> str:
    """The API's own error message, scrubbed and bounded."""
    body = getattr(exc, "body", None)
    message = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message = err["message"]
    if message is None:
        message = getattr(exc, "message", None) or str(exc)
    message = scrub(str(message)).strip()
    return message[:MAX_ERROR_TEXT]


def _stream_error_type(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str):
            return err["type"]
    return None


class _Retry(Exception):
    def __init__(self, delay: float | None, kind: str, text: str, http_status: int | None = None) -> None:
        super().__init__(text)
        self.delay = delay
        self.kind = kind
        self.text = text
        self.http_status = http_status


class ClaudeProvider:
    models = MODELS
    default_model = DEFAULT_MODEL

    def __init__(self, key: Secret, tools: list[ToolSpec], system_prompt: str, *,
                 transport: Any = None, compaction: bool = True, backoff_scale: float = 1.0) -> None:
        http_client = DefaultAsyncHttpxClient(transport=transport) if transport is not None else None
        self._client = AsyncAnthropic(
            api_key=key.reveal(),
            base_url=API_BASE_URL,
            max_retries=0,
            timeout=anthropic.Timeout(600.0, connect=10.0),
            open_telemetry=False,
            http_client=http_client,
        )
        self._tools_strict = [api_tool(t, strict=True) for t in tools]
        self._tools_plain = [api_tool(t, strict=False) for t in tools]
        self._system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        self._compaction = compaction
        self._backoff_scale = max(0.0, backoff_scale)

    def new_conversation(self) -> ClaudeConversation:
        return ClaudeConversation()

    async def aclose(self) -> None:
        await self._client.close()

    def params(self, conversation: ClaudeConversation, settings: Settings) -> dict:
        """The request, in a fixed order: tools, system, then the conversation."""
        params: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": MAX_TOKENS,
            "tools": self._tools_strict if conversation.strict else self._tools_plain,
            "tool_choice": {"type": "auto"},
            "system": self._system,
            "messages": conversation.messages,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": settings.effort},
            "cache_control": {"type": "ephemeral"},
        }
        betas = []
        if settings.model not in NO_SERVER_FALLBACK:
            params["fallbacks"] = "default"
            betas.append(BETA_FALLBACK)
        if self._compaction:
            params["context_management"] = {"edits": [{"type": "compact_20260112"}]}
            betas.append(BETA_COMPACTION)
        if betas:
            params["betas"] = betas
        return params

    def _backoff(self, kind: str, attempt: int) -> float:
        base = {"connection": 1.0, "server": 2.0, "rate_limit": 5.0, "bad_json": 0.5}.get(kind, 2.0)
        delay = min(MAX_BACKOFF_S, base * (2 ** (attempt - 1)))
        return delay * random.uniform(0.75, 1.25) * self._backoff_scale

    def _classify(self, exc: Exception, settings: Settings, conversation: ClaudeConversation) -> _Retry:
        """Raises AgentError for a failure that retrying cannot fix; returns _Retry otherwise.

        The order follows the SDK's error chain: rate limit, then the other status
        errors, then the connection errors.
        """
        if isinstance(exc, anthropic.RateLimitError):
            return _Retry(_retry_after(exc), "rate_limit", "is rate limited (HTTP 429)", 429)
        if isinstance(exc, anthropic.APIStatusError):
            status = exc.status_code
            detail = _api_message(exc)
            if status == 401:
                raise AgentError("agent.key_rejected",
                                 "The Claude API did not accept the API key (HTTP 401). "
                                 "Check or replace the key in the agent pane's settings.", rpc_code=-32000,
                                 data={"http_status": 401})
            if status == 402:
                raise AgentError("agent.billing", f"The Claude API refused the request for billing reasons (HTTP 402): {detail}",
                                 data={"http_status": 402})
            if status == 403:
                raise AgentError("agent.forbidden", f"The Claude API refused the request for this key (HTTP 403): {detail}",
                                 data={"http_status": 403})
            if status == 404:
                raise AgentError("agent.model_unavailable",
                                 f"The model {settings.model} is not available to this key (HTTP 404): {detail}",
                                 data={"http_status": 404})
            if status == 413:
                raise AgentError("agent.request_too_large",
                                 "The conversation is too large for one request (HTTP 413). Start a new session.",
                                 data={"http_status": 413})
            if status == 400:
                raise AgentError("agent.bad_request", f"The Claude API refused the request (HTTP 400): {detail}",
                                 data={"http_status": 400})
            if status in (408, 409) or status >= 500:
                return _Retry(_retry_after(exc), "server", f"answered HTTP {status}", status)
            if status == 200:
                kind = _stream_error_type(exc)
                if kind in _RETRYABLE_STREAM_ERRORS:
                    return _Retry(_retry_after(exc), "server", f"reported {kind} during the reply", None)
                raise AgentError("agent.stream_error", f"The Claude API reported an error during the reply: {detail}")
            raise AgentError("agent.api_error", f"The Claude API answered HTTP {status}: {detail}",
                             data={"http_status": status})
        if isinstance(exc, anthropic.APITimeoutError):
            return _Retry(None, "connection", "did not answer in time")
        if isinstance(exc, anthropic.APIConnectionError):
            return _Retry(None, "connection", "could not be reached")
        if isinstance(exc, ValueError):
            return _Retry(None, "bad_json", "sent tool input that could not be parsed")
        raise AgentError("agent.internal", f"The request failed: {describe_exception(exc)}")

    async def request(self, conversation: ClaudeConversation, settings: Settings, sink: EventSink) -> RequestResult:
        attempt = 0
        started = time.monotonic()
        while True:
            attempt += 1
            reached_pane = [False]
            try:
                message = await self._stream_once(conversation, settings, sink, reached_pane)
                break
            except asyncio.CancelledError:
                raise
            except AgentError:
                raise
            except Exception as exc:  # the SDK's typed errors, and ValueError from its parser
                if (isinstance(exc, anthropic.BadRequestError) and conversation.strict
                        and not conversation.has_assistant_turn and "strict" in _api_message(exc).lower()):
                    conversation.strict = False
                    sink.status("strict_off", "The Claude API refused the strict tool schemas; this session "
                                              "continues without strict mode, and Caliper still checks every input.")
                    attempt -= 1
                    continue
                retry = self._classify(exc, settings, conversation)
                if reached_pane[0]:
                    raise AgentError("agent.stream_interrupted",
                                     f"The reply broke off because the Claude API {retry.text}. "
                                     "Send the prompt again to retry.", retryable=True,
                                     data={"http_status": retry.http_status})
                if attempt >= MAX_ATTEMPTS:
                    waited = time.monotonic() - started
                    raise AgentError("agent.retries_exhausted",
                                     f"The Claude API {retry.text}; Caliper tried {attempt} times over "
                                     f"{waited:.0f} seconds. Try again later.", retryable=True,
                                     data={"http_status": retry.http_status, "attempts": attempt})
                delay = retry.delay if retry.delay is not None else self._backoff(retry.kind, attempt)
                delay = min(delay, MAX_RETRY_AFTER_S)
                sink.status("retry",
                            f"The Claude API {retry.text}. Retrying in {delay:.0f} s "
                            f"(attempt {attempt + 1} of {MAX_ATTEMPTS}).",
                            reason=retry.kind, http_status=retry.http_status, attempt=attempt + 1,
                            max_attempts=MAX_ATTEMPTS, delay_s=round(delay, 3))
                await asyncio.sleep(delay)
        return self._finish(message, conversation, sink)

    async def _stream_once(self, conversation: ClaudeConversation, settings: Settings, sink: EventSink,
                           reached_pane: list) -> Any:
        params = self.params(conversation, settings)
        async with self._client.beta.messages.stream(**params) as stream:
            async for event in stream:
                if self._translate(event, sink):
                    reached_pane[0] = True
            return await stream.get_final_message()

    @staticmethod
    def _translate(event: Any, sink: EventSink) -> bool:
        """Sends what the pane should see; True when something was sent."""
        kind = getattr(event, "type", None)
        if kind == "content_block_start":
            block = event.content_block
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                sink.tool_started(block.id, block.name)
                return True
            if btype == "text" and getattr(block, "text", ""):
                sink.message(block.text)
                return True
            if btype == "fallback":
                trigger = getattr(block, "trigger", None)
                sink.status("fallback",
                            f"{block.from_.model} declined this request; {block.to.model} continued it.",
                            from_model=block.from_.model, to_model=block.to.model,
                            category=getattr(trigger, "category", None))
                return True
            if btype == "compaction":
                sink.status("compaction", "The earlier conversation was summarized to make room.")
                return True
            return False
        if kind == "content_block_delta":
            delta = event.delta
            dtype = getattr(delta, "type", None)
            if dtype == "text_delta" and delta.text:
                sink.message(delta.text)
                return True
            if dtype == "thinking_delta" and delta.thinking:
                sink.thought(delta.thinking)
                return True
        return False

    def _finish(self, message: Any, conversation: ClaudeConversation, sink: EventSink) -> RequestResult:
        usage = message.usage
        counts = {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        }
        iterations = getattr(usage, "iterations", None) or []
        served_by_fallback = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
        stop = _STOPS.get(message.stop_reason or "end_turn", "end_turn")
        result = RequestResult(stop=stop, usage=counts, model=message.model, served_by_fallback=served_by_fallback)

        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            result.refusal_category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            result.refusal_explanation = scrub(explanation)[:MAX_ERROR_TEXT] if isinstance(explanation, str) else None
            return result  # a declined reply is not added to the history

        content = echo_content([b.to_dict(mode="json") for b in message.content])
        if content:
            conversation.add_assistant(content)
        result.tool_calls = [ToolCall(id=b["id"], name=b["name"], input=b.get("input"))
                             for b in content if b.get("type") == "tool_use"]
        return result


def tool_input_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
