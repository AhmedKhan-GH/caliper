"""The Claude API provider: every Claude-specific detail of the sidecar lives here.

Requests go through Pydantic AI's Anthropic model, which uses the official
`anthropic` SDK (2.x) underneath and always streams. Most of the request is set
through Pydantic AI's documented `AnthropicModelSettings`:

- `anthropic_thinking`: adaptive thinking with summarized display, so the pane
  can show the model's reasoning as thoughts.
- `anthropic_effort`: `output_config.effort`, set explicitly on every request.
- `anthropic_eager_input_streaming`: every tool streams its input as written.
- `anthropic_cache_instructions` and `anthropic_cache`: a breakpoint on the
  system block (which caches the tools and the system prompt, rendered first)
  and the top-level `cache_control` that caches the conversation.
- `anthropic_context_management`: server-side compaction (`compact_20260112`);
  Pydantic AI adds its beta itself.
- `max_tokens` 64000; Pydantic AI streams the request.

Strict tools are Pydantic AI's `ToolDefinition.strict`, chosen by tools.py.

Server-side refusal fallbacks are not part of Pydantic AI's settings, so the
request carries them in `extra_body` (`fallbacks: "default"`) with the beta in
`anthropic_betas`. Pydantic AI drops the `fallback` block from a streamed reply
(its `AnthropicStreamedResponse` has no case for it) and has no part that maps
back to one, so after a mid-reply fallback it would echo the turn without the
boundary the API needs and would run the declining model's tool calls. The
workaround is `CaliperAnthropicModel` below: it keeps each `fallback` block as an
empty text part that carries the block in `provider_details`, maps that part
back to the block when the history is sent, and `echo_after_fallback()` (run by
the loop's `after_model_request` hook) leaves out what the API says to leave out.
It overrides two private methods of Pydantic AI 2.68, `_map_content_block_start`
of the streamed response and `_map_text_part` of the model, and a test pins
both. Claude Haiku 5.5 has no server-side fallback, so its requests carry none.

The client gets the key as an explicit argument, so the SDK never looks for
another credential, the base URL is fixed to the Claude API's, and the SDK's own
retries are off (`max_retries=0`): the loop retries through Pydantic AI's
`on_model_request_error` hook, where it can tell the pane what it waits for.
"""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.anthropic._streaming import AnthropicStreamedResponse
from pydantic_ai.providers.anthropic import AnthropicProvider

from .keyguard import Secret
from .provider import Capabilities, ModelInfo, Settings

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
COMPACTION = {"edits": [{"type": "compact_20260112"}]}

# The key under which a kept `fallback` block rides in a text part's provider_details.
FALLBACK_KEY = "caliper_fallback_block"


def model_ids() -> tuple[str, ...]:
    return tuple(m.id for m in MODELS)


def claude_settings(model_id: str, effort: str, *, compaction: bool = True) -> AnthropicModelSettings:
    """The Pydantic AI settings of one request to `model_id`."""
    settings = AnthropicModelSettings(
        max_tokens=MAX_TOKENS,
        anthropic_thinking={"type": "adaptive", "display": "summarized"},
        anthropic_effort=effort,  # type: ignore[typeddict-item]
        anthropic_eager_input_streaming=True,
        anthropic_cache_instructions=True,
        anthropic_cache=True,
    )
    if model_id not in NO_SERVER_FALLBACK:
        settings["extra_body"] = {"fallbacks": "default"}
        settings["anthropic_betas"] = [BETA_FALLBACK]
    if compaction:
        settings["anthropic_context_management"] = COMPACTION  # type: ignore[typeddict-item]
    return settings


def fallback_block(part: Any) -> dict | None:
    """The `fallback` block a text part stands for, or None."""
    if isinstance(part, TextPart) and not part.content and isinstance(part.provider_details, dict):
        block = part.provider_details.get(FALLBACK_KEY)
        if isinstance(block, dict):
            return block
    return None


def echo_after_fallback(response: ModelResponse) -> ModelResponse:
    """The response as it may be sent back after a mid-reply fallback.

    Before the last `fallback` block, only text and the fallback blocks
    themselves are kept: the declining model's thinking and tool calls are left
    out, as the API requires, and so are never run. The sidecar sends no server
    tools, so there are no paired server-tool blocks to keep.
    """
    last = -1
    for i, part in enumerate(response.parts):
        if fallback_block(part) is not None:
            last = i
    if last < 0:
        return response
    kept = [part for i, part in enumerate(response.parts)
            if i >= last or (isinstance(part, TextPart) and (part.content or fallback_block(part) is not None))]
    if len(kept) == len(response.parts):
        return response
    response.parts = kept
    return response


def declined_tool_calls(before: ModelResponse, after: ModelResponse) -> list[str]:
    """The ids of the tool calls `echo_after_fallback` left out."""
    kept = {p.tool_call_id for p in after.parts if isinstance(p, ToolCallPart)}
    return [p.tool_call_id for p in before.parts if isinstance(p, ToolCallPart) and p.tool_call_id not in kept]


class _FallbackKeepingStream(AnthropicStreamedResponse):
    """Keeps a `fallback` block as an empty text part instead of dropping it."""

    def _map_content_block_start(self, event):  # type: ignore[override]
        block = event.content_block
        if getattr(block, "type", None) == "fallback":
            self._current_block = block
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=TextPart(content="", provider_name=self.provider_name,
                              provider_details={FALLBACK_KEY: block.to_dict(mode="json")}))
            return
        yield from super()._map_content_block_start(event)


class CaliperAnthropicModel(AnthropicModel):
    """Pydantic AI's Anthropic model, plus the round trip of `fallback` blocks."""

    async def _process_streamed_response(self, *args: Any, **kwargs: Any) -> AnthropicStreamedResponse:
        streamed = await super()._process_streamed_response(*args, **kwargs)
        streamed.__class__ = _FallbackKeepingStream
        return streamed

    def _map_text_part(self, message, part, assistant_content_params) -> None:  # type: ignore[override]
        block = fallback_block(part)
        if block is not None:
            assistant_content_params.append(block)  # back where it appeared
            return
        super()._map_text_part(message, part, assistant_content_params)


class ClaudeProvider:
    """The Claude API through Pydantic AI, with the user's key."""

    name = "Claude API"
    models = MODELS
    default_model = DEFAULT_MODEL

    def __init__(self, key: Secret, *, transport: Any = None, compaction: bool = True,
                 backoff_scale: float = 1.0) -> None:
        http_client = DefaultAsyncHttpxClient(transport=transport) if transport is not None else None
        self._client = AsyncAnthropic(
            api_key=key.reveal(),
            base_url=API_BASE_URL,
            max_retries=0,
            timeout=anthropic.Timeout(600.0, connect=10.0),
            open_telemetry=False,
            http_client=http_client,
        )
        self._provider = AnthropicProvider(anthropic_client=self._client)
        self._models: dict[str, CaliperAnthropicModel] = {}
        self._compaction = compaction
        self.backoff_scale = max(0.0, backoff_scale)
        self.extra: dict[str, Any] = {}

    def model(self, model_id: str) -> CaliperAnthropicModel:
        if model_id not in self._models:
            self._models[model_id] = CaliperAnthropicModel(model_id, provider=self._provider)
        return self._models[model_id]

    def model_settings(self, settings: Settings, *, strict: bool) -> dict:
        return dict(claude_settings(settings.model, settings.effort, compaction=self._compaction))

    def capabilities(self, model_id: str) -> Capabilities:
        return Capabilities(tools=True, vision=True)

    def lower(self, model_id: str, *, tools: bool | None = None, vision: bool | None = None) -> None:
        pass  # every model in the picker calls tools and reads images

    async def aclose(self) -> None:
        await self._client.close()
