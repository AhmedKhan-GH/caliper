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

Server-side refusal fallbacks are off (`SERVER_SIDE_FALLBACKS`). Pydantic AI
drops the `fallback` block from a streamed reply and has no part that maps back
to one, so after a fallback in the middle of a reply the next request would
lose the boundary the Claude API needs to check the turn, and the tool calls of
the model that declined would run. A refusal therefore ends the turn with ACP's
`refusal` stop reason (loop.py). The switch can be turned on once Pydantic AI
keeps the block; the request would then carry `fallbacks: "default"` in
`extra_body` and its beta in `anthropic_betas`, except on Claude Haiku 5.5,
which has no server-side fallback.

The client gets the key as an explicit argument, so the SDK never looks for
another credential, the base URL is fixed to the Claude API's, and the SDK's own
retries are off (`max_retries=0`): the loop retries through Pydantic AI's
`on_model_request_error` hook, where it can tell the pane what it waits for.
"""

from __future__ import annotations

from typing import Any, Callable

import anthropic
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from .keyguard import Secret
from .offline import check_provider, trust_env
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

# Server-side refusal fallbacks. Off: Pydantic AI drops the reply's `fallback`
# block, which the next request needs after a fallback in the middle of a
# reply. Turn this on once Pydantic AI keeps the block.
SERVER_SIDE_FALLBACKS = False
# Models without server-side refusal fallbacks: the parameter is left out.
NO_SERVER_FALLBACK = frozenset({"claude-haiku-5-5"})
BETA_FALLBACK = "server-side-fallback-2026-07-01"

MAX_TOKENS = 64000
COMPACTION = {"edits": [{"type": "compact_20260112"}]}


def known_capabilities(context_length: int | None = None, context_source: str = "unknown") -> Capabilities:
    """What every model of the picker can do, without a probe."""
    return Capabilities(tools=True, vision=True, context_length=context_length,
                        sources={"tools": "known", "vision": "known", "context_length": context_source})


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
    if SERVER_SIDE_FALLBACKS and model_id not in NO_SERVER_FALLBACK:
        settings["extra_body"] = {"fallbacks": "default"}
        settings["anthropic_betas"] = [BETA_FALLBACK]
    if compaction:
        settings["anthropic_context_management"] = COMPACTION  # type: ignore[typeddict-item]
    return settings


class ClaudeProvider:
    """The Claude API through Pydantic AI, with the user's key."""

    name = "Claude API"
    models = MODELS
    default_model = DEFAULT_MODEL

    def __init__(self, key: Secret, *, transport: Any = None, compaction: bool = True,
                 backoff_scale: float = 1.0, facts: Callable[[str], Capabilities] | None = None) -> None:
        # With CALIPER_AGENT_OFFLINE set, only the project's stub transport may stand in for the API.
        check_provider("claude", transport=transport)
        # Offline, the client ignores the environment's proxy settings (offline.trust_env).
        http_client = (DefaultAsyncHttpxClient(transport=transport, trust_env=trust_env())
                       if transport is not None or not trust_env() else None)
        self._client = AsyncAnthropic(
            api_key=key.reveal(),
            base_url=API_BASE_URL,
            max_retries=0,
            timeout=anthropic.Timeout(600.0, connect=10.0),
            open_telemetry=False,
            http_client=http_client,
        )
        self._provider = AnthropicProvider(anthropic_client=self._client)
        self._models: dict[str, AnthropicModel] = {}
        self._compaction = compaction
        self._facts = facts
        self.backoff_scale = max(0.0, backoff_scale)
        self.extra: dict[str, Any] = {}

    def model(self, model_id: str) -> AnthropicModel:
        if model_id not in self._models:
            self._models[model_id] = AnthropicModel(model_id, provider=self._provider)
        return self._models[model_id]

    def model_settings(self, settings: Settings, *, strict: bool, role: str = "general") -> dict:
        return dict(claude_settings(settings.model, settings.effort, compaction=self._compaction))

    def capabilities(self, model_id: str) -> Capabilities:
        """Every model in the picker calls tools and reads images; the context
        length is known once the probe has read the Models API (capabilities.py)."""
        if self._facts is not None:
            return self._facts(model_id)
        return known_capabilities()

    def lower(self, model_id: str, *, tools: bool | None = None, vision: bool | None = None) -> None:
        pass  # every model in the picker calls tools and reads images

    async def aclose(self) -> None:
        await self._client.close()
