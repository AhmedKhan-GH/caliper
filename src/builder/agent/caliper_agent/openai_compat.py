"""The OpenAI-compatible provider: any model server that speaks OpenAI's chat API.

Requests go through Pydantic AI's `OpenAIChatModel` with an
`OpenAICompatibleProvider` around an `AsyncOpenAI` client that the sidecar
builds itself: the base URL comes from `--base-url`, the key (if the server
needs one) from `CALIPER_MODEL_API_KEY`, and the SDK's own retries are off.
`OPENAI_API_KEY` and the other `OPENAI_*` variables are removed at start, so an
OpenAI key never reaches another server.

Servers differ in what they support, so the provider keeps what it knows about
the model in `Capabilities` and lowers it when the server says no:

- **Tool calling.** A server that refuses a request with tools (HTTP 400 that
  names tools) marks the model as without tools; the session then says that
  the model cannot drive Caliper, instead of calling it. `--model-tools no`
  says so up front.
- **Vision.** A server that refuses an image (HTTP 400 that names images or
  vision) marks the model as without vision, and the request is sent again
  with each image replaced by a line of text, which Pydantic AI writes when the
  model's profile has `supports_image_input` off. `--model-vision no` says so up
  front.
- **Lenient servers** that ignore `strict`, return malformed arguments or
  never make parallel calls need nothing here: every input is checked against
  the command's full schema before it runs, a malformed one becomes an error
  result the model can correct, and calls run one at a time anyway. Arguments
  that arrive whole or in pieces are assembled by Pydantic AI before the call
  is checked, and reasoning in a `reasoning` or `reasoning_content` field or in
  `<think>` tags becomes thoughts.
- **What the probe learned** (probe.py, capabilities.py) arrives at start:
  whether the model calls tools and reads images, its context length, and two
  quirks the provider follows: a server that refuses `strict` gets tools
  without it, and one that refuses `max_completion_tokens` gets `max_tokens`.
  When the context length is known, a request asks for at most half of it as
  output, so that a small model's context is not taken by the output budget.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from .keyguard import Secret
from .offline import check_provider
from .provider import Capabilities, ModelInfo, Settings

KEY_ENV = "CALIPER_MODEL_API_KEY"
MAX_TOKENS = 16384
PROVIDER_NAME = "caliper-openai-compatible"


class BaseUrlError(ValueError):
    pass


MAX_BASE_URL = 2048


def check_base_url(url: str) -> str:
    """The base URL as hostile input: an http or https address of at most 2,048
    printable characters, with a host, and without a user name, a password, a
    query or a fragment, which could carry a credential into the cache or a log."""
    if not isinstance(url, str) or not url or len(url) > MAX_BASE_URL or any(
            ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        raise BaseUrlError("--base-url must be an http or https address of at most 2,048 characters, "
                           "without spaces.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise BaseUrlError("--base-url is not a valid address.") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BaseUrlError("--base-url must be an http or https address, such as http://127.0.0.1:11434/v1.")
    if parts.username or parts.password:
        raise BaseUrlError("--base-url must not carry a user name or password; pass a key in "
                           f"{KEY_ENV} instead.")
    if parts.query or parts.fragment or url.endswith(("?", "#")):
        raise BaseUrlError("--base-url must not carry a query or a fragment.")
    del port
    return url.rstrip("/")


class OpenAICompatProvider:
    """One model on an OpenAI-compatible server."""

    def __init__(self, base_url: str, model_id: str, key: Secret | None, *, tools: bool = True,
                 vision: bool = True, transport: Any = None, backoff_scale: float = 1.0,
                 max_tokens: int = MAX_TOKENS, context_length: int | None = None,
                 sources: dict | None = None, quirks: dict | None = None) -> None:
        self.base_url = check_base_url(base_url)
        # With CALIPER_AGENT_OFFLINE set, only a server on loopback (or a test's transport).
        check_provider("openai-compatible", base_url=self.base_url, stubbed=transport is not None)
        host = urlsplit(self.base_url).hostname or "the model server"
        self.name = f"The model server at {host}"
        self.models = (ModelInfo(model_id, model_id, f"{model_id} on {host}"),)
        self.default_model = model_id
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=key.reveal() if key is not None else "not-needed",
            max_retries=0,
            timeout=httpx2.Timeout(600.0, connect=10.0),
            http_client=httpx2.AsyncClient(transport=transport) if transport is not None else None,
        )
        self._caps = {model_id: Capabilities(tools=tools, vision=vision, context_length=context_length,
                                             sources=dict(sources or {}))}
        self._quirks = dict(quirks or {})
        self._models: dict[str, OpenAIChatModel] = {}
        self._max_tokens = max_tokens
        self.backoff_scale = max(0.0, backoff_scale)
        self.extra: dict[str, Any] = {}

    def capabilities(self, model_id: str) -> Capabilities:
        return self._caps.setdefault(model_id, Capabilities())

    def lower(self, model_id: str, *, tools: bool | None = None, vision: bool | None = None) -> None:
        caps = self.capabilities(model_id)
        if tools is False:
            caps.tools = False
            caps.sources["tools"] = "observed"
        if vision is False:
            caps.vision = False
            caps.sources["vision"] = "observed"
        self._models.pop(model_id, None)

    def model(self, model_id: str) -> OpenAIChatModel:
        if model_id not in self._models:
            caps = self.capabilities(model_id)
            profile = OpenAIModelProfile(supports_image_input=caps.vision, supports_tools=caps.tools)
            if self._quirks.get("strict") is False:
                profile["openai_supports_strict_tool_definition"] = False
            if self._quirks.get("max_tokens_field") == "max_tokens":
                profile["openai_chat_supports_max_completion_tokens"] = False
            provider = OpenAICompatibleProvider(PROVIDER_NAME, openai_client=self._client, profile=profile)
            self._models[model_id] = OpenAIChatModel(model_id, provider=provider)
        return self._models[model_id]

    def output_budget(self, model_id: str) -> int:
        """The most output tokens a request asks for: at most half a known context length."""
        caps = self.capabilities(model_id)
        if caps.context_length and caps.sources.get("context_length") not in (None, "default", "unknown"):
            return max(256, min(self._max_tokens, caps.context_length // 2))
        return self._max_tokens

    def model_settings(self, settings: Settings, *, strict: bool) -> dict:
        return {"max_tokens": self.output_budget(settings.model)}

    async def aclose(self) -> None:
        await self._client.close()


def refusal_kind(message: str) -> str | None:
    """What a 400 from an OpenAI-compatible server refuses: "tools", "vision" or None."""
    text = message.lower()
    if "tool" in text or "function call" in text or "function_call" in text:
        return "tools"
    if "image" in text or "vision" in text or "multimodal" in text:
        return "vision"
    return None
