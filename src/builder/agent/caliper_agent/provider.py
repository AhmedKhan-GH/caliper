"""The shapes shared by the turn loop and the model providers.

The loop (loop.py) runs on Pydantic AI's agent graph; a provider says which
Pydantic AI model serves a session and with which settings. Two providers exist:
the Claude API through Pydantic AI's Anthropic model (claude.py) and any
OpenAI-compatible server, cloud or local, through Pydantic AI's OpenAI chat
model (openai_compat.py). Tests add a scripted provider (scripted.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    description: str


@dataclass
class Settings:
    model: str
    effort: str


@dataclass
class Capabilities:
    """What a model can do, as far as the sidecar knows. A provider may learn
    more from the server's answers and lower these for the rest of the sidecar."""

    tools: bool = True
    vision: bool = True


class AgentError(Exception):
    """A failure that ends the turn with an ACP error. The message is shown in
    the pane, so it is plain, scrubbed and never carries a secret."""

    def __init__(self, code: str, message: str, *, rpc_code: int = -32603, retryable: bool = False,
                 data: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.rpc_code = rpc_code
        self.retryable = retryable
        self.data = dict(data or {})


@dataclass
class Retry:
    """A failed model request that may be sent again."""

    kind: str                 # rate_limit, server, connection
    text: str                 # completes "The <service> ..."
    http_status: int | None = None
    delay: float | None = None   # what the server asked for, in seconds


class Provider(Protocol):
    """Where a session's model requests go."""

    name: str                        # "Claude API", shown in statuses and errors
    models: tuple[ModelInfo, ...]
    default_model: str
    backoff_scale: float
    extra: dict[str, Any]

    def model(self, model_id: str) -> Any: ...
    def model_settings(self, settings: Settings, *, strict: bool) -> dict: ...
    def capabilities(self, model_id: str) -> Capabilities: ...
    def lower(self, model_id: str, *, tools: bool | None = None, vision: bool | None = None) -> None: ...
    async def aclose(self) -> None: ...


@dataclass
class TurnUsage:
    """The token counts of one turn, as the pane and the usage log see them."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    per_request: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"requests": self.requests, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_creation_input_tokens": self.cache_creation_input_tokens,
                "cache_read_input_tokens": self.cache_read_input_tokens, "per_request": list(self.per_request)}
