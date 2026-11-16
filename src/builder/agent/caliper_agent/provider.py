"""The provider-neutral shapes between the turn loop and a model provider.

The loop (loop.py) runs tools, asks for approval and talks ACP; a provider runs
one model request at a time and owns its conversation format. The Claude API is
the only provider (claude.py). Other providers (OpenAI, Gemini, local models)
are a later, separate decision, each through its own official SDK and never
through a compatibility shim; each would be one more module with these shapes.
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
class ToolCall:
    id: str
    name: str
    input: Any


@dataclass
class ToolOutcome:
    """The answer to one tool call, as the model will see it."""

    call_id: str
    text: str
    is_error: bool = False
    png: bytes | None = None
    cancelled: bool = False


# How a request ended, in the loop's words:
#   end_turn      the model finished its reply
#   tool_use      the model asked for tools
#   max_tokens    the reply hit the output limit
#   context_full  the conversation no longer fits the model's context window
#   refusal       the model (and any fallback) declined
#   pause         the server paused a long turn; send the conversation again
STOPS = ("end_turn", "tool_use", "max_tokens", "context_full", "refusal", "pause")


@dataclass
class RequestResult:
    stop: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    refusal_category: str | None = None
    refusal_explanation: str | None = None
    served_by_fallback: bool = False


class EventSink(Protocol):
    """Where a provider sends what the pane should see while a request streams."""

    def message(self, text: str) -> None: ...
    def thought(self, text: str) -> None: ...
    def tool_started(self, call_id: str, name: str) -> None: ...
    def status(self, kind: str, text: str, **meta: Any) -> None: ...


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


class Conversation(Protocol):
    def add_user_prompt(self, text: str) -> None: ...
    def rollback_prompt(self) -> bool: ...
    def add_tool_results(self, outcomes: list[ToolOutcome]) -> None: ...


class Provider(Protocol):
    models: tuple[ModelInfo, ...]
    default_model: str

    def new_conversation(self) -> Conversation: ...
    async def request(self, conversation: Conversation, settings: Settings, sink: EventSink) -> RequestResult: ...
    async def aclose(self) -> None: ...
