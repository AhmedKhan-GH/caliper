"""A scripted model for tests, on Pydantic AI's `FunctionModel`.

When the environment variable CALIPER_AGENT_TEST_SCRIPT names a JSON file, the
sidecar answers its model requests from that script instead of a model server.
Nothing leaves the process. The switch can only make the sidecar talk to a
file; the key store removes the variable at spawn.

The script:

    {"steps": [<step>, ...],
     "record": "<optional path of a JSON Lines file of the requests seen>",
     "backoff_scale": <optional factor for retry delays, 0 in tests>,
     "route": "function" | "anthropic"}

With `route` "function" (the default), each step is turned into a streamed
reply of a `FunctionModel` (`ScriptedModel` below): Pydantic AI's own test model,
so the turn loop, the tools, approvals, cancellation, retries and stop reasons
are exercised without any wire format. With `route` "anthropic" the steps are
answered as Claude API events by stub_transport.py under the real Anthropic
model, for the tests of what is sent on the wire.

A step is one of:

    {"message": {"content": [<block>, ...], "stop_reason": "end_turn",
                 "usage": {...}, "stop_details": {...}, "model": "..."},
     "hang_after": <optional number of chunks after which the stream stalls>}
    {"error": {"status": 429, "headers": {...}, "body": {...}}}
    {"connect_error": "<message>"}
    {"stream_error": {"text": "<text shown before the error>", "body": {...}}}

A block is a text, thinking or tool_use block in the Claude API's response
shape; a tool_use block may give "raw_input", the exact JSON text to stream
instead of its "input". The function route records, for each request, the
model, the settings, the tools and the messages as Pydantic AI serializes them.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.exceptions import (ModelConnectionError, ModelHTTPError, ModelOverloadedError,
                                    ModelRateLimitError, ModelRequestRejectedError, ModelServerError,
                                    ModelUnavailableError)
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, DeltaToolCall, FunctionModel, FunctionStreamedResponse
from pydantic_ai.usage import RequestUsage

from .claude import DEFAULT_MODEL, MODELS, claude_settings
from .keyguard import scrub
from .provider import Capabilities, Settings

SCRIPT_ENV = "CALIPER_AGENT_TEST_SCRIPT"
MAX_SCRIPT_BYTES = 4 << 20

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_call", "max_tokens": "length",
           "model_context_window_exceeded": "length", "refusal": "content_filter", "pause_turn": "stop"}

_CATEGORY = {429: ModelRateLimitError, 529: ModelOverloadedError, 503: ModelUnavailableError}


def load_script(path: str) -> dict:
    p = Path(path)
    if p.stat().st_size > MAX_SCRIPT_BYTES:
        raise ValueError("The test script is larger than 4 MiB.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
        raise ValueError("The test script has no list of steps.")
    return doc


def http_error(status: int, model: str, body: Any, headers: dict | None = None, *, in_stream: bool = False):
    """The Pydantic AI error a provider adapter raises for an HTTP status."""
    if status in _CATEGORY:
        category = _CATEGORY[status]
    elif status >= 500:
        category = ModelServerError
    elif status in (400, 401, 403, 404, 413, 422):
        category = None if status != 400 else ModelRequestRejectedError
    else:
        category = None
    err = body.get("error") if isinstance(body, dict) else None
    return ModelHTTPError.for_category(category, status_code=status, model_name=model, body=body,
                                       headers=headers, in_stream=in_stream,
                                       provider_error_type=err.get("type") if isinstance(err, dict) else None)


@dataclass
class _ScriptedStream(FunctionStreamedResponse):
    """A FunctionModel stream that ends with the step's stop reason and usage."""

    _step: dict = field(default_factory=dict)

    async def _get_event_iterator(self):
        async for event in super()._get_event_iterator():
            yield event
        spec = self._step
        stop = spec.get("stop_reason", "end_turn")
        self.finish_reason = _FINISH.get(stop, "stop")  # type: ignore[assignment]
        details: dict[str, Any] = {"finish_reason": stop}
        stop_details = spec.get("stop_details") or {}
        if isinstance(stop_details, dict):
            if stop_details.get("explanation"):
                details["refusal"] = stop_details["explanation"]
            if stop_details.get("category"):
                details["refusal_category"] = stop_details["category"]
        self.provider_details = details
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, **spec.get("usage", {})}
        self._usage = RequestUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                                   cache_write_tokens=usage["cache_creation_input_tokens"],
                                   cache_read_tokens=usage["cache_read_input_tokens"],
                                   details={k: int(v) for k, v in usage.items() if isinstance(v, int)})


def _chunks(text: str, size: int = 24) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


class ScriptState:
    """The steps left and the record of requests, shared by the models of one sidecar."""

    def __init__(self, script: dict) -> None:
        self.steps = list(script.get("steps", []))
        self.record = script.get("record")
        self.backoff_scale = float(script.get("backoff_scale", 1.0))
        self.capabilities = dict(script.get("capabilities", {}))
        self._lock = threading.Lock()

    def next_step(self) -> dict:
        with self._lock:
            if not self.steps:
                return {"error": {"status": 500, "body": {"type": "error", "error": {
                    "type": "api_error", "message": "The test script has no more steps."}}}}
            return self.steps.pop(0)

    def log(self, entry: dict) -> None:
        if not self.record:
            return
        with self._lock, open(self.record, "a", encoding="utf-8") as f:
            f.write(scrub(json.dumps(entry, sort_keys=True, default=str)) + "\n")


class ScriptedModel(FunctionModel):
    """A FunctionModel whose replies come from the script's steps."""

    def __init__(self, state: ScriptState, model_name: str) -> None:
        super().__init__(stream_function=self._unused, model_name=model_name)
        self._state = state

    async def _unused(self, messages: list, info: AgentInfo) -> AsyncIterator[Any]:  # pragma: no cover
        yield ""

    async def _stream(self, step: dict) -> AsyncIterator[Any]:
        spec = step.get("message", {})
        hang_after = step.get("hang_after")
        if "stream_error" in step:
            failure = step["stream_error"]
            text = failure.get("text", "")
            if text:
                yield text
            body = failure.get("body", {"type": "error", "error": {"type": "overloaded_error",
                                                                   "message": "Overloaded"}})
            raise http_error(529, self.model_name, body, in_stream=True)
        count = 0
        for index, block in enumerate(spec.get("content", [])):
            kind = block.get("type")
            pieces: list[Any] = []
            if kind == "text":
                pieces = [p for p in _chunks(block.get("text", "")) if p]
            elif kind == "thinking":
                pieces = [{index: DeltaThinkingPart(content=p)} for p in _chunks(block.get("thinking", ""))]
                pieces.append({index: DeltaThinkingPart(signature=block.get("signature", "sig"))})
            elif kind == "tool_use":
                raw = block["raw_input"] if "raw_input" in block else json.dumps(block.get("input", {}))
                pieces = [{index: DeltaToolCall(name=block["name"], tool_call_id=block["id"])}]
                pieces += [{index: DeltaToolCall(json_args=p)} for p in _chunks(raw, 16)]
            for piece in pieces:
                if hang_after is not None and count >= hang_after:
                    await asyncio.sleep(3600)
                yield piece
                count += 1
                await asyncio.sleep(0)
        if count == 0:
            yield ""

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        step = self._state.next_step()
        self._state.log({
            "route": "function",
            "model": self.model_name,
            "settings": model_settings,
            "tools": [{"name": t.name, "strict": t.strict} for t in model_request_parameters.function_tools],
            "messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json"),
        })
        if "connect_error" in step:
            raise ModelConnectionError(self.model_name, str(step["connect_error"]), phase="connect")
        if "error" in step:
            err = step["error"]
            raise http_error(int(err.get("status", 500)), self.model_name,
                             err.get("body", {"type": "error", "error": {"type": "api_error", "message": "stub"}}),
                             err.get("headers"))
        spec = step.get("message", {})
        stream = _ScriptedStream(model_request_parameters=model_request_parameters,
                                 _model_name=spec.get("model", self.model_name),
                                 _iter=self._stream(step), _step=spec)
        yield stream


class ScriptedProvider:
    """The Claude settings and model list, with replies from the script."""

    name = "Claude API"
    models = MODELS
    default_model = DEFAULT_MODEL

    def __init__(self, script: dict, *, compaction: bool = True) -> None:
        self._state = ScriptState(script)
        self._models: dict[str, ScriptedModel] = {}
        self._compaction = compaction
        self.backoff_scale = self._state.backoff_scale
        self.extra: dict[str, Any] = {}

    def model(self, model_id: str) -> ScriptedModel:
        if model_id not in self._models:
            self._models[model_id] = ScriptedModel(self._state, model_id)
        return self._models[model_id]

    def model_settings(self, settings: Settings, *, strict: bool, role: str = "general") -> dict:
        return dict(claude_settings(settings.model, settings.effort, compaction=self._compaction))

    def capabilities(self, model_id: str) -> Capabilities:
        caps = self._state.capabilities
        return Capabilities(tools=caps.get("tools", True), vision=caps.get("vision", True),
                            context_length=caps.get("context_length"),
                            sources={k: "script" for k in ("tools", "vision", "context_length") if k in caps})

    def lower(self, model_id: str, *, tools: bool | None = None, vision: bool | None = None) -> None:
        if tools is False:
            self._state.capabilities["tools"] = False
        if vision is False:
            self._state.capabilities["vision"] = False

    async def aclose(self) -> None:
        pass
