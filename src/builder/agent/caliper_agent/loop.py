"""One prompt turn on Pydantic AI's agent graph: model requests, tool calls, approvals and their ACP updates.

The turn drives `agent.iter()` node by node. A `ModelRequestNode` is streamed,
and its events become ACP updates (text, thoughts, tool calls, statuses). The
`CallToolsNode` that follows holds the finished reply, and the turn decides
there, before any tool runs:

- `stop` or `tool_call`: the node runs; each tool call goes to
  `CaliperToolset`, which hands it back to `TurnRunner.run_tool()` (validated
  against the command's full schema, previewed, approved when it must be, then
  sent to Caliper API), one at a time and in order. What Caliper reviews itself
  (`review.status`: spec edits with review on, destructive commands) is not
  asked again through `session/request_permission`: Caliper holds the call
  until the person decides there, so each change gets one card. A reply without tool calls
  ends the run with ACP stop reason `end_turn`.
- `length` (`max_tokens` or a full context window): the node does not run, so
  no tool of a cut-off reply runs; each call gets an error result in the
  history, and the turn ends with `max_tokens`.
- `content_filter` (the model declined): the node does not run, the history
  goes back to what it was before the prompt, as ACP's `refusal` means, and the
  turn ends with `refusal`. The pane gets `REFUSAL_TEXT`: neutral, since a
  refusal can be a false positive, with three ways forward. Server-side
  fallbacks to another model are off (claude.py says why).

`CaliperHooks`, a Pydantic AI capability, sits in the model request: its
`on_model_request_error` hook sees every failed attempt that reached the pane
with nothing yet, tells the pane what it waits for and raises
`RetryModelRequest`, or ends the turn with an error; its
`prepare_model_request` hook sends the tools without `strict` once the API has
refused strict mode for the session.

Cancellation uses `AgentRun.cancel()`: Pydantic AI closes the stream or
cancels the running tool, and raises `RunCancelled` with everything that
finished. The turn then keeps that history, leaves out a reply the stream did
not finish, and gives every tool call that has no result one that says the
person cancelled it, so the next request is valid.

A widget session (widget.py) runs the same turn with its own agent (the widget
system prompt, its scoped tools and `handoff`) and three differences: every
call carries the session's preview in `X-Caliper-Preview` and every edit runs
without `dry_run`, so it stacks on the preview, with no permission request,
since the preview card in the app is the review; a call outside the scope is
refused even when the model names it; and the turn budget is small (3 model
requests, 8 tool calls, 8,000 output tokens). A `handoff` call ends the turn at
once and is reported in the prompt's answer for the app's "Ask in chat".
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
import random
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any

from pydantic_ai import Agent, BinaryContent, RetryModelRequest, UsageLimitExceeded, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import (ModelAPIError, ModelAuthenticationError, ModelConnectionError, ModelNotFoundError,
                                    ModelQuotaExceededError, ModelRateLimitError, ModelRequestRejectedError,
                                    ModelServerError, ModelTimeoutError, ModelUnavailableError, RunCancelled,
                                    ToolFailed, UnexpectedModelBehavior)
from pydantic_ai.messages import (CompactionPart, FunctionToolResultEvent, ModelMessage, ModelRequest, ModelResponse,
                                  PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart,
                                  ThinkingPartDelta, ToolCallPart, ToolReturnPart)

from .acp import Connection, PeerClosed, RpcError
from .caliper_api import CaliperApi, ToolOutput
from .keyguard import scrub
from .openai_compat import refusal_kind
from .provider import AgentError, Provider, Retry, Settings, TurnUsage
from .schema import validate
from .tools import PREVIEW_ARG, ToolSpec, acp_tool_kind
from .toolset import CaliperToolset, invalid_json
from .udiff import apply_unified_diff
from .widget import (HANDOFF_KINDS, MAX_HANDOFF_REASON, WIDGET_MAX_REQUESTS, WIDGET_MAX_TOKENS,
                     WIDGET_MAX_TOOL_CALLS, WidgetSpec)

log = logging.getLogger("caliper_agent.loop")

MAX_REQUESTS_PER_TURN = 50
MAX_PANE_TEXT = 4096
MAX_ATTEMPTS = 5
MAX_RETRY_AFTER_S = 60.0
MAX_BACKOFF_S = 30.0
MAX_ERROR_TEXT = 500

# What the pane shows when the model declines a request. A refusal can be a
# false positive (machine learning in the physical sciences touches chemistry,
# biology and security data), so it is neutral and offers three ways forward.
REFUSAL_TEXT = ("The model declined this request. You can rephrase it, try it again, or switch to another "
                "model or provider, such as a local model.")
REFUSAL_CHANGES_STAY = " Changes this turn already made in Caliper stay; you can undo them in the app."

REVIEW_SPEC_CHANGES = "spec_changes"
REVIEW_DESTRUCTIVE_ONLY = "destructive_only"


@dataclass
class SessionState:
    """What a turn needs from its session."""

    id: str
    cwd: str
    settings: Settings
    history: list[ModelMessage] = field(default_factory=list)
    review: str = REVIEW_SPEC_CHANGES
    turns: int = 0
    strict: bool = True
    # A prepared block of context that is sent with each prompt, ahead of it,
    # never in the system prompt. The widget agent's context bundle comes in the
    # prompt's own content blocks instead.
    context_block: str | None = None
    # The target, the scope and the preview of a widget session; None for the
    # general role.
    widget: WidgetSpec | None = None


@dataclass
class TurnDeps:
    """The `deps` of a Pydantic AI run: the turn that runs it."""

    runner: "TurnRunner"

    @property
    def strict(self) -> bool:
        return self.runner.session.strict


class _NoTools(Exception):
    """The model server cannot call tools."""


# ---- the agent ---------------------------------------------------------------------------------


@dataclass
class CaliperHooks(AbstractCapability[TurnDeps]):
    """The turn's hooks into each model request (see the module docstring)."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def prepare_model_request(self, ctx, request_context):
        if not ctx.deps.strict and any(t.strict for t in request_context.model_request_parameters.function_tools):
            params = request_context.model_request_parameters
            request_context.model_request_parameters = replace(
                params, function_tools=[replace(t, strict=False) for t in params.function_tools])
        return request_context

    async def on_model_request_error(self, ctx, *, request_context, error):
        return await ctx.deps.runner.on_request_error(request_context, error)


def build_agent(tools: list[ToolSpec], system_prompt: str, extra_toolsets: list | None = None) -> Agent:
    """One agent per sidecar for the general role, and one per tool scope for the
    widget role: the system prompt and the tools never change, so the cached
    prefix stays the same; the model and its settings come per run."""
    import pydantic_ai

    pydantic_ai.BANNER_ENABLED = False  # stderr is the app's log, not a terminal
    return Agent(
        output_type=str | None,  # a reply with only thinking ends the turn too
        instructions=system_prompt,
        deps_type=TurnDeps,
        toolsets=[CaliperToolset(tools), *(extra_toolsets or [])],
        capabilities=[CaliperHooks()],
        retries={"tools": 50, "output": 2},
        defer_model_check=True,
        name="caliper-agent",
    )


# ---- the pane ----------------------------------------------------------------------------------


class AcpSink:
    """Turns what happens into ACP session updates."""

    def __init__(self, conn: Connection, session_id: str) -> None:
        self._conn = conn
        self._sid = session_id
        self.started_calls: dict[str, str] = {}   # tool call id -> tool name, as shown in the pane
        self.open_calls: set[str] = set()         # shown and not yet completed or failed
        self.shown = False                        # whether this request has shown anything

    def update(self, update: dict) -> None:
        self._conn.notify("session/update", {"sessionId": self._sid, "update": update})

    def message(self, text: str) -> None:
        self.shown = True
        self.update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}})

    def thought(self, text: str) -> None:
        self.shown = True
        self.update({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}})

    def status(self, kind: str, text: str, **meta: Any) -> None:
        # A status reaches the pane as a thought that carries `_meta`, so any ACP
        # client shows it, and Caliper's pane can show it as a status line.
        self.update({"sessionUpdate": "agent_thought_chunk",
                     "content": {"type": "text", "text": scrub(text)},
                     "_meta": {"caliper": {"status": kind, **meta}}})

    def tool_started(self, call_id: str, name: str) -> None:
        if call_id in self.started_calls:
            return
        self.shown = True
        self.started_calls[call_id] = name
        self.open_calls.add(call_id)
        self.update({"sessionUpdate": "tool_call", "toolCallId": call_id, "title": name, "name": name,
                     "kind": "other", "status": "pending"})

    def finish(self, call_id: str, status: str, content: list[dict] | None = None,
               raw_output: Any = None) -> None:
        update: dict = {"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": status}
        if content:
            update["content"] = content
        if raw_output is not None:
            update["rawOutput"] = raw_output
        self.update(update)
        if status in ("completed", "failed"):
            self.open_calls.discard(call_id)

    def fail_open_calls(self, reason: str, keep: set[str] | None = None) -> None:
        for call_id in sorted(self.open_calls - (keep or set())):
            self.finish(call_id, "failed", [text_content(reason)])


def text_content(text: str) -> dict:
    return {"type": "content", "content": {"type": "text", "text": text}}


def _pane_text(text: str) -> str:
    return text if len(text) <= MAX_PANE_TEXT else text[:MAX_PANE_TEXT] + "\n[cut]"


def _describe_call(command: str, args: Any) -> str:
    try:
        pretty = json.dumps(args, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        pretty = repr(args)
    return _pane_text(f"{command}\n{pretty}")


def _title(command: str, args: Any) -> str:
    if isinstance(args, dict):
        for key in ("id", "name", "panel", "path"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return f"{command} {value[:64]}"
    return command


def _diff_path(cwd: str, relative: str) -> str:
    """An absolute path under the session's folder for a spec diff."""
    rel = PurePosixPath(relative.replace("\\", "/"))
    parts = [p for p in rel.parts if p not in ("", ".", "..", "/")]
    return posixpath.join(cwd, *parts) if parts else posixpath.join(cwd, "open.spec.toml")


# ---- the history --------------------------------------------------------------------------------


def _without_unfinished_reply(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The history without a last reply that the stream did not finish."""
    if messages and isinstance(messages[-1], ModelResponse) and messages[-1].state in ("interrupted", "incomplete"):
        return messages[:-1]
    return messages


def close_tool_calls(messages: list[ModelMessage], reason: str) -> list[ModelMessage]:
    """The history with an error result for every call of the last reply that has none."""
    out = list(messages)
    last = next((i for i in range(len(out) - 1, -1, -1) if isinstance(out[i], ModelResponse)), None)
    if last is None:
        return out
    calls = [p for p in out[last].parts if isinstance(p, ToolCallPart)]
    if not calls:
        return out
    following = out[last + 1] if last + 1 < len(out) and isinstance(out[last + 1], ModelRequest) else None
    answered = {getattr(p, "tool_call_id", None) for p in following.parts} if following is not None else set()
    missing = [ToolReturnPart(tool_name=c.tool_name, content=reason, tool_call_id=c.tool_call_id, outcome="failed")
               for c in calls if c.tool_call_id not in answered]
    if following is not None:
        out[last + 1] = replace(following, parts=[*following.parts, *missing], state="complete")
    elif missing:
        out.insert(last + 1, ModelRequest(parts=missing))
    return out


def usage_counts(response: ModelResponse) -> dict:
    """A reply's token counts in the Claude API's names, from Pydantic AI's usage."""
    usage = response.usage
    details = usage.details or {}
    return {
        "input_tokens": int(details.get("input_tokens", usage.input_tokens) or 0),
        "output_tokens": int(details.get("output_tokens", usage.output_tokens) or 0),
        "cache_creation_input_tokens": int(details.get("cache_creation_input_tokens", usage.cache_write_tokens) or 0),
        "cache_read_input_tokens": int(details.get("cache_read_input_tokens", usage.cache_read_tokens) or 0),
    }


# ---- errors --------------------------------------------------------------------------------------


def _error_message(error: BaseException) -> str:
    """The provider's own error message, scrubbed and bounded."""
    body = getattr(error, "body", None)
    message = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message = err["message"]
        elif isinstance(body.get("message"), str):
            message = body["message"]
    if message is None:
        message = getattr(error, "message", None) or str(error)
    return scrub(str(message)).strip()[:MAX_ERROR_TEXT]


def classify(error: BaseException, service: str, model_id: str) -> Retry:
    """A Retry for a failure worth another attempt; raises AgentError for one that is not.

    The order follows Pydantic AI's error categories: authentication, quota,
    the status errors that retrying cannot fix, then rate limits, unavailable
    and server errors, and connection errors.
    """
    status = getattr(error, "status_code", None)
    detail = _error_message(error)
    if isinstance(error, ModelAuthenticationError) or status == 401:
        raise AgentError("agent.key_rejected",
                         f"{service} did not accept the API key (HTTP 401). "
                         "Check or replace the key in the agent pane's settings.", rpc_code=-32000,
                         data={"http_status": 401})
    if isinstance(error, ModelQuotaExceededError) or status == 402:
        raise AgentError("agent.billing", f"{service} refused the request for billing reasons: {detail}",
                         data={"http_status": status})
    if status == 403:
        raise AgentError("agent.forbidden", f"{service} refused the request for this key (HTTP 403): {detail}",
                         data={"http_status": 403})
    if isinstance(error, ModelNotFoundError) or status == 404:
        raise AgentError("agent.model_unavailable",
                         f"The model {model_id} is not available (HTTP 404): {detail}", data={"http_status": 404})
    if status == 413:
        raise AgentError("agent.request_too_large",
                         "The conversation is too large for one request (HTTP 413). Start a new session.",
                         data={"http_status": 413})
    retry_after = getattr(error, "retry_after", None)
    if isinstance(error, ModelRateLimitError) or status == 429:
        return Retry("rate_limit", "is rate limited (HTTP 429)", 429, retry_after)
    if isinstance(error, (ModelUnavailableError, ModelServerError)) or status in (408, 409) or (
            isinstance(status, int) and status >= 500):
        what = f"answered HTTP {status}" if status else "reported an error"
        kind = getattr(error, "provider_error_type", None)
        if getattr(error, "in_stream", False) and kind:
            what = f"reported {kind} during the reply"
        return Retry("server", what, status, retry_after)
    if isinstance(error, ModelTimeoutError):
        return Retry("connection", "did not answer in time")
    if isinstance(error, ModelConnectionError):
        return Retry("connection", "could not be reached")
    if isinstance(error, ModelRequestRejectedError) or status == 400:
        raise AgentError("agent.bad_request", f"{service} refused the request (HTTP 400): {detail}",
                         data={"http_status": 400})
    if isinstance(error, ModelAPIError):
        raise AgentError("agent.api_error", f"{service} answered with an error: {detail}",
                         data={"http_status": status})
    raise AgentError("agent.internal", f"The request failed: {type(error).__name__}: {detail}")


def _backoff(kind: str, attempt: int, scale: float) -> float:
    base = {"connection": 1.0, "server": 2.0, "rate_limit": 5.0}.get(kind, 2.0)
    delay = min(MAX_BACKOFF_S, base * (2 ** (attempt - 1)))
    return delay * random.uniform(0.75, 1.25) * scale


# ---- the turn ------------------------------------------------------------------------------------


class TurnRunner:
    def __init__(self, conn: Connection, provider: Provider, agent: Agent, api: CaliperApi,
                 tools: dict[str, ToolSpec], session: SessionState, usage_log: str | None = None) -> None:
        self._conn = conn
        self._provider = provider
        self._agent = agent
        self._api = api
        self._tools = tools
        self.session = session
        self._usage_log = usage_log
        self._sink = AcpSink(conn, session.id)
        self.usage = TurnUsage()
        self._run = None
        self._task: asyncio.Task | None = None
        self.cancel_requested = False
        self._group = ""
        # What Caliper asks the person itself (review.status), read once per turn.
        self._app_review: tuple[bool, bool] | None = None
        # A widget session's outcome: the edits that stacked on its preview this
        # turn, and the handoff the model asked for.
        self.widget_edits = 0
        self.handoff_info: dict | None = None
        self.limit_reached = False

    # ---- cancellation ------------------------------------------------------------------------------

    def cancel(self) -> None:
        """Stops whatever runs: the stream, a tool call, a permission request or a retry delay."""
        self.cancel_requested = True
        if self._run is not None:
            self._run.cancel()
        elif self._task is not None and not self._task.done():
            self._task.cancel()

    # ---- the run -----------------------------------------------------------------------------------

    async def run(self, prompt: Any) -> str:
        """Runs the turn and returns its ACP stop reason. AgentError ends it with an error.

        `prompt` is the user prompt as Pydantic AI takes it: a string, or a list of
        strings and images (widget.prompt_content)."""
        s = self.session
        s.turns += 1
        self._group = f"{s.id}-t{s.turns}"
        self._app_review = None
        self._task = asyncio.current_task()
        model_id = s.settings.model
        caps = self._provider.capabilities(model_id)
        if not caps.tools:
            self._say_no_tools(model_id)
            return "end_turn"
        before = list(s.history)
        if s.context_block:
            user_prompt: Any = [s.context_block, *(prompt if isinstance(prompt, list) else [prompt])]
        else:
            user_prompt = prompt
        messages: list[ModelMessage] = before
        settings = self._provider.model_settings(s.settings, strict=s.strict)
        if s.widget is not None:
            limits = UsageLimits(request_limit=WIDGET_MAX_REQUESTS, tool_calls_limit=WIDGET_MAX_TOOL_CALLS)
            settings = {**settings, "max_tokens": min(int(settings.get("max_tokens") or WIDGET_MAX_TOKENS),
                                                      WIDGET_MAX_TOKENS)}
        else:
            limits = UsageLimits(request_limit=MAX_REQUESTS_PER_TURN)
        try:
            async with self._agent.iter(
                    user_prompt,
                    message_history=before,
                    model=self._provider.model(model_id),
                    model_settings=settings,
                    deps=TurnDeps(self),
                    usage_limits=limits,
            ) as run:
                self._run = run
                if self.cancel_requested:
                    run.cancel()
                async for node in run:
                    if Agent.is_model_request_node(node):
                        self._sink.shown = False
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                self._on_model_event(event)
                    elif Agent.is_call_tools_node(node):
                        stop = self._on_reply(node.model_response, model_id)
                        if stop == "refusal":
                            s.history = before  # the prompt and everything after it leave the history
                            return "refusal"
                        if stop == "max_tokens":
                            s.history = close_tool_calls(run.all_messages(),
                                                         "Not run: the reply was cut off at the output limit.")
                            return "max_tokens"
                        async with node.stream(run.ctx) as tool_stream:
                            async for event in tool_stream:
                                self._on_tool_event(event)
                        if self.handoff_info is not None:
                            break  # the request goes to the side chat; no further request
                messages = run.all_messages()
            s.history = messages
            return "end_turn"
        except RunCancelled as exc:
            s.history = close_tool_calls(_without_unfinished_reply(exc.all_messages()),
                                         "Cancelled by the user before it finished.")
            self._sink.fail_open_calls("Cancelled.")
            return "cancelled"
        except asyncio.CancelledError:
            if self._run is not None:
                s.history = close_tool_calls(_without_unfinished_reply(self._run.all_messages()),
                                             "Cancelled by the user before it finished.")
            self._sink.fail_open_calls("Cancelled.")
            return "cancelled"
        except UsageLimitExceeded:
            s.history = close_tool_calls(_without_unfinished_reply(self._run.all_messages()),
                                         "Not run: the turn reached its limit of model requests.")
            self._sink.fail_open_calls("Not run: the turn reached its limit.")
            self.limit_reached = True
            if s.widget is not None:
                self._sink.status("limit", f"The request stopped at the widget agent's limit of {WIDGET_MAX_REQUESTS} "
                                           f"model requests and {WIDGET_MAX_TOOL_CALLS} tool calls. Refine it, or ask "
                                           "in the side chat.", max_requests=WIDGET_MAX_REQUESTS,
                                  max_tool_calls=WIDGET_MAX_TOOL_CALLS)
            else:
                self._sink.status("limit", f"The turn stopped after {MAX_REQUESTS_PER_TURN} model requests.")
            return "max_turn_requests"
        except _NoTools:
            s.history = before
            self._sink.fail_open_calls("Not run.")
            self._say_no_tools(model_id)
            return "end_turn"
        except AgentError:
            self._keep_failed_turn()
            raise
        except ModelAPIError as exc:
            # A failure after the pane saw part of the reply: Pydantic AI does not
            # retry it, and neither does the turn, since a retried reply would
            # repeat what the pane already shows.
            self._keep_failed_turn()
            retry = classify(exc, self._provider.name, model_id)
            raise AgentError("agent.stream_interrupted",
                             f"The reply broke off because {self._provider.name} {retry.text}. "
                             "Send the prompt again to retry.", retryable=True,
                             data={"http_status": retry.http_status}) from None
        except UnexpectedModelBehavior as exc:
            self._keep_failed_turn()
            raise AgentError("agent.model_error", f"The model's reply could not be used: {scrub(str(exc))[:500]}")
        finally:
            self._run = None

    def _keep_failed_turn(self) -> None:
        if self._run is not None:
            self.session.history = close_tool_calls(_without_unfinished_reply(self._run.all_messages()),
                                                    "Not run: the turn failed.")
        self._sink.fail_open_calls("Not run: the turn failed.")

    def _say_no_tools(self, model_id: str) -> None:
        text = (f"The model {model_id} cannot call tools, so it cannot drive Caliper. "
                "Choose a model that supports tool calling.")
        self._sink.status("no_tools", text, model=model_id)
        self._sink.message(text)

    # ---- model events -----------------------------------------------------------------------------

    def _on_model_event(self, event: Any) -> None:
        sink = self._sink
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                if part.content:
                    sink.message(part.content)
            elif isinstance(part, ThinkingPart):
                if part.content:
                    sink.thought(part.content)
            elif isinstance(part, ToolCallPart):
                sink.tool_started(part.tool_call_id, part.tool_name)
            elif isinstance(part, CompactionPart):
                sink.status("compaction", "The earlier conversation was summarized to make room.")
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta) and delta.content_delta:
                sink.message(delta.content_delta)
            elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                sink.thought(delta.content_delta)

    def _on_reply(self, response: ModelResponse, model_id: str) -> str | None:
        """Accounts for a finished reply and says whether the turn stops before its tools."""
        counts = usage_counts(response)
        self._add_usage(counts, response.model_name or model_id)
        details = response.provider_details or {}
        if response.finish_reason == "content_filter":
            self._sink.fail_open_calls("Not run: the model declined the request.")
            text = REFUSAL_TEXT + (REFUSAL_CHANGES_STAY if self.usage.requests > 1 else "")
            # The provider's category and explanation travel in `_meta` for the
            # record; the text names neither.
            self._sink.status("refusal", text, category=details.get("refusal_category"),
                              explanation=details.get("refusal"), turn_removed=True)
            return "refusal"
        if response.finish_reason == "length":
            self._sink.fail_open_calls("Not run: the reply was cut off at the output limit.")
            return "max_tokens"
        return None

    def _add_usage(self, counts: dict, model: str) -> None:
        u = self.usage
        u.requests += 1
        u.input_tokens += counts["input_tokens"]
        u.output_tokens += counts["output_tokens"]
        u.cache_creation_input_tokens += counts["cache_creation_input_tokens"]
        u.cache_read_input_tokens += counts["cache_read_input_tokens"]
        u.per_request.append({**counts, "model": model})
        if self._usage_log:
            # Token counts only, for the manual check of prompt caching; no content.
            line = json.dumps({"session": self.session.id, "turn": self.session.turns, "model": model, **counts},
                              sort_keys=True)
            try:
                with open(self._usage_log, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                log.warning("the usage log cannot be written")

    def _on_tool_event(self, event: Any) -> None:
        if isinstance(event, FunctionToolResultEvent):
            part = event.part
            if part.tool_call_id in self._sink.open_calls:
                # A call that never reached the toolset: an unknown tool, or an
                # input that is not a JSON object.
                text = part.model_response_str(wrap_if_error=False)
                self._sink.finish(part.tool_call_id, "failed", [text_content(_pane_text(scrub(text)))])

    # ---- retries -----------------------------------------------------------------------------------

    async def on_request_error(self, request_context: Any, error: Exception) -> Any:
        """Pydantic AI's `on_model_request_error`: retry with a status, or end the turn."""
        s = self.session
        provider = self._provider
        model_id = s.settings.model
        if isinstance(error, AgentError):
            raise error
        status = getattr(error, "status_code", None)
        message = _error_message(error)
        rejected = isinstance(error, ModelRequestRejectedError) or status == 400
        if rejected and "strict" in message.lower() and s.strict and not any(
                isinstance(m, ModelResponse) for m in request_context.messages):
            # Only before the first reply: later it would change the tools under
            # thinking blocks that were made with them.
            s.strict = False
            self._sink.status("strict_off", f"{provider.name} refused the strict tool schemas; this session "
                                            "continues without strict mode, and Caliper still checks every input.")
            raise RetryModelRequest()
        if rejected and provider.name != "Claude API":
            kind = refusal_kind(message)
            if kind == "tools":
                provider.lower(model_id, tools=False)
                raise _NoTools()
            if kind == "vision" and provider.capabilities(model_id).vision:
                provider.lower(model_id, vision=False)
                self._sink.status("no_vision", f"The model {model_id} cannot read images; captures and rendered "
                                               "panels are described to it in text from now on.", model=model_id)
                raise RetryModelRequest(provider.model(model_id))
        retry = classify(error, provider.name, model_id)
        attempt = request_context.attempt
        if attempt >= MAX_ATTEMPTS:
            raise AgentError("agent.retries_exhausted",
                             f"{provider.name} {retry.text}; Caliper tried {attempt} times. Try again later.",
                             retryable=True, data={"http_status": retry.http_status, "attempts": attempt})
        delay = retry.delay if retry.delay is not None else _backoff(retry.kind, attempt, provider.backoff_scale)
        delay = min(max(0.0, delay), MAX_RETRY_AFTER_S)
        self._sink.status("retry", f"{provider.name} {retry.text}. Retrying in {delay:.0f} s "
                                   f"(attempt {attempt + 1} of {MAX_ATTEMPTS}).",
                          reason=retry.kind, http_status=retry.http_status, attempt=attempt + 1,
                          max_attempts=MAX_ATTEMPTS, delay_s=round(delay, 3))
        await asyncio.sleep(delay)
        raise RetryModelRequest()

    # ---- tools -------------------------------------------------------------------------------------

    async def run_tool(self, spec: ToolSpec, args: dict, ctx: Any) -> Any:
        """One tool call, from `CaliperToolset.call_tool`. Returns what the model
        reads; raises ToolFailed for an error result."""
        sink = self._sink
        call_id = ctx.tool_call_id or ""
        group = self._group
        command = spec.command
        sink.tool_started(call_id, spec.name)
        problem = validate(args, command.args)
        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call_id,
                     "title": _title(command.name, args), "kind": acp_tool_kind(command), "rawInput": args})
        if problem:
            # Treated like invalid JSON: not run, and the model sees what it sent.
            sink.finish(call_id, "failed", [text_content(f"Not run: {problem}")])
            raise ToolFailed(invalid_json(args, problem))
        if self.session.widget is not None:
            return await self._run_widget_tool(spec, args, call_id)

        previewable = command.previewable and args.get(PREVIEW_ARG) is not True
        # Caliper may hold the call for the person itself; what it asks, the
        # sidecar does not ask again, so one change gets one card.
        app_edits, app_destructive = (await self._app_reviews()) if previewable or command.destructive \
            else (False, False)
        own_review = previewable and self.session.review == REVIEW_SPEC_CHANGES and not app_edits
        before = await self._api.spec_text() if previewable else None

        preview_diff: str | None = None
        if own_review:
            preview = await self._api.call(command.name, {**args, PREVIEW_ARG: True}, group)
            if not preview.ok:
                # The change itself is refused; the real call would be too.
                sink.finish(call_id, "failed", [text_content(_pane_text(preview.text))],
                            raw_output={"error": preview.error_code})
                raise ToolFailed(scrub(preview.text))
            diff = preview.result.get("diff") if isinstance(preview.result, dict) else None
            preview_diff = diff if isinstance(diff, str) else None
        # A preview that changes nothing needs no approval.
        ask = (command.destructive and not app_destructive) or (own_review and preview_diff != "")
        if ask:
            decision = await self._ask(call_id, command, args, before, preview_diff)
            if decision == "cancelled":
                sink.finish(call_id, "failed", [text_content("Cancelled.")])
                self.cancel()
                raise ToolFailed("Cancelled by the user.")
            if decision == "rejected":
                sink.finish(call_id, "failed", [text_content("Denied by the user.")])
                raise ToolFailed(json.dumps({"error": {
                    "code": "agent.denied",
                    "message": "The user denied this call. Do not try it again in another form; "
                               "say what you wanted to do and ask how to go on."}}))

        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "in_progress"})
        out = await self._api.call(command.name, args, group)
        await self._report(call_id, out, before, previewable)
        # Workspace data could hold a secret; the model never sees one.
        text = scrub(out.text)
        if not out.ok:
            raise ToolFailed(text)
        if out.png is not None:
            return [text, BinaryContent(data=out.png, media_type="image/png")]
        return text

    async def _app_reviews(self) -> tuple[bool, bool]:
        """Whether Caliper itself holds this sidecar's spec edits and its destructive
        commands for the person, from `review.status`, read once per turn. An app
        without that command, or an answer without the fields, holds neither, and
        the sidecar asks as before."""
        if self._app_review is None:
            out = await self._api.call("review.status", {}, self._group)
            result = out.result if out.ok and isinstance(out.result, dict) else {}
            self._app_review = (result.get("spec_edits") is True, result.get("destructive") is True)
        return self._app_review

    async def _run_widget_tool(self, spec: ToolSpec, args: dict, call_id: str) -> Any:
        """One call of a widget session: inside the scope only, with the preview's
        header, and an edit without `dry_run`, so it stacks on the preview."""
        sink = self._sink
        widget = self.session.widget
        command = spec.command
        if command.name not in widget.scope:
            # The session's agent offers only the scope's tools; this guards the
            # call path itself, whatever the model names.
            sink.finish(call_id, "failed", [text_content("Not run: the command is outside this target's scope.")])
            raise ToolFailed(json.dumps({"error": {
                "code": "agent.out_of_scope",
                "message": f"{command.name} is not one of this widget session's tools. Use the handoff tool when "
                           "the request needs more than this target."}}))
        if self.handoff_info is not None:
            sink.finish(call_id, "failed", [text_content("Not run: the request was handed to the side chat.")])
            raise ToolFailed("Not run: the request was handed to the side chat.")
        args = {k: v for k, v in args.items() if k != PREVIEW_ARG}
        edit = command.previewable
        before = await self._api.spec_text(widget.preview) if edit else None
        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "in_progress"})
        out = await self._api.call(command.name, args, self._group, widget.preview)
        if out.ok and edit and isinstance(out.result, dict) and out.result.get("applied") is True \
                and out.result.get("diff"):
            self.widget_edits += 1
        await self._report(call_id, out, before, edit)
        text = scrub(out.text)
        if not out.ok:
            raise ToolFailed(text)
        if out.png is not None:
            return [text, BinaryContent(data=out.png, media_type="image/png")]
        return text

    async def handoff(self, args: dict, ctx: Any) -> str:
        """The `handoff` tool: records why the request belongs to the side chat; the
        turn ends after this reply's calls (`run`)."""
        call_id = ctx.tool_call_id or ""
        sink = self._sink
        sink.tool_started(call_id, "handoff")
        kind = args.get("kind")
        reason = args.get("reason")
        if kind not in HANDOFF_KINDS or not isinstance(reason, str) or not reason.strip() or set(args) - {
                "kind", "reason"}:
            problem = f"kind must be one of {', '.join(HANDOFF_KINDS)} and reason one sentence"
            sink.finish(call_id, "failed", [text_content(f"Not run: {problem}")])
            raise ToolFailed(invalid_json(args, problem))
        reason = scrub(" ".join(reason.split()))[:MAX_HANDOFF_REASON]
        self.handoff_info = {"kind": kind, "reason": reason}
        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "title": "Ask in chat",
                     "kind": "other", "rawInput": {"kind": kind, "reason": reason}})
        sink.finish(call_id, "completed", [text_content(reason)], raw_output=self.handoff_info)
        sink.status("handoff", f"This request belongs in the side chat: {reason}", category=kind, reason=reason)
        return "Handed to the side chat; the person can ask there. The turn ends now."

    def widget_outcome(self) -> dict | None:
        """The `_meta.caliper.widget` of the prompt's answer, for a widget session."""
        widget = self.session.widget
        if widget is None:
            return None
        return {"preview": widget.preview, "edits": self.widget_edits, "handoff": self.handoff_info,
                "limit": self.limit_reached}

    def _diff_content(self, before: tuple[str, str] | None, diff: str | None) -> dict | None:
        """The change as an ACP diff (old and new text), else as the unified diff in a text block."""
        if not isinstance(diff, str) or diff == "":
            return None
        if before is not None:
            after = apply_unified_diff(before[0], diff)
            if after is not None:
                return {"type": "diff", "path": _diff_path(self.session.cwd, before[1]), "oldText": before[0],
                        "newText": after}
        return text_content(_pane_text("```diff\n" + diff.rstrip("\n") + "\n```"))

    async def _ask(self, call_id: str, command, args: dict, before: tuple[str, str] | None,
                   preview_diff: str | None) -> str:
        """Sends session/request_permission; returns allowed, rejected or cancelled."""
        content = [text_content(_describe_call(command.name, args))]
        shown = self._diff_content(before, preview_diff)
        if shown is not None:
            content.insert(0, shown)
        if command.destructive:
            options = [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                       {"optionId": "reject", "name": "Deny", "kind": "reject_once"}]
        else:
            options = [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                       {"optionId": "allow-session", "name": "Allow spec changes for this session",
                        "kind": "allow_always"},
                       {"optionId": "reject", "name": "Deny", "kind": "reject_once"}]
        tool_call = {"toolCallId": call_id, "title": _title(command.name, args), "kind": acp_tool_kind(command),
                     "status": "pending", "rawInput": args, "content": content}
        try:
            answer = await self._conn.request("session/request_permission",
                                              {"sessionId": self.session.id, "toolCall": tool_call,
                                               "options": options})
        except (RpcError, PeerClosed):
            return "rejected"
        outcome = answer.get("outcome") if isinstance(answer, dict) else None
        if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
            return "cancelled"
        kinds = {o["optionId"]: o["kind"] for o in options}
        kind = kinds.get(outcome.get("optionId"))
        if kind == "allow_always":
            self.session.review = REVIEW_DESTRUCTIVE_ONLY
            return "allowed"
        if kind == "allow_once":
            return "allowed"
        return "rejected"

    async def _report(self, call_id: str, out: ToolOutput, before, previewable: bool) -> None:
        content: list[dict] = []
        if out.ok and previewable and before is not None:
            diff = out.result.get("diff") if isinstance(out.result, dict) else None
            shown = self._diff_content(before, diff) if isinstance(diff, str) else None
            if shown is None and not isinstance(diff, str):
                # An app whose commands return no diff: compare the spec before and after.
                widget = self.session.widget
                after = await self._api.spec_text(widget.preview if widget is not None else "")
                if after is not None and after[0] != before[0]:
                    shown = {"type": "diff", "path": _diff_path(self.session.cwd, after[1]),
                             "oldText": before[0], "newText": after[0]}
            if shown is not None:
                content.append(shown)
        content.append(text_content(_pane_text(out.text)))
        if out.png is not None:
            content.append({"type": "content", "content": {"type": "image", "mimeType": "image/png",
                                                           "data": base64.b64encode(out.png).decode("ascii")}})
        raw = out.result if out.ok else {"error": out.error_code}
        self._sink.finish(call_id, "completed" if out.ok else "failed", content, raw_output=raw)
