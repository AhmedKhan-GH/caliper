"""The ACP agent: initialize, sessions, prompts, cancellation and config options.

Methods served (ACP version 1, agent side):

- `initialize`: answers version 1 whatever the client offers first, declares
  images and embedded context in prompts and `session/close`, and lists no
  login method (the key comes from the app).
- `session/new`: a session with its own history, model, effort and review
  setting, offered to the pane as config options (`model`, `effort` for the
  Claude API, `review`). The answer's `_meta.caliper.model` says what the
  sidecar assumes of the model (tool calling, vision, the context length and
  where each came from), so the app can size the context bundle; a model
  known to have no tool calling refuses the session with `agent.no_tools`. The `mcpServers` of the request are not started: this
  agent reaches Caliper through Caliper API itself. With
  `_meta.caliper.widget` the session is a widget session (widget.py): one
  target, a tool scope and the app's preview, the widget system prompt, the
  fast default model at low effort, and no `review` option, since the preview
  card in the app is the review.
- `session/set_config_option`: changes one of them; the new value applies from
  the next model request.
- `session/prompt`: runs one turn and answers with its stop reason, and with the
  turn's token usage under `_meta.caliper.usage`. Without a key the turn is
  refused with error -32000 and a message the pane shows.
  A widget session's answer also carries `_meta.caliper.widget`: the preview,
  the edits that stacked on it, the handoff the model asked for, and whether
  the turn reached its limit.
- `session/cancel` (notification): stops the running turn, which then answers
  with the stop reason `cancelled`. A widget session's preview stays as it is.
- `session/close`: cancels a running turn, waits for it to end, and forgets the
  session. The preview belongs to the app, which drops or commits it.

Everything else is answered with method-not-found.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from pydantic_ai import Agent as GraphAgent

from . import __version__
from .acp import AUTH_REQUIRED, INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, Connection, RpcError
from .caliper_api import CaliperApi
from .keyguard import KeyStatus
from .loop import REVIEW_DESTRUCTIVE_ONLY, REVIEW_SPEC_CHANGES, SessionState, TurnRunner, build_agent
from .prompt import WIDGET_SYSTEM_PROMPT
from .provider import AgentError, Provider, Settings
from .tools import Command, ToolSpec
from .toolset import HandoffToolset
from .widget import ScopeRule, WidgetError, parse_widget_meta, prompt_content, widget_tools

log = logging.getLogger("caliper_agent.agent")

PROTOCOL_VERSION = 1
MAX_SESSIONS = 16
MAX_PROMPT_CHARS = 200_000

EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLOSE_WAIT_S = 10.0

# A widget session's defaults on the Claude API (design/widget-agent.md, The turn
# budget, the model and the effort): the fastest and cheapest model of the
# picker, at low effort, for a small edit with its context given in full.
WIDGET_CLAUDE_MODEL = "claude-haiku-5-5"
WIDGET_CLAUDE_EFFORT = "low"

_EFFORT_NAMES = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Maximum"}

_NO_KEY_MESSAGE = ("No Claude API key is set. Add your key in the agent pane's settings to use the "
                   "Claude API.")
_BAD_KEY_MESSAGE = ("The API key that Caliper passed is not well formed. Enter the key again "
                    "in the agent pane's settings.")


class Session:
    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.runner: TurnRunner | None = None
        self.idle = asyncio.Event()
        self.idle.set()
        self.closed = False


class Agent:
    def __init__(self, conn: Connection, provider: Provider | None, key_status: str, tools: list[ToolSpec],
                 graph: GraphAgent, api: CaliperApi, *, default_model: str, default_effort: str,
                 usage_log: str | None = None, key_required: bool = True, commands: list[Command] | None = None,
                 scope_rule: ScopeRule | None = None, refusal: Exception | None = None,
                 provider_meta: dict | None = None, widget_model: str | None = None,
                 widget_effort: str | None = None, widget_requests: int | None = None) -> None:
        self._widget_requests = widget_requests
        self._usage_log = usage_log
        # Why no provider may run (offline.py): every session/new and
        # session/prompt is then answered with this error.
        self._refusal = refusal
        # Where model requests go, with the warnings the pane shows
        # (presets.py), for initialize's `_meta.caliper.provider`.
        self._provider_meta = provider_meta
        # A widget session's model and effort on the Claude API when the
        # command line names them (the conformance suite does); else the
        # fast defaults below.
        self._widget_model = widget_model
        self._widget_effort = widget_effort
        self._conn = conn
        self._provider = provider
        self._key_status = key_status
        self._key_required = key_required
        self._tools = {t.name: t for t in tools}
        self._graph = graph
        self._api = api
        self._default_model = default_model
        self._default_effort = default_effort
        self._sessions: dict[str, Session] = {}
        # The widget role: the whole table (the general role's tools may be
        # filtered), the scope rule, and one agent per tool set, so that each
        # kind of target keeps one cached prefix.
        self._commands = commands or [t.command for t in tools]
        self._scope_rule = scope_rule
        self._widget_graphs: dict[tuple[str, ...], tuple[GraphAgent, dict[str, ToolSpec]]] = {}

    @property
    def _claude(self) -> bool:
        return self._provider is None or self._provider.name == "Claude API"

    @property
    def _model_ids(self) -> tuple[str, ...]:
        if self._provider is None:
            return (self._default_model,)
        return tuple(m.id for m in self._provider.models)

    async def aclose(self) -> None:
        if self._provider is not None:
            await self._provider.aclose()
        await self._api.aclose()

    # ---- dispatch ------------------------------------------------------------------

    async def on_request(self, method: str, params: Any) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "session/new":
            return self._new_session(params)
        if method == "session/prompt":
            return await self._prompt(params)
        if method == "session/set_config_option":
            return self._set_config_option(params)
        if method == "session/close":
            return await self._close_session(params)
        if method == "authenticate":
            raise RpcError(INVALID_PARAMS, "This agent has no login methods; the app passes the key.")
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def on_notification(self, method: str, params: Any) -> None:
        if method == "session/cancel" and isinstance(params, dict):
            session = self._sessions.get(params.get("sessionId"))
            if session is not None and session.runner is not None:
                session.runner.cancel()

    # ---- methods -------------------------------------------------------------------

    def _initialize(self, params: Any) -> dict:
        title = "Caliper agent (Claude API)" if self._claude else f"Caliper agent ({self._provider.name})"
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                "mcpCapabilities": {"http": False, "sse": False},
                "sessionCapabilities": {"close": {}},
            },
            "authMethods": [],
            "agentInfo": {"name": "caliper-agent", "title": title, "version": __version__},
            **({"_meta": {"caliper": {"provider": self._provider_meta}}} if self._provider_meta is not None else {}),
        }

    def _config_options(self, s: SessionState) -> list[dict]:
        names = {m.id: m.name for m in self._provider.models} if self._provider is not None else {}
        options = [
            {"id": "model", "name": "Model", "description": "The model this session uses.",
             "category": "model", "type": "select", "currentValue": s.settings.model,
             "options": [{"value": m, "name": names.get(m, m)} for m in self._model_ids]},
        ]
        if self._claude:
            options.append(
                {"id": "effort", "name": "Effort",
                 "description": "How much the model thinks before it acts. Higher is slower and costs more.",
                 "category": "thought_level", "type": "select", "currentValue": s.settings.effort,
                 "options": [{"value": e, "name": _EFFORT_NAMES[e]} for e in EFFORTS]})
        if s.widget is not None:
            return options  # the preview card in the app is the review
        options.append(
            {"id": "review", "name": "Review spec changes",
             "description": "Ask before each change to the open spec, or only before deleting or overwriting saved data.",
             "type": "select", "currentValue": s.review,
             "options": [{"value": REVIEW_SPEC_CHANGES, "name": "Ask before each spec change"},
                         {"value": REVIEW_DESTRUCTIVE_ONLY, "name": "Ask only before deleting or overwriting"}]})
        return options

    def _refuse_if_offline(self) -> None:
        if self._refusal is not None:
            raise RpcError(INTERNAL_ERROR, str(self._refusal),
                           {"code": getattr(self._refusal, "code", "agent.offline"), "retryable": False})

    def _new_session(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/new needs params.")
        self._refuse_if_offline()
        cwd = params.get("cwd")
        if not isinstance(cwd, str) or not cwd or len(cwd) > 4096 or not os.path.isabs(cwd) or "\x00" in cwd:
            raise RpcError(INVALID_PARAMS, "session/new needs an absolute cwd.")
        if len(self._sessions) >= MAX_SESSIONS:
            raise RpcError(INVALID_PARAMS, f"This agent keeps at most {MAX_SESSIONS} sessions.")
        try:
            widget = parse_widget_meta(params.get("_meta"), self._commands, self._scope_rule)
        except WidgetError as exc:
            raise RpcError(INVALID_PARAMS, str(exc), {"code": "agent.bad_widget"}) from None
        sid = ("widget-" if widget is not None else "byok-") + uuid.uuid4().hex[:16]
        model, effort = self._default_model, self._default_effort
        if widget is not None and self._claude:
            model = self._widget_model or WIDGET_CLAUDE_MODEL
            effort = self._widget_effort or WIDGET_CLAUDE_EFFORT
        caliper_meta: dict = {}
        if self._provider is not None:
            caps = self._provider.capabilities(model)
            if not caps.tools:
                # A model without tool calling cannot drive Caliper: no session
                # is made, and the reason is one the pane can show.
                raise RpcError(INVALID_PARAMS, f"The model {model} cannot call tools, so it cannot drive Caliper. "
                                               "Choose a model that supports tool calling.",
                               {"code": "agent.no_tools", "model": model, "capability": "tools",
                                "source": caps.sources.get("tools", "default"), "retryable": False})
            caliper_meta["model"] = caps.as_meta(model)
        state = SessionState(id=sid, cwd=cwd.rstrip("/") or "/", settings=Settings(model=model, effort=effort),
                             widget=widget)
        if self._widget_requests:
            state.widget_max_requests = self._widget_requests
        self._sessions[sid] = Session(state)
        result: dict = {"sessionId": sid, "configOptions": self._config_options(state)}
        if widget is not None:
            caliper_meta["widget"] = {"target": widget.target, "scope": list(widget.scope), "preview": widget.preview}
        if caliper_meta:
            result["_meta"] = {"caliper": caliper_meta}
        return result

    def _set_config_option(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/set_config_option needs params.")
        session = self._sessions.get(params.get("sessionId"))
        if session is None:
            raise RpcError(INVALID_PARAMS, "Unknown session.")
        config_id = params.get("configId")
        value = params.get("value")
        s = session.state
        if config_id == "model" and value in self._model_ids:
            s.settings.model = value
        elif config_id == "effort" and self._claude and value in EFFORTS:
            s.settings.effort = value
        elif config_id == "review" and s.widget is None and value in (REVIEW_SPEC_CHANGES, REVIEW_DESTRUCTIVE_ONLY):
            s.review = value
        else:
            raise RpcError(INVALID_PARAMS, f"Unknown config option or value: {config_id!s:.64}.")
        return {"configOptions": self._config_options(s)}

    def _vision(self, s: SessionState) -> bool:
        return self._provider is None or bool(self._provider.capabilities(s.settings.model).vision)

    def _turn_tools(self, s: SessionState) -> tuple[GraphAgent, dict[str, ToolSpec]]:
        """The agent and the tools of a session's next turn: the general role's, or
        the widget scope's for the model as it is now (with or without vision)."""
        if s.widget is None:
            return self._graph, self._tools
        assert self._scope_rule is not None
        tools = widget_tools(self._commands, s.widget.scope, vision=self._vision(s), rule=self._scope_rule)
        key = tuple(t.name for t in tools)
        if key not in self._widget_graphs:
            self._widget_graphs[key] = (build_agent(tools, WIDGET_SYSTEM_PROMPT, [HandoffToolset()]),
                                        {t.name: t for t in tools})
        return self._widget_graphs[key]

    async def _close_session(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/close needs params.")
        session = self._sessions.get(params.get("sessionId"))
        if session is None:
            raise RpcError(INVALID_PARAMS, "Unknown session.")
        session.closed = True
        if session.runner is not None:
            session.runner.cancel()
            try:
                await asyncio.wait_for(session.idle.wait(), timeout=CLOSE_WAIT_S)
            except asyncio.TimeoutError:
                log.warning("a turn did not end within %.0f s of session/close", CLOSE_WAIT_S)
        self._sessions.pop(session.state.id, None)
        return {}

    async def _prompt(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/prompt needs params.")
        session = self._sessions.get(params.get("sessionId"))
        if session is None or session.closed:
            raise RpcError(INVALID_PARAMS, "Unknown session.")
        if session.runner is not None:
            raise RpcError(INVALID_PARAMS, "A turn is already running in this session.")
        self._refuse_if_offline()
        try:
            content = prompt_content(params.get("prompt"), vision=self._vision(session.state),
                                     max_chars=MAX_PROMPT_CHARS)
        except WidgetError as exc:
            raise RpcError(INVALID_PARAMS, str(exc)) from None
        if self._key_status == KeyStatus.MALFORMED or (self._key_required and self._key_status != KeyStatus.PRESENT):
            message = _BAD_KEY_MESSAGE if self._key_status == KeyStatus.MALFORMED else _NO_KEY_MESSAGE
            raise RpcError(AUTH_REQUIRED, message, {"code": "agent.bad_key" if self._key_status == KeyStatus.MALFORMED
                                                    else "agent.no_key"})
        assert self._provider is not None
        graph, tools = self._turn_tools(session.state)
        runner = TurnRunner(self._conn, self._provider, graph, self._api, tools, session.state, self._usage_log)
        session.runner = runner
        session.idle.clear()
        try:
            stop = await runner.run(content.user_prompt())
        except AgentError as exc:
            data = {"code": exc.code, "retryable": exc.retryable, **exc.data}
            raise RpcError(exc.rpc_code, exc.message, data) from None
        finally:
            session.runner = None
            session.idle.set()
        if runner.cancel_requested:
            stop = "cancelled"
        meta: dict = {"usage": runner.usage.as_dict()}
        outcome = runner.widget_outcome()
        if outcome is not None:
            meta["widget"] = outcome
        return {"stopReason": stop, "_meta": {"caliper": meta}}
