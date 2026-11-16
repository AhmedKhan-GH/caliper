"""The ACP agent: initialize, sessions, prompts, cancellation and config options.

Methods served (ACP version 1, agent side):

- `initialize`: answers version 1 whatever the client offers first, declares
  no optional capability, and lists no login method (the key comes from the app).
- `session/new`: a session with its own conversation, model, effort and review
  setting, offered to the pane as config options (`model`, `effort`, `review`).
  The `mcpServers` of the request are not started: this agent reaches Caliper
  through Caliper API itself.
- `session/set_config_option`: changes one of the three; the new value applies
  from the next model request.
- `session/prompt`: runs one turn and answers with its stop reason, and with the
  turn's token usage under `_meta.caliper.usage`. Without a key the turn is
  refused with error -32000 and a message the pane shows.
- `session/cancel` (notification): stops the running turn, which then answers
  with the stop reason `cancelled`.

Everything else is answered with method-not-found.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable

from . import __version__
from .acp import AUTH_REQUIRED, INVALID_PARAMS, METHOD_NOT_FOUND, Connection, RpcError
from .caliper_api import CaliperApi
from .keyguard import KeyStatus
from .loop import REVIEW_DESTRUCTIVE_ONLY, REVIEW_SPEC_CHANGES, SessionState, TurnRunner
from .provider import AgentError, Provider, Settings
from .tools import ToolSpec

log = logging.getLogger("caliper_agent.agent")

PROTOCOL_VERSION = 1
MAX_SESSIONS = 16
MAX_PROMPT_CHARS = 200_000

EFFORTS = ("low", "medium", "high", "xhigh", "max")

_EFFORT_NAMES = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Maximum"}

_NO_KEY_MESSAGE = ("No Claude API key is set. Add your key in the agent pane's settings to use the "
                   "Claude API backend.")
_BAD_KEY_MESSAGE = ("The Claude API key that Caliper passed is not well formed. Enter the key again "
                    "in the agent pane's settings.")


class Session:
    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.task: asyncio.Task | None = None
        self.cancel_requested = False


class Agent:
    def __init__(self, conn: Connection, provider_factory: Callable[[], Provider] | None, key_status: str,
                 tools: list[ToolSpec], api: CaliperApi, *, model_ids: tuple[str, ...], model_names: dict,
                 default_model: str, default_effort: str, usage_log: str | None = None) -> None:
        self._usage_log = usage_log
        self._conn = conn
        self._provider_factory = provider_factory
        self._provider: Provider | None = None
        self._key_status = key_status
        self._tools = {t.name: t for t in tools}
        self._api = api
        self._model_ids = model_ids
        self._model_names = model_names
        self._default_model = default_model
        self._default_effort = default_effort
        self._sessions: dict[str, Session] = {}

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
        if method == "authenticate":
            raise RpcError(INVALID_PARAMS, "This agent has no login methods; the app passes the key.")
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def on_notification(self, method: str, params: Any) -> None:
        if method == "session/cancel" and isinstance(params, dict):
            session = self._sessions.get(params.get("sessionId"))
            if session is not None and session.task is not None and not session.task.done():
                session.cancel_requested = True
                session.task.cancel()

    # ---- methods -------------------------------------------------------------------

    def _initialize(self, params: Any) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False},
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "authMethods": [],
            "agentInfo": {"name": "caliper-agent", "title": "Caliper agent (Claude API)", "version": __version__},
        }

    def _config_options(self, s: SessionState) -> list[dict]:
        return [
            {"id": "model", "name": "Model", "description": "The Claude model this session uses.",
             "category": "model", "type": "select", "currentValue": s.settings.model,
             "options": [{"value": m, "name": self._model_names.get(m, m)} for m in self._model_ids]},
            {"id": "effort", "name": "Effort",
             "description": "How much the model thinks before it acts. Higher is slower and costs more.",
             "category": "thought_level", "type": "select", "currentValue": s.settings.effort,
             "options": [{"value": e, "name": _EFFORT_NAMES[e]} for e in EFFORTS]},
            {"id": "review", "name": "Review spec changes",
             "description": "Ask before each change to the open spec, or only before deleting or overwriting saved data.",
             "type": "select", "currentValue": s.review,
             "options": [{"value": REVIEW_SPEC_CHANGES, "name": "Ask before each spec change"},
                         {"value": REVIEW_DESTRUCTIVE_ONLY, "name": "Ask only before deleting or overwriting"}]},
        ]

    def _new_session(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/new needs params.")
        cwd = params.get("cwd")
        if not isinstance(cwd, str) or not cwd or len(cwd) > 4096 or not os.path.isabs(cwd) or "\x00" in cwd:
            raise RpcError(INVALID_PARAMS, "session/new needs an absolute cwd.")
        if len(self._sessions) >= MAX_SESSIONS:
            raise RpcError(INVALID_PARAMS, f"This agent keeps at most {MAX_SESSIONS} sessions.")
        provider = self._ensure_provider()
        sid = "byok-" + uuid.uuid4().hex[:16]
        conversation = provider.new_conversation() if provider is not None else None
        state = SessionState(id=sid, cwd=cwd.rstrip("/") or "/", conversation=conversation,
                             settings=Settings(model=self._default_model, effort=self._default_effort))
        self._sessions[sid] = Session(state)
        return {"sessionId": sid, "configOptions": self._config_options(state)}

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
        elif config_id == "effort" and value in EFFORTS:
            s.settings.effort = value
        elif config_id == "review" and value in (REVIEW_SPEC_CHANGES, REVIEW_DESTRUCTIVE_ONLY):
            s.review = value
        else:
            raise RpcError(INVALID_PARAMS, f"Unknown config option or value: {config_id!s:.64}.")
        return {"configOptions": self._config_options(s)}

    def _ensure_provider(self) -> Provider | None:
        if self._provider is None and self._provider_factory is not None:
            self._provider = self._provider_factory()
        return self._provider

    @staticmethod
    def _prompt_text(blocks: Any) -> str:
        if not isinstance(blocks, list) or not blocks:
            raise RpcError(INVALID_PARAMS, "session/prompt needs a non-empty prompt.")
        parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif kind == "resource_link" and isinstance(block.get("uri"), str):
                name = block.get("name") if isinstance(block.get("name"), str) else block["uri"]
                parts.append(f"[Linked resource: {name} ({block['uri']})]")
        text = "\n\n".join(p for p in parts if p)
        if not text.strip():
            raise RpcError(INVALID_PARAMS, "The prompt holds no text.")
        if len(text) > MAX_PROMPT_CHARS:
            raise RpcError(INVALID_PARAMS, f"The prompt is longer than {MAX_PROMPT_CHARS} characters.")
        return text

    async def _prompt(self, params: Any) -> dict:
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "session/prompt needs params.")
        session = self._sessions.get(params.get("sessionId"))
        if session is None:
            raise RpcError(INVALID_PARAMS, "Unknown session.")
        if session.task is not None and not session.task.done():
            raise RpcError(INVALID_PARAMS, "A turn is already running in this session.")
        text = self._prompt_text(params.get("prompt"))
        if self._key_status != KeyStatus.PRESENT:
            message = _NO_KEY_MESSAGE if self._key_status == KeyStatus.MISSING else _BAD_KEY_MESSAGE
            raise RpcError(AUTH_REQUIRED, message, {"code": "agent.no_key" if self._key_status == KeyStatus.MISSING
                                                    else "agent.bad_key"})
        provider = self._ensure_provider()
        assert provider is not None
        session.task = asyncio.current_task()
        session.cancel_requested = False
        runner = TurnRunner(self._conn, provider, self._api, self._tools, session.state, self._usage_log)
        try:
            stop = await runner.run(text)
        except AgentError as exc:
            data = {"code": exc.code, "retryable": exc.retryable, **exc.data}
            raise RpcError(exc.rpc_code, exc.message, data) from None
        finally:
            session.task = None
        if session.cancel_requested:
            stop = "cancelled"
        return {"stopReason": stop, "_meta": {"caliper": {"usage": runner.usage}}}
