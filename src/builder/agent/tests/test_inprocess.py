"""The turn in process, with Pydantic AI's own test models (TestModel, FunctionModel).

These drive `TurnRunner` directly, without the stdio protocol: a fake ACP
connection collects the updates and allows every permission request, and the
fake Caliper API of conftest answers the tools.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from conftest import COMMAND_TABLE, general_tool_count

from caliper_agent.caliper_api import CaliperApi
from caliper_agent.loop import SessionState, TurnRunner, build_agent
from caliper_agent.prompt import SYSTEM_PROMPT
from caliper_agent.provider import Capabilities, Settings
from caliper_agent.tools import derive_tools, load_command_table


class FakeConn:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.asked: list[dict] = []

    def notify(self, method: str, params: dict) -> None:
        self.updates.append(params["update"])

    async def request(self, method: str, params: dict) -> dict:
        self.asked.append(params)
        return {"outcome": {"outcome": "selected", "optionId": "allow"}}


class OneModel:
    """A provider that always answers with the given Pydantic AI model."""

    name = "Test"
    models = ()
    default_model = "test"
    backoff_scale = 0.0
    extra: dict = {}

    def __init__(self, model) -> None:
        self._model = model

    def model(self, model_id: str):
        return self._model

    def model_settings(self, settings: Settings, *, strict: bool, role: str = "general") -> dict:
        return {}

    def capabilities(self, model_id: str) -> Capabilities:
        return Capabilities()

    def lower(self, model_id: str, **kwargs) -> None:
        pass

    async def aclose(self) -> None:
        pass


def _turn(model, caliper, session: SessionState | None = None, tools=None):
    tools = tools if tools is not None else derive_tools(load_command_table(COMMAND_TABLE))
    conn = FakeConn()
    session = session or SessionState(id="s1", cwd="/tmp/ws", settings=Settings(model="test", effort="high"))
    api = CaliperApi(caliper.api_file)
    runner = TurnRunner(conn, OneModel(model), build_agent(tools, SYSTEM_PROMPT), api, {t.name: t for t in tools},
                        session)
    return runner, conn, api


def test_testmodel_calls_tools_generated_from_the_table(caliper):
    model = TestModel(call_tools=["catalog_list", "app_info", "spec_get"])
    runner, conn, api = _turn(model, caliper)

    async def go():
        try:
            return await runner.run("What is there?")
        finally:
            await api.aclose()

    assert asyncio.run(go()) == "end_turn"
    params = model.last_model_request_parameters
    tools = {t.name: t for t in params.function_tools}
    assert len(tools) == general_tool_count() and all(t.sequential for t in tools.values())
    assert 0 < sum(bool(t.strict) for t in tools.values()) <= 20
    assert [c["path"] for c in caliper.calls if c["path"] != "/v0/commands/spec.get"][:2] == [
        "/v0/commands/catalog.list", "/v0/info"]
    done = [u["toolCallId"] for u in conn.updates if u.get("status") == "completed"]
    assert len(done) == 3
    # Every call of the history has its result.
    calls = [p.tool_call_id for m in runner.session.history for p in getattr(m, "parts", [])
             if type(p).__name__ == "ToolCallPart"]
    results = [p.tool_call_id for m in runner.session.history for p in getattr(m, "parts", [])
               if type(p).__name__ == "ToolReturnPart"]
    assert sorted(calls) == sorted(results) and len(calls) == 3


def test_a_context_block_goes_with_the_prompt_and_never_into_the_system_prompt(caliper):
    seen: list = []

    async def stream(messages, info: AgentInfo):
        seen.append((messages, info.instructions))
        yield "Noted."

    session = SessionState(id="s2", cwd="/tmp/ws", settings=Settings(model="test", effort="high"),
                           context_block="Widget context: a heatmap of tensor:attn_weights.")
    runner, conn, api = _turn(FunctionModel(stream_function=stream), caliper, session)

    async def go():
        try:
            return await runner.run("Explain this widget.")
        finally:
            await api.aclose()

    assert asyncio.run(go()) == "end_turn"
    [(messages, instructions)] = seen
    prompt = messages[-1].parts[-1].content
    assert prompt == ["Widget context: a heatmap of tensor:attn_weights.", "Explain this widget."]
    assert instructions == SYSTEM_PROMPT.strip() or instructions.startswith(SYSTEM_PROMPT[:200])
    assert "Widget context" not in instructions
