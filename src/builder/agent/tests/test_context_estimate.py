"""Generic request sizes and context estimates from synthetic inputs."""
from __future__ import annotations

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from caliper_agent.loop import RequestSize, estimate_prompt_tokens, request_size

PNG = b"\x89PNG\r\n\x1a\n" + bytes(64)


def _tools(chars: int) -> list[ToolDefinition]:
    """Tool definitions whose name, description and schema take `chars` characters, as the estimate counts them."""
    if chars == 0:
        return []
    schema = {"type": "object", "properties": {}}
    fixed = len("t") + len('{"type": "object", "properties": {}}')
    return [ToolDefinition(name="t", description="d" * (chars - fixed), parameters_json_schema=schema)]


def _messages(instructions: int, text: int, images: int, history: int = 1) -> list:
    """A request with the instructions on each of `history` requests, as Pydantic AI keeps them."""
    out: list = []
    for i in range(history - 1):
        out.append(ModelRequest(parts=[UserPromptPart(content="")], instructions="i" * instructions))
        out.append(ModelResponse(parts=[TextPart(content="")]))
    content: list = ["x" * text] + [BinaryContent(data=PNG, media_type="image/png")] * images
    out.append(ModelRequest(parts=[UserPromptPart(content=content)], instructions="i" * instructions))
    return out


def test_a_request_is_measured_by_kind_and_its_instructions_once():
    size = request_size(_messages(2213, 2476, 1, history=3), _tools(15438))
    assert size == RequestSize(instructions=2213, tools=15438, text=2476, images=1)
    calls = [ModelResponse(parts=[ToolCallPart(tool_name="panel_patch", args={"id": "curves"}, tool_call_id="c1")]),
             ModelRequest(parts=[ToolReturnPart(tool_name="panel_patch", content="ok", tool_call_id="c1")])]
    grown = request_size(_messages(2213, 2476, 1) + calls, _tools(15438))
    assert grown.text == 2476 + len("panel_patch") + len('{"id":"curves"}') + len("ok")
    assert grown.instructions == 2213


def test_a_history_that_shrank_is_estimated_whole():
    before = RequestSize(instructions=2213, tools=15438, text=9000, images=2)
    now = RequestSize(instructions=2213, tools=15438, text=2476, images=1)
    assert estimate_prompt_tokens(now, (9000, before)) == estimate_prompt_tokens(now)
