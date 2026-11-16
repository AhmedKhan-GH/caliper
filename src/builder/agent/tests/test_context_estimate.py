"""The context check's estimate of a request's prompt tokens (loop.estimate_prompt_tokens), against real counts.

The figures below were recorded from real requests of `qwen3.8:27b` on Ollama 0.40.16 through the Ollama preset: a
loopback proxy in front of Ollama noted, for each `/api/chat` request, the characters of the instructions, of the
tool definitions and of the messages' text, the images, and the `prompt_eval_count` Ollama answered with. They are
⌘K's first requests on the four entry points of the walk-through, the requests after them in the same turns, and a
side chat session's first request with every command of the table as a tool. No content was kept; the requests are
rebuilt here from their sizes, since the estimate reads nothing else.

The estimate must never be below the server's count, or a request the check lets through could still be cut, and
should not be far above it, or a request that fits would be refused: within 30 percent here.
"""

from __future__ import annotations

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from caliper_agent.loop import RequestSize, estimate_prompt_tokens, request_size

MARGIN = 1.30
PNG = b"\x89PNG\r\n\x1a\n" + bytes(64)

# (what, instruction characters, text characters, tool characters, images, prompt_eval_count)
FIRST_REQUESTS = [
    ("⌘K on the curves", 2213, 2476, 15438, 1, 6464),
    ("⌘K on the heatmap", 2213, 1625, 15438, 1, 5814),
    ("⌘K on a selection", 2213, 3930, 13066, 1, 7046),
    ("⌘K on a property", 2213, 1980, 6229, 1, 3335),
    ("the side chat, 34 tools", 3136, 58, 35144, 0, 10773),
    ("the probe's tool call", 0, 69, 152, 0, 281),
]

# The turns' later requests: (what, the request before as above, this request's sizes, its count)
LATER_REQUESTS = [
    ("curves, second", (2213, 2476, 15438, 1, 6464), (2213, 3374, 15438, 1), 6779),
    ("curves, third, with a render", (2213, 3374, 15438, 1, 6779), (2525, 3684, 15438, 2), 7401),
    ("heatmap, second", (2213, 1625, 15438, 1, 5814), (2213, 2681, 15438, 1), 6181),
    ("selection, second", (2213, 3930, 13066, 1, 7046), (2213, 7899, 13066, 1), 8162),
    ("selection, third", (2213, 7899, 13066, 1, 8162), (2213, 10067, 13066, 1), 8796),
    ("property, second", (2213, 1980, 6229, 1, 3335), (2213, 2871, 6229, 1), 3659),
]


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


def test_the_first_request_is_estimated_at_or_above_the_servers_count():
    for what, instructions, text, tools, images, counted in FIRST_REQUESTS:
        estimate = estimate_prompt_tokens(request_size(_messages(instructions, text, images), _tools(tools)))
        assert counted <= estimate <= counted * MARGIN, (what, estimate, counted)


def test_a_later_request_is_estimated_from_the_last_count_at_or_above_the_servers():
    for what, (bi, bx, bt, bn, before_counted), (ni, nx, nt, nn), counted in LATER_REQUESTS:
        before = RequestSize(instructions=bi, tools=bt, text=bx, images=bn)
        now = RequestSize(instructions=ni, tools=nt, text=nx, images=nn)
        estimate = estimate_prompt_tokens(now, (before_counted, before))
        assert counted <= estimate <= counted * MARGIN, (what, estimate, counted)


def test_a_history_that_shrank_is_estimated_whole():
    before = RequestSize(instructions=2213, tools=15438, text=9000, images=2)
    now = RequestSize(instructions=2213, tools=15438, text=2476, images=1)
    assert estimate_prompt_tokens(now, (9000, before)) == estimate_prompt_tokens(now)
