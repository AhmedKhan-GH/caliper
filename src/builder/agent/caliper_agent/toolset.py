"""Caliper's commands as a Pydantic AI toolset.

`CaliperToolset` is an `AbstractToolset` whose tools are generated at run time
from the command descriptor table (tools.py): each tool's definition is the
command's name, summary and stripped JSON schema, `strict` as `choose_strict()`
decided (and off for the whole session after the API refused strict mode), and
`sequential`, so the calls of one reply run one at a time, in order, as the
approval of spec edits needs.

Pydantic AI parses a call's arguments with the tool's `args_validator`; this
one only turns the JSON text into an object, and a text that is not a JSON
object becomes a failed result (`ToolFailed`, which does not use up a retry)
that shows the model what it sent. Everything else happens in `call_tool`,
which hands the call to the turn (loop.py): the check against the command's
full schema, the preview and the approval, the call to Caliper API and the
pane's updates.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from .tools import ToolSpec

# A call that fails only uses the tool's retry budget when the model asked for
# an unknown tool; everything else fails with ToolFailed, which uses none.
MAX_TOOL_RETRIES = 50


def invalid_json(raw: Any, problem: str) -> str:
    """The error result of an input that is not run: what the model sent and why."""
    sent = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=repr)
    return json.dumps({"INVALID_JSON": sent, "problem": problem}, ensure_ascii=False)


class JsonObjectArgs:
    """An args validator that accepts any JSON object and nothing else."""

    def validate_json(self, input: str | bytes | bytearray, *, allow_partial: Any = False, **kwargs: Any) -> Any:
        text = input.decode("utf-8", "replace") if isinstance(input, (bytes, bytearray)) else input
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            raise ToolFailed(invalid_json(text, "the input is not valid JSON")) from None
        return self.validate_python(value)

    def validate_python(self, input: Any, *, allow_partial: Any = False, **kwargs: Any) -> Any:
        if not isinstance(input, dict):
            raise ToolFailed(invalid_json(input, "the input is not a JSON object"))
        return input


class CaliperToolset(AbstractToolset[Any]):
    """The command table's tools. `ctx.deps` is the running turn (loop.TurnDeps)."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {t.name: t for t in tools}
        self._validator = JsonObjectArgs()

    @property
    def id(self) -> str:
        return "caliper"

    @property
    def specs(self) -> dict[str, ToolSpec]:
        return self._tools

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        strict_ok = getattr(ctx.deps, "strict", True)
        out: dict[str, ToolsetTool[Any]] = {}
        for name, spec in self._tools.items():
            definition = ToolDefinition(
                name=name,
                description=spec.description,
                parameters_json_schema=spec.input_schema,
                strict=bool(spec.send_strict and strict_ok),
                sequential=True,
                metadata={"command": spec.command.name},
            )
            out[name] = ToolsetTool(toolset=self, tool_def=definition, max_retries=MAX_TOOL_RETRIES,
                                    args_validator=self._validator)
        return out

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx, tool: ToolsetTool[Any]) -> Any:
        return await ctx.deps.runner.run_tool(self._tools[name], tool_args, ctx)


class HandoffToolset(AbstractToolset[Any]):
    """The widget session's one tool of its own, `handoff`: the model says that the
    request belongs to the side chat (widget.py), and the turn ends there. It is
    not a command, and its name has no underscore, so it can never be taken for one."""

    def __init__(self) -> None:
        from .widget import HANDOFF_DESCRIPTION, HANDOFF_SCHEMA, HANDOFF_TOOL

        self._name = HANDOFF_TOOL
        self._description = HANDOFF_DESCRIPTION
        self._schema = HANDOFF_SCHEMA
        self._validator = JsonObjectArgs()

    @property
    def id(self) -> str:
        return "caliper-handoff"

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        definition = ToolDefinition(name=self._name, description=self._description,
                                    parameters_json_schema=self._schema,
                                    strict=bool(getattr(ctx.deps, "strict", True)), sequential=True,
                                    metadata={"handoff": True})
        return {self._name: ToolsetTool(toolset=self, tool_def=definition, max_retries=MAX_TOOL_RETRIES,
                                        args_validator=self._validator)}

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx, tool: ToolsetTool[Any]) -> Any:
        return await ctx.deps.runner.handoff(tool_args, ctx)
