"""The sidecar's tools, derived from Caliper's command descriptor table.

The table is the one source of tool definitions (contracts section 12). The
commands module keeps its JSON form in src/builder/commands/command_table.json,
and this module reads it generically: it knows the descriptor's fields and the
naming rule, never a particular command's schema.

- A tool is named after its command with `_` for the first `.`: `panel_create`
  for `panel.create`. A group has no underscore, so the first underscore of a
  tool name marks the dot and the mapping is reversible (the rule `caliper-mcp`
  uses).
- A tool's input schema is the command's `args` schema without the keywords a
  strict Claude API tool does not take (`minimum`, `maximum`, `minLength`,
  `maxLength`, `minItems`, `maxItems`, `pattern`, `default` and their kin). The
  full schema is kept for the check the sidecar runs before each call.
- A tool is strict when every object in its schema is closed
  (`additionalProperties: false`) and lists `required`, and has no `$ref`.
  Otherwise it is sent without `strict`, so one open schema cannot make the
  whole request fail.

The file is hostile input as far as the loader is concerned: its size, the
number of commands, names, nesting and types are all checked.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_TABLE_BYTES = 8 << 20
MAX_COMMANDS = 512
MAX_SUMMARY = 8192
MAX_SCHEMA_DEPTH = 32

_NAME = re.compile(r"^[a-z][a-z0-9]*\.[a-z][a-z0-9_]*$")

# Keywords that strict tools do not take. The command layer still checks them.
STRICT_UNSUPPORTED = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "pattern", "default",
})

# The command groups whose mutating commands edit the open spec in place.
# spec.new, spec.open and applet.open replace the open spec rather than edit it.
SPEC_EDIT_GROUPS = frozenset({"source", "panel", "layout"})
SPEC_EDIT_COMMANDS = frozenset({"spec.set_text"})


class TableError(Exception):
    """The command table could not be read or is malformed."""


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    args: dict
    result: dict
    mutates: bool
    destructive: bool
    frame_thread: bool

    @property
    def group(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def verb(self) -> str:
        return self.name.split(".", 1)[1]

    @property
    def edits_spec(self) -> bool:
        """The command changes the open spec in place (and can be undone)."""
        return self.mutates and (self.group in SPEC_EDIT_GROUPS or self.name in SPEC_EDIT_COMMANDS)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its name, the command behind it and the schemas."""

    name: str
    command: Command
    input_schema: dict = field(repr=False)   # stripped, as sent to the model
    strict: bool = True

    @property
    def description(self) -> str:
        text = self.command.summary.strip()
        if self.command.destructive:
            text += " This command deletes or overwrites saved data, so Caliper asks the user before it runs."
        return text


def tool_name(command_name: str) -> str:
    return command_name.replace(".", "_", 1)


def command_name(name: str) -> str:
    return name.replace("_", ".", 1)


def _schema_depth_ok(schema: Any, depth: int = 0) -> bool:
    if depth > MAX_SCHEMA_DEPTH:
        return False
    if isinstance(schema, dict):
        return all(_schema_depth_ok(v, depth + 1) for v in schema.values())
    if isinstance(schema, list):
        return all(_schema_depth_ok(v, depth + 1) for v in schema)
    return True


def strip_for_strict(schema: Any) -> Any:
    """A copy of `schema` without the keywords strict tools do not take.

    Only schema positions are touched: the names inside `properties` are kept
    even when a property happens to be called `pattern` or `default`.
    """
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    out: dict = {}
    for key, value in schema.items():
        if key in STRICT_UNSUPPORTED:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: strip_for_strict(sub) for name, sub in value.items()}
        elif key in ("items", "additionalProperties", "not") and isinstance(value, dict):
            out[key] = strip_for_strict(value)
        elif key in ("anyOf", "allOf", "oneOf", "prefixItems") and isinstance(value, list):
            out[key] = [strip_for_strict(v) for v in value]
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: strip_for_strict(sub) for name, sub in value.items()}
        else:
            out[key] = copy.deepcopy(value)
    return out


def is_strict_compatible(schema: Any) -> bool:
    """Every object is closed and lists `required`, and nothing is referenced."""
    if not isinstance(schema, dict):
        return True
    if "$ref" in schema:
        return False
    types = schema.get("type")
    is_object = types == "object" or (isinstance(types, list) and "object" in types) or "properties" in schema
    if is_object:
        if schema.get("additionalProperties", None) is not False:
            return False
        if not isinstance(schema.get("required"), list):
            return False
    for value in (schema.get("properties") or {}).values() if isinstance(schema.get("properties"), dict) else ():
        if not is_strict_compatible(value):
            return False
    for key in ("items", "additionalProperties", "not"):
        if isinstance(schema.get(key), dict) and not is_strict_compatible(schema[key]):
            return False
    for key in ("anyOf", "allOf", "oneOf"):
        if isinstance(schema.get(key), list) and not all(is_strict_compatible(v) for v in schema[key]):
            return False
    return True


def _command_from_json(item: Any, index: int) -> Command:
    if not isinstance(item, dict):
        raise TableError(f"commands[{index}] is not an object.")
    name = item.get("name")
    if not isinstance(name, str) or len(name) > 64 or not _NAME.match(name):
        raise TableError(f"commands[{index}] has a bad name.")
    summary = item.get("summary", "")
    if not isinstance(summary, str) or len(summary) > MAX_SUMMARY:
        raise TableError(f"{name}: the summary is missing or too long.")
    args = item.get("args")
    if not isinstance(args, dict) or args.get("type") != "object":
        raise TableError(f"{name}: args is not an object schema.")
    result = item.get("result", {})
    if not isinstance(result, dict):
        raise TableError(f"{name}: result is not a schema.")
    if not _schema_depth_ok(args) or not _schema_depth_ok(result):
        raise TableError(f"{name}: a schema is nested too deeply.")
    flags = {}
    for flag in ("mutates", "destructive", "frame_thread"):
        value = item.get(flag, False)
        if not isinstance(value, bool):
            raise TableError(f"{name}: {flag} is not a boolean.")
        flags[flag] = value
    return Command(name=name, summary=summary, args=args, result=result, **flags)


def parse_command_table(data: bytes | str) -> list[Command]:
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) > MAX_TABLE_BYTES:
        raise TableError("The command table is larger than 8 MiB.")
    try:
        doc = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise TableError(f"The command table is not JSON: {exc}") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("commands"), list):
        raise TableError("The command table has no list of commands.")
    if doc.get("version", 0) != 0:
        raise TableError(f"The command table has version {doc.get('version')!r}; this sidecar reads version 0.")
    items = doc["commands"]
    if len(items) > MAX_COMMANDS:
        raise TableError(f"The command table has more than {MAX_COMMANDS} commands.")
    commands = [_command_from_json(item, i) for i, item in enumerate(items)]
    seen = set()
    for c in commands:
        if c.name in seen:
            raise TableError(f"The command {c.name} appears twice.")
        seen.add(c.name)
    return commands


def load_command_table(path: str | Path) -> list[Command]:
    p = Path(path)
    try:
        size = p.stat().st_size
        if size > MAX_TABLE_BYTES:
            raise TableError("The command table is larger than 8 MiB.")
        with p.open("rb") as f:
            data = f.read(MAX_TABLE_BYTES + 1)
    except OSError as exc:
        raise TableError(f"The command table cannot be read: {exc.strerror or exc}") from None
    return parse_command_table(data)


def derive_tools(commands: list[Command]) -> list[ToolSpec]:
    """One tool per command, in the table's order (stable, for prompt caching)."""
    tools = []
    for c in commands:
        stripped = strip_for_strict(c.args)
        tools.append(ToolSpec(name=tool_name(c.name), command=c, input_schema=stripped,
                              strict=is_strict_compatible(stripped)))
    return tools


def acp_tool_kind(command: Command) -> str:
    """The ACP tool kind the pane uses to pick an icon."""
    if not command.mutates:
        return "read"
    if command.verb in ("remove", "delete"):
        return "delete"
    if command.verb == "run":
        return "execute"
    return "edit"
