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
- A type list (`"type": ["string", "null"]`) becomes an `anyOf` of one schema
  per type, with the keywords that belong to each, because the Claude SDK's
  `transform_schema()`, which Pydantic AI runs on every strict tool, accepts only
  a single type. The meaning is the same.
- A tool can be strict when every object in its schema is closed
  (`additionalProperties: false`) and lists `required`, and has no `$ref`.
  Otherwise it is sent without `strict`, so one open schema cannot make the
  whole request fail.
- The Claude API compiles all strict schemas of a request together and refuses
  a request over its schema complexity limits: 20 strict tools, 24 optional
  parameters and 16 parameters with union types in all (the limits Pydantic AI
  checks before it sends, in `pydantic_ai/models/_anthropic_schema_limits.py`).
  `choose_strict()` picks the strict tools within those limits once per
  sidecar, the commands that change something first, in table order; the
  others are sent without `strict` and checked the same way.
- A session may use a subset of the tools (`include`, a filter over the table),
  for roles that need fewer commands.

The file is hostile input as far as the loader is concerned: its size, the
number of commands, names, nesting and types are all checked.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

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

# The argument through which a command previews its change: with it set, the
# command checks the change and returns its diff without applying it.
PREVIEW_ARG = "dry_run"

# Command groups that are never offered to the model. The preview commands
# belong to the app and the sidecar: edits reach a stacked preview through the
# X-Caliper-Preview header, which the sidecar sets, and only the person
# commits one.
NOT_FOR_THE_MODEL = frozenset({"preview"})


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
    def previewable(self) -> bool:
        """The command changes something and can preview the change: its args
        schema has a boolean `dry_run`. Read from the schema, so the set of such
        commands follows the table (today, every edit of the open spec)."""
        props = self.args.get("properties")
        arg = props.get(PREVIEW_ARG) if isinstance(props, dict) else None
        if not self.mutates or not isinstance(arg, dict):
            return False
        types = arg.get("type")
        return types == "boolean" or (isinstance(types, list) and "boolean" in types)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its name, the command behind it and the schemas."""

    name: str
    command: Command
    input_schema: dict = field(repr=False)   # stripped, as sent to the model
    strict: bool = True                      # the schema allows strict mode
    send_strict: bool = False                # chosen to be sent strict (choose_strict)

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


_TYPE_KEYWORDS = {
    "object": ("properties", "required", "additionalProperties"),
    "array": ("items", "prefixItems"),
}


def _value_has_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name in ("integer", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    return True


def split_type_lists(schema: Any) -> Any:
    """A copy of `schema` in which every type list is an `anyOf` of single types."""
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    out: dict = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: split_type_lists(sub) for name, sub in value.items()}
        elif key in ("items", "additionalProperties", "not") and isinstance(value, dict):
            out[key] = split_type_lists(value)
        elif key in ("anyOf", "allOf", "oneOf", "prefixItems") and isinstance(value, list):
            out[key] = [split_type_lists(v) for v in value]
        else:
            out[key] = copy.deepcopy(value)
    types = out.get("type")
    if not isinstance(types, list) or len(types) < 2 or "anyOf" in out:
        if isinstance(types, list) and len(types) == 1:
            out["type"] = types[0]
        return out
    shared = {k: v for k, v in out.items()
              if k not in ("type", "enum", "description", "title") and not any(k in kws for kws in _TYPE_KEYWORDS.values())}
    branches = []
    for name in types:
        branch: dict = {"type": name}
        if name != "null":
            branch.update(shared)
            for kw in _TYPE_KEYWORDS.get(name, ()):
                if kw in out:
                    branch[kw] = out[kw]
            if isinstance(out.get("enum"), list):
                values = [v for v in out["enum"] if _value_has_type(v, name)]
                if not values:
                    continue
                branch["enum"] = values
        branches.append(branch)
    result: dict = {"anyOf": branches}
    for k in ("description", "title"):
        if k in out:
            result[k] = out[k]
    return result


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


MAX_STRICT_TOOLS = 20
MAX_STRICT_OPTIONAL = 24
MAX_STRICT_UNION = 16


def strict_counts(schema: Any, depth: int = 0) -> tuple[int, int]:
    """The optional parameters and the parameters with union types of a schema,
    counted over every object reached without a `$ref`, as the limits count them."""
    if not isinstance(schema, dict) or depth > MAX_SCHEMA_DEPTH:
        return 0, 0
    optional = union = 0
    props = schema.get("properties")
    if isinstance(props, dict):
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for name, prop in props.items():
            if name not in required:
                optional += 1
            if isinstance(prop, dict) and ("anyOf" in prop or isinstance(prop.get("type"), list)):
                union += 1
            o, u = strict_counts(prop, depth + 1)
            optional, union = optional + o, union + u
    for key in ("items", "additionalProperties", "not"):
        if isinstance(schema.get(key), dict):
            o, u = strict_counts(schema[key], depth + 1)
            optional, union = optional + o, union + u
    for key in ("anyOf", "allOf", "oneOf"):
        if isinstance(schema.get(key), list):
            for sub in schema[key]:
                o, u = strict_counts(sub, depth + 1)
                optional, union = optional + o, union + u
    return optional, union


def choose_strict(tools: list[ToolSpec]) -> list[ToolSpec]:
    """Marks the tools to send strict, within the Claude API's limits for one request."""
    order = sorted(range(len(tools)), key=lambda i: (not tools[i].command.mutates, i))
    chosen: set[int] = set()
    optional = union = 0
    for i in order:
        if not tools[i].strict or len(chosen) >= MAX_STRICT_TOOLS:
            continue
        o, u = strict_counts(tools[i].input_schema)
        if optional + o > MAX_STRICT_OPTIONAL or union + u > MAX_STRICT_UNION:
            continue
        chosen.add(i)
        optional, union = optional + o, union + u
    return [replace(t, send_strict=i in chosen) for i, t in enumerate(tools)]


def derive_tools(commands: list[Command], include: Callable[[Command], bool] | None = None) -> list[ToolSpec]:
    """One tool per command, in the table's order (stable, for prompt caching).

    `include` keeps only the commands it accepts: the seam for a session whose
    role needs fewer tools. The groups of NOT_FOR_THE_MODEL are always left
    out. The strict tools are chosen among those kept.
    """
    tools = []
    for c in commands:
        if c.group in NOT_FOR_THE_MODEL or (include is not None and not include(c)):
            continue
        stripped = split_type_lists(strip_for_strict(c.args))
        tools.append(ToolSpec(name=tool_name(c.name), command=c, input_schema=stripped,
                              strict=is_strict_compatible(stripped)))
    return choose_strict(tools)


def name_filter(patterns: str) -> Callable[[Command], bool]:
    """A filter from a comma-separated list of command names and groups
    (`catalog,spec.get`): a command is kept when its name or its group is listed."""
    wanted = {p.strip() for p in patterns.split(",") if p.strip()}
    return lambda c: c.name in wanted or c.group in wanted


def acp_tool_kind(command: Command) -> str:
    """The ACP tool kind the pane uses to pick an icon."""
    if not command.mutates:
        return "read"
    if command.verb in ("remove", "delete"):
        return "delete"
    if command.verb == "run":
        return "execute"
    return "edit"
