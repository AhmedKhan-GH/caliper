"""A validator for the JSON Schema subset of Caliper's command descriptors.

The command layer checks every argument object against its full schema, with
the keywords listed in docs/design/command-layer.md (`type`, `properties`,
`required`, `additionalProperties`, `items`, `minItems`, `maxItems`, `enum`,
`minimum`, `maximum`, `minLength`, `maxLength`, `pattern`). The sidecar runs the
same checks on each tool input the model produced before it runs the tool,
because eager input streaming leaves validation to the client and the SDK's
tolerant parser can return a truncated object. The command layer stays the
authority; this check only stops malformed input early.

The validator bounds its own work, since its input comes from a model: a value
nested deeper than 64 levels or holding more than 1,048,576 values is refused
before it is walked, and `pattern` is applied only to strings of at most 4096
bytes. `validate()` returns the first error as "<where>: <message>", or None.
Other keywords are annotations and are ignored.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any

MAX_DEPTH = 64
MAX_VALUES = 1_048_576
MAX_SCHEMA_DEPTH = 32
PATTERN_MAX_BYTES = 4096


def measure(value: Any) -> tuple[int, int]:
    """The depth and the number of values of a JSON value, without recursion."""
    count = 0
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > deepest:
            deepest = depth
        if count > MAX_VALUES or deepest > MAX_DEPTH:
            return deepest, count
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
    return deepest, count


def _type_matches(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if name == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    return False


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _check(value: Any, schema: Any, where: str, depth: int) -> str | None:
    if depth > MAX_SCHEMA_DEPTH:
        return f"{where}: the schema is nested too deeply"
    if not isinstance(schema, dict):
        return None  # `true` or a non-schema: nothing to check

    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types) if isinstance(types, list) else []
        if names and not any(_type_matches(value, n) for n in names):
            return f"{where}: expected {' or '.join(names)}, got {_type_name(value)}"

    if "enum" in schema and isinstance(schema["enum"], list):
        if not any(_json_equal(value, option) for option in schema["enum"]):
            allowed = ", ".join(json.dumps(o) for o in schema["enum"][:16])
            return f"{where}: must be one of {allowed}"

    if isinstance(value, str):
        length = len(value)  # code points
        if isinstance(schema.get("minLength"), int) and length < schema["minLength"]:
            return f"{where}: must have at least {schema['minLength']} characters"
        if isinstance(schema.get("maxLength"), int) and length > schema["maxLength"]:
            return f"{where}: must have at most {schema['maxLength']} characters"
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            if len(value.encode("utf-8")) > PATTERN_MAX_BYTES:
                return f"{where}: is longer than {PATTERN_MAX_BYTES} bytes, the most a pattern is checked on"
            compiled = _compile(pattern)
            if compiled is not None and compiled.search(value) is None:
                return f"{where}: does not match the pattern {pattern}"

    if _type_matches(value, "number"):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            return f"{where}: must be at least {schema['minimum']}"
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            return f"{where}: must be at most {schema['maximum']}"

    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            return f"{where}: must have at least {schema['minItems']} items"
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            return f"{where}: must have at most {schema['maxItems']} items"
        items = schema.get("items")
        if isinstance(items, dict):
            for i, element in enumerate(value):
                err = _check(element, items, f"{where}[{i}]", depth + 1)
                if err:
                    return err

    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for name in sorted(n for n in required if isinstance(n, str)):
            if name not in value:
                return f"{where}: the property {name} is required"
        extra = schema.get("additionalProperties", True)
        for name in sorted(value):
            child = f"{where}.{name}"
            if name in properties:
                err = _check(value[name], properties[name], child, depth + 1)
            elif extra is False:
                err = f"{child}: is not a known property"
            elif isinstance(extra, dict):
                err = _check(value[name], extra, child, depth + 1)
            else:
                err = None
            if err:
                return err
    return None


def validate(value: Any, schema: dict, where: str = "args") -> str | None:
    """The first error of `value` against `schema`, or None when it passes."""
    depth, count = measure(value)
    if depth > MAX_DEPTH:
        return f"{where}: is nested deeper than {MAX_DEPTH} levels"
    if count > MAX_VALUES:
        return f"{where}: holds more than {MAX_VALUES} values"
    return _check(value, schema, where, 0)
