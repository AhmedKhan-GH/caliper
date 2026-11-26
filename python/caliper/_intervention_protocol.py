"""Bounded intervention metadata shared by request and lifecycle validation."""
import json
import math

if __package__:
    from ._interpret_protocol import integer, text
else:
    # The standalone wire codec loads this sibling without the Caliper package.
    import importlib.util as _import_util
    from pathlib import Path as _Path

    _spec = _import_util.spec_from_file_location(
        f"{__name__}_interpret_protocol", _Path(__file__).with_name("_interpret_protocol.py"))
    _protocol = _import_util.module_from_spec(_spec)
    _spec.loader.exec_module(_protocol)
    integer, text = _protocol.integer, _protocol.text

KINDS = ("ablate", "add", "scale", "clamp", "patch")
MAX_VECTOR = 16384
MAX_POSITIONS = 32
MAX_BYTES = 262144
FIELDS = {"ablate": set(), "add": {"vector", "coefficient", "positions"},
          "scale": {"feature", "multiplier", "positions", "dictionary"},
          "clamp": {"feature", "value", "positions", "dictionary"},
          "patch": {"source_prompt", "source_position", "positions"}, "steer": set()}


def finite(value):
    if type(value) not in (int, float):
        return False
    if type(value) is int and abs(value) > 2**53 - 1:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def specification(value, *, lifecycle=False):
    if type(value) is not dict:
        return False
    op, kind = value.get("op"), value.get("kind")
    if op is not None and kind is not None and op != kind:
        return False
    kind = kind if kind is not None else op
    if not isinstance(kind, str) or kind not in FIELDS or not text(value.get("point"), 255, False):
        return False
    allowed = {"point", "op", "kind", "from_step", "until_step"} | FIELDS[kind]
    if lifecycle:
        allowed |= {"since_step", "state", "error", "error_code"}
        if value.get("state") not in ("pending", "active", "ended"):
            return False
        if "error" in value and not text(value["error"], 1024):
            return False
        if "error_code" in value and not text(value["error_code"], 63, False):
            return False
    if set(value) - allowed:
        return False
    for key in ("from_step", "until_step", "since_step"):
        if value.get(key) is not None and not integer(value[key], 0, 2**53 - 1):
            return False
    start, end = value.get("from_step"), value.get("until_step")
    if start is not None and end is not None and end <= start:
        return False
    if "positions" in value:
        positions = value["positions"]
        if (type(positions) is not list or len(positions) > MAX_POSITIONS
                or any(not integer(i, 0, 31) for i in positions) or len(set(positions)) != len(positions)):
            return False
    if "dictionary" in value and not text(value["dictionary"], 255, False):
        return False
    if kind == "add":
        vector = value.get("vector")
        if (type(vector) is not list or not 0 < len(vector) <= MAX_VECTOR
                or not all(finite(x) for x in vector) or not finite(value.get("coefficient"))):
            return False
    if kind in ("scale", "clamp"):
        if not integer(value.get("feature"), 0, 2**31 - 1):
            return False
        if not finite(value.get("multiplier" if kind == "scale" else "value")):
            return False
    if kind == "patch":
        if not text(value.get("source_prompt"), 4096, False):
            return False
        if "source_position" in value and not integer(value["source_position"], 0, 31):
            return False
    try:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) <= MAX_BYTES
    except (ValueError, UnicodeError, TypeError):
        return False
