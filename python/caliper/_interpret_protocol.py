"""Bounded structured metadata for interpretation CONTROL requests."""
import json
import math

CAPABILITY = "interpretation"
OP = "interpret"
VERSION = 1
MAX_BYTES = 262144
MAX_STRING = 4096
MAX_LIST = 512
MAX_FIELDS = 64
MAX_DEPTH = 8
MAX_VALUES = 16384
READ_METHODS = frozenset(("feature.dictionary", "feature.show", "feature.examples", "trace.status", "trace.read"))
METHODS = READ_METHODS | {"trace.run"}


def text(value, limit=MAX_STRING, empty=True):
    if not isinstance(value, str) or (not empty and not value):
        return False
    try:
        return len(value.encode("utf-8")) <= limit and "\x00" not in value
    except UnicodeError:
        return False


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def bounded(value):
    """Check object bytes, depth, fields, arrays, strings, and finite numbers."""
    count = 0

    def visit(v, depth):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > MAX_VALUES:
            return False
        if v is None or type(v) is bool:
            return True
        if type(v) is int:
            return abs(v) <= 2**53 - 1
        if type(v) is float:
            return math.isfinite(v)
        if isinstance(v, str):
            return text(v)
        if type(v) is list:
            return len(v) <= MAX_LIST and all(visit(x, depth + 1) for x in v)
        if type(v) is dict:
            return len(v) <= MAX_FIELDS and all(text(k, 63, False) and visit(x, depth + 1) for k, x in v.items())
        return False

    return (type(value) is dict and visit(value, 0)
            and len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) <= MAX_BYTES)


def request(value):
    if not bounded(value) or set(value) != {"version", "method", "params"}:
        return False
    if type(value["version"]) is not int or value["version"] != VERSION:
        return False
    method, p = value["method"], value["params"]
    if not isinstance(method, str) or method not in METHODS or type(p) is not dict:
        return False
    allowed = {
        "feature.dictionary": set(), "trace.status": set(),
        "feature.show": {"layer", "feature", "node"},
        "feature.examples": {"layer", "feature", "node", "limit"},
        "trace.run": {"prompt", "max_targets", "max_nodes", "max_edges"},
        "trace.read": {"node_offset", "node_limit", "edge_offset", "edge_limit"},
    }[method]
    if set(p) - allowed:
        return False
    bounds = {"layer": (0, 4095), "feature": (0, 2**31 - 1), "node": (0, 511),
              "limit": (0, 16), "max_targets": (0, 512), "max_nodes": (1, 512),
              "max_edges": (0, 8192), "node_offset": (0, 512), "edge_offset": (0, 8192),
              "node_limit": (0, 256), "edge_limit": (0, 256)}
    if any(k in p and not integer(p[k], *b) for k, b in bounds.items()):
        return False
    if method in ("feature.show", "feature.examples"):
        return (("node" in p and "layer" not in p and "feature" not in p)
                or ("node" not in p and "layer" in p and "feature" in p))
    if method == "trace.run":
        return text(p.get("prompt"), empty=False)
    return True
