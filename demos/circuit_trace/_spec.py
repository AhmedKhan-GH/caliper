"""Atomic scene panel and library piece wiring through public spec commands."""
from __future__ import annotations

import json
import math


def toml_value(value) -> str:
    """Serialize a JSON-compatible spec value into inline TOML."""
    if isinstance(value, str) or type(value) in (bool, int):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{toml_value(k)} = {toml_value(v)}" for k, v in value.items()) + " }"
    raise ValueError("The instrument spec contains an unsupported TOML value.")


def scene_programs(
    session,
    programs: list[dict],
    *,
    remove_examples: bool = False,
    state_defaults: dict | None = None,
) -> None:
    """Attach library pieces to scene panels and commit the updated spec."""
    for program in programs:
        pin = program["piece"]
        found = session.mcp.body("piece.show", name=pin["name"], version=pin["version"], **{"from": "library"})
        pin.update(hash=found["piece"]["hash"], **{"from": found["from"]})
    spec = session.mcp.body("spec.get")["json"]
    if state_defaults:
        for st in spec.get("state", []):
            if st["id"] in state_defaults:
                st["default"] = state_defaults[st["id"]]
    identifiers = {program["id"] for program in programs}
    if remove_examples:
        identifiers.add("example_tokens")
    spec["program"] = [p for p in spec.get("program", []) if p["id"] not in identifiers]
    spec["program"].extend(programs)
    targets = {program["panel"] for program in programs}
    for panel in spec["panel"]:
        if panel["id"] in targets:
            panel["primitive"] = "scene"
            panel["options"] = {"axes": False, "labels": True, "scale_bar": False}
            if panel["id"] == "graph":
                panel["options"]["select"] = {"state": "selected_node", "field": "index"}
        elif remove_examples and panel["id"] == "examples":
            panel["primitive"] = "text"
            panel["options"] = {"text": "No feature is selected."}
    text = "\n".join(f"{toml_value(k)} = {toml_value(v)}" for k, v in spec.items()) + "\n"
    session.mcp.body("spec.set_text", text=text)
