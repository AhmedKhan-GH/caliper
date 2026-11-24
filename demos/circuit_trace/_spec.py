"""Atomic scene panel and piece edits through the public spec command."""
import json
import math


def toml_value(value):
    if isinstance(value, str) or type(value) in (bool, int):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(toml_value(key) + " = " + toml_value(item)
                                 for key, item in value.items()) + " }"
    raise ValueError("The instrument spec contains an unsupported TOML value.")


def scene_programs(session, programs, *, remove_examples=False):
    spec = session.mcp.body("spec.get")["json"]
    identifiers = {program["id"] for program in programs}
    if remove_examples:
        identifiers.add("example_tokens")
    spec["program"] = [program for program in spec.get("program", []) if program["id"] not in identifiers]
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
    text = "\n".join(toml_value(key) + " = " + toml_value(value) for key, value in spec.items()) + "\n"
    session.mcp.body("spec.set_text", text=text)
