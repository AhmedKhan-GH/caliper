"""Command table compatibility and effect boundaries."""

import json
from dataclasses import replace

import pytest

from conftest import COMMAND_TABLE
from caliper_agent.tools import (TableError, derive_tools, load_command_table, not_for_agents,
                                 parse_command_table)
from caliper_agent.widget import ScopeRule


def test_real_run_effects_keep_permission_flags_and_widget_scope():
    source = json.loads(COMMAND_TABLE.read_text())["commands"]
    raw = {item["name"]: item for item in source if item.get("effect") == "run"}
    commands = load_command_table(COMMAND_TABLE)
    run = [command for command in commands if command.effect == "run"]
    assert {command.name for command in run} == set(raw)
    assert {"run.set_param", "run.action", "run.pause", "intervene.set", "intervene.clear"} <= set(raw)
    for command in run:
        for flag in ("mutates", "destructive", "frame_thread", "agents"):
            assert getattr(command, flag) == raw[command.name][flag]
        assert not command.mutates and not command.previewable
    assert {tool.command.name for tool in derive_tools(run)} == {command.name for command in run if command.agents}
    closed = [replace(command, agents=False) for command in run]
    assert derive_tools(closed) == []
    assert not_for_agents(closed) == set(raw)
    rule = ScopeRule.load(COMMAND_TABLE.parent / "widget_scopes.json")
    for target in rule.targets:
        assert not set(rule.scope(target, commands)) & set(raw)


def test_unknown_effect_is_refused_and_an_absent_effect_stays_change():
    entry = {"name": "sample.action", "args": {"type": "object", "properties": {}}}
    document = {"version": 0, "commands": [entry]}
    assert parse_command_table(json.dumps(document))[0].effect == "change"
    for effect in ("execute", "write", "run ", None, False):
        entry["effect"] = effect
        with pytest.raises(TableError, match="effect"):
            parse_command_table(json.dumps(document))


@pytest.mark.parametrize("field", [{}, {"description": "Any JSON value."}])
def test_unrestricted_fields_remain_available_without_strict_mode(field):
    args = {"type": "object", "additionalProperties": False, "required": ["value"],
            "properties": {"value": field}}
    entry = {"name": "sample.edit", "effect": "edit", "mutates": True, "args": args}
    commands = parse_command_table(json.dumps({"version": 0, "commands": [entry]}))
    tool, = derive_tools(commands)
    assert not tool.strict and not tool.send_strict
    assert tool.input_schema == args == tool.command.args
    assert tool.command.effect == "edit" and tool.command.mutates
