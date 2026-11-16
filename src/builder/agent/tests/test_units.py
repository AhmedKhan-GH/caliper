"""Unit tests of the sidecar's pure parts: tools, schema checks, the key, the API file."""

from __future__ import annotations

import io
import json
import logging
import sys
import threading

import pytest

from conftest import COMMAND_TABLE, FAKE_KEY, general_tool_count, tiny_png

from caliper_agent import keyguard
from caliper_agent.caliper_api import (ApiFileError, MAX_IMAGE_BYTES, default_api_file, output_from_envelope,
                                       parse_api_file, png_size)
from pydantic_ai.exceptions import ModelHTTPError, ModelRateLimitError, ToolFailed
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models._anthropic_schema_limits import exceeded_limit

from caliper_agent.claude import (NO_SERVER_FALLBACK, SERVER_SIDE_FALLBACKS, claude_settings, model_ids)
from caliper_agent.loop import classify, close_tool_calls
from caliper_agent.openai_compat import BaseUrlError, check_base_url, refusal_kind
from caliper_agent.provider import AgentError
from caliper_agent.schema import MAX_DEPTH, validate
from caliper_agent.toolset import JsonObjectArgs
from caliper_agent.udiff import apply_unified_diff
from caliper_agent.tools import (NOT_FOR_THE_MODEL, STRICT_UNSUPPORTED, TableError, acp_tool_kind, choose_strict,
                                 command_name, derive_tools, is_strict_compatible, load_command_table, name_filter, not_for_agents, offered_to_the_model,
                                 parse_command_table, split_type_lists, strip_for_strict, tool_name)


# ---- tools ---------------------------------------------------------------------------------


def _walk_all(schema):
    yield schema
    for sub in (schema.get("properties") or {}).values():
        yield from _walk_all(sub)
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            yield from _walk_all(schema[key])
    for sub in schema.get("anyOf") or []:
        yield from _walk_all(sub)


def test_commands_not_for_agents_are_offered_in_no_session_and_never_sent(caliper):
    import asyncio

    from caliper_agent.caliper_api import CaliperApi
    from caliper_agent.widget import ScopeRule, widget_tools

    commands = load_command_table(COMMAND_TABLE)
    by_name = {c.name: c for c in commands}
    # The table's `agents` flag: today false for undo and redo, the person's own.
    closed = not_for_agents(commands)
    assert closed == {"history.undo", "history.redo"}
    assert not offered_to_the_model(by_name["history.undo"]) and offered_to_the_model(by_name["history.list"])
    # A general session, one narrowed to the history group, and a widget whose
    # scope names every command: none offers a command closed to agents.
    rule = ScopeRule.load(COMMAND_TABLE.parent / "widget_scopes.json")
    sessions = {"general": derive_tools(commands), "history": derive_tools(commands, name_filter("history")),
                "widget": widget_tools(commands, tuple(by_name), vision=True, rule=rule)}
    for session, tools in sessions.items():
        assert all(t.command.agents for t in tools) and not {t.command.name for t in tools} & closed, session
    assert "history.list" in {t.command.name for t in sessions["history"]}
    # The tool count the tests compare against follows the flag.
    offered = [c for c in commands if offered_to_the_model(c)]
    assert general_tool_count() == len(offered) == \
        len([c for c in commands if c.group not in NOT_FOR_THE_MODEL]) - len(closed)
    # A call made anyway is refused by the sidecar, with the command layer's
    # code for it, and never reaches Caliper.
    api = CaliperApi(caliper.api_file, not_for_agents=closed)
    try:
        for name in sorted(closed):
            out = asyncio.run(api.call(name, {}))
            assert not out.ok and out.error_code == "command.not_allowed", name
        assert asyncio.run(api.call("history.list", {})).ok
    finally:
        asyncio.run(api.aclose())
    assert [c["path"] for c in caliper.commands(dry=None)] == ["/v0/commands/history.list"]


def test_the_agents_flag_is_read_from_the_table_generically():
    def table(**flags):
        return json.dumps({"version": 0, "commands": [
            {"name": "a.read", "summary": "Reads.", "args": {"type": "object", "properties": {}}, **flags},
            {"name": "a.other", "summary": "Other.", "args": {"type": "object", "properties": {}}}]})

    # A table without the key predates it: every command is open to agents.
    assert [t.command.name for t in derive_tools(parse_command_table(table()))] == ["a.read", "a.other"]
    assert all(c.agents for c in parse_command_table(table()))
    # Any command marked false is left out, whatever its name.
    commands = parse_command_table(table(agents=False))
    assert not_for_agents(commands) == {"a.read"}
    assert [t.command.name for t in derive_tools(commands)] == ["a.other"]
    with pytest.raises(TableError, match="agents is not a boolean"):
        parse_command_table(table(agents="no"))


def test_every_command_of_the_real_table_becomes_a_tool_strict_within_the_limits():
    commands = load_command_table(COMMAND_TABLE)
    tools = derive_tools(commands)
    # The preview and review commands are the app's and the sidecar's, never the model's.
    assert {c.name for c in commands if c.group in NOT_FOR_THE_MODEL} == {
        "preview.begin", "preview.diff", "preview.commit", "preview.discard", "review.status"}
    commands = [c for c in commands if offered_to_the_model(c)]   # without undo and redo, the person's own
    assert len(tools) == len(commands) >= 34
    for tool, command in zip(tools, commands):
        assert tool.name == tool_name(command.name) and command_name(tool.name) == command.name
        assert tool.strict, f"{command.name} should be strict-compatible"
        for schema in _walk_all(tool.input_schema):
            assert not (STRICT_UNSUPPORTED & set(schema)), command.name
            assert not isinstance(schema.get("type"), list), command.name
        assert tool.description.startswith(command.summary.strip()[:20])
    # The tools sent strict fit the Claude API's limits for one request, by
    # Pydantic AI's own count; the commands that change something come first.
    strict = [t for t in tools if t.send_strict]
    assert 10 <= len(strict) <= 20
    assert exceeded_limit([t.input_schema for t in strict], None) is None
    assert exceeded_limit([t.input_schema for t in tools], None) is not None
    assert {"spec_set_text", "source_add", "loader_run", "applet_open"} <= {t.name for t in strict}
    by_name = {t.name: t for t in tools}
    assert "Caliper asks the user" in by_name["applet_save"].description
    assert "Caliper asks the user" not in by_name["catalog_list"].description
    # Previewable commands come from the schema: a boolean dry_run on a mutating command.
    previewable = sorted(t.command.name for t in tools if t.command.previewable)
    assert previewable == sorted(c["name"] for c in json.loads(COMMAND_TABLE.read_text())["commands"]
                                 if c["mutates"] and "dry_run" in c["args"].get("properties", {}))
    assert "panel.create" in previewable and "spec.set_text" in previewable
    assert "spec.new" not in previewable and "catalog.list" not in previewable


def test_names_are_reversible_and_kinds_follow_the_flags():
    assert tool_name("spec.set_text") == "spec_set_text"
    assert command_name("spec_set_text") == "spec.set_text"
    tools = {t.name: t for t in derive_tools(load_command_table(COMMAND_TABLE))}
    assert acp_tool_kind(tools["catalog_list"].command) == "read"
    assert acp_tool_kind(tools["panel_remove"].command) == "delete"
    assert acp_tool_kind(tools["loader_run"].command) == "execute"
    assert acp_tool_kind(tools["panel_create"].command) == "edit"


def test_stripping_keeps_property_names_that_look_like_keywords():
    schema = {"type": "object", "additionalProperties": False, "required": ["pattern"],
              "properties": {"pattern": {"type": "string", "pattern": "^a", "maxLength": 3},
                             "default": {"type": "integer", "minimum": 1, "default": 2},
                             "list": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4}}}
    out = strip_for_strict(schema)
    assert set(out["properties"]) == {"pattern", "default", "list"}
    assert out["properties"]["pattern"] == {"type": "string"}
    assert out["properties"]["default"] == {"type": "integer"}
    assert out["properties"]["list"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["pattern"]["pattern"] == "^a"  # the original is untouched


def test_an_open_object_makes_only_its_tool_non_strict():
    table = {"version": 0, "commands": [
        {"name": "panel.patch", "summary": "Patch.", "mutates": True, "destructive": False, "frame_thread": True,
         "args": {"type": "object", "additionalProperties": False, "required": ["patch"],
                  "properties": {"patch": {"type": "object", "additionalProperties": True}}}, "result": {}},
        {"name": "app.info", "summary": "Info.", "args": {"type": "object", "additionalProperties": False,
                                                          "required": [], "properties": {}}, "result": {}}]}
    patch, info = derive_tools(parse_command_table(json.dumps(table)))
    assert not patch.strict and info.strict
    assert not patch.send_strict and info.send_strict
    assert not is_strict_compatible({"type": "object", "properties": {}, "additionalProperties": False})
    assert not is_strict_compatible({"$ref": "#/x"})


@pytest.mark.parametrize("doc, message", [
    ("not json", "not JSON"),
    ('{"commands": 3}', "no list"),
    ('{"version": 1, "commands": []}', "version"),
    ('{"commands": [{"name": "Bad.Name", "args": {"type": "object"}}]}', "bad name"),
    ('{"commands": [{"name": "a.b", "args": {"type": "string"}}]}', "object schema"),
    ('{"commands": [{"name": "a.b", "args": {"type": "object"}, "mutates": "yes"}]}', "boolean"),
    ('{"commands": [{"name": "a.b", "args": {"type": "object"}}, {"name": "a.b", "args": {"type": "object"}}]}',
     "twice"),
])
def test_hostile_tables_are_refused(doc, message):
    with pytest.raises(TableError, match=message):
        parse_command_table(doc)


def test_table_limits():
    with pytest.raises(TableError, match="512"):
        parse_command_table(json.dumps({"commands": [{"name": f"g.c{i}", "args": {"type": "object"}}
                                                      for i in range(513)]}))
    deep: dict = {"type": "object"}
    for _ in range(40):
        deep = {"type": "object", "properties": {"x": deep}}
    with pytest.raises(TableError, match="deeply"):
        parse_command_table(json.dumps({"commands": [{"name": "a.b", "args": deep}]}))
    with pytest.raises(TableError, match="8 MiB"):
        parse_command_table(b" " * ((8 << 20) + 1))


# ---- schema checks ---------------------------------------------------------------------------


PANEL = {"type": "object", "additionalProperties": False, "required": ["id", "bind"],
         "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$", "maxLength": 64},
                        "ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95},
                        "where": {"type": "string", "enum": ["beside", "above"]},
                        "bind": {"type": "object", "additionalProperties": False, "required": [],
                                 "properties": {"y": {"type": ["string", "array"], "items": {"type": "string"}}}},
                        "n": {"type": "integer"}, "cols": {"type": "array", "items": {"type": "string"},
                                                           "maxItems": 2}}}


@pytest.mark.parametrize("value, error", [
    ({"id": "heat", "bind": {}}, None),
    ({"id": "heat", "bind": {"y": ["a.b", "c.d"]}, "ratio": 0.5, "n": 3.0}, None),
    ({"bind": {}}, "args: the property id is required"),
    ({"id": "Heat", "bind": {}}, "args.id: does not match"),
    ({"id": "heat", "bind": {}, "ratio": 2}, "args.ratio: must be at most 0.95"),
    ({"id": "heat", "bind": {}, "where": "left"}, "args.where: must be one of"),
    ({"id": "heat", "bind": {"y": [1]}}, "args.bind.y[0]: expected string"),
    ({"id": "heat", "bind": {"z": "a"}}, "args.bind.z: is not a known property"),
    ({"id": "heat", "bind": {}, "n": True}, "args.n: expected integer, got boolean"),
    ({"id": "heat", "bind": {}, "n": 1.5}, "args.n: expected integer"),
    ({"id": "heat", "bind": {}, "cols": ["a", "b", "c"]}, "args.cols: must have at most 2 items"),
    ({"id": "h" * 70, "bind": {}}, "args.id: must have at most 64 characters"),
    ([], "args: expected object, got array"),
])
def test_validation(value, error):
    result = validate(value, PANEL)
    if error is None:
        assert result is None
    else:
        assert result is not None and result.startswith(error), result


def test_validation_bounds_hostile_values():
    deep: object = "x"
    for _ in range(MAX_DEPTH + 5):
        deep = [deep]
    assert "nested deeper" in validate({"id": "a", "bind": {}, "cols": deep}, PANEL)
    wide = {"id": "a", "bind": {}, "cols": [0] * 1_100_000}
    assert "more than" in validate(wide, PANEL)
    long_id = {"id": "a" * 5000, "bind": {}}
    assert "4096 bytes" in validate(long_id, {"type": "object", "properties": {"id": {"type": "string",
                                                                                       "pattern": "^a+$"}}})


# ---- the key -------------------------------------------------------------------------------


def test_the_key_is_taken_and_every_override_removed():
    env = {"ANTHROPIC_API_KEY": FAKE_KEY, "ANTHROPIC_BASE_URL": "http://elsewhere", "ANTHROPIC_AUTH_TOKEN": "t",
           "ANTHROPIC_CUSTOM_HEADERS": "X: y", "ANTHROPIC_PROFILE": "p", "HOME": "/home/x"}
    key, status = keyguard.take_key_from_environment(env)
    assert status == keyguard.KeyStatus.PRESENT and key.reveal() == FAKE_KEY
    assert env == {"HOME": "/home/x"}
    assert FAKE_KEY not in repr(key) and FAKE_KEY not in str(key) and FAKE_KEY not in f"{key}"
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(key)
    assert keyguard.take_key_from_environment({})[1] == keyguard.KeyStatus.MISSING
    assert keyguard.take_key_from_environment({"ANTHROPIC_API_KEY": ""})[1] == keyguard.KeyStatus.MISSING
    for bad in ("has space", "x" * 2000, "tab\there", "é-key"):
        assert keyguard.take_key_from_environment({"ANTHROPIC_API_KEY": bad}) == (None, keyguard.KeyStatus.MALFORMED)


def test_the_scrubber_replaces_secrets_in_text_logs_and_exceptions(capsys):
    scrubber = keyguard.Scrubber()
    scrubber.register(keyguard.Secret(FAKE_KEY))
    scrubber.register("short")  # too short to register safely
    assert scrubber.scrub(f"a {FAKE_KEY} b") == "a [redacted] b"
    assert scrubber.scrub("a sk-ant-other0123456789 b") == "a [redacted] b"
    assert scrubber.scrub("short stays") == "short stays"
    assert scrubber.contains_secret(FAKE_KEY) and not scrubber.contains_secret("plain")

    keyguard.SCRUBBER.register(FAKE_KEY)
    message = keyguard.describe_exception(RuntimeError(f"bad key {FAKE_KEY}\nmore"))
    assert message == "RuntimeError: bad key [redacted]"

    stream = io.StringIO()
    old = sys.stderr
    saved = (sys.excepthook, threading.excepthook, sys.unraisablehook, logging.getLogger().handlers[:],
             logging.getLogger().level)
    sys.stderr = stream
    try:
        keyguard.install_guards()
        logging.getLogger("test").warning("the key is %s", FAKE_KEY)
        try:
            raise ValueError(f"oops {FAKE_KEY}")
        except ValueError:
            sys.excepthook(*sys.exc_info())
            logging.getLogger("test").exception("failed with %s", FAKE_KEY)
    finally:
        sys.stderr = old
        sys.excepthook, threading.excepthook, sys.unraisablehook = saved[:3]
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved[3]:
            root.addHandler(handler)
        root.setLevel(saved[4])
    written = stream.getvalue()
    assert FAKE_KEY not in written and written.count("[redacted]") >= 3


# ---- Caliper API ---------------------------------------------------------------------------


def _api(**fields):
    doc = {"version": 1, "pid": 1, "port": 7341, "token": "abcdefghijklmnop_-1234567890ABCD",
           "url": "http://127.0.0.1:7341/v0"}
    doc.update(fields)
    return json.dumps(doc).encode()


def test_the_api_file_is_checked():
    endpoint = parse_api_file(_api())
    assert endpoint.base_url == "http://127.0.0.1:7341"
    assert "abcdefghijklmnop" not in repr(endpoint)
    assert parse_api_file(_api(url="http://localhost:7341/v0")).host == "localhost"
    assert parse_api_file(_api(url="http://[::1]:7341/v0")).base_url == "http://[::1]:7341"
    assert parse_api_file(_api(url=None)).port == 7341
    for bad, code in ((_api(url="http://10.0.0.5:7341/v0"), "caliper.not_loopback"),
                      (_api(url="http://evil.example:7341/v0"), "caliper.not_loopback"),
                      (_api(url="https://127.0.0.1:7341/v0"), "caliper.bad_api_file"),
                      (_api(token="short"), "caliper.bad_api_file"),
                      (_api(token="has space in it 1234567"), "caliper.bad_api_file"),
                      (_api(port=99999, url=None), "caliper.bad_api_file"),
                      (_api(port=1234), "caliper.bad_api_file"),
                      (b"[1, 2]", "caliper.bad_api_file"),
                      (b"{not json", "caliper.bad_api_file"),
                      (b" " * (65 << 10), "caliper.bad_api_file")):
        with pytest.raises(ApiFileError) as err:
            parse_api_file(bad)
        assert err.value.code == code


def test_the_api_file_location(monkeypatch, tmp_path):
    assert default_api_file({"CALIPER_DATA_DIR": str(tmp_path)}) == tmp_path / "api.json"
    path = default_api_file({})
    assert path.name == "api.json" and path.parent.name.lower() == "caliper"


def test_envelopes_and_images():
    out = output_from_envelope({"ok": True, "result": {"entries": []}})
    assert out.ok and json.loads(out.text) == {"entries": []}
    out = output_from_envelope({"ok": False, "error": {"code": "panel.not_found", "message": "No panel.",
                                                       "where": "args.id"}})
    assert not out.ok and out.error_code == "panel.not_found" and "args.id" in out.text
    out = output_from_envelope({"ok": False, "error": {"code": "command.needs_approval", "message": "Ask.",
                                                       "where": "", "approval": "a-1"}})
    assert json.loads(out.text)["error"]["approval"] == "a-1"
    import base64
    png = tiny_png(6, 4)
    out = output_from_envelope({"ok": True, "result": {"png_base64": base64.b64encode(png).decode(), "width": 6,
                                                       "height": 4}})
    assert out.ok and out.png == png and (out.width, out.height) == (6, 4) and "png_base64" not in out.text
    assert not output_from_envelope({"ok": True, "result": {"png_base64": "!!!"}}).ok
    assert not output_from_envelope({"ok": True, "result": {"png_base64": base64.b64encode(b"GIF89a").decode()}}).ok
    big = png[:24] + b"\x00" * MAX_IMAGE_BYTES
    out = output_from_envelope({"ok": True, "result": {"png_base64": base64.b64encode(big).decode()}})
    assert not out.ok and out.error_code == "caliper.image_too_large"
    assert not output_from_envelope("nonsense").ok
    assert png_size(png) == (6, 4) and png_size(b"\x89PNG") is None
    long = output_from_envelope({"ok": True, "result": {"rows": ["x" * 1000] * 200}})
    assert long.text.endswith("characters.]")


# ---- the Claude provider's pure parts --------------------------------------------------------


def test_server_side_fallbacks_are_off():
    # Pydantic AI drops the reply's `fallback` block, so the setting stays off.
    assert SERVER_SIDE_FALLBACKS is False
    for model in model_ids():
        settings = claude_settings(model, "high")
        assert "extra_body" not in settings and "anthropic_betas" not in settings


def test_models():
    assert model_ids() == ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2")
    assert NO_SERVER_FALLBACK == {"claude-haiku-5-5"}


# ---- the turn's pure parts --------------------------------------------------------------------


def test_type_lists_become_any_of_and_keep_their_meaning():
    schema = {"type": "object", "additionalProperties": False, "required": ["a", "b"],
              "properties": {"a": {"type": ["integer", "array"], "items": {"type": "integer"}, "description": "d"},
                             "b": {"type": ["string", "null"], "enum": ["x", "y", None]},
                             "c": {"type": ["number"]}}}
    out = split_type_lists(schema)
    assert out["properties"]["a"] == {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                                      "description": "d"}
    assert out["properties"]["b"] == {"anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]}
    assert out["properties"]["c"] == {"type": "number"}
    for value, ok in (({"a": 1, "b": None}, True), ({"a": [1, 2], "b": "x"}, True), ({"a": "1", "b": "x"}, False)):
        assert (validate(value, schema) is None) == ok


def test_choose_strict_stays_within_the_limits():
    def command_table(n, optional):
        return {"version": 0, "commands": [
            {"name": f"g.c{i}", "summary": "S.", "mutates": i % 2 == 0,
             "args": {"type": "object", "additionalProperties": False, "required": [],
                      "properties": {f"p{j}": {"type": "string"} for j in range(optional)}}}
            for i in range(n)]}
    tools = derive_tools(parse_command_table(json.dumps(command_table(30, 0))))
    assert sum(t.send_strict for t in tools) == 20
    assert all(t.send_strict for t in tools if t.command.mutates)  # the mutating ones first
    tools = derive_tools(parse_command_table(json.dumps(command_table(10, 5))))
    assert sum(t.send_strict for t in tools) == 4  # 4 x 5 optional parameters <= 24
    assert exceeded_limit([t.input_schema for t in tools if t.send_strict], None) is None
    assert choose_strict([]) == []


def test_the_tool_filter_keeps_names_and_groups():
    commands = load_command_table(COMMAND_TABLE)
    tools = derive_tools(commands, name_filter("catalog, spec.get"))
    assert sorted(t.name for t in tools) == ["catalog_describe", "catalog_list", "catalog_sample", "spec_get"]


def test_tool_calls_without_results_are_closed():
    call = ToolCallPart(tool_name="a", args="{}", tool_call_id="c1")
    call2 = ToolCallPart(tool_name="b", args="{}", tool_call_id="c2")
    history = [ModelRequest(parts=[]), ModelResponse(parts=[call, call2])]
    out = close_tool_calls(history, "Cancelled.")
    [closing] = out[2:]
    assert [(p.tool_call_id, p.outcome, p.content) for p in closing.parts] == [("c1", "failed", "Cancelled."),
                                                                               ("c2", "failed", "Cancelled.")]
    partial = history + [ModelRequest(parts=[ToolReturnPart(tool_name="a", content="ok", tool_call_id="c1")],
                                      state="interrupted")]
    out = close_tool_calls(partial, "Cancelled.")
    assert [p.tool_call_id for p in out[-1].parts] == ["c1", "c2"] and out[-1].state == "complete"
    assert close_tool_calls(out, "again") == out
    assert close_tool_calls([], "x") == []


def test_errors_are_classified_by_pydantic_ais_categories():
    retry = classify(ModelHTTPError.for_category(ModelRateLimitError, status_code=429, model_name="m",
                                                 headers={"retry-after": "7"}), "Claude API", "m")
    assert (retry.kind, retry.http_status, retry.delay) == ("rate_limit", 429, 7.0)
    for status, code in ((401, "agent.key_rejected"), (402, "agent.billing"), (403, "agent.forbidden"),
                         (404, "agent.model_unavailable"), (413, "agent.request_too_large"),
                         (400, "agent.bad_request")):
        with pytest.raises(AgentError) as err:
            classify(ModelHTTPError(status, "m", {"error": {"message": f"key {FAKE_KEY}"}}), "Claude API", "m")
        assert err.value.code == code and FAKE_KEY not in err.value.message
    assert classify(ModelHTTPError(503, "m"), "Claude API", "m").kind == "server"
    with pytest.raises(AgentError):
        classify(RuntimeError("boom"), "Claude API", "m")


def test_openai_compatible_refusals_and_addresses():
    assert refusal_kind("registry.ollama.ai/library/x does not support tools") == "tools"
    assert refusal_kind("This model does not support image input") == "vision"
    assert refusal_kind("context length exceeded") is None
    assert check_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434/v1"
    for bad in ("ftp://x/v1", "http://user:pw@host/v1", "not a url"):
        with pytest.raises(BaseUrlError):
            check_base_url(bad)


def test_the_args_validator_takes_json_objects_only():
    v = JsonObjectArgs()
    assert v.validate_json('{"a": 1}') == {"a": 1}
    for bad in ('{"a": ', "[1]", "3", "nul"):
        with pytest.raises(ToolFailed) as err:
            v.validate_json(bad)
        assert "INVALID_JSON" in json.loads(err.value.message)


# ---- applying a unified diff ------------------------------------------------------------------


def test_unified_diffs_apply_and_mismatches_are_refused():
    import difflib
    old = "".join(f"line {i}\n" for i in range(30))
    new = old.replace("line 3\n", "line three\n").replace("line 20\n", "").replace("line 29\n",
                                                                                    "line 29\nline 30\n")
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "before", "after"))
    assert apply_unified_diff(old, diff) == new
    assert apply_unified_diff(old, "") == old
    assert apply_unified_diff("", "--- before\n+++ after\n@@ -0,0 +1,2 @@\n+a\n+b\n") == "a\nb\n"
    # The marker of a last line without a line feed, as `diff -u` writes it.
    assert apply_unified_diff("a\nb", "--- before\n+++ after\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n"
                              "+c\n\\ No newline at end of file\n") == "a\nc"
    assert apply_unified_diff("x\n" + old, diff) is None          # the context no longer matches
    assert apply_unified_diff(old, "--- before\n+++ after\n@@ the diff could not be computed @@\n") is None
    assert apply_unified_diff(old, "--- before\n+++ after\n") is None
    assert apply_unified_diff(old, diff.replace("@@ -1,7", "@@ -1,9")) is None  # counts that disagree


def test_a_null_on_an_optional_argument_is_dropped_as_the_command_layer_drops_it():
    # The call a local model sent: every optional argument of source.add null.
    from caliper_agent.schema import drop_null_optionals
    from caliper_agent.tools import load_command_table
    from conftest import COMMAND_TABLE
    table = {c.name: c for c in load_command_table(COMMAND_TABLE)}
    add = table["source.add"].args
    call = {"id": "layer0", "catalog": "live:entropy.l0", "sql": None, "from": None, "transform": None,
            "dry_run": False}
    assert validate(call, add) is not None
    understood, dropped = drop_null_optionals(call, add)
    assert dropped == 3 and understood == {"id": "layer0", "catalog": "live:entropy.l0", "dry_run": False}
    assert validate(understood, add) is None
    assert call["sql"] is None   # the caller's value is not changed
    # A required null stays and is refused; nested optional nulls go; a null the schema admits stays.
    assert drop_null_optionals({"id": None, "catalog": "x"}, add) == ({"id": None, "catalog": "x"}, 0)
    nested, n = drop_null_optionals({"id": "e", "transform": [{"op": "reduce", "axis": None}, {"op": None}]}, add)
    assert n == 1 and nested["transform"] == [{"op": "reduce"}, {"op": None}]
    patch = table["panel.patch"].args
    kept, n = drop_null_optionals({"id": "loss", "patch": {"title": None, "bind": None}}, patch)
    assert n == 1 and kept == {"id": "loss", "patch": {"title": None}}
