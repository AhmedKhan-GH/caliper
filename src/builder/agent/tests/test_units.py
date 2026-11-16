"""Unit tests of the sidecar's pure parts: tools, schema checks, the key, the API file."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading

import pytest

from conftest import COMMAND_TABLE, FAKE_KEY, tiny_png

from caliper_agent import keyguard
from caliper_agent.caliper_api import (ApiFileError, MAX_IMAGE_BYTES, default_api_file, output_from_envelope,
                                       parse_api_file, png_size)
from caliper_agent.claude import NO_SERVER_FALLBACK, api_tool, echo_content, model_ids
from caliper_agent.schema import MAX_DEPTH, validate
from caliper_agent.tools import (STRICT_UNSUPPORTED, TableError, acp_tool_kind, command_name, derive_tools,
                                 is_strict_compatible, load_command_table, parse_command_table, strip_for_strict,
                                 tool_name)


# ---- tools ---------------------------------------------------------------------------------


def _walk_schemas(schema):
    yield schema
    for sub in (schema.get("properties") or {}).values():
        yield from _walk_schemas(sub)
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            yield from _walk_schemas(schema[key])


def test_every_command_of_the_real_table_becomes_a_strict_tool():
    commands = load_command_table(COMMAND_TABLE)
    tools = derive_tools(commands)
    assert len(tools) == len(commands) >= 35
    for tool, command in zip(tools, commands):
        assert tool.name == tool_name(command.name) and command_name(tool.name) == command.name
        assert tool.strict, f"{command.name} should be a strict tool"
        for schema in _walk_schemas(tool.input_schema):
            assert not (STRICT_UNSUPPORTED & set(schema)), command.name
        api = api_tool(tool, strict=True)
        assert api["strict"] is True and api["eager_input_streaming"] is True
        assert api["description"].startswith(command.summary.strip()[:20])
    by_name = {t.name: t for t in tools}
    assert "Caliper asks the user" in by_name["applet_save"].description
    assert "Caliper asks the user" not in by_name["catalog_list"].description
    assert by_name["panel_create"].command.edits_spec and not by_name["spec_new"].command.edits_spec
    assert by_name["spec_set_text"].command.edits_spec and not by_name["catalog_list"].command.edits_spec


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
    assert "strict" not in api_tool(patch, strict=True)
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


def test_echo_after_a_mid_reply_fallback():
    fallback = {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}}
    blocks = [{"type": "thinking", "thinking": "x", "signature": "s"}, {"type": "text", "text": "partial"},
              {"type": "redacted_thinking", "data": "r"}, {"type": "tool_use", "id": "t1", "name": "n", "input": {}},
              {"type": "server_tool_use", "id": "s1"}, {"type": "web_search_tool_result", "tool_use_id": "s1"},
              {"type": "server_tool_use", "id": "s2"}, {"type": "mystery"}, fallback,
              {"type": "thinking", "thinking": "y", "signature": "t"}, {"type": "text", "text": "answer"},
              {"type": "tool_use", "id": "t2", "name": "n", "input": {}}]
    out = echo_content(blocks)
    assert [b["type"] for b in out] == ["text", "server_tool_use", "web_search_tool_result", "fallback", "thinking",
                                        "text", "tool_use"]
    assert out[-1]["id"] == "t2"
    plain = [{"type": "thinking", "thinking": "x", "signature": "s"}, {"type": "text", "text": "t"}]
    assert echo_content(plain) == plain


def test_models():
    assert model_ids() == ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2")
    assert NO_SERVER_FALLBACK == {"claude-haiku-5-5"}
