"""The provider presets (presets.json, presets.py): equal to the design page's table, checked as hostile input.

Also the scrubbing of every key shape that a provider's documentation shows,
and `--preset` on the sidecar's and the probe's command lines.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from conftest import AGENT_DIR, COMMAND_TABLE, AgentProcess
from fake_openai import FakeOpenAI

from caliper_agent.keyguard import REDACTED, Scrubber
from caliper_agent.presets import PRESETS, PRESETS_FILE, BaseUrlError, PresetError, check_base_url, choose, \
    custom_key_account

DESIGN_PAGE = AGENT_DIR.parent.parent.parent / "design" / "widget-agent.md"


def _table_after(text: str, heading: str) -> list[list[str]]:
    section = text.split(heading, 1)[1]
    rows = []
    for line in section.splitlines()[1:]:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]  # without the header and the rule


@pytest.mark.skipif(not DESIGN_PAGE.is_file(), reason="the design page is not beside the package")
def test_the_presets_equal_the_design_page():
    text = DESIGN_PAGE.read_text(encoding="utf-8")
    rows = _table_after(text, "### Presets")
    by_name = {p.name: p for p in PRESETS.values()}
    assert [r[0] for r in rows] == [p.name for p in PRESETS.values()]
    for name, base, key in rows:
        preset = by_name[name]
        urls = re.findall(r"`([^`]+)`", base)
        assert (urls[0] if urls else None) == preset.base_url, name
        expected = "none" if key == "None" else "optional" if key == "Optional" else "required"
        assert preset.key == expected, name
    # The second table: every service that a preset reaches, with its documented base URL.
    sources = _table_after(text, "Every base URL was read from")
    sources = [r for r in sources if r[2] != "Custom"]
    assert sorted(r[2] for r in sources) == sorted(p.name for p in PRESETS.values() if p.id not in ("anthropic", "custom"))
    for service, base, preset_name, _source in sources:
        assert re.findall(r"`([^`]+)`", base)[0] == by_name[preset_name].base_url, service
    # The data's own invariants.
    for p in PRESETS.values():
        assert (p.key == "none") == p.local and (p.key == "none") == (p.key_env is None)
        assert p.key_env in (None, "ANTHROPIC_API_KEY" if p.api == "claude" else "CALIPER_MODEL_API_KEY")
    assert json.loads(PRESETS_FILE.read_text())["version"] == 0


def test_the_claude_preset_lists_the_sidecars_models_and_efforts():
    from caliper_agent.claude import EFFORTS, MODELS
    doc = json.loads(PRESETS_FILE.read_text())
    claude = next(p for p in doc["presets"] if p["id"] == "anthropic")
    assert [(m["id"], m["name"]) for m in claude["models"]] == [(m.id, m.name) for m in MODELS]
    assert claude["efforts"] == list(EFFORTS)
    assert all("models" not in p and "efforts" not in p for p in doc["presets"] if p["id"] != "anthropic")


def test_a_custom_entry_is_named_by_its_base_url():
    # The same value as custom_key_account() in key_store.cpp (test_key_store.cpp).
    assert custom_key_account("https://api.mistral.ai/v1") == "custom-7c6535712768"
    assert custom_key_account("https://api.mistral.ai/v1/") == "custom-7c6535712768"


@pytest.mark.parametrize("url", [
    "", "ftp://x/v1", "file:///etc/passwd", "http:///v1", "http://user:pw@host/v1", "http://host/v1?key=1",
    "http://host/v1#f", "http://host/v1?", "http://ho st/v1", "http://host/v1\n", "http://host:99999/v1",
    "http://host/" + "a" * 2100, "javascript:alert(1)", "http://host/\x7f"])
def test_hostile_base_urls_are_refused(url):
    with pytest.raises(BaseUrlError):
        check_base_url(url)


def test_choices_of_preset_provider_and_base_url():
    assert choose(None, None, None).api == "claude" and choose(None, None, None).base_url is None
    c = choose("ollama", None, None)
    assert (c.api, c.base_url, c.warnings()) == ("openai-compatible", "http://localhost:11434/v1", [])
    c = choose("lmstudio", "openai-compatible", "http://192.0.2.7:1234/v1/")
    assert c.base_url == "http://192.0.2.7:1234/v1"
    assert [w["code"] for w in c.warnings()] == ["remote_host", "unencrypted"]
    assert "leave this machine" in c.warnings()[0]["message"]
    c = choose("custom", None, "https://models.example.com/v1")
    assert [w["code"] for w in c.warnings()] == ["remote_host"]
    assert choose("custom", None, "http://127.0.0.1:8080/v1").warnings() == []
    assert choose("openai", None, "https://api.openai.com/v1/").warnings() == []   # hosted: data leaves by design
    meta = choose("gemini", None, None).as_meta()
    assert meta == {"preset": "gemini", "api": "openai-compatible",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "host": "generativelanguage.googleapis.com", "loopback": False, "warnings": []}
    for bad in [("nope", None, None), ("anthropic", None, "http://127.0.0.1/v1"), ("anthropic", "openai-compatible", None),
                ("openai", None, "https://elsewhere.example.com/v1"), ("custom", None, None),
                (None, "openai-compatible", None), (None, "claude", "http://127.0.0.1/v1")]:
        with pytest.raises(PresetError):
            choose(*bad)


def test_every_documented_key_shape_is_scrubbed():
    s = Scrubber()
    keys = {
        "anthropic": "sk-ant-api03-" + "A1b2" * 10,
        "openrouter": "sk-or-v1-" + "0123456789abcdef" * 4,
        "google": "AIza" + "SyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe",  # Google Cloud's example key
    }
    for name, key in keys.items():
        assert s.scrub(f"the key is {key}.") == f"the key is {REDACTED}.", name
        assert s.contains_secret(key), name
    # Near misses stay: a short sk-or-v1- string, uppercase hex, a truncated Google key.
    for text in ("sk-or-v1-abc", "sk-or-v1-" + "ABCDEF0123456789" * 4, "AIzaShort", "a word like AIzawhatever"):
        assert s.scrub(text) == text, text
    # Any key the sidecar was given is scrubbed whatever its shape.
    s.register("plain-provider-key-0123456789")
    assert REDACTED in s.scrub("x plain-provider-key-0123456789 y")


def _exit_code(*args: str) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env["PYTHONPATH"] = str(AGENT_DIR)
    run = subprocess.run([sys.executable, "-P", "-m", "caliper_agent", "--command-table", str(COMMAND_TABLE), *args],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=60)
    return run.returncode, run.stderr


def test_the_sidecar_refuses_a_bad_choice():
    for args in (["--preset", "nope"], ["--preset", "anthropic", "--base-url", "http://127.0.0.1/v1"],
                 ["--preset", "openrouter", "--base-url", "https://elsewhere.example.com/v1", "--model", "m"],
                 ["--preset", "custom", "--model", "m"], ["--preset", "ollama"],
                 ["--provider", "openai-compatible", "--base-url", "http://u:p@127.0.0.1/v1", "--model", "m"]):
        code, err = _exit_code(*args)
        assert code == 2 and "caliper-agent:" in err, (args, err)


def test_a_preset_reaches_initialize_with_its_warnings(caliper):
    server = FakeOpenAI([{"text": "Hello."}])
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--preset", "ollama", "--base-url", server.base_url, "--model", "fake-model",
                                     "--tools", "catalog.list"])
    try:
        response, _ = agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        provider = response["result"]["_meta"]["caliper"]["provider"]
        assert provider["preset"] == "ollama" and provider["loopback"] is True and provider["warnings"] == []
        sid = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []})[0]["result"]["sessionId"]
        reply, _ = agent.prompt(sid, "Hi")
        assert reply["result"]["stopReason"] == "end_turn" and len(server.requests) == 1
    finally:
        agent.close()
        server.close()
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--preset", "lmstudio", "--base-url", "http://192.0.2.10:1234/v1", "--model", "m"])
    try:
        response, _ = agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        provider = response["result"]["_meta"]["caliper"]["provider"]
        assert [w["code"] for w in provider["warnings"]] == ["remote_host", "unencrypted"]
        assert provider["loopback"] is False and provider["host"] == "192.0.2.10"
        # The tests run offline, so the remote server is never reached.
        response, _ = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []})
        assert response["error"]["data"]["code"] == "agent.offline"
        assert "another machine" in agent.stderr()
    finally:
        agent.close()


def test_the_probe_takes_a_preset():
    server = FakeOpenAI(models=[{"id": "local-model"}], lmstudio=8192)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CALIPER_MODEL"))}
    env["PYTHONPATH"] = str(AGENT_DIR)
    try:
        run = subprocess.run([sys.executable, "-P", "-m", "caliper_agent", "probe", "--preset", "lmstudio",
                              "--base-url", server.base_url, "--model", "local-model", "--no-vision-probe"],
                             capture_output=True, text=True, env=env, timeout=60)
        doc = json.loads(run.stdout)
        assert doc["preset"] == "lmstudio" and doc["capabilities"]["context_length"]["value"] == 8192
        assert any(p.startswith("/api/v0/models/local-model") for p in server.paths)
    finally:
        server.close()
