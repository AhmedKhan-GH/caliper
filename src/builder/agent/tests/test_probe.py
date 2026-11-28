"""The capability probe (probe.py), its cache (capabilities.py), `caliper-agent probe`, and the sidecar's use of them.

The probe runs against the fake OpenAI-compatible server in each capability
mode and with each quirk, and against the stubbed Claude transport, which
answers the Models API. The cache is tested for its expiry, the overrides,
corruption and its bounds. Nothing leaves the machine and no real key is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import AGENT_DIR, FAKE_KEY, AgentProcess, message, text, widget_meta
from fake_openai import FakeOpenAI

from caliper_agent.capabilities import (CACHE_FILE, MAX_ENTRIES, TTL_S, CacheError, Entry, ProbeCache, resolve)
from caliper_agent.keyguard import SCRUBBER, Secret
from caliper_agent.probe import Capability, ProbeResult, probe_claude, probe_openai_compatible, solid_png
from caliper_agent.stub_transport import StubTransport

MODEL_KEY = "sk-model-TESTONLY-0123456789abcdefghijklmn"


def probe(server: FakeOpenAI, model: str = "fake-model", **kwargs) -> ProbeResult:
    return asyncio.run(probe_openai_compatible(server.base_url, model, kwargs.pop("key", None), **kwargs))


@pytest.fixture
def fake():
    servers: list[FakeOpenAI] = []

    def start(**flags) -> FakeOpenAI:
        server = FakeOpenAI(**flags)
        servers.append(server)
        return server

    yield start
    for s in servers:
        s.close()


# ---- the probe against the fake server ------------------------------------------------------------


def test_a_capable_model(fake):
    server = fake(models=[{"id": "fake-model", "context_length": 32768}, {"id": "other"}])
    r = probe(server)
    assert r.models == ["fake-model", "other"]
    assert (r.tools.value, r.tools.source) == (True, "probe")
    assert (r.vision.value, r.vision.source) == (True, "probe")
    assert (r.context_length.value, r.context_length.source) == (32768, "server")
    assert r.quirks["strict"] is True and r.quirks["max_tokens_field"] == "max_completion_tokens"
    assert r.errors == []
    # Three requests on the wire: the list, the tool call and the image, each bounded.
    tool_req, vision_req = server.requests
    assert tool_req["stream"] is True and tool_req["max_completion_tokens"] == 1024
    assert tool_req["tools"][0]["function"]["name"] == "record_number"
    assert tool_req["tools"][0]["function"]["strict"] is True
    assert "image_url" in json.dumps(vision_req["messages"]) and "tools" not in vision_req


@pytest.mark.parametrize("colour", ["red", "green", "blue", "yellow"])
def test_the_vision_probe_needs_the_right_colour(fake, colour):
    assert probe(fake(), colour=colour).vision.value is True


def test_a_model_without_tool_calling(fake):
    r = probe(fake(refuse_tools=True))
    assert (r.tools.value, r.tools.source) == (False, "probe") and "does not support tools" in r.tools.detail
    r = probe(fake(tool_mode="text"))
    assert r.tools.value is False and "text" in r.tools.detail


def test_a_reply_cut_off_at_the_limit_leaves_tool_calling_unknown(fake):
    r = probe(fake(tool_mode="cutoff"), vision=False)
    assert r.tools.value is None and r.errors[-1]["code"] == "probe.cut_off"


def test_a_model_without_vision(fake):
    r = probe(fake(refuse_images=True))
    assert r.vision.value is False and "refused a request with an image" in r.vision.detail
    r = probe(fake(vision_mode="blind"))
    assert r.vision.value is False and "did not name" in r.vision.detail
    r = probe(fake(), vision=False)
    assert r.vision.value is None


def test_quirks_strict_and_max_tokens(fake):
    server = fake(refuse_strict=True)
    r = probe(server, vision=False)
    assert r.tools.value is True and r.quirks["strict"] is False
    assert "strict" not in server.requests[-1]["tools"][0]["function"]
    server = fake(refuse_max_completion_tokens=True)
    r = probe(server)
    assert r.tools.value is True and r.vision.value is True and r.quirks["max_tokens_field"] == "max_tokens"
    assert server.requests[-1]["max_tokens"] == 1024 and "max_completion_tokens" not in server.requests[-1]


@pytest.mark.parametrize("style,expected", [("whole", "whole"), ("split", "whole"), ("pieces", "pieces")])
def test_streamed_arguments_in_any_shape(fake, style, expected):
    r = probe(fake(args_style=style), vision=False)
    assert r.tools.value is True and r.quirks["streamed_arguments"] == expected


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "think_tags"])
def test_where_the_reasoning_came(fake, field):
    r = probe(fake(reasoning=field), colour="blue")
    assert r.quirks["reasoning_field"] == field and r.vision.value is True


def test_context_length_from_the_server(fake):
    for name in ("context_length", "context_window", "max_context_length", "max_model_len"):
        assert probe(fake(models=[{"id": "fake-model", name: 4096}]), vision=False).context_length.value == 4096
    r = probe(fake(models=None), vision=False)
    assert r.models is None and r.context_length.value is None
    assert r.errors and r.errors[0]["capability"] == "models"
    r = probe(fake(models=[{"id": "fake-model"}], lmstudio=65536), preset="lmstudio", vision=False)
    assert r.context_length.value == 65536


def test_a_server_that_needs_a_key(fake):
    server = fake(require_key=MODEL_KEY, models=[{"id": "fake-model"}])
    r = probe(server, key=Secret(MODEL_KEY))
    assert r.tools.value is True and r.vision.value is True
    assert all(h["authorization"] == f"Bearer {MODEL_KEY}" for h in server.headers)
    SCRUBBER.register(MODEL_KEY)
    r = probe(server, key=Secret("sk-model-TESTONLY-wrong-0123456789abcdef"))
    assert r.tools.value is None and r.vision.value is None
    assert {e["code"] for e in r.errors} == {"probe.key_rejected"}
    assert MODEL_KEY not in json.dumps(r.as_dict())


def test_an_unreachable_server_leaves_everything_unknown():
    r = asyncio.run(probe_openai_compatible("http://127.0.0.1:9/v1", "m", None))
    assert r.tools.value is None and r.vision.value is None
    assert {e["code"] for e in r.errors} <= {"probe.unreachable", "probe.timeout"}


def test_the_probe_png_is_one_colour():
    png = solid_png((1, 2, 3), 4)
    assert png.startswith(b"\x89PNG") and len(png) < 200


# ---- the Claude API through the stubbed transport ---------------------------------------------------


CLAUDE_MODELS = [
    {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "max_input_tokens": 1000000, "max_tokens": 128000,
     "capabilities": None},
    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "max_input_tokens": 1000000, "max_tokens": 128000},
]


def _claude_caps(vision: bool) -> dict:
    support = {"supported": True}
    return {"batch": support, "citations": support, "code_execution": support,
            "context_management": {"supported": True}, "effort": {"supported": True},
            "image_input": {"supported": vision}, "pdf_input": support, "server_tools": {"supported": True},
            "structured_outputs": support, "thinking": {"supported": True, "types": {}}}


def test_the_claude_probe_reads_the_models_api():
    models = [dict(CLAUDE_MODELS[0], capabilities=_claude_caps(True)), dict(CLAUDE_MODELS[1], capabilities=_claude_caps(False))]
    stub = StubTransport({"steps": [], "models": models})
    r = asyncio.run(probe_claude("claude-haiku-5-5", Secret(FAKE_KEY), transport=stub))
    assert r.models == ["claude-haiku-5-5", "claude-opus-5-5"]
    assert (r.tools.value, r.tools.source) == (True, "known")
    assert (r.vision.value, r.vision.source) == (True, "server")
    assert (r.context_length.value, r.context_length.source) == (1000000, "server")
    assert [(q["method"], q["path"]) for q in stub.requests] == [("GET", "/v1/models"), ("GET", "/v1/models/claude-haiku-5-5")]
    assert all(q["has_credential"] for q in stub.requests) and set(stub.hosts) == {"api.anthropic.com"}
    # No message request, no tokens.
    assert not any(q["path"].startswith("/v1/messages") for q in stub.requests)
    r = asyncio.run(probe_claude("claude-opus-5-5", Secret(FAKE_KEY), transport=StubTransport({"steps": [], "models": models})))
    assert r.vision.value is False


def test_the_claude_probe_without_a_key_or_the_model():
    r = asyncio.run(probe_claude("claude-haiku-5-5", None, transport=StubTransport({"steps": []})))
    assert r.errors[0]["code"] == "probe.no_key" and r.tools.value is None
    r = asyncio.run(probe_claude("claude-sonnet-5-5", Secret(FAKE_KEY), transport=StubTransport({"steps": [], "models": CLAUDE_MODELS})))
    assert r.tools.value is None and r.errors[-1]["code"] == "probe.model_unavailable"


def test_the_claude_probe_is_refused_offline():
    r = asyncio.run(probe_claude("claude-haiku-5-5", Secret(FAKE_KEY)))   # no stub: the real API
    assert r.errors[0]["code"] == "agent.offline" and r.models is None


# ---- the cache ----------------------------------------------------------------------------------


def _result(base: str = "http://127.0.0.1:1/v1", model: str = "m", *, tools=True, vision=True, context=None,
            at: float | None = None) -> ProbeResult:
    r = ProbeResult("openai-compatible", base, model, probed_at=at if at is not None else time.time())
    r.tools = Capability(tools, "probe")
    r.vision = Capability(vision, "probe")
    if context:
        r.context_length = Capability(context, "server")
    r.quirks = {"strict": False, "max_tokens_field": "max_tokens", "junk": object()}
    return r


def test_the_cache_keeps_results_and_overrides(tmp_path):
    cache = ProbeCache(tmp_path / "settings")
    cache.put_result(_result(tools=True, vision=False, context=4096))
    cache.set_override("http://127.0.0.1:1/v1", "m", "context_length", 16384)
    assert cache.save()
    path = tmp_path / "settings" / CACHE_FILE
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    again = ProbeCache(tmp_path / "settings")
    entry = again.get("http://127.0.0.1:1/v1", "m")
    known = resolve(entry)
    assert (known.tools.value, known.tools.source) == (True, "probe")
    assert (known.vision.value, known.vision.source) == (False, "probe")
    assert (known.context_length.value, known.context_length.source) == (16384, "override")
    assert known.quirks == {"strict": False, "max_tokens_field": "max_tokens"}
    # An option on the command line wins over the override, which wins over the probe.
    known = resolve(entry, options={"context_length": 2048, "vision": True})
    assert known.context_length.source == "option" and known.vision.value is True
    # "auto" clears the override.
    again.set_override("http://127.0.0.1:1/v1", "m", "context_length", None)
    assert resolve(again.get("http://127.0.0.1:1/v1", "m")).context_length.value == 4096


def test_the_cache_expires_results_but_not_overrides(tmp_path):
    cache = ProbeCache(tmp_path)
    cache.put_result(_result(tools=False, at=time.time() - TTL_S - 60))
    cache.set_override("http://127.0.0.1:1/v1", "m", "vision", False)
    entry = cache.get("http://127.0.0.1:1/v1", "m")
    assert entry.stale()
    known = resolve(entry)
    assert (known.tools.value, known.tools.source) == (True, "default")   # the stale "no" is not relied on
    assert (known.vision.value, known.vision.source) == (False, "override")
    assert known.context_length.value == 8192 and known.context_length.source == "default"
    assert known.quirks == {}


@pytest.mark.parametrize("content", [b"{not json", b"[]", json.dumps({"version": 7, "entries": []}).encode(),
                                     b"\xff\xfe\x00", b"x" * ((1 << 20) + 10)])
def test_a_corrupt_cache_counts_as_empty_and_is_replaced(tmp_path, content):
    (tmp_path / CACHE_FILE).write_bytes(content)
    cache = ProbeCache(tmp_path)
    assert cache.entries() == [] and cache.warnings
    cache.put_result(_result())
    assert cache.save()
    assert len(ProbeCache(tmp_path).entries()) == 1


def test_malformed_entries_are_dropped(tmp_path):
    good = Entry.from_result(_result()).as_dict()
    doc = {"version": 1, "entries": [good, {"base_url": 3}, {"base_url": "http://x/v1", "model": "a" * 300},
                                     dict(good, model="n", result={"tools": {"value": "maybe"}},
                                          overrides={"context_length": -5})]}
    (tmp_path / CACHE_FILE).write_text(json.dumps(doc))
    cache = ProbeCache(tmp_path)
    assert [e.model for e in cache.entries()] == ["m", "n"]
    assert len(cache.warnings) == 2
    n = cache.get("http://x/v1", "n") or cache.entries()[1]
    assert "tools" not in n.result and n.overrides == {}


def test_the_cache_is_bounded(tmp_path):
    cache = ProbeCache(tmp_path)
    for i in range(MAX_ENTRIES + 20):
        entry = cache.put_result(_result(model=f"model-{i}"))
        entry.used_at = 1000.0 + i
    assert cache.save()
    kept = ProbeCache(tmp_path).entries()
    assert len(kept) == MAX_ENTRIES
    assert "model-0" not in {e.model for e in kept} and f"model-{MAX_ENTRIES + 19}" in {e.model for e in kept}
    # The file stays small because it keeps at most MAX_ENTRIES entries and no list of models.
    assert "models" not in (tmp_path / CACHE_FILE).read_text()  # the list of models is not kept


def test_the_cache_refuses_what_could_hold_a_secret(tmp_path):
    cache = ProbeCache(tmp_path)
    for url in ("http://user:pw@127.0.0.1/v1", "http://127.0.0.1/v1?key=abc", "http://127.0.0.1/v1#x", "ftp://x/v1"):
        with pytest.raises(CacheError):
            cache.set_override(url, "m", "tools", True)
    with pytest.raises(CacheError):
        cache.set_override("http://127.0.0.1/v1", "m", "speed", True)
    with pytest.raises(CacheError):
        cache.set_override("http://127.0.0.1/v1", "m", "tools", "yes")
    secret = "sk-model-TESTONLY-secret-in-a-model-id-0123456789"
    SCRUBBER.register(secret)
    cache.put_result(_result(model=secret))
    assert cache.save() is False and "secret" in cache.warnings[-1]
    assert not (tmp_path / CACHE_FILE).exists()


# ---- `caliper-agent probe` ------------------------------------------------------------------------


def run_probe_cli(*args: str, env_extra: dict | None = None) -> tuple[int, dict | None, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CALIPER_MODEL", "OPENAI_"))}
    env.pop("CALIPER_AGENT_TEST_SCRIPT", None)
    env["PYTHONPATH"] = str(AGENT_DIR)
    env["CALIPER_AGENT_OFFLINE"] = "1"
    env.update(env_extra or {})
    run = subprocess.run([sys.executable, "-P", "-m", "caliper_agent", "probe", *args],
                         capture_output=True, text=True, env=env, timeout=120)
    doc = json.loads(run.stdout) if run.stdout.strip() else None
    return run.returncode, doc, run.stderr


def test_the_probe_command_prints_json_and_caches(fake, tmp_path):
    server = fake(models=[{"id": "fake-model", "context_length": 6000}], require_key=MODEL_KEY, reasoning="reasoning")
    cache = tmp_path / "settings"
    code, doc, err = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url + "/",
                                   "--model", "fake-model", "--cache-dir", str(cache),
                                   env_extra={"CALIPER_MODEL_API_KEY": MODEL_KEY})
    assert code == 0, err
    assert doc["base_url"] == server.base_url and doc["models"] == ["fake-model"]
    assert doc["capabilities"]["tools"] == {"value": True, "source": "probe",
                                            "detail": "The model called record_number as asked."}
    assert doc["capabilities"]["context_length"]["value"] == 6000
    assert doc["can_drive_caliper"] is True and doc["from_cache"] is False
    assert doc["quirks"]["reasoning_field"] == "reasoning"
    assert "below 8,192" in " ".join(doc["warnings"])
    on_disk = (cache / CACHE_FILE).read_text()
    assert MODEL_KEY not in on_disk and MODEL_KEY not in json.dumps(doc) and MODEL_KEY not in err
    requests = len(server.requests)
    # A second run answers from the cache, without a request.
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url,
                                 "--model", "fake-model", "--cache-dir", str(cache))
    assert doc["from_cache"] is True and len(server.requests) == requests
    # The person's override: no request, kept in the cache, cleared with auto.
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url,
                                 "--model", "fake-model", "--cache-dir", str(cache), "--set", "vision=no",
                                 "--set", "context_length=12000")
    assert doc["capabilities"]["vision"] == {"value": False, "source": "override"}
    assert doc["overrides"] == {"vision": False, "context_length": 12000} and len(server.requests) == requests
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url,
                                 "--model", "fake-model", "--cache-dir", str(cache), "--set", "vision=auto")
    assert doc["capabilities"]["vision"]["source"] == "probe"
    # --refresh probes again.
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url,
                                 "--model", "fake-model", "--cache-dir", str(cache), "--refresh",
                                 env_extra={"CALIPER_MODEL_API_KEY": MODEL_KEY})
    assert doc["from_cache"] is False and len(server.requests) == requests + 2


def test_the_probe_command_for_a_model_without_tools_and_the_list(fake):
    server = fake(refuse_tools=True, models=[{"id": "a"}, {"id": "b"}])
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url, "--model", "a")
    assert doc["can_drive_caliper"] is False and doc["capabilities"]["tools"]["value"] is False
    code, doc, _ = run_probe_cli("--provider", "openai-compatible", "--base-url", server.base_url, "--model", "a",
                                 "--list-models")
    assert doc["models"] == ["a", "b"] and "capabilities" not in doc


def test_the_probe_command_for_claude_through_the_stub(tmp_path):
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"steps": [], "route": "anthropic",
                                  "models": [dict(CLAUDE_MODELS[0], capabilities=_claude_caps(True))]}))
    code, doc, err = run_probe_cli("--model", "claude-haiku-5-5", "--cache-dir", str(tmp_path / "c"),
                                   env_extra={"ANTHROPIC_API_KEY": FAKE_KEY, "CALIPER_AGENT_TEST_SCRIPT": str(script)})
    assert code == 0, err
    assert doc["base_url"] == "https://api.anthropic.com"
    assert doc["capabilities"]["context_length"] == {"value": 1000000, "source": "server", "detail": "From the Models API."}
    assert doc["capabilities"]["tools"]["source"] == "known" and doc["can_drive_caliper"] is True
    assert FAKE_KEY not in json.dumps(doc) and FAKE_KEY not in err


def test_the_probe_command_refuses_bad_usage(fake):
    for args in (["--provider", "openai-compatible", "--model", "m"],
                 ["--provider", "openai-compatible", "--base-url", "file:///etc", "--model", "m"],
                 ["--provider", "openai-compatible", "--base-url", "http://127.0.0.1/v1?k=1", "--model", "m"],
                 ["--model", "claude-haiku-5-5", "--base-url", "http://127.0.0.1/v1"],
                 ["--model", "m", "--set", "vision=no"],
                 ["--model", "m", "--cache-dir", "/nonexistent-dir-for-test", "--set", "speed=fast"]):
        code, doc, err = run_probe_cli(*args)
        assert code == 2 and doc is None and "caliper-agent probe" in err, args


# ---- the sidecar uses the cache -------------------------------------------------------------------


def _write_cache(folder: Path, base_url: str, model: str, **kwargs) -> None:
    cache = ProbeCache(folder)
    r = _result(base_url, model, **{k: v for k, v in kwargs.items() if k in ("tools", "vision", "context", "at")})
    r.quirks = kwargs.get("quirks", {})
    cache.put_result(r)
    for name, value in kwargs.get("overrides", {}).items():
        cache.set_override(base_url, model, name, value)
    assert cache.save()


def _sidecar(caliper, server: FakeOpenAI, cache: Path, *extra: str) -> AgentProcess:
    return AgentProcess(script=None, api_file=caliper.api_file, key=None,
                        extra_args=["--provider", "openai-compatible", "--base-url", server.base_url,
                                    "--model", "fake-model", "--cache-dir", str(cache), *extra])


def _session_new(agent: AgentProcess, meta: dict | None = None) -> dict:
    agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    params: dict = {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []}
    if meta is not None:
        params["_meta"] = meta
    return agent.request("session/new", params)[0]


def test_a_cached_model_without_tools_refuses_the_session(caliper, fake, tmp_path):
    server = fake()
    _write_cache(tmp_path, server.base_url, "fake-model", tools=False)
    agent = _sidecar(caliper, server, tmp_path)
    try:
        response = _session_new(agent)
        assert response["error"]["data"] == {"code": "agent.no_tools", "model": "fake-model", "capability": "tools",
                                             "source": "probe", "retryable": False}
        assert server.requests == []
    finally:
        agent.close()
    # The person's override wins over the probe.
    _write_cache(tmp_path, server.base_url, "fake-model", tools=False, overrides={"tools": True})
    agent = _sidecar(caliper, server, tmp_path)
    try:
        assert "result" in _session_new(agent)
    finally:
        agent.close()


def test_cached_vision_and_context_reach_the_session_and_the_request(caliper, fake, tmp_path):
    server = fake(replies=[{"text": "Done."}])
    _write_cache(tmp_path, server.base_url, "fake-model", vision=False, context=4096,
                 quirks={"strict": False, "max_tokens_field": "max_tokens"})
    agent = _sidecar(caliper, server, tmp_path)
    try:
        response = _session_new(agent, widget_meta())
        model = response["result"]["_meta"]["caliper"]["model"]
        assert model == {"id": "fake-model", "tools": True, "vision": False, "context_length": 4096,
                         "sources": {"tools": "probe", "vision": "probe", "context_length": "server"}}
        scope = response["result"]["_meta"]["caliper"]["widget"]["scope"]
        sid = response["result"]["sessionId"]
        png = __import__("base64").b64encode(solid_png((1, 2, 3))).decode()
        reply, _ = agent.prompt(sid, [{"type": "text", "text": "Use a linear y axis"},
                                      {"type": "image", "data": png, "mimeType": "image/png"}])
        assert reply["result"]["stopReason"] == "end_turn"
        sent = server.requests[0]
        names = [t["function"]["name"] for t in sent["tools"]]
        assert "panel_render" not in names and "panel_patch" in names and "panel.render" in scope
        assert "image_url" not in json.dumps(sent["messages"]) and "left out" in json.dumps(sent["messages"])
        # The quirks: no strict, max_tokens, and at most half the context as output.
        assert all("strict" not in t["function"] for t in sent["tools"])
        assert sent["max_tokens"] == 2048 and "max_completion_tokens" not in sent
    finally:
        agent.close()


def test_options_win_and_a_stale_cache_is_not_relied_on(caliper, fake, tmp_path):
    server = fake()
    _write_cache(tmp_path, server.base_url, "fake-model", tools=False, at=time.time() - TTL_S - 5)
    agent = _sidecar(caliper, server, tmp_path, "--context-length", "32000", "--model-vision", "no")
    try:
        model = _session_new(agent)["result"]["_meta"]["caliper"]["model"]
        assert model["tools"] is True and model["sources"]["tools"] == "default"
        assert (model["vision"], model["sources"]["vision"]) == (False, "option")
        assert (model["context_length"], model["sources"]["context_length"]) == (32000, "option")
    finally:
        agent.close()
    # Without anything known, an OpenAI-compatible model is assumed to have 8,192 tokens.
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--provider", "openai-compatible", "--base-url", server.base_url,
                                     "--model", "fake-model"])
    try:
        model = _session_new(agent)["result"]["_meta"]["caliper"]["model"]
        assert (model["context_length"], model["sources"]["context_length"]) == (8192, "default")
    finally:
        agent.close()


def test_a_claude_session_reports_the_cached_context_length(run_agent, tmp_path):
    _write_cache(tmp_path / "c", "https://api.anthropic.com", "claude-haiku-5-5", context=1000000)
    agent, _ = run_agent([message(text("Hi."))], route="anthropic",
                         extra_args=["--cache-dir", str(tmp_path / "c")])
    model = _session_new(agent, widget_meta())["result"]["_meta"]["caliper"]["model"]
    assert model == {"id": "claude-haiku-5-5", "tools": True, "vision": True, "context_length": 1000000,
                     "sources": {"tools": "known", "vision": "probe", "context_length": "server"}}
    general = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []})[0]
    assert general["result"]["_meta"]["caliper"]["model"]["context_length"] is None
