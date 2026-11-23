"""The Ollama preset on Ollama's own chat API (ollama_native.py), against the fake server.

Ollama's OpenAI-compatible endpoint ignores a context length, so the server
loads the model at its full context; the Ollama preset sends its chat requests
to `/api/chat` instead, with `options.num_ctx` and `keep_alive` in each. These
tests check that every chat request of a turn and of the probe goes there with
the preset's context length or the person's, that tool calls, thoughts, images
and errors behave as on the OpenAI-compatible endpoint, and that the other
presets stay on `/v1`. Nothing leaves the machine and no key is used.
"""

from __future__ import annotations

import json

import pytest

from conftest import AgentProcess, message_text, statuses, widget_meta
from fake_openai import HYBRID_27B_INFO, FakeOpenAI
from test_probe import run_probe_cli

from caliper_agent.ollama_native import native_chat_url, native_context, native_request
from caliper_agent.presets import PRESETS

OLLAMA_CONTEXT = PRESETS["ollama"].context_length["widget"]
GENERAL_CONTEXT = PRESETS["ollama"].context_length["general"]


@pytest.fixture
def ollama_agent(caliper):
    started: list[tuple[AgentProcess, FakeOpenAI]] = []

    def start(replies: list[dict], *extra: str, preset: str = "ollama", tools: str | None = "catalog.list",
              **flags) -> tuple[AgentProcess, FakeOpenAI]:
        server = FakeOpenAI(replies, **flags)
        # Protocol fixtures need only this command. The full table has a separate context-limit check.
        tool_args = ["--tools", tools] if tools is not None else []
        agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                             extra_args=["--preset", preset, "--base-url", server.base_url, "--model", "qwen-test",
                                         *tool_args, *extra])
        started.append((agent, server))
        return agent, server

    yield start
    for agent, server in started:
        agent.close()
        server.close()


def _call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def test_the_ollama_preset_carries_a_context_length_of_its_own():
    assert PRESETS["ollama"].chat_api == "ollama" and (OLLAMA_CONTEXT, GENERAL_CONTEXT) == (16384, 32768)
    assert all(p.chat_api == "openai" for p in PRESETS.values() if p.id != "ollama")
    assert native_context(PRESETS["ollama"], None, "default") == OLLAMA_CONTEXT
    assert native_context(PRESETS["ollama"], None, "default", "general") == GENERAL_CONTEXT
    assert native_context(PRESETS["ollama"], 262144, "probe") == OLLAMA_CONTEXT   # never the model's maximum
    assert native_context(PRESETS["ollama"], 8192, "override") == 8192            # the person's setting
    assert native_context(PRESETS["lmstudio"], 8192, "override") is None
    assert native_chat_url("http://localhost:11434/v1") == "http://localhost:11434/api/chat"


def test_a_chat_request_becomes_a_native_one():
    body = {"model": "m", "stream": True, "max_completion_tokens": 900, "temperature": 0.2,
            "reasoning_effort": "low", "stream_options": {"include_usage": True}, "tool_choice": "auto",
            "tools": [{"type": "function", "function": {"name": "panel_patch", "description": "d",
                                                        "parameters": {"type": "object"}, "strict": True}}],
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "Look"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]},
                {"role": "assistant", "content": None, "reasoning": "Thinking.",
                 "tool_calls": [{"id": "call_9", "type": "function",
                                 "function": {"name": "panel_patch", "arguments": "{\"id\": \"curves\"}"}}]},
                {"role": "tool", "tool_call_id": "call_9", "content": "ok"}]}
    n = native_request(body, num_ctx=16384, keep_alive="60s")
    assert n["options"] == {"num_ctx": 16384, "num_predict": 900, "temperature": 0.2}
    assert n["keep_alive"] == "60s" and n["stream"] is True and n["think"] == "low"
    assert n["tools"] == [{"type": "function", "function": {"name": "panel_patch", "description": "d",
                                                            "parameters": {"type": "object"}}}]
    system, user, assistant, tool = n["messages"]
    assert system == {"role": "system", "content": "Be brief."}
    assert user == {"role": "user", "content": "Look", "images": ["QUJD"]}
    assert assistant["thinking"] == "Thinking."
    assert assistant["tool_calls"] == [{"id": "call_9", "function": {"name": "panel_patch",
                                                                     "arguments": {"id": "curves"}}}]
    assert tool == {"role": "tool", "content": "ok", "tool_call_id": "call_9", "tool_name": "panel_patch"}
    assert "tool_choice" not in n and "stream_options" not in n


@pytest.mark.parametrize("ids", [True, False])
def test_a_turn_runs_on_the_native_api_with_the_context_length(ollama_agent, caliper, ids):
    agent, server = ollama_agent([{"tool_calls": [_call("call_1", "catalog_list", '{"kind": "table"}')],
                                   "thought": "The person wants the tables."},
                                  {"text": "There is one table."}], native_ids=ids, reasoning="reasoning")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "end_turn"
    assert message_text(updates) == "There is one table."
    assert [c["origin"] for c in caliper.commands("catalog.list")] == ["agent"]
    # Every chat request went to /api/chat, none to the OpenAI-compatible endpoint.
    assert "/v1/chat/completions" not in server.paths and server.paths.count("/api/chat") == 2
    first, second = server.native_requests
    for request in (first, second):   # a general session: the general role's context length
        assert request["options"]["num_ctx"] == GENERAL_CONTEXT and request["keep_alive"] == "60s"
        assert request["stream"] is True and request["model"] == "qwen-test"
    assert "catalog_list" in [t["function"]["name"] for t in first["tools"]]
    call = next(m for m in second["messages"] if m["role"] == "assistant")["tool_calls"][0]
    assert call["function"] == {"name": "catalog_list", "arguments": {"kind": "table"}}
    result = next(m for m in second["messages"] if m["role"] == "tool")
    assert result["tool_call_id"] == call["id"] and result["tool_name"] == "catalog_list"
    assert "run42_metrics" in result["content"]
    thoughts = "".join(u["content"]["text"] for u in updates
                       if u.get("sessionUpdate") == "agent_thought_chunk" and "_meta" not in u)
    assert "The person wants the tables." in thoughts
    usage = response["result"]["_meta"]["caliper"]["usage"]
    assert usage["requests"] == 2 and usage["input_tokens"] == 22 and usage["output_tokens"] == 6


def test_several_calls_in_one_native_reply_run_in_order(ollama_agent, caliper):
    agent, server = ollama_agent([{"tool_calls": [_call("call_a", "catalog_list", '{"kind": "table"}'),
                                                  _call("call_b", "catalog_list", '{"kind": "tensor"}')]},
                                  {"text": "One table and one tensor."}], native_ids=False)
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "end_turn"
    assert [c["args"] for c in caliper.commands("catalog.list")] == [{"kind": "table"}, {"kind": "tensor"}]
    second = server.native_requests[1]
    calls = next(m for m in second["messages"] if m["role"] == "assistant")["tool_calls"]
    assert [c["function"]["arguments"] for c in calls] == [{"kind": "table"}, {"kind": "tensor"}]
    assert len({c["id"] for c in calls}) == 2   # an id made for each call Ollama sent without one
    results = [m for m in second["messages"] if m["role"] == "tool"]
    assert [r["tool_call_id"] for r in results] == [c["id"] for c in calls]


def test_each_role_has_its_own_context_length(ollama_agent):
    agent, server = ollama_agent([{"text": "A linear axis."}, {"text": "Hello."}])
    widget = agent.start_session(meta=widget_meta())
    response, _ = agent.prompt(widget, "Use a linear y axis")
    assert response["result"]["stopReason"] == "end_turn"
    general = agent.start_session()
    response, _ = agent.prompt(general, "Say hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert [r["options"]["num_ctx"] for r in server.native_requests] == [OLLAMA_CONTEXT, GENERAL_CONTEXT]
    # At most half of it is asked for as output, within the widget agent's own budget.
    assert server.native_requests[0]["options"]["num_predict"] <= OLLAMA_CONTEXT // 2
    assert server.native_requests[1]["options"]["num_predict"] == GENERAL_CONTEXT // 2


def test_a_request_that_would_not_fit_is_not_sent(ollama_agent, caliper):
    # The first reply's prompt took 15,900 of 16,384 tokens: the tool's result would
    # leave no room for the reply, so the second request is never sent.
    agent, server = ollama_agent([{"tool_calls": [_call("call_1", "catalog_list", '{"kind": "table"}')]},
                                  {"text": "never"}], "--context-length", "16384", prompt_tokens=[15900])
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "max_tokens"   # a limit, not a finished turn
    assert len(server.native_requests) == 1
    assert [c["origin"] for c in caliper.commands("catalog.list")] == ["agent"]   # what ran stays
    text = message_text(updates)
    assert "no longer fits the model's context of 16,384 tokens" in text
    assert "Start a new chat, or raise the context length" in text and "--context-length" in text
    full = [s for s in statuses(updates) if s["status"] == "context_full"]
    assert full and full[0]["context_length"] == 16384
    # The session stays usable, and the next prompt is checked the same way.
    response, updates = agent.prompt(sid, "And now?")
    assert len(server.native_requests) == 1 and "no longer fits" in message_text(updates)


def test_a_widget_request_that_would_not_fit_names_the_settings(ollama_agent):
    agent, server = ollama_agent([{"text": "never"}], "--context-length", "1024", tools=None)
    sid = agent.start_session(meta=widget_meta())
    response, updates = agent.prompt(sid, "Use a linear y axis")
    assert response["result"]["stopReason"] == "max_tokens"
    assert response["result"]["_meta"]["caliper"]["widget"]["limit"] is True   # ⌘K shows LIMIT
    assert server.native_requests == []   # the tools alone take more than 1,024 tokens
    text = message_text(updates)
    assert "context of 1,024 tokens" in text and "Close the prompt and ask again" in text
    assert "widget agent's settings" in text


def test_the_persons_context_length_replaces_the_presets(ollama_agent):
    agent, server = ollama_agent([{"text": "Hello."}], "--context-length", "20000")
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Say hello")
    assert response["result"]["stopReason"] == "end_turn"
    request = server.native_requests[0]
    assert request["options"]["num_ctx"] == 20000
    # At most half of the context length is asked for as output.
    assert request["options"]["num_predict"] <= 10000   # and no more than the first prompt leaves of 20,000

    # A general session's first request with the full table does not fit 8,192 tokens: nothing is sent.
    agent, server = ollama_agent([{"text": "never"}], "--context-length", "8192", tools=None)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Say hello")
    assert server.native_requests == [] and "context of 8,192 tokens" in message_text(updates)


def test_the_full_command_table_is_refused_before_the_default_general_request(ollama_agent):
    agent, server = ollama_agent([{"text": "never"}], tools=None)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Say hello")
    assert response["result"]["stopReason"] == "max_tokens"
    assert server.native_requests == []
    [full] = [s for s in statuses(updates) if s["status"] == "context_full"]
    assert full["context_length"] == GENERAL_CONTEXT
    assert full["estimate"] > GENERAL_CONTEXT


def test_a_refusal_of_tools_and_a_cut_off_reply_keep_their_meaning(ollama_agent):
    agent, server = ollama_agent([{"text": "never"}], refuse_tools=True)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Build a dashboard")
    assert response["result"]["stopReason"] == "end_turn"
    assert "cannot drive Caliper" in message_text(updates)
    assert "no_tools" in [s["status"] for s in statuses(updates)]

    agent, server = ollama_agent([{"text": "Let me think about", "finish": "length"}])
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Say hello")
    assert response["result"]["stopReason"] == "max_tokens"


def test_the_server_may_neither_cut_the_prompt_nor_shift_the_reply():
    n = native_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, num_ctx=16384, keep_alive="60s")
    # Ollama 0.40 cuts a prompt longer than num_ctx from its front unless truncate is false, and then refuses it.
    assert n["truncate"] is False and n["shift"] is False


def test_the_reply_asks_for_no_more_than_the_context_leaves(ollama_agent, caliper):
    # The first reply's prompt took 13,000 of 16,384 tokens: the next request may ask for what is left, not 8,192.
    agent, server = ollama_agent([{"tool_calls": [_call("call_1", "catalog_list", '{"kind": "table"}')]},
                                  {"text": "One table."}], "--context-length", "16384", prompt_tokens=[13000])
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "end_turn"
    first, second = server.native_requests
    assert first["options"]["num_predict"] <= 8192   # at most half the context, and what the first prompt leaves
    assert 13000 + second["options"]["num_predict"] <= 16384
    assert second["options"]["num_predict"] >= 2048   # the least reply worth asking for


def test_a_reply_cut_at_the_end_of_the_context_is_a_limit_even_with_a_tool_call(ollama_agent, caliper):
    agent, server = ollama_agent([{"tool_calls": [_call("call_1", "catalog_list", '{"kind": "table"}')],
                                   "finish": "length"}])
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "max_tokens"
    assert len(server.native_requests) == 1
    assert caliper.commands("catalog.list") == []   # no call of a cut-off reply runs


def test_a_prompt_the_server_refuses_as_too_long_ends_the_turn_as_a_limit(ollama_agent):
    agent, server = ollama_agent([{"text": "never"}], native_error=(
        400, "request (20512 tokens) exceeds the available context size (16384 tokens), try increasing it"))
    sid = agent.start_session(meta=widget_meta())
    response, updates = agent.prompt(sid, "Use a linear y axis")
    assert response["result"]["stopReason"] == "max_tokens"
    assert response["result"]["_meta"]["caliper"]["widget"]["limit"] is True
    assert "no longer fits the model's context of 16,384 tokens" in message_text(updates)
    [full] = [s for s in statuses(updates) if s["status"] == "context_full"]
    assert full["estimate"] == 20512


def test_lm_studio_stays_on_the_openai_compatible_endpoint(ollama_agent):
    agent, server = ollama_agent([{"text": "Hello."}], preset="lmstudio")
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Say hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert server.paths == ["/v1/chat/completions"] and server.native_requests == []
    assert "num_ctx" not in json.dumps(server.requests)


def test_the_probe_loads_the_model_with_the_presets_context_length(tmp_path):
    server = FakeOpenAI(models=[{"id": "qwen-test"}], reasoning="reasoning")
    try:
        cache = tmp_path / "settings"
        code, doc, err = run_probe_cli("--preset", "ollama", "--base-url", server.base_url, "--model", "qwen-test",
                                       "--cache-dir", str(cache))
        assert code == 0, err
        assert doc["capabilities"]["tools"]["value"] is True and doc["capabilities"]["vision"]["value"] is True
        assert doc["capabilities"]["context_length"] == {
            "value": OLLAMA_CONTEXT, "source": "preset", "detail": "The context length the preset loads the model with."}
        assert doc["quirks"]["reasoning_field"] == "reasoning"
        # Both test calls went to /api/chat with the context length; the list of models stays on /v1.
        assert "/v1/models" in server.paths and "/v1/chat/completions" not in server.paths
        assert [r["options"]["num_ctx"] for r in server.native_requests] == [OLLAMA_CONTEXT, OLLAMA_CONTEXT]
        assert all(r["keep_alive"] == "60s" for r in server.native_requests)
        assert server.native_requests[1]["messages"][0]["images"]   # the vision call's picture

        # The person's context length, kept as an override, is what the next probe loads.
        code, doc, err = run_probe_cli("--preset", "ollama", "--base-url", server.base_url, "--model", "qwen-test",
                                       "--cache-dir", str(cache), "--set", "context_length=8192", "--refresh")
        assert code == 0, err
        assert doc["capabilities"]["context_length"]["value"] == 8192
        assert server.native_requests[-1]["options"]["num_ctx"] == 8192
    finally:
        server.close()




def test_the_probe_reads_how_much_cache_a_token_takes(tmp_path):
    server = FakeOpenAI(models=[{"id": "qwen-test"}], show=HYBRID_27B_INFO)
    try:
        code, doc, err = run_probe_cli("--preset", "ollama", "--base-url", server.base_url, "--model", "qwen-test",
                                       "--cache-dir", str(tmp_path))
        assert code == 0, err
        # 17 attention layers x 4 key and value heads x (256 + 256) x 2 bytes: 68 KiB a token.
        assert doc["quirks"]["kv_bytes_per_token"] == 17 * 4 * 512 * 2
        assert doc["capabilities"]["context_length"]["value"] == OLLAMA_CONTEXT   # the preset's, not 262,144
        assert "/api/show" in server.paths

        # A Custom address of an Ollama server: the model's metadata gives the most it can take, not what the server
        # runs it with, so it is not the context length; without the server's word the length stays unknown.
        code, doc, err = run_probe_cli("--preset", "custom", "--base-url", server.base_url, "--model", "qwen-test",
                                       "--cache-dir", str(tmp_path / "custom"))
        assert code == 0, err
        assert doc["probe"]["context_length"]["value"] is None
        assert doc["capabilities"]["context_length"]["source"] == "default"
        assert doc["quirks"]["kv_bytes_per_token"] == 17 * 4 * 512 * 2
    finally:
        server.close()
    # With the model loaded, Ollama's /api/ps states the context in force, which the probe takes.
    server = FakeOpenAI(models=[{"id": "qwen-test"}], show=HYBRID_27B_INFO, ps_context=32768)
    try:
        code, doc, err = run_probe_cli("--preset", "custom", "--base-url", server.base_url, "--model", "qwen-test",
                                       "--cache-dir", str(tmp_path / "custom-ps"))
        assert code == 0, err
        assert doc["capabilities"]["context_length"] == {
            "value": 32768, "source": "server", "detail": "The context the server runs the model with (Ollama's /api/ps)."}
        assert server.paths.index("/api/ps") > server.paths.index("/v1/chat/completions")   # after the test calls
    finally:
        server.close()
