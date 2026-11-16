"""The turn loop through the real stdio protocol.

Each test starts the sidecar as a child process with a fake key, a script of
model replies and a fake Caliper API, then checks the ACP messages the sidecar
sent, the requests the model saw and the calls that reached Caliper API.

Most tests use the "function" route: Pydantic AI's FunctionModel answers, and
the record holds the messages as Pydantic AI hands them to the model. The tests
that must see the Claude API's wire format (the request body, the history as
sent, the cache prefix, fallbacks, strict mode, the key and the host) use the
"anthropic" route: the real Anthropic model behind a stubbed HTTP transport.
"""

from __future__ import annotations

import base64
import json
import threading
import time

from pydantic_ai.models._anthropic_schema_limits import exceeded_limit

from caliper_agent.loop import REFUSAL_TEXT
from conftest import (general_tool_count, FAKE_KEY, allow, deny, message, message_text, prompts, read_requests, statuses, text,
                      thinking, tool_results, tool_use)

CACHE_1 = {"input_tokens": 40, "output_tokens": 12, "cache_creation_input_tokens": 9000, "cache_read_input_tokens": 0}
CACHE_2 = {"input_tokens": 25, "output_tokens": 8, "cache_creation_input_tokens": 60, "cache_read_input_tokens": 9000}
OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


# ---- the Claude API wire format (anthropic route) ---------------------------------------------


def test_plain_turn_streams_and_sends_the_documented_request(run_agent):
    agent, record = run_agent([message(thinking("The person says hello."), text("Hello. What shall we build?"))],
                              route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert message_text(updates) == "Hello. What shall we build?"
    thoughts = [u for u in updates if u["sessionUpdate"] == "agent_thought_chunk" and "_meta" not in u]
    assert "".join(t["content"]["text"] for t in thoughts) == "The person says hello."

    [req] = read_requests(record)
    assert req["host"] == "api.anthropic.com"
    assert req["path"] == "/v1/messages"
    assert req["has_credential"]
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["stream"] is True
    assert body["max_tokens"] == 64000
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["tool_choice"] == {"type": "auto"}
    assert body["cache_control"]["type"] == "ephemeral"
    assert body["system"][-1]["cache_control"]["type"] == "ephemeral"
    assert body["context_management"] == {"edits": [{"type": "compact_20260112"}]}
    betas = req["headers"]["anthropic-beta"].split(",")
    assert "compact-2026-01-12" in betas
    # Server-side refusal fallbacks are off: neither the setting nor its beta is sent.
    assert "fallbacks" not in body and not any(b.startswith("server-side-fallback") for b in betas)
    for forbidden in ("temperature", "top_p", "top_k", "thread"):
        assert forbidden not in body
    # Tools: one per command, named with _ for ., eagerly streamed, and strict
    # within the Claude API's limits for one request.
    names = [t["name"] for t in body["tools"]]
    assert "panel_create" in names and "capture_frame" in names and len(names) == len(set(names)) == general_tool_count()
    for tool in body["tools"]:
        assert tool["eager_input_streaming"] is True
    strict = [t for t in body["tools"] if t.get("strict")]
    assert 0 < len(strict) <= 20
    assert exceeded_limit([t["input_schema"] for t in strict], None) is None
    # The conversation sent: one user message with the prompt, and nothing before the system prompt is volatile.
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


def test_multi_tool_turn_runs_calls_in_order_and_returns_results_together(run_agent, caliper):
    caliper.handlers["catalog.describe"] = lambda a: {"ok": True, "result": {"entry": {"id": a["id"], "kind": "table"}}}
    steps = [
        message(thinking("Look first."), text("I will look at the catalog."),
                tool_use("toolu_1", "catalog_list", {"kind": "table"}),
                tool_use("toolu_2", "app_info", {}), stop="tool_use"),
        message(tool_use("toolu_3", "catalog_describe", {"id": "table:run42_metrics"}), stop="tool_use"),
        message(text("There is one table, run42_metrics.")),
    ]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "end_turn"
    assert [c["path"] for c in caliper.calls] == ["/v0/commands/catalog.list", "/v0/info",
                                                  "/v0/commands/catalog.describe"]
    first = caliper.commands("catalog.list")[0]
    assert first["origin"] == "agent" and first["group"].startswith(sid)
    assert caliper.commands("catalog.describe")[0]["args"] == {"id": "table:run42_metrics"}

    completed = [u for u in updates if u["sessionUpdate"] == "tool_call_update" and u.get("status") == "completed"]
    assert [u["toolCallId"] for u in completed] == ["toolu_1", "toolu_2", "toolu_3"]
    started = [u for u in updates if u["sessionUpdate"] == "tool_call"]
    assert [u["name"] for u in started] == ["catalog_list", "app_info", "catalog_describe"]

    requests = read_requests(record)
    assert len(requests) == 3
    second = requests[1]["body"]["messages"]
    # The assistant turn goes back whole (thinking with its signature included),
    # and both results come back in one user message, in order.
    assistant = second[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "Look first.", "signature": "sig-1"}
    assert [b["type"] for b in assistant["content"]] == ["thinking", "text", "tool_use", "tool_use"]
    results = second[2]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["toolu_1", "toolu_2"]
    assert "run42_metrics" in results["content"][0]["content"][0]["text"]
    # The history is append-only: the third request starts with the second's messages.
    third = requests[2]["body"]["messages"]
    assert third[: len(second)] == second


def test_capture_comes_back_as_an_image(run_agent, caliper):
    steps = [message(tool_use("toolu_cap", "capture_frame", {}), tool_use("toolu_r", "panel_render", {"id": "heat"}),
                     stop="tool_use"),
             message(text("The heatmap is drawn."))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Show me")
    assert response["result"]["stopReason"] == "end_turn"
    results = read_requests(record)[1]["body"]["messages"][-1]["content"]
    for result, size in zip(results, ((4, 3), (8, 5))):
        assert not result.get("is_error")
        textblock, image = result["content"]
        assert json.loads(textblock["text"])["width"] == size[0]
        assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
        assert base64.b64decode(image["source"]["data"]).startswith(b"\x89PNG")
    shown = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_cap"
                 and u.get("status") == "completed")
    assert any(c["type"] == "content" and c["content"]["type"] == "image" for c in shown["content"])
    assert caliper.calls[0]["path"] == "/v0/capture.png"


def test_cache_fields_across_two_turns(run_agent, tmp_path):
    steps = [message(text("First."), usage=CACHE_1), message(text("Second."), usage=CACHE_2)]
    usage_log = tmp_path / "usage.jsonl"
    agent, record = run_agent(steps, route="anthropic", extra_args=["--usage-log", str(usage_log)])
    sid = agent.start_session()
    first, _ = agent.prompt(sid, "One")
    second, _ = agent.prompt(sid, "Two")
    u1 = first["result"]["_meta"]["caliper"]["usage"]
    u2 = second["result"]["_meta"]["caliper"]["usage"]
    assert u1["cache_creation_input_tokens"] == 9000 and u1["cache_read_input_tokens"] == 0
    assert u2["cache_read_input_tokens"] == 9000 and u2["cache_read_input_tokens"] > 0
    assert u2["per_request"][0]["model"] == "claude-opus-5-5"
    logged = [json.loads(line) for line in usage_log.read_text().splitlines()]
    assert [line["cache_read_input_tokens"] for line in logged] == [0, 9000]
    assert set(logged[0]) == {"session", "turn", "model", "input_tokens", "output_tokens",
                              "cache_creation_input_tokens", "cache_read_input_tokens"}
    # The cached prefix is byte-identical between the turns: tools, then system.
    r1, r2 = (r["body"] for r in read_requests(record))
    assert json.dumps(r1["tools"]) == json.dumps(r2["tools"])
    assert r1["system"] == r2["system"]
    assert r2["messages"][: len(r1["messages"])] == r1["messages"]


def test_model_and_effort_are_config_options(run_agent):
    agent, record = run_agent([message(text("Fast.")), message(text("Fable."))], route="anthropic")
    sid = agent.start_session()
    response, _ = agent.request("session/set_config_option",
                                {"sessionId": sid, "configId": "model", "value": "claude-haiku-5-5"})
    options = {o["id"]: o for o in response["result"]["configOptions"]}
    assert options["model"]["currentValue"] == "claude-haiku-5-5"
    assert [o["value"] for o in options["model"]["options"]] == [
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"]
    agent.request("session/set_config_option", {"sessionId": sid, "configId": "effort", "value": "low"})
    agent.prompt(sid, "Quick")
    agent.request("session/set_config_option", {"sessionId": sid, "configId": "model", "value": "claude-fable-5-2"})
    agent.prompt(sid, "Deep")
    haiku, fable = (r for r in read_requests(record))
    assert haiku["body"]["model"] == "claude-haiku-5-5" and haiku["body"]["output_config"] == {"effort": "low"}
    assert fable["body"]["model"] == "claude-fable-5-2" and fable["body"]["output_config"] == {"effort": "low"}
    for request in (haiku, fable):
        assert "fallbacks" not in request["body"]
        assert "server-side-fallback" not in request["headers"].get("anthropic-beta", "")
    bad, _ = agent.request("session/set_config_option", {"sessionId": sid, "configId": "model", "value": "gpt-x"})
    assert bad["error"]["code"] == -32602


def test_compaction_is_shown_and_sent_back(run_agent):
    compaction = {"type": "compaction", "content": "Summary of the earlier conversation."}
    steps = [message(compaction, text("Continuing.")), message(text("Next."))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Long session")
    assert response["result"]["stopReason"] == "end_turn"
    assert "compaction" in [s["status"] for s in statuses(updates)]
    agent.prompt(sid, "Again")
    # The compaction block goes back, and Pydantic AI sends nothing before it,
    # since the summary replaces the earlier conversation.
    sent = read_requests(record)[1]["body"]["messages"]
    assert sent[0]["role"] == "assistant"
    assert sent[0]["content"][0] == {"type": "compaction", "content": "Summary of the earlier conversation."}
    assert sent[-1] == {"role": "user", "content": [{"type": "text", "text": "Again"}]}


def test_refusal_and_cut_offs_on_the_wire(run_agent, caliper):
    steps = [message(text("I can"), stop="refusal",
                     stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."}),
             message(text("Part"), tool_use("t_cut", "panel_create", raw='{"id": "he'), stop="max_tokens"),
             message(text("Full"), stop="model_context_window_exceeded"),
             message(text("Fine."))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Declined prompt")
    assert response["result"]["stopReason"] == "refusal"
    [status] = [s for s in statuses(updates) if s["status"] == "refusal"]
    # The provider's category and explanation are kept in `_meta`, out of the text.
    assert status["category"] == "cyber" and status["explanation"] == "Declined."
    assert status["text"] == REFUSAL_TEXT
    response, _ = agent.prompt(sid, "Cut")
    assert response["result"]["stopReason"] == "max_tokens"
    response, _ = agent.prompt(sid, "Full")
    assert response["result"]["stopReason"] == "max_tokens"
    agent.prompt(sid, "Last")
    assert caliper.commands() == []
    history = read_requests(record)[3]["body"]["messages"]
    # The refused prompt is gone; the cut-off call has an error result.
    texts = json.dumps(history)
    assert "Declined prompt" not in texts and "Cut" in texts
    results = [b for m in history if m["role"] == "user" for b in m["content"] if b.get("type") == "tool_result"]
    assert [(r["tool_use_id"], r["is_error"]) for r in results] == [("t_cut", True)]


def test_a_rate_limit_from_the_wire_reaches_the_retry_hook(run_agent):
    steps = [{"error": {"status": 429, "headers": {"retry-after": "0"},
                        "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "Slow down."}}}},
             {"stream_error": {"after": 1, "body": OVERLOADED}},
             message(text("Done after waiting."))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    retries = [s for s in statuses(updates) if s["status"] == "retry"]
    assert [(r["reason"], r["attempt"], r["max_attempts"]) for r in retries] == [("rate_limit", 2, 5), ("server", 3, 5)]
    assert retries[0]["http_status"] == 429 and retries[0]["delay_s"] == 0
    # One HTTP request per attempt: the SDK's own retries are off.
    assert len(read_requests(record)) == 3


def test_a_stream_that_breaks_after_the_pane_saw_it_is_not_retried(run_agent):
    steps = [{"stream_error": {"after": 4, "body": OVERLOADED}}, message(text("never"))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert message_text(updates) == "partial"
    assert response["error"]["data"]["code"] == "agent.stream_interrupted"
    assert "Send the prompt again" in response["error"]["message"]
    assert len(read_requests(record)) == 1


def test_the_key_never_appears_in_any_output(run_agent, caliper):
    leak = f"invalid x-api-key: {FAKE_KEY}"
    caliper.handlers["catalog.list"] = lambda a: {"ok": False, "error": {"code": "x", "message": FAKE_KEY, "where": ""}}
    steps = [message(tool_use("t", "catalog_list", {}), stop="tool_use"),
             message(text(f"The tool said {FAKE_KEY}")),
             {"error": {"status": 401, "body": {"type": "error", "error": {"type": "authentication_error",
                                                                           "message": leak}}}}]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Try")
    assert response["result"]["stopReason"] == "end_turn"
    response, _ = agent.prompt(sid, "Again")
    assert response["error"]["data"]["code"] == "agent.key_rejected"
    assert response["error"]["code"] == -32000
    agent.close()
    output = agent.all_output()
    assert FAKE_KEY not in output
    assert "[redacted]" in output
    # The key reached the transport as a credential, and was never logged by it.
    assert all(r["has_credential"] for r in read_requests(record))
    # Tool output is scrubbed before it reaches the model, and the transport's
    # record of the requests holds no key either.
    assert FAKE_KEY not in record.read_text()


def test_only_the_claude_api_host_is_contacted(run_agent):
    agent, record = run_agent([message(text("One.")), message(text("Two."))], route="anthropic")
    sid = agent.start_session()
    agent.prompt(sid, "a")
    agent.prompt(sid, "b")
    assert {r["host"] for r in read_requests(record)} == {"api.anthropic.com"}


def test_strict_mode_is_dropped_when_the_api_refuses_it_on_the_first_request(run_agent):
    steps = [{"error": {"status": 400, "body": {"type": "error", "error": {
        "type": "invalid_request_error", "message": "Too many strict tools in one request."}}}},
             message(text("Fine without strict."))]
    agent, record = run_agent(steps, route="anthropic")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    first, second = read_requests(record)
    assert any(t.get("strict") for t in first["body"]["tools"])
    assert not any(t.get("strict") for t in second["body"]["tools"])
    assert "strict_off" in [s["status"] for s in statuses(updates)]


# ---- the loop (function route) ------------------------------------------------------------------


def test_input_that_fails_its_schema_is_not_run(run_agent, caliper):
    steps = [
        message(tool_use("toolu_bad", "panel_create", {"id": "Bad Id!", "primitive": "heatmap", "bind": {"values": "entropy"}}), stop="tool_use"),
        message(tool_use("toolu_trunc", "catalog_describe", raw='{"id": "table:run'), stop="tool_use"),
        message(text("I will fix the id.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Make a heatmap")
    assert response["result"]["stopReason"] == "end_turn"
    assert caliper.commands("panel.create", dry=None) == [] and caliper.commands("catalog.describe") == []
    failed = {u["toolCallId"] for u in updates if u["sessionUpdate"] == "tool_call_update" and u.get("status") == "failed"}
    assert failed == {"toolu_bad", "toolu_trunc"}
    requests = read_requests(record)
    bad = tool_results(requests[1]["messages"])["toolu_bad"]
    assert bad["outcome"] == "failed"
    payload = json.loads(bad["content"])
    assert "INVALID_JSON" in payload and "args.id" in payload["problem"]
    truncated = tool_results(requests[2]["messages"])["toolu_trunc"]
    assert truncated["outcome"] == "failed" and "INVALID_JSON" in json.loads(truncated["content"])


def test_an_unknown_tool_is_an_error_result(run_agent, caliper):
    steps = [message(tool_use("toolu_x", "no_such_tool", {}), stop="tool_use"), message(text("Sorry."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Go")
    assert response["result"]["stopReason"] == "end_turn"
    assert any(u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_x" and u["status"] == "failed"
               for u in updates)
    result = tool_results(read_requests(record)[1]["messages"])["toolu_x"]
    assert result["outcome"] == "retried" and "Unknown tool name" in str(result["content"])


def test_max_tokens_runs_no_tool_and_keeps_the_history_valid(run_agent, caliper):
    steps = [
        message(text("Creating the panel"), tool_use("toolu_cut", "panel_create", {"id": "heat"}), stop="max_tokens"),
        message(text("Ready.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Make a heatmap")
    assert response["result"]["stopReason"] == "max_tokens"
    assert caliper.commands(dry=None) == []
    assert any(u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_cut" and u["status"] == "failed"
               for u in updates)
    response, _ = agent.prompt(sid, "Go on")
    assert response["result"]["stopReason"] == "end_turn"
    history = read_requests(record)[1]["messages"]
    assert [m["kind"] for m in history][:2] == ["request", "response"]
    cut = tool_results(history)["toolu_cut"]
    assert cut["outcome"] == "failed" and "cut off" in cut["content"]
    assert prompts(history) == ["Make a heatmap", "Go on"]


def test_refusal_is_surfaced_and_the_turn_is_taken_back(run_agent, caliper):
    steps = [
        message(text("I can"), stop="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."}),
        message(text("Here is the plan.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Something the classifier declines")
    assert response["result"]["stopReason"] == "refusal"
    [status] = [s for s in statuses(updates) if s["status"] == "refusal"]
    assert status["category"] == "cyber" and status["turn_removed"] is True
    assert status["text"] == REFUSAL_TEXT and "cyber" not in status["text"]
    response, _ = agent.prompt(sid, "Plan a loss dashboard")
    assert response["result"]["stopReason"] == "end_turn"
    # The declined prompt and its partial reply are not in the next request.
    history = read_requests(record)[1]["messages"]
    assert prompts(history) == ["Plan a loss dashboard"] and len(history) == 1


def test_the_refusal_text_is_neutral_and_offers_three_ways_forward():
    # A refusal can be a false positive, so the text blames no one and names
    # the three ways forward: rephrase, retry, or another model or provider.
    text = REFUSAL_TEXT.lower()
    assert text.startswith("the model declined this request.")
    assert "rephrase" in text and "try it again" in text
    assert "another model or provider" in text and "local model" in text
    for word in ("policy", "violat", "harm", "unsafe", "not allowed", "inappropriate", "you asked", "your request"):
        assert word not in text
    assert len(REFUSAL_TEXT) < 160


def test_a_refusal_later_in_a_turn_removes_the_whole_turn(run_agent, caliper):
    # ACP's refusal: the prompt and everything after it are left out of the next request.
    steps = [message(text("First.")),
             message(text("Looking."), tool_use("t1", "catalog_list", {}), stop="tool_use"),
             message(text("I won't"), stop="refusal"),
             message(text("Fine."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    agent.prompt(sid, "One")
    response, updates = agent.prompt(sid, "Two")
    assert response["result"]["stopReason"] == "refusal"
    [status] = [s for s in statuses(updates) if s["status"] == "refusal"]
    assert "stay" in status["text"] and status["turn_removed"] is True
    assert len(caliper.commands("catalog.list")) == 1  # the tool round did run
    agent.prompt(sid, "Three")
    history = read_requests(record)[3]["messages"]
    assert prompts(history) == ["One", "Three"]
    assert [m["kind"] for m in history] == ["request", "response", "request"]


def test_rate_limit_then_success_reports_the_retry(run_agent):
    steps = [
        {"error": {"status": 429, "headers": {"retry-after": "0"},
                   "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "Slow down."}}}},
        {"error": {"status": 529, "body": OVERLOADED}},
        message(text("Done after waiting.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    retries = [s for s in statuses(updates) if s["status"] == "retry"]
    assert [r["http_status"] for r in retries] == [429, 529]
    assert [r["attempt"] for r in retries] == [2, 3]
    assert retries[0]["reason"] == "rate_limit" and "rate limited" in retries[0]["text"]
    assert len(read_requests(record)) == 3
    assert message_text(updates) == "Done after waiting."


def test_connection_errors_are_retried_then_reported(run_agent):
    steps = [{"connect_error": "connection refused"}, message(text("Back online."))]
    agent, _ = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert [s["reason"] for s in statuses(updates) if s["status"] == "retry"] == ["connection"]

    steps = [{"connect_error": "connection refused"}] * 5
    agent, _ = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    error = response["error"]
    assert error["data"]["code"] == "agent.retries_exhausted" and error["data"]["retryable"] is True
    assert "could not be reached" in error["message"] and "5 times" in error["message"]
    assert len([s for s in statuses(updates) if s["status"] == "retry"]) == 4


def test_a_stream_that_breaks_is_retried_only_while_the_pane_saw_nothing(run_agent):
    steps = [{"stream_error": {"text": "", "body": OVERLOADED}}, message(text("Second try."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert [s["reason"] for s in statuses(updates) if s["status"] == "retry"] == ["server"]
    assert len(read_requests(record)) == 2

    steps = [{"stream_error": {"text": "partial", "body": OVERLOADED}}, message(text("never"))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert message_text(updates) == "partial"
    assert response["error"]["data"]["code"] == "agent.stream_interrupted"
    assert len(read_requests(record)) == 1


def test_cancel_in_the_middle_of_a_stream(run_agent):
    steps = [
        {**message(text("This is a long answer that the person will stop part of the way through, "
                        "well before it ends.")), "hang_after": 2},
        message(text("Fresh start.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    seen = threading.Event()

    def on_update(update):
        if update.get("sessionUpdate") == "agent_message_chunk" and not seen.is_set():
            seen.set()
            agent.notify("session/cancel", {"sessionId": sid})

    started = time.monotonic()
    response, updates = agent.prompt(sid, "Tell me everything", on_update=on_update, timeout=20)
    assert seen.is_set()
    assert response["result"]["stopReason"] == "cancelled"
    assert time.monotonic() - started < 10
    # The next turn works, and the interrupted reply is not in the history.
    response, updates = agent.prompt(sid, "Short answer please")
    assert response["result"]["stopReason"] == "end_turn"
    history = read_requests(record)[1]["messages"]
    assert [m["kind"] for m in history] == ["request", "request"] or [m["kind"] for m in history] == ["request"]
    assert prompts(history) == ["Tell me everything", "Short answer please"]
    assert "long answer" not in json.dumps(history)


def test_cancel_while_a_tool_runs_gives_every_call_a_result(run_agent, caliper):
    release = threading.Event()

    def slow(args):
        release.wait(10)
        return {"ok": True, "result": {"entries": []}}

    caliper.handlers["catalog.list"] = slow
    steps = [message(tool_use("t_slow", "catalog_list", {}), tool_use("t_after", "app_info", {}), stop="tool_use"),
             message(text("Again."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()

    def on_update(update):
        if update.get("sessionUpdate") == "tool_call_update" and update.get("status") == "in_progress":
            agent.notify("session/cancel", {"sessionId": sid})

    started = time.monotonic()
    response, updates = agent.prompt(sid, "List", on_update=on_update, timeout=20)
    release.set()
    assert response["result"]["stopReason"] == "cancelled"
    assert time.monotonic() - started < 8
    assert {u["toolCallId"] for u in updates if u["sessionUpdate"] == "tool_call_update"
            and u.get("status") == "failed"} == {"t_slow", "t_after"}
    agent.prompt(sid, "Again")
    results = tool_results(read_requests(record)[1]["messages"])
    assert set(results) == {"t_slow", "t_after"}
    assert all(r["outcome"] == "failed" and "Cancelled" in r["content"] for r in results.values())


def test_a_cancelled_permission_request_cancels_the_turn(run_agent, caliper):
    steps = [message(tool_use("t_save", "applet_save", {"id": "demo"}), stop="tool_use"), message(text("Ok."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Save", on_permission=lambda p: {"outcome": {"outcome": "cancelled"}})
    assert response["result"]["stopReason"] == "cancelled"
    assert caliper.commands("applet.save") == []
    agent.prompt(sid, "Next")
    assert tool_results(read_requests(record)[1]["messages"])["t_save"]["outcome"] == "failed"


def test_destructive_command_asks_and_acts_on_the_answer(run_agent, caliper):
    save = {"id": "demo", "overwrite": True}
    steps = [message(text("Saving."), tool_use("toolu_save", "applet_save", save), stop="tool_use"),
             message(text("Saved.")),
             message(text("Saving again."), tool_use("toolu_save2", "applet_save", save), stop="tool_use"),
             message(text("I did not save it."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    asked = []

    def answer_allow(params):
        asked.append(params)
        return allow(params)

    response, _ = agent.prompt(sid, "Save the applet", on_permission=answer_allow)
    assert response["result"]["stopReason"] == "end_turn"
    [params] = asked
    assert params["sessionId"] == sid and params["toolCall"]["toolCallId"] == "toolu_save"
    assert params["toolCall"]["rawInput"] == save
    assert {o["kind"] for o in params["options"]} == {"allow_once", "reject_once"}
    assert len(caliper.commands("applet.save")) == 1

    response, updates = agent.prompt(sid, "Save it again", on_permission=deny)
    assert response["result"]["stopReason"] == "end_turn"
    assert len(caliper.commands("applet.save")) == 1  # the denied call never reached Caliper
    denied = tool_results(read_requests(record)[3]["messages"])["toolu_save2"]
    assert denied["outcome"] == "failed" and "denied" in denied["content"]


def test_spec_edits_are_previewed_and_shown_as_diffs_before_they_run(run_agent, caliper):
    before_text = caliper.spec_text
    new_text = before_text + '\n[[source]]\nid = "loss"\nsql = "SELECT 1"\n'
    panel = {"id": "heat", "primitive": "heatmap", "bind": {"values": "entropy"}}
    steps = [message(tool_use("toolu_p", "panel_create", panel),
                     tool_use("toolu_t", "spec_set_text", {"text": new_text}), stop="tool_use"),
             message(text("Done."))]
    agent, _ = run_agent(steps)
    sid = agent.start_session("/tmp/ws")
    asked = []

    def answer(params):
        asked.append(params)
        # When the person is asked, the change has not been applied yet.
        assert len(caliper.commands(params["toolCall"]["rawInput"] and
                                    {"toolu_p": "panel.create", "toolu_t": "spec.set_text"}[
                                        params["toolCall"]["toolCallId"]])) == 0
        return allow(params)

    response, updates = agent.prompt(sid, "Add a heatmap", on_permission=answer)
    assert response["result"]["stopReason"] == "end_turn"
    assert [p["toolCall"]["toolCallId"] for p in asked] == ["toolu_p", "toolu_t"]
    # Each edit was previewed with dry_run and then run once, after the answer.
    for name in ("panel.create", "spec.set_text"):
        assert len(caliper.commands(name, dry=True)) == 1 and len(caliper.commands(name)) == 1
        assert caliper.commands(name, dry=True)[0]["group"] == caliper.commands(name)[0]["group"]
    panel_diff = asked[0]["toolCall"]["content"][0]
    assert panel_diff["type"] == "diff" and panel_diff["path"] == "/tmp/ws/applets/demo/demo.spec.toml"
    assert panel_diff["oldText"] == before_text and 'id = "heat"' in panel_diff["newText"]
    text_diff = asked[1]["toolCall"]["content"][0]
    assert text_diff["oldText"] == panel_diff["newText"] and text_diff["newText"] == new_text
    # The completed call shows what changed, from the command's own diff.
    done = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_p"
                and u.get("status") == "completed")
    assert done["content"][0] == panel_diff


def test_a_preview_without_change_or_refused_asks_nothing(run_agent, caliper):
    caliper.spec_text += '\n[[panel]]\nid = "demo"\nprimitive = "text"\n'
    same = caliper.spec_text
    steps = [message(tool_use("t_same", "spec_set_text", {"text": same}),
                     tool_use("t_dup", "panel_create", {"id": "demo", "primitive": "text"}),
                     tool_use("t_own", "panel_create", {"id": "heat", "primitive": "heatmap",
                                                        "bind": {"values": "x"}, "dry_run": True}),
                     stop="tool_use"),
             message(text("Nothing to do."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    asked = []
    response, updates = agent.prompt(sid, "Tidy", on_permission=lambda p: asked.append(p) or allow(p))
    assert response["result"]["stopReason"] == "end_turn"
    assert asked == []
    # An empty diff runs the edit, which changes nothing; a refused preview is
    # the call's result and the edit never runs; the model's own dry run is
    # sent once, as asked.
    assert len(caliper.commands("spec.set_text")) == 1
    assert len(caliper.commands("panel.create", dry=None)) == 2
    assert len(caliper.commands("panel.create", dry=True)) == 2
    results = tool_results(read_requests(record)[1]["messages"])
    assert results["t_dup"]["outcome"] == "failed" and "panel.exists" in results["t_dup"]["content"]
    assert results["t_own"]["outcome"] == "success" and '"applied": false' in results["t_own"]["content"]


def test_allow_always_stops_asking_for_spec_changes_but_not_for_destructive_ones(run_agent, caliper):
    steps = [message(tool_use("t1", "panel_create", {"id": "a", "primitive": "series", "bind": {"y": "loss.loss"}}), stop="tool_use"),
             message(tool_use("t2", "panel_create", {"id": "b", "primitive": "series", "bind": {"y": "loss.loss"}}),
                     tool_use("t3", "loader_delete", {"id": "old"}), stop="tool_use"),
             message(text("Done."))]
    agent, _ = run_agent(steps)
    sid = agent.start_session()
    asked = []

    def always(params):
        asked.append(params["toolCall"]["toolCallId"])
        option = next((o for o in params["options"] if o["kind"] == "allow_always"),
                      next(o for o in params["options"] if o["kind"] == "allow_once"))
        return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}

    response, _ = agent.prompt(sid, "Two panels", on_permission=always)
    assert response["result"]["stopReason"] == "end_turn"
    assert asked == ["t1", "t3"]
    assert len(caliper.commands("panel.create")) == 2 and len(caliper.commands("loader.delete")) == 1
    # Only the reviewed edit was previewed.
    assert len(caliper.commands("panel.create", dry=True)) == 1


def test_what_caliper_reviews_itself_is_not_asked_twice(run_agent, caliper):
    # The app holds spec edits and destructive commands for the person, so the
    # sidecar sends no permission request and no dry run for them.
    caliper.handlers["review.status"] = lambda args: {"ok": True, "result": {
        "origin": "agent", "spec_edits": True, "destructive": True}}
    panel = {"id": "heat", "primitive": "heatmap", "bind": {"values": "entropy"}}
    steps = [message(tool_use("t_p", "panel_create", panel),
                     tool_use("t_d", "loader_delete", {"id": "old"}), stop="tool_use"),
             message(text("Done.")),
             message(tool_use("t_q", "panel_create", {"id": "q", "primitive": "text"}),
                     tool_use("t_e", "loader_delete", {"id": "older"}), stop="tool_use"),
             message(text("Done again."))]
    agent, _ = run_agent(steps)
    sid = agent.start_session("/tmp/ws")
    asked = []
    response, updates = agent.prompt(sid, "Add a heatmap", on_permission=lambda p: asked.append(p) or allow(p))
    assert response["result"]["stopReason"] == "end_turn"
    assert asked == []
    assert len(caliper.commands("panel.create", dry=None)) == 1 and len(caliper.commands("loader.delete")) == 1
    assert len(caliper.commands("review.status")) == 1  # once for the turn
    # The completed edit still shows its change, from the command's own diff.
    done = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "t_p"
                and u.get("status") == "completed")
    assert done["content"][0]["type"] == "diff" and 'id = "heat"' in done["content"][0]["newText"]

    # When the app reviews spec edits only, the sidecar still asks before a
    # destructive command, and the setting is read again in the next turn.
    caliper.handlers["review.status"] = lambda args: {"ok": True, "result": {
        "origin": "agent", "spec_edits": True, "destructive": False}}
    response, _ = agent.prompt(sid, "Again", on_permission=lambda p: asked.append(p) or allow(p))
    assert response["result"]["stopReason"] == "end_turn"
    assert [p["toolCall"]["toolCallId"] for p in asked] == ["t_e"]
    assert len(caliper.commands("review.status")) == 2
    assert len(caliper.commands("panel.create", dry=True)) == 0


def test_capture_unavailable_is_a_clear_tool_error(tmp_path):
    from conftest import AgentProcess, FakeCaliper, write_script
    fake = FakeCaliper(tmp_path, capture_status=501)
    try:
        script, record = write_script(tmp_path, [message(tool_use("c", "capture_frame", {}), stop="tool_use"),
                                                 message(text("No capture here."))])
        agent = AgentProcess(script=script, api_file=fake.api_file)
        sid = agent.start_session()
        agent.prompt(sid, "Capture")
        result = tool_results(read_requests(record)[1]["messages"])["c"]
        assert result["outcome"] == "failed" and "capture.unavailable" in result["content"]
        agent.close()
    finally:
        fake.close()


def test_no_key_refuses_the_turn_with_an_error_the_pane_can_show(run_agent):
    agent, record = run_agent([message(text("never"))], key=None)
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Hello")
    assert response["error"]["code"] == -32000
    assert response["error"]["data"]["code"] == "agent.no_key"
    assert "No Claude API key is set" in response["error"]["message"]
    assert read_requests(record) == []

    agent, _ = run_agent([message(text("never"))], key="sk-ant with spaces\n")
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Hello")
    assert response["error"]["data"]["code"] == "agent.bad_key"


def test_a_command_layer_that_is_missing_is_a_tool_error(tmp_path):
    from conftest import AgentProcess, FakeCaliper, write_script
    fake = FakeCaliper(tmp_path, command_layer=False)
    try:
        script, record = write_script(tmp_path, [message(tool_use("c", "catalog_list", {}), tool_use("i", "app_info", {}),
                                                         stop="tool_use"), message(text("Older app."))])
        agent = AgentProcess(script=script, api_file=fake.api_file)
        sid = agent.start_session()
        agent.prompt(sid, "List")
        results = tool_results(read_requests(record)[1]["messages"])
        assert results["c"]["outcome"] == "failed" and "caliper.no_command_layer" in results["c"]["content"]
        assert results["i"]["outcome"] == "success" and "0.6.0" in results["i"]["content"]
        agent.close()
    finally:
        fake.close()


def test_the_tool_set_is_a_filter_over_the_table(run_agent):
    agent, record = run_agent([message(text("Hi."))], extra_args=["--tools", "catalog,spec.get"])
    sid = agent.start_session()
    agent.prompt(sid, "Hello")
    [req] = read_requests(record)
    assert sorted(t["name"] for t in req["tools"]) == ["catalog_describe", "catalog_list", "catalog_sample", "spec_get"]


def test_unknown_methods_and_bad_requests(run_agent):
    agent, _ = run_agent([])
    response, _ = agent.request("session/load", {"sessionId": "x"})
    assert response["error"]["code"] == -32601
    response, _ = agent.request("session/new", {"cwd": "relative/path", "mcpServers": []})
    assert response["error"]["code"] == -32602
    agent.send({"jsonrpc": "2.0", "id": 99, "method": "initialize"})  # no params: still answered
    agent.proc.stdin.write(b"{not json\n[1,2]\n")
    agent.proc.stdin.flush()
    response, _ = agent.request("initialize", {"protocolVersion": 7})
    assert response["result"]["protocolVersion"] == 1
    sid = agent.start_session()
    response, _ = agent.request("session/prompt", {"sessionId": sid, "prompt": []})
    assert response["error"]["code"] == -32602
