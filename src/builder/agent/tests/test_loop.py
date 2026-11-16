"""The turn loop through the real stdio protocol, against the scripted transport.

Each test starts the sidecar as a child process with a fake key, a script of
Claude API replies and a fake Caliper API, then checks the ACP messages the
sidecar sent, the requests the scripted transport received and the calls that
reached Caliper API.
"""

from __future__ import annotations

import base64
import json
import threading
import time

from conftest import (FAKE_KEY, allow, deny, message, message_text, read_requests, statuses, text, thinking,
                      tool_use)

CACHE_1 = {"input_tokens": 40, "output_tokens": 12, "cache_creation_input_tokens": 9000, "cache_read_input_tokens": 0}
CACHE_2 = {"input_tokens": 25, "output_tokens": 8, "cache_creation_input_tokens": 60, "cache_read_input_tokens": 9000}


def test_plain_turn_streams_and_sends_the_documented_request(run_agent):
    agent, record = run_agent([message(thinking("The person says hello."), text("Hello. What shall we build?"))])
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
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert body["context_management"] == {"edits": [{"type": "compact_20260112"}]}
    betas = req["headers"]["anthropic-beta"].split(",")
    assert "server-side-fallback-2026-07-01" in betas and "compact-2026-01-12" in betas
    for forbidden in ("temperature", "top_p", "top_k", "thread"):
        assert forbidden not in body
    # Tools: one per command, named with _ for ., strict and eagerly streamed.
    names = [t["name"] for t in body["tools"]]
    assert "panel_create" in names and "capture_frame" in names and len(names) == len(set(names))
    for tool in body["tools"]:
        assert tool["eager_input_streaming"] is True
        assert tool["strict"] is True
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
    agent, record = run_agent(steps)
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
    assert caliper.commands("panel.create") == [] and caliper.commands("catalog.describe") == []
    failed = {u["toolCallId"] for u in updates if u["sessionUpdate"] == "tool_call_update" and u.get("status") == "failed"}
    assert failed == {"toolu_bad", "toolu_trunc"}
    requests = read_requests(record)
    result = requests[1]["body"]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    payload = json.loads(result["content"][0]["text"])
    assert "INVALID_JSON" in payload and "args.id" in payload["problem"]
    truncated = requests[2]["body"]["messages"][-1]["content"][0]
    assert truncated["is_error"] is True
    assert "INVALID_JSON" in json.loads(truncated["content"][0]["text"])


def test_max_tokens_runs_no_tool_and_keeps_the_history_valid(run_agent, caliper):
    steps = [
        message(text("Creating the panel"), tool_use("toolu_cut", "panel_create", {"id": "heat"}), stop="max_tokens"),
        message(text("Ready.")),
    ]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Make a heatmap")
    assert response["result"]["stopReason"] == "max_tokens"
    assert caliper.commands() == []
    assert any(u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_cut" and u["status"] == "failed"
               for u in updates)
    response, _ = agent.prompt(sid, "Go on")
    assert response["result"]["stopReason"] == "end_turn"
    history = read_requests(record)[1]["body"]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "user"]
    assert history[2]["content"][0]["tool_use_id"] == "toolu_cut" and history[2]["content"][0]["is_error"]


def test_refusal_is_surfaced_and_the_prompt_is_taken_back(run_agent, caliper):
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
    assert status["category"] == "cyber" and status["prompt_removed"] is True
    response, _ = agent.prompt(sid, "Plan a loss dashboard")
    assert response["result"]["stopReason"] == "end_turn"
    # The declined prompt and its partial reply are not in the next request.
    history = read_requests(record)[1]["body"]["messages"]
    assert history == [{"role": "user", "content": [{"type": "text", "text": "Plan a loss dashboard"}]}]


def test_rate_limit_then_success_reports_the_retry(run_agent):
    steps = [
        {"error": {"status": 429, "headers": {"retry-after": "0"},
                   "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "Slow down."}}}},
        {"error": {"status": 529, "body": {"type": "error", "error": {"type": "overloaded_error",
                                                                      "message": "Overloaded"}}}},
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
    # Exactly one HTTP request per attempt: the SDK's own retries are off.
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
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    steps = [{"stream_error": {"after": 1, "body": overloaded}}, message(text("Second try."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    assert [s["reason"] for s in statuses(updates) if s["status"] == "retry"] == ["server"]
    assert len(read_requests(record)) == 2

    steps = [{"stream_error": {"after": 4, "body": overloaded}}, message(text("never"))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert message_text(updates) == "partial"
    assert response["error"]["data"]["code"] == "agent.stream_interrupted"
    assert "Send the prompt again" in response["error"]["message"]
    assert len(read_requests(record)) == 1


def test_cancel_in_the_middle_of_a_stream(run_agent):
    steps = [
        {**message(text("This is a long answer that the person will stop part of the way through, "
                        "well before it ends.")), "hang_after": 4},
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
    history = read_requests(record)[1]["body"]["messages"]
    assert [m["role"] for m in history] == ["user", "user"]


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
    denied = read_requests(record)[3]["body"]["messages"][-1]["content"][0]
    assert denied["is_error"] and "denied" in denied["content"][0]["text"]


def test_spec_changes_are_reviewed_and_shown_as_diffs(run_agent, caliper):
    new_text = caliper.spec_text + '\n[[source]]\nid = "loss"\nsql = "SELECT 1"\n'
    steps = [message(tool_use("toolu_p", "panel_create", {"id": "heat", "primitive": "heatmap", "bind": {"values": "entropy"}}),
                     tool_use("toolu_t", "spec_set_text", {"text": new_text}), stop="tool_use"),
             message(text("Done."))]
    agent, _ = run_agent(steps)
    sid = agent.start_session("/tmp/ws")
    asked = []
    response, updates = agent.prompt(sid, "Add a heatmap", on_permission=lambda p: asked.append(p) or allow(p))
    assert response["result"]["stopReason"] == "end_turn"
    assert [p["toolCall"]["toolCallId"] for p in asked] == ["toolu_p", "toolu_t"]
    # The spec.set_text request shows the change before it is applied.
    diff = asked[1]["toolCall"]["content"][0]
    assert diff["type"] == "diff" and diff["path"] == "/tmp/ws/applets/demo/demo.spec.toml"
    assert diff["newText"] == new_text and diff["oldText"].startswith("caliper_spec = 0")
    # panel.create shows what changed once it has run.
    done = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_p"
                and u.get("status") == "completed")
    after_diff = done["content"][0]
    assert after_diff["type"] == "diff" and 'id = "heat"' in after_diff["newText"]
    assert 'id = "heat"' not in after_diff["oldText"]


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


def test_capture_comes_back_as_an_image(run_agent, caliper):
    steps = [message(tool_use("toolu_cap", "capture_frame", {}), tool_use("toolu_r", "panel_render", {"id": "heat"}),
                     stop="tool_use"),
             message(text("The heatmap is drawn."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Show me")
    assert response["result"]["stopReason"] == "end_turn"
    results = read_requests(record)[1]["body"]["messages"][-1]["content"]
    for result, size in zip(results, ((4, 3), (8, 5))):
        assert "is_error" not in result
        textblock, image = result["content"]
        assert json.loads(textblock["text"])["width"] == size[0]
        assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
        assert base64.b64decode(image["source"]["data"]).startswith(b"\x89PNG")
    shown = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_cap"
                 and u.get("status") == "completed")
    assert any(c["type"] == "content" and c["content"]["type"] == "image" for c in shown["content"])
    assert caliper.calls[0]["path"] == "/v0/capture.png"


def test_capture_unavailable_is_a_clear_tool_error(tmp_path):
    from conftest import AgentProcess, FakeCaliper, write_script
    fake = FakeCaliper(tmp_path, capture_status=501)
    try:
        script, record = write_script(tmp_path, [message(tool_use("c", "capture_frame", {}), stop="tool_use"),
                                                 message(text("No capture here."))])
        agent = AgentProcess(script=script, api_file=fake.api_file)
        sid = agent.start_session()
        response, _ = agent.prompt(sid, "Capture")
        result = read_requests(record)[1]["body"]["messages"][-1]["content"][0]
        assert result["is_error"] and "capture.unavailable" in result["content"][0]["text"]
        agent.close()
    finally:
        fake.close()


def test_cache_fields_across_two_turns(run_agent, tmp_path):
    steps = [message(text("First."), usage=CACHE_1), message(text("Second."), usage=CACHE_2)]
    usage_log = tmp_path / "usage.jsonl"
    agent, record = run_agent(steps, extra_args=["--usage-log", str(usage_log)])
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


def test_model_and_effort_are_config_options_and_haiku_has_no_fallback(run_agent):
    agent, record = run_agent([message(text("Fast.")), message(text("Fable."))])
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
    assert "fallbacks" not in haiku["body"]
    assert "server-side-fallback" not in haiku["headers"].get("anthropic-beta", "")
    assert fable["body"]["model"] == "claude-fable-5-2" and fable["body"]["fallbacks"] == "default"
    bad, _ = agent.request("session/set_config_option", {"sessionId": sid, "configId": "model", "value": "gpt-x"})
    assert bad["error"]["code"] == -32602


def test_fallback_mid_reply_drops_the_declined_blocks_and_their_tool_calls(run_agent, caliper):
    fallback = {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"},
                "trigger": {"type": "refusal", "category": "cyber"}}
    steps = [message(thinking("Plan."), text("Let me "), tool_use("toolu_declined", "loader_delete", {"id": "x"}),
                     fallback, text("Here is the answer."), tool_use("toolu_kept", "catalog_list", {}),
                     stop="tool_use", model="claude-opus-5",
                     iterations=[{"type": "message", "input_tokens": 5, "output_tokens": 2},
                                 {"type": "fallback_message", "input_tokens": 5, "output_tokens": 9}]),
             message(text("Done."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Go")
    assert response["result"]["stopReason"] == "end_turn"
    assert caliper.commands("loader.delete") == [] and len(caliper.commands("catalog.list")) == 1
    found = [s["status"] for s in statuses(updates)]
    assert "fallback" in found and "served_by" in found
    echoed = read_requests(record)[1]["body"]["messages"][1]["content"]
    assert [b["type"] for b in echoed] == ["text", "fallback", "text", "tool_use"]
    assert echoed[3]["id"] == "toolu_kept"
    assert any(u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "toolu_declined"
               and u["status"] == "failed" for u in updates)


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


def test_the_key_never_appears_in_any_output(run_agent, caliper):
    leak = f"invalid x-api-key: {FAKE_KEY}"
    caliper.handlers["catalog.list"] = lambda a: {"ok": False, "error": {"code": "x", "message": FAKE_KEY, "where": ""}}
    steps = [message(tool_use("t", "catalog_list", {}), stop="tool_use"),
             message(text(f"The tool said {FAKE_KEY}")),
             {"error": {"status": 401, "body": {"type": "error", "error": {"type": "authentication_error",
                                                                           "message": leak}}}}]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Try")
    assert response["result"]["stopReason"] == "end_turn"
    response, _ = agent.prompt(sid, "Again")
    assert response["error"]["data"]["code"] == "agent.key_rejected"
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
    agent, record = run_agent([message(text("One.")), message(text("Two."))])
    sid = agent.start_session()
    agent.prompt(sid, "a")
    agent.prompt(sid, "b")
    assert {r["host"] for r in read_requests(record)} == {"api.anthropic.com"}


def test_strict_mode_is_dropped_when_the_api_refuses_it_on_the_first_request(run_agent):
    steps = [{"error": {"status": 400, "body": {"type": "error", "error": {
        "type": "invalid_request_error", "message": "Too many strict tools in one request."}}}},
             message(text("Fine without strict."))]
    agent, record = run_agent(steps)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert response["result"]["stopReason"] == "end_turn"
    first, second = read_requests(record)
    assert all(t.get("strict") for t in first["body"]["tools"])
    assert not any(t.get("strict") for t in second["body"]["tools"])
    assert "strict_off" in [s["status"] for s in statuses(updates)]


def test_a_command_layer_that_is_missing_is_a_tool_error(tmp_path):
    from conftest import AgentProcess, FakeCaliper, write_script
    fake = FakeCaliper(tmp_path, command_layer=False)
    try:
        script, record = write_script(tmp_path, [message(tool_use("c", "catalog_list", {}), tool_use("i", "app_info", {}),
                                                         stop="tool_use"), message(text("Older app."))])
        agent = AgentProcess(script=script, api_file=fake.api_file)
        sid = agent.start_session()
        agent.prompt(sid, "List")
        missing, info = read_requests(record)[1]["body"]["messages"][-1]["content"]
        assert missing["is_error"] and "caliper.no_command_layer" in missing["content"][0]["text"]
        assert "is_error" not in info and "0.6.0" in info["content"][0]["text"]
        agent.close()
    finally:
        fake.close()


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
