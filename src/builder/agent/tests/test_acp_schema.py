"""Every ACP message the sidecar sends, validated against the protocol's own models.

The models come from the official ACP Python SDK (`agent-client-protocol`,
generated from the protocol's schema), installed as a test dependency only:
the sidecar keeps its own JSON-RPC layer (acp.py). Scenarios cover a tool turn
with a capture, spec diffs in permission requests, statuses, a refusal, a cut-off
reply, exhausted retries, a missing key and a refusal on the Claude API's wire
format. Skipped when
the SDK is not installed.
"""

from __future__ import annotations

import json
import typing

import pytest

acp_schema = pytest.importorskip("acp.schema")

from conftest import allow, message, text, thinking, tool_use  # noqa: E402

RESPONSES = {
    "initialize": acp_schema.InitializeResponse,
    "session/new": acp_schema.NewSessionResponse,
    "session/set_config_option": acp_schema.SetSessionConfigOptionResponse,
    "session/prompt": acp_schema.PromptResponse,
}


def _validate_all(agent) -> int:
    """Validates every line the sidecar wrote; returns how many there were."""
    methods: dict[int, str] = {}
    for raw in agent.sent:
        methods[raw["id"]] = raw["method"]
    count = 0
    for line in agent.lines:
        msg = json.loads(line)
        assert msg.get("jsonrpc") == "2.0"
        if msg.get("method") == "session/update":
            acp_schema.SessionNotification.model_validate(msg["params"])
            # The SDK's models drop an optional enum value they do not know
            # instead of refusing it, so the enums are checked here.
            update = msg["params"]["update"]
            if "status" in update:
                assert update["status"] in typing.get_args(acp_schema.ToolCallStatus), update
            if "kind" in update:
                assert update["kind"] in typing.get_args(acp_schema.ToolKind), update
        elif msg.get("method") == "session/request_permission":
            acp_schema.RequestPermissionRequest.model_validate(msg["params"])
        elif "error" in msg:
            acp_schema.Error.model_validate(msg["error"])
        elif "result" in msg:
            model = RESPONSES[methods[msg["id"]]]
            model.model_validate(msg["result"])
        else:
            raise AssertionError(f"unexpected message {line[:200]}")
        count += 1
    return count


@pytest.fixture
def tracked(run_agent):
    """run_agent, with the requests the test sent remembered by id."""

    def start(steps, **kwargs):
        agent, record = run_agent(steps, **kwargs)
        agent.sent = []
        send = agent.send

        def tracking_send(msg):
            if "method" in msg and "id" in msg:
                agent.sent.append(msg)
            send(msg)

        agent.send = tracking_send
        return agent, record

    return start


def test_the_messages_of_many_turns_follow_the_protocol(tracked, caliper):
    steps = [
        message(thinking("Look."), text("Adding a panel."),
                tool_use("p1", "panel_create", {"id": "heat", "primitive": "heatmap", "bind": {"values": "x"}}),
                tool_use("c1", "capture_frame", {}), stop="tool_use"),
        {"error": {"status": 529, "body": {"type": "error", "error": {"type": "overloaded_error",
                                                                      "message": "Overloaded"}}}},
        message(text("Done.")),
        message(text("No"), stop="refusal", stop_details={"type": "refusal", "category": "cyber"}),
        message(text("Cut"), tool_use("x1", "catalog_list", {}), stop="max_tokens"),
        *[{"connect_error": "refused"}] * 5,
    ]
    agent, _ = tracked(steps)
    sid = agent.start_session("/tmp/ws")
    agent.request("session/set_config_option", {"sessionId": sid, "configId": "effort", "value": "max"})
    agent.prompt(sid, "Add a heatmap", on_permission=allow)
    agent.prompt(sid, "Refused")
    agent.prompt(sid, "Cut off")
    response, _ = agent.prompt(sid, "Offline")
    assert response["error"]["data"]["code"] == "agent.retries_exhausted"
    agent.request("session/load", {"sessionId": sid})
    assert _validate_all(agent) >= 25


def test_the_models_reject_wrong_shapes():
    with pytest.raises(Exception):
        acp_schema.SessionNotification.model_validate(
            {"sessionId": "s", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text",
                                                                                            "text": 1}}})
    with pytest.raises(Exception):
        acp_schema.PromptResponse.model_validate({"stopReason": "finished"})
    with pytest.raises(Exception):
        acp_schema.RequestPermissionRequest.model_validate(
            {"sessionId": "s", "toolCall": {"toolCallId": "t"},
             "options": [{"optionId": "a", "name": "A", "kind": "allow_sometimes"}]})


def test_the_messages_without_a_key_and_of_a_wire_refusal_follow_the_protocol(tracked, caliper):
    agent, _ = tracked([message(text("never"))], key=None)
    sid = agent.start_session()
    agent.prompt(sid, "Hello")
    assert _validate_all(agent) == 3

    agent, _ = tracked([message(text("Let me "), tool_use("d1", "loader_delete", {"id": "x"}), stop="refusal",
                                stop_details={"type": "refusal", "category": "cyber"})], route="anthropic")
    sid = agent.start_session()
    response, _ = agent.prompt(sid, "Go", on_permission=allow)
    assert response["result"]["stopReason"] == "refusal"
    assert _validate_all(agent) > 5
