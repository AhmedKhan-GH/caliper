"""The OpenAI-compatible provider against an in-process fake server.

The sidecar runs with `--provider openai-compatible` and a base URL on
loopback, where a fake server speaks OpenAI's chat completions API (streamed,
as Server-Sent Events) with the differences real servers have: a model without
tool calling, a model without vision, and a lenient server that ignores
`strict`, sends malformed arguments and never makes parallel calls. Nothing
leaves the machine and no key is used.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import AgentProcess, message_text, statuses


class FakeOpenAI:
    """POST /v1/chat/completions, streamed. Replies are taken in order."""

    def __init__(self, replies: list[dict], *, refuse_tools: bool = False, refuse_images: bool = False) -> None:
        self.replies = list(replies)
        self.refuse_tools = refuse_tools
        self.refuse_images = refuse_images
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status: int, doc: dict) -> None:
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length))
                fake.requests.append(body)
                fake.headers.append({k.lower(): v for k, v in self.headers.items()})
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "not found"}})
                    return
                if fake.refuse_tools and body.get("tools"):
                    self._json(400, {"error": {"message": "fake-model does not support tools",
                                               "type": "invalid_request_error"}})
                    return
                if fake.refuse_images and "image_url" in json.dumps(body.get("messages")):
                    self._json(400, {"error": {"message": "image input is not supported by this model",
                                               "type": "invalid_request_error"}})
                    return
                reply = fake.replies.pop(0) if fake.replies else {"text": "(no more replies)"}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in fake.chunks(reply, body.get("model", "")):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def chunks(reply: dict, model: str) -> list[dict]:
        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        out = [chunk({"role": "assistant"})]
        if "text" in reply:
            out += [chunk({"content": reply["text"][i:i + 8]}) for i in range(0, len(reply["text"]), 8)]
        for n, call in enumerate(reply.get("tool_calls", [])):
            out.append(chunk({"tool_calls": [{"index": n, "id": call["id"], "type": "function",
                                              "function": {"name": call["name"], "arguments": ""}}]}))
            out.append(chunk({"tool_calls": [{"index": n, "function": {"arguments": call["arguments"]}}]}))
        # Some servers say "stop" even after tool calls; the sidecar runs them anyway.
        out.append(chunk({}, reply.get("finish", "tool_calls" if reply.get("tool_calls") else "stop")))
        out.append({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})
        return out

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def openai_agent(caliper):
    started: list[tuple[AgentProcess, FakeOpenAI]] = []

    def start(replies: list[dict], *extra: str, **flags) -> tuple[AgentProcess, FakeOpenAI]:
        server = FakeOpenAI(replies, **flags)
        agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                             extra_args=["--provider", "openai-compatible", "--base-url", server.base_url,
                                         "--model", "fake-model", *extra])
        started.append((agent, server))
        return agent, server

    yield start
    for agent, server in started:
        agent.close()
        server.close()


def _call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def test_tools_work_against_an_openai_compatible_server(openai_agent, caliper):
    agent, server = openai_agent([{"tool_calls": [_call("call_1", "catalog_list", '{"kind": "table"}')],
                                   "finish": "stop"},
                                  {"text": "There is one table."}])
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "What data do I have?")
    assert response["result"]["stopReason"] == "end_turn"
    assert message_text(updates) == "There is one table."
    assert [c["origin"] for c in caliper.commands("catalog.list")] == ["agent"]
    first, second = server.requests
    assert first["model"] == "fake-model" and first["stream"] is True
    names = [t["function"]["name"] for t in first["tools"]]
    assert "catalog_list" in names and len(names) == 35
    assert "authorization" in server.headers[0]  # a placeholder: OPENAI_API_KEY is never read
    tool_message = next(m for m in second["messages"] if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1" and "run42_metrics" in tool_message["content"]
    usage = response["result"]["_meta"]["caliper"]["usage"]
    assert usage["requests"] == 2 and usage["input_tokens"] == 22


def test_a_server_without_tool_calling_says_it_cannot_drive_caliper(openai_agent):
    agent, server = openai_agent([{"text": "never"}], refuse_tools=True)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Build a dashboard")
    assert response["result"]["stopReason"] == "end_turn"
    assert "cannot drive Caliper" in message_text(updates)
    assert "no_tools" in [s["status"] for s in statuses(updates)]
    assert len(server.requests) == 1
    # The model is known to be without tools now: no further request is made.
    response, updates = agent.prompt(sid, "Try again")
    assert "cannot drive Caliper" in message_text(updates) and len(server.requests) == 1

    agent, server = openai_agent([{"text": "never"}], "--model-tools", "no")
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Hello")
    assert "cannot drive Caliper" in message_text(updates) and server.requests == []


def test_a_server_without_vision_gets_the_capture_as_text(openai_agent, caliper):
    agent, server = openai_agent([{"tool_calls": [_call("call_c", "capture_frame", "{}")]},
                                  {"text": "The capture was taken."}], refuse_images=True)
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "Capture the frame")
    assert response["result"]["stopReason"] == "end_turn"
    assert "no_vision" in [s["status"] for s in statuses(updates)]
    assert len(server.requests) == 3
    refused, retried = server.requests[1], server.requests[2]
    assert "image_url" in json.dumps(refused["messages"])
    sent = json.dumps(retried["messages"])
    assert "image_url" not in sent and "cannot read images" in sent
    # The pane still gets the image.
    done = next(u for u in updates if u["sessionUpdate"] == "tool_call_update" and u["toolCallId"] == "call_c"
                and u.get("status") == "completed")
    assert any(c["type"] == "content" and c["content"]["type"] == "image" for c in done["content"])


def test_a_lenient_server_gets_error_results_it_can_correct(openai_agent, caliper):
    agent, server = openai_agent([
        {"tool_calls": [_call("call_1", "catalog_list", '{"kind": "tabl')]},             # malformed JSON
        {"tool_calls": [_call("call_2", "catalog_list", '{"kind": "table", "bogus": 1}')]},  # strict ignored
        {"tool_calls": [_call("call_3", "catalog_list", '{"kind": "table"}')]},
        {"text": "Done."},
    ])
    sid = agent.start_session()
    response, updates = agent.prompt(sid, "List the tables")
    assert response["result"]["stopReason"] == "end_turn"
    assert len(caliper.commands("catalog.list")) == 1
    tool_messages = [m for m in server.requests[3]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2", "call_3"]
    assert "INVALID_JSON" in tool_messages[0]["content"] and "not valid JSON" in tool_messages[0]["content"]
    assert "INVALID_JSON" in tool_messages[1]["content"] and "args.bogus" in tool_messages[1]["content"]
    assert "run42_metrics" in tool_messages[2]["content"]
    # What a server that does not know `strict` was sent.
    strict_sent = [t["function"].get("strict") for t in server.requests[0]["tools"]]
    assert set(strict_sent) <= {True, None}
