"""caliper-agent as the side chat's agent, in its general role.

The side chat starts any ACP agent with one MCP server in `session/new`, named
`caliper`: this build's caliper-mcp with `--api-file <the session's own API
file>`, whose token has the origin `agent` and which the app revokes when the
agent's session ends. caliper-agent starts no MCP server; it reaches Caliper
API itself, so it must take that file from the server's arguments, or its
commands would run with the token of `api.json` (the origin `mcp`) and the
side chat would not mirror their approvals. Each call of a Caliper command
carries the command in `_meta.caliper.command`, so that the side chat shows it
as one and folds its approval into it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conftest import AgentProcess, FakeCaliper, message, text, tool_use, write_script


def caliper_server(api_file: Path | str, name: str = "caliper", *, joined: bool = False) -> dict:
    """The MCP server as the side chat hands it (side_chat.cpp, build_chat_launch)."""
    args = ["--origin", "agent", f"--api-file={api_file}"] if joined else ["--origin", "agent", "--api-file",
                                                                           str(api_file)]
    return {"name": name, "command": "/opt/caliper/bin/caliper-mcp", "args": args, "env": []}


def start(agent: AgentProcess, servers: list) -> str:
    response, _ = agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    assert response["result"]["protocolVersion"] == 1
    response, _ = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": servers})
    assert "result" in response, response
    return response["result"]["sessionId"]


@pytest.fixture
def two_apps(tmp_path):
    """Two fake Caliper APIs: the API file given on the command line, and the session's own."""
    (tmp_path / "default").mkdir()
    (tmp_path / "session").mkdir()
    default, session = FakeCaliper(tmp_path / "default"), FakeCaliper(tmp_path / "session")
    yield default, session
    default.close()
    session.close()


def test_the_session_uses_the_api_file_of_the_caliper_server_it_is_handed(tmp_path, two_apps):
    default, session = two_apps
    steps = [message(tool_use("t1", "catalog_list", {}), stop="tool_use"), message(text("One table."))]
    script, _ = write_script(tmp_path, steps)
    agent = AgentProcess(script=script, api_file=default.api_file)
    try:
        sid = start(agent, [caliper_server(session.api_file)])
        response, updates = agent.prompt(sid, "What data is there?")
        assert response["result"]["stopReason"] == "end_turn"
        assert [c["path"] for c in session.commands("catalog.list")] == ["/v0/commands/catalog.list"]
        assert default.calls == []
        # The call names its command for the side chat.
        started = next(u for u in updates if u["sessionUpdate"] == "tool_call" and u["toolCallId"] == "t1")
        assert started["_meta"] == {"caliper": {"command": "catalog.list"}}
    finally:
        agent.close()


def test_without_a_caliper_server_the_session_keeps_the_command_lines_file(tmp_path, two_apps):
    default, session = two_apps
    steps = [message(tool_use("t1", "catalog_list", {}), stop="tool_use"), message(text("One table."))]
    script, _ = write_script(tmp_path, steps)
    agent = AgentProcess(script=script, api_file=default.api_file)
    try:
        other = {"name": "files", "command": "/usr/local/bin/files-mcp", "args": ["--api-file", str(session.api_file)],
                 "env": []}
        sid = start(agent, [other])
        agent.prompt(sid, "What data is there?")
        assert len(default.commands("catalog.list")) == 1
        assert session.calls == []
    finally:
        agent.close()


def test_each_session_keeps_its_own_file(tmp_path, two_apps):
    default, session = two_apps
    steps = [message(tool_use("a", "catalog_list", {}), stop="tool_use"), message(text("First.")),
             message(tool_use("b", "spec_get", {}), stop="tool_use"), message(text("Second."))]
    script, _ = write_script(tmp_path, steps)
    agent = AgentProcess(script=script, api_file=default.api_file)
    try:
        first = start(agent, [caliper_server(session.api_file, joined=True)])
        second = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []})[0]
        second = second["result"]["sessionId"]
        agent.prompt(first, "List")
        agent.prompt(second, "Spec")
        assert [c["path"] for c in session.calls] == ["/v0/commands/catalog.list"]
        assert [c["path"] for c in default.calls] == ["/v0/commands/spec.get"]
        # A closed session's client goes with it; the other session still works.
        assert "result" in agent.request("session/close", {"sessionId": first})[0]
    finally:
        assert agent.close() == 0


def test_a_caliper_server_without_its_api_file_refuses_the_session(tmp_path, two_apps):
    # The session's own file is the agent token's; api.json has the token of
    # outside agents (origin mcp). A Caliper server handed without a usable
    # file must not fall back to it silently.
    default, _ = two_apps
    script, _ = write_script(tmp_path, [message(text("never"))])
    agent = AgentProcess(script=script, api_file=default.api_file)
    try:
        agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        for server in ({"name": "caliper", "command": "/opt/caliper/bin/caliper-mcp", "args": ["--origin", "agent"],
                        "env": []}, caliper_server("relative/api.json"),
                       # Caliper's server in the HTTP form, which carries no API file at all.
                       {"name": "caliper", "type": "http", "url": "http://127.0.0.1:1/mcp", "headers": []}):
            response, _ = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": [server]})
            assert response["error"]["code"] == -32602
            assert response["error"]["data"]["code"] == "agent.no_session_api_file"
            assert "no API file of its own" in response["error"]["message"]
        assert default.calls == []
    finally:
        agent.close()


def test_the_api_file_of_a_server_list():
    from caliper_agent.caliper_api import session_api_file

    good = str(Path(tempfile.gettempdir()).resolve() / "api-agent-7-1.json")   # absolute on every platform
    other = str(Path(tempfile.gettempdir()).resolve() / "other.json")
    assert session_api_file([caliper_server(good)]) == (True, Path(good))
    assert session_api_file([caliper_server(good, joined=True)]) == (True, Path(good))
    # Named otherwise, but caliper-mcp by its command.
    assert session_api_file([caliper_server(good, name="caliper-app")]) == (True, Path(good))
    # The first Caliper server counts.
    assert session_api_file([caliper_server(good), caliper_server(other)]) == (True, Path(good))
    # A Caliper server without a usable file.
    for servers in ([{"name": "caliper", "command": "x", "args": "--api-file"}],
                    [caliper_server("relative/api.json")], [caliper_server("")], [caliper_server(good + "\x00")],
                    [caliper_server("/" + "a" * 5000)],
                    [{"name": "caliper", "command": "/bin/caliper-mcp", "args": ["--api-file"]}],
                    [{"name": "caliper", "command": "/bin/caliper-mcp", "args": ["--api-file", 7]}],
                    [{"name": "files", "command": "/bin/caliper-mcp", "args": []}],
                    # A server named caliper is Caliper's in any form: without a command (HTTP, SSE) it has no
                    # API file to give, so the session is refused rather than fall back to api.json.
                    [{"name": "caliper"}],
                    [{"name": "caliper", "type": "http", "url": "http://127.0.0.1:1/mcp", "headers": []}],
                    [{"name": "caliper", "type": "sse", "url": "http://127.0.0.1:1/sse", "headers": []}],
                    [{"name": "caliper", "command": 7, "args": ["--api-file", good]}]):
        assert session_api_file(servers) == (True, None), json.dumps(servers)
    # No Caliper server: the sidecar's own file.
    for servers in (None, "caliper", [], [{"name": "Caliper-docs", "type": "http", "url": "http://127.0.0.1:1/mcp"}],
                    [{"name": "other", "command": "/bin/other", "args": ["--api-file", good]}]):
        assert session_api_file(servers) == (False, None), json.dumps(servers)
