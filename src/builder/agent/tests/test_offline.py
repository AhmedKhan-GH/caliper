"""The offline switch (caliper_agent/offline.py): with CALIPER_AGENT_OFFLINE=1 nothing reaches a host off loopback.

The bug this guards against: a test started the sidecar with a fake key but
without its script switch, which the key store strips at spawn, so the
sidecar took its real Claude route and sent requests to the real API. With
the switch on, the provider is never built and every socket off loopback is
refused. A proxy trap on loopback (HTTPS_PROXY and HTTP_PROXY point at it)
records any request the sidecar would have tried, since the SDKs honour the
system's proxy settings: it must see none.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time

import pytest

from conftest import AGENT_DIR, FAKE_KEY, AgentProcess, message_text
from test_openai_compat import FakeOpenAI


class ProxyTrap:
    """A TCP listener on loopback that records every connection and closes it."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            self.connections += 1
            conn.close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._stop = True
        self.sock.close()


@pytest.fixture
def trap(monkeypatch):
    t = ProxyTrap()
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.setenv(name, t.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    yield t
    t.close()


def _new_session(agent: AgentProcess) -> dict:
    response, _ = agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    assert response["result"]["protocolVersion"] == 1
    response, _ = agent.request("session/new", {"cwd": "/tmp/caliper-test-workspace", "mcpServers": []})
    return response


def test_the_claude_route_without_a_script_is_refused_offline(caliper, trap):
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=FAKE_KEY)
    try:
        response = _new_session(agent)
        assert response["error"]["data"]["code"] == "agent.offline"
        assert "CALIPER_AGENT_OFFLINE" in response["error"]["message"]
        time.sleep(0.3)
        assert trap.connections == 0
        assert "CALIPER_AGENT_OFFLINE is set" in agent.stderr()
    finally:
        agent.close()
    assert FAKE_KEY not in agent.all_output()


def test_an_openai_compatible_server_off_loopback_is_refused_offline(caliper, trap):
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--provider", "openai-compatible", "--base-url", "http://192.0.2.10:8000/v1",
                                     "--model", "remote-model"])
    try:
        response = _new_session(agent)
        assert response["error"]["data"]["code"] == "agent.offline"
        assert "192.0.2.10" in response["error"]["message"]
        time.sleep(0.3)
        assert trap.connections == 0
    finally:
        agent.close()


def test_a_loopback_openai_compatible_server_still_works_offline(caliper):
    server = FakeOpenAI([{"text": "Hello from loopback."}])
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--provider", "openai-compatible", "--base-url", server.base_url,
                                     "--model", "fake-model"])
    try:
        sid = agent.start_session()
        response, updates = agent.prompt(sid, "Say hello")
        assert response["result"]["stopReason"] == "end_turn"
        assert message_text(updates) == "Hello from loopback."
        assert len(server.requests) == 1
    finally:
        agent.close()
        server.close()


def test_the_providers_refuse_to_be_built_offline():
    from caliper_agent.claude import ClaudeProvider
    from caliper_agent.keyguard import Secret
    from caliper_agent.offline import OfflineError, offline
    from caliper_agent.openai_compat import OpenAICompatProvider
    from caliper_agent.stub_transport import StubTransport

    assert offline()  # conftest turns the switch on for this process
    with pytest.raises(OfflineError) as caught:
        ClaudeProvider(Secret(FAKE_KEY))
    assert caught.value.code == "agent.offline"
    ClaudeProvider(Secret(FAKE_KEY), transport=StubTransport({"steps": []}))   # a stub is allowed
    with pytest.raises(OfflineError):
        OpenAICompatProvider("https://api.example.com/v1", "m", None)
    for url in ("http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8080/v1"):
        OpenAICompatProvider(url, "m", None)


def test_the_offline_switch_values():
    from caliper_agent.offline import is_loopback_host, offline

    assert not offline({}) and not offline({"CALIPER_AGENT_OFFLINE": ""}) and not offline({"CALIPER_AGENT_OFFLINE": "0"})
    assert offline({"CALIPER_AGENT_OFFLINE": "1"}) and offline({"CALIPER_AGENT_OFFLINE": "yes"})
    for host in ("127.0.0.1", "127.8.9.10", "::1", "[::1]", "localhost", "LOCALHOST"):
        assert is_loopback_host(host), host
    for host in ("", None, "api.anthropic.com", "10.0.0.1", "0.0.0.0", "localhost.example.com", "192.0.2.1"):
        assert not is_loopback_host(host), host


def test_the_socket_guard_refuses_every_host_off_loopback():
    # In a child process, since the guard patches the socket module for good.
    code = r"""
import socket, sys
from caliper_agent.offline import OfflineError, install_socket_guard
install_socket_guard()
listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=2).close()
for target in [("192.0.2.1", 443), ("api.anthropic.com", 443)]:
    try:
        socket.create_connection(target, timeout=2)
    except OfflineError:
        print("refused", target[0])
try:
    socket.getaddrinfo("api.openai.com", 443)
except OfflineError:
    print("not resolved")
s = socket.socket()
try:
    s.connect(("192.0.2.1", 80))
except OfflineError:
    print("connect refused")
print("ok")
"""
    env = {"PYTHONPATH": str(AGENT_DIR), "PATH": "/usr/bin:/bin"}
    run = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(AGENT_DIR)!r})\n" + code],
                         capture_output=True, text=True, timeout=30, env=env)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split("\n")[:5] == ["refused 192.0.2.1", "refused api.anthropic.com", "not resolved",
                                          "connect refused", "ok"]
    assert "refused a connection to 192.0.2.1" in run.stderr
