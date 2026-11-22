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

import json
import os
import select
import socket
import subprocess
import sys
import threading

import pytest

from conftest import AGENT_DIR, FAKE_KEY, AgentProcess, message_text
from fake_openai import FakeOpenAI


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

    def pending(self) -> bool:
        """A connection waits to be accepted."""
        try:
            return bool(select.select([self.sock], [], [], 0)[0])
        except (OSError, ValueError):
            return False

    def untouched(self) -> bool:
        """No connection came, and none waits."""
        return self.connections == 0 and not self.pending()

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
        assert "CALIPER_AGENT_OFFLINE is set" in agent.stderr()
    finally:
        agent.close()
    # The sidecar has exited, so every connection it made has been made.
    assert trap.untouched()
    assert FAKE_KEY not in agent.all_output()


def test_an_openai_compatible_server_off_loopback_is_refused_offline(caliper, trap):
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--provider", "openai-compatible", "--base-url", "http://192.0.2.10:8000/v1",
                                     "--model", "remote-model"])
    try:
        response = _new_session(agent)
        assert response["error"]["data"]["code"] == "agent.offline"
        assert "192.0.2.10" in response["error"]["message"]
    finally:
        agent.close()
    assert trap.untouched()


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
    import httpx2

    from caliper_agent.claude import ClaudeProvider
    from caliper_agent.keyguard import Secret
    from caliper_agent.offline import OfflineError, check_provider, is_stub_transport, offline
    from caliper_agent.openai_compat import OpenAICompatProvider
    from caliper_agent.stub_transport import StubTransport

    assert offline()  # conftest turns the switch on for this process
    with pytest.raises(OfflineError) as caught:
        ClaudeProvider(Secret(FAKE_KEY))
    assert caught.value.code == "agent.offline"
    ClaudeProvider(Secret(FAKE_KEY), transport=StubTransport({"steps": []}))   # the project's stub is allowed
    # Any other transport is not a stub: a real one, or one dressed as the stub.

    class LookAlike(StubTransport):
        pass

    for transport in (httpx2.AsyncHTTPTransport(), LookAlike({"steps": []})):
        assert not is_stub_transport(transport)
        with pytest.raises(OfflineError):
            ClaudeProvider(Secret(FAKE_KEY), transport=transport)
    with pytest.raises(OfflineError):
        OpenAICompatProvider("https://models.invalid/v1", "m", None)
    with pytest.raises(OfflineError):
        OpenAICompatProvider("https://models.invalid/v1", "m", None, transport=httpx2.AsyncHTTPTransport())
    for url in ("http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8080/v1"):
        OpenAICompatProvider(url, "m", None)
    # The classification itself, without any client or socket.
    with pytest.raises(OfflineError):
        check_provider("claude")
    with pytest.raises(OfflineError):
        check_provider("openai-compatible", base_url="https://models.invalid/v1")
    check_provider("openai-compatible", base_url="http://127.0.0.1:1/v1")


def test_the_clients_ignore_the_environment_offline():
    from caliper_agent.keyguard import Secret
    from caliper_agent.offline import trust_env
    from caliper_agent.openai_compat import OpenAICompatProvider
    from caliper_agent.probe import _openai_client

    assert trust_env() is False
    provider = OpenAICompatProvider("http://127.0.0.1:9/v1", "m", None)
    assert provider._client._client.trust_env is False
    assert _openai_client("http://127.0.0.1:9/v1", Secret(FAKE_KEY))._client.trust_env is False


def test_a_proxy_on_loopback_is_not_used_offline(caliper, trap):
    # The proxy variables point at the trap on loopback, which the socket guard
    # lets through; a client that read them would send the request there, from
    # where a proxy could forward it off the machine.
    server = FakeOpenAI([{"text": "Straight to the server."}])
    agent = AgentProcess(script=None, api_file=caliper.api_file, key=None,
                         extra_args=["--provider", "openai-compatible", "--base-url", server.base_url,
                                     "--model", "fake-model"])
    try:
        sid = agent.start_session()
        response, updates = agent.prompt(sid, "Hello")
        assert response["result"]["stopReason"] == "end_turn"
        assert message_text(updates) == "Straight to the server."
        assert len(server.requests) == 1
    finally:
        agent.close()
        server.close()
    assert trap.untouched()


def test_the_pytest_process_is_guarded():
    import caliper_agent.offline as offline

    assert offline._installed   # conftest installs the guard in this process too
    assert not any(name in os.environ for name in offline.PROXY_ENV)
    # Only now that the guard is known to be there: refused before any lookup.
    with pytest.raises(offline.OfflineError):
        socket.getaddrinfo("models.invalid", 443)


def test_the_offline_switch_values():
    from caliper_agent.offline import is_loopback_host, offline

    assert not offline({}) and not offline({"CALIPER_AGENT_OFFLINE": ""}) and not offline({"CALIPER_AGENT_OFFLINE": "0"})
    assert offline({"CALIPER_AGENT_OFFLINE": "1"}) and offline({"CALIPER_AGENT_OFFLINE": "yes"})
    for host in ("127.0.0.1", "127.8.9.10", "::1", "[::1]", "localhost", "LOCALHOST"):
        assert is_loopback_host(host), host
    for host in ("", None, "models.invalid", "10.0.0.1", "0.0.0.0", "localhost.example.com", "192.0.2.1"):
        assert not is_loopback_host(host), host


# The child of the socket guard's test. The real connect, sendto and lookups
# are replaced by recorders before the guard is installed, so that whatever the
# guard lets through is recorded rather than sent: no packet leaves the machine
# whether the guard works or not. Every target is a .invalid name or a TEST-NET
# address.
GUARD_CHILD = r"""
import json, socket, sys
from caliper_agent.offline import OfflineError, install_socket_guard

reached = []

def record(kind):
    def f(*args, **kwargs):
        reached.append([kind, repr(args[1] if kind in ("connect", "connect_ex") else args[-1] if kind == "sendto"
                                   else args[4] if kind == "sendmsg" else args[0])])
        if kind == "getaddrinfo":
            return []
        raise ConnectionRefusedError("recorded")
    return f

socket.socket.connect = record("connect")
socket.socket.connect_ex = record("connect_ex")
socket.socket.sendto = record("sendto")
socket.socket.sendmsg = record("sendmsg")
socket.getaddrinfo = record("getaddrinfo")
socket.gethostbyname = record("gethostbyname")
socket.gethostbyname_ex = record("gethostbyname_ex")
install_socket_guard()

refused = []
def attempt(label, fn):
    try:
        fn()
    except OfflineError:
        refused.append(label)
    except ConnectionRefusedError:
        pass

tcp = lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udp = lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
attempt("str", lambda: tcp().connect(("192.0.2.1", 9)))
attempt("bytes", lambda: tcp().connect((b"192.0.2.1", 9)))
attempt("bytearray", lambda: tcp().connect((bytearray(b"192.0.2.1"), 9)))
attempt("bytes name", lambda: tcp().connect((b"models.invalid", 9)))
attempt("non-ascii bytes", lambda: tcp().connect((bytes([0xFF, 0xFE]), 9)))
attempt("int host", lambda: tcp().connect((3221225985, 9)))
attempt("connect_ex", lambda: tcp().connect_ex((b"192.0.2.1", 9)))
attempt("sendto", lambda: udp().sendto(b"x", ("192.0.2.1", 9)))
attempt("sendto flags", lambda: udp().sendto(b"x", 0, (b"192.0.2.1", 9)))
attempt("sendmsg", lambda: udp().sendmsg([b"x"], [], 0, (b"192.0.2.1", 9)))
attempt("getaddrinfo", lambda: socket.getaddrinfo("models.invalid", 443))
attempt("getaddrinfo bytes", lambda: socket.getaddrinfo(b"models.invalid", 443))
attempt("getaddrinfo int", lambda: socket.getaddrinfo(12345, 443))
attempt("gethostbyname", lambda: socket.gethostbyname("models.invalid"))
attempt("gethostbyname_ex", lambda: socket.gethostbyname_ex(b"models.invalid"))
attempt("create_connection", lambda: socket.create_connection(("models.invalid", 443), timeout=2))
# Loopback passes the guard and reaches the (recording) socket API.
attempt("loopback", lambda: tcp().connect((b"127.0.0.1", 9)))
attempt("loopback name", lambda: socket.getaddrinfo("localhost", 9))
print(json.dumps({"refused": refused, "reached": reached}))
"""


def test_the_socket_guard_refuses_every_host_off_loopback():
    # In a child process, since the guard patches the socket module for good.
    run = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(AGENT_DIR)!r})\n" + GUARD_CHILD],
                         capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    assert run.returncode == 0, run.stderr
    doc = json.loads(run.stdout.strip().splitlines()[-1])
    assert doc["refused"] == ["str", "bytes", "bytearray", "bytes name", "non-ascii bytes", "int host", "connect_ex",
                              "sendto", "sendto flags", "sendmsg", "getaddrinfo", "getaddrinfo bytes", "getaddrinfo int",
                              "gethostbyname", "gethostbyname_ex", "create_connection"]
    # Only the loopback attempts got past the guard.
    assert doc["reached"] == [["connect", "(b'127.0.0.1', 9)"], ["getaddrinfo", "'localhost'"]]
    assert "refused a connection to 192.0.2.1" in run.stderr
