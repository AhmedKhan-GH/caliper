"""The offline switch: with CALIPER_AGENT_OFFLINE=1 the sidecar never reaches a model server off this machine.

Every test and every tool in `src/builder/` that starts the sidecar sets the
switch, so a test that forgets its script, or loses it on the way (the key
store removes CALIPER_AGENT_TEST_SCRIPT at spawn, by design), cannot send a
request to a real API. With the switch on:

- **No real client to another host.** `check_provider()` refuses, before any
  client or transport exists, the Claude provider on its real transport and
  an OpenAI-compatible provider whose base URL is not a loopback address. Only
  the scripted model, the stubbed transports and servers on loopback run. The
  refusal is an `OfflineError` with the code `agent.offline`, which the
  sidecar answers to `session/new` and `session/prompt` as a JSON-RPC error.
- **No socket to another host either.** `install_socket_guard()` wraps
  `socket.getaddrinfo` and the connect methods of `socket.socket`, so a name
  other than a loopback name is never resolved and an address other than a
  loopback address is never connected to, whatever library asks; each attempt
  fails with `OfflineError` and is noted on stderr.

The key store passes the switch through to the sidecar unchanged
(`place_key_in_environment()` in key_store.cpp), and the sidecar does not
remove it from its environment, so code under test cannot strip it.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from urllib.parse import urlsplit

OFFLINE_ENV = "CALIPER_AGENT_OFFLINE"
OFFLINE_CODE = "agent.offline"
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.", "ip6-localhost", "ip6-loopback"})


class OfflineError(ConnectionRefusedError):
    """A model server or a socket that the offline switch refuses; the message is safe to show."""

    code = OFFLINE_CODE


def offline(environ: dict | None = None) -> bool:
    """Whether the switch is on: CALIPER_AGENT_OFFLINE is set to anything but empty or 0."""
    value = (os.environ if environ is None else environ).get(OFFLINE_ENV, "")
    return value.strip() not in ("", "0")


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    name = host.strip("[]").lower()
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def check_provider(kind: str, *, base_url: str | None = None, stubbed: bool = False) -> None:
    """Raises OfflineError when the switch is on and the provider would reach another host.

    `kind` is "claude" or "openai-compatible"; `stubbed` is true for a stubbed
    transport that never opens a socket (stub_transport.py)."""
    if not offline() or stubbed:
        return
    if kind == "claude":
        raise OfflineError(f"{OFFLINE_ENV} is set, so the sidecar does not reach the Claude API; a test must give "
                           "the scripted model (CALIPER_AGENT_TEST_SCRIPT) or a loopback server.")
    host = urlsplit(base_url or "").hostname
    if not is_loopback_host(host):
        raise OfflineError(f"{OFFLINE_ENV} is set, so the sidecar reaches only a model server on a loopback "
                           f"address, and not {host or 'this base URL'}.")


def _check_address(address) -> None:
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        host = address[0]
        if not is_loopback_host(host):
            sys.stderr.write(f"caliper-agent: {OFFLINE_ENV} refused a connection to {host}\n")
            raise OfflineError(f"{OFFLINE_ENV} is set: no connection to {host}.")


_installed = False


def install_socket_guard() -> None:
    """Refuses every name resolution and connection off loopback, for the life of the process."""
    global _installed
    if _installed:
        return
    _installed = True
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        name = host.decode("ascii", "replace") if isinstance(host, (bytes, bytearray)) else host
        if name is not None and not is_loopback_host(str(name)):
            sys.stderr.write(f"caliper-agent: {OFFLINE_ENV} refused to resolve {name}\n")
            raise OfflineError(f"{OFFLINE_ENV} is set: {name} is not resolved.")
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address)
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
