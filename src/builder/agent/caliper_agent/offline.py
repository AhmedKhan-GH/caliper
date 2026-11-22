"""The offline switch: with CALIPER_AGENT_OFFLINE=1 the sidecar never reaches a model server off this machine.

Every test and every tool in `src/builder/` that starts the sidecar sets the
switch, so a test that forgets its script, or loses it on the way (the key
store removes CALIPER_AGENT_TEST_SCRIPT at spawn, by design), cannot send a
request to a real API. With the switch on:

- **No real client to another host.** `check_provider()` refuses, before any
  client or transport exists, the Claude provider on its real transport and
  an OpenAI-compatible provider whose base URL is not a loopback address. Only
  the scripted model, the project's stub transport and servers on loopback run. The
  refusal is an `OfflineError` with the code `agent.offline`, which the
  sidecar answers to `session/new` and `session/prompt` as a JSON-RPC error.
- **No socket to another host either.** `install_socket_guard()` wraps
  `socket.getaddrinfo`, `gethostbyname`, `gethostbyname_ex` and the connect
  `sendto` and `sendmsg` methods of `socket.socket`, so a name other than a loopback name
  is never resolved and an address other than a loopback address is never
  connected or sent to, whatever library asks. A host given as text or as bytes
  is checked; anything the guard cannot classify (another type, bytes that are
  not ASCII, a socket family other than IPv4, IPv6 and Unix) is refused. Each
  attempt fails with `OfflineError` and is noted on stderr.
- **No proxy.** `enter_offline()` removes the proxy variables (`HTTP_PROXY`,
  `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, in both cases) from the process's
  environment before any client is built, since a proxy on loopback would pass
  the guard and forward a request off the machine, and `trust_env()` is false,
  so every HTTP client the sidecar builds ignores the environment's proxy,
  `.netrc` and certificate settings. Outside the switch the clients keep
  trusting the environment, so that a person behind a proxy can reach the
  provider.
- **Only the project's stub stands in for a provider.** `is_stub_transport()`
  accepts exactly `stub_transport.StubTransport`, which never opens a socket,
  and no other transport, a subclass of it included.

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


PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy",
             "no_proxy")


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


def trust_env() -> bool:
    """Whether the sidecar's HTTP clients read the environment's proxy and certificate settings: not offline."""
    return not offline()


def is_stub_transport(transport) -> bool:
    """True only for the project's own stub transport (stub_transport.py), never for another transport."""
    if transport is None:
        return False
    from .stub_transport import StubTransport
    return type(transport) is StubTransport


def enter_offline() -> bool:
    """With the switch on: removes the proxy variables and installs the socket guard. True when it is on."""
    if not offline():
        return False
    for name in PROXY_ENV:
        os.environ.pop(name, None)
    install_socket_guard()
    return True


def check_provider(kind: str, *, base_url: str | None = None, transport=None) -> None:
    """Raises OfflineError when the switch is on and the provider would reach another host.

    `kind` is "claude" or "openai-compatible". The Claude route passes only
    with the project's stub transport (`is_stub_transport`), which never opens
    a socket; an OpenAI-compatible server passes only on a loopback address,
    whatever transport it is given."""
    if not offline():
        return
    if kind == "claude":
        if is_stub_transport(transport):
            return
        raise OfflineError(f"{OFFLINE_ENV} is set, so the sidecar does not reach the Claude API; a test must give "
                           "the scripted model (CALIPER_AGENT_TEST_SCRIPT) or a loopback server.")
    host = urlsplit(base_url or "").hostname
    if not is_loopback_host(host):
        raise OfflineError(f"{OFFLINE_ENV} is set, so the sidecar reaches only a model server on a loopback "
                           f"address, and not {host or 'this base URL'}.")


def _host_text(host) -> str | None:
    """A host as text, or None when it is of a type or an encoding the guard cannot classify."""
    if isinstance(host, str):
        return host
    if isinstance(host, (bytes, bytearray, memoryview)):
        try:
            return bytes(host).decode("ascii")
        except UnicodeDecodeError:
            return None
    return None


def _refuse(what: str) -> None:
    sys.stderr.write(f"caliper-agent: {OFFLINE_ENV} refused {what}\n")
    raise OfflineError(f"{OFFLINE_ENV} is set: {what} is refused.")


def _check_address(family, address) -> None:
    if family == getattr(socket, "AF_UNIX", object()):
        return
    if family not in (socket.AF_INET, socket.AF_INET6):
        _refuse(f"a socket of family {family!r}")
    host = _host_text(address[0]) if isinstance(address, tuple) and address else None
    if host is None:
        _refuse(f"a connection to an address it cannot read ({type(address).__name__})")
    if not is_loopback_host(host):
        _refuse(f"a connection to {host}")


def _check_name(host) -> None:
    if host is None:
        return   # the local host
    name = _host_text(host)
    if name is None:
        _refuse(f"resolving a name it cannot read ({type(host).__name__})")
    if not is_loopback_host(name):
        _refuse(f"resolving {name}")


_installed = False


def install_socket_guard() -> None:
    """Refuses every name resolution and connection off loopback, for the life of the process."""
    global _installed
    if _installed:
        return
    _installed = True
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_sendmsg = getattr(socket.socket, "sendmsg", None)

    def getaddrinfo(host, *args, **kwargs):
        _check_name(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        _check_name(host)
        return real_gethostbyname(host)

    def gethostbyname_ex(host):
        _check_name(host)
        return real_gethostbyname_ex(host)

    def connect(self, address):
        _check_address(self.family, address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _check_address(self.family, address)
        return real_connect_ex(self, address)

    def sendto(self, data, *args):
        if args:   # sendto(data, address) or sendto(data, flags, address)
            _check_address(self.family, args[-1])
        return real_sendto(self, data, *args)

    def sendmsg(self, buffers, *args):
        if len(args) >= 3:   # sendmsg(buffers, ancdata, flags, address)
            _check_address(self.family, args[2])
        return real_sendmsg(self, buffers, *args)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    if real_sendmsg is not None:
        socket.socket.sendmsg = sendmsg
