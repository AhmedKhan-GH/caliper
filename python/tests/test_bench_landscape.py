"""How python/bench/bench_landscape.py confirms its viewer, without a
measurement: the drain viewer fails loudly when it cannot reach the server, and
wait_subscribed returns once the server holds the viewer's subscription to the
scene, and raises when the viewer ends first."""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("websockets")

import caliper  # noqa: E402
from caliper import _scene  # noqa: E402
from caliper._net_scene import route_of  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "bench" / "bench_landscape.py"


def bench():
    spec = importlib.util.spec_from_file_location("bench_landscape", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def closed_port() -> int:
    """A loopback port that nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "0")
    s = caliper.serve(port=0)
    yield s
    s.close()
    for name in list(_scene.SCENES):
        caliper.remove(name)


def test_the_drain_viewer_fails_when_it_cannot_reach_the_server():
    url = f"ws://127.0.0.1:{closed_port()}/v3/stream"
    p = subprocess.run([sys.executable, str(BENCH), "--drain", url], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(BENCH.parents[1])})
    assert p.returncode != 0, p.stdout + p.stderr


def test_wait_subscribed_returns_once_the_viewer_holds_the_scene(server):
    b = bench()
    caliper.scene("landscape", up="+z", units="").points("current", torch.zeros(1, 3))
    assert not route_of(server.ring).watched("landscape")
    view = b.viewer(server.stream_url)
    try:
        b.wait_subscribed(server, view, timeout=60)
        assert route_of(server.ring).watched("landscape")
    finally:
        server.close()
        assert view.wait(30) == 0                         # a closed server ends it quietly


def test_wait_subscribed_raises_when_the_viewer_ends_first(server):
    b = bench()
    caliper.scene("landscape", up="+z", units="").points("current", torch.zeros(1, 3))
    view = b.viewer(f"ws://127.0.0.1:{closed_port()}/v3/stream")   # never reaches the server
    with pytest.raises(SystemExit, match="ended with exit status"):
        b.wait_subscribed(server, view, timeout=60)
    assert view.returncode != 0
