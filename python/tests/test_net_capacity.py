"""Bounded loopback evidence for the internal native capacity interface."""
from types import SimpleNamespace
import json

import pytest

from caliper import _wire as w
from test_net_scene import RawViewer

nio_module = pytest.importorskip("caliper._netio")


@pytest.fixture
def connection():
    identity = {"id": "capacity-test", "script": "test", "pid": 1, "start_unix_ms": 1}
    token = "t" * 32
    nio = nio_module.Server(0, token, json.dumps(identity), "{}", [], 2, 16,
                            5000, 60000, 120000)
    viewer = RawViewer(SimpleNamespace(port=nio.port, token=token))
    viewer.s.settimeout(5)
    try:
        viewer.send(w.Hello("viewer", [w.CAP_RESOURCES]))
        viewer.expect(lambda msg: isinstance(msg, w.Views), "VIEWS")
        events = nio.control(0)
        conn, = [conn for kind, conn, _ in events if kind == "attached"]
        yield nio, viewer, conn
    finally:
        viewer.close()
        nio.close()


def queued_resources(nio, conn):
    # Two bounded frames exceed the loopback receive buffer without a send loop.
    data = bytes(6 << 20)
    frame = w.encode(w.Resource("capacity", "values", 1, len(data) // 4, 4, data))
    assert nio.send(conn, frame)
    assert nio.send(conn, frame)
    return len(frame)


def test_native_capacity_delivery_and_coalescing(connection):
    nio, viewer, conn = connection
    size = queued_resources(nio, conn)
    assert not nio.watch_capacity(conn, size // 2, 1)
    assert not nio.watch_capacity(conn, size // 2, 2)
    first = viewer.expect(lambda msg: isinstance(msg, w.Resource), "RESOURCE")
    assert first.count == (6 << 20) // 4
    events = nio.control(3000)
    assert events == [("capacity", conn, b"2")]
    assert nio.control_bytes(conn) <= size // 2
    viewer.expect(lambda msg: isinstance(msg, w.Resource), "RESOURCE")
    assert nio.control(0) == []
    assert nio.watch_capacity(conn, 0, 3)
    assert nio.control(0) == []


@pytest.mark.parametrize("after_drain", [False, True])
def test_native_capacity_cancel_and_close(connection, after_drain):
    nio, viewer, conn = connection
    queued_resources(nio, conn)
    assert not nio.watch_capacity(conn, 0, 1)
    if not after_drain:
        assert nio.watch_capacity(conn, -1, 2)
    viewer.expect(lambda msg: isinstance(msg, w.Resource), "RESOURCE")
    viewer.expect(lambda msg: isinstance(msg, w.Resource), "RESOURCE")
    if after_drain:
        assert nio.watch_capacity(conn, -1, 2)
    assert nio.control(0) == []
    queued_resources(nio, conn)
    assert not nio.watch_capacity(conn, 0, 3)
    viewer.close()
    events = nio.control(3000)
    assert events == [("closed", conn, b"")]
    assert nio.watch_capacity(conn, 0, 4)
    assert nio.control(0) == []


def test_native_detach_cancels_an_observation(connection):
    nio, viewer, conn = connection
    queued_resources(nio, conn)
    assert not nio.watch_capacity(conn, 0, 1)
    viewer.send(w.Attach(None))
    assert nio.control(3000) == [("detached", conn, b"")]
    viewer.expect(lambda msg: isinstance(msg, w.Views), "VIEWS")
    assert nio.control(0) == []
    assert nio.watch_capacity(conn, 0, 2)
