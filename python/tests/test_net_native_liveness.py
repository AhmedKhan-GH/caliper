"""CPU checks for native heartbeats, control replies, and close diagnostics."""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import threading

import pytest

from caliper import _controls, _wire as w

nio_module = pytest.importorskip("caliper._netio")
pytest.importorskip("websockets")
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


IDENTITY = {"id": "heartbeat-test", "script": "test", "pid": 1, "start_unix_ms": 1}
CAPS = [w.CAP_CONTROLS, w.CAP_INTERPRETATION]


async def welcome(ws):
    hello = w.decode(await ws.recv())
    assert isinstance(hello, w.Hello) and hello.role == "producer"
    await ws.send(w.encode(w.Welcome("hub", CAPS, session=IDENTITY, control=True)))


async def receive(ws, cls, predicate=lambda message: True):
    while True:
        message = w.decode(await ws.recv())
        if isinstance(message, cls) and predicate(message):
            return message
        if isinstance(message, w.Ping):
            await ws.send(w.encode(w.Pong(message.nonce)))


def test_native_heartbeats_and_replies_survive_blocked_control_handler(monkeypatch):
    from caliper._net_native import NativeServer

    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    monkeypatch.setattr(w, "PING_INTERVAL_MS", 100)
    monkeypatch.setattr(w, "IDLE_TIMEOUT_MS", 500)
    producer = NativeServer(port=None)
    producer.nio.enable_interpretation(True)
    release = threading.Event()
    entered = threading.Event()
    port = producer.producer.controls

    def control(link, meta, *, acknowledge):
        rid = meta["request_id"]
        if rid == 1:
            acknowledge(w.ControlReply(rid, "interpret", "queued").to_meta())
            entered.set()
            assert release.wait(5), "The test did not release the control handler."
            # Completion can precede the return from submit().
            port.reply(link, w.ControlReply(rid, "interpret", "ok", result={}).to_meta())
            return None
        return w.ControlReply(rid, "interpret", "ok", result={}).to_meta()

    monkeypatch.setattr(port.route, "on_control", control)

    async def check():
        done = asyncio.get_running_loop().create_future()

        async def hub(ws):
            try:
                await welcome(ws)
                await receive(ws, w.Controls)
                request = {"version": 1, "method": "feature.dictionary", "params": {}}
                await ws.send(w.encode(w.Control(1, "interpret", interpretation=request)))
                first = await receive(ws, w.ControlReply)
                assert (first.request_id, first.status) == (1, "queued")
                assert entered.is_set()
                await ws.send(w.encode(w.Control(2, "interpret", interpretation=request)))
                # Six native heartbeat periods exceed the test's idle limit.
                # Each period also checks the native response to a hub PING.
                for nonce in range(6):
                    ping = await receive(ws, w.Ping)
                    await ws.send(w.encode(w.Pong(ping.nonce)))
                    await ws.send(w.encode(w.Ping(100 + nonce)))
                    await receive(ws, w.Pong, lambda message: message.nonce == 100 + nonce)
                    assert not release.is_set()
                assert producer.hub.stats()["connects"] == 1
                release.set()
                replies = [await receive(ws, w.ControlReply) for _ in range(2)]
                assert sorted((r.request_id, r.status) for r in replies) == [(1, "ok"), (2, "ok")]
                assert producer.counters["control_errors"] == 0
                assert producer.counters["pump_errors"] == 0
                done.set_result(None)
                await receive(ws, w.Bye)
            except Exception as exc:
                if not done.done():
                    done.set_exception(exc)
                else:
                    raise

        async with serve(hub, "127.0.0.1", 0, ping_interval=None) as server:
            producer.connect_hub(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream")
            try:
                await asyncio.wait_for(done, 5)
            finally:
                release.set()
                await asyncio.to_thread(producer.close)

    try:
        asyncio.run(check())
    finally:
        release.set()
        producer.close()


@pytest.mark.parametrize("local", [False, True], ids=["hub-close", "producer-idle"])
def test_native_close_code_and_direction_survive_reconnect(local):
    nio = nio_module.Server(-1, "t" * 32, json.dumps(IDENTITY), "{}", [], 4, 16,
                            1000, 100, 300)
    initial = nio.upstream()
    assert initial["last_close_code"] == 0
    assert initial["last_close_local"] is False

    async def check():
        reconnected = asyncio.Event()
        attempts = 0

        async def hub(ws):
            nonlocal attempts
            attempts += 1
            attempt = attempts
            await welcome(ws)
            await receive(ws, w.Views)
            if attempt == 1:
                if local:
                    # Ignore wire PING. The producer must enforce its idle limit.
                    try:
                        while True:
                            await ws.recv()
                    except ConnectionClosed as exc:
                        assert exc.rcvd.code == w.CLOSE_IDLE_TIMEOUT
                else:
                    await ws.close(w.CLOSE_IDLE_TIMEOUT, "idle")
            else:
                reconnected.set()
                await receive(ws, w.Bye)

        async with serve(hub, "127.0.0.1", 0, ping_interval=None) as server:
            nio.connect_upstream(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream",
                                 json.dumps(IDENTITY))
            try:
                await asyncio.wait_for(reconnected.wait(), 5)
                state = nio.upstream()
                assert state["connected"] and state["connects"] == 2
                assert state["last_close_code"] == w.CLOSE_IDLE_TIMEOUT
                assert state["last_close_local"] is local
            finally:
                await asyncio.to_thread(nio.disconnect_upstream)
            state = nio.upstream()
            assert state["last_close_code"] == w.CLOSE_GOING_AWAY
            assert state["last_close_local"] is True

    try:
        asyncio.run(check())
    finally:
        nio.close()


async def gil_probe_hub():
    """A separate process answers heartbeats while the parent holds the GIL."""
    done = asyncio.get_running_loop().create_future()

    async def hub(ws):
        try:
            await welcome(ws)
            await receive(ws, w.Views)
            await ws.send(w.encode(w.Control(1, "pause")))
            for nonce in range(6):
                ping = await receive(ws, w.Ping)
                await ws.send(w.encode(w.Pong(ping.nonce)))
                await ws.send(w.encode(w.Ping(100 + nonce)))
                await receive(ws, w.Pong, lambda message: message.nonce == 100 + nonce)
            done.set_result(None)
        except Exception as exc:
            if not done.done():
                done.set_exception(exc)

    async with serve(hub, "127.0.0.1", 0, ping_interval=None, close_timeout=1) as server:
        print(server.sockets[0].getsockname()[1], flush=True)
        await asyncio.wait_for(done, 5)


@pytest.mark.skipif(os.name != "posix", reason="The GIL probe uses POSIX waitpid.")
def test_native_heartbeats_do_not_need_the_gil():
    nio = nio_module.Server(-1, "t" * 32, json.dumps(IDENTITY), "{}", [], 4, 16,
                            1000, 100, 500)
    nio.enable_controls(True)
    child = subprocess.Popen([sys.executable, __file__, "--gil-probe"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        port = child.stdout.readline().strip()
        assert port.isdecimal(), child.stderr.read()
        # PyDLL retains the GIL for the whole wait. The child bounds its own run.
        waitpid = ctypes.PyDLL(None).waitpid
        waitpid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        waitpid.restype = ctypes.c_int
        status = ctypes.c_int()
        nio.connect_upstream(f"ws://127.0.0.1:{port}/stream", json.dumps(IDENTITY))
        assert waitpid(child.pid, ctypes.byref(status), 0) == child.pid
        child.returncode = os.waitstatus_to_exitcode(status.value)
        assert child.returncode == 0, child.stderr.read()
        messages = nio.control(0)
        requests = [w.decode(frame) for kind, _, frame in messages if kind == "msg"]
        assert len(requests) == 1 and requests[0].request_id == 1
        assert nio.upstream()["connects"] == 1
    finally:
        if child.poll() is None:
            child.terminate()
        child.communicate(timeout=5)
        nio.close()


if __name__ == "__main__" and sys.argv[1:] == ["--gil-probe"]:
    asyncio.run(gil_probe_hub())
