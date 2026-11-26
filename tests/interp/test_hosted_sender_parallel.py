"""CPU ownership for control callbacks and asynchronous sender work."""
import asyncio
from types import SimpleNamespace

import pytest

from caliper import _controls, _net, _net_controls, _net_ring as N, _wire
from .test_hosted_sender_overlap import sender_cpu


def measured_ring():
    ring = N.NetRing(engine=object())
    account = N._SenderAccount()
    ring._sender.watch(account)
    return ring, account


@pytest.mark.parametrize("nested", [False, True], ids=["scheduled", "native-pump"])
def test_control_flush_owns_encoding_once(monkeypatch, sender_cpu, nested):
    ring, account = measured_ring()
    producer = _net._Producer(ring, {}, None)
    link = _net.Link(producer)
    link.attached = True
    producer.links.add(link)
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    encode = _wire.encode_frame

    def encoded(*args, **kwargs):
        sender_cpu.value += .00002
        return encode(*args, **kwargs)

    monkeypatch.setattr(_wire, "encode_frame", encoded)

    async def run():
        server = SimpleNamespace(ring=ring, producer=producer, _loop=asyncio.get_running_loop())
        port = _net_controls.ControlsPort(server, True, False)
        try:
            reply = {"request_id": 1, "op": "pause", "status": "ok"}
            if nested:
                with ring.sender_cost():
                    sender_cpu.value += .00001
                    port._replies.append((link, reply))
                    port.flush()
                    sender_cpu.value += .00003
            else:
                port.reply(link, reply)
                await asyncio.sleep(0)
            assert len(link.control) == 1
        finally:
            port.detach()

    asyncio.run(run())
    expected = .00006 if nested else .00002
    assert account.sample() == pytest.approx(expected)
    assert ring.publish_s == pytest.approx(expected)
    assert not account.active


@pytest.mark.parametrize("direction", ["reader", "writer"])
def test_listener_owns_cpu_but_not_suspended_task_work(monkeypatch, sender_cpu, direction):
    ring, account = measured_ring()
    producer = _net._Producer(ring, {}, None)
    link = _net.Link(producer)
    listener = _net._Listener(SimpleNamespace(ring=ring))
    packet = _wire.encode(_wire.Ping(7))
    operation = _wire.decode if direction == "reader" else _wire.encode
    calls = []
    decoded = asyncio.Event()

    def codec(*args, **kwargs):
        calls.append(1)
        decoded.set()
        sender_cpu.value += .00002
        return operation(*args, **kwargs)

    monkeypatch.setattr(_wire, "decode" if direction == "reader" else "encode", codec)

    async def run():
        waiting, release = asyncio.Event(), asyncio.Event()

        received = False

        async def recv():
            nonlocal received
            if received:
                await asyncio.Future()
            received = True
            waiting.set()
            await release.wait()
            return packet

        async def send(frame):
            assert _wire.decode(frame).nonce == 7
            sender_cpu.value += .00003
            waiting.set()
            await release.wait()

        if direction == "writer":
            link.pong = 7
        ws = SimpleNamespace(recv=recv, send=send)
        task = asyncio.create_task(getattr(listener, "_" + direction)(ws, link))
        try:
            await asyncio.wait_for(waiting.wait(), 1.)
            # Unrelated work on the event loop while this sender is suspended.
            sender_cpu.value += .001
            if direction == "reader":
                release.set()
                await asyncio.wait_for(decoded.wait(), 1.)
            assert len(calls) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    expected = .00002 if direction == "reader" else .00005
    assert account.sample() == pytest.approx(expected)
    assert ring.publish_s == pytest.approx(expected)
    assert not account.active


@pytest.mark.parametrize("outcome", ["return", "error", "cancel"])
def test_coroutine_segments_keep_nested_cpu_and_cleanup(sender_cpu, outcome):
    ring, account = measured_ring()

    async def run():
        ready, resume = asyncio.Event(), asyncio.Event()

        async def inner():
            try:
                sender_cpu.value += .00002
                ready.set()
                await resume.wait()
                sender_cpu.value += .00003
                if outcome == "error":
                    raise ValueError("sender failure")
                # A bare yield resumes without a Future.
                await asyncio.sleep(0)
                return 7
            finally:
                sender_cpu.value += .00005

        async def outer():
            sender_cpu.value += .00001
            return await ring.sender_task(inner())

        task = asyncio.ensure_future(ring.sender_task(outer()))
        await asyncio.wait_for(ready.wait(), 1.)
        sender_cpu.value += .001
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            resume.set()
            if outcome == "error":
                with pytest.raises(ValueError, match="sender failure"):
                    await task
            else:
                assert await task == 7

    asyncio.run(run())
    expected = .00008 if outcome == "cancel" else .00011
    assert ring.publish_s == pytest.approx(expected)
    assert account.sample() == pytest.approx(expected)
    assert not account.active


def test_native_loop_retains_nested_flush_cost(monkeypatch, sender_cpu):
    from caliper import _net_native

    ring, account = measured_ring()
    producer = _net._Producer(ring, {}, None)
    link = _net.Link(producer)
    link.attached, link.conn = True, 1
    producer.links.add(link)
    server = SimpleNamespace(ring=ring, producer=producer, closed=False, counters={"pump_errors": 0})
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    packets, pumps = [], []

    def send(conn, frame):
        assert conn == 1
        packets.append(frame)
        sender_cpu.value += .00003

    def pump():
        pumps.append(1)
        if len(pumps) == 1:
            return False  # The initial view list precedes the measured loop.
        sender_cpu.value += .00002
        server.closed = True
        return True

    def control(wait):
        sender_cpu.value += .001  # The blocking native poll stays outside the domain.
        return []

    server.nio = SimpleNamespace(send=send, control=control, wake=lambda: None)
    port = _net_controls.ControlsPort(server, True, True)
    producer.controls = port
    producer.pump_once, producer.wait_ms = pump, lambda: 0
    try:
        port.reply(link, {"request_id": 1, "op": "pause", "status": "ok"})
        _net_native.NativeServer._run(server)
    finally:
        port.detach()
    assert len(packets) == 1
    assert server.counters["pump_errors"] == 0
    assert ring.publish_s == pytest.approx(.00005)
    assert account.sample() == pytest.approx(.00005)
    assert not account.active


@pytest.mark.parametrize("direction", ["reader", "writer"])
def test_listener_reports_cpu_from_a_real_thread(monkeypatch, direction):
    import threading
    import time

    ring, account = measured_ring()
    producer = _net._Producer(ring, {}, None)
    link = _net.Link(producer)
    listener = _net._Listener(SimpleNamespace(ring=ring))
    packet = _wire.encode(_wire.Ping(7))
    operation = _wire.decode if direction == "reader" else _wire.encode
    codec_cpu, failures = [], []

    def codec(*args, **kwargs):
        start = time.thread_time()
        result = operation(*args, **kwargs)
        codec_cpu.append(time.thread_time() - start)
        return result

    monkeypatch.setattr(_wire, "decode" if direction == "reader" else "encode", codec)

    async def run():
        done = asyncio.Event()
        received = False

        async def recv():
            nonlocal received
            if received:
                done.set()
                await asyncio.Future()
            received = True
            return packet

        async def send(frame):
            assert _wire.decode(frame).nonce == 7
            done.set()
            await asyncio.Future()

        link.pong = 7 if direction == "writer" else None
        task = asyncio.create_task(getattr(listener, "_" + direction)(
            SimpleNamespace(recv=recv, send=send), link))
        try:
            await asyncio.wait_for(done.wait(), 1.)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    def worker():
        try:
            asyncio.run(run())
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(5.)
    assert not thread.is_alive()
    assert not failures
    assert len(codec_cpu) == 1 and codec_cpu[0] >= 0.
    if codec_cpu[0] == 0.:
        pytest.skip("The thread CPU clock did not resolve this bounded codec call.")
    # A codec is a strict subset of sender work, independent of timer placement.
    assert ring.publish_s >= codec_cpu[0]
    assert account.sample() >= codec_cpu[0]
    assert not account.active
