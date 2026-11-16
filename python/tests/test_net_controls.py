"""The controls of a running script over caliper.serve() (wire protocol 3.1),
on both senders, with a websockets viewer: the capabilities and WELCOME's
control flag, CONTROLS after VIEWS, pause, resume and step of a loop that
calls caliper.step(), parameter values with clamping, actions with their two
replies, a read-only server, a script without a checkpoint, a flood of
requests, a 3.0 viewer, linked selection from sampled points to row ids with
details, the last step's snapshot of a paused run, and the cost of
caliper.step() with a viewer attached."""
from __future__ import annotations

import asyncio
import threading
import time
import timeit

import pytest
import torch

import caliper
from caliper import _controls as C
from caliper import _wire as w

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

try:
    from caliper import _netio  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False
MODES = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not NATIVE, reason="caliper._netio is not built (python scripts/build_netio.py)"))]
CAPS = [w.CAP_CONTROLS, w.CAP_SELECTION]


@pytest.fixture(params=MODES)
def mode(request, monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1" if request.param == "native" else "0")
    return request.param


@pytest.fixture
def ctl(monkeypatch):
    c = C.Controls()
    monkeypatch.setattr(C, "CONTROLS", c)
    monkeypatch.setattr(caliper, "_CTL", c)
    monkeypatch.setattr(C, "WAIT_POLL_S", 0.01)
    yield c
    with c.lock:
        c.state = C.RUN_RUNNING
        c.cond.notify_all()


@pytest.fixture
def make_server(mode, ctl):
    made = []

    def make(control=True):
        s = caliper.serve(port=0, control=control)
        assert (type(s).__name__ == "NativeServer") == (mode == "native")
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


@pytest.fixture
def server(make_server):
    return make_server()


def run(coro, timeout=60.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Viewer:
    """A viewer that announces ``caps``; it keeps every message it decodes."""

    def __init__(self, server, caps=CAPS):
        self.url = server.stream_url
        self.caps = caps
        self.inbox: list = []
        self.seen: list = []                    # every message, in order
        self._rid = 0

    async def __aenter__(self):
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None,
                                close_timeout=1).__aenter__()
        await self.send(w.Hello("viewer", list(self.caps), {"name": "test", "version": "0"}))
        self.welcome = await self.expect(w.Welcome)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send(self, msg):
        await self.ws.send(w.encode(msg))

    async def expect(self, cls, pred=lambda m: True, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            for i, m in enumerate(self.inbox):
                if isinstance(m, cls) and pred(m):
                    return self.inbox.pop(i)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {cls.__name__}; got {[type(m).__name__ for m in self.inbox]}")
            try:
                m = w.decode(await asyncio.wait_for(self.ws.recv(), min(left, 0.05)))
            except asyncio.TimeoutError:
                continue
            self.inbox.append(m)
            self.seen.append(m)

    async def drain(self, seconds=0.2):
        try:
            await self.expect(type(None), timeout=seconds)
        except TimeoutError:
            pass

    async def control(self, op, **kw):
        """Send a CONTROL; returns its first CONTROL_REPLY."""
        self._rid += 1
        rid = self._rid
        await self.send(w.Control(rid, op, **kw))
        return await self.expect(w.ControlReply, lambda m: m.request_id == rid)

    async def state(self, pred, timeout=10.0):
        return await self.expect(w.Controls, pred, timeout)


class Loop:
    """A training loop on a thread: caliper.step(), then ``body(i)``."""

    def __init__(self, body=None, delay=0.002):
        self.count = 0
        self.stop = False
        self.body = body
        self.delay = delay
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            while not self.stop:
                caliper.step()
                if self.body is not None:
                    self.body(self.count)
                self.count += 1
                time.sleep(self.delay)
        except BaseException as e:            # noqa: BLE001
            self.error = e

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop = True
        caliper._CTL._release()
        self.thread.join(5)
        assert self.error is None, self.error


async def until(fn, timeout=10.0):
    end = time.monotonic() + timeout
    while not fn():
        if time.monotonic() > end:
            raise TimeoutError
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# the handshake and the state a viewer receives
# ---------------------------------------------------------------------------
def test_capabilities_welcome_and_state_after_views(server, ctl):
    caliper.param("lr", 2e-3, min=1e-4, max=1e-1, log=True)

    async def go():
        async with Viewer(server) as v:
            assert set(CAPS) <= set(v.welcome.capabilities)
            assert v.welcome.control is True
            sessions = await v.expect(w.Sessions)
            assert set(CAPS) <= set(sessions.sessions[0]["capabilities"])
            await v.expect(w.Views)
            c = await v.state(lambda m: True)
            assert c.run == {"state": "none"} and c.params[0]["name"] == "lr"
            kinds = [type(m) for m in v.seen]
            assert kinds.index(w.Views) < kinds.index(w.Controls)
    run(go())


def test_a_30_viewer_is_served_as_before(server, ctl):
    caliper.param("lr", 2e-3)
    z = torch.zeros(10, 2)

    async def go():
        async with Viewer(server, caps=[]) as v:
            assert v.welcome.control is None
            with Loop(body=lambda i: caliper.points(z, name="embedding", items="test")):
                views = await v.expect(w.Views, lambda m: m.views)
                assert "items" not in str(views.views)
                await v.send(w.Control(1, "pause"))       # a 3.0 viewer has no business here
                err = await v.expect(w.Error)
                assert err.code == w.ERR_BAD_REQUEST and err.request_id == 1 and not err.fatal
                await v.send(w.Ping(5))
                await v.expect(w.Pong, lambda m: m.nonce == 5)
                await v.drain()
                assert not any(isinstance(m, (w.Controls, w.Selection, w.ControlReply))
                               for m in v.seen)
                async with Viewer(server) as v2:           # a 3.1 viewer gets the item spaces
                    views = await v2.expect(w.Views, lambda m: m.views)
                    dims = views.views[0]["sources"][0]["dims"]
                    assert dims[-1].get("items") == "test"
    run(go())


def test_a_read_only_server(make_server, ctl):
    s = make_server(control=False)
    p = caliper.param("lr", 2e-3)
    caliper.step()
    assert s.control is False

    async def go():
        async with Viewer(s) as v:
            assert v.welcome.control is False
            await v.state(lambda m: True)                   # still shown
            for op, kw in (("pause", {}), ("set", {"name": "lr", "value": 0.5}),
                           ("run", {"name": "x"})):
                r = await v.control(op, **kw)
                assert r.status == "error" and r.error["code"] == w.ERR_FORBIDDEN
    run(go())
    assert p.value == 2e-3 and ctl.state == "running"


# ---------------------------------------------------------------------------
# pause, resume, step, values, actions
# ---------------------------------------------------------------------------
def test_without_a_checkpoint(server, ctl, monkeypatch):
    monkeypatch.setattr(C, "ACTION_WAIT_S", 0.3)
    p = caliper.param("lr", 2e-3, min=1e-4, max=1e-1)
    caliper.action("save", lambda: "saved")

    async def go():
        async with Viewer(server) as v:
            r = await v.control("pause")
            assert r.status == "error" and r.error["code"] == w.ERR_UNSUPPORTED
            r = await v.control("set", name="lr", value=0.01)
            assert r.status == "ok" and p.value == 0.01           # at once: no checkpoint
            r = await v.control("run", name="save")
            assert r.status == "queued"
            last = await v.expect(w.ControlReply, lambda m: m.request_id == r.request_id)
            assert last.status == "error" and last.error["code"] == w.ERR_TIMEOUT
    run(go())


def test_pause_step_and_resume_a_training_loop(server, ctl):
    async def go():
        async with Viewer(server) as v:
            with Loop() as loop:
                await v.state(lambda m: m.run["state"] == "running")
                assert (await v.control("pause")).status == "ok"
                paused = await v.state(lambda m: m.run["state"] == "paused")
                n = loop.count
                await asyncio.sleep(0.2)
                assert loop.count == n and paused.run["step"] == ctl.n
                assert (await v.control("step", count=3)).status == "ok"
                await v.state(lambda m: m.run["state"] == "stepping")
                again = await v.state(lambda m: m.run["state"] == "paused")
                await asyncio.sleep(0.2)
                assert loop.count == n + 3 and again.run["step"] == paused.run["step"] + 3
                assert (await v.control("resume")).status == "ok"
                await v.state(lambda m: m.run["state"] == "running")
                await until(lambda: loop.count > n + 10)
    run(go())


def test_values_are_clamped_echoed_and_used(server, ctl):
    lr = caliper.param("lr", 2e-3, min=1e-4, max=1e-1, log=True)
    used = []

    async def go():
        async with Viewer(server) as v:
            with Loop(body=lambda i: used.append(lr.value)):
                await v.state(lambda m: m.run["state"] == "running")
                r = await v.control("set", name="lr", value=0.5)
                assert r.status == "ok" and r.value == 0.1         # clamped, as applied
                await v.state(lambda m: m.params[0]["value"] == 0.1 and "pending" not in m.params[0])
                await until(lambda: used and used[-1] == 0.1)
                r = await v.control("set", name="lr", value="fast")
                assert r.error["code"] == w.ERR_BAD_VALUE
                r = await v.control("set", name="nope", value=1)
                assert r.error["code"] == w.ERR_UNKNOWN_CONTROL
    run(go())
    assert lr.value == 0.1


def test_actions_reply_twice_and_report_errors(server, ctl):
    ran = []

    def save():
        ran.append(threading.current_thread().name)
        return "saved ckpt.pt"

    caliper.action("save checkpoint", save)
    caliper.action("boom", lambda: 1 / 0)

    async def go():
        async with Viewer(server) as v:
            with Loop() as loop:
                await v.state(lambda m: m.run["state"] == "running")
                r = await v.control("run", name="save checkpoint")
                assert r.status == "queued"
                done = await v.expect(w.ControlReply, lambda m: m.request_id == r.request_id)
                assert done.status == "ok" and done.message == "saved ckpt.pt"
                assert ran == [loop.thread.name]
                r = await v.control("run", name="boom")
                with pytest.warns(RuntimeWarning):
                    failed = await v.expect(w.ControlReply, lambda m: m.request_id == r.request_id)
                assert failed.error["code"] == w.ERR_ACTION_FAILED
                assert "ZeroDivisionError" in failed.error["message"]
                busy = [m for m in v.seen if isinstance(m, w.Controls)
                        and any(a["running"] or a["queued"] for a in m.actions)]
                assert busy                                       # every viewer sees it busy
    run(go())


def test_a_flood_of_requests(server, ctl):
    p = caliper.param("p", 0, min=0, max=10 ** 6)
    n = 2000

    async def go():
        async with Viewer(server) as v:
            with Loop():
                await v.state(lambda m: m.run["state"] == "running")
                for i in range(n):
                    await v.send(w.Control(i + 1, "set", name="p", value=i))
                replies = []
                while len(replies) < n:
                    replies.append(await v.expect(w.ControlReply))
                statuses = {r.status for r in replies}
                assert statuses <= {"ok", "error"}
                assert all(r.error["code"] == w.ERR_LIMIT for r in replies if r.status == "error")
                last_ok = max(r.value for r in replies if r.status == "ok")
                await until(lambda: p.value == last_ok and p.pending is None)
                await v.send(w.Ping(9))
                await v.expect(w.Pong, lambda m: m.nonce == 9)
                await v.drain(0.3)
                controls = sum(isinstance(m, w.Controls) for m in v.seen)
                print(f"{n} set requests: {controls} CONTROLS sent")
                assert controls < n                              # coalesced
    run(go())


# ---------------------------------------------------------------------------
# linked selection
# ---------------------------------------------------------------------------
def test_selection_maps_sampled_points_to_row_ids_with_details(server, ctl):
    z = torch.randn(1000, 2)
    caliper.items("test", details={"label": torch.arange(1000) % 6})

    def body(i):
        caliper.points(z, name="embedding", items="test")

    async def go():
        async with Viewer(server) as v:
            with Loop(body=body):
                views = await v.expect(w.Views, lambda m: m.views)
                assert views.views[0]["sources"][0]["dims"][0]["items"] == "test"
                await v.send(w.Subscribe(1, "embedding", 1, None, {"budget": {"max_points": 4}}))
                rep = await v.expect(w.ViewReply)
                assert rep.data_view["budget"]["max_points"] == 4
                snap = await v.expect(w.Snapshot)
                (axis,) = snap.axes
                assert (axis["count"], axis["size"]) == (1000, 4)
                # only the indices in the sent sample: the producer maps them
                await v.send(w.Event("embedding", "select", {"indices": [1, 3]}, sub=1,
                                     seq=snap.seq))
                sel = await v.expect(w.Selection)
                assert sel.items == [250, 750] and sel.view == "embedding"
                assert sel.details == {"fields": ["label"], "items": [250, 750],
                                       "values": [[250 % 6], [750 % 6]]}
                # a 3.1 viewer names the space and the items itself
                await v.send(w.Event("embedding", "select", {"indices": [0], "row_ids": [0],
                                                             "space": "test", "items": [7, 3]}))
                sel = await v.expect(w.Selection, lambda m: m.items == [3, 7])
                assert ctl.spaces["test"].selection == [3, 7]
                # a late viewer gets the selection after CONTROLS
                async with Viewer(server) as late:
                    await late.expect(w.Selection, lambda m: m.items == [3, 7])
                    kinds = [type(m) for m in late.seen]
                    assert kinds.index(w.Views) < kinds.index(w.Controls) < kinds.index(w.Selection)
    run(go())


def test_the_script_sets_a_selection(server, ctl):
    space = caliper.items("test")

    async def go():
        async with Viewer(server) as v:
            await v.expect(w.Controls)
            space.select([5, 1])
            sel = await v.expect(w.Selection)
            assert sel.items == [1, 5] and sel.view is None
    run(go())


# ---------------------------------------------------------------------------
# the training thread's side
# ---------------------------------------------------------------------------
def test_a_paused_run_shows_its_last_step(server, ctl):
    """The values a step appended before the pause reach the viewer while the
    loop waits: the open tick is closed on the training thread."""
    def body(i):
        caliper.series(torch.tensor(float(i)), "loss")

    async def go():
        async with Viewer(server) as v:
            with Loop(body=body, delay=0.0005) as loop:
                await v.expect(w.Views, lambda m: any(x["name"] == "loss" for x in m.views))
                await v.send(w.Subscribe(1, "loss", 1))
                await v.expect(w.ViewReply)
                await v.control("pause")
                await v.state(lambda m: m.run["state"] == "paused")
                total = loop.count
                got = await v.expect(w.Series, lambda m: m.total == total, timeout=5)
                assert got.values.array()[-1] == total - 1
    run(go())


def test_close_releases_a_paused_run(make_server, ctl):
    s = make_server()

    async def go():
        async with Viewer(s) as v:
            with Loop() as loop:
                await v.state(lambda m: m.run["state"] == "running")
                await v.control("pause")
                await v.state(lambda m: m.run["state"] == "paused")
                n = loop.count
                s.close()
                await until(lambda: loop.count > n + 3)
    run(go())


def test_step_stays_cheap_with_a_viewer(server, ctl):
    caliper.param("lr", 2e-3)

    async def go():
        async with Viewer(server) as v:
            await v.expect(w.Controls)
            caliper.step()
            assert ctl.busy is False
            number = 200_000
            best = min(timeit.repeat("step()", globals={"step": caliper.step}, number=number,
                                     repeat=5)) / number
            print(f"caliper.step() with a viewer attached: {best * 1e9:.1f} ns per call")
            assert best < 1e-6
    run(go())
