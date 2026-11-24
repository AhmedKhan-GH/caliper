"""caliper._controls: the controls of a running script and linked selection
(wire protocol 3.1), without a socket. A fake transport records what the
controls broadcast and reply, checks every message against caliper._wire, and
injects CONTROL requests and select events as the transport's thread would,
while a loop calls caliper.step() on another thread or on this one.

Needs numpy and pytest only (anywidget for the panel widget's tests):

    PYTHONPATH=python .venv/bin/python -m pytest --noconftest python/tests/test_controls.py
"""
from __future__ import annotations

import _thread
import importlib.util
import itertools
import math
import os
import subprocess
import sys
import threading
import time
import timeit
import types
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import caliper  # noqa: E402
from caliper import _controls as C  # noqa: E402
from caliper import _wire as w  # noqa: E402

needs_anywidget = pytest.mark.skipif(importlib.util.find_spec("anywidget") is None,
                                     reason="anywidget is not installed (caliper[notebook])")


# ---------------------------------------------------------------------------
# a transport that records, and checks every message against the codec
# ---------------------------------------------------------------------------
def valid(type_, meta):
    msg = w.decode(w.encode_frame(type_, meta))
    assert int(type(msg).TYPE) == int(type_), (type_, meta)
    return msg


class Fake:
    def __init__(self, allow=True, controllers=1):
        self.lock = threading.Lock()
        self.sent, self.replies = [], []
        self.ordered = []                     # after_data of each broadcast
        self.allow, self.n, self.views = allow, controllers, 0

    def broadcast(self, type_, meta, after_data=False):
        valid(type_, meta)
        with self.lock:
            self.sent.append((int(type_), meta))
            self.ordered.append(after_data)

    def reply(self, conn, meta):
        valid(w.MsgType.CONTROL_REPLY, meta)
        with self.lock:
            self.replies.append((conn, meta))

    def may_control(self, conn):
        return self.allow(conn) if callable(self.allow) else self.allow

    def controllers(self):
        return self.n

    def views_changed(self):
        self.views += 1

    def controls(self):
        with self.lock:
            return [m for t, m in self.sent if t == w.MsgType.CONTROLS]

    def selections(self):
        with self.lock:
            return [m for t, m in self.sent if t == w.MsgType.SELECTION]


@pytest.fixture
def ctl(monkeypatch):
    """A fresh Controls behind the public calls."""
    c = C.Controls()
    monkeypatch.setattr(C, "CONTROLS", c)
    monkeypatch.setattr(caliper, "_CTL", c)
    yield c
    for r in list(c.routes):
        r.detach()
    with c.lock:
        c.state = C.RUN_RUNNING
        c.cond.notify_all()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(C, "RELEASE_GRACE_S", 0.3)
    monkeypatch.setattr(C, "WAIT_POLL_S", 0.01)
    monkeypatch.setattr(C, "ACTION_WAIT_S", 0.3)
    monkeypatch.setattr(C, "DETAILS_TIMEOUT_S", 0.3)


_rid = itertools.count(1)


def control(route, op, conn="c1", **kw):
    meta = {"request_id": next(_rid), "op": op, **kw}
    reply = route.on_control(conn, meta)
    m = valid(w.MsgType.CONTROL_REPLY, reply)
    assert m.request_id == meta["request_id"] and m.op == op
    return reply


def wait_for(fn, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.005)
    raise AssertionError("timed out")


class Loop:
    """A training loop on its own thread: caliper.step(), then one count."""

    def __init__(self, delay=0.001, warm=True):
        self.warm = warm
        self.count = 0
        self.stop = False
        self.error = None
        self.delay = delay
        self.in_wait = False                  # the thread is in a pause's wait (a wait hook ran)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _hook(self):
        if threading.current_thread() is self.thread:
            self.in_wait = True

    def _run(self):
        try:
            while not self.stop:
                caliper.step()
                self.in_wait = False
                self.count += 1
                time.sleep(self.delay)
        except BaseException as e:            # noqa: BLE001
            self.error = e

    def __enter__(self):
        self.ctl = caliper._CTL
        self.ctl.wait_hooks.append(self._hook)
        self.thread.start()
        if self.warm:
            wait_for(lambda: self.count > 2)
        return self

    def __exit__(self, *exc):
        self.stop = True
        caliper._CTL._release()
        self.thread.join(5)
        self.ctl.wait_hooks.remove(self._hook)
        assert not self.thread.is_alive()
        assert self.error is None, self.error

    def still(self, seconds=0.15) -> bool:
        """The loop is held: its thread has entered the pause's wait (waited
        for, so a slow thread cannot pass this by not having got there yet),
        and no step follows during ``seconds``."""
        wait_for(lambda: self.in_wait)
        a = self.count
        time.sleep(seconds)
        return self.count == a and self.in_wait


# ---------------------------------------------------------------------------
# the module stands alone and agrees with the protocol
# ---------------------------------------------------------------------------
def test_strings_and_limits_mirror_the_codec():
    assert C.CAPABILITIES == (w.CAP_CONTROLS, w.CAP_SELECTION)
    assert (C.RUN_NONE, C.RUN_RUNNING, C.RUN_PAUSING, C.RUN_PAUSED, C.RUN_STEPPING) == \
        w.RUN_STATES
    for name in ("NAME_MAX", "MAX_PARAMS", "MAX_ACTIONS", "MAX_CHOICES", "TEXT_MAX",
                 "MAX_ITEM_SPACES", "MAX_DETAIL_ITEMS", "MAX_DETAIL_FIELDS", "DETAIL_TEXT_MAX"):
        assert getattr(C, name) == getattr(w, name), name
    assert C.MAX_SELECTION == w.MAX_EVENT_INDICES
    assert C.ACTION_WAIT_S * 1000 == w.ACTION_WAIT_MS
    assert C.DETAILS_TIMEOUT_S * 1000 == w.DETAILS_TIMEOUT_MS
    assert C.RELEASE_GRACE_S * 1000 == w.RELEASE_GRACE_MS


def test_intervention_strings_and_limits_mirror_the_codec():
    if not hasattr(w, "OP_INTERVENE"):
        pytest.skip("this codec has no CONTROL op intervene (protocol 3.2 draft)")
    assert C.OP_INTERVENE == w.OP_INTERVENE
    from caliper._intervention_protocol import KINDS, specification
    from caliper.interp._intervene import KINDS as EXECUTOR_KINDS
    assert C.INTERVENE_OPS == w.INTERVENE_OPS == KINDS == EXECUTOR_KINDS
    for fields in ({"op": "ablate"}, {"kind": "add", "vector": [1., -1.], "coefficient": .5},
                   {"kind": "scale", "feature": 1, "multiplier": 2.},
                   {"kind": "clamp", "feature": 1, "value": 0.},
                   {"kind": "patch", "source_prompt": "AB", "positions": [0], "source_position": 1}):
        spec = {"point": "layers.0.mlp_out", **fields}
        assert specification(spec)
        control = w.Control(1, "intervene", intervene={"set": spec})
        assert w.decode(w.encode(control)).intervene == {"set": spec}
    assert (C.IV_PENDING, C.IV_ACTIVE, C.IV_ENDED) == w.INTERVENTION_STATES
    for name in ("MAX_POINTS_LISTED", "MAX_INTERVENTIONS", "POINT_MAX"):
        assert getattr(C, name) == getattr(w, name), name


def test_a_script_without_a_watched_model_sends_no_interventions_and_refuses_intervene():
    ctl = C.Controls()
    assert "interventions" not in ctl.controls_meta()
    with ctl.lock:
        assert ctl._intervene({"clear_all": True})[0] == C.ERR_NOT_SUPPORTED
    assert ctl.summary()["interventions"] == []


def test_using_controls_imports_nothing_heavy():
    code = ("import sys, caliper\n"
            "p = caliper.param('lr', 1e-3, min=1e-5, max=0.1, log=True)\n"
            "a = caliper.action('save', lambda: 'ok')\n"
            "s = caliper.items('test', details={'label': [1, 2]})\n"
            "for i in range(3): caliper.step()\n"
            "assert a() == 'ok' and p.value == 1e-3\n"
            "print(','.join(m for m in ('numpy', 'torch', 'websockets', 'caliper._wire', "
            "'caliper._native', 'caliper._widget') if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ,
                       "PYTHONPATH": str(ROOT / "python")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_step_costs_almost_nothing_when_idle(ctl):
    caliper.step()                              # the first call leaves the state none
    assert ctl.busy is False
    n = 200_000
    best = min(timeit.repeat("step()", globals={"step": caliper.step}, number=n, repeat=5)) / n
    empty = min(timeit.repeat("f()", globals={"f": lambda: None}, number=n, repeat=5)) / n
    print(f"caliper.step() idle: {best * 1e9:.1f} ns per call (an empty call: "
          f"{empty * 1e9:.1f} ns)")
    assert best < 1e-6
    assert ctl.n == 1 + 5 * n


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def test_parameter_types(ctl):
    lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1, log=True, help="learning rate")
    assert (lr.type, lr.value, lr.min, lr.max, lr.log) == ("number", 1e-3, 1e-5, 0.1, True)
    batch = caliper.param("batch", 64, min=8, max=512)
    assert batch.type == "integer" and isinstance(batch.value, int)
    assert caliper.param("augment", True).type == "boolean"
    opt = caliper.param("optimizer", "adam", choices=["adam", "sgd"])
    assert (opt.type, opt.value) == ("choice", "adam")
    assert caliper.param("seed", 3).entry() == {"name": "seed", "type": "integer", "value": 3}
    assert [p for p in ctl.params] == ["lr", "batch", "augment", "optimizer", "seed"]


@pytest.mark.parametrize("args, kw, exc", [
    (("x", "adam"), {"min": 0}, ValueError),
    (("x", [1]), {}, TypeError),
    (("x", "c"), {"choices": ["a", "b"]}, ValueError),
    (("x", "a"), {"choices": []}, ValueError),
    (("x", "a"), {"choices": ["a", "a"]}, ValueError),
    (("x", "a"), {"choices": ["a"], "min": 1}, ValueError),
    (("x", True), {"max": 1}, ValueError),
    (("x", 0.5), {"log": True}, ValueError),
    (("x", 0.5), {"min": 0.0, "max": 1.0, "log": True}, ValueError),
    (("x", 5), {"min": 6}, ValueError),
    (("x", 1.0), {"min": 2.0, "max": 1.0}, ValueError),
    (("x", 1), {"min": 0.5}, ValueError),
    (("x", 2 ** 60), {}, ValueError),
    (("x", math.nan), {}, ValueError),
    (("x", math.inf), {}, ValueError),
    (("", 1.0), {}, ValueError),
    (("x" * 64, 1.0), {}, ValueError),
    (("x", 1.0), {"help": "h" * 1025}, ValueError),
])
def test_bad_parameter(ctl, args, kw, exc):
    with pytest.raises(exc):
        caliper.param(*args, **kw)
    assert not ctl.params


def test_parameter_limits_and_replacement(ctl):
    for i in range(C.MAX_PARAMS):
        caliper.param(f"p{i}", 1.0)
    with pytest.raises(ValueError, match="at most 64"):
        caliper.param("one more", 1.0)
    old = ctl.params["p0"]
    new = caliper.param("p0", 5, min=0, max=9)          # a name again replaces it
    assert old.removed and not new.removed and ctl.params["p0"] is new
    assert new.remove() and "p0" not in ctl.params and not new.remove()


def test_set_from_the_script(ctl):
    t = Fake()
    ctl.attach(t)
    lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1)
    assert lr.set(0.5) == 0.1 and lr.value == 0.1
    assert lr.set(3) == 3e-0 * 0 + 0.1 and isinstance(lr.value, float)
    batch = caliper.param("batch", 64, min=8, max=512)
    assert batch.set(64.0) == 64 and isinstance(batch.value, int)
    import numpy as np
    assert batch.set(np.int64(4)) == 8
    for bad in (True, "1", 64.5, 2 ** 60):
        with pytest.raises(ValueError):
            batch.set(bad)
    assert t.controls()[-1]["params"][1] == {"name": "batch", "type": "integer", "value": 8,
                                             "min": 8, "max": 512}


def test_viewer_values_before_and_after_the_first_checkpoint(ctl):
    t = Fake()
    r = ctl.attach(t)
    lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1, log=True)
    # Before the first checkpoint a value is set at once.
    reply = control(r, "set", name="lr", value=0.01)
    assert reply == {"request_id": reply["request_id"], "op": "set", "status": "ok",
                     "name": "lr", "value": 0.01}
    assert lr.value == 0.01 and lr.pending is None
    caliper.step()
    # After it, a value waits for the next checkpoint; the latest wins.
    for v in (0.02, 0.05, 7.0):
        reply = control(r, "set", name="lr", value=v)
    assert reply["value"] == 0.1                          # clamped to max, as the reply says
    assert lr.value == 0.01 and lr.pending == 0.1
    entry = t.controls()[-1]["params"][0]
    assert (entry["value"], entry["pending"]) == (0.01, 0.1)
    assert ctl.busy
    caliper.step()
    assert lr.value == 0.1 and lr.pending is None and not ctl.busy
    assert "pending" not in t.controls()[-1]["params"][0]


@pytest.mark.parametrize("meta, code", [
    ({"op": "set", "name": "nope", "value": 1}, "unknown_control"),
    ({"op": "run", "name": "nope"}, "unknown_control"),
    ({"op": "set", "name": "lr", "value": True}, "bad_value"),
    ({"op": "set", "name": "lr", "value": "fast"}, "bad_value"),
    ({"op": "set", "name": "batch", "value": 2 ** 60}, "bad_value"),
    ({"op": "set", "name": "batch", "value": 1.5}, "bad_value"),
    ({"op": "set", "name": "opt", "value": "SGD"}, "bad_value"),
    ({"op": "set", "name": "opt", "value": 0}, "bad_value"),
    ({"op": "set", "name": "aug", "value": 1}, "bad_value"),
])
def test_refused_controls(ctl, meta, code):
    t = Fake()
    r = ctl.attach(t)
    caliper.param("lr", 1e-3, min=1e-5, max=1e-1)
    caliper.param("batch", 64, min=8, max=512)
    caliper.param("opt", "adam", choices=["adam", "sgd"])
    caliper.param("aug", False)
    before = ctl.summary()
    reply = control(r, **meta)
    assert reply["status"] == "error" and reply["error"]["code"] == code, reply
    assert ctl.summary()["params"] == before["params"] and not ctl.queue


def test_huge_values_are_clamped(ctl):
    r = ctl.attach(Fake())
    lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1)
    free = caliper.param("free", 0.0)
    assert control(r, "set", name="lr", value=1e308)["value"] == 0.1
    assert control(r, "set", name="lr", value=-1e308)["value"] == 1e-5
    assert control(r, "set", name="free", value=-1e308)["value"] == -1e308
    assert lr.value == 1e-5 and free.value == -1e308


@pytest.mark.parametrize("meta", [
    {"op": "pause"},
    {"request_id": 1, "op": "explode"},
    {"request_id": 1, "op": "set", "value": 1},
    {"request_id": 1, "op": "set", "name": "lr", "value": {"$": "code"}},
    {"request_id": 1, "op": "set", "name": "x" * 64, "value": 1},
    {"request_id": 1, "op": "run"},
    {"request_id": -1, "op": "resume"},
    [],
    "pause",
])
def test_malformed_controls_raise_bad_request(ctl, meta):
    r = ctl.attach(Fake())
    with pytest.raises(w.MessageError) as e:
        r.on_control("c1", meta)
    assert e.value.code == w.ERR_BAD_REQUEST


def test_forbidden(ctl):
    t = Fake(allow=lambda conn: conn == "owner")
    r = ctl.attach(t)
    caliper.step()
    caliper.param("lr", 1e-3)
    for op, kw in (("pause", {}), ("resume", {}), ("step", {"count": 2}),
                   ("set", {"name": "lr", "value": 1.0}), ("run", {"name": "x"})):
        reply = control(r, op, conn="viewer", **kw)
        assert reply["error"]["code"] == "forbidden"
    assert ctl.state == "running" and ctl.params["lr"].value == 1e-3
    assert control(r, "pause", conn="owner")["status"] == "ok"
    assert ctl.state == "pausing"


def test_a_flood_of_values_from_several_threads(ctl):
    t = Fake()
    r = ctl.attach(t)
    p = caliper.param("p", 0, min=0, max=10 ** 6)
    caliper.step()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            caliper.step()

    lt = threading.Thread(target=loop)
    lt.start()
    start = time.perf_counter()

    def flood(base):
        for i in range(5000):
            control(r, "set", conn=f"c{base}", name="p", value=base * 10 ** 5 + i)

    threads = [threading.Thread(target=flood, args=(k,)) for k in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    seconds = time.perf_counter() - start
    stop.set()
    lt.join()
    caliper.step()
    print(f"20,000 set requests from 4 threads in {seconds:.2f} s while the loop stepped")
    assert p.value in {k * 10 ** 5 + 4999 for k in range(4)} and p.pending is None
    assert ctl._npending == 0 and not ctl.busy
    assert seconds < 20


# ---------------------------------------------------------------------------
# pause, resume and step
# ---------------------------------------------------------------------------
def test_pause_and_step_need_a_checkpoint(ctl):
    r = ctl.attach(Fake())
    assert control(r, "pause")["error"]["code"] == "unsupported"
    assert control(r, "step")["error"]["code"] == "unsupported"
    assert control(r, "resume")["status"] == "ok"
    assert ctl.state == "none"


def test_pause_and_resume_from_another_thread(ctl, fast):
    t = Fake()
    r = ctl.attach(t)
    with Loop() as loop:
        assert control(r, "pause")["status"] == "ok"
        wait_for(lambda: ctl.state == "paused")
        assert loop.still()
        states = [m["run"]["state"] for m in t.controls()]
        assert states[-2:] == ["pausing", "paused"]
        # only the announcement of the pause waits for the data before it
        flags = [f for (k, _), f in zip(t.sent, t.ordered) if k == w.MsgType.CONTROLS]
        assert flags[-2:] == [False, True]
        assert t.controls()[-1]["run"]["step"] == ctl.n
        assert control(r, "pause")["status"] == "ok" and ctl.state == "paused"   # no change
        assert control(r, "resume")["status"] == "ok"
        a = loop.count
        wait_for(lambda: loop.count > a + 5)
        assert ctl.state == "running" and t.controls()[-1]["run"]["state"] == "running"


@pytest.mark.parametrize("count", [1, 3, 7])
def test_single_steps(ctl, fast, count):
    t = Fake()
    r = ctl.attach(t)
    with Loop() as loop:
        control(r, "pause")
        wait_for(lambda: ctl.state == "paused")
        before = loop.count
        reply = control(r, "step", count=count)
        assert reply["status"] == "ok"
        wait_for(lambda: ctl.state == "paused" and loop.count > before)
        assert loop.still()
        assert loop.count - before == count
        assert any(m["run"].get("steps_left") == count for m in t.controls())


def test_step_from_running_pauses_after_that_many_checkpoints(ctl, fast):
    r = ctl.attach(Fake())
    with Loop(delay=0.02) as loop:
        before = loop.count
        control(r, "step", count=4)
        wait_for(lambda: ctl.state == "paused")
        assert loop.still()
        assert 3 <= loop.count - before <= 5             # the loop was between checkpoints


def test_a_running_run_reports_the_step_of_its_latest_checkpoint(ctl, fast, monkeypatch):
    # The run.step of an app attached to a watched script stayed at 1 while the script
    # was past step 50: CONTROLS went out only when the run state changed.
    monkeypatch.setattr(C, "RUN_REPORT_S", 0.5, raising=False)
    t = Fake()
    r = ctl.attach(t)
    for _ in range(50):
        caliper.step()
    assert t.controls()[0]["run"] == {"state": "running", "step": 1}
    assert len(t.controls()) <= 5                        # no CONTROLS per step (50 steps)
    end = time.monotonic() + 15
    while t.controls()[-1]["run"]["step"] == 1 and time.monotonic() < end:
        caliper.step()
        time.sleep(0.01)
    assert t.controls()[-1]["run"] == {"state": "running", "step": ctl.n}
    assert ctl.n > 50 and not ctl.busy
    with Loop() as loop:                                 # a paused run stands at its step
        control(r, "pause")
        wait_for(lambda: ctl.state == "paused")
        assert loop.still(seconds=2 * C.RUN_REPORT_S)
        assert t.controls()[-1]["run"] == {"state": "paused", "step": ctl.n}


def test_values_and_actions_reach_a_paused_run(ctl, fast):
    t = Fake()
    r = ctl.attach(t)
    p = caliper.param("p", 1.0)
    ran = []
    caliper.action("save", lambda: ran.append(threading.current_thread().name) or "saved")
    with Loop() as loop:
        control(r, "pause")
        wait_for(lambda: ctl.state == "paused")
        control(r, "set", name="p", value=2.0)
        wait_for(lambda: p.value == 2.0)
        control(r, "run", name="save")
        wait_for(lambda: len(t.replies) == 1)
        assert loop.still() and ctl.state == "paused"
        assert ran == [loop.thread.name]                  # on the training thread
        assert t.replies[0][1]["status"] == "ok" and t.replies[0][1]["message"] == "saved"


def test_ctrl_c_ends_a_paused_checkpoint(ctl, fast):
    r = ctl.attach(Fake())
    caliper.step()
    control(r, "pause")
    timer = threading.Timer(0.3, _thread.interrupt_main)
    timer.start()
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        caliper.step()
    assert 0.25 < time.monotonic() - start < 10       # the interrupt, not a wait for the run
    assert ctl.state == "running"
    caliper.step()                                     # the next call does not wait


def test_a_closed_server_releases_a_paused_run(ctl, fast):
    r = ctl.attach(Fake())
    with Loop() as loop:
        control(r, "pause")
        wait_for(lambda: ctl.state == "paused")
        assert loop.still(0.4)
        r.detach()
        a = loop.count
        wait_for(lambda: loop.count > a + 3)
        assert ctl.state == "running"


def test_the_last_controller_leaving_releases_after_the_grace(ctl, fast, monkeypatch):
    # a grace well above the 0.1 s gap below, so a slow runner stays within it
    monkeypatch.setattr(C, "RELEASE_GRACE_S", 1.0)
    t = Fake()
    r = ctl.attach(t)
    with Loop() as loop:
        control(r, "pause")
        wait_for(lambda: ctl.state == "paused")
        t.n = 0
        time.sleep(0.1)
        t.n = 1                                        # a page reload: back within the grace
        assert loop.still(0.5) and ctl.state == "paused"
        t.n = 0
        start = time.monotonic()
        wait_for(lambda: ctl.state == "running")
        assert time.monotonic() - start >= 0.9
        a = loop.count
        wait_for(lambda: loop.count > a)


def test_pause_from_the_script(ctl, fast):
    with pytest.warns(RuntimeWarning, match="no stream server"):
        assert caliper.pause() is False
    caliper.step()                                     # nothing could resume it: no wait
    t = Fake()
    r = ctl.attach(t)
    assert caliper.pause() is True and ctl.state == "pausing"
    with Loop(warm=False) as loop:
        wait_for(lambda: ctl.state == "paused")
        control(r, "resume")
        a = loop.count
        wait_for(lambda: loop.count > a)
    t.n = 0
    with pytest.warns(RuntimeWarning, match="resumes after"):
        caliper.pause()


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def test_actions_run_in_order_at_the_checkpoint(ctl):
    t = Fake()
    r = ctl.attach(t)
    ran = []

    @caliper.action("save checkpoint", help="writes a checkpoint")
    def save():
        ran.append("save")
        return "saved ckpt.pt"

    evaluate = caliper.action("evaluate", lambda: ran.append("evaluate"))
    assert isinstance(save, caliper.Action) and save() == "saved ckpt.pt" and ran == ["save"]
    ran.clear()
    caliper.step()
    q1 = control(r, "run", conn="a", name="evaluate")
    q2 = control(r, "run", conn="b", name="save checkpoint")
    q3 = control(r, "run", conn="c", name="evaluate")      # joins the queued run
    assert [q["status"] for q in (q1, q2, q3)] == ["queued"] * 3
    entries = {a["name"]: a for a in t.controls()[-1]["actions"]}
    assert entries["evaluate"]["queued"] and entries["save checkpoint"]["help"]
    assert not t.replies
    caliper.step()
    assert ran == ["evaluate", "save"]
    got = {conn: m for conn, m in t.replies}
    assert set(got) == {"a", "b", "c"}
    assert got["a"]["request_id"] == q1["request_id"] and got["c"]["request_id"] == q3["request_id"]
    assert got["b"]["message"] == "saved ckpt.pt" and "message" not in got["a"]
    assert all(m["status"] == "ok" for m in got.values())
    states = [(m["actions"][1]["queued"], m["actions"][1]["running"]) for m in t.controls()
              if len(m["actions"]) == 2]
    assert (False, True) in states and states[-1] == (False, False)
    assert evaluate.entry() == {"name": "evaluate", "running": False, "queued": False}


@pytest.mark.parametrize("same_action", [False, True])
def test_late_actions_observe_prior_parameter_edits_at_the_next_pass(ctl, same_action):
    route = ctl.attach(Fake())
    value = caliper.param("value", 0)
    observed = []

    def enqueue():
        assert control(route, "set", name="value", value=7)["status"] == "ok"
        assert control(route, "run", name="evaluate")["status"] == "queued"

    def evaluate():
        observed.append(value.value)
        if same_action and len(observed) == 1:
            enqueue()

    caliper.action("evaluate", evaluate)
    caliper.action("enqueue", enqueue)
    caliper.step()
    control(route, "run", name="evaluate" if same_action else "enqueue")
    caliper.step()
    assert observed == ([0] if same_action else [])
    assert ctl.actions["evaluate"].queued and value.value == 0
    caliper.step()
    assert observed == ([0, 7] if same_action else [7])
    assert not ctl.actions["evaluate"].queued


@pytest.mark.parametrize("edit", [False, True])
def test_late_waiters_defer_the_coalesced_action_and_its_followers(ctl, edit):
    transport = Fake()
    route = ctl.attach(transport)
    value = caliper.param("value", 0)
    observed, joined = [], []

    def enqueue():
        observed.append(("enqueue", value.value))
        for new_value in (3, 7):
            if edit:
                assert control(route, "set", name="value", value=new_value)["status"] == "ok"
            joined.append(control(route, "run", name="evaluate"))

    caliper.action("enqueue", enqueue)
    evaluate = caliper.action("evaluate", lambda: observed.append(("evaluate", value.value)))
    caliper.action("after", lambda: observed.append(("after", value.value)))
    caliper.step()
    first = control(route, "run", name="enqueue")
    earlier = [control(route, "run", name="evaluate") for _ in range(2)]
    last = control(route, "run", name="after")
    run_id = evaluate.run_id

    caliper.step()
    assert observed == [("enqueue", 0)]
    assert [reply["request_id"] for _, reply in transport.replies] == [first["request_id"]]
    assert evaluate.queued and evaluate.run_id == run_id
    assert len(evaluate.waiters) == 4
    assert value.value == 0

    caliper.step()
    expected = 7 if edit else 0
    assert observed == [("enqueue", 0), ("evaluate", expected), ("after", expected)]
    assert [reply["request_id"] for _, reply in transport.replies] == [
        request["request_id"] for request in [first, *earlier, *joined, last]]
    assert all(reply["status"] == "ok" for _, reply in transport.replies)
    assert not ctl.queue and not evaluate.waiters
    caliper.step()
    assert ctl.counters["actions_run"] == 3


def test_an_action_that_raises(ctl):
    t = Fake()
    r = ctl.attach(t)
    caliper.action("boom", lambda: 1 / 0)
    caliper.action("chatty", lambda: "é" * 2000)
    caliper.step()
    control(r, "run", name="boom")
    control(r, "run", name="chatty")
    with pytest.warns(RuntimeWarning, match="boom"):
        caliper.step()
    (_, err), (_, ok) = t.replies
    assert err["status"] == "error" and err["error"]["code"] == "action_failed"
    assert "ZeroDivisionError" in err["error"]["message"]
    assert len(ok["message"].encode("utf-8")) <= 1024 and ok["message"] == "é" * 512


def test_actions_wait_for_a_checkpoint_and_time_out(ctl, fast):
    t = Fake()
    r = ctl.attach(t)
    caliper.action("save", lambda: "saved")
    assert ctl.state == "none"
    assert control(r, "run", name="save")["status"] == "queued"
    assert t.controls()[-1]["run"]["state"] == "none" and t.controls()[-1]["actions"][0]["queued"]
    wait_for(lambda: len(t.replies) == 1)
    reply = t.replies[0][1]
    assert reply["status"] == "error" and reply["error"]["code"] == "timeout"
    assert not ctl.queue and not t.controls()[-1]["actions"][0]["queued"]
    caliper.step()                                     # nothing left to run
    assert len(t.replies) == 1


def test_removing_or_replacing_a_queued_action(ctl):
    t = Fake()
    r = ctl.attach(t)
    caliper.step()
    a = caliper.action("save", lambda: "first")
    control(r, "run", conn="x", name="save")
    caliper.action("save", lambda: "second")          # the waiting run moves to the new one
    caliper.step()
    assert t.replies[-1][1]["message"] == "second"
    b = caliper.action("other", lambda: None)
    control(r, "run", conn="y", name="other")
    assert b.remove() and not a.remove()
    assert t.replies[-1][1]["error"]["code"] == "unknown_control"
    caliper.step()
    assert len(t.replies) == 2


def test_action_limits(ctl):
    for i in range(C.MAX_ACTIONS):
        caliper.action(f"a{i}", lambda: None)
    with pytest.raises(ValueError):
        caliper.action("one more", lambda: None)
    with pytest.raises(TypeError):
        caliper.action("a0", "not callable")
    with pytest.raises(ValueError):
        caliper.action("", lambda: None)


def test_without_a_transport_everything_works_locally(ctl):
    lr = caliper.param("lr", 0.1)
    for _ in range(3):
        caliper.step()
    assert ctl.state == "running" and not ctl.busy
    assert lr.set(0.2) == 0.2 and lr.value == 0.2
    assert caliper.action("hello", lambda: "hi")() == "hi"
    s = caliper.items("test", details={"label": [3, 4]})
    assert s.select([1, 0, 1]) == [0, 1] and s.selection == [0, 1]


# ---------------------------------------------------------------------------
# item spaces, linked selection and details
# ---------------------------------------------------------------------------
HEAT = {"id": 1, "name": "logits", "kind": 1, "sources": [
    {"name": "data", "dims": [{"name": "row", "size": 512}, {"name": "col", "size": 6}]}]}
POINTS = {"id": 2, "name": "embedding", "kind": 3, "sources": [
    {"name": "data", "dims": [{"name": "point", "size": 1000}]}]}
LOSS = {"id": 3, "name": "loss", "kind": 4, "sources": []}
OFFERED = {"id": 4, "name": "acts", "kind": 1, "sources": [
    {"name": "a", "dims": [{"name": "dim0", "size": 4}, {"name": "dim1", "size": 8}]},
    {"name": "b", "dims": [{"name": "sample", "size": 4}, {"name": "unit", "size": 3}]}]}


@pytest.fixture
def routes(ctl, monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setattr(caliper, "_auto_tried", False)
    t = Fake()
    return t, ctl.attach(t)


def test_items_keyword_on_the_view_calls(ctl, routes):
    t, r = routes
    assert caliper.show(None, "logits", items="test") is False   # nothing attached: no snapshot
    assert caliper.points(None, None, "embedding", items="test") is False
    assert caliper.series(0.5, "loss", items="step") is False
    assert ctl.view_items == {"logits": {0: "test"}, "embedding": {-1: "test"},
                              "loss": {None: "step"}}
    assert t.views == 3
    caliper.show(None, "logits", items="test")                   # unchanged: no new VIEWS
    assert t.views == 3
    with pytest.raises(ValueError):
        caliper.offer("h", {}, kind="histogram", items="test")
    with pytest.raises(ValueError):
        caliper.series(0.5, "loss2", items={"step": "x"})
    with pytest.raises(TypeError):
        caliper.show(None, "bad", items=3)
    caliper.remove("logits")
    assert "logits" not in ctl.view_items


def test_views_carry_their_item_spaces(ctl, routes):
    t, r = routes
    ctl.view_items_call("logits", "heatmap", "test")
    ctl.view_items_call("embedding", "points", "test")
    ctl.view_items_call("loss", "series", "step")
    ctl.view_items_call("acts", "heatmap", {"sample": "test"})
    heat = r.annotate_view(HEAT)
    assert heat["sources"][0]["dims"] == [{"name": "row", "size": 512, "items": "test"},
                                          {"name": "col", "size": 6}]
    assert "items" not in HEAT["sources"][0]["dims"][0]            # the entry is not changed
    assert r.annotate_view(POINTS)["sources"][0]["dims"][0]["items"] == "test"
    assert r.annotate_view(LOSS)["items"] == "step"
    acts = r.annotate_view(OFFERED)
    assert [d.get("items") for d in acts["sources"][0]["dims"]] == [None, None]
    assert [d.get("items") for d in acts["sources"][1]["dims"]] == ["test", None]
    other = {"id": 9, "name": "other", "kind": 1, "sources": []}
    assert r.annotate_view(other) is other
    views = w.Views([heat, r.annotate_view(POINTS), r.annotate_view(LOSS), acts], "s")
    assert w.decode(w.encode(views)) == views
    # a str on an offered heatmap names the first dim of each source
    ctl.view_items_call("acts", "heatmap", "units")
    acts = r.annotate_view(OFFERED)
    assert [s["dims"][0].get("items") for s in acts["sources"]] == ["units", "units"]


def select(r, view, detail, axes=None):
    r.on_select("c1", {"view": view, "type": "select", "detail": detail}, axes)


def test_a_selection_reaches_every_viewer(ctl, routes):
    t, r = routes
    ctl.view_items_call("embedding", "points", "test")
    ctl.view_items_call("logits", "heatmap", "test")
    test = caliper.items("test")
    select(r, "embedding", {"indices": [2, 0], "row_ids": [500, 0], "space": "test",
                            "items": [500, 0, 500]})
    (sel,) = t.selections()
    assert sel == {"space": "test", "items": [0, 500], "view": "embedding"}
    assert test.selection == [0, 500]
    kinds = [type_ for type_, _ in r.snapshot_state()]
    assert kinds == [w.MsgType.CONTROLS, w.MsgType.SELECTION]
    assert r.snapshot_state()[1][1] == sel
    # a 3.0-style event: the row ids of the points are the items
    select(r, "embedding", {"indices": [1], "row_ids": [7]})
    assert t.selections()[-1]["items"] == [7]
    # heatmap rows, from the payload axes the transport passes, or the plain dims
    axes = [{"dim": "row", "start": 0, "step": 1, "count": 512, "size": 512},
            {"dim": "col", "start": 0, "step": 1, "count": 6, "size": 6}]
    select(r, "logits", {"rows": [10, 13], "cols": [0, 6]}, axes)
    assert t.selections()[-1] == {"space": "test", "items": [10, 11, 12], "view": "logits"}
    # {} clears the spaces of the view; an empty items list clears its space
    select(r, "logits", {})
    assert t.selections()[-1]["items"] == [] and test.selection == []
    assert [k for k, _ in r.snapshot_state()] == [w.MsgType.CONTROLS]


def test_sampled_points_map_to_row_ids_through_the_point_axis(ctl, routes):
    """A viewer that sends only the indices of the sent sample: the producer
    maps them to row ids with the snapshot's payload axis (1000 points
    sampled to 4 with a stride), and the row ids are the items."""
    t, r = routes
    view = {"id": 2, "name": "embedding", "kind": 3, "sources": [
        {"name": "z", "dims": [{"name": "point", "size": 1000}]}],
            "default": {"budget": {"max_points": 4}}}
    ctl.view_items_call("embedding", "points", "test")
    r.annotate_view(view)
    resolved = w.resolve_data_view(view, {"index": {"point": [100, 1000, 3]}})
    axes = w.payload_layout(view, resolved)["axes"]
    select(r, "embedding", {"indices": [1, 3, 9]}, axes)        # 9 is past the sample
    assert t.selections()[-1]["items"] == [325, 775]
    select(r, "embedding", {"indices": [0]})                    # no axes: nothing to map with
    assert len(t.selections()) == 1


def test_series_steps_and_bounds(ctl, routes):
    t, r = routes
    ctl.view_items_call("loss", "series", "step")
    select(r, "loss", {"steps": [100, 104]})
    assert t.selections()[-1]["items"] == [100, 101, 102, 103]
    select(r, "loss", {"steps": [0, 10 ** 12]})                 # a huge range is cut
    last = t.selections()[-1]
    assert len(last["items"]) == C.MAX_SELECTION and last["truncated"] is True
    select(r, "loss", {"space": "step", "items": list(range(C.MAX_SELECTION + 5))[::-1]})
    assert t.selections()[-1]["truncated"] is True
    n = len(t.selections())
    for detail in ({"space": "other", "items": [1]},             # a space the view does not map
                   {"space": "step", "items": "1,2"},
                   {"steps": [5]}, {"steps": ["a", "b"]}, {"rows": [0, 3]}, {"value": 3}):
        select(r, "loss", detail)
    select(r, "unknown view", {"steps": [0, 3]})
    r.on_select("c1", {"view": "loss", "type": "hover", "detail": {"step": 3}})
    r.on_select("c1", "not an event")
    assert len(t.selections()) == n


def test_item_space_limit(ctl, routes):
    t, r = routes
    for i in range(C.MAX_ITEM_SPACES):
        caliper.items(f"s{i}")
    with pytest.raises(ValueError):
        caliper.items("one more")
    with pytest.raises(ValueError):
        ctl.view_items_call("v", "points", "yet another")


def test_static_details(ctl, routes):
    import numpy as np
    t, r = routes
    ctl.view_items_call("embedding", "points", "test")
    caliper.items("test", details={"label": np.arange(100) % 6, "loss": [math.nan] + [0.5] * 99,
                                   "name": ["x" * 300] * 100, "ok": [True] * 100})
    select(r, "embedding", {"space": "test", "items": list(range(50)) + [500]})
    sel = t.selections()[-1]
    d = sel["details"]
    assert d["fields"] == ["label", "loss", "name", "ok"] and d["items"] == list(range(32))
    assert d["values"][0] == [0, None, "x" * 256, True] and d["values"][7][0] == 1
    select(r, "embedding", {"space": "test", "items": [500]})   # past the table
    assert t.selections()[-1]["details"]["values"] == [[None, None, None, None]]
    with pytest.raises(ValueError):
        caliper.items("test", details={f"f{i}": [1] for i in range(17)})
    with pytest.raises(TypeError):
        caliper.items("test", details=[1, 2])


def test_details_function_runs_off_the_training_thread(ctl, routes, fast):
    t, r = routes
    ctl.view_items_call("embedding", "points", "test")
    seen = []

    def details(ids):
        seen.append((threading.current_thread().name, list(ids)))
        return {"label": [i % 6 for i in ids], "note": [f"item {i}" for i in ids]}

    caliper.items("test", details=details)
    select(r, "embedding", {"space": "test", "items": [3, 1]})
    wait_for(lambda: len(t.selections()) == 2)
    first, second = t.selections()
    assert "details" not in first and second["items"] == [1, 3]
    assert second["details"] == {"fields": ["label", "note"], "items": [1, 3],
                                 "values": [[1, "item 1"], [3, "item 3"]]}
    assert seen == [("caliper.details", [1, 3])]
    # one dict per item works too
    caliper.items("test", details=lambda ids: [{"label": i} for i in ids])
    select(r, "embedding", {"space": "test", "items": [4]})
    wait_for(lambda: len(t.selections()) == 4)
    assert t.selections()[-1]["details"]["values"] == [[4]]


def test_details_errors_time_outs_and_stale_results(ctl, routes, fast):
    t, r = routes
    ctl.view_items_call("embedding", "points", "test")
    caliper.items("test", details=lambda ids: 1 / 0)
    select(r, "embedding", {"space": "test", "items": [1]})
    wait_for(lambda: len(t.selections()) == 2)
    assert "ZeroDivisionError" in t.selections()[-1]["details_error"]
    gate = threading.Event()

    def slow(ids):
        gate.wait(5)
        return {"label": list(ids)}

    caliper.items("test", details=slow)
    select(r, "embedding", {"space": "test", "items": [2]})
    wait_for(lambda: len(t.selections()) == 4)
    assert t.selections()[-1]["details_error"].startswith("timeout")
    select(r, "embedding", {"space": "test", "items": [5]})     # waits behind the slow one
    gate.set()
    # the details of [5] come after whatever the slow job for [2] would send,
    # since the jobs run in order: wait for them rather than for a while
    wait_for(lambda: any(s["items"] == [5] and ("details" in s or "details_error" in s)
                         for s in t.selections()))
    final = t.selections()[-1]
    assert final["items"] == [5] and "details" not in final or final["details"]["items"] == [5]
    assert all(s.get("details", {}).get("items", [5]) != [2] for s in t.selections())


def test_the_script_sets_a_selection(ctl, routes):
    t, r = routes
    s = caliper.items("test", details={"label": [0, 1, 2]})
    assert s.select([2, 0]) == [0, 2]
    assert t.selections()[-1] == {"space": "test", "items": [0, 2], "details": {
        "fields": ["label"], "items": [0, 2], "values": [[0], [2]]}}
    with pytest.raises(ValueError):
        s.select([-1])
    s.clear()
    assert t.selections()[-1]["items"] == []


# ---------------------------------------------------------------------------
# the panel widget
# ---------------------------------------------------------------------------
@pytest.fixture
def server(monkeypatch):
    srv = types.SimpleNamespace(port=40123, token="tok/en")
    monkeypatch.setattr(caliper, "_server", srv, raising=False)
    return srv


@needs_anywidget
def test_controls_panel_without_its_module(server, monkeypatch, tmp_path):
    import anywidget
    import caliper._widget  # noqa: F401  (its view widget reads the real assets)
    monkeypatch.setenv("CALIPER_WEB_DIR", str(tmp_path))
    with pytest.warns(RuntimeWarning, match="caliper-controls-widget.js"):
        panel = caliper.controls()
    assert set(panel.keys) - set(anywidget.AnyWidget().keys) == {"src", "height"}
    assert panel.src == "ws://127.0.0.1:40123/v3/stream?token=tok%2Fen" and panel.height == 160
    assert "is not in this build" in panel._esm


@needs_anywidget
def test_controls_panel_with_its_module(server, monkeypatch, tmp_path):
    module = "export default { render({ model, el }) { el.textContent = 'panel'; } };\n"
    (tmp_path / "caliper-controls-widget.js").write_text(module)
    import caliper._widget  # noqa: F401
    monkeypatch.setenv("CALIPER_WEB_DIR", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        panel = caliper.controls(height=200)
    assert panel._esm == module and panel.height == 200
    with pytest.raises(Exception):
        caliper.controls(height=0)


def test_free_text_parameter_round_trip_and_utf8_bounds(ctl):
    prompt = caliper.param("prompt", "")
    assert prompt.type == "text" and prompt.value == ""
    transport = Fake()
    route = ctl.attach(transport)
    text = "A café near the river " * 20
    reply = route.on_control(None, {"request_id": 1, "op": "set", "name": "prompt", "value": text})
    assert reply["status"] == "ok" and prompt.value == text
    ctl.step(1)
    route.on_control(None, {"request_id": 2, "op": "set", "name": "prompt", "value": "next"})
    assert prompt.value == text and prompt.pending == "next"
    ctl.step(2)
    assert prompt.value == "next"
    for value in ("é" * 2049, "\ud800", "a\0b"):
        with pytest.raises(ValueError):
            caliper.param("invalid", value)
        with pytest.raises(w.MessageError):
            route.on_control(None, {"request_id": 3, "op": "set", "name": "prompt", "value": value})
    assert caliper.param("choice", "yes", choices=["yes", "no"]).type == "choice"
