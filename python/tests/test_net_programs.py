"""The wire path of GPU programs in the script (caliper._net_programs) over caliper.serve(), with a
websockets viewer: the capabilities programs and interventions only under
caliper.watch(model, optimizer), PROGRAM set and remove reaching the facade and PROGRAM_STATE
answering them, the refusals of a viewer without the capability, of a read-only server and of a
wrong hash, the refused item of a script that cannot run programs, the unrequested status on a
change, the removal of every program after the last viewer left, and the key interventions of
CONTROLS only for a viewer that announced it."""
from __future__ import annotations

import hashlib
import threading
import time

import pytest

from caliper import _net_programs as NP
from caliper import _wire as w

pytest.importorskip("websockets")
from test_net_controls import Viewer, ctl, make_server, mode, run, server  # noqa: E402,F401

PROGRAM_CAPS = [w.CAP_CONTROLS, w.CAP_SELECTION, w.CAP_PROGRAMS, w.CAP_INTERVENTIONS]
WGSL = "@compute @workgroup_size(64) fn main() {}\n"
SHA = hashlib.sha256(WGSL.encode()).hexdigest()
PIXELS = {"inputs": [{"var": "w", "source": "live:fc.weight"}],
          "output": {"kind": "pixels", "width": 8, "height": 8}}


class FakeFacade:
    """caliper._programs as the producer runtime gives it: install, remove, status."""

    def __init__(self):
        self.lock = threading.Lock()
        self.installed: dict = {}
        self.calls: list = []
        self.state = "compiled"

    def install(self, program_id, declaration, wgsl, *, scene=None, sha256=""):
        with self.lock:
            self.calls.append(("install", program_id, declaration, wgsl, scene, sha256))
            self.installed[program_id] = sha256
        return self.status()[-1]

    def remove(self, program_id):
        with self.lock:
            self.calls.append(("remove", program_id))
            self.installed.pop(program_id, None)

    def status(self):
        with self.lock:
            return [{"id": k, "kind": "pixels", "state": self.state, "every": None,
                     "every_min": 1, "runs": 0, "memory_bytes": 0,
                     "memory_limit_bytes": 64 << 20, "error": None, "sha256": v}
                    for k, v in self.installed.items()]


@pytest.fixture
def watching(monkeypatch):
    """A caliper.watch(model, optimizer) is on."""
    monkeypatch.setattr(NP, "_watches", 0)
    NP.enable()
    yield
    NP.disable()


@pytest.fixture
def flows(mode):
    """The program flows run on both senders: the native server's links carry the capabilities
    programs and interventions to Python."""


@pytest.fixture
def facade(monkeypatch):
    f = FakeFacade()
    monkeypatch.setattr(NP, "_facade", lambda: f)
    return f


def wait_for(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        time.sleep(0.01)


def _set(rid, pid="img", sha=SHA):
    return w.Program(rid, w.PROGRAM_SET, pid, PIXELS, WGSL, sha)


def test_capabilities_only_under_watch(monkeypatch, server):
    monkeypatch.setattr(NP, "_watches", 0)
    NP._native_caps(False)

    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            assert w.CAP_PROGRAMS not in v.welcome.capabilities
            assert w.CAP_INTERVENTIONS not in v.welcome.capabilities
        NP.enable()
        try:
            async with Viewer(server, PROGRAM_CAPS) as v:
                assert {w.CAP_PROGRAMS, w.CAP_INTERVENTIONS} <= set(v.welcome.capabilities)
        finally:
            NP.disable()
    run(go())


def test_set_and_remove_reach_the_facade(flows, watching, facade, server):
    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            await v.send(_set(5))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 5)
            assert [x["id"] for x in st.programs] == ["img"]
            assert st.programs[0]["sha256"] == SHA and "host" not in st.programs[0]
            assert facade.calls[0] == ("install", "img", PIXELS, WGSL, None, SHA)
            await v.send(w.Program(6, w.PROGRAM_REMOVE, "img"))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 6)
            assert st.programs == [] and facade.calls[-1] == ("remove", "img")
    run(go())


def test_script_without_runtime_refuses_the_program(flows, watching, server, monkeypatch):
    class NoRuntime(FakeFacade):
        def install(self, *a, **k):               # a facade that cannot install raises
            raise NotImplementedError("no producer runtime")
    monkeypatch.setattr(NP, "_facade", lambda f=NoRuntime(): f)

    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            await v.send(_set(7))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 7)
            (item,) = st.programs
            assert item["state"] == "refused" and item["error"]["code"] == NP.UNAVAILABLE
            await v.send(w.Program(8, w.PROGRAM_REMOVE, "img"))
            assert (await v.expect(w.ProgramState, lambda m: m.request_id == 8)).programs == []
    run(go())


@pytest.mark.parametrize("case", ["no_capability", "no_watch", "bad_hash"])
def test_refusals(monkeypatch, mode, facade, server, case):
    if mode == "native" and case == "bad_hash":
        pytest.skip("the native server's links do not carry the capability programs yet")
    monkeypatch.setattr(NP, "_watches", 0)
    if case != "no_watch":
        NP.enable()
        cleanup = NP.disable
    else:
        NP._native_caps(False)
        cleanup = None
    caps = [w.CAP_CONTROLS] if case == "no_capability" else PROGRAM_CAPS
    want = w.ERR_BAD_REQUEST if case == "bad_hash" else w.ERR_UNSUPPORTED

    async def go():
        async with Viewer(server, caps) as v:
            await v.send(_set(9, sha="0" * 64 if case == "bad_hash" else SHA))
            e = await v.expect(w.Error)
            assert (e.code, e.in_reply_to, e.request_id) == (want, int(w.MsgType.PROGRAM), 9)
    try:
        run(go())
    finally:
        if cleanup is not None:
            cleanup()
    assert facade.calls == []


def test_read_only_server_forbids_programs(flows, watching, facade, make_server):
    s = make_server(control=False)

    async def go():
        async with Viewer(s, PROGRAM_CAPS) as v:
            await v.send(_set(3))
            assert (await v.expect(w.Error)).code == w.ERR_FORBIDDEN
    run(go())
    assert facade.calls == []


def test_unrequested_state_after_a_change(flows, watching, facade, server):
    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            await v.send(_set(1))
            await v.expect(w.ProgramState, lambda m: m.request_id == 1)
            facade.state = "running"
            st = await v.expect(w.ProgramState, lambda m: m.request_id is None
                                and m.programs and m.programs[0]["state"] == "running")
            assert st.programs[0]["id"] == "img"
    run(go())


def test_programs_end_after_the_last_viewer(flows, monkeypatch, watching, facade, server):
    monkeypatch.setattr(NP, "LINGER_S", 0.2)

    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            await v.send(_set(1))
            await v.expect(w.ProgramState, lambda m: m.request_id == 1)
    run(go())
    wait_for(lambda: ("remove", "img") in facade.calls)
    assert facade.installed == {}


def test_interventions_key_only_for_its_viewers(flows, watching, server):
    iv = {"points": [{"point": "fc", "kind": "module", "module": "fc"}], "active": []}
    meta = {"run": {"state": "running", "step": 0, "steps_left": 0}, "params": [],
            "actions": [], "interventions": iv}

    async def go():
        async with Viewer(server, PROGRAM_CAPS) as a, \
                Viewer(server, [w.CAP_CONTROLS, w.CAP_SELECTION]) as b:
            await a.drain()
            await b.drain()
            server.producer.controls.broadcast(w.MsgType.CONTROLS, meta)
            got_a = await a.expect(w.Controls, lambda m: m.interventions is not None)
            assert got_a.interventions == iv
            got_b = await b.expect(w.Controls, lambda m: m.run.get("step") == 0)
            assert got_b.interventions is None
    run(go())
