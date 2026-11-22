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

    def set_params(self, program_id, params):
        with self.lock:
            self.calls.append(("set_params", program_id, params))

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


# ---- the params op and the operator id -----------------------------------------------------------
def test_the_params_op_form():
    msg = w.Program(4, w.PROGRAM_PARAMS, "img", params={"level": 0.5, "k_2": 3})
    got = w.decode(w.encode(msg))
    assert (got.op, got.id, got.params) == ("params", "img", {"level": 0.5, "k_2": 3})
    assert got.declaration is None and "declaration" not in msg.to_meta() and msg.to_sections() == []


@pytest.mark.parametrize("params", [
    None, {}, [], {"a": "1"}, {"a": float("nan")}, {"a": float("inf")}, {"a": True}, {"1a": 1}, {"__x": 1},
    {"a-b": 1}, {"": 1}, {"x" * 64: 1}, {f"p{i}": 1 for i in range(17)}])
def test_the_params_op_refuses_a_bad_form(params):
    meta = {"request_id": 1, "op": "params", "id": "img"}
    if params is not None:
        meta["params"] = params
    with pytest.raises(w.MessageError):
        w.Program.from_frame(w.Frame(w.MsgType.PROGRAM, meta, []))


def test_the_params_op_takes_16_names_of_63_bytes():
    ps = {("p" * 62 + str(i % 10)) if i == 0 else f"p{i}": 1.5 for i in range(16)}
    meta = {"request_id": 1, "op": "params", "id": "img", "params": ps}
    assert w.Program.from_frame(w.Frame(w.MsgType.PROGRAM, meta, [])).params == ps
    assert w.Program.from_frame(w.Frame(w.MsgType.PROGRAM, dict(meta, params={"x" * 63: 1}), [])).params


@pytest.mark.parametrize("pid,ok", [
    ("operator.terrain.0", True), ("operator.a-b_c1.12345", True), ("operator.t.123456", False),
    ("operator.t.", False), ("operator..1", False), ("operator.T.1", False), ("operator.1a.1", False),
    ("operator.t", False), ("operator.t.x", False), ("operator." + "a" * 45 + ".1", False),
    ("operator." + "a" * 44 + ".1", True), ("img", True), ("Img", False), ("a.b", False)])
def test_the_operator_id_form(pid, ok):
    assert w._program_id(pid) is ok


def _decl(**kw):
    d = {"output": {"kind": "tensor", "shape": [4, 4]}}
    d.update(kw)
    return d


@pytest.mark.parametrize("decl,ok", [
    (_decl(), True), (_decl(params={"level": 1}, builtin="surface@1"), True),
    (_decl(builtin=""), False), (_decl(builtin="b" * 33), False), (_decl(builtin=3), False),
    (_decl(params={}), False), (_decl(params={"__x": 1}), False),
    ({"output": {"kind": "tensor", "shape": [1000, 1000]}}, True),
    ({"output": {"kind": "tensor", "shape": [1001, 1000]}}, False),
    ({"output": {"kind": "tensor", "shape": []}}, False),
    ({"output": {"kind": "tensor", "shape": [1, 1, 1, 1, 1]}}, False),
    ({"output": {"kind": "tensor", "shape": [0, 4]}}, False),
    ({"output": {"kind": "tensor"}}, False),
    ({"output": {"kind": "tensor", "shape": [4], "width": 4}}, False)])
def test_a_set_declaration_with_a_tensor_output(decl, ok):
    meta = {"request_id": 1, "op": "set", "id": "operator.t.0", "declaration": decl, "sha256": SHA}
    sec = w.Section.from_array(w.SECTION_WGSL, __import__("numpy").frombuffer(WGSL.encode(), dtype="uint8"))
    frame = w.Frame(w.MsgType.PROGRAM, meta, [sec])
    if ok:
        assert w.Program.from_frame(frame).declaration == decl
    else:
        with pytest.raises(w.MessageError):
            w.Program.from_frame(frame)


def test_the_params_op_is_applied_and_answered(mode, watching, facade, server):
    if mode == "native":
        pytest.skip("the native server's links do not carry the params op yet")

    async def go():
        async with Viewer(server, PROGRAM_CAPS) as v:
            await v.send(w.Program(1, w.PROGRAM_SET, "operator.t.0",
                                   {"output": {"kind": "tensor", "shape": [2, 2]}, "builtin": "surface@1",
                                    "params": {"level": 1}}, WGSL, SHA))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 1)
            assert [x["id"] for x in st.programs] == ["operator.t.0"]
            await v.send(w.Program(2, w.PROGRAM_PARAMS, "operator.t.0", params={"level": 2.5}))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 2)
            assert facade.calls[-1] == ("set_params", "operator.t.0", {"level": 2.5})
            assert [x["id"] for x in st.programs] == ["operator.t.0"]
            await v.send(w.Program(3, w.PROGRAM_PARAMS, "gone", params={"level": 1}))
            st = await v.expect(w.ProgramState, lambda m: m.request_id == 3)
            assert [x["id"] for x in st.programs] == ["operator.t.0"]
    run(go())
