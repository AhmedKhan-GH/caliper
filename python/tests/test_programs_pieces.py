"""The derive half of pieces in the script's process (caliper._programs, design/programmable-graphics.md).

A piece's instance runs several compute passes at the pass steps of the shared cadence and sends
each published resource as one RESOURCE message (type 0x0082) to the viewers that announced
``resources``, after the draw half (PROGRAM without the section ``derive``). The checks of a
manifest run anywhere. The passes need an MPS device and the native module: the arcs piece of
the plan reads two fixed attention tensors of a small model, in place, and its arcs, scores and
labels are compared with a reference on the CPU. The senders' test runs both servers with a
viewer that keeps each raw frame.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _programs as P
from caliper import _wire as w

MPS = torch.backends.mps.is_available()
try:
    import caliper._producer as _producer
    NATIVE = int(getattr(_producer, "ABI_VERSION", 0)) >= 3
except ImportError:
    NATIVE = False
needs_device = pytest.mark.skipif(not (MPS and NATIVE), reason="needs MPS and caliper._producer 3")

L, H, T, P_LEN, S = 2, 4, 32, 12, 4

COMMON = """const L: u32 = 2u;
const H: u32 = 4u;
const T: u32 = 32u;
const P: u32 = 12u;
const S: u32 = 4u;

fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(cos(a), f32(layer) * 0.6, sin(a));
}
"""

DERIVE = """fn attn(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    let k = (head * T + i) * T + j;
    if (layer == 0u) {
        return attn0[k];
    }
    return attn1[k];
}

@compute @workgroup_size(8)
fn scores(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H) {
        return;
    }
    var sum = 0.0;
    for (var t = S + P; t < S + 2u * P; t = t + 1u) {
        sum = sum + attn(id.x / H, id.x % H, t, t - P + 1u);
    }
    heads[id.x] = sum / f32(P);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H * T * T) {
        return;
    }
    let j = id.x % T;
    let i = (id.x / T) % T;
    let head = (id.x / (T * T)) % H;
    let layer = id.x / (T * T * H);
    if (j >= i) {
        return;
    }
    let w = attn(layer, head, i, j);
    if (w < params.threshold) {
        return;
    }
    let slot = atomicAdd(&arcs_count, 1u);
    if (slot >= arrayLength(&arcs)) {
        return;
    }
    arcs[slot] = Arc(j, i, layer, head, w, heads[layer * H + head]);
}

@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    labels[id.x] = Label(vec4<f32>(ring_point(0u, id.x) * 1.15, 1.0), id.x);
}
"""

DRAW = "// the draw half: the viewer runs it\n"

ARC_FIELDS = ["from_token: u32", "to_token: u32", "layer: u32", "head: u32", "weight: f32", "score: f32"]
MANIFEST = {
    "name": "attention_arcs", "version": 1,
    "description": "Arcs between the tokens of one sentence, one ring for each layer",
    "imports": [],
    "inputs": [{"name": "attn0", "kind": "tensor", "rank_min": 4, "rank_max": 4},
               {"name": "attn1", "kind": "tensor", "rank_min": 4, "rank_max": 4}],
    "params": [{"name": "threshold", "default": 0.05, "min": 0.0, "max": 0.5},
               {"name": "glow", "default": 1.0, "min": 0.0, "max": 4.0}],
    "resources": [{"id": "heads", "kind": "buffer", "element": "f32", "count": 8},
                  {"id": "arcs", "kind": "buffer", "struct": "Arc", "fields": ARC_FIELDS, "count": 4096,
                   "counted": True},
                  {"id": "labels", "kind": "buffer", "struct": "Label",
                   "fields": ["position: vec4<f32>", "string: u32"], "count": 32}],
    "derive": {"passes": [{"entry": "scores", "writes": ["heads"]},
                          {"entry": "build", "workgroups": [128, 1, 1], "writes": ["arcs"]},
                          {"entry": "place_labels"}]},
    "draw": {"camera": "orbit", "passes": [{"kind": "render", "fragment": "f"}]},
}
PIN = {"name": "attention_arcs", "version": 1, "hash": "sha256:" + "0" * 64, "from": "workspace"}
BIND = {"attn0": "live:attn0", "attn1": "live:weights.act"}
ARC = np.dtype([("from_token", "<u4"), ("to_token", "<u4"), ("layer", "<u4"), ("head", "<u4"),
                ("weight", "<f4"), ("score", "<f4")])


def weights(layer: int) -> np.ndarray:
    """The fixed weights of one layer, (1, H, T, T), causal, in [0, 0.16]."""
    h, i, j = np.meshgrid(np.arange(H), np.arange(T), np.arange(T), indexing="ij")
    v = ((i * 7 + j * 13 + h * 5 + layer * 3) % 17) / 100.0
    return np.where(j <= i, v, 0.0).astype(np.float32)[None]


def reference(threshold=np.float32(0.05)):
    """The scores of the heads and the arcs as sorted (layer, head, to, from) rows, on the CPU."""
    a = [weights(0)[0], weights(1)[0]]
    t = np.arange(S + P_LEN, S + 2 * P_LEN)
    heads = np.array([a[x // H][x % H, t, t - P_LEN + 1].sum(dtype=np.float32) / np.float32(P_LEN)
                      for x in range(L * H)], np.float32)
    rows = sorted((lay, hd, i, j) for lay in range(L) for hd in range(H) for i in range(T) for j in range(i)
                  if not a[lay][hd, i, j] < threshold)
    return heads, rows


class ArcModel(torch.nn.Module):
    """A small model whose buffer attn0 and whose module ``weights`` hold the fixed weights; the
    output of ``weights`` is new storage at each step, thus that input moves."""

    def __init__(self, width=64):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)
        self.register_buffer("attn0", torch.from_numpy(weights(0)))
        self.register_buffer("a1", torch.from_numpy(weights(1)))
        self.weights = torch.nn.Identity()

    def forward(self, x):
        w1 = self.weights(self.a1 * 1.0)
        return self.lin(x).relu().square().mean() + 0.0 * w1.sum()


class FakeLink:
    def __init__(self, caps):
        self.caps = frozenset(caps)
        self.attached = True
        self.closed = False


class FakeControls:
    """A controls port of a server with one viewer; it keeps every post."""

    def __init__(self, caps=(w.CAP_PROGRAMS, w.CAP_RESOURCES)):
        self.link = FakeLink(caps)
        self.server = type("S", (), {"producer": type("Pr", (), {"links": [self.link]})()})()
        self.posts = []

    def post(self, link, frame, cap):
        self.posts.append((link, w.decode_frame(bytes(frame)), cap))


def _train(model, opt, steps, first=0):
    x = torch.randn(256, 64, device=next(model.parameters()).device)
    for i in range(first, first + steps):
        loss = model(x)
        opt.zero_grad()
        loss.backward()
        caliper.step(i)            # the checkpoint of watch(model, optimizer): before the update
        opt.step()


@pytest.fixture
def attached(monkeypatch):
    """The arc model on MPS attached to the facade, and a controls port that keeps the posts."""
    ctl = FakeControls()
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    torch.manual_seed(0)
    model = ArcModel().to("mps")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    P.attach(model, opt)
    yield model, opt, ctl
    P.detach()
    rt = P._STATE["rt"]
    if rt is not None:
        rt.close()
    P._STATE["rt"] = None
    P._CAD.reset()


def _install(pid="tower_arcs", manifest=MANIFEST, bind=BIND, derive=DERIVE, **kw):
    return P.install_piece(pid, piece=PIN, manifest=manifest, bind=bind, params={"threshold": 0.05},
                           panel="tower", segments=[{"part": "common.wgsl", "first_line": 1}],
                           common=COMMON, derive=derive, draw=DRAW, **kw)


# ---- the checks of a manifest, anywhere ---------------------------------------------------------
@pytest.mark.parametrize("change, code", [
    (lambda m, b: b.pop("attn1"), "piece.bad_binding"),
    (lambda m, b: b.update(attn0="attn0"), "piece.bad_binding"),
    (lambda m, b: m["derive"]["passes"].append({"entry": "x", "writes": ["nothing"]}), "program.unknown_resource"),
    (lambda m, b: m["derive"].update(passes=[{"entry": "x"}] * 9), "program.bad_pass"),
    (lambda m, b: m["derive"]["passes"][1].update(workgroups=[65536, 1, 1]), "piece.bad_manifest"),
    (lambda m, b: m["resources"][1].update(count=4194305), "piece.bad_manifest"),
    (lambda m, b: m["resources"].extend([{"id": f"r{i}", "element": "f32", "count": 1} for i in range(6)]),
     "program.bad_resource"),
    (lambda m, b: m["derive"].update(time_ms=51), "program.over_limits"),
    (lambda m, b: m["inputs"][0].update(dtype="f16"), "piece.bad_manifest"),
])
def test_a_bad_manifest_is_refused_before_the_runtime(change, code):
    import copy
    m, b = copy.deepcopy(MANIFEST), dict(BIND)
    change(m, b)
    item = _install("bad", manifest=m, bind=b)
    try:
        assert item["state"] == "refused"
        assert item["error"]["code"] == code
        assert item["kind"] == "piece" and item["piece"]["name"] == "attention_arcs"
    finally:
        P.remove("bad")
    assert "bad" not in {x["id"] for x in P.status()}


def test_a_piece_without_a_derive_half_is_only_relayed(monkeypatch):
    ctl = FakeControls()
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    m = {k: v for k, v in MANIFEST.items() if k != "derive"}
    item = _install("drawn", manifest=m, derive="")
    try:
        assert item["state"] == "compiled"
        assert not P.ACTIVE                          # nothing runs at caliper.step()
        (link, f, cap), = ctl.posts
        assert f.type == w.MsgType.PROGRAM and cap == w.CAP_RESOURCES and link is ctl.link
        assert f.meta["op"] == "set" and f.meta["panel"] == "tower"
        assert {s.name for s in f.sections} == {w.SECTION_COMMON, w.SECTION_DRAW}
    finally:
        P.remove("drawn")
    assert ctl.posts[-1][1].meta["op"] == "remove"


def test_a_viewer_behind_skips_versions_and_a_version_above_its_queue_goes_nowhere(monkeypatch):
    ctl = FakeControls()
    behind = FakeLink((w.CAP_PROGRAMS, w.CAP_RESOURCES))
    behind.backlog = lambda: P.RESOURCE_BACKLOG_BYTES - 100
    ctl.server.producer.links.append(behind)
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    p = P._Program("r", {}, "", None, "")
    small = w.encode_frame(P.RESOURCE_TYPE, {"program": "r", "resource": "a", "step": 1, "count": 1, "stride": 4},
                           [w.Section(w.SECTION_RESOURCE, int(w.DType.U8), (4,), memoryview(b"abcd"))])
    P._send_resources(p, [small])
    got = [(ln, f.type) for ln, f, _c in ctl.posts]
    assert (ctl.link, P.RESOURCE_TYPE) in got and (behind, P.RESOURCE_TYPE) not in got
    assert got[0][1] == w.MsgType.PROGRAM            # the draw half first, to both viewers
    assert p.skipped == 1 and p.error is None
    big = b"x" * (P.RESOURCE_BACKLOG_BYTES + 1)
    ctl.posts.clear()
    P._send_resources(p, [big])
    assert not ctl.posts and p.error["code"] == "program.resource_too_large"


# ---- the passes on the MPS device ----------------------------------------------------------------
def _resources(ctl, name):
    return [f for _l, f, _c in ctl.posts if f.type == P.RESOURCE_TYPE and f.meta["resource"] == name]


@needs_device
def test_the_arcs_piece_publishes_its_resources_in_the_cadence(attached):
    model, opt, ctl = attached
    before = model.attn0.clone()
    item = _install()
    assert item["state"] in ("compiled", "waiting"), item["error"]
    _train(model, opt, 600)
    st = next(x for x in P.status() if x["id"] == "tower_arcs")
    assert st["state"] == "running", st["error"]
    assert st["runs"] >= 2

    # The draw half goes first, without the section derive.
    first = ctl.posts[0][1]
    assert first.type == w.MsgType.PROGRAM
    assert first.meta["manifest"]["name"] == "attention_arcs" and first.meta["bind"] == BIND
    assert {s.name for s in first.sections} == {w.SECTION_COMMON, w.SECTION_DRAW}
    assert bytes(first.section(w.SECTION_DRAW).data).decode() == DRAW

    heads, rows = reference()
    arcs = _resources(ctl, "arcs")
    assert len(arcs) == st["runs"]                  # a written resource goes at each run
    for f in arcs:
        assert f.meta["program"] == "tower_arcs" and f.meta["stride"] == 24
        data = f.section(w.SECTION_RESOURCE)
        assert data.dtype == w.DType.U8 and data.shape == (f.meta["count"] * 24,)
        assert f.meta["count"] == len(rows)
    a = np.frombuffer(bytes(arcs[-1].section(w.SECTION_RESOURCE).data), ARC)
    assert sorted(zip(a["layer"].tolist(), a["head"].tolist(), a["to_token"].tolist(),
                      a["from_token"].tolist())) == rows
    np.testing.assert_allclose(a["score"], heads[a["layer"] * H + a["head"]], rtol=1e-5)
    h = np.frombuffer(bytes(_resources(ctl, "heads")[-1].section(w.SECTION_RESOURCE).data), np.float32)
    np.testing.assert_allclose(h, heads, rtol=1e-5)
    labels = _resources(ctl, "labels")
    assert len(labels) == 1                          # static: after the first run only
    assert labels[0].meta["stride"] == 32 and labels[0].meta["count"] == T
    steps = [f.meta["step"] for f in arcs]
    assert steps == sorted(steps) and steps[-1] == st["last_step"]

    # The cadence: every pass of the piece is a pass of the shared cadence, inside its share.
    assert st["every"] >= P.N_MIN
    assert st["overhead_share"] is not None and 0 < st["overhead_share"] <= 2 * P.SHARE
    print(f"arcs: {st['runs']} runs in 600 steps, every {st['every']}, share {st['overhead_share']:.4f}, "
          f"step {st['step_ms']:.3f} ms, cost {st['last_cost_ms']:.3f} ms, pass {st['last_pass_ms']} ms")
    assert torch.equal(model.attn0, before)          # read in place, never written


@needs_device
def test_two_pieces_share_the_pass_steps_and_a_new_viewer_gets_the_draw_half_first(attached):
    model, opt, ctl = attached
    sums = {"name": "row_sums", "version": 1, "inputs": [{"name": "t", "kind": "tensor"}],
            "resources": [{"id": "sums", "kind": "buffer", "element": "f32", "count": 128}],
            "derive": {"passes": [{"entry": "add", "workgroups": [2, 1, 1], "writes": ["sums"]}]}}
    derive = """@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 128u) { return; }
    var s = 0.0;
    for (var j = 0u; j < 32u; j++) { s = s + t[id.x * 32u + j]; }
    sums[id.x] = s;
}
"""
    _install()
    P.install_piece("sums", piece=dict(PIN, name="row_sums"), manifest=sums, bind={"t": "live:attn0"},
                    derive=derive)
    _train(model, opt, 300)
    late = FakeLink((w.CAP_PROGRAMS, w.CAP_RESOURCES))
    ctl.server.producer.links.append(late)
    _train(model, opt, 300, first=300)
    a, b = (next(x for x in P.status() if x["id"] == i) for i in ("tower_arcs", "sums"))
    assert a["state"] == b["state"] == "running"
    assert a["last_step"] == b["last_step"] and a["every"] == b["every"] and a["runs"] == b["runs"]
    s = np.frombuffer(bytes(_resources(ctl, "sums")[-1].section(w.SECTION_RESOURCE).data), np.float32)
    np.testing.assert_allclose(s, weights(0).reshape(128, 32).sum(axis=1), rtol=1e-5)
    to_late = [(f.type, f.meta.get("resource")) for ln, f, _c in ctl.posts if ln is late]
    assert to_late[0] == (w.MsgType.PROGRAM, None)  # the draw half, then the static labels
    assert (P.RESOURCE_TYPE, "labels") in to_late


# ---- both senders -------------------------------------------------------------------------------
websockets = pytest.importorskip("websockets")
from test_net_controls import ctl, make_server, mode, server  # noqa: E402,F401
from caliper import _net_programs as NP  # noqa: E402


@pytest.fixture
def watching(monkeypatch):
    monkeypatch.setattr(NP, "_watches", 0)
    NP.enable()
    yield
    NP.disable()


async def _collect(url, caps, until, timeout=60.0):
    """A viewer that announces ``caps`` and keeps every frame until ``until(frames)``."""
    from websockets.asyncio.client import connect
    frames = []
    async with connect(url, max_size=None, compression=None, ping_interval=None, close_timeout=1) as ws:
        await ws.send(w.encode(w.Hello("viewer", list(caps), {"name": "test", "version": "0"})))
        deadline = time.monotonic() + timeout
        while not until(frames) and time.monotonic() < deadline:
            try:
                frames.append(w.decode_frame(await asyncio.wait_for(ws.recv(), 0.05)))
            except asyncio.TimeoutError:
                continue
    return frames


@needs_device
def test_both_senders_relay_the_draw_half_and_send_resources_to_viewers_with_resources(
        server, watching, monkeypatch):
    monkeypatch.setattr(P, "SHARE", 0.5)            # a pass every few steps: a short run suffices
    torch.manual_seed(0)
    model = ArcModel().to("mps")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    P.attach(model, opt)
    stop = threading.Event()

    def train():
        i = 0
        while not stop.is_set() and i < 20000:
            _train(model, opt, 10, first=i)
            i += 10

    caps = [w.CAP_CONTROLS, w.CAP_SELECTION, w.CAP_PROGRAMS, w.CAP_INTERVENTIONS]
    arcs_in = lambda fs: sum(1 for f in fs if f.type == P.RESOURCE_TYPE and f.meta["resource"] == "arcs") >= 2

    async def both():
        with_res = asyncio.ensure_future(_collect(server.stream_url, caps + [w.CAP_RESOURCES], arcs_in))
        without = asyncio.ensure_future(_collect(server.stream_url, caps, lambda fs: with_res.done(), 90.0))
        await asyncio.sleep(0)
        return await with_res, await without

    trainer = None
    try:
        def start():
            nonlocal trainer
            # The viewers are attached before the install, thus the install relays to them.
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and len(
                    [ln for c in P._controls_ports() for ln in P._resource_links(c)]) < 1:
                time.sleep(0.01)
            _install()
            trainer = threading.Thread(target=train, daemon=True)
            trainer.start()
        threading.Thread(target=start, daemon=True).start()
        got, other = asyncio.run(asyncio.wait_for(both(), 120))
    finally:
        stop.set()
        if trainer is not None:
            trainer.join(30)
        P.detach()
        rt = P._STATE["rt"]
        if rt is not None:
            rt.close()
        P._STATE["rt"] = None
        P._CAD.reset()
    assert w.CAP_RESOURCES in next(f for f in got if f.type == w.MsgType.WELCOME).meta["capabilities"]
    mine = [f for f in got if f.type in (w.MsgType.PROGRAM, P.RESOURCE_TYPE)]
    assert mine[0].type == w.MsgType.PROGRAM and w.SECTION_DERIVE not in {s.name for s in mine[0].sections}
    _heads, rows = reference()
    arcs = [f for f in mine if f.type == P.RESOURCE_TYPE and f.meta["resource"] == "arcs"]
    assert len(arcs) >= 2
    assert all(f.meta["count"] == len(rows) and f.section(w.SECTION_RESOURCE).shape == (len(rows) * 24,)
               for f in arcs)
    assert not [f for f in other if f.type in (w.MsgType.PROGRAM, P.RESOURCE_TYPE)]
