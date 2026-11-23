"""The derive half of pieces in the script's process (caliper._programs, design/programmable-graphics.md).

A piece's instance runs several compute passes at the pass steps of the shared cadence and sends
each published resource as one RESOURCE message (type 0x0082) to the viewers that announced
``resources``, after the draw half (PROGRAM with the sections ``common``, ``derive`` and ``draw``). The checks of a
manifest run anywhere. The passes need an MPS device and the native module: the arcs piece of
the plan reads two fixed attention tensors of a small model, in place, and its arcs, scores and
labels are compared with a reference on the CPU. The senders' test runs both servers with a
viewer that keeps each raw frame.
"""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _programs as P
from caliper import _wire as w

import overhead_targets

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
    (lambda m, b: m["resources"][1].update(count=33554433), "piece.bad_manifest"),
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


def _resource_frame(rid, data, pid="r", stride=4, step=1):
    return w.encode_frame(P.RESOURCE_TYPE, {"program": pid, "resource": rid, "step": step,
                                            "count": len(data) // stride, "stride": stride},
                          [w.Section(w.SECTION_RESOURCE, int(w.DType.U8), (len(data),), memoryview(data))])


def test_a_viewer_behind_skips_a_whole_pass_and_gets_a_later_one_whole(monkeypatch):
    ctl = FakeControls()
    behind = FakeLink((w.CAP_PROGRAMS, w.CAP_RESOURCES))
    ctl.server.producer.links.append(behind)
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    p = P._Program("r", {}, "", None, "")
    frames = [_resource_frame("a", b"abcd" * 64), _resource_frame("b", b"efgh" * 64)]
    queued = [P.RESOURCE_BACKLOG_BYTES - len(frames[0]) - 10]   # room for the first frame only
    behind.backlog = lambda: queued[0]
    P._send_resources(p, frames)
    got = [(ln, f.type, f.meta.get("resource")) for ln, f, _c in ctl.posts]
    assert got[0][1] == w.MsgType.PROGRAM            # the draw half first, to both viewers
    assert [r for ln, t, r in got if ln is ctl.link and t == P.RESOURCE_TYPE] == ["a", "b"]
    assert [r for ln, t, r in got if ln is behind and t == P.RESOURCE_TYPE] == []   # no pass in part
    assert p.skipped == 1 and p.error is None
    queued[0] = 0                                    # it caught up: the next pass goes whole
    ctl.posts.clear()
    P._send_resources(p, frames)
    assert [f.meta["resource"] for ln, f, _c in ctl.posts if ln is behind] == ["a", "b"]


def test_a_version_of_one_part_reaches_a_viewer_with_an_empty_queue_in_one_frame(monkeypatch):
    """262,144 points of 32 bytes, 8 MiB, PART_BYTES: the largest version that goes in one frame."""
    ctl = FakeControls()
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    p = P._Program("points", {}, "", None, "")
    P._send_resources(p, [_resource_frame("points", bytes(262144 * 32), "points", 32)])
    res = [f for _l, f, _c in ctl.posts if f.type == P.RESOURCE_TYPE]
    assert len(res) == 1 and res[0].meta["count"] == 262144
    assert len(res[0].section(w.SECTION_RESOURCE).data) == 262144 * 32
    assert p.error is None and p.skipped == 0
    assert P.PART_BYTES == 8 << 20 == 262144 * 32 and P.RESOURCE_BACKLOG_BYTES == P.PART_BYTES + (64 << 10)


class _Runtime:
    """The resources of a fake runtime: id -> bytes; each call of resource() is counted."""

    def __init__(self, data, stride=4):
        self.data, self.stride, self.calls = data, stride, 0

    def resource(self, handle, rid):
        self.calls += 1
        d = self.data[rid]
        return d, len(d) // self.stride, self.stride, 0, 0, None


def _piece_program(pid, ids, monkeypatch=None):
    p = P._Program(pid, {}, "", None, "")
    p.resources = [{"id": i, "kind": "buffer", "publish": True} for i in ids]
    p.written, p.sent_static, p.last_step, p.handle, p.state = set(ids), True, 3, 1, "running"
    if monkeypatch is not None:
        monkeypatch.setitem(P._PROGS, pid, p)
    return p


def test_versions_above_the_published_bytes_are_a_named_refusal_not_a_silent_drop():
    """One resource above PUBLISHED_RESOURCE_MAX (128 MiB), or the resources of a pass above
    PUBLISHED_BYTES_MAX (512 MiB) together, from a manifest that skipped the app's check."""
    assert P.PUBLISHED_RESOURCE_MAX == w.RESOURCE_VERSION_MAX_BYTES == 128 << 20
    assert P.PUBLISHED_BYTES_MAX == 512 << 20
    over = bytearray(P.PUBLISHED_RESOURCE_MAX + 4)
    with pytest.raises(P._Refused) as e:
        P._resource_frames(_Runtime({"a": over}), _piece_program("big", ["a"]))
    assert e.value.error["code"] == "program.resource_too_large"
    del over
    third = bytearray(96 << 20)                      # six versions pass 512 MiB together
    with pytest.raises(P._Refused) as e:
        P._resource_frames(_Runtime(dict.fromkeys("abcdef", third)), _piece_program("big", "abcdef"))
    assert e.value.error["code"] == "program.resource_too_large"
    frames = P._resource_frames(_Runtime({"a": third, "b": third}), _piece_program("big", "ab"))
    assert len(frames) == 24 and all(f.split["parts"] == 12 for f in frames)   # 96 MiB in parts of 8 MiB


def test_a_large_version_goes_in_parts_within_the_backlog_and_comes_back_whole(monkeypatch):
    """A version of 40 MiB and a small one: the pass's first chunk goes with the pass, the others at
    the next steps, each only while the viewer's queue stays within RESOURCE_BACKLOG_BYTES; the
    viewer takes no other pass meanwhile and the pass makes no copy for it; the parts come back
    whole with the small version after them."""
    ctl = FakeControls()
    ln = ctl.link
    queued = [0]
    ln.backlog = lambda: queued[0]
    posted = ctl.post

    def post(link, frame, cap):                      # a queue that keeps what it gets until drained
        queued[0] += len(frame)
        posted(link, frame, cap)
    ctl.post = post
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    monkeypatch.setattr(P, "_OUTBOX", {})
    big = np.arange((40 << 20) // 4, dtype=np.uint32).tobytes()
    rt = _Runtime({"big": big, "info": np.array([7, 8, 9, 10], np.uint32).tobytes()}, stride=16)
    p = _piece_program("large", ["big", "info"], monkeypatch)
    frames = P._resource_frames(rt, p)
    assert rt.calls == 2 and len(frames) == 6        # five parts of 8 MiB, then info in one frame
    P._send_resources(p, frames)
    res = lambda: [w.Resource.from_frame(f) for _l, f, _c in ctl.posts if f.type == P.RESOURCE_TYPE]  # noqa: E731
    assert [r.split["part"] for r in res()] == [0]   # the draw half, then the first chunk
    assert id(ln) in P._OUTBOX
    extra = P._CAD.extra
    for k in range(1, 5):
        P._pump()                                    # the queue holds a part: nothing more goes
        assert len(res()) == k and queued[0] > P.RESOURCE_BACKLOG_BYTES - P.PART_BYTES
        # Meanwhile the viewer takes no other pass, and a pass for it makes no copy.
        other = _piece_program("other", ["info"], monkeypatch)
        P._send_resources(other, [_resource_frame("info", b"abcd" * 4, "other", 16)])
        assert other.skipped == 1 and ln in other.behind
        assert P._resource_frames(rt, p) == [] and rt.calls == 2
        queued[0] = 0                                # the viewer drained its queue
        P._pump()
    assert id(ln) not in P._OUTBOX and P._CAD.extra > extra
    got = res()
    assert [r.resource for r in got] == ["big"] * 5 + ["info"]
    sr = w.SplitResources()
    wholes = [x for x in (sr.add(r) for r in got) if x is not None]
    assert [x.resource for x in wholes] == ["big", "info"]
    assert bytes(wholes[0].data) == big and wholes[0].count == len(big) // 16


def test_an_outbox_goes_with_its_viewer_or_its_program(monkeypatch):
    ctl = FakeControls()
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    monkeypatch.setattr(P, "_OUTBOX", {})
    rt = _Runtime({"big": bytes(20 << 20)}, stride=16)
    p = _piece_program("large", ["big"], monkeypatch)
    P._send_resources(p, P._resource_frames(rt, p))
    ctl.link.backlog = lambda: P.RESOURCE_BACKLOG_BYTES      # a full queue: the outbox waits
    P._pump()
    assert id(ctl.link) in P._OUTBOX
    ctl.link.closed = True
    P._pump()
    assert not P._OUTBOX
    ctl.link.closed = False
    ctl.link.backlog = lambda: 0
    P._send_resources(p, P._resource_frames(rt, p))
    assert id(ctl.link) in P._OUTBOX
    ctl.link.backlog = lambda: P.RESOURCE_BACKLOG_BYTES
    monkeypatch.delitem(P._PROGS, "large")           # removed
    P._pump()
    assert not P._OUTBOX


def test_a_relay_frame_carries_common_derive_and_draw_in_order():
    p = P._Program("rel", {}, "fn derive_text() {}", None, "")
    p.piece, p.manifest = {"name": "n", "version": 1, "hash": "sha256:" + "a" * 64}, {"name": "n"}
    p.common, p.draw = "fn common_text() {}", "fn draw_text() {}"
    f = w.decode_frame(P._relay_frame(p))
    assert [s.name for s in f.sections] == [w.SECTION_COMMON, w.SECTION_DERIVE, w.SECTION_DRAW]
    assert bytes(f.section(w.SECTION_DERIVE).data).decode() == "fn derive_text() {}"
    assert w.Program.from_frame(f).derive == "fn derive_text() {}"
    p.wgsl = ""
    f = w.decode_frame(P._relay_frame(p))
    assert [s.name for s in f.sections] == [w.SECTION_COMMON, w.SECTION_DRAW]


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

    # The draw half goes first, with all three sections.
    first = ctl.posts[0][1]
    assert first.type == w.MsgType.PROGRAM
    assert first.meta["manifest"]["name"] == "attention_arcs" and first.meta["bind"] == BIND
    assert {s.name for s in first.sections} == {w.SECTION_COMMON, w.SECTION_DERIVE, w.SECTION_DRAW}
    assert bytes(first.section(w.SECTION_DERIVE).data).decode() == DERIVE
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
    failure = overhead_targets.overhead_share("tower_arcs", st["overhead_share"], P.SHARE)
    assert failure is None, failure
    print(f"arcs: {st['runs']} runs in 600 steps, every {st['every']}, share {st['overhead_share']:.4f}, "
          f"step {st['step_ms']:.3f} ms, cost {st['last_cost_ms']:.3f} ms, pass {st['last_pass_ms']} ms")
    assert torch.equal(model.attn0, before)          # read in place, never written


LOOK_DERIVE = """@compute @workgroup_size(1)
fn look() {
    shape[0] = shapes.extra;
    shape[1] = shapes.attn0;
    shape[2] = vec4<u32>(bitcast<u32>(extra[5u]), u32(attn0[0] >= 0.0), 0u, 0u);
}
"""
LOOK = {"name": "look", "version": 1,
        "inputs": [{"name": "attn0", "kind": "tensor", "rank_min": 4, "rank_max": 4},
                   {"name": "extra", "kind": "tensor", "dtype": "i32", "optional": True}],
        "resources": [{"id": "shape", "kind": "buffer", "element": "vec4<u32>", "count": 3}],
        "derive": {"passes": [{"entry": "look", "writes": ["shape"]}]}}


TYPED_DERIVE = """@compute @workgroup_size(1)
fn run() {
    for (var i = 0u; i < 16u; i = i + 1u) {
        out[i] = vec4<f32>(f32(depth_at(i)), f32(depth_type), f32(grey_at(i)), f32(shapes.depth__len));
    }
    // The grey image (H, W) as an r8unorm texture of its size, four texels a word.
    tex_size = vec2<u32>(shapes.grey.w, shapes.grey.z);
    for (var k = 0u; k < 4u; k = k + 1u) {
        tex[k] = grey[k];
    }
    // The first 5 depths as u16 elements, two a word.
    ids[0] = depth_at(0u) | (depth_at(1u) << 16u);
    ids[1] = depth_at(2u) | (depth_at(3u) << 16u);
    ids[2] = depth_at(4u);
    atomicStore(&ids_count, 5u);
}
"""
TYPED = {"name": "typed", "version": 1,
         "inputs": [{"name": "depth", "kind": "tensor", "dtype": "u16|u8", "rank_min": 2, "rank_max": 2},
                    {"name": "grey", "kind": "tensor", "dtype": "u8", "rank_min": 2, "rank_max": 2}],
         "resources": [{"id": "out", "kind": "buffer", "element": "vec4<f32>", "count": 16},
                       {"id": "tex", "kind": "texture", "format": "r8unorm", "width": 8, "height": 8,
                        "texels": 40},
                       {"id": "ids", "kind": "buffer", "element": "u16", "count": 16, "counted": True}],
         "derive": {"passes": [{"entry": "run", "writes": ["out", "tex", "ids"]}]}}


@needs_device
def test_u8_and_u16_tensors_bind_in_place_and_a_texture_and_narrow_buffer_carry_their_size(attached):
    """A u16 tensor from DLPack and a u8 tensor reach the derive half without a copy through their
    checked accessors; an r8unorm texture's version carries the size that the derive half set and its
    format, and a u16 buffer its elements, in the RESOURCE frames."""
    model, opt, ctl = attached
    # Made on the CPU, because MPS has no arange of u16, and moved to the device as a training tensor is.
    model.register_buffer("depth16", torch.arange(1000, 1015, dtype=torch.int32).to(torch.uint16).reshape(3, 5).to("mps"))
    model.register_buffer("grey8", torch.arange(1, 7, dtype=torch.uint8).reshape(2, 3).to("mps"))
    pin = {"name": "typed", "version": 1, "hash": "sha256:" + "0" * 64, "from": "workspace"}
    item = P.install_piece("typed", piece=pin, manifest=TYPED, bind={"depth": "live:depth16", "grey": "live:grey8"},
                           params={}, panel=None, segments=[], common="", derive=TYPED_DERIVE, draw="")
    try:
        assert item["state"] in ("compiled", "waiting"), item["error"]
        _train(model, opt, 300)
        st = next(x for x in P.status() if x["id"] == "typed")
        assert st["state"] == "running", st["error"]
        out = _resources(ctl, "out")[-1]
        v = np.frombuffer(bytes(out.section(w.SECTION_RESOURCE).data), np.float32).reshape(16, 4)
        assert v[:15, 0].tolist() == list(range(1000, 1015))   # the u16 depths, exact
        assert v[15, 0] == 0                                   # past the 15 elements: zero
        assert set(v[:, 1].tolist()) == {4.0}                  # CALIPER_U16
        assert v[:6, 2].tolist() == [1, 2, 3, 4, 5, 6] and v[6, 2] == 0
        assert set(v[:, 3].tolist()) == {15.0}
        tex = _resources(ctl, "tex")[-1]
        assert (tex.meta["width"], tex.meta["height"], tex.meta["format"]) == (3, 2, "r8unorm")
        assert tex.meta["count"] == 2 and tex.meta["stride"] == 4
        assert bytes(tex.section(w.SECTION_RESOURCE).data)[:6] == bytes([1, 2, 3, 4, 5, 6])
        ids = _resources(ctl, "ids")[-1]
        assert (ids.meta["element"], ids.meta["elements"], ids.meta["count"]) == ("u16", 5, 3)
        assert np.frombuffer(bytes(ids.section(w.SECTION_RESOURCE).data), np.uint16)[:5].tolist() == \
            list(range(1000, 1005))
    finally:
        P.remove("typed")
    # A tensor of a dtype that the input does not name is refused with piece.bad_binding.
    model.register_buffer("depth32", torch.zeros(3, 5, dtype=torch.int32, device="mps"))
    item = P.install_piece("typed2", piece=pin, manifest=TYPED, bind={"depth": "live:depth32", "grey": "live:grey8"},
                           params={}, panel=None, segments=[], common="", derive=TYPED_DERIVE, draw="")
    try:
        assert item["state"] == "refused"
        assert item["error"]["code"] == "piece.bad_binding"
    finally:
        P.remove("typed2")


@needs_device
def test_an_optional_tensor_input_left_unbound_reads_as_empty_and_a_required_one_is_refused(attached):
    import copy
    model, opt, ctl = attached
    pin = {"name": "look", "version": 1, "hash": "sha256:" + "0" * 64, "from": "workspace"}

    def install(pid, manifest):
        return P.install_piece(pid, piece=pin, manifest=manifest, bind={"attn0": "live:attn0"}, params={},
                               panel=None, segments=[], common="", derive=LOOK_DERIVE, draw="")
    item = install("look", LOOK)
    try:
        assert item["state"] in ("compiled", "waiting"), item["error"]
        _train(model, opt, 300)
        st = next(x for x in P.status() if x["id"] == "look")
        assert st["state"] == "running", st["error"]
        frames = [f for _, f, _ in ctl.posts if f.type == P.RESOURCE_TYPE and f.meta["resource"] == "shape"]
        assert frames
        v = np.frombuffer(bytes(frames[-1].section(w.SECTION_RESOURCE).data), np.uint32).reshape(3, 4)
        assert v[0].tolist() == [0, 0, 0, 0]                 # the unbound input's shape
        assert v[1].tolist() == [1, H, T, T]                 # the bound input's shape
        assert v[2].tolist() == [0, 1, 0, 0]                 # its reads give zero
    finally:
        P.remove("look")
    required = copy.deepcopy(LOOK)
    required["inputs"][1].pop("optional")
    item = install("strict", required)
    try:
        assert item["state"] == "refused"
        assert item["error"]["code"] == "piece.bad_binding"
    finally:
        P.remove("strict")


@needs_device
def test_a_piece_that_publishes_64_mib_every_pass_keeps_its_overhead_share(attached):
    """A published buffer of 4,194,304 vec4<u32>, 64 MiB, written at every pass: it goes to the viewer
    in eight parts, one a step while the viewer drains its queue, and the cadence keeps the passes, the
    copies and the parts within the share of the training time."""
    model, opt, ctl = attached
    ln = ctl.link
    queued = [0]
    ln.backlog = lambda: queued[0]
    parts: dict = {}                                 # version -> its part frames, without a copy

    def post(link, frame, cap):                      # the viewer drains its queue at each step
        queued[0] += len(frame)
        f = w.decode_frame(frame)
        if f.type == P.RESOURCE_TYPE and "split" in f.meta:
            parts.setdefault(f.meta["split"]["version"], []).append(frame)
            for v in sorted(parts)[:-2]:
                del parts[v]
    ctl.post = post
    n = 4194304
    big = {"name": "big_words", "version": 1, "inputs": [{"name": "t", "kind": "tensor"}],
           "resources": [{"id": "words", "kind": "buffer", "element": "vec4<u32>", "count": n}],
           "derive": {"memory_mb": 128,
                      "passes": [{"entry": "fill", "workgroups": [1024, 1, 1], "writes": ["words"]}]}}
    derive = """@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = u32(t[0] * 0.0);
    for (var k = 0u; k < 64u; k++) {
        let i = id.x * 64u + k;
        words[i] = vec4<u32>(i, i + 1u, i + 2u, 7u + z);
    }
}
"""
    item = P.install_piece("big_words", piece=dict(PIN, name="big_words"), manifest=big,
                           bind={"t": "live:attn0"}, derive=derive)
    assert item["state"] in ("compiled", "waiting"), item["error"]
    x = torch.randn(256, 64, device="mps")
    steps = 12000                                    # long enough for the cadence to settle: a step takes ~0.15 ms
    for i in range(steps):
        loss = model(x)
        opt.zero_grad()
        loss.backward()
        caliper.step(i)
        opt.step()
        queued[0] = 0
    st = next(x for x in P.status() if x["id"] == "big_words")
    assert st["state"] == "running", st["error"]
    whole = None
    for v in sorted(parts, reverse=True):
        sr = w.SplitResources()
        got = [sr.add(w.decode(f)) for f in parts[v]]
        if got[-1] is not None:
            whole = got[-1]
            break
    assert whole is not None and whole.count == n and whole.stride == 16
    words = np.frombuffer(bytes(whole.data), np.uint32).reshape(n, 4)
    i = np.arange(n, dtype=np.uint32)
    assert np.array_equal(words[:, 0], i) and np.array_equal(words[:, 2], i + 2) and (words[:, 3] == 7).all()
    failure = overhead_targets.overhead_share("big_words", st["overhead_share"], P.SHARE)
    print(f"big_words: {st['runs']} runs in {steps} steps, every {st['every']}, share {st['overhead_share']:.4f}, "
          f"step {st['step_ms']:.3f} ms, cost {st['last_cost_ms']:.3f} ms, pass {st['last_pass_ms']} ms, "
          f"passes skipped for a busy viewer {P._PROGS['big_words'].skipped}")
    assert failure is None, failure


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


@needs_device
def test_a_piece_whose_versions_no_viewer_takes_stops_with_a_named_error(attached, monkeypatch):
    """A manifest that skipped the app's check exceeds a reduced viewer budget."""
    model, opt, ctl = attached
    monkeypatch.setattr(P, "PUBLISHED_BYTES_MAX", 256 << 20)
    ids = [f"v{k}" for k in range(5)]
    big = {"name": "big", "version": 1, "inputs": [{"name": "t", "kind": "tensor"}],
           "resources": [{"id": i, "kind": "buffer", "element": "vec4<u32>", "count": 4194304} for i in ids],
           "derive": {"memory_mb": 512,
                      "passes": [{"entry": "fill", "workgroups": [1, 1, 1], "writes": ids}]}}
    derive = """@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    v0[id.x] = vec4<u32>(u32(t[id.x]));
    v1[id.x] = v0[id.x];
    v2[id.x] = v0[id.x];
    v3[id.x] = v0[id.x];
    v4[id.x] = v0[id.x];
}
"""
    item = P.install_piece("big", piece=dict(PIN, name="big"), manifest=big, bind={"t": "live:attn0"},
                           derive=derive)
    assert item["state"] in ("compiled", "waiting"), item["error"]
    _train(model, opt, 300)
    st = next(x for x in P.status() if x["id"] == "big")
    assert st["state"] == "stopped"
    assert st["error"]["code"] == "program.resource_too_large"
    assert not [f for _l, f, _c in ctl.posts if f.type == P.RESOURCE_TYPE]


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
    assert mine[0].type == w.MsgType.PROGRAM and w.SECTION_DERIVE in {s.name for s in mine[0].sections}
    _heads, rows = reference()
    arcs = [f for f in mine if f.type == P.RESOURCE_TYPE and f.meta["resource"] == "arcs"]
    assert len(arcs) >= 2
    assert all(f.meta["count"] == len(rows) and f.section(w.SECTION_RESOURCE).shape == (len(rows) * 24,)
               for f in arcs)
    assert not [f for f in other if f.type in (w.MsgType.PROGRAM, P.RESOURCE_TYPE)]


class _Reader(threading.Thread):
    """A viewer with ``resources`` on a thread of its own: it keeps the program and step of each
    RESOURCE and the close code of a connection that the server closed."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.got = []
        self.wholes = []                             # (program, step, bytes, sha256) of each whole version
        self.parts = w.SplitResources()
        self.closed = None
        self.stop = threading.Event()

    def run(self):
        asyncio.run(self._read())

    async def _read(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed
        caps = [w.CAP_CONTROLS, w.CAP_SELECTION, w.CAP_PROGRAMS, w.CAP_RESOURCES]
        try:
            async with connect(self.url, max_size=None, compression=None, ping_interval=None,
                               close_timeout=1) as ws:
                await ws.send(w.encode(w.Hello("viewer", caps, {"name": "test", "version": "0"})))
                while not self.stop.is_set():
                    try:
                        f = w.decode_frame(await asyncio.wait_for(ws.recv(), 0.05))
                    except asyncio.TimeoutError:
                        continue
                    if f.type == P.RESOURCE_TYPE:
                        self.got.append((f.meta["program"], f.meta["step"]))
                        r = self.parts.add(w.Resource.from_frame(f))
                        if r is not None:
                            self.wholes.append((r.program, r.step, len(r.data),
                                                hashlib.sha256(r.data).hexdigest()))
        except ConnectionClosed as e:
            self.closed = e.rcvd.code if e.rcvd is not None else -1


def _wait(cond, what, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def test_both_senders_carry_a_version_in_parts_to_a_viewer_whole(server, watching, monkeypatch):
    """A version of 40 MiB goes in five parts of 8 MiB through either sender, one chunk while the
    viewer's queue has room, and the viewer puts each version together whole and stays connected."""
    reader = _Reader(server.stream_url)
    reader.start()
    try:
        _wait(lambda: [ln for c in P._controls_ports() if c.server is server for ln in P._resource_links(c)],
              "the viewer attaches")
        monkeypatch.setattr(P, "_OUTBOX", {})
        p = _piece_program("big", ["big"], monkeypatch)
        p.kind = "piece"
        data = np.arange((40 << 20) // 4, dtype=np.uint32).tobytes()
        rt = _Runtime({"big": data}, stride=16)
        for step in (1, 2):
            p.last_step = step
            P._publish_pass([(p, P._resource_frames(rt, p))])
            _wait(lambda: P._pump() is None and (reader.closed is not None or
                                                  any(x[1] == step for x in reader.wholes)),
                  f"the viewer puts the version of step {step} together")
            assert reader.closed is None, f"the server closed the viewer with {reader.closed}"
        assert [x[:3] for x in reader.wholes] == [("big", 1, len(data)), ("big", 2, len(data))]
        assert reader.wholes[1][3] == hashlib.sha256(data).hexdigest()
        assert len(reader.got) == 10 and reader.parts.dropped == reader.parts.discarded == 0
    finally:
        reader.stop.set()
        reader.join(10)


def test_two_programs_of_one_chunk_each_reach_one_viewer_in_turn(server, watching, monkeypatch):
    """Two instances that publish 8 MiB each at one pass, as two of the library's points@1 do. The frames
    posted for the viewer and not queued yet count toward its bound, thus the second program skips the
    pass where both would overflow the 16 MiB queue, the viewer stays connected, and the program that it
    skipped goes first at the next pass."""
    reader = _Reader(server.stream_url)
    reader.start()
    try:
        _wait(lambda: [ln for c in P._controls_ports() if c.server is server for ln in P._resource_links(c)],
              "the viewer attaches")
        ctl, = [c for c in P._controls_ports() if c.server is server]
        gate = threading.Lock()
        flush = ctl.flush

        def gated(*a, **kw):
            with gate:
                return flush(*a, **kw)
        monkeypatch.setattr(ctl, "flush", gated)
        progs = [P._Program(pid, {}, "", None, "") for pid in ("a", "b")]
        for p in progs:
            p.kind = "piece"
        for step in range(1, 5):
            with gate:                                   # the whole pass is posted before one flush
                P._publish_pass([(p, [_resource_frame("points", bytes(P.PART_BYTES), p.id, 32, step)])
                                 for p in progs])
            _wait(lambda: reader.closed is not None or any(s == step for _p, s in reader.got),
                  f"the viewer gets a version of step {step}")
            assert reader.closed is None, f"the server closed the viewer with {reader.closed}"
        assert reader.got == [("a", 1), ("b", 2), ("a", 3), ("b", 4)]
        assert [p.skipped for p in progs] == [2, 2]
    finally:
        reader.stop.set()
        reader.join(10)


def test_published_textures_have_a_separate_byte_limit(monkeypatch):
    assert P.PUBLISHED_TEXTURE_MAX == w.RESOURCE_TEXTURE_VERSION_MAX_BYTES == 256 << 20
    monkeypatch.setattr(P, "PUBLISHED_RESOURCE_MAX", 8)
    monkeypatch.setattr(P, "PUBLISHED_TEXTURE_MAX", 16)
    class TextureRuntime:
        def resource(self, handle, rid):
            return bytes(16), 4, 4, 2, 2, None
    p = _piece_program("tex", ["a"])
    p.resources[0]["kind"] = "texture"
    frames = P._resource_frames(TextureRuntime(), p)
    assert len(frames) == 1
    assert w.decode(frames[0]).width == 2
    p.resources[0]["kind"] = "buffer"
    with pytest.raises(P._Refused, match="in one resource"):
        P._resource_frames(TextureRuntime(), p)


@pytest.mark.parametrize("delivery", ["deferred", "outbox"])
def test_late_viewer_keeps_static_resources_when_the_draw_frame_fills_the_queue(monkeypatch, delivery):
    """A draw frame must not consume the bytes reserved for a late viewer's static resources."""
    ctl = FakeControls()
    link = ctl.link
    monkeypatch.setattr(P, "_OUTBOX", {})
    p = _piece_program("static", ["labels"], monkeypatch)
    resource = _resource_frame("labels", bytes(128), p.id)
    p.static_frames["labels"] = [resource]
    program = P._relay_frame(p)
    budget = (len(program) + len(resource) if delivery == "deferred" else max(len(program), len(resource))) + 16
    monkeypatch.setattr(P, "RESOURCE_BACKLOG_BYTES", budget)
    queued = [20 if delivery == "deferred" else 0]
    link.backlog = lambda: queued[0]
    original_post = ctl.post
    def post(viewer, frame, cap):
        queued[0] += len(frame)
        assert queued[0] <= budget
        original_post(viewer, frame, cap)
    ctl.post = post

    P._relay_to(p, ctl, [link])
    if delivery == "deferred":
        assert ctl.posts == []
        assert link not in p.relayed
    else:
        assert [frame.type for _, frame, _ in ctl.posts] == [w.MsgType.PROGRAM]
        assert id(link) in P._OUTBOX
    queued[0] = 0
    P._relay_to(p, ctl, [link])
    P._pump()
    assert [frame.type for _, frame, _ in ctl.posts] == [w.MsgType.PROGRAM, P.RESOURCE_TYPE]
    assert ctl.posts[1][1].meta["resource"] == "labels"
    assert link in p.relayed and id(link) not in P._OUTBOX
    queued[0] = 0
    P._relay_to(p, ctl, [link])
    P._pump()
    assert len(ctl.posts) == 2
