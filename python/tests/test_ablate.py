"""caliper._ablate and the interventions of caliper._controls (design/gpu-programs.md,
Interventions): the points of a model, the ablation of a module and of a head through one hook,
the loss of a fixed batch that changes while an ablation is set and returns exactly when it is
cleared, no hook left after a clear, an error or the end, and a hook error that never stops a
training step.

Needs torch and pytest:

    PYTHONPATH=python .venv/bin/python -m pytest --noconftest python/tests/test_ablate.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from caliper import _ablate  # noqa: E402
from caliper import _controls as C  # noqa: E402
from caliper._adapter import TORCH  # noqa: E402

nn = torch.nn


class Attention(nn.Module):
    """A GPT-style attention block: n_head and an output projection c_proj."""

    def __init__(self, dim=8, n_head=2):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=-1)
        h = self.n_head
        q, k, v = (z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v))
        y = torch.softmax(q @ k.transpose(-1, -2) / (d // h) ** 0.5, dim=-1) @ v
        return self.c_proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attn = Attention(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(inplace=True), nn.Linear(16, dim))

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class Tiny(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim), Block(dim)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x).mean(dim=(1, 2))


def _model():
    torch.manual_seed(0)
    return Tiny()


def _batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(4, 5, 8, generator=g), torch.randn(4, generator=g)


def _loss(model, batch):
    x, y = batch
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y)


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------
def test_points_list_each_module_and_each_head():
    model = _model()
    pts = TORCH.points(model)
    names = [p["point"] for p in pts]
    assert "" not in names                              # the root is not a point
    assert names[:3] == ["blocks", "blocks.0", "blocks.0.attn"]
    assert names[3:5] == ["blocks.0.attn#0", "blocks.0.attn#1"]
    assert "blocks.1.mlp.1" in names and "head" in names
    head = pts[names.index("blocks.0.attn#1")]
    assert head == {"point": "blocks.0.attn#1", "kind": "head", "module": "blocks.0.attn",
                    "head": 1, "type": "Attention", "shape": None}
    mod = pts[names.index("blocks.0.attn.c_proj")]
    assert mod["kind"] == "module" and mod["head"] is None and mod["type"] == "Linear"
    assert all(p["shape"] is None for p in pts)         # before a forward pass


def test_shapes_after_one_forward_pass_and_no_hook_left():
    model = _model()
    off = _ablate.record_shapes(model)
    assert _hooks(model) > 0
    _loss(model, _batch())
    off()
    assert _hooks(model) == 0
    shapes = {p["point"]: p["shape"] for p in TORCH.points(model)}
    assert shapes["blocks.0.attn"] == [4, 5, 8]
    assert shapes["head"] == [4, 5, 1]
    assert shapes["blocks.0.attn#0"] is None            # a head has no shape of its own


def test_multihead_attention_gives_no_heads():
    # nn.MultiheadAttention passes out_proj.weight to its functional forward and never calls
    # out_proj, so a hook there would never run: its heads are not points.
    model = nn.Sequential(nn.MultiheadAttention(8, 2, batch_first=True))
    names = [p["point"] for p in TORCH.points(model)]
    assert names == ["0", "0.out_proj"]


def test_heads_need_one_projection_that_divides():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.o_proj = nn.Linear(8, 8)
            self.proj = nn.Linear(8, 8)

    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 3
            self.out_proj = nn.Linear(8, 8)

    class Flag(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_head = True                          # a bool is not a head count
            self.proj = nn.Linear(8, 8)

    model = nn.ModuleDict({"two": Two(), "odd": Odd(), "flag": Flag()})
    assert not [p for p in TORCH.points(model) if p["kind"] == "head"]


@pytest.mark.parametrize("point", ["", "nope", "blocks.0.attn#2", "blocks.0.attn#01",
                                   "blocks.0.attn#-1", "blocks.0.mlp#0", "#0"])
def test_unknown_point_raises_key_error(point):
    with pytest.raises(KeyError):
        TORCH.ablate(_model(), point)


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------
def test_module_ablation_changes_the_loss_and_clearing_returns_it_exactly():
    model, batch = _model(), _batch()
    base = _loss(model, batch)
    off = TORCH.ablate(model, "blocks.0.attn")
    assert _hooks(model) == 1
    ablated = _loss(model, batch)
    assert not torch.equal(ablated, base)
    off()
    assert _hooks(model) == 0
    assert torch.equal(_loss(model, batch), base)
    off()                                               # a second call does nothing
    assert _hooks(model) == 0


def test_module_ablation_zeroes_the_output():
    model, (x, _) = _model(), _batch()
    off = TORCH.ablate(model, "blocks.0.attn")
    with torch.no_grad():
        out = model.blocks[0].attn(x)
    off()
    assert out.shape == (4, 5, 8) and torch.count_nonzero(out) == 0


def test_head_ablation_equals_zeroing_that_slice_of_the_projection_input():
    model, (x, _) = _model(), _batch()
    attn = model.blocks[0].attn
    inputs = []
    h = attn.c_proj.register_forward_pre_hook(lambda m, a: inputs.append(a[0].clone()))
    with torch.no_grad():
        base = attn(x)
    h.remove()
    off = TORCH.ablate(model, "blocks.0.attn#1")
    with torch.no_grad():
        ablated = attn(x)
    off()
    want_in = inputs[0].clone()
    want_in[..., 4:8] = 0
    with torch.no_grad():
        want = attn.c_proj(want_in)
    assert torch.allclose(ablated, want) and not torch.allclose(ablated, base)
    assert _hooks(model) == 0


def test_backward_works_when_the_ablated_output_is_all_the_loss_sees():
    model, (x, y) = _model(), _batch()
    off = TORCH.ablate(model, "head")
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()                                     # no "does not require grad" error
    off()
    assert model.head.weight.grad is not None
    assert torch.count_nonzero(model.head.weight.grad) == 0


def test_in_place_op_after_an_ablated_module_still_works():
    model, (x, y) = _model(), _batch()
    off = TORCH.ablate(model, "blocks.0.mlp.0")         # followed by ReLU(inplace=True)
    torch.nn.functional.mse_loss(model(x), y).backward()
    off()


def test_nan_input_gives_exact_zeros():
    lin = nn.Linear(3, 3)
    off = TORCH.ablate(nn.Sequential(lin), "0")
    with torch.no_grad():
        lin.bias.fill_(float("nan"))
        out = lin(torch.ones(2, 3))
    off()
    assert torch.count_nonzero(out) == 0 and not torch.isnan(out).any()


def test_a_failing_hook_never_stops_the_forward_pass(monkeypatch):
    model, batch = _model(), _batch()
    base = _loss(model, batch)
    a = TORCH.ablate(model, "blocks.0.attn")

    def boom(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(a, "_mask", boom)
    assert torch.equal(_loss(model, batch), base)       # the tensor is left as it was
    assert a.errors == 1 and "boom" in a.error
    a()
    assert _hooks(model) == 0


# ---------------------------------------------------------------------------
# the interventions of caliper._controls at the step boundary
# ---------------------------------------------------------------------------
def _intervene(ctl, req):
    with ctl.lock:
        out = ctl._intervene(req)
        ctl._update_busy()
    return out


def _train(model, batch, ctl, steps=1):
    """Plain training steps with the checkpoint, as caliper.watch(model, optimizer) runs them."""
    opt = torch.optim.SGD(model.parameters(), lr=0.0)    # lr 0: the weights stay, the loss is fixed
    x, y = batch
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        ctl.step()                                      # the checkpoint before optimizer.step()
        opt.step()


def test_controls_apply_an_ablation_at_the_next_step_and_clear_it_exactly():
    model, batch = _model(), _batch()
    ctl = C.Controls()
    handle = ctl.interventions_on(model)
    base = _loss(model, batch)
    _train(model, batch, ctl)                           # step 1: shapes recorded, hooks off
    assert _hooks(model) == 0
    assert _intervene(ctl, {"set": {"point": "blocks.1.attn#0", "op": "ablate"}}) is None
    assert _hooks(model) == 0                           # nothing before the next checkpoint
    assert ctl.summary()["interventions"][0]["state"] == "pending"
    _train(model, batch, ctl)                           # step 2: the hook goes on
    iv = ctl.summary()["interventions"][0]
    assert iv["state"] == "active" and iv["since_step"] == 3 and _hooks(model) == 1
    ablated = _loss(model, batch)
    assert not torch.equal(ablated, base)
    assert _intervene(ctl, {"clear": "blocks.1.attn#0"}) is None
    assert torch.equal(_loss(model, batch), ablated)    # still on until the next step
    _train(model, batch, ctl)
    assert _hooks(model) == 0 and ctl.summary()["interventions"] == []
    assert torch.equal(_loss(model, batch), base)
    handle.remove()
    assert ctl._iv_model is None


def test_controls_refuse_unknown_points_reserved_ops_and_bad_ranges():
    model = _model()
    ctl = C.Controls()
    assert _intervene(ctl, {"set": {"point": "head", "op": "ablate"}})[0] == C.ERR_NOT_SUPPORTED
    ctl.interventions_on(model)
    assert _intervene(ctl, {"set": {"point": "blocks.9", "op": "ablate"}})[0] == C.ERR_UNKNOWN_POINT
    assert _intervene(ctl, {"set": {"point": "head", "op": "steer"}})[0] == C.ERR_OP_RESERVED
    assert _intervene(ctl, {"set": {"point": "head", "op": "ablate", "from_step": 5,
                                    "until_step": 5}})[0] == C.ERR_BAD_RANGE
    assert _intervene(ctl, {"set": {"point": "head", "op": "ablate",
                                    "until_step": 1}})[0] == C.ERR_BAD_RANGE   # the next step is 1
    assert _intervene(ctl, {"clear": "blocks.9"})[0] == C.ERR_UNKNOWN_POINT
    assert ctl.summary()["interventions"] == []


def test_controls_keep_a_step_range():
    model, batch = _model(), _batch()
    ctl = C.Controls()
    ctl.interventions_on(model)
    _train(model, batch, ctl)                           # step 1
    assert _intervene(ctl, {"set": {"point": "head", "op": "ablate", "from_step": 4,
                                    "until_step": 6}}) is None
    states = []
    for _ in range(5):                                  # checkpoints 2..6 decide steps 3..7
        _train(model, batch, ctl)
        iv = ctl.summary()["interventions"][0]
        states.append((iv["state"], _hooks(model)))
    assert states == [("pending", 0), ("active", 1), ("active", 1), ("ended", 0), ("ended", 0)]
    assert ctl.summary()["interventions"][0]["since_step"] == 4


def test_a_failing_hook_ends_the_intervention_and_the_training_goes_on(monkeypatch):
    model, batch = _model(), _batch()
    ctl = C.Controls()
    ctl.interventions_on(model)
    _train(model, batch, ctl)
    _intervene(ctl, {"set": {"point": "blocks.0.attn", "op": "ablate"}})
    _train(model, batch, ctl)
    iv = next(iter(ctl._ivs.values()))
    monkeypatch.setattr(iv.hook, "_mask", lambda *a: 1 / 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _train(model, batch, ctl, steps=2)              # the step with the error still trains
    s = ctl.summary()["interventions"][0]
    assert s["state"] == "ended" and "ZeroDivisionError" in s["error"]
    assert _hooks(model) == 0
    assert any("failed and ended" in str(c.message) for c in caught)


def test_removing_the_watch_takes_every_hook_off():
    model, batch = _model(), _batch()
    ctl = C.Controls()

    class Handle:
        _handles: list = []

    h = Handle()
    h._handles = []
    ctl.interventions_on(model, h)
    _train(model, batch, ctl)
    for p in ("head", "blocks.0.attn#1", "blocks.1.mlp"):
        _intervene(ctl, {"set": {"point": p, "op": "ablate"}})
    _train(model, batch, ctl)
    assert _hooks(model) == 3
    for x in h._handles:
        x.remove()
    assert _hooks(model) == 0 and ctl.summary()["interventions"] == []
    assert not ctl._iv_due


def test_clear_all():
    model, batch = _model(), _batch()
    ctl = C.Controls()
    ctl.interventions_on(model)
    _train(model, batch, ctl)
    for p in ("head", "blocks.0.attn"):
        _intervene(ctl, {"set": {"point": p, "op": "ablate"}})
    _train(model, batch, ctl)
    assert _hooks(model) == 2
    _intervene(ctl, {"clear_all": True})
    _train(model, batch, ctl)
    assert _hooks(model) == 0 and ctl.summary()["interventions"] == []


# ---------------------------------------------------------------------------
# the wire: CONTROL op intervene and the CONTROLS key interventions
# ---------------------------------------------------------------------------
class _Transport:
    def __init__(self):
        self.out = []

    def broadcast(self, type_, meta, after_data=False):
        self.out.append((type_, meta))

    def reply(self, conn, meta):
        self.out.append(("reply", meta))

    def may_control(self, conn):
        return True

    def controllers(self):
        return 1


def test_wire_control_intervene_and_controls_interventions():
    w = pytest.importorskip("caliper._wire")
    if not hasattr(w, "OP_INTERVENE"):
        pytest.skip("this codec has no CONTROL op intervene (protocol 3.2 draft)")
    model, batch = _model(), _batch()
    ctl = C.Controls()
    t = _Transport()
    route = ctl.attach(t)
    ctl.interventions_on(model)
    _train(model, batch, ctl)
    meta = {"request_id": 7, "op": "intervene",
            "intervene": {"set": {"point": "blocks.0.attn#1", "op": "ablate"}}}
    reply = route.on_control("c", meta)
    w.ControlReply.from_frame(w.Frame(w.MsgType.CONTROL_REPLY, reply))
    assert reply["status"] == "ok" and reply["op"] == "intervene"
    bad = route.on_control("c", {"request_id": 8, "op": "intervene",
                                 "intervene": {"set": {"point": "blocks.9", "op": "ablate"}}})
    assert bad["status"] == "error" and bad["error"]["code"] == "intervene.unknown_point"
    _train(model, batch, ctl)
    controls = [m for ty, m in t.out if ty == w.MsgType.CONTROLS]
    last = w.Controls.from_frame(w.Frame(w.MsgType.CONTROLS, controls[-1])).to_meta()
    iv = last["interventions"]
    assert iv["active"] == [{"point": "blocks.0.attn#1", "op": "ablate", "from_step": None,
                             "until_step": None, "since_step": 3, "state": "active"}]
    pts = {p["point"]: p for p in iv["points"]}
    assert pts["blocks.0.attn"]["shape"] == [4, 5, 8]
    assert pts["blocks.0.attn#1"]["kind"] == "head"
    route.on_control("c", {"request_id": 9, "op": "intervene", "intervene": {"clear_all": True}})
    _train(model, batch, ctl)
    assert _hooks(model) == 0
    route.detach()


class _Controllers(_Transport):
    """A transport whose count of connections that may control the test sets."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def controllers(self):
        return self.n


class _Clock:
    """The time module with a monotonic clock that the test moves forward. The linger time is
    measured on it, so a slow machine cannot use the linger up between two steps."""
    ahead = 0.0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return time.monotonic() + self.ahead


def test_interventions_end_when_no_controller_stays_for_the_linger_time(monkeypatch):
    from caliper import _net_programs

    clock = _Clock()
    monkeypatch.setattr(C, "time", clock)
    monkeypatch.setattr(_net_programs, "LINGER_S", 600.0)
    monkeypatch.setattr(C, "LINGER_CHECK_S", 0.0)
    model, batch = _model(), _batch()
    ctl = C.Controls()
    t = _Controllers(1)
    route = ctl.attach(t)
    ctl.interventions_on(model)
    base = _loss(model, batch)
    _train(model, batch, ctl)
    assert _intervene(ctl, {"set": {"point": "blocks.1.attn", "op": "ablate"}}) is None
    _train(model, batch, ctl)
    assert _hooks(model) == 1
    t.n = 0                                             # the viewer that set it goes
    _train(model, batch, ctl)
    assert _hooks(model) == 1                           # inside the linger time
    t.n = 1                                             # a reload comes back in time
    _train(model, batch, ctl)
    t.n = 0
    _train(model, batch, ctl)
    clock.ahead += 400.0
    _train(model, batch, ctl)
    assert _hooks(model) == 1                           # the linger started again at the return
    clock.ahead += 700.0
    with pytest.warns(RuntimeWarning, match="interventions on blocks.1.attn ended"):
        _train(model, batch, ctl)
    assert _hooks(model) == 0 and torch.equal(_loss(model, batch), base)
    iv = ctl.summary()["interventions"][0]
    assert iv["state"] == "ended" and "no connection that may control" in iv["error"]
    assert ctl.summary()["interventions_lingered"] == 1
    controls = [m for ty, m in t.out if ty != "reply"]
    assert controls[-1]["interventions"]["active"][0]["state"] == "ended"
    route.detach()


def test_interventions_end_when_the_server_goes(monkeypatch):
    from caliper import _net_programs
    clock = _Clock()
    monkeypatch.setattr(C, "time", clock)
    monkeypatch.setattr(_net_programs, "LINGER_S", 600.0)
    monkeypatch.setattr(C, "LINGER_CHECK_S", 0.0)
    model, batch = _model(), _batch()
    ctl = C.Controls()
    route = ctl.attach(_Controllers(1))
    ctl.interventions_on(model)
    _train(model, batch, ctl)
    assert _intervene(ctl, {"set": {"point": "blocks.0.attn", "op": "ablate"}}) is None
    _train(model, batch, ctl)
    assert _hooks(model) == 1
    route.detach()                                      # no route at all: nobody may control
    _train(model, batch, ctl)
    clock.ahead += 700.0
    with pytest.warns(RuntimeWarning):
        _train(model, batch, ctl)
    assert _hooks(model) == 0


# ---------------------------------------------------------------------------
# per-filter ConvNd channel intervention points
# ---------------------------------------------------------------------------
class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.single = nn.Conv2d(4, 1, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        _ = self.single(h)
        return self.head(h.mean(dim=(-2, -1)))


def _conv_model():
    torch.manual_seed(0)
    return TinyConv()


def _conv_batch():
    g = torch.Generator().manual_seed(2)
    return torch.randn(3, 1, 6, 6, generator=g), torch.randint(0, 3, (3,), generator=g)


def test_conv_points_list_each_output_filter_and_skip_single_channel():
    model = _conv_model()
    pts = TORCH.points(model)
    by_name = {p["point"]: p for p in pts}
    for c in range(4):
        assert by_name[f"conv1#{c}"] == {
            "point": f"conv1#{c}",
            "kind": "module",
            "module": "conv1",
            "head": None,
            "type": "Conv2d",
            "shape": None,
        }
    assert "single" in by_name
    assert "single#0" not in by_name
    assert "conv1#4" not in by_name
    for bad in ("conv1#4", "conv1#01", "conv1#-1", "single#0"):
        with pytest.raises(KeyError):
            TORCH.ablate(model, bad)


def test_conv_channel_ablation_zeroes_only_one_filter_batched_and_unbatched():
    model, (x, y) = _conv_model(), _conv_batch()
    with torch.no_grad():
        base_b = model.conv1(x)
        base_u = model.conv1(x[0])
        base_logits = model(x)

    off = TORCH.ablate(model, "conv1#2")
    assert _hooks(model) == 1
    with torch.no_grad():
        abl_b = model.conv1(x)
        abl_u = model.conv1(x[0])
        abl_logits = model(x)

    assert torch.count_nonzero(abl_b[:, 2]) == 0
    assert torch.count_nonzero(abl_u[2]) == 0
    for c in (0, 1, 3):
        assert torch.equal(abl_b[:, c], base_b[:, c])
        assert torch.equal(abl_u[c], base_u[c])
    assert not torch.equal(abl_logits, base_logits)

    # Backward + inplace ReLU after the ablated conv channel must still work.
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert model.conv1.weight.grad is not None
    assert torch.count_nonzero(model.conv1.weight.grad[2]) == 0

    off()
    assert _hooks(model) == 0
    with torch.no_grad():
        assert torch.equal(model(x), base_logits)


def test_conv_channel_controls_and_wire_round_trip():
    model, (x, y) = _conv_model(), _conv_batch()
    ctl = C.Controls()
    w = pytest.importorskip("caliper._wire")
    if not hasattr(w, "OP_INTERVENE"):
        pytest.skip("this codec has no CONTROL op intervene")
    t = _Transport()
    route = ctl.attach(t)
    ctl.interventions_on(model)

    def step_once():
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        ctl.step()
        opt.step()

    with torch.no_grad():
        base = model(x)
    step_once()

    # Non-ablate op on a channel point is rejected cleanly by InterventionExecutor.
    ex = _ablate.InterventionExecutor(model)
    with pytest.raises(_ablate.InterventionError, match="ablate only"):
        ex.validate({"point": "conv1#1", "op": "add", "vector": [1.0], "coefficient": 1.0})

    reply = route.on_control("c", {
        "request_id": 11,
        "op": "intervene",
        "intervene": {"set": {"point": "conv1#1", "op": "ablate"}},
    })
    w.ControlReply.from_frame(w.Frame(w.MsgType.CONTROL_REPLY, reply))
    assert reply["status"] == "ok" and reply["op"] == "intervene"

    step_once()
    assert _hooks(model) == 1
    with torch.no_grad():
        ablated = model(x)
    assert not torch.equal(ablated, base)

    controls = [m for ty, m in t.out if ty == w.MsgType.CONTROLS]
    last = w.Controls.from_frame(w.Frame(w.MsgType.CONTROLS, controls[-1])).to_meta()
    iv = last["interventions"]
    assert iv["active"] == [{
        "point": "conv1#1",
        "op": "ablate",
        "from_step": None,
        "until_step": None,
        "since_step": 3,
        "state": "active",
    }]
    pts = {p["point"]: p for p in iv["points"]}
    assert pts["conv1#1"]["kind"] == "module" and pts["conv1#1"]["module"] == "conv1"

    route.on_control("c", {"request_id": 12, "op": "intervene", "intervene": {"clear": "conv1#1"}})
    step_once()
    assert _hooks(model) == 0
    with torch.no_grad():
        assert torch.equal(model(x), base)
    route.detach()
