"""The framework adapter (caliper._adapter): the state of a model and its optimizer saved and
restored, so that a restored run continues exactly as the original, the loss tap that it owns,
the export, sync point and hooks of the producer runtime, and the probe that is not available yet."""
from __future__ import annotations

import io

import pytest
import torch

from caliper._adapter import TORCH, TorchAdapter


def _make(seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
    return net, torch.optim.Adam(net.parameters(), lr=0.01)


def _steps(net, opt, x, y, n):
    out = []
    for _ in range(n):
        loss = (net(x) - y).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append(loss.detach().clone())
    return torch.stack(out)


def test_a_restored_run_continues_as_the_original():
    """The weights and the Adam moments come back, so the next steps give the same losses."""
    x, y = torch.randn(32, 6), torch.randn(32, 1)
    net, opt = _make()
    _steps(net, opt, x, y, 5)
    data = TORCH.state(net, opt, {"step": 5, "params": {"lr": 0.01}})
    expected = _steps(net, opt, x, y, 3)

    net2, opt2 = _make(seed=1)                       # other weights before the restore
    extra = TORCH.restore(data, net2, opt2)
    assert extra == {"step": 5, "params": {"lr": 0.01}}
    assert torch.equal(_steps(net2, opt2, x, y, 3), expected)


def test_the_state_is_a_plain_torch_file():
    """torch.load(..., weights_only=True) reads it, and its model entry is a state_dict."""
    net, opt = _make()
    got = torch.load(io.BytesIO(TORCH.state(net, opt, {"step": 1})), weights_only=True)
    assert set(got) == {"model", "optimizer", "step"}
    net2, _ = _make(seed=2)
    net2.load_state_dict(got["model"])
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), net2.parameters()))


def test_without_an_optimizer():
    """An inference state has no optimizer, and its restore leaves a given optimizer as it is."""
    net, opt = _make()
    data = TORCH.state(net)
    assert torch.load(io.BytesIO(data), weights_only=True)["optimizer"] is None
    net2, opt2 = _make(seed=3)
    before = opt2.state_dict()
    assert TORCH.restore(data, net2, opt2) == {}
    assert opt2.state_dict() == before


def test_refusals():
    net, opt = _make()
    with pytest.raises(ValueError):
        TORCH.state(net, opt, {"model": 1})
    buf = io.BytesIO()
    torch.save({"weights": 1}, buf)
    with pytest.raises(ValueError):
        TORCH.restore(buf.getvalue(), net, opt)
    other = torch.nn.Linear(3, 3)
    with pytest.raises(RuntimeError):
        TORCH.restore(TORCH.state(other), net)


def test_the_loss_tap_is_counted_and_comes_off():
    a = TorchAdapter()
    orig = torch.autograd.backward
    a.tap_on()
    a.tap_on()
    w = torch.ones(3, requires_grad=True)
    (w * 2).sum().backward()
    assert a.loss[0] == 1 and a.loss[1].item() == 6.0
    a.tap_off()
    assert torch.autograd.backward is not orig       # one user is left
    a.tap_off()
    assert torch.autograd.backward is orig and a.loss[1] is None


def test_probe_is_not_available():
    with pytest.raises(NotImplementedError):
        TorchAdapter().probe(None, None)


def test_export_gives_the_storage_without_a_copy_and_refuses_a_strided_view():
    a = TorchAdapter()
    w = torch.nn.Linear(4, 3).weight                 # requires grad: export detaches, no copy
    back = torch.from_dlpack(a.export(w))
    assert back.data_ptr() == w.data_ptr() and back.shape == w.shape
    with pytest.raises(ValueError, match="not contiguous"):
        a.export(w.t())


def test_framework_done_returns_after_the_queued_work():
    a = TorchAdapter()
    a.framework_done()                               # CPU: returns at once
    if torch.backends.mps.is_available():
        x = torch.ones(256, 256, device="mps")
        y = x @ x
        a.framework_done()
        a.framework_done()                           # one event, recorded again
        assert float(y[0, 0].cpu()) == 256.0


def test_hooks_call_back_and_come_off():
    net, opt = _make()
    a = TorchAdapter()
    outs, grads, steps, writes = [], [], [], []
    with pytest.raises(RuntimeError, match="optimizer"):
        a.before_write(lambda: writes.append(1))
    a.hooks(net, opt, lambda n, o: outs.append(n), lambda n, g: grads.append((n, g.shape)),
            lambda: steps.append(1))
    a.before_write(lambda: writes.append(1))
    x, y = torch.randn(5, 6), torch.randn(5, 1)
    _steps(net, opt, x, y, 2)
    assert outs[:4] == ["0", "1", "2", ""] and len(outs) == 8
    assert ("0.weight", torch.Size([12, 6])) in grads and len(grads) == 8
    assert steps == [1, 1] and writes == [1, 1]
    a.unhook(net)
    a.unhook(opt)
    outs.clear(), grads.clear(), steps.clear(), writes.clear()
    _steps(net, opt, x, y, 1)
    assert outs == grads == steps == writes == []


def test_hooks_without_callbacks_add_nothing():
    net, opt = _make()
    a = TorchAdapter()
    a.hooks(net, None, None, None, None)
    assert all(not m._forward_hooks for m in net.modules())
    a.unhook(net)
