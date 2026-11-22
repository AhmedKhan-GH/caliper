"""The framework adapter (caliper._adapter): the state of a model and its optimizer saved and
restored, so that a restored run continues exactly as the original, the loss tap that it owns,
and the functions that wait for M2 and M11."""
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


@pytest.mark.parametrize("call", [
    lambda a: a.export(torch.zeros(1)),
    lambda a: a.framework_done(),
    lambda a: a.before_write(lambda: None),
    lambda a: a.hooks(None, None, None, None, None),
    lambda a: a.probe(None, None),
])
def test_the_functions_of_m2_wait_for_it(call):
    with pytest.raises(NotImplementedError):
        call(TorchAdapter())
