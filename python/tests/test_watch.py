"""caliper.watch(model): hooks are registered, produce views, and come off."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper
from conftest import live_views_or_skip, mps_available

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


@pytest.fixture(scope="module", autouse=True)
def canvas():
    caliper.offscreen(256, 256)
    yield
    caliper.close()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.fc1 = nn.Linear(6 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def _step(net, device):
    x = torch.randn(4, 1, 16, 16, device=device)
    loss = net(x).square().mean()
    loss.backward()


def _hook_count(net):
    return sum(len(m._forward_hooks) for m in net.modules())


@pytest.mark.parametrize("device", DEVICES)
def test_watch_registers_produces_and_removes(device):
    live_views_or_skip()
    torch.manual_seed(0)
    net = Net().to(device)
    h = caliper.watch(net, activations=True, gradients=True)
    try:
        assert h.modules == ["conv1", "fc1", "fc2"]
        assert set(h.views) == {"conv1.act", "fc1.act", "fc2.act",
                                "conv1.weight.grad", "fc1.weight.grad", "fc2.weight.grad"}
        assert _hook_count(net) == 3 and len(h) == 6
        _step(net, device)
        conv = caliper.view_info("conv1.act")              # (4,6,16,16) -> item 0 mosaic 3x2
        assert (conv["width"], conv["height"]) == (48, 32) and conv["updates"] == 1
        assert (caliper.view_info("fc1.act")["width"], caliper.view_info("fc1.act")["height"]) == (16, 4)
        g = caliper.view_info("conv1.weight.grad")         # (6,1,3,3) -> (6, 9)
        assert (g["width"], g["height"]) == (9, 6)
        assert caliper.view_info("fc1.weight.grad")["imported"] is True
    finally:
        h.remove()
    assert _hook_count(net) == 0 and not h.active
    before = caliper.view_info("conv1.act")["updates"]
    _step(net, device)
    assert caliper.view_info("conv1.act")["updates"] == before
    for v in h.views:
        caliper.remove(v)


def test_watch_include_and_every():
    live_views_or_skip()
    net = Net()
    with caliper.watch(net, include=["conv*"], every=2) as h:
        assert h.modules == ["conv1"] and h.views == ["conv1.act"]
        for _ in range(4):
            net(torch.randn(2, 1, 16, 16))
        assert caliper.view_info("conv1.act")["updates"] == 2     # every 2nd forward
    assert _hook_count(net) == 0
    pick = caliper.watch(net, include=lambda name, m: isinstance(m, nn.Linear), activations=True)
    assert pick.modules == ["fc1", "fc2"]
    pick.remove()
    caliper.remove("conv1.act")


def test_watch_is_free_when_nothing_is_attached():
    net = Net()
    caliper.close()                       # no transport at all: the hooks call a no-op
    try:
        import os
        os.environ["CALIPER_SESSION"] = "0"
        caliper._auto_tried = False
        with caliper.watch(net, gradients=True):
            _step(net, "cpu")
        assert caliper._native_core() is None
    finally:
        os.environ.pop("CALIPER_SESSION", None)
        caliper.offscreen(256, 256)
