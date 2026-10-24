"""Pixel-exactness of the heatmap path against the CPU reference
(src/host/tensor_bridge.cpp map_f32_to_rgba8 + the core's own LUT)."""
import numpy as np
import pytest
import torch

import caliper
from conftest import mps_available, reference_heatmap

W, H = 64, 48
DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


@pytest.fixture(scope="module", autouse=True)
def canvas_64x48():
    caliper.offscreen(W, H)     # a canvas exactly the view's size; RAW draws 1:1 at (0,0)
    yield
    caliper.close()


@pytest.mark.parametrize("device", DEVICES)
def test_ramp_matches_cpu_reference(device):
    x = torch.arange(W, dtype=torch.float32) / (W - 1)
    ramp = x.repeat(H, 1)                                   # (48, 64), value x/(W-1)
    ref = reference_heatmap(ramp.numpy(), "viridis", 0.0, 1.0)

    st = caliper.show(ramp.to(device), name="ramp", cmap="viridis", vmin=0, vmax=1, _raw=True)
    if device == "mps":
        assert st.path.startswith("compute") and st.imported
    caliper.frame()
    caliper.frame()
    px = caliper.read_pixels()
    assert px.shape == (H, W, 4)

    diff = np.abs(px.astype(np.int16) - ref.astype(np.int16))
    print(f"[{device}] read_pixels vs CPU reference: max abs diff {diff.max()} "
          f"(rgb {diff[..., :3].max()}), mismatching texels {(diff.max(-1) > 0).sum()}")
    assert diff[..., :3].max() <= 1

    tex = caliper.view_readback("ramp")
    tdiff = np.abs(tex.astype(np.int16) - ref.astype(np.int16))
    print(f"[{device}] view_readback vs CPU reference: max abs diff {tdiff.max()}")
    assert np.array_equal(tex, ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cmap", ["magma", "rdbu"])
def test_other_colormaps_and_nan(device, cmap):
    torch.manual_seed(0)
    v = torch.randn(H, W) * 2
    v[3, 5] = float("nan")
    ref = reference_heatmap(v.numpy(), cmap, -1.5, 2.5)
    caliper.show(v.to(device), name="cm", cmap=cmap, vmin=-1.5, vmax=2.5, _raw=True)
    assert np.array_equal(caliper.view_readback("cm"), ref)
