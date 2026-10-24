import numpy as np
import pytest
import torch

import caliper
from conftest import mps_available

pytestmark = pytest.mark.usefixtures("offscreen256")
H, W = 32, 48
DEVICES = ["cpu"] + (["mps"] if mps_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_rgba_u8_roundtrip(device):
    g = torch.Generator().manual_seed(1)
    img = torch.randint(0, 256, (H, W, 4), dtype=torch.uint8, generator=g)
    st = caliper.show(img.to(device), name="rgba")
    info = caliper.view_info("rgba")
    # Changed: a CPU image is copied once into a shared MTLBuffer slot that the core
    # blits in place, so CPU reports "blit-imported" like MPS (was "cpu-staged").
    assert info["path"].startswith("blit") and info["imported"] is True, info
    assert st.path == info["path"]
    assert (info["width"], info["height"]) == (W, H)
    assert np.array_equal(caliper.view_readback("rgba"), img.numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_rgb_and_chw_float_images(device):
    g = torch.Generator().manual_seed(2)
    rgb = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, generator=g)
    caliper.show(rgb.to(device), name="rgb")
    tex = caliper.view_readback("rgb")
    assert np.array_equal(tex[..., :3], rgb.numpy()) and (tex[..., 3] == 255).all()

    chw = torch.rand(3, H, W, generator=g)
    caliper.show(chw.to(device), name="chw")
    tex = caliper.view_readback("chw")
    expect = (chw.permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
    assert np.array_equal(tex[..., :3], expect) and (tex[..., 3] == 255).all()
