import numpy as np
import pytest
import torch

import caliper
from conftest import reference_heatmap, requires_mps

pytestmark = [requires_mps, pytest.mark.usefixtures("offscreen256")]


def test_show_mps_ramp_imported():
    w, h = 64, 48
    ramp = (torch.arange(w, dtype=torch.float32, device="mps") / (w - 1)).repeat(h, 1)
    st = caliper.show(ramp, name="mps_ramp", vmin=0, vmax=1)
    info = caliper.view_info("mps_ramp")
    assert info["path"].startswith("compute"), info
    assert info["imported"] is True and st.imported
    assert (info["width"], info["height"]) == (w, h)
    assert info["device"] == "metal"
    ref = reference_heatmap(ramp.cpu().numpy(), "viridis", 0, 1)
    assert np.array_equal(caliper.view_readback("mps_ramp"), ref)


def test_row_slice_byte_offset_imported():
    big = torch.rand(128, 64, device="mps")
    s = big[16:64]
    assert s.is_contiguous() and s.storage_offset() == 16 * 64
    cap_off = s.__dlpack__()  # sanity: DLPack carries the slice as a byte_offset
    del cap_off
    caliper.show(s, name="slice", vmin=0, vmax=1)
    info = caliper.view_info("slice")
    assert info["imported"] is True, info
    assert (info["width"], info["height"]) == (64, 48)
    ref = reference_heatmap(s.cpu().numpy(), "viridis", 0, 1)
    assert np.array_equal(caliper.view_readback("slice"), ref)


def test_non_contiguous_mps_slice_still_works():
    big = torch.rand(64, 64, device="mps")
    s = big[8:, 8:]          # non-contiguous: show() makes it contiguous on-device
    assert not s.is_contiguous()
    caliper.show(s, name="nc", vmin=0, vmax=1)
    info = caliper.view_info("nc")
    assert (info["width"], info["height"]) == (56, 56) and info["imported"] is True
    ref = reference_heatmap(s.cpu().numpy(), "viridis", 0, 1)
    assert np.array_equal(caliper.view_readback("nc"), ref)


def test_mps_autoscale_from_gpu_stats():
    # Changed: the wait_gpu= option is gone (no show waits for the GPU any more);
    # the autoscale range now comes from aminmax(out=) into the slot's stats.
    t = torch.randn(32, 40, device="mps")
    st = caliper.show(t, name="nowait")
    assert st.imported
    caliper.frame()
    lo, hi = torch.aminmax(t)
    ref = reference_heatmap(t.cpu().numpy(), "viridis", lo.item(), hi.item())
    assert np.array_equal(caliper.view_readback("nowait"), ref)


def test_mps_repeated_show_no_device_growth():
    t = torch.rand(256, 256, device="mps")
    caliper.show(t, name="grow", vmin=0, vmax=1)
    torch.mps.synchronize()
    before = torch.mps.current_allocated_memory()
    for _ in range(20):
        caliper.show(t, name="grow", vmin=0, vmax=1)
    torch.mps.synchronize()
    assert torch.mps.current_allocated_memory() == before
