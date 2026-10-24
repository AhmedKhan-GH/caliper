import threading

import numpy as np
import pytest
import torch

import caliper

pytestmark = pytest.mark.usefixtures("offscreen256")


def test_show_cpu_ramp():
    h, w = 64, 96
    ramp = torch.linspace(0, 1, w).repeat(h, 1)
    st = caliper.show(ramp, name="cpu_ramp")
    # Changed: the CPU tensor is copied once into a shared MTLBuffer slot that the
    # core colormaps in place: "compute-imported", imported (was "cpu-staged").
    assert st and st.path == "compute-imported" and st.imported is True
    info = caliper.view_info("cpu_ramp")
    assert info["path"] == "compute-imported"
    assert (info["width"], info["height"]) == (w, h)
    assert info["imported"] is True
    caliper.frame()
    caliper.frame()
    px = caliper.read_pixels()
    assert px.shape == (256, 256, 4) and px.dtype == np.uint8
    # Non-blank: the view's colormapped texels exceed the dark clear color.
    assert px[..., :3].max() > 100
    assert len(np.unique(px.reshape(-1, 4), axis=0)) > 16


def test_update_in_place_counts():
    a = torch.zeros(8, 8)
    caliper.show(a, name="upd", vmin=0, vmax=1)
    u0 = caliper.view_info("upd")["updates"]
    caliper.show(a + 1, name="upd", vmin=0, vmax=1)
    assert caliper.view_info("upd")["updates"] == u0 + 1
    assert caliper.view_count() >= 1


def test_conversion_rules_cpu():
    # (C,H,W) C=5 -> mosaic: 3 columns x 2 rows of 10x12 tiles
    caliper.show(torch.randn(5, 10, 12), name="mosaic")
    info = caliper.view_info("mosaic")
    assert (info["width"], info["height"]) == (36, 20)
    # (N,C,H,W) with C=3 float -> first item as an RGBA image
    caliper.show(torch.rand(2, 3, 16, 20), name="nchw")
    info = caliper.view_info("nchw")
    assert (info["width"], info["height"]) == (20, 16)
    # (1,H,W) -> heatmap; int dtype -> .float(); constant tensor (vmin == vmax)
    caliper.show(torch.ones(1, 7, 9, dtype=torch.int32), name="const")
    assert (caliper.view_info("const")["width"], caliper.view_info("const")["height"]) == (9, 7)
    # 1-D -> a 1xN strip; numpy arrays accepted
    caliper.show(np.arange(10, dtype=np.float32), name="vec")
    assert caliper.view_info("vec")["height"] == 1


def test_show_from_a_thread_works():
    # Changed: show() from a non-main thread is allowed now (caliper.run() trains on
    # a worker thread); only window()/offscreen() need the main thread.
    errs, res = [], []

    def worker():
        try:
            res.append(caliper.show(torch.full((4, 4), 0.5), name="thread", vmin=0, vmax=1))
        except Exception as e:      # noqa: BLE001
            errs.append(e)

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert not errs and res and res[0]
    assert caliper.view_info("thread")["updates"] == 1
    errs.clear()

    def opener():
        try:
            caliper.offscreen(8, 8)
        except RuntimeError as e:
            errs.append(str(e))
    th = threading.Thread(target=opener)
    th.start()
    th.join()
    assert errs and "main thread" in errs[0]


def test_bad_dtype_layout_raises():
    with pytest.raises(ValueError):
        caliper.show(torch.zeros(4, 4), name="x", cmap="nope")


def test_autoscaled_same_shape_reuses_texture():
    # 50 autoscaled shows of a same-shape tensor: one texture creation, 50 updates.
    for i in range(50):
        caliper.show(torch.randn(24, 40) * (i + 1), name="reuse")
    info = caliper.view_info("reuse")
    assert info["creates"] == 1 and info["updates"] == 50
    caliper.show(torch.randn(24, 41), name="reuse")      # shape change -> new texture
    assert caliper.view_info("reuse")["creates"] == 2


def test_nan_inf_autoscale_and_bad_args():
    v = torch.randn(16, 16)
    v[0, 0], v[1, 1], v[2, 2] = float("nan"), float("inf"), -float("inf")
    caliper.show(v, name="naninf")                       # finite range picked from finite texels
    with pytest.raises(ValueError):
        caliper.show(v, name="naninf", vmin=0, vmax=float("inf"))
    with pytest.raises(ValueError):
        caliper.show(v, name="bad##name")


def test_core_error_surfaces_verbatim():
    # The core's last_error text, unwrapped.
    with pytest.raises(RuntimeError, match=r"^view_info: no view named 'missing'"):
        caliper.view_info("missing")
    core = caliper._native_core()
    with pytest.raises(RuntimeError, match=r"^view_tensor\('bad'\): ") as e:
        core.view_tensor("bad", torch.zeros(4, 4, dtype=torch.int32).__dlpack__())
    print("core refusal:", e.value)
