"""Every view kind through the synchronous offscreen core (caliper.offscreen):
the window transport's shared-MTLBuffer slots, caliper_core_live_apply, and the
core's readbacks."""
import numpy as np
import pytest
import torch

import caliper
from caliper._convert import SERIES_CAPACITY
from conftest import live_views_or_skip, mps_available, reference_heatmap

DEVICES = ["cpu"] + (["mps"] if mps_available() else [])
CW, CH = 320, 240


@pytest.fixture(scope="module", autouse=True)
def canvas():
    caliper.offscreen(CW, CH)
    yield
    caliper.close()


@pytest.fixture(autouse=True)
def _need_live_views():
    live_views_or_skip()


@pytest.mark.parametrize("device", DEVICES)
def test_heatmap_byte_exact_with_storage_offset(device):
    g = torch.Generator().manual_seed(3)
    big = torch.rand(100, 64, generator=g).to(device)
    s = big[10:58]
    assert s.storage_offset() == 640
    st = caliper.show(s, name="hm_off", cmap="magma", vmin=0.1, vmax=0.9)
    assert st.path == "compute-imported" and st.imported      # CPU too: the slot is shared
    ref = reference_heatmap(s.cpu().numpy(), "magma", 0.1, 0.9)
    assert np.array_equal(caliper.view_readback("hm_off"), ref)
    # autoscaled: the range is the GPU aminmax the core read from the slot's stats
    caliper.show(s, name="hm_auto")
    lo, hi = torch.aminmax(s)
    ref = reference_heatmap(s.cpu().numpy(), "viridis", lo.item(), hi.item())
    assert np.array_equal(caliper.view_readback("hm_auto"), ref)
    info = caliper.live_view_info("hm_auto")
    assert info["kind"] == 1 and info["imported"] and info["updates"] == 1


@pytest.mark.parametrize("device", DEVICES)
def test_rgba_byte_exact_with_storage_offset(device):
    g = torch.Generator().manual_seed(4)
    big = torch.randint(0, 256, (40, 24, 4), dtype=torch.uint8, generator=g).to(device)
    s = big[7:33]
    assert s.storage_offset() > 0
    st = caliper.show(s, name="img_off")
    assert st.path.startswith("blit") and st.imported
    assert np.array_equal(caliper.view_readback("img_off"), s.cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_points_texture_shows_label_colours(device):
    torch.manual_seed(0)
    k = 4
    y = torch.arange(800) % k
    centres = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    z = centres[y] + 0.05 * torch.randn(800, 2)
    st = caliper.points(z.to(device), labels=y.to(device), name="emb", classes=k)
    assert st.path.startswith("points") and st.kind == "points"
    caliper.frame()
    caliper.frame()
    tex = caliper.live_view_readback("emb")
    assert tex.ndim == 3 and tex.shape[2] == 4 and tex.size > 0
    colours, counts = np.unique(tex.reshape(-1, 4), axis=0, return_counts=True)
    bg = colours[counts.argmax()]
    fg = [c for c, n in zip(colours, counts) if not np.array_equal(c, bg) and n >= 10]
    assert len(fg) >= k, (len(fg), colours)            # one colour per class at least
    # each quadrant's dominant non-background colour differs: labels decide colour
    h, w = tex.shape[:2]
    quads = []
    for qy in (slice(0, h // 2), slice(h // 2, h)):
        for qx in (slice(0, w // 2), slice(w // 2, w)):
            q = tex[qy, qx].reshape(-1, 4)
            q = q[~(q == bg).all(1)]
            if len(q):
                c, n = np.unique(q, axis=0, return_counts=True)
                quads.append(tuple(c[n.argmax()]))
    assert len(quads) == 4 and len(set(quads)) == 4, quads
    info = caliper.live_view_info("emb")
    assert info["count"] == 800 and info["kind"] == 3
    assert info["imported"] is True                    # CPU too: shared-MTLBuffer slot


@pytest.mark.parametrize("device", DEVICES)
def test_series_values_round_trip_with_wrap(device):
    n = SERIES_CAPACITY + 1000
    if device == "mps":
        vals = torch.arange(n, dtype=torch.float32, device="mps")
        for i in range(n):
            caliper.series(vals[i], "curve")
    else:
        for i in range(n):
            caliper.series(float(i), "curve")
    got = caliper.live_view_values("curve")
    assert np.array_equal(got, np.arange(n - SERIES_CAPACITY, n, dtype=np.float32))
    info = caliper.live_view_info("curve")
    assert info["count"] == SERIES_CAPACITY and info["path"] == "cpu-read"
    # 1-D tensors append several values at once (wrapping)
    caliper.series(torch.arange(3000, dtype=torch.float32, device=device) + n, "curve")
    got = caliper.live_view_values("curve")
    assert np.array_equal(got, np.arange(n + 3000 - SERIES_CAPACITY, n + 3000, dtype=np.float32))


@pytest.mark.parametrize("device", DEVICES)
def test_histogram_counts_round_trip(device):
    g = torch.Generator().manual_seed(9)
    x = torch.randn(512, 333, generator=g).to(device)
    st = caliper.histogram(x, "dist", bins=40)
    assert st.kind == "histogram"
    ref = torch.histc(x.cpu(), 40, x.min().item(), x.max().item()).numpy()
    got = caliper.live_view_values("dist")
    assert np.array_equal(got, ref) and got.sum() == x.numel()
    assert caliper.live_view_info("dist")["count"] == 40


def test_render_and_pause():
    caliper.show(torch.linspace(0, 1, 64).repeat(32, 1), name="rp")
    img = caliper.render()
    assert img.shape == (CH, CW, 4) and img.dtype == np.uint8
    assert img[..., :3].max() > 100 and len(np.unique(img.reshape(-1, 4), axis=0)) > 16
    for i in range(3):
        caliper.series(float(i), "ps")
    caliper.live_pause(True)
    try:
        assert caliper.live_paused()
        caliper.series(99.0, "ps")
        info = caliper.live_view_info("ps")
        assert info["dropped"] >= 1
        assert list(caliper.live_view_values("ps")) == [0.0, 1.0, 2.0]
    finally:
        caliper.live_pause(False)
    caliper.series(5.0, "ps")
    assert list(caliper.live_view_values("ps")) == [0.0, 1.0, 2.0, 99.0, 5.0]


def test_kind_change_and_remove():
    caliper.show(torch.rand(8, 8), name="kc")
    caliper.histogram(torch.rand(100), name="kc", bins=4)
    assert caliper.live_view_info("kc")["kind"] == 5
    assert caliper.remove("kc")
    with pytest.raises(RuntimeError):
        caliper.live_view_info("kc")
