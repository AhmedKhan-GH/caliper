"""Shared fixtures. One CaliperCore per process: a module fixture (re)creates
the offscreen core at the size that module needs; views are removed after every
test so tests do not see each other's state."""
import numpy as np
import pytest
import torch

import caliper


def mps_available() -> bool:
    import torch
    return torch.backends.mps.is_available()


requires_mps = pytest.mark.skipif(not mps_available(), reason="no MPS device")


@pytest.fixture(scope="module")
def offscreen256():
    caliper.offscreen(256, 256)
    yield
    caliper.close()


@pytest.fixture(autouse=True)
def _remove_views(monkeypatch):
    """Record every view a test creates and remove it afterwards."""
    names = set()

    def recording(fn, default):
        def wrapped(t, *a, **kw):
            names.add(kw["name"] if "name" in kw else (a[0] if a and isinstance(a[0], str) else default))
            return fn(t, *a, **kw)
        return wrapped

    monkeypatch.setattr(caliper, "show", recording(caliper.show, "tensor"))
    monkeypatch.setattr(caliper, "series", recording(caliper.series, "series"))
    monkeypatch.setattr(caliper, "histogram", recording(caliper.histogram, "histogram"))

    real_points = caliper.points

    def recording_points(z, labels=None, name="points", *a, **kw):
        names.add(name)
        return real_points(z, labels, name, *a, **kw)
    monkeypatch.setattr(caliper, "points", recording_points)
    yield
    for n in names:
        caliper.remove(n)


def live_views_or_skip():
    """Skip unless the core's typed live views (caliper_core_live_apply for
    points/series/histogram) are implemented in this build."""
    try:
        caliper.histogram(torch.zeros(4), name="__probe__", bins=2)
    except RuntimeError as e:
        if "not implemented" in str(e):
            pytest.skip(f"typed live views not in this libcaliper yet: {e}")
        raise
    finally:
        caliper.remove("__probe__")


def unpack_lut(lut: np.ndarray) -> np.ndarray:
    """uint32[256] (R | G<<8 | B<<16 | A<<24) -> uint8[256, 4]."""
    lut = lut.astype(np.uint32)
    return np.stack([(lut >> s) & 0xFF for s in (0, 8, 16, 24)], -1).astype(np.uint8)


def reference_heatmap(v: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """CPU reference, the exact rule of src/host/tensor_bridge.cpp map_f32_to_rgba8:
    t = clamp((v - vmin) / (vmax - vmin), 0, 1) in f32; idx = (int)(t*255 + 0.5);
    NaN -> 0; dst = lut[idx] unpacked little-endian."""
    v = np.asarray(v, dtype=np.float32)
    lo, hi = np.float32(vmin), np.float32(vmax)
    denom = np.float32(hi - lo)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (v - lo) / denom if denom != 0 else np.zeros_like(v)
    t = np.clip(t, np.float32(0), np.float32(1)).astype(np.float32)
    nan = np.isnan(t)
    t = np.where(nan, np.float32(0), t)
    idx = (t * np.float32(255.0) + np.float32(0.5)).astype(np.int32)
    idx = np.where(nan, 0, np.clip(idx, 0, 255))
    return unpack_lut(caliper.colormap_lut(cmap))[idx]
