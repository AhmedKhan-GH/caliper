"""Tensor -> snapshot preparation, done on the tensor's own device.

Every function here builds a :class:`Plan`: the payload layout of a view kind
plus a ``write(slot)`` that fills a ring slot with torch ops queued on the
tensor's own stream. Nothing here reads a value back: no ``.item()``,
``.cpu()``, ``.numpy()``, ``.tolist()`` and no synchronize, so a snapshot of an
MPS tensor is a handful of queued kernels and the training loop never waits for
the GPU. CPU tensors are written with a plain ``copy_`` into the slot, which is
CPU-readable shared memory (the one copy).

Layouts (the payload a slot holds):
  * heatmap    (H, W) float32, colormapped; autoscale = ``aminmax(out=stats[0:2])``
  * image      (H, W, 4) uint8 RGBA
  * points     (N, 3) float32 (2-D input padded z = 0), aux (N,) float32 labels,
               stats[0:2] = label range (continuous labels), stats[2:8] = bbox
  * series     (M,) float32 append-only ring (see SeriesPlan)
  * histogram  (bins,) float32 counts over [stats[0], stats[1]]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from ._ring import (FLAG_AUTO_VMAX, FLAG_AUTO_VMIN, FLAG_RAW, KIND_HEATMAP, KIND_HISTOGRAM,
                    KIND_IMAGE, KIND_POINTS, KIND_SERIES, STATS_FLOATS)

CMAPS = {"viridis": 0, "magma": 1, "rdbu": 2}
SERIES_CAPACITY = 4096


def cmap_index(cmap) -> int:
    if isinstance(cmap, int):
        if cmap not in CMAPS.values():
            raise ValueError(f"unknown colormap {cmap!r}; expected one of {sorted(CMAPS)}")
        return cmap
    try:
        return CMAPS[str(cmap).lower()]
    except KeyError:
        raise ValueError(f"unknown colormap {cmap!r}; expected one of {sorted(CMAPS)}") from None


def check_range(vmin, vmax) -> None:
    for v in (vmin, vmax):
        if v is not None and not math.isfinite(float(v)):
            raise ValueError(f"vmin/vmax must be finite, got {v!r}")


@dataclass(slots=True)
class Plan:
    """How one snapshot is written into a ring slot.

    ``key`` is the view signature: a change re-creates the view's slots
    (``ViewSpec`` is fixed for a view's lifetime)."""
    kind: int
    shape: tuple                    # payload shape
    dtype: Any                      # torch dtype of the payload
    device: Any                     # torch.device the producer writes from
    key: tuple
    write: Callable[[Any], None]    # write(slot): queued device ops into the slot
    init: Callable[[Any], None] | None = None   # once per new slot (NaN pads, alpha = 255)
    aux_shape: tuple | None = None
    colormap: int = 0
    vmin: float = 0.0
    vmax: float = 1.0
    flags: int = 0
    points_dims: int = 0
    label_count: int = 0
    stats_count: int = 2


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def _as_tensor(t):
    import torch
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    t = t.detach()
    if t.device.type not in ("cpu", "mps"):
        # CUDA and other devices have no zero-copy path on macOS: one host copy.
        t = t.to("cpu")
    if t.is_complex():
        t = t.abs()
    return t


def _is_image_hwc(t) -> bool:
    import torch
    # (H, W, 3|4) with a non-channel-like leading dim, or any uint8 (H, W, 3|4).
    if t.shape[-1] not in (3, 4):
        return False
    return t.dtype == torch.uint8 or t.shape[0] not in (1, 3, 4)


def _image(x):
    """(H, W, 3|4) -> contiguous (H, W, 4) uint8 on x's device (kept for
    caliper.live, the session prototype)."""
    import torch
    if x.dtype == torch.uint8:
        u8 = x
    elif x.is_floating_point():
        u8 = (x.float().clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    else:
        u8 = x.clamp(0, 255).to(torch.uint8)
    if u8.shape[-1] == 3:
        alpha = torch.full((*u8.shape[:-1], 1), 255, dtype=torch.uint8, device=u8.device)
        u8 = torch.cat([u8, alpha], -1)
    return SimpleNamespace(tensor=u8.contiguous())   # the spike's Prepared shape


# ---------------------------------------------------------------------------
# heatmap / image (show)
# ---------------------------------------------------------------------------
def _range_flags(vmin, vmax):
    flags = 0
    if vmin is None:
        flags |= FLAG_AUTO_VMIN
    if vmax is None:
        flags |= FLAG_AUTO_VMAX
    lo = float(vmin) if vmin is not None else 0.0
    hi = float(vmax) if vmax is not None else 1.0
    return flags, lo, hi


def _heatmap_plan(x, cm, vmin, vmax, raw, stats_src=None, mosaic=None):
    import torch
    flags, lo, hi = _range_flags(vmin, vmax)
    if raw:
        flags |= FLAG_RAW
    auto = flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX)
    if mosaic is None:
        shape = tuple(x.shape)

        def write(slot):
            slot.payload.copy_(x)                    # casts to f32 in the same kernel
            if auto:
                torch.aminmax(slot.payload, out=(slot.lo, slot.hi))
        init = None
        mode = "plain"
    else:
        rows, cols, h, w = mosaic
        c = x.shape[0]
        full, rem = divmod(c, cols)
        shape = (rows * h, cols * w)

        def write(slot):
            grid = slot.payload.view(rows, h, cols, w)
            if full:
                grid[:full].copy_(x[:full * cols].reshape(full, cols, h, w).permute(0, 2, 1, 3))
            if rem:
                grid[full, :, :rem].copy_(x[full * cols:].permute(1, 0, 2))
            if auto:
                torch.aminmax(stats_src, out=(slot.lo, slot.hi))

        def init(slot):
            slot.payload.fill_(float("nan"))         # pad cells
        mode = ("mosaic", cols)
    return Plan(KIND_HEATMAP, shape, torch.float32, x.device,
                (KIND_HEATMAP, shape, x.device, mode), write, init,
                colormap=cm, vmin=lo, vmax=hi, flags=flags, stats_count=2)


def _image_plan(x, raw):
    """x: (H, W, 3|4) -> (H, W, 4) uint8 slot; floats clamped to [0, 1]."""
    import torch
    h, w, c = x.shape
    if x.dtype == torch.uint8:
        src = x
    elif x.is_floating_point():
        src = x.float().clamp(0.0, 1.0).mul(255.0).add(0.5)   # copy_ to u8 truncates
    else:
        src = x.clamp(0, 255)
    init = None
    if c == 3:
        def init(slot):
            slot.payload[..., 3].fill_(255)

        def write(slot):
            slot.payload[..., :3].copy_(src)
    else:
        def write(slot):
            slot.payload.copy_(src)
    shape = (h, w, 4)
    return Plan(KIND_IMAGE, shape, torch.uint8, x.device, (KIND_IMAGE, shape, x.device, c),
                write, init, flags=FLAG_RAW if raw else 0, stats_count=0)


def plan_show(t, cm: int, vmin=None, vmax=None, raw: bool = False) -> Plan:
    """Map any tensor/array onto a heatmap or an RGBA image, on its own device.

    2-D -> heatmap; (H,W,3|4) and (C,H,W) with C in (3, 4) -> RGBA; (1,H,W) ->
    heatmap; other (C,H,W) -> channel mosaic (ceil(sqrt(C)) columns, NaN pads,
    range from the channels); (N,C,H,W) -> item 0; more dims stacked as channels;
    0-/1-D -> a 1xN strip."""
    import torch
    t = _as_tensor(t)
    if t.dtype == torch.bool:
        t = t.to(torch.uint8) if t.ndim == 3 and _is_image_hwc(t) else t.float()
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.reshape(1, -1)
    elif t.ndim == 4:
        t = t[0]
    elif t.ndim > 4:
        t = t.reshape(-1, *t.shape[-2:])
    if t.ndim == 2:
        return _heatmap_plan(t, cm, vmin, vmax, raw)
    if _is_image_hwc(t):
        return _image_plan(t, raw)
    c, h, w = t.shape
    if c == 1:
        return _heatmap_plan(t[0], cm, vmin, vmax, raw)
    if c in (3, 4):
        return _image_plan(t.permute(1, 2, 0), raw)
    x = t if t.dtype == torch.float32 else t.float()
    cols = math.ceil(math.sqrt(c))
    rows = math.ceil(c / cols)
    return _heatmap_plan(x, cm, vmin, vmax, raw, stats_src=x, mosaic=(rows, cols, h, w))


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------
def plan_points(z, labels=None, classes=None, cm: int = 0, vmin=None, vmax=None) -> Plan:
    """(N, 2|3) positions -> (N, 3) float32 payload (z = 0 for 2-D input);
    labels (N,) -> aux float32 (cast on the device). ``classes`` = number of
    categorical labels 0..classes-1 (categorical palette); without it labels
    are colormapped over their range (stats[0:2], autoscaled unless vmin/vmax).
    stats[2:8] = per-axis (min, max), one aminmax on the device."""
    import torch
    z = _as_tensor(z)
    if z.ndim != 2 or z.shape[1] not in (2, 3):
        raise ValueError(f"points: expected an (N, 2) or (N, 3) tensor, got shape {tuple(z.shape)}")
    if z.dtype != torch.float32:
        z = z.float()
    n, d = z.shape
    if n == 0:
        raise ValueError("points: empty point set")
    lab = None
    if labels is not None:
        lab = labels.detach() if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        lab = lab.reshape(-1)
        if lab.numel() != n:
            raise ValueError(f"points: {lab.numel()} labels for {n} points")
        if lab.device != z.device:
            if lab.device.type != "cpu":
                raise ValueError(f"points: labels on {lab.device} but points on {z.device}")
            lab = lab.to(z.device)     # host -> device upload, never device -> host
    if classes is not None:
        classes = int(classes)
        if classes < 1:
            raise ValueError("points: classes must be >= 1")
        if lab is None:
            raise ValueError("points: classes= needs labels=")
    label_count = classes or 0
    flags, lo, hi = (0, 0.0, 1.0)
    if lab is not None and not label_count:
        flags, lo, hi = _range_flags(vmin, vmax)
    auto = flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX)

    def init(slot):
        slot.payload.zero_()
        slot.stats.zero_()

    def write(slot):
        if d == 3:
            slot.payload.copy_(z)
        else:
            slot.payload[:, :2].copy_(z)
        torch.aminmax(z, dim=0, out=(slot.axis_min[:d], slot.axis_max[:d]))
        if lab is not None:
            slot.aux.copy_(lab)
            if auto:
                torch.aminmax(slot.aux, out=(slot.lo, slot.hi))

    shape = (n, 3)
    return Plan(KIND_POINTS, shape, torch.float32, z.device,
                (KIND_POINTS, shape, z.device, d, lab is not None), write, init,
                aux_shape=(n,) if lab is not None else None, colormap=cm, vmin=lo, vmax=hi,
                flags=flags, points_dims=d, label_count=label_count, stats_count=STATS_FLOATS)


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------
_ONES: dict = {}


def plan_histogram(t, bins: int = 64) -> Plan:
    """(bins,) float32 counts of every element of ``t`` over [min, max]
    (constant data: [v - 1, v + 1], as torch.histc). The range is written to
    stats[0:2] by aminmax on the device.

    MPS: binned on the device (floor((x - lo) * bins / (hi - lo)) clamped to the
    last bin, then index_add_), because torch.histc on MPS reads its range back
    to the host (a full drain of the queue, measured 15-53 ms). The bin rule is
    histc's; NaN values fall into bin 0. CPU: torch.histc."""
    import torch
    bins = int(bins)
    if not 1 <= bins <= 1 << 16:
        raise ValueError("histogram: bins must be in [1, 65536]")
    x = _as_tensor(t)
    if x.dtype != torch.float32:
        x = x.float()
    x = x.reshape(-1)
    if x.numel() == 0:
        raise ValueError("histogram: empty tensor")
    dev = x.device
    n = x.numel()

    def write(slot):
        lo, hi = slot.lo, slot.hi
        torch.aminmax(x, out=(lo, hi))
        pad = (hi == lo).to(torch.float32)
        lo.sub_(pad)
        hi.add_(pad)
        if dev.type == "cpu":
            try:
                torch.histc(x, bins, 0, 0, out=slot.payload)
                return
            except RuntimeError:       # non-finite range: fall through to the binning rule
                pass
        one = _ONES.get(dev)
        if one is None:
            one = _ONES[dev] = torch.ones(1, dtype=torch.float32, device=dev)
        idx = (x - lo).mul_(bins / (hi - lo)).nan_to_num_(0.0, bins - 1, 0.0)
        idx = idx.clamp_(0, bins - 1).to(torch.int64)
        slot.payload.zero_()
        slot.payload.index_add_(0, idx, one.expand(n))

    shape = (bins,)
    return Plan(KIND_HISTOGRAM, shape, torch.float32, dev, (KIND_HISTOGRAM, shape, dev), write,
                flags=FLAG_AUTO_VMIN | FLAG_AUTO_VMAX, stats_count=2)


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SeriesPlan:
    """Values to append to a series view. ``value`` is a Python float or a
    1-D float32 tensor of ``count`` values on ``device``."""
    value: Any
    count: int
    device: Any

    def append(self, payload, start: int) -> None:
        """Write the values at [start, start + count) mod capacity, on the device."""
        m = payload.shape[0]
        v = self.value
        if self.count == 1:
            if isinstance(v, float):
                payload[start] = v               # CPU store, or a fill kernel on MPS
            else:
                payload[start:start + 1].copy_(v)
            return
        first = min(self.count, m - start)
        payload[start:start + first].copy_(v[:first])
        if first < self.count:
            payload[:self.count - first].copy_(v[first:])


def plan_series(value) -> SeriesPlan:
    import torch
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return SeriesPlan(float(value), 1, torch.device("cpu"))
    v = _as_tensor(value)
    v = v.reshape(-1)
    if v.numel() == 0:
        raise ValueError("series: empty value")
    if v.numel() > SERIES_CAPACITY:
        v = v[-SERIES_CAPACITY:]
    if v.dtype != torch.float32:
        v = v.float()
    return SeriesPlan(v, v.numel(), v.device)


SERIES_KIND = KIND_SERIES
