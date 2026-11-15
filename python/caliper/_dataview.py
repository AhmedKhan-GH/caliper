"""Data views: which part of a tensor travels to a viewer.

The rules are those of the wire protocol (docs/references/wire-protocol.md,
"Data views"); this module applies them in the producer. A view offers one or
more named sources, each a tensor with named dims. A data view names a source,
indexes dims, reduces dims, can tile one dim of a heatmap, and sets a budget:

    {"source": "attention",
     "index":  {"layer": 3, "query": [0, 128], "key": [0, 512, 2]},
     "reduce": {"head": "mean"},
     "tile":   "batch",
     "budget": {"max_extent": 512, "pool": "mean", "max_points": 100000}}

An integer index selects one position and drops the dim (a negative one counts
from the end). A range ``[start, stop]`` or ``[start, stop, step]`` keeps it
(bounds may be null, are clamped, and the step is at least 1). A reduction
(``mean``, ``sum``, ``max``, ``min`` or ``norm``, the square root of the sum of
squares) removes the dim. The dims that remain must fit the live view: 2 for a
heatmap (3 with ``tile``), 2 for an image, 1 for points (the point dim), any
number for a histogram. The channels of an image source and the coordinates of
a points source are not dims: an image source is a tensor with one more axis
(RGB or RGBA, found by the rule of ``show``), a points source one whose last
axis holds 2 or 3 coordinates. A series takes no data view but its backfill.

Resolution (:func:`resolve`): members a request leaves out come from the view's
default data view; a member the request names replaces the default's member
as a whole, except ``budget``, whose keys are merged one by one. Unknown
members are ignored. The resolved data view has every default filled in, with
integer indices made non-negative and ranges written ``[start, stop, step]``
with clamped bounds. Nothing in a request is evaluated.

Application (:func:`apply`), on the tensor's device, in the protocol's order:
index, reduce one dim at a time in the source's dim order, lay out the tile
mosaic, then meet the budget. A heatmap or image axis of ``count`` positions
above ``max_extent`` is pooled to ``size = max_extent`` cells, cell i covering
positions [floor(i * count / size), ceil((i + 1) * count / size)), the rule of
torch's adaptive pooling (``budget.pool``: mean, max or min; images always
mean). Points above ``max_points`` are sampled with a stride: point i is
position floor(i * count / size), so the sample stays the same while the size
does not change, and the source row of each point is known on the host
(:func:`source_rows`). Pooling uses a reshape when ``count`` is a multiple of
``size``; otherwise segment sums (``index_add_``) for the mean and a gather of
the windows for max and min, with index tensors made on the device by arange
arithmetic and cached. It never uses torch's adaptive pooling, which MPS cannot
do for every size, and nothing reads a value back from the device.

The registry (:class:`Registry`, one per process: ``REGISTRY``) holds what
each view offers and the view's default data view. A data view set from Python
is the view's default: the in-process routes draw it and a viewer's
subscription starts from it (the protocol lets each subscription choose its
own; that is the network route's part). A request swaps a small immutable
state under a lock (any thread); all tensor work happens in the view call.

Importing this module does not import torch.
"""
from __future__ import annotations

import json
import math
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass

__all__ = [
    "DataViewError", "SourceSpec", "Offer", "Program", "Source", "Registry", "REGISTRY",
    "parse_request", "resolve", "layout", "compile_view", "apply", "take_rows", "source_rows",
    "DataCall", "DVWrite", "payload_plan", "finish", "lazy_plan", "net_plan", "dtype_name",
    "KINDS", "REDUCE_OPS", "POOL_OPS",
]

# The values of the wire protocol (caliper/live_net.h).
KINDS = {"heatmap": 1, "image": 2, "points": 3, "series": 4, "histogram": 5}
KIND_NAMES = {v: k for k, v in KINDS.items()}
REDUCE_OPS = ("mean", "sum", "max", "min", "norm")
POOL_OPS = ("mean", "max", "min")
DEFAULT_MAX_EXTENT = 512
LIMIT_MAX_EXTENT = 4096
DEFAULT_MAX_POINTS = 100_000
LIMIT_MAX_POINTS = 1_000_000
SERIES_RETAIN_MIN = 4096
NAME_MAX = 63                      # bytes of a view, source or dim name
MAX_SOURCES = 64
MAX_DIMS = 8
DTYPES = ("f32", "f16", "bf16", "f64", "i64", "i32", "i16", "i8", "u8", "bool")

REQUEST_BYTES_LIMIT = 64 * 1024    # JSON text of one request given to parse_request
PENDING_LIMIT = 64                 # requests kept for views that do not exist yet
_MEMBERS = ("source", "index", "reduce", "tile")


class DataViewError(ValueError):
    """A data view that does not resolve (the protocol's ``bad_data_view``).
    The message is plain text that names the rule, fit to send to a viewer."""


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _say(v) -> str:
    """A short rendering of a value from a request, for an error message."""
    if isinstance(v, str):
        return json.dumps(v if len(v) <= 40 else v[:37] + "...")
    if v is None or isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, int):
        return str(v) if -10 ** 18 < v < 10 ** 18 else "a very large integer"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else "a non-finite number"
    if isinstance(v, dict):
        return "an object"
    if isinstance(v, (list, tuple)):
        if len(v) <= 3 and all(x is None or _is_int(x) for x in v):
            return "[" + ", ".join("null" if x is None else _say(x) for x in v) + "]"
        return "a list"
    return "a value of type " + type(v).__name__


def _names(seq, limit: int = 8) -> str:
    seq = list(seq)
    s = ", ".join(_say(x) for x in seq[:limit])
    if len(seq) > limit:
        s += f" and {len(seq) - limit} more"
    return s or "none"


def _check_name(v, what: str) -> str:
    """A view, source or dim name given by Python code: 1 to 63 bytes."""
    if not isinstance(v, str) or not 0 < len(v.encode("utf-8", "surrogatepass")) <= NAME_MAX:
        raise ValueError(f"invalid {what} {v!r} (a str of 1 to {NAME_MAX} bytes)")
    return v


def parse_request(request) -> dict:
    """The members a data view request names (``source``, ``index``,
    ``reduce``, ``tile``, ``budget``), as given; unknown members, such as a
    newer protocol's, are ignored. ``request`` is a dict, its JSON text (str or
    bytes, at most REQUEST_BYTES_LIMIT) or None (the default). Raises
    DataViewError when it is not an object."""
    if request is None:
        return {}
    if isinstance(request, (bytes, bytearray, memoryview)):
        if len(request) > REQUEST_BYTES_LIMIT:
            raise DataViewError(f"a data view is limited to {REQUEST_BYTES_LIMIT} bytes")
        try:
            request = bytes(request).decode("utf-8")
        except UnicodeDecodeError:
            raise DataViewError("a data view must be UTF-8 JSON") from None
    if isinstance(request, str):
        if len(request) > REQUEST_BYTES_LIMIT:
            raise DataViewError(f"a data view is limited to {REQUEST_BYTES_LIMIT} bytes")
        try:
            request = json.loads(request)
        except (ValueError, RecursionError):
            raise DataViewError("a data view must be valid JSON") from None
    if not isinstance(request, dict):
        raise DataViewError(f"a data view is a JSON object, not {_say(request)}")
    out = {k: request[k] for k in _MEMBERS if k in request}
    budget = request.get("budget")
    if budget is not None:
        if not isinstance(budget, dict):
            raise DataViewError("budget must be an object")
        out["budget"] = budget
    return out


def _kind(kind) -> str:
    if _is_int(kind) and kind in KIND_NAMES:
        return KIND_NAMES[kind]
    if isinstance(kind, str) and kind in KINDS:
        return kind
    raise ValueError(f"unknown view kind {kind!r}; expected one of {tuple(KINDS)}")


# ---------------------------------------------------------------------------
# offers
# ---------------------------------------------------------------------------
_DTYPE_NAMES: dict | None = None


def dtype_name(dtype) -> str | None:
    """The protocol's name of a torch dtype ("f32", "bf16", "u8", ...), or None."""
    global _DTYPE_NAMES
    if _DTYPE_NAMES is None:
        import torch
        _DTYPE_NAMES = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
                        torch.float64: "f64", torch.int64: "i64", torch.int32: "i32",
                        torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8",
                        torch.bool: "bool"}
    return _DTYPE_NAMES.get(dtype)


@dataclass(frozen=True)
class SourceSpec:
    """One source of an offer: its dims, their sizes and its dtype (or None)."""
    name: str
    dims: tuple
    sizes: tuple
    dtype: str | None = None

    def to_json(self) -> dict:
        d = {"name": self.name,
             "dims": [{"name": n, "size": s} for n, s in zip(self.dims, self.sizes)]}
        if self.dtype is not None:
            d["dtype"] = self.dtype
        return d


@dataclass(frozen=True)
class Offer:
    """A view as a viewer sees it: its kind, its sources (those that have a
    tensor) and its default data view (None: ``{}``)."""
    kind: str = "heatmap"
    sources: tuple = ()
    default: dict | None = None
    name: str = ""

    def source(self, name) -> SourceSpec | None:
        for s in self.sources:
            if s.name == name:
                return s
        return None

    def to_json(self) -> dict:
        """The view object of a VIEWS message, without its ``id``."""
        d = {"name": self.name, "kind": KINDS[self.kind],
             "sources": [s.to_json() for s in self.sources]}
        if self.default is not None:
            d["default"] = self.default
        return d

    @classmethod
    def from_json(cls, view) -> "Offer":
        """An Offer from a view object as in a VIEWS message."""
        if not isinstance(view, dict):
            raise DataViewError("a view is a JSON object")
        try:
            kind = _kind(view.get("kind"))
            specs = tuple(SourceSpec(s["name"], tuple(d["name"] for d in s["dims"]),
                                     tuple(int(d["size"]) for d in s["dims"]), s.get("dtype"))
                          for s in view.get("sources") or ())
        except (KeyError, TypeError, ValueError) as e:
            raise DataViewError(f"malformed view object ({type(e).__name__})") from None
        default = view.get("default")
        return cls(kind, specs, default if isinstance(default, dict) else None,
                   str(view.get("name", "")))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def resolve(offer, request=None, kind=None, *, retain: int = SERIES_RETAIN_MIN) -> dict:
    """Resolve ``request`` (a dict, its JSON text, or None for ``{}``) against
    ``offer`` (an :class:`Offer` or a view object as in VIEWS): the request with
    every default filled in, ``{"source", "index", "reduce", "budget"}`` plus
    ``"tile"`` for a tiled heatmap (a series: ``{"budget": {"max_backfill"}}``,
    with ``retain`` the producer's retention). ``kind`` overrides the offer's.
    Raises DataViewError (``bad_data_view``) naming the broken rule."""
    off = offer if isinstance(offer, Offer) else Offer.from_json(offer)
    return _resolve(off, parse_request(request), off.kind if kind is None else _kind(kind),
                    retain)


def _budget_int(budget: dict, key: str, default: int, limit: int) -> int:
    v = budget.get(key)
    if v is None:
        return default
    if not _is_int(v) or v < 1:
        raise DataViewError(f"budget {key} must be a positive integer, not {_say(v)}")
    return min(v, limit)


def _resolve(off: Offer, req: dict, kind: str, retain: int = SERIES_RETAIN_MIN) -> dict:
    default = off.default or {}
    dv = {}
    for key in _MEMBERS:
        if key in req:
            dv[key] = req[key]
        elif key in default:
            dv[key] = default[key]
    budget = dict(default.get("budget") or {})
    budget.update(req.get("budget") or {})
    if kind == "series":
        for key in ("index", "reduce", "tile"):
            if dv.get(key):
                raise DataViewError(f"a series takes no {key}")
        mb = budget.get("max_backfill")
        if mb is not None and (not _is_int(mb) or mb < 0):
            raise DataViewError(f"budget max_backfill must be an integer of 0 or more, not {_say(mb)}")
        return {"budget": {"max_backfill": retain if mb is None else min(mb, retain)}}
    if not off.sources:
        raise DataViewError("this view has no source with data yet")
    name = dv.get("source")
    if name is None:
        src = off.sources[0]
    else:
        src = off.source(name) if isinstance(name, str) else None
        if src is None:
            raise DataViewError(f"no source {_say(name)}; this view offers "
                                f"{_names(s.name for s in off.sources)}")
    sizes = dict(zip(src.dims, src.sizes))

    index = dv.get("index") or {}
    if not isinstance(index, dict):
        raise DataViewError("index must be an object that maps dims to an integer or a "
                            "[start, stop] or [start, stop, step] range")
    rindex = {}
    for dim, sel in index.items():
        if dim not in sizes:
            raise DataViewError(f"source {_say(src.name)} has no dim {_say(dim)}; its dims are "
                                f"{_names(src.dims)}")
        size = sizes[dim]
        if _is_int(sel):
            i = sel + size if sel < 0 else sel
            if not 0 <= i < size:
                raise DataViewError(f"index {_say(sel)} is out of range for {_say(dim)}, "
                                    f"which has {size} positions")
            rindex[dim] = i
        elif isinstance(sel, (list, tuple)) and len(sel) in (2, 3) and all(
                x is None or _is_int(x) for x in sel):
            start, stop = sel[0], sel[1]
            step = sel[2] if len(sel) == 3 and sel[2] is not None else 1
            if step < 1:
                raise DataViewError(f"the range of {_say(dim)} has step {_say(step)}; "
                                    "a step is at least 1")
            start = 0 if start is None else (start + size if start < 0 else start)
            stop = size if stop is None else (stop + size if stop < 0 else stop)
            start, stop = min(max(start, 0), size), min(max(stop, 0), size)
            if stop <= start:
                raise DataViewError(f"the range {_say(sel)} of {_say(dim)} selects nothing; "
                                    f"it has {size} positions")
            rindex[dim] = [start, stop, step]
        else:
            raise DataViewError(f"the index of {_say(dim)} must be an integer or a [start, stop] "
                                f"or [start, stop, step] range, not {_say(sel)}")

    reduce = dv.get("reduce") or {}
    if not isinstance(reduce, dict):
        raise DataViewError("reduce must be an object that maps dims to mean, sum, max, min "
                            "or norm")
    for dim, op in reduce.items():
        if dim not in sizes:
            raise DataViewError(f"source {_say(src.name)} has no dim {_say(dim)}; its dims are "
                                f"{_names(src.dims)}")
        if _is_int(rindex.get(dim)):
            raise DataViewError(f"{_say(dim)} is indexed with an integer, so it cannot be reduced")
        if op not in REDUCE_OPS:
            raise DataViewError(f"the reduction of {_say(dim)} must be mean, sum, max, min or "
                                f"norm, not {_say(op)}")

    kept = [d for d in src.dims if not _is_int(rindex.get(d)) and d not in reduce]
    tile = dv.get("tile")
    if tile is not None:
        if kind != "heatmap":
            raise DataViewError("only a heatmap can tile")
        if tile not in kept:
            raise DataViewError(f"the tile dim {_say(tile)} is not kept; the kept dims are "
                                f"{_names(kept)}")
    need = {"heatmap": 3 if tile is not None else 2, "image": 2, "points": 1}.get(kind)
    if need is not None and len(kept) != need:
        what = {"heatmap": "a heatmap keeps 2 dims (3 with tile)" if tile is None else
                "a tiled heatmap keeps 3 dims", "image": "an image keeps 2 dims",
                "points": "points keep 1 dim, the point dim"}[kind]
        raise DataViewError(f"{what}, and this data view keeps {len(kept)} ({_names(kept)}); "
                            "index or reduce the others")

    if kind in ("heatmap", "image"):
        rbudget = {"max_extent": _budget_int(budget, "max_extent", DEFAULT_MAX_EXTENT,
                                             LIMIT_MAX_EXTENT)}
        if kind == "heatmap":
            pool = budget.get("pool") or "mean"
            if pool not in POOL_OPS:
                raise DataViewError(f"budget pool must be mean, max or min, not {_say(pool)}")
            rbudget["pool"] = pool
    elif kind == "points":
        rbudget = {"max_points": _budget_int(budget, "max_points", DEFAULT_MAX_POINTS,
                                             LIMIT_MAX_POINTS)}
    else:
        rbudget = {}
    out = {"source": src.name, "index": rindex, "reduce": dict(reduce), "budget": rbudget}
    if tile is not None:
        out["tile"] = tile
    _layout(src, kind, out)                      # the payload must fit the hard limit
    return out


def layout(offer, resolved) -> dict:
    """How a resolved data view's payload maps onto its source, as SNAPSHOT
    reports it: ``{"extent": [...], "axes": [...], "tile": {...} or None}``,
    each axis ``{"dim", "start", "step", "count", "size"}`` (see the module
    docstring for the window rule)."""
    off = offer if isinstance(offer, Offer) else Offer.from_json(offer)
    if off.kind in ("series", "histogram"):
        return {"extent": [], "axes": [], "tile": None}
    return _layout(off.source(resolved["source"]), off.kind, resolved)


def _layout(src: SourceSpec, kind: str, resolved: dict) -> dict:
    if kind in ("series", "histogram"):
        return {"extent": [], "axes": [], "tile": None}
    index, reduce, tile = resolved["index"], resolved["reduce"], resolved.get("tile")
    axes = {}
    for name, size in zip(src.dims, src.sizes):
        sel = index.get(name)
        if _is_int(sel) or name in reduce:
            continue
        start, stop, step = sel if sel is not None else (0, size, 1)
        axes[name] = {"dim": name, "start": start, "step": step,
                      "count": len(range(start, stop, step))}
    budget = resolved["budget"]
    if kind == "points":
        (a,) = axes.values()
        a["size"] = min(a["count"], budget["max_points"])
        return {"extent": [a["size"]], "axes": [a], "tile": None}
    ext = budget["max_extent"]
    if tile is not None:
        t = axes.pop(tile)
        t["size"] = t["count"]
        cols = math.isqrt(t["count"] - 1) + 1
        rows = -(-t["count"] // cols)
        r, c = axes.values()
        r["size"] = min(r["count"], max(1, ext // rows))
        c["size"] = min(c["count"], max(1, ext // cols))
        extent = [rows * r["size"], cols * c["size"]]
        if max(extent) > LIMIT_MAX_EXTENT:
            raise DataViewError(f"a mosaic of {extent[0]} by {extent[1]} exceeds "
                                f"{LIMIT_MAX_EXTENT}")
        return {"extent": extent, "axes": [t, r, c], "tile": {"dim": tile, "rows": rows,
                                                               "cols": cols}}
    r, c = axes.values()
    r["size"] = min(r["count"], ext)
    c["size"] = min(c["count"], ext)
    return {"extent": [r["size"], c["size"]], "axes": [r, c], "tile": None}


# ---------------------------------------------------------------------------
# application on the device
# ---------------------------------------------------------------------------
class Program:
    """A resolved data view compiled for :func:`apply` against one source's
    dims and sizes. ``resolved`` is the data view, ``axes`` and ``tile`` the
    payload layout a snapshot reports."""
    __slots__ = ("resolved", "kind", "sizes", "index", "steps", "perm", "pool", "axes", "tile",
                 "extent", "sample", "row_size", "row_slice")


def compile_view(offer, resolved) -> Program:
    """Compile ``resolved`` (from :func:`resolve`) for :func:`apply`."""
    off = offer if isinstance(offer, Offer) else Offer.from_json(offer)
    src = off.source(resolved.get("source"))
    if src is None:
        raise DataViewError(f"no source {_say(resolved.get('source'))} in this offer")
    lay = _layout(src, off.kind, resolved)
    index, reduce = resolved["index"], resolved["reduce"]
    p = Program()
    p.resolved, p.kind, p.sizes = resolved, off.kind, tuple(src.sizes)
    p.axes, p.tile, p.extent = lay["axes"], lay["tile"], lay["extent"]
    sel = [slice(None) if index.get(d) is None else
           index[d] if _is_int(index[d]) else slice(*index[d]) for d in src.dims]
    while sel and sel[-1] == slice(None):
        sel.pop()
    p.index = tuple(sel)
    remaining = [d for d in src.dims if not _is_int(index.get(d))]
    steps = []                 # one dim at a time in the source's order; a run of one op at once
    for d in remaining:
        op = reduce.get(d)
        if op is None:
            continue
        pos = sum(1 for e in remaining[:remaining.index(d)] if e not in reduce)
        if steps and steps[-1][0] == op:
            steps[-1][1].append(pos + len(steps[-1][1]))
        else:
            steps.append((op, [pos]))
    p.steps = tuple((op, tuple(ax)) for op, ax in steps)
    kept = [d for d in remaining if d not in reduce]
    p.perm = None
    p.pool = None
    p.sample = None
    p.row_size = p.row_slice = None
    if off.kind in ("heatmap", "image"):
        op = resolved["budget"].get("pool", "mean") if off.kind == "heatmap" else "mean"
        cell = p.axes[1:] if p.tile is not None else p.axes
        if p.tile is not None:
            order = [kept.index(a["dim"]) for a in p.axes]
            p.perm = tuple(order) if order != [0, 1, 2] else None
            first = 1
        else:
            first = 0
        p.pool = (op, tuple((first + j, a["count"], a["size"]) for j, a in enumerate(cell)
                            if a["size"] < a["count"]))
    elif off.kind == "points":
        a = p.axes[0]
        if a["size"] < a["count"]:
            p.sample = (a["count"], a["size"])
        p.row_size = dict(zip(src.dims, src.sizes))[a["dim"]]
        r = index.get(a["dim"])
        p.row_slice = slice(*r) if r is not None else None
    return p


def _channel_axis(shape, dtype_u8: bool):
    """The RGB(A) axis of an image tensor by the rule of show: (..., H, W, 3|4)
    when uint8 or H is not 1, 3 or 4, else (..., 3|4, H, W); None if neither."""
    if len(shape) < 3:
        return None
    if shape[-1] in (3, 4) and (dtype_u8 or shape[-3] not in (1, 3, 4)):
        return len(shape) - 1
    if shape[-3] in (3, 4):
        return len(shape) - 3
    return None


def dim_sizes(kind: str, shape, dtype_u8: bool = False):
    """The sizes of a source's dims from its tensor's shape: every axis, but the
    RGB(A) axis of an image and the coordinate axis of points. None when the
    shape cannot be such a source."""
    shape = tuple(shape)
    if kind == "image":
        ch = _channel_axis(shape, dtype_u8)
        return None if ch is None else shape[:ch] + shape[ch + 1:]
    if kind == "points":
        return shape[:-1] if shape and shape[-1] in (2, 3) else None
    return shape


def apply(tensor, program: Program):
    """Compute the data view of ``program`` (from :func:`compile_view`) on
    ``tensor`` on its own device: index, reduce, tile, then pool or sample.
    Only device operations are queued; nothing is read back. Returns the
    payload tensor: (H, W) for a heatmap, (H, W, C) for an image, (N, D) for
    points, any shape for a histogram. Raises DataViewError when the tensor
    does not have the source's shape."""
    from ._convert import _as_tensor
    return _apply(_as_tensor(tensor), program)


def _apply(x, p: Program, pre: bool = False):
    """``apply`` on a detached tensor. With ``pre``, returns (payload, the data
    view before it is tiled, pooled or sampled), whose range a heatmap reports."""
    import torch
    if p.kind == "image":
        ch = _channel_axis(x.shape, x.dtype == torch.uint8)
        if ch is not None and ch != x.ndim - 1:
            x = x.movedim(ch, -1)
        lead = None if ch is None else x.shape[:-1]
    elif p.kind == "points":
        lead = x.shape[:-1] if x.ndim and x.shape[-1] in (2, 3) else None
    else:
        lead = x.shape
    if lead != p.sizes:
        raise DataViewError(f"the tensor has shape {tuple(x.shape)}, which does not fit the "
                            f"source's dims of sizes {p.sizes}")
    if p.index:
        x = x[p.index]
    dtype0 = x.dtype
    if p.steps or (p.pool and p.pool[1]) or p.tile is not None:
        if dtype0 not in (torch.float32, torch.float64):
            x = x.float()                   # integers, bool and half types work in f32
        for op, ax in p.steps:
            x = _reduce(x, ax, op)
        before = x
        if p.perm is not None:
            x = x.permute(p.perm)
        if p.pool and p.pool[1]:
            for axis, count, size in p.pool[1]:
                x = _pool_axis(x, axis, count, size, p.pool[0])
        if p.tile is not None:
            x = _mosaic(x, p.tile["rows"], p.tile["cols"])
        if p.kind == "image" and not dtype0.is_floating_point:
            x = x.round_()                  # an integer image keeps its dtype (show's 0..255 rule)
            if dtype0 == torch.uint8:
                x = x.clamp_(0, 255)
            x = x.to(dtype0)
    else:
        before = x
    if p.sample is not None:
        x = x.index_select(0, _stride(p.sample[0], p.sample[1], x.device))
    return (x, before) if pre else x


def _reduce(x, ax, op):
    import torch
    if op == "mean":
        return x.mean(dim=ax)
    if op == "sum":
        return x.sum(dim=ax)
    if op == "max":
        return x.amax(dim=ax)
    if op == "min":
        return x.amin(dim=ax)
    return torch.linalg.vector_norm(x, 2, dim=ax)


def _pool_axis(x, axis, count, size, op):
    """Pool ``axis`` from count positions to size cells by the window rule.

    Even windows: a reshape and a reduction. Uneven ones, mean: window i is the
    segment [s_i, s_{i+1}) (s_i = floor(i * count / size)) plus position
    s_{i+1} when (i + 1) * count / size is not an integer, so one index_add_
    of segment sums, one small gather for the shared positions and a division.
    Uneven max and min: a gather of the windows, each padded with its own last
    position (index_reduce_ is still beta in torch)."""
    import torch
    shape = tuple(x.shape)
    if count % size == 0:
        k = count // size
        return _reduce(x.reshape(shape[:axis] + (size, k) + shape[axis + 1:]), (axis + 1,), op)
    tail = (1,) * (x.ndim - axis - 1)
    if op == "mean":
        seg, nxt, shared, n = _segments(count, size, x.device)
        out = torch.zeros(shape[:axis] + (size,) + shape[axis + 1:], dtype=x.dtype, device=x.device)
        out.index_add_(axis, seg, x)
        extra = x.index_select(axis, nxt)
        out += torch.where(shared.reshape((size,) + tail), extra, 0.0)
        return out.div_(n.reshape((size,) + tail))
    idx, width = _windows(count, size, x.device)
    g = x.index_select(axis, idx).reshape(shape[:axis] + (size, width) + shape[axis + 1:])
    return _reduce(g, (axis + 1,), op)


def _mosaic(x, rows, cols):
    """(C, H, W) cells -> (rows * H, cols * W), cell t at (t // cols, t % cols),
    the cells left over NaN (the channel mosaic of show)."""
    import torch
    c, h, w = x.shape
    out = torch.full((rows * h, cols * w), float("nan"), dtype=x.dtype, device=x.device)
    grid = out.view(rows, h, cols, w)
    full, rem = divmod(c, cols)
    if full:
        grid[:full].copy_(x[:full * cols].reshape(full, cols, h, w).permute(0, 2, 1, 3))
    if rem:
        grid[full, :, :rem].copy_(x[full * cols:].permute(1, 0, 2))
    return out


# Index tensors of the window rule, made on the device by arange arithmetic (no
# upload) once per (count, size, device) and cached.
_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: OrderedDict = OrderedDict()
_INDEX_CACHE_SIZE = 64


def _cached(key, make):
    with _INDEX_LOCK:
        v = _INDEX_CACHE.get(key)
        if v is not None:
            _INDEX_CACHE.move_to_end(key)
            return v
    v = make()
    with _INDEX_LOCK:
        _INDEX_CACHE[key] = v
        while len(_INDEX_CACHE) > _INDEX_CACHE_SIZE:
            _INDEX_CACHE.popitem(last=False)
    return v


def _stride(count: int, size: int, device):
    """Positions floor(i * count / size) for i in [0, size), on the device."""
    def make():
        import torch
        i = torch.arange(size, device=device, dtype=torch.int64)
        return torch.div(i * count, size, rounding_mode="floor")
    return _cached(("stride", count, size, device), make)


def _windows(count: int, size: int, device):
    """The windows [floor(i * count / size), ceil((i + 1) * count / size)) as a
    gather index (size * width,), each window padded with its last position."""
    def make():
        import numpy as np
        import torch
        i = np.arange(size, dtype=np.int64)
        width = int((-((-(i + 1) * count) // size) - (i * count) // size).max())
        it = torch.arange(size, device=device, dtype=torch.int64)
        start = torch.div(it * count, size, rounding_mode="floor")
        stop = torch.div((it + 1) * count + (size - 1), size, rounding_mode="floor")
        pos = start[:, None] + torch.arange(width, device=device, dtype=torch.int64)[None, :]
        return torch.minimum(pos, (stop - 1)[:, None]).reshape(-1), width
    return _cached(("windows", count, size, device), make)


def _segments(count: int, size: int, device):
    """For the uneven mean: the segment of each position, floor(((j + 1) * size
    - 1) / count); the position s_{i+1} each window may share with the next,
    whether it does, and the window lengths."""
    def make():
        import torch
        j = torch.arange(count, device=device, dtype=torch.int64)
        seg = torch.div((j + 1) * size - 1, count, rounding_mode="floor")
        it = torch.arange(size, device=device, dtype=torch.int64)
        start = torch.div(it * count, size, rounding_mode="floor")
        stop = torch.div((it + 1) * count + (size - 1), size, rounding_mode="floor")
        nxt = torch.div((it + 1) * count, size, rounding_mode="floor")
        shared = stop > nxt
        return seg, nxt.clamp_(max=count - 1), shared, (stop - start).float()
    return _cached(("segments", count, size, device), make)


def take_rows(t, program: Program):
    """The values of a per-point tensor ``t`` ((n,) labels, aligned with the
    kept dim of a points data view) that the payload's points carry: the
    dim's range, then the same stride sample. On t's device."""
    from ._convert import _as_tensor
    if program.row_size is None:
        raise DataViewError("take_rows() needs a points data view")
    x = _as_tensor(t).reshape(-1)
    if x.shape[0] != program.row_size:
        raise DataViewError(f"{x.shape[0]} values for a point dim of {program.row_size}")
    if program.row_slice is not None:
        x = x[program.row_slice]
    if program.sample is not None:
        x = x.index_select(0, _stride(program.sample[0], program.sample[1], x.device))
    return x


def source_rows(program: Program, indices=None):
    """The source index along the point dim of each point of a points payload
    (or of the points at ``indices``), as a numpy int64 array computed on the
    host: start + step * floor(i * count / size). This maps a selection back."""
    import numpy as np
    if program.row_size is None:
        raise DataViewError("source_rows() needs a points data view")
    a = program.axes[0]
    i = np.arange(a["size"], dtype=np.int64) if indices is None else \
        np.asarray(indices, dtype=np.int64).reshape(-1)
    if i.size and (i.min() < 0 or i.max() >= a["size"]):
        raise DataViewError(f"a point index is out of range for {a['size']} points")
    return a["start"] + a["step"] * ((i * a["count"]) // a["size"])


# ---------------------------------------------------------------------------
# view calls and their snapshot plans
# ---------------------------------------------------------------------------
class DataCall:
    """One view call of a view with data views, as the routes see it: the
    registry entry, the source the call carries, its tensor (``tensor()``
    fetches a registered source once, when a route needs it) and the call's
    options (``cmap``, ``vmin``, ``vmax``, ``raw``, ``labels``, ``classes``,
    ``bins``)."""
    __slots__ = ("entry", "source", "x", "opts", "fetched", "_labels")

    def __init__(self, entry, source: str, x, opts: dict):
        self.entry, self.source, self.opts = entry, source, opts
        self.x, self.fetched, self._labels = x, x is not None, ()

    def tensor(self):
        if not self.fetched:
            self.fetched = True
            src = self.entry.sources.get(self.source)
            t = None if src is None else src.fetch()
            if t is not None:
                from ._convert import _as_tensor
                t = _as_tensor(t)
                REGISTRY._observe(self.entry.kind, src, t)
            self.x = t
        return self.x

    def labels(self):
        if self._labels == ():
            lab = self.opts.get("labels")
            self._labels = lab() if callable(lab) else lab
        return self._labels


class DVWrite:
    """The ``write`` of a data view's snapshot plan. It computes the payload on
    the device only when a route writes a slot; the network route finds the
    view call here (``call``) and computes each subscription's data view."""
    __slots__ = ("call", "program")

    def __init__(self, call: DataCall, program):
        self.call, self.program = call, program

    def __call__(self, slot) -> None:
        p = self.program
        y, pre = _apply(self.call.tensor(), p, pre=True)
        finish(slot, p, pre, payload_plan(p, y, self.call))


def payload_plan(p: Program, y, call: DataCall):
    """The caliper._convert plan that writes payload ``y`` of data view ``p``."""
    from ._convert import _image_plan, plan_histogram, plan_points, plan_show
    o = call.opts
    if p.kind == "heatmap":
        return plan_show(y, o.get("cmap", 0), o.get("vmin"), o.get("vmax"), o.get("raw", False))
    if p.kind == "image":
        return _image_plan(y, o.get("raw", False))
    if p.kind == "points":
        lab = call.labels()
        if lab is not None:
            lab = take_rows(lab, p)
        return plan_points(y, lab, o.get("classes"), o.get("cmap", 0), o.get("vmin"), o.get("vmax"))
    return plan_histogram(y, int(o.get("bins", 64)))


def finish(slot, p: Program, pre, plan) -> None:
    """Write ``plan`` into ``slot`` and record the data view on the slot
    (``slot.data_view``, ``slot.axes``, ``slot.tile``). A pooled or tiled
    heatmap reports the range of the data before pooling, padding excluded
    (the rule of protocol v2), not that of its payload."""
    import torch
    from ._ring import FLAG_AUTO_VMAX, FLAG_AUTO_VMIN
    plan.write(slot)
    if p.kind == "heatmap" and plan.flags & (FLAG_AUTO_VMIN | FLAG_AUTO_VMAX) and (
            p.tile is not None or (p.pool and p.pool[1])):
        torch.aminmax(pre if pre.dtype == torch.float32 else pre.float(), out=(slot.lo, slot.hi))
    slot.data_view, slot.axes, slot.tile = p.resolved, p.axes, p.tile


def lazy_plan(call: DataCall, p: Program):
    """The snapshot plan of the view's default data view ``p`` for the routes
    that draw it (window, session): the layout of the caliper._convert plan the
    payload will have, with a write that computes it. Building it queues
    nothing on the device."""
    import torch
    from ._convert import Plan, _range_flags
    from ._ring import (FLAG_AUTO_VMAX, FLAG_AUTO_VMIN, FLAG_RAW, KIND_HEATMAP, KIND_HISTOGRAM,
                        KIND_IMAGE, KIND_POINTS, STATS_FLOATS)
    x, o = call.x, call.opts
    dev = x.device
    write = DVWrite(call, p)
    raw = FLAG_RAW if o.get("raw") else 0
    if p.kind == "heatmap":
        flags, lo, hi = _range_flags(o.get("vmin"), o.get("vmax"))
        shape = tuple(p.extent)
        return Plan(KIND_HEATMAP, shape, torch.float32, dev, (KIND_HEATMAP, shape, dev, "dv"), write,
                    None, colormap=o.get("cmap", 0), vmin=lo, vmax=hi, flags=flags | raw,
                    stats_count=2)
    if p.kind == "image":
        c = x.shape[_channel_axis(x.shape, x.dtype == torch.uint8)]
        shape = (*p.extent, 4)

        def init(slot):
            slot.payload[..., 3].fill_(255)
        return Plan(KIND_IMAGE, shape, torch.uint8, dev, (KIND_IMAGE, shape, dev, c), write,
                    init if c == 3 else None, flags=raw, stats_count=0)
    if p.kind == "points":
        n, d = p.extent[0], x.shape[-1]
        has = call.labels() is not None
        classes = o.get("classes")
        count = int(classes) if classes else 0
        flags, lo, hi = _range_flags(o.get("vmin"), o.get("vmax")) if has and not count \
            else (0, 0.0, 1.0)

        def init(slot):
            slot.payload.zero_()
            slot.stats.zero_()
        return Plan(KIND_POINTS, (n, 3), torch.float32, dev, (KIND_POINTS, (n, 3), dev, d, has),
                    write, init, aux_shape=(n,) if has else None, colormap=o.get("cmap", 0),
                    vmin=lo, vmax=hi, flags=flags, points_dims=d, label_count=count,
                    stats_count=STATS_FLOATS)
    bins = int(o.get("bins", 64))
    return Plan(KIND_HISTOGRAM, (bins,), torch.float32, dev, (KIND_HISTOGRAM, (bins,), dev), write,
                flags=FLAG_AUTO_VMIN | FLAG_AUTO_VMAX, stats_count=2)


def net_plan(call: DataCall):
    """A plan that only carries ``call`` to the network route (a source that
    only a subscription selects): no layout, no device work."""
    from ._convert import Plan
    kind = KINDS[call.entry.kind]
    return Plan(kind, (), None, None, (kind, "dv-net"), DVWrite(call, None))


# ---------------------------------------------------------------------------
# default data views
# ---------------------------------------------------------------------------
def auto_default(kind: str, dims, given=None) -> dict:
    """The default data view a view gets from its dims: ``given`` (members set
    by Python code) and, when it sets no index, an index of 0 for the leading
    dims it does not reduce or tile, until the kind fits: a heatmap keeps its
    last two dims (three with ``tile``), an image two, points the last one. A
    batch thus shows item 0, as ``show`` does. No tile is added, so a request
    that only names an index replaces this default cleanly."""
    req = dict(given or {})
    if "index" in req or kind not in ("heatmap", "image", "points"):
        return req
    reduce = req.get("reduce") or {}
    tile = req.get("tile")
    need = {"heatmap": 3 if tile is not None else 2, "image": 2, "points": 1}[kind]
    kept = [d for d in dims if d not in reduce]
    index = {}
    for d in list(kept):
        if len(kept) <= need:
            break
        if d != tile:
            index[d] = 0
            kept.remove(d)
    req["index"] = index
    return req


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
class Source:
    """A source registered for a view: its dims, the dim sizes and dtype its
    tensor last had (``meta``, replaced as a whole), and how to get the tensor.

    ``get`` is None (the view call or a hook passes it) or a callable returning
    the current tensor or None (called under no_grad). ``namer(n)`` names the
    dims of a source whose rank is only known from its tensor."""
    __slots__ = ("name", "meta", "get", "namer")

    def __init__(self, name, dims=None, sizes=None, dtype=None, get=None, namer=None):
        _check_name(name, "source name")
        if dims is not None:
            dims = _check_dims(dims)
        self.name = name
        self.meta = (dims, None if sizes is None else tuple(sizes), dtype)
        self.get = get
        self.namer = namer

    def spec(self) -> SourceSpec | None:
        dims, sizes, dtype = self.meta
        return None if dims is None or sizes is None else SourceSpec(self.name, dims, sizes, dtype)

    def fetch(self):
        if self.get is None:
            return None
        import torch
        with torch.no_grad():
            return self.get()


def _check_dims(dims, n=None) -> tuple:
    if isinstance(dims, str):
        raise ValueError("dims must be a sequence of dim names, not one string")
    try:
        dims = tuple(dims)
    except TypeError:
        raise ValueError(f"dims must be a sequence of dim names, not {dims!r}") from None
    for d in dims:
        _check_name(d, "dim name")
    if len(set(dims)) != len(dims):
        raise ValueError(f"dim names must be unique, got {dims}")
    if len(dims) > MAX_DIMS:
        raise ValueError(f"a source has at most {MAX_DIMS} dims, got {len(dims)}")
    if n is not None and len(dims) != n:
        raise ValueError(f"{len(dims)} dim names {dims} for {n} dims")
    return dims


def _auto_dims(n: int) -> tuple:
    return tuple(f"dim{i}" for i in range(n))


def _explicit(rv: dict) -> dict:
    """A resolved data view as a request that names every member, so that
    resolving it again takes nothing from a default."""
    return {"source": rv["source"], "index": rv["index"], "reduce": rv["reduce"],
            "tile": rv.get("tile"), "budget": rv["budget"]}


class _Entry:
    """A view in the registry. ``state`` = (version, current, selected) is
    replaced as a whole: ``current`` is the data view set from Python (an
    explicit request) or None for the registered default, ``selected`` the
    source it reads. ``cache`` belongs to the view call (the caliper lock)."""
    __slots__ = ("name", "kind", "sources", "given", "rate", "options", "call", "state",
                 "pending", "cache", "warned")

    def __init__(self, name, kind, sources, given, rate, options, call):
        self.name = name
        self.kind = kind
        self.sources = {s.name: s for s in sources}
        self.given = given
        self.rate = rate
        self.options = options or {}
        self.call = call
        first = next(iter(self.sources), None)
        self.state = (0, None, given.get("source") if isinstance(given.get("source"), str)
                      and given.get("source") in self.sources else first)
        self.pending = None
        self.cache = None
        self.warned = -1

    def specs(self) -> tuple:
        return tuple(s for s in (src.spec() for src in self.sources.values()) if s is not None)

    def base(self, specs, source=None) -> dict:
        """The registered default resolved (for ``source``, or its own source)."""
        given = self.given
        name = source or (given.get("source") if isinstance(given.get("source"), str) else None) \
            or (specs[0].name if specs else None)
        spec = next((s for s in specs if s.name == name), None)
        if spec is None:
            raise DataViewError(f"source {_say(name)} has no data yet")
        req = auto_default(self.kind, spec.dims, given)
        req["source"] = name
        return _resolve(Offer(self.kind, specs), req, self.kind)

    def close(self) -> None:
        self.state = (self.state[0] + 1, None, None)      # selects nothing: hooks stop


class Registry:
    """What each view offers and its default data view (one per process).

    The transport's side, any thread: :meth:`offer`, :meth:`offers`,
    :meth:`request`, :meth:`resolved`, :meth:`rows` and ``generation`` (bumped
    when an offer or a default changes). The view call's side, under caliper's
    lock: :meth:`for_call`, :meth:`observe`, :meth:`prepare`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._views: dict[str, _Entry] = {}
        self._pending: dict[str, dict] = {}
        self.generation = 0

    # -- registration --------------------------------------------------------------
    def register(self, name, kind, sources, *, default=None, rate=None, options=None,
                 call=None) -> _Entry:
        """Register (or replace) view ``name`` with ``sources`` (Source objects)
        and the members of its default data view set by Python (``default``,
        completed by :func:`auto_default`). A replaced view's data view, or one
        set before the view existed, is kept while it still resolves."""
        kind = _kind(kind)
        _check_name(name, "view name")
        sources = list(sources)
        if len(sources) > MAX_SOURCES:
            raise ValueError(f"a view has at most {MAX_SOURCES} sources, got {len(sources)}")
        if len({s.name for s in sources}) != len(sources):
            raise ValueError(f"view {name!r}: source names must be unique")
        given = parse_request(default)
        if kind == "series":
            raise ValueError("a series view has no sources")
        e = _Entry(name, kind, sources, given, rate, options, call)
        specs = e.specs()
        if given and specs and (not isinstance(given.get("source"), str)
                                or any(s.name == given["source"] for s in specs)):
            e.base(specs)                         # the given default must resolve
        with self._lock:
            old = self._views.get(name)
            carried = self._pending.pop(name, None)
            if carried is not None:
                e.pending = carried
            elif old is not None and old.state[1] is not None:
                cur = old.state[1]
                if cur["source"] in e.sources:
                    e.state = (old.state[0] + 1, cur, cur["source"])
            elif old is not None and old.pending is not None:
                e.pending = old.pending
            if old is not None:
                old.close()
            self._views[name] = e
            self.generation += 1
        if e.pending is not None and specs:
            self._settle(e)
        return e

    def offer_view(self, name, sources, kind="heatmap", dims=None, *, default=None, rate=None,
                   options=None) -> _Entry:
        """Register view ``name`` from a mapping of source names to tensors or
        callables (``caliper.offer``). Each callable runs once here, under
        no_grad, to learn its shape."""
        from ._convert import _as_tensor
        kind = _kind(kind)
        if kind == "series":
            raise ValueError("a series view has no sources; use caliper.series")
        if not isinstance(sources, dict):
            sources = {name: sources}
        if not sources:
            raise ValueError("offer() needs at least one source")
        out = []
        for sname, v in sources.items():
            sd = dims.get(sname) if isinstance(dims, dict) else dims
            get = v if callable(v) else (lambda t=v: t)
            src = Source(sname, None if sd is None else _check_dims(sd), get=get,
                         namer=_auto_dims if sd is None else None)
            t = src.fetch()
            if t is not None:
                self._observe(kind, src, _as_tensor(t))
            out.append(src)
        return self.register(name, kind, out, default=default, rate=rate, options=options)

    def for_call(self, name, call: str, dims, data_view, value):
        """The view of a ``show``, ``points`` or ``histogram`` call with ``dims``
        or ``data_view``: registered at the first call and again when the dims,
        the default or the kind change. Returns (entry, value as a detached
        torch tensor)."""
        from ._convert import _as_tensor
        import torch
        x = _as_tensor(value)
        if call == "show":
            n = x.ndim if dims is None else len(dims)
            kind = "heatmap" if n == x.ndim else "image"
        else:
            kind = call
        e = self._views.get(name)
        if e is not None and e.call is not None and e.kind == kind and e.call[0] == dims \
                and (e.call[1] is data_view or e.call[1] == data_view):
            return e, x
        u8 = x.dtype == torch.uint8
        sizes = dim_sizes(kind, x.shape, u8)
        if sizes is None:
            want = {"image": "an image tensor (..., H, W, 3|4) or (..., 3|4, H, W)",
                    "points": "a points tensor (..., N, 2|3)"}.get(kind, "")
            raise ValueError(f"{name!r}: dims {dims} do not fit a tensor of shape "
                             f"{tuple(x.shape)}; {kind} needs {want}")
        d = _auto_dims(len(sizes)) if dims is None else _check_dims(dims, len(sizes))
        src = Source(name, d, sizes, dtype_name(x.dtype))
        return self.register(name, kind, [src], default=data_view, call=(dims, data_view)), x

    def remove(self, name, entry=None) -> bool:
        """Drop view ``name`` (only if it is still ``entry``, when given)."""
        with self._lock:
            self._pending.pop(name, None)
            e = self._views.get(name)
            if e is None or (entry is not None and e is not entry):
                return False
            del self._views[name]
            e.close()
            self.generation += 1
            return True

    def clear(self) -> None:
        with self._lock:
            for e in self._views.values():
                e.close()
            self._views.clear()
            self._pending.clear()
            self.generation += 1

    # -- the transport's side -------------------------------------------------------
    def get(self, name) -> _Entry | None:
        return self._views.get(name)

    def names(self) -> list:
        return list(self._views)

    def offer(self, name) -> Offer | None:
        """View ``name`` as viewers see it (``.to_json()`` is its VIEWS entry
        without the id), with the current default data view; None when unknown
        or before any source has a tensor."""
        e = self._views.get(name)
        if e is None:
            return None
        specs = e.specs()
        if not specs:
            return None
        try:
            default = self._effective(e, specs, warn=False)
        except DataViewError:
            default = None
        return Offer(e.kind, specs, default, e.name)

    def offers(self) -> dict:
        """{view name: Offer} for every view with data."""
        out = {}
        for n in list(self._views):
            o = self.offer(n)
            if o is not None:
                out[n] = o
        return out

    def request(self, name, request) -> dict | None:
        """Set the default data view of view ``name`` from ``request`` (any
        thread): resolved against the current default, so ``{}`` keeps it, and
        None returns to the registered default. Returns the resolved data view,
        or None when it cannot be checked yet (the view does not exist yet or
        has no data yet); it is then checked at the first snapshot. Raises
        DataViewError, and the view keeps the data view it had."""
        req = None if request is None else parse_request(request)
        with self._lock:
            e = self._views.get(name)
            if e is None:
                if req is None:
                    self._pending.pop(name, None)
                    return None
                if not isinstance(name, str) or not 0 < len(name.encode("utf-8", "replace")) <= NAME_MAX:
                    raise DataViewError(f"unknown view {_say(name)}")
                if name not in self._pending and len(self._pending) >= PENDING_LIMIT:
                    raise DataViewError(f"unknown view {_say(name)}")
                self._pending[name] = req
                return None
            specs = e.specs()
            if req is None:
                e.pending = None
                e.state = (e.state[0] + 1, None, e.state[2] if e.state[1] is None else
                           (e.given.get("source") if isinstance(e.given.get("source"), str)
                            else next(iter(e.sources))))
                self.generation += 1
                return self._effective(e, specs, warn=False) if specs else None
            src = req.get("source")
            if not specs or (isinstance(src, str) and src in e.sources
                             and e.sources[src].spec() is None):
                e.pending = req
                return None
            rv = _resolve(Offer(e.kind, specs, self._effective(e, specs, warn=False)), req, e.kind)
            e.pending = None
            e.state = (e.state[0] + 1, _explicit(rv), rv["source"])
            self.generation += 1
        return rv

    def resolved(self, name) -> dict | None:
        """The current default data view of ``name``, resolved; None when the
        view is unknown or has no data yet."""
        o = self.offer(name)
        return None if o is None else o.default

    def rows(self, name, indices=None):
        """Source indices of the points of the last points snapshot of ``name``
        (see :func:`source_rows`), or None."""
        e = self._views.get(name)
        c = None if e is None else e.cache
        if c is None or c[2].row_size is None:
            return None
        return source_rows(c[2], indices)

    # -- the view call's side --------------------------------------------------------------
    def observe(self, entry: _Entry, name, t) -> None:
        """Record the dim sizes and dtype of source ``name`` (metadata only)."""
        src = entry.sources.get(name)
        if src is not None:
            self._observe(entry.kind, src, t)

    def _observe(self, kind, src: Source, t) -> None:
        dims, sizes, dtype = src.meta
        s = t.shape
        dt = dtype_name(t.dtype)
        if kind in ("heatmap", "histogram"):
            if s == sizes and dt == dtype:
                return
            new = tuple(s)
        else:
            new = dim_sizes(kind, s, dt == "u8")
            if new == sizes and dt == dtype:
                return
            if new is None:
                raise DataViewError(f"source {src.name!r}: a tensor of shape {tuple(s)} is not "
                                    f"{'an image' if kind == 'image' else 'a points'} source")
        if src.namer is not None and (dims is None or len(dims) != len(new)):
            dims = tuple(src.namer(len(new)))
        if dims is None or len(dims) != len(new):
            raise DataViewError(f"source {src.name!r} has the dims {dims}, but its tensor has "
                                f"shape {tuple(s)}")
        src.meta = (dims, new, dt)
        self.generation += 1

    def _settle(self, e: _Entry) -> None:
        """Apply a request kept until the view had data; warn if it does not fit."""
        req, e.pending = e.pending, None
        try:
            with self._lock:
                specs = e.specs()
                rv = _resolve(Offer(e.kind, specs, self._effective(e, specs, warn=False)), req,
                              e.kind)
                e.state = (e.state[0] + 1, _explicit(rv), rv["source"])
                self.generation += 1
        except DataViewError as err:
            warnings.warn(f"caliper: the data view set for {e.name!r} does not fit it ({err})",
                          RuntimeWarning, stacklevel=4)

    def _effective(self, e: _Entry, specs, warn: bool = True) -> dict:
        """The view's default data view for the current sizes: the one set from
        Python, else the registered default; if the sizes changed so that it no
        longer fits, the registered default of the same source, then the
        automatic one (with a warning, once per change)."""
        version, current, sel = e.state
        err = None
        if current is not None:
            try:
                return _resolve(Offer(e.kind, specs), current, e.kind)
            except DataViewError as x:
                err = x
        try:
            rv = e.base(specs, sel)
        except DataViewError as x:
            err = err or x
            spec = next((s for s in specs if s.name == sel), None)
            if spec is None:
                raise
            req = auto_default(e.kind, spec.dims)
            req["source"] = sel
            rv = _resolve(Offer(e.kind, specs), req, e.kind)
        if err is not None and warn and e.warned != version:
            e.warned = version
            warnings.warn(f"caliper: the data view of {e.name!r} does not fit its tensor "
                          f"({err}); using the default", RuntimeWarning, stacklevel=4)
        return rv

    def default_program(self, entry: _Entry, x):
        """The Program of the view's default data view for ``x``, a tensor of its
        selected source, or None when a request settled just now selects
        another source. The resolution is cached per default and dim sizes."""
        version, _, sel = entry.state
        src = entry.sources.get(sel)
        if src is None:
            return None
        self._observe(entry.kind, src, x)
        if entry.pending is not None:
            self._settle(entry)
            version, _, now = entry.state
            if now != sel:                        # the request picked another source: next call
                return None
        key = (version, sel, src.meta)
        c = entry.cache
        if c is None or c[0] != key:
            specs = entry.specs()
            rv = self._effective(entry, specs)
            c = entry.cache = (key, rv, compile_view(Offer(entry.kind, specs), rv))
        return c[2]

    def default_plan(self, call: DataCall):
        """The lazy snapshot plan of the view's default data view for ``call``
        (see :func:`lazy_plan`), or None when there is no tensor."""
        if call.source != call.entry.state[2] or call.tensor() is None:
            return None
        p = self.default_program(call.entry, call.x)
        return None if p is None else lazy_plan(call, p)

    def prepare(self, entry: _Entry, tensor=None):
        """Apply the view's default data view to its selected source: ``tensor``
        when given, else the source's callable. Returns (payload tensor on its
        device, Program), or (None, None) when there is no tensor."""
        src = entry.sources.get(entry.state[2])
        if src is None:
            return None, None
        if tensor is None:
            tensor = src.fetch()
            if tensor is None:
                return None, None
        from ._convert import _as_tensor
        x = _as_tensor(tensor)
        p = self.default_program(entry, x)
        return (None, None) if p is None else (_apply(x, p), p)

    def fetchable(self) -> list:
        """Views whose selected source the registry fetches itself (callables,
        as from ``caliper.offer`` and watch's weights)."""
        out = []
        for e in list(self._views.values()):
            s = e.sources.get(e.state[2])
            if s is not None and s.get is not None:
                out.append(e)
        return out


REGISTRY = Registry()
