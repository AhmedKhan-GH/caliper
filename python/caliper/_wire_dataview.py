"""Wire protocol v3 data view resolution, payload frame bounds, and layout calculation."""
from __future__ import annotations

import math
import sys as _sys

if __package__:
    from . import _wire_frame as _f, _wire_recordings as _r, _wire_scenes as _s
else:
    _prefix = __name__.rpartition(".")[0]
    _f = _sys.modules[f"{_prefix}._wire_frame"]
    _r = _sys.modules[f"{_prefix}._wire_recordings"]
    _s = _sys.modules[f"{_prefix}._wire_scenes"]

DEFAULT_MAX_EXTENT = _f.DEFAULT_MAX_EXTENT
DEFAULT_MAX_POINTS = _f.DEFAULT_MAX_POINTS
ERR_BAD_DATA_VIEW = _f.ERR_BAD_DATA_VIEW
HEADER_BYTES = _f.HEADER_BYTES
Kind = _f.Kind
LIMIT_MAX_EXTENT = _f.LIMIT_MAX_EXTENT
LIMIT_MAX_POINTS = _f.LIMIT_MAX_POINTS
MAX_FRAME_BYTES = _f.MAX_FRAME_BYTES
MAX_META_BYTES = _f.MAX_META_BYTES
MAX_SECTIONS = _f.MAX_SECTIONS
MessageError = _f.MessageError
POOL_OPS = _f.POOL_OPS
REDUCE_OPS = _f.REDUCE_OPS
SECTION_ALIGN = _f.SECTION_ALIGN
SERIES_RETAIN_MIN = _f.SERIES_RETAIN_MIN
_is_int = _f._is_int
_obj = _f._obj

MAX_BARS = _r.MAX_BARS
resolve_scene_data_view = _s.resolve_scene_data_view


def _dv_bad(msg: str):
    raise MessageError(ERR_BAD_DATA_VIEW, msg)


# A SNAPSHOT is one frame of at most MAX_FRAME_BYTES, and its head may take
# the fixed header, a section table of MAX_SECTIONS entries, metadata of up to
# MAX_META_BYTES and each section's alignment padding; this is what is left
# for its payload. Only a heatmap or an image (4 bytes a cell: float32, or
# RGBA8) near LIMIT_MAX_EXTENT on both sides can exceed it: points at
# LIMIT_MAX_POINTS with labels take 16 MB and a histogram at MAX_BINS 256 KiB.
MAX_PAYLOAD_BYTES = (MAX_FRAME_BYTES - MAX_META_BYTES - HEADER_BYTES
                     - MAX_SECTIONS * (4 + SECTION_ALIGN - 1))


def payload_frame_error(kind, max_extent: int, extent_for) -> str | None:
    """The data view rule that follows from MAX_FRAME_BYTES, shared by this
    reference resolver and Caliper LIB's (caliper._dataview): None when the
    payload of a data view fits one frame, else the refusal's message, which
    names the largest ``max_extent`` that would fit. ``kind`` is a Kind or its
    name; ``extent_for(m)`` returns the payload's extent with ``max_extent``
    ``m`` (and may raise when that budget breaks another rule)."""
    if kind not in (Kind.HEATMAP, Kind.IMAGE, "heatmap", "image"):
        return None

    def fits(m: int) -> bool:
        try:
            return 4 * math.prod(extent_for(m)) <= MAX_PAYLOAD_BYTES
        except Exception:
            return False
    extent = extent_for(max_extent)
    if 4 * math.prod(extent) <= MAX_PAYLOAD_BYTES:
        return None
    lo, hi = 1, max_extent
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
    return (f"a {extent[0]} by {extent[1]} payload is too large to send in one frame; "
            f"ask for a smaller max_extent, at most {lo} here")


def _merge(request: dict | None, default: dict | None) -> dict:
    """Members the request leaves out come from the view's default data view;
    budget members are merged one by one."""
    request = request or {}
    default = default or {}
    out = {}
    for key in ("source", "index", "reduce", "tile"):
        if key in request:
            out[key] = request[key]
        elif key in default:
            out[key] = default[key]
    budget = dict(default.get("budget") or {})
    rb = request.get("budget")
    if rb is not None:
        if not _obj(rb):
            _dv_bad("'budget' must be an object")
        budget.update(rb)
    out["budget"] = budget
    return out


def _budget_int(budget: dict, key: str, default: int, limit: int) -> int:
    v = budget.get(key)
    if v is None:
        return default
    if not _is_int(v) or v < 1:
        _dv_bad(f"budget.{key} must be a positive integer, got {v!r}")
    return min(v, limit)


def resolve_data_view(view: dict, request: dict | None, *,
                      retain: int = SERIES_RETAIN_MIN) -> dict:
    """The resolved form of ``request`` for the view entry ``view`` (as in a
    VIEWS message): the request's shape with every default filled in, indices
    normalised and budgets clamped. ``retain`` is the producer's series
    retention. Raises :class:`MessageError` (``bad_data_view``)."""
    if view.get("kind") == Kind.SCENE:          # 3.2 draft
        return resolve_scene_data_view(view, request)
    if request is not None and not _obj(request):
        _dv_bad("a data view is a JSON object")
    dv = _merge(request, view.get("default"))
    kind = view.get("kind")
    budget = dv["budget"]
    if kind in (Kind.SERIES, Kind.TEXT):           # 3.2 draft: a text log's is a series'
        for key in ("index", "reduce", "tile"):
            if dv.get(key):
                _dv_bad(f"a series takes no '{key}'")
        mb = budget.get("max_backfill")
        if mb is not None and (not _is_int(mb) or mb < 0):
            _dv_bad(f"budget.max_backfill must be an integer >= 0, got {mb!r}")
        return {"budget": {"max_backfill": retain if mb is None else min(mb, retain)}}
    sources = view.get("sources") or []
    if not sources:
        _dv_bad(f"view {view.get('name')!r} offers no source")
    name = dv.get("source")
    if name is None:
        src = sources[0]
    else:
        src = next((s for s in sources if s.get("name") == name), None)
        if src is None:
            _dv_bad(f"no source {name!r}")
    dims = {d["name"]: d["size"] for d in src["dims"]}
    order = [d["name"] for d in src["dims"]]

    index = dv.get("index") or {}
    if not _obj(index):
        _dv_bad("'index' must be an object")
    rindex = {}
    for dim, sel in index.items():
        if dim not in dims:
            _dv_bad(f"source {src['name']!r} has no dim {dim!r}")
        size = dims[dim]
        if _is_int(sel):
            i = sel + size if sel < 0 else sel
            if not 0 <= i < size:
                _dv_bad(f"index {sel} out of range for {dim!r} of size {size}")
            rindex[dim] = i
        elif isinstance(sel, list) and len(sel) in (2, 3) and all(
                x is None or _is_int(x) for x in sel):
            start, stop = sel[0], sel[1]
            step = sel[2] if len(sel) == 3 and sel[2] is not None else 1
            if step < 1:
                _dv_bad(f"step {step} for {dim!r}; steps are >= 1")
            start = 0 if start is None else (start + size if start < 0 else start)
            stop = size if stop is None else (stop + size if stop < 0 else stop)
            start, stop = min(max(start, 0), size), min(max(stop, 0), size)
            if stop <= start:
                _dv_bad(f"empty range {sel} for {dim!r} of size {size}")
            rindex[dim] = [start, stop, step]
        else:
            _dv_bad(f"index for {dim!r} must be an integer or [start, stop(, step)], got {sel!r}")

    reduce = dv.get("reduce") or {}
    if not _obj(reduce):
        _dv_bad("'reduce' must be an object")
    for dim, op in reduce.items():
        if dim not in dims:
            _dv_bad(f"source {src['name']!r} has no dim {dim!r}")
        if _is_int(rindex.get(dim)):
            _dv_bad(f"dim {dim!r} is indexed and cannot be reduced")
        if op not in REDUCE_OPS:
            _dv_bad(f"reduce op {op!r}; expected one of {list(REDUCE_OPS)}")

    kept = [d for d in order if not _is_int(rindex.get(d)) and d not in reduce]
    tile = dv.get("tile")
    if tile is not None:
        if kind != Kind.HEATMAP:
            _dv_bad("only a heatmap can tile")
        if tile not in kept:
            _dv_bad(f"tile dim {tile!r} is not kept")
    need = {Kind.HEATMAP: 3 if tile is not None else 2, Kind.IMAGE: 2, Kind.POINTS: 1,
            Kind.BARS: 1}.get(kind)
    if need is not None and len(kept) != need:
        _dv_bad(f"kind {kind} needs {need} kept dims, the data view keeps {kept}")

    if kind in (Kind.HEATMAP, Kind.IMAGE):
        rbudget = {"max_extent": _budget_int(budget, "max_extent", DEFAULT_MAX_EXTENT,
                                             LIMIT_MAX_EXTENT)}
        if kind == Kind.HEATMAP:
            pool = budget.get("pool") or "mean"
            if pool not in POOL_OPS:
                _dv_bad(f"budget.pool {pool!r}; expected one of {list(POOL_OPS)}")
            rbudget["pool"] = pool
    elif kind == Kind.POINTS:
        rbudget = {"max_points": _budget_int(budget, "max_points", DEFAULT_MAX_POINTS,
                                             LIMIT_MAX_POINTS)}
    else:
        rbudget = {}
    out = {"source": src["name"], "index": rindex, "reduce": dict(reduce), "budget": rbudget}
    if tile is not None:
        out["tile"] = tile
    payload_layout(view, out)                    # the result must fit the hard limits
    if "max_extent" in rbudget:
        err = payload_frame_error(kind, rbudget["max_extent"], lambda m: payload_layout(
            view, dict(out, budget=dict(rbudget, max_extent=m)))["extent"])
        if err is not None:
            _dv_bad(err)
    return out


def payload_layout(view: dict, resolved: dict) -> dict:
    """How a resolved data view's payload maps onto the source.

    Returns ``{"extent": [...], "axes": [...], "tile": {...} or None}``:
    ``extent`` is the payload size along each kept axis (heatmap rows and
    columns, image height and width, the point count; a histogram has none),
    and each axis is ``{"dim", "start", "step", "count", "size"}``. Payload
    index i of an axis stands for the selected positions
    [floor(i * count / size), ceil((i + 1) * count / size)); position p is
    source index start + p * step. A pooled axis aggregates those positions,
    a sampled axis (points) takes the first."""
    kind = view.get("kind")
    if kind in (Kind.SERIES, Kind.HISTOGRAM, Kind.TEXT):
        return {"extent": [], "axes": [], "tile": None}
    src = next(s for s in view["sources"] if s["name"] == resolved["source"])
    index, reduce, tile = resolved["index"], resolved["reduce"], resolved.get("tile")
    axes = {}
    for d in src["dims"]:
        name, size = d["name"], d["size"]
        sel = index.get(name)
        if _is_int(sel) or name in reduce:
            continue
        start, stop, step = sel if sel is not None else (0, size, 1)
        axes[name] = {"dim": name, "start": start, "step": step,
                      "count": len(range(start, stop, step))}
    budget = resolved["budget"]
    tinfo = None
    if kind == Kind.POINTS:
        (a,) = axes.values()
        a["size"] = min(a["count"], budget["max_points"])
        order = [a]
    elif kind == Kind.BARS:                                                    # 3.2 draft
        (a,) = axes.values()
        if not 1 <= a["count"] <= MAX_BARS:
            _dv_bad(f"a bar chart shows 1 to {MAX_BARS} bars, the data view selects {a['count']}")
        a["size"] = a["count"]
        order = [a]
    else:
        ext = budget["max_extent"]
        if tile is not None:
            t = axes.pop(tile)
            t["size"] = t["count"]
            cols = math.isqrt(t["count"] - 1) + 1 if t["count"] > 0 else 1
            rows = -(-t["count"] // cols)
            r, c = axes.values()
            r["size"] = min(r["count"], max(1, ext // rows))
            c["size"] = min(c["count"], max(1, ext // cols))
            tinfo = {"dim": tile, "rows": rows, "cols": cols}
            order = [t, r, c]
            extent = [rows * r["size"], cols * c["size"]]
            if max(extent) > LIMIT_MAX_EXTENT:
                _dv_bad(f"a mosaic of {extent} exceeds {LIMIT_MAX_EXTENT}")
            return {"extent": extent, "axes": order, "tile": tinfo}
        r, c = axes.values()
        r["size"] = min(r["count"], ext)
        c["size"] = min(c["count"], ext)
        order = [r, c]
    return {"extent": [a["size"] for a in order], "axes": order, "tile": tinfo}
