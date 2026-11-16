"""caliper: live tensor views from a normal PyTorch script.

    import caliper
    for step in range(steps):
        ...train...
        caliper.show(model.fc1.weight, name="fc1.weight")   # heatmap, returns at once
        caliper.points(z, labels=y, classes=10)              # embedding scatter
        caliper.series(loss)                                 # loss curve
        caliper.histogram(model.fc1.weight.grad)             # value distribution
    caliper.spin()                                           # optional: keep it on screen

ONE set of calls, routed by context (training code never changes):

* Attach to the app (primary). The script's first call opens a local live
  session (shared memory + a session file, caliper.live). The Caliper app or
  caliper-viewer attaches to it; with no viewer attached a call returns after
  a check, no snapshot, no GPU work.
* Own window. ``caliper.window()`` opens a Caliper window in this process (a
  render thread repaints it); ``caliper.run(train)`` runs ``train`` on a worker
  thread so the main thread can keep the window interactive (macOS delivers
  input only to the main thread). ``caliper.offscreen()`` is the synchronous
  test mode (every call applied and readable at once).
* Every active transport receives each call; with none attached a call is a
  near no-op.

A snapshot is a few queued torch ops into a slot on the tensor's own device
(rate limited per view, default 60 Hz): MPS tensors are never read back to the
host, nothing synchronizes, the training loop never waits for the GPU or the
viewer.

Importing this package does not touch GLFW, AppKit, Metal or torch; the native
extension loads on first use.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass

from ._convert import (CMAPS, SERIES_CAPACITY, check_range, cmap_index, plan_histogram,
                       plan_points, plan_series, plan_show)
from . import _dataview
from ._dataview import DataViewError
from ._ring import (DEFAULT_RATE_HZ, KIND_HEATMAP, KIND_HISTOGRAM, KIND_IMAGE, KIND_POINTS,
                    KIND_SERIES, LOCK, Publisher, SnapshotRing, rate_interval)

__all__ = [
    "show", "points", "series", "histogram", "watch", "window", "offscreen", "session", "run",
    "spin", "flush", "close", "remove", "stats", "render", "frame", "read_pixels",
    "view_info", "view_readback", "view_remove", "view_count", "colormap_lut", "log_lines",
    "live_view_info", "live_view_values", "live_view_readback", "live_pause", "live_paused",
    "ShowStats", "CMAPS", "SERIES_CAPACITY", "DEFAULT_RATE_HZ",
]
__version__ = "0.1.0"

_KIND_NAMES = {KIND_HEATMAP: "heatmap", KIND_IMAGE: "image", KIND_POINTS: "points",
               KIND_SERIES: "series", KIND_HISTOGRAM: "histogram"}
_CMAP_FAST = dict(CMAPS)
_MAIN_IDENT = threading.main_thread().ident
_native_mod = None


def _load_native():
    global _native_mod
    if _native_mod is None:
        from . import _native as n
        _native_mod = n
    return _native_mod


def __getattr__(name):
    if name == "Scene":                          # scenes: imported on first use
        from ._scene import Scene
        return Scene
    if name in ("HAVE_VIEWS", "HAVE_LIVE_VIEWS", "VIEW_FLAG_RAW", "VIEW_FLAG_WAIT_GPU",
                "EMBED_API_VERSION", "HEATMAP_IMAGE_VIA_LIVE_APPLY"):
        return getattr(_load_native(), name)
    raise AttributeError(f"module 'caliper' has no attribute {name!r}")


def _check_main_thread(what: str) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            f"caliper.{what}() must be called from the main thread "
            "(macOS AppKit/GLFW windows are main-thread only)")


@dataclass
class ShowStats:
    """What a call returned in the synchronous offscreen mode (always truthy)."""
    name: str
    kind: str
    path: str          # renderer path: "compute-imported" / "blit-imported" / "points-imported" ...
    imported: bool     # True = the core read the slot's MTLBuffer in place
    total_ms: float

    def __bool__(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# module state: the active routes (transport + its ring)
# ---------------------------------------------------------------------------
class _Route:
    """A transport and the ring that feeds it (session and test transports;
    the in-process Window is its own route)."""
    sync = False

    def __init__(self, transport, rate: float = DEFAULT_RATE_HZ):
        self.transport = transport
        self.ring = SnapshotRing(transport, rate, on_pending=_publisher.kick)
        self.closed = False

    def close(self) -> None:
        with LOCK:
            if self.closed:
                return
            self.closed = True
            try:
                self.ring.flush(timeout=1.0)
            except Exception:
                pass
            self.ring.close()
        close_fn = getattr(self.transport, "close", None)
        if close_fn is not None:
            close_fn()


_publisher = Publisher(LOCK)
_routes: tuple = ()          # read without the lock on the hot path (atomic swap)
_window = None               # the in-process Window (window or offscreen), or None
_session_route: _Route | None = None
_extra_routes: list = []     # _attach_transport() (tests, custom transports)
_auto_tried = False


def _set_routes() -> None:
    global _routes
    rs = []
    if _session_route is not None and not _session_route.closed:
        rs.append(_session_route)
    if _window is not None and not _window.closed:
        rs.append(_window)
    rs.extend(r for r in _extra_routes if not r.closed)
    _publisher.rings = [r.ring for r in rs if not r.sync]
    _routes = tuple(rs)


def _auto_routes() -> tuple:
    """First call with no transport: open the live session (attach to app)."""
    global _auto_tried
    if _auto_tried:
        return _routes
    with LOCK:
        if _auto_tried:
            return _routes
        _auto_tried = True
        if os.environ.get("CALIPER_SESSION", "1") == "0":
            return _routes
        try:
            session()
        except _SessionUnavailable as e:
            warnings.warn(f"caliper: {e}; calls do nothing until caliper.window() or "
                          "caliper.offscreen() is opened", RuntimeWarning, stacklevel=3)
    return _routes


def _attach_transport(transport, rate: float = DEFAULT_RATE_HZ) -> _Route:
    """Route every call to ``transport`` too (implements caliper._ring.Transport)."""
    with LOCK:
        r = _Route(transport, rate)
        _extra_routes.append(r)
        _set_routes()
        return r


def _detach_transport(route: _Route) -> None:
    with LOCK:
        if route in _extra_routes:
            _extra_routes.remove(route)
        _set_routes()
    route.close()


def _sync_result(w, name: str, kind: int, t0: float):
    path, imported = "", False
    try:
        if kind in (KIND_HEATMAP, KIND_IMAGE) and not w.core.heatmap_live_apply:
            info = w.core.view_info(name)
        else:
            info = w.core.live_view_info(name)
        path, imported = info["path"], bool(info["imported"])
    except RuntimeError:
        pass
    return ShowStats(name, _KIND_NAMES.get(kind, "?"), path, imported,
                     (time.perf_counter() - t0) * 1e3)


def _after(w) -> None:
    """Main-thread window pump (60 Hz) when a window is open outside run()."""
    if w is not None and threading.get_ident() == _MAIN_IDENT and w.pump_due():
        w.pump()


def _dispatch(kind, name, rate, make, a, b, c, d, e):
    """Offer one snapshot to every attached route (the module lock held)."""
    now = time.perf_counter()
    taken = False
    sync = None
    with LOCK:
        plan = None
        for r in _routes:
            if not r.transport.attached():
                continue
            ring = r.ring
            iv = ring.default_interval if rate is None else rate_interval(rate)
            if not ring.due(name, iv, now):
                continue
            if plan is None:
                plan = make(a, b, c, d, e)
                if plan is None:                 # a data view source without a tensor yet
                    break
            take_call = getattr(ring, "take_call", None)   # the serve() route also wants the call
            if (take_call(name, plan, now, (a, b, c, d, e)) if take_call is not None
                    else ring.take(name, plan, now)):
                taken = True
                if r.sync:
                    sync = r
        if sync is not None:
            sync.complete(name)
            res = _sync_result(sync, name, plan.kind, now)
    _after(_window)
    return res if sync is not None else taken


def _name_ok(name) -> None:
    if not isinstance(name, str) or not name or "##" in name:
        raise ValueError(f"invalid view name {name!r} (non-empty str, no '##')")


def _cmap(cmap) -> int:
    cm = _CMAP_FAST.get(cmap) if cmap.__class__ is str else None
    return cmap_index(cmap) if cm is None else cm


# ---------------------------------------------------------------------------
# the calls
# ---------------------------------------------------------------------------
def show(t, name: str = "tensor", cmap="viridis", vmin=None, vmax=None, rate=None, *,
         dims=None, data_view=None, items=None, _raw: bool = False):
    """Show/update view ``name`` with tensor ``t``. Returns at once.

    * 2-D -> heatmap (float32; autoscaled on the GPU unless vmin/vmax given)
    * (C,H,W) with C in (3,4) / (H,W,3|4) -> RGBA image (floats clamped to [0,1])
    * (C,H,W) other C -> channel mosaic heatmap;  (N,C,H,W) -> first item
    ``rate``: snapshots per second for this view (default 60, 0 = every call;
    the offscreen test mode defaults to every call).
    ``dims``: one name per dimension of t (for an image, of all but the RGB(A)
    one); the view then offers t as a source that a viewer can index, reduce
    and tile, and shows its data view instead of the rules above.
    ``data_view``: the view's default data view (see :func:`set_data_view`).
    ``items``: the item space of the rows (a str), or {dim: space}, for
    linked selection (see :func:`items`).

    Returns True if a snapshot was taken, False when the call was rate limited,
    the ring was full or nothing is attached; a :class:`ShowStats` in the
    synchronous offscreen mode."""
    _name_ok(name)
    if items is not None:
        _CTL.view_items_call(name, "heatmap", items)
    cm = _cmap(cmap)
    if vmin is not None or vmax is not None:
        check_range(vmin, vmax)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    if dims is not None or data_view is not None:
        return _dispatch(KIND_HEATMAP, name, rate, _dv_show_plan, t, cm, vmin, vmax,
                         (_raw, name, dims, data_view))
    return _dispatch(KIND_HEATMAP, name, rate, plan_show, t, cm, vmin, vmax, _raw)


def _points_plan(z, labels, classes, cm, rng):
    return plan_points(z, labels, classes, cm, rng[0], rng[1])


def points(z, labels=None, name: str = "points", classes=None, cmap="viridis", vmin=None,
           vmax=None, rate=None, *, dims=None, data_view=None, items=None):
    """Scatter of ``z`` (N, 2) or (N, 3) (2-D: pan/zoom, 3-D: orbit), coloured by
    ``labels`` (N,): categorical with ``classes=K`` (labels 0..K-1), otherwise
    colormapped over their range. ``dims``: one name per dimension of z but the
    coordinates, ending with the point dimension, which makes z a source of
    data views (labels follow the kept dimension, and above the budget points
    and labels are sampled with a stride); ``data_view``: the default data
    view; ``items``: the item space of the points (a str), or {dim: space},
    for linked selection (see :func:`items`). Same return values as
    :func:`show`."""
    _name_ok(name)
    if items is not None:
        _CTL.view_items_call(name, "points", items)
    cm = _cmap(cmap)
    if vmin is not None or vmax is not None:
        check_range(vmin, vmax)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    if dims is not None or data_view is not None:
        return _dispatch(KIND_POINTS, name, rate, _dv_points_plan, z, labels, classes, cm,
                         (vmin, vmax, name, dims, data_view))
    return _dispatch(KIND_POINTS, name, rate, _points_plan, z, labels, classes, cm, (vmin, vmax))


def _hist_plan(t, bins, _c, _d, _e):
    return plan_histogram(t, bins)


def histogram(t, name: str = "histogram", bins: int = 64, rate=None, *, dims=None,
              data_view=None):
    """Histogram of every element of ``t`` over its [min, max], binned on the
    tensor's device. ``dims`` and ``data_view`` as for :func:`show`; the
    histogram bins the data view. Same return values as :func:`show`."""
    _name_ok(name)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    if dims is not None or data_view is not None:
        return _dispatch(KIND_HISTOGRAM, name, rate, _dv_hist_plan, t, bins, name, dims,
                         data_view)
    return _dispatch(KIND_HISTOGRAM, name, rate, _hist_plan, t, bins, None, None, None)


def series(value, name: str = "series", rate=None, *, items=None):
    """Append ``value`` (a float, a 0-d tensor or a 1-D tensor of values) to the
    series ``name`` (a ring of the last 4096 values, on the value's device).
    Every call appends; the view is updated at most ``rate`` times per second.
    ``items``: the item space that the series' index numbers, for linked
    selection (see :func:`items`)."""
    _name_ok(name)
    if items is not None:
        _CTL.view_items_call(name, "series", items)
    tap = _series_tap               # caliper.serve() keeps every value for late viewers
    tapped = tap is not None and tap(name, value)
    for r in _routes or _auto_routes():
        if r.transport.attached() and not getattr(r, "taps_series", False):
            break
    else:
        return tapped
    now = time.perf_counter()
    taken = tapped
    sync = None
    with LOCK:
        sp = None
        for r in _routes:
            if not r.transport.attached() or getattr(r, "taps_series", False):
                continue
            if sp is None:
                sp = plan_series(value)
            ring = r.ring
            iv = ring.default_interval if rate is None else rate_interval(rate)
            if ring.append(name, sp, now, iv):
                taken = True
                if r.sync:
                    sync = r
        if sync is not None:
            sync.complete(name)
            res = _sync_result(sync, name, KIND_SERIES, now)
    _after(_window)
    return res if sync is not None else taken


def watch(model, activations: bool = True, gradients: bool = False, every: int = 1,
          include=None, rate=None, groups=True):
    """Show a module's activations (forward hooks, channel mosaics) and/or its
    parameter gradients (heatmaps) as views named ``<module>.act`` and
    ``<module>.<param>.grad``. Returns a handle; ``handle.remove()`` removes the hooks.
    ``include``: module names / glob patterns, or a predicate(name, module).
    ``groups``: also offer the grouped views ``weights``, ``activations`` and
    ``gradients``, one heatmap each whose data view picks the module or
    parameter (a str prefixes their names with ``<str>.``; False: none)."""
    from ._watch import watch as _w
    return _w(model, activations=activations, gradients=gradients, every=every,
              include=include, rate=rate, groups=groups)


# ---------------------------------------------------------------------------
# data views: which data travels (a slice, selection, projection or
# aggregation of a tensor, computed on its device; see caliper._dataview)
# ---------------------------------------------------------------------------
__all__ += ["offer", "update", "data_view", "set_data_view", "DataViewError"]

_DV_KINDS = {"heatmap": KIND_HEATMAP, "image": KIND_IMAGE, "points": KIND_POINTS,
             "histogram": KIND_HISTOGRAM}
_DV_OPTIONS = {"heatmap": ("cmap", "vmin", "vmax"), "image": (),
               "points": ("labels", "classes", "cmap", "vmin", "vmax"), "histogram": ("bins",)}


def offer(name: str, sources, kind: str = "heatmap", dims=None, *, data_view=None, rate=None,
          items=None, **options):
    """Offer ``sources`` as the sources of view ``name``; a viewer, or
    :func:`set_data_view`, picks one of them and a data view of it.

    ``sources`` maps source names to tensors (read at each snapshot, so a tensor
    updated in place stays current) or to callables that return the current
    tensor or None (called under ``torch.no_grad()``: once now to learn the
    shape, then only when the source is selected and a snapshot is due). A
    single tensor or callable is one source named ``name``. ``dims`` names the
    dimensions: one tuple for every source, or a dict of tuples per source
    (default ``dim0``, ``dim1``, ...); an image source has one more dimension,
    RGB(A), and a points source a last one with 2 or 3 coordinates, which are
    not named. ``kind`` is "heatmap", "image", "points" or "histogram".
    ``data_view`` is the view's default data view. ``options`` are the matching
    call's arguments: ``cmap``, ``vmin`` and ``vmax`` (heatmap, points),
    ``labels`` (a tensor or callable, along the kept dimension) and ``classes``
    (points), ``bins`` (histogram). ``items``: the item space of the item axis
    (a heatmap's or image's first dim, the point dim), or {dim: space}, for
    linked selection (see :func:`items`).

    The view takes snapshots when the training loop calls :func:`update`.
    Replaces an earlier view of the same name. Returns the default data view,
    resolved, or None while no source has a tensor."""
    _name_ok(name)
    kind = _dataview._kind(kind)
    if kind == "series":
        raise ValueError("a series view has no sources; use caliper.series")
    if items is not None:
        _CTL.view_items_call(name, kind, items)
    opts = _dv_options(kind, options)
    _dataview.REGISTRY.offer_view(name, sources, kind, dims, default=data_view, rate=rate,
                                  options=opts)
    return _dataview.REGISTRY.resolved(name)


def update(name: str | None = None):
    """Take a snapshot of every view registered with :func:`offer` (and the
    ``weights`` view of :func:`watch`) that is due, or of view ``name`` only:
    of the source each route needs (the default data view's for the window and
    the session, the subscriptions' for a server). Same return values as
    :func:`show`; for every view, True if any took one."""
    entries = None
    if name is not None:
        e = _dataview.REGISTRY.get(name)
        if e is None:
            raise ValueError(f"no view {name!r} was offered")
        entries = [e]
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    taken = False
    for e in entries or _dataview.REGISTRY.fetchable():
        for src, s in list(e.sources.items()):
            if s.get is not None:
                got = _dv_dispatch(e, src)
                if name is not None and got.__class__ is not bool:
                    return got                       # a ShowStats in the offscreen mode
                taken = bool(got) or taken
    return taken


def data_view(name: str):
    """The default data view of view ``name``, resolved: a dict with
    ``source``, ``index``, ``reduce`` and ``budget``, and ``tile`` for a tiled
    heatmap. None when the view offers no data views or has no data yet."""
    return _dataview.REGISTRY.resolved(name)


def set_data_view(name: str, request=None):
    """Set the default data view of view ``name`` (any thread; used from the
    next snapshot). ``request`` is a dict, or its JSON text, with any of
    ``source``, ``index``, ``reduce``, ``tile`` and ``budget``. A member it
    names replaces the current one as a whole (``budget`` key by key), and the
    others stay, so ``{}`` changes nothing; None returns to the default the view
    was registered with. Returns the resolved data view; raises
    :class:`DataViewError` when it does not fit the view, which keeps the data
    view it had. A request for a view that does not exist yet, or has no data
    yet, is kept and checked at its first snapshot; then None is returned."""
    return _dataview.REGISTRY.request(name, request)


def _dv_options(kind, options):
    allowed = _DV_OPTIONS[kind]
    for k in options:
        if k not in allowed:
            raise TypeError(f"offer(kind={kind!r}) got an unexpected option {k!r}")
    o = dict(options)
    if kind in ("heatmap", "points"):
        o["cmap"] = _cmap(o.get("cmap", "viridis"))
    if o.get("vmin") is not None or o.get("vmax") is not None:
        check_range(o.get("vmin"), o.get("vmax"))
    if "bins" in o and not 1 <= int(o["bins"]) <= 1 << 16:
        raise ValueError("histogram: bins must be in [1, 65536]")
    return o


def _dv_call_plan(call):
    """make() of a call with dims or data_view: the lazy plan of the view's
    default data view (no device work until a route writes it)."""
    p = _dataview.REGISTRY.default_program(call.entry, call.x)
    return None if p is None else _dataview.lazy_plan(call, p)


def _dv_show_plan(t, cm, vmin, vmax, extra):
    raw, name, dims, dv = extra
    e, x = _dataview.REGISTRY.for_call(name, "show", dims, dv, t)
    return _dv_call_plan(_dataview.DataCall(e, name, x, {"cmap": cm, "vmin": vmin, "vmax": vmax,
                                                          "raw": raw}))


def _dv_points_plan(z, labels, classes, cm, extra):
    vmin, vmax, name, dims, dv = extra
    e, x = _dataview.REGISTRY.for_call(name, "points", dims, dv, z)
    return _dv_call_plan(_dataview.DataCall(e, name, x, {"labels": labels, "classes": classes,
                                                          "cmap": cm, "vmin": vmin, "vmax": vmax}))


def _dv_hist_plan(t, bins, name, dims, dv):
    e, x = _dataview.REGISTRY.for_call(name, "histogram", dims, dv, t)
    return _dv_call_plan(_dataview.DataCall(e, name, x, {"bins": bins}))


def _dv_dispatch(entry, source, tensor=None):
    """Offer source ``source`` of registry view ``entry`` (offer(), watch's
    groups) to every attached route that is due and needs it: the window and
    the session draw the view's default data view, so only its source; a
    server needs the sources its subscriptions select (``ring.wants``). The
    tensor of a registered source is fetched only when a route needs it."""
    now = time.perf_counter()
    taken = False
    sync = None
    name = entry.name
    with LOCK:
        call = plan = net = None
        for r in _routes:
            if not r.transport.attached():
                continue
            ring = r.ring
            take_call = getattr(ring, "take_call", None)
            if take_call is None:
                if entry.state[2] != source:
                    continue
            elif not ring.wants(name, source):
                continue
            iv = ring.default_interval if entry.rate is None else rate_interval(entry.rate)
            if not ring.due(name, iv, now):
                continue
            if call is None:
                call = _dataview.DataCall(entry, source, tensor, entry.options)
            if take_call is None:
                if plan is None:
                    plan = _dataview.REGISTRY.default_plan(call)
                    if plan is None:
                        continue
                ok = ring.take(name, plan, now)
            else:
                if net is None:
                    net = plan if plan is not None else _dataview.net_plan(call)
                ok = take_call(name, net, now, None)
            if ok:
                taken = True
                if r.sync:
                    sync = r
        if sync is not None:
            sync.complete(name)
            res = _sync_result(sync, name, plan.kind, now)
    _after(_window)
    return res if sync is not None else taken


def _dv_hook(entry, source, t):
    """watch's hooks: ``t`` (None: fetch it) is source ``source`` of the grouped
    view ``entry``. A source no route needs records its shape (metadata) and
    returns; nothing happens while nothing is attached."""
    for r in _routes:
        if r.transport.attached():
            break
    else:
        return False
    if t is not None:
        _dataview.REGISTRY.observe(entry, source, t)
    return _dv_dispatch(entry, source, t)


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------
class _SessionUnavailable(RuntimeError):
    pass


def _script_name() -> str:
    arg0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    return os.path.basename(arg0) or "python"


def session(auto_launch: bool = False, viewer_path: str | None = None,
            script_name: str | None = None):
    """Open (or return) the live session other processes attach to (the
    Caliper app, caliper-viewer). The first call of a script opens it
    implicitly with ``auto_launch=False``; ``auto_launch=True`` also starts a
    viewer. Returns the SessionTransport."""
    global _session_route
    with LOCK:
        if _session_route is not None and not _session_route.closed:
            tr = _session_route.transport
            if auto_launch and not tr.viewer_alive():
                tr.launch_viewer()
            return tr
        try:
            from .live import SessionTransport
        except ImportError as e:
            raise _SessionUnavailable(
                "the live session transport is not available in this build "
                f"(caliper.live.SessionTransport is missing: {e})") from None
        tr = SessionTransport.open(auto_launch=auto_launch, viewer_path=viewer_path,
                                   script_name=script_name or _script_name())
        _session_route = _Route(tr, DEFAULT_RATE_HZ)
        _set_routes()
        return tr


def window(width: int = 1280, height: int = 800, title: str = "caliper", *,
           offscreen: bool = False, fps: float = 60.0, rate=None, activate: bool = True,
           heartbeat: bool = False):
    """Open the in-process Caliper window (main thread only) and route every
    call to it. ``offscreen=True`` renders on an offscreen canvas (same render
    thread, no window: benchmarks, CI, :func:`render`). Returns the Window
    (``.run(fn)``, ``.stats()``, ``.close()``); an open one is returned as is.

    Limits: input waits during long steps unless :func:`run` gives the main
    thread to Caliper; a crash of the script takes the window down too."""
    _check_main_thread("window")
    global _window
    from ._inproc import Window
    with LOCK:
        w = _window
        if w is not None and not w.closed:
            if not w.sync and w.offscreen == offscreen:
                return w
            w.close()
        _window = Window(width, height, title, offscreen=offscreen, fps=fps, rate=rate,
                         activate=activate, heartbeat=heartbeat, on_pending=_publisher.kick)
        _set_routes()
        return _window


def offscreen(width: int = 256, height: int = 256):
    """Synchronous offscreen canvas (tests, notebooks): every call waits for its
    snapshot, applies it and is readable at once (view_info, view_readback,
    read_pixels, render). Replaces an open window (one core per process)."""
    _check_main_thread("offscreen")
    global _window
    from ._inproc import Window
    with LOCK:
        if _window is not None:
            _window.close()
            _window = None
        _window = Window(width, height, offscreen=True, sync=True)
        _set_routes()
        return _window


def run(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)``. With a window open, fn runs on a worker
    thread and the main thread keeps the window interactive; otherwise fn just
    runs. Returns fn's result."""
    w = _window
    if w is not None and not w.closed and not w.offscreen:
        return w.run(fn, *args, **kwargs)
    return fn(*args, **kwargs)


def flush(timeout: float = 5.0) -> int:
    """Publish every pending snapshot (polls their GPU events; not for the
    training loop). Returns how many are still pending after ``timeout``."""
    with LOCK:
        return sum(r.ring.flush(timeout=timeout) for r in _routes if _has_viewer(r.transport))


def _has_viewer(transport) -> bool:
    """A viewer is attached to the route. The network transport's attached()
    also says whether its next tick is due, which a flush must not wait for:
    right after a step's view calls it is not, and the step's gathered series
    values would wait for the next call."""
    n = getattr(transport, "n_attached", None)
    return n > 0 if n is not None else transport.attached()


def spin(seconds: float | None = None) -> None:
    """Flush pending snapshots, then keep the view alive: with a window, run
    its event loop until it is closed (or ``seconds``); with a session, wait
    for the viewer to exit (or ``seconds``)."""
    flush()
    w = _window
    if w is not None and not w.closed and not w.sync:
        w.loop(seconds)
        return
    s = _session_route
    if s is not None and not s.closed:
        s.transport.wait_viewer(timeout=seconds)
        return
    if w is not None and not w.closed:
        w.core.frame()
    elif seconds:
        time.sleep(seconds)


def close() -> None:
    """Close every transport (window, offscreen canvas, session). A later call
    opens a new session again."""
    global _window, _session_route, _auto_tried
    with LOCK:
        w, _window = _window, None
        s, _session_route = _session_route, None
        extras = list(_extra_routes)
        _extra_routes.clear()
        _auto_tried = False
        _set_routes()
    if w is not None:
        w.close()
    if s is not None:
        s.close()
    for r in extras:
        r.close()


def remove(name: str) -> bool:
    """Remove view ``name`` from every transport. Returns True if the in-process
    core had it."""
    found = False
    _dataview.REGISTRY.remove(name)
    sc = sys.modules.get(__name__ + "._scene")     # scenes, when a script made one
    if sc is not None:
        sc.forget(name)
    _CTL.forget_view(name)
    with LOCK:
        for r in _routes:
            r.ring.forget(name)
        w = _window
        if w is not None and not w.closed:
            try:
                found = w.core.live_view_remove(name) or found
            except RuntimeError:
                pass
            try:
                w.core.view_remove(name)
                found = True
            except RuntimeError:
                pass
    return found


def stats() -> dict:
    """Ring counters per route plus the window's render stats."""
    with LOCK:
        out = {}
        for r in _routes:
            key = "window" if r is _window else ("session" if r is _session_route else repr(r))
            d = {"ring": dict(r.ring.counters)}
            if r is _window and not r.closed:
                d.update(r.stats())
            elif hasattr(r.transport, "stats"):
                d["transport"] = r.transport.stats()
            out[key] = d
        return out


# ---------------------------------------------------------------------------
# readback and compatibility (the in-process core)
# ---------------------------------------------------------------------------
def _require_window(what: str):
    w = _window
    if w is None or w.closed:
        raise RuntimeError(f"caliper.{what}(): no live core (call caliper.offscreen() or "
                           "caliper.window() first)")
    return w


def frame() -> float:
    """Render one frame now. Returns its wall time in ms."""
    w = _require_window("frame")
    if not w.offscreen:
        _check_main_thread("frame")
    w._last_frame = time.perf_counter()
    return w.core.frame()


def render():
    """(H, W, 4) uint8 of a freshly composited frame (offscreen canvases)."""
    return _require_window("render").render()


def read_pixels():
    """(H, W, 4) uint8 copy of the last composited frame (offscreen canvases)."""
    return _require_window("read_pixels").core.read_pixels()


def view_info(name: str) -> dict:
    """The embed v1.2 view of a heatmap/image (path, size, updates, creates)."""
    return _require_window("view_info").core.view_info(name)


def view_readback(name: str):
    """(H, W, 4) uint8 readback of a heatmap/image view's texture."""
    return _require_window("view_readback").core.view_readback(name)


def view_remove(name: str) -> None:
    remove(name)


def view_count() -> int:
    w = _window
    return w.core.view_count() if w is not None and not w.closed else 0


def live_view_info(name: str) -> dict:
    """The typed live view (caliper_core_live_view_info): kind, path, updates,
    latency_ms, read_done, count..."""
    return _require_window("live_view_info").core.live_view_info(name)


def live_view_values(name: str):
    """Series values (oldest first) or histogram counts the view holds."""
    return _require_window("live_view_values").core.live_view_values(name)


def live_view_readback(name: str):
    """(H, W, 4) uint8 of a texture-backed typed view (heatmap, image, points)."""
    return _require_window("live_view_readback").core.live_view_readback(name)


def live_pause(paused: bool = True) -> None:
    """Freeze the display (snapshots keep flowing and are dropped)."""
    _require_window("live_pause").core.live_set_paused(paused)


def live_paused() -> bool:
    w = _window
    return bool(w.core.live_paused()) if w is not None and not w.closed else False


def colormap_lut(name="viridis"):
    """The core's 256-entry LUT as uint32 (R | G<<8 | B<<16 | A<<24)."""
    return _load_native().colormap_lut(cmap_index(name))


def log_lines() -> list[str]:
    w = _window
    return w.core.log_lines() if w is not None and not w.closed else []


def _native_core():
    """The live ``_native.Core`` (or None), for benchmarks and tests."""
    w = _window
    return w.core if w is not None and not w.closed else None


atexit.register(close)


# ---------------------------------------------------------------------------
# notebook widget (the notebook extra: pip install 'caliper[notebook]')
# ---------------------------------------------------------------------------
__all__ += ["widget"]


def widget(name: str, kind: str | None = None, *, height: int = 320):
    """A live view of the stream inline in a notebook cell (an anywidget fed by
    the stream socket, not by Jupyter's message channel). ``kind`` is "series",
    "heatmap", "image", "points", "histogram" or "scene"; None takes it from the stream.
    Starts :func:`serve` when no server runs. The widget's ``selection`` and
    ``data_view`` traits carry what the viewer picks back to Python."""
    from ._widget import widget as _widget
    return _widget(name, kind, height=height)


# ---------------------------------------------------------------------------
# network stream (caliper.serve: wire protocol v3 over a WebSocket on 127.0.0.1)
# ---------------------------------------------------------------------------
__all__ += ["serve"]
_server = None                      # the running caliper._net.NetServer (read by the widget)
_series_tap = None                  # its series tap while it runs


def serve(port: int = 0, *, control: bool = True):
    """Stream every view of this process to viewers on this machine: the
    browser page at the URL this prints (``http://127.0.0.1:<port>/?token=<token>``),
    the notebook widget and the Caliper app. ``port=0`` picks a free port.
    ``control=False`` makes every viewer read-only: none may pause the run,
    change a parameter or run an action (see :func:`step`).
    Returns the server (``url``, ``port``, ``token``, ``stream_url``,
    ``session``, ``control``, ``stats()``, ``on_event(fn)``, ``events()``,
    ``close()``; also a context manager); a second call returns the running
    server. Loopback only, with a session token. :func:`close` stops it."""
    from ._net import serve as _serve
    return _serve(port, control=control)


__all__ += ["connect"]


def connect(url: str | None = None):
    """Stream every view of this process to a `caliper serve` hub on this
    machine, as well as to serve()'s viewers if it runs. ``url`` is the hub's
    stream URL with its token (or the viewer link it prints); without it,
    $CALIPER_HUB_URL, then the hub file ($CALIPER_HUB_FILE, or hub.json in
    Caliper's app-data folder). Returns at once; the connection retries with
    back-off and never blocks training. Returns the connection (``url``,
    ``session``, ``state``, ``connected``, ``stats()``, ``close()``). Needs the
    native sender (caliper._netio). :func:`close` ends it."""
    from ._net import connect as _connect
    return _connect(url)


# ---------------------------------------------------------------------------
# controls of the running script and linked selection (wire protocol 3.1;
# caliper._controls holds the state and the seam its transports attach to)
# ---------------------------------------------------------------------------
__all__ += ["step", "pause", "param", "action", "items", "controls", "Param", "Action",
            "ItemSpace"]
from . import _controls                                        # noqa: E402
from ._controls import Action, ItemSpace, Param                # noqa: E402
_CTL = _controls.CONTROLS


def step(step=None) -> None:
    """The checkpoint of the training loop: call it once per step, on the
    training thread. It sets the parameter values that viewers changed, runs
    the actions they asked for, in order, and waits while a viewer has paused
    the run, until a viewer resumes or steps it, the server closes, or no
    connection that may control has been open for 10 s. Ctrl-C or a
    notebook's interrupt ends the wait. ``step``: this step's number, shown to
    viewers (default: one more than the last). With nothing waiting it only
    counts the step."""
    c = _CTL
    if step is None:
        c.n += 1
    else:
        c.n = step
    if c.busy:
        c.checkpoint()


def pause() -> bool:
    """Pause the run at the next :func:`step`, as a viewer's Pause button does;
    a viewer resumes or steps it. Returns False, with a warning, when no stream
    server runs, since no viewer could resume it."""
    return _CTL.pause()


def param(name: str, value, *, min=None, max=None, log=False, choices=None, help=None) -> Param:
    """A parameter that viewers can change while the script runs. The type
    comes from ``value``: a bool is a toggle, an int an integer, a float a
    number, and a str one of ``choices``. ``min`` and ``max`` bound a number
    or an integer (a viewer's value is clamped to them), ``log=True`` draws
    its slider on a log scale (it needs min > 0 and max), and ``help`` is a
    tooltip. Returns a :class:`Param`: read ``.value`` in the loop; a
    viewer's change reaches it at the next :func:`step`, or at once while the
    script has called no step yet. ``.set(v)`` changes it from the script and
    ``.remove()`` takes it away. Registering a name again replaces it."""
    return _CTL.param(name, value, min=min, max=max, log=log, choices=choices, help=help)


def action(name: str, fn=None, *, help=None):
    """Register ``fn``, a function without arguments, as an action that a
    viewer can run: it runs at the next :func:`step`, on the training thread,
    and the text of what it returns, or of the error it raises, goes back to
    the viewer that asked. Works as a decorator, ``@caliper.action("save
    checkpoint")``. Returns an :class:`Action`, which calls ``fn`` when called
    and has ``.remove()``. Registering a name again replaces it."""
    if fn is None:
        return lambda f: _CTL.action(name, f, help=help)
    return _CTL.action(name, fn, help=help)


def items(space: str, details=None) -> ItemSpace:
    """The item space ``space``: items numbered from 0, such as the samples
    of a test set, that the ``items=`` of several views index, so that a
    selection in one view highlights the same items in the others.
    ``details`` gives a viewer a few fields of the selected items: a mapping
    {field: sequence indexed by item id} (copied once, now), or a function
    ``fn(ids)`` that returns {field: values} or one dict per item. The
    function runs on the thread "caliper.details", never on the training
    thread, so it must only read data that the loop does not change in place.
    Returns an :class:`ItemSpace` with ``.selection`` (the item ids viewers
    selected, current even while a cell runs) and ``.select(ids)``."""
    return _CTL.items(space, details)


def controls(*, height: int = 160):
    """A notebook widget with the run controls, every parameter and every
    action of this script (the notebook extra). Starts :func:`serve` when no
    server runs."""
    from ._widget import controls as _controls_widget
    return _controls_widget(height=height)


# ---------------------------------------------------------------------------
# scenes (the scenes subset of wire protocol 3.2, draft; caliper._scene)
# ---------------------------------------------------------------------------
__all__ += ["scene", "Scene"]


def scene(name: str = "scene", *, space: str = "3d", up=None, units: str = "m", rate=None,
          data_view=None, view_camera=None) -> "Scene":
    """The scene ``name``, made on the first call: a tree of entities, named
    by entity paths such as "world/cloud", each holding a batch of points,
    lines and a mesh, in one 3-D (``space="3d"``) or 2-D space. ``up`` is the
    axis a 3-D view keeps upright when it opens ("+x" to "-z", default "+y"),
    or in 2-D the direction of y on screen ("-y" default, "+y"); ``units`` a
    note of at most 15 bytes shown with the view (default "m"); ``rate`` the
    default rate limit of each batch (default 60 per second, 0 every call).
    A second call returns the same scene and updates ``up``, ``units`` and
    ``rate``; a different ``space`` raises ValueError. Log to it with
    ``scene.points``, ``scene.lines`` and ``scene.mesh``. Scenes travel over
    :func:`serve` (the notebook widget with ``kind="scene"``, the browser page
    and a hub); with only the shared-memory session or the in-process window
    open, a scene call warns once and does nothing there. ``data_view`` and
    ``view_camera`` are not built yet and raise NotImplementedError. Scene
    is imported on first use, so importing caliper stays light."""
    from . import _scene
    return _scene.scene(name, space=space, up=up, units=units, rate=rate, data_view=data_view,
                        view_camera=view_camera)
