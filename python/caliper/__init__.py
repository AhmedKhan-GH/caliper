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
            if ring.take(name, plan, now):
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
         _raw: bool = False):
    """Show/update view ``name`` with tensor ``t``. Returns at once.

    * 2-D -> heatmap (float32; autoscaled on the GPU unless vmin/vmax given)
    * (C,H,W) with C in (3,4) / (H,W,3|4) -> RGBA image (floats clamped to [0,1])
    * (C,H,W) other C -> channel mosaic heatmap;  (N,C,H,W) -> first item
    ``rate``: snapshots per second for this view (default 60, 0 = every call;
    the offscreen test mode defaults to every call).

    Returns True if a snapshot was taken, False when the call was rate limited,
    the ring was full or nothing is attached; a :class:`ShowStats` in the
    synchronous offscreen mode."""
    _name_ok(name)
    cm = _cmap(cmap)
    if vmin is not None or vmax is not None:
        check_range(vmin, vmax)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    return _dispatch(KIND_HEATMAP, name, rate, plan_show, t, cm, vmin, vmax, _raw)


def _points_plan(z, labels, classes, cm, rng):
    return plan_points(z, labels, classes, cm, rng[0], rng[1])


def points(z, labels=None, name: str = "points", classes=None, cmap="viridis", vmin=None,
           vmax=None, rate=None):
    """Scatter of ``z`` (N, 2) or (N, 3) (2-D: pan/zoom, 3-D: orbit), coloured by
    ``labels`` (N,): categorical with ``classes=K`` (labels 0..K-1), otherwise
    colormapped over their range. Same return values as :func:`show`."""
    _name_ok(name)
    cm = _cmap(cmap)
    if vmin is not None or vmax is not None:
        check_range(vmin, vmax)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    return _dispatch(KIND_POINTS, name, rate, _points_plan, z, labels, classes, cm, (vmin, vmax))


def _hist_plan(t, bins, _c, _d, _e):
    return plan_histogram(t, bins)


def histogram(t, name: str = "histogram", bins: int = 64, rate=None):
    """Histogram of every element of ``t`` over its [min, max], binned on the
    tensor's device. Same return values as :func:`show`."""
    _name_ok(name)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    return _dispatch(KIND_HISTOGRAM, name, rate, _hist_plan, t, bins, None, None, None)


def series(value, name: str = "series", rate=None):
    """Append ``value`` (a float, a 0-d tensor or a 1-D tensor of values) to the
    series ``name`` (a ring of the last 4096 values, on the value's device).
    Every call appends; the view is updated at most ``rate`` times per second."""
    _name_ok(name)
    for r in _routes or _auto_routes():
        if r.transport.attached():
            break
    else:
        return False
    now = time.perf_counter()
    taken = False
    sync = None
    with LOCK:
        sp = None
        for r in _routes:
            if not r.transport.attached():
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
          include=None, rate=None):
    """Show a module's activations (forward hooks, channel mosaics) and/or its
    parameter gradients (heatmaps) as views named ``<module>.act`` and
    ``<module>.<param>.grad``. Returns a handle; ``handle.remove()`` removes the hooks.
    ``include``: module names / glob patterns, or a predicate(name, module)."""
    from ._watch import watch as _w
    return _w(model, activations=activations, gradients=gradients, every=every,
              include=include, rate=rate)


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
        return sum(r.ring.flush(timeout=timeout) for r in _routes if r.transport.attached())


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
