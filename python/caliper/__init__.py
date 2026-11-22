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
from typing import TYPE_CHECKING

# A process that the app's model runner started ends when the app ends, also when the app crashes
# before the script calls watch() or param(). Without CALIPER_RUN_APP_PID nothing of this loads.
if os.environ.get("CALIPER_RUN_APP_PID"):
    from . import _managed
    _managed.watch_parent()

from ._convert import (CMAPS, SERIES_CAPACITY, check_range, cmap_index, plan_histogram,
                       plan_points, plan_series, plan_show)
from . import _dataview
from ._dataview import DataViewError
from ._ring import (DEFAULT_RATE_HZ, KIND_HEATMAP, KIND_HISTOGRAM, KIND_IMAGE, KIND_POINTS,
                    KIND_SERIES, LOCK, Publisher, SnapshotRing, rate_interval)

if TYPE_CHECKING:
    # Scene is imported on first use (__getattr__ below); this names it for type checkers and the API reference.
    from ._scene import Scene

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
    """The result of a view call in the synchronous offscreen mode, always truthy.

    Attributes:
        name (str): The view id.
        kind (str): The kind of the view, such as `"heatmap"`.
        path (str): The renderer path, such as `"compute-imported"`.
        imported (bool): `True` when the core read the slot buffer in place.
        total_ms (float): The time of the call in milliseconds.
    """
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
    """Show or update view `name` with tensor `t` as a heatmap or an RGBA image.

    The call returns at once. With `dims` or `data_view`, the view shows a data view of `t`
    instead of the layouts below ([Data Views](data-view-calls.md#data-views)).

    | Input | Shown as |
    |---|---|
    | 0-D or 1-D | A 1xN heatmap strip |
    | 2-D `(H, W)` | Heatmap (cast to float32 in the copy kernel) |
    | `(H, W, 3)` or `(H, W, 4)` | RGBA image |
    | `(1, H, W)` | Heatmap |
    | `(3, H, W)` or `(4, H, W)` | RGBA image |
    | `(C, H, W)`, any other C | Channel mosaic heatmap: ceil(sqrt(C)) columns, empty cells NaN, range over all channels |
    | `(N, C, H, W)` | Item 0, then as above |
    | More than 4 dims | Leading dims stacked as channels, `(-1, H, W)` |

    A 3-D tensor whose last dimension is 3 or 4 is read as `(H, W, C)` if it is uint8 or its first
    dimension is not 1, 3 or 4. Otherwise it is read as `(C, H, W)`.

    Image values follow these rules: uint8 stays as it is, floating point is clamped to [0, 1] and
    scaled to 255, and other integers are clamped to 0..255. A missing alpha channel is 255.
    Booleans
    become float32 (uint8 for an image), and complex tensors become their magnitude. With autoscale,
    a non-finite range keeps the last finite one, and a flat range widens by +-0.5. NaN values draw
    as colormap index 0.

    Args:
        t (torch.Tensor): The tensor, in one of the layouts above.
        name (str): View id. The same name updates the view in place. Non-empty, with no `##`.
        cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"` (case-insensitive), or 0, 1 or 2.
        vmin (float | None): Fixed lower end of the colormap range. `None` takes that end from a GPU
            `aminmax` of the snapshot. Each end is independent, and both must be finite.
        vmax (float | None): Fixed upper end of the colormap range, as for `vmin`.
        rate (float | None): Snapshots per second for this view ([Rate Limit][rate-limit]).
        dims (Sequence[str] | None): One name for each dimension of `t`, or for an image all but
            the color one ([Data Views](data-view-calls.md#data-views)).
        data_view (dict | None): The default data view of the view, in the form of
            [`set_data_view`][caliper.set_data_view].
        items (str | dict | None): The item space of the rows, as a name or as `{dim: space}`, for
            linked selection. See [`items`][caliper.items].

    Returns:
        bool | ShowStats: `True` if a route took a snapshot. `False` when the call was rate
            limited, all slots were busy or nothing is attached. A [`ShowStats`][caliper.ShowStats]
            in the synchronous offscreen mode.

    Raises:
        ValueError: When `name` is empty or holds `##`, when `cmap` is unknown, when `vmin` or
            `vmax` is not finite, or when `rate` is negative or NaN.
        DataViewError: When `data_view` does not fit `t`.
    """
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
    """Show or update view `name` as a scatter of the points `z`.

    `z` is `(N, 2)` (pan and zoom) or `(N, 3)` (orbit). Any dtype is cast to float32 on its device,
    and N must be at least 1. The bounding box for auto-fit is computed on the device. With `dims`
    or `data_view`, `z` is a source of data views and `labels` follows its kept dimension. Above
    the point budget, points and labels are sampled with a stride. See
    [Data Views](data-view-calls.md#data-views).

    Args:
        z (torch.Tensor): The points, `(N, 2)` or `(N, 3)`. With `dims`, one more leading
            dimension per name.
        labels (torch.Tensor | None): Per-point labels or values, shape `(N,)`, on the device of `z`
            or on the CPU. CPU labels for MPS points are copied to the device without waiting for
            its queue, from a private copy. The caller may reuse its tensor at once. Nothing is
            copied the other way. `None` draws one flat color.
        name (str): View id. The same name updates the view in place.
        classes (int | None): K categorical labels 0..K-1, drawn with a fixed 16-color palette
            (label L gets color L mod 16). Needs `labels`.
        cmap (str | int): Without `classes`, labels are colormapped as for [`show`][caliper.show].
        vmin (float | None): Lower end of the label range. Without `classes`, `None` autoscales
            as for `show`.
        vmax (float | None): Upper end of the label range, as for `vmin`.
        rate (float | None): Snapshots per second for this view ([Rate Limit][rate-limit]).
        dims (Sequence[str] | None): One name for each dimension of `z` but the last, ending with
            the point dimension.
        data_view (dict | None): The default data view of the view.
        items (str | dict | None): The item space of the points, as a name or as `{dim: space}`,
            for linked selection. See [`items`][caliper.items].

    Returns:
        bool | ShowStats: As for [`show`][caliper.show].

    Raises:
        ValueError: When `z` has a wrong shape, when the label count is not N, when `labels` is
            on a device other than the device of `z` or the CPU, when `classes` is below 1, or when
            `classes` is given without `labels`.
        DataViewError: When `data_view` does not fit `z`.
    """
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
    """Show or update view `name` as the bin counts of every element of `t` over its [min, max].

    The tensor is flattened and cast to float32, and it needs at least one element. The range comes
    from a GPU `aminmax`. Constant data uses [v - 1, v + 1] as `torch.histc` does, and NaN values
    fall into bin 0.

    On MPS the binning runs on the device (`index_add_`) because `torch.histc` on MPS reads its
    range back to the host and drains the queue. On the CPU it is `torch.histc`. With `dims` or
    `data_view`, the histogram bins the data view of `t`. See
    [Data Views](data-view-calls.md#data-views).

    Args:
        t (torch.Tensor): The tensor.
        name (str): View id. The same name updates the view in place.
        bins (int): Number of bins, in 1..65536.
        rate (float | None): Snapshots per second for this view ([Rate Limit][rate-limit]).
        dims (Sequence[str] | None): One name for each dimension of `t`.
        data_view (dict | None): The default data view of the view.

    Returns:
        bool | ShowStats: As for [`show`][caliper.show].

    Raises:
        ValueError: When `bins` is outside 1..65536 or `t` is empty.
        DataViewError: When `data_view` does not fit `t`.
    """
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
    """Append `value` to the series `name`.

    The value is a Python `int` or `float`, a 0-d tensor, or a tensor of several values. Pass
    `loss.detach()`, not `loss.item()`, to avoid a GPU sync. A tensor of several values is
    flattened, and more than 4096 values keep the last 4096.

    The series is a ring of the last [`SERIES_CAPACITY`][caliper.SERIES_CAPACITY] (4096) float32
    values on the device of the value. A CPU series that later receives device values moves to the
    device with one host-to-device copy of its history. The x axis is the append index. Values
    offered while nothing is attached are not recorded. Every call appends on every attached route,
    and the view is republished at most `rate` times per second ([Rate Limit][rate-limit]).

    Args:
        value (float | torch.Tensor): The value or values to append.
        name (str): View id. The same name appends to the same series.
        rate (float | None): Republications per second for this view.
        items (str | None): The item space that the series index numbers, for linked selection.
            A series has no dims, so this is a name only. See [`items`][caliper.items].

    Returns:
        bool | ShowStats: As for [`show`][caliper.show]. A series also returns `True` when a
            server of [`serve`][caliper.serve] recorded the value.

    Raises:
        ValueError: When `value` is an empty tensor or `name` is invalid.
    """
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


def watch(model, optimizer=None, *, activations: bool = True, gradients: bool = False,
          every: int = 1, include=None, rate=None, groups=True, scene: bool = True):
    """Hook a module so that `show` displays its outputs and gradients, and with an optimizer, the run.

    The call returns a [`WatchHandle`][caliper._watch.WatchHandle]. The root module is labeled with
    its class name in lower case.

    ```python
    caliper.watch(model, optimizer)
    for x, y in loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

    With an optimizer, `watch()` also does these things:

    1. It starts the stream server with [`serve`][caliper.serve] when no server runs, so that the
       Caliper app can attach with control. The shared-memory session does not open, so no view
       appears twice in the app.
    2. It makes each `optimizer.step()` the checkpoint of the run, through a step pre-hook that
       calls [`step`][caliper.step]. Pending parameter values apply and queued actions run there,
       and a pause holds the run there, before the weights change. A script that passes an
       optimizer does not also call `caliper.step()`.
    3. It registers the parameter `lr` and publishes the series `loss` and `lr` and the scene
       `model`.

    In a managed run, which the model runner of the Caliper app starts with the environment variable
    `CALIPER_RUN_DIR` set, `watch()` also starts the server when it is not started. It loads the
    checkpoint of `CALIPER_RUN_RESUME` into the model and the optimizer before the first step and
    continues the step count. It applies the values of `CALIPER_RUN_PARAMS` and of the checkpoint
    when each parameter is registered. It registers the actions `caliper.checkpoint`, which saves
    the model, the optimizer, the step and the parameter values to `<run dir>/checkpoint-<step>.pt`
    and returns the path, and `caliper.stop`, which ends the script with exit code 0 at its next
    checkpoint, before the weights change. Last, it writes the stream link into
    `CALIPER_RUN_LINK_FILE`. Without an optimizer, each forward pass of the model is the checkpoint.
    See [Managed Runs](../agents/managed-runs.md).

    | Name | Kind | Content |
    |---|---|---|
    | `loss` | Series | The last 0-d tensor that `backward()` ran on before each step, or inside the closure of `optimizer.step(closure)` |
    | `lr` | Series | The learning rate that parameter group 0 used at each step |
    | `lr` | Parameter | A number on a log scale, from the start rate of group 0 divided by 1,000 to the start rate multiplied by 10 |
    | `model` | Scene | The entity `modules`: one box for each module of `WatchHandle.modules`, along +y in that order |

    A new value of `lr` applies to the next optimizer step, and every parameter group moves by the
    same factor, so the ratios between the groups stay. Without a learning rate scheduler, the new
    rates stay in `optimizer.param_groups`. With a torch scheduler, the value scales the
    scheduler's schedule from that step on: before each step the hook multiplies the scheduled
    rates by the factor, and after the step it puts them back, so the scheduler neither undoes nor
    compounds the change. `optimizer.param_groups` and `scheduler.get_last_lr()` then show the
    scheduled rates, and the series `lr` shows the rate that each step used.

    The size of a box shows the parameter count of its module, and its color the mean absolute
    value of the module's last output, computed at most 10 times a second and only while a viewer
    watches the scene. The row id of a box is the index of its module in `WatchHandle.modules`,
    so a pick on a box names the module, and each box has the module's name as its label.

    The grouped views are heatmaps with [data views](data-view-calls.md#data-views). Their sources
    are the selected modules or their parameters, so one widget can switch between layers. The same
    hooks feed them, so `watch()` adds no hook for them.

    | View | Sources | Dims | Default data view |
    |---|---|---|---|
    | `weights` | `<module>.<param>`, every parameter with 2 or more dims | `out`, `in` (higher-rank weights reshaped to `(out, -1)`) | The first source, whole |
    | `activations` | `<module>`, with `activations=True` | From the rank: `batch`, `feature`, or `batch`, `position`, `feature`, or `batch`, `channel`, `height`, `width`, or `batch`, `channel`, `length` for a 1-D convolution | The first module, its leading dims at index 0 |
    | `gradients` | `<module>.<param>`, with `gradients=True`, as the `.grad` views | As `weights` | The first source, whole |

    The weights and gradients have the same two dims, so `{"source": "fc2.weight"}` alone switches
    between them. Activations of different ranks have different dims. A request for another module
    must name its own `index`, and `{}` for a 2-D output.

    `weights` takes a snapshot when its module runs forward and on [`update`][caliper.update]. With
    `activations=False`, it takes the snapshot after its gradient accumulates. A source does tensor
    work only when a route needs it. The window and the session need the source of the default data
    view, and a server needs the sources that its subscriptions select. The hook of any other source
    records the shape and returns.

    A view has at most 64 sources, so a group keeps the first 64. Sources with names longer than 63
    bytes are left out, each with a `RuntimeWarning`.

    Args:
        model (torch.nn.Module): The module to hook.
        optimizer (torch.optim.Optimizer | None): The optimizer of the run. `None` watches the
            module only, as above.
        activations (bool): A forward hook per selected module shows its output as `<module>.act`.
            A tuple output gives its first tensor. Only floating-point, non-empty tensors show.
            `(N, C, H, W)` outputs show item 0 as a channel mosaic, and `(N, F)` outputs show a
            heatmap.
        gradients (bool): Each selected module shows the gradient of its own parameters that have 2
            or more dims and `requires_grad`, as `<module>.<param>.grad` with colormap rdbu.
            Higher-rank weights are reshaped to `(out, -1)`.
        every (int): Show 1 of every N hook calls (per hook). Must be at least 1.
        include (str | Sequence[str] | Callable[[str, torch.nn.Module], bool] | None): `None`
            selects leaf modules. Strings are `fnmatch` patterns over `named_modules()` names. A
            callable is `predicate(name, module) -> bool`.
        rate (float | None): Passed to `show`.
        groups (bool | str): Also offer the grouped views above. A string prefixes their names with
            `<string>.`, and `False` offers none.
        scene (bool): With an optimizer, also publish the scene `model`. Without one, it does
            nothing.

    Returns:
        WatchHandle: The handle of the hooks. Its `remove()` takes them off, with the parameter
            `lr`.

    Raises:
        TypeError: When `model` is not a `torch.nn.Module`, or `optimizer` is not an optimizer.
        ValueError: When `every` is below 1.
    """
    from ._watch import watch as _w
    h = _w(model, optimizer, activations=activations, gradients=gradients, every=every,
           include=include, rate=rate, groups=groups, scene=scene)
    if optimizer is not None:                     # GPU programs read this model's tensors
        _programs.attach(model, optimizer, h)
    return h


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
    """Register view `name` with several sources, for tensors that no view call shows.

    Examples are attention maps or the output of a layer on a fixed probe batch. A viewer, or
    [`set_data_view`][caliper.set_data_view], picks one source and a data view of it. The view takes
    snapshots when the training loop calls [`update`][caliper.update]. A later `offer()` of the same
    name replaces the view.

    A tensor is read at each snapshot, so a tensor updated in place, such as a parameter, stays
    current. A callable runs under `torch.no_grad()`, once at the call to learn the shape and then
    only when a route needs that source and a snapshot is due.

    Args:
        name (str): View id.
        sources (torch.Tensor | Callable[[], object] | dict): Maps source names to tensors or to
            callables that return the current tensor or `None`. A single tensor or callable is one
            source named `name`.
        kind (str): `"heatmap"`, `"image"`, `"points"` or `"histogram"`. A series has no sources.
        dims (Sequence[str] | dict | None): One tuple of dimension names for every source, or a dict
            of tuples per source. The default is `dim0`, `dim1`, ... An image source has one more
            dimension, RGB(A), and a points source a last one with 2 or 3 coordinates. Neither is
            named.
        data_view (dict | None): The default data view of the view.
        rate (float | None): Snapshots per second for this view ([Rate Limit][rate-limit]).
        items (str | dict | None): The item space of the item axis (the first dimension of a heatmap
            or image, or the point dimension), as a name or as `{dim: space}`, for linked selection.
            See [`items`][caliper.items].
        **options (object): The arguments of the matching call: `cmap`, `vmin` and `vmax`
            (heatmap, points), `labels` (a tensor or a callable, along the kept dimension) and
            `classes` (points), and `bins` (histogram).

    Returns:
        dict | None: The default data view, resolved, or `None` while the default source has no
            tensor.

    Raises:
        TypeError: When an option is unknown for `kind`.
        ValueError: When `kind` is unknown or is a series, or when an option value is invalid.
        DataViewError: When `data_view` does not fit the sources.
    """
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
    """Take a snapshot of every offered view that is due, and of the `weights` of `watch()`.

    With `name`, the call takes a snapshot of that view only. It takes each snapshot of the source
    that each route needs: the one of the default data view for the window and the session, and the
    ones of the subscriptions for a server. Call it in the training loop.

    Args:
        name (str | None): The view to update. `None` updates every view that is due.

    Returns:
        bool | ShowStats: As for [`show`][caliper.show]. For every view, `True` if any took a
            snapshot.

    Raises:
        ValueError: When `name` was not offered.
    """
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
    """Return the default data view of view `name`, resolved.

    Args:
        name (str): View id.

    Returns:
        dict | None: A dict with `source`, `index`, `reduce` and `budget`, and `tile` when the
            heatmap tiles. `None` when the view offers no data views or has no data yet.
    """
    return _dataview.REGISTRY.resolved(name)


def set_data_view(name: str, request=None):
    """Set the default data view of view `name`, from any thread.

    The next snapshot uses it, and the next VIEWS message carries it. A subscription with its own
    data view keeps it. The fields of `request` replace the fields of the current default as a
    whole (`budget` key by key), and the other fields stay, so `{}` changes nothing. A request for
    a view that does not exist yet, or has no data yet, is kept and checked at its first snapshot.
    See [Data Views](data-view-calls.md#data-views) for the fields.

    Args:
        name (str): View id.
        request (dict | str | None): A dict, or its JSON text, with any of `source`, `index`,
            `reduce`, `tile` and `budget`. `None` returns to the default that the view was
            registered with.

    Returns:
        dict | None: The resolved data view, or `None` when the view does not exist yet or has no
            data yet.

    Raises:
        DataViewError: When the request does not fit the view. The view keeps the data view it had.
    """
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
    """Open the live session, or return the open one.

    Attach to the session from the Caliper app (F4 Attach to process) or from
    `caliper-viewer`. The first view call of a script opens it implicitly with
    `auto_launch=False`, unless `CALIPER_SESSION=0`.

    Args:
        auto_launch (bool): `True` also spawns `caliper-viewer --session <token>`, unless a spawned
            viewer is running or a viewer is attached.
        viewer_path (str | None): Overrides the viewer binary.
        script_name (str | None): The name that viewers show. `None` takes the basename of
            `sys.argv[0]`.

    Returns:
        caliper.live.SessionTransport: The open session.

    Raises:
        RuntimeError: When the build has no session transport.
    """
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
    """Open the in-process window and route every call to it.

    An open window of the same kind is returned as is, and any other canvas is replaced.
    Call this function on the main thread only.

    The core of the window keeps its data in `$CALIPER_DATA_DIR`, else in a private temporary
    directory that closes with the window. The data are its stores, the data folders of the
    applets and its `imgui.ini`.

    The window has these limits:

    - macOS delivers input only to the main thread. Input waits during long steps unless
      [`run`][caliper.run] gives the main thread to Caliper. Outside `run()`, a view call on the
      main thread pumps events at most 60 times a second.
    - A crash of the script takes the window down, and a hung window hangs the script.
    - A process has one core, so only one window or offscreen canvas exists at a time.

    While open, the window holds an `NSProcessInfo` user-initiated, latency-critical activity.
    Without it, macOS App Nap throttles the process of an occluded window.

    Args:
        width (int): Width of the window.
        height (int): Height of the window.
        title (str): Title of the window.
        offscreen (bool): `True` drives an offscreen canvas from the same render thread (CI,
            `render()`).
        fps (float): Frame rate of the render thread.
        rate (float | None): Default snapshot rate of the route. `None` takes 60 per second.
        activate (bool): `False` shows the window without taking focus from the terminal.
        heartbeat (bool): `True` adds a view named `caliper.render` that scrolls one texel per
            rendered frame. A screenshot then shows whether the canvas repaints.

    Returns:
        caliper._inproc.Window: The window. Its `run`, `stats`, `render` and `close` members work as
            the functions of the same name.

    Raises:
        RuntimeError: When called off the main thread.
    """
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
    """Open the synchronous offscreen canvas for tests and notebooks.

    It replaces an open window. Call this function on the main thread only.

    The canvas has no render thread. Each call waits for the GPU event of its snapshot,
    applies it on the calling thread and renders a frame (at most 60 per second). Thus
    [`view_info`][caliper.view_info], [`view_readback`][caliper.view_readback] and
    [`render`][caliper.render] see the call at once. The default rate is every call. In a
    notebook, `caliper.render()` returns the composited frame as an `(H, W, 4)` uint8 array
    to display.

    Args:
        width (int): Width of the canvas.
        height (int): Height of the canvas.

    Returns:
        caliper._inproc.Window: The canvas.

    Raises:
        RuntimeError: When called off the main thread.
    """
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
    """Run `fn(*args, **kwargs)`, on a worker thread when a window is open.

    With a window open, the main thread keeps the window interactive while `fn` runs.
    Otherwise `run` calls `fn` directly.

    The worker is a daemon thread named `caliper.train`. Ctrl-C reaches the thread as
    `KeyboardInterrupt`. If the user closes the window, `fn` keeps running and its view calls
    become no-ops. The window closes when `fn` returns.

    Args:
        fn (Callable): The function to run.
        *args (object): Positional arguments of `fn`.
        **kwargs (object): Keyword arguments of `fn`.

    Returns:
        object: The result of `fn`.

    Raises:
        RuntimeError: When a window is open and the call is not on the main thread.
        Exception: Any exception of `fn`, re-raised on the calling thread.
    """
    w = _window
    if w is not None and not w.closed and not w.offscreen:
        return w.run(fn, *args, **kwargs)
    return fn(*args, **kwargs)


def flush(timeout: float = 5.0) -> int:
    """Publish every pending snapshot and poll their GPU events.

    The call is not for the training loop.

    Args:
        timeout (float): Seconds to wait at most.

    Returns:
        int: How many snapshots are still pending after `timeout`.
    """
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
    """Flush pending snapshots, then keep the view alive.

    With a window, run its event loop until it closes or `seconds` pass. With a session, wait
    until the viewer exits or `seconds` pass. The wait ends at once when no viewer is attached
    and none was spawned.

    Args:
        seconds (float | None): Seconds to wait at most. `None` waits without a limit.

    Raises:
        RuntimeError: When a window is open and the call is not on the main thread.
    """
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
    """Close every route: the window, the offscreen canvas and the session.

    The next view call opens a new session. The call also runs at interpreter exit.
    """
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
    """Remove view `name` from every route.

    Args:
        name (str): The view id.

    Returns:
        bool: `True` if the in-process core had the view.
    """
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
    """Return the ring counters per route plus the render stats of the window.

    The call works without a core. The result has one entry per route. The key `"session"`
    holds `ring` counters and `transport` (the result of
    [`SessionTransport.stats()`][caliper.live.SessionTransport.stats]). The key `"window"` holds
    `ring` plus the render stats below. Any other route has its `repr` as key.

    The ring counters are `taken`, `published`, `skipped` (rate limit), `ring_full`,
    `superseded` (a newer snapshot completed first), `appended` (series values), `allocs` and
    `recreated`.

    The window render stats are `running`, `fps`, `frames`, `published`, `superseded`,
    `drawn`, `deferred_takes`, `errors`, `last_error`, `stats_unreadable`, `nonfinite_range`,
    `queued`, `max_frame_gap_ms`, `events_applied` and `heatmap_live_apply`. The sample lists
    are `latency_ms` (snapshot to first frame drawn with it), `tick_ms` and `frame_gap_ms`.
    `Window.stats(reset=True)` clears them.

    Returns:
        dict: The stats per route.
    """
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
    """Render one frame now.

    For a real window, call this function on the main thread.

    Returns:
        float: The wall time of the frame in ms.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`, or for a real window off the main
            thread.
    """
    w = _require_window("frame")
    if not w.offscreen:
        _check_main_thread("frame")
    w._last_frame = time.perf_counter()
    return w.core.frame()


def render():
    """Render a frame and return its pixels (offscreen canvases).

    Returns:
        numpy.ndarray: A freshly composited frame, `(H, W, 4)` uint8.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`, or when the canvas is not offscreen.
    """
    return _require_window("render").render()


def read_pixels():
    """Return a copy of the last composited frame (offscreen canvases).

    Returns:
        numpy.ndarray: The frame, `(H, W, 4)` uint8.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("read_pixels").core.read_pixels()


def view_info(name: str) -> dict:
    """Return the state of a heatmap or image view.

    The result has `path`, `width`, `height`, `device` (`"metal"` or `"cpu"`), `updates`,
    `last_update_ms`, `imported` and `creates`.

    Args:
        name (str): The view id.

    Returns:
        dict: The state of the view.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("view_info").core.view_info(name)


def view_readback(name: str):
    """Read back the texture of a heatmap or image view.

    Args:
        name (str): The view id.

    Returns:
        numpy.ndarray: The texture, `(H, W, 4)` uint8.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("view_readback").core.view_readback(name)


def view_remove(name: str) -> None:
    """Remove view `name`, as [`remove`][caliper.remove] does.

    Args:
        name (str): The view id.
    """
    remove(name)


def view_count() -> int:
    """Count the heatmap and image views.

    Returns:
        int: The number of views. 0 without a core.
    """
    w = _window
    return w.core.view_count() if w is not None and not w.closed else 0


def live_view_info(name: str) -> dict:
    """Return the state of any typed view.

    The result has `kind` (1 heatmap, 2 image, 3 points, 4 series, 5 histogram), `path`,
    `width`, `height`, `updates`, `creates`, `dropped`, `seq`, `stamp_ns`, `last_apply_ms`,
    `latency_ms`, `imported`, `read_done` and `count`.

    Args:
        name (str): The view id.

    Returns:
        dict: The state of the view.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("live_view_info").core.live_view_info(name)


def live_view_values(name: str):
    """Return the series values or the histogram counts that a view holds.

    Series values come oldest first.

    Args:
        name (str): The view id.

    Returns:
        numpy.ndarray: A float32 array of the values or counts.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("live_view_values").core.live_view_values(name)


def live_view_readback(name: str):
    """Read back a heatmap, image or points view.

    Args:
        name (str): The view id.

    Returns:
        numpy.ndarray: The texture, `(H, W, 4)` uint8.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    return _require_window("live_view_readback").core.live_view_readback(name)


def live_pause(paused: bool = True) -> None:
    """Freeze the display of every typed view.

    Snapshots keep flowing and are dropped.

    Args:
        paused (bool): `True` freezes the display, `False` resumes it.

    Raises:
        RuntimeError: Without a `window()` or `offscreen()`.
    """
    _require_window("live_pause").core.live_set_paused(paused)


def live_paused() -> bool:
    """Tell whether the display of the typed views is frozen.

    Returns:
        bool: `True` while paused. `False` without a core.
    """
    w = _window
    return bool(w.core.live_paused()) if w is not None and not w.closed else False


def colormap_lut(name="viridis"):
    """Return a colormap as a look-up table.

    The call needs no core.

    Args:
        name (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.

    Returns:
        numpy.ndarray: A `(256,)` uint32 array. Each entry is `R | G<<8 | B<<16 | A<<24`.

    Raises:
        ValueError: When `name` is not a known colormap.
    """
    return _load_native().colormap_lut(cmap_index(name))


def log_lines() -> list[str]:
    """Return the log lines of the core.

    Returns:
        list[str]: The lines. `[]` without a core.
    """
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
    """Show the live view `name` in a notebook cell, as an anywidget widget.

    The widget reads the WebSocket stream of the server directly, not the message channel of the
    notebook, so it keeps updating while a cell runs. It is an ipywidgets widget, so `HBox` and
    `VBox` lay several out together.

    When no server runs, `widget()` starts one with [`serve`][caliper.serve]. Without anywidget
    installed, the call raises `ImportError` and names the extra `caliper[notebook]`.

    Args:
        name (str): The view to show. The widget may be created before the first call of the view.
        kind (str | None): `"series"`, `"heatmap"`, `"image"`, `"points"`, `"histogram"` or
            `"scene"`. `None` takes the kind from the stream.
        height (int): Height of the whole widget, header included, in CSS pixels.

    Returns:
        CaliperWidget: The widget, with the traits and the `on_select` method listed below.

    Raises:
        ValueError: When `name` is not a non-empty string or `kind` is unknown.
        ImportError: When anywidget is not installed.
    """
    from ._widget import widget as _widget
    return _widget(name, kind, height=height)


# ---------------------------------------------------------------------------
# network stream (caliper.serve: wire protocol v3 over a WebSocket on 127.0.0.1)
# ---------------------------------------------------------------------------
__all__ += ["serve"]
_server = None                      # the running caliper._net.NetServer (read by the widget)
_series_tap = None                  # its series tap while it runs


def serve(port: int = 0, *, control: bool = True):
    """Start the stream server of this process, print its link, and return the server.

    The link is `http://127.0.0.1:<port>/?token=<token>`. Viewers on this machine use it: the
    browser page, the notebook widget and the Caliper app. A second call while the server runs
    returns the same server. [`close`][caliper.close] stops it.

    The server listens on 127.0.0.1 only and makes a new random session token. While it listens,
    it writes the stream file `streams/<pid>.json` in the app-data folder of Caliper
    (`$CALIPER_DATA_DIR` when set, see [Environment](../reference/folders-and-environment.md)).
    The file has mode 0600 in a folder of mode 0700. It holds the pid, the machine name, the
    session id, the script name, the start time, the port, the stream link with its token,
    `control` and the view names. The server rewrites the file when the views change.

    The Caliper app lists the stream from this file in `F4 Attach to process`, so it attaches
    without a pasted link. `close()` removes the file, and the next listing of the app removes
    the file of a killed script. A file that cannot be written leaves the server running, and the
    printed link still attaches.

    The server is one more route. Every view call is offered to it. A view takes a snapshot for
    it only while a viewer subscribes to that view, once for each data view that its
    subscriptions ask for. The snapshots are computed on the device and copied into pinned host
    memory without waiting, so the call still returns at once.

    Args:
        port (int): TCP port on 127.0.0.1. 0 lets the system choose a free port.
        control (bool): Whether viewers may pause the run, change parameters and run actions
            ([Controls](run-control-calls.md#controls)). `False` shows them read-only.

    Returns:
        NetServer: The running server. It works as a context manager that closes it.
    """
    from ._net import serve as _serve
    return _serve(port, control=control)


__all__ += ["connect"]


def connect(url: str | None = None):
    """Stream every view of this process, as a producer, to a `caliper serve` hub on this machine.

    The call returns the connection. [Hub Setup](hub-setup.md) explains the hub.

    `connect()` returns at once and never blocks the training loop, because the native sender
    connects in the background. When the hub cannot be reached or the connection drops, the
    sender tries again after 0.5 seconds and doubles the wait up to 30 seconds. It proposes the
    session id that the hub assigned, so the hub resumes the session. It stops when the hub ends
    the connection with BYE `replaced`.

    The views reach the hub through the same calls as for [`serve`][caliper.serve]. A view takes
    a snapshot for the hub only while a viewer of the hub watches it. Caliper LIB keeps one
    producer per process, so `serve()` and `connect()` in one script share it, and every view
    call reaches both. [`close`][caliper.close] ends the connection as well.

    Args:
        url (str | None): The stream URL of the hub with its token,
            `ws://127.0.0.1:<port>/v3/stream?token=<token>`, or the viewer link that the hub
            prints. `None` reads `CALIPER_HUB_URL`, then the hub file: `CALIPER_HUB_FILE`, or
            `hub.json` in the app-data folder (`~/Library/Application Support/Caliper/hub.json`
            on macOS).

    Returns:
        HubConnection: The connection to the hub.

    Raises:
        ValueError: When the URL has no token or names a host other than 127.0.0.1 or localhost.
        RuntimeError: When no URL is given and there is no hub file, or when the native sender
            (`caliper._netio`) is not built.
    """
    from ._net import connect as _connect
    return _connect(url)


# ---------------------------------------------------------------------------
# controls of the running script and linked selection (wire protocol 3.1;
# caliper._controls holds the state and the seam its transports attach to)
# ---------------------------------------------------------------------------
__all__ += ["step", "pause", "param", "action", "items", "controls", "Param", "Action",
            "ItemSpace"]
from . import _controls                                        # noqa: E402
from . import _programs                                        # noqa: E402
from ._controls import Action, ItemSpace, Param                # noqa: E402
_CTL = _controls.CONTROLS


def step(step=None) -> None:
    """Mark the checkpoint of the training loop, where the controls change the script.

    Call it once per step, on the training thread, between two steps. It is the only place where
    the controls change the script. There it sets the parameter values that viewers changed, runs
    the actions they asked for in the order they asked, and waits while a viewer has paused the
    run.

    A waiting `step()` returns in these cases:

    - A viewer resumes the run or asks for a number of single steps.
    - The server closes.
    - No connection that may control has been open for 10 seconds. A page reload stays well
      within this time.

    While it waits, it still sets values and runs actions as they arrive. Ctrl-C, or the
    interrupt of a notebook, ends the wait with `KeyboardInterrupt`, and the run is no longer
    paused afterward. Right before it waits, it sends the snapshots and series values of the step
    where it paused. The run state goes to each viewer after them, so a view that reads `paused`
    already shows that step ([What a viewer may rely
    on](../reference/run-control-messages.md#what-a-viewer-may-rely-on-when-a-run-pauses)).

    With nothing waiting, a call only counts the step and reads one flag. While the run runs and
    a server is open, the first call after each second also sends the step number to the viewers.

    Before the first `step()` of the script, viewers can change parameters, which then take
    effect at once. They cannot pause the run, and actions wait for the first checkpoint.

    Args:
        step (int | None): The number of the step, which viewers show where the run paused.
            `None` counts the calls.
    """
    c = _CTL
    if step is None:
        c.n += 1
    else:
        c.n = step
    if c.busy:
        c.checkpoint()
    if _programs.ACTIVE:                                       # GPU programs (caliper._programs)
        _programs.at_step(c.n)


def pause() -> bool:
    """Pause the run at the next [`step`][caliper.step], as the Pause button of a viewer does.

    Use it for example when the loss stops being finite. A viewer resumes the run. When no stream
    server runs, no viewer could resume it. The call then does nothing and warns with a
    `RuntimeWarning`. When a server runs but no connection may control it, the call warns that the
    run resumes after 10 seconds.

    Returns:
        bool: `True` when the run is paused, `False` when no stream server runs.
    """
    return _CTL.pause()


def param(name: str, value, *, min=None, max=None, log=False, choices=None, help=None) -> Param:
    """Register a parameter that viewers can change while the script runs.

    ```python
    lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1, log=True, help="learning rate")
    for batch in loader:
        caliper.step()
        for group in opt.param_groups:
            group["lr"] = lr.value
    ```

    The type of the value sets the type of the parameter:

    | `value` | Type | A viewer shows |
    |---|---|---|
    | `bool` | `boolean` | A toggle |
    | `int` | `integer` | A slider or a field |
    | `float` | `number` | A slider or a field |
    | `str`, with `choices` | `choice` | A menu |

    A value from a viewer is checked and clamped. A number or an integer is clamped to `min` and
    `max`, an integer must be a whole number, and a choice must be one of `choices`. The value
    reaches `Param.value` at the next [`step`][caliper.step], or at once before the first `step()`
    of the script. When a viewer sends several values before a `step()`, the last one wins.
    Registering a name again replaces the parameter. A script has at most 64 parameters.

    Args:
        name (str): 1 to 63 bytes. Registering a name again replaces the parameter.
        value (bool | int | float | str): The first value. Its type sets the type of the parameter.
        min (int | float | None): The lower end of the range of a number or an integer. A value
            from a viewer is clamped to it.
        max (int | float | None): The upper end of the range of a number or an integer. A value
            from a viewer is clamped to it.
        log (bool): Draw the slider on a log scale. Needs `min` greater than 0 and `max`.
        choices (Sequence[str] | None): The choices of a `str` value, 1 to 64.
        help (str | None): A tooltip, at most 1,024 bytes.

    Returns:
        Param: The registered parameter. Read its `value` in the loop.

    Raises:
        ValueError: When an argument is out of range, for example `log` without `min` and `max`,
            a `value` outside `min` and `max`, or more than 64 parameters.
        TypeError: When `value` is a `str` without `choices`, or of another unsupported type.
    """
    return _CTL.param(name, value, min=min, max=max, log=log, choices=choices, help=help)


def action(name: str, fn=None, *, help=None):
    """Register a function that viewers can run, as an action.

    ```python
    @caliper.action("save checkpoint", help="saves the model's weights")
    def save_checkpoint():
        torch.save(model.state_dict(), "checkpoint.pt")
        return "saved checkpoint.pt"
    ```

    `caliper.action(name, fn)` registers `fn`, a function without arguments. `caliper.action(name)`
    does the same as a decorator. A request of a viewer runs the function at the next
    [`step`][caliper.step], on the training thread, after the parameter values are set. Requests
    run one at a time in the order they arrived. A request for an action that already waits joins
    that run.

    The text that `fn` returns goes back to the viewer that asked, cut to 1,024 bytes. An
    exception goes back to that viewer with its message and appears in the script as a
    `RuntimeWarning`. It does not stop the loop. A request that no `step()` has run within 10
    minutes fails with a time-out. A script has at most 64 actions, and registering a name again
    replaces the action.

    Args:
        name (str): The name of the action, 1 to 63 bytes.
        fn (Callable[[], object] | None): The function to run. `None` returns a decorator.
        help (str | None): A tooltip, at most 1,024 bytes.

    Returns:
        Action: The registered action, or a decorator that returns it when `fn` is `None`.

    Raises:
        ValueError: When `name` is not 1 to 63 bytes, or when there are more than 64 actions.
        TypeError: When `fn` is not callable.
    """
    if fn is None:
        return lambda f: _CTL.action(name, f, help=help)
    return _CTL.action(name, fn, help=help)


def items(space: str, details=None) -> ItemSpace:
    """Register the item space `space` for linked selection, with optional details.

    ```python
    caliper.items("eval", details={"class": eval_labels})
    caliper.points(z, eval_labels, "embedding", classes=10, items="eval")
    caliper.show(probabilities, "predictions", items="eval")
    ```

    An item space is a set of items numbered from 0 that several views index, such as the samples
    of an evaluation set. A view names the item space of its item axis with `items=`, a keyword of
    `show`, `points`, `series` and `offer`:

    | Call | `items="eval"` names the space of |
    |---|---|
    | `show` | Its rows: the first dimension, or the first of `dims` |
    | `points` | Its points: the point dimension |
    | `series` | Its index, the number of values appended before each value |
    | `offer` | The first dimension of each source, or the point dimension of points |

    `items={"sample": "eval"}` names the dimensions of `dims` instead. A histogram has no item
    axis. A selection made on an axis with an item space highlights the same items in every view
    of the session that names the space: the points, the rows of a heatmap and markers on a
    series. A call without `items=` has no extra cost.

    `details` gives a viewer a few fields of the selected items, shown in a small table under the
    view where the selection was made. It is one of two forms:

    - A mapping `{field: sequence}` indexed by item id, copied to the host once, at the call.
    - A function `fn(ids)` that returns `{field: values}` in the order of `ids`, or one dict per
      item.

    The function runs on the thread `caliper.details`, never on the training thread, and has 2
    seconds. It must read only data that the loop does not change in place, such as labels or
    arrays that the script computed before. Details cover the first 32 selected items and at most
    16 fields. Each value is a number, a string of at most 256 bytes, a boolean or `None`.

    A session has at most 16 item spaces. Python code also receives each select event through
    `server.on_event`, `server.events()`, `widget.selection` and `on_select`. Its `detail` has
    `space` and `items` too.

    Args:
        space (str): The name of the item space, 1 to 63 bytes.
        details (dict | Callable[[Sequence[int]], object] | None): The details of the selected
            items, in one of the two forms above.

    Returns:
        ItemSpace: The item space, with the current selection and the calls to change it.

    Raises:
        ValueError: When a name is invalid, a details column has too many fields, or the session
            would have more than 16 item spaces.
        TypeError: When `details` is neither a mapping nor a function.
    """
    return _CTL.items(space, details)


def controls(*, height: int = 160):
    """Show the run controls, parameters and actions of the script in a notebook widget.

    The widget is like [`widget`][caliper.widget]. It shows the run state, the Pause, Resume and
    Step buttons, every parameter and every action of the script, and the result of the last
    action it ran. It reads the stream directly, so it works while a cell runs, and it starts
    [`serve`][caliper.serve] when no server runs. Its traits are `src` and `height`. The header of
    every view widget also shows the run state and the Pause, Resume and Step buttons.

    A build whose browser assets lack the module of the panel, `caliper-controls-widget.js`,
    warns and shows a note in its place. The call needs the notebook extra `caliper[notebook]`.

    Args:
        height (int): Height of the whole panel in CSS pixels.

    Returns:
        anywidget.AnyWidget: The controls panel.
    """
    from ._widget import controls as _controls_widget
    return _controls_widget(height=height)


# ---------------------------------------------------------------------------
# scenes (the scenes subset of wire protocol 3.2, draft; caliper._scene)
# ---------------------------------------------------------------------------
__all__ += ["scene", "Scene"]


def scene(name: str = "scene", *, space: str = "3d", up=None, units: str = "m", rate=None,
          data_view=None, view_camera=None) -> "Scene":
    """Make or fetch the scene `name`, a tree of entities that every viewer of the network route draws.

    A scene holds entities, named by entity paths such as `"world/cloud"`. Each entity holds
    batches of points, lines, arrows, boxes, a mesh or an image, or a pinhole camera, and an entity
    transform places it. The space is 3-D or 2-D, fixed when the scene is made. Log to the returned
    [`Scene`][caliper.Scene] with its `points`, `lines`, `arrows`, `boxes`, `mesh`, `image`,
    `depth`, `segmentation`, `camera`, `transform`, `classes` and `clear` methods
    ([Scene Calls](scene-calls.md)).

    Scenes travel over [`serve`][caliper.serve]: the notebook widget with `kind="scene"`, the
    browser page and a hub. With only the shared-memory session or the in-process window open, a
    scene call warns once and does nothing there. The scene module is imported on first use, so
    importing caliper stays light.

    A second call with the same `name` returns the same scene and updates `up`, `units`, `rate` and
    `view_camera`. `up`, `units` and `rate` are set from the arguments of that call, so an omitted
    one returns to its default. `view_camera=None` leaves the starting camera as it is, and
    `view_camera={}` returns it to the fit.

    Args:
        name (str): The scene id: 1 to 63 bytes, with no `##`.
        space (str): `"3d"` or `"2d"`. Positions are `(N, 3)` in a 3-D scene and `(N, 2)` in a 2-D
            one. A scene keeps its space.
        up (str | None): In 3-D, the axis a view keeps upright when it opens, from `"+x"` to
            `"-z"`. In 2-D, the direction of y on screen, `"-y"` or `"+y"`. `None` is `"+y"` in 3-D
            and `"-y"` in 2-D.
        units (str): A note of at most 15 bytes shown with the view. Nothing is converted.
        rate (float | None): The default rate limit of each batch, in snapshots per second. `None`
            is 60 per second, and 0 means every call ([Rate Limits](scene-calls.md#rate-limits)).
        data_view (dict | None): Not built yet.
        view_camera (dict | str | None): The camera a 3-D view opens on: a dict with any of `target`
            (3 numbers), `distance` (above 0), `azimuth` and `elevation` (degrees, from -89 to 89),
            or the entity path of a camera to look through. Keys left out are filled from the fit.

    Returns:
        Scene: The scene.

    Raises:
        ValueError: When `name`, `space`, `up` or `units` is invalid, `rate` is negative or not
            finite, `view_camera` has an unknown key, a number that is not finite, a distance not
            above 0, an elevation outside -89 to 89 or a path outside the entity path grammar, or
            `view_camera` is given in a 2-D scene, or `space` differs from the space of the
            existing scene.
        TypeError: When `rate` is not a number, or `view_camera` is neither a dict nor a string,
            or one of its values has the wrong type.
        NotImplementedError: When `data_view` is given.
    """
    from . import _scene
    return _scene.scene(name, space=space, up=up, units=units, rate=rate, data_view=data_view,
                        view_camera=view_camera)
