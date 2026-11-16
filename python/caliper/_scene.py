"""caliper.scene: scenes of points, lines and meshes on the network route.

This is the part of the scene calls that is built, on
the scenes subset of the protocol 3.2 draft (docs/references/wire-protocol-3-2.md):
``caliper.scene(name, space=, up=, units=, rate=)`` and, on the scene it
returns, ``points``, ``lines`` (segments or strips) and ``mesh``, each with
``static=`` and ``rate=``. Only ``caliper.serve()`` and a hub that it connects
to carry scenes so far; with only the shared-memory session or the
in-process window open, a scene call warns once and does nothing there.

A call, on the thread that makes it:

* checks every argument from host metadata only (shapes, dtypes, devices and
  the scene's limits), so a bad call raises at once, attached or not;
* with no viewer of the scene attached, returns False after those checks,
  unless the batch is static: a static batch is the scene's setup, so it is
  snapshotted once at the call even with nothing attached, and the producer
  retains it for every viewer that subscribes later;
* otherwise, within the batch's rate limit, builds the snapshot once: the
  casts (positions, values and radii to f32, indices and strip lengths to
  i32, colours to RGBA u8) and the per-axis bounds and value range run on the
  arrays' own device, the default data view's budget samples points and
  segments by stride, and each array is copied with ``non_blocking=True`` into
  pinned host memory, after which one ``torch.mps.Event`` is recorded. The
  network thread (caliper/_net_scene.py) reads the copies once the event has
  completed. Nothing here waits for the device: there is no ``.item()``,
  ``.cpu()`` of a device tensor, ``.numpy()``, ``.tolist()`` or
  ``synchronize()``.

The arguments and calls of the full scene API that are not built yet raise
NotImplementedError with a message that says so.
"""
from __future__ import annotations

import math
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from . import _wire
from ._convert import check_range, cmap_index

__all__ = ["scene", "Scene", "SceneState", "Batch", "SCENES", "forget"]

KIND_SCENE = int(_wire.Kind.SCENE)
_CMAPS = {"viridis": 0, "magma": 1, "rdbu": 2}
DEFAULT_RATE_HZ = 60.0
META_MARGIN = 4096                       # bytes of metadata a batch frame may need, at most
PART_ORDER = {_wire.PART_POINTS: 0, _wire.PART_LINES: 1, _wire.PART_MESH: 2}


def _not_built(call: str, what: str) -> NotImplementedError:
    """The error of an argument or call of the scene API that is not built yet."""
    return NotImplementedError(f"{call} is not built yet ({what})")


SCENES: dict[str, "SceneState"] = {}     # name -> the scene (training thread writes; the network
                                         # thread reads its static batches)
_lock = threading.Lock()
_warned = False
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


# ---------------------------------------------------------------------------
# a snapshot on its way to the network
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Batch:
    """One snapshot of one part of one entity. ``tensors`` holds the wire
    arrays in host memory (pinned for a device source, written by queued
    copies); ``stats`` the device-computed pieces of the stats array; the
    network thread reads them once ``ready()`` and fills ``arrays``."""
    scene: str
    path: str
    part: str
    seq: int
    stamp_ns: int
    static: bool
    params: dict
    tensors: dict
    stats: list                           # [values (2,) or None, bounds (2 * D,) or None, D]
    axes: list | None = None
    event: Any = None
    nbytes: int = 0
    arrays: dict | None = None            # name -> numpy, once finished (network thread)

    def ready(self) -> bool:
        return self.event is None or self.event.query()


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------
class SceneState:
    """Everything the producer holds of one scene."""

    def __init__(self, name: str, space: str, up: str, units: str, rate):
        self.name = name
        self.space = space
        self.dims = 3 if space == _wire.SPACE_3D else 2
        self.up = up
        self.units = units
        self.rate = rate
        self.interval = _interval(rate if rate is not None else DEFAULT_RATE_HZ)
        self.version = 0                  # grows when up, units or rate change
        self.seq = 0                      # the scene's batch counter
        self.static: dict = {}            # (path, part) -> Batch, the retained static batches
        self.last: dict = {}              # (path, part) -> perf_counter of its last snapshot
        self.batches: set = set()         # (path, part) logged so far
        self.entities: set = set()        # paths logged so far
        self.counters: dict = {}

    def info(self) -> dict:
        return {"space": self.space, "up": self.up, "units": self.units}

    def count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1


def _interval(rate) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError(f"rate must be a number of snapshots per second or None, got {rate!r}")
    rate = float(rate)
    if not rate >= 0 or math.isinf(rate):
        raise ValueError(f"rate must be a finite number >= 0, got {rate!r}")
    return 1.0 / rate if rate > 0 else 0.0


def _check_name(name) -> None:
    if not isinstance(name, str) or not name or "##" in name or \
            len(name.encode("utf-8")) > _wire.NAME_MAX:
        raise ValueError(f"invalid scene name {name!r} (1 to {_wire.NAME_MAX} bytes, no '##')")


def scene(name: str = "scene", *, space: str = "3d", up=None, units: str = "m", rate=None,
          data_view=None, view_camera=None) -> "Scene":
    """See caliper.scene."""
    _check_name(name)
    if data_view is not None:
        raise _not_built("caliper.scene(data_view=...)", "scene data views")
    if view_camera is not None:
        raise _not_built("caliper.scene(view_camera=...)", "the starting camera")
    if space not in (_wire.SPACE_3D, _wire.SPACE_2D):
        raise ValueError(f"space must be '3d' or '2d', got {space!r}")
    axes = _wire.UP_AXES_3D if space == _wire.SPACE_3D else _wire.UP_AXES_2D
    if up is None:
        up = "+y" if space == _wire.SPACE_3D else "-y"
    if up not in axes:
        raise ValueError(f"up must be one of {list(axes)} in a {space} scene, got {up!r}")
    if not isinstance(units, str) or len(units.encode("utf-8")) > _wire.UNITS_MAX:
        raise ValueError(f"units must be a str of at most {_wire.UNITS_MAX} bytes, got {units!r}")
    if rate is not None:
        _interval(rate)
    import caliper
    routes = caliper._routes or caliper._auto_routes()
    with _lock:
        st = SCENES.get(name)
        if st is None:
            st = SCENES[name] = SceneState(name, space, up, units, rate)
        else:
            if st.space != space:
                raise ValueError(f"scene {name!r} is {st.space}; a scene keeps its space")
            if (st.up, st.units, st.rate) != (up, units, rate):
                st.up, st.units, st.rate = up, units, rate
                st.interval = _interval(rate if rate is not None else DEFAULT_RATE_HZ)
                st.version += 1
    for sr in _scene_routes(routes):
        sr.ensure(st)
    return Scene(st)


def forget(name: str) -> None:
    """caliper.remove(name): drop the scene and what it retains. A later call
    on its Scene object makes it again, empty, as a view call makes its view."""
    with _lock:
        st = SCENES.pop(name, None)
        if st is not None:
            st.static, st.last, st.batches, st.entities = {}, {}, set(), set()


def attach_ring(ring) -> None:
    """A network route opened: list every scene made so far in its VIEWS."""
    from ._net_scene import route_of
    sr = route_of(ring)
    for st in list(SCENES.values()):
        sr.ensure(st)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
_route_cache: tuple = ((), (), False)     # (routes tuple, scene routes, other routes open)
_CALIPER = None


def _caliper():
    global _CALIPER
    import caliper
    _CALIPER = caliper
    return caliper


def _scene_routes(routes) -> tuple:
    """The SceneRoutes of the open network routes (cached per routes tuple),
    warning once when only other routes are open."""
    global _route_cache, _warned
    cached = _route_cache
    if cached[0] is routes:
        srs, others = cached[1], cached[2]
    else:
        from ._net_scene import route_of
        srs, others = [], False
        from ._net_ring import NetRing
        for r in routes:
            ring = getattr(r, "ring", None)
            if isinstance(ring, NetRing):        # caliper.serve() or caliper.connect()
                srs.append(route_of(ring))
            else:
                others = True
        srs = tuple(srs)
        _route_cache = (routes, srs, others)
    if others and not srs and not _warned:
        _warned = True
        warnings.warn("caliper: scenes reach only caliper.serve() viewers (the notebook widget, "
                      "the browser page and a hub) in this version; the shared-memory session "
                      "and the in-process window do not show them yet", RuntimeWarning,
                      stacklevel=4)
    return srs


# ---------------------------------------------------------------------------
# argument checks (host metadata only)
# ---------------------------------------------------------------------------
_FLOATS = None


def _float_dtypes():
    global _FLOATS
    if _FLOATS is None:
        torch = _torch()
        _FLOATS = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    return _FLOATS


def _array(x, what: str, kind: str):
    """A tensor from ``x`` without touching a device's data: a tensor
    (detached), a numpy array or a nested sequence (host data). ``kind`` is
    "float" or "int": what the array must hold."""
    torch = _TORCH or _torch()
    if isinstance(x, torch.Tensor):
        t = x.detach() if x.requires_grad else x
    else:
        import numpy as np
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x) if x.dtype != np.float64 or kind != "float" else \
                torch.from_numpy(x.astype(np.float32))
        elif isinstance(x, (list, tuple)):
            try:
                t = torch.as_tensor(x, dtype=torch.float32 if kind == "float" else
                                    torch.int32 if kind == "int" else None)
                if kind == "color" and not t.dtype.is_floating_point:
                    if t.numel() and (int(t.min()) < 0 or int(t.max()) > 255):
                        raise ValueError("integer colour components are 0 to 255")
                    t = t.to(torch.uint8)
                elif kind == "color":
                    t = t.to(torch.float32)
            except (TypeError, ValueError, RuntimeError) as e:
                raise TypeError(f"{what} must be a tensor, a numpy array or a sequence of "
                                f"numbers: {e}") from None
        else:
            raise TypeError(f"{what} must be a tensor, a numpy array or a sequence of numbers, "
                            f"got {type(x).__name__}")
    if kind == "float":
        if t.dtype not in _float_dtypes():
            raise ValueError(f"{what} must hold floats, got {t.dtype}")
    elif kind == "int" and (t.dtype.is_floating_point or t.dtype.is_complex or
                            t.dtype == torch.bool):
        raise ValueError(f"{what} must hold integers, got {t.dtype}")
    dev = t.device.type
    if dev not in ("cpu", "mps"):
        t = t.to("cpu")                      # macOS: any other device goes through the host
    return t


def _check_path(path) -> None:
    if not isinstance(path, str):
        raise TypeError(f"the entity path must be a str, got {type(path).__name__}")
    if not _wire.entity_path_ok(path):
        raise ValueError(f"invalid entity path {path!r}: 0 to {_wire.PATH_SEGMENTS_MAX} segments "
                         f"of [A-Za-z0-9_.-]{{1,63}} joined by '/', at most {_wire.PATH_MAX} "
                         "bytes, no '.' or '..'")


def _check_positions(st: SceneState, positions, what="positions"):
    t = _array(positions, what, "float")
    if t.ndim != 2 or t.shape[1] != st.dims:
        raise ValueError(f"{what} must be (N, {st.dims}) in a {st.space} scene, got "
                         f"{tuple(t.shape)}")
    if t.shape[0] > _wire.MAX_BATCH_POINTS:
        raise ValueError(f"{t.shape[0]} {what}, more than {_wire.MAX_BATCH_POINTS} in a batch")
    return t


def _one_colour(c):
    """One colour for every item: a tuple or list of 3 or 4 integers from 0
    to 255 or floats from 0 to 1, as [r, g, b, a] of 0..255."""
    if len(c) not in (3, 4) or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                       for x in c):
        raise ValueError(f"one colour is 3 or 4 numbers, got {c!r}")
    if all(isinstance(x, int) for x in c):
        if not all(0 <= x <= 255 for x in c):
            raise ValueError(f"colour components are 0 to 255, got {c!r}")
        out = [int(x) for x in c]
    else:
        if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in c):
            raise ValueError(f"float colour components are 0 to 1, got {c!r}")
        out = [int(float(x) * 255 + 0.5) for x in c]
    return out + [255] if len(out) == 3 else out


def _flat(colors) -> bool:
    """One colour for all: a flat tuple or list of numbers."""
    return isinstance(colors, (tuple, list)) and len(colors) > 0 and \
        not isinstance(colors[0], (tuple, list))


def _rows(x) -> int:
    return x.shape[0] if hasattr(x, "shape") and len(x.shape) else len(x)


def _check_colors(colors, values, rows_ok, what_rows):
    """(colour key, colors tensor, values tensor). ``rows_ok(n)`` says whether
    an array of n rows fits."""
    if colors is not None and values is not None:
        raise ValueError("colors and values together: an item's colour comes from one of them")
    torch = _TORCH or _torch()
    if colors is not None and _flat(colors):
        return _one_colour(colors), None, None
    ct = vt = None
    if colors is not None:
        ct = _array(colors, "colors", "color")
        if ct.ndim != 2 or ct.shape[1] not in (3, 4) or not rows_ok(ct.shape[0]):
            raise ValueError(f"colors must be ({what_rows}, 3) or ({what_rows}, 4), got "
                             f"{tuple(ct.shape)}")
        if ct.dtype != torch.uint8 and ct.dtype not in _float_dtypes():
            raise ValueError(f"colors must be uint8 or floats in [0, 1], got {ct.dtype}")
    if values is not None:
        vt = _array(values, "values", "float")
        if vt.ndim != 1 or not rows_ok(vt.shape[0]):
            raise ValueError(f"values must be ({what_rows},), got {tuple(vt.shape)}")
    return None, ct, vt


def _check_range_args(cmap, vmin, vmax):
    cm = _CMAPS.get(cmap) if cmap.__class__ is str else None
    if cm is None:
        cm = cmap_index(cmap)
    if vmin is not None or vmax is not None:
        for v in (vmin, vmax):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise TypeError(f"vmin/vmax must be numbers or None, got {v!r}")
        check_range(vmin, vmax)
    return cm


def _check_static_rate(static, rate):
    if not isinstance(static, bool):
        raise TypeError(f"static must be True or False, got {static!r}")
    return None if rate is None else _interval(rate)


# ---------------------------------------------------------------------------
# the scene object
# ---------------------------------------------------------------------------
class Scene:
    """A scene of entities, each holding batches of points, lines and a mesh,
    drawn in one 3-D or 2-D space. What :func:`caliper.scene` returns. Every
    call logs one part of one entity and returns True if a route took a
    snapshot, False when every route skipped it (nothing attached, the rate
    limit)."""
    __slots__ = ("_s",)

    def __init__(self, state: SceneState):
        self._s = state

    name = property(lambda self: self._s.name)
    space = property(lambda self: self._s.space)
    up = property(lambda self: self._s.up)
    units = property(lambda self: self._s.units)
    rate = property(lambda self: self._s.rate)

    def __repr__(self) -> str:
        s = self._s
        return f"<caliper.Scene {s.name!r} {s.space} up={s.up} units={s.units!r}>"

    # -- the calls that are built ----------------------------------------------
    def points(self, path, positions, *, colors=None, values=None, cmap="viridis", vmin=None,
               vmax=None, class_ids=None, radii=None, labels=None, items=None, static=False,
               rate=None) -> bool:
        """Points at ``positions`` (N, D), coloured by ``colors`` (N, 3 or 4),
        one colour (a tuple), or ``values`` (N,) through ``cmap`` over
        [vmin, vmax] (an open end is computed on the device); ``radii`` (N,)
        or one number, in scene units (default: discs of a few pixels)."""
        if class_ids is not None:
            raise _not_built("scene.points(class_ids=...)", "the class table")
        if labels is not None:
            raise _not_built("scene.points(labels=...)", "labels")
        if items is not None:
            raise _not_built("scene.points(items=...)", "picking and linked selection")
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        pos = _check_positions(st, positions)
        n = pos.shape[0]
        color, ct, vt = _check_colors(colors, values, lambda r: r == n, "N")
        cm = _check_range_args(cmap, vmin, vmax)
        params, rt = {}, None
        if radii is not None:
            if isinstance(radii, (int, float)) and not isinstance(radii, bool):
                if not (math.isfinite(radii) and radii > 0):
                    raise ValueError(f"radii must be above 0, got {radii!r}")
                params["radius"] = float(radii)
            else:
                rt = _array(radii, "radii", "float")
                if rt.ndim != 1 or rt.shape[0] != n:
                    raise ValueError(f"radii must be (N,) or one number, got {tuple(rt.shape)}")
        if color is not None:
            params["color"] = color
        spec = _Spec(_wire.PART_POINTS, pos, ct, vt, cm, vmin, vmax, params,
                     extra={"radii": rt} if rt is not None else {})
        return _log(st, path, spec, static, iv)

    def lines(self, path, positions, *, indices=None, strip_lengths=None, colors=None,
              values=None, cmap="viridis", vmin=None, vmax=None, color_per=None, width=2.0,
              static=False, rate=None) -> bool:
        """Lines through ``positions`` (V, D): segments (vertices 2k and
        2k + 1), or strips with ``strip_lengths`` (S,) (a list or an integer
        tensor); coloured per vertex or per strip (``color_per``, read from
        the row count when None), or with one colour, or by ``values``
        through ``cmap``; ``width`` in pixels."""
        if indices is not None:
            raise _not_built("scene.lines(indices=...)", "index-pair edges")
        if color_per == "edge":
            raise _not_built("scene.lines(color_per='edge')", "index-pair edges")
        if color_per not in (None, _wire.COLOR_PER_VERTEX, _wire.COLOR_PER_STRIP):
            raise ValueError(f"color_per must be 'vertex', 'strip' or None, got {color_per!r}")
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        pos = _check_positions(st, positions)
        v = pos.shape[0]
        if isinstance(width, bool) or not isinstance(width, (int, float)) or \
                not (math.isfinite(width) and width > 0):
            raise ValueError(f"width must be a number above 0 (pixels), got {width!r}")
        params = {"width": float(width)}
        extra = {}
        if strip_lengths is None:
            if v % 2:
                raise ValueError(f"segments need an even number of vertices, got {v}")
            if color_per == _wire.COLOR_PER_STRIP:
                raise ValueError("color_per='strip' needs strip_lengths")
            params["mode"] = _wire.LINES_SEGMENTS
            s_count = None
            rows_ok, rows = (lambda r: r == v), "V"
        else:
            if isinstance(strip_lengths, (list, tuple)) and all(
                    x.__class__ is int for x in strip_lengths):
                sl = strip_lengths                   # host ints: checked here, a tensor later
                s_count = len(sl)
                total, low = sum(sl), min(sl, default=1)
            else:
                sl = _array(strip_lengths, "strip_lengths", "int")
                if sl.ndim != 1:
                    raise ValueError(f"strip_lengths must be (S,), got {tuple(sl.shape)}")
                s_count = sl.shape[0]
                total = low = None
                if sl.device.type == "cpu" and s_count:    # host data: the sum is checked here
                    total, low = int(sl.sum()), int(sl.min())
            if s_count > _wire.MAX_BATCH_ITEMS:
                raise ValueError(f"{s_count} strips, more than {_wire.MAX_BATCH_ITEMS} in a batch")
            if total is not None:
                if s_count and low < 1:
                    raise ValueError("every strip length is at least 1")
                if total != v:
                    raise ValueError(f"the strip lengths add up to {total}, not to the {v} "
                                     "vertices of positions")
            params["mode"] = _wire.LINES_STRIPS
            extra["strip_lengths"] = sl
            if color_per is None and (values is not None or
                                      (colors is not None and not _flat(colors))):
                r = _rows(colors if colors is not None else values)
                if r == v and r == s_count:
                    raise ValueError(f"{r} colour rows fit both the vertices and the strips; "
                                     "pass color_per='vertex' or 'strip'")
                color_per = _wire.COLOR_PER_STRIP if r == s_count else _wire.COLOR_PER_VERTEX
            if color_per == _wire.COLOR_PER_STRIP:
                params["color_per"] = _wire.COLOR_PER_STRIP
                rows_ok, rows = (lambda r: r == s_count), "S"
            else:
                rows_ok, rows = (lambda r: r == v), "V"
        color, ct, vt = _check_colors(colors, values, rows_ok, rows)
        cm = _check_range_args(cmap, vmin, vmax)
        if color is not None:
            params["color"] = color
        spec = _Spec(_wire.PART_LINES, pos, ct, vt, cm, vmin, vmax, params, extra=extra)
        return _log(st, path, spec, static, iv)

    def mesh(self, path, positions, indices=None, *, normals=None, colors=None, values=None,
             cmap="viridis", vmin=None, vmax=None, uvs=None, texture=None, static=False,
             rate=None) -> bool:
        """A triangle mesh of ``positions`` (V, D) and ``indices`` (T, 3), any
        integer dtype (without indices each three vertices form a triangle),
        coloured per vertex by ``colors`` (V, 3 or 4), one colour, or
        ``values`` (V,) through ``cmap``; faces are shaded flat. An index out
        of range draws as the last vertex."""
        if normals is not None:
            raise _not_built("scene.mesh(normals=...)", "smooth shading")
        if uvs is not None or texture is not None:
            raise _not_built("scene.mesh(uvs=..., texture=...)", "textured meshes")
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        pos = _check_positions(st, positions)
        v = pos.shape[0]
        extra = {}
        if indices is None:
            if v % 3:
                raise ValueError(f"a mesh without indices needs a multiple of 3 vertices, got {v}")
        else:
            it = _array(indices, "indices", "int")
            if it.ndim != 2 or it.shape[1] != 3:
                raise ValueError(f"indices must be (T, 3), got {tuple(it.shape)}")
            if it.shape[0] > _wire.MAX_BATCH_POINTS:
                raise ValueError(f"{it.shape[0]} triangles, more than {_wire.MAX_BATCH_POINTS} "
                                 "in a batch")
            extra["indices"] = it
        color, ct, vt = _check_colors(colors, values, lambda r: r == v, "V")
        cm = _check_range_args(cmap, vmin, vmax)
        params = {"color": color} if color is not None else {}
        spec = _Spec(_wire.PART_MESH, pos, ct, vt, cm, vmin, vmax, params, extra=extra)
        return _log(st, path, spec, static, iv)

    # -- the calls that are not built yet --------------------------------------
    def _later(self, call: str, what: str):
        raise _not_built(f"scene.{call}()", what)

    def arrows(self, *a, **kw):
        self._later("arrows", "arrows")

    def boxes(self, *a, **kw):
        self._later("boxes", "boxes")

    def image(self, *a, **kw):
        self._later("image", "images")

    def depth(self, *a, **kw):
        self._later("depth", "depth images")

    def segmentation(self, *a, **kw):
        self._later("segmentation", "segmentation images")

    def camera(self, *a, **kw):
        self._later("camera", "pinhole cameras")

    def transform(self, *a, **kw):
        self._later("transform", "entity transforms")

    def classes(self, *a, **kw):
        self._later("classes", "the class table")

    def clear(self, *a, **kw):
        self._later("clear", "clearing entities")


@dataclass(eq=False)
class _Spec:
    """A checked call: what the snapshot needs, all host metadata."""
    part: str
    positions: Any
    colors: Any
    values: Any
    cmap: int
    vmin: Any
    vmax: Any
    params: dict
    extra: dict = field(default_factory=dict)   # radii, strip_lengths or indices


# ---------------------------------------------------------------------------
# the call: limits, rate, snapshot
# ---------------------------------------------------------------------------
def _frame_bytes(spec: _Spec) -> int:
    """The bytes of the batch's arrays on the wire, from their shapes, after
    the default data view's budget."""
    n, d = spec.positions.shape
    rows = n
    if spec.part == _wire.PART_POINTS:
        rows = min(n, _wire.DEFAULT_MAX_POINTS)
    elif spec.part == _wire.PART_LINES and spec.params["mode"] == _wire.LINES_SEGMENTS:
        rows = 2 * min(n // 2, _wire.SCENE_DEFAULT_MAX_ITEMS)
    strip = spec.params.get("color_per") == _wire.COLOR_PER_STRIP
    out = rows * d * 4
    for name, t in (("colors", spec.colors), ("values", spec.values)):
        if t is not None:
            out += (t.shape[0] if strip else rows) * 4       # RGBA u8 or f32
    for name, t in spec.extra.items():
        out += (rows if name == "radii" else len(t) if isinstance(t, (list, tuple))
                else t.numel()) * 4
    return out


def _log(st: SceneState, path: str, spec: _Spec, static: bool, iv) -> bool:
    caliper = _CALIPER or _caliper()
    if SCENES.get(st.name) is not st:               # removed: a call makes it again
        with _lock:
            if SCENES.setdefault(st.name, st) is not st:
                raise ValueError(f"scene {st.name!r} was removed and made again; log to the "
                                 "Scene object that caliper.scene returns now")
    nbytes = _frame_bytes(spec)
    if nbytes + META_MARGIN > _wire.MAX_FRAME_BYTES:
        raise ValueError(f"this batch needs {nbytes} bytes, more than one frame of "
                         f"{_wire.MAX_FRAME_BYTES >> 20} MiB; split updates are not built yet")
    key = (path, spec.part)
    if key not in st.batches:
        if path not in st.entities and len(st.entities) >= _wire.MAX_ENTITIES:
            raise ValueError(f"scene {st.name!r} holds {_wire.MAX_ENTITIES} entities, its limit")
        if len(st.batches) >= _wire.MAX_BATCHES:
            raise ValueError(f"scene {st.name!r} holds {_wire.MAX_BATCHES} batches, its limit")
        st.entities.add(path)
        st.batches.add(key)
    srs = _scene_routes(caliper._routes or caliper._auto_routes())
    watching = []
    for sr in srs:
        sr.ensure(st)
        if sr.watched(st.name):
            watching.append(sr)
    if not static:
        if not watching:
            st.count("unwatched")
            return False
        now = time.perf_counter()
        if now - st.last.get(key, -math.inf) < (st.interval if iv is None else iv):
            st.count("rate_limited")
            return False
        st.last[key] = now
    b = _snapshot(st, path, spec, static)
    if static:
        st.static[key] = b
    for sr in watching:
        sr.queue(b)
    st.count("taken")
    return True


def _stride(count: int, size: int, device):
    from ._dataview import _stride as stride
    return stride(count, size, device)


def _snapshot(st: SceneState, path: str, spec: _Spec, static: bool) -> Batch:
    """Build the batch on the arrays' devices and queue its copies into host
    memory. Never waits for a device."""
    torch = _TORCH or _torch()
    f32, i32, u8 = torch.float32, torch.int32, torch.uint8
    pos = spec.positions
    n = pos.shape[0]
    params = dict(spec.params)
    arrays: dict = {}
    axes = None
    sample = None                       # the default budget's stride: (count, size, per segment)
    if spec.part == _wire.PART_POINTS and n > _wire.DEFAULT_MAX_POINTS:
        sample = (n, _wire.DEFAULT_MAX_POINTS, False)
        axes = [{"dim": "point", "start": 0, "step": 1, "count": n, "size": sample[1]}]
    elif spec.part == _wire.PART_LINES and params["mode"] == _wire.LINES_SEGMENTS and \
            n // 2 > _wire.SCENE_DEFAULT_MAX_ITEMS:
        sample = (n // 2, _wire.SCENE_DEFAULT_MAX_ITEMS, True)
        axes = [{"dim": "segment", "start": 0, "step": 1, "count": n // 2, "size": sample[1]}]

    def rows(t):
        """The rows of ``t`` the budget keeps, gathered on t's own device."""
        if sample is None:
            return t
        idx = _stride(sample[0], sample[1], t.device)
        if sample[2]:                   # both vertices of each kept segment
            idx = (idx[:, None] * 2 + torch.arange(2, device=t.device)).reshape(-1)
        return t.index_select(0, idx)

    arrays["positions"] = rows(pos if pos.dtype == f32 else pos.to(f32))
    per_vertex = params.get("color_per") != _wire.COLOR_PER_STRIP
    if spec.colors is not None:
        c = spec.colors
        if c.dtype != u8:
            c = (c.to(f32).clamp(0.0, 1.0) * 255.0 + 0.5).to(u8)
        if c.shape[1] == 3:
            c = torch.cat([c, torch.full((c.shape[0], 1), 255, dtype=u8, device=c.device)], 1)
        arrays["colors"] = rows(c) if per_vertex else c
    vals = None
    if spec.values is not None:
        vals = spec.values if spec.values.dtype == f32 else spec.values.to(f32)
        vals = rows(vals) if per_vertex else vals
        arrays["values"] = vals
        params["colormap"] = spec.cmap
        params["flags"] = (_wire.SNAP_AUTO_VMIN if spec.vmin is None else 0) | \
                          (_wire.SNAP_AUTO_VMAX if spec.vmax is None else 0)
        params["vmin"] = 0.0 if spec.vmin is None else float(spec.vmin)
        params["vmax"] = 1.0 if spec.vmax is None else float(spec.vmax)
    for name, t in spec.extra.items():
        if name == "radii":
            arrays["radii"] = rows(t if t.dtype == f32 else t.to(f32))
        elif name in ("strip_lengths", "indices"):
            if isinstance(t, (list, tuple)):
                t = torch.tensor(t, dtype=i32)
            arrays[name] = t if t.dtype == i32 else t.to(i32)
    # The stats: value range and per-axis bounds, on the arrays' devices.
    sent = arrays["positions"]
    bounds = None
    if sent.shape[0]:
        bounds = torch.stack([sent.amin(0), sent.amax(0)], 1).reshape(-1)
    vrange = None
    if vals is not None and vals.shape[0]:
        vrange = torch.stack(torch.aminmax(vals))
    nbytes = sum(t.numel() * t.element_size() for t in arrays.values())
    # The copies into host memory: pinned and non-blocking from a device.
    host = {}
    mps = False
    for name, t in list(arrays.items()) + [("_bounds", bounds), ("_vrange", vrange)]:
        if t is None:
            continue
        dev = t.device.type != "cpu"
        mps = mps or t.device.type == "mps"
        out = torch.empty(t.shape, dtype=t.dtype, pin_memory=dev)
        out.copy_(t, non_blocking=dev)
        host[name] = out
    event = None
    if mps:
        event = torch.mps.Event()
        event.record()
    st.seq += 1
    return Batch(st.name, path, spec.part, st.seq, time.monotonic_ns(), static, params,
                 {k: v for k, v in host.items() if not k.startswith("_")},
                 [host.get("_vrange"), host.get("_bounds"), st.dims], axes, event, nbytes)
