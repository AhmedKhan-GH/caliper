"""caliper.scene: scenes on the network route.

The scene calls of Caliper LIB on the scenes of the protocol 3.2 draft
(docs/reference/wire-protocol-3-2.md): ``caliper.scene(name, space=, up=,
units=, rate=, view_camera=)`` and, on the scene it returns, ``points``,
``lines`` (segments, strips and index-pair edges), ``arrows``, ``boxes``,
``mesh``, ``image``, ``depth``, ``segmentation``, ``camera``, ``transform``,
``classes`` and ``clear``. Only ``caliper.serve()`` and a hub that it connects
to carry scenes so far; with only the shared-memory session or the
in-process window open, a scene call warns once and does nothing there.

A call, on the thread that makes it:

* checks every argument from host metadata only (shapes, dtypes, devices, the
  scene's limits and what the entity can hold), so a bad call raises at once,
  attached or not;
* for a primitive with no viewer of the scene attached, returns False after
  those checks, unless the batch is static: a static batch is the scene's
  setup, so it is snapshotted once at the call even with nothing attached, and
  the producer retains it for every viewer that subscribes later;
* otherwise, within the batch's rate limit, builds the snapshot once: the
  casts (floats to f32, integers to i32, colours and colour images to RGBA
  u8), the per-axis bounds and value range, the composition of an entity
  transform given as tensors, all on the arrays' own device; the default
  data view's budget samples points and segments by stride; and each array is
  copied with ``non_blocking=True`` into pinned host memory, after which one
  ``torch.mps.Event`` is recorded. The network thread (caliper/_net_scene.py)
  reads the copies once the event has completed. Nothing here waits for the
  device: there is no ``.item()``, ``.cpu()`` of a device tensor, ``.numpy()``,
  ``.tolist()`` or ``synchronize()``.

Cameras, entity transforms, the class table and clears are small: the
producer keeps them whether or not anything is attached, and a later viewer
receives them in the scene's state. They are never dropped by a rate limit.
A clear applies at once to what the producer keeps.

``data_view=``, ``items=``, ``blend=`` and the instance arrays of a mesh are not
built yet and raise NotImplementedError.
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

__all__ = ["scene", "Scene", "SceneState", "Batch", "Transform", "Classes", "Clear", "SCENES",
           "forget"]

KIND_SCENE = int(_wire.Kind.SCENE)
_CMAPS = {"viridis": 0, "magma": 1, "rdbu": 2}
DEFAULT_RATE_HZ = 60.0
META_MARGIN = 4096                       # bytes of metadata a batch frame may need, at most
PART_ORDER = {p: i for i, p in enumerate(_wire.PARTS)}
PRIMITIVES = tuple(p for p in _wire.PARTS if p != _wire.PART_CAMERA)
IMAGE_FORMS = {b"\x89PNG\r\n\x1a\n": _wire.FORMAT_PNG, b"\xff\xd8\xff": _wire.FORMAT_JPEG}
# The 16 colours that categorical points draw with (src/host/live/live_views_draw.cpp,
# kPalette): a class id without a colour of its own takes colour id mod 16.
PALETTE = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
           (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
           (196, 156, 148))


def _not_built(call: str, what: str) -> NotImplementedError:
    """The error of an argument of the scene API that is not built yet."""
    return NotImplementedError(f"{call} is not built yet ({what})")


SCENES: dict[str, "SceneState"] = {}     # name -> the scene (training thread writes; the network
                                         # thread reads what it retains)
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
# what a call hands to the network
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Batch:
    """One SCENE_UPDATE: a snapshot of one primitive of one entity, or a
    camera (no arrays). ``tensors`` holds the wire arrays in host memory
    (pinned for a device source, written by queued copies); ``stats`` the
    device-computed pieces of the stats array ([values (2,) or None, bounds
    (2 * D,) or None, D], or [range (2,), None, 0] for a depth image, or None);
    ``cast`` the numpy dtypes some arrays take on the host once copied. The
    network thread reads them once ``ready()`` and fills ``arrays``."""
    scene: str
    path: str
    part: str
    seq: int
    stamp_ns: int
    static: bool
    params: dict
    tensors: dict
    stats: list | None
    axes: list | None = None
    event: Any = None
    nbytes: int = 0
    cast: dict = field(default_factory=dict)
    arrays: dict | None = None            # name -> numpy, once finished (network thread)
    refused: bool = False                 # finished and found invalid: never sent
    step: int | None = None               # the run's step at the call (the step of a piece form's RESOURCE)
    packed: Any = None                    # the piece form's resources, packed once (caliper._scene_pieces)

    def ready(self) -> bool:
        return self.event is None or self.event.query()


@dataclass(eq=False)
class Transform:
    """One entity transform: ``matrix`` is the 4 x 4 matrix stored
    column-major, (16,) f32 in host memory (pinned and written by a queued
    copy when it was composed on a device); ``values`` the numpy copy once
    finished."""
    scene: str
    path: str
    seq: int
    stamp_ns: int
    static: bool
    matrix: Any
    event: Any = None
    values: Any = None

    def ready(self) -> bool:
        return self.event is None or self.event.query()


@dataclass(eq=False)
class Classes:
    """A scene's class table: ``classes`` are {id, label, color}, ascending."""
    scene: str
    seq: int
    stamp_ns: int
    static: bool
    classes: list

    def ready(self) -> bool:
        return True


@dataclass(eq=False)
class Clear:
    """A clear of entity ``path``, its subtree too with ``recursive``."""
    scene: str
    seq: int
    stamp_ns: int
    path: str
    recursive: bool
    static: bool

    def ready(self) -> bool:
        return True

    def reaches(self, path: str) -> bool:
        if path == self.path:
            return True
        return self.recursive and (self.path == "" or path.startswith(self.path + "/"))

    def removes(self, path: str, static: bool) -> bool:
        """Whether this clear removes a part at ``path``: static parts only when
        the clear is static too."""
        return self.reaches(path) and (self.static or not static)


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------
class SceneState:
    """Everything the producer holds of one scene."""

    def __init__(self, name: str, space: str, up: str, units: str, rate, view_camera=None):
        self.name = name
        self.space = space
        self.dims = 3 if space == _wire.SPACE_3D else 2
        self.up = up
        self.units = units
        self.rate = rate
        self.view_camera = view_camera
        self.interval = _interval(rate if rate is not None else DEFAULT_RATE_HZ)
        self.version = 0                  # grows when up, units, rate or view_camera change
        self.seq = 0                      # the scene's message counter
        self.static: dict = {}            # (path, part) -> Batch, the retained static batches
        self.cameras: dict = {}           # path -> Batch of the camera part, retained
        self.transforms: dict = {}        # path -> Transform, retained
        self.classes: Classes | None = None
        self.kinds: dict = {}             # path -> the parts it holds (a camera excludes primitives)
        self.last: dict = {}              # (path, part) -> perf_counter of its last snapshot
        self.batches: set = set()         # (path, part) logged so far
        self.entities: set = set()        # paths logged so far
        self.counters: dict = {}

    def info(self) -> dict:
        out = {"space": self.space, "up": self.up, "units": self.units}
        if self.view_camera is not None:
            out["view_camera"] = self.view_camera
        return out

    def count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def under_camera(self, path: str) -> bool:
        """Whether a proper ancestor of ``path`` holds a camera: its children
        live in the camera's 2-D pixel space."""
        if not self.cameras or not path:
            return False
        segs = path.split("/")
        return any("/".join(segs[:k]) in self.cameras for k in range(1, len(segs)))

    def dims_at(self, path: str) -> int:
        return 2 if self.dims == 2 or self.under_camera(path) else 3


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


def _real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_view_camera(vc, space: str):
    """The starting camera of a scene, as scenes.md defines it: an orbit dict
    of target, distance, azimuth and elevation (degrees), or the entity path
    of a camera. Returns the value as VIEWS carries it."""
    if isinstance(vc, str):
        if space != _wire.SPACE_3D:
            raise ValueError("a 2-D scene takes no view_camera: its fit frames it whole")
        if vc == "" or not _wire.entity_path_ok(vc):
            raise ValueError(f"view_camera {vc!r} is not the entity path of a camera")
        return vc
    if not isinstance(vc, dict):
        raise TypeError(f"view_camera must be a dict or an entity path, got {type(vc).__name__}")
    if space != _wire.SPACE_3D:
        raise ValueError("a 2-D scene takes no view_camera: its fit frames it whole")
    out = {}
    for key, v in vc.items():
        if key not in _wire.VIEW_CAMERA_KEYS:
            raise ValueError(f"unknown view_camera key {key!r}; the keys are "
                             f"{list(_wire.VIEW_CAMERA_KEYS)}")
        if key == "target":
            try:
                vals = list(v)
            except TypeError:
                raise TypeError(f"view_camera target must be 3 numbers, got {v!r}") from None
            if len(vals) != 3 or not all(_real(x) for x in vals):
                raise TypeError(f"view_camera target must be 3 numbers, got {v!r}")
            if not all(math.isfinite(x) for x in vals):
                raise ValueError(f"view_camera target must be finite, got {v!r}")
            out[key] = [float(x) for x in vals]
            continue
        if not _real(v):
            raise TypeError(f"view_camera {key} must be a number, got {v!r}")
        if not math.isfinite(v):
            raise ValueError(f"view_camera {key} must be finite, got {v!r}")
        if key == "distance" and not v > 0:
            raise ValueError(f"view_camera distance must be above 0, got {v!r}")
        if key == "elevation" and not -89 <= v <= 89:
            raise ValueError(f"view_camera elevation is from -89 to 89 degrees, got {v!r}")
        out[key] = float(v)
    return out


def scene(name: str = "scene", *, space: str = "3d", up=None, units: str = "m", rate=None,
          data_view=None, view_camera=None) -> "Scene":
    """See caliper.scene."""
    _check_name(name)
    if data_view is not None:
        raise _not_built("caliper.scene(data_view=...)", "scene data views")
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
    vc = None if view_camera is None else _check_view_camera(view_camera, space)
    import caliper
    routes = caliper._routes or caliper._auto_routes()
    with _lock:
        st = SCENES.get(name)
        if st is None:
            st = SCENES[name] = SceneState(name, space, up, units, rate, vc)
        else:
            if st.space != space:
                raise ValueError(f"scene {name!r} is {st.space}; a scene keeps its space")
            new_vc = st.view_camera if view_camera is None else vc
            if (st.up, st.units, st.rate, st.view_camera) != (up, units, rate, new_vc):
                st.up, st.units, st.rate, st.view_camera = up, units, rate, new_vc
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
            st.cameras, st.transforms, st.classes, st.kinds = {}, {}, None, {}


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


def _live(st: SceneState) -> None:
    """A call on a scene that caliper.remove dropped makes it again."""
    if SCENES.get(st.name) is not st:
        with _lock:
            if SCENES.setdefault(st.name, st) is not st:
                raise ValueError(f"scene {st.name!r} was removed and made again; log to the "
                                 "Scene object that caliper.scene returns now")


def _watching(st: SceneState) -> list:
    caliper = _CALIPER or _caliper()
    out = []
    for sr in _scene_routes(caliper._routes or caliper._auto_routes()):
        sr.ensure(st)
        if sr.watched(st.name):
            out.append(sr)
    return out


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
    "float", "int", "color" or "any": what the array must hold."""
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


def _check_holds(st: SceneState, path: str, part: str) -> None:
    """What an entity can hold: a camera excludes primitives, the root holds no
    camera, a camera is never under another, and in a 3-D scene an image sits
    under a camera."""
    held = st.kinds.get(path, ())
    if part == _wire.PART_CAMERA:
        if path == "":
            raise ValueError("the root holds no camera: give the camera an entity path")
        if any(p != _wire.PART_CAMERA for p in held):
            raise ValueError(f"{path!r} holds primitives; an entity that holds a camera holds no "
                             "primitive, so put the camera beside them")
        if st.under_camera(path):
            raise ValueError(f"{path!r} lies under a camera, in its 2-D pixel space; a camera "
                             "cannot be placed there")
        return
    if _wire.PART_CAMERA in held:
        raise ValueError(f"{path!r} holds a camera; an entity that holds a camera holds no "
                         "primitive: log to a child, which lives in the camera's pixel space")
    if part == _wire.PART_IMAGE and st.dims == 3 and not st.under_camera(path):
        raise ValueError(f"an image in a 3-D scene sits under a camera, its place coming from the "
                         f"intrinsics; log a camera at a parent of {path!r} first")


def _check_rows(st: SceneState, path: str, x, what: str, limit: int, kind: str = "float"):
    """A required (N, D) array, D the dims of the entity's space."""
    d = st.dims_at(path)
    t = _array(x, what, kind)
    if t.ndim != 2 or t.shape[1] != d:
        where = "under a camera" if d == 2 and st.dims == 3 else f"in a {st.space} scene"
        raise ValueError(f"{what} must be (N, {d}) {where}, got {tuple(t.shape)}")
    if t.shape[0] > limit:
        raise ValueError(f"{t.shape[0]} {what}, more than {limit} in a batch")
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


def _check_static(static) -> None:
    if not isinstance(static, bool):
        raise TypeError(f"static must be True or False, got {static!r}")


def _check_width(width) -> float:
    if isinstance(width, bool) or not isinstance(width, (int, float)) or \
            not (math.isfinite(width) and width > 0):
        raise ValueError(f"width must be a number above 0 (pixels), got {width!r}")
    return float(width)


def _check_labels(labels, n: int):
    """One label for all items (a str) or one per item (a list of n str), at
    most MAX_LABELS of NAME_MAX bytes each. Returns the wire list."""
    if isinstance(labels, str):
        out = [labels]
    elif isinstance(labels, (list, tuple)) and all(isinstance(x, str) for x in labels):
        out = list(labels)
        if len(out) != n:
            raise ValueError(f"{len(out)} labels for {n} items: give one label (a str) for all "
                             "or one per item")
    else:
        raise TypeError(f"labels must be a str or a list of str, got {type(labels).__name__}")
    if len(out) > _wire.MAX_LABELS:
        raise ValueError(f"{len(out)} labels, more than {_wire.MAX_LABELS} in a batch")
    for x in out:
        if len(x.encode("utf-8")) > _wire.NAME_MAX:
            raise ValueError(f"a label is at most {_wire.NAME_MAX} bytes, got {x!r}")
    return out


def _check_opacity(opacity) -> float:
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or \
            not (math.isfinite(opacity) and 0 <= opacity <= 1):
        raise ValueError(f"opacity is a number from 0 to 1, got {opacity!r}")
    return float(opacity)


def _check_side(t, what: str, rank_ok) -> None:
    if not rank_ok(t):
        raise ValueError(f"{what} has the shape {tuple(t.shape)}")
    if t.shape[0] > _wire.MAX_IMAGE_SIDE or t.shape[1] > _wire.MAX_IMAGE_SIDE:
        raise ValueError(f"{what} is at most {_wire.MAX_IMAGE_SIDE} pixels on a side, got "
                         f"{tuple(t.shape)}")


def _no(**kw) -> None:
    """Arguments of the scene API that are not built yet: refused when given."""
    for name, (value, default, what) in kw.items():
        if value is not default and value != default:
            raise _not_built(f"{name}=", what)


# ---------------------------------------------------------------------------
# the scene object
# ---------------------------------------------------------------------------
class Scene:
    """A scene of entities, drawn in one 3-D or 2-D space, returned by [`scene`][caliper.scene].

    Each entity holds batches of primitives, a camera or an entity transform. A primitive call
    (`points`, `lines`, `arrows`, `boxes`, `mesh`, `image`, `depth`, `segmentation`) logs one part
    of one entity and returns `True` if a route took a snapshot. It returns `False` when every
    route skipped it: nothing is attached, or the rate limit dropped it. `camera`, `transform`,
    `classes` and `clear` are kept whether or not anything is attached, and return `True`.

    Every call checks its arguments from host metadata only, so a bad call raises at once, attached
    or not ([Scene Calls](scene-calls.md#primitives)). A call on a scene that
    [`remove`][caliper.remove] dropped makes it again, empty. Only [`serve`][caliper.serve] and a
    hub that it connects to carry scenes; with only the shared-memory session or the in-process
    window open, a scene call warns once and does nothing there. `data_view=`, `items=`, `blend=`
    and the instance arrays of a mesh are not built yet and raise `NotImplementedError`.

    Attributes:
        name (str): The scene id.
        space (str): `"3d"` or `"2d"`.
        up (str): The axis a 3-D view keeps upright, or the direction of y in 2-D.
        units (str): The unit note shown with the view.
        rate (float | None): The default rate limit of each batch, `None` for 60 per second.
        view_camera (dict | str | None): The camera a 3-D view opens on, `None` for the fit.
    """
    __slots__ = ("_s",)

    def __init__(self, state: SceneState):
        self._s = state

    name = property(lambda self: self._s.name)
    space = property(lambda self: self._s.space)
    up = property(lambda self: self._s.up)
    units = property(lambda self: self._s.units)
    rate = property(lambda self: self._s.rate)
    view_camera = property(lambda self: self._s.view_camera)

    def __repr__(self) -> str:
        s = self._s
        return f"<caliper.Scene {s.name!r} {s.space} up={s.up} units={s.units!r}>"

    def _watched(self) -> bool:
        """Whether a viewer of a network route watches the scene now (lists the scene first)."""
        _live(self._s)
        return bool(_watching(self._s))

    # -- primitives ------------------------------------------------------------
    def points(self, path, positions, *, colors=None, values=None, cmap="viridis", vmin=None,
               vmax=None, class_ids=None, radii=None, labels=None, items=None, blend="normal",
               static=False, rate=None) -> bool:
        """Log a batch of points at entity `path`, replacing the points it held.

        Points are drawn as discs of a few pixels unless `radii` is given. Give at most one of
        `colors` and `values`. A static batch is the setup of the scene ([Static Items](scene-calls.md#static-items)).

        Args:
            path (str): The entity path: 0 to 16 segments of `[A-Za-z0-9_.-]{1,63}` joined by `/`, at
                most 255 bytes. The empty string is the root.
            positions (Tensor | ndarray | Sequence): Positions `(N, D)`, with D 3 in a 3-D scene and 2 in
                a 2-D scene or under a camera. At most 4,194,304 points.
            colors (Tensor | tuple | None): Colors `(N, 3)` or `(N, 4)` as uint8 or floats in [0, 1], or
                one color for all as a tuple of 3 or 4 integers from 0 to 255 or floats from 0 to 1. A
                missing alpha is 255.
            values (Tensor | None): Values `(N,)` drawn through `cmap`.
            cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.
            vmin (float | None): Fixed lower end of the value range. `None` computes that end on the
                device. Each end is independent.
            vmax (float | None): Fixed upper end of the value range, as for `vmin`.
            class_ids (Tensor | None): Integer class ids `(N,)`, colored by the class table of
                [`classes`][caliper.Scene.classes].
            radii (Tensor | float | None): Radii `(N,)` or one number above 0, in scene units.
            labels (str | Sequence[str] | None): One label for all points, or one per point. Each is at
                most 63 bytes, and at most 4,096 labels.
            items (None): Not built yet (picking and linked selection).
            blend (str): Only `"normal"`. Additive blending is not built yet.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene ([Rate Limits](scene-calls.md#rate-limits)).

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `colors` and `values` are given together, a
                shape, dtype or row count does not fit, a batch is over the limits, the entity cannot
                hold points, or the scene is at its limit of 1,024 entities or 2,048 batches.
            NotImplementedError: When `items` or `blend` is not its default.
        """
        _no(items=(items, None, "picking and linked selection"),
            blend=(blend, "normal", "additive blending"))
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_POINTS)
        pos = _check_rows(st, path, positions, "positions", _wire.MAX_BATCH_POINTS)
        n = pos.shape[0]
        color, ct, vt = _check_colors(colors, values, lambda r: r == n, "N")
        cm = _check_range_args(cmap, vmin, vmax)
        params, extra = {}, {}
        if radii is not None:
            if _real(radii):
                if not (math.isfinite(radii) and radii > 0):
                    raise ValueError(f"radii must be above 0, got {radii!r}")
                params["radius"] = float(radii)
            else:
                rt = _array(radii, "radii", "float")
                if rt.ndim != 1 or rt.shape[0] != n:
                    raise ValueError(f"radii must be (N,) or one number, got {tuple(rt.shape)}")
                extra["radii"] = rt
        if class_ids is not None:
            extra["class_ids"] = _check_ids(class_ids, n, "N")
        if labels is not None:
            params["labels"] = _check_labels(labels, n)
        if color is not None:
            params["color"] = color
        spec = _Spec(_wire.PART_POINTS, pos, ct, vt, cm, vmin, vmax, params, extra=extra)
        return _log(st, path, spec, static, iv)

    def lines(self, path, positions, *, indices=None, strip_lengths=None, colors=None,
              values=None, cmap="viridis", vmin=None, vmax=None, color_per=None, width=2.0,
              blend="normal", static=False, rate=None) -> bool:
        """Log a batch of lines at entity `path`, replacing the lines it held.

        `positions` holds the vertices. Without `indices` and `strip_lengths`, vertices 2k and 2k + 1
        form segment k. With `strip_lengths`, consecutive vertices form strips. With `indices`, each
        row is an edge between two vertices, as PyTorch Geometric's `edge_index.T`.

        Colors belong to the vertices, the strips or the edges. `color_per` says which; `None` reads it
        from the row count of `colors` or `values`, and a count that fits two raises `ValueError`.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            positions (Tensor | ndarray | Sequence): Vertices `(V, D)`, with D 3 in a 3-D scene and 2 in
                a 2-D scene or under a camera. At most 4,194,304 vertices. Segments need an even V.
            indices (Tensor | None): Integer index pairs `(E, 2)`, at most 4,194,304 edges. A `(2, E)`
                array raises `ValueError` and names the fix. Not together with `strip_lengths`.
            strip_lengths (Sequence[int] | Tensor | None): The vertices of each strip, `(S,)`, at least
                1 each and adding up to V, at most 1,048,576 strips. Lengths in a device tensor are
                checked once their copy reaches host memory; a batch that fails there is dropped with a
                warning, after the call returned `True`.
            colors (Tensor | tuple | None): Colors `(rows, 3)` or `(rows, 4)` as uint8 or floats in
                [0, 1], or one color for all, as for [`points`][caliper.Scene.points]. The rows are V,
                S or E.
            values (Tensor | None): Values `(rows,)` drawn through `cmap`. Not together with `colors`.
            cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.
            vmin (float | None): Fixed lower end of the value range. `None` computes that end on the
                device.
            vmax (float | None): Fixed upper end of the value range, as for `vmin`.
            color_per (str | None): `"vertex"`, `"strip"` or `"edge"`. `"strip"` needs `strip_lengths`
                and `"edge"` needs `indices`.
            width (float): Line width in pixels, above 0.
            blend (str): Only `"normal"`. Additive blending is not built yet.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `color_per` is unknown or does not fit the
                batch, `indices` and `strip_lengths` are given together, the vertex count is odd for
                segments, the strip lengths do not add up to V (when they are host data) or one is
                below 1, `colors` and `values` are given together, a shape, dtype or row count does not
                fit, a batch is over the limits, the entity cannot hold lines, or the scene is at its
                limit of 1,024 entities or 2,048 batches.
            NotImplementedError: When `blend` is not `"normal"`.
        """
        _no(blend=(blend, "normal", "additive blending"))
        if color_per not in (None, _wire.COLOR_PER_VERTEX, _wire.COLOR_PER_STRIP,
                             _wire.COLOR_PER_EDGE):
            raise ValueError(f"color_per must be 'vertex', 'strip', 'edge' or None, got "
                             f"{color_per!r}")
        if indices is not None and strip_lengths is not None:
            raise ValueError("indices and strip_lengths together: a batch holds segments, strips "
                             "or edges")
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_LINES)
        pos = _check_rows(st, path, positions, "positions", _wire.MAX_BATCH_POINTS)
        v = pos.shape[0]
        params = {"width": _check_width(width)}
        extra = {}
        colored = values is not None or (colors is not None and not _flat(colors))
        if indices is not None:
            it = _array(indices, "indices", "int")
            if it.ndim != 2 or it.shape[1] != 2:
                fix = (f"; pass edge_index.T, an (E, 2) array" if it.ndim == 2 and it.shape[0] == 2
                       else "")
                raise ValueError(f"indices has shape {tuple(it.shape)}{fix}" if fix else
                                 f"indices must be (E, 2), got {tuple(it.shape)}")
            e = it.shape[0]
            if e > _wire.MAX_BATCH_POINTS:
                raise ValueError(f"{e} edges, more than {_wire.MAX_BATCH_POINTS} in a batch")
            if color_per == _wire.COLOR_PER_STRIP:
                raise ValueError("color_per='strip' needs strip_lengths")
            if color_per is None and colored:
                r = _rows(colors if colors is not None else values)
                if r == v and r == e:
                    raise ValueError(f"{r} colour rows fit both the vertices and the edges; "
                                     "pass color_per='vertex' or 'edge'")
                color_per = _wire.COLOR_PER_EDGE if r == e else _wire.COLOR_PER_VERTEX
            params["mode"] = _wire.LINES_EDGES
            extra["indices"] = it
            if color_per == _wire.COLOR_PER_EDGE:
                params["color_per"] = _wire.COLOR_PER_EDGE
                rows_ok, rows = (lambda r: r == e), "E"
            else:
                rows_ok, rows = (lambda r: r == v), "V"
        elif strip_lengths is None:
            if v % 2:
                raise ValueError(f"segments need an even number of vertices, got {v}")
            if color_per in (_wire.COLOR_PER_STRIP, _wire.COLOR_PER_EDGE):
                raise ValueError(f"color_per={color_per!r} needs "
                                 f"{'strip_lengths' if color_per == 'strip' else 'indices'}")
            params["mode"] = _wire.LINES_SEGMENTS
            rows_ok, rows = (lambda r: r == v), "V"
        else:
            if color_per == _wire.COLOR_PER_EDGE:
                raise ValueError("color_per='edge' needs indices")
            if isinstance(strip_lengths, (list, tuple)) and all(
                    x.__class__ is int for x in strip_lengths):
                sl = strip_lengths                   # host ints: checked here
                s_count = len(sl)
                total, low = sum(sl), min(sl, default=1)
            else:
                sl = _array(strip_lengths, "strip_lengths", "int")
                if sl.ndim != 1:
                    raise ValueError(f"strip_lengths must be (S,), got {tuple(sl.shape)}")
                s_count = sl.shape[0]
                total = low = None
                # host data: the sum is checked here; a device tensor's once its copy is done
                # (_net_scene.finish drops the batch then)
                if sl.device.type == "cpu" and s_count:
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
            if color_per is None and colored:
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
        spec = _Spec(_wire.PART_LINES, pos, ct, vt, cm, vmin, vmax, params, extra=extra,
                     per_item=params.get("color_per") in (_wire.COLOR_PER_STRIP,
                                                           _wire.COLOR_PER_EDGE))
        return _log(st, path, spec, static, iv)

    def arrows(self, path, origins, vectors, *, colors=None, values=None, cmap="viridis",
               vmin=None, vmax=None, width=2.0, static=False, rate=None) -> bool:
        """Log a batch of arrows at entity `path`, replacing the arrows it held.

        Each arrow starts at an origin and runs along its vector. The head is a fifth of the length.
        Colors work as for [`points`][caliper.Scene.points].

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            origins (Tensor | ndarray | Sequence): Origins `(N, D)`, with D 3 in a 3-D scene and 2 in a
                2-D scene or under a camera. At most 1,048,576 arrows.
            vectors (Tensor | ndarray | Sequence): Vectors `(N, D)`, with the shape of `origins`.
            colors (Tensor | tuple | None): Colors `(N, 3)` or `(N, 4)`, or one color for all.
            values (Tensor | None): Values `(N,)` drawn through `cmap`. Not together with `colors`.
            cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.
            vmin (float | None): Fixed lower end of the value range. `None` computes that end on the
                device.
            vmax (float | None): Fixed upper end of the value range, as for `vmin`.
            width (float): Arrow width in pixels, above 0.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `vectors` does not have the shape of
                `origins`, `colors` and `values` are given together, a shape, dtype or row count does
                not fit, a batch is over the limits, the entity cannot hold arrows, or the scene is at
                its limit of 1,024 entities or 2,048 batches.
        """
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_ARROWS)
        o = _check_rows(st, path, origins, "origins", _wire.MAX_BATCH_ITEMS)
        vec = _check_rows(st, path, vectors, "vectors", _wire.MAX_BATCH_ITEMS)
        n = o.shape[0]
        if vec.shape[0] != n:
            raise ValueError(f"vectors must be ({n}, {o.shape[1]}), like origins, got "
                             f"{tuple(vec.shape)}")
        color, ct, vt = _check_colors(colors, values, lambda r: r == n, "N")
        cm = _check_range_args(cmap, vmin, vmax)
        params = {"width": _check_width(width)}
        if color is not None:
            params["color"] = color
        spec = _Spec(_wire.PART_ARROWS, o, ct, vt, cm, vmin, vmax, params,
                     extra={"vectors": vec}, first="origins")
        return _log(st, path, spec, static, iv)

    def boxes(self, path, centers=None, half_sizes=None, *, corners=None, rotations=None,
              colors=None, class_ids=None, labels=None, fill=False, width=2.0, items=None,
              static=False, rate=None) -> bool:
        """Log a batch of boxes at entity `path`, replacing the boxes it held.

        Give `centers` and `half_sizes`, or in 2-D `corners`. A 3-D box may be rotated. Edges are drawn
        with `width`, and `fill` adds translucent faces.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            centers (Tensor | ndarray | Sequence | None): Centers `(N, D)`, with D 3 in a 3-D scene and
                2 in a 2-D scene or under a camera. At most 1,048,576 boxes.
            half_sizes (Tensor | ndarray | Sequence | None): Half sizes `(N, D)`, with the shape of
                `centers`.
            corners (Tensor | None): 2-D boxes `(N, 4)` as `(x0, y0, x1, y1)`, converted to centers and
                half sizes on the device. Replaces `centers` and `half_sizes`.
            rotations (Tensor | None): Quaternions `(N, 4)` as x y z w, for 3-D boxes only.
            colors (Tensor | tuple | None): Colors `(N, 3)` or `(N, 4)`, or one color for all.
            class_ids (Tensor | None): Integer class ids `(N,)`, colored by the class table of
                [`classes`][caliper.Scene.classes].
            labels (str | Sequence[str] | None): One label for all boxes, or one per box. Each is at
                most 63 bytes, and at most 4,096 labels.
            fill (bool): Also draw translucent faces.
            width (float): Edge width in pixels, above 0.
            items (None): Not built yet (picking and linked selection).
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, `fill` is not a bool,
                neither `corners` nor both `centers` and `half_sizes` are given, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `corners` is given together with `centers`
                or `half_sizes` or in a 3-D space, `rotations` is given in 2-D or does not fit, a
                shape, dtype or row count does not fit, a batch is over the limits, the entity cannot
                hold boxes, or the scene is at its limit of 1,024 entities or 2,048 batches.
            NotImplementedError: When `items` is given.
        """
        _no(items=(items, None, "picking and linked selection"))
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_BOXES)
        d = st.dims_at(path)
        if corners is not None:
            if centers is not None or half_sizes is not None:
                raise ValueError("corners replace centers and half_sizes: give one or the other")
            if d != 2:
                raise ValueError("corners give 2-D boxes; a 3-D box takes centers and half_sizes")
            ct4 = _array(corners, "corners", "float")
            if ct4.ndim != 2 or ct4.shape[1] != 4:
                raise ValueError(f"corners must be (N, 4) as (x0, y0, x1, y1), got "
                                 f"{tuple(ct4.shape)}")
            if ct4.shape[0] > _wire.MAX_BATCH_ITEMS:
                raise ValueError(f"{ct4.shape[0]} boxes, more than {_wire.MAX_BATCH_ITEMS} in a "
                                 "batch")
            c, h = None, None
            n = ct4.shape[0]
        else:
            if centers is None or half_sizes is None:
                raise TypeError("boxes takes centers and half_sizes, or corners")
            c = _check_rows(st, path, centers, "centers", _wire.MAX_BATCH_ITEMS)
            h = _check_rows(st, path, half_sizes, "half_sizes", _wire.MAX_BATCH_ITEMS)
            n = c.shape[0]
            if h.shape[0] != n:
                raise ValueError(f"half_sizes must be ({n}, {d}), like centers, got "
                                 f"{tuple(h.shape)}")
            ct4 = None
        extra = {}
        if rotations is not None:
            if d != 3:
                raise ValueError("rotations are for 3-D boxes only")
            rt = _array(rotations, "rotations", "float")
            if rt.ndim != 2 or rt.shape != (n, 4):
                raise ValueError(f"rotations must be ({n}, 4) quaternions x y z w, got "
                                 f"{tuple(rt.shape)}")
            extra["rotations"] = rt
        color, cc, _ = _check_colors(colors, None, lambda r: r == n, "N")
        if class_ids is not None:
            extra["class_ids"] = _check_ids(class_ids, n, "N")
        params = {"width": _check_width(width)}
        if not isinstance(fill, bool):
            raise TypeError(f"fill must be True or False, got {fill!r}")
        if fill:
            params["fill"] = True
        if labels is not None:
            params["labels"] = _check_labels(labels, n)
        if color is not None:
            params["color"] = color
        spec = _Spec(_wire.PART_BOXES, c, cc, None, 0, None, None, params, extra=extra,
                     first="centers", half_sizes=h, corners=ct4)
        return _log(st, path, spec, static, iv)

    def mesh(self, path, positions, indices=None, *, normals=None, colors=None, values=None,
             cmap="viridis", vmin=None, vmax=None, uvs=None, texture=None, instance_poses=None,
             instance_colors=None, instance_values=None, static=False, rate=None) -> bool:
        """Log a triangle mesh at entity `path`, replacing the mesh it held.

        Without `indices`, each three vertices form a triangle. Faces are flat without `normals`. The
        color comes from one of `colors`, `values`, or `uvs` with a `texture`. An index out of range
        draws as the last vertex.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            positions (Tensor | ndarray | Sequence): Vertices `(V, D)`, with D 3 in a 3-D scene and 2 in
                a 2-D scene or under a camera. At most 4,194,304 vertices. Without `indices`, V is a
                multiple of 3.
            indices (Tensor | None): Triangles `(T, 3)` in any integer dtype, at most 4,194,304.
            normals (Tensor | None): Per-vertex normals `(V, 3)` for smooth shading.
            colors (Tensor | tuple | None): Per-vertex colors `(V, 3)` or `(V, 4)`, or one color for
                all.
            values (Tensor | None): Per-vertex values `(V,)` drawn through `cmap`.
            cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.
            vmin (float | None): Fixed lower end of the value range. `None` computes that end on the
                device.
            vmax (float | None): Fixed upper end of the value range, as for `vmin`.
            uvs (Tensor | None): Texture coordinates `(V, 2)`. Given together with `texture`, and
                instead of `colors` and `values`.
            texture (Tensor | None): A texture `(H, W, 3)` or `(H, W, 4)`, uint8 or floats in [0, 1],
                at most 8,192 pixels on a side.
            instance_poses (None): Not built yet (instanced meshes).
            instance_colors (None): Not built yet (instanced meshes).
            instance_values (None): Not built yet (instanced meshes).
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, a mesh without `indices` has a vertex count
                that is not a multiple of 3, `uvs` and `texture` are not given together or are given
                with `colors` or `values`, `colors` and `values` are given together, a shape, dtype or
                row count does not fit, a batch is over the limits, the entity cannot hold a mesh, or
                the scene is at its limit of 1,024 entities or 2,048 batches.
            NotImplementedError: When an instance argument is given.
        """
        _no(instance_poses=(instance_poses, None, "instanced meshes"),
            instance_colors=(instance_colors, None, "instanced meshes"),
            instance_values=(instance_values, None, "instanced meshes"))
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_MESH)
        pos = _check_rows(st, path, positions, "positions", _wire.MAX_BATCH_POINTS)
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
        if normals is not None:
            nt = _array(normals, "normals", "float")
            if nt.ndim != 2 or nt.shape != (v, 3):
                raise ValueError(f"normals must be ({v}, 3), got {tuple(nt.shape)}")
            extra["normals"] = nt
        if (uvs is None) != (texture is None):
            raise ValueError("uvs and texture go together")
        if uvs is not None:
            if colors is not None or values is not None:
                raise ValueError("a mesh takes one of colors, values and uvs with a texture")
            ut = _array(uvs, "uvs", "float")
            if ut.ndim != 2 or ut.shape != (v, 2):
                raise ValueError(f"uvs must be ({v}, 2), got {tuple(ut.shape)}")
            tt = _check_color_image(texture, "texture", gray=False)
            extra["uvs"] = ut
            extra["texture"] = tt
        color, ct, vt = _check_colors(colors, values, lambda r: r == v, "V")
        cm = _check_range_args(cmap, vmin, vmax)
        params = {"color": color} if color is not None else {}
        spec = _Spec(_wire.PART_MESH, pos, ct, vt, cm, vmin, vmax, params, extra=extra)
        return _log(st, path, spec, static, iv)

    # -- images ------------------------------------------------------------------
    def image(self, path, image, *, opacity=1.0, static=False, rate=None) -> bool:
        """Log a color image at entity `path`, replacing the image it held.

        In a 3-D scene an image sits under a camera, in its pixel space, so log the
        [`camera`][caliper.Scene.camera] at a parent of `path` first.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            image (Tensor | ndarray | bytes): A tensor `(H, W, 3)`, `(H, W, 4)` or `(H, W)` as uint8 or
                floats in [0, 1] (floats are converted on the device), at most 8,192 pixels on a side.
                Or the bytes of one JPEG or PNG frame, recognised by its signature, of at most 16 MiB.
            opacity (float): From 0 to 1.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `opacity` is outside 0 to 1, the bytes are
                not one JPEG or PNG frame or are over 16 MiB, the shape or dtype does not fit, the
                image is over 8,192 pixels on a side, the entity cannot hold an image (in a 3-D scene,
                one that is not under a camera), or the scene is at its limit of 1,024 entities or
                2,048 batches.
        """
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_IMAGE)
        params = {"opacity": _check_opacity(opacity)}
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            form = next((f for sig, f in IMAGE_FORMS.items() if data.startswith(sig)), None)
            if form is None:
                raise ValueError("encoded image bytes must be one JPEG or PNG frame")
            if len(data) > _wire.MAX_ENCODED_BYTES:
                raise ValueError(f"an encoded frame of {len(data)} bytes, more than "
                                 f"{_wire.MAX_ENCODED_BYTES}")
            params.update(form=_wire.FORM_ENCODED, format=form)
            spec = _Spec(_wire.PART_IMAGE, None, None, None, 0, None, None, params,
                         encoded=data)
            return _log(st, path, spec, static, iv)
        t = _check_color_image(image, "image", gray=True)
        params["form"] = _wire.FORM_COLOR
        spec = _Spec(_wire.PART_IMAGE, None, None, None, 0, None, None, params,
                     extra={"pixels": t})
        return _log(st, path, spec, static, iv)

    def depth(self, path, depth, *, scale=None, cmap="viridis", vmin=None, vmax=None, cloud=None,
              static=False, rate=None) -> bool:
        """Log a depth image at entity `path`, replacing the image it held.

        A depth image is an image part, so it replaces a color image at the same path, and it sits
        under a camera in a 3-D scene as an image does.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            depth (Tensor | ndarray): Depths `(H, W)` as floats (cast to float32 on the device) or
                uint16, at most 8,192 pixels on a side.
            scale (float | None): Scene units per stored unit, above 0. `None` is 1 for floats and
                0.001 for uint16, millimetres as metres.
            cmap (str | int): `"viridis"`, `"magma"` or `"rdbu"`, or 0, 1 or 2.
            vmin (float | None): Fixed lower end of the range, in scene units. `None` computes that end
                on the device.
            vmax (float | None): Fixed upper end of the range, as for `vmin`.
            cloud (bool | None): Also draw the back-projected point cloud. `None` draws it under a
                camera.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, `cloud` is not a bool or
                `None`, or an argument has the wrong type.
            ValueError: When `path` is not an entity path, `depth` is not floats or uint16 or not
                `(H, W)`, is over 8,192 pixels on a side, `scale` is not above 0, the entity cannot
                hold an image, or the scene is at its limit of 1,024 entities or 2,048 batches.
        """
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_IMAGE)
        torch = _TORCH or _torch()
        t = _array(depth, "depth", "any")
        u16 = t.dtype == torch.uint16
        if not u16 and t.dtype not in _float_dtypes():
            raise ValueError(f"depth must hold floats or uint16, got {t.dtype}")
        _check_side(t, "depth", lambda x: x.ndim == 2)
        if scale is None:
            scale = 0.001 if u16 else 1.0
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or \
                not (math.isfinite(scale) and scale > 0):
            raise ValueError(f"scale must be a number above 0, got {scale!r}")
        if cloud is None:
            cloud = st.under_camera(path)
        if not isinstance(cloud, bool):
            raise TypeError(f"cloud must be True, False or None, got {cloud!r}")
        cm = _check_range_args(cmap, vmin, vmax)
        params = {"form": _wire.FORM_DEPTH, "scale": float(scale), "cloud": cloud}
        spec = _Spec(_wire.PART_IMAGE, None, None, None, cm, vmin, vmax, params,
                     extra={"pixels": t}, depth=True)
        return _log(st, path, spec, static, iv)

    def segmentation(self, path, class_ids, *, opacity=0.5, static=False, rate=None) -> bool:
        """Log a segmentation image at entity `path`, replacing the image it held.

        Each pixel is a class id, colored by the class table of [`classes`][caliper.Scene.classes]. Ids
        travel as uint8 when the image is uint8 and as uint16 otherwise; an id outside 0 to 65,535 is
        clamped. The image sits under a camera in a 3-D scene.

        Args:
            path (str): The entity path, as for [`points`][caliper.Scene.points].
            class_ids (Tensor | ndarray): Integer class ids `(H, W)`, at most 8,192 pixels on a side.
            opacity (float): From 0 to 1.
            static (bool): Copy the batch once at the call and retain it for every viewer.
            rate (float | None): Snapshots per second for this batch. `None` takes the rate of the
                scene.

        Returns:
            bool: `True` if a route took a snapshot, `False` if every route skipped it.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or an argument has the
                wrong type.
            ValueError: When `path` is not an entity path, `class_ids` does not hold integers or is not
                `(H, W)`, is over 8,192 pixels on a side, `opacity` is outside 0 to 1, the entity
                cannot hold an image, or the scene is at its limit of 1,024 entities or 2,048 batches.
        """
        st = self._s
        _check_path(path)
        iv = _check_static_rate(static, rate)
        _check_holds(st, path, _wire.PART_IMAGE)
        torch = _TORCH or _torch()
        t = _array(class_ids, "class_ids", "any")
        if t.dtype.is_floating_point or t.dtype.is_complex or t.dtype == torch.bool:
            raise ValueError(f"class_ids must hold integers, got {t.dtype}")
        _check_side(t, "class_ids", lambda x: x.ndim == 2)
        params = {"form": _wire.FORM_SEGMENTATION, "opacity": _check_opacity(opacity)}
        spec = _Spec(_wire.PART_IMAGE, None, None, None, 0, None, None, params,
                     extra={"pixels": t}, segmentation=True)
        return _log(st, path, spec, static, iv)

    # -- cameras, entity transforms, the class table, clears ----------------------
    def camera(self, path, *, width, height, focal=None, fov=None, principal=None,
               convention="rdf", plane_distance=None, static=False) -> bool:
        """Log a pinhole camera at entity `path`.

        The entity transform of `path` poses the camera ([`transform`][caliper.Scene.transform]). The
        entity holds no primitive, the root holds no camera, and a camera is never under another. Its
        children live in its pixel space, where images and 2-D points, lines, arrows and boxes sit.
        The producer keeps the camera whether or not a viewer is attached, and the rate limit never
        drops it.

        Args:
            path (str): The entity path of the camera, not the root.
            width (int): Image width in pixels, from 1 to 8,192.
            height (int): Image height in pixels, from 1 to 8,192.
            focal (float | tuple[float, float] | None): `fx` and `fy` in pixels, or one value for both.
                Give exactly one of `focal` and `fov`.
            fov (float | None): The vertical field of view in degrees, between 0 and 180, which sets
                `fx = fy`.
            principal (tuple[float, float] | None): `(cx, cy)` in pixels. `None` is the image centre.
            convention (str): `"rdf"` (x right, y down, z forward) or `"rub"` (x right, y up, looking
                along -z).
            plane_distance (float | None): How far in front of the camera its image plane is drawn, in
                scene units, above 0.
            static (bool): Retain the camera as part of the scene setup, which a later `clear` removes
                only when it is static too.

        Returns:
            bool: Always `True`.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, `width` or `height` is not
                an int, or `focal` or `principal` has the wrong type.
            ValueError: When `path` is not an entity path or is the root, the entity holds primitives
                or lies under a camera, `width` or `height` is outside 1 to 8,192, both or neither of
                `focal` and `fov` are given, `fov`, `focal`, `principal` or `plane_distance` is out of
                range, `convention` is unknown, or the scene is at its limit of 1,024 entities or
                2,048 batches.
        """
        st = self._s
        _check_path(path)
        _check_static(static)
        _check_holds(st, path, _wire.PART_CAMERA)
        for name, v in (("width", width), ("height", height)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} must be an int, got {v!r}")
            if not 1 <= v <= _wire.MAX_IMAGE_SIDE:
                raise ValueError(f"{name} is 1 to {_wire.MAX_IMAGE_SIDE} pixels, got {v}")
        if (focal is None) == (fov is None):
            raise ValueError("give exactly one of focal and fov")
        if fov is not None:
            if not _real(fov) or not (math.isfinite(fov) and 0 < fov < 180):
                raise ValueError(f"fov is the vertical field of view, between 0 and 180 degrees, "
                                 f"got {fov!r}")
            fx = fy = (height / 2.0) / math.tan(math.radians(fov) / 2.0)
        else:
            pair = (focal, focal) if _real(focal) else tuple(focal) if isinstance(
                focal, (list, tuple)) else None
            if pair is None or len(pair) != 2 or not all(_real(x) for x in pair):
                raise TypeError(f"focal is one number or a pair (fx, fy), got {focal!r}")
            if not all(math.isfinite(x) and x > 0 for x in pair):
                raise ValueError(f"focal lengths are above 0, got {focal!r}")
            fx, fy = (float(x) for x in pair)
        if principal is None:
            cx, cy = width / 2.0, height / 2.0
        else:
            if not isinstance(principal, (list, tuple)) or len(principal) != 2 or \
                    not all(_real(x) for x in principal):
                raise TypeError(f"principal is a pair (cx, cy), got {principal!r}")
            if not all(math.isfinite(x) for x in principal):
                raise ValueError(f"principal must be finite, got {principal!r}")
            cx, cy = (float(x) for x in principal)
        if convention not in (_wire.CONVENTION_RDF, _wire.CONVENTION_RUB):
            raise ValueError(f"convention is 'rdf' or 'rub', got {convention!r}")
        params = {"fx": float(fx), "fy": float(fy), "cx": cx, "cy": cy, "width": width,
                  "height": height, "convention": convention}
        if plane_distance is not None:
            if not _real(plane_distance) or not (math.isfinite(plane_distance) and
                                                  plane_distance > 0):
                raise ValueError(f"plane_distance must be above 0, got {plane_distance!r}")
            params["plane_distance"] = float(plane_distance)
        _live(st)
        _admit(st, path, _wire.PART_CAMERA)
        b = Batch(st.name, path, _wire.PART_CAMERA, st.next_seq(), time.monotonic_ns(), static,
                  params, {}, None)
        b.arrays = {}
        st.cameras[path] = b
        for sr in _watching(st):
            sr.queue(b)
        st.count("cameras")
        return True

    def transform(self, path, *, translation=None, rotation=None, scale=None, matrix=None,
                  static=False) -> bool:
        """Set the entity transform of `path`, mapping its space into the space of its parent.

        The parts `translation`, `rotation` and `scale` are applied as T R S. Or give `matrix`, which
        excludes the parts. Each is numbers, an array or a tensor on any device; the transform of
        tensors is composed on their device and the transform of numbers on the host. A call with none
        of them resets the transform to the identity. The producer keeps the transform whether or not a
        viewer is attached, and the rate limit never drops it.

        Args:
            path (str): The entity path, not the root.
            translation (Sequence[float] | Tensor | None): 3 numbers in 3-D, 2 in 2-D and under a
                camera.
            rotation (Sequence[float] | Tensor | None): In 3-D a quaternion x y z w, normalized by the
                call. In 2-D and under a camera one angle in radians.
            scale (float | Sequence[float] | Tensor | None): 1 number for all axes, or one per axis
                (3 in 3-D, 2 in 2-D and under a camera).
            matrix (Sequence | Tensor | None): A `(4, 4)` matrix in the usual row and column sense.
            static (bool): Retain the transform as part of the scene setup, which a later `clear`
                removes only when it is static too.

        Returns:
            bool: Always `True`.

        Raises:
            TypeError: When `path` is not a string, `static` is not a bool, or a part is not numbers,
                an array or a tensor.
            ValueError: When `path` is not an entity path or is the root, `matrix` is given together
                with a part, a part or `matrix` has the wrong size or holds a type that is not a
                number, or the scene is at its limit of 1,024 entities.
        """
        st = self._s
        _check_path(path)
        _check_static(static)
        if path == "":
            raise ValueError("the root holds no entity transform: it is the scene's own space")
        d = st.dims_at(path)
        if matrix is not None and (translation is not None or rotation is not None or
                                   scale is not None):
            raise ValueError("matrix excludes translation, rotation and scale")
        m, event = _compose(d, translation, rotation, scale, matrix)
        _live(st)
        if path not in st.entities and len(st.entities) >= _wire.MAX_ENTITIES:
            raise ValueError(f"scene {st.name!r} holds {_wire.MAX_ENTITIES} entities, its limit")
        st.entities.add(path)
        tr = Transform(st.name, path, st.next_seq(), time.monotonic_ns(), static, m, event)
        st.transforms[path] = tr
        for sr in _watching(st):
            sr.queue(tr)
        st.count("transforms")
        return True

    def classes(self, table, *, static=True) -> bool:
        """Replace the class table of the scene.

        Points and boxes with `class_ids` and segmentation images are colored by it. An id without a
        color takes color `id % 16` of the categorical palette. The producer keeps the table whether or
        not a viewer is attached, and the rate limit never drops it.

        Args:
            table (dict): Maps a class id, an int from 0 to 65,535, to a label (at most 63 bytes), or to
                `(label, color)`. A color is 3 or 4 integers from 0 to 255 or floats from 0 to 1. At
                most 4,096 entries.
            static (bool): Retain the table as part of the scene setup, which is the default.

        Returns:
            bool: Always `True`.

        Raises:
            TypeError: When `table` is not a dict, `static` is not a bool, or a class id is not an int
                or a value is neither a label nor `(label, color)`.
            ValueError: When the table has more than 4,096 classes, a class id is outside 0 to 65,535,
                a label is over 63 bytes, or a color is not 3 or 4 numbers in range.
        """
        st = self._s
        _check_static(static)
        if not isinstance(table, dict):
            raise TypeError(f"table must be a dict of class id to a label or (label, colour), "
                            f"got {type(table).__name__}")
        if len(table) > _wire.MAX_CLASSES:
            raise ValueError(f"{len(table)} classes, more than {_wire.MAX_CLASSES}")
        out = []
        for cid, v in table.items():
            if isinstance(cid, bool) or not isinstance(cid, int):
                raise TypeError(f"a class id is an int, got {cid!r}")
            if not 0 <= cid <= _wire.MAX_CLASS_ID:
                raise ValueError(f"a class id is 0 to {_wire.MAX_CLASS_ID}, got {cid}")
            if isinstance(v, str):
                label, color = v, None
            elif isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], str):
                label, color = v[0], _one_colour(v[1])
            else:
                raise TypeError(f"class {cid}: a label or (label, colour), got {v!r}")
            if len(label.encode("utf-8")) > _wire.NAME_MAX:
                raise ValueError(f"class {cid}: a label is at most {_wire.NAME_MAX} bytes")
            if color is None:
                color = list(PALETTE[cid % 16]) + [255]
            out.append({"id": cid, "label": label, "color": color})
        out.sort(key=lambda e: e["id"])
        _live(st)
        k = Classes(st.name, st.next_seq(), time.monotonic_ns(), static, out)
        st.classes = k
        for sr in _watching(st):
            sr.queue(k)
        st.count("classes")
        return True

    def clear(self, path="", *, recursive=False, static=False) -> bool:
        """Clear entity `path`: its primitives, camera and entity transform.

        With `recursive`, the subtree of `path` is cleared too, and `clear("", recursive=True)` empties
        the live data of the scene. Static parts stay unless the clear is static too. A later log at
        the path brings a part back. The clear applies at once to what the producer keeps, and the rate
        limit never drops it.

        Args:
            path (str): The entity path. The empty string is the root.
            recursive (bool): Also clear the subtree.
            static (bool): Also clear static parts.

        Returns:
            bool: Always `True`.

        Raises:
            TypeError: When `path` is not a string, or `recursive` or `static` is not a bool.
            ValueError: When `path` is not an entity path.
        """
        st = self._s
        _check_path(path)
        _check_static(static)
        if not isinstance(recursive, bool):
            raise TypeError(f"recursive must be True or False, got {recursive!r}")
        _live(st)
        c = Clear(st.name, st.next_seq(), time.monotonic_ns(), path, recursive, static)
        _apply_clear(st, c)
        for sr in _watching(st):
            sr.queue(c)
        st.count("clears")
        return True


def _check_ids(ids, n: int, what: str):
    t = _array(ids, "class_ids", "int")
    if t.ndim != 1 or t.shape[0] != n:
        raise ValueError(f"class_ids must be ({what},), got {tuple(t.shape)}")
    return t


def _check_color_image(x, what: str, gray: bool):
    """A colour image (H, W, 3 or 4), or (H, W) when ``gray``: u8, or floats
    in [0, 1]."""
    torch = _TORCH or _torch()
    t = _array(x, what, "any")
    if t.dtype != torch.uint8 and t.dtype not in _float_dtypes():
        raise ValueError(f"{what} must be uint8 or floats in [0, 1], got {t.dtype}")
    _check_side(t, what, lambda u: (u.ndim == 3 and u.shape[2] in (3, 4)) or
                (gray and u.ndim == 2))
    return t


def _apply_clear(st: SceneState, c: Clear) -> None:
    """A clear removes what the producer keeps of the entities it reaches:
    stamped parts always, static ones only when the clear is static."""
    for key in [k for k, b in st.static.items() if c.removes(k[0], True)]:
        del st.static[key]
    for path in [p for p, b in st.cameras.items() if c.removes(p, b.static)]:
        del st.cameras[path]
    for path in [p for p, t in st.transforms.items() if c.removes(p, t.static)]:
        del st.transforms[path]
    for path in [p for p in st.kinds if c.reaches(p)]:
        parts = st.kinds[path]
        keep = {part for part in parts if (path, part) in st.static or
                (part == _wire.PART_CAMERA and path in st.cameras)}
        if keep:
            st.kinds[path] = keep
        else:
            del st.kinds[path]
    for key in [k for k in st.batches if c.reaches(k[0]) and k not in st.static and
                not (k[1] == _wire.PART_CAMERA and k[0] in st.cameras)]:
        st.batches.discard(key)


def _admit(st: SceneState, path: str, part: str) -> None:
    """Count a new entity part against the scene's limits."""
    key = (path, part)
    if key not in st.batches:
        if path not in st.entities and len(st.entities) >= _wire.MAX_ENTITIES:
            raise ValueError(f"scene {st.name!r} holds {_wire.MAX_ENTITIES} entities, its limit")
        if len(st.batches) >= _wire.MAX_BATCHES:
            raise ValueError(f"scene {st.name!r} holds {_wire.MAX_BATCHES} batches, its limit")
        st.entities.add(path)
        st.batches.add(key)
    st.kinds.setdefault(path, set()).add(part)


def _compose(d: int, translation, rotation, scale, matrix):
    """The column-major 4 x 4 matrix of an entity transform as (16,) f32, on
    the host for numbers and arrays, on the device for a tensor (a pinned copy
    and an event). Returns (matrix, event or None)."""
    torch = _TORCH or _torch()
    given = [x for x in (translation, rotation, scale, matrix) if x is not None]
    dev = next((x.device for x in given if isinstance(x, torch.Tensor) and
                x.device.type != "cpu"), None)
    if dev is not None and dev.type != "mps":
        dev = None                                   # macOS: other devices go through the host
    target = dev if dev is not None else torch.device("cpu")

    def vec(x, what, sizes):
        if isinstance(x, torch.Tensor):
            t = (x.detach() if x.requires_grad else x)
            if t.dtype not in _float_dtypes() and (t.dtype.is_complex or t.dtype == torch.bool):
                raise ValueError(f"{what} must hold numbers, got {t.dtype}")
            t = t.to(device=target, dtype=torch.float32)
        else:
            if _real(x):
                x = [x]
            try:
                t = torch.as_tensor(x, dtype=torch.float32, device=target)
            except (TypeError, ValueError, RuntimeError) as e:
                raise TypeError(f"{what} must be numbers, an array or a tensor: {e}") from None
        if t.ndim > 1:
            t = t.reshape(-1) if t.numel() in sizes and t.ndim == 2 and 1 in t.shape else t
        if t.ndim != 1 or t.shape[0] not in sizes:
            raise ValueError(f"{what} must be {' or '.join(map(str, sizes))} numbers, got shape "
                             f"{tuple(t.shape)}")
        return t

    if matrix is not None:
        if isinstance(matrix, torch.Tensor):
            m = matrix.detach() if matrix.requires_grad else matrix
            m = m.to(device=target, dtype=torch.float32)
        else:
            try:
                m = torch.as_tensor(matrix, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as e:
                raise TypeError(f"matrix must be a (4, 4) array or tensor: {e}") from None
        if tuple(m.shape) != (4, 4):
            raise ValueError(f"matrix must be (4, 4), got {tuple(m.shape)}")
        out = m.t().reshape(16)
    else:
        out = torch.eye(4, dtype=torch.float32, device=target)
        if d == 3:
            t = vec(translation, "translation", (3,)) if translation is not None else None
            q = vec(rotation, "rotation", (4,)) if rotation is not None else None
            s = vec(scale, "scale", (1, 3)) if scale is not None else None
        else:
            t = vec(translation, "translation", (2,)) if translation is not None else None
            q = vec(rotation, "rotation", (1,)) if rotation is not None else None
            s = vec(scale, "scale", (1, 2)) if scale is not None else None
        r = torch.eye(3, dtype=torch.float32, device=target)
        if q is not None and d == 3:
            q = q / torch.linalg.vector_norm(q).clamp_min(1e-30)
            x, y, z, w = q[0], q[1], q[2], q[3]
            r = torch.stack([
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])
        elif q is not None:
            c, sn = torch.cos(q[0]), torch.sin(q[0])
            zero, one = torch.zeros_like(c), torch.ones_like(c)
            r = torch.stack([torch.stack([c, -sn, zero]), torch.stack([sn, c, zero]),
                             torch.stack([zero, zero, one])])
        if s is not None:
            sv = s.expand(d) if s.shape[0] == 1 else s
            if d == 2:
                sv = torch.cat([sv, torch.ones(1, dtype=torch.float32, device=target)])
            r = r * sv.reshape(1, 3)
        out[:3, :3] = r
        if t is not None:
            out[:d, 3] = t
        out = out.t().reshape(16)
    if out.device.type == "cpu":
        return out.contiguous().clone(), None
    host = torch.empty(16, dtype=torch.float32, pin_memory=True)
    host.copy_(out, non_blocking=True)
    event = torch.mps.Event()
    event.record()
    return host, event


@dataclass(eq=False)
class _Spec:
    """A checked primitive call: what the snapshot needs, all host metadata.
    ``positions`` is the first array of the part (``first`` names it on the
    wire); ``extra`` the part's other arrays."""
    part: str
    positions: Any
    colors: Any
    values: Any
    cmap: int
    vmin: Any
    vmax: Any
    params: dict
    extra: dict = field(default_factory=dict)
    first: str = "positions"
    per_item: bool = False                # lines: colours per strip or per edge
    half_sizes: Any = None                # boxes
    corners: Any = None                   # 2-D boxes given as corners
    encoded: bytes | None = None          # an encoded image
    depth: bool = False
    segmentation: bool = False


# ---------------------------------------------------------------------------
# the call: limits, rate, snapshot
# ---------------------------------------------------------------------------
def _frame_bytes(spec: _Spec) -> int:
    """The bytes of the batch's arrays on the wire, from their shapes, after
    the default data view's budget."""
    if spec.encoded is not None:
        return len(spec.encoded)
    out = 0
    if spec.positions is not None:
        n, d = spec.positions.shape
        rows = n
        if spec.part == _wire.PART_POINTS:
            rows = min(n, _wire.DEFAULT_MAX_POINTS)
        elif spec.part == _wire.PART_LINES and spec.params["mode"] == _wire.LINES_SEGMENTS:
            rows = 2 * min(n // 2, _wire.SCENE_DEFAULT_MAX_ITEMS)
        out += rows * d * 4
    else:
        rows = (spec.corners.shape[0] if spec.corners is not None else 0)
        out += rows * 16
    for t in (spec.colors, spec.values):
        if t is not None:
            out += (t.shape[0] if spec.per_item else rows) * 4       # RGBA u8 or f32
    if spec.half_sizes is not None:
        out += spec.half_sizes.numel() * 4
    for name, t in spec.extra.items():
        if isinstance(t, (list, tuple)):
            out += len(t) * 4
        elif name == "radii":
            out += rows * 4
        elif name in ("pixels", "texture"):
            h, w = t.shape[0], t.shape[1]
            out += h * w * (4 if t.ndim == 3 else 2)
        else:
            out += t.numel() * 4
    return out


def _log(st: SceneState, path: str, spec: _Spec, static: bool, iv) -> bool:
    _live(st)
    nbytes = _frame_bytes(spec)
    if nbytes + META_MARGIN > _wire.MAX_FRAME_BYTES:
        if nbytes > _wire.MAX_UPDATE_BYTES:
            raise ValueError(f"this batch needs {nbytes} bytes, more than the "
                             f"{_wire.MAX_UPDATE_BYTES >> 20} MiB of a split update")
    key = (path, spec.part)
    _admit(st, path, spec.part)
    watching = _watching(st)
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
    b.step = _run_step()
    if static:
        st.static[key] = b
    for sr in watching:
        sr.queue(b)
    st.count("taken")
    return True


def _run_step():
    from ._net_ring import _run_step as run_step
    return run_step()


def _stride(count: int, size: int, device):
    from ._dataview import _stride as stride
    return stride(count, size, device)


def _rgba(t, torch, image: bool = False):
    """Colours (N, 3 or 4) or a colour image (H, W, 3 or 4) as RGBA u8, on its
    device: floats in [0, 1] scaled, a missing alpha 255. A grey image (H, W)
    stays as it is."""
    f32, u8 = torch.float32, torch.uint8
    if t.dtype != u8:
        t = (t.to(f32).clamp(0.0, 1.0) * 255.0 + 0.5).to(u8)
    if t.shape[-1] == 3 and t.ndim == (3 if image else 2):
        pad = torch.full(tuple(t.shape[:-1]) + (1,), 255, dtype=u8, device=t.device)
        t = torch.cat([t, pad], -1)
    return t


def _snapshot(st: SceneState, path: str, spec: _Spec, static: bool) -> Batch:
    """Build the batch on the arrays' devices and queue its copies into host
    memory. Never waits for a device."""
    torch = _TORCH or _torch()
    f32, i32, u8 = torch.float32, torch.int32, torch.uint8
    params = dict(spec.params)
    arrays: dict = {}
    cast: dict = {}
    axes = None
    stats = None
    if spec.encoded is not None:                     # host bytes: no device work
        import numpy as np
        st.seq += 1
        b = Batch(st.name, path, spec.part, st.seq, time.monotonic_ns(), static, params,
                  {}, None, nbytes=len(spec.encoded))
        b.arrays = {"pixels": np.frombuffer(spec.encoded, np.uint8)}
        return b
    pos = spec.positions
    sample = None                       # the default budget's stride: (count, size, per segment)
    if pos is not None:
        n = pos.shape[0]
        if spec.part == _wire.PART_POINTS and n > _wire.DEFAULT_MAX_POINTS:
            sample = (n, _wire.DEFAULT_MAX_POINTS, False)
            axes = [{"dim": "point", "start": 0, "step": 1, "count": n, "size": sample[1]}]
        elif spec.part == _wire.PART_LINES and params["mode"] == _wire.LINES_SEGMENTS and \
                n // 2 > _wire.SCENE_DEFAULT_MAX_ITEMS:
            sample = (n // 2, _wire.SCENE_DEFAULT_MAX_ITEMS, True)
            axes = [{"dim": "segment", "start": 0, "step": 1, "count": n // 2,
                     "size": sample[1]}]

    def rows(t):
        """The rows of ``t`` the budget keeps, gathered on t's own device."""
        if sample is None:
            return t
        idx = _stride(sample[0], sample[1], t.device)
        if sample[2]:                   # both vertices of each kept segment
            idx = (idx[:, None] * 2 + torch.arange(2, device=t.device)).reshape(-1)
        return t.index_select(0, idx)

    def flt(t):
        return t if t.dtype == f32 else t.to(f32)

    def intg(t):
        if isinstance(t, (list, tuple)):
            return torch.tensor(t, dtype=i32)
        return t if t.dtype == i32 else t.to(i32)

    bounds_src = []
    if spec.part == _wire.PART_BOXES:
        if spec.corners is not None:                 # 2-D corners to centres and half sizes
            c4 = flt(spec.corners)
            lo = torch.minimum(c4[:, :2], c4[:, 2:])
            hi = torch.maximum(c4[:, :2], c4[:, 2:])
            centers, half = (lo + hi) * 0.5, (hi - lo) * 0.5
        else:
            centers, half = flt(spec.positions), flt(spec.half_sizes)
        arrays["centers"], arrays["half_sizes"] = centers, half
        reach = half.abs()
        if "rotations" in spec.extra:               # a rotated box: its bounding sphere
            reach = torch.linalg.vector_norm(half, dim=1, keepdim=True).expand_as(half)
        bounds_src = [centers - reach, centers + reach]
    elif pos is not None:
        arrays[spec.first] = rows(flt(pos))
        bounds_src = [arrays[spec.first]]
    if spec.colors is not None:
        c = _rgba(spec.colors, torch)
        arrays["colors"] = c if spec.per_item else rows(c)
    vals = None
    if spec.values is not None:
        vals = flt(spec.values)
        vals = vals if spec.per_item else rows(vals)
        arrays["values"] = vals
    if spec.values is not None or spec.depth:
        params["colormap"] = spec.cmap
        params["flags"] = (_wire.SNAP_AUTO_VMIN if spec.vmin is None else 0) | \
                          (_wire.SNAP_AUTO_VMAX if spec.vmax is None else 0)
        params["vmin"] = 0.0 if spec.vmin is None else float(spec.vmin)
        params["vmax"] = 1.0 if spec.vmax is None else float(spec.vmax)
    for name, t in spec.extra.items():
        if name in ("radii", "rotations"):
            arrays[name] = rows(flt(t)) if name == "radii" else flt(t)
        elif name == "vectors":
            arrays[name] = flt(t)
            bounds_src.append(arrays["origins"] + arrays[name])
        elif name in ("normals", "uvs"):
            arrays[name] = flt(t)
        elif name == "class_ids":
            arrays[name] = rows(intg(t))
        elif name in ("strip_lengths", "indices"):
            arrays[name] = intg(t)
        elif name == "texture":
            arrays[name] = _rgba(t, torch, image=True)
        elif name == "pixels":
            if spec.depth:
                arrays[name] = t if t.dtype == torch.uint16 else flt(t)
            elif spec.segmentation:
                if t.dtype == u8:
                    arrays[name] = t
                else:                                # u16 on the wire, cast on the host once copied
                    arrays[name] = t.to(torch.int64).clamp(0, _wire.MAX_CLASS_ID).to(i32)
                    cast[name] = "<u2"
            else:
                arrays[name] = _rgba(t, torch, image=True)
    # The stats: value range and per-axis bounds, on the arrays' devices.
    bounds = vrange = None
    if bounds_src and bounds_src[0].shape[0]:
        allp = torch.cat(bounds_src, 0) if len(bounds_src) > 1 else bounds_src[0]
        bounds = torch.stack([allp.amin(0), allp.amax(0)], 1).reshape(-1)
    if vals is not None and vals.shape[0]:
        vrange = torch.stack(torch.aminmax(vals))
    if spec.depth:                                   # the depths in scene units, zeros and
        v = arrays["pixels"].to(f32) * params["scale"]          # non-finite values left out
        ok = torch.isfinite(v) & (v != 0)
        inf = torch.full((), math.inf, dtype=f32, device=v.device)   # a fill: no copy from the host
        vrange = torch.stack([torch.where(ok, v, inf).amin(), torch.where(ok, v, -inf).amax()])
        stats = [None, None, 0]
    elif spec.part in (_wire.PART_IMAGE,):
        stats = None
    else:
        stats = [None, None, st.dims_at(path)]
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
    if stats is not None:
        stats[0], stats[1] = host.get("_vrange"), host.get("_bounds")
    st.seq += 1
    return Batch(st.name, path, spec.part, st.seq, time.monotonic_ns(), static, params,
                 {k: v for k, v in host.items() if not k.startswith("_")}, stats, axes, event,
                 nbytes, cast)
