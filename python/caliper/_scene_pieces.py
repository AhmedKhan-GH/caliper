"""The piece form of a scene's batches: the arrays of an entity part as the resources of its
library piece (wire protocol 3.2, capability "scene_pieces").

A scene call keeps its checks, its rate limit and its device copies (caliper._scene). On the
network thread, a batch that goes to a link in the piece form becomes one version of the
resources of the part's library piece, written here from the host copies exactly as the piece's
derive half writes them, and the SCENE_UPDATE that follows names them: the scene program
``scene.<n>`` of its scene, path and part (:func:`program_id`), the library pin, each resource's
count and the arrays the batch held. Nothing here touches a device or a model tensor, and a
batch is packed once, for every link that takes the form, and only when it is sent.

``CALIPER_SCENE_SHAPES=builtin`` in the environment keeps the per-shape parts for every link.
A part without a piece route here keeps them too.
"""
from __future__ import annotations

import itertools
import os
import threading

from . import _wire

__all__ = ["enabled", "program_id", "route", "pack", "unpack_points", "PINS"]

# The library pieces of the parts that have a piece route, by name: the version whose resource
# layout the packers write (src/builder/programs/library/pieces/<name>/<version>/piece.toml).
PINS = {"points": {"name": "points", "version": 1}, "image": {"name": "image", "version": 1}}
POINT_STRIDE = 32           # points@1 resource "points": Point {position: vec4<f32>, bits: u32, index: u32}
POINTS_MAX = 4194304        # its count
PIXELS_MAX = 33554432       # image@1 resource "pixels": an rgba8unorm texture, its most texels
INFO_STRIDE = 80            # image@1 resource "info": ImageInfo {size: vec4<u32>, c0..c3: vec4<f32>}
NAN_BITS = 0x7FC00000       # the bits of a point without a color source, as the scene view writes them

_ids: dict = {}
_ids_lock = threading.Lock()
_next = itertools.count(1)


def enabled() -> bool:
    """Whether the piece form is on: not switched off by CALIPER_SCENE_SHAPES=builtin."""
    return os.environ.get("CALIPER_SCENE_SHAPES", "").strip().lower() != "builtin"


def program_id(scene: str, path: str, part: str) -> str:
    """The scene program of an entity part, ``scene.<n>``, kept for the life of the process."""
    key = (scene, path, part)
    with _ids_lock:
        pid = _ids.get(key)
        if pid is None:
            pid = _ids[key] = f"{_wire.SCENE_PROGRAM_PREFIX}{next(_next)}"
        return pid


def route(b) -> str | None:
    """The library piece that draws batch ``b`` in the piece form, or None when the part keeps
    the per-shape part (no route yet, or above the piece's capacity)."""
    if b.part == _wire.PART_POINTS:
        pos = (b.arrays or {}).get("positions")
        if pos is not None and pos.ndim == 2 and pos.shape[0] <= POINTS_MAX:
            return "points"
    elif b.part == _wire.PART_IMAGE and b.params.get("form") == _wire.FORM_COLOR:
        px = (b.arrays or {}).get("pixels")
        if px is not None and px.dtype.itemsize == 1 and (px.ndim == 2 or (px.ndim == 3 and px.shape[2] == 4)) \
                and 0 < px.shape[0] * px.shape[1] <= PIXELS_MAX \
                and max(px.shape[0], px.shape[1]) <= _wire.RESOURCE_SIDE_MAX:
            return "image"
    return None


def pack(b, piece: str) -> tuple:
    """(names, [(resource id, count, stride, bytes, width, height)]) of batch ``b`` for ``piece``,
    width and height of a texture and None for a buffer, cached on the batch, thus packed once for
    every link."""
    got = getattr(b, "packed", None)
    if got is not None:
        return got
    if piece == "points":
        got = (_names(b), [("points", int(b.arrays["positions"].shape[0]), POINT_STRIDE, pack_points(b), None, None)])
    elif piece == "image":
        pixels, info = pack_image(b)
        h, w = int(b.arrays["pixels"].shape[0]), int(b.arrays["pixels"].shape[1])
        got = (_names(b), [("pixels", w * h, 4, pixels, w, h), ("info", 1, INFO_STRIDE, info, None, None)])
    else:
        raise ValueError(f"no piece route for {piece!r}")
    b.packed = got
    return got


def _names(b) -> list:
    return [n for n in _wire.PART_SECTIONS[b.part] if (b.arrays or {}).get(n) is not None]


def pack_points(b):
    """points@1's resource ``points``: each row as Point {position: vec4<f32>, bits: u32, index: u32}
    with 8 bytes of padding, as the derive half and the scene view write it. ``position`` holds
    x, y and z (0 in 2-D) and in w the radius: the row's radius, else the batch's ``radius``, else 1.
    ``bits`` holds the packed RGBA of ``colors`` (r | g << 8 | b << 16 | a << 24), else the bits of
    the f32 of ``values``, else the class id of ``class_ids`` (whose color the viewer writes from its
    class table), else NaN bits."""
    import numpy as np
    a = b.arrays
    pos = a["positions"]
    n, d = int(pos.shape[0]), int(pos.shape[1])
    out = np.zeros((n, POINT_STRIDE // 4), dtype="<u4")
    f = out.view("<f4")
    f[:, :d] = pos
    radii = a.get("radii")
    if radii is not None:
        f[:, 3] = radii
    else:
        f[:, 3] = float(b.params.get("radius") or 1.0)
    colors, values, ids = a.get("colors"), a.get("values"), a.get("class_ids")
    if colors is not None:
        c = np.ascontiguousarray(colors, dtype=np.uint8).reshape(-1, 4).view("<u4").reshape(-1)
        out[:, 4] = c if c.shape[0] == n else c[0]
    elif values is not None:
        out[:, 4] = np.ascontiguousarray(values, dtype="<f4").view("<u4")
    elif ids is not None:
        out[:, 4] = np.ascontiguousarray(ids, dtype="<i4").view("<u4")
    else:
        out[:, 4] = NAN_BITS
    out[:, 5] = np.arange(n, dtype="<u4")
    return memoryview(out).cast("B")


def pack_image(b):
    """image@1's resources of a color image, as its derive half and the scene view write them:
    ``pixels`` the texels of an rgba8unorm texture of the image's width and height, one word
    (r | g << 8 | b << 16 | a << 24) a pixel, row by row, a grey image (H, W) repeating its value in
    red, green and blue with alpha 255; ``info`` the width, the height,
    the channels (1 or 4) and 0 as u32, then the identity pose as four f32 columns (the entity's
    transform places the image)."""
    import numpy as np
    px = np.ascontiguousarray(b.arrays["pixels"], dtype=np.uint8)
    h, w = int(px.shape[0]), int(px.shape[1])
    if px.ndim == 2:
        g = px.reshape(-1).astype("<u4")
        pixels = memoryview(g | (g << 8) | (g << 16) | np.uint32(0xFF000000)).cast("B")
        channels = 1
    else:
        pixels = memoryview(px.reshape(-1)).cast("B")
        channels = 4
    info = np.zeros(INFO_STRIDE // 4, dtype="<u4")
    info[:4] = (w, h, channels, 0)
    f = info.view("<f4")
    for k in range(4):
        f[4 + 4 * k + k] = 1.0
    return pixels, memoryview(info).cast("B")


def unpack_points(data, names: list, dims: int) -> dict:
    """The arrays of a points batch from points@1's resource bytes and the names the update lists:
    what a viewer rebuilds (used by the tests)."""
    import numpy as np
    u = np.frombuffer(data, dtype="<u4").reshape(-1, POINT_STRIDE // 4)
    f = u.view("<f4")
    out = {"positions": f[:, :dims].copy()}
    bits = u[:, 4]
    if "colors" in names:
        out["colors"] = bits.copy().view(np.uint8).reshape(-1, 4)
    if "values" in names:
        out["values"] = bits.view("<f4").copy()
    if "class_ids" in names:
        out["class_ids"] = bits.view("<i4").copy()
    if "radii" in names:
        out["radii"] = f[:, 3].copy()
    return out


def bundle(sub: int, view: str, b, state: bool) -> list:
    """The frames of batch ``b`` in the piece form for one subscription: the RESOURCE versions of
    its scene program (in parts of 8 MiB when larger), then the SCENE_UPDATE that names them.
    None when the batch has no piece route."""
    piece = route(b)
    if piece is None:
        return None
    from . import _programs
    names, resources = pack(b, piece)
    pid = program_id(b.scene, b.path, b.part)
    step = int(getattr(b, "step", None) or 0)
    frames, counts = [], {}
    for rid, count, stride, data, width, height in resources:
        r = _wire.Resource(pid, rid, step, count, stride, data, width, height)
        frames.extend(_wire.resource_parts(r, next(_programs._VERSIONS), _programs.PART_BYTES))
        counts[rid] = count
    frames.append(_wire.SceneUpdate(sub, view, b.seq, b.stamp_ns, b.path, b.part, {}, b.params,
                                    static=b.static, state=state, axes=b.axes, program=pid,
                                    piece=dict(PINS[piece]), resources=counts, names=names))
    return frames
