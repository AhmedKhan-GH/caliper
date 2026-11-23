"""The piece form of a scene's batches: the arrays of an entity part as the resources of its
library piece (wire protocol 3.2, capability "scene_pieces").

A scene call keeps its checks, its rate limit and its device copies (caliper._scene). On the
network thread, a batch that goes to a link in the piece form becomes one version of the
resources of the part's library piece, written here from the host copies exactly as the piece's
derive half writes them, and the SCENE_UPDATE that follows names them: the scene program
``scene.<n>`` of its scene, path and part (:func:`program_id`), the library pin, each resource's
count and the arrays the batch held. Nothing here touches a device or a model tensor, and a
batch is packed once, for every link that takes the form, and only when it is sent.

Every drawable part uses this form. Links without the capability receive scene structure only.
"""
from __future__ import annotations

import itertools
import threading

from . import _wire

__all__ = [ "program_id", "route", "pack", "unpack_points", "PINS"]

# The library pieces of the parts that have a piece route, by name: the version whose resource
# layout the packers write (src/builder/programs/library/pieces/<name>/<version>/piece.toml).
PINS = {name: {"name": name, "version": 1} for name in
        ("points", "lines", "arrows", "boxes", "mesh", "image", "depth", "segmentation")}
POINT_STRIDE = 32           # points@1 resource "points": Point {position: vec4<f32>, bits: u32, index: u32}
POINTS_MAX = 4194304        # its count
PIXELS_MAX = 33554432       # image@1 resource "pixels": an rgba8unorm texture, its most texels
INFO_STRIDE = 80            # image@1 resource "info": ImageInfo {size: vec4<u32>, c0..c3: vec4<f32>}
NAN_BITS = 0x7FC00000       # the bits of a point without a color source, as the scene view writes them

_ids: dict = {}
_ids_lock = threading.Lock()
_next = itertools.count(1)


def program_id(scene: str, path: str, part: str) -> str:
    """The scene program of an entity part, ``scene.<n>``, kept for the life of the process."""
    key = (scene, path, part)
    with _ids_lock:
        pid = _ids.get(key)
        if pid is None:
            pid = _ids[key] = f"{_wire.SCENE_PROGRAM_PREFIX}{next(_next)}"
        return pid


def route(b) -> str | None:
    """The library piece for a drawable batch. Cameras remain scene structure."""
    if b.part == _wire.PART_CAMERA:
        return None
    if b.part == _wire.PART_IMAGE:
        if b.params.get("form") == _wire.FORM_ENCODED:
            decode_image(b)
        return _wire.FORM_PIECE[b.params["form"]]
    if b.part not in PINS:
        raise ValueError(f"no piece for scene part {b.part!r}")
    return b.part


def decode_image(b):
    """Decode a bounded PNG or JPEG on the network thread before texture packing."""
    import io
    import numpy as np
    from PIL import Image
    try:
        with Image.open(io.BytesIO(memoryview(b.arrays["pixels"]).cast("B"))) as image:
            w, h = image.size
            if not (0 < w <= _wire.RESOURCE_SIDE_MAX and 0 < h <= _wire.RESOURCE_SIDE_MAX
                    and w * h <= PIXELS_MAX):
                raise ValueError("encoded image exceeds the image piece capacity")
            pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("encoded image is not a valid bounded PNG or JPEG") from error
    b.arrays = {"pixels": pixels}
    b.params = {k: v for k, v in b.params.items() if k != "format"}
    b.params["form"] = _wire.FORM_COLOR


def pack(b, piece: str) -> tuple:
    """(names, [(resource id, count, stride, bytes, width, height)]) of batch ``b`` for ``piece``,
    width and height of a texture and None for a buffer, cached on the batch, thus packed once for
    every link."""
    got = getattr(b, "packed", None)
    if got is not None:
        return got
    if piece == "points":
        resources = [("points", int(b.arrays["positions"].shape[0]), POINT_STRIDE, pack_points(b), None, None)]
        if "class_ids" in b.arrays and ("colors" in b.arrays or "values" in b.arrays):
            resources.append(_resource("source_class_ids", b.arrays["class_ids"]))
        got = (_names(b), resources)
    elif piece in ("image", "depth", "segmentation"):
        got = (_names(b), pack_pixels(b, piece))
    else:
        got = (_names(b), globals()["pack_" + piece](b))
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
    return memoryview(out).cast("B") if out.size else memoryview(b"")


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


def _resource(name, array, stride=None):
    import numpy as np
    a = np.ascontiguousarray(array)
    stride = stride or (a.dtype.itemsize * (a.shape[1] if a.ndim == 2 else 1))
    data = memoryview(a).cast("B") if a.size else memoryview(b"")
    return (name, data.nbytes // stride, stride, data, None, None)


def _bits(b):
    import numpy as np
    a = b.arrays
    if "colors" in a:
        return np.ascontiguousarray(a["colors"], dtype=np.uint8).reshape(-1, 4).view("<u4").ravel()
    if "values" in a:
        return np.ascontiguousarray(a["values"], dtype="<f4").view("<u4")
    return np.array([NAN_BITS], dtype="<u4")


def _info(bits):
    import numpy as np
    vals = bits.view("<f4")
    valid = vals[~np.isnan(vals)]
    info = np.zeros(3, dtype="<u4")
    if valid.size:
        info.view("<f4")[:2] = (valid.min(), valid.max())
    info[2] = int(np.isnan(vals).any() or not valid.size) | (2 if ((bits >> 24) < 255).any() else 0)
    return _resource("info", info, 12)


def pack_lines(b):
    import numpy as np
    a = b.arrays
    pos = a["positions"]
    n, d = pos.shape
    bits = _bits(b)
    mode = b.params.get("mode", _wire.LINES_SEGMENTS)
    if mode == _wire.LINES_EDGES:
        ids = np.minimum(a["indices"].astype("<u4"), max(0, n - 1)) if n else np.empty((0, 2), "<u4")
        keep = np.ones(len(ids), bool)
        per = b.params.get("color_per") == _wire.COLOR_PER_EDGE
        color_ids = np.repeat(np.arange(len(ids))[:, None], 2, axis=1) if per else ids
    elif mode == _wire.LINES_STRIPS:
        ids = np.column_stack((np.arange(max(0, n - 1)), np.arange(1, n))).astype("<u4")
        ends = np.cumsum(a["strip_lengths"], dtype=np.int64)
        keep = ~np.isin(ids[:, 0] + 1, ends)
        per = b.params.get("color_per") == _wire.COLOR_PER_STRIP
        strip = np.repeat(np.arange(len(ends)), a["strip_lengths"])
        color_ids = np.repeat(strip[:-1, None], 2, axis=1) if per else ids
    else:
        ids = np.arange(n, dtype="<u4").reshape(-1, 2)
        keep = np.ones(len(ids), bool)
        color_ids = ids
    out = np.zeros((len(ids), 10), dtype="<u4")
    f = out.view("<f4")
    if len(ids):
        f[:, :d] = pos[ids[:, 0]]
        f[:, 4:4+d] = pos[ids[:, 1]]
        out[:, 3] = bits[0] if bits.size == 1 else bits[color_ids[:, 0]]
        out[:, 7] = bits[0] if bits.size == 1 else bits[color_ids[:, 1]]
        out[:, 8:10] = ids
        out[~keep, 9] = 0xffffffff
    # Expanded segments omit unused vertices and singleton strips. These host resources preserve
    # the source batch for bounds, selection and labels, and never enter the draw graph.
    sources = [_resource("source_" + name, a[name]) for name in _names(b)]
    return [_resource("segments", out[:, :8], 32), _resource("picks", out[:, 8:10], 8), _info(bits), *sources]


def pack_arrows(b):
    import numpy as np
    a = b.arrays
    n, d = a["origins"].shape
    out = np.zeros((n, 8), dtype="<u4")
    f = out.view("<f4")
    f[:, :d], f[:, 4:4+d] = a["origins"], a["vectors"]
    bits = _bits(b)
    out[:, 3], out[:, 7] = bits, np.arange(n, dtype="<u4")
    return [_resource("arrows", out, 32), _info(bits)]


def pack_boxes(b):
    import numpy as np
    a = b.arrays
    n, d = a["centers"].shape
    out = np.zeros((n, 16), dtype="<u4")
    f = out.view("<f4")
    f[:, :d], f[:, 3], f[:, 4:4+d], f[:, 11] = a["centers"], d, a["half_sizes"], 1
    if "rotations" in a:
        f[:, 8:12] = a["rotations"]
    if "colors" in a:
        out[:, 12] = _bits(b)
        if n and ((out[:, 12] >> 24) < 255).any():
            f[0, 7] = 1
    out[:, 14], out[:, 15] = np.arange(n, dtype="<u4"), 0xffffffff
    resources = [_resource("boxes", out, 64), _resource("names", np.empty(0, "<u4"), 64)]
    if "class_ids" in a:
        resources.append(_resource("source_class_ids", a["class_ids"]))
    return resources


def pack_mesh(b):
    import numpy as np
    a = b.arrays
    n, d = a["positions"].shape
    vertices = np.zeros((n, 8), dtype="<u4")
    f = vertices.view("<f4")
    f[:, :d] = a["positions"]
    vertices[:, 3] = NAN_BITS
    if "values" in a:
        f[:, 3] = a["values"]
    if "normals" in a:
        f[:, 4:7] = a["normals"]
    if "colors" in a:
        vertices[:, 7] = _bits(b)
    attributes = np.zeros((n, 3), dtype="<u4")
    attributes[:, 0] = np.arange(n, dtype="<u4")
    if "uvs" in a:
        attributes.view("<f4")[:, 1:3] = a["uvs"]
    alpha = "colors" in a and ((vertices[:, 7] >> 24) < 255).any()
    if "texture" in a:
        tex = np.ascontiguousarray(a["texture"], dtype=np.uint8)
        h, w = tex.shape[:2]
        texture = ("tex", h*w, 4, memoryview(tex).cast("B"), w, h)
        alpha = alpha or (tex[..., 3] < 255).any()
    else:
        texture = _resource("tex", np.empty(0, "<u4"))
    info = np.array([int(alpha)], dtype="<u4")
    faces = np.minimum(a["indices"].astype("<u4"), max(0, n-1)) if "indices" in a else np.arange(n, dtype="<u4")
    return [texture, _resource("vertices", vertices, 32), _resource("attributes", attributes, 12),
            _resource("info", info), _resource("faces", faces.ravel(), 4)]


def pack_pixels(b, piece):
    import numpy as np
    if piece == "image":
        pixels, info = pack_image(b)
        h, w = b.arrays["pixels"].shape[:2]
        return [("pixels", w*h, 4, pixels, w, h), ("info", 1, 80, info, None, None)]
    px = np.ascontiguousarray(b.arrays["pixels"])
    h, w = px.shape
    info = np.zeros(24 if piece == "depth" else 20, dtype="<u4")
    info[:4] = (w, h, 1, px.dtype.itemsize)
    f = info.view("<f4")
    f[4:20] = np.eye(4, dtype="<f4").ravel()
    if piece == "depth":
        source = px.astype("<f4").ravel()
        pixels = source * np.float32(b.params.get("scale", 0.001 if px.dtype.itemsize == 2 else 1.0))
        valid = pixels[np.isfinite(pixels) & (pixels != 0)]
        if valid.size:
            f[20:23] = (valid.min(), valid.max(), 1)
        return [_resource("pixels", pixels), _resource("info", info, 96), _resource("source_pixels", source)]
    return [_resource("pixels", px.astype("<u4").ravel()),
            _resource("palette", np.zeros(4096, "<u4")), _resource("info", info, 80)]
