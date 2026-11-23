"""Array reconstruction for sender tests from the public library resource layouts."""
import numpy as np
from caliper import _wire as w
from caliper import _scene_pieces


def arrays(update, resources, dims=3):
    def words(name):
        return np.frombuffer(resources[name].data, dtype="<u4")
    piece = update.piece["name"]
    names = update.names
    if piece == "points":
        out = _scene_pieces.unpack_points(resources["points"].data, names, dims)
        if "source_class_ids" in resources:
            out["class_ids"] = words("source_class_ids").view("<i4").copy()
        return out
    if piece == "lines":
        out = {}
        for name in names:
            r = resources["source_" + name]
            dtype = np.uint8 if name == "colors" else "<i4" if name in ("indices", "strip_lengths") else "<f4"
            data = np.frombuffer(r.data, dtype=dtype).copy()
            if name == "positions": data = data.reshape(-1, r.stride // 4)
            if name == "colors": data = data.reshape(-1, 4)
            if name == "indices": data = data.reshape(-1, 2)
            out[name] = data
        return out
    if piece in ("arrows", "boxes", "mesh"):
        arrow, box = piece == "arrows", piece == "boxes"
        u = words("arrows" if arrow else "boxes" if box else "vertices").reshape(-1, 16 if box else 8)
        f = u.view("<f4")
        out = {"origins" if arrow else "centers" if box else "positions": f[:, :dims].copy()}
        if arrow: out["vectors"] = f[:, 4:4+dims].copy()
        if box: out["half_sizes"] = f[:, 4:4+dims].copy()
        for name, off, count in (("rotations", 8, 4), ("normals", 4, 3), ("unused_uvs", 10, 2)):
            if name in names: out[name] = f[:, off:off+count].copy()
        if "colors" in names: out["colors"] = u[:, 3 if arrow else 12 if box else 7].copy().view(np.uint8).reshape(-1, 4)
        if "values" in names: out["values"] = f[:, 3].copy()
        if "class_ids" in names: out["class_ids"] = words("source_class_ids").view("<i4").copy()
        if "indices" in names: out["indices"] = words("faces").view("<i4").reshape(-1, 3).copy()
        if "uvs" in names:
            out["uvs"] = words("attributes").reshape(-1, 3).view("<f4")[:, 1:3].copy()
        if "texture" in names:
            tex = resources["tex"]
            out["texture"] = np.frombuffer(tex.data, np.uint8).reshape(tex.height, tex.width, 4).copy()
        return out
    width, height, channels, size = words("info")[:4]
    if piece == "image":
        pixels = words("pixels").view(np.uint8).reshape(height, width, 4)
        return {"pixels": pixels[..., 0].copy() if channels == 1 else pixels.copy()}
    if piece == "depth":
        pixels = words("source_pixels").view("<f4")
        return {"pixels": pixels.astype("<u2" if size == 2 else "<f4").reshape(height, width)}
    return {"pixels": words("pixels").astype(np.uint8 if size == 1 else "<u2").reshape(height, width)}


class Inbox:
    def __init__(self):
        self.parts = w.SplitResources()
        self.held = {}

    def take(self, message, dims=3):
        if isinstance(message, w.Resource):
            whole = self.parts.add(message)
            if whole is not None:
                self.held[(whole.program, whole.resource)] = whole
        if isinstance(message, w.SceneUpdate) and message.program:
            resources = {name: self.held.pop((message.program, name)) for name in message.resources}
            got = arrays(message, resources, dims)
            message.arrays = {name: w._section(name, value) for name, value in got.items()}
        return message
