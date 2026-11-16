"""A stand-in for torch in the Python loader tests, so that they need no torch.

It has only what loader_host.py uses: load(), Tensor and the bfloat16 dtype.
load() refuses any call but load(file, map_location="cpu", weights_only=True),
and it never unpickles anything: a test checkpoint is the pickle protocol
marker b"\\x80\\x04" (so that Caliper's native readers refuse the file as a
pickle) followed by JSON of the form

    {"__record__": "<path where load() writes the arguments it got>",
     "layers.0.weight": {"dtype": "float32", "shape": [2, 3], "data": [...]},
     "nested": {"bias": {...}}, "epoch": 3}

where a dict with "dtype" is a tensor, another dict nests, and anything else is
a plain value. The dtype "bfloat16" gives a tensor whose numpy() fails, as
torch's does, until float() converts it.
"""
import json

import numpy


class dtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "torch." + self.name


bfloat16 = dtype("bfloat16")
float32 = dtype("float32")


class Tensor:
    def __init__(self, array, kind=None):
        self._array = array
        self.dtype = kind or dtype(str(array.dtype))

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self._array.astype(numpy.float32), float32)

    def numpy(self):
        if self.dtype is bfloat16:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self._array


def _build(value):
    if isinstance(value, dict) and "dtype" in value:
        if value["dtype"] == "bfloat16":
            return Tensor(numpy.array(value["data"], dtype=numpy.float32).reshape(value["shape"]), bfloat16)
        return Tensor(numpy.array(value["data"], dtype=value["dtype"]).reshape(value["shape"]))
    if isinstance(value, dict):
        return {k: _build(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_build(v) for v in value]
    return value


def load(f, map_location=None, weights_only=None, **kwargs):
    if weights_only is not True:
        raise RuntimeError("the test torch loads only with weights_only=True")
    if map_location != "cpu":
        raise RuntimeError("the test torch loads only with map_location='cpu'")
    with open(f, "rb") as handle:
        data = handle.read()
    if data[:2] != b"\x80\x04":
        raise RuntimeError("not a test checkpoint")
    spec = json.loads(data[2:].decode("utf-8"))
    record = spec.pop("__record__", None)
    if record:
        with open(record, "w") as out:
            json.dump({"map_location": map_location, "weights_only": weights_only, "other": sorted(kwargs)}, out)
    return _build(spec)
