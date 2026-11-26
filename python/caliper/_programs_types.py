"""Shared constants and data structures for GPU programs and programmable pieces."""
from __future__ import annotations

from collections import deque

_ARRAYS = ("positions", "colors", "values", "radii")
_STOP_TEXT = {
    "program.over_time": "The program went over limits.time_ms on 3 passes in a row.",
    "program.import_refused": "An input of the program was released.",
    "program.layer_refused": "The graphics layer refused the pass.",
}

RESOURCE_TYPE = 0x0082  # RESOURCE (include/caliper/live_net.h), one published resource
PIECE_PASSES_MAX = 8
PIECE_RESOURCES_MAX = 8
PIECE_FIELDS_MAX = 16
PIECE_COUNT_MAX = 33554432      # spec/piece.h, kCountMax: 128 MiB of 4-byte elements
PIECE_SIDE_MAX = 8192          # spec/piece.h, kTextureSideMax: WebGPU's default largest 2-D texture
TEXTURE_FORMATS = ("rgba8unorm", "r8unorm", "r32uint")
WORKGROUPS_MAX = 65535
_DTYPES = ("f32", "i32", "u32")
_TENSOR_DTYPES = ("f32", "i32", "u32", "u8", "u16")


def _tensor_dtype_ok(dtype) -> bool:
    """A tensor input's dtype: one of f32, i32, u32, u8 and u16, or two to five of them joined by "|",
    each once; the bound tensor holds one of them."""
    if not isinstance(dtype, str) or len(dtype) > 32:
        return False
    types = dtype.split("|")
    return all(t in _TENSOR_DTYPES for t in types) and len(set(types)) == len(types)


class _Import:
    """One tensor of the script imported in place: its handle and the storage it binds."""
    __slots__ = ("name", "handle", "key")

    def __init__(self, name, handle, key):
        self.name, self.handle, self.key = name, handle, key


class _LinkSet:
    """The viewers that got a draw half, by identity, without keeping a closed link alive."""

    def __init__(self):
        import weakref
        self._refs = []
        self._weakref = weakref.ref

    def __contains__(self, link) -> bool:
        return any(r() is link for r in self._refs)

    def __bool__(self) -> bool:
        return any(r() is not None for r in self._refs)

    def add(self, link) -> None:
        if link in self:
            return
        try:
            ref = self._weakref(link)
        except TypeError:                            # a link without weak references
            ref = (lambda link=link: link)
        self._refs = [r for r in self._refs if r() is not None] + [ref]

    def discard(self, link) -> None:
        self._refs = [r for r in self._refs if r() is not None and r() is not link]


class _Program:
    __slots__ = ("id", "decl", "wgsl", "sha256", "scene", "kind", "inputs", "output", "every_min",
                 "time_ms", "memory_mb", "handle", "state", "error", "warning", "runs", "last_step",
                 "last_pass_ms", "memory_bytes", "warm", "every", "t0", "bound", "pass_times",
                 "params", "builtin", "shape", "piece", "manifest", "bind", "panel", "segments",
                 "common", "draw", "types", "resources", "passes", "param_types", "written",
                 "sent_static", "static_frames", "static_relayed", "relayed", "skipped", "behind",
                 "output_retry", "input_version", "params_dirty", "act_dirty")

    def __init__(self, pid, decl, wgsl, scene, sha256):
        from . import _programs as _mod

        self.id = pid
        self.decl = decl
        self.wgsl = wgsl
        self.sha256 = sha256
        self.scene = scene
        out = decl.get("output") or {}
        self.output = out
        self.kind = out.get("kind", "geometry")
        self.shape = out.get("shape")
        self.params = dict(decl.get("params") or {})   # name -> number: the uniform's values
        self.builtin = decl.get("builtin") or None
        self.inputs = [(str(i.get("var", "")), str(i.get("source", ""))) for i in decl.get("inputs") or []]
        lim = decl.get("limits") or {}
        self.every_min = int(lim.get("every") or 1)
        self.time_ms = float(lim.get("time_ms") or _mod.TIME_MS_DEFAULT)
        self.memory_mb = int(lim.get("memory_mb") or _mod.MEMORY_MB_DEFAULT)
        self.handle = None
        self.state = "waiting"
        self.error = None
        self.warning = None
        self.runs = 0
        self.last_step = None
        self.input_version = -1
        self.params_dirty = True
        self.act_dirty = False
        self.last_pass_ms = None
        self.memory_bytes = 0
        self.warm = _mod.WARMUP_STEPS
        self.every = None
        self.t0 = _mod._clock()
        self.bound = {}         # var -> the _Import it was compiled with
        self.pass_times = deque(maxlen=_mod.RATE_PASSES + 1)   # the clock at its last passes
        self.piece = None       # a piece's instance: the pin {name, version, hash, from}
        self.manifest = None
        self.bind, self.panel, self.segments = {}, None, None
        self.common, self.draw = "", ""
        self.types = {}         # input name -> (kind, dtype)
        self.resources = []     # the manifest's resources, checked
        self.passes = []        # (entry, workgroups, writes) of the derive half
        self.param_types = {}   # param name -> f32, i32 or u32
        self.written = set()    # the resources that a pass names in `writes`
        self.sent_static = False
        self.static_frames = {}  # resource id -> the RESOURCE frame of a static resource
        self.static_relayed = _LinkSet()
        self.relayed = _LinkSet()  # the viewers that got the draw half
        self.skipped = 0        # passes whose RESOURCE frames a viewer behind did not get
        self.behind = _LinkSet()  # the viewers whose last pass of it was skipped
        self.output_retry = False  # a discarded pass must resend its static resources

    def passes_per_s(self):
        """The passes a second on the run's wall clock over the last ``RATE_PASSES`` passes, or
        None before the second pass and while the program does not run."""
        t = self.pass_times
        if self.state != "running" or len(t) < 2 or t[-1] <= t[0]:
            return None
        return (len(t) - 1) / (t[-1] - t[0])

    def item(self) -> dict:
        from . import _programs as _mod

        cad = _mod._CAD
        b, d, share = cad.step_s(), cad.cost_s(), cad.share()
        return {
            "id": self.id, "kind": "operator" if self.builtin else self.kind, "state": self.state,
            **({"piece": dict(self.piece)} if self.piece else {}),
            "every": self.every,
            "every_min": self.every_min, "last_pass_ms": self.last_pass_ms,
            "last_cost_ms": None if d is None else d * 1e3,
            "step_ms": None if b is None else b * 1e3,
            "overhead_share": None if share is None else share,
            "passes_per_s": self.passes_per_s(),
            "last_step": self.last_step, "runs": self.runs, "memory_bytes": self.memory_bytes,
            "memory_limit_bytes": self.memory_mb << 20, "error": self.error,
            "warning": self.warning,
            "sha256": self.sha256,
        }


class _Refused(Exception):
    def __init__(self, code, message, line=None, column=None):
        super().__init__(message)
        self.error = {"code": code, "message": message, "line": line, "column": column}


class _Waiting(Exception):
    """An input that exists but has no tensor yet: a gradient before the first backward pass, or
    a module's output before its first forward pass."""


class _OutputRemoval:
    """An output address without tensors, resource frames, or a retired program reference."""
    __slots__ = ("id", "kind", "runs", "scene", "output")

    def __init__(self, p):
        self.id, self.kind, self.runs, self.scene = p.id, p.kind, p.runs, p.scene
        self.output = {"entity": p.output.get("entity")}

    def target(self):
        from . import _programs as _mod

        if self.kind == "piece":
            return "piece", self.id
        if self.kind in ("tensor", "pixels"):
            return "view", self.id
        return "scene", self.scene, _mod._entity(self)


class _Detach:
    """A hook handle whose ``remove()`` detaches the facade from its model."""

    def __init__(self, model):
        self._model = model

    def remove(self) -> None:
        from . import _programs as _mod
        _mod.detach(self._model)
