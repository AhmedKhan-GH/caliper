"""Live intervention points and hooks for watched PyTorch and TransformerLens models.

``TorchAdapter.points`` and ``TorchAdapter.ablate`` in ``caliper._adapter`` call :func:`points`
and :func:`ablate`. ``caliper._controls`` validates and installs structured interventions
(``ablate``, ``add``, ``scale``, ``clamp``, ``patch``) at step boundaries through
:class:`InterventionExecutor`.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import math
import weakref

from . import _intervention_protocol as metadata
from ._intervention_protocol import (
     FIELDS,
    KINDS,
    MAX_POSITIONS,
    MAX_VECTOR,
    InterventionError,
    internal_forward,
)

__all__ = [
    "points",
    "ablate",
    "record_shapes",
    "Ablation",
    "HEAD_SEPARATOR",
    "KINDS",
    "InterventionError",
    "InterventionExecutor",
    "InterventionReplay",
    "suspend",
    "rows",
    "selected_rows",
    "row_mask",
]

HEAD_SEPARATOR = "#"
MAX_PROMPT_BYTES = 4096
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_ENTRIES = 64

_HEAD_COUNTS = ("num_heads", "n_heads", "n_head")
_PROJECTIONS = ("out_proj", "o_proj", "c_proj", "proj")
_MAX_CHANNELS = 256
_TL_PROBES = (
    ("resid_pre", "hook_resid_pre"),
    ("attn_out", "hook_attn_out"),
    ("attn_pattern", "attn.hook_pattern"),
    ("mlp_in", "hook_mlp_in"),
    ("mlp_out", "hook_mlp_out"),
)

_shapes: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_SUSPENDED = ContextVar("caliper_interventions_suspended", default=frozenset())
_ROWS = ContextVar("caliper_intervention_rows", default={})
_MUTED_HOOKS = ContextVar("caliper_muted_intervention_hooks", default=frozenset())
_ACTIVE_REPLAYS = ContextVar("caliper_intervention_replays", default=())


def _indices(value, name: str, *, limit: int | None = None) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple))
            or any(type(i) is not int or i < 0 or (limit is not None and i >= limit) for i in value)
            or len(set(value)) != len(value)
            or (limit is not None and len(value) > limit)):
        raise InterventionError(f"{name} must contain unique nonnegative integer indices")
    return tuple(value)


def _text_checked(value, name: str, maximum: int, *, code: str = "intervene.bad_request") -> str:
    if not metadata.text(value, maximum, False):
        raise InterventionError(f"{name} must contain 1 to {maximum} UTF-8 bytes", code)
    return value


def _finite_checked(value, name: str) -> float:
    if not metadata.finite(value):
        raise InterventionError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InterventionError(f"{name} must be a finite number")
    return result


def selected_rows(model):
    """Return None for all rows, or a tuple for the current model context."""
    if model is not None and id(model) in _SUSPENDED.get():
        return ()
    return _ROWS.get().get(id(model)) if model is not None else None


def _hook_muted(hook) -> bool:
    return hook in _MUTED_HOOKS.get()


@contextmanager
def _mute_hooks(hooks):
    token = _MUTED_HOOKS.set(_MUTED_HOOKS.get().union(hooks))
    try:
        yield
    finally:
        _MUTED_HOOKS.reset(token)


@contextmanager
def suspend(model):
    """Suspend every intervention on ``model`` within the current context."""
    token = _SUSPENDED.set(_SUSPENDED.get() | {id(model)})
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


@contextmanager
def rows(model, indices):
    """Restrict interventions on ``model`` to batch rows, intersected with any outer scope."""
    chosen = _indices(indices, "rows")
    outer = _ROWS.get().get(id(model))
    if outer is not None:
        chosen = tuple(i for i in chosen if i in outer)
    token = _ROWS.set({**_ROWS.get(), id(model): chosen})
    try:
        yield
    finally:
        _ROWS.reset(token)


def row_mask(model, tensor):
    """Build the batch mask for module and head hooks."""
    import torch
    chosen = selected_rows(model)
    if chosen is None:
        return None
    if tensor.ndim == 0:
        raise InterventionError("batch row selection requires a batch dimension")
    if any(i >= tensor.shape[0] for i in chosen):
        raise InterventionError("a selected batch row is outside the tensor")
    mask = torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    if chosen:
        mask[list(chosen)] = True
    return mask.reshape((-1,) + (1,) * (tensor.ndim - 1))


def _head_count(module) -> int:
    for attr in _HEAD_COUNTS:
        v = getattr(module, attr, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    cfg = getattr(module, "config", None) or getattr(module, "cfg", None)
    for attr in ("num_attention_heads", "n_heads"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return 0


def _projection(module, heads: int):
    """The one output projection of ``module`` that splits into ``heads`` slices, or None."""
    import torch
    if heads <= 0 or isinstance(module, torch.nn.MultiheadAttention):
        return None
    children = getattr(module, "_modules", {}) or {}
    found = [children[n] for n in _PROJECTIONS
             if isinstance(children.get(n), torch.nn.Linear)]
    if len(found) != 1 or found[0].in_features % heads:
        return None
    return found[0]


def _tl_hook_z(module, heads: int):
    """Return TransformerLens ``hook_z`` child when ``module`` is an attention block."""
    if heads <= 0:
        return None
    children = getattr(module, "_modules", {}) or {}
    hook_z = children.get("hook_z")
    if hook_z is not None and hasattr(module, "W_O"):
        return hook_z
    return None


def _conv_channels(module) -> int:
    """Return output channel count of a bounded ConvNd module, else 0."""
    import torch
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        channels = getattr(module, "out_channels", 0)
        if isinstance(channels, int) and not isinstance(channels, bool) and 1 < channels <= _MAX_CHANNELS:
            return channels
    return 0


def _type_name(module) -> str:
    return type(module).__name__[:63] or "Module"


def _tl_semantic_map(model) -> dict[str, tuple[str, int, str]]:
    """Map ``layers.<l>.<probe>`` names onto ``HookedTransformer`` hook modules."""
    cfg = getattr(model, "cfg", None)
    hook_dict = getattr(model, "hook_dict", None)
    n_layers = getattr(cfg, "n_layers", None)
    if not isinstance(hook_dict, dict) or not isinstance(n_layers, int) or n_layers <= 0:
        return {}
    use_mlp_in = bool(getattr(cfg, "use_hook_mlp_in", False))
    out: dict[str, tuple[str, int, str]] = {}
    for layer in range(n_layers):
        for short, suffix in _TL_PROBES:
            if short == "mlp_in" and not use_mlp_in:
                continue
            target = f"blocks.{layer}.{suffix}"
            if target in hook_dict:
                out[f"layers.{layer}.{short}"] = (target, layer, short)
    return out


def points(model) -> list:
    """List semantic, module, and attention-head intervention points."""
    shapes = _shapes.get(model, {})
    semantic = []
    tl_map = _tl_semantic_map(model)
    if tl_map:
        mods = dict(model.named_modules())
        for point, (target, _layer, _short) in tl_map.items():
            mod = mods.get(target)
            if mod is None:
                continue
            shape = shapes.get(target) or shapes.get(point)
            semantic.append({
                "point": point,
                "kind": "module",
                "module": target,
                "head": None,
                "type": _type_name(mod),
                "shape": list(shape) if shape is not None else None,
            })
    out = []
    for name, mod in model.named_modules():
        if not name:
            continue
        shape = shapes.get(name)
        out.append({"point": name, "kind": "module", "module": name, "head": None,
                    "type": _type_name(mod), "shape": list(shape) if shape is not None else None})
        heads = _head_count(mod)
        if _projection(mod, heads) is not None or _tl_hook_z(mod, heads) is not None:
            out += [{"point": f"{name}{HEAD_SEPARATOR}{h}", "kind": "head", "module": name, "head": h,
                     "type": _type_name(mod), "shape": None} for h in range(heads)]
        else:
            channels = _conv_channels(mod)
            if channels > 0:
                out += [{"point": f"{name}{HEAD_SEPARATOR}{c}", "kind": "module", "module": name, "head": None,
                         "type": _type_name(mod), "shape": None} for c in range(channels)]
    return semantic + out


def _find(model, point: str):
    """``(module, head_spec)`` of ``point``, with ``head_spec`` ``None`` for a module, or KeyError."""
    if not isinstance(point, str) or not point:
        raise KeyError(point)
    tl_map = _tl_semantic_map(model)
    if point in tl_map:
        target, _layer, short = tl_map[point]
        if short == "attn_pattern":
            raise ValueError("attention patterns are read-only points")
        return dict(model.named_modules())[target], None
    name, sep, head = point.rpartition(HEAD_SEPARATOR)
    if not sep:
        name, head = point, None
    mod = dict(model.named_modules()).get(name) if name else None
    if mod is None:
        raise KeyError(point)
    if head is None:
        if name.endswith(".hook_pattern") or name.endswith(".attn_pattern"):
            raise ValueError("attention patterns are read-only points")
        return mod, None
    if not head.isdigit() or (len(head) > 1 and head[0] == "0"):
        raise KeyError(point)
    h, heads = int(head), _head_count(mod)
    if h < heads:
        proj = _projection(mod, heads)
        if proj is not None:
            return proj, ("linear", h, proj.in_features // heads, proj.in_features)
        hook_z = _tl_hook_z(mod, heads)
        if hook_z is not None:
            return hook_z, ("hook_z", h, heads)
    channels = _conv_channels(mod)
    if h < channels:
        return mod, ("channel", h, channels)
    raise KeyError(point)


class Ablation:
    """One ablation's hook, as :func:`ablate` returns it."""

    def __init__(self, point: str, model=None):
        self.point = point
        self._model = weakref.ref(model) if model is not None else (lambda: None)
        self.on_error = None
        self.calls = 0
        self.errors = 0
        self.error = None
        self._handle = None
        self._masks: dict = {}

    def __call__(self) -> None:
        h, self._handle = self._handle, None
        if h is not None:
            h.remove()
        self._masks.clear()
        self.on_error = None

    @property
    def removed(self) -> bool:
        return self._handle is None

    def _suspended(self) -> bool:
        return self.removed or _hook_muted(self) or selected_rows(self._model()) == ()

    def _failed(self, e: BaseException) -> None:
        self.errors += 1
        self.error = f"{type(e).__name__}: {e}"[:300]
        callback = self.on_error
        self()
        if callback is not None:
            try:
                callback(self.error)
            except Exception:
                pass

    def _zero(self, t, mask_key):
        import torch
        if isinstance(t, torch.Tensor):
            if not (t.is_floating_point() or t.is_complex()):
                return t
            mask = self._masks.get((t.device, mask_key))
            if mask is None:
                mask = self._mask(t.device, mask_key)
            selected = row_mask(self._model(), t)
            if selected is not None:
                mask = mask & selected
            return t.masked_fill(mask, 0)
        if isinstance(t, tuple) and hasattr(t, "_fields"):
            return type(t)(*(self._zero(x, mask_key) for x in t))
        if isinstance(t, (tuple, list)):
            return type(t)(self._zero(x, mask_key) for x in t)
        if isinstance(t, dict):
            return {k: self._zero(v, mask_key) for k, v in t.items()}
        return t

    def _mask(self, device, mask_key):
        import torch
        if mask_key is None:
            mask = torch.ones((), dtype=torch.bool, device=device)
        elif mask_key[0] == "hook_z":
            _, h, heads = mask_key
            mask = torch.zeros((heads, 1), dtype=torch.bool, device=device)
            mask[h, 0] = True
        elif mask_key[0] == "channel":
            _, c, channels, ndim, axis = mask_key
            shape = [1] * ndim
            shape[axis] = channels
            mask = torch.zeros(shape, dtype=torch.bool, device=device)
            idx = [0] * ndim
            idx[axis] = c
            mask[tuple(idx)] = True
        else:
            _, h, width, features = mask_key
            mask = torch.zeros(features, dtype=torch.bool, device=device)
            mask[h * width:(h + 1) * width] = True
        self._masks[(device, mask_key)] = mask
        return mask

    def _forward_hook(self, module, args, output):
        if self._suspended():
            return None
        self.calls += 1
        try:
            return self._zero(output, None)
        except Exception as e:
            self._failed(e)
            return None

    def _hook_z_forward(self, h: int, heads: int):
        key = ("hook_z", h, heads)

        def hook(module, args, output):
            if self._suspended():
                return None
            self.calls += 1
            try:
                if output.ndim < 3 or output.shape[-2] != heads:
                    raise ValueError(f"hook_z output has shape {tuple(output.shape)}, expected {heads} heads")
                return self._zero(output, key)
            except Exception as e:
                self._failed(e)
                return None

        return hook

    def _pre_hook(self, h: int, width: int, features: int):
        key = ("linear", h, width, features)

        def hook(module, args):
            if self._suspended():
                return None
            self.calls += 1
            try:
                if not args:
                    return None
                x = args[0]
                if x.shape[-1] != features:
                    raise ValueError(f"the projection's input has {x.shape[-1]} features, not {features}")
                return (self._zero(x, key),) + tuple(args[1:])
            except Exception as e:
                self._failed(e)
                return None

        return hook


    def _channel_forward(self, c: int, channels: int):
        def hook(module, args, output):
            if self._suspended():
                return None
            self.calls += 1
            try:
                expected_batched = len(getattr(module, "kernel_size", (1, 1))) + 2
                if output.ndim == expected_batched and output.shape[1] == channels:
                    axis = 1
                elif output.ndim == expected_batched - 1 and output.shape[0] == channels:
                    axis = 0
                else:
                    raise ValueError(
                        f"conv output has shape {tuple(output.shape)}, expected {channels} channels"
                    )
                return self._zero(output, ("channel", c, channels, output.ndim, axis))
            except Exception as e:
                self._failed(e)
                return None

        return hook


def ablate(model, point: str):
    """Ablate one point of ``model`` and return the :class:`Ablation` remover."""
    target, head_spec = _find(model, point)
    a = Ablation(point, model)
    if head_spec is None:
        a._handle = target.register_forward_hook(a._forward_hook)
    elif head_spec[0] == "hook_z":
        _, h, heads = head_spec
        a._handle = target.register_forward_hook(a._hook_z_forward(h, heads))
    elif head_spec[0] == "channel":
        _, c, channels = head_spec
        a._handle = target.register_forward_hook(a._channel_forward(c, channels))
    else:
        _, h, width, features = head_spec
        a._handle = target.register_forward_pre_hook(a._pre_hook(h, width, features))
    return a


def _shape_of(output):
    import torch
    t = output
    if isinstance(t, (tuple, list)):
        t = next((x for x in t if isinstance(x, torch.Tensor)), None)
    if isinstance(t, torch.Tensor):
        return tuple(int(d) for d in t.shape)
    return None


def record_shapes(model):
    """Record the output shape of each module of ``model`` at its next forward pass."""
    store = _shapes.setdefault(model, {})
    tl_map = _tl_semantic_map(model)
    reverse_tl = {target: point for point, (target, _, _) in tl_map.items()}
    handles = []

    def make(name):
        def hook(module, args, output):
            try:
                s = _shape_of(output)
                if s is not None:
                    store[name] = s
                    alias = reverse_tl.get(name)
                    if alias is not None:
                        store[alias] = s
            except Exception:
                pass
        return hook

    try:
        for name, mod in model.named_modules():
            if name:
                handles.append(mod.register_forward_hook(make(name)))
    except BaseException:
        for h in handles:
            h.remove()
        raise

    def remove() -> None:
        while handles:
            handles.pop().remove()
    return remove


def _tokenize_prompt(model, tokenizer, prompt: str):
    from collections.abc import Mapping
    import torch
    if hasattr(model, "to_tokens") and tokenizer is getattr(model, "tokenizer", None):
        prepend = getattr(getattr(model, "cfg", None), "default_prepend_bos", True)
        ids = model.to_tokens(prompt, prepend_bos=prepend)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if not 1 <= int(ids.shape[1]) <= MAX_POSITIONS:
            raise ValueError(f"prompt token length {int(ids.shape[1])} must be in 1..{MAX_POSITIONS}")
        return {"input_ids": ids}
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1 or not 1 <= int(ids.shape[1]) <= MAX_POSITIONS:
        raise ValueError(f"prompt token length {int(ids.shape[1])} must be in 1..{MAX_POSITIONS}")
    if isinstance(encoded, Mapping) and "attention_mask" in encoded:
        return {"input_ids": ids, "attention_mask": encoded["attention_mask"]}
    return {"input_ids": ids}


def _run_clean_forward(model, inputs: dict):
    device = next(model.parameters(), None)
    dev = device.device if device is not None else "cpu"
    moved = {k: v.to(dev) for k, v in inputs.items()}
    if hasattr(model, "hook_dict") and hasattr(model, "cfg"):
        return model(moved["input_ids"])
    return model(**moved)


class _TensorHook:
    """Removable forward hook for ``add``, ``patch``, and feature interventions."""

    def __init__(self, executor: "InterventionExecutor", spec: dict, width: int, on_error):
        self.executor = executor
        self.spec = spec
        self.width = width
        self.on_error = on_error
        self.handles: list = []
        self.calls = 0
        self.errors = 0
        self.error = None

    @property
    def removed(self) -> bool:
        return not self.handles

    def __call__(self) -> None:
        self._remove()

    def _remove(self, *, retain_source: bool = False) -> None:
        while self.handles:
            self.handles.pop().remove()
        if self.spec["op"] == "patch" and not retain_source:
            key = (self.spec["source_prompt"], self.spec["point"])
            if key not in self.executor._source_keys():
                self.executor._cache.pop(key, None)
        self.on_error = None

    def _failed(self, error: BaseException) -> None:
        from .interp._memory import CleanStop
        self.errors += 1
        self.error = f"{type(error).__name__}: {error}"[:300]
        callback = self.on_error
        self()
        if callback is not None:
            try:
                callback(self.error)
            except CleanStop:
                raise
            except Exception:
                pass

    def post(self, module, args, output):
        from .interp._memory import CleanStop
        if self.removed or _hook_muted(self) or selected_rows(self.executor.model) == ():
            return None
        self.calls += 1
        try:
            return self._apply(output)
        except Exception as error:
            self._failed(error)
            if isinstance(error, CleanStop):
                raise
            return None

    def _apply(self, value):
        import torch
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != self.width:
            raise InterventionError("point must produce a batch-token-width tensor of the expected width")
        if not value.is_floating_point():
            raise InterventionError("point tensor must be floating point")
        chosen = selected_rows(self.executor.model)
        batch = list(range(value.shape[0])) if chosen is None else list(chosen)
        positions = self.spec.get("positions", list(range(value.shape[1])))
        if any(i >= value.shape[0] for i in batch) or any(i >= value.shape[1] for i in positions):
            raise InterventionError("selected row or position is outside the live tensor")
        if not batch or not positions:
            return value
        b = torch.tensor(batch, device=value.device, dtype=torch.long)[:, None]
        p = torch.tensor(positions, device=value.device, dtype=torch.long)[None, :]
        selected = value[b, p]
        kind = self.spec["op"]
        if kind == "add":
            vector = value.new_tensor(self.spec["vector"])
            changed = selected + self.spec["coefficient"] * vector
        elif kind == "patch":
            source = self.executor._source(self.spec)
            src_pos = ([self.spec["source_position"]] * len(positions)
                       if "source_position" in self.spec else positions)
            if max(src_pos) >= source.shape[1] or source.shape[-1] != self.width:
                raise InterventionError("source positions or width do not match the patch")
            changed = source[:, src_pos].to(device=value.device, dtype=value.dtype).expand(len(batch), -1, -1)
        else:
            raise InterventionError(f"unsupported tensor hook op {kind!r}")
        if not torch.isfinite(changed).all().item():
            raise InterventionError("intervention produced a nonfinite tensor")
        result = value.clone()
        result[b, p] = changed
        return result


class InterventionExecutor:
    """Validate specifications and install live hooks on a watched model."""

    def __init__(self, model, *, adapter=None):
        self.model = model
        self.adapter = adapter
        self._ablation_points: frozenset[str] | None = None
        self.dictionary = None
        self.dictionary_id: str | None = None
        self.tokenizer = None
        self.resource_guard = None
        self.backend = None
        self._cache: OrderedDict = OrderedDict()
        self.preparing_source = False
        self._hooks: weakref.WeakSet = weakref.WeakSet()
        self._hook_order: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._next_hook_order = 0
        self._feature_groups: dict = {}

    def suspend(self):
        return suspend(self.model)

    def rows(self, indices):
        return rows(self.model, indices)

    @property
    def suspended(self) -> bool:
        return id(self.model) in _SUSPENDED.get()

    def configure(self, *, dictionary=None, tokenizer=None, dictionary_id=None,
                  resource_guard=None, backend=None) -> None:
        if resource_guard is not None and not callable(getattr(resource_guard, "check", None)):
            raise InterventionError("resource_guard must provide check(stage, **kwargs)")
        if dictionary is None and dictionary_id is not None:
            raise InterventionError("dictionary identifier requires a dictionary", "intervene.unknown_dictionary")
        if tokenizer is not None and not callable(tokenizer) and not hasattr(tokenizer, "encode"):
            raise InterventionError("tokenizer must be callable")
        if dictionary is not None and dictionary_id is None:
            meta = getattr(dictionary, "metadata", None)
            dictionary_id = (meta.get("dictionary_id") if isinstance(meta, dict) else None) or "default"
        if dictionary_id is not None:
            _text_checked(dictionary_id, "dictionary identifier", 255)
        self.close()
        self.dictionary = dictionary
        self.dictionary_id = dictionary_id
        self.tokenizer = tokenizer
        self.resource_guard = resource_guard
        self.backend = backend

    def invalidate_cache(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        for hook in list(self._hooks):
            hook()
        self._feature_groups.clear()
        self.invalidate_cache()

    def _source_keys(self) -> set[tuple[str, str]]:
        return {(h.spec["source_prompt"], h.spec["point"]) for h in self._hooks
                if isinstance(h, _TensorHook) and not h.removed and h.spec["op"] == "patch"}

    def _discard_unused_sources(self) -> None:
        required = self._source_keys()
        for key in list(self._cache):
            if key not in required:
                del self._cache[key]

    def _target(self, point: str):
        tl_map = _tl_semantic_map(self.model)
        if point in tl_map:
            target_name, layer, short = tl_map[point]
            if short == "attn_pattern":
                raise InterventionError("attention patterns are read-only points")
            module = dict(self.model.named_modules())[target_name]
            width = int(self.model.cfg.d_model)
            return module, width, (layer, short)
        try:
            module, head_spec = _find(self.model, point)
        except ValueError as error:
            raise InterventionError(str(error)) from None
        except KeyError:
            raise InterventionError("unknown intervention point", "intervene.unknown_point") from None
        if head_spec is not None:
            if head_spec[0] == "channel":
                raise InterventionError("channel points support ablate only")
            raise InterventionError("head points support ablate only")
        shape = _shapes.get(self.model, {}).get(point)
        cfg = getattr(self.model, "cfg", None)
        d_model = getattr(cfg, "d_model", None) if point.startswith("blocks.") else None
        width = getattr(module, "out_features", shape[-1] if shape else d_model)
        if width is None:
            raise InterventionError("point width is unknown; capture its output first")
        return module, int(width), None

    def _feature_count(self, layer: int, point: str) -> int:
        if self.backend is not None and hasattr(self.backend, "feature_count"):
            return int(self.backend.feature_count(layer, point=point))
        layers = getattr(self.dictionary, "layers", None)
        if layers is not None and 0 <= layer < len(layers):
            w_dec = getattr(layers[layer], "W_dec", None)
            if w_dec is not None:
                return int(w_dec.shape[0])
        d_sae = getattr(getattr(self.dictionary, "cfg", None), "d_sae", None)
        if isinstance(d_sae, int) and d_sae > 0:
            return d_sae
        d_transcoder = getattr(self.dictionary, "d_transcoder", None)
        if isinstance(d_transcoder, int) and d_transcoder > 0:
            return d_transcoder
        raise InterventionError("dictionary does not expose feature count for point", "intervene.unknown_dictionary")

    def _resolve_feature_layer(self, point: str, semantic) -> int:
        if self.backend is not None and hasattr(self.backend, "resolve_feature_layer"):
            return int(self.backend.resolve_feature_layer(point))
        if semantic is not None and semantic[1] == "mlp_out":
            return semantic[0]
        meta = getattr(self.dictionary, "metadata", None)
        if isinstance(meta, dict):
            writes = meta.get("write_points") or ()
            if point in writes:
                return list(writes).index(point)
        hook_name = getattr(getattr(getattr(self.dictionary, "cfg", None), "metadata", None), "hook_name", None)
        if hook_name and point == hook_name:
            return 0
        raise InterventionError("features require a configured dictionary write point")

    def validate(self, request, *, resources: bool = True) -> dict:
        import torch
        if not isinstance(request, dict):
            raise InterventionError("intervention set must be an object")
        op, kind = request.get("op"), request.get("kind")
        if op is not None and kind is not None and op != kind:
            raise InterventionError("op and kind conflict")
        kind = kind if kind is not None else op
        if not isinstance(kind, str) or kind not in KINDS:
            raise InterventionError("unsupported intervention kind", "intervene.op_reserved")
        allowed = {"point", "op", "kind", "from_step", "until_step"} | FIELDS[kind]
        if set(request) - allowed:
            raise InterventionError("unexpected intervention fields")
        point = request.get("point")
        _text_checked(point, "point", 255, code="intervene.unknown_point")
        spec = deepcopy(request)
        spec.update(op=kind, kind=kind)
        for key in ("from_step", "until_step"):
            value = spec.get(key)
            if value is not None and (type(value) is not int or not 0 <= value < 2**53):
                raise InterventionError("schedule values must be nonnegative steps", "intervene.bad_range")
        if (spec.get("from_step") is not None and spec.get("until_step") is not None
                and spec["until_step"] <= spec["from_step"]):
            raise InterventionError("until_step must exceed from_step", "intervene.bad_range")
        if not metadata.specification(spec):
            raise InterventionError("intervention metadata exceeds the wire bounds")
        if not resources:
            return spec
        if kind == "ablate":
            names = self._ablation_points
            if names is None:
                listed = points(self.model) if self.adapter is None else self.adapter.points(self.model)
                names = frozenset(p["point"] for p in listed)
            if point not in names:
                raise InterventionError("unknown intervention point", "intervene.unknown_point")
            if point.endswith(".attn_pattern") or point.endswith(".hook_pattern"):
                raise InterventionError("attention patterns are read-only points")
            return spec
        module, width, semantic = self._target(point)
        if "positions" in spec:
            spec["positions"] = list(_indices(spec["positions"], "positions", limit=MAX_POSITIONS))
        if kind == "add":
            vector = spec.get("vector")
            if not isinstance(vector, list) or len(vector) != width or not 0 < len(vector) <= MAX_VECTOR:
                raise InterventionError("vector must match the point width")
            spec["vector"] = [_finite_checked(v, "vector element") for v in vector]
            spec["coefficient"] = _finite_checked(spec.get("coefficient"), "coefficient")
            if not all(math.isfinite(v * spec["coefficient"]) for v in spec["vector"]):
                raise InterventionError("scaled vector must be finite")
        elif kind in ("scale", "clamp"):
            if self.dictionary is None or spec.get("dictionary", self.dictionary_id) != self.dictionary_id:
                raise InterventionError("unknown or absent dictionary", "intervene.unknown_dictionary")
            layer = self._resolve_feature_layer(point, semantic)
            n_features = self._feature_count(layer, point)
            feature = spec.get("feature")
            if type(feature) is not int or not 0 <= feature < n_features:
                raise InterventionError("unknown dictionary feature", "intervene.unknown_feature")
            key = "multiplier" if kind == "scale" else "value"
            spec[key] = _finite_checked(spec.get(key), key)
            spec["dictionary"] = self.dictionary_id
        elif kind == "patch":
            prompt = spec.get("source_prompt")
            _text_checked(prompt, "source_prompt", MAX_PROMPT_BYTES)
            if self.tokenizer is None:
                raise InterventionError("patch requires a configured tokenizer")
            if "source_position" in spec:
                _indices([spec["source_position"]], "source_position", limit=MAX_POSITIONS)
            self._source_inputs(spec)
        parameter = next(module.parameters(), next(self.model.parameters(), None))
        if parameter is not None and parameter.is_floating_point():
            maximum = torch.finfo(parameter.dtype).max
            values = ([*spec["vector"], spec["coefficient"],
                       *(v * spec["coefficient"] for v in spec["vector"])] if kind == "add" else
                      [spec["multiplier"]] if kind == "scale" else
                      [spec["value"]] if kind == "clamp" else [])
            if any(abs(v) > maximum for v in values):
                raise InterventionError("intervention value exceeds the model dtype range")
        if not metadata.specification(spec):
            raise InterventionError("canonical intervention metadata exceeds the wire bounds")
        return spec

    def _source_inputs(self, spec: dict) -> dict:
        try:
            inputs = _tokenize_prompt(self.model, self.tokenizer, spec["source_prompt"])
        except ValueError as error:
            raise InterventionError("invalid patch source: " + str(error)) from error
        except Exception as error:
            raise InterventionError("source prompt tokenization failed") from error
        ids = inputs["input_ids"]
        positions = [spec["source_position"]] if "source_position" in spec else spec.get("positions", [])
        if positions and max(positions) >= ids.shape[1]:
            raise InterventionError("source position is outside the clean prompt")
        return inputs

    def _source(self, spec: dict):
        import torch
        from .interp._memory import autorelease_scope
        prompt, point = spec["source_prompt"], spec["point"]
        key = (prompt, point)
        if key not in self._cache:
            inputs = self._source_inputs(spec)
            module, _width, _semantic = self._target(point)
            captured = {}
            device = next(self.model.parameters(), None)
            dev = device.device if device is not None else torch.device("cpu")

            def save(_mod, _args, output):
                if not isinstance(output, torch.Tensor):
                    raise InterventionError("patch source point must produce a tensor")
                captured[point] = output.detach().clone()

            handle = module.register_forward_hook(save)
            self.preparing_source = True
            try:
                with internal_forward(), self.suspend(), torch.no_grad(), autorelease_scope(dev):
                    if self.resource_guard is not None:
                        self.resource_guard.check("patch source forward start")
                    _run_clean_forward(self.model, inputs)
                    if self.resource_guard is not None:
                        self.resource_guard.check("patch source forward complete")
            finally:
                self.preparing_source = False
                handle.remove()
            tensor = captured.get(point)
            if tensor is None or tensor.ndim != 3 or not torch.isfinite(tensor).all().item():
                raise InterventionError("patch source must be a finite batch-token-width tensor")
            size = tensor.numel() * tensor.element_size()
            pinned = self._source_keys()
            total = sum(v.numel() * v.element_size() for v in self._cache.values()) + size
            for old_key in list(self._cache):
                if total <= MAX_SOURCE_BYTES and len(self._cache) + 1 <= MAX_SOURCE_ENTRIES:
                    break
                if old_key not in pinned:
                    old = self._cache.pop(old_key)
                    total -= old.numel() * old.element_size()
            if total > MAX_SOURCE_BYTES or len(self._cache) + 1 > MAX_SOURCE_ENTRIES:
                res = "source_bytes" if total > MAX_SOURCE_BYTES else "source_entries"
                raise InterventionError("active patch sources exceed the bounded source cache", resource=res)
            self._cache[key] = tensor
        self._cache.move_to_end(key)
        return self._cache[key]

    def _remember_hook_order(self, hook) -> None:
        self._hook_order[hook] = self._next_hook_order
        self._next_hook_order += 1

    def execution_order(self, hook):
        order = self._hook_order.get(hook, math.inf)
        group = getattr(hook, "group", None)
        group_order = self._hook_order.get(group, order) if group is not None else order
        return group_order, order

    def install(self, request, *, on_error=None):
        spec = self.validate(request)
        if spec["op"] == "ablate":
            apply = ablate if self.adapter is None else self.adapter.ablate
            hook = apply(self.model, spec["point"])
            if hasattr(hook, "on_error"):
                hook.on_error = on_error
            self._remember_hook_order(hook)
            self._hooks.add(hook)
            return hook
        module, width, semantic = self._target(spec["point"])
        if spec["op"] in ("scale", "clamp"):
            layer = self._resolve_feature_layer(spec["point"], semantic)
            if self.backend is not None and hasattr(self.backend, "install_feature_hook"):
                hook = self.backend.install_feature_hook(
                    self, spec, layer=layer, width=width, module=module, on_error=on_error,
                    is_muted=_hook_muted, selected_rows=selected_rows)
                self._remember_hook_order(hook)
                self._hooks.add(hook)
                return hook
            raise InterventionError("feature interventions require an interpretability backend",
                                    "intervene.not_supported")
        if spec["op"] == "patch":
            source = self._source(spec)
            positions = spec.get("positions")
            src_pos = [spec["source_position"]] if "source_position" in spec else positions
            if src_pos and max(src_pos) >= source.shape[1]:
                raise InterventionError("source position is outside the clean prompt")
        hook = _TensorHook(self, spec, width, on_error)
        try:
            hook.handles.append(module.register_forward_hook(hook.post))
        except BaseException:
            hook()
            raise
        self._remember_hook_order(hook)
        self._hooks.add(hook)
        return hook


class InterventionReplay:
    """Temporary snapshot replay with muted live hooks and an independent source cache."""

    def __init__(self, executor: InterventionExecutor):
        self.model = executor.model
        self.executor = InterventionExecutor(self.model, adapter=executor.adapter)
        self.executor._ablation_points = executor._ablation_points
        self.executor.dictionary = executor.dictionary
        self.executor.dictionary_id = executor.dictionary_id
        self.executor.tokenizer = executor.tokenizer
        self.executor.resource_guard = executor.resource_guard
        self.executor.backend = executor.backend
        self._effects: list = []

    @property
    def dictionary(self):
        return self.executor.dictionary

    @property
    def tokenizer(self):
        return self.executor.tokenizer

    def specifications(self) -> list[dict]:
        effects = sorted(self._effects, key=lambda effect: self.executor.execution_order(effect[1]))
        return [deepcopy(spec) for spec, hook in effects if not getattr(hook, "removed", False)]

    @contextmanager
    def _scope(self, live_hooks, specifications):
        outer = _ACTIVE_REPLAYS.get()
        hooks = [*live_hooks, *(hook for replay in outer if replay.model is self.model
                                for _, hook in replay._effects)]
        with _mute_hooks(hooks):
            token = _ACTIVE_REPLAYS.set((*outer, self))
            try:
                for spec in specifications:
                    self._effects.append((spec, self.executor.install(spec)))
                self._effects.sort(key=lambda effect: self.executor.execution_order(effect[1]))
                yield self
                for _, hook in self._effects:
                    if getattr(hook, "error", None) is not None:
                        raise InterventionError("snapshot hook failed: " + hook.error,
                                                "intervene.hook_failed")
            finally:
                try:
                    self.executor.close()
                finally:
                    _ACTIVE_REPLAYS.reset(token)
