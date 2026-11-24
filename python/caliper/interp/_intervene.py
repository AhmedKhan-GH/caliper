"""Validated interventions on live model tensors, with scoped batch selection."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import math
import weakref

import torch

from ._points import _architecture, _input, capture

MAX_POSITIONS = 32
MAX_VECTOR = 16384
MAX_PROMPT_BYTES = 4096
KINDS = ("ablate", "add", "scale", "clamp", "patch")
_SUSPENDED = ContextVar("caliper_interventions_suspended", default=frozenset())
_ROWS = ContextVar("caliper_intervention_rows", default={})
_MUTED_HOOKS = ContextVar("caliper_muted_intervention_hooks", default=frozenset())
_ACTIVE_REPLAYS = ContextVar("caliper_intervention_replays", default=())


class InterventionError(ValueError):
    """A request refusal with a stable wire error code."""

    def __init__(self, message, code="intervene.bad_request"):
        super().__init__(message)
        self.code = code


def _indices(value, name, *, limit=None):
    if not isinstance(value, (list, tuple)) or any(
        type(i) is not int or i < 0 or (limit is not None and i >= limit) for i in value
    ) or len(set(value)) != len(value) or (limit is not None and len(value) > limit):
        raise InterventionError(f"{name} must contain unique nonnegative integer indices")
    return tuple(value)


def _text(value, name, maximum, *, code="intervene.bad_request"):
    try:
        valid = isinstance(value, str) and 0 < len(value.encode("utf-8")) <= maximum
    except UnicodeEncodeError:
        valid = False
    if not valid:
        raise InterventionError(f"{name} must contain 1 to {maximum} UTF-8 bytes", code)
    return value


def _finite(value, name):
    if type(value) not in (int, float):
        raise InterventionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise InterventionError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise InterventionError(f"{name} must be a finite number")
    return result


def selected_rows(model):
    """Return None for all rows, or a tuple for the current model context."""
    if id(model) in _SUSPENDED.get():
        return ()
    return _ROWS.get().get(id(model))


def _hook_muted(hook):
    return hook in _MUTED_HOOKS.get()


@contextmanager
def _mute_hooks(hooks):
    """Mute these hook objects in this context without a lifecycle change."""
    token = _MUTED_HOOKS.set(_MUTED_HOOKS.get().union(hooks))
    try:
        yield
    finally:
        _MUTED_HOOKS.reset(token)


@contextmanager
def suspend(model):
    """Suspend every intervention on this model within the current context."""
    token = _SUSPENDED.set(_SUSPENDED.get() | {id(model)})
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


@contextmanager
def rows(model, indices):
    """Restrict interventions to batch rows, intersected with any outer scope."""
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
    """Build the batch mask for legacy module and head ablation."""
    chosen = selected_rows(model)
    if chosen is None:
        return None
    if tensor.ndim == 0:
        raise InterventionError("batch row selection requires a batch dimension")
    if any(i >= tensor.shape[0] for i in chosen):
        raise InterventionError("a selected batch row is outside the tensor")
    mask = torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    mask[list(chosen)] = True
    return mask.reshape((-1,) + (1,) * (tensor.ndim - 1))


class InterventionService:
    """Validate specifications and install removable hooks on one real model.

    Dictionary weights stay on their device. Specifications contain only JSON values.
    Clean patch sources use a separate forward pass under ``suspend()``.
    """

    def __init__(self, model):
        self.model = model
        self.dictionary = None
        self.dictionary_id = None
        self.tokenizer = None
        self._cache = OrderedDict()
        self._cache_identity = None
        self.preparing_source = False
        self._hooks = weakref.WeakSet()

    def suspend(self):
        return suspend(self.model)

    def rows(self, indices):
        return rows(self.model, indices)

    @property
    def suspended(self):
        return id(self.model) in _SUSPENDED.get()

    def configure(self, *, dictionary=None, tokenizer=None, dictionary_id=None):
        """Bind a validated dictionary and tokenizer before requests arrive.

        Reconfiguration removes existing hooks. Controls must also end their records.
        A failed validation leaves the prior configuration and hooks intact.
        """
        if dictionary is not None:
            from ._dictionary import Dictionary, model_identity
            if not isinstance(dictionary, Dictionary):
                raise InterventionError("dictionary must be a loaded Dictionary", "intervene.unknown_dictionary")
            meta = dictionary.metadata
            if meta.get("model_hash") != model_identity(self.model):
                raise InterventionError("dictionary model identity mismatch", "intervene.unknown_dictionary")
            probes = _architecture(self.model)
            reads = [p.point for p in probes.values() if p.point.endswith(".mlp_in")]
            writes = [p.point for p in probes.values() if p.point.endswith(".mlp_out")]
            width = self.model.config.hidden_size
            if (meta.get("read_points") != reads or meta.get("write_points") != writes
                    or len(dictionary.layers) != len(writes) or meta.get("width") != width):
                raise InterventionError("dictionary probes or dimensions differ from the model")
            for layer in dictionary.layers:
                if (layer.W_enc.ndim != 2 or layer.W_dec.shape != layer.W_enc.shape
                        or layer.W_dec.shape[1] != width
                        or layer.b_enc.shape != (layer.W_enc.shape[0],)
                        or layer.b_dec.shape != (width,)):
                    raise InterventionError("dictionary weight dimensions differ from its probes")
            dictionary_id = meta["model_hash"] if dictionary_id is None else dictionary_id
            _text(dictionary_id, "dictionary identifier", 255)
        elif dictionary_id is not None:
            raise InterventionError("dictionary identifier requires a dictionary", "intervene.unknown_dictionary")
        if tokenizer is not None and not callable(tokenizer):
            raise InterventionError("tokenizer must be callable")
        self.close()
        self.dictionary, self.dictionary_id, self.tokenizer = dictionary, dictionary_id, tokenizer

    def invalidate_cache(self):
        """Release patch sources after a prompt or model change."""
        self._cache.clear()
        self._cache_identity = None

    def close(self):
        for hook in list(self._hooks):
            hook()
        self.invalidate_cache()

    def _target(self, point):
        from .._ablate import _find, _shapes
        try:
            probes = _architecture(self.model)
        except NotImplementedError:
            probes = {}
        probe = probes.get(point)
        if probe is not None:
            if probe.event == "pattern":
                raise InterventionError("attention patterns are read-only points")
            return (dict(self.model.named_modules())[probe.module], probe.event,
                    self.model.config.hidden_size, probe)
        try:
            module, head = _find(self.model, point)
        except KeyError:
            raise InterventionError("unknown intervention point", "intervene.unknown_point") from None
        if head is not None:
            raise InterventionError("head points support ablate only")
        shape = _shapes.get(self.model, {}).get(point)
        width = getattr(module, "out_features", shape[-1] if shape else None)
        if width is None:
            raise InterventionError("point width is unknown; capture its output first")
        return module, "output", width, None

    def validate(self, request):
        """Return a detached, canonical specification without a hook or forward pass."""
        if not isinstance(request, dict):
            raise InterventionError("intervention set must be an object")
        op, kind = request.get("op"), request.get("kind")
        if op is not None and kind is not None and op != kind:
            raise InterventionError("op and kind conflict")
        kind = kind if kind is not None else op
        if kind not in KINDS:
            raise InterventionError("unsupported intervention kind", "intervene.op_reserved")
        fields = {"point", "op", "kind", "from_step", "until_step"}
        fields |= {"add": {"vector", "coefficient", "positions"},
                   "scale": {"feature", "multiplier", "positions", "dictionary"},
                   "clamp": {"feature", "value", "positions", "dictionary"},
                   "patch": {"source_prompt", "source_position", "positions"},
                   "ablate": set()}[kind]
        if set(request) - fields:
            raise InterventionError("unexpected intervention fields")
        point = request.get("point")
        _text(point, "point", 255, code="intervene.unknown_point")
        spec = deepcopy(request)
        spec.update(op=kind, kind=kind)
        for key in ("from_step", "until_step"):
            value = spec.get(key)
            if value is not None and (type(value) is not int or not 0 <= value < 2**53):
                raise InterventionError("schedule values must be nonnegative steps", "intervene.bad_range")
        if (spec.get("from_step") is not None and spec.get("until_step") is not None
                and spec["until_step"] <= spec["from_step"]):
            raise InterventionError("until_step must exceed from_step", "intervene.bad_range")
        if kind == "ablate":
            from .._ablate import points
            if point not in {p["point"] for p in points(self.model)}:
                raise InterventionError("unknown intervention point", "intervene.unknown_point")
            if point.endswith(".attn_pattern"):
                raise InterventionError("attention patterns are read-only points")
            return spec
        module, _, width, probe = self._target(point)
        if "positions" in spec:
            spec["positions"] = list(_indices(spec["positions"], "positions", limit=MAX_POSITIONS))
        if kind == "add":
            vector = spec.get("vector")
            if not isinstance(vector, list) or len(vector) != width or not 0 < len(vector) <= MAX_VECTOR:
                raise InterventionError("vector must match the point width")
            spec["vector"] = [_finite(v, "vector element") for v in vector]
            spec["coefficient"] = _finite(spec.get("coefficient"), "coefficient")
            if not all(math.isfinite(v * spec["coefficient"]) for v in spec["vector"]):
                raise InterventionError("scaled vector must be finite")
        elif kind in ("scale", "clamp"):
            if self.dictionary is None or spec.get("dictionary", self.dictionary_id) != self.dictionary_id:
                raise InterventionError("unknown or absent dictionary", "intervene.unknown_dictionary")
            if probe is None or not point.endswith(".mlp_out"):
                raise InterventionError("features require an mlp_out point")
            feature = spec.get("feature")
            if type(feature) is not int or not 0 <= feature < self.dictionary.layers[probe.layer].W_dec.shape[0]:
                raise InterventionError("unknown dictionary feature", "intervene.unknown_feature")
            key = "multiplier" if kind == "scale" else "value"
            spec[key] = _finite(spec.get(key), key)
            spec["dictionary"] = self.dictionary_id
        elif kind == "patch":
            prompt = spec.get("source_prompt")
            _text(prompt, "source_prompt", MAX_PROMPT_BYTES)
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
        return spec

    def _source_inputs(self, spec):
        try:
            inputs = self.tokenizer(spec["source_prompt"], return_tensors="pt")
            ids = inputs.get("input_ids")
        except Exception as error:
            raise InterventionError("source prompt tokenization failed") from error
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= MAX_POSITIONS:
            raise InterventionError("patch source must contain 1 to 32 tokens in one row")
        positions = [spec["source_position"]] if "source_position" in spec else spec.get("positions", [])
        if positions and max(positions) >= ids.shape[1]:
            raise InterventionError("source position is outside the clean prompt")
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise InterventionError("tokenizer output values must be tensors")
        return inputs

    def _identity(self, prompt):
        # Version counters detect optimizer steps and normal in-place weight updates.
        tensors = []
        for value in (*self.model.parameters(), *self.model.buffers()):
            try:
                version = value._version
            except RuntimeError:  # tensors created inside inference_mode have no counter
                version = None
            tensors.append((id(value), version, value.device, value.dtype, tuple(value.shape)))
        return (id(self.model), id(self.tokenizer), prompt,
                tuple((id(m), m.training) for m in self.model.modules()), tuple(tensors))

    def _source(self, spec):
        identity = self._identity(spec["source_prompt"])
        if identity != self._cache_identity:
            self.invalidate_cache()
            self._cache_identity = identity
        point = spec["point"]
        if point not in self._cache:
            inputs = self._source_inputs(spec)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            self.preparing_source = True
            try:
                with self.suspend(), torch.no_grad(), capture(self.model, [point]) as values:
                    self.model(**inputs)
                    source = values[point].detach().clone()
                values.close(clear=True)
            finally:
                self.preparing_source = False
            if source.ndim != 3 or not torch.isfinite(source).all().item():
                raise InterventionError("patch source must be a finite batch-token-width tensor")
            self._cache[point] = source
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(point)
        return self._cache[point]

    def install(self, request, *, on_error=None):
        """Install a validated hook. The caller selects the step boundary."""
        spec = self.validate(request)
        if spec["op"] == "ablate":
            from .._ablate import ablate
            hook = ablate(self.model, spec["point"])
            hook.on_error = on_error
        else:
            module, event, width, probe = self._target(spec["point"])
            if spec["op"] == "patch":
                source = self._source(spec)
                positions = spec.get("positions")
                sources = [spec["source_position"]] if "source_position" in spec else positions
                if sources and max(sources) >= source.shape[1]:
                    raise InterventionError("source position is outside the clean prompt")
            hook = Intervention(self, spec, width, probe, on_error)
            try:
                if event == "input":
                    hook.handles.append(module.register_forward_pre_hook(hook.pre, with_kwargs=True))
                else:
                    hook.handles.append(module.register_forward_hook(hook.post, with_kwargs=True))
            except BaseException:
                hook()
                raise
        self._hooks.add(hook)
        return hook


class _InterventionReplay:
    """Private snapshot effects with borrowed resources and an independent source cache."""

    def __init__(self, executor):
        self.model = executor.model
        self.executor = InterventionService(self.model)
        self.executor.dictionary = executor.dictionary
        self.executor.dictionary_id = executor.dictionary_id
        self.executor.tokenizer = executor.tokenizer
        self._effects = []

    @property
    def dictionary(self):
        return self.executor.dictionary

    @property
    def tokenizer(self):
        return self.executor.tokenizer

    def specifications(self):
        """Return detached specifications for effects that remain installed."""
        return [deepcopy(spec) for spec, hook in self._effects if not hook.removed]

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
                yield self
                for _, hook in self._effects:
                    if hook.error is not None:
                        raise InterventionError("snapshot hook failed: " + hook.error,
                                                "intervene.hook_failed")
            finally:
                try:
                    self.executor.close()
                finally:
                    _ACTIVE_REPLAYS.reset(token)


class Intervention:
    """One failure-contained hook with no retained live activation."""

    def __init__(self, service, spec, width, probe, on_error):
        self.service, self.spec = service, spec
        self.width, self.probe = width, probe
        self.on_error = on_error
        self.handles = []
        self.calls, self.errors, self.error = 0, 0, None

    @property
    def removed(self):
        return not self.handles

    def __call__(self):
        while self.handles:
            self.handles.pop().remove()
        self.service.invalidate_cache()
        self.on_error = None

    def _failed(self, error):
        self.errors += 1
        self.error = f"{type(error).__name__}: {error}"[:300]
        callback = self.on_error
        self()
        if callback is not None:
            try:
                callback(self.error)
            except Exception:
                pass  # Error reporting must never escape into the model forward.

    def _apply(self, value, live_input):
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != self.width:
            raise InterventionError("point must produce a batch-token-width tensor of the expected width")
        if not value.is_floating_point():
            raise InterventionError("point tensor must be floating point")
        chosen = selected_rows(self.service.model)
        batch = list(range(value.shape[0])) if chosen is None else list(chosen)
        positions = self.spec.get("positions", list(range(value.shape[1])))
        if any(i >= value.shape[0] for i in batch) or any(i >= value.shape[1] for i in positions):
            raise InterventionError("selected row or position is outside the live tensor")
        if not batch or not positions:
            return value
        b = torch.tensor(batch, device=value.device)[:, None]
        p = torch.tensor(positions, device=value.device)[None, :]
        selected = value[b, p]
        kind = self.spec["op"]
        if kind == "add":
            vector = value.new_tensor(self.spec["vector"])
            changed = selected + self.spec["coefficient"] * vector
        elif kind in ("scale", "clamp"):
            if not isinstance(live_input, torch.Tensor) or live_input.shape != value.shape:
                raise InterventionError("live MLP input does not match the output shape")
            layer = self.probe.layer
            dictionary = self.service.dictionary
            weights = dictionary.layers[layer]
            x = live_input[b, p].to(device=weights.W_enc.device, dtype=weights.W_enc.dtype)
            activation = dictionary.encode(layer, x)[..., self.spec["feature"]]
            amount = ((self.spec["multiplier"] - 1) * activation if kind == "scale" else
                      self.spec["value"] - activation)
            decoder = weights.W_dec[self.spec["feature"]]
            delta = (amount.unsqueeze(-1) * decoder).to(device=value.device, dtype=value.dtype)
            changed = selected + delta
        else:
            source = self.service._source(self.spec)
            source_positions = [self.spec["source_position"]] * len(positions) if "source_position" in self.spec else positions
            if max(source_positions) >= source.shape[1] or source.shape[-1] != self.width:
                raise InterventionError("source positions or width do not match the patch")
            changed = source[:, source_positions].to(device=value.device, dtype=value.dtype).expand(len(batch), -1, -1)
        if not torch.isfinite(changed).all().item():
            raise InterventionError("intervention produced a nonfinite tensor")
        result = value.clone()
        result[b, p] = changed
        return result

    def pre(self, module, args, kwargs):
        if self.removed or _hook_muted(self) or selected_rows(self.service.model) == ():
            return None
        self.calls += 1
        try:
            value = self._apply(_input(args, kwargs), None)
            if args:
                return (value,) + args[1:], kwargs
            key = "hidden_states" if "hidden_states" in kwargs else "x"
            return args, dict(kwargs, **{key: value})
        except Exception as error:
            self._failed(error)
            return None

    def post(self, module, args, kwargs, output):
        if self.removed or _hook_muted(self) or selected_rows(self.service.model) == ():
            return None
        self.calls += 1
        try:
            return self._apply(output, _input(args, kwargs))
        except Exception as error:
            self._failed(error)
            return None
