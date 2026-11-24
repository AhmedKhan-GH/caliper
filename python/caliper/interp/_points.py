"""Semantic decoder probes with device-resident capture buffers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import weakref


@dataclass(frozen=True)
class Probe:
    """A semantic point and its actual module path."""

    point: str
    module: str
    event: str
    layer: int
    family: str


_ACTIVE = weakref.WeakKeyDictionary()
_EAGER = {}


def _architecture(model):
    config = getattr(model, "config", None)
    family = getattr(config, "model_type", None)
    modules = dict(model.named_modules())
    if family == "caliper_tiny":
        prefix = "layers"
    elif family == "qwen3_5_text":
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5TextModel,
        )
        decoders = [(name, mod) for name, mod in modules.items() if isinstance(mod, Qwen3_5TextModel)]
        if len(decoders) != 1:
            raise ValueError("Qwen probes require exactly one text decoder")
        prefix = f"{decoders[0][0]}.layers".lstrip(".")
    else:
        raise NotImplementedError(f"semantic probes do not support decoder {family or type(model).__name__!r}")
    layers = modules.get(prefix)
    if layers is None or len(layers) != config.num_hidden_layers:
        raise ValueError("decoder layer paths do not match the model config")
    probes = {}
    for index, block in enumerate(layers):
        base = f"{prefix}.{index}"
        if family == "qwen3_5_text":
            if not isinstance(block, Qwen3_5DecoderLayer):
                raise ValueError(f"unsupported Qwen decoder block at {base}")
            mixer = "linear_attn" if block.block_type == "linear_attention" else "self_attn"
            pattern_family = "delta" if mixer == "linear_attn" else "qwen_full"
        else:
            mixer, pattern_family = "self_attn", "fixture"
        for suffix, path, event in (
            ("resid_pre", base, "input"),
            ("mlp_in", f"{base}.mlp", "input"),
            ("mlp_out", f"{base}.mlp", "output"),
            ("attn_pattern", f"{base}.{mixer}", "pattern"),
        ):
            if path not in modules:
                raise ValueError(f"missing decoder module {path}")
            if event == "pattern" and pattern_family == "fixture" and f"{path}.pattern" not in modules:
                raise ValueError(f"missing fixture pattern module {path}.pattern")
            point = f"{base}.{suffix}"
            probes[point] = Probe(point, path, event, index, pattern_family)
    return probes


def semantic_points(model, *, required=True):
    """Return semantic metadata, or no additions for an unsupported ordinary module."""
    from .._ablate import _shapes
    try:
        probes = _architecture(model)
    except NotImplementedError:
        if required:
            raise
        return []
    modules = dict(model.named_modules())
    shapes = _shapes.get(model, {})
    return [{"point": p.point, "kind": "module", "module": p.module, "head": None,
             "type": type(modules[p.module]).__name__,
             "shape": list(shapes[p.point]) if p.point in shapes else None,
             "probe": p.point.rsplit(".", 1)[1], "layer": p.layer,
             "pattern": ("signed_value_mixing" if p.family == "delta" else "softmax")
             if p.event == "pattern" else None} for p in probes.values()]


def points(model):
    """Return module, head, and semantic point dictionaries for a supported decoder.

    Semantic rows add ``probe``, ``layer``, and ``pattern`` to the adapter fields.
    DeltaNet patterns map post-convolution values to the recurrent output before normalization.
    Their shape is ``(batch, value_heads, target_position, source_position)``.
    """
    _architecture(model)
    from .._ablate import points as adapter_points
    return adapter_points(model)


def _first_tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((v for v in value if isinstance(v, torch.Tensor)), None)
    return None


def _input(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))


def _delta_pattern(module, values):
    """Compute the frozen signed value operator from the actual projection outputs.

    The functional convolution bypasses module hooks in Transformers. Repeat that convolution
    from its captured input and weights. The recurrence below transports value coefficients.
    It uses the same float32 normalization, decay, and beta as the installed DeltaNet kernels.
    """
    import torch
    from torch.nn import functional as F
    from transformers.activations import ACT2FN

    qkv = values["qkv"].transpose(1, 2)
    length = qkv.shape[-1]
    qkv = F.conv1d(qkv.to(module.conv1d.weight.dtype), module.conv1d.weight,
                   module.conv1d.bias, padding=module.conv_kernel_size - 1,
                   groups=module.conv_dim)[..., :length]
    qkv = ACT2FN[module.activation](qkv).to(values["qkv"].dtype).transpose(1, 2)
    q, k, _ = qkv.split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
    batch = q.shape[0]
    q, k = [x.reshape(batch, length, module.num_k_heads, module.head_k_dim)
            .transpose(1, 2).float() for x in (q, k)]
    q, k = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    repeat = module.num_v_heads // module.num_k_heads
    q, k = [x.repeat_interleave(repeat, dim=1) for x in (q, k)]
    q = q * module.head_k_dim ** -0.5
    beta = values["b"].sigmoid().transpose(1, 2).float()
    decay = (-module.A_log.float().exp() * F.softplus(values["a"].float() + module.dt_bias))
    decay = decay.transpose(1, 2).exp()
    # Column j of state is the coefficient of value j in the recurrent state.
    state = q.new_zeros(batch, module.num_v_heads, module.head_k_dim, length)
    rows = []
    for t in range(length):
        state = state * decay[:, :, t, None, None]
        kt = k[:, :, t, :, None]
        memory = (state * kt).sum(-2)
        delta = -memory * beta[:, :, t, None]
        delta[:, :, t] += beta[:, :, t]
        state = state + kt * delta.unsqueeze(-2)
        rows.append((state * q[:, :, t, :, None]).sum(-2))
    return torch.stack(rows, dim=-2)


class Capture(dict):
    """A context and point-to-tensor map for the latest forward pass.

    Values detach from autograd and share device storage with the observed tensors.
    Exit removes hooks and retains results. An error or watch removal also clears results.
    ``close(clear=True)`` releases captured tensors explicitly. Contexts are single-use.
    """

    def __init__(self, model, requested):
        super().__init__()
        self._model = weakref.ref(model)
        self._probes = _architecture(model)
        if isinstance(requested, str):
            requested = [requested]
        self.requested = tuple(dict.fromkeys(p["point"] if isinstance(p, Mapping) else p for p in requested))
        from .._ablate import _find
        for point in self.requested:
            if point not in self._probes:
                _find(model, point)
        self._handles = []
        self._configs = []
        self._pending = {}
        self._entered = False
        self._closed = False

    def _save(self, name, value):
        tensor = _first_tensor(value)
        if tensor is None:
            raise TypeError(f"probe {name!r} did not produce a tensor")
        self[name] = tensor.detach()
        from .._ablate import _shapes
        model = self._model()
        if model is not None:
            _shapes.setdefault(model, {})[name] = tuple(tensor.shape)

    def _output(self, name):
        def hook(module, args, output):
            self._save(name, output)
        return hook

    def _pre(self, name, head=None):
        def hook(module, args, kwargs):
            tensor = _input(args, kwargs)
            if head is not None:
                h, width = head
                tensor = tensor[..., h * width:(h + 1) * width]
            self._save(name, tensor)
        return hook

    def _delta(self, probe, module):
        values = self._pending[probe.point] = {}

        def start(mod, args, kwargs):
            values.clear()
            cache = kwargs.get("cache", kwargs.get("cache_params", args[1] if len(args) > 1 else None))
            if cache is not None and cache.get_conv_state(mod.layer_idx, state_idx=0) is not None:
                raise ValueError("DeltaNet pattern capture requires a complete prompt without a populated cache")
            if kwargs.get("cu_seq_lens_q") is not None:
                raise ValueError("DeltaNet pattern capture does not support packed sequences")

        self._handles.append(module.register_forward_pre_hook(start, with_kwargs=True))
        for name, child in (("qkv", module.in_proj_qkv), ("a", module.in_proj_a), ("b", module.in_proj_b)):
            def keep(mod, args, output, name=name):
                values[name] = output.detach()
            self._handles.append(child.register_forward_hook(keep))

        def finish(mod, args, output):
            import torch
            with torch.no_grad():
                self._save(probe.point, _delta_pattern(mod, values))
            values.clear()
        self._handles.append(module.register_forward_hook(finish))

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("a capture context can only be entered once")
        self._entered = True
        model = self._model()
        if model is None:
            raise RuntimeError("the capture model no longer exists")
        modules = dict(model.named_modules())
        from .._ablate import _find
        _ACTIVE.setdefault(model, weakref.WeakValueDictionary())[id(self)] = self
        try:
            # A new call must not leave any result from a prior or partial call.
            self._handles.append(model.register_forward_pre_hook(lambda *_: self.clear()))
            for name in self.requested:
                probe = self._probes.get(name)
                if probe is None:
                    module, head = _find(model, name)
                    handle = (module.register_forward_hook(self._output(name)) if head is None else
                              module.register_forward_pre_hook(self._pre(name, head), with_kwargs=True))
                    self._handles.append(handle)
                    continue
                module = modules[probe.module]
                if probe.event == "input":
                    self._handles.append(module.register_forward_pre_hook(self._pre(name), with_kwargs=True))
                elif probe.event == "output":
                    self._handles.append(module.register_forward_hook(self._output(name)))
                elif probe.family == "fixture":
                    self._handles.append(module.pattern.register_forward_hook(self._output(name)))
                elif probe.family == "delta":
                    self._delta(probe, module)
                else:
                    # Eager attention exposes the real weights, including masks and dropout.
                    # The shared decoder config also selects its causal mask implementation.
                    config = module.config
                    if not any(c is config for c in self._configs):
                        state = _EAGER.setdefault(id(config), [config, config._attn_implementation, 0])
                        state[2] += 1
                        self._configs.append(config)
                        config._attn_implementation = "eager"
                    def pattern(mod, args, output, name=name):
                        self._save(name, output[1])
                    # Observe weights before output hooks can ablate the attention tuple.
                    self._handles.append(module.register_forward_hook(pattern, prepend=True))
        except BaseException:
            self.close(clear=True)
            raise
        return self

    def close(self, *, clear=False):
        """Remove this context's hooks and restore its attention configuration."""
        while self._handles:
            self._handles.pop().remove()
        while self._configs:
            config = self._configs.pop()
            state = _EAGER[id(config)]
            state[2] -= 1
            if state[2] == 0:
                config._attn_implementation = state[1]
                del _EAGER[id(config)]
        self._pending.clear()
        model = self._model()
        if model is not None and model in _ACTIVE:
            _ACTIVE[model].pop(id(self), None)
        if clear:
            self.clear()
        self._closed = True

    def __exit__(self, exc_type, exc, traceback):
        self.close(clear=exc_type is not None)
        return False


def capture(model, points):
    """Create a capture context from point names or point dictionaries."""
    return Capture(model, points)


def clear_captures(model):
    """Remove active capture hooks and tensors when a watch ends."""
    for context in list(_ACTIVE.get(model, {}).values()):
        context.close(clear=True)
