"""Frozen value paths for the fixture and the Qwen3.5 text decoder."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import json
import sys
import time

import torch
from torch.nn import functional as F

from ._dictionary import model_identity
from ._points import _architecture, capture
from ._memory import MemoryGuard as ProcessMemoryGuard, CleanStop, free_percent, autorelease_scope


def _validate_dictionary(model, dictionary):
    """Check identity, dimensions, and probes before any model forward."""
    data = dictionary.metadata
    if not isinstance(data, dict) or data.get("model_hash") != model_identity(model):
        raise ValueError("dictionary model identity mismatch")
    config = model.config
    if (data.get("layers"), data.get("width")) != (config.num_hidden_layers, config.hidden_size):
        raise ValueError("dictionary dimensions differ from the model")
    if len(dictionary.layers) != data["layers"]:
        raise ValueError("dictionary layer count differs from the model")
    probes = _architecture(model)
    for field, suffix in (("read_points", "mlp_in"), ("write_points", "mlp_out")):
        expected = [name for name in probes if name.endswith("." + suffix)]
        if data.get(field) != expected:
            raise ValueError(f"dictionary {field} differ from model probes")
    latents, k = data.get("latents"), data.get("k")
    if (data.get("activation") != "topk" or type(latents) is not int or type(k) is not int
            or not 1 <= k <= latents):
        raise ValueError("dictionary requires valid TopK metadata")
    shapes = {"W_enc": (latents, config.hidden_size), "W_dec": (latents, config.hidden_size),
              "b_enc": (latents,), "b_dec": (config.hidden_size,)}
    device = next(model.parameters()).device
    for layer in dictionary.layers:
        if layer.k != k:
            raise ValueError("dictionary TopK differs from its metadata")
        for name, shape in shapes.items():
            value = getattr(layer, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"dictionary {name} shape differs from its metadata")
            if value.device != device:
                raise ValueError("dictionary and model must share one device")
    return probes


class MemoryGuard:
    """Keep recent stage records and the process peak, and enforce the demo limit."""

    limit = 12_000_000_000
    record_limit = 128

    def __init__(self, device):
        self.device = torch.device(device)
        self.records = deque(maxlen=self.record_limit)
        self._pressure_time = -float("inf")
        self._pressure_value = 100.
        self._guard = ProcessMemoryGuard(12, self.device, pressure=self._pressure)
        self.check("trace start")

    def _pressure(self):
        now = time.monotonic()
        if now - self._pressure_time >= 15 or self._pressure_value < 20:
            self._pressure_value = free_percent()
            self._pressure_time = now
        return self._pressure_value

    def check(self, stage):
        # The shared guard checks physical footprint and any previously observed peak.
        self._guard.check(stage)
        allocated = torch.mps.current_allocated_memory() if self.device.type == "mps" else 0
        driver = torch.mps.driver_allocated_memory() if self.device.type == "mps" else 0
        rss = self._guard.resident()
        physical = self._guard.footprint()
        current = max(physical if physical is not None else rss + driver, rss, driver, allocated)
        record = {"stage": stage, "allocated_bytes": allocated, "driver_bytes": driver,
                  "process_bytes": current, "peak_memory_bytes": self._guard.peak,
                  "limit_bytes": self.limit}
        self.records.append(record)
        if self.device.type == "mps":
            print("caliper.trace.memory " + json.dumps(record), file=sys.stderr, flush=True)
        if max(current, self._guard.peak) > self.limit:
            raise CleanStop("trace exceeded the 12 GB process or MPS memory limit")
        return current

    def batch_size(self, requested, *, layers, tokens, width, mixer_width):
        if self.device.type != "mps":
            return requested
        current = self.check("target batch admission")
        # Bound live values, adjoints, convolution work, and allocator headroom.
        per_target = 4 * tokens * (width * (layers + 1) * 12 + mixer_width * layers * 8)
        available = self.limit - current - 512_000_000
        admitted = min(requested, int(available // max(1, per_target)))
        if admitted < 1:
            raise CleanStop("no target batch fits the 12 GB MPS memory limit")
        return admitted


def _tensor_input(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))


def _bias(module):
    value = getattr(module, "bias", None)
    return value.detach() if value is not None else None


class FrozenNorm:
    """A recorded RMS denominator, fixed gain, and explicit affine bias."""

    def __init__(self, module, x, *, qwen=False):
        x = x.detach()
        work = x.float() if qwen else x
        scale = torch.rsqrt(work.square().mean(-1, keepdim=True) + module.eps)
        if not qwen and getattr(module, "frozen_scale", None) is not None:
            scale = module.frozen_scale.detach()
        self.scale = scale.detach()
        self.weight = (1 + module.weight.detach().float()) if qwen else module.weight.detach()
        self.bias = _bias(module)
        self.qwen = qwen

    def __call__(self, x, bias=1.):
        work = x.float() if self.qwen else x
        result = ((work * self.scale) * self.weight).to(x.dtype)
        if self.bias is not None:
            result = result + bias * self.bias
        return result


class FrozenAttentionOutput:
    """An output projection with head masks and ordered output transforms."""

    def __init__(self, module, *, ablated=False, heads=()):
        self.weight = module.weight.detach()
        self.bias = _bias(module)
        self.head_mask = torch.zeros(module.in_features, device=self.weight.device, dtype=torch.bool)
        for head, width in heads:
            self.head_mask[head * width:(head + 1) * width] = True
        self.output_mask = torch.tensor(ablated, device=self.weight.device)
        self.transforms = []

    def __call__(self, x, biases, intervention=False):
        x = x.masked_fill(self.head_mask, 0)
        result = F.linear(x, self.weight, self.bias if biases else None)
        result = result.masked_fill(self.output_mask, 0)
        for transform in self.transforms:
            result = transform(result, intervention)
        return result


def _attention_output(model, name, projection_name, interventions, recorded=None):
    from .._ablate import _find
    module = model.get_submodule(name)
    projection = getattr(module, projection_name)
    ablated, heads = False, []
    for spec in interventions:
        if spec.get("op", spec.get("kind")) != "ablate":
            continue
        point = spec["point"]
        if point in (name, f"{name}.{projection_name}"):
            ablated = True
        elif point.startswith(name + "#"):
            # Resolve the same supported slices as the live ablation interface.
            target, head = _find(model, point)
            if target is not projection or head is None:
                raise ValueError("attention head does not target its output projection")
            heads.append(head)
    result = FrozenAttentionOutput(projection, ablated=ablated if recorded is None else False, heads=heads)
    if recorded is not None:
        # Child output hooks run before parent output hooks, even if installed later.
        for point in (f"{name}.{projection_name}", name):
            specifications = [s for s in interventions if s["point"] == point]
            result.transforms.append(FrozenInput(recorded[point + ".before"],
                                                  recorded[point + ".after"], specifications))
    return result


def _validate_ablation_points(model, interventions):
    """Refuse ablations outside the frozen path before a model forward."""
    ablations = [spec["point"] for spec in interventions
                 if spec.get("op", spec.get("kind")) == "ablate"]
    if not ablations:
        return
    from .._ablate import _find
    probes = _architecture(model)
    supported, attention = set(), {}
    for point in probes:
        if not point.endswith(".mlp_in"):
            continue
        base = point.rsplit(".", 1)[0]
        block = model.get_submodule(base)
        delta = getattr(block, "block_type", None) == "linear_attention"
        name = f"{base}.linear_attn" if delta else f"{base}.self_attn"
        projection = "out_proj" if delta else "o_proj"
        supported.update((f"{base}.mlp", f"{base}.mlp_out", f"{base}.mlp_in",
                          f"{base}.resid_pre", name, f"{name}.{projection}"))
        attention[name] = getattr(model.get_submodule(name), projection)
    for point in ablations:
        if point in supported:
            continue
        name, separator, _ = point.rpartition("#")
        if separator and name in attention:
            try:
                target, head = _find(model, point)
            except KeyError:
                pass
            else:
                if target is attention[name] and head is not None:
                    continue
        raise ValueError(f"trace does not support ablation point: {point}")


class FrozenAttention:
    """Linear values with fixed softmax and optional Qwen output gate."""

    def __init__(self, module, x, pattern, *, gate_projection=None, output=None):
        self.pattern = pattern.detach()
        self.head_dim = module.head_dim
        self.heads = pattern.shape[1]
        self.v_weight = module.v_proj.weight.detach()
        self.v_bias = _bias(module.v_proj)
        self.output = output if output is not None else FrozenAttentionOutput(module.o_proj)
        self.gate = None
        if gate_projection is not None:
            projected = gate_projection.reshape(1, x.shape[1], self.heads, self.head_dim * 2)
            self.gate = projected.chunk(2, dim=-1)[1].reshape(1, x.shape[1], -1).sigmoid().detach()
        with torch.no_grad():
            self.offset = self._apply(torch.zeros_like(x), True).detach()
            self.intervention_offset = self._apply(torch.zeros_like(x), False, True).detach()

    def _apply(self, x, biases, intervention=False):
        batch, length, _ = x.shape
        values = F.linear(x, self.v_weight, self.v_bias if biases else None)
        values = values.reshape(batch, length, -1, self.head_dim).transpose(1, 2)
        values = values.repeat_interleave(self.heads // values.shape[1], dim=1)
        result = (self.pattern @ values).transpose(1, 2).reshape(batch, length, -1)
        if self.gate is not None:
            result = result * self.gate
        return self.output(result, biases, intervention)

    def __call__(self, x, bias=1., intervention=1.):
        return self._apply(x, False) + bias * self.offset + intervention * self.intervention_offset


class FrozenDeltaNet:
    """Linear value projection, causal convolution, and signed recurrence.

    The captured operator fixes queries, keys, decay, and beta. Value SiLU
    retains its linear input with a fixed sigmoid. Output RMS and SiLU gates
    use their recorded factors. No routing gradient enters this path.
    """

    def __init__(self, module, x, pattern, qkv, norm_input, output_gate, *, output=None):
        if module.activation not in ("silu", "swish"):
            raise NotImplementedError("DeltaNet tracing requires the SiLU value gate")
        start = 2 * module.key_dim
        self.value_dim = module.value_dim
        self.head_dim = module.head_v_dim
        self.heads = module.num_v_heads
        self.kernel = module.conv_kernel_size
        self.v_weight = module.in_proj_qkv.weight.detach()[start:]
        projection_bias = _bias(module.in_proj_qkv)
        self.v_bias = projection_bias[start:] if projection_bias is not None else None
        self.conv_weight = module.conv1d.weight.detach()[start:]
        conv_bias = _bias(module.conv1d)
        self.conv_bias = conv_bias[start:] if conv_bias is not None else None
        self.output = output if output is not None else FrozenAttentionOutput(module.out_proj)
        self.pattern = pattern.detach()
        length = x.shape[1]
        with torch.no_grad():
            projected = qkv[..., start:].transpose(1, 2).to(self.conv_weight.dtype)
            convolved = F.conv1d(projected, self.conv_weight, self.conv_bias,
                                 padding=self.kernel - 1, groups=self.value_dim)[..., :length]
            self.value_gate = convolved.sigmoid().transpose(1, 2).detach()
            core = norm_input.reshape(1, length, self.heads, self.head_dim).float()
            self.norm_scale = torch.rsqrt(core.square().mean(-1, keepdim=True) + module.norm.variance_epsilon)
            self.norm_weight = module.norm.weight.detach()
            self.output_gate = F.silu(output_gate.float()).reshape(1, length, self.heads, self.head_dim).detach()
            self.offset = self._apply(torch.zeros_like(x), True).detach()
            self.intervention_offset = self._apply(torch.zeros_like(x), False, True).detach()

    def _apply(self, x, biases, intervention=False):
        batch, length, _ = x.shape
        values = F.linear(x, self.v_weight, self.v_bias if biases else None).transpose(1, 2)
        values = F.conv1d(values.to(self.conv_weight.dtype), self.conv_weight,
                          self.conv_bias if biases else None, padding=self.kernel - 1,
                          groups=self.value_dim)[..., :length].transpose(1, 2)
        values = (values * self.value_gate).to(x.dtype)
        values = values.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        core = (self.pattern @ values.float()).transpose(1, 2).to(x.dtype)
        normalized = (core.float() * self.norm_scale).to(x.dtype) * self.norm_weight
        result = (normalized * self.output_gate).to(x.dtype).reshape(batch, length, self.value_dim)
        return self.output(result, biases, intervention)

    def __call__(self, x, bias=1., intervention=1.):
        return self._apply(x, False) + bias * self.offset + intervention * self.intervention_offset


class FrozenInput:
    """An affine input intervention with an explicit constant contribution."""

    def __init__(self, before, after, specifications):
        self.scale = before.new_ones((1, before.shape[1], 1))
        for spec in specifications:
            if spec["op"] in ("patch", "ablate"):
                positions = spec.get("positions", range(before.shape[1]))
                self.scale[:, positions] = 0
        self.offset = (after - before * self.scale).detach() if specifications else torch.zeros_like(before)

    def __call__(self, value, bias=1.):
        return value * self.scale + bias * self.offset


class FrozenFeatures:
    """Compact affine feature writes within the original TopK region."""

    def __init__(self, original, layer, specifications):
        indices = original[0].topk(layer.k, dim=-1).indices
        edited = sorted({s["feature"] for s in specifications if s["op"] in ("scale", "clamp")})
        if edited:
            extra = indices.new_tensor(edited).expand(indices.shape[0], -1)
            indices = torch.cat((indices, extra), dim=-1)
        self.indices = indices.sort(dim=-1).values
        unique = torch.ones_like(self.indices, dtype=torch.bool)
        unique[:, 1:] = self.indices[:, 1:] != self.indices[:, :-1]
        self.original = original[0].gather(-1, self.indices) * unique
        self.region = (self.original > 0).detach()
        self.coefficients = self.region.to(original.dtype)
        self.constants = torch.zeros_like(self.original)
        self.scale = original.new_ones((1, original.shape[1], 1))
        for spec in specifications:
            positions = list(spec.get("positions", range(original.shape[1])))
            kind = spec["op"]
            if kind in ("ablate", "patch"):
                self.coefficients[positions] = 0
                self.constants[positions] = 0
                self.scale[:, positions] = 0
            elif kind in ("scale", "clamp"):
                selected = (self.indices == spec["feature"]) & unique
                rows = torch.zeros_like(selected)
                rows[positions] = True
                selected &= rows
                amount = spec["multiplier"] - 1 if kind == "scale" else -1
                self.coefficients += amount * (selected & self.region).to(original.dtype)
                if kind == "clamp":
                    self.constants += spec["value"] * selected.to(original.dtype)
        self.activations = self.coefficients * self.original + self.constants

    def evaluate(self, read, layer):
        encoder = layer.W_enc.detach()[self.indices].to(read.dtype)
        preactivation = (torch.einsum("btw,tkw->btk", read, encoder) +
                         layer.b_enc.detach()[self.indices].to(read.dtype))
        return preactivation * self.coefficients + self.constants


def _validate_interventions(model, blocks, prefix, interventions):
    """Refuse module edits outside the explicit affine paths."""
    allowed = {}
    for i, block in enumerate(blocks):
        base = f"{prefix}.{i}"
        for suffix in ("resid_pre", "mlp_in", "mlp", "mlp_out"):
            allowed[f"{base}.{suffix}"] = {"ablate", "add", "patch"}
        allowed[f"{base}.mlp_out"] |= {"scale", "clamp"}
        delta = getattr(block, "block_type", None) == "linear_attention"
        name = f"{base}.{'linear_attn' if delta else 'self_attn'}"
        projection = "out_proj" if delta else "o_proj"
        allowed[name] = {"ablate", "add", "patch"}
        if model.config.model_type == "qwen3_5_text" and not delta:
            # The live add and patch hooks require a tensor, but this module returns a tuple.
            allowed[name] = {"ablate"}
        allowed[f"{name}.{projection}"] = {"ablate", "add", "patch"}
    from .._ablate import _find
    for spec in interventions:
        point, kind = spec["point"], spec["op"]
        if kind in allowed.get(point, ()):
            continue
        if kind == "ablate" and "#" in point and point.rsplit("#", 1)[0] in allowed:
            _find(model, point)
            continue
        raise NotImplementedError(f"trace has no exact affine intervention path for {kind} at {point}")


class FrozenPath:
    """Prompt-local affine residual path with independent MLP output writes.

    ``run`` never calls an MLP or a dictionary encoder. Feature targets read
    recorded MLP inputs. Its write arguments are independent source leaves.
    """

    def __init__(self, model, input_ids, dictionary, *, guard=None, interventions=()):
        _validate_ablation_points(model, interventions)
        probes = _validate_dictionary(model, dictionary)
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.guard = guard or MemoryGuard(self.device)
        if model.training:
            raise ValueError("trace requires an evaluation model")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("trace requires float32 or float64 model weights")
        self.qwen = model.config.model_type == "qwen3_5_text"
        if self.qwen and self.dtype != torch.float32:
            raise ValueError("Qwen tracing requires float32 model weights")
        modules = dict(model.named_modules())
        prefix = next(iter(probes)).rsplit(".", 2)[0]
        self.blocks = modules[prefix]
        interventions = [dict(s, op=s.get("op", s.get("kind"))) for s in interventions]
        _validate_interventions(model, self.blocks, prefix, interventions)
        root = model.model if self.qwen else model
        self.layers = len(self.blocks)
        self.tokens = input_ids.shape[1]
        self.width = model.config.hidden_size
        self.mixer_width = max((getattr(getattr(b, "linear_attn", None), "value_dim", self.width)
                                for b in self.blocks), default=self.width)
        recorded = {}
        self.norms, self.post_norms, self.mixers = [], [], []
        with ExitStack() as stack:
            stack.enter_context(autorelease_scope(self.device))
            values = stack.enter_context(capture(model, list(probes)))
            def pre(name):
                def hook(mod, args, kwargs):
                    recorded[name] = _tensor_input(args, kwargs).detach()
                    if name.endswith(".delta_norm"):
                        recorded[name + ".gate"] = (args[1] if len(args) > 1 else kwargs["gate"]).detach()
                return hook
            def out(name):
                def hook(mod, args, result):
                    recorded[name] = (result[0] if isinstance(result, tuple) else result).detach()
                return hook
            def install(handle):
                stack.callback(handle.remove)
            frames = []
            def enter_forward(mod, args):
                # Clean patch sources can run a nested model forward inside an intervention.
                frames.append((dict(values), {key: dict(value) for key, value in values._pending.items()},
                               dict(recorded)) if frames else None)
            def exit_forward(mod, args, output):
                frame = frames.pop()
                if frame is not None:
                    saved_values, saved_pending, saved_recorded = frame
                    values.clear()
                    values.update(saved_values)
                    for key, pending in values._pending.items():
                        pending.clear()
                        pending.update(saved_pending[key])
                    recorded.clear()
                    recorded.update(saved_recorded)
            install(model.register_forward_pre_hook(enter_forward, prepend=True))
            install(model.register_forward_hook(exit_forward, always_call=True))
            install(root.norm.register_forward_pre_hook(pre("final_norm"), with_kwargs=True))
            install(root.embed_tokens.register_forward_hook(out("embedding")))
            for i, block in enumerate(self.blocks):
                install(block.register_forward_pre_hook(pre(f"{i}.residual_source"), with_kwargs=True, prepend=True))
                install(block.mlp.register_forward_pre_hook(pre(f"{i}.mlp_source"), with_kwargs=True, prepend=True))
                install(block.mlp.register_forward_hook(out(f"{i}.mlp_raw"), prepend=True))
                delta = self.qwen and block.block_type == "linear_attention"
                attention_name = f"{prefix}.{i}.{'linear_attn' if delta else 'self_attn'}"
                projection_name = "out_proj" if delta else "o_proj"
                for name in (f"{attention_name}.{projection_name}", attention_name):
                    install(modules[name].register_forward_hook(out(name + ".before"), prepend=True))
                    install(modules[name].register_forward_hook(out(name + ".after")))
                install(block.input_layernorm.register_forward_pre_hook(pre(f"{i}.input_norm"), with_kwargs=True))
                install(block.post_attention_layernorm.register_forward_pre_hook(pre(f"{i}.post_norm"), with_kwargs=True))
                if self.qwen and block.block_type == "linear_attention":
                    install(block.linear_attn.in_proj_qkv.register_forward_hook(out(f"{i}.qkv")))
                    install(block.linear_attn.norm.register_forward_pre_hook(pre(f"{i}.delta_norm"), with_kwargs=True))
                elif self.qwen:
                    install(block.self_attn.q_proj.register_forward_hook(out(f"{i}.q_gate")))
                def stage_hook(*_, i=i):
                    self.guard.check(f"record layer {i}")
                install(block.register_forward_hook(stage_hook))
            with torch.no_grad():
                kwargs = {"use_cache": False, "logits_to_keep": 1} if self.qwen else {}
                self.logits = model(input_ids=input_ids, **kwargs).logits[:, -1].detach()
        self.guard.check("record complete")
        self.embedding = recorded["embedding"]
        self.inputs, self.outputs, self.features, self.errors, self.decoder_biases = [], [], [], [], []
        self.feature_outputs, self.output_offsets = [], []
        self.residual_inputs, self.mlp_inputs = [], []
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                with autorelease_scope(self.device):
                    base = f"{prefix}.{i}"
                    x = values[f"{base}.mlp_in"]
                    y = values[f"{base}.mlp_out"]
                    layer = dictionary.layers[i]
                    a = dictionary.encode(i, x.to(layer.W_enc.dtype)).detach()
                    raw = recorded[f"{i}.mlp_raw"]
                    output = FrozenFeatures(a.to(self.dtype), layer, [s for s in interventions
                        if s["point"] in (f"{base}.mlp", f"{base}.mlp_out")])
                    self.feature_outputs.append(output)
                    transformed = torch.zeros_like(a, dtype=self.dtype).scatter_add(
                        -1, output.indices[None], output.activations[None])
                    reconstruction = dictionary.decode(i, a).to(self.dtype)
                    error = output.scale * (raw - reconstruction)
                    decoder_bias = output.scale * layer.b_dec.detach().to(self.dtype)
                    # Only known output constants enter this term. Errors describe the raw MLP.
                    decoded = transformed.to(layer.W_dec.dtype) @ layer.W_dec.detach()
                    offset = y - decoded.to(self.dtype) - decoder_bias - error
                    self.inputs.append(x)
                    self.outputs.append(y)
                    self.features.append(transformed)
                    self.errors.append(error.detach())
                    self.decoder_biases.append(decoder_bias)
                    self.output_offsets.append(offset.detach())
                    self.norms.append(FrozenNorm(block.input_layernorm, recorded[f"{i}.input_norm"], qwen=self.qwen))
                    self.post_norms.append(FrozenNorm(block.post_attention_layernorm, recorded[f"{i}.post_norm"], qwen=self.qwen))
                    self.residual_inputs.append(FrozenInput(recorded[f"{i}.residual_source"],
                        recorded[f"{i}.input_norm"], [s for s in interventions if s["point"] == f"{base}.resid_pre"]))
                    self.mlp_inputs.append(FrozenInput(recorded[f"{i}.mlp_source"], x,
                        [s for s in interventions if s["point"] == f"{base}.mlp_in"]))
                    pattern = values[f"{base}.attn_pattern"]
                    if self.qwen and block.block_type == "linear_attention":
                        mixer = FrozenDeltaNet(block.linear_attn, recorded[f"{i}.input_norm"], pattern,
                                               recorded[f"{i}.qkv"], recorded[f"{i}.delta_norm"],
                                               recorded[f"{i}.delta_norm.gate"], output=_attention_output(
                                                   model, f"{base}.linear_attn", "out_proj", interventions, recorded))
                    else:
                        mixer = FrozenAttention(block.self_attn, recorded[f"{i}.input_norm"], pattern,
                                                gate_projection=recorded.get(f"{i}.q_gate"), output=_attention_output(
                                                    model, f"{base}.self_attn", "o_proj", interventions, recorded))
                    self.mixers.append(mixer)
                    self.guard.check(f"freeze layer {i}")
            with autorelease_scope(self.device):
                self.final_norm = FrozenNorm(root.norm, recorded["final_norm"], qwen=self.qwen)
                head = model.get_output_embeddings()
                self.unembedding = head.weight.detach()
                self.unembedding_mean = self.unembedding.mean(0)
                self.unembedding_bias = _bias(head)
                self.writes = [(y - b - c).detach() for y, b, c in
                               zip(self.outputs, self.decoder_biases, self.output_offsets)]
                reads, final = self.run(self.embedding, self.writes)
                # Compare every read and all next-token logits, before any edge calculation.
                self.forward_residual = max([float((a - b).abs().max()) for a, b in zip(reads, self.inputs)] +
                                            [float((F.linear(final[:, -1], self.unembedding,
                                                             self.unembedding_bias) - self.logits).abs().max())])
                self.direct_forward_residual = self.forward_residual
                replacement_reads, replacement_final = self.replacement(self.embedding, dictionary)
                self.replacement_forward_residual = max(
                    max(float((a - b).abs().max()) for a, b in zip(replacement_reads, self.inputs)),
                    float((F.linear(replacement_final[:, -1], self.unembedding,
                                    self.unembedding_bias) - self.logits).abs().max()))
                self.forward_residual = max(self.direct_forward_residual, self.replacement_forward_residual)
                tolerance = 1e-9 if self.dtype == torch.float64 else 2e-2
                if not self.forward_residual <= tolerance:
                    raise ValueError(f"frozen replacement forward mismatch: {self.forward_residual:.6g}")
                self.guard.check("forward agreement")
                # Encoder and unembedding biases are target-local. Input edits have their own constant term.
                zero = torch.zeros_like(self.embedding).expand(4, -1, -1)
                masks = torch.eye(4, device=self.device, dtype=self.dtype).reshape(4, 4, 1, 1)
                self.bias_reads, self.bias_final = self.run(zero, [zero] * self.layers, bias_masks=masks)

    def run(self, embedding, writes, *, bias_masks=None):
        """Return MLP reads and final normalized residual for independent writes.

        Bias mask rows are decoder, normalization, attention, and intervention contributions.
        A zero mask gives the strictly linear residual path.
        """
        masks = (1., 1., 1., 1.) if bias_masks is None else bias_masks
        decoder, norm, attention = masks[:3]
        intervention = masks[3] if len(masks) == 4 else 0.
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer, write) in enumerate(zip(self.norms, self.post_norms, self.mixers, writes)):
            with autorelease_scope(self.device):
                x = self.residual_inputs[i](x, intervention)
                x = x + mixer(input_norm(x, norm), attention, intervention)
                reads.append(self.mlp_inputs[i](post_norm(x, norm), intervention))
                x = x + write + decoder * self.decoder_biases[i] + intervention * self.output_offsets[i]
                self.guard.check(f"frozen layer {i}")
        return reads, self.final_norm(x, norm)


    def replacement(self, embedding, dictionary):
        """Evaluate transcoders with the recorded active masks and constant errors.

        This affine forward verifies the complete local replacement. Direct
        edges use ``run`` with independent writes and never differentiate here.
        """
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer) in enumerate(zip(self.norms, self.post_norms, self.mixers)):
            with autorelease_scope(self.device):
                x = self.residual_inputs[i](x)
                x = x + mixer(input_norm(x))
                read = self.mlp_inputs[i](post_norm(x))
                reads.append(read)
                layer = dictionary.layers[i]
                output = self.feature_outputs[i]
                decoder = layer.W_dec.detach()[output.indices].to(self.dtype)
                activation = output.evaluate(read, layer)
                x = (x + torch.einsum("btk,tkw->btw", activation, decoder) +
                     self.decoder_biases[i] + self.errors[i] + self.output_offsets[i])
                self.guard.check(f"replacement layer {i}")
        return reads, self.final_norm(x)
