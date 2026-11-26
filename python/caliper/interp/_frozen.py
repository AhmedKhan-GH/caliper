"""Frozen value paths for supported decoder families."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import json
import math
import sys
import time

import torch
from torch.nn import functional as F

from ._dictionary import _encoder_matches_dtype, _reference_coordinates, model_identity
from ._families import model_family
from ._points import _architecture, capture
from ._output import internal_forward
from ._memory import (MemoryGuard as ProcessMemoryGuard, BudgetMemory, CleanStop,
                      allocation_bytes, free_percent, autorelease_scope, family_memory_limit,
                      host_reserve_for_model)


# Mixed storage precision needs at most this many converted encoder elements,
# or one row when the residual width exceeds the block limit.
_ENCODER_BLOCK_ELEMENTS = 1_048_576
_FINITE_BLOCK_BYTES = 32 * 1024 * 1024
_FINITE_BATCH_BLOCKS = 4


def _no_check(stage):
    pass


def _require_finite(value, name, check=_no_check, memory=None):
    """Check every value with bounded workspace and scalar readbacks."""
    ieee = {torch.bfloat16: (torch.int16, 0x7F80),
            torch.float16: (torch.int16, 0x7C00),
            torch.float32: (torch.int32, 0x7F800000)}.get(value.dtype)
    if value.layout != torch.strided or value.is_neg() or value.is_conj():
        ieee = None
    # Allow two input temporaries at float32 width or wider, plus four int32
    # masks per element. This includes copies, promotion, abs, and reduction.
    # Four blocks reserve at most 128 MiB of workspace, plus scalar results.
    # Retain this admission and block size for the integer path as well.
    # Its dtype view aliases the input. The explicit exponent buffer has N
    # elements. Its maximum and comparison are scalars. No floating arithmetic changes.
    element_bytes = 2 * max(4, value.element_size()) + 4 * 4
    block_elements = max(1, _FINITE_BLOCK_BYTES // element_bytes)
    budget = None if memory is None else BudgetMemory(memory, memory.limit / 1e9)

    def checked(stage, workspace):
        check(stage)
        if budget is not None:
            from ._admission import _host_reserve
            budget.check(stage)
            if allocation_bytes(budget) + workspace > budget.limit:
                raise CleanStop("finite validation allocation exceeds the process memory limit")
            _host_reserve(workspace, stage, memory=budget)
            # Keep any allocator-specific admission from the supplied owner.
            reserve = getattr(memory, "reserve", None)
            if reserve is not None:
                reserve(stage, workspace)

    pending = [value.detach()]
    while pending:
        with autorelease_scope(value.device):
            blocks = []
            while pending and len(blocks) < _FINITE_BATCH_BLOCKS:
                block = pending.pop()
                if block.numel() > block_elements:
                    axis = max(range(block.ndim), key=lambda i: block.shape[i])
                    middle = block.shape[axis] // 2
                    pending.append(block.narrow(axis, middle, block.shape[axis] - middle))
                    pending.append(block.narrow(axis, 0, middle))
                    continue
                blocks.append(block)
            # Reserve the whole batch before its first mask. Device commands can
            # retain temporaries until readback, even after Python releases them.
            workspace = sum(block.numel() for block in blocks) * element_bytes + 2 * len(blocks) + 1
            results = []
            for block in blocks:
                checked("finite block", workspace)
                if ieee is not None and block.numel():
                    integer, exponent = ieee
                    # All exponent bits are one exactly for infinity or NaN.
                    # The mask removes sign and payload bits. Its nonnegative
                    # maximum equals the mask exactly when any value is bad.
                    results.append(torch.bitwise_and(block.view(integer), exponent).amax() != exponent)
                else:
                    results.append(torch.isfinite(block).all())
                checked("finite block complete", workspace)
            finite = (results[0] if len(results) == 1 else torch.stack(results).all()).item()
            del results
            checked("finite block readback", 0)
            if not finite:
                raise ValueError(f"{name} must be finite")


def _finite_max(values, name):
    """Reject each nonfinite residual before a maximum can hide a later NaN."""
    largest = 0.
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        largest = max(largest, value)
    return largest


def _trace_features(layer, x, guard=None, check=_no_check, *, refresh=True):
    """Encode features in live input precision with bounded weight copies.

    Storage precision defines the weight values. Live input precision defines
    encoder arithmetic and activation identity in hooks, traces, and validation.
    """
    if _encoder_matches_dtype(layer, x.dtype):
        return layer.encode(x)
    memory = BudgetMemory(guard, guard.limit / 1e9) if guard is not None else None
    if memory is not None:
        memory.check("dictionary encoder admission", force=refresh)
    latents, width = layer.W_enc.shape
    rows = max(1, _ENCODER_BLOCK_ELEMENTS // width)
    if memory is not None:
        positions, itemsize = math.prod(x.shape[:-1]), x.element_size()
        output_bytes = positions * latents * itemsize
        # Include converted weights, linear output, activation copies, and TopK indices.
        scratch = max(min(rows, latents) * (width + 1 + positions) * itemsize,
                      positions * latents * (4 * itemsize + 8) + latents * itemsize)
        initial = allocation_bytes(memory)
        if initial + output_bytes + scratch > memory.limit:
            raise CleanStop("dictionary encoder allocation exceeds the process memory limit")
    preactivation = x.new_empty(*x.shape[:-1], latents)
    for start in range(0, latents, rows):
        with autorelease_scope(x.device):
            if memory is not None:
                memory.check("dictionary encoder block", force=False)
                if allocation_bytes(memory, floor=initial + output_bytes) + scratch > memory.limit:
                    raise CleanStop("dictionary encoder allocation exceeds the process memory limit")
            check("dictionary encoder block")
            stop = min(start + rows, latents)
            preactivation[..., start:stop] = F.linear(
                x, layer.W_enc.detach()[start:stop].to(x.dtype),
                layer.b_enc.detach()[start:stop].to(x.dtype))
            if memory is not None:
                memory.check("dictionary encoder block complete", force=False)
                if allocation_bytes(memory, floor=initial + output_bytes) + scratch > memory.limit:
                    raise CleanStop("dictionary encoder allocation exceeds the process memory limit")
            check("dictionary encoder block complete")
    if hasattr(layer, "threshold") and layer.threshold.dtype != preactivation.dtype:
        def activation_check(stage):
            check(stage)
            if memory is not None:
                memory.check(stage, force=False)
                if allocation_bytes(memory, floor=initial + output_bytes) + scratch > memory.limit:
                    raise CleanStop("dictionary encoder allocation exceeds the process memory limit")
        return layer.activate(preactivation, check=activation_check)
    if callable(getattr(layer, "activate", None)):
        return layer.activate(preactivation)
    # Compatibility with existing TopK instances before activation metadata.
    values, indices = preactivation.relu_().topk(layer.k, dim=-1)
    return torch.zeros_like(preactivation).scatter_(-1, indices, values)


def _trace_reconstruction(layer, features, *, check=_no_check):
    """Decode in model precision with bounded copies of decoder rows."""
    check("dictionary decoder admission")
    if layer.W_dec.dtype == features.dtype and layer.b_dec.dtype == features.dtype:
        return layer.decode(features)
    flat = features.reshape(-1, features.shape[-1])
    indices = _active_columns(flat)
    result = _decode_selected(layer, indices, flat.gather(-1, indices), check=check)
    check("dictionary decoder bias")
    return result.reshape(*features.shape[:-1], layer.W_dec.shape[1]) + layer.b_dec.detach().to(features.dtype)


def _active_columns(values, edited=()):
    """Select all active columns, with distinct inactive columns as padding."""
    included = values != 0
    if edited:
        included[:, edited] = True
    count = int(included.sum(-1).max())
    return included.to(torch.int32).topk(count, dim=-1).indices.sort(dim=-1).values


def _decode_selected(layer, indices, activations, *, check=_no_check):
    """Decode compact features with a bounded copy of selected decoder rows."""
    tokens, columns = indices.shape
    width = layer.W_dec.shape[1]
    result = activations.new_zeros(*activations.shape[:-1], width)
    step = max(1, _ENCODER_BLOCK_ELEMENTS // max(1, tokens * width))
    for start in range(0, columns, step):
        with autorelease_scope(activations.device):
            check("selected decoder block")
            rows = layer.W_dec.detach()[indices[:, start:start + step]].to(activations.dtype)
            result += torch.einsum("...tk,tkw->...tw", activations[..., start:start + step], rows)
            del rows
    return result


def _decode_reference(layer, indices, activations, *, check=_no_check, include_zeros=False, guard=None):
    """Use one joint sparse accumulation, or refuse its complete workspace."""
    tokens, columns = indices.shape
    width = layer.W_dec.shape[1]
    batches = math.prod(activations.shape[:-2])
    itemsize = activations.element_size()
    with autorelease_scope(activations.device):
        check("reference decoder selection")
        if guard is not None:
            # Masks, coordinate pairs, and contiguous coordinate copies. No
            # decoder row or result allocation precedes workspace admission.
            selection_bytes = tokens * columns * (batches + 1 + 4 * 8)
            guard.reserve("reference decoder selection", selection_bytes)
        if include_zeros:
            selected = torch.ones_like(indices, dtype=torch.bool)
        else:
            active = activations != 0
            selected = active.any(tuple(range(active.ndim - 2))) if active.ndim > 2 else active
        positions, slots = selected.nonzero(as_tuple=True)
        positions, slots = positions.contiguous(), slots.contiguous()
        count = positions.numel()
        check("reference decoder workspace")
        if guard is not None:
            # Include stored and converted rows, values, scaled rows, and the
            # result. Reserve decoder adjoint scratch when gradients are live.
            row_bytes = count * width * (layer.W_dec.element_size() + itemsize)
            value_bytes = batches * count * itemsize
            scaled_bytes = value_bytes * width
            output_bytes = batches * tokens * width * itemsize
            workspace = count * 8 + row_bytes + value_bytes + scaled_bytes + output_bytes
            if torch.is_grad_enabled() and activations.requires_grad:
                workspace += scaled_bytes + value_bytes + output_bytes
            guard.reserve("reference decoder joint workspace", workspace)
        features = indices[positions, slots]
        values = activations[..., positions, slots]
        check("reference decoder rows")
        rows = layer.W_dec.detach()[features].to(activations.dtype)
        check("reference decoder scaling")
        scaled = rows * values[..., None]
        check("reference decoder accumulation")
        result = activations.new_zeros(*activations.shape[:-1], width)
        # The pinned decode_sparse uses this single call across all positions.
        # Position or feature blocks change BF16 sums on some backends. Keep
        # BOS rows here because full-source diagnostics require those sources.
        result.index_add_(-2, positions, scaled)
        check("reference decoder complete")
        return result


def _validate_dictionary(model, dictionary, check=_no_check, *, memory=None):
    """Check identity, dimensions, and probes before any model forward."""
    check("dictionary admission")
    def identity_check(stage):
        # The caller reserves hash scratch and checks its deadline at each chunk.
        check("model_identity")
    data = dictionary.metadata
    limit = None if memory is None else family_memory_limit(model, memory=memory)
    if not isinstance(data, dict) or data.get("model_hash") != model_identity(
            model, memory=memory, memory_limit_gb=limit, check=identity_check):
        raise ValueError("dictionary model identity mismatch")
    check("dictionary identity")
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
    latents, k, activation = data.get("latents"), data.get("k"), data.get("activation")
    if type(latents) is not int or latents < 1:
        raise ValueError("dictionary requires a positive latent count")
    if activation == "topk":
        if type(k) is not int or not 1 <= k <= latents:
            raise ValueError("dictionary requires valid TopK metadata")
    elif activation == "jumprelu":
        if k is not None:
            raise ValueError("JumpReLU dictionary must not specify TopK")
    else:
        raise ValueError("dictionary requires valid activation metadata")
    shapes = {"W_enc": (latents, config.hidden_size), "W_dec": (latents, config.hidden_size),
              "b_enc": (latents,), "b_dec": (config.hidden_size,)}
    if activation == "jumprelu":
        shapes["threshold"] = (latents,)
    reference = _reference_coordinates(data)
    if reference is not None:
        shapes["W_enc_original"] = shapes["W_enc"]
        if str(next(model.parameters()).dtype) != "torch." + reference["dtype"]:
            raise ValueError("original encoder artifact dtype differs from model coordinates")
    device = next(model.parameters()).device
    for i, layer in enumerate(dictionary.layers):
        if getattr(layer, "activation", "topk") != activation or layer.k != k:
            raise ValueError("dictionary activation or TopK differs from its metadata")
        if activation == "jumprelu" and not callable(getattr(layer, "activate", None)):
            raise ValueError("JumpReLU dictionary requires activate(preactivation)")
        for name, shape in shapes.items():
            value = getattr(layer, name, None)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"dictionary {name} shape differs from its metadata")
            if not value.is_floating_point():
                raise ValueError("dictionary tensors must use floating point storage")
            if reference is not None and str(value.dtype) != "torch." + reference["dtype"]:
                raise ValueError("original encoder tensor dtype differs from its coordinate contract")
            if value.device != device:
                raise ValueError("dictionary and model must share one device")
            if name == "threshold" and (value <= 0).any().item():
                raise ValueError("dictionary JumpReLU threshold must be positive")
            _require_finite(value, f"dictionary layer {i} {name}", check, memory)
    return probes


class _DictionaryAdmission:
    """Single-use validation for one synchronous trace request."""

    def __init__(self, model, dictionary, check, *, memory):
        self._probes = _validate_dictionary(model, dictionary, check, memory=memory)
        self._model, self._dictionary = model, dictionary
        self._active = False

    def __enter__(self):
        if self._active or self._probes is None:
            raise ValueError("trace dictionary admission cannot be reused")
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        self._probes = None

    def consume(self, model, dictionary):
        if (not self._active or self._probes is None
                or model is not self._model or dictionary is not self._dictionary):
            raise ValueError("trace dictionary admission is expired or belongs to different resources")
        probes, self._probes = self._probes, None
        return probes


class MemoryGuard:
    """Keep recent stage records and the process peak, and enforce the demo limit."""

    _limit = 12_000_000_000
    record_limit = 128

    def __init__(self, device, *, memory_limit_gb=12, host_reserve_bytes=0):
        if (isinstance(memory_limit_gb, bool) or not isinstance(memory_limit_gb, (int, float))
                or not math.isfinite(memory_limit_gb) or not 0 < memory_limit_gb <= 40):
            raise ValueError("trace memory limit must be positive and at most 40 GB")
        self.limit = int(memory_limit_gb * 1_000_000_000)
        self.device = torch.device(device)
        self._host_reserve_bytes = host_reserve_bytes
        self.records = deque(maxlen=self.record_limit)
        self._pressure_time = -float("inf")
        self._pressure_value = 100.
        self._guard = ProcessMemoryGuard(memory_limit_gb, self.device, pressure=self._pressure)
        self.check("trace start")

    @property
    def host_reserve_bytes(self):
        return max(self._host_reserve_bytes, getattr(getattr(self, "_guard", None), "host_reserve_bytes", 0))

    @property
    def limit(self):
        return min(self._limit, getattr(getattr(self, "_guard", None), "limit", self._limit))

    @limit.setter
    def limit(self, value):
        self._limit = value

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
            raise CleanStop("trace exceeded the process or MPS memory limit")
        return current

    def _mps_headroom(self):
        """Use the owned allocator cap and the latest forced device readings."""
        ceiling = getattr(self._guard, "mps_allocator_limit", None)
        if ceiling is None:
            raise CleanStop("MPS allocator ceiling is unavailable")
        # A lower supplied process budget also lowers prospective device use.
        ceiling = min(ceiling, int(0.8 * self.limit))
        record = self.records[-1]
        # Driver bytes include cache and framework use, and overlap tensor bytes.
        return ceiling - max(record["allocated_bytes"], record["driver_bytes"])

    def reserve(self, stage, size):
        """Refuse a known allocation before it can exceed either memory limit."""
        current = self.check(stage)
        if current + size > self.limit:
            raise CleanStop(f"{stage} allocation exceeds the trace memory limit")
        if self.device.type == "mps" and size > self._mps_headroom():
            raise CleanStop(f"{stage} allocation exceeds the MPS allocator limit")
        from ._admission import _host_reserve
        _host_reserve(size, stage, memory=self)

    def _check_interval(self, stage):
        """Check bounded encoder work without another forced process read."""
        self._guard.check(stage, force=False)
        current = allocation_bytes(self._guard)
        if max(current, self._guard.peak) > self.limit:
            raise CleanStop("trace exceeded the process or MPS memory limit")
        return current

    def batch_size(self, requested, *, layers, tokens, width, mixer_width, itemsize=4):
        current = self.check("target batch admission")
        if self.device.type == "mps":
            # Release only unoccupied cache at the synchronous batch boundary.
            # Keep prior failures and use fresh counters for every admission.
            torch.mps.empty_cache()
            current = self.check("target batch admission after cache release")
        # Bound live values, adjoints, convolution work, and allocator headroom.
        per_target = max(4, itemsize) * tokens * (width * (layers + 1) * 12 + mixer_width * layers * 8)
        available = self.limit - current - 512_000_000
        import psutil
        if self.host_reserve_bytes:
            available = min(available, psutil.virtual_memory().available - self.host_reserve_bytes - 512_000_000)
        if self.device.type == "mps":
            available = min(available, self._mps_headroom() - 512_000_000)
        admitted = min(requested, int(available // max(1, per_target)))
        if admitted < 1:
            limit = "process or MPS allocator" if self.device.type == "mps" else "process"
            raise CleanStop(f"no target batch fits the {limit} memory limit")
        return admitted


class _CappedMemoryGuard(MemoryGuard):
    """Keep a private path cap and all supplied guard checks and history."""

    def __init__(self, device, *, memory_limit_gb, memory, host_reserve_bytes=0):
        memory.check("frozen supplied guard")
        super().__init__(device, memory_limit_gb=memory_limit_gb, host_reserve_bytes=host_reserve_bytes)
        self._guard = BudgetMemory(self._guard, memory_limit_gb, extra=memory)
        self.check("frozen supplied budget")

    def check(self, stage):
        return max(super().check(stage), allocation_bytes(self._guard))


def _tensor_input(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))


def _bias(module):
    value = getattr(module, "bias", None)
    return value.detach() if value is not None else None


class FrozenNorm:
    """A recorded RMS denominator, fixed gain, and explicit affine bias."""

    def __init__(self, module, x, *, unit_offset=False, reference=False):
        x = x.detach()
        work = x.float() if unit_offset else x
        scale = torch.rsqrt(work.square().mean(-1, keepdim=True) + module.eps)
        if not unit_offset and getattr(module, "frozen_scale", None) is not None:
            scale = module.frozen_scale.detach()
        self.scale = scale.detach()
        self.denominator = (work.square().mean(-1, keepdim=True) + module.eps).sqrt().detach()
        self.weight = (1 + module.weight.detach().float()) if unit_offset else module.weight.detach()
        self.promote_output = False
        if reference:
            from ._gemma_reference import ReferenceMLPOutputNorm
            self.weight = self.weight.to(module.weight.dtype)
            self.promote_output = type(module) is ReferenceMLPOutputNorm
        self.bias = _bias(module)
        self.unit_offset = unit_offset
        self.reference = reference

    def __call__(self, x, bias=1.):
        work = x.float() if self.unit_offset else x
        result = (self.pure(x) * self.weight if self.reference else
                  ((work * self.scale) * self.weight).to(x.dtype))
        if self.bias is not None:
            result = result + bias * self.bias
        return result.float() if self.promote_output else result

    def pure(self, x):
        """Return the original Scope input before gain and encoder conversion."""
        dtype = self.weight.dtype if self.reference else x.dtype
        return (x.float() / self.denominator).to(dtype)


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
    family = model_family(model)
    supported, attention = set(), {}
    for point in probes:
        if not point.endswith(".mlp_in"):
            continue
        base = point.rsplit(".", 1)[0]
        name = f"{base}.self_attn"
        projection = "o_proj"
        supported.update((f"{base}.mlp_out", f"{base}.mlp_in",
                          f"{base}.resid_pre", name, f"{name}.{projection}"))
        if family.mlp_write_module == "mlp":
            supported.add(f"{base}.mlp")
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
    """Linear values with fixed softmax."""

    def __init__(self, module, x, pattern, *, output=None):
        self.pattern = pattern.detach()
        self.head_dim = module.head_dim
        self.heads = pattern.shape[1]
        self.v_weight = module.v_proj.weight.detach()
        self.v_bias = _bias(module.v_proj)
        self.output = output if output is not None else FrozenAttentionOutput(module.o_proj)
        with torch.no_grad():
            # The recorded residual can be float32 while the preceding norm
            # supplies model-dtype values to the attention projection.
            zero = torch.zeros_like(x, dtype=self.v_weight.dtype)
            self.offset = self._apply(zero, True).detach()
            self.intervention_offset = self._apply(zero, False, True).detach()

    def _apply(self, x, biases, intervention=False):
        batch, length, _ = x.shape
        values = F.linear(x, self.v_weight, self.v_bias if biases else None)
        values = values.reshape(batch, length, -1, self.head_dim).transpose(1, 2)
        values = values.repeat_interleave(self.heads // values.shape[1], dim=1)
        result = (self.pattern @ values).transpose(1, 2).reshape(batch, length, -1)
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
    """Compact affine feature writes within the recorded activation region."""

    def __init__(self, original, layer, specifications):
        from ._intervene import _effective_feature_operations

        # Select every nonzero activation and every explicit edit. Each row has
        # unique indices, including inactive padding. JumpReLU has no fixed k.
        edited = sorted({s["feature"] for s in specifications if s["op"] in ("scale", "clamp")})
        self.indices = _active_columns(original[0], edited)
        self.original = original[0].gather(-1, self.indices)
        self.region = (self.original != 0).detach()
        self.coefficients = self.region.to(original.dtype)
        self.constants = torch.zeros_like(self.original)
        self.scale = original.new_ones((1, original.shape[1], 1))
        for spec in _effective_feature_operations(specifications):
            positions = list(spec.get("positions", range(original.shape[1])))
            kind = spec["op"]
            if kind in ("ablate", "patch"):
                self.coefficients[positions] = 0
                self.constants[positions] = 0
                self.scale[:, positions] = 0
            elif kind in ("scale", "clamp"):
                selected = self.indices == spec["feature"]
                rows = torch.zeros_like(selected)
                rows[positions] = True
                selected &= rows
                amount = spec["multiplier"] - 1 if kind == "scale" else -1
                self.coefficients += amount * (selected & self.region).to(original.dtype)
                if kind == "clamp":
                    self.constants += spec["value"] * selected.to(original.dtype)
        self.activations = self.coefficients * self.original + self.constants

    def evaluate(self, read, layer, *, original=False):
        if original:
            # The pinned encoder uses one F.linear, including its bias. A
            # selected einsum followed by bias addition changes BF16 rounding.
            # The reference workspace admits this token-by-feature buffer.
            preactivation = F.linear(read, layer.W_enc_original.detach(), layer.b_enc.detach())
            values = preactivation.gather(-1, self.indices.expand(*read.shape[:-1], -1))
            return values * self.coefficients + self.constants
        values = read.new_empty(*read.shape[:-1], self.indices.shape[-1])
        step = max(1, _ENCODER_BLOCK_ELEMENTS // max(1, self.indices.shape[0] * read.shape[-1]))
        for start in range(0, self.indices.shape[-1], step):
            with autorelease_scope(read.device):
                indices = self.indices[:, start:start + step]
                encoder = layer.W_enc.detach()[indices].to(read.dtype)
                values[..., start:start + step] = (torch.einsum("btw,tkw->btk", read, encoder) +
                                                    layer.b_enc.detach()[indices].to(read.dtype))
        return values * self.coefficients + self.constants


def _validate_intervention_positions(tokens, interventions):
    """Refuse invalid target positions before a forward or cache change."""
    for spec in interventions:
        for position in spec.get("positions", ()):
            if type(position) is not int or not 0 <= position < tokens:
                raise ValueError(f"intervention position {position!r} is outside the prompt of {tokens} tokens")


def _validate_interventions(model, blocks, prefix, interventions):
    """Refuse module edits outside the explicit affine paths."""
    allowed = {}
    family = model_family(model)
    for i, block in enumerate(blocks):
        base = f"{prefix}.{i}"
        for suffix in ("resid_pre", "mlp_in", "mlp", "mlp_out"):
            allowed[f"{base}.{suffix}"] = {"ablate", "add", "patch"}
        allowed[f"{base}.mlp_out"] |= {"scale", "clamp"}
        if family.mlp_write_module != "mlp":
            # Raw GeGLU edits precede a nonlinear norm. They are not residual writes.
            del allowed[f"{base}.mlp"]
        name = f"{base}.self_attn"
        projection = "o_proj"
        allowed[name] = {"ablate", "add", "patch"}
        if family.name == "gemma2":
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

    def __init__(self, model, input_ids, dictionary, *, guard=None, interventions=(), check=_no_check,
                 memory_limit_gb=None, _admission=None, profile="caliper"):
        _validate_ablation_points(model, interventions)
        _validate_intervention_positions(input_ids.shape[1], interventions)
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.family = model_family(model)
        self.reference = profile == "circuit_tracer_82996f14"
        self.raw_reference_input = self.reference and self.family.name == "gemma2"
        self.reference_arithmetic = False
        if self.family.name == "gemma2":
            from ._gemma_reference import profile_metadata
            self.reference_arithmetic = profile_metadata(model) is not None
            if self.raw_reference_input and self.dtype == torch.bfloat16 and not self.reference_arithmetic:
                raise ValueError("bfloat16 reference tracing requires a qualified Gemma arithmetic profile")
        if self.raw_reference_input and _reference_coordinates(dictionary.metadata) is None:
            raise ValueError("Gemma reference tracing requires the original pre-gain encoder artifact")
        memory_limit_gb = family_memory_limit(model, memory_limit_gb, guard)
        host_reserve = host_reserve_for_model(model)
        self.guard = guard if guard is not None else MemoryGuard(
            self.device, memory_limit_gb=memory_limit_gb, host_reserve_bytes=host_reserve)
        if guard is not None and (int(memory_limit_gb * 1e9) < guard.limit or
                                  host_reserve > getattr(guard, "host_reserve_bytes", 0)):
            self.guard = _CappedMemoryGuard(self.device, memory_limit_gb=memory_limit_gb, memory=guard,
                                          host_reserve_bytes=host_reserve)

        def memory_check(stage):
            check(stage)
            getattr(self.guard, "_check_interval", self.guard.check)(stage)

        if _admission is None:
            probes = _validate_dictionary(model, dictionary, memory_check,
                                          memory=BudgetMemory(self.guard, memory_limit_gb))
        else:
            if not isinstance(_admission, _DictionaryAdmission):
                raise TypeError("trace requires a private dictionary admission")
            probes = _admission.consume(model, dictionary)
        if model.training:
            raise ValueError("trace requires an evaluation model")
        if self.dtype not in (torch.float32, torch.float64) and not (
                self.reference and self.family.name == "gemma2" and self.dtype == torch.bfloat16):
            raise ValueError("trace requires float32 or float64 model weights")
        if self.family.unit_offset_norm and self.dtype != torch.float32 and not (
                self.reference and self.family.name == "gemma2" and self.dtype == torch.bfloat16):
            raise ValueError(f"{self.family.name} tracing requires float32 model weights")
        modules = dict(model.named_modules())
        root, prefix, self.blocks = self.family.locate(model)
        interventions = [dict(s, op=s.get("op", s.get("kind"))) for s in interventions]
        _validate_interventions(model, self.blocks, prefix, interventions)
        self.layers = len(self.blocks)
        self.tokens = input_ids.shape[1]
        self.width = model.config.hidden_size
        self.mixer_width = self.width
        if self.reference:
            # Record tensors, feature buffers, decoder blocks, and vocabulary work.
            self.guard.reserve("reference forward workspace", self.tokens * self.layers * (
                self.width * 64 + dictionary.metadata["latents"] * 16) +
                model.config.vocab_size * 32 + _ENCODER_BLOCK_ELEMENTS * 16)
            if interventions and self.raw_reference_input:
                raise ValueError("original-coordinate reference traces require an unchanged baseline")
        recorded = {}
        self.norms, self.post_norms, self.mixers = [], [], []
        self.attention_post_norms, self.mlp_post_norms = [], []
        with ExitStack() as stack:
            stack.enter_context(autorelease_scope(self.device))
            values = stack.enter_context(capture(model, list(probes)))
            def pre(name):
                def hook(mod, args, kwargs):
                    recorded[name] = _tensor_input(args, kwargs).detach()
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
                frames.append((dict(values), dict(recorded)) if frames else None)
            def exit_forward(mod, args, output):
                frame = frames.pop()
                if frame is not None:
                    saved_values, saved_recorded = frame
                    values.clear()
                    values.update(saved_values)
                    recorded.clear()
                    recorded.update(saved_recorded)
            install(model.register_forward_pre_hook(enter_forward, prepend=True))
            install(model.register_forward_hook(exit_forward, always_call=True))
            install(root.norm.register_forward_pre_hook(pre("final_norm"), with_kwargs=True))
            install(root.embed_tokens.register_forward_hook(out("embedding")))
            for i, block in enumerate(self.blocks):
                install(block.register_forward_pre_hook(pre(f"{i}.residual_source"), with_kwargs=True, prepend=True))
                install(block.mlp.register_forward_pre_hook(pre(f"{i}.mlp_source"), with_kwargs=True, prepend=True))
                # Gemma's actual GeGLU forward and fourth norm produce the
                # residual write. Replacement writes must not traverse that norm again.
                write_module = getattr(block, self.family.mlp_write_module)
                install(write_module.register_forward_hook(out(f"{i}.mlp_raw"), prepend=True))
                if self.family.attention_write_norm is not None:
                    install(block.post_feedforward_layernorm.register_forward_pre_hook(pre(f"{i}.mlp_post_norm"), with_kwargs=True))
                    install(getattr(block, self.family.attention_write_norm).register_forward_pre_hook(
                        pre(f"{i}.attention_post_norm"), with_kwargs=True))
                attention_name = f"{prefix}.{i}.self_attn"
                projection_name = "o_proj"
                for name in (f"{attention_name}.{projection_name}", attention_name):
                    install(modules[name].register_forward_hook(out(name + ".before"), prepend=True))
                    install(modules[name].register_forward_hook(out(name + ".after")))
                install(block.input_layernorm.register_forward_pre_hook(pre(f"{i}.input_norm"), with_kwargs=True))
                install(getattr(block, self.family.mlp_read_norm).register_forward_pre_hook(pre(f"{i}.post_norm"), with_kwargs=True))
                def stage_hook(*_, i=i):
                    self.guard.check(f"record layer {i}")
                    check(f"record layer {i}")
                install(block.register_forward_hook(stage_hook))
            with torch.no_grad():
                check("record admission")
                self.guard.check("record admission")
                kwargs = {"use_cache": False, "logits_to_keep": 1} if self.family.name != "caliper_tiny" else {}
                with internal_forward():
                    self.logits = model(input_ids=input_ids, **kwargs).logits[:, -1].detach()
        self.guard.check("record complete")
        check("record complete")
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
                    check(f"freeze layer {i} admission")
                    self.guard.check(f"freeze layer {i} admission")
                    if self.raw_reference_input:
                        raw_input = recorded[f"{i}.post_norm"].float()
                        denominator = (raw_input.square().mean(-1, keepdim=True) +
                                       getattr(block, self.family.mlp_read_norm).eps).sqrt()
                        x = (raw_input / denominator).to(self.dtype)
                        a = layer.activate(F.linear(x, layer.W_enc_original, layer.b_enc))
                    else:
                        a = _trace_features(layer, x, self.guard, memory_check)
                    raw = recorded[f"{i}.mlp_raw"]
                    output_specs = [s for s in interventions
                                    if s["point"] in (f"{base}.mlp", f"{base}.mlp_out")
                                    and s.get("positions", (0,))]
                    output = FrozenFeatures(a, layer, output_specs)
                    self.feature_outputs.append(output)
                    transformed = torch.zeros_like(a).scatter_add(
                        -1, output.indices[None], output.activations[None])
                    decode = _decode_reference if self.reference else _decode_selected
                    if self.reference:
                        original = FrozenFeatures(a, layer, [])
                        reconstruction = decode(layer, original.indices, original.activations,
                                                check=memory_check, guard=self.guard)[None]
                        reconstruction = reconstruction + layer.b_dec.detach().to(self.dtype)
                    else:
                        reconstruction = _trace_reconstruction(layer, a)
                    error = output.scale * (raw - reconstruction)
                    decoder_bias = output.scale * layer.b_dec.detach().to(self.dtype)
                    # Explicit edits can add signed features outside the recorded region.
                    decoded = decode(layer, output.indices, output.activations, check=memory_check,
                                     **({"guard": self.guard} if self.reference else {}))[None]
                    # Only external vector writes need an output constant. Feature edits
                    # already have affine coefficients and constants in FrozenFeatures.
                    external = any(s["op"] in ("add", "patch") for s in output_specs)
                    offset = y - decoded - decoder_bias - error if external else torch.zeros_like(y)
                    self.inputs.append(x)
                    self.outputs.append(y)
                    self.features.append(transformed)
                    self.errors.append(error.detach())
                    _require_finite(self.features[-1], f"layer {i} features", memory_check, self.guard)
                    _require_finite(self.errors[-1], f"layer {i} reconstruction residual", memory_check, self.guard)
                    self.decoder_biases.append(decoder_bias)
                    self.output_offsets.append(offset.detach())
                    self.norms.append(FrozenNorm(block.input_layernorm, recorded[f"{i}.input_norm"], unit_offset=self.family.unit_offset_norm, reference=self.reference_arithmetic))
                    self.post_norms.append(FrozenNorm(getattr(block, self.family.mlp_read_norm), recorded[f"{i}.post_norm"], unit_offset=self.family.unit_offset_norm, reference=self.reference_arithmetic))
                    self.attention_post_norms.append(None if self.family.attention_write_norm is None else
                        FrozenNorm(getattr(block, self.family.attention_write_norm), recorded[f"{i}.attention_post_norm"], unit_offset=True, reference=self.reference_arithmetic))
                    self.mlp_post_norms.append(None if self.family.mlp_write_module == "mlp" else
                        FrozenNorm(block.post_feedforward_layernorm, recorded[f"{i}.mlp_post_norm"], unit_offset=True, reference=self.reference_arithmetic))
                    self.residual_inputs.append(FrozenInput(recorded[f"{i}.residual_source"],
                        recorded[f"{i}.input_norm"], [s for s in interventions if s["point"] == f"{base}.resid_pre"]))
                    self.mlp_inputs.append(FrozenInput(recorded[f"{i}.mlp_source"], x,
                        [s for s in interventions if s["point"] == f"{base}.mlp_in"]))
                    pattern = values[f"{base}.attn_pattern"]
                    mixer = FrozenAttention(block.self_attn, recorded[f"{i}.input_norm"], pattern,
                                            output=_attention_output(
                                                model, f"{base}.self_attn", "o_proj", interventions, recorded))
                    self.mixers.append(mixer)
                    self.guard.check(f"freeze layer {i}")
                    check(f"freeze layer {i}")
            with autorelease_scope(self.device):
                self.final_norm = FrozenNorm(root.norm, recorded["final_norm"], unit_offset=self.family.unit_offset_norm, reference=self.reference_arithmetic)
                head = model.get_output_embeddings()
                self.unembedding = head.weight.detach()
                self.unembedding_mean = self.unembedding.mean(0)
                self.unembedding_bias = _bias(head)
                # Pinned attribution retains complete MLP values with stopped
                # gradients. Subtraction of BF16 biases loses recorded bits.
                self.writes = (list(self.outputs) if self.reference else
                               [(y - b - c).detach() for y, b, c in
                                zip(self.outputs, self.decoder_biases, self.output_offsets)])
                reads, final = self.run(self.embedding, self.writes, check=check)
                self.reference_final = self.final_norm(recorded["final_norm"]).detach()
                self._freeze_logits(model, self.final_norm(recorded["final_norm"])[:, -1], memory_check)
                # Compare every read and all next-token logits, before any edge calculation.
                self.forward_residuals = {
                    "direct_reads": [float((a - b).abs().max()) for a, b in zip(reads, self.inputs)],
                    "direct_logits": float((self.logits_from_final(final[:, -1]) - self.logits).abs().max()),
                }
                self.forward_residual = _finite_max(
                    self.forward_residuals["direct_reads"] + [self.forward_residuals["direct_logits"]],
                    "direct forward residual")
                self.direct_forward_residual = self.forward_residual
                check("direct forward agreement")
                replacement_reads, replacement_final = self.replacement(self.embedding, dictionary, check=check)
                self.forward_residuals.update(
                    replacement_reads=[float((a - b).abs().max()) for a, b in zip(replacement_reads, self.inputs)],
                    replacement_logits=float((self.logits_from_final(replacement_final[:, -1]) - self.logits).abs().max()))
                self.replacement_forward_residual = _finite_max(
                    self.forward_residuals["replacement_reads"] + [self.forward_residuals["replacement_logits"]],
                    "replacement forward residual")
                self.forward_residual = max(self.direct_forward_residual, self.replacement_forward_residual)
                self.guard.check("forward agreement")
                check("forward agreement")
                # Encoder and unembedding biases are target-local. Input edits have their own constant term.
                zero = torch.zeros_like(self.embedding).expand(4, -1, -1)
                masks = torch.eye(4, device=self.device, dtype=self.dtype).reshape(4, 4, 1, 1)
                self.bias_reads, self.bias_final = self.run(zero, [zero] * self.layers, bias_masks=masks, check=check)
                check("bias preparation")

    def _freeze_logits(self, model, final, check):
        """Preserve softcap value and derivative with a prompt-local tangent."""
        self.logit_scale = self.logit_offset = None
        cap = getattr(model.config, "final_logit_softcapping", None) if self.family.name == "gemma2" else None
        self.reference_logit_cap = cap if self.reference else None
        if cap is None:
            return
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("final logit softcap must be finite and positive")
        raw = F.linear(final, self.unembedding, self.unembedding_bias)[0]
        tangent = (raw / cap).tanh()
        self.logit_scale = (1 - tangent.square()).detach()
        # f(z) = c*tanh(z/c). At z0, f(z0) + f'(z0)*(z-z0)
        # equals the live value and has its derivative. The offset is an
        # explicit target bias, including its vocabulary mean for centering.
        self.logit_offset = (cap * tangent - self.logit_scale * raw).detach()
        # Keep the existing unembedding storage. Two vocabulary vectors hold
        # the tangent and offset. One width vector holds the centered row mean.
        self.unembedding_mean = self.unembedding.new_zeros(self.width)
        rows = max(1, _ENCODER_BLOCK_ELEMENTS // self.width)
        for start in range(0, len(raw), rows):
            with autorelease_scope(self.device):
                check("softcap centered direction")
                self.unembedding_mean += self.logit_scale[start:start + rows] @ self.unembedding[start:start + rows]
        self.unembedding_mean /= len(raw)

    def logits_from_final(self, final):
        """Return forward logits or the ordinary prompt-local affine logits."""
        self.guard.check("frozen logits admission")
        logits = F.linear(final, self.unembedding, self.unembedding_bias)
        if self.reference_logit_cap is not None:
            # Published targets use pre-softcap directions in logit_target.
            # Forward checks use the actual softcap, without a BF16 tangent
            # offset whose subtraction and addition can change the live value.
            cap = self.reference_logit_cap
            return cap * (logits / cap).tanh()
        if self.logit_scale is not None:
            logits = logits * self.logit_scale + self.logit_offset
        return logits

    def logit_target(self, token, *, reference=False):
        """Return the centered logit derivative and explicit target bias."""
        self.guard.check("logit target admission")
        direction = self.unembedding[token]
        bias = self.unembedding_bias
        if reference:
            constant = self.embedding.new_zeros(()) if bias is None else bias[token] - bias.mean()
            return direction - self.unembedding.mean(0), constant
        if self.logit_scale is not None:
            direction = direction * self.logit_scale[token]
            bias = self.logit_offset if bias is None else self.logit_scale * bias + self.logit_offset
        constant = self.embedding.new_zeros(()) if bias is None else bias[token] - bias.mean()
        return direction - self.unembedding_mean, constant

    def run(self, embedding, writes, *, bias_masks=None, check=_no_check):
        """Return MLP reads and final normalized residual for independent writes.

        Bias mask rows are decoder, normalization, attention, and intervention contributions.
        A zero mask gives the strictly linear residual path.
        Reference writes include the complete output unless bias masks are supplied.
        """
        self.guard.check("frozen path admission")
        masks = (1., 1., 1., 1.) if bias_masks is None else bias_masks
        decoder, norm, attention = masks[:3]
        intervention = masks[3] if len(masks) == 4 else 0.
        complete_writes = self.reference and bias_masks is None
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer, write) in enumerate(zip(self.norms, self.post_norms, self.mixers, writes)):
            with autorelease_scope(self.device):
                x = self.residual_inputs[i](x, intervention)
                mixed = mixer(input_norm(x, norm), attention, intervention)
                if self.attention_post_norms[i] is not None:
                    mixed = self.attention_post_norms[i](mixed, norm)
                x = x + mixed
                reads.append(post_norm.pure(x) if self.raw_reference_input else
                             self.mlp_inputs[i](post_norm(x, norm), intervention))
                x = (x + write if complete_writes else
                     x + write + decoder * self.decoder_biases[i] + intervention * self.output_offsets[i])
                self.guard.check(f"frozen layer {i}")
                check(f"frozen layer {i}")
        return reads, self.final_norm(x, norm)


    def replacement(self, embedding, dictionary, *, check=_no_check):
        """Evaluate transcoders with the recorded active masks and constant errors.

        This affine forward verifies the complete local replacement. Direct
        edges use ``run`` with independent writes and never differentiate here.
        The reference uses decoded activation deltas around complete recorded
        writes. Its rounded reconstruction errors remain separate graph sources.
        """
        self.guard.check("replacement path admission")
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer) in enumerate(zip(self.norms, self.post_norms, self.mixers)):
            with autorelease_scope(self.device):
                x = self.residual_inputs[i](x)
                mixed = mixer(input_norm(x))
                if self.attention_post_norms[i] is not None:
                    mixed = self.attention_post_norms[i](mixed)
                x = x + mixed
                read = post_norm.pure(x) if self.raw_reference_input else self.mlp_inputs[i](post_norm(x))
                reads.append(read)
                layer = dictionary.layers[i]
                output = self.feature_outputs[i]
                activation = output.evaluate(read, layer, original=self.raw_reference_input)
                decode = _decode_reference if self.reference else _decode_selected
                if self.reference:
                    # Pinned intervention hooks subtract activations before
                    # decoder scaling, then add the delta to the MLP output.
                    # A rounded reconstruction plus its rounded error cannot
                    # recover that output exactly in BF16.
                    delta = decode(layer, output.indices, activation - output.activations,
                                   check=check, include_zeros=True, guard=self.guard)
                    x = x + (self.outputs[i] + delta)
                else:
                    x = (x + decode(layer, output.indices, activation, check=check) +
                         self.decoder_biases[i] + self.errors[i] + self.output_offsets[i])
                self.guard.check(f"replacement layer {i}")
                check(f"replacement layer {i}")
        return reads, self.final_norm(x)
