"""Exact evidence shared by quality evaluation and live qualification."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from ._dictionary import _canonical_device, _checkpoint_slices, _dense_state, _hf_digest, _state_hash, weight_names
from ._forward_identity import _ARITHMETIC_CONTEXT, arithmetic_settings, forward_settings
from ._memory import CleanStop, allocation_bytes, autorelease_scope
from ._trace import _forward_settings


_CHUNK_BYTES = 1024 * 1024
_DICTIONARY_CHANGED = "evaluation dictionary or settings changed: artifact binding differs"


def _admit_dictionary_tensor(tensor, memory):
    if tensor.device.type != "cpu":
        if memory is None:
            raise ValueError("evaluation device tensors require a process memory guard")
        if _canonical_device(tensor.device) != _canonical_device(memory.device):
            raise ValueError("evaluation tensor and memory guard devices differ")
    if memory:
        memory.check("evaluation_binding", force=False)
        if allocation_bytes(memory) + 4 * _CHUNK_BYTES > memory.limit:
            raise CleanStop("evaluation binding exceeds the process memory limit")


def evaluation_dictionary_evidence(dictionary, *, memory=None, folder=None):
    """Hash live tensors in cursor order, with an optional converted-file check.

    The version 1 digest preserves existing evaluation cursors. Each copy stays
    within one chunk, including file conversion and device readback.
    """
    digest = hashlib.sha256(b"caliper.evaluation.tensors.v1\0")
    for i, layer in enumerate(dictionary.layers):
        weights = weight_names(layer)
        tensors = _dense_state({name: getattr(layer, name) for name in weights})
        context = (nullcontext(None) if folder is None else
                   safe_open(Path(folder) / f"layer_{i:02d}.safetensors", framework="pt", device="cpu"))
        with context as source:
            if source is not None and set(source.keys()) != set(weights):
                raise ValueError(_DICTIONARY_CHANGED)
            for name, tensor in tensors.items():
                stored = source.get_slice(name) if source is not None else None
                if stored is not None and list(tensor.shape) != stored.get_shape():
                    raise ValueError(_DICTIONARY_CHANGED)
                header = json.dumps([i, name, str(tensor.dtype), list(tensor.shape)]).encode()
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
                # Eight bytes cover the largest supported source or target dtype.
                for selection in _checkpoint_slices(tensor.shape, _CHUNK_BYTES // 8):
                    _admit_dictionary_tensor(tensor, memory)
                    with autorelease_scope(tensor.device):
                        chunk = tensor.detach()[selection].to(device="cpu").contiguous()
                        raw = chunk.view(torch.uint8)
                        if stored is not None:
                            expected = stored[selection].to(dtype=tensor.dtype).contiguous()
                            if not torch.equal(raw, expected.view(torch.uint8)):
                                raise ValueError(_DICTIONARY_CHANGED)
                            del expected
                        digest.update(memoryview(raw.numpy()))
                        del raw, chunk
    return {"version": 1, "tensor_hash": digest.hexdigest()}


def _evaluation_settings(model, memory):
    # Shared trace settings include runtime arithmetic. The canonical default
    # adds no fields, so legacy evaluation and index digests stay exact.
    settings = _forward_settings(model, memory)
    if getattr(getattr(model, "config", None), "model_type", None) == "gemma2":
        # Include every Gemma checkpoint setting, such as activation approximation.
        # Keep the existing Qwen cursor digest and trace object checks unchanged.
        checkpoint = forward_settings(model, tensor_hash=lambda state: _state_hash(state, memory=memory))
        checkpoint.update(settings)
        settings = checkpoint
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        if hasattr(config, "_attn_implementation"):
            # Evaluation uses the configured backend for both full forwards.
            key = (name + "." if name else "") + "config._attn_implementation"
            settings[key] = json.dumps(config._attn_implementation, sort_keys=True, allow_nan=False)
    return settings


def _settings_digest(model, settings):
    persistent = dict(settings)
    for name, module in model.named_modules():
        if hasattr(module, "act_fn"):
            # Trace snapshots use object identity. Resume needs a stable type.
            # Module fields and tensors remain bound by their existing evidence.
            activation = module.act_fn
            if not isinstance(activation, torch.nn.Module):
                raise ValueError("evaluation requires module activation functions")
            key = (name + "." if name else "") + "act_fn"
            persistent[key] = json.dumps(type(activation).__module__ + "." + type(activation).__qualname__)
    raw = json.dumps(persistent, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(b"caliper.evaluation.forward.v1\0" + raw).hexdigest()


def evaluation_model_evidence(model, model_hash, *, memory=None, settings=None):
    """Describe exact tensors and settings after the caller verifies model_hash."""
    if settings is None:
        settings = _evaluation_settings(model, memory)
    elif settings.get(_ARITHMETIC_CONTEXT) != arithmetic_settings(model).get(_ARITHMETIC_CONTEXT):
        # A supplied snapshot must describe the current arithmetic before any
        # tensor readback. Never relabel stale evidence with the current context.
        raise ValueError("evaluation arithmetic context changed after settings capture")
    return {"version": 2, "model_hash": model_hash,
            "tensor_hash": _hf_digest(model, memory=memory),
            "forward_settings_hash": _settings_digest(model, settings)}
