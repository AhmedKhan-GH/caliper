"""Stable model evidence shared by quality evaluation and live qualification."""
from __future__ import annotations

import hashlib
import json

import torch

from ._dictionary import _hf_digest, _state_hash
from ._forward_identity import forward_settings
from ._trace import _forward_settings


def _evaluation_settings(model, memory):
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
    settings = _evaluation_settings(model, memory) if settings is None else settings
    return {"version": 2, "model_hash": model_hash,
            "tensor_hash": _hf_digest(model, memory=memory),
            "forward_settings_hash": _settings_digest(model, settings)}
