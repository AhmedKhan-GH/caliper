"""Bounded evidence for the artifact and tensors in one evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from caliper.interp._dictionary import (
    _canonical_device, _checkpoint_slices, _dense_state, _hf_digest,
    dictionary_fingerprint, model_identity, weight_names,
)
from caliper.interp._trace import _forward_settings
from .memory_limits import CleanStop, autorelease_scope


_CHUNK_BYTES = 1024 * 1024
_CHANGED = "evaluation dictionary or settings changed: artifact binding differs"
_MODEL_CHANGED = "evaluation model identity mismatch: runtime binding differs"


def _metadata_bytes(metadata):
    return json.dumps({k: v for k, v in metadata.items() if k != "quality"},
                      sort_keys=True, allow_nan=False).encode()


def _evaluation_settings(model, memory):
    settings = _forward_settings(model, memory)
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


class EvaluationBinding:
    """Bind converted tensors to disk without a second dictionary allocation."""

    def __init__(self, folder, dictionary, identity, *, model, memory=None):
        self.folder, self.dictionary = Path(folder), dictionary
        self.identity, self.memory = identity, memory
        self.model = model
        self.model_hash = dictionary.metadata["model_hash"]
        self._check_model_identity()
        # Keep original modules and tensors alive through temporary MLP replacements.
        self.model_modules = tuple(model.modules())
        self.model_tensors = tuple(self._model_tensors().values())
        self.model_stamps = self._model_stamps()
        self.model_settings = _evaluation_settings(model, memory)
        self.model_evidence = {"version": 2, "model_hash": self.model_hash,
                               "tensor_hash": _hf_digest(model, memory=memory),
                               "forward_settings_hash": _settings_digest(model, self.model_settings)}
        self.metadata = _metadata_bytes(dictionary.metadata)
        self.layers = tuple(dictionary.layers)
        # Retain references so a replacement cannot reuse an old tensor identity.
        self.tensors = tuple(getattr(layer, name) for layer in self.layers for name in weight_names(layer))
        self.stamps = self._tensor_stamps()
        self.files = self._file_stamps()
        self.check()
        self.evidence = {"version": 1, "tensor_hash": self._digest(compare=True)}
        self._check_files()
        self.check()

    def _check_model_identity(self):
        limit = self.memory.limit / 1e9 if self.memory else None
        if model_identity(self.model, memory_limit_gb=limit) != self.model_hash:
            raise ValueError(_MODEL_CHANGED)

    def _model_tensors(self):
        return {**dict(self.model.named_parameters(remove_duplicate=False)),
                **dict(self.model.named_buffers(remove_duplicate=False))}

    def _model_stamps(self):
        tensors = _dense_state(self._model_tensors())
        return (tuple((name, id(module)) for name, module in self.model.named_modules()), tuple(
            (name, id(t), t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape), t.stride())
            for name, t in tensors.items()))

    def _tensor_stamps(self):
        layers = self.dictionary.layers
        data = self.dictionary.metadata
        if len(layers) != data["layers"] or any(
            layer.k != data["k"] or layer.activation != data["activation"] for layer in layers
        ):
            raise ValueError(_CHANGED)
        tensors = {f"{i}.{name}": getattr(layer, name)
                   for i, layer in enumerate(layers) for name in weight_names(layer)}
        _dense_state(tensors)
        return (tuple(id(layer) for layer in layers), tuple(
            (id(t), t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape), t.stride())
            for t in tensors.values()))

    def _file_stamps(self):
        result = []
        for i in range(len(self.layers)):
            stat = (self.folder / f"layer_{i:02d}.safetensors").stat()
            result.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return result

    def check(self):
        """Refuse changes to forward settings, metadata, files, or tensors."""
        if self._model_stamps() != self.model_stamps:
            raise ValueError(_MODEL_CHANGED)
        settings = _evaluation_settings(self.model, self.memory)
        if settings != self.model_settings:
            changed = next(name for name in sorted(settings.keys() | self.model_settings.keys())
                           if settings.get(name) != self.model_settings.get(name))
            raise ValueError(f"{_MODEL_CHANGED}; forward setting changed: {changed}")
        disk = json.loads((self.folder / "dictionary.json").read_text(encoding="utf-8"))
        if (_metadata_bytes(self.dictionary.metadata) != self.metadata
                or _metadata_bytes(disk) != self.metadata
                or self._tensor_stamps() != self.stamps
                or self._file_stamps() != self.files):
            raise ValueError(_CHANGED)

    def _admit(self, tensor):
        if tensor.device.type != "cpu":
            if self.memory is None:
                raise ValueError("evaluation device tensors require a process memory guard")
            if _canonical_device(tensor.device) != _canonical_device(self.memory.device):
                raise ValueError("evaluation tensor and memory guard devices differ")
        if self.memory:
            # Admit every readback. File hashes force a fresh stage measurement.
            self.memory.check("evaluation_binding", force=False)
            if self.memory.peak + 4 * _CHUNK_BYTES > self.memory.limit:
                raise CleanStop("evaluation binding exceeds the process memory limit")

    def _digest(self, *, compare=False):
        digest = hashlib.sha256(b"caliper.evaluation.tensors.v1\0")
        for i, layer in enumerate(self.layers):
            with safe_open(self.folder / f"layer_{i:02d}.safetensors",
                           framework="pt", device="cpu") as source:
                if set(source.keys()) != set(weight_names(layer)):
                    raise ValueError(_CHANGED)
                for name in weight_names(layer):
                    tensor = getattr(layer, name)
                    stored = source.get_slice(name)
                    if list(tensor.shape) != stored.get_shape():
                        raise ValueError(_CHANGED)
                    header = json.dumps([i, name, str(tensor.dtype), list(tensor.shape)]).encode()
                    digest.update(len(header).to_bytes(8, "little"))
                    digest.update(header)
                    # Eight bytes cover the largest supported source or target dtype.
                    for selection in _checkpoint_slices(tensor.shape, _CHUNK_BYTES // 8):
                        self._admit(tensor)
                        with autorelease_scope(tensor.device):
                            chunk = tensor.detach()[selection].to(device="cpu").contiguous()
                            raw = chunk.view(torch.uint8)
                            if compare:
                                # Convert the file slice as the loader converts the parameter.
                                expected = stored[selection].to(dtype=tensor.dtype).contiguous()
                                if not torch.equal(raw, expected.view(torch.uint8)):
                                    raise ValueError(_CHANGED)
                                del expected
                            digest.update(memoryview(raw.numpy()))
                            del raw, chunk
        return digest.hexdigest()

    def _check_files(self):
        if dictionary_fingerprint(self.folder, self.dictionary.metadata, memory=self.memory) != self.identity:
            raise ValueError(_CHANGED)

    def verify(self):
        """Recheck bytes before publication, including aliases without version updates."""
        self.check()
        self._check_model_identity()
        if _hf_digest(self.model, memory=self.memory) != self.model_evidence["tensor_hash"]:
            raise ValueError(_MODEL_CHANGED)
        self._check_files()
        if self._digest() != self.evidence["tensor_hash"]:
            raise ValueError(_CHANGED)
        self.check()
