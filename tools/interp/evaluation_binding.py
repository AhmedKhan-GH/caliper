"""Bounded evidence for the artifact and tensors in one evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from caliper.interp._dictionary import (
    _canonical_device, _checkpoint_slices, _dense_state, _WEIGHTS,
    dictionary_fingerprint,
)
from .memory_limits import CleanStop, autorelease_scope


_CHUNK_BYTES = 1024 * 1024
_CHANGED = "evaluation dictionary or settings changed: artifact binding differs"


def _metadata_bytes(metadata):
    return json.dumps({k: v for k, v in metadata.items() if k != "quality"},
                      sort_keys=True, allow_nan=False).encode()


class EvaluationBinding:
    """Bind converted tensors to disk without a second dictionary allocation."""

    def __init__(self, folder, dictionary, identity, *, memory=None):
        self.folder, self.dictionary = Path(folder), dictionary
        self.identity, self.memory = identity, memory
        self.metadata = _metadata_bytes(dictionary.metadata)
        self.layers = tuple(dictionary.layers)
        # Retain references so a replacement cannot reuse an old tensor identity.
        self.tensors = tuple(getattr(layer, name) for layer in self.layers for name in _WEIGHTS)
        self.stamps = self._tensor_stamps()
        self.files = self._file_stamps()
        self.check()
        self.evidence = {"version": 1, "tensor_hash": self._digest(compare=True)}
        self._check_files()
        self.check()

    def _tensor_stamps(self):
        layers = self.dictionary.layers
        data = self.dictionary.metadata
        if len(layers) != data["layers"] or any(layer.k != data["k"] for layer in layers):
            raise ValueError(_CHANGED)
        tensors = {f"{i}.{name}": getattr(layer, name)
                   for i, layer in enumerate(layers) for name in _WEIGHTS}
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
        """Refuse metadata, file, tensor replacement, or tensor version changes."""
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
                if set(source.keys()) != set(_WEIGHTS):
                    raise ValueError(_CHANGED)
                for name in _WEIGHTS:
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
        self._check_files()
        if self._digest() != self.evidence["tensor_hash"]:
            raise ValueError(_CHANGED)
        self.check()
