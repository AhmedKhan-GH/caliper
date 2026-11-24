"""Bounded evidence for the artifact and tensors in one evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from caliper.interp._dictionary import (
    _dense_state, _hf_digest,
    dictionary_fingerprint, model_identity, weight_names,
)
from caliper.interp._evaluation_identity import (
    _evaluation_settings, evaluation_dictionary_evidence, evaluation_model_evidence,
)
from caliper.interp._gemma_import_identity import verified_import_binding


_CHANGED = "evaluation dictionary or settings changed: artifact binding differs"
_MODEL_CHANGED = "evaluation model identity mismatch: runtime binding differs"


def _metadata_bytes(metadata):
    return json.dumps({k: v for k, v in metadata.items() if k != "quality"},
                      sort_keys=True, allow_nan=False).encode()


class EvaluationBinding:
    """Bind converted tensors to disk without a second dictionary allocation."""

    def __init__(self, folder, dictionary, identity, *, model, memory=None):
        self.folder, self.dictionary = Path(folder), dictionary
        self.identity, self.memory = identity, memory
        self.model = model
        self.model_hash = dictionary.metadata["model_hash"]
        self._check_model_identity()
        self.import_evidence = verified_import_binding(
            self.folder, dictionary.metadata, model=model, memory=memory)
        # Keep original modules and tensors alive through temporary MLP replacements.
        self.model_modules = tuple(model.modules())
        self.model_tensors = tuple(self._model_tensors().values())
        self.model_stamps = self._model_stamps()
        self.model_settings = _evaluation_settings(model, memory)
        self.model_evidence = evaluation_model_evidence(
            model, self.model_hash, memory=memory, settings=self.model_settings)
        self.metadata = _metadata_bytes(dictionary.metadata)
        self.layers = tuple(dictionary.layers)
        # Retain references so a replacement cannot reuse an old tensor identity.
        self.tensors = tuple(getattr(layer, name) for layer in self.layers for name in weight_names(layer))
        self.stamps = self._tensor_stamps()
        self.files = self._file_stamps()
        self.check()
        self.evidence = evaluation_dictionary_evidence(dictionary, memory=memory, folder=self.folder)
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

    def _check_files(self):
        if dictionary_fingerprint(self.folder, self.dictionary.metadata, memory=self.memory) != self.identity:
            raise ValueError(_CHANGED)

    def verify(self):
        """Recheck bytes before publication, including aliases without version updates."""
        self.check()
        self._check_model_identity()
        if verified_import_binding(self.folder, self.dictionary.metadata, model=self.model,
                                   memory=self.memory) != self.import_evidence:
            raise ValueError(_CHANGED)
        if _hf_digest(self.model, memory=self.memory) != self.model_evidence["tensor_hash"]:
            raise ValueError(_MODEL_CHANGED)
        self._check_files()
        if evaluation_dictionary_evidence(self.dictionary, memory=self.memory) != self.evidence:
            raise ValueError(_CHANGED)
        self.check()
