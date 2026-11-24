"""Portable checkpoint file flush checks without a trainer run."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tools.interp import checkpoint_state


def test_tensor_file_sync_uses_writable_handle(tmp_path, monkeypatch):
    path = tmp_path / "weights.safetensors"
    opened = Path.open
    handles = {}
    synced = []
    sync = checkpoint_state.os.fsync

    def track(file, *args, **kwargs):
        handle = opened(file, *args, **kwargs)
        handles[handle.fileno()] = handle
        return handle

    def writable_sync(descriptor):
        assert handles[descriptor].writable(), "file sync requires write access"
        synced.append(descriptor)
        sync(descriptor)

    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(checkpoint_state.os, "fsync", writable_sync)
    values = {"weight": torch.arange(6).reshape(2, 3).float()}
    checkpoint_state._save_tensors(values, path)
    assert len(synced) == 1 and handles[synced[0]].closed
    assert torch.equal(load_file(path)["weight"], values["weight"])


def test_tensor_file_sync_failure_closes_handle(tmp_path, monkeypatch):
    path = tmp_path / "weights.safetensors"
    opened = Path.open
    handles = []

    def track(file, *args, **kwargs):
        handle = opened(file, *args, **kwargs)
        handles.append(handle)
        return handle

    def fail(_):
        raise OSError("fixture sync failure")

    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(checkpoint_state.os, "fsync", fail)
    with pytest.raises(OSError, match="sync failure"):
        checkpoint_state._save_tensors({"weight": torch.ones(2)}, path)
    assert handles and all(handle.closed for handle in handles)
    path.unlink()


@pytest.mark.parametrize("saved,current", [
    (None, {"identity": "a"}), ({"identity": "a"}, None),
    ({"identity": "a"}, {"identity": "b"}),
])
def test_checkpoint_refuses_changed_fallback_evidence(saved, current):
    metadata = {"model_hash": "a" * 64, "training": {}}
    previous = dict(metadata)
    if saved is not None:
        previous["fallback_source"] = saved
    if current is not None:
        metadata["fallback_source"] = current
    with pytest.raises(ValueError, match="fallback source evidence"):
        checkpoint_state.validate_checkpoint({"settings": {}, "metadata": previous}, {}, metadata, "source")


@pytest.mark.parametrize("evidence", [None, {"identity": "a"}])
def test_checkpoint_preserves_optional_fallback_evidence(evidence):
    from copy import deepcopy
    metadata = {"model_hash": "a" * 64, "training": {}}
    if evidence is not None:
        metadata["fallback_source"] = evidence
    checkpoint_state.validate_checkpoint({"settings": {}, "metadata": deepcopy(metadata)}, {}, metadata, "source")
