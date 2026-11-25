"""Pinned local snapshot locations for explicit reference qualification."""
import json
from pathlib import Path
from weakref import WeakKeyDictionary

import torch

from ._dictionary import _GEMMA_IDENTITY, _VERIFIED_MODELS
from ._forward_identity import _reference_profile
from ._gemma_import_identity import _REFERENCE_PROFILE, _SOURCE


_MODEL_REPO, _MODEL_REVISION = _GEMMA_IDENTITY.removeprefix("hf:").split("@")
_MODEL_SNAPSHOTS = WeakKeyDictionary()


def _pinned_snapshot(path, repo, revision):
    try:
        snapshot = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError("reference qualification requires an available local snapshot") from exc
    if (not snapshot.is_dir() or snapshot.name != revision or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--" + repo.replace("/", "--")):
        raise ValueError("reference qualification requires the pinned local snapshot")
    return snapshot


def model_snapshot(path):
    return _pinned_snapshot(path, _MODEL_REPO, _MODEL_REVISION)


def scope_snapshot():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = snapshot_download(repo_id=_SOURCE["repo"], revision=_SOURCE["revision"], local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise ValueError("reference qualification requires the pinned Scope snapshot in the local HF cache") from exc
    return _pinned_snapshot(path, _SOURCE["repo"], _SOURCE["revision"])


def qualified_reference_profile(model):
    """Check installed operations and their verified metadata without tensor copies."""
    profile = _reference_profile(model)
    if profile is None:
        return None
    if profile["profile"] != _REFERENCE_PROFILE or profile["dtype"] != "bfloat16":
        raise ValueError("reference qualification requires the explicit BF16 model profile")
    devices = {"cpu", *(parameter.device.type for parameter in model.parameters())}
    if (any(torch.is_autocast_enabled(device) for device in devices)
            or torch.get_float32_matmul_precision() != "highest"):
        raise ValueError("reference qualification requires disabled autocast and highest matmul precision")
    verified = _VERIFIED_MODELS.get(model)
    if verified is None or verified.identity != _GEMMA_IDENTITY:
        raise ValueError("reference qualification requires the verified pinned model")
    try:
        recorded = json.loads(verified.settings.get("__gemma_reference__", "null"))
    except (TypeError, ValueError) as exc:
        raise ValueError("reference qualification model profile metadata changed") from exc
    if recorded != profile:
        raise ValueError("reference qualification model profile metadata changed")
    return profile


def remember_model_snapshot(model, snapshot):
    """Retain a verified loader source location, without a qualification cache."""
    if qualified_reference_profile(model) is None:
        raise ValueError("reference qualification requires an explicit model profile")
    _MODEL_SNAPSHOTS[model] = model_snapshot(snapshot)


def reference_import_arguments(model):
    profile = qualified_reference_profile(model)
    if profile is None:
        return {}
    location = _MODEL_SNAPSHOTS.get(model, getattr(model.config, "_name_or_path", None))
    snapshot = model_snapshot(location)
    return {"profile": profile["profile"], "source": scope_snapshot(), "model_snapshot": snapshot}
