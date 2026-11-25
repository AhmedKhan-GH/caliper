"""Offline snapshot resolution without checkpoint or Scope payload reads."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _reference_snapshots as snapshots
from caliper.interp._dictionary import _GEMMA_IDENTITY, _VERIFIED_MODELS
from caliper.interp._gemma_reference import PROFILE, install_reference_profile, profile_metadata
from .test_gemma_reference import reference_fixture


def snapshot_folder(root, repo, revision):
    folder = root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def local_snapshots(tmp_path, monkeypatch):
    model = snapshot_folder(tmp_path, snapshots._MODEL_REPO, snapshots._MODEL_REVISION)
    scope = snapshot_folder(tmp_path, snapshots._SOURCE["repo"], snapshots._SOURCE["revision"])
    calls = []
    def lookup(**kwargs):
        calls.append(kwargs)
        return str(scope)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lookup)
    return SimpleNamespace(model=model, scope=scope, calls=calls)


@pytest.fixture
def qualified_model():
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    # This fixture records metadata only. It does not certify checkpoint bytes.
    _VERIFIED_MODELS[model] = SimpleNamespace(identity=_GEMMA_IDENTITY,
        settings={"__gemma_reference__": json.dumps(profile_metadata(model))})
    return model


def test_local_lookup_uses_only_pinned_scope_and_loader_source(local_snapshots, qualified_model):
    snapshots.remember_model_snapshot(qualified_model, local_snapshots.model)
    arguments = snapshots.reference_import_arguments(qualified_model)
    assert arguments == {"profile": PROFILE, "source": local_snapshots.scope,
                         "model_snapshot": local_snapshots.model}
    assert local_snapshots.calls == [{"repo_id": snapshots._SOURCE["repo"],
        "revision": snapshots._SOURCE["revision"], "local_files_only": True}]
    assert all(not list(folder.iterdir()) for folder in (local_snapshots.model, local_snapshots.scope))


def test_external_verified_model_can_retain_source_location(local_snapshots, qualified_model):
    qualified_model.config._name_or_path = str(local_snapshots.model)
    assert snapshots.reference_import_arguments(qualified_model)["model_snapshot"] == local_snapshots.model


def test_missing_verified_source_refuses_without_a_cache_guess(local_snapshots, qualified_model):
    with pytest.raises(ValueError, match="local snapshot"):
        snapshots.reference_import_arguments(qualified_model)
    assert not local_snapshots.calls


@pytest.mark.parametrize("change", ["missing", "revision", "repo", "nested", "file", "symlink"])
def test_model_snapshot_location_refusal(tmp_path, local_snapshots, change):
    folder = local_snapshots.model
    if change == "missing":
        folder.rmdir()
    elif change == "revision":
        folder = snapshot_folder(tmp_path, snapshots._MODEL_REPO, "f" * 40)
    elif change == "repo":
        folder = snapshot_folder(tmp_path, "other/model", snapshots._MODEL_REVISION)
    elif change == "nested":
        folder = folder / "nested"
        folder.mkdir()
    elif change == "file":
        folder.rmdir()
        folder.touch()
    else:
        target = tmp_path / "unqualified"
        target.mkdir()
        folder.rmdir()
        folder.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="local snapshot"):
        snapshots.model_snapshot(folder)


def test_valid_snapshot_alias_resolves_canonically(tmp_path, local_snapshots):
    alias = tmp_path / "model-alias"
    alias.symlink_to(local_snapshots.model, target_is_directory=True)
    assert snapshots.model_snapshot(alias) == local_snapshots.model


@pytest.mark.parametrize("missing", [True, False])
def test_scope_cache_missing_or_mismatched(local_snapshots, monkeypatch, missing):
    from huggingface_hub.errors import LocalEntryNotFoundError
    def lookup(**kwargs):
        assert kwargs["local_files_only"] is True
        if missing:
            raise LocalEntryNotFoundError("offline fixture cache miss")
        return str(local_snapshots.model)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lookup)
    with pytest.raises(ValueError, match="local HF cache|pinned local snapshot"):
        snapshots.scope_snapshot()


def test_saved_source_location_is_rechecked(local_snapshots, qualified_model):
    snapshots.remember_model_snapshot(qualified_model, local_snapshots.model)
    local_snapshots.model.rmdir()
    with pytest.raises(ValueError, match="available local snapshot"):
        snapshots.reference_import_arguments(qualified_model)


@pytest.mark.parametrize("change", ["unverified", "identity", "metadata", "dtype", "forward", "profile", "missing"])
def test_reference_profile_requires_actual_adapter_and_verified_metadata(qualified_model, change):
    model = qualified_model
    if change == "unverified":
        del _VERIFIED_MODELS[model]
    elif change == "identity":
        _VERIFIED_MODELS[model].identity = "hf:other/model@" + "f" * 40
    elif change == "metadata":
        metadata = deepcopy(profile_metadata(model))
        metadata["operation_sha256"] = {}
        _VERIFIED_MODELS[model].settings["__gemma_reference__"] = json.dumps(metadata)
    elif change == "dtype":
        model.to(torch.float32)
    elif change == "forward":
        model.model.layers[0].pre_feedforward_layernorm.forward = lambda value: value
    elif change == "profile":
        model._caliper_reference_profile = "unsupported"
    else:
        del model._caliper_reference_profile
    with pytest.raises(ValueError, match="reference"):
        snapshots.qualified_reference_profile(model)


def test_arbitrary_attached_profile_does_not_qualify_an_ordinary_model():
    model = reference_fixture()
    model._caliper_reference_profile = PROFILE
    with pytest.raises(ValueError, match="reference"):
        snapshots.qualified_reference_profile(model)


def test_explicit_float32_adapter_does_not_select_bf16_artifacts():
    model = reference_fixture(torch.float32)
    install_reference_profile(model, PROFILE)
    with pytest.raises(ValueError, match="BF16"):
        snapshots.qualified_reference_profile(model)


@pytest.mark.parametrize("context", ["autocast", "precision"])
def test_reference_qualification_checks_runtime_context(qualified_model, context):
    if context == "autocast":
        with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="disabled autocast"):
            snapshots.qualified_reference_profile(qualified_model)
    else:
        prior = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with pytest.raises(ValueError, match="highest matmul"):
                snapshots.qualified_reference_profile(qualified_model)
        finally:
            torch.set_float32_matmul_precision(prior)


def test_ordinary_bf16_model_never_resolves_reference_snapshots(monkeypatch):
    model = reference_fixture()
    monkeypatch.setattr(snapshots, "scope_snapshot", lambda: pytest.fail("ordinary profile queried the cache"))
    assert snapshots.reference_import_arguments(model) == {}
