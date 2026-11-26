"""CPU dispatch checks for the explicit index model profile."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.interp import build_index as command, model_load
from caliper.interp._gemma_reference import PROFILE
from caliper.interp import _reference_snapshots as snapshots
from .test_model_load_modes import logical_guard
from .test_reference_snapshots import local_snapshots, qualified_model


@pytest.mark.parametrize("profile", [None, PROFILE])
def test_gemma_loader_selects_explicit_reference_only(local_snapshots, qualified_model, monkeypatch, profile):
    calls = []
    model, tokenizer = qualified_model, object()

    def snapshot(model_id, **kwargs):
        calls.append(("snapshot", model_id, kwargs))
        return str(local_snapshots.model)

    def load(folder, **kwargs):
        calls.append(("model", folder, kwargs))
        return model

    def tokenize(folder, **kwargs):
        calls.append(("tokenizer", folder, kwargs))
        return tokenizer

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenize)))
    monkeypatch.setattr(model_load, "_load_gemma_model", load)
    guard = logical_guard(8, {"current": 0, "peak": 0})
    actual = model_load.load_text_model(device="cpu", memory=guard,
        model_id=model_load.GEMMA_ID, reference_profile=profile)
    assert actual == (model, tokenizer)
    assert calls[0] == ("snapshot", model_load.GEMMA_ID,
                        {"revision": model_load.GEMMA_REVISION, "local_files_only": True})
    options = calls[1][2]
    assert options["device"] == "cpu" and options["memory"].limit == guard.limit
    assert options["memory"].memory is guard
    assert {key: value for key, value in options.items() if key not in ("device", "memory")} == (
        {} if profile is None else {"dtype": torch.bfloat16, "reference_profile": PROFILE})
    assert calls[1][1] == local_snapshots.model
    assert calls[2] == ("tokenizer", local_snapshots.model,
                        {"local_files_only": True, "trust_remote_code": False})
    if profile is None:
        assert model not in snapshots._MODEL_SNAPSHOTS
    else:
        assert snapshots._MODEL_SNAPSHOTS[model] == local_snapshots.model
        # The loader registry, rather than mutable config text, supplies the path.
        model.config._name_or_path = "untrusted-location"
        monkeypatch.setattr(snapshots, "scope_snapshot", lambda: local_snapshots.scope)
        assert snapshots.reference_import_arguments(model) == {
            "profile": PROFILE, "source": local_snapshots.scope,
            "model_snapshot": local_snapshots.model}


@pytest.mark.parametrize("model_id,profile", [
    ("example/unsupported", PROFILE),
    *[(model_load.GEMMA_ID, value) for value in ("caliper", "other", False, {}, 1)],
])
def test_invalid_reference_selection_refuses_before_resources(monkeypatch, model_id, profile):
    monkeypatch.setattr(model_load, "MemoryGuard", lambda *args: pytest.fail("resource admission ran"))
    with pytest.raises(ValueError):
        model_load.load_text_model(model_id=model_id, reference_profile=profile)


def test_cli_default_and_invalid_reference_model(tmp_path):
    args = ["--dictionary", str(tmp_path)]
    assert command.parser().parse_args(args).profile == "caliper"
    explicit = command.parser().parse_args([*args, "--model", model_load.GEMMA_ID, "--profile", PROFILE])
    assert explicit.profile == PROFILE and explicit.memory_gb == 40
    with pytest.raises(SystemExit) as error:
        command.parser().parse_args([*args, "--model", "example/unsupported", "--profile", PROFILE])
    assert error.value.code == 2


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("profile", ["caliper", PROFILE])
def test_cli_passes_reference_selection_to_loader(tmp_path, monkeypatch, query, profile):
    metadata = {"layers": 2, "latents": 4, "activation": "jumprelu"}
    if profile == PROFILE:
        metadata["reference_input"] = {"coordinate": "gemma_scope_pre_gain",
                                       "encoder": "W_enc_original", "dtype": "bfloat16"}
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "validate_partitions", lambda data: {"partition": "index"})
    monkeypatch.setattr(command, "_index_stream", lambda *args, **kwargs: object())
    monkeypatch.setattr(command, "MemoryGuard", lambda limit, device: logical_guard(limit, {"current": 0, "peak": 0}))
    calls = []

    class Boundary(Exception):
        pass

    def load(**kwargs):
        calls.append(kwargs)
        raise Boundary

    monkeypatch.setattr(command, "load_text_model", load)
    args = ["--dictionary", str(tmp_path), "--device", "cpu", "--model", model_load.GEMMA_ID,
            "--profile", profile]
    if query:
        args += ["--query", "0", "0"]
    with pytest.raises(Boundary):
        command.main(args)
    assert len(calls) == 1
    options = {key: value for key, value in calls[0].items() if key != "memory"}
    assert options == {"device": "cpu", "model_id": model_load.GEMMA_ID,
                       **({"reference_profile": PROFILE} if profile == PROFILE else {})}


@pytest.mark.parametrize("profile,coordinate", [("caliper", "bfloat16"), (PROFILE, None), (PROFILE, "float32")])
def test_cli_refuses_artifact_profile_mismatch_before_source(tmp_path, monkeypatch, profile, coordinate):
    metadata = {"activation": "jumprelu"}
    if coordinate is not None:
        metadata["reference_input"] = {"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original",
                                       "dtype": coordinate}
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "validate_partitions", lambda data: {"partition": "index"})
    monkeypatch.setattr(command, "_index_stream", lambda *args, **kwargs: pytest.fail("source admission ran"))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("model load ran"))
    with pytest.raises(ValueError, match="artifact"):
        command.main(["--dictionary", str(tmp_path), "--model", model_load.GEMMA_ID, "--profile", profile])
