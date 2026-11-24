"""Synthetic CLI admission checks without model or optimizer allocation."""
import json

import pytest

from tools.interp import train_transcoders as training


def snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def checkpoint(folder):
    """Write only the metadata that startup admission reads."""
    args = training.parser().parse_args(["--output", str(folder), "--device", "cpu"])
    metadata = {"layers": 2, "width": 8, "latents": args.latents, "k": args.k,
                "model_hash": "synthetic", "training": {"token_budget": args.tokens}}
    record = {"version": 1, "metadata": metadata,
              "settings": training.trainer_settings(metadata, **training._trainer_arguments(args)),
              "stream": {"source_id": "synthetic"}, "progress": training.initial_progress()}
    generation = folder / "checkpoints" / "checkpoint-synthetic"
    generation.mkdir(parents=True)
    (generation / "state.json").write_text(json.dumps(record))
    (generation.parent / "latest.json").write_text(json.dumps({"generation": generation.name}))
    (folder / "dictionary.json").write_text('{"quality": {"status": "passed"}}')
    (folder / "layer_00.safetensors").write_bytes(b"prior synthetic weights")
    return args, record, generation / "state.json"


def forbid_allocations(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("startup refusal reached source or model allocation")
    for name in ("MemoryGuard", "load_text_model", "Trainer", "fineweb_documents", "train_online"):
        monkeypatch.setattr(training, name, fail)


@pytest.mark.parametrize("option,value", [
    ("latents", "8192"), ("k", "16"), ("learning-rate", "0.001"),
    ("microbatch-tokens", "128"), ("auxiliary-weight", "0"), ("dead-tokens", "123"),
    ("context", "128"), ("batch-tokens", "2048"), ("tokens", "20000000"),
])
def test_all_layer_resume_refuses_changed_settings_before_allocations(tmp_path, monkeypatch, option, value):
    checkpoint(tmp_path)
    before = snapshot(tmp_path)
    forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="settings|token budget"):
        training.main(["--output", str(tmp_path), "--resume", "--device", "cpu", f"--{option}", value])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("artifact", ["dictionary.json", "layer_00.safetensors", "evaluation/state.json",
                                       "index/index.json", "checkpoints/latest.json", "unrelated.txt"])
def test_fresh_run_refuses_nonempty_output_before_allocations(tmp_path, monkeypatch, artifact):
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"prior artifact")
    before = snapshot(tmp_path)
    forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="output is not empty"):
        training.main(["--output", str(tmp_path), "--device", "cpu"])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("option,value", [("learning-rate", "nan"), ("auxiliary-weight", "inf"),
                                          ("dead-tokens", "0"), ("memory-gb", "25")])
def test_invalid_settings_refuse_before_allocations(tmp_path, monkeypatch, option, value):
    forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="settings or memory"):
        training.main(["--output", str(tmp_path), "--device", "cpu", f"--{option}", value])
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("resume", [False, True])
def test_matching_settings_reach_resource_admission(tmp_path, monkeypatch, resume):
    if resume:
        checkpoint(tmp_path)
    before = snapshot(tmp_path)
    forbid_allocations(monkeypatch)

    def admitted(*args, **kwargs):
        raise training.CleanStop("synthetic admission boundary")

    monkeypatch.setattr(training, "MemoryGuard", admitted)
    arguments = ["--output", str(tmp_path), "--device", "cpu"] + (["--resume"] if resume else [])
    assert training.main(arguments) == 2
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("requested,current", [(None, 18), (24, 18), (8, 6)])
def test_trainer_selects_training_load_budget_before_optimizer(tmp_path, monkeypatch, capsys, requested, current):
    from caliper.interp import _dictionary
    from tools.interp import model_load
    from .test_model_load import hf_model, mocked_loader
    from .test_model_load_modes import logical_guard

    model = hf_model()
    calls = mocked_loader(tmp_path, monkeypatch, model, {})
    limit = 24 if requested is None else requested
    guard = logical_guard(limit, {"current": int(current * 1e9), "peak": 0})

    def memory(limit_gb, device):
        assert limit_gb == limit and device == "cpu"
        return guard

    def load(*, device, memory, mode):
        assert device == "cpu" and memory is guard and mode == "training"
        return model_load.load_text_model(device=device, memory=memory, mode=mode)

    def admitted(loaded, *args):
        assert loaded is model
        assert calls == ["snapshot", "weights", "tokenizer"]
        retained = _dictionary._VERIFIED_MODELS[model].memory
        assert retained.memory is guard and retained.limit == guard.limit == int(limit * 1e9)
        assert _dictionary.model_identity(model) == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
        raise training.CleanStop("fixture admission complete")

    def forbidden(*args, **kwargs):
        pytest.fail("trainer allocation or training ran")

    monkeypatch.setattr(training, "MemoryGuard", memory)
    monkeypatch.setattr(training, "load_text_model", load)
    monkeypatch.setattr(training, "make_manifest", admitted)
    for name in ("Trainer", "fineweb_documents", "train_online"):
        monkeypatch.setattr(training, name, forbidden)
    output = tmp_path / "output"
    argv = ["--output", str(output), "--device", "cpu"]
    if requested is not None:
        argv += ["--memory-gb", str(requested)]
    assert training.main(argv) == 2
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["stage"] == "clean_stop" and result["reason"] == "fixture admission complete"
    assert not output.exists()
