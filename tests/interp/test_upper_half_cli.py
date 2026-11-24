"""Synthetic fallback admission, resume identity, and aggregate time checks."""
import copy
import json
import shutil

import pytest
import torch

from tools.interp import checkpoint_state, train_transcoders as training
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_training_stream import documents
from .test_upper_half import tiny_model


def synthetic_source(folder):
    """Create artificial completed-run records without a learning or quality run."""
    model = tiny_model()
    metadata = training.make_manifest(model, 8, 2)
    stream = TextStream(documents(), TinyTokenizer(), "train")
    metadata["training"].update(source_id=stream.source_id, token_budget=16, tokens=16,
                                elapsed_seconds=100, stop_reason="token_budget",
                                partial_batch_tokens=0, next_layer=0)
    trainer = training.Trainer(metadata, context=4, batch_tokens=8, microbatch_tokens=4)
    progress = training.initial_progress()
    progress.update(tokens=16, steps=2, elapsed_seconds=110)
    generation = checkpoint_state.save_checkpoint(folder / "checkpoints", trainer, stream, progress)
    checkpoint_state.export_dictionary(folder, trainer, progress)
    checkpoint_state.write_json(folder / "runtime.json", {"generation": generation.name, "elapsed_seconds": 130})
    metadata["quality"] = {"fvu": 0.31, "dead_fraction": 0.1, "cross_entropy_delta": 0.2,
                           "evaluated_tokens": 200_000, "status": "failed", "partition": "quality",
                           "stop_reason": "complete", "all_mlps_replaced": 4}
    checkpoint_state.write_json(folder / "dictionary.json", metadata)
    return model, metadata, stream.source_id


def arguments(source, output):
    return ["--output", str(output), "--upper-half-from", str(source), "--prior-wall-seconds", "140",
            "--device", "cpu", "--latents", "8", "--k", "2", "--context", "4", "--batch-tokens", "8",
            "--microbatch-tokens", "4", "--tokens", "16"]


def setup_fallback(tmp_path):
    source, output = tmp_path / "source", tmp_path / "fallback"
    model, metadata, source_id = synthetic_source(source)
    argv = arguments(source, output)
    args = training.parser().parse_args(argv)
    prepared = training.prepare_upper_half(args, source_id=source_id)
    return source, output, model, metadata, source_id, argv, args, prepared


def test_admission_binds_source_bytes_and_excludes_quality(tmp_path):
    source, _, _, metadata, source_id, _, args, prepared = setup_fallback(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    fallback = prepared["training"]["fallback"]
    assert fallback["source_elapsed_seconds"] == 130
    assert fallback["prior_wall_seconds"] == 140
    assert fallback["source_training_tokens"] == fallback["token_budget"] == 16
    assert prepared["training"]["tokens"] == prepared["training"]["elapsed_seconds"] == 0
    assert fallback["source_dictionary_hash"] == checkpoint_state.dictionary_fingerprint(source, metadata)
    metadata["quality"]["fvu"] = 0.4
    assert checkpoint_state.dictionary_fingerprint(source, metadata) == fallback["source_dictionary_hash"]
    metadata["training"]["seed"] = 77
    assert checkpoint_state.dictionary_fingerprint(source, metadata) != fallback["source_dictionary_hash"]
    assert training.prepare_upper_half(args, source_id=source_id) == prepared
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("section,key,value", [
    ("quality", "status", "partial"), ("quality", "evaluated_tokens", 199999),
    ("quality", "partition", "train"), ("quality", "all_mlps_replaced", 2),
    ("quality", "fvu", 0.2), ("quality", "cross_entropy_delta", None),
    ("training", "tokens", 8), ("training", "token_budget", 32),
    ("training", "partial_batch_tokens", 8), ("training", "stop_reason", "wall time reserve"),
    ("training", "source_id", "other"), ("training", "dataset_revision", "other"),
])
def test_invalid_source_refuses_before_model_load(tmp_path, monkeypatch, section, key, value):
    source, _, _, metadata, _, argv, _, _ = setup_fallback(tmp_path)
    metadata[section][key] = value
    checkpoint_state.write_json(source / "dictionary.json", metadata)
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: documents())
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError):
        training.main(argv)


@pytest.mark.parametrize("extra", [
    ["--prior-wall-seconds", "129"], ["--prior-wall-seconds", "nan"], ["--prior-wall-seconds", "-1"],
    ["--prior-wall-seconds", "35940"], ["--prior-wall-seconds", "36000"],
    ["--max-wall-seconds", "36001"], ["--latents", "16"], ["--k", "1"],
])
def test_invalid_cli_refuses_before_model_load(tmp_path, monkeypatch, extra):
    _, _, _, _, _, argv, _, _ = setup_fallback(tmp_path)
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: documents())
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError):
        training.main([*argv, *extra])


@pytest.mark.parametrize("extra", [["--prior-wall-seconds", "10"], ["--upper-half-from", "source"]])
def test_fallback_options_are_required_together(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError, match="required together"):
        training.main(["--output", str(tmp_path), *extra])


def test_source_alias_and_existing_output_refuse(tmp_path):
    source, output, _, _, source_id, _, args, _ = setup_fallback(tmp_path)
    output.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        training.prepare_upper_half(args, source_id=source_id)
    output.unlink()
    output.mkdir()
    (output / "dictionary.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        training.prepare_upper_half(args, source_id=source_id)


def fallback_checkpoint(tmp_path):
    source, output, model, metadata, source_id, argv, args, prepared = setup_fallback(tmp_path)
    trainer = training.Trainer(prepared, **training._trainer_arguments(args))
    trainer.initialize_upper(source)
    stream = TextStream(documents(), TinyTokenizer(), "train")
    folder = checkpoint_state.save_checkpoint(output / "checkpoints", trainer, stream, training.initial_progress())
    return source, output, folder, argv


@pytest.mark.parametrize("change", ["tokens", "cap", "prior", "source_path", "source_manifest", "source_weights",
                                   "active_layers", "warm_start", "learning_rate", "context", "mode"])
def test_changed_resume_contract_refuses_before_model_load(tmp_path, monkeypatch, change):
    source, _, folder, argv = fallback_checkpoint(tmp_path)
    argv.append("--resume")
    if change in ("tokens", "cap", "prior", "learning_rate", "context"):
        flag, value = {"tokens": ("--tokens", "32"), "cap": ("--max-wall-seconds", "35000"),
                       "prior": ("--prior-wall-seconds", "150"), "learning_rate": ("--learning-rate", "0.001"),
                       "context": ("--context", "2")}[change]
        argv.extend([flag, value])
    elif change == "source_path":
        shutil.copytree(source, tmp_path / "copy")
        argv.extend(["--upper-half-from", str(tmp_path / "copy")])
    elif change == "source_manifest":
        metadata = json.loads((source / "dictionary.json").read_text())
        metadata["training"]["seed"] = 99
        checkpoint_state.write_json(source / "dictionary.json", metadata)
    elif change == "source_weights":
        from safetensors.torch import load_file, save_file
        values = load_file(source / "layer_03.safetensors")
        values["b_dec"][0] += 1
        save_file(values, source / "layer_03.safetensors")
    elif change == "mode":
        for option in ("--upper-half-from", "--prior-wall-seconds"):
            index = argv.index(option)
            del argv[index:index + 2]
    else:
        record = json.loads((folder / "state.json").read_text())
        record["metadata"]["training"]["fallback"][change] = [3] if change == "active_layers" else "reuse_adam"
        checkpoint_state.write_json(folder / "state.json", record)
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: documents())
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError):
        training.main(argv)


def test_resume_time_includes_prior_and_runtime_export(tmp_path, monkeypatch):
    _, output, folder, argv = fallback_checkpoint(tmp_path)
    # The checkpoint is short, but its final export consumed the remaining time.
    checkpoint_state.write_json(output / "runtime.json", {"generation": folder.name, "elapsed_seconds": 35810})
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: pytest.fail("source opened after budget exhaustion"))
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError, match="reserve"):
        training.main([*argv, "--resume"])


def test_main_uses_checkpoint_on_resume_and_keeps_warm_source_unchanged(tmp_path, monkeypatch):
    source, output, model, _, _, argv, _, _ = setup_fallback(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: documents())
    monkeypatch.setattr(training, "load_text_model", lambda **_: (model, TinyTokenizer()))
    # This is two synthetic CPU batches, with no external model or learning benchmark.
    training.main(argv)
    exported = json.loads((output / "dictionary.json").read_text())
    assert exported["training"]["status"] == "fallback_complete"
    assert exported["training"]["tokens"] == 16
    assert "upper_half_from" not in exported["training"]["optimizer"]
    monkeypatch.setattr(training.Trainer, "initialize_upper", lambda *_: pytest.fail("source weights reinitialized"))
    training.main([*argv, "--resume"])
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_aggregate_allowance_counts_setup_checkpoint_and_export(tmp_path, monkeypatch):
    source, output, model, _, _, _, args, metadata = setup_fallback(tmp_path)
    metadata["training"]["fallback"].update(source_elapsed_seconds=100, prior_wall_seconds=100,
                                            aggregate_wall_limit_seconds=200)
    trainer = training.Trainer(metadata, **training._trainer_arguments(args))
    trainer.initialize_upper(source)
    stream = TextStream(documents(), TinyTokenizer(), "train")
    progress = training.initial_progress()
    progress["elapsed_seconds"] = 20

    class Clock:
        value = 30.0  # Synthetic model-load and setup time in this invocation.
        def __call__(self):
            return self.value

    clock = Clock()
    save, export = training.save_checkpoint, training.export_dictionary

    def counted_save(*args, **kwargs):
        clock.value += 2
        return save(*args, **kwargs)

    def counted_export(*args, **kwargs):
        clock.value += 3
        return export(*args, **kwargs)

    def forward_time(*_):
        clock.value += 35

    monkeypatch.setattr(training, "save_checkpoint", counted_save)
    monkeypatch.setattr(training, "export_dictionary", counted_export)
    handle = model.layers[0].register_forward_pre_hook(forward_time)
    try:
        result = training.train_online(model, trainer, stream, output, token_budget=16,
                                        progress=progress, max_wall_seconds=200, started=0, clock=clock)
    finally:
        handle.remove()
    assert result["tokens"] == 0 and result["pending_ids"] is not None
    assert result["elapsed_seconds"] == 92
    assert trainer.metadata["training"]["status"] == "fallback_incomplete"
    assert trainer.metadata["training"]["stop_reason"] == "wall time reserve"
    assert trainer.metadata["training"]["elapsed_seconds"] + 100 == 192
    assert checkpoint_state.checkpoint_record(output / "checkpoints")[1]["progress"]["elapsed_seconds"] == 92


@pytest.mark.parametrize("damage", ["lower_index", "missing_batch", "uninitialized", "over_budget"])
def test_invalid_fallback_cursor_refuses_before_model_load(tmp_path, monkeypatch, damage):
    _, _, folder, argv = fallback_checkpoint(tmp_path)
    record = json.loads((folder / "state.json").read_text())
    if damage == "lower_index":
        record["progress"].update(next_layer=1, pending_ids=[1, 2, 3, 4])
    elif damage == "missing_batch":
        record["progress"]["next_layer"] = 3
    elif damage == "uninitialized":
        record["initialized"][2] = False
    else:
        record["progress"]["tokens"] = 17
    checkpoint_state.write_json(folder / "state.json", record)
    monkeypatch.setattr(training, "fineweb_documents", lambda **_: documents())
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    with pytest.raises(ValueError, match="progress|initialization"):
        training.main([*argv, "--resume"])
