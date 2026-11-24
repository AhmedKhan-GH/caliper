"""Tiny CPU checks for the upper-half dictionary and optimizer contracts."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import load_dictionary
from caliper.interp._dictionary import _manifest
from tools.interp import checkpoint_state, train_transcoders as training
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import fixture_manifest, write_fixture
from .test_training_stream import documents


def tiny_model(layers=4):
    return TinyLM(layers=layers, width=4, heads=2, latents=8, k=2, dtype=torch.float32)


def fallback_manifest(model):
    metadata = fixture_manifest(model)
    count = metadata["layers"]
    metadata["training"]["source_id"] = "synthetic-text-v1"
    metadata["training"]["fallback"] = {
        "version": 1, "mode": "upper_half",
        "active_layers": list(range(count // 2, count)), "zero_layers": list(range(count // 2)),
        "source_dictionary_hash": "a" * 64, "source_checkpoint_generation": "checkpoint-" + "b" * 32,
        "source_training_tokens": 16, "source_elapsed_seconds": 100.0, "prior_wall_seconds": 120.0,
        "aggregate_wall_limit_seconds": 36000.0, "token_budget": 16,
        "warm_start": "upper_weights_fresh_adam", "source_id": "synthetic-text-v1",
        "lower_behavior": "zero_dictionary_original_mlp",
    }
    return metadata


def upper_trainer(metadata):
    return training.Trainer(copy.deepcopy(metadata), context=4, batch_tokens=8, microbatch_tokens=4)


def text_stream():
    return TextStream(documents(), TinyTokenizer(), "train", source_id="synthetic-text-v1")


def assert_zero_lower(trainer):
    for i in trainer.metadata["training"]["fallback"]["zero_layers"]:
        assert trainer.optimizers[i] is None
        assert not trainer.last_active[i].count_nonzero()
        for parameter in trainer.layers[i].parameters():
            assert not parameter.requires_grad and not parameter.count_nonzero()


@pytest.mark.parametrize("layers", [2, 3, 4])
def test_warm_weights_fresh_adam_and_zero_lower_export(tmp_path, layers):
    model = tiny_model(layers)
    source = tmp_path / "source"
    write_fixture(source, model)
    trainer = upper_trainer(fallback_manifest(model))
    trainer.initialize_upper(source)
    assert trainer.active_layers == tuple(range(layers // 2, layers))
    assert_zero_lower(trainer)
    for i in trainer.active_layers:
        assert trainer.initialized[i]
        assert not trainer.optimizers[i].state
        for a, b in zip(trainer.layers[i].parameters(), model.layers[i].mlp.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="inactive"):
        trainer.step_layer(0, torch.ones(4, 4), torch.ones(4, 4), 0)
    stream = text_stream()
    progress = training.initial_progress()
    folder = checkpoint_state.save_checkpoint(tmp_path / "checkpoints", trainer, stream, progress)
    for i in range(layers):
        for name, _ in trainer.layers[i].named_parameters():
            assert set(load_file(folder / f"layer_{i:02d}_{name}.safetensors")) == {"weight"}
    i = trainer.active_layers[0]
    trainer.step_layer(i, torch.randn(4, 4), torch.randn(4, 4), 0)
    assert trainer.optimizers[i].state
    assert_zero_lower(trainer)
    checkpoint_state.export_dictionary(tmp_path / "export", trainer, progress)
    dictionary = load_dictionary(tmp_path / "export", model=model)
    assert len(dictionary.layers) == layers
    for i in range(layers // 2):
        assert all(not p.count_nonzero() for p in dictionary.layers[i].parameters())


@pytest.mark.parametrize("field,value", [
    ("active_layers", [1, 2, 3]), ("active_layers", [3, 2]), ("zero_layers", [0]),
    ("zero_layers", [False, 1]), ("warm_start", "old_adam"), ("mode", "all"),
    ("version", True), ("source_dictionary_hash", "bad"), ("source_checkpoint_generation", "../source"),
    ("source_id", "different"), ("prior_wall_seconds", 99), ("prior_wall_seconds", float("nan")),
    ("aggregate_wall_limit_seconds", 36001), ("token_budget", 0),
])
def test_manifest_refuses_invalid_fallback(field, value):
    metadata = fallback_manifest(tiny_model())
    metadata["training"]["fallback"][field] = value
    with pytest.raises(ValueError, match="fallback"):
        _manifest(metadata)


@pytest.mark.parametrize("name", ["W_enc", "b_enc", "W_dec", "b_dec"])
def test_loader_refuses_nonzero_lower_tensors_before_conversion(tmp_path, name):
    model = tiny_model()
    trainer = upper_trainer(fallback_manifest(model))
    checkpoint_state.export_dictionary(tmp_path, trainer, training.initial_progress())
    path = tmp_path / "layer_00.safetensors"
    values = load_file(path)
    values[name].view(-1)[0] = 1e-40  # Conversion to float16 must not hide a lower feature.
    save_file(values, path)
    with pytest.raises(ValueError, match="nonzero"):
        load_dictionary(tmp_path, model=model, dtype=torch.float16)


def test_active_batch_resume_is_exact_and_captures_only_upper(tmp_path, monkeypatch):
    model = tiny_model()
    write_fixture(tmp_path / "source", model)
    metadata = fallback_manifest(model)
    expected = upper_trainer(metadata)
    expected.initialize_upper(tmp_path / "source")
    source = text_stream()
    capture = training.capture
    calls = []

    def active_capture(model, requested):
        assert requested == [metadata[field][i] for field in ("read_points", "write_points") for i in (2, 3)]
        calls.append(requested)
        return capture(model, requested)

    monkeypatch.setattr(training, "capture", active_capture)
    full = training.train_online(model, expected, source, tmp_path / "full", token_budget=16, clock=lambda: 0)
    first = upper_trainer(metadata)
    first.initialize_upper(tmp_path / "source")
    first_source = text_stream()

    class Stop:
        def check(self):
            if first.optimizers[2].state and not first.optimizers[3].state:
                raise CleanStop("synthetic interruption")

    partial = training.train_online(model, first, first_source, tmp_path / "resume", token_budget=16,
                                    stop=Stop(), clock=lambda: 0)
    assert partial["next_layer"] == 3 and partial["tokens"] == partial["steps"] == 0
    assert len(partial["pending_ids"]) == 8
    folder, _ = checkpoint_state.checkpoint_record(tmp_path / "resume" / "checkpoints")
    assert set(load_file(folder / "layer_00_W_enc.safetensors")) == {"weight"}
    assert set(load_file(folder / "layer_02_W_enc.safetensors")) == {"weight", "step", "exp_avg", "exp_avg_sq"}
    expected_rng = (torch.rand(3), random.random(), np.random.random())
    expected_tokens = first_source.next_tokens(7).input_ids
    resumed = upper_trainer(metadata)
    resumed_source = text_stream()
    # Resume must need no warm source tensor reads.
    monkeypatch.setattr(resumed, "initialize_upper", lambda *_: pytest.fail("warm source reopened"))
    progress = checkpoint_state.restore_checkpoint(tmp_path / "resume" / "checkpoints", resumed, resumed_source)
    actual_rng = (torch.rand(3), random.random(), np.random.random())
    torch.testing.assert_close(actual_rng[0], expected_rng[0], rtol=0, atol=0)
    assert actual_rng[1:] == expected_rng[1:]
    assert torch.equal(resumed_source.next_tokens(7).input_ids, expected_tokens)
    # Reset the cursor after its independent next-token assertion.
    progress = checkpoint_state.restore_checkpoint(tmp_path / "resume" / "checkpoints", resumed, resumed_source)
    completed = training.train_online(model, resumed, resumed_source, tmp_path / "resume", token_budget=16,
                                      progress=progress, clock=lambda: 0)
    assert completed["tokens"] == full["tokens"] == 16
    assert completed["steps"] == full["steps"] == 2
    assert calls
    assert_zero_lower(resumed)
    for i, (original, actual) in enumerate(zip(expected.layers, resumed.layers)):
        torch.testing.assert_close(expected.last_active[i], resumed.last_active[i], rtol=0, atol=0)
        for a, b in zip(original.parameters(), actual.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if i in resumed.active_layers:
                for key in expected.optimizers[i].state[a]:
                    torch.testing.assert_close(expected.optimizers[i].state[a][key],
                                               resumed.optimizers[i].state[b][key], rtol=0, atol=0)


def test_legacy_fixture_manifest_and_settings_stay_unchanged(tmp_path):
    model, metadata = write_fixture(tmp_path, tiny_model())
    before = (tmp_path / "dictionary.json").read_bytes()
    assert load_dictionary(tmp_path, model=model).metadata == metadata
    assert (tmp_path / "dictionary.json").read_bytes() == before
    trainer = training.Trainer(metadata)
    assert set(trainer.settings) == {"learning_rate", "microbatch_tokens", "auxiliary_weight", "dead_tokens",
                                     "context", "batch_tokens", "decoder_normalization", "layers", "width",
                                     "latents", "k", "dtype"}
    assert all(optimizer is not None for optimizer in trainer.optimizers)
