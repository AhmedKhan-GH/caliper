"""Fresh-weight learning and exact interrupted-training resume on CPU."""
import random

import numpy as np
import pytest
import torch

from caliper.interp import capture, load_dictionary
from tools.interp.checkpoint_state import checkpoint_record, restore_checkpoint, save_checkpoint
from tools.interp.memory_limits import CleanStop
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream
from tools.interp.train_transcoders import Trainer, initial_progress, train_online
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import fixture_manifest
from .test_training_stream import documents


def fixture_text():
    """Distinct neutral documents for the real tokenizer and partition helper."""
    generator = random.Random(222)
    colors = ("blue", "green", "red", "yellow", "purple", "white", "black", "orange")
    shapes = ("square", "circle", "triangle", "rectangle", "cube", "sphere", "line", "point")
    directions = ("left", "right", "up", "down", "north", "south", "east", "west")
    rows = []
    for index in range(10_000):
        sentences = [f"Pattern {index} contains several objects."]
        for _ in range(3):
            color, shape, direction = (generator.choice(items) for items in (colors, shapes, directions))
            sentences.append(f"The {color} {shape} moves {direction} across the grid.")
        rows.append({"text": " ".join(sentences)})
    return rows


def test_fresh_dictionary_learns_held_out_fixture(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(123)
        model = TinyLM(dtype=torch.float32)
        metadata = fixture_manifest(model)
        # Match the real dictionary's 16-times-width expansion and k=32.
        metadata.update(latents=1024, k=32)
        trainer = Trainer(metadata, learning_rate=1e-3,
                          microbatch_tokens=128, auxiliary_weight=1 / 32, dead_tokens=16_384,
                          context=32, batch_tokens=128)
        originals = [layer.W_enc.detach().clone() for layer in trainer.layers]
        assert all(not torch.equal(originals[i], block.mlp.W_enc) for i, block in enumerate(model.layers))
        source = fixture_text()
        tokenizer = TinyTokenizer()
        quality = TextStream(source, tokenizer, "quality", source_id="fixture-neutral-v1")
        held_out = quality.next_tokens(2048).input_ids.reshape(-1, 32)
        selected = trainer.metadata["read_points"] + trainer.metadata["write_points"]
        with capture(model, selected) as values, torch.no_grad():
            hidden_forward(model, held_out)
        def held_out_fvu():
            result = []
            with torch.no_grad():
                for i, layer in enumerate(trainer.layers):
                    x = values[trainer.metadata["read_points"][i]].reshape(-1, 64)
                    y = values[trainer.metadata["write_points"][i]].reshape(-1, 64)
                    result.append(float((layer(x) - y).square().sum() / (y - y.mean(0)).square().sum()))
            return result
        initial = held_out_fvu()
        training = TextStream(source, tokenizer, "train", source_id="fixture-neutral-v1")
        # Numerical correctness uses fixed updates. Local timing acceptance is separate.
        progress = train_online(model, trainer, training, tmp_path, token_budget=640_000,
                                clock=lambda: 0)
        fvu = held_out_fvu()
        values.close(clear=True)
        print({"initial_fvu": initial, "held_out_fvu": fvu, "tokens": progress["tokens"]}, flush=True)
        assert progress["tokens"] == 640_000 and progress["steps"] == 5000
        assert trainer.metadata["training"]["stop_reason"] == "token_budget"
        assert sum(initial) / 4 > 0.1
        assert sum(fvu) / 4 < 0.1
        for i, layer in enumerate(trainer.layers):
            assert not torch.equal(originals[i], layer.W_enc)
            assert torch.isfinite(layer.W_dec).all()
    finally:
        torch.set_num_threads(previous_threads)


def small_trainer(seed=5):
    torch.manual_seed(seed)
    model = TinyLM(layers=2, width=8, heads=2, latents=16, k=2, dtype=torch.float32)
    trainer = Trainer(fixture_manifest(model), learning_rate=1e-3, microbatch_tokens=8,
                      context=4, batch_tokens=16, dead_tokens=16)
    return model, trainer, TextStream(documents(), TinyTokenizer(), "train")


def test_checkpoint_restores_optimizer_rng_and_stream(tmp_path):
    model, trainer, stream = small_trainer()
    inputs = torch.randn(16, 8)
    with torch.no_grad():
        targets = model.layers[0].mlp(inputs)
    trainer.step_layer(0, inputs, targets, 0)
    stream.next_tokens(17)
    progress = initial_progress()
    progress.update(tokens=16, steps=1, next_layer=1, pending_ids=[4, 5, 6, 7])
    save_checkpoint(tmp_path, trainer, stream, progress)
    expected_random = (torch.rand(7), random.random(), np.random.random())
    expected_next = stream.next_tokens(31)
    trainer.step_layer(0, inputs, targets, 16)
    _, resumed, resumed_stream = small_trainer(seed=77)
    actual_progress = restore_checkpoint(tmp_path, resumed, resumed_stream)
    actual_random = (torch.rand(7), random.random(), np.random.random())
    torch.testing.assert_close(actual_random[0], expected_random[0], rtol=0, atol=0)
    assert actual_random[1:] == expected_random[1:]
    assert actual_progress == progress
    assert torch.equal(resumed_stream.next_tokens(31).input_ids, expected_next.input_ids)
    resumed.step_layer(0, inputs, targets, 16)
    for first, second in zip(trainer.layers[0].parameters(), resumed.layers[0].parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for key in trainer.optimizers[0].state[first]:
            torch.testing.assert_close(trainer.optimizers[0].state[first][key], resumed.optimizers[0].state[second][key], rtol=0, atol=0)


def test_online_resume_mid_batch_matches_uninterrupted(tmp_path):
    model, expected, source = small_trainer()
    full = train_online(model, expected, source, tmp_path / "full", token_budget=32)
    model, first, source = small_trainer()
    class Stop:
        def check(self):
            # Stop after the first layer's first optimizer update.
            if first.optimizers[0].state and not first.optimizers[1].state:
                raise CleanStop("fixture interruption")
    partial = train_online(model, first, source, tmp_path / "resume", token_budget=32, stop=Stop())
    assert partial["next_layer"] == 1 and partial["tokens"] == 0
    assert len(partial["pending_ids"]) == 16
    model, resumed, resumed_source = small_trainer(seed=99)
    progress = restore_checkpoint(tmp_path / "resume" / "checkpoints", resumed, resumed_source)
    completed = train_online(model, resumed, resumed_source, tmp_path / "resume", token_budget=32, progress=progress)
    assert completed["tokens"] == full["tokens"] == 32
    assert completed["steps"] == full["steps"] == 2
    for original, actual in zip(expected.layers, resumed.layers):
        for a, b in zip(original.parameters(), actual.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    dictionary = load_dictionary(tmp_path / "resume", model=model)
    assert dictionary.metadata["training"]["tokens"] == 32
    assert dictionary.metadata["quality"]["fvu"] is None


def test_failed_save_keeps_prior_complete_generation(tmp_path, monkeypatch):
    from tools.interp import checkpoint_state
    _, trainer, source = small_trainer()
    progress = initial_progress()
    first = save_checkpoint(tmp_path, trainer, source, progress)
    def fail(*_):
        raise OSError("fixture write failure")
    monkeypatch.setattr(checkpoint_state, "_save_tensors", fail)
    with pytest.raises(OSError, match="write failure"):
        save_checkpoint(tmp_path, trainer, source, progress)
    folder, record = checkpoint_record(tmp_path)
    assert folder == first
    assert record["progress"] == progress


def test_wall_limit_and_periodic_saves_preserve_pending_batch(tmp_path, monkeypatch):
    from tools.interp import train_transcoders
    model, trainer, source = small_trainer()
    class Clock:
        value = 0
        def __call__(self):
            return self.value
    clock = Clock()
    def advance(*_):
        clock.value += 2
    hook = model.layers[0].register_forward_pre_hook(advance)
    times = []
    implementation = train_transcoders.save_checkpoint
    def record(*args, **kwargs):
        times.append(clock())
        return implementation(*args, **kwargs)
    monkeypatch.setattr(train_transcoders, "save_checkpoint", record)
    try:
        progress = train_online(model, trainer, source, tmp_path, token_budget=64,
                                max_wall_seconds=10, save_seconds=1, clock=clock)
    finally:
        hook.remove()
    assert progress["tokens"] == 48
    assert progress["elapsed_seconds"] < 10
    assert len(progress["pending_ids"]) == 16
    assert trainer.metadata["training"]["stop_reason"] == "wall time reserve"
    assert len(times) >= 3 and max(b - a for a, b in zip(times, times[1:])) <= 2
    _, saved = checkpoint_record(tmp_path / "checkpoints")
    assert saved["progress"]["pending_ids"] == progress["pending_ids"]


@pytest.mark.parametrize("arguments", [
    ["--latents", "1000000"], ["--microbatch-tokens", "8192"], ["--batch-tokens", "8192"],
    ["--context", "1024"], ["--save-seconds", "301"], ["--max-wall-seconds", "36001"],
])
def test_cli_refuses_unbounded_work_before_model_load(tmp_path, monkeypatch, arguments):
    from tools.interp import train_transcoders
    monkeypatch.setattr(train_transcoders, "load_text_model", lambda **_: pytest.fail("model load ran"))
    with pytest.raises(ValueError):
        train_transcoders.main(["--output", str(tmp_path), *arguments])


def test_exhausted_resume_refuses_model_load(tmp_path, monkeypatch):
    from tools.interp import train_transcoders
    _, trainer, stream = small_trainer()
    progress = initial_progress()
    progress["elapsed_seconds"] = 36_000
    save_checkpoint(tmp_path / "checkpoints", trainer, stream, progress)
    monkeypatch.setattr(train_transcoders, "load_text_model", lambda **_: pytest.fail("model load ran"))
    with pytest.raises(ValueError, match="reserve"):
        train_transcoders.main(["--output", str(tmp_path), "--resume"])


@pytest.mark.parametrize("delay,stop_reason", [(1, None), (270, None), (1, "synthetic resource stop")])
def test_checkpoint_deadline_starts_after_complete_save(tmp_path, monkeypatch, capsys, delay, stop_reason):
    from caliper._interpret_protocol import bounded
    from tools.interp import train_transcoders as training
    from .test_dictionary_startup import startup
    import json

    model, _, _ = small_trainer()
    startup(monkeypatch, training, model, "no_stop")

    class Clock:
        value = 0.0
        def __call__(self):
            return self.value

    clock = Clock()
    save, online = training.save_checkpoint, training.train_online

    def delayed_save(*args, **kwargs):
        result = save(*args, **kwargs)
        clock.value += 300  # Artificial duration, with no sleep or timing measurement.
        return result

    def step(self, index, x, y, tokens_before, *, health):
        clock.value += delay
        health()
        if stop_reason:
            raise CleanStop(stop_reason)
        return {"fvu": 0.0, "gradient_norm": 0.0}

    def train(*args, **kwargs):
        kwargs.update(started=0, clock=clock)
        return online(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", delayed_save)
    monkeypatch.setattr(training.Trainer, "step_layer", step)
    monkeypatch.setattr(training, "train_online", train)
    # Supply deterministic CPU tokens without a dataset iterator or optimizer step.
    from tools.interp.text_stream import TokenBatch
    monkeypatch.setattr(TextStream, "next_tokens", lambda self, count, **k: TokenBatch(
        torch.ones(count, dtype=torch.int64), torch.zeros(count, dtype=torch.int64),
        torch.arange(count)))
    code = training.main(["--output", str(tmp_path), "--device", "cpu", "--latents", "16", "--k", "2",
                          "--tokens", "8", "--context", "4", "--batch-tokens", "8"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = events[-1]
    assert result["stage"] == "train_end" and bounded({"result": result})
    _, saved = checkpoint_record(tmp_path / "checkpoints")
    assert all(event["save_seconds"] == 300 for event in events if event["stage"] == "checkpoint")
    if delay == 270 or stop_reason:
        assert code == 2 and result["status"] == "interrupted"
        assert result["reason"] == (stop_reason or "checkpoint deadline") and result["tokens"] == 0
        assert saved["progress"]["pending_ids"] == [1] * 8
        assert saved["progress"]["next_layer"] == 0
    else:
        assert code == 0 and result["status"] == "complete"
        assert result["tokens"] == saved["progress"]["tokens"] == 8
        assert saved["progress"]["pending_ids"] is None
