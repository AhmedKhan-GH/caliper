"""Controlled CPU wall clocks and file costs without optimizer updates."""
from types import SimpleNamespace

import pytest
import torch

from tools.interp import checkpoint_state, train_transcoders as training
from tools.interp.text_stream import TextStream, TokenBatch
from .fixture import TinyTokenizer
from .test_dictionary import fixture_manifest, write_fixture
from .test_upper_half import fallback_manifest, tiny_model


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def controlled_run(tmp_path, monkeypatch, *, fallback=False, remaining=36000):
    model = tiny_model()
    metadata = fallback_manifest(model) if fallback else fixture_manifest(model)
    if fallback:
        metadata["training"]["fallback"].update(prior_wall_seconds=100,
                                                 aggregate_wall_limit_seconds=150 + remaining)
    trainer = training.Trainer(metadata, device="cpu", context=4, batch_tokens=8, microbatch_tokens=4)
    if fallback:
        write_fixture(tmp_path / "source", model)
        trainer.initialize_upper(tmp_path / "source")
    stream = TextStream([], TinyTokenizer(), "train", source_id="synthetic-text-v1")
    monkeypatch.setattr(stream, "next_tokens", lambda count, **k: TokenBatch(
        torch.ones(count, dtype=torch.int64, device="cpu"), torch.zeros(count, dtype=torch.int64),
        torch.arange(count)))
    monkeypatch.setattr(trainer, "step_layer", lambda *a, **k: {"fvu": 0.0})
    progress = training.initial_progress()
    progress["elapsed_seconds"] = 50 if fallback else 0
    output = tmp_path / "output"
    checkpoint_state.save_checkpoint(output / "checkpoints", trainer, stream, progress)
    return model, trainer, stream, output, progress


@pytest.mark.parametrize("remaining", [61, 64, 120, 239])
def test_short_aggregate_window_retains_admission_floor(tmp_path, monkeypatch, remaining):
    model, trainer, stream, output, progress = controlled_run(tmp_path, monkeypatch, fallback=True, remaining=remaining)
    clock = Clock()
    args = SimpleNamespace(max_wall_seconds=150 + remaining, prior_wall_seconds=100, run_seconds=None)
    monkeypatch.setattr(training.time, "monotonic", clock)
    training._fallback_reserve(args, progress, 0)

    def forward(*_):
        clock.value = remaining - 60

    def step(*args, **kwargs):
        pytest.fail("layer work entered the final save allowance")

    monkeypatch.setattr(trainer, "step_layer", step)
    save, export = training.save_checkpoint, training.export_dictionary
    writes = []

    def save_cost(*args, **kwargs):
        writes.append(("checkpoint", clock()))
        clock.value += 4
        return save(*args, **kwargs)

    def export_cost(*args, **kwargs):
        writes.append(("export", clock()))
        clock.value += 6
        return export(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", save_cost)
    monkeypatch.setattr(training, "export_dictionary", export_cost)
    handle = model.layers[0].register_forward_pre_hook(forward)
    try:
        result = training.train_online(model, trainer, stream, output, token_budget=16, progress=progress,
                                        max_wall_seconds=args.max_wall_seconds, clock=clock)
    finally:
        handle.remove()
    assert writes == [("checkpoint", remaining - 60), ("export", remaining - 56)]
    assert result["tokens"] == 0 and result["pending_ids"] == [1] * 8
    assert trainer.metadata["training"]["stop_reason"] == "wall time reserve"
    assert result["elapsed_seconds"] + 100 == args.max_wall_seconds - 50
    saved = checkpoint_state.checkpoint_record(output / "checkpoints")[1]["progress"]
    assert saved["checkpoint_seconds"] == 4 and saved["export_seconds"] == 6
    assert saved["elapsed_seconds"] == result["elapsed_seconds"]


@pytest.mark.parametrize("checkpoint,export", [(0, 0), (10, 35), (40, 0)])
def test_setup_and_run_cap_cannot_spend_the_reserve(tmp_path, monkeypatch, checkpoint, export):
    model, trainer, stream, output, progress = controlled_run(tmp_path, monkeypatch)
    progress.update(checkpoint_seconds=checkpoint, export_seconds=export)
    reserve = max(60, 2 * (checkpoint + export))
    clock = Clock()
    clock.value = 5
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="reserve"):
        training.train_online(model, trainer, stream, output, token_budget=8, progress=progress,
                              run_seconds=reserve + 5, started=0, clock=clock)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(training.time, "monotonic", clock)
    args = SimpleNamespace(max_wall_seconds=36000, prior_wall_seconds=100, run_seconds=reserve + 5)
    with pytest.raises(ValueError, match="reserve"):
        training._fallback_reserve(args, progress, 0)


def test_observed_export_cost_controls_next_invocation(tmp_path, monkeypatch):
    model, trainer, stream, output, progress = controlled_run(tmp_path, monkeypatch)
    clock = Clock()
    save, export = training.save_checkpoint, training.export_dictionary

    def save_cost(*args, **kwargs):
        clock.value += 10
        return save(*args, **kwargs)

    def export_cost(*args, **kwargs):
        clock.value += 35
        return export(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", save_cost)
    monkeypatch.setattr(training, "export_dictionary", export_cost)
    result = training.train_online(model, trainer, stream, output, token_budget=8, progress=progress, clock=clock)
    assert result["tokens"] == 8 and result["elapsed_seconds"] == 45
    restored = checkpoint_state.checkpoint_record(output / "checkpoints")[1]["progress"]
    assert restored["checkpoint_seconds"] == 10 and restored["export_seconds"] == 35
    assert training._save_reserve(restored) == 90
    with pytest.raises(ValueError, match="reserve"):
        training.train_online(model, trainer, stream, output, token_budget=16, progress=restored,
                              run_seconds=90, clock=clock)


def test_initial_checkpoint_cost_expands_reserve_before_work(tmp_path, monkeypatch):
    model, trainer, stream, _, progress = controlled_run(tmp_path, monkeypatch)
    clock = Clock()
    save = training.save_checkpoint

    def save_cost(*args, **kwargs):
        clock.value += 40
        return save(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", save_cost)
    monkeypatch.setattr(training, "hidden_forward", lambda *a: pytest.fail("forward consumed the save allowance"))
    result = training.train_online(model, trainer, stream, tmp_path / "fresh", token_budget=8,
                                    progress=progress, run_seconds=100, clock=clock)
    assert result["tokens"] == 0 and result["pending_ids"] is None
    assert result["checkpoint_seconds"] == 40 and result["elapsed_seconds"] == 80
    assert trainer.metadata["training"]["stop_reason"] == "wall time reserve"


@pytest.mark.parametrize("key", ["checkpoint_seconds", "export_seconds"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_save_cost_refuses_admission(key, value):
    with pytest.raises(ValueError, match="reserve"):
        training._save_reserve({key: value})


def test_runtime_preserves_largest_save_costs(tmp_path, monkeypatch):
    _, _, _, output, _ = controlled_run(tmp_path, monkeypatch)
    folder, record = checkpoint_state.checkpoint_record(output / "checkpoints")
    record["progress"].update(checkpoint_seconds=25, export_seconds=30)
    checkpoint_state.write_json(folder / "state.json", record)
    checkpoint_state.write_json(output / "runtime.json", {
        "generation": folder.name, "elapsed_seconds": 100, "checkpoint_seconds": 10, "export_seconds": 35,
    })
    progress = checkpoint_state.checkpoint_record(output / "checkpoints")[1]["progress"]
    assert progress["elapsed_seconds"] == 100
    assert progress["checkpoint_seconds"] == 25 and progress["export_seconds"] == 35
    assert training._save_reserve(progress) == 120
