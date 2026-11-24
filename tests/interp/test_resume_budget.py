"""Resume schedule admission without model loads or optimizer steps."""
import copy
from types import SimpleNamespace

import pytest

from tools.interp import train_transcoders as training
from tools.interp.checkpoint_state import checkpoint_record, write_json
from .test_dictionary_startup import snapshot
from .test_train_transcoders import small_trainer
from .test_upper_half_cli import synthetic_source


@pytest.mark.parametrize("saved,requested", [(16, 8), (16, 32), (None, 16), ("16", 16), (True, 1)])
def test_cli_refuses_changed_or_missing_budget_before_allocations(tmp_path, monkeypatch, saved, requested):
    synthetic_source(tmp_path)
    folder, record = checkpoint_record(tmp_path / "checkpoints")
    if saved is None:
        del record["metadata"]["training"]["token_budget"]
    else:
        record["metadata"]["training"]["token_budget"] = saved
    write_json(folder / "state.json", record)
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("resume crossed budget admission")

    for name in ("MemoryGuard", "load_text_model", "Trainer", "fineweb_documents", "restore_checkpoint", "train_online"):
        monkeypatch.setattr(training, name, forbidden)
    with pytest.raises(ValueError, match="token budget.*learning-rate schedule"):
        training.main(["--output", str(tmp_path), "--resume", "--device", "cpu", "--tokens", str(requested)])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("requested", [20_000_000, 80_000_000])
def test_callable_refuses_budget_change_before_any_write(tmp_path, monkeypatch, requested):
    metadata = {"training": {"token_budget": 40_000_000}}
    trainer = SimpleNamespace(metadata=metadata, fallback_source=None)
    progress = training.initial_progress()
    progress.update(tokens=36_000_000, steps=1, next_layer=1, pending_ids=[1, 2, 3, 4])
    before = copy.deepcopy((metadata, progress))
    monkeypatch.setattr(training, "save_checkpoint", lambda *a, **k: pytest.fail("checkpoint write ran"))
    with pytest.raises(ValueError, match="token budget.*learning-rate schedule"):
        training.train_online(None, trainer, None, tmp_path, token_budget=requested, progress=progress)
    assert (metadata, progress) == before
    assert not list(tmp_path.iterdir())


def test_saved_budget_preserves_decay_at_the_resume_cursor():
    _, trainer, stream = small_trainer()
    try:
        budget, tokens = 40_000_000, 36_000_000
        trainer.metadata["training"]["token_budget"] = budget
        before = copy.deepcopy(trainer.metadata)
        training._validate_token_budget(trainer.metadata, budget, resume=True)
        trainer.set_learning_rate(tokens / budget)
        for optimizer in trainer.optimizers:
            assert not optimizer.state
            assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55 * trainer.settings["learning_rate"])
        assert trainer.metadata == before
    finally:
        stream.close()


def test_matching_cli_budget_reaches_resource_admission(tmp_path, monkeypatch):
    synthetic_source(tmp_path)
    before = snapshot(tmp_path)

    def admitted(*args, **kwargs):
        raise training.CleanStop("synthetic admission stop")

    monkeypatch.setattr(training, "MemoryGuard", admitted)
    monkeypatch.setattr(training, "load_text_model", lambda **k: pytest.fail("model load ran"))
    assert training.main(["--output", str(tmp_path), "--resume", "--device", "cpu", "--tokens", "16",
                          "--latents", "8", "--k", "2", "--context", "4",
                          "--batch-tokens", "8", "--microbatch-tokens", "4"]) == 2
    assert snapshot(tmp_path) == before
