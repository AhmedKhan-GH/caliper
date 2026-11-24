"""Pressure stops preserve checkpoints and partial diagnostics retain their identity."""
import json
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import load_dictionary
from tools.interp import checkpoint_state, eval_dictionary as evaluation, train_transcoders as training
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_durability_reserve import controlled_run


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stop_at", ["after_forward", "mid_batch", "before_dictionary_export"])
def test_pressure_stop_skips_export_and_preserves_prior_artifact(tmp_path, monkeypatch, capsys, existing, stop_at):
    model, trainer, stream, output, progress = controlled_run(tmp_path, monkeypatch)
    if existing:
        checkpoint_state.export_dictionary(output, trainer, progress)
        trainer.metadata["quality"].update(status="partial", fvu=0.1, evaluated_tokens=4)
        checkpoint_state.write_json(output / "dictionary.json", trainer.metadata)
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    completed = []

    def step(index, *args, **kwargs):
        # A controlled parameter change represents one completed layer update.
        with torch.no_grad():
            trainer.layers[index].b_dec.add_(1)
        completed.append(index)
        return {"fvu": 0.0}

    class Guard:
        peak = 0

        def check(self, stage, *, force=False):
            if stage == "before_dictionary_export":
                assert force
            if (stop_at == "mid_batch" and stage == "train" and completed) or stage == stop_at:
                raise CleanStop("synthetic memory pressure")

    monkeypatch.setattr(trainer, "step_layer", step)
    monkeypatch.setattr(training, "export_dictionary", lambda *a: pytest.fail("export ran after memory refusal"))
    result = training.train_online(model, trainer, stream, output, token_budget=8,
                                    progress=progress, memory=Guard(), clock=lambda: 0)
    assert result["dictionary_export"] == "skipped"
    assert all((output / name).read_bytes() == data for name, data in before.items())
    assert (output / "dictionary.json").exists() == existing
    assert len(list(output.glob("layer_*.safetensors"))) == (4 if existing else 0)
    _, saved = checkpoint_state.checkpoint_record(output / "checkpoints")
    expected_count = 8 if stop_at == "before_dictionary_export" else 0
    assert saved["progress"]["tokens"] == expected_count
    assert saved["progress"]["next_layer"] == (1 if stop_at == "mid_batch" else 0)
    assert saved["progress"]["pending_ids"] == (None if expected_count else [1] * 8)
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["dictionary_export"] == "skipped" and event["export_stop_reason"] == "synthetic memory pressure"
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["dictionary_export"] == "skipped" and runtime["export_stop_reason"] == event["export_stop_reason"]
    # Recovery uses the complete checkpoint without a new optimizer update.
    restored = checkpoint_state.restore_checkpoint(output / "checkpoints", trainer, stream)
    assert restored["tokens"] == expected_count
    checkpoint_state.export_dictionary(output, trainer, restored)
    recovered = load_dictionary(output, model=model, device="cpu")
    for actual, expected in zip(recovered.layers, trainer.layers):
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cli_reports_incomplete_export_after_full_token_count(tmp_path, monkeypatch):
    from .test_dictionary_startup import startup
    from .test_upper_half import tiny_model

    startup(monkeypatch, training, tiny_model(), "no_stop")
    monkeypatch.setattr(training, "train_online", lambda *a, **k: {"tokens": 8, "dictionary_export": "skipped"})
    assert training.main(["--output", str(tmp_path), "--device", "cpu", "--tokens", "8", "--latents", "8", "--k", "2"]) == 2


def test_mid_batch_diagnostic_binds_exact_artifact_without_claiming_completed_training(tmp_path, monkeypatch):
    model, trainer, stream, output, progress = controlled_run(tmp_path, monkeypatch)
    completed = []

    def step(index, *args, **kwargs):
        with torch.no_grad():
            trainer.layers[index].b_dec.add_(1)
        completed.append(index)
        return {"fvu": 0.0}

    def stop():
        if completed:
            raise CleanStop("synthetic operator stop")

    monkeypatch.setattr(trainer, "step_layer", step)
    result = training.train_online(model, trainer, stream, output, token_budget=8, progress=progress,
                                    stop=SimpleNamespace(check=stop), clock=lambda: 0)
    assert result["dictionary_export"] == "complete" and result["next_layer"] == 1
    dictionary = load_dictionary(output, model=model, device="cpu")
    identity = checkpoint_state.dictionary_fingerprint(output, dictionary.metadata)
    before = dict(dictionary.metadata["training"])
    assert before["tokens"] == 0 and before["token_budget"] == 8
    assert before["partial_batch_tokens"] == 8 and before["next_layer"] == 1
    assert before["stop_reason"] == "synthetic operator stop"
    for actual, expected in zip(dictionary.layers, trainer.layers):
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Artificial complete statistics test the report contract, not dictionary quality.
    stats = evaluation.quality_stats(dictionary.metadata)
    stats.count = 200_000
    stats.ce_tokens = 100_000
    stats.variance_sum.fill_(1)
    stats.hits.fill_(1)
    source = TextStream([], TinyTokenizer(), "quality", source_id="synthetic-text-v1")
    monkeypatch.setattr(source, "next_tokens", stream.next_tokens)
    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *a, **k: stats)
    quality = evaluation.evaluate(model, dictionary, source, output, context=4, batch_tokens=8)
    assert quality["status"] == "passed" and quality["stop_reason"] == "complete"
    assert quality["dictionary_hash"] == identity
    assert dictionary.metadata["training"] == before
    saved = json.loads((output / "dictionary.json").read_text())
    assert saved["training"] == before and saved["quality"] == quality
