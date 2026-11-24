"""Startup that consumes the save reserve cannot replace complete artifacts."""
import pytest

from tools.interp import checkpoint_state, train_transcoders as training
from .test_dictionary_startup import snapshot, startup
from .test_durability_reserve import Clock
from .test_upper_half import tiny_model
from .test_upper_half_cli import arguments, synthetic_source


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_startup_exhaustion_preserves_generation_and_export(tmp_path, monkeypatch, resume, fallback):
    output = tmp_path / "output"
    if fallback:
        source = tmp_path / "source"
        model, _, source_id = synthetic_source(source)
        argv = [*arguments(source, output), "--max-wall-seconds", "201"]
        args = training.parser().parse_args(argv)
        if resume:
            metadata = training.prepare_upper_half(args, source_id=source_id)
            trainer = training.Trainer(metadata, device="cpu", **training._trainer_arguments(args))
            trainer.initialize_upper(source)
            stream = training.TextStream([], None, "train", source_id=source_id)
            progress = training.initial_progress()
            progress["checkpoint_seconds"] = 5
            checkpoint_state.save_checkpoint(output / "checkpoints", trainer, stream, progress)
            checkpoint_state.export_dictionary(output, trainer, progress)
    else:
        model = tiny_model()
        argv = ["--output", str(output), "--device", "cpu", "--latents", "8", "--k", "2",
                "--context", "4", "--batch-tokens", "8", "--microbatch-tokens", "4", "--tokens", "32"]
        if resume:
            model, _, _ = synthetic_source(output)
            argv[argv.index("--tokens") + 1] = "16"  # Resume keeps the checkpoint learning-rate schedule.
            folder, record = checkpoint_state.checkpoint_record(output / "checkpoints")
            record["progress"].update(elapsed_seconds=35_939, checkpoint_seconds=5)
            checkpoint_state.write_json(folder / "state.json", record)
        else:
            argv += ["--run-seconds", "61"]
    if resume:
        argv += ["--resume"]
    before = snapshot(tmp_path)
    _, _, seen, streams, sources = startup(monkeypatch, training, model, "no_stop")
    clock = Clock()
    load, online = training.load_text_model, training.train_online

    def slow_startup(**kwargs):
        result = load(**kwargs)
        clock.value = 65
        return result

    def checked_online(*args, **kwargs):
        return online(*args, **kwargs, clock=clock)

    def forbidden(*args, **kwargs):
        pytest.fail("work consumed an exhausted startup reserve")

    monkeypatch.setattr(training.time, "monotonic", clock)
    monkeypatch.setattr(training, "load_text_model", slow_startup)
    monkeypatch.setattr(training, "train_online", checked_online)
    monkeypatch.setattr(training, "save_checkpoint", forbidden)
    monkeypatch.setattr(training, "export_dictionary", forbidden)
    monkeypatch.setattr(training, "hidden_forward", forbidden)
    monkeypatch.setattr(training.TextStream, "next_tokens", forbidden)
    with pytest.raises(ValueError, match="reserve"):
        training.main(argv)
    assert "after_resume" in seen
    assert snapshot(tmp_path) == before
    assert output.exists() == resume
    assert streams and all(stream.closed for stream in streams)
    assert sources and all(source.closed for source in sources)
