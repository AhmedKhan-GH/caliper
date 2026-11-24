"""Injected CPU startup stops preserve complete artifacts and close sources."""
import json
import signal
from types import SimpleNamespace

import pytest

from caliper._interpret_protocol import bounded
from tools.interp import eval_dictionary as evaluation, train_transcoders as training
from tools.interp.memory_limits import CleanStop, StopControl
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_upper_half_cli import synthetic_source, arguments


@pytest.fixture(autouse=True)
def cpu_guard(monkeypatch):
    from caliper.interp import _dictionary
    from tools.interp.memory_limits import MemoryGuard
    monkeypatch.setattr(_dictionary, "MemoryGuard", lambda limit, device: MemoryGuard(
        limit, device, resident=lambda: 0, footprint=lambda: 0, peak=lambda: 0, pressure=lambda: 50,
        driver=lambda: 0))


def snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def startup(monkeypatch, command, model, target, *, signal_only=False):
    """Inject one guard failure or signal flag without OS signals or pressure."""
    seen, streams, sources = [], [], []
    control = StopControl()
    reason = "synthetic stop " + "é" * 5000

    def stage(name):
        seen.append(name)
        if target == name:
            if signal_only:
                control._signal(signal.SIGTERM, None)
            else:
                raise CleanStop(reason)

    def wrap(name, function):
        def call(*args, **kwargs):
            stage(name)
            return function(*args, **kwargs)
        return call

    def guard(limit, device):
        assert str(device) == "cpu"
        stage("guard")
        return SimpleNamespace(peak=0, limit=int(limit * 1e9), device=device,
                               check=lambda name, **_: stage(name))

    class Source(list):
        closed = False
        def close(self):
            self.closed = True

    def source(**_):
        stage("source")
        value = Source()
        sources.append(value)
        return value

    class Stream(TextStream):
        def __init__(self, *args, **kwargs):
            stage("stream")
            super().__init__(*args, **kwargs)
            streams.append(self)

    def load(*, device, memory, mode="inference"):
        assert device == memory.device == "cpu"
        assert mode == ("training" if command is training else "inference")
        return model, TinyTokenizer()

    monkeypatch.setattr(command, "StopControl", lambda: control)
    monkeypatch.setattr(command, "MemoryGuard", guard)
    monkeypatch.setattr(command, "load_text_model", wrap("model", load))
    monkeypatch.setattr(command, "fineweb_documents", source)
    monkeypatch.setattr(command, "TextStream", Stream)
    return stage, wrap, seen, streams, sources


def assert_clean_stop(capsys):
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    result = json.loads(captured.out.splitlines()[-1])
    assert result["stage"] == "clean_stop"
    assert result["status"] == "interrupted"
    assert len(result["reason"].encode()) <= 1024
    assert bounded({"result": result})


@pytest.mark.parametrize("target", [
    "checkpoint", "guard", "start", "model", "manifest", "trainer",
    "after_dictionary_init", "source", "stream", "restore", "after_resume",
])
def test_trainer_startup_stop_preserves_checkpoint(tmp_path, monkeypatch, capsys, target):
    model, _, _ = synthetic_source(tmp_path)
    before = snapshot(tmp_path)
    stage, wrap, _, streams, sources = startup(monkeypatch, training, model, target)
    for name, attribute in (("checkpoint", "checkpoint_record"), ("manifest", "make_manifest"),
                            ("trainer", "Trainer")):
        monkeypatch.setattr(training, attribute, wrap(name, getattr(training, attribute)))

    def restore(root, trainer, stream):
        # A stop after an in-memory mutation must not save a new checkpoint.
        trainer.metadata["training"]["tokens"] = 1
        stage("restore")
        return training.initial_progress()

    monkeypatch.setattr(training, "restore_checkpoint", restore)
    monkeypatch.setattr(training, "train_online", lambda *a, **k: pytest.fail("batch loop ran"))
    assert training.main(["--output", str(tmp_path), "--resume", "--device", "cpu",
                          "--latents", "8", "--k", "2", "--tokens", "16",
                          "--context", "4", "--batch-tokens", "8", "--microbatch-tokens", "4"]) == 2
    assert snapshot(tmp_path) == before
    assert all(stream.closed for stream in streams)
    assert all(source.closed for source in sources)
    assert_clean_stop(capsys)


@pytest.mark.parametrize("target", ["source", "stream", "admission", "model", "warm"])
def test_fallback_startup_stop_leaves_output_absent(tmp_path, monkeypatch, capsys, target):
    source, output = tmp_path / "source", tmp_path / "fallback"
    model, _, _ = synthetic_source(source)
    before = snapshot(source)
    _, wrap, _, streams, sources = startup(monkeypatch, training, model, target)
    monkeypatch.setattr(training, "prepare_upper_half", wrap("admission", training.prepare_upper_half))
    monkeypatch.setattr(training.Trainer, "initialize_upper", wrap("warm", training.Trainer.initialize_upper))
    monkeypatch.setattr(training, "train_online", lambda *a, **k: pytest.fail("batch loop ran"))
    assert training.main(arguments(source, output)) == 2
    assert not output.exists()
    assert snapshot(source) == before
    assert all(stream.closed for stream in streams)
    assert all(source.closed for source in sources)
    assert_clean_stop(capsys)


@pytest.mark.parametrize("target", ["guard", "start", "model", "dictionary", "after_dictionary_load",
                                   "source", "stream", "fingerprint", "binding"])
def test_evaluator_stop_before_artifact_admission_preserves_prior_report(tmp_path, monkeypatch, capsys, target):
    model, _, _ = synthetic_source(tmp_path)
    before = snapshot(tmp_path)
    _, wrap, _, streams, sources = startup(monkeypatch, evaluation, model, target)
    # The command uses real tiny tensors, but no model forward or device admission.
    from caliper.interp import load_dictionary
    dictionary = load_dictionary(tmp_path, model=model)
    monkeypatch.setattr(evaluation, "load_dictionary", wrap("dictionary", lambda *a, **k: dictionary))
    monkeypatch.setattr(evaluation, "dictionary_fingerprint", wrap("fingerprint", evaluation.dictionary_fingerprint))
    monkeypatch.setattr(evaluation, "EvaluationBinding", wrap("binding", evaluation.EvaluationBinding))
    assert evaluation.main(["--dictionary", str(tmp_path), "--resume", "--device", "cpu"]) == 2
    assert snapshot(tmp_path) == before
    assert all(stream.closed for stream in streams)
    assert all(source.closed for source in sources)
    assert_clean_stop(capsys)


def test_signal_after_stream_creation_preserves_prior_report_before_admission(tmp_path, monkeypatch, capsys):
    model, _, _ = synthetic_source(tmp_path)
    before = snapshot(tmp_path)
    _, _, seen, streams, sources = startup(monkeypatch, evaluation, model, "stream", signal_only=True)
    from caliper.interp import load_dictionary
    dictionary = load_dictionary(tmp_path, model=model, device="cpu")
    monkeypatch.setattr(evaluation, "load_dictionary", lambda *a, **k: dictionary)
    # Stream creation does not admit evaluation. The prior complete report remains valid.
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: pytest.fail("evaluation admission ran"))
    assert evaluation.main(["--dictionary", str(tmp_path), "--resume", "--device", "cpu"]) == 2
    assert "stream" in seen and streams and all(stream.closed for stream in streams)
    assert all(source.closed for source in sources)
    assert snapshot(tmp_path) == before
    assert_clean_stop(capsys)


@pytest.mark.parametrize("command,target,forbidden", [
    (training, "model", "manifest"), (training, "trainer", "source"),
    (evaluation, "model", "dictionary"), (evaluation, "dictionary", "source"),
])
def test_signal_flag_stops_before_next_costly_stage(tmp_path, monkeypatch, capsys, command, target, forbidden):
    model, _, _ = synthetic_source(tmp_path)
    before = snapshot(tmp_path)
    _, wrap, seen, _, _ = startup(monkeypatch, command, model, target, signal_only=True)
    if command is training:
        for name, attribute in (("manifest", "make_manifest"), ("trainer", "Trainer")):
            monkeypatch.setattr(command, attribute, wrap(name, getattr(command, attribute)))
        argv = ["--output", str(tmp_path), "--resume", "--latents", "8", "--k", "2", "--tokens", "16",
                "--context", "4", "--batch-tokens", "8", "--microbatch-tokens", "4"]
    else:
        monkeypatch.setattr(command, "load_dictionary", wrap("dictionary", lambda *a, **k: None))
        argv = ["--dictionary", str(tmp_path)]
    assert command.main([*argv, "--device", "cpu"]) == 2
    assert target in seen and forbidden not in seen
    assert snapshot(tmp_path) == before
    assert_clean_stop(capsys)
