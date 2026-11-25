"""Exact CPU quality, full MLP replacement, and interrupted evaluation."""
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import load_dictionary
from tools.interp import eval_dictionary as evaluation
from tools.interp.memory_limits import CleanStop
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import memory_readers, write_fixture
from .test_training_stream import documents


@pytest.mark.parametrize("fail_sync", [False, True])
def test_evaluation_sync_precedes_binding_recheck_and_cursor(tmp_path, monkeypatch, fail_sync):
    from tools.interp import checkpoint_state
    from tools.interp.evaluation_binding import EvaluationBinding

    model = TinyLM(layers=2, width=4, heads=2, latents=8, k=2)
    _, metadata = write_fixture(tmp_path, model)
    dictionary = load_dictionary(tmp_path, model=model, device="cpu")
    identity = checkpoint_state.dictionary_fingerprint(tmp_path, metadata)
    binding = EvaluationBinding(tmp_path, dictionary, identity, model=model)
    stream = TextStream(documents(), TinyTokenizer(), "quality")
    stats = evaluation.quality_stats(metadata)
    settings = {"tokens": 4, "batch_tokens": 2}
    folder = tmp_path / "evaluation"
    evaluation.save_evaluation(folder, stats, stream, None, identity, settings, 0, binding)
    previous = {p.name: p.read_bytes() for p in folder.iterdir()}
    pending = stream.next_tokens(4).input_ids[2:].tolist()
    events, handles = [], {}
    opened, sync = Path.open, checkpoint_state.os.fsync
    save, verify, write = checkpoint_state.save_file, binding.verify, evaluation.write_json

    def track(path, *args, **kwargs):
        handle = opened(path, *args, **kwargs)
        handles[handle.fileno()] = path
        return handle

    def save_tensors(*args):
        events.append("write")
        return save(*args)

    def sync_tensors(descriptor):
        if handles[descriptor].suffix == ".safetensors":
            events.append("sync")
            if fail_sync:
                raise OSError("synthetic statistics sync failure")
        sync(descriptor)

    def recheck():
        events.append("verify")
        verify()

    def cursor(*args):
        events.append("cursor")
        return write(*args)

    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(checkpoint_state, "save_file", save_tensors)
    monkeypatch.setattr(checkpoint_state.os, "fsync", sync_tensors)
    monkeypatch.setattr(binding, "verify", recheck)
    monkeypatch.setattr(evaluation, "write_json", cursor)
    stats.count = 2
    stats.mean.fill_(3)
    if fail_sync:
        with pytest.raises(OSError, match="statistics sync failure"):
            evaluation.save_evaluation(folder, stats, stream, pending, identity, settings, 1, binding)
        assert events == ["verify", "write", "sync"]
        assert all((folder / name).read_bytes() == value for name, value in previous.items())
        restored = evaluation.quality_stats(metadata)
        assert evaluation.restore_evaluation(folder, restored, stream, identity, settings, binding) == (None, 0)
        assert restored.count == 0 and torch.equal(restored.mean, torch.zeros_like(restored.mean))
        assert stream.next_tokens(4).input_ids[2:].tolist() == pending
        fail_sync = False
        events.clear()
    evaluation.save_evaluation(folder, stats, stream, pending, identity, settings, 1, binding)
    assert events == ["verify", "write", "sync", "verify", "cursor"]
    assert json.loads((folder / "state.json").read_text())["tensors"] == "stats-b.safetensors"
    restored = evaluation.quality_stats(metadata)
    assert evaluation.restore_evaluation(folder, restored, stream, identity, settings, binding) == (pending, 1)
    assert restored.count == 2 and torch.equal(restored.mean, stats.mean)
    stats.count = 4
    evaluation.save_evaluation(folder, stats, stream, None, identity, settings, 2, binding)
    assert json.loads((folder / "state.json").read_text())["tensors"] == "stats-a.safetensors"


@pytest.mark.parametrize("metric,value,status", [
    ("fvu", 0.40, "passed"), ("fvu", 0.401, "failed"),
    ("dead", 0.15, "passed"), ("dead", 0.20, "failed"),
    ("ce", 0.50, "passed"), ("ce", 0.501, "failed"),
])
def test_completed_quality_thresholds(metric, value, status):
    stats = evaluation.QualityStats(1, 2, 20)
    stats.count, stats.ce_tokens = 200_000, 100_000
    stats.variance_sum.fill_(10)
    stats.error_sum.fill_(4)
    stats.hits.fill_(1)
    stats.hits[:, :3] = 0
    stats.original_ce, stats.replaced_ce = 100_000, 150_000
    if metric == "fvu":
        stats.error_sum.fill_(value * 10)
    elif metric == "dead":
        stats.hits[:, :round(value * 20)] = 0
    else:
        stats.replaced_ce = stats.original_ce + value * stats.ce_tokens
    result = stats.metrics()
    assert result["status"] == status
    assert result["thresholds"] == {"fvu": .40, "dead_fraction": .15,
                                    "cross_entropy_delta": .5, "evaluated_tokens": 200_000}


@pytest.mark.parametrize("active_layers", [None, [1, 2]])
def test_zero_variance_cannot_pass_at_token_budget(active_layers):
    stats = evaluation.QualityStats(3, 2, 2, active_layers=active_layers)
    stats.count, stats.ce_tokens = 200_000, 100_000
    stats.variance_sum[:] = torch.tensor([10, 0, 10])
    stats.error_sum[:] = torch.tensor([1, 0, 2])
    stats.hits.fill_(1)
    stats.original_ce = stats.replaced_ce = 100_000
    quality = stats.metrics()
    assert quality["status"] == ("partial" if active_layers is None else "fallback_partial")
    assert quality["fvu"] is None
    assert quality["per_layer_fvu"] == ([.1, None, .2] if active_layers is None else [None, None, .2])
    assert quality["dead_fraction"] == quality["cross_entropy_delta"] == 0
    assert quality["original_cross_entropy"] == quality["replaced_cross_entropy"] == 1


def test_zero_variance_stop_saves_partial_cursor_and_resumes(tmp_path, capsys):
    from safetensors.torch import load_file
    from tools.interp.checkpoint_state import dictionary_fingerprint, write_json

    class Tokenizer(TinyTokenizer):
        def encode(self, *args, **kwargs):
            return [1, 1, 2, 3]

    model = TinyLM(layers=2, width=4, heads=2, latents=8, k=2)
    for name in ("full", "partial"):
        write_fixture(tmp_path / name, model)

    def run(name, *, interrupt=False, **kwargs):
        class Stream(TextStream):
            def next_tokens(self, *args, **kwargs):
                if interrupt and self.emitted == 2:
                    raise CleanStop("zero variance batch complete")
                return super().next_tokens(*args, **kwargs)

        folder = tmp_path / name
        dictionary = load_dictionary(folder, model=model)
        stream = Stream(documents(), Tokenizer(), "quality")
        return evaluation.evaluate(model, dictionary, stream, folder, tokens=4,
                                   context=2, batch_tokens=2, chunk_tokens=2, **kwargs)

    expected = run("full")
    folder = tmp_path / "partial"
    metadata = json.loads((folder / "dictionary.json").read_text())
    identity = dictionary_fingerprint(folder, metadata)
    metadata["quality"].update(status="passed", fvu=.1, dead_fraction=.1,
                               cross_entropy_delta=.1, evaluated_tokens=200_000)
    write_json(folder / "dictionary.json", metadata)
    partial = run("partial", interrupt=True)
    assert partial["status"] == "partial" and partial["evaluated_tokens"] == 2
    assert partial["fvu"] is None and partial["per_layer_fvu"] == [None, None]
    assert partial["cross_entropy_tokens"] == 1 and partial["cross_entropy_delta"] == 0
    assert partial["dead_fraction"] is not None
    assert partial["dictionary_hash"] == identity
    assert partial["stop_reason"] == "zero variance batch complete"
    assert json.loads((folder / "dictionary.json").read_text())["quality"] == partial
    end = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert end == {"stage": "quality_end", **partial}
    cursor = json.loads((folder / "evaluation" / "state.json").read_text())
    assert cursor["count"] == cursor["stream"]["emitted"] == 2
    assert cursor["ce_tokens"] == 1 and cursor["pending_ids"] is None
    tensors = load_file(folder / "evaluation" / cursor["tensors"])
    assert torch.equal(tensors["variance_sum"], torch.zeros(2, dtype=torch.float64))
    resumed = run("partial", resume=True)
    for key in ("fvu", "dead_fraction", "cross_entropy_delta", "evaluated_tokens", "cross_entropy_tokens"):
        assert resumed[key] == expected[key]
    assert dictionary_fingerprint(folder, metadata) == identity


def test_chunked_cross_entropy_matches_full_logits():
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    hidden = hidden_forward(model, ids)
    expected = F.cross_entropy(model.lm_head(hidden)[0, :-1].float(), ids[0, 1:], reduction="sum")
    expected += F.cross_entropy(model.lm_head(hidden)[1, :1].float(), ids[1, 1:2], reduction="sum")
    value, count = evaluation.cross_entropy_sum(model, hidden, ids, 6, chunk_tokens=2)
    assert count == 4
    assert value == pytest.approx(expected.item(), abs=3e-6)


def test_every_mlp_replaced_and_restored_on_exception(tmp_path):
    model, _ = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    original = [block.mlp for block in model.layers]
    with pytest.raises(RuntimeError, match="fixture"):
        with evaluation.replace_all_mlps(model, dictionary):
            assert all(block.mlp.layer is dictionary.layers[i] for i, block in enumerate(model.layers))
            raise RuntimeError("fixture")
    assert all(block.mlp is original[i] for i, block in enumerate(model.layers))


def test_exact_fixture_quality_and_stable_variance(tmp_path):
    model, _ = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    stats = evaluation.evaluate_batch(model, dictionary, ids, 8, chunk_tokens=3)
    metrics = stats.metrics()
    assert metrics["fvu"] < 1e-28
    assert metrics["cross_entropy_delta"] == 0
    assert metrics["evaluated_tokens"] == 8
    assert metrics["cross_entropy_tokens"] == 6
    assert metrics["status"] == "partial"
    assert len(metrics["per_layer_fvu"]) == 4
    first, second = evaluation.QualityStats(1, 2, 2), evaluation.QualityStats(1, 2, 2)
    first.count, second.count = 2, 3
    first.mean[:] = torch.tensor([1, 2])
    second.mean[:] = torch.tensor([4, 6])
    first.merge(second)
    targets = torch.tensor([[1, 2], [1, 2], [4, 6], [4, 6], [4, 6]], dtype=torch.float64)
    torch.testing.assert_close(first.mean[0], targets.mean(0))
    torch.testing.assert_close(first.variance_sum[0], (targets - targets.mean(0)).square().sum())


def test_evaluation_resume_keeps_pending_partition_batch(tmp_path, monkeypatch):
    model = TinyLM(layers=2, width=8, heads=2, latents=16, k=2)
    for name in ("full", "partial"):
        write_fixture(tmp_path / name, model)
    dictionary = load_dictionary(tmp_path / "full", model=model)
    expected = evaluation.evaluate(model, dictionary, TextStream(documents(), TinyTokenizer(), "quality"),
                                   tmp_path / "full", tokens=48, context=4, batch_tokens=16, chunk_tokens=4)
    implementation = evaluation.evaluate_batch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        result = implementation(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise CleanStop("fixture interruption")
        return result
    monkeypatch.setattr(evaluation, "evaluate_batch", interrupt)
    dictionary = load_dictionary(tmp_path / "partial", model=model)
    partial = evaluation.evaluate(model, dictionary, TextStream(documents(), TinyTokenizer(), "quality"),
                                  tmp_path / "partial", tokens=48, context=4, batch_tokens=16, chunk_tokens=4)
    assert partial["evaluated_tokens"] == 16
    state = json.loads((tmp_path / "partial" / "evaluation" / "state.json").read_text())
    assert len(state["pending_ids"]) == 16
    monkeypatch.setattr(evaluation, "evaluate_batch", implementation)
    resumed = evaluation.evaluate(model, dictionary, TextStream(documents(), TinyTokenizer(), "quality"),
                                  tmp_path / "partial", tokens=48, context=4, batch_tokens=16,
                                  chunk_tokens=4, resume=True)
    for name in ("evaluated_tokens", "fvu", "dead_fraction", "cross_entropy_delta", "cross_entropy_tokens"):
        assert resumed[name] == expected[name]


@pytest.mark.parametrize("arguments", [["--chunk-tokens", "4096"], ["--memory-gb", "24"],
                                       ["--context", "1024"], ["--tokens", "200001"]])
def test_evaluator_refuses_unbounded_work_before_model_load(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(evaluation, "load_text_model", lambda **_: pytest.fail("model load ran"))
    with pytest.raises(ValueError):
        evaluation.main(["--dictionary", str(tmp_path), "--device", "cpu", *arguments])


@pytest.mark.parametrize("stop_at", ["before_batch", "pending_batch", "one_token", "restore"])
def test_stop_after_artifact_admission_records_partial_quality(tmp_path, monkeypatch, capsys, stop_at):
    from caliper._interpret_protocol import bounded
    from tools.interp.checkpoint_state import dictionary_fingerprint, write_json
    from tools.interp.text_stream import TokenBatch

    model = TinyLM(layers=2, width=4, heads=2, latents=8, k=2)
    _, metadata = write_fixture(tmp_path, model)
    dictionary = load_dictionary(tmp_path, model=model)
    dictionary.metadata["quality"].update(status="passed", fvu=0, dead_fraction=0,
                                           cross_entropy_delta=0, evaluated_tokens=200_000)
    write_json(tmp_path / "dictionary.json", dictionary.metadata)
    identity = dictionary_fingerprint(tmp_path, metadata)
    stream = TextStream([], TinyTokenizer(), "quality")

    class Stop:
        def check(self):
            if stop_at == "before_batch":
                raise CleanStop("synthetic pre-batch stop")

    def next_tokens(*args, **kwargs):
        size = 1 if stop_at == "one_token" else 2
        return TokenBatch(torch.ones(size, dtype=torch.int64), torch.zeros(size), torch.zeros(size))

    def batch(*args, **kwargs):
        if stop_at == "pending_batch":
            raise CleanStop("synthetic pending\0stop\ud800" + "é" * 5000)
        result = evaluation.quality_stats(metadata)
        result.count = 1
        monkeypatch.setattr(stream, "next_tokens", lambda *a, **k: (_ for _ in ()).throw(CleanStop("one token")))
        return result

    monkeypatch.setattr(stream, "next_tokens", next_tokens)
    monkeypatch.setattr(evaluation, "evaluate_batch", batch)
    if stop_at == "restore":
        state = tmp_path / "evaluation" / "state.json"
        state.parent.mkdir()
        state.write_text('{"complete_previous_cursor": true}\n')
        before = state.read_bytes()

        def restore(folder, stats, *args):
            stats.count = 200_000
            raise CleanStop("synthetic partial restore")

        monkeypatch.setattr(evaluation, "restore_evaluation", restore)
    if stop_at == "restore":
        prior_quality = (tmp_path / "dictionary.json").read_bytes()
        with pytest.raises(CleanStop, match="partial restore"):
            evaluation.evaluate(model, dictionary, stream, tmp_path, tokens=4, context=2,
                                batch_tokens=2, stop=Stop(), resume=True)
        assert (tmp_path / "dictionary.json").read_bytes() == prior_quality
        assert state.read_bytes() == before
        assert list(state.parent.iterdir()) == [state]
        assert not capsys.readouterr().out
        return
    result = evaluation.evaluate(model, dictionary, stream, tmp_path, tokens=4, context=2,
                                 batch_tokens=2, stop=Stop())
    assert result["status"] == "partial"
    assert result["evaluated_tokens"] == (1 if stop_at == "one_token" else 0)
    assert result["fvu"] is result["dead_fraction"] is result["cross_entropy_delta"] is None
    assert not {"scope", "active_layers", "per_layer_status"} & result.keys()
    assert result["all_mlps_replaced"] == 2
    assert result["dictionary_hash"] == identity
    assert json.loads((tmp_path / "dictionary.json").read_text())["quality"] == result
    end = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert end == {"stage": "quality_end", **result}
    assert bounded({"result": end})
    if stop_at == "pending_batch":
        state = json.loads((tmp_path / "evaluation" / "state.json").read_text())
        assert state["pending_ids"] == [1, 1] and state["count"] == 0


def test_early_and_observed_all_layer_quality_have_the_same_schema(tmp_path):
    model = TinyLM(layers=2, width=4, heads=2, latents=8, k=2)
    results = []
    for name, source in (("early", []), ("observed", documents())):
        folder = tmp_path / name
        write_fixture(folder, model)
        dictionary = load_dictionary(folder, model=model)
        results.append(evaluation.evaluate(
            model, dictionary, TextStream(source, TinyTokenizer(), "quality"), folder,
            tokens=4, context=4, batch_tokens=4, chunk_tokens=2))
    early, observed = results
    assert early.keys() == observed.keys()
    assert early["status"] == observed["status"] == "partial"
    assert early["evaluated_tokens"] == 0 and observed["evaluated_tokens"] == 4
    assert early["thresholds"] == observed["thresholds"]
    for name in ("original_cross_entropy", "replaced_cross_entropy"):
        assert early[name] is None and observed[name] is not None
    assert not {"scope", "active_layers", "per_layer_status"} & early.keys()
