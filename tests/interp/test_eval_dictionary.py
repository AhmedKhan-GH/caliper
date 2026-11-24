"""Exact CPU quality, full MLP replacement, and interrupted evaluation."""
import json

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import load_dictionary
from tools.interp import eval_dictionary as evaluation
from tools.interp.memory_limits import CleanStop
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_training_stream import documents


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
            assert all(block.mlp is dictionary.layers[i] for i, block in enumerate(model.layers))
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
        evaluation.main(["--dictionary", str(tmp_path), *arguments])
