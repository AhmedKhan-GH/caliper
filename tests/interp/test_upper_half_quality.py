"""Small synthetic CPU checks for upper-only quality and evaluation resume."""
import json

import pytest
import torch

from caliper.interp import load_dictionary
from tools.interp import checkpoint_state, eval_dictionary as evaluation, train_transcoders as training
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_dictionary import write_fixture
from .test_training_stream import documents
from .test_upper_half import fallback_manifest, tiny_model, upper_trainer


def dictionary_fixture(folder):
    model = tiny_model()
    write_fixture(folder / "source", model)
    trainer = upper_trainer(fallback_manifest(model))
    trainer.initialize_upper(folder / "source")
    checkpoint_state.export_dictionary(folder / "dictionary", trainer, training.initial_progress())
    return model, load_dictionary(folder / "dictionary", model=model), folder / "dictionary"


def quality_stream():
    return TextStream(documents(), TinyTokenizer(), "quality", source_id="synthetic-text-v1")


def test_lower_mlps_stay_original_and_upper_replacements_restore(tmp_path, monkeypatch):
    model, dictionary, _ = dictionary_fixture(tmp_path)
    original = [block.mlp for block in model.layers]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with evaluation.replace_all_mlps(model, dictionary):
            assert [block.mlp for block in model.layers[:2]] == original[:2]
            assert [block.mlp for block in model.layers[2:]] == list(dictionary.layers[2:])
            raise RuntimeError("synthetic failure")
    assert [block.mlp for block in model.layers] == original
    forward = evaluation.hidden_forward
    calls = 0

    def fail_second(model, ids):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert [block.mlp for block in model.layers[:2]] == original[:2]
            raise RuntimeError("synthetic forward failure")
        return forward(model, ids)

    monkeypatch.setattr(evaluation, "hidden_forward", fail_second)
    with pytest.raises(RuntimeError, match="forward failure"):
        evaluation.evaluate_batch(model, dictionary, torch.tensor([[1, 2, 3, 4]]), 4, chunk_tokens=2)
    assert [block.mlp for block in model.layers] == original
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_upper_quality_ignores_zero_lower_modules(tmp_path, monkeypatch):
    model, dictionary, _ = dictionary_fixture(tmp_path)
    capture = evaluation.capture

    def upper_capture(model, requested):
        assert requested == [dictionary.metadata[field][i] for field in ("read_points", "write_points") for i in (2, 3)]
        return capture(model, requested)

    monkeypatch.setattr(evaluation, "capture", upper_capture)
    stats = evaluation.evaluate_batch(model, dictionary, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
                                       8, chunk_tokens=2)
    metrics = stats.metrics()
    assert metrics["fvu"] == metrics["cross_entropy_delta"] == 0
    assert metrics["scope"] == "upper_half" and metrics["active_layers"] == [2, 3]
    assert metrics["status"] == "fallback_partial"
    assert metrics["per_layer_fvu"] == [None, None, 0.0, 0.0]
    assert metrics["per_layer_dead_fraction"][:2] == [None, None]
    assert metrics["per_layer_status"] == ["error_only", "error_only", "active", "active"]


@pytest.mark.parametrize("metric,value,status", [
    ("fvu", 0.3, "fallback_passed"), ("fvu", 0.301, "fallback_failed"),
    ("dead", 0.25, "fallback_failed"), ("ce", 0.501, "fallback_failed"),
])
def test_completed_metrics_have_only_fallback_status(metric, value, status):
    # Statistics are artificial. No 200000-token evaluation runs in this test.
    stats = evaluation.QualityStats(4, 4, 8, active_layers=[2, 3])
    stats.count, stats.ce_tokens = 200_000, 100_000
    stats.variance_sum[2:] = 10
    stats.error_sum[2:] = 3
    stats.hits[2:] = 1
    stats.hits[2:, 0] = 0
    stats.original_ce = 100_000
    stats.replaced_ce = 150_000
    # Lower entries must not affect active metrics or variance admission.
    stats.error_sum[:2] = float("nan")
    if metric == "fvu":
        stats.error_sum[2:] = value * 10
    elif metric == "dead":
        stats.hits[2:, 1] = 0
    else:
        stats.replaced_ce = stats.original_ce + value * stats.ce_tokens
    result = stats.metrics()
    assert result["status"] == status
    assert result["per_layer_fvu"][:2] == [None, None]
    assert result["per_layer_dead_fraction"][:2] == [None, None]
    assert result["dead_fraction"] == (0.25 if metric == "dead" else 0.125)
    stats.count -= 1
    assert stats.metrics()["status"] == "fallback_partial"


def test_upper_evaluation_resume_preserves_metrics_and_scope(tmp_path, monkeypatch):
    model, dictionary, folder = dictionary_fixture(tmp_path / "full")
    expected = evaluation.evaluate(model, dictionary, quality_stream(), folder,
                                    tokens=16, context=4, batch_tokens=8, chunk_tokens=2)
    model, dictionary, folder = dictionary_fixture(tmp_path / "partial")
    implementation = evaluation.evaluate_batch
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = implementation(*args, **kwargs)
        if calls == 2:
            raise CleanStop("synthetic interruption")
        return result

    monkeypatch.setattr(evaluation, "evaluate_batch", interrupt)
    partial = evaluation.evaluate(model, dictionary, quality_stream(), folder,
                                   tokens=16, context=4, batch_tokens=8, chunk_tokens=2)
    assert partial["evaluated_tokens"] == 8 and partial["status"] == "fallback_partial"
    state = json.loads((folder / "evaluation" / "state.json").read_text())
    assert state["settings"]["scope"] == "upper_half"
    assert state["settings"]["active_layers"] == [2, 3]
    assert len(state["pending_ids"]) == 8
    monkeypatch.setattr(evaluation, "evaluate_batch", implementation)
    # Reload the updated manifest to prove that mutable quality does not alter identity.
    dictionary = load_dictionary(folder, model=model)
    resumed = evaluation.evaluate(model, dictionary, quality_stream(), folder,
                                   tokens=16, context=4, batch_tokens=8, chunk_tokens=2, resume=True)
    for key in ("fvu", "dead_fraction", "cross_entropy_delta", "evaluated_tokens", "cross_entropy_tokens",
                "scope", "active_layers", "status", "per_layer_fvu", "per_layer_dead_fraction"):
        assert resumed[key] == expected[key]
    assert resumed["replaced_mlps"] == 2 and "all_mlps_replaced" not in resumed
    assert json.loads((folder / "dictionary.json").read_text())["quality"] == resumed


@pytest.mark.parametrize("change", ["active_layers", "scope", "all_layer_manifest", "dictionary_weights"])
def test_evaluation_resume_refuses_changed_scope_or_identity(tmp_path, monkeypatch, change):
    model, dictionary, folder = dictionary_fixture(tmp_path)
    evaluation.evaluate(model, dictionary, quality_stream(), folder,
                        tokens=8, context=4, batch_tokens=8, chunk_tokens=2)
    state_path = folder / "evaluation" / "state.json"
    state = json.loads(state_path.read_text())
    if change in ("active_layers", "scope"):
        state["settings"][change] = [3] if change == "active_layers" else "all_layers"
        checkpoint_state.write_json(state_path, state)
    elif change == "all_layer_manifest":
        del dictionary.metadata["training"]["fallback"]
    else:
        from safetensors.torch import load_file, save_file
        values = load_file(folder / "layer_03.safetensors")
        values["b_dec"][0] += 1
        save_file(values, folder / "layer_03.safetensors")
    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *_a, **_k: pytest.fail("evaluation ran after scope change"))
    with pytest.raises(ValueError, match="dictionary or settings changed"):
        evaluation.evaluate(model, dictionary, quality_stream(), folder,
                            tokens=8, context=4, batch_tokens=8, chunk_tokens=2, resume=True)


def test_interruption_before_first_batch_is_fallback_partial(tmp_path):
    model, dictionary, folder = dictionary_fixture(tmp_path)

    class Stop:
        def check(self):
            raise CleanStop("synthetic stop")

    result = evaluation.evaluate(model, dictionary, quality_stream(), folder, tokens=8, context=4,
                                 batch_tokens=8, chunk_tokens=2, stop=Stop())
    assert result["status"] == "fallback_partial" and result["evaluated_tokens"] == 0
    assert result["scope"] == "upper_half" and result["active_layers"] == [2, 3]
    assert result["fvu"] is result["dead_fraction"] is result["cross_entropy_delta"] is None
    assert result["per_layer_fvu"] == [None] * 4
    assert load_dictionary(folder, model=model).metadata["quality"] == result
