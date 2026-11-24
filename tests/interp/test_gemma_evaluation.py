"""Gemma semantic replacement, quality gates, and imported source contracts."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tools.interp import build_index, eval_dictionary as evaluation, text_stream
from tools.interp.model_load import GEMMA_ID, GEMMA_REVISION, MODEL_ID, hidden_forward
from .test_gemma_runtime import gemma_model, gemma_semantics, manual_gemma, runtime_dictionary


def test_gemma_replacement_matches_independent_full_forward(gemma_semantics):
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    # Nonunit post norms make an accidental second norm observable.
    for block in model.model.layers:
        block.post_feedforward_layernorm.weight.data.copy_(torch.linspace(-.5, 1, 16))
    ids = torch.tensor([[1, 4, 5, 9, 7, 2], [1, 3, 6, 8, 9, 2]])
    originals = tuple(block.mlp for block in model.model.layers)
    with torch.no_grad():
        expected = manual_gemma(model, ids, dictionary, replace=True)
        before = model(ids, use_cache=False).logits
        with evaluation.replace_all_mlps(model, dictionary):
            actual = model(ids, use_cache=False).logits
            assert tuple(block.mlp for block in model.model.layers) == originals
        restored = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored, before, rtol=0, atol=0)
    assert not torch.allclose(actual, before)
    assert all(not block.mlp._forward_hooks and not block.post_feedforward_layernorm._forward_hooks
               for block in model.model.layers)


def test_gemma_replacement_removes_hooks_on_failure(gemma_semantics, monkeypatch):
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    def fail(*args):
        raise RuntimeError("synthetic decoder failure")
    monkeypatch.setattr(dictionary.layers[0], "decode", fail)
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        with evaluation.replace_all_mlps(model, dictionary), torch.no_grad():
            model(torch.tensor([[1, 4, 5]]), use_cache=False)
    assert all(not block.mlp._forward_hooks and not block.post_feedforward_layernorm._forward_hooks
               for block in model.model.layers)


def test_gemma_chunked_cross_entropy_matches_softcapped_model_logits():
    model = gemma_model()
    model.lm_head.weight.data.mul_(20)
    ids = torch.tensor([[1, 4, 5, 7], [1, 8, 6, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        hidden = hidden_forward(model, ids)
        observed, count = evaluation.cross_entropy_sum(model, hidden, ids, 7, chunk_tokens=2)
    positions = torch.tensor([0, 1, 2, 4, 5])
    loss = F.cross_entropy(expected.reshape(-1, 37)[positions], ids.reshape(-1)[positions + 1], reduction="sum")
    assert count == 5
    assert observed == pytest.approx(loss.item(), rel=1e-6)
    raw = model.lm_head(hidden).reshape(-1, 37)
    uncapped = F.cross_entropy(raw[positions], ids.reshape(-1)[positions + 1], reduction="sum")
    assert abs(observed - uncapped.item()) > .1


@pytest.mark.parametrize("model_id,bound", [(GEMMA_ID, .30), (MODEL_ID, .40)])
@pytest.mark.parametrize("fvu,expected", [(.30, "passed"), (.35, None), (.401, "failed")])
def test_quality_thresholds_keep_gemma_and_qwen_separate(model_id, bound, fvu, expected):
    stats = evaluation.quality_stats({"model_id": model_id, "layers": 1, "width": 2,
                                      "latents": 20, "training": {}})
    stats.count, stats.ce_tokens = 200_000, 199_000
    stats.variance_sum.fill_(1)
    stats.error_sum.fill_(fvu)
    stats.hits.fill_(1)
    stats.replaced_ce = .5 * stats.ce_tokens
    result = stats.metrics()
    assert result["thresholds"] == {"fvu": bound, "dead_fraction": .15,
                                     "cross_entropy_delta": .5, "evaluated_tokens": 200_000}
    assert result["status"] == (expected or ("failed" if model_id == GEMMA_ID else "passed"))
    stats.count = 199_999
    assert stats.metrics()["status"] == "partial"
    stats.count = 200_001
    assert stats.metrics()["status"] == "partial"


@pytest.mark.parametrize("dead,delta,status", [(3, .5, "passed"), (4, .5, "failed"), (3, .501, "failed")])
def test_gemma_quality_retains_dead_feature_and_cross_entropy_gates(dead, delta, status):
    stats = evaluation.quality_stats({"model_id": GEMMA_ID, "layers": 1, "width": 2,
                                      "latents": 20, "training": {}})
    stats.count, stats.ce_tokens = 200_000, 199_000
    stats.variance_sum.fill_(1)
    stats.error_sum.fill_(.3)
    stats.hits.fill_(1)
    stats.hits[0, :dead] = 0
    stats.replaced_ce = delta * stats.ce_tokens
    assert stats.metrics()["status"] == status


def test_gemma_quality_refuses_qwen_upper_half_fallback():
    from .test_upper_half import fallback_manifest, tiny_model
    metadata = fallback_manifest(tiny_model())
    metadata["model_id"] = GEMMA_ID
    with pytest.raises(ValueError, match="Gemma quality requires every layer"):
        evaluation.quality_stats(metadata)


def imported_metadata(source_id=None):
    source = {"dataset_id": text_stream.DATASET_ID, "dataset_config": text_stream.DATASET_CONFIG,
              "dataset_revision": text_stream.DATASET_REVISION, "split": "train",
              "partitions": text_stream.partition_metadata(),
              "source_id": source_id or f"{text_stream.DATASET_ID}/{text_stream.DATASET_CONFIG}@{text_stream.DATASET_REVISION}:train"}
    return {"model_id": GEMMA_ID, "revision": GEMMA_REVISION, "layers": 2, "latents": 20,
            "training": {"origin": "imported", "dataset_id": "unknown", "dataset_config": "unknown",
                         "dataset_revision": "unknown", "split": "unknown", "partitions": {}, "tokens": 0,
                         "local_training": False, "publisher_tokens": None,
                         "publisher_training_disjointness": "unknown"},
            "import_source": {"kind": "gemma_scope", "license": "CC-BY-4.0",
                              "source": {"repo": "google/gemma-scope-2b-pt-transcoders",
                                         "revision": "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"},
                              "original_input": "pure_rms_normalized_pre_mlp_before_learned_gain",
                              "input_conversion": "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]",
                              "output": "after_post_feedforward_layernorm", "output_conversion": "none",
                              "layers": [{"layer": 0, "source_sha256": "a" * 64, "sha256": "b" * 64}]},
            "evaluation_source": source}


def test_imported_source_is_independent_and_has_no_training_disjointness_claim():
    meta = imported_metadata()
    before = deepcopy(meta)
    source = text_stream.evaluation_source_metadata(meta)
    assert source is meta["evaluation_source"] and source["source_id"]
    result = build_index.validate_partitions(meta)
    assert result["publisher_training_overlap"] == "unknown"
    assert result["partitions"] == text_stream.partition_metadata()
    assert meta == before and "source_id" not in meta["training"]


@pytest.mark.parametrize("field,value", [("partitions", {}), ("dataset_revision", "f" * 40), ("source_id", None)])
def test_imported_source_refuses_changed_or_missing_records(field, value):
    meta = imported_metadata()
    meta["evaluation_source"][field] = value
    with pytest.raises(ValueError, match="evaluation source"):
        build_index.validate_partitions(meta)


@pytest.mark.parametrize("field", ["dataset_id", "dataset_config", "dataset_revision", "split",
                                  "source_id", "partitions", "publisher_training_overlap"])
def test_imported_source_requires_exact_canonical_fields(tmp_path, field):
    meta = imported_metadata()
    if field == "publisher_training_overlap":
        meta["evaluation_source"][field] = "disjoint"
    else:
        del meta["evaluation_source"][field]
    with pytest.raises(ValueError, match="evaluation source"):
        build_index.validate_partitions(meta)
    dictionary = SimpleNamespace(metadata=meta)
    stream = SimpleNamespace(partition="quality", source_id="unused")
    with pytest.raises(ValueError, match="evaluation source"):
        evaluation.evaluate(None, dictionary, stream, tmp_path)
    assert not list(tmp_path.iterdir())


def test_qwen_source_checks_still_require_its_training_source():
    meta = imported_metadata()
    qwen = {"model_id": MODEL_ID, "training": deepcopy(meta["evaluation_source"])}
    assert "publisher_training_overlap" not in build_index.validate_partitions(qwen)
    qwen["training"]["partitions"]["index"] = [9400, 10000]
    with pytest.raises(ValueError, match="training dataset"):
        build_index.validate_partitions(qwen)


def test_imported_index_source_requires_same_pinned_evaluation_identity(monkeypatch):
    meta = imported_metadata()
    monkeypatch.setattr(build_index, "fineweb_documents", lambda **kw: [])
    args = SimpleNamespace(partition="index", text_jsonl=None, local_dataset_manifest=None, dataset_cache=None)
    with ExitStack() as resources:
        provenance = build_index.validate_partitions(meta)
        stream = build_index._index_stream(args, meta, provenance, resources)
        assert stream.source_id == meta["evaluation_source"]["source_id"]
        assert provenance["publisher_training_overlap"] == "unknown"
    meta["evaluation_source"]["source_id"] += ":changed"
    with ExitStack() as resources, pytest.raises(ValueError, match="evaluation source"):
        build_index._index_stream(args, meta, {}, resources)


def test_imported_quality_refuses_mismatched_source_before_artifact_read(tmp_path):
    dictionary = SimpleNamespace(metadata=imported_metadata())
    stream = SimpleNamespace(partition="quality", source_id="different")
    with pytest.raises(ValueError, match="evaluation source"):
        evaluation.evaluate(None, dictionary, stream, tmp_path)


def test_index_cli_model_selection_keeps_old_defaults_and_gemma_ceiling():
    parsed = build_index.parser().parse_args(["--dictionary", "unused"])
    assert parsed.model == MODEL_ID and parsed.memory_gb == 12
    parsed = build_index.parser().parse_args(["--dictionary", "unused", "--model", GEMMA_ID])
    assert parsed.memory_gb == 28 and parsed.tokens == 2_000_000
    with pytest.raises(SystemExit):
        build_index.parser().parse_args(["--dictionary", "unused", "--model", GEMMA_ID, "--memory-gb", "29"])


@pytest.mark.parametrize("field", ["import_source", "evaluation_source"])
def test_imported_metadata_requires_both_provenance_records(field):
    metadata = imported_metadata()
    del metadata[field]
    with pytest.raises(ValueError, match="import_source|evaluation source"):
        text_stream.evaluation_source_metadata(metadata)


def test_imported_evaluation_publishes_independent_source_and_resumes(tmp_path, gemma_semantics):
    import json
    from safetensors.torch import save_file
    from caliper.interp import _dictionary
    from .test_training_stream import documents
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    if not hasattr(_dictionary, "JumpReLUTranscoder"):
        # Base-branch binding only supports TopK. Integration uses real JumpReLU.
        dictionary.layers = torch.nn.ModuleList([_dictionary.TopKTranscoder(
            layer.W_enc, layer.b_enc, layer.W_dec, layer.b_dec, 2) for layer in dictionary.layers])
        dictionary.metadata.update(activation="topk", k=2)
    provenance = imported_metadata()
    dictionary.metadata.update({key: provenance[key] for key in ("training", "import_source", "evaluation_source")})
    before = deepcopy(dictionary.metadata["training"])
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    for i, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), tmp_path / f"layer_{i:02d}.safetensors")
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [3 + ord(ch) % 34 for ch in text], eos_token_id=2)
    with text_stream.TextStream(documents(), tokenizer, "quality") as stream:
        quality = evaluation.evaluate(model, dictionary, stream, tmp_path, tokens=8,
                                      context=4, batch_tokens=8, chunk_tokens=2)
    assert quality["status"] == "partial" and quality["evaluated_tokens"] == 8
    assert quality["thresholds"]["fvu"] == .30
    assert quality["evaluation_source"] == provenance["evaluation_source"]
    assert quality["publisher_training_overlap"] == "unknown"
    assert dictionary.metadata["training"] == before
    state = json.loads((tmp_path / "evaluation" / "state.json").read_text())
    assert state["settings"]["evaluation_source"] == provenance["evaluation_source"]
    with text_stream.TextStream(documents(), tokenizer, "quality") as stream:
        resumed = evaluation.evaluate(model, dictionary, stream, tmp_path, tokens=8,
                                      context=4, batch_tokens=8, chunk_tokens=2, resume=True)
    for field in ("fvu", "dead_fraction", "cross_entropy_delta", "evaluated_tokens", "evaluation_source"):
        assert resumed[field] == quality[field]


def test_evaluator_explicit_gemma_selection_passes_budget_and_preserves_old_loader_api(tmp_path, monkeypatch):
    from .test_gemma_runtime import memory
    observed = []
    guard = memory(28)
    def loader(*, device, memory, model_id):
        observed.append((device, memory, model_id))
        return object(), object()
    def dictionary(*args, **kwargs):
        assert kwargs["memory_limit_gb"] == 28
        return object()
    monkeypatch.setattr(evaluation, "MemoryGuard", lambda limit, device: guard)
    monkeypatch.setattr(evaluation, "load_text_model", loader)
    monkeypatch.setattr(evaluation, "load_dictionary", dictionary)
    monkeypatch.setattr(evaluation, "fineweb_documents", lambda **kw: [])
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: {"status": "failed"})
    assert evaluation.main(["--dictionary", str(tmp_path), "--device", "cpu", "--model", GEMMA_ID]) == 2
    assert observed == [("cpu", guard, GEMMA_ID)]
