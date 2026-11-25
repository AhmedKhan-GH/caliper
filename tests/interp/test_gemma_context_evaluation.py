"""CPU evidence for content-only Gemma quality, exact targets, and resume policy."""
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import capture
from caliper.interp._context import context_policy, prepare_contexts
from tools.interp import eval_dictionary as evaluation
from tools.interp.memory_limits import CleanStop
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream, document_identity
from .test_gemma_import_binding import artifact
from .test_gemma_runtime import gemma_model, runtime_dictionary
from .test_training_stream import documents


def test_partial_batch_statistics_and_ce_use_exact_source_positions():
    model = gemma_model()
    dictionary = runtime_dictionary(model)
    batch = prepare_contexts([[4, 5, 6], [7]], context_policy(model, 4), device="cpu", pad_to=4)
    points = dictionary.metadata["read_points"] + dictionary.metadata["write_points"]
    with torch.no_grad(), capture(model, points) as saved:
        logits = model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
        expected = []
        for i, layer in enumerate(dictionary.layers):
            x = saved[dictionary.metadata["read_points"][i]][batch.content_mask]
            y = saved[dictionary.metadata["write_points"][i]][batch.content_mask]
            a = layer.encode(x)
            expected.append((y.mean(0), (y - y.mean(0)).square().sum(),
                             (y - layer.decode(a)).square().sum(), (a > 0).sum(0)))
    targets = torch.tensor([4, 5, 6, 7])
    expected_ce = F.cross_entropy(logits[:, :-1][batch.prediction_mask], targets, reduction="sum")
    actual = evaluation.evaluate_batch(model, dictionary, batch.input_ids, 4, prepared=batch, chunk_tokens=2)
    assert actual.count == actual.ce_tokens == 4
    assert actual.original_ce == pytest.approx(expected_ce.item(), rel=1e-6)
    with torch.no_grad(), evaluation.replace_all_mlps(model, dictionary, content_mask=batch.content_mask):
        replaced = model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
    replaced_ce = F.cross_entropy(replaced[:, :-1][batch.prediction_mask], targets, reduction="sum")
    assert actual.replaced_ce == pytest.approx(replaced_ce.item(), rel=1e-6)
    for i, (mean, variance, error, hits) in enumerate(expected):
        torch.testing.assert_close(actual.mean[i], mean.double())
        torch.testing.assert_close(actual.variance_sum[i], variance.double())
        torch.testing.assert_close(actual.error_sum[i], error.double())
        torch.testing.assert_close(actual.hits[i], hits)


def test_replacement_preserves_actual_prefix_and_padding_outputs():
    model = gemma_model()
    dictionary = runtime_dictionary(model)
    batch = prepare_contexts([[4, 5, 6], [7]], context_policy(model, 4), device="cpu", pad_to=4)
    before, after, handles = {}, {}, []
    try:
        for i, block in enumerate(model.model.layers):
            handles.append(block.post_feedforward_layernorm.register_forward_hook(
                lambda m, args, out, i=i: before.__setitem__(i, out.clone())))
        with evaluation.replace_all_mlps(model, dictionary, content_mask=batch.content_mask):
            for i, block in enumerate(model.model.layers):
                handles.append(block.post_feedforward_layernorm.register_forward_hook(
                    lambda m, args, out, i=i: after.__setitem__(i, out.clone())))
            with torch.no_grad():
                replaced = model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
        for i in before:
            torch.testing.assert_close(before[i][~batch.content_mask], after[i][~batch.content_mask], rtol=0, atol=0)
            assert not torch.allclose(before[i][batch.content_mask], after[i][batch.content_mask])
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        original = model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
    torch.testing.assert_close(original[:, 0], replaced[:, 0], rtol=0, atol=0)


def tokenizer():
    return SimpleNamespace(encode=lambda text, **kw: [3, 4, 5, 6], eos_token_id=2,
                           bos_token_id=1, pad_token_id=0)


def run(artifact, **kwargs):
    folder, model, dictionary, _ = artifact
    with TextStream(documents() * 2, tokenizer(), "quality") as stream:
        return evaluation.evaluate(model, dictionary, stream, folder, tokens=13,
                                   context=4, batch_tokens=8, chunk_tokens=2, **kwargs)


def test_partial_context_resume_keeps_tokens_documents_and_targets(artifact, monkeypatch, tmp_path):
    folder, model, dictionary, snapshot = artifact
    clean_folder = tmp_path / "uninterrupted"
    shutil.copytree(folder, clean_folder)
    uninterrupted = run((clean_folder, model, dictionary, snapshot))
    implementation = evaluation.evaluate_batch
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(kwargs["prepared"].input_ids.tolist())
        if len(calls) == 2:
            raise CleanStop("fixture context interruption")
        return implementation(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate_batch", interrupted)
    partial = run(artifact)
    assert partial["evaluated_tokens"] == partial["cross_entropy_tokens"] == 5
    folder = artifact[0]
    state = json.loads((folder / "evaluation/state.json").read_text())
    assert state["pending_ids"]["ids"] == [3, 4, 5, 6, 2, 3]
    assert state["pending_ids"]["positions"] == [0, 1, 2, 3, 4, 0]
    monkeypatch.setattr(evaluation, "evaluate_batch", implementation)
    complete = run(artifact, resume=True)
    assert complete["evaluated_tokens"] == complete["cross_entropy_tokens"] == 13
    ids = [document_identity(row["text"])[1] for row in documents() * 2
           if document_identity(row["text"])[0] == "quality"]
    final_state = json.loads((folder / "evaluation/state.json").read_text())
    assert final_state["content_segments"] == [[ids[0], 0, 3], [ids[0], 3, 2],
                                               [ids[1], 0, 3], [ids[1], 3, 2], [ids[2], 0, 3]]
    assert complete["context_policy"] == state["settings"]["context_policy"]
    assert complete["tensor_binding"] == partial["tensor_binding"]
    assert complete["model_binding"] == partial["model_binding"]
    for field in ("fvu", "dead_fraction", "original_cross_entropy", "replaced_cross_entropy", "content_segmentation"):
        assert complete[field] == uninterrupted[field]


@pytest.mark.parametrize("change", ["missing", "prefix", "ce", "segmentation"])
def test_incompatible_context_resume_refuses_before_statistics_load(artifact, monkeypatch, change):
    run(artifact)
    folder = artifact[0]
    path = folder / "evaluation/state.json"
    state = json.loads(path.read_text())
    if change == "missing":
        del state["settings"]["context_policy"]
    else:
        key = {"prefix": "prefix_ids", "ce": "cross_entropy", "segmentation": "segmentation"}[change]
        state["settings"]["context_policy"][key] = "incompatible"
    path.write_text(json.dumps(state))
    before = path.read_bytes(), (folder / "dictionary.json").read_bytes()
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **kw: pytest.fail("statistics load ran"))
    with pytest.raises(ValueError, match="settings changed"):
        run(artifact, resume=True)
    assert before == (path.read_bytes(), (folder / "dictionary.json").read_bytes())


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_final_softcap_is_refused(cap):
    model = gemma_model()
    model.config.final_logit_softcapping = cap
    ids = torch.tensor([[1, 4]])
    with torch.no_grad():
        hidden = hidden_forward(model, ids)
    with pytest.raises(ValueError, match="softcap"):
        evaluation.cross_entropy_sum(model, hidden, ids, 2)
