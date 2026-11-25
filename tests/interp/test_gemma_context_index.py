"""CPU checks for Gemma index prefixes, source offsets, and stale policy refusal."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from caliper.interp import capture
from caliper.interp._index import FeatureIndex, build_index
from .test_gemma_import_binding import artifact
from .test_index_precision import rewrite_manifest


def tokenizer():
    return SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0,
                           convert_ids_to_tokens=lambda token: str(token),
                           decode=lambda ids, **kw: " ".join(map(str, ids)))


def test_index_counts_and_offsets_exclude_prefixes_and_keep_partial_documents(artifact):
    folder, model, dictionary, _ = artifact
    documents = [("a", [3, 4, 5, 6, 7]), ("b", [8]), ("c", [9, 10, 11, 12])]
    expected = [[] for _ in dictionary.layers]
    # The token cap retains two tokens from the final document.
    for _, source in [documents[0], documents[1], ("c", [9, 10])]:
        for start in range(0, len(source), 3):
            ids = torch.tensor([[1, *source[start:start + 3]]])
            with torch.no_grad(), capture(model, dictionary.metadata["read_points"]) as saved:
                model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                for i, point in enumerate(dictionary.metadata["read_points"]):
                    expected[i].append(dictionary.layers[i].encode(saved[point][0, 1:]).numpy())
    manifest = build_index(folder, documents, model=model, tokenizer=tokenizer(), dictionary=dictionary,
                           provenance={"partition": "fixture"}, context_length=4, max_tokens=8)
    assert manifest["context_policy"]["content_context"] == 3
    assert manifest["example_context_policy"]["model_context"] == 34
    assert manifest["tokens"] == manifest["token_budget"] == 8
    with FeatureIndex(folder, model=model, tokenizer=tokenizer(), dictionary=dictionary) as index:
        assert index.tokens.tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
        assert index.documents["length"].tolist() == [5, 1, 2]
        assert index.documents["truncated"].tolist() == [False, False, True]
        for layer, chunks in enumerate(expected):
            values = np.concatenate(chunks)
            np.testing.assert_array_equal(index.counts[layer], (values > 0).sum(0))
            for feature in range(values.shape[-1]):
                ranked = sorted((i for i in range(8) if values[i, feature] > 0),
                                key=lambda i: (-values[i, feature], i))
                np.testing.assert_array_equal(index.table[layer, feature]["offset"][:len(ranked)], ranked)
                np.testing.assert_allclose(index.table[layer, feature]["activation"][:len(ranked)],
                                           values[ranked, feature], rtol=1e-6)


def test_example_windows_match_public_bos_context_and_exclude_padding(artifact):
    folder, model, dictionary, _ = artifact
    build_index(folder, [("short", [3]), ("long", [4, 5, 6, 7])], model=model, tokenizer=tokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(folder, model=model, tokenizer=tokenizer(), dictionary=dictionary) as index:
        feature = int(np.argmax(index.counts[0]))
        rows = index.examples(0, feature)
        assert rows and {row["document_id"] for row in rows} == {"short", "long"}
        ids = torch.zeros(len(rows), 34, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            content = [token for token in row["token_ids"] if token is not None]
            ids[i, :len(content) + 1] = torch.tensor([1, *content])
            mask[i, :len(content) + 1] = 1
        point = dictionary.metadata["read_points"][0]
        with torch.no_grad(), capture(model, [point]) as saved:
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            expected = dictionary.layers[0].encode(saved[point])[..., feature]
        for i, row in enumerate(rows):
            actual = [value for value in row["activations"] if value is not None]
            assert actual == pytest.approx(expected[i, 1:len(actual) + 1].tolist(), rel=1e-6)
            assert len(actual) == sum(row["valid"])
            assert all(value is None for value, valid in zip(row["activations"], row["valid"]) if not valid)


@pytest.mark.parametrize("change", ["missing", "prefix", "window", "context", "boolean", "float"])
def test_stale_index_context_requires_rebuild(artifact, monkeypatch, change):
    folder, model, dictionary, _ = artifact
    manifest = build_index(folder, [("a", [3, 4])], model=model, tokenizer=tokenizer(),
                           dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    if change == "missing":
        changes = {"context_policy": None, "example_context_policy": None}
    elif change == "context":
        changes = {"context_length": 5}
    else:
        key = "example_context_policy" if change == "window" else "context_policy"
        policy = deepcopy(manifest[key])
        if change in ("boolean", "float"):
            policy["pad_id"] = False if change == "boolean" else 0.0
        else:
            policy["prefix_ids"] = []
        changes = {key: policy}
    rewrite_manifest(folder, **changes)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("stale index arrays were opened"))
    with pytest.raises(ValueError, match="context policy differs; rebuild"):
        FeatureIndex(folder, model=model, tokenizer=tokenizer(), dictionary=dictionary)


def test_invalid_gemma_context_refuses_before_resource_admission(tmp_path, monkeypatch):
    from caliper.interp import _index
    from tools.interp.build_index import main

    model = SimpleNamespace(config=SimpleNamespace(model_type="gemma2"))
    monkeypatch.setattr(_index, "family_memory_limit", lambda *a, **kw: pytest.fail("late admission refusal"))
    with pytest.raises(ValueError, match="reserve"):
        build_index(tmp_path, [], model=model, tokenizer=tokenizer(),
                    provenance={"partition": "fixture"}, context_length=1)
    with pytest.raises(SystemExit) as refusal:
        main(["--dictionary", str(tmp_path / "absent"), "--model", "google/gemma-2-2b", "--context", "1"])
    assert refusal.value.code == 2
