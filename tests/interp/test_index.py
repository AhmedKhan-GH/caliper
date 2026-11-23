"""Exact CPU evidence for feature tables and document windows."""
from collections import Counter

import numpy as np
import pytest
import torch

from caliper.interp import capture, load_dictionary
from caliper.interp._index import FeatureIndex, _TopTable, build_index
from .fixture import TinyTokenizer
from .test_dictionary import write_fixture


@pytest.fixture
def indexed(tmp_path):
    model, _ = write_fixture(tmp_path)
    tokenizer = TinyTokenizer()
    documents = [("first", [65, 66, 65, 67] * 12), ("second", [70, 71, 72]), ("last", [65])]
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, documents, model=model, tokenizer=tokenizer, dictionary=dictionary,
                provenance={"partition": "fixture-examples"}, context_length=17)
    return FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary), documents


def test_streamed_table_equals_brute_force(indexed):
    index, documents = indexed
    activations = [[] for _ in index.dictionary.layers]
    with torch.inference_mode():
        for _, tokens in documents:
            for start in range(0, len(tokens), 17):
                with capture(index.model, index.dictionary.metadata["read_points"]) as saved:
                    index.model(torch.tensor([tokens[start:start + 17]]))
                    for layer, point in enumerate(index.dictionary.metadata["read_points"]):
                        activations[layer].append(index.dictionary.encode(layer, saved[point])[0].numpy())
    for layer, chunks in enumerate(activations):
        values = np.concatenate(chunks)
        for feature in range(values.shape[1]):
            positive = [i for i in range(len(values)) if values[i, feature] > 0]
            ranked = sorted(positive, key=lambda i: (-values[i, feature], i))[:16]
            expected_values = [values[i, feature] for i in ranked] + [0] * (16 - len(ranked))
            expected_offsets = ranked + [-1] * (16 - len(ranked))
            np.testing.assert_array_equal(index.table[layer, feature]["activation"], expected_values)
            np.testing.assert_array_equal(index.table[layer, feature]["offset"], expected_offsets)
            assert index.counts[layer, feature] == len(positive)
    assert isinstance(index.table, np.memmap)
    assert isinstance(index.tokens, np.memmap)
    assert index.tokens.dtype == np.dtype("uint32")


def test_boundary_ties_across_batches(tmp_path):
    table = _TopTable(tmp_path, 1, 3)
    values = torch.tensor([[2., 0., -1.]] * 37, dtype=torch.float64)
    values[22, 0] = 4
    values[29, 0] = 3
    values[25, 1] = 1
    table.update(0, values[:20], 0)
    table.update(0, values[20:], 20)
    np.testing.assert_array_equal(table.table[0, 0]["offset"], [22, 29] + list(range(14)))
    assert table.counts.tolist() == [[37, 1, 0]]
    assert table.table[0, 1, 0]["offset"] == 25
    assert np.all(table.table[0, 2]["offset"] == -1)
    assert np.all(table.table[0, 2]["activation"] == 0)


def test_document_windows_and_real_token_activations(indexed):
    index, documents = indexed
    short_offset = len(documents[0][1])
    candidates = np.argwhere(index.table["offset"] == short_offset)
    layer, feature, _ = map(int, candidates[0])
    rows = index.examples(layer, feature)
    assert len(rows) <= 16
    assert [r["activation"] for r in rows] == sorted((r["activation"] for r in rows), reverse=True)
    source = dict(documents)
    for row in rows:
        assert all(len(row[k]) == 33 for k in ("token_ids", "tokens", "activations", "valid", "positions"))
        assert row["positions"][16] == row["position"]
        tokens = source[row["document_id"]]
        for slot, position in enumerate(row["positions"]):
            if position is None:
                assert row["token_ids"][slot] is None and row["activations"][slot] is None
            else:
                assert row["token_ids"][slot] == tokens[position]
    point = index.dictionary.metadata["read_points"][layer]
    # Independent batch reconstruction preserves the exact CPU kernel shape.
    ids = torch.full((len(rows), 33), 258)
    for i, row in enumerate(rows):
        valid = [v for v in row["token_ids"] if v is not None]
        ids[i, :len(valid)] = torch.tensor(valid)
    with torch.inference_mode(), capture(index.model, [point]) as saved:
        index.model(ids)
        expected = index.dictionary.encode(layer, saved[point])[..., feature]
    for row, values in zip(rows, expected):
        actual = [v for v in row["activations"] if v is not None]
        assert actual == values[:len(actual)].tolist()
    short = next(r for r in rows if r["document_id"] == "second" and r["position"] == 0)
    assert short["token_ids"][:16] == [None] * 16
    assert short["token_ids"][16:19] == [70, 71, 72]
    assert short["token_ids"][19:] == [None] * 14


def test_label_values_and_query_cache(indexed):
    index, _ = indexed
    feature = int(np.argmax(index.counts[0]))
    info = index.show(0, feature)
    weight = index.model.lm_head.weight.detach()
    direction = index.dictionary.layers[0].W_dec[feature]
    scores = (weight @ direction).tolist()
    promoted = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
    assert [t["token_id"] for t in info["promoted_tokens"]] == promoted
    assert [t["score"] for t in info["promoted_tokens"]] == [scores[i] for i in promoted]
    calls = []
    handle = index.model.register_forward_pre_hook(lambda module, args: calls.append(True))
    rows = index.examples(0, feature, limit=1)
    assert len(calls) == 1
    all_rows = index.examples(0, feature)
    assert len(calls) == 1 and len(all_rows) == min(16, info["count"])
    frequencies = Counter(i for row in all_rows for i in row["token_ids"] if i is not None)
    expected = sorted(frequencies, key=lambda i: (-frequencies[i], i))[:3]
    assert [t["token_id"] for t in info["example_tokens"]] == expected
    assert info["label"] == "Output: " + " | ".join(index._token(i) for i in promoted) + " / Examples: " + " | ".join(index._token(i) for i in expected)
    rows[0]["token_ids"][16] = -1
    info["maximum"] = -1
    assert index.examples(0, feature, 1)[0]["token_ids"][16] >= 0
    assert index.show(0, feature)["maximum"] >= 0
    index.clear_cache()
    index.examples(0, feature)
    assert len(calls) == 2
    handle.remove()


@pytest.mark.parametrize("layer,feature,limit", [(-1, 0, 16), (4, 0, 16), (0, -1, 16),
                                               (0, 256, 16), (0, 0, 17), (0, 0, -1),
                                               (True, 0, 16), (0, 1.5, 16)])
def test_invalid_queries(indexed, layer, feature, limit):
    index, _ = indexed
    with pytest.raises(ValueError):
        index.examples(layer, feature, limit)
