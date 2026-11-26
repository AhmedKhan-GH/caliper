"""Bounded CPU checks for graph feature labels from immutable feature indexes."""
from collections import Counter
from copy import deepcopy
from itertools import repeat
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from caliper.interp import _index
from caliper.interp._backend import make_tiny_backend
from caliper.interp._feature_labels import label_graph_nodes
from caliper.interp._graph import Graph, TOKEN_CAP
from caliper.interp._index import FeatureIndex, LABEL_FEATURES, build_index
from caliper.interp._memory import CleanStop, MemoryGuard
from .test_token_labels import unpack_label


@pytest.fixture
def label_index(tmp_path):
    backend = make_tiny_backend(n_layers=2, n_features=300)
    build_index(
        tmp_path,
        [("first", [2, 37, 38, 37, 39] * 7), ("edge", [42, 43])],
        model=backend.model,
        tokenizer=backend.tokenizer,
        dictionary=backend.dictionary,
        provenance={"partition": "fixture"},
        context_length=17,
    )
    memory = MemoryGuard(
        12, "cpu", resident=lambda: 0, footprint=lambda: 0,
        peak=lambda: 0, driver=lambda: 0, pressure=lambda: 50,
    )
    with FeatureIndex(
        tmp_path, model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, memory=memory,
    ) as index:
        yield index


def refuse(*args, **kwargs):
    pytest.fail("label queries must not execute this operation")


def test_batch_labels_match_scores_and_document_windows(label_index, monkeypatch):
    index = label_index
    keys = [tuple(map(int, pair)) for pair in np.argwhere(index.counts > 0)[:4]]
    keys.append((1, int(np.argmin(index.counts[1]))))
    expected = {}
    for layer, feature in keys:
        scores = (index.model.W_U.T @ index.dictionary.layers[layer].W_dec[feature]).tolist()
        promoted = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
        rows = index._windows(layer, feature)
        counts = Counter(i for row in rows for i in row["token_ids"] if i is not None)
        common = sorted(counts, key=lambda i: (-counts[i], i))[:3]
        expected[layer, feature] = (
            "Output: " + " | ".join(index._token(i) for i in promoted)
            + " / Examples: " + (" | ".join(index._token(i) for i in common) or "none")
        )
    monkeypatch.setattr(index.model, "forward", refuse)
    monkeypatch.setattr(index, "_windows", refuse)
    calls = []
    original = index._promoted_many

    def promoted(features, **kwargs):
        calls.append(features)
        return original(features, **kwargs)

    monkeypatch.setattr(index, "_promoted_many", promoted)
    assert index.labels([*keys, keys[0]]) == expected
    assert calls == [keys]
    assert index.labels(keys) == expected
    assert index.show(*keys[0])["label"] == expected[keys[0]]
    assert len(calls) == 1
    assert index._cache == {}
    assert index.label_identity == ("caliper.feature-labels.v1", index.metadata["index_hash"])


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 9, 4096])
def test_batched_promoted_scores_keep_exact_ties_and_bounded_transfers(chunk_size):
    scores = torch.tensor([
        [0., 4., 0.], [2., 0., 0.], [0., 4., 0.], [2., 1., 0.],
        [0., 4., 0.], [0., 2., 0.], [2., 4., 0.], [0., 2., 0.], [2., 1., 0.],
    ], dtype=torch.float64)
    scores[8, 0] = torch.nextafter(scores[8, 0], torch.tensor(3., dtype=torch.float64))
    index = SimpleNamespace(
        model=SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=scores)),
        dictionary=SimpleNamespace(layers=[SimpleNamespace(W_dec=torch.eye(3, dtype=torch.float64))]),
        _token=lambda token: f"#{token}",
    )
    rows = FeatureIndex._promoted_many(index, [(0, 0), (0, 1), (0, 2)], chunk_size=chunk_size)
    for column, tokens in enumerate(rows):
        order = sorted(range(9), key=lambda i: (-scores[i, column].item(), i))[:3]
        assert tokens == [{"token_id": i, "token": f"#{i}", "score": scores[i, column].item()} for i in order]


def test_full_label_batch_cache_bound_and_window_independence(label_index, monkeypatch):
    index = label_index
    key = tuple(map(int, np.argwhere(index.counts > 0)[0]))
    index.examples(*key, limit=1)
    window_cache = dict(index._cache)
    keys = [(layer, feature) for layer in range(2) for feature in range(300)]
    assert len(index.labels(keys[:LABEL_FEATURES])) == LABEL_FEATURES
    assert len(index._summaries) == LABEL_FEATURES
    assert index._cache == window_cache
    promoted = index._promoted_many
    monkeypatch.setattr(index, "_promoted_many", refuse)
    assert len(index.labels(keys[:LABEL_FEATURES])) == LABEL_FEATURES
    monkeypatch.setattr(index, "_promoted_many", promoted)
    index.labels([keys[0]])
    index.labels([keys[LABEL_FEATURES]])
    assert keys[0] in index._summaries and keys[1] not in index._summaries
    assert len(index._summaries) == LABEL_FEATURES
    index.clear_cache()
    assert index._cache == index._summaries == {}


@pytest.mark.parametrize("features", [
    [(-1, 0)], [(0, -1)], [(2, 0)], [(0, 300)],
    [(True, 0)], [(0, 1.5)], [0], [(0,)], [(0, 0, 0)], repeat((0, 0)),
])
def test_invalid_label_batches_refuse_before_projection(label_index, monkeypatch, features):
    monkeypatch.setattr(label_index, "_promoted_many", refuse)
    with pytest.raises(ValueError):
        label_index.labels(features)
    assert label_index._summaries == {}


def test_graph_json_and_publication_share_decorated_labels(label_index, monkeypatch):
    index = label_index
    common = {"activation": 1.0, "influence": 0.5, "effect": 0.25}
    nodes = [
        dict(common, kind="token", id="t:0", layer=-1, position=0, token_id=37, label="A"),
        dict(common, kind="feature", id="f:0:0:0", layer=0, position=0, feature=0, label="L0 F0"),
        dict(common, kind="feature", id="f:0:1:0", layer=0, position=1, feature=0, label="L0 F0"),
        dict(common, kind="error", id="e:0:0", layer=0, position=0, label="L0 error"),
        dict(common, kind="logit", id="l:38", layer=2, position=1, token_id=38, label="B", probability=0.75),
    ]
    original = deepcopy(nodes)
    labeled = label_graph_nodes(nodes, index)
    expected = index.show(0, 0)["label"]
    assert labeled[1]["label"] == labeled[2]["label"] == expected
    assert nodes == original
    published = {}
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
    graph = Graph.build(
        nodes=labeled, edges=[{"source": 0, "target": 1, "weight": 0.25}],
        tokens=[{"position": 0, "token_id": 37, "text": "A"}], layers=2,
        completeness={}, omitted_influence=0.0, error_share=0.1, feature_share=0.9,
        trace_id="fixture-labels", device="cpu",
    )
    graph.publish()
    assert json.loads(graph.to_json())["nodes"][1]["label"] == expected
    packed = expected.encode("utf-8")[:63].decode("utf-8", errors="ignore")
    assert unpack_label(published["trace.label_text"][TOKEN_CAP + 1]) == packed
    assert unpack_label(published["trace.logit_label_text"][0]) == "B"
