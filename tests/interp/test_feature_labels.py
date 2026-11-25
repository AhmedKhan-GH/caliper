"""Bounded CPU evidence for graph labels from immutable feature indexes."""
from collections import Counter
from copy import deepcopy
from itertools import repeat
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from caliper.interp import load_dictionary
from caliper.interp import _index
from caliper.interp._feature_labels import label_graph_nodes
from caliper.interp._graph import Graph, NODE_CAP, REFERENCE_NODE_CAP, TOKEN_CAP
from caliper.interp._index import FeatureIndex, LABEL_FEATURES, build_index
from caliper.interp._labels import bounded_label
from caliper.interp._memory import CleanStop, MemoryGuard
from caliper.interp._reference_graph import PROFILE
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_token_labels import unpack_label


@pytest.fixture
def label_index(tmp_path, request):
    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=300, k=2))
    tokenizer = TinyTokenizer()
    token_text = getattr(request, "param", None)
    if token_text is not None:
        tokenizer.decode = lambda ids: "".join(token_text + str(token) for token in ids)
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, [("first", [65, 66, 65, 67] * 9), ("edge", [70, 71])],
                model=model, tokenizer=tokenizer, dictionary=dictionary,
                provenance={"partition": "fixture"}, context_length=17)
    memory = MemoryGuard(12, "cpu", resident=lambda: 0, footprint=lambda: 0,
                         peak=lambda: 0, driver=lambda: 0, pressure=lambda: 50)
    with FeatureIndex(tmp_path, model=model, tokenizer=tokenizer,
                      dictionary=dictionary, memory=memory) as index:
        yield index


def refuse(*args, **kwargs):
    pytest.fail("label queries must not execute this operation")


def test_batch_labels_match_scores_and_document_windows(label_index, monkeypatch):
    index = label_index
    keys = [tuple(map(int, pair)) for pair in np.argwhere(index.counts > 0)[:4]]
    keys.append((1, int(np.argmin(index.counts[1]))))
    expected = {}
    for layer, feature in keys:
        scores = (index.model.lm_head.weight @ index.dictionary.layers[layer].W_dec[feature]).tolist()
        promoted = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
        rows = index._windows(layer, feature)
        counts = Counter(i for row in rows for i in row["token_ids"] if i is not None)
        common = sorted(counts, key=lambda i: (-counts[i], i))[:3]
        expected[layer, feature] = bounded_label(
            "Output: " + " | ".join(index._token(i) for i in promoted)
            + " / Examples: " + (" | ".join(index._token(i) for i in common) or "none"))
    contents = {file.name: file.read_bytes() for file in (index.root / "index").iterdir()}
    for name in ("_dictionary_hash", "_source_hash", "_tokenizer_hash", "model_identity", "weights_hash"):
        monkeypatch.setattr(_index, name, refuse)
    monkeypatch.setattr(index.model, "forward", refuse)
    monkeypatch.setattr(index, "_windows", refuse)
    monkeypatch.setattr(index, "_promoted", refuse)
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
    assert {file.name: file.read_bytes() for file in (index.root / "index").iterdir()} == contents
    shown = index.show(*keys[0])
    shown["promoted_tokens"].clear()
    assert len(index.show(*keys[0])["promoted_tokens"]) == 3


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 9, 4096])
def test_batched_promoted_scores_keep_exact_ties_and_bounded_transfers(monkeypatch, chunk_size):
    scores = torch.tensor([[0., 4., 0.], [2., 0., 0.], [0., 4., 0.], [2., 1., 0.],
                           [0., 4., 0.], [0., 2., 0.], [2., 4., 0.], [0., 2., 0.],
                           [2., 1., 0.]], dtype=torch.float64)
    scores[8, 0] = torch.nextafter(scores[8, 0], torch.tensor(3., dtype=torch.float64))
    index = SimpleNamespace(
        model=SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=scores)),
        dictionary=SimpleNamespace(layers=[SimpleNamespace(W_dec=torch.eye(3, dtype=torch.float64))]),
        _token=lambda token: f"#{token}")
    transfers, calls = [], []
    original_cpu, original_topk = torch.Tensor.cpu, _index._stable_topk

    def cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        assert tensor.shape[0] == 3 and tensor.shape[1] <= 6
        return original_cpu(tensor, *args, **kwargs)

    def topk(values, offsets, k):
        calls.append(tuple(values.shape))
        return original_topk(values, offsets, k)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(_index, "_stable_topk", topk)
    rows = FeatureIndex._promoted_many(index, [(0, 0), (0, 1), (0, 2)], chunk_size=chunk_size)
    for column, tokens in enumerate(rows):
        order = sorted(range(9), key=lambda i: (-scores[i, column].item(), i))[:3]
        assert tokens == [{"token_id": i, "token": f"#{i}", "score": scores[i, column].item()} for i in order]
    assert calls[::2] == [(3, min(chunk_size, 9 - start)) for start in range(0, 9, chunk_size)]
    assert len(transfers) == 4 * len(calls[::2])


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
    assert index._cache == window_cache
    identity = index.label_identity
    index.clear_cache()
    assert index._cache == index._summaries == {}
    assert index.label_identity == identity


@pytest.mark.parametrize("features", [[(-1, 0)], [(0, -1)], [(2, 0)], [(0, 300)],
                                      [(True, 0)], [(0, 1.5)], [0], [(0,)],
                                      [(0, 0, 0)], repeat((0, 0))])
def test_invalid_label_batches_refuse_before_projection(label_index, monkeypatch, features):
    monkeypatch.setattr(label_index, "_promoted_many", refuse)
    with pytest.raises(ValueError):
        label_index.labels(features)
    assert label_index._summaries == {}


def test_empty_closed_and_training_label_queries(label_index, monkeypatch):
    index = label_index
    monkeypatch.setattr(index, "_promoted_many", refuse)
    assert index.labels([]) == {}
    index.model.train()
    with pytest.raises(ValueError, match="eval"):
        index.labels([])
    index.model.eval()
    index.close()
    assert index._summaries == {}
    with pytest.raises(ValueError, match="closed"):
        index.labels([])
    with pytest.raises(ValueError, match="closed"):
        _ = index.label_identity


def test_label_memory_admission_precedes_projection(label_index, monkeypatch):
    index = label_index
    index.memory.limit = 1
    monkeypatch.setattr(index, "_promoted_many", refuse)
    with pytest.raises(CleanStop, match="feature label allocation"):
        index.labels([(0, 0)])
    assert index._summaries == {}


@pytest.mark.parametrize("stage", ["feature label chunk", "feature label examples", "feature labels complete"])
def test_label_cancellation_preserves_prior_cache(label_index, stage):
    index = label_index
    prior = index.labels([(0, 0)])
    def check(current):
        if current == stage:
            raise CleanStop("fixture label cancellation")
    with pytest.raises(CleanStop, match="label cancellation"):
        index.labels([(0, 0), (1, 0)], check=check)
    assert set(index._summaries) == {(0, 0)}
    assert index.labels([(0, 0)]) == prior


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_promoted_batch_refuses_cache(label_index, bad):
    index = label_index
    # Deliberately violate the immutable-resource contract to exercise numerical refusal.
    with torch.no_grad():
        index.model.lm_head.weight[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        index.labels([(0, 0), (1, 0)])
    assert index._summaries == {}


def display_nodes():
    common = {"activation": 1., "influence": .5, "effect": .25}
    return [
        dict(common, kind="token", id="t:0", layer=-1, position=0, token_id=65, label="A"),
        dict(common, kind="feature", id="f:0:0:0", layer=0, position=0, feature=0, label="L0 F0"),
        dict(common, kind="feature", id="f:0:1:0", layer=0, position=1, feature=0, label="L0 F0"),
        dict(common, kind="error", id="e:0:0", layer=0, position=0, label="L0 error"),
        dict(common, kind="logit", id="l:66", layer=2, position=1, token_id=66, label="B", probability=.75),
    ]


@pytest.mark.parametrize("label_index,expected_label", [
    (None, None), ("é" * 40, "Output: " + "é" * 26 + "..."),
], indirect=["label_index"])
def test_graph_json_and_publication_share_decorated_labels(label_index, monkeypatch, expected_label):
    index = label_index
    nodes = display_nodes()
    original = deepcopy(nodes)
    labeled = label_graph_nodes(nodes, index)
    expected = index.show(0, 0)["label"]
    assert len(expected.encode("utf-8")) <= 63
    if expected_label is not None:
        assert expected == expected_label
        shown = index.show(0, 0)
        assert shown["promoted_tokens"] == index._promoted(0, 0)
        for row in shown["promoted_tokens"] + shown["example_tokens"]:
            assert row["token"] == "é" * 40 + str(row["token_id"])
    assert labeled[1]["label"] == labeled[2]["label"] == expected
    assert labeled[0] == nodes[0] and labeled[3:] == nodes[3:]
    assert nodes == original
    assert all(left is not right for left, right in zip(nodes, labeled))
    assert [{k: v for k, v in n.items() if k != "label"} for n in labeled] == [
        {k: v for k, v in n.items() if k != "label"} for n in original]
    published = {}
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
    graph = Graph.build(nodes=labeled, edges=[{"source": 0, "target": 1, "weight": .25}],
                        tokens=[{"position": 0, "token_id": 65, "text": "A"}], layers=2,
                        completeness={}, omitted_influence=0., error_share=.1, feature_share=.9,
                        trace_id="fixture-labels", device="cpu")
    graph.publish()
    assert graph.read()["nodes"] == labeled
    assert json.loads(graph.to_json())["nodes"][1]["label"] == expected
    assert unpack_label(published["trace.label_text"][TOKEN_CAP + 1]) == expected
    assert unpack_label(published["trace.label_text"][TOKEN_CAP + 2]) == expected
    assert unpack_label(published["trace.label_text"][0]) == "A"
    assert unpack_label(published["trace.label_text"][TOKEN_CAP + 3]) == "L0 error"
    assert unpack_label(published["trace.logit_label_text"][0]) == "B"
    first = {name: value.clone() for name, value in published.items()}
    monkeypatch.setattr(index, "labels", refuse)
    graph.publish()  # A cached graph already owns its labels.
    assert all(torch.equal(first[name], value) for name, value in published.items())


def test_graph_label_decoration_is_atomic_and_bounded():
    nodes = display_nodes()
    original = deepcopy(nodes)
    calls = []
    def labels(features, **kwargs):
        calls.append(features)
        raise CleanStop("fixture graph cancellation")
    index = SimpleNamespace(labels=labels)
    with pytest.raises(CleanStop, match="graph cancellation"):
        label_graph_nodes(nodes, index)
    assert calls == [[(0, 0)]]
    assert nodes == original
    with pytest.raises(ValueError, match="label nodes"):
        label_graph_nodes([nodes[1]] * 513, index)
    assert len(calls) == 1


def test_indexless_graph_keeps_explicit_feature_ids():
    nodes = display_nodes()
    nodes[1]["label"] = None
    labeled = label_graph_nodes(nodes)
    assert labeled[1]["label"] == labeled[2]["label"] == "L0 F0"
    assert nodes[1]["label"] is None


@pytest.mark.parametrize("profile,capacity", [("caliper", NODE_CAP), (PROFILE, REFERENCE_NODE_CAP)])
def test_graph_label_profile_boundaries(profile, capacity):
    node = display_nodes()[1]
    labeled = label_graph_nodes([node] * capacity, profile=profile)
    assert len(labeled) == capacity
    assert all(record["label"] == "L0 F0" and record is not node for record in labeled)
    with pytest.raises(ValueError, match="label nodes"):
        label_graph_nodes([node] * (capacity + 1), SimpleNamespace(labels=refuse), profile=profile)
    with pytest.raises(ValueError, match="profile"):
        label_graph_nodes([], profile="unknown")


def test_reference_labels_keep_unique_order_and_all_batches(label_index, monkeypatch):
    index = label_index
    keys = [(layer, feature) for layer in range(2) for feature in reversed(range(300))]
    requests = [*keys[:510], keys[0], *keys[510:], keys[1]]
    nodes = [dict(display_nodes()[1], id=f"feature:{i}", layer=layer, feature=feature)
             for i, (layer, feature) in enumerate(requests)]
    original = deepcopy(nodes)
    batches, expected, stages = [], {}, []
    labels = index.labels

    # Keep one callback object to check that each batch receives the same deadline.
    check = stages.append

    def record_batch(features, **kwargs):
        assert kwargs["check"] is check
        batches.append(list(features))
        result = labels(features, **kwargs)
        expected.update(result)
        return result

    monkeypatch.setattr(index, "labels", record_batch)
    labeled = label_graph_nodes(nodes, index, profile=PROFILE, check=check)
    assert batches == [keys[:LABEL_FEATURES], keys[LABEL_FEATURES:]]
    assert [node["label"] for node in labeled] == [expected[key] for key in requests]
    assert len(index._summaries) == LABEL_FEATURES and keys[0] not in index._summaries
    assert nodes == original
    assert all(before is not after for before, after in zip(nodes, labeled))
    assert [{k: v for k, v in n.items() if k != "label"} for n in labeled] == [
        {k: v for k, v in n.items() if k != "label"} for n in original]
    assert stages.count("graph label batch admission") == stages.count("graph label batch complete") == 2
    assert stages[-1] == "graph labels complete"
    with pytest.raises(ValueError, match="at most 512 requests"):
        labels(keys[:LABEL_FEATURES + 1])


@pytest.mark.parametrize("failure", ["index", "admission", "complete"])
def test_later_label_batch_failure_keeps_input_unchanged(failure):
    nodes = [dict(display_nodes()[1], feature=i) for i in range(LABEL_FEATURES + 1)]
    original = deepcopy(nodes)
    batches, stages = [], Counter()

    def check(stage):
        stages[stage] += 1
        if stage == "graph label batch " + failure and stages[stage] == 2:
            raise CleanStop("later batch cancellation")

    def labels(features, **kwargs):
        batches.append(list(features))
        if failure == "index" and len(batches) == 2:
            raise CleanStop("later batch cancellation")
        return {key: f"L{key[0]} F{key[1]}" for key in features}

    with pytest.raises(CleanStop, match="later batch cancellation"):
        label_graph_nodes(nodes, SimpleNamespace(labels=labels), profile=PROFILE, check=check)
    assert len(batches) == (1 if failure == "admission" else 2)
    assert nodes == original
    assert stages["graph labels complete"] == 0
