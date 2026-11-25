"""Bounded example batches with live memory and retained snapshot accounting."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from caliper.interp import _index
from caliper.interp._memory import CleanStop
from .test_index import indexed
from .test_memory_phases import process_guard


def test_gemma_example_admission_counts_live_snapshot_and_keeps_reserve():
    tensor = torch.empty(0)
    case = SimpleNamespace(
        context_policy={"prefix_ids": [2]},
        model=SimpleNamespace(parameters=lambda: iter([tensor]), config=SimpleNamespace(
            model_type="gemma2", num_attention_heads=8, head_dim=256,
            intermediate_size=9216, vocab_size=256000)),
        dictionary=SimpleNamespace(metadata={"width": 2304, "latents": 16384},
                                   layers=[SimpleNamespace(W_enc=tensor)] * 26))
    case._example_bytes = lambda rows: _index.FeatureIndex._example_bytes(case, rows)
    # These are scalar accounting inputs, not allocated weights or measurements.
    weights, dictionary, snapshot = 10_457_367_552, 7_855_384_576, 7_855_384_576
    readings = {"current": weights + dictionary + snapshot + 1_000_000_000,
                "peak": 27_900_000_000}
    case.memory = process_guard(28_000_000_000, readings)
    count = _index.FeatureIndex._admit_examples(case, 16)
    assert 1 <= count < 16
    assert readings["current"] + case._example_bytes(count) <= case.memory.limit
    assert readings["current"] + case._example_bytes(count + 1) > case.memory.limit
    assert case._example_bytes(1) > 512_000_000
    prefixed_bytes = case._example_bytes(1)
    case.context_policy = None
    unprefixed_bytes = case._example_bytes(1)
    case.context_policy = {"prefix_ids": [2]}
    assert prefixed_bytes > unprefixed_bytes
    readings["current"] = case.memory.limit - unprefixed_bytes
    with pytest.raises(CleanStop, match="example allocation"):
        _index.FeatureIndex._admit_examples(case, 1)
    readings["current"] = case.memory.limit - prefixed_bytes
    assert _index.FeatureIndex._admit_examples(case, 1) == 1
    readings["current"] = case.memory.limit - case._example_bytes(1) + 1
    with pytest.raises(CleanStop, match="example allocation"):
        _index.FeatureIndex._admit_examples(case, 16)
    readings.update(current=weights + dictionary, peak=28_000_000_001)
    with pytest.raises(CleanStop, match="memory limit"):
        _index.FeatureIndex._admit_examples(case, 1)


def test_example_chunks_preserve_order_values_cache_and_tensor_lifetime(indexed, monkeypatch):
    index, _ = indexed
    feature = int(np.argmax(index.counts[0]))
    expected = index.examples(0, feature)
    assert len(expected) == 16
    index.clear_cache()
    base = 1_000
    limit = base + index._example_bytes(3)
    readings = {"current": base, "peak": limit - 1}
    index.memory = process_guard(limit, readings)
    batches, references, active, scopes = [], [], [], []
    forward, encode = _index._forward, _index._trace_features

    @contextmanager
    def scope(device):
        assert not active and all(ref() is None for ref in references)
        active.append(device)
        scopes.append(device)
        try:
            yield
        finally:
            assert all(ref() is None for ref in references)
            active.pop()

    def observed(model, ids, mask):
        assert active
        references.extend([weakref.ref(ids), weakref.ref(mask)])
        batches.append(len(ids))
        return forward(model, ids, mask)

    def encoded(layer, values, **kwargs):
        result = encode(layer, values, **kwargs)
        references.extend([weakref.ref(values), weakref.ref(result)])
        return result

    monkeypatch.setattr(_index, "_device_scope", scope)
    monkeypatch.setattr(_index, "_forward", observed)
    monkeypatch.setattr(_index, "_trace_features", encoded)
    prefix = index.examples(0, feature, limit=1)
    actual = index.examples(0, feature)
    assert batches == [3, 3, 3, 3, 3, 1] and len(scopes) == 6 and not active
    assert prefix == actual[:1]
    for left, right in zip(actual, expected):
        assert {k: v for k, v in left.items() if k != "activations"} == {
            k: v for k, v in right.items() if k != "activations"}
        assert [v is None for v in left["activations"]] == [v is None for v in right["activations"]]
        np.testing.assert_allclose([v for v in left["activations"] if v is not None],
                                   [v for v in right["activations"] if v is not None], rtol=1e-12, atol=1e-12)
    prefix[0]["activations"][16] = -1
    assert index.examples(0, feature, 1) == actual[:1]


@pytest.mark.parametrize("failure", ["current", "historical", "stop", "nonfinite"])
def test_failed_chunk_releases_scope_and_never_caches_partial_examples(indexed, monkeypatch, failure):
    index, _ = indexed
    feature = int(np.argmax(index.counts[0]))
    limit = 1_000 + index._example_bytes(2)
    readings = {"current": 1_000, "peak": 1_000}
    index.memory = process_guard(limit, readings)
    batches, active = [], []
    original = index._example_values

    @contextmanager
    def scope(device):
        active.append(device)
        try:
            yield
        finally:
            active.pop()

    def values(rows, layer, selected, device):
        batches.append(len(rows))
        if len(batches) == 2 and failure in ("stop", "nonfinite"):
            raise CleanStop("batch stop") if failure == "stop" else ValueError("nonfinite batch")
        result = original(rows, layer, selected, device)
        if failure == "current":
            readings["current"] = limit - index._example_bytes(1) + 1
        elif failure == "historical":
            readings["peak"] = limit + 1
        return result

    monkeypatch.setattr(_index, "_device_scope", scope)
    monkeypatch.setattr(index, "_example_values", values)
    with pytest.raises(ValueError if failure == "nonfinite" else CleanStop):
        index.examples(0, feature)
    assert not active and "examples" not in index._cache[0, feature]
    assert batches == ([2, 2] if failure in ("stop", "nonfinite") else [2])


def test_example_refusal_precedes_first_tensor_allocation(indexed, monkeypatch):
    index, _ = indexed
    feature = int(np.argmax(index.counts[0]))
    limit = 1_000 + index._example_bytes(1) - 1
    index.memory = process_guard(limit, {"current": 1_000, "peak": 1_000})
    monkeypatch.setattr(torch, "full", lambda *a, **kw: pytest.fail("unadmitted example tensor"))
    assert index.examples(0, feature, limit=0) == []
    with pytest.raises(CleanStop, match="example allocation"):
        index.examples(0, feature)
    assert "examples" not in index._cache[0, feature]
