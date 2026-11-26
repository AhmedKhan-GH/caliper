"""CPU graph labels share trace identity, cache entries, and atomic publication."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import Tracer, _publish
from caliper.interp._service import InterpretationService
from caliper.interp._memory import CleanStop
from .test_service import actual
from .test_token_labels import unpack_label


@pytest.fixture
def trace_index(actual):
    return actual.index


@pytest.mark.parametrize("lazy", [False, True])
def test_service_binds_labels_before_trace_identity_and_publication(trace_index, monkeypatch, lazy):
    index = trace_index
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    tracer = Tracer(index.model, index.tokenizer, index.dictionary)
    plain = tracer.trace("AB", max_targets=8)
    calls = []
    def factory():
        calls.append("index")
        return index
    service = InterpretationService(tracer=tracer, **({"index_factory": factory} if lazy else {"index": index}))
    try:
        result = service._execute("trace.run", {"prompt": "AB", "max_targets": 8}, 1)
        graph = service.graph
        assert result["trace_id"] == graph.trace_id != plain.trace_id
        features = [(i, n) for i, n in enumerate(graph.nodes) if n["kind"] == "feature"]
        assert features
        for i, node in features:
            expected = index.show(node["layer"], node["feature"])["label"]
            assert node["label"] == expected
            assert unpack_label(published["trace.label_text"][32 + i]) == expected.encode()[:63].decode(errors="ignore")
        original = published["trace.label_text"].clone()
        monkeypatch.setattr(index, "labels", lambda *a, **k: pytest.fail("cached labels were recomputed"))
        service._execute("trace.run", {"prompt": "AB", "max_targets": 8}, 2)
        assert service.graph is graph and torch.equal(published["trace.label_text"], original)
        assert service._execute("trace.read", {}, 3)["nodes"] == graph.read()["nodes"]
        assert calls == (["index"] if lazy else [])
    finally:
        service.close()


def test_label_failure_preserves_published_graph_and_its_cache(trace_index, monkeypatch):
    index = trace_index
    monkeypatch.setattr(_publish, "_TENSORS", {})
    tracer = Tracer(index.model, index.tokenizer, index.dictionary)
    plain = tracer.trace("AB", max_targets=8)
    before = dict(_publish._TENSORS)
    tracer.bind_index(index)
    def fail(*args, **kwargs):
        raise CleanStop("fixture label stop")
    monkeypatch.setattr(index, "labels", fail)
    with pytest.raises(CleanStop, match="fixture label stop"):
        tracer.trace("AB", max_targets=8)
    assert set(_publish._TENSORS) == set(before)
    assert all(_publish._TENSORS[name] is tensor for name, tensor in before.items())
    assert list(tracer._cache.values()) == [plain]


@pytest.mark.parametrize("field", ["model", "tokenizer", "dictionary"])
def test_trace_index_mismatch_refuses_before_trace(trace_index, field):
    index = trace_index
    tracer = Tracer(index.model, index.tokenizer, index.dictionary)
    resources = {name: getattr(index, name) for name in ("model", "tokenizer", "dictionary")}
    resources[field] = object()
    with pytest.raises(ValueError, match="same " + field):
        tracer.bind_index(SimpleNamespace(**resources, label_identity=("fixture", "other")))
    assert tracer.index is None and not tracer._cache
