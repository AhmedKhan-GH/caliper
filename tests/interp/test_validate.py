"""Tests for Spearman correlation and trace validation."""
import pytest

from caliper import _controls
from caliper.interp import Tracer, _publish
from caliper.interp._backend import make_tiny_backend
from caliper.interp._validate import spearman, validate_trace


def test_spearman_ties_and_zero_variance():
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0], [1.0]) is None


def test_validate_trace_correlates_ablations_with_direct_effects(monkeypatch):
    monkeypatch.setattr(_publish, "_TENSORS", {})
    backend = make_tiny_backend()
    ctl = _controls.Controls()
    ivs = ctl.interventions_on(backend.model)
    ivs.configure(dictionary=backend.dictionary, tokenizer=backend.tokenizer, backend=backend)
    tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend, controls=ctl)
    try:
        graph = tracer.trace("ABCD", max_targets=32, node_threshold=0.99, edge_threshold=0.99)
        res = validate_trace(backend, ivs, graph, prompt="ABCD")
        assert res["trace_id"] == graph.trace_id
        assert res["count"] == len(res["measurements"])
        assert all("predicted" in m and "measured" in m for m in res["measurements"])

        with pytest.raises(ValueError, match="does not match"):
            validate_trace(backend, ivs, graph, prompt="EFGH")
    finally:
        ivs.remove()
