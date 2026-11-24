"""Graph identities cover requests, visited targets, and displayed content."""
import pytest

from caliper.interp._graph import Graph
from caliper.interp._trace import Tracer
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@pytest.fixture
def tracer(monkeypatch):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = TinyLM(layers=3, width=8, heads=2, latents=16, k=3)
    return Tracer(model, TinyTokenizer(), exact_dictionary(model))


def test_cached_and_fresh_graphs_keep_identity(tracer):
    first = tracer.trace("AB", max_targets=6, max_nodes=32, max_edges=64)
    assert tracer.trace("AB", 6, 32, 64) is first
    tracer._cache.clear()
    tracer._forward_cache = None
    fresh = tracer.trace("AB", 6, 32, 64)
    assert fresh is not first
    assert fresh.nodes == first.nodes and fresh.edges == first.edges
    assert fresh.completeness == first.completeness
    assert fresh.trace_id == first.trace_id


@pytest.mark.parametrize("changed", [{"max_targets": 2}, {"max_nodes": 2}, {"max_edges": 1}])
def test_graph_identity_distinguishes_target_and_pruning_caps(tracer, changed):
    options = {"max_targets": 6, "max_nodes": 32, "max_edges": 64}
    first = tracer.trace("AB", **options)
    options.update(changed)
    limited = tracer.trace("AB", **options)
    assert limited.trace_id != first.trace_id
    assert (limited.nodes, limited.edges, limited.completeness) != (first.nodes, first.edges, first.completeness)
    assert tracer.trace("AB", **options) is limited


def test_graph_identity_distinguishes_configuration_with_equal_content(tracer):
    first = tracer.trace("AB", max_targets=0, max_nodes=511, max_edges=8192)
    second = tracer.trace("AB", max_targets=0, max_nodes=512, max_edges=8192)
    assert second.nodes == first.nodes and second.edges == first.edges
    assert second.trace_id != first.trace_id
    tracer.target_batch_size = 63
    third = tracer.trace("AB", max_targets=0, max_nodes=512, max_edges=8192)
    assert third is not second and third.trace_id != second.trace_id
    assert third.nodes == second.nodes and third.edges == second.edges


def test_graph_identity_distinguishes_adaptive_visited_targets(tracer, monkeypatch):
    import torch
    from caliper.interp import _trace

    class SyntheticEffects:
        """A three-feature DAG whose ranking changes after the first target."""

        def __init__(self, path, dictionary, ids, tokenizer):
            self.path = path
            self.tokens = [{"position": i, "token_id": token, "text": str(token)} for i, token in enumerate(ids)]
            self.nodes = [{"id": f"token:{i}", "kind": "token", "layer": -1,
                           "position": i, "activation": 1.} for i in range(len(ids))]
            self.feature_nodes = [len(ids) + i for i in range(3)]
            self.nodes.extend({"id": f"feature:{i}", "kind": "feature", "layer": i,
                               "position": 0, "feature": i, "activation": 1.} for i in range(3))
            self.logit_nodes = [len(self.nodes)]
            self.nodes.append({"id": "logit:0", "kind": "logit", "layer": path.layers,
                               "position": 0, "activation": 1., "probability": 1.})
            self.weights = torch.zeros(len(self.nodes), len(self.nodes), dtype=path.dtype)
            upstream, other, downstream = self.feature_nodes
            logit = self.logit_nodes[0]
            self.weights[downstream, logit] = 1.
            self.weights[other, logit] = .9
            self.weights[upstream, logit] = .8
            self.weights[upstream, downstream] = 1.
            self.weights[0, upstream] = 1.
            self.weights[0, other] = 1.

        def backward(self, indices):
            return self.weights[:, indices], [
                {"id": self.nodes[i]["id"], "absolute_residual": 0., "relative_residual": 0.} for i in indices]

    # The real influence algorithm visits downstream + other in one batch.
    # With one target per batch, upstream overtakes other after downstream.
    monkeypatch.setattr(_trace, "_DirectEffects", SyntheticEffects)
    monkeypatch.setattr(tracer.guard, "batch_size", lambda *args, **kwargs: 2)
    first = tracer.trace("AB", max_targets=2, max_nodes=32, max_edges=64)
    tracer._cache.clear()
    monkeypatch.setattr(tracer.guard, "batch_size", lambda *args, **kwargs: 1)
    second = tracer.trace("AB", max_targets=2, max_nodes=32, max_edges=64)
    first_targets = [row["id"] for row in first.completeness["targets"]]
    second_targets = [row["id"] for row in second.completeness["targets"]]
    assert first_targets == ["logit:0", "feature:2", "feature:1"]
    assert second_targets == ["logit:0", "feature:2", "feature:0"]
    assert second.trace_id != first.trace_id
    assert tracer.trace("AB", max_targets=2, max_nodes=32, max_edges=64) is second
