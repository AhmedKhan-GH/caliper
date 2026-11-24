"""Exact source influence and bounded path search on small graph oracles."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._influence import influence
from caliper.interp._trace import _prune, _token_path


@pytest.mark.parametrize("reachable", [False, True])
def test_token_path_visits_shared_ancestors_once(reachable):
    visits = Counter()

    class Incoming(dict):
        def get(self, key, default=None):
            visits[key] += 1
            return super().get(key, default)

    nodes = [{"kind": "token"}] + [{"kind": "feature"}] * 4 + [{"kind": "logit"}]
    incoming = Incoming({5: [(4, 0), (3, 1)], 4: [(2, 2), (1, 3)],
                         3: [(2, 4), (1, 5)], 2: [(1, 6)], 1: [(0, 7)] if reachable else []})
    assert _token_path(nodes, incoming, 5) == ([7, 3, 0] if reachable else None)
    assert visits == Counter({5: 1, 4: 1, 3: 1, 2: 1, 1: 1})


@pytest.mark.parametrize("isolated_logit", [False, True])
@pytest.mark.parametrize("max_nodes", [2, 3, 7])
def test_pruning_threshold_uses_source_influence(isolated_logit, max_nodes):
    nodes = [{"id": "token", "kind": "token"}]
    nodes += [{"id": f"feature:{i}", "kind": "feature"} for i in range(4)]
    logits = [5, 6] if isolated_logit else [5]
    nodes += [{"id": f"logit:{i}", "kind": "logit"} for i in logits]
    targets = torch.tensor(logits)
    weights = torch.zeros(len(nodes), len(logits), dtype=torch.float64)
    weights[1:5, -1] = torch.tensor([.4, .2, .2, .2], dtype=torch.float64)
    seeds = torch.zeros(len(nodes), dtype=torch.float64)
    seeds[targets] = torch.tensor([.8, .2] if isolated_logit else [1.], dtype=torch.float64)
    scores, normalized = influence(weights, targets, seeds)
    effects = SimpleNamespace(nodes=nodes, logit_nodes=logits, path=SimpleNamespace(tokens=1))
    displayed, edges, omitted, _, connectivity = _prune(effects, weights, logits, scores, normalized,
                                                       max_nodes=max_nodes, max_edges=8)
    assert connectivity == {"status": "unavailable", "minimum_nodes": None, "minimum_edges": None}
    kept = [n["id"] for n in displayed if n["kind"] == "feature"]
    count = max(1, min(3, max_nodes - len(logits)))
    assert kept == [f"feature:{i}" for i in range(count)]
    assert omitted == pytest.approx(1 - (.4 + .2 * (count - 1)))
    if max_nodes == 7:
        assert omitted <= .2 + 1e-12
    assert not any(n["kind"] == "token" for n in displayed)
    assert all(displayed[e["source"]]["kind"] == "feature" for e in edges)


def test_pruning_zero_source_influence_has_no_invented_path():
    nodes = [{"kind": "token"}, {"kind": "feature"}, {"kind": "logit"}]
    effects = SimpleNamespace(nodes=nodes, logit_nodes=[2], path=SimpleNamespace(tokens=1))
    weights = torch.zeros(3, 1, dtype=torch.float64)
    scores, normalized = influence(weights, torch.tensor([2]), weights.new_tensor([0., 0., 1.]))
    displayed, edges, omitted, edge_omitted, connectivity = _prune(
        effects, weights, [2], scores, normalized, max_nodes=3, max_edges=3)
    assert len(displayed) == 1 and displayed[0]["kind"] == "logit"
    assert edges == [] and omitted == edge_omitted == 0
    assert connectivity["status"] == "unavailable"


def indirect_graph():
    nodes = [{"id": f"token:{i}", "kind": "token"} for i in range(2)]
    nodes += [{"id": f"feature:{i}", "kind": "feature"} for i in range(5)]
    logits = list(range(7, 17))
    nodes += [{"id": f"logit:{i}", "kind": "logit"} for i in logits]
    targets = logits + [2]
    weights = torch.zeros(len(nodes), len(targets), dtype=torch.float64, device="cpu")
    weights[2:7, 0] = .2
    weights[:2, -1] = .5
    seeds = weights.new_zeros(len(nodes))
    seeds[logits] = weights.new_tensor([.9] + [.1 / 9] * 9)
    scores, normalized = influence(weights, torch.tensor(targets), seeds)
    effects = SimpleNamespace(nodes=nodes, logit_nodes=logits, path=SimpleNamespace(tokens=2))
    return effects, weights, targets, scores, normalized


def has_token_path(nodes, edges):
    # Independent forward reachability, without the production reverse search.
    reached = {i for i, node in enumerate(nodes) if node["kind"] == "token"}
    pending = list(reached)
    while pending:
        source = pending.pop()
        for edge in edges:
            if edge["source"] == source and edge["target"] not in reached:
                reached.add(edge["target"])
                pending.append(edge["target"])
    return any(nodes[i]["id"] == "logit:7" for i in reached)


@pytest.mark.parametrize("max_nodes,max_edges,connected", [
    (512, 8192, True), (10, 8, True), (4, 2, True), (3, 2, True),
    (2, 2, False), (3, 1, False), (1, 0, False), (3, 0, False),
])
def test_feature_only_token_paths_survive_pruning_caps(max_nodes, max_edges, connected):
    effects, weights, targets, scores, normalized = indirect_graph()
    assert not weights[:2, :10].count_nonzero()
    nodes, edges, _, _, connectivity = _prune(effects, weights, targets, scores, normalized,
                                             max_nodes=max_nodes, max_edges=max_edges)
    assert len(nodes) <= max_nodes and len(edges) <= max_edges
    assert has_token_path(nodes, edges) == connected
    assert connectivity == {"status": "retained" if connected else "capped",
                            "minimum_nodes": 3, "minimum_edges": 2}
    if connected:
        assert {"token:0", "feature:0", "logit:7"} <= {node["id"] for node in nodes}
    if max_nodes > 1:
        assert any(node["kind"] != "logit" for node in nodes)
    original = {node["id"]: i for i, node in enumerate(effects.nodes)}
    columns = {target: column for column, target in enumerate(targets)}
    for node in nodes:
        assert node["influence"] == float(scores[original[node["id"]]])
    for edge in edges:
        source = original[nodes[edge["source"]]["id"]]
        target = original[nodes[edge["target"]]["id"]]
        assert edge["weight"] == float(weights[source, columns[target]])


def test_shorter_weak_path_takes_priority_over_stronger_long_path():
    nodes = [{"id": "token:0", "kind": "token"}]
    nodes += [{"id": f"feature:{i}", "kind": "feature"} for i in range(3)]
    nodes += [{"id": "logit:7", "kind": "logit"}]
    targets = [4, 1, 2, 3]
    weights = torch.zeros(5, 4, dtype=torch.float64, device="cpu")
    weights[2, 0] = 100
    weights[3, 0] = -.001
    weights[0, 1] = weights[1, 2] = 1
    weights[0, 3] = -.001
    seeds = weights.new_tensor([0., 0., 0., 0., 1.])
    scores, normalized = influence(weights, torch.tensor(targets), seeds)
    effects = SimpleNamespace(nodes=nodes, logit_nodes=[4], path=SimpleNamespace(tokens=1))
    displayed, edges, _, _, connectivity = _prune(
        effects, weights, targets, scores, normalized, max_nodes=3, max_edges=2)
    assert [node["id"] for node in displayed] == ["token:0", "feature:2", "logit:7"]
    assert has_token_path(displayed, edges)
    assert [edge["weight"] for edge in edges] == [-.001, -.001]
    assert connectivity == {"status": "retained", "minimum_nodes": 3, "minimum_edges": 2}


def test_path_reservation_is_deterministic_and_reports_public_caps():
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    from .test_trace import exact_dictionary

    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2).cpu()
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    graph = tracer.trace("A", max_targets=4, max_nodes=2, max_edges=1)
    assert graph.completeness["token_path"] == {"status": "retained", "minimum_nodes": 2, "minimum_edges": 1}
    assert {node["kind"] for node in graph.nodes} == {"token", "logit"}
    assert len(graph.edges) == 1
    tracer._cache.clear()
    repeated = tracer.trace("A", max_targets=4, max_nodes=2, max_edges=1)
    assert repeated.nodes == graph.nodes and repeated.edges == graph.edges
    assert repeated.trace_id == graph.trace_id
    capped = tracer.trace("A", max_targets=4, max_nodes=1, max_edges=0)
    assert capped.read()["completeness"]["token_path"] == {
        "status": "capped", "minimum_nodes": 2, "minimum_edges": 1}
