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
@pytest.mark.parametrize("max_nodes", [3, 7])
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
    displayed, edges, omitted, _ = _prune(effects, weights, logits, scores, normalized,
                                         max_nodes=max_nodes, max_edges=8)
    kept = [n["id"] for n in displayed if n["kind"] == "feature"]
    count = min(3, max_nodes - len(logits))
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
    displayed, edges, omitted, edge_omitted = _prune(
        effects, weights, [2], scores, normalized, max_nodes=3, max_edges=3)
    assert len(displayed) == 1 and displayed[0]["kind"] == "logit"
    assert edges == [] and omitted == edge_omitted == 0
