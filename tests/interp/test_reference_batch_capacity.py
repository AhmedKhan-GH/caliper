"""Bounded CPU checks for reference queues and physical memory admission."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _reference_graph
from caliper.interp._frozen import MemoryGuard
from caliper.interp._memory import CleanStop
from .test_direct_decoder_cache import fixture, oracle


OPTIONS = dict(layers=26, tokens=11, width=2304, mixer_width=2304, itemsize=2)
# The existing model keeps float32 work, twelve residual-width buffers per
# layer, eight mixer-width buffers per layer, and fixed allocator headroom.
PER_TARGET = 4 * 11 * (2304 * 27 * 12 + 2304 * 26 * 8)


class ScalarGuard(MemoryGuard):
    """Use actual admission arithmetic with deterministic scalar readers."""

    def __init__(self, current):
        self.device = torch.device("cpu")
        self.limit = 40_000_000_000
        self._host_reserve_bytes = 8_000_000_000
        self.current = current
        self.requests, self.reserves = [], []

    def check(self, stage):
        return self.current

    def reserve(self, stage, size):
        super().reserve(stage, size)
        self.reserves.append((stage, size))

    def batch_size(self, requested, **kwargs):
        result = super().batch_size(requested, **kwargs)
        self.requests.append((requested, result))
        return result


@pytest.mark.parametrize("capacity", [1, 64, 129, 512])
@pytest.mark.parametrize("bound", ["process", "host"])
def test_reference_capacity_preserves_conservative_factors(monkeypatch, capacity, bound):
    current = 40_000_000_000 - 512_000_000 - capacity * PER_TARGET if bound == "process" else 0
    available = 8_000_000_000 + 512_000_000 + capacity * PER_TARGET if bound == "host" else 64_000_000_000
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    guard = ScalarGuard(current)
    assert guard.batch_size(512, **OPTIONS) == capacity
    assert guard.batch_size(1, **OPTIONS) == 1
    # One byte below a one-target capacity must refuse before any tensor work.
    guard.current = 40_000_000_000 - 512_000_000 - PER_TARGET + 1
    with pytest.raises(CleanStop, match="no target batch fits"):
        guard.batch_size(512, **OPTIONS)


def test_reference_capacity_uses_the_prospective_host_reserve(monkeypatch):
    available = 15_515_828_224
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    guard = ScalarGuard(25_344_707_464)
    assert PER_TARGET == 53_932_032
    assert guard.batch_size(512, **OPTIONS) == 129
    available = 8_000_000_000 + 512_000_000 + PER_TARGET - 1
    with pytest.raises(CleanStop, match="no target batch fits"):
        guard.batch_size(512, **OPTIONS)


class GraphEffects:
    """A token, feature, and logit DAG with explicit BOS projection rows."""

    def __init__(self, capacity):
        self.path = SimpleNamespace(**{k: v for k, v in OPTIONS.items() if k != "itemsize"},
            device=torch.device("cpu"), embedding=torch.zeros(1, dtype=torch.bfloat16),
            guard=ScalarGuard(40_000_000_000 - 512_000_000 - capacity * PER_TARGET))
        self.feature_nodes = list(range(521))
        self.nodes = [{"id": f"feature:{i}", "kind": "feature", "position": int(i != 0)} for i in range(521)]
        self.nodes += [{"id": "error:bos", "kind": "error", "position": 0},
                       {"id": "error:1", "kind": "error", "position": 1},
                       {"id": "token:1", "kind": "token", "position": 1},
                       {"id": "logit:1", "kind": "logit", "position": 1, "probability": 1.}]
        self.logit_nodes = [524]
        self.prepared = False
        self.calls = []

    def prepare_decoders(self):
        self.prepared = True

    def release_decoders(self):
        self.prepared = False

    def backward(self, indices):
        assert self.prepared
        assert len(indices) <= self.path.guard.requests[-1][1]
        self.calls.append(indices)
        incoming = torch.zeros(len(self.nodes), len(indices), dtype=torch.bfloat16)
        # Every target includes these BOS rows in its full-source diagnostics.
        incoming[0] = 2.
        incoming[521] = 4.
        incoming[522] = .25
        incoming[523] = .5
        for column, target in enumerate(indices):
            if target == 524:
                incoming[1:521, column] = torch.arange(1, 521).to(incoming.dtype)
        diagnostics = [{"id": self.nodes[i]["id"], "incoming": float(incoming[:, col].float().sum()),
                        "biases": {"encoder": .125}} for col, i in enumerate(indices)]
        return incoming, diagnostics


def build(effects, max_targets=515):
    return _reference_graph.build(effects, max_targets=max_targets, max_nodes=8192, max_edges=262144,
        batch_size=512, update_interval=1, node_threshold=.7, edge_threshold=.9, check=lambda stage: None)


@pytest.mark.parametrize("max_targets", [515, 520])
def test_reference_graph_keeps_queues_edges_and_evidence_when_capacity_splits(monkeypatch, max_targets):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=64_000_000_000))
    full, split = GraphEffects(512), GraphEffects(64)
    reference, divided = build(full, max_targets), build(split, max_targets)
    # Pruned union edges, all source diagnostics, target order, and logical queues match.
    assert reference[:3] == divided[:3]
    assert reference[4] == divided[4]
    assert reference[3] == [1, 512, max_targets - 512]
    assert max(divided[3]) == 64
    assert full.path.guard.requests[1] == (512, 512)
    assert split.path.guard.requests[1] == (512, 64)
    assert len(reference[1]) == max_targets + 1
    assert all(row["incoming"] > row["reference_projection"]["selected_source_incoming"] for row in reference[1])
    assert all(row["biases"] == {"encoder": .125} for row in reference[1])
    assert not full.prepared and not split.prepared


def test_reference_graph_refuses_before_backward_when_no_target_fits(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=64_000_000_000))
    effects = GraphEffects(0)
    with pytest.raises(CleanStop, match="no target batch fits"):
        build(effects)
    assert effects.calls == []


def test_reference_direct_batch_allows_512_and_keeps_the_ordinary_limit():
    effects = fixture()
    targets = effects.logit_nodes * 256
    actual, diagnostics = effects.backward(targets)
    expected = oracle(effects, effects.logit_nodes).repeat(1, 256)
    assert torch.equal(actual, expected)
    assert len(diagnostics) == 512
    with pytest.raises(ValueError, match="target batch"):
        effects.backward(targets + targets[:1])
    effects.reference = False
    with pytest.raises(ValueError, match="target batch"):
        effects.backward(targets[:65])
