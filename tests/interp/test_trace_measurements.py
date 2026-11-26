"""Complete trace diagnostics with a hang timeout and no error cutoff."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from caliper.interp import Tracer, _publish, _trace
from .test_trace_safety import resources


def test_trace_deadline_uses_the_full_safety_interval(monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    deadline = _trace._Deadline()
    assert deadline.expires == 700.
    clock.now = 699.
    deadline("complete source account")
    clock.now = 700.
    with pytest.raises(_trace.TraceTimeout, match="600 second safety timeout"):
        deadline("complete source account")


def test_trace_preserves_large_residual_measurements(monkeypatch):
    model, tokenizer, dictionary = resources()
    baseline = Tracer(model, tokenizer, dictionary).trace("AB", max_targets=0)
    backward = _trace._DirectEffects.backward
    records = []

    def changed(self, indices):
        weights, diagnostics = backward(self, indices)
        for row in diagnostics:
            row["expected"] += 1.
            row["absolute_residual"] = abs(math.fsum(
                [row["incoming"], *row["biases"].values(), -row["expected"]]))
            row["relative_residual"] = row["absolute_residual"] / max(1., abs(row["expected"]))
        records.extend(deepcopy(diagnostics))
        return weights, diagnostics

    monkeypatch.setattr(_trace._DirectEffects, "backward", changed)
    graph = Tracer(model, tokenizer, dictionary).trace("AB", max_targets=0)
    assert graph.nodes == baseline.nodes and graph.edges == baseline.edges
    assert graph.completeness["targets"] == records
    assert graph.completeness["max_absolute_residual"] > .5
    assert graph.completeness["numeric_status"] == "measured"
    groups = graph.completeness["forward_residuals"]
    assert set(groups) == {"direct_reads", "direct_logits", "replacement_reads", "replacement_logits"}
    assert len(groups["direct_reads"]) == len(groups["replacement_reads"]) == len(model.layers)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_residual_keeps_prior_publication(monkeypatch, value):
    model, tokenizer, dictionary = resources()
    monkeypatch.setattr(_publish, "_TENSORS", {})
    Tracer(model, tokenizer, dictionary).trace("AB", max_targets=0)
    published = dict(_publish._TENSORS)
    backward = _trace._DirectEffects.backward

    def changed(self, indices):
        weights, diagnostics = backward(self, indices)
        diagnostics[0]["absolute_residual"] = value
        return weights, diagnostics

    monkeypatch.setattr(_trace._DirectEffects, "backward", changed)
    tracer = Tracer(model, tokenizer, dictionary)
    with pytest.raises(ValueError, match="finite"):
        tracer.trace("AB", max_targets=0)
    assert not tracer._cache
    assert all(_publish._TENSORS[key] is tensor for key, tensor in published.items())
