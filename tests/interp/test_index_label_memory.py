"""CPU label admission with scalar memory limits and small tensor fixtures."""
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
import torch

from caliper.interp import _index
from caliper.interp._memory import BudgetMemory, CleanStop, MemoryGuard


@pytest.fixture
def label_memory(monkeypatch):
    # Gemma dimensions select the family reserve. All tensors remain small CPU fixtures.
    output = torch.tensor([[0., 4., 0.], [2., 0., 0.], [0., 4., 0.], [2., 1., 0.],
                           [0., 4., 0.], [0., 2., 0.], [2., 4., 0.]], dtype=torch.float64)
    readings = {"current": 1000, "peak": 1000, "available": 20_000_000_000}
    supplied = MemoryGuard(40, "cpu", resident=lambda: readings["current"],
                           footprint=lambda: readings["current"], peak=lambda: readings["peak"],
                           driver=lambda: 0, pressure=lambda: 50)
    supplied.host_reserve_bytes = 0
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=readings["available"]))
    index = object.__new__(_index.FeatureIndex)
    index._closed = False
    index._cache, index._summaries = OrderedDict(), OrderedDict()
    index._label_identity = ("caliper.feature-labels.v1", "fixture-index")
    index.model = SimpleNamespace(
        training=False, modules=lambda: (), get_output_embeddings=lambda: SimpleNamespace(weight=output),
        config=SimpleNamespace(model_type="gemma2", num_hidden_layers=26, hidden_size=2304))
    index.dictionary = SimpleNamespace(layers=[SimpleNamespace(W_dec=torch.eye(3, dtype=torch.float64))])
    index.metadata = {"layers": 1, "features": 3}
    index.memory = BudgetMemory(supplied, 40)
    index._token = lambda token: f"#{token}"
    index.table = np.zeros((1, 3, _index.TOP_EXAMPLES), dtype=_index._TABLE)
    index.table["activation"][0, :, 0] = [3, 2, 1]
    index.table["offset"][0, :, 0] = [0, 1, 2]
    index.counts = np.ones((1, 3), dtype=np.uint64)
    index.tokens = np.array([0, 1, 2], dtype=np.int32)
    index.documents = np.array([(b"fixture", 0, 3, False)], dtype=_index._DOCUMENT)
    yield index, supplied, readings
    index.close()


def workspace(count):
    # The fixture has three decoder columns, seven vocabulary rows, and eight-byte values.
    return count * (4 * 3 + 16 * 7) * 8 + 64_000_000


def refuse(*args, **kwargs):
    pytest.fail("label admission must precede allocation and projection")


def forbid_allocations(monkeypatch):
    monkeypatch.setattr(np, "empty", refuse)
    monkeypatch.setattr(torch, "stack", refuse)
    monkeypatch.setattr(torch.Tensor, "to", refuse)
    monkeypatch.setattr(torch.Tensor, "__matmul__", refuse)


@pytest.mark.parametrize("caller_reserve", [0, 12_000_000_000])
@pytest.mark.parametrize("query", ["labels", "show"])
def test_prospective_host_refusal_precedes_allocation(label_memory, monkeypatch, caller_reserve, query):
    index, supplied, readings = label_memory
    # Change the supplied guard after its BudgetMemory adapter exists.
    supplied.host_reserve_bytes = caller_reserve
    reserve = max(caller_reserve, 8_000_000_000)
    readings["available"] = reserve + workspace(1) - 1
    forbid_allocations(monkeypatch)
    with pytest.raises(CleanStop, match="feature label allocation.*host reserve"):
        index.labels([(0, 0)]) if query == "labels" else index.show(0, 0)
    assert index._summaries == {}


@pytest.mark.parametrize("caller_reserve", [0, 12_000_000_000])
def test_exact_host_and_process_boundaries_preserve_results(label_memory, caller_reserve):
    index, supplied, readings = label_memory
    supplied.host_reserve_bytes = caller_reserve
    readings["available"] = max(caller_reserve, 8_000_000_000) + workspace(2)
    supplied.limit = readings["current"] + workspace(2)
    assert index.labels([(0, 0), (0, 1), (0, 0)]) == {
        (0, 0): "Output: #1 | #3 | #6 / Examples: #0 | #1 | #2",
        (0, 1): "Output: #0 | #2 | #4 / Examples: #0 | #1 | #2",
    }
    assert index.show(0, 0)["promoted_tokens"] == [
        {"token_id": token, "token": f"#{token}", "score": 2.0} for token in (1, 3, 6)]
    assert index.label_identity == ("caliper.feature-labels.v1", "fixture-index")


@pytest.mark.parametrize("owner", ["caller", "adapter"])
def test_process_workspace_cap_precedes_allocation(label_memory, monkeypatch, owner):
    index, supplied, readings = label_memory
    guard = supplied if owner == "caller" else index.memory
    guard.limit = readings["current"] + workspace(1) - 1
    forbid_allocations(monkeypatch)
    with pytest.raises(CleanStop, match="feature label allocation.*process memory limit"):
        index.labels([(0, 0)])
    assert index._summaries == {}


def test_guard_failure_remains_latched_before_allocation(label_memory, monkeypatch):
    index, supplied, readings = label_memory
    readings["peak"] = supplied.limit + 1
    forbid_allocations(monkeypatch)
    with pytest.raises(CleanStop, match="memory limit"):
        index.labels([(0, 0)])
    readings["peak"] = readings["current"]
    with pytest.raises(CleanStop, match="memory limit"):
        index.labels([(0, 0)])
    assert supplied.peak > supplied.limit
    assert index._summaries == {}


def test_cached_summaries_need_no_projection_reserve(label_memory, monkeypatch):
    index, supplied, readings = label_memory
    expected = index.labels([(0, 0), (0, 1)])
    summary = index.show(0, 0)
    readings["available"] = 0
    supplied.limit = 1
    forbid_allocations(monkeypatch)
    monkeypatch.setattr(psutil, "virtual_memory", refuse)
    monkeypatch.setattr(index, "_promoted_many", refuse)
    assert index.labels([(0, 1), (0, 0)]) == expected
    assert index.show(0, 0) == summary


@pytest.mark.parametrize("change", ["available", "caller_reserve", "process_cap", "historical_peak"])
def test_late_chunk_refusal_preserves_prior_cache(label_memory, monkeypatch, change):
    index, supplied, readings = label_memory
    index.labels([(0, 2), (0, 0)])
    prior = deepcopy(list(index._summaries.items()))
    index._cache[0, 2] = {"examples": ["prior fixture"]}
    prior_windows = deepcopy(index._cache)
    promoted = index._promoted_many
    monkeypatch.setattr(index, "_promoted_many", lambda features, **kwargs: promoted(
        features, chunk_size=3, **kwargs))
    projections, chunks = [], []
    matmul = torch.Tensor.__matmul__

    def project(left, right):
        projections.append(tuple(left.shape))
        return matmul(left, right)

    def check(stage):
        if stage != "feature label chunk":
            return
        chunks.append(stage)
        if len(chunks) == 2:
            if change == "available":
                readings["available"] = 8_000_000_000 + workspace(1) - 1
            elif change == "caller_reserve":
                supplied.host_reserve_bytes = readings["available"] - workspace(1) + 1
            elif change == "process_cap":
                supplied.limit = readings["current"] + workspace(1) - 1
            else:
                supplied.peak = supplied.limit + 1

    monkeypatch.setattr(torch.Tensor, "__matmul__", project)
    reason = "host reserve" if change in ("available", "caller_reserve") else "process memory limit"
    with pytest.raises(CleanStop, match=reason):
        index.labels([(0, 0), (0, 1)], check=check)
    assert len(chunks) == 2 and projections == [(3, 3)]
    assert list(index._summaries.items()) == prior
    assert index._cache == prior_windows


def test_request_cap_precedes_projection(label_memory, monkeypatch):
    index, _, _ = label_memory
    forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="at most 512 requests"):
        index.labels([(0, 0)] * 513)
    assert index._summaries == {}
