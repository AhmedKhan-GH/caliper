"""CPU checks for bounded finite scans and exact request admission."""
import math

import pytest
import torch

from caliper.interp import _frozen, _trace
from caliper.interp._memory import CleanStop
from .test_trace_safety import resources


def test_finite_batches_bound_masks_and_readbacks_for_strided_views(monkeypatch):
    value = torch.arange(324, dtype=torch.float64).reshape(18, 18).T[::2]
    finite, item = torch.isfinite, torch.Tensor.item
    sizes, readbacks, stages = [], [], []

    def bounded(block):
        sizes.append(block.numel())
        assert block.numel() <= 8
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        return finite(block)

    def readback(result):
        readbacks.append(result.numel())
        return item(result)

    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_ELEMENTS", 8)
    monkeypatch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 4)
    monkeypatch.setattr(torch, "isfinite", bounded)
    monkeypatch.setattr(torch.Tensor, "item", readback)
    _frozen._require_finite(value, "fixture", stages.append)
    assert sum(sizes) == value.numel()
    assert readbacks == [1] * math.ceil(len(sizes) / 4)
    assert stages.count("finite block") == stages.count("finite block complete") == len(sizes)
    assert stages.count("finite batch complete") == len(readbacks)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", [0, 7, 8, 16])
def test_finite_batches_reject_nonfinite_blocks_and_partial_tail(monkeypatch, bad, position):
    value = torch.ones(34, dtype=torch.float64)[::2]
    value[position] = bad
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_ELEMENTS", 1)
    monkeypatch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 8)
    with pytest.raises(ValueError, match="fixture must be finite"):
        _frozen._require_finite(value, "fixture")


@pytest.mark.parametrize("error", [CleanStop, _trace.TraceTimeout])
def test_finite_batches_keep_checks_inside_batch(monkeypatch, error):
    seen = []

    def check(stage):
        seen.append(stage)
        if seen.count("finite block") == 3:
            raise error("fixture stop")

    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_ELEMENTS", 1)
    monkeypatch.setattr(torch.Tensor, "item", lambda value: pytest.fail("stop followed a readback"))
    with pytest.raises(error, match="fixture stop"):
        _frozen._require_finite(torch.ones(8), "fixture", check)


def test_finite_batches_check_deadline_after_readback(monkeypatch):
    item = torch.Tensor.item
    copied = False

    def readback(value):
        nonlocal copied
        result = item(value)
        copied = True
        return result

    def check(stage):
        if copied:
            assert stage == "finite batch complete"
            raise _trace.TraceTimeout("fixture readback")

    monkeypatch.setattr(torch.Tensor, "item", readback)
    with pytest.raises(_trace.TraceTimeout, match="fixture readback"):
        _frozen._require_finite(torch.ones(8), "fixture", check)


def test_finite_batches_preserve_exact_graph(monkeypatch):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_ELEMENTS", 8)
    with monkeypatch.context() as patch:
        patch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 1)
        reference = tracer.trace("AB", max_targets=4)
    tracer._cache.clear()
    tracer._forward_cache = None
    batched = tracer.trace("AB", max_targets=4)
    assert reference.trace_id == batched.trace_id
    assert reference.nodes == batched.nodes and reference.edges == batched.edges
    assert reference.completeness == batched.completeness
    assert reference.tokens == batched.tokens
    for name, tensor in reference.tensors.items():
        assert torch.equal(tensor, batched.tensors[name])
