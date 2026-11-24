"""Generic source copies precede mutation and obey allocation limits."""
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from caliper.interp import _intervene
from caliper.interp._intervene import InterventionError, InterventionService
from caliper.interp._memory import CleanStop
from .test_generic_patch_sources import SPEC, tokenizer
from .test_intervention_replay import hook_objects
from .test_native_source_capture import NativeActivationModel


HIDDEN = dict(SPEC, point="hidden")
SOURCE_BYTES = 1 * 4 * 2 * 4


@pytest.fixture
def native_source():
    model = NativeActivationModel(nn.LeakyReLU(0.1, inplace=True)).eval()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    hooks = hook_objects(model)
    try:
        yield model, service
    finally:
        service.close()
    assert not service._cache and service._cache_identity is None
    assert not service.preparing_source and hook_objects(model) == hooks


def test_intermediate_patch_matches_clean_forward_before_inplace_activation(native_source, monkeypatch):
    model, service = native_source
    with torch.no_grad():
        expected_hidden = model.hidden(model.embed(tokenizer("source")["input_ids"]))
        expected = model(**tokenizer("source"))
        baseline = model(**tokenizer("target"))
    assert (expected_hidden < 0).any() and (expected < 0).any()
    assert not torch.equal(expected_hidden, expected)
    assert not torch.equal(baseline, expected)
    clones, clone = [], torch.Tensor.clone

    def record_copy(tensor, *args, **kwargs):
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            clones.append(weakref.ref(result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", record_copy)
        hook = service.install(HIDDEN)
    cached = service._cache["source", "hidden"]
    torch.testing.assert_close(cached, expected_hidden, rtol=0, atol=0)
    assert len(clones) == 1 and clones[0]() is cached
    with torch.no_grad():
        actual = model(**tokenizer("target"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cached, expected_hidden, rtol=0, atol=0)
    assert service._source(HIDDEN) is cached and hook.error is None


@pytest.mark.parametrize("bound", ["bytes", "entries"])
def test_cache_admission_precedes_copy_and_downstream_forward(native_source, monkeypatch, bound):
    model, service = native_source
    field = "MAX_SOURCE_BYTES" if bound == "bytes" else "MAX_SOURCE_ENTRIES"
    monkeypatch.setattr(_intervene, field, SOURCE_BYTES - 1 if bound == "bytes" else 0)
    clone = torch.Tensor.clone

    def forbid_copy(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "source copy preceded cache admission"
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", forbid_copy)
    downstream = []
    observer = model.nonlinearity.register_forward_pre_hook(lambda *args: downstream.append(True))
    try:
        with pytest.raises(InterventionError, match="bounded source cache"):
            service.install(HIDDEN)
        assert not downstream and not service._cache and not service.preparing_source
        assert not model.hidden._forward_hooks and not service._cache_identity[1]
    finally:
        observer.remove()


class CaptureBudget:
    """Expose a lower outer or inner limit and stale allocation readings."""

    def __init__(self, lower, capacity):
        low = 8 * 1024 * 1024
        self.limit = low if lower == "outer" else 2 * low
        self._guard = SimpleNamespace(limit=low if lower == "inner" else 2 * low, peak=0, current=0)
        self.device = torch.device("cpu")
        self.peak = self.current = 0
        self.baseline = low - capacity
        self.captures = 0

    def check(self, stage):
        if stage == "patch source capture":
            self.captures += 1
            self.current = self.baseline
            self.peak = max(self.peak, self.current)
        return self.current


@pytest.mark.parametrize("lower", ["outer", "inner"])
def test_lower_process_budget_refuses_before_copy_and_accepts_exact_capacity(native_source, monkeypatch, lower):
    model, service = native_source
    guard = CaptureBudget(lower, SOURCE_BYTES - 1)
    service.resource_guard = guard
    clone = torch.Tensor.clone

    def forbid_copy(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "source copy preceded process admission"
        return clone(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbid_copy)
        with pytest.raises(CleanStop, match="patch source allocation exceeds"):
            service.install(HIDDEN)
    assert guard.captures == 1
    assert not service._cache and not service._cache_identity[1] and not service.preparing_source
    assert not any(pre or post for pre, post in hook_objects(model))
    # Restore the prior phase reading before the next content hash.
    guard.current = 0
    selected = guard if lower == "outer" else guard._guard
    selected.limit += 1
    source = service._source(HIDDEN)
    with torch.no_grad():
        expected = model.hidden(model.embed(tokenizer("source")["input_ids"]))
    torch.testing.assert_close(source, expected, rtol=0, atol=0)
    assert guard.captures == 2


@pytest.mark.parametrize("lower", ["outer", "inner"])
def test_multiple_captures_reserve_prior_copies_and_publish_no_partial_cache(native_source, monkeypatch, lower):
    model, service = native_source
    service.install(HIDDEN)
    service.install(SPEC)
    hooks = hook_objects(model)
    service.invalidate_cache()
    guard = CaptureBudget(lower, 2 * SOURCE_BYTES - 1)
    service.resource_guard = guard
    copies, clone = [], torch.Tensor.clone

    def record_copy(tensor, *args, **kwargs):
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            copies.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", record_copy)
    with pytest.raises(CleanStop, match="patch source allocation exceeds"):
        service._source(SPEC)
    assert guard.captures == 2 and len(copies) == 1
    assert copies[0]() is None
    assert not service._cache and not service._cache_identity[1] and not service.preparing_source
    assert hook_objects(model) == hooks


@pytest.mark.parametrize("failure", ["forward", "copy", "nonfinite"])
def test_failed_capture_preserves_prior_cache_and_releases_temporary_hooks(native_source, monkeypatch, failure):
    model, service = native_source
    original = service._source(SPEC)
    evidence = service._cache_identity
    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 1)
    copies, clone = [], torch.Tensor.clone

    def record_copy(tensor, *args, **kwargs):
        if tensor.ndim == 3 and failure == "copy":
            raise RuntimeError("source copy failed")
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            copies.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", record_copy)
    if failure == "forward":
        def fail(*args, **kwargs):
            raise RuntimeError("downstream forward failed")
        monkeypatch.setattr(model.head, "forward", fail)
    elif failure == "nonfinite":
        monkeypatch.setattr(model.hidden, "forward", lambda values: values * float("nan"))
    with pytest.raises((RuntimeError, InterventionError)):
        service._source(HIDDEN)
    assert list(service._cache) == [("source", "head")]
    assert service._cache["source", "head"] is original and service._cache_identity is evidence
    assert all(reference() is None for reference in copies)
    assert not any(pre or post for pre, post in hook_objects(model))
    assert not service.preparing_source
