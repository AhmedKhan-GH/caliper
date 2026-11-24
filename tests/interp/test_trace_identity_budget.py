"""Trace deadlines and lower limits cover every exact identity hash chunk."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _frozen, _publish, _trace
from caliper.interp._memory import CleanStop
from .test_gemma_runtime import gemma_model, local_gemma
from .test_memory_phases import process_guard
from .test_standalone_budgets import Tokenizer
from .test_trace import random_dictionary
from .test_trace_safety import resources


@pytest.mark.parametrize("family", ["tiny", "gemma"])
@pytest.mark.parametrize("fault", ["deadline", "lower"])
def test_trace_stops_inside_model_hash_and_preserves_prior_publication(tmp_path, monkeypatch, family, fault):
    readings = {"current": 1_000, "peak": 1_000}
    supplied = process_guard(12_000_000_000, readings)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", lambda limit, device, **kwargs:
                        process_guard(int(limit * 1e9), readings))
    if family == "gemma":
        model = gemma_model()
        _dictionary.verify_hf_model(model, local_gemma(tmp_path, model))
        tokenizer, dictionary = Tokenizer(), random_dictionary(model, latents=8, k=2)
    else:
        model, tokenizer, dictionary = resources()
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(_publish, "_TENSORS", {})
    tracer = _trace.Tracer(model, tokenizer, dictionary, memory=supplied)
    prior = tracer.trace("A", max_targets=0)
    cache, published, forward_cache = dict(tracer._cache), dict(_publish._TENSORS), tracer._forward_cache
    name = sorted(model.state_dict())[0]
    tensor = model.state_dict()[name]
    assert tensor.numel() * tensor.element_size() > 256
    pointer = tensor.data_ptr()
    convert = torch.Tensor.to
    chunks = []
    def expire(value, *args, **kwargs):
        result = convert(value, *args, **kwargs)
        if (value.ndim == 1 and pointer <= value.data_ptr() < pointer + tensor.numel() * tensor.element_size()
                and kwargs.get("device") == "cpu"):
            chunks.append(value.numel())
            if fault == "deadline":
                clock.now += 20
            else:
                supplied.limit = readings["current"] + 4 * _trace._HASH_CHUNK_BYTES - 1
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 256)
        patch.setattr(torch.Tensor, "to", expire)
        patch.setattr(model, "forward", lambda *a, **kw: pytest.fail("identity stop reached the forward"))
        with pytest.raises(_trace.TraceTimeout if fault == "deadline" else CleanStop):
            tracer.trace("B", max_targets=0)
    assert chunks == [256 // tensor.element_size()]
    assert tracer._cache == cache and tracer._forward_cache is forward_cache
    assert all(_publish._TENSORS[key] is value for key, value in published.items())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    if fault == "deadline":
        assert tracer.trace("A", max_targets=0) is prior


def test_model_identity_checks_deadline_after_final_chunk(monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    seen = []
    copied = False
    convert = torch.Tensor.to
    def copy(value, *args, **kwargs):
        nonlocal copied
        result = convert(value, *args, **kwargs)
        copied = True
        return result
    def check(stage):
        seen.append(stage)
        if copied:
            raise _trace.TraceTimeout("fixture final chunk")
    monkeypatch.setattr(torch.Tensor, "to", copy)
    with pytest.raises(_trace.TraceTimeout, match="final chunk"):
        _dictionary.model_identity(model, check=check)
    assert copied and seen == ["model_identity"] * 3


def test_direct_frozen_path_checks_supplied_guard_before_model_hash(monkeypatch):
    model, _, dictionary = resources()
    memory = process_guard(100_000_000, {"current": 1_000, "peak": 1_000})
    memory.limit = 1_000 + 4 * _dictionary._HASH_CHUNK_BYTES - 1
    monkeypatch.setattr(torch.Tensor, "to", lambda *a, **k: pytest.fail("hash copy preceded admission"))
    with pytest.raises(CleanStop, match="model identity allocation"):
        _frozen.FrozenPath(model, torch.tensor([[1, 2]]), dictionary, guard=memory)
