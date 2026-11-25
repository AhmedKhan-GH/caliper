"""CPU identity admission checks at caller boundaries."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _frozen, _gemma_import_identity, _hosted, _trace
from caliper.interp._intervene import InterventionService
from caliper.interp._memory import CleanStop
from tools.interp.evaluation_binding import EvaluationBinding
from .test_gemma_runtime import gemma_model, local_gemma
from .test_memory_phases import process_guard
from .test_service_memory import RequestGuard
from .test_trace import random_dictionary
from .test_trace_safety import resources


@pytest.fixture
def cpu_memory(monkeypatch):
    def create(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        return process_guard(int(limit * 1e9), {"current": 1_000, "peak": 1_000})
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", create)
    monkeypatch.setattr(_dictionary, "MemoryGuard", create)


def hosted(model, guard):
    run = _hosted.HostedInference.__new__(_hosted.HostedInference)
    run.model, run.guard = model, guard
    run._failure = run._verified_bindings = run.service = None
    run.model_name, run.dictionary_path = "cpu-fixture", "fixture"
    run.index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"})
    return run


def forbid_work(monkeypatch, model):
    def fail(*args, **kwargs):
        pytest.fail("identity refusal followed a readback, forward, or new hook")
    monkeypatch.setattr(torch.Tensor, "to", fail)
    monkeypatch.setattr(model, "forward", fail)
    monkeypatch.setattr(torch.nn.Module, "register_forward_hook", fail)
    monkeypatch.setattr(torch.nn.Module, "register_forward_pre_hook", fail)


@pytest.mark.parametrize("family,route", [
    (family, route) for family in ("tiny", "verified_gemma")
    for route in ("configure", "evaluation", "hosted", "import")
    if family != "tiny" or route != "import"
])
@pytest.mark.parametrize("fault", ["lower", "stop", "historical"])
def test_identity_callers_refuse_before_tensor_reads(tmp_path, monkeypatch, cpu_memory, family, route, fault):
    if family == "verified_gemma":
        model = gemma_model()
        _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=40)
        dictionary = random_dictionary(model, latents=8, k=2)
    else:
        model, _, dictionary = resources()
    memory = RequestGuard()
    stopped = CleanStop("caller identity stop")
    if fault == "lower":
        memory._guard.limit = 4_000_000
    elif fault == "stop":
        memory.failure = stopped
    else:
        memory._guard.peak = memory.limit + 1
    service = InterventionService(model)
    run = hosted(model, _hosted._RunGuard(memory))
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop) as caught:
        if route == "configure":
            service.configure(dictionary=dictionary, dictionary_id="caller-owned", resource_guard=memory)
        elif route == "evaluation":
            EvaluationBinding(tmp_path, dictionary, "fixture", model=model, memory=memory)
        elif route == "hosted":
            run._bindings()
        else:
            _gemma_import_identity.verified_import_binding(tmp_path, dictionary.metadata, model=model, memory=memory)
    if fault == "stop":
        assert caught.value is stopped
    assert memory.stages == ["model_identity"]
    assert service.dictionary is service.resource_guard is None
    assert run._verified_bindings is None
    if route == "hosted":
        assert run._failure is caught.value
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_hosted_display_guard_stop_precedes_first_binding(monkeypatch):
    model, _, _ = resources()
    model_guard, display_guard = RequestGuard(), RequestGuard()
    failure = CleanStop("display guard stop")
    display_guard.failure = failure
    run = hosted(model, _hosted._RunGuard(model_guard, extra=display_guard))
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop) as caught:
        run._bindings()
    assert caught.value is run._failure is failure
    assert model_guard.stages == display_guard.stages == ["model_identity"]
    assert run._verified_bindings is None


@pytest.mark.parametrize("route", ["constructor", "request", "frozen"])
@pytest.mark.parametrize("supplied", [False, True])
def test_trace_identity_receives_effective_cpu_guard(monkeypatch, cpu_memory, route, supplied):
    model, tokenizer, dictionary = resources()
    memory = RequestGuard(1_000_000_000)
    tracer = _trace.Tracer(model, tokenizer, dictionary, memory=memory if supplied else None)
    identify, seen = _frozen.model_identity, []
    limit = 1_000_000_000 if supplied else 12_000_000_000

    def identity(value, **kwargs):
        active = kwargs["memory"]
        assert active is not None and active.device == torch.device("cpu")
        assert active.limit == limit and kwargs["memory_limit_gb"] == limit / 1e9
        seen.append(active)
        return identify(value, **kwargs)

    monkeypatch.setattr(_frozen, "model_identity", identity)
    if route == "constructor":
        _trace.Tracer(model, tokenizer, dictionary, memory=memory if supplied else None)
    elif route == "request":
        tracer.trace("A", max_targets=0)
    else:
        _frozen.FrozenPath(model, torch.tensor([[1, 2]]), dictionary, guard=tracer.guard if supplied else None)
    assert len(seen) == 1


def test_default_frozen_guard_refuses_before_identity_readback(monkeypatch, cpu_memory):
    model, _, dictionary = resources()
    ids = torch.tensor([[1, 2]])
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop, match="model identity allocation"):
        _frozen.FrozenPath(model, ids, dictionary, memory_limit_gb=.004)


def test_frozen_explicit_lower_identity_budget_precedes_readback(monkeypatch, cpu_memory):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    ids = torch.tensor([[1, 2]])
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop, match="model identity allocation"):
        _frozen.FrozenPath(model, ids, dictionary, guard=tracer.guard, memory_limit_gb=.004)


@pytest.mark.parametrize("family", ["tiny", "verified_gemma"])
def test_hosted_cpu_bindings_recheck_alias_changes(tmp_path, monkeypatch, cpu_memory, family):
    if family == "verified_gemma":
        model = gemma_model()
        _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=40)
    else:
        model, _, _ = resources()
    run = hosted(model, _hosted._RunGuard(RequestGuard()))
    first = run._bindings()
    parameter = next(model.parameters())
    version = parameter._version
    parameter.data.view(-1)[0].add_(.125)
    assert parameter._version == version
    if family == "verified_gemma":
        with pytest.raises(ValueError, match="identity mismatch"):
            run._bindings()
        assert run._verified_bindings == first
    else:
        assert run._bindings()["model_hash"] != first["model_hash"]


def test_verified_cpu_identity_retains_loader_stop(tmp_path, monkeypatch, cpu_memory):
    model = gemma_model()
    _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=40)
    dictionary = random_dictionary(model, latents=8, k=2)
    loader_memory = _dictionary._VERIFIED_MODELS[model].memory
    loader_memory.peak = loader_memory.limit + 1
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop, match="memory limit"):
        InterventionService(model).configure(dictionary=dictionary, dictionary_id="caller-owned",
                                             resource_guard=RequestGuard())


def test_verified_cpu_identity_retains_loader_budget_without_extra_guard(tmp_path, cpu_memory):
    model = gemma_model()
    _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=40)
    dictionary = random_dictionary(model, latents=8, k=2)
    loader_memory = _dictionary._VERIFIED_MODELS[model].memory
    # This scalar tests budget precedence. The model and all tensors remain small CPU fixtures.
    loader_memory.peak = 13_000_000_000
    service = InterventionService(model)
    service.configure(dictionary=dictionary, dictionary_id="caller-owned")
    assert service.dictionary is dictionary
    assert hosted(model, None)._bindings()["model_hash"] == dictionary.metadata["model_hash"]
    _frozen._validate_dictionary(model, dictionary)


def test_identity_caller_caps_supplied_budget_at_family_limit(monkeypatch):
    model, _, dictionary = resources()
    memory = RequestGuard(40_000_000_000)
    memory._guard.peak = 13_000_000_000
    forbid_work(monkeypatch, model)
    with pytest.raises(CleanStop, match="memory limit"):
        InterventionService(model).configure(dictionary=dictionary, dictionary_id="caller-owned",
                                             resource_guard=memory)
