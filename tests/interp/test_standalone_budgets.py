"""Explicit standalone budgets with small CPU tensors and coupled memory readers."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from caliper._controls import Controls
from caliper.interp import _dictionary, _frozen, _index, _trace, _validate
from caliper.interp._memory import BudgetMemory, CleanStop, family_memory_limit
from caliper.interp._output import forward_logits
from caliper.interp._points import capture
from caliper.interp._service import register_service, unregister_service
from .test_gemma_runtime import gemma_model, local_gemma
from .test_memory_phases import process_guard
from .test_trace import random_dictionary
from .test_trace_safety import resources


class Tokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 3, 5]])}

    def convert_ids_to_tokens(self, token_id):
        return str(token_id)

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture
def gemma_resources(tmp_path, monkeypatch):
    # Scalars model a released transfer peak and a live model plus dictionary.
    # All actual tensors use the small random CPU configuration.
    readings = {"current": 18_312_752_128, "peak": 24_000_000_000}
    guards = []
    def guard(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        result = process_guard(int(limit * 1e9), readings)
        guards.append(result)
        return result
    for module in (_dictionary, _index, _validate):
        monkeypatch.setattr(module, "MemoryGuard", guard)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", guard)
    model = gemma_model()
    _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=28)
    dictionary = random_dictionary(model, latents=8, k=2)
    for layer in dictionary.layers:
        layer.b_enc.data.fill_(1)
    folder = tmp_path / "dictionary"
    folder.mkdir()
    (folder / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    for number, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), folder / f"layer_{number:02d}.safetensors")
    return SimpleNamespace(model=model, dictionary=dictionary, tokenizer=Tokenizer(),
                           folder=folder, readings=readings, guards=guards)


def test_standalone_gemma_index_load_build_hash_and_examples(gemma_resources):
    case = gemma_resources
    options = {"model": case.model, "tokenizer": case.tokenizer, "memory_limit_gb": 28}
    _index.build_index(case.folder, [("first", [1, 3, 5, 7])], provenance={"partition": "fixture"}, **options)
    with _index.FeatureIndex(case.folder, **options) as index:
        assert index.memory.limit == 28_000_000_000
        assert index.dictionary is not case.dictionary
        assert index.dictionary.metadata == case.dictionary.metadata
        assert _index._dictionary_hash(index.dictionary, memory_limit_gb=28) == index.metadata["dictionary_hash"]
        assert index.examples(0, int(index.counts[0].argmax()))
    assert case.guards and all(guard.limit == 28_000_000_000 for guard in case.guards)


def test_standalone_gemma_service_and_validation_budget(gemma_resources):
    case = gemma_resources
    controls = Controls()
    service = register_service(controls=controls, model=case.model, tokenizer=case.tokenizer,
                               dictionary=case.dictionary, memory_limit_gb=28)
    try:
        tracer = service._resource("tracer")
        assert tracer.guard.limit == 28_000_000_000 and tracer.controls is controls
        graph = tracer.trace("fixture", max_targets=0, max_nodes=32)
        # Use actual last-layer activations for this resource-route check.
        # This synthetic graph supplies no evidence about attribution quality.
        point = case.dictionary.metadata["read_points"][-1]
        with torch.inference_mode(), capture(case.model, [point]) as saved:
            forward_logits(case.model, case.tokenizer("fixture"))
            activations = case.dictionary.encode(1, saved[point])[0, -1]
        nodes = [node for node in graph.nodes if node["kind"] == "logit"]
        for feature in activations.nonzero().flatten().tolist():
            nodes.append({"kind": "feature", "layer": 1, "position": 2, "feature": feature,
                          "activation": float(activations[feature]), "influence": 1., "effect": 0.})
        validation_graph = SimpleNamespace(nodes=nodes, tokens=graph.tokens, trace_id=graph.trace_id)
        interventions = controls.interventions_on(case.model)
        try:
            interventions.configure(dictionary=case.dictionary, tokenizer=case.tokenizer)
            result = _validate.validate_trace(case.model, case.tokenizer, case.dictionary,
                                              interventions, validation_graph, prompt="fixture", memory_limit_gb=28)
            assert result["count"] == 2 and not result["passed"]
        finally:
            interventions.remove()
    finally:
        unregister_service(service, controls=controls)


@pytest.mark.parametrize("route", ["index", "build", "service", "validation"])
def test_standalone_routes_preserve_supplied_lower_guard_before_work(gemma_resources, monkeypatch, route):
    case = gemma_resources
    lower = process_guard(25_000_000_000, case.readings)
    # The limit shrinks after creation. The supplied stop and actual history remain active.
    lower.limit = 23_000_000_000
    def forbidden(*args, **kwargs):
        pytest.fail("a lower guard must refuse before dictionary or prompt work")
    monkeypatch.setattr(_index, "load_dictionary", forbidden)
    monkeypatch.setattr(_trace, "_validate_dictionary", forbidden)
    monkeypatch.setattr(_validate, "prompt_inputs", forbidden)
    with pytest.raises(CleanStop, match="memory limit"):
        if route == "index":
            _index.FeatureIndex(case.folder, model=case.model, tokenizer=case.tokenizer,
                                memory=lower, memory_limit_gb=28)
        elif route == "build":
            _index.build_index(case.folder, [], model=case.model, tokenizer=case.tokenizer,
                               provenance={"partition": "fixture"}, memory=lower, memory_limit_gb=28)
        elif route == "service":
            controls = Controls()
            service = register_service(controls=controls, model=case.model, tokenizer=case.tokenizer,
                                       dictionary=case.dictionary, memory=lower, memory_limit_gb=28)
            try:
                service._resource("tracer")
            finally:
                unregister_service(service, controls=controls)
        else:
            _validate.validate_trace(case.model, case.tokenizer, case.dictionary, SimpleNamespace(),
                                      None, prompt="fixture", guard=lower, memory_limit_gb=28)


def test_trace_reserves_and_identity_keep_dynamic_inner_lower_guard(monkeypatch):
    model, tokenizer, dictionary = resources()
    readings = {"current": 1_000, "peak": 5_000}
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", lambda limit, device, **kwargs:
                        process_guard(int(limit * 1e9), readings))
    inner = process_guard(100_000_000, readings)
    supplied = SimpleNamespace(device=torch.device("cpu"), limit=100_000_000, _guard=inner,
                               check=lambda stage: inner.check(stage))
    tracer = _trace.Tracer(model, tokenizer, dictionary, memory=supplied, memory_limit_gb=1)
    assert tracer.guard.limit == 100_000_000
    inner.limit = 10_000_000
    assert tracer.guard.limit == 10_000_000
    with pytest.raises(CleanStop, match="trace memory limit"):
        tracer.guard.reserve("pending trace", 9_999_001)
    inner.limit = 4 * _dictionary._HASH_CHUNK_BYTES
    with pytest.raises(CleanStop, match="model identity allocation"):
        _dictionary._state_hash(dictionary.state_dict(), memory=tracer.guard._guard)


@pytest.mark.parametrize("family", ["qwen3_5_text", "caliper_tiny"])
@pytest.mark.parametrize("route", ["index", "build", "tracer", "validation"])
def test_explicit_gemma_budget_cannot_raise_other_family_limits(tmp_path, family, route):
    model = SimpleNamespace(config=SimpleNamespace(model_type=family))
    with pytest.raises(ValueError, match="at most 12 GB"):
        if route == "index":
            _index.FeatureIndex(tmp_path, model=model, tokenizer=None, memory_limit_gb=28)
        elif route == "build":
            _index.build_index(tmp_path, [], model=model, tokenizer=None, provenance={"partition": "fixture"},
                               memory_limit_gb=28)
        elif route == "tracer":
            _trace.Tracer(model, None, object(), memory_limit_gb=28)
        else:
            _validate._validation_guard(model, None, SimpleNamespace(),
                                         process_guard(28_000_000_000, {"current": 0, "peak": 0}), 28)
    assert family_memory_limit(model) == 12


def test_budget_adapter_preserves_stage_only_latch_and_historical_readings():
    values = iter([900, 100])
    guard = SimpleNamespace(limit=1_000, check=lambda stage: next(values))
    memory = BudgetMemory(guard, .000001)
    assert memory.check("first") == 900
    assert memory.check("released") == 100 and memory.peak == 900
    guard.limit = 899
    with pytest.raises(CleanStop, match="historical allocation"):
        from caliper.interp._memory import allocation_bytes
        allocation_bytes(memory)
