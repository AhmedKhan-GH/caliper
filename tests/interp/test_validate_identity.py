"""CPU checks for exact dictionary evidence and bounded validation resources."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _intervene, _validate
from caliper.interp._memory import CleanStop
from .fixture import make_fixture
from .test_intervene import dictionary_for
from .test_validate import frozen_graph, validation_case


def hooks(model):
    return {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
            for name, module in model.named_modules()}


@pytest.fixture
def case():
    model, tokenizer, dictionary, controls, service = validation_case()
    graph = frozen_graph(model, tokenizer, "AB")
    try:
        yield model, tokenizer, dictionary, controls, service, graph
    finally:
        service.remove()


@pytest.mark.parametrize("replay", [False, True])
def test_validation_reuses_exact_snapshot_without_content_readback(case, monkeypatch, replay):
    model, tokenizer, dictionary, controls, service, graph = case
    service.executor._prepare_dictionary_identity()
    snapshot = service.executor._dictionary_snapshot
    before, state = hooks(model), service.state_dict()
    checks = []
    matches = snapshot.matches
    def checked(value, identifier, guard):
        assert value is dictionary and identifier == service.executor.dictionary_id
        checks.append(1)
        return matches(value, identifier, guard)
    monkeypatch.setattr(snapshot, "matches", checked)
    monkeypatch.setattr(_intervene, "_content_identity",
                        lambda *a, **kw: pytest.fail("validation read the dictionary back"))
    with model.frozen_attention(norms=True):
        if replay:
            with service.temporary_state([]) as active:
                result = _validate.validate_trace(model, tokenizer, dictionary, active, graph, prompt="AB")
        else:
            result = _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
    assert result["count"] == 8 and result["correct_signs"] == 8
    assert len(checks) >= result["count"] + 1
    assert service.executor._dictionary_snapshot is snapshot
    assert service.state_dict() == state and hooks(model) == before and controls.n == 0


@pytest.mark.parametrize("source, expected_hashes", [
    ("unprepared", 1), ("opaque", 2), ("other_dictionary", 2), ("other_model", 2)])
def test_private_identity_preparation_occurs_once_with_exact_ownership(case, monkeypatch, source, expected_hashes):
    model, tokenizer, dictionary, controls, service, graph = case
    owner = service.executor
    if source == "opaque":
        service.configure(dictionary=dictionary, tokenizer=tokenizer, dictionary_id="caller-owned")
    elif source == "other_dictionary":
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
        owner._prepare_dictionary_identity()
    elif source == "other_model":
        owner._prepare_dictionary_identity()
        owner.model, _ = make_fixture()
    snapshot = owner._dictionary_snapshot
    hashes = []
    content = _intervene._content_identity
    def hashed(*args, **kwargs):
        hashes.append(1)
        return content(*args, **kwargs)
    monkeypatch.setattr(_intervene, "_content_identity", hashed)
    with model.frozen_attention(norms=True):
        result = _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
    assert result["count"] == 8 and len(hashes) == expected_hashes
    assert owner._dictionary_snapshot is snapshot


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("mutation", ["data", "numpy", "signed_zero", "topk", "metadata"])
def test_validation_refuses_stale_content_before_any_forward(case, monkeypatch, snapshot, mutation):
    model, tokenizer, dictionary, _, service, graph = case
    if snapshot:
        service.executor._prepare_dictionary_identity()
    tensor = dictionary.layers[0].W_dec
    version = tensor._version
    if mutation == "data":
        tensor.data[0, 0].add_(.25)
        assert tensor._version == version
    elif mutation == "numpy":
        tensor.numpy()[0, 0] += .25
        assert tensor._version == version
    elif mutation == "signed_zero":
        dictionary.layers[0].b_dec.numpy()[0] = -0.
    elif mutation == "topk":
        dictionary.layers[0].k += 1
    else:
        dictionary.metadata["training"]["tokens"] += 1
    before = hooks(model)
    monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("stale content reached a forward"))
    with pytest.raises(ValueError, match="dictionary content changed"):
        _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
    assert hooks(model) == before


@pytest.mark.parametrize("snapshot", [False, True])
def test_alias_write_between_measurements_is_refused_without_live_state_changes(case, monkeypatch, snapshot):
    model, tokenizer, dictionary, controls, service, graph = case
    if snapshot:
        service.executor._prepare_dictionary_identity()
    before, state = hooks(model), service.state_dict()
    forward = model.forward
    calls = []
    def mutate(*args, **kwargs):
        result = forward(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            dictionary.layers[0].W_dec.data[0, 0].add_(.25)
        return result
    monkeypatch.setattr(model, "forward", mutate)
    with model.frozen_attention(norms=True), pytest.raises(ValueError, match="dictionary content changed"):
        _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
    assert len(calls) == 2
    assert hooks(model) == before and service.state_dict() == state and controls.n == 0


def test_snapshot_checks_keep_explicit_lower_guard_and_stop(case):
    model, tokenizer, dictionary, _, service, graph = case
    service.executor._prepare_dictionary_identity()
    before, snapshot = hooks(model), service.executor._dictionary_snapshot
    stages = []
    guard = SimpleNamespace(device=torch.device("cpu"), limit=1, peak=0,
                            check=lambda stage, **kwargs: stages.append(stage))
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB", guard=guard)
    assert stages and hooks(model) == before
    assert service.executor._dictionary_snapshot is snapshot


def test_snapshot_admission_falls_back_to_bounded_full_content_checks(case, monkeypatch):
    from caliper.interp._dictionary import _HASH_CHUNK_BYTES
    model, tokenizer, dictionary, _, service, graph = case
    stages, hashes = [], []
    guard = SimpleNamespace(device=torch.device("cpu"), limit=4 * _HASH_CHUNK_BYTES, peak=0,
                            check=lambda stage, **kwargs: stages.append(stage))
    content = _intervene._content_identity
    def hashed(state, memory):
        assert memory is guard
        hashes.append(1)
        return content(state, memory)
    monkeypatch.setattr(_intervene, "_content_identity", hashed)
    with model.frozen_attention(norms=True):
        result = _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB", guard=guard)
    assert result["count"] == 8 and len(hashes) == 9
    assert "dictionary identity" in stages and "model_identity" in stages
    assert service.executor._dictionary_snapshot is None


def test_snapshot_refuses_inference_tensor_alias_mutation(case):
    model, tokenizer, dictionary, _, service, graph = case
    with torch.inference_mode():
        dictionary.layers[0].W_dec = torch.nn.Parameter(dictionary.layers[0].W_dec.clone(), requires_grad=False)
    service.executor._prepare_dictionary_identity()
    with torch.inference_mode():
        dictionary.layers[0].W_dec[0, 0].add_(.25)
    with pytest.raises(ValueError, match="dictionary content changed"):
        _validate.validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("location", ["model", "dictionary"])
def test_default_device_guard_precedes_prompt_setup_without_device_allocation(monkeypatch, device, location):
    # Device labels exercise admission only. No device tensor or API is used.
    cpu, accelerator = (SimpleNamespace(device=torch.device(d)) for d in ("cpu", device))
    model = SimpleNamespace(parameters=lambda: iter([accelerator if location == "model" else cpu]))
    dictionary = SimpleNamespace(state_dict=lambda: {"weight": accelerator if location == "dictionary" else cpu})
    failure = CleanStop("default admission")
    calls = []
    def guard(limit, selected):
        calls.append((limit, selected))
        raise failure
    monkeypatch.setattr(_validate, "MemoryGuard", guard)
    monkeypatch.setattr(_validate, "prompt_inputs", lambda *a: pytest.fail("prompt preceded guard setup"))
    with pytest.raises(CleanStop) as caught:
        _validate.validate_trace(model, None, dictionary, SimpleNamespace(), None, prompt="AB")
    assert caught.value is failure and calls == [(12, torch.device(device))]


def test_guard_selection_preserves_unguarded_cpu_and_matching_owner(case, monkeypatch):
    model, _, dictionary, _, service, _ = case
    monkeypatch.setattr(_validate, "MemoryGuard", lambda *a: pytest.fail("CPU gained an implicit memory guard"))
    assert _validate._validation_guard(model, dictionary, service, None) is None
    owned = SimpleNamespace(check=lambda stage: None)
    service.executor.resource_guard = owned
    assert _validate._validation_guard(model, dictionary, service, None) is owned
    explicit = SimpleNamespace(check=lambda stage: None)
    assert _validate._validation_guard(model, dictionary, service, explicit) is explicit
    assert _validate._validation_guard(model, dictionary_for(model), service, None) is None
