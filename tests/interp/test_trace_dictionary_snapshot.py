"""Trace identity reuses private content only after a current exact comparison."""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import _dictionary, _frozen, _intervene, _publish, _trace
from caliper.interp._memory import CleanStop
from .test_memory_phases import process_guard
from .test_trace_safety import resources


@pytest.fixture
def case(monkeypatch):
    readings = {"current": 1_000, "peak": 1_000}
    process = process_guard(12_000_000_000, readings)
    supplied = process_guard(12_000_000_000, readings)
    owner_guard = process_guard(12_000_000_000, readings)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", lambda *a, **kw: process)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    model, tokenizer, dictionary = resources()
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=owner_guard)
    executor = service.executor
    executor._prepare_dictionary_identity()
    assert type(executor._dictionary_snapshot) is _intervene._DictionarySnapshot
    tracer = _trace.Tracer(model, tokenizer, dictionary, controls=controls, memory=supplied)
    yield SimpleNamespace(model=model, tokenizer=tokenizer, dictionary=dictionary, controls=controls,
                          service=service, executor=executor, tracer=tracer, supplied=supplied,
                          process=process, owner_guard=owner_guard)
    service.remove()


def forbid(*args, **kwargs):
    pytest.fail("unexpected full hash, copy, or forward")


def record_hash(monkeypatch):
    calls = []
    original = _trace.weights_hash

    def hashed(dictionary, **kwargs):
        result = original(dictionary, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(_trace, "weights_hash", hashed)
    return calls


def test_exact_digest_without_hash_copy_or_new_snapshot(case, monkeypatch):
    expected = _dictionary.weights_hash(case.dictionary)
    snapshot = case.executor._dictionary_snapshot
    assert snapshot.content == expected
    sizes, stages = [], []
    equal = torch.equal

    def compare(left, right):
        assert left.dtype == right.dtype == torch.uint8
        assert left.data_ptr() != right.data_ptr()
        sizes.append(left.numel())
        return equal(left, right)

    monkeypatch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", 32)
    monkeypatch.setattr(torch, "equal", compare)
    monkeypatch.setattr(torch.Tensor, "to", forbid)
    monkeypatch.setattr(torch.Tensor, "clone", forbid)
    monkeypatch.setattr(_trace, "weights_hash", forbid)
    monkeypatch.setattr(_intervene._DictionarySnapshot, "create", forbid)
    assert case.tracer._dictionary_content(stages.append) == expected
    assert max(sizes) <= 32
    assert sum(sizes) == sum(t.numel() * t.element_size() for t in case.dictionary.state_dict().values())
    assert stages.count("dictionary content dictionary identity") >= 2 * len(sizes)
    assert stages[-1] == "dictionary content dictionary snapshot complete"


def test_each_request_compares_after_admission_before_graph_or_forward_cache(case, monkeypatch):
    snapshot = case.executor._dictionary_snapshot
    case.executor._dictionary_snapshot = None
    baseline = case.tracer.trace("AB", max_targets=0)
    case.executor._dictionary_snapshot = snapshot
    order = []
    validate, matches = _frozen._validate_dictionary, snapshot.matches

    def admitted(*args, **kwargs):
        result = validate(*args, **kwargs)
        order.append("admission")
        return result

    def compared(*args, **kwargs):
        assert order[-1] == "admission"
        order.append("comparison")
        return matches(*args, **kwargs)

    monkeypatch.setattr(_frozen, "_validate_dictionary", admitted)
    monkeypatch.setattr(snapshot, "matches", compared)
    monkeypatch.setattr(_trace, "weights_hash", forbid)
    monkeypatch.setattr(case.model, "forward", forbid)
    assert case.tracer.trace("AB", max_targets=0) is baseline
    path = case.tracer._forward_cache[1]
    changed_caps = case.tracer.trace("AB", max_targets=0, max_nodes=511)
    assert changed_caps is not baseline and case.tracer._forward_cache[1] is path
    assert order == ["admission", "comparison"] * 2
    case.tracer._cache.clear()
    fresh = case.tracer.trace("AB", max_targets=0)
    assert fresh.trace_id == baseline.trace_id
    assert fresh.nodes == baseline.nodes and fresh.edges == baseline.edges
    assert fresh.completeness == baseline.completeness


@pytest.mark.parametrize("write", ["data", "numpy", "alias", "signed_zero"])
def test_unversioned_mutation_falls_back_and_invalidates_both_caches(case, monkeypatch, write):
    prior = case.tracer.trace("AB", max_targets=0)
    path = case.tracer._forward_cache[1]
    parameter = case.dictionary.layers[-1].b_dec
    version = parameter._version
    # An earlier admission must not authorize reuse after a callback changes bytes.
    with case.executor.dictionary_admission():
        if write == "data":
            parameter.data[-1] += .125
        elif write == "numpy":
            parameter.detach().numpy()[-1] += .125
        elif write == "alias":
            alias = torch.empty(0, dtype=parameter.dtype).set_(parameter.untyped_storage())
            alias[-1] += .125
        else:
            assert parameter[-1] == 0
            parameter.data[-1] = -0.
        assert parameter._version == version
        expected = _dictionary.weights_hash(case.dictionary)
        assert expected != case.executor._dictionary_snapshot.content
        calls = record_hash(monkeypatch)
        graph = case.tracer.trace("AB", max_targets=0)
    assert calls == [expected]
    assert graph is not prior and graph.trace_id != prior.trace_id
    assert case.tracer._forward_cache[1] is not path
    assert case.tracer.trace("AB", max_targets=0) is graph
    assert calls == [expected, expected]


@pytest.mark.parametrize("fault", ["missing", "closed", "owner_model", "executor_model", "dictionary",
                                    "snapshot", "snapshot_type", "caller_id"])
def test_unrelated_or_expired_snapshot_uses_full_hash(case, monkeypatch, fault):
    executor = case.executor
    if fault == "missing":
        executor._dictionary_snapshot = None
    elif fault == "closed":
        executor.close()
    elif fault == "owner_model":
        monkeypatch.setattr(case.service, "model", object())
    elif fault == "executor_model":
        monkeypatch.setattr(executor, "model", object())
    elif fault == "dictionary":
        executor.dictionary = deepcopy(case.dictionary)
    elif fault == "snapshot":
        other = deepcopy(case.dictionary)
        other.layers[-1].b_dec.data[-1] += .125
        identifier = _intervene._dictionary_identifier(other, case.owner_guard)
        executor._dictionary_snapshot = _intervene._DictionarySnapshot.create(other, identifier, case.owner_guard)
    elif fault == "snapshot_type":
        executor._dictionary_snapshot = SimpleNamespace(content="sha256:untrusted", matches=lambda *a: True)
    else:
        # Even an explicit ID equal to the default ID cannot authorize reuse.
        executor._default_dictionary_id = False
    expected = _dictionary.weights_hash(case.dictionary)
    calls = record_hash(monkeypatch)
    assert case.tracer._dictionary_content(lambda stage: None) == expected
    assert calls == [expected]


@pytest.mark.parametrize("fault", ["metadata", "topk", "names", "dtype", "shape", "source_device"])
def test_snapshot_compatibility_mismatch_retains_full_hash(case, monkeypatch, fault):
    if fault == "metadata":
        case.dictionary.metadata["revision"] = "changed"
    elif fault == "topk":
        case.dictionary.layers[-1].k -= 1
    elif fault == "names":
        case.dictionary.register_buffer("extra", torch.zeros(1))
    elif fault == "dtype":
        tensor = case.dictionary.layers[-1].b_dec
        tensor.data = tensor.data.float()
    elif fault == "shape":
        tensor = case.dictionary.layers[-1].b_dec
        tensor.data = tensor.data.reshape(2, -1)
    else:
        snapshot = case.executor._dictionary_snapshot
        name = next(iter(snapshot.source_devices))
        snapshot.source_devices[name] = torch.device("meta")
    expected = _dictionary.weights_hash(case.dictionary)
    calls = record_hash(monkeypatch)
    assert case.tracer._dictionary_content(lambda stage: None) == expected
    assert calls == [expected]


def test_metadata_change_updates_graph_identity_with_same_content(case, monkeypatch):
    prior = case.tracer.trace("AB", max_targets=0)
    content = case.executor._dictionary_snapshot.content
    case.dictionary.metadata["revision"] = "changed"
    calls = record_hash(monkeypatch)
    graph = case.tracer.trace("AB", max_targets=0)
    assert calls == [content]
    assert graph.trace_id != prior.trace_id


def test_noncontiguous_storage_keeps_existing_refusal(case, monkeypatch):
    tensor = case.dictionary.layers[-1].W_dec
    tensor.data = tensor.data.T
    monkeypatch.setattr(torch.Tensor, "clone", forbid)
    with pytest.raises(ValueError, match="contiguous dense"):
        _dictionary.weights_hash(case.dictionary)
    with pytest.raises(ValueError, match="contiguous dense"):
        case.tracer._dictionary_content(lambda stage: None)


@pytest.mark.parametrize("fault", ["deadline", "trace_limit", "owner_limit"])
@pytest.mark.parametrize("last", [False, True])
def test_comparison_refusal_keeps_prior_cache_publication_and_hooks(case, monkeypatch, fault, last):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    prior = case.tracer.trace("AB", max_targets=0)
    cache, forward = dict(case.tracer._cache), case.tracer._forward_cache
    published = dict(_publish._TENSORS)
    hooks = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
    size = sum(t.numel() * t.element_size() for t in case.dictionary.state_dict().values())
    compared = []
    equal = torch.equal

    def compare(left, right):
        result = equal(left, right)
        compared.append(left.numel())
        if not last or sum(compared) == size:
            if fault == "deadline":
                clock.now += 20
            else:
                guard = case.supplied if fault == "trace_limit" else case.owner_guard
                guard.limit = 1
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", 32)
        patch.setattr(torch, "equal", compare)
        patch.setattr(_trace, "weights_hash", forbid)
        patch.setattr(case.model, "forward", forbid)
        with pytest.raises(_trace.TraceTimeout if fault == "deadline" else CleanStop):
            case.tracer.trace("AB", max_targets=0)
    assert sum(compared) == (size if last else 32)
    assert case.tracer._cache == cache and case.tracer._forward_cache is forward
    assert _publish._TENSORS.keys() == published.keys()
    assert all(_publish._TENSORS[key] is value for key, value in published.items())
    assert hooks == [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
    if fault == "deadline":
        assert case.tracer.trace("AB", max_targets=0) is prior


def test_snapshot_is_not_retained_by_tracer(case):
    snapshot = case.executor._dictionary_snapshot
    reference = weakref.ref(snapshot)
    tensor = weakref.ref(next(iter(snapshot.state.values())))
    case.tracer.trace("AB", max_targets=0)
    del snapshot
    case.executor.close()
    assert reference() is None and tensor() is None
