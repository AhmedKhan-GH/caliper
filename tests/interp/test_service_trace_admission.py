"""CPU checks for deferred service admission and exact request validation."""
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import _frozen, _memory, _publish, _service, _trace
from caliper.interp._memory import CleanStop
from .fixture import TinyTokenizer
from .test_memory_phases import process_guard
from .test_service_memory import RequestGuard, resources as indexed_resources
from .test_trace_safety import resources


@pytest.fixture
def case(monkeypatch):
    readings = {"current": 1_000, "peak": 1_000}

    def guard(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        return process_guard(int(limit * 1e9), readings)

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", guard)
    monkeypatch.setattr(_memory, "MemoryGuard", guard)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    model, tokenizer, dictionary = resources()
    service = _service.InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary)
    service._bind_controls(Controls())
    yield SimpleNamespace(model=model, tokenizer=tokenizer, dictionary=dictionary, service=service)
    service.close()


def enqueue(service, **params):
    replies = []
    service.submit({"version": 1, "method": "trace.run",
                    "params": {"prompt": "AB", "max_targets": 0, **params}},
                   lambda *args: replies.append(args))
    return replies


def run(service, **params):
    replies = enqueue(service, **params)
    assert service.step() == 1
    assert replies[0][0] == "ok", replies
    return service.graph


def unchanged_publication(published):
    assert _publish._TENSORS.keys() == published.keys()
    assert all(_publish._TENSORS[name] is value for name, value in published.items())


@pytest.mark.parametrize("source", ["default", "dictionary_factory", "tracer_factory"])
def test_lazy_service_checks_once_per_request_before_both_caches(case, monkeypatch, source):
    service = case.service
    counts = dict(identity=0, finite=0, content=0, forwards=0)
    identity, finite, content = _frozen.model_identity, _frozen._require_finite, _trace.weights_hash
    forward, factories = case.model.forward, []

    def checked_identity(value, **kwargs):
        counts["identity"] += 1
        return identity(value, **kwargs)

    def checked_finite(value, name, *args):
        if name.startswith("dictionary layer"):
            counts["finite"] += 1
        return finite(value, name, *args)

    def checked_content(value, **kwargs):
        counts["content"] += 1
        return content(value, **kwargs)

    def checked_forward(*args, **kwargs):
        counts["forwards"] += 1
        return forward(*args, **kwargs)

    def factory():
        factories.append(source)
        assert service.status["state"] == "running"
        if source == "dictionary_factory":
            return case.dictionary
        return _trace.Tracer(case.model, case.tokenizer, case.dictionary, controls=service.controls)

    monkeypatch.setattr(_frozen, "model_identity", checked_identity)
    monkeypatch.setattr(_frozen, "_require_finite", checked_finite)
    monkeypatch.setattr(_trace, "weights_hash", checked_content)
    monkeypatch.setattr(case.model, "forward", checked_forward)
    if source != "default":
        service.factories[source.removesuffix("_factory")] = factory
        if source == "dictionary_factory":
            service.dictionary = None
    replies = enqueue(service)
    assert counts == dict(identity=0, finite=0, content=0, forwards=0) and not factories
    service.step()
    assert replies[0][0] == "ok", replies
    scans = 2 if source == "tracer_factory" else 1
    assert counts == dict(identity=scans, finite=8 * scans, content=1, forwards=1)
    first, path = service.graph, service.tracer._forward_cache[1]
    assert run(service) is first
    assert counts == dict(identity=scans + 1, finite=8 * (scans + 1), content=2, forwards=1)
    run(service, max_nodes=511)
    assert service.tracer._forward_cache[1] is path
    assert counts == dict(identity=scans + 2, finite=8 * (scans + 2), content=3, forwards=1)
    run(service, prompt="CD")
    assert counts == dict(identity=scans + 3, finite=8 * (scans + 3), content=4, forwards=2)
    assert factories == ([] if source == "default" else [source])


@pytest.mark.parametrize("cache", ["cold", "graph", "forward", "new_prompt"])
@pytest.mark.parametrize("resource", ["model", "dictionary"])
@pytest.mark.parametrize("callback", ["prompt", "patch"])
def test_tokenizer_alias_mutation_refuses_before_service_publication(case, monkeypatch, cache, resource, callback):
    tensor = case.model.embed_tokens.weight if resource == "model" else case.dictionary.layers[1].W_dec

    class MutatingTokenizer(TinyTokenizer):
        armed = False

        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if self.armed and (callback == "prompt" or text == "EF"):
                version = tensor._version
                tensor.data[-1, -1] = 7. if resource == "model" else float("nan")
                assert tensor._version == version
            return result

    tokenizer = case.service.tokenizer = MutatingTokenizer()
    intervention = None
    if callback == "patch":
        intervention = case.service.controls.interventions_on(case.model)
        intervention.configure(dictionary=case.dictionary, tokenizer=tokenizer)
        intervention.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "EF"})
        case.service.controls.step()
    try:
        if cache != "cold":
            run(case.service)
        prior, published = case.service.graph, dict(_publish._TENSORS)
        tokenizer.armed = True
        hooks = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
        monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("alias reached a forward"))
        replies = enqueue(case.service, prompt="CD" if cache == "new_prompt" else "AB",
                          max_nodes=511 if cache == "forward" else 512)
        case.service.step()
        assert replies[0][0] == "error"
        message = "model identity mismatch" if resource == "model" else "dictionary layer 1 W_dec must be finite"
        assert message in replies[0][2]["message"]
        assert case.service.graph is prior
        assert not case.service.tracer._cache and case.service.tracer._forward_cache is None
        unchanged_publication(published)
        assert hooks == [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
    finally:
        if intervention is not None:
            intervention.remove()


@pytest.mark.parametrize("cache", ["graph", "forward"])
def test_finite_tokenizer_alias_change_refreshes_service_cache(case, cache):
    class MutatingTokenizer(TinyTokenizer):
        armed = False

        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            if self.armed:
                tensor = case.dictionary.layers[1].b_dec
                version = tensor._version
                tensor.data[-1] = -0.
                assert tensor._version == version
                self.armed = False
            return result

    tokenizer = case.service.tokenizer = MutatingTokenizer()
    first = run(case.service)
    path = case.service.tracer._forward_cache[1]
    tokenizer.armed = True
    second = run(case.service, max_nodes=511 if cache == "forward" else 512)
    assert second.trace_id != first.trace_id
    assert case.service.tracer._forward_cache[1] is not path


@pytest.mark.parametrize("fault,message", [
    ("model", "model identity mismatch"),
    ("metadata", "dimensions differ"),
    ("probes", "read_points differ"),
    ("shape", "shape differs"),
    ("nonfinite", "must be finite"),
])
def test_invalid_resources_refuse_in_service_and_standalone_constructor(case, monkeypatch, fault, message):
    if fault == "model":
        case.model.embed_tokens.weight.data[-1, -1] = 7.
    elif fault == "metadata":
        case.dictionary.metadata["width"] += 1
    elif fault == "probes":
        case.dictionary.metadata["read_points"] = []
    elif fault == "shape":
        case.dictionary.layers[1].b_dec.data = torch.zeros(1)
    else:
        case.dictionary.layers[1].W_dec.data[-1, -1] = float("inf")
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("invalid resources reached a forward"))
    with pytest.raises(ValueError, match=message):
        _trace.Tracer(case.model, case.tokenizer, case.dictionary)
    with pytest.raises(ValueError, match=message):
        _frozen.FrozenPath(case.model, torch.tensor([[65, 66]]), case.dictionary)
    replies = enqueue(case.service)
    case.service.step()
    assert replies[0][0] == "error" and message in replies[0][2]["message"]
    assert case.service.graph is None and not case.service.tracer._cache
    assert case.service.tracer._forward_cache is None and not _publish._TENSORS


@pytest.mark.parametrize("fault", ["dictionary_factory", "tracer_factory", "settings"])
def test_failed_lazy_construction_retries_without_partial_tracer(case, monkeypatch, fault):
    attempts = []
    settings = _trace._forward_settings

    def fail_once(*args, **kwargs):
        attempts.append(fault)
        if len(attempts) == 1:
            raise ValueError("fixture construction refusal")
        if fault == "dictionary_factory":
            return case.dictionary
        if fault == "tracer_factory":
            return _trace.Tracer(case.model, case.tokenizer, case.dictionary, controls=case.service.controls)
        return settings(*args, **kwargs)

    if fault == "settings":
        monkeypatch.setattr(_trace, "_forward_settings", fail_once)
    else:
        case.service.factories[fault.removesuffix("_factory")] = fail_once
        if fault == "dictionary_factory":
            case.service.dictionary = None
    replies = enqueue(case.service)
    case.service.step()
    assert replies[0][0] == "error" and "construction refusal" in replies[0][2]["message"]
    assert case.service.tracer is case.service.graph is None
    assert case.service.failure is None and not _publish._TENSORS
    assert not case.service._borrowed and case.service._request_stack is None
    assert run(case.service) is not None and len(attempts) >= 2


@pytest.mark.parametrize("stage", ["dictionary admission", "model_identity", "finite block"])
@pytest.mark.parametrize("fault", ["memory", "deadline"])
def test_first_request_admission_refusals_keep_publication_atomic(case, monkeypatch, stage, fault):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    deadline = _trace._Deadline.__call__
    seen = []

    def expire(owner, current):
        if current == stage:
            seen.append(current)
            clock.now += _trace.TRACE_TIMEOUT_SECONDS
        deadline(owner, current)

    def stop(current):
        if current == stage:
            seen.append(current)
            raise CleanStop("fixture supplied guard stop")

    if fault == "deadline":
        monkeypatch.setattr(_trace._Deadline, "__call__", expire)
    else:
        memory = RequestGuard()
        memory.at_stage = stop
        case.service.memory = memory
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("refusal reached a forward"))
    replies = enqueue(case.service)
    if fault == "memory":
        with pytest.raises(CleanStop, match="supplied guard stop"):
            case.service.step()
    else:
        case.service.step()
    assert seen == [stage] and replies[0][0] == "error"
    assert replies[0][2]["code"] == ("interpret.stopped" if fault == "memory" else "interpret.trace_timeout")
    assert case.service.graph is None and not case.service.tracer._cache
    assert case.service.tracer._forward_cache is None and not _publish._TENSORS
    assert not case.service._borrowed and case.service._request_stack is None


def test_service_timer_includes_lazy_setup_and_index_binding(indexed_resources, monkeypatch):
    case = indexed_resources
    clock, stages = SimpleNamespace(now=100.), []
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(_trace, "time", fake_time)
    monkeypatch.setattr(_service, "time", fake_time)
    initialize, bind, forward = _trace.Tracer._initialize, _trace.Tracer.bind_index, case.model.forward

    def dictionary():
        stages.append("dictionary")
        clock.now += 5
        return case.dictionary

    def index():
        stages.append("index")
        clock.now += 7
        return case.index

    def construct(owner, *args, **kwargs):
        stages.append("constructor")
        clock.now += 3
        initialize(owner, *args, **kwargs)

    def bind_index(owner, value):
        if value is not None:
            stages.append("bind")
            clock.now += 11
        bind(owner, value)

    def model_forward(*args, **kwargs):
        stages.append("forward")
        clock.now += 2
        return forward(*args, **kwargs)

    monkeypatch.setattr(_trace.Tracer, "_initialize", construct)
    monkeypatch.setattr(_trace.Tracer, "bind_index", bind_index)
    monkeypatch.setattr(case.model, "forward", model_forward)
    service = _service.InterpretationService(model=case.model, tokenizer=case.tokenizer,
                                             dictionary_factory=dictionary, index_factory=index)
    service._bind_controls(case.controls)
    try:
        replies = enqueue(service)
        assert clock.now == 100 and not stages
        service.step()
        assert replies[0][0] == "ok", replies
        assert stages == ["dictionary", "constructor", "index", "bind", "forward"]
        assert replies[0][1]["seconds"] == 28
        assert service.tracer.index is case.index
        assert service.index.memory is case.index_memory
    finally:
        service.close()


def test_deferred_service_admission_preserves_standalone_graph(case):
    deferred = run(case.service, max_targets=4)
    standalone = _trace.Tracer(case.model, case.tokenizer, case.dictionary,
                               controls=case.service.controls).trace("AB", max_targets=4)
    assert deferred.trace_id == standalone.trace_id
    assert deferred.nodes == standalone.nodes and deferred.edges == standalone.edges
    assert deferred.tokens == standalone.tokens and deferred.completeness == standalone.completeness
    assert deferred.tensors.keys() == standalone.tensors.keys()
    assert all(torch.equal(value, standalone.tensors[name]) for name, value in deferred.tensors.items())


@pytest.mark.parametrize("fault", ["closed", "model", "tokenizer", "dictionary"])
def test_lazy_index_refusal_precedes_forward_and_publication(indexed_resources, monkeypatch, fault):
    case = indexed_resources
    if fault == "closed":
        case.index.close()
    else:
        monkeypatch.setattr(case.index, fault, object())
    message = "closed" if fault == "closed" else "share"
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("invalid index reached a forward"))
    with pytest.raises(ValueError, match=message):
        _trace.Tracer(case.model, case.tokenizer, case.dictionary, index=case.index)
    service = _service.InterpretationService(model=case.model, tokenizer=case.tokenizer,
                                             dictionary=case.dictionary, index_factory=lambda: case.index)
    service._bind_controls(case.controls)
    try:
        replies = enqueue(service)
        service.step()
        assert replies[0][0] == "error" and message in replies[0][2]["message"]
        assert service.graph is None and not service.tracer._cache
        assert service.tracer._forward_cache is None and not _publish._TENSORS
        assert case.index.memory is case.index_memory
        assert not service._borrowed and service._request_stack is None
    finally:
        service.close()


def test_first_request_dictionary_refusal_requires_fresh_admission_after_repair(case, monkeypatch):
    tensor = case.dictionary.layers[1].b_dec
    original = tensor[-1].item()
    tensor.data[-1] = float("nan")
    replies = enqueue(case.service)
    case.service.step()
    assert replies[0][0] == "error" and "must be finite" in replies[0][2]["message"]
    tracer = case.service.tracer
    assert tracer is not None and not tracer._cache and tracer._forward_cache is None
    tensor.data[-1] = original
    run(case.service)
    assert case.service.tracer is tracer
    published = dict(_publish._TENSORS)
    tensor.data[-1] = float("nan")
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("invalid repair reached a forward"))
    replies = enqueue(case.service)
    case.service.step()
    assert replies[0][0] == "error" and "must be finite" in replies[0][2]["message"]
    assert not tracer._cache and tracer._forward_cache is None
    unchanged_publication(published)
