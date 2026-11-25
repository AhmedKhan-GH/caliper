"""Hosted Gemma admission through queued requests with random CPU resources."""
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import _frozen, _hosted, _publish, _trace
from caliper.interp._memory import CleanStop
from .test_gemma_frozen import Tokenizer, setup
from .test_memory_phases import process_guard
from .test_service_memory import RequestGuard
from .test_service_trace_admission import enqueue, run, unchanged_publication


@pytest.fixture
def case(monkeypatch):
    readings = {"current": 1_000, "peak": 1_000}
    process_guards = []

    def guard(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        memory = process_guard(int(limit * 1e9), readings)
        process_guards.append(memory)
        return memory

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", guard)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    model, dictionary, _ = setup(jump=True)

    class CallbackTokenizer(Tokenizer):
        callback = None

        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if self.callback is not None:
                self.callback(text)
            return result

    tokenizer, controls = CallbackTokenizer(), Controls()
    interventions = controls.interventions_on(model)
    interventions.configure(dictionary=dictionary, tokenizer=tokenizer)
    index = SimpleNamespace(
        model=model, tokenizer=tokenizer, dictionary=dictionary,
        memory=RequestGuard(40_000_000_000), label_identity=("hosted-fixture", "labels"),
        labels=lambda features, **kw: {key: f"Feature {key[0]}:{key[1]}" for key in features})
    failures = []
    owner = SimpleNamespace(
        model=model, tokenizer=tokenizer, dictionary=dictionary, index=index,
        guard=RequestGuard(40_000_000_000), controls=controls, interventions=interventions,
        sweep=lambda **kw: None, validate=lambda **kw: None,
        prompt=SimpleNamespace(set=lambda value: None), status=SimpleNamespace(set=lambda value: None),
        _check_stop=lambda: None, _latch_failure=failures.append,
        update_controls=controls.step, update_output=lambda: None, _reconcile_selection=lambda graph: None)
    service = _hosted.HostedService(owner)
    service._bind_controls(controls)
    yield SimpleNamespace(model=model, dictionary=dictionary, tokenizer=tokenizer, owner=owner,
                          service=service, index=index, readings=readings,
                          process_guards=process_guards, failures=failures)
    service.close()
    interventions.remove()


def test_hosted_factory_admits_once_per_request_before_both_caches(case, monkeypatch):
    counts = dict(identity=0, content=0, forwards=0)
    tensors, tokens = [], []
    identity, finite, content = _frozen.model_identity, _frozen._require_finite, _trace.weights_hash
    forward = case.model.forward
    expected = {f"dictionary layer {i} {name}": id(getattr(layer, name))
                for i, layer in enumerate(case.dictionary.layers)
                for name in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}
    case.tokenizer.callback = tokens.append

    def checked_identity(value, **kwargs):
        counts["identity"] += 1
        assert len(tokens) == 2 * counts["identity"]
        return identity(value, **kwargs)

    def checked_finite(value, name, *args):
        if name.startswith("dictionary layer"):
            tensors.append((name, id(value)))
        return finite(value, name, *args)

    def checked_content(value, **kwargs):
        counts["content"] += 1
        assert len(tensors) == len(expected) * counts["content"]
        assert dict(tensors[-len(expected):]) == expected
        return content(value, **kwargs)

    def checked_forward(*args, **kwargs):
        counts["forwards"] += 1
        assert counts["identity"] == counts["content"]
        return forward(*args, **kwargs)

    monkeypatch.setattr(_frozen, "model_identity", checked_identity)
    monkeypatch.setattr(_frozen, "_require_finite", checked_finite)
    monkeypatch.setattr(_trace, "weights_hash", checked_content)
    monkeypatch.setattr(case.model, "forward", checked_forward)
    replies = enqueue(case.service)
    assert case.service.tracer is None and not case.process_guards
    assert counts == dict(identity=0, content=0, forwards=0) and not tensors
    assert case.service.step() == 1 and replies[0][0] == "ok", replies
    assert counts == dict(identity=1, content=1, forwards=1)
    tracer = case.service.tracer
    assert tracer._memory is case.owner.guard and tracer.index is case.index
    assert tracer.controls is case.owner.controls
    first, path = case.service.graph, tracer._forward_cache[1]
    assert run(case.service) is first
    assert counts == dict(identity=2, content=2, forwards=1)
    assert run(case.service, max_nodes=511) is not first
    assert tracer._forward_cache[1] is path
    assert counts == dict(identity=3, content=3, forwards=1)
    run(case.service, prompt="CD")
    assert tracer._forward_cache[1] is not path
    assert counts == dict(identity=4, content=4, forwards=2)
    assert tokens == ["AB"] * 6 + ["CD"] * 2
    assert len(case.process_guards) == 1


@pytest.mark.parametrize("limit", [40, 28])
def test_hosted_factory_preserves_cap_and_dynamic_owner_limit(case, monkeypatch, limit):
    case.owner.guard._guard.limit = int(limit * 1e9)
    run(case.service)
    tracer = case.service.tracer
    assert tracer.memory_limit_gb == limit
    assert tracer.guard.limit == int(limit * 1e9)
    assert tracer.guard._guard.guards == (case.process_guards[0], case.owner.guard)
    assert case.process_guards[0].limit == int(limit * 1e9)
    assert "dictionary admission" in case.owner.guard.stages
    published = dict(_publish._TENSORS)
    case.owner.guard._guard.limit = 999
    assert tracer.guard.limit == 999
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("lower cap reached a forward"))
    replies = enqueue(case.service)
    with pytest.raises(CleanStop, match="memory limit"):
        case.service.step()
    assert replies[0][0] == "error" and replies[0][2]["code"] == "interpret.stopped"
    assert case.failures and case.service.failure is not None
    unchanged_publication(published)


@pytest.mark.parametrize("source", ["owner", "process"])
def test_hosted_admission_checks_both_guards_after_tokenizer(case, monkeypatch, source):
    def breach(text):
        if case.service._request_stack is not None:
            readings = case.owner.guard.readings if source == "owner" else case.readings
            readings.update(current=40_000_000_001, peak=40_000_000_001)

    case.tokenizer.callback = breach
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("memory breach reached a forward"))
    replies = enqueue(case.service)
    with pytest.raises(CleanStop, match="memory limit"):
        case.service.step()
    assert replies[0][0] == "error" and replies[0][2]["code"] == "interpret.stopped"
    assert case.service.tracer is not None and case.service.graph is None
    assert not _publish._TENSORS
    unaffected = case.readings if source == "owner" else case.owner.guard.readings
    assert unaffected == {"current": 1_000, "peak": 1_000}


@pytest.mark.parametrize("fault,message", [
    ("metadata", "dimensions differ"),
    ("shape", "threshold shape differs"),
    ("nonfinite", "threshold must be finite"),
])
def test_hosted_malformed_dictionary_refuses_and_public_constructor_stays_eager(case, monkeypatch, fault, message):
    if fault == "metadata":
        case.dictionary.metadata["width"] += 1
    elif fault == "shape":
        case.dictionary.layers[-1].threshold.data = torch.ones(1)
    else:
        case.dictionary.layers[-1].threshold.data[-1] = float("nan")
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("malformed dictionary reached a forward"))
    with pytest.raises(ValueError, match=message):
        _trace.Tracer(case.model, case.tokenizer, case.dictionary, memory=case.owner.guard, memory_limit_gb=40)
    replies = enqueue(case.service)
    assert case.service.step() == 1
    assert replies[0][0] == "error" and message in replies[0][2]["message"]
    assert case.service.tracer is not None and case.service.graph is None
    assert not case.service.tracer._cache and case.service.tracer._forward_cache is None
    assert not _publish._TENSORS


@pytest.mark.parametrize("cache", ["cold", "graph", "forward"])
@pytest.mark.parametrize("resource", ["model", "dictionary"])
@pytest.mark.parametrize("callback", ["prompt", "patch"])
def test_hosted_tokenizer_alias_mutation_refuses_before_forward(case, monkeypatch, cache, resource, callback):
    tensor = (case.model.model.embed_tokens.weight if resource == "model"
              else case.dictionary.layers[-1].W_dec)
    if callback == "patch":
        case.owner.interventions.set({"kind": "patch", "point": "model.layers.0.resid_pre",
                                      "source_prompt": "EF"})
    case.owner.controls.step()
    if cache != "cold":
        run(case.service)
    prior, published = case.service.graph, dict(_publish._TENSORS)
    mutations = []

    def mutate(text):
        if (case.service._request_stack is not None and not mutations
                and text == ("AB" if callback == "prompt" else "EF")):
            version = tensor._version
            tensor.data[-1, -1] = 7. if resource == "model" else float("nan")
            assert tensor._version == version
            mutations.append(text)

    case.tokenizer.callback = mutate
    hooks = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
    monkeypatch.setattr(case.model, "forward", lambda *a, **kw: pytest.fail("alias mutation reached a forward"))
    replies = enqueue(case.service, max_nodes=511 if cache == "forward" else 512)
    assert case.service.step() == 1
    assert mutations == ["AB" if callback == "prompt" else "EF"]
    message = "model identity mismatch" if resource == "model" else "W_dec must be finite"
    assert replies[0][0] == "error" and message in replies[0][2]["message"]
    assert case.service.graph is prior
    assert not case.service.tracer._cache and case.service.tracer._forward_cache is None
    unchanged_publication(published)
    assert hooks == [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in case.model.modules()]
    assert not case.service._borrowed and case.service._request_stack is None


def test_hosted_deferred_graph_equals_eager_standalone_graph(case):
    deferred = run(case.service, max_targets=4)
    eager = _trace.Tracer(case.model, case.tokenizer, case.dictionary, controls=case.owner.controls,
                          index=case.index, memory=case.owner.guard, memory_limit_gb=40).trace("AB", max_targets=4)
    assert deferred.trace_id == eager.trace_id
    assert deferred.nodes == eager.nodes and deferred.edges == eager.edges
    assert deferred.tokens == eager.tokens and deferred.completeness == eager.completeness
    assert deferred.tensors.keys() == eager.tensors.keys()
    assert all(torch.equal(value, eager.tensors[name]) for name, value in deferred.tensors.items())
