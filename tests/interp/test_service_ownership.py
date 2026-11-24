"""CPU checks for service ownership and intervention-aware trace caches."""
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from caliper import _controls
from caliper.interp import Tracer
from caliper.interp._service import InterpretationService, Refused, register_service, unregister_service
from .fixture import TinyLM, TinyTokenizer
from .test_service import Transport, call
from .test_trace import exact_dictionary


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setattr("caliper.interp._publish._TENSORS", {})
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2).to("cpu")
    return model, TinyTokenizer(), exact_dictionary(model)


@pytest.mark.parametrize("default_owner", [True, False])
def test_supplied_tracer_refuses_different_owner_without_cache_rebinding(resources, default_owner):
    owner = _controls.Controls()
    tracer = Tracer(*resources, **({} if default_owner else {"controls": _controls.Controls()}))
    baseline = tracer.trace("AB", max_targets=0)
    prior_owner, forward = tracer.controls, tracer._forward_cache
    service = InterpretationService(tracer=tracer)
    try:
        with pytest.raises(Refused) as caught:
            register_service(service, controls=owner)
        assert caught.value.code == "interpret.controls_mismatch"
        assert owner.interpretation_service is None
        assert not hasattr(service, "controls") and not service.closed
        assert tracer.controls is prior_owner and tracer._forward_cache is forward
        assert tracer.trace("AB", max_targets=0) is baseline
    finally:
        service.close()


@pytest.mark.parametrize("source", ["automatic", "supplied", "factory"])
def test_registered_tracer_cache_tracks_owner_ablation_and_repeated_prompt(resources, source):
    model, tokenizer, dictionary = resources
    owner, transport = _controls.Controls(), Transport()
    route = owner.attach(transport)
    watched = owner.interventions_on(model)
    if source == "automatic":
        service = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary)
    else:
        tracer = Tracer(*resources, controls=owner)
        service = InterpretationService(**({"tracer": tracer} if source == "supplied" else {
            "tracer_factory": lambda: tracer}))
    register_service(service, controls=owner)

    def trace():
        assert call(route, "trace.run", {"prompt": "AB", "max_targets": 0})["status"] == "queued"
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        return service.graph

    try:
        baseline = trace()
        baseline_logits = service.tracer._forward_cache[1].logits.clone()
        assert service.tracer.controls is owner
        assert service.model is model and service.tokenizer is tokenizer and service.dictionary is dictionary
        assert trace() is baseline
        route.on_control(None, {"request_id": 2, "op": "intervene", "intervene": {
            "set": {"point": "layers.0.mlp", "op": "ablate"}}})
        owner.step()
        ablated = trace()
        assert ablated is not baseline and ablated.trace_id != baseline.trace_id
        assert not torch.equal(service.tracer._forward_cache[1].logits, baseline_logits)
        assert trace() is ablated
        route.on_control(None, {"request_id": 3, "op": "intervene", "intervene": {"clear_all": True}})
        owner.step()
        assert trace() is baseline
    finally:
        unregister_service(service, controls=owner)
        watched.remove()
        route.detach()


@pytest.mark.parametrize("method", ["trace.run", "feature.dictionary"])
def test_tracer_factory_refuses_mismatch_before_resource_binding(resources, monkeypatch, method):
    tracer = Tracer(*resources)
    baseline = tracer.trace("AB", max_targets=0)
    forward = tracer._forward_cache
    owner, transport = _controls.Controls(), Transport()
    route = owner.attach(transport)
    calls = []

    def factory():
        calls.append("factory")
        return tracer

    def unexpected_trace(**kwargs):
        pytest.fail("A mismatched factory tracer must not execute.")

    monkeypatch.setattr(tracer, "trace", unexpected_trace)
    service = register_service(tracer_factory=factory, controls=owner)
    assert not calls
    params = {"prompt": "AB", "max_targets": 0} if method == "trace.run" else {}
    try:
        assert call(route, method, params)["status"] == "queued"
        assert not calls
        owner.step()
        assert calls == ["factory"]
        assert transport.replies[-1].status == "error"
        assert transport.replies[-1].error["code"] == "interpret.controls_mismatch"
        assert service.tracer is None and service.model is None and service.dictionary is None
        assert tracer.controls is _controls.CONTROLS and tracer._forward_cache is forward
        assert list(tracer._cache.values()) == [baseline]
        assert service.failure is None and not service.pending
        service.factories["tracer"] = lambda: Tracer(*resources, controls=owner)
        assert call(route, method, params)["status"] == "queued"
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert service.tracer.controls is owner
    finally:
        unregister_service(service, controls=owner)
        route.detach()


def test_failed_replacement_preserves_live_queue_and_native_capability(resources, monkeypatch):
    owner, transport = _controls.CONTROLS, Transport()
    route = owner.attach(transport)
    refreshed = []
    native = SimpleNamespace(refresh_interpretation=lambda: refreshed.append(owner.interpretation_service))
    monkeypatch.setitem(sys.modules, "caliper._net_native", native)
    service = register_service(tracer=Tracer(*resources))
    candidate = InterpretationService(tracer=Tracer(*resources, controls=_controls.Controls()))
    try:
        assert refreshed == [service]
        assert call(route, "trace.run", {"prompt": "AB", "max_targets": 0})["status"] == "queued"
        with pytest.raises(Refused) as caught:
            register_service(candidate)
        assert caught.value.code == "interpret.controls_mismatch"
        assert owner.interpretation_service is service and not service.closed
        assert service.pending and owner.busy and not transport.replies
        assert not hasattr(candidate, "controls") and not candidate.closed
        assert refreshed == [service]
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
        assert service.graph is not None
        assert register_service(service) is service
        assert refreshed == [service, service]
    finally:
        unregister_service(service)
        candidate.close()
        route.detach()
    assert refreshed == [service, service, None]


@pytest.mark.parametrize("resolved", [False, True])
def test_bound_service_refuses_owner_transfer_and_preserves_both_queues(resources, resolved):
    first, second = _controls.Controls(), _controls.Controls()
    calls, replies = [], []

    def factory():
        calls.append("factory")
        return Tracer(*resources, controls=first)

    service = register_service(controls=first, **({"tracer": factory()} if resolved else {"tracer_factory": factory}))
    previous = register_service(controls=second, dictionary=resources[2])
    request = {"version": 1, "method": "feature.dictionary", "params": {}}
    service.submit(request, lambda *args: replies.append(args))
    previous.submit(request, lambda *args: replies.append(args))
    try:
        assert register_service(service, controls=first) is service
        with pytest.raises(Refused) as caught:
            register_service(service, controls=second)
        assert caught.value.code == "interpret.controls_mismatch"
        assert first.interpretation_service is service and second.interpretation_service is previous
        assert service.controls is first and previous.controls is second
        assert not service.closed and not previous.closed
        assert service.pending and previous.pending and not replies
        assert calls == (["factory"] if resolved else [])
        service.step()
        previous.step()
        assert [reply[0] for reply in replies] == ["ok", "ok"]
        assert calls == ["factory"] and service.tracer.controls is first
    finally:
        unregister_service(service, controls=first)
        unregister_service(previous, controls=second)


def test_owner_claim_precedes_concurrent_replacement(monkeypatch):
    first, second = _controls.Controls(), _controls.Controls()
    previous = register_service(controls=first)
    untouched = register_service(controls=second)
    candidate = InterpretationService()
    entered, release = threading.Event(), threading.Event()
    failures = []
    close = previous.close

    def delayed_close():
        entered.set()
        assert release.wait(5)
        close()

    def replace():
        try:
            register_service(candidate, controls=first)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(previous, "close", delayed_close)
    worker = threading.Thread(target=replace)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(Refused) as caught:
            register_service(candidate, controls=second)
        assert caught.value.code == "interpret.controls_mismatch"
        assert second.interpretation_service is untouched and not untouched.closed
    finally:
        release.set()
        worker.join(5)
        unregister_service(candidate, controls=first)
        unregister_service(untouched, controls=second)
    assert not worker.is_alive() and not failures
    assert previous.closed and candidate.controls is first
