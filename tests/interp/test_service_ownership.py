"""CPU checks for service ownership and intervention-aware trace caches."""
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
    import caliper.interp
    monkeypatch.setattr(caliper.interp, "publish", lambda *args: None)
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
