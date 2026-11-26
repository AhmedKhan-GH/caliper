"""CPU checks for service ownership and intervention-aware trace caches."""
import sys
import threading
from types import SimpleNamespace

import pytest

from caliper import _controls
from caliper.interp import Tracer
from caliper.interp._backend import make_tiny_backend
from caliper.interp._service import InterpretationService, Refused, register_service, unregister_service
from .test_service import Transport, call


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setattr("caliper.interp._publish._TENSORS", {})
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    backend = make_tiny_backend()
    return backend.model, backend.tokenizer, backend.dictionary


@pytest.mark.parametrize("default_owner", [True, False])
def test_supplied_tracer_refuses_different_owner(resources, default_owner):
    owner = _controls.Controls()
    tracer = Tracer(*resources, **({} if default_owner else {"controls": _controls.Controls()}))
    baseline = tracer.trace("AB", max_targets=0)
    prior_owner = tracer.controls
    service = InterpretationService(tracer=tracer)
    try:
        with pytest.raises(Refused) as caught:
            register_service(service, controls=owner)
        assert caught.value.code == "interpret.controls_mismatch"
        assert owner.interpretation_service is None
        assert not hasattr(service, "controls") and not service.closed
        assert tracer.controls is prior_owner
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
        assert service.tracer.controls is owner
        assert service.model is model and service.tokenizer is tokenizer and service.dictionary is dictionary
        assert trace() is baseline
        route.on_control(None, {"request_id": 2, "op": "intervene", "intervene": {
            "set": {"point": "layers.0.mlp_out", "op": "ablate"}}})
        owner.step()
        ablated = trace()
        assert ablated is not baseline and ablated.trace_id != baseline.trace_id
        assert trace() is ablated
        route.on_control(None, {"request_id": 3, "op": "intervene", "intervene": {"clear_all": True}})
        owner.step()
        assert trace() is baseline
    finally:
        unregister_service(service, controls=owner)
        watched.remove()
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
        owner.step()
        assert transport.replies[-1].status == "ok", transport.replies[-1].error
    finally:
        unregister_service(service)
        candidate.close()
        route.detach()


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
    finally:
        release.set()
        worker.join(5)
        unregister_service(candidate, controls=first)
        unregister_service(untouched, controls=second)
    assert not worker.is_alive() and not failures
