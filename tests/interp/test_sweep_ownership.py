"""CPU sweep publication across queued and active watch changes."""
from contextvars import Context
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from caliper import _programs as P
from caliper._controls import Controls
from caliper._watch import WatchHandle
from caliper.interp import _publish, publish
from caliper.interp._hosted import HostedInference, HostedService
from caliper.interp._service import InterpretationService, register_service, unregister_service
from caliper.interp._sweep import patch_sweep
from .fixture import TinyTokenizer
from .test_publication import registry
from .test_service import Transport, call
from .test_sweep import PlantedCopy


PARAMS = {"clean_prompt": "AZ", "corrupted_prompt": "BZ",
          "metric": {"positive_token": 65, "negative_token": 66}}


@pytest.fixture
def sweep_case(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "0")
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    cases = []

    def make(kind="hosted", *, watched=True):
        model, tokenizer, controls = PlantedCopy(), TinyTokenizer(), Controls()
        handle = WatchHandle()
        if watched:
            P.attach(model, None, handle)
        interventions = controls.interventions_on(model, handle)
        interventions.configure(tokenizer=tokenizer)
        owner = None
        if kind == "hosted":
            owner = HostedInference("tiny-lm", tmp_path, "AZ", resources=(model, tokenizer, None, None, None))
            owner.controls, owner.interventions = controls, interventions
            for attribute, name, value in (
                ("prompt", "prompt", "AZ"), ("selected", "selected_node", -1),
                ("clamp", "clamp", 0.), ("clamp_status", "clamp_status", ""),
                ("status", "output.status", "ready"),
            ):
                setattr(owner, attribute, controls.param(name, value))
            owner._clamp_version = owner.clamp._version
            owner._selected_version = owner.selected._version
            # Keep both hosted ownership checks without unrelated output forwards.
            monkeypatch.setattr(owner, "_update_output", lambda **kwargs: None)
            service = owner.service = HostedService(owner)
        else:
            service = InterpretationService(model=model, handlers={
                "patch.sweep": partial(patch_sweep, model, tokenizer, interventions)})
        register_service(service, controls=controls)
        transport = Transport()
        route = controls.attach(transport)
        forwards = []
        hook = model.register_forward_hook(lambda *args: forwards.append(interventions.suspended))
        case = SimpleNamespace(model=model, tokenizer=tokenizer, controls=controls, handle=handle,
                               interventions=interventions, owner=owner, service=service,
                               transport=transport, route=route, forwards=forwards, hook=hook)
        cases.append(case)
        return case

    yield make
    for case in reversed(cases):
        unregister_service(case.service, controls=case.controls)
        case.hook.remove()
        case.route.detach()
        case.handle.remove()
        current = case.controls.intervention_service
        if current is not None:
            current.remove()


def current_publication():
    publish("patch.grid", torch.full((1, 1), 7.))
    publish("custom.current", torch.ones(1))
    return dict(_publish._TENSORS), dict(_publish._VERSIONS)


def change_publication_watch(model, transition="replace"):
    def change():
        P.detach(model)
        if transition != "remove":
            P.attach(model if transition == "reattach" else PlantedCopy(), None)
        return current_publication()

    # The replacement publication belongs to a separate request context.
    return Context().run(change)


def assert_publication_unchanged(before):
    tensors, versions = before
    assert set(_publish._TENSORS) == set(tensors)
    assert all(_publish._TENSORS[name] is value for name, value in tensors.items())
    assert _publish._VERSIONS == versions


def assert_terminal_refusal(case, code):
    reply, = case.transport.replies
    assert reply.request_id == 1 and reply.status == "error", (reply, _publish._resolve("patch.grid"))
    assert reply.result is None and reply.error["code"] == code
    assert not case.service.pending and not case.service.running
    assert case.service.running_thread is None and case.service.failure is None
    assert case.service._request_stack is None and not case.service._borrowed
    assert _publish._LEASE.get() is None
    assert not case.interventions.suspended and case.interventions._readers == 0
    assert all(not block._forward_pre_hooks for block in case.model.layers)
    if case.owner is not None:
        assert not case.service.boundary_running and case.service.boundary_thread is None
    case.controls.step()
    assert len(case.transport.replies) == 1


@pytest.mark.parametrize("kind,transition", [
    ("hosted", "replace"), ("hosted", "remove"), ("hosted", "reattach"), ("generic", "replace"),
])
def test_queued_sweep_refuses_old_watch_before_model_work(sweep_case, kind, transition):
    case = sweep_case(kind)
    assert call(case.route, "patch.sweep", PARAMS)["status"] == "queued"
    case.handle.remove()
    replacement = None if transition == "remove" else (
        case.model if transition == "reattach" else PlantedCopy())
    if replacement is not None:
        P.attach(replacement, None)
        case.controls.interventions_on(replacement)
    before = current_publication()
    case.controls.step()
    code = "interpret.watch_changed" if transition == "replace" else "interpret.refused"
    assert_terminal_refusal(case, code)
    assert not case.forwards
    assert P._STATE["model"] is replacement
    assert_publication_unchanged(before)


@pytest.mark.parametrize("kind", ["hosted", "generic"])
@pytest.mark.parametrize("transition", ["replace", "remove", "reattach"])
def test_active_sweep_refuses_changed_publication_watch(sweep_case, monkeypatch, kind, transition):
    case = sweep_case(kind)
    original = case.model.forward
    before = []

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(case.forwards) == 1:
            before.append(change_publication_watch(case.model, transition))
        return result

    monkeypatch.setattr(case.model, "forward", forward)
    assert call(case.route, "patch.sweep", PARAMS)["status"] == "queued"
    case.controls.step()
    assert_terminal_refusal(case, "interpret.watch_changed")
    assert case.forwards == [True, True, True]
    assert_publication_unchanged(before[0])


@pytest.mark.parametrize("kind", ["hosted", "generic"])
@pytest.mark.parametrize("watched", [False, True])
def test_sweep_keeps_valid_service_and_headless_publication(sweep_case, kind, watched):
    case = sweep_case(kind, watched=watched)
    case.interventions.set({"kind": "ablate", "point": "layers.3.mlp"})
    case.controls.step()
    before = case.interventions.state_dict()
    assert call(case.route, "patch.sweep", PARAMS)["status"] == "queued"
    case.controls.step()
    reply, = case.transport.replies
    assert reply.status == "ok", reply.error
    assert reply.result["normalizer"] == .5
    assert reply.result["recovery"] == _publish._resolve("patch.grid").tolist()
    assert case.forwards == [True, True, True]
    assert case.interventions.state_dict() == before
    assert not case.interventions.suspended and case.interventions._readers == 0
    assert not case.service.pending and _publish._LEASE.get() is None
    if not watched:
        grid = _publish._resolve("patch.grid")
        P.attach(case.model, None)
        assert _publish._resolve("patch.grid") is grid


def test_service_scope_precedes_sweep_handler_work(sweep_case):
    case = sweep_case("generic")
    sweep = case.service.handlers["patch.sweep"]
    before = []

    def handler(**params):
        before.append(change_publication_watch(case.model, "reattach"))
        return sweep(**params)

    case.service.handlers["patch.sweep"] = handler
    assert call(case.route, "patch.sweep", PARAMS)["status"] == "queued"
    case.controls.step()
    assert_terminal_refusal(case, "interpret.watch_changed")
    assert not case.forwards
    assert_publication_unchanged(before[0])


@pytest.mark.parametrize("phase", ["before", "forward", "commit"])
def test_direct_sweep_retains_model_and_publication_epoch(sweep_case, monkeypatch, phase):
    case = sweep_case("generic")
    before = []
    if phase == "before":
        before.append(change_publication_watch(case.model))
    elif phase == "forward":
        forward = case.model.forward

        def changed(*args, **kwargs):
            result = forward(*args, **kwargs)
            if len(case.forwards) == 1:
                before.append(change_publication_watch(case.model, "reattach"))
            return result

        monkeypatch.setattr(case.model, "forward", changed)
    else:
        prepare = _publish._prepare_many

        def prepared(values):
            result = prepare(values)
            if "patch.grid" in values and values["patch.grid"].shape == (4, 3):
                before.append(change_publication_watch(case.model))
            return result

        monkeypatch.setattr(_publish, "_prepare_many", prepared)
    with pytest.raises(RuntimeError) as caught:
        patch_sweep(case.model, case.tokenizer, case.interventions, **PARAMS)
    assert caught.value.code == "interpret.watch_changed"
    assert case.forwards == ([] if phase == "before" else [True, True, True])
    assert_publication_unchanged(before[0])
    assert not case.interventions.suspended and _publish._LEASE.get() is None
    assert all(not block._forward_pre_hooks for block in case.model.layers)


def test_hosted_detach_retains_resources_until_sweep_cleanup(sweep_case, monkeypatch):
    case = sweep_case()
    forward = case.model.forward
    before = []

    def detached(*args, **kwargs):
        result = forward(*args, **kwargs)
        if len(case.forwards) == 1:
            case.handle.remove()
            assert case.interventions._readers == 1
            assert case.interventions in case.controls._iv_retiring
            assert case.interventions.executor.tokenizer is case.tokenizer
            before.append(change_publication_watch(case.model))
        return result

    monkeypatch.setattr(case.model, "forward", detached)
    assert call(case.route, "patch.sweep", PARAMS)["status"] == "queued"
    case.controls.step()
    assert_terminal_refusal(case, "interpret.watch_changed")
    assert case.forwards == [True, True, True]
    assert not case.controls._iv_retiring and case.interventions._retired is None
    assert case.interventions.executor.tokenizer is None
    assert_publication_unchanged(before[0])
    case.controls.interventions_on(P._STATE["model"])
