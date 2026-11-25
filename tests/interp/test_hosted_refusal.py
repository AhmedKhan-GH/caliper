"""Hosted refusal preserves committed effects and action evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from caliper import _controls, _wire
from caliper.interp._intervene import Intervention
from caliper.interp._publish import _resolve
from .concurrency_isolation import isolated_test
from .test_controls_concurrency import other_thread
from .test_hosted import hosted_run
from .test_hosted_controls import _feature_rows, _queue_action
from .test_intervention_replay import hook_objects


@pytest.fixture(autouse=True)
def local_transport(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))
    monkeypatch.setattr(_controls._Timers, "at", lambda *args: None)


def edit(route, name, value):
    reply = route.on_control(None, _wire.Control(100, "set", name=name, value=value).to_meta())
    assert reply["status"] == "ok"


def selected_clamp(run, route):
    rows = _feature_rows(run)
    edit(route, "selected_node", rows[0])
    edit(route, "clamp", 0.)
    run.boundary()
    return next(row for row in rows if run.service.graph.nodes[row]["layer"] != run.selection["layer"])


def refuse_hook(run, monkeypatch, layer):
    module = run.model.layers[layer].mlp
    register = module.register_forward_hook

    def refused(callback, **kwargs):
        if isinstance(getattr(callback, "__self__", None), Intervention):
            raise RuntimeError("hosted hook refusal")
        return register(callback, **kwargs)

    monkeypatch.setattr(module, "register_forward_hook", refused)


def test_cross_layer_refusal_retains_old_clamp(hosted_run, monkeypatch):
    run, route = hosted_run
    run.interventions.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "AC"})
    run.boundary()
    second = selected_clamp(run, route)
    record = run._owned_record()
    hook, applied, hooks = record.hook, record.applied_spec, hook_objects(run.model)
    published = _resolve("output.probs")
    comparison = deepcopy(run.inference_status()["comparison"])
    cache = dict(run.interventions.executor._cache)
    identity = run.interventions.executor._cache_identity
    specifications = run.interventions.specifications()
    edit(route, "selected_node", second)
    edit(route, "clamp", 1.)
    refuse_hook(run, monkeypatch, run.service.graph.nodes[second]["layer"])
    with pytest.warns(RuntimeWarning, match="hosted hook refusal"):
        run.boundary()
    assert record.hook is hook and record.applied_spec is applied and not hook.removed
    assert hook_objects(run.model) == hooks
    assert run._owned_record() is record
    assert run.interventions.specifications() == specifications
    assert run.interventions.executor._cache_identity == identity
    assert list(run.interventions.executor._cache) == list(cache)
    assert all(run.interventions.executor._cache[key] is value for key, value in cache.items())
    assert run.inference_status()["comparison"] == comparison
    assert _resolve("output.probs") is published


def test_refused_external_replacement_retains_slider_ownership(hosted_run, monkeypatch):
    run, route = hosted_run
    selected_clamp(run, route)
    record = run._owned_record()
    hook, applied = record.hook, record.applied_spec
    replacement = dict(record.spec, value=record.spec["value"] + 1)
    reply = route.on_control(None, _wire.Control(1, "intervene", intervene={"set": replacement}).to_meta())
    assert reply["status"] == "ok"
    with monkeypatch.context() as patch:
        refuse_hook(run, patch, run.selection["layer"])
        with pytest.warns(RuntimeWarning, match="hosted hook refusal"):
            run.boundary()
    assert record.hook is hook and record.applied_spec is applied and not hook.removed
    assert run._owned_record() is record
    assert "Live slider clamp:" in run.clamp_status.value
    _queue_action(route, "reset", 2)
    run.boundary()
    assert hook.removed and not run.interventions.specifications()


@pytest.mark.parametrize("outcome", ["refused", "deferred"])
@pytest.mark.parametrize("prior", ["none", "same_point", "cross_layer"])
@isolated_test("hosted_run", "monkeypatch")
def test_explicit_clamp_requires_committed_effect(hosted_run, monkeypatch, outcome, prior):
    run, route = hosted_run
    if prior == "none":
        row = _feature_rows(run)[0]
    else:
        row = selected_clamp(run, route)
        if prior == "same_point":
            row = next(i for i, node in enumerate(run.service.graph.nodes)
                       if node["kind"] == "feature" and node["layer"] == run.selection["layer"]
                       and (node["feature"], node["position"]) != (run.selection["feature"], run.selection["position"]))
    edit(route, "selected_node", row)
    run.boundary()
    published = _resolve("output.probs")
    comparison = deepcopy(run.inference_status()["comparison"])
    if outcome == "refused":
        refuse_hook(run, monkeypatch, run.selection["layer"])
    else:
        install = run.controls._install_intervention

        def installed(service, iv, spec, **kwargs):
            install(service, iv, spec, **kwargs)
            if iv.op != "clamp":
                return
            reply = other_thread(lambda: route.on_control(None, _wire.Control(3, "intervene", intervene={
                "set": {"kind": "ablate", "point": "layers.0.resid_pre"}}).to_meta()))
            assert reply["status"] == "ok"

        monkeypatch.setattr(run.controls, "_install_intervention", installed)
    clamps = [spec for spec in run.interventions.specifications() if spec["kind"] == "clamp"]
    _queue_action(route, "apply_clamp", 2)
    with pytest.warns(RuntimeWarning, match="action 'apply_clamp' raised ValueError"):
        run.boundary()
    reply, = [reply for reply in route.transport.replies if reply.request_id == 2]
    assert reply.status == "error" and reply.error["code"] == "action_failed"
    assert run.inference_status()["comparison"] == comparison
    assert _resolve("output.probs") is published
    # Resolve the unrelated concurrent request without a delayed clamp effect.
    run._apply()
    assert [spec for spec in run.interventions.specifications() if spec["kind"] == "clamp"] == clamps
    assert len([reply for reply in route.transport.replies if reply.request_id == 2]) == 1
    assert not run.controls.actions["apply_clamp"].queued


@pytest.mark.parametrize("traffic", ["unrelated", "target", "old_owner", "clear"])
@isolated_test("hosted_run", "monkeypatch")
def test_cross_layer_deferral_preserves_newer_requests(hosted_run, monkeypatch, traffic):
    run, route = hosted_run
    second = selected_clamp(run, route)
    old = run._owned_record()
    hook, applied = old.hook, old.applied_spec
    install = run.controls._install_intervention
    concurrent = []

    def installed(service, iv, spec, **kwargs):
        install(service, iv, spec, **kwargs)
        if concurrent:
            return
        if traffic == "unrelated":
            request = {"set": {"kind": "ablate", "point": "layers.0.resid_pre"}}
        elif traffic == "clear":
            request = {"clear": spec["point"]}
        else:
            request = {"set": dict(spec if traffic == "target" else applied, value=3.)}
        reply = other_thread(lambda: route.on_control(None, _wire.Control(5, "intervene", intervene=request).to_meta()))
        assert reply["status"] == "ok"
        concurrent.append(request)

    edit(route, "selected_node", second)
    edit(route, "clamp", 1.)
    with monkeypatch.context() as patch:
        patch.setattr(run.controls, "_install_intervention", installed)
        run.boundary()
    assert old.hook is hook and old.applied_spec is applied and not hook.removed
    request, = concurrent
    if traffic != "old_owner":
        assert run._owned_record() is old
    if "set" in request:
        spec = request["set"]
        record = run.controls._ivs[(spec["point"], spec["kind"])]
        assert all(record.spec[key] == value for key, value in spec.items())
    else:
        assert run.controls._ivs[(request["clear"], "clamp")].clear
    run.boundary()
    if traffic in ("unrelated", "clear"):
        assert run._owned_record() is old
        assert [spec for spec in run.interventions.specifications() if spec["kind"] == "clamp"] == [applied]
    _queue_action(route, "reset", 7)
    run.boundary()
    expected = [] if traffic == "clear" else [run.interventions.executor.validate(request["set"], resources=False)]
    assert run.interventions.specifications() == expected


def test_reset_does_not_orphan_a_pending_external_refusal(hosted_run, monkeypatch):
    run, route = hosted_run
    selected_clamp(run, route)
    old = run._owned_record()
    spec = dict(old.spec, value=old.spec["value"] + 1)
    assert route.on_control(None, _wire.Control(5, "intervene", intervene={"set": spec}).to_meta())["status"] == "ok"
    with monkeypatch.context() as patch:
        refuse_hook(run, patch, run.selection["layer"])
        with pytest.warns(RuntimeWarning, match="hosted hook refusal"):
            run.reset()
    assert run._owned_record() is old
    run.reset()
    assert not run.interventions.specifications()


@pytest.mark.parametrize("prior", [False, True])
def test_explicit_clamp_stop_keeps_identity_and_single_replies(hosted_run, monkeypatch, prior):
    import json
    from pathlib import Path
    from caliper.interp._memory import CleanStop

    run, route = hosted_run
    row = selected_clamp(run, route) if prior else _feature_rows(run)[0]
    edit(route, "selected_node", row)
    run.boundary()
    owned = run._owned_record()
    hook = None if owned is None else owned.hook
    comparison = deepcopy(run.inference_status()["comparison"])
    failure = CleanStop("explicit clamp resource stop")
    install = run.controls._install_intervention

    def stopped(service, iv, spec, **kwargs):
        if iv.op == "clamp":
            raise failure
        return install(service, iv, spec, **kwargs)

    monkeypatch.setattr(run.controls, "_install_intervention", stopped)
    _queue_action(route, "apply_clamp", 1)
    _queue_action(route, "apply_clamp", 2)
    _queue_action(route, "generate", 3)
    with pytest.raises(CleanStop) as caught:
        run.boundary()
    assert caught.value is failure and run.service.failure is failure
    assert sorted((reply.request_id, reply.status) for reply in route.transport.replies) == [(1, "error"), (2, "error"), (3, "error")]
    assert all(reply.error == {"code": "action_failed", "message": str(failure)} for reply in route.transport.replies)
    assert run.inference_status()["comparison"] == comparison
    if prior:
        assert run._owned_record() is owned and owned.hook is hook and not hook.removed
        assert run.state_dict()["owned_clamp"] == owned.applied_spec
        assert len(run.controls._ivs) == 1
    with pytest.raises(CleanStop) as repeated:
        run.boundary()
    assert repeated.value is failure and len(route.transport.replies) == 3
    checkpoint, = Path(run.managed.dir).glob("checkpoint-*.json")
    if prior:
        assert json.loads(checkpoint.read_text())["owned_clamp"] == owned.applied_spec


def test_accepted_equal_external_request_cannot_regain_slider_ownership(hosted_run, monkeypatch):
    run, route = hosted_run
    selected_clamp(run, route)
    spec = deepcopy(run._owned_clamp)
    run.interventions.set(spec)
    run._apply()
    run.interventions.set(dict(spec, value=spec["value"] + 1))
    with monkeypatch.context() as patch:
        refuse_hook(run, patch, run.selection["layer"])
        with pytest.warns(RuntimeWarning, match="hosted hook refusal"):
            run.boundary()
    assert run._owned_record() is None
    _queue_action(route, "reset", 2)
    run.boundary()
    assert run.interventions.specifications() == [spec]
