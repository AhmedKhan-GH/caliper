"""CPU checks for hosted clamp edits and explicit actions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from caliper import _wire
from caliper.interp._publish import _resolve
from .test_hosted import hosted_run, _trace_at_boundary


@pytest.fixture(autouse=True)
def in_process_transport(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))


@pytest.mark.parametrize("later_success", [False, True])
def test_slider_conflict_defers_generate_and_follower_for_the_whole_pass(hosted_run, monkeypatch, later_success):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    completed, boundaries, injected = [], [], []
    run.actions.append(run.controls.action(
        "after_generate", lambda: completed.append(run.inference_status()["generation"]["id"])))
    install = run.controls._install_intervention
    boundary = run.controls._intervention_boundary

    def installed(service, iv, spec, **kwargs):
        install(service, iv, spec, **kwargs)
        if iv.op == "clamp" and not injected:
            injected.append(spec)
            reply = route.on_control(None, _wire.Control(5, "intervene", intervene={
                "set": {"kind": "ablate", "point": "layers.0.mlp_out"}}).to_meta())
            assert reply["status"] == "ok"

    def applied(*args, **kwargs):
        result = boundary(*args, **kwargs)
        boundaries.append(result[1])
        return result

    update = run.update_controls

    def updated():
        update()
        if later_success:
            run._apply()

    monkeypatch.setattr(run.controls, "_install_intervention", installed)
    monkeypatch.setattr(run.controls, "_intervention_boundary", applied)
    monkeypatch.setattr(run, "update_controls", updated)
    reply = route.on_control(None, _wire.Control(1, "set", name="clamp", value=2.).to_meta())
    assert reply["status"] == "ok" and run.clamp.pending == 2.
    _queue_action(route, "generate", 2)
    _queue_action(route, "after_generate", 3)
    action = run.controls.actions["generate"]
    run_id = action.run_id
    run.boundary()
    assert boundaries == ([True, False, True] if later_success else [True, False])
    assert len(injected) == 1
    assert action.queued and action.run_id == run_id
    assert run.inference_status()["generation"]["id"] == 0
    assert not route.transport.replies and not completed
    assert bool(run.interventions.specifications()) == later_success
    run.boundary()
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(2, "ok"), (3, "ok")]
    assert completed == [1] and not run.controls.queue
    generation = run.inference_status()["generation"]
    assert generation["counts"] == [24, 24]
    assert generation["interventions_hash"] == run.inference_status()["interventions_hash"]
    assert generation["completed_step"] == run.controls.n
    assert run.interventions.specifications()[0] == injected[0]
    run.boundary()
    assert completed == [1] and len(route.transport.replies) == 2


@pytest.mark.parametrize("resource_stop", [False, True])
def test_hosted_replacement_refusal_and_resource_stop_have_distinct_outcomes(hosted_run, resource_stop):
    from caliper.interp._memory import CleanStop

    run, route = hosted_run
    old = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AB"}
    run.interventions.set(old)
    run.boundary()
    record = run.controls._ivs[(old["point"], "patch")]
    hook, specification = record.hook, record.applied_spec
    cache = dict(run.interventions.executor._cache)
    identity = run.interventions.executor._cache_identity
    attempts, recovered = [], []
    failure = CleanStop("source resource stop") if resource_stop else RuntimeError("source refusal")

    def fail(*args):
        if run.interventions.executor.preparing_source:
            attempts.append(1)
            raise failure

    def recover():
        run.interventions.clear(old["point"])
        run._apply()
        recovered.append(1)

    run.actions.append(run.controls.action("recover", recover))
    run.interventions.set(dict(old, source_prompt="AC"))
    _queue_action(route, "recover", 1)
    _queue_action(route, "generate", 2)
    _queue_action(route, "generate", 3)
    observer = run.model.register_forward_pre_hook(fail)
    try:
        if resource_stop:
            with pytest.raises(CleanStop) as caught:
                run.boundary()
            assert caught.value is failure and run.service.failure is failure
            assert record.replace and record.spec["source_prompt"] == "AC" and record.error is None
            replies = route.transport.replies
            assert [(reply.request_id, reply.status) for reply in replies] == [(1, "error"), (2, "error"), (3, "error")]
            assert all(reply.error == {"code": "action_failed", "message": "source resource stop"}
                       for reply in replies)
            with pytest.raises(CleanStop) as repeated:
                run.boundary()
            assert repeated.value is failure and len(replies) == 3 and not run.controls.queue
        else:
            with pytest.warns(RuntimeWarning, match="source refusal"):
                run.boundary()
            assert not record.replace and record.spec == specification
            assert record.error_code == "intervene.hook_failed" and run.service.failure is None
            assert not route.transport.replies
        assert not recovered and run.inference_status()["generation"]["id"] == 0
        assert record.hook is hook and record.applied_spec is specification and not hook.removed
        assert run.interventions.executor._cache_identity is identity
        assert list(run.interventions.executor._cache) == list(cache)
        assert all(run.interventions.executor._cache[key] is value for key, value in cache.items())
        if not resource_stop:
            run.boundary()
            assert recovered == [1] and hook.removed and not run.interventions.specifications()
            assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok"), (2, "ok"), (3, "ok")]
            assert run.inference_status()["generation"]["id"] == 1
            assert not run.controls.queue
        assert attempts == [1]
    finally:
        observer.remove()


@pytest.mark.parametrize("refusal", ["selection", "maximum", "intervention", "row", "prompt"])
def test_refused_slider_edit_cannot_clamp_a_later_selection(hosted_run, monkeypatch, refusal):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    rows = [i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"]
    assert len(rows) >= 2
    run.selected.set(rows[0])
    run.boundary()
    baseline = _resolve("output.probs").clone()
    comparison = deepcopy(run.inference_status()["comparison"])
    with monkeypatch.context() as patch:
        if refusal == "selection":
            run.selected.set(-1)
        elif refusal == "maximum":
            patch.setattr(run.index, "show", lambda *args: {"maximum": 0.})
        elif refusal == "intervention":
            patch.setattr(run.controls, "_intervene", lambda *args: ("refused", "test clamp refusal"))
        elif refusal == "row":
            run.selected.set(511)
        else:
            run.prompt.set("A" * 33)
        reply = route.on_control(None, _wire.Control(1, "set", name="clamp", value=2.).to_meta())
        assert reply["status"] == "ok"
        run.boundary()
        assert not run.interventions.specifications()
        assert run.clamp._version == run._clamp_version
        assert run.status.value != "ready"
    run.prompt.set("AB")
    run.selected.set(rows[1])
    run.boundary()
    assert not run.interventions.specifications()
    assert run.inference_status()["comparison"] == comparison
    assert torch.equal(_resolve("output.probs"), baseline)


def _feature_rows(run):
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    return [i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"]


def _queue_action(route, name, request_id):
    reply = route.on_control(None, _wire.Control(request_id, "run", name=name).to_meta())
    assert reply["status"] == "queued"


@pytest.mark.parametrize("continuation", ["reset", "restore", "fork"])
def test_slider_registration_refusal_preserves_owner_and_reset(hosted_run, monkeypatch, tmp_path, continuation):
    import caliper
    from caliper import _controls, _managed
    from caliper.interp import load_dictionary
    from caliper.interp._hosted import HostedInference
    from caliper.interp._index import FeatureIndex
    from caliper.interp._intervene import Intervention
    from .test_intervention_replay import hook_objects
    from .test_service import Transport

    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.clamp.set(0.)
    run.boundary()
    record = run._owned_record()
    hook, applied = record.hook, record.applied_spec
    before, hooks = _resolve("output.probs").clone(), hook_objects(run.model)
    assert not torch.equal(before[0], before[1])
    module = run.model.layers[run.selection["layer"]].mlp
    register, attempts = module.register_forward_hook, []

    def fail(callback, **kwargs):
        if isinstance(getattr(callback, "__self__", None), Intervention):
            attempts.append(1)
            raise RuntimeError("slider hook registration refused")
        return register(callback, **kwargs)

    reply = route.on_control(None, _wire.Control(1, "set", name="clamp", value=1.).to_meta())
    assert reply["status"] == "ok"
    _queue_action(route, "reset", 2)
    with monkeypatch.context() as patch:
        patch.setattr(module, "register_forward_hook", fail)
        with pytest.warns(RuntimeWarning, match="slider hook registration refused"):
            run.boundary()
    assert attempts == [1] and not route.transport.replies
    assert record.hook is hook and record.applied_spec is applied and not hook.removed
    assert hook_objects(run.model) == hooks
    assert run._owned_record() is record and run._owned_clamp is record.spec
    assert record.spec == applied and not record.replace
    assert record.error == "RuntimeError: slider hook registration refused"
    assert record.error_code == "intervene.hook_failed"
    assert "Live slider clamp:" in run.clamp_status.value
    torch.testing.assert_close(_resolve("output.probs"), before, rtol=0, atol=0)
    checkpoint = run.managed.checkpoint(run.model, None)
    saved = json.loads(Path(checkpoint).read_bytes())
    assert saved["owned_clamp"] == applied
    assert saved["interventions"] == run.interventions.state_dict()
    child = None
    child_route = None
    try:
        if continuation == "restore":
            run.load_checkpoint(saved)
            run.interventions.load_state_dict(saved["interventions"], restore_active=True)
            run._restore_owned_clamp(run.restored["owned_clamp"])
            assert run.interventions.state_dict() == saved["interventions"]
            assert run._owned_record() is not None
        elif continuation == "fork":
            model = type(run.model)()
            model.load_state_dict(run.model.state_dict())
            tokenizer, folder = run.tokenizer, run.dictionary_path
            run.close()
            controls = _controls.Controls()
            monkeypatch.setattr(caliper, "_CTL", controls)
            monkeypatch.setattr(_controls, "CONTROLS", controls)
            fork_folder = tmp_path / "refused-fork"
            fork_folder.mkdir()
            managed = _managed.ManagedRun("refused-fork", str(fork_folder),
                                         str(fork_folder / "link"), resume=checkpoint)
            monkeypatch.setattr(_managed, "current", lambda: managed)
            dictionary = load_dictionary(folder, model=model)
            index = FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary)
            child = HostedInference("tiny-lm", folder, "AB", resources=(model, tokenizer, dictionary, index, None))
            child_route = controls.attach(Transport())
            child.start()
            run, route = child, child_route
            assert run.interventions.state_dict() == saved["interventions"]
            assert run.state_dict()["owned_clamp"] == applied
            assert run._owned_record() is not None
            torch.testing.assert_close(_resolve("output.probs"), before, rtol=0, atol=0)
            _queue_action(route, "reset", 2)
        run.boundary()
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(2, "ok")]
        assert not run.interventions.specifications() and run._owned_record() is None
        assert run.clamp.value == 0. and "No slider clamp." in run.clamp_status.value
        torch.testing.assert_close(_resolve("output.probs")[0], _resolve("output.probs")[1], rtol=0, atol=0)
        run.boundary()
        assert len(route.transport.replies) == 1 and not run.controls.queue
    finally:
        if child is not None:
            child.close()
        if child_route is not None:
            child_route.detach()


@pytest.mark.parametrize("draft", ["", "A" * 33])
def test_reset_removes_clamp_with_invalid_draft_without_false_output_evidence(hosted_run, draft):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    assert run._owned_clamp is not None
    previous = deepcopy(run.inference_status()["comparison"])
    published = _resolve("output.probs")
    run.prompt.set(draft)
    run.boundary()
    _queue_action(route, "reset", 2)
    run.boundary()
    reply, = [reply for reply in route.transport.replies if reply.request_id == 2]
    assert reply.status == "ok"
    assert run.prompt.value == draft and run.checkpoint_prompt == "AB"
    assert run._owned_clamp is None and not run.interventions.specifications()
    assert run.clamp.value == 0 and "No slider clamp" in run.clamp_status.value
    assert run.status.value != "ready"
    assert run.inference_status()["comparison"] == previous
    assert _resolve("output.probs") is published
    run.prompt.set("AB")
    run.boundary()
    assert run.status.value == "ready"
    assert run.inference_status()["comparison"]["interventions_hash"] != previous["interventions_hash"]
    torch.testing.assert_close(_resolve("output.probs")[0], _resolve("output.probs")[1], rtol=0, atol=0)


def _count_clamp_work(run, monkeypatch):
    calls = {"requests": [], "installations": [], "comparisons": []}
    intervene = run.controls._intervene
    install = run.interventions.executor.install
    compare = run.output.compare

    def requested(request):
        calls["requests"].append(deepcopy(request))
        return intervene(request)

    def installed(request, **kwargs):
        calls["installations"].append(deepcopy(request))
        return install(request, **kwargs)

    def compared(prompt):
        calls["comparisons"].append(prompt)
        return compare(prompt)

    monkeypatch.setattr(run.controls, "_intervene", requested)
    monkeypatch.setattr(run.interventions.executor, "install", installed)
    monkeypatch.setattr(run.output, "compare", compared)
    return calls


def _assert_action_completion(run, route, request_id, previous):
    reply, = [reply for reply in route.transport.replies if reply.request_id == request_id]
    assert reply.op == "run" and reply.name == "apply_clamp" and reply.status == "ok"
    status = route.on_control(None, _wire.Control(request_id + 100, "interpret", interpretation={
        "version": 1, "method": "trace.status", "params": {}}).to_meta())
    assert status["status"] == "ok"
    evidence = status["result"]["inference"]
    assert evidence["comparison"]["completed_step"] == run.controls.n
    assert evidence["comparison"]["completed_step"] > previous["completed_step"]
    assert evidence["comparison"]["prompt_hash"] == evidence["prompt_hash"]
    assert evidence["comparison"]["interventions_hash"] == evidence["interventions_hash"]


@pytest.mark.parametrize("value", [0., 2.])
def test_pending_edit_and_action_share_one_clamp_and_comparison(hosted_run, monkeypatch, value):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    previous = deepcopy(run.inference_status()["comparison"])
    calls = _count_clamp_work(run, monkeypatch)
    reply = route.on_control(None, _wire.Control(1, "set", name="clamp", value=value).to_meta())
    assert reply["status"] == "ok" and run.clamp.pending == value
    _queue_action(route, "apply_clamp", 2)
    assert not any(calls.values())
    run.boundary()
    assert len(calls["requests"]) == len(calls["installations"]) == 1
    assert calls["comparisons"] == ["AB"]
    assert run.interventions.specifications() == [run._owned_clamp]
    _assert_action_completion(run, route, 2, previous)


def test_repeated_zero_action_publishes_one_fresh_comparison(hosted_run, monkeypatch):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    owned = run._owned_clamp
    previous = deepcopy(run.inference_status()["comparison"])
    published = _resolve("output.probs")
    calls = _count_clamp_work(run, monkeypatch)
    _queue_action(route, "apply_clamp", 2)
    run.boundary()
    assert calls == {"requests": [], "installations": [], "comparisons": ["AB"]}
    assert run._owned_clamp is owned
    assert run.interventions.specifications() == [owned] and owned["value"] == 0.
    assert _resolve("output.probs") is not published
    assert torch.equal(_resolve("output.probs"), published)
    _assert_action_completion(run, route, 2, previous)


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("refusal", ["prompt", "selection", "row", "maximum", "nonfinite"])
def test_repeated_action_refusal_preserves_owner_and_consumes_edit(hosted_run, monkeypatch, edit, refusal):
    run, route = hosted_run
    rows = _feature_rows(run)
    run.selected.set(rows[0])
    run.boundary()
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    owned = run._owned_clamp
    previous = deepcopy(run.inference_status()["comparison"])
    published = _resolve("output.probs")
    with monkeypatch.context() as patch:
        if refusal == "prompt":
            run.prompt.set("A" * 33)
        elif refusal == "selection":
            run.selected.set(-1)
        elif refusal == "row":
            run.selected.set(511)
        else:
            maximum = float("nan") if refusal == "nonfinite" else 0.
            patch.setattr(run.index, "show", lambda *args: {"maximum": maximum})
        if edit:
            reply = route.on_control(None, _wire.Control(2, "set", name="clamp", value=2.).to_meta())
            assert reply["status"] == "ok"
        _queue_action(route, "apply_clamp", 3)
        with pytest.warns(RuntimeWarning, match="action 'apply_clamp' raised ValueError"):
            run.boundary()
        reply, = [reply for reply in route.transport.replies if reply.request_id == 3]
        assert reply.status == "error" and reply.error["code"] == _wire.ERR_ACTION_FAILED
        assert run.clamp._version == run._clamp_version
        assert run._owned_clamp is owned
        assert run.inference_status()["comparison"] == previous
        assert _resolve("output.probs") is published
    run.prompt.set("AB")
    run.selected.set(rows[1])
    run.boundary()
    assert run.interventions.specifications() == [owned]
    assert run.inference_status()["comparison"] == previous
    assert _resolve("output.probs") is published


@pytest.mark.parametrize("origin", ["prior", "comparison"])
def test_repeated_action_resource_stop_has_no_success_evidence(hosted_run, monkeypatch, origin):
    from caliper.interp._memory import CleanStop
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    previous = deepcopy(run.inference_status()["comparison"])
    published = _resolve("output.probs")
    owned = run._owned_clamp
    calls = _count_clamp_work(run, monkeypatch)
    failure = CleanStop("clamp resource stop")

    def stop(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run.output, "guard", SimpleNamespace(check=stop))
    monkeypatch.setattr(run.output, "generate", lambda *args: pytest.fail("later action ran"))
    _queue_action(route, "apply_clamp", 2)
    _queue_action(route, "generate", 3)
    if origin == "prior":
        run._latch_failure(failure)
    with pytest.raises(CleanStop) as caught:
        run.boundary()
    assert caught.value is failure
    assert sorted((reply.request_id, reply.status) for reply in route.transport.replies
                  if reply.request_id in (2, 3)) == [(2, "error"), (3, "error")]
    assert not run.controls.queue and run._owned_clamp is owned
    assert run.inference_status()["comparison"] == previous
    assert _resolve("output.probs") is published
    assert calls["requests"] == calls["installations"] == []
    assert calls["comparisons"] == ([] if origin == "prior" else ["AB"])


def test_apply_clamp_ablates_initial_and_reset_zero_at_the_boundary(hosted_run, monkeypatch):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    baseline = _resolve("output.probs").clone()
    replies = []
    monkeypatch.setattr(route, "_reply", lambda conn, metadata: replies.append(metadata))
    assert run.clamp.value == 0.
    version = run.clamp._version
    _queue_action(route, "apply_clamp", 1)
    assert not run.interventions.specifications()
    run.boundary()
    assert run.clamp._version == version
    assert run.interventions.specifications()[0]["value"] == 0.
    assert replies[-1]["status"] == "ok"
    assert not torch.equal(_resolve("output.probs"), baseline)
    assert run.inference_status()["comparison"]["completed_step"] == run.controls.n
    _queue_action(route, "reset", 2)
    run.boundary()
    assert run.clamp.value == 0. and not run.interventions.specifications()
    assert "No slider clamp." in run.clamp_status.value
    assert torch.equal(_resolve("output.probs"), baseline)
    version = run.clamp._version
    _queue_action(route, "apply_clamp", 3)
    run.boundary()
    assert run.clamp._version == version
    assert run.interventions.specifications()[0]["value"] == 0.
    assert replies[-1]["status"] == "ok"
    assert not torch.equal(_resolve("output.probs"), baseline)


@pytest.mark.parametrize("same_point", [False, True])
def test_feature_switch_keeps_owner_and_explicit_apply_moves_clamp(hosted_run, same_point):
    run, route = hosted_run
    rows = _feature_rows(run)
    first = run.service.graph.nodes[rows[0]]
    second = next(i for i in rows[1:]
                  if (run.service.graph.nodes[i]["layer"] == first["layer"]) == same_point)
    run.selected.set(rows[0])
    run.clamp.set(2.)
    run.boundary()
    owned = deepcopy(run._owned_clamp)
    before = _resolve("output.probs").clone()
    comparison = deepcopy(run.inference_status()["comparison"])
    run.selected.set(second)
    run.boundary()
    assert run.clamp.value == 0.
    assert run.interventions.specifications() == [owned]
    assert torch.equal(_resolve("output.probs"), before)
    assert run.inference_status()["comparison"] == comparison
    selection = run.selection
    assert (f"Selected layer {selection['layer']}, feature {selection['feature']}, "
            f"position {selection['position']}.") in run.clamp_status.value
    assert (f"Live slider clamp: {owned['point']}, feature {owned['feature']}, "
            f"position {owned['positions'][0]}, value {owned['value']:g}.") in run.clamp_status.value
    assert "Clamp input: 0 times" in run.clamp_status.value
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    spec, = run.interventions.specifications()
    assert spec["feature"] == selection["feature"] and spec["positions"] == [selection["position"]]
    assert spec["point"] == run.dictionary.metadata["write_points"][selection["layer"]]
    assert spec["value"] == 0.
    assert spec != owned


def test_selection_preserves_slider_edit_received_during_boundary(hosted_run, monkeypatch):
    run, route = hosted_run
    rows = _feature_rows(run)
    run.selected.set(rows[0])
    run.clamp.set(2.)
    run.boundary()
    owned = deepcopy(run._owned_clamp)
    run.selected.set(rows[1])
    update_selection = run._update_selection
    def select_and_receive():
        update_selection()
        reply = route.on_control(None, _wire.Control(1, "set", name="clamp", value=1.).to_meta())
        assert reply["status"] == "ok"
    with monkeypatch.context() as patch:
        patch.setattr(run, "_update_selection", select_and_receive)
        run.boundary()
    assert run.clamp.pending == 1.
    assert run.interventions.specifications() == [owned]
    run.boundary()
    assert run.clamp.value == 1. and run.clamp.pending is None
    assert run._owned_clamp != owned


def test_apply_clamp_refusal_has_final_reply_and_no_later_effect(hosted_run, monkeypatch):
    run, route = hosted_run
    rows = _feature_rows(run)
    replies = []
    monkeypatch.setattr(route, "_reply", lambda conn, metadata: replies.append(metadata))
    _queue_action(route, "apply_clamp", 1)
    with pytest.warns(RuntimeWarning, match="clamp requires a selected feature"):
        run.boundary()
    assert replies[-1]["status"] == "error"
    assert replies[-1]["error"]["code"] == _wire.ERR_ACTION_FAILED
    run.selected.set(rows[0])
    run.boundary()
    assert not run.interventions.specifications()
