"""Hosted clamps retain watch ownership across requests and boundaries."""
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from caliper import _ablate, _controls
from caliper.interp._intervene import InterventionError
from caliper.interp._publish import _resolve
from .concurrency_isolation import isolated_test
from .fixture import TinyLM
from .test_controls_concurrency import Job, other_thread
from .test_hosted import hosted_run
from .test_hosted_controls import _feature_rows, _queue_action
from .test_intervention_replay import hook_objects


@pytest.fixture(autouse=True)
def local_transport(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))
    monkeypatch.setattr(_controls._Timers, "at", lambda *args: None)


def select_clamp(run, prior):
    rows = _feature_rows(run)
    run.selected.set(rows[0])
    run.boundary()
    if prior != "fresh":
        run.apply_clamp()
    if prior == "cross_layer":
        run.selected.set(next(row for row in rows
                              if run.service.graph.nodes[row]["layer"] != run.selection["layer"]))
        run.boundary()
    run.clamp.set(0. if prior == "unchanged" else 1.)


def replacement_watch(run):
    model = TinyLM()
    model.load_state_dict(run.model.state_dict())
    dictionary = deepcopy(run.dictionary)
    service = run.controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=run.tokenizer,
                      dictionary_id=run.interventions.executor.dictionary_id)
    assert model is not run.model and dictionary is not run.dictionary
    assert service.executor.dictionary_id == run.interventions.executor.dictionary_id
    point = dictionary.metadata["write_points"][run.selection["layer"]]
    spec = {"kind": "clamp", "point": point, "feature": run.selection["feature"], "value": 3.}
    service.set(spec)
    with run.controls.lock:
        run.controls._apply_interventions()
    service.set(dict(spec, value=4.))
    service.set({"kind": "ablate", "point": "layers.0.resid_pre", "from_step": run.controls.n + 20})
    return service


def watch_snapshot(run, service):
    ctl = run.controls
    with ctl.lock:
        records = tuple((key, iv, iv.spec, iv.hook, iv.applied_spec) for key, iv in ctl._ivs.items())
        return {
            "state": service._state_dict(), "records": records,
            "revision": ctl._iv_revision, "due": ctl._iv_due,
            "failures": ctl._iv_boundary_failures, "counters": dict(ctl.counters),
            "hooks": hook_objects(service.model), "old_hooks": hook_objects(run.model),
            "dictionary": service.executor._dictionary_status(),
            "comparison": deepcopy(run._comparison), "published": _resolve("output.probs"),
        }


def assert_unchanged(run, service, before):
    after = watch_snapshot(run, service)
    assert after.pop("published") is before["published"]
    assert after == {key: value for key, value in before.items() if key != "published"}
    for key, iv, spec, hook, applied in before["records"]:
        assert run.controls._ivs[key] is iv and iv.spec is spec
        assert iv.hook is hook and iv.applied_spec is applied
        if hook is not None:
            assert not hook.removed


@pytest.mark.parametrize("prior", ["fresh", "same_point", "cross_layer", "unchanged"])
@pytest.mark.parametrize("entry", ["action", "edit"])
def test_detached_clamp_preserves_replacement_watch(hosted_run, prior, entry):
    run, _ = hosted_run
    select_clamp(run, prior)
    replacement = replacement_watch(run)
    before = watch_snapshot(run, replacement)
    try:
        assert run.interventions.executor.dictionary is None
        if entry == "action":
            # Exercise explicit submission without a pending slider edit.
            run._clamp_version = run.clamp._version
            with pytest.warns(RuntimeWarning, match="no longer attached"):
                ok, message = run.controls._run_action(run.controls.actions["apply_clamp"])
            assert not ok and "no longer attached" in message
            # A refused action still counts as one action invocation.
            before["counters"]["actions_run"] = before["counters"].get("actions_run", 0) + 1
        else:
            with pytest.raises(ValueError, match="no longer attached"):
                run.update_controls()
        with pytest.raises(ValueError, match="no longer attached"):
            run.update_output(force=True)
        assert_unchanged(run, replacement, before)
    finally:
        replacement.remove()


@pytest.mark.parametrize("prior", ["fresh", "same_point"])
def test_detached_clamp_has_one_terminal_action_error(hosted_run, monkeypatch, prior):
    run, route = hosted_run
    select_clamp(run, prior)
    run._clamp_version = run.clamp._version
    action = run.controls.actions["apply_clamp"]
    apply = action.fn
    replacements = []

    def detached():
        replacement = replacement_watch(run)
        replacements.append(replacement)
        before = watch_snapshot(run, replacement)
        try:
            return apply()
        finally:
            assert_unchanged(run, replacement, before)

    monkeypatch.setattr(action, "fn", detached)
    _queue_action(route, "apply_clamp", 11)
    try:
        with pytest.warns(RuntimeWarning, match="no longer attached"):
            run.boundary()
        reply, = [reply for reply in route.transport.replies if reply.request_id == 11]
        assert reply.status == "error" and reply.error["code"] == "action_failed"
        assert "no longer attached" in reply.error["message"]
        assert not action.queued and not action.running
    finally:
        for replacement in replacements:
            replacement.remove()


@isolated_test("hosted_run", "monkeypatch")
def test_detachment_during_clamp_installation_does_not_count_a_new_watch_failure(hosted_run, monkeypatch):
    run, _ = hosted_run
    select_clamp(run, "fresh")
    install = run.controls._install_intervention
    boundary = run.controls._intervention_boundary
    replacements, snapshots = [], []

    def installed(service, iv, spec, **kwargs):
        install(service, iv, spec, **kwargs)
        other_thread(service.remove)

    def detached_boundary():
        with monkeypatch.context() as patch:
            patch.setattr(run.controls, "_install_intervention", installed)
            result = boundary()
        # The detached operation now released its resource lease. Complete a
        # new watch before the hosted caller consumes the boundary result.
        with monkeypatch.context() as patch:
            patch.setattr(run.controls, "_intervention_boundary", boundary)
            with run.controls._outside_lock():
                replacement = other_thread(lambda: replacement_watch(run))
                replacements.append(replacement)
                snapshots.append(watch_snapshot(run, replacement))
        return result

    monkeypatch.setattr(run.controls, "_intervention_boundary", detached_boundary)
    try:
        with pytest.raises(ValueError, match="no longer attached"):
            run._set_clamp(require_commit=True)
        assert len(replacements) == 1
        assert_unchanged(run, replacements[0], snapshots[0])
    finally:
        for replacement in replacements:
            replacement.remove()


@pytest.mark.parametrize("prior", ["fresh", "same_point", "unchanged"])
@pytest.mark.parametrize("operation", ["configure", "watch"])
@isolated_test("hosted_run", "monkeypatch")
def test_clamp_refuses_lifecycle_before_request_validation(hosted_run, monkeypatch, prior, operation):
    run, _ = hosted_run
    select_clamp(run, prior)
    service = run.interventions
    entered, release = Event(), Event()
    validations = []
    validate = service.executor.validate

    def validated(*args, **kwargs):
        validations.append(kwargs.get("resources", True))
        return validate(*args, **kwargs)

    def gate():
        entered.set()
        assert release.wait(5), "lifecycle fixture exceeded its finite wait"

    monkeypatch.setattr(service.executor, "validate", validated)
    if operation == "configure":
        configure = service.executor.configure

        def configured(**kwargs):
            gate()
            return configure(**kwargs)

        monkeypatch.setattr(service.executor, "configure", configured)
        invoke = lambda: service.configure(dictionary=run.dictionary, tokenizer=run.tokenizer)
    else:
        model = TinyLM()
        shapes = _ablate.record_shapes

        def recorded(target):
            assert target is model
            gate()
            return shapes(target)

        monkeypatch.setattr(_ablate, "record_shapes", recorded)
        invoke = lambda: run.controls.interventions_on(model)
    job = Job(invoke)
    try:
        assert entered.wait(5), "lifecycle fixture did not reach its barrier"
        before = watch_snapshot(run, service)

        def request():
            with pytest.raises(ValueError, match="lifecycle operation is in progress"):
                run._set_clamp(require_commit=True)

        other_thread(request)
        assert not validations
        assert_unchanged(run, service, before)
    finally:
        release.set()
        job.join()
        if operation == "watch" and job.result is not None:
            job.result.remove()
    job.success()


@pytest.mark.parametrize("phase", ["before_boundary", "after_boundary"])
@pytest.mark.parametrize("prior", ["fresh", "same_point", "cross_layer"])
@isolated_test("hosted_run", "monkeypatch")
def test_watch_replacement_between_clamp_phases_preserves_new_requests(hosted_run, monkeypatch, phase, prior):
    run, _ = hosted_run
    select_clamp(run, prior)
    apply = run._apply
    replacements, snapshots = [], []

    def replaced():
        if phase == "after_boundary":
            apply()
        replacement = other_thread(lambda: replacement_watch(run))
        replacements.append(replacement)
        snapshots.append(watch_snapshot(run, replacement))
        if phase == "before_boundary":
            apply()

    monkeypatch.setattr(run, "_apply", replaced)
    try:
        with pytest.raises(ValueError, match="no longer attached"):
            run._set_clamp(require_commit=True)
        assert len(replacements) == 1
        assert_unchanged(run, replacements[0], snapshots[0])
    finally:
        for replacement in replacements:
            replacement.remove()


@pytest.mark.parametrize("phase", ["boundary", "output"])
@pytest.mark.parametrize("operation", ["configure", "watch"])
@isolated_test("hosted_run", "monkeypatch")
def test_clamp_publication_pins_watch_without_control_lock(hosted_run, monkeypatch, phase, operation):
    run, _ = hosted_run
    select_clamp(run, "fresh")
    service = run.interventions
    model = TinyLM()
    attempts = []
    target = run.controls if phase == "boundary" else run.output
    name = "_publish" if phase == "boundary" else "compare"
    original = getattr(target, name)

    def published(*args, **kwargs):
        def replace():
            assert run.controls.lock.acquire(blocking=False), "publication holds Controls.lock"
            run.controls.lock.release()
            with pytest.raises(InterventionError, match="lifecycle operation is in progress"):
                if operation == "watch":
                    run.controls.interventions_on(model)
                else:
                    service.configure(dictionary=run.dictionary, tokenizer=run.tokenizer)
            attempts.append(operation)

        other_thread(replace)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, name, published)
        if phase == "boundary":
            run._set_clamp(require_commit=True)
        else:
            run._clamp_version = run.clamp._version
            run.apply_clamp()
    assert attempts == [operation]
    assert run.controls._iv_model is service and service._readers == 0
    assert run._owned_record().hook is not None
