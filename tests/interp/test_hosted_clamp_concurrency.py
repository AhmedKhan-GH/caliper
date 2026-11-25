"""Unchanged clamp validation permits control traffic without stale ownership."""
from copy import deepcopy
from threading import Event

import pytest

from caliper import _wire
from caliper._controls import intervention_key
from caliper.interp._intervene import InterventionError
from .concurrency_isolation import isolated_test
from .test_controls_concurrency import Job, other_thread, source_gate
from .test_hosted import hosted_run
from .test_hosted_controls import _feature_rows, _queue_action


@pytest.mark.parametrize("change", ["read", "clear", "replace", "detach", "configure"])
@isolated_test("hosted_run", "monkeypatch")
def test_unchanged_clamp_validation_releases_lock_and_retains_resources(hosted_run, monkeypatch, change):
    run, route = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    _queue_action(route, "apply_clamp", 1)
    run.boundary()
    service = run.interventions
    owned = run.controls._ivs[intervention_key(run._owned_clamp)]
    hook = owned.hook
    before = service.state_dict()
    comparison = deepcopy(run.inference_status()["comparison"])
    entered, release = Event(), Event()
    validate = service.executor.validate

    def delayed(spec, *, resources=True):
        if resources:
            assert run.controls.lock.acquire(blocking=False), "clamp validation holds Controls.lock"
            run.controls.lock.release()
            entered.set()
            assert release.wait(5), "clamp validation exceeded its finite wait"
        return validate(spec, resources=resources)

    monkeypatch.setattr(service.executor, "validate", delayed)
    job = Job(run._set_clamp)
    queued = None
    try:
        assert entered.wait(5), "clamp validation did not reach its barrier"

        def control():
            assert service.state_dict() == before
            assert service.specifications() == [run._owned_clamp]
            assert route.on_control(None, _wire.Control(2, "resume").to_meta())["status"] == "ok"
            if change == "clear":
                service.clear()
            elif change == "replace":
                # Use the real validator for this independent admission request.
                monkeypatch.setattr(service.executor, "validate", validate)
                service.set(dict(run._owned_clamp, value=run._owned_clamp["value"] + 1))
            elif change == "detach":
                service.remove()
                assert service.executor.dictionary is not None and not hook.removed
            elif change == "configure":
                with pytest.raises(InterventionError, match="operation is in progress"):
                    service.configure()
            return None if change == "detach" else service.state_dict()

        queued = other_thread(control)
        assert not job.done.is_set()
    finally:
        release.set()
        job.join()
    assert run._comparison == comparison
    if change in ("read", "configure"):
        job.success()
        assert service.state_dict() == before and owned.hook is hook and not hook.removed
    else:
        assert isinstance(job.error, ValueError) and "state changed" in str(job.error)
        if change == "detach":
            assert hook.removed and service.executor.dictionary is None
            assert not run.controls._ivs and not run.controls._iv_retiring
        else:
            assert service.state_dict() == queued and owned.hook is hook and not hook.removed


@isolated_test("hosted_run")
def test_owned_clamp_clear_invalidates_inflight_preparation(hosted_run):
    run, _ = hosted_run
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    run._set_clamp()
    owned = run.controls._ivs[intervention_key(run._owned_clamp)]
    hook = owned.hook
    run.interventions.set({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AC"})

    def clear():
        with run.controls.lock:
            run._discard_owned_clamp()
        return run.interventions.state_dict()

    with source_gate(run.model, run._apply) as job:
        queued = other_thread(clear)
    job.success()
    assert run.interventions.state_dict() == queued
    assert owned.clear and owned.hook is hook and not hook.removed
    run._apply()
    assert hook.removed and run._owned_record() is None
    assert [spec["kind"] for spec in run.interventions.specifications()] == ["patch"]
