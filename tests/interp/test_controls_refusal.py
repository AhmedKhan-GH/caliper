"""Small CPU checks for refusal notices and concurrent lifecycle ownership."""
import threading
import warnings

import pytest
import torch

from caliper import _controls, _wire
from .fixture import make_fixture
from .test_intervention_replay import hook_objects
from .test_service import Transport, call


@pytest.fixture
def case(monkeypatch):
    model, tokenizer = make_fixture(layers=1, width=4, heads=1, latents=8, k=2)
    controls = _controls.Controls()
    timers = []
    monkeypatch.setattr(controls._timers, "at", lambda when, fn, *args: timers.append((when, fn, args)))
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    route = controls.attach(Transport())
    try:
        yield model, tokenizer, controls, service, route, timers
    finally:
        route.detach()
        service.remove()


def patch(point="layers.0.resid_pre", **fields):
    return dict(kind="patch", point=point, source_prompt="a", **fields)


def addition(**fields):
    return dict(kind="add", point="layers.0.mlp_out", vector=[1.] * 4, coefficient=1, **fields)


@pytest.mark.parametrize("scheduled", [False, True])
def test_refusal_warning_observes_restored_hooks_sources_and_request(case, monkeypatch, scheduled):
    from caliper.interp import _intervene

    model, tokenizer, controls, service, route, _ = case
    service.set(patch(**({"from_step": 3} if scheduled else {})))
    service.set(patch("layers.0.mlp_in"))
    controls.step(1)
    record, survivor = controls._ivs.values()
    hooks = hook_objects(model)
    cache = service.executor._cache
    sources, identity = dict(cache), service.executor._cache_identity
    before = model(**tokenizer("c")).logits
    old = record.hook
    monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES", 128)
    service.set(dict(patch(**({"from_step": 3} if scheduled else {})), source_prompt="bb"))
    request = record.spec
    observed, ran = [], []

    def notice(message, *args, **kwargs):
        assert controls.lock.acquire(blocking=False), "warning holds the controls lock"
        controls.lock.release()
        observed.append((str(message), hook_objects(model), dict(cache), service.executor._cache_identity,
                         model(**tokenizer("c")).logits, record.spec, record.hook, survivor.error))
        # A warning callback can submit a new request after rollback.
        service.set(addition())

    controls.action("inspect", lambda: ran.append(service.specifications()))
    route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", notice)
        controls.step(2)
    assert len(observed) == 1
    message, seen_hooks, seen_cache, seen_identity, output, seen_request, seen_hook, error = observed[0]
    assert seen_hooks == hooks
    assert service.executor._cache is cache and seen_identity is identity
    assert list(seen_cache) == list(sources)
    assert all(seen_cache[key] is value for key, value in sources.items())
    torch.testing.assert_close(output, before, rtol=0, atol=0)
    assert seen_request is request and seen_hook is old and error is None
    assert f"at {record.point!r}" in message
    assert not ran and not route.transport.replies
    controls.step()
    assert ran == [service.specifications()] and ran[0][-1]["kind"] == "add"
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]


def test_runtime_failure_notice_is_once_across_another_refusal(case, monkeypatch):
    _, _, controls, service, _, _ = case
    service.set(addition())
    service.set(patch())
    controls.step()
    failed, retained = controls._ivs.values()
    failed.hook.on_error("runtime failure")
    service.set(dict(patch(), source_prompt="bb"))

    def refuse(*args, **kwargs):
        raise RuntimeError("replacement refusal")

    with warnings.catch_warnings(record=True) as reports:
        warnings.simplefilter("always")
        with monkeypatch.context() as changes:
            changes.setattr(service.executor, "install", refuse)
            controls.step()
        controls.step()
    assert sum("runtime failure" in str(item.message) for item in reports) == 1
    assert sum("replacement refusal" in str(item.message) for item in reports) == 1
    assert not failed.report and failed.state == "ended"
    assert retained.state == "active" and retained.error_code == "intervene.hook_failed"


@pytest.mark.parametrize("edit", ["failure", "replace", "clear"])
@pytest.mark.parametrize("terminal", ["run", "expire"])
def test_scheduled_hook_ownership_conflict_defers_with_original_deadline(case, monkeypatch, edit, terminal):
    model, _, controls, service, route, timers = case
    service.set(patch(from_step=3))
    service.set(addition())
    service.set(patch("layers.0.mlp_in"))
    controls.step(1)
    scheduled, failed, survivor = controls._ivs.values()
    old, retained = failed.hook, survivor.hook
    removed_ids = {handle.id for handle in old.handles}
    retained_hooks = [tuple(tuple((key, fn) for key, fn in group if key not in removed_ids)
                            for group in module) for module in hook_objects(model)]
    cache = service.executor._cache
    sources, identity = dict(cache), service.executor._cache_identity
    observed, prepared, queued, errors = [], [], [], []
    action = controls.action("inspect", lambda: observed.append(service.specifications()))
    route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
    deadline, expire, args = next(item for item in timers if item[2] == (action, action.run_id))
    install = service.executor.install

    def conflict():
        try:
            old.on_error("concurrent hook failure")
            if edit == "replace":
                service.set(dict(addition(), coefficient=2))
            elif edit == "clear":
                service.clear(failed.point)
            queued.append((service.state_dict(), failed.spec, hook_objects(model)))
        except BaseException as error:
            errors.append(error)

    def installed(spec, **kwargs):
        result = install(spec, **kwargs)
        prepared.append(result)
        if spec["point"] == scheduled.point and not queued:
            thread = threading.Thread(target=conflict)
            thread.start()
            thread.join(5)
            assert not thread.is_alive(), "hook failure callback did not finish"
        return result

    with monkeypatch.context() as changes:
        changes.setattr(service.executor, "install", installed)
        controls.step(2)
    assert not errors and len(queued) == 1
    assert service.state_dict() == queued[0][0] and failed.spec is queued[0][1]
    assert scheduled.hook is None and old.removed
    assert survivor.hook is retained and not retained.removed
    assert all(hook.removed for hook in prepared)
    assert hook_objects(model) == retained_hooks
    assert service.executor._cache is cache and service.executor._cache_identity is identity
    assert list(cache) == list(sources) and all(cache[key] is value for key, value in sources.items())
    assert not observed and not route.transport.replies and action.queued
    assert [(when, fn, values) for when, fn, values in timers if fn == expire] == [(deadline, expire, args)]
    if terminal == "expire":
        expire(*args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        controls.step()
    assert scheduled.hook is not None and scheduled.state == "active"
    if edit == "failure":
        assert failed.state == "ended" and failed.error == "concurrent hook failure"
    elif edit == "replace":
        assert failed.applied_spec["coefficient"] == 2 and failed.spec is queued[0][1]
    else:
        assert (failed.point, failed.op) not in controls._ivs
    reply, = route.transport.replies
    assert (reply.request_id, reply.status) == (1, "ok" if terminal == "run" else "error")
    assert observed == ([service.specifications()] if terminal == "run" else [])
    if terminal == "expire":
        assert reply.error["code"] == "timeout"
    expire(*args)
    controls.step()
    assert len(route.transport.replies) == 1 and not controls.queue


def test_normal_expiration_clears_active_replacement_refusal(case, monkeypatch):
    _, _, controls, service, _, _ = case
    service.set(addition(until_step=5))
    controls.step(1)
    record = next(iter(controls._ivs.values()))
    hook = record.hook
    service.set(dict(addition(), coefficient=2))

    def refuse(*args, **kwargs):
        raise RuntimeError("replacement refusal")

    with monkeypatch.context() as changes:
        changes.setattr(service.executor, "install", refuse)
        with pytest.warns(RuntimeWarning, match="replacement refusal"):
            controls.step(2)
    assert record.state == "active" and record.error_code == "intervene.hook_failed"
    controls.step(4)
    assert record.state == "ended" and hook.removed
    assert record.error is None and record.error_code is None
    saved, = service.state_dict()["interventions"]
    assert saved["error"] is None and saved["error_code"] is None
    entry, = controls.controls_meta()["interventions"]["active"]
    assert "error" not in entry and "error_code" not in entry


@pytest.mark.parametrize("conflict", [False, True])
def test_interpretation_work_observes_restored_effect_after_boundary_deferral(case, monkeypatch, conflict):
    from caliper.interp._service import InterpretationService, register_service, unregister_service

    model, tokenizer, controls, service, route, _ = case
    service.set(addition())
    service.set(patch())
    controls.step()
    hooks = hook_objects(model)
    cache = dict(service.executor._cache)
    identity = service.executor._cache_identity
    before = model(**tokenizer("c")).logits
    observations = []

    def inspect(**kwargs):
        observations.append((hook_objects(model), dict(service.executor._cache), service.executor._cache_identity,
                             service.specifications(), model(**tokenizer("c")).logits, service._operation_owner))
        return {"observed": True}

    reader = InterpretationService(handlers={"patch.sweep": inspect})
    register_service(reader, controls=controls)
    service.set(dict(addition(), coefficient=2))
    service.set(dict(patch(), source_prompt="bb"))
    install = service.executor.install

    def installed(spec, **kwargs):
        if spec["kind"] == "patch":
            if conflict:
                service.set(dict(patch(), source_prompt="c"))
            raise RuntimeError("source refusal")
        return install(spec, **kwargs)

    try:
        assert call(route, "patch.sweep", {"clean_prompt": "a", "corrupted_prompt": "b",
                                        "metric": {"positive_token": 1, "negative_token": 2}})["status"] == "queued"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            monkeypatch.setattr(service.executor, "install", installed)
            controls.step()
        assert len(observations) == 1
        seen_hooks, sources, seen_identity, specs, output, owner = observations[0]
        assert seen_hooks == hooks and seen_identity is identity and owner is None
        assert list(sources) == list(cache) and all(sources[key] is value for key, value in cache.items())
        assert specs == [iv.applied_spec for iv in controls._ivs.values()]
        torch.testing.assert_close(output, before, rtol=0, atol=0)
        reply, = route.transport.replies
        assert reply.status == "ok" and reply.result == {"observed": True}
    finally:
        unregister_service(reader, controls=controls)
