"""Bounded CPU checks for action admission after intervention refusals."""
from copy import deepcopy
import warnings

import pytest
import torch

from caliper import _wire
from .test_intervene import add_spec, controlled
from .test_intervention_replay import hook_objects
from .test_service import Transport


def assert_callback_unlocked(controls):
    assert controls.lock.acquire(blocking=False), "install callback holds the controls lock"
    controls.lock.release()


def test_install_callback_lock_guard_fails_without_a_wait():
    from caliper._controls import Controls

    controls = Controls()
    with controls.lock, pytest.raises(AssertionError, match="install callback holds the controls lock"):
        assert_callback_unlocked(controls)
    assert_callback_unlocked(controls)


def test_resource_stop_does_not_retain_a_speculative_source_in_attribution(monkeypatch):
    import weakref
    from caliper.interp._memory import CleanStop
    from caliper.interp._service import InterpretationService, register_service, unregister_service
    from .test_source_admission import patch_capacity

    with patch_capacity(2) as (model, _, controls, service, specs):
        interpretation = register_service(InterpretationService(model=model), controls=controls)
        route = controls.attach(Transport())
        cache = service.executor._cache
        sources, identity, hooks = list(cache.items()), service.executor._cache_identity, hook_objects(model)
        original = [(iv, iv.hook, iv.applied_spec) for iv in controls._ivs.values()]
        service.set(dict(specs[0], source_prompt="b"))
        controls.action("recover", lambda: pytest.fail("resource stop ran an action"))
        route.on_control(None, _wire.Control(1, "run", name="recover").to_meta())
        install, prepared = service.executor.install, []
        failure = CleanStop("source registration resource stop")

        def installed(spec, **kwargs):
            if spec["point"] == specs[1]["point"]:
                raise failure
            hook = install(spec, **kwargs)
            prepared.append(weakref.ref(cache[spec["source_prompt"], spec["point"]]))
            return hook

        try:
            monkeypatch.setattr(service.executor, "install", installed)
            with pytest.raises(CleanStop) as caught:
                controls.step()
            assert caught.value is failure and interpretation.failure is failure
            assert len(prepared) == 1 and prepared[0]() is None
            assert all(iv.hook is hook and iv.applied_spec is applied and not hook.removed
                       for iv, hook, applied in original)
            assert original[0][0].replace and all(iv.error is None for iv, _, _ in original)
            assert hook_objects(model) == hooks
            assert service.executor._cache is cache and service.executor._cache_identity is identity
            assert list(cache) == [key for key, _ in sources]
            assert all(cache[key] is tensor for key, tensor in sources)
            reply, = route.transport.replies
            assert reply.status == "error" and reply.error["code"] == "action_failed"
            with pytest.raises(CleanStop) as repeated:
                controls.step()
            assert repeated.value is failure and len(route.transport.replies) == 1
        finally:
            route.detach()
            unregister_service(interpretation, controls=controls)


def test_source_capacity_refusal_preserves_an_independent_add_request(monkeypatch):
    from caliper._controls import Controls
    from caliper.interp import _intervene
    from .fixture import make_fixture

    model, tokenizer = make_fixture(layers=1, width=4, heads=1, latents=8, k=2)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    first = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "a"}
    add = {"kind": "add", "point": "layers.0.mlp_out", "vector": [1.] * 4, "coefficient": 1}
    last = {"kind": "patch", "point": "layers.0.mlp_in", "source_prompt": "a"}
    route = controls.attach(Transport())
    observed = []
    try:
        for spec in (first, add, last):
            service.set(spec)
        controls.step()
        original = [(iv, iv.hook, iv.applied_spec) for iv in controls._ivs.values()]
        hooks = hook_objects(model)
        cache = service.executor._cache
        sources, identity = list(cache.items()), service.executor._cache_identity
        assert [tensor.numel() * tensor.element_size() for _, tensor in sources] == [64, 64]
        before = model(**tokenizer("c")).logits
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES", 128)
        service.set(dict(first, source_prompt="bb"))
        service.set(dict(add, coefficient=2))
        request = original[1][0].spec
        controls.action("inspect", lambda: observed.append(service.specifications()))
        route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
        with pytest.warns(RuntimeWarning, match="bounded source cache"):
            controls.step()
        patch_record, add_record, last_record = [iv for iv, *_ in original]
        assert patch_record.error == "InterventionError: active patch sources exceed the bounded source cache"
        assert patch_record.error_code == "intervene.hook_failed" and not patch_record.replace
        assert patch_record.spec == patch_record.applied_spec
        assert add_record.spec is request and add_record.spec["coefficient"] == 2
        assert add_record.replace and add_record.error is None and last_record.error is None
        assert all(iv.hook is hook and iv.applied_spec is applied and not hook.removed
                   for iv, hook, applied in original)
        assert hook_objects(model) == hooks
        assert service.executor._cache is cache and service.executor._cache_identity is identity
        assert list(cache) == [key for key, _ in sources]
        assert all(cache[key] is tensor for key, tensor in sources)
        torch.testing.assert_close(model(**tokenizer("c")).logits, before, rtol=0, atol=0)
        assert not observed and not route.transport.replies
        controls.step()
        assert observed == [service.specifications()]
        assert observed[0][0]["source_prompt"] == "a" and observed[0][1]["coefficient"] == 2
        assert not add_record.replace and add_record.error is None
        assert patch_record.hook is original[0][1]
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]
        controls.step()
        assert len(route.transport.replies) == 1 and not controls.queue
    finally:
        route.detach()
        service.remove()


@pytest.mark.parametrize("capacity", ["bytes", "entries"])
def test_deferred_restore_refusal_ends_saved_effect_without_a_hook(monkeypatch, capacity):
    from caliper._controls import Controls
    from caliper.interp import _intervene
    from .fixture import make_fixture

    model, tokenizer = make_fixture(layers=1, width=4, heads=1, latents=8, k=2)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    route = controls.attach(Transport())
    observed, due = [], []
    try:
        service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "bb"})
        service.set({"kind": "patch", "point": "layers.0.mlp_in", "source_prompt": "a", "from_step": 1})
        controls.step(2)
        saved = service.state_dict()
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES" if capacity == "bytes" else "MAX_SOURCE_ENTRIES",
                            128 if capacity == "bytes" else 1)
        service.load_state_dict(saved, restore_active=False)
        first, second = controls._ivs.values()
        assert all(iv.hook is None and iv.applied_spec is not None for iv in (first, second))
        install = service.executor.install

        def installed(spec, **kwargs):
            # Capture the committed due effect before speculative source preparation.
            if spec["point"] == first.point:
                due.append((second.hook, hook_objects(model), list(service.executor._cache.items()),
                            service.executor._cache_identity))
            return install(spec, **kwargs)

        monkeypatch.setattr(service.executor, "install", installed)
        cache = service.executor._cache
        controls.action("recover", lambda: observed.append(service.specifications()))
        route.on_control(None, _wire.Control(1, "run", name="recover").to_meta())
        with pytest.warns(RuntimeWarning, match="bounded source cache"):
            controls.step()
        assert first.state == "ended" and first.hook is None and first.applied_spec is None
        assert not first.replace and first.error_code == "intervene.hook_failed"
        assert first.error == "InterventionError: active patch sources exceed the bounded source cache"
        hook, hooks, sources, identity = due[0]
        assert second.hook is hook and not hook.removed and second.error is None
        assert hook_objects(model) == hooks
        assert service.executor._cache is cache and service.executor._cache_identity is identity
        assert list(cache) == [key for key, _ in sources]
        assert all(cache[key] is tensor for key, tensor in sources)
        entry = controls.controls_meta()["interventions"]["active"][0]
        assert entry["state"] == "ended" and entry["error"] == first.error
        assert not observed and not route.transport.replies
        resolved = service.state_dict()
        controls.step()
        assert len(due) == 1 and service.state_dict() == resolved
        assert observed == [[second.applied_spec]]
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]
        controls.step()
        assert len(route.transport.replies) == 1 and not controls.queue
        assert second.hook is hook and service.executor._cache_identity is identity
    finally:
        route.detach()
        service.remove()


@pytest.mark.parametrize("request_kind", ["replacement", "activation", "scheduled"])
@pytest.mark.parametrize("request_index", [0, 1])
def test_reinstallation_refusal_resolves_the_request_that_changes_the_final_set(monkeypatch, request_kind, request_index):
    from caliper.interp import _intervene
    from .test_source_admission import patch_capacity

    with patch_capacity(2) as (model, tokenizer, controls, service, specs):
        monkeypatch.setattr(controls._timers, "at", lambda *args: None)
        cache = service.executor._cache
        sources, identity = list(cache.items()), service.executor._cache_identity
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES",
                            sum(t.numel() * t.element_size() for _, t in sources))
        hooks = hook_objects(model)
        before = model(**tokenizer("c")).logits
        first = controls._ivs[(specs[request_index]["point"], "patch")]
        if request_kind != "replacement":
            saved = service.state_dict()
            pending = saved["interventions"][request_index]
            pending.update(state="pending", applied_specification=None, since_step=None)
            pending["specification"]["source_prompt"] = "bb"
            if request_kind == "scheduled":
                pending["specification"]["from_step"] = controls.n + 2
            service.load_state_dict(saved, restore_active=True)
            first = controls._ivs[(specs[request_index]["point"], "patch")]
            sources, identity = list(cache.items()), service.executor._cache_identity
            hooks = hook_objects(model)
            before = model(**tokenizer("c")).logits
        else:
            service.set(dict(specs[request_index], source_prompt="bb"))
        old_hook, applied = first.hook, first.applied_spec
        transport, observed = Transport(), []
        route = controls.attach(transport)
        controls.action("inspect", lambda: observed.append(model(**tokenizer("c")).logits))
        route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
        try:
            with pytest.warns(RuntimeWarning, match="bounded source cache") as reports:
                controls.step()
            assert len(reports) == 1 and f"at {first.point!r}" in str(reports[0].message)
            assert first.error_code == "intervene.hook_failed"
            assert not first.replace
            assert first.state == ("active" if request_kind == "replacement" else "ended")
            assert first.hook is old_hook and first.applied_spec is applied
            assert hook_objects(model) == hooks
            deferred = request_index == 0 or request_kind == "replacement"
            assert service.executor._cache is cache
            if deferred:
                assert service.executor._cache_identity is identity
            else:
                # Commit cleanup can rebuild the immutable source identity tuple.
                assert service.executor._cache_identity == identity
            assert list(cache) == [key for key, _ in sources]
            assert all(cache[key] is tensor for key, tensor in sources)
            # A failed new tail hook needs no rollback of earlier live effects.
            assert len(transport.replies) == len(observed) == (0 if deferred else 1)
            controls.step()
            assert [(reply.request_id, reply.status) for reply in transport.replies] == [(1, "ok")]
            torch.testing.assert_close(observed[0], before, rtol=0, atol=0)
            assert not controls.queue
        finally:
            route.detach()


def test_terminal_replacement_refusal_preserves_effect_and_allows_recovery(controlled):
    model, tokenizer, controls, service = controlled
    transport = Transport()
    route = controls.attach(transport)
    old = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "abc"}
    service.set(old)
    controls.step()
    record = controls._ivs[(old["point"], "patch")]
    specification, hook = record.applied_spec, record.hook
    cache, identity = dict(service.executor._cache), service.executor._cache_identity
    expected = model(**tokenizer("ABC")).logits
    attempts, observed = [], []

    def fail(*args):
        if service.executor.preparing_source:
            attempts.append(1)
            raise RuntimeError("persistent source refusal")

    def recover():
        observed.append(model(**tokenizer("ABC")).logits)
        service.clear(old["point"])

    controls.action("recover", recover)
    service.set(dict(old, source_prompt="xyz"))
    assert route.on_control(None, _wire.Control(1, "run", name="recover").to_meta())["status"] == "queued"
    observer = model.register_forward_pre_hook(fail)
    hooks = hook_objects(model)
    try:
        with pytest.warns(RuntimeWarning, match="persistent source refusal"):
            controls.step()
        assert record.hook is hook and not hook.removed
        assert hook_objects(model) == hooks
        assert record.applied_spec is specification
        assert record.spec == specification and not record.replace
        assert record.state == "active" and record.error_code == "intervene.hook_failed"
        assert record.error == "RuntimeError: persistent source refusal"
        entry, = controls.controls_meta()["interventions"]["active"]
        assert entry["error"] == record.error and entry["source_prompt"] == "abc"
        _wire.decode(_wire.encode_frame(_wire.MsgType.CONTROLS, controls.controls_meta()))
        assert service.executor._cache_identity is identity
        assert list(service.executor._cache) == list(cache)
        assert all(service.executor._cache[key] is value for key, value in cache.items())
        assert not transport.replies
        resolved = deepcopy(service.state_dict())
        controls.step()
        assert attempts == [1]
        assert [(reply.request_id, reply.status) for reply in transport.replies] == [(1, "ok")]
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
        assert record.clear and not controls.queue
        controls.step()
        assert hook.removed and not service.specifications() and not service.executor._cache
        assert len(transport.replies) == 1
        assert resolved["interventions"][0]["specification"] == specification
    finally:
        observer.remove()
        route.detach()
    # The source observer was the only additional model hook during refusal.
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


def test_scheduled_reinstallation_conflict_preserves_the_newer_request(monkeypatch):
    from caliper.interp import _intervene
    from .test_source_admission import patch_capacity

    with patch_capacity(2) as (model, tokenizer, controls, service, specs):
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES", 128)
        saved = service.state_dict()
        pending = saved["interventions"][0]
        pending.update(state="pending", applied_specification=None, since_step=None)
        pending["specification"].update(source_prompt="bb", from_step=controls.n + 2)
        service.load_state_dict(saved, restore_active=True)
        hooks = hook_objects(model)
        cache, identity = list(service.executor._cache.items()), service.executor._cache_identity
        install, queued = service.executor.install, []

        def installed(spec, **kwargs):
            assert_callback_unlocked(controls)
            hook = install(spec, **kwargs)
            if spec["point"] == specs[0]["point"] and not queued:
                service.set(dict(specs[0], source_prompt="b"))
                queued.append(service.state_dict())
            return hook

        with monkeypatch.context() as patch:
            patch.setattr(service.executor, "install", installed)
            with warnings.catch_warnings(record=True) as reports:
                warnings.simplefilter("always")
                controls.step()
        assert len(queued) == 1 and not reports
        assert service.state_dict() == queued[0]
        assert hook_objects(model) == hooks
        assert service.executor._cache_identity is identity
        assert list(service.executor._cache) == [key for key, _ in cache]
        assert all(service.executor._cache[key] is tensor for key, tensor in cache)
        controls.step()
        assert [spec["source_prompt"] for spec in service.specifications()] == ["b", "a"]
        assert all(record.error is None for record in controls._ivs.values())


def test_refusal_restores_all_hooks_and_preserves_other_pending_requests(controlled):
    model, tokenizer, controls, service = controlled
    route = controls.attach(Transport())
    patch = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "abc"}
    service.set(add_spec())
    service.set(patch)
    controls.step()
    original = [(iv, iv.hook, iv.applied_spec) for iv in controls._ivs.values()]
    cache, identity = dict(service.executor._cache), service.executor._cache_identity
    expected = model(**tokenizer("ABC")).logits
    service.set(add_spec(coefficient=2))
    service.set(dict(patch, source_prompt="xyz"))
    attempts, ran = [], []

    def fail(*args):
        if service.executor.preparing_source:
            attempts.append(1)
            raise RuntimeError("replacement source refusal")

    controls.action("inspect", lambda: ran.append(service.specifications()))
    route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
    observer = model.register_forward_pre_hook(fail)
    hooks = hook_objects(model)
    try:
        with pytest.warns(RuntimeWarning, match="replacement source refusal"):
            controls.step()
        assert hook_objects(model) == hooks
        assert all(iv.hook is hook and iv.applied_spec is spec and not hook.removed
                   for iv, hook, spec in original)
        assert service.executor._cache_identity is identity
        assert list(service.executor._cache) == list(cache)
        assert all(service.executor._cache[key] is value for key, value in cache.items())
        torch.testing.assert_close(model(**tokenizer("ABC")).logits, expected, rtol=0, atol=0)
        add, patch_record = [iv for iv, _, _ in original]
        assert add.replace and add.spec["coefficient"] == 2 and add.error is None
        assert not patch_record.replace and patch_record.error_code == "intervene.hook_failed"
        assert not ran
        controls.step()
        assert attempts == [1] and ran == [service.specifications()]
        assert ran[0][0]["coefficient"] == 2 and ran[0][1]["source_prompt"] == "abc"
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]
    finally:
        observer.remove()
        route.detach()


@pytest.mark.parametrize("edit", ["replace", "clear", "unrelated"])
def test_revision_conflict_does_not_resolve_a_newer_request_as_a_refusal(controlled, edit):
    model, _, controls, service = controlled
    route = controls.attach(Transport())
    patch = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "abc"}
    service.set(patch)
    controls.step()
    record = controls._ivs[(patch["point"], "patch")]
    old_hook = record.hook
    service.set(dict(patch, source_prompt="xyz"))
    attempts, ran = [], []

    def conflict(*args):
        if service.executor.preparing_source and not attempts:
            attempts.append(1)
            if edit == "replace":
                service.set(dict(patch, source_prompt="def"))
            elif edit == "clear":
                service.clear(patch["point"])
            else:
                service.set(add_spec())
            raise RuntimeError("obsolete source refusal")

    controls.action("inspect", lambda: ran.append(service.specifications()))
    route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())
    observer = model.register_forward_pre_hook(conflict)
    try:
        with warnings.catch_warnings(record=True) as reports:
            warnings.simplefilter("always")
            controls.step()
        assert not reports and not ran and not route.transport.replies
        assert record.hook is old_hook and not old_hook.removed
        assert record.error is None and record.error_code is None and record.replace
        assert record.spec["source_prompt"] == ("def" if edit == "replace" else "xyz")
        assert record.clear == (edit == "clear")
        controls.step()
        assert attempts == [1] and ran == [service.specifications()]
        if edit == "clear":
            assert not ran[0]
        else:
            assert ran[0][0]["source_prompt"] == ("def" if edit == "replace" else "xyz")
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]
    finally:
        observer.remove()
        route.detach()
