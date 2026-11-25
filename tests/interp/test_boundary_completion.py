"""Small CPU cases for boundary completion and fatal hook errors."""
import pytest
import torch

from caliper.interp._memory import CleanStop
from tests.interp.test_intervene import add_spec, controlled


@pytest.mark.parametrize("kind", ["clamp", "scale"])
@pytest.mark.parametrize("fatal", [False, True])
def test_encoder_stop_cleans_hook_before_forward_continues(controlled, kind, fatal):
    model, tokenizer, ctl, service = controlled
    dictionary = service.dictionary
    dictionary.layers[0].float()
    error = CleanStop("encoder resource stop") if fatal else RuntimeError("encoder hook error")

    class Guard:
        limit, peak, current = 12_000_000_000, 0, 0
        device = torch.device("cpu")

        def check(self, stage):
            if stage == "dictionary encoder block":
                raise error

    service.configure(dictionary=dictionary, tokenizer=tokenizer,
                      dictionary_id="fixture", resource_guard=Guard())
    value = {"value": 0} if kind == "clamp" else {"multiplier": 0}
    service.set({"kind": kind, "point": "layers.0.mlp_out", "feature": 0, **value})
    ctl.step()
    record = ctl._ivs[("layers.0.mlp_out", kind)]
    hook = record.hook
    downstream = []
    observer = model.layers[1].register_forward_pre_hook(lambda *args: downstream.append(True))
    try:
        if fatal:
            with pytest.raises(CleanStop) as caught:
                model(**tokenizer("abc"))
            assert caught.value is error
            assert downstream == []
        else:
            output = model(**tokenizer("abc")).logits
            assert downstream == [True]
            with service.suspend():
                expected = model(**tokenizer("abc")).logits
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert hook.removed and hook.errors == 1
        assert hook.error == f"{type(error).__name__}: {error}"
        assert record.state == "ended" and record.hook is None
        assert record.error_code == "intervene.hook_failed"
        assert service.specifications() == []
    finally:
        observer.remove()


@pytest.mark.parametrize("started", [False, True])
def test_prospective_step_overflow_preserves_all_control_state(started):
    from caliper._controls import Controls
    from python.tests.test_controls_budget import control, encoded, fill_budget

    ctl = Controls()
    param = ctl.param("text", "original")
    calls = []
    action = ctl.action("queued", lambda: calls.append(True))
    if started:
        ctl.step()
        assert control(ctl, "set", name=param.name, value="pending")["status"] == "ok"
    assert control(ctl, "run", name=action.name)["status"] == "queued"
    fill_budget(ctl)
    before = ctl.controls_meta()
    pending = (ctl.n, ctl.busy, ctl._npending, param._version, list(ctl.queue))
    with pytest.raises(ValueError, match="wire limit"):
        ctl.step(10**1000)
    assert ctl.controls_meta() == before
    assert (ctl.n, ctl.busy, ctl._npending, param._version, list(ctl.queue)) == pending
    assert calls == [] and action.queued
    encoded(ctl)
    ctl.step()
    assert calls == [True]
    assert param.value == ("pending" if started else "original")
    encoded(ctl)


def test_large_step_and_implicit_increment_remain_supported():
    from caliper._controls import Controls
    from python.tests.test_controls_budget import encoded

    ctl = Controls()
    ctl.step(10**100)
    ctl.param("text", "ordinary value")
    ctl.step()
    assert ctl.n == 10**100 + 1
    encoded(ctl)


def test_scheduled_start_reserve_preserves_valid_timestamp_endpoints(controlled):
    from python.tests.test_controls_budget import encoded, fill_budget

    _, _, ctl, service = controlled
    ctl.step(1)
    service.set(add_spec(from_step=3))
    service.set(add_spec(point="layers.2.mlp_out", from_step=3))
    pad = fill_budget(ctl)
    # Keep the metadata near its cap while it reserves both valid timestamps.
    assert len(pad.value) > 25
    pad.set(pad.value[:-25])
    before, state = ctl.controls_meta(), service.state_dict()
    with pytest.raises(ValueError, match="since_step"):
        ctl.step(10**100)
    assert ctl.n == 1 and ctl.controls_meta() == before
    assert service.state_dict() == state and service.specifications() == []

    ctl.step(2**53 - 2)
    assert len(service.specifications()) == 2
    assert all(record.since_step == 2**53 - 1 for record in ctl._ivs.values())
    encoded(ctl)


def test_implicit_step_growth_is_admitted_before_counter_change():
    from caliper._controls import Controls
    from python.tests.test_controls_budget import encoded, fill_budget

    ctl = Controls()
    ctl.step(10**100 - 1)
    pad = fill_budget(ctl)
    # Escaped padding leaves at most five bytes. Consume each remaining byte.
    for _ in range(6):
        try:
            pad.set(pad.value + "x")
        except ValueError:
            break
    else:
        pytest.fail("metadata did not reach its admission bound")
    before = ctl.controls_meta()
    with pytest.raises(ValueError, match="wire limit"):
        ctl.step()
    assert ctl.n == 10**100 - 1 and ctl.controls_meta() == before
    encoded(ctl)


PATCH = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "xyz"}


def assert_unlocked(ctl):
    """Fail before a reentrant test callback can wait on the controls lock."""
    assert ctl.lock.acquire(blocking=False), "source callback holds the controls lock"
    ctl.lock.release()


@pytest.mark.parametrize("transition", ["expire", "clear", "clear_all"])
def test_due_removal_completes_before_conflicting_addition(controlled, transition):
    model, tokenizer, ctl, service = controlled
    baseline = model(**tokenizer("ABC")).logits
    service.set(add_spec(**({"until_step": 3} if transition == "expire" else {})))
    ctl.step(1)
    record = ctl._ivs[("layers.0.mlp_out", "add")]
    old = record.hook
    assert not torch.equal(model(**tokenizer("ABC")).logits, baseline)
    if transition == "clear":
        service.clear(record.point)
    elif transition == "clear_all":
        service.clear()
    service.set(PATCH)
    queued = []

    def request(*args):
        assert_unlocked(ctl)
        service.set(add_spec(point="layers.3.mlp_out"))
        queued.append(service.state_dict())

    observer = model.register_forward_pre_hook(request)
    try:
        ctl.step(2)
    finally:
        observer.remove()
    assert len(queued) == 1 and service.state_dict() == queued[0]
    assert old.removed
    if transition == "expire":
        assert record.state == "ended"
    else:
        assert (record.point, record.op) not in ctl._ivs
    assert service.specifications() == [] and service.executor._cache == {}
    assert len(service.specifications(active_only=False)) == 2
    torch.testing.assert_close(model(**tokenizer("ABC")).logits, baseline, rtol=0, atol=0)
    ctl.step()
    assert len(service.specifications()) == 2


def test_due_activation_keeps_overlap_order_after_unrelated_conflict(controlled):
    model, tokenizer, ctl, service = controlled
    service.set({"kind": "ablate", "point": "layers.0.mlp_out", "from_step": 3})
    service.set(add_spec())
    ctl.step(1)
    old = ctl._ivs[("layers.0.mlp_out", "add")].hook
    service.set(PATCH)
    queued = []

    def request(*args):
        assert_unlocked(ctl)
        service.set(add_spec(coefficient=2))
        queued.append(True)

    observer = model.register_forward_pre_hook(request)
    try:
        ctl.step(2)
    finally:
        observer.remove()
    assert queued == [True] and old.removed
    ablation = ctl._ivs[("layers.0.mlp_out", "ablate")]
    addition = ctl._ivs[("layers.0.mlp_out", "add")]
    assert ablation.state == "active" and ablation.since_step == 3
    assert addition.applied_spec["coefficient"] == 1 and addition.spec["coefficient"] == 2
    values = []
    observer = model.layers[0].mlp.register_forward_hook(lambda m, a, y: values.append(y.detach().clone()))
    try:
        model(**tokenizer("ABC"))
        value = values.pop()
        torch.testing.assert_close(value, torch.full_like(value, 0.1), rtol=0, atol=0)
        observer.remove()
        ctl.step()
        observer = model.layers[0].mlp.register_forward_hook(lambda m, a, y: values.append(y.detach().clone()))
        model(**tokenizer("ABC"))
        value = values.pop()
        torch.testing.assert_close(value, torch.full_like(value, 0.2), rtol=0, atol=0)
    finally:
        observer.remove()


@pytest.mark.parametrize("edit", ["new", "replace", "clear"])
def test_scheduled_patch_completes_once_and_preserves_source_callback_requests(controlled, edit):
    model, tokenizer, ctl, service = controlled
    scheduled = dict(PATCH, from_step=3)
    service.set(scheduled)
    ctl.step(1)
    record = ctl._ivs[(PATCH["point"], "patch")]
    calls = []

    def request(*args):
        assert_unlocked(ctl)
        calls.append(True)
        if edit == "new":
            service.set(add_spec())
        elif edit == "replace":
            service.set(dict(PATCH, source_prompt="ABC"))
        else:
            service.clear(PATCH["point"])

    observer = model.register_forward_pre_hook(request)
    try:
        ctl.step(2)
    finally:
        observer.remove()
    assert calls == [True]
    assert record.state == "active" and record.since_step == 3
    assert record.applied_spec["source_prompt"] == "xyz"
    assert len(service.specifications()) == 1
    if edit == "new":
        assert ctl._ivs[("layers.0.mlp_out", "add")].state == "pending"
    elif edit == "replace":
        assert record.replace and record.spec["source_prompt"] == "ABC"
    else:
        assert record.clear
    source = service.executor._cache[("xyz", PATCH["point"])]
    observed = []
    observer = model.layers[1].register_forward_pre_hook(lambda m, a: observed.append(a[0].detach().clone()))
    try:
        model(**tokenizer("abc"))
    finally:
        observer.remove()
    torch.testing.assert_close(observed[0], source, rtol=0, atol=0)
    ctl.step()
    if edit == "new":
        assert len(service.specifications()) == 2
    elif edit == "replace":
        assert record.applied_spec["source_prompt"] == "ABC"
    else:
        assert service.specifications() == []
        assert (record.point, record.op) not in ctl._ivs


def test_old_schedule_expires_while_future_replacement_waits(controlled):
    model, tokenizer, ctl, service = controlled
    baseline = model(**tokenizer("abc")).logits
    service.set(add_spec(until_step=3))
    ctl.step(1)
    record = ctl._ivs[("layers.0.mlp_out", "add")]
    old = record.hook
    service.set(add_spec(from_step=10, coefficient=2))
    ctl.step(2)
    assert old.removed and record.hook is None and record.state == "pending"
    assert record.spec["from_step"] == 10
    torch.testing.assert_close(model(**tokenizer("abc")).logits, baseline, rtol=0, atol=0)
    ctl.step(9)
    assert record.state == "active" and record.since_step == 10


def test_scheduled_source_stop_rolls_back_before_script_resumes(controlled):
    from tests.interp.test_intervention_replay import hook_objects

    model, _, ctl, service = controlled
    service.set(dict(PATCH, from_step=3))
    service.set(add_spec())
    ctl.step(1)
    before, hooks = service.state_dict(), hook_objects(model)
    stop = CleanStop("scheduled source stop")

    class Guard:
        def check(self, stage):
            if stage == "patch source forward start":
                raise stop

    service.executor.resource_guard = Guard()
    with pytest.raises(CleanStop) as caught:
        ctl.step(2)
    assert caught.value is stop
    assert service.state_dict() == before and hook_objects(model) == hooks
    assert not service.executor._cache and not service.executor.preparing_source
    assert service._operation_owner is None


def test_scheduled_reorder_failure_retains_old_hooks_without_retry(controlled, monkeypatch):
    from caliper import _wire
    from tests.interp.test_intervention_replay import hook_objects
    from tests.interp.test_service import Transport

    model, tokenizer, ctl, service = controlled
    monkeypatch.setattr(ctl._timers, "at", lambda *args: None)
    route = ctl.attach(Transport())
    service.set({"kind": "ablate", "point": "layers.0.mlp_out", "from_step": 3})
    service.set(add_spec())
    ctl.step(1)
    state, hooks = service.state_dict(), hook_objects(model)
    expected = model(**tokenizer("abc")).logits
    scheduled, retained = ctl._ivs.values()
    old = retained.hook
    calls, observed = [], []
    ctl.action("inspect", lambda: observed.append(model(**tokenizer("abc")).logits))
    route.on_control(None, _wire.Control(1, "run", name="inspect").to_meta())

    install = service.executor.install
    prepared = []

    def fail(spec, **kwargs):
        # Refuse the retained add reinstall after the scheduled ablation installs.
        if spec["op"] == "ablate":
            hook = install(spec, **kwargs)
            prepared.append(hook)
            return hook
        assert spec["op"] == "add"
        calls.append(True)
        raise RuntimeError("reorder installation refused")

    try:
        monkeypatch.setattr(service.executor, "install", fail)
        with pytest.warns(RuntimeWarning, match="reorder installation refused"):
            ctl.step(2)
        assert calls == [True] and not observed and not route.transport.replies
        assert len(prepared) == 1 and prepared[0].removed
        assert scheduled.state == "ended" and scheduled.hook is None
        assert scheduled.applied_spec is None and not scheduled.replace
        assert scheduled.error_code == "intervene.hook_failed"
        assert scheduled.error == "RuntimeError: reorder installation refused"
        assert service.state_dict()["interventions"][1] == state["interventions"][1]
        assert retained.hook is old and not old.removed and hook_objects(model) == hooks
        assert service._operation_owner is None and not service.executor._retiring_sources
        resolved = service.state_dict()
        ctl.step()
        assert calls == [True] and service.state_dict() == resolved
        assert len(observed) == 1
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
        assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok")]
        ctl.step()
        assert len(route.transport.replies) == 1 and not ctl.queue
    finally:
        route.detach()


@pytest.mark.parametrize("point", ["layers.0.resid_pre", "layers.0.mlp_out"])
def test_patch_refresh_stop_escapes_input_and_output_hooks(controlled, point):
    model, tokenizer, ctl, service = controlled
    service.set(dict(PATCH, point=point))
    ctl.step()
    record = ctl._ivs[(point, "patch")]
    hook = record.hook
    service.invalidate_source_cache()
    stop = CleanStop("patch refresh stop")

    class Guard:
        def check(self, stage):
            if stage == "patch source forward start":
                raise stop

    service.executor.resource_guard = Guard()
    downstream = []
    observer = model.layers[1].register_forward_pre_hook(lambda *args: downstream.append(True))
    try:
        with pytest.raises(CleanStop) as caught:
            model(**tokenizer("abc"))
        assert caught.value is stop and downstream == []
        assert hook.removed and hook.errors == 1
        assert record.state == "ended" and record.hook is None
        assert not service.executor._cache and not service.executor.preparing_source
    finally:
        observer.remove()
