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
        assert old.removed
        service.set(add_spec(point="layers.3.mlp_out"))
        queued.append(service.state_dict())

    observer = model.register_forward_pre_hook(request)
    try:
        ctl.step(2)
    finally:
        observer.remove()
    assert len(queued) == 1 and service.state_dict() == queued[0]
    assert record.state == "ended" if transition == "expire" else (record.point, record.op) not in ctl._ivs
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
    from caliper.interp._intervene import InterventionError
    from tests.interp.test_intervention_replay import hook_objects

    model, _, ctl, service = controlled
    service.set({"kind": "ablate", "point": "layers.0.mlp_out", "from_step": 3})
    service.set(add_spec())
    ctl.step(1)
    state, hooks = service.state_dict(), hook_objects(model)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("reorder installation refused")

    monkeypatch.setattr(service.executor, "install", fail)
    with pytest.raises(InterventionError, match="scheduled intervention preparation failed"):
        ctl.step(2)
    assert calls == [True]
    assert service.state_dict() == state and hook_objects(model) == hooks
    assert service._operation_owner is None and not service.executor._retiring_sources
