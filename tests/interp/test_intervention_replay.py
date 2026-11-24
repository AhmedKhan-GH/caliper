"""CPU checks for snapshot replay without live lifecycle changes."""
from contextvars import Context
from copy import deepcopy
from threading import Event, Thread

import pytest
import torch

from caliper.interp._intervene import InterventionError, InterventionService, _mute_hooks
from tests.interp.test_intervene import add_spec, controlled


@pytest.mark.parametrize("spec", [
    add_spec(point="layers.0.resid_pre"),
    add_spec(),
    {"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0, "value": 10},
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "xyz"},
    {"kind": "ablate", "point": "layers.0.mlp"},
    {"kind": "ablate", "point": "layers.0.self_attn#0"},
    {"kind": "ablate", "point": "layers.0.mlp_in"},
])
def test_mute_scope_selects_objects_and_preserves_other_contexts(controlled, spec):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set(spec)
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    active = model(**inputs).logits
    assert not torch.equal(active, baseline)
    calls = hook.calls
    cache = dict(service.executor._cache)
    with _mute_hooks([hook]):
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
        assert hook.calls == calls
        torch.testing.assert_close(Context().run(model, **inputs).logits, active, rtol=0, atol=0)
        private = InterventionService(model)
        private.configure(dictionary=service.dictionary, tokenizer=tokenizer, dictionary_id="fixture")
        try:
            duplicate = private.install(spec)
            torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)
            with service.suspend():
                torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
            assert duplicate.calls == 1
        finally:
            private.close()
    assert not hook.removed and hook.calls == calls + 1
    assert all(service.executor._cache[key] is value for key, value in cache.items())
    torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)


def test_nested_mute_scope_resets_after_exception(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set(add_spec())
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    active = model(**inputs).logits
    with _mute_hooks([hook]):
        with pytest.raises(RuntimeError, match="scope exit"):
            with _mute_hooks([]):
                raise RuntimeError("scope exit")
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
    torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)


def snapshot(*specifications):
    return {"version": 1, "interventions": [
        {"specification": spec, "state": "active", "since_step": 1,
         "clear": False, "error": None, "error_code": None} for spec in specifications]}


def hook_objects(model):
    return [(tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
            for module in model.modules()]


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("point", ["layers.0.mlp", "layers.0.self_attn#0"])
def test_replay_restores_exact_live_objects_and_pending_replacement(controlled, fail, point):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    service.set({"kind": "ablate", "point": point, "until_step": 3})
    service.set({"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0,
                 "value": 10, "until_step": 3})
    ctl.step()
    traced = service.state_dict()
    traced_copy = deepcopy(traced)
    expected_trace = model(**inputs).logits
    ctl.step()  # The two trace effects end at this boundary.
    service.set(add_spec())
    service.set({"kind": "patch", "point": "layers.2.resid_pre", "source_prompt": "xyz"})
    ctl.step()
    expected_live = model(**inputs).logits
    service.set(add_spec(coefficient=2))  # The old hook still applies coefficient one.
    service.set(add_spec(point="layers.3.mlp_out", from_step=20))
    service.clear("layers.2.resid_pre")  # The patch stays active until a real boundary.
    lifecycle = service.state_dict()
    records = ctl._ivs
    originals = list(records.items())
    hooks = hook_objects(model)
    active = [(iv.hook, iv.hook.calls, iv.applied_spec) for iv in records.values() if iv.hook]
    cache = service.executor._cache
    sources = list(cache.items())
    cache_identity = service.executor._cache_identity
    step = ctl.n
    try:
        with service.temporary_state(traced) as replay:
            assert replay.model is model and replay.dictionary is service.dictionary
            assert replay.tokenizer is tokenizer and replay.executor is not service.executor
            assert all("from_step" not in spec and "until_step" not in spec
                       for spec in replay.specifications())
            assert len(replay.specifications()) == 2
            replay.specifications()[0]["point"] = "invalid"
            torch.testing.assert_close(model(**inputs).logits, expected_trace, rtol=0, atol=0)
            assert service.state_dict() == lifecycle and ctl.n == step
            if fail:
                raise RuntimeError("validation forward failed")
    except RuntimeError as error:
        assert fail and str(error) == "validation forward failed"
    assert ctl._ivs is records and list(records.items()) == originals
    assert hook_objects(model) == hooks and service.state_dict() == lifecycle
    assert traced == traced_copy and ctl.n == step
    assert service.executor._cache is cache and service.executor._cache_identity is cache_identity
    assert list(cache) == [key for key, _ in sources]
    assert all(cache[key] is value for key, value in sources)
    assert all(not hook.removed and hook.calls == calls for hook, calls, _ in active)
    assert all(iv.applied_spec is applied for iv in records.values() if iv.hook
               for hook, _, applied in active if hook is iv.hook)
    torch.testing.assert_close(model(**inputs).logits, expected_live, rtol=0, atol=0)


def test_replay_filters_inactive_records_and_keeps_queued_clear_effect(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    service.set(add_spec())
    ctl.step()
    expected = model(**inputs).logits
    state = service.state_dict()
    state["interventions"][0]["clear"] = True
    pending = snapshot(add_spec(point="layers.1.mlp_out"))["interventions"][0]
    pending.update(state="pending", since_step=None)
    ended = snapshot(add_spec(point="layers.2.mlp_out"))["interventions"][0]
    ended.update(state="ended", error="prior failure", error_code="intervene.hook_failed")
    state["interventions"].extend([pending, ended])
    with service.temporary_state(state) as replay:
        assert len(replay.specifications()) == 1
        torch.testing.assert_close(model(**inputs).logits, expected, rtol=0, atol=0)


def test_replay_patch_source_is_clean_and_cache_is_private(controlled):
    model, tokenizer, ctl, service = controlled
    clean = model(**tokenizer("abc")).logits
    service.set({"kind": "ablate", "point": "layers.0.mlp"})
    service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "xyz"})
    ctl.step()
    cached = service.executor._cache["xyz", "layers.0.resid_pre"]
    original = hook_objects(model)
    traced = snapshot(add_spec(), {"kind": "patch", "point": "layers.2.resid_pre", "source_prompt": "abc"})
    live = service.state_dict()
    forwards = []

    def inspect_lock(*args):
        assert ctl.lock.acquire(blocking=False), "source and replay forwards must release Controls.lock"
        ctl.lock.release()
        forwards.append(service.state_dict())

    observer = model.register_forward_pre_hook(inspect_lock)
    try:
        with service.temporary_state(traced) as replay:
            assert replay.executor._cache is not service.executor._cache
            torch.testing.assert_close(model(**tokenizer("xyz")).logits, clean, rtol=0, atol=0)
            assert replay.executor._cache["abc", "layers.2.resid_pre"] is not cached
    finally:
        observer.remove()
    assert forwards == [live, live]
    assert not replay.executor._cache and hook_objects(model) == original
    assert service.executor._cache["xyz", "layers.0.resid_pre"] is cached


def test_replay_allows_private_validation_clamp_and_nested_scopes(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set({"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0, "value": 10})
    ctl.step()
    steered = model(**inputs).logits
    with service.temporary_state(snapshot()) as outer:
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
        with service.temporary_state(service.state_dict()) as inner:
            assert outer.specifications() == [] and len(inner.specifications()) == 1
            torch.testing.assert_close(model(**inputs).logits, steered, rtol=0, atol=0)
        private = InterventionService(model)
        private.configure(dictionary=outer.dictionary, tokenizer=outer.tokenizer)
        try:
            private.install({"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0, "value": 10})
            torch.testing.assert_close(model(**inputs).logits, steered, rtol=0, atol=0)
        finally:
            private.close()
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
    torch.testing.assert_close(model(**inputs).logits, steered, rtol=0, atol=0)


def test_nested_replay_mutes_outer_effects_and_restores_them_after_failure(controlled):
    model, tokenizer, _, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    with service.temporary_state(snapshot(add_spec())):
        outer = model(**inputs).logits
        assert not torch.equal(outer, baseline)
        with pytest.raises(RuntimeError, match="inner validation failure"):
            with service.temporary_state(snapshot()):
                torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
                raise RuntimeError("inner validation failure")
        torch.testing.assert_close(model(**inputs).logits, outer, rtol=0, atol=0)
    torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)


class ReplayTransport:
    def broadcast(self, *args, **kwargs):
        pass

    def reply(self, *args):
        pass

    def controllers(self):
        return 1

    def may_control(self, connection):
        return True


def test_network_requests_during_replay_wait_for_real_boundary(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set(add_spec())
    service.set({"kind": "ablate", "point": "layers.1.mlp"})
    ctl.step()
    previous = model(**inputs).logits
    add = ctl._ivs["layers.0.mlp_out", "add"]
    ablate = ctl._ivs["layers.1.mlp", "ablate"]
    original_hooks = add.hook, ablate.hook
    step, records = ctl.n, ctl._ivs
    route = ctl.attach(ReplayTransport())
    ready, done = Event(), Event()
    replies, errors = [], []

    def network():
        try:
            assert ready.wait(5)
            service.set(add_spec(coefficient=2))
            for request_id, request in enumerate([
                {"clear": "layers.1.mlp"},
                {"set": {"op": "ablate", "point": "layers.2.mlp"}},
            ]):
                replies.append(route.on_control("fixture", {
                    "request_id": request_id + 1, "op": "intervene", "intervene": request}))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    def rendezvous(*args):
        ready.set()
        assert done.wait(5), "requests must complete during the replay forward"

    thread = Thread(target=network, daemon=True)
    thread.start()
    forward = None
    try:
        with service.temporary_state(snapshot()):
            forward = model.register_forward_pre_hook(rendezvous)
            torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
            forward.remove()
            assert not errors and [reply["status"] for reply in replies] == ["ok", "ok"]
            assert ctl._ivs is records and ctl.n == step
            assert (add.hook, ablate.hook) == original_hooks
            assert add.spec["coefficient"] == 2 and add.applied_spec["coefficient"] == 1
            assert add.replace and ablate.clear
            assert ctl._ivs["layers.2.mlp", "ablate"].state == "pending"
        torch.testing.assert_close(model(**inputs).logits, previous, rtol=0, atol=0)
        ctl.step()
        assert ctl.n == step + 1 and ctl._ivs is records
        assert all(hook.removed for hook in original_hooks)
        assert add.applied_spec["coefficient"] == 2 and not add.replace
        assert ("layers.1.mlp", "ablate") not in ctl._ivs
        assert ctl._ivs["layers.2.mlp", "ablate"].state == "active"
        assert not torch.equal(model(**inputs).logits, previous)
    finally:
        ready.set()
        if forward is not None:
            forward.remove()
        thread.join(5)
        route.detach()
    assert not thread.is_alive()


@pytest.mark.parametrize("mutation", [
    "version_bool", "version_float", "extra_field", "records_type", "too_many", "record_fields",
    "bad_spec", "bad_schedule", "active_step", "pending_step", "bad_clear", "bad_error", "duplicate",
])
def test_malformed_snapshot_refuses_before_any_installation(controlled, monkeypatch, mutation):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before = service.state_dict()
    hooks = hook_objects(model)
    state = snapshot({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"},
                     add_spec())
    record = state["interventions"][1]
    if mutation == "version_bool":
        state["version"] = True
    elif mutation == "version_float":
        state["version"] = 1.0
    elif mutation == "extra_field":
        state["extra"] = None
    elif mutation == "records_type":
        state["interventions"] = {}
    elif mutation == "too_many":
        from caliper._controls import MAX_INTERVENTIONS
        state["interventions"] *= MAX_INTERVENTIONS
    elif mutation == "record_fields":
        record.pop("error")
    elif mutation == "bad_spec":
        record["specification"]["point"] = "unknown"
    elif mutation == "bad_schedule":
        record["specification"]["from_step"] = True
    elif mutation == "active_step":
        record["since_step"] = None
    elif mutation == "pending_step":
        record["state"] = "pending"
    elif mutation == "bad_clear":
        record["clear"] = 1
    elif mutation == "bad_error":
        record["error"] = 42
    else:
        state["interventions"].append(deepcopy(record))

    def forbidden(*args, **kwargs):
        pytest.fail("a malformed snapshot must not install hooks or run the model")

    monkeypatch.setattr(InterventionService, "install", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(InterventionError):
        with service.temporary_state(state):
            pytest.fail("a malformed snapshot must not yield")
    assert hook_objects(model) == hooks and service.state_dict() == before


@pytest.mark.parametrize("kind", ["clamp", "patch", "ablate"])
def test_failed_replay_hook_does_not_end_original_intervention(controlled, monkeypatch, kind):
    model, tokenizer, ctl, service = controlled
    spec = {"clamp": {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 10},
            "patch": {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "xyz"},
            "ablate": {"kind": "ablate", "point": "layers.0.mlp"}}[kind]
    service.set(spec)
    ctl.step()
    state = service.state_dict()
    original = next(iter(ctl._ivs.values())).hook
    hooks = hook_objects(model)
    expected = model(**tokenizer("abc")).logits
    calls = original.calls
    with pytest.raises(InterventionError, match="snapshot hook failed") as error:
        with service.temporary_state(state) as replay:
            temporary = next(iter(replay.executor._hooks))

            def fail(*args):
                raise RuntimeError("fixture hook failure")

            monkeypatch.setattr(temporary, "_zero" if kind == "ablate" else "_apply", fail)
            model(**tokenizer("abc"))
            assert temporary.removed and replay.specifications() == []
    assert error.value.code == "intervene.hook_failed"
    assert not original.removed and original.calls == calls and original.error is None
    assert service.state_dict() == state and hook_objects(model) == hooks
    torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)


def test_source_forward_failure_cleans_partial_replay_and_resets_mute(controlled, monkeypatch):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    state = service.state_dict()
    expected = model(**tokenizer("abc")).logits
    hooks = hook_objects(model)
    forward = model.forward

    def fail(*args, **kwargs):
        raise RuntimeError("fixture source failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture source failure"):
        with service.temporary_state(snapshot(add_spec(), {
            "kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "xyz"
        })):
            pytest.fail("source failure must prevent entry")
    monkeypatch.setattr(model, "forward", forward)
    assert service.state_dict() == state and hook_objects(model) == hooks
    torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)


def test_detached_service_refuses_replay(controlled):
    _, _, _, service = controlled
    service.remove()
    with pytest.raises(InterventionError, match="no longer attached"):
        with service.temporary_state(snapshot()):
            pytest.fail("a detached service must not yield")


def test_applied_specification_list_replays_effect_before_queued_replacement(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    service.set(add_spec(until_step=4))
    ctl.step()
    expected = model(**inputs).logits
    service.set(add_spec(coefficient=5))
    traced = service.specifications()
    assert traced[0]["coefficient"] == 1
    assert service.state_dict()["interventions"][0]["specification"]["coefficient"] == 5
    ctl.step()
    current = model(**inputs).logits
    assert not torch.equal(current, expected)
    with service.temporary_state(traced) as replay:
        assert replay.specifications()[0]["coefficient"] == 1
        assert "until_step" not in replay.specifications()[0]
        torch.testing.assert_close(model(**inputs).logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(model(**inputs).logits, current, rtol=0, atol=0)
    assert traced[0]["until_step"] == 4


@pytest.mark.parametrize("specifications", [[None], [add_spec(), add_spec()], [add_spec(value=0)]])
def test_malformed_specification_list_refuses_before_installation(controlled, monkeypatch, specifications):
    _, _, _, service = controlled

    def forbidden(*args, **kwargs):
        pytest.fail("a malformed specification list must not install hooks")

    monkeypatch.setattr(InterventionService, "install", forbidden)
    with pytest.raises(InterventionError):
        with service.temporary_state(specifications):
            pytest.fail("a malformed specification list must not yield")


def test_clear_feature_preserves_unrelated_active_pending_and_ended_records(controlled):
    model, tokenizer, ctl, service = controlled
    point = "layers.0.mlp_out"
    service.set({"kind": "clamp", "point": point, "feature": 0, "value": 10})
    service.set({"kind": "scale", "point": point, "feature": 0, "multiplier": 2})
    service.set({"kind": "ablate", "point": "layers.1.mlp", "until_step": 3})
    ctl.step()
    ctl.step()
    service.set(add_spec(point="layers.2.mlp_out"))
    records = ctl._ivs
    selected = records[point, "clamp"]
    hook = selected.hook
    originals = list(records.items())
    before = service.state_dict()
    hooks = hook_objects(model)
    assert service.clear_feature(point, 0) is True
    after = deepcopy(before)
    after["interventions"][0]["clear"] = True
    assert service.state_dict() == after and hook_objects(model) == hooks
    assert ctl._ivs is records and list(records.items()) == originals
    assert not hook.removed and ctl._iv_due and ctl.busy
    assert service.clear_feature(point, 0) is False
    ctl.step()
    assert hook.removed and (point, "clamp") not in records
    assert records[point, "scale"].state == "active"
    assert records["layers.1.mlp", "ablate"].state == "ended"
    assert records["layers.2.mlp_out", "add"].state == "active"


@pytest.mark.parametrize("kind", ["clamp", "scale"])
def test_clear_feature_matches_latest_specification_and_handles_pending_records(controlled, kind):
    _, _, ctl, service = controlled
    point = "layers.0.mlp_out"
    parameter = {"value": 10} if kind == "clamp" else {"multiplier": 2}
    spec = {"kind": kind, "point": point, "feature": 0, **parameter}
    service.set(spec)
    ctl.step()
    record = ctl._ivs[point, kind]
    hook = record.hook
    service.set(dict(spec, feature=1))
    before = service.state_dict()
    assert service.clear_feature(point, 0, kind=kind) is False
    assert service.state_dict() == before and not hook.removed
    assert service.clear_feature(point, 1, kind=kind) is True
    ctl.step()
    assert hook.removed and (point, kind) not in ctl._ivs
    service.set(spec)
    assert ctl._ivs[point, kind].state == "pending"
    assert service.clear_feature(point, 0, kind=kind) is True
    ctl.step()
    assert (point, kind) not in ctl._ivs


@pytest.mark.parametrize("replacement", [False, True])
def test_clear_feature_notifies_and_preserves_requests_before_publication(controlled, monkeypatch, replacement):
    _, _, ctl, service = controlled
    point = "layers.0.mlp_out"
    spec = {"kind": "clamp", "point": point, "feature": 0, "value": 10}
    service.set(spec)
    ctl.step()
    record = ctl._ivs[point, "clamp"]
    ready, done = Event(), Event()
    errors = []

    def network():
        try:
            with ctl.cond:
                ready.set()
                assert ctl.cond.wait_for(lambda: record.clear, timeout=5)
            service.set(dict(spec, feature=1) if replacement else add_spec(point="layers.1.mlp_out"))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    publish = ctl._publish

    def inspect_publication():
        assert ctl.lock.acquire(blocking=False), "publication must release the controls lock first"
        ctl.lock.release()
        assert done.wait(5), "a notified request must finish before publication"
        publish()

    thread = Thread(target=network, daemon=True)
    thread.start()
    try:
        assert ready.wait(5)
        with monkeypatch.context() as patch:
            patch.setattr(ctl, "_publish", inspect_publication)
            assert service.clear_feature(point, 0) is True
        assert not errors and record.hook is not None and not record.hook.removed
        if replacement:
            assert record.spec["feature"] == 1 and not record.clear
        else:
            assert ctl._ivs["layers.1.mlp_out", "add"].state == "pending"
        ctl.step()
        if replacement:
            assert record.applied_spec["feature"] == 1
        else:
            assert (point, "clamp") not in ctl._ivs
            assert ctl._ivs["layers.1.mlp_out", "add"].state == "active"
    finally:
        thread.join(5)
    assert not thread.is_alive()


@pytest.mark.parametrize("fields", [
    {"point": "unknown", "feature": 0},
    {"point": "layers.0.mlp_out", "feature": True},
    {"point": "layers.0.mlp_out", "feature": -1},
    {"point": "layers.0.mlp_out", "feature": 0, "kind": "add"},
])
def test_clear_feature_refuses_invalid_selection_without_lifecycle_changes(controlled, fields):
    _, _, _, service = controlled
    before = service.state_dict()
    with pytest.raises(InterventionError):
        service.clear_feature(**fields)
    assert service.state_dict() == before


def test_clear_feature_preserves_ended_records_and_refuses_detached_service(controlled):
    _, _, ctl, service = controlled
    point = "layers.0.mlp_out"
    service.set({"kind": "clamp", "point": point, "feature": 0, "value": 10, "until_step": 3})
    ctl.step()
    ctl.step()
    before = service.state_dict()
    assert service.clear_feature(point, 0) is False
    assert service.state_dict() == before and ctl._ivs[point, "clamp"].state == "ended"
    service.remove()
    with pytest.raises(InterventionError, match="no longer attached"):
        service.clear_feature(point, 0)
