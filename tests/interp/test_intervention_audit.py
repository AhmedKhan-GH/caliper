"""Bounded CPU reproductions for intervention audit corrections."""
from copy import deepcopy

import pytest
import torch

from caliper import _wire as wire
from caliper._controls import Controls
from caliper.interp._intervene import InterventionError
from tests.interp.test_intervene import add_spec, controlled
from tests.interp.fixture import make_fixture


@pytest.mark.parametrize("spec", [
    add_spec(coefficient=2**53 + 1),
    add_spec(vector=[2**53 + 1] * 64),
    add_spec(positions=(1,)),
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "A\0B"},
    {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 2**53 + 1},
])
def test_local_requests_obey_wire_metadata_before_conversion(controlled, spec):
    _, _, ctl, service = controlled
    assert not wire._intervene_request({"set": spec})
    before = service.state_dict()
    with pytest.raises(InterventionError):
        service.set(spec)
    assert service.state_dict() == before
    wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, ctl.controls_meta()))


def test_adapter_only_ablation_point_preserves_legacy_execution():
    class Model:
        active = False

        def __call__(self):
            return 0 if self.active else 7

    class Adapter:
        def points(self, model):
            return [{"point": "adapter.output", "kind": "module", "module": "adapter.output",
                     "head": None, "type": "Fixture", "shape": None}]

        def ablate(self, model, point):
            assert point == "adapter.output"
            model.active = True

            def remove():
                model.active = False

            return remove

    model, ctl = Model(), Controls()
    service = ctl.interventions_on(model, adapter=Adapter())
    try:
        service.set({"op": "ablate", "point": "adapter.output"})
        assert model() == 7
        ctl.step()
        assert model() == 0
        saved = service.state_dict()
        service.load_state_dict(saved)
        assert model() == 7
        ctl.step()
        assert model() == 0
        service.clear("adapter.output")
        assert model() == 0
        ctl.step()
        assert model() == 7
        with pytest.raises(InterventionError, match="unknown intervention point"):
            service.set({"op": "ablate", "point": "other.output"})
    finally:
        service.remove()


def test_adapter_alias_uses_adapter_in_replay(controlled):
    from caliper._ablate import ablate
    model, tokenizer, ctl, _ = controlled

    class Adapter:
        def points(self, model):
            return [{"point": "adapter.output", "kind": "module", "module": "layers.0.mlp",
                     "head": None, "type": "Fixture", "shape": None}]

        def ablate(self, model, point):
            assert point == "adapter.output"
            return ablate(model, "layers.0.mlp")

    inputs = tokenizer("AB")
    baseline = model(**inputs).logits
    service = ctl.interventions_on(model, adapter=Adapter())
    try:
        service.set({"kind": "ablate", "point": "adapter.output"})
        ctl.step()
        active = model(**inputs).logits
        assert not torch.equal(active, baseline)
        with service.temporary_state(service.specifications()):
            torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)
        service.clear()
        ctl.step()
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
    finally:
        service.remove()


def test_retired_dictionary_records_restore_and_replay(controlled):
    model, tokenizer, ctl, service = controlled
    service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 4})
    service.set({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AB"})
    ctl.step()
    service.configure(dictionary=service.dictionary, dictionary_id="new")
    state = service.state_dict()
    baseline = model(**tokenizer("AB")).logits
    assert all(record["state"] == "ended" for record in state["interventions"])
    service.load_state_dict(state)
    assert service.state_dict() == state
    ctl.step()
    with service.temporary_state(state) as replay:
        assert not replay.specifications()
        torch.testing.assert_close(model(**tokenizer("AB")).logits, baseline, rtol=0, atol=0)
    assert service.state_dict() == state
    wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, ctl.controls_meta()))


@pytest.mark.parametrize("operation", ["restore", "replay"])
@pytest.mark.parametrize("changes", [
    {"specification": add_spec(coefficient=2**53 + 1)},
    {"specification": add_spec(positions=[32])},
    {"error": "bad\0text"}, {"error": "é" * 513}, {"error_code": "x" * 64},
])
def test_ended_records_still_require_valid_structure(controlled, operation, changes):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before = service.state_dict()
    state = deepcopy(before)
    state["interventions"][0].update(state="ended", **changes)
    with pytest.raises(InterventionError):
        if operation == "restore":
            service.load_state_dict(state)
        else:
            with service.temporary_state(state):
                pytest.fail("invalid ended record reached replay")
    assert service.state_dict() == before


@pytest.mark.parametrize("late_start", [False, True])
def test_overlap_order_matches_replay_restore_and_direct_oracle(controlled, late_start):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("AB")
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out"}
    service.set(dict(ablation, from_step=4) if late_start else ablation)
    service.set(add_spec())
    ctl.step()
    if late_start:
        ctl.step(3)
    else:
        service.set(dict(ablation, until_step=40))
        ctl.step()
    # Ablation precedes addition in the saved order. This independent hook replaces
    # their combined MLP output with the expected constant tensor.
    with service.suspend():
        handle = model.layers[0].mlp.register_forward_hook(lambda m, a, y: torch.full_like(y, 0.1))
        try:
            expected = model(**inputs).logits
        finally:
            handle.remove()
    actual = model(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with service.temporary_state(service.specifications()):
        torch.testing.assert_close(model(**inputs).logits, actual, rtol=0, atol=0)
    other, _ = make_fixture()
    fork = Controls()
    restored = fork.interventions_on(other)
    try:
        fork.n = ctl.n
        restored.load_state_dict(service.state_dict())
        fork.step()
        torch.testing.assert_close(other(**inputs).logits, actual, rtol=0, atol=0)
    finally:
        restored.remove()


def test_cancelled_replacement_preserves_hooks_output_and_cache_identity(controlled):
    import json
    model, tokenizer, ctl, service = controlled
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out"}
    service.set(ablation)
    service.set(add_spec())
    ctl.step()
    before = model(**tokenizer("AB")).logits
    hooks = [iv.hook for iv in ctl._ivs.values()]
    key = json.dumps(service.specifications(), sort_keys=True)
    service.set(dict(ablation, until_step=40))
    service.set(ablation)
    ctl.step()
    assert key == json.dumps(service.specifications(), sort_keys=True)
    torch.testing.assert_close(model(**tokenizer("AB")).logits, before, rtol=0, atol=0)
    assert hooks == [iv.hook for iv in ctl._ivs.values()]


@pytest.mark.parametrize("queued", ["replacement", "clear", "future"])
def test_script_restore_preserves_applied_output_and_queued_boundary(controlled, queued):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("AB")
    service.set({"kind": "ablate", "point": "layers.0.mlp_out"})
    service.set(add_spec())
    service.set(add_spec(point="layers.2.mlp_out", from_step=100))
    ctl.step(5)
    before = model(**inputs).logits
    if queued == "clear":
        service.clear("layers.0.mlp_out")
    else:
        service.set(add_spec(coefficient=2, **({"from_step": 100} if queued == "future" else {})))
    state = service.state_dict()
    other, _ = make_fixture()
    fork = Controls()
    restored = fork.interventions_on(other)
    fork.n = ctl.n - 1
    try:
        restored.load_state_dict(state, restore_active=True)
        assert fork.n == 4 and restored.state_dict() == state
        torch.testing.assert_close(other(**inputs).logits, before, rtol=0, atol=0)
        with service.temporary_state(state):
            torch.testing.assert_close(model(**inputs).logits, before, rtol=0, atol=0)
        ctl.step(5)
        fork.step()
        assert fork.n == ctl.n == 5
        after = model(**inputs).logits
        assert not torch.equal(before, after)
        torch.testing.assert_close(other(**inputs).logits, after, rtol=0, atol=0)
        assert restored.state_dict() == service.state_dict()
    finally:
        restored.remove()


def test_script_restore_accepts_older_checkpoint_records(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    expected = model(**tokenizer("AB")).logits
    state = service.state_dict()
    for record in state["interventions"]:
        record.pop("applied_specification", None)
    service.load_state_dict(state, restore_active=True)
    torch.testing.assert_close(model(**tokenizer("AB")).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("applied", [None, add_spec(point="layers.1.mlp_out"), add_spec(coefficient=2**53 + 1)])
def test_invalid_applied_snapshot_refuses_before_live_replacement(controlled, applied):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    before = service.state_dict()
    state = deepcopy(before)
    state["interventions"][0]["applied_specification"] = applied
    with pytest.raises(InterventionError):
        service.load_state_dict(state, restore_active=True)
    assert service.state_dict() == before


class SourceGuard:
    def __init__(self, stop=None):
        self.stop = stop
        self.stages = []

    def check(self, stage, **kwargs):
        self.stages.append(stage)
        if self.stop in ("all", stage):
            raise RuntimeError("source guard stop")


def patch_spec(**fields):
    return dict({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "AB"}, **fields)


def test_clean_source_obeys_latched_guard_before_forward(controlled):
    model, tokenizer, _, service = controlled
    service.configure(tokenizer=tokenizer, resource_guard=SourceGuard("all"))
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(RuntimeError, match="source guard stop"):
            service.executor.install(patch_spec())
        assert calls == [] and not service.executor._cache
        assert not service.executor.preparing_source
    finally:
        handle.remove()


@pytest.mark.parametrize("limit", ["bytes", "entries"])
def test_source_cache_refuses_overflow_without_evicting_active_sources(controlled, monkeypatch, limit):
    from caliper.interp import _intervene
    model, tokenizer, _, service = controlled
    executor = service.executor
    expected = model(**tokenizer("AB")).logits
    if limit == "bytes":
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES", 3 * 64 * 8)
    else:
        monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 1)
    first = executor.install(patch_spec())
    source = executor._cache["AB", "layers.0.resid_pre"]
    with pytest.raises(InterventionError, match="bounded source cache"):
        executor.install(patch_spec(point="layers.1.resid_pre"))
    assert executor._cache["AB", "layers.0.resid_pre"] is source and not first.removed
    assert sum(value.numel() * value.element_size() for value in executor._cache.values()) <= _intervene.MAX_SOURCE_BYTES
    assert len(executor._cache) <= _intervene.MAX_SOURCE_ENTRIES
    torch.testing.assert_close(model(**tokenizer("XY")).logits, expected, rtol=0, atol=0)
    first()
    assert not executor._cache
    second = executor.install(patch_spec(point="layers.1.resid_pre"))
    assert not second.removed and len(executor._cache) == 1


def test_source_cache_evicts_only_unused_sources(controlled, monkeypatch):
    from caliper.interp import _intervene
    _, _, _, service = controlled
    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 1)
    executor = service.executor
    executor._source(patch_spec())
    hook = executor.install(patch_spec(source_prompt="CD"))
    assert list(executor._cache) == [("CD", "layers.0.resid_pre")]
    with pytest.raises(InterventionError, match="bounded source cache"):
        executor._source(patch_spec())
    assert list(executor._cache) == [("CD", "layers.0.resid_pre")] and not hook.removed


def test_order_rebind_and_unrelated_removal_retain_active_patch_sources(controlled):
    model, tokenizer, ctl, service = controlled
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out"}
    service.set(ablation)
    service.set(patch_spec(point="layers.1.resid_pre"))
    service.set(patch_spec(point="layers.2.resid_pre", source_prompt="CD"))
    ctl.step()
    cached = dict(service.executor._cache)
    sources = []
    handle = model.register_forward_pre_hook(
        lambda *_: sources.append(1) if service.executor.preparing_source else None)
    try:
        service.set(dict(ablation, until_step=40))
        ctl.step()
        model(**tokenizer("XY"))
        assert sources == []
        assert all(service.executor._cache[key] is value for key, value in cached.items())
        service.clear("layers.1.resid_pre")
        ctl.step()
        assert list(service.executor._cache) == [("CD", "layers.2.resid_pre")]
        assert service.executor._cache["CD", "layers.2.resid_pre"] is cached["CD", "layers.2.resid_pre"]
    finally:
        handle.remove()


def test_clean_source_uses_internal_scope_and_decoder_options(controlled, monkeypatch):
    from caliper.interp import _intervene, _points
    from caliper.interp._output import is_internal_forward
    model, _, _, service = controlled
    probes = _points._architecture(model)
    # This fixture checks keyword dispatch only. Its probes retain the fixture map.
    monkeypatch.setattr(_points, "_architecture", lambda _: probes)
    monkeypatch.setattr(_intervene, "_architecture", lambda _: probes)
    model.config.model_type = "qwen3_5_text"
    original = model.forward
    observed = []

    def forward(*args, **kwargs):
        observed.append((is_internal_forward(), kwargs.get("use_cache"), kwargs.get("logits_to_keep")))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    service.executor.install(patch_spec())
    assert observed == [(True, False, 1)]
    assert not is_internal_forward()


def test_source_guard_stages_scope_and_replay_inheritance(controlled, monkeypatch):
    from contextlib import contextmanager
    from caliper.interp import _memory
    from caliper.interp._output import is_internal_forward
    model, tokenizer, ctl, service = controlled
    guard = SourceGuard()
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    scopes = []

    @contextmanager
    def scope(device):
        assert device.type == "cpu" and is_internal_forward()
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")

    monkeypatch.setattr(_memory, "autorelease_scope", scope)
    service.set(patch_spec())
    ctl.step()
    stages = ["patch source forward start", "patch source forward complete"]
    assert guard.stages == stages and scopes == ["enter", "exit"]
    with service.temporary_state(service.specifications()) as replay:
        assert replay.executor.resource_guard is guard
    assert guard.stages == stages * 2 and scopes == ["enter", "exit"] * 2
    guard.stop = "all"
    with pytest.raises(RuntimeError, match="source guard stop"):
        with service.temporary_state(service.specifications()):
            pytest.fail("latched guard allowed replay")
    assert not is_internal_forward() and ctl.n == 1


def test_source_guard_failure_after_forward_releases_capture(controlled):
    model, tokenizer, _, service = controlled
    guard = SourceGuard("patch source forward complete")
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    hooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match="source guard stop"):
        service.executor.install(patch_spec())
    assert guard.stages == ["patch source forward start", "patch source forward complete"]
    assert hooks == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    assert not service.executor._cache and not service.executor.preparing_source


def test_source_refill_checks_guard_and_isolates_hook_failure(controlled):
    model, tokenizer, ctl, service = controlled
    guard = SourceGuard()
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    service.set(patch_spec())
    ctl.step()
    service.invalidate_source_cache()
    guard.stop = "all"
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        model(**tokenizer("AC"))
        assert calls == [1]
        assert ctl.summary()["interventions"][0]["state"] == "ended"
        assert not service.executor._cache and not service.executor.preparing_source
    finally:
        handle.remove()


@pytest.mark.parametrize("multiple_prompts", [False, True])
def test_active_patch_sources_do_not_thrash_and_refill_once_per_prompt(controlled, multiple_prompts):
    model, tokenizer, ctl, service = controlled
    if multiple_prompts:
        specs = [patch_spec(), patch_spec(point="layers.2.resid_pre", source_prompt="CD")]
        expected = model(**tokenizer("CD")).logits
    else:
        specs = [patch_spec(point=f"layers.{layer}.{suffix}") for layer in range(3)
                 for suffix in ("resid_pre", "mlp_in", "mlp_out")]
        expected = model(**tokenizer("AB")).logits
    sources = []
    handle = model.register_forward_pre_hook(
        lambda *_: sources.append(1) if service.executor.preparing_source else None)
    try:
        for spec in specs:
            service.set(spec)
        ctl.step()
        sources.clear()
        first = model(**tokenizer("XY")).logits
        second = model(**tokenizer("XY")).logits
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, expected, rtol=0, atol=0)
        assert sources == []
        service.invalidate_source_cache()
        torch.testing.assert_close(model(**tokenizer("XY")).logits, expected, rtol=0, atol=0)
        assert len(sources) == (2 if multiple_prompts else 1)
        assert len(service.executor._cache) == len(specs)
    finally:
        handle.remove()


def test_nested_source_refill_preserves_outer_probe_capture(controlled):
    from caliper.interp import capture
    model, tokenizer, ctl, service = controlled
    service.set(patch_spec(point="layers.3.mlp_in", source_prompt="AC"))
    ctl.step()
    points = ["layers.0.mlp_in", "layers.0.mlp_out", "layers.3.mlp_in", "layers.3.mlp_out"]
    with capture(model, points) as warm:
        expected = model(**tokenizer("AB")).logits
    service.invalidate_source_cache()
    with capture(model, points) as cold:
        actual = model(**tokenizer("AB")).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for point in points:
        torch.testing.assert_close(cold[point], warm[point], rtol=0, atol=0)


def test_explicit_fork_overrides_leave_saved_requests_queued(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    service.set(add_spec(point="layers.2.mlp_out", coefficient=3))
    ctl.step(5)
    service.set(add_spec(coefficient=2))
    service.clear("layers.2.mlp_out")
    service.set(add_spec(point="layers.3.mlp_out", coefficient=4, from_step=5))
    state = service.state_dict()
    ctl.n = 4
    service.load_state_dict(state, restore_active=True)
    service.apply_overrides([add_spec(point="layers.1.mlp_out", coefficient=7)])
    assert ctl.n == 4
    current = service.state_dict()["interventions"]
    assert current[:3] == state["interventions"]

    def oracle(amounts):
        handles = [model.layers[layer].mlp.register_forward_hook(
            lambda m, a, y, amount=amount: y + y.new_tensor([0.1] * 64) * amount)
            for layer, amount in amounts]
        try:
            with service.suspend():
                return model(**tokenizer("AB")).logits
        finally:
            for handle in handles:
                handle.remove()

    expected = oracle([(0, 1), (1, 7), (2, 3)])
    torch.testing.assert_close(model(**tokenizer("AB")).logits, expected, rtol=0, atol=0)
    ctl.step()
    assert ctl.n == 5
    expected = oracle([(0, 2), (1, 7), (3, 4)])
    torch.testing.assert_close(model(**tokenizer("AB")).logits, expected, rtol=0, atol=0)


def test_explicit_override_replaces_only_its_record_and_obeys_schedule(controlled):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step(5)
    service.set(add_spec(coefficient=2))
    service.apply_overrides([add_spec(coefficient=3), add_spec(point="layers.1.mlp_out", from_step=10)])
    assert ctl.n == 5 and service.specifications()[0]["coefficient"] == 3
    assert service.state_dict()["interventions"][1]["state"] == "pending"
    ctl.step()
    assert service.specifications()[0]["coefficient"] == 3
    assert len(service.specifications()) == 1
    ctl.step(9)
    assert len(service.specifications()) == 2


def test_invalid_explicit_override_batch_preserves_live_records(controlled):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    service.set(add_spec(coefficient=2))
    state = service.state_dict()
    hook = next(iter(ctl._ivs.values())).hook
    with pytest.raises(InterventionError):
        service.apply_overrides([add_spec(point="layers.1.mlp_out"), add_spec(coefficient=2**53 + 1)])
    assert service.state_dict() == state and next(iter(ctl._ivs.values())).hook is hook
    service.apply_overrides([])
    assert service.state_dict() == state and next(iter(ctl._ivs.values())).hook is hook
