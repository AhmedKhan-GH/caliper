"""Trace caches describe effective replay hooks without lifecycle changes."""
from copy import deepcopy

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._graph import Graph
from caliper.interp._intervene import InterventionError, _ACTIVE_REPLAYS, _MUTED_HOOKS
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@pytest.fixture
def replay_trace(monkeypatch):
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2).cpu()
    tokenizer = TinyTokenizer()
    dictionary = exact_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    published = []
    monkeypatch.setattr(Graph, "publish", lambda graph: published.append(graph) or graph)
    tracer = Tracer(model, tokenizer, dictionary, controls=owner)
    try:
        yield tracer, owner, service, published
    finally:
        service.remove()


def assert_graph_equal(actual, expected):
    assert actual.nodes == expected.nodes and actual.edges == expected.edges
    assert actual.completeness == expected.completeness


@pytest.mark.parametrize("kind", ["add", "clamp", "ablate"])
@pytest.mark.parametrize("warm", [True, False])
def test_empty_and_nested_replay_use_effective_specs_and_restore_live_cache(replay_trace, kind, warm):
    tracer, owner, service, _ = replay_trace
    baseline = tracer.trace("AB", max_targets=12)
    spec = {"kind": kind, "point": "layers.0.mlp_out"}
    if kind == "add":
        spec.update(vector=[.2] * 8, coefficient=1.)
    elif kind == "clamp":
        spec.update(feature=0, value=-.7)
    service.set(spec)
    owner.step()
    live = tracer.trace("AB", max_targets=12)
    assert live.trace_id != baseline.trace_id
    saved = deepcopy(service.state_dict())
    live_hooks = [iv.hook for iv in owner._ivs.values()]
    calls = [hook.calls for hook in live_hooks]
    if not warm:
        tracer._cache.clear()
        tracer._forward_cache = None
    with service.temporary_state([]):
        clean = tracer.trace("AB", max_targets=12)
        assert clean.trace_id == baseline.trace_id and clean is not live
        assert_graph_equal(clean, baseline)
        assert service.state_dict() == saved
        changed = [{"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1,
                    "positions": [1], "value": -.9}]
        with service.temporary_state(changed) as replay:
            different = tracer.trace("AB", max_targets=12)
            assert different.trace_id not in (baseline.trace_id, live.trace_id)
            assert different.completeness["max_absolute_residual"] < 1e-9
            assert different.completeness["forward_residual"] < 1e-9
            assert any(abs(row["biases"]["intervention"]) > 1e-9
                       for row in different.completeness["targets"])
            path = tracer._forward_cache[1]
            assert path.feature_outputs[0].constants.count_nonzero() == 1
            with torch.no_grad():
                logits = tracer.model(tracer.tokenizer("AB")["input_ids"]).logits[:, -1]
            torch.testing.assert_close(path.logits, logits, atol=1e-9, rtol=0)
            with service.temporary_state([]):
                assert tracer.trace("AB", max_targets=12) is clean
            with service.temporary_state(changed):
                assert tracer.trace("AB", max_targets=12) is different
            # Replay object identity must not enter the cache key.
            assert tracer.trace("AB", max_targets=12) is different
            with service.rows([0]):
                selected = tracer.trace("AB", max_targets=12)
                assert_graph_equal(selected, different)
                with service.rows([]):
                    excluded = tracer.trace("AB", max_targets=12)
                    assert_graph_equal(excluded, baseline)
            with service.suspend():
                assert tracer.trace("AB", max_targets=12) is excluded
            assert tracer.trace("AB", max_targets=12) is different
            assert replay.specifications() == tracer._intervention_state()["specifications"]
        assert tracer.trace("AB", max_targets=12) is clean
        assert [hook.calls for hook in live_hooks] == calls
        assert service.state_dict() == saved
    assert service.state_dict() == saved
    assert not _ACTIVE_REPLAYS.get() and not _MUTED_HOOKS.get()
    restored = tracer.trace("AB", max_targets=12)
    assert restored.trace_id == live.trace_id
    assert_graph_equal(restored, live)
    if warm:
        assert restored is live
    assert all(not hook.removed and hook.error is None for hook in live_hooks)


@pytest.mark.parametrize("failure", ["body", "hook", "position"])
def test_replay_errors_preserve_lifecycle_and_restore_live_state(replay_trace, failure, monkeypatch):
    tracer, owner, service, published = replay_trace
    service.set({"kind": "add", "point": "layers.0.mlp_out", "vector": [.2] * 8, "coefficient": 1.})
    owner.step()
    live = tracer.trace("AB", max_targets=4)
    saved = deepcopy(service.state_dict())
    cached = list(tracer._cache.items())
    spec = {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 1.}
    if failure == "position":
        spec["positions"] = [8]
    published.clear()
    error = RuntimeError if failure == "body" else (ValueError if failure == "position" else InterventionError)
    with pytest.raises(error):
        with service.temporary_state([spec]) as replay:
            if failure == "body":
                raise RuntimeError("scope body failed")
            if failure == "hook":
                monkeypatch.setattr(replay._effects[0][1], "_apply",
                                    lambda *args: (_ for _ in ()).throw(ValueError("fixture hook failure")))
                with pytest.raises(RuntimeError, match="intervention state changed"):
                    tracer.trace("AB", max_targets=4)
            else:
                tracer.trace("AB", max_targets=4)
    assert published == []
    assert list(tracer._cache.items()) == cached
    assert service.state_dict() == saved
    assert not _ACTIVE_REPLAYS.get() and not _MUTED_HOOKS.get()
    assert tracer.trace("AB", max_targets=4) is live


def test_muted_live_hook_does_not_enter_trace_identity(replay_trace):
    from caliper.interp._intervene import _mute_hooks
    tracer, owner, service, _ = replay_trace
    baseline = tracer.trace("AB", max_targets=4)
    service.set({"kind": "add", "point": "layers.0.mlp_out", "vector": [.2] * 8, "coefficient": 1.})
    owner.step()
    live = tracer.trace("AB", max_targets=4)
    with _mute_hooks([iv.hook for iv in owner._ivs.values()]):
        assert tracer.trace("AB", max_targets=4) is baseline
    assert tracer.trace("AB", max_targets=4) is live


def test_lifecycle_replay_uses_applied_specs_without_consuming_requests(replay_trace):
    tracer, owner, service, _ = replay_trace
    spec = {"kind": "add", "point": "layers.0.mlp_out", "vector": [.2] * 8, "coefficient": 1.}
    service.set(spec)
    owner.step()
    live = tracer.trace("AB", max_targets=4)
    service.set(dict(spec, coefficient=3.))
    snapshot = service.state_dict()
    with service.temporary_state(snapshot):
        replayed = tracer.trace("AB", max_targets=4)
        assert_graph_equal(replayed, live)
        assert service.state_dict() == snapshot
        service.clear()
        queued = service.state_dict()
        assert tracer.trace("AB", max_targets=4) is replayed
        assert service.state_dict() == queued
    assert tracer.trace("AB", max_targets=4) is live
    assert service.state_dict() == queued


def test_replay_prompt_change_preserves_live_patch_cache(replay_trace):
    tracer, owner, service, _ = replay_trace
    service.set({"kind": "patch", "point": "layers.0.mlp_out", "positions": [1], "source_prompt": "CD"})
    owner.step()
    tracer.trace("AB", max_targets=4)
    cache = dict(service.executor._cache)
    assert cache
    state = service.state_dict()
    with service.temporary_state([]):
        tracer.trace("EF", max_targets=4)
        assert service.executor._cache == cache
    assert service.state_dict() == state
    assert service.executor._cache == cache


@pytest.mark.parametrize("rows", [[1], [0, 1]])
def test_replay_rows_outside_single_prompt_refuse_before_forward(replay_trace, rows, monkeypatch):
    tracer, _, service, published = replay_trace
    spec = {"kind": "add", "point": "layers.0.mlp_out", "vector": [.2] * 8, "coefficient": 1.}
    with service.temporary_state([spec]) as replay, service.rows(rows):
        hook = replay._effects[0][1]
        monkeypatch.setattr(tracer.model, "forward", lambda *a, **kw: pytest.fail("unexpected forward"))
        with pytest.raises(ValueError, match="rows.*single prompt"):
            tracer.trace("AB", max_targets=4)
        assert hook.calls == 0 and hook.error is None and not hook.removed
        assert not tracer._cache and tracer._forward_cache is None and tracer._prompt is None
        assert published == []


def test_muted_replay_hook_and_unrelated_replay_preserve_effective_state(replay_trace):
    from caliper.interp._intervene import _mute_hooks
    tracer, _, service, _ = replay_trace
    baseline = tracer.trace("AB", max_targets=4)
    spec = {"kind": "add", "point": "layers.0.mlp_out", "vector": [.2] * 8, "coefficient": 1.}
    other_owner = Controls()
    other = other_owner.interventions_on(TinyLM(layers=1, width=8, heads=2, latents=8, k=2).cpu())
    try:
        with service.temporary_state([spec]) as replay:
            changed = tracer.trace("AB", max_targets=4)
            with other.temporary_state([]):
                assert tracer.trace("AB", max_targets=4) is changed
            with _mute_hooks([hook for _, hook in replay._effects]):
                assert tracer.trace("AB", max_targets=4) is baseline
            assert tracer.trace("AB", max_targets=4) is changed
    finally:
        other.remove()
