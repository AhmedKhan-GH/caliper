"""CPU regressions for hosted latency without weaker content or memory checks."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._hosted import _RunGuard
from caliper.interp._intervene import _content_identity
from caliper.interp._memory import CleanStop, MemoryGuard
from caliper.interp import _dictionary
from .test_hosted import hosted_run


def guarded_memory():
    samples = []
    clock = [0.]
    guard = MemoryGuard(1, "cpu", clock=lambda: clock[0], pressure=lambda: 100,
                        resident=lambda: samples.append(clock[0]) or 1024,
                        footprint=lambda: 1024, peak=lambda: 1024)
    return guard, samples, clock


@pytest.mark.parametrize("hosted", [False, True])
def test_identity_chunks_retain_interval_and_full_content(monkeypatch, hosted):
    guard, samples, clock = guarded_memory()
    owner = _RunGuard(guard) if hosted else guard
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 128)
    tensor = torch.arange(1024, dtype=torch.float32)
    state = {"weight": tensor}
    original = torch.Tensor.to
    copies = []

    def copied(value, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            copies.append(value.numel() * value.element_size())
            if len(copies) == 16:
                clock[0] = 16.
        return original(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", copied)
        identity = _content_identity(state, owner)
    assert len(copies) == 32 and sum(copies) == tensor.numel() * tensor.element_size()
    assert max(copies) == 128 and samples == [0., 16.]
    assert identity == _dictionary._state_hash(state)
    version = tensor._version
    tensor.data[-1].add_(1)
    assert tensor._version == version
    assert _content_identity(state, owner) != identity


def test_interval_guard_retains_legacy_checks_and_stop_latch(monkeypatch):
    guard, samples, _ = guarded_memory()
    checks, stopped = [], []
    failure = CleanStop("chunk limit")

    def legacy(stage):
        checks.append(stage)
        if len(checks) == 3:
            raise failure

    extra = SimpleNamespace(check=legacy, limit=guard.limit, peak=0)
    owner = _RunGuard(guard, extra=extra, on_stop=stopped.append)
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 128)
    with pytest.raises(CleanStop) as caught:
        _content_identity({"weight": torch.ones(1024)}, owner)
    assert caught.value is failure and stopped == [failure]
    assert len(checks) == 3 and samples == [0.]
    with pytest.raises(CleanStop) as caught:
        owner._check_interval("later")
    assert caught.value is failure and len(checks) == 3


def test_interval_admission_observes_a_lower_limit_before_each_copy(monkeypatch):
    guard, samples, _ = guarded_memory()
    owner = _RunGuard(guard)
    original = torch.Tensor.to
    copies = []
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 128)

    def copied(value, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            copies.append(value.numel())
            guard.limit = 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copied)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _content_identity({"weight": torch.ones(1024)}, owner)
    assert copies == [32] and samples == [0.]


def test_new_clamp_checks_bytes_at_admission_and_installation(hosted_run, monkeypatch):
    from .test_hosted import _trace_at_boundary
    run, _ = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    snapshot = run.interventions.executor._dictionary_snapshot
    assert snapshot is not None
    original = snapshot.matches
    checks = []
    def identity(*args):
        checks.append(1)
        return original(*args)
    monkeypatch.setattr(snapshot, "matches", identity)
    run.apply_clamp()
    assert checks == [1, 1]
    owned = run._owned_clamp
    checks.clear()
    run.apply_clamp()
    assert checks == [1] and run._owned_clamp is owned
    # Alias writes do not change Tensor._version and must still refuse reuse.
    run.dictionary.layers[0].W_dec.data[0, 0].add_(.125)
    with pytest.raises(ValueError, match="dictionary content changed"):
        run.apply_clamp()
    assert run._owned_clamp is owned


def test_clamp_diagnostics_retain_completion_and_do_not_validate_on_status(hosted_run):
    from caliper import _wire
    from .test_hosted import _trace_at_boundary
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    before = run.inference_status()
    assert route.on_control(None, _wire.Control(1, "set", name="clamp", value=0.).to_meta())["status"] == "ok"
    assert route.on_control(None, _wire.Control(2, "run", name="apply_clamp").to_meta())["status"] == "queued"
    run.boundary()
    after = run.inference_status()
    identity = after["dictionary_validation"]
    assert identity["mode"] == "snapshot"
    assert identity["snapshot_bytes"] == sum(t.numel() * t.element_size() for t in run.dictionary.state_dict().values())
    assert identity["checks"] - before["dictionary_validation"]["checks"] == 3
    assert identity["seconds"] > before["dictionary_validation"]["seconds"]
    comparison = after["comparison"]
    assert comparison["compute_publish_seconds"] > 0
    assert comparison["completed_step"] == run.controls.n > before["comparison"]["completed_step"]
    assert comparison["interventions_hash"] == after["interventions_hash"]
    assert comparison["prompt_hash"] == after["prompt_hash"]
    assert run.inference_status() == after
    identity["checks"] = -1
    assert run.inference_status()["dictionary_validation"]["checks"] >= 3


def test_cross_layer_action_replaces_hook_and_publishes_comparison_with_two_scans(hosted_run):
    from caliper import _wire
    from .test_hosted_controls import _feature_rows, _queue_action

    run, route = hosted_run
    rows = _feature_rows(run)
    run.selected.set(rows[0])
    run.clamp.set(0.)
    run.boundary()
    old = run._owned_record()
    row = next(row for row in rows if run.service.graph.nodes[row]["layer"] != run.selection["layer"])
    before = run.inference_status()["dictionary_validation"]["checks"]
    reply = route.on_control(None, _wire.Control(1, "set", name="selected_node", value=row).to_meta())
    assert reply["status"] == "ok"
    _queue_action(route, "apply_clamp", 2)
    run.boundary()
    reply, = [reply for reply in route.transport.replies if reply.request_id == 2]
    assert reply.status == "ok"
    assert old.hook.removed and len(run.interventions.specifications()) == 1
    status = run.inference_status()
    assert status["comparison"]["completed_step"] == run.controls.n
    assert status["comparison"]["interventions_hash"] == status["interventions_hash"]
    assert status["dictionary_validation"]["checks"] - before == 2
