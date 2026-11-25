"""Bounded CPU coverage for complete hosted intervention checkpoints."""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import caliper
from caliper import _controls, _managed, _wire
from caliper._intervention_protocol import specification
from caliper.interp._hosted import (
    CHECKPOINT_BYTES, CHECKPOINT_VALUES, HostedInference, HostedService, TinyTokenizer,
    _check_checkpoint_interventions, checkpoint_metadata,
)
from caliper.interp._intervene import InterventionError
from caliper.interp._service import register_service


class WideBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Sequential(OrderedDict(pattern=nn.Identity()))
        self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class WideFixture(nn.Module):
    """Supply semantic points and wide outputs with one scalar parameter."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.layers = nn.ModuleList(WideBlock() for _ in range(22))
        self.config = SimpleNamespace(model_type="caliper_tiny", hidden_size=width,
                                      num_hidden_layers=22, vocab_size=512)
        self.eval()

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


def _points():
    return [f"layers.{layer}.{point}" for layer in range(22)
            for point in ("resid_pre", "mlp_in", "mlp_out")]


def _add(point, width, value=0., **schedule):
    return dict(kind="add", point=point, vector=[value] * width, coefficient=1., **schedule)


@pytest.fixture
def wide_run(tmp_path, monkeypatch):
    def create(width):
        model = WideFixture(width)
        controls = _controls.Controls()
        monkeypatch.setattr(caliper, "_CTL", controls)
        index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"},
                                close=lambda: None)
        run = HostedInference("checkpoint-fixture", tmp_path, "AB",
                              resources=(model, TinyTokenizer(), None, index, None))
        run.controls = controls
        run.interventions = controls.interventions_on(model)
        run.service = register_service(HostedService(run), controls=controls)
        # These tests exercise admission, hooks, and checkpoints without output or trace work.
        monkeypatch.setattr(run, "update_controls", lambda: None)
        monkeypatch.setattr(run, "update_output", lambda: None)
        managed = _managed.ManagedRun("capacity", str(tmp_path))
        managed._pkg = SimpleNamespace(_CTL=controls)
        managed.inference = run
        managed.kept.extension = ".json"
        # A fixed clock makes each selected checkpoint due without a wall-clock wait.
        managed.kept.clock = lambda: 0.
        run.managed = managed
        controls.param("prompt", "AB")
        runs.append(run)
        return run

    runs = []
    yield create
    for run in runs:
        run.interventions.remove()
        run.close()


def _read_checkpoint(run):
    path = Path(run.managed.checkpoint(run.model, None))
    assert path.stat().st_size <= CHECKPOINT_BYTES
    record = json.loads(path.read_bytes())
    assert checkpoint_metadata(record)
    return record


@pytest.mark.parametrize(("width", "requested", "accepted"), [(2304, 57, 56), (1920, 64, 64)])
def test_full_active_snapshot_and_automatic_checkpoints(wide_run, width, requested, accepted):
    run = wide_run(width)
    service = run.interventions
    for point in _points()[:accepted]:
        service.set(_add(point, width))
    if requested > accepted:
        before = service.state_dict()
        with pytest.raises(InterventionError, match="checkpoint metadata") as refused:
            service.set(_add(_points()[accepted], width))
        assert refused.value.code == _wire.ERR_LIMIT
        assert service.state_dict() == before
        assert not service.executor._hooks

    pending = _read_checkpoint(run)
    assert len(pending["interventions"]["interventions"]) == accepted
    assert all(record["applied_specification"] is None for record in pending["interventions"]["interventions"])
    run.boundary()
    active = _read_checkpoint(run)
    records = active["interventions"]["interventions"]
    assert len(records) == accepted
    assert all(record["state"] == "active" and record["specification"] == record["applied_specification"]
               for record in records)
    assert len(service.executor._hooks) == accepted
    with torch.no_grad():
        torch.testing.assert_close(run.model(torch.zeros(1, 1, width)), torch.zeros(1, 1, width))
    assert not run.managed.kept.off
    kept = Path(run.managed.dir) / "kept" / "checkpoint-1.json"
    assert json.loads(kept.read_bytes()) == active

    if width == 2304:
        # This replacement fits now, but its second copy cannot fit after application.
        future = deepcopy(active)
        for record in future["interventions"]["interventions"]:
            record["specification"]["positions"] = list(range(32))
        assert checkpoint_metadata(future)
        hooks = set(service.executor._hooks)
        with pytest.raises(InterventionError, match="checkpoint metadata") as refused:
            service.load_state_dict(future["interventions"], restore_active=True)
        assert refused.value.code == _wire.ERR_LIMIT
        assert set(service.executor._hooks) == hooks
        assert service.state_dict() == active["interventions"]

    # Distinct pending replacements retain all old applied vectors until the boundary.
    replacement = _add(_points()[0], width, .5, from_step=5)
    service.set(replacement)
    queued = _read_checkpoint(run)
    first = queued["interventions"]["interventions"][0]
    assert first["specification"]["vector"][0] == .5
    assert first["applied_specification"]["vector"][0] == 0.
    for record in queued["interventions"]["interventions"]:
        record["specification"].update(vector=[.5] * width, from_step=5)
    run.load_checkpoint(queued)
    assert run.restored == queued
    service.load_state_dict(queued["interventions"], restore_active=True)
    assert service.state_dict() == queued["interventions"]
    assert _read_checkpoint(run) == queued
    run.boundary()
    assert not run.managed.kept.off
    kept = Path(run.managed.dir) / "kept" / "checkpoint-2.json"
    assert json.loads(kept.read_bytes()) == _read_checkpoint(run)

    # Reserve maximum error text, timestamps, and every text parameter slot.
    worst = deepcopy(queued)
    worst["params"] = {str(i): "\x01" * 4096 for i in range(_controls.MAX_PARAMS)}
    for record in worst["interventions"]["interventions"]:
        record.update(since_step=2**53 - 1, error="\x01" * 1024, error_code="\x01" * 63)
    assert checkpoint_metadata(worst)


def test_review_snapshot_exceeds_values_before_bytes(wide_run):
    run = wide_run(2304)
    # A plain intervention service reproduces the former admission capacity.
    run.controls.interpretation_service = None
    for point in _points()[:57]:
        run.interventions.set(_add(point, 2304))
    run.controls.step(1)
    snapshot = run.state_dict()
    records = snapshot["interventions"]["interventions"]
    assert len(records) == 57 < _controls.MAX_INTERVENTIONS
    assert all(record["state"] == "active" for record in records)
    assert 57 * 2 * 2304 > CHECKPOINT_VALUES
    assert len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")) < CHECKPOINT_BYTES
    assert not checkpoint_metadata(snapshot)
    run.controls.interpretation_service = run.service
    with pytest.raises(InterventionError, match="checkpoint metadata"):
        run.interventions.load_state_dict(snapshot["interventions"], restore_active=True)


def test_full_ended_history_remains_checkpointable(wide_run):
    run = wide_run(2304)
    records = [{"specification": dict(_add(point, 2304), op="add"),
                "state": "ended", "applied_specification": None, "since_step": 1,
                "clear": False, "error": None, "error_code": None}
               for point in _points()[:64]]
    state = {"version": 1, "interventions": records}
    run.interventions.load_state_dict(state, restore_active=True)
    assert run.interventions.state_dict() == state
    assert not run.interventions.specifications()
    assert _read_checkpoint(run)["interventions"] == state
    run.boundary()
    assert not run.managed.kept.off
    assert run.interventions.state_dict() == state


def test_checkpoint_admission_bounds_bytes_and_both_replacement_outcomes():
    records = {}
    for point in _points()[:10]:
        spec = _add(point, 8192, -1.7976931348623157e308)
        assert specification(spec)
        iv = _controls._Intervention(point, "add", None, None, spec)
        records[(point, "add")] = iv
    # Both copies have fewer than the maximum values, but their bytes exceed the budget.
    assert 10 * 2 * (8192 + 5) < CHECKPOINT_VALUES
    with pytest.raises(ValueError, match="checkpoint metadata"):
        _check_checkpoint_interventions(records)
    for iv in records.values():
        iv.applied_spec = iv.spec
        iv.spec = dict(_add(iv.point, 8192), positions=list(range(32)))
    # A smaller request must reserve the larger retained effect and a possible rollback.
    with pytest.raises(ValueError, match="checkpoint metadata"):
        _check_checkpoint_interventions(records)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2**53, object()])
def test_checkpoint_rejects_unsafe_values(value):
    assert not checkpoint_metadata({"vector": [value]})


def test_checkpoint_preserves_structural_and_byte_limits():
    assert checkpoint_metadata({"vector": [0.] * 16384})
    assert not checkpoint_metadata({"vector": [0.] * 16385})
    assert not checkpoint_metadata({str(i): None for i in range(65)})
    assert not checkpoint_metadata({"x" * 64: None})
    assert not checkpoint_metadata({"text": "x" * 4097})
    assert not checkpoint_metadata({"text": "\ud800"})
    assert not checkpoint_metadata({"tuple": (1, 2)})
    nested = 0
    for _ in range(8):
        nested = [nested]
    assert not checkpoint_metadata({"depth": nested})
    assert checkpoint_metadata({"depth": nested[0]})
    too_many_bytes = {str(i): ["\x01" * 4096] * 3 for i in range(64)}
    assert len(json.dumps(too_many_bytes).encode("utf-8")) > CHECKPOINT_BYTES
    assert not checkpoint_metadata(too_many_bytes)


def test_checkpoint_value_count_retains_its_exact_boundary():
    value = {str(i): [0] * 16384 for i in range(15)}
    value["last"] = [0] * (CHECKPOINT_VALUES - 15 * 16384 - 17)
    assert checkpoint_metadata(value)
    value["last"].append(0)
    assert not checkpoint_metadata(value)


def test_version_one_records_without_applied_copy_remain_loadable(wide_run):
    run = wide_run(16)
    run.interventions.set(_add(_points()[0], 16))
    run.boundary()
    state = run.interventions.state_dict()
    legacy = deepcopy(state)
    del legacy["interventions"][0]["applied_specification"]
    run.interventions.load_state_dict(legacy, restore_active=True)
    assert run.interventions.state_dict() == state
    assert checkpoint_metadata(_read_checkpoint(run))
