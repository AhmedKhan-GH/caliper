"""Frozen path caps with deterministic CPU tensors and scalar memory readers.

Scalar readers provide no GPU or large-memory evidence.
"""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen
from caliper.interp._memory import CleanStop, MemoryGuard as ProcessMemoryGuard
from .fixture import TinyLM
from .test_trace import exact_dictionary


class SuppliedGuard(_frozen.MemoryGuard):
    def __init__(self, limit):
        self.failure = None
        self.stages = []
        super().__init__("cpu", memory_limit_gb=limit)

    def _stop(self, stage):
        self.stages.append(stage)
        if self.failure is not None:
            raise self.failure

    def check(self, stage):
        self._stop(stage)
        return super().check(stage)

    def _check_interval(self, stage):
        self._stop(stage)
        return super()._check_interval(stage)


@pytest.fixture
def case(monkeypatch):
    readings = {"current": 1_000, "peak": 1_000, "clock": 0.}
    created, identities = [], []

    def create(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        memory = ProcessMemoryGuard(
            limit, device, pressure=lambda: 80, resident=lambda: readings["current"],
            footprint=lambda: readings["current"], peak=lambda: readings["peak"],
            driver=lambda: 0, clock=lambda: readings["clock"])
        created.append(memory)
        return memory

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", create)
    model = TinyLM(layers=2, width=8, heads=2, vocab=32, latents=8, k=2)
    dictionary = exact_dictionary(model)
    identify = _frozen.model_identity

    def identity(model, **kwargs):
        result = identify(model, **kwargs)
        identities.append(kwargs["memory"].limit)
        return result

    monkeypatch.setattr(_frozen, "model_identity", identity)
    supplied = SuppliedGuard(12)
    ids = torch.tensor([[1, 2]])

    def path(limit=1, guard=supplied, **kwargs):
        return _frozen.FrozenPath(model, ids, dictionary, guard=guard, memory_limit_gb=limit, **kwargs)

    return SimpleNamespace(model=model, dictionary=dictionary, ids=ids, supplied=supplied,
                           readings=readings, created=created, identities=identities, path=path)


def batch_options(path):
    return dict(layers=path.layers, tokens=path.tokens, width=path.width,
                mixer_width=path.mixer_width, itemsize=path.embedding.element_size())


def forbidden(*args, **kwargs):
    pytest.fail("memory refusal must precede tensor work")


def test_lower_cap_refuses_reserve_after_successful_identity(case, monkeypatch):
    path = case.path()
    assert case.identities == [1_000_000_000]
    case.supplied.reserve("borrowed reserve", 1_000_000_000)
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        path.guard.reserve("path reserve", 1_000_000_000)
        torch.empty(1)
    assert path.guard.limit == 1_000_000_000
    assert case.supplied.limit == case.supplied._guard.limit == 12_000_000_000


def test_lower_cap_refuses_target_batch_before_tensor_work(case, monkeypatch):
    path = case.path()
    assert case.identities == [1_000_000_000]
    case.readings["current"] = 600_000_000
    options = batch_options(path)
    assert case.supplied.batch_size(64, **options) == 64
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(CleanStop, match="no target batch fits"):
        count = path.guard.batch_size(64, **options)
        torch.empty(count)


def test_lower_cap_reduces_target_batch(case):
    path = case.path()
    options = batch_options(path)
    per_target = options["itemsize"] * path.tokens * (
        path.width * (path.layers + 1) * 12 + path.mixer_width * path.layers * 8)
    case.readings["current"] = 1_000_000_000 - 512_000_000 - 2 * per_target
    assert path.guard.batch_size(64, **options) == 2
    assert case.supplied.batch_size(64, **options) == 64


@pytest.mark.parametrize("supplied_limit,requested", [(1, 1), (.75, 1), (1, None)])
def test_equal_or_stricter_supplied_cap_reuses_guard(case, supplied_limit, requested):
    supplied = SuppliedGuard(supplied_limit)
    count = len(case.created)
    path = case.path(requested, supplied)
    assert path.guard is supplied and len(case.created) == count
    assert case.identities == [int(supplied_limit * 1e9)]
    with pytest.raises(CleanStop, match="allocation exceeds"):
        path.guard.reserve("supplied cap", int(supplied_limit * 1e9))


@pytest.mark.parametrize("when", ["before_identity", "later"])
@pytest.mark.parametrize("fault", ["stop", "historical", "pressure"])
def test_supplied_stops_and_history_remain_active(case, monkeypatch, when, fault):
    path = case.path() if when == "later" else None
    memory = case.supplied._guard
    failure = CleanStop("borrowed stop")
    if fault == "stop":
        case.supplied.failure = failure
    elif fault == "historical":
        memory.peak = 12_000_000_001
    else:
        memory.pressure = lambda: 10
        memory._pressure_time = -float("inf")
    monkeypatch.setattr(_frozen, "model_identity", forbidden)
    with pytest.raises(CleanStop) as caught:
        case.path() if path is None else path.guard.reserve("later reserve", 1)
    if fault == "stop":
        assert caught.value is case.supplied.failure is failure
    elif fault == "historical":
        assert memory.peak == 12_000_000_001
    else:
        assert "memory pressure" in str(caught.value)
    assert case.supplied._guard is memory
    assert case.supplied.limit == memory.limit == 12_000_000_000
    assert case.identities == ([] if path is None else [1_000_000_000])


@pytest.mark.parametrize("owner", ["outer", "inner"])
@pytest.mark.parametrize("supplied_limit", [12, 1, .75])
def test_dynamic_supplied_limit_controls_reserves_and_batches(case, owner, supplied_limit):
    supplied = SuppliedGuard(supplied_limit)
    path = case.path(guard=supplied)
    memory = supplied if owner == "outer" else supplied._guard
    memory.limit = 250_000_000
    assert path.guard.limit == 250_000_000
    with pytest.raises(CleanStop, match="allocation exceeds"):
        path.guard.reserve("dynamic reserve", 250_000_000)
    with pytest.raises(CleanStop, match="no target batch fits"):
        path.guard.batch_size(1, **batch_options(path))
    assert memory.limit == 250_000_000


def test_reserves_and_batches_use_higher_borrowed_current(case):
    path = case.path()
    memory = case.supplied._guard
    memory.resident = memory.footprint = memory.peak_reader = lambda: 600_000_000
    with pytest.raises(CleanStop, match="allocation exceeds"):
        path.guard.reserve("borrowed current", 500_000_000)
    with pytest.raises(CleanStop, match="no target batch fits"):
        path.guard.batch_size(1, **batch_options(path))
    assert case.readings["current"] == 1_000


def test_private_cap_preserves_borrowed_objects_and_peak(case):
    memory, records, stages = case.supplied._guard, case.supplied.records, case.supplied.stages
    memory.peak = 900_000_000
    path = case.path()
    path.guard.reserve("released peak", 500_000_000)
    assert case.supplied._guard is memory and case.supplied.records is records
    assert case.supplied.stages is stages and "released peak" in stages
    assert case.supplied.limit == memory.limit == 12_000_000_000
    assert memory.peak == 900_000_000 and case.supplied.failure is None
    assert "finite block" in stages and "frozen layer 0" in stages
    memory.limit = 800_000_000
    with pytest.raises(CleanStop, match="memory limit"):
        path.guard.reserve("prior supplied peak", 1)
    assert memory.peak == 900_000_000


def test_private_cap_retains_its_own_historical_breach(case):
    path = case.path()
    case.readings["peak"] = 1_000_000_001
    with pytest.raises(CleanStop, match="memory limit"):
        path.guard.check("private breach")
    case.readings["peak"] = 1_000
    with pytest.raises(CleanStop, match="memory limit"):
        path.guard.reserve("released private breach", 1)
    assert case.supplied._guard.peak == 1_000


@pytest.mark.parametrize("phase", ["dictionary", "features"])
def test_finite_checks_keep_lower_cap_after_identity(case, monkeypatch, phase):
    finite, isfinite = _frozen._require_finite, torch.isfinite
    triggered = []

    def require(value, name, check):
        if not triggered and (name.startswith("dictionary") if phase == "dictionary" else name == "layer 0 features"):
            assert case.identities == [1_000_000_000]
            triggered.append(name)
            case.readings.update(current=1_000_000_001, clock=20.)
        return finite(value, name, check)

    def evaluate(value):
        if triggered:
            forbidden()
        return isfinite(value)

    monkeypatch.setattr(_frozen, "_require_finite", require)
    monkeypatch.setattr(torch, "isfinite", evaluate)
    with pytest.raises(CleanStop, match="memory limit"):
        case.path()
    assert triggered
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())


def test_lower_cap_refuses_record_allocation_after_identity(case, monkeypatch):
    def check(stage):
        if stage == "record admission":
            assert case.identities == [1_000_000_000]
            case.readings["current"] = 1_000_000_001

    monkeypatch.setattr(case.model, "forward", forbidden)
    with pytest.raises(CleanStop, match="memory limit"):
        case.path(check=check)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())


@pytest.mark.parametrize("operation", ["run", "replacement", "logits", "target"])
def test_later_path_work_refuses_before_tensor_operations(case, monkeypatch, operation):
    path = case.path()
    assert case.identities == [1_000_000_000]
    case.readings["current"] = 1_000_000_001
    monkeypatch.setattr(_frozen.FrozenNorm, "__call__", forbidden)
    monkeypatch.setattr(_frozen.F, "linear", forbidden)
    monkeypatch.setattr(torch.Tensor, "new_zeros", forbidden)
    with pytest.raises(CleanStop, match="memory limit"):
        if operation == "run":
            path.run(path.embedding, path.writes)
        elif operation == "replacement":
            path.replacement(path.embedding, case.dictionary)
        elif operation == "logits":
            path.logits_from_final(path.embedding[:, -1])
        else:
            path.logit_target(0)
