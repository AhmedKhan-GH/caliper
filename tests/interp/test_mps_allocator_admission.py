"""Allocator admission with scalar metrics and no real MPS method calls."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen, _memory


GB = 1_000_000_000
MARGIN = 512_000_000
OPTIONS = dict(layers=26, tokens=11, width=2304, mixer_width=2304, itemsize=2)
PER_TARGET = 53_932_032


@pytest.fixture
def metrics(monkeypatch):
    state = SimpleNamespace(recommended=64 * GB, allocated=0, driver=0,
                            current=0, peak=0, host=64 * GB, fractions=[])

    def forbidden(*args, **kwargs):
        pytest.fail("a real MPS method must not run")

    # Module-local facades supply every device method. The real APIs stay blocked.
    names = ("recommended_max_memory", "current_allocated_memory", "driver_allocated_memory",
             "set_per_process_memory_fraction", "empty_cache", "synchronize")
    for name in names:
        monkeypatch.setattr(torch.mps, name, forbidden)
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden)
    facade = SimpleNamespace(device=torch.device, backends=SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True)), mps=SimpleNamespace(
            recommended_max_memory=lambda: state.recommended,
            current_allocated_memory=lambda: state.allocated,
            driver_allocated_memory=lambda: state.driver,
            set_per_process_memory_fraction=state.fractions.append))
    monkeypatch.setattr(_memory, "torch", facade)
    monkeypatch.setattr(_frozen, "torch", facade)
    monkeypatch.setattr(_memory, "_mps_allocator_limit", None)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=state.host))

    def process(limit, device, **kwargs):
        return _memory.MemoryGuard(limit, device, pressure=lambda: 80,
            resident=lambda: state.current, footprint=lambda: state.current,
            peak=lambda: state.peak, clock=lambda: 0)

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", process)
    state.process = process
    state.trace = lambda limit=40, **kw: _frozen.MemoryGuard("mps", memory_limit_gb=limit, **kw)
    return state


@pytest.mark.parametrize("budget,recommended,ceiling", [(40, 64, 32), (12, 64, 9.6), (40, 20, 20)])
def test_owned_ceiling_matches_successful_setter(metrics, budget, recommended, ceiling):
    metrics.recommended = recommended * GB
    guard = metrics.process(budget, "mps")
    assert guard.mps_allocator_limit == int(ceiling * GB)
    assert len(metrics.fractions) == 1
    assert 0 < metrics.fractions[0] <= 1
    assert int(metrics.fractions[0] * metrics.recommended) == guard.mps_allocator_limit


def test_later_guards_lower_the_owned_cap_without_a_restore(metrics):
    original = metrics.trace()
    lower = metrics.trace(12)
    later = metrics.trace()
    assert metrics.fractions == [.5, .15]
    assert original.limit == later.limit == 40 * GB
    assert lower.limit == 12 * GB
    assert original._guard.mps_allocator_limit == later._guard.mps_allocator_limit == 9_600_000_000
    metrics.allocated = metrics.driver = 9_600_000_000 - MARGIN - PER_TARGET
    assert original.batch_size(512, **OPTIONS) == later.batch_size(512, **OPTIONS) == 1
    assert metrics.fractions == [.5, .15]


@pytest.mark.parametrize("existing", [None, 32 * GB])
def test_failed_setter_does_not_record_an_unowned_ceiling(metrics, monkeypatch, existing):
    monkeypatch.setattr(_memory, "_mps_allocator_limit", existing)

    def fail(fraction):
        raise RuntimeError("setter failure")

    monkeypatch.setattr(_memory.torch.mps, "set_per_process_memory_fraction", fail)
    with pytest.raises(RuntimeError, match="setter failure"):
        metrics.process(12, "mps")
    assert _memory._mps_allocator_limit == existing


def test_subbyte_ceiling_cannot_disable_the_allocator_cap(metrics):
    with pytest.raises(_memory.CleanStop, match="ceiling must be positive"):
        metrics.process(1e-9, "mps")
    assert metrics.fractions == []
    assert _memory._mps_allocator_limit is None


@pytest.mark.parametrize("capacity", [1, 64, 129, 512])
def test_batch_cap_boundary_and_single_byte_overflow(metrics, capacity):
    guard = metrics.trace()
    metrics.driver = 32 * GB - MARGIN - capacity * PER_TARGET
    metrics.allocated = metrics.driver - GB
    assert guard.batch_size(512, **OPTIONS) == capacity
    assert guard.batch_size(1, **OPTIONS) == 1
    metrics.driver += 1
    if capacity == 1:
        with pytest.raises(_memory.CleanStop, match="no target batch fits"):
            guard.batch_size(512, **OPTIONS)
    else:
        assert guard.batch_size(512, **OPTIONS) == capacity - 1
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("allocated,driver", [(24, 26), (26, 24), (26, 26)])
def test_driver_cache_and_tensor_bytes_overlap(metrics, allocated, driver):
    guard = metrics.trace()
    metrics.allocated, metrics.driver = allocated * GB, driver * GB
    assert guard.batch_size(512, **OPTIONS) == (6 * GB - MARGIN) // PER_TARGET
    assert guard.records[-1]["allocated_bytes"] == allocated * GB
    assert guard.records[-1]["driver_bytes"] == driver * GB
    # A later forced read must see cache growth even if tensor use stays constant.
    metrics.driver = 32 * GB - MARGIN
    with pytest.raises(_memory.CleanStop, match="no target batch fits"):
        guard.batch_size(1, **OPTIONS)


def test_reported_second_batch_uses_allocator_headroom(metrics):
    guard = metrics.trace(host_reserve_bytes=8 * GB)
    metrics.allocated = 22_479_155_712
    metrics.driver = 23_138_598_912
    metrics.current = 25_259_641_568
    # Give the host more space than the device needs to isolate the allocator cap.
    assert guard.batch_size(512, **OPTIONS) == 154
    assert metrics.driver + MARGIN + 154 * PER_TARGET <= 32 * GB
    assert metrics.driver + MARGIN + 155 * PER_TARGET > 32 * GB


@pytest.mark.parametrize("owner", ["trace", "process", "supplied", "budget"])
def test_lower_live_budgets_bound_admission_without_resetting_allocator(metrics, owner):
    guard = metrics.trace()
    supplied = SimpleNamespace(limit=40 * GB, peak=0, current=0, check=lambda stage: 0)
    guard._guard = _memory.BudgetMemory(guard._guard, 40, extra=supplied)
    target = {"trace": guard, "process": guard._guard.memory,
              "supplied": supplied, "budget": guard._guard}[owner]
    target.limit = 12 * GB
    metrics.driver = metrics.allocated = 9_600_000_000 - MARGIN - PER_TARGET
    assert guard.limit == 12 * GB
    assert guard.batch_size(512, **OPTIONS) == 1
    guard.reserve("lower budget", MARGIN + PER_TARGET)
    with pytest.raises(_memory.CleanStop, match="MPS allocator limit"):
        guard.reserve("lower budget", MARGIN + PER_TARGET + 1)
    assert guard._guard.mps_allocator_limit == 32 * GB
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("bound", ["process", "host"])
def test_device_space_cannot_override_process_or_host_reserves(metrics, bound):
    guard = metrics.trace(host_reserve_bytes=8 * GB)
    if bound == "process":
        metrics.current = 40 * GB - MARGIN - PER_TARGET
    else:
        metrics.host = 8 * GB + MARGIN + PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == 1
    if bound == "process":
        metrics.current += 1
    else:
        metrics.host -= 1
    with pytest.raises(_memory.CleanStop, match="no target batch fits"):
        guard.batch_size(1, **OPTIONS)


@pytest.mark.parametrize("method", ["reserve", "batch"])
def test_supplied_stop_and_historical_breach_remain_latched(metrics, method):
    guard = metrics.trace()
    failure = _memory.CleanStop("supplied stop")

    def stop(stage):
        raise failure

    supplied = SimpleNamespace(limit=40 * GB, peak=0, current=0, check=stop)
    guard._guard = _memory.BudgetMemory(guard._guard, 40, extra=supplied)

    def admit():
        return guard.reserve("known allocation", 1) if method == "reserve" else guard.batch_size(1, **OPTIONS)

    with pytest.raises(_memory.CleanStop) as caught:
        admit()
    assert caught.value is failure
    supplied.check = lambda stage: 0
    supplied.peak = 40 * GB + 1
    with pytest.raises(_memory.CleanStop, match="memory limit"):
        admit()
    supplied.peak = 0
    with pytest.raises(_memory.CleanStop, match="memory limit"):
        admit()


def test_released_process_peak_does_not_consume_device_headroom(metrics):
    metrics.peak = 39 * GB
    guard = metrics.trace()
    metrics.driver = 32 * GB - MARGIN - 2 * PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == 2
    assert guard._guard.peak == 39 * GB


@pytest.mark.parametrize("metric", ["allocated", "driver"])
def test_known_allocation_reserve_obeys_exact_allocator_boundary(metrics, metric):
    guard = metrics.trace()
    setattr(metrics, metric, 32 * GB - 4096)
    guard.reserve("decoder workspace", 4096)
    with pytest.raises(_memory.CleanStop, match="MPS allocator limit"):
        guard.reserve("decoder workspace", 4097)
    setattr(metrics, metric, 32 * GB + 1)
    with pytest.raises(_memory.CleanStop, match="MPS allocator limit"):
        guard.reserve("no device space", 1)


def test_known_allocation_retains_host_reserve(metrics):
    guard = metrics.trace(host_reserve_bytes=8 * GB)
    metrics.host = 8 * GB + 4096
    guard.reserve("workspace", 4096)
    metrics.host -= 1
    with pytest.raises(_memory.CleanStop, match="8 GB host reserve"):
        guard.reserve("workspace", 4096)


def test_unknown_owned_cap_refuses_admission(metrics, monkeypatch):
    guard = metrics.trace()
    monkeypatch.setattr(_memory, "_mps_allocator_limit", None)
    with pytest.raises(_memory.CleanStop, match="ceiling is unavailable"):
        guard.batch_size(1, **OPTIONS)
    with pytest.raises(_memory.CleanStop, match="ceiling is unavailable"):
        guard.reserve("workspace", 1)


def test_cpu_admission_never_reads_or_changes_mps_state(metrics):
    for name in vars(_memory.torch.mps):
        setattr(_memory.torch.mps, name, lambda *a: pytest.fail("CPU admission called MPS"))
    guard = _frozen.MemoryGuard("cpu")
    metrics.current = 12 * GB - MARGIN - 2 * PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == 2
    assert guard.batch_size(512, **(OPTIONS | {"itemsize": 8})) == 1
    guard.reserve("CPU workspace", MARGIN + 2 * PER_TARGET)
    with pytest.raises(_memory.CleanStop, match="trace memory limit"):
        guard.reserve("CPU workspace", MARGIN + 2 * PER_TARGET + 1)
    metrics.current = 12 * GB - MARGIN
    with pytest.raises(_memory.CleanStop, match="^no target batch fits the process memory limit$"):
        guard.batch_size(1, **OPTIONS)
    assert guard._guard.mps_allocator_limit is None
    assert _memory._mps_allocator_limit is None and metrics.fractions == []
