"""Scalar cache scenarios verify admission, without device or timing evidence."""
from types import SimpleNamespace

import pytest

from caliper.interp import _frozen, _memory
from .test_mps_allocator_admission import GB, MARGIN, OPTIONS, PER_TARGET, metrics


@pytest.fixture
def cache(metrics, monkeypatch):
    state = SimpleNamespace(calls=0, changes={})

    def release():
        state.calls += 1
        vars(metrics).update(state.changes)

    monkeypatch.setattr(_frozen.torch.mps, "empty_cache", release)
    return state


@pytest.mark.parametrize("reclaimed,expected", [(False, 45), (True, 154)])
def test_cache_release_uses_measured_driver_bytes(metrics, cache, reclaimed, expected):
    guard = metrics.trace(host_reserve_bytes=8 * GB)
    metrics.allocated = 22_478_893_568
    metrics.driver = 29_010_624_512
    metrics.current = 31_145_429_752
    if reclaimed:
        # These post-release values are hypothetical, not a device measurement.
        cache.changes.update(driver=23_138_598_912, current=25_259_641_568)
    assert guard.batch_size(512, **OPTIONS) == expected
    assert cache.calls == 1
    assert metrics.allocated == 22_478_893_568
    assert guard.records[-1]["driver_bytes"] == metrics.driver
    assert guard._guard.peak == 31_145_429_752
    assert guard._guard.mps_allocator_limit == 32 * GB
    assert metrics.driver + MARGIN + expected * PER_TARGET <= 32 * GB
    assert metrics.driver + MARGIN + (expected + 1) * PER_TARGET > 32 * GB
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("driver", [32 * GB - MARGIN, 32 * GB + 1])
def test_unreclaimable_driver_memory_still_refuses(metrics, cache, driver):
    guard = metrics.trace()
    metrics.allocated, metrics.driver = 22 * GB, driver
    with pytest.raises(_memory.CleanStop, match="no target batch fits the process or MPS allocator memory limit"):
        guard.batch_size(512, **OPTIONS)
    assert cache.calls == 1
    assert guard.records[-1]["driver_bytes"] == driver
    assert metrics.fractions == [.5]


def test_release_refreshes_all_counters_before_admission(metrics, cache, monkeypatch):
    guard = metrics.trace(host_reserve_bytes=8 * GB)
    metrics.allocated, metrics.driver, metrics.current = 24 * GB, 29 * GB, 31 * GB
    reads = []

    def read(name, field):
        reads.append(name)
        return getattr(metrics, field)

    monkeypatch.setattr(guard._guard, "resident", lambda: read("resident", "current"))
    monkeypatch.setattr(guard._guard, "footprint", lambda: read("footprint", "current"))
    monkeypatch.setattr(guard._guard, "peak_reader", lambda: read("peak", "peak"))
    monkeypatch.setattr(_frozen.torch.mps, "current_allocated_memory", lambda: read("allocated", "allocated"))
    monkeypatch.setattr(_frozen.torch.mps, "driver_allocated_memory", lambda: read("driver", "driver"))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=read("host", "host")))
    release = _frozen.torch.mps.empty_cache

    def flush():
        reads.append("release")
        release()

    monkeypatch.setattr(_frozen.torch.mps, "empty_cache", flush)
    cache.changes.update(allocated=22 * GB, driver=23 * GB, current=25 * GB, peak=33 * GB, host=9 * GB)
    assert guard.batch_size(512, **OPTIONS) == (GB - MARGIN) // PER_TARGET
    before, after = reads[:reads.index("release")], reads[reads.index("release") + 1:]
    assert set(before) == {"resident", "footprint", "peak", "allocated", "driver"}
    assert set(after) == set(before) | {"host"}
    assert guard.records[-1] == dict(stage="target batch admission after cache release",
        allocated_bytes=22 * GB, driver_bytes=23 * GB, process_bytes=25 * GB,
        peak_memory_bytes=33 * GB, limit_bytes=40 * GB)
    assert guard._guard.current == 25 * GB
    assert guard._guard.peak == 33 * GB


@pytest.mark.parametrize("metric", ["allocated", "driver"])
@pytest.mark.parametrize("capacity", [1, 64, 129, 512])
def test_refreshed_device_boundary_keeps_margin_and_request_bound(metrics, cache, metric, capacity):
    guard = metrics.trace()
    cache.changes[metric] = 32 * GB - MARGIN - capacity * PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == capacity
    assert guard.batch_size(1, **OPTIONS) == 1
    cache.changes[metric] += 1
    if capacity == 1:
        with pytest.raises(_memory.CleanStop, match="no target batch fits"):
            guard.batch_size(512, **OPTIONS)
    else:
        assert guard.batch_size(512, **OPTIONS) == capacity - 1
    assert cache.calls == 3
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("itemsize,factor", [(2, 1), (4, 1), (8, 2)])
def test_cache_release_preserves_per_target_formula(metrics, cache, itemsize, factor):
    guard = metrics.trace()
    cache.changes["driver"] = 32 * GB - MARGIN - 4 * PER_TARGET
    assert guard.batch_size(512, **(OPTIONS | {"itemsize": itemsize})) == 4 // factor
    assert guard.batch_size(1, **(OPTIONS | {"itemsize": itemsize})) == 1


def test_cache_error_propagates_without_post_release_admission(metrics, monkeypatch):
    guard = metrics.trace()
    error = RuntimeError("device cache failure")

    def fail():
        raise error

    monkeypatch.setattr(_frozen.torch.mps, "empty_cache", fail)
    with pytest.raises(RuntimeError) as caught:
        guard.batch_size(512, **OPTIONS)
    assert caught.value is error
    assert [record["stage"] for record in guard.records] == ["trace start", "target batch admission"]
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("fault", ["current", "history", "lower_history", "supplied", "budget_history", "pressure"])
def test_prior_guard_refusal_precedes_cache_release(metrics, cache, fault):
    guard = metrics.trace()
    if fault == "current":
        metrics.current = 40 * GB + 1
        cache.changes["current"] = 0
    elif fault in ("history", "lower_history"):
        guard._guard.peak = 40 * GB + 1 if fault == "history" else 13 * GB
        if fault == "lower_history":
            guard.limit = 12 * GB
    elif fault == "pressure":
        guard._guard.pressure = lambda: 14
        guard._guard._pressure_time = -float("inf")
    else:
        def stop(stage):
            raise _memory.CleanStop("supplied stop")

        supplied = SimpleNamespace(limit=12 * GB, current=0, peak=0, check=stop)
        guard._guard = _memory.BudgetMemory(guard._guard, 40, extra=supplied)
        if fault == "budget_history":
            supplied.check = lambda stage: 0
            supplied.peak = 13 * GB
            with pytest.raises(_memory.CleanStop, match="memory limit"):
                guard.check("prior supplied breach")
            supplied.peak = 0
    with pytest.raises(_memory.CleanStop):
        guard.batch_size(512, **OPTIONS)
    assert cache.calls == 0


def test_post_release_breach_remains_latched(metrics, cache):
    guard = metrics.trace()
    cache.changes["peak"] = 40 * GB + 1
    with pytest.raises(_memory.CleanStop, match="memory limit"):
        guard.batch_size(512, **OPTIONS)
    assert cache.calls == 1
    metrics.peak = 0
    cache.changes.clear()
    with pytest.raises(_memory.CleanStop, match="memory limit"):
        guard.batch_size(512, **OPTIONS)
    assert cache.calls == 1
    assert guard._guard.peak == 40 * GB + 1


@pytest.mark.parametrize("owner", ["trace", "process", "supplied", "budget", "prior_allocator"])
def test_release_retains_lower_caps_and_history(metrics, cache, owner):
    guard = metrics.trace()
    supplied = SimpleNamespace(limit=40 * GB, current=0, peak=11 * GB, check=lambda stage: 0)
    guard._guard = _memory.BudgetMemory(guard._guard, 40, extra=supplied)
    if owner == "prior_allocator":
        metrics.trace(12)
    else:
        target = {"trace": guard, "process": guard._guard.memory,
                  "supplied": supplied, "budget": guard._guard}[owner]
        target.limit = 12 * GB
    metrics.driver = 9_600_000_000 - MARGIN
    cache.changes["driver"] = metrics.driver - PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == 1
    assert guard._guard.peak == supplied.peak == 11 * GB
    assert guard.limit == (40 if owner == "prior_allocator" else 12) * GB
    assert guard._guard.mps_allocator_limit == (9_600_000_000 if owner == "prior_allocator" else 32 * GB)
    assert metrics.fractions == ([.5, .15] if owner == "prior_allocator" else [.5])


@pytest.mark.parametrize("bound", ["process", "host", "borrowed"])
def test_released_device_space_keeps_fresh_process_and_host_bounds(metrics, cache, bound):
    supplied = SimpleNamespace(limit=40 * GB, peak=0, check=lambda stage: metrics.borrowed)
    metrics.borrowed = 0
    guard = _frozen._CappedMemoryGuard("mps", memory_limit_gb=40, memory=supplied, host_reserve_bytes=8 * GB)
    metrics.driver = 29 * GB
    cache.changes["driver"] = 23 * GB
    metric = {"process": "current", "host": "host", "borrowed": "borrowed"}[bound]
    cache.changes[metric] = (8 * GB + MARGIN + PER_TARGET if bound == "host"
                             else 40 * GB - MARGIN - PER_TARGET)
    assert guard.batch_size(512, **OPTIONS) == 1
    cache.changes[metric] += -1 if bound == "host" else 1
    with pytest.raises(_memory.CleanStop, match="no target batch fits"):
        guard.batch_size(512, **OPTIONS)
    assert cache.calls == 2


def test_reserves_and_interval_checks_do_not_release_cache(metrics, cache):
    guard = metrics.trace()
    guard.reserve("decoder workspace", 4096)
    guard._check_interval("decoder block")
    guard.check("decoder complete")
    assert cache.calls == 0


def test_service_owner_rechecks_after_each_cache_release(metrics, monkeypatch):
    from caliper.interp._service import _ServiceTraceGuard

    owner = metrics.trace()
    supplied = SimpleNamespace(limit=40 * GB, peak=0, current=0, check=lambda stage: 0)
    guard = _ServiceTraceGuard(owner, supplied)
    releases = []

    def release():
        # The service admits through its combined budget and then its owner.
        releases.append(None)
        metrics.driver = 32 * GB - MARGIN - (3 if len(releases) == 1 else 1) * PER_TARGET

    monkeypatch.setattr(_frozen.torch.mps, "empty_cache", release)
    assert guard.batch_size(512, **OPTIONS) == 1
    assert len(releases) == 2
    assert owner.records[-1]["driver_bytes"] == 32 * GB - MARGIN - PER_TARGET
    assert metrics.fractions == [.5]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_other_devices_keep_original_admission_without_cache_calls(metrics, cache, device, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary CPU or CUDA admission called MPS")

    for name in vars(_frozen.torch.mps):
        monkeypatch.setattr(_frozen.torch.mps, name, forbidden)
    guard = _frozen.MemoryGuard(device)
    metrics.current = 12 * GB - MARGIN - 2 * PER_TARGET
    assert guard.batch_size(512, **OPTIONS) == 2
    assert guard.batch_size(512, **(OPTIONS | {"itemsize": 8})) == 1
    assert [record["stage"] for record in guard.records] == [
        "trace start", "target batch admission", "target batch admission"]
    metrics.current += 2 * PER_TARGET
    with pytest.raises(_memory.CleanStop, match="^no target batch fits the process memory limit$"):
        guard.batch_size(1, **OPTIONS)
    assert cache.calls == 0
    assert metrics.fractions == []
