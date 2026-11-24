"""Pressure sampling and forced allocation checks with an injected clock."""
import pytest

from caliper.interp._memory import CleanStop, MemoryGuard


def test_forced_checks_cache_pressure_but_refresh_allocation():
    now = [0]
    calls = []
    reads = []
    footprint = [0]

    def pressure():
        calls.append(now[0])
        return 50

    def process():
        reads.append(now[0])
        return footprint[0]

    guard = MemoryGuard(1, "cpu", clock=lambda: now[0], pressure=pressure,
                        resident=lambda: 0, footprint=process, peak=lambda: 0, driver=lambda: 0)
    for tick in (0, 1, 2, 14):
        now[0] = tick
        guard.check("batch")
    assert calls == [0]
    assert reads == [0, 1, 2, 14]
    now[0] = 15
    guard.check("next sample")
    assert calls == [0, 15]
    footprint[0] = 1_000_000_001
    now[0] = 16
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("allocation")
    assert calls == [0, 15]
    assert reads[-1] == 16


def test_pressure_stop_uses_fresh_low_samples():
    samples = iter([19, 21, 19, 19])
    now = [0]
    calls = []

    def pressure():
        calls.append(now[0])
        return next(samples)

    guard = MemoryGuard(1, "cpu", clock=lambda: now[0], pressure=pressure,
                        resident=lambda: 0, footprint=lambda: 0, peak=lambda: 0, driver=lambda: 0)
    guard.check("first low")
    assert guard.low == 1
    now[0] = 1
    guard.check("recovery")
    assert guard.low == 0
    now[0] = 2
    guard.check("cached recovery")
    assert calls == [0, 1] and guard.low == 0
    now[0] = 16
    guard.check("new low")
    assert guard.low == 1
    now[0] = 17
    with pytest.raises(CleanStop, match="pressure"):
        guard.check("second low")
    assert calls == [0, 1, 16, 17] and guard.low == 2


def test_critical_pressure_stops_at_next_due_sample():
    values = iter([50, 14])
    now = [0]
    guard = MemoryGuard(1, "cpu", clock=lambda: now[0], pressure=lambda: next(values),
                        resident=lambda: 0, footprint=lambda: 0, peak=lambda: 0, driver=lambda: 0)
    guard.check("start")
    now[0] = 15
    with pytest.raises(CleanStop, match="pressure"):
        guard.check("critical")
    assert guard.low == 1


def test_memory_diagnostics_preserve_stdout_results_and_stop_limits(capsys):
    import json
    footprint = [100]
    guard = MemoryGuard(1, "cpu", clock=lambda: 0, pressure=lambda: 50,
                        resident=lambda: 100, footprint=lambda: footprint[0], peak=lambda: 100)
    guard.check("ready")
    print(json.dumps({"result": "ready"}))
    footprint[0] = 1_000_000_001
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("over limit")
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": "ready"}
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert [record["stage"] for record in records] == ["ready", "over limit"]
    assert records[-1]["peak_memory_bytes"] == 1_000_000_001


def test_trace_diagnostics_use_stderr_with_cpu_accounting_stubs(monkeypatch, capsys):
    import json
    from types import SimpleNamespace
    from caliper.interp import _frozen

    def process_guard(limit, device, **kwargs):
        return MemoryGuard(limit, device, pressure=lambda: 50, resident=lambda: 100,
                           footprint=lambda: 100, peak=lambda: 100, clock=lambda: 0)

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", process_guard)
    guard = _frozen.MemoryGuard("cpu")
    # Exercise the device diagnostic format through local scalar stubs only.
    guard.device = SimpleNamespace(type="mps")
    monkeypatch.setattr(_frozen, "torch", SimpleNamespace(mps=SimpleNamespace(
        current_allocated_memory=lambda: 7, driver_allocated_memory=lambda: 11)))
    guard.check("device record")
    captured = capsys.readouterr()
    assert captured.out == ""
    prefix, record = captured.err.splitlines()[-1].split(" ", 1)
    assert prefix == "caliper.trace.memory"
    assert json.loads(record)["driver_bytes"] == 11
    assert json.loads(record)["stage"] == "device record"
