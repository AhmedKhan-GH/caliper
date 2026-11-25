"""CPU fixture isolation with unchanged pressure and allocation stop rules."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen, _memory
from caliper.interp._publish import _resolve
from .cpu_pressure import cpu_fixture_pressure
from .test_hosted import hosted_run, _trace_at_boundary


def test_hosted_fixture_completes_under_low_ambient_pressure(monkeypatch, request):
    samples = []

    def ambient_pressure():
        samples.append(18.)
        return 18.

    monkeypatch.setitem(_memory.MemoryGuard.__init__.__kwdefaults__, "pressure", ambient_pressure)
    monkeypatch.setattr(_frozen, "free_percent", ambient_pressure)
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))
    unisolated = _memory.MemoryGuard(12, "cpu")
    unisolated.check("first ambient sample")
    with pytest.raises(_memory.CleanStop, match="pressure"):
        unisolated.check("second ambient sample")
    assert samples == [18., 18.] and unisolated.low == 2

    run, _ = request.getfixturevalue("hosted_run")
    graph = _trace_at_boundary(run, prompt="AB", max_targets=16)
    assert any(node["kind"] == "feature" for node in graph.nodes)
    before = _resolve("output.probs").clone()
    run.update_output()
    torch.testing.assert_close(_resolve("output.probs"), before, rtol=0, atol=0)
    assert samples == [18., 18.]


@pytest.mark.parametrize("pressure,checks", [(18., 2), (14., 1)])
def test_fixture_isolation_preserves_explicit_pressure_stops(pressure, checks):
    with cpu_fixture_pressure():
        guard = _memory.MemoryGuard(12, "cpu", pressure=lambda: pressure)
        if checks == 2:
            guard.check("first explicit sample")
        with pytest.raises(_memory.CleanStop, match="pressure"):
            guard.check("explicit stop")
        assert guard.low == checks


def test_fixture_isolation_keeps_real_allocation_limits_and_restores_readers():
    default = _memory.MemoryGuard.__init__.__kwdefaults__["pressure"]
    trace_reader = _frozen.free_percent
    with cpu_fixture_pressure():
        guard = _memory.MemoryGuard(.000000001, "cpu")
        with pytest.raises(_memory.CleanStop, match="memory limit"):
            guard.check("one byte process limit")
        assert guard.peak > guard.limit == 1
    assert _memory.MemoryGuard.__init__.__kwdefaults__["pressure"] is default
    assert _frozen.free_percent is trace_reader


def test_fixture_isolation_preserves_explicit_trace_pressure(monkeypatch):
    with cpu_fixture_pressure(), monkeypatch.context() as patch:
        patch.setattr(_frozen, "free_percent", lambda: 18.)
        guard = _frozen.MemoryGuard("cpu")
        with pytest.raises(_memory.CleanStop, match="pressure"):
            guard.check("second trace sample")
        assert guard._guard.low == 2
