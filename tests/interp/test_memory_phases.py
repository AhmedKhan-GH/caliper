"""Current allocation, retained copies, and historical peak admission on CPU."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _intervene
from caliper.interp._memory import CleanStop, MemoryGuard, allocation_bytes
from .fixture import TinyLM
from .test_dictionary import write_fixture
from .test_intervene import dictionary_for


def process_guard(limit, readings):
    return MemoryGuard(limit / 1e9, "cpu", pressure=lambda: 50,
                       resident=lambda: readings["current"], footprint=lambda: readings["current"],
                       peak=lambda: readings["peak"], driver=lambda: 0)


def test_released_peak_does_not_reserve_live_bytes_but_breach_stays_latched():
    readings = {"current": 10_000, "peak": 900_000}
    guard = process_guard(1_000_000, readings)
    assert _dictionary._admit(guard, 500_000, "next phase") == 10_000
    assert guard.peak == 900_000
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _dictionary._admit(guard, 500_000, "pending pages", floor=500_001)
    readings["current"] = 500_001
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _dictionary._admit(guard, 500_000, "retained bytes")
    readings.update(current=10_000, peak=1_000_001)
    with pytest.raises(CleanStop, match="memory limit"):
        _dictionary._admit(guard, 1, "historical breach")
    readings["peak"] = 0
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("breach remains latched")


def test_stage_only_adapter_uses_latest_current_and_preserves_lower_limits():
    readings = {"current": 100, "peak": 900}
    inner = process_guard(1_000, readings)
    outer = SimpleNamespace(_guard=inner, limit=950, check=lambda stage: inner.check(stage))
    memory = _intervene._IdentityMemory(outer)
    memory.check("phase")
    assert allocation_bytes(memory) == 100 and memory.peak == 900 and memory.limit == 950
    readings["current"] = 50
    memory.check("released")
    assert allocation_bytes(memory) == 50
    outer.limit = 899
    with pytest.raises(CleanStop, match="historical allocation"):
        allocation_bytes(memory)


@pytest.mark.parametrize("fault", [None, "current", "historical", "pending"])
def test_dictionary_load_keeps_full_reserve_across_layers(tmp_path, monkeypatch, fault):
    model, metadata = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    readings = {"current": 10_000, "peak": 999_999}
    guards = []
    def factory(limit, device):
        guard = process_guard(int(limit * 1e9), readings)
        guards.append(guard)
        return guard
    monkeypatch.setattr(_dictionary, "MemoryGuard", factory)
    copies = []
    convert = torch.Tensor.to
    def copied(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        if fault == "pending":
            # The next layer must retain the first destination despite a stale reader.
            guards[-1].limit = 10_000 + _dictionary._metadata_budget(
                (tmp_path / "dictionary.json").stat().st_size) + 1120 + 5312
        return convert(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", copied)
    if fault == "current":
        readings["current"] = 999_999
    if fault == "historical":
        readings["peak"] = 1_000_001
    if fault == "pending":
        readings["peak"] = readings["current"]
    if fault:
        with pytest.raises(CleanStop, match="memory limit"):
            _dictionary.load_dictionary(tmp_path, model_hash=metadata["model_hash"],
                                        dtype=torch.float32, memory_limit_gb=.001)
        assert len(copies) == (4 if fault == "pending" else 0)
    else:
        dictionary = _dictionary.load_dictionary(tmp_path, model_hash=metadata["model_hash"],
                                                 dtype=torch.float32, memory_limit_gb=.001)
        assert len(dictionary.layers) == 2 and len(copies) == 8
        assert all(guard.peak == 999_999 for guard in guards)


def test_snapshot_uses_current_bytes_and_full_retained_floor(monkeypatch):
    model = TinyLM(layers=2, width=8, heads=2, latents=16, k=2)
    dictionary = dictionary_for(model)
    readings = {"current": 1_000, "peak": 9_999_999}
    guard = process_guard(10_000_000, readings)
    identifier = _intervene._dictionary_identifier(dictionary, guard)
    total = sum(t.numel() * t.element_size() for t in dictionary.state_dict().values())
    snapshot = _intervene._DictionarySnapshot.create(dictionary, identifier, guard)
    assert snapshot is not None and snapshot.matches(dictionary, identifier, guard)
    readings["current"] += total
    assert snapshot.matches(dictionary, identifier, guard)
    dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
    assert not snapshot.matches(dictionary, identifier, guard)
    # A valid high-water value alone must not force full-hash fallback.
    assert guard.peak == 9_999_999
    readings["current"] = 9_999_999
    monkeypatch.setattr(torch.Tensor, "clone", lambda *a, **kw: pytest.fail("unadmitted snapshot copy"))
    assert _intervene._DictionarySnapshot.create(dictionary, identifier, guard) is None
    readings.update(current=1_000, peak=10_000_001)
    with pytest.raises(CleanStop, match="memory limit"):
        _intervene._DictionarySnapshot.create(dictionary, identifier, guard)


def test_snapshot_pending_copy_floor_survives_a_stale_reader(monkeypatch):
    dictionary = dictionary_for(TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    readings = {"current": 1_000, "peak": 1_000}
    guard = process_guard(10_000_000, readings)
    identifier = _intervene._dictionary_identifier(dictionary, guard)
    total = sum(t.numel() * t.element_size() for t in dictionary.state_dict().values())
    clone, copies = torch.Tensor.clone, []
    def copied(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        guard.limit = readings["current"] + total + 4 * _dictionary._HASH_CHUNK_BYTES - 1
        return clone(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", copied)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _intervene._DictionarySnapshot.create(dictionary, identifier, guard)
    assert len(copies) == 1
