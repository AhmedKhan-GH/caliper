"""CPU admission fixtures for bounded snapshot queues under host pressure."""
from types import SimpleNamespace

import psutil
import pytest
import torch

from caliper.interp import _dictionary, _intervene
from caliper.interp._memory import CleanStop
from .test_dictionary_snapshot import device_snapshot
from .test_intervene import controlled


@pytest.fixture
def headroom_snapshot(device_snapshot, monkeypatch):
    case = device_snapshot()
    label = type(next(iter(case.dictionary.state_dict().values())))
    monkeypatch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", 16)
    monkeypatch.setattr(_intervene, "_SNAPSHOT_BATCH_BLOCKS", 64)
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 16)
    case.guard.limit = 40_000_000_000
    case.guard.current = case.guard.peak = 1_000_000
    case.available = 64_000_000_000
    case.batches = []
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=case.available))
    compare = _intervene._DictionarySnapshot._device_equal

    def compared(snapshot, blocks, memory):
        case.batches.append([left.numel() for left, _ in blocks])
        return compare(snapshot, blocks, memory)

    monkeypatch.setattr(_intervene._DictionarySnapshot, "_device_equal", compared)

    def prepare(state=None):
        state = {"weight": torch.arange(64 * 16 + 13, dtype=torch.uint8)} if state is None else state
        case.state = {name: tensor.as_subclass(label) for name, tensor in state.items()}
        case.total = sum(t.numel() * t.element_size() for t in case.state.values())
        monkeypatch.setattr(case.dictionary, "state_dict", lambda: case.state)
        case.executor.dictionary_id = _intervene._dictionary_identifier(case.dictionary, case.guard)
        case.transfers.clear()
        return case

    return prepare


@pytest.mark.parametrize("capacity", [64, 3, 1])
def test_snapshot_creation_adapts_queue_to_host_headroom(headroom_snapshot, capacity):
    case = headroom_snapshot()
    case.available = 8_000_000_000 + case.total + _intervene._snapshot_workspace(16, capacity)
    case.executor._prepare_dictionary_identity()
    snapshot = case.executor._dictionary_snapshot
    assert snapshot is not None
    assert sum(size for size, _ in case.copies) == case.total
    assert sum(map(sum, case.batches)) == case.total
    assert max(map(len, case.batches)) == capacity
    assert case.drains == list(range(capacity, 65, capacity)) + ([65] if 65 % capacity else [])
    assert not case.scopes


@pytest.mark.parametrize("capacity", [64, 3, 1])
def test_snapshot_comparison_adapts_queue_to_host_headroom(headroom_snapshot, capacity):
    case = headroom_snapshot()
    case.executor._prepare_dictionary_identity()
    case.batches.clear()
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, capacity)
    snapshot = case.executor._dictionary_snapshot
    assert snapshot.matches(case.dictionary, case.executor.dictionary_id, case.guard)
    assert sum(map(sum, case.batches)) == case.total
    assert max(map(len, case.batches)) == capacity
    assert len(case.batches[-1]) == 65 % capacity or len(case.batches[-1]) == capacity


@pytest.mark.parametrize("phase", ["create", "compare"])
def test_snapshot_refuses_when_one_block_cannot_fit(headroom_snapshot, phase):
    case = headroom_snapshot()
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
    case.allocations.clear()
    case.copies.clear()
    case.comparisons.clear()
    pending = case.total if phase == "create" else 0
    case.available = 8_000_000_000 + pending + _intervene._snapshot_workspace(16, 1) - 1
    with pytest.raises(CleanStop, match="host reserve"):
        if phase == "create":
            case.executor._prepare_dictionary_identity()
        else:
            case.executor._dictionary_matches()
    assert not case.allocations and not case.copies and not case.comparisons and not case.scopes
