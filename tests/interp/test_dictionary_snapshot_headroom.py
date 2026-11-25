"""CPU admission fixtures for bounded snapshot queues under host pressure."""
from types import SimpleNamespace
import weakref

import psutil
import pytest
import torch

from caliper.interp import _dictionary, _intervene
from caliper.interp._memory import CleanStop
from .test_dictionary_snapshot import device_snapshot
from .test_intervene import controlled
from .test_intervention_dictionary_identity import feature_spec
from .test_intervention_identity import IdentityGuard


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
    assert case.drains == list(range(capacity, 65, capacity)) + [65]
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


@pytest.mark.parametrize("bound", ["host", "process"])
@pytest.mark.parametrize("capacity", [64, 7, 1])
def test_snapshot_admission_uses_production_block_costs(headroom_snapshot, bound, capacity):
    case = headroom_snapshot()
    block = 16 * 1024 * 1024
    workspace = capacity * (4 * block + 64 * 1024)
    retained = 6_000_000_000
    if bound == "host":
        case.available = 8_000_000_000 + retained + workspace
    else:
        case.guard.limit = case.guard.current + retained + workspace
    memory = _intervene._DeviceSnapshotMemory(case.guard)
    count, reserve = _intervene._DictionarySnapshot._admit_batch(
        memory, [block] * 65, retained=retained)
    assert (count, reserve) == (capacity, workspace)
    assert memory.reserve == retained + workspace
    assert memory.limit <= 40_000_000_000


@pytest.mark.parametrize("phase", ["create", "compare"])
@pytest.mark.parametrize("owner", ["outer", "inner", "decimal_cap"])
def test_snapshot_queue_obeys_the_lowest_process_budget(headroom_snapshot, phase, owner):
    case = headroom_snapshot()
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
    case.batches.clear()
    pending = case.total if phase == "create" else 0
    workspace = _intervene._snapshot_workspace(16, 3)
    if owner == "decimal_cap":
        case.guard.limit = 50_000_000_000
        case.guard.current = case.guard.peak = 40_000_000_000 - pending - workspace
    else:
        selected = case.guard
        if owner == "inner":
            selected = case.guard._guard = IdentityGuard(limit=40_000_000_000)
        selected.limit = case.guard.current + pending + workspace
    if phase == "create":
        case.executor._prepare_dictionary_identity()
    else:
        assert case.executor._dictionary_matches()
    assert case.executor._dictionary_snapshot is not None
    assert sum(map(sum, case.batches)) == case.total
    assert max(map(len, case.batches)) == 3


@pytest.mark.parametrize("phase", ["create", "compare"])
def test_snapshot_minimum_process_refusal_preserves_hash_fallback(headroom_snapshot, phase):
    case = headroom_snapshot()
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
    case.allocations.clear()
    case.comparisons.clear()
    case.copies.clear()
    pending = case.total if phase == "create" else 0
    case.guard.limit = case.guard.current + pending + _intervene._snapshot_workspace(16, 1) - 1
    if phase == "create":
        case.executor._prepare_dictionary_identity()
        assert case.executor._dictionary_snapshot is None
        assert case.executor._dictionary_matches()
    else:
        with pytest.raises(CleanStop, match="process memory limit"):
            case.executor._dictionary_matches()
    assert not case.allocations and not case.comparisons and not case.copies


def test_snapshot_copy_keeps_pending_and_untouched_pages_reserved(headroom_snapshot, monkeypatch):
    case = headroom_snapshot({"first": torch.arange(128, dtype=torch.uint8),
                              "second": torch.arange(128, dtype=torch.uint8)})
    monkeypatch.setattr(_intervene, "_SNAPSHOT_FLAG_RESERVE", 64)
    case.available = 8_000_000_000 + case.total + _intervene._snapshot_workspace(16, 2)
    # The reader never observes destination pages. Both tensors still consume
    # the full retained reserve before allocation and after each copy.
    case.executor._prepare_dictionary_identity()
    assert case.guard.current == 1_000_000
    assert case.drains == list(range(2, 17, 2))
    assert [len(batch) for batch in case.batches] == [2] * 8
    assert sum(map(sum, case.batches)) == case.total


@pytest.mark.parametrize("reading", ["outer", "inner", "returned"])
def test_snapshot_headroom_subtracts_unobserved_floor_once(headroom_snapshot, reading, monkeypatch):
    case = headroom_snapshot()
    observed, pending = case.guard.current, 123_456
    if reading == "inner":
        case.guard._guard = IdentityGuard(limit=40_000_000_000)
        case.guard._guard.current = observed
        case.guard.current = 0
    elif reading == "returned":
        case.guard.current = 0
        monkeypatch.setattr(case.guard, "check", lambda stage: observed)
    memory = _intervene._DeviceSnapshotMemory(case.guard)
    memory.floor = observed + pending // 2
    workspace = _intervene._snapshot_workspace(16, 3)
    case.available = 8_000_000_000 + pending + workspace
    count, reserve = _intervene._DictionarySnapshot._admit_batch(
        memory, [16] * 64, floor=observed + pending)
    assert (count, reserve) == (3, workspace)
    assert memory.headroom == workspace


@pytest.mark.parametrize("phase", ["create", "compare"])
def test_snapshot_replans_after_headroom_falls_between_batches(headroom_snapshot, monkeypatch, phase):
    case = headroom_snapshot()
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
    case.batches.clear()
    case.drains.clear()
    pending = case.total if phase == "create" else 0
    case.available = 8_000_000_000 + pending + _intervene._snapshot_workspace(16, 4)

    def lower_headroom():
        case.available = 8_000_000_000 + pending + _intervene._snapshot_workspace(16, 2)

    if phase == "create":
        drain = torch.mps.synchronize

        def drained():
            drain()
            lower_headroom()

        monkeypatch.setattr(torch.mps, "synchronize", drained)
        case.executor._prepare_dictionary_identity()
        assert case.drains[:3] == [4, 6, 8]
        assert case.drains[-1] == 65
        assert max(map(len, case.batches)) == 2
    else:
        compare = _intervene._DictionarySnapshot._device_equal

        def compared(snapshot, blocks, memory):
            equal = compare(snapshot, blocks, memory)
            lower_headroom()
            return equal

        monkeypatch.setattr(_intervene._DictionarySnapshot, "_device_equal", compared)
        assert case.executor._dictionary_matches()
        assert [len(batch) for batch in case.batches[:3]] == [4, 2, 2]
    assert sum(map(sum, case.batches)) == case.total
    assert not case.scopes


@pytest.mark.parametrize("phase", ["create", "compare"])
def test_snapshot_stops_and_cleans_up_when_admitted_queue_loses_headroom(headroom_snapshot, monkeypatch, phase):
    case = headroom_snapshot()
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
    case.allocations.clear()
    case.copies.clear()
    case.comparisons.clear()
    case.drains.clear()
    work = case.copies if phase == "create" else case.comparisons
    pending = case.total if phase == "create" else 0
    case.available = 8_000_000_000 + pending + _intervene._snapshot_workspace(16, 4)
    check = case.guard.check

    def checked(stage):
        check(stage)
        if work:
            case.available = 8_000_000_000 + pending + _intervene._snapshot_workspace(16, 1)

    monkeypatch.setattr(case.guard, "check", checked)
    with pytest.raises(CleanStop, match="host reserve") as caught:
        if phase == "create":
            case.executor._prepare_dictionary_identity()
        else:
            case.executor._dictionary_matches()
    assert caught.value.__traceback__ is not None
    assert len(work) == 1 and len(case.drains) == 1 and not case.scopes
    assert all(reference() is None for reference in case.allocations)
    assert not case.executor._hooks
    if phase == "create":
        assert case.executor._dictionary_snapshot is None


def test_snapshot_admission_preserves_the_hosted_guard_latch(headroom_snapshot, monkeypatch):
    from caliper.interp._hosted import _RunGuard
    case = headroom_snapshot()
    guard = _RunGuard(case.guard)
    case.executor.resource_guard = guard
    case.available = 8_000_000_000 + case.total + _intervene._snapshot_workspace(16, 2)
    case.executor._prepare_dictionary_identity()
    assert guard.failure is None
    failure = CleanStop("owning guard stop")

    def stopped(stage):
        raise failure

    monkeypatch.setattr(case.guard, "check", stopped)
    for attempt in range(2):
        with pytest.raises(CleanStop) as caught:
            case.executor._dictionary_matches()
        assert caught.value is guard.failure is failure


@pytest.mark.parametrize("mutation", [False, True])
def test_reduced_snapshot_keeps_both_complete_action_scans(headroom_snapshot, mutation):
    case = headroom_snapshot()
    case.executor._prepare_dictionary_identity()
    case.batches.clear()
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, 3)
    admitted = case.executor.validate(feature_spec())
    assert sum(map(sum, case.batches)) == case.total
    case.batches.clear()
    if mutation:
        case.state["weight"][0] ^= 128
        with pytest.raises(_intervene.InterventionError, match="dictionary content changed"):
            case.executor.install(admitted)
        assert not case.executor._hooks
    else:
        hook = case.executor.install(admitted)
        hook()
    assert sum(map(sum, case.batches)) == case.total
    assert max(map(len, case.batches)) == 3


def test_reduced_snapshot_checks_mixed_alignment_and_tails(headroom_snapshot):
    raw = torch.arange(32, dtype=torch.uint8)
    words = torch.tensor([0, -2147483648, 0x7FC00001, -1], dtype=torch.int32)
    case = headroom_snapshot({"empty": raw[:0], "words": words.view(torch.float32),
                              "offset": raw[1:20], "tail": raw[:3]})
    case.executor._prepare_dictionary_identity()
    case.batches.clear()
    case.operands.clear()
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, 3)
    words[0] = -2147483648
    assert not case.executor._dictionary_matches()
    assert case.batches == [[16, 16, 3], [3]]
    assert case.operands == [(torch.int32, 4), (torch.uint8, 16), (torch.uint8, 3), (torch.uint8, 3)]
    assert sum(map(sum, case.batches)) == case.total


def test_snapshot_admits_the_actual_short_tail_cost(headroom_snapshot):
    case = headroom_snapshot({"first": torch.arange(3, dtype=torch.uint8),
                              "second": torch.arange(3, dtype=torch.uint8)})
    case.available = 8_000_000_000 + case.total + _intervene._snapshot_workspace(3, 2)
    case.executor._prepare_dictionary_identity()
    assert case.batches == [[3, 3]]
    case.batches.clear()
    case.available -= case.total
    assert case.executor._dictionary_matches()
    assert case.batches == [[3, 3]]


@pytest.mark.parametrize("margin", [-1, 0])
def test_snapshot_creation_retains_hash_reserve_but_comparison_needs_no_hash(headroom_snapshot, monkeypatch, margin):
    case = headroom_snapshot()
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 1024)
    monkeypatch.setattr(_intervene, "_SNAPSHOT_FLAG_RESERVE", 64)
    case.available = 8_000_000_000 + case.total + 4 * 1024 + margin
    if margin < 0:
        with pytest.raises(CleanStop, match="host reserve"):
            case.executor._prepare_dictionary_identity()
        assert not case.allocations
        return
    case.executor._prepare_dictionary_identity()
    assert max(map(len, case.batches)) == 32
    case.batches.clear()
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, 1)
    assert case.executor._dictionary_matches()
    assert len(case.batches) == 65 and sum(map(sum, case.batches)) == case.total
    saved = weakref.ref(next(iter(case.executor._dictionary_snapshot.state.values())))
    case.executor.close()
    assert saved() is None


@pytest.mark.parametrize("mismatch", ["device", "cpu"])
def test_device_snapshot_finishes_cpu_blocks_after_a_mismatch(headroom_snapshot, monkeypatch, mismatch):
    case = headroom_snapshot({"device": torch.arange(17, dtype=torch.uint8)})
    case.state.update(cpu=torch.arange(33, dtype=torch.uint8), last=torch.arange(7, dtype=torch.uint8))
    case.executor.dictionary_id = _intervene._dictionary_identifier(case.dictionary, case.guard)
    case.executor._prepare_dictionary_identity()
    case.batches.clear()
    case.state[mismatch][0] ^= 128
    sizes = []
    equal = torch.equal

    def compared(left, right):
        sizes.append(left.numel())
        return equal(left, right)

    monkeypatch.setattr(torch, "equal", compared)
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, 1)
    assert not case.executor._dictionary_matches()
    assert case.batches == [[16], [1]]
    assert sizes == [16, 16, 1, 7]


@pytest.fixture
def guarded_snapshot(headroom_snapshot, monkeypatch):
    """Use real process, trace, and owner guards with small CPU storage."""
    from caliper.interp import _frozen
    from caliper.interp._hosted import _RunGuard
    from caliper.interp._memory import MemoryGuard

    case = headroom_snapshot()
    case.now, case.process_bytes = 0, 1_000_000
    case.samples = []

    def resident():
        case.samples.append(len(case.comparisons))
        return case.process_bytes

    def process_guard(limit, device, **kwargs):
        return MemoryGuard(limit, device, clock=lambda: case.now, pressure=lambda: 50,
                           resident=resident, footprint=lambda: case.process_bytes,
                           peak=lambda: case.process_bytes, driver=lambda: 0)

    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", process_guard)
    case.trace_guard = _frozen.MemoryGuard("cpu", memory_limit_gb=40)
    case.owner = _RunGuard(case.guard, extra=case.trace_guard)
    case.executor.resource_guard = case.owner
    case.executor._prepare_dictionary_identity()
    case.samples.clear()
    case.batches.clear()
    case.allocations.clear()
    case.comparisons.clear()
    case.copies.clear()
    case.drains.clear()
    return case


@pytest.mark.parametrize("capacity", [64, 3])
@pytest.mark.parametrize("mismatch", [False, True])
def test_snapshot_comparison_reads_process_at_entry_and_each_completed_batch(guarded_snapshot, capacity, mismatch):
    case = guarded_snapshot
    case.available = 8_000_000_000 + _intervene._snapshot_workspace(16, capacity)
    if mismatch:
        case.state["weight"][0] ^= 128
    assert case.executor._dictionary_matches() is not mismatch
    completed = list(range(capacity, 65, capacity)) + [65]
    # The trace guard reads RSS through its process guard and its own record.
    assert case.samples == [count for count in [0, *completed] for _ in range(2)]
    assert sum(map(sum, case.batches)) == case.total
    assert max(map(len, case.batches)) == capacity
    assert not case.scopes


@pytest.mark.parametrize("change", ["host", "limit", "latch", "interval"])
def test_snapshot_comparison_admission_preserves_live_stops(guarded_snapshot, monkeypatch, change):
    case = guarded_snapshot
    compare = _intervene._DictionarySnapshot._device_equal
    failure = CleanStop("owner stop after readback")

    def compared(snapshot, blocks, memory):
        equal = compare(snapshot, blocks, memory)
        if change == "host":
            case.available = 8_000_000_000 + _intervene._snapshot_workspace(13, 1) - 1
        elif change == "limit":
            case.trace_guard.limit = case.process_bytes + _intervene._snapshot_workspace(13, 1) - 1
        elif change == "latch":
            case.owner.failure = failure
        else:
            case.now += 15
            case.process_bytes = 40_000_000_001
        return equal

    monkeypatch.setattr(_intervene._DictionarySnapshot, "_device_equal", compared)
    with pytest.raises(CleanStop) as caught:
        case.executor._dictionary_matches()
    assert len(case.comparisons) == 64
    assert all(reference() is None for reference in case.allocations)
    assert not case.scopes and not case.executor._hooks
    if change == "latch":
        assert caught.value is failure
    elif change == "interval":
        assert caught.value is case.owner.failure


def test_snapshot_completed_batch_forces_a_process_stop_within_the_interval(guarded_snapshot, monkeypatch):
    case = guarded_snapshot
    reduce = torch.all

    def reduced(*args, **kwargs):
        result = reduce(*args, **kwargs)
        case.process_bytes = 40_000_000_001
        return result

    monkeypatch.setattr(torch, "all", reduced)
    with pytest.raises(CleanStop) as caught:
        case.executor._dictionary_matches()
    assert caught.value is case.owner.failure
    assert case.now == 0 and len(case.comparisons) == 64
    assert all(reference() is None for reference in case.allocations)
    assert case.readback() is None and not case.scopes and not case.executor._hooks


def test_snapshot_copy_admission_keeps_forced_process_reads(guarded_snapshot, monkeypatch):
    case = guarded_snapshot
    drain = torch.mps.synchronize

    def drained():
        drain()
        case.process_bytes = 40_000_000_001

    monkeypatch.setattr(torch.mps, "synchronize", drained)
    with pytest.raises(CleanStop) as caught:
        case.executor._prepare_dictionary_identity()
    assert caught.value is case.owner.failure
    assert case.now == 0 and len(case.copies) == 64
    assert not case.comparisons and not case.scopes
    assert case.executor._dictionary_snapshot is None
    assert all(reference() is None for reference in case.allocations)
