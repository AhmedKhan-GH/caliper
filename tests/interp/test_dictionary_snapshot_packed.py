"""Exact word comparisons through CPU seams for the bounded MPS snapshot path."""
import pytest
import torch

from caliper.interp import _intervene
from .test_dictionary_snapshot import device_snapshot
from .test_intervene import controlled


@pytest.fixture
def packed_snapshot(device_snapshot, monkeypatch):
    case = device_snapshot()
    label = type(next(iter(case.dictionary.state_dict().values())))

    def prepare(state, *, saved=None, block_bytes=16, batch_blocks=2):
        state = {name: tensor.as_subclass(label) for name, tensor in state.items()}
        monkeypatch.setattr(case.dictionary, "state_dict", lambda: state)
        monkeypatch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", block_bytes)
        monkeypatch.setattr(_intervene, "_SNAPSHOT_BATCH_BLOCKS", batch_blocks)
        case.executor.dictionary_id = _intervene._dictionary_identifier(case.dictionary, case.guard)
        case.executor._prepare_dictionary_identity()
        snapshot = case.executor._dictionary_snapshot
        assert snapshot is not None
        if saved is not None:
            snapshot.state = {name: tensor.as_subclass(label) for name, tensor in saved.items()}
        case.comparisons.clear()
        case.reductions.clear()
        case.operands.clear()
        case.word_views.clear()
        case.allocations.clear()
        case.transfers.clear()
        case.drains.clear()
        return case, snapshot

    return prepare


@pytest.mark.parametrize("length,left_offset,right_offset,operands", [
    (0, 0, 0, []), (0, 1, 3, []),
    (1, 0, 0, [(torch.uint8, 1)]),
    (2, 0, 0, [(torch.uint8, 2)]),
    (3, 0, 0, [(torch.uint8, 3)]),
    (4, 0, 0, [(torch.int32, 1)]),
    (8, 4, 8, [(torch.int32, 2)]),
    (8, 1, 0, [(torch.uint8, 8)]),
    (8, 0, 1, [(torch.uint8, 8)]),
    (8, 2, 0, [(torch.uint8, 8)]),
    (8, 0, 3, [(torch.uint8, 8)]),
    (8, 3, 2, [(torch.uint8, 8)]),
    (5, 0, 0, [(torch.uint8, 5)]),
    (6, 4, 0, [(torch.uint8, 6)]),
    (7, 0, 0, [(torch.uint8, 7)]),
    (17, 4, 8, [(torch.int32, 4), (torch.uint8, 1)]),
    (33, 0, 0, [(torch.int32, 4), (torch.int32, 4), (torch.uint8, 1)]),
])
def test_packed_snapshot_preserves_offsets_and_complete_tails(
        packed_snapshot, length, left_offset, right_offset, operands):
    left_storage = torch.arange(64, dtype=torch.uint8)
    right_storage = torch.zeros_like(left_storage)
    left = left_storage[left_offset:left_offset + length]
    right = right_storage[right_offset:right_offset + length]
    right.copy_(left)
    case, snapshot = packed_snapshot({"weight": left}, saved={"weight": right})
    identifier = case.executor.dictionary_id
    assert snapshot.matches(case.dictionary, identifier, case.guard)
    assert case.operands == operands
    assert sum(size for size, _ in case.comparisons) == length
    assert [size for size, _ in case.reductions] == [size for _, size in operands]
    assert sum(size for size, _ in case.transfers) == len(operands)
    assert all(size <= 2 for size, _ in case.transfers)
    assert not case.scopes and not case.drains
    assert all(reference() is None for reference in case.word_views)
    # Bytes outside each tensor view must not affect its content identity.
    left_storage[-1] ^= 128
    right_storage[-1] ^= 128
    assert snapshot.matches(case.dictionary, identifier, case.guard)
    if length:
        left[-1] ^= 128
        assert not snapshot.matches(case.dictionary, identifier, case.guard)
    else:
        assert not case.allocations and not case.comparisons


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("byte", range(16))
def test_packed_snapshot_checks_each_byte_of_signed_words_and_nan_payloads(packed_snapshot, dtype, byte):
    bits = torch.tensor([0, -2147483648, 0x7FC00001, -1], dtype=torch.int32)
    case, snapshot = packed_snapshot({"weight": bits.view(dtype)})
    identifier = case.executor.dictionary_id
    assert snapshot.matches(case.dictionary, identifier, case.guard)
    assert case.operands == [(torch.int32, 4)]
    raw = bits.view(torch.uint8)
    raw[byte] ^= 128
    assert not snapshot.matches(case.dictionary, identifier, case.guard)
    raw[byte] ^= 128
    assert snapshot.matches(case.dictionary, identifier, case.guard)
    assert all(reference() is None for reference in case.word_views)


def test_packed_snapshot_keeps_all_mixed_blocks_after_an_early_mismatch(packed_snapshot):
    left = torch.arange(69, dtype=torch.uint8)
    case, snapshot = packed_snapshot({"empty": left[:0], "weight": left})
    left[0] ^= 128
    assert not snapshot.matches(case.dictionary, case.executor.dictionary_id, case.guard)
    assert case.operands == [(torch.int32, 4)] * 4 + [(torch.uint8, 5)]
    assert [size for size, _ in case.comparisons] == [16, 16, 16, 16, 5]
    assert [size for size, _ in case.transfers] == [2, 2, 1]
    assert not case.scopes and not case.drains
    assert all(reference() is None for reference in case.allocations)


def test_packed_snapshot_keeps_odd_block_boundaries_without_padding(packed_snapshot):
    left = torch.arange(40, dtype=torch.uint8)
    case, snapshot = packed_snapshot({"weight": left}, block_bytes=17)
    assert snapshot.matches(case.dictionary, case.executor.dictionary_id, case.guard)
    assert case.operands == [(torch.uint8, 17), (torch.uint8, 17), (torch.uint8, 6)]
    left[-1] ^= 128
    assert not snapshot.matches(case.dictionary, case.executor.dictionary_id, case.guard)
