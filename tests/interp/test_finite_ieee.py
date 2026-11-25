"""Finite classification against independent IEEE boundary bytes on the CPU."""
import struct

import pytest
import torch

from caliper.interp import _frozen


# Literal IEEE encodings cover zeros, subnormals, normal boundaries, adjacent
# values near one, finite extrema, infinities, and distinct NaN payloads.
FORMATS = [
    (torch.float16, "H", 0x8000, [
        0x0000, 0x0001, 0x03FF, 0x0400, 0x3BFF, 0x3C00, 0x3C01,
        0x7BFF, 0x7C00, 0x7C01, 0x7D00, 0x7E00, 0x7FFF]),
    (torch.bfloat16, "H", 0x8000, [
        0x0000, 0x0001, 0x007F, 0x0080, 0x3F7F, 0x3F80, 0x3F81,
        0x7F7F, 0x7F80, 0x7F81, 0x7FA0, 0x7FC0, 0x7FFF]),
    (torch.float32, "I", 0x80000000, [
        0x00000000, 0x00000001, 0x007FFFFF, 0x00800000,
        0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x7F7FFFFF,
        0x7F800000, 0x7F800001, 0x7FA00000, 0x7FC00000, 0x7FFFFFFF]),
]
DTYPES = [case[0] for case in FORMATS]


def ieee_values(case):
    dtype, code, sign, positive = case
    words = positive + [word | sign for word in positive]
    raw = bytearray(struct.pack("=" + code * len(words), *words))
    return torch.frombuffer(raw, dtype=dtype)


def require_matches_oracle(value):
    expected = bool(torch.isfinite(value).all())
    if expected:
        _frozen._require_finite(value, "IEEE fixture")
    else:
        with pytest.raises(ValueError, match="^IEEE fixture must be finite$"):
            _frozen._require_finite(value, "IEEE fixture")


@pytest.mark.parametrize("case", FORMATS, ids=[str(dtype) for dtype in DTYPES])
def test_each_signed_encoding_matches_torch_isfinite(case):
    value = ieee_values(case)
    before = value.view(torch.uint8).clone()
    for scalar in value.unbind():
        require_matches_oracle(scalar)
    assert torch.equal(value.view(torch.uint8), before)


@pytest.mark.parametrize("case", FORMATS, ids=[str(dtype) for dtype in DTYPES])
@pytest.mark.parametrize("layout", ["odd", "transpose", "stride", "expand", "overlap"])
def test_current_bytes_in_bounded_views_match_oracle(monkeypatch, case, layout):
    samples = ieee_values(case)
    finite = samples[torch.isfinite(samples)]
    storage = finite.repeat(8)[:105].reshape(3, 5, 7)
    value = {
        "odd": storage,
        "transpose": storage.permute(2, 0, 1),
        "stride": storage[1:, ::2, 1::2],
        "expand": storage[:1].expand(3, -1, -1),
        "overlap": storage.flatten().unfold(0, 7, 3),
    }[layout]
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 5 * 24)
    require_matches_oracle(value)
    # A live alias changes the final logical element after a successful scan.
    position = tuple(size - 1 for size in value.shape)
    original = value[position].clone()
    for bad in samples[~torch.isfinite(samples)].unbind():
        value[position] = bad
        before = storage.view(torch.uint8).clone()
        require_matches_oracle(value)
        assert torch.equal(storage.view(torch.uint8), before)
    value[position] = original
    require_matches_oracle(value)


@pytest.mark.parametrize("dtype", DTYPES + [torch.float64])
@pytest.mark.parametrize("shape", [(0,), (3, 0, 5)])
def test_empty_values_keep_all_checks(dtype, shape):
    stages = []
    value = torch.empty(shape, dtype=dtype)
    assert bool(torch.isfinite(value).all())
    _frozen._require_finite(value, "empty", stages.append)
    assert stages == ["finite block", "finite block complete", "finite block readback"]


def test_identical_storage_has_dtype_specific_finiteness():
    raw = bytearray(struct.pack("=4H", 0x7C00, 0xFC00, 0x7BFF, 0xFBFF))
    half = torch.frombuffer(raw, dtype=torch.float16)
    bfloat = torch.frombuffer(raw, dtype=torch.bfloat16)
    assert not bool(torch.isfinite(half).all())
    assert bool(torch.isfinite(bfloat).all())
    require_matches_oracle(half)
    require_matches_oracle(bfloat)


@pytest.mark.parametrize("dtype", DTYPES + [torch.float64])
def test_negative_views_and_unused_nonfinite_storage(dtype):
    storage = torch.tensor([float("nan"), 0., float("inf"), -0.,
                            -float("inf"), torch.finfo(dtype).max], dtype=dtype)
    alias = torch._neg_view(storage[1::2])
    assert alias.is_neg()
    require_matches_oracle(alias)
    storage[-1] = float("nan")
    require_matches_oracle(alias)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transpose", [False, True])
def test_cpu_tensor_workspace_is_smaller_without_larger_blocks(dtype, transpose):
    from torch.utils._python_dispatch import TorchDispatchMode

    value = torch.ones(7, 15, dtype=dtype)
    if transpose:
        value = value.T

    class Allocations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.storage = {value.untyped_storage().data_ptr()}
            self.outputs = []
            self.bytes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                storage = result.untyped_storage()
                if storage.data_ptr() not in self.storage:
                    self.storage.add(storage.data_ptr())
                    self.bytes.append(storage.nbytes())
                # Retain all outputs, including released intermediates. The sum
                # bounds tensor storage even if every operation overlaps.
                self.outputs.append(result)
            return result

    with Allocations() as baseline:
        assert bool(torch.isfinite(value).all())
    with Allocations() as candidate:
        _frozen._require_finite(value, "workspace")
    assert sum(candidate.bytes) < sum(baseline.bytes)
    assert sum(candidate.bytes) <= 24 * value.numel() + 3
    assert [size for size in candidate.bytes if size > 4] == [value.numel() * value.element_size()]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reason", ["outer", "inner", "extra", "current", "peak", "host", "latch", "cancel", "batch"])
def test_integer_scan_refuses_before_allocation(monkeypatch, dtype, reason):
    from types import SimpleNamespace
    from caliper.interp._memory import BudgetMemory, CleanStop
    from .test_finite_workspace import Guard, forbidden

    first, second = Guard(), Guard()
    supplied = BudgetMemory(first, 10_000 / 1e9, extra=second)
    failure = CleanStop("latched refusal")
    # Nine elements reserve 216 bytes plus three scalar bytes.
    if reason == "outer":
        supplied.limit = 318
    elif reason == "inner":
        first.limit = 318
    elif reason == "extra":
        second.limit = 318
    elif reason == "current":
        second.current = 9_782
    elif reason == "peak":
        second.peak = 10_001
    elif reason == "host":
        second.host_reserve_bytes = 8_000_000_000
        monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=8_000_000_218))
    elif reason == "latch":
        second.failure = failure
    elif reason == "batch":
        monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 24)
        first.limit = 204

    def check(stage):
        if reason == "cancel":
            raise failure

    monkeypatch.setattr(torch, "bitwise_and", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(CleanStop) as error:
        _frozen._require_finite(torch.ones(9, dtype=dtype), "guard", check, supplied)
    if reason in ("latch", "cancel"):
        assert error.value is failure


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reason", ["cap", "current", "host", "peak", "latch", "cancel", "deadline"])
def test_integer_scan_keeps_late_refusals(monkeypatch, dtype, reason):
    from types import SimpleNamespace
    from caliper.interp._memory import BudgetMemory, CleanStop
    from caliper.interp._trace import TraceTimeout
    from .test_finite_workspace import Guard, forbidden

    first, second = Guard(reserve=8_000_000_000), Guard()
    supplied = BudgetMemory(first, 10_000 / 1e9, extra=second)
    available = SimpleNamespace(available=8_000_010_000)
    monkeypatch.setattr("psutil.virtual_memory", lambda: available)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 24)
    calls = []
    bitwise_and = torch.bitwise_and
    failure = TraceTimeout("deadline") if reason == "deadline" else CleanStop("cancel or latch")

    def evaluate(block, exponent):
        calls.append(block.numel())
        assert len(calls) == 1
        return bitwise_and(block, exponent)

    def check(stage):
        if stage == "finite block complete":
            # Four one-element blocks reserve 96 bytes and nine scalar bytes.
            if reason == "cap":
                second.limit = 204
            elif reason == "current":
                second.current = 9_896
            elif reason == "host":
                available.available = 8_000_000_104
            elif reason == "peak":
                second.peak = 10_001
            elif reason == "latch":
                second.failure = failure
            else:
                raise failure

    monkeypatch.setattr(torch, "bitwise_and", evaluate)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    with pytest.raises((CleanStop, TraceTimeout)) as error:
        _frozen._require_finite(torch.ones(9, dtype=dtype), "guard", check, supplied)
    assert calls == [1]
    if reason in ("latch", "cancel", "deadline"):
        assert error.value is failure


@pytest.mark.parametrize("dtype", DTYPES)
def test_integer_scan_keeps_exact_process_and_host_boundary(monkeypatch, dtype):
    from types import SimpleNamespace
    from .test_finite_workspace import Guard

    guard = Guard(limit=319, current=100, peak=318, reserve=8_000_000_000)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=8_000_000_219))
    _frozen._require_finite(torch.ones(9, dtype=dtype), "boundary", memory=guard)
    assert guard.peak == 318 and guard.limit == 319
    assert guard.stages == ["finite block", "finite block complete", "finite block readback"]
