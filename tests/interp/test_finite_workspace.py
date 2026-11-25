"""Exact CPU coverage and prospective workspace admission for finite scans."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen
from caliper.interp._memory import BudgetMemory, CleanStop
from caliper.interp._trace import TraceTimeout
from .test_mps_allocator_admission import metrics


class Guard:
    def __init__(self, *, limit=10_000, current=100, peak=100, reserve=0):
        self.limit, self.current, self.peak = limit, current, peak
        self.host_reserve_bytes = reserve
        self.stages = []
        self.failure = None

    def check(self, stage):
        self.stages.append(stage)
        if self.failure is not None:
            raise self.failure
        return self.current


def forbidden(*args, **kwargs):
    pytest.fail("refusal must precede the next tensor operation")


def test_default_workspace_uses_two_reductions_and_one_readback(monkeypatch):
    value = torch.ones(2_097_153, dtype=torch.float32)
    finite, item = torch.isfinite, torch.Tensor.item
    sizes, readbacks = [], []

    def inspect(block):
        sizes.append(block.numel())
        assert block.numel() * (2 * max(4, block.element_size()) + 16) <= _frozen._FINITE_BLOCK_BYTES
        return finite(block)

    def readback(result):
        readbacks.append(result.shape)
        return item(result)

    monkeypatch.setattr(torch, "isfinite", inspect)
    monkeypatch.setattr(torch.Tensor, "item", readback)
    _frozen._require_finite(value, "fixture")
    assert sizes == [1_048_576, 1_048_577] and readbacks == [()]
    value[-1] = float("nan")
    with pytest.raises(ValueError, match="^fixture must be finite$"):
        _frozen._require_finite(value, "fixture")
    assert sizes == [1_048_576, 1_048_577] * 2 and readbacks == [(), ()]


@pytest.mark.parametrize("layout", ["odd", "transpose", "stride", "expand"])
def test_every_logical_value_reaches_a_bounded_view(monkeypatch, layout):
    original = torch.arange(7 * 11 * 5, dtype=torch.float64).reshape(7, 11, 5)
    value = {"odd": original, "transpose": original.permute(2, 0, 1),
             "stride": original[1::2, ::3, 1::2], "expand": original[:1].expand(3, -1, -1)}[layout]
    expected = sorted(value.reshape(-1).tolist())
    finite, item = torch.isfinite, torch.Tensor.item
    seen, readbacks = [], []
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 5 * 32)

    def inspect(block):
        assert block.numel() <= 5
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        seen.extend(block.reshape(-1).tolist())
        return finite(block)

    def readback(result):
        assert result.shape == () and result.dtype == torch.bool
        readbacks.append(result.shape)
        return item(result)

    monkeypatch.setattr(torch, "isfinite", inspect)
    monkeypatch.setattr(torch.Tensor, "item", readback)
    _frozen._require_finite(value, "fixture")
    assert sorted(seen) == expected
    assert len(readbacks) > 1
    assert torch.equal(original, torch.arange(original.numel(), dtype=original.dtype).reshape(original.shape))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", [0, 17, 34])
def test_bad_values_in_first_later_and_tail_chunks(monkeypatch, dtype, bad, position):
    value = torch.ones(70, dtype=dtype)[::2]
    value[position] = bad
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 3 * (2 * max(4, value.element_size()) + 16))
    with pytest.raises(ValueError, match="^fixture must be finite$"):
        _frozen._require_finite(value, "fixture")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_finite_values_and_unused_storage(monkeypatch, dtype):
    limits = torch.finfo(dtype)
    storage = torch.full((12,), float("nan"), dtype=dtype)
    value = storage[::2]
    value[:] = torch.tensor([limits.max, limits.min, limits.tiny, -limits.tiny, 0., -0.], dtype=dtype)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 2 * max(4, value.element_size()) + 16)
    _frozen._require_finite(value, "fixture")
    assert torch.isnan(storage[1::2]).all()


@pytest.mark.parametrize("shape", [(), (0,), (3, 0, 5)])
def test_scalar_and_empty_inputs_keep_checks(monkeypatch, shape):
    value = torch.ones(shape, dtype=torch.float64)
    stages = []
    guard = Guard(limit=1_000)
    _frozen._require_finite(value, "fixture", stages.append, guard)
    assert stages == guard.stages == ["finite block", "finite block complete", "finite block readback"]


@pytest.mark.parametrize("kind", ["outer", "inner", "extra", "current", "peak"])
def test_supplied_limits_refuse_before_masks(monkeypatch, kind):
    first, second = Guard(), Guard()
    supplied = BudgetMemory(first, 10_000 / 1e9, extra=second)
    # Nine float64 elements need 288 workspace bytes and three scalar bytes.
    if kind == "outer":
        supplied.limit = 390
    elif kind == "inner":
        first.limit = 390
    elif kind == "extra":
        second.limit = 390
    elif kind == "current":
        second.current = 9_710
    else:
        second.peak = 10_001
    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(CleanStop, match="memory limit"):
        _frozen._require_finite(torch.ones(9, dtype=torch.float64), "fixture", memory=supplied)
    assert first.stages == second.stages == ["finite block"]


def test_exact_workspace_boundary_and_current_below_prior_peak():
    value = torch.ones(9, dtype=torch.float64)
    guard = Guard(limit=391, current=100, peak=388)
    _frozen._require_finite(value, "fixture", memory=guard)
    assert guard.limit == 391 and guard.peak == 388


def test_whole_batch_workspace_precedes_first_mask(monkeypatch):
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 32)
    # Four one-element blocks need 128 workspace bytes and nine scalar bytes.
    guard = Guard(limit=236)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(CleanStop, match="finite validation allocation"):
        _frozen._require_finite(torch.ones(8, dtype=torch.float64), "fixture", memory=guard)


@pytest.mark.parametrize("late", ["cap", "current", "host", "latch"])
def test_late_resource_changes_stop_before_next_mask(monkeypatch, late):
    guard = Guard(reserve=8_000_000_000)
    extra = Guard()
    supplied = BudgetMemory(guard, 10_000 / 1e9, extra=extra)
    available = SimpleNamespace(available=8_000_010_000)
    monkeypatch.setattr("psutil.virtual_memory", lambda: available)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 32)
    finite = torch.isfinite
    calls = []
    failure = CleanStop("supplied pressure latch")

    def evaluate(block):
        calls.append(block.numel())
        assert len(calls) == 1
        return finite(block)

    def check(stage):
        if stage == "finite block complete":
            if late == "cap":
                extra.limit = 236
            elif late == "current":
                extra.current = 10_001
            elif late == "host":
                available.available = 8_000_000_136
            else:
                extra.failure = failure

    monkeypatch.setattr(torch, "isfinite", evaluate)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    with pytest.raises(CleanStop) as error:
        _frozen._require_finite(torch.ones(8, dtype=torch.float64), "fixture", check, supplied)
    assert calls == [1]
    if late == "latch":
        assert error.value is failure


@pytest.mark.parametrize("stage", ["finite block", "finite block complete", "finite block readback"])
def test_late_deadline_and_autorelease_cleanup(monkeypatch, stage):
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 32)
    active, scopes, counts = [], [], {}
    failure = TraceTimeout("late deadline")
    finite, narrow = torch.isfinite, torch.Tensor.narrow

    @contextmanager
    def scope(device):
        assert device.type == "cpu" and not active
        active.append(True)
        scopes.append(True)
        try:
            yield
        finally:
            active.pop()

    def check(current):
        assert active
        counts[current] = counts.get(current, 0) + 1
        # Stop in the second batch, including after its final readback.
        if current == stage and counts[current] == (2 if stage.endswith("readback") else 5):
            raise failure

    def evaluate(block):
        assert active
        return finite(block)

    def view(value, *args):
        assert active
        return narrow(value, *args)

    monkeypatch.setattr(_frozen, "autorelease_scope", scope)
    monkeypatch.setattr(torch, "isfinite", evaluate)
    monkeypatch.setattr(torch.Tensor, "narrow", view)
    with pytest.raises(TraceTimeout) as error:
        _frozen._require_finite(torch.ones(9, dtype=torch.float64), "fixture", check)
    assert error.value is failure and len(scopes) == 2 and not active


def test_deadline_after_bad_readback_precedes_value_error(monkeypatch):
    failure = TraceTimeout("late readback")

    def check(stage):
        if stage == "finite block readback":
            raise failure

    with pytest.raises(TraceTimeout) as error:
        _frozen._require_finite(torch.tensor(float("nan")), "fixture", check)
    assert error.value is failure


@pytest.mark.parametrize("late", [False, True])
def test_finite_scan_retains_owned_allocator_admission(metrics, monkeypatch, late):
    guard = metrics.trace()
    value = torch.ones(9, dtype=torch.float32)
    workspace = value.numel() * 24 + 3
    metrics.driver = 32_000_000_000 - workspace + int(not late)
    calls = []

    def evaluate(block):
        calls.append(block.numel())
        assert late and len(calls) == 1
        metrics.driver += 1
        return torch.isfinite(block)

    monkeypatch.setattr(_frozen.torch, "isfinite", evaluate, raising=False)
    with pytest.raises(CleanStop, match="MPS allocator limit"):
        _frozen._require_finite(value, "fixture", memory=guard)
    assert calls == ([9] if late else [])
    assert metrics.fractions == [.5]
