"""CPU target batches use the same preallocation gate as device batches."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._frozen import MemoryGuard
from caliper.interp._memory import CleanStop


def test_cpu_target_admission_preserves_headroom_and_float64_cost():
    checks = []
    guard = SimpleNamespace(device=torch.device("cpu"), limit=12_000_000_000, host_reserve_bytes=0,
                            check=lambda stage: checks.append(stage) or 9_000_000_000)
    dimensions = dict(layers=24, tokens=32, width=1024, mixer_width=2048)
    single = MemoryGuard.batch_size(guard, 64, itemsize=4, **dimensions)
    double = MemoryGuard.batch_size(guard, 64, itemsize=8, **dimensions)
    assert 1 <= double < single < 64
    assert checks == ["target batch admission", "target batch admission"]
    guard.check = lambda stage: 11_800_000_000
    with pytest.raises(CleanStop, match="no target batch fits"):
        MemoryGuard.batch_size(guard, 1, itemsize=8, **dimensions)
