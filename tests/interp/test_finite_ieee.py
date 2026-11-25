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
