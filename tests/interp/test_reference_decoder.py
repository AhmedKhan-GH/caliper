"""Bounded reference decoder checks with synthetic CPU tensors only."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen


@pytest.fixture
def pinned_decode():
    """Execute the actual pinned method without package imports or payloads."""
    source = os.environ.get("CALIPER_PINNED_TRANSCODER_SOURCE")
    if source is None:
        pytest.skip("the pinned source file is required for the external oracle")
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "8fb5719906c96653efed86353bbf83fd7217d646edf9f7c05e6ab157cd1a55e7"
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "SingleLayerTranscoder")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "decode_sparse")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), source, "exec"), namespace)
    return namespace["decode_sparse"]


def synthetic_case(dtype=torch.bfloat16):
    weights = torch.tensor([[256., .1, -.75], [1., .2, 2.], [-256., .3, .5],
                            [.25, -1.5, 3.], [1.5, .75, -2.]], dtype=dtype)
    dense = torch.tensor([[1., 1., 1., 0., 0.], [0., 0., 1., 0., 0.],
                          [0., .5, 0., 2., 1.], [0., 0., 0., 0., 0.]], dtype=dtype)
    layer = SimpleNamespace(W_dec=weights, b_dec=torch.zeros(3, dtype=dtype),
                            d_model=3, W_skip=None, _get_decoder_vectors=lambda ids: weights[ids])
    indices = _frozen._active_columns(dense)
    return layer, dense, indices, dense.gather(-1, indices)


@pytest.mark.parametrize("block_elements", [1_048_576, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_decoder_matches_pinned_source(pinned_decode, monkeypatch, dtype, block_elements):
    layer, dense, indices, values = synthetic_case(dtype)
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", block_elements)
    calls = []
    index_add = torch.Tensor.index_add_

    def record(result, dim, destinations, scaled, *args, **kwargs):
        calls.append((dim % result.ndim, destinations.clone(), scaled.clone()))
        return index_add(result, dim, destinations, scaled, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_add_", record)
    expected, _ = pinned_decode(layer, dense.to_sparse())
    wanted, = calls
    calls.clear()
    actual = _frozen._decode_reference(layer, indices, values)
    assert len(calls) == 1, "the reference requires one joint accumulation across positions"
    observed, = calls
    assert observed[0] == wanted[0]
    torch.testing.assert_close(observed[1], wanted[1], atol=0, rtol=0)
    torch.testing.assert_close(observed[2], wanted[2], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_bos_sources_remain_distinct_from_published_suppression(pinned_decode):
    layer, dense, indices, values = synthetic_case()
    original = dense.clone()
    full, _ = pinned_decode(layer, dense.to_sparse())
    published = dense.clone()
    published[0] = 0
    suppressed, _ = pinned_decode(layer, published.to_sparse())
    actual = _frozen._decode_reference(layer, indices, values)
    torch.testing.assert_close(actual, full, atol=0, rtol=0)
    torch.testing.assert_close(actual[1:], suppressed[1:], atol=0, rtol=0)
    assert actual[0].count_nonzero() > 0
    assert suppressed[0].count_nonzero() == 0
    torch.testing.assert_close(dense, original, atol=0, rtol=0)


def test_joint_decoder_preserves_zero_delta_derivatives():
    layer, _, indices, values = synthetic_case(torch.float64)
    deltas = torch.zeros(2, *values.shape, dtype=values.dtype, requires_grad=True)
    actual = _frozen._decode_reference(layer, indices, deltas, include_zeros=True)
    assert actual.shape == (2, 4, 3)
    assert actual.count_nonzero() == 0
    direction = torch.tensor([1., -.5, .25], dtype=values.dtype)
    (actual * direction).sum().backward()
    expected = (layer.W_dec[indices] * direction).sum(-1).expand_as(deltas)
    torch.testing.assert_close(deltas.grad, expected, atol=0, rtol=0)
