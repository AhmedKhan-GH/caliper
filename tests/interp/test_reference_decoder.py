"""Bounded reference decoder checks with synthetic CPU tensors only."""
import ast
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen
from caliper.interp._memory import CleanStop
from .test_gemma_frozen_reference import reference_case


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


@pytest.mark.parametrize("include_zeros", [False, True])
def test_joint_operation_has_ordered_support_and_one_call(monkeypatch, include_zeros):
    layer, _, indices, values = synthetic_case()
    calls = []
    index_add = torch.Tensor.index_add_

    def record(result, dim, positions, scaled):
        calls.append((positions.tolist(), scaled.shape))
        return index_add(result, dim, positions, scaled)

    monkeypatch.setattr(torch.Tensor, "index_add_", record)
    _frozen._decode_reference(layer, indices, values, include_zeros=include_zeros)
    wanted = [0] * 3 + [1] * (3 if include_zeros else 1) + [2] * 3
    if include_zeros:
        wanted += [3] * 3
    assert calls == [(wanted, (len(wanted), 3))]


@pytest.mark.parametrize("shape", [(3, 0), (0, 0)])
def test_empty_reference_support_has_zero_output_and_derivative(shape):
    layer, _, _, _ = synthetic_case()
    indices = torch.empty(shape, dtype=torch.long)
    values = torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
    decoded = _frozen._decode_reference(layer, indices, values, include_zeros=True)
    assert decoded.shape == (shape[0], 3)
    assert decoded.count_nonzero() == 0
    decoded.sum().backward()
    assert values.grad.shape == values.shape


def scalar_guard(monkeypatch, *, current=1_000, available=64_000_000_000):
    """Use unchanged production reserve logic with small scalar readers."""
    import psutil
    guard = _frozen.MemoryGuard.__new__(_frozen.MemoryGuard)
    guard.limit = 40_000_000_000
    guard._host_reserve_bytes = 8_000_000_000
    guard.check = lambda stage: current
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    return guard


@pytest.mark.parametrize("bound", ["process", "host"])
@pytest.mark.parametrize("phase", ["selection", "workspace"])
def test_exact_workspace_refuses_before_decoder_allocation(monkeypatch, bound, phase):
    layer, _, indices, values = synthetic_case()
    layer.W_dec = layer.W_dec.repeat(1, 32)
    room = 1 if phase == "selection" else 500
    guard = scalar_guard(monkeypatch, **({"current": 40_000_000_000 - room} if bound == "process"
                                        else {"available": 8_000_000_000 + room}))
    stages = []

    def forbidden(*args, **kwargs):
        pytest.fail("the decoder must refuse before rows or output allocation")

    monkeypatch.setattr(torch.Tensor, "detach", forbidden)
    monkeypatch.setattr(torch.Tensor, "new_zeros", forbidden)
    match = "memory limit" if bound == "process" else "8 GB host reserve"
    with pytest.raises(CleanStop, match=match):
        _frozen._decode_reference(layer, indices, values, guard=guard, check=stages.append)
    assert stages == (["reference decoder selection"] if phase == "selection" else
                      ["reference decoder selection", "reference decoder workspace"])


def test_zero_delta_workspace_includes_retained_rows_and_adjoint(monkeypatch):
    layer, _, indices, values = synthetic_case()
    guard = scalar_guard(monkeypatch, current=40_000_000_000 - 600)
    deltas = torch.zeros(2, *values.shape, dtype=values.dtype, requires_grad=True)
    with torch.no_grad():
        result = _frozen._decode_reference(layer, indices, deltas, include_zeros=True, guard=guard)
    assert result.count_nonzero() == 0
    # Forward work fits, but the retained support and adjoint do not fit.
    with pytest.raises(CleanStop, match="joint workspace allocation exceeds"):
        _frozen._decode_reference(layer, indices, deltas, include_zeros=True, guard=guard)


@pytest.mark.parametrize("stage", ["selection", "workspace", "rows", "scaling", "accumulation", "complete"])
@pytest.mark.parametrize("reason", ["cancelled", "deadline"])
def test_decoder_stop_preserves_autorelease_cleanup(monkeypatch, stage, reason):
    layer, _, indices, values = synthetic_case()
    scopes = []

    @contextmanager
    def scope(device):
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")

    def check(name):
        if name == "reference decoder " + stage:
            raise CleanStop(reason)

    monkeypatch.setattr(_frozen, "autorelease_scope", scope)
    with pytest.raises(CleanStop, match=reason):
        _frozen._decode_reference(layer, indices, values, check=check)
    assert scopes == ["enter", "exit"]


def test_reference_path_passes_its_guard_to_every_decoder_call(reference_case, monkeypatch):
    case = reference_case
    calls = []
    decode = _frozen._decode_reference

    def record(*args, **kwargs):
        calls.append(kwargs)
        return decode(*args, **kwargs)

    monkeypatch.setattr(_frozen, "_decode_reference", record)
    path = _frozen.FrozenPath(case.model, case.ids, case.dictionary, profile="circuit_tracer_82996f14")
    assert len([call for call in calls if not call.get("include_zeros")]) == 2 * path.layers
    assert all(call["guard"] is path.guard for call in calls)
    calls.clear()
    path.replacement(path.embedding, case.dictionary)
    assert len(calls) == path.layers
    assert all(call["guard"] is path.guard and call["include_zeros"] for call in calls)


def test_signed_batched_values_preserve_all_batch_sources():
    layer, _, indices, values = synthetic_case(torch.float64)
    values = torch.stack((values, -values))
    values[0, 0] = 0
    values[1, 2] = 0
    values.requires_grad_()
    actual = _frozen._decode_reference(layer, indices, values)
    expected = (values[..., None] * layer.W_dec[indices]).sum(-2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    expected_grad = layer.W_dec[indices].sum(-1).expand_as(values).clone()
    expected_grad *= (values.detach() != 0).any(0)
    torch.testing.assert_close(values.grad, expected_grad, atol=0, rtol=0)
