"""CPU checks for the lifetime and geometry of reference gradient casts."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from caliper.interp import _trace
from caliper.interp._memory import CleanStop
from .test_direct_decoder_cache import fixture
from .test_reference_batch_differential import pinned_context


def position_case(dtype=torch.bfloat16):
    effects = fixture(dtype)
    path = effects.path
    path.errors = [value.float() for value in path.errors]
    factors = torch.tensor([1., -.375, 2.125]).reshape(1, path.tokens, 1)

    def run(embedding, writes, **kwargs):
        x, reads = embedding, []
        for layer, write in enumerate(writes):
            reads.append(x.sum(1, keepdim=True).expand(-1, path.tokens, -1))
            x = x + write * (factors + layer)
        return reads, x.sum(1, keepdim=True).expand(-1, path.tokens, -1).to(dtype)

    path.run = run
    _, path.reference_final = run(path.embedding, path.errors)
    return effects


class Geometry(TorchDispatchMode):
    """Record exact gather and product operands without device work."""

    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if (func == torch.ops.aten.bmm.default or
                (func == torch.ops.aten.index.Tensor and args[0].ndim == 3)):
            tensors = [args[0], args[1], result] if func == torch.ops.aten.bmm.default else [args[0], result]
            self.calls.append((str(func), [(tuple(x.shape), x.stride(), x.dtype, x.clone()) for x in tensors]))
        return result


def compare_geometry(actual, expected):
    assert len(actual) == len(expected)
    for (name, operands), (ref_name, ref_operands) in zip(actual, expected):
        assert name == ref_name
        for (shape, stride, dtype, value), (ref_shape, ref_stride, ref_dtype, ref_value) in zip(operands, ref_operands):
            assert (shape, stride, dtype) == (ref_shape, ref_stride, ref_dtype)
            torch.testing.assert_close(value, ref_value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_elements", [4, 8, 64])
def test_layer_cast_matches_pinned_sources_and_accounts(pinned_context, monkeypatch, dtype, block_elements):
    effects = position_case(dtype)
    path = effects.path
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", block_elements)
    effects.prepare_decoders()
    nodes = [dict(node) for node in effects.nodes]
    targets = effects.logit_nodes + [effects.feature_nodes[0], effects.feature_nodes[-1]]
    gradients = []
    grad = torch.autograd.grad

    def record(*args, **kwargs):
        result = grad(*args, **kwargs)
        gradients.extend(result)
        return result

    monkeypatch.setattr(torch.autograd, "grad", record)
    with Geometry() as actual_geometry:
        actual, accounts = effects.backward(targets)
    reference = object.__new__(pinned_context)
    reference._batch_buffer = torch.zeros_like(actual)

    def score(vector, rows, gradient, read_index=np.s_[:]):
        _, hook = reference._compute_score_hook("synthetic", vector, rows, read_index)
        hook(gradient, None)

    with Geometry() as expected_geometry:
        score(path.embedding[0], slice(0, path.tokens), gradients[0])
        for layer, blocks in enumerate(effects._decoder_blocks):
            gradient = gradients[layer + 1]
            assert not torch.equal(gradient[:, 0], gradient[:, 2])
            for rows, position, decoder in blocks:
                score(decoder, rows, gradient, np.s_[:, torch.full((len(decoder),), position, dtype=torch.long)])
            score(path.errors[layer][0], effects.source_layers[layer][1], gradient)
    expected = reference._batch_buffer
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Bias accounts follow the source contractions and use separate products.
    compare_geometry(actual_geometry.calls[:len(expected_geometry.calls)], expected_geometry.calls)
    incoming = expected.sum(0, dtype=torch.float32)
    for index, account in enumerate(accounts):
        target = effects.target(targets[index])
        assert account["expected"] == float(target.expected)
        assert account["incoming"] == float(incoming[index])
        assert account["biases"] == dict(decoder=0., normalization=0., attention=0., intervention=0.,
                                         encoder=float(target.encoder_bias), unembedding=float(target.unembedding_bias))
        residual = float(abs(np.float32(np.float32(float(incoming[index])) + np.float32(float(target.encoder_bias)))
                             - np.float32(float(target.expected))))
        assert account["absolute_residual"] == residual
        assert account["relative_residual"] == residual / max(1., abs(float(target.expected)))
    assert effects.nodes == nodes


def observe_casts(monkeypatch, effects, batch):
    original = torch.Tensor.to
    refs, calls = [], []

    def cast(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if (value.shape == (batch, effects.path.tokens, effects.path.width)
                and args == (effects.path.dtype,) and value.dtype != result.dtype and not value.requires_grad):
            refs.append(weakref.ref(result))
            calls.append((value.shape, value.stride(), result.shape, result.stride()))
        return result

    monkeypatch.setattr(torch.Tensor, "to", cast)
    return refs, calls


@pytest.mark.parametrize("block_elements", [4, 8, 64])
def test_cast_reuse_keeps_each_block_reserve_and_releases_each_layer(monkeypatch, block_elements):
    effects = position_case()
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", block_elements)
    effects.prepare_decoders()
    batch = len(effects.logit_nodes)
    refs, calls = observe_casts(monkeypatch, effects, batch)
    blocks = [block for layer in effects._decoder_blocks for block in layer]
    sizes, live = [], []

    def reserve(stage, size):
        if stage == "direct decoder contraction":
            decoder = blocks[len(sizes)][2]
            count, itemsize = len(decoder), decoder.element_size()
            assert size == batch * effects.path.tokens * effects.path.width * itemsize + batch * count * (effects.path.width + 1) * itemsize + count * 8
            sizes.append(size)
            live.append(sum(ref() is not None for ref in refs))
        elif stage == "direct decoder bias accounting":
            assert all(ref() is None for ref in refs)

    effects.path.guard.reserve = reserve
    effects.backward(effects.logit_nodes)
    assert len(calls) == effects.path.layers
    assert len(sizes) == len(blocks) > len(calls)
    first_indices = [0, len(effects._decoder_blocks[0])]
    assert live == [0 if index in first_indices else 1 for index in range(len(blocks))]
    assert all(ref() is None for ref in refs)
    sizes.clear()
    live.clear()
    effects.backward(effects.logit_nodes)
    assert len(calls) == 2 * effects.path.layers
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("block", [1, 2])
@pytest.mark.parametrize("reason", [CleanStop, _trace.TraceTimeout])
def test_refusal_precedes_block_allocations_and_releases_cast(monkeypatch, block, reason):
    effects = position_case()
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 4)
    effects.prepare_decoders()
    refs, calls = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    full = torch.full
    allocations, reserves, scopes = [], [], []

    def positions(*args, **kwargs):
        allocations.append(args)
        return full(*args, **kwargs)

    @contextmanager
    def scope(device):
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")

    def reserve(stage, size):
        if stage == "direct decoder contraction":
            reserves.append(size)
            if len(reserves) == block:
                raise reason("block refusal")

    effects.path.guard.reserve = reserve
    monkeypatch.setattr(torch, "full", positions)
    monkeypatch.setattr(_trace, "autorelease_scope", scope)
    with pytest.raises(reason, match="block refusal") as failure:
        effects.backward(effects.logit_nodes)
    assert failure.value.__traceback__ is not None
    assert len(allocations) == block - 1
    assert len(calls) == min(1, block - 1)
    assert all(ref() is None for ref in refs)
    assert scopes == [entry for _ in range(block) for entry in ("enter", "exit")]
    effects.release_decoders()
    assert effects._decoder_blocks is None


@pytest.mark.parametrize("stage", ["direct decoder block", "finite block readback", "direct edge products"])
@pytest.mark.parametrize("reason", [CleanStop, _trace.TraceTimeout])
def test_late_failures_keep_checks_and_cleanup(monkeypatch, stage, reason):
    effects = position_case()
    effects.prepare_decoders()
    refs, _ = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    seen = []

    def check(name):
        seen.append(name)
        if name == stage and (stage != "direct decoder block" or seen.count(name) == 2):
            raise reason("late refusal")

    effects.check = check
    with pytest.raises(reason, match="late refusal") as failure:
        effects.backward(effects.logit_nodes)
    assert failure.value.__traceback__ is not None
    assert refs and all(ref() is None for ref in refs)
    assert seen[-1] == stage
    effects.release_decoders()
    assert effects._decoder_blocks is None
