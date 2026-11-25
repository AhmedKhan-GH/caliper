"""Synthetic CPU comparisons against the pinned attribution source."""
import ast
import contextlib
from functools import partial
import hashlib
import os
from pathlib import Path
from typing import Callable
import weakref

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from caliper.interp._memory import CleanStop
from .test_direct_decoder_cache import fixture
from .reference_source_oracle import (pinned_decoder, whole_layer_scores, contraction_evidence,
                                      compare_source_products)


@pytest.fixture
def pinned_context():
    """Load only the verified class, without model imports or payload access."""
    source = os.environ.get("CALIPER_PINNED_ATTRIBUTION_SOURCE")
    if source is None:
        pytest.skip("the pinned attribution source is required for the external oracle")
    # An explicit source opts into the oracle and requires its dependency.
    from einops import einsum

    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "1a60242cc9b9acc94fa995b3ec8d90984675878c86712bb45f39a14f2717e9f6"
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AttributionContext")
    namespace = dict(torch=torch, np=np, einsum=einsum, partial=partial,
                     weakref=weakref, contextlib=contextlib, HookPoint=object, Callable=Callable)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), source, "exec"), namespace)
    return namespace["AttributionContext"]


class Products(TorchDispatchMode):
    """Record the contraction geometry that selects each device kernel."""

    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.calls.append((str(func), tuple(args[0].shape), tuple(args[1].shape)))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("counts", [None, [2, 1, 1], [2, 1, 2], [4, 0, 1]])
@pytest.mark.parametrize("row_bound", [1, 2, 5])
def test_feature_contraction_uses_pinned_batched_dot_geometry(
        pinned_context, pinned_decoder, monkeypatch, dtype, counts, row_bound):
    from caliper.interp import _trace

    effects = fixture(dtype, counts)
    if dtype == torch.float32:
        generator = torch.Generator().manual_seed(1729)
        path = effects.path
        path.width = path.mixer_width = 64
        path.embedding = torch.randn(1, path.tokens, path.width, generator=generator)
        path.errors = [path.embedding / 4, -path.embedding / 8]
        path.decoder_biases = [torch.zeros_like(path.embedding) for _ in range(path.layers)]
        path.direction = torch.randn(path.width, generator=generator)
        path.bias_final = path.embedding.new_zeros(4, path.tokens, path.width)
        path.bias_reads = [path.bias_final.clone() for _ in range(path.layers)]
        _, path.reference_final = path.run(path.embedding, path.errors)
        for layer in effects.dictionary.layers:
            layer.W_dec = torch.randn(5, path.width, generator=generator)
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", row_bound * effects.path.width)
    targets = effects.logit_nodes * 3
    with Products() as actual_products:
        (actual, _), gradients, calls, _ = contraction_evidence(effects, targets, monkeypatch)
    # SumPath has the same independent derivative at every source position.
    directions = torch.stack([effects.target(index).direction for index in targets])
    gradient = directions[:, None].expand(-1, effects.path.tokens, -1)
    for observed in gradients:
        torch.testing.assert_close(observed, gradient, rtol=0, atol=0)
    expected, scopes = whole_layer_scores(effects, [gradient] * len(gradients), pinned_context, pinned_decoder)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compare_source_products(effects, calls, scopes, len(targets))
    assert calls
    assert not any(name == "aten.mm.default" and left == (len(targets), effects.path.width)
                   and right[0] == effects.path.width
                   for name, left, right in actual_products.calls)


def test_pinned_seed_rows_and_padding_are_independent(pinned_context):
    context = object.__new__(pinned_context)
    source = torch.zeros(8, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    early = source * 2
    late = early * 3
    context._resid_activations = [early, late]
    context._row_size = 3
    vector = torch.tensor([[1., 2., 3., 4.]] * 3, dtype=source.dtype)
    _, score = context._compute_score_hook("source", vector, slice(None))
    handle = source.register_hook(lambda grad: score(grad, None))
    layers = torch.tensor([1, 0, 1])
    positions = torch.tensor([2, 0, 1])
    values = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], dtype=source.dtype)
    result = context.compute_batch(layers, positions, values)
    handle.remove()
    assert result.shape == (3, 3)
    assert result.tolist() == [[0., 0., 6.], [4., 0., 0.], [0., 18., 0.]]
    assert torch.count_nonzero(source.grad[3:]) == 0


@pytest.mark.parametrize("physical_width", [3, 8])
def test_direct_source_gradients_match_pinned_mixed_target_seeds(pinned_context, monkeypatch, physical_width):
    effects = fixture()
    path = effects.path
    path.errors = [value.float() for value in path.errors]

    def run(embedding, writes, **kwargs):
        x, reads = embedding, []
        for write in writes:
            read = x * 2
            reads.append(read)
            # The pinned no-skip replacement retains a zero derivative branch.
            x = x + write + read * 0
        return reads, x

    path.run = run
    indices = [effects.feature_nodes[-1], effects.feature_nodes[0], effects.logit_nodes[0]]
    targets = [effects.target(index) for index in indices]
    sources = [source.new_zeros(physical_width, path.tokens, path.width, requires_grad=True)
               for source in [path.embedding, *path.errors]]
    reads, final = run(sources[0], sources[1:])
    context = object.__new__(pinned_context)
    context._resid_activations = reads + [final]
    context._row_size = 1
    context.compute_batch(torch.tensor([target.layer for target in targets]),
                          torch.tensor([target.position for target in targets]),
                          torch.stack([target.direction for target in targets]), retain_graph=False)
    expected = [source.grad for source in sources]
    observed = []
    original = torch.autograd.grad

    def record(*args, **kwargs):
        gradients = original(*args, **kwargs)
        observed.append(gradients)
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", record)
    effects.backward(indices)
    actual, = observed
    for first, second in zip(actual, expected):
        assert first.dtype == second.dtype
        torch.testing.assert_close(first, second[:len(indices)], atol=0, rtol=0)
        assert torch.count_nonzero(second[len(indices):]) == 0


def test_source_contraction_refuses_workspace_before_product():
    effects = fixture()
    effects.prepare_decoders()

    def reserve(stage, size):
        if stage == "direct decoder contraction":
            assert size > 0
            raise CleanStop("source contraction refusal")

    effects.path.guard.reserve = reserve
    with pytest.raises(CleanStop, match="source contraction refusal"):
        effects.backward(effects.logit_nodes)
