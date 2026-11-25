"""CPU geometry evidence for singleton-preserving source groups."""
import numpy as np
import pytest
import torch

from .test_reference_batch_differential import pinned_context
from .test_reference_gradient_cast import Geometry


def pinned_scores(context_class, gradient, vectors, positions, sizes):
    """Execute source hooks with explicit independent partition boundaries."""
    context = object.__new__(context_class)
    context._batch_buffer = vectors.new_zeros(len(vectors), len(gradient))
    offset = 0
    with Geometry() as geometry:
        for size in sizes:
            rows = slice(offset, offset + size)
            _, hook = context._compute_score_hook(
                "synthetic", vectors[rows], rows, np.s_[:, positions[rows]])
            hook(gradient, None)
            offset += size
    assert offset == len(vectors)
    return context._batch_buffer, geometry.calls


def per_source_products(calls):
    """Expose actual BMM matrix operands without a predicted lowering rule."""
    result = []
    for name, operands in calls:
        if name == "aten.bmm.default":
            left, right, output = operands
            for source in range(left[0][0]):
                result.append((left[3][source], right[3][source], output[3][source]))
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_restricted_groups_preserve_pinned_matrix_operands(pinned_context, dtype, batch):
    generator = torch.Generator().manual_seed(144)
    gradient = torch.randn(batch, 5, 37, generator=generator).to(dtype)
    vectors = torch.randn(13, 37, generator=generator).to(dtype)
    positions = torch.tensor([0, 1, 1, 3, 3, 3, 4, 0, 0, 4, 4, 4, 4])
    baseline, original = pinned_scores(pinned_context, gradient, vectors, positions, [1, 2, 3, 1, 2, 4])
    grouped, candidate = pinned_scores(pinned_context, gradient, vectors, positions, [1, 5, 1, 6])
    original_products = per_source_products(original)
    candidate_products = per_source_products(candidate)
    assert len(original_products) == len(candidate_products) == len(vectors)
    for before, after in zip(original_products, candidate_products):
        for left, right in zip(before, after):
            assert left.shape == right.shape
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(grouped, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_singleton_boundary_has_independent_geometry(pinned_context, batch):
    gradient = torch.arange(batch * 3 * 7, dtype=torch.float32).reshape(batch, 3, 7)
    vectors = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7)
    positions = torch.tensor([0, 1, 2])
    _, singles = pinned_scores(pinned_context, gradient, vectors, positions, [1, 1, 1])
    _, joint = pinned_scores(pinned_context, gradient, vectors, positions, [3])
    single_operands = per_source_products(singles)
    joint_operands = per_source_products(joint)
    assert len(single_operands) == len(joint_operands) == 3
    for separate, combined in zip(single_operands, joint_operands):
        if batch == 1:
            assert [tuple(x.shape) for x in separate] == [tuple(x.shape) for x in combined]
        else:
            assert [tuple(x.shape) for x in separate] != [tuple(x.shape) for x in combined]
