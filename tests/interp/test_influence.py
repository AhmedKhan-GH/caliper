"""Influence orientation, mixed signs, disconnected nodes, and bounds."""
import pytest
import torch

from caliper.interp._influence import influence, normalized_incoming


def test_power_matches_inverse_with_mixed_signs_and_disconnected_nodes():
    weights = torch.tensor([[0., -2., 1., 0., 0., 0.],
                            [0., 0., -4., 3., 0., 0.],
                            [0., 0., 0., -2., 1., 0.],
                            [0., 0., 0., 0., -5., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.]], dtype=torch.float64)
    seeds = torch.tensor([0., 0., 0., .2, .8, 0.], dtype=torch.float64)
    targets = torch.arange(6)
    actual, normalized = influence(weights, targets, seeds)
    expected = torch.linalg.solve(torch.eye(6, dtype=torch.float64) - normalized, seeds)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert actual[0] > 0 and actual[5] == 0
    assert not torch.allclose(actual, torch.linalg.solve(torch.eye(6) - normalized.T, seeds))
    compact, _ = influence(weights[:, 1:5], targets[1:5], seeds)
    torch.testing.assert_close(compact, actual, atol=1e-12, rtol=0)
    torch.testing.assert_close(normalized.sum(0), torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.float64))


def test_zero_incoming_weights():
    seeds = torch.tensor([0., 1.])
    actual, normalized = influence(torch.zeros(2, 1), torch.tensor([1]), seeds)
    assert torch.equal(actual, seeds) and normalized.count_nonzero() == 0


@pytest.mark.parametrize("weights", [torch.ones(2), torch.ones(2, 2, dtype=torch.int64),
                                    torch.tensor([[float('nan')]])])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        normalized_incoming(weights)


def test_invalid_target_and_seed_bounds():
    with pytest.raises(ValueError, match="bounds"):
        influence(torch.zeros(2, 1), torch.tensor([2]), torch.ones(2))
    with pytest.raises(ValueError, match="unique"):
        influence(torch.zeros(2, 2), torch.tensor([1, 1]), torch.ones(2))
    with pytest.raises(ValueError, match="nonnegative"):
        influence(torch.zeros(2, 1), torch.tensor([1]), torch.tensor([0., -1.]))
