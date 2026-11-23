"""Fixture identity and exact transcoder checks."""
from pathlib import Path

import torch

from .fixture import TinyLM, forward_hash


def test_saved_forward_hash():
    expected = Path(__file__).with_name("fixture.sha256").read_text().strip()
    assert forward_hash() == expected


def test_exact_transcoders_and_causality():
    model = TinyLM()
    for block in model.layers:
        x = torch.arange(128, dtype=torch.float64).reshape(1, 2, 64) / 128
        a = block.mlp.encode(x)
        assert torch.equal(block.mlp(x), block.mlp.decode(a))
        assert (a.count_nonzero(dim=-1) <= 8).all()
    first = model(torch.tensor([[1, 2, 3]])).logits
    changed = model(torch.tensor([[1, 2, 4]])).logits
    torch.testing.assert_close(first[:, :2], changed[:, :2], atol=0, rtol=0)


def test_construction_preserves_random_state():
    before = torch.random.get_rng_state().clone()
    TinyLM()
    assert torch.equal(before, torch.random.get_rng_state())
