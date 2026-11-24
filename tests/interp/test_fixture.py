"""Fixture identity and exact transcoder checks."""
import hashlib
import json
from pathlib import Path

import torch

from .fixture import TinyLM, forward_logits


def test_saved_forward_reference():
    payload = Path(__file__).with_name("fixture_logits.json").read_bytes().strip()
    expected_hash = Path(__file__).with_name("fixture.sha256").read_text().strip()
    assert hashlib.sha256(payload).hexdigest() == expected_hash
    expected = torch.tensor(json.loads(payload), dtype=torch.float64) / 1e8
    # Parameters start as float32. CPU initialization and kernels can differ in
    # their last bits, even when later computation uses float64.
    torch.testing.assert_close(forward_logits(), expected, atol=1e-6, rtol=1e-6)


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
