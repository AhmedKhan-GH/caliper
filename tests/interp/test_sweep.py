"""Real residual patch math and scoped cleanup on CPU fixtures."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from caliper._controls import Controls
from caliper.interp._sweep import patch_sweep
from caliper.interp._publish import _clear, _resolve
from .fixture import TinyLM, TinyTokenizer


class CopyBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.mlp = nn.Identity()
        self.self_attn = nn.Module()
        self.self_attn.pattern = nn.Identity()

    def forward(self, x):
        x = x.clone()
        if self.layer == 0:
            x[:, -1, 0] = x[:, 1, 0]
            x[:, 1, 1] = x[:, 1, 0]
            x[:, 1, 0] = 0
        elif self.layer == 1:
            x[:, -1, 0] -= 0.5 * x[:, 1, 1]
            x[:, :, 1] = 0
        return x


class PlantedCopy(nn.Module):
    """Copy one source token, then subtract a separate distractor channel."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="caliper_tiny", num_hidden_layers=4, vocab_size=512)
        self.embed_tokens = nn.Embedding(512, 2)
        self.layers = nn.ModuleList(CopyBlock(i) for i in range(4))
        with torch.no_grad():
            self.embed_tokens.weight.zero_()
            self.embed_tokens.weight[65, 0] = 1
        self.eval()

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        logits = x.new_zeros(*x.shape[:2], 512)
        logits[:, :, 65] = x[:, :, 0]
        return SimpleNamespace(logits=logits)


def test_planted_copy_recovers_known_cell_with_actual_math():
    model, tokenizer = PlantedCopy(), TinyTokenizer()
    controls = Controls()
    service = controls.interventions_on(model)
    try:
        result = patch_sweep(model, tokenizer, service, clean_prompt="AZ", corrupted_prompt="BZ",
                             metric={"positive_token": 65, "negative_token": 66})
        grid = _resolve("patch.grid")
        assert grid.dtype == torch.float32 and grid.shape == (4, 3)
        assert result["clean_metric"] == 0.5 and result["corrupted_metric"] == 0
        assert int(grid.argmax()) == 1 * 3 + 2
        assert grid[1, 2] == 2
        assert grid[0, 1] == 1 and grid[1, 1] == -1
        assert result["recovery"] == grid.tolist()
        assert all(not b._forward_pre_hooks for b in model.layers)
    finally:
        service.remove()
        _clear()


def test_sweep_batches_32_cells_and_preserves_intervention_state():
    model, tokenizer = TinyLM(), TinyTokenizer()
    controls = Controls()
    service = controls.interventions_on(model)
    service.set({"kind": "ablate", "point": "layers.3.mlp"})
    controls.step()
    before = service.state_dict()
    calls = []
    hook = model.register_forward_hook(lambda m, a, y: calls.append((y.logits.shape[0], service.suspended)))
    try:
        patch_sweep(model, tokenizer, service, clean_prompt="ABCDEFGHI", corrupted_prompt="JBCDEFGHI",
                    metric={"positive_token": 65, "negative_token": 66})
        assert calls == [(1, True), (1, True), (32, True), (8, True)]
        assert service.state_dict() == before and not service.suspended
        assert all(not b._forward_pre_hooks for b in model.layers)
    finally:
        hook.remove()
        service.remove()
        _clear()


@pytest.mark.parametrize("failure", ["length", "zero", "forward"])
def test_sweep_refusals_and_failure_remove_hooks(failure, monkeypatch):
    model, tokenizer = PlantedCopy(), TinyTokenizer()
    controls = Controls()
    service = controls.interventions_on(model)
    original = model.forward
    count = 0
    if failure == "forward":
        def fail(*a, **kw):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("patch forward failed")
            return original(*a, **kw)
        monkeypatch.setattr(model, "forward", fail)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            patch_sweep(model, tokenizer, service, clean_prompt="AZ",
                        corrupted_prompt={"length": "Z", "zero": "AZ", "forward": "BZ"}[failure],
                        metric={"positive_token": 65, "negative_token": 66})
        assert all(not b._forward_pre_hooks for b in model.layers)
        assert not service.suspended and controls.n == 0
    finally:
        service.remove()
