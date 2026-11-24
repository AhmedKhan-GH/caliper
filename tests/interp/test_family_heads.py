"""Head points zero the corresponding live output-projection input slice."""
import pytest
import torch

from caliper._ablate import ablate, points
from .test_gemma_family import gemma_model
from .test_points import qwen


@pytest.mark.parametrize("family", ["gemma", "qwen_full", "qwen_delta"])
def test_hf_head_discovery_and_live_ablation(family):
    model = gemma_model(1) if family == "gemma" else qwen()
    model.config._attn_implementation = "eager"
    if family == "gemma":
        name, heads, projection = "model.layers.0.self_attn", 2, "o_proj"
    elif family == "qwen_full":
        name, heads, projection = "model.layers.3.self_attn", 2, "o_proj"
    else:
        name, heads, projection = "model.layers.0.linear_attn", 2, "out_proj"
    advertised = {row["point"] for row in points(model) if row["kind"] == "head"}
    assert {f"{name}#{head}" for head in range(heads)} <= advertised
    assert f"{name}#{heads}" not in advertised
    output = getattr(model.get_submodule(name), projection)
    seen = []
    collect = lambda module, args: seen.append(args[0].detach().clone())
    handle = output.register_forward_pre_hook(collect)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    handle.remove()
    baseline = seen.pop()
    width = output.in_features // heads
    assert baseline[..., width:].count_nonzero() > 0
    off = ablate(model, name + "#1")
    handle = output.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        actual = seen.pop()
        torch.testing.assert_close(actual[..., :width], baseline[..., :width], rtol=0, atol=0)
        assert actual[..., width:].count_nonzero() == 0
    finally:
        handle.remove()
        off()


def test_arbitrary_config_does_not_advertise_attention_heads():
    from types import SimpleNamespace
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=2)
            self.proj = torch.nn.Linear(8, 8)
    model = torch.nn.Sequential(Projection())
    assert not [row for row in points(model) if row["kind"] == "head"]
