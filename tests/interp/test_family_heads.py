"""Head points zero the corresponding live output-projection input slice."""
import pytest
import torch

from caliper._ablate import ablate, points
from caliper.interp._gemma_reference import PROFILE, ReferenceAttention, install_reference_profile
from .test_gemma_family import gemma_model
from .test_gemma_reference import reference_fixture
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
    with pytest.raises(KeyError):
        ablate(model, f"{name}#{heads}")
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head", range(4))
def test_reference_head_discovery_and_live_ablation(dtype, head):
    model = reference_fixture(dtype)
    expected_points = {
        f"model.layers.{layer}.self_attn#{index}"
        for layer in range(2) for index in range(4)
    }
    assert {row["point"] for row in points(model) if row["kind"] == "head"} == expected_points
    install_reference_profile(model, PROFILE)
    rows = [row for row in points(model) if row["kind"] == "head"]
    assert {row["point"] for row in rows} == expected_points
    assert all(row["type"] == "ReferenceAttention" for row in rows)
    attention = model.model.layers[0].self_attn
    assert type(attention) is ReferenceAttention
    assert not any(hasattr(attention, attr) for attr in ("num_heads", "n_heads", "n_head"))
    output = attention.o_proj
    seen = []
    collect = lambda module, args: seen.append(args[0].detach().clone())
    ids = torch.tensor([[1, 4, 6, 9, 2, 11, 3]])
    handle = output.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            baseline_logits = model(input_ids=ids).logits
        baseline, = seen
    finally:
        handle.remove()
    seen.clear()
    width = output.in_features // model.config.num_attention_heads
    start, stop = head * width, (head + 1) * width
    assert baseline[..., start:stop].count_nonzero() > 0
    expected = baseline.clone()
    expected[..., start:stop] = 0
    off = ablate(model, f"model.layers.0.self_attn#{head}")
    handle = output.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            model(input_ids=ids)
        actual, = seen
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert off.calls == 1 and off.errors == 0 and off.error is None
        off()
        seen.clear()
        with torch.no_grad():
            restored_logits = model(input_ids=ids).logits
        restored, = seen
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
        torch.testing.assert_close(restored_logits, baseline_logits, rtol=0, atol=0)
    finally:
        handle.remove()
        off()
    assert not output._forward_pre_hooks


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("count", [None, False, True, 0, -1, 1.5, "4", 5, 25])
def test_gemma_invalid_head_counts_are_not_points(adapted, count):
    model = reference_fixture()
    if adapted:
        install_reference_profile(model, PROFILE)
    # Bypass the config validator to test head discovery with malformed state.
    object.__setattr__(model.config, "num_attention_heads", count)
    assert not [row for row in points(model) if row["kind"] == "head"]
    with pytest.raises(KeyError):
        ablate(model, "model.layers.0.self_attn#0")
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks


@pytest.mark.parametrize("head", ["4", "-1", "01", "True", "1.0", ""])
def test_reference_invalid_head_ids_are_refused(head):
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    with pytest.raises(KeyError):
        ablate(model, f"model.layers.0.self_attn#{head}")
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks


def test_arbitrary_config_does_not_advertise_attention_heads():
    from types import SimpleNamespace
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=2)
            self.proj = torch.nn.Linear(8, 8)
    model = torch.nn.Sequential(Projection())
    assert not [row for row in points(model) if row["kind"] == "head"]
    with pytest.raises(KeyError):
        ablate(model, "0#0")
