"""Probe identities, actual architecture paths, and capture lifecycle checks."""
import gc
import weakref
from contextlib import ExitStack

import pytest
import torch

from caliper import interp
from caliper._adapter import TORCH
from caliper.interp._points import _delta_pattern
from .fixture import TinyLM


def hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def semantic(model):
    return [p for p in interp.points(model) if "probe" in p]


def qwen(layers=4):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_conv_kernel_dim=3, max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return Qwen3_5ForCausalLM(config).eval()


def test_fixture_residual_identity_and_pattern():
    model = TinyLM()
    requested = semantic(model) + [{"point": f"layers.{i}.self_attn"} for i in range(4)]
    requested += [{"point": f"layers.{i}"} for i in range(4)]
    with interp.capture(model, requested) as values:
        model(torch.tensor([[1, 2, 3, 4]]))
    for i in range(4):
        base = f"layers.{i}"
        actual = values[f"{base}.resid_pre"] + values[f"{base}.self_attn"] + values[f"{base}.mlp_out"]
        torch.testing.assert_close(actual, values[base], atol=1e-6, rtol=0)
        if i < 3:
            torch.testing.assert_close(actual, values[f"layers.{i + 1}.resid_pre"], atol=1e-6, rtol=0)
        torch.testing.assert_close(values[f"{base}.attn_pattern"], model.layers[i].self_attn.last_pattern)
        torch.testing.assert_close(model.layers[i].mlp(values[f"{base}.mlp_in"]), values[f"{base}.mlp_out"])
    assert hooks(model) == 0


def test_capture_shares_storage_and_never_copies_to_cpu(monkeypatch):
    model = TinyLM(layers=1)
    x = torch.randn(1, 3, 64, dtype=torch.float64, requires_grad=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("capture must not copy tensors to CPU")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        with interp.capture(model, ["layers.0.resid_pre", "layers.0.mlp_out"]) as values:
            model(inputs_embeds=x)
    assert values["layers.0.resid_pre"].data_ptr() == x.data_ptr()
    assert values["layers.0.resid_pre"].device == x.device
    assert not any(v.requires_grad for v in values.values())
    values.close(clear=True)
    assert not values


def test_capture_uses_device_storage_without_device_transfer(monkeypatch):
    # Meta tensors expose device placement without a GPU or a data transfer.
    model = TinyLM(layers=1).to("meta")
    original = torch.Tensor.to
    def checked(self, *args, **kwargs):
        assert not args or str(args[0]) not in ("cpu", "cuda", "mps")
        assert "device" not in kwargs
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", checked)
    with interp.capture(model, semantic(model)) as values:
        model(torch.empty(1, 3, dtype=torch.long, device="meta"))
    assert all(t.device.type == "meta" for t in values.values())


def test_modules_heads_and_semantic_ablation():
    model = TinyLM(layers=1)
    names = {p["point"] for p in TORCH.points(model)}
    assert {"layers.0.mlp", "layers.0.self_attn#1", "layers.0.mlp_in"} <= names
    with interp.capture(model, ["layers.0.self_attn#1", "layers.0.mlp_out"]) as values:
        off = TORCH.ablate(model, "layers.0.mlp_out")
        try:
            model(torch.tensor([[2, 3]]))
            # The capture hook predates ablation and thus observes the original output.
            assert values["layers.0.self_attn#1"].shape == (1, 2, 16)
        finally:
            off()
    for point in ("layers.0.mlp_in", "layers.0.resid_pre", "layers.0.mlp_out"):
        off = TORCH.ablate(model, point)
        with interp.capture(model, [point]) as values:
            model(torch.tensor([[2, 3]]))
        off()
        assert values[point].count_nonzero() == 0
    with pytest.raises(ValueError, match="read-only"):
        TORCH.ablate(model, "layers.0.attn_pattern")


def test_capture_cleanup_after_error_and_partial_install(monkeypatch):
    model = TinyLM(layers=1)
    original = model.layers[0].mlp.register_forward_hook
    before = hooks(model)
    with pytest.raises(RuntimeError, match="failed"):
        with interp.capture(model, semantic(model)) as values:
            model(torch.tensor([[1, 2]]))
            raise RuntimeError("failed")
    assert not values and hooks(model) == before
    def failed(*args, **kwargs):
        raise RuntimeError("failed install")
    monkeypatch.setattr(model.layers[0].mlp, "register_forward_hook", failed)
    with pytest.raises(RuntimeError, match="failed install"):
        with interp.capture(model, ["layers.0.resid_pre", "layers.0.mlp_out"]):
            pass
    assert hooks(model) == before
    monkeypatch.setattr(model.layers[0].mlp, "register_forward_hook", original)
    with pytest.raises(IndexError):
        with interp.capture(model, semantic(model)) as failed_values:
            model(torch.tensor([[9999]]))
    assert not failed_values and hooks(model) == before


def test_capture_replaces_prior_forward_and_releases_results():
    model = TinyLM(layers=1)
    with interp.capture(model, ["layers.0.mlp_out"]) as values:
        model(torch.tensor([[1, 2]]))
        old = weakref.ref(values["layers.0.mlp_out"])
        model(torch.tensor([[3]]))
        assert old() is None
        assert values["layers.0.mlp_out"].shape[1] == 1
    current = weakref.ref(values["layers.0.mlp_out"])
    values.close(clear=True)
    gc.collect()
    assert current() is None
    with pytest.raises(RuntimeError, match="once"):
        values.__enter__()


def test_generic_decoder_refusal_preserves_existing_module_points():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert TORCH.points(model)[0]["point"] == "0"
    with pytest.raises(NotImplementedError, match="do not support decoder"):
        interp.points(model)
    with pytest.raises(NotImplementedError, match="do not support decoder"):
        interp.capture(model, ["0"])
    with pytest.raises(KeyError):
        interp.capture(TinyLM(), ["layers.0.missing"])


def test_qwen_actual_paths_and_hybrid_layer_counts():
    model = qwen(24)
    rows = semantic(model)
    assert len(rows) == 24 * 4
    patterns = [p for p in rows if p["probe"] == "attn_pattern"]
    assert sum(p["pattern"] == "signed_value_mixing" for p in patterns) == 18
    assert sum(p["pattern"] == "softmax" for p in patterns) == 6
    modules = dict(model.named_modules())
    assert all(p["module"] in modules for p in rows)
    assert patterns[0]["module"] == "model.layers.0.linear_attn"
    assert patterns[3]["module"] == "model.layers.3.self_attn"
    assert rows[0]["point"] == "model.layers.0.resid_pre"


def test_qwen_pattern_reproduces_real_recurrent_output_and_restores_config():
    model = qwen()
    observed = {}
    handles = []
    for i, block in enumerate(model.model.layers):
        if hasattr(block, "linear_attn"):
            mixer = block.linear_attn
            def projection(mod, args, output, i=i):
                observed[i, "qkv"] = output.detach()
            def core(mod, args, i=i):
                observed[i, "core"] = args[0].detach()
            handles.append(mixer.in_proj_qkv.register_forward_hook(projection))
            handles.append(mixer.norm.register_forward_pre_hook(core))
    implementation = model.config._attn_implementation
    baseline = model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False).logits
    try:
        with interp.capture(model, semantic(model)) as values:
            actual = model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False).logits
        torch.testing.assert_close(actual, baseline, atol=1e-6, rtol=1e-5)
        assert model.config._attn_implementation == implementation
        for i, block in enumerate(model.model.layers):
            pattern = values[f"model.layers.{i}.attn_pattern"]
            assert pattern.shape == (1, 2, 5, 5)
            assert torch.count_nonzero(pattern.triu(1)) == 0
            if hasattr(block, "linear_attn"):
                mixer = block.linear_attn
                qkv = observed[i, "qkv"].transpose(1, 2)
                convolved = torch.nn.functional.silu(mixer.conv1d(qkv)[..., :5]).transpose(1, 2)
                value = convolved[..., 2 * mixer.key_dim:].reshape(1, 5, 2, 4).transpose(1, 2)
                reconstructed = (pattern @ value).transpose(1, 2).reshape(-1, 4)
                torch.testing.assert_close(reconstructed, observed[i, "core"], atol=1e-6, rtol=1e-5)
                assert torch.count_nonzero(pattern) > 0
            else:
                torch.testing.assert_close(pattern.sum(-1), torch.ones(1, 2, 5))
    finally:
        for handle in handles:
            handle.remove()
    assert hooks(model) == 0


def test_delta_operator_is_signed_and_matches_kernel():
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    mixer = qwen(1).model.layers[0].linear_attn
    with torch.no_grad():
        mixer.conv1d.weight.zero_()
        mixer.conv1d.weight[:, :, -1] = 1
        mixer.A_log.fill_(-3)
        mixer.dt_bias.zero_()
    # Opposite queries and keys force signed transport even through SiLU.
    qkv = torch.zeros(1, 3, mixer.conv_dim)
    qkv[..., :mixer.key_dim] = 1
    qkv[..., mixer.key_dim:2 * mixer.key_dim] = -1
    qkv[..., 2 * mixer.key_dim:] = torch.arange(24).reshape(1, 3, 8) / 24
    values = {"qkv": qkv, "a": torch.zeros(1, 3, 2), "b": torch.zeros(1, 3, 2)}
    with torch.no_grad():
        pattern = _delta_pattern(mixer, values)
        convolved = torch.nn.functional.silu(mixer.conv1d(qkv.transpose(1, 2))[..., :3]).transpose(1, 2)
        q, k, v = convolved.split((4, 4, 8), dim=-1)
        q, k, v = q.reshape(1, 3, 1, 4), k.reshape(1, 3, 1, 4), v.reshape(1, 3, 2, 4)
        g = -mixer.A_log.exp() * torch.nn.functional.softplus(values["a"] + mixer.dt_bias)
        output, _ = torch_recurrent_gated_delta_rule(q, k, v, g, values["b"].sigmoid(), use_qk_l2norm_in_kernel=True)
    assert (pattern < 0).any()
    torch.testing.assert_close((pattern @ v.transpose(1, 2)).transpose(1, 2), output, atol=1e-6, rtol=1e-5)


def test_qwen_error_restores_attention_config():
    model = qwen()
    implementation = model.config._attn_implementation
    with pytest.raises(RuntimeError, match="failure"):
        with interp.capture(model, ["model.layers.3.attn_pattern"]) as values:
            assert model.config._attn_implementation == "eager"
            raise RuntimeError("failure")
    assert hooks(model) == 0 and not values
    assert model.config._attn_implementation == implementation


@pytest.mark.parametrize("capture_first", [False, True])
def test_qwen_full_pattern_precedes_module_ablation(capture_first):
    model = qwen()
    attention = model.model.layers[3].self_attn
    point = "model.layers.3.attn_pattern"
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    implementation = model.config._attn_implementation
    with torch.no_grad(), interp.capture(model, [point]) as baseline:
        original = model(ids, use_cache=False).logits
    with torch.no_grad(), ExitStack() as stack:
        if capture_first:
            values = stack.enter_context(interp.capture(model, [point]))
        ablation = TORCH.ablate(model, "model.layers.3.self_attn")
        stack.callback(ablation)
        if not capture_first:
            values = stack.enter_context(interp.capture(model, [point]))
        output = []
        stack.callback(attention.register_forward_hook(
            lambda module, args, result: output.append(result)).remove)
        actual = model(ids, use_cache=False).logits
        assert ablation.calls == 1 and ablation.errors == 0
        assert all(t.count_nonzero() == 0 for t in output[-1])
        torch.testing.assert_close(values[point], baseline[point], atol=0, rtol=0)
        torch.testing.assert_close(values[point].sum(-1), torch.ones(2, 2, 3))
        assert values[point].triu(1).count_nonzero() == 0
        assert not values[point].requires_grad
        assert values[point].device == ids.device
        values.close()
        expected = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert not torch.equal(actual, original)
        assert ablation.calls == 2 and ablation.errors == 0
    assert model.config._attn_implementation == implementation
    assert hooks(model) == 0


def test_qwen_full_pattern_precedes_selected_row_output_hook():
    model = qwen()
    attention = model.model.layers[3].self_attn
    point = "model.layers.3.attn_pattern"
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    implementation = model.config._attn_implementation
    with torch.no_grad(), interp.capture(model, [point]) as baseline:
        original = model(ids, use_cache=False).logits
    # A small output hook represents a scoped ablation without an M8 dependency.
    def ablate_row(module, args, output):
        return tuple(t.masked_fill(torch.tensor([False, True]).reshape(
            2, *([1] * (t.ndim - 1))), 0) for t in output)

    with torch.no_grad(), ExitStack() as stack:
        stack.callback(attention.register_forward_hook(ablate_row).remove)
        with interp.capture(model, [point]) as values:
            actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(values[point], baseline[point], atol=0, rtol=0)
        torch.testing.assert_close(actual[0], original[0], atol=0, rtol=0)
        assert not torch.equal(actual[1], original[1])
        # Keep eager weights in the tuple for the synthetic scoped output hook.
        stack.callback(setattr, model.config, "_attn_implementation", implementation)
        model.config._attn_implementation = "eager"
        expected = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model.config._attn_implementation == implementation
    assert hooks(model) == 0


def test_overlapping_capture_cleanup_preserves_existing_hooks():
    from caliper.interp._points import clear_captures
    model = qwen()
    implementation = model.config._attn_implementation
    existing = model.model.layers[0].register_forward_hook(lambda *args: None)
    one = interp.capture(model, ["model.layers.3.attn_pattern"])
    two = interp.capture(model, ["model.layers.3.attn_pattern"])
    try:
        one.__enter__()
        two.__enter__()
        model(torch.tensor([[1, 2]]), use_cache=False)
        one.close()
        assert model.config._attn_implementation == "eager"
        clear_captures(model)
        assert not two
        assert model.config._attn_implementation == implementation
        assert hooks(model) == 1
    finally:
        one.close(clear=True)
        two.close(clear=True)
        existing.remove()


def test_delta_capture_refuses_populated_cache():
    model = qwen()
    cache = model(torch.tensor([[1, 2]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="complete prompt"):
        with interp.capture(model, ["model.layers.0.attn_pattern"]):
            model(torch.tensor([[3]]), past_key_values=cache, use_cache=True)
    assert hooks(model) == 0


def test_semantic_points_round_trip_through_real_controls_codec():
    from caliper import _wire
    from caliper._controls import Controls
    from caliper._watch import watch
    model = TinyLM(layers=1)
    controls = Controls()
    handle = watch(model, activations=False, groups=False, scene=False)
    controls.interventions_on(model, handle)
    try:
        model(torch.tensor([[1, 2]]))
        controls.checkpoint()
        message = _wire.Controls.from_frame(_wire.Frame(_wire.MsgType.CONTROLS, controls.controls_meta()))
        decoded = _wire.decode(_wire.encode(message))
        rows = {p["point"]: p for p in decoded.to_meta()["interventions"]["points"]}
        assert rows["layers.0.mlp_out"]["kind"] == "module"
        assert rows["layers.0.mlp_out"]["probe"] == "mlp_out"
        assert rows["layers.0.attn_pattern"]["pattern"] == "softmax"
        assert rows["layers.0.self_attn#0"]["kind"] == "head"
    finally:
        handle.remove()
    assert hooks(model) == 0
