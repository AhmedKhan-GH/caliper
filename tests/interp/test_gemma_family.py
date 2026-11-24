"""Gemma semantic probes against the installed random decoder."""
from contextlib import ExitStack

import torch

from caliper.interp._families import model_family
from caliper.interp._points import _architecture, capture


def gemma_model(layers=3, *, softcap=.15, window=2):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    config = Gemma2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                          num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=8, max_position_embeddings=32, sliding_window=window,
                          query_pre_attn_scalar=5, attn_logit_softcapping=.2,
                          final_logit_softcapping=softcap, attention_bias=True,
                          hidden_activation="gelu_pytorch_tanh")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        model = Gemma2ForCausalLM(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.copy_(torch.linspace(-.3, .7, parameter.numel()))
                elif name.endswith("bias"):
                    parameter.copy_(torch.linspace(-.02, .03, parameter.numel()))
            # Make attention softcap observable instead of almost linear.
            for block in model.model.layers:
                block.self_attn.q_proj.weight.mul_(8)
                block.self_attn.k_proj.weight.mul_(8)
    return model


def test_family_entries_retain_fixture_and_qwen_paths():
    from .fixture import TinyLM
    from .test_points import qwen
    for model, name, prefix in ((TinyLM(layers=1), "caliper_tiny", "layers"),
                                (qwen(layers=1), "qwen3_5_text", "model.layers"),
                                (gemma_model(1), "gemma2", "model.layers")):
        family = model_family(model)
        assert family.name == name
        assert family.locate(model)[1] == prefix
        assert _architecture(model)[f"{prefix}.0.resid_pre"].module == f"{prefix}.0"
        assert _architecture(model)[f"{prefix}.0.mlp_in"].module == f"{prefix}.0.mlp"


def test_gemma_semantic_residuals_gains_geglu_and_scaled_embeddings():
    model = gemma_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    requested = list(_architecture(model)) + ["model.embed_tokens"]
    for i in range(3):
        requested += [f"model.layers.{i}", f"model.layers.{i}.post_attention_layernorm"]
    with torch.no_grad(), capture(model, requested) as values:
        model(input_ids=ids, use_cache=False)
    torch.testing.assert_close(values["model.layers.0.resid_pre"],
                               model.model.embed_tokens.weight[ids] * 16 ** .5)
    for i, block in enumerate(model.model.layers):
        base = f"model.layers.{i}"
        read = values[base + ".mlp_in"]
        raw = block.mlp.down_proj(block.mlp.act_fn(block.mlp.gate_proj(read)) * block.mlp.up_proj(read))
        write = block.post_feedforward_layernorm(raw)
        torch.testing.assert_close(values[base + ".mlp_out"], write)
        residual = values[base + ".resid_pre"] + values[base + ".post_attention_layernorm"]
        torch.testing.assert_close(values[base + ".mlp_in"], block.pre_feedforward_layernorm(residual))
        torch.testing.assert_close(values[base], residual + write)
        assert _architecture(model)[base + ".mlp_out"].module == base + ".post_feedforward_layernorm"
        assert not torch.allclose(raw, write)


def test_gemma_capture_uses_actual_softcapped_alternating_patterns(monkeypatch):
    from transformers.models.gemma2 import modeling_gemma2 as implementation
    model = gemma_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    original = implementation.eager_attention_forward
    observed, uncapped = {}, {}
    def attention(module, query, key, value, mask, **kwargs):
        output = original(module, query, key, value, mask, **kwargs)
        observed[module.layer_idx] = output[1].detach()
        uncapped[module.layer_idx] = original(module, query, key, value, mask,
                                             **dict(kwargs, softcap=None))[1].detach()
        return output
    monkeypatch.setattr(implementation, "eager_attention_forward", attention)
    model.config._attn_implementation = "flex_attention"
    with torch.no_grad(), capture(model, list(_architecture(model))) as values:
        assert model.config._attn_implementation == "eager"
        model(input_ids=ids, use_cache=False)
    assert model.config._attn_implementation == "flex_attention"
    for i in range(3):
        pattern = values[f"model.layers.{i}.attn_pattern"]
        torch.testing.assert_close(pattern, observed[i], rtol=0, atol=0)
        assert not torch.allclose(pattern, uncapped[i], atol=1e-4)
        assert pattern.triu(1).count_nonzero() == 0
        if i % 2 == 0:
            assert pattern[..., -1, :-2].count_nonzero() == 0
        else:
            assert (pattern[..., -1, :-2] > 0).all()


def test_gemma_semantic_ablation_uses_residual_write():
    from caliper._ablate import ablate
    model = gemma_model(1)
    with ExitStack() as stack:
        stack.callback(ablate(model, "model.layers.0.mlp_out"))
        values = stack.enter_context(capture(model, ["model.layers.0.mlp_out", "model.layers.0.mlp"]))
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2]]), use_cache=False)
        assert values["model.layers.0.mlp_out"].count_nonzero() == 0
        assert values["model.layers.0.mlp"].count_nonzero() > 0
