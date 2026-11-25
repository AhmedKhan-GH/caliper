"""Published BF16 forward values and independent frozen derivatives on CPU."""
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from caliper.interp._frozen import FrozenPath, _decode_reference
from caliper.interp._gemma_reference import PROFILE, install_reference_profile
from caliper.interp._points import capture
from .test_gemma_frozen import original_encoder_dictionary


@pytest.fixture
def reference_case():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    config = Gemma2Config(vocab_size=37, hidden_size=24, intermediate_size=40,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, max_position_embeddings=32, sliding_window=3,
        query_pre_attn_scalar=6, attn_logit_softcapping=1.7,
        final_logit_softcapping=30., attention_bias=False,
        hidden_activation="gelu_pytorch_tanh", use_cache=False)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = Gemma2ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.copy_(torch.linspace(-.373, .713, parameter.numel()))
                else:
                    parameter.mul_(5)
        model.bfloat16()
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    install_reference_profile(model, PROFILE)
    dictionary = original_encoder_dictionary(model, dtype=torch.bfloat16)
    ids = torch.tensor([[1, 4, 6, 9, 2, 11, 3]])
    norm_inputs = {}
    points = [f"model.layers.{i}.attn_pattern" for i in range(6)]
    with ExitStack() as stack, capture(model, points) as patterns:
        for name, module in model.named_modules():
            if name.endswith("norm"):
                def record(module, args, name=name):
                    norm_inputs[name] = args[0].detach().clone()
                stack.callback(module.register_forward_pre_hook(record).remove)
        path = FrozenPath(model, ids, dictionary, profile=PROFILE)
    return SimpleNamespace(model=model, dictionary=dictionary, ids=ids, path=path,
                           norm_inputs=norm_inputs, patterns=patterns)


def independent_forward(case, embedding, writes=None):
    """Use pinned RMS division, complete writes, and sparse activation deltas."""
    model, path, dictionary = case.model, case.path, case.dictionary
    def norm(value, name, *, pure=False):
        module = model.get_submodule(name)
        recorded = case.norm_inputs[name].float()
        denominator = (recorded.square().mean(-1, keepdim=True) + module.eps).sqrt()
        normalized = (value.float() / denominator).to(module.weight.dtype)
        return normalized if pure else normalized * (1 + module.weight.float()).to(module.weight.dtype)
    x, reads = embedding, []
    for i, block in enumerate(model.model.layers):
        base = f"model.layers.{i}"
        value = block.self_attn.v_proj(norm(x, base + ".input_layernorm"))
        value = value.reshape(*x.shape[:-1], 2, 6).transpose(1, 2).repeat_interleave(2, dim=1)
        mixed = (case.patterns[base + ".attn_pattern"] @ value).transpose(1, 2).reshape_as(x)
        x = x + norm(block.self_attn.o_proj(mixed), base + ".post_attention_layernorm")
        read = norm(x, base + ".pre_feedforward_layernorm", pure=True)
        reads.append(read)
        if writes is None:
            layer = dictionary.layers[i]
            baseline = F.linear(path.inputs[i], layer.W_enc_original, layer.b_enc)
            active = baseline > layer.threshold
            current = F.linear(read, layer.W_enc_original, layer.b_enc)
            differences = (current - baseline) * active
            delta = torch.zeros_like(x, dtype=layer.W_dec.dtype)
            # Keep the recorded support at zero delta to retain its derivative.
            positions, features = active[0].nonzero(as_tuple=True)
            scaled = differences[:, positions, features, None] * layer.W_dec[features]
            delta.index_add_(1, positions, scaled)
            write = path.outputs[i] + delta
        else:
            write = writes[i]
        x = x + write
    return reads, norm(x, "model.norm")


def test_reference_baseline_preserves_every_recorded_value(reference_case):
    case, path = reference_case, reference_case.path
    assert path.forward_residual == 0
    assert path.embedding.dtype == torch.bfloat16
    assert all(write.dtype == torch.float32 for write in path.outputs)
    assert all(value.dtype == torch.bfloat16 for value in path.inputs)
    assert path.forward_residuals == {"direct_reads": [0.] * 6, "direct_logits": 0.,
                                     "replacement_reads": [0.] * 6, "replacement_logits": 0.}
    for write, output in zip(path.writes, path.outputs):
        assert write.data_ptr() == output.data_ptr()
    reads, final = independent_forward(case, path.embedding, path.outputs)
    for actual, expected in zip(reads, path.inputs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(final, path.reference_final, atol=0, rtol=0)
    reconstructed_residuals = []
    for i, layer in enumerate(case.dictionary.layers):
        preactivation = F.linear(path.inputs[i], layer.W_enc_original, layer.b_enc)
        activations = preactivation * (preactivation > layer.threshold)
        positions, features = activations[0].nonzero(as_tuple=True)
        reconstruction = torch.zeros_like(path.outputs[i][0], dtype=layer.W_dec.dtype)
        reconstruction.index_add_(0, positions, layer.W_dec[features] * activations[0, positions, features, None])
        reconstruction = reconstruction[None] + layer.b_dec
        # Graph errors subtract the rounded BF16 reconstruction from float32 writes.
        torch.testing.assert_close(path.errors[i], path.outputs[i] - reconstruction, atol=0, rtol=0)
        torch.testing.assert_close(path.decoder_biases[i], layer.b_dec.expand_as(path.decoder_biases[i]), atol=0, rtol=0)
        reconstructed_residuals.append(float((reconstruction + path.errors[i] - path.outputs[i]).abs().max()))
        output = path.feature_outputs[i]
        torch.testing.assert_close(output.evaluate(path.inputs[i], layer, original=True)[0],
                                   output.activations, atol=0, rtol=0)
    # Float32 errors recover these BF16 writes without a second BF16 rounding.
    assert reconstructed_residuals == [0.] * 6


@pytest.mark.parametrize("replacement", [False, True])
def test_reference_perturbations_and_derivatives_match_independent_source(reference_case, replacement):
    case, path = reference_case, reference_case.path
    inputs = [path.embedding, *path.outputs]
    actual = [value.clone().requires_grad_() for value in inputs]
    expected = [value.clone().requires_grad_() for value in inputs]
    # A nonzero perturbation tests the delta path beyond baseline equality.
    perturbation = torch.linspace(-.125, .125, path.width).bfloat16()
    observed = (path.replacement(actual[0] + perturbation, case.dictionary) if replacement else
                path.run(actual[0] + perturbation, actual[1:]))
    wanted = independent_forward(case, expected[0] + perturbation, None if replacement else expected[1:])
    for first, second in zip(observed[0] + [observed[1]], wanted[0] + [wanted[1]]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    direction, _ = path.logit_target(3, reference=True)
    first = (observed[1][0, -1] * direction).sum()
    second = (wanted[1][0, -1] * direction).sum()
    actual_grad = torch.autograd.grad(first, actual, allow_unused=True)
    expected_grad = torch.autograd.grad(second, expected, allow_unused=True)
    for first, second in zip(actual_grad, expected_grad):
        if first is None or second is None:
            assert first is second
        else:
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert actual_grad[0].count_nonzero() > 0


def test_zero_reference_delta_retains_decoder_derivative():
    layer = SimpleNamespace(W_dec=torch.tensor([[1.25, -.5], [-.75, 2.]], dtype=torch.bfloat16))
    activation = torch.zeros(1, 1, 2, dtype=torch.bfloat16, requires_grad=True)
    decoded = _decode_reference(layer, torch.tensor([[0, 1]]), activation, include_zeros=True)
    assert decoded.count_nonzero() == 0
    decoded.sum().backward()
    torch.testing.assert_close(activation.grad[0, 0], layer.W_dec.sum(-1), atol=0, rtol=0)


def test_published_softcap_forward_avoids_tangent_roundtrip():
    path = FrozenPath.__new__(FrozenPath)
    path.reference = True
    path.device = torch.device("cpu")
    path.family = SimpleNamespace(name="gemma2")
    path.guard = SimpleNamespace(check=lambda stage: None)
    path.width = 1
    path.unembedding = torch.linspace(-60, 60, 257).bfloat16()[:, None]
    path.unembedding_bias = None
    final = torch.ones(1, 1, dtype=torch.bfloat16)
    path.embedding = final[:, None]
    path._freeze_logits(SimpleNamespace(config=SimpleNamespace(final_logit_softcapping=30.)), final, lambda stage: None)
    raw = F.linear(final, path.unembedding)
    expected = 30 * (raw / 30).tanh()
    old_tangent = raw * path.logit_scale + path.logit_offset
    assert float((old_tangent - expected).abs().max()) > .02
    torch.testing.assert_close(path.logits_from_final(final), expected, atol=0, rtol=0)
    for token in (0, 128, 256):
        direction, bias = path.logit_target(token, reference=True)
        torch.testing.assert_close(direction, path.unembedding[token] - path.unembedding.mean(0), atol=0, rtol=0)
        assert bias == 0


def test_reference_edges_keep_each_source_gradient_dtype(reference_case, monkeypatch):
    from caliper.interp._trace import _DirectEffects
    case, path = reference_case, reference_case.path
    effects = _DirectEffects(path, case.dictionary, case.ids[0].tolist(),
                             SimpleNamespace(decode=lambda *args, **kwargs: "x"), profile=PROFILE)
    index = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 2)
    node = effects.nodes[index]
    direction = case.dictionary.layers[node["layer"]].W_enc_original[node["feature"]]
    sources = [torch.zeros_like(value).requires_grad_() for value in [path.embedding, *path.outputs]]
    reads, _ = independent_forward(case, sources[0], sources[1:])
    objective = (reads[node["layer"]][0, node["position"]] * direction).sum()
    expected = torch.autograd.grad(objective, sources, allow_unused=True)
    original = torch.autograd.grad
    observed = []
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result
    monkeypatch.setattr(torch.autograd, "grad", record)
    effects.backward([index])
    actual, = observed
    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == torch.float32
    for first, second in zip(actual, expected):
        if first is None or second is None:
            assert first is second
        else:
            torch.testing.assert_close(first, second, atol=0, rtol=0)
