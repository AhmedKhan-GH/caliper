"""Random Gemma frozen paths and independent CPU derivative oracles."""
from contextlib import ExitStack, contextmanager

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from caliper.interp._frozen import (FrozenFeatures, FrozenPath, _trace_features,
                                    _trace_reconstruction, _validate_dictionary)
from caliper.interp._trace import _DirectEffects, Tracer
from .test_gemma_family import gemma_model
from .test_trace import random_dictionary


class JumpLayer(nn.Module):
    """Test-only implementation of the dictionary worker's JumpReLU interface."""

    activation = "jumprelu"
    k = None

    def __init__(self, source):
        super().__init__()
        for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
            self.register_buffer(name, getattr(source, name).detach().clone())
        self.register_buffer("threshold", torch.linspace(.03, .2, len(self.b_enc), dtype=self.b_enc.dtype))

    def activate(self, value):
        return value.relu() * (value > self.threshold.to(value.dtype))

    def encode(self, x):
        return self.activate(F.linear(x, self.W_enc, self.b_enc))

    def decode(self, x):
        return x @ self.W_dec + self.b_dec


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    def decode(self, ids):
        return str(ids[0])


def setup(*, jump=False, dtype=torch.float32, softcap=.15):
    model = gemma_model(softcap=softcap)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    dictionary = random_dictionary(model, latents=19, k=4)
    if jump:
        from caliper.interp import _dictionary
        jump_class = getattr(_dictionary, "JumpReLUTranscoder", None)
        layers = [JumpLayer(layer) for layer in dictionary.layers]
        if jump_class is not None:
            layers = [jump_class(**{name: getattr(layer, name) for name in
                                  ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}) for layer in layers]
        dictionary.layers = nn.ModuleList(layers)
        dictionary.metadata.update(activation="jumprelu", k=None)
    dictionary.to(dtype=dtype)
    return model, dictionary, torch.tensor([[1, 2, 3, 4, 5]])


@contextmanager
def frozen_live(model, path):
    """Freeze live norm denominators and attention weights in actual modules."""
    with ExitStack() as stack:
        for i, block in enumerate(model.model.layers):
            for module, norm in ((block.input_layernorm, path.norms[i]),
                                 (block.post_attention_layernorm, path.attention_post_norms[i]),
                                 (block.pre_feedforward_layernorm, path.post_norms[i]),
                                 (block.post_feedforward_layernorm, path.mlp_post_norms[i])):
                stack.enter_context(pytest.MonkeyPatch.context()).setattr(module, "forward", norm)
            attention = block.self_attn
            pattern = path.mixers[i].pattern
            def forward(hidden_states, *, attention=attention, pattern=pattern, **kwargs):
                batch, tokens, _ = hidden_states.shape
                value = attention.v_proj(hidden_states).reshape(batch, tokens, -1, attention.head_dim).transpose(1, 2)
                value = value.repeat_interleave(attention.num_key_value_groups, dim=1)
                mixed = (pattern @ value).transpose(1, 2).reshape(batch, tokens, -1)
                return attention.o_proj(mixed), pattern
            stack.enter_context(pytest.MonkeyPatch.context()).setattr(attention, "forward", forward)
        stack.enter_context(pytest.MonkeyPatch.context()).setattr(model.model.norm, "forward", path.final_norm)
        yield


@pytest.mark.parametrize("jump,dtype", [(False, torch.float32), (True, torch.float32), (True, torch.float16)])
def test_gemma_baseline_replacement_and_independent_write_linearity(jump, dtype):
    model, dictionary, ids = setup(jump=jump, dtype=dtype)
    path = FrozenPath(model, ids, dictionary)
    with torch.no_grad():
        actual = model(input_ids=ids, use_cache=False).logits[:, -1]
    torch.testing.assert_close(path.logits, actual, atol=1e-6, rtol=1e-5)
    assert path.direct_forward_residual < 2e-5
    assert path.replacement_forward_residual < 2e-5
    assert all(error.count_nonzero() for error in path.errors)
    torch.testing.assert_close(path.embedding, model.model.embed_tokens(ids))
    for output in path.feature_outputs:
        assert (output.indices[:, 1:] != output.indices[:, :-1]).all()
    x = torch.linspace(-.03, .04, path.embedding.numel()).reshape_as(path.embedding)
    y = x.flip(-1)
    writes_a = [x * (i + 1) for i in range(path.layers)]
    writes_b = [y * (i - 2) for i in range(path.layers)]
    first = path.run(x, writes_a, bias_masks=(0.,) * 4)
    second = path.run(y, writes_b, bias_masks=(0.,) * 4)
    combined = path.run(1.3 * x - .4 * y,
                        [1.3 * a - .4 * b for a, b in zip(writes_a, writes_b)], bias_masks=(0.,) * 4)
    for a, b, c in zip(first[0] + [first[1]], second[0] + [second[1]], combined[0] + [combined[1]]):
        torch.testing.assert_close(c, 1.3 * a - .4 * b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("softcap", [.15, None])
def test_softcap_tangent_preserves_live_value_derivative_and_centered_bias(softcap):
    model, dictionary, ids = setup(softcap=softcap)
    model.lm_head.bias = nn.Parameter(torch.linspace(-.07, .08, model.config.vocab_size))
    dictionary = random_dictionary(model)
    path = FrozenPath(model, ids, dictionary)
    _, final = path.run(path.embedding, path.writes)
    z = final[:, -1].detach().requires_grad_(True)
    raw = model.lm_head(z)
    actual = raw if softcap is None else softcap * (raw / softcap).tanh()
    affine = path.logits_from_final(z)
    torch.testing.assert_close(affine, actual, atol=1e-6, rtol=1e-5)
    assert path.unembedding.data_ptr() == model.lm_head.weight.data_ptr()
    if softcap is not None:
        assert path.logit_offset.abs().max() > .01
        assert path.logit_scale.min() < .5
    token = int(actual.argmax())
    direction, bias = path.logit_target(token)
    centered = actual[0, token] - actual[0].mean()
    derivative = torch.autograd.grad(centered, z)[0][0]
    torch.testing.assert_close(direction, derivative, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close((z[0] * direction).sum() + bias, centered, atol=1e-6, rtol=1e-5)
    if softcap is not None:
        assert not torch.allclose(direction, path.unembedding[token] - path.unembedding.mean(0))
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())
    _, diagnostics = effects.backward(effects.logit_nodes + effects.feature_nodes[:6])
    assert max(row["absolute_residual"] for row in diagnostics) < 2e-5


@pytest.mark.parametrize("jump", [False, True])
def test_centered_logit_edges_match_live_independent_writes(jump):
    model, dictionary, ids = setup(jump=jump)
    path = FrozenPath(model, ids, dictionary)
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())
    sources = [0, next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0)]
    target = effects.logit_nodes[0]
    weights, diagnostics = effects.backward([target])
    assert diagnostics[0]["absolute_residual"] < 2e-5
    with frozen_live(model, path), ExitStack() as stack:
        magnitude = torch.zeros((), requires_grad=True)
        write_values = list(path.outputs)
        for i, block in enumerate(model.model.layers):
            stack.callback(block.post_feedforward_layernorm.register_forward_hook(
                lambda module, args, output, i=i: write_values[i]).remove)
        for source in sources:
            node = effects.nodes[source]
            embedding = path.embedding
            if node["kind"] == "token":
                perturbation = torch.zeros_like(embedding)
                perturbation[:, node["position"]] = embedding[:, node["position"]]
                embedding = embedding + magnitude * perturbation
            else:
                perturbation = torch.zeros_like(embedding)
                perturbation[:, node["position"]] = (node["activation"] * dictionary.layers[node["layer"]].W_dec[node["feature"]])
                write_values[node["layer"]] = path.outputs[node["layer"]] + magnitude * perturbation
            logits = model(inputs_embeds=embedding, use_cache=False).logits[0, -1]
            objective = logits[effects.nodes[target]["token_id"]] - logits.mean()
            expected = torch.autograd.grad(objective, magnitude)[0]
            torch.testing.assert_close(weights[source, 0], expected, atol=2e-5, rtol=2e-5)
            assert expected.abs() > 1e-7


def test_jumprelu_all_active_features_padding_edits_and_mixed_precision(monkeypatch):
    import caliper.interp._frozen as frozen
    model, dictionary, ids = setup(jump=True, dtype=torch.float16)
    layer = dictionary.layers[0]
    monkeypatch.setattr(frozen, "_ENCODER_BLOCK_ELEMENTS", 32)
    x = torch.linspace(-1., 1., 3 * 16).reshape(1, 3, 16)
    expected = layer.activate(F.linear(x, layer.W_enc.float(), layer.b_enc.float()))
    torch.testing.assert_close(_trace_features(layer, x), expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(_trace_reconstruction(layer, expected),
                               expected @ layer.W_dec.float() + layer.b_dec.float(), atol=1e-7, rtol=1e-6)
    original = torch.zeros(1, 3, 19)
    original[0, 0] = torch.arange(1., 20.)
    original[0, 1, [0, 7]] = torch.tensor([.2, .6])
    specs = [{"op": "scale", "feature": 7, "positions": [1], "multiplier": -2.},
             {"op": "clamp", "feature": 14, "positions": [1, 2], "value": -.4}]
    output = FrozenFeatures(original, layer, specs)
    assert output.indices.shape == (3, 19)
    assert (output.indices[:, 1:] > output.indices[:, :-1]).all()
    actual = torch.zeros_like(original[0]).scatter_add(-1, output.indices, output.activations)
    wanted = original[0].clone()
    wanted[1, 7] *= -2
    wanted[1:, 14] = -.4
    torch.testing.assert_close(actual, wanted)
    column = (output.indices[1] == 7).nonzero()[0, 0]
    assert output.coefficients[1, column] == -2
    empty = FrozenFeatures(torch.zeros_like(original), layer, [])
    assert empty.indices.shape == (3, 0)
    torch.testing.assert_close(empty.evaluate(x, layer), torch.empty(1, 3, 0))
    # Fewer active columns force padding, while every selected index stays unique.
    sparse = FrozenFeatures(original[:, 1:], layer, [])
    assert sparse.indices.shape == (2, 2)
    assert (sparse.indices[:, 1:] != sparse.indices[:, :-1]).all()


@pytest.mark.parametrize("defect", ["shape", "zero", "negative", "nonfinite", "activation", "k"])
def test_jump_dictionary_admission_rejects_invalid_threshold_and_metadata(defect):
    model, dictionary, _ = setup(jump=True, dtype=torch.float16)
    layer = dictionary.layers[0]
    if defect == "shape":
        layer.threshold.data = layer.threshold.data[:1]
    elif defect == "zero":
        layer.threshold[0] = 0
    elif defect == "negative":
        layer.threshold[0] = -.1
    elif defect == "nonfinite":
        layer.threshold[0] = float("nan")
    elif defect == "activation":
        layer.activation = "topk"
    else:
        dictionary.metadata["k"] = 3
    with pytest.raises(ValueError, match="threshold|activation|TopK"):
        _validate_dictionary(model, dictionary)


@pytest.mark.parametrize("kind", ["add", "patch"])
def test_raw_gemma_mlp_interventions_refuse_before_forward(kind, monkeypatch):
    model, dictionary, ids = setup()
    monkeypatch.setattr(model, "forward", lambda **kwargs: pytest.fail("unexpected forward"))
    with pytest.raises(NotImplementedError, match="no exact affine intervention path"):
        FrozenPath(model, ids, dictionary, interventions=[{"point": "model.layers.0.mlp", "kind": kind}])


@pytest.mark.parametrize("key", ["op", "kind"])
def test_raw_gemma_mlp_ablation_refuses_before_identity(key, monkeypatch):
    from caliper.interp import _frozen, _trace
    model, dictionary, ids = setup()
    tracer = Tracer(model, Tokenizer(), dictionary)
    spec = {"point": "model.layers.0.mlp", key: "ablate"}
    monkeypatch.setattr(tracer, "_intervention_state", lambda *_: [spec])

    def refuse(*args, **kwargs):
        pytest.fail("raw Gemma MLP ablation reached identity or model work")

    monkeypatch.setattr(_frozen, "_validate_dictionary", refuse)
    monkeypatch.setattr(_trace, "_validate_dictionary", refuse)
    monkeypatch.setattr(torch.Tensor, "to", refuse)
    monkeypatch.setattr(model, "forward", refuse)
    with pytest.raises(ValueError, match="trace does not support ablation point"):
        tracer.trace("fixture", max_targets=0)
    with pytest.raises(ValueError, match="trace does not support ablation point"):
        FrozenPath(model, ids, dictionary, interventions=[spec])
    assert not tracer._cache and tracer._forward_cache is None
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_gemma_trace_and_forward_settings_admission():
    model, dictionary, _ = setup(jump=True)
    tracer = Tracer(model, Tokenizer(), dictionary)
    graph = tracer.trace("fixture", max_targets=20)
    assert graph.completeness["max_absolute_residual"] < 2e-5
    model.config.final_logit_softcapping = .3
    with pytest.raises(ValueError, match="identity mismatch|forward setting changed"):
        tracer.trace("fixture", max_targets=20)


def test_explicit_gemma_memory_budget_retains_other_family_defaults(monkeypatch):
    import caliper.interp._frozen as frozen
    from caliper.interp._families import model_family
    from .fixture import TinyLM
    model, dictionary, _ = setup()
    default = Tracer(model, Tokenizer(), dictionary)
    assert default.guard.limit == 12_000_000_000
    observed = []
    original = frozen.ProcessMemoryGuard
    def process(limit, device, **kwargs):
        observed.append(limit)
        return original(limit, device, **kwargs)
    monkeypatch.setattr(frozen, "ProcessMemoryGuard", process)
    tracer = Tracer(model, Tokenizer(), dictionary, memory_limit_gb=40)
    assert observed == [40]
    assert tracer.guard.limit == tracer.guard._guard.limit == 40_000_000_000
    with pytest.raises(ValueError, match="at most 40"):
        Tracer(model, Tokenizer(), dictionary, memory_limit_gb=41)
    with pytest.raises(ValueError, match="at most 12"):
        model_family(TinyLM(layers=1)).memory_limit(40)


@pytest.mark.parametrize("kind,point", [("ablate", "mlp_out"), ("ablate", "resid_pre"),
                                        ("add", "mlp_in"), ("patch", "mlp_out"),
                                        ("add", "self_attn.o_proj")])
def test_gemma_semantic_interventions_match_live_output(kind, point):
    from caliper._controls import Controls
    model, dictionary, ids = setup()
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=Tokenizer())
    spec = {"kind": kind, "point": f"model.layers.1.{point}"}
    if kind == "add":
        spec.update(vector=[.02] * 16, coefficient=2., positions=[1])
    if kind == "patch":
        spec.update(source_prompt="fixture", source_position=2, positions=[1])
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        service.set(spec)
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        assert path.forward_residual < 2e-5
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())
        _, diagnostics = effects.backward(effects.logit_nodes + effects.feature_nodes[:5])
        assert max(row["absolute_residual"] for row in diagnostics) < 2e-5
    finally:
        service.remove()


@pytest.mark.parametrize("kind,value,inactive", [("clamp", -.4, False), ("scale", -2., False),
                                                ("clamp", .3, True)])
def test_gemma_feature_edits_at_semantic_residual_write(kind, value, inactive):
    # The runtime worker supplies this separate semantic-read capture to live hooks.
    # This oracle tests the frozen contract without editing the intervention module.
    model, dictionary, ids = setup(jump=True, dtype=torch.float16)
    baseline = FrozenPath(model, ids, dictionary)
    layer = dictionary.layers[1]
    original = _trace_features(layer, baseline.inputs[1])[0, 1]
    feature = int((original == 0).nonzero()[0]) if inactive else int(original.argmax())
    spec = {"op": kind, "point": "model.layers.1.mlp_out", "feature": feature, "positions": [1],
            "value" if kind == "clamp" else "multiplier": value}
    read = []
    def capture_read(module, args):
        read[:] = [args[0]]
    def edit(module, args, output):
        activation = _trace_features(layer, read[0])[0, 1, feature]
        delta = value - activation if kind == "clamp" else (value - 1) * activation
        result = output.clone()
        result[:, 1] += delta * layer.W_dec[feature].float()
        return result
    with ExitStack() as stack:
        block = model.model.layers[1]
        stack.callback(block.mlp.register_forward_pre_hook(capture_read).remove)
        stack.callback(block.post_feedforward_layernorm.register_forward_hook(edit).remove)
        path = FrozenPath(model, ids, dictionary, interventions=[spec])
    expected = torch.tensor(value) if kind == "clamp" else value * original[feature]
    torch.testing.assert_close(path.features[1][0, 1, feature], expected)
    assert path.forward_residual < 2e-5
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())
    target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
    weights, diagnostics = effects.backward([target, effects.logit_nodes[0]])
    assert max(row["absolute_residual"] for row in diagnostics) < 2e-5
    if kind == "clamp":
        assert weights[:, 0].count_nonzero() == 0
        assert diagnostics[0]["biases"]["intervention"] == pytest.approx(value)
    else:
        torch.testing.assert_close(effects.target(target).direction, value * layer.W_enc[feature].float())


def test_dense_jump_sources_refuse_budget_before_node_allocation(monkeypatch):
    from caliper.interp._memory import CleanStop
    model, dictionary, ids = setup(jump=True)
    path = FrozenPath(model, ids, dictionary)
    monkeypatch.setattr(path.guard, "check", lambda stage: path.guard.limit - 1)
    with pytest.raises(CleanStop, match="source nodes allocation"):
        _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())


def test_feature_read_edges_match_live_independent_writes():
    model, dictionary, ids = setup(jump=True)
    path = FrozenPath(model, ids, dictionary)
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer())
    source = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0)
    target = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 1)
    weights, diagnostics = effects.backward([target])
    assert diagnostics[0]["absolute_residual"] < 2e-5
    node = effects.nodes[source]
    target_node = effects.nodes[target]
    amplitude = torch.zeros((), requires_grad=True)
    perturbation = torch.zeros_like(path.embedding)
    perturbation[:, node["position"]] = node["activation"] * dictionary.layers[0].W_dec[node["feature"]]
    reads = []
    with frozen_live(model, path), ExitStack() as stack:
        for i, block in enumerate(model.model.layers):
            def write(module, args, output, i=i):
                return path.outputs[i] + amplitude * perturbation if i == 0 else path.outputs[i]
            stack.callback(block.post_feedforward_layernorm.register_forward_hook(write).remove)
        stack.callback(model.model.layers[1].mlp.register_forward_pre_hook(
            lambda module, args: reads.append(args[0])).remove)
        model(input_ids=ids, use_cache=False)
        actual = dictionary.layers[1].encode(reads[0])[0, target_node["position"], target_node["feature"]]
        derivative = torch.autograd.grad(actual, amplitude)[0]
    torch.testing.assert_close(weights[source, 0], derivative, atol=2e-5, rtol=2e-5)


def test_softcap_centered_rows_use_bounded_blocks(monkeypatch):
    import caliper.interp._frozen as frozen
    model, dictionary, ids = setup()
    monkeypatch.setattr(frozen, "_ENCODER_BLOCK_ELEMENTS", 32)
    path = FrozenPath(model, ids, dictionary)
    expected = (path.logit_scale[:, None] * model.lm_head.weight).mean(0)
    torch.testing.assert_close(path.unembedding_mean, expected, atol=1e-8, rtol=1e-5)
    assert path.forward_residual < 2e-5


def original_encoder_dictionary(model, *, dtype=torch.float32):
    from caliper.interp._dictionary import JumpReLUTranscoder, model_identity
    dictionary = random_dictionary(model, latents=19, k=4)
    # These original rows are independently specified fixture inputs. They do
    # not come from an attempted inverse of an existing converted artifact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(319)
        layers = []
        for block, layer in zip(model.model.layers, dictionary.layers):
            original = torch.randn_like(layer.W_enc) * .03
            gain = 1 + block.pre_feedforward_layernorm.weight.detach().float()
            layers.append(JumpReLUTranscoder(
                W_enc=(original / gain).to(dtype), W_enc_original=original.to(dtype),
                b_enc=layer.b_enc.to(dtype), W_dec=layer.W_dec.to(dtype), b_dec=layer.b_dec.to(dtype),
                threshold=torch.full_like(layer.b_enc, .03).to(dtype)))
    dictionary.layers = nn.ModuleList(layers)
    dictionary.metadata.update(model_hash=model_identity(model), activation="jumprelu", k=None,
        reference_input={"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original",
                         "dtype": str(dtype).removeprefix("torch.")})
    return dictionary


def test_reference_original_encoder_and_pre_softcap_objective():
    model = gemma_model(layers=1)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    dictionary = original_encoder_dictionary(model)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    raw = {}
    handle = model.model.layers[0].pre_feedforward_layernorm.register_forward_pre_hook(
        lambda module, args: raw.update(value=args[0].detach().clone()))
    try:
        path = FrozenPath(model, ids, dictionary, profile="circuit_tracer_82996f14")
    finally:
        handle.remove()
    # Independent formula uses the live norm input and the original source rows.
    x = raw["value"].float()
    x = x / (x.square().mean(-1, keepdim=True) + model.config.rms_norm_eps).sqrt()
    layer = dictionary.layers[0]
    preactivation = F.linear(x, layer.W_enc_original, layer.b_enc)
    expected = preactivation * (preactivation > layer.threshold)
    torch.testing.assert_close(path.features[0], expected)
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer(),
                             profile="circuit_tracer_82996f14", max_n_logits=2)
    target = effects.target(effects.logit_nodes[0])
    expected_direction = model.lm_head.weight[effects.nodes[target.node]["token_id"]]
    expected_direction = expected_direction - model.lm_head.weight.mean(0)
    torch.testing.assert_close(target.direction, expected_direction)
    assert not torch.allclose(target.direction, path.logit_target(effects.nodes[target.node]["token_id"])[0])
    _, diagnostics = effects.backward(effects.logit_nodes + effects.feature_nodes[:2])
    assert max(row["absolute_residual"] for row in diagnostics) < 2e-5


def test_reference_requires_original_encoder_before_forward(monkeypatch):
    model, dictionary, ids = setup(jump=True)
    monkeypatch.setattr(model, "forward", lambda **kwargs: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError, match="original pre-gain encoder artifact"):
        FrozenPath(model, ids, dictionary, profile="circuit_tracer_82996f14")


def test_bfloat16_gain_conversion_cannot_recover_original_encoder():
    original = torch.tensor([1.0049])
    gain = torch.tensor([1.01])
    converted = original / gain
    reconstructed = converted.bfloat16() * gain.bfloat16()
    assert not torch.equal(reconstructed, original.bfloat16())


def test_reference_decoder_scales_before_accumulation_in_bfloat16():
    from types import SimpleNamespace
    from caliper.interp._frozen import _decode_reference
    rows = torch.tensor([[1.125, -.5], [-.625, 2.25], [3., .25]], dtype=torch.bfloat16)
    values = torch.tensor([[.75, 1.5], [-.5, 2.]], dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [1, 2]])
    expected = torch.zeros(2, 2, dtype=torch.bfloat16)
    for position, selected in enumerate(indices):
        for column, feature in enumerate(selected):
            expected[position] += rows[feature] * values[position, column]
    torch.testing.assert_close(_decode_reference(SimpleNamespace(W_dec=rows), indices, values), expected, atol=0, rtol=0)


def test_reference_edges_match_independent_pre_softcap_source_derivatives():
    from caliper.interp._points import capture
    model = gemma_model(layers=2)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    dictionary = original_encoder_dictionary(model)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    # Record independent live inputs. The oracle does not call FrozenNorm,
    # FrozenAttention, FrozenPath.run, or the production decoder helper.
    norm_inputs, baseline_writes = {}, {}
    norms = {name: module for name, module in model.named_modules()
             if name.endswith("norm")}
    pattern_points = [f"model.layers.{i}.attn_pattern" for i in range(2)]
    with ExitStack() as stack, capture(model, pattern_points) as recorded:
        for name, module in norms.items():
            stack.callback(module.register_forward_pre_hook(
                lambda module, args, name=name: norm_inputs.update({name: args[0].detach().clone()})).remove)
        for i, block in enumerate(model.model.layers):
            stack.callback(block.post_feedforward_layernorm.register_forward_hook(
                lambda module, args, output, i=i: baseline_writes.update({i: output.detach().clone()})).remove)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    patterns = [recorded[point].detach() for point in pattern_points]
    embedding_source = model.model.embed_tokens(ids).detach()
    raw_read = norm_inputs["model.layers.0.pre_feedforward_layernorm"].float()
    raw_read = raw_read / (raw_read.square().mean(-1, keepdim=True) + model.config.rms_norm_eps).sqrt()
    layer = dictionary.layers[0]
    preactivation = F.linear(raw_read, layer.W_enc_original, layer.b_enc)
    activations = preactivation * (preactivation > layer.threshold)
    reconstruction = (activations[..., None] * layer.W_dec[None, None]).sum(-2) + layer.b_dec
    errors = baseline_writes[0] - reconstruction
    path = FrozenPath(model, ids, dictionary, profile="circuit_tracer_82996f14")
    effects = _DirectEffects(path, dictionary, ids[0].tolist(), Tokenizer(),
                             profile="circuit_tracer_82996f14", max_n_logits=1)
    weights, _ = effects.backward(effects.logit_nodes)
    token = effects.nodes[effects.logit_nodes[0]]["token_id"]
    feature = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0)
    error = next(i for i, n in enumerate(effects.nodes) if n["kind"] == "error" and n["layer"] == 0)
    with ExitStack() as stack:
        for name, module in norms.items():
            scale = (norm_inputs[name].float().square().mean(-1, keepdim=True) + module.eps).rsqrt()
            gain = 1 + module.weight.detach().float()
            def norm(value, scale=scale, gain=gain):
                return (value.float() * scale * gain).to(value.dtype)
            stack.enter_context(pytest.MonkeyPatch.context()).setattr(module, "forward", norm)
        for block, pattern in zip(model.model.layers, patterns):
            attention = block.self_attn
            def mix(hidden_states, attention=attention, pattern=pattern, **kwargs):
                batch, tokens, _ = hidden_states.shape
                values = attention.v_proj(hidden_states).reshape(batch, tokens, -1, attention.head_dim).transpose(1, 2)
                values = values.repeat_interleave(attention.num_key_value_groups, dim=1)
                mixed = (pattern @ values).transpose(1, 2).reshape(batch, tokens, -1)
                return attention.o_proj(mixed), pattern
            stack.enter_context(pytest.MonkeyPatch.context()).setattr(attention, "forward", mix)
        writes = dict(baseline_writes)
        captured = {}
        for i, block in enumerate(model.model.layers):
            stack.callback(block.post_feedforward_layernorm.register_forward_hook(
                lambda module, args, output, i=i: writes[i]).remove)
        stack.callback(model.model.norm.register_forward_hook(
            lambda module, args, output: captured.update(final=output)).remove)
        for source in (0, feature, error):
            magnitude = torch.zeros((), requires_grad=True)
            node = effects.nodes[source]
            perturbation = torch.zeros_like(embedding_source)
            embedding = embedding_source
            if node["kind"] == "token":
                perturbation[:, node["position"]] = embedding[:, node["position"]]
                embedding = embedding + magnitude * perturbation
            else:
                value = (layer.W_dec[node["feature"]] * activations[0, node["position"], node["feature"]]
                         if node["kind"] == "feature" else errors[0, node["position"]])
                perturbation[:, node["position"]] = value
                writes[0] = baseline_writes[0] + magnitude * perturbation
            model(inputs_embeds=embedding, use_cache=False)
            direction = model.lm_head.weight[token] - model.lm_head.weight.mean(0)
            objective = (captured["final"][0, -1] * direction).sum()
            expected = torch.autograd.grad(objective, magnitude)[0]
            torch.testing.assert_close(weights[source, 0], expected, atol=2e-5, rtol=2e-5)
            writes[0] = baseline_writes[0]


def test_original_encoder_artifact_roundtrip_and_dtype_refusal(tmp_path):
    import json
    from safetensors.torch import save_file
    from caliper.interp._dictionary import load_dictionary, weight_names, weights_hash
    model = gemma_model(layers=1).to(dtype=torch.bfloat16)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    dictionary = original_encoder_dictionary(model, dtype=torch.bfloat16)
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    layer = dictionary.layers[0]
    save_file({name: value.detach().contiguous() for name, value in layer.state_dict().items()},
              tmp_path / "layer_00.safetensors")
    loaded = load_dictionary(tmp_path, model=model)
    assert "W_enc_original" in weight_names(loaded.metadata)
    assert "W_enc_original" in weight_names(loaded.layers[0])
    assert weights_hash(loaded) == weights_hash(dictionary)
    with pytest.raises(ValueError, match="artifact dtype"):
        load_dictionary(tmp_path, model=model, dtype=torch.float32)
    # The legacy artifact cannot silently become an original-coordinate encoder.
    del dictionary.metadata["reference_input"]
    dictionary.metadata["import_source"] = {"kind": "gemma_scope"}
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    with pytest.raises(ValueError, match="converted encoder cast is insufficient"):
        load_dictionary(tmp_path, model=model)
