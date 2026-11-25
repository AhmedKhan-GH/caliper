"""Independent CPU oracles for ordered output intervention derivatives."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from caliper._controls import Controls
from caliper.interp import Tracer, capture
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary, qwen_for_trace, random_dictionary


@contextmanager
def setup(*, qwen=False):
    model = qwen_for_trace() if qwen else TinyLM(layers=3, width=12, heads=3, latents=16, k=3)
    tokenizer = TinyTokenizer()
    if qwen:
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3] if text == "AB" else [1, 4, 5]])}

            def decode(self, ids):
                return str(ids[0])
        tokenizer = Tokenizer()
    dictionary = random_dictionary(model) if qwen else exact_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    try:
        yield model, tokenizer, dictionary, owner, service
    finally:
        service.remove()


@contextmanager
def fixed_patch_sources(model, tokenizer, service):
    """Keep original clean tensors fixed while the oracle changes target math.

    A clean prompt must not run under the target's frozen norm denominators or
    temporary oracle hooks. Capture it independently from the unmodified model.
    """
    specs = [spec for spec in service.specifications() if spec.get("op", spec.get("kind")) == "patch"]
    saved = [(module, name, getattr(module, name)) for module in model.modules()
             for name in ("last_scale", "last_pattern") if hasattr(module, name)]
    sources = {}
    try:
        for prompt in sorted({spec["source_prompt"] for spec in specs}):
            points = sorted({spec["point"] for spec in specs if spec["source_prompt"] == prompt})
            with service.suspend(), capture(model, points) as values, torch.no_grad():
                model(**tokenizer(prompt), use_cache=False)
                sources.update({(prompt, point): values[point].detach().clone() for point in points})
        for spec in specs:
            torch.testing.assert_close(service.executor._source(spec), sources[spec["source_prompt"], spec["point"]],
                                       rtol=0, atol=0)
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
    with patch.object(service.executor, "_source", lambda spec: sources[spec["source_prompt"], spec["point"]]):
        yield


def feature_at(model, dictionary, ids, layer=1, *, inactive=False):
    blocks = model.model.layers if model.config.model_type == "qwen3_5_text" else model.layers
    reads = []
    handle = blocks[layer].mlp.register_forward_pre_hook(lambda m, a: reads.append(a[0].detach()))
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    values = dictionary.encode(layer, reads[0])[0, 1]
    return int((values == 0).nonzero()[0]) if inactive else int(values.argmax())


def specs_for(case, point, feature, width):
    clamp = {"kind": "clamp", "point": point, "feature": feature, "positions": [1], "value": -.4}
    scale = {"kind": "scale", "point": point, "feature": feature, "positions": [1], "multiplier": -2.}
    add = {"kind": "add", "point": point, "positions": [1], "vector": [.03] * width, "coefficient": 2.}
    patch = {"kind": "patch", "point": point, "positions": [1], "source_prompt": "CD", "source_position": 2}
    ablate = {"kind": "ablate", "point": point}
    # Positioned edits retain the last delta at each group coordinate.
    return {
        "zero": ([dict(clamp, value=0)], 0., 0.),
        "constant": ([clamp], 0., -.4),
        "negative": ([scale], -2., 0.),
        "revived": ([ablate, add, clamp], -1., -.4),
        "reset": ([clamp, scale, add, ablate], 0., 0.),
        "patch_first": ([patch, add, scale, clamp], -1., -.4),
        "patch_last": ([scale, clamp, add, patch], 0., 0.),
    }[case]


@pytest.mark.parametrize("case", ["zero", "constant", "negative", "revived", "reset", "patch_first", "patch_last"])
def test_tiny_output_affine_matches_live_frozen_oracle(case):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specs, coefficient, constant = specs_for(case, "layers.1.mlp_out", feature, 12)
        for spec in specs:
            service.set(spec)
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        original = dictionary.encode(1, path.inputs[1])[0, 1, feature]
        assert original > 0
        torch.testing.assert_close(path.features[1][0, 1, feature], coefficient * original + constant)
        assert all(error.count_nonzero() == 0 for error in path.errors)
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        node_id = f"feature:1:1:{feature}"
        matches = [i for i in effects.feature_nodes if effects.nodes[i]["id"] == node_id]
        if coefficient == constant == 0:
            assert not matches
        else:
            assert len(matches) == 1
            target = effects.target(matches[0])
            torch.testing.assert_close(target.direction, coefficient * dictionary.layers[1].W_enc[feature])
            incoming, diagnostics = effects.backward(matches)
            assert diagnostics[0]["absolute_residual"] < 1e-9
            if coefficient == 0:
                assert incoming.count_nonzero() == 0
                assert diagnostics[0]["biases"]["intervention"] == pytest.approx(constant)

        direction = torch.linspace(-.3, .4, path.embedding.numel(), dtype=path.dtype).reshape_as(path.embedding)
        with fixed_patch_sources(model, tokenizer, service), model.frozen_attention(norms=True), torch.no_grad():
            # Two small perturbations exercise actual live output hooks and the nonlinear TinyLM.
            for sign in (-1, 1):
                embedding = path.embedding + sign * 1e-6 * direction
                expected = model(inputs_embeds=embedding).logits[:, -1]
                _, final = path.replacement(embedding, dictionary)
                actual = model.lm_head(final[:, -1])
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("inactive", [False, True])
def test_constant_clamp_keeps_a_signed_node_without_an_incoming_derivative(inactive):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids, inactive=inactive)
        service.set({"kind": "clamp", "point": "layers.1.mlp_out", "feature": feature,
                     "positions": [1], "value": -.4})
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
        incoming, diagnostics = effects.backward([target])
        assert incoming.count_nonzero() == 0
        assert effects.nodes[target]["activation"] == pytest.approx(-.4)
        torch.testing.assert_close(path.features[1][0, 1, feature], path.embedding.new_tensor(-.4), atol=0, rtol=0)
        assert diagnostics[0]["biases"]["intervention"] == pytest.approx(-.4)
        assert diagnostics[0]["absolute_residual"] < 1e-9
        # A constant feature still writes its decoder vector into later targets.
        outgoing, _ = effects.backward(effects.logit_nodes)
        assert outgoing[target].abs().max() > 1e-6
        assert path.features[1].count_nonzero() > 0
        assert path.errors[1].count_nonzero() == 0


def test_fractional_scale_keeps_the_model_coefficient_precision():
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        service.set({"kind": "scale", "point": "layers.1.mlp_out", "feature": feature,
                     "positions": [1], "multiplier": .1})
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
        torch.testing.assert_close(effects.target(target).direction, .1 * dictionary.layers[1].W_enc[feature],
                                   atol=1e-15, rtol=0)


@pytest.mark.parametrize("legacy_owner", [False, True])
def test_pending_replacement_rows_and_cold_source_cache_keep_applied_edges(legacy_owner, monkeypatch):
    with setup() as (model, tokenizer, dictionary, owner, service):
        if legacy_owner:
            monkeypatch.delattr(Controls, "intervention_service", raising=False)
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        tracer._new_prompt("AB")
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specs, _, _ = specs_for("patch_first", "layers.1.mlp_out", feature, 12)
        # Input edits remain active with output edits and share the clean source prompt.
        specs.insert(0, {"kind": "patch", "point": "layers.1.mlp_in", "positions": [0], "source_prompt": "CD"})
        specs.insert(1, {"kind": "add", "point": "layers.1.resid_pre", "positions": [2],
                         "vector": [.01] * 12, "coefficient": 1.})
        for spec in specs:
            service.set(spec)
        owner.step()
        service.invalidate_source_cache()
        calls = []
        handle = model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            cold = tracer.trace("AB", max_targets=48)
            assert len(calls) == 2  # Target forward and one shared clean source forward.
            sources = dict(service.executor._cache)
            assert len(sources) == 2
            assert tracer.trace("AB", max_targets=48) is cold
            assert len(calls) == 2
            tracer._forward_cache = None
            tracer._cache.clear()
            warm = tracer.trace("AB", max_targets=48)
            assert len(calls) == 3
            assert all(service.executor._cache[key] is value for key, value in sources.items())
            assert warm.nodes == cold.nodes and warm.edges == cold.edges
            service.set(dict(specs[-1], value=2.))
            assert tracer.trace("AB", max_targets=48) is warm
            assert len(calls) == 3
            with service.rows([]):
                excluded = tracer.trace("AB", max_targets=48)
            with service.suspend():
                suspended = tracer.trace("AB", max_targets=48)
            assert excluded.nodes == suspended.nodes and excluded.edges == suspended.edges
            with service.rows([0]):
                selected = tracer.trace("AB", max_targets=48)
            assert selected.nodes == warm.nodes and selected.edges == warm.edges
            owner.step()
            changed = tracer.trace("AB", max_targets=48)
            assert changed.trace_id != warm.trace_id
            assert changed.completeness["max_absolute_residual"] < 1e-9
        finally:
            handle.remove()


@contextmanager
def qwen_oracle(model, dictionary, ids, monkeypatch):
    """Freeze real module factors and use the Transformers recurrence as the oracle."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    records = {}
    blocks = model.model.layers
    with ExitStack() as capture:
        def before(name):
            def hook(module, args):
                records[name] = tuple(value.detach().clone() for value in args if isinstance(value, torch.Tensor))
            return hook

        def after(name):
            def hook(module, args, output):
                records[name] = output.detach().clone()
            return hook

        norms = [model.model.norm]
        for i, block in enumerate(blocks):
            norms.extend((block.input_layernorm, block.post_attention_layernorm))
            capture.callback(block.mlp.register_forward_pre_hook(before((i, "read"))).remove)
            capture.callback(block.mlp.register_forward_hook(after((i, "raw")), prepend=True).remove)
            if block.block_type == "full_attention":
                for name in ("q_proj", "k_proj"):
                    capture.callback(getattr(block.self_attn, name).register_forward_hook(after((i, name))).remove)
            else:
                for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"):
                    capture.callback(getattr(block.linear_attn, name).register_forward_hook(after((i, name))).remove)
                capture.callback(block.linear_attn.norm.register_forward_pre_hook(before((i, "core"))).remove)
        for norm in norms:
            capture.callback(norm.register_forward_pre_hook(before(id(norm))).remove)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    with monkeypatch.context() as patch, ExitStack() as hooks:
        for norm in norms:
            denominator = torch.rsqrt(records[id(norm)][0].float().square().mean(-1, keepdim=True) + norm.eps)
            def norm_forward(x, norm=norm, denominator=denominator):
                return (x.float() * denominator * (1 + norm.weight.float())).to(x.dtype)
            patch.setattr(norm, "forward", norm_forward)
        for i, block in enumerate(blocks):
            layer = dictionary.layers[i]
            original = dictionary.encode(i, records[i, "read"][0]).detach()
            active = original != 0
            error = (records[i, "raw"] - dictionary.decode(i, original)).detach()
            def mlp_forward(x, layer=layer, active=active, error=error):
                activation = F.linear(x, layer.W_enc, layer.b_enc) * active
                return activation @ layer.W_dec + layer.b_dec + error
            patch.setattr(block.mlp, "forward", mlp_forward)
            if block.block_type == "full_attention":
                for name in ("q_proj", "k_proj"):
                    frozen = records[i, name]
                    def fixed_projection(module, args, output, frozen=frozen):
                        return frozen
                    hooks.callback(getattr(block.self_attn, name).register_forward_hook(fixed_projection).remove)
                continue
            module = block.linear_attn
            length = ids.shape[1]
            with torch.no_grad():
                convolution = module.conv1d(records[i, "in_proj_qkv"].transpose(1, 2))[..., :length].transpose(1, 2)
                q, k, _ = F.silu(convolution).split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
                query = q.reshape(1, length, -1, module.head_k_dim)
                key = k.reshape(1, length, -1, module.head_k_dim)
                sigmoid = convolution[..., 2 * module.key_dim:].sigmoid()
                beta = records[i, "in_proj_b"].sigmoid()
                decay = -module.A_log.float().exp() * F.softplus(records[i, "in_proj_a"].float() + module.dt_bias)
                core, gate = records[i, "core"]
                denominator = torch.rsqrt(core.float().square().mean(-1, keepdim=True) + module.norm.variance_epsilon)
                gate = F.silu(gate.float())
            def delta_forward(hidden_states, module=module, query=query, key=key, sigmoid=sigmoid,
                              beta=beta, decay=decay, denominator=denominator, gate=gate, **kwargs):
                x = hidden_states
                projected = module.in_proj_qkv(x).transpose(1, 2)
                convolution = module.conv1d(projected)[..., :length].transpose(1, 2)
                value = (convolution[..., 2 * module.key_dim:] * sigmoid).reshape(1, length, -1, module.head_v_dim)
                core, _ = torch_recurrent_gated_delta_rule(query, key, value, g=decay, beta=beta,
                                                          use_qk_l2norm_in_kernel=True)
                normalized = (core.reshape(-1, module.head_v_dim).float() * denominator).to(x.dtype)
                output = (normalized * module.norm.weight * gate).to(x.dtype).reshape(1, length, -1)
                return module.out_proj(output)
            patch.setattr(module, "forward", delta_forward)
        yield


@pytest.mark.parametrize("layer", [0, 3])
def test_qwen_output_transform_matches_independent_frozen_derivatives(layer, monkeypatch):
    with setup(qwen=True) as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids, layer)
        point = f"model.layers.{layer}.mlp_out"
        specs, _, _ = specs_for("revived", point, feature, model.config.hidden_size)
        for spec in specs:
            service.set(spec)
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        assert path.forward_residual < 2e-5
        assert path.errors[layer].count_nonzero() == 0
        with qwen_oracle(model, dictionary, ids, monkeypatch):
            actual_input = path.embedding.detach().clone().requires_grad_(True)
            actual = model(inputs_embeds=actual_input, use_cache=False).logits[:, -1]
            frozen_input = path.embedding.detach().clone().requires_grad_(True)
            _, final = path.replacement(frozen_input, dictionary)
            frozen = model.lm_head(final[:, -1])
            torch.testing.assert_close(frozen, actual, atol=2e-5, rtol=1e-5)
            direction = torch.linspace(-.2, .3, actual.shape[-1])[None]
            oracle_gradient = torch.autograd.grad((actual * direction).sum(), actual_input)[0]
            frozen_gradient = torch.autograd.grad((frozen * direction).sum(), frozen_input)[0]
            torch.testing.assert_close(frozen_gradient, oracle_gradient, atol=2e-5, rtol=1e-5)
            assert oracle_gradient.abs().max() > 1e-4
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        target = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:{layer}:1:{feature}")
        torch.testing.assert_close(effects.target(target).direction, -dictionary.layers[layer].W_enc[feature])
        incoming, diagnostics = effects.backward([target])
        assert incoming.abs().max() > 1e-5
        assert diagnostics[0]["absolute_residual"] < 2e-5
