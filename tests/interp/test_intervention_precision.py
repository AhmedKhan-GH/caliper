"""Live feature identity and affine effects at mixed precision rank boundaries."""
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from caliper.interp._validate import validate_trace
from .fixture import TinyLM, TinyTokenizer
from .test_trace import random_dictionary
from .test_trace_affine_precision import dense_features


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["none", "scale", "clamp"])
def test_rank_boundary_uses_one_feature_identity(model_dtype, storage_dtype, kind):
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1, dtype=model_dtype).cpu()
    tokenizer = TinyTokenizer()
    ids = tokenizer("AB")["input_ids"]
    dictionary = random_dictionary(model, latents=4, k=1).to(dtype=storage_dtype)
    initial = FrozenPath(model, ids, dictionary)
    layer = dictionary.layers[0]
    position = 2
    x = initial.inputs[0][0, position]
    # Low precision ties two unit activations. The other row wins in model precision.
    tied = torch.tensor([1., 1., 0., 0.], dtype=storage_dtype).topk(1).indices.item()
    feature = 1 - tied
    column = int(x.abs().argmax())
    with torch.no_grad():
        layer.W_enc.zero_()
        layer.b_enc.fill_(-2.)
        layer.b_enc[:2] = 1.
        layer.W_enc[feature, column] = torch.finfo(storage_dtype).eps / (4 * x[column])
    original = dense_features(layer, initial.inputs[0])
    low = layer.encode(initial.inputs[0].to(storage_dtype))
    assert original[0, position].argmax() == feature
    assert low[0, position].argmax() == tied
    assert original[0, position, feature] > 1
    stored = {name: (p.data_ptr(), p.detach().clone()) for name, p in dictionary.named_parameters()}
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    tolerance = 1e-9 if model_dtype == torch.float64 else 2e-5
    try:
        if kind != "none":
            spec = dict(kind=kind, point="layers.0.mlp_out", feature=feature, positions=[position])
            spec.update({"multiplier": -2.} if kind == "scale" else {"value": -.4})
            service.set(spec)
            owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        expected = original.clone()
        if kind == "scale":
            expected[0, position, feature] *= -2
        elif kind == "clamp":
            expected[0, position, feature] = -.4
        torch.testing.assert_close(path.features[0], expected, atol=tolerance, rtol=0)
        assert torch.count_nonzero(path.output_offsets[0]) == 0
        assert path.forward_residual < tolerance

        # A manual model-precision write checks the live hook independently.
        def edit(module, args, output):
            changed = output.clone()
            changed[:, position] += ((expected - original)[0, position, feature]
                                    * layer.W_dec[feature].to(model_dtype))
            return changed
        with service.suspend(), torch.no_grad():
            handle = model.layers[0].mlp.register_forward_hook(edit)
            try:
                manual = model(ids).logits[:, -1]
            finally:
                handle.remove()
        torch.testing.assert_close(path.logits, manual, atol=tolerance, rtol=0)

        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        targets = effects.logit_nodes + effects.feature_nodes
        _, diagnostics = effects.backward(targets)
        assert all(row["absolute_residual"] < tolerance for row in diagnostics)
        if kind in ("none", "scale"):
            assert all(row["biases"]["intervention"] == 0 for row in diagnostics)
        target_index = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0
                            and effects.nodes[i]["position"] == position
                            and effects.nodes[i]["feature"] == feature)
        target = effects.target(target_index)
        direction = torch.linspace(-.3, .4, path.width, dtype=model_dtype)
        epsilon = .1 if model_dtype == torch.float32 else 1e-4
        plus, minus = path.embedding.clone(), path.embedding.clone()
        plus[0, position] += epsilon * direction
        minus[0, position] -= epsilon * direction
        after = path.run(plus, path.writes)[0][0]
        before = path.run(minus, path.writes)[0][0]
        finite_difference = (path.feature_outputs[0].evaluate(after, layer)
                             - path.feature_outputs[0].evaluate(before, layer)) / (2 * epsilon)
        selected = effects.feature_columns[target_index]
        gradient_input = path.embedding.clone().requires_grad_(True)
        read = path.run(gradient_input, path.writes)[0][0]
        gradient = torch.autograd.grad(read[0, position] @ target.direction, gradient_input)[0]
        derivative = gradient[0, position] @ direction
        if kind != "clamp":
            assert derivative != 0
        derivative_tolerance = 2e-6 if model_dtype == torch.float32 else 1e-9
        assert float(finite_difference[0, position, selected]) == pytest.approx(
            float(derivative), abs=derivative_tolerance)

        # The raw target validates even if display pruning excludes this boundary feature.
        top = effects.logit_nodes[0]
        top_weights, _ = effects.backward([top])
        graph = SimpleNamespace(trace_id="boundary", tokens=effects.tokens,
            nodes=[effects.nodes[top], dict(effects.nodes[target_index], influence=1.,
                                          effect=float(top_weights[target_index, 0]))])
        result = validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        assert result["count"] == 1
        assert result["measurements"][0]["feature"] == feature
        def remove_feature(module, args, output):
            changed = output.clone()
            changed[:, position] -= expected[0, position, feature] * layer.W_dec[feature].to(model_dtype)
            return changed
        with torch.no_grad():
            handle = model.layers[0].mlp.register_forward_hook(remove_feature)
            try:
                ablated = model(ids).logits[0, -1]
            finally:
                handle.remove()
        token = result["token"]
        measured = float(ablated[token] - ablated.mean()) - result["baseline_centered_logit"]
        assert result["measurements"][0]["measured"] == pytest.approx(measured, abs=tolerance)
        graph.nodes[-1]["activation"] += torch.finfo(storage_dtype).eps / 10
        with pytest.raises(ValueError, match="current output contribution"):
            validate_trace(model, tokenizer, dictionary, service, graph, prompt="AB")
        actual = Tracer(model, tokenizer, dictionary, controls=owner).trace("AB", max_targets=32)
        validate_trace(model, tokenizer, dictionary, service, actual, prompt="AB")
        for name, parameter in dictionary.named_parameters():
            pointer, value = stored[name]
            assert parameter.data_ptr() == pointer and torch.equal(parameter, value)
    finally:
        service.remove()
