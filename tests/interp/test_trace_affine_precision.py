"""Mixed storage precision preserves actual affine effects and target accounts."""
from contextlib import ExitStack

import pytest
import torch
from torch.nn import functional as F

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._graph import Graph
from caliper.interp._trace import _DirectEffects
from .fixture import TinyLM, TinyTokenizer
from .test_trace import random_dictionary


def dense_features(layer, inputs):
    preactivation = F.linear(inputs, layer.W_enc.to(inputs.dtype), layer.b_enc.to(inputs.dtype))
    values, indices = preactivation.relu().topk(layer.k, dim=-1)
    return torch.zeros_like(preactivation).scatter(-1, indices, values)


@pytest.mark.parametrize("model_dtype,storage_dtype", [
    (torch.float64, torch.float32), (torch.float32, torch.float16),
    (torch.float64, torch.float64), (torch.float32, torch.float32),
])
@pytest.mark.parametrize("case", ["input_add", "output_add", "empty_output", "clamp", "scale", "patch",
                                  "revived", "ordered"])
def test_affine_effects_preserve_precision_forward_and_every_target(model_dtype, storage_dtype, case, monkeypatch):
    model = TinyLM(layers=2, width=12, heads=3, latents=12, k=2, dtype=model_dtype).cpu()
    tokenizer = TinyTokenizer()
    ids = tokenizer("AB")["input_ids"]
    dictionary = random_dictionary(model, latents=12, k=2).to(dtype=storage_dtype)
    stored = {name: (p.data_ptr(), p.detach().clone()) for name, p in dictionary.named_parameters()}
    tolerance = 1e-9 if model_dtype == torch.float64 else 2e-5
    baseline = FrozenPath(model, ids, dictionary)
    original = dense_features(dictionary.layers[0], baseline.inputs[0])
    feature = int((original[0, 1] == 0).nonzero()[0]) if case == "revived" else int(original[0, 1].argmax())
    common = {"point": "layers.0.mlp_out", "positions": [1]}
    clamp = dict(common, kind="clamp", feature=feature, value=-.4)
    scale = dict(common, kind="scale", feature=feature, multiplier=-2.)
    patch = dict(common, kind="patch", source_prompt="CD", source_position=2)
    add = dict(common, kind="add", vector=[.07 * (i + 1) for i in range(12)], coefficient=.5)
    specs = {
        "input_add": [dict(add, point="layers.0.mlp_in")], "output_add": [add],
        "empty_output": [dict(clamp, positions=[])],
        "clamp": [clamp], "scale": [scale], "patch": [patch],
        "revived": [{"kind": "ablate", "point": common["point"]}, clamp],
        "ordered": [patch, scale, clamp],
    }[case]
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    try:
        for spec in specs:
            service.set(spec)
        owner.step()
        raw = {}
        with ExitStack() as stack:
            for index, block in enumerate(model.layers):
                def record(module, args, output, index=index):
                    raw[index] = output.detach()
                stack.callback(block.mlp.register_forward_hook(record, prepend=True).remove)
            path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        with torch.no_grad():
            actual_logits = model(ids).logits[:, -1]
        if case == "empty_output":
            assert torch.equal(actual_logits, baseline.logits)
        else:
            assert not torch.equal(actual_logits, baseline.logits)
        assert path.forward_residual < tolerance
        reads, final = path.replacement(path.embedding, dictionary)
        torch.testing.assert_close(F.linear(final[:, -1], model.lm_head.weight), actual_logits,
                                   atol=tolerance, rtol=0)
        expected_features = []
        for index, layer in enumerate(dictionary.layers):
            original = dense_features(layer, path.inputs[index])
            transformed = original.clone()
            output_scale = torch.ones_like(original[..., :1])
            if index == 0:
                if case == "revived":
                    transformed.zero_()
                    output_scale.zero_()
                if case in ("patch", "ordered"):
                    transformed[:, 1] = 0
                    output_scale[:, 1] = 0
                if case in ("scale", "ordered"):
                    transformed[:, 1, feature] -= 3 * original[:, 1, feature]
                if case in ("clamp", "revived", "ordered"):
                    transformed[:, 1, feature] += -.4 - original[:, 1, feature]
            expected_features.append(transformed)
            reconstruction = original @ layer.W_dec.to(model_dtype) + layer.b_dec.to(model_dtype)
            expected_error = output_scale * (raw[index] - reconstruction)
            torch.testing.assert_close(path.features[index], transformed, atol=tolerance, rtol=0)
            torch.testing.assert_close(path.errors[index], expected_error, atol=tolerance, rtol=0)
            torch.testing.assert_close(reads[index], path.inputs[index], atol=tolerance, rtol=0)
            decoded = transformed @ layer.W_dec.to(model_dtype)
            torch.testing.assert_close(decoded + path.decoder_biases[index] + expected_error + path.output_offsets[index],
                                       path.outputs[index], atol=tolerance, rtol=0)
            if index != 0 or case in ("input_add", "empty_output"):
                assert torch.count_nonzero(path.output_offsets[index]) == 0
        if case == "output_add":
            expected_offset = torch.zeros_like(path.output_offsets[0])
            expected_offset[:, 1] = .5 * expected_offset.new_tensor(add["vector"])
            torch.testing.assert_close(path.output_offsets[0], expected_offset, atol=tolerance, rtol=0)
        if case in ("clamp", "revived", "ordered"):
            assert path.features[0][0, 1, feature] < 0

        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        targets = effects.logit_nodes + effects.feature_nodes
        weights, diagnostics = effects.backward(targets)
        for column, (target, diagnostic) in enumerate(zip(targets, diagnostics)):
            node = effects.nodes[target]
            expected = (expected_features[node["layer"]][0, node["position"], node["feature"]]
                        if node["kind"] == "feature" else
                        actual_logits[0, node["token_id"]] - actual_logits.mean())
            actual = float(weights[:, column].sum()) + sum(diagnostic["biases"].values())
            assert actual == pytest.approx(float(expected), abs=tolerance, rel=0)
            assert diagnostic["absolute_residual"] < tolerance
            if (case in ("clamp", "revived") and node["kind"] == "feature" and node["layer"] == 0
                    and node["position"] == 1 and node["feature"] == feature):
                assert torch.count_nonzero(weights[:, column]) == 0
                assert diagnostic["biases"]["intervention"] == pytest.approx(-.4, abs=tolerance, rel=0)
        if case == "empty_output":
            assert all(row["biases"]["intervention"] == 0 for row in diagnostics)
        elif case != "scale":
            assert any(abs(row["biases"]["intervention"]) > 1e-9 for row in diagnostics)
        graph = Tracer(model, tokenizer, dictionary, controls=owner).trace("AB", max_targets=64)
        assert graph.completeness["target_count"] == len(targets)
        assert graph.completeness["max_absolute_residual"] < tolerance
        assert graph.completeness["forward_residual"] < tolerance
        for name, parameter in dictionary.named_parameters():
            pointer, value = stored[name]
            assert parameter.data_ptr() == pointer and parameter.dtype == storage_dtype
            assert torch.equal(parameter, value)
    finally:
        service.remove()
