"""Independent CPU oracles for dictionary storage and trace arithmetic."""
import pytest
import torch
from torch.nn import functional as F

from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from .fixture import TinyLM, TinyTokenizer
from .test_trace import random_dictionary


def test_encoder_conversion_uses_bounded_blocks(monkeypatch):
    from caliper.interp import _frozen
    from caliper.interp._dictionary import TopKTranscoder
    from types import SimpleNamespace

    layer = TopKTranscoder(torch.arange(60, dtype=torch.float16).reshape(15, 4) / 60,
                          torch.linspace(-.2, .2, 15, dtype=torch.float16),
                          torch.ones(15, 4, dtype=torch.float16), torch.zeros(4, dtype=torch.float16), k=3)
    x = torch.tensor([[[.1, -.2, .3, .4], [-.5, .2, -.3, .4]]], dtype=torch.float32, device="cpu")
    preactivation = F.linear(x, layer.W_enc.float(), layer.b_enc.float()).relu()
    values, indices = preactivation.topk(3, dim=-1)
    expected = torch.zeros_like(preactivation).scatter(-1, indices, values)
    linear = F.linear
    blocks = []
    def bounded_linear(inputs, weight, bias):
        blocks.append(weight.numel())
        assert weight.numel() <= 16
        return linear(inputs, weight, bias)
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 16)
    monkeypatch.setattr(F, "linear", bounded_linear)
    actual = _frozen._trace_features(layer, x, SimpleNamespace(check=lambda stage: None))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
    assert blocks == [16, 16, 16, 12]


@pytest.mark.parametrize("model_dtype,storage_dtype", [
    (torch.float64, torch.float32), (torch.float32, torch.float16),
    (torch.float64, torch.float64), (torch.float32, torch.float32),
])
def test_storage_precision_preserves_forward_and_every_target(model_dtype, storage_dtype):
    model = TinyLM(layers=2, width=12, heads=3, latents=12, k=2, dtype=model_dtype).cpu()
    dictionary = random_dictionary(model, latents=12, k=2).to(dtype=storage_dtype)
    stored = {name: (p.data_ptr(), p.detach().clone()) for name, p in dictionary.named_parameters()}
    ids = torch.tensor([[1, 2, 3]], device="cpu")
    with torch.no_grad():
        expected_logits = model(ids).logits[:, -1]
    path = FrozenPath(model, ids, dictionary)
    tolerance = 1e-9 if model_dtype == torch.float64 else 2e-5
    assert all(torch.count_nonzero(offset) == 0 for offset in path.output_offsets)
    assert path.forward_residual < tolerance
    reads, final = path.replacement(path.embedding, dictionary)
    torch.testing.assert_close(F.linear(final[:, -1], model.lm_head.weight), expected_logits,
                               atol=tolerance, rtol=0)
    expected_features = []
    for index, layer in enumerate(dictionary.layers):
        # A full dense oracle is safe for this tiny dictionary. Production uses bounded copies.
        preactivation = F.linear(path.inputs[index], layer.W_enc.to(model_dtype), layer.b_enc.to(model_dtype))
        values, indices = preactivation.relu().topk(layer.k, dim=-1)
        features = torch.zeros_like(preactivation).scatter(-1, indices, values)
        expected_features.append(features)
        reconstruction = features @ layer.W_dec.to(model_dtype) + layer.b_dec.to(model_dtype)
        torch.testing.assert_close(path.features[index], features, atol=tolerance, rtol=0)
        assert torch.equal(path.features[index] > 0, features > 0)
        torch.testing.assert_close(path.errors[index], path.outputs[index] - reconstruction, atol=tolerance, rtol=0)
        torch.testing.assert_close(reads[index], path.inputs[index], atol=tolerance, rtol=0)
        assert path.errors[index].dtype == model_dtype and not path.errors[index].requires_grad
        if model_dtype == storage_dtype:
            assert torch.equal(path.features[index], dictionary.encode(index, path.inputs[index]))
            assert torch.equal(path.errors[index], path.outputs[index] - dictionary.decode(index, features))

    effects = _DirectEffects(path, dictionary, ids[0].tolist(), TinyTokenizer())
    targets = effects.logit_nodes + effects.feature_nodes
    weights, diagnostics = effects.backward(targets)
    assert len(targets) <= 64
    for column, (target, diagnostic) in enumerate(zip(targets, diagnostics)):
        node = effects.nodes[target]
        if node["kind"] == "feature":
            expected = expected_features[node["layer"]][0, node["position"], node["feature"]]
        else:
            expected = expected_logits[0, node["token_id"]] - expected_logits.mean()
        # Read raw edge sums and explicit biases, without the tracer's residual calculation.
        actual = float(weights[:, column].sum()) + sum(diagnostic["biases"].values())
        assert actual == pytest.approx(float(expected), abs=tolerance, rel=0)
        assert diagnostic["absolute_residual"] < tolerance
    assert any(abs(row["biases"]["encoder"]) > 0 for row in diagnostics)
    assert any(abs(row["biases"]["decoder"]) > 0 for row in diagnostics)
    assert all(row["biases"]["intervention"] == 0 for row in diagnostics)
    assert any(error.count_nonzero() for error in path.errors)

    zero = path.replacement(torch.zeros_like(path.embedding), dictionary)[1]
    positive = path.replacement(3 * path.embedding, dictionary)[1]
    negative = path.replacement(-2 * path.embedding, dictionary)[1]
    torch.testing.assert_close(positive + negative - zero, final, atol=tolerance, rtol=0)
    for name, parameter in dictionary.named_parameters():
        pointer, value = stored[name]
        assert parameter.data_ptr() == pointer and parameter.dtype == storage_dtype
        assert torch.equal(parameter, value)
