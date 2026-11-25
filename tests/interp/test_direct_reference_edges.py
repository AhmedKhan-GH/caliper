"""Small independent oracles for published direct edges and completeness."""
from math import fsum
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._reference_graph import PROFILE
from caliper.interp._trace import _DirectEffects


SOURCE = [1.25, -.0177001953125, 1.8359375, -1.40625]
DIRECTION = [-1.328125, -.71484375, 1.6796875, 1.6953125]


def rounded_dot(left, right):
    """Round an exact sum of four binary products once to the source dtype."""
    return left.new_tensor(fsum(float(a) * float(b) for a, b in zip(left, right)))


class LinearPath:
    """One independent residual write with no model or frozen-path dependency."""

    def __init__(self, embedding, error, direction, *, feature_value=None):
        self.embedding = torch.tensor([[embedding]], dtype=torch.bfloat16)
        self.errors = [self.embedding.new_tensor([[error]])]
        self.direction = self.embedding.new_tensor(direction)
        self.tokens, self.layers, self.width, self.mixer_width = 1, 1, len(direction), len(direction)
        self.device, self.dtype = self.embedding.device, self.embedding.dtype
        self.guard = SimpleNamespace(reserve=lambda *args: None, check=lambda *args: None)
        self.logits = self.embedding.new_tensor([[1.]])
        self.reference_final = self.embedding + self.errors[0]
        self.bias_final = self.embedding.new_zeros(4, self.tokens, self.width)
        self.bias_reads = [self.bias_final.clone()]
        self.raw_reference_input = True
        shape = (self.tokens, int(feature_value is not None))
        activations = self.embedding.new_full(shape, feature_value or 0.)
        self.feature_outputs = [SimpleNamespace(
            indices=torch.zeros(shape, dtype=torch.long), activations=activations,
            coefficients=torch.ones_like(activations), constants=torch.zeros_like(activations))]

    def logit_target(self, token, *, reference=False):
        return self.direction, self.embedding.new_zeros(())

    def run(self, embedding, writes, **kwargs):
        return [embedding], embedding + writes[0]


def effects_for(path, *, decoder=SOURCE):
    original = path.embedding.new_tensor([DIRECTION])
    dictionary = SimpleNamespace(layers=[SimpleNamespace(
        W_dec=path.embedding.new_tensor([decoder]), W_enc_original=original,
        W_enc=-original, b_enc=path.embedding.new_tensor([.125]))])
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "x")
    return _DirectEffects(path, dictionary, [0], tokenizer, profile=PROFILE)


@pytest.mark.parametrize("kind", ["token", "error"])
def test_reference_source_edge_rounds_after_the_contraction(kind):
    zero = [0.] * 4
    path = LinearPath(SOURCE if kind == "token" else zero,
                      SOURCE if kind == "error" else zero, DIRECTION)
    effects = effects_for(path)
    weights, _ = effects.backward(effects.logit_nodes)
    source = path.embedding[0, 0] if kind == "token" else path.errors[0][0, 0]
    expected = rounded_dot(source, path.direction)
    assert float(expected) == -.94921875
    assert float((source * path.direction).sum()) == -.95703125
    row = next(i for i, node in enumerate(effects.nodes) if node["kind"] == kind)
    assert torch.equal(weights[row, 0], expected)


def test_reference_logit_expectation_uses_one_contraction():
    path = LinearPath(SOURCE, [0.] * 4, DIRECTION)
    effects = effects_for(path)
    expected = rounded_dot(path.reference_final[0, 0], path.direction)
    target = effects.target(effects.logit_nodes[0])
    assert torch.equal(target.expected, expected)
    assert effects.nodes[effects.logit_nodes[0]]["activation"] == float(expected)


def test_reference_feature_edge_scales_before_contraction():
    path = LinearPath([0.] * 4, [0.] * 4, DIRECTION, feature_value=1.3046875)
    effects = effects_for(path)
    weights, _ = effects.backward(effects.logit_nodes)
    decoder = path.embedding.new_tensor(SOURCE)
    activation = path.feature_outputs[0].activations[0, 0]
    # Each scaled coordinate has its own BF16 rounding boundary in the source.
    scaled = decoder.new_tensor([float(x) * float(activation) for x in decoder])
    expected = rounded_dot(scaled, path.direction)
    assert expected != rounded_dot(decoder, path.direction) * activation
    assert expected != (scaled * path.direction).sum()
    assert torch.equal(weights[effects.feature_nodes[0], 0], expected)
    # The original encoder supplies the target direction before normalization gain.
    target = effects.target(effects.feature_nodes[0])
    assert torch.equal(target.direction, path.direction)
    assert float(target.encoder_bias) == .125
    assert float(target.expected) == float(activation)


@pytest.mark.parametrize("account", ["sources", "bias_groups", "intervention"])
def test_completeness_preserves_residual_below_bf16_sum_spacing(account):
    path = LinearPath([32.], [.125] if account == "sources" else [0.], [1.])
    if account == "bias_groups":
        path.embedding.zero_()
        path.bias_final[0, 0, 0] = 32.
        path.bias_final[1, 0, 0] = .125
    elif account == "intervention":
        path = LinearPath([0.], [0.], [1.], feature_value=32.)
        path.feature_outputs[0].constants[0, 0] = .125
        path.bias_reads[0][3, 0, 0] = 32.
    effects = effects_for(path, decoder=[0.])
    if account == "intervention":
        effects.dictionary.layers[0].W_enc_original = path.embedding.new_tensor([[1.]])
        effects.dictionary.layers[0].b_enc.zero_()
        targets = effects.feature_nodes
    else:
        targets = effects.logit_nodes
    weights, diagnostics = effects.backward(targets)
    row = diagnostics[0]
    independent = fsum(float(x) for x in weights[:, 0]) + fsum(row["biases"].values()) - row["expected"]
    assert row["expected"] == 32.
    assert independent == .125
    assert row["absolute_residual"] == .125
    assert row["absolute_residual"] > .02
    if account == "sources":
        assert row["incoming"] == 32.125
