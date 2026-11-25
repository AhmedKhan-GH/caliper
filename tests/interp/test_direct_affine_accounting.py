"""CPU oracles for explicit decoder bias and rounded completeness."""
from math import fsum
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._reference_graph import PROFILE
from caliper.interp._trace import _DirectEffects


class BiasPath:
    """Two scalar writes with a rounded read after each residual sum."""

    def __init__(self, dtype, *, gain=1.):
        self.dtype, self.device = dtype, torch.device("cpu")
        self.tokens, self.layers, self.width = 1, 2, 1
        self.embedding = torch.zeros(1, 1, 1, dtype=dtype)
        self.errors = [torch.zeros(1, 1, 1, dtype=torch.float64 if dtype == torch.float64
                                   else torch.float32) for _ in range(self.layers)]
        self.gain = self.errors[0].new_tensor([gain])
        self.decoder_biases = [self.embedding.new_tensor([[[32.]]]),
                               self.embedding.new_tensor([[[.125]]])]
        self.guard = SimpleNamespace(reserve=lambda *args: None, check=lambda *args: None)
        self.raw_reference_input = True
        self.direction = self.embedding.new_ones(1)
        self.logits = self.embedding.new_ones(1, 1)
        reads, self.reference_final = self.run(self.embedding, self.errors)
        self.feature_outputs = [SimpleNamespace(
            indices=torch.zeros(1, 1, dtype=torch.long),
            activations=read[0] + .5, coefficients=torch.ones(1, 1, dtype=dtype),
            constants=torch.zeros(1, 1, dtype=dtype)) for read in reads]
        zero = self.embedding.expand(4, -1, -1)
        masks = torch.eye(4, dtype=dtype).reshape(4, 4, 1, 1)
        self.bias_reads, self.bias_final = self.run(zero, [zero] * self.layers, bias_masks=masks)

    def run(self, embedding, writes, *, bias_masks=None, **kwargs):
        decoder = 1. if bias_masks is None else bias_masks[0]
        x, reads = embedding, []
        for write, bias in zip(writes, self.decoder_biases):
            reads.append(x.to(self.dtype))
            x = x + write + decoder * bias
        return reads, (x * self.gain).to(self.dtype)

    def logit_target(self, token, **kwargs):
        return self.direction, self.embedding.new_zeros(())


def effects_for(path):
    layers = [SimpleNamespace(W_dec=path.embedding.new_zeros(1, path.width),
                              W_enc_original=path.direction[None], W_enc=path.direction[None],
                              b_enc=path.embedding.new_tensor([.5])) for _ in range(path.layers)]
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "x")
    return _DirectEffects(path, SimpleNamespace(layers=layers), [0], tokenizer, profile=PROFILE)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_decoder_bias_uses_actual_write_gradients(dtype):
    path = BiasPath(dtype)
    effects = effects_for(path)
    targets = effects.feature_nodes + effects.logit_nodes
    weights, rows = effects.backward(targets)
    # The independent scalar forward has derivatives one for preceding writes.
    # Each target excludes its own decoder bias and all later decoder biases.
    assert [row["biases"]["decoder"] for row in rows] == [0., 32., 32.125]
    assert [row["expected"] for row in rows] == [.5, 32.5, 32. if dtype == torch.bfloat16 else 32.125]
    assert not torch.count_nonzero(weights)
    assert [row["biases"]["encoder"] for row in rows] == [.5, .5, 0.]
    assert rows[-1]["absolute_residual"] == (.125 if dtype == torch.bfloat16 else 0.)
    assert all(set(row["biases"]) == {"decoder", "normalization", "attention", "intervention",
                                      "encoder", "unembedding"} for row in rows)
    split = [effects.backward([target]) for target in targets]
    assert rows == [result[1][0] for result in split]
    assert torch.equal(weights, torch.cat([result[0] for result in split], dim=1))


def test_separate_bf16_bias_forward_can_hide_a_completeness_failure():
    path = BiasPath(torch.bfloat16)
    # An independent forward and backward use complete float32 writes.
    first = torch.tensor([32.], requires_grad=True)
    second = torch.tensor([.125], requires_grad=True)
    output = (first + second).to(torch.bfloat16)
    gradients = torch.autograd.grad(output.sum(), (first, second))
    contribution = fsum(float(g) * float(b) for g, b in zip(gradients, (first.detach(), second.detach())))
    assert float(output.detach()) == 32. and contribution == 32.125
    assert float(path.bias_final[0, 0] @ path.direction) == 32.
    effects = effects_for(path)
    _, rows = effects.backward(effects.logit_nodes)
    assert rows[0]["expected"] == float(output.detach())
    assert rows[0]["biases"]["decoder"] == contribution
    assert rows[0]["absolute_residual"] == .125 > .02


def test_discrete_edge_rounding_alone_prevents_exact_completeness():
    from .test_direct_reference_edges import LinearPath, effects_for as linear_effects
    path = LinearPath([32., .125], [-31.875, 0.], [1., 1.])
    path.decoder_biases = [torch.zeros_like(path.embedding)]
    path.errors[0] = path.errors[0].float()
    effects = linear_effects(path, decoder=[0., 0.])
    embedding = path.embedding.detach().requires_grad_(True)
    write = path.errors[0].detach().requires_grad_(True)
    # Every forward operation and derivative here is exact, including the cast.
    value = (embedding + write).to(torch.bfloat16) @ path.direction
    gradients = torch.autograd.grad(value.sum(), (embedding, write))
    exact = [fsum(float(a) * float(b) for a, b in zip(source.flatten(), gradient.flatten()))
             for source, gradient in zip((embedding.detach(), write.detach()), gradients)]
    assert exact == [32.125, -31.875] and fsum(exact) == float(value.detach()) == .25
    weights, rows = effects.backward(effects.logit_nodes)
    assert weights[:, 0].tolist() == [32., -31.875, 0.]
    assert rows[0]["expected"] == .25
    assert rows[0]["incoming"] == .125
    assert not any(rows[0]["biases"].values())
    assert rows[0]["absolute_residual"] == .125 > .02


def test_decoder_contraction_retains_the_recorded_gradient_precision():
    path = BiasPath(torch.bfloat16, gain=1.00390625)
    first = torch.tensor([32.], requires_grad=True)
    second = torch.tensor([.125], requires_grad=True)
    output = ((first + second) * 1.00390625).to(torch.bfloat16)
    gradients = torch.autograd.grad(output.sum(), (first, second))
    assert all(float(gradient) == 1.00390625 for gradient in gradients)
    actual = fsum(float(g) * float(b) for g, b in zip(gradients, (first.detach(), second.detach())))
    rounded = fsum(float(g.bfloat16()) * float(b) for g, b in zip(gradients, (first.detach(), second.detach())))
    assert actual == 32.25048828125 and rounded == 32.125
    effects = effects_for(path)
    _, rows = effects.backward(effects.logit_nodes)
    assert rows[0]["biases"]["decoder"] == actual
    assert rows[0]["expected"] == float(output.detach()) == 32.25
    assert rows[0]["absolute_residual"] == .00048828125


def test_decoder_correction_preserves_the_other_accounting_channels():
    path = BiasPath(torch.bfloat16)
    # Distinct sentinels test the existing channel layout, not another forward oracle.
    path.bias_final[1:, 0, 0] = path.embedding.new_tensor([.25, .5, 1.])
    path.logit_target = lambda *args, **kwargs: (path.direction, path.embedding.new_tensor(2.))
    effects = effects_for(path)
    _, rows = effects.backward(effects.logit_nodes)
    assert rows[0]["expected"] == 34. and rows[0]["incoming"] == 0.
    assert rows[0]["biases"] == {"decoder": 32.125, "normalization": .25, "attention": .5,
                                 "intervention": 1., "encoder": 0., "unembedding": 2.}
    assert rows[0]["absolute_residual"] == 1.875
