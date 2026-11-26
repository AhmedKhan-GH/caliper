"""Semantic ablations remove the corresponding derivatives and feature writes."""
from contextlib import ExitStack

import pytest
import torch

from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects

from .fixture import TinyLM, TinyTokenizer
from .test_attention_ablation import active_ablation
from .test_trace import exact_dictionary, random_dictionary


@pytest.mark.parametrize("suffix", ["resid_pre", "mlp_in"])
def test_input_ablation_removes_upstream_direct_effects(suffix):
    model, tokenizer = TinyLM(), TinyTokenizer()
    dictionary = random_dictionary(model)
    ids = tokenizer("AB")["input_ids"]
    point = f"layers.1.{suffix}"
    with active_ablation(model, [point]) as owner:
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        assert tracer.trace("AB", max_targets=32) is graph
        path = tracer._forward_cache[1]
        assert path.forward_residual < 1e-9
        assert graph.completeness["max_absolute_residual"] < 1e-9
        assert path.inputs[1].count_nonzero() == 0
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        targets = [i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 1]
        weights, diagnostics = effects.backward(targets + effects.logit_nodes[:1])
        upstream = [i for i, node in enumerate(effects.nodes) if node["layer"] < 1]
        assert weights[upstream, :-1].count_nonzero() == 0
        if suffix == "resid_pre":
            assert weights[upstream, -1].count_nonzero() == 0
        else:
            # An MLP input ablation preserves the residual bypass to the logit.
            assert weights[upstream, -1].abs().max() > 1e-6
        assert max(d["absolute_residual"] for d in diagnostics) < 1e-9
        for target in diagnostics[:-1]:
            assert target["incoming"] == 0
            assert target["expected"] == pytest.approx(target["biases"]["encoder"], abs=1e-9)

        def actual_read(scale):
            reads = []
            with ExitStack() as stack:
                stack.enter_context(torch.no_grad())
                stack.enter_context(model.frozen_attention(norms=True))
                for layer, block in enumerate(model.layers):
                    def fixed_write(module, args, output, layer=layer):
                        return path.outputs[layer]
                    stack.callback(block.mlp.register_forward_hook(fixed_write).remove)
                stack.callback(model.layers[1].mlp.register_forward_pre_hook(
                    lambda module, args: reads.append(args[0].detach().clone())).remove)
                model(inputs_embeds=(1 + scale) * path.embedding)
            return reads[0]

        measured = (actual_read(1e-4) - actual_read(-1e-4)) / 2e-4
        assert measured.count_nonzero() == 0


@pytest.mark.parametrize("suffix", ["mlp", "mlp_out"])
@pytest.mark.parametrize("imperfect", [False, True])
def test_mlp_output_ablation_has_no_feature_or_error_cancellation(suffix, imperfect):
    model, tokenizer = TinyLM(), TinyTokenizer()
    dictionary = random_dictionary(model) if imperfect else exact_dictionary(model)
    ids = tokenizer("AB")["input_ids"]
    if imperfect:
        assert dictionary.layers[1].b_dec.count_nonzero() > 0
    point = f"layers.1.{suffix}"
    with active_ablation(model, [point]) as owner:
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        path = tracer._forward_cache[1]
        assert path.forward_residual < 1e-9
        assert graph.completeness["max_absolute_residual"] < 1e-9
        assert all(value.count_nonzero() == 0 for value in (
            path.outputs[1], path.features[1], path.errors[1], path.decoder_biases[1]))
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        assert not any(n["kind"] == "feature" and n["layer"] == 1 for n in effects.nodes)
        assert not any(n["kind"] == "feature" and n["layer"] == 1 for n in graph.nodes)
        weights, diagnostics = effects.backward(effects.logit_nodes[:1])
        errors = [i for i, n in enumerate(effects.nodes) if n["kind"] == "error" and n["layer"] == 1]
        assert weights[errors].count_nonzero() == 0
        assert max(d["absolute_residual"] for d in diagnostics) < 1e-9
        # The ablated real MLP has zero derivative even under a nonzero input perturbation.
        value = path.inputs[1].clone().requires_grad_(True)
        actual = model.layers[1].mlp(value)
        assert actual.count_nonzero() == 0
        assert torch.autograd.grad(actual.sum(), value)[0].count_nonzero() == 0
