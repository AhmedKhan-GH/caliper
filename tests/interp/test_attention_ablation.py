"""Attention ablations preserve frozen forwards and remove direct value effects."""
from contextlib import ExitStack, contextmanager

import pytest
import torch

from caliper import _controls
from caliper.interp import Tracer

from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@contextmanager
def active_ablation(model, points):
    owner = _controls.Controls()
    watched = owner.interventions_on(model)
    try:
        for point in points:
            with owner.lock:
                assert owner._intervene({"set": {"point": point, "op": "ablate"}}) is None
        owner.step()
        yield owner
    finally:
        watched.remove()


@pytest.mark.parametrize("points", [
    ["layers.0.self_attn"],
    ["layers.0.self_attn#1"],
])
@pytest.mark.parametrize("legacy_owner", [False, True])
def test_attention_ablation_trace_forward_and_cache(points, monkeypatch, legacy_owner):
    # Use the original four-layer fixture with its default seed and weights.
    model, tokenizer = TinyLM(), TinyTokenizer()
    dictionary = exact_dictionary(model)
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    if legacy_owner:
        monkeypatch.delattr(_controls.Controls, "intervention_service", raising=False)
    with active_ablation(model, points) as owner:
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        path = tracer._forward_cache[1]
        assert path.forward_residual < 1e-9
        assert graph.completeness["max_absolute_residual"] < 1e-9
        assert all(error.count_nonzero() == 0 for error in path.errors)
        with torch.no_grad():
            logits = model(**tokenizer("AB")).logits[:, -1]
        torch.testing.assert_close(path.logits, logits, atol=1e-9, rtol=0)
        assert tracer.trace("AB", max_targets=32) is graph
        cold = Tracer(model, tokenizer, dictionary, controls=owner).trace("AB", max_targets=32)
        assert cold.nodes == graph.nodes and cold.edges == graph.edges
        for name in graph.tensors:
            torch.testing.assert_close(cold.tensors[name], graph.tensors[name], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["module", "head", "all_heads"])
def test_attention_ablation_direct_effect_matches_real_frozen_forward(mode):
    from caliper.interp._frozen import FrozenPath
    from caliper.interp._trace import _DirectEffects

    model, tokenizer = TinyLM(), TinyTokenizer()
    dictionary = exact_dictionary(model)
    ids = tokenizer("AB")["input_ids"]
    name = "layers.1.self_attn"
    points = [name] if mode == "module" else [f"{name}#{h}" for h in (
        range(model.layers[1].self_attn.num_heads) if mode == "all_heads" else [1])]
    with active_ablation(model, points):
        path = FrozenPath(model, ids, dictionary,
                          interventions=[{"op": "ablate", "point": p} for p in points])
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        sources = [i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0
                   and effects.nodes[i]["position"] == 0]
        targets = [i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 1
                   and effects.nodes[i]["position"] == path.tokens - 1]
        weights, diagnostics = effects.backward(targets)
        assert max(d["absolute_residual"] for d in diagnostics) < 1e-9
        assert all(error.count_nonzero() == 0 for error in path.errors)
        # These earlier-position writes can reach this read only through layer 1 attention.
        if mode in ("module", "all_heads"):
            assert weights[sources].count_nonzero() == 0
            assert path.mixers[1].offset.count_nonzero() == 0
        else:
            assert weights[sources].abs().max() > 1e-6

        source = sources[0]
        node = effects.nodes[source]
        vector = node["activation"] * dictionary.layers[0].W_dec[node["feature"]]
        directions = torch.stack([effects.target(i).direction for i in targets])
        epsilon = 1e-4

        def actual_read(scale):
            reads = []
            with ExitStack() as stack:
                stack.enter_context(torch.no_grad())
                stack.enter_context(model.frozen_attention(norms=True))
                for layer, block in enumerate(model.layers):
                    def fixed_write(module, args, result, layer=layer):
                        value = path.outputs[layer].clone()
                        if layer == 0:
                            value[:, 0] += scale * vector
                        return value
                    stack.callback(block.mlp.register_forward_hook(fixed_write).remove)
                stack.callback(model.layers[1].mlp.register_forward_pre_hook(
                    lambda module, args: reads.append(args[0].detach().clone())).remove)
                model(input_ids=ids)
            return reads[0][0, -1] @ directions.T

        measured = (actual_read(epsilon) - actual_read(-epsilon)) / (2 * epsilon)
        torch.testing.assert_close(weights[source], measured, atol=1e-6, rtol=0)
        if mode in ("module", "all_heads"):
            assert measured.count_nonzero() == 0
        # The unablated path uses the same fixture and source direction.
        norm_input = path.norms[1](path.embedding)
        from caliper.interp._frozen import FrozenAttention
        unmasked = FrozenAttention(model.layers[1].self_attn, norm_input, path.mixers[1].pattern)
        perturbation = torch.zeros_like(norm_input)
        perturbation[:, 0] = vector
        assert unmasked(perturbation, 0.)[:, -1].abs().max() > 1e-6


@pytest.mark.parametrize("whole_module", [False, True])
def test_attention_projection_bias_obeys_ablation_scope(whole_module):
    from caliper._ablate import ablate
    from caliper.interp._frozen import _attention_output

    class Attention(torch.nn.Module):
        num_heads = 2

        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(4, 3, dtype=torch.float64)

        def forward(self, x):
            return self.o_proj(x)

    # This separate projection checks nonzero bias without changes to the fixture weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = torch.nn.ModuleDict({"attention": Attention()})
    points = ["attention"] if whole_module else ["attention#0", "attention#1"]
    with ExitStack() as stack:
        for point in points:
            stack.callback(ablate(model, point))
        frozen = _attention_output(model, "attention", "o_proj",
                                   [{"op": "ablate", "point": p} for p in points])
        x = torch.ones(2, 3, 4, dtype=torch.float64, requires_grad=True)
        actual = model["attention"](x)
        torch.testing.assert_close(frozen(x, True), actual, atol=0, rtol=0)
        assert torch.autograd.grad(frozen(x, True).sum(), x)[0].count_nonzero() == 0
        if whole_module:
            assert actual.count_nonzero() == 0
        else:
            assert actual.count_nonzero() > 0
            torch.testing.assert_close(actual, model["attention"].o_proj.bias.expand_as(actual), atol=0, rtol=0)


def test_attention_ablation_trace_respects_suspension_and_selected_rows():
    model, tokenizer = TinyLM(), TinyTokenizer()
    dictionary = exact_dictionary(model)
    with active_ablation(model, ["layers.0.self_attn#1"]) as owner:
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        ablated = tracer.trace("AB", max_targets=32)
        with owner.intervention_service.suspend():
            suspended = tracer.trace("AB", max_targets=32)
            suspended_logits = tracer._forward_cache[1].logits.clone()
            assert suspended is not ablated
        assert tracer.trace("AB", max_targets=32) is ablated
        with owner.intervention_service.rows([]):
            excluded = tracer.trace("AB", max_targets=32)
            torch.testing.assert_close(tracer._forward_cache[1].logits, suspended_logits, atol=0, rtol=0)
            assert excluded.nodes == suspended.nodes and excluded.edges == suspended.edges
        with owner.intervention_service.rows([0]):
            selected = tracer.trace("AB", max_targets=32)
            assert selected.nodes == ablated.nodes and selected.edges == ablated.edges
