"""Attention output scope and direct edge oracles on small CPU decoders."""
from contextlib import ExitStack

import pytest
import torch

from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from .test_output_affine import setup, feature_at, specs_for, fixed_patch_sources


def direct_oracle(model, dictionary, ids, path, effects, source, target, *, coefficient=1., constant=0.):
    """Measure one direct source edge through actual modules with independent MLP writes."""
    blocks = model.layers
    source_node, target_node = effects.nodes[source], effects.nodes[target]
    magnitude = torch.zeros((), dtype=path.dtype, requires_grad=True)
    perturbation = torch.zeros_like(path.embedding)
    if source_node["kind"] == "token":
        position = source_node["position"]
        perturbation[:, position] = path.embedding[:, position]
        embedding = path.embedding + magnitude * perturbation
    else:
        embedding = path.embedding
        perturbation[:, source_node["position"]] = (source_node["activation"] *
            dictionary.layers[source_node["layer"]].W_dec[source_node["feature"]])
    reads = []
    with ExitStack() as stack:
        for layer, block in enumerate(blocks):
            def write(module, args, output, layer=layer):
                value = path.outputs[layer]
                if source_node["kind"] == "feature" and source_node["layer"] == layer:
                    value = value + magnitude * perturbation
                return value
            stack.callback(block.mlp.register_forward_hook(write).remove)
        if target_node["kind"] == "feature":
            stack.callback(blocks[target_node["layer"]].mlp.register_forward_pre_hook(
                lambda module, args: reads.append(args[0])).remove)
        logits = model(inputs_embeds=embedding, use_cache=False).logits
        if target_node["kind"] == "logit":
            row = logits[0, -1]
            objective = row[target_node["token_id"]] - row.mean()
        else:
            activation = dictionary.encode(target_node["layer"], reads[0])[0, target_node["position"], target_node["feature"]]
            objective = coefficient * activation + constant
        measured = torch.autograd.grad(objective, magnitude, allow_unused=True)[0]
    return torch.zeros_like(magnitude) if measured is None else measured


@pytest.mark.parametrize("case", ["constant", "revived", "patch_first"])
def test_tiny_incoming_and_outgoing_edges_match_module_derivatives(case):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        feature = feature_at(model, dictionary, ids)
        specifications, coefficient, constant = specs_for(case, "layers.1.mlp_out", feature, 12)
        for spec in specifications:
            service.set(spec)
        owner.step()
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        node = next(i for i in effects.feature_nodes if effects.nodes[i]["id"] == f"feature:1:1:{feature}")
        logit = effects.logit_nodes[0]
        weights, diagnostics = effects.backward([node, logit])
        with fixed_patch_sources(model, tokenizer, service), model.frozen_attention(norms=True):
            incoming = direct_oracle(model, dictionary, ids, path, effects, 0, node,
                                     coefficient=coefficient, constant=constant)
            outgoing = direct_oracle(model, dictionary, ids, path, effects, node, logit)
        torch.testing.assert_close(weights[0, 0], incoming, atol=1e-9, rtol=0)
        torch.testing.assert_close(weights[node, 1], outgoing, atol=1e-9, rtol=0)
        assert abs(float(outgoing)) > 1e-6
        assert max(row["absolute_residual"] for row in diagnostics) < 1e-9


def attention_specs(case, name, projection, width):
    add = {"kind": "add", "point": name, "positions": [1], "vector": [.02] * width, "coefficient": 2.}
    patch = {"kind": "patch", "point": name, "positions": [1], "source_prompt": "CD", "source_position": 2}
    # These bounded cases distinguish structural hook order from installation order.
    return {
        "parent_reset": [{"kind": "ablate", "point": name}, dict(add, point=projection)],
        "child_reset": [add, {"kind": "ablate", "point": projection}],
        "patch_then_add": [patch, add, dict(add, point=projection, positions=[0, 2])],
        "add_then_patch": [add, patch, dict(add, point=projection, positions=[0, 2])],
    }[case]


@pytest.mark.parametrize("case", ["parent_reset", "child_reset", "patch_then_add", "add_then_patch"])
def test_tiny_attention_output_order_and_position_derivatives(case):
    with setup() as (model, tokenizer, dictionary, owner, service):
        ids = tokenizer("AB")["input_ids"]
        # Populate ordinary module widths through the installed shape recorder.
        with torch.no_grad():
            model(input_ids=ids)
        name = "layers.1.self_attn"
        for spec in attention_specs(case, name, name + ".o_proj", 12):
            service.set(spec)
        owner.step()
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        path = tracer._forward_cache[1]
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        source = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0
                      and effects.nodes[i]["position"] == 0)
        targets = [next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 1
                        and effects.nodes[i]["position"] == position) for position in (1, 2)]
        weights, _ = effects.backward(targets)
        # A nested clean source can update fixture last_* fields. Record target constants with a warm source.
        with torch.no_grad():
            target_logits = model(input_ids=ids).logits[:, -1]
        torch.testing.assert_close(target_logits, path.logits, atol=1e-9, rtol=0)
        with fixed_patch_sources(model, tokenizer, service), model.frozen_attention(norms=True):
            expected = torch.stack([direct_oracle(model, dictionary, ids, path, effects, source, target)
                                    for target in targets])
        torch.testing.assert_close(weights[source], expected, atol=1e-9, rtol=0)
        assert weights[source, 0] == 0  # Position 1 has no attention value derivative in each case.
        if "patch" in case:
            assert weights[source, 1].abs() > 1e-6
        else:
            assert weights[source, 1] == 0
        assert graph.completeness["max_absolute_residual"] < 1e-9
        assert tracer.trace("AB", max_targets=32) is graph
        assert all(error.count_nonzero() == 0 for error in path.errors)


@pytest.mark.parametrize("point", ["layers.0.self_attn.v_proj", "layers.0", "norm"])
def test_unsupported_module_forms_refuse_before_forward(point, monkeypatch):
    with setup() as (model, tokenizer, dictionary, owner, service):
        service.set({"kind": "ablate", "point": point})
        owner.step()
        def forbidden(*args, **kwargs):
            raise AssertionError("unsupported intervention must fail before a forward")
        monkeypatch.setattr(model, "forward", forbidden)
        with pytest.raises(ValueError, match="trace does not support ablation point"):
            Tracer(model, tokenizer, dictionary, controls=owner).trace("AB", max_targets=0)
