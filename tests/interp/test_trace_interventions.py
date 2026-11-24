"""Input interventions retain the actual frozen forward and source derivatives."""
from contextlib import contextmanager

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._trace import _DirectEffects
from .fixture import make_fixture
from .test_trace import exact_dictionary
from .test_output_affine import fixed_patch_sources


@contextmanager
def controlled(point, kind, *, selected=True):
    model, tokenizer = make_fixture(layers=3, width=12, heads=3, latents=16, k=3)
    dictionary = exact_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    spec = {"kind": kind, "point": "layers.1." + point}
    if kind == "patch":
        spec.update(source_prompt="CD", source_position=2)
    elif kind == "add":
        spec.update(vector=[.07 * (i + 1) for i in range(12)], coefficient=.5)
    if kind != "ablate" and selected:
        spec["positions"] = [1]
    try:
        yield model, tokenizer, dictionary, owner, service, spec
    finally:
        service.remove()


@pytest.mark.parametrize("point", ["resid_pre", "mlp_in"])
@pytest.mark.parametrize("kind", ["add", "patch", "ablate"])
@pytest.mark.parametrize("warm", [False, True])
def test_trace_retains_input_effects_and_explicit_constants(point, kind, warm):
    with controlled(point, kind) as (model, tokenizer, dictionary, owner, service, spec):
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        tracer._new_prompt("AB")
        service.set(spec)
        owner.step()
        if not warm:
            service.invalidate_source_cache()
        graph = tracer.trace("AB", max_targets=32)
        assert graph.completeness["forward_residual"] < 1e-9
        assert graph.completeness["max_absolute_residual"] < 1e-9
        targets = graph.completeness["targets"]
        assert all(abs(row["incoming"] + sum(row["biases"].values()) - row["expected"]) < 1e-9
                   for row in targets)
        if kind != "ablate":
            assert any(abs(row["biases"]["intervention"]) > 1e-9 for row in targets)


@pytest.mark.parametrize("point", ["resid_pre", "mlp_in"])
@pytest.mark.parametrize("kind", ["add", "patch", "ablate"])
def test_frozen_input_path_matches_independent_model_perturbations(point, kind):
    with controlled(point, kind) as (model, tokenizer, dictionary, owner, service, spec):
        service.set(spec)
        owner.step()
        ids = tokenizer("AB")["input_ids"]
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        direction = torch.linspace(-.3, .4, path.embedding.numel(), dtype=path.dtype).reshape_as(path.embedding)
        with fixed_patch_sources(model, tokenizer, service), model.frozen_attention(norms=True), torch.no_grad():
            for sign in (-1, 1):
                embedding = path.embedding + sign * 1e-6 * direction
                expected = model(inputs_embeds=embedding).logits[:, -1]
                _, final = path.replacement(embedding, dictionary)
                actual = model.lm_head(final[:, -1])
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("point", ["resid_pre", "mlp_in"])
def test_full_input_patch_removes_upstream_direct_edges(point):
    with controlled(point, "patch", selected=False) as (model, tokenizer, dictionary, owner, service, spec):
        service.set(spec)
        owner.step()
        ids = tokenizer("AB")["input_ids"]
        path = FrozenPath(model, ids, dictionary, interventions=service.specifications())
        effects = _DirectEffects(path, dictionary, ids[0].tolist(), tokenizer)
        targets = [i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 1]
        incoming, diagnostics = effects.backward(targets)
        assert torch.count_nonzero(incoming) == 0
        assert max(row["absolute_residual"] for row in diagnostics) < 1e-9
        assert all(abs(row["biases"]["intervention"] + row["biases"]["encoder"] - row["expected"]) < 1e-9
                   for row in diagnostics)


@pytest.mark.parametrize("patch_first", [False, True])
def test_overlapping_input_effects_keep_live_order_and_row_scope(patch_first):
    with controlled("mlp_in", "patch") as (model, tokenizer, dictionary, owner, service, patch):
        add = {"kind": "add", "point": patch["point"], "positions": [1],
               "vector": [.2] * 12, "coefficient": 1.}
        for spec in ([patch, add] if patch_first else [add, patch]):
            service.set(spec)
        owner.step()
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        graph = tracer.trace("AB", max_targets=32)
        assert graph.completeness["forward_residual"] < 1e-9
        with service.rows([]):
            without_rows = tracer.trace("AB", max_targets=32)
        with service.suspend():
            suspended = tracer.trace("AB", max_targets=32)
        assert without_rows.completeness["forward_residual"] < 1e-9
        assert without_rows.nodes == suspended.nodes
        assert without_rows.edges == suspended.edges


@pytest.mark.parametrize("point", ["model.layers.2.resid_pre", "model.layers.3.mlp_in"])
def test_random_cpu_qwen_input_patch_retains_hybrid_forward(point):
    from .test_trace import qwen_for_trace, random_dictionary
    model = qwen_for_trace()
    dictionary = random_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)

    def tokenizer(text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3] if text == "target" else [1, 4, 5]])}

    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    try:
        service.set({"kind": "patch", "point": point, "source_prompt": "source", "positions": [1]})
        owner.step()
        path = FrozenPath(model, tokenizer("target")["input_ids"], dictionary,
                          interventions=service.specifications())
        assert path.forward_residual < 2e-5
        assert next(model.parameters()).device.type == "cpu"
        x = path.embedding
        first = path.run(x, path.writes, bias_masks=(0., 0., 0., 0.))
        doubled = path.run(2 * x, [2 * y for y in path.writes], bias_masks=(0., 0., 0., 0.))
        for a, b in zip(first[0] + [first[1]], doubled[0] + [doubled[1]]):
            torch.testing.assert_close(b, 2 * a, atol=2e-6, rtol=1e-5)
    finally:
        service.remove()
