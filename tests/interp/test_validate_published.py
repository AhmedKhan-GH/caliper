"""CPU checks for live validation beside prepared published actions."""
from copy import deepcopy
from contextlib import nullcontext
from contextvars import Context
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import Tracer, _replication_actions as actions
from caliper.interp._hosted import HostedInference, HostedService
from caliper.interp._memory import CleanStop
from .test_intervention_replay import hook_objects
from .test_reference_interventions import (reference, independent as reference_forward,
                                           objective as reference_objective, specification, traced_graph)
from .test_validate import validation_case


def independent(model, ids, specifications=(), *, patterns=None):
    """Compute fixture outputs from weights without model or intervention hooks."""
    def norm(x, module):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps) * module.weight

    with torch.no_grad():
        x = F.embedding(ids, model.embed_tokens.weight)
        activations = []
        for layer, block in enumerate(model.layers):
            attention = block.self_attn
            coordinate = norm(x, block.input_layernorm)
            batch, length, width = coordinate.shape
            def project(module):
                return F.linear(coordinate, module.weight).view(
                    batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
            q, k, v = (project(module) for module in (attention.q_proj, attention.k_proj, attention.v_proj))
            if patterns is None:
                scores = q @ k.transpose(-1, -2) / attention.head_dim ** .5
                mask = torch.ones(length, length, dtype=torch.bool).triu(1)
                pattern = scores.masked_fill(mask, -torch.inf).softmax(-1)
            else:
                pattern = patterns[f"layers.{layer}.attn_pattern"]
            mixed = (pattern @ v).transpose(1, 2).reshape(batch, length, width)
            x = x + F.linear(mixed, attention.o_proj.weight)
            mlp = block.mlp
            z = F.linear(norm(x, block.post_attention_layernorm), mlp.W_enc, mlp.b_enc).relu()
            values, indices = z.topk(mlp.k, dim=-1)
            active = torch.zeros_like(z).scatter(-1, indices, values)
            activations.append(active.clone())
            output = active @ mlp.W_dec + mlp.b_dec
            for spec in specifications:
                if spec["point"] == f"layers.{layer}.mlp_out":
                    feature = spec["feature"]
                    for position in spec["positions"]:
                        output[:, position] += (spec["value"] - active[:, position, feature, None]) * mlp.W_dec[feature]
            x = x + output
        return F.linear(norm(x, model.norm), model.lm_head.weight), activations


@pytest.fixture
def published_validation(monkeypatch, tmp_path):
    model, tokenizer, dictionary, controls, interventions = validation_case(decoder_scale=8)
    inputs = tokenizer("AB")
    _, active = independent(model, inputs["input_ids"])
    features = active[0][0, 1].topk(8).indices.tolist()
    # Preserve every named action and factor, with coordinates sized for the fixture.
    groups = {name: [(0, features[index % len(features)], 1)]
              for index, name in enumerate(actions.GROUPS)}
    monkeypatch.setattr(actions, "GROUPS", groups)
    monkeypatch.setattr(actions, "PROMPTS", dict(dallas="AB", oakland="CD", shanghai="EF", vancouver="GH"))
    owner = HostedInference("tiny-lm", tmp_path, "AB", resources=(model, tokenizer, dictionary, None, None),
                            replication_profile=actions.PROTOCOL, private_output_directory=tmp_path)
    owner.controls, owner.interventions = controls, interventions
    owner.service = HostedService(owner)
    runtime = actions.PublishedCases(model, tokenizer, interventions, output_directory=tmp_path)
    owner.replication = runtime
    try:
        runtime.prepare()
        yield SimpleNamespace(model=model, tokenizer=tokenizer, dictionary=dictionary, controls=controls,
                              interventions=interventions, runtime=runtime, owner=owner, inputs=inputs)
    finally:
        owner.service.close()
        runtime.close()
        interventions.remove()


def trace(case):
    with actions.attention_patterns(case.model, case.runtime.patterns):
        graph = Tracer(case.model, case.tokenizer, case.dictionary, controls=case.controls).trace(
            "AB", max_targets=32, max_n_logits=1, node_threshold=1., edge_threshold=1.)
    # Early writes change later attention. Their input precedes every fixture group edit.
    graph.nodes[:] = [node for node in graph.nodes if node["kind"] == "logit" or
                      (node["kind"] == "feature" and node["layer"] == 0 and node["position"] == 1)]
    assert any(node["kind"] == "feature" for node in graph.nodes)
    case.owner.service.graph = graph
    case.owner.traced_prompt = "AB"
    case.owner.traced_state = deepcopy(case.interventions.specifications())
    return graph


def validate(case):
    return case.owner.service._execute("trace.validate", {}, case.controls.n)


def centered(logits, token):
    row = logits[0, -1]
    return float(row[token] - row.mean())


@pytest.mark.parametrize("cell", [10, 26])
def test_hosted_validation_matches_live_attention_and_restores_published_state(published_validation, cell):
    case = published_validation
    case.runtime.run(cell)
    graph = trace(case)
    specs = case.owner.traced_state
    baseline, active = independent(case.model, case.inputs["input_ids"], specs)
    # A later slider and a pending edit must survive validation of the trace snapshot.
    feature = int(active[0][0, 1].argmax())
    case.interventions.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": feature,
                            "positions": [1], "value": 7.})
    case.controls.step()
    case.interventions.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": feature,
                            "positions": [1], "value": 9.})
    state, hooks = case.interventions.state_dict(), hook_objects(case.model)
    current = case.model(**case.inputs).logits.detach().clone()
    result = validate(case)
    token = result["token"]
    assert result["baseline_centered_logit"] == pytest.approx(centered(baseline, token), abs=1e-12)
    differences = []
    for measured in result["measurements"]:
        node = next(node for node in graph.nodes if node["kind"] == "feature" and
                    all(node[key] == measured[key] for key in ("layer", "position", "feature")))
        extra = {"point": "layers.0.mlp_out", "feature": node["feature"], "positions": [1],
                 "value": float(active[0][0, 1, node["feature"]]) - node["activation"]}
        live, _ = independent(case.model, case.inputs["input_ids"], [*specs, extra])
        frozen, _ = independent(case.model, case.inputs["input_ids"], [*specs, extra], patterns=case.runtime.patterns)
        expected = centered(live, token) - centered(baseline, token)
        assert measured["measured"] == pytest.approx(expected, abs=1e-12)
        differences.append(abs(centered(live, token) - centered(frozen, token)))
    assert max(differences) > 1e-4
    assert case.interventions.state_dict() == state and hook_objects(case.model) == hooks
    assert not case.runtime._forward_scopes and case.interventions.executor._published_forward_depth == 0
    torch.testing.assert_close(case.model(**case.inputs).logits, current, rtol=0, atol=0)
    case.runtime.run(10)
    torch.testing.assert_close(case.model(**case.inputs).logits[0, -1], case.runtime.baseline_logits, rtol=0, atol=0)


@pytest.mark.parametrize("cell", [14, 16, 18, 20, 24, 26, 29])
def test_every_named_action_keeps_frozen_attention_around_validation(published_validation, cell):
    case = published_validation
    trace(case)
    result = case.runtime.run(cell)
    assert result["case_id"] == f"{actions.PROTOCOL}.cell{cell}"
    specs, _ = case.runtime.case(cell)
    expected, _ = independent(case.model, case.inputs["input_ids"], specs, patterns=case.runtime.patterns)
    live, _ = independent(case.model, case.inputs["input_ids"], specs)
    assert (expected - live).abs().max() > 1e-4
    before = case.model(**case.inputs).logits.detach().clone()
    torch.testing.assert_close(before, expected, rtol=0, atol=1e-12)
    state, hooks = case.interventions.state_dict(), hook_objects(case.model)
    validate(case)
    assert case.interventions.state_dict() == state and hook_objects(case.model) == hooks
    torch.testing.assert_close(case.model(**case.inputs).logits, before, rtol=0, atol=0)
    for layer, block in enumerate(case.model.layers):
        torch.testing.assert_close(block.self_attn.last_pattern, case.runtime.patterns[f"layers.{layer}.attn_pattern"],
                                   rtol=0, atol=0)


@pytest.mark.parametrize("exception", [RuntimeError, CleanStop])
@pytest.mark.parametrize("failure_call", [1, 2])
def test_hosted_validation_error_cleans_clamps_and_attention_scopes(published_validation, exception, failure_call):
    case = published_validation
    trace(case)
    case.runtime.run(26)
    state, hooks = case.interventions.state_dict(), hook_objects(case.model)
    before = case.model(**case.inputs).logits.detach().clone()
    calls = 0
    def fail(module, args):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise exception("fixture validation stop")
    handle = case.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(exception, match="fixture validation stop"):
            validate(case)
        assert calls == failure_call
        assert not actions._LIVE_ATTENTION.get() and not case.runtime._forward_scopes
        assert case.interventions.executor._published_forward_depth == 0
    finally:
        handle.remove()
    assert case.interventions.state_dict() == state and hook_objects(case.model) == hooks
    torch.testing.assert_close(case.model(**case.inputs).logits, before, rtol=0, atol=0)
    assert validate(case)["measurements"]


def test_hosted_validation_guard_refusal_restores_scope(published_validation):
    case = published_validation
    trace(case)
    stages = []
    case.owner.guard = SimpleNamespace(device=torch.device("cpu"), limit=1, peak=0,
                                       check=lambda stage, **kwargs: stages.append(stage))
    state, hooks = case.interventions.state_dict(), hook_objects(case.model)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        validate(case)
    assert "validation admission" in stages
    assert not actions._LIVE_ATTENTION.get() and not case.runtime._forward_scopes
    assert case.interventions.state_dict() == state and hook_objects(case.model) == hooks


def test_nested_live_scope_keeps_other_models_and_contexts_frozen(published_validation, tmp_path):
    case = published_validation
    case.runtime.run(26)
    model, tokenizer, _, _, service = validation_case(decoder_scale=8)
    other = actions.PublishedCases(model, tokenizer, service, output_directory=tmp_path / "other")
    try:
        other.prepare()
        other.run(26)
        frozen = case.model(**case.inputs).logits.detach().clone()
        other_frozen = model(**case.inputs).logits.detach().clone()
        live, _ = independent(case.model, case.inputs["input_ids"], case.interventions.specifications())
        assert (live - frozen).abs().max() > 1e-4
        hooks = hook_objects(case.model)
        # An outer published scope must also yield to validation on this model.
        with actions.attention_patterns(case.model, case.runtime.patterns), actions._live_attention(case.model):
            torch.testing.assert_close(case.model(**case.inputs).logits, live, rtol=0, atol=1e-12)
            torch.testing.assert_close(model(**case.inputs).logits, other_frozen, rtol=0, atol=0)
            torch.testing.assert_close(Context().run(case.model, **case.inputs).logits, frozen, rtol=0, atol=0)
            with pytest.raises(RuntimeError, match="nested scope"):
                with actions._live_attention(case.model), actions._live_attention(model):
                    assert actions._LIVE_ATTENTION.get() == {id(case.model), id(model)}
                    raise RuntimeError("nested scope")
            torch.testing.assert_close(case.model(**case.inputs).logits, live, rtol=0, atol=1e-12)
            torch.testing.assert_close(model(**case.inputs).logits, other_frozen, rtol=0, atol=0)
        assert not actions._LIVE_ATTENTION.get() and hook_objects(case.model) == hooks
        torch.testing.assert_close(case.model(**case.inputs).logits, frozen, rtol=0, atol=0)
    finally:
        other.close()
        service.remove()


@pytest.mark.parametrize("outer_attention", [False, True])
def test_prepared_reference_validation_uses_live_pre_softcap_objective(reference, monkeypatch, tmp_path,
                                                                     outer_attention):
    case = reference
    model, service = case.model, case.service
    graph = traced_graph(case)
    ids = case.tokenizer("prompt")["input_ids"]
    _, baseline = reference_forward(case, ids)
    _, live = reference_forward(case, ids, specification(case))
    token = graph.nodes[0]["token_id"]
    expected = reference_objective(case, live, token) - reference_objective(case, baseline, token)
    monkeypatch.setattr(actions, "PROMPTS", dict(dallas="prompt", oakland="fork", shanghai="fork", vancouver="fork"))
    monkeypatch.setattr(actions, "GROUPS", {name: [(0, case.feature, case.position)] for name in actions.GROUPS})
    stages = []
    guard = SimpleNamespace(device=torch.device("cpu"), limit=12_000_000_000, peak=0,
                            check=lambda stage, **kwargs: stages.append(stage))
    runtime = actions.PublishedCases(model, case.tokenizer, service, guard=guard, output_directory=tmp_path)
    owner = HostedInference("random-gemma-fixture", tmp_path, "prompt",
                            resources=(model, case.tokenizer, case.dictionary, None, guard),
                            replication_profile=actions.PROTOCOL, private_output_directory=tmp_path)
    owner.controls, owner.interventions, owner.replication = case.controls, service, runtime
    owner.service = HostedService(owner)
    owner.service.graph, owner.traced_prompt, owner.traced_state = graph, "prompt", []
    try:
        runtime.prepare()
        with service.suspend(), actions.attention_patterns(model, runtime.patterns):
            _, frozen = reference_forward(case, ids, specification(case))
        # BF16 can round the selected scalar to the same value. Compare the full readout too.
        assert not torch.equal(live["final"], frozen["final"])
        state, hooks = service.state_dict(), hook_objects(model)
        readouts = []
        handle = model.model.norm.register_forward_hook(
            lambda module, args, output: readouts.append(output.detach().clone()))
        try:
            with actions.attention_patterns(model, runtime.patterns) if outer_attention else nullcontext():
                result = owner.service._execute("trace.validate", {}, 0)
        finally:
            handle.remove()
        assert result["baseline_centered_logit"] == reference_objective(case, baseline, token)
        assert result["measurements"][0]["measured"] == expected
        assert len(readouts) == 2
        torch.testing.assert_close(readouts[0], baseline["final"], rtol=0, atol=0)
        torch.testing.assert_close(readouts[1], live["final"], rtol=0, atol=0)
        assert "validation admission" in stages
        assert service.state_dict() == state and hook_objects(model) == hooks
        assert not actions._LIVE_ATTENTION.get() and not runtime._forward_scopes
        assert service.executor._published_forward_depth == 0
        torch.testing.assert_close(model(ids).logits[0, -1], runtime.baseline_logits, rtol=0, atol=0)
    finally:
        owner.service.close()
        runtime.close()
