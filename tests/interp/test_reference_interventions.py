"""Bounded CPU regressions for original feature coordinates and trace objectives."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from caliper._controls import Controls
from caliper.interp._dictionary import model_identity
from caliper.interp._frozen import FrozenPath
from caliper.interp._gemma_reference import PROFILE, install_reference_profile, profile_metadata
from caliper.interp._intervene import InterventionService
from caliper.interp._output import Output
from caliper.interp._trace import _DirectEffects
from caliper.interp._validate import validate_trace
from .test_gemma_frozen import original_encoder_dictionary
from .test_gemma_reference import reference_fixture
from .test_intervention_replay import hook_objects


class Tokenizer:
    def __call__(self, text, **kwargs):
        ids = [1, 4, 6, 9, 2, 11, 3] if text in ("prompt", "generation") else [1, 3, 11, 2, 9, 6, 4, 7]
        if text == "generation":
            ids += [8, 5, 10, 12]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


@pytest.fixture
def reference():
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    dictionary = original_encoder_dictionary(model, dtype=torch.bfloat16)
    tokenizer = Tokenizer()
    coordinates = {}
    with ExitStack() as stack, torch.no_grad():
        for i, block in enumerate(model.model.layers):
            def read(module, args, i=i):
                x = args[0].float()
                coordinates[i] = (x / (x.square().mean(-1, keepdim=True) + module.eps).sqrt()).bfloat16()
            stack.callback(block.pre_feedforward_layernorm.register_forward_pre_hook(read).remove)
        model(**tokenizer("prompt"))
        layer = dictionary.layers[0]
        layer.W_enc_original.mul_(64)
        gain = (model.model.layers[0].pre_feedforward_layernorm.weight.float() + 1).bfloat16()
        layer.W_enc.copy_((layer.W_enc_original.float() / gain.float()).bfloat16())
        original = F.linear(coordinates[0], layer.W_enc_original, layer.b_enc)
        folded = F.linear(coordinates[0] * gain, layer.W_enc, layer.b_enc)
        candidates = ((original > folded) & (original > 1))[0]
        candidates[0] = False
        position, feature = candidates.nonzero()[0].tolist()
        layer.threshold[feature] = folded[0, position, feature]
        assert original[0, position, feature] > layer.threshold[feature] > 0
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    case = SimpleNamespace(model=model, dictionary=dictionary, tokenizer=tokenizer,
                           controls=controls, service=service, position=position, feature=feature)
    try:
        yield case
    finally:
        service.remove()


def specification(case, kind="clamp", positioned=True):
    spec = {"kind": kind, "point": "model.layers.0.mlp_out", "feature": case.feature}
    spec.update(value=0.) if kind == "clamp" else spec.update(multiplier=-1.)
    if positioned:
        spec["positions"] = [case.position]
    return spec


def independent(case, ids, spec=None, *, rows=None):
    """Apply one source-coordinate edit with raw norm inputs and ordinary hooks."""
    model, layer = case.model, case.dictionary.layers[0]
    recorded = {}
    def read(module, args):
        x = args[0].float()
        x = (x / (x.square().mean(-1, keepdim=True) + module.eps).sqrt()).bfloat16()
        pre = F.linear(x, layer.W_enc_original, layer.b_enc)
        recorded["activation"] = pre * (pre > layer.threshold)

    def write(module, args, output):
        if spec is None:
            return
        active = recorded["activation"][..., spec["feature"]]
        amount = spec["value"] - active if spec["kind"] == "clamp" else (spec["multiplier"] - 1) * active
        vectors = amount[..., None] * layer.W_dec[spec["feature"]]
        result = output.clone()
        for row in range(len(ids)) if rows is None else rows:
            for position in spec.get("positions", range(ids.shape[1])):
                result[row, position] += vectors[row, position]
        return result

    with ExitStack() as stack, torch.no_grad():
        block = model.model.layers[0]
        stack.callback(block.pre_feedforward_layernorm.register_forward_pre_hook(read).remove)
        stack.callback(block.post_feedforward_layernorm.register_forward_hook(write).remove)
        stack.callback(block.post_feedforward_layernorm.register_forward_hook(
            lambda module, args, output: recorded.update(write=output.detach().clone())).remove)
        stack.callback(model.model.norm.register_forward_hook(
            lambda module, args, output: recorded.update(final=output.detach().clone())).remove)
        logits = model(input_ids=ids, use_cache=False).logits
    return logits, recorded


@pytest.mark.parametrize("positioned", [False, True])
@pytest.mark.parametrize("kind", ["clamp", "scale"])
def test_standalone_original_coordinates(reference, positioned, kind):
    case = reference
    spec = specification(case, kind, positioned)
    ids = case.tokenizer("prompt")["input_ids"].repeat(2, 1)
    expected, _ = independent(case, ids, spec, rows=[1])
    hooks = hook_objects(case.model)
    executor = InterventionService(case.model)
    executor.configure(dictionary=case.dictionary, tokenizer=case.tokenizer)
    try:
        hook = executor.install(spec)
        with executor.rows([1]):
            actual = case.model(ids).logits
        assert hook.error is None and executor._published_forward_depth == 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        executor.close()
    assert hook_objects(case.model) == hooks


@pytest.mark.parametrize("prompt", ["prompt", "fork"])
@pytest.mark.parametrize("positioned", [False, True])
def test_original_coordinates_survive_restore_replay_and_prompt_fork(reference, prompt, positioned):
    case = reference
    spec = specification(case, positioned=positioned)
    ids = case.tokenizer(prompt)["input_ids"]
    expected, _ = independent(case, ids, spec)
    case.service.set(spec)
    case.controls.step()
    state = case.service.state_dict()
    case.service.load_state_dict(state, restore_active=True)
    hooks = hook_objects(case.model)
    with case.service.temporary_state(state):
        actual = case.model(ids).logits
    assert hook_objects(case.model) == hooks
    assert case.service.state_dict() == state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("positioned", [False, True])
def test_generation_retains_original_coordinates_at_every_length(reference, positioned):
    case = reference
    spec = specification(case, positioned=positioned)
    case.service.set(spec)
    case.controls.step()
    observed, ids = [], []
    block = case.model.model.layers[0]
    with ExitStack() as stack:
        stack.callback(case.model.register_forward_pre_hook(
            lambda module, args, kwargs: ids.append(kwargs["input_ids"].clone()), with_kwargs=True).remove)
        stack.callback(block.post_feedforward_layernorm.register_forward_hook(
            lambda module, args, output: observed.append(output.detach().clone())).remove)
        result = Output(case.model, case.tokenizer, case.service).generate("generation")
    assert len(observed) == len(ids) == len(result["token_ids"][1]) == 24
    assert ids[-1].shape[1] == ids[0].shape[1] + 23 == 34
    with case.service.suspend():
        for inputs, actual in zip(ids, observed):
            _, expected = independent(case, inputs, spec, rows=[1])
            torch.testing.assert_close(actual, expected["write"], rtol=0, atol=0)
    for hook in case.service.executor._hooks:
        reader = getattr(hook, "group", hook)
        assert reader.live_input is None and not reader.read_stack


def test_generic_slider_uses_index_coordinate_without_a_published_case(reference):
    from caliper.interp._hosted import HostedInference
    case = reference
    ids = case.tokenizer("prompt")["input_ids"]
    _, values = independent(case, ids)
    maximum = float(values["activation"][0, case.position, case.feature])
    run = object.__new__(HostedInference)
    run.model, run.dictionary, run.tokenizer = case.model, case.dictionary, case.tokenizer
    run.controls, run.interventions = case.controls, case.service
    run.guard = run._failure = run._owned_clamp = run._owned_applied_spec = None
    run.selection = {"layer": 0, "position": case.position, "feature": case.feature}
    run.clamp = SimpleNamespace(value=.5)
    run.index = SimpleNamespace(show=lambda layer, feature: {"maximum": maximum})
    spec = dict(specification(case), value=.5 * maximum)
    expected, _ = independent(case, ids, spec)
    assert run._set_clamp(require_commit=True)
    owned = run._owned_record()
    assert owned is not None and owned.applied_spec["value"] == .5 * maximum
    torch.testing.assert_close(case.model(ids).logits, expected, rtol=0, atol=0)
    assert case.service.executor._published_forward_depth == 0


def traced_graph(case, *, features=True):
    ids = case.tokenizer("prompt")["input_ids"]
    path = FrozenPath(case.model, ids, case.dictionary, profile=PROFILE)
    effects = _DirectEffects(path, case.dictionary, ids[0].tolist(), case.tokenizer,
                             profile=PROFILE, max_n_logits=1)
    weights, _ = effects.backward(effects.logit_nodes)
    logit = deepcopy(effects.nodes[effects.logit_nodes[0]])
    nodes = [logit]
    if features:
        index = next(i for i in effects.feature_nodes if effects.nodes[i]["layer"] == 0
                     and effects.nodes[i]["position"] == case.position
                     and effects.nodes[i]["feature"] == case.feature)
        nodes.append(dict(effects.nodes[index], influence=1., effect=float(weights[index, 0])))
    return SimpleNamespace(nodes=nodes, tokens=effects.tokens, layers=path.layers, trace_id="reference-fixture",
                           settings={"profile": PROFILE, "objective": "centered_pre_softcap_logit"})


def objective(case, recorded, token):
    weight = case.model.lm_head.weight
    return float(recorded["final"][0, -1] @ (weight[token] - weight.mean(0)))


def test_reference_validation_uses_traced_objective(reference):
    case = reference
    graph = traced_graph(case, features=False)
    logits, recorded = independent(case, case.tokenizer("prompt")["input_ids"])
    token = graph.nodes[0]["token_id"]
    expected = objective(case, recorded, token)
    assert abs(expected - float(logits[0, -1, token] - logits[0, -1].mean())) > .01
    before = (profile_metadata(case.model), deepcopy(case.model.config.to_dict()), hook_objects(case.model),
              model_identity(case.model))
    result = validate_trace(case.model, case.tokenizer, case.dictionary, case.service, graph, prompt="prompt")
    assert result["baseline_centered_logit"] == expected
    assert before == (profile_metadata(case.model), case.model.config.to_dict(), hook_objects(case.model),
                      model_identity(case.model))


def test_reference_validation_uses_traced_activation_and_clamp(reference):
    case = reference
    graph = traced_graph(case)
    ids = case.tokenizer("prompt")["input_ids"]
    _, baseline = independent(case, ids)
    _, ablated = independent(case, ids, specification(case))
    token = graph.nodes[0]["token_id"]
    expected = objective(case, ablated, token) - objective(case, baseline, token)
    state, hooks = case.service.state_dict(), hook_objects(case.model)
    result = validate_trace(case.model, case.tokenizer, case.dictionary, case.service, graph, prompt="prompt")
    assert result["measurements"][0]["measured"] == expected
    assert result["measurements"][0]["predicted"] == -graph.nodes[1]["effect"]
    assert case.service.state_dict() == state and hook_objects(case.model) == hooks


def test_reference_validation_replay_preserves_live_slider_and_pending_edit(reference):
    case = reference
    graph = traced_graph(case)
    baseline = validate_trace(case.model, case.tokenizer, case.dictionary, case.service, graph, prompt="prompt")
    case.service.set(dict(specification(case), value=-2.))
    case.controls.step()
    case.service.set(dict(specification(case), value=3.))
    state, hooks = case.service.state_dict(), hook_objects(case.model)
    with case.service.temporary_state([]) as replay:
        result = validate_trace(case.model, case.tokenizer, case.dictionary, replay, graph, prompt="prompt")
    assert result == baseline
    assert case.service.state_dict() == state and hook_objects(case.model) == hooks


@pytest.mark.parametrize("positioned", [False, True])
def test_reference_read_cleanup_after_failure_and_nested_forward(reference, positioned):
    from caliper.interp._memory import CleanStop
    case = reference
    spec = specification(case, positioned=positioned)
    ids = case.tokenizer("prompt")["input_ids"]
    expected, _ = independent(case, ids, spec)
    hook = case.service.executor.install(spec)
    reader = getattr(hook, "group", hook)
    nested = False
    def enter(module, args):
        nonlocal nested
        if not nested:
            nested = True
            case.model(case.tokenizer("fork")["input_ids"])
    handle = case.model.model.layers[0].mlp.register_forward_pre_hook(enter)
    try:
        torch.testing.assert_close(case.model(ids).logits, expected, rtol=0, atol=0)
        assert reader.live_input is None and not reader.read_stack
    finally:
        handle.remove()
    def stop(module, args):
        raise CleanStop("fixture cancellation")
    handle = case.model.model.layers[0].mlp.register_forward_pre_hook(stop)
    try:
        with pytest.raises(CleanStop, match="fixture cancellation"):
            case.model(ids)
        assert reader.live_input is None and not reader.read_stack
    finally:
        handle.remove()
    torch.testing.assert_close(case.model(ids).logits, expected, rtol=0, atol=0)
    hook()


@pytest.mark.parametrize("failure_call", [1, 2])
def test_reference_validation_cancellation_retains_live_and_pending_edits(reference, failure_call):
    from caliper.interp._memory import CleanStop
    from caliper.interp import _intervene
    case = reference
    graph = traced_graph(case)
    case.service.set({"kind": "add", "point": "model.layers.0.mlp_out", "vector": [0.] * 24, "coefficient": 1.})
    case.service.set(dict(specification(case), from_step=20))
    case.controls.step()
    state = case.service.state_dict()
    calls = 0
    def stop(module, args):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise CleanStop("fixture validation cancellation")
    handle = case.model.model.norm.register_forward_pre_hook(stop)
    hooks = hook_objects(case.model)
    try:
        with pytest.raises(CleanStop, match="fixture validation cancellation"):
            validate_trace(case.model, case.tokenizer, case.dictionary, case.service, graph, prompt="prompt")
        assert case.service.state_dict() == state and hook_objects(case.model) == hooks
        assert not _intervene._REFERENCE_INPUTS.get() and case.controls.n == 1
    finally:
        handle.remove()


def test_original_coordinate_admission_refuses_a_changed_adapter(reference):
    case = reference
    case.model.config.final_logit_softcapping += 1
    hooks = hook_objects(case.model)
    with pytest.raises(ValueError, match="configuration changed"):
        case.service.executor.install(specification(case))
    assert hook_objects(case.model) == hooks and not case.service.executor._feature_groups
