"""Published attention policy for live ablations on small CPU models."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.nn import functional as F

from caliper._adapter import TORCH
from caliper._controls import Controls
from caliper.interp import _replication_actions as published
from caliper.interp._output import Output, prompt_inputs
from .fixture import make_fixture, TinyTokenizer
from .test_intervene import dictionary_for


def reference_fixture():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from caliper.interp._gemma_reference import PROFILE, install_reference_profile
    from .test_gemma_frozen import original_encoder_dictionary

    config = Gemma2Config(vocab_size=512, hidden_size=24, intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, max_position_embeddings=32, sliding_window=3,
        query_pre_attn_scalar=6, attn_logit_softcapping=1.7,
        final_logit_softcapping=30., attention_bias=False,
        hidden_activation="gelu_pytorch_tanh", use_cache=False)
    config._attn_implementation = "eager"
    # Fixed before product evaluation, as in the reference arithmetic fixture.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = Gemma2ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.copy_(torch.linspace(-.373, .713, parameter.numel()))
                else:
                    parameter.mul_(5)
        model.bfloat16()
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    install_reference_profile(model, PROFILE)
    return model, TinyTokenizer(), original_encoder_dictionary(model, dtype=torch.bfloat16)


def feature_values(weights, value, reference):
    """Compute features directly from weights for the independent oracle."""
    pre = F.linear(value, weights.W_enc_original if reference else weights.W_enc, weights.b_enc)
    if reference:
        return pre * (pre > weights.threshold)
    values, indices = pre.relu().topk(weights.k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, indices, values)


def explicit_logits(case, specifications=(), *, frozen=False):
    """Apply explicit hooks to a separate model without an intervention service."""
    model, dictionary = case.oracle, case.dictionary
    blocks = model.model.layers if case.reference else model.layers
    with ExitStack() as stack, torch.no_grad():
        if frozen:
            # Freeze only clean attention. Norms and feature inputs remain live.
            for block, pattern in zip(blocks, case.clean_patterns):
                attention = block.self_attn
                if case.reference:
                    values = []
                    def read_value(module, args, output, values=values):
                        values[:] = [output]
                    def mix(module, args, values=values, pattern=pattern, attention=attention):
                        value = values.pop().reshape(*args[0].shape[:2], -1, attention.head_dim)
                        value = value.transpose(1, 2).repeat_interleave(attention.num_key_value_groups, dim=1)
                        return ((pattern @ value).transpose(1, 2).reshape_as(args[0]),)
                    stack.callback(attention.v_proj.register_forward_hook(read_value).remove)
                    stack.callback(attention.o_proj.register_forward_pre_hook(mix).remove)
                else:
                    def replace(module, args, output, pattern=pattern):
                        return pattern
                    stack.callback(attention.pattern.register_forward_hook(replace).remove)
        for spec in specifications:
            if spec["kind"] == "ablate":
                if "#" in spec["point"]:
                    point, head = spec["point"].split("#")
                    attention = model.get_submodule(point)
                    start = int(head) * attention.head_dim
                    end = start + attention.head_dim
                    def zero_head(module, args, start=start, end=end):
                        value = args[0].clone()
                        value[..., start:end] = 0
                        return (value,) + args[1:]
                    stack.callback(attention.o_proj.register_forward_pre_hook(zero_head).remove)
                else:
                    stack.callback(model.get_submodule(spec["point"]).register_forward_hook(
                        lambda module, args, output: torch.zeros_like(output)).remove)
                continue
            layer = int(spec["point"].split(".")[-2])
            block, weights, inputs = blocks[layer], dictionary.layers[layer], []
            if case.reference:
                def read(module, args, output, inputs=inputs):
                    inputs[:] = [output]
                stack.callback(block.pre_feedforward_layernorm.reference_input.register_forward_hook(read).remove)
                write = block.post_feedforward_layernorm
            else:
                def read(module, args, inputs=inputs):
                    inputs[:] = [args[0]]
                stack.callback(block.mlp.register_forward_pre_hook(read).remove)
                write = block.mlp
            def clamp(module, args, output, spec=spec, weights=weights, inputs=inputs):
                active = feature_values(weights, inputs.pop(), case.reference)
                delta = torch.zeros_like(output, dtype=active.dtype)
                for position in spec["positions"]:
                    change = spec["value"] - active[:, position, spec["feature"]]
                    delta[:, position] = change[:, None] * weights.W_dec[spec["feature"]]
                return output + delta
            stack.callback(write.register_forward_hook(clamp).remove)
        inputs = {name: value.repeat(2, 1) for name, value in case.inputs.items()}
        return model(**inputs, use_cache=False).logits[0, -1].clone()


@pytest.fixture(params=["fixture", "reference"])
def case(request, monkeypatch, tmp_path):
    reference = request.param == "reference"
    if reference:
        from caliper.interp._memory import MemoryGuard
        model, tokenizer, dictionary = reference_fixture()
        guard = MemoryGuard(2, "cpu")
    else:
        model, tokenizer = make_fixture(seed=7, layers=2, width=8, heads=2, latents=16, k=2)
        model.requires_grad_(False)
        dictionary, guard = dictionary_for(model), None
    oracle = deepcopy(model)
    inputs = prompt_inputs(model, tokenizer, "AB")
    blocks = oracle.model.layers if reference else oracle.layers
    clean_patterns, features = [], []
    with ExitStack() as stack, torch.no_grad():
        for block in blocks:
            def record(module, args, output):
                clean_patterns.append((output[1] if reference else output).detach().clone())
            point = block.self_attn if reference else block.self_attn.pattern
            stack.callback(point.register_forward_hook(record).remove)
        if reference:
            def read(module, args, output):
                features.append(feature_values(dictionary.layers[0], output, True))
            stack.callback(blocks[0].pre_feedforward_layernorm.reference_input.register_forward_hook(read).remove)
        else:
            def read(module, args):
                features.append(feature_values(dictionary.layers[0], args[0], False))
            stack.callback(blocks[0].mlp.register_forward_pre_hook(read).remove)
        oracle(**inputs, use_cache=False)
    # Select from the independent clean oracle, before Controls or preparation.
    feature = int(features[0][0, 1].argmax())
    assert features[0][0, 1, feature] > 0
    monkeypatch.setattr(published, "PROMPTS", dict(dallas="AB", oakland="CD", shanghai="EF", vancouver="GH"))
    monkeypatch.setattr(published, "GROUPS", {name: [(0, feature, 1)] for name in published.GROUPS})
    controls, adapter_calls = Controls(), []
    def install_ablation(current, point):
        assert current is model
        adapter_calls.append(point)
        return TORCH.ablate(current, point)
    adapter = SimpleNamespace(points=TORCH.points, ablate=install_ablation)
    service = controls.interventions_on(model, adapter=adapter)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=guard)
    runtime = published.PublishedCases(model, tokenizer, service, guard=guard, output_directory=tmp_path)
    output = Output(model, tokenizer, service, guard=guard)
    output.attention_scope = runtime.output_scope
    result = SimpleNamespace(model=model, oracle=oracle, dictionary=dictionary, reference=reference,
        tokenizer=tokenizer, inputs=inputs, clean_patterns=clean_patterns, controls=controls,
        service=service, runtime=runtime, output=output, logits=[], adapter_calls=adapter_calls,
        feature_activation=float(features[0][0, 1, feature]))
    publish = output.publish_logits
    def record_logits(logits):
        result.logits.append(logits.clone())
        return publish(logits)
    monkeypatch.setattr(output, "publish_logits", record_logits)
    try:
        runtime.prepare()
        yield result
    finally:
        runtime.close()
        service.remove()
        assert not runtime._handles and not runtime._forward_scopes and not runtime.patterns
        assert not service.executor._feature_groups
        assert all(hook.removed for hook in service.executor._hooks)
        for current in (model, oracle):
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in current.modules())


def ablation(case, kind):
    prefix = "model.layers.0" if case.reference else "layers.0"
    return {"kind": "ablate", "point": prefix + (".mlp" if kind == "module" else ".self_attn#0")}


def active_hooks(case):
    return {hook for hook in case.service.executor._hooks if not hook.removed}


def compare(case, expected, *, changed=True):
    baseline = explicit_logits(case)
    if changed:
        assert (expected - baseline).abs().max() > 1e-6, "the independent effect must be nonzero"
    result = case.output.compare("AB")
    actual = case.logits[-1]
    tolerance = 0 if case.reference else 1e-12
    print({"reference": case.reference, "expected_effect": float((expected - baseline).abs().max()),
           "actual_effect": float((actual[1] - actual[0]).abs().max()),
           "oracle_error": float((actual[1] - expected).abs().max())})
    torch.testing.assert_close(actual[0], baseline, rtol=0, atol=tolerance)
    torch.testing.assert_close(actual[1], expected, rtol=0, atol=tolerance)
    if changed:
        assert not torch.equal(actual[0], actual[1])
    torch.testing.assert_close(torch.tensor(result["probabilities"]),
                               actual.softmax(-1)[:, result["tokens"]].float(), rtol=0, atol=0)
    assert not case.runtime._forward_scopes
    assert case.service.executor._published_forward_depth == 0
    return actual


@pytest.mark.parametrize("kind", ["module", "head"])
@pytest.mark.parametrize("group", [False, True])
def test_live_ablation_uses_live_attention(case, kind, group):
    specifications = [ablation(case, kind)] + (case.runtime.case(14)[0] if group else [])
    expected = explicit_logits(case, specifications)
    for spec in specifications:
        case.service.set(spec)
    assert not active_hooks(case)
    case.controls.step()
    compare(case, expected)
    hooks = [record.hook for record in case.controls._ivs.values()]
    assert active_hooks(case) == set(hooks)
    assert [case.service.executor.execution_order(hook) for hook in hooks] == sorted(
        case.service.executor.execution_order(hook) for hook in hooks)
    assert all(hook.calls and hook.errors == 0 for hook in hooks)
    assert case.adapter_calls == [specifications[0]["point"]]


def test_baseline_and_published_group_keep_frozen_attention(case, monkeypatch):
    scopes = []
    attention_patterns = published.attention_patterns
    @contextmanager
    def record_scope(*args):
        scopes.append(True)
        with attention_patterns(*args):
            yield
    monkeypatch.setattr(published, "attention_patterns", record_scope)
    compare(case, explicit_logits(case), changed=False)
    assert scopes and not active_hooks(case)
    specifications, _ = case.runtime.case(14)
    assert specifications[0]["value"] == -2 * case.feature_activation
    expected = explicit_logits(case, specifications, frozen=True)
    live = explicit_logits(case, specifications)
    assert (expected - live).abs().max() > 1e-6
    for spec in specifications:
        case.service.set(spec)
    case.controls.step()
    scopes.clear()
    compare(case, expected)
    assert scopes
    assert case.runtime.matching_case("AB", case.service.specifications()) == 14
    assert len(active_hooks(case)) == len(specifications)


@pytest.mark.parametrize("kind", ["module", "head"])
@pytest.mark.parametrize("group", [False, True])
def test_restore_fork_replay_and_clear_preserve_live_ablation(case, monkeypatch, kind, group):
    specifications = [ablation(case, kind)] + (case.runtime.case(14)[0] if group else [])
    expected = explicit_logits(case, specifications)
    for spec in specifications:
        case.service.set(spec)
    case.controls.step()
    compare(case, expected)
    saved = case.service.state_dict()
    hooks = active_hooks(case)
    refs = [weakref.ref(hook) for hook in hooks]
    with case.service.temporary_state(saved) as replay, monkeypatch.context() as patch:
        replay_hooks = set(replay.executor._hooks)
        assert len(replay_hooks) == len(hooks) and not replay_hooks & hooks
        patch.setattr(case.output, "attention_scope",
                      lambda prompt, specs: case.runtime.output_scope(prompt, replay.specifications()))
        compare(case, expected)
        assert active_hooks(case) == hooks
    assert all(hook.removed for hook in replay_hooks)
    assert case.service.state_dict() == saved
    case.service.clear()
    # A queued clear retains each effect until the next boundary.
    compare(case, expected)
    case.controls.step()
    assert all(hook.removed for hook in hooks) and not active_hooks(case)
    hooks.clear()
    assert all(ref() is None for ref in refs)
    assert not case.service.executor._hook_order
    compare(case, explicit_logits(case), changed=False)
    case.service.load_state_dict(saved, restore_active=True)
    assert len(active_hooks(case)) == len(specifications)
    compare(case, expected)
    case.service.clear()
    case.controls.step()
    case.service.load_state_dict(saved)
    assert not active_hooks(case)
    compare(case, explicit_logits(case), changed=False)
    case.controls.step()
    compare(case, expected)
    # A fork adds a distinct head and retains all restored effects.
    override = dict(ablation(case, "head"), point=ablation(case, "head")["point"][:-1] + "1")
    case.service.apply_overrides([override])
    assert len(active_hooks(case)) == len(specifications) + 1
    compare(case, explicit_logits(case, [*specifications, override]))
    case.service.clear()
    case.controls.step()
    assert not active_hooks(case)
    assert not case.service.executor._feature_groups
    compare(case, explicit_logits(case), changed=False)


@pytest.mark.parametrize("kind", ["module", "head"])
def test_adapter_error_ends_live_ablation_and_releases_hook(case, monkeypatch, kind):
    spec = ablation(case, kind)
    case.service.set(spec)
    case.controls.step()
    record, = case.controls._ivs.values()
    hook = record.hook
    assert hook in active_hooks(case) and hook.on_error is not None
    assert case.adapter_calls == [spec["point"]]
    def refuse(*args):
        raise ValueError("synthetic ablation refusal")
    with monkeypatch.context() as patch:
        patch.setattr(hook, "_zero", refuse)
        compare(case, explicit_logits(case), changed=False)
    assert record.hook is None and record.state == "ended"
    assert record.error_code == "intervene.hook_failed"
    assert "synthetic ablation refusal" in record.error
    assert hook.removed and hook.errors == 1 and hook.on_error is None
    assert not active_hooks(case)
    ref = weakref.ref(hook)
    del hook
    assert ref() is None
    assert not case.service.executor._hooks and not case.service.executor._hook_order
    compare(case, explicit_logits(case), changed=False)
