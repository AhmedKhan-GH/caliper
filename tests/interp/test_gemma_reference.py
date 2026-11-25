"""Tiny CPU qualification for the explicit Gemma arithmetic profile."""
from contextlib import ExitStack
import json

import pytest
import torch
from torch.nn import functional as F

from caliper.interp._gemma_reference import (PROFILE, ReferenceRMSNorm, capture_reference_inputs,
    install_reference_profile, profile_metadata)
from caliper.interp._forward_identity import forward_settings
from caliper.interp._dictionary import _state_hash, model_identity, verify_hf_model


def reference_fixture(dtype=torch.bfloat16):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    config = Gemma2Config(vocab_size=37, hidden_size=24, intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, max_position_embeddings=32, sliding_window=3,
        query_pre_attn_scalar=6, attn_logit_softcapping=1.7,
        final_logit_softcapping=2.3, attention_bias=False,
        hidden_activation="gelu_pytorch_tanh", use_cache=False)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = Gemma2ForCausalLM(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.copy_(torch.linspace(-.373, .713, parameter.numel()))
                else:
                    parameter.mul_(5)
        model.to(dtype=dtype).requires_grad_(False)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    return model


def test_reference_norm_exposes_rounding_and_preserves_source_bytes():
    model = reference_fixture()
    norm = model.model.layers[0].pre_feedforward_layernorm
    x = torch.linspace(-2.71, 3.13, 7 * 24).reshape(1, 7, 24).bfloat16()
    weight = norm.weight.detach().clone()
    ordinary = norm(x)
    install_reference_profile(model, PROFILE)
    assert type(norm) is ReferenceRMSNorm
    raw = (x.float() / (x.float().square().mean(-1, keepdim=True) + norm.eps).sqrt()).bfloat16()
    gain = (weight.float() + 1).bfloat16()
    expected = raw * gain
    assert torch.equal(norm(x), expected)
    assert torch.equal(norm.weight, weight)
    assert not torch.equal(expected, ordinary)
    assert not torch.equal(expected / gain, raw)
    assert not torch.equal(gain.float(), weight.float() + 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_profile_identity_and_default(dtype):
    model = reference_fixture(dtype)
    before = forward_settings(model, tensor_hash=_state_hash)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    ids = torch.tensor([[1, 4, 6, 9, 2, 11, 3]])
    ordinary = model(ids).logits
    identity = model_identity(model)
    install_reference_profile(model, None)
    assert forward_settings(model, tensor_hash=_state_hash) == before
    assert torch.equal(ordinary, model(ids).logits)
    install_reference_profile(model, PROFILE)
    metadata = json.loads(forward_settings(model, tensor_hash=_state_hash)["__gemma_reference__"])
    assert metadata["profile"] == PROFILE
    assert metadata["dtype"] == str(dtype).removeprefix("torch.")
    assert model_identity(model) != identity
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert model(ids).logits.shape == ordinary.shape


@pytest.mark.parametrize("profile", ["caliper", "circuit_tracer", False, {}, 1])
def test_unknown_profiles_refused_before_changes(profile):
    model = reference_fixture()
    cls = type(model)
    with pytest.raises(ValueError, match="unsupported Gemma reference profile"):
        install_reference_profile(model, profile)
    assert type(model) is cls


@pytest.mark.parametrize("mutation", ["profile", "dtype", "forward", "norm", "attention", "missing"])
def test_mutated_adapter_refusal(mutation):
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    norm = model.model.layers[0].pre_feedforward_layernorm
    if mutation == "profile":
        model._caliper_reference_profile = "other"
    elif mutation == "dtype":
        model.to(torch.float32)
    elif mutation == "forward":
        norm.forward = lambda x: x
    elif mutation == "norm":
        norm.eps *= 2
    elif mutation == "attention":
        model.model.layers[0].self_attn.num_key_value_groups += 1
    else:
        del model._caliper_reference_profile
    with pytest.raises(ValueError, match="reference"):
        forward_settings(model, tensor_hash=_state_hash)
    with pytest.raises(ValueError, match="reference"):
        model(torch.tensor([[1, 2, 3]]))


def test_raw_capture_matches_live_residual_and_removes_hooks():
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    norm = model.model.layers[0].pre_feedforward_layernorm
    residuals, outputs = [], []
    with ExitStack() as stack:
        stack.callback(norm.register_forward_pre_hook(lambda mod, args: residuals.append(args[0].clone())).remove)
        stack.callback(norm.register_forward_hook(lambda mod, args, output: outputs.append(output.clone())).remove)
        count = len(norm._forward_pre_hooks)
        with capture_reference_inputs(model) as values:
            model(torch.tensor([[1, 4, 6, 9, 2, 11, 3]]))
            x = residuals[-1].float()
            wanted = (x / (x.square().mean(-1, keepdim=True) + norm.eps).sqrt()).bfloat16()
            raw = values["model.layers.0.mlp_in"]
            assert torch.equal(raw, wanted)
            assert not raw.requires_grad
            gain = (norm.weight.float() + 1).bfloat16()
            assert not torch.equal(outputs[-1] / gain, raw)
            encoder = torch.linspace(-.31, .27, 5 * 24).reshape(5, 24).bfloat16()
            assert torch.equal(F.linear(raw, encoder), F.linear(wanted, encoder))
            model(torch.tensor([[1, 2]]))
            assert all(value.shape[1] == 2 for value in values.values())
            with pytest.raises(ValueError, match="uncached"):
                model(torch.tensor([[1, 2]]), use_cache=True)
            assert not values
        assert len(norm._forward_pre_hooks) == count


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adapted_checkpoint_verification_and_parameter_mutation(tmp_path, dtype):
    from safetensors.torch import save_file
    model = reference_fixture(dtype)
    snapshot = tmp_path / "models--caliper-test--random-gemma2" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(model.config.to_json_string())
    state = {key: value.clone().contiguous() for key, value in model.state_dict().items()
             if key != "lm_head.weight"}
    save_file(state, snapshot / "model.safetensors")
    install_reference_profile(model, PROFILE)
    expected = "hf:caliper-test/random-gemma2@" + "b" * 40
    assert verify_hf_model(model, snapshot) == expected
    assert model_identity(model) == expected
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.weight[0, 0] += 1
    with pytest.raises(ValueError, match="weights changed"):
        model_identity(model)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_hf_model(model, snapshot)


def test_loader_profile_refuses_before_snapshot_access(tmp_path):
    from caliper.interp._admission import _load_gemma_model
    with pytest.raises(ValueError, match="unsupported Gemma reference profile"):
        _load_gemma_model(tmp_path / "absent", device="cpu", memory=None, reference_profile="other")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loader_opt_in_preserves_complete_verification(tmp_path, monkeypatch, dtype):
    from caliper.interp import _admission, _dictionary
    from .test_gemma_runtime import mock_gemma_load, memory
    model, snapshot, calls = mock_gemma_load(tmp_path, monkeypatch)
    model.to(dtype=dtype)
    loaded = _admission._load_gemma_model(snapshot, device="cpu", memory=memory(),
                                         dtype=dtype, reference_profile=PROFILE)
    assert loaded is model and calls == ["weights"]
    assert profile_metadata(loaded)["dtype"] == str(dtype).removeprefix("torch.")
    assert _dictionary.model_identity(loaded) == f"hf:{_admission.GEMMA_ID}@{_admission.GEMMA_REVISION}"
    model.model.layers[0].pre_feedforward_layernorm.weight.data.add_(1)
    with pytest.raises(ValueError, match="weights changed"):
        _dictionary.model_identity(loaded)


def test_configuration_and_hook_mutation_refused():
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    model.config.final_logit_softcapping *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        model(torch.tensor([[1, 2]]))
    model.config.final_logit_softcapping /= 2
    model.model.layers[0].pre_feedforward_layernorm.reference_input = torch.nn.Identity().eval()
    with pytest.raises(ValueError, match="input hook changed"):
        model(torch.tensor([[1, 2]]))


def test_reference_runtime_rejects_implicit_precision_changes():
    model = reference_fixture(torch.float32)
    install_reference_profile(model, PROFILE)
    ids = torch.tensor([[1, 2]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="disabled autocast"):
            model(ids)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError, match="highest matmul precision"):
            model(ids)
    finally:
        torch.set_float32_matmul_precision(precision)


def test_capture_nested_forward_retains_outer_coordinates():
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    outer = torch.tensor([[1, 4, 6]])
    with capture_reference_inputs(model) as expected:
        model(outer)
    active = False
    def nested(module, args, output):
        nonlocal active
        if not active:
            active = True
            try:
                model(torch.tensor([[1, 8]]))
            finally:
                active = False
    with ExitStack() as stack:
        stack.callback(model.model.layers[0].mlp.register_forward_hook(nested).remove)
        with capture_reference_inputs(model) as actual:
            model(outer)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], value) for key, value in actual.items())


def test_checkpoint_buffer_mutation_is_not_hidden_by_adapter(tmp_path):
    from .test_gemma_runtime import local_gemma, gemma_model
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    install_reference_profile(model, PROFILE)
    model.model.rotary_emb.inv_freq.data.add_(1)
    with pytest.raises(ValueError, match="nonpersistent buffers"):
        verify_hf_model(model, snapshot)


def test_identity_binds_executable_operations_and_refuses_code_mutation(monkeypatch):
    from caliper.interp import _gemma_reference as reference
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    settings = profile_metadata(model)
    key = "caliper.interp._gemma_reference.normalized_input"
    assert len(settings["operation_sha256"][key]) == 64
    original = reference.normalized_input.__code__
    assert reference._code_identity(original) == reference._code_identity(
        original.replace(co_filename="different.py", co_firstlineno=42))
    def changed(module, x):
        return x
    monkeypatch.setattr(reference.normalized_input, "__code__", changed.__code__)
    with pytest.raises(ValueError, match="implementation changed"):
        model_identity(model)
    with pytest.raises(ValueError, match="implementation changed"):
        model(torch.tensor([[1, 2]]))


@pytest.mark.parametrize("base", [False, True])
def test_class_forward_mutation_refused(monkeypatch, base):
    from transformers import Gemma2ForCausalLM
    from caliper.interp._gemma_reference import ReferenceRotaryEmbedding
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    monkeypatch.setattr(Gemma2ForCausalLM if base else ReferenceRotaryEmbedding, "forward", lambda *args: None)
    with pytest.raises(ValueError, match="class operation changed"):
        forward_settings(model, tensor_hash=_state_hash)
