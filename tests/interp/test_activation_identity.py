"""Effective activation options bind checkpoint and clean source identity."""
from functools import partial
import json
import subprocess
import sys

import pytest
import torch

from caliper.interp._dictionary import model_identity, verify_hf_model
from caliper.interp._evaluation_identity import evaluation_model_evidence
from caliper.interp._intervene import InterventionService
from .test_gemma_runtime import gemma_model, local_gemma


SPEC = {"kind": "patch", "point": "model.layers.0.mlp_out", "source_prompt": "source"}


def tokenizer(*args, **kwargs):
    return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


@pytest.mark.parametrize("verified", [False, True])
def test_mutable_gelu_option_refuses_checkpoint_before_tensor_read(tmp_path, monkeypatch, verified):
    import safetensors

    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    if verified:
        verify_hf_model(model, snapshot)
    activation = model.model.layers[0].mlp.act_fn
    function = activation.act
    function.keywords["approximate"] = "none"
    assert activation.act is function
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **kw: pytest.fail("checkpoint tensor read ran"))
    if verified:
        with pytest.raises(ValueError, match="forward settings changed"):
            model_identity(model)
    with pytest.raises(ValueError, match="forward settings differ from the local checkpoint"):
        verify_hf_model(model, snapshot)


@pytest.mark.parametrize("change", ["keyword", "callable", "bound_method"])
def test_source_refreshes_effective_activation_and_matches_direct_forward(change):
    model = gemma_model()
    with torch.no_grad():
        model.model.layers[0].mlp.gate_proj.weight.mul_(12)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        original = service._source(SPEC)
        activation = model.model.layers[0].mlp.act_fn
        function = activation.act
        if change == "keyword":
            function.keywords["approximate"] = "none"
        elif change == "callable":
            activation.act = partial(torch.nn.functional.relu, inplace=False)
        else:
            activation.act = activation._gelu_tanh_python
        assert model.model.layers[0].mlp.act_fn is activation
        if change == "keyword":
            assert activation.act is function
        values = []
        observer = model.model.layers[0].post_feedforward_layernorm.register_forward_hook(
            lambda module, args, output: values.append(output.detach().clone()))
        try:
            with torch.no_grad():
                model(**tokenizer(), use_cache=False)
        finally:
            observer.remove()
        refreshed = service._source(SPEC)
        assert refreshed is not original
        if change != "bound_method":
            assert not torch.equal(refreshed, original)
        torch.testing.assert_close(refreshed, values[0], rtol=0, atol=0)
        assert service._source(SPEC) is refreshed
    finally:
        service.close()


@pytest.mark.parametrize("change", ["closure", "partial_function", "partial_args", "partial_option", "module", "forward"])
def test_unknown_activation_state_refuses_identity_and_cached_source(tmp_path, change):
    model = gemma_model()
    verify_hf_model(model, local_gemma(tmp_path, model))
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        service._source(SPEC)
        activation = model.model.layers[0].mlp.act_fn
        if change == "closure":
            activation.act = lambda value: value * .5
        elif change == "partial_function":
            activation.act = partial(lambda value, approximate: value, approximate="tanh")
        elif change == "partial_args":
            activation.act = partial(torch.nn.functional.gelu, torch.ones(1))
        elif change == "partial_option":
            activation.act.keywords["approximate"] = object()
        elif change == "module":
            activation.act = torch.nn.ReLU().eval()
        else:
            activation.forward = lambda value: value
        with pytest.raises(ValueError, match="unsupported effective activation"):
            model_identity(model)
        with pytest.raises(ValueError, match="unsupported effective activation"):
            service._source(SPEC)
    finally:
        service.close()


@pytest.mark.parametrize("approximation", ["tanh", "none"])
def test_activation_quality_digest_is_stable_across_processes(approximation):
    model = gemma_model()
    identity = model_identity(model)
    original = evaluation_model_evidence(model, identity)
    model.model.layers[0].mlp.act_fn.act.keywords["approximate"] = approximation
    expected = evaluation_model_evidence(model, identity)
    assert (expected == original) == (approximation == "tanh")
    code = """
import json
import sys
from caliper.interp._dictionary import model_identity
from caliper.interp._evaluation_identity import evaluation_model_evidence
from tests.interp.test_gemma_runtime import gemma_model
model = gemma_model()
identity = model_identity(model)
model.model.layers[0].mlp.act_fn.act.keywords["approximate"] = sys.argv[1]
print(json.dumps(evaluation_model_evidence(model, identity)))
"""
    process = subprocess.run([sys.executable, "-c", code, approximation], capture_output=True,
                             text=True, check=True, timeout=30)
    assert json.loads(process.stdout) == expected
