"""Native source identity and immutable intermediate captures use CPU forwards."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from caliper.interp import _intervene
from caliper.interp._intervene import InterventionError, InterventionService
from .test_generic_patch_sources import SPEC, tokenizer


class NativeActivationModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32, model_type="generic")
        self.embed = nn.Embedding(32, 2)
        self.hidden = nn.Linear(2, 2, bias=False)
        self.nonlinearity = activation
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            values = torch.arange(1, 33, dtype=torch.float32) / 4
            self.embed.weight.copy_(torch.stack((-values, values), dim=-1))
            self.hidden.weight.copy_(torch.eye(2))
            self.head.weight.copy_(torch.eye(2))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.head(self.nonlinearity(self.hidden(self.embed(input_ids))))


@pytest.mark.parametrize("change", ["slope", "approximation", "inplace"])
def test_standalone_activation_option_refreshes_source_before_target_forward(change):
    activation = nn.GELU() if change == "approximation" else nn.LeakyReLU(0.1)
    model = NativeActivationModel(activation).eval()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        hook = service.install(SPEC)
        original = service._cache["source", "head"]
        evidence = service._cache_identity
        parameters = [(id(value), value._version) for value in model.parameters()]
        assert (original < 0).any()
        if change == "slope":
            activation.negative_slope = 0.5
        elif change == "approximation":
            activation.approximate = "tanh"
        else:
            activation.inplace = True
        with service.suspend(), torch.no_grad():
            expected = model(**tokenizer("source"))
            baseline = model(**tokenizer("target"))
        with torch.no_grad():
            actual = model(**tokenizer("target"))
        assert model.nonlinearity is activation
        assert parameters == [(id(value), value._version) for value in model.parameters()]
        assert hook.error is None and not torch.equal(baseline, expected)
        assert (expected < 0).any()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        refreshed = service._cache["source", "head"]
        assert refreshed is not original and service._cache_identity != evidence
        assert service._source(SPEC) is refreshed
        if change != "inplace":
            assert not torch.equal(original, refreshed)
    finally:
        service.close()
    assert not model.head._forward_hooks and not service._cache


@pytest.mark.parametrize("change", ["option", "forward", "call", "compiled", "subclass"])
def test_standalone_activation_refuses_unknown_state_before_cache_reuse(monkeypatch, change):
    model = NativeActivationModel(nn.LeakyReLU(0.1)).eval()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        original = service._source(SPEC)
        evidence = service._cache_identity
        if change == "option":
            model.nonlinearity.negative_slope = object()
        elif change == "subclass":
            class CustomLeakyReLU(nn.LeakyReLU):
                pass
            model.nonlinearity = CustomLeakyReLU(0.1)
        else:
            field = {"forward": "forward", "call": "_call_impl", "compiled": "_compiled_call_impl"}[change]
            setattr(model.nonlinearity, field, lambda value: value)
        monkeypatch.setattr(_intervene, "_content_identity",
                            lambda *args: pytest.fail("unsupported activation reached the weight hash"))
        with pytest.raises(InterventionError, match="unsupported effective activation") as refused:
            service._source(SPEC)
        assert refused.value.code == "intervene.not_supported"
        assert service._cache["source", "head"] is original and service._cache_identity is evidence
        assert not service.preparing_source
        assert all(not module._forward_hooks for module in model.modules())
    finally:
        service.close()
