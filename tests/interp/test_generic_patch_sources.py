"""Generic patch sources bind supported callables and effective decoder settings."""
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from caliper.interp._dictionary import _state_hash, model_identity, verify_hf_model
from caliper.interp._forward_identity import forward_settings
from caliper.interp._intervene import InterventionService
from caliper.interp._points import capture


class TokenTransformer(nn.Module):
    """Adapt small native PyTorch layers to the patch prompt contract."""

    def __init__(self, kind, activation, *, backend="eager"):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32, model_type="generic", _attn_implementation=backend)
        self.embed = nn.Embedding(32, 8)
        layer = nn.TransformerEncoderLayer if kind == "encoder" else nn.TransformerDecoderLayer
        # An odd head count exercises the callable instead of the encoder fast path.
        self.layer = layer(8, 1, dim_feedforward=12, dropout=0, batch_first=True, activation=activation)
        self.head = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        values = self.embed(input_ids)
        if isinstance(self.layer, nn.TransformerDecoderLayer):
            values = self.layer(values, values)
        else:
            values = self.layer(values)
        return self.head(values)


def tokenizer(prompt, **kwargs):
    return {"input_ids": torch.tensor([[1, 2, 3, 4] if prompt == "source" else [5, 6, 7, 8]])}


def generic_model(kind, activation, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        return TokenTransformer(kind, activation, **kwargs).eval()


SPEC = {"kind": "patch", "point": "head", "source_prompt": "source"}


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("activation", ["relu", "gelu", "partial", "module"])
def test_native_transformer_patch_matches_clean_forward_and_reuses_source(kind, activation):
    activation = {"relu": F.relu, "gelu": F.gelu,
                  "partial": partial(F.gelu, approximate="tanh"), "module": nn.GELU()}[activation]
    model = generic_model(kind, activation)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        with torch.no_grad():
            expected = model(**tokenizer("source"))
            baseline = model(**tokenizer("target"))
        hook = service.install(SPEC)
        source = service._cache["source", "head"]
        with torch.no_grad():
            actual = model(**tokenizer("target"))
        assert hook.error is None and not torch.equal(baseline, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert service._source(SPEC) is source
    finally:
        service.close()


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("change", ["callable", "partial", "module", "norm_first"])
def test_native_transformer_source_refreshes_effective_settings(kind, change):
    activation = nn.GELU() if change == "module" else partial(F.gelu, approximate="none")
    model = generic_model(kind, activation)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        hook = service.install(SPEC)
        original = service._cache["source", "head"]
        if change == "callable":
            model.layer.activation = F.relu
        elif change == "partial":
            activation.keywords["approximate"] = "tanh"
        elif change == "module":
            activation.approximate = "tanh"
        else:
            model.layer.norm_first = True
        with service.suspend(), torch.no_grad():
            expected = model(**tokenizer("source"))
        with torch.no_grad():
            actual = model(**tokenizer("target"))
        assert hook.error is None and not torch.equal(original, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        refreshed = service._cache["source", "head"]
        assert refreshed is not original and service._source(SPEC) is refreshed
    finally:
        service.close()


def test_generic_backend_evidence_does_not_grant_checkpoint_admission():
    model = generic_model("encoder", F.relu, backend="flash_attention_2")
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        original = service._source(SPEC)
        evidence = service._cache_identity
        model.config._attn_implementation = "sdpa"
        refreshed = service._source(SPEC)
        assert refreshed is not original and service._cache_identity != evidence
        # This native layer runs on CPU. Its config label is source evidence only.
        with torch.no_grad():
            expected = model(**tokenizer("source"))
        torch.testing.assert_close(refreshed, expected, rtol=0, atol=0)
        model.config._attn_implementation = "flash_attention_2"
        with pytest.raises(ValueError, match="checkpoint identity supports eager or sdpa"):
            forward_settings(model, tensor_hash=_state_hash)
    finally:
        service.close()


@pytest.mark.parametrize("change", ["bidirectional", "causal_config", "causal_module"])
def test_gemma_source_tracks_bidirectional_settings_without_checkpoint_relaxation(tmp_path, change):
    from .test_gemma_runtime import gemma_model, local_gemma

    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    verify_hf_model(model, snapshot)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    spec = dict(SPEC, point="model.layers.0.mlp_out")
    point = spec["point"]
    try:
        original = service._source(spec)
        evidence = service._cache_identity
        if change == "bidirectional":
            model.config.use_bidirectional_attention = True
        elif change == "causal_config":
            model.config.is_causal = False
        else:
            model.model.layers[0].self_attn.is_causal = False
        with pytest.raises(ValueError, match="forward settings changed"):
            model_identity(model)
        with pytest.raises(ValueError, match="forward settings differ"):
            verify_hf_model(model, snapshot)
        with torch.no_grad(), capture(model, [point]) as values:
            model(**tokenizer("source"), use_cache=False)
            expected = values[point].clone()
        refreshed = service._source(spec)
        assert refreshed is not original and service._cache_identity != evidence
        torch.testing.assert_close(refreshed, expected, rtol=0, atol=0)
        if change == "causal_config":
            assert not torch.equal(original, refreshed)
        assert service._source(spec) is refreshed
    finally:
        service.close()
