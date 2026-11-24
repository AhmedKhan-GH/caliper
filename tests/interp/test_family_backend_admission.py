"""Backend admission follows the supported family's actual attention path."""
import pytest
import torch

from caliper.interp._dictionary import _state_hash, model_identity, verify_hf_model
from caliper.interp._forward_identity import forward_settings
from caliper.interp._intervene import InterventionService
from .test_gemma_family import gemma_model
from .test_gemma_runtime import gemma_model as checkpoint_model, local_gemma
from .test_points import qwen


@pytest.mark.parametrize("backend", [None, "sdpa"])
def test_gemma_refuses_backend_after_eager_verification(tmp_path, backend):
    model = checkpoint_model()
    snapshot = local_gemma(tmp_path, model)
    expected = verify_hf_model(model, snapshot)
    assert model_identity(model) == expected
    model.config._attn_implementation = backend
    with pytest.raises(ValueError, match="Gemma.*eager.*softcap"):
        model_identity(model)
    with pytest.raises(ValueError, match="Gemma.*eager.*softcap"):
        verify_hf_model(model, snapshot)


def test_gemma_null_backend_loses_causal_and_sliding_masks(monkeypatch):
    from transformers.models.gemma2 import modeling_gemma2 as implementation

    model = gemma_model(2, window=2)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    original = implementation.eager_attention_forward
    observed = {}

    def attention(module, query, key, value, mask, **kwargs):
        output = original(module, query, key, value, mask, **kwargs)
        observed[module.layer_idx] = (mask, output[1].detach())
        return output

    monkeypatch.setattr(implementation, "eager_attention_forward", attention)
    for backend in ("eager", None):
        model.config._attn_implementation = backend
        # A direct forward must exercise the configured mask registry.
        with torch.no_grad():
            model(tokens, use_cache=False)
        assert set(observed) == {0, 1}
        for layer, (mask, pattern) in observed.items():
            if backend == "eager":
                assert mask is not None
                assert pattern.triu(1).count_nonzero() == 0
                assert (pattern[..., -1, :-2].count_nonzero() == 0) == (layer == 0)
            else:
                assert mask is None
                assert pattern.triu(1).count_nonzero() > 0
                assert pattern[..., -1, :-2].count_nonzero() > 0


def test_gemma_null_backend_cannot_reuse_clean_source():
    model = checkpoint_model()
    service = InterventionService(model)
    service.configure(tokenizer=lambda *a, **kw: {"input_ids": torch.tensor([[1, 2, 3]])})
    spec = {"kind": "patch", "point": "model.layers.0.mlp_out", "source_prompt": "source"}
    try:
        service._source(spec)
        model.config._attn_implementation = None
        with pytest.raises(ValueError, match="Gemma.*eager.*softcap"):
            service._source(spec)
    finally:
        service.close()


def test_qwen_retains_eager_and_sdpa_identity():
    model = qwen()
    model.config._attn_implementation = "eager"
    eager = forward_settings(model, tensor_hash=_state_hash)
    model.config._attn_implementation = "sdpa"
    assert forward_settings(model, tensor_hash=_state_hash) == eager


@pytest.mark.parametrize("first_backend", ["eager", "sdpa"])
def test_qwen_source_cache_tracks_actual_backend_without_checkpoint_change(tmp_path, monkeypatch, first_backend):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    from tools.interp import model_load
    from .test_model_load import local_snapshot

    model = qwen()
    model.config._name_or_path = model_load.MODEL_ID
    model.config._commit_hash = model_load.MODEL_REVISION
    model.config._attn_implementation = first_backend
    checkpoint = verify_hf_model(model, local_snapshot(tmp_path, model))
    eager, sdpa = implementation.eager_attention_forward, torch.nn.functional.scaled_dot_product_attention
    calls = []

    def observe_eager(*args, **kwargs):
        calls.append("eager")
        return eager(*args, **kwargs)

    def observe_sdpa(*args, **kwargs):
        calls.append("sdpa")
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(implementation, "eager_attention_forward", observe_eager)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe_sdpa)
    service = InterventionService(model)
    tokens = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    service.configure(tokenizer=lambda *a, **kw: tokens)
    spec = {"kind": "patch", "point": "model.layers.3.mlp_out", "source_prompt": "source"}
    try:
        original = service._source(spec)
        evidence = service._cache_identity
        assert calls == [first_backend]
        assert service._source(spec) is original
        second_backend = "sdpa" if first_backend == "eager" else "eager"
        model.config._attn_implementation = second_backend
        assert model_identity(model) == checkpoint
        refreshed = service._source(spec)
        assert calls == [first_backend, second_backend]
        assert refreshed is not original and service._cache_identity != evidence
        values = []
        observer = model.model.layers[3].mlp.register_forward_hook(
            lambda module, args, output: values.append(output.detach().clone()))
        try:
            with torch.no_grad():
                model(**tokens, use_cache=False)
        finally:
            observer.remove()
        torch.testing.assert_close(refreshed, values[0], rtol=0, atol=0)
        assert service._source(spec) is refreshed
        assert calls == [first_backend, second_backend, second_backend]
    finally:
        service.close()
