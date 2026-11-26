"""Backend admission follows the supported family's actual attention path."""
import pytest
import torch

from caliper.interp._dictionary import _state_hash, model_identity, verify_hf_model
from caliper.interp._forward_identity import forward_settings
from caliper.interp._intervene import InterventionService
from .test_gemma_family import gemma_model
from .test_gemma_runtime import gemma_model as checkpoint_model, local_gemma


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
