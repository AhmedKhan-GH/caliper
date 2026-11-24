"""Clean patch sources follow scalar settings and current tokenized inputs."""
import pytest
import torch

from caliper.interp._intervene import InterventionService
from caliper.interp._points import capture
from .fixture import TinyLM


class MutableTokenizer:
    add_bos_token = True

    def __call__(self, text, **kwargs):
        tokens = [65, 66] if text == "source" else [67, 68]
        if self.add_bos_token:
            tokens.insert(0, 256)
        ids = torch.tensor([tokens])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("change", ["epsilon", "tokens"])
def test_patch_refreshes_after_scalar_or_tokenizer_change(change):
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1).eval()
    tokenizer = MutableTokenizer()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    point = "layers.1.resid_pre" if change == "epsilon" else "layers.0.resid_pre"
    spec = {"kind": "patch", "point": point, "source_prompt": "source", "source_position": 0}
    hook = service.install(spec)
    previous = service._cache["source", point].clone()
    if change == "epsilon":
        model.layers[0].input_layernorm.eps = 2.0
    else:
        tokenizer.add_bos_token = False
    try:
        with service.suspend(), capture(model, [point]) as values, torch.no_grad():
            model(**tokenizer("source"))
            expected = values[point].clone()
        with capture(model, [point]) as values, torch.no_grad():
            model(**tokenizer("target"))
            actual = values[point].clone()
        assert hook.error is None
        torch.testing.assert_close(actual, expected[:, :1].expand_as(actual), rtol=0, atol=0)
        torch.testing.assert_close(service._cache["source", point], expected, rtol=0, atol=0)
        assert previous.shape != expected.shape or not torch.equal(previous, expected)
        assert service._cache_identity[1]
        hook()
        assert not service._cache and not service._cache_identity[1]
    finally:
        service.close()
    assert service._cache_identity is None and not service._cache


def test_unchanged_tokenized_source_reuses_cached_tensor():
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).eval()
    service = InterventionService(model)
    service.configure(tokenizer=MutableTokenizer())
    spec = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "source"}
    try:
        first = service._source(spec)
        assert service._source(spec) is first
    finally:
        service.close()


def test_source_evidence_restores_with_the_control_transaction():
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).eval()
    tokenizer = MutableTokenizer()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    spec = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "source"}
    try:
        original = service._source(spec)
        cache, identity = service._cache.copy(), service._cache_identity
        tokenizer.add_bos_token = False
        changed = service._source(spec)
        assert changed.shape[1] == 2
        # Restore the same pair retained by control preparation and rollback.
        service._cache.clear()
        service._cache.update(cache)
        service._cache_identity = identity
        refreshed = service._source(spec)
        assert refreshed is not original
        torch.testing.assert_close(refreshed, changed, rtol=0, atol=0)
    finally:
        service.close()
