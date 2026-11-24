"""Patch source admission uses the normal untruncated prompt contract."""
from collections import UserDict

import pytest
import torch
from transformers import BatchEncoding

from caliper.interp._intervene import InterventionError, InterventionService
from caliper.interp._output import prompt_inputs
from .fixture import TinyLM


RETURN_TYPES = [dict, UserDict, BatchEncoding]
SPEC = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "source"}


def hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


@pytest.mark.parametrize("return_type", RETURN_TYPES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [1, 32])
def test_patch_accepts_supported_tokenizer_mappings(return_type, dtype, masked, length):
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1).cpu()
    calls = []
    values = {"input_ids": torch.full((1, length), 65, dtype=dtype)}
    if masked:
        values["attention_mask"] = torch.ones(1, length, dtype=torch.int64)
    # Normal prompt admission forwards only input_ids and attention_mask.
    values["token_type_ids"] = torch.zeros(1, length, dtype=torch.int64)
    def tokenizer(text, **kwargs):
        calls.append(kwargs)
        return return_type(values)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    before = hook_counts(model)
    try:
        normal = prompt_inputs(model, tokenizer, "source")
        source = service._source_inputs(SPEC)
        assert set(normal) == set(source) == ({"input_ids", "attention_mask"} if masked else {"input_ids"})
        for name in normal:
            assert torch.equal(source[name], normal[name])
        hook = service.install(SPEC)
        with torch.no_grad():
            model(**normal)
        assert hook.error is None and hook.calls == 1
        assert all(call == {"return_tensors": "pt", "truncation": False} for call in calls)
    finally:
        service.close()
    assert hook_counts(model) == before


@pytest.mark.parametrize("return_type", RETURN_TYPES)
@pytest.mark.parametrize("invalid", ["long", "empty", "batch", "rank", "float", "bool", "int16",
                                     "negative", "vocabulary", "mask_shape", "padding", "mask_value"])
def test_patch_refuses_invalid_prompt_before_any_install(return_type, invalid, monkeypatch):
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1).cpu()
    ids = torch.tensor([[65, 66]])
    mask = torch.ones_like(ids)
    if invalid == "long":
        ids = torch.full((1, 33), 65)
    elif invalid == "empty":
        ids = ids[:, :0]
    elif invalid == "batch":
        ids = ids.repeat(2, 1)
    elif invalid == "rank":
        ids = ids[0]
    elif invalid in ("float", "bool", "int16"):
        ids = ids.to({"float": torch.float32, "bool": torch.bool, "int16": torch.int16}[invalid])
    elif invalid == "negative":
        ids[0, 0] = -1
    elif invalid == "vocabulary":
        ids[0, 0] = model.config.vocab_size
    elif invalid == "mask_shape":
        mask = mask[:, :1]
    elif invalid in ("padding", "mask_value"):
        mask[0, 0] = 0 if invalid == "padding" else 2
    calls = []
    def tokenizer(text, *, truncation=True, **kwargs):
        calls.append(truncation)
        # A default truncation policy must never make the long source valid.
        result = ids[:, :32] if invalid == "long" and truncation else ids
        return return_type({"input_ids": result, "attention_mask": mask})
    def forbidden(*args, **kwargs):
        raise AssertionError("Prompt refusal must precede a model forward")
    monkeypatch.setattr(model, "forward", forbidden)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    before = hook_counts(model)
    try:
        with pytest.raises(ValueError) as normal:
            prompt_inputs(model, tokenizer, "source")
        for action in (service.validate, service.install):
            with pytest.raises(InterventionError) as refused:
                action(SPEC)
            assert refused.value.code == "intervene.bad_request"
            assert str(normal.value) in str(refused.value)
        assert calls == [False, False, False]
        assert hook_counts(model) == before
        assert not service._hooks and not service._cache
    finally:
        service.close()


@pytest.mark.parametrize("result", [None, [], torch.tensor([[65]]), {}, {"input_ids": [[65]]},
                                    {"input_ids": None}, {"input_ids": torch.tensor([[65]]), "attention_mask": [1]}])
def test_malformed_tokenizer_returns_keep_structured_refusal(result):
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).cpu()
    service = InterventionService(model)
    service.configure(tokenizer=lambda *args, **kwargs: result)
    before = hook_counts(model)
    try:
        with pytest.raises(InterventionError) as refused:
            service.install(SPEC)
        assert refused.value.code == "intervene.bad_request"
        assert hook_counts(model) == before and not service._cache
    finally:
        service.close()


@pytest.mark.parametrize("positions", [{"positions": [2]}, {"source_position": 2}])
def test_patch_positions_remain_bounded_by_validated_source(positions):
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).cpu()
    service = InterventionService(model)
    service.configure(tokenizer=lambda *args, **kwargs: {"input_ids": torch.tensor([[65, 66]])})
    try:
        with pytest.raises(InterventionError, match="source position is outside"):
            service.install(dict(SPEC, **positions))
        assert not service._hooks and not service._cache
    finally:
        service.close()


def test_tokenizer_exception_keeps_structured_refusal():
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).cpu()
    def fail(*args, **kwargs):
        raise RuntimeError("fixture tokenizer failure")
    service = InterventionService(model)
    service.configure(tokenizer=fail)
    try:
        with pytest.raises(InterventionError, match="source prompt tokenization failed") as refused:
            service.install(SPEC)
        assert refused.value.code == "intervene.bad_request"
        assert not service._hooks and not service._cache
    finally:
        service.close()
