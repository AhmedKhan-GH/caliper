"""Arithmetic context evidence with small CPU tensors and local checkpoints."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._dictionary import _state_hash, model_identity, verify_hf_model
from caliper.interp._forward_identity import arithmetic_settings, forward_settings, source_settings
from caliper.interp._intervene import InterventionService
from .test_gemma_runtime import gemma_model, local_gemma


@contextmanager
def arithmetic(mode):
    precision = torch.get_float32_matmul_precision()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    try:
        torch.set_float32_matmul_precision(mode if mode in ("high", "medium") else "highest")
        # Nested dtype changes must not reuse PyTorch's cached weight casts.
        with torch.autocast("cpu", enabled=dtype is not None, dtype=dtype or torch.bfloat16,
                            cache_enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.fixture(autouse=True)
def default_arithmetic():
    with arithmetic("default"):
        yield


class LinearPatchModel(torch.nn.Module):
    """Expose a rounding difference at one scalar activation."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=2)
        self.embed = torch.nn.Embedding(2, 1)
        self.head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor([[1.003], [2.0]]))
            self.head.weight.fill_(1.003)
            self.head.bias.fill_(-1)
        self.eval()

    def forward(self, input_ids, use_cache=False):
        return self.head(self.embed(input_ids))


def tokenizer(prompt, **kwargs):
    return {"input_ids": torch.tensor([[0 if prompt == "source" else 1]])}


def test_canonical_default_adds_no_fields_and_ignores_disabled_dtype():
    model = LinearPatchModel()
    baseline = source_settings(model, tensor_hash=_state_hash)
    assert arithmetic_settings(model) == {}
    with torch.autocast("cpu", enabled=False, dtype=torch.float16):
        assert arithmetic_settings(model) == {}
        assert source_settings(model, tensor_hash=_state_hash) == baseline


@pytest.mark.parametrize(("factory", "checkpoint", "source"), [
    (gemma_model, "7e5f76d90a9b78b9397b7c57b3dcfbbc5d42950533fb3d7b345732f9d9adbe0a",
     "f1165d903a00f7e44de0e68fafe80dd177d03d2ff851072079612e6f1794ebdf"),
])
def test_default_checkpoint_and_source_hashes_match_legacy(factory, checkpoint, source):
    # These hashes come from the parent revision, before arithmetic evidence.
    model = factory()
    for function, expected in ((forward_settings, checkpoint), (source_settings, source)):
        raw = json.dumps(function(model, tensor_hash=_state_hash), sort_keys=True).encode()
        assert hashlib.sha256(raw).hexdigest() == expected


@pytest.mark.parametrize("mode", ["bf16", "fp16", "high", "medium"])
def test_source_identity_binds_arithmetic_and_restores_default(mode):
    model = LinearPatchModel()
    baseline = source_settings(model, tensor_hash=_state_hash)
    checkpoint = forward_settings(model, tensor_hash=_state_hash)
    with arithmetic(mode):
        assert arithmetic_settings(model)
        assert source_settings(model, tensor_hash=_state_hash) != baseline
        assert forward_settings(model, tensor_hash=_state_hash) == checkpoint
    assert arithmetic_settings(model) == {}
    assert source_settings(model, tensor_hash=_state_hash) == baseline


def test_autocast_dtype_changes_source_identity():
    model = LinearPatchModel()
    with arithmetic("bf16"):
        first = source_settings(model, tensor_hash=_state_hash)
        with arithmetic("fp16"):
            assert source_settings(model, tensor_hash=_state_hash) != first
        assert source_settings(model, tensor_hash=_state_hash) == first


def test_device_specific_autocast_uses_parameter_and_buffer_devices(monkeypatch):
    # Device descriptors require no GPU initialization or tensor allocation.
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:1"))]),
        buffers=lambda: iter([SimpleNamespace(device=torch.device("mps"))]))
    active = {"cpu": False, "cuda": True, "mps": True}
    dtypes = {"cuda": torch.float16, "mps": torch.bfloat16}
    checked = []

    def enabled(device):
        checked.append(device)
        return active[device]

    monkeypatch.setattr(torch, "is_autocast_enabled", enabled)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: dtypes[device])
    first = arithmetic_settings(model)
    assert checked == ["cpu", "cuda", "mps"]
    assert json.loads(first["__arithmetic_context__"])["autocast"] == {
        "cuda": "torch.float16", "mps": "torch.bfloat16"}
    dtypes["cuda"] = torch.bfloat16
    assert arithmetic_settings(model) != first
    active.update(cuda=False, mps=False)
    assert arithmetic_settings(model) == {}


def test_parameterless_source_binds_cpu_autocast():
    model = torch.nn.Identity()
    assert arithmetic_settings(model) == {}
    with arithmetic("bf16"):
        assert arithmetic_settings(model)


def test_patch_recaptures_actual_cpu_activation_across_arithmetic_contexts():
    model = LinearPatchModel()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    spec = {"kind": "patch", "point": "head", "source_prompt": "source"}
    try:
        hook = service.install(spec)
        original = service._cache["source", "head"]
        assert original.dtype == torch.float32
        assert original.item() == pytest.approx(0.006009, abs=1e-7)
        previous = original
        for mode, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16),
                            ("high", torch.float32), ("default", torch.float32)):
            with arithmetic(mode), torch.no_grad():
                with service.suspend():
                    expected = model(**tokenizer("source"))
                actual = model(**tokenizer("target"))
                assert hook.error is None
                assert actual.dtype == dtype
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                refreshed = service._cache["source", "head"]
                assert refreshed is not previous
                assert service._source(spec) is refreshed
                if mode == "bf16":
                    assert expected.item() == 0
                    assert original.to(dtype).item() != expected.item()
                previous = refreshed
        torch.testing.assert_close(previous, original, rtol=0, atol=0)
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["bf16", "high"])
def test_checkpoint_verification_keeps_weight_and_setting_checks(tmp_path, mode):
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    identity = verify_hf_model(model, snapshot)
    with arithmetic(mode):
        assert verify_hf_model(model, snapshot) == identity
        assert model_identity(model) == identity
        original = model.config.final_logit_softcapping
        model.config.final_logit_softcapping = original + 1
        with pytest.raises(ValueError, match="forward settings"):
            model_identity(model)
        model.config.final_logit_softcapping = original
        model.model.layers[0].mlp.down_proj.weight.data.add_(.125)
        with pytest.raises(ValueError, match="weights changed"):
            model_identity(model)
        with pytest.raises(ValueError, match="weights|tensor"):
            verify_hf_model(model, snapshot)
