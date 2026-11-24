"""Offline Gemma runtime checks with small random CPU models."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _admission, _dictionary, _hosted, _index
from caliper.interp._admission import GEMMA_ID, GEMMA_REVISION
from caliper.interp._memory import CleanStop, MemoryGuard
from caliper.interp._output import forward_logits
from tools.interp import model_load


def gemma_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma2Config(
        vocab_size=37, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, sliding_window=4, attn_logit_softcapping=3.0,
        final_logit_softcapping=2.0, query_pre_attn_scalar=8,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = transformers.Gemma2ForCausalLM(config).eval().requires_grad_(False)
    model.config._name_or_path, model.config._commit_hash = GEMMA_ID, GEMMA_REVISION
    return model


def memory(limit=12, **kwargs):
    return MemoryGuard(limit, "cpu", resident=lambda: 0, footprint=lambda: 0,
                       peak=lambda: 0, driver=lambda: 0, pressure=lambda: 80, **kwargs)


def local_gemma(tmp_path, model):
    snapshot = tmp_path / "models--google--gemma-2-2b" / "snapshots" / GEMMA_REVISION
    model.save_pretrained(snapshot, safe_serialization=True)
    return snapshot


def test_gemma_memory_ceiling_and_staged_admission():
    guard = memory(28)
    guard.peak = 20_000_000_000
    _admission._admit_load(guard, 12_000_000_000, "transfer", staged=True)
    with pytest.raises(CleanStop, match="model allocation"):
        _admission._admit_load(guard, 12_000_000_000, "legacy")
    with pytest.raises(ValueError, match="28 GB"):
        memory(28.01)
    over = MemoryGuard(28, "cpu", resident=lambda: 28_000_000_001, footprint=lambda: None,
                       peak=lambda: 0, driver=lambda: 0, pressure=lambda: 80)
    with pytest.raises(CleanStop, match="memory limit"):
        over.check("over budget")


def test_gemma_checkpoint_verifies_tied_weights_and_nonpersistent_buffers(tmp_path):
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert "model.embed_tokens.embed_scale" not in model.state_dict()
    identity = _dictionary.verify_hf_model(model, snapshot)
    assert identity == f"hf:{GEMMA_ID}@{GEMMA_REVISION}"
    model.model.embed_tokens.embed_scale.data.add_(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        _dictionary.model_identity(model)
    with pytest.raises(ValueError, match="nonpersistent buffers"):
        _dictionary.verify_hf_model(model, snapshot)


@pytest.mark.parametrize("change", ["softcap", "sliding", "scale", "tied"])
def test_gemma_checkpoint_refuses_effective_setting_or_tied_weight_changes(tmp_path, change):
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    _dictionary.verify_hf_model(model, snapshot)
    if change == "softcap":
        model.config.final_logit_softcapping *= 2
    elif change == "sliding":
        model.model.layers[0].self_attn.sliding_window = 7
    elif change == "scale":
        model.model.embed_tokens.scalar_embed_scale *= 2
    else:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone() + 1)
    with pytest.raises(ValueError, match="identity mismatch"):
        _dictionary.model_identity(model)
    with pytest.raises(ValueError, match="settings|weights|tensor|checkpoint"):
        _dictionary.verify_hf_model(model, snapshot)


def test_gemma_output_uses_real_softcapped_forward_and_last_token():
    model = gemma_model()
    ids = torch.tensor([[1, 5, 9, 7, 6], [1, 5, 9, 7, 6]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        actual = forward_logits(model, {"input_ids": ids, "attention_mask": mask})
    assert actual.shape == (2, 1, 37)
    torch.testing.assert_close(actual[:, -1], expected[:, -1])
    assert actual.abs().max() <= model.config.final_logit_softcapping


def test_gemma_hidden_forward_preserves_masks_without_head_projection(monkeypatch):
    model = gemma_model()
    ids = torch.tensor([[1, 5, 9, 0], [1, 7, 6, 2]])
    mask = (ids != 0).long()
    with torch.no_grad():
        expected = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    monkeypatch.setattr(model.lm_head, "forward", lambda *a: pytest.fail("vocabulary projection ran"))
    calls = []
    handle = model.model.layers[0].mlp.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        with torch.no_grad():
            actual = model_load.hidden_forward(model, ids, mask)
        torch.testing.assert_close(actual, expected)
        assert calls == [1]
    finally:
        handle.remove()


def mock_gemma_load(tmp_path, monkeypatch, *, info=None):
    import huggingface_hub
    import transformers
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    calls = []

    def cached(repo, *, revision, local_files_only):
        assert (repo, revision, local_files_only) == (GEMMA_ID, GEMMA_REVISION, True)
        calls.append("snapshot")
        return str(snapshot)

    def loaded(folder, **kwargs):
        assert folder == snapshot
        assert kwargs["local_files_only"] and kwargs["use_safetensors"]
        assert not kwargs["trust_remote_code"] and kwargs["attn_implementation"] == "eager"
        calls.append("weights")
        return model, info or {}

    def tokenizer(folder, **kwargs):
        assert folder == snapshot and kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        calls.append("tokenizer")
        return object()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    monkeypatch.setattr(transformers, "Gemma2Config", lambda **kw: SimpleNamespace(
        model_type="gemma2", num_hidden_layers=26, hidden_size=2304, vocab_size=256000))
    monkeypatch.setattr(transformers.Gemma2ForCausalLM, "from_pretrained", loaded)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    return model, snapshot, calls


def test_explicit_gemma_loader_uses_only_pinned_local_snapshot(tmp_path, monkeypatch):
    model, _, calls = mock_gemma_load(tmp_path, monkeypatch)
    loaded, _ = model_load.load_text_model(model_id=GEMMA_ID, device="cpu", memory=memory())
    assert loaded is model and calls == ["snapshot", "weights", "tokenizer"]
    assert _dictionary.model_identity(model) == f"hf:{GEMMA_ID}@{GEMMA_REVISION}"


def test_gemma_incomplete_load_never_records_identity(tmp_path, monkeypatch):
    model, _, calls = mock_gemma_load(tmp_path, monkeypatch, info={"missing_keys": ["model.norm.weight"]})
    with pytest.raises(ValueError, match="weights do not match"):
        model_load.load_text_model(model_id=GEMMA_ID, device="cpu", memory=memory())
    assert model not in _dictionary._VERIFIED_MODELS
    assert calls == ["snapshot", "weights"]


def test_gemma_hosted_load_passes_budget_to_dictionary_and_index(tmp_path, monkeypatch):
    model, snapshot, calls = mock_gemma_load(tmp_path, monkeypatch)
    guard = memory(28)
    from caliper.interp import _memory
    monkeypatch.setattr(_memory, "MemoryGuard", lambda *a: guard)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    verify = _dictionary.verify_hf_model

    def verify_cpu(model, folder, *, memory_limit_gb):
        assert memory_limit_gb == 28
        # The dictionary worker supplies the wider verifier cap at integration.
        return verify(model, folder, memory_limit_gb=12)

    def dictionary(*a, **kwargs):
        assert kwargs["memory_limit_gb"] == 28 and kwargs["model"] is model
        calls.append("dictionary")
        return object()

    def index(*a, **kwargs):
        assert kwargs["memory"] is guard and kwargs["model"] is model
        calls.append("index")
        return object()

    monkeypatch.setattr(_dictionary, "verify_hf_model", verify_cpu)
    monkeypatch.setattr(_dictionary, "load_dictionary", dictionary)
    monkeypatch.setattr(_index, "FeatureIndex", index)
    loaded, _, _, _, actual_guard = _hosted.load_resources(str(snapshot), tmp_path)
    assert loaded is model and actual_guard is guard
    assert calls == ["weights", "tokenizer", "dictionary", "index"]
