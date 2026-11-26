"""Initial checkpoint identity for buffers that affect Gemma forward execution."""
import pytest
import torch

from caliper.interp import _dictionary as dictionary
from caliper.interp import _forward_identity as forward_identity
from caliper.interp._memory import CleanStop
from tools.interp import model_load
from .test_model_load import FixtureHFModel, hf_model, local_snapshot, memory
from .test_gemma_family import gemma_model


def rotary_snapshot(tmp_path, *, layers=1):
    model = gemma_model(layers=layers)
    model.config._name_or_path = model_load.MODEL_ID
    model.config._commit_hash = model_load.MODEL_REVISION
    return model, local_snapshot(tmp_path, model)


def test_initial_rotary_mutation_changes_logits_and_refuses_identity(tmp_path, memory, monkeypatch):
    import safetensors

    model, snapshot = rotary_snapshot(tmp_path, layers=4)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        original = model(tokens, use_cache=False).logits
        model.model.rotary_emb.inv_freq.zero_()
        changed = model(tokens, use_cache=False).logits
    assert (changed - original).abs().max().item() > 1e-6
    assert "model.rotary_emb.inv_freq" not in model.state_dict()
    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: pytest.fail("checkpoint read ran"))
    with pytest.raises(ValueError, match="nonpersistent buffers"):
        dictionary.verify_hf_model(model, snapshot)
    assert model not in dictionary._VERIFIED_MODELS


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotary_reference_allocates_only_bounded_buffers(tmp_path, memory, monkeypatch, dtype):
    from transformers import Gemma2ForCausalLM
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding

    model, snapshot = rotary_snapshot(tmp_path)
    model.to(dtype=dtype)
    initialize_model = Gemma2ForCausalLM.__init__
    initialize_rotary = Gemma2RotaryEmbedding.__init__
    parameter_devices, buffer_devices, allocations = [], [], []

    def reference(instance, config):
        initialize_model(instance, config)
        parameter_devices.extend(value.device.type for value in instance.parameters())
        buffer_devices.extend(value.device.type for value in instance.buffers())

    def rotary(instance, config):
        initialize_rotary(instance, config)
        if instance.inv_freq.device.type == "cpu":
            assert not list(instance.parameters())
            allocations.append(sum(value.numel() * value.element_size() for value in instance.buffers()))

    monkeypatch.setattr(Gemma2ForCausalLM, "__init__", reference)
    monkeypatch.setattr(Gemma2RotaryEmbedding, "__init__", rotary)
    rng = torch.get_rng_state()
    assert dictionary.verify_hf_model(model, snapshot).startswith("hf:")
    assert set(parameter_devices) == set(buffer_devices) == {"meta"}
    assert allocations == [32]
    assert torch.equal(torch.get_rng_state(), rng)
    model.model.rotary_emb.inv_freq.data.add_(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.model_identity(model)


@pytest.mark.parametrize("damage", ["original", "missing", "none", "extra_alias", "persistent"])
def test_rotary_buffer_values_and_registration_require_trusted_baseline(tmp_path, memory, damage):
    model, snapshot = rotary_snapshot(tmp_path)
    rotary = model.model.rotary_emb
    if damage == "original":
        rotary.original_inv_freq.add_(1)
    elif damage == "missing":
        del rotary.inv_freq
    elif damage == "none":
        rotary.inv_freq = None
    elif damage == "extra_alias":
        rotary.register_buffer("extra", rotary.inv_freq, persistent=False)
    else:
        rotary._non_persistent_buffers_set.remove("inv_freq")
    with pytest.raises(ValueError, match="nonpersistent buffers"):
        dictionary.verify_hf_model(model, snapshot)
    assert model not in dictionary._VERIFIED_MODELS


def test_unknown_buffer_constructor_has_no_cpu_reconstruction(tmp_path, memory, monkeypatch):
    class UnknownBuffers(FixtureHFModel):
        def __init__(self, config):
            super().__init__(config)
            self.register_buffer("transient", torch.arange(2, dtype=torch.float32), persistent=False)

    model = UnknownBuffers(hf_model().config).eval()
    snapshot = local_snapshot(tmp_path, model)
    initialize = UnknownBuffers.__init__

    def reference(instance, config):
        assert torch.empty(0).device.type == "meta"
        initialize(instance, config)

    monkeypatch.setattr(UnknownBuffers, "__init__", reference)
    with pytest.raises(ValueError, match="unsupported checkpoint nonpersistent buffers"):
        dictionary.verify_hf_model(model, snapshot)


@pytest.mark.parametrize("limit", ["bytes", "headroom"])
def test_rotary_reconstruction_refuses_before_cpu_allocation(tmp_path, memory, monkeypatch, limit):
    from transformers import Gemma2ForCausalLM
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding

    model, snapshot = rotary_snapshot(tmp_path)
    initialize_model = Gemma2ForCausalLM.__init__
    initialize_rotary = Gemma2RotaryEmbedding.__init__

    def reference(instance, config):
        initialize_model(instance, config)
        if limit == "headroom":
            memory.footprint = lambda: memory.limit - 1

    def rotary(instance, config):
        assert torch.empty(0).device.type == "meta"
        initialize_rotary(instance, config)

    monkeypatch.setattr(Gemma2ForCausalLM, "__init__", reference)
    monkeypatch.setattr(Gemma2RotaryEmbedding, "__init__", rotary)
    if limit == "bytes":
        monkeypatch.setattr(forward_identity, "_MAX_CHECKPOINT_BUFFER_BYTES", 1)
    with pytest.raises(ValueError if limit == "bytes" else CleanStop, match="reconstruction"):
        dictionary.verify_hf_model(model, snapshot)
    assert model not in dictionary._VERIFIED_MODELS


def test_buffer_mutation_during_header_read_cannot_become_the_baseline(tmp_path, memory, monkeypatch):
    import safetensors

    model, snapshot = rotary_snapshot(tmp_path)
    original = safetensors.safe_open

    def mutate(*args, **kwargs):
        model.model.rotary_emb.inv_freq.zero_()
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", mutate)
    with pytest.raises(ValueError, match="nonpersistent buffers changed during"):
        dictionary.verify_hf_model(model, snapshot)
    assert model not in dictionary._VERIFIED_MODELS
