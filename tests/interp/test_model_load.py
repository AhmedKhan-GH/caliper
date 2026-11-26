"""Offline checkpoint provenance and bounded content checks on CPU tensors."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import _dictionary as dictionary
from caliper.interp._memory import CleanStop, MemoryGuard
from tools.interp import model_load


@pytest.fixture
def memory(monkeypatch):
    guard = MemoryGuard(12, "cpu", resident=lambda: 0, footprint=lambda: 0,
                        peak=lambda: 0, pressure=lambda: 50, driver=lambda: 0)
    monkeypatch.setattr(dictionary, "MemoryGuard", lambda *_: guard)
    return guard


class FixtureConfig(SimpleNamespace):
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return vars(self).copy()


class FixtureHFModel(torch.nn.Linear):
    def __init__(self, config):
        super().__init__(3, 2)
        self.config = config
        self.variance_epsilon = config.rms_norm_eps
        self.register_buffer("persistent", torch.arange(3, dtype=torch.float32))


def hf_model():
    config = FixtureConfig(model_type="synthetic_checkpoint", _name_or_path=model_load.MODEL_ID,
                           _commit_hash=model_load.MODEL_REVISION, rms_norm_eps=1e-6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        return FixtureHFModel(config).eval()


def change_weights(model, change):
    if change == "inplace":
        with torch.no_grad():
            model.weight.add_(1)
    elif change == "optimizer":
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
    elif change == "parameter":
        model.weight = torch.nn.Parameter(model.weight.detach() + 1)
    elif change == "data_replace":
        model.weight.data = model.weight.detach() + 1
    elif change == "data_mutate":
        model.weight.data.add_(1)
    elif change == "detached_alias":
        model.weight.detach().add_(1)
    elif change == "numpy_alias":
        model.weight.detach().numpy()[0, 0] += 1
    elif change == "persistent":
        model.persistent.data.add_(1)
    elif change == "transient":
        model.transient.data.add_(1)
    elif change == "dtype":
        model.double()
    elif change == "random":
        model.reset_parameters()
    else:
        raise AssertionError(change)


CHANGES = ["inplace", "optimizer", "parameter", "data_replace", "data_mutate",
           "detached_alias", "numpy_alias", "persistent", "transient", "dtype", "random"]


@pytest.mark.parametrize("change", CHANGES)
def test_verified_identity_rejects_all_weight_changes(memory, change):
    model = hf_model()
    if change == "transient":
        model.register_buffer("transient", torch.arange(2, dtype=torch.float32), persistent=False)
    dictionary._record_verified_hf_load(model, memory=memory)
    identity = f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
    assert dictionary.model_identity(model) == identity
    config = vars(model.config).copy()
    change_weights(model, change)
    assert vars(model.config) == config
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.model_identity(model)
    # Refusal precedes all dictionary file access and the trace forward.
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.load_dictionary("unused", model=model)
    from caliper.interp._frozen import _validate_dictionary
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_dictionary(model, SimpleNamespace(metadata={"model_hash": identity}))


@pytest.mark.parametrize("change", CHANGES)
def test_unverified_config_cannot_hide_changed_weights(change):
    model = hf_model()
    if change == "transient":
        model.register_buffer("transient", torch.arange(2, dtype=torch.float32), persistent=False)
    original = dictionary.model_identity(model)
    assert original.startswith("sha256:")
    change_weights(model, change)
    assert dictionary.model_identity(model) != original


def test_provenance_belongs_to_one_model_instance(memory):
    model = hf_model()
    dictionary._record_verified_hf_load(model, memory=memory)
    copied = deepcopy(model)
    assert dictionary.model_identity(model).startswith("hf:")
    assert dictionary.model_identity(copied).startswith("sha256:")
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.model_identity(model)


@pytest.mark.parametrize("field", ["variance_epsilon", "config_epsilon", "training"])
def test_trusted_forward_mutation_refuses_dictionary_before_file_access(memory, field):
    model = hf_model()
    dictionary._record_verified_hf_load(model, memory=memory)
    if field == "variance_epsilon":
        model.variance_epsilon = .1
    elif field == "config_epsilon":
        model.config.rms_norm_eps = .1
    else:
        model.train()
    with pytest.raises(ValueError, match="forward settings changed"):
        dictionary.load_dictionary("unused", model=model)


def test_unverified_model_refusal_precedes_content_transfer(monkeypatch):
    model = hf_model()
    monkeypatch.setattr(dictionary, "_UNVERIFIED_CPU_BYTES", 1)
    monkeypatch.setattr(torch.Tensor, "to", lambda *_a, **_k: pytest.fail("weight transfer ran"))
    with pytest.raises(ValueError, match="verify_hf_model"):
        dictionary.model_identity(model)


def test_hash_preserves_fixture_bytes():
    from .fixture import TinyLM
    model = TinyLM(layers=1, width=8, heads=2, latents=16, k=2)
    digest = hashlib.sha256(b"caliper.weights.v1\0")
    for name, value in sorted(model.state_dict().items()):
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(memoryview(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
    assert dictionary.weights_hash(model) == "sha256:" + digest.hexdigest()


@pytest.mark.parametrize("left,right,equal", [
    ("mps", "mps:0", True), ("mps:0", "mps", True),
    ("mps:0", "mps:0", True), ("mps", "mps:1", False),
    ("mps", "cpu", False), ("mps", "cuda:0", False),
    ("cuda", "cuda:0", False), ("cuda:0", "cuda:1", False),
])
def test_identity_device_normalization_without_device_access(left, right, equal, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("device normalization must not query or allocate on a device")

    monkeypatch.setattr(torch.backends.mps, "is_available", refuse)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", refuse)
    monkeypatch.setattr(torch.Tensor, "to", refuse)
    assert torch.device("mps") != torch.device("mps:0")
    assert (dictionary._canonical_device(torch.device(left)) ==
            dictionary._canonical_device(torch.device(right))) is equal


def test_digest_chunks_and_admission(memory, monkeypatch):
    model = torch.nn.Linear(1024, 513, bias=False)
    transfers = []
    convert = torch.Tensor.to

    def bounded(value, *args, **kwargs):
        transfers.append(value.numel() * value.element_size())
        return convert(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", bounded)
    dictionary.weights_hash(model, memory=memory)
    assert len(transfers) == 3
    assert max(transfers) <= dictionary._HASH_CHUNK_BYTES
    memory.peak = memory.limit - 1
    dictionary.weights_hash(model, memory=memory)
    memory.footprint = lambda: memory.limit - 1
    transfers.clear()
    with pytest.raises(CleanStop, match="identity allocation"):
        dictionary.weights_hash(model, memory=memory)
    assert not transfers


def test_noncontiguous_weights_refuse_before_copy(memory, monkeypatch):
    model = hf_model()
    model.weight.data = model.weight.data.T
    monkeypatch.setattr(torch.Tensor, "to", lambda *_a, **_k: pytest.fail("weight transfer ran"))
    monkeypatch.setattr(torch.Tensor, "contiguous", lambda *_a, **_k: pytest.fail("full copy ran"))
    with pytest.raises(ValueError, match="contiguous"):
        dictionary.weights_hash(model, memory=memory)


def local_snapshot(tmp_path, model, *, prefix=""):
    snapshot = tmp_path / "models--google--gemma-2-2b" / "snapshots" / model_load.MODEL_REVISION
    snapshot.mkdir(parents=True)
    save_file({prefix + name: value.detach().clone() for name, value in model.state_dict().items()},
              snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps(model.config.to_dict()))
    return snapshot


@pytest.mark.parametrize("prefix", ["", "source."])
def test_external_snapshot_verification_and_subsequent_change(tmp_path, memory, prefix):
    from caliper.interp import verify_hf_model
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model, prefix=prefix)
    mapping = {prefix: ""} if prefix else None
    assert dictionary.model_identity(model).startswith("sha256:")
    identity = verify_hf_model(model, snapshot, key_mapping=mapping)
    assert identity == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
    assert dictionary.model_identity(model) == identity
    model.weight.data.add_(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.model_identity(model)


def test_external_settings_reference_has_no_tensor_storage(tmp_path, memory, monkeypatch):
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model)
    initialize = FixtureHFModel.__init__
    devices = []

    def reference(instance, config):
        initialize(instance, config)
        devices.extend(value.device.type for value in list(instance.parameters()) + list(instance.buffers()))

    monkeypatch.setattr(FixtureHFModel, "__init__", reference)
    rng = torch.get_rng_state()
    assert dictionary.verify_hf_model(model, snapshot).startswith("hf:")
    assert devices and set(devices) == {"meta"}
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("damage", ["random", "missing", "extra", "shape", "revision", "mapping"])
def test_external_snapshot_refuses_unverified_weights(tmp_path, memory, damage):
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    mapping = None
    if damage == "random":
        model.reset_parameters()
    elif damage == "missing":
        del state["weight"]
    elif damage == "extra":
        state["extra"] = torch.zeros(1)
    elif damage == "shape":
        state["weight"] = state["weight"][:1]
    elif damage == "revision":
        model.config._commit_hash = "b" * 40
    else:
        mapping = {"weight": "weight", "w": "w"}
    save_file(state, snapshot / "model.safetensors")
    with pytest.raises(ValueError):
        dictionary.verify_hf_model(model, snapshot, key_mapping=mapping)
    assert model not in dictionary._VERIFIED_MODELS


def test_snapshot_verification_handles_tied_weights_and_dtype(tmp_path, memory):
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model)
    model.register_parameter("tied", model.weight)
    model.double()
    assert dictionary.verify_hf_model(model, snapshot).startswith("hf:")
    assert dictionary.model_identity(model).startswith("hf:")


def test_snapshot_chunks_cover_wide_rows_scalars_and_empty_tensors(tmp_path, memory, monkeypatch):
    model = hf_model()
    model.register_buffer("scalar", torch.tensor(1.0))
    model.register_buffer("empty", torch.empty(0))
    snapshot = local_snapshot(tmp_path, model)
    monkeypatch.setattr(dictionary, "_HASH_CHUNK_BYTES", 8)
    assert dictionary.verify_hf_model(model, snapshot).startswith("hf:")
    model.weight.data[1, 2] += 1
    with pytest.raises(ValueError, match="local checkpoint"):
        dictionary.verify_hf_model(model, snapshot)


def test_verified_digest_honors_lower_load_limit(memory, monkeypatch):
    model = hf_model()
    dictionary._record_verified_hf_load(model, memory=memory)
    memory.peak = 6_000_000_000
    dictionary.model_identity(model, memory_limit_gb=6)
    memory.footprint = lambda: 6_000_000_000
    monkeypatch.setattr(torch.Tensor, "to", lambda *_a, **_k: pytest.fail("weight transfer ran"))
    with pytest.raises(CleanStop, match="identity allocation"):
        dictionary.model_identity(model, memory_limit_gb=6)


def test_public_model_dictionary_verification_and_random_weight_refusal(tmp_path, memory):
    from .test_trace import random_dictionary
    from .test_gemma_family import gemma_model

    model = gemma_model(layers=1)
    model.config._name_or_path = model_load.MODEL_ID
    model.config._commit_hash = model_load.MODEL_REVISION
    snapshot = local_snapshot(tmp_path, model)
    # A direct Windows snapshot name has the same checkpoint identity.
    model.config._name_or_path = str(snapshot).replace("/", "\\")
    dictionary.verify_hf_model(model, snapshot)
    source = random_dictionary(model, latents=8, k=2)
    source.metadata["model_id"] = model_load.MODEL_ID
    folder = tmp_path / "dictionary"
    folder.mkdir()
    (folder / "dictionary.json").write_text(json.dumps(source.metadata))
    save_file(source.layers[0].state_dict(), folder / "layer_00.safetensors")
    assert dictionary.load_dictionary(folder, model=model).metadata == source.metadata
    other = deepcopy(model)
    other.model.layers[0].mlp.down_proj.weight.data.add_(1)
    assert vars(other.config) == vars(model.config)
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.load_dictionary(folder, model=other)
    with pytest.raises(ValueError, match="local checkpoint"):
        dictionary.verify_hf_model(other, snapshot)


def mocked_loader(tmp_path, monkeypatch, model, info):
    snapshot = local_snapshot(tmp_path, model)
    calls = []

    def download(repository, **kwargs):
        assert repository == model_load.MODEL_ID
        assert kwargs == {"revision": model_load.MODEL_REVISION, "local_files_only": True}
        calls.append("snapshot")
        return str(snapshot)

    def load(path, **kwargs):
        assert path == snapshot
        assert kwargs["local_files_only"] is True
        assert kwargs["output_loading_info"] is True
        assert kwargs["dtype"] == torch.float32
        assert kwargs["trust_remote_code"] is False
        calls.append("weights")
        return model, info

    def tokenizer(path, **kwargs):
        assert path == snapshot and kwargs == {"local_files_only": True, "trust_remote_code": False}
        calls.append("tokenizer")
        return object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Gemma2Config=lambda **kwargs: SimpleNamespace(model_type="gemma2", num_hidden_layers=26, hidden_size=2304, vocab_size=256000),
        Gemma2ForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    return calls


def test_trusted_loader_preserves_existing_manifest_identity(tmp_path, monkeypatch, memory):
    model = hf_model()
    calls = mocked_loader(tmp_path, monkeypatch, model, {})
    loaded, _ = model_load.load_text_model(device="cpu", memory=memory)
    assert loaded is model
    assert calls == ["snapshot", "weights", "tokenizer"]
    assert dictionary.model_identity(model) == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
    assert not model.training and not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_trusted_load_never_records_provenance(tmp_path, monkeypatch, memory, key):
    model = hf_model()
    mocked_loader(tmp_path, monkeypatch, model, {key: ["weight"]})
    with pytest.raises(ValueError, match="weights do not match"):
        model_load.load_text_model(device="cpu", memory=memory)
    assert model not in dictionary._VERIFIED_MODELS
