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


def hf_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = torch.nn.Linear(3, 2)
    model.config = SimpleNamespace(model_type="qwen3_5_text", _name_or_path=model_load.MODEL_ID,
                                   _commit_hash=model_load.MODEL_REVISION)
    model.register_buffer("persistent", torch.arange(3, dtype=torch.float32))
    model.register_buffer("transient", torch.arange(2, dtype=torch.float32), persistent=False)
    return model


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
    snapshot = tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / model_load.MODEL_REVISION
    snapshot.mkdir(parents=True)
    save_file({prefix + name: value.detach().clone() for name, value in model.state_dict().items()},
              snapshot / "model.safetensors")
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


def text_subset_snapshot(tmp_path):
    model = torch.nn.Module()
    model.config = hf_model().config
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2, dtype=torch.bfloat16)
    model.model.norm = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.lm_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
    model.lm_head.weight = model.model.embed_tokens.weight
    state = {"model.language_model." + name: value.detach().clone()
             for name, value in model.model.state_dict().items()}
    _, excluded = dictionary._checkpoint_subset(
        model, dictionary._hf_identity(model.config), "qwen3.5-0.8b-text", None)
    assert len(excluded) == 168
    state.update({name: torch.zeros(1) for name in excluded})
    snapshot = tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / model_load.MODEL_REVISION
    snapshot.mkdir(parents=True)
    save_file(state, snapshot / "model.safetensors")
    return model.float(), snapshot, state


def test_public_text_subset_verifies_tied_weights_without_source_changes(tmp_path, memory, monkeypatch):
    import safetensors
    from caliper.interp import verify_hf_model
    model, snapshot, state = text_subset_snapshot(tmp_path)
    checkpoint = snapshot / "model.safetensors"
    before = checkpoint.read_bytes()
    original = safetensors.safe_open

    class TextOnly:
        def __init__(self, *args, **kwargs):
            self.source = original(*args, **kwargs)

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def keys(self):
            return self.source.keys()

        def get_slice(self, key):
            assert key.startswith("model.language_model.")
            return self.source.get_slice(key)

    monkeypatch.setattr(safetensors, "safe_open", TextOnly)
    identity = verify_hf_model(model, snapshot, checkpoint_subset="qwen3.5-0.8b-text")
    assert identity == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
    assert dictionary.model_identity(model) == identity
    assert checkpoint.read_bytes() == before
    model.lm_head.weight.data[0, 0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        dictionary.model_identity(model)


@pytest.mark.parametrize("damage", ["strict", "text_extra", "vision_extra", "mtp_extra", "missing_text",
                                  "missing_vision", "untied_head", "duplicate", "mapping", "subset", "weights"])
def test_public_text_subset_refuses_incomplete_or_undeclared_weights(tmp_path, memory, damage):
    from caliper.interp import verify_hf_model
    model, snapshot, state = text_subset_snapshot(tmp_path)
    options = {"checkpoint_subset": "qwen3.5-0.8b-text"}
    if damage == "strict":
        options = {"key_mapping": {"model.language_model.": "model."}}
    elif damage in ("text_extra", "vision_extra", "mtp_extra"):
        prefix = {"text_extra": "model.language_model.", "vision_extra": "model.visual.", "mtp_extra": "mtp."}[damage]
        state[prefix + "undeclared.weight"] = torch.zeros(1)
    elif damage == "missing_text":
        del state["model.language_model.norm.weight"]
    elif damage == "missing_vision":
        del state["model.visual.pos_embed.weight"]
    elif damage == "untied_head":
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    elif damage == "duplicate":
        save_file({"model.visual.pos_embed.weight": torch.zeros(1)}, snapshot / "extra.safetensors")
    elif damage == "mapping":
        options["key_mapping"] = {"model.language_model.": "model."}
    elif damage == "subset":
        options["checkpoint_subset"] = "unknown"
    elif damage == "weights":
        model.model.norm.weight.data[0, 0] += 1
    save_file(state, snapshot / "model.safetensors")
    with pytest.raises(ValueError):
        verify_hf_model(model, snapshot, **options)
    assert model not in dictionary._VERIFIED_MODELS


@pytest.mark.parametrize("field,value", [("_commit_hash", "b" * 40), ("model_type", "other")])
def test_text_subset_requires_the_declared_revision_and_architecture(field, value):
    model = hf_model()
    setattr(model.config, field, value)
    with pytest.raises(ValueError, match="unsupported checkpoint subset"):
        dictionary._checkpoint_subset(model, dictionary._hf_identity(model.config), "qwen3.5-0.8b-text", None)


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
    monkeypatch.setattr(torch.Tensor, "to", lambda *_a, **_k: pytest.fail("weight transfer ran"))
    with pytest.raises(CleanStop, match="identity allocation"):
        dictionary.model_identity(model, memory_limit_gb=6)


def test_public_model_dictionary_verification_and_random_weight_refusal(tmp_path, memory):
    from .test_trace import qwen_for_trace, random_dictionary

    model = qwen_for_trace(layers=1)
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
    (snapshot / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 24, "hidden_size": 1024}}))
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
        assert kwargs["key_mapping"] == {r"^model\.language_model\.": "model."}
        model.config = kwargs["config"]
        model.config.model_type = "qwen3_5_text"
        calls.append("weights")
        return model, info

    def tokenizer(path, **kwargs):
        assert path == snapshot and kwargs == {"local_files_only": True}
        calls.append("tokenizer")
        return object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5TextConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Qwen3_5ForCausalLM=SimpleNamespace(from_pretrained=load),
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
    with pytest.raises(ValueError, match="incomplete"):
        model_load.load_text_model(device="cpu", memory=memory)
    assert model not in dictionary._VERIFIED_MODELS
