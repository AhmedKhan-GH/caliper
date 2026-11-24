"""Dictionary identity, safe storage, and exact fixture reconstruction."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import load_dictionary
from caliper.interp._dictionary import model_identity, weights_hash
from .fixture import TinyLM


def fixture_manifest(model):
    count = len(model.layers)
    return {
        "format_version": 1, "model_id": "tiny-lm", "model_hash": weights_hash(model),
        "revision": "fixture-v1-seed-7", "layers": count, "width": model.config.hidden_size,
        "latents": model.config.latents, "activation": "topk", "k": model.config.k,
        "read_points": [f"layers.{i}.mlp_in" for i in range(count)],
        "write_points": [f"layers.{i}.mlp_out" for i in range(count)],
        "training": {"dataset_id": "fixture", "dataset_config": "random-inputs",
                     "dataset_revision": "seed-17", "split": "train", "partitions": {},
                     "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
        "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                    "evaluated_tokens": 0},
    }


def write_fixture(folder, model=None):
    model = model if model is not None else TinyLM()
    folder.mkdir(exist_ok=True, parents=True)
    data = fixture_manifest(model)
    (folder / "dictionary.json").write_text(json.dumps(data))
    for i, block in enumerate(model.layers):
        save_file({name: p.detach().clone() for name, p in block.mlp.named_parameters()},
                  folder / f"layer_{i:02d}.safetensors")
    return model, data


def test_exact_fixture_and_input_gradients(tmp_path):
    model, data = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    assert dictionary.metadata == data
    assert len(dictionary.layers) == 4
    for i, block in enumerate(model.layers):
        x = torch.randn(2, 7, 64, dtype=torch.float64, requires_grad=True)
        a = dictionary.encode(i, x)
        assert torch.equal(a, block.mlp.encode(x))
        assert torch.equal(dictionary.decode(i, a), block.mlp(x))
        assert dictionary.error(i, x, block.mlp(x)).count_nonzero() == 0
        assert (a.count_nonzero(-1) <= 8).all()
        assert dictionary.layers[i].W_dec.device == x.device
        assert not dictionary.layers[i].W_dec.requires_grad
        dictionary.decode(i, a).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_identity_hash_covers_all_weights(tmp_path):
    model, _ = write_fixture(tmp_path)
    same = TinyLM()
    assert weights_hash(model) == weights_hash(same)
    with torch.no_grad():
        same.lm_head.weight[-1, -1] += 1
    assert weights_hash(model) != weights_hash(same)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_dictionary(tmp_path, model=same)


def test_expected_hash_and_watch(tmp_path, monkeypatch):
    from caliper import _programs
    model, data = write_fixture(tmp_path)
    monkeypatch.setitem(_programs._STATE, "model", None)
    with pytest.raises(ValueError, match="supply a model"):
        load_dictionary(tmp_path)
    assert load_dictionary(tmp_path, model_hash=data["model_hash"]).metadata == data
    with pytest.raises(ValueError, match="expected model_hash differ"):
        load_dictionary(tmp_path, model=model, model_hash="sha256:" + "0" * 64)
    monkeypatch.setitem(_programs._STATE, "model", model)
    assert load_dictionary(tmp_path).metadata == data


@pytest.mark.parametrize("field,value", [
    ("activation", "relu"), ("k", 257), ("k", 0), ("latents", 128),
    ("width", 32), ("layers", 3), ("revision", ""), ("training", {}),
    ("quality", {}), ("format_version", 2),
    ("read_points", [f"wrong.{i}.mlp_in" for i in range(4)]),
])
def test_reject_manifest_mismatch(tmp_path, field, value):
    model, data = write_fixture(tmp_path)
    data[field] = value
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model)


@pytest.mark.parametrize("damage", ["nan", "inf", "shape", "extra", "integer", "overflow"])
def test_reject_unsafe_layer_values(tmp_path, damage):
    model, _ = write_fixture(tmp_path)
    tensors = {n: p.detach().clone() for n, p in model.layers[0].mlp.named_parameters()}
    if damage in ("nan", "inf"):
        tensors["W_enc"][0, 0] = float(damage)
    elif damage == "shape":
        tensors["W_dec"] = tensors["W_dec"].T.contiguous()
    elif damage == "extra":
        tensors["pickle"] = torch.ones(1)
    elif damage == "integer":
        tensors["b_enc"] = tensors["b_enc"].to(torch.int64)
    else:
        tensors["W_enc"][0, 0] = 1e30
    save_file(tensors, tmp_path / "layer_00.safetensors")
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model, dtype=torch.float16 if damage == "overflow" else None)


def test_nonfinite_metadata(tmp_path):
    model, data = write_fixture(tmp_path)
    data["quality"]["fvu"] = float("nan")
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nonfinite"):
        load_dictionary(tmp_path, model=model)


@pytest.mark.parametrize("section,field,value", [
    ("training", "elapsed_seconds", "1e309"),
    ("training", "peak_memory_bytes", "1e309"),
    ("quality", "fvu", "1e309"), ("quality", "dead_fraction", "1e309"),
    ("quality", "cross_entropy_delta", "-1e309"),
    ("training", "elapsed_seconds", "-1"), ("training", "peak_memory_bytes", "-1"),
    ("quality", "fvu", "-0.1"), ("quality", "dead_fraction", "-0.1"),
    ("quality", "dead_fraction", "1.1"),
])
def test_metadata_numeric_ranges(tmp_path, section, field, value):
    model, data = write_fixture(tmp_path)
    data[section][field] = "REPLACE_NUMBER"
    text = json.dumps(data).replace('"REPLACE_NUMBER"', value)
    (tmp_path / "dictionary.json").write_text(text)
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model)


def test_hf_identity_requires_resolved_checkpoint():
    revision = "a" * 40
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text",
                            _name_or_path="Qwen/Qwen3.5-0.8B", _commit_hash=revision))
    assert model_identity(model) == f"hf:Qwen/Qwen3.5-0.8B@{revision}"
    model.config._commit_hash = "main"
    with pytest.raises(ValueError, match="exact HF"):
        model_identity(model)
    model.config._name_or_path = f"cache/models--Qwen--Qwen3.5-0.8B/snapshots/{revision}"
    model.config._commit_hash = None
    assert model_identity(model).endswith("@" + revision)
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="conflicts"):
        model_identity(model)


def test_partial_export_refuses_load_and_retry_resets_quality(tmp_path, monkeypatch):
    from copy import deepcopy
    from safetensors.torch import load_file
    from tools.interp import checkpoint_state

    model, data = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    data["quality"].update(fvu=0.01, dead_fraction=0, cross_entropy_delta=0,
                           evaluated_tokens=200_000, status="passed")
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    assert load_dictionary(tmp_path, model=model).metadata["quality"]["status"] == "passed"
    trainer = SimpleNamespace(metadata=deepcopy(data), device="cpu",
                              layers=deepcopy([block.mlp for block in model.layers]))
    with torch.no_grad():
        for layer in trainer.layers:
            layer.W_enc.add_(1)
    progress = {"tokens": 32, "elapsed_seconds": 1}
    save = checkpoint_state._save_tensors
    calls = []

    def fail_second(values, path):
        assert not (tmp_path / "dictionary.json").exists()
        calls.append(path)
        if len(calls) == 2:
            raise OSError("fixture export failure")
        save(values, path)

    monkeypatch.setattr(checkpoint_state, "_save_tensors", fail_second)
    with pytest.raises(OSError, match="export failure"):
        checkpoint_state.export_dictionary(tmp_path, trainer, progress)
    assert len(calls) == 2
    assert torch.equal(load_file(tmp_path / "layer_00.safetensors")["W_enc"], trainer.layers[0].W_enc)
    assert torch.equal(load_file(tmp_path / "layer_01.safetensors")["W_enc"], model.layers[1].mlp.W_enc)
    with pytest.raises(FileNotFoundError):
        load_dictionary(tmp_path, model=model)

    monkeypatch.setattr(checkpoint_state, "_save_tensors", save)
    checkpoint_state.export_dictionary(tmp_path, trainer, progress)
    dictionary = load_dictionary(tmp_path, model=model)
    assert dictionary.metadata["training"]["tokens"] == 32
    assert dictionary.metadata["quality"] == {
        "fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
        "evaluated_tokens": 0, "status": "not_evaluated",
    }
    for actual, expected in zip(dictionary.layers, trainer.layers):
        for name, value in expected.named_parameters():
            assert torch.equal(getattr(actual, name), value)
