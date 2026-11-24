"""CPU checks for JumpReLU storage, activation, and evaluation evidence."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import load_dictionary
from caliper.interp import _dictionary
from caliper.interp._dictionary import JumpReLUTranscoder, TopKTranscoder, dictionary_fingerprint, weight_names
from caliper.interp._memory import CleanStop
from tools.interp.evaluation_binding import EvaluationBinding
from .fixture import TinyLM
from .test_dictionary import memory_readers, write_fixture


def jump_fixture(folder):
    model, data = write_fixture(folder, TinyLM(layers=2, width=4, heads=2, latents=8, k=2))
    data.update(activation="jumprelu", k=None)
    (folder / "dictionary.json").write_text(json.dumps(data))
    for i in range(data["layers"]):
        path = folder / f"layer_{i:02d}.safetensors"
        tensors = {name: value.clone() for name, value in load_file(path).items()}
        tensors["threshold"] = torch.linspace(0.25, 2, 8, dtype=torch.float64)
        save_file(tensors, path)
    return model, data


def test_activation_boundary_and_gradient():
    weights = torch.eye(3, dtype=torch.float64)
    layer = JumpReLUTranscoder(weights, torch.zeros(3, dtype=torch.float64), weights.clone(),
                              torch.zeros(3, dtype=torch.float64), torch.tensor([1., 2., 3.]))
    values = torch.tensor([[-1., 2., 4.], [1., 2.5, 3.]], dtype=torch.float64, requires_grad=True)
    expected = torch.tensor([[0., 0., 4.], [0., 2.5, 0.]], dtype=torch.float64)
    assert layer.activation == "jumprelu" and layer.k is None
    assert torch.equal(layer.activate(values), expected)
    assert torch.equal(layer.encode(values), expected)
    assert torch.equal(layer(values), expected)
    layer(values).sum().backward()
    assert torch.equal(values.grad, torch.tensor([[0., 0., 1.], [0., 1., 0.]], dtype=torch.float64))


def test_topk_api_keeps_original_numerics():
    weights = torch.eye(4)
    layer = TopKTranscoder(weights, torch.zeros(4), weights.clone(), torch.zeros(4), 2)
    values = torch.tensor([[-3., -1., -2., -4.], [2., 2., 1., -1.]])
    old = values.relu()
    selected, indices = old.topk(2, dim=-1)
    expected = torch.zeros_like(old).scatter(-1, indices, selected)
    assert layer.activation == "topk"
    assert weight_names(layer) == _dictionary._WEIGHTS
    assert torch.equal(layer.activate(values), expected)
    assert torch.equal(layer.encode(values), expected)


def test_round_trip_and_fingerprint_include_threshold(tmp_path):
    model, metadata = jump_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    assert dictionary.metadata == metadata
    first = dictionary.layers[0]
    assert weight_names(metadata) == weight_names(first) == (*_dictionary._WEIGHTS, "threshold")
    assert not first.threshold.requires_grad
    assert torch.equal(first.threshold, torch.linspace(0.25, 2, 8, dtype=torch.float64))
    before = dictionary_fingerprint(tmp_path, metadata)
    path = tmp_path / "layer_00.safetensors"
    tensors = {name: value.clone() for name, value in load_file(path).items()}
    tensors["threshold"][0] += 0.25
    save_file(tensors, path)
    assert dictionary_fingerprint(tmp_path, metadata) != before


@pytest.mark.parametrize("kind", ["missing", "shape", "zero", "negative", "nan", "inf", "integer", "underflow"])
def test_malformed_threshold_refusal(tmp_path, kind):
    model, _ = jump_fixture(tmp_path)
    path = tmp_path / "layer_00.safetensors"
    tensors = {name: value.clone() for name, value in load_file(path).items()}
    if kind == "missing":
        del tensors["threshold"]
    elif kind == "shape":
        tensors["threshold"] = tensors["threshold"].reshape(2, 4)
    elif kind == "integer":
        tensors["threshold"] = tensors["threshold"].to(torch.int64)
    else:
        tensors["threshold"][0] = {"zero": 0, "negative": -1, "nan": float("nan"),
                                   "inf": float("inf"), "underflow": 1e-100}[kind]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="threshold|tensor names"):
        load_dictionary(tmp_path, model=model, dtype=torch.float32)


@pytest.mark.parametrize("k", [0, 2, "null"])
def test_jumprelu_requires_null_k(tmp_path, k):
    model, data = jump_fixture(tmp_path)
    data["k"] = k
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="k=null"):
        load_dictionary(tmp_path, model=model)


def test_threshold_is_in_memory_admission(tmp_path, monkeypatch):
    model, data = jump_fixture(tmp_path)
    budgets = []
    original = _dictionary._layer_bytes

    def record(stored, shapes, device, dtype):
        result = original(stored, shapes, device, dtype)
        assert shapes["threshold"] == (8,) and "threshold" in stored
        budgets.append(result)
        return result

    monkeypatch.setattr(_dictionary, "_layer_bytes", record)
    load_dictionary(tmp_path, model=model)
    assert budgets[0][0] == (2 * 8 * 4 + 8 + 4 + 8) * 8
    monkeypatch.setattr(_dictionary, "_metadata_budget", lambda size: 0)
    # The weights alone fit this limit only if the threshold tensors are omitted.
    limit = (2 * (2 * 8 * 4 + 8 + 4) * 8 + budgets[0][1]) / 1e9
    with pytest.raises(CleanStop, match="allocation"):
        load_dictionary(tmp_path, model=model, memory_limit_gb=limit)


@pytest.mark.parametrize("kind", ["version", "alias", "replacement", "activation", "disk"])
def test_binding_refuses_threshold_changes(tmp_path, kind):
    model, _ = jump_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
    bound = EvaluationBinding(tmp_path, dictionary, identity, model=model)
    bound.verify()
    layer = dictionary.layers[0]
    if kind == "version":
        layer.threshold.detach().add_(1)
    elif kind == "alias":
        version = layer.threshold._version
        layer.threshold.data.add_(1)
        assert layer.threshold._version == version
    elif kind == "replacement":
        layer.threshold = torch.nn.Parameter(layer.threshold.detach().clone(), requires_grad=False)
    elif kind == "activation":
        layer.activation = "topk"
    else:
        path = tmp_path / "layer_00.safetensors"
        tensors = {name: value.clone() for name, value in load_file(path).items()}
        tensors["threshold"][0] += 1
        save_file(tensors, path)
    with pytest.raises(ValueError, match="artifact binding"):
        bound.verify()


def test_binding_compares_threshold_to_checkpoint(tmp_path):
    model, _ = jump_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    dictionary.layers[0].threshold.data.add_(1)
    with pytest.raises(ValueError, match="artifact binding"):
        EvaluationBinding(tmp_path, dictionary, dictionary_fingerprint(tmp_path, dictionary.metadata), model=model)


def test_expanded_budget_is_only_for_pinned_gemma(tmp_path):
    model, _ = jump_fixture(tmp_path)
    with pytest.raises(ValueError, match="pinned Gemma"):
        load_dictionary(tmp_path, model=model, memory_limit_gb=28)
    for limit in (0, 29, float("nan"), True):
        with pytest.raises(ValueError, match="memory limit"):
            load_dictionary(tmp_path, model=model, memory_limit_gb=limit)
