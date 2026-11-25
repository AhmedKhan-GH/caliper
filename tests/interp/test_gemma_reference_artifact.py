"""Original-source CPU fixtures for explicit Gemma reference artifacts."""
from copy import deepcopy
import hashlib
import json
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import _gemma_import_identity as imported
from caliper.interp._dictionary import _state_hash, load_dictionary, verify_hf_model
from tools.interp import convert_gemma_scope as conversion
from .cpu_pressure import deterministic_cpu_pressure
from .test_gemma_runtime import gemma_model, local_gemma, memory


PROFILE = "circuit_tracer_82996f14"


def source_arrays(seed=12, dtype=np.float32):
    random = np.random.default_rng(seed)
    return {name: random.uniform(low, high, shape).astype(dtype) for name, shape, low, high in (
        ("W_enc", (16, 20), -3, 3), ("W_dec", (20, 16), -2, 2),
        ("b_enc", (20,), -1, 1), ("b_dec", (16,), -1, 1), ("threshold", (20,), .1, 2))}


def bf16_bits(array):
    # Independent scalar round-to-nearest-even oracle for finite float32 inputs.
    rounded = []
    for value in np.asarray(array).flat:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounded.append(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF)
    return np.array(rounded, dtype=np.uint16).reshape(array.shape)


def assert_source_cast(tensor, array):
    assert tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
    np.testing.assert_array_equal(tensor.view(torch.uint16).numpy(), bf16_bits(array))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_reference_encoder_is_direct_source_cast(dtype):
    arrays = source_arrays(dtype=dtype)
    gain = torch.linspace(.701, 1.713, 16).to(torch.bfloat16)
    result = conversion.convert_arrays(arrays, gain, width=16, latents=20, profile=PROFILE)
    assert set(result) == {*arrays, "W_enc_original"}
    assert_source_cast(result["W_enc_original"], arrays["W_enc"].T)
    for name in ("W_dec", "b_enc", "b_dec", "threshold"):
        assert_source_cast(result[name], arrays[name])
    # A converted encoder cannot recover the original rows after dtype rounding.
    inverted = (result["W_enc"].float() * gain.float()).to(torch.bfloat16)
    assert not torch.equal(inverted, result["W_enc_original"])
    expected = arrays["W_enc"].T.astype(np.float64 if dtype == np.float64 else np.float32)
    expected = expected / gain.float().numpy()[None, :]
    assert_source_cast(result["W_enc"], expected)


@pytest.mark.parametrize("field,value", [("threshold", 1e-50), ("W_dec", 1e100)])
def test_reference_cast_refuses_underflow_and_overflow(field, value):
    arrays = source_arrays(dtype=np.float64)
    arrays[field].flat[0] = value
    with pytest.raises(ValueError, match="positive|overflows"):
        conversion.convert_arrays(arrays, torch.ones(16, dtype=torch.bfloat16),
                                  width=16, latents=20, profile=PROFILE)


@pytest.fixture
def reference_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(conversion, "WIDTH", 16)
    monkeypatch.setattr(conversion, "LATENTS", 20)
    monkeypatch.setattr(conversion, "L0", (76, 65))
    monkeypatch.setattr(conversion, "SOURCE_PATHS", conversion.SOURCE_PATHS[:2])
    monkeypatch.setattr(imported, "_WIDTH", 16)
    monkeypatch.setattr(imported, "_LATENTS", 20)
    monkeypatch.setattr(imported, "_L0", (76, 65))
    model = gemma_model().bfloat16()
    with torch.no_grad():
        for block in model.model.layers:
            block.pre_feedforward_layernorm.weight.copy_(torch.linspace(-.297, .713, 16))
    snapshot = local_gemma(tmp_path, model)
    model.float()
    verify_hf_model(model, snapshot)
    source = tmp_path / "models--google--gemma-scope-2b-pt-transcoders" / "snapshots" / conversion.SOURCE_REVISION
    for index, name in enumerate(conversion.SOURCE_PATHS):
        path = source / name
        path.parent.mkdir(parents=True)
        np.savez(path, **source_arrays(12 + index))
    output = tmp_path / "reference"
    metadata = conversion.convert_gemma_scope(source, output, model=model,
                                               model_snapshot=snapshot, profile=PROFILE)
    return source, output, snapshot, model, metadata


def test_reference_publication_binds_source_gain_and_all_output_tensors(reference_artifact):
    source, folder, snapshot, model, metadata = reference_artifact
    assert metadata == json.loads((folder / "dictionary.json").read_text())
    assert metadata["reference_input"] == {
        "coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original", "dtype": "bfloat16"}
    assert metadata["import_source"]["reference_conversion"]["profile"] == PROFILE
    for index, row in enumerate(metadata["import_source"]["layers"]):
        tensors = load_file(folder / row["file"])
        assert row["original_dtype"] == "float32"
        assert row["source_gain_dtype"] == "bfloat16"
        assert row["source_sha256"] == hashlib.sha256((source / row["source_file"]).read_bytes()).hexdigest()
        assert row["sha256"] == hashlib.sha256((folder / row["file"]).read_bytes()).hexdigest()
        assert row["tensor_hashes"] == {name: _state_hash({name: value}) for name, value in tensors.items()}
        assert_source_cast(tensors["W_enc_original"], source_arrays(12 + index)["W_enc"].T)
        weight = model.model.layers[index].pre_feedforward_layernorm.weight.detach()
        assert row["source_gain_hash"] == _state_hash({"weight": weight.to(torch.bfloat16)})
        assert row["float32_gain_hash"] == _state_hash({"gain": 1 + weight})
        assert row["input_gain_hash"] == _state_hash({"gain": (1 + weight).to(torch.bfloat16)})
    model.bfloat16()
    verify_hf_model(model, snapshot)
    dictionary = load_dictionary(folder, model=model)
    assert all(layer.W_enc_original.dtype == torch.bfloat16 for layer in dictionary.layers)


def test_reference_profile_requires_explicit_pinned_gain_source(tmp_path):
    with pytest.raises(ValueError, match="pinned local Gemma model snapshot"):
        conversion.convert_gemma_scope(tmp_path, tmp_path / "output", model=object(), profile=PROFILE)
    with pytest.raises(ValueError, match="unsupported"):
        conversion.convert_gemma_scope(tmp_path, tmp_path / "output", model=object(), profile="auto")
    with pytest.raises(ValueError, match="rounded bfloat16 gain"):
        conversion.convert_arrays(source_arrays(), torch.ones(16), width=16, latents=20, profile=PROFILE)
