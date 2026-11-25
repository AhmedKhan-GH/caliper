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
from caliper.interp._memory import CleanStop
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
    model = gemma_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(parameter.to(torch.bfloat16))
    for block in model.model.layers:
        block.pre_feedforward_layernorm.weight = torch.nn.Parameter(
            torch.linspace(-.297, .713, 16), requires_grad=False)
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
    reference_model_dtype(model, snapshot)
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
        assert row["source_gain_dtype"] == "float32"
        assert row["source_sha256"] == hashlib.sha256((source / row["source_file"]).read_bytes()).hexdigest()
        assert row["sha256"] == hashlib.sha256((folder / row["file"]).read_bytes()).hexdigest()
        assert row["tensor_hashes"] == {name: _state_hash({name: value}) for name, value in tensors.items()}
        assert_source_cast(tensors["W_enc_original"], source_arrays(12 + index)["W_enc"].T)
        weight = model.model.layers[index].pre_feedforward_layernorm.weight.detach()
        assert row["source_gain_hash"] == _state_hash({"weight": weight})
        assert row["float32_gain_hash"] == _state_hash({"gain": 1 + weight})
        assert row["input_gain_hash"] == _state_hash({"gain": (1 + weight).to(torch.bfloat16)})
    reference_model_dtype(model, snapshot)
    dictionary = load_dictionary(folder, model=model)
    assert all(layer.W_enc_original.dtype == torch.bfloat16 for layer in dictionary.layers)


def test_reference_profile_requires_explicit_pinned_gain_source(tmp_path):
    with pytest.raises(ValueError, match="pinned local Gemma model snapshot"):
        conversion.convert_gemma_scope(tmp_path, tmp_path / "output", model=object(), profile=PROFILE)
    with pytest.raises(ValueError, match="unsupported"):
        conversion.convert_gemma_scope(tmp_path, tmp_path / "output", model=object(), profile="auto")
    with pytest.raises(ValueError, match="rounded bfloat16 gain"):
        conversion.convert_arrays(source_arrays(), torch.ones(16), width=16, latents=20, profile=PROFILE)


def reference_model_dtype(model, snapshot):
    # Match source norm precision. This fixture does not install a model adapter.
    norms = [block.pre_feedforward_layernorm.weight.detach().clone() for block in model.model.layers]
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    for block, norm in zip(model.model.layers, norms):
        block.pre_feedforward_layernorm.weight = torch.nn.Parameter(norm, requires_grad=False)
    verify_hf_model(model, snapshot)


def qualify(artifact, **kwargs):
    source, folder, snapshot, model, metadata = artifact
    return imported.verified_import_binding(folder, metadata, model=model, memory=memory(),
                                            source=source, model_snapshot=snapshot, profile=PROFILE, **kwargs)


def test_reference_qualification_uses_original_sources(reference_artifact):
    _, _, snapshot, model, metadata = reference_artifact
    result = qualify(reference_artifact)
    assert result["version"] == 2 and result["profile"] == PROFILE
    assert result["reference_input"] == metadata["reference_input"]
    assert len(result["layers"]) == 2
    for result_row, row in zip(result["layers"], metadata["import_source"]["layers"]):
        assert all(row[key] == value for key, value in result_row.items())


def test_reference_profile_refuses_float32_runtime(reference_artifact):
    _, _, snapshot, model, _ = reference_artifact
    model.float()
    verify_hf_model(model, snapshot)
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


@pytest.mark.parametrize("fault", [
    "missing_reference", "coordinate", "encoder", "dtype", "operations", "operation_version", "boolean_version",
    "model_revision", "source_revision", "selection", "reference_revision", "source_verification",
    "original_dtype", "source_gain_dtype", "source_gain_hash", "float32_gain_hash", "input_gain_hash",
    "tensor_hashes", "sha256", "source_sha256", "source_file",
])
def test_reference_metadata_mismatch_is_refused(reference_artifact, fault):
    metadata = reference_artifact[-1]
    source = metadata["import_source"]
    row = source["layers"][0]
    if fault == "missing_reference":
        del metadata["reference_input"]
    elif fault in ("coordinate", "encoder", "dtype"):
        metadata["reference_input"][fault] = "incorrect"
    elif fault == "operations":
        source["reference_conversion"]["tensors"]["W_enc_original"] = "W_enc * gain"
    elif fault in ("operation_version", "boolean_version"):
        source["reference_conversion"]["version"] = 2 if fault == "operation_version" else True
    elif fault == "model_revision":
        metadata["revision"] = "b" * 40
    elif fault in ("source_revision", "selection", "reference_revision"):
        source[{"source_revision": "source", "selection": "selection", "reference_revision": "reference"}[fault]]["revision"] = "b" * 40
    elif fault == "source_verification":
        source[fault] = "unchecked"
    else:
        row[fault] = {} if fault == "tensor_hashes" else "incorrect"
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


def test_legacy_and_reference_profiles_do_not_alias(reference_artifact):
    source, folder, snapshot, model, metadata = reference_artifact
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, metadata, model=model)
    legacy = conversion._metadata(deepcopy(metadata["import_source"]["layers"]))
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, legacy, model=model, profile=PROFILE,
                                        source=source, model_snapshot=snapshot)
    for missing in ("source", "model_snapshot"):
        kwargs = {"source": source, "model_snapshot": snapshot}
        del kwargs[missing]
        with pytest.raises(ValueError, match="import binding"):
            imported.verified_import_binding(folder, metadata, model=model, profile=PROFILE, **kwargs)


@pytest.mark.parametrize("fault", ["byte", "inverse", "dtype", "compatibility", "decoder", "threshold"])
def test_reforged_output_hashes_cannot_replace_source_bytes(reference_artifact, fault):
    _, folder, _, model, metadata = reference_artifact
    row = metadata["import_source"]["layers"][0]
    path = folder / row["file"]
    tensors = load_file(path)
    if fault == "inverse":
        gain = (1 + model.model.layers[0].pre_feedforward_layernorm.weight.detach()).to(torch.bfloat16)
        tensors["W_enc_original"] = (tensors["W_enc"].float() * gain.float()).to(torch.bfloat16)
    elif fault == "dtype":
        tensors["W_enc_original"] = tensors["W_enc_original"].float()
    else:
        key = {"byte": "W_enc_original", "compatibility": "W_enc", "decoder": "W_dec", "threshold": "threshold"}[fault]
        tensors[key].view(-1)[0] += .25
    save_file(tensors, path)
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["tensor_hashes"] = {name: _state_hash({name: value}) for name, value in tensors.items()}
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


@pytest.mark.parametrize("reforge", [False, True])
def test_source_npz_mutation_is_refused(reference_artifact, reforge):
    source, _, _, _, metadata = reference_artifact
    row = metadata["import_source"]["layers"][0]
    arrays = source_arrays()
    arrays["W_enc"][0, 0] += .5
    path = source / row["source_file"]
    np.savez(path, **arrays)
    if reforge:
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


def test_gain_alias_mutation_cannot_rebaseline_reference_metadata(reference_artifact):
    model, metadata = reference_artifact[-2:]
    weight = model.model.layers[0].pre_feedforward_layernorm.weight
    version = weight._version
    weight.data.add_(.125)
    assert weight._version == version
    row = metadata["import_source"]["layers"][0]
    row["source_gain_hash"] = _state_hash({"weight": weight.detach()})
    row["float32_gain_hash"] = _state_hash({"gain": 1 + weight.detach()})
    row["input_gain_hash"] = _state_hash({"gain": (1 + weight.detach()).to(torch.bfloat16)})
    with pytest.raises(ValueError, match="identity mismatch"):
        qualify(reference_artifact)


def test_early_norm_rounding_refuses_a_different_effective_gain(reference_artifact):
    _, _, snapshot, model, _ = reference_artifact
    weight = model.model.layers[0].pre_feedforward_layernorm.weight.detach()
    expected = (1 + weight).to(torch.bfloat16)
    rounded_first = (1 + weight.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert not torch.equal(expected, rounded_first)
    model.bfloat16()
    verify_hf_model(model, snapshot)
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


def test_reference_ordinary_float32_regression(reference_artifact, tmp_path):
    source, _, snapshot, model, _ = reference_artifact
    model.float()
    verify_hf_model(model, snapshot)
    output = tmp_path / "ordinary"
    metadata = conversion.convert_gemma_scope(source, output, model=model)
    assert "reference_input" not in metadata
    assert "reference_conversion" not in metadata["import_source"]
    result = imported.verified_import_binding(output, metadata, model=model)
    assert result["version"] == 1
    for index, row in enumerate(metadata["import_source"]["layers"]):
        tensors = load_file(output / row["file"])
        gain = 1 + model.model.layers[index].pre_feedforward_layernorm.weight.detach().numpy()
        expected = source_arrays(12 + index)["W_enc"].T / gain[None, :]
        np.testing.assert_array_equal(tensors["W_enc"].numpy(), expected)
        assert all(value.dtype == torch.float32 for value in tensors.values())
        assert "W_enc_original" not in tensors


def test_reference_layer_admission_precedes_source_arrays(reference_artifact, monkeypatch):
    source, folder, _, _, metadata = reference_artifact
    row = metadata["import_source"]["layers"][0]
    guard = memory()
    guard.limit = 4 * 1024 * 1024
    monkeypatch.setattr(imported, "_source_arrays", lambda *a: pytest.fail("source arrays allocated"))
    with pytest.raises(CleanStop, match="allocation"):
        imported._reference_layer(folder / row["file"], source / row["source_file"], row,
                                  torch.ones(16, dtype=torch.bfloat16), guard)


def test_reference_conversion_admission_includes_original_encoder(tmp_path, monkeypatch):
    from .test_convert_gemma_scope import Guard
    path = tmp_path / "params.npz"
    np.savez(path, **source_arrays())
    guard = Guard()
    guard.limit = 4 * 1024 * 1024
    monkeypatch.setattr(conversion, "_arrays", lambda *a: pytest.fail("source arrays allocated"))
    with pytest.raises(CleanStop, match="allocation"):
        conversion.load_scope_layer(path, torch.ones(16, dtype=torch.bfloat16),
                                    width=16, latents=20, memory=guard, profile=PROFILE)


def test_reference_qualification_leaves_prospective_host_reserve(reference_artifact, monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8_000_000_000))
    monkeypatch.setattr(imported, "model_identity", lambda *a, **k: pytest.fail("model hashing began"))
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        qualify(reference_artifact)


def test_source_checkpoint_gain_mutation_is_refused(reference_artifact):
    _, _, snapshot, _, metadata = reference_artifact
    path = next(snapshot.glob("*.safetensors"))
    tensors = {name: value.clone() for name, value in load_file(path).items()}
    key = "model.layers.0.pre_feedforward_layernorm.weight"
    tensors[key][0] += .125
    save_file(tensors, path)
    row = metadata["import_source"]["layers"][0]
    row["source_gain_hash"] = _state_hash({"weight": tensors[key]})
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


def test_reference_file_replacement_during_verification_is_refused(reference_artifact, monkeypatch):
    _, folder, _, _, _ = reference_artifact
    read = imported._source_arrays

    def replace_after_digest(path):
        arrays = read(path)
        destination = folder / "layer_00.safetensors"
        tensors = {name: value.clone() for name, value in load_file(destination).items()}
        save_file(tensors, destination)
        return arrays

    monkeypatch.setattr(imported, "_source_arrays", replace_after_digest)
    with pytest.raises(ValueError, match="import binding"):
        qualify(reference_artifact)


def test_gain_header_admission_precedes_checkpoint_reads(reference_artifact, monkeypatch):
    from pathlib import Path
    _, _, snapshot, model, _ = reference_artifact
    guard = memory()
    guard.limit = 10_000_000
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("checkpoint read began"))
    with pytest.raises(CleanStop, match="allocation"):
        imported._reference_gain(snapshot, 0, model.model.layers[0].pre_feedforward_layernorm.weight, guard)


def test_reference_cli_selects_explicit_profile_without_dtype_inference(tmp_path, monkeypatch):
    source = tmp_path / "models--google--gemma-scope-2b-pt-transcoders" / "snapshots" / conversion.SOURCE_REVISION
    snapshot = tmp_path / "models--google--gemma-2-2b" / "snapshots" / conversion.MODEL_REVISION
    from caliper.interp import _admission
    model = object()
    calls = []
    monkeypatch.setattr(conversion, "MemoryGuard", lambda *a: memory())
    monkeypatch.setattr(_admission, "_load_gemma_model", lambda *a, **k: model)
    monkeypatch.setattr(conversion, "convert_gemma_scope", lambda *a, **k: calls.append((a, k)))
    conversion.main(["--source", str(source), "--output", str(tmp_path / "reference"),
                     "--model-snapshot", str(snapshot), "--profile", PROFILE])
    assert len(calls) == 1
    assert calls[0][1]["model"] is model
    assert calls[0][1]["profile"] == PROFILE
    assert calls[0][1]["model_snapshot"] == snapshot
