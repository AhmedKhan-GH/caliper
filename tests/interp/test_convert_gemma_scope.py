"""Synthetic CPU checks for the pinned Gemma Scope converter."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from caliper.interp._memory import CleanStop
from tools.interp import convert_gemma_scope as conversion


class Guard:
    peak, limit = 0, 10**9
    device = torch.device("cpu")

    def check(self, stage, **kwargs):
        pass


def arrays(dtype=np.float64):
    return {"W_enc": np.array([[1, 2, -1], [3, -2, 2]], dtype=dtype),
            "b_enc": np.array([0.5, -1, 1], dtype=dtype),
            "W_dec": np.array([[1, 2], [-3, 1], [2, -2]], dtype=dtype),
            "b_dec": np.array([0.25, -0.5], dtype=dtype),
            "threshold": np.array([0.5, 1, 2], dtype=dtype)}


@pytest.mark.parametrize("gain", [[2., -4.], [-0.5, 3.], [1., 1.]])
def test_conversion_matches_independent_scalar_reference(gain):
    source = arrays()
    converted = conversion.convert_arrays(source, torch.tensor(gain, dtype=torch.float64), width=2, latents=3)
    from caliper.interp._dictionary import JumpReLUTranscoder
    layer = JumpReLUTranscoder(**converted)
    for pure in ([0., 0.], [1., -2.], [-3., 0.25]):
        expected_features = []
        for feature in range(3):
            pre = float(source["b_enc"][feature])
            for column in range(2):
                pre += pure[column] * float(source["W_enc"][column, feature])
            expected_features.append(pre if pre > source["threshold"][feature] else 0.)
        expected_output = []
        for column in range(2):
            total = float(source["b_dec"][column])
            for feature in range(3):
                total += expected_features[feature] * float(source["W_dec"][feature, column])
            expected_output.append(total)
        actual_input = torch.tensor([[pure[i] * gain[i] for i in range(2)]], dtype=torch.float64)
        torch.testing.assert_close(layer.encode(actual_input)[0], torch.tensor(expected_features, dtype=torch.float64), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(layer(actual_input)[0], torch.tensor(expected_output, dtype=torch.float64), rtol=1e-14, atol=1e-14)
    for name in ("b_enc", "W_dec", "b_dec", "threshold"):
        assert np.array_equal(converted[name].numpy(), source[name])
    assert all(value.is_contiguous() for value in converted.values())


def test_random_init_gemma_gain_uses_one_plus_weight():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    config = Gemma2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=16)
    model = Gemma2ForCausalLM(config).float().eval()
    with torch.no_grad():
        model.model.layers[0].pre_feedforward_layernorm.weight.copy_(torch.tensor([1., -3., 0., 2., 3., -2., 0.5, -0.5]))
    gain = 1 + model.model.layers[0].pre_feedforward_layernorm.weight.detach()
    source = {"W_enc": np.arange(24, dtype=np.float32).reshape(8, 3),
              "b_enc": np.ones(3, dtype=np.float32), "W_dec": np.ones((3, 8), dtype=np.float32),
              "b_dec": np.zeros(8, dtype=np.float32), "threshold": np.ones(3, dtype=np.float32)}
    converted = conversion.convert_arrays(source, gain, width=8, latents=3)
    for i in range(8):
        for j in range(3):
            assert converted["W_enc"][j, i].item() == pytest.approx(float(source["W_enc"][i, j]) / gain[i].item())
    # Random configurations cannot enter the real conversion path.
    model.config._name_or_path = conversion.MODEL_ID
    model.config._commit_hash = conversion.MODEL_REVISION
    with pytest.raises(ValueError, match="verified pinned"):
        conversion._model_admission(model, 12)


@pytest.mark.parametrize("gain", [[0., 1.], [float("nan"), 1.], [float("inf"), 1.], [1.]])
def test_bad_gain_refusal(gain):
    with pytest.raises(ValueError, match="gain"):
        conversion.convert_arrays(arrays(), torch.tensor(gain), width=2, latents=3)


@pytest.mark.parametrize("compressed", [False, True])
def test_npz_conversion_hash_and_no_pickle(tmp_path, monkeypatch, compressed):
    path = tmp_path / "params.npz"
    (np.savez_compressed if compressed else np.savez)(path, **arrays())
    original = np.load
    calls = []

    def safe(*args, **kwargs):
        calls.append(kwargs["allow_pickle"])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", safe)
    converted, digest = conversion.load_scope_layer(path, torch.tensor([2., -4.]), width=2, latents=3, memory=Guard())
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [False] * 5
    assert torch.equal(converted["threshold"], torch.from_numpy(arrays()["threshold"]))


@pytest.mark.parametrize("kind", ["object", "integer", "shape", "extra", "missing", "zero", "negative", "nan", "mixed", "overflow"])
def test_malformed_arrays_fail_before_output(tmp_path, kind):
    source = arrays(np.float32)
    gain = torch.tensor([2., -4.])
    if kind == "object":
        source["b_enc"] = np.array([object(), object(), object()], dtype=object)
    elif kind == "integer":
        source["b_enc"] = source["b_enc"].astype(np.int64)
    elif kind == "shape":
        source["W_enc"] = source["W_enc"].T
    elif kind == "extra":
        source["extra"] = np.ones(1)
    elif kind == "missing":
        del source["threshold"]
    elif kind == "mixed":
        source["threshold"] = source["threshold"].astype(np.float64)
    elif kind == "overflow":
        gain[0] = 1e-40
    else:
        source["threshold"][0] = {"zero": 0, "negative": -1, "nan": float("nan")}[kind]
    path = tmp_path / "params.npz"
    np.savez(path, **source)
    with pytest.raises(ValueError):
        conversion.load_scope_layer(path, gain, width=2, latents=3, memory=Guard())


def test_duplicate_zip_members_are_refused(tmp_path):
    path = tmp_path / "params.npz"
    np.savez(path, **arrays())
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning):
            archive.writestr("threshold.npy", b"duplicate")
    with pytest.raises(ValueError, match="bounds|members"):
        conversion.load_scope_layer(path, torch.ones(2), width=2, latents=3, memory=Guard())


def test_resource_refusal_precedes_array_load(tmp_path, monkeypatch):
    path = tmp_path / "params.npz"
    np.savez(path, **arrays())
    guard = Guard()
    guard.limit = 1_000_000
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array allocation ran"))
    with pytest.raises(CleanStop, match="allocation"):
        conversion.load_scope_layer(path, torch.ones(2), width=2, latents=3, memory=guard)


def test_header_length_is_bounded_before_read():
    class Header(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 10_000
            return super().read(size)

    source = Header(b"\x93NUMPY\x02\x00" + (2**32 - 1).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="header exceeds"):
        conversion._npy_header(source)


def test_changed_header_is_refused_before_numpy_allocation(tmp_path, monkeypatch):
    path = tmp_path / "params.npz"
    np.savez(path, **arrays())
    original = conversion._hash_file
    changed = False

    def mutate(source, memory):
        nonlocal changed
        result = original(source, memory)
        if not changed:
            changed = True
            other = arrays()
            other["threshold"] = np.ones(5, dtype=np.float64)
            np.savez(path, **other)
        return result

    monkeypatch.setattr(conversion, "_hash_file", mutate)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array allocation ran"))
    # Change the first member too, so the loader refuses before any NumPy allocation.
    original_arrays = conversion._arrays

    def changed_first(source, headers):
        other = arrays()
        other["W_enc"] = np.ones((3, 3), dtype=np.float64)
        np.savez(path, **other)
        return original_arrays(source, headers)

    monkeypatch.setattr(conversion, "_arrays", changed_first)
    with pytest.raises(ValueError, match="header changed|grew"):
        conversion.load_scope_layer(path, torch.ones(2), width=2, latents=3, memory=Guard())


def test_source_pin_refusal_precedes_model_access(tmp_path):
    source = tmp_path / "models--google--gemma-scope-2b-pt-transcoders" / "snapshots" / ("0" * 40)
    with pytest.raises(ValueError, match="revision differs"):
        conversion.convert_gemma_scope(source, tmp_path / "output", model=object())
    assert not (tmp_path / "output").exists()


def test_pins_and_selection_are_exact():
    assert conversion.MODEL_REVISION == "c5ebcd40d208330abc697524c919956e692655cf"
    assert conversion.SOURCE_REVISION == "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"
    assert conversion.SELECTION_REVISION == "9250a2d4860ce5ed5c96c14d5882b7d8162809a3"
    expected = [76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41]
    assert list(conversion.L0) == expected
    for index, l0 in enumerate(expected):
        assert conversion.SOURCE_PATHS[index] == f"layer_{index}/width_16k/average_l0_{l0}/params.npz"
    metadata = conversion._metadata([])
    assert metadata["training"]["tokens"] == 0
    assert metadata["training"]["publisher_tokens"] is None
    assert metadata["training"]["publisher_training_disjointness"] == "unknown"
    assert metadata["training"]["partitions"] == {}
    assert "source_id" not in metadata["training"]
    assert metadata["import_source"]["license"] == "CC-BY-4.0"
    assert metadata["import_source"]["output_conversion"] == "none"


def test_evaluation_source_is_explicit_pinned_and_separate():
    from tools.interp.text_stream import DATASET_ID, DATASET_CONFIG, DATASET_REVISION, partition_metadata
    source = {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
              "dataset_revision": DATASET_REVISION, "split": "train",
              "source_id": "local-parquet:verified-source", "partitions": partition_metadata()}
    metadata = conversion._metadata([], source)
    assert metadata["evaluation_source"] == source
    assert metadata["training"]["dataset_id"] == "unknown"
    assert metadata["training"]["local_training"] is False
    source["partitions"]["quality"][0] = 0
    assert metadata["evaluation_source"]["partitions"]["quality"][0] == 9400
    with pytest.raises(ValueError, match="pinned evaluation source"):
        conversion._metadata([], source)


@pytest.mark.parametrize("fail", [True, False])
def test_atomic_publication_and_layer_release(tmp_path, monkeypatch, fail):
    # This tests publication orchestration with a small fake layer loader.
    # Only convert_arrays and load_scope_layer accept synthetic dimensions.
    source = tmp_path / "models--google--gemma-scope-2b-pt-transcoders" / "snapshots" / conversion.SOURCE_REVISION
    output = tmp_path / "converted"
    model = torch.nn.Linear(1, 1)
    blocks = [SimpleNamespace(pre_feedforward_layernorm=SimpleNamespace(weight=torch.zeros(conversion.WIDTH)))
              for _ in conversion.L0]
    admissions = []
    fsync = conversion.os.fsync

    def writable_fsync(descriptor):
        # A zero-byte write verifies Windows-compatible descriptor access.
        conversion.os.write(descriptor, b"")
        fsync(descriptor)

    monkeypatch.setattr(conversion.os, "fsync", writable_fsync)

    def admitted(model, limit):
        admissions.append(limit)
        return blocks

    monkeypatch.setattr(conversion, "_model_admission", admitted)
    monkeypatch.setattr(conversion, "MemoryGuard", lambda *a: Guard())
    calls = []
    references = []

    def load(path, gain, **kwargs):
        import weakref
        assert not output.exists()
        assert all(ref() is None for ref in references)
        calls.append(Path(path).relative_to(source).as_posix())
        if fail and len(calls) == 2:
            raise CleanStop("synthetic layer refusal")
        tensors = conversion.convert_arrays(arrays(), torch.tensor([2., -4.]), width=2, latents=3)
        references.extend(weakref.ref(value) for value in tensors.values())
        return tensors, "0" * 64

    monkeypatch.setattr(conversion, "load_scope_layer", load)
    if fail:
        with pytest.raises(CleanStop, match="synthetic"):
            conversion.convert_gemma_scope(source, output, model=model)
        assert not output.exists()
    else:
        result = conversion.convert_gemma_scope(source, output, model=model)
        assert admissions == [28, 28]
        assert calls == list(conversion.SOURCE_PATHS)
        assert json.loads((output / "dictionary.json").read_text()) == result
        for row in result["import_source"]["layers"]:
            assert row["sha256"] == hashlib.sha256((output / row["file"]).read_bytes()).hexdigest()
            assert row["input_gain_hash"].startswith("sha256:")
        before = (output / "dictionary.json").read_bytes()
        with pytest.raises(FileExistsError):
            conversion.convert_gemma_scope(source, output, model=model)
        assert (output / "dictionary.json").read_bytes() == before
    assert not output.with_name("." + output.name + ".conversion-lock").exists()


@pytest.mark.parametrize("requested", [None, 12, 28, 28.1])
def test_cli_budget_admits_exact_gemma_load_reserve_and_preserves_lower_cap(tmp_path, monkeypatch, requested):
    from caliper.interp import _admission
    from .test_memory_phases import process_guard
    source = tmp_path / "models--google--gemma-scope-2b-pt-transcoders" / "snapshots" / conversion.SOURCE_REVISION
    snapshot = tmp_path / "models--google--gemma-2-2b" / "snapshots" / conversion.MODEL_REVISION
    arguments = ["--source", str(source), "--output", str(tmp_path / "output"), "--model-snapshot", str(snapshot)]
    if requested is not None:
        arguments += ["--memory-limit-gb", str(requested)]
    calls, converted = [], []
    model = object()
    monkeypatch.setattr(conversion, "MemoryGuard", lambda limit, device: process_guard(
        int(limit * 1e9), {"current": 512_000_000, "peak": 1_000_000_000}))

    def load(path, *, memory, device):
        assert path == snapshot and device == "cpu"
        calls.append(memory.limit)
        # Scalar accounting only. No gated model or large array is loaded.
        _admission._admit_load(memory, 18_557_347_328, "Gemma CPU reserve", staged=True)
        return model

    def convert(source_path, output, **kwargs):
        assert source_path == source and kwargs["model"] is model
        converted.append(kwargs["memory_limit_gb"])

    monkeypatch.setattr(_admission, "_load_gemma_model", load)
    monkeypatch.setattr(conversion, "convert_gemma_scope", convert)
    if requested in (12, 28.1):
        with pytest.raises(CleanStop if requested == 12 else ValueError, match="memory limit"):
            conversion.main(arguments)
        assert not converted
        assert calls == ([12_000_000_000] if requested == 12 else [])
    else:
        conversion.main(arguments)
        assert calls == [28_000_000_000] and converted == [28]
