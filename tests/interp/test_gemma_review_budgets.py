"""Shared admission and retained verification limits with small CPU resources."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _hosted, _index, _validate, load_dictionary
from caliper.interp._memory import CleanStop
from tools.interp import model_load
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_gemma_runtime import gemma_model, local_gemma
from .test_memory_phases import process_guard
from .test_trace import qwen_for_trace, random_dictionary


def test_public_dictionary_load_preserves_supplied_history(tmp_path):
    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=8, k=2))
    readings = {"current": 1_000, "peak": 2_000}
    guard = process_guard(100_000_000, readings)
    dictionary = load_dictionary(tmp_path, model=model, memory=guard)
    assert len(dictionary.layers) == 2 and guard.peak == 2_000
    readings["peak"] = guard.limit + 1
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model=model, memory=guard)
    readings["peak"] = 0
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model=model, memory=guard)


@pytest.mark.parametrize("explicit", [False, True])
def test_validation_refuses_wrong_device_before_prompt(monkeypatch, explicit):
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    dictionary = random_dictionary(model, latents=8, k=2)
    guard = process_guard(12_000_000_000, {"current": 0, "peak": 0})
    guard.device = torch.device("mps")
    monkeypatch.setattr(_validate, "prompt_inputs", lambda *a: pytest.fail("prompt preceded device admission"))
    with pytest.raises(ValueError, match="guard for its model device"):
        _validate.validate_trace(model, TinyTokenizer(), dictionary, SimpleNamespace(), None,
                                 prompt="A", guard=guard, memory_limit_gb=12 if explicit else None)


@pytest.mark.parametrize("family", ["tiny", "qwen"])
def test_validation_caps_implicit_family_budget_and_tracks_dynamic_lower_limit(family):
    model = (TinyLM(layers=2, width=8, heads=2, latents=8, k=2) if family == "tiny"
             else qwen_for_trace(layers=4))
    dictionary = random_dictionary(model, latents=8, k=2)
    readings = {"current": 1_000, "peak": 2_000}
    supplied = process_guard(28_000_000_000, readings)
    guard = _validate._validation_guard(model, dictionary, SimpleNamespace(), supplied)
    assert guard.limit == 12_000_000_000
    guard.check("admitted")
    supplied.limit = 1_999
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("lowered")


@pytest.mark.parametrize("route", ["build", "index", "hash"])
def test_unverified_gemma_dictionary_cannot_admit_wider_index_budget(tmp_path, route):
    model = gemma_model()
    dictionary = random_dictionary(model, latents=8, k=2)
    guard = process_guard(28_000_000_000, {"current": 1_000, "peak": 2_000})
    options = dict(model=model, tokenizer=TinyTokenizer(), dictionary=dictionary, memory=guard)
    with pytest.raises(ValueError, match="requires the pinned Gemma model"):
        if route == "build":
            _index.build_index(tmp_path, [], provenance={"partition": "fixture"}, **options)
        elif route == "index":
            _index.FeatureIndex(tmp_path, **options)
        else:
            _index._dictionary_hash(dictionary, memory=guard)
    assert not (tmp_path / "index").exists()


def test_cached_verification_keeps_lower_budget_and_historical_stop(tmp_path, monkeypatch):
    model = gemma_model()
    readings = {"current": 1_000_000_000, "peak": 1_000_000_000}
    guard = process_guard(12_000_000_000, readings)
    monkeypatch.setattr(_dictionary, "MemoryGuard", lambda *a: guard)
    identity = _dictionary.verify_hf_model(model, local_gemma(tmp_path, model))
    later = process_guard(28_000_000_000, readings)
    assert _dictionary.model_identity(model, memory_limit_gb=28, memory=later) == identity
    readings.update(current=18_000_000_000, peak=18_000_000_000)
    with pytest.raises(CleanStop, match="memory limit"):
        _dictionary.model_identity(model, memory_limit_gb=28, memory=later)
    readings.update(current=1_000_000_000, peak=1_000_000_000)
    with pytest.raises(CleanStop, match="memory limit"):
        _dictionary.model_identity(model, memory_limit_gb=28, memory=later)
    verified = _dictionary._VERIFIED_MODELS[model]
    assert verified.memory is guard and guard.limit == 12_000_000_000
    assert guard.peak == 18_000_000_000


@pytest.mark.parametrize("model_id", [model_load.MODEL_ID, model_load.GEMMA_ID])
def test_text_loader_refuses_wrong_device_before_source_access(monkeypatch, model_id):
    import huggingface_hub
    guard = process_guard(28_000_000_000, {"current": 0, "peak": 0})
    guard.device = torch.device("mps")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("source admission skipped"))
    with pytest.raises(ValueError, match="guard for its device"):
        model_load.load_text_model(model_id=model_id, device="cpu", memory=guard)


def test_qwen_loader_caps_supplied_budget_before_source_access(monkeypatch):
    import huggingface_hub
    guard = process_guard(28_000_000_000, {"current": 13_000_000_000, "peak": 13_000_000_000})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("source admission skipped"))
    with pytest.raises(CleanStop, match="supplied process memory limit"):
        model_load.load_text_model(device="cpu", memory=guard)


def test_hosted_gemma_factory_retains_owner_guard(monkeypatch):
    import caliper.interp
    model = gemma_model()
    dictionary = random_dictionary(model, latents=8, k=2)
    guard = process_guard(28_000_000_000, {"current": 1_000, "peak": 2_000})
    owner = SimpleNamespace(model=model, tokenizer=TinyTokenizer(), dictionary=dictionary,
                            index=None, guard=guard, sweep=lambda **kw: None, validate=lambda **kw: None)
    service = _hosted.HostedService(owner)
    def tracer(*args, **kwargs):
        assert kwargs["memory"] is guard and kwargs["memory_limit_gb"] == 40
        return "tracer"
    monkeypatch.setattr(caliper.interp, "Tracer", SimpleNamespace(_for_service=tracer))
    assert service._gemma_tracer() == "tracer"


def test_evaluation_cli_uses_one_guard_through_real_dictionary_load(tmp_path, monkeypatch):
    import json
    from tools.interp import eval_dictionary as command
    from .test_training_stream import documents
    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=8, k=2, dtype=torch.float32))
    memory = process_guard(12_000_000_000, {"current": 1_000, "peak": 2_000})
    monkeypatch.setattr(command, "MemoryGuard", lambda *a: memory)
    monkeypatch.setattr(command, "load_text_model", lambda **kw: (model, TinyTokenizer()))
    monkeypatch.setattr(command, "fineweb_documents", lambda **kw: documents())
    monkeypatch.setattr(_dictionary, "MemoryGuard", lambda *a: pytest.fail("dictionary replaced the supplied guard"))
    result = command.main(["--dictionary", str(tmp_path), "--device", "cpu", "--tokens", "4",
                           "--context", "2", "--batch-tokens", "2", "--chunk-tokens", "1"])
    quality = json.loads((tmp_path / "dictionary.json").read_text())["quality"]
    assert quality["evaluated_tokens"] == 4
    assert result == (0 if quality["status"] == "passed" else 2)
    assert memory.limit == 12_000_000_000 and memory.peak == 2_000
