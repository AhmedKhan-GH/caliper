"""Loader budgets with logical memory readers and small CPU fixtures."""
import sys
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary as dictionary
from caliper.interp._memory import CleanStop, MemoryGuard
from tools.interp import model_load
from .test_model_load import hf_model, mocked_loader


ROUTES = [
    pytest.param({}, 12, id="qwen-default"),
    pytest.param({"mode": "inference"}, 12, id="qwen-inference"),
    pytest.param({"mode": "training"}, 24, id="qwen-training"),
    pytest.param({"model_id": model_load.GEMMA_ID}, 40, id="gemma-inference"),
]


def logical_guard(limit, readings):
    return MemoryGuard(limit, "cpu", resident=lambda: readings["current"],
                       footprint=lambda: readings["current"], peak=lambda: readings["peak"],
                       pressure=lambda: readings.get("pressure", 50), driver=lambda: 0)


class SourceBoundary(Exception):
    pass


@pytest.fixture
def source_boundary(monkeypatch):
    calls = []

    def source(model_id, *, revision, local_files_only):
        calls.append((model_id, revision, local_files_only))
        raise SourceBoundary

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=source))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=object(), Qwen3_5ForCausalLM=object(), Qwen3_5TextConfig=object()))
    return calls


@pytest.mark.parametrize("selection,ceiling", ROUTES)
@pytest.mark.parametrize("supplied", [False, True])
def test_loader_enforces_mode_ceiling_before_source(monkeypatch, source_boundary, selection, ceiling, supplied):
    readings = {"current": int(ceiling * 1e9), "peak": 0}
    created = []

    def factory(limit, device):
        assert device == "cpu" and limit == ceiling
        created.append(limit)
        return logical_guard(limit, readings)

    monkeypatch.setattr(model_load, "MemoryGuard", factory)
    guard = logical_guard(40, readings) if supplied else None
    with pytest.raises(SourceBoundary):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    model_id = selection.get("model_id", model_load.MODEL_ID)
    revision = model_load.GEMMA_REVISION if model_id == model_load.GEMMA_ID else model_load.MODEL_REVISION
    assert source_boundary == [(model_id, revision, True)]
    assert created == ([] if supplied else [ceiling])
    readings["current"] += 1
    with pytest.raises(CleanStop, match="memory limit"):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    assert len(source_boundary) == 1
    if supplied:
        assert guard.limit == 40_000_000_000


@pytest.mark.parametrize("selection,ceiling", ROUTES)
def test_loader_preserves_lower_supplied_limit(source_boundary, selection, ceiling):
    readings = {"current": 8_000_000_000, "peak": 0}
    guard = logical_guard(8, readings)
    with pytest.raises(SourceBoundary):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    guard.limit -= 1
    with pytest.raises(CleanStop, match="memory limit"):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    assert len(source_boundary) == 1 and guard.limit == 7_999_999_999


@pytest.mark.parametrize("selection,ceiling", ROUTES)
@pytest.mark.parametrize("stop", ["history", "pressure", "prior"])
def test_loader_preserves_supplied_stops(monkeypatch, source_boundary, selection, ceiling, stop):
    readings = {"current": 0, "peak": 0}
    guard = logical_guard(8, readings)
    if stop == "history":
        readings["peak"] = guard.limit + 1
        with pytest.raises(CleanStop, match="memory limit"):
            guard.check("prior breach")
        readings["peak"] = 0
        message = "memory limit"
    elif stop == "pressure":
        readings["pressure"] = 14
        message = "memory pressure"
    else:
        def prior_stop(stage):
            assert stage == "text model admission"
            raise CleanStop("prior stop")
        monkeypatch.setattr(guard, "check", prior_stop)
        message = "prior stop"
    with pytest.raises(CleanStop, match=message):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    assert source_boundary == []


@pytest.mark.parametrize("selection,ceiling", ROUTES)
def test_loader_refuses_wrong_guard_device(source_boundary, selection, ceiling):
    guard = logical_guard(40, {"current": 0, "peak": 0})
    guard.device = torch.device("mps")
    with pytest.raises(ValueError, match="guard for its device"):
        model_load.load_text_model(device="cpu", memory=guard, **selection)
    assert source_boundary == []


@pytest.mark.parametrize("mode", [None, True, False, 24, "", "train", "TRAINING", [], {}])
def test_loader_refuses_invalid_mode_before_guard(monkeypatch, source_boundary, mode):
    monkeypatch.setattr(model_load, "MemoryGuard", lambda *a: pytest.fail("invalid mode created a guard"))
    with pytest.raises(ValueError, match="unsupported text model load mode"):
        model_load.load_text_model(device="cpu", mode=mode)
    assert source_boundary == []


def test_loader_refuses_gemma_training_before_guard(monkeypatch, source_boundary):
    monkeypatch.setattr(model_load, "MemoryGuard", lambda *a: pytest.fail("invalid model mode created a guard"))
    with pytest.raises(ValueError, match="training mode requires the Qwen text model"):
        model_load.load_text_model(device="cpu", model_id=model_load.GEMMA_ID, mode="training")
    assert source_boundary == []


@pytest.mark.parametrize("mode,limit,current,expected", [
    ("inference", 40, 10, 12), ("training", 24, 18, 24),
    ("training", 40, 18, 24), ("training", 8, 6, 8),
])
def test_qwen_fixture_retains_mode_budget_and_prior_stops(tmp_path, monkeypatch, mode, limit, current, expected):
    model = hf_model()
    calls = mocked_loader(tmp_path, monkeypatch, model, {})
    readings = {"current": int(current * 1e9), "peak": 0}
    guard = logical_guard(limit, readings)
    loaded, _ = model_load.load_text_model(device="cpu", memory=guard, mode=mode)
    assert loaded is model and calls == ["snapshot", "weights", "tokenizer"]
    retained = dictionary._VERIFIED_MODELS[model].memory
    assert retained.memory is guard and retained.limit == int(expected * 1e9)
    assert guard.limit == int(limit * 1e9)
    assert dictionary.model_identity(model) == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
    readings["current"] = retained.limit + 1
    with pytest.raises(CleanStop, match="memory limit"):
        dictionary.model_identity(model)
    readings["current"] = 0
    with pytest.raises(CleanStop, match="memory limit"):
        dictionary.model_identity(model)


def test_qwen_training_fixture_retains_dynamic_lower_limit(tmp_path, monkeypatch):
    model = hf_model()
    mocked_loader(tmp_path, monkeypatch, model, {})
    readings = {"current": 18_000_000_000, "peak": 0}
    guard = logical_guard(24, readings)
    model_load.load_text_model(device="cpu", memory=guard, mode="training")
    guard.limit = readings["current"] - 1
    with pytest.raises(CleanStop, match="memory limit"):
        dictionary.model_identity(model)
    assert dictionary._VERIFIED_MODELS[model].memory.limit == guard.limit


@pytest.mark.parametrize("mode,current", [("inference", 11.6), ("training", 23.6)])
def test_qwen_fixture_refuses_prospective_load_before_weights(tmp_path, monkeypatch, mode, current):
    model = hf_model()
    calls = mocked_loader(tmp_path, monkeypatch, model, {})
    guard = logical_guard(40, {"current": int(current * 1e9), "peak": 0})
    with pytest.raises(CleanStop, match="model allocation exceeds"):
        model_load.load_text_model(device="cpu", memory=guard, mode=mode)
    assert calls == ["snapshot"] and model not in dictionary._VERIFIED_MODELS
