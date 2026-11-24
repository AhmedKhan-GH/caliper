"""CPU admission checks for hosted loads and owner stop propagation."""
from types import SimpleNamespace
import json
import sys

import pytest
import torch

from caliper.interp import _dictionary, _hosted, _index
from caliper.interp._memory import CleanStop
from tools.interp import model_load
from .test_admission import guard
from .test_hosted import hosted_run
from .test_model_load import hf_model, local_snapshot


def hosted_loader(tmp_path, monkeypatch, *, info=None, near_limit=False, transfer_limit=False):
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model)
    (snapshot / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 24, "hidden_size": 1024}}))
    memory = guard()
    if near_limit:
        memory.peak = 11_700_000_000
    calls = []

    def load(*args, **kwargs):
        calls.append("weights")
        assert kwargs["local_files_only"] and kwargs["use_safetensors"]
        model.config = kwargs["config"]
        model.config.model_type = "qwen3_5_text"
        if transfer_limit:
            memory.peak = 11_700_000_000
        return model, info or {}

    def transfer(*args, **kwargs):
        calls.append("transfer")
        return model

    def dictionary(*args, **kwargs):
        calls.append("dictionary")
        assert _dictionary.model_identity(kwargs["model"]) == f"hf:{model_load.MODEL_ID}@{model_load.MODEL_REVISION}"
        return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5TextConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Qwen3_5ForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(model, "to", transfer)
    monkeypatch.setattr(_dictionary, "load_dictionary", dictionary)
    monkeypatch.setattr(_index, "FeatureIndex", lambda *a, **k: object())
    from caliper.interp import _memory
    monkeypatch.setattr(_memory, "MemoryGuard", lambda *a, **k: memory)
    return model, snapshot, calls, memory


@pytest.mark.parametrize("near_limit,transfer_limit", [(True, False), (False, True)])
def test_hosted_loader_refuses_before_each_allocation_stage(tmp_path, monkeypatch, near_limit, transfer_limit):
    model, snapshot, calls, _ = hosted_loader(tmp_path, monkeypatch, near_limit=near_limit, transfer_limit=transfer_limit)
    with pytest.raises(CleanStop, match="model allocation"):
        _hosted.load_resources(str(snapshot), tmp_path)
    assert calls == ([] if near_limit else ["weights"])
    if near_limit:
        assert model not in _dictionary._VERIFIED_MODELS


def test_hosted_loader_records_verified_weights_without_a_cap_change(tmp_path, monkeypatch):
    model, snapshot, calls, memory = hosted_loader(tmp_path, monkeypatch)
    loaded, _, _, _, actual_guard = _hosted.load_resources(str(snapshot), tmp_path)
    assert loaded is model and actual_guard is memory and memory.limit == 12_000_000_000
    assert calls == ["weights", "transfer", "dictionary"]
    assert _dictionary.model_identity(model).startswith("hf:")
    model.weight.data.add_(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        _dictionary.model_identity(model)


def test_incomplete_hosted_load_never_records_verified_weights(tmp_path, monkeypatch):
    model, snapshot, calls, _ = hosted_loader(tmp_path, monkeypatch, info={"missing_keys": ["weight"]})
    with pytest.raises(ValueError, match="weights do not match"):
        _hosted.load_resources(str(snapshot), tmp_path)
    assert calls == ["weights"] and model not in _dictionary._VERIFIED_MODELS


def test_separate_index_admission_stop_reaches_hosted_owner(hosted_run, monkeypatch):
    from caliper import _wire
    run, route = hosted_run
    memory = guard()
    memory.footprint = lambda: memory.limit - 1
    run.index.memory = memory
    saved = []
    monkeypatch.setattr(run.managed, "checkpoint", lambda *args: saved.append(1))
    feature = int(run.index.counts[0].argmax())
    reply = route.on_control(None, _wire.Control(7, "interpret", interpretation={"version": 1,
        "method": "feature.examples", "params": {"layer": 0, "feature": feature}}).to_meta())
    assert reply["status"] == "queued"
    # One real boundary suffices. A missing stop must fail instead of entering
    # the hosted idle loop and leaving this regression unbounded.
    with pytest.raises(CleanStop, match="example allocation") as caught:
        run.boundary()
    assert run._failure is caught.value and saved == [1]
    assert route.transport.replies[-1].error["code"] == "interpret.stopped"
