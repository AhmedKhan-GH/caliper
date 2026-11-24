"""CPU admission checks for portable weight loads and example windows."""
from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch

from caliper.interp import _admission, _dictionary, _index
from caliper.interp._memory import CleanStop, MemoryGuard
from tools.interp import model_load
from .test_index import indexed
from .test_model_load import hf_model, local_snapshot, mocked_loader


def guard():
    return MemoryGuard(12, "cpu", resident=lambda: 0, footprint=lambda: 0,
                       peak=lambda: 0, pressure=lambda: 50, driver=lambda: 0)


def test_model_admission_reads_only_bounded_headers(tmp_path, monkeypatch):
    model = hf_model()
    snapshot = local_snapshot(tmp_path, model)
    memory = guard()
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("tensor allocation preceded admission"))
    loaded, largest = _admission._admit_model_load(snapshot, memory, "cpu")
    weights = list(model.state_dict().values())
    assert loaded == sum(value.numel() * value.element_size() for value in weights)
    assert largest == max(value.numel() * value.element_size() for value in weights)
    memory.peak = 11_700_000_000
    with pytest.raises(CleanStop, match="model allocation"):
        _admission._admit_model_load(snapshot, memory, "cpu")


@pytest.mark.parametrize("damage", ["header", "shape", "offsets", "dtype"])
def test_model_admission_rejects_unbounded_or_invalid_headers(tmp_path, damage):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    entry = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    if damage == "shape":
        entry["shape"] = [2**31]
    elif damage == "offsets":
        entry["data_offsets"] = [0, 12]
    elif damage == "dtype":
        entry["dtype"] = "unknown"
    raw = json.dumps({"weight": entry}).encode()
    length = 2**30 if damage == "header" else len(raw)
    (folder / "model.safetensors").write_bytes(length.to_bytes(8, "little") + raw + b"abcd")
    with pytest.raises(ValueError, match="header|storage"):
        _admission._admit_model_load(folder, guard(), "cpu")


def test_tool_loader_refuses_before_weight_allocation(tmp_path, monkeypatch):
    model = hf_model()
    calls = mocked_loader(tmp_path, monkeypatch, model, {})
    memory = guard()
    memory.peak = 11_700_000_000
    with pytest.raises(CleanStop, match="model allocation"):
        model_load.load_text_model(device="cpu", memory=memory)
    assert calls == ["snapshot"] and model not in _dictionary._VERIFIED_MODELS


def test_example_admission_precedes_tensor_allocation_and_forward(indexed, monkeypatch):
    index, _ = indexed
    memory = guard()
    memory.peak = memory.limit - 1
    index.memory = memory
    layer, feature = 0, int(index.counts[0].argmax())
    monkeypatch.setattr(torch, "full", lambda *a, **k: pytest.fail("window tensor allocation ran"))
    monkeypatch.setattr(_index, "_forward", lambda *a, **k: pytest.fail("window forward ran"))
    with pytest.raises(CleanStop, match="example allocation"):
        index.examples(layer, feature)
    assert "examples" not in index._cache[(layer, feature)]


def test_example_estimate_covers_encode_and_logits_and_checks_completion(indexed, monkeypatch):
    index, _ = indexed
    memory = guard()
    index.memory = memory
    layer, feature = 0, int(index.counts[0].argmax())
    rows = len(index._windows(layer, feature))
    estimate = index._example_bytes(rows)
    encoded = rows * 33 * index.dictionary.metadata["latents"] * 8
    logits = rows * 33 * index.model.config.vocab_size * 8
    assert estimate > 512_000_000 + encoded + logits
    stages = []
    check = memory.check
    def observed(stage):
        stages.append(stage)
        check(stage)
    monkeypatch.setattr(memory, "check", observed)
    result = index.examples(layer, feature)
    assert result and stages == ["example windows admission", "example windows complete"]
    monkeypatch.setattr(_index, "_forward", lambda *a, **k: pytest.fail("cached query forwarded"))
    assert index.examples(layer, feature) == result
    assert len(stages) == 2


def test_training_tool_import_has_no_hosted_runtime_dependency():
    script = """
import importlib.abc
import sys
sys.path[:0] = sys.argv[1:]
class NoHostedRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "caliper.interp._hosted":
            raise AssertionError("training import reached the M8 hosted runtime")
sys.meta_path.insert(0, NoHostedRuntime())
from tools.interp import model_load
assert "caliper.interp._hosted" not in sys.modules
assert model_load.MODEL_REVISION
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", script, str(root), str(root / "python")],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
