"""CPU admission checks for local safetensors weight headers and example windows."""
import json

import pytest
from safetensors.torch import save_file
import torch

from caliper.interp import _admission
from caliper.interp._backend import make_tiny_backend
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._memory import CleanStop, MemoryGuard


def guard(limit_gb=12):
    return MemoryGuard(
        limit_gb, "cpu", resident=lambda: 0, footprint=lambda: 0,
        peak=lambda: 0, pressure=lambda: 50, driver=lambda: 0,
    )


def test_model_admission_reads_only_bounded_headers(tmp_path, monkeypatch):
    backend = make_tiny_backend()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tensors = {k: v.detach().cpu().contiguous() for k, v in backend.model.state_dict().items()}
    save_file(tensors, str(snapshot / "model.safetensors"))
    memory = guard()
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("tensor allocation preceded admission"))
    loaded, largest = _admission._admit_model_load(snapshot, memory, "cpu")
    weights = list(tensors.values())
    assert loaded == sum(v.numel() * max(4, v.element_size()) for v in weights)
    assert largest == max(v.numel() * max(4, v.element_size()) for v in weights)
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


def test_example_window_caching_and_stage_checks(tmp_path, monkeypatch):
    backend = make_tiny_backend()
    docs = [("doc-0", [2, 37, 38, 39, 40] * 4)]
    build_index(
        tmp_path, docs, model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, provenance={"partition": "fixture"}, context_length=16,
    )
    memory = guard()
    stages = []
    orig_check = memory.check
    monkeypatch.setattr(memory, "check", lambda stage, **kw: (stages.append(stage), orig_check(stage, **kw))[1])
    with FeatureIndex(
        tmp_path, model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, memory=memory,
    ) as index:
        feat = int(index.counts[0].argmax())
        rows = index.examples(0, feat)
        assert rows and stages == ["example windows admission", "example windows complete"]
        assert index.examples(0, feat) == rows
        assert len(stages) == 2
