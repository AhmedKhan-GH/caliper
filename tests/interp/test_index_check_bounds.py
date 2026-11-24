"""Bound process reads by batch boundaries, with real CPU encoders and indexes."""
import json
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen, _index, load_dictionary
from caliper.interp._memory import CleanStop, MemoryGuard
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture


def counted_guard():
    state = SimpleNamespace(reads=0, now=0., current=1_000, peak=2_000, free=80)
    def resident():
        state.reads += 1
        return state.current
    guard = MemoryGuard(12, "cpu", resident=resident, footprint=lambda: state.current,
                        peak=lambda: state.peak, pressure=lambda: state.free,
                        driver=lambda: 0, clock=lambda: state.now)
    return guard, state


def mixed_resources(tmp_path):
    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float64
    return model, dictionary


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_index_process_reads_and_logs_do_not_scale_with_encoder_blocks(tmp_path, monkeypatch, capsys, rows):
    model, dictionary = mixed_resources(tmp_path)
    memory, state = counted_guard()
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", rows * 8)
    stages = []
    capsys.readouterr()
    manifest = _index.build_index(
        tmp_path, [("first", [65, 66, 67, 68]), ("second", [69, 70])], model=model,
        tokenizer=TinyTokenizer(), dictionary=dictionary, memory=memory, memory_guard=stages.append,
        provenance={"partition": "fixture"}, context_length=2)
    assert manifest["tokens"] == 6
    assert stages.count("before batch") == stages.count("model batch") == 3
    assert stages.count("dictionary encoder block") == 3 * 2 * (16 // rows)
    assert stages.count("dictionary encoder block complete") == stages.count("dictionary encoder block")
    # Four identity/admission reads, two per batch, and two final reads.
    assert state.reads == 4 + 2 * 3 + 2
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(records) == state.reads
    assert not any(row["stage"] in ("document", "dictionary batch", "dictionary encoder block",
                                    "dictionary encoder block complete") for row in records)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("fault", ["lower", "cancel", "pressure", "current"])
def test_index_inner_checks_stop_without_completion_or_hooks(tmp_path, monkeypatch, fault):
    model, dictionary = mixed_resources(tmp_path)
    memory, state = counted_guard()
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 8)
    completed = []
    def check(stage):
        if stage != "dictionary encoder block complete":
            return
        completed.append(stage)
        if fault == "lower":
            memory.limit = state.peak - 1
        elif fault == "cancel":
            raise CleanStop("fixture cancellation")
        else:
            state.now += 16
            if fault == "pressure":
                state.free = 14
            else:
                state.current = memory.limit + 1
    with pytest.raises(CleanStop):
        _index.build_index(tmp_path, [("first", [65, 66])], model=model, tokenizer=TinyTokenizer(),
                           dictionary=dictionary, memory=memory, memory_guard=check,
                           provenance={"partition": "fixture"}, context_length=2)
    assert len(completed) == 1
    assert not (tmp_path / "index" / "index.json").exists()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("trace_guard", [False, True])
def test_encoder_guard_uses_one_admission_and_interval_checks(tmp_path, monkeypatch, capsys, trace_guard):
    _, dictionary = mixed_resources(tmp_path)
    memory, state = counted_guard()
    if trace_guard:
        monkeypatch.setattr(_frozen, "ProcessMemoryGuard", lambda *a, **k: memory)
        guard = _frozen.MemoryGuard("cpu")
    else:
        guard = memory
    state.reads = 0
    capsys.readouterr()
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 8)
    x = torch.arange(16, dtype=torch.float64).reshape(2, 8) / 16
    result = _frozen._trace_features(dictionary.layers[0], x, guard)
    layer = dictionary.layers[0]
    expected = layer.activate(torch.nn.functional.linear(x, layer.W_enc.double(), layer.b_enc.double()))
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)
    assert state.reads == (2 if trace_guard else 1)
    assert len(capsys.readouterr().err.splitlines()) == 1


@pytest.mark.parametrize("fault", ["lower", "latch"])
def test_encoder_keeps_dynamic_limits_and_stage_only_stop_callbacks(tmp_path, monkeypatch, fault):
    _, dictionary = mixed_resources(tmp_path)
    memory, state = counted_guard()
    stopped = False
    def supplied_check(stage):
        if stopped:
            raise CleanStop("fixture latch")
        memory.check(stage)
    guard = (memory if fault == "lower" else SimpleNamespace(
        limit=memory.limit, device=memory.device, _guard=memory, check=supplied_check))
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 8)
    completed = []
    def check(stage):
        nonlocal stopped
        if stage == "dictionary encoder block complete":
            completed.append(stage)
            if fault == "lower":
                memory.limit = state.peak - 1
            else:
                stopped = True
    with pytest.raises(CleanStop):
        _frozen._trace_features(dictionary.layers[0], torch.ones(2, 8, dtype=torch.float64), guard, check)
    assert len(completed) == 1
