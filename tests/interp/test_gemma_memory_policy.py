"""Decimal memory boundaries with scalar evidence and no model allocations."""
from types import SimpleNamespace

import pytest

from caliper.interp import _admission, _dictionary
from caliper.interp._memory import BudgetMemory, CleanStop, MemoryGuard, family_memory_limit
from tools.interp import build_index, text_stream


GB = 1_000_000_000
GEMMA = SimpleNamespace(config=SimpleNamespace(model_type="gemma2"))


def guard(limit, evidence):
    return MemoryGuard(limit, "cpu", resident=lambda: evidence.get("rss", 0),
                       footprint=lambda: evidence.get("current", 0),
                       peak=lambda: evidence.get("peak", 0), driver=lambda: evidence.get("driver", 0),
                       pressure=lambda: 80, clock=lambda: 0)


def test_gemma_accepts_exact_40_decimal_gb_and_retains_a_breach():
    evidence = {"current": 40 * GB}
    memory = guard(family_memory_limit(GEMMA, 40), evidence)
    memory.check("at cap")
    assert memory.current == memory.peak == memory.limit == 40 * GB
    evidence["current"] += 1
    with pytest.raises(CleanStop, match="memory limit"):
        memory.check("over cap")
    evidence["current"] = 0
    with pytest.raises(CleanStop, match="memory limit"):
        memory.check("released breach")


@pytest.mark.parametrize("limit", [40.000000001, float("inf"), float("nan"), 0, -1, True])
def test_invalid_limits_refuse_before_any_memory_read(limit):
    with pytest.raises(ValueError):
        family_memory_limit(GEMMA, limit)
    with pytest.raises(ValueError):
        guard(limit, {})
    with pytest.raises(ValueError):
        _dictionary._memory_limit(limit, _dictionary._GEMMA_IDENTITY, "verification")


def test_qwen_keeps_12_gb_and_gemma_requires_an_explicit_larger_budget():
    qwen = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text"))
    assert family_memory_limit(qwen) == family_memory_limit(GEMMA) == 12
    assert family_memory_limit(qwen, 12) == 12
    with pytest.raises(ValueError, match="at most 12 GB"):
        family_memory_limit(qwen, 12.000000001)


@pytest.mark.parametrize("identity", [
    _dictionary._GEMMA_IDENTITY,
    "hf:google/gemma-2-2b@" + "0" * 40,
    "hf:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
])
def test_larger_dictionary_budget_requires_exact_pinned_identity(identity):
    if identity == _dictionary._GEMMA_IDENTITY:
        _dictionary._memory_limit(40, identity, "verification")
    else:
        with pytest.raises(ValueError, match="requires the pinned Gemma model"):
            _dictionary._memory_limit(40, identity, "verification")


@pytest.mark.parametrize("limit", [8, 12, 28])
def test_40_gb_request_preserves_supplied_lower_and_dynamic_caps(limit):
    supplied = guard(limit, {"current": limit * GB})
    assert family_memory_limit(GEMMA, 40, supplied) == limit
    memory = BudgetMemory(supplied, 40)
    memory.limit = 40 * GB
    memory.check("at supplied cap")
    assert memory.limit == supplied.limit == limit * GB
    supplied.limit -= 1
    with pytest.raises(CleanStop, match="memory limit"):
        memory.check("lower supplied cap")


def test_model_allocation_still_requires_prospective_process_headroom():
    memory = guard(40, {"current": 39 * GB})
    _admission._admit_load(memory, GB, "at cap", staged=True)
    with pytest.raises(CleanStop, match="model allocation"):
        _admission._admit_load(memory, GB + 1, "over cap", staged=True)


@pytest.mark.parametrize("total", [64 * GB, 128 * GB])
def test_parquet_leaves_exactly_8_decimal_gb_after_allocation(monkeypatch, total):
    required = 128_904_264
    host = SimpleNamespace(total=total, available=8 * GB + required)
    monkeypatch.setattr("psutil.virtual_memory", lambda: host)
    # The old guard admits this cap, which isolates the host reserve regression.
    memory = guard(28, {"current": 20 * GB})
    text_stream._admit_parquet_decode(memory, required)
    host.available -= 1
    with pytest.raises(CleanStop, match="headroom"):
        text_stream._admit_parquet_decode(memory, required)


def test_host_headroom_cannot_override_prospective_process_cap(monkeypatch):
    evidence = {"current": 39 * GB}
    memory = guard(40, evidence)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=64 * GB, available=60 * GB))
    text_stream._admit_parquet_decode(memory, GB)
    with pytest.raises(CleanStop, match="headroom"):
        text_stream._admit_parquet_decode(memory, GB + 1)
    # The interval cache must not hide a new current reading.
    evidence["current"] += 1
    with pytest.raises(CleanStop, match="headroom"):
        text_stream._admit_parquet_decode(memory, GB)


@pytest.mark.parametrize("requested", [None, 8, 28, 40])
def test_index_gemma_budget_defaults_to_40_and_keeps_smaller_requests(requested):
    args = ["--dictionary", "fixture", "--model", _admission.GEMMA_ID]
    if requested is not None:
        args += ["--memory-gb", str(requested)]
    parsed = build_index.parser().parse_args(args)
    assert parsed.memory_gb == (40 if requested is None else requested)


@pytest.mark.parametrize("model,limit", [(_admission.GEMMA_ID, 40.000000001), (build_index.MODEL_ID, 12.000000001)])
def test_index_rejects_budget_above_selected_family_cap(model, limit):
    with pytest.raises(SystemExit):
        build_index.parser().parse_args(["--dictionary", "fixture", "--model", model, "--memory-gb", str(limit)])


def test_trace_allocations_keep_prospective_host_reserve(monkeypatch):
    from caliper.interp._frozen import MemoryGuard as TraceMemory
    required = 12345
    host = SimpleNamespace(available=8 * GB + required)
    monkeypatch.setattr("psutil.virtual_memory", lambda: host)
    monkeypatch.setattr(TraceMemory, "check", lambda self, stage: 0)
    memory = TraceMemory("cpu", memory_limit_gb=40, host_reserve_bytes=8 * GB)
    memory.reserve("at reserve", required)
    host.available -= 1
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        memory.reserve("below reserve", required)


def test_checkpoint_conversion_keeps_supplied_guard_and_exact_bytes(tmp_path):
    import torch
    from .test_gemma_runtime import gemma_model, local_gemma
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    model.to(dtype=torch.bfloat16)
    evidence = {"current": 0}
    supplied = guard(12, evidence)
    identity = _dictionary.verify_hf_model(model, snapshot, memory=supplied)
    assert identity == _dictionary._GEMMA_IDENTITY
    assert supplied in _dictionary._VERIFIED_MODELS[model].memory.guards
    model.model.layers[0].mlp.up_proj.weight.data[0, 0] += 1
    with pytest.raises(ValueError, match="weights differ"):
        _dictionary.verify_hf_model(model, snapshot, memory=supplied)
    evidence["current"] = 12 * GB + 1
    with pytest.raises(CleanStop, match="memory limit"):
        _dictionary.verify_hf_model(model, snapshot, memory=supplied)


@pytest.mark.parametrize("admit", [_admission._admit_load, _dictionary._admit])
def test_resource_load_leaves_prospective_host_reserve(monkeypatch, admit):
    required = 4096
    host = SimpleNamespace(available=8 * GB + required)
    monkeypatch.setattr("psutil.virtual_memory", lambda: host)
    memory = BudgetMemory(guard(40, {"current": 0}), 40, host_reserve_bytes=8 * GB)
    admit(memory, required, "exact reserve")
    host.available -= 1
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        admit(memory, required, "reserve breach")


def test_small_cpu_admission_does_not_require_a_gemma_host(monkeypatch):
    from caliper.interp._frozen import MemoryGuard as TraceMemory
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=2 * GB))
    memory = guard(12, {"current": 0})
    _admission._admit_load(memory, 4096, "small model")
    _dictionary._admit(memory, 4096, "small dictionary")
    monkeypatch.setattr(TraceMemory, "check", lambda self, stage: 0)
    trace = TraceMemory("cpu")
    trace.reserve("small trace", 4096)
    assert trace.batch_size(2, layers=2, tokens=3, width=8, mixer_width=8) == 2
    # Staged Gemma construction retains its reserve even with a lower process cap.
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        _admission._admit_load(memory, 4096, "Gemma model", staged=True)


def test_full_gemma_reserve_survives_lower_and_nested_budgets(monkeypatch):
    from caliper.interp._memory import host_reserve_for_model
    full = SimpleNamespace(config=SimpleNamespace(model_type="gemma2", num_hidden_layers=26, hidden_size=2304))
    tiny = SimpleNamespace(config=SimpleNamespace(model_type="gemma2", num_hidden_layers=2, hidden_size=8))
    assert host_reserve_for_model(full) == 8 * GB
    assert host_reserve_for_model(tiny) == 0
    memory = BudgetMemory(guard(40, {}), 40, host_reserve_bytes=host_reserve_for_model(full))
    lower = BudgetMemory(guard(12, {}), 12, extra=memory)
    assert lower.host_reserve_bytes == 8 * GB and lower.limit == 12 * GB
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=8 * GB + 4095))
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        _dictionary._admit(lower, 4096, "nested reserve")
