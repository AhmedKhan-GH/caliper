"""CPU phase admission through the hosted guard and intervention snapshot."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _hosted, _intervene
from caliper.interp._memory import CleanStop, MemoryGuard, allocation_bytes
from .fixture import TinyLM, TinyTokenizer
from .test_intervene import dictionary_for
from .test_intervention_dictionary_identity import feature_spec
from .test_memory_phases import process_guard


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    import caliper
    from caliper import _controls, _managed
    model = TinyLM(layers=2, width=8, heads=2, latents=16, k=2)
    tokenizer, controls = TinyTokenizer(), _controls.Controls()
    dictionary = dictionary_for(model)
    readings = {"current": 1_000, "peak": 9_999_999}
    process = process_guard(10_000_000, readings)
    index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"},
                            close=lambda: None)
    monkeypatch.delenv("CALIPER_FIXTURE_DISPLAY_MPS", raising=False)
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_managed, "current", lambda: None)
    monkeypatch.setattr(caliper, "watch", lambda *a, **kw: SimpleNamespace(
        remove=controls.interventions_on(model).remove))
    monkeypatch.setattr(caliper, "param", controls.param)
    monkeypatch.setattr(caliper, "action", controls.action)
    run = _hosted.HostedInference("tiny-lm", tmp_path, "AB", resources=(
        model, tokenizer, dictionary, index, process))
    monkeypatch.setattr(run, "boundary", lambda: None)
    yield run, readings, process
    run.close()


def test_hosted_start_retains_snapshot_after_released_peak(hosted):
    run, readings, process = hosted
    expected = _intervene._dictionary_identifier(run.dictionary, process)
    run.start()
    executor = run.interventions.executor
    snapshot = executor._dictionary_snapshot
    assert isinstance(run.guard, _hosted._RunGuard)
    assert executor.resource_guard is run.guard
    assert executor.dictionary_id == expected
    assert snapshot is not None
    assert run.inference_status()["dictionary_validation"]["mode"] == "snapshot"
    assert snapshot.matches(run.dictionary, expected, run.guard)
    assert run.guard.current == 1_000 and run.guard.peak == 9_999_999
    readings["current"] = 500
    assert run.guard.check("released snapshot phase") == 500
    assert allocation_bytes(run.guard) == 500
    run.dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
    with pytest.raises(_intervene.InterventionError, match="dictionary content changed"):
        executor.validate(feature_spec())


def test_hosted_snapshot_keeps_pending_copy_floor(hosted, monkeypatch):
    run, readings, process = hosted
    readings["peak"] = readings["current"]
    identifier = _intervene._dictionary_identifier(run.dictionary, run.guard)
    total = sum(t.numel() * t.element_size() for t in run.dictionary.state_dict().values())
    clone, copies = torch.Tensor.clone, []

    def copied(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        process.limit = readings["current"] + total + 4 * _dictionary._HASH_CHUNK_BYTES - 1
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", copied)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        _intervene._DictionarySnapshot.create(run.dictionary, identifier, run.guard)
    assert len(copies) == 1


def test_hosted_current_reserve_and_historical_stop(hosted, monkeypatch):
    run, readings, _ = hosted
    run.start()
    identifier = run.interventions.executor.dictionary_id
    readings["current"] = 9_999_999
    monkeypatch.setattr(torch.Tensor, "clone", lambda *a, **kw: pytest.fail("unadmitted copy"))
    assert _intervene._DictionarySnapshot.create(run.dictionary, identifier, run.guard) is None
    readings.update(current=1_000, peak=10_000_001)
    with pytest.raises(CleanStop, match="memory limit") as stopped:
        _intervene._DictionarySnapshot.create(run.dictionary, identifier, run.guard)
    assert run._failure is run.guard.failure is run.service.failure is stopped.value
    readings["peak"] = 0
    with pytest.raises(CleanStop) as latched:
        run.guard.check("after release", force=False)
    assert latched.value is stopped.value


def test_hosted_wrapper_preserves_force_interval_and_pressure_cache():
    clock, values, calls = [0], {"current": 100, "peak": 900}, []
    process = MemoryGuard(.000001, "cpu", interval=15, clock=lambda: clock[0],
                          pressure=lambda: calls.append("pressure") or 50,
                          resident=lambda: calls.append("resident") or values["current"],
                          footprint=lambda: values["current"], peak=lambda: values["peak"])
    guard = _hosted._RunGuard(process)
    assert guard.check("first", force=False) == 100
    values["current"] = 50
    assert guard._check_interval("cached") == 100
    assert calls == ["resident", "pressure"]
    assert guard.check("forced") == 50
    assert calls == ["resident", "pressure", "resident"]
    clock[0] = 15
    assert guard.check("interval", force=False) == 50
    assert calls == ["resident", "pressure", "resident", "resident", "pressure"]
    assert guard.peak == 900


def test_hosted_wrapper_retains_stage_only_readings_and_lower_extra_limit():
    values = iter([900, 100, 50])
    primary = SimpleNamespace(check=lambda stage: next(values), limit=1_000)
    extra = SimpleNamespace(check=lambda stage: None, limit=950, current=25, peak=30)
    stops = []
    guard = _hosted._RunGuard(primary, extra=extra, on_stop=stops.append)
    assert guard.check("first") == 900
    assert guard.check("released", force=False) == 100
    assert guard.peak == 900
    extra.limit = 899
    with pytest.raises(CleanStop, match="memory limit") as stopped:
        guard.check("lower cap")
    assert guard.current == 50 and stops == [stopped.value]
    with pytest.raises(CleanStop) as latched:
        guard._check_interval("stopped")
    assert latched.value is stopped.value and stops == [stopped.value]


@pytest.mark.parametrize("keyword_only", [False, True])
def test_hosted_wrapper_passes_force_without_an_interval_method(keyword_only):
    calls = []

    def explicit(stage, *, force=False):
        calls.append((stage, force))
        return 100

    def keywords(stage, **kwargs):
        calls.append((stage, kwargs["force"]))
        return 100

    guard = _hosted._RunGuard(SimpleNamespace(check=explicit if keyword_only else keywords))
    assert guard.check("forced") == 100
    assert guard.check("deferred", force=False) == 100
    assert guard._check_interval("interval") == 100
    assert calls == [("forced", True), ("deferred", False), ("interval", False)]


def test_hosted_wrapper_does_not_retry_a_guard_type_error():
    calls = []

    def check(stage, *, force=True):
        calls.append((stage, force))
        raise TypeError("guard implementation error")

    guard = _hosted._RunGuard(SimpleNamespace(check=check))
    with pytest.raises(TypeError, match="guard implementation error"):
        guard.check("once", force=False)
    assert calls == [("once", False)] and guard.failure is None


@pytest.mark.parametrize("extra", [False, True])
def test_hosted_wrapper_propagates_supplied_stop_once(extra):
    failure = CleanStop("explicit stop")
    calls, stops = [], []

    def stop(stage):
        calls.append(stage)
        raise failure

    stopped = SimpleNamespace(check=stop)
    guard = _hosted._RunGuard(None if extra else stopped, extra=stopped if extra else None,
                              on_stop=stops.append)
    for force in (False, True):
        with pytest.raises(CleanStop) as caught:
            guard.check("request", force=force)
        assert caught.value is failure
    assert calls == ["request"] and stops == [failure]
