"""CPU command checks for the source admission health callback."""
import json
import signal

import pytest

from caliper._interpret_protocol import bounded
from tools.interp import eval_dictionary as evaluation
from tools.interp.memory_limits import MemoryGuard, StopControl
from .fixture import TinyTokenizer
from .test_dictionary import memory_readers
from .test_evaluation_binding import fixture, publications


@pytest.mark.parametrize("stop_kind", ["none", "signal", "peak", "pressure", "sustained_pressure"])
def test_source_admission_checks_stop_and_memory_before_evaluation(tmp_path, monkeypatch, capsys, stop_kind):
    model, dictionary = fixture(tmp_path)
    cursor = tmp_path / "evaluation" / "state.json"
    cursor.parent.mkdir()
    cursor.write_text('{"prior_cursor": true}\n')
    before = publications(tmp_path)
    readings = {"bytes": 8_000_000, "free": 80}
    guard = MemoryGuard(.064, "cpu", interval=0, pressure=lambda: readings["free"],
                        resident=lambda: readings["bytes"], footprint=lambda: readings["bytes"],
                        peak=lambda: readings["bytes"])
    control = StopControl()
    events = []
    manifest = tmp_path / "source.json"
    cache = tmp_path / "cache"

    class Source:
        source_id = "fixture-source"

        def close(self):
            events.append("close")

    def admit_source(*, cache_dir, local_manifest, health):
        assert cache_dir == cache and local_manifest == manifest
        health()
        if stop_kind == "signal":
            control._signal(signal.SIGTERM, None)
        elif stop_kind == "peak":
            readings["bytes"] = guard.limit + 1
        elif stop_kind in ("pressure", "sustained_pressure"):
            readings["free"] = 14 if stop_kind == "pressure" else 19
        # The source worker invokes this callback between bounded admission steps.
        health()
        health()
        events.append("source admitted")
        return Source()

    def evaluate(*args, memory, stop, **kwargs):
        assert stop_kind == "none"
        assert memory is guard and stop is control
        events.append("evaluate")
        return {"status": "passed"}

    monkeypatch.setattr(evaluation, "StopControl", lambda: control)
    monkeypatch.setattr(evaluation, "MemoryGuard", lambda *a: guard)
    monkeypatch.setattr(evaluation, "load_text_model", lambda **k: (model, TinyTokenizer()))
    monkeypatch.setattr(evaluation, "load_dictionary", lambda *a, **k: dictionary)
    monkeypatch.setattr(evaluation, "fineweb_documents", admit_source)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    capsys.readouterr()
    code = evaluation.main(["--dictionary", str(tmp_path), "--device", "cpu", "--memory-gb", ".064",
                            "--dataset-cache", str(cache), "--local-dataset-manifest", str(manifest)])
    assert publications(tmp_path) == before
    if stop_kind == "none":
        assert code == 0 and events == ["source admitted", "evaluate", "close"]
    else:
        assert code == 2 and events == []
        result = json.loads(capsys.readouterr().out)
        assert result["stage"] == "clean_stop" and result["status"] == "interrupted"
        assert bounded({"result": result})
        reason_kind = "signal" if stop_kind == "signal" else "memory"
        assert reason_kind in result["reason"]
