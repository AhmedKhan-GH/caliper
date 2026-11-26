"""Operational readiness and persistent errors at the offline app boundary."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_circuit_trace import driver
from _visual import visual_checks


@pytest.fixture
def live_driver(tmp_path, monkeypatch):
    state = {"live": True, "closed": False, "fault": None, "holds": []}
    spec = {"caliper_spec": 1, "applet": {"id": "circuit-trace", "title": "Published Dallas"},
            "layout": {"split": "vertical", "children": ["clamp_status", "graph"]}}

    def body(method, **args):
        if method == "spec.get":
            return {"text": "# Fixture Instrument\n", "json": deepcopy(spec)}
        if method == "run.state":
            return {"attached": True}
        assert method in ("applet.save", "spec.set_text")
        if method == "spec.set_text":
            state["error_spec"] = args["text"]
        return {}

    def capture(name, steps):
        program = {"id": "probabilities", "last_pass_ms": 51 if state["fault"] == "budget" else 1,
                   "memory_bytes": 1024, "memory_limit_bytes": 64 << 20,
                   "draw": {"frame_ms": 1, "frame_limit_ms": 4,
                            "memory_bytes": 1024, "memory_limit_bytes": 64 << 20}}
        session.program_evidence[name] = {"status": {"programs": [program]},
            "expected_steps": {"probabilities": 1}, "pieces": {"probabilities": {
                "derive": {"time_ms": 50, "memory_mb": 64}, "draw": {"frame_ms": 4, "memory_mb": 64}}}}
        return session.out / (name + ".png"), None if session.show or state["fault"] == "metadata" else {}

    def capture_checks(*args, **kwargs):
        ids = np.zeros((20, 100, 2), dtype=np.uint32)
        for index in range(10):
            ids[5:15, index * 10:index * 10 + 8] = [1, index]
        panel = {"programs": [{"pick_id": 1}], "clip_from_world": np.eye(4).reshape(-1).tolist(),
                 "labels": [{"text": f"token{index}", "box": [index * 10, 0, index * 10 + 8, 4]}
                            for index in range(9 if state["fault"] == "labels" else 10)]}
        return visual_checks("output", panel, ids, {}, node_count=0), {}

    def keep_alive(bridge):
        assert state["live"] and not state["closed"]
        state["holds"].append((session.out / "demo-error.json").exists())
        state["live"] = False

    def close():
        assert not state["live"]
        state["closed"] = True

    def launch(args, out, *, show, interactive):
        session.out, session.show = out, show
        out.mkdir()
        return session

    session = SimpleNamespace(mcp=SimpleNamespace(body=body), live=lambda: state["live"], close=close,
        run_id="fixture", cleanup_error=None, replication_profile=driver.published.PROFILE, program_evidence={},
        private_output_directory=tmp_path, ready_programs=lambda names: None, capture=capture,
        example_page=lambda page: ({"step": 1}, {}),
        read=lambda *args, **kwargs: {"step": 1, "elements": [[0, 0, 1, 1, .5, .5, 0, i] for i in range(10)]},
        events=SimpleNamespace(complete=lambda *args, **kwargs: {"step": 1, "seconds": 21}))
    graph = {"counts": {}, "completeness": {"max_absolute_residual": .03}}
    monkeypatch.setattr(driver, "Session", launch)
    monkeypatch.setattr(driver, "check_assets", lambda s: {"fixture": {"status": 200}})
    monkeypatch.setattr(driver, "instrument", lambda s: None)
    monkeypatch.setattr(driver, "programs", lambda s: None)
    monkeypatch.setattr(driver, "published_action", lambda *args: {
        "completed": True, "evidence": {"result": {"argmax_token_ids": [0, 0]}}})
    monkeypatch.setattr(driver.published, "replication_state", lambda status: ({}, {}))
    monkeypatch.setattr(driver.published, "preparation", lambda *args: {"id": "fixture", "prompts": {"dallas": {"token_ids": []}}})
    monkeypatch.setattr(driver.published, "read_graph", lambda *args, **kwargs: (graph, []))
    monkeypatch.setattr(driver.published, "published_selection", lambda graph: {"index": 0})
    monkeypatch.setattr(driver.Bridge, "select", lambda *args: {})
    monkeypatch.setattr(driver.Bridge, "keep_alive", keep_alive)
    monkeypatch.setattr(driver, "capture_checks", capture_checks)
    monkeypatch.setattr(driver, "memory_bytes", lambda *args: 1024)

    def run(fault=None, *, show=False):
        state["fault"] = fault
        model = tmp_path / "models--google--gemma-2-2b" / "snapshots" / driver.GEMMA_REVISION
        out = tmp_path / "driver"
        code = driver.main(["--model", str(model), "--dictionary", str(tmp_path / "dictionary"),
            "--replication-profile", driver.published.PROFILE, "--private-output-directory", str(tmp_path),
            "--out", str(out), "--show" if show else "--demo"])
        return code, json.loads((out / "result.json").read_text()), session.out, state
    return run


@pytest.mark.parametrize("fault,check", [("labels", "cell10.output_label_count"),
                                         ("budget", "cell10.probabilities.derive_time_ms"),
                                         ("metadata", "cell10.native_pixels")])
def test_hidden_live_refuses_completed_failed_display(live_driver, fault, check):
    code, result, out, state = live_driver(fault)
    assert check in result["failed_checks"]
    assert code == 1
    assert not result.get("demo_ready") and not (out / "demo-ready.json").exists()
    assert check in result["error"] and check in state["error_spec"]
    assert state["holds"] == [True] and state["closed"]


@pytest.mark.parametrize("show", [False, True])
def test_live_display_readiness_is_separate_from_scientific_acceptance(live_driver, show):
    code, result, out, state = live_driver(show=show)
    assert code == 0 and result["demo_ready"]
    assert (out / "demo-ready.json").exists()
    assert {"trace_time", "completeness", "external_replication_review"} <= set(result["failed_checks"])
    assert not result["accepted_replication"] and not result["gemma_acceptance"]
    assert state["holds"] == [False] and state["closed"]


def test_visible_live_refuses_a_measured_budget_failure(live_driver):
    code, result, out, state = live_driver("budget", show=True)
    assert code == 1 and not result.get("demo_ready")
    assert not (out / "demo-ready.json").exists()
    assert state["holds"] == [True] and state["closed"]
