"""Tests for the Circuit Trace MCP demo interface, controls, and tiny-lm E2E flow."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demos" / "circuit_trace"


def _load_demo_module(name: str):
    spec = importlib.util.spec_from_file_location(f"demo_{name}", DEMO_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"demo_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mounted_ids(layout: dict) -> set[str]:
    node_ids = {node["id"] for node in layout["nodes"]}
    leaves: set[str] = set()
    for node in layout["nodes"]:
        for child in node["children"]:
            if child not in node_ids:
                leaves.add(child)
    return leaves


def test_spec_toml_and_published_manifest():
    spec_mod = _load_demo_module("_spec")
    ui_mod = _load_demo_module("_published_ui")

    encoded = spec_mod.toml_value({"panel": "graph", "count": 8, "ratio": 0.5, "flags": [True, False]})
    assert '"panel" = "graph"' in encoded
    assert '"count" = 8' in encoded

    manifest = ui_mod.load_dallas_manifest()
    assert manifest["prompts"]["dallas"] == "Fact: the capital of the state containing Dallas is"
    cells = [c["cell"] for c in manifest["interventions"]]
    assert cells == [14, 16, 18, 20, 24, 26, 29]
    text = ui_mod.group_text(manifest, 0)
    assert text.startswith("capital:") and "L15 F4494 P4" in text

    default_opts = ui_mod.published_trace_options(show_more=False)
    assert default_opts["node_threshold"] == 0.7
    assert default_opts["edge_threshold"] == 0.9
    assert default_opts["max_targets"] == 5000
    assert default_opts["max_n_logits"] == 10
    assert default_opts["desired_logit_prob"] == 0.95

    more_opts = ui_mod.published_trace_options(show_more=True)
    assert more_opts["node_threshold"] > default_opts["node_threshold"]
    assert more_opts["edge_threshold"] > default_opts["edge_threshold"]


def test_read_only_outputs_and_live_controls():
    ui_mod = _load_demo_module("_published_ui")

    class RecordingMCP:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def body(self, name: str, **kwargs):
            self.calls.append((name, kwargs))
            return {}

    mcp = RecordingMCP()
    session = type("FakeSession", (), {"mcp": mcp})()
    manifest = ui_mod.build_instrument(session, dallas=False)
    assert manifest is None

    controls = {kw["id"]: kw for name, kw in mcp.calls if name == "control.create"}
    states = {kw["id"]: kw for name, kw in mcp.calls if name == "state.create"}
    layout = next(kw["layout"] for name, kw in mcp.calls if name == "layout.set")
    mounted = _mounted_ids(layout)

    # Editable input vs read-only output rows.
    assert controls["prompt"]["kind"] == "text"
    assert controls["prompt"]["bind"] == "param:prompt"
    assert "prompt" in mounted

    for cid, expected_param in (
        ("baseline_text", "param:output.baseline"),
        ("steered_text", "param:output.steered"),
    ):
        assert cid in mounted
        assert controls[cid]["kind"] == "readout"
        assert controls[cid]["kind"] != controls["prompt"]["kind"]
        backing_id = controls[cid]["options"]["fields"]["text"]["control"]
        assert backing_id not in mounted
        assert controls[backing_id]["bind"] == expected_param

    # Real-time Run button, Show More toggle, Preset picker, and Status/Progress row.
    assert "run" in mounted
    assert controls["run"]["kind"] == "button"
    assert controls["run"]["bind"] == "action:run"
    assert controls["run"]["title"] == "Run"

    assert "show_more" in mounted
    assert controls["show_more"]["kind"] == "toggle"
    assert controls["show_more"]["bind"] == "param:show_more"

    assert "trace_status" in mounted
    assert controls["trace_status"]["kind"] == "readout"
    status_backing = controls["trace_status"]["options"]["fields"]["status"]["control"]
    assert status_backing not in mounted
    assert controls[status_backing]["bind"] == "param:output.status"

    assert "preset" in mounted
    assert controls["preset"]["kind"] == "picker"
    assert controls["preset"]["bind"] == "state:prompt_preset"
    assert states["prompt_preset"]["default"] == 0
    choice_labels = [c["label"] for c in controls["preset"]["options"]["choices"]]
    assert "Default" in choice_labels
    assert "Published Dallas" in choice_labels

    # Interactive default spec does not bake Dallas-only cell controls into the layout.
    assert "cell14" not in controls

    # Acceptance mode keeps the published Dallas controls and preset default.
    dallas_mcp = RecordingMCP()
    dallas_manifest = ui_mod.build_instrument(type("FakeSession", (), {"mcp": dallas_mcp})(), dallas=True)
    assert dallas_manifest is not None
    dallas_controls = {kw["id"]: kw for name, kw in dallas_mcp.calls if name == "control.create"}
    dallas_states = {kw["id"]: kw for name, kw in dallas_mcp.calls if name == "state.create"}
    assert "cell14" in dallas_controls
    assert dallas_states["prompt_preset"]["default"] == 1


def test_default_non_city_prompt_presets_and_index_discovery(tmp_path: Path):
    run_mod = _load_demo_module("run")

    expected_default = "He saw a carrot and had to grab it, his hunger was like a starving"
    assert run_mod.DEFAULT_PROMPT == expected_default
    assert run_mod.resolve_startup_prompt("gemma-2-2b") == expected_default
    assert run_mod.resolve_startup_prompt("gemma-2-2b", preset="default") == expected_default
    assert run_mod.resolve_startup_prompt("gemma-2-2b", preset="dallas") == run_mod.DALLAS_PROMPT
    assert run_mod.resolve_startup_prompt("gemma-2-2b", acceptance=True) == run_mod.DALLAS_PROMPT
    assert run_mod.resolve_startup_prompt("gemma-2-2b", prompt="A custom prompt") == "A custom prompt"
    assert run_mod.resolve_startup_prompt("tiny-lm") == run_mod.FIXTURE_PROMPT
    with pytest.raises(ValueError, match="cannot be used together"):
        run_mod.resolve_startup_prompt("gemma-2-2b", prompt="A", preset="dallas")

    dict_dir = tmp_path / "sae"
    dict_dir.mkdir()
    assert run_mod.resolve_feature_index(str(dict_dir)) is None

    sibling_index = tmp_path / "index"
    sibling_index.mkdir()
    (sibling_index / "index.json").write_text("{}", encoding="utf-8")
    assert run_mod.resolve_feature_index(str(dict_dir)) == sibling_index.resolve()

    nested_index = dict_dir / "index"
    nested_index.mkdir()
    (nested_index / "index.json").write_text("{}", encoding="utf-8")
    assert run_mod.resolve_feature_index(str(dict_dir)) == nested_index.resolve()

    explicit_index = tmp_path / "custom-index"
    explicit_index.mkdir()
    assert run_mod.resolve_feature_index(str(dict_dir), str(explicit_index)) == explicit_index.resolve()


def test_prompt_change_and_show_more_retrace_with_pruning_and_regenerate():
    run_mod = _load_demo_module("run")

    trace_calls: list[dict] = []
    actions_run: list[str] = []
    progress_log: list[str] = []
    selected_nodes: list[int] = []
    state_sets: list[tuple[str, object]] = []
    params_set: list[tuple[str, object]] = []
    hosted_prompt = run_mod.DEFAULT_PROMPT
    hosted_show_more = False

    class FakeMCP:
        def body(self, name: str, **kwargs):
            if name == "state.set":
                state_sets.append((kwargs["id"], kwargs["value"]))
            elif name == "run.state":
                return {
                    "params": [
                        {"name": "prompt", "value": hosted_prompt, "pending": None},
                        {"name": "show_more", "value": hosted_show_more, "pending": None},
                    ]
                }
            return {}

    class FakeEvents:
        def complete(self, name: str, **kwargs):
            if name == "trace.run":
                trace_calls.append(kwargs)
                return {"seconds": 0.01}
            if name == "trace.status":
                return {"prompt": trace_calls[-1]["prompt"] if trace_calls else run_mod.DEFAULT_PROMPT}
            raise AssertionError(f"Unexpected complete call: {name}")

    s = run_mod.Session.__new__(run_mod.Session)
    s.mcp = FakeMCP()
    s.events = FakeEvents()
    s.started = True
    s.run_id = "circuit-trace"
    s.current_prompt = run_mod.DEFAULT_PROMPT
    s.show_more = False
    s.progress = "ready"
    s.selected_node = 0
    s.examples_installed = True
    s.hosted_params = {"prompt", "selected_node", "clamp", "retrace"}
    s.hosted_actions = {"apply_clamp", "reset", "generate"}
    s.set_progress = lambda status: progress_log.append(status)
    s.set_param = lambda name, value: params_set.append((name, value))
    s.run_action = lambda name: actions_run.append(name) or {"status": "ok"}
    s.read_full_graph = lambda: {"nodes": [{"index": 3, "kind": "feature"}], "edges": []}
    s.select_feature_node = lambda idx: selected_nodes.append(idx) or {"feature": {"maximum": 2.0}, "examples": {"examples": []}}

    # Prompt change retraces with published pruning, refreshes examples, resets clamp, and regenerates text.
    new_prompt = "The baker pulled a warm loaf from the"
    hosted_prompt = new_prompt
    s.handle_event({"kind": "run", "change": "param", "name": "prompt", "value": new_prompt}, None)

    assert len(trace_calls) == 1
    assert trace_calls[0]["prompt"] == new_prompt
    assert trace_calls[0]["node_threshold"] == 0.7
    assert trace_calls[0]["edge_threshold"] == 0.9
    assert selected_nodes == [3]
    assert actions_run == ["reset", "generate"]
    assert progress_log == ["tracing", "ready"]
    assert ("prompt_preset", 2) in state_sets

    # Toggling Show More retraces with expanded thresholds and regenerates text.
    hosted_show_more = True
    s.handle_event({"kind": "control", "control": "show_more", "value": True}, None)

    assert len(trace_calls) == 2
    assert trace_calls[1]["prompt"] == new_prompt
    assert trace_calls[1]["node_threshold"] == 0.78
    assert trace_calls[1]["edge_threshold"] == 0.93
    assert actions_run == ["reset", "generate", "reset", "generate"]
    assert progress_log == ["tracing", "ready", "tracing", "ready"]

    # Clicking Run when action:run is not yet on the hosted worker retraces, applies clamp, and regenerates text.
    s.handle_event({"kind": "run", "change": "action", "name": "run", "status": "error", "text": "unknown"}, None)
    assert len(trace_calls) == 3
    assert actions_run[-2:] == ["apply_clamp", "generate"]


def test_show_mode_error_keeps_window_alive(monkeypatch):
    run_mod = _load_demo_module("run")

    events_seen: list[str] = []

    class FakeApp:
        def __init__(self):
            self.alive = True
            self.stopped = False

        def poll(self):
            return None if self.alive else 0

    class FakeSession:
        def __init__(self, show: bool):
            self.args = type("Args", (), {"prompt": "ABCD", "model": "tiny-lm", "browser_capture": False})()
            self.show = show
            self.app = FakeApp()
            self.mcp = None
            self.started = True
            self.run_id = "circuit-trace"

        def show_error(self, msg: str):
            events_seen.append(f"error:{msg}")

        def keep_alive(self, _manifest):
            events_seen.append(f"keep_alive:alive={self.app.alive}")
            self.app.alive = False

    monkeypatch.setattr(run_mod, "build_instrument", lambda *_a, **_kw: (_ for _ in ()).throw(run_mod.Failure("boom")))
    fake = FakeSession(show=True)
    summary = run_mod.run_standard(fake)
    assert summary["error"] == "boom"
    assert events_seen == ["error:boom", "keep_alive:alive=True"]

    stopped_procs: list[tuple[object, dict]] = []
    monkeypatch.setattr(run_mod, "stop", lambda proc, **kw: stopped_procs.append((proc, kw)))
    s = run_mod.Session.__new__(run_mod.Session)
    s.mcp = None
    s.app = FakeApp()
    s.app_log = None
    s.started = False
    s.close(stop_app=False)
    assert stopped_procs == []
    assert s.app is not None
    s.close(stop_app=True)
    assert len(stopped_procs) == 1
    assert stopped_procs[0][1] == {"group": True}


@pytest.mark.skipif(sys.platform != "darwin" or not torch.backends.mps.is_available(), reason="requires macOS Metal")
def test_circuit_trace_tiny_lm_headless_mcp(tmp_path: Path):
    app_exe = ROOT / "cmake-build-release" / "caliper"
    mcp_exe = ROOT / "cmake-build-release" / "src" / "builder" / "mcp" / "caliper-mcp"
    if not app_exe.is_file() or not mcp_exe.is_file():
        pytest.skip("requires built caliper and caliper-mcp binaries")

    out_dir = tmp_path / "run"
    env = os.environ.copy()
    env["CALIPER_TEST_NO_WINDOW"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT)])

    proc = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py"), "--model", "tiny-lm", "--output", str(out_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["mode"] == "standard"
    assert result["counts"]["nodes"] > 0
    assert result["selected_node"] >= 0
    assert result["example_count"] > 0
    for key in ("before", "after"):
        png = Path(result["captures"][key])
        assert png.is_file() and png.stat().st_size > 1024
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
