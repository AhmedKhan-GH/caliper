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
        def wait(self, _timeout=0.05):
            return []

        def until(self, description, read, *, timeout=120.0):
            val = read()
            if val:
                return val
            raise AssertionError(f"Timed out in fake until: {description}")

        def complete(self, name: str, **kwargs):
            if name == "trace.run":
                trace_calls.append(kwargs)
                return {"seconds": 0.01}
            if name == "trace.status":
                return {
                    "state": "done" if trace_calls else "idle",
                    "prompt": trace_calls[-1]["prompt"] if trace_calls else run_mod.DEFAULT_PROMPT,
                }
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


def test_startup_waits_for_automatic_trace_and_show_mode_recovers_panels(monkeypatch):
    run_mod = _load_demo_module("run")

    trace_run_calls: list[dict] = []
    timeline: list[str] = []
    status_polls = 0

    def _default_spec(spec_id: str = "circuit-trace") -> dict:
        return {
            "applet": {"id": spec_id},
            "state": [{"id": "selected_node", "default": -1}, {"id": "activation_max", "default": 1.0}],
            "panel": [
                {"id": "graph", "primitive": "text", "options": {"text": "The trace is not available."}},
                {"id": "output", "primitive": "text", "options": {"text": "The trace is not available."}},
                {"id": "examples", "primitive": "text", "options": {"text": "No feature is selected."}},
            ],
            "control": [
                {"id": "clamp_status", "kind": "readout"},
                {"id": "clamp_status_value", "kind": "text"},
            ],
            "program": [],
        }

    class FakeMCP:
        def __init__(self, *, open_spec: bool = True):
            self.calls: list[tuple[str, dict]] = []
            self.open = open_spec
            self.spec = _default_spec()

        def body(self, cmd: str, /, **kwargs):
            self.calls.append((cmd, kwargs))
            timeline.append(cmd if cmd != "piece.show" else f"piece.show:{kwargs.get('name')}")
            if cmd == "spec.new":
                self.open = True
                self.spec = {
                    "applet": {"id": kwargs["id"]},
                    "state": [],
                    "panel": [],
                    "control": [],
                    "program": [],
                }
                return {}
            if cmd == "panel.create":
                self.spec["panel"].append(
                    {"id": kwargs["id"], "primitive": kwargs["primitive"], "options": dict(kwargs.get("options") or {})}
                )
                return {}
            if cmd == "panel.patch":
                for panel in self.spec["panel"]:
                    if panel["id"] == kwargs["id"]:
                        opts = dict(panel.get("options") or {})
                        opts.update((kwargs.get("patch") or {}).get("options") or {})
                        panel["options"] = opts
                return {}
            if cmd == "state.create":
                self.spec["state"].append({"id": kwargs["id"], "default": kwargs.get("default")})
                return {}
            if cmd == "piece.show":
                return {"from": "library", "piece": {"hash": "sha256:00"}}
            if cmd == "spec.set_text":
                text = kwargs.get("text", "")
                progs = []
                for pid, panel_id in (
                    ("attribution", "graph"),
                    ("probabilities", "output"),
                    ("example_tokens", "examples"),
                ):
                    if f'"{pid}"' in text:
                        progs.append({"id": pid, "panel": panel_id})
                        for panel in self.spec["panel"]:
                            if panel["id"] == panel_id:
                                panel["primitive"] = "scene"
                self.spec["program"] = progs
                return {}
            if cmd == "run.state":
                return {
                    "attached": True,
                    "params": [
                        {"name": "prompt", "value": run_mod.DEFAULT_PROMPT, "pending": None},
                        {"name": "selected_node", "value": -1, "pending": None},
                        {"name": "clamp", "value": 0.0, "pending": None},
                        {"name": "retrace", "value": False, "pending": None},
                        {"name": "show_more", "value": False, "pending": None},
                        {"name": "output.status", "value": "ready", "pending": None},
                    ],
                    "actions": [{"name": n} for n in ("run", "generate", "reset", "apply_clamp")],
                }
            if cmd == "run.list":
                return {"runs": [{"id": "circuit-trace", "state": "running"}]}
            if cmd == "program.status":
                return {"programs": []}
            if cmd == "spec.get":
                if not self.open:
                    return {"open": False}
                return {
                    "open": True,
                    "id": self.spec["applet"]["id"],
                    "json": json.loads(json.dumps(self.spec)),
                }
            return {}

    class FakeEvents:
        def __init__(self, *, pending_events: list[dict] | None = None):
            self.pending = list(pending_events or [])

        def wait(self, _timeout=0.05):
            return []

        def until(self, description, read, *, timeout=120.0):
            for _ in range(5):
                val = read()
                if val:
                    return val
            raise run_mod.Failure(f"Timed out: {description}")

        def complete(self, name: str, **kwargs):
            nonlocal status_polls
            timeline.append(name)
            if name == "trace.status":
                status_polls += 1
                if status_polls == 1:
                    return {
                        "state": "running",
                        "seconds": 0.0,
                        "inference": {"state": "tracing", "traced_prompt": None, "traced_show_more": None},
                    }
                return {
                    "state": "done",
                    "seconds": 1.25,
                    "inference": {
                        "state": "ready",
                        "traced_prompt": run_mod.DEFAULT_PROMPT,
                        "traced_show_more": False,
                        "selected_node": 2,
                    },
                }
            if name == "trace.run":
                trace_run_calls.append(kwargs)
                raise AssertionError("ensure_trace must reuse the completed automatic trace without calling trace.run")
            if name == "trace.read":
                return {
                    "counts": {"nodes": 3, "edges": 2, "tokens": 2, "layers": 2},
                    "nodes": [{"index": 2, "kind": "feature", "layer": 1, "feature": 7, "position": 1}],
                    "edges": [],
                    "next_node_offset": None,
                    "next_edge_offset": None,
                }
            if name == "feature.show":
                return {"maximum": 5.0}
            if name == "feature.examples":
                return {"examples": [{"example_id": 0}]}
            raise AssertionError(f"Unexpected complete({name})")

    def _make_session(mcp: FakeMCP, events: FakeEvents):
        s = run_mod.Session.__new__(run_mod.Session)
        s.args = type("Args", (), {"prompt": run_mod.DEFAULT_PROMPT, "model": "gemma-2-2b", "browser_capture": False, "acceptance": False})()
        s.show = True
        s.mcp = mcp
        s.events = events
        s.started = True
        s.run_id = "circuit-trace"
        s.current_prompt = run_mod.DEFAULT_PROMPT
        s.show_more = False
        s.progress = "ready"
        s.selected_node = -1
        s.instrument_built = False
        s.base_programs_installed = False
        s.examples_installed = False
        s.initial_ready = False
        s.error_shown = False
        s.last_error = None
        s.applet_saved = False
        s.load_seconds = 0.5
        s.hosted_params = {"prompt", "selected_node", "clamp", "retrace", "show_more", "output.status"}
        s.hosted_actions = {"run", "apply_clamp", "reset", "generate"}
        s.set_param = lambda *_a, **_kw: {}
        s.wait_programs_rendered = lambda *_a, **_kw: {}
        s.capture_native = lambda name, **_kw: Path(f"/nonexistent/{name}.png")
        return s

    class OneShotApp:
        def __init__(self, target):
            self.target = target
            self.polls = 0

        def poll(self):
            self.polls += 1
            if (
                getattr(self.target, "initial_ready", False)
                and not getattr(self.target, "error_shown", False)
                and self.polls >= 4
            ):
                return 0
            return None if self.polls <= 30 else 0

    def fake_build(sess, **_kw):
        sess.mcp.open = True
        sess.mcp.spec = _default_spec("circuit-trace")
        return None

    s = _make_session(FakeMCP(open_spec=False), FakeEvents())
    s.app = OneShotApp(s)
    monkeypatch.setattr(run_mod, "build_instrument", fake_build)

    summary = run_mod.run_standard(s)
    assert "error" not in summary
    assert trace_run_calls == []
    assert status_polls >= 2
    assert s.base_programs_installed is True
    assert s.examples_installed is True
    assert s.initial_ready is True
    # All three programs must be installed and finalized before the first trace status or run call.
    first_trace_idx = timeline.index("trace.status")
    for piece_event in ("piece.show:attribution_graph", "piece.show:logit_bars", "piece.show:token_strip"):
        assert timeline.index(piece_event) < first_trace_idx
    assert timeline.index("spec.set_text") < first_trace_idx
    assert timeline.index("applet.save") < first_trace_idx
    assert timeline.index("applet.open") < first_trace_idx

    # Verify ensure_trace handles an interpret.busy race by waiting and reusing the completed trace.
    busy_calls = 0

    class BusyOnceEvents(FakeEvents):
        def complete(self, name: str, **kwargs):
            nonlocal busy_calls
            if name == "trace.status":
                if busy_calls == 0:
                    return {
                        "state": "idle",
                        "seconds": 0.0,
                        "inference": {"state": "ready", "traced_prompt": None, "traced_show_more": None},
                    }
                return {
                    "state": "done",
                    "seconds": 0.8,
                    "inference": {"state": "ready", "traced_prompt": run_mod.DEFAULT_PROMPT, "traced_show_more": False},
                }
            if name == "trace.run":
                busy_calls += 1
                if busy_calls == 1:
                    raise run_mod.Failure('trace.run: {"error":{"code":"interpret.busy","message":"A trace already waits for completion."}}')
                raise AssertionError("trace.run must not be called a second time once the trace completed")
            return super().complete(name, **kwargs)

    s_busy = _make_session(FakeMCP(), BusyOnceEvents())
    s_busy.hosted_actions = {"generate", "reset", "apply_clamp"}
    res = s_busy.ensure_trace(run_mod.DEFAULT_PROMPT, show_more=False)
    assert res["state"] == "done"
    assert busy_calls == 1

    # Verify keep_alive recovers from circuit-trace-error fallback when build_instrument fails transiently twice.
    s2 = _make_session(FakeMCP(open_spec=False), FakeEvents())
    s2.app = OneShotApp(s2)

    attempts = 0

    def flaky_build(sess, **kw):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise run_mod.Failure("transient build failure")
        return fake_build(sess, **kw)

    monkeypatch.setattr(run_mod, "build_instrument", flaky_build)
    monkeypatch.setattr(run_mod.time, "monotonic", lambda: 1e9)
    summary2 = run_mod.run_standard(s2)
    assert summary2["error"] == "transient build failure"
    assert attempts >= 3
    assert any(cmd == "spec.new" and kw.get("id") == "circuit-trace-error" for cmd, kw in s2.mcp.calls)
    assert s2.mcp.spec["applet"]["id"] == "circuit-trace"
    assert s2.instrument_built is True
    assert s2.base_programs_installed is True
    assert s2.examples_installed is True
    assert s2.initial_ready is True
    assert s2.error_shown is False
    assert {p["id"] for p in s2.mcp.spec["program"]} == {"attribution", "probabilities", "example_tokens"}

    # Verify show_error writes state into unbound text panels when program installation fails,
    # unrelated events do not clear the error, and keep_alive retries until panels are bound.
    install_attempts = 0
    orig_install = run_mod.install_programs

    def flaky_install(sess, **kw):
        nonlocal install_attempts
        install_attempts += 1
        if install_attempts <= 2:
            raise run_mod.Failure("piece library temporarily unavailable")
        return orig_install(sess, **kw)

    monkeypatch.setattr(run_mod, "build_instrument", fake_build)
    monkeypatch.setattr(run_mod, "install_programs", flaky_install)
    s3_mcp = FakeMCP(open_spec=False)
    s3 = _make_session(s3_mcp, FakeEvents(pending_events=[{"kind": "spec", "event": "do"}]))
    s3.app = OneShotApp(s3)
    # Also verify an unrelated event does not claim handled or wipe error state before recovery.
    s3.error_shown = True
    assert s3.handle_event({"kind": "spec", "event": "do"}, None) is False
    assert s3.error_shown is True
    s3.error_shown = False

    summary3 = run_mod.run_standard(s3)
    assert summary3["error"] == "piece library temporarily unavailable"
    patched_panels = {kw["id"]: kw["patch"]["options"]["text"] for cmd, kw in s3_mcp.calls if cmd == "panel.patch"}
    assert set(patched_panels) == {"graph", "output", "examples"}
    for text in patched_panels.values():
        assert "piece library temporarily unavailable" in text
        assert "Retrying" in text
    assert install_attempts >= 3
    assert s3.base_programs_installed is True
    assert s3.examples_installed is True
    assert s3.initial_ready is True
    assert s3.error_shown is False
    assert {p["id"] for p in s3_mcp.spec["program"]} == {"attribution", "probabilities", "example_tokens"}


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
