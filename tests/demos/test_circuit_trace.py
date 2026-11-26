"""Tests for the simplified Circuit Trace MCP demo helpers and tiny-lm E2E flow."""
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
