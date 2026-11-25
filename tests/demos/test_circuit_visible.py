"""Visible demo failures preserve the app and browser assets stay explicit."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import threading
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import jsonschema
import pytest

from test_circuit_trace import driver, ROOT
from _mcp import Failure
from _visible import ASSETS, configure_assets, check_assets, hold_after_error, show_error


@pytest.mark.parametrize("explicit", [False, True])
def test_browser_assets_use_app_environment_variable(tmp_path, explicit):
    env = {"CALIPER_WASM_DIR": str(tmp_path / "parity")}
    if explicit:
        env["CALIPER_WEB_DIR"] = str(tmp_path / "browser")
    configure_assets(env)
    assert env["CALIPER_WEB_DIR"] == str(tmp_path / ("browser" if explicit else "parity"))


@pytest.mark.parametrize("missing", [None, *ASSETS])
def test_actual_browser_routes_require_all_modules(tmp_path, missing):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path.split("?")[0])
            name = self.path.split("?")[0].rsplit("/", 1)[-1]
            self.send_response(404 if name == missing else 200)
            self.end_headers()
            self.wfile.write(b"\x00asm\x01\x00\x00\x00" if name.endswith(".wasm") else b"export const ready = true;")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/api.json").write_text(json.dumps({
        "spec_url": f"http://127.0.0.1:{server.server_port}/v0/spec?token=fixture-token"}))
    try:
        if missing:
            with pytest.raises(Failure, match="browser asset.*unavailable") as failure:
                check_assets(SimpleNamespace(out=tmp_path))
            assert "fixture-token" not in str(failure.value)
        else:
            evidence = check_assets(SimpleNamespace(out=tmp_path))
            assert set(evidence) == set(ASSETS)
            assert all(item["status"] == 200 and item["bytes"] for item in evidence.values())
            assert requests == ["/v0/spec/" + name for name in ASSETS]
            assert "fixture-token" not in json.dumps(evidence)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("existing", [False, True])
def test_error_panel_is_valid_before_and_after_instrument_creation(tmp_path, existing):
    spec = {"caliper_spec": 1, "applet": {"id": "circuit-trace", "title": "Circuit Trace"}}
    if existing:
        spec.update(panel=[{"id": "graph", "primitive": "text", "options": {"text": "Existing Graph"}}],
                    control=[{"id": "prompt", "kind": "text", "title": "Prompt", "bind": "param:prompt"},
                             {"id": "clamp_status", "kind": "readout", "options": {"format": "Ready"}}],
                    layout={"split": "vertical", "children": [
                        {"split": "vertical", "children": ["prompt", "clamp_status"]}, "graph"]})
    original = deepcopy(spec)
    changed = []
    def body(method, **args):
        if method == "spec.get":
            return {"json": spec}
        if method == "applet.save":
            return {}
        assert method == "spec.set_text"
        changed.append(tomllib.loads(args["text"]))
    show_error(SimpleNamespace(out=tmp_path, mcp=SimpleNamespace(body=body)), Failure("Dictionary {request} failed."))
    assert not (tmp_path / "demo-error-display.json").exists()
    assert spec == original
    document = changed[0]
    schema = json.loads((ROOT / "src/builder/spec/schema/caliper-spec-v1.schema.json").read_text())
    jsonschema.validate(document, schema)
    error = next(p for p in document["panel"] if p["id"] == "demo_error")
    assert error["options"]["text"] == "Dictionary {request} failed."
    if existing:
        assert document["panel"][0] == spec["panel"][0]
        assert document["layout"]["children"][0]["children"] == ["prompt", "demo_error"]
    else:
        assert document["layout"] == {"panel": "demo_error"}


@pytest.mark.parametrize("bridge", [False, True])
def test_driver_error_holds_owned_app_even_without_a_working_bridge(tmp_path, bridge):
    state = {"live": True, "waits": 0}
    def wait(timeout):
        assert timeout == 1
        state["waits"] += 1
        if state["waits"] == 1:
            raise subprocess.TimeoutExpired("owned-app", timeout)
        state["live"] = False
    session = SimpleNamespace(out=tmp_path, mcp=Mock(), live=lambda: state["live"], app=SimpleNamespace(wait=wait))
    session.mcp.body.side_effect = Failure("Command transport failed.")
    if bridge:
        session.bridge = SimpleNamespace(keep_alive=Mock(side_effect=Failure("Bridge failed.")))
    hold_after_error(session, Failure("Dictionary failed."))
    assert state["waits"] == 2
    assert (tmp_path / "demo-error-display.json").exists()
    if bridge:
        session.bridge.keep_alive.assert_called_once()


@pytest.mark.parametrize("show", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="The owned child uses a POSIX executable.")
def test_live_startup_failure_retains_process_before_bridge_exists(tmp_path, monkeypatch, show):
    import psutil
    child = tmp_path / "owned-child"
    child.write_text(f"#!{sys.executable}\n" + "\n".join([
        "import json, os, pathlib, signal",
        "folder = pathlib.Path(os.environ['CALIPER_DATA_DIR'])",
        "(folder / 'probe.json').write_text(json.dumps({'pid': os.getpid()}))",
        "signal.pause()",
    ]) + "\n")
    child.chmod(0o700)
    actual_wait = driver.wait_file
    def fail_ready(path, **kwargs):
        actual_wait(path.parent / "probe.json", timeout=5, **kwargs)
        raise Failure("Ready file failed before the bridge.")
    monkeypatch.setattr(driver, "executable", lambda *args: child)
    monkeypatch.setattr(driver, "wait_file", fail_ready)
    args = SimpleNamespace(model="tiny-lm", dictionary=str(tmp_path), prompt="Test", demo=True)
    with pytest.raises(driver.StartupFailure) as raised:
        driver.Session(args, tmp_path / "session", show=show, interactive=not show)
    session = raised.value.session
    pid = json.loads((session.out / "data/probe.json").read_text())["pid"]
    try:
        assert session.live() and psutil.pid_exists(pid)
        assert not hasattr(session, "bridge")
        assert not session.app_log.closed
    finally:
        session.close()
    assert not psutil.pid_exists(pid) and session.app_log.closed


@pytest.mark.parametrize("exits", [False, True])
def test_close_response_requires_actual_owned_process_exit(exits):
    session = SimpleNamespace(live=lambda: True, app=Mock(), events=Mock())
    failure = Failure("mcp.app_busy: the app is closing")
    session.events.wait.side_effect = failure
    if not exits:
        session.app.wait.side_effect = subprocess.TimeoutExpired("owned-app", 5)
        with pytest.raises(Failure, match="app is closing"):
            driver.Bridge(session).keep_alive()
    else:
        driver.Bridge(session).keep_alive()
    session.app.wait.assert_called_once_with(timeout=5)
