"""caliper.widget(): the anywidget wiring, and the widget drawn in a real
JupyterLab driven by headless Chromium.

Runs without torch or the native extensions. The JupyterLab test needs
jupyterlab, ipykernel, playwright (with its Chromium) and Pillow, and is
skipped without them. It keeps its screenshots in $CALIPER_TEST_ARTIFACTS
when that is set."""
import importlib.util
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

import pytest

import caliper
from caliper import _assets

REPO = Path(__file__).resolve().parents[2]
TRAITS = {"src", "view", "kind", "height", "data_view", "selection"}

needs_anywidget = pytest.mark.skipif(importlib.util.find_spec("anywidget") is None,
                                     reason="anywidget is not installed (caliper[notebook])")


def run_python(code: str, **env) -> subprocess.CompletedProcess:
    full = {**os.environ, "PYTHONPATH": str(REPO / "python"), **env}
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True)


def test_import_loads_no_optional_dependency():
    code = ("import sys, caliper; print(','.join(m for m in ('anywidget', 'ipywidgets', "
            "'traitlets', 'websockets', 'torch', 'caliper._native', 'caliper._widget') "
            "if m in sys.modules))")
    r = run_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_missing_anywidget_names_the_extra():
    code = ("import sys; sys.modules['anywidget'] = None\n"
            "import caliper\n"
            "try:\n    caliper.widget('loss')\n"
            "except ImportError as e:\n    print(e)\n")
    r = run_python(code)
    assert r.returncode == 0, r.stderr
    assert "caliper[notebook]" in r.stdout


@pytest.fixture
def server(monkeypatch):
    """A stand-in for caliper.serve()'s server, through the seam caliper._server."""
    srv = types.SimpleNamespace(port=40123, token="tok/en")
    monkeypatch.setattr(caliper, "_server", srv, raising=False)
    return srv


@needs_anywidget
def test_widget_traits(server):
    import anywidget
    w = caliper.widget("loss")
    assert set(w.keys) - set(anywidget.AnyWidget().keys) == TRAITS
    assert w.src == "ws://127.0.0.1:40123/v3/stream?token=tok%2Fen"
    assert (w.view, w.kind, w.height, w.data_view, w.selection) == ("loss", None, 320, {}, {})
    w = caliper.widget("fc1.weight", "heatmap", height=200)
    assert (w.view, w.kind, w.height) == ("fc1.weight", "heatmap", 200)
    w.data_view = {"index": {"head": 3}}
    assert w.get_state()["data_view"] == {"index": {"head": 3}}


@needs_anywidget
def test_widget_esm_is_the_packaged_module(server):
    w = caliper.widget("loss")
    assert w._esm == _assets.read_asset("caliper-widget.js").decode("utf-8")
    assert w.get_state()["_esm"] == w._esm


@needs_anywidget
def test_stream_url_seam(monkeypatch):
    from caliper._widget import _current_stream_url
    monkeypatch.setattr(caliper, "_server", None, raising=False)
    assert _current_stream_url() is None
    srv = types.SimpleNamespace(port=1, token="a", stream_url="ws://127.0.0.1:2/v3/stream?token=b")
    monkeypatch.setattr(caliper, "_server", srv)
    assert _current_stream_url() == "ws://127.0.0.1:2/v3/stream?token=b"
    srv.closed = True
    assert _current_stream_url() is None


@needs_anywidget
def test_widget_starts_serve_when_no_server_runs(monkeypatch):
    monkeypatch.setattr(caliper, "_server", None, raising=False)
    calls = []

    def serve(port=0):
        calls.append(port)
        caliper._server = types.SimpleNamespace(port=40999, token="t")
        return caliper._server
    monkeypatch.setattr(caliper, "serve", serve, raising=False)
    w = caliper.widget("loss")
    assert calls == [0]
    assert w.src == "ws://127.0.0.1:40999/v3/stream?token=t"
    caliper.widget("loss")
    assert calls == [0]      # the running server is reused


@needs_anywidget
def test_widget_without_a_server(monkeypatch):
    monkeypatch.setattr(caliper, "_server", None, raising=False)
    monkeypatch.delattr(caliper, "serve", raising=False)
    with pytest.raises(RuntimeError, match=r"caliper\.serve\(\)"):
        caliper.widget("loss")
    monkeypatch.setattr(caliper, "serve", lambda port=0: None, raising=False)
    with pytest.raises(RuntimeError, match="no stream server is running"):
        caliper.widget("loss")


@needs_anywidget
def test_widget_arguments_are_checked(server):
    import traitlets
    for bad in ("", None, 3):
        with pytest.raises(ValueError):
            caliper.widget(bad)
    with pytest.raises(ValueError, match="kind"):
        caliper.widget("loss", "lines")
    with pytest.raises(traitlets.TraitError):
        caliper.widget("loss", height=0)
    w = caliper.widget("loss")
    with pytest.raises(traitlets.TraitError):
        w.kind = "lines"


@needs_anywidget
def test_on_select(server):
    w = caliper.widget("points", "points")
    got, other = [], []

    @w.on_select
    def keep(sel):
        got.append(sel)
    w.on_select(keep)                             # registered once
    w.on_select(other.append)
    sel = {"view": "points", "type": "select", "detail": {"indices": [1, 5]}}
    w.selection = sel                             # as the browser writes it
    assert got == [sel] and other == [sel] and w.selection == sel
    assert w.on_select(keep, remove=True) is keep
    w.selection = {"view": "points", "type": "pick", "detail": {"index": 2}}
    assert got == [sel] and len(other) == 2


@needs_anywidget
def test_repr_hides_the_token(server):
    w = caliper.widget("loss", "series")
    text = repr(w)
    assert "tok%2Fen" not in text and "tok/en" not in text
    assert "'loss'" in text and "series" in text and "token=..." in text
    assert "127.0.0.1:40123" in text


@needs_anywidget
def test_missing_widget_module_names_the_build_command(tmp_path):
    code = ("import types, caliper\n"
            "caliper._server = types.SimpleNamespace(port=1, token='t')\n"
            "try:\n    caliper.widget('loss')\n"
            "except FileNotFoundError as e:\n    print(e)\n")
    r = run_python(code, CALIPER_WEB_DIR=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "caliper-widget.js" in r.stdout and "scripts/build_web.py" in r.stdout


# ---------------------------------------------------------------------------
# the widget in a real JupyterLab, driven by headless Chromium
# ---------------------------------------------------------------------------
# The first cell runs a mock stream server in the kernel: it refuses a wrong
# token and sends a binary frame to each socket 30 times a second. It hands its
# URL to the widget through the seam caliper._server.
CELL_STREAM = r'''
import asyncio, http, json, os, secrets, threading, types
from pathlib import Path
from websockets.asyncio.server import serve
import caliper

OUT = Path(os.environ["CALIPER_E2E_OUT"])
TOKEN = secrets.token_urlsafe(16)
seen = {"origins": [], "refused": 0}

def check(connection, request):
    if request.path != "/v3/stream?token=" + TOKEN:
        seen["refused"] += 1
        return connection.respond(http.HTTPStatus.FORBIDDEN, "wrong token\n")

async def stream(ws):
    seen["origins"].append(ws.request.headers.get("Origin"))
    (OUT / "stream.json").write_text(json.dumps(seen))
    n = 0
    while True:
        n += 1
        await ws.send(n.to_bytes(8, "little") * 8)
        await asyncio.sleep(1 / 30)

ready = threading.Event()

def run():
    async def main():
        async with serve(stream, "127.0.0.1", 0, process_request=check) as server:
            port = server.sockets[0].getsockname()[1]
            caliper._server = types.SimpleNamespace(
                stream_url=f"ws://127.0.0.1:{port}/v3/stream?token={TOKEN}")
            ready.set()
            await server.serve_forever()
    asyncio.run(main())

threading.Thread(target=run, daemon=True).start()
assert ready.wait(10)
'''

CELL_WIDGET = r'''
w = caliper.widget("loss")
w.on_select(lambda sel: (OUT / "selection.json").write_text(json.dumps(sel)))
w.observe(lambda change: (OUT / "data_view.json").write_text(json.dumps(change["new"])),
          names="data_view")
w
'''


def notebook(*cells: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "id": f"c{i}", "metadata": {},
                   "outputs": [], "source": c.strip()} for i, c in enumerate(cells)],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_call(url: str, token: str, method: str = "GET", timeout: float = 5.0):
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"token {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def wait_for(fn, timeout: float, what: str):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            v = fn()
            if v:
                return v
        except OSError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def processes_mentioning(text: str) -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines()
            if text in line and str(os.getpid()) != line.split()[0]]


def drawn_and_changed(png1: bytes, png2: bytes) -> tuple[int, bool]:
    import io
    from PIL import Image, ImageChops
    a = Image.open(io.BytesIO(png1)).convert("RGB")
    b = Image.open(io.BytesIO(png2)).convert("RGB")
    colors = len(a.getcolors(maxcolors=1 << 20) or [])
    return colors, ImageChops.difference(a, b).getbbox() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="the test manages POSIX process groups")
@needs_anywidget
def test_widget_draws_in_jupyterlab(tmp_path):
    for mod in ("jupyterlab", "ipykernel", "websockets", "PIL"):
        pytest.importorskip(mod)
    sync_api = pytest.importorskip("playwright.sync_api")
    shots = Path(os.environ.get("CALIPER_TEST_ARTIFACTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    out, nbdir, jdir = tmp_path / "out", tmp_path / "notebooks", tmp_path / "jupyter"
    for d in (out, nbdir, jdir):
        d.mkdir()
    (nbdir / "widget.ipynb").write_text(json.dumps(notebook(CELL_STREAM, CELL_WIDGET)))
    # No news or update checks (they would fetch from the network).
    notify = jdir / "lab-settings" / "@jupyterlab" / "apputils-extension"
    notify.mkdir(parents=True)
    (notify / "notification.jupyterlab-settings").write_text(
        json.dumps({"fetchNews": "false", "checkForUpdates": False}))

    # Every Jupyter directory inside tmp_path; the kernel imports this checkout.
    env = {**os.environ,
           "JUPYTER_CONFIG_DIR": str(jdir / "config"), "JUPYTER_DATA_DIR": str(jdir / "data"),
           "JUPYTER_RUNTIME_DIR": str(jdir / "runtime"), "IPYTHONDIR": str(jdir / "ipython"),
           "JUPYTERLAB_SETTINGS_DIR": str(jdir / "lab-settings"),
           "JUPYTERLAB_WORKSPACES_DIR": str(jdir / "lab-workspaces"),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "python"),
                                                       os.environ.get("PYTHONPATH")])),
           "CALIPER_E2E_OUT": str(out)}
    port, token = free_port(), secrets.token_hex(16)
    base = f"http://127.0.0.1:{port}"
    # --no-browser: the server opens no window; only headless Chromium visits it.
    cmd = [sys.executable, "-m", "jupyterlab", "--no-browser", "--ip=127.0.0.1",
           f"--port={port}", "--ServerApp.port_retries=0", f"--IdentityProvider.token={token}",
           f"--ServerApp.root_dir={nbdir}", "--LabApp.expose_app_in_browser=True",
           "--LabApp.news_url=None", "--LabApp.extension_manager=readonly",
           "--LabApp.check_for_updates_class=jupyterlab.NeverCheckForUpdate"]
    log = open(tmp_path / "jupyter.log", "wb")
    proc = subprocess.Popen(cmd, cwd=nbdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        wait_for(lambda: http_call(f"{base}/api/status", token) == 200, 90, "JupyterLab to start")
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except sync_api.Error as e:
                pytest.skip(f"Playwright's Chromium is not available: {e}")
            try:
                page = browser.new_page(viewport={"width": 1100, "height": 900},
                                        device_scale_factor=1)
                page.goto(f"{base}/lab/tree/widget.ipynb?token={token}")
                page.wait_for_function(
                    "() => window.jupyterapp?.shell.currentWidget?.sessionContext"
                    "?.session?.kernel?.status === 'idle'", timeout=90_000)
                page.evaluate("() => window.jupyterapp.commands.execute('notebook:run-all-cells')")

                view = page.locator(".jp-OutputArea-output caliper-view")
                view.wait_for(state="attached", timeout=60_000)
                cell = page.locator(".jp-CodeCell").filter(has=view)
                canvas = view.locator("canvas")
                canvas.wait_for(state="visible", timeout=30_000)
                stream = json.loads(wait_for(lambda: (out / "stream.json").read_text(), 20,
                                             "the widget to open the stream socket"))
                page.wait_for_timeout(1500)
                cell.scroll_into_view_if_needed()
                first = canvas.screenshot()
                cell.screenshot(path=shots / "widget-cell-1.png")
                page.screenshot(path=shots / "widget-page-1.png")
                page.wait_for_timeout(800)
                second = canvas.screenshot()
                cell.screenshot(path=shots / "widget-cell-2.png")
                page.screenshot(path=shots / "widget-page-2.png")

                # Drawn, inside the cell's output area, at the default height, changing.
                colors, changed = drawn_and_changed(first, second)
                assert colors >= 3, f"the canvas looks blank ({colors} colours)"
                assert changed, "the canvas did not change between the screenshots"
                c = canvas.bounding_box()
                o = cell.locator(".jp-Cell-outputArea").bounding_box()
                assert o["x"] <= c["x"] and c["x"] + c["width"] <= o["x"] + o["width"] + 1
                assert o["y"] <= c["y"] and c["y"] + c["height"] <= o["y"] + o["height"] + 1
                assert abs(c["height"] - 320) <= 2 and c["width"] > 200

                # The page is Jupyter's origin; the stream socket is another port.
                assert stream["origins"] == [base]
                assert stream["refused"] == 0

                # Events from the element come back to Python as traits.
                sel = {"view": "loss", "type": "select", "detail": {"index": [3, 9]}}
                dv = {"source": "loss", "index": {"step": [0, 100]}}
                view.evaluate("(el, d) => el.dispatchEvent(new CustomEvent('caliper-select',"
                              " {detail: d, bubbles: true, composed: true}))", sel)
                view.evaluate("(el, d) => el.dispatchEvent(new CustomEvent('caliper-dataview',"
                              " {detail: d, bubbles: true, composed: true}))", dv)
                got = wait_for(lambda: (out / "selection.json").read_text(), 20, "the selection")
                assert json.loads(got) == sel
                got = wait_for(lambda: (out / "data_view.json").read_text(), 20, "the data view")
                assert json.loads(got) == dv
            finally:
                browser.close()
    finally:
        try:
            http_call(f"{base}/api/shutdown", token, method="POST")
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
        # Kernels run in their own sessions; their command lines name tmp_path.
        end = time.monotonic() + 15
        while processes_mentioning(str(tmp_path)) and time.monotonic() < end:
            time.sleep(0.2)
        left = processes_mentioning(str(tmp_path))
        for line in left:
            os.kill(int(line.split()[0]), signal.SIGKILL)
        assert not left, f"processes left running: {left}"
