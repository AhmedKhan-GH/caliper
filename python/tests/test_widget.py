"""caliper.widget(): the anywidget wiring, and the widget drawn in a real
JupyterLab driven by headless Chromium.

Runs without torch or the native extensions. The JupyterLab test needs
jupyterlab, ipykernel, playwright (with its Chromium) and Pillow, and is
skipped without them. It keeps its screenshots in $CALIPER_TEST_ARTIFACTS
when that is set."""
import importlib.util
import json
import os
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
# The widget in JupyterLab under headless Chromium, through the harness in
# tests/headless: the first cell starts caliper.serve() and a thread that shows
# a 4-D tensor with named dims 30 times a second (a thread, so the kernel stays
# idle and traits can travel while the view updates); the second shows it with
# caliper.widget().
CELL_PRODUCER = r"""
import json, os, threading, time
from pathlib import Path
import torch
import caliper

OUT = Path(os.environ["CALIPER_E2E_OUT"])
server = caliper.serve()
g = torch.Generator().manual_seed(0)
base = torch.rand(2, 3, 24, 32, generator=g)

def produce():
    step = 0
    while True:
        step += 1
        x = base + 0.5 * torch.sin(torch.tensor(step / 10.0)) * torch.rand(2, 3, 24, 32)
        caliper.show(x, "heat", dims=("layer", "head", "query", "key"),
                     data_view={"index": {"layer": 0, "head": 0}})
        time.sleep(1 / 30)

threading.Thread(target=produce, daemon=True).start()
"""

CELL_WIDGET = r"""
w = caliper.widget("heat")
w.on_select(lambda sel: (OUT / "selection.json").write_text(json.dumps(sel)))
w.observe(lambda change: (OUT / "data_view.json").write_text(json.dumps(change["new"])),
          names="data_view")
w
"""

STATE_JS = """() => { const v = __caliperHarness.deepQueryAll('caliper-view')[0];
                     return v ? v.debugState() : null; }"""


def notebook(*cells: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "id": f"c{i}", "metadata": {},
                   "outputs": [], "source": c.strip()} for i, c in enumerate(cells)],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


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


def changed(png1: bytes, png2: bytes) -> bool:
    import io
    from PIL import Image, ImageChops
    a = Image.open(io.BytesIO(png1)).convert("RGB")
    b = Image.open(io.BytesIO(png2)).convert("RGB")
    return ImageChops.difference(a, b).getbbox() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="the harness manages POSIX process groups")
@needs_anywidget
@pytest.mark.parametrize("mode", ["metal", "canvas2d"])
def test_widget_draws_in_jupyterlab(tmp_path, mode):
    """The widget draws a real view of caliper.serve() inside its cell, keeps
    updating, takes the data view chosen in its picker or set from Python, and
    reports selections and data views back to Python. "metal" draws with the
    WebAssembly views on WebGPU, "canvas2d" with the element's own views."""
    for mod in ("jupyterlab", "ipykernel", "websockets", "PIL", "torch", "playwright"):
        pytest.importorskip(mod)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tests.headless.notebook import notebook_session
    from tests.headless.screenshot import png_stats
    shots = Path(os.environ.get("CALIPER_TEST_ARTIFACTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    out.mkdir()
    path = tmp_path / "widget.ipynb"
    path.write_text(json.dumps(notebook(CELL_PRODUCER, CELL_WIDGET)))
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "python"),
                                                       os.environ.get("PYTHONPATH")])),
           "CALIPER_E2E_OUT": str(out), "CALIPER_SESSION": "0"}
    with notebook_session(path, env=env, mode=mode, viewport=(1100, 900)) as nb:
        nb.wait("() => { const v = __caliperHarness.deepQueryAll('caliper-view')[0];"
                " return !!v && v.debugState().status === 'connected'; }", timeout=90)
        state = nb.page.evaluate(STATE_JS)
        assert state["renderer"] == ("wasm" if mode == "metal" else "canvas2d"), state
        view = nb.page.locator(".jp-OutputArea-output caliper-view")
        area = view.locator("canvas:visible").first
        cell = nb.page.locator(".jp-CodeCell").filter(has=view)
        cell.scroll_into_view_if_needed()
        nb.page.wait_for_timeout(500)
        first = area.screenshot()
        nb.page.wait_for_timeout(800)
        second = area.screenshot()
        cell.screenshot(path=shots / f"widget-{mode}-cell.png")
        assert png_stats(first).nonblank(), png_stats(first).to_dict()
        assert changed(first, second), "the view did not change between the screenshots"
        c = area.bounding_box()
        o = cell.locator(".jp-Cell-outputArea").bounding_box()
        assert o["x"] <= c["x"] and c["x"] + c["width"] <= o["x"] + o["width"] + 1
        assert o["y"] <= c["y"] and c["y"] + c["height"] <= o["y"] + o["height"] + 1
        v = view.bounding_box()                               # height= is the whole element
        assert abs(v["height"] - 320) <= 2 and c["height"] > 200 and c["width"] > 200

        # The picker: layer 1, head 2; the trait reaches Python, the view follows.
        view.locator('select[data-dim="layer"]').select_option("i:1")
        view.locator('select[data-dim="head"]').select_option("i:2")
        want = {"layer": 1, "head": 2}
        got = wait_for(lambda: json.loads((out / "data_view.json").read_text() or "{}")
                       .get("index") == want and (out / "data_view.json").read_text(), 20,
                       "the picked data view in Python")
        assert json.loads(got)["source"] == "heat"
        nb.wait("() => { const s = (" + STATE_JS + ")();"
                " return s.dataView.index.layer === 1 && s.dataView.index.head === 2; }",
                timeout=20)

        # Set from Python: the widget requests it, the picker shows it.
        nb.execute('w.data_view = {"index": {"layer": 0, "head": 1}}')
        nb.wait("() => { const s = (" + STATE_JS + ")();"
                " return s.dataView.index.layer === 0 && s.dataView.index.head === 1; }",
                timeout=20)
        assert view.locator('select[data-dim="head"]').input_value() == "i:1"
        stats = json.loads(nb.execute("print(json.dumps(server.stats()))"))
        assert stats["views"]["heat"]["groups"] >= 1

        # A selection reported by the element reaches Python.
        sel = {"view": "heat", "type": "select", "detail": {"rows": [3, 9], "cols": [0, 4]}}
        view.evaluate("(el, d) => el.dispatchEvent(new CustomEvent('caliper-select',"
                      " {detail: d, bubbles: true, composed: true}))", sel)
        got = wait_for(lambda: (out / "selection.json").read_text(), 20, "the selection")
        assert json.loads(got) == sel
        assert json.loads(nb.execute("print(json.dumps(w.selection))")) == sel
