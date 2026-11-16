"""Self-tests of the headless harness.

Run from the repository root with `python -m pytest tests/headless`. Screenshots
go to `$CALIPER_HEADLESS_DIR/shots` (see README.md), never into the repository.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from .browser import HeadedBrowserRefused, launch, probe_webgpu, work_dir
from .notebook import (
    NotebookCellError,
    _alive,
    _children,
    notebook_session,
    run_notebook,
    screenshot_notebook,
)
from .screenshot import png_difference, png_stats, screenshot_element, screenshot_page

HERE = Path(__file__).parent
SELFTEST = HERE / "mock_widget" / "selftest.ipynb"
CANVAS = ".mock-view canvas"  # inside the widget's open shadow root
RECEIVED = "() => Number(__caliperHarness.deepQueryAll('.mock-view')[0].dataset.received || 0)"

CANVAS_2D_PAGE = """<!doctype html>
<body style="margin:0;background:#181715">
<canvas id="c" width="320" height="200"></canvas>
<script>
  const ctx = document.getElementById('c').getContext('2d');
  ctx.fillStyle = '#272622'; ctx.fillRect(0, 0, 320, 200);
  for (let i = 0; i < 16; i++) {
    ctx.fillStyle = `rgb(${120 + i * 8}, ${90 + i * 5}, 60)`;
    ctx.fillRect(10 + i * 18, 190 - i * 11, 14, i * 11);
  }
  window.drawn = true;
</script>"""

WEBGPU_PAGE = """<!doctype html>
<body style="margin:0;background:#181715">
<canvas id="c" width="256" height="256"></canvas>
<script type="module">
  window.state = 'start';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const ctx = document.getElementById('c').getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({device, format, alphaMode: 'opaque'});
    const module = device.createShaderModule({code: `
      @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        var p = array<vec2f, 3>(vec2f(0.0, 0.7), vec2f(-0.7, -0.7), vec2f(0.7, -0.7));
        return vec4f(p[i], 0.0, 1.0);
      }
      @fragment fn fs() -> @location(0) vec4f { return vec4f(0.7, 0.55, 0.35, 1.0); }`});
    const pipeline = device.createRenderPipeline({layout: 'auto',
      vertex: {module, entryPoint: 'vs'},
      fragment: {module, entryPoint: 'fs', targets: [{format}]}});
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({colorAttachments: [{
      view: ctx.getCurrentTexture().createView(), clearValue: {r: 0.15, g: 0.15, b: 0.13, a: 1},
      loadOp: 'clear', storeOp: 'store'}]});
    pass.setPipeline(pipeline); pass.draw(3); pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    window.state = 'drawn';
  } catch (e) { window.state = 'error: ' + e; }
</script>"""


@pytest.fixture(scope="session")
def shots() -> Path:
    path = work_dir() / "shots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def page_dir() -> Path:
    """A fresh folder for test pages and notebooks, inside the work directory."""
    path = Path(tempfile.mkdtemp(prefix="pages-", dir=work_dir()))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_headed_browser_is_refused():
    with pytest.raises(HeadedBrowserRefused):
        with launch(headless=False):
            pass
    with pytest.raises(HeadedBrowserRefused):
        with launch(extra_args=["--headless=false"]):
            pass


@pytest.mark.parametrize("mode,software", [("metal", False), ("swiftshader", True)])
def test_probe_webgpu_finds_an_adapter(mode, software):
    result = probe_webgpu(mode)
    assert result["mode"] == mode
    assert result["adapter"] and result["device"], result
    assert result["software"] is software, result


def test_probe_webgpu_canvas2d_mode_hides_webgpu():
    result = probe_webgpu("canvas2d")
    assert result["has_gpu"] is False and result["adapter"] is False, result


def test_static_canvas_screenshot_is_nonblank(page_dir, shots):
    (page_dir / "canvas2d.html").write_text(CANVAS_2D_PAGE)
    page = str(page_dir / "canvas2d.html")
    stats = screenshot_page(page, shots / "static-page.png", wait_for="() => window.drawn === true",
                            nonblank="#c", viewport=(400, 240))
    assert stats.nonblank(), stats
    element = screenshot_element(page, "#c", shots / "static-canvas.png", mode="canvas2d")
    assert element.nonblank() and element.distinct_colors >= 10, element
    assert (element.width, element.height) == (320, 200)


@pytest.mark.parametrize("mode", ["metal", "swiftshader"])
def test_webgpu_canvas_appears_in_screenshots(mode, page_dir, shots):
    (page_dir / "webgpu.html").write_text(WEBGPU_PAGE)
    stats = screenshot_element(str(page_dir / "webgpu.html"), "#c", shots / f"webgpu-{mode}.png",
                               wait_for="() => window.state !== 'start'", mode=mode)
    assert stats.nonblank(), stats
    # The triangle covers about a quarter of the canvas.
    assert 0.2 < stats.nonbackground_fraction < 0.3, stats


def _write_notebook(path: Path, *sources: str) -> Path:
    nb = new_notebook(cells=[new_code_cell(s) for s in sources])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, path)
    return path


def test_run_notebook_runs_a_trivial_notebook(page_dir):
    path = _write_notebook(page_dir / "trivial.ipynb", "x = 6 * 7", "print(x)")
    nb = run_notebook(path, timeout=60)
    assert nb.cells[1].outputs[0]["text"].strip() == "42"


def test_run_notebook_reports_the_failing_cell(page_dir):
    path = _write_notebook(page_dir / "failing.ipynb", "y = 1", "y / 0", "print('never')")
    with pytest.raises(NotebookCellError) as info:
        run_notebook(path, timeout=60)
    assert info.value.cell == 1 and info.value.ename == "ZeroDivisionError"
    assert "y / 0" in info.value.traceback


def test_notebook_session_fails_fast_and_cleans_up(page_dir):
    path = _write_notebook(page_dir / "failing.ipynb", "import time", "1 / 0", "time.sleep(600)")
    with pytest.raises(NotebookCellError) as info:
        with notebook_session(path, work=page_dir) as session:
            server = session.server
            kernels = _children(server.process.pid)
            session.wait(idle=True, timeout=60)
    assert info.value.cell == 1 and info.value.ename == "ZeroDivisionError"
    assert server.process.poll() is not None
    assert kernels and not any(_alive(k) for k in kernels)
    # After an error the session folder is kept for its server log.
    assert any(str(server.state) in note for note in info.value.__notes__)
    assert (server.state / "server.log").exists()


def test_screenshot_notebook_shows_the_widget(shots):
    result = screenshot_notebook(SELFTEST, shots / "selftest.png", nonblank=CANVAS, outputs=True,
                                 env={"MOCK_LOOP_SECONDS": "30"})
    assert result.stats and all(s.nonblank() for s in result.stats), result.stats
    assert result.pngs[0].exists() and png_stats(result.pngs[0]).nonblank()
    widget_outputs = [p for p in result.pngs[1:] if "cell1" in p.name]
    assert widget_outputs and png_stats(widget_outputs[0]).nonblank()


def test_widget_stays_live_while_a_later_cell_runs(shots):
    with notebook_session(SELFTEST, env={"MOCK_LOOP_SECONDS": "60"}) as session:
        session.wait("() => __caliperHarness.deepQueryAll('canvas')"
                     ".some(c => c.getRootNode() instanceof ShadowRoot)", timeout=60)
        session.wait(nonblank=CANVAS, timeout=60)
        # Cell 2 is the loop; take both shots while it is running.
        session.wait(lambda page: any(c["running"] and c["index"] == 2
                                      for c in session.state()["cells"]), timeout=30)
        first = session.page.locator(CANVAS).screenshot(path=str(shots / "live-a.png"))
        # Three more frames of the stream arrive (the widget draws each as it comes).
        got = session.page.evaluate(RECEIVED)
        session.wait(f"() => ({RECEIVED})() >= {got + 3}", timeout=30)
        second = session.page.locator(CANVAS).screenshot(path=str(shots / "live-b.png"))
        assert session.state()["cells"][2]["running"], "the loop ended before the second shot"
        assert png_stats(first).nonblank() and png_stats(second).nonblank()
        assert png_difference(first, second) > 0.05
        session.screenshot(shots / "live-notebook.png")

        webgpu = session.probe_webgpu()
        csp = session.headers.get("content-security-policy", "")

        (session.server.root / "stop-loop").write_text("")
        session.wait(idle=True, timeout=60)
        report = json.loads((session.server.root / "mock_stream_report.json").read_text())

    assert webgpu["adapter"] and webgpu["device"], webgpu
    # JupyterLab's only policy is frame-ancestors; it does not restrict sockets.
    assert "connect-src" not in csp and "default-src" not in csp, csp
    [connection] = report["connections"]
    # The socket to another loopback port carries the JupyterLab page's origin.
    assert connection["origin"] == session.server.url
    assert connection["received"] > 0
    assert connection["webgpu"]["adapter"] is True
    assert report["measured_rate"] > 20, report
