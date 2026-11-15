"""Self-tests of the headless harness.

Run from the repository root with `python -m pytest tests/headless`. Screenshots
go to `$CALIPER_HEADLESS_DIR/shots` (see README.md), never into the repository.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from .browser import HeadedBrowserRefused, launch, probe_webgpu, work_dir
from .screenshot import screenshot_element, screenshot_page

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
