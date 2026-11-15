"""Headless Chromium for Caliper's browser tests.

Every browser this module starts is headless. A request for a headed browser
raises `HeadedBrowserRefused`, because tests must never open a window.

Three rendering modes are available:

- ``metal`` (the default): WebGPU on the Mac's GPU through Metal. The adapter
  is a hardware adapter (vendor ``apple``).
- ``swiftshader``: WebGPU on SwiftShader, Chromium's software renderer. The
  adapter is a fallback (software) adapter.
- ``canvas2d``: WebGPU is hidden (``navigator.gpu`` is undefined and no adapter
  is available), so a viewer takes its Canvas 2D path.

The README in this folder records how these flag sets were chosen.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

# Flag sets measured with Playwright's chrome-headless-shell on macOS (arm64).
# `--use-angle=metal` alone selects the Metal adapter; `--enable-unsafe-webgpu`
# keeps WebGPU exposed even where Chromium's GPU blocklist would hide it.
# SwiftShader needs `--use-angle=swiftshader` as well, otherwise WebGPU draws
# but the canvas composites as black in screenshots.
MODES: dict[str, tuple[str, ...]] = {
    "metal": ("--enable-unsafe-webgpu", "--use-angle=metal"),
    "swiftshader": (
        "--enable-unsafe-webgpu",
        "--use-angle=swiftshader",
        "--use-webgpu-adapter=swiftshader",
    ),
    "canvas2d": (),
}
DEFAULT_MODE = "metal"

# No Chromium flag removes `navigator.gpu` from the headless shell, so the
# Canvas 2D mode deletes it before any page script runs. Without WebGPU flags
# the headless shell returns no adapter either, which also covers workers.
_HIDE_WEBGPU = """
for (const proto of [globalThis.Navigator, globalThis.WorkerNavigator]) {
  if (proto && proto.prototype && 'gpu' in proto.prototype) delete proto.prototype.gpu;
}
"""

# Helpers for page predicates. `__caliperHarness.deepQueryAll(selector)` also
# searches open shadow roots. Unlike Playwright's CSS locators it matches the
# selector inside each root separately, so `deepQueryAll('canvas')` finds a
# canvas in a shadow root but `deepQueryAll('caliper-view canvas')` does not.
HARNESS_JS = """
(() => {
  if (window.__caliperHarness) return;
  function deepQueryAll(selector, root = document) {
    const found = [...root.querySelectorAll(selector)];
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) found.push(...deepQueryAll(selector, el.shadowRoot));
    }
    return found;
  }
  window.__caliperHarness = {
    deepQueryAll,
    deepQuery: (selector, root) => deepQueryAll(selector, root)[0] || null,
  };
})();
"""

# Runs inside a page and reports what WebGPU offers there.
PROBE_JS = """
async () => {
  const out = {mode: null, has_gpu: !!navigator.gpu, adapter: false, device: false,
               software: null, info: null, error: null, secure_context: isSecureContext};
  if (!navigator.gpu) return out;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { out.error = 'requestAdapter() returned null'; return out; }
    out.adapter = true;
    const info = adapter.info || {};
    out.info = {vendor: info.vendor || '', architecture: info.architecture || '',
                device: info.device || '', description: info.description || ''};
    const fallback = info.isFallbackAdapter ?? adapter.isFallbackAdapter;
    out.software = !!fallback || /swiftshader/i.test(out.info.architecture + out.info.vendor);
    try {
      const device = await adapter.requestDevice();
      out.device = !!device;
      device.destroy();
    } catch (e) { out.error = 'requestDevice() failed: ' + e; }
  } catch (e) { out.error = String(e); }
  return out;
}
"""


WORK_DIR_ENV = "CALIPER_HEADLESS_DIR"


def work_dir(base: str | Path | None = None) -> Path:
    """The folder for everything the harness writes: `base`, else
    `$CALIPER_HEADLESS_DIR`, else a folder in the system temporary folder."""
    path = Path(base or os.environ.get(WORK_DIR_ENV) or Path(tempfile.gettempdir()) / "caliper-headless")
    path.mkdir(parents=True, exist_ok=True)
    return path


class HeadedBrowserRefused(RuntimeError):
    """Raised when code asks for a browser that would open a window."""


def _args_for(mode: str, extra_args: tuple[str, ...] | list[str]) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    args = [*MODES[mode], *extra_args]
    for arg in args:
        # Playwright sets the headless switch itself; nobody else may change it.
        if arg.startswith(("--headless", "--auto-open-devtools-for-tabs")):
            raise HeadedBrowserRefused(f"refusing browser argument {arg!r}")
    return args


@contextlib.contextmanager
def launch(
    mode: str = DEFAULT_MODE,
    *,
    headless: bool = True,
    extra_args: tuple[str, ...] | list[str] = (),
    playwright: Playwright | None = None,
) -> Iterator[Browser]:
    """Start headless Chromium in `mode` and close it on exit.

    `headless` exists only so that a caller who passes ``headless=False``
    gets a clear error instead of a window.
    """
    if headless is not True:
        raise HeadedBrowserRefused("tests run headless Chromium only; a headed browser is refused")
    args = _args_for(mode, extra_args)
    with contextlib.ExitStack() as stack:
        pw = playwright or stack.enter_context(sync_playwright())
        browser = pw.chromium.launch(headless=True, args=args)
        browser._caliper_mode = mode  # read by new_page()
        try:
            yield browser
        finally:
            browser.close()


def new_page(
    browser: Browser,
    *,
    viewport: tuple[int, int] = (1280, 800),
    device_scale_factor: float = 1,
) -> Page:
    """Open a page in a fresh context with the harness helpers installed."""
    context = browser.new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        device_scale_factor=device_scale_factor,
    )
    if getattr(browser, "_caliper_mode", None) == "canvas2d":
        context.add_init_script(_HIDE_WEBGPU)
    context.add_init_script(HARNESS_JS)
    return context.new_page()


@contextlib.contextmanager
def open_page(
    url: str | None = None,
    *,
    mode: str = DEFAULT_MODE,
    viewport: tuple[int, int] = (1280, 800),
    device_scale_factor: float = 1,
    timeout: float = 30.0,
    extra_args: tuple[str, ...] | list[str] = (),
) -> Iterator[Page]:
    """Start a browser, open one page (at `url` if given) and close both on exit."""
    with launch(mode, extra_args=extra_args) as browser:
        page = new_page(browser, viewport=viewport, device_scale_factor=device_scale_factor)
        page.set_default_timeout(timeout * 1000)
        if url is not None:
            page.goto(url, wait_until="load")
        yield page


def probe_page(page: Page) -> dict:
    """Report what WebGPU offers in an already open page."""
    return page.evaluate(PROBE_JS)


def probe_webgpu(mode: str = DEFAULT_MODE, *, timeout: float = 30.0) -> dict:
    """Ask headless Chromium in `mode` for a WebGPU adapter and device.

    Returns a dict with ``has_gpu`` (``navigator.gpu`` exists), ``adapter`` and
    ``device`` (whether each was obtained), ``software`` (whether the adapter
    is a software adapter), ``info`` (vendor, architecture, device and
    description) and ``error``. WebGPU needs a secure context, so the probe
    page is served from a loopback HTTP server.
    """
    with static_server(None) as base, open_page(base + "/", mode=mode, timeout=timeout) as page:
        result = probe_page(page)
    result["mode"] = mode
    return result


class _FileHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
        pass


class _BlankHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
        pass

    def do_GET(self):
        body = b"<!doctype html><title>blank</title>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def static_server(directory: str | Path | None) -> Iterator[str]:
    """Serve `directory` on 127.0.0.1 and yield its base URL (no trailing slash).

    A page served from loopback is a secure context (WebGPU needs one) and,
    unlike a page opened from a ``file://`` URL, can import ES modules beside
    it. With ``directory=None`` every path returns an empty HTML page.
    """
    if directory is None:
        handler = _BlankHandler
    else:
        handler = functools.partial(_FileHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="static-server", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe WebGPU in headless Chromium.")
    parser.add_argument("--mode", choices=sorted(MODES), action="append",
                        help="mode to probe (repeatable; default: every mode)")
    ns = parser.parse_args()
    for mode in ns.mode or list(MODES):
        print(json.dumps(probe_webgpu(mode)))


if __name__ == "__main__":
    _main()
