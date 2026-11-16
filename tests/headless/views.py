"""Waits on what Caliper's <caliper-view> elements in a page have drawn and
received, for the notebook and widget checks. A check waits on frames drawn,
snapshots received or published state, never on a fixed time: a software
adapter on a CI runner draws a frame in a second or more.

`view` names an element by its `view` attribute. The page must have the
harness's helpers (`__caliperHarness.deepQueryAll`), as every page the harness
opens does."""
from __future__ import annotations

from playwright.sync_api import Page

FIND = ("(name) => __caliperHarness.deepQueryAll('caliper-view')"
        ".find((e) => e.getAttribute('view') === name)")
# How many snapshots a view has received: the WebAssembly views' count of
# applied updates (as the module publishes it), a series' steps, and the
# newest snapshot's sequence number for the element's other views.
RECEIVED = ("(s) => s.drawer === 'wasm' ? Number(s.updates)"
            " : s.kind === 'series' ? Number(s.steps && s.steps.hi) : Number(s.seq)")
MODULE_FRAMES = f"(n) => Number(((({FIND})(n).debugState().wasmStats) || {{}}).frames || 0)"
# An element's own view has drawn what it holds once settled() resolves, or once
# it has drawn a frame since the call while data keeps arriving.
DREW = ("const drew = async (e) => { const frames = () => Number(e.debugState().frames ?? NaN);"
        " const f0 = frames(); let done = false; e.settled().then(() => { done = true; });"
        " while (!done && !(frames() > f0)) await new Promise((ok) => requestAnimationFrame(() => ok(null))); };")


def received(page: Page, view: str) -> float:
    return page.evaluate(f"(n) => ({RECEIVED})(({FIND})(n).debugState())", view)


def snapshots_pass(page: Page, view: str, k: int, timeout: float = 60.0) -> None:
    """Waits until the view has received k more snapshots."""
    want = received(page, view) + k
    page.wait_for_function(f"([n, want]) => ({RECEIVED})(({FIND})(n).debugState()) >= want",
                           arg=[view, want], polling="raf", timeout=timeout * 1000)


def drawn(page: Page, view: str, frames: int = 2, timeout: float = 60.0) -> None:
    """Waits until the view has drawn what it holds. The element's own views
    draw on demand: settled() resolves once no frame is pending, and while data
    keeps arriving faster than a slow machine draws, a frame drawn after this
    call is enough (its frames count grows). A WebAssembly view's module draws
    on every animation frame and publishes its frame count at the end of a
    frame: the count must change, which a frame that began after this call
    does, and pass the module's 30 start-up frames; then it must grow by
    frames - 1 more."""
    count = page.evaluate(f"async (n) => {{ const v = ({FIND})(n); if (v.debugState().drawer === 'wasm')"
                          f" return ({MODULE_FRAMES})(n); {DREW} await drew(v); return null; }}", view)
    if count is None:
        return
    first = page.wait_for_function(f"([n, min]) => {{ const f = ({MODULE_FRAMES})(n); return f >= min && f; }}",
                                   arg=[view, max(count + 1, 31)], polling="raf",
                                   timeout=timeout * 1000).json_value()
    if frames > 1:
        page.wait_for_function(f"([n, min]) => ({MODULE_FRAMES})(n) >= min", arg=[view, first + frames - 1],
                               polling="raf", timeout=timeout * 1000)


def animation_frames(page: Page, n: int) -> None:
    """Waits for n animation frames of the page: a hold measured in frames."""
    page.evaluate("async (n) => { for (let i = 0; i < n; i++)"
                  " await new Promise((ok) => requestAnimationFrame(() => ok(null))); }", n)
