"""Headless screenshots of pages and elements, and pixel checks on them.

Command line, from the repository root:

    python -m tests.headless.screenshot <url-or-file> <out.png> [--wait-for X] [--nonblank SEL]

A local file path is served from a loopback HTTP server, because Chromium
does not let a ``file://`` page import ES modules beside it. The command
prints pixel statistics of the screenshot (or of the ``--element``) as JSON.

Pixels of a WebGPU canvas cannot be read back through the page once a frame
is presented, so every pixel check here works on a screenshot (PNG), which
shows exactly what was composited.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Union

import numpy as np
from PIL import Image
from playwright.sync_api import Locator, Page

from .browser import DEFAULT_MODE, MODES, open_page, static_server

# A predicate is a CSS selector (wait until an element matching it is
# visible; open shadow roots are searched), a JavaScript function source such
# as "() => window.ready === true", or a Python callable taking the page.
WaitFor = Union[str, Callable[[Page], bool], None]


@dataclass
class PixelStats:
    """Basic statistics of an image, used to tell whether something was drawn."""

    width: int
    height: int
    background: tuple[int, int, int]
    nonbackground_fraction: float
    distinct_colors: int

    def nonblank(self, min_fraction: float = 0.005, min_colors: int = 2) -> bool:
        """True when enough pixels differ from the background colour."""
        return self.nonbackground_fraction >= min_fraction and self.distinct_colors >= min_colors

    def to_dict(self) -> dict:
        return asdict(self) | {"nonblank": self.nonblank()}


def _load(png: bytes | str | Path | Image.Image) -> Image.Image:
    if isinstance(png, Image.Image):
        return png.convert("RGB")
    if isinstance(png, (bytes, bytearray)):
        return Image.open(io.BytesIO(png)).convert("RGB")
    return Image.open(png).convert("RGB")


def png_stats(
    png: bytes | str | Path | Image.Image,
    *,
    background: tuple[int, int, int] | None = None,
    tolerance: int = 8,
) -> PixelStats:
    """Compute pixel statistics of a PNG (path, bytes or PIL image).

    `background` defaults to the most common colour. A pixel counts as drawn
    when any channel differs from the background by more than `tolerance`.
    """
    image = _load(png)
    w, h = image.size
    colors = image.getcolors(maxcolors=w * h) or []
    if background is None:
        background = max(colors)[1] if colors else (0, 0, 0)
    br, bg, bb = background
    drawn = sum(
        count
        for count, (r, g, b) in colors
        if abs(r - br) > tolerance or abs(g - bg) > tolerance or abs(b - bb) > tolerance
    )
    return PixelStats(
        width=w,
        height=h,
        background=tuple(background),
        nonbackground_fraction=drawn / float(w * h) if w * h else 0.0,
        distinct_colors=len(colors),
    )


def png_difference(a: bytes | str | Path | Image.Image, b: bytes | str | Path | Image.Image,
                   *, tolerance: int = 8) -> float:
    """Fraction of pixels that differ between two images of the same size."""
    ia, ib = _load(a), _load(b)
    if ia.size != ib.size:
        raise ValueError(f"image sizes differ: {ia.size} and {ib.size}")
    pa = np.asarray(ia, dtype=np.int16)
    pb = np.asarray(ib, dtype=np.int16)
    changed = (np.abs(pa - pb) > tolerance).any(axis=-1)
    return float(changed.mean()) if changed.size else 0.0


def element_png(page: Page, selector: str | Locator, out_png: str | Path | None = None) -> bytes:
    """Screenshot the first element matching `selector` (open shadow roots included)."""
    locator = page.locator(selector).first if isinstance(selector, str) else selector
    data = locator.screenshot(path=str(out_png) if out_png else None, animations="allow")
    return data


def canvas_stats(page: Page, selector: str | Locator = "canvas", **kwargs) -> PixelStats:
    """Pixel statistics of an element as it appears on screen.

    Works for Canvas 2D, WebGL and WebGPU canvases alike, because it reads a
    screenshot of the element rather than the canvas's own buffer.
    """
    return png_stats(element_png(page, selector), **kwargs)


def _is_js(text: str) -> bool:
    t = text.lstrip()
    return t.startswith(("(", "function", "async")) or "=>" in t


def check(page: Page, predicate: WaitFor) -> bool:
    """Evaluate `predicate` once; see `WaitFor` for the accepted forms."""
    if predicate is None:
        return True
    if callable(predicate):
        return bool(predicate(page))
    if _is_js(predicate):
        return bool(page.evaluate(predicate))
    locator = page.locator(predicate).first
    return locator.count() > 0 and locator.is_visible()


def wait_for(page: Page, predicate: WaitFor, *, timeout: float = 30.0) -> None:
    """Wait until `predicate` holds; see `WaitFor` for the accepted forms."""
    deadline = time.monotonic() + timeout
    while not check(page, predicate):
        if time.monotonic() > deadline:
            raise TimeoutError(f"predicate {predicate!r} did not hold within {timeout} s")
        page.wait_for_timeout(100)


def nonblank_now(page: Page, selector: str) -> list[PixelStats]:
    """Pixel statistics of every visible element matching `selector`, or [] if
    none is ready. Hidden matches are left out: an element may hold a canvas it
    does not show (``<caliper-view>`` keeps one for each renderer), and a
    screenshot of a hidden element would wait for it to appear."""
    locator = page.locator(selector)
    try:
        shown = [locator.nth(i) for i in range(locator.count()) if locator.nth(i).is_visible()]
        return [png_stats(element_png(page, el)) for el in shown]
    except Exception:  # an element was re-rendered or hidden while we looked at it
        return []


def wait_nonblank(
    page: Page,
    selector: str,
    *,
    timeout: float = 30.0,
    min_fraction: float = 0.005,
    min_colors: int = 2,
) -> list[PixelStats]:
    """Wait until every element matching `selector` (at least one) is non-blank.

    Returns the statistics of each element at the moment the wait succeeded.
    """
    deadline = time.monotonic() + timeout
    while True:
        stats = nonblank_now(page, selector)
        if stats and all(s.nonblank(min_fraction, min_colors) for s in stats):
            return stats
        if time.monotonic() > deadline:
            raise TimeoutError(f"{selector!r} not non-blank within {timeout} s "
                               f"(stats {[s.to_dict() for s in stats]})")
        page.wait_for_timeout(200)


@contextlib.contextmanager
def _resolve(url: str) -> Iterator[str]:
    """Turn a local file path into a loopback URL; leave real URLs as they are."""
    if "://" in url:
        yield url
        return
    path = Path(url).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with static_server(path.parent) as base:
        yield f"{base}/{path.name}"


def screenshot_page(
    url: str,
    out_png: str | Path,
    *,
    wait_for: WaitFor = None,
    nonblank: str | None = None,
    timeout: float = 30.0,
    viewport: tuple[int, int] = (1280, 800),
    device_scale_factor: float = 1,
    full_page: bool = True,
    mode: str = DEFAULT_MODE,
    settle: float = 0.0,
) -> PixelStats:
    """Open `url` in headless Chromium, wait, and save a screenshot of the page.

    `wait_for` is a selector, a JavaScript predicate or a Python callable;
    `nonblank` is a selector whose elements must be non-blank first; `settle`
    adds a final pause in seconds. Returns the statistics of the screenshot.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with _resolve(url) as real_url, open_page(
        real_url, mode=mode, viewport=viewport, device_scale_factor=device_scale_factor,
        timeout=timeout,
    ) as page:
        _wait(page, wait_for, nonblank, timeout, settle)
        data = page.screenshot(path=str(out_png), full_page=full_page, animations="allow")
    return png_stats(data)


def screenshot_element(
    url: str,
    selector: str,
    out_png: str | Path,
    *,
    wait_for: WaitFor = None,
    nonblank: str | None = None,
    timeout: float = 30.0,
    viewport: tuple[int, int] = (1280, 800),
    device_scale_factor: float = 1,
    mode: str = DEFAULT_MODE,
    settle: float = 0.0,
) -> PixelStats:
    """Like `screenshot_page`, but save only the first element matching `selector`."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with _resolve(url) as real_url, open_page(
        real_url, mode=mode, viewport=viewport, device_scale_factor=device_scale_factor,
        timeout=timeout,
    ) as page:
        _wait(page, wait_for or selector, nonblank, timeout, settle)
        data = element_png(page, selector, out_png)
    return png_stats(data)


def _wait(page: Page, predicate: WaitFor, nonblank: str | None, timeout: float, settle: float) -> None:
    wait_for(page, predicate, timeout=timeout)
    if nonblank:
        wait_nonblank(page, nonblank, timeout=timeout)
    if settle:
        page.wait_for_timeout(settle * 1000)


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tests.headless.screenshot",
        description="Screenshot a URL or a local HTML file in headless Chromium.",
    )
    parser.add_argument("url", help="http(s) URL, or a local HTML file (served on 127.0.0.1)")
    parser.add_argument("out", help="output PNG path")
    parser.add_argument("--wait-for", help="CSS selector or JavaScript predicate to wait for")
    parser.add_argument("--nonblank", help="selector whose elements must be non-blank first")
    parser.add_argument("--element", help="save only the first element matching this selector")
    parser.add_argument("--mode", choices=sorted(MODES), default=DEFAULT_MODE)
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds (default 30)")
    parser.add_argument("--viewport", type=_size, default=(1280, 800), help="WIDTHxHEIGHT")
    parser.add_argument("--scale", type=float, default=1.0, help="device scale factor")
    parser.add_argument("--settle", type=float, default=0.0, help="extra seconds before the shot")
    ns = parser.parse_args(argv)
    common = dict(wait_for=ns.wait_for, nonblank=ns.nonblank, timeout=ns.timeout,
                  viewport=ns.viewport, device_scale_factor=ns.scale, mode=ns.mode,
                  settle=ns.settle)
    if ns.element:
        stats = screenshot_element(ns.url, ns.element, ns.out, **common)
    else:
        stats = screenshot_page(ns.url, ns.out, **common)
    print(json.dumps({"out": str(Path(ns.out).resolve()), **stats.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
