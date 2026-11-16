"""Step 6 of the walk-through in the hidden app: save the dashboard, reopen it
from the landing screen, and open it in a browser.

    python examples/build_a_dashboard/dry_run/step6_save_reopen_browser.py [--out DIR] [--sizes 1440x900,2560x1440]

The app runs without --api. The dashboard is built by hand as in step 2, then
⌘S saves it as the applet training-run in the workspace, ⌘B leaves build mode,
esc gives up the focus, and ⌘O Open in browser in the page bar is clicked: the
app starts Caliper API on a free loopback port, as it does for the side chat,
and gives the link of its browser viewer for the saved dashboard,
http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=training-run, to
the browser. Then esc goes Home, where Projects lists the dashboard and the
landing selects it; ⌘K, its title and Enter open it again. The saved and the
reopened dashboards are compared pixel by pixel, and the saved spec with
expected.spec.toml beside this script.

Meanwhile the browser part: headless Chromium (Playwright, from the web
package) opens the link once the app has given it, and the page is captured at
the window's size and laid beside the reopened app's capture, whole and panel
by panel.

The captures are step6-<n>-<what>-<W>x<H>.png in DIR/shots. Needs node, and
Playwright from `cd web && npm ci` (or NODE_PATH naming a node_modules that has
@playwright/test).

What a person does differently: the hidden app opens no browser, so the link
goes to the file that CALIPER_TEST_OPEN_URL names, the test hook in place of
the person's default browser, and this script hands it to headless Chromium.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

from hidden_app import APPLET, HERE, REPO, SIZES, TITLE, HiddenApp, build_steps, check_data, png_size, settle_and_shot

NAMES = ["saved", "landing", "reopened"]


def png_rows(path):
    """The size of a PNG and its scanlines as stored (each row's filter byte and
    its filtered bytes), undecoded."""
    data = Path(path).read_bytes()
    pos, idat, size = 8, b"", None
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            size = (int.from_bytes(body[0:4], "big"), int.from_bytes(body[4:8], "big"), body[8], body[9], body[12])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    h = size[1]
    stride = len(raw) // h
    return size, [raw[y * stride:(y + 1) * stride] for y in range(h)]


def compare_png(a, b):
    """Whether two captures by the same encoder hold the same pixels. Equal
    scanlines throughout mean equal pixels; a differing row is reported with
    its span (a filtered row also depends on the row above it, so the span can
    reach one row past the pixels that differ)."""
    sa, ra = png_rows(a)
    sb, rb = png_rows(b)
    if sa != sb:
        return f"sizes differ: {sa[0]}x{sa[1]} and {sb[0]}x{sb[1]}"
    diff = [y for y in range(len(ra)) if ra[y] != rb[y]]
    if not diff:
        return "identical pixels"
    return f"{len(diff)} of {len(ra)} rows differ, from y={diff[0]} to y={diff[-1]}"


def node_capture(*args):
    script = HERE / "browser_capture.mjs"
    r = subprocess.run(["node", str(script), *map(str, args)], capture_output=True, text=True, timeout=180)
    sys.stdout.write("".join("  " + l + "\n" for l in r.stdout.splitlines()))
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
    return r.returncode == 0


def run(size, out):
    root = out / size
    shots = out / "shots"
    done = root / "browser-done"
    link_file = root / "browser-link.txt"
    steps = build_steps() + [
        "key cmd+s", "wait 20", "key cmd+b", "key escape",
        # ⌘O Open in browser: the link goes to the test hook's file.
        "click page:browser", f"wait-file {link_file}", "wait 10", *settle_and_shot("saved"),
        "key escape", "page landing", "wait 30", f"landing selected local.{APPLET}",
        f"landing projects local.{APPLET}", *settle_and_shot("landing"),
        "key cmd+k", f"type {TITLE}", "key enter", "page spec", "wait 40", *settle_and_shot("reopened"),
        f"wait-file {done}", "quit"]

    def read_link():
        try:
            lines = link_file.read_text().splitlines()
        except OSError:
            return None
        return lines[0] if lines and re.fullmatch(r"http://127\.0\.0\.1:\d+/v0/spec\?token=[A-Za-z0-9_-]+&applet=[a-z0-9-]+",
                                                  lines[0]) else None

    app = HiddenApp(root, size, steps, shots, extra_env={"CALIPER_TEST_OPEN_URL": str(link_file)})
    ok = True
    browser_png = shots / f"step6-4-browser-{size}.png"
    state = root / "browser-state.json"
    # The browser opens the link once the app has given it, while the app goes Home and reopens the dashboard.
    deadline = time.time() + 300
    url = None
    while app.proc.poll() is None and time.time() < deadline and not url:
        url = read_link()
        time.sleep(0.2)
    if not url:
        print("  no link from Open in browser")
        ok = False
    else:
        print(f"  link: {url.split('?')[0]}?token=<viewer token>&applet={url.rsplit('=', 1)[1]}")
        ok = url.endswith(f"&applet={APPLET}") and node_capture("capture", url, size, browser_png, state)
    while app.proc.poll() is None and time.time() < deadline and not app.raw_shot("reopened").is_file():
        time.sleep(0.2)
    done.touch()
    rc = app.wait()
    print(f"  app: exit {rc}, {'done' if app.ui_done() else 'stopped'}")
    for line in app.ui_lines():
        if "stopped" in line or "shot " in line:
            print("   ", line)
    collected = app.collect(6, NAMES)
    ok = ok and rc == 0 and app.ui_done() and len(collected) == len(NAMES)
    if app.raw_shot("reopened").is_file() and browser_png.is_file():
        ok = node_capture("compare", app.raw_shot("reopened"), browser_png, state,
                          shots / f"step6-5-side-{size}.png", shots / f"step6-6-panels-{size}.png") and ok
    for s in sorted(shots.glob(f"step6-*-{size}.png")):
        w, h = png_size(s)
        print(f"  {s.name} {w}x{h}")
    # The saved and the reopened dashboards, whole windows.
    if app.raw_shot("saved").is_file() and app.raw_shot("reopened").is_file():
        print(f"  saved and reopened: {compare_png(app.raw_shot('saved'), app.raw_shot('reopened'))}")
    # The saved spec.
    saved = root / "workspace" / "applets" / APPLET / f"{APPLET}.spec.toml"
    expected = HERE / "expected.spec.toml"
    if not saved.is_file():
        print(f"  no saved spec at applets/{APPLET}/")
        ok = False
    else:
        same = saved.read_text() == expected.read_text()
        print(f"  saved spec {'equals' if same else 'differs from'} expected.spec.toml")
        ok = ok and same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "build" / "build_a_dashboard" / "step6"))
    ap.add_argument("--sizes", default=",".join(SIZES), help="window sizes in points, comma separated")
    args = ap.parse_args()
    check_data()
    if not shutil.which("node"):
        sys.exit("needs node for the browser part")
    out = Path(args.out)
    failed = False
    for size in args.sizes.split(","):
        print(f"{size}:")
        failed = not run(size, out) or failed
    print("step 6: " + ("FAILED" if failed else "ok"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
