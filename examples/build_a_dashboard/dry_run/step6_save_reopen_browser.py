"""Step 6 of the walk-through in the hidden app: save the dashboard, reopen it
from the landing screen, and open it in a browser.

    python examples/build_a_dashboard/dry_run/step6_save_reopen_browser.py [--out DIR] [--work DIR]
        [--size 1440x900[,2560x1440]] [--caliper APP] [--workspace DIR --app-data DIR]

Without --workspace the
dashboard is built by hand as in step 2, in a new workspace, then ⌘S saves it
as the applet training-run, ⌘B leaves build mode, esc gives up the focus and
esc again goes Home, where the landing selects the dashboard under Projects,
and Enter opens it again. The saved and the reopened dashboards are compared pixel
by pixel, and the saved spec with expected.spec.toml beside this script.

With --workspace and --app-data, as in run_all.py --story, the dashboard is the
one that the earlier steps built and changed in that workspace: it is opened
from the landing screen, saved with ⌘S in build mode, and reopened, and the
saved spec must equal the spec the workspace held when the step began.

Meanwhile the browser part: once the dashboard is saved, the input script
clicks ⌘O Open in browser in the page bar. The hidden app opens no browser; its
test hook CALIPER_TEST_OPEN_URL writes the link instead, http://127.0.0.1:<port>
/v0/spec?token=<viewer token>&applet=training-run, which headless Chromium
(Playwright, from the web package) opens, and the page is captured at the
window's size and laid beside the reopened app's capture, whole and panel by
panel. After esc Home the landing must select the dashboard and list it under
Projects. The app's input script ends with the reopened capture and the app
stays open for the browser, which may take as long as it needs; the script then
closes the app.

The captures are step6-<n>-<what>-<W>x<H>.png in --out. Needs node, and
Playwright from `cd web && npm ci` (or NODE_PATH naming a node_modules that has
@playwright/test). The options are those of hidden_app.py.

What a person does differently: ⌘O opens the dashboard in their own default
browser.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

from hidden_app import (APPLET, EXPECTED_SPEC, FAILED, HERE, INTERRUPTED, MISSING, OK, TITLE, Children,
                        HiddenApp, Stop, add_options, build_steps, caliper_problem, clean_env, data_problem,
                        fresh_folder, png_size, saved_spec, settle_and_shot, work_folder)

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


def node(children, *args, timeout=180):
    """browser_capture.mjs under node, recorded and stopped by its pid if it outlasts the timeout."""
    p = children.start("node browser_capture.mjs", ["node", str(HERE / "browser_capture.mjs"), *map(str, args)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
                       env=clean_env())
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        children.stop(p)
        out, err = p.communicate()
        print(f"  node took longer than {timeout} s and was stopped")
    sys.stdout.write("".join("  " + l + "\n" for l in out.splitlines()))
    if p.returncode != 0:
        sys.stdout.write(err[-2000:])
    return p.returncode == 0


def until(app, name, deadline):
    """Waits for a raw capture, while the app runs and its input script has not stopped."""
    while (app.proc.poll() is None and time.time() < deadline and not app.raw_shot(name).is_file()
           and not any(l.startswith("[ui] stopped") for l in app.ui_lines())):
        time.sleep(0.2)
    return app.raw_shot(name).is_file()


def run(children, args, size, work, out):
    story = bool(args.workspace)
    root = fresh_folder(work, size)
    workspace = Path(args.workspace).resolve() if story else root / "workspace"
    found = saved_spec(workspace).read_text() if story else None
    link_file = root / "browser-link.txt"
    if story:
        # The story's dashboard, opened from the landing screen and saved in build mode. Nothing is focused
        # then, so no esc follows: with nothing to give up, esc goes Home.
        steps = ["wait 60", "page landing", "key cmd+k", f"type {TITLE}", "key enter", "page spec", "wait 40",
                 "key cmd+b", "wait 20", "key cmd+s", "wait 20", "key cmd+b", "wait 10"]
    else:
        # After the build the table is focused; esc gives up the focus.
        steps = build_steps() + ["key cmd+s", "wait 20", "key cmd+b", "key escape"]
    steps += [
        # ⌘O Open in browser: the hidden app writes the link to the test hook's file.
        "click page:browser", f"wait-file {link_file}", "wait 10", *settle_and_shot("saved"),
        # Home, where the landing selects the dashboard under Projects, and Enter opens it again.
        "key escape", "page landing", "wait 30", f"landing selected local.{APPLET}",
        f"landing projects local.{APPLET}", *settle_and_shot("landing"),
        "key enter", "page spec", "wait 40", *settle_and_shot("reopened")]

    def read_link():
        try:
            lines = link_file.read_text().splitlines()
        except OSError:
            return None
        pattern = r"http://127\.0\.0\.1:\d+/v0/spec\?token=[A-Za-z0-9_-]+&applet=[a-z0-9-]+"
        return lines[0] if lines and re.fullmatch(pattern, lines[0]) else None

    app = HiddenApp(children, args.caliper, root, size, steps, workspace=workspace,
                    app_data=Path(args.app_data).resolve() if args.app_data else None,
                    extra_env={"CALIPER_TEST_OPEN_URL": str(link_file)})
    ok = True
    browser_png = out / f"step6-4-browser-{size}.png"
    state = root / "browser-state.json"
    # The browser opens the link once the app has given it, while the app goes Home and reopens the dashboard.
    deadline = time.time() + 300
    url = None
    while app.proc.poll() is None and time.time() < deadline and not url and not any(
            l.startswith("[ui] stopped") for l in app.ui_lines()):
        url = read_link()
        time.sleep(0.2)
    if not url:
        print("  no link from Open in browser")
        ok = False
    else:
        print(f"  link: {url.split('?')[0]}?token=<viewer token>&applet={url.rsplit('=', 1)[1]}")
        ok = url.endswith(f"&applet={APPLET}") and node(children, "capture", url, size, browser_png, state)
    until(app, "reopened", deadline)
    # The input script ends with the reopened capture; the app stays open until it is closed here.
    end = time.time() + 30
    while app.proc.poll() is None and time.time() < end and not app.ui_done() and not any(
            l.startswith("[ui] stopped") for l in app.ui_lines()):
        time.sleep(0.2)
    done = app.ui_done()
    rc = app.stop()
    print(f"  app: closed with exit {rc}, its input script {'done' if done else 'stopped'}")
    for line in app.ui_lines():
        if "stopped" in line or "shot " in line:
            print("   ", line)
    collected = app.collect(6, NAMES, out)
    ok = ok and rc == 0 and done and len(collected) == len(NAMES)
    if app.raw_shot("reopened").is_file() and browser_png.is_file():
        ok = node(children, "compare", app.raw_shot("reopened"), browser_png, state,
                  out / f"step6-5-side-{size}.png", out / f"step6-6-panels-{size}.png") and ok
    for s in sorted(out.glob(f"step6-*-{size}.png")):
        w, h = png_size(s)
        print(f"  {s.name} {w}x{h}")
    # The saved and the reopened dashboards, whole windows.
    if app.raw_shot("saved").is_file() and app.raw_shot("reopened").is_file():
        print(f"  saved and reopened: {compare_png(app.raw_shot('saved'), app.raw_shot('reopened'))}")
    # The saved spec: as built by hand, or as the story's workspace held it.
    saved = saved_spec(workspace)
    if not saved.is_file():
        print(f"  no saved spec at applets/{APPLET}/")
        ok = False
    elif story:
        same = saved.read_text() == found
        print(f"  saved spec {'equals' if same else 'differs from'} the spec the workspace held at the start")
        ok = ok and same
    else:
        same = saved.read_text() == EXPECTED_SPEC.read_text()
        print(f"  saved spec {'equals' if same else 'differs from'} expected.spec.toml")
        ok = ok and same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_options(ap, sizes=True)
    args = ap.parse_args()
    sizes = args.size.split(",")
    problems = [data_problem(), caliper_problem(args.caliper)]
    if not shutil.which("node"):
        problems.append("needs node for the browser part")
    if bool(args.workspace) != bool(args.app_data):
        problems.append("--workspace and --app-data go together")
    if args.workspace and len(sizes) > 1:
        problems.append("one --size with --workspace")
    if args.workspace and not saved_spec(args.workspace).is_file():
        problems.append(f"the workspace holds no saved {APPLET}; step 2 saves it there")
    for problem in filter(None, problems):
        print(f"step 6: {problem}", file=sys.stderr)
        return MISSING
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    failed = False
    with Children() as children:
        try:
            work = work_folder(args, 6)
            print(f"step 6: files in {work}")
            for size in sizes:
                print(f"{size}:")
                failed = not run(children, args, size, work, out) or failed
        except Stop as why:
            print(f"step 6: interrupted by {why}")
            return INTERRUPTED
    print("step 6: " + ("FAILED" if failed else "ok"))
    return FAILED if failed else OK


if __name__ == "__main__":
    sys.exit(main())
