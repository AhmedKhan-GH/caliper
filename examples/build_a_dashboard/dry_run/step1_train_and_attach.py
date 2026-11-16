"""Dry run of step 1: train.py streams, and the hidden Caliper app attaches to it.

    python examples/build_a_dashboard/dry_run/step1_train_and_attach.py \\
        --train-python <a Python with torch and caliper> [--caliper cmake-build-release/caliper] \\
        [--out build/build_a_dashboard] [--work <folder>] [--size 1440x900] [--seconds 90] [--settle 15]

It does what a person does in step 1, without a window: it starts train.py,
reads the stream link that the script prints, and starts the app hidden with
that link on its command line (`caliper --hidden --url <link>`), which opens
the live page on the stream as F4 and the pasted link do. It then checks, over
Caliper API, that the catalog lists the stream's views as live sources with
their shapes and rates, reads the live page back into
<out>/step1-live-<W>x<H>.png, writes the catalog's live entries to
<out>/step1-catalog.json, and stops both. It prints one line per check and
exits with 0 when every check passed, 1 when one failed, 2 when something it
needs is missing, and 130 when it was interrupted.

With --url it starts no training and attaches to a stream that is already
running, such as the training run that run_all.py --story starts, and leaves
that stream running. --workspace and --app-data then name the app's workspace
and data folder, so that later steps continue in them.

It needs nothing but the standard library itself; the Python given with
--train-python runs train.py. The app gets the environment of hidden_app.py
(CALIPER_AGENT_OFFLINE=1, a PATH of the system's folders, CALIPER_EXIT_AFTER)
and never shows a window. The app's files go into a new subfolder of --work,
or into a temporary folder that is removed at the end. Every process it starts
is stopped by its pid when the run ends, fails or is interrupted.

The capture waits on the input driver's wait-file step, which gives up after
1800 frames, about 30 s on the live page. That wait covers the catalog checks
and --settle, so the script refuses a --settle above 20 s, and stops with a
message when the catalog checks left too little of the wait for the settle.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from hidden_app import (FAILED, INTERRUPTED, MISSING, OK, Children, Stop, add_options, app_env, caliper_problem,
                        clean_env, work_folder)

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train.py"

# The input driver's wait-file step gives up after kPatience frames
# (src/host/builder/spec_ui_driver.h), about 30 s on the live page. The time
# from the live page to the capture is the catalog checks plus the settle; it
# is kept a few seconds under that wait.
PATIENCE_S = 30.0
BUDGET_S = 24.0
MAX_SETTLE_S = 20.0


class Lines:
    """A child's output, gathered by a thread line by line, each line with the time it came."""

    def __init__(self, p: subprocess.Popen):
        self.p = p
        self.lines: list[tuple[float, str]] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stdout:
            self.lines.append((time.monotonic(), line.rstrip("\n")))

    def texts(self) -> list[str]:
        return [line for _, line in list(self.lines)]

    def find(self, prefix: str, timeout: float) -> str | None:
        """The first line that starts with prefix, waiting up to timeout seconds."""
        hit = self.when(prefix, timeout)
        return hit[1] if hit else None

    def when(self, prefix: str, timeout: float) -> tuple[float, str] | None:
        """The first line that starts with prefix and the time it came, waiting up to timeout seconds."""
        end = time.monotonic() + timeout
        while True:
            for t, line in list(self.lines):
                if line.startswith(prefix):
                    return t, line
            if self.p.poll() is not None:
                time.sleep(0.2)                     # the reader may still hold the last lines
                return next(((t, x) for t, x in list(self.lines) if x.startswith(prefix)), None)
            if time.monotonic() >= end:
                return None
            time.sleep(0.1)


def command(port: int, token: str, name: str, args: dict) -> dict:
    """One command through Caliper API's command endpoint."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v0/commands/{name}", method="POST",
                                 data=json.dumps(args).encode(),
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run(children: Children, args, work: Path, out: Path, check) -> bool:
    """The step itself; False when it stopped early."""
    data = Path(args.app_data).resolve() if args.app_data else work / "data"
    workspace = Path(args.workspace).resolve() if args.workspace else work / "ws"
    (work / "user-applets").mkdir(exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    ready = work / "ready"                  # in a folder of this run's own, so never one from before
    shot = f"step1-live-{args.size}"
    pipes = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1,
                 errors="replace")

    # 1. The training script; it prints its stream link first.
    train = None
    if args.url:
        link = args.url
        if not check(link.startswith("http://127.0.0.1:"), f"the stream link is on this machine ({link.split('?')[0]})"):
            return False
    else:
        train = Lines(children.start("train.py", [args.train_python, "-u", str(TRAIN), "--seconds", str(args.seconds)],
                                     stop_signal=signal.SIGINT, env=clean_env(), **pipes))
        link = train.find("http://127.0.0.1:", 60)
        if not check(link is not None, "train.py printed its stream link"):
            print("\n".join(train.texts()[-20:]))
            return False

    # 2. The app, hidden, attached to the stream from its command line. The
    # input script reads the live page back once the checks below have passed.
    env = app_env(data, work / "user-applets", args.size, f"page live;wait-file {ready};wait 30;shot {shot};quit",
                  out, exit_after=60 + 60 + args.settle + 60)
    app = Lines(children.start("the app", [args.caliper, "--hidden", "--url", link, "--api", "--api-port", "0",
                                           "--workspace", str(workspace)], env=env, **pipes))
    try:
        check(app.find("[api] Caliper API on ", 60) is not None, "the app started Caliper API")
        check(app.find("[live] attached", 30) is not None, "the app attached to the stream")
        live_page = app.when("[ui] page live", 30)
        check(live_page is not None, "the app shows the live page")
        if not check((data / "api.json").is_file(), "the app wrote its API file"):
            return False
        api = json.loads((data / "api.json").read_text())

        # 3. The catalog lists the stream's views as live sources, with their
        # shapes once a snapshot arrived and a rate once a few did.
        want = ["live:attn", "live:attn.entropy", "live:loss"] + [f"live:entropy.l{k}" for k in range(4)]
        live: dict[str, dict] = {}
        end = time.monotonic() + 60
        while time.monotonic() < end:
            listed = command(api["port"], api["token"], "catalog.list", {"kind": "live"})
            live = {e["id"]: e for e in listed.get("result", {}).get("entries", [])}
            if all(i in live and (live[i].get("rate_hz") or 0) > 0 for i in want):
                break
            time.sleep(0.5)
        for i in want:
            check(i in live and (live[i].get("rate_hz") or 0) > 0,
                  f"the catalog lists {i} with a rate ({(live.get(i) or {}).get('rate_hz')})")
        attn_shape = (live.get("live:attn") or {}).get("shape")
        check(attn_shape is not None and len(attn_shape) == 3 and attn_shape[0] == 4,
              f"live:attn is one map per layer, {attn_shape}")
        check((live.get("live:attn.entropy") or {}).get("shape") == [4, 4],
              f"live:attn.entropy is (layer, head), {(live.get('live:attn.entropy') or {}).get('shape')}")
        (out / "step1-catalog.json").write_text(json.dumps(live, indent=2) + "\n")

        # 4. Let it train a while, then take the capture and quit. The settle
        # must end before the input driver's wait for the ready file gives up.
        spent = time.monotonic() - live_page[0] if live_page else 0.0
        fits = spent + args.settle <= BUDGET_S
        if not check(fits, f"the settle of {args.settle:g} s fits the input driver's wait for the capture, about "
                           f"{PATIENCE_S:g} s, after {spent:.0f} s of catalog checks"
                           + ("" if fits else "; give a shorter --settle")):
            return False
        time.sleep(args.settle)
        ready.touch()
        end = time.monotonic() + 120
        while app.p.poll() is None and time.monotonic() < end and not any(
                x.startswith("[ui] stopped") for x in app.texts()):
            time.sleep(0.2)
        stopped = next((x for x in app.texts() if x.startswith("[ui] stopped")), None)
        check(stopped is None and app.find("[ui] done", 5) is not None,
              "the input script read the live page back" + (f" ({stopped})" if stopped else ""))
        try:
            code = app.p.wait(30) if stopped is None else children.stop(app.p)
        except subprocess.TimeoutExpired:
            code = children.stop(app.p)
        check(code == 0, f"the app quit with 0 ({code})")
        png = out / f"{shot}.png"
        check(png.exists() and png.stat().st_size > 10000, f"the capture {png.name} was written")
        return True
    finally:
        children.stop(app.p)
        if train is not None:
            children.stop(train.p)
            for line in train.texts():
                if line.startswith(("step ", "done", "stopped", "training on")):
                    print("train.py: " + line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_options(ap, train=True, url=True)
    ap.add_argument("--seconds", type=float, default=90.0, help="how long train.py trains")
    ap.add_argument("--settle", type=float, default=15.0,
                    help=f"seconds of training the capture waits for after the catalog checks, at most {MAX_SETTLE_S:g}")
    args = ap.parse_args()

    problems = [caliper_problem(args.caliper)]
    if not args.url and not args.train_python:
        problems.append("give --train-python, or --url of a stream that is already running")
    if bool(args.workspace) != bool(args.app_data):
        problems.append("--workspace and --app-data go together")
    if args.settle > MAX_SETTLE_S:
        problems.append(f"--settle {args.settle:g} is more than this script can wait for. The capture waits on the "
                        f"input driver's wait-file step, which gives up after about {PATIENCE_S:g} s on the live page "
                        f"and must also cover the catalog checks; give at most {MAX_SETTLE_S:g} s.")
    for problem in filter(None, problems):
        print(f"step 1: {problem}", file=sys.stderr)
        return MISSING

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    def check(ok: bool, what: str) -> bool:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            failed.append(what)
        return ok

    with Children() as children:
        try:
            work = work_folder(args, 1)
            if not run(children, args, work, out, check) and not failed:
                failed.append("stopped early")
        except Stop as why:
            print(f"step 1: interrupted by {why}", flush=True)
            return INTERRUPTED
    print(("dry run of step 1 passed" if not failed else f"{len(failed)} checks failed") + f"; files in {out}")
    return OK if not failed else FAILED


if __name__ == "__main__":
    sys.exit(main())
