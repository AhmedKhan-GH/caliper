"""Dry run of step 1: train.py streams, and the hidden Caliper app attaches to it.

    python examples/build_a_dashboard/dry_run/step1_train_and_attach.py \\
        --train-python <a Python with torch and caliper> [--caliper cmake-build-release/caliper] \\
        [--out build/dry_run] [--work <folder>] [--size 1440x900] [--seconds 90] [--settle 15]

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
running, such as the training run that run_all.py --story keeps from step 1
to step 4, and leaves that stream running. --workspace and --app-data then
name the app's workspace and data folder, so that later steps continue in
them.

It needs nothing but the standard library itself; the Python given with
--train-python runs train.py. It sets CALIPER_AGENT_OFFLINE=1 and never shows
a window. The app's files go to --work, or to a temporary folder that is
removed at the end. Every process it starts is stopped by its pid when the
run ends, fails or is interrupted (SIGINT, SIGTERM or SIGHUP), and the app
also closes itself after CALIPER_EXIT_AFTER seconds should this script be
killed outright.

The capture waits on the input driver's wait-file step, which gives up after
1800 frames, about 30 s on the live page. That wait covers the catalog checks
and --settle, so the script refuses a --settle above 20 s, and stops with a
message when the catalog checks left too little of the wait for the settle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TRAIN = HERE.parent / "train.py"

# The input driver's wait-file step gives up after kPatience frames
# (src/host/builder/spec_ui_driver.h), about 30 s on the live page. The time
# from the live page to the capture is the catalog checks plus the settle; it
# is kept a few seconds under that wait.
PATIENCE_S = 30.0
BUDGET_S = 24.0
MAX_SETTLE_S = 20.0


class Stop(Exception):
    """A signal asked the dry run to stop."""


def on_signal(signum, frame):
    raise Stop(signal.Signals(signum).name)


class Proc:
    """A child process whose output is gathered by a thread, line by line,
    each line with the time it came."""

    def __init__(self, name: str, argv: list[str], env: dict[str, str]):
        self.name = name
        self.lines: list[tuple[float, str]] = []
        self.p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, text=True, bufsize=1, errors="replace")
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

    def stop(self, sig: int, timeout: float = 30.0) -> int | None:
        """Stops the process by its pid, with sig and then SIGKILL, and says so."""
        if self.p.poll() is None:
            print(f"stopping {self.name} (pid {self.p.pid}) with {signal.Signals(sig).name}", flush=True)
            self.p.send_signal(sig)
            try:
                return self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                print(f"stopping {self.name} (pid {self.p.pid}) with SIGKILL", flush=True)
                self.p.kill()
        return self.p.wait()


def command(port: int, token: str, name: str, args: dict) -> dict:
    """One command through Caliper API's command endpoint."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v0/commands/{name}", method="POST",
                                 data=json.dumps(args).encode(),
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-python", "--python", dest="train_python",
                    help="the Python that runs train.py (torch and caliper); not needed with --url")
    ap.add_argument("--url", help="attach to this stream, which is already running, instead of starting train.py")
    ap.add_argument("--caliper", default=os.environ.get("CALIPER_BIN") or str(ROOT / "cmake-build-release" / "caliper"),
                    help="the caliper executable (default: CALIPER_BIN, or cmake-build-release/caliper)")
    ap.add_argument("--out", default=str(ROOT / "build" / "dry_run"), help="where the captures go")
    ap.add_argument("--work", help="where the app's files go (default: a temporary folder, removed at the end)")
    ap.add_argument("--workspace", help="the app's workspace (default: one under --work)")
    ap.add_argument("--app-data", help="the app's data folder, CALIPER_DATA_DIR (default: one under --work)")
    ap.add_argument("--size", default="1440x900", help="the window's size in points")
    ap.add_argument("--seconds", type=float, default=90.0, help="how long train.py trains")
    ap.add_argument("--settle", type=float, default=15.0,
                    help=f"seconds of training the capture waits for after the catalog checks, at most {MAX_SETTLE_S:g}")
    args = ap.parse_args()

    if not args.url and not args.train_python:
        print("step 1: give --train-python, or --url of a stream that is already running", file=sys.stderr)
        return 2
    if args.settle > MAX_SETTLE_S:
        print(f"step 1: --settle {args.settle:g} is more than this script can wait for. The capture waits on the "
              f"input driver's wait-file step, which gives up after about {PATIENCE_S:g} s on the live page and "
              f"must also cover the catalog checks; give at most {MAX_SETTLE_S:g} s.", file=sys.stderr)
        return 2
    caliper = Path(args.caliper)
    if not caliper.is_file() or not os.access(caliper, os.X_OK):
        print(f"step 1: no caliper executable at {caliper}; build the target caliper, or give --caliper",
              file=sys.stderr)
        return 2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    temporary = not args.work
    work = Path(tempfile.mkdtemp(prefix="caliper-step1-")) if temporary else Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = Path(args.app_data).resolve() if args.app_data else work / "data"
    data.mkdir(parents=True, exist_ok=True)
    (work / "user-applets").mkdir(exist_ok=True)
    workspace = Path(args.workspace).resolve() if args.workspace else work / "ws"
    ready = work / "ready"
    ready.unlink(missing_ok=True)
    shot = f"step1-live-{args.size}"
    env = dict(os.environ, CALIPER_AGENT_OFFLINE="1")
    failed: list[str] = []

    def check(ok: bool, what: str) -> bool:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            failed.append(what)
        return ok

    train: Proc | None = None
    app: Proc | None = None
    interrupted = None
    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        # 1. The training script; it prints its stream link first.
        if args.url:
            link = args.url
            check(link.startswith("http://127.0.0.1:"), f"the stream link is on this machine ({link.split('?')[0]})")
        else:
            train = Proc("train.py", [args.train_python, "-u", str(TRAIN), "--seconds", str(args.seconds)], env)
            link = train.find("http://127.0.0.1:", 60)
            if not check(link is not None, "train.py printed its stream link"):
                print("\n".join(train.texts()[-20:]))
                return 1

        # 2. The app, hidden, attached to the stream from its command line. The
        # input script reads the live page back once the checks below have
        # passed; CALIPER_EXIT_AFTER closes it should nothing else.
        exit_after = 60 + 60 + args.settle + 60
        app_env = dict(env, CALIPER_DATA_DIR=str(data),
                       CALIPER_USER_APPLETS_DIR=str(work / "user-applets"),
                       CALIPER_WINDOW_SIZE=args.size,
                       CALIPER_EXIT_AFTER=f"{exit_after:g}",
                       CALIPER_TEST_SPEC_UI=f"page live;wait-file {ready};wait 30;shot {shot};quit",
                       CALIPER_TEST_SPEC_UI_SHOTS=str(out))
        app = Proc("the app", [str(caliper), "--hidden", "--url", link, "--api", "--api-port", "0",
                               "--workspace", str(workspace)], app_env)
        check(app.find("[api] Caliper API on ", 60) is not None, "the app started Caliper API")
        check(app.find("[live] attached", 30) is not None, "the app attached to the stream")
        live_page = app.when("[ui] page live", 30)
        check(live_page is not None, "the app shows the live page")
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
            return 1
        time.sleep(args.settle)
        ready.touch()
        end = time.monotonic() + 120
        while app.p.poll() is None and time.monotonic() < end and not any(
                x.startswith("[ui] stopped") for x in app.texts()):
            time.sleep(0.2)
        stopped = next((x for x in app.texts() if x.startswith("[ui] stopped")), None)
        check(stopped is None and app.find("[ui] done", 5) is not None,
              "the input script read the live page back" + (f" ({stopped})" if stopped else ""))
        code = app.p.wait(30) if app.p.poll() is None and stopped is None else app.p.poll()
        check(code == 0, f"the app quit with 0 ({code})")
        png = out / f"{shot}.png"
        check(png.exists() and png.stat().st_size > 10000, f"the capture {png.name} was written")
    except Stop as why:
        interrupted = str(why)
        print(f"step 1: interrupted by {why}", flush=True)
    except subprocess.TimeoutExpired:
        check(False, "the app quit within 30 s of its input script's end")
    finally:
        # Every process this script started, stopped by its pid. A second
        # signal while it stops them does not cut this short.
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        if app is not None:
            app.stop(signal.SIGTERM)
        if train is not None:
            train.stop(signal.SIGINT)
        if temporary:
            shutil.rmtree(work, ignore_errors=True)
        for s, handler in previous.items():
            signal.signal(s, handler)
    if train is not None:
        for line in train.texts():
            if line.startswith(("step ", "done", "stopped", "training on")):
                print("train.py: " + line)
    if interrupted:
        return 130
    print(("dry run of step 1 passed" if not failed else f"{len(failed)} checks failed") + f"; files in {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
