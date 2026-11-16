"""Dry run of step 1: train.py streams, and the hidden Caliper app attaches to it.

    python examples/build_a_dashboard/dry_run/step1_train_and_attach.py \\
        --train-python <a Python with torch and caliper> [--caliper cmake-build-release/caliper] \\
        [--out build/build_a_dashboard] [--work <folder>] [--size 1440x900] [--seconds 90] [--settle 15]

It does what a person does in step 1, without a window: it starts train.py
with the same app-data folder as the app, so that the app lists the script's
stream from the stream file that caliper.serve() writes there, and starts the
app hidden. Through the app's input driver it presses F4 on the landing
screen, reads the session list back into <out>/step1-f4-<W>x<H>.png and presses
Enter on the stream, which opens its live page. It then checks, over Caliper
API, that the catalog lists the stream's views as live sources with their
shapes and rates, lets the model train for --settle seconds, reads the live
page back into <out>/step1-live-<W>x<H>.png, goes Home and checks that the app
stays attached and the catalog still lists the live sources, writes the
catalog's live entries to <out>/step1-catalog.json, and stops both. It prints
one line per check and exits with 0 when every check passed, 1 when one
failed, 2 when something it needs is missing, and 130 when interrupted.

With --url it starts no training and attaches to a stream that is already
running, such as the training run that run_all.py --story starts, and leaves
that stream running: through F4 when the stream's file is in the app-data
folder (the runner starts train.py on the story's), else with the link on the
app's command line (`caliper --hidden --url <link>`), as a person may start
the app. --workspace and --app-data name the app's workspace and data folder,
so that later steps continue in them.

It needs nothing but the standard library itself; the Python given with
--train-python runs train.py. The app gets the environment of hidden_app.py
(CALIPER_AGENT_OFFLINE=1, a PATH of the system's folders, CALIPER_EXIT_AFTER)
and never shows a window. The app's files go into a new subfolder of --work,
or into a temporary folder that is removed at the end. Every process it starts
is stopped by its pid when the run ends, fails or is interrupted.
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
from urllib.parse import urlsplit

from hidden_app import (FAILED, INTERRUPTED, MISSING, OK, Children, Stop, add_options, app_env, caliper_problem,
                        clean_env, work_folder)

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train.py"
WANT = ["live:attn", "live:attn.entropy", "live:loss"] + [f"live:entropy.l{k}" for k in range(4)]


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
        end = time.monotonic() + timeout
        while True:
            hit = next((x for x in self.texts() if x.startswith(prefix)), None)
            if hit is not None:
                return hit
            if self.p.poll() is not None:
                time.sleep(0.2)                     # the reader may still hold the last lines
                return next((x for x in self.texts() if x.startswith(prefix)), None)
            if time.monotonic() >= end:
                return None
            time.sleep(0.1)


def command(port: int, token: str, name: str, args: dict) -> dict:
    """One command through Caliper API's command endpoint."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v0/commands/{name}", method="POST",
                                 data=json.dumps(args).encode(),
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read())


def live_sources(api: dict) -> dict[str, dict]:
    listed = command(api["port"], api["token"], "catalog.list", {"kind": "live"})
    return {e["id"]: e for e in listed.get("result", {}).get("entries", [])}


def listed_stream(data: Path, link: str, timeout: float) -> bool:
    """Whether the stream at link has its stream file in the app-data folder, where F4 lists it."""
    port = urlsplit(link).port
    end = time.monotonic() + timeout
    while True:
        for f in (data / "streams").glob("*.json"):
            try:
                if json.loads(f.read_text()).get("port") == port:
                    return True
            except (OSError, ValueError):
                pass
        if time.monotonic() >= end:
            return False
        time.sleep(0.2)


def run(children: Children, args, work: Path, out: Path, check) -> bool:
    """The step itself; False when it stopped early."""
    data = Path(args.app_data).resolve() if args.app_data else work / "data"
    workspace = Path(args.workspace).resolve() if args.workspace else work / "ws"
    (work / "user-applets").mkdir(exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    ready, home = work / "ready", work / "home"     # in a folder of this run's own, so never ones from before
    shot, f4_shot = f"step1-live-{args.size}", f"step1-f4-{args.size}"
    pipes = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1,
                 errors="replace")

    # 1. The training script, on the app's app-data folder; it prints its stream link first, once its
    # stream file is there.
    train = None
    if args.url:
        link = args.url
        if not check(link.startswith("http://127.0.0.1:"), f"the stream link is on this machine ({link.split('?')[0]})"):
            return False
    else:
        train = Lines(children.start("train.py", [args.train_python, "-u", str(TRAIN), "--seconds", str(args.seconds)],
                                     stop_signal=signal.SIGINT, env=dict(clean_env(), CALIPER_DATA_DIR=str(data)),
                                     **pipes))
        link = train.find("http://127.0.0.1:", 60)
        if not check(link is not None, "train.py printed its stream link"):
            print("\n".join(train.texts()[-20:]))
            return False
    # caliper.serve() writes its stream file before it prints the link; a given link's is there already.
    by_f4 = listed_stream(data, link, 2 if args.url else 10)
    if not args.url:
        check(by_f4, "train.py's stream file is in the app-data folder, where F4 lists it")

    # 2. The app, hidden. With the stream listed, the input script presses F4 and Enter on it, as a person
    # does; else the app opens on the link. The settle is a wait in seconds (patience), and after the
    # live capture the script goes Home.
    settle_wait = int(args.settle) + 120
    attach = (f"page landing;key f4;wait 20;shot {f4_shot};key enter;" if by_f4 else "")
    steps = (f"{attach}page live;patience {settle_wait};wait-file {ready};patience 0;wait 30;shot {shot};"
             f"key escape;page landing;touch {home}")
    env = app_env(data, work / "user-applets", args.size, steps, out, exit_after=settle_wait + 240)
    argv = [args.caliper, "--hidden", "--api", "--api-port", "0", "--workspace", str(workspace)]
    if not by_f4:
        argv += ["--url", link]
    app = Lines(children.start("the app", argv, env=env, **pipes))
    try:
        check(app.find("[api] Caliper API on ", 60) is not None, "the app started Caliper API")
        check(app.find("[live] attached", 60) is not None,
              "F4 listed the stream and Enter attached the app to it" if by_f4 else "the app attached to the link")
        check(app.find("[ui] page live", 30) is not None, "the app shows the live page")
        if not check((data / "api.json").is_file(), "the app wrote its API file"):
            return False
        api = json.loads((data / "api.json").read_text())

        # 3. The catalog lists the stream's views as live sources, with their
        # shapes once a snapshot arrived and a rate once a few did.
        live: dict[str, dict] = {}
        end = time.monotonic() + 60
        while time.monotonic() < end:
            live = live_sources(api)
            if all(i in live and (live[i].get("rate_hz") or 0) > 0 for i in WANT):
                break
            time.sleep(0.5)
        for i in WANT:
            check(i in live and (live[i].get("rate_hz") or 0) > 0,
                  f"the catalog lists {i} with a rate ({(live.get(i) or {}).get('rate_hz')})")
        attn_shape = (live.get("live:attn") or {}).get("shape")
        check(attn_shape is not None and len(attn_shape) == 3 and attn_shape[0] == 4,
              f"live:attn is one map per layer, {attn_shape}")
        check((live.get("live:attn.entropy") or {}).get("shape") == [4, 4],
              f"live:attn.entropy is (layer, head), {(live.get('live:attn.entropy') or {}).get('shape')}")
        (out / "step1-catalog.json").write_text(json.dumps(live, indent=2) + "\n")

        # 4. Let it train a while, take the capture and go Home: the app stays
        # attached, so the live sources stay in the catalog.
        time.sleep(args.settle)
        ready.touch()
        end = time.monotonic() + 120
        while (app.p.poll() is None and time.monotonic() < end and not home.exists()
               and not any(x.startswith("[ui] stopped") for x in app.texts())):
            time.sleep(0.2)
        stopped = next((x for x in app.texts() if x.startswith("[ui] stopped")), None)
        check(home.exists() and stopped is None,
              "the input script read the live page back and went Home" + (f" ({stopped})" if stopped else ""))
        if home.exists():
            after_home = live_sources(api)
            check(all(i in after_home for i in WANT), "after Home the catalog still lists the live sources")
            check(not any(x.startswith("[live] detached") for x in app.texts()), "Home did not detach the stream")
        for name in ([f4_shot] if by_f4 else []) + [shot]:
            png = out / f"{name}.png"
            check(png.exists() and png.stat().st_size > 10000, f"the capture {png.name} was written")
        code = children.stop(app.p)                 # SIGTERM closes the hidden app as its close button does
        check(code == 0, f"the app closed with 0 ({code})")
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
                    help="seconds of training the live capture waits for after the catalog checks")
    args = ap.parse_args()

    problems = [caliper_problem(args.caliper)]
    if not args.url and not args.train_python:
        problems.append("give --train-python, or --url of a stream that is already running")
    if bool(args.workspace) != bool(args.app_data):
        problems.append("--workspace and --app-data go together")
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
