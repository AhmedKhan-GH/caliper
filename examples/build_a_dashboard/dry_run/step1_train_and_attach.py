"""Dry run of step 1: train.py streams, and the hidden Caliper app attaches to it.

    python examples/build_a_dashboard/dry_run/step1_train_and_attach.py \\
        --python <a Python with torch and caliper> [--caliper cmake-build-release/caliper] \\
        [--out build/dry_run] [--size 1440x900] [--seconds 90]

It does what a person does in step 1, without a window: it starts train.py,
waits for the stream link that the script prints, and starts the app hidden
with the same app-data folder, so that the app lists the script's stream from
the stream file that caliper.serve() writes. Through the app's input driver it
presses F4 on the landing screen, reads the session list back into
<out>/step1-f4-<W>x<H>.png and presses Enter on the stream, which opens its
live page. It then checks, over Caliper API, that the catalog lists the
stream's views as live sources with their shapes and rates, reads the live
page back into <out>/step1-live-<W>x<H>.png, goes Home and checks that the app
stays attached and the live sources stay, writes the catalog's live entries to
<out>/step1-catalog.json, and stops both. It prints one line per check and
exits with 0 when every check passed.

It needs nothing but the standard library itself; the Python given with
--python runs train.py. It sets CALIPER_AGENT_OFFLINE=1, keeps the app's data
and workspace in a temporary folder, and never shows a window.
"""
from __future__ import annotations

import argparse
import json
import os
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


class Proc:
    """A child process whose output is gathered by a thread, line by line."""

    def __init__(self, argv: list[str], env: dict[str, str]):
        self.lines: list[str] = []
        self.p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stdout:
            self.lines.append(line.rstrip("\n"))

    def find(self, prefix: str, timeout: float) -> str | None:
        """The first line that starts with prefix, waiting up to timeout seconds."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for line in list(self.lines):
                if line.startswith(prefix):
                    return line
            if self.p.poll() is not None and not any(x.startswith(prefix) for x in self.lines):
                time.sleep(0.2)                     # the reader may still hold the last lines
                return next((x for x in self.lines if x.startswith(prefix)), None)
            time.sleep(0.1)
        return None

    def stop(self, sig: int, timeout: float = 30.0) -> int | None:
        if self.p.poll() is None:
            self.p.send_signal(sig)
        try:
            return self.p.wait(timeout)
        except subprocess.TimeoutExpired:
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
    ap.add_argument("--python", required=True, help="the Python that runs train.py (torch and caliper)")
    ap.add_argument("--caliper", default=str(ROOT / "cmake-build-release" / "caliper"),
                    help="the caliper executable (default: cmake-build-release/caliper)")
    ap.add_argument("--out", default=str(ROOT / "build" / "dry_run"), help="where the capture goes")
    ap.add_argument("--size", default="1440x900", help="the window's size in points")
    ap.add_argument("--seconds", type=float, default=90.0, help="how long train.py trains")
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds of training the capture waits for after the attach")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="caliper-step1-"))
    (work / "data").mkdir()
    (work / "user-applets").mkdir()
    ready = work / "ready"
    home = work / "home"
    shot = f"step1-live-{args.size}"
    f4_shot = f"step1-f4-{args.size}"
    env = dict(os.environ, CALIPER_AGENT_OFFLINE="1")
    failed: list[str] = []

    def check(ok: bool, what: str) -> bool:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            failed.append(what)
        return ok

    # 1. The training script; it prints its stream link first, once its stream
    # file is in the app-data folder the app reads too.
    train = Proc([args.python, "-u", str(TRAIN), "--seconds", str(args.seconds)],
                 dict(env, CALIPER_DATA_DIR=str(work / "data")))
    link = train.find("http://127.0.0.1:", 60)
    if not check(link is not None, "train.py printed its stream link"):
        print("\n".join(train.lines[-20:]))
        train.stop(signal.SIGINT)
        return 1

    # 2. The app, hidden, on the same app-data folder. The input script presses
    # F4 and Enter on the listed stream, reads the live page back once the
    # checks below have passed, and goes Home.
    app_env = dict(env, CALIPER_DATA_DIR=str(work / "data"),
                   CALIPER_USER_APPLETS_DIR=str(work / "user-applets"),
                   CALIPER_WINDOW_SIZE=args.size,
                   CALIPER_TEST_SPEC_UI=(f"page landing;key f4;wait 20;shot {f4_shot};key enter;page live;"
                                         f"wait-file {ready};wait 30;shot {shot};key escape;page landing;"
                                         f"touch {home}"),
                   CALIPER_TEST_SPEC_UI_SHOTS=str(out))
    app = Proc([args.caliper, "--hidden", "--api", "--api-port", "0",
                "--workspace", str(work / "ws")], app_env)
    try:
        check(app.find("[api] Caliper API on ", 60) is not None, "the app started Caliper API")
        check(app.find("[live] attached", 60) is not None, "F4 listed the stream and Enter attached the app to it")
        api = json.loads((work / "data" / "api.json").read_text())

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

        # 4. Let it train a while, take the capture and go Home: the app stays
        # attached, so the live sources stay in the catalog.
        time.sleep(args.settle)
        ready.touch()
        check(app.find("[ui] done", 120) is not None, "the input script read the live page back and went Home")
        listed = command(api["port"], api["token"], "catalog.list", {"kind": "live"})
        after_home = {e["id"] for e in listed.get("result", {}).get("entries", [])}
        check(all(i in after_home for i in want), "after Home the catalog still lists the live sources")
        check(not any(x.startswith("[live] detached") for x in app.lines), "Home did not detach the stream")
        for name in (f4_shot, shot):
            png = out / f"{name}.png"
            check(png.exists() and png.stat().st_size > 10000, f"the capture {png.name} was written")
        code = app.stop(signal.SIGTERM)
        check(code == 0, f"the app quit with 0 ({code})")
    finally:
        app.stop(signal.SIGTERM)
        train.stop(signal.SIGINT)
    for line in train.lines:
        if line.startswith(("step ", "done", "stopped", "training on")):
            print("train.py: " + line)
    print(("dry run of step 1 passed" if not failed else f"{len(failed)} checks failed") + f"; files in {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
