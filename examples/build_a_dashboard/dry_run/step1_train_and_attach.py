"""Dry run of step 1: train.py streams, and the hidden Caliper app attaches to it.

    python examples/build_a_dashboard/dry_run/step1_train_and_attach.py \\
        --python <a Python with torch and caliper> [--caliper cmake-build-release/caliper] \\
        [--out build/dry_run] [--size 1440x900] [--seconds 90]

It does what a person does in step 1, without a window: it starts train.py,
reads the stream link that the script prints, and starts the app hidden with
that link on its command line (`caliper --hidden --url <link>`), which opens
the live page on the stream as F4 and the pasted link do. It then checks, over
Caliper API, that the catalog lists the stream's views as live sources with
their shapes and rates, reads the live page back into
<out>/step1-live-<W>x<H>.png, writes the catalog's live entries to
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
    shot = f"step1-live-{args.size}"
    env = dict(os.environ, CALIPER_AGENT_OFFLINE="1")
    failed: list[str] = []

    def check(ok: bool, what: str) -> bool:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            failed.append(what)
        return ok

    # 1. The training script; it prints its stream link first.
    train = Proc([args.python, "-u", str(TRAIN), "--seconds", str(args.seconds)], env)
    link = train.find("http://127.0.0.1:", 60)
    if not check(link is not None, "train.py printed its stream link"):
        print("\n".join(train.lines[-20:]))
        train.stop(signal.SIGINT)
        return 1

    # 2. The app, hidden, attached to the stream from its command line. The
    # input script reads the live page back once the checks below have passed.
    app_env = dict(env, CALIPER_DATA_DIR=str(work / "data"),
                   CALIPER_USER_APPLETS_DIR=str(work / "user-applets"),
                   CALIPER_WINDOW_SIZE=args.size,
                   CALIPER_TEST_SPEC_UI=f"page live;wait-file {ready};wait 30;shot {shot};quit",
                   CALIPER_TEST_SPEC_UI_SHOTS=str(out))
    app = Proc([args.caliper, "--hidden", "--url", link, "--api", "--api-port", "0",
                "--workspace", str(work / "ws")], app_env)
    try:
        check(app.find("[api] Caliper API on ", 60) is not None, "the app started Caliper API")
        check(app.find("[live] attached", 30) is not None, "the app attached to the stream")
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

        # 4. Let it train a while, then take the capture and quit.
        time.sleep(args.settle)
        ready.touch()
        code = app.p.wait(120)
        check(app.find("[ui] done", 5) is not None, "the input script read the live page back")
        check(code == 0, f"the app quit with 0 ({code})")
        png = out / f"{shot}.png"
        check(png.exists() and png.stat().st_size > 10000, f"the capture {png.name} was written")
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
