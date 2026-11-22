#!/usr/bin/env python3
"""The driver of the M1 acceptance test and the founder check.

It starts tests/m1/train.py, starts the hidden app on the script's stream link and caliper-mcp
as a child over stdio, and sends tests/m1/transcript.jsonl through the MCP tools. It waits on the
script's output, on events and on published state, never for a fixed time. Without a window and
without the network. Exit code 0 passes, 77 skips (no app, no torch, no Metal), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir>
                 [--capture-dir <dir>] [--transcript <file>] [--limit-s 280]
"""
import argparse
import base64
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "m1-check", "version": "1"}}
# The decisions the person gives at the cards, through the app's test seam.
APPROVALS = ["approve spec.new mcp", "approve panel.create mcp", "approve control.create mcp",
             "approve run.set_param mcp"]
# The scripted input of the hidden app: a click in the middle of the model's scene view. It waits
# for the frames that the MCP steps need first, and for the view to be drawn.
UI_SCRIPT = "wait 300; pick model 0.5 0.5; wait 30"


class Fail(Exception):
    pass


class Skip(Exception):
    pass


class Lines:
    """A child's output on a thread: every line kept, and a queue for waits."""

    def __init__(self, stream, path):
        self.q = queue.Queue()
        self.lines = []
        self.path = path
        self.t = threading.Thread(target=self._pump, args=(stream,), daemon=True)
        self.t.start()

    def _pump(self, stream):
        with open(self.path, "w") as f:
            for raw in stream:
                line = raw.rstrip("\n")
                f.write(line + "\n")
                f.flush()
                self.lines.append(line)
                self.q.put(line)

    def wait_for(self, pattern, timeout, alive):
        """The first line that matches the regular expression, from the lines read so far."""
        rx = re.compile(pattern)
        end = time.monotonic() + timeout
        seen = 0
        while True:
            while seen < len(self.lines):
                m = rx.search(self.lines[seen])
                seen += 1
                if m:
                    return m
            left = end - time.monotonic()
            if left <= 0:
                raise Fail(f"no output line matches {pattern!r}; last lines: {self.lines[-5:]}")
            if not alive():
                if seen >= len(self.lines):
                    raise Fail(f"the process ended before a line matched {pattern!r}: {self.lines[-5:]}")
                continue
            try:
                self.q.get(timeout=min(left, 0.5))
            except queue.Empty:
                pass


class Mcp:
    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env=env)
        self.n = 0

    def request(self, method, params):
        self.n += 1
        params = dict(params, _meta=META)
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise Fail("caliper-mcp closed its output")
            m = json.loads(line)
            if m.get("id") == self.n and "method" not in m:
                return m

    def call(self, tool, arguments):
        r = self.request("tools/call", {"name": tool, "arguments": arguments})
        if "result" not in r:
            raise Fail(f"{tool}: {r}")
        return r["result"]


def stop(p):
    """Stops a child by its own pid: SIGTERM, then SIGKILL after 20 seconds."""
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def matches(event, want):
    return all(event.get(k) == v for k, v in want.items())


def run(a):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    signal.alarm(a.limit_s)
    work = Path(a.work).resolve()
    if work.exists():
        subprocess.run(["rm", "-rf", str(work)], check=True)
    (work / "data").mkdir(parents=True)
    (work / "user").mkdir()
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")}
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "PYTHONPATH": str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")})
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).exists():
            raise Skip(f"{exe or 'a program'} is not built")
    pre = subprocess.run([a.python, "-c", "import torch, caliper, caliper._netio"], env=env,
                         capture_output=True, text=True)
    if pre.returncode != 0:
        raise Skip("torch or the native network sender is missing")
    decisions = work / "approvals.txt"
    decisions.write_text("\n".join(APPROVALS) + "\n")
    steps = [json.loads(l) for l in Path(a.transcript).read_text().splitlines() if l.strip()]
    procs = []
    result = {}
    try:
        script = subprocess.Popen([a.python, "-u", str(HERE / "train.py")], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        procs.append(script)
        sout = Lines(script.stdout, work / "script.log")
        link = sout.wait_for(r"^(https?://\S+)$", 120, lambda: script.poll() is None).group(1)
        sout.wait_for(r"^lr \S+ step \d+$", 120, lambda: script.poll() is None)   # the loop runs

        app_env = dict(env, CALIPER_TEST_APPROVALS=str(decisions), CALIPER_TEST_UI=UI_SCRIPT)
        app = subprocess.Popen([a.app, "--hidden", "--url", link, "--api", "--api-port", "0"], env=app_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(app)
        aout = Lines(app.stdout, work / "app.log")
        api_file = work / "data" / "api.json"
        end = time.monotonic() + 90
        while not api_file.exists():
            for line in aout.lines:
                m = re.match(r"\[renderer\] (\S+)", line)
                if m and m.group(1) != "metal":
                    raise Skip(f"the renderer is {m.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(aout.lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens

        mcp = Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        cursor = 0
        for s in steps:
            name = s["step"]
            if "capture" in s:
                r = mcp.call("capture_frame", {})
                if r.get("isError"):
                    raise Fail(f"{name}: capture_frame failed: {r}")
                img = next(c for c in r["content"] if c["type"] == "image")
                if a.capture_dir:
                    Path(a.capture_dir).mkdir(parents=True, exist_ok=True)
                    out = Path(a.capture_dir) / (s["capture"] + ".png")
                    out.write_bytes(base64.b64decode(img["data"]))
                    result["capture"] = str(out)
                continue
            if "script_prints" in s:
                want = s["script_prints"]["lr"]
                m = sout.wait_for(rf"^lr {want:g} step (\d+)$", 60,
                                  lambda: script.poll() is None)
                result["lr_at_next_step"] = want
                result["lr_step"] = int(m.group(1))
                continue
            args = dict(s.get("arguments", {}))
            if s["tool"] == "events_wait" and "cursor" not in args:
                args["cursor"] = cursor
            if "approve" in s:
                pass   # the decision line is already in the decisions file, read when the card waits
            r = mcp.call(s["tool"], args)
            if r.get("isError"):
                raise Fail(f"{name}: {s['tool']} answered an error: {json.dumps(r)[:600]}")
            body = r.get("structuredContent", {})
            for k, v in s.get("expect", {}).items():
                if body.get(k) != v:
                    raise Fail(f"{name}: {k} is {body.get(k)!r}, expected {v!r}")
            if s["tool"] == "events_wait":
                found = None
                end = time.monotonic() + 60
                while True:
                    cursor = body["cursor"]
                    found = next((e for e in body["events"] if matches(e, s["until"])), None)
                    if found or time.monotonic() > end:
                        break
                    args["cursor"] = cursor
                    body = mcp.call("events_wait", args).get("structuredContent", {})
                if not found:
                    raise Fail(f"{name}: no event matches {s['until']}")
                result[name] = found
        if "pick" in result:
            for k in ("entity", "primitive", "index", "row_id", "position"):
                if k not in result["pick"]:
                    raise Fail(f"the pick event lacks {k}: {result['pick']}")
        return result
    finally:
        for p in reversed(procs):
            if isinstance(p, subprocess.Popen):
                try:
                    if p.stdin:
                        p.stdin.close()
                except OSError:
                    pass
                stop(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app")
    ap.add_argument("--mcp")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work", required=True)
    ap.add_argument("--capture-dir")
    ap.add_argument("--transcript", default=str(HERE / "transcript.jsonl"))
    ap.add_argument("--limit-s", type=int, default=280)
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_m1_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_m1_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=2))
    print("caliper_m1_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
