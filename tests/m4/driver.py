#!/usr/bin/env python3
"""The driver of the hosted runs acceptance test and its check script.

It starts the hidden app with no script and caliper-mcp as a child over stdio, copies tests/m4/train.py
and tests/m4/infer.py into a throwaway workspace, and sends tests/m4/transcript.jsonl through the MCP
tools. The model runner of the app starts, forks and stops the scripts. The driver waits on events and on
published state, never for a fixed time. Without a window and without the network. Exit code 0 passes,
77 skips (no app, no torch, no Metal), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir>
                 [--capture-dir <dir>] [--transcript <file>] [--limit-s 280]
"""
import argparse
import array
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "hosted-runs-check", "version": "1"}}
# The decisions the person gives at the cards, through the app's test seam. Each line decides the oldest
# waiting card of its command: two starts and their stops, one fork and the edits of the spec.
APPROVALS = (["approve spec.new mcp", "approve source.add mcp", "approve source.add mcp",
              "approve panel.create mcp"]
             + ["approve run.start mcp"] * 2 + ["approve run.fork mcp"] + ["approve run.stop mcp"] * 3)


class Fail(Exception):
    pass


class Skip(Exception):
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

    def body(self, tool, arguments):
        r = self.call(tool, arguments)
        if r.get("isError"):
            raise Fail(f"{tool} answered an error: {json.dumps(r)[:600]}")
        return r.get("structuredContent", {})


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


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie has no code to run: it counts as ended.
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def pids_of(script_path):
    """The pids of the processes whose command line names the script file."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if str(script_path) in cmd and int(pid) != os.getpid():
            found.append(int(pid))
    return found


class Check:
    """One run of the transcript: the connection, the event cursor and the pids seen."""

    def __init__(self, mcp, ws, capture_dir):
        self.mcp = mcp
        self.ws = ws
        self.capture_dir = capture_dir
        self.cursor = 0
        self.ctx = {}
        self.pids = set()
        self.result = {}

    def sub(self, v):
        if isinstance(v, str):
            for k, x in self.ctx.items():
                v = v.replace("{" + k + "}", x)
            return v
        if isinstance(v, list):
            return [self.sub(i) for i in v]
        if isinstance(v, dict):
            return {self.sub(k): self.sub(i) for k, i in v.items()}
        return v

    def runs(self):
        return {r["id"]: r for r in self.mcp.body("run_list", {}).get("runs", [])}

    def pace(self, seconds):
        """Waits for a run event, at most the time: the pace of a poll of published state."""
        b = self.mcp.body("events_wait", {"cursor": self.cursor, "kinds": ["run"], "timeout_s": seconds})
        self.cursor = b.get("cursor", self.cursor)

    def wait_state(self, run, states, limit=120):
        end = time.monotonic() + limit
        while True:
            r = self.runs().get(run)
            if r and r["state"] in states:
                return r
            if r and r["state"] == "failed":
                raise Fail(f"run {run} failed: {r.get('error')}")
            if time.monotonic() > end:
                raise Fail(f"run {run} is {r and r['state']}, not one of {states}")
            self.pace(2)

    def steps_grow(self, run, limit=120):
        end = time.monotonic() + limit
        first = None
        while True:
            step = self.mcp.body("run_state", {"run": run}).get("step")
            if step is not None:
                if first is None:
                    first = step
                elif step > first:
                    return first, step
            if time.monotonic() > end:
                raise Fail(f"the step of {run} did not grow: {first} then {step}")
            self.pace(1)

    def child_steps(self, run, factor, limit=60):
        """Waits on the step that the run reports, until it is `factor` times the first step seen."""
        end = time.monotonic() + limit
        first = None
        while True:
            step = self.mcp.body("run_state", {"run": run}).get("step")
            if step is not None:
                first = step if first is None else first
                if step >= factor * first:
                    return first, step
            if time.monotonic() > end:
                raise Fail(f"the step of {run} did not reach {factor} times {first}: {step}")
            self.pace(1)

    def loss_gap(self, panel, fork, parent, margin):
        """The fork learns more slowly: at the last step that both lines have, its loss is higher than its
        parent's by the factor `margin`. Its line also spans a third of the plot's width or more."""
        r = self.mcp.call("panel_data", {"panel": panel})
        lines = {l["binding"]: l for l in ((r.get("structuredContent") or {}).get("data") or {}).get("lines", [])}

        def series(binding):
            l = lines.get(binding)
            if not l:
                raise Fail(f"panel {panel} has no line {binding}: {list(lines)}")
            out = []
            for k in ("x", "y"):
                v = array.array("d")
                v.frombytes(base64.b64decode(l[k]["base64"]))
                out.append(list(v))
            return out
        fx, fy = series(fork)
        px, py = series(parent)
        if len(fx) < 2 or len(px) < 2:
            raise Fail(f"panel {panel}: a line has fewer than two points")
        common = min(fx[-1], px[-1])

        def at(xs, ys):
            return ys[min(range(len(xs)), key=lambda i: abs(xs[i] - common))]
        f_loss, p_loss = at(fx, fy), at(px, py)
        lo, hi = min(fx[0], px[0]), max(fx[-1], px[-1])
        share = (fx[-1] - fx[0]) / (hi - lo)
        if share < 1 / 3:
            raise Fail(f"the fork's line spans {share:.2f} of the plot's width, less than a third")
        if f_loss < margin * p_loss:
            raise Fail(f"at step {common:.0f} the fork's loss {f_loss:.3f} is not {margin} times "
                       f"the parent's {p_loss:.3f}")
        return {"step": common, "fork_loss": round(f_loss, 3), "parent_loss": round(p_loss, 3),
                "fork_share_of_width": round(share, 2)}

    def panel_has_lines(self, panel, count, limit=90):
        end = time.monotonic() + limit
        last = None
        while True:
            r = self.mcp.call("panel_data", {"panel": panel})
            d = (r.get("structuredContent") or {}).get("data") or {}
            lines = d.get("lines") or []
            last = [(l.get("binding"), l.get("count")) for l in lines]
            if len(lines) >= count and all((l.get("count") or 0) > 1 for l in lines):
                return last
            if time.monotonic() > end:
                raise Fail(f"panel {panel} has no {count} lines with data: {last}")
            self.pace(1)

    def no_children(self, limit=60):
        end = time.monotonic() + limit
        while True:
            left = [p for p in sorted(self.pids) if alive(p)]
            if not left:
                return
            if time.monotonic() > end:
                raise Fail(f"child processes are left: {left}")
            self.pace(1)

    def step(self, s):
        name = s["step"]
        if "wait_state" in s:
            w = self.sub(s["wait_state"])
            self.result[name] = self.wait_state(w["run"], w["states"])
            return
        if "steps_grow" in s:
            first, now = self.steps_grow(self.sub(s["steps_grow"]["run"]))
            self.result[name] = {"first": first, "later": now}
            return
        if "record_pids" in s:
            found = pids_of(self.ws / s["record_pids"])
            if not found:
                raise Fail(f"{name}: no process runs {s['record_pids']}")
            self.pids.update(found)
            self.result[name] = sorted(self.pids)
            return
        if "steps_factor" in s:
            w = self.sub(s["steps_factor"])
            first, now = self.child_steps(w["run"], w["factor"])
            self.result[name] = {"first": first, "later": now}
            return
        if "loss_gap" in s:
            w = self.sub(s["loss_gap"])
            self.result[name] = self.loss_gap(w["panel"], w["fork"], w["parent"], w["margin"])
            return
        if "panel_has_lines" in s:
            w = s["panel_has_lines"]
            self.result[name] = self.panel_has_lines(w["panel"], w["lines"])
            return
        if "no_children" in s:
            self.no_children()
            return
        if "capture" in s:
            r = self.mcp.call("capture_frame", {})
            if r.get("isError"):
                raise Fail(f"{name}: capture_frame failed: {r}")
            img = next(c for c in r["content"] if c["type"] == "image")
            if self.capture_dir:
                Path(self.capture_dir).mkdir(parents=True, exist_ok=True)
                out = Path(self.capture_dir) / (s["capture"] + ".png")
                out.write_bytes(base64.b64decode(img["data"]))
                self.result["capture"] = str(out)
            return
        if name == "source_b":   # the run that the app does not show
            runs = self.runs()
            shown = next((i for i, r in runs.items() if r.get("attached")), None)
            others = [i for i, r in runs.items() if r.get("managed") and i != shown]
            if shown is None or len(others) != 1:
                raise Fail(f"{name}: expected one attached run and one other, got {json.dumps(runs)[:400]}")
            self.ctx["other"] = others[0]
            self.result["attached"] = shown
        args = self.sub(dict(s.get("arguments", {})))
        body = self.mcp.body(s["tool"], args)
        if "expect_run" in s:
            run = body.get("run", {})
            for k, v in s["expect_run"].items():
                if run.get(k) != v:
                    raise Fail(f"{name}: run {k} is {run.get(k)!r}, expected {v!r}")
            if s["tool"] == "run_fork":
                self.ctx["child"] = run["id"]
        if "expect_runs" in s:
            runs = {r["id"]: r for r in body.get("runs", [])}
            for rid, want in self.sub(s["expect_runs"]).items():
                got = runs.get(rid)
                if not got:
                    raise Fail(f"{name}: run_list lacks {rid}: {list(runs)}")
                for k, v in want.items():
                    if got.get(k) != v:
                        raise Fail(f"{name}: {rid} {k} is {got.get(k)!r}, expected {v!r}")
            self.result[name] = {i: r["state"] for i, r in runs.items()}
        if s.get("expect_lines"):
            if not body.get("lines"):
                raise Fail(f"{name}: run_logs returned no lines")
            self.result[name] = len(body["lines"])


def run(a):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    signal.alarm(a.limit_s)
    work = Path(a.work).resolve()
    if work.exists():
        subprocess.run(["rm", "-rf", str(work)], check=True)
    (work / "data").mkdir(parents=True)
    (work / "user").mkdir()
    ws = work / "workspace"
    ws.mkdir()
    for f in ("train.py", "infer.py"):
        (ws / f).write_text((HERE / f).read_text())
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")
           and not k.startswith("CALIPER_RUN_")}
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python,
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
    check = None
    try:
        app_env = dict(env, CALIPER_TEST_APPROVALS=str(decisions))
        alog = open(work / "app.log", "w")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"], env=app_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(app)
        lines = []

        def drain():
            for raw in app.stdout:
                alog.write(raw)
                alog.flush()
                lines.append(raw.rstrip("\n"))
        import threading
        threading.Thread(target=drain, daemon=True).start()
        api_file = work / "data" / "api.json"
        end = time.monotonic() + 90
        while not api_file.exists():
            for line in list(lines):
                m = re.match(r"\[renderer\] (\S+)", line)
                if m and m.group(1) != "metal":
                    raise Skip(f"the renderer is {m.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        check = Check(mcp, ws, a.capture_dir)
        for s in steps:
            check.step(s)
        return check.result
    finally:
        for p in reversed(procs):
            if isinstance(p, subprocess.Popen):
                try:
                    if p.stdin:
                        p.stdin.close()
                except OSError:
                    pass
                stop(p)
        if check is not None:   # the app is gone: no child it started may remain
            for pid in sorted(check.pids):
                if alive(pid):
                    print(f"a child process is left after the app ended: {pid}", file=sys.stderr)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass


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
        print(f"caliper_m4_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_m4_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=2))
    print("caliper_m4_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
