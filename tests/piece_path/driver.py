#!/usr/bin/env python3
"""The end-to-end check of a piece's derive half, through the command layer, without a draw half.

It starts the hidden app with no script and caliper-mcp as a child over stdio, and lets the model runner start the
small training script of the GPU programs test (tests/gpu_programs/train.py) in a throwaway workspace. Then it
writes a workspace piece with `piece_update`, creates an instance of it with `program_create`, and checks the whole
path: the app sends the instance to the script as PROGRAM, the script's derive pass runs at `caliper.step()` on the
live weight `0.weight` and publishes its resource as RESOURCE, the app keeps the versions in the step history, and
`program_read` returns the elements. The piece computes the norm of each row of the weight, beside the row's index,
so that a wrong layout, an unread tensor or a lost version shows. The driver waits on events and on published
state, never for a fixed time, and runs without a window and without the network.

Exit code 0 passes, 77 skips (no app, no torch, no Metal, no native sender, no producer module), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir> [--limit-s 240]
"""
import argparse
import importlib.util
import json
import math
import os
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

# The client, the process helpers and the waits of the GPU programs test.
_spec = importlib.util.spec_from_file_location("gpu_programs_driver", ROOT / "tests" / "gpu_programs" / "driver.py")
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)
Fail, Skip = gp.Fail, gp.Skip

# A derive half that reads the 64 x 8 weight in place and writes one element for each row.
DERIVE = """@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let rows = shapes.w.z;
    let cols = shapes.w.w;
    if (i >= rows || i >= 64u) {
        return;
    }
    var sum = 0.0;
    for (var j = 0u; j < cols; j = j + 1u) {
        let v = w[i * cols + j];
        sum = sum + v * v;
    }
    norms[i] = Row(i, sqrt(sum) * params.gain);
}
"""
PIECE = {"name": "row_norms",
         "description": "The norm of each row of a matrix",
         "inputs": [{"name": "w", "kind": "tensor", "rank_min": 2, "rank_max": 2}],
         "params": [{"name": "gain", "default": 1.0, "min": 0.0, "max": 4.0}],
         "resources": [{"id": "norms", "kind": "buffer", "struct": "Row",
                        "fields": ["index: u32", "norm: f32"], "count": 64}],
         "derive": {"passes": [{"entry": "measure", "workgroups": [1, 1, 1], "writes": ["norms"]}]},
         "files": {"derive": DERIVE}}
ROWS = 64


def check_elements(read):
    """The 64 rows of the weight, each with its index and a norm in the range of the layer's initialization.
    program_read gives an element as the values of its fields in their order: [index, norm]."""
    els = read.get("elements") or []
    if read.get("count") != ROWS or len(els) != ROWS:
        raise Fail(f"program_read gave {read.get('count')} elements and {len(els)} values, not {ROWS}")
    for i, e in enumerate(els):
        if not isinstance(e, list) or len(e) != 2 or e[0] != i:
            raise Fail(f"element {i} is {e}: the layout or the pass is wrong")
        n = e[1]
        # nn.Linear(8, 64) draws from U(-1/sqrt(8), 1/sqrt(8)): a row's norm is near 0.58 and never 0.
        if not isinstance(n, (int, float)) or not math.isfinite(n) or not 0.05 < n < 5.0:
            raise Fail(f"element {i} has the norm {n}: the pass did not read the weight")


def steps(c, result):
    m = c.mcp
    m.body("spec_new", {"id": "piece_path", "title": "Piece Path"})
    m.body("run_start", {"script": "train.py"})
    c.run = c.until("a run", lambda: next(iter(c.runs()), None), 60)
    c.until("the run is running and attached",
            lambda: (lambda r: r and r["state"] == "running" and r.get("attached"))(c.runs().get(c.run)), 120)
    c.pids.update(gp.pids_of(c.ws / "train.py"))

    # The piece goes into the workspace; the program pins it with its hash and binds the live weight.
    piece_hash = m.body("piece_update", {"piece": PIECE})["hash"]
    result["hash"] = piece_hash
    m.body("program_create", {"id": "norms", "title": "Row Norms",
                              "piece": {"name": "row_norms", "from": "workspace"},
                              "bind": {"w": "live:0.weight"}})
    try:
        c.event("the derive half compiled in the script",
                {"kind": "program", "program": "norms", "change": "compiled", "host": "script", "half": "derive"},
                90)
    except Fail as e:
        seen = [ev for ev in c.events if ev.get("kind") == "program"]
        raise Fail(f"{e}; status {c.status('norms')}; program events {seen}") from e
    st = c.until("norms runs", lambda: (lambda x: x if x and x.get("state") == "running" and
                                         (x.get("runs") or 0) >= 2 else None)(c.status("norms")), 90)
    if st.get("kind") != "piece" or st.get("host") != "script":
        raise Fail(f"the status of norms is not a piece in the script: {st}")
    if st.get("sha256") != piece_hash.removeprefix("sha256:"):
        raise Fail(f"the script runs {st.get('sha256')}, and the pin is {piece_hash}")
    if (st.get("piece") or {}).get("hash") != piece_hash:
        raise Fail(f"the status names the pin {st.get('piece')}")
    result["status"] = {k: st.get(k) for k in ("state", "runs", "every", "overhead_share", "memory_bytes")}

    # The versions reach the step history: program_read gives the newest one's elements.
    def read():
        r = m.call("program_read", {"id": "norms", "resource": "norms", "count": ROWS})
        b = r.get("structuredContent") or {}
        return b if not r.get("isError") and b.get("count") else None
    first = c.until("a version of norms in the step history", read, 60)
    check_elements(first)
    later = c.until("a later version of norms",
                    lambda: (lambda b: b if b and b["step"] > first["step"] else None)(read()), 60)
    check_elements(later)
    if later["kept"]["versions"] < 2:
        raise Fail(f"the history keeps {later['kept']['versions']} versions of norms")
    result["read"] = {"first_step": first["step"], "later_step": later["step"], "kept": later["kept"],
                      "norm_0": later["elements"][0][1]}

    # An older step reads the version that was newest then.
    old = m.body("program_read", {"id": "norms", "resource": "norms", "step": first["step"], "count": 1})
    if old.get("step") is None or old["step"] > first["step"]:
        raise Fail(f"program_read at step {first['step']} gave the version of step {old.get('step')}")

    m.body("run_stop", {"run": c.run})
    c.until("the run stopped", lambda: (c.runs().get(c.run) or {}).get("state") not in ("running", "starting"), 60)


def run(a):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    signal.alarm(a.limit_s)
    work = Path(a.work).resolve()
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).exists():
            raise Skip(f"{exe or 'a program'} is not built")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")
           and not k.startswith("CALIPER_RUN_")}
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    pre = subprocess.run([a.python, "-c", "import numpy, torch, caliper, caliper._netio, caliper._producer"],
                         env=env, capture_output=True, text=True)
    if pre.returncode != 0:
        why = pre.stderr.strip().splitlines()[-1] if pre.stderr.strip() else "an import failed"
        raise Skip(f"torch, numpy, the native network sender or the producer module is missing: {why}")
    gpu = subprocess.run([a.python, "-c", "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 3)"],
                         env=env, capture_output=True, text=True)
    if gpu.returncode != 0:
        raise Skip("this machine has no Metal device for torch")
    if work.exists():
        subprocess.run(["rm", "-rf", str(work)], check=True)
    (work / "data").mkdir(parents=True)
    (work / "user").mkdir()
    ws = work / "workspace"
    ws.mkdir()
    (ws / "train.py").write_text((ROOT / "tests" / "gpu_programs" / "train.py").read_text())
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python})
    # The person's decisions at the cards; a line that finds no card waits and does no harm.
    decisions = work / "approvals.txt"
    decisions.write_text("\n".join(f"approve {c} mcp" for c in ("spec.new", "run.start", "piece.update",
                                                                  "program.create", "run.stop")) + "\n")
    procs, own, c = [], set(), None
    result = {}
    try:
        alog = open(work / "app.log", "w")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"],
                               env=dict(env, CALIPER_TEST_APPROVALS=str(decisions)),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(app)
        lines = []

        def drain():
            for raw in app.stdout:
                alog.write(raw)
                alog.flush()
                lines.append(raw.rstrip("\n"))
        threading.Thread(target=drain, daemon=True).start()
        api_file = work / "data" / "api.json"
        end = time.monotonic() + 90
        while not api_file.exists():
            for line in list(lines):
                mm = re.match(r"\[renderer\] (\S+)", line)
                if mm and mm.group(1) != "metal":
                    raise Skip(f"the renderer is {mm.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = gp.Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        c = gp.Check(mcp, ws, work, work / "captures", HERE)
        own.update(p.pid for p in procs)
        steps(c, result)
        return result
    finally:
        for p in reversed(procs):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
            gp.stop(p)
        if c is not None:   # the app is gone: no child it started may remain
            left = [pid for pid in sorted(c.pids | own) if gp.alive(pid)]
            for pid in left:
                print(f"a child process is left after the app ended: {pid}", file=sys.stderr)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            if left and sys.exc_info()[0] is None:
                raise Fail(f"child processes are left: {left}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app")
    ap.add_argument("--mcp")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit-s", type=int, default=240)
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_piece_path_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_piece_path_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    print("caliper_piece_path_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
