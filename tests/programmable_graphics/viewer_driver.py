"""The browser's dashboard viewer end to end: the real spec page of the hidden app draws the programs of a live run.

The hidden app and caliper-mcp start as in driver.py, the model runner starts tests/acceptance/arc_tower_train.py, and a
dashboard gets two scene panels that hold programs alone: the arcs of attention_arcs@1 pinned from the WORKSPACE, and
the attention grid of attention_grid@1 written INLINE into the spec. The dashboard is saved, the run trains until the
arcs are published, and tests/programmable_graphics/viewer_page.mjs opens the viewer link of the app's API file, which
carries the viewer token, in headless Chromium, with a WebGPU adapter and with SwiftShader. In the page, in the same task
as each frame of the module, it counts the pixels of each program panel that differ from the panel's background, and
reads the page's list of programs. Each program must be running, and each panel must hold pixels of its program.

Then the app opens another spec, and the page of the saved dashboard must say that the app does not have it open: the
banner, and the state of each panel.

Waits on events and published state only. Exit 0 passes, 77 skips (no app, no torch or Metal, no Node with the web
package, no WebAssembly views), 1 fails.

usage: viewer_driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir> [--projects webgpu,swiftshader]
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77
_spec = importlib.util.spec_from_file_location("programmable_graphics_driver", HERE / "driver.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)
gp, Fail, Skip = d.gp, d.Fail, d.Skip

# The pixels that a panel's program draws at least: the arcs and their labels, the grid's cells and labels.
MIN_PIXELS = 500
NOT_OPEN = "The app does not have this dashboard open. It shows data when the dashboard is open in the app."


def page(url, project, mode):
    p = subprocess.run(["node", str(HERE / "viewer_page.mjs"), "--web", str(ROOT / "web"), "--url", url,
                        "--project", project, "--mode", mode], capture_output=True, text=True, timeout=300)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise Fail(f"viewer_page.mjs printed no result (exit {p.returncode}): {p.stderr[-800:]}")
    return json.loads(lines[-1])


def build(r, c):
    r.step("start the run and pause it")
    r.call("spec_new", {"id": "viewer_check", "title": "Viewer Check"})
    r.call("run_start", {"script": d.SCRIPT})
    c.run = c.until("a run", lambda: next(iter(c.runs()), None), 60)
    c.until("the run is running and attached",
            lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(c.runs().get(c.run)), 120)
    c.pids.update(gp.pids_of(c.ws / d.SCRIPT))
    c.until("the first step of the run", lambda: r.run_step(c.run) >= 1, 60)
    r.call("run_pause", {"op": "pause", "run": c.run})
    r.step("a workspace piece and an inline piece in two scene panels, saved")
    arcs = d.piece_args(ROOT / "tests" / "acceptance" / "arc_tower" / "attention_arcs")
    grid = d.piece_args(ROOT / "tests" / "acceptance" / "arc_tower" / "attention_grid")
    r.call("piece_update", {"piece": arcs})
    for pid, title in (("tower", "Arcs"), ("pattern", "Grid")):
        r.call("panel_create", {"id": pid, "primitive": "scene", "title": title, "bind": {"scene": "live:model"},
                                "options": {"exclude": ["**"]}})
    binds = {"attn0": "live:blocks.0.attn.weights.act", "attn1": "live:blocks.1.attn.weights.act"}
    r.call("program_create", {"id": "tower_arcs", "title": "Arcs", "piece": {"name": "attention_arcs", "version": 1,
           "from": "workspace"}, "panel": "tower", "bind": binds, "params": {"threshold": 0.05}})
    r.call("program_create", {"id": "head_pattern", "title": "Grid", "inline": grid, "panel": "pattern", "bind": binds,
           "params": {"layer": 1, "head": 0, "gain": 2.0}})
    for pid in ("tower", "pattern"):
        r.call("panel_patch", {"id": pid, "patch": {"bind": {"scene": None}, "options": {"exclude": None}}})
    spec = r.call("spec_get", {})["json"]
    froms = {p["id"]: p["piece"]["from"] for p in spec.get("program", [])}
    if froms != {"tower_arcs": "workspace", "head_pattern": "inline"}:
        raise Fail(f"the programs pin {froms}, not a workspace piece and an inline piece")
    r.call("applet_save", {})
    r.step("both programs compile, the run goes on")
    r.call("run_pause", {"op": "resume", "run": c.run})
    for prog in ("tower_arcs", "head_pattern"):
        for half in ("derive", "draw"):
            c.event(f"{prog} {half} compiled", {"kind": "program", "program": prog, "change": "compiled", "half": half}, 120)
    c.until("an arc version", lambda: (lambda b: b if b.get("count") else None)(r.read("arcs", 1024)), 90)


def run(a):
    work = Path(a.work).resolve()
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).exists():
            raise Skip(f"{exe or 'a program'} is not built")
    if not shutil.which("node") or not (ROOT / "web" / "node_modules" / "@playwright" / "test").exists():
        raise Skip("needs Node and the web package's dependencies (npm ci in web/) for headless Chromium")
    views = ROOT / "python" / "caliper" / "_web"
    if not all((views / f).exists() for f in ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")):
        raise Skip(f"needs the WebAssembly views in {views} (scripts/build_wasm.py or scripts/fetch_wasm.py)")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")
           and not k.startswith("CALIPER_RUN_")}
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    pre = subprocess.run([a.python, "-c", "import numpy, torch, caliper, caliper._netio, caliper._producer"],
                         env=env, capture_output=True, text=True)
    if pre.returncode != 0:
        raise Skip("torch, numpy, the native network sender or the producer module is missing")
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
    (ws / d.SCRIPT).write_text((ROOT / "tests" / "acceptance" / d.SCRIPT).read_text())
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python})
    approvals = work / "approvals.txt"
    approvals.write_text("")
    procs, c, result = [], None, {}
    transcript = open(work / "transcript.jsonl", "w")
    try:
        alog = open(work / "app.log", "w")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"],
                               env=dict(env, CALIPER_TEST_APPROVALS=str(approvals)), stdout=alog,
                               stderr=subprocess.STDOUT, text=True)
        procs.append(app)
        api_file = work / "data" / "api.json"
        end = time.monotonic() + 90
        while not api_file.exists():
            if app.poll() is not None or time.monotonic() > end:
                raise Fail("the app ended or wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = gp.Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        c = gp.Check(mcp, ws, work, work / "captures", HERE)
        c.app, c.mcp_path = a.app, a.mcp
        r = d.Run(c, approvals, transcript)
        build(r, c)
        url = json.loads(api_file.read_text())["spec_url"] + "&applet=viewer_check"
        for project in a.projects.split(","):
            r.step(f"the page draws both programs, {project}")
            got = page(url, project, "programs")
            result[project] = {k: got.get(k) for k in ("counts", "programs", "error")}
            if got.get("error"):
                raise Fail(f"{project}: {got['error']}; console {got.get('console')}")
            states = {p["id"]: p["module"] for m in got["programs"].values() for p in m["programs"]}
            if states != {"tower_arcs": "running", "head_pattern": "running"}:
                raise Fail(f"{project}: the page's programs are {got['programs']}, not both running; console {got.get('console')}")
            for panel in ("tower", "pattern"):
                n = (got.get("counts") or {}).get(panel, 0)
                if n < MIN_PIXELS:
                    raise Fail(f"{project}: the panel {panel} holds {n} pixels of its program, under {MIN_PIXELS}")
        r.step("a saved dashboard that the app does not have open says so")
        r.call("spec_new", {"id": "elsewhere", "title": "Elsewhere"})
        got = page(url, a.projects.split(",")[0], "not-open")
        result["not_open"] = got
        if got.get("error") or got.get("banner") != NOT_OPEN:
            raise Fail(f"the page of a dashboard that is not open shows {got}")
        if any(p["state"] != "Dashboard not open in the app" for p in got["panels"]):
            raise Fail(f"a panel of a dashboard that is not open shows another state: {got['panels']}")
        return result
    finally:
        transcript.close()
        for p in reversed(procs):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
            gp.stop(p)
        if c is not None:
            left = [pid for pid in sorted(c.pids) if gp.alive(pid)]
            if left and sys.exc_info()[0] is None:
                raise Fail(f"child processes are left: {left}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app")
    ap.add_argument("--mcp")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work", required=True)
    ap.add_argument("--projects", default="webgpu,swiftshader")
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_dashboard_viewer_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_dashboard_viewer_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=1, default=str)[:4000])
    print("caliper_dashboard_viewer_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
