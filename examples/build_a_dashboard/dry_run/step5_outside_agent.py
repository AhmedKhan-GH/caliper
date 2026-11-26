"""Step 5 of the walk-through in the hidden app: an outside agent through caliper-mcp.

An agent outside the app (outside_agent.py beside this folder) starts
caliper-mcp, speaks MCP to it over stdio, lists the tools, reads the catalog
and the open spec, and adds a panel to the open dashboard. caliper-mcp reads
the app's API file, whose token runs commands as the origin `mcp`, so the app
holds the edit on an approval card ("MCP agent"); the person approves it, and
the panel lands in the dashboard.

    python3 step5_outside_agent.py [--caliper <app>] [--out <folder>] [--work <folder>] [--size 1440x900]
        [--workspace <folder> --app-data <folder>] [--model example-model --ollama http://127.0.0.1:11434]

The options are those of hidden_app.py.
- caliper-mcp is found beside the caliper executable, or under
  `src/builder/mcp/` of its build folder, as a development build has it.
- The dashboard: without --workspace, the one of expected.spec.toml (the one
  step 2 builds by hand), built at start in a new workspace by the app's
  command script (CALIPER_COMMAND_SCRIPT) from the two data files of
  make_data.py and opened. With --workspace and --app-data, as in run_all.py
  --story, the dashboard training-run that the earlier steps built in that
  workspace is opened, and once the edit has landed it is saved with ⌘S in
  build mode, so that step 6 reopens it with the new panel.
- Without `--model` the outside agent is scripted (no model); with it, a model
  that Ollama serves on this machine chooses the calls.

The person's part is played through the hidden app's input script
(CALIPER_TEST_SPEC_UI): it waits for the spec page and captures the dashboard,
waits until the agent's edit is held and the approval strip shows its card,
captures the card, approves it through the hidden app's test-only decisions
file (CALIPER_TEST_APPROVALS, the same path as the card's Approve button), and
captures the dashboard with the new panel. With `--model` the agent may send
other edits first, so a thread here approves each of its cards after the first
has been captured.

Captures are written to `--out` as step5-<what>-<W>x<H>.png (dashboard, card,
landed), W x H the window's size in points. The run's files go into a new
subfolder of --work, or into a temporary folder that is removed at the end.
The script prints the outside agent's lines and one JSON object, and exits with
0 when the edit was approved and landed, 1 when it did not, 2 when something it
needs is missing, and 130 when interrupted. Every process it starts (the app,
the outside agent and through it caliper-mcp) is stopped by its pid when it
ends, fails or is interrupted.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from hidden_app import (APPLET, FAILED, INTERRUPTED, MISSING, OK, Children, Stop, add_options, app_env,
                        caliper_problem, clean_env, data_problem, fixture_commands, saved_spec, work_folder)

HERE = Path(__file__).resolve().parent
APPROVAL = re.compile(r"^\[approval\] (\S+) (\S+) \((\w+)\) (.*)$")


def find_mcp(caliper: Path) -> Path | None:
    """caliper-mcp beside caliper (an installed app), else where a build folder puts it."""
    for candidate in (caliper.parent / "caliper-mcp", caliper.parent / "src" / "builder" / "mcp" / "caliper-mcp"):
        if candidate.is_file():
            return candidate
    return None


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.lock = threading.Lock()

    def follow(self, stream) -> None:
        with self.path.open("w") as out:
            for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                out.write(line + "\n")
                out.flush()
                with self.lock:
                    self.lines.append(line)

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self.lines)

    def has(self, text: str) -> bool:
        return any(text in line for line in self.snapshot())


def wait_for(cond, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_options(p)
    p.add_argument("--model", help="a model that Ollama serves on this machine chooses the agent's calls")
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="with --model: Ollama on this machine")
    p.add_argument("--minutes", type=int, default=10, help="how long the agent may take")
    args = p.parse_args()
    caliper = Path(args.caliper)
    problems = [caliper_problem(caliper), data_problem()]
    if not caliper_problem(caliper) and find_mcp(caliper) is None:
        problems.append(f"no caliper-mcp beside {caliper} or under src/builder/mcp/ of its build folder")
    if bool(args.workspace) != bool(args.app_data):
        problems.append("--workspace and --app-data go together")
    if args.workspace and not saved_spec(args.workspace).is_file():
        problems.append(f"the workspace holds no saved {APPLET}; step 2 saves it there")
    for problem in filter(None, problems):
        print(f"step5: {problem}", file=sys.stderr)
        return MISSING
    with Children() as children:
        try:
            return run(children, args, caliper)
        except Stop as why:
            print(f"step5: interrupted by {why}", flush=True)
            return INTERRUPTED


def run(children: Children, args, caliper: Path) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = work_folder(args, 5)
    story = bool(args.workspace)
    ws = Path(args.workspace).resolve() if story else work / "ws"
    data = Path(args.app_data).resolve() if story else work / "data"
    apps, raw = work / "apps", work / "raw"
    for d in (data, apps, raw):
        d.mkdir(parents=True, exist_ok=True)
    found = saved_spec(ws).read_text() if story else None
    api_file = data / "api.json"
    started_at = time.time()
    decisions, held, client_done = work / "decisions.txt", work / "edit-held", work / "agent-done"
    decisions.touch()

    # The person's part. Scripted, the edit is a panel.create, which the
    # input script approves itself; with a model, the thread below approves.
    approve = (f"append {decisions} approve panel.create mcp" if not args.model else f"touch {work / 'card-captured'}")
    ui = ["page spec", "wait 60", "shot dashboard", f"patience {args.minutes * 60}", f"wait-file {held}",
          "cards 1", "shot card", approve, f"wait-file {client_done}", "wait 60", "shot landed"]
    if story:
        ui += ["key cmd+b", "wait 20", "key cmd+s", "wait 20"]     # saved, for step 6
    ui += ["quit"]
    extra = {"CALIPER_TEST_APPROVALS": str(decisions)}
    if story:
        extra["CALIPER_AUTOLAUNCH"] = f"local.{APPLET}"
    else:
        script = work / "build.jsonl"
        commands = fixture_commands() + [{"name": "applet.open", "args": {"id": APPLET}}]
        script.write_text("".join(json.dumps(c) + "\n" for c in commands))
        extra.update(CALIPER_COMMAND_SCRIPT=str(script), CALIPER_COMMAND_RESULTS=str(work / "build-results.jsonl"))
    env = app_env(data, apps, args.size, ui, raw, exit_after=args.minutes * 60 + 300, extra=extra)
    app = children.start("the app", [str(caliper), "--hidden", "--api", "--api-port", "0", "--workspace", str(ws)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env)
    log = Log(work / "app.log")
    threading.Thread(target=log.follow, args=(app.stderr,), daemon=True).start()
    report: dict = {"agent": "model " + args.model if args.model else "scripted", "story": story, "folder": str(work)}
    agent_lines: list[str] = []
    agent_code = None

    def ready() -> bool:
        written = api_file.exists() and api_file.stat().st_mtime >= started_at - 1
        return written and (log.has("[ui] page spec") if story else log.has("[session] script done"))

    try:
        if not wait_for(lambda: ready() or app.poll() is not None, 120) or not ready():
            report["outcome"] = "the dashboard was not open: " + " | ".join(log.snapshot()[-5:])
            return finish(children, report, app, raw, out, args.size)
        report["opened"] = next((l for l in log.snapshot() if "[session] script done" in l), "from the workspace")

        # Held edits: the first is captured, then each is approved (with a model).
        stop = threading.Event()

        def watch() -> None:
            seen: set[str] = set()
            while not stop.is_set():
                for line in log.snapshot():
                    m = APPROVAL.match(line)
                    if not m or m.group(3) != "mcp" or m.group(4) != "waiting" or m.group(1) in seen:
                        continue
                    seen.add(m.group(1))
                    held.touch()
                    if args.model:
                        wait_for(lambda: (work / "card-captured").exists() or stop.is_set(), args.minutes * 60)
                        time.sleep(1.0)
                        with decisions.open("a") as f:
                            f.write(f"approve {m.group(2)} mcp\n")
                time.sleep(0.2)

        threading.Thread(target=watch, daemon=True).start()
        argv = [sys.executable, str(HERE.parent / "outside_agent.py"), "--mcp", str(find_mcp(caliper)),
                "--api-file", str(api_file)]
        if args.model:
            argv += ["--model", args.model, "--ollama", args.ollama]
        # SIGINT lets the outside agent close caliper-mcp itself.
        agent = children.start("outside_agent.py", argv, stop_signal=signal.SIGINT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=clean_env())
        for raw_line in agent.stdout:
            line = raw_line.decode("utf-8", "replace").rstrip("\n")
            agent_lines.append(line)
            print(f"  agent: {line}", flush=True)
        agent_code = agent.wait()
        client_done.touch()
        stop.set()
        app.wait(timeout=180)
    except subprocess.TimeoutExpired:
        children.stop(app)
    report["agent_exit"] = agent_code
    report["agent_lines"] = len(agent_lines)
    report["approvals"] = [l for l in log.snapshot() if l.startswith("[approval]")]
    landed = agent_code == 0 and any(l.endswith("(mcp) approved") for l in report["approvals"])
    if story and landed:
        # The edit saved with the dashboard, for step 6: one panel more than the spec held before.
        now = saved_spec(ws).read_text() if saved_spec(ws).is_file() else ""
        report["saved"] = now.count("[[panel]]") == found.count("[[panel]]") + 1
        landed = report["saved"]
    report["outcome"] = "the edit was approved and landed" if landed else "the edit did not land"
    return finish(children, report, app, raw, out, args.size)


def finish(children: Children, report: dict, app: subprocess.Popen, raw: Path, out: Path, size: str) -> int:
    children.stop(app)
    report["app_exit"] = app.returncode
    captures = []
    for name in ("dashboard", "card", "landed"):
        src = raw / f"{name}.png"
        if src.exists():
            dst = out / f"step5-{name}-{size}.png"
            shutil.copyfile(src, dst)
            captures.append(str(dst))
    report["captures"] = captures
    print(json.dumps(report), flush=True)
    return OK if report.get("outcome") == "the edit was approved and landed" else FAILED


if __name__ == "__main__":
    sys.exit(main())
