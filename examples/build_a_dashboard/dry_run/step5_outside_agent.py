"""Step 5 of the walk-through in the hidden app: an outside agent through caliper-mcp.

An agent outside the app (outside_agent.py beside this folder) starts
caliper-mcp, speaks MCP to it over stdio, lists the tools, reads the catalog
and the open spec, and adds a panel to the open dashboard. caliper-mcp reads
the app's API file, whose token runs commands as the origin `mcp`, so the app
holds the edit on an approval card ("MCP agent"); the person approves it, and
the panel lands in the dashboard.

    python3 step5_outside_agent.py --build <build folder> --out <folder> [--work <folder>] [--size 1440x900]
        [--model qwen3.8:27b --ollama http://127.0.0.1:11434]

- `--build` is a build folder of the repository with `caliper` and `caliper-mcp`
  (beside it, or under `src/builder/mcp/` as a development build has it).
- The dashboard is built at start by the app's command script
  (CALIPER_COMMAND_SCRIPT) from `data/run_metrics.parquet` beside this folder
  when it is there, else from a small CSV file of the same columns that the
  script writes: one series panel of the loss and the accuracy by step.
- Without `--model` the outside agent is scripted (no model); with it, a model
  that Ollama serves on this machine chooses the calls.

The person's part is played through the hidden app's input script
(CALIPER_TEST_SPEC_UI): it waits for the spec page and captures the dashboard,
waits until the agent's edit is held and the approval strip shows its card,
captures the card, approves it through the
hidden app's test-only decisions file (CALIPER_TEST_APPROVALS, the same path as
the card's Approve button), and captures the dashboard with the new panel. With
`--model` the agent may send other edits first, so a thread here approves each
of its cards after the first has been captured.

Captures are written to `--out` as step5-<what>-<W>x<H>.png (dashboard, card,
landed). The script prints the outside agent's lines and one JSON object, and
exits with 0 when the edit was approved and landed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPROVAL = re.compile(r"^\[approval\] (\S+) (\S+) \((\w+)\) (.*)$")


# ---- the processes this script starts ---------------------------------------------------------
# Every child is registered when it starts and stopped by its pid when the
# script ends, however it ends: normally, on an error, or on SIGINT or SIGTERM.
# The hidden app also has CALIPER_EXIT_AFTER as a backstop.
CHILDREN: list[subprocess.Popen] = []


def spawn(argv: list[str], **kwargs) -> subprocess.Popen:
    proc = subprocess.Popen(argv, **kwargs)
    CHILDREN.append(proc)
    return proc


def stop_children() -> None:
    for proc in reversed(CHILDREN):
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 15
    for proc in reversed(CHILDREN):
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def on_signal(signum, frame) -> None:
    stop_children()
    sys.exit(128 + signum)


def new_folder(parent: str | None, prefix: str) -> Path:
    """A new, empty folder of this script's own: inside `parent` when given (the
    caller's folder, whose contents are left alone), else a temporary one."""
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    return Path(tempfile.mkdtemp(prefix="caliper-" + prefix))


def find_mcp(build: Path) -> Path:
    """caliper-mcp beside caliper (an installed app), else where a build folder puts it."""
    for candidate in (build / "caliper-mcp", build / "src" / "builder" / "mcp" / "caliper-mcp"):
        if candidate.is_file():
            return candidate
    return build / "caliper-mcp"


def env_for_children() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "CALIPER_MODEL_API_KEY", "CALIPER_TEST_", "CALIPER_API"))}
    env["CALIPER_AGENT_OFFLINE"] = "1"
    return env


def write_data(ws: Path) -> str:
    """The run's metrics in the workspace; the file's path relative to it."""
    (ws / "data").mkdir(parents=True, exist_ok=True)
    parquet = HERE.parent / "data" / "run_metrics.parquet"
    if parquet.exists():
        shutil.copyfile(parquet, ws / "data" / "run_metrics.parquet")
        return "data/run_metrics.parquet"
    with (ws / "data" / "run_metrics.csv").open("w") as f:
        f.write("step,loss,accuracy,lr\n")
        for i in range(300):
            loss = 0.2 + 2.0 * math.exp(-i / 60.0)
            accuracy = 0.1 + 0.85 * (1.0 - math.exp(-i / 50.0))
            lr = 3e-4 * min(1.0, (i + 1) / 30.0) * (0.5 + 0.5 * math.cos(math.pi * i / 300.0))
            f.write(f"{i},{loss:.6f},{accuracy:.6f},{lr:.8f}\n")
    return "data/run_metrics.csv"


def build_script(path: Path, data_file: str) -> None:
    commands = [
        {"name": "data.import", "args": {"paths": [data_file], "run": True}},
        {"name": "spec.new", "args": {"id": "training-run", "title": "Training run"}},
        {"name": "source.add", "args": {"id": "metrics", "catalog": "table:run_metrics"}},
        {"name": "panel.create", "args": {"id": "curves", "primitive": "series", "title": "Loss and accuracy",
                                          "bind": {"x": "metrics.step", "y": ["metrics.loss", "metrics.accuracy"]}}},
        {"name": "applet.save", "args": {"overwrite": True}},
        {"name": "applet.open", "args": {"id": "training-run"}},
    ]
    path.write_text("".join(json.dumps(c) + "\n" for c in commands))


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
    p.add_argument("--build", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", help="a folder in which the run makes a new folder of its own for its files "
                                  "(default: a new temporary folder); nothing in it is removed")
    p.add_argument("--size", default="1440x900")
    p.add_argument("--model", help="a model that Ollama serves on this machine chooses the agent's calls")
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="with --model: Ollama on this machine")
    p.add_argument("--minutes", type=int, default=10, help="how long the agent may take")
    args = p.parse_args()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        return run(args)
    finally:
        stop_children()


def run(args) -> int:
    build, out = Path(args.build), Path(args.out)
    work = new_folder(args.work, "step5-")
    ws, data, apps, raw = work / "ws", work / "data", work / "apps", work / "raw"
    for d in (ws, data, apps, raw, out):
        d.mkdir(parents=True, exist_ok=True)
    decisions, held, client_done = work / "decisions.txt", work / "edit-held", work / "agent-done"
    decisions.touch()
    build_script(work / "build.jsonl", write_data(ws))

    # The person's part. Scripted, the edit is a panel.create, which the
    # input script approves itself; with a model, the thread below approves.
    approve = (f"append {decisions} approve panel.create mcp" if not args.model else f"touch {work / 'card-captured'}")
    ui = "; ".join(["page spec", "wait 60", "shot dashboard", f"patience {args.minutes * 60}", f"wait-file {held}",
                    "cards 1", "shot card", approve, f"wait-file {client_done}", "wait 60", "shot landed", "quit"])
    env = env_for_children()
    env.update({"CALIPER_DATA_DIR": str(data), "CALIPER_USER_APPLETS_DIR": str(apps), "CALIPER_WINDOW_SIZE": args.size,
                "CALIPER_COMMAND_SCRIPT": str(work / "build.jsonl"),
                "CALIPER_COMMAND_RESULTS": str(work / "build-results.jsonl"),
                "CALIPER_TEST_APPROVALS": str(decisions), "CALIPER_TEST_SPEC_UI": ui,
                "CALIPER_TEST_SPEC_UI_SHOTS": str(raw), "CALIPER_EXIT_AFTER": str(args.minutes * 60 + 300)})
    app = spawn([str(build / "caliper"), "--hidden", "--api", "--api-port", "0", "--workspace", str(ws)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    log = Log(work / "app.log")
    threading.Thread(target=log.follow, args=(app.stderr,), daemon=True).start()
    report: dict = {"agent": "model " + args.model if args.model else "scripted", "folder": str(work)}
    agent_lines: list[str] = []
    agent_code = None
    try:
        if not wait_for(lambda: log.has("[session] script done") and (data / "api.json").exists(), 120):
            report["outcome"] = "the dashboard was not built: " + " | ".join(log.snapshot()[-5:])
            return finish(report, app, raw, out, args.size)
        report["build_script"] = next(l for l in log.snapshot() if "[session] script done" in l)

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
        argv = [sys.executable, str(HERE.parent / "outside_agent.py"), "--mcp", str(find_mcp(build)),
                "--api-file", str(data / "api.json")]
        if args.model:
            argv += ["--model", args.model, "--ollama", args.ollama]
        agent = spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env_for_children())
        for raw_line in agent.stdout:
            line = raw_line.decode("utf-8", "replace").rstrip("\n")
            agent_lines.append(line)
            print(f"  agent: {line}", flush=True)
        agent_code = agent.wait()
        client_done.touch()
        stop.set()
        app.wait(timeout=180)
    except subprocess.TimeoutExpired:
        app.terminate()
    report["agent_exit"] = agent_code
    report["agent_lines"] = len(agent_lines)
    report["approvals"] = [l for l in log.snapshot() if l.startswith("[approval]")]
    report["outcome"] = ("the edit was approved and landed" if agent_code == 0 and
                         any(l.endswith("(mcp) approved") for l in report["approvals"]) else "the edit did not land")
    return finish(report, app, raw, out, args.size)


def finish(report: dict, app: subprocess.Popen, raw: Path, out: Path, size: str) -> int:
    if app.poll() is None:
        app.terminate()
        try:
            app.wait(timeout=30)
        except subprocess.TimeoutExpired:
            app.kill()
    report["app_exit"] = app.returncode
    w, h = size.split("x")
    captures = []
    for name in ("dashboard", "card", "landed"):
        src = raw / f"{name}.png"
        if not src.exists():
            continue
        dst = out / f"step5-{name}-{size}.png"
        if shutil.which("sips"):
            subprocess.run(["sips", "-z", h, w, str(src), "--out", str(dst)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        else:
            shutil.copyfile(src, dst)
        captures.append(str(dst))
    report["captures"] = captures
    print(json.dumps(report), flush=True)
    return 0 if report.get("outcome") == "the edit was approved and landed" else 1


if __name__ == "__main__":
    sys.exit(main())
