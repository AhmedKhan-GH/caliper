"""Step 3 of the build-a-dashboard walk-through, run headless: ⌘K on a panel with the widget agent on a local model.

    python examples/build_a_dashboard/dry_run/step3_cmdk.py --caliper <build>/caliper \
        --agent-python <python> --out <folder> [--runs N] [--request curves|attention|selection|property|all]

It needs Ollama running on 127.0.0.1:11434 with the model qwen3.8:27b (or the one given with --model), the
caliper executable built with the builder, a Python interpreter that has caliper-agent's dependencies
(--agent-python, or CALIPER_AGENT_PYTHON), and the two data files of make_data.py in data/ beside this folder
(or in --data). It stops with exit code 2 and says what is missing when any of them is not there.

What it does, all in the hidden app (`caliper --hidden`), which never shows a window:

1. Builds a workspace in --work: imports the two data files and makes a dashboard of two panels, the loss and
   accuracy curves on a log y axis (`curves`) and the attention of one head as a heatmap (`attention`), through
   a command script, and saves it as a spec applet.
2. For each request, opens the dashboard in a fresh hidden app, opens the widget agent's settings, chooses the
   Ollama preset and the model, lists the server's models and probes the model, as a person does the first
   time, and closes the settings. Then it focuses the panel, presses ⌘K, types the request and presses Return,
   waits for the preview, accepts it, and undoes it with ⌘Z in build mode (⌘B), with a capture at each step.
   The requests are "Use a linear y axis" on the curves and "Use the diverging colormap" on the heatmap; with
   --request selection or property, also "Make a table of these points" on a selection of the curves, and
   "Use the diverging colormap" from the heatmap's colormap in Properties.
3. Checks the workspace's journal: the agent's edit committed by the person as one step, then undone.

The agent is caliper-agent, the real sidecar, on the real model: no test script. CALIPER_AGENT_OFFLINE=1 is set
for everything it starts, so the sidecar reaches nothing but a model server on loopback, and every API key and
endpoint variable of the environment is removed. The hidden app keeps keys in memory only.

The captures go to --out as step3-<request>-<what>-<W>x<H>.png; with --runs above 1, only the first run's.
Each run's log (the app's standard error with the time of each line) and a summary of all runs
(summary/step3-summary.json) go to --work, each run in a new folder of its own: the script never deletes anything, and a
name that exists gets a suffix. Every process it starts is stopped by its pid when the script ends or is
interrupted, and each hidden app also ends by itself after a while (CALIPER_EXIT_AFTER). The exit code is 0 when every run ended with the change accepted and
undone, 1 when one did not, and 2 when something it needs is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
REPO = EXAMPLE.parent.parent

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3.8:27b"
APPLET = "training-run"
WINDOW = "1440x900"

# The variables that could carry a key or send the sidecar elsewhere; none reaches the app.
REMOVED = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CALIPER_MODEL_API_KEY",
           "OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CALIPER_AGENT_TEST_SCRIPT",
           "CALIPER_TEST_UI", "CALIPER_COMMAND_SCRIPT", "CALIPER_AUTOLAUNCH")


class Missing(Exception):
    """Something the step needs is not there; the message says what to do."""


# ---- what it needs ------------------------------------------------------------------------------


def _get(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=5) as response:  # loopback only
        return json.loads(response.read())


def check_ollama(model: str) -> str:
    try:
        version = _get(OLLAMA + "/api/version").get("version", "?")
        tags = _get(OLLAMA + "/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Missing(f"Ollama does not answer on {OLLAMA} ({exc}). Start it with `ollama serve`.") from None
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if model not in names:
        raise Missing(f"Ollama has no model {model}. Download it with `ollama pull {model}`.")
    try:
        caps = _get(OLLAMA + "/api/show", {"model": model}).get("capabilities", [])
    except (urllib.error.URLError, OSError, ValueError):
        caps = []
    if caps and "tools" not in caps:
        raise Missing(f"Ollama says that {model} does not call tools ({', '.join(caps)}); choose a model that does.")
    return f"Ollama {version}, {model}"


def check_files(args) -> tuple[Path, Path, Path, Path]:
    caliper = Path(args.caliper or REPO / "cmake-build-release" / "caliper")
    if not caliper.is_file() or not os.access(caliper, os.X_OK):
        raise Missing(f"No caliper executable at {caliper}. Build the target caliper, or give --caliper.")
    python = args.agent_python or os.environ.get("CALIPER_AGENT_PYTHON", "")
    if not python:
        raise Missing("No interpreter for caliper-agent: give --agent-python or set CALIPER_AGENT_PYTHON to a Python "
                      "that has the dependencies of src/builder/agent/pyproject.toml.")
    probe = subprocess.run([python, "-c", "import pydantic_ai, anthropic, openai"], capture_output=True, text=True,
                           timeout=60)
    if probe.returncode != 0:
        raise Missing(f"{python} lacks caliper-agent's dependencies (pydantic-ai-slim, anthropic, openai).")
    data = Path(args.data or EXAMPLE / "data")
    parquet, tensors = data / "run_metrics.parquet", data / "attention.safetensors"
    for f in (parquet, tensors):
        if not f.is_file():
            raise Missing(f"No {f}. Run examples/build_a_dashboard/make_data.py first, or give --data.")
    if args.out.is_dir() and any(args.out.glob("step3-*.png")):
        raise Missing(f"{args.out} already holds captures of step 3, which this script does not replace; give "
                      "another --out.")
    return caliper, Path(python), parquet, tensors


# ---- the hidden app -----------------------------------------------------------------------------


def app_env(root: Path, python: Path, extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in REMOVED}
    env.update({
        "CALIPER_AGENT_OFFLINE": "1",
        "CALIPER_AGENT_PYTHON": str(python),
        "CALIPER_DATA_DIR": str(root / "app-data"),
        "CALIPER_USER_APPLETS_DIR": str(root / "applets"),
        "CALIPER_WINDOW_SIZE": WINDOW,
    })
    env.update(extra)
    (root / "app-data").mkdir(parents=True, exist_ok=True)
    (root / "applets").mkdir(parents=True, exist_ok=True)
    return env


# Every process the script started and has not seen end, stopped by its pid on SIGINT and SIGTERM.
STARTED: list[subprocess.Popen] = []


def stop_started(*_args) -> None:
    for proc in list(STARTED):
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 30
    for proc in list(STARTED):
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def on_signal(signum, _frame) -> None:
    stop_started()
    sys.exit(128 + signum)


def fresh_folder(parent: Path, name: str) -> Path:
    """A new folder `name` in `parent`, or `name-2`, `name-3`, ... when it exists; nothing is ever cleared."""
    for n in range(1, 1000):
        folder = parent / (name if n == 1 else f"{name}-{n}")
        try:
            folder.mkdir(parents=True)
            return folder
        except FileExistsError:
            continue
    raise RuntimeError(f"no free folder name for {name} in {parent}")


class AppRun:
    """One hidden app; its standard error is read line by line, each line with its time."""

    def __init__(self, argv: list[str], env: dict, log: Path) -> None:
        self.t0 = time.monotonic()
        self.lines: list[tuple[float, str]] = []
        self._log = log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True, errors="replace")
        STARTED.append(self.proc)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for line in self.proc.stderr:
            t = time.monotonic() - self.t0
            self.lines.append((t, line.rstrip("\n")))
            self._log.write(f"{t:8.2f} {line}")
            self._log.flush()

    def wait(self, timeout: float) -> int:
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.send_signal(signal.SIGTERM)
            try:
                code = self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                code = self.proc.wait()
        self._reader.join(timeout=5)
        self._log.close()
        if self.proc in STARTED:
            STARTED.remove(self.proc)
        return code

    def stop(self) -> int:
        self.proc.send_signal(signal.SIGTERM)
        return self.wait(30)

    def first(self, text: str, after: float = 0.0) -> float | None:
        return next((t for t, line in self.lines if t >= after and text in line), None)

    def has(self, text: str) -> bool:
        return any(text in line for _, line in self.lines)


def build_workspace(caliper: Path, python: Path, parquet: Path, tensors: Path, work: Path) -> Path:
    """The dashboard of two panels, built by a command script and saved as a spec applet."""
    base = fresh_folder(work, "base")
    commands = [
        {"name": "data.import", "args": {"paths": [str(parquet), str(tensors)], "run": True}},
        {"name": "spec.new", "args": {"id": APPLET, "title": "Training run"}},
        {"name": "source.add", "args": {"id": "metrics", "catalog": "table:run_metrics"}},
        # attn is layers x heads x query x key; layer 1, head 2 is an induction head.
        {"name": "source.add", "args": {"id": "attn", "catalog": "tensor:attention_attn",
                                        "transform": [{"op": "slice", "axis": 0, "index": 1},
                                                      {"op": "slice", "axis": 0, "index": 2}]}},
        {"name": "panel.create", "args": {"id": "curves", "primitive": "series", "title": "Loss and accuracy",
                                          "bind": {"x": "metrics.step", "y": ["metrics.loss", "metrics.accuracy"]},
                                          "options": {"x_label": "Step", "y_log": True}}},
        {"name": "panel.create", "args": {"id": "attention", "primitive": "heatmap",
                                          "title": "Attention of layer 1, head 2", "bind": {"values": "attn"},
                                          "options": {"colormap": "magma", "x_label": "Key", "y_label": "Query"}}},
        {"name": "layout.set", "args": {"layout": {"root": "main", "nodes": [
            {"id": "main", "split": "horizontal", "children": ["curves", "attention"], "ratio": 0.6}]}}},
        {"name": "applet.save", "args": {"overwrite": True}},
    ]
    script, results = base / "commands.jsonl", base / "results.jsonl"
    script.write_text("".join(json.dumps(c) + "\n" for c in commands), encoding="utf-8")
    env = app_env(base, python, {"CALIPER_COMMAND_SCRIPT": str(script), "CALIPER_COMMAND_RESULTS": str(results),
                                 "CALIPER_EXIT_AFTER": "120"})
    run = AppRun([str(caliper), "--hidden", "--workspace", str(base / "ws")], env, base / "build.log")
    deadline = time.monotonic() + 110
    done = None
    while time.monotonic() < deadline and run.proc.poll() is None:
        if results.is_file():
            for line in results.read_text(encoding="utf-8").splitlines():
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                if doc.get("done"):
                    done = doc
        if done:
            break
        time.sleep(0.2)
    code = run.stop()
    if not done or done.get("failed"):
        failed = [line for line in results.read_text(encoding="utf-8").splitlines()
                  if '"ok": false' in line] if results.is_file() else []
        raise RuntimeError(f"the dashboard was not built (exit {code}): {failed or 'no results'}; see {base}")
    return base


# ---- one request --------------------------------------------------------------------------------

REQUESTS = {
    "curves": {"panel": "curves", "text": "Use a linear y axis", "change": ("panel.patch",)},
    "attention": {"panel": "attention", "text": "Use the diverging colormap", "change": ("panel.patch",)},
    "selection": {"panel": "curves", "text": "Make a table of these points",
                  "change": ("panel.create", "source.add", "layout.place")},
    "property": {"panel": "attention", "text": "Use the diverging colormap", "change": ("panel.patch",)},
}


def ui_script(name: str, model: str, turn_seconds: int) -> str:
    r = REQUESTS[name]
    settings = (f"wait 60; settings open; settings until open; settings preset ollama; settings model {model}; "
                f"settings list; settings until done 60; settings probe; settings until done 300; wait 10; "
                f"shot settings; settings close; wait 5")
    if name == "selection":
        aim = "select curves 0.3 0.45; wait 10; leave; wait 5; shot target"
    elif name == "property":
        aim = ("key cmd+b; wait 20; click panel:attention 0.5 0.7; pane focus attention; wait 10; shot target; "
               "click cmdk:prop:options.colormap")
    else:
        aim = f"focus {r['panel']}; wait 3; leave; wait 10; shot target"
    key = "" if name == "property" else "key cmd+k; "
    turn = (f"{key}until prompt; wait 3; type {r['text']}; wait 10; shot prompt; key enter; wait 600; shot running; "
            f"until preview {turn_seconds}; wait 10; shot preview; key enter; until applied; wait 10; "
            f"shot accepted")
    # ⌘Z undoes in build mode, where the page bar shows ⌘Z Undo.
    undo = ("" if name == "property" else "key cmd+b; wait 20; ") + "leave; key cmd+z; wait 20; shot undone"
    return f"{settings}; {aim}; {turn}; {undo}; quit"


def journal(ws: Path) -> list[dict]:
    out = []
    f = ws / "journal" / "commands.jsonl"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def run_request(name: str, n: int, args, caliper: Path, python: Path, base: Path, work: Path) -> dict:
    root = fresh_folder(work, f"{name}-{n}")
    shutil.copytree(base / "ws", root / "ws")
    shots = root / "shots"
    shots.mkdir(parents=True)
    ui = ui_script(name, args.model, args.turn_seconds)
    env = app_env(root, python, {"CALIPER_AUTOLAUNCH": f"local.{APPLET}", "CALIPER_TEST_SPEC_UI": ui,
                                 "CALIPER_TEST_SPEC_UI_SHOTS": str(shots),
                                 "CALIPER_EXIT_AFTER": str(args.turn_seconds + 600)})
    run = AppRun([str(caliper), "--hidden", "--api", "--api-port", "0", "--workspace", str(root / "ws")], env,
                 root / "app.log")
    code = run.wait(args.turn_seconds + 660)
    sent = run.first("[ui] pressed enter")
    preview = run.first("[ui] the surface is preview")
    outcome = {"request": name, "run": n, "exit": code, "done": run.has("[ui] done"),
               "stopped": next((line.split("stopped: ", 1)[1] for _, line in run.lines if "[ui] stopped: " in line),
                               None),
               "probe": next((line for _, line in run.lines if "[ui] the settings" in line and "probe" in line), None),
               "turn_seconds": round(preview - sent, 1) if sent is not None and preview is not None else None,
               "surface": [line.split("[ui] the surface is ", 1)[1] for _, line in run.lines
                           if "[ui] the surface is " in line]}
    events = journal(root / "ws")
    outcome["agent_edits"] = [e.get("name") for e in events if e.get("origin") == "agent"]
    outcome["committed"] = any(e.get("name") == "preview.commit" and e.get("origin") == "ui" for e in events)
    outcome["undone"] = any(e.get("name") == "history.undo" for e in events)
    outcome["limit"] = any("tag LIMIT" in state for state in outcome["surface"])
    # The change asked for, accepted as one step and undone again.
    outcome["ok"] = bool(outcome["done"] and outcome["committed"] and outcome["undone"] and code == 0
                         and any(e in REQUESTS[name]["change"] for e in outcome["agent_edits"]))
    if n == 1:
        args.out.mkdir(parents=True, exist_ok=True)
        for png in sorted(shots.glob("*.png")):
            w, h = png_size(png)
            shutil.copy2(png, args.out / f"step3-{name}-{png.stem}-{w}x{h}.png")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 3 of the walk-through: ⌘K with the widget agent on a local "
                                                 "model, in the hidden app.")
    parser.add_argument("--caliper", help="the caliper executable (default: cmake-build-release/caliper)")
    parser.add_argument("--agent-python", help="a Python with caliper-agent's dependencies "
                                               "(default: CALIPER_AGENT_PYTHON)")
    parser.add_argument("--data", help="the folder of make_data.py's two files (default: data/ of the example)")
    parser.add_argument("--model", default=MODEL, help=f"the Ollama model (default: {MODEL})")
    parser.add_argument("--out", type=Path, required=True, help="the folder for the captures")
    parser.add_argument("--work", type=Path, help="the folder for the workspaces and logs (default: a new "
                                                  "temporary folder, kept)")
    parser.add_argument("--request", default="both", choices=["curves", "attention", "both", "selection",
                                                              "property", "all"])
    parser.add_argument("--runs", type=int, default=1, help="runs of each request (default: 1)")
    parser.add_argument("--turn-seconds", type=int, default=300, help="the longest a turn may take (default: 300)")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        server = check_ollama(args.model)
        caliper, python, parquet, tensors = check_files(args)
    except Missing as exc:
        print(f"step3_cmdk: {exc}", file=sys.stderr)
        return 2
    work = args.work or Path(tempfile.mkdtemp(prefix="caliper-step3-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"step3_cmdk: {server}; work in {work}")
    base = build_workspace(caliper, python, parquet, tensors, work)
    names = {"both": ["curves", "attention"], "all": ["curves", "attention", "selection", "property"]}.get(
        args.request, [args.request])
    outcomes = []
    for name in names:
        for n in range(1, args.runs + 1):
            o = run_request(name, n, args, caliper, python, base, work)
            outcomes.append(o)
            print(f"step3_cmdk: {name} run {n}: {'ok' if o['ok'] else 'FAILED'}{' at the LIMIT' if o['limit'] else ''}, "
                  f"turn {o['turn_seconds']} s, "
                  f"surface {' > '.join(o['surface'])}, agent {o['agent_edits']}"
                  + (f", stopped: {o['stopped']}" if o["stopped"] else ""))
    (fresh_folder(work, "summary") / "step3-summary.json").write_text(json.dumps(outcomes, indent=1),
                                                                     encoding="utf-8")
    print(f"step3_cmdk: {sum(o['ok'] for o in outcomes)} of {len(outcomes)} runs ok; captures in {args.out}")
    return 0 if all(o["ok"] for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
