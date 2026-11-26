"""Step 3 of the build-a-dashboard walk-through, run headless: ⌘K on a panel with the widget agent on a local model.

    python examples/build_a_dashboard/dry_run/step3_cmdk.py --agent-python <python> [--caliper <app>]
        [--out <folder>] [--work <folder>] [--size 1440x900] [--workspace <folder> --app-data <folder>]
        [--runs N] [--request series|heatmap|both|selection|property|all]

It needs Ollama running on 127.0.0.1:11434 (or --ollama) with the model example-model (or the one given with --model), the
caliper executable built with the builder, a Python interpreter that has caliper-agent's dependencies
(--agent-python, or CALIPER_AGENT_PYTHON), and the two data files of make_data.py in data/ beside this folder
(or in --data). It stops with exit code 2 and says what is missing when any of them is not there.

What it does, all in the hidden app (`caliper --hidden`), which never shows a window:

1. The dashboard: without --workspace, a new workspace in the step's own folder, where a command script imports
   the two data files and makes the dashboard of expected.spec.toml beside this script (the one step 2 builds
   by hand: the series `series` of loss and accuracy, the heatmap `heatmap` of one attention head and the
   table `table`) and saves it as the spec applet training-run. With --workspace and --app-data, as in
   run_all.py --story, the dashboard is the one step 2 built and saved in that workspace.
2. For each request, opens the dashboard in a fresh hidden app, opens the widget agent's settings, chooses the
   Ollama preset and the model, lists the server's models and probes the model, as a person does the first
   time, and closes the settings. Then it focuses the panel, presses ⌘K, types the request and presses Return,
   waits for the preview, accepts it, and undoes it with ⌘Z in build mode (⌘B), with a capture at each step.
   The requests are "Use a log scale for the y axis" on the series and "Use the diverging colormap" on the
   heatmap; with --request selection or property, also "Make a table of these points" on a selection of the
   series, and "Use the diverging colormap" from the heatmap's colormap in Properties.
3. Checks the workspace's journal: the agent's edit committed by the person as one step, then undone.

The agent is caliper-agent, the real sidecar, on the real model: no test script. The app gets the environment of
hidden_app.py (CALIPER_AGENT_OFFLINE=1, so the sidecar reaches nothing but a model server on loopback, a PATH of
the system's folders, no API key or endpoint variable, CALIPER_EXIT_AFTER), and keeps keys in memory only.

The captures go to --out as step3-<request>-<what>-<W>x<H>.png, W x H the window's size in points; with --runs
above 1, only the first run's. An --out that already holds captures of step 3 is refused. Each run's log (the
app's standard error with the time of each line) and a summary of all runs (step3-summary.json) go into a new
subfolder of --work, or into a temporary folder that is removed at the end. Every process it starts is stopped
by its pid when the script ends or is interrupted. The exit code is 0 when every run ended with the change
accepted and undone, 1 when one did not, 2 when something it needs is missing, and 130 when interrupted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from hidden_app import (APPLET, EXPECTED_SPEC, FAILED, INTERRUPTED, MISSING, OK, Children, Stop, add_options, app_env,
                        caliper_problem, fixture_commands, fresh_folder, saved_spec, work_folder)

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
OLLAMA = "http://127.0.0.1:11434"


class Missing(Exception):
    """Something the step needs is not there; the message says what to do."""


# ---- what it needs ------------------------------------------------------------------------------


def _get(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:  # loopback only
        return json.loads(response.read())


def check_ollama(base: str, model: str) -> str:
    """Lists the server's models and reads the model's metadata; it never calls the model."""
    if (urllib.parse.urlsplit(base).hostname or "") not in ("127.0.0.1", "localhost", "::1"):
        raise Missing(f"--ollama {base} is not a server on this machine.")
    try:
        version = _get(base + "/api/version").get("version", "?")
        tags = _get(base + "/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Missing(f"Ollama does not answer on {base} ({exc}). Start it with `ollama serve`.") from None
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if model not in names:
        raise Missing(f"Ollama has no model {model}. Download it with `ollama pull {model}`.")
    try:
        caps = _get(base + "/api/show", {"model": model}).get("capabilities", [])
    except (urllib.error.URLError, OSError, ValueError):
        caps = []
    if caps and "tools" not in caps:
        raise Missing(f"Ollama says that {model} does not call tools ({', '.join(caps)}); choose a model that does.")
    return f"Ollama {version}, {model}"


def check_files(args) -> tuple[Path, Path, Path]:
    why = caliper_problem(args.caliper)
    if why:
        raise Missing(why)
    python = args.agent_python or ""
    if not python:
        raise Missing("No interpreter for caliper-agent: give --agent-python or set CALIPER_AGENT_PYTHON to a Python "
                      "that has the dependencies of src/builder/agent/pyproject.toml.")
    probe = subprocess.run([python, "-c", "import pydantic_ai, anthropic, openai"], capture_output=True, text=True,
                           timeout=60, stdin=subprocess.DEVNULL)
    if probe.returncode != 0:
        raise Missing(f"{python} lacks caliper-agent's dependencies (pydantic-ai-slim, anthropic, openai).")
    data = Path(args.data or EXAMPLE / "data")
    parquet, tensors = data / "run_metrics.parquet", data / "attention.safetensors"
    for f in (parquet, tensors):
        if not f.is_file():
            raise Missing(f"No {f}. Run examples/build_a_dashboard/make_data.py first, or give --data.")
    if bool(args.workspace) != bool(args.app_data):
        raise Missing("--workspace and --app-data go together.")
    if args.workspace and not saved_spec(args.workspace).is_file():
        raise Missing(f"The workspace holds no saved {APPLET}; step 2 saves it there.")
    out = Path(args.out)
    if out.is_dir() and any(out.glob("step3-*.png")):
        raise Missing(f"{out} already holds captures of step 3, which this script does not replace; give "
                      "another --out.")
    return Path(python), parquet, tensors


# ---- the hidden app -----------------------------------------------------------------------------


class AppRun:
    """One hidden app; its standard error is read line by line, each line with its time."""

    def __init__(self, children: Children, argv: list[str], env: dict, log: Path) -> None:
        self.children = children
        self.t0 = time.monotonic()
        self.lines: list[tuple[float, str]] = []
        self._log = log.open("w", encoding="utf-8")
        self.proc = children.start("the app", argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, errors="replace")
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
            code = self.children.stop(self.proc)
        self._reader.join(timeout=5)
        self._log.close()
        return code

    def stop(self) -> int:
        self.children.stop(self.proc)
        return self.wait(30)

    def first(self, text: str, after: float = 0.0) -> float | None:
        return next((t for t, line in list(self.lines) if t >= after and text in line), None)

    def has(self, text: str) -> bool:
        return any(text in line for _, line in list(self.lines))


def environment(app_data: Path, user_applets: Path, python: Path, size: str, extra: dict, exit_after: int) -> dict:
    app_data.mkdir(parents=True, exist_ok=True)
    user_applets.mkdir(parents=True, exist_ok=True)
    return app_env(app_data, user_applets, size, exit_after=exit_after,
                   extra=dict(extra, CALIPER_AGENT_PYTHON=str(python)))


def build_workspace(children: Children, args, python: Path, parquet: Path, tensors: Path, work: Path) -> Path:
    """The dashboard of expected.spec.toml, built by a command script and saved as a spec applet."""
    base = fresh_folder(work, "base")
    script, results = base / "commands.jsonl", base / "results.jsonl"
    script.write_text("".join(json.dumps(c) + "\n" for c in fixture_commands(parquet, tensors)), encoding="utf-8")
    env = environment(base / "app-data", base / "applets", python, args.size,
                      {"CALIPER_COMMAND_SCRIPT": str(script), "CALIPER_COMMAND_RESULTS": str(results)}, 120)
    run = AppRun(children, [str(args.caliper), "--hidden", "--workspace", str(base / "ws")], env, base / "build.log")
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
    saved = saved_spec(base / "ws")
    same = saved.is_file() and saved.read_text() == EXPECTED_SPEC.read_text()
    print(f"step3_cmdk: the dashboard {'equals' if same else 'differs from'} expected.spec.toml", flush=True)
    return base / "ws"


# ---- one request --------------------------------------------------------------------------------

REQUESTS = {
    "series": {"panel": "series", "text": "Use a log scale for the y axis", "change": ("panel.patch",)},
    "heatmap": {"panel": "heatmap", "text": "Use the diverging colormap", "change": ("panel.patch",)},
    "selection": {"panel": "series", "text": "Make a table of these points",
                  "change": ("panel.create", "source.add", "layout.place")},
    "property": {"panel": "heatmap", "text": "Use the diverging colormap", "change": ("panel.patch",)},
}


def ui_script(name: str, model: str, turn_seconds: int) -> str:
    r = REQUESTS[name]
    settings = (f"wait 60; settings open; settings until open; settings preset ollama; settings model {model}; "
                f"settings list; settings until done 60; settings probe; settings until done 300; wait 10; "
                f"shot settings; settings close; wait 5")
    if name == "selection":
        aim = "select series 0.3 0.45; wait 10; leave; wait 5; shot target"
    elif name == "property":
        aim = ("key cmd+b; wait 20; click panel:heatmap 0.5 0.7; pane focus heatmap; wait 10; shot target; "
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


def run_request(children: Children, name: str, n: int, args, python: Path, base_ws: Path, work: Path) -> dict:
    root = fresh_folder(work, f"{name}-{n}")
    story = bool(args.workspace)
    if story:
        ws, app_data = Path(args.workspace).resolve(), Path(args.app_data).resolve()
    else:
        ws, app_data = root / "ws", root / "app-data"
        shutil.copytree(base_ws, ws)
    before = len(journal(ws))                     # in the story, the workspace's journal already holds step 2
    shots = root / "shots"
    shots.mkdir(parents=True)
    ui = ui_script(name, args.model, args.turn_seconds)
    env = environment(app_data, root / "applets", python, args.size,
                      {"CALIPER_AUTOLAUNCH": f"local.{APPLET}", "CALIPER_TEST_SPEC_UI": ui,
                       "CALIPER_TEST_SPEC_UI_SHOTS": str(shots)}, args.turn_seconds + 600)
    run = AppRun(children, [str(args.caliper), "--hidden", "--api", "--api-port", "0", "--workspace", str(ws)], env,
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
    events = journal(ws)[before:]
    outcome["agent_edits"] = [e.get("name") for e in events if e.get("origin") == "agent"]
    outcome["committed"] = any(e.get("name") == "preview.commit" and e.get("origin") == "ui" for e in events)
    outcome["undone"] = any(e.get("name") == "history.undo" for e in events)
    outcome["limit"] = any("tag LIMIT" in state for state in outcome["surface"])
    # The change asked for, accepted as one step and undone again.
    outcome["ok"] = bool(outcome["done"] and outcome["committed"] and outcome["undone"] and code == 0
                         and any(e in REQUESTS[name]["change"] for e in outcome["agent_edits"]))
    if n == 1:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for png in sorted(shots.glob("*.png")):
            shutil.copy2(png, out / f"step3-{name}-{png.stem}-{args.size}.png")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 3 of the walk-through: ⌘K with the widget agent on a local "
                                                 "model, in the hidden app.")
    add_options(parser, agent=True, model=True)
    parser.add_argument("--ollama", default=OLLAMA, help=f"Ollama on this machine (default {OLLAMA})")
    parser.add_argument("--data", help="the folder of make_data.py's two files (default: data/ of the example)")
    parser.add_argument("--request", default="both", choices=["series", "heatmap", "both", "selection",
                                                              "property", "all"])
    parser.add_argument("--runs", type=int, default=1, help="runs of each request (default: 1)")
    parser.add_argument("--turn-seconds", type=int, default=300, help="the longest a turn may take (default: 300)")
    args = parser.parse_args(argv)

    try:
        server = check_ollama(args.ollama.rstrip("/"), args.model)
        python, parquet, tensors = check_files(args)
    except Missing as exc:
        print(f"step3_cmdk: {exc}", file=sys.stderr)
        return MISSING
    names = {"both": ["series", "heatmap"], "all": ["series", "heatmap", "selection", "property"]}.get(
        args.request, [args.request])
    outcomes = []
    with Children() as children:
        try:
            work = work_folder(args, 3)
            print(f"step3_cmdk: {server}; work in {work}", flush=True)
            base_ws = None if args.workspace else build_workspace(children, args, python, parquet, tensors, work)
            for name in names:
                for n in range(1, args.runs + 1):
                    o = run_request(children, name, n, args, python, base_ws, work)
                    outcomes.append(o)
                    print(f"step3_cmdk: {name} run {n}: {'ok' if o['ok'] else 'FAILED'}"
                          f"{' at the LIMIT' if o['limit'] else ''}, turn {o['turn_seconds']} s, "
                          f"surface {' > '.join(o['surface'])}, agent {o['agent_edits']}"
                          + (f", stopped: {o['stopped']}" if o["stopped"] else ""), flush=True)
            (work / "step3-summary.json").write_text(json.dumps(outcomes, indent=1), encoding="utf-8")
        except Stop as why:
            print(f"step3_cmdk: interrupted by {why}", flush=True)
            return INTERRUPTED
        except RuntimeError as why:
            print(f"step3_cmdk: FAILED, {why}", flush=True)
            return FAILED
    print(f"step3_cmdk: {sum(o['ok'] for o in outcomes)} of {len(outcomes)} runs ok; captures in {args.out}")
    return OK if outcomes and all(o["ok"] for o in outcomes) else FAILED


if __name__ == "__main__":
    sys.exit(main())
