"""Runs the six dry runs of the build-a-dashboard walk-through in order and writes one summary.

    python examples/build_a_dashboard/dry_run/run_all.py --out <new folder> [--size 1024x640]
        [--steps 1-6] [--story] [--caliper <app>] [--train-python <python>]
        [--agent-python <python>] [--model qwen3.8:27b] [--script <N>=<file>]

Each step is a script of its own in this folder, runnable alone, which plays
the person's part in the hidden app (`caliper --hidden`), with the browser of
step 6 headless, and writes its captures:

    1  step1_train_and_attach     train.py streams and the app attaches to it
    2  step2_drop_and_build       the data files dropped and a dashboard built by hand
    3  step3_cmdk                 ⌘K on a panel: preview, Accept and undo, on the local model
    4  step4_side_chat            the side chat plots the training run, on the local model
    5  step5_outside_agent        an outside agent's edit through caliper-mcp, held and approved
    6  step6_save_reopen_browser  save, reopen from the landing screen, and open in a browser

A step whose script is not in this folder yet is reported as not there yet.
--script N=<file> runs another file as step N, such as a draft or a stand-in.

By default each step starts from its own fixture, in its own workspace. With
--story the steps make one continuous run: one workspace and one app-data
folder for all of them, so that the dashboard built in step 2 is the one that
⌘K changes in step 3, that the outside agent adds to in step 5 and that step 6
saves and reopens; and one training run, which the runner starts before step 1
and stops after step 4, so that step 1 attaches to the stream that the side
chat plots in step 4. A step that does not take the story's options yet runs
on its own fixture, and the summary says which options it lacks.

What every step is given:

    --caliper <app> --out <captures folder> --work <its own folder> --size <W>x<H>
    --train-python <python>      steps 1 and 4, which run train.py
    --agent-python <python>      steps 3 and 4, which start caliper-agent
    --model <name>               steps 3 and 4, which call the local model
    --workspace <folder> --app-data <folder>    with --story
    --url <stream link>          with --story, steps 1 and 4

A step whose --help does not list these options is run with the options it
had before, where the runner knows them, and the summary says so.

The output folder must be new or empty. It receives:

    summary.txt, summary.json   which step passed, which failed and where, and how long each took
    captures/                   the captures of every step, step<N>-<what>-<W>x<H>.png
    logs/                       each step's output, and train.log with --story
    work/                       each step's own folder, and the story's workspace and app data

Heavy jobs and the local model. When CALIPER_DEMO_HEAVY is set, each step
runs through it, as `$CALIPER_DEMO_HEAVY "<label>" <command...>`. When
CALIPER_DEMO_MODEL is set, steps 3 and 4 run through it the same way,
together as one job, so that the model stays loaded from one to the other;
with --story, steps 1 to 4 are that one job, since the training run lives
from step 1 to step 4. Unset, the steps run plainly. Before it starts
anything, the runner checks what the selected steps need: the app, the data
files, the Pythons, Node with Playwright, and for steps 3 and 4 that Ollama
answers on this machine and lists the model. It never calls the model itself.

Every process a step starts (the training script, the app and its sidecar,
the producer, caliper-mcp, the browser) is watched by pid while the step
runs. Whatever a step leaves running when it ends, fails, runs out of time or
is interrupted is stopped by its pid, and the summary lists each one. The
runner sets CALIPER_AGENT_OFFLINE=1 for everything it starts and leaves API
keys and test hooks out of their environment.

Exit status: 0 when every selected step passed, 1 when one failed, 2 when
something a step needs is missing (nothing was started), 3 when none failed
but some are not there yet, and 130 when interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
REPO = EXAMPLE.parent.parent
TRAIN = EXAMPLE / "train.py"
DATA = [EXAMPLE / "data" / "run_metrics.parquet", EXAMPLE / "data" / "attention.safetensors"]
MODEL = "qwen3.8:27b"
OLLAMA = "http://127.0.0.1:11434"
STORY_TRAIN_SECONDS = 7200          # the story's training run; the runner stops it after step 4
SHELLS = {"sh", "zsh", "bash", "dash"}
# Left out of every environment the runner gives: keys, and hooks that would
# change what the hidden app does.
DROPPED = ("ANTHROPIC_", "OPENAI_", "GEMINI_", "OPENROUTER_", "CALIPER_MODEL_API_KEY", "CALIPER_TEST_",
           "CALIPER_COMMAND_", "CALIPER_AUTOLAUNCH", "CALIPER_DEMO_")


class Stop(Exception):
    """A signal asked the run to stop."""


def on_signal(signum, frame):
    raise Stop(signal.Signals(signum).name)


# ---- the steps ----------------------------------------------------------------------------------


class Step:
    def __init__(self, n, name, title, needs=(), model=False, url=False, after=None, old=None):
        self.n, self.name, self.title = n, name, title
        self.needs = set(needs)     # train, agent, data, mcp, node
        self.model = model          # calls the local model
        self.url = url              # takes the story's stream link
        self.after = after          # in the story, the step whose dashboard it continues
        self.old = old              # the options the script took before the convention
        self.script: Path | None = None
        self.stand_in = False


def old_step1(ctx, work):
    # Until step 1 takes --work its temporary folder goes under the step's folder.
    return (["--python", ctx.train_python, "--caliper", ctx.caliper, "--out", ctx.captures, "--size", ctx.size],
            {"TMPDIR": str(work / "tmp")})


def old_hidden_app(ctx, work):
    # Steps 2 and 6 as first committed: CALIPER_BIN, --sizes, and --out for both their files and captures/shots.
    return ["--out", work, "--sizes", ctx.size], {"CALIPER_BIN": str(ctx.caliper)}


def old_step3(ctx, work):
    # Step 3 as first committed: no --size; the window is 1440x900.
    return (["--caliper", ctx.caliper, "--agent-python", ctx.agent_python, "--out", ctx.captures, "--work", work,
             "--model", ctx.model], {})


def old_step4(ctx, work):
    # Step 4's draft: a build folder, and its own training run.
    return (["--build", Path(ctx.caliper).parent, "--agent-python", ctx.agent_python, "--out", ctx.captures,
             "--work", work, "--size", ctx.size, "--model", ctx.model, "--stream", "train",
             "--train-python", ctx.train_python], {})


def old_step5(ctx, work):
    # Step 5 as first committed: a build folder; the outside agent scripted.
    return ["--build", Path(ctx.caliper).parent, "--out", ctx.captures, "--work", work, "--size", ctx.size], {}


STEPS = [
    Step(1, "step1_train_and_attach", "Train a model and attach the app", needs=("train",), url=True,
         old=old_step1),
    Step(2, "step2_drop_and_build", "Drop the data files and build a dashboard by hand", needs=("data",),
         old=old_hidden_app),
    Step(3, "step3_cmdk", "Change a panel with ⌘K, accept it and undo it", needs=("data", "agent"), model=True,
         after=2, old=old_step3),
    Step(4, "step4_side_chat", "Ask the side chat for the attention entropy per layer", needs=("agent", "train"),
         model=True, url=True, old=old_step4),
    Step(5, "step5_outside_agent", "Let an outside agent add a panel through caliper-mcp", needs=("data", "mcp"),
         after=2, old=old_step5),
    Step(6, "step6_save_reopen_browser", "Save, open again and open in a browser", needs=("data", "node"), after=2,
         old=old_hidden_app),
]


def find_script(step: Step, overrides: dict[int, Path]) -> Path | None:
    if step.n in overrides:
        step.stand_in = True
        return overrides[step.n]
    for ext in (".py", ".sh"):
        if (HERE / f"{step.name}{ext}").is_file():
            return HERE / f"{step.name}{ext}"
    return None


def interpreter(script: Path) -> list[str]:
    if script.suffix == ".py":
        return [sys.executable, str(script)]
    return [str(script)] if os.access(script, os.X_OK) else ["/bin/sh", str(script)]


# ---- what the steps need ------------------------------------------------------------------------


def clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROPPED)}
    env.update(CALIPER_AGENT_OFFLINE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def python_lacks(python: str, code: str, env: dict) -> str:
    """'' when the Python runs code, else the last line of its error."""
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                           stdin=subprocess.DEVNULL)
    except OSError as exc:
        return str(exc)
    except subprocess.TimeoutExpired:
        return "it did not answer within 120 s"
    if r.returncode == 0:
        return ""
    return (r.stderr.strip().splitlines() or ["exit " + str(r.returncode)])[-1]


def get_json(url: str):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as r:
        return json.loads(r.read())


def ollama_lacks(base: str, model: str) -> str:
    """'' when Ollama answers on this machine and lists the model. It lists; it never calls the model."""
    if (urlsplit(base).hostname or "") not in ("127.0.0.1", "localhost", "::1"):
        return f"--ollama {base} is not a server on this machine"
    try:
        version = get_json(base.rstrip("/") + "/api/version").get("version", "?")
        tags = get_json(base.rstrip("/") + "/api/tags")
    except (OSError, ValueError) as exc:
        return f"Ollama does not answer on {base} ({exc}); start it with `ollama serve`"
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if model not in names:
        return f"Ollama {version} on {base} does not list the model {model}; download it with `ollama pull {model}`"
    return ""


def find_mcp(caliper: Path) -> Path | None:
    for candidate in (caliper.parent / "caliper-mcp", caliper.parent / "src" / "builder" / "mcp" / "caliper-mcp"):
        if candidate.is_file():
            return candidate
    return None


def has_playwright(env: dict) -> bool:
    places = [REPO / "web" / "node_modules"] + [Path(p) for p in env.get("NODE_PATH", "").split(os.pathsep) if p]
    return any((p / "@playwright" / "test").is_dir() for p in places)


def wrapper(var: str) -> list[str] | None:
    value = os.environ.get(var, "").strip()
    return shlex.split(value) if value else None


def prerequisites(ctx, steps: list[Step], env: dict) -> list[str]:
    """What the selected steps need and is missing, as sentences; empty when all is there."""
    problems = []
    needs = set().union(*(s.needs for s in steps)) if steps else set()
    if not ctx.caliper.is_file() or not os.access(ctx.caliper, os.X_OK):
        problems.append(f"No caliper executable at {ctx.caliper}: build the target caliper, or give --caliper.")
    if "data" in needs:
        for f in DATA:
            if not f.is_file():
                problems.append(f"No {f.relative_to(EXAMPLE)}: run python examples/build_a_dashboard/make_data.py.")
    if "train" in needs or (ctx.story and any(s.url for s in steps)):
        if not ctx.train_python:
            problems.append("No Python for train.py: give --train-python, a Python with torch and Caliper LIB.")
        else:
            why = python_lacks(ctx.train_python, "import torch", env)
            if why:
                problems.append(f"The Python for train.py, {ctx.train_python}, cannot import torch: {why}")
            elif python_lacks(ctx.train_python, "import caliper", env):
                # From a checkout, the package in python/ serves as well as the wheel.
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO / "python"), env.get("PYTHONPATH")]))
                why = python_lacks(ctx.train_python, "import caliper", env)
                if why:
                    problems.append(f"The Python for train.py cannot import caliper, installed or from python/: {why}")
                else:
                    ctx.notes.append("train.py imports caliper from the checkout's python/ folder.")
    if "agent" in needs:
        if not ctx.agent_python:
            problems.append("No Python for caliper-agent: give --agent-python or set CALIPER_AGENT_PYTHON.")
        else:
            why = python_lacks(ctx.agent_python, "import pydantic_ai, openai", env)
            if why:
                problems.append(f"The Python for caliper-agent, {ctx.agent_python}, lacks its dependencies "
                                f"(src/builder/agent/pyproject.toml): {why}")
    if "mcp" in needs and ctx.caliper.is_file() and find_mcp(ctx.caliper) is None:
        problems.append(f"No caliper-mcp beside {ctx.caliper} or under src/builder/mcp/ of its build folder.")
    if "node" in needs:
        if not shutil.which("node", path=env.get("PATH")):
            problems.append("No node on PATH, which step 6 needs for the browser.")
        elif not has_playwright(env):
            problems.append("No Playwright for step 6: run `npm ci` in web/, or set NODE_PATH to a node_modules "
                            "that has @playwright/test.")
    if any(s.model for s in steps):
        why = ollama_lacks(ctx.ollama, ctx.model)
        if why:
            problems.append(why + ".")
        if ctx.heavy and not ctx.model_wrapper:
            problems.append("CALIPER_DEMO_HEAVY is set and CALIPER_DEMO_MODEL is not: steps 3 and 4 call the local "
                            "model and run through CALIPER_DEMO_MODEL; set both or neither.")
    for var, words in (("CALIPER_DEMO_HEAVY", ctx.heavy), ("CALIPER_DEMO_MODEL", ctx.model_wrapper)):
        if words and not (shutil.which(words[0]) or Path(words[0]).is_file()):
            problems.append(f"{var} names {words[0]}, which is not a program.")
    return problems


# ---- the options each step takes ----------------------------------------------------------------


def listed_options(script: Path, env: dict) -> tuple[set[str] | None, str]:
    """The options a Python step's --help lists; None for a shell step, which is taken to follow the convention."""
    if script.suffix != ".py":
        return None, ""
    try:
        r = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, env=env,
                           timeout=60, stdin=subprocess.DEVNULL, cwd=script.parent)
    except subprocess.TimeoutExpired:
        return set(), "its --help did not answer within 60 s"
    if r.returncode != 0:
        return set(), "its --help failed: " + ((r.stderr.strip().splitlines() or ["?"])[-1])
    return set(re.findall(r"--[a-z][a-z0-9-]*", r.stdout)), ""


def plan(step: Step, ctx, joined: dict[int, bool], env: dict) -> None:
    """The command line of one step: the convention's options when the step lists them all, else its old
    ones; and with --story, whether it continues the story's workspace."""
    work = ctx.work / f"step{step.n}"
    conv: list = ["--caliper", ctx.caliper, "--out", ctx.captures, "--work", work, "--size", ctx.size]
    if "train" in step.needs:
        conv += ["--train-python", ctx.train_python]
    if "agent" in step.needs:
        conv += ["--agent-python", ctx.agent_python]
    if step.model:
        conv += ["--model", ctx.model]
    options, broken = listed_options(step.script, env)

    def lacking(argv):
        return [] if options is None else [a for a in argv if isinstance(a, str) and a.startswith("--")
                                           and a not in options]

    step.problem = broken
    step.interface = "the runner's options"
    step.env = {}
    missing = lacking(conv)
    if broken:
        step.argv = []
    elif not missing:
        step.argv = conv
    elif step.old and not lacking(step.old(ctx, work)[0]):
        step.argv, step.env = step.old(ctx, work)
        step.interface = "its old options; it lacks " + ", ".join(missing)
    else:
        old_missing = lacking(step.old(ctx, work)[0]) if step.old else []
        step.problem = ("it takes neither the runner's options (it lacks " + ", ".join(missing) + ")" +
                        (" nor its old ones (it lacks " + ", ".join(old_missing) + ")" if step.old else ""))
        step.argv = []
    step.argv = [str(a) for a in step.argv]
    step.work = work
    step.story = ""
    if ctx.story and not step.problem:
        wanted = ["--workspace", "--app-data"] + (["--url"] if step.url else [])
        story_missing = lacking(wanted)
        if story_missing:
            step.story = "its own fixture; it lacks " + ", ".join(story_missing)
        elif step.after and not joined.get(step.after):
            step.story = f"its own fixture, since step {step.after} does not build the story's dashboard"
        else:
            step.story = "the story"
            step.argv += ["--workspace", str(ctx.story_dir / "workspace"), "--app-data", str(ctx.story_dir / "app-data")]
    joined[step.n] = step.story == "the story"


# ---- processes ----------------------------------------------------------------------------------


def ps_table() -> dict[int, tuple[int, int, str, str]]:
    """Every process that has not ended: pid -> (parent pid, group, start time, command line). A process
    that has ended and waits for its parent to collect it (a zombie) is left out."""
    r = subprocess.run(["/bin/ps", "-A", "-ww", "-o", "pid=,ppid=,pgid=,stat=,lstart=,command="],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    table = {}
    for line in r.stdout.splitlines():
        f = line.split(None, 9)
        if len(f) == 10 and f[0].isdigit() and not f[3].startswith("Z"):
            table[int(f[0])] = (int(f[1]), int(f[2]), " ".join(f[4:9]), f[9])
    return table


def program(command: str) -> str:
    """A process's name for the summary: the program, and the script an interpreter runs."""
    words = command.split()
    name = Path(words[0]).name if words else "?"
    if name.lower().startswith("python") or name in SHELLS or name == "node":
        script = next((w for w in words[1:] if not w.startswith("-")), "")
        return f"{name} {Path(script).name}".strip()
    return name


def is_shell(command: str) -> bool:
    words = command.split()
    return bool(words) and Path(words[0]).name in SHELLS


class Watched:
    """One command, started in a session of its own, and every process under it, watched by pid from start
    to end: its children by their parent, its process group, and every process once seen while it lives."""

    def __init__(self, label: str, argv: list[str], env: dict, log: Path, main: str):
        self.label, self.main = label, main
        self.seen: dict[int, tuple[str, str]] = {}
        self.lock = threading.Lock()
        self.leader_start = None
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.logf = open(log, "a", encoding="utf-8")
        self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=self.logf,
                                     stderr=subprocess.STDOUT, start_new_session=True, cwd=str(HERE))
        self.started = time.monotonic()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.done.is_set():
            self.members()
            self.done.wait(1.0)

    def members(self) -> dict[int, tuple[str, str]]:
        """The processes of this command that are alive now: pid -> (start time, command line)."""
        self.proc.poll()                            # collects the command itself once it has ended
        table = ps_table()
        leader = table.get(self.proc.pid)
        if leader and self.leader_start is None:
            self.leader_start = leader[2]
        group_ok = not leader or leader[2] == self.leader_start   # the group's number is not someone else's
        children: dict[int, list[int]] = {}
        for pid, (ppid, _, _, _) in table.items():
            children.setdefault(ppid, []).append(pid)
        with self.lock:
            known = dict(self.seen)
        roots = {self.proc.pid} if group_ok else set()
        if group_ok:
            roots |= {pid for pid, (_, pgid, _, _) in table.items() if pgid == self.proc.pid}
        roots |= {pid for pid, (start, _) in known.items() if pid in table and table[pid][2] == start}
        found, todo = set(), list(roots)
        while todo:
            pid = todo.pop()
            if pid in found or pid not in table or pid == os.getpid():
                continue
            found.add(pid)
            todo.extend(children.get(pid, ()))
        # The runner's own look at the process table, from a job's inner runner, is not the step's.
        found = {pid for pid in found if not table[pid][3].startswith("/bin/ps ")}
        live = {pid: (table[pid][2], table[pid][3]) for pid in found}
        with self.lock:
            self.seen.update(live)
        return live

    def wait(self, timeout: float | None) -> int | None:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def _signal(self, pids, sig, stopped, why):
        for pid in pids:
            start, command = self.seen.get(pid, ("", ""))
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
            entry = next((e for e in stopped if e["pid"] == pid), None)
            if entry is None:
                stopped.append({"pid": pid, "program": program(command), "signal": signal.Signals(sig).name,
                                "why": why})
            else:
                entry["signal"] = signal.Signals(sig).name

    def _wait_gone(self, pids, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            live = self.members()
            if not any(p in live for p in pids):
                return
            time.sleep(0.3)

    def stop(self, why: str, main_grace: float = 25.0) -> list[dict]:
        """Stops by pid what is still running, gently first: SIGINT to the step's own script, so that its
        clean-up runs; then SIGTERM, and SIGKILL, to every other program; the wrappers' shells end by
        themselves once their command has, which releases what they hold. Returns what it signalled."""
        stopped: list[dict] = []
        live = self.members()
        mains = [p for p, (_, cmd) in live.items() if self.main in cmd and not is_shell(cmd)]
        if mains:
            self._signal(mains, signal.SIGINT, stopped, why)
            self._wait_gone(mains, main_grace)
        for sig, seconds in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            live = self.members()
            programs = [p for p, (_, cmd) in live.items() if not is_shell(cmd)]
            if not programs:
                break
            self._signal(programs, sig, stopped, why)
            self._wait_gone(programs, seconds)
        if self.wait(90) is None:                   # a wrapper that did not end by itself
            self._signal([self.proc.pid], signal.SIGTERM, stopped, why)
            if self.wait(15) is None:
                self._signal([self.proc.pid], signal.SIGKILL, stopped, why)
                self.wait(5)
        live = self.members()
        if live:
            self._signal(list(live), signal.SIGKILL, stopped, why)
        return stopped

    def close(self):
        self.done.set()
        self.thread.join(timeout=5)
        self.logf.close()


# ---- one job: steps run one after another, plainly ----------------------------------------------


def where(log: Path) -> str:
    """The line of a step's output that says where it stopped."""
    try:
        lines = [l.rstrip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    if any(l.startswith("Traceback") for l in lines):
        return lines[-1]
    pattern = re.compile(r"^(FAIL\b|.*\bFAILED\b|.*\[ui\] stopped|.*\bmissing capture\b|usage: |.*: error: )")
    hit = next((l for l in lines if pattern.match(l)), None)
    return hit or (lines[-1] if lines else "")


def sweep(step: Step, ctx) -> list[str]:
    """Copies the step's captures that it wrote under its own folder into the captures folder, and lists
    them all."""
    for f in sorted(step.work.rglob(f"step{step.n}-*")) if step.work.is_dir() else []:
        if f.is_file() and f.suffix in (".png", ".json") and not (ctx.captures / f.name).exists():
            shutil.copyfile(f, ctx.captures / f.name)
    return sorted(p.name for p in ctx.captures.glob(f"step{step.n}-*"))


def minutes(seconds) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def say(text: str):
    print(text, flush=True)


def start_training(ctx, env) -> tuple[Watched | None, str]:
    """The story's training run, and its stream link."""
    log = ctx.logs / "train.log"
    train = Watched("train.py", [ctx.train_python, "-u", str(TRAIN), "--seconds", str(STORY_TRAIN_SECONDS)], env,
                    log, str(TRAIN))
    end = time.monotonic() + 120
    while time.monotonic() < end and train.proc.poll() is None:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("http://127.0.0.1:"):
                return train, line.strip()
        time.sleep(0.3)
    return train, ""


def run_job_plainly(steps: list[Step], ctx, env, results: dict, stopped: list) -> None:
    """Runs the steps one after another without a wrapper, each watched by pid; with the story, also its
    training run for the steps that take its link."""
    train, link = None, ""
    users = [s.n for s in steps if s.story == "the story" and s.url]
    current: Watched | None = None
    try:
        if users:
            say("story: starting train.py for steps " + ", ".join(map(str, users)))
            train, link = start_training(ctx, env)
            say(f"story: train.py (pid {train.proc.pid}) streams at {link.split('?')[0]}" if link
                else "story: train.py printed no stream link; see logs/train.log")
        for step in steps:
            r = results[step.n]
            if ctx.story and step.after and joined_failed(step, results):
                r.update(status="skipped", where=f"step {step.after}, which builds the story's dashboard, did not pass")
                say(f"step {step.n}: skipped, since step {step.after} did not pass")
                continue
            argv = interpreter(step.script) + step.argv
            if step.story == "the story" and step.url:
                if not link:
                    r.update(status="failed", where="the story's training run printed no stream link")
                    continue
                argv += ["--url", link]
            step.work.mkdir(parents=True, exist_ok=True)
            for key, value in step.env.items():
                if key == "TMPDIR":
                    Path(value).mkdir(parents=True, exist_ok=True)
            log = ctx.logs / f"step{step.n}.log"
            say(f"step {step.n}: started, {step.title.lower()} ({step.script.name}); log in logs/{log.name}")
            r["started_at"] = time.time()
            current = Watched(f"step {step.n}", argv, dict(env, **step.env), log, str(step.script))
            code = current.wait(ctx.step_timeout)
            r["seconds"] = time.monotonic() - current.started
            if code is None:
                r.update(status="failed", where=f"it ran out of its {minutes(ctx.step_timeout)} and was stopped")
                left = current.stop(f"step {step.n} ran out of time")
            else:
                r["exit"] = code
                r.update(status="passed" if code == 0 else "missing" if code == 2 else "failed")
                if code != 0:
                    r["where"] = where(log)
                left = current.stop(f"left running by step {step.n}")
            for e in left:
                e["step"] = step.n
            stopped.extend(left)
            current.close()
            current = None
            r["captures"] = sweep(step, ctx)
            say(f"step {step.n}: {r['status']} in {minutes(r.get('seconds'))}" +
                (f"; {r['where']}" if r.get("where") else "") +
                (f"; stopped {len(left)} process(es) it left running" if left else ""))
            if train is not None and users and step.n == users[-1]:
                left = train.stop("the story's training run, after its last step")
                stopped.extend(dict(e, step=step.n) for e in left)
                say(f"story: stopped train.py (pid {train.proc.pid}) after step {step.n}")
                train.close()
                train = None
    except Stop as why:
        if current is not None:
            r = results[current_step(steps, results)]
            r.update(status="interrupted", where=f"interrupted by {why}",
                     seconds=time.monotonic() - current.started)
        raise
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        if current is not None:
            stopped.extend(current.stop("the run was interrupted"))
            current.close()
        if train is not None:
            stopped.extend(train.stop("the story's training run, at the end of the run"))
            train.close()
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, on_signal)


def current_step(steps, results) -> int:
    started = [s.n for s in steps if results[s.n].get("started_at") and results[s.n]["status"] == "pending"]
    return started[-1] if started else steps[0].n


def joined_failed(step: Step, results: dict) -> bool:
    if step.story != "the story":
        return False
    return results.get(step.after, {}).get("status") != "passed"


# ---- the jobs, through the wrappers -------------------------------------------------------------


def group_jobs(steps: list[Step], story: bool) -> list[list[Step]]:
    groups = [[1, 2, 3, 4], [5], [6]] if story else [[1], [2], [3, 4], [5], [6]]
    return [chosen for g in groups if (chosen := [s for s in steps if s.n in g])]


def run_job_wrapped(k: int, job: list[Step], words: list[str], ctx, env, results, stopped) -> None:
    """Runs the job's steps through a wrapper: this script again, inside the wrapper, runs them plainly and
    writes what happened to a file, which is read back here."""
    numbers = ",".join(str(s.n) for s in job)
    result_file = ctx.work / f"job-{k}.json"
    label = f"build_a_dashboard dry run: step{'s' if len(job) > 1 else ''} {numbers}"
    inner = [sys.executable, str(Path(__file__).resolve()), "--in-job", numbers, "--job-file", str(result_file)]
    inner += ctx.forwarded
    log = ctx.logs / f"job-{k}.log"
    say(f"job {k}: steps {numbers} through {Path(words[-1]).name}; it may wait for a slot first")
    launched = time.time()
    job_proc = Watched(f"job {k}", words + [label] + inner, env, log, str(Path(__file__).resolve()))
    relay = threading.Thread(target=tail, args=(log, job_proc), daemon=True)
    relay.start()
    try:
        job_proc.wait(None)
    except Stop:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        stopped.extend(job_proc.stop("the run was interrupted", main_grace=150.0))
        job_proc.close()
        read_job(result_file, job, results, stopped, launched)
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, on_signal)
        raise
    stopped.extend(job_proc.stop(f"left running by job {k}"))
    job_proc.close()
    relay.join(timeout=3)
    read_job(result_file, job, results, stopped, launched)


def read_job(path: Path, job, results, stopped, launched):
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        for s in job:
            if results[s.n]["status"] == "pending":
                results[s.n].update(status="failed", where="the job ended without a result; see its log in logs/")
        return
    for n, r in doc.get("results", {}).items():
        results[int(n)].update(r)
    stopped.extend(doc.get("stopped", []))
    first = next((s.n for s in job if results[s.n].get("started_at")), None)
    if first is not None and doc.get("began_at"):
        results[first]["waited"] = max(0.0, doc["began_at"] - launched)


def tail(log: Path, job_proc: Watched):
    """Prints the job's own lines as they come."""
    pos = 0
    while True:
        try:
            with open(log, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            chunk = ""
        for line in chunk.splitlines():
            print("  " + line, flush=True)
        if job_proc.proc.poll() is not None and not chunk:
            return
        time.sleep(0.5)


# ---- the summary --------------------------------------------------------------------------------


def summarise(ctx, steps, results, stopped, interrupted) -> tuple[str, int]:
    counts = {}
    for s in steps:
        counts[results[s.n]["status"]] = counts.get(results[s.n]["status"], 0) + 1
    waited = any(results[s.n].get("waited") for s in steps)
    lines = [f"Dry run of the build-a-dashboard walk-through, {time.strftime('%Y-%m-%d %H:%M', time.localtime(ctx.began))}",
             ("Story: one workspace and one training run" if ctx.story else "Steps: each from its own fixture") +
             f"; window {ctx.size}; output in {ctx.out}", ""]
    header = f"{'step':<5} {'result':<14} {'time':>6} " + (f"{'waited':>7} " if waited else "") + "where"
    lines.append(header)
    for s in steps:
        r = results[s.n]
        row = f"{s.n:<5} {r['status']:<14} {minutes(r.get('seconds')):>6} "
        if waited:
            row += f"{minutes(r.get('waited')) if r.get('waited') else '':>7} "
        row += (r.get("where") or "")[:200]
        if r.get("where") and r["status"] in ("failed", "missing", "interrupted") and r.get("log"):
            row += f" ({r['log']})"
        lines.append(row.rstrip())
    lines.append("")
    for s in steps:
        r = results[s.n]
        notes = []
        if s.script is not None and s.stand_in:
            notes.append(f"ran {s.script} in its place")
        if r.get("interface") and r["interface"] != "the runner's options":
            notes.append("ran with " + r["interface"])
        if ctx.story and r.get("story") and r["story"] != "the story":
            notes.append("ran on " + r["story"])
        if notes:
            lines.append(f"Step {s.n} " + "; ".join(notes) + ".")
    for note in ctx.notes:
        lines.append(note)
    captures = sorted(p.name for p in ctx.captures.iterdir()) if ctx.captures.is_dir() else []
    lines.append(f"Captures ({len(captures)}) in captures/: " + (", ".join(captures) if captures else "none") + ".")
    if stopped:
        lines.append("Stopped by pid: " + "; ".join(
            f"{e['pid']} {e['program']} with {e['signal']} ({e['why']})" for e in stopped) + ".")
    else:
        lines.append("Every process the steps started had ended; none was left to stop.")
    order = ["passed", "failed", "missing", "interrupted", "skipped", "not there yet"]
    lines.append("Result: " + ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k)) + ".")
    if interrupted:
        code = 130
    elif counts.get("failed") or counts.get("missing") or counts.get("skipped"):
        code = 1
    elif counts.get("not there yet"):
        code = 3
    else:
        code = 0
    return "\n".join(lines) + "\n", code


# ---- main ---------------------------------------------------------------------------------------


def parse_steps(text: str) -> list[int]:
    chosen = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        chosen.update(range(int(a), int(b or a) + 1))
    if not chosen or not chosen <= set(range(1, 7)):
        raise argparse.ArgumentTypeError(f"steps are 1 to 6: {text}")
    return sorted(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the output folder, new or empty")
    ap.add_argument("--size", default="1024x640", help="the window's size in points (default 1024x640; 1440x900)")
    ap.add_argument("--steps", type=parse_steps, default=list(range(1, 7)), help="which steps, such as 1-6 or 1,3")
    ap.add_argument("--story", action="store_true", help="one workspace and one training run for all steps")
    ap.add_argument("--caliper", default=os.environ.get("CALIPER_BIN") or str(REPO / "cmake-build-release" / "caliper"),
                    help="the caliper executable (default: CALIPER_BIN, or cmake-build-release/caliper)")
    ap.add_argument("--train-python", help="a Python with torch and Caliper LIB, for train.py")
    ap.add_argument("--agent-python", default=os.environ.get("CALIPER_AGENT_PYTHON"),
                    help="a Python with caliper-agent's dependencies (default: CALIPER_AGENT_PYTHON)")
    ap.add_argument("--model", default=MODEL, help=f"the model Ollama serves for steps 3 and 4 (default {MODEL})")
    ap.add_argument("--ollama", default=OLLAMA, help=f"Ollama on this machine (default {OLLAMA})")
    ap.add_argument("--script", action="append", default=[], metavar="N=FILE",
                    help="run FILE as step N, such as a draft or a stand-in")
    ap.add_argument("--step-timeout", type=float, default=3600.0, help="seconds a step may run (default 3600)")
    ap.add_argument("--in-job", help=argparse.SUPPRESS)
    ap.add_argument("--job-file", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not re.fullmatch(r"\d+x\d+", args.size):
        ap.error(f"--size takes <W>x<H>, such as 1024x640: {args.size}")
    overrides = {}
    for item in args.script:
        n, _, file = item.partition("=")
        if not n.isdigit() or not 1 <= int(n) <= 6 or not file:
            ap.error(f"--script takes N=FILE with N from 1 to 6: {item}")
        overrides[int(n)] = Path(file).resolve()

    class Ctx:
        pass

    ctx = Ctx()
    ctx.began = time.time()
    ctx.out = Path(args.out).resolve()
    ctx.captures, ctx.logs, ctx.work = ctx.out / "captures", ctx.out / "logs", ctx.out / "work"
    ctx.story_dir = ctx.work / "story"
    ctx.size, ctx.story, ctx.model, ctx.ollama = args.size, args.story, args.model, args.ollama
    ctx.caliper = Path(args.caliper).resolve()
    ctx.train_python, ctx.agent_python = args.train_python, args.agent_python
    ctx.step_timeout = args.step_timeout
    ctx.heavy, ctx.model_wrapper = wrapper("CALIPER_DEMO_HEAVY"), wrapper("CALIPER_DEMO_MODEL")
    ctx.notes = []
    ctx.forwarded = ["--out", str(ctx.out), "--size", ctx.size, "--caliper", str(ctx.caliper), "--model", ctx.model,
                     "--ollama", ctx.ollama, "--step-timeout", str(ctx.step_timeout)]
    ctx.forwarded += ["--train-python", ctx.train_python] if ctx.train_python else []
    ctx.forwarded += ["--agent-python", ctx.agent_python] if ctx.agent_python else []
    ctx.forwarded += ["--story"] if ctx.story else []
    for n, f in overrides.items():
        ctx.forwarded += ["--script", f"{n}={f}"]

    inside = args.in_job is not None
    chosen = parse_steps(args.in_job) if inside else args.steps
    steps = [s for s in STEPS if s.n in chosen]
    results = {s.n: {"step": s.n, "title": s.title, "status": "pending"} for s in steps}
    for s in steps:
        s.script = find_script(s, overrides)
        if s.script is None:
            results[s.n].update(status="not there yet", where=f"{s.name}.py is not in dry_run/ yet")
        elif not s.script.is_file():
            results[s.n].update(status="failed", where=f"no file {s.script}")
            s.script = None
    present = [s for s in steps if s.script is not None]
    env = clean_env()

    if not inside:
        if ctx.out.exists() and any(ctx.out.iterdir()):
            print(f"run_all: {ctx.out} is not empty; name a new or empty folder", file=sys.stderr)
            return 2
        problems = prerequisites(ctx, present, env)
        if problems:
            print("run_all: nothing was started, since the selected steps need what is missing:", file=sys.stderr)
            for p in problems:
                print("  " + p, file=sys.stderr)
            return 2
        for d in (ctx.captures, ctx.logs, ctx.work):
            d.mkdir(parents=True, exist_ok=True)
        if ctx.story:
            (ctx.story_dir / "app-data").mkdir(parents=True, exist_ok=True)
    elif ctx.train_python and "PYTHONPATH" not in env and python_lacks(ctx.train_python, "import caliper", env):
        env["PYTHONPATH"] = str(REPO / "python")

    joined: dict[int, bool] = {}
    for s in present:
        plan(s, ctx, joined, env)
        results[s.n].update(interface=s.interface, story=s.story, log=f"logs/step{s.n}.log")
        if s.problem:
            results[s.n].update(status="failed", where=s.problem)
    runnable = [s for s in present if not s.problem]

    stopped: list[dict] = []
    interrupted = None
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        if inside:
            began = time.time()
            try:
                run_job_plainly(runnable, ctx, env, results, stopped)
            except Stop as why:
                interrupted = str(why)
            finally:
                Path(args.job_file).write_text(json.dumps(
                    {"began_at": began, "results": {s.n: results[s.n] for s in steps}, "stopped": stopped}, indent=1))
            return 130 if interrupted else 0
        if not ctx.story:
            say(f"run_all: steps {', '.join(str(s.n) for s in steps)} at {ctx.size}, each from its own fixture")
        else:
            say(f"run_all: the story, steps {', '.join(str(s.n) for s in steps)} at {ctx.size}")
        for s in steps:
            if results[s.n]["status"] != "pending":
                say(f"step {s.n}: {results[s.n]['status']}; {results[s.n]['where']}")
        for k, job in enumerate(group_jobs(runnable, ctx.story), 1):
            words = ctx.model_wrapper if any(s.model for s in job) else ctx.heavy
            if words:
                run_job_wrapped(k, job, words, ctx, env, results, stopped)
            else:
                run_job_plainly(job, ctx, env, results, stopped)
    except Stop as why:
        interrupted = str(why)
        say(f"run_all: interrupted by {why}; stopping what runs")
        for r in results.values():
            if r["status"] == "pending":
                r.update(status="interrupted" if r.get("started_at") else "skipped",
                         where=r.get("where") or f"interrupted by {why}")
    finally:
        if not inside:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            for r in results.values():
                if r["status"] == "pending":
                    r.update(status="skipped", where=r.get("where") or "the run ended before it")
            text, code = summarise(ctx, steps, results, stopped, interrupted)
            (ctx.out / "summary.txt").write_text(text, encoding="utf-8")
            (ctx.out / "summary.json").write_text(json.dumps(
                {"began_at": ctx.began, "story": ctx.story, "size": ctx.size, "out": str(ctx.out),
                 "steps": [results[s.n] for s in steps], "stopped": stopped, "notes": ctx.notes,
                 "exit": code}, indent=1) + "\n", encoding="utf-8")
            print("\n" + text, end="", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
