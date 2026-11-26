"""Runs the six dry runs of the build-a-dashboard walk-through in order and writes one summary.

    python examples/build_a_dashboard/dry_run/run_all.py --out <new folder> [--size 1024x640]
        [--steps 1-6] [--story] [--stream producer|train] [--caliper <app>]
        [--train-python <python>] [--agent-python <python>] [--model example-model]
        [--script <N>=<file>]

Each step is a script of its own in this folder, runnable alone, which plays
the person's part in the hidden app (`caliper --hidden`), with the browser of
step 6 headless, and writes its captures:

    1  step1_train_and_attach     train.py streams and the app attaches to it
    2  step2_drop_and_build       the data files dropped and a dashboard built by hand
    3  step3_cmdk                 ⌘K on a panel: preview, Accept and undo, on the local model
    4  step4_side_chat            the side chat plots the stream, on the local model
    5  step5_outside_agent        an outside agent's edit through caliper-mcp, held and approved
    6  step6_save_reopen_browser  save, reopen from the landing screen, and open in a browser

A step whose script is not in this folder yet is reported as not there yet.
--script N=<file> runs another file as step N, such as a draft or a stand-in.

By default each step starts from its own fixture, in its own workspace. With
--story the steps make one continuous run: one workspace and one app-data
folder for all of them, so that the dashboard built and saved in step 2 (the
one of expected.spec.toml) is the one that ⌘K changes in step 3, that the
outside agent adds to in step 5 and that step 6 saves and reopens. A step that
does not take the story's options yet runs on its own fixture, and the
summary says which options it lacks.

The stream. Step 1 starts its own train.py on its app-data folder and
attaches the app to it through F4, as a person does. The runner starts the
stream that step 4's side chat plots and gives step 4 its link (--url): with
--stream producer (the default) the test tool caliper_attention_producer,
which streams the same views as train.py with no model and no torch; with
--stream train, train.py itself. With --story --stream train, the runner
starts one train.py on the story's app-data folder before step 1 and gives its
link to steps 1 and 4: step 1 attaches to it through F4 and step 4 plots it,
so steps 1 to 4 are one job and train.py trains beside the local model.

What every step is given, the convention of hidden_app.py:

    --caliper <app> --out <captures folder> --work <folder> --size <W>x<H>
    --train-python <python>      steps 1 and 4, which run train.py
    --agent-python <python>      steps 3 and 4, which start caliper-agent
    --model <name>               steps 3 and 4, which call the local model
    --workspace <folder> --app-data <folder>    with --story
    --url <stream link>          step 4, and step 1 with --story --stream train

A step whose --help does not list these options is not run, and the summary
says which it lacks.

The output folder must be new or empty. It receives:

    summary.txt, summary.json   which step passed, which failed and where, and how long each took
    captures/                   the captures of every step, step<N>-<what>-<W>x<H>.png
    logs/                       each step's output, the stream's, and each job's
    work/                       each step's own folder, and the story's workspace and app data

Heavy jobs and the local model. When CALIPER_DEMO_HEAVY is set, each step
runs through it, as `$CALIPER_DEMO_HEAVY "<label>" <command...>`. When
CALIPER_DEMO_MODEL is set, steps 3 and 4 run through it the same way,
together as one job, so that the model stays loaded from one to the other
(steps 1 to 4 with --story --stream train). Unset, the steps run plainly.
Before it starts anything, the runner checks what the selected steps need:
the app, the data files, the Pythons, the producer, Node with Playwright, and
for steps 3 and 4 that Ollama answers on this machine and lists the model. It
never calls the model itself.

The memory watch. While steps 3 and 4 run, the runner reads the free memory
every 5 seconds (memory_pressure, on macOS) and prints it with the resident
size of the model server and of train.py. Two readings in a row below
--memory-low (20 % free), or one below --memory-critical (15 %), stop the
step, the stream and their processes by pid and skip what remains of the job.
The summary gives the lowest reading and the largest resident sizes. With
--memory-baseline, a job with a model step first waits inside its slot, up to
--baseline-wait seconds, for that much memory free, and its steps are skipped
when it does not come.

Every process a step starts (the training script, the app and its sidecar,
the producer, caliper-mcp, the browser) is watched by pid while the step
runs. Whatever a step leaves running when it ends, fails, runs out of time or
is interrupted is stopped by its pid, and the summary lists each one. The
runner sets CALIPER_AGENT_OFFLINE=1 for everything it starts, leaves API keys
and test hooks out of their environment, and deletes nothing.

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
MODEL = "example-model"
OLLAMA = "http://127.0.0.1:11434"
PRODUCER = "caliper_attention_producer"
PRODUCER_ARGS = ["--layers", "4", "--heads", "4", "--tokens", "48", "--hz", "8"]   # train.py's shapes
STORY_TRAIN_SECONDS = 7200          # the story's training run; the runner stops it after its last step
SHELLS = {"sh", "zsh", "bash", "dash"}
# Left out of every environment the runner gives: keys, and hooks that would
# change what the hidden app does.
DROPPED = ("ANTHROPIC_", "OPENAI_", "GEMINI_", "OPENROUTER_", "CALIPER_MODEL_API_KEY", "CALIPER_TEST_",
           "CALIPER_COMMAND_", "CALIPER_AUTOLAUNCH", "CALIPER_DEMO_")


class Stop(Exception):
    """A signal asked the run to stop."""


def on_signal(signum, frame):
    raise Stop(signal.Signals(signum).name)


def ignore_signals():
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)


def catch_signals():
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)


# ---- the steps ----------------------------------------------------------------------------------


class Step:
    def __init__(self, n, name, title, needs=(), model=False, url=False, after=None):
        self.n, self.name, self.title = n, name, title
        self.needs = set(needs)     # train, agent, data, mcp, node
        self.model = model          # calls the local model
        self.url = url              # takes a stream's link
        self.after = after          # in the story, the step whose dashboard it continues
        self.script: Path | None = None
        self.stand_in = False
        self.stream = ""            # the stream the runner gives it: train or producer, or none


STEPS = [
    Step(1, "step1_train_and_attach", "Train a model and attach the app", needs=("train",), url=True),
    Step(2, "step2_drop_and_build", "Drop the data files and build a dashboard by hand", needs=("data",)),
    Step(3, "step3_cmdk", "Change a panel with ⌘K, accept it and undo it", needs=("data", "agent"), model=True,
         after=2),
    Step(4, "step4_side_chat", "Ask the side chat for the attention entropy per layer", needs=("agent",),
         model=True, url=True),
    Step(5, "step5_outside_agent", "Let an outside agent add a panel through caliper-mcp", needs=("data", "mcp"),
         after=2),
    Step(6, "step6_save_reopen_browser", "Save, open again and open in a browser", needs=("data", "node"), after=2),
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
    numbers = {s.n for s in steps}
    needs = set().union(*(s.needs for s in steps)) if steps else set()
    if 4 in numbers and ctx.stream == "train":
        needs.add("train")
    if not ctx.caliper.is_file() or not os.access(ctx.caliper, os.X_OK):
        problems.append(f"No caliper executable at {ctx.caliper}: build the target caliper, or give --caliper.")
    if "data" in needs:
        for f in DATA:
            if not f.is_file():
                problems.append(f"No {f.relative_to(EXAMPLE)}: run python examples/build_a_dashboard/make_data.py.")
    if "train" in needs:
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
    if 4 in numbers and ctx.stream == "producer" and not (ctx.caliper.parent / PRODUCER).is_file():
        problems.append(f"No {PRODUCER} beside {ctx.caliper}: build that target, or give --stream train.")
    if "agent" in needs:
        if not ctx.agent_python:
            problems.append("No Python for caliper-agent: give --agent-python or set CALIPER_AGENT_PYTHON.")
        else:
            why = python_lacks(ctx.agent_python, "import pydantic_ai, openai, anthropic", env)
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
        if not shutil.which("memory_pressure"):
            ctx.notes.append("This machine has no memory_pressure, so the memory watch was off.")
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
    """The command line of one step, in the convention's options, which its --help must list; whether it
    continues the story's workspace; and which stream it is given."""
    conv: list = ["--caliper", ctx.caliper, "--out", ctx.captures, "--work", ctx.work, "--size", ctx.size]
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
        step.argv = [str(a) for a in conv]
    else:
        step.problem = "it does not take the runner's options: it lacks " + ", ".join(missing)
        step.argv = []
    step.story = ""
    step.stream = ""
    takes_url = step.url and not step.problem and (options is None or "--url" in options)
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
            if step.url and (step.n == 4 or ctx.stream == "train"):
                step.stream = ctx.stream
    if step.n == 4 and not step.stream and takes_url:
        step.stream = ctx.stream                  # step 4 is given the runner's stream in either mode
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


# ---- the memory watch ---------------------------------------------------------------------------


def free_percent() -> int | None:
    """The system's free memory in percent, as memory_pressure reports it; None where it cannot."""
    try:
        r = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)\s*%", r.stdout)
    return int(m.group(1)) if m else None


def resident_kb(match) -> int:
    """The summed resident size, in KB, of the processes whose command line match() accepts."""
    r = subprocess.run(["/bin/ps", "-A", "-ww", "-o", "rss=,command="], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    total = 0
    for line in r.stdout.splitlines():
        f = line.split(None, 1)
        if len(f) == 2 and f[0].isdigit() and match(f[1]):
            total += int(f[0])
    return total


def gb(kb) -> str:
    return f"{kb / 1048576:.1f} GB" if kb >= 1048576 else f"{kb / 1024:.0f} MB"


class MemoryWatch:
    """Free memory every 5 s while a model step runs, with the resident sizes of the model server and of
    train.py, and the stop lines: `low` twice in a row, or `critical` once."""

    def __init__(self, ctx, memory: dict):
        self.ctx, self.memory = ctx, memory
        self.tripped = ""
        self.on_trip = None
        self._done = threading.Event()
        self._thread = None

    def start(self, label: str, on_trip):
        if not shutil.which("memory_pressure"):
            return
        self.on_trip = on_trip
        self._done.clear()
        self._thread = threading.Thread(target=self._run, args=(label,), daemon=True)
        self._thread.start()

    def stop(self):
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=40)
            self._thread = None

    def _run(self, label: str):
        low_in_a_row = 0
        while not self._done.is_set():
            free = free_percent()
            # Ollama serves a model from a runner process of its own (llama-server),
            # which holds the weights and the context; `ollama serve` itself is small.
            model = resident_kb(lambda c: Path(c.split()[0]).name in ("ollama", "llama-server"))
            train = resident_kb(lambda c: str(TRAIN) in c and not is_shell(c))
            m = self.memory
            if free is not None:
                m["readings"] = m.get("readings", 0) + 1
                m["lowest_free"] = free if m.get("lowest_free") is None else min(m["lowest_free"], free)
            m["model_server_kb"] = max(m.get("model_server_kb", 0), model)
            m["train_kb"] = max(m.get("train_kb", 0), train)
            say(f"memory, {label}: {free if free is not None else '?'}% free, model server {gb(model)}, "
                f"train.py {gb(train)}")
            if free is not None and not self.tripped:
                low_in_a_row = low_in_a_row + 1 if free < self.ctx.memory_low else 0
                if free < self.ctx.memory_critical:
                    self.tripped = f"{free}% free, below {self.ctx.memory_critical}%"
                elif low_in_a_row >= 2:
                    self.tripped = f"two readings in a row below {self.ctx.memory_low}% free ({free}%)"
                if self.tripped:
                    m["stopped"] = self.tripped
                    say(f"memory: {self.tripped}; stopping {label} and the stream by pid")
                    self.on_trip(self.tripped)
            self._done.wait(5.0)


def wait_for_baseline(ctx) -> int | None:
    """Waits inside the job's slot, up to --baseline-wait seconds, for --memory-baseline percent of memory
    free, reading every 10 s; returns the last reading."""
    end = time.monotonic() + ctx.baseline_wait
    while True:
        free = free_percent()
        say(f"memory, before the job: {free if free is not None else '?'}% free, the baseline is {ctx.memory_baseline}%")
        if free is None or free >= ctx.memory_baseline or time.monotonic() >= end:
            return free
        time.sleep(10)


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
    """The step's captures in the captures folder."""
    return sorted(p.name for p in ctx.captures.glob(f"step{step.n}-*"))


def minutes(seconds) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def say(text: str):
    print(text, flush=True)


def start_stream(kind: str, ctx, env) -> tuple[Watched, str]:
    """A stream and its link: train.py, or the test producer."""
    if kind == "train":
        argv, marker = [ctx.train_python, "-u", str(TRAIN), "--seconds", str(STORY_TRAIN_SECONDS)], str(TRAIN)
        pattern = re.compile(r"^(http://127\.0\.0\.1:\S+)")
        # Its stream file goes to the story's app-data folder, where step 1's F4 lists it, or to a folder of
        # the run's own; never to the person's app data.
        data = ctx.story_dir / "app-data" if ctx.story else ctx.work / "stream-data"
        data.mkdir(parents=True, exist_ok=True)
        env = dict(env, CALIPER_DATA_DIR=str(data))
    else:
        exe = ctx.caliper.parent / PRODUCER
        argv, marker = [str(exe), *PRODUCER_ARGS], str(exe)
        pattern = re.compile(r"\[producer\] hub ((?:ws|http)://127\.0\.0\.1:\S+)")
    log = ctx.logs / f"stream-{kind}.log"
    stream = Watched(kind, argv, env, log, marker)
    end = time.monotonic() + 120
    while time.monotonic() < end and stream.proc.poll() is None:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pattern.search(line)
            if m:
                return stream, m.group(1)
        time.sleep(0.3)
    return stream, ""


def run_job_plainly(steps: list[Step], ctx, env, results: dict, stopped: list, memory: dict) -> None:
    """Runs the steps one after another without a wrapper, each watched by pid, with the streams the
    steps are given, and the memory watch while a model step runs."""
    streams: dict[str, tuple[Watched, str]] = {}
    last_user = {s.stream: s.n for s in steps if s.stream}
    current: Watched | None = None
    watch = MemoryWatch(ctx, memory)
    tripped = ""
    try:
        if ctx.memory_baseline and any(s.model for s in steps):
            free = wait_for_baseline(ctx)
            memory["baseline"] = free
            if free is not None and free < ctx.memory_baseline:
                why = (f"free memory stayed below the baseline of {ctx.memory_baseline}% for "
                       f"{minutes(ctx.baseline_wait)} ({free}% free)")
                memory["stopped"] = why
                say(f"memory: {why}; the job's steps do not start")
                for s in steps:
                    results[s.n].update(status="skipped", where=why)
                return
        for step in steps:
            r = results[step.n]
            if tripped:
                r.update(status="skipped", where=f"the memory watch stopped the job: {tripped}")
                continue
            if ctx.story and step.after and joined_failed(step, results, steps):
                r.update(status="skipped", where=f"step {step.after}, which builds the story's dashboard, did not pass")
                say(f"step {step.n}: skipped, since step {step.after} did not pass")
                continue
            argv = interpreter(step.script) + step.argv
            if step.stream:
                if step.stream not in streams:
                    say(f"stream: starting {'train.py' if step.stream == 'train' else PRODUCER}")
                    streams[step.stream] = start_stream(step.stream, ctx, env)
                    w, link = streams[step.stream]
                    say(f"stream: {step.stream} (pid {w.proc.pid}) at {link.split('?')[0]}" if link
                        else f"stream: {step.stream} printed no link; see logs/stream-{step.stream}.log")
                link = streams[step.stream][1]
                if not link:
                    r.update(status="failed", where=f"the stream ({step.stream}) printed no link")
                    continue
                argv += ["--url", link]
            log = ctx.logs / f"step{step.n}.log"
            say(f"step {step.n}: started, {step.title[0].lower() + step.title[1:]} ({step.script.name}); "
                f"log in logs/{log.name}")
            r["started_at"] = time.time()
            current = Watched(f"step {step.n}", argv, dict(env, **step.env), log, str(step.script))
            if step.model:
                def trip(why, w=current, n=step.n):
                    stopped.extend(dict(e, step=n) for e in w.stop(f"the memory watch: {why}"))
                    for kind, (sw, _) in streams.items():
                        stopped.extend(dict(e, step=n) for e in sw.stop(f"the memory watch: {why}"))
                watch.start(f"step {step.n}", trip)
            code = current.wait(ctx.step_timeout)
            watch.stop()
            r["seconds"] = time.monotonic() - current.started
            if watch.tripped:
                tripped = watch.tripped
                r.update(status="failed", where=f"stopped by the memory watch: {tripped}")
                left = current.stop(f"left running by step {step.n}")
            elif code is None:
                r.update(status="failed", where=f"it ran out of its {minutes(ctx.step_timeout)} and was stopped")
                left = current.stop(f"step {step.n} ran out of time")
            else:
                r["exit"] = code
                r.update(status="passed" if code == 0 else "missing" if code == 2 else "failed")
                if code != 0:
                    r["where"] = where(log)
                left = current.stop(f"left running by step {step.n}")
            stopped.extend(dict(e, step=step.n) for e in left)
            current.close()
            current = None
            r["captures"] = sweep(step, ctx)
            say(f"step {step.n}: {r['status']} in {minutes(r.get('seconds'))}" +
                (f"; {r['where']}" if r.get("where") else "") +
                (f"; stopped {len(left)} process(es) it left running" if left else ""))
            for kind, (w, _) in list(streams.items()):
                if last_user.get(kind) == step.n or tripped:
                    stopped.extend(dict(e, step=step.n) for e in w.stop(f"the stream ({kind}), after its last step"))
                    say(f"stream: stopped {kind} (pid {w.proc.pid}) after step {step.n}")
                    w.close()
                    del streams[kind]
    except Stop as why:
        if current is not None:
            r = results[current_step(steps, results)]
            r.update(status="interrupted", where=f"interrupted by {why}", seconds=time.monotonic() - current.started)
        raise
    finally:
        ignore_signals()
        watch.stop()
        if current is not None:
            stopped.extend(current.stop("the run was interrupted"))
            current.close()
        for kind, (w, _) in streams.items():
            stopped.extend(w.stop(f"the stream ({kind}), at the end of the run"))
            w.close()
        catch_signals()


def current_step(steps, results) -> int:
    started = [s.n for s in steps if results[s.n].get("started_at") and results[s.n]["status"] == "pending"]
    return started[-1] if started else steps[0].n


def joined_failed(step: Step, results: dict, job: list[Step]) -> bool:
    """Whether a step that continues the story's dashboard must be skipped since the step that builds it,
    in the same job, did not pass. A step of an earlier job is checked before that job starts."""
    if step.story != "the story" or step.after not in [s.n for s in job]:
        return False
    return results.get(step.after, {}).get("status") != "passed"


# ---- the jobs, through the wrappers -------------------------------------------------------------


def group_jobs(steps: list[Step], ctx) -> list[list[Step]]:
    if ctx.story and ctx.stream == "train":
        groups = [[1, 2, 3, 4], [5], [6]]    # the training run lives from step 1 to step 4
    else:
        groups = [[1], [2], [3, 4], [5], [6]]
    return [chosen for g in groups if (chosen := [s for s in steps if s.n in g])]


def run_job_wrapped(k: int, job: list[Step], words: list[str], ctx, env, results, stopped, memory) -> None:
    """Runs the job's steps through a wrapper: this script again, inside the wrapper, runs them plainly and
    writes what happened to a file, which is read back here."""
    numbers = ",".join(str(s.n) for s in job)
    result_file = ctx.work / f"job-{k}.json"
    label = f"build_a_dashboard dry run: step{'s' if len(job) > 1 else ''} {numbers}"
    inner = [sys.executable, str(Path(__file__).resolve()), "--in-job", numbers, "--job-file", str(result_file)]
    inner += ctx.forwarded
    log = ctx.logs / f"job-{k}.log"
    say(f"job {k}: step{'s' if len(job) > 1 else ''} {numbers} through {Path(words[-1]).name}; "
        f"it may wait for a slot first")
    launched = time.time()
    job_proc = Watched(f"job {k}", words + [label] + inner, env, log, str(Path(__file__).resolve()))
    relay = threading.Thread(target=tail, args=(log, job_proc), daemon=True)
    relay.start()
    try:
        job_proc.wait(None)
    except Stop:
        ignore_signals()
        stopped.extend(job_proc.stop("the run was interrupted", main_grace=150.0))
        job_proc.close()
        read_job(result_file, job, results, stopped, memory, launched)
        catch_signals()
        raise
    stopped.extend(job_proc.stop(f"left running by job {k}"))
    job_proc.close()
    relay.join(timeout=3)
    read_job(result_file, job, results, stopped, memory, launched)


def read_job(path: Path, job, results, stopped, memory, launched):
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
    merge_memory(memory, doc.get("memory", {}))
    first = next((s.n for s in job if results[s.n].get("started_at")), None)
    if first is not None and doc.get("began_at"):
        results[first]["waited"] = max(0.0, doc["began_at"] - launched)


def merge_memory(into: dict, more: dict):
    if more.get("lowest_free") is not None:
        into["lowest_free"] = more["lowest_free"] if into.get("lowest_free") is None else min(
            into["lowest_free"], more["lowest_free"])
    for key in ("model_server_kb", "train_kb"):
        into[key] = max(into.get(key, 0), more.get(key, 0))
    into["readings"] = into.get("readings", 0) + more.get("readings", 0)
    if more.get("baseline") is not None:
        into["baseline"] = more["baseline"]
    if more.get("stopped"):
        into["stopped"] = more["stopped"]


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


def summarise(ctx, steps, results, stopped, memory, interrupted) -> tuple[str, int]:
    counts = {}
    for s in steps:
        counts[results[s.n]["status"]] = counts.get(results[s.n]["status"], 0) + 1
    waited = any(results[s.n].get("waited") for s in steps)
    began = time.strftime("%Y-%m-%d %H:%M", time.localtime(ctx.began))
    mode = (f"Story: one workspace, the stream {ctx.stream}" if ctx.story
            else f"Steps: each from its own fixture, the stream {ctx.stream}")
    lines = [f"Dry run of the build-a-dashboard walk-through, {began}",
             f"{mode}; window {ctx.size}; output in {ctx.out}", ""]
    lines.append(f"{'step':<5} {'result':<14} {'time':>6} " + (f"{'waited':>7} " if waited else "") + "where")
    for s in steps:
        r = results[s.n]
        row = f"{s.n:<5} {r['status']:<14} {minutes(r.get('seconds')):>6} "
        if waited:
            row += f"{minutes(r.get('waited')) if r.get('waited') else '':>7} "
        row += (r.get("where") or "")[:200]
        if r.get("where") and r["status"] in ("failed", "missing", "interrupted") and r.get("started_at"):
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
        if r.get("stream"):
            notes.append(f"was given the stream of {'train.py' if r['stream'] == 'train' else PRODUCER}")
        if notes:
            lines.append(f"Step {s.n} " + "; ".join(notes) + ".")
    for note in ctx.notes:
        lines.append(note)
    if memory.get("baseline") is not None:
        lines.append(f"Memory before the model job: {memory['baseline']}% free, against a baseline of "
                     f"{ctx.memory_baseline}%.")
    if memory.get("readings"):
        readings = memory["readings"]
        lines.append(f"Memory during the model steps: lowest {memory['lowest_free']}% free in {readings} "
                     f"reading{'s' if readings != 1 else ''}; the model server at most {gb(memory.get('model_server_kb', 0))} resident, train.py "
                     f"at most {gb(memory.get('train_kb', 0))}" +
                     (f"; the watch stopped the job at {memory['stopped']}." if memory.get("stopped") else "."))
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
    ap.add_argument("--story", action="store_true", help="one workspace for all steps, and step 1's stream from the runner")
    ap.add_argument("--stream", choices=("producer", "train"), default="producer",
                    help="what step 4's side chat plots: the test producer (default), or train.py, which with "
                         "--story is step 1's training run and trains beside the local model")
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
    ap.add_argument("--memory-low", type=int, default=20,
                    help="stop the model steps at two readings in a row below this free percentage (default 20)")
    ap.add_argument("--memory-critical", type=int, default=15,
                    help="stop the model steps at one reading below this free percentage (default 15)")
    ap.add_argument("--memory-baseline", type=int, default=0,
                    help="before a job with a model step, wait inside its slot for this free percentage, and skip "
                         "the job when it does not come (default 0, no wait)")
    ap.add_argument("--baseline-wait", type=float, default=300.0,
                    help="seconds to wait for --memory-baseline (default 300)")
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
    ctx.size, ctx.story, ctx.stream, ctx.model, ctx.ollama = args.size, args.story, args.stream, args.model, args.ollama
    ctx.caliper = Path(args.caliper).resolve()
    ctx.train_python, ctx.agent_python = args.train_python, args.agent_python
    ctx.step_timeout = args.step_timeout
    ctx.memory_low, ctx.memory_critical = args.memory_low, args.memory_critical
    ctx.memory_baseline, ctx.baseline_wait = args.memory_baseline, args.baseline_wait
    ctx.heavy, ctx.model_wrapper = wrapper("CALIPER_DEMO_HEAVY"), wrapper("CALIPER_DEMO_MODEL")
    ctx.notes = []
    ctx.forwarded = ["--out", str(ctx.out), "--size", ctx.size, "--caliper", str(ctx.caliper), "--model", ctx.model,
                     "--ollama", ctx.ollama, "--step-timeout", str(ctx.step_timeout), "--stream", ctx.stream,
                     "--memory-low", str(ctx.memory_low), "--memory-critical", str(ctx.memory_critical),
                     "--memory-baseline", str(ctx.memory_baseline), "--baseline-wait", str(ctx.baseline_wait),
                     "--steps", ",".join(str(n) for n in args.steps)]
    ctx.forwarded += ["--train-python", ctx.train_python] if ctx.train_python else []
    ctx.forwarded += ["--agent-python", ctx.agent_python] if ctx.agent_python else []
    ctx.forwarded += ["--story"] if ctx.story else []
    for n, f in overrides.items():
        ctx.forwarded += ["--script", f"{n}={f}"]

    # Inside a job every selected step is planned, so that the plan is the same in each job, and the job's
    # own steps run.
    inside = args.in_job is not None
    in_job = parse_steps(args.in_job) if inside else []
    steps = [s for s in STEPS if s.n in args.steps]
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
        results[s.n].update(interface=s.interface, story=s.story, stream=s.stream, log=f"logs/step{s.n}.log")
        if s.problem:
            results[s.n].update(status="failed", where=s.problem)
    runnable = [s for s in present if not s.problem]

    stopped: list[dict] = []
    memory: dict = {}
    interrupted = None
    code = 1
    catch_signals()
    try:
        if inside:
            began = time.time()
            job = [s for s in runnable if s.n in in_job]
            try:
                run_job_plainly(job, ctx, env, results, stopped, memory)
            except Stop as why:
                interrupted = str(why)
            finally:
                Path(args.job_file).write_text(json.dumps(
                    {"began_at": began, "results": {s.n: results[s.n] for s in job}, "stopped": stopped,
                     "memory": memory}, indent=1))
            return 130 if interrupted else 0
        say(f"run_all: {'the story' if ctx.story else 'steps'} {', '.join(str(s.n) for s in steps)} at {ctx.size}, "
            f"the stream {ctx.stream}")
        for s in steps:
            if results[s.n]["status"] != "pending":
                say(f"step {s.n}: {results[s.n]['status']}; {results[s.n]['where']}")
        for k, job in enumerate(group_jobs(runnable, ctx), 1):
            for s in job:
                if ctx.story and s.story == "the story" and s.after and s.after not in [j.n for j in job] \
                        and results.get(s.after, {}).get("status") != "passed":
                    results[s.n].update(status="skipped",
                                        where=f"step {s.after}, which builds the story's dashboard, did not pass")
                    say(f"step {s.n}: skipped, since step {s.after} did not pass")
            if memory.get("stopped"):
                for s in job:
                    results[s.n].update(status="skipped", where=f"the memory watch stopped the run: {memory['stopped']}")
            job = [s for s in job if results[s.n]["status"] == "pending"]
            if not job:
                continue
            words = ctx.model_wrapper if any(s.model for s in job) else ctx.heavy
            if words:
                run_job_wrapped(k, job, words, ctx, env, results, stopped, memory)
            else:
                run_job_plainly(job, ctx, env, results, stopped, memory)
    except Stop as why:
        interrupted = str(why)
        say(f"run_all: interrupted by {why}; stopping what runs")
        for r in results.values():
            if r["status"] == "pending":
                r.update(status="interrupted" if r.get("started_at") else "skipped",
                         where=r.get("where") or f"interrupted by {why}")
    finally:
        if not inside:
            ignore_signals()
            for r in results.values():
                if r["status"] == "pending":
                    r.update(status="skipped", where=r.get("where") or "the run ended before it")
            text, code = summarise(ctx, steps, results, stopped, memory, interrupted)
            (ctx.out / "summary.txt").write_text(text, encoding="utf-8")
            (ctx.out / "summary.json").write_text(json.dumps(
                {"began_at": ctx.began, "story": ctx.story, "stream": ctx.stream, "size": ctx.size,
                 "out": str(ctx.out), "steps": [results[s.n] for s in steps], "stopped": stopped,
                 "memory": memory, "notes": ctx.notes, "exit": code}, indent=1) + "\n", encoding="utf-8")
            print("\n" + text, end="", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
