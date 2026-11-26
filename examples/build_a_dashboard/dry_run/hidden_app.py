"""What the dry runs of the build-a-dashboard walk-through share.

The options every step takes (add_options), one convention for all six:

    --caliper <app> --out <captures> --work <folder> --size <W>x<H>
    --train-python <python>      the steps that run train.py
    --agent-python <python>      the steps that start caliper-agent
    --model <name>               the steps that call the local model
    --workspace <folder> --app-data <folder>    the story's, which later steps continue in
    --url <stream link>          a stream that is already running, for the steps that attach to one

Captures go straight into --out as step<N>-<what>-<W>x<H>.png, W x H the
window's size in points (the pixels are twice that on a Retina display). A
step's own files go into a new subfolder of --work, or into a temporary folder
that is removed at the end. Exit status: 0 when every check passed, 1 when one
failed, 2 when something the step needs is missing, 130 when interrupted.

The environment of a hidden app (app_env): CALIPER_AGENT_OFFLINE=1, a PATH of
the system's folders only, so that the side chat finds no installed agent, no
API key, test hook or Caliper API setting from the caller's environment (the
app starts Caliper API only when asked), and CALIPER_EXIT_AFTER, so
that the app closes itself should its script be killed outright.

The processes a step starts (Children): each is recorded as it starts and
stopped by its pid when the step ends, fails or is interrupted (SIGINT,
SIGTERM or SIGHUP).

The folders a step deletes (make_temporary, remove_temporary): only the
temporary folders it made itself, recorded when made, and only when the
resolved path is still inside the system's temporary root and is no symbolic
link. A folder the caller names (--out, --work, --workspace, --app-data) is
never removed or cleared, and neither is anything in it from an earlier run: a
step writes into a fresh subfolder of it (fresh_folder).

The hidden app (HiddenApp) plays a script of input steps through ImGui's own
event queue, as a person's clicks and keys arrive (the input driver,
src/host/builder/spec_ui_driver.h, lists them), and writes a PNG for each
`shot` step. Nothing is shown on screen. Paths given to the `drop` step must
not contain spaces or semicolons, since the driver splits its script on both.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
REPO = EXAMPLE.parent.parent
METRICS = EXAMPLE / "data" / "run_metrics.parquet"
ATTENTION = EXAMPLE / "data" / "attention.safetensors"
EXPECTED_SPEC = HERE / "expected.spec.toml"
DEFAULT_SIZE = "1440x900"
MODEL = "example-model"
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

OK, FAILED, MISSING, INTERRUPTED = 0, 1, 2, 130

# The dashboard's title, and the applet id that New dashboard derives from it.
TITLE = "Training run"
APPLET = "training-run"

# Left out of every environment a step gives: keys, and hooks that would change
# what the hidden app does.
DROPPED = ("ANTHROPIC_", "OPENAI_", "GEMINI_", "OPENROUTER_", "CALIPER_MODEL_API_KEY", "CALIPER_TEST_",
           "CALIPER_COMMAND_", "CALIPER_AUTOLAUNCH", "CALIPER_DEMO_", "CALIPER_API")


# ---- options ------------------------------------------------------------------------------------


def add_options(ap, train=False, agent=False, model=False, url=False, sizes=False):
    """Adds the options of the convention to an argument parser."""
    ap.add_argument("--caliper", default=os.environ.get("CALIPER_BIN") or str(REPO / "cmake-build-release" / "caliper"),
                    help="the caliper executable (default: CALIPER_BIN, or cmake-build-release/caliper)")
    ap.add_argument("--out", default=str(REPO / "build" / "build_a_dashboard"), help="where the captures go")
    ap.add_argument("--work", help="a folder in which the step makes a new subfolder for its own files "
                                   "(default: a temporary folder, removed at the end)")
    ap.add_argument("--size", default=DEFAULT_SIZE,
                    help="the window's size in points" + (", or several separated by commas" if sizes else ""))
    ap.add_argument("--workspace", help="the story's workspace, which later steps continue in")
    ap.add_argument("--app-data", help="the story's app-data folder (CALIPER_DATA_DIR)")
    if train:
        ap.add_argument("--train-python", "--python", dest="train_python",
                        help="a Python with torch and Caliper LIB, which runs train.py")
    if agent:
        ap.add_argument("--agent-python", default=os.environ.get("CALIPER_AGENT_PYTHON"),
                        help="a Python with caliper-agent's dependencies (default: CALIPER_AGENT_PYTHON)")
    if model:
        ap.add_argument("--model", default=MODEL, help=f"the model Ollama serves (default {MODEL})")
    if url:
        ap.add_argument("--url", help="attach to this stream, which is already running")


def caliper_problem(path):
    """Why the caliper executable cannot be used, or ''."""
    exe = Path(path)
    if not exe.is_file() or not os.access(exe, os.X_OK):
        return f"no caliper executable at {exe}: build the target caliper, or give --caliper"
    return ""


def data_problem():
    """Why the data files cannot be used, or ''."""
    for f in (METRICS, ATTENTION):
        if not f.is_file():
            return f"no {f.name}: run python examples/{EXAMPLE.name}/make_data.py"
        if any(c in str(f) for c in " ;"):
            return f"the driver's drop step cannot take a path with a space or ';': {f}"
    return ""


# ---- folders ------------------------------------------------------------------------------------

_TEMPORARY = {}          # each temporary folder this process made: its path as made -> its resolved path


def temporary_root():
    return Path(tempfile.gettempdir()).resolve()


def make_temporary(prefix):
    """A new temporary folder, recorded as made by this process, so that remove_temporary may remove it."""
    made = Path(tempfile.mkdtemp(prefix=prefix))
    _TEMPORARY[made] = made.resolve()
    return made


def remove_temporary(path):
    """Removes a temporary folder this process made with make_temporary. Refuses, and returns why, for any
    other path: one not recorded, one that is now a symbolic link, or one whose resolved path is not
    inside the temporary root. Returns '' when it removed the folder."""
    path = Path(path)
    if path not in _TEMPORARY:
        return f"refused to remove {path}: this script did not make it"
    if path.is_symlink():
        return f"refused to remove {path}: it is a symbolic link"
    resolved = path.resolve()
    root = temporary_root()
    if resolved != _TEMPORARY[path] or resolved == root or root not in resolved.parents:
        return f"refused to remove {path}: it is not inside the temporary root {root}"
    shutil.rmtree(resolved, ignore_errors=True)
    del _TEMPORARY[path]
    return ""


def remove_all_temporary():
    for made in list(_TEMPORARY):
        why = remove_temporary(made)
        if why:
            print(why, flush=True)


def fresh_folder(parent, name):
    """A new, empty folder under parent: parent/name, or parent/name-2, -3 and so on when that name is
    taken. Nothing already in parent is changed."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        folder = parent / (name if n == 1 else f"{name}-{n}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise RuntimeError(f"no free name for {name} in {parent}")


def work_folder(args, step):
    """The step's own folder: a fresh subfolder of --work, or a temporary folder removed at the end."""
    if args.work:
        return fresh_folder(Path(args.work).resolve(), f"step{step}")
    return make_temporary(f"caliper-step{step}-")


# ---- processes ----------------------------------------------------------------------------------


class Stop(Exception):
    """A signal asked the step to stop."""


class Children:
    """Every process a step starts, recorded as it starts, and stopped by its pid when the step ends,
    fails or is interrupted. Used as a context manager, it turns SIGINT, SIGTERM and SIGHUP into Stop,
    and on the way out stops what still runs and removes the temporary folders the step made."""

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self):
        self.procs = []          # (name, Popen, the signal that stops it)
        self._starting = False
        self._pending = None
        self._previous = {}

    def _on_signal(self, signum, frame):
        if self._starting:       # raised once the new process is recorded
            self._pending = signal.Signals(signum).name
            return
        raise Stop(signal.Signals(signum).name)

    def __enter__(self):
        self._previous = {s: signal.signal(s, self._on_signal) for s in self.SIGNALS}
        return self

    def __exit__(self, kind, value, tb):
        for s in self.SIGNALS:
            signal.signal(s, signal.SIG_IGN)
        try:
            self.stop_all()
            remove_all_temporary()
        finally:
            for s, handler in self._previous.items():
                signal.signal(s, handler)
        return False

    def start(self, name, argv, stop_signal=signal.SIGTERM, **popen):
        """Starts a process and records it before a signal can interrupt the step."""
        self._starting = True
        try:
            p = subprocess.Popen(argv, **popen)
            self.procs.append((name, p, stop_signal))
        finally:
            self._starting = False
        if self._pending:
            why, self._pending = self._pending, None
            raise Stop(why)
        return p

    def stop(self, p, timeout=30.0):
        """Stops one recorded process by its pid, with its signal and then SIGKILL, and says so."""
        for name, q, sig in self.procs:
            if q is p and p.poll() is None:
                print(f"stopping {name} (pid {p.pid}) with {sig.name}", flush=True)
                p.send_signal(sig)
                try:
                    p.wait(timeout)
                except subprocess.TimeoutExpired:
                    print(f"stopping {name} (pid {p.pid}) with SIGKILL", flush=True)
                    p.kill()
                    p.wait()
        return p.poll()

    def stop_all(self):
        for _, p, _ in reversed(self.procs):
            self.stop(p)


def clean_env():
    """The caller's environment without keys and test hooks, and with CALIPER_AGENT_OFFLINE=1."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROPPED)}
    env["CALIPER_AGENT_OFFLINE"] = "1"
    return env


def app_env(app_data, user_applets, size, steps=None, shots=None, exit_after=300, extra=None):
    """The environment of a hidden app."""
    env = clean_env()
    env.update({
        # The side chat finds agents on PATH; a PATH of the system folders keeps every installed agent
        # out of reach of an app that opens the chat without naming its agent.
        "PATH": SYSTEM_PATH,
        "CALIPER_DATA_DIR": str(app_data),
        "CALIPER_USER_APPLETS_DIR": str(user_applets),
        "CALIPER_WINDOW_SIZE": size,
        "CALIPER_EXIT_AFTER": str(int(exit_after)),
    })
    if steps is not None:
        env["CALIPER_TEST_SPEC_UI"] = steps if isinstance(steps, str) else "; ".join(steps)
    if shots is not None:
        env["CALIPER_TEST_SPEC_UI_SHOTS"] = str(shots)
    env.update(extra or {})
    return env


class HiddenApp:
    """One run of `caliper --hidden` with a script of input steps, in the folder root (made by the caller
    and kept), on the workspace and app-data folder given or new ones under root. Its standard output
    and error are one log, root/app.log, read line by line."""

    def __init__(self, children, caliper, root, size, steps, workspace=None, app_data=None, api=False,
                 exit_after=300, args=(), extra_env=None):
        self.root = Path(root)
        self.size = size
        self.workspace = Path(workspace) if workspace else self.root / "workspace"
        self.app_data = Path(app_data) if app_data else self.root / "app-data"
        for d in (self.app_data, self.root / "applets", self.root / "raw"):
            d.mkdir(parents=True, exist_ok=True)
        self.log_lines = []
        self.listeners = []
        env = app_env(self.app_data, self.root / "applets", size, steps, self.root / "raw", exit_after, extra_env)
        argv = [str(caliper), "--hidden", "--workspace", str(self.workspace), *args]
        if api:
            argv += ["--api", "--api-port", "0"]
        self.children = children
        # Standard output (where --api prints its link) and error, as one log.
        self.proc = children.start("the app", argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        with open(self.root / "app.log", "w") as log:
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                self.log_lines.append(line.rstrip("\n"))
                for f in list(self.listeners):
                    f(line.rstrip("\n"))

    def wait(self, timeout=360):
        """Waits for the app to quit, and stops it by its pid when it has not by then."""
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = self.children.stop(self.proc)
        self.reader.join(timeout=10)
        return rc

    def stop(self):
        """Closes the app as its close button does (SIGTERM, which the hidden app takes as a close)."""
        rc = self.children.stop(self.proc)
        self.reader.join(timeout=10)
        return rc

    def ui_lines(self):
        return [l for l in self.log_lines if l.startswith("[ui] ")]

    def ui_done(self):
        return "[ui] done" in self.log_lines

    def raw_shot(self, name):
        return self.root / "raw" / f"{name}.png"

    def collect(self, step, names, out):
        """Copies the captures into out as step<N>-<n>-<what>-<W>x<H>.png."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        copied = []
        for i, name in enumerate(names, 1):
            src = self.raw_shot(name)
            if not src.is_file():
                print(f"  missing capture {name}")
                continue
            dst = out / f"step{step}-{i}-{name}-{self.size}.png"
            shutil.copyfile(src, dst)
            copied.append(dst)
        return copied


# ---- the dashboard built by hand ----------------------------------------------------------------


def build_steps(shot=None):
    """The steps from the landing screen to the finished dashboard in build mode. `shot(name)` gives the
    steps that capture a moment; without it the steps capture nothing and skip the one try that changes
    nothing (a click on Heatmap while the tensor still has four dimensions)."""
    try_heatmap = ["click primitive:heatmap", "wait 40", "shot heatmap-refused"] if shot else []
    shot = shot or (lambda name: [])
    return [
        # The landing screen; ⌘K puts the keyboard in its command line, whose
        # text F3 New dashboard takes as the title. The page opens in build mode.
        "wait 60", "page landing", *shot("landing"),
        "key cmd+k", f"type {TITLE}", "key f3", "page spec", *shot("new-dashboard"),
        # Both files dropped on the window: they are imported and listed in Sources.
        f"drop {METRICS} {ATTENTION}",
        "pane source table:run_metrics", "pane source tensor:attention_attn", *shot("sources"),
        # ⌘T: the spec text under the canvas, which follows every edit.
        "key cmd+t", *shot("spec-text-empty"),
        # The series: the metrics table, then Series. It takes step as x and the
        # other columns as lines; the lr line is removed, a title and an x label set.
        "click source:table:run_metrics 0.6 0.5", "click primitive:series", "pane focus series",
        "click bind.remove:y:2",
        "click prop:title", "type Loss and accuracy", "key enter",
        "click option:x_label", "type Step", "key enter", *shot("series"),
        # The heatmap: the attention tensor is [layer, head, query, key], and a
        # heatmap takes two dimensions, so Heatmap is faint and a click on it
        # sends nothing; two slices pick layer 1 (axis 0, index 1) and then
        # head 2 of it (axis 0 again, index 2).
        "click source:tensor:attention_attn 0.6 0.5", *try_heatmap,
        "click transform:add:slice", "click transform:add:slice",
        "click transform:0:index", "type 1", "key enter",
        "click transform:1:index", "type 2", "key enter", *shot("slices"),
        "click primitive:heatmap", "pane focus heatmap",
        "click prop:title", "type Attention of layer 1, head 2", "key enter",
        "click option:x_label", "type Key", "key enter",
        "click option:y_label", "type Query", "key enter", *shot("heatmap"),
        # The table: the metrics source again, Table dragged to the bottom edge of
        # the canvas, under both panels, and the divider above it dragged down.
        "click source:spec:run_metrics 0.6 0.5",
        "drag primitive:table to canvas 0.5 -4", "pane focus table",
        "drag divider: 0.5 0.0 to canvas 0.5 0.62", *shot("table"),
    ]


def fixture_commands(metrics=METRICS, attention=ATTENTION):
    """The dashboard of expected.spec.toml, the one step 2 builds by hand, as a command script: the data files
    imported, the spec made and saved as the applet training-run. Steps that start from their own fixture
    build this one, so that alone and in the story they work on the same dashboard."""
    return [
        {"name": "data.import", "args": {"paths": [str(metrics), str(attention)], "run": True}},
        {"name": "spec.new", "args": {"id": APPLET, "title": TITLE}},
        {"name": "source.add", "args": {"id": "run_metrics", "catalog": "table:run_metrics"}},
        # attn is layers x heads x query x key; layer 1, head 2 is an induction head.
        {"name": "source.add", "args": {"id": "attention_attn", "catalog": "tensor:attention_attn",
                                        "transform": [{"op": "slice", "axis": 0, "index": 1},
                                                      {"op": "slice", "axis": 0, "index": 2}]}},
        {"name": "panel.create", "args": {"id": "series", "primitive": "series", "title": "Loss and accuracy",
                                          "bind": {"x": "run_metrics.step",
                                                   "y": ["run_metrics.loss", "run_metrics.accuracy"]},
                                          "options": {"x_label": "Step"}}},
        {"name": "panel.create", "args": {"id": "heatmap", "primitive": "heatmap",
                                          "title": "Attention of layer 1, head 2", "bind": {"values": "attention_attn"},
                                          "options": {"x_label": "Key", "y_label": "Query"}}},
        {"name": "panel.create", "args": {"id": "table", "primitive": "table", "bind": {"rows": "run_metrics"}}},
        {"name": "layout.set", "args": {"layout": {"root": "n1", "nodes": [
            {"id": "n1", "split": "vertical", "ratio": 0.62, "children": ["n2", "table"]},
            {"id": "n2", "split": "horizontal", "children": ["series", "heatmap"]}]}}},
        {"name": "applet.save", "args": {"overwrite": True}},
    ]


def settle_and_shot(name):
    """A capture with nothing hovered."""
    return ["wait 10", "leave", "wait 5", f"shot {name}"]


def saved_spec(workspace):
    """The saved dashboard's spec file in a workspace."""
    return Path(workspace) / "applets" / APPLET / f"{APPLET}.spec.toml"


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def journal_summary(workspace):
    """The commands of the workspace's journal, counted by origin."""
    counts = {}
    path = Path(workspace) / "journal" / "commands.jsonl"
    if not path.is_file():
        return counts
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        key = (e.get("origin", "?"), e.get("name", "?"))
        counts[key] = counts.get(key, 0) + 1
    return counts
