"""The hidden app for the dry runs of the build-a-dashboard walk-through.

Holds where the app and the data are, the input steps a person takes to build
the dashboard by hand, and one run of `caliper --hidden` with those steps. The
app plays the steps through ImGui's own event queue, as a person's clicks and
keys arrive (the input driver, src/host/builder/spec_ui_driver.h, lists them),
and writes a PNG for each `shot` step. Nothing is shown on screen.

The app is CALIPER_BIN, or cmake-build-release/caliper in the repository.
Paths given to the `drop` step must not contain spaces or semicolons, since
the driver splits its script on both.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
REPO = EXAMPLE.parent.parent
METRICS = EXAMPLE / "data" / "run_metrics.parquet"
ATTENTION = EXAMPLE / "data" / "attention.safetensors"
SIZES = ["1440x900", "2560x1440"]

# The dashboard's title, and the applet id that New dashboard derives from it.
TITLE = "Training run"
APPLET = "training-run"


def caliper_bin():
    exe = Path(os.environ.get("CALIPER_BIN") or REPO / "cmake-build-release" / "caliper")
    if not exe.is_file():
        sys.exit(f"no Caliper app at {exe}: build the release preset, or set CALIPER_BIN")
    return exe


def check_data():
    for f in (METRICS, ATTENTION):
        if not f.is_file():
            sys.exit(f"no {f.name}: run python examples/{EXAMPLE.name}/make_data.py")
        if any(c in str(f) for c in " ;"):
            sys.exit(f"the driver's drop step cannot take a path with a space or ';': {f}")


def build_steps(shot=None):
    """The steps from the landing screen to the finished dashboard in build
    mode. `shot(name)` gives the steps that capture a moment; without it the
    steps capture nothing and skip the one try that changes nothing (a click
    on Heatmap while the tensor still has four dimensions)."""
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


def settle_and_shot(name):
    """A capture with nothing hovered."""
    return ["wait 10", "leave", "wait 5", f"shot {name}"]


class HiddenApp:
    """One run of `caliper --hidden` on a new workspace under `root`."""

    def __init__(self, root, size, steps, shots, api=False, exit_after=300, extra_env=None):
        self.root = Path(root)
        self.size = size
        self.shots = Path(shots)
        if self.root.exists():
            shutil.rmtree(self.root)
        for d in ("app-data", "applets", "raw"):
            (self.root / d).mkdir(parents=True)
        self.shots.mkdir(parents=True, exist_ok=True)
        self.log_lines = []
        self.listeners = []
        env = dict(os.environ)
        env.update({
            "CALIPER_AGENT_OFFLINE": "1",
            # The side chat finds agents on PATH; these runs open no chat, and a
            # PATH of the system folders keeps every installed agent out of reach.
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "CALIPER_DATA_DIR": str(self.root / "app-data"),
            "CALIPER_USER_APPLETS_DIR": str(self.root / "applets"),
            "CALIPER_WINDOW_SIZE": size,
            "CALIPER_TEST_SPEC_UI": "; ".join(steps),
            "CALIPER_TEST_SPEC_UI_SHOTS": str(self.root / "raw"),
            "CALIPER_EXIT_AFTER": str(exit_after),
        })
        env.pop("CALIPER_API", None)   # the app starts Caliper API only when asked
        env.update(extra_env or {})
        argv = [str(caliper_bin()), "--hidden", "--workspace", str(self.root / "workspace")]
        if api:
            argv += ["--api", "--api-port", "0"]
        # Standard output (where --api prints its link) and error, as one log.
        self.proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors="replace")
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
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()   # closes the window as its close button does
            rc = self.proc.wait(timeout=30)
        self.reader.join(timeout=10)
        return rc

    def ui_lines(self):
        return [l for l in self.log_lines if l.startswith("[ui] ")]

    def ui_done(self):
        return "[ui] done" in self.log_lines

    def raw_shot(self, name):
        return self.root / "raw" / f"{name}.png"

    def collect(self, step, names):
        """Copies the captures as step<N>-<what>-<W>x<H>.png, W x H the window's
        size in points (the pixels are twice that on a Retina display)."""
        out = []
        for i, name in enumerate(names, 1):
            src = self.raw_shot(name)
            if not src.is_file():
                print(f"  missing capture {name}")
                continue
            dst = self.shots / f"step{step}-{i}-{name}-{self.size}.png"
            shutil.copyfile(src, dst)
            out.append(dst)
        return out


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def journal_summary(workspace):
    """The commands of the workspace's journal, counted by origin."""
    import json
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
