#!/usr/bin/env python3
"""The acceptance test of instrument building (design/instrument-interface.md, Acceptance Test) and its check script.

It starts the hidden app with no script and caliper-mcp as a child over stdio, copies examples/mnist/train.py
into a throwaway workspace, and builds an instrument on the managed run of that script with MCP tools
only: a terrain from the surface operator over a live module output (live:fc1.act), the loss as a line in the same
scene, state variables, controls, a heatmap that follows a pick through a reference, a time cursor with a transport,
bound text, annotations and layout shares. It checks each part by state, events and panel data, not by eye. Each step
is one method of Test, named step_<letter>_<name>, and the test stops with Fail(...) at the first wrong answer.
The message names the step, the tool and the answer of the tool. It waits on events and on published state,
never for a fixed time, and it asserts no update rate. It records the SHA-256 of the app, caliper-mcp and the
Python package before and after, and checks by pid that no child process is left. Every number it records goes
to <work>/result.json, also after a failure. The test runs without a window and without the network.
Exit code 0 passes, 77 skips, 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir>
                 [--capture-dir <dir>] [--stop-after <step letter or name>] [--limit-s 540]
"""
import argparse
import base64
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "instrument-building-check", "version": "1"}}

# The edit commands that the test sends, counted. Each one needs one decision at its card. The test seam of
# the app reads one line "approve <command> <origin>" for each card. edit() refuses a send over the count.
EDITS = {"spec.new": 1, "state.create": 3, "source.add": 9, "control.create": 9, "panel.create": 5,
         "annotation.set": 4, "layout.set": 1, "entity.set": 1, "entity.remove": 1, "source.remove": 1}
# Commands of the effect "run". A line that finds no card waits for one and does no harm.
RUNS = {"run.start": 1, "run.stop": 1, "run.set_param": 1, "intervene.set": 1, "intervene.clear": 1}
APPROVALS = [f"approve {c} mcp" for table in (EDITS, RUNS) for c, n in table.items() for _ in range(n)]

READOUT_FORMAT = "Step {step:,d}  Unit {unit:d}  Value {value:.3f}"
READOUT_FIELDS = {"step": {"state": "cursor", "field": "step"}, "unit": {"state": "unit"},
                  "value": {"selection": "world", "entity": "terrain", "field": "value"}}
ACT_ID = "live:fc1.act"
HEAD_ID = "live:fc2.act"
LOSS_ID = "live:loss"
MATERIAL_WGSL = """fn material(in: MaterialIn) -> vec4<f32> {
  if (abs(in.value - params.level) < 0.01) { return vec4<f32>(1.0, 0.6, 0.2, 1.0); }
  return in.color;
}
"""


class Fail(Exception):
    pass


class Skip(Exception):
    pass


class Mcp:
    """caliper-mcp over stdio: one JSON line for each request and one for each answer."""

    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env=env)
        self.n = 0
        self.first_call = None   # time.monotonic() of the first tool call

    def request(self, method, params):
        self.n += 1
        params = dict(params, _meta=META)
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise Fail("caliper-mcp closed its output")
            m = json.loads(line)
            if m.get("id") == self.n and "method" not in m:
                return m

    def call(self, tool, arguments):
        if self.first_call is None:
            self.first_call = time.monotonic()
        r = self.request("tools/call", {"name": tool, "arguments": arguments})
        if "result" not in r:
            raise Fail(f"{tool}: protocol answer without a result: {json.dumps(r)[:600]}")
        return r["result"]


def stop(p):
    """Stops a child by its own pid: SIGTERM, then SIGKILL after 20 seconds."""
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")   # a zombie has no code to run


def pids_of(script_path):
    """The pids of the processes whose command line names the script file."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if str(script_path) in cmd and int(pid) != os.getpid():
            found.append(int(pid))
    return found


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_files():
    """The files of the Python package: the sources and the native modules, not the caches."""
    pkg = ROOT / "python" / "caliper"
    return sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def decode_png(data):
    """The pixels of an 8-bit RGB or RGBA PNG, read with the decoder of the GPU programs test."""
    spec = importlib.util.spec_from_file_location("caliper_gpu_programs_driver", ROOT / "tests" / "gpu_programs" / "driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode_png(data)


def pixel_difference(a, b):
    """How many pixels differ between two PNG images of the same size."""
    import numpy as np
    x, y = decode_png(a), decode_png(b)
    if x.shape != y.shape:
        raise Fail(f"two renders have the shapes {x.shape} and {y.shape}")
    return int((np.abs(x.astype(int) - y.astype(int)).max(axis=2) > 8).sum())


def tool_name(command):
    return command.replace(".", "_")


def command_name(tool):
    return tool.replace("_", ".", 1)   # run_set_param is run.set_param


def find_entity(node, path):
    """The first dict in a panel_data answer that describes the entity `path`. The shape of the answer is
    not fixed yet, so the search is by key and by value."""
    if isinstance(node, dict):
        if node.get("entity") == path or node.get("path") == path:
            return node
        if isinstance(node.get(path), dict):
            return node[path]
        for v in node.values():
            found = find_entity(v, path)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_entity(v, path)
            if found is not None:
                return found
    return None


class Test:
    def __init__(self, mcp, ws, work, capture_dir, args):
        self.mcp = mcp
        self.ws = ws
        self.work = work
        self.capture_dir = capture_dir
        self.args = args
        self.cur = "start"
        self.cursor = 0           # event cursor
        self.events = []
        self.sent = collections.Counter()
        self.pids = set()
        self.procs = {}           # name -> Popen, for step o
        self.shots = 0
        self.run = None
        self.loss_id = None
        self.toggle_point = None
        self.res = {"steps": {}}

    def digest(self):
        """The SHA-256 of the app, caliper-mcp and each file of the Python package."""
        files = [Path(self.args.app).resolve(), Path(self.args.mcp).resolve()] + package_files()
        return {str(p): sha256(p) for p in files}

    # -- calls -------------------------------------------------------------------------------------------
    def answer(self, tool, arguments, allow_error=False):
        """One tool call. The raw result when allow_error, else the structured body, or Fail."""
        r = self.mcp.call(tool, arguments)
        if allow_error:
            return r
        if r.get("isError"):
            raise Fail(f"step {self.cur}: {tool} {json.dumps(arguments)[:400]} answered an error: "
                       f"{json.dumps(r)[:1200]}")
        return r.get("structuredContent", {})

    def edit(self, tool, arguments, allow_error=False):
        cmd = command_name(tool)
        self.sent[cmd] += 1
        if self.sent[cmd] > EDITS.get(cmd, 0):
            raise Fail(f"step {self.cur}: {tool} is sent {self.sent[cmd]} times, and the approvals cover "
                       f"{EDITS.get(cmd, 0)}")
        body = self.answer(tool, arguments, allow_error)
        if not allow_error and body.get("applied") is False:
            raise Fail(f"step {self.cur}: {tool} did not apply: {json.dumps(body)[:1200]}")
        # A source with no panel yet is a warning while the instrument is being built.
        warned = [w for w in body.get("warnings") or [] if w.get("code") != "spec.unused_source"]
        if not allow_error and warned:
            print(f"  warnings of {tool}: {json.dumps(warned)[:600]}")
        return body

    # -- waits on events and on published state ----------------------------------------------------------
    def pace(self, seconds):
        b = self.answer("events_wait", {"cursor": self.cursor, "timeout_s": seconds})
        self.cursor = b.get("cursor", self.cursor)
        self.events.extend(b.get("events", []))

    def until(self, what, pred, limit=60):
        end = time.monotonic() + limit
        while True:
            v = pred()
            if v:
                return v
            if time.monotonic() > end:
                raise Fail(f"step {self.cur}: {what}: not reached within {limit} s")
            self.pace(1)

    def mark(self):
        """A position in the event list: event(since=mark) ignores earlier events."""
        return len(self.events)

    def event(self, what, match, since=0, limit=60):
        def find():
            for e in self.events[since:]:
                if all(e.get(k) == v for k, v in match.items()):
                    return e
            return None
        return self.until(f"event {what} {match}", find, limit)

    def runs(self):
        return {r["id"]: r for r in self.answer("run_list", {}).get("runs", [])}

    def run_step(self):
        return self.answer("run_state", {"run": self.run}).get("step") or 0

    def cursor_now(self):
        return self.answer("cursor_get", {})["cursor"]

    def pdata(self, panel):
        return self.answer("panel_data", {"panel": panel})

    def entity(self, panel, path):
        """One entity of a scene panel's data, or None."""
        for ent in (self.pdata(panel).get("data") or {}).get("entities", []):
            if ent.get("path") == path:
                return ent
        return None

    def terrain(self):
        """The terrain entity of world: its row count, its x values and the panel data's version."""
        b = self.pdata("world")
        (self.work / "world_data.json").write_text(json.dumps(b)[:200000])
        rows, cols, xs = 0, 0, []
        for ent in (b.get("data") or {}).get("entities", []):
            if ent.get("path") != "terrain":
                continue
            for slot in ent.get("slots", []):
                if slot.get("slot") == "height" and slot.get("shape"):
                    rows = slot["shape"][0]
                    cols = slot["shape"][1] if len(slot["shape"]) > 1 else 0
                if slot.get("slot") == "x":
                    xs = [v for v in slot.get("values", []) if v is not None]
        return {"rows": rows, "cols": cols, "x": xs, "version": b.get("version"), "state": b.get("state"),
                "error": b.get("error")}

    def kept_gap(self):
        """The largest gap between two kept steps of the source act, from the x values of the terrain. The script
        takes a snapshot at each pass of the shared cadence, a few times a second, and the app keeps the ones that
        it reads, thus kept steps stand apart by that gap, not by one step."""
        xs = sorted(set(self.terrain()["x"]))
        return int(max((b - a for a, b in zip(xs, xs[1:])), default=0))

    # -- captures ----------------------------------------------------------------------------------------
    @staticmethod
    def png_of(result):
        for c in result.get("content", []):
            if c.get("type") == "image":
                return base64.b64decode(c["data"])
        raise Fail(f"a result holds no image: {json.dumps(result)[:400]}")

    def render(self, panel="world"):
        r = self.answer("panel_render", {"id": panel, "width": 960, "height": 640}, allow_error=True)
        if r.get("isError"):
            raise Fail(f"step {self.cur}: panel_render {panel} answered an error: {json.dumps(r)[:800]}")
        return self.png_of(r)

    def save(self, name, data):
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.shots += 1
        out = self.capture_dir / f"{self.shots:02d}-{name}.png"
        out.write_bytes(data)
        self.res.setdefault("captures", []).append(str(out))
        return out

    def capture(self, name, panel_png=None):
        """capture_frame as NN-<name>-frame.png, and a render of world as NN+1-<name>.png unless given."""
        r = self.answer("capture_frame", {}, allow_error=True)
        if r.get("isError"):
            raise Fail(f"step {self.cur}: capture_frame answered an error: {json.dumps(r)[:800]}")
        self.save(name + "-frame", self.png_of(r))
        if panel_png is None:
            panel_png = self.render("world")
        self.save(name, panel_png)

    # -- the steps ---------------------------------------------------------------------------------------
    def step_a_start(self):
        """The app and caliper-mcp are running. The time of the first tool call starts the clock."""
        info = self.answer("app_info", {})
        self.res["t_first_call_monotonic"] = self.mcp.first_call
        self.res["t_first_call_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.res["processes"] = {k: p.pid for k, p in self.procs.items()}
        self.res["app_info"] = info
        self.res["versions"] = {"app_info": info.get("version")}
        self.hashes = self.digest()
        self.res["sha256_files"] = len(self.hashes)

    def step_b_run(self):
        """spec_new, run_start of train.py, the attach, a growing step and the catalog."""
        self.edit("spec_new", {"id": "instrument_building_check", "title": "Instrument Building Check"})
        r = self.answer("run_start", {
            "script": "train.py",
            "args": ["--watch-activations", "--probe-size", "64", "--test-batch-size", "64",
                     "--probe-every", "50", "--pace", "0.015"],
        })
        self.run = r["run"]["id"]
        self.res["run"] = self.run
        self.until("the run is running and attached",
                   lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(self.runs().get(self.run)),
                   120)
        found = pids_of(self.ws / "train.py")
        if not found:
            raise Fail(f"step {self.cur}: no process runs train.py")
        self.pids.update(found)
        self.res["train_pids"] = sorted(self.pids)
        first = self.until("a first step", self.run_step, 60)
        self.until("the step grows", lambda: self.run_step() > first, 60)
        self.res["steps"]["run_first"], self.res["steps"]["run_later"] = first, self.run_step()
        seen = []

        def tiled():
            entries = self.answer("catalog_list", {"limit": 1000}).get("entries", [])
            by = {e["id"]: e for e in entries}
            shape = by.get(ACT_ID, {}).get("shape")
            if not seen or seen[-1] != shape:
                seen.append(shape)
            return by if shape == [64, 128] else None
        try:
            by_id = self.until(f"{ACT_ID} lists as (64, 128)", tiled, 60)
        finally:
            self.res["act_shapes_seen"] = seen
            print(f"  shapes of {ACT_ID} seen: {seen}")
        for need in (ACT_ID, HEAD_ID):
            if need not in by_id:
                raise Fail(f"step {self.cur}: catalog_list lacks {need}; it lists {sorted(by_id)[:40]}")
        self.res["shapes"] = {i: by_id[i].get("shape") for i in (ACT_ID, HEAD_ID)}
        if by_id[ACT_ID].get("shape") != [64, 128]:
            raise Fail(f"step {self.cur}: {ACT_ID} has shape {by_id[ACT_ID].get('shape')}, not [64, 128]")
        if LOSS_ID not in by_id:
            raise Fail(f"step {self.cur}: catalog_list lacks {LOSS_ID}; it lists {sorted(by_id)[:40]}")
        pick = LOSS_ID
        self.res["loss_entry"] = by_id[LOSS_ID]
        self.loss_id = pick
        self.res["loss_id_used"] = pick
        print(f"  loss id used: {pick}")

    def step_c_state(self):
        """State variables unit, measure and palette, and the source act with 2048 kept steps."""
        self.edit("state_create", {"id": "unit", "title": "Unit", "type": "integer", "min": 0, "max": 63,
                                   "default": 0})
        self.edit("state_create", {"id": "measure", "title": "Measure", "type": "text",
                                   "choices": ["raw", "zscore"], "default": "raw"})
        self.edit("state_create", {"id": "palette", "title": "Palette", "type": "text",
                                   "choices": ["viridis", "magma"], "default": "viridis"})
        # The batch norm of each of the first 64 hidden units of fc1.act, kept for 2048 steps.
        self.edit("source_add", {"id": "act", "catalog": ACT_ID, "history": 2048,
                                 "transform": [{"op": "slice", "axis": 1, "start": 0, "stop": 64},
                                               {"op": "reduce", "fn": "norm", "axis": 0}]})

    def step_d_scene(self):
        """The range window, the terrain, the kept loss and its window, the scene world and the slider level."""
        self.edit("control_create", {
            "id": "window", "kind": "range", "title": "Steps", "bind": "value",
            "options": {"min": {"state": "cursor", "field": "first"}, "max": {"state": "cursor", "field": "latest"},
                        "value": [0, 2048], "min_span": 16, "follow": True}})
        self.edit("source_add", {"id": "terrain", "from": "act", "transform": [
            {"op": "window", "start": {"control": "window", "field": "lo"},
             "stop": {"control": "window", "field": "hi"}},
            {"op": "surface", "max_cells": 32768}]})
        # The loss series keeps its newest value at each step only for a source that states history.
        self.edit("source_add", {"id": "loss", "catalog": self.loss_id, "history": 2048})
        self.edit("source_add", {"id": "loss_window", "from": "loss", "transform": [{"op": "window", "steps": 2048}]})
        self.edit("panel_create", {
            "id": "world", "primitive": "scene", "title": "Activation Terrain",
            "entity": [
                {"path": "terrain", "draw": "surface", "bind": "terrain",
                 "options": {"colormap": {"state": "palette"}}},
                {"path": "loss", "draw": "lines", "title": "Loss",
                 "bind": {"x": "loss_window.step", "z": "loss_window.value"},
                 "options": {"mode": "strip", "y": "max", "own": ["z"], "width_px": 2.0}}],
            "options": {"axes": True, "up": "+z", "box": [3.0, 1.0, 0.8],
                        "select": {"state": "unit", "field": "column", "entity": "terrain"}}})
        self.edit("control_create", {"id": "level", "kind": "slider", "title": "Level", "bind": "value",
                                     "options": {"min": 0.0, "max": 200.0, "value": 1.0}})

    def step_e_controls(self):
        """Sources raw and zscore, the panel detail, the controls measure, palette, now, transport, pause,
        the toggle and the lr slider."""
        # The picked unit pair (unit, unit + 64) across the first 16 batch items, raw and z-scored per unit.
        unit_slice = [{"op": "slice", "axis": 0, "start": 0, "stop": 16},
                      {"op": "slice", "axis": 1, "start": {"state": "unit"}, "step": 64}]
        self.edit("source_add", {"id": "raw", "catalog": ACT_ID, "transform": unit_slice})
        self.edit("source_add", {"id": "zscore", "catalog": ACT_ID, "transform": unit_slice + [
            {"op": "normalise", "mode": "zscore", "axis": 0}]})
        self.edit("panel_create", {"id": "detail", "primitive": "heatmap", "title": "Unit Detail",
                                   "bind": {"values": {"state": "measure"}},
                                   "options": {"x_label": "Pair", "y_label": "Batch"}})
        # The picked unit's value over the window's steps, for the series panel trace.
        self.edit("source_add", {"id": "trace", "from": "act", "transform": [
            {"op": "window", "start": {"control": "window", "field": "lo"},
             "stop": {"control": "window", "field": "hi"}},
            {"op": "slice", "axis": 1, "index": {"state": "unit"}}]})
        # The picked unit's norm at the cursor, as one value, for the bound text.
        self.edit("source_add", {"id": "focus", "from": "act", "transform": [
            {"op": "slice", "axis": 0, "start": {"state": "unit"}, "step": 64}]})
        self.edit("control_create", {
            "id": "measure", "kind": "segmented", "title": "Measure", "bind": "state:measure",
            "options": {"choices": [{"value": "raw", "label": "Raw", "key": "r"},
                                    {"value": "zscore", "label": "Z-Score", "key": "z"}]}})
        self.edit("control_create", {
            "id": "palette", "kind": "segmented", "title": "Palette", "bind": "state:palette",
            "options": {"choices": [{"value": "viridis", "label": "Viridis"}, {"value": "magma", "label": "Magma"}]}})
        self.edit("control_create", {
            "id": "now", "kind": "readout", "title": "Now",
            "options": {"format": READOUT_FORMAT, "fields": READOUT_FIELDS}})
        self.edit("control_create", {"id": "transport", "kind": "transport", "title": "Time", "bind": "cursor"})
        self.edit("control_create", {"id": "pause", "kind": "button", "title": "Pause Run", "bind": "run:pause"})
        # The toggle binds an intervention at a module output, because a toggle may bind intervene:, param:,
        # value or state:, and the run registers no boolean parameter. It ablates the hidden fc1 layer.
        pts = self.answer("intervene_points", {"match": "fc1"}).get("points", [])
        if not pts:
            pts = self.answer("intervene_points", {}).get("points", [])
        if not pts:
            raise Fail(f"step {self.cur}: intervene_points lists no point")
        self.toggle_point = next((p["point"] for p in pts if p.get("module") == "fc1"),
                                 pts[0]["point"])
        self.res["toggle_point"] = self.toggle_point
        self.edit("control_create", {"id": "ablate", "kind": "toggle", "title": "Ablate fc1",
                                     "bind": "intervene:" + self.toggle_point, "options": {"value": False}})
        self.edit("control_create", {"id": "lr", "kind": "slider", "title": "Learning Rate", "bind": "param:lr",
                                     "options": {"min": 1e-3, "max": 10.0, "log": True, "value": 1.0}})

    def step_f_annotations(self):
        """The series panels loss and trace, a band, a threshold, two markers and the layout."""
        self.edit("panel_create", {"id": "loss", "primitive": "series", "title": "Loss",
                                   "bind": {"x": "loss_window.step", "y": "loss_window.value"},
                                   "options": {"x_label": "Step", "y_label": "Loss"}})
        self.edit("panel_create", {"id": "trace", "primitive": "series", "title": "Picked Unit",
                                   "bind": {"x": "trace.step", "y": "trace.value"},
                                   "options": {"x_label": "Step", "y_label": "Norm"}})
        # The bound text of the readout control as a text panel too, because panel_data reads a text panel's value.
        self.edit("panel_create", {"id": "readout_text", "primitive": "text", "title": "Readout",
                                   "bind": {"value": "focus"},
                                   "options": {"format": READOUT_FORMAT, "fields": {
                                       k: v for k, v in READOUT_FIELDS.items() if k != "value"}}})
        self.edit("annotation_set", {"panel": "loss", "annotation": {
            "id": "window", "kind": "band", "axis": "x", "range": {"control": "window"}, "text": "Window"}})
        self.edit("annotation_set", {"panel": "trace", "annotation": {
            "id": "level", "kind": "threshold", "axis": "y", "value": {"control": "level"},
            "text": "Level {level:.2f}", "fields": {"level": {"control": "level"}}}})
        self.edit("annotation_set", {"panel": "trace", "annotation": {
            "id": "current", "kind": "marker",
            "at": [{"selection": "world", "entity": "terrain", "field": "x"},
                   {"selection": "world", "entity": "terrain", "field": "z"}],
            "text": "Unit {unit:d} at {value:.3f}",
            "fields": {"unit": {"state": "unit"},
                       "value": {"selection": "world", "entity": "terrain", "field": "z"}}}})
        self.edit("annotation_set", {"panel": "world", "annotation": {
            "id": "picked", "kind": "marker", "at": {"selection": "world", "entity": "terrain"},
            "text": "Unit {unit:d}", "fields": {"unit": {"state": "unit"}}}})
        self.edit("layout_set", {"layout": {"root": "main", "nodes": [
            {"id": "main", "split": "vertical", "shares": [6, 3, 1, 1], "children": ["top", "middle", "row1", "row2"],
             "mins": {}},
            {"id": "top", "split": "horizontal", "shares": [3, 2], "children": ["world", "detail"],
             "mins": {"world": [48, 20]}},
            {"id": "middle", "split": "horizontal", "shares": [1, 1], "children": ["loss", "trace"]},
            {"id": "row1", "split": "horizontal", "children": ["window", "level", "lr", "ablate", "pause"]},
            {"id": "row2", "split": "horizontal", "children": ["measure", "palette", "transport", "now", "readout_text"]}]}})

    def step_g_terrain(self):
        """The first drawn terrain, the time to it, two renders at least 20 steps apart, the loss line, and the
        warning of the shared cadence in the catalog."""
        def terrain_rows():
            t = self.terrain()
            return t if t["rows"] > 0 else None
        first = self.until("the terrain has rows", terrain_rows, 120)
        early_png = self.render("world")
        t_render = time.monotonic() - self.mcp.first_call
        print(f"  seconds from the first tool call to the first render of a non-empty terrain: {t_render:.2f}")
        self.res["seconds_to_first_terrain_render"] = t_render
        l1 = self.cursor_now()["latest"]
        self.res["terrain_early"] = {"rows": first["rows"], "version": first["version"], "latest": l1}
        self.capture("terrain-early", early_png)
        # A row comes with a kept step of act, at a pass of the shared cadence that the app reads, not at every
        # step. The test thus waits for 20 more steps and for a row that came after the first render.
        self.until(f"the latest step reaches {l1 + 20} and the terrain has a new row",
                   lambda: self.cursor_now()["latest"] >= l1 + 20 and self.terrain()["rows"] > first["rows"], 120)
        later_png = self.render("world")
        later = self.terrain()
        l2 = self.cursor_now()["latest"]
        self.res["terrain_later"] = {"rows": later["rows"], "version": later["version"], "latest": l2}
        if later["rows"] <= first["rows"]:
            raise Fail(f"step {self.cur}: terrain rows did not grow: {first['rows']} then {later['rows']} "
                       f"(latest {l1} then {l2})")
        self.capture("terrain-later", later_png)
        # The script takes a snapshot at each pass of the shared cadence, a few times a second, and the app keeps
        # one value of a view for each frame that it draws, about once a second in the hidden window. A row thus
        # comes at the slower of the two, so the test waits for 40 rows: the terrain then reads as a terrain.
        full = self.until("the terrain has 40 rows", lambda: (lambda t: t if t["rows"] >= 40 else None)(self.terrain()),
                          150)
        self.res["terrain_full"] = {"rows": full["rows"], "latest": self.cursor_now()["latest"]}
        # The loss is a line in the same scene, from the loss window: both of its slots hold values.
        loss = self.entity("world", "loss")
        slots = {s["slot"]: s for s in (loss or {}).get("slots", [])}
        if not loss or loss.get("error") or "x" not in slots or "z" not in slots:
            raise Fail(f"step {self.cur}: the scene has no loss line with x and z: {json.dumps(loss)[:600]}")
        n = slots["x"]["shape"][0]
        if n < 2 or slots["z"]["shape"][0] != n or not all(isinstance(v, (int, float)) for v in slots["z"]["values"]):
            raise Fail(f"step {self.cur}: the loss line holds {n} points and its z slot {slots['z'].get('shape')}")
        self.res["loss_line"] = {"points": n, "draw": loss.get("draw"), "z_min": slots["z"].get("min"),
                                 "z_max": slots["z"].get("max")}
        # Everything that viewers ask of the run keeps to 2 % of its training time, by the shared cadence of the
        # programs. The live catalog gives each view that the cadence thins a warning that says how often it
        # updates and why: every N steps for the budget, or passes of the views alone for the floor.
        def cadence_note():
            entries = {e["id"]: e for e in self.answer("catalog_list", {"limit": 1000}).get("entries", [])}
            notes = [w for w in entries.get(ACT_ID, {}).get("warnings", []) if w.startswith("Updates ")]
            return notes[0] if notes else None
        note = self.until(f"catalog_list gives {ACT_ID} the warning of the shared cadence", cadence_note, 60)
        self.res["cadence_note"] = note
        every = re.search(r"every (\d+) steps", note)
        budget = note.startswith("Updates every ") and "keep to 2 % of the training time" in note
        floor = note.startswith("Updates on passes of the views alone")
        if not every or int(every.group(1)) < 3 or not (budget or floor):
            raise Fail(f"step {self.cur}: the cadence warning of {ACT_ID} is {note!r}")
        print(f"  cadence warning of {ACT_ID}: {note}")
        self.capture("terrain-full")

    def step_h_window(self):
        """A narrower window: the terrain follows, and its x values stay inside the window."""
        c = self.cursor_now()
        before = self.terrain()["rows"]
        hi = c["latest"] - 20
        lo = c["first"] + (c["latest"] - c["first"]) * 2 // 5
        self.res["window_set"] = [lo, hi]
        held = self.answer("control_set", {"id": "window", "value": [lo, hi]}).get("value")
        if list(held or []) != [lo, hi]:
            raise Fail(f"step {self.cur}: control_set of window answered {held}, not {[lo, hi]}")

        def inside():
            t = self.terrain()
            return t if t["x"] and min(t["x"]) >= lo and max(t["x"]) <= hi else None
        t = self.until(f"terrain x values inside [{lo}, {hi}]", inside, 60)
        if t["rows"] >= before:
            raise Fail(f"step {self.cur}: the narrow window left {t['rows']} rows, the wide one had {before}")
        self.res["window_terrain"] = {"rows": t["rows"], "before_rows": before, "x_min": min(t["x"]),
                                      "x_max": max(t["x"])}
        self.capture("window-narrow")

    def detail(self):
        """The heatmap detail: its version, its shape and its values as base64 text."""
        d = self.pdata("detail")
        data = d.get("data") or {}
        vals = (data.get("values") or {}).get("base64")
        if d.get("error") or not vals:
            raise Fail(f"step {self.cur}: panel_data of detail has no values: {json.dumps(d)[:800]}")
        return {"version": d.get("version"), "values": vals, "rows": data.get("source_rows"),
                "cols": data.get("source_cols")}

    def step_i_pick(self):
        """A pick on the terrain writes the state unit, and the heatmap detail follows it through a reference."""
        before = self.detail()
        m = self.mark()
        rows = self.terrain()["rows"]
        index = (rows // 2) * 64 + 20   # the middle row of the window, unit 20
        r = self.answer("selection_set", {"panel": "world", "entity": "terrain", "index": index})
        self.res["pick_fields"] = r.get("fields")
        pick = self.event("pick on world", {"kind": "pick", "panel": "world"}, m)
        fields = pick.get("fields") or {}
        if fields.get("column") != 20 or fields.get("entity") != "terrain":
            raise Fail(f"step {self.cur}: the pick event has the fields {json.dumps(fields)[:600]}, not column 20")
        state = self.event("state unit", {"kind": "state", "state": "unit"}, m)
        if state.get("value") != 20:
            raise Fail(f"step {self.cur}: the unit event holds {state.get('value')}, not 20")
        self.res["pick_event"], self.res["unit_event"] = pick, state
        unit = {s["id"]: s for s in self.answer("state_get", {}).get("states", [])}.get("unit", {})
        if unit.get("value") != 20 or "options.select" not in " ".join(map(str, unit.get("writers", []))):
            raise Fail(f"step {self.cur}: state_get gives the unit {json.dumps(unit)[:400]}")
        after = self.until("the detail panel changes", lambda: (lambda d: d if d["version"] != before["version"]
                                                                else None)(self.detail()), 30)
        if after["values"] == before["values"]:
            raise Fail(f"step {self.cur}: the picked unit left the values of detail unchanged")
        if (after["rows"], after["cols"]) != (16, 2):
            raise Fail(f"step {self.cur}: detail is {after['rows']} by {after['cols']}, not 16 by 2")
        self.res["detail_after_pick"] = {"before": before["version"], "after": after["version"],
                                         "shape": [after["rows"], after["cols"]]}
        self.capture("pick")
        self.save("pick-trace", self.render("trace"))
        self.save("pick-detail", self.render("detail"))

    def step_j_cursor(self):
        """The transport holds the cursor at an earlier kept step of act: each panel shows its value at that step.
        One step back, each panel shows its newest value at or before the new step."""
        live = {"detail": self.detail(), "world": self.terrain(), "trace": self.pdata("trace"),
                "loss": self.pdata("loss")}
        c = self.cursor_now()
        # The sources keep different steps: act keeps the step stamp of each snapshot that the app reads, at the
        # passes of the shared cadence, and loss keeps the run's step when a chunk of its series arrives, because a
        # series value has no stamp. The cursor goes to the kept step of any source nearest the step asked for, and
        # each panel shows the newest value of its source at or before the cursor. Detail and the terrain thus show
        # the held step itself only at a step that act keeps, so the test holds the cursor at the newest kept step
        # of act at least 100 steps back: a row of the terrain, whose x values are those steps while its grid keeps
        # all 64 columns (an unpooled grid).
        t = live["world"]
        if t["cols"] != 64:
            raise Fail(f"step {self.cur}: the terrain has {t['cols']} columns, so it pools its rows and its x values "
                       f"are not kept steps")
        rows = sorted(int(x) for x in t["x"] if x == int(x) and x <= c["latest"] - 100)
        if not rows:
            raise Fail(f"step {self.cur}: the terrain has no row at or before step {c['latest'] - 100}: x "
                       f"{t['x'][:3]} to {t['x'][-3:]}")
        target = rows[-1]
        self.res["cursor_kept_gap"] = self.kept_gap()
        held = self.answer("cursor_set", {"step": target, "live": False})["cursor"]
        step = held.get("step")
        if held.get("live") or step != target:
            raise Fail(f"step {self.cur}: cursor_set of the kept step {target} gave {json.dumps(held)}, not that step "
                       f"held")

        def at_step():
            d, w = self.detail(), self.terrain()
            return (d, w) if d["version"] != live["detail"]["version"] and w["version"] != live["world"]["version"] \
                else None
        d, w = self.until("detail and world change at the held step", at_step, 30)
        # A scene's terrain is a window, so its data states the window that ends at the held step; the heatmap
        # detail is one value and states the step of that value. Both agree on the held step, and the terrain's
        # newest row is that step.
        wc = self.pdata("world")["data"].get("cursor") or {}
        dc = (self.pdata("detail").get("data") or {}).get("cursor") or {}
        if (wc.get("window") or [None, None])[1] != step or dc.get("step") != step:
            raise Fail(f"step {self.cur}: at step {step} world states {wc} and detail states {dc}")
        if not w["x"] or max(w["x"]) != step:
            raise Fail(f"step {self.cur}: the terrain at step {step} holds x values up to "
                       f"{max(w['x']) if w['x'] else None}, not up to the held step")
        for panel in ("trace", "loss"):
            xs = self.series_x(panel)
            if xs and max(xs) > step:
                raise Fail(f"step {self.cur}: {panel} at step {step} holds steps up to {max(xs)}")
        steady = self.detail()
        if steady["values"] != d["values"]:
            raise Fail(f"step {self.cur}: the detail changed while the cursor was held")
        # One step back: the cursor goes to the previous step that any source keeps, which act need not keep. Both
        # panels change again, each shows the newest value of act at or before that step, and the two agree on it.
        stepped = self.answer("cursor_step", {"count": -1})["cursor"]
        if stepped.get("live") or stepped.get("step") is None or stepped["step"] >= step:
            raise Fail(f"step {self.cur}: cursor_step -1 gave {json.dumps(stepped)}, not a step before {step}")
        d2 = self.until("detail changes one step back", lambda: (lambda x: x if x["version"] != d["version"]
                                                                  else None)(self.detail()), 30)
        w2 = self.until("world changes one step back", lambda: (lambda x: x if x["version"] != w["version"]
                                                                 else None)(self.terrain()), 30)
        wc2 = self.pdata("world")["data"].get("cursor") or {}
        dc2 = (self.pdata("detail").get("data") or {}).get("cursor") or {}
        newest = max(w2["x"]) if w2["x"] else None
        if (wc2.get("window") or [None, None])[1] != stepped["step"] or newest is None \
                or dc2.get("step") != newest or newest > stepped["step"]:
            raise Fail(f"step {self.cur}: at step {stepped['step']} world states {wc2}, the terrain's newest row is "
                       f"{newest} and detail states {dc2}")
        self.res["cursor"] = {"held_step": step, "stepped_to": stepped["step"], "stepped_value_step": newest,
                              "versions": {"live": [live["detail"]["version"], live["world"]["version"]],
                                           "held": [d["version"], w["version"]],
                                           "stepped": [d2["version"], w2["version"]]}}
        self.res["bound_text_held"] = self.readout_text(stepped["step"])
        self.capture("cursor-held")

    @staticmethod
    def floats(v):
        """An array of the panel data: a list, or an object of base64 bytes (f32 or f64)."""
        if isinstance(v, dict) and v.get("base64") is not None:
            raw = base64.b64decode(v["base64"])
            size = 8 if v.get("dtype") == "f64" else 4
            return list(struct.unpack(f"<{len(raw) // size}{'d' if size == 8 else 'f'}", raw))
        return list(v) if isinstance(v, list) else []

    def series(self, panel):
        """The first line of a series panel: its step values and its values."""
        b = self.pdata(panel)
        (self.work / f"{panel}_data.json").write_text(json.dumps(b)[:100000])
        for line in (b.get("data") or {}).get("lines", []):
            return self.floats(line.get("x")), self.floats(line.get("y"))
        return [], []

    def series_x(self, panel):
        """The step values of a series panel."""
        return self.series(panel)[0]

    def readout_text(self, step):
        """The bound text of the readout. A text panel's data holds the bound value, not the formatted string,
        so the test checks the three inputs of the text and forms the string the format gives for them: the
        step of the cursor, the picked token and the picked token's value at that step."""
        b = self.pdata("readout_text")
        value = (b.get("data") or {}).get("value")
        if b.get("error") or not isinstance(value, (int, float)):
            raise Fail(f"step {self.cur}: the bound value of the readout is {value!r}: {json.dumps(b)[:500]}")
        xs, ys = self.series("trace")
        if not xs or xs[-1] > step or abs(ys[-1] - value) > 1e-4 * max(1.0, abs(value)):
            raise Fail(f"step {self.cur}: the bound value {value} differs from the last value of trace at step "
                       f"{step}: x {xs[-3:]} y {ys[-3:]}")
        unit = {s["id"]: s for s in self.answer("state_get", {}).get("states", [])}["unit"]["value"]
        cursor = self.cursor_now()
        if unit != 20 or cursor["step"] != step:
            raise Fail(f"step {self.cur}: unit {unit} and cursor {cursor} do not match step {step} and unit 20")
        return f"Step {step:,d}  Unit {unit:d}  Value {value:.3f}"

    def step_k_live(self):
        """Live again: the cursor follows the run, and the bound text follows the newest step."""
        c = self.answer("cursor_set", {"live": True})["cursor"]
        if not c.get("live"):
            raise Fail(f"step {self.cur}: cursor_set live gave {json.dumps(c)}")
        self.until("the detail follows the run again", lambda: (self.pdata("detail").get("data") or {}).get("cursor", {})
                   .get("step") is None, 30)
        # The window goes back to the whole run, so that the last point of trace is the newest one. The bound
        # value of the picked token is the newest one again too.

        def newest():
            # The two panels read the run at different frames, so the bound value may be one point ahead.
            # control.set clamps the end to the newest step, where it then follows the run. An end of 2048
            # stood on the newest step only while the run was below step 2048, which made the step depend
            # on the speed of the run.
            self.answer("control_set", {"id": "window", "value": [0, 2**53 - 1]})
            # The newest point of trace is the newest kept step of act, which stands behind the newest step that
            # the app received by up to the gap of the cadence.
            behind = max(200, 2 * self.kept_gap())
            xs, ys = self.series("trace")
            value = (self.pdata("readout_text").get("data") or {}).get("value")
            self.res["last_try_of_live_text"] = {"x": xs[-2:], "y": ys[-4:], "value": value, "cursor": self.cursor_now()}
            if xs and xs[-1] >= self.cursor_now()["latest"] - behind and isinstance(value, (int, float)) \
                    and any(abs(y - value) <= 1e-4 * max(1.0, abs(value)) for y in ys[-4:]):
                return {"step": xs[-1], "value": value}
            return None
        self.res["bound_text_live"] = self.until("the bound value is the newest one", newest, 30)
        self.res["live_cursor"] = c

    def step_l_modes(self):
        """The segmented groups switch the measure and the palette: the heatmap data and the terrain's colormap
        follow. The cursor is held for the colormap, so that only the colormap changes the render."""
        d0 = self.detail()
        m = self.mark()
        self.answer("control_set", {"id": "measure", "value": "zscore"})
        self.event("state measure", {"kind": "state", "state": "measure", "value": "zscore"}, m)
        d1 = self.until("detail follows the measure", lambda: (lambda d: d if d["version"] != d0["version"] else None)(
            self.detail()), 30)
        if d1["values"] == d0["values"]:
            raise Fail(f"step {self.cur}: the measure zscore left the values of detail unchanged")
        self.res["detail_version_measure"] = [d0["version"], d1["version"]]
        c = self.cursor_now()
        self.answer("cursor_set", {"step": c["latest"] - 100, "live": False})
        self.until("world holds the step", lambda: (self.pdata("world")["data"].get("cursor") or {}).get("window"), 30)
        viridis = self.render("world")
        v0 = self.pdata("world").get("version")
        m = self.mark()
        self.answer("control_set", {"id": "palette", "value": "magma"})
        self.event("state palette", {"kind": "state", "state": "palette", "value": "magma"}, m)
        self.until("world follows the palette", lambda: self.pdata("world").get("version") != v0, 30)
        magma = self.render("world")
        changed = pixel_difference(viridis, magma)
        self.res["palette"] = {"world_versions": [v0, self.pdata("world").get("version")], "pixels_changed": changed}
        if changed < 500:
            raise Fail(f"step {self.cur}: the palette magma changed only {changed} pixels of the terrain render")
        self.save("palette-viridis", viridis)
        self.save("palette-magma", magma)
        self.answer("cursor_set", {"live": True})

    def step_m_annotations(self):
        """The threshold line, the current-item marker, the band and the layout shares exist in the spec, and the
        slider level moves the threshold in the render of its series panel."""
        spec = self.answer("spec_get", {}).get("json", {})
        (self.work / "spec.json").write_text(json.dumps(spec, indent=1))
        notes = {(p.get("id"), a.get("id")): a for p in spec.get("panel", []) for a in p.get("annotation", [])}
        for key, kind in ((("trace", "level"), "threshold"), (("trace", "current"), "marker"),
                          (("loss", "window"), "band"), (("world", "picked"), "marker")):
            if notes.get(key, {}).get("kind") != kind:
                raise Fail(f"step {self.cur}: the spec has no {kind} {key}: {json.dumps(sorted(notes))[:400]}")
        splits, mins = [], []

        def walk(node, name):
            if isinstance(node, dict) and "children" in node:
                if node.get("shares"):
                    splits.append((name, node["shares"]))
                for i, child in enumerate(node["children"]):
                    walk(child, f"{name}.{i}")
            elif isinstance(node, dict) and node.get("min"):
                mins.append((node.get("panel"), node["min"]))
        walk(spec.get("layout") or {}, "root")
        if len(splits) < 3 or ("world", [48, 20]) not in [(p, list(m)) for p, m in mins]:
            raise Fail(f"step {self.cur}: the layout holds splits with shares {splits} and minimums {mins}: "
                       f"{json.dumps(spec.get('layout'))[:500]}")
        self.res["annotations"] = sorted(f"{p}/{i}:{a['kind']}" for (p, i), a in notes.items())
        self.res["layout_shares"] = dict(splits)
        self.res["layout_minimums"] = dict(mins)
        # The panel data exposes no label rectangles, so the test checks that the annotations are present and
        # that they draw (the render changes with the threshold), not that the placed labels avoid each other.
        self.res["label_overlap_check"] = "not exposed by panel.data or panel.render: presence only"
        # Hold the cursor while checking the threshold render so only the threshold line moves, and pick levels
        # inside trace's plotted y-range (MNIST fc1 batch L2 norms sit around 20..60, above the initial 1.0).
        c = self.cursor_now()
        self.answer("cursor_set", {"step": c["latest"], "live": False})
        _, ys = self.series("trace")
        if ys:
            y_lo, y_hi = min(ys), max(ys)
            span = max(y_hi - y_lo, 1.0)
            candidates = (
                round(min(200.0, max(0.0, y_lo + 0.35 * span)), 3),
                round(min(200.0, max(0.0, y_lo + 0.70 * span)), 3),
            )
        else:
            z = (self.res.get("pick_fields") or {}).get("z") or 1.0
            candidates = (round(min(200.0, z * 0.8), 3), round(min(200.0, z * 1.2 + 0.5), 3))
        before = self.render("trace")
        for level in candidates:
            self.answer("control_set", {"id": "level", "value": level})
            after = self.render("trace")
            if pixel_difference(before, after) >= 50:
                break
        else:
            self.answer("cursor_set", {"live": True})
            raise Fail(f"step {self.cur}: the threshold did not move in the render of trace (ys={ys[-5:] if ys else []}, candidates={candidates})")
        self.res["level_set"] = level
        self.save("level-trace", after)
        self.save("loss-band", self.render("loss"))
        self.capture("annotations")
        self.answer("cursor_set", {"live": True})

    def step_n_run_controls(self):
        """A person's drag of the lr slider and two clicks of the toggle reach the run."""
        before = {p["name"]: p.get("value") for p in self.answer("run_state", {"run": self.run}).get("params", [])}
        self.res["lr_before"] = before.get("lr")
        (self.work / "lr_drag").write_text("")

        def took_lr():
            for p in self.answer("run_state", {"run": self.run}).get("params", []):
                if p["name"] == "lr" and p.get("value") is not None and before.get("lr") is not None \
                        and abs(p["value"] - before["lr"]) > 1e-9 * max(1.0, abs(before["lr"])):
                    return p
            return None
        # Waits for the run to report a changed lr; 90 s is only a deadline, the loop returns as soon as it holds.
        self.res["lr_after"] = self.until("the run took a new lr from the slider", took_lr, 90)
        self.capture("lr-dragged")
        m = self.mark()
        (self.work / "toggle_on").write_text("")
        # Waits for the run's "intervene ablate" event; 90 s is only a deadline, it returns when the event comes.
        on = self.event("the ablation applied", {"kind": "run", "change": "intervene", "point": self.toggle_point,
                                                 "op": "ablate"}, m, 90)
        self.res["toggle_on_event"] = on
        self.capture("toggle-on")
        m = self.mark()
        (self.work / "toggle_off").write_text("")
        # Waits for the run's event that clears the ablation; 90 s is only a deadline, it returns when the event comes.
        self.res["toggle_off_event"] = self.event("the ablation cleared", {"kind": "run", "change": "intervene",
                                                                         "point": self.toggle_point, "op": None},
                                                  m, 90)
        # The clear event shows that the second click reached the run, so the UI script may end.
        (self.work / "script_done").write_text("")

    def step_o_operator(self):
        """An operator source from live:head.act drawn on world, then removed."""
        self.edit("source_add", {"id": "head_surface", "catalog": HEAD_ID,
                                 "transform": [{"op": "surface", "max_cells": 4096}]})
        self.edit("entity_set", {"panel": "world", "entity": {"path": "head", "draw": "surface",
                                                              "bind": "head_surface"}})

        def listed():
            for p in self.answer("program_status", {}).get("programs", []):
                if p.get("id") == "operator.head_surface.0" and p.get("kind") == "operator" \
                        and p.get("host") == "script" and p.get("state") == "running" and (p.get("runs") or 0) >= 1:
                    return p
            return None
        # Waits for the operator program to run in the script; 90 s is only a deadline, it returns when it runs.
        try:
            self.res["operator_status"] = self.until("operator.head_surface.0 runs in the script", listed, 90)
        except Fail as e:
            progs = self.answer("program_status", {}).get("programs", [])
            raise Fail(f"{e}; programs: {json.dumps(progs)[:800]}")
        # The script's pooled grid feeds the source: the entity head resolves to a surface.
        def fed():
            for ent in (self.pdata("world").get("data") or {}).get("entities", []):
                if ent.get("path") == "head" and ent.get("slots") and not ent.get("error"):
                    return ent
            return None
        head = self.until("the entity head draws the operator's output", fed, 60)
        self.res["operator_entity"] = {s["slot"]: s.get("shape") for s in head["slots"]}
        self.save("operator-head", self.render("world"))
        self.edit("entity_remove", {"panel": "world", "path": "head"})
        self.edit("source_remove", {"id": "head_surface"})

    def step_p_stop(self):
        """run_stop, the exit of the run, no live pid of the script, then app and mcp stop by their pids."""
        self.answer("run_stop", {"run": self.run})
        # Waits for the run to report "exited"; 90 s is only a deadline, it returns as soon as the state holds.
        self.until("the run is exited", lambda: self.runs().get(self.run, {}).get("state") == "exited", 90)
        self.until("no pid of the script is alive", lambda: not [p for p in sorted(self.pids) if alive(p)], 60)
        self.res["train_pids_gone"] = sorted(self.pids)
        for name in ("mcp", "app"):
            p = self.procs[name]
            if p.stdin:
                try:
                    p.stdin.close()
                except OSError:
                    pass
            stop(p)
            if alive(p.pid):
                raise Fail(f"step {self.cur}: {name} pid {p.pid} is still alive")
        self.res["sent_edits"] = dict(self.sent)

    def step_q_script_lines(self):
        """The canonical MNIST script calls caliper.watch and contains no instrument/spec builder code."""
        text = (ROOT / "examples" / "mnist" / "train.py").read_text()
        lines = [l.strip() for l in text.splitlines() if re.search("caliper", l, re.I)]
        self.res["caliper_lines"] = lines
        if not any("caliper.watch(" in l for l in lines):
            raise Fail(f"step {self.cur}: examples/mnist/train.py does not call caliper.watch")
        forbidden = [tok for tok in ("panel_create", "control_create", "spec_new", "layout_set", "caliper.scene")
                     if tok in text]
        if forbidden:
            raise Fail(f"step {self.cur}: examples/mnist/train.py contains instrument code: {forbidden}")

    def step_r_unchanged(self):
        """The SHA-256 of the app, caliper-mcp and the package files is the same as before the run."""
        now = self.digest()
        changed = sorted(p for p in now if now[p] != self.hashes.get(p))
        if changed or set(now) != set(self.hashes):
            raise Fail(f"step {self.cur}: these files changed during the run: {changed}")
        self.res["unchanged_files"] = len(now)

    STEPS = [("a", "start"), ("b", "run"), ("c", "state"), ("d", "scene"), ("e", "controls"),
             ("f", "setup_notes"), ("g", "terrain"), ("h", "window"), ("i", "pick"), ("j", "cursor"),
             ("k", "live"), ("l", "modes"), ("m", "annotations"), ("n", "run_controls"), ("o", "operator"),
             ("p", "stop"), ("q", "script_lines"), ("r", "unchanged")]
    METHODS = {"a": step_a_start, "b": step_b_run, "c": step_c_state, "d": step_d_scene, "e": step_e_controls,
               "f": step_f_annotations, "g": step_g_terrain, "h": step_h_window, "i": step_i_pick,
               "j": step_j_cursor, "k": step_k_live, "l": step_l_modes, "m": step_m_annotations,
               "n": step_n_run_controls, "o": step_o_operator, "p": step_p_stop, "q": step_q_script_lines,
               "r": step_r_unchanged}

    def go(self, stop_after):
        for letter, name in self.STEPS:
            self.cur = f"{letter}_{name}"
            t0 = time.monotonic()
            print(f"step {self.cur}")
            self.METHODS[letter](self)
            self.res["steps"][self.cur + "_seconds"] = round(time.monotonic() - t0, 2)
            if stop_after in (letter, name, self.cur):
                print(f"stopped after {self.cur}")
                return


def run(a):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    # The alarm fires at least 60 s before the ctest TIMEOUT of 600 s (tests/CMakeLists.txt), so the cleanup
    # in the finally block below (stop the app, caliper-mcp and the script) ends before ctest kills the test.
    signal.alarm(a.limit_s)
    work = Path(a.work).resolve()
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).exists():
            raise Skip(f"{exe or 'a program'} is not built")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_INSTRUMENT_PATH")
           and not k.startswith("CALIPER_RUN_")}
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    pre = subprocess.run([a.python, "-c", "import numpy, torch, torchvision, caliper, caliper._netio, caliper._producer"],
                         env=env, capture_output=True, text=True)
    if pre.returncode != 0:
        why = pre.stderr.strip().splitlines()[-1] if pre.stderr.strip() else "an import failed"
        raise Skip(f"torch, torchvision or the native network sender is missing: {why}")
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
    (ws / "train.py").write_text((ROOT / "examples" / "mnist" / "train.py").read_text())
    mnist_data = Path(
        os.environ.get("CALIPER_MNIST_DATA_DIR", str(Path.home() / ".cache" / "caliper" / "mnist"))
    ).expanduser()
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_INSTRUMENTS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1600x1000",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python,
                "CALIPER_MNIST_DATA_DIR": str(mnist_data)})
    decisions = work / "approvals.txt"
    decisions.write_text("\n".join(APPROVALS) + "\n")
    capture_dir = Path(a.capture_dir) if a.capture_dir else work / "captures"
    procs = {}
    test = None
    err = None
    try:
        alog = open(work / "app.log", "w")
        # A person's input on the toggle and the lr slider, through the app's own input queue, each
        # when the test writes its file (src/host/builder/spec_ui_driver.h).
        ui = (f"patience 900; wait-file {work}/lr_drag; drag panel:lr 0.62 0.5 to panel:lr 0.25 0.5; "
              f"wait-file {work}/toggle_on; click panel:ablate; wait-file {work}/toggle_off; click panel:ablate; "
              f"wait-file {work}/script_done")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"],
                               env=dict(env, CALIPER_TEST_APPROVALS=str(decisions), CALIPER_TEST_SPEC_UI=ui),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs["app"] = app
        lines = []

        def drain():
            for raw in app.stdout:
                alog.write(raw)
                alog.flush()
                lines.append(raw.rstrip("\n"))
        threading.Thread(target=drain, daemon=True).start()
        api_file = work / "data" / "api.json"
        # Waits for the app to write its API file; 90 s is only a deadline, the loop ends as soon as the file exists.
        end = time.monotonic() + 90
        while not api_file.exists():
            for line in list(lines):
                m = re.match(r"\[renderer\] (\S+)", line)
                if m and m.group(1) != "metal":
                    raise Skip(f"the renderer is {m.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs["mcp"] = mcp.p
        mcp.request("server/discover", {})
        test = Test(mcp, ws, work, capture_dir, a)
        test.procs = procs
        test.go(a.stop_after)
        return test.res
    except (Fail, Skip) as e:
        err = e
        if test is not None:
            test.res["failed"] = str(e)
        raise
    finally:
        for p in reversed(list(procs.values())):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
            stop(p)
        if test is not None:
            left = [pid for pid in sorted(test.pids | {p.pid for p in procs.values()}) if alive(pid)]
            for pid in left:
                print(f"a process is left after the test: {pid}", file=sys.stderr)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            test.res["left_after_cleanup"] = left
            (work / "result.json").write_text(json.dumps(test.res, indent=2, default=str))
            if left and err is None:
                raise Fail(f"processes are left: {left}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--mcp", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work", required=True)
    ap.add_argument("--capture-dir")
    ap.add_argument("--stop-after", help="a step letter or name: a, b, ... or start, run, ...")
    ap.add_argument("--limit-s", type=int, default=540)
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_instrument_building_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"instrument building acceptance FAILED: {e}")
        print(f"result so far: {Path(a.work) / 'result.json'}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    print("caliper_instrument_building_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
