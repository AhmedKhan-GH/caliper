#!/usr/bin/env python3
"""The acceptance test of programmable graphics, through caliper-mcp only.

It starts the hidden app with no script and caliper-mcp as a child over stdio, then drives an agent session
through caliper-mcp alone on a managed run of the canonical MNIST script (examples/mnist/train.py). The agent
builds an MNIST probe dashboard from the workspace pieces under tests/programmable_graphics/pieces: a 3-D
per-digit accuracy ring (digit_ring@1) in one scene panel and a 10x10 row-normalized confusion matrix
(confusion_grid@1) in another, with no built-in primitive in either frame, a multi-line chart of per-class
probe log-probabilities with a legend, a two-tab loss/accuracy strip, and a toggle bound to an intervention
point. The driver checks foreign-spec piece review, piece_check and piece_update, cursor scrubbing between an
early kept step and trained step, selection_set picking with linked state and panel update, intervention
shifting accuracy/loss, run_fork divergence under ablation, an over-time draw warning while training continues,
overhead_share within target, unchanged file hashes, and zero leftover child processes.

The driver waits on events and on published state, never for a fixed time, and runs without a window and without the
network. Exit code 0 passes, 77 skips (no app, no torch, no Metal, no native sender, no producer module), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir> [--capture-dir <dir>]
                 [--limit-s 900]
"""
import argparse
import base64
import importlib.util
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77

# The client, the process helpers and the waits of the GPU programs test.
_spec = importlib.util.spec_from_file_location("gpu_programs_driver", ROOT / "tests" / "gpu_programs" / "driver.py")
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)
Fail, Skip = gp.Fail, gp.Skip

SCRIPT = "train.py"
PIECES_DIR = HERE / "pieces"
# Four per-digit probe log-probability series for the multi-line legend chart.
SCORE_LINES = [f"digit_{d}" for d in range(4)]
# Training thresholds on canonical MNIST (seed=1, Adadelta lr=0.005 so early kept steps
# stay near chance while reaching > 0.75 within ~350 steps):
TRAIN_ACC_TARGET = 0.75
TRAIN_LOSS_TARGET = 0.90
TRAIN_CAP = 800
# Ablating fc1 zeroes the hidden layer before fc2, dropping probe accuracy to ~0.10 (chance)
# and raising training NLL loss toward ln(10) ~= 2.30.
ABLATED_ACC_MAX = 0.30
FORK_STEPS = 40
FORK_CAP_STEPS = 200
STALL_S = 120
TRAIN_WAIT_S = 600
LOG = re.compile(r"step (\d+) loss ([0-9.eE+-]+)")
# The commands that wait on a card of the person; the driver approves each as the person would.
CARDS = {"spec_new", "spec_open", "run_start", "run_pause", "run_set_param", "run_fork", "run_stop", "piece_update",
         "state_create", "state_set", "panel_create", "panel_patch", "program_create", "source_add", "layout_place",
         "layout_set", "control_create", "cursor_set", "selection_set", "intervene_set", "intervene_clear",
         "program_remove", "panel_remove"}

FOREIGN = '''caliper_spec = 1

[applet]
id = "shared"
title = "Shared Instrument"

[[piece]]
name = "shared_copy"
version = 1
description = "Copies the first values of a matrix"

[[piece.inputs]]
name = "w"
kind = "tensor"
rank_min = 2
rank_max = 2

[[piece.resources]]
id = "vals"
kind = "buffer"
element = "f32"
count = 64

[piece.derive]
passes = [{ entry = "fill", writes = ["vals"] }]

[piece.files]
derive = """
@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 64u) {
        vals[id.x] = w[id.x];
    }
}
"""

[[program]]
id = "shared_vals"
piece = { name = "shared_copy", version = 1, from = "inline", hash = "@HASH@" }
bind = { w = "live:fc1.weight" }

[provenance]
foreign = true
from = "a spec shared by a colleague"
reviewed = []
'''


def piece_args(folder):
    """The piece of a folder (piece.toml and its WGSL files) as the argument of piece.check and piece.update."""
    p = tomllib.loads((folder / "piece.toml").read_text())
    p["files"] = {k: (folder / f"{k}.wgsl").read_text() for k in ("common", "derive", "draw")
                  if (folder / f"{k}.wgsl").exists()}
    return p


def f64s(obj):
    raw = base64.b64decode(obj["base64"])
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


def mnist_data_dir() -> Path:
    return Path(
        os.environ.get("CALIPER_MNIST_DATA_DIR", str(Path.home() / ".cache" / "caliper" / "mnist"))
    ).expanduser()


class Run:
    """The state of the test run: the connection, the approvals, the captures and the steps done."""

    def __init__(self, c, approvals, transcript):
        self.c = c
        self.m = c.mcp
        self.approvals = approvals
        self.transcript = transcript
        self.n = 0
        self.result = {}

    def call(self, tool, args=None, **kw):
        """One tool call as an agent sends it; a card is approved first, as the person would."""
        args = args or {}
        if tool in CARDS:
            with open(self.approvals, "a") as f:
                f.write(f"approve {tool.replace('_', '.', 1)} mcp\n")
        r = self.m.call(tool, args)
        self.transcript.write(json.dumps({"tool": tool, "args": args, "error": bool(r.get("isError"))}) + "\n")
        if r.get("isError") and not kw.get("allow_error"):
            raise Fail(f"{tool} answered an error: {json.dumps(r)[:700]}")
        return r.get("structuredContent", {}) if not r.get("isError") else r

    def step(self, name):
        self.n += 1
        print(f"step {self.n}: {name}", flush=True)
        self.result.setdefault("steps", []).append(name)

    # ---- reads
    def read(self, resource, count=10, step=None, first=None, prog="digit_bars"):
        a = {"id": prog, "resource": resource, "count": count}
        if step is not None:
            a["step"] = step
        if first is not None:
            a["first"] = first
        return self.call("program_read", a)

    def metrics(self, step=None):
        """Read the 10 DigitMetric rows [digit, total, correct, errors, has_errors, accuracy, mean_conf]."""
        b = self.read("metrics", 10, step)
        els = b.get("elements") or []
        accs = [float(e[5]) for e in els]
        return b, els, accs

    def chart(self):
        """The lines of class_scores in the order of SCORE_LINES, each as a list of (step, value)."""
        lines = [list(zip(f64s(d["x"]), f64s(d["y"])))
                 for d in self.call("panel_data", {"panel": "class_scores"})["data"]["lines"]]
        if len(lines) != len(SCORE_LINES):
            raise Fail(f"the chart class_scores has {len(lines)} lines, not {len(SCORE_LINES)}")
        return lines

    def series_lines(self, panel):
        d = self.call("panel_data", {"panel": panel})["data"]["lines"][0]
        return list(zip(f64s(d["x"]), f64s(d["y"])))

    def logs(self, run):
        out = []
        for line in self.call("run_logs", {"run": run, "lines": 200}).get("lines", []):
            m = LOG.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
        return out

    def run_step(self, run):
        return self.call("run_state", {"run": run}).get("step") or 0

    def status(self, pid):
        return self.c.status(pid)

    def step_n(self, run, count):
        prev = self.run_step(run)
        target = prev + count
        self.call("run_pause", {"op": "step", "count": count, "run": run})
        self.c.until(
            f"step {count} past {prev}",
            lambda t=target: self.run_step(run) >= t and (self.c.runs().get(run) or {}).get("state") == "paused",
            90,
        )
        return self.run_step(run)

    def sync_paused(self, run):
        """Step a paused run so a pending intervention applies to probe.logits and digit_bars runs on it."""
        self.step_n(run, 1)
        ready = self.run_step(run)
        for _ in range(4):
            st = self.status("digit_bars") or {}
            every = max(int(st.get("every") or 1), 1)
            burst = min(max(every + 4, 4), 100)
            self.step_n(run, burst)
            st2 = self.status("digit_bars") or {}
            if int(st2.get("last_step") or 0) >= ready:
                self.c.pace(0.5)
                return self.run_step(run)
        raise Fail(f"digit_bars did not run past step {ready}: {self.status('digit_bars')}")


def steps(r):
    c = r.c
    cap = c.capture

    # 1. The files that the test must leave unchanged.
    r.step("hash the app, caliper-mcp and the package")
    before = {str(f): gp.sha256(f) for f in [Path(c.app), Path(c.mcp_path)] + gp.package_files()}
    (c.work / "before.json").write_text(json.dumps(before, indent=1))

    # 2. A foreign spec's piece waits in review: no card exists yet that reviews it.
    r.step("a piece of a foreign spec waits in review")
    shared = c.ws / "shared"
    shared.mkdir()
    foreign_hash = r.call("piece_check", {"piece": tomllib.loads(FOREIGN)["piece"][0], "headers": False})["hash"]
    (shared / "foreign.spec.toml").write_text(FOREIGN.replace("@HASH@", foreign_hash))
    r.call("spec_open", {"path": "shared/foreign.spec.toml"})
    st = c.status("shared_vals")
    if not st or st.get("state") != "review":
        raise Fail(f"the program of a foreign spec's piece is not in review: {st}")
    r.result["foreign"] = {k: st.get(k) for k in ("state", "kind", "host")}

    # 3. Start the canonical MNIST training script as a managed run and pause while building.
    r.step("start the MNIST run and pause it")
    r.call("spec_new", {"id": "mnist_dashboard", "title": "MNIST Probe Dashboard"})
    r.call("run_start", {
        "script": SCRIPT,
        "args": ["--log-every", "5", "--probe-every", "1", "--probe-size", "512",
                 "--probe-series", "--seed", "1", "--lr", "0.005", "--pace", "0.015"],
    })
    c.run = c.until("a run", lambda: next(iter(c.runs()), None), 60)
    c.until("the run is running and attached",
            lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(c.runs().get(c.run)), 120)
    c.pids.update(gp.pids_of(c.ws / SCRIPT))
    c.until("step 2 of the run (after step 1 probe and accuracy)", lambda: r.run_step(c.run) >= 2, 90)
    r.call("run_pause", {"op": "pause", "run": c.run})
    c.until("the run paused", lambda: (c.runs().get(c.run) or {}).get("state") == "paused", 60)
    r.call("catalog_list", {})

    # 4. The library.
    r.step("the library lists points@1, image_grid@1 and the four lib modules")
    lst = r.call("piece_list", {"from": "library"})
    names = {(p["name"], p["version"]) for p in lst["pieces"]}
    for need in (("points", 1), ("image_grid", 1)):
        if need not in names:
            raise Fail(f"piece_list lacks {need[0]}@{need[1]}: {sorted(names)}")
    want = {"colormap@1", "pick@1", "camera@1", "easing@1"}
    if not want <= set(lst["modules"]):
        raise Fail(f"piece_list lacks lib modules: {lst['modules']}")
    shown = r.call("piece_show", {"name": "points", "from": "library"})
    if not (shown["piece"].get("files") or {}).get("draw"):
        raise Fail("piece_show of points has no draw file")

    # 5. Check and write both MNIST workspace pieces as version 1.
    r.step("check and write both MNIST pieces as version 1")
    pieces = {n: piece_args(PIECES_DIR / n) for n in ("digit_ring", "confusion_grid")}
    for n, p in pieces.items():
        ck = r.call("piece_check", {"piece": p})
        if not ck["ok"]:
            raise Fail(f"piece_check of {n}: {ck['messages']}")
    for n, p in pieces.items():
        up = r.call("piece_update", {"piece": p})
        if up["version"] != 1:
            raise Fail(f"piece_update of {n} wrote version {up['version']}")
        r.result.setdefault("hashes", {})[n] = up["hash"]

    # 6. Build the MNIST dashboard: two scene panels with pieces alone, state, multi-line series,
    #    two-tab loss/accuracy strip, and a toggle bound to an intervention point.
    r.step("build the MNIST instrument from pieces")
    def catalog_ready():
        ids = {e["id"] for e in r.call("catalog_list", {"kind": "live", "limit": 1000}).get("entries", [])}
        need = {"live:loss", "live:accuracy"} | {f"live:probe.{line}" for line in SCORE_LINES}
        return need <= ids or None
    c.until("MNIST live sources in the catalog", catalog_ready, 60)

    r.call("state_create", {"id": "digit", "type": "integer", "default": 0, "min": 0, "max": 9, "title": "Digit"})
    for pid, title in (("digits", "Digit Accuracy Ring"), ("matrix", "Confusion Matrix")):
        r.call("panel_create", {"id": pid, "primitive": "scene", "title": title, "bind": {"scene": "live:model"},
                                "options": {"exclude": ["**"]}})
    binds = {"logits": "live:probe.logits", "targets": "live:probe.labels"}
    r.call("program_create", {
        "id": "digit_bars", "title": "Digit Accuracy Ring",
        "piece": {"name": "digit_ring", "version": 1, "from": "workspace"},
        "panel": "digits", "bind": binds, "params": {"scale": 1.0},
    })
    r.call("program_create", {
        "id": "confusion_view", "title": "Confusion Matrix",
        "piece": {"name": "confusion_grid", "version": 1, "from": "workspace"},
        "panel": "matrix", "bind": binds,
        "params": {"digit": {"state": "digit"}, "picked": {"selection": "digits", "field": "digit", "default": 0},
                   "gain": 1.0},
    })
    r.call("panel_patch", {"id": "digits", "patch": {"bind": {"scene": None}, "options": {
        "exclude": None, "axes": False, "up": "+y", "select": {"state": "digit", "field": "digit"}}}})
    r.call("panel_patch", {"id": "matrix", "patch": {"bind": {"scene": None}, "options": {"exclude": None}}})

    r.call("source_add", {"id": "loss", "catalog": "live:loss", "history": 4096})
    r.call("source_add", {"id": "loss_window", "from": "loss", "transform": [{"op": "window", "steps": 4096}]})
    r.call("source_add", {"id": "acc", "catalog": "live:accuracy", "history": 4096})
    r.call("source_add", {"id": "acc_window", "from": "acc", "transform": [{"op": "window", "steps": 4096}]})
    r.call("panel_create", {"id": "loss_curve", "primitive": "series", "title": "Loss",
                            "bind": {"x": "loss_window.step", "y": "loss_window.value"},
                            "options": {"x_label": "Step", "y_label": "Loss"}})
    r.call("panel_create", {"id": "acc_curve", "primitive": "series", "title": "Accuracy",
                            "bind": {"x": "acc_window.step", "y": "acc_window.value"},
                            "options": {"x_label": "Step", "y_label": "Accuracy"}})
    for line in SCORE_LINES:
        r.call("source_add", {"id": f"{line}_raw", "catalog": f"live:probe.{line}", "history": 4096})
        r.call("source_add", {"id": line, "from": f"{line}_raw", "transform": [{"op": "window", "steps": 4096}]})
    r.call("panel_create", {
        "id": "class_scores", "primitive": "series", "title": "Probe Accuracy by Digit",
        "bind": {"x": f"{SCORE_LINES[0]}.step", "y": [f"{line}.value" for line in SCORE_LINES]},
        "options": {"x_label": "Step", "y_label": "Accuracy", "legend": True},
    })
    r.call("control_create", {
        "id": "ablate", "kind": "toggle", "title": "Ablate fc1",
        "bind": "intervene:fc1", "options": {"value": False},
    })
    r.call("layout_set", {"layout": {"root": "page", "nodes": [
        {"id": "page", "split": "vertical", "children": ["main", "ablate"], "ratio": None},
        {"id": "main", "split": "horizontal", "children": ["left", "matrix"], "ratio": 0.5},
        {"id": "left", "split": "vertical", "children": ["digits", "charts"], "ratio": 0.45},
        {"id": "charts", "split": "vertical", "children": ["class_scores", "curves"], "ratio": 0.55},
        {"id": "curves", "split": "tabs", "children": ["loss_curve", "acc_curve"], "ratio": None},
    ]}})
    if "[[panel.entity]]" in r.call("spec_get", {}).get("text", ""):
        raise Fail("the spec holds an entity: a primitive of the scene is in the frame")
    cap("empty")

    # 7. Both halves of both programs compile; the run resumes.
    r.step("both halves of both programs compile")
    r.call("run_pause", {"op": "resume", "run": c.run})
    for prog in ("digit_bars", "confusion_view"):
        for half in ("derive", "draw"):
            try:
                c.event(f"{prog} {half} compiled",
                        {"kind": "program", "program": prog, "change": "compiled", "half": half}, 120)
            except Fail as e:
                seen = [ev for ev in c.events if ev.get("kind") == "program"]
                raise Fail(f"{e}; status {c.status(prog)}; program events {seen[-8:]}") from e

    # 8. Resources are published and the digit ring draws.
    r.step("the digit metrics are published and the ring draws")
    got = c.until("a metrics version", lambda: (lambda b: b if b.get("count") == 10 else None)(r.read("metrics", 10)), 90)
    for idx, e in enumerate(got["elements"]):
        if e[0] != idx or e[1] <= 0 or not (0.0 <= e[5] <= 1.0):
            raise Fail(f"unexpected digit metric row {idx}: {e}")
    c.until("digit_bars draw runs",
            lambda: (lambda d: d if d.get("state") == "running" and d.get("resource_step") is not None
                     else None)((c.status("digit_bars") or {}).get("draw") or {}), 60)
    cap("digits-early")
    n = c.differs("digits-early", "empty", 2000)
    r.result["ring_pixels"] = n
    if "[[panel.entity]]" in r.call("spec_get", {}).get("text", ""):
        raise Fail("the panel has an entity")

    # 9. Training improves probe accuracy and drops loss; cursor_set scrubs between early and trained steps.
    r.step("training improves accuracy and cursor_set scrubs between early and trained steps")
    def trained():
        b, els, accs = r.metrics()
        log = r.logs(c.run)
        if len(accs) == 10 and (sum(accs) / 10.0) >= TRAIN_ACC_TARGET and log and log[-1][1] <= TRAIN_LOSS_TARGET:
            return b, els, accs, log
        return None
    newest, els_new, accs_new, log = until_trained(
        c, r, c.run,
        f"mean probe accuracy >= {TRAIN_ACC_TARGET} and loss <= {TRAIN_LOSS_TARGET} by step {TRAIN_CAP}",
        trained, TRAIN_CAP,
    )
    first_step = newest["kept"]["first"]
    old, _, accs_old = r.metrics(first_step)
    mean_old = sum(accs_old) / len(accs_old)
    mean_new = sum(accs_new) / len(accs_new)
    if old["step"] != first_step or mean_new - mean_old < 0.25:
        raise Fail(f"accuracy did not improve from first kept step {old['step']} ({mean_old:.3f}) "
                   f"to step {newest['step']} ({mean_new:.3f})")
    r.call("cursor_set", {"step": first_step})
    cap("scrub-early")
    r.call("cursor_set", {"live": True})
    cap("scrub-live")
    n = c.differs("scrub-early", "scrub-live", 500)
    lines = r.chart()
    if any(len(line) < 2 for line in lines):
        raise Fail(f"class_scores lines lack history: {[len(l) for l in lines]}")
    r.result["training"] = {
        "first_kept": first_step, "newest": newest["step"],
        "mean_acc_early": round(mean_old, 4), "mean_acc_trained": round(mean_new, 4),
        "loss_first": log[0][1], "loss_last": log[-1][1],
        "scrub_pixels_changed": n, "chart_points": len(lines[0]),
    }

    # 10. Pick on the digit ring updates state and confusion grid.
    r.step("selection_set pick returns digit fields, writes state, and updates the confusion matrix")
    r.call("run_pause", {"op": "pause", "run": c.run})
    c.until("the run paused", lambda: (c.runs().get(c.run) or {}).get("state") == "paused", 60)
    _, cur_els, cur_accs = r.metrics()
    best = max(range(10), key=lambda i: cur_accs[i])
    r.call("state_set", {"id": "digit", "value": (best + 1) % 10})
    c.pace(0.5)
    cap("pick-before")
    picked_resp = r.call("selection_set", {"panel": "digits", "program": "digit_bars", "item": best})
    fields = picked_resp["fields"]
    ev = c.event("the pick event", {"kind": "pick", "program": "digit_bars"}, 30)
    want = cur_els[best]
    flds = ev.get("fields") or {}
    for key, val in (("digit", best), ("total", int(want[1])), ("correct", int(want[2])), ("errors", int(want[3]))):
        got_ev = flds.get(key)
        got_resp = fields.get("fields", fields).get(key)
        if got_ev != val or got_resp != val:
            raise Fail(f"the pick has {key}={got_ev}/{got_resp}, not {val}; event {ev}")
    state = {s["id"]: s["value"] for s in r.call("state_get", {})["states"]}
    if state.get("digit") != best:
        raise Fail(f"the state digit is {state.get('digit')}, not {best}")
    def matrix_changed():
        cap("pick-after")
        return _differs(c, "pick-before", "pick-after", 1000)
    c.until("the confusion matrix follows the pick", matrix_changed, 30)
    r.result["pick"] = {"item": best, "digit": best, "accuracy": round(cur_accs[best], 4)}
    cap("digits-parent")

    # 11. Intervention (ablating fc1) drops probe accuracy and raises loss, then clearing recovers.
    r.step("ablating fc1 shifts accuracy and loss, and clearing recovers")
    base_acc = sum(cur_accs) / 10.0
    r.call("intervene_set", {"point": "fc1", "op": "ablate"})
    r.sync_paused(c.run)
    _, _, ablated_accs = r.metrics()
    ablated_acc = sum(ablated_accs) / 10.0
    if ablated_acc > ABLATED_ACC_MAX or base_acc - ablated_acc < 0.40:
        raise Fail(f"ablating fc1 left mean accuracy at {ablated_acc:.3f} (baseline {base_acc:.3f})")
    r.call("intervene_clear", {"point": "fc1"})
    r.sync_paused(c.run)
    _, _, restored_accs = r.metrics()
    restored_acc = sum(restored_accs) / 10.0
    if restored_acc < base_acc - 0.15:
        raise Fail(f"clearing fc1 did not restore accuracy: {restored_acc:.3f} vs baseline {base_acc:.3f}")
    r.call("run_pause", {"op": "resume", "run": c.run})
    r.result["ablation"] = {
        "point": "fc1", "baseline_acc": round(base_acc, 4),
        "ablated_acc": round(ablated_acc, 4), "restored_acc": round(restored_acc, 4),
    }

    # 12. Fork from early kept step with fc1 ablated diverges from parent (never learns).
    r.step("a fork from an early step with fc1 ablated stays unlearned and diverges from parent")
    parent = c.run
    fork = r.call("run_fork", {"run": parent, "step": first_step,
                               "interventions": [{"point": "fc1", "op": "ablate"}]})["run"]
    child = fork["id"]
    c.run = child
    c.until("the child runs and is attached",
            lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(c.runs().get(child)), 120)
    c.pids.update(gp.pids_of(c.ws / SCRIPT))
    target = newest["step"] + FORK_STEPS
    def child_reached():
        b, _, hh = r.metrics()
        return (b, hh) if (b.get("step") or 0) >= target and len(hh) == 10 else None
    cb, ch = until_trained(
        c, r, child,
        f"the forked child reaches step {target} (cap {target + FORK_CAP_STEPS})",
        child_reached, target + FORK_CAP_STEPS,
    )
    child_acc = sum(ch) / 10.0
    if child_acc >= ABLATED_ACC_MAX:
        raise Fail(f"the ablated child reached mean accuracy {child_acc:.3f} at step {cb['step']}")
    cap("digits-child")
    n = c.differs("digits-parent", "digits-child", 300)
    r.result["fork"] = {
        "child": child, "step": cb["step"], "child_mean_acc": round(child_acc, 4),
        "ring_pixels_changed": n,
    }

    # 13. An over-time draw piece records a program.draw_over_time warning while training goes on.
    r.step("an over-time draw records a warning and keeps drawing while training goes on")
    slow = dict(pieces["digit_ring"])
    slow["name"] = "digit_ring_slow"
    slow["params"] = list(slow["params"]) + [{"name": "burn", "default": 25000.0, "min": 0.0, "max": 100000.0}]
    slow["derive"] = dict(slow["derive"], time_ms=50.0)
    slow["draw"] = dict(slow["draw"], passes=list(slow["draw"]["passes"]) + [
        {"kind": "render", "fragment": "burn_fs", "blend": "alpha", "depth": "none"}
    ])
    burn_code = """
@fragment
fn burn_fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    var acc = 0.0;
    for (var k = 0u; k < u32(params.burn); k = k + 1u) {
        acc = acc + sin(f32(k) * 0.001 + in.uv.x) * 0.0000001;
    }
    return vec4<f32>(0.0, 0.0, 0.0, clamp(abs(acc), 0.0, 0.01));
}
"""
    slow["files"] = dict(slow["files"], draw=slow["files"]["draw"] + burn_code)
    r.call("piece_update", {"piece": slow})
    r.call("panel_create", {"id": "slow", "primitive": "scene", "title": "Slow Ring", "bind": {"scene": "live:model"},
                            "options": {"exclude": ["**"]}})
    step_at = r.run_step(child)
    r.call("program_create", {
        "id": "slow_ring", "title": "Slow Ring",
        "piece": {"name": "digit_ring_slow", "version": 1, "from": "workspace"},
        "panel": "slow", "bind": binds, "params": {"scale": 1.0},
    })
    r.call("panel_patch", {"id": "slow", "patch": {"bind": {"scene": None}, "options": {"exclude": None}}})
    warned = c.event("the slow draw warned",
                     {"kind": "program", "program": "slow_ring", "change": "warned", "half": "draw"}, 180)
    st = c.status("slow_ring") or {}
    draw = st.get("draw") or {}
    warn = draw.get("warning") or {}
    if draw.get("state") != "running" or draw.get("error") is not None:
        raise Fail(f"slow_ring did not stay running: {draw}; event {warned}")
    if warn.get("code") != "program.draw_over_time" or "slow_ring" not in warn.get("message", ""):
        raise Fail(f"slow_ring warned with {warn}; event {warned}")
    if not re.fullmatch(r"draw\.passes\[\d+\]", warn.get("where", "")):
        raise Fail(f"the warning names no pass: {warn}")
    dr = (c.status("digit_bars") or {}).get("draw") or {}
    if dr.get("state") != "running":
        raise Fail(f"digit_bars does not draw: {dr}")
    c.until("training goes on", lambda: r.run_step(child) > step_at + 5, 60)
    r.result["draw_warning"] = {"code": warn["code"], "where": warn["where"], "message": warn["message"]}

    # 14. The derive halves stay inside their overhead target of 2 percent of the run.
    r.step("the derive halves stay near their target of 2 percent of the run")
    shares = {}
    for prog in ("digit_bars", "confusion_view"):
        s = (c.status(prog) or {}).get("overhead_share")
        failure = gp.targets.overhead_share(prog, s, 0.02)
        if failure:
            raise Fail(failure)
        shares[prog] = s
    r.result["overhead_share"] = shares

    # 15. Stop the runs; nothing the test started stays alive, and no file changed.
    r.step("stop the runs and compare the hashes")
    for run in (child, parent):
        r.call("run_stop", {"run": run})
    c.until("the runs stopped", lambda: all(x["state"] not in ("running", "starting", "paused", "pausing")
                                             for x in c.runs().values()), 90)
    after_hashes = {str(f): gp.sha256(f) for f in [Path(c.app), Path(c.mcp_path)] + gp.package_files()}
    if after_hashes != before:
        changed = sorted(f for f in set(before) | set(after_hashes) if before.get(f) != after_hashes.get(f))
        raise Fail(f"a file of the app, caliper-mcp or the package changed: {changed}")


def _differs(c, a, b, n):
    try:
        c.differs(a, b, n)
        return True
    except Fail:
        return False


def until_trained(c, r, run, what, pred, cap):
    """Poll pred() while `run` trains: fails when step exceeds cap or stalls for STALL_S seconds."""
    seen, since = -1, time.monotonic()
    end = since + TRAIN_WAIT_S
    while True:
        v = pred()
        if v:
            return v
        step, now = r.run_step(run), time.monotonic()
        if step > cap:
            raise Fail(f"{what}: not reached by step {cap} (step {step})")
        if step > seen:
            seen, since = step, now
        elif now - since > STALL_S:
            raise Fail(f"{what}: the run stayed at step {step} for {STALL_S} s")
        if now > end:
            raise Fail(f"{what}: not reached within {TRAIN_WAIT_S} s (step {step})")
        c.pace(1)


def run(a):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    signal.alarm(a.limit_s)
    work = Path(a.work).resolve()
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).exists():
            raise Skip(f"{exe or 'a program'} is not built")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")
           and not k.startswith("CALIPER_RUN_")}
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    pre = subprocess.run([a.python, "-c", "import numpy, torch, torchvision, caliper, caliper._netio, caliper._producer"],
                         env=env, capture_output=True, text=True)
    if pre.returncode != 0:
        why = pre.stderr.strip().splitlines()[-1] if pre.stderr.strip() else "an import failed"
        raise Skip(f"torch, torchvision, numpy, the native network sender or the producer module is missing: {why}")
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
    (ws / SCRIPT).write_text((ROOT / "examples" / "mnist" / "train.py").read_text())
    env.update({
        "CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
        "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
        "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python,
        "CALIPER_MNIST_DATA_DIR": str(mnist_data_dir()),
    })
    approvals = work / "approvals.txt"
    approvals.write_text("")
    procs, own, c = [], set(), None
    result = {}
    transcript = open(work / "transcript.jsonl", "w")
    try:
        alog = open(work / "app.log", "w")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"],
                               env=dict(env, CALIPER_TEST_APPROVALS=str(approvals)),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(app)
        lines = []

        def drain():
            for raw in app.stdout:
                alog.write(raw)
                alog.flush()
                lines.append(raw.rstrip("\n"))
        threading.Thread(target=drain, daemon=True).start()
        api_file = work / "data" / "api.json"
        end = time.monotonic() + 90
        while not api_file.exists():
            for line in list(lines):
                mm = re.match(r"\[renderer\] (\S+)", line)
                if mm and mm.group(1) != "metal":
                    raise Skip(f"the renderer is {mm.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = gp.Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        c = gp.Check(mcp, ws, work, a.capture_dir or work / "captures", HERE)
        c.app, c.mcp_path = a.app, a.mcp
        own.update(p.pid for p in procs)
        r = Run(c, approvals, transcript)
        try:
            steps(r)
        finally:
            result = dict(r.result, captures=c.result.get("captures"))
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
        if c is not None:   # the app is gone: no child it started may remain
            left = [pid for pid in sorted(c.pids | own) if gp.alive(pid)]
            for pid in left:
                print(f"a child process is left after the app ended: {pid}", file=sys.stderr)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            if left and sys.exc_info()[0] is None:
                raise Fail(f"child processes are left: {left}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app")
    ap.add_argument("--mcp")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work", required=True)
    ap.add_argument("--capture-dir")
    ap.add_argument("--limit-s", type=int, default=900)
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_programmable_graphics_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_programmable_graphics_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    print("caliper_programmable_graphics_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
