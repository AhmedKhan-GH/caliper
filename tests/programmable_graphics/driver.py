#!/usr/bin/env python3
"""The acceptance test of programmable graphics, through caliper-mcp only.

It starts the hidden app with no script and caliper-mcp as a child over stdio, then follows the steps of
tests/programmable_graphics/transcript.jsonl in order, each step with its own checks. An agent that holds nothing but
the MCP tools builds the Attention Arc Tower from the pieces of demos/arc_tower: the arcs of a two-layer model's
attention in one scene panel, and a fragment-only field of one head in another, with no built-in primitive in either
frame. The model runner starts examples/python/arc_tower_train.py, whose only Caliper line is
`caliper.watch(model, optimizer)`. The driver then scrubs to the first kept step and the bump of the loss, picks an
arc, ablates the head that formed, forks the run from a past step with the head ablated, stops a draw that is over its
time while training goes on, and checks that a foreign spec's piece waits in review.

The driver waits on events and on published state, never for a fixed time, and runs without a window and without the
network. Exit code 0 passes, 77 skips (no app, no torch, no Metal, no native sender, no producer module), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir> [--capture-dir <dir>]
                 [--limit-s 840] [--transcript <file>]
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

SCRIPT = "arc_tower_train.py"
# The head of layer 1 that the toggle names, before the agent knows which head forms.
TOGGLED = 0
# The rise of the in-context loss over its settled value at the first steps with the strongest head ablated. Two runs
# with the script's seed differ in the last digits from step 5 on (the Apple GPU's kernels are not bitwise
# deterministic), and the step at which the driver ablates depends on timing. Measured on the script without the app
# at 15 steps from 1,050 to 1,400, after the loss settled below 0.2: the first ablated step gave 4.0 to 8.8 times its
# value, an in-context loss of 0.74 to 0.96, while the other heads make up for the ablated one within tens of steps.
# Thus a settled value below 0.2 gives at least 3.7 times, and 3 keeps a margin.
RISE = 3.0
# The steps the run single-steps with the head ablated, each a pass of the views; the other heads make up for the
# ablated one within about 10 to 30 steps.
ABLATED_STEPS = 3
# The steps of the fork child past the parent's step at the bump, at which the ablated head must still be unformed.
FORK_STEPS = 500
LOG = re.compile(r"step (\d+) loss ([0-9.]+) in-context ([0-9.]+)")
# The commands that wait on a card of the person; the driver approves each as the person would.
CARDS = {"spec_new", "spec_open", "run_start", "run_pause", "run_fork", "run_stop", "piece_update", "state_create",
         "state_set", "panel_create", "panel_patch", "program_create", "source_add", "layout_place", "control_create",
         "cursor_set", "selection_set", "intervene_set", "intervene_clear", "program_remove", "panel_remove"}

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
bind = { w = "live:0.weight" }

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
    def read(self, resource, count=8, step=None, first=None, prog="tower_arcs"):
        a = {"id": prog, "resource": resource, "count": count}
        if step is not None:
            a["step"] = step
        if first is not None:
            a["first"] = first
        return self.call("program_read", a)

    def heads(self, step=None):
        b = self.read("heads", 8, step)
        return b, [e[0] for e in b["elements"]]

    def arcs(self):
        b = self.read("arcs", 1024, first=0)
        out = list(b["elements"])
        first = 1024
        while b["count"] > first:
            out += self.read("arcs", 1024, first=first)["elements"]
            first += 1024
        return b["step"], out

    def icl(self):
        d = self.call("panel_data", {"panel": "icl_curve"})["data"]["lines"][0]
        return list(zip(f64s(d["x"]), f64s(d["y"])))

    def logs(self, run):
        out = []
        for line in self.call("run_logs", {"run": run, "lines": 200}).get("lines", []):
            m = LOG.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        return out

    def run_step(self, run):
        return self.call("run_state", {"run": run}).get("step") or 0

    def status(self, pid):
        return self.c.status(pid)


def steps(r):
    c, m = r.c, r.m
    cap = c.capture

    # 1. The files that the test must leave unchanged.
    r.step("hash the app, caliper-mcp and the package")
    before = {str(f): gp.sha256(f) for f in [Path(c.app), Path(c.mcp_path)] + gp.package_files()}
    (c.work / "before.json").write_text(json.dumps(before, indent=1))

    # 2. A foreign spec's piece waits in review: no card exists yet that reviews it.
    r.step("a piece of a foreign spec waits in review")
    shared = c.ws / "shared"
    shared.mkdir()
    # The colleague's file pins the piece by its hash, which piece_check gives for the piece of the file.
    foreign_hash = r.call("piece_check", {"piece": tomllib.loads(FOREIGN)["piece"][0], "headers": False})["hash"]
    (shared / "foreign.spec.toml").write_text(FOREIGN.replace("@HASH@", foreign_hash))
    r.call("spec_open", {"path": "shared/foreign.spec.toml"})
    st = c.status("shared_vals")
    if not st or st.get("state") != "review":
        raise Fail(f"the program of a foreign spec's piece is not in review: {st}")
    r.result["foreign"] = {k: st.get(k) for k in ("state", "kind", "host")}

    # 3. The run, paused while the agent builds the instrument.
    r.step("start the run and pause it")
    r.call("spec_new", {"id": "arc_tower", "title": "Attention Arc Tower"})
    r.call("run_start", {"script": SCRIPT})
    c.run = c.until("a run", lambda: next(iter(c.runs()), None), 60)
    c.until("the run is running and attached",
            lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(c.runs().get(c.run)), 120)
    c.pids.update(gp.pids_of(c.ws / SCRIPT))
    c.until("the first step of the run", lambda: r.run_step(c.run) >= 1, 60)
    r.call("run_pause", {"op": "pause", "run": c.run})
    r.call("catalog_list", {})

    # 4. The library.
    r.step("the library lists points@1 and the four lib modules")
    lst = r.call("piece_list", {"from": "library"})
    names = {(p["name"], p["version"]) for p in lst["pieces"]}
    if ("points", 1) not in names:
        raise Fail(f"piece_list lacks points@1: {sorted(names)}")
    want = {"colormap@1", "pick@1", "camera@1", "easing@1"}
    if not want <= set(lst["modules"]):
        raise Fail(f"piece_list lacks lib modules: {lst['modules']}")
    shown = r.call("piece_show", {"name": "points", "from": "library"})
    if not (shown["piece"].get("files") or {}).get("draw"):
        raise Fail("piece_show of points has no draw file")

    # 5. Check and write the pieces of the demo.
    r.step("check and write both pieces as version 1")
    pieces = {n: piece_args(ROOT / "demos" / "arc_tower" / n) for n in ("attention_arcs", "attention_field")}
    for n, p in pieces.items():
        ck = r.call("piece_check", {"piece": p})
        if not ck["ok"]:
            raise Fail(f"piece_check of {n}: {ck['messages']}")
    for n, p in pieces.items():
        up = r.call("piece_update", {"piece": p})
        if up["version"] != 1:
            raise Fail(f"piece_update of {n} wrote version {up['version']}")
        r.result.setdefault("hashes", {})[n] = up["hash"]

    # 6. The instrument: two scene panels that hold programs alone, state, sources, curves and a toggle.
    r.step("build the instrument from pieces")
    r.call("state_create", {"id": "head", "type": "integer", "default": 0, "min": 0, "max": 3, "title": "Head"})
    for pid, title in (("tower", "Attention Arc Tower"), ("pattern", "Head Pattern")):
        r.call("panel_create", {"id": pid, "primitive": "scene", "title": title, "bind": {"scene": "live:model"},
                                "options": {"exclude": ["**"]}})
    binds = {"attn0": "live:blocks.0.attn.weights.act", "attn1": "live:blocks.1.attn.weights.act"}
    r.call("program_create", {"id": "tower_arcs", "title": "Tower Arcs", "piece": {"name": "attention_arcs",
           "version": 1, "from": "workspace"}, "panel": "tower", "bind": binds, "params": {"threshold": 0.05}})
    r.call("program_create", {"id": "head_pattern", "title": "Head Pattern", "piece": {"name": "attention_field",
           "version": 1, "from": "workspace"}, "panel": "pattern", "bind": binds,
           "params": {"layer": {"selection": "tower", "field": "layer", "default": 1}, "head": {"state": "head"},
                      "gain": 2.0}})
    r.call("panel_patch", {"id": "tower", "patch": {"bind": {"scene": None}, "options": {
        "exclude": None, "axes": False, "up": "+y", "select": {"state": "head", "field": "head"}}}})
    r.call("panel_patch", {"id": "pattern", "patch": {"bind": {"scene": None}, "options": {"exclude": None}}})
    r.call("source_add", {"id": "loss", "catalog": "live:loss", "history": 4096})
    r.call("source_add", {"id": "loss_window", "from": "loss", "transform": [{"op": "window", "steps": 4096}]})
    r.call("source_add", {"id": "icl", "catalog": "live:token_loss.act", "history": 4096, "transform": [
        {"op": "slice", "axis": 0, "index": 0}, {"op": "slice", "axis": 0, "start": 16, "stop": 27},
        {"op": "reduce", "axis": 0, "fn": "mean"}]})
    r.call("source_add", {"id": "icl_window", "from": "icl", "transform": [{"op": "window", "steps": 4096}]})
    for pid, title, src in (("loss_curve", "Loss", "loss_window"), ("icl_curve", "In-Context Loss", "icl_window")):
        r.call("panel_create", {"id": pid, "primitive": "series", "title": title,
                                "bind": {"x": f"{src}.step", "y": f"{src}.value"},
                                "options": {"x_label": "Step", "y_label": "Loss"}})
    r.call("layout_place", {"panel": "pattern", "target": "tower", "where": "beside", "ratio": 0.4})
    r.call("layout_place", {"panel": "loss_curve", "target": "pattern", "where": "below", "ratio": 0.5})
    r.call("layout_place", {"panel": "icl_curve", "target": "loss_curve", "where": "below", "ratio": 0.5})
    r.call("control_create", {"id": "ablate", "kind": "toggle", "title": f"Ablate Head {TOGGLED} of Layer 1",
                              "bind": f"intervene:blocks.1.attn#{TOGGLED}", "options": {"value": False}})
    if "[[panel.entity]]" in r.call("spec_get", {}).get("text", ""):
        raise Fail("the spec holds an entity: a primitive of the scene is in the frame")
    cap("empty")

    # 7. Both halves of both programs compile; the run goes on.
    r.step("both halves of both programs compile")
    r.call("run_pause", {"op": "resume", "run": c.run})
    for prog in ("tower_arcs", "head_pattern"):
        for half, host in (("derive", "script"), ("draw", "viewer")):
            try:
                c.event(f"{prog} {half} compiled",
                        {"kind": "program", "program": prog, "change": "compiled", "half": half}, 120)
            except Fail as e:
                seen = [ev for ev in c.events if ev.get("kind") == "program"]
                raise Fail(f"{e}; status {c.status(prog)}; program events {seen[-8:]}") from e

    # 8. The arcs exist and the tower draws.
    r.step("the arcs are published and the tower draws")
    got = c.until("an arc version", lambda: (lambda b: b if b.get("count") else None)(r.read("arcs", 1024)), 90)
    for e in got["elements"]:
        ft, tt = e[0], e[1]
        if not (ft < 32 and tt < 32 and ft < tt):
            raise Fail(f"an arc has the tokens {ft} to {tt}")
    cap("tower-early")
    n = c.differs("tower-early", "empty", 2000)
    r.result["tower_pixels"] = n
    if "[[panel.entity]]" in r.call("spec_get", {}).get("text", ""):
        raise Fail("the panel has an entity")

    # 9. The bump: the newest version scores above 0.4, and the first kept version holds no score above 0.2.
    r.step("the heads form at the bump, and the first kept version holds none")
    def formed():
        b, h = r.heads()
        return (b, h) if max(h[4:]) > 0.4 else None
    newest, h_new = c.until("a head with a score above 0.4", formed, 420)
    first_step = newest["kept"]["first"]
    old, h_old = r.heads(first_step)
    if old["step"] != first_step or max(h_old) >= 0.2:
        raise Fail(f"the first kept version (step {old['step']}) holds scores {h_old}")
    r.call("cursor_set", {"step": first_step})
    cap("bump-before")
    r.call("cursor_set", {"live": True})
    cap("bump-after")
    n = c.differs("bump-before", "bump-after", 500)
    log = r.logs(c.run)
    if not log or log[0][1] < 3.5 or min(x[2] for x in log) > 1.5 or log[-1][1] > 3.4:
        raise Fail(f"the loss shows no bump: first {log[:1]}, last {log[-1:]}")
    r.result["bump"] = {"first_kept": first_step, "newest": newest["step"], "scores": h_new,
                        "loss_first": log[0][1], "loss_last": log[-1][1], "tower_pixels_changed": n}

    # 10. A pick of the brightest arc of the best head.
    r.step("a pick returns the layer, the head and the token pair")
    r.call("run_pause", {"op": "pause", "run": c.run})
    c.until("the run paused", lambda: c.runs()[c.run]["state"] == "paused", 60)
    _, h = r.heads()
    best = max(range(4), key=lambda i: h[4 + i])
    r.call("state_set", {"id": "head", "value": (best + 1) % 4})
    _, arcs = r.arcs()
    mine = [(e[4], i) for i, e in enumerate(arcs) if e[2] == 1 and e[3] == best]
    if not mine:
        raise Fail(f"no arc of head {best} of layer 1 among {len(arcs)} arcs")
    item = max(mine)[1]
    cap("pick-before")
    fields = r.call("selection_set", {"panel": "tower", "program": "tower_arcs", "item": item})["fields"]
    ev = c.event("the pick event", {"kind": "pick", "program": "tower_arcs"}, 30)
    want = arcs[item]
    flds = ev.get("fields") or {}
    for key, val in (("layer", 1), ("head", best), ("from_token", want[0]), ("to_token", want[1])):
        if flds.get(key) != val or fields.get("fields", fields).get(key) != val:
            raise Fail(f"the pick has {key} {flds.get(key)}, not {val}; event {ev}")
    state = {s["id"]: s["value"] for s in r.call("state_get", {})["states"]}
    if state.get("head") != best:
        raise Fail(f"the state head is {state.get('head')}, not {best}")
    def pattern_changed():
        cap("pick-after")
        return _differs(c, "pick-before", "pick-after", 1000)
    c.until("the pattern panel follows the pick", pattern_changed, 30)
    r.call("run_pause", {"op": "resume", "run": c.run})
    r.result["pick"] = {"item": item, "layer": 1, "head": best, "tokens": [want[0], want[1]]}
    cap("tower-parent")

    # 11. Ablating the strongest head raises the in-context loss at once. The run is paused, the head with the highest
    # score of layer 1 in the newest published version is ablated, and the run single-steps, so that each ablated step
    # is a pass of the views and its in-context loss is published, whatever the shared cadence of the views and the
    # time at which the ablation reaches the script. The other heads make up for the ablated one within tens of steps.
    r.step("the ablation of the strongest head raises the in-context loss")
    def converged():
        ys = [y for _, y in r.icl()[-5:]]
        return sum(ys) / len(ys) if len(ys) == 5 and sum(ys) / 5 < 0.2 else None
    base = c.until("the in-context loss settles below 0.2", converged, 400)
    r.call("run_pause", {"op": "pause", "run": c.run})
    c.until("the run paused", lambda: c.runs()[c.run]["state"] == "paused", 60)
    _, hs = r.heads()
    strong = max(range(4), key=lambda i: hs[4 + i])
    point = f"blocks.1.attn#{strong}"
    last = r.icl()[-1][0]
    r.call("intervene_set", {"point": point, "op": "ablate"})
    for _ in range(ABLATED_STEPS):
        at = r.run_step(c.run)
        r.call("run_pause", {"op": "step", "count": 1, "run": c.run})
        c.until("the single step", lambda: r.run_step(c.run) > at and c.runs()[c.run]["state"] == "paused", 60)
        c.until("the in-context loss of the step", lambda: any(x > at for x, _ in r.icl()), 60)
    after = [(x, y) for x, y in r.icl() if x > last]
    rise = max(y for _, y in after) / base
    if rise < RISE:
        raise Fail(f"with head {strong} of layer 1 ablated the in-context loss rose {rise:.2f} times, from {base} to "
                   f"{after}; scores {hs}")
    r.call("intervene_clear", {"point": point})
    r.call("run_pause", {"op": "resume", "run": c.run})
    r.result["ablation"] = {"point": point, "scores": hs[4:], "before": base, "after": after, "rise": rise}

    # 12. A fork from the first kept step with the strong head ablated never forms that head.
    r.step("a fork from a past step with the head ablated")
    parent = c.run
    kids_before = set(c.runs())
    fork = r.call("run_fork", {"run": parent, "step": first_step,
                               "interventions": [{"point": point, "op": "ablate"}]})["run"]
    child = fork["id"]
    c.run = child
    c.until("the child runs and is attached",
            lambda: (lambda x: x and x["state"] == "running" and x.get("attached"))(c.runs().get(child)), 120)
    c.pids.update(gp.pids_of(c.ws / SCRIPT))
    target = newest["step"] + FORK_STEPS
    def child_formed():
        b, hh = r.heads()
        return (b, hh) if (b.get("step") or 0) >= target and max(hh[4 + i] for i in range(4) if i != strong) > 0.4 \
            else None
    cb, ch = c.until(f"the other heads of the child form by step {target}", child_formed, 420)
    if ch[4 + strong] >= 0.2:
        raise Fail(f"the ablated head {strong} scores {ch[4 + strong]} in the child at step {cb['step']}")
    cap("tower-child")
    n = c.differs("tower-parent", "tower-child", 300)
    r.result["fork"] = {"child": child, "step": cb["step"], "scores": ch, "tower_pixels_changed": n}

    # 13. A draw over its time is removed and named, and training goes on.
    r.step("a draw over its time is stopped and named while training goes on")
    slow = dict(pieces["attention_arcs"])
    slow["name"] = "attention_arcs_slow"
    slow["params"] = list(slow["params"]) + [{"name": "burn", "default": 20000.0, "min": 0.0, "max": 100000.0}]
    # The draw half's loop holds the GPU that the script's derive half shares, which can put a derive pass of 4 ms,
    # the default, over its limit three times in a row before the draw half's 30 frames: the copy's derive half
    # takes the highest limit, and the wait below takes the draw half's stop only.
    slow["derive"] = dict(slow["derive"], time_ms=50.0)
    d = re.sub(r"( *)return (sum \* [0-9.]+ \* params\.glow);",
               r"\1var burn = 0.0;\n\1for (var i = 0u; i < u32(params.burn); i = i + 1u) {\n"
               r"\1    burn = burn + sin(f32(i) * 0.001 + in.uv.x) * 0.0000001;\n\1}\n"
               r"\1return \2 + vec4<f32>(burn);", slow["files"]["draw"])
    if "burn" not in d:
        raise Fail("the bloom of the arcs did not change")
    slow["files"] = dict(slow["files"], draw=d)
    r.call("piece_update", {"piece": slow})
    r.call("panel_create", {"id": "slow", "primitive": "scene", "title": "Slow Arcs", "bind": {"scene": "live:model"},
                            "options": {"exclude": ["**"]}})
    step_at = r.run_step(child)
    r.call("program_create", {"id": "slow_arcs", "title": "Slow Arcs", "piece": {"name": "attention_arcs_slow",
           "version": 1, "from": "workspace"}, "panel": "slow", "bind": binds, "params": {"threshold": 0.05}})
    r.call("panel_patch", {"id": "slow", "patch": {"bind": {"scene": None}, "options": {"exclude": None}}})
    stop = c.event("the slow draw stopped",
                   {"kind": "program", "program": "slow_arcs", "change": "stopped", "half": "draw"}, 180)
    st = c.status("slow_arcs") or {}
    draw = st.get("draw") or {}
    err = draw.get("error") or {}
    if err.get("code") != "program.draw_over_time" or "slow_arcs" not in err.get("message", ""):
        raise Fail(f"slow_arcs stopped with {err}; event {stop}")
    if not re.fullmatch(r"draw\.passes\[\d+\]", err.get("where", "")):
        raise Fail(f"the stop names no pass: {err}")
    tw = (c.status("tower_arcs") or {}).get("draw") or {}
    if tw.get("state") != "running":
        raise Fail(f"tower_arcs does not draw: {tw}")
    c.until("training goes on", lambda: r.run_step(child) > step_at + 5, 60)
    r.result["draw_stop"] = {"code": err["code"], "where": err["where"], "message": err["message"]}

    # 14. The derive halves stay inside their share of the run.
    r.step("the derive halves stay at or below 2 percent of the run")
    shares = {}
    for prog in ("tower_arcs", "head_pattern"):
        s = (c.status(prog) or {}).get("overhead_share")
        if s is None or s > 0.02:
            raise Fail(f"{prog} has an overhead_share of {s}")
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
        raise Fail("a file of the app, caliper-mcp or the package changed")


def _differs(c, a, b, n):
    try:
        c.differs(a, b, n)
        return True
    except Fail:
        return False


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
    pre = subprocess.run([a.python, "-c", "import numpy, torch, caliper, caliper._netio, caliper._producer"],
                         env=env, capture_output=True, text=True)
    if pre.returncode != 0:
        why = pre.stderr.strip().splitlines()[-1] if pre.stderr.strip() else "an import failed"
        raise Skip(f"torch, numpy, the native network sender or the producer module is missing: {why}")
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
    (ws / SCRIPT).write_text((ROOT / "examples" / "python" / SCRIPT).read_text())
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python})
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
    ap.add_argument("--limit-s", type=int, default=840)
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
