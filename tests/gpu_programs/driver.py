#!/usr/bin/env python3
"""The driver of the GPU programs acceptance test and its check script.

It starts the hidden app with no script and caliper-mcp as a child over stdio, copies tests/gpu_programs/train.py
into a throwaway workspace, and sends tests/gpu_programs/transcript.jsonl through the MCP tools. The model runner of
the app starts the training script, and the script runs the programs of the agent in its own process on the live
tensors. The driver waits on events and on published state, never for a fixed time. It runs without a window
and without the network. Exit code 0 passes, 77 skips (no app, no torch, no Metal, no producer module, no WGSL
compiler library), 1 fails.

usage: driver.py --app <caliper> --mcp <caliper-mcp> --python <python> --work <dir>
                 [--capture-dir <dir>] [--transcript <file>] [--limit-s 280]
"""
import argparse
import base64
import hashlib
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
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKIP = 77
# The overhead targets that the acceptance tests share: a measured share is printed with its target, a
# miss is a warning, and only a share above twice the target fails.
_targets_spec = importlib.util.spec_from_file_location("overhead_targets",
                                                       ROOT / "python" / "tests" / "overhead_targets.py")
targets = importlib.util.module_from_spec(_targets_spec)
_targets_spec.loader.exec_module(targets)
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "gpu-programs-check", "version": "1"}}
# The decisions the person gives at the cards, through the app's test seam. Each line decides the oldest
# waiting card of its command. A line that finds no card waits for one and does no harm.
APPROVALS = (["approve spec.new mcp", "approve run.start mcp", "approve run.stop mcp",
              "approve control.create mcp"]
             + ["approve panel.create mcp"] * 3 + ["approve source.add mcp"] * 2
             + ["approve piece.update mcp"] * 2 + ["approve program.create mcp"] * 5)
LOSS = re.compile(r"step (\d+) loss ([0-9.eE+-]+)")


class Fail(Exception):
    pass


class Skip(Exception):
    pass


class Mcp:
    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env=env)
        self.n = 0

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
        r = self.request("tools/call", {"name": tool, "arguments": arguments})
        if "result" not in r:
            raise Fail(f"{tool}: {r}")
        return r["result"]

    def body(self, tool, arguments):
        r = self.call(tool, arguments)
        if r.get("isError"):
            raise Fail(f"{tool} answered an error: {json.dumps(r)[:800]}")
        return r.get("structuredContent", {})


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
    # A zombie has no code to run: it counts as ended.
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


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
    """The files of the installed Python package: the sources and the native modules, not the caches."""
    pkg = ROOT / "python" / "caliper"
    return sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def decode_png(data):
    """The pixels of an 8-bit RGB or RGBA PNG as an array of height x width x channels."""
    import numpy as np
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise Fail("a capture is not a PNG")
    p, idat, hdr = 8, [], None
    while p < len(data):
        n, t = struct.unpack(">I4s", data[p:p + 8])
        if t == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data[p + 8:p + 21])
        elif t == b"IDAT":
            idat.append(data[p + 8:p + 8 + n])
        p += 12 + n
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        raise Fail(f"a capture is a PNG of a kind the driver does not read: {hdr}")
    bpp = 3 if ctype == 2 else 4
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(h):
        base = y * (stride + 1)
        f = raw[base]
        cur = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=base + 1).copy()
        if f == 1:
            cur = cur.reshape(w, bpp).cumsum(axis=0, dtype=np.uint8).reshape(-1)
        elif f == 2:
            cur += prev
        elif f in (3, 4):
            c, pv = cur.tolist(), prev.tolist()
            for i in range(stride):
                a = c[i - bpp] if i >= bpp else 0
                b = pv[i]
                if f == 3:
                    c[i] = (c[i] + ((a + b) >> 1)) & 255
                else:
                    cc = pv[i - bpp] if i >= bpp else 0
                    pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
                    c[i] = (c[i] + (a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc))) & 255
            cur = np.array(c, dtype=np.uint8)
        out[y] = cur
        prev = cur
    return out.reshape(h, w, bpp)


class Check:
    """One run of the transcript: the connection, the event cursor, the captures and the pids seen."""

    def __init__(self, mcp, ws, work, capture_dir, programs_dir):
        self.mcp = mcp
        self.ws = ws
        self.work = work
        self.capture_dir = Path(capture_dir) if capture_dir else work / "captures"
        self.programs_dir = programs_dir
        self.cursor = 0
        self.events = []
        self.pids = set()
        self.images = {}
        self.hashes = {}
        self.result = {}
        self.run = None

    # -- waits on events and on published state -------------------------------------------------------
    def pace(self, seconds):
        """Waits for an event, at most the time: the pace of a poll of published state."""
        b = self.mcp.body("events_wait", {"cursor": self.cursor, "timeout_s": seconds})
        self.cursor = b.get("cursor", self.cursor)
        self.events.extend(b.get("events", []))

    def until(self, what, pred, limit=60):
        end = time.monotonic() + limit
        while True:
            v = pred()
            if v:
                return v
            if time.monotonic() > end:
                raise Fail(f"{what}: not reached within {limit} s")
            self.pace(1)

    def event(self, what, match, limit=60):
        """The first event, seen or to come, whose fields hold every key of `match`."""
        def find():
            for e in self.events:
                if all(e.get(k) == v for k, v in match.items()):
                    return e
            return None
        return self.until(what, find, limit)

    def runs(self):
        return {r["id"]: r for r in self.mcp.body("run_list", {}).get("runs", [])}

    def step_of_run(self):
        return self.mcp.body("run_state", {"run": self.run}).get("step") or 0

    def status(self, pid):
        b = self.mcp.body("program_status", {})
        for item in b.get("programs", []):
            if item.get("id") == pid:
                return item
        return None

    def losses(self):
        lines = self.mcp.body("run_logs", {"run": self.run, "lines": 200}).get("lines", [])
        out = []
        for line in lines:
            m = LOSS.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
        return out

    def loss_after(self, step, count=5, limit=60):
        """The first `count` losses that the run printed after `step`, once it printed them."""
        def enough():
            got = [x for x in self.losses() if x[0] > step]
            return got[:count] if len(got) >= count else None
        return self.until(f"{count} losses after step {step}", enough, limit)

    def passes(self, pid, more, limit=60):
        """Waits until the program ran `more` passes after now, so that the app has its newest output."""
        first = (self.status(pid) or {}).get("runs") or 0
        return self.until(f"{more} more passes of {pid}",
                          lambda: ((self.status(pid) or {}).get("runs") or 0) >= first + more, limit)

    # -- captures ---------------------------------------------------------------------------------------
    def capture(self, name, panel=None):
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        r = self.mcp.call("capture_frame", {})
        if r.get("isError"):
            raise Fail(f"capture_frame failed: {json.dumps(r)[:400]}")
        data = base64.b64decode(next(c for c in r["content"] if c["type"] == "image")["data"])
        out = self.capture_dir / (name + ".png")
        out.write_bytes(data)
        self.images[name] = decode_png(data)
        paths = [str(out)]
        if panel:
            r = self.mcp.call("panel_render", {"id": panel, "width": 640, "height": 480})
            if r.get("isError"):
                raise Fail(f"panel_render {panel} failed: {json.dumps(r)[:400]}")
            data = base64.b64decode(next(c for c in r["content"] if c["type"] == "image")["data"])
            out = self.capture_dir / (name + "-panel.png")
            out.write_bytes(data)
            paths.append(str(out))
        self.result.setdefault("captures", {})[name] = paths

    def differs(self, a, b, min_pixels, tolerance=24):
        """The count of pixels whose color differs between two captures by more than `tolerance` in a channel."""
        import numpy as np
        x, y = self.images[a][..., :3].astype(int), self.images[b][..., :3].astype(int)
        if x.shape != y.shape:
            raise Fail(f"the captures {a} and {b} differ in size")
        n = int((np.abs(x - y).max(axis=2) > tolerance).sum())
        if n < min_pixels:
            raise Fail(f"only {n} pixels differ between the captures {a} and {b}, fewer than {min_pixels}")
        return n

    def yellow(self, name):
        """The count of pixels in the yellow and orange range that the material piece draws, in the lower left
        quarter of the frame, where the points of the geometry program are and the yellow outline of a module is
        not."""
        im = self.images[name].astype(int)
        im = im[im.shape[0] // 2:, :im.shape[1] // 2]
        return int(((im[..., 0] > 200) & (im[..., 1] > 120) & (im[..., 2] < 90)).sum())

    # -- the steps --------------------------------------------------------------------------------------
    def file_hashes(self):
        files = [Path(self.app), Path(self.mcp_path)] + package_files()
        return {str(f): sha256(f) for f in files}

    def toggle(self, w):
        m = self.mcp
        pts = m.body("intervene_points", {})
        names = [p.get("point") for p in pts.get("points", [])]
        if w["point"] not in names:
            raise Fail(f"intervene_points lacks {w['point']}: {names[:20]}")
        m.body("control_create", {"id": w["id"], "kind": "toggle", "title": "Ablate Layer " + w["point"],
                                  "bind": "intervene:" + w["point"], "place": {"target": w["below"], "where": "below"}})
        base = self.loss_after(self.step_of_run(), 5)
        (self.work / "toggle_on").write_text("on\n")      # the scripted input of the app clicks the toggle
        on = self.event("the ablation applied", {"kind": "run", "change": "intervene", "point": w["point"],
                                                  "op": "ablate"})
        on_step = on.get("step") or self.step_of_run()
        during = self.loss_after(on_step + 10, 5)
        self.capture(w["capture"])
        (self.work / "toggle_off").write_text("off\n")    # the scripted input clicks it again
        off = self.event("the ablation cleared", {"kind": "run", "change": "intervene", "point": w["point"],
                                                   "op": None})
        off_step = off.get("step") or self.step_of_run()
        after = self.loss_after(off_step + 30, 5)
        mean = lambda xs: sum(v for _, v in xs) / len(xs)
        res = {"mean_before": mean(base), "mean_during": mean(during), "mean_after": mean(after),
               "on_step": on_step, "off_step": off_step}
        self.result["toggle"] = res
        if not res["mean_during"] > w["raise_factor"] * res["mean_before"]:
            raise Fail(f"the loss did not rise by {w['raise_factor']} times under the ablation: {res}")
        if not res["mean_after"] < w["return_factor"] * res["mean_during"]:
            raise Fail(f"the loss did not fall below {w['return_factor']} of its ablated value after the "
                       f"release: {res}")

    def with_wgsl(self, s, args):
        if "wgsl_file" in s:
            args["wgsl"] = (self.programs_dir / s["wgsl_file"]).read_text()
        if "piece_files" in s:
            # A piece as the agent writes it: the manifest's keys and its WGSL files, each edit of
            # `piece_edit` (a file and an [old, new] pair) applied, as an agent's typo is.
            files = s["piece_files"]
            piece = json.loads((self.programs_dir / files["manifest"]).read_text())
            piece["files"] = {k: (self.programs_dir / v).read_text() for k, v in files.items() if k != "manifest"}
            for part, (old, new) in s.get("piece_edit", {}).items():
                if old not in piece["files"][part]:
                    raise Fail(f"the edit of {part} finds no {old!r}")
                piece["files"][part] = piece["files"][part].replace(old, new)
            args["piece"] = piece
        return args

    def step(self, s):
        name = s["step"]
        m = self.mcp
        if "hash" in s:
            self.hashes[s["hash"]] = self.file_hashes()
            self.result["hashed_files"] = len(self.hashes[s["hash"]])
            return
        if "hash_same" in s:
            before, now = self.hashes[s["hash_same"]], self.file_hashes()
            changed = sorted(set(before) ^ set(now)) + [f for f in before if f in now and before[f] != now[f]]
            if changed:
                raise Fail(f"the SHA-256 of these files changed: {changed}")
            self.result["unchanged_files"] = len(now)
            return
        if "wait_attached" in s:
            self.run = s["wait_attached"]["run"]
            self.until("the run is running and attached",
                       lambda: (lambda r: r and r["state"] == "running" and r.get("attached"))(
                           self.runs().get(self.run)), 120)
            return
        if "steps_grow" in s:
            first = self.until("a first step", self.step_of_run, 60)
            self.until("the step grows", lambda: self.step_of_run() > first, 60)
            self.result[name] = {"first": first, "later": self.step_of_run()}
            return
        if "record_pids" in s:
            found = pids_of(self.ws / s["record_pids"])
            if not found:
                raise Fail(f"{name}: no process runs {s['record_pids']}")
            self.pids.update(found)
            self.result[name] = sorted(self.pids)
            return
        if "wait_program" in s:
            w = s["wait_program"]
            st = self.until(f"{w['id']} is {w['states']}",
                            lambda: (lambda x: x if x and x.get("state") in w["states"] and
                                     (x.get("runs") or 0) >= w["runs"] else None)(self.status(w["id"])), 90)
            self.result[name] = st
            return
        if "wait_event" in s:
            try:
                self.result[name] = self.event(name, s["wait_event"], s.get("limit_s", 30))
            except Fail as e:
                seen = [ev for ev in self.events if ev.get("kind") == "program"][-8:]
                raise Fail(f"{e}; the last program events {seen}") from e
            return
        if "wait_draw" in s:
            pid = s["wait_draw"]["id"]
            self.result[name] = self.until(
                f"the draw half of {pid} runs with a resource version",
                lambda: (lambda d: d if d.get("state") == "running" and d.get("resource_step") is not None
                         else None)((self.status(pid) or {}).get("draw") or {}), 60)
            return
        if "piece_budget" in s:
            # Both halves of an instance inside their budgets: the derive half runs at the shared cadence
            # within its memory, and the draw half runs within its memory. A draw over its time on 30
            # frames in a row, or a derive pass over its time, would have stopped it.
            pid = s["piece_budget"]["id"]
            st = self.status(pid) or {}
            draw = st.get("draw") or {}
            if st.get("kind") != "piece" or st.get("host") != "script" or st.get("state") != "running":
                raise Fail(f"{name}: {pid} is not a running piece in the script: {json.dumps(st)[:600]}")
            if not isinstance(st.get("every"), int) or st["every"] < 1:
                raise Fail(f"{name}: the cadence N of {pid} is {st.get('every')!r}")
            for half, d in (("derive", st), ("draw", draw)):
                mem, limit = d.get("memory_bytes"), d.get("memory_limit_bytes")
                if not isinstance(mem, int) or mem <= 0 or (limit and mem > limit):
                    raise Fail(f"{name}: the {half} half of {pid} holds {mem!r} bytes against {limit!r}")
            if draw.get("state") != "running":
                raise Fail(f"{name}: the draw half of {pid} is {draw.get('state')!r}: {json.dumps(draw)[:400]}")
            self.result[name] = {"every": st.get("every"), "overhead_share": st.get("overhead_share"),
                                 "memory_bytes": st.get("memory_bytes"), "draw_frame_ms": draw.get("frame_ms"),
                                 "draw_frame_limit_ms": draw.get("frame_limit_ms"),
                                 "draw_memory_bytes": draw.get("memory_bytes")}
            return
        if "passes" in s:
            self.passes(s["passes"]["id"], s["passes"]["more"])
            return
        if "wait_catalog" in s:
            self.until(f"{s['wait_catalog']} is in the catalog",
                       lambda: not m.call("catalog_describe", {"id": s["wait_catalog"]}).get("isError"), 60)
            return
        if "wait_panel_data" in s:
            def has_data():
                r = m.call("panel_data", {"panel": s["wait_panel_data"]["panel"]})
                return bool(((r.get("structuredContent") or {}).get("data") or {})) and not r.get("isError")
            self.until("the image panel has data", has_data, 60)
            return
        if "capture" in s:
            self.capture(s["capture"], s.get("panel"))
            return
        if "differs" in s:
            w = s["differs"]
            self.result[name] = self.differs(w["a"], w["b"], w["min_pixels"])
            return
        if "yellow_gain" in s:
            w = s["yellow_gain"]
            before, after = self.yellow(w["before"]), self.yellow(w["after"])
            self.result[name] = {"before": before, "after": after}
            if after < w["min_after"] or before * w["ratio"] > after:
                raise Fail(f"the material piece did not change the look: {before} yellow pixels before, "
                           f"{after} after (at least {w['min_after']}, and {w['ratio']} times the first)")
            return
        if "expect_error" in s:
            w = s["expect_error"]
            r = m.call(w["tool"], self.with_wgsl(w, dict(w["arguments"])))
            text = json.dumps(r)
            if not r.get("isError") or w["code"] not in text:
                self.compiler_check(r)
                raise Fail(f"{name}: expected the error {w['code']}, got {text[:600]}")
            for word in w.get("names", []):
                if word not in text:
                    raise Fail(f"{name}: the error {w['code']} does not name {word}: {text[:600]}")
            self.result[name] = {"code": w["code"], "message": text[:400]}
            return
        if "no_status" in s:
            if self.status(s["no_status"]) is not None:
                raise Fail(f"{name}: the refused program {s['no_status']} is in program.status")
            return
        if "over_time" in s:
            w = s["over_time"]
            args = self.with_wgsl(w, dict(w["arguments"]))
            before = self.step_of_run()
            r = m.call("program_create", args)
            if r.get("isError"):
                self.compiler_check(r)
                raise Fail(f"program_create {w['id']} answered an error: {json.dumps(r)[:600]}")
            ev = self.event("the slow program stopped", {"kind": "program", "program": w["id"],
                                                         "change": "stopped"}, 120)
            st = self.status(w["id"])
            if not st or (st.get("error") or {}).get("code") != "program.over_time":
                raise Fail(f"{w['id']} did not stop with program.over_time: {st}")
            at = self.step_of_run()
            self.until("training goes on after the stop", lambda: self.step_of_run() > at + w["steps_after"], 60)
            self.result[name] = {"event": ev, "before_install": before, "at_stop": at, "later": self.step_of_run()}
            return
        if "status_fields" in s:
            w = s["status_fields"]
            st = self.status(w["id"])
            if not st:
                raise Fail(f"{name}: program.status has no {w['id']}")
            if not isinstance(st.get("every"), int) or st["every"] < w["every_min"]:
                raise Fail(f"{name}: the cadence N is {st.get('every')!r}, expected at least {w['every_min']}")
            if st.get("last_pass_ms") is None:
                raise Fail(f"{name}: no last pass time: {st}")
            mem, limit = st.get("memory_bytes"), st.get("memory_limit_bytes")
            if not mem or mem <= 0 or (limit and mem > limit):
                raise Fail(f"{name}: memory in use {mem!r} against the limit {limit!r}")
            if "share_target" in w:
                # The measured overhead share comes down to the budget once the cadence settles; a
                # pass of a stopped program or a slow cycle can lift it for a few cycles before that.
                # The driver waits up to settle_s for the target, then judges the last share: a miss
                # is a warning, and only a share above twice the target fails.
                last, target = {}, w["share_target"]
                end = time.monotonic() + w.get("settle_s", 90)

                def settled():
                    last["st"] = self.status(w["id"]) or {}
                    share = last["st"].get("overhead_share")
                    return (isinstance(share, (int, float)) and 0 <= share <= target) or time.monotonic() > end
                self.until(f"the overhead share of {w['id']} reaches its target {target}", settled,
                           w.get("settle_s", 90) + 30)
                st = last["st"]
                failure = targets.overhead_share(w["id"], st.get("overhead_share"), target)
                if failure:
                    raise Fail(f"{name}: {failure}; last status {json.dumps(st)[:600]}")
            self.result[name] = {k: st.get(k) for k in ("every", "last_pass_ms", "last_cost_ms", "step_ms",
                                                         "overhead_share", "passes_per_s", "memory_bytes",
                                                         "memory_limit_bytes", "runs")}
            return
        if "toggle" in s:
            self.toggle(s["toggle"])
            return
        if "wait_state" in s:
            w = s["wait_state"]
            self.until(f"run {w['run']} is {w['states']}",
                       lambda: (self.runs().get(w["run"]) or {}).get("state") in w["states"], 90)
            return
        if "no_children" in s:
            self.until("no child process is left", lambda: not [p for p in sorted(self.pids) if alive(p)], 60)
            return
        args = self.with_wgsl(s, dict(s.get("arguments", {})))
        r = m.call(s["tool"], args)
        if r.get("isError"):
            self.compiler_check(r)
            raise Fail(f"{name}: {s['tool']} answered an error: {json.dumps(r)[:800]}")
        body = r.get("structuredContent", {})
        if "expect_run" in s:
            run = body.get("run", {})
            for k, v in s["expect_run"].items():
                if run.get(k) != v:
                    raise Fail(f"{name}: run {k} is {run.get(k)!r}, expected {v!r}")

    def compiler_check(self, r):
        if "program.compiler_unavailable" in json.dumps(r):
            raise Skip("the build has no WGSL compiler library (CALIPER_WGSL_FROM_SOURCE=ON builds it)")


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
    (ws / "train.py").write_text((HERE / "train.py").read_text())
    env.update({"CALIPER_AGENT_OFFLINE": "1", "CALIPER_HIDDEN": "1", "CALIPER_DATA_DIR": str(work / "data"),
                "CALIPER_USER_APPLETS_DIR": str(work / "user"), "CALIPER_WINDOW_SIZE": "1280x800",
                "CALIPER_WORKSPACE": str(ws), "CALIPER_RUN_PYTHON": a.python})
    decisions = work / "approvals.txt"
    decisions.write_text("\n".join(APPROVALS) + "\n")
    # The scripted input of the app clicks the toggle when the driver writes the first file, and again when
    # it writes the second one.
    ui = (f"patience 280; wait-file {work / 'toggle_on'}; click panel:ablate; "
          f"wait-file {work / 'toggle_off'}; click panel:ablate")
    steps = [json.loads(l) for l in Path(a.transcript).read_text().splitlines() if l.strip()]
    procs = []
    own = set()
    check = None
    try:
        app_env = dict(env, CALIPER_TEST_APPROVALS=str(decisions), CALIPER_TEST_SPEC_UI=ui)
        alog = open(work / "app.log", "w")
        app = subprocess.Popen([a.app, "--hidden", "--api", "--api-port", "0"], env=app_env,
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
                m = re.match(r"\[renderer\] (\S+)", line)
                if m and m.group(1) != "metal":
                    raise Skip(f"the renderer is {m.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("the app ended before it wrote its API file: " + str(lines[-5:]))
            if time.monotonic() > end:
                raise Fail("the app wrote no API file within 90 seconds")
            time.sleep(0.05)   # polls published state: the file the app writes when its API listens
        mcp = Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        mcp.request("server/discover", {})
        check = Check(mcp, ws, work, a.capture_dir, HERE)
        check.app, check.mcp_path = a.app, a.mcp
        own.update(p.pid for p in procs)   # the app and caliper-mcp end too
        check.result["processes"] = sorted(own)
        for s in steps:
            check.step(s)
        return check.result
    finally:
        for p in reversed(procs):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
            stop(p)
        if check is not None:   # the app is gone: no child it started may remain
            left = [pid for pid in sorted(check.pids | own) if alive(pid)]
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
    ap.add_argument("--transcript", default=str(HERE / "transcript.jsonl"))
    ap.add_argument("--limit-s", type=int, default=280)
    a = ap.parse_args()
    try:
        res = run(a)
    except Skip as e:
        print(f"caliper_gpu_programs_acceptance skipped: {e}")
        return SKIP
    except Fail as e:
        print(f"caliper_gpu_programs_acceptance FAILED: {e}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    print("caliper_gpu_programs_acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
