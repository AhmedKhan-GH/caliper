#!/usr/bin/env python3
"""Agent-built MNIST instrument and first-layer filter intervention through caliper-mcp.

Drives the hidden Caliper app exclusively through ``caliper-mcp`` on a managed run of
``examples/mnist/train.py`` (the canonical PyTorch MNIST CNN from pytorch/examples/mnist):

1. Builds a live instrument containing:
   - a 10x10 row-normalized confusion matrix (one-pass WGSL GPU tensor program + existing
     ``heatmap`` panel primitive);
   - a paginated page of misclassified digits with predicted and true labels (library piece
     ``image_grid@1`` in a ``scene`` panel, bound to a ``page`` state slider);
   - a live training accuracy curve (existing ``series`` panel primitive);
   - a live toggle bound to the first-layer filter intervention point.
2. Captures the visible-size (1280x800) instrument before intervention, switches off one
   first-layer conv filter through the intervention points published by ``caliper.watch``,
   verifies that per-digit accuracy and misclassified digits change, captures the instrument
   after intervention, and clears the intervention to verify recovery.

Usage:
    python examples/mnist/agent_instrument.py [<capture_dir>]
        [--app <caliper>] [--mcp <caliper-mcp>] [--python <python>]
        [--data-dir <mnist_cache>] [--work <work_dir>] [--limit-s 300] [--ctest]

Exit codes:
    0   passed
    1   failed
    2   cannot run here (standalone mode)
    77  skipped (``--ctest`` mode, for CTest SKIP_RETURN_CODE)
"""
from __future__ import annotations

import argparse
import base64
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

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "examples" / "mnist" / "train.py"
DEFAULT_DATA_DIR = Path(
    os.environ.get("CALIPER_MNIST_DATA_DIR", str(Path.home() / ".cache" / "caliper" / "mnist"))
).expanduser()

PLACES = {"caliper": "caliper", "caliper-mcp": "src/builder/mcp/caliper-mcp"}
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "mnist-agent-instrument", "version": "1"},
}
# Commands that may wait for a person's approval card in the hidden app.
CARDS = {
    "spec_new", "spec_open", "run_start", "run_pause", "run_set_param", "run_fork", "run_stop",
    "piece_update", "state_create", "state_set", "panel_create", "panel_patch",
    "program_create", "source_add", "layout_place", "layout_set", "control_create",
    "control_patch", "cursor_set", "selection_set", "intervene_set", "intervene_clear",
    "program_remove", "panel_remove",
}

# One-pass WGSL compute program producing a 10x10 row-normalized confusion matrix
# (row = true digit, column = predicted digit) directly from live probe tensors on GPU.
CONFUSION_WGSL = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(10)
fn confusion(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= 100u) { return; }
    let row = idx / 10u;
    let col = idx % 10u;
    let n = arrayLength(&labels);
    var row_total = 0u;
    var cell = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (labels[i] == i32(row)) {
            row_total = row_total + 1u;
            let base = i * 10u;
            var best = 0u;
            var best_val = logits[base];
            for (var c = 1u; c < 10u; c = c + 1u) {
                let v = logits[base + c];
                if (v > best_val) {
                    best_val = v;
                    best = c;
                }
            }
            if (best == col) {
                cell = cell + 1u;
            }
        }
    }
    out[idx] = select(0.0, f32(cell) / f32(row_total), row_total > 0u);
}
""".strip() + "\n"

# Candidate first-layer filters to inspect in priority order (conv1#3 has the largest L2 norm
# under the canonical seed=1 PyTorch MNIST CNN initialization).
CANDIDATE_FILTERS = (3, 15, 16, 9, 6, 21, 18, 0)
TRAIN_STEPS_BEFORE_PAUSE = 80


class Fail(Exception):
    pass


class Skip(Exception):
    pass


def find_built(name: str) -> str | None:
    """Locate a binary in the newest build directory containing both caliper and caliper-mcp."""
    folders = [
        f for f in list(ROOT.glob("cmake-build-*")) + list(ROOT.glob("build*"))
        if all((f / place).is_file() for place in PLACES.values())
    ]
    if not folders:
        return None
    newest = max(folders, key=lambda f: (f / PLACES["caliper"]).stat().st_mtime)
    return str(newest / PLACES[name])


class Mcp:
    def __init__(self, argv: list[str], env: dict[str, str]):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=env,
        )
        self.n = 0

    def request(self, method: str, params: dict) -> dict:
        self.n += 1
        body = dict(params, _meta=META)
        assert self.p.stdin is not None and self.p.stdout is not None
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": body}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise Fail("caliper-mcp closed its output")
            msg = json.loads(line)
            if msg.get("id") == self.n and "method" not in msg:
                return msg

    def call(self, tool: str, arguments: dict) -> dict:
        r = self.request("tools/call", {"name": tool, "arguments": arguments})
        if "result" not in r:
            raise Fail(f"{tool}: {r}")
        return r["result"]


def stop_proc(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def pids_of(script_path: Path) -> list[int]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if str(script_path) in cmd and int(pid) != os.getpid():
            found.append(int(pid))
    return found


def decode_png(data: bytes):
    import numpy as np

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise Fail("capture is not a PNG")
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data[pos + 8:pos + 21])
        elif tag == b"IDAT":
            idat.append(data[pos + 8:pos + 8 + n])
        pos += 12 + n
    assert hdr is not None
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        raise Fail(f"unsupported PNG format: {hdr}")
    bpp = 3 if ctype == 2 else 4
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(h):
        base = y * (stride + 1)
        filt = raw[base]
        cur = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=base + 1).copy()
        if filt == 1:
            cur = cur.reshape(w, bpp).cumsum(axis=0, dtype=np.uint8).reshape(-1)
        elif filt == 2:
            cur += prev
        elif filt in (3, 4):
            c, pv = cur.tolist(), prev.tolist()
            for i in range(stride):
                a = c[i - bpp] if i >= bpp else 0
                b = pv[i]
                if filt == 3:
                    c[i] = (c[i] + ((a + b) >> 1)) & 255
                else:
                    cc = pv[i - bpp] if i >= bpp else 0
                    pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
                    c[i] = (c[i] + (a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc))) & 255
            cur = np.array(c, dtype=np.uint8)
        out[y] = cur
        prev = cur
    return out.reshape(h, w, bpp)


def decode_f32(arr_obj: dict) -> list[float]:
    raw = base64.b64decode(arr_obj["base64"])
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class Session:
    def __init__(self, mcp: Mcp, approvals: Path, capture_dir: Path, transcript):
        self.mcp = mcp
        self.approvals = approvals
        self.capture_dir = capture_dir
        self.transcript = transcript
        self.cursor = 0
        self.events: list[dict] = []
        self.images: dict[str, object] = {}
        self.captures: dict[str, str] = {}
        self.step_no = 0

    def log_step(self, msg: str) -> None:
        self.step_no += 1
        print(f"step {self.step_no}: {msg}", flush=True)

    def call(self, tool: str, args: dict | None = None, *, allow_error: bool = False) -> dict:
        args = args or {}
        if tool in CARDS:
            with open(self.approvals, "a") as f:
                f.write(f"approve {tool.replace('_', '.', 1)} mcp\n")
        r = self.mcp.call(tool, args)
        self.transcript.write(json.dumps({"tool": tool, "args": args, "error": bool(r.get("isError"))}) + "\n")
        self.transcript.flush()
        if r.get("isError") and not allow_error:
            if "program.compiler_unavailable" in json.dumps(r):
                raise Skip("build has no WGSL compiler library (CALIPER_WGSL_FROM_SOURCE=ON)")
            raise Fail(f"{tool} failed: {json.dumps(r)[:800]}")
        return r.get("structuredContent", {}) if not r.get("isError") else r

    def pace(self, seconds: float = 0.5) -> None:
        b = self.call("events_wait", {"cursor": self.cursor, "timeout_s": seconds})
        self.cursor = b.get("cursor", self.cursor)
        self.events.extend(b.get("events", []))

    def until(self, what: str, pred, limit: float = 90.0):
        end = time.monotonic() + limit
        while True:
            val = pred()
            if val:
                return val
            if time.monotonic() > end:
                raise Fail(f"{what}: timed out after {limit:.0f} s")
            self.pace(0.5)

    def wait_event(self, what: str, match: dict, limit: float = 90.0) -> dict:
        def find():
            for ev in self.events:
                if all(ev.get(k) == v for k, v in match.items()):
                    return ev
            return None

        return self.until(what, find, limit)

    def runs(self) -> dict[str, dict]:
        return {r["id"]: r for r in self.call("run_list", {}).get("runs", [])}

    def run_step(self, run_id: str) -> int:
        return int(self.call("run_state", {"run": run_id}).get("step") or 0)

    def program_status(self, pid: str) -> dict | None:
        for item in self.call("program_status", {}).get("programs", []):
            if item.get("id") == pid:
                return item
        return None

    def read_confusion(self) -> tuple[str, list[list[float]], list[float]]:
        """Read the 10x10 confusion matrix from the heatmap panel and return (version, matrix, diag)."""
        body = self.call("panel_data", {"panel": "confusion"})
        data = body.get("data") or {}
        vals = data.get("values")
        if not vals or vals.get("shape") != [10, 10]:
            raise Fail(f"confusion panel data unexpected: {json.dumps(body)[:400]}")
        flat = decode_f32(vals)
        matrix = [flat[r * 10:(r + 1) * 10] for r in range(10)]
        diag = [matrix[d][d] for d in range(10)]
        version = str(body.get("version") or "")
        return version, matrix, diag

    def read_grid(self) -> tuple[int, dict, list[list]]:
        """Read image_grid@1 summary and tile rows via program_read."""
        s_body = self.call("program_read", {"id": "misclassified_grid", "resource": "summary", "count": 1})
        t_body = self.call("program_read", {"id": "misclassified_grid", "resource": "tiles", "count": 16})
        s_el = s_body["elements"][0]
        summary = {"total": int(s_el[0]), "errors": int(s_el[1]), "shown": int(s_el[2]), "page": int(s_el[3])}
        return int(s_body["step"]), summary, list(t_body["elements"])

    def step_n(self, run_id: str, count: int) -> int:
        """Advance a paused run by ``count`` steps and wait until it pauses again."""
        prev = self.run_step(run_id)
        target = prev + count
        self.call("run_pause", {"op": "step", "count": count, "run": run_id})
        self.until(
            f"step {count} past {prev}",
            lambda t=target: self.run_step(run_id) >= t and self.runs()[run_id]["state"] == "paused",
            90,
        )
        return self.run_step(run_id)

    def step_and_sync(self, run_id: str, *, apply_first: bool = True) -> int:
        """Step a paused run so pending changes apply to probe.logits and both GPU programs draw them.

        When ``apply_first`` is True (after an intervention or parameter change):
        - Step 1 unpauses at the checkpoint (applying pending interventions/params), runs
          ``optimizer.step()``, and calls ``publish_probe()`` which publishes updated ``probe.logits``
          before pausing at the next step boundary.
        - A bounded burst of ``every + 6`` steps then advances across the adaptive cadence window
          so both ``confusion`` and ``misclassified_grid`` run on the newly published ``probe.logits``
          and the viewer draws the updated resources.
        """
        if apply_first:
            self.step_n(run_id, 1)

        step_ready = self.run_step(run_id)

        def both_synced():
            c_st = self.program_status("confusion") or {}
            g_st = self.program_status("misclassified_grid") or {}
            g_draw = g_st.get("draw") or {}
            c_step = int(c_st.get("last_step") or 0)
            g_step = int(g_st.get("last_step") or 0)
            d_step = int(g_draw.get("resource_step") or 0)
            if c_step >= step_ready and g_step >= step_ready and d_step >= step_ready:
                p = self.call("panel_data", {"panel": "confusion"})
                if (p.get("data") or {}).get("values"):
                    return True
            return False

        for _ in range(4):
            c_st = self.program_status("confusion") or {}
            g_st = self.program_status("misclassified_grid") or {}
            every = max(int(c_st.get("every") or 1), int(g_st.get("every") or 1), 1)
            burst = min(max(every + 6, 4), 200)
            cur = self.step_n(run_id, burst)
            try:
                self.until(f"programs synced past step {step_ready}", both_synced, 6.0)
                self.pace(0.3)
                return cur
            except Fail:
                continue
        c_st = self.program_status("confusion")
        g_st = self.program_status("misclassified_grid")
        raise Fail(f"programs did not sync past step {step_ready}: confusion={c_st} grid={g_st}")

    def capture(self, name: str) -> str:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        r = self.mcp.call("capture_frame", {})
        if r.get("isError"):
            raise Fail(f"capture_frame failed: {json.dumps(r)[:400]}")
        data = base64.b64decode(next(c for c in r["content"] if c["type"] == "image")["data"])
        out = self.capture_dir / f"{name}.png"
        out.write_bytes(data)
        self.images[name] = decode_png(data)
        self.captures[name] = str(out)
        return str(out)

    def pixel_diff(self, a: str, b: str, tolerance: int = 8) -> int:
        import numpy as np

        xa = self.images[a][..., :3].astype(int)
        xb = self.images[b][..., :3].astype(int)
        if xa.shape != xb.shape:
            raise Fail(f"capture dimensions differ: {a}={xa.shape} vs {b}={xb.shape}")
        return int((np.abs(xa - xb).max(axis=2) > tolerance).sum())


def drive(s: Session, data_dir: Path, ws: Path, pids: set[int]) -> dict:
    result: dict = {}

    # 1. Start the canonical MNIST training script as a managed run and pause while building.
    s.log_step("create spec and start managed MNIST training run")
    s.call("spec_new", {"id": "mnist_agent", "title": "MNIST Agent Instrument"})
    started = s.call("run_start", {
        "script": "train.py",
        "args": [
            "--epochs", "14",
            "--probe-every", "1",
            "--probe-size", "512",
            "--seed", "1",
            "--pace", "0.015",
            "--data-dir", str(data_dir),
        ],
    })
    run_id = started["run"]["id"]
    s.until(
        "MNIST run running and attached",
        lambda: (lambda r: r and r["state"] == "running" and r.get("attached"))(s.runs().get(run_id)),
        120,
    )
    pids.update(pids_of(ws / "train.py"))
    s.until("first MNIST step", lambda: s.run_step(run_id) >= 1, 90)
    s.call("run_pause", {"op": "pause", "run": run_id})
    s.until("MNIST run paused", lambda: s.runs()[run_id]["state"] == "paused", 60)

    # 2. Verify library piece image_grid@1 and per-filter intervention points from caliper.watch.
    s.log_step("verify image_grid@1 library piece and conv1 filter intervention points")
    lib = s.call("piece_list", {"from": "library"})
    lib_pieces = {(p["name"], p["version"]) for p in lib.get("pieces", [])}
    if ("image_grid", 1) not in lib_pieces:
        raise Fail(f"library lacks image_grid@1: {sorted(lib_pieces)}")
    pts = s.call("intervene_points", {}).get("points", [])
    point_names = {p["point"] for p in pts}
    missing = [f"conv1#{c}" for c in range(32) if f"conv1#{c}" not in point_names]
    if missing:
        raise Fail(f"caliper.watch did not publish conv1 filter points: missing {missing[:8]}")
    result["conv1_points_count"] = 32

    # 3. Create the GPU confusion-matrix program and the misclassified-digits library piece.
    s.log_step("build confusion matrix GPU program and image_grid@1 misclassified digits panel")
    s.call("state_create", {
        "id": "page", "type": "integer", "default": 0, "min": 0, "max": 7, "title": "Error Page",
    })
    s.call("program_create", {
        "id": "confusion",
        "title": "Confusion Matrix",
        "wgsl": CONFUSION_WGSL,
        "inputs": [
            {"var": "logits", "source": "live:probe.logits"},
            {"var": "labels", "source": "live:probe.labels"},
        ],
        "output": {"kind": "tensor", "shape": [10, 10]},
    })
    s.call("panel_create", {
        "id": "misclassified",
        "primitive": "scene",
        "title": "Misclassified Digits",
        "bind": {"scene": "live:model"},
        "options": {"exclude": ["**"]},
    })
    s.call("program_create", {
        "id": "misclassified_grid",
        "title": "Misclassified Digits Grid",
        "piece": {"name": "image_grid", "version": 1, "from": "library"},
        "panel": "misclassified",
        "bind": {
            "images": "live:probe.images",
            "targets": "live:probe.labels",
            "logits": "live:probe.logits",
        },
        "params": {
            "only_errors": 1,
            "page": {"state": "page"},
            "cols": 4,
            "rows": 4,
            "scale": 1.0,
        },
    })
    s.call("panel_patch", {
        "id": "misclassified",
        "patch": {"bind": {"scene": None}, "options": {"exclude": None}},
    })

    # 4. Resume training so both GPU programs compile and MNIST trains past step 80.
    s.log_step(f"resume training until programs compile and step reaches {TRAIN_STEPS_BEFORE_PAUSE}")
    s.call("run_pause", {"op": "resume", "run": run_id})
    s.wait_event("confusion compiled", {"kind": "program", "program": "confusion", "change": "compiled"}, 90)
    s.wait_event(
        "misclassified_grid derive compiled",
        {"kind": "program", "program": "misclassified_grid", "change": "compiled", "half": "derive"},
        90,
    )
    s.wait_event(
        "misclassified_grid draw compiled",
        {"kind": "program", "program": "misclassified_grid", "change": "compiled", "half": "draw"},
        90,
    )
    s.until(
        "live:program.confusion in catalog",
        lambda: not s.call("catalog_describe", {"id": "live:program.confusion"}, allow_error=True).get("isError"),
        60,
    )
    s.until(
        "live:accuracy in catalog",
        lambda: not s.call("catalog_describe", {"id": "live:accuracy"}, allow_error=True).get("isError"),
        60,
    )

    # Wire the confusion heatmap panel, accuracy series panel, controls, and clean 1280x800 layout.
    s.log_step("add confusion heatmap, accuracy series, controls, and layout")
    s.call("source_add", {"id": "confusion_src", "catalog": "live:program.confusion"})
    s.call("source_add", {"id": "acc_raw", "catalog": "live:accuracy", "history": 4096})
    s.call("source_add", {"id": "acc_window", "from": "acc_raw", "transform": [{"op": "window", "steps": 4096}]})
    s.call("panel_create", {
        "id": "confusion",
        "primitive": "heatmap",
        "title": "Confusion Matrix (Row-Normalized)",
        "bind": {"values": "confusion_src"},
        "options": {"colormap": "viridis", "range": [0.0, 1.0], "x_label": "Predicted", "y_label": "True"},
    })
    s.call("panel_create", {
        "id": "accuracy",
        "primitive": "series",
        "title": "Training Batch Accuracy",
        "bind": {"x": "acc_window.step", "y": "acc_window.value"},
        "options": {"x_label": "Step", "y_label": "Accuracy"},
    })
    initial_point = f"conv1#{CANDIDATE_FILTERS[0]}"
    s.call("control_create", {
        "id": "ablate_filter",
        "kind": "toggle",
        "title": "First-Layer Filter",
        "bind": f"intervene:{initial_point}",
        "options": {"value": False},
    })
    s.call("control_create", {
        "id": "page_slider",
        "kind": "slider",
        "title": "Misclassified Page",
        "bind": "state:page",
        "options": {"min": 0, "max": 7, "step": 1},
    })
    s.call("layout_set", {
        "layout": {
            "root": "page",
            "nodes": [
                {"id": "page", "split": "vertical", "children": ["top", "bottom"], "ratio": 0.72},
                {"id": "top", "split": "horizontal", "children": ["confusion", "misclassified"], "ratio": 0.44},
                {"id": "bottom", "split": "horizontal", "children": ["accuracy", "controls"], "ratio": 0.65},
                {"id": "controls", "split": "vertical", "children": ["ablate_filter", "page_slider"], "ratio": 0.52},
            ],
        }
    })

    s.until(f"training reaches step {TRAIN_STEPS_BEFORE_PAUSE}", lambda: s.run_step(run_id) >= TRAIN_STEPS_BEFORE_PAUSE, 120)
    s.call("run_pause", {"op": "pause", "run": run_id})
    s.until("run paused after training", lambda: s.runs()[run_id]["state"] == "paused", 60)

    # Lower lr to the watch minimum (0.001) while paused so single-stepping holds trained weights
    # steady while comparing baseline, ablated, and restored probe evaluations.
    s.call("run_set_param", {"run": run_id, "name": "lr", "value": 0.001})
    base_step = s.step_and_sync(run_id, apply_first=True)

    # 5. Check pagination on misclassified digits first, then return to page 0 and take the
    # baseline confusion matrix, misclassified page, and before-intervention capture right before
    # applying the intervention so no extra training steps sit between baseline and ablation.
    s.log_step("check error pagination, read baseline confusion matrix and grid, and capture before-intervention")
    _, init_summary, init_tiles = s.read_grid()
    pagination_checked = False
    if init_summary["errors"] > 16:
        s.call("state_set", {"id": "page", "value": 1})
        s.step_and_sync(run_id, apply_first=False)
        _, p1_summary, p1_tiles = s.read_grid()
        if p1_summary["page"] != 1 or p1_summary["shown"] <= 0:
            raise Fail(f"pagination to page 1 failed: {p1_summary}")
        p0_ids = {int(t[8]) for t in init_tiles}
        p1_ids = {int(t[8]) for t in p1_tiles}
        if p0_ids & p1_ids:
            raise Fail(f"page 0 and page 1 share sample indices: {p0_ids & p1_ids}")
        s.call("state_set", {"id": "page", "value": 0})
        base_step = s.step_and_sync(run_id, apply_first=False)
        pagination_checked = True

    conf_ver, base_mat, base_diag = s.read_confusion()
    for d in range(10):
        row_sum = sum(base_mat[d])
        if abs(row_sum - 1.0) > 1e-3:
            raise Fail(f"baseline confusion matrix row {d} sums to {row_sum:.4f}, expected 1.0")
    grid_step, base_summary, base_tiles = s.read_grid()
    if base_summary["total"] != 512 or base_summary["errors"] <= 0 or base_summary["shown"] <= 0:
        raise Fail(f"unexpected baseline image_grid summary: {base_summary}")
    for tile in base_tiles:
        # Flattened Tile layout: rect(4 f32), uv_rect(4 f32), sample_index(u32), expected(i32), predicted(i32), wrong(u32)
        expected_d, pred_d, wrong = int(tile[9]), int(tile[10]), int(tile[11])
        if wrong != 1 or expected_d == pred_d:
            raise Fail(f"misclassified page contains non-error tile: {tile}")

    before_path = s.capture("before-intervention")
    h_px, w_px = s.images["before-intervention"].shape[:2]
    if w_px < 1280 or h_px < 800:
        raise Fail(f"before capture size is {w_px}x{h_px}, expected at least 1280x800")

    # 6. Apply a first-layer filter ablation through caliper.watch intervention points.
    s.log_step("switch off one first-layer conv1 filter and inspect changed per-digit accuracy")
    chosen_point = None
    after_diag = None
    after_summary = None
    after_tiles = None
    after_step = None
    for idx in CANDIDATE_FILTERS:
        pt = f"conv1#{idx}"
        if pt != initial_point:
            s.call("control_patch", {
                "id": "ablate_filter",
                "patch": {"title": "First-Layer Filter", "bind": f"intervene:{pt}"},
            })
        s.call("intervene_set", {"point": pt, "op": "ablate"})
        # Step 1 activates the hook at checkpoint and publishes ablated probe.logits at end of step 1;
        # next pass executes confusion and misclassified_grid on the ablated probe.logits.
        after_step = s.step_and_sync(run_id, apply_first=True)
        active = s.call("intervene_list", {}).get("interventions", [])
        if not any(iv["point"] == pt and iv["state"] == "active" for iv in active):
            raise Fail(f"intervention on {pt} is not active: {active}")
        _, after_mat, cand_diag = s.read_confusion()
        _, cand_summary, cand_tiles = s.read_grid()
        deltas = [cand_diag[d] - base_diag[d] for d in range(10)]
        if any(abs(delta) > 1e-6 for delta in deltas):
            chosen_point = pt
            after_diag = cand_diag
            after_summary = cand_summary
            after_tiles = cand_tiles
            break
        s.call("intervene_clear", {"point": pt})
        s.step_and_sync(run_id, apply_first=True)

    if chosen_point is None or after_diag is None or after_summary is None:
        raise Fail("no candidate first-layer conv1 filter changed per-digit accuracy on probe set")

    after_path = s.capture("after-intervention")
    diff_pixels = s.pixel_diff("before-intervention", "after-intervention")
    if diff_pixels <= 0:
        raise Fail("before-intervention and after-intervention captures are identical")

    # 7. Clear the intervention and verify causal recovery toward baseline per-digit accuracy.
    s.log_step(f"clear intervention on {chosen_point} and verify recovery")
    s.call("intervene_clear", {"point": chosen_point})
    restored_step = s.step_and_sync(run_id, apply_first=True)
    remaining = [iv for iv in s.call("intervene_list", {}).get("interventions", []) if iv.get("state") == "active"]
    if remaining:
        raise Fail(f"intervention remained active after clear: {remaining}")
    _, _, restored_diag = s.read_confusion()
    _, restored_summary, _ = s.read_grid()

    deltas_during = [round(after_diag[d] - base_diag[d], 4) for d in range(10)]
    deltas_restored = [round(restored_diag[d] - base_diag[d], 4) for d in range(10)]
    deltas_clear = [round(after_diag[d] - restored_diag[d], 4) for d in range(10)]
    max_during = max(abs(x) for x in deltas_during)
    max_restored = max(abs(x) for x in deltas_restored)
    l1_during = sum(abs(x) for x in deltas_during)
    l1_restored = sum(abs(x) for x in deltas_restored)
    if not any(abs(x) > 1e-6 for x in deltas_clear) or l1_restored >= l1_during:
        raise Fail(
            f"per-digit accuracy did not recover after clearing {chosen_point}: "
            f"l1_during={l1_during:.4f} l1_restored={l1_restored:.4f} "
            f"during={deltas_during} restored={deltas_restored}"
        )

    # 8. Stop the managed run cleanly.
    s.log_step("stop managed run")
    s.call("run_stop", {"run": run_id})
    s.until("MNIST run exited", lambda: s.runs()[run_id]["state"] == "exited", 90)

    result.update({
        "run_id": run_id,
        "point": chosen_point,
        "steps": {"baseline": base_step, "confusion_ver": conf_ver, "grid": grid_step,
                  "after": after_step, "restored": restored_step},
        "pagination_checked": pagination_checked,
        "baseline": {
            "per_digit_acc": [round(x, 4) for x in base_diag],
            "summary": base_summary,
            "first_tiles": [[int(t[8]), int(t[9]), int(t[10])] for t in base_tiles[:8]],
        },
        "after_intervention": {
            "per_digit_acc": [round(x, 4) for x in after_diag],
            "deltas": deltas_during,
            "max_abs_delta": round(max_during, 4),
            "changed_digits": [d for d in range(10) if abs(deltas_during[d]) > 0],
            "summary": after_summary,
            "first_tiles": [[int(t[8]), int(t[9]), int(t[10])] for t in (after_tiles or [])[:8]],
        },
        "restored": {
            "per_digit_acc": [round(x, 4) for x in restored_diag],
            "deltas_vs_baseline": deltas_restored,
            "max_abs_delta": round(max_restored, 4),
            "summary": restored_summary,
        },
        "captures": {
            "before": before_path,
            "after": after_path,
            "size": [w_px, h_px],
            "diff_pixels": diff_pixels,
        },
    })
    return result


def run_all(a: argparse.Namespace) -> dict:
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Fail("over the time limit")))
    signal.alarm(a.limit_s)
    for exe in (a.app, a.mcp):
        if not exe or not Path(exe).is_file():
            raise Skip(f"{exe or 'caliper / caliper-mcp'} is not built")
    if not TRAIN_SCRIPT.is_file():
        raise Fail(f"missing training script: {TRAIN_SCRIPT}")

    env = {
        k: v for k, v in os.environ.items()
        if k not in ("CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_APPLET_PATH")
        and not k.startswith("CALIPER_RUN_")
    }
    env["PYTHONPATH"] = str(ROOT / "python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    pre = subprocess.run(
        [a.python, "-c", "import numpy, torch, torchvision, caliper, caliper._netio, caliper._producer"],
        env=env, capture_output=True, text=True,
    )
    if pre.returncode != 0:
        why = pre.stderr.strip().splitlines()[-1] if pre.stderr.strip() else "import failed"
        raise Skip(f"torch, torchvision, numpy, caliper._netio, or caliper._producer is missing: {why}")
    gpu = subprocess.run(
        [a.python, "-c", "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 3)"],
        env=env, capture_output=True, text=True,
    )
    if gpu.returncode != 0:
        raise Skip("this machine has no Metal device for torch")

    work = Path(a.work).resolve()
    if work.exists():
        subprocess.run(["rm", "-rf", str(work)], check=True)
    (work / "data").mkdir(parents=True)
    (work / "user").mkdir()
    ws = work / "workspace"
    ws.mkdir()
    (ws / "train.py").write_text(TRAIN_SCRIPT.read_text())

    data_dir = Path(a.data_dir).expanduser().resolve()
    env.update({
        "CALIPER_AGENT_OFFLINE": "1",
        "CALIPER_HIDDEN": "1",
        "CALIPER_DATA_DIR": str(work / "data"),
        "CALIPER_USER_APPLETS_DIR": str(work / "user"),
        "CALIPER_WINDOW_SIZE": "1280x800",
        "CALIPER_WORKSPACE": str(ws),
        "CALIPER_RUN_PYTHON": a.python,
        "CALIPER_MNIST_DATA_DIR": str(data_dir),
    })
    approvals = work / "approvals.txt"
    approvals.write_text("")

    procs: list[subprocess.Popen] = []
    own_pids: set[int] = set()
    child_pids: set[int] = set()
    transcript = open(work / "transcript.jsonl", "w")
    alog = open(work / "app.log", "w")
    try:
        app = subprocess.Popen(
            [a.app, "--hidden", "--api", "--api-port", "0"],
            env=dict(env, CALIPER_TEST_APPROVALS=str(approvals)),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        procs.append(app)
        own_pids.add(app.pid)
        lines: list[str] = []

        def drain():
            assert app.stdout is not None
            for raw in app.stdout:
                alog.write(raw)
                alog.flush()
                lines.append(raw.rstrip("\n"))

        threading.Thread(target=drain, daemon=True).start()
        api_file = work / "data" / "api.json"
        deadline = time.monotonic() + 90
        while not api_file.exists():
            for line in list(lines):
                m = re.match(r"\[renderer\] (\S+)", line)
                if m and m.group(1) != "metal":
                    raise Skip(f"the renderer is {m.group(1)}, not Metal")
            if app.poll() is not None:
                raise Fail("caliper exited before writing api.json: " + str(lines[-8:]))
            if time.monotonic() > deadline:
                raise Fail("caliper did not write api.json within 90 s")
            time.sleep(0.05)

        mcp = Mcp([a.mcp, "--api-file", str(api_file)], env)
        procs.append(mcp.p)
        own_pids.add(mcp.p.pid)
        mcp.request("server/discover", {})

        sess = Session(mcp, approvals, Path(a.capture_dir).resolve(), transcript)
        return drive(sess, data_dir, ws, child_pids)
    finally:
        signal.alarm(0)
        transcript.close()
        for p in reversed(procs):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
            stop_proc(p)
        alog.close()
        left = [pid for pid in sorted(child_pids | own_pids) if alive(pid)]
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if left and sys.exc_info()[0] is None:
            raise Fail(f"child processes left alive: {left}")


def build_parser() -> argparse.ArgumentParser:
    app = find_built("caliper")
    default_dir = str(Path(app).parent / "mnist-agent-instrument") if app else "mnist-agent-instrument"
    ap = argparse.ArgumentParser(description="Build the MNIST agent instrument and run a filter intervention via caliper-mcp.")
    ap.add_argument("capture_dir", nargs="?", default=default_dir,
                    help="folder that receives before-intervention.png and after-intervention.png")
    ap.add_argument("--app", default=app, help="path to the caliper executable")
    ap.add_argument("--mcp", default=find_built("caliper-mcp"), help="path to the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="Python executable with torch and torchvision")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="MNIST dataset cache folder")
    ap.add_argument("--work", default=None, help="scratch work directory (default: <capture_dir>/work)")
    ap.add_argument("--limit-s", type=int, default=300, help="overall timeout in seconds")
    ap.add_argument("--ctest", action="store_true",
                    help="emit CTest pass/skip lines and exit 77 when skipped")
    ap.add_argument("--keep-work", action="store_true", help="keep the work directory after exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    owns_work = a.work is None
    if a.work is None:
        a.work = str(Path(a.capture_dir) / "work")
    try:
        res = run_all(a)
    except Skip as e:
        if a.ctest:
            print(f"caliper_mnist_agent_acceptance skipped: {e}", flush=True)
            return 77
        print(f"Cannot run here: {e}", file=sys.stderr)
        return 2
    except Fail as e:
        if a.ctest:
            print(f"caliper_mnist_agent_acceptance FAILED: {e}", flush=True)
        else:
            print(f"Failed: {e}", file=sys.stderr)
        return 1
    finally:
        if owns_work and not a.keep_work:
            subprocess.run(["rm", "-rf", str(a.work)], check=False)

    print(json.dumps(res, indent=2), flush=True)
    if a.ctest:
        print("caliper_mnist_agent_acceptance passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
