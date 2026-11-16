"""A synthetic producer for the browser tests.

It serves web/dist and a wire protocol v3 stream on 127.0.0.1 with a random
session token, using the protocol's own Python codec (_wire.py) and the
websockets package, and makes the data of a small training run: a decaying
loss, a drifting weight matrix, attention maps with layers and heads, an
embedding whose clusters separate, a histogram of the weights and an input
image. It computes every data view a viewer asks for with numpy, as a real
producer would on its device.

    python mock_server.py --wire path/to/_wire.py [--port 0] [--dist web/dist] [--wasm DIR]

With --wasm it also serves Caliper's WebAssembly views from DIR (the three
files caliper-views-loader.js, caliper-views.js and caliper-views.wasm), as
caliper.serve() does from the package.

With --scene clusters or --scene depth the embedding is a fixed 3-D scene
instead of the moving clusters (scene() describes both), for the tests and
screenshots of the 3-D points views.

With --fixed it serves instead the fixed data of the element's view tests
(web/test/views.spec.ts), sent once per subscription: two 4 x 8 heatmaps, one
with the identity axes and one pooled from query and key dims, four 2-D
points at the corners of the unit square sampled with stride 3, a series of
the values 0 to 999, and a histogram of ten bins over [-5, 5].

Wire protocol 3.1 (--controls): the producer announces the capabilities
controls and selection. Its run calls a checkpoint at every step, with the
run state, four parameters (lr, a number on a log scale; batch, an integer;
augment, a boolean; optimizer, a choice) and an action, "save checkpoint".
Viewers that announce selection get item spaces in VIEWS (the embedding's
samples and the loss's steps; in --fixed mode the points' and the heatmap
h's rows, space "test", and the series' steps) and a SELECTION, with item
details for the sample spaces, for each select event that names a space.
--read-only refuses every CONTROL (WELCOME's control false, replies
forbidden), --no-checkpoint keeps the run state none, and --v30 makes it a
3.0 producer: frames at minor 0 and none of the 3.1 additions.

It prints the viewer page's URL on the first line, then one JSON line for
each EVENT and each CONTROL a viewer sends.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import mimetypes
import os
import re
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

W = None  # the _wire module

WASM_FILES = ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")


def load_wire(path: str):
    spec = importlib.util.spec_from_file_location("_wire", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wire"] = mod  # dataclasses resolve their annotations through it
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the simulated run
# ---------------------------------------------------------------------------
def scene(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 3-D points for the embedding, and their labels. "clusters": eight
    classes in clusters at different depths, placed by golden-ratio sequences
    (tests/test_live_views.cpp, cluster_scene, makes the same points). "depth":
    eight anchors at the corners of [-1, 1]^3 (label 2), one point in front of
    the centre as the default camera sees it (label 1) and one behind it
    (label 3)."""
    if name == "depth":
        corners = [[1.0 if i & 1 else -1.0, 1.0 if i & 2 else -1.0, 1.0 if i & 4 else -1.0] for i in range(8)]
        pos = np.array(corners + [[0.45, 0.45, 0.45], [-0.45, -0.45, -0.45]], np.float32)
        return pos, np.array([2.0] * 8 + [1.0, 3.0], np.float32)
    n = 3000
    pos = np.empty((n, 3), np.float32)
    frac = lambda x: x - math.floor(x)
    for i in range(n):
        k, j = i % 8, i // 8
        a = 2.0 * math.pi * k / 8.0
        cx, cy, cz = 4.0 * math.cos(a), 2.2 * ((k % 3) - 1), 4.0 * math.sin(a)
        u = frac((j + 0.37 * k) * 0.6180339887498949)
        v = frac((j + 0.53 * k) * 0.7548776662466927)
        w = frac((j + 0.29 * k) * 0.5698402909980532)
        r, th, ph = 0.95 * u ** (1.0 / 3.0), 2.0 * math.pi * v, math.acos(1.0 - 2.0 * w)
        pos[i] = (cx + r * math.sin(ph) * math.cos(th), cy + r * math.cos(ph), cz + r * math.sin(ph) * math.sin(th))
    return pos, (np.arange(n) % 8).astype(np.float32)


class Run:
    def __init__(self, points: int, dims: int, seed: int, scene_name: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.loss: list[float] = []
        self.weights = (self.rng.standard_normal((48, 64)) * 0.5).astype(np.float32)
        self.scene = scene(scene_name) if scene_name else None
        if self.scene is not None:
            points, dims = len(self.scene[0]), 3
        self.n, self.dims = points, dims
        self.labels = (np.arange(points) % 8).astype(np.float32) if self.scene is None else self.scene[1]
        ang = 2 * np.pi * np.arange(8) / 8
        self.centers = np.stack([np.cos(ang) * 4, np.sin(ang) * 4, np.where(np.arange(8) % 2, 2.0, -2.0)], 1)
        self.offsets = self.rng.standard_normal((points, 3)).astype(np.float32)
        for _ in range(40):
            self.advance()

    def advance(self):
        self.step += 1
        t = self.step
        for _ in range(4):
            i = len(self.loss)
            base = 0.08 + 2.2 * math.exp(-i / 600) + 0.3 * math.exp(-i / 60)
            self.loss.append(float(base * (1 + 0.12 * self.rng.standard_normal())))
        y, x = np.mgrid[0:48, 0:64]
        target = 0.8 * np.sin(x / 7 + t / 40) * np.cos(y / 5 - t / 60)
        self.weights += (0.04 * (target - self.weights) + 0.02 * self.rng.standard_normal(self.weights.shape)).astype(np.float32)

    def source(self, view: str, name: str) -> np.ndarray:
        t = self.step
        if view in ("weights", "weight_hist"):
            return self.weights
        if view == "attention":
            q = np.arange(32)[:, None]
            k = np.arange(32)[None, :]
            out = np.empty((4, 8, 32, 32), np.float32)
            for layer in range(4):
                for h in range(8):
                    width = 1.5 + h * 0.8 + layer
                    d = k - q - ((h % 3) - 1) + math.sin(t / 30 + h) * 2
                    s = -(d * d) / (2 * width * width) + np.where(k == 0, layer * 0.7, 0)
                    out[layer, h] = s if name == "scores" else np.exp(s) / np.exp(s).sum(1, keepdims=True)
            return out
        if view == "embedding" and self.scene is not None:
            return np.stack([self.scene[0]] * 3)
        if view == "embedding":
            p = 1 - math.exp(-t / 150)
            out = np.empty((3, self.n, self.dims), np.float32)
            for layer in range(3):
                sep = p * (0.4 + 0.3 * layer)
                c = self.centers[self.labels.astype(int)][:, :self.dims]
                out[layer] = c * sep + self.offsets[:, :self.dims] * (1.2 - 0.5 * sep)
            return out
        if view == "input":
            y, x = np.mgrid[0:32, 0:32]
            r = np.hypot(x - 16 - 6 * math.cos(t / 10), y - 16 - 6 * math.sin(t / 10))
            img = np.stack([np.clip(255 - r * 18, 0, 255), (x * 8) % 256, (y * 8) % 256, np.full_like(x, 255)], -1)
            return img.astype(np.uint8)
        raise KeyError(view)


def views(run: Run) -> list[dict]:
    K = W.Kind
    return [
        {"id": 1, "name": "loss", "kind": int(K.SERIES), "sources": []},
        {"id": 2, "name": "weights", "kind": int(K.HEATMAP),
         "sources": [{"name": "fc1.weight", "dims": [{"name": "out", "size": 48}, {"name": "in", "size": 64}], "dtype": "f32"}]},
        {"id": 3, "name": "attention", "kind": int(K.HEATMAP),
         "sources": [{"name": n, "dtype": "f32", "dims": [{"name": "layer", "size": 4}, {"name": "head", "size": 8},
                                                       {"name": "query", "size": 32}, {"name": "key", "size": 32}]}
                     for n in ("attn", "scores")],
         "default": {"source": "attn", "index": {"layer": 0, "head": 0}}},
        {"id": 4, "name": "embedding", "kind": int(K.POINTS),
         "sources": [{"name": "hidden", "dtype": "f32", "dims": [{"name": "layer", "size": 3}, {"name": "sample", "size": run.n}]}],
         "default": {"index": {"layer": 2}}},
        {"id": 5, "name": "weight_hist", "kind": int(K.HISTOGRAM),
         "sources": [{"name": "fc1.weight", "dims": [{"name": "out", "size": 48}, {"name": "in", "size": 64}], "dtype": "f32"}]},
        {"id": 6, "name": "input", "kind": int(K.IMAGE),
         "sources": [{"name": "batch", "dims": [{"name": "height", "size": 32}, {"name": "width", "size": 32}], "dtype": "u8"}]},
    ]


# ---------------------------------------------------------------------------
# the fixed data (--fixed)
# ---------------------------------------------------------------------------
HEAT = [float("nan") if i == 31 else float(i % 31) for i in range(32)]   # 4 x 8


def fixed_views() -> list[dict]:
    K = W.Kind
    return [
        {"id": 1, "name": "h", "kind": int(K.HEATMAP),
         "sources": [{"name": "h", "dims": [{"name": "row", "size": 4}, {"name": "col", "size": 8}]}]},
        {"id": 2, "name": "ha", "kind": int(K.HEATMAP),
         "sources": [{"name": "ha", "dims": [{"name": "query", "size": 64}, {"name": "key", "size": 41}]}]},
        {"id": 3, "name": "p", "kind": int(K.POINTS),
         "sources": [{"name": "p", "dims": [{"name": "sample", "size": 10}]}]},
        {"id": 4, "name": "s", "kind": int(K.SERIES), "sources": []},
        {"id": 5, "name": "hist", "kind": int(K.HISTOGRAM),
         "sources": [{"name": "hist", "dims": [{"name": "value", "size": 91}]}]},
    ]


def fixed_snapshot(view: dict, sub: int, seq: int, dv: dict):
    K = W.Kind
    name, t = view["name"], time.monotonic_ns()
    heat = np.asarray(HEAT, np.float32).reshape(4, 8)
    if name == "h":
        axes = [{"dim": "row", "start": 0, "step": 1, "count": 4, "size": 4},
                {"dim": "col", "start": 0, "step": 1, "count": 8, "size": 8}]
        return W.Snapshot(sub, name, seq, t, K.HEATMAP, dv, heat, vmin=0.0, vmax=31.0, axes=axes)
    if name == "ha":
        # Pooled from query rows 0..63 and key columns 10, 12, ..., 40.
        axes = [{"dim": "query", "start": 0, "step": 1, "count": 64, "size": 4},
                {"dim": "key", "start": 10, "step": 2, "count": 16, "size": 8}]
        return W.Snapshot(sub, name, seq, t, K.HEATMAP, dv, heat, vmin=0.0, vmax=31.0, axes=axes)
    if name == "p":
        data = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
        axes = [{"dim": "sample", "start": 0, "step": 3, "count": 4, "size": 4}]
        return W.Snapshot(sub, name, seq, t, K.POINTS, dv, data, aux=np.arange(4, dtype=np.float32),
                          stats=[0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], axes=axes, points_dims=2, label_count=4)
    counts = np.asarray([1, 2, 4, 8, 16, 32, 16, 8, 4, 0], np.float32)
    return W.Snapshot(sub, name, seq, t, K.HISTOGRAM, dv, counts, stats=[-5.0, 5.0])


FIXED_SERIES = np.arange(1000, dtype=np.float32)

# Item spaces of 3.1 (for viewers that announce selection): view -> dim -> space,
# and the series views' spaces.
ITEMS = {"fixed": ({"p": {"sample": "test"}, "h": {"row": "test"}}, {"s": "step"}),
         "run": ({"embedding": {"sample": "sample"}}, {"loss": "step"})}


def with_items(views: list[dict], fixed: bool) -> list[dict]:
    """The views with the item spaces of their dims and series."""
    dims, series = ITEMS["fixed" if fixed else "run"]
    out = []
    for v in views:
        v = json.loads(json.dumps(v))
        if v["name"] in series:
            v["items"] = series[v["name"]]
        for s in v["sources"]:
            for d in s["dims"]:
                space = dims.get(v["name"], {}).get(d["name"])
                if space:
                    d["items"] = space
        out.append(v)
    return out


def item_details(space: str, items: list[int]) -> dict | None:
    """The details of the first selected items of a sample space."""
    if space not in ("test", "sample"):
        return None
    first = items[:W.MAX_DETAIL_ITEMS]
    return {"fields": ["label", "score"], "items": first,
            "values": [[i % 8, round(math.cos(i) * 0.5 + 0.5, 3)] for i in first]}


# ---------------------------------------------------------------------------
# controls (3.1)
# ---------------------------------------------------------------------------
class Controls:
    """The run state, parameters and actions of the simulated script, with the
    producer's rules of wire protocol 3.1 ("Controls")."""

    def __init__(self, broadcast):
        self.broadcast = broadcast                 # async () -> None: CONTROLS to every viewer
        self.state, self.step, self.steps_left = W.RUN_NONE, None, None
        self.params = [
            {"name": "lr", "type": "number", "value": 1e-3, "min": 1e-5, "max": 0.1, "log": True,
             "help": "Learning rate"},
            {"name": "batch", "type": "integer", "value": 64, "min": 8, "max": 512, "help": "Batch size"},
            {"name": "augment", "type": "boolean", "value": True, "help": "Augment the inputs"},
            {"name": "optimizer", "type": "choice", "value": "adam", "choices": ["adam", "sgd", "lion"]},
        ]
        self.actions = [{"name": "save checkpoint", "running": False, "queued": False,
                         "help": "Write the weights to disk"}]
        self.waiting: dict[str, list] = {}         # action -> [(send, request_id)]
        self.queue: list[str] = []

    def message(self):
        run = {"state": self.state}
        if self.step is not None:
            run["step"] = self.step
        if self.state == W.RUN_STEPPING:
            run["steps_left"] = self.steps_left
        return W.Controls(run, [dict(p) for p in self.params], [dict(a) for a in self.actions])

    async def handle(self, msg, send, allowed: bool):
        async def reply(status, **kw):
            await send(W.ControlReply(msg.request_id, msg.op, status, **kw))

        if not allowed:
            return await reply(W.STATUS_ERROR, error={"code": W.ERR_FORBIDDEN, "message": "this producer refuses controls"})
        op = msg.op
        if op in (W.OP_PAUSE, W.OP_STEP) and self.state == W.RUN_NONE:
            return await reply(W.STATUS_ERROR, error={"code": W.ERR_UNSUPPORTED, "message": "the script has no checkpoint"})
        if op == W.OP_PAUSE:
            if self.state in (W.RUN_RUNNING, W.RUN_STEPPING):
                self.state = W.RUN_PAUSING
        elif op == W.OP_RESUME:
            if self.state != W.RUN_NONE:
                self.state = W.RUN_RUNNING
        elif op == W.OP_STEP:
            self.state, self.steps_left = W.RUN_STEPPING, msg.count or 1
        elif op == W.OP_SET:
            p = next((p for p in self.params if p["name"] == msg.name), None)
            if p is None:
                return await reply(W.STATUS_ERROR, name=msg.name, error={"code": W.ERR_UNKNOWN_CONTROL, "message": msg.name})
            try:
                v = W.param_value(p, msg.value)
            except W.MessageError as e:
                return await reply(W.STATUS_ERROR, name=msg.name, error={"code": e.code, "message": e.message})
            if self.state == W.RUN_NONE:
                p["value"] = v
            else:
                p["pending"] = v
            await reply(W.STATUS_OK, name=msg.name, value=v)
            return await self.broadcast()
        elif op == W.OP_RUN:
            a = next((a for a in self.actions if a["name"] == msg.name), None)
            if a is None:
                return await reply(W.STATUS_ERROR, name=msg.name, error={"code": W.ERR_UNKNOWN_CONTROL, "message": msg.name})
            self.waiting.setdefault(a["name"], []).append((send, msg.request_id))
            if not a["queued"]:
                a["queued"] = True
                self.queue.append(a["name"])
            await reply(W.STATUS_QUEUED, name=msg.name)
            return await self.broadcast()
        await reply(W.STATUS_OK)
        await self.broadcast()

    async def checkpoint(self, step: int) -> bool:
        """The script's checkpoint at step: apply pending values, run queued
        actions, and hold the loop while paused. False: skip this step."""
        changed = self.state == W.RUN_NONE
        if self.state == W.RUN_NONE:
            self.state = W.RUN_RUNNING
        for p in self.params:
            if "pending" in p:
                p["value"] = p.pop("pending")
                changed = True
        while self.queue:
            name = self.queue.pop(0)
            a = next(a for a in self.actions if a["name"] == name)
            a["queued"], a["running"] = False, True
            await self.broadcast()
            await asyncio.sleep(0.2)
            a["running"] = False
            for send, rid in self.waiting.pop(name, []):
                await send(W.ControlReply(rid, W.OP_RUN, W.STATUS_OK, name=name, message=f"saved step {step}"))
            changed = True
        if self.state == W.RUN_PAUSING:
            self.state, self.step, changed = W.RUN_PAUSED, step, True
        elif self.state == W.RUN_STEPPING:
            if self.steps_left <= 0:
                self.state, self.step, self.steps_left, changed = W.RUN_PAUSED, step, None, True
            else:
                self.steps_left -= 1
                changed = True
        if changed:
            await self.broadcast()
        return self.state != W.RUN_PAUSED


# ---------------------------------------------------------------------------
# data views, computed with numpy
# ---------------------------------------------------------------------------
REDUCE = {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min,
          "norm": lambda a, axis: np.sqrt(np.sum(a * a, axis=axis))}


def apply(view: dict, dv: dict, x: np.ndarray, point_dims: int = 0):
    """index, then reduce in dim order, then the tile mosaic and the points budget."""
    src = next(s for s in view["sources"] if s["name"] == dv["source"])
    names = [d["name"] for d in src["dims"]]
    sl = []
    for n in names:
        sel = dv["index"].get(n)
        sl.append(sel if isinstance(sel, int) else slice(*sel) if sel is not None else slice(None))
    if point_dims:
        sl.append(slice(None))
    x = x[tuple(sl)]
    kept = [n for n in names if not isinstance(dv["index"].get(n), int)]
    for n in [n for n in names if n in dv["reduce"]]:
        ax = kept.index(n)
        x = REDUCE[dv["reduce"][n]](x, axis=ax)
        kept.pop(ax)
    layout = W.payload_layout(view, dv)
    if dv.get("tile"):
        c, h, w = x.shape
        t = layout["tile"]
        m = np.full((t["rows"] * h, t["cols"] * w), np.nan, np.float32)
        for k in range(c):
            r, q = divmod(k, t["cols"])
            m[r * h:(r + 1) * h, q * w:(q + 1) * w] = x[k]
        x = m
    if view["kind"] == W.Kind.POINTS:
        a = layout["axes"][0]
        idx = (np.arange(a["size"]) * a["count"]) // a["size"]
        x = x[idx]
    return np.ascontiguousarray(x), layout["axes"], layout["tile"]


def snapshot(run: Run, view: dict, sub: int, seq: int, dv: dict):
    K = W.Kind
    kind = view["kind"]
    if kind == K.IMAGE:
        data, axes, _ = apply(view, dv, run.source("input", "batch"), point_dims=1)
        return W.Snapshot(sub, view["name"], seq, time.monotonic_ns(), kind, dv, data, axes=axes)
    x = run.source(view["name"], dv.get("source", ""))
    if kind == K.POINTS:
        data, axes, _ = apply(view, dv, x, point_dims=1)
        a = axes[0]
        aux = run.labels[a["start"] + ((np.arange(a["size"]) * a["count"]) // a["size"]) * a["step"]]
        bb = [float(v) for d in range(3) for v in ((data[:, d].min(), data[:, d].max()) if d < run.dims else (0.0, 0.0))]
        return W.Snapshot(sub, view["name"], seq, time.monotonic_ns(), kind, dv, data, aux=aux,
                          stats=[0.0, 7.0] + bb, axes=axes, points_dims=run.dims, label_count=8)
    data, axes, tile = apply(view, dv, x)
    if kind == K.HISTOGRAM:
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        counts, _ = np.histogram(data, bins=64, range=(lo, hi))
        return W.Snapshot(sub, view["name"], seq, time.monotonic_ns(), kind, dv, counts.astype(np.float32), stats=[lo, hi])
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if view["name"] == "weights":
        m = max(abs(lo), abs(hi))
        return W.Snapshot(sub, view["name"], seq, time.monotonic_ns(), kind, dv, data, colormap=2, vmin=-m, vmax=m,
                          stats=[lo, hi], axes=axes)
    return W.Snapshot(sub, view["name"], seq, time.monotonic_ns(), kind, dv, data, colormap=0,
                      flags=W.SNAP_AUTO_VMIN | W.SNAP_AUTO_VMAX, stats=[lo, hi], axes=axes, tile=tile)


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------
ORIGIN = re.compile(r"^(https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?|vscode-webview://[^/]+|null)$", re.I)


class Producer:
    def __init__(self, run: Run, token: str, dist: Path, session_id: str, wasm: Path | None = None,
                 fixed: bool = False, controls: bool = False, read_only: bool = False, v30: bool = False):
        self.run, self.token, self.dist, self.wasm, self.fixed = run, token, dist, wasm, fixed
        self.v30 = v30
        self.has_controls = controls and not v30
        self.read_only = read_only
        self.conns: dict = {}                      # send -> set of the viewer's capabilities
        self.ctl = Controls(self.broadcast_controls) if self.has_controls else None
        self.selections: dict[str, object] = {}  # space -> the latest SELECTION
        self.identity = {"id": session_id, "script": "train_mock.py", "pid": os.getpid(),
                         "start_unix_ms": int(time.time() * 1000)}
        self.seq = 0
        self.port = 0

    def check(self, request) -> Response | None:
        host = request.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            return self.text(403, "bad host")
        origin = request.headers.get("Origin")
        if origin is not None and not ORIGIN.match(origin):
            return self.text(403, "bad origin")
        return None

    @staticmethod
    def text(status: int, body: str) -> Response:
        return Response(status, "Forbidden" if status == 403 else "Not Found",
                        Headers([("Content-Type", "text/plain"), ("Content-Length", str(len(body)))]), body.encode())

    def process_request(self, connection, request):
        bad = self.check(request)
        if bad:
            return bad
        url = urlsplit(request.path)
        token = parse_qs(url.query).get(W.TOKEN_PARAM, [""])[0]
        if url.path == W.PATH:
            return None if secrets.compare_digest(token, self.token) else self.text(403, "bad token")
        name = "index.html" if url.path == "/" else url.path.lstrip("/")
        if name == "index.html" and not secrets.compare_digest(token, self.token):
            return self.text(403, "bad token")
        f = self.dist / name
        if self.wasm is not None and name in WASM_FILES:
            f = self.wasm / name
        if "/" in name or not f.is_file():
            return self.text(404, "not found")
        body = f.read_bytes()
        ctype = ("text/javascript" if name.endswith(".js") else "application/wasm" if name.endswith(".wasm")
                 else mimetypes.guess_type(name)[0] or "application/octet-stream")
        return Response(200, "OK", Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                                            ("Access-Control-Allow-Origin", "*"), ("Cache-Control", "no-store")]), body)

    async def handler(self, ws):
        subs: dict[int, dict] = {}
        lock = asyncio.Lock()

        minor = 0 if self.v30 else W.MINOR

        async def send(msg):
            async with lock:
                await ws.send(W.encode(msg, minor=minor))

        first = W.decode(await asyncio.wait_for(ws.recv(), W.HANDSHAKE_TIMEOUT_MS / 1000))
        if not isinstance(first, W.Hello) or first.role != W.ROLE_VIEWER:
            await ws.close(W.CLOSE_ROLE_REFUSED)
            return
        # 3.1 additions only for what the viewer announced and this producer implements.
        caps = set(first.capabilities) & set(self.capabilities())
        control = (not self.read_only) if W.CAP_CONTROLS in caps else None
        await send(W.Welcome(W.ROLE_PRODUCER, self.capabilities(), {"name": "mock_server", "version": "0"}, self.identity,
                             control=control))
        entry = dict(self.identity, state=W.SESSION_LIVE, views=len(self.views()))
        if not self.v30:
            entry["capabilities"] = self.capabilities()
        await send(W.Sessions([entry]))
        await send(W.Views(self.views(caps), self.identity["id"]))
        if W.CAP_CONTROLS in caps:
            await send(self.ctl.message())
        if W.CAP_SELECTION in caps:
            for sel in self.selections.values():
                if sel.items:
                    await send(sel)
        self.conns[send] = caps
        pump = asyncio.create_task(self.pump(subs, send))
        try:
            async for raw in ws:
                try:
                    msg = W.decode(raw)
                except W.WireError as e:
                    await ws.close(e.close_code)
                    return
                except W.MessageError as e:
                    await send(W.Error(e.code, e.message))
                    continue
                await self.answer(msg, subs, send, ws)
        except ConnectionClosed:
            pass
        finally:
            self.conns.pop(send, None)
            pump.cancel()

    def views(self, caps=frozenset()) -> list[dict]:
        vs = fixed_views() if self.fixed else views(self.run)
        return with_items(vs, self.fixed) if self.has_controls and W.CAP_SELECTION in caps else vs

    def capabilities(self) -> list[str]:
        if self.v30:
            return [W.CAP_EVENTS]
        caps = [W.CAP_EVENTS]
        if self.has_controls:
            caps += [W.CAP_CONTROLS, W.CAP_SELECTION]
        return caps

    async def broadcast_controls(self):
        for send, caps in list(self.conns.items()):
            if W.CAP_CONTROLS in caps:
                await send(self.ctl.message())

    async def broadcast_selection(self, sel):
        for send, caps in list(self.conns.items()):
            if W.CAP_SELECTION in caps:
                await send(sel)

    def retained(self) -> int:
        return len(FIXED_SERIES) if self.fixed else len(self.run.loss)

    async def answer(self, msg, subs, send, ws):
        vs = {v["name"]: v for v in self.views()}
        if isinstance(msg, (W.Subscribe, W.ViewRequest)):
            v = vs.get(msg.view)
            if v is None:
                await send(W.ViewReply(msg.request_id, msg.sub, msg.view, error={"code": W.ERR_UNKNOWN_VIEW, "message": msg.view}))
                return
            try:
                dv = W.resolve_data_view(v, msg.data_view, retain=self.retained())
            except W.MessageError as e:
                await send(W.ViewReply(msg.request_id, msg.sub, msg.view, error={"code": e.code, "message": e.message}))
                return
            await send(W.ViewReply(msg.request_id, msg.sub, msg.view, data_view=dv))
            if isinstance(msg, W.Subscribe):
                back = dv.get("budget", {}).get("max_backfill", 0)
                subs[msg.sub] = {"view": v, "dv": dv, "sent": max(0, self.retained() - back), "fresh": True}
            elif msg.sub in subs:
                subs[msg.sub]["dv"] = dv
                subs[msg.sub]["fresh"] = True
        elif isinstance(msg, W.Unsubscribe):
            subs.pop(msg.sub, None)
        elif isinstance(msg, W.Event):
            print(json.dumps({"event": {"view": msg.view, "type": msg.type, "detail": msg.detail, "sub": msg.sub,
                                        "seq": msg.seq}}), flush=True)
            space = msg.detail.get("space") if msg.type == "select" else None
            if self.has_controls and isinstance(space, str):
                items, cut = W.selection_items(msg.detail.get("items") or [])
                sel = W.Selection(space, items, view=msg.view, truncated=cut, details=item_details(space, items) if items else None)
                self.selections[space] = sel
                await self.broadcast_selection(sel)
        elif isinstance(msg, W.Control):
            print(json.dumps({"control": msg.to_meta()}), flush=True)
            if self.ctl is None or W.CAP_CONTROLS not in self.conns.get(send, set()):
                await send(W.Error(W.ERR_UNSUPPORTED, "this producer has no controls", in_reply_to=int(W.MsgType.CONTROL)))
            else:
                await self.ctl.handle(msg, send, not self.read_only)
        elif isinstance(msg, W.Ping):
            if type(msg) is W.Ping:
                await send(W.Pong(msg.nonce))
        elif isinstance(msg, W.Bye):
            await ws.close(W.CLOSE_NORMAL)

    async def pump(self, subs, send):
        while self.fixed:
            for sub, s in list(subs.items()):
                v = s["view"]
                if v["kind"] == W.Kind.SERIES and s["sent"] < len(FIXED_SERIES):
                    start = s["sent"]
                    s["sent"] = len(FIXED_SERIES)
                    await send(W.Series(sub, v["name"], start, len(FIXED_SERIES), time.monotonic_ns(),
                                        FIXED_SERIES[start:]))
                elif v["kind"] != W.Kind.SERIES and s.pop("fresh", False):
                    self.seq += 1
                    await send(fixed_snapshot(v, sub, self.seq, s["dv"]))
            await asyncio.sleep(0.05)
        while True:
            for sub, s in list(subs.items()):
                v = s["view"]
                if v["kind"] == W.Kind.SERIES:
                    loss = self.run.loss
                    if s["sent"] < len(loss):
                        start = s["sent"]
                        s["sent"] = len(loss)
                        await send(W.Series(sub, v["name"], start, len(loss), time.monotonic_ns(),
                                            np.asarray(loss[start:], np.float32)))
                else:
                    self.seq += 1
                    await send(snapshot(self.run, v, sub, self.seq, s["dv"]))
            await asyncio.sleep(0.05)


async def main():
    global W
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", required=True, help="path of python/caliper/_wire.py")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--dist", default=str(here.parent / "dist"))
    ap.add_argument("--wasm", default=None, help="folder of the WebAssembly views' three files")
    ap.add_argument("--points", type=int, default=4000)
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scene", choices=("clusters", "depth"), default=None,
                    help="fixed 3-D points for the embedding (see scene())")
    ap.add_argument("--fixed", action="store_true", help="serve the fixed data of the view tests")
    ap.add_argument("--controls", action="store_true", help="3.1: run state, parameters, an action and linked selection")
    ap.add_argument("--read-only", action="store_true", help="refuse every CONTROL")
    ap.add_argument("--no-checkpoint", action="store_true", help="never call the checkpoint (run state none)")
    ap.add_argument("--v30", action="store_true", help="a 3.0 producer: minor 0, none of the 3.1 additions")
    ap.add_argument("--step-ms", type=float, default=50.0, help="the time of one training step")
    a = ap.parse_args()
    W = load_wire(a.wire)
    run = Run(a.points, a.dims, a.seed, a.scene)
    token = secrets.token_urlsafe(24)
    prod = Producer(run, token, Path(a.dist), "mock", Path(a.wasm) if a.wasm else None, fixed=a.fixed,
                    controls=a.controls, read_only=a.read_only, v30=a.v30)
    async with serve(prod.handler, "127.0.0.1", a.port, process_request=prod.process_request,
                     compression=None, max_size=W.MAX_FRAME_BYTES) as server:
        prod.port = server.sockets[0].getsockname()[1]
        print(f"http://127.0.0.1:{prod.port}/?token={token}", flush=True)

        async def train():
            while True:
                await asyncio.sleep(a.step_ms / 1000)
                if prod.ctl is not None and not a.no_checkpoint and not await prod.ctl.checkpoint(run.step):
                    continue
                run.advance()

        await asyncio.gather(train(), server.serve_forever())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
