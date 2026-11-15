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

It prints the viewer page's URL on the first line, then one JSON line for
each EVENT a viewer sends.
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
class Run:
    def __init__(self, points: int, dims: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.loss: list[float] = []
        self.weights = (self.rng.standard_normal((48, 64)) * 0.5).astype(np.float32)
        self.n, self.dims = points, dims
        self.labels = (np.arange(points) % 8).astype(np.float32)
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
    def __init__(self, run: Run, token: str, dist: Path, session_id: str, wasm: Path | None = None):
        self.run, self.token, self.dist, self.wasm = run, token, dist, wasm
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

        async def send(msg):
            async with lock:
                await ws.send(W.encode(msg))

        first = W.decode(await asyncio.wait_for(ws.recv(), W.HANDSHAKE_TIMEOUT_MS / 1000))
        if not isinstance(first, W.Hello) or first.role != W.ROLE_VIEWER:
            await ws.close(W.CLOSE_ROLE_REFUSED)
            return
        await send(W.Welcome(W.ROLE_PRODUCER, [W.CAP_EVENTS], {"name": "mock_server", "version": "0"}, self.identity))
        await send(W.Sessions([dict(self.identity, state=W.SESSION_LIVE, views=len(views(self.run)))]))
        await send(W.Views(views(self.run), self.identity["id"]))
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
            pump.cancel()

    async def answer(self, msg, subs, send, ws):
        vs = {v["name"]: v for v in views(self.run)}
        if isinstance(msg, (W.Subscribe, W.ViewRequest)):
            v = vs.get(msg.view)
            if v is None:
                await send(W.ViewReply(msg.request_id, msg.sub, msg.view, error={"code": W.ERR_UNKNOWN_VIEW, "message": msg.view}))
                return
            try:
                dv = W.resolve_data_view(v, msg.data_view, retain=len(self.run.loss))
            except W.MessageError as e:
                await send(W.ViewReply(msg.request_id, msg.sub, msg.view, error={"code": e.code, "message": e.message}))
                return
            await send(W.ViewReply(msg.request_id, msg.sub, msg.view, data_view=dv))
            if isinstance(msg, W.Subscribe):
                back = dv.get("budget", {}).get("max_backfill", 0)
                subs[msg.sub] = {"view": v, "dv": dv, "sent": max(0, len(self.run.loss) - back)}
            elif msg.sub in subs:
                subs[msg.sub]["dv"] = dv
        elif isinstance(msg, W.Unsubscribe):
            subs.pop(msg.sub, None)
        elif isinstance(msg, W.Event):
            print(json.dumps({"event": {"view": msg.view, "type": msg.type, "detail": msg.detail, "sub": msg.sub,
                                        "seq": msg.seq}}), flush=True)
        elif isinstance(msg, W.Ping):
            if type(msg) is W.Ping:
                await send(W.Pong(msg.nonce))
        elif isinstance(msg, W.Bye):
            await ws.close(W.CLOSE_NORMAL)

    async def pump(self, subs, send):
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
    a = ap.parse_args()
    W = load_wire(a.wire)
    run = Run(a.points, a.dims, a.seed)
    token = secrets.token_urlsafe(24)
    prod = Producer(run, token, Path(a.dist), "mock", Path(a.wasm) if a.wasm else None)
    async with serve(prod.handler, "127.0.0.1", a.port, process_request=prod.process_request,
                     compression=None, max_size=W.MAX_FRAME_BYTES) as server:
        prod.port = server.sockets[0].getsockname()[1]
        print(f"http://127.0.0.1:{prod.port}/?token={token}", flush=True)

        async def train():
            while True:
                await asyncio.sleep(0.05)
                run.advance()

        await asyncio.gather(train(), server.serve_forever())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
