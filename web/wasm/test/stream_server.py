"""A synthetic producer for testing Caliper's browser views (web/wasm).

It serves one session over wire protocol v3 on 127.0.0.1, with the shared
codec in python/caliper/_wire.py and the websockets package, the way
caliper.serve() does: the WebSocket endpoint /v3/stream?token=<token>, and the
files of a build directory (then of test/pages) over plain HTTP on the same
port, so a test page and its module come from the producer's own origin, as
with caliper.serve(). Static files carry Access-Control-Allow-Origin: *, which
a page of another origin needs to load the module (README.md, "Serving").

Views:
  activations  heatmap (H, W) f32, a moving 2-D pattern; its source has a
               "layer" dim of 4, and the data view's index picks the layer
  loss         series, a noisy decaying curve, several values per tick
  histogram    histogram of 64 bins over [-4, 4], a drifting normal sample
  image        image (H, W, 4) u8, a moving colour pattern

Usage:
  python stream_server.py --site <build dir> [--port 0] [--rate 60]
                          [--heat 256x192] [--seconds N]
It prints one line "READY <port> <token>" when it listens, and "EVENT ..."
for every viewer event. It never binds anything but 127.0.0.1.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import math
import mimetypes
import os
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
from caliper import _wire as w  # noqa: E402

STEPS_PER_TICK = 4
RETAIN = 16384


class Session:
    """The producer's state: the views and the series history."""

    def __init__(self, heat_w: int, heat_h: int):
        self.id = "synthetic"
        self.start_ms = int(time.time() * 1000)
        self.w, self.h = heat_w, heat_h
        self.t0 = time.monotonic()
        self.seq = {"activations": 0, "histogram": 0, "image": 0}
        self.series = np.zeros(0, dtype=np.float32)
        self.total = 0
        self.rng = np.random.default_rng(12345)
        self.views = [
            {"id": 1, "name": "activations", "kind": int(w.Kind.HEATMAP),
             "sources": [{"name": "activations", "dtype": "f32",
                          "dims": [{"name": "layer", "size": 4}, {"name": "row", "size": heat_h},
                                   {"name": "col", "size": heat_w}]}],
             "default": {"index": {"layer": 0}}},
            {"id": 2, "name": "loss", "kind": int(w.Kind.SERIES), "sources": []},
            {"id": 3, "name": "histogram", "kind": int(w.Kind.HISTOGRAM),
             "sources": [{"name": "values", "dtype": "f32", "dims": [{"name": "sample", "size": 4096}]}]},
            {"id": 4, "name": "image", "kind": int(w.Kind.IMAGE),
             "sources": [{"name": "image", "dtype": "u8",
                          "dims": [{"name": "row", "size": heat_h}, {"name": "col", "size": heat_w}]}]},
        ]

    def identity(self) -> dict:
        return {"id": self.id, "script": "stream_server.py", "pid": os.getpid(),
                "start_unix_ms": self.start_ms}

    def view(self, name: str) -> dict | None:
        return next((v for v in self.views if v["name"] == name), None)

    def heatmap(self, layer: int) -> np.ndarray:
        t = time.monotonic() - self.t0
        y, x = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cx = self.w * (0.5 + 0.3 * math.cos(t * 0.9))
        cy = self.h * (0.5 + 0.3 * math.sin(t * 0.9))
        f = 1.0 + 0.5 * layer
        v = (0.6 * np.sin(0.07 * f * x + 1.4 * t) * np.cos(0.09 * f * y - 0.8 * t)
             + 0.3 * np.sin(0.025 * (x + 2 * y) + 0.5 * t)
             + 1.4 * np.exp(-(((x - cx) / 22.0) ** 2 + ((y - cy) / 22.0) ** 2)))
        return v.astype(np.float32)

    def image(self) -> np.ndarray:
        t = time.monotonic() - self.t0
        y, x = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        rgba = np.empty((self.h, self.w, 4), dtype=np.uint8)
        rgba[..., 0] = (127.5 + 127.5 * np.sin(0.05 * x + t)).astype(np.uint8)
        rgba[..., 1] = (127.5 + 127.5 * np.sin(0.04 * y - 0.7 * t)).astype(np.uint8)
        rgba[..., 2] = (127.5 + 127.5 * np.cos(0.03 * (x - y) + 0.4 * t)).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba

    def histogram(self) -> np.ndarray:
        t = time.monotonic() - self.t0
        sample = self.rng.normal(math.sin(t * 0.5), 1.0 + 0.3 * math.cos(t * 0.3), 4096)
        counts, _ = np.histogram(sample, bins=64, range=(-4.0, 4.0))
        return counts.astype(np.float32)

    def append_series(self) -> None:
        s = self.total + np.arange(STEPS_PER_TICK, dtype=np.float64)
        v = (2.2 * np.exp(-s / 450.0) + 0.12
             + 0.06 * np.exp(-s / 1500.0) * self.rng.standard_normal(STEPS_PER_TICK)
             + 0.01 * self.rng.standard_normal(STEPS_PER_TICK)).astype(np.float32)
        self.series = np.concatenate([self.series, v])[-RETAIN:]
        self.total += STEPS_PER_TICK


class Viewer:
    """One viewer connection: its subscriptions and what it was last sent."""

    def __init__(self):
        self.subs: dict[int, dict] = {}   # sub -> {"view", "data_view", "sent_total"}


async def serve_viewer(ws, session: Session, rate: float):
    viewer = Viewer()
    try:
        hello = w.decode(await asyncio.wait_for(ws.recv(), 10))
    except (asyncio.TimeoutError, ConnectionClosed):
        return
    if not isinstance(hello, w.Hello) or hello.role != "viewer":
        await ws.close(w.CLOSE_ROLE_REFUSED)
        return
    await ws.send(w.encode(w.Welcome("producer", ["events"], {"name": "stream_server", "version": "1"},
                                     session.identity())))
    await ws.send(w.encode(w.Sessions([{**session.identity(), "state": "live",
                                        "views": len(session.views)}])))
    await ws.send(w.encode(w.Views(list(session.views), session.id)))

    async def receive():
        async for raw in ws:
            msg = w.decode(raw)
            if isinstance(msg, w.Subscribe):
                view = session.view(msg.view)
                if view is None:
                    await ws.send(w.encode(w.ViewReply(msg.request_id, msg.sub, msg.view, error={
                        "code": "unknown_view", "message": f"no view {msg.view!r}"})))
                    continue
                try:
                    dv = w.resolve_data_view(view, msg.data_view, retain=RETAIN)
                except w.MessageError as e:
                    await ws.send(w.encode(w.ViewReply(msg.request_id, msg.sub, msg.view, error={
                        "code": e.code, "message": str(e)})))
                    continue
                viewer.subs[msg.sub] = {"view": msg.view, "data_view": dv,
                                        "sent_total": max(0, session.total - len(session.series))}
                await ws.send(w.encode(w.ViewReply(msg.request_id, msg.sub, msg.view, dv)))
            elif isinstance(msg, w.ViewRequest):
                sub = viewer.subs.get(msg.sub)
                view = session.view(msg.view)
                try:
                    if sub is None or view is None:
                        raise w.MessageError("unknown_subscription", f"no subscription {msg.sub}")
                    dv = w.resolve_data_view(view, msg.data_view, retain=RETAIN)
                except w.MessageError as e:
                    await ws.send(w.encode(w.ViewReply(msg.request_id, msg.sub, msg.view, error={
                        "code": e.code, "message": str(e)})))
                    continue
                sub["data_view"] = dv
                await ws.send(w.encode(w.ViewReply(msg.request_id, msg.sub, msg.view, dv)))
            elif isinstance(msg, w.Unsubscribe):
                viewer.subs.pop(msg.sub, None)
            elif isinstance(msg, w.Ping):
                await ws.send(w.encode(w.Pong(msg.nonce)))
            elif isinstance(msg, w.Event):
                print("EVENT", msg.view, msg.type, msg.detail, flush=True)
            elif isinstance(msg, w.Bye):
                return

    async def send_loop():
        period = 1.0 / rate
        while True:
            t = time.monotonic()
            stamp = time.monotonic_ns()
            for sub_id, sub in list(viewer.subs.items()):
                name = sub["view"]
                if name == "loss":
                    lo = max(sub["sent_total"], session.total - len(session.series))
                    if lo < session.total:
                        vals = session.series[len(session.series) - (session.total - lo):]
                        for i in range(0, len(vals), 65536):
                            chunk = vals[i:i + 65536]
                            await ws.send(w.encode(w.Series(sub_id, name, lo + i, session.total, stamp, chunk)))
                        sub["sent_total"] = session.total
                    continue
                session.seq[name] += 1
                if name == "activations":
                    layer = sub["data_view"].get("index", {}).get("layer", 0)
                    data = session.heatmap(layer if isinstance(layer, int) else 0)
                    snap = w.Snapshot(sub_id, name, session.seq[name], stamp, int(w.Kind.HEATMAP),
                                      sub["data_view"], data, colormap=0, flags=3,
                                      stats=[float(data.min()), float(data.max())])
                elif name == "histogram":
                    snap = w.Snapshot(sub_id, name, session.seq[name], stamp, int(w.Kind.HISTOGRAM),
                                      sub["data_view"], session.histogram(), stats=[-4.0, 4.0])
                elif name == "image":
                    snap = w.Snapshot(sub_id, name, session.seq[name], stamp, int(w.Kind.IMAGE),
                                      sub["data_view"], session.image())
                else:
                    continue
                await ws.send(w.encode(snap))
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t)))

    recv_task = asyncio.create_task(receive())
    send_task = asyncio.create_task(send_loop())
    try:
        done, _ = await asyncio.wait({recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception() and not isinstance(t.exception(), ConnectionClosed):
                print("viewer error:", repr(t.exception()), file=sys.stderr, flush=True)
    finally:
        recv_task.cancel()
        send_task.cancel()


def make_request_handler(site: Path | None, token: str, pages: Path | None = None):
    def process_request(connection, request):
        parts = urlsplit(request.path)
        if parts.path == "/v3/stream":
            got = parse_qs(parts.query).get("token", [""])[0]
            if not hmac.compare_digest(got.encode(), token.encode()):
                return connection.respond(403, "Forbidden\n")
            return None   # go on with the upgrade
        name = parts.path.lstrip("/") or "demo.html"
        path = None
        for root in (site, pages):   # the build directory first, then the test pages
            if root is None:
                continue
            p = (root / name).resolve()
            if p.is_file() and root.resolve() in p.parents:
                path = p
                break
        if path is None:
            return connection.respond(404, "Not Found\n")
        ctype = {".wasm": "application/wasm", ".js": "text/javascript", ".html": "text/html",
                 ".ttf": "font/ttf"}.get(path.suffix) or mimetypes.guess_type(name)[0] or "application/octet-stream"
        body = path.read_bytes()
        headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                           ("Cache-Control", "no-store"), ("Access-Control-Allow-Origin", "*")])
        return Response(200, "OK", headers, body)
    return process_request


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=None)
    ap.add_argument("--pages", type=Path, default=HERE / "pages")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--heat", default="256x192")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    heat_w, heat_h = (int(x) for x in args.heat.split("x"))
    token = args.token or secrets.token_urlsafe(16)
    session = Session(heat_w, heat_h)

    async def ticker():   # the producer appends to the series whether or not anyone watches
        while True:
            session.append_series()
            await asyncio.sleep(1.0 / args.rate)

    async def handler(ws):
        await serve_viewer(ws, session, args.rate)

    async with serve(handler, "127.0.0.1", args.port, process_request=make_request_handler(args.site, token, args.pages),
                     compression=None, max_size=w.MAX_FRAME_BYTES, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        print("READY", port, token, flush=True)
        tick = asyncio.create_task(ticker())
        if args.seconds > 0:
            await asyncio.sleep(args.seconds)
        else:
            await asyncio.Future()
        tick.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
