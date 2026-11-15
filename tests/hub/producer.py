"""A test producer for the hub's end-to-end check (tests/hub/e2e.py).

It trains a small PyTorch model and streams it to a `caliper serve` hub over
wire protocol v3, connecting out as a producer the way the reference page's
"The producer connection to a hub" describes. It is built on the package's
codec (caliper/_wire.py) and the websockets package only, so that the hub can
be checked end to end before the package's own outbound connection exists.

Views: "fc1.weight" and "fc2.weight" (heatmaps of the two weight matrices)
and "loss" (a series). The training loop runs on the main thread and never
waits for the network: it stores the newest weights and appends the loss,
and a network thread sends what its subscriptions ask for, at their rate.

    python tests/hub/producer.py --hub-file <hub.json> --id e2e-a --seconds 30

With --bench it streams one 512 x 512 heatmap, "bench", at --rate snapshots
per second instead of training.

When it exits it prints one JSON line: {"id", "steps", "loss_values",
"snapshots_sent", "events"}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time

import numpy as np

from caliper import _wire as W


def hub_address(hub_file: str) -> str:
    with open(hub_file) as f:
        h = json.load(f)
    return f"{h['stream_url']}?{W.TOKEN_PARAM}={h['token']}"


def heat_view(vid: int, name: str, rows: int, cols: int) -> dict:
    return {"id": vid, "name": name, "kind": int(W.Kind.HEATMAP), "default": {},
            "sources": [{"name": name, "dtype": "f32",
                         "dims": [{"name": "out", "size": rows}, {"name": "in", "size": cols}]}]}


def pool(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Adaptive mean pooling of a 2-D array to (rows, cols), as the protocol states."""
    if a.shape == (rows, cols):
        return a
    out = np.empty((rows, cols), dtype=np.float32)
    H, W_ = a.shape
    for i in range(rows):
        r0, r1 = (i * H) // rows, -(-((i + 1) * H) // rows)
        for j in range(cols):
            c0, c1 = (j * W_) // cols, -(-((j + 1) * W_) // cols)
            out[i, j] = a[r0:r1, c0:c1].mean()
    return out


class Stream:
    """The producer side of one connection to a hub (network thread)."""

    def __init__(self, url: str, identity: dict, views: list):
        self.url = url
        self.identity = identity
        self.views = {v["name"]: v for v in views}
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[int, np.ndarray]] = {}   # view -> (seq, array)
        self.loss: list[float] = []
        self.seq = 0
        self.subs: dict[int, dict] = {}
        self.events = 0
        self.snapshots_sent = 0
        self.closed = threading.Event()
        self.done = False
        self.wake = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)

    # -- the training thread ------------------------------------------------------
    def publish(self, name: str, array: np.ndarray) -> None:
        with self.lock:
            self.seq += 1
            self.latest[name] = (self.seq, array)
        self._poke()

    def append(self, value: float) -> None:
        with self.lock:
            self.loss.append(value)
        self._poke()

    def _poke(self) -> None:
        if self.wake is not None:
            self.loop.call_soon_threadsafe(self.wake.set)

    # -- the network thread ---------------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.done = True
        self._poke()
        if self.thread.is_alive():
            self.thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:                      # the hub went away: training goes on
            print(f"producer {self.identity['id']}: connection ended: {e!r}", file=sys.stderr)
        finally:
            self.closed.set()

    async def _main(self) -> None:
        import websockets
        self.wake = asyncio.Event()
        async with websockets.connect(self.url, compression=None, max_size=W.MAX_FRAME_BYTES,
                                      ping_interval=None) as ws:
            await ws.send(W.encode(W.Hello(W.ROLE_PRODUCER, [W.CAP_EVENTS],
                                           {"name": "hub-e2e-producer", "version": "1"},
                                           self.identity)))
            welcome = W.decode(await ws.recv())
            assert isinstance(welcome, W.Welcome) and welcome.role == W.ROLE_HUB, welcome
            self.identity = dict(welcome.session)
            await ws.send(W.encode(W.Views(list(self.views.values()))))
            reader = asyncio.ensure_future(self._reader(ws))
            try:
                await self._writer(ws)
            finally:
                reader.cancel()
            if self.done:
                await ws.send(W.encode(W.Bye(W.BYE_NORMAL, "the script finished")))

    async def _reader(self, ws) -> None:
        async for frame in ws:
            msg = W.decode(frame)
            if isinstance(msg, W.Subscribe):
                self._subscribe(msg, ws)
            elif isinstance(msg, W.Unsubscribe):
                self.subs.pop(msg.sub, None)
            elif isinstance(msg, W.ViewRequest):
                s = self.subs.get(msg.sub)
                if s is None:
                    reply = W.ViewReply(msg.request_id, msg.sub, msg.view,
                                        error={"code": W.ERR_UNKNOWN_SUBSCRIPTION, "message": "no sub"})
                else:
                    try:
                        s["resolved"] = W.resolve_data_view(self.views[s["view"]], msg.data_view)
                        reply = W.ViewReply(msg.request_id, msg.sub, msg.view, data_view=s["resolved"])
                    except W.MessageError as e:
                        reply = W.ViewReply(msg.request_id, msg.sub, msg.view,
                                            error={"code": e.code, "message": e.message})
                await ws.send(W.encode(reply))
            elif isinstance(msg, W.Event):
                self.events += 1
            elif isinstance(msg, W.Ping):
                await ws.send(W.encode(W.Pong(msg.nonce)))
            elif isinstance(msg, W.Bye):
                self.done = True
                self.wake.set()
                return
        self.done = True
        self.wake.set()

    def _subscribe(self, msg: W.Subscribe, ws) -> None:
        v = self.views.get(msg.view)
        if v is None:
            reply = W.ViewReply(msg.request_id, msg.sub, msg.view,
                                error={"code": W.ERR_UNKNOWN_VIEW, "message": f"no view {msg.view!r}"})
        else:
            try:
                resolved = W.resolve_data_view(v, msg.data_view, retain=1 << 20)
                rate = min(float(msg.max_rate or W.DEFAULT_RATE_HZ), float(W.LIMIT_RATE_HZ))
                back = resolved.get("budget", {}).get("max_backfill", 0) if v["kind"] == W.Kind.SERIES else 0
                with self.lock:
                    start = max(0, len(self.loss) - back)
                self.subs[msg.sub] = {"view": msg.view, "resolved": resolved, "interval": 1.0 / rate,
                                      "last": -1.0, "seq": 0, "next": start}
                reply = W.ViewReply(msg.request_id, msg.sub, msg.view, data_view=resolved)
            except W.MessageError as e:
                reply = W.ViewReply(msg.request_id, msg.sub, msg.view,
                                    error={"code": e.code, "message": e.message})
        asyncio.ensure_future(ws.send(W.encode(reply)))
        self.wake.set()

    async def _writer(self, ws) -> None:
        while not self.done:
            self.wake.clear()
            now = time.monotonic()
            wait = 0.05
            for sub, s in list(self.subs.items()):
                due = s["last"] + s["interval"] - now
                if due > 0:
                    wait = min(wait, due)
                    continue
                view = self.views[s["view"]]
                if view["kind"] == W.Kind.SERIES:
                    with self.lock:
                        total = len(self.loss)
                        vals = np.asarray(self.loss[s["next"]:s["next"] + W.SERIES_MAX_VALUES],
                                          dtype=np.float32)
                    if len(vals) == 0:
                        continue
                    await ws.send(W.encode(W.Series(sub, s["view"], s["next"], total,
                                                    time.monotonic_ns(), vals)))
                    s["next"] += len(vals)
                    s["last"] = now
                    continue
                with self.lock:
                    seq, a = self.latest.get(s["view"], (0, None))
                if a is None or seq == s["seq"]:
                    continue
                layout = W.payload_layout(view, s["resolved"])
                (r, c) = layout["axes"]
                part = a[r["start"]:r["start"] + r["count"] * r["step"]:r["step"],
                         c["start"]:c["start"] + c["count"] * c["step"]:c["step"]]
                data = pool(np.ascontiguousarray(part, dtype=np.float32), r["size"], c["size"])
                lo, hi = float(data.min()), float(data.max())
                await ws.send(W.encode(W.Snapshot(
                    sub, s["view"], seq, time.monotonic_ns(), int(W.Kind.HEATMAP), s["resolved"], data,
                    colormap=2, flags=W.SNAP_AUTO_VMIN | W.SNAP_AUTO_VMAX, stats=[lo, hi],
                    axes=layout["axes"])))
                self.snapshots_sent += 1
                s["seq"] = seq
                s["last"] = now
            try:
                await asyncio.wait_for(self.wake.wait(), max(wait, 0.001))
            except asyncio.TimeoutError:
                pass


def train(stream: Stream, seconds: float, device: str) -> int:
    import torch
    torch.manual_seed(int(os.getpid()) % 1000)
    x = torch.randn(4096, 32, device=device)
    y = (x[:, :4].sum(dim=1) > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    end = time.monotonic() + seconds
    steps = 0
    while time.monotonic() < end and not stream.closed.is_set():
        if steps == 1:
            stream.start()   # connect once the model trains, so a listed session streams at once
        idx = torch.randint(0, x.shape[0], (256,), device=device)
        loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        stream.append(float(loss.item()))
        stream.publish("fc1.weight", model[0].weight.detach().cpu().numpy())
        stream.publish("fc2.weight", model[2].weight.detach().cpu().numpy())
        time.sleep(0.004)
    return steps


def bench(stream: Stream, seconds: float, rate: float) -> int:
    rng = np.random.default_rng(1)
    frames = [rng.standard_normal((512, 512), dtype=np.float32) for _ in range(4)]
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < seconds and not stream.closed.is_set():
        stream.publish("bench", frames[n % 4])
        n += 1
        nxt = t0 + n / rate
        time.sleep(max(0.0, nxt - time.monotonic()))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-file", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rate", type=float, default=60)
    a = ap.parse_args()
    identity = {"id": a.id, "script": os.path.basename(__file__), "pid": os.getpid(),
                "start_unix_ms": int(time.time() * 1000)}
    if a.bench:
        views = [heat_view(1, "bench", 512, 512)]
    else:
        views = [heat_view(1, "fc1.weight", 64, 32), heat_view(2, "fc2.weight", 2, 64),
                 {"id": 3, "name": "loss", "kind": int(W.Kind.SERIES), "sources": []}]
    stream = Stream(hub_address(a.hub_file), identity, views)
    if a.bench:
        stream.start()
        steps = bench(stream, a.seconds, a.rate)
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        steps = train(stream, a.seconds, device)
    time.sleep(0.3)   # let the last values go out
    stream.stop()
    print(json.dumps({"id": a.id, "steps": steps, "loss_values": len(stream.loss),
                      "snapshots_sent": stream.snapshots_sent, "events": stream.events}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
