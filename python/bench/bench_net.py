"""What streaming snapshots over a WebSocket costs a training loop: throughput
against a commit-matched control, per-call cost, snapshots and bytes per second
on the socket, and step-to-receipt latency at a viewer in a second process.

    PYTHONPATH=python CALIPER_SESSION=0 .venv/bin/python python/bench/bench_net.py \\
        --transport serve|standin [--host-copy nonblocking,direct,event-locked,event] \\
        [--device mps|cpu] [--size 1024] [--points 10000,100000] [--viewers 3] \\
        [--arms ...] [--repeat 3] [--seconds 8] [--warmup 1] [--quick]

One heatmap size per invocation (as in bench_show.py); the log goes to
python/bench/results/N-<transport>-<device>-<size>[-<host copy>].txt as well as
stdout. Stand-in results always carry "standin" in their name.

Workload (bench_show.py's): the TinyCNN of examples/python/train_show.py on
64-image batches plus a weight W (size x size, f32, updated in place every
step). The points views show E (100,000 x 3 f32, static) with 4-class labels;
the 10,000-point view is its first 10,000 rows. The series view is the loss.

Arms (each --seconds long after --warmup; every arm runs once per repeat, the
arms interleaved, and every ratio is taken within one repeat):
  baseline        the loop alone (the stream is open and idle)
  control         MPS: torch.mps.Event.record() at the snapshot rate, no snapshot.
                  Recording an event commits torch's command buffer, which by itself
                  raises this small model's throughput, so streaming is compared
                  with this arm, not only with the baseline.
  noviewer        every view called every step, nothing connected
  heatmap-<S>     show(W) every step, one viewer subscribed
  series          series(loss) every step, one viewer subscribed
  points-<N>k     points(E[:N], labels, classes=4) every step, one viewer subscribed
  all             all of the views above every step, one viewer subscribed to all
  viewers-<K>     as "all", with K viewers

Each viewer is net_client.py in its own process. Latency is the viewer's
receipt time minus the stamp the producer put in the snapshot at the view call
(both time.monotonic_ns, mach_absolute_time on macOS).

--transport standin uses a deliberately simple transport written in this file
(StandinStream), not the product's: for each view call at up to 60 Hz it
reduces the tensor on the device, queues it, and a background thread moves it
to the host and sends it. --host-copy picks how the reduced snapshot reaches the
host without the training thread waiting for it; with several, comma-separated,
every arm with a viewer runs once per method, the methods side by side in each
repeat (arm names "<arm>/<method>"):
  event         (a) the training thread queues a device copy and records a
                torch.mps.Event; the background thread polls the event, then .cpu().
                With torch 2.14.1 this aborts the process within seconds (a Metal
                assertion: torch's MPS stream is driven from two threads at once).
  event-locked  (a) made safe: one lock serializes the training thread's torch work
                (step and view calls) with the background thread's .cpu()
  direct        (b) the training thread calls .cpu() on the reduced tensor (drains)
  nonblocking   (c) the training thread queues copy_(non_blocking=True) into a pinned
                CPU buffer and records an event; the background thread polls the
                event and reads the buffer. No .cpu() and no torch call that encodes
                GPU work happens off the training thread.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import secrets
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http import HTTPStatus
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
NET_CLIENT = os.path.join(HERE, "net_client.py")
sys.path.insert(0, HERE)
from net_client import standin_encode, standin_decode   # noqa: E402  (the stand-in codec)


class Calls(NamedTuple):
    show: object        # show(t, name) -> truthy when it took a snapshot
    series: object      # series(value, name)
    points: object      # points(z, labels, name, classes=)


# ---------------------------------------------------------------------------
# adapters: the only code that knows the sprint's transport (caliper.serve).
# Fit these two functions when the transport is committed.
# ---------------------------------------------------------------------------
def open_stream(args):
    """Start the producer side. Returns (server, calls, codec): server has url,
    port, token and close(); calls are the view calls with caliper's signatures;
    codec is net_client.py's --codec for this stream."""
    if args.transport == "serve":
        import caliper
        if not hasattr(caliper, "serve"):
            raise SystemExit("caliper.serve() is not in this checkout; --transport standin "
                             "measures the stand-in transport")
        server = caliper.serve(port=0)
        return server, Calls(caliper.show, caliper.series, caliper.points), "wire"
    s = StandinStream(args.device, args.host_copies[0], args.rate, args.max_extent, args.max_points)
    return s, Calls(s.show, s.series, s.points), "standin"


def stream_url(server) -> str:
    """The viewer's WebSocket endpoint (wire protocol v3)."""
    return f"ws://127.0.0.1:{server.port}/v3/stream?token={server.token}"


# ---------------------------------------------------------------------------
# the training workload (examples/python/train_show.py's TinyCNN + a weight W)
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def make_batch(n, device):
    labels = torch.randint(0, 4, (n,), device=device)
    cy = 7.0 + 14.0 * (labels // 2).float() + torch.randn(n, device=device) * 2
    cx = 7.0 + 14.0 * (labels % 2).float() + torch.randn(n, device=device) * 2
    g = torch.arange(28, device=device, dtype=torch.float32)
    yy = (g[None, :, None] - cy[:, None, None]) ** 2
    xx = (g[None, None, :] - cx[:, None, None]) ** 2
    img = torch.exp(-(yy + xx) / 18.0) + 0.15 * torch.randn(n, 28, 28, device=device)
    return img[:, None], labels


class Workload:
    def __init__(self, device, size, n_points, dim):
        torch.manual_seed(0)
        self.device = torch.device(device)
        self.model = TinyCNN().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.W = torch.randn(size, size, device=self.device)
        self.E = torch.randn(n_points, dim, device=self.device)
        self.labels = torch.randint(0, 4, (n_points,), device=self.device)
        self.loss = None

    def step(self):
        x, y = make_batch(64, self.device)
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.W.add_(0.001)
        self.loss = loss.detach()

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()


def train(wl, seconds, hook=None, marks=None, lock=None):
    """Train for `seconds`, hook() after every step. Returns (steps, elapsed
    incl. the final device sync). marks gets perf_counter_ns at every step
    boundary (step times, including the view calls). lock, if given, is held
    around each step with its view calls and around the final sync."""
    pc = time.perf_counter_ns
    steps = 0
    t0 = time.perf_counter()
    t_end = t0 + seconds
    if marks is not None:
        marks.append(pc())
    while time.perf_counter() < t_end:
        if lock is not None:
            lock.acquire()
        wl.step()
        if hook is not None:
            hook()
        if lock is not None:
            lock.release()
        steps += 1
        if marks is not None:
            marks.append(pc())
    if lock is not None:
        with lock:
            wl.sync()
    else:
        wl.sync()
    return steps, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# the stand-in transport (a measuring stick, not the product's transport)
# ---------------------------------------------------------------------------
KIND_HEATMAP, KIND_POINTS, KIND_SERIES = 1, 3, 4
POOL_SLOTS = 3          # pinned host buffers per view for --host-copy nonblocking
HISTORY = 4096          # retained series values, sent first on subscribe
POLL_S = 0.001          # background thread's event poll interval while work is pending


class _Entry:
    __slots__ = ("view", "kind", "seq", "t_ns", "meta", "dev", "host", "buf", "event")

    def __init__(self, view, kind, seq, t_ns, meta):
        self.view, self.kind, self.seq, self.t_ns, self.meta = view, kind, seq, t_ns, meta
        self.dev = self.host = self.buf = self.event = None

    def ready(self):
        return self.event is None or self.event.query()


class _Conn:
    def __init__(self):
        self.views = set()
        self.fifo = collections.deque()      # control messages and series deltas, in order
        self.latest = {}                     # view -> newest snapshot frame (latest wins)
        self.series_next = {}                # view -> next series index this viewer needs
        self.wake = asyncio.Event()


class StandinStream:
    """A deliberately simple transport for proving the harness. For each view
    call at up to `rate` Hz it reduces the tensor on the device (heatmaps pooled
    to max_extent, points strided to max_points), queues it, and a background
    thread moves it to the host, encodes it with net_client.py's stand-in codec
    and hands it to a websockets server on 127.0.0.1. Latest snapshot wins per
    view and viewer; series go as deltas. A view nobody subscribes to costs a
    dict lookup."""

    def __init__(self, device, host_copy, rate=60.0, max_extent=512, max_points=100_000):
        self.device = torch.device(device)
        self.host_copy = host_copy if self.device.type == "mps" else "cpu"
        self.torch_lock = threading.Lock() if self.host_copy == "event-locked" else None
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.max_extent, self.max_points = max_extent, max_points
        self.token = secrets.token_urlsafe(16)
        self._subs = {}                      # view -> subscriber count (read lock-free)
        self._last = {}                      # view -> perf_counter of its last snapshot
        self._seq = {}
        self._series = {}                    # view -> [pending values, last tick, next index]
        self._known = set()
        self._cond = threading.Condition()
        self._pending = {}                   # view -> _Entry (latest wins)
        self._fifo = {}                      # series view -> deque of _Entry
        self._retiring = []                  # superseded entries whose host buffer is in flight
        self._pools = {}                     # view -> deque of free pinned buffers
        self._allocated = collections.Counter()
        self._history = {}                   # series view -> deque of (start, values)
        self._conns = set()
        self.counters = collections.Counter()
        self.spans_copy = []                 # (start, end) ns of each background .cpu()
        self.spans_publish = []              # (start, end) ns of each background publish
        self._stopping = False
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._net = threading.Thread(target=self._run_loop, name="standin-net", daemon=True)
        self._net.start()
        if not self._ready.wait(10):
            raise RuntimeError("stand-in server did not start")
        self.url = f"http://127.0.0.1:{self.port}/?token={self.token}"
        self._worker = threading.Thread(target=self._work, name="standin-host", daemon=True)
        self._worker.start()

    def set_host_copy(self, host_copy):
        """Switch how snapshots reach the host, once the earlier ones are sent."""
        t_end = time.monotonic() + 10.0
        while (self._pending or self._retiring or any(self._fifo.values())) \
                and time.monotonic() < t_end:
            time.sleep(0.005)
        self.host_copy = host_copy if self.device.type == "mps" else "cpu"
        self.torch_lock = threading.Lock() if self.host_copy == "event-locked" else None

    # -- the view calls (training thread) ---------------------------------
    def _due(self, name):
        if not self._subs.get(name):
            return None
        now = time.perf_counter()
        if now - self._last.get(name, -1e9) < self.interval:
            return None
        if self.host_copy == "nonblocking" and not self._pool_ok(name):
            self.counters["busy"] += 1
            return None
        self._last[name] = now
        return now

    def show(self, t, name="tensor", cmap="viridis", vmin=None, vmax=None, rate=None):
        if self._due(name) is None:
            return False
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        h, w = t.shape
        k = max(-(-h // self.max_extent), -(-w // self.max_extent))
        r = F.avg_pool2d(t[None, None], k, ceil_mode=True)[0, 0] if k > 1 else t
        mn, mx = torch.aminmax(r)
        payload = torch.cat((r.reshape(-1), mn.reshape(1), mx.reshape(1)))
        return self._queue(name, KIND_HEATMAP, payload, {
            "dtype": "f32", "shape": list(r.shape), "cmap": 0,
            "data_view": {"budget": {"max_extent": self.max_extent}, "pool": k}})

    def points(self, z, labels=None, name="points", classes=None, cmap="viridis", vmin=None,
               vmax=None, rate=None):
        if self._due(name) is None:
            return False
        z = z.detach()
        if z.dtype != torch.float32:
            z = z.float()
        n, d = z.shape
        step = -(-n // self.max_points)
        if step > 1:
            z = z[::step]
            labels = labels[::step] if labels is not None else None
        lo, hi = torch.aminmax(z, dim=0)
        parts = [z.reshape(-1)]
        if labels is not None:
            parts.append(labels.reshape(-1).float())
        payload = torch.cat(parts + [lo, hi])
        return self._queue(name, KIND_POINTS, payload, {
            "dtype": "f32", "shape": [z.shape[0], d], "dims": d, "labels": labels is not None,
            "classes": classes or 0, "data_view": {"budget": {"max_points": self.max_points},
                                                   "stride": step}})

    def series(self, value, name="series", rate=None):
        if not self._subs.get(name):
            return False
        st = self._series.get(name)
        if st is None:
            st = self._series[name] = [[], -1e9, 0]
        st[0].append(value)
        now = time.perf_counter()
        if now - st[1] < self.interval:
            return False
        if self.host_copy == "nonblocking" and not self._pool_ok(name):
            self.counters["busy"] += 1
            return False
        st[1] = now
        vals, st[0] = st[0], []
        if isinstance(vals[0], torch.Tensor):
            payload = torch.stack(vals).reshape(-1).float()
        else:
            payload = torch.tensor(vals, dtype=torch.float32)
        start = st[2]
        st[2] += len(vals)
        return self._queue(name, KIND_SERIES, payload, {"start": start, "n": len(vals)})

    def _pool_ok(self, name):
        return bool(self._pools.get(name)) or self._allocated[name] < POOL_SLOTS

    def _queue(self, name, kind, payload, meta):
        seq = self._seq[name] = self._seq.get(name, 0) + 1
        e = _Entry(name, kind, seq, time.monotonic_ns(), meta)
        if payload.device.type == "cpu":
            e.host = payload
        elif self.host_copy == "direct":                       # (b)
            e.host = payload.cpu()
        elif self.host_copy in ("event", "event-locked"):      # (a)
            e.dev = payload
            e.event = torch.mps.Event()
            e.event.record()
        else:                                                  # (c)
            pool = self._pools.setdefault(name, collections.deque())
            n = payload.numel()
            try:
                buf = pool.pop()
            except IndexError:
                buf = None
            if buf is None or buf.numel() < n:
                if buf is None:
                    self._allocated[name] += 1
                buf = torch.empty(max(n, HISTORY if kind == KIND_SERIES else n), pin_memory=True)
            e.buf = buf
            e.host = buf[:n]
            e.host.copy_(payload, non_blocking=True)
            e.event = torch.mps.Event()
            e.event.record()
        self.counters["taken"] += 1
        with self._cond:
            self._known.add(name)
            if kind == KIND_SERIES:
                self._fifo.setdefault(name, collections.deque()).append(e)
            else:
                old = self._pending.get(name)
                if old is not None:
                    self.counters["superseded"] += 1
                    if old.buf is not None:
                        self._retiring.append(old)
                self._pending[name] = e
            self._cond.notify()
        return True

    # -- the background thread: device -> host, encode, hand to the server --
    def _work(self):
        mono = time.monotonic_ns
        while True:
            with self._cond:
                while not (self._pending or self._retiring or any(self._fifo.values())):
                    if self._stopping:
                        return
                    self._cond.wait(0.1)
                if self._stopping:
                    return
                cands = list(self._pending.values()) + [q[0] for q in self._fifo.values() if q]
                retiring = list(self._retiring)
            for e in retiring:
                if e.event.query():
                    with self._cond:
                        self._retiring.remove(e)
                    self._pools[e.view].append(e.buf)
            progressed = False
            for e in sorted(cands, key=lambda x: x.t_ns):
                if not e.ready():
                    continue
                with self._cond:
                    if e.kind == KIND_SERIES:
                        q = self._fifo[e.view]
                        if not q or q[0] is not e:
                            continue
                        q.popleft()
                    else:
                        if self._pending.get(e.view) is not e:
                            continue
                        del self._pending[e.view]
                t0 = mono()
                if e.host is None:                             # (a): the event fired, copy now
                    if self.torch_lock is not None:
                        with self.torch_lock:
                            e.host = e.dev.cpu()
                            e.dev = None
                    else:
                        e.host = e.dev.cpu()
                    self.spans_copy.append((t0, mono()))
                self._publish(e)
                self.spans_publish.append((t0, mono()))
                progressed = True
            if not progressed:
                time.sleep(POLL_S)

    def _publish(self, e):
        a = e.host.numpy()
        meta = {"type": "snapshot", "view": e.view, "seq": e.seq, "t_ns": e.t_ns}
        if e.kind == KIND_HEATMAP:
            h, w = e.meta["shape"]
            meta.update(e.meta, kind=KIND_HEATMAP, vmin=float(a[h * w]), vmax=float(a[h * w + 1]))
            sections = [a[:h * w]]
        elif e.kind == KIND_POINTS:
            n, d = e.meta["shape"]
            off = n * d + (n if e.meta["labels"] else 0)
            meta.update(e.meta, kind=KIND_POINTS, range=[a[off:off + d].tolist(), a[off + d:off + 2 * d].tolist()])
            sections = [a[:n * d]] + ([a[n * d:off]] if e.meta["labels"] else [])
        else:
            meta = {"type": "series", "view": e.view, "start": e.meta["start"], "n": e.meta["n"],
                    "t_ns": e.t_ns}
            sections = [a]
        frame = standin_encode(meta, sections)
        start = end = None
        if e.kind == KIND_SERIES:
            start, end = e.meta["start"], e.meta["start"] + e.meta["n"]
        with self._cond:
            if e.kind == KIND_SERIES:
                h = self._history.setdefault(e.view, collections.deque())
                h.append((start, a.copy()))
                while sum(len(v) for _, v in h) > HISTORY and len(h) > 1:
                    h.popleft()
            if e.buf is not None:
                self._pools[e.view].append(e.buf)
            self._loop.call_soon_threadsafe(self._deliver, e.view, frame, start, end)
        self.counters["published"] += 1

    # -- the server (asyncio thread) ----------------------------------------
    def _run_loop(self):
        from websockets.asyncio.server import serve
        asyncio.set_event_loop(self._loop)

        async def start():
            self._server = await serve(self._handler, "127.0.0.1", 0, compression=None,
                                       max_size=2**20, ping_interval=None,
                                       process_request=self._check)
            self.port = self._server.sockets[0].getsockname()[1]
        self._loop.run_until_complete(start())
        self._ready.set()
        self._loop.run_forever()

    def _check(self, connection, request):
        u = urllib.parse.urlsplit(request.path)
        if u.path != "/v3/stream":
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        tok = urllib.parse.parse_qs(u.query).get("token", [""])[0]
        if not secrets.compare_digest(tok, self.token):
            return connection.respond(HTTPStatus.FORBIDDEN, "bad session token\n")
        return None

    def _deliver(self, view, frame, start, end):
        for c in self._conns:
            if view not in c.views:
                continue
            if start is None:
                c.latest[view] = frame
            else:
                if start < c.series_next.get(view, 0):
                    continue                           # already in the history it was sent
                c.fifo.append(frame)
                c.series_next[view] = end
            c.wake.set()

    def _subscribe(self, c, view):
        with self._cond:
            if view in c.views:
                return
            c.views.add(view)
            self._subs[view] = self._subs.get(view, 0) + 1
            h = self._history.get(view)
            if h:
                start = h[0][0]
                vals = np.concatenate([v for _, v in h])
                c.fifo.append(standin_encode({"type": "series", "view": view, "start": start,
                                              "n": len(vals)}, [vals]))
                c.series_next[view] = start + len(vals)
        c.wake.set()

    def _unsubscribe(self, c, view):
        with self._cond:
            if view in c.views:
                c.views.discard(view)
                self._subs[view] -= 1

    async def _handler(self, ws):
        from websockets.exceptions import ConnectionClosed
        c = _Conn()
        self._conns.add(c)
        sender = asyncio.create_task(self._sender(ws, c))
        try:
            async for raw in ws:
                msg = standin_decode(raw)
                if msg.type == "hello":
                    c.fifo.append(standin_encode({"type": "welcome", "standin": True,
                                                  "views": sorted(self._known)}))
                    c.wake.set()
                elif msg.type == "subscribe":
                    self._subscribe(c, msg.meta["view"])
                elif msg.type == "unsubscribe":
                    self._unsubscribe(c, msg.meta["view"])
        except ConnectionClosed:
            pass
        finally:
            self._conns.discard(c)
            for v in list(c.views):
                self._unsubscribe(c, v)
            sender.cancel()

    async def _sender(self, ws, c):
        from websockets.exceptions import ConnectionClosed
        try:
            while True:
                await c.wake.wait()
                c.wake.clear()
                while c.fifo or c.latest:
                    if c.fifo:
                        f = c.fifo.popleft()
                    else:
                        f = c.latest.pop(next(iter(c.latest)))
                    await ws.send(f)
        except (ConnectionClosed, asyncio.CancelledError):
            pass

    def reset_stats(self):
        self.counters.clear()
        self.spans_copy = []
        self.spans_publish = []

    def close(self):
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._worker.join(5)

        async def aclose():
            self._server.close()
            await self._server.wait_closed()
        try:
            asyncio.run_coroutine_threadsafe(aclose(), self._loop).result(10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._net.join(5)


# ---------------------------------------------------------------------------
# viewers (net_client.py processes)
# ---------------------------------------------------------------------------
def launch_viewers(url, views, codec, k, tmp, tag, keep_busy, timeout=30.0):
    """Start k viewers and return once every one has subscribed. keep_busy()
    runs meanwhile: a v3 view exists only once the script has called it, so
    the script keeps training and calling its views while viewers subscribe."""
    procs = []
    for i in range(k):
        out = os.path.join(tmp, f"{tag}-v{i}.json")
        err = open(out + ".stderr", "w")
        p = subprocess.Popen([sys.executable, NET_CLIENT, "--url", url, "--views", ",".join(views),
                              "--codec", codec, "--out", out], stdout=subprocess.PIPE, stderr=err,
                             text=True)
        err.close()
        procs.append((p, out))
    waiting = {p.stdout: (p, out) for p, out in procs}
    t_end = time.monotonic() + timeout
    while waiting:
        for f in select.select(list(waiting), [], [], 0)[0]:
            if "READY" in f.readline():
                del waiting[f]
        for p, out in waiting.values():
            if time.monotonic() > t_end or p.poll() is not None:
                stop_viewers(procs)
                raise RuntimeError(f"viewer did not subscribe: {open(out + '.stderr').read()[-2000:]}")
        if waiting:
            keep_busy()
    return procs


def stop_viewers(procs):
    """SIGTERM every viewer (it writes its records and exits); returns the records."""
    for p, _ in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    recs = []
    for p, out in procs:
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        try:
            with open(out) as f:
                recs.append(json.load(f))
        except (OSError, ValueError):
            recs.append(None)
    return recs


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def q(xs, p):
    if xs is None or len(xs) == 0:
        return float("nan")
    xs = np.sort(np.asarray(xs))
    return float(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))])


def r1(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def viewer_numbers(recs, t0, t1, seconds):
    """Received messages/s, bytes/s and latencies (ms) within [t0, t1]."""
    got = nbytes = errors = 0
    lat, dec, missing = [], [], 0
    per_view = collections.Counter()
    for rec in recs:
        if rec is None:
            errors += 1
            continue
        errors += rec["errors"]
        recv = np.asarray(rec["recv_ns"], dtype=np.int64)
        if len(recv) == 0:
            continue
        stamp = np.asarray(rec["stamp_ns"], dtype=np.int64)
        data = np.asarray([t in ("snapshot", "series") for t in rec["type"]])
        win = (recv >= t0) & (recv <= t1)
        got += int((win & data).sum())
        nbytes += int(np.asarray(rec["nbytes"])[win].sum())
        for v in np.asarray(rec["view"])[win & data]:
            per_view[rec["views"][v] if v >= 0 else "?"] += 1
        stamped = data & (stamp > 0) & (stamp >= t0) & (stamp <= t1)
        missing += int((data & win & (stamp <= 0)).sum())
        lat.extend(((recv[stamped] - stamp[stamped]) / 1e6).tolist())
        dec.extend((np.asarray(rec["decode_ns"])[win & data] / 1e3).tolist())
    return {"received_per_s": got / seconds, "MB_per_s": nbytes / seconds / 1e6,
            "received_by_view": dict(per_view), "viewer_errors": errors,
            "unstamped": missing, "decode_us_med": q(dec, 0.5)}, lat


def overlap_split(marks, spans):
    """Step durations (us) split by whether the step overlapped a span."""
    b = np.asarray(marks, dtype=np.int64)
    lo, hi, dur = b[:-1], b[1:], np.diff(b) / 1e3
    if not spans:
        return np.empty(0), dur
    s = np.asarray([a for a, _ in spans], dtype=np.int64)
    e = np.asarray([z for _, z in spans], dtype=np.int64)
    idx = np.searchsorted(e, lo, side="right")
    ok = idx < len(s)
    hit = np.zeros(len(lo), dtype=bool)
    hit[ok] = s[idx[ok]] < hi[ok]
    return dur[hit], dur[~hit]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def arm_plan(args):
    """[(arm, views, viewers, host copy)]; views name what the arm calls every
    step; host copy is the stand-in's method for the arm (None for serve)."""
    pts = [f"emb{n // 1000}k" for n in args.points]
    every = ["W", "loss"] + pts
    plan = [("baseline", [], 0)]
    if args.device == "mps":
        plan.append(("control", [], 0))
    plan += [("noviewer", every, 0), (f"heatmap-{args.size}", ["W"], 1), ("series", ["loss"], 1)]
    plan += [(f"points-{n // 1000}k", [p], 1) for n, p in zip(args.points, pts)]
    plan += [("all", every, 1)]
    if args.viewers > 1:
        plan.append((f"viewers-{args.viewers}", every, args.viewers))
    if args.arms:
        want = args.arms.split(",")
        plan = [a for a in plan if any(a[0] == w or a[0].startswith(w + "-") for w in want)]
    hcs = args.host_copies if args.transport == "standin" else [None]
    out = []
    for name, vs, n in plan:
        if n == 0 or len(hcs) == 1:
            out.append((name, vs, n, hcs[0]))
        else:
            out += [(f"{name}/{h}", vs, n, h) for h in hcs]
    return out


def make_hook(arm, views, wl, calls, rate, costs):
    """The per-step view calls of an arm, each timed singly."""
    pc = time.perf_counter_ns
    fns = []
    for v in views:
        if v == "W":
            fns.append((v, lambda: calls.show(wl.W, "W")))
        elif v == "loss":
            fns.append((v, lambda: calls.series(wl.loss, "loss")))
        else:
            n = int(v[3:-1]) * 1000
            z, lab = wl.E[:n], wl.labels[:n]
            fns.append((v, lambda z=z, lab=lab, v=v: calls.points(z, lab, v, classes=4)))
    if arm == "control":
        ev, last, iv = torch.mps.Event(), [0.0], 1.0 / rate

        def commit():
            now = time.perf_counter()
            if now - last[0] >= iv:
                last[0] = now
                ev.record()
                return True
            return False
        fns = [("commit", commit)]
    if not fns:
        return None
    slots = [(fn, costs.setdefault(v, ([], []))) for v, fn in fns]

    def hook():
        for fn, c in slots:
            a = pc()
            took = fn()
            c[1 if took else 0].append(pc() - a)
    return hook


def run_arm(args, wl, server, calls, codec, arm, views, viewers, hc, rep, tmp, pooled):
    tag = f"{arm.replace('/', '-')}-r{rep}"
    standin = isinstance(server, StandinStream)
    if standin:
        server.set_host_copy(hc)
    lock = server.torch_lock if standin else None
    warm = make_hook(arm, views, wl, calls, args.rate, {})
    procs = []
    if viewers:
        procs = launch_viewers(stream_url(server), views, codec, viewers, tmp, tag,
                               lambda: train(wl, 0.02, warm, lock=lock))
    try:
        train(wl, args.warmup, warm, lock=lock)
        if standin:
            server.reset_stats()
        costs, marks = {}, []
        hook = make_hook(arm, views, wl, calls, args.rate, costs)
        load0 = os.getloadavg()[0]
        t0 = time.monotonic_ns()
        steps, el = train(wl, args.seconds, hook, marks, lock)
        t1 = time.monotonic_ns()
        load1 = os.getloadavg()[0]
        if procs:
            time.sleep(0.3)          # frames in flight reach the viewers
    finally:
        recs = stop_viewers(procs) if procs else []
    row = {"arm": arm, "rep": rep, "host_copy": hc, "views": views, "viewers": viewers,
           "steps": steps,
           "seconds": round(el, 3), "steps_per_s": round(steps / el, 1),
           "load": [round(load0, 2), round(load1, 2)]}
    pool = pooled.setdefault(arm, {"calls": {}, "lat": [], "steps": [], "ov": [], "free": [],
                                   "copy_ms": []})
    taken = 0
    row["calls"] = {}
    for v, (skip, snap) in costs.items():
        pc_ = pool["calls"].setdefault(v, ([], []))
        pc_[0].extend(skip)
        pc_[1].extend(snap)
        if v != "commit":
            taken += len(snap)
        row["calls"][v] = {"snap": len(snap), "skip": len(skip),
                           "snap_us_med": r1(q(snap, 0.5) / 1e3), "snap_us_p99": r1(q(snap, 0.99) / 1e3),
                           "skip_us_med": r1(q(skip, 0.5) / 1e3, 2),
                           "skip_us_p99": r1(q(skip, 0.99) / 1e3, 2)}
    row["taken_per_s"] = round(taken / el, 1)
    dur = np.diff(np.asarray(marks, dtype=np.int64)) / 1e3
    pool["steps"].extend(dur.tolist())
    row["step_us"] = {"p50": r1(q(dur, 0.5)), "p99": r1(q(dur, 0.99)), "max": r1(float(dur.max()))}
    if viewers:
        vn, lat = viewer_numbers(recs, t0, t1, el)
        pool["lat"].extend(lat)
        row.update({k: (r1(x, 2) if isinstance(x, float) else x) for k, x in vn.items()})
        row.update(lat_ms_med=r1(q(lat, 0.5), 2), lat_ms_p95=r1(q(lat, 0.95), 2), lat_n=len(lat))
    if standin:
        ov, free = overlap_split(marks, server.spans_copy)
        pool["ov"].extend(ov.tolist())
        pool["free"].extend(free.tolist())
        cp = [(b - a) / 1e6 for a, b in server.spans_copy]
        pool["copy_ms"].extend(cp)
        pub = [(b - a) / 1e6 for a, b in server.spans_publish]
        row["standin"] = dict(server.counters)
        row["worker"] = {"copies_per_s": round(len(cp) / el, 1), "copy_ms_med": r1(q(cp, 0.5), 3),
                         "publish_ms_med": r1(q(pub, 0.5), 3),
                         "steps_in_copy": len(ov), "step_in_copy_us_med": r1(q(ov, 0.5)),
                         "step_in_copy_us_p99": r1(q(ov, 0.99)), "step_other_us_med": r1(q(free, 0.5)),
                         "step_other_us_p99": r1(q(free, 0.99))}
    return row


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def spread(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return f"{fmt.format(float(np.median(xs)))} ({fmt.format(min(xs))} to {fmt.format(max(xs))})"


def med_or_dash(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    return fmt.format(float(np.median(xs))) if xs else "-"


def report(out, args, plan, rows, pooled):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["rep"]][r["arm"]] = r
    reps = sorted(by)

    def ratio(arm, ref):
        xs = []
        for rep in reps:
            a, b = by[rep].get(arm), by[rep].get(ref)
            if a and b:
                xs.append(100.0 * a["steps_per_s"] / b["steps_per_s"])
        return xs

    out("")
    out(f"{args.transport}{'' if args.transport == 'serve' else ' (' + args.host_copy_label + ')'} "
        f"{args.device} W {args.size}x{args.size}: median of {len(reps)} repeat(s), "
        f"(min to max) across repeats; ratios are taken within each repeat")
    out("| arm | viewers | steps/s | % of control | % of baseline | snapshots taken/s | received/s "
        "| MB/s on the socket | latency ms median / p95 [n] | step µs p50 / p99 / max |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for arm, views, viewers, hc in plan:
        rs = [by[rep][arm] for rep in reps if arm in by[rep]]
        if not rs:
            continue
        p = pooled[arm]
        st = np.asarray(p["steps"])
        lat = p["lat"]
        ctl = spread(ratio(arm, "control")) if any("control" in by[x] for x in reps) else "-"
        latc = f"{q(lat, 0.5):.2f} / {q(lat, 0.95):.2f} [{len(lat)}]" if lat else "-"
        out(f"| {arm} | {viewers} | {spread([r['steps_per_s'] for r in rs])} | {ctl} "
            f"| {spread(ratio(arm, 'baseline'))} "
            f"| {med_or_dash([r['taken_per_s'] for r in rs]) if views else '-'} "
            f"| {med_or_dash([r.get('received_per_s') for r in rs])} "
            f"| {med_or_dash([r.get('MB_per_s') for r in rs], '{:.2f}')} | {latc} "
            f"| {q(st, 0.5):.0f} / {q(st, 0.99):.0f} / {st.max():.0f} |")
    unstamped = sum(r.get("unstamped", 0) for r in rows)
    if unstamped:
        out(f"{unstamped} messages in the measured windows carried no producer stamp (t_ns), so "
            "their latency is not measured; see README-net.md.")
    out("")
    out("Per call, pooled over the repeats (µs; 'skipped' is the rate limit, or nobody "
        "subscribed in the noviewer arm):")
    out("| arm | view | calls | took a snapshot: median / p99 [n] | skipped: median / p99 [n] |")
    out("|---|---|---|---|---|")
    for arm, views, viewers, hc in plan:
        for v, (skip, snap) in pooled.get(arm, {}).get("calls", {}).items():
            sn = f"{q(snap, 0.5) / 1e3:.1f} / {q(snap, 0.99) / 1e3:.1f} [{len(snap)}]" if snap else "-"
            sk = f"{q(skip, 0.5) / 1e3:.2f} / {q(skip, 0.99) / 1e3:.2f} [{len(skip)}]" if skip else "-"
            out(f"| {arm} | {v} | {len(snap) + len(skip)} | {sn} | {sk} |")
    if args.transport == "standin":
        out("")
        out("Stand-in background thread, pooled over the repeats: its device-to-host .cpu() "
            "calls (host copy 'event' and 'event-locked' only) and the training steps that "
            "overlapped one:")
        out("| arm | .cpu() calls/s | .cpu() ms median / p99 | steps overlapping a .cpu(): µs "
            "median / p99 [n] | other steps: µs median / p99 [n] | published/s | superseded/s |")
        out("|---|---|---|---|---|---|---|")
        for arm, views, viewers, hc in plan:
            rs = [by[rep][arm] for rep in reps if arm in by[rep]]
            if not rs or not viewers:
                continue
            p = pooled[arm]
            cp, ov, fr = p["copy_ms"], p["ov"], p["free"]
            secs = sum(r["seconds"] for r in rs)
            pub = sum(r["standin"].get("published", 0) for r in rs) / secs
            sup = sum(r["standin"].get("superseded", 0) for r in rs) / secs
            cpc = f"{q(cp, 0.5):.2f} / {q(cp, 0.99):.2f}" if cp else "-"
            ovc = f"{q(ov, 0.5):.0f} / {q(ov, 0.99):.0f} [{len(ov)}]" if ov else "-"
            out(f"| {arm} | {len(cp) / secs:.1f} | {cpc} | {ovc} | {q(fr, 0.5):.0f} / {q(fr, 0.99):.0f} "
                f"[{len(fr)}] | {pub:.1f} | {sup:.1f} |")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transport", choices=("serve", "standin"), default="serve")
    ap.add_argument("--host-copy", default="nonblocking",
                    help="stand-in only: how a reduced snapshot reaches the host, one or more "
                    "of nonblocking, direct, event-locked and event, comma-separated")
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--size", type=int, default=1024, help="heatmap W is size x size")
    ap.add_argument("--points", default="10000,100000", help="point counts of the points views")
    ap.add_argument("--points-dim", type=int, choices=(2, 3), default=3)
    ap.add_argument("--viewers", type=int, default=3, help="viewers in the viewers-K arm (<2: no arm)")
    ap.add_argument("--arms", default="", help="comma-separated arm names or prefixes (default: all)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=60.0,
                    help="snapshot rate of the stand-in and of the control's event record (Hz)")
    ap.add_argument("--max-extent", type=int, default=512, help="stand-in heatmap budget")
    ap.add_argument("--max-points", type=int, default=100_000, help="stand-in points budget")
    ap.add_argument("--quick", action="store_true", help="a smoke test of a few seconds")
    ap.add_argument("--tag", default="", help="suffix of the result file name")
    ap.add_argument("--results", default=RESULTS, help="folder of the result file")
    ap.add_argument("--tmp", default=None, help="folder for the viewers' records (default: a "
                    "temporary folder, removed at the end)")
    args = ap.parse_args()
    if args.quick:
        args.repeat, args.seconds, args.warmup = 1, 0.5, 0.25
        args.viewers = min(args.viewers, 2)
        args.tag = args.tag or "-quick"
    args.points = [int(x) for x in args.points.split(",") if x]
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("no MPS device", file=sys.stderr)
        return 2
    args.host_copies = [h for h in args.host_copy.split(",") if h] if args.device == "mps" else ["cpu"]
    bad = set(args.host_copies) - {"nonblocking", "direct", "event-locked", "event", "cpu"}
    if bad or not args.host_copies:
        ap.error(f"--host-copy: unknown method {', '.join(sorted(bad))}")
    args.host_copy_label = ",".join(args.host_copies)
    label = args.host_copies[0] if len(args.host_copies) == 1 else "hostcopy"
    name = f"N-{args.transport}-{args.device}-{args.size}"
    if args.transport == "standin":
        name = f"N-standin-{label}-{args.size}" if args.device == "cpu" else \
            f"N-standin-{label}-{args.device}-{args.size}"
    os.makedirs(args.results, exist_ok=True)
    out = Tee(os.path.join(args.results, f"{name}{args.tag}.txt"))
    tmp = args.tmp or tempfile.mkdtemp(prefix="caliper-bench-net-")
    os.makedirs(tmp, exist_ok=True)

    plan = arm_plan(args)
    out(f"# bench_net.py --transport {args.transport}"
        + (f" --host-copy {args.host_copy_label}" if args.transport == "standin" else "")
        + f" --device {args.device} --size {args.size} --seconds {args.seconds:g} --repeat "
        f"{args.repeat}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    out(f"torch {torch.__version__}, W {args.size}x{args.size} f32 updated in place every step, "
        f"points {'/'.join(str(n) for n in args.points)} x {args.points_dim} f32 with 4-class labels, "
        f"series of the loss; snapshot rate {args.rate:g} Hz; {args.seconds:g} s per arm after "
        f"{args.warmup:g} s warm-up; arms {', '.join(a[0] for a in plan)}; load average "
        f"{' '.join(f'{x:.1f}' for x in os.getloadavg())} ({os.cpu_count()} cores)")
    if args.transport == "standin":
        out("STAND-IN transport written in bench_net.py (a measuring stick, not the product's "
            f"transport): heatmaps pooled to {args.max_extent}, points strided to {args.max_points}, "
            f"host copy '{args.host_copy_label}'")

    wl = Workload(args.device, args.size, max(args.points), args.points_dim)
    server, calls, codec = open_stream(args)
    rows, pooled = [], {}
    try:
        train(wl, max(args.warmup, 1.0), lock=getattr(server, "torch_lock", None))
        for rep in range(1, args.repeat + 1):
            for arm, views, viewers, hc in plan:
                r = run_arm(args, wl, server, calls, codec, arm, views, viewers, hc, rep, tmp,
                            pooled)
                rows.append(r)
                out("RESULT " + json.dumps(r))
    finally:
        server.close()
        if args.tmp is None:
            shutil.rmtree(tmp, ignore_errors=True)
    out(f"load average at the end {' '.join(f'{x:.1f}' for x in os.getloadavg())}")
    report(out, args, plan, rows, pooled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
