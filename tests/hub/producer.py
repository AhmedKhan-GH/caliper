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

With --controls it takes the controls of protocol 3.1 as a script that calls
a checkpoint once per training step does: it announces `controls`, registers
the parameter "lr" (the optimiser's learning rate) and the action "save", and
pauses, resumes and steps at its checkpoint. A parameter value it accepts is
set, and a queued action runs, at the next checkpoint, on the training thread.

With --selection it takes the linked selection of protocol 3.1: it announces
`selection`, declares the item space "units" (the hidden units: the rows of
"fc1.weight" and the columns of "fc2.weight") and "step" (the loss series),
keeps the selection a viewer makes in each space and sends it to every viewer
in SELECTION, with the details of the selected units (the norms of their
input and output weights) computed off the network thread.

With --all-kinds it also streams "units", a points view of the hidden
units placed by their two output weights and coloured by the class they vote
for, the larger of the two (its point dim indexes "units" with --selection),
and "fc1 values", a histogram of the first layer's weights whose "out" and
"in" dims a viewer can index or reduce. "fc1.weight" then also offers the
source "fc1.grad", the weights' gradient, for a viewer's source menu.

When it exits it prints one JSON line: {"id", "steps", "loss_values",
"snapshots_sent", "events", "lr", "saves", "selects", "selection"}.
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


def heat_view(vid: int, name: str, rows: int, cols: int, row_items: str = "", col_items: str = "") -> dict:
    out = {"name": "out", "size": rows}
    inn = {"name": "in", "size": cols}
    if row_items:
        out["items"] = row_items
    if col_items:
        inn["items"] = col_items
    return {"id": vid, "name": name, "kind": int(W.Kind.HEATMAP), "default": {},
            "sources": [{"name": name, "dtype": "f32", "dims": [out, inn]}]}


def apply_index_reduce(a: np.ndarray, dims: list, dv: dict) -> np.ndarray:
    """The source array `a`, whose axes are `dims` in order, under a resolved
    data view's index (an integer drops the dim, a range keeps it) and reduce."""
    ops = {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min,
           "norm": lambda x, axis: np.sqrt(np.sum(x * x, axis=axis))}
    index, reduce = dv.get("index") or {}, dv.get("reduce") or {}
    names = [d["name"] for d in dims]
    for name in list(names):
        ax = names.index(name)
        sel = index.get(name)
        if isinstance(sel, int):
            a = np.take(a, sel, axis=ax)
            names.pop(ax)
        elif isinstance(sel, list):
            a = np.take(a, range(*sel), axis=ax)
        elif name in reduce:
            a = ops[reduce[name]](a, axis=ax)
            names.pop(ax)
    return a


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


class ScriptControls:
    """The run state, the parameter "lr" and the action "save" of protocol 3.1,
    as a script registers them. handle() runs on the network thread and
    answers a CONTROL; checkpoint() runs on the training thread once per step
    and applies what was accepted. Both send CONTROLS when anything changes."""

    def __init__(self, stream: "Stream"):
        self.stream = stream
        self.lock = threading.Lock()
        self.state, self.step, self.steps_left = W.RUN_NONE, 0, 0
        self.lr = {"name": "lr", "type": W.PARAM_NUMBER, "value": 1e-3, "min": 1e-5, "max": 0.1,
                   "log": True, "help": "the learning rate"}
        self.save = {"name": "save", "running": False, "queued": False, "help": "save a checkpoint"}
        self.waiting: list[int] = []        # request ids answered `queued`
        self.saves = 0

    def message(self) -> W.Controls:
        with self.lock:
            run = {"state": self.state, "step": self.step}
            if self.state == W.RUN_STEPPING:
                run["steps_left"] = self.steps_left
            return W.Controls(run, [dict(self.lr)], [dict(self.save)])

    # -- the network thread -------------------------------------------------------
    def handle(self, m: W.Control) -> W.ControlReply:
        def error(code, text):
            return W.ControlReply(m.request_id, m.op, W.STATUS_ERROR, m.name, error={"code": code, "message": text})
        with self.lock:
            if m.op == W.OP_SET:
                if m.name != "lr":
                    return error(W.ERR_UNKNOWN_CONTROL, f"no parameter {m.name!r}")
                try:
                    v = W.param_value(self.lr, m.value)
                except W.MessageError as e:
                    return error(e.code, e.message)
                if self.state == W.RUN_NONE:
                    self.lr["value"] = v
                else:
                    self.lr["pending"] = v
                return W.ControlReply(m.request_id, m.op, W.STATUS_OK, m.name, value=v)
            if m.op == W.OP_RUN:
                if m.name != "save":
                    return error(W.ERR_UNKNOWN_CONTROL, f"no action {m.name!r}")
                self.save["queued"] = True
                self.waiting.append(m.request_id)
                return W.ControlReply(m.request_id, m.op, W.STATUS_QUEUED, m.name)
            if self.state == W.RUN_NONE and m.op in (W.OP_PAUSE, W.OP_STEP):
                return error(W.ERR_UNSUPPORTED, "the script has called no checkpoint yet")
            if m.op == W.OP_PAUSE and self.state in (W.RUN_RUNNING, W.RUN_STEPPING):
                self.state = W.RUN_PAUSING
            elif m.op == W.OP_RESUME:
                self.state = W.RUN_RUNNING
            elif m.op == W.OP_STEP:
                self.state, self.steps_left = W.RUN_STEPPING, m.count or 1
            return W.ControlReply(m.request_id, m.op, W.STATUS_OK)

    # -- the training thread ------------------------------------------------------
    def checkpoint(self, step: int, opt) -> None:
        changed = False
        with self.lock:
            self.step = step
            if self.state == W.RUN_NONE:
                self.state, changed = W.RUN_RUNNING, True
        while True:
            changed |= self._apply(opt)
            with self.lock:
                if self.state == W.RUN_STEPPING:
                    self.steps_left -= 1
                    changed = True
                    if self.steps_left <= 0:
                        self.state = W.RUN_PAUSED
                elif self.state == W.RUN_PAUSING:
                    self.state, changed = W.RUN_PAUSED, True
                paused = self.state == W.RUN_PAUSED
            if changed:
                self.stream.post(self.message())
                changed = False
            if not paused or self.stream.closed.is_set():
                return
            time.sleep(0.01)    # waits in the checkpoint until resume or step

    def _apply(self, opt) -> bool:
        with self.lock:
            pending = self.lr.pop("pending", None)
            if pending is not None:
                self.lr["value"] = pending
            run = self.save["queued"]
            if run:
                self.save["queued"], self.save["running"] = False, True
            waiting, self.waiting = self.waiting, []
        if pending is not None:
            for g in opt.param_groups:
                g["lr"] = pending
        if run:
            self.stream.post(self.message())
            self.saves += 1
            text = f"saved at step {self.step}"
            with self.lock:
                self.save["running"] = False
            for rid in waiting:
                self.stream.post(W.ControlReply(rid, W.OP_RUN, W.STATUS_OK, "save", message=text))
        return pending is not None or run


class Stream:
    """The producer side of one connection to a hub (network thread)."""

    def __init__(self, url: str, identity: dict, views: list, controls: bool = False,
                 selection: bool = False):
        self.url = url
        self.selection = selection
        self.selected: dict[str, list] = {}   # item space -> the current selection
        self.selects = 0                       # select events that changed a selection
        self.identity = identity
        self.views = {v["name"]: v for v in views}
        self.controls = ScriptControls(self) if controls else None
        self.hub_controls = False           # the hub announced `controls`
        self.outbox: list = []               # control messages from the training thread
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

    # Any thread: a control message (CONTROLS, CONTROL_REPLY) to send next.
    def post(self, msg) -> None:
        if isinstance(msg, W.Controls) and not self.hub_controls:
            return
        with self.lock:
            self.outbox.append(W.encode(msg))
        self._poke()

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
            caps = [W.CAP_EVENTS] + ([W.CAP_CONTROLS] if self.controls else []) + \
                ([W.CAP_SELECTION] if self.selection else [])
            await ws.send(W.encode(W.Hello(W.ROLE_PRODUCER, caps,
                                           {"name": "hub-e2e-producer", "version": "1"},
                                           self.identity)))
            welcome = W.decode(await ws.recv())
            assert isinstance(welcome, W.Welcome) and welcome.role == W.ROLE_HUB, welcome
            self.identity = dict(welcome.session)
            self.hub_controls = bool(self.controls) and W.CAP_CONTROLS in welcome.capabilities
            await ws.send(W.encode(W.Views(list(self.views.values()))))
            if self.hub_controls:
                await ws.send(W.encode(self.controls.message()))
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
                if self.selection and msg.type == "select" and "space" in msg.detail:
                    await self._select(ws, msg)
            elif isinstance(msg, W.Control) and self.controls is not None:
                await ws.send(W.encode(self.controls.handle(msg)))
                await ws.send(W.encode(self.controls.message()))
            elif isinstance(msg, W.Ping):
                await ws.send(W.encode(W.Pong(msg.nonce)))
            elif isinstance(msg, W.Bye):
                self.done = True
                self.wake.set()
                return
        self.done = True
        self.wake.set()

    async def _select(self, ws, msg: W.Event) -> None:
        """A viewer selected in an item space: keep it and tell every viewer,
        first at once, then again with the details of the selected units."""
        space = msg.detail["space"]
        ids, cut = W.selection_items(msg.detail.get("items", []))
        cut = cut or bool(msg.detail.get("truncated"))
        self.selected[space] = ids
        self.selects += 1
        await ws.send(W.encode(W.Selection(space, ids, view=msg.view, truncated=cut)))
        if space != "units" or not ids:
            return
        details = await asyncio.get_running_loop().run_in_executor(None, self._unit_details, ids)
        if self.selected.get(space) == ids:
            await ws.send(W.encode(W.Selection(space, ids, view=msg.view, truncated=cut, details=details)))

    def _unit_details(self, ids: list) -> dict:
        """The norms of each selected unit's input weights (a row of fc1) and
        output weights (a column of fc2), for the first selected units."""
        with self.lock:
            fc1 = self.latest.get("fc1.weight", (0, None))[1]
            fc2 = self.latest.get("fc2.weight", (0, None))[1]
        first = ids[:W.MAX_DETAIL_ITEMS]
        rows = []
        for u in first:
            w_in = float(np.linalg.norm(fc1[u])) if fc1 is not None and u < fc1.shape[0] else None
            w_out = float(np.linalg.norm(fc2[:, u])) if fc2 is not None and u < fc2.shape[1] else None
            rows.append([round(w_in, 4) if w_in is not None else None,
                         round(w_out, 4) if w_out is not None else None])
        return {"fields": ["in_norm", "out_norm"], "items": first, "values": rows}

    def _derived(self, sub: int, s: dict, view: dict, seq: int, a) -> W.Snapshot:
        """A snapshot of the points view (positions, values) or the histogram (weights)."""
        if view["kind"] == W.Kind.HISTOGRAM:
            src = view["sources"][0]
            w = np.asarray(apply_index_reduce(np.asarray(a), src["dims"], s["resolved"]), dtype=np.float32).ravel()
            lo, hi = float(w.min()), float(w.max())
            counts, _ = np.histogram(w, bins=32, range=(lo, hi))
            return W.Snapshot(sub, s["view"], seq, time.monotonic_ns(), int(W.Kind.HISTOGRAM), s["resolved"],
                              counts.astype(np.float32), stats=[lo, hi])
        pos, cls = a
        (ax,) = W.payload_layout(view, s["resolved"])["axes"]
        rows = slice(ax["start"], ax["start"] + ax["count"] * ax["step"], ax["step"])
        pos = np.ascontiguousarray(pos[rows][:ax["size"]], dtype=np.float32)
        cls = np.ascontiguousarray(cls[rows][:ax["size"]], dtype=np.float32)
        stats = [0.0, 1.0, float(pos[:, 0].min()), float(pos[:, 0].max()),
                 float(pos[:, 1].min()), float(pos[:, 1].max())]
        return W.Snapshot(sub, s["view"], seq, time.monotonic_ns(), int(W.Kind.POINTS), s["resolved"], pos,
                          aux=cls, stats=stats, axes=[ax], points_dims=2, label_count=2)

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
            with self.lock:
                out, self.outbox = self.outbox, []
            for frame in out:
                await ws.send(frame)
            now = time.monotonic()
            wait = 0.05
            for sub, s in list(self.subs.items()):
                due = s["last"] + s["interval"] - now
                if due > 0:
                    wait = min(wait, due)
                    continue
                view = self.views[s["view"]]
                if view["kind"] in (W.Kind.POINTS, W.Kind.HISTOGRAM):
                    with self.lock:
                        seq, a = self.latest.get(s["view"], (0, None))
                    if a is None or seq == s["seq"]:
                        continue
                    await ws.send(W.encode(self._derived(sub, s, view, seq, a)))
                    self.snapshots_sent += 1
                    s["seq"] = seq
                    s["last"] = now
                    continue
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
                # The source the data view names (a view may offer several).
                key = s["resolved"].get("source", s["view"])
                with self.lock:
                    seq, a = self.latest.get(key, (0, None))
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


def told_to_stop(path: str | None) -> bool:
    """--stop-file: the test ends the script by creating the file, not a timer."""
    return path is not None and os.path.exists(path)


def train(stream: Stream, seconds: float, device: str, stop_file: str | None = None) -> int:
    import torch
    torch.manual_seed(int(os.getpid()) % 1000)
    x = torch.randn(4096, 32, device=device)
    y = (x[:, :4].sum(dim=1) > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    end = time.monotonic() + seconds
    steps = 0
    while time.monotonic() < end and not stream.closed.is_set() and not told_to_stop(stop_file):
        if steps == 1:
            stream.start()   # connect once the model trains, so a listed session streams at once
        idx = torch.randint(0, x.shape[0], (256,), device=device)
        loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if stream.controls is not None:
            stream.controls.checkpoint(steps, opt)
        stream.append(float(loss.item()))
        w1 = model[0].weight.detach().cpu().numpy()
        w2 = model[2].weight.detach().cpu().numpy()
        stream.publish("fc1.weight", w1)
        stream.publish("fc2.weight", w2)
        if "units" in stream.views:
            stream.publish("fc1.grad", model[0].weight.grad.detach().cpu().numpy())
            stream.publish("units", (w2.T.copy(), np.argmax(w2, axis=0).astype(np.float32)))
            stream.publish("fc1 values", w1)
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
    ap.add_argument("--seconds", type=float, default=20, help="the longest the script trains")
    ap.add_argument("--stop-file", help="train until this file exists (or --seconds pass)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rate", type=float, default=60)
    ap.add_argument("--controls", action="store_true", help="take the controls of protocol 3.1")
    ap.add_argument("--selection", action="store_true", help="take the linked selection of protocol 3.1")
    ap.add_argument("--all-kinds", action="store_true", help="also stream a points view and a histogram")
    a = ap.parse_args()
    identity = {"id": a.id, "script": os.path.basename(__file__), "pid": os.getpid(),
                "start_unix_ms": int(time.time() * 1000)}
    if a.bench:
        views = [heat_view(1, "bench", 512, 512)]
    else:
        units = "units" if a.selection else ""
        views = [heat_view(1, "fc1.weight", 64, 32, row_items=units),
                 heat_view(2, "fc2.weight", 2, 64, col_items=units),
                 {"id": 3, "name": "loss", "kind": int(W.Kind.SERIES), "sources": []}]
        if a.selection:
            views[2]["items"] = "step"
        if a.all_kinds:
            grad = dict(views[0]["sources"][0], name="fc1.grad")
            views[0]["sources"].append(grad)
            point = {"name": "unit", "size": 64}
            if a.selection:
                point["items"] = "units"
            views.append({"id": 4, "name": "units", "kind": int(W.Kind.POINTS), "default": {},
                          "sources": [{"name": "units", "dtype": "f32", "dims": [point]}]})
            views.append({"id": 5, "name": "fc1 values", "kind": int(W.Kind.HISTOGRAM), "default": {},
                          "sources": [{"name": "fc1.weight", "dtype": "f32",
                                       "dims": [{"name": "out", "size": 64}, {"name": "in", "size": 32}]}]})
    stream = Stream(hub_address(a.hub_file), identity, views, controls=a.controls, selection=a.selection)
    if a.bench:
        stream.start()
        steps = bench(stream, a.seconds, a.rate)
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        steps = train(stream, a.seconds, device, a.stop_file)
    time.sleep(0.3)   # let the last values go out
    stream.stop()
    print(json.dumps({"id": a.id, "steps": steps, "loss_values": len(stream.loss),
                      "snapshots_sent": stream.snapshots_sent, "events": stream.events,
                      "lr": stream.controls.lr["value"] if stream.controls else None,
                      "saves": stream.controls.saves if stream.controls else 0,
                      "selects": stream.selects,
                      "selection": {k: v[:16] for k, v in stream.selected.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
