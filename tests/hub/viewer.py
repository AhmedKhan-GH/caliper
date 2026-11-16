"""A Python viewer client of wire protocol v3 for the hub's end-to-end check.

It connects to a hub, reads the session list, attaches to one session,
subscribes to every view of it and counts what arrives, on the package's codec
(caliper/_wire.py) and the websockets package. tests/hub/e2e.py imports it;
it also runs on its own:

    python tests/hub/viewer.py --hub-file <hub.json> --session e2e-a --seconds 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from caliper import _wire as W


async def sessions(url: str) -> list:
    """The hub's session list, as SESSIONS gives it right after WELCOME."""
    import websockets
    async with websockets.connect(url, compression=None, ping_interval=None) as ws:
        await ws.send(W.encode(W.Hello(W.ROLE_VIEWER)))
        while True:
            msg = W.decode(await asyncio.wait_for(ws.recv(), 10))
            if isinstance(msg, W.Sessions):
                await ws.send(W.encode(W.Bye(W.BYE_NORMAL, "done")))
                return msg.sessions


async def watch(url: str, session: str | None, seconds: float, *, max_rate: float | None = None,
                views: list | None = None) -> dict:
    """Attach to `session` (or follow the hub's default when None) and count
    what arrives for `seconds`. Returns a summary dict."""
    import websockets
    out = {"session": None, "state": None, "sessions": [], "views": [], "snapshots": {}, "series": {},
           "errors": [], "latency_ms": [], "bytes": 0}
    async with websockets.connect(url, compression=None, max_size=W.MAX_FRAME_BYTES, ping_interval=None) as ws:
        await ws.send(W.encode(W.Hello(W.ROLE_VIEWER, [], {"name": "hub-e2e-viewer", "version": "1"})))
        sub_of: dict[int, str] = {}
        series: dict[str, dict[int, float]] = {}
        req = 0
        attached = False
        subscribed = False
        if session is not None:
            await ws.send(W.encode(W.Attach(session)))
        end = None
        while end is None or time.monotonic() < end:
            timeout = 10.0 if end is None else max(0.0, end - time.monotonic())
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout)
            except asyncio.TimeoutError:
                if end is None:
                    out["errors"].append("timed out waiting for the session's views")
                break
            now = time.monotonic_ns()
            msg = W.decode(frame)
            if isinstance(msg, W.Sessions):
                out["sessions"] = msg.sessions
                for s in msg.sessions:
                    if s["id"] == (session or out["session"]):
                        out["state"] = s["state"]
            elif isinstance(msg, W.Views):
                if msg.session is None:
                    continue
                if session is not None and msg.session != session:
                    continue
                out["session"] = msg.session
                out["views"] = [v["name"] for v in msg.views]
                for s in out["sessions"]:
                    if s["id"] == msg.session:
                        out["state"] = s["state"]
                attached = True
                if not subscribed and msg.views:
                    subscribed = True
                    for i, v in enumerate(msg.views):
                        if views is not None and v["name"] not in views:
                            continue
                        req += 1
                        sub_of[i + 1] = v["name"]
                        await ws.send(W.encode(W.Subscribe(i + 1, v["name"], req, max_rate=max_rate)))
                    end = time.monotonic() + seconds
            elif isinstance(msg, W.ViewReply):
                if msg.error:
                    out["errors"].append(f"{msg.view}: {msg.error}")
            elif isinstance(msg, W.Snapshot):
                name = sub_of.get(msg.sub, "?")
                out["snapshots"][name] = out["snapshots"].get(name, 0) + 1
                out["latency_ms"].append((now - msg.stamp_ns) / 1e6)
                out["bytes"] += len(frame)
            elif isinstance(msg, W.Series):
                name = sub_of.get(msg.sub, "?")
                vals = msg.values.array()
                d = series.setdefault(name, {})
                for k, x in enumerate(vals.tolist()):
                    d[msg.start + k] = x
            elif isinstance(msg, W.Ping):
                await ws.send(W.encode(W.Pong(msg.nonce)))
            elif isinstance(msg, W.Error):
                out["errors"].append(f"error {msg.code}: {msg.message}")
            elif isinstance(msg, W.Bye):
                break
        if not attached:
            out["errors"].append("never attached")
        for name, d in series.items():
            idx = sorted(d)
            out["series"][name] = {"values": len(d), "first": idx[0], "last": idx[-1],
                                   "contiguous": idx[-1] - idx[0] + 1 == len(d)}
        await ws.send(W.encode(W.Bye(W.BYE_NORMAL, "done")))
    return out


async def control(url: str, session: str, lr: float, seconds: float = 8.0) -> dict:
    """Protocol 3.1 through a hub: announce `controls` and `selection`, attach
    to `session`, wait for its CONTROLS, then set the parameter "lr" and run
    the action "save". Returns what came back: WELCOME's `control`, the
    replies by request id, and the parameter as the last CONTROLS shows it."""
    import websockets
    out = {"control": None, "capabilities": [], "replies": {1: [], 2: []}, "lr": None, "pending": None,
           "controls": 0, "errors": []}
    async with websockets.connect(url, compression=None, max_size=W.MAX_FRAME_BYTES, ping_interval=None) as ws:
        await ws.send(W.encode(W.Hello(W.ROLE_VIEWER, [W.CAP_CONTROLS, W.CAP_SELECTION],
                                       {"name": "hub-e2e-viewer", "version": "1"})))
        await ws.send(W.encode(W.Attach(session)))
        sent = False
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                msg = W.decode(await asyncio.wait_for(ws.recv(), max(0.0, end - time.monotonic())))
            except asyncio.TimeoutError:
                break
            if isinstance(msg, W.Welcome):
                out["control"], out["capabilities"] = msg.control, msg.capabilities
            elif isinstance(msg, W.Controls):
                out["controls"] += 1
                lr = next((p for p in msg.params if p["name"] == "lr"), None)
                if lr is not None:
                    out["lr"], out["pending"] = lr["value"], lr.get("pending")
                if not sent:
                    sent = True
                    await ws.send(W.encode(W.Control(1, W.OP_SET, "lr", lr_value(lr))))
                    await ws.send(W.encode(W.Control(2, W.OP_RUN, "save")))
            elif isinstance(msg, W.ControlReply):
                out["replies"].setdefault(msg.request_id, []).append(
                    {"status": msg.status, "value": msg.value, "message": msg.message,
                     "error": (msg.error or {}).get("code")})
            elif isinstance(msg, W.Ping):
                await ws.send(W.encode(W.Pong(msg.nonce)))
            elif isinstance(msg, W.Error):
                out["errors"].append(f"error {msg.code}: {msg.message}")
            r1, r2 = out["replies"][1], out["replies"][2]
            done_set = r1 and (r1[-1]["status"] == W.STATUS_ERROR or out["lr"] == r1[-1]["value"])
            done_run = r2 and r2[-1]["status"] != W.STATUS_QUEUED
            if done_set and done_run:
                break
        await ws.send(W.encode(W.Bye(W.BYE_NORMAL, "done")))
    return out


def lr_value(param) -> float:
    """The value the e2e check sets: half the parameter's maximum."""
    return (param or {}).get("max", 0.1) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-file", required=True)
    ap.add_argument("--session")
    ap.add_argument("--seconds", type=float, default=3)
    a = ap.parse_args()
    with open(a.hub_file) as f:
        h = json.load(f)
    url = f"{h['stream_url']}?{W.TOKEN_PARAM}={h['token']}"
    r = asyncio.run(watch(url, a.session, a.seconds))
    r.pop("latency_ms")
    print(json.dumps(r))
    return 0 if not r["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
