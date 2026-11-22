"""A producer of wire protocol 3.2 that serves the fixture of the pixel comparison to a viewer, for
the live path of the browser module (web/test/draw_parity.spec.ts).

    python serve_fixture.py --fixture <folder of caliper_draw_parity_app>

It listens on 127.0.0.1 at a free port and prints one JSON line, {"url": "ws://..."}. A viewer that
says HELLO gets WELCOME with the capabilities scenes and resources, SESSIONS, and VIEWS with one
empty 3-D scene view for each panel of the fixture ("pattern" and "tower"), then the fixture's
frames as they are: the instance form of PROGRAM set for each program and RESOURCE for each
version, as a script relays them to a viewer that announced resources
(design/programmable-graphics.md, Resource Message). A SUBSCRIBE to a scene view gets its
VIEW_REPLY and a reset of the empty scene. Needs numpy and websockets, and caliper._wire from
python/ (PYTHONPATH).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path

import websockets

from caliper import _wire as W

PANELS = ("pattern", "tower")


def scene_view(view_id: int, name: str) -> dict:
    return {"id": view_id, "name": name, "kind": 6, "sources": [], "scene": {"space": "3d", "up": "+y", "units": "m"}}


def frames_of(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, at = [], 0
    while at < len(data):
        (n,) = struct.unpack_from("<I", data, at)
        at += 4
        out.append(data[at:at + n])
        at += n
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--fixture", type=Path, required=True)
    args = ap.parse_args()
    frames = frames_of(args.fixture / "frames.bin")
    identity = {"id": "fixture", "script": "draw_parity_fixture", "pid": os.getpid(),
                "start_unix_ms": int(time.time() * 1000)}
    views = [scene_view(100 + i, name) for i, name in enumerate(PANELS)]
    caps = [W.CAP_EVENTS, W.CAP_SCENES, W.CAP_RESOURCES]

    async def handler(ws):
        lock = asyncio.Lock()

        async def send(msg):
            async with lock:
                await ws.send(W.encode(msg, minor=W.MINOR_3_2))

        first = W.decode(await asyncio.wait_for(ws.recv(), W.HANDSHAKE_TIMEOUT_MS / 1000))
        if not isinstance(first, W.Hello) or first.role != W.ROLE_VIEWER:
            await ws.close(W.CLOSE_ROLE_REFUSED)
            return
        await send(W.Welcome(W.ROLE_PRODUCER, caps, {"name": "serve_fixture", "version": "0"}, identity))
        await send(W.Sessions([dict(identity, state=W.SESSION_LIVE, views=len(views), capabilities=caps)]))
        await send(W.Views(views, identity["id"]))
        if W.CAP_RESOURCES in first.capabilities:
            async with lock:
                for f in frames:
                    await ws.send(f)
        try:
            async for raw in ws:
                msg = W.decode(raw)
                if isinstance(msg, W.Subscribe):
                    v = next((x for x in views if x["name"] == msg.view), None)
                    if v is None:
                        await send(W.ViewReply(msg.request_id, msg.sub, msg.view,
                                               error={"code": W.ERR_UNKNOWN_VIEW, "message": msg.view}))
                        continue
                    dv = W.resolve_data_view(v, msg.data_view, retain=0)
                    await send(W.ViewReply(msg.request_id, msg.sub, msg.view, data_view=dv))
                    await send(W.SceneClear(msg.sub, msg.view, 1, time.monotonic_ns(), reset=True, state=True))
                elif isinstance(msg, W.Ping):
                    await send(W.Pong(msg.nonce))
                elif isinstance(msg, W.Bye):
                    await ws.close(W.CLOSE_NORMAL)
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(handler, "127.0.0.1", 0, max_size=None) as server:
        port = server.sockets[0].getsockname()[1]
        print(json.dumps({"url": f"ws://127.0.0.1:{port}/v3/stream"}), flush=True)
        await asyncio.Future()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
