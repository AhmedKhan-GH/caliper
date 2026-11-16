"""A stand-in for Caliper's stream, used by the harness self-test.

`MockStreamServer` runs a WebSocket server on 127.0.0.1 in a background
thread of the kernel and pushes one binary frame to every client at a fixed
rate (30 Hz by default). `MockCanvas` is an anywidget whose ESM connects to
that server directly and draws each frame on a canvas inside a shadow root.

This is the situation sprint 2 needs: a widget in a notebook cell fed by a
WebSocket from the kernel process over loopback, not by Jupyter's message
channel, so it keeps drawing while a later cell runs a long loop.

Frame layout (little-endian): uint32 frame index, uint32 width, uint32
height, uint32 reserved, then width * height uint8 values.

Text messages from the browser are JSON objects. The widget sends
``{"received": n}`` every second and ``{"webgpu": {...}}`` once.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import anywidget
import numpy as np
import traitlets
from websockets.asyncio.server import broadcast, serve

HEADER = struct.Struct("<IIII")


class MockStreamServer:
    """WebSocket server on 127.0.0.1 that pushes frames from its own thread."""

    def __init__(self, rate: float = 30.0, size: int = 64):
        self.rate = rate
        self.size = size
        self.port: int | None = None
        self.frames_generated = 0
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self.connections: list[dict] = []
        self._clients: set = set()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mock-stream", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("mock stream server did not start")
        if self._error:
            raise RuntimeError("mock stream server failed") from self._error

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/mock"

    def frame(self, index: int) -> bytes:
        """Build frame `index`: a pattern that moves with the frame index."""
        n = self.size
        y, x = np.mgrid[0:n, 0:n].astype(np.float32)
        t = 0.25 * index
        v = np.sin(0.21 * x + 0.13 * y - t) * np.cos(0.17 * y - 0.6 * t) + 0.35 * np.sin(0.09 * (x + y) + t)
        field = np.clip((v + 1.35) / 2.7 * 255.0, 0, 255).astype(np.uint8)
        return HEADER.pack(index, n, n, 0) + field.tobytes()

    def report(self) -> dict:
        """What the server saw: its address, the frame count and every connection."""
        span = (self._last_frame_at or 0.0) - (self._first_frame_at or 0.0)
        return {
            "url": self.url,
            "rate": self.rate,
            "frames_generated": self.frames_generated,
            "measured_rate": (self.frames_generated - 1) / span if span > 0 else None,
            "connections": [dict(c) for c in self.connections],
        }

    def write_report(self, path: str | Path) -> dict:
        report = self.report()
        Path(path).write_text(json.dumps(report, indent=2))
        return report

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # reported to the constructor if it happens at startup
            self._error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _main(self) -> None:
        async with serve(self._handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            period = 1.0 / self.rate
            next_time = time.monotonic()
            while not self._stop.is_set():
                if self._clients:
                    broadcast(self._clients, self.frame(self.frames_generated))
                    self.frames_generated += 1
                    self._last_frame_at = time.monotonic()
                    if self._first_frame_at is None:
                        self._first_frame_at = self._last_frame_at
                next_time += period
                await asyncio.sleep(max(0.0, next_time - time.monotonic()))

    async def _handler(self, connection) -> None:
        request = connection.request
        info = {
            "path": request.path,
            "origin": request.headers.get("Origin"),
            "host": request.headers.get("Host"),
            "remote": connection.remote_address[0],
            "user_agent": request.headers.get("User-Agent", ""),
            "received": 0,
            "webgpu": None,
        }
        self.connections.append(info)
        self._clients.add(connection)
        try:
            async for message in connection:
                if isinstance(message, str):
                    data = json.loads(message)
                    if "received" in data:
                        info["received"] = int(data["received"])
                    if "webgpu" in data:
                        info["webgpu"] = data["webgpu"]
        finally:
            self._clients.discard(connection)


class MockCanvas(anywidget.AnyWidget):
    """Draws the frames of a MockStreamServer; `src` is the server's ws:// URL."""

    _esm = Path(__file__).with_name("widget.js")
    src = traitlets.Unicode("").tag(sync=True)
    height = traitlets.Int(240).tag(sync=True)
