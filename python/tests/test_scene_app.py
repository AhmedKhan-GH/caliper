"""The Caliper app draws a scene that caliper.serve() sends (the scenes of the
wire protocol 3.2 draft), end to end, with nothing on screen.

A script logs demo 0's kind of scene with CPU tensors: a static box of segments
at the scene's root, a mesh over a 41 x 41 grid coloured through viridis, the
optimizer's path as a line strip and the current point as a red marker. A
websockets viewer is subscribed while the stamped batches are logged, so the
producer holds them; the app then attaches late with ``caliper --hidden --url``
and receives the scene's state (the reset, the static box and the held
batches). The test reads the scene view's data area and camera from the app's
UI driver (``where``), takes the window's frame from Caliper API
(``GET /v0/capture.png``), and checks that it is the scene: not blank, the
marker's colour at the marker's projected pixel, the box's colour on a box
edge, and the colour of the surface's flat centre equal to the colormap of the
centre value, lit as a flat face is. One mesh update then recolours the
surface, and the app draws it.

It needs the app (``CALIPER_TEST_CALIPER_EXE``) and Metal; it skips without
either. ``CALIPER_SCENE_CAPTURE_DIR`` keeps the two frames as
scene-view-app.png and scene-view-update.png."""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib

import numpy as np
import pytest
import torch

import caliper
from caliper import _scene, _wire as w

EXE = os.environ.get("CALIPER_TEST_CALIPER_EXE", "")
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not EXE or not os.path.exists(EXE),
                                reason="needs the Caliper app (CALIPER_TEST_CALIPER_EXE) on macOS")
websockets = pytest.importorskip("websockets")
from websockets.sync.client import connect  # noqa: E402

N = 41
MARKER = (255, 80, 80)
BOX = (180, 180, 180)
PATH = (235, 235, 235)
VMIN, VMAX = -4.0, 2.0
WINDOW = (1280, 800)


# ---------------------------------------------------------------------------
# the reference: viridis as tensor_bridge.cpp makes it, the index rule of
# map_f32_to_rgba8, and a PNG reader
# ---------------------------------------------------------------------------
VIRIDIS = [
    (0.2777273272234177, 0.005407344544966578, 0.3340998053353061),
    (0.1050930431085774, 1.404613529898575, 1.384590162594685),
    (-0.3308618287255563, 0.214847559468213, 0.09509516302823659),
    (-4.634230498983486, -5.799100973351585, -19.33244095627987),
    (6.228269936347081, 14.17993336680509, 56.69055260068105),
    (4.776384997670288, -13.74514537774601, -65.35303263337234),
    (-5.435455855934631, 4.645852612178535, 26.3124352495832),
]


def viridis(i: int) -> tuple:
    f = np.float32
    t = f(i) / f(255)
    c = [f(x) for x in VIRIDIS[0]]
    tp = f(1)
    for k in range(1, 7):
        tp = f(tp * t)
        c = [f(c[j] + f(f(VIRIDIS[k][j]) * tp)) for j in range(3)]
    return tuple(int(min(max(x, f(0)), f(1)) * f(255) + f(0.5)) for x in c)


def index_of(v: float, lo: float, hi: float) -> int:
    t = (np.float32(v) - np.float32(lo)) / (np.float32(hi) - np.float32(lo))
    return int(min(max(float(t), 0.0), 1.0) * 255.0 + 0.5)


def read_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w_, h_, ctype = 8, b"", 0, 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w_, h_, depth, ctype = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and ctype in (2, 6)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    ch = 4 if ctype == 6 else 3
    raw = zlib.decompress(idat)
    stride = w_ * ch
    out = np.zeros((h_, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    for y in range(h_):
        f = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        cur = np.zeros(stride, np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        else:
            for x in range(stride):
                a = cur[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + b) // 2
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else b if pb <= pc else c
                cur[x] = (line[x] + p) & 255
        out[y] = cur
        prev = cur
    return out.reshape(h_, w_, ch)[:, :, :3]


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------
def height(x, y):
    """A bowl with a flat floor of radius 0.35 around the centre."""
    r = torch.sqrt(x * x + y * y)
    return 0.8 * torch.clamp(r - 0.35, min=0.0) ** 2 - 0.5


def surface(shift: float = 0.0):
    g = torch.linspace(-1.0, 1.0, N)
    y, x = torch.meshgrid(g, g, indexing="ij")
    z = height(x, y)
    positions = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    values = (z.reshape(-1) * 4.0 + shift).clamp(VMIN, VMAX)   # the floor's value is -2 + shift
    i = torch.arange(N - 1)
    a = (i[:, None] * N + i[None, :]).reshape(-1)
    tri = torch.stack([torch.stack([a, a + 1, a + N + 1], -1), torch.stack([a, a + N + 1, a + N], -1)], 1)
    return positions, tri.reshape(-1, 3).to(torch.int32), values


def frame_box():
    lo, hi = (-1.05, -1.05, -0.6), (1.05, 1.05, 0.4)
    corner = lambda c: [hi[k] if c >> k & 1 else lo[k] for k in range(3)]   # noqa: E731
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return torch.tensor([corner(c) for e in edges for c in e], dtype=torch.float32)


def path_points():
    t = torch.linspace(0.0, 1.0, 30)
    x, y = -0.9 + 0.3 * t, 0.9 - 0.35 * t      # down the far wall, which faces the camera
    z = height(x, y) + 0.04
    return torch.stack([x, y, z], -1)


# ---------------------------------------------------------------------------
# the camera of the scene view, as the app reports it
# ---------------------------------------------------------------------------
class Where:
    def __init__(self, line: str):
        f = line.split()
        i = f.index("area")
        self.area = [float(v) for v in f[i + 1:i + 5]]
        self.scale = float(f[f.index("scale") + 1])
        t = f.index("target")
        self.target = np.array([float(v) for v in f[t + 1:t + 4]])
        self.distance = float(f[f.index("distance") + 1])
        self.azimuth = float(f[f.index("azimuth") + 1])
        self.elevation = float(f[f.index("elevation") + 1])
        self.up = f[f.index("up") + 1]
        self.fovy = float(f[f.index("fovy") + 1])
        p = f.index("target_px")
        self.target_px = (int(f[p + 1]), int(f[p + 2]))

    def direction(self):
        assert self.up == "+z"
        u, a = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
        b = np.cross(a, u)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        return (a * math.cos(self.azimuth) + b * math.sin(self.azimuth)) * ce + u * se

    def pixel(self, p, frame_scale: float):
        """The frame pixel (x, y) where world point p is drawn."""
        eye = self.target + self.direction() * self.distance
        fwd = (self.target - eye) / np.linalg.norm(self.target - eye)
        side = np.cross(fwd, [0.0, 0, 1])
        side /= np.linalg.norm(side)
        up = np.cross(side, fwd)
        d = np.asarray(p, float) - eye
        xv, yv, zv = d @ side, d @ up, -(d @ fwd)
        aspect = self.target_px[0] / self.target_px[1]
        f = 1.0 / math.tan(self.fovy / 2)
        nx, ny = f / aspect * xv / -zv, f * yv / -zv
        ax, ay, aw, ah = self.area
        return ((ax + (nx * 0.5 + 0.5) * aw) * frame_scale, (ay + (0.5 - ny * 0.5) * ah) * frame_scale)


def near(img, xy, rgb, tol, r=1):
    """The smallest largest-channel difference from rgb within r pixels of xy, and that pixel."""
    x0, y0 = int(xy[0]), int(xy[1])
    best = (999, None)
    for y in range(y0 - r, y0 + r + 1):
        for x in range(x0 - r, x0 + r + 1):
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                d = int(np.abs(img[y, x].astype(int) - np.array(rgb)).max())
                if d < best[0]:
                    best = (d, tuple(int(c) for c in img[y, x]))
    return best[0] <= tol, best


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
class Subscriber(threading.Thread):
    """A viewer with the capability scenes, subscribed to the scene, so that the producer
    holds the newest batch of each entity part for a viewer that attaches later."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.stop, self.updates = url, threading.Event(), []
        self.ready = threading.Event()

    def run(self):
        with connect(self.url, max_size=None, compression=None) as ws:
            ws.send(w.encode(w.Hello("viewer", [w.CAP_SCENES, w.CAP_SCENE_PIECES])))
            subscribed = False
            while not self.stop.is_set():
                try:
                    m = w.decode(ws.recv(timeout=0.1))
                except TimeoutError:
                    continue
                if isinstance(m, w.Views) and not subscribed and any(v["name"] == "landscape" for v in m.views):
                    ws.send(w.encode(w.Subscribe(1, "landscape", 7, None, None)))
                    subscribed = True
                elif isinstance(m, w.ViewReply):
                    self.ready.set()
                elif isinstance(m, w.SceneUpdate):
                    self.updates.append((m.path, m.part))


class App:
    def __init__(self, url, data, ui):
        env = dict(os.environ, CALIPER_DATA_DIR=os.path.join(data, "data"),
                   CALIPER_USER_INSTRUMENTS_DIR=os.path.join(data, "user"), CALIPER_TEST_UI=ui,
                   CALIPER_WINDOW_SIZE=f"{WINDOW[0]}x{WINDOW[1]}")
        os.makedirs(env["CALIPER_DATA_DIR"])
        os.makedirs(env["CALIPER_USER_INSTRUMENTS_DIR"])
        self.data = env["CALIPER_DATA_DIR"]
        self.p = subprocess.Popen([EXE, "--hidden", "--url", url, "--api", "--api-port", "0"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.lines: list[str] = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.p.stderr:
            self.lines.append(line.rstrip("\n"))

    def wait_for(self, pred, timeout=60.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for line in list(self.lines):
                if pred(line):
                    return line
            if self.p.poll() is not None:
                break
            time.sleep(0.05)
        return None

    def capture(self) -> bytes:
        with open(os.path.join(self.data, "api.json")) as f:
            api = json.load(f)
        url = f"http://127.0.0.1:{api['port']}/v0/capture.png?token={api['token']}"
        with urllib.request.urlopen(url, timeout=20) as r:
            assert r.status == 200
            return r.read()

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(20)
            except subprocess.TimeoutExpired:
                self.p.kill()
        return self.p.returncode


def keep(name, png):
    folder = os.environ.get("CALIPER_SCENE_CAPTURE_DIR")
    if folder:
        with open(os.path.join(folder, name), "wb") as f:
            f.write(png)


def test_the_app_draws_a_scene_from_caliper_serve(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "0")
    s = caliper.serve(port=0)
    tmp = tempfile.mkdtemp(prefix="caliper-scene-app-")
    app = sub = None
    try:
        scene = caliper.scene("landscape", up="+z", units="")
        assert scene.lines("", frame_box(), colors=BOX, width=3.0, static=True)   # before anyone attaches

        sub = Subscriber(s.stream_url)
        sub.start()
        assert sub.ready.wait(10), "the websockets viewer did not subscribe"
        positions, tri, values = surface()
        assert scene.mesh("surface", positions, tri, values=values, cmap="viridis", vmin=VMIN, vmax=VMAX)
        path = path_points()
        assert scene.lines("path", path, strip_lengths=[len(path)], colors=PATH, width=3.0)
        assert scene.points("current", path[-1:], colors=MARKER)
        end = time.monotonic() + 10
        while time.monotonic() < end and {"surface", "path", "current"} - {p for p, _ in sub.updates}:
            time.sleep(0.05)

        # A late attach: the app receives the reset, the static box and the batches held.
        app = App(s.stream_url, tmp, "where landscape; wait 40; where landscape; wait 400")
        renderer = app.wait_for(lambda l: l.startswith("[renderer] "), 60)
        if renderer is None or "metal" not in renderer:
            pytest.skip(f"the app has no Metal renderer here ({renderer})")
        lines = []
        end = time.monotonic() + 60
        while time.monotonic() < end and len(lines) < 2:
            lines = [l for l in app.lines if l.startswith("[ui] where landscape ")]
            time.sleep(0.05)
        assert len(lines) == 2, "\n".join(app.lines[-40:])
        where = Where(lines[1])
        assert where.target_px[0] > 100 and where.target_px[1] > 100

        png = app.capture()
        keep("scene-view-app.png", png)
        img = read_png(png)
        k = img.shape[1] / WINDOW[0]                                   # framebuffer pixels per point
        ax, ay, aw, ah = (v * k for v in where.area)
        area = img[int(ay):int(ay + ah), int(ax):int(ax + aw)]
        assert len({tuple(c) for c in area[::7, ::7].reshape(-1, 3)}) > 20, "the scene view is blank"

        # The marker, unlit and flat: its colour at its projected pixel.
        ok, got = near(img, where.pixel(path[-1].tolist(), k), MARKER, 1)
        assert ok, f"marker: {got}"
        # The box: its colour on the middle of its top front edge.
        top = frame_box()
        ok, got = near(img, where.pixel(((top[8] + top[9]) / 2).tolist(), k), BOX, 2, r=2)
        assert ok, f"box: {got}"
        # The surface's flat floor at the centre: viridis at the floor's value, lit as a face
        # whose normal is the up axis (0.30 + 0.70 |n.z| in view space, so sin(elevation)).
        def floor_colour(shift):
            v = float(values.new_tensor(-2.0 + shift).clamp(VMIN, VMAX))
            lit = 0.30 + 0.70 * abs(math.sin(where.elevation))
            return tuple(int(round(c / 255.0 * lit * 255.0)) for c in viridis(index_of(v, VMIN, VMAX)))
        centre = where.pixel([0.0, 0.0, float(height(torch.tensor(0.0), torch.tensor(0.0)))], k)
        ok, got = near(img, centre, floor_colour(0.0), 3)
        assert ok, f"surface centre: {got}, expected {floor_colour(0.0)}"

        # One mesh update of the same size recolours the surface, and the app draws it.
        positions, tri, values = surface(shift=2.5)
        assert scene.mesh("surface", positions, tri, values=values, cmap="viridis", vmin=VMIN, vmax=VMAX)
        end, ok = time.monotonic() + 15, False
        while time.monotonic() < end and not ok:
            png = app.capture()
            img = read_png(png)
            ok, got = near(img, centre, floor_colour(2.5), 3)
        keep("scene-view-update.png", png)
        assert ok, f"surface centre after the update: {got}, expected {floor_colour(2.5)}"
        assert near(img, where.pixel(path[-1].tolist(), k), MARKER, 1)[0]
    finally:
        if app:
            code = app.close()
            assert code == 0, "\n".join(app.lines[-40:])
        if sub:
            sub.stop.set()
            sub.join(5)
        s.close()
        for name in list(_scene.SCENES):
            caliper.remove(name)
        shutil.rmtree(tmp, ignore_errors=True)
