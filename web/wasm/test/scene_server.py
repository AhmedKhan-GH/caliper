"""A script that logs demo 0's kind of scene through caliper.serve(), for the
headless check of a scene in the browser (scene.mjs).

The scene "landscape" (3-D, +z up) holds a static box of segments at its root,
a mesh over a 41 x 41 grid coloured through viridis on the range -4 to 2, a
line strip down the bowl's far wall and a red marker at its end, all from CPU
tensors. The script logs the stamped batches again every 0.25 s, as a training
loop does, so a viewer that attaches at any moment draws them. It prints
"READY <page link>" once it serves, and on the line "update" from its standard
input it shifts the surface's values by 2.5, which recolours it; it ends when
its standard input closes. Run with Caliper LIB on PYTHONPATH."""
from __future__ import annotations

import os
import sys
import threading
import time

import torch

import caliper

N = 41


def height(x, y):
    r = torch.sqrt(x * x + y * y)
    return 0.8 * torch.clamp(r - 0.35, min=0.0) ** 2 - 0.5


def surface(shift):
    g = torch.linspace(-1.0, 1.0, N)
    y, x = torch.meshgrid(g, g, indexing="ij")
    z = height(x, y)
    positions = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    values = (z.reshape(-1) * 4.0 + shift).clamp(-4.0, 2.0)
    i = torch.arange(N - 1)
    a = (i[:, None] * N + i[None, :]).reshape(-1)
    tri = torch.stack([torch.stack([a, a + 1, a + N + 1], -1), torch.stack([a, a + N + 1, a + N], -1)], 1)
    return positions, tri.reshape(-1, 3).to(torch.int32), values


def main():
    os.environ.setdefault("CALIPER_SESSION", "0")
    s = caliper.serve(port=0)
    scene = caliper.scene("landscape", up="+z", units="")
    lo, hi = (-1.05, -1.05, -0.6), (1.05, 1.05, 0.4)
    corner = lambda c: [hi[k] if c >> k & 1 else lo[k] for k in range(3)]   # noqa: E731
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    scene.lines("", torch.tensor([corner(c) for e in edges for c in e]), colors=(180, 180, 180), width=3.0,
                static=True)
    t = torch.linspace(0.0, 1.0, 30)
    x, y = -0.9 + 0.3 * t, 0.9 - 0.35 * t
    path = torch.stack([x, y, height(x, y) + 0.04], -1)
    shift = [0.0]
    done = threading.Event()

    def commands():
        for line in sys.stdin:
            if line.strip() == "update":
                shift[0] = 2.5
        done.set()
    threading.Thread(target=commands, daemon=True).start()
    print("READY", s.url, flush=True)
    while not done.is_set():
        positions, tri, values = surface(shift[0])
        scene.mesh("surface", positions, tri, values=values, cmap="viridis", vmin=-4.0, vmax=2.0)
        scene.lines("path", path, strip_lengths=[len(path)], colors=(235, 235, 235), width=3.0)
        scene.points("current", path[-1:], colors=(255, 80, 80))
        time.sleep(0.25)
    s.close()


if __name__ == "__main__":
    main()
