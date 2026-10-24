"""Render the example's four live views to PNG frames without a screen.

    .venv/bin/python python/bench/render_frames.py [--out media] [--seconds 8] [--every 1.0]

Runs examples/python/train_show.py's Trainer with caliper.window(offscreen=True)
(the in-process render thread on an offscreen canvas: the same view layer and
drawing code as the window and the viewer) and saves caliper.render() every
`every` seconds as <out>/offscreen-frame-NN.png, plus the last frame as
<out>/offscreen-four-views.png. Pure-Python PNG writer (no Pillow).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import sys
import time
import zlib

import numpy as np
import torch

import caliper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def write_png(path: str, rgba: np.ndarray) -> None:
    h, w, c = rgba.shape
    assert c == 4 and rgba.dtype == np.uint8
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(raw, 6)) +
           chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def load_example():
    spec = importlib.util.spec_from_file_location(
        "train_show", os.path.join(ROOT, "examples", "python", "train_show.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "media"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--size", default="1600x1000")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    w, h = (int(v) for v in args.size.split("x"))

    ex = load_example()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tr = ex.Trainer(device)
    win = caliper.window(w, h, "caliper", offscreen=True)
    print(f"device {device}, offscreen canvas {w}x{h}")

    tr.phase(0.5, "off")
    tr.phase(0.5, "caliper")       # views exist and have their first snapshots
    t0 = time.perf_counter()
    next_shot = t0 + args.every
    n = 0
    steps = 0
    while time.perf_counter() - t0 < args.seconds:
        steps += int(tr.phase(0.25, "caliper")[0])
        if time.perf_counter() >= next_shot:
            frame = np.asarray(caliper.render())
            path = os.path.join(args.out, f"offscreen-frame-{n:02d}.png")
            write_png(path, frame)
            print(f"  {path}  ({frame.shape[1]}x{frame.shape[0]}, step {steps})")
            n += 1
            next_shot += args.every
    frame = np.asarray(caliper.render())
    write_png(os.path.join(args.out, "offscreen-four-views.png"), frame)
    print(f"stats: {win.stats()}")
    for name in ("fc1.weight", "conv1.act", "embedding", "loss"):
        try:
            print(f"  {name}: {caliper.live_view_info(name)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: {e}")
    caliper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
