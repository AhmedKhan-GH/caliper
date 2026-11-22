"""The cost of a view call with nothing attached, as the cost table of
docs/development/performance.md measures it: 100,000 calls per row, each timed
singly with perf_counter_ns, median and p99 in microseconds.

    PYTHONPATH=python .venv/bin/python python/bench/bench_call_cost.py [--calls 100000] [--scenes]

A network route is open (caliper.serve(), the Python sender) with no viewer,
so every call returns False after its check. The rows: caliper.show of a
1024 x 1024 MPS tensor (or 256 x 256 on the CPU), caliper.series(1.0),
caliper.points and caliper.histogram; with --scenes, the three scene calls of
demo 0 (the 41 x 41 surface mesh with values, the path strip with one colour
and the one-point marker). Prints one line per row and the 1-minute load.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("CALIPER_SESSION", "0")
os.environ.setdefault("CALIPER_NETIO", "0")

import torch  # noqa: E402

import caliper  # noqa: E402


def measure(fn, n: int) -> tuple[float, float]:
    pc = time.perf_counter_ns
    ts = [0] * n
    for i in range(n):
        a = pc()
        r = fn()
        ts[i] = pc() - a
        if r is not False:
            raise SystemExit(f"a call took a snapshot ({r!r}): something is attached")
    ts.sort()
    return statistics.median(ts) / 1e3, ts[int(n * 0.99)] / 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calls", type=int, default=100_000)
    ap.add_argument("--scenes", action="store_true")
    args = ap.parse_args()
    mps = torch.backends.mps.is_available()
    dev = "mps" if mps else "cpu"
    s = caliper.serve(port=0)
    try:
        side = 1024 if mps else 256
        x = torch.rand(side, side, device=dev)
        z = torch.rand(1000, 3, device=dev)
        rows = [(f"show {side}x{side} {dev}", lambda: caliper.show(x, "x")),
                ("series(1.0)", lambda: caliper.series(1.0, "s")),
                (f"points 1000x3 {dev}", lambda: caliper.points(z, None, "p")),
                (f"histogram {side}x{side} {dev}", lambda: caliper.histogram(x, "h"))]
        if args.scenes:
            sc = caliper.scene("landscape", up="+z", units="")
            pos = torch.rand(1681, 3, device=dev)
            tri = torch.randint(0, 1681, (3200, 3), device=dev, dtype=torch.int32)
            vals = torch.rand(1681, device=dev)
            path = torch.rand(200, 3, device=dev)
            cur = torch.rand(1, 3, device=dev)
            rows += [
                (f"scene.mesh 1681 vertices {dev}",
                 lambda: sc.mesh("surface", pos, tri, values=vals, cmap="viridis", vmin=-4.0, vmax=2.0)),
                (f"scene.lines strip of 200 {dev}",
                 lambda: sc.lines("path", path, strip_lengths=[200], colors=(235, 235, 235), rate=10)),
                (f"scene.points 1 point {dev}",
                 lambda: sc.points("current", cur, colors=(255, 80, 80), rate=10)),
            ]
        for name, fn in rows:
            for _ in range(2000):                        # warm-up
                fn()
            med, p99 = measure(fn, args.calls)
            print(f"{name:32s} median {med:7.3f} us   p99 {p99:7.3f} us", flush=True)
        print(f"load average {os.getloadavg()[0]:.2f} (1 min), torch {torch.__version__}", flush=True)
    finally:
        s.close()


if __name__ == "__main__":
    main()
