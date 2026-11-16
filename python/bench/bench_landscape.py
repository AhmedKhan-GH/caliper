"""The cost row of demo 0, the loss landscape (demos/loss_landscape), on MPS:
the notebook's training run with the landscape at the default N of 500
against the same run without it, the surface's own time and its peak memory,
vmap against the loop over perturbed copies, and the per-step cost of the
path and marker calls.

    PYTHONPATH=python .venv/bin/python python/bench/bench_landscape.py [--steps 8000] [--repeats 2]

Each training run is a fresh process that runs the notebook's own code cells
(all but the training run, the random-directions cell and the close) with a
viewer attached in a process of its own, which subscribes to the scene and
the series and discards what it receives, as a notebook widget would read
them. The runs alternate, with and without the landscape. Without the
landscape the loop is the same, with land.point and land.due made no-ops. The
budget of design/demos.md is at most 1.5 times as long with the landscape.
Prints one line per row with the 1-minute load average beside it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "demos" / "loss_landscape" / "loss_landscape.ipynb"


def load() -> float:
    return os.getloadavg()[0]


def notebook(steps: int) -> dict:
    """The notebook's namespace after its setup cells."""
    os.environ["CALIPER_SESSION"] = "0"
    os.environ["CALIPER_NOTEBOOK_STEPS"] = str(steps)
    sys.path.insert(0, str(NOTEBOOK.parent))
    ns = {"__name__": "loss_landscape"}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code" and cell["id"] not in ("widgets", "run", "random", "close"):
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec"), ns)
    return ns


def viewer(url: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, __file__, "--drain", url],
                            env={**os.environ, "PYTHONPATH": str(REPO / "python")})


def drain(url: str) -> None:
    """A viewer that subscribes to every view and discards what it receives."""
    from websockets.asyncio.client import connect

    from caliper import _wire as w

    async def go():
        async with connect(url, max_size=None, compression=None, ping_interval=None) as ws:
            await ws.send(w.encode(w.Hello("viewer", [w.CAP_SCENES, w.CAP_CONTROLS],
                                           {"name": "bench", "version": "0"})))
            subscribed = set()
            async for raw in ws:
                m = w.decode(raw)
                if isinstance(m, w.Views):
                    for v in m.views:
                        if v["name"] not in subscribed:
                            subscribed.add(v["name"])
                            sub = len(subscribed)
                            await ws.send(w.encode(w.Subscribe(sub, v["name"], sub, None, None)))
    try:
        asyncio.run(go())
    except Exception:  # noqa: BLE001  the server closed
        pass


def run(variant: str, steps: int) -> dict:
    """One training run of the notebook; prints a JSON line."""
    import torch
    ns = notebook(steps)
    assert ns["device"].type == "mps", "the cost row runs on MPS"
    land = ns["land"]
    view = viewer(ns["server"].stream_url)
    time.sleep(1.0)                                         # the viewer subscribes
    calls = []
    if variant == "without":
        land.point = lambda model, step: None
        land.due = lambda step, every, directions: False
    else:
        show_path = ns["show_path"]

        def timed_path(rate=10):
            t = time.perf_counter_ns()
            show_path(rate)
            calls.append(time.perf_counter_ns() - t)
        ns["show_path"] = timed_path
    ns["train"](20)                                         # warm-up: kernels, the first batches
    torch.mps.synchronize()
    load0 = load()
    t0 = time.perf_counter()
    ns["train"](steps)
    torch.mps.synchronize()
    seconds = time.perf_counter() - t0
    out = {"variant": variant, "steps": steps, "seconds": seconds, "load": [load0, load()]}
    if variant == "with":
        calls.sort()
        out["path_calls_us"] = {"median": statistics.median(calls) / 1e3,
                                "p99": calls[int(len(calls) * 0.99)] / 1e3,
                                "mean": sum(calls) / len(calls) / 1e3, "count": len(calls),
                                "slowest": [c / 1e3 for c in calls[-5:]]}
        out["refits"] = [s for s, _ in land.refits]
        out["method"] = land.method
        out["share"] = float(land.share)
        out["snapshots"] = len(land.snapshot_steps)
    ns["server"].close()
    view.wait(10)
    print(json.dumps(out), flush=True)
    return out


def picture(steps: int) -> None:
    """What the picture shows over a run: at each refit in the path's plane,
    the step, the plane share and the loss; at the end, how far the weights
    moved from the anchor and where that lands in random directions."""
    import torch
    ns = notebook(steps)
    land = ns["land"]
    rows = []
    show_surface = ns["show_surface"]

    def recording():
        show_surface()
        rows.append((land.step, land.share, land._height))
    ns["show_surface"] = recording
    ns["train"](steps)
    out = {"refits": [(s, float(sh), 10 ** (2 * float(h))) for s, sh, h in rows]}
    theta = land.flat(ns["model"])
    d = land.random_directions
    coef = land._random_proj @ (theta - land.anchor)
    out["moved"] = float((theta - land.anchor).norm() / land.anchor.norm())
    out["random_coords"] = coef.tolist()
    resid = (theta - land.anchor) - coef @ d
    out["random_in_plane"] = float(1 - resid.norm() ** 2 / (theta - land.anchor).norm() ** 2)
    out["path_box"] = [float(x) for x in land.path[:, :2].abs().amax(0)]
    ns["server"].close()
    del torch
    print(json.dumps(out), flush=True)


def surface(steps: int) -> None:
    """The surface's own time, synchronised, in both modes and with the loop,
    and its peak memory above what training holds."""
    import torch
    ns = notebook(steps)
    land, model, span = ns["land"], ns["model"], ns["span"]
    due = land.due
    land.due = lambda step, every, directions: False         # no surface while training
    ns["train"](steps)
    land.due = due
    torch.mps.synchronize()
    held0 = torch.mps.current_allocated_memory()

    def timed(mode, n):
        out = []
        for _ in range(n):
            torch.mps.synchronize()
            t = time.perf_counter()
            land.refit(model, span.value, mode)
            torch.mps.synchronize()
            out.append(time.perf_counter() - t)
        return out
    rows = {"load": [load()]}
    rows["first path s"] = timed("path", 1)
    # The peak: the allocator's allocated bytes, sampled after every module's
    # forward during one surface, above what was held before it.
    peak = [held0]
    hooks = [m.register_forward_hook(
        lambda *a: peak.__setitem__(0, max(peak[0], torch.mps.current_allocated_memory())))
        for m in model.modules()]
    timed("path", 1)
    for h in hooks:
        h.remove()
    rows["peak MB"] = (peak[0] - held0) / 1e6
    rows["held MB"] = held0 / 1e6
    rows["snapshots"] = len(land.snapshot_steps)
    rows["vmap path s"] = timed("path", 5)
    rows["vmap random s"] = timed("random", 5)
    land.method = "loop"
    timed("path", 1)
    rows["loop path s"] = timed("path", 3)
    rows["load"].append(load())
    ns["server"].close()
    print(json.dumps(rows), flush=True)


def child(args: list[str]) -> dict:
    p = subprocess.run([sys.executable, __file__] + args, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO / "python")}, timeout=1800)
    if p.returncode:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"{args} failed")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--drain")
    ap.add_argument("--run", choices=["with", "without"])
    ap.add_argument("--surface", action="store_true")
    ap.add_argument("--picture", action="store_true")
    args = ap.parse_args()
    if args.drain:
        return drain(args.drain)
    if args.run:
        return run(args.run, args.steps) and None
    if args.surface:
        return surface(600)
    if args.picture:
        return picture(args.steps)
    times = {"with": [], "without": []}
    for i in range(args.repeats):
        for variant in ("without", "with") if i % 2 == 0 else ("with", "without"):
            r = child(["--run", variant, "--steps", str(args.steps)])
            times[variant].append(r["seconds"])
            extra = ""
            if variant == "with":
                c = r["path_calls_us"]
                extra = (f"  path+marker calls median {c['median']:.1f} us, p99 {c['p99']:.1f} us,"
                         f" mean {c['mean']:.1f} us over {c['count']}, slowest "
                         + ", ".join(f"{x:.0f}" for x in c["slowest"])
                         + f" us; {len(r['refits'])} refits, {r['method']}, share {r['share']:.3f}")
            print(f"{variant:8s} {args.steps} steps {r['seconds']:7.2f} s "
                  f"(load {r['load'][0]:.1f} to {r['load'][1]:.1f}){extra}", flush=True)
    ratio = statistics.median(times["with"]) / statistics.median(times["without"])
    print(f"ratio with / without: {ratio:.3f} (budget 1.5)", flush=True)
    s = child(["--surface"])
    fmt = lambda xs: ", ".join(f"{x:.3f}" for x in xs)  # noqa: E731
    print(f"surface: first {fmt(s['first path s'])} s; vmap path {fmt(s['vmap path s'])} s;"
          f" vmap random {fmt(s['vmap random s'])} s;"
          f" loop path {fmt(s['loop path s'])}"
          + f" s; surface peak {s['peak MB']:.0f} MB above the {s['held MB']:.0f} MB held "
          f"(load {s['load'][0]:.1f} to {s['load'][1]:.1f})", flush=True)


if __name__ == "__main__":
    main()
