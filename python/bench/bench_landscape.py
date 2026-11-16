"""The cost row of demo 0, the loss landscape (demos/loss_landscape), on MPS:
the notebook's training run with the landscape at its defaults (a surface
every 2,000 steps, a path point every 10th step) against the baseline arm
"committed", the same loop without the landscape whose work reaches the
device as it runs; the surface's own time and its peak memory; vmap against
the loop over perturbed copies; and the per-step cost of the path and marker
calls.

    PYTHONPATH=python .venv/bin/python python/bench/bench_landscape.py [--steps 8000] [--repeats 3]
        [--grid 41] [--diagnose]

The baseline. On MPS, a training loop that never waits for the device and
records no event leaves its work queued until it ends: the notebook's loop
without the landscape spends 14 s on the host for 8,000 steps and then 14 s
more while the device drains, so a wall-clock comparison against that loop
measures the drain, not the landscape. The landscape's calls record events
and so let the device work beside the host. Every cost row therefore compares
against "committed", the loop without the landscape with a timed MPS event
recorded after every step, which hands its work to the device as it runs; the
plain loop without it ("plain", --diagnose) is reported for the record only.

Each training run is a fresh process that runs the notebook's own code cells
(all but the training run, the random-directions cell and the close) with a
viewer attached in a process of its own, which subscribes to the scene and
the series and discards what it receives, as a notebook widget would read
them; a run starts only once the server holds the viewer's subscription to
the scene, and fails if the viewer ends early or fails. The arms alternate. Without the landscape the loop is the same, with
land.point and land.due made no-ops. Each run's clock stops after
torch.mps.synchronize(), so it includes every kernel the loop queued. The
budget of design/demos.md is at most 1.5 times as long with the landscape.
--grid overrides the notebook's 41 points a side. Prints one line per row with
the 1-minute load average beside it, and each run's measurements as JSON.
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
    grid = int(os.environ.get("CALIPER_BENCH_GRID", "0"))
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code" and cell["id"] not in ("widgets", "run", "random", "close"):
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec"), ns)
            if cell["id"] == "serve" and grid:                # --grid: another resolution
                made = ns["Landscape"]
                ns["Landscape"] = lambda *a, **kw: made(*a, **{**kw, "grid": grid})
    return ns


def viewer(url: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, __file__, "--drain", url],
                            env={**os.environ, "PYTHONPATH": str(REPO / "python")})


def wait_subscribed(server, view: subprocess.Popen, scene: str = "landscape",
                    timeout: float = 60.0) -> None:
    """Waits until the server holds the viewer's subscription to the scene,
    so that a run measures the landscape with a subscriber. Raises SystemExit
    when the viewer process ends first, or after ``timeout`` seconds."""
    from caliper._net_scene import route_of
    route = route_of(server.ring)
    end = time.monotonic() + timeout
    while not route.watched(scene):
        if view.poll() is not None:
            raise SystemExit(f"the viewer ended with exit status {view.returncode} before it subscribed "
                             f"to {scene!r}")
        if time.monotonic() > end:
            raise SystemExit(f"the viewer did not subscribe to {scene!r} within {timeout:.0f} s")
        time.sleep(0.01)


def drain(url: str) -> None:
    """A viewer that subscribes to every view and discards what it receives.
    It ends quietly when the server closes the connection; any other error
    ends the process with a traceback and exit status 1, which run() sees."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

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
    except ConnectionClosed:                                # the server closed
        pass


def run(variant: str, steps: int, mode: str = "plain") -> dict:
    """One training run of the notebook; prints a JSON line.

    The clock starts after a synchronised warm-up and stops after
    torch.mps.synchronize(), so the run time includes every kernel the loop
    queued; it is the training run's time, not the time until a viewer has
    drawn the last surface. Every step's end is stamped on the host, so the
    time from the last step's host work to the end of the device's work (the
    drain) is reported too. ``mode`` "events" also records a timed MPS event
    after every step and around every surface, without synchronising, to see
    when the device reaches each one; "sync" synchronises before and after
    every surface and times it on the host, which serialises the surfaces;
    "bounded" synchronises after every 25th step, so that the host never runs
    more than 25 steps ahead of the device."""
    import torch
    ns = notebook(steps)
    assert ns["device"].type == "mps", "the cost row runs on MPS"
    land = ns["land"]
    view = viewer(ns["server"].stream_url)
    wait_subscribed(ns["server"], view)
    calls, stamps, events, surfaces = [], [], [], []
    timing = mode == "events"
    point = land.point if variant == "with" else (lambda model, step: None)

    def stamped_point(model, step):
        r = point(model, step)
        if mode == "bounded" and len(stamps) % 25 == 24:
            torch.mps.synchronize()
        stamps.append(time.perf_counter())
        if timing:
            e = torch.mps.Event(enable_timing=True)
            e.record()
            events.append(e)
        return r
    land.point = stamped_point
    if variant == "without":
        land.due = lambda step, every, directions: False
    else:
        show_path, show_surface = ns["show_path"], ns["show_surface"]

        def timed_path(rate=10):
            t = time.perf_counter_ns()
            show_path(rate)
            calls.append(time.perf_counter_ns() - t)

        def timed_surface():
            k = len(stamps) - 1                             # the step this surface follows
            if mode == "sync":
                torch.mps.synchronize()
                t = time.perf_counter()
                show_surface()
                torch.mps.synchronize()
                surfaces.append((k, time.perf_counter() - t))
            elif timing:
                a, b = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                a.record()
                show_surface()
                b.record()
                surfaces.append((k, a, b))
            else:
                show_surface()
                surfaces.append((k, None))
        ns["show_path"], ns["show_surface"] = timed_path, timed_surface
    ns["train"](20)                                         # warm-up: kernels, the first batches
    torch.mps.synchronize()
    calls.clear(), stamps.clear(), events.clear(), surfaces.clear()
    load0 = load()
    start = torch.mps.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    ns["train"](steps)
    torch.mps.synchronize()
    seconds = time.perf_counter() - t0
    host_end = stamps[-1] - t0
    out = {"variant": variant, "mode": mode, "steps": steps, "seconds": seconds,
           "host_end": host_end, "drain": seconds - host_end, "load": [load0, load()]}
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    out["step_ms_median"] = statistics.median(gaps) * 1e3
    if timing:
        # When the device reached each step's event, against when the host
        # finished that step: the device's lag behind the host, and the
        # device's own time a step wherever it never waited for the host.
        gpu = [start.elapsed_time(e) / 1e3 for e in events]
        lag = [g - (h - t0) for g, h in zip(gpu, stamps)]
        out["lag_end_s"] = lag[-1]
        out["device_end_s"] = gpu[-1]
        busy = [b - a for a, b, la in zip(gpu, gpu[1:], lag) if la > 0.01]
        out["device_step_ms_when_behind"] = statistics.median(busy) * 1e3 if busy else None
    if variant == "with":
        calls.sort()
        out["path_calls_us"] = {"median": statistics.median(calls) / 1e3,
                                "p99": calls[int(len(calls) * 0.99)] / 1e3,
                                "mean": sum(calls) / len(calls) / 1e3, "count": len(calls),
                                "slowest": [c / 1e3 for c in calls[-5:]]}
        out["refits"] = [s for s, _ in land.refits]
        out["method"] = land.method
        out["share"] = float(land.share)
        # The host's step that holds each surface call, and the slowest of the
        # 50 steps after it, against the median step.
        out["surface_step_ms"] = [gaps[k] * 1e3 for k, *_ in surfaces if k < len(gaps)]
        out["next50_max_ms"] = [max(gaps[k + 1:k + 51]) * 1e3 for k, *_ in surfaces
                                if k + 51 <= len(gaps)]
        if mode == "sync":
            out["surface_s"] = [t for _, t in surfaces]
        if timing:
            out["surface_s"] = [a.elapsed_time(b) / 1e3 for _, a, b in surfaces]
            after, catch, busy = [], [], []
            for k, *_ in surfaces:
                if k + 1 >= len(lag):
                    continue
                after.append(lag[k + 1])
                n = next((i for i in range(k + 1, len(lag)) if lag[i] < 0.01), len(lag)) - k - 1
                catch.append(n)
                if n > 10:                                  # the device's own time a step,
                    j = k + 1 + n                           # while it works off the backlog
                    busy.append((gpu[j - 1] - gpu[k + 2]) / (j - 3 - k) * 1e3)
            far = [lag[i] for i in range(len(lag)) if all(not (k < i <= k + c + 1) for (k, *_), c
                                                          in zip(surfaces, catch))]
            out["lag_after_surface_s"] = after
            out["catch_up_steps"] = catch
            out["device_step_ms_in_backlog"] = busy
            out["lag_elsewhere_ms_median"] = statistics.median(far) * 1e3 if far else None
    ns["server"].close()
    if view.wait(10) != 0:
        raise SystemExit(f"the viewer failed during the run (exit status {view.returncode})")
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", type=int, default=0, help="points a side, instead of the notebook's")
    ap.add_argument("--drain")
    ap.add_argument("--run", choices=["with", "without"])
    ap.add_argument("--mode", choices=["plain", "events", "sync", "bounded"], default="plain")
    ap.add_argument("--diagnose", action="store_true",
                    help="also run the arm with the landscape in the modes events and sync")
    ap.add_argument("--surface", action="store_true")
    ap.add_argument("--picture", action="store_true")
    args = ap.parse_args()
    if args.grid:
        os.environ["CALIPER_BENCH_GRID"] = str(args.grid)        # read by every child
    if args.drain:
        return drain(args.drain)
    if args.run:
        return run(args.run, args.steps, args.mode) and None
    if args.surface:
        return surface(600)
    if args.picture:
        return picture(args.steps)
    arms = {"landscape": ["--run", "with"], "committed": ["--run", "without", "--mode", "events"]}
    times = {arm: [] for arm in arms}
    for i in range(args.repeats):
        for arm in ("committed", "landscape") if i % 2 == 0 else ("landscape", "committed"):
            r = child(arms[arm] + ["--steps", str(args.steps)])
            times[arm].append(r["seconds"])
            extra = ""
            if arm == "landscape":
                c = r["path_calls_us"]
                extra = (f"; path+marker calls median {c['median']:.1f} us, p99 {c['p99']:.1f} us,"
                         f" slowest {max(c['slowest']):.0f} us; {len(r['refits'])} surfaces at "
                         f"{r['refits']}, {r['method']}, share {r['share']:.3f}")
            print(f"{arm:9s} {args.steps} steps {r['seconds']:7.2f} s to the device's end "
                  f"(host loop {r['host_end']:.2f} s, drain {r['drain'] * 1e3:.0f} ms, step "
                  f"{r['step_ms_median']:.2f} ms median; load {r['load'][0]:.1f} to "
                  f"{r['load'][1]:.1f}){extra}", flush=True)
            print("json: " + json.dumps(r), flush=True)
    ratio = statistics.median(times["landscape"]) / statistics.median(times["committed"])
    print(f"ratio landscape / committed: {ratio:.3f} (budget 1.5); each run "
          f"{min(times['landscape']) / max(times['committed']):.3f} to "
          f"{max(times['landscape']) / min(times['committed']):.3f}", flush=True)
    if args.diagnose:
        base = statistics.median(times["committed"])
        for variant, mode in (("without", "plain"), ("with", "events"), ("with", "sync"),
                              ("without", "bounded"), ("with", "bounded")):
            r = child(["--run", variant, "--steps", str(args.steps), "--mode", mode])
            print("json: " + json.dumps(r), flush=True)
            name = {"without": "plain" if mode == "plain" else "without", "with": "landscape"}
            line = (f"{name[variant]}, {mode}: {r['seconds']:.2f} s to the device's end (host loop "
                    f"{r['host_end']:.2f} s, drain {r['drain'] * 1e3:.0f} ms), "
                    f"{r['seconds'] - base:+.2f} s against the median committed run; host "
                    f"step {r['step_ms_median']:.2f} ms median")
            if mode == "events":
                line += (f"; device {r['device_step_ms_when_behind']:.2f} ms a step while behind"
                         f" the host, {r['lag_end_s']:.2f} s behind at the last step")
            if variant == "with" and mode in ("events", "sync"):
                line += (f"; surfaces {sum(r['surface_s']):.2f} s in all ({min(r['surface_s']):.3f}"
                         f" to {max(r['surface_s']):.3f} s each); the host step holding a surface "
                         f"{min(r['surface_step_ms']):.0f} to {max(r['surface_step_ms']):.0f} ms")
            if variant == "with" and mode == "events":
                line += (f"; device lag after a surface {min(r['lag_after_surface_s']):.3f} to "
                         f"{max(r['lag_after_surface_s']):.3f} s, worked off in "
                         f"{min(r['catch_up_steps'])} to {max(r['catch_up_steps'])} steps")
            print(line + f" (load {r['load'][0]:.1f} to {r['load'][1]:.1f})", flush=True)
    s = child(["--surface"])
    fmt = lambda xs: ", ".join(f"{x:.3f}" for x in xs)  # noqa: E731
    print(f"surface: first {fmt(s['first path s'])} s; vmap path {fmt(s['vmap path s'])} s;"
          f" vmap random {fmt(s['vmap random s'])} s;"
          f" loop path {fmt(s['loop path s'])}"
          + f" s; surface peak {s['peak MB']:.0f} MB above the {s['held MB']:.0f} MB held "
          f"(load {s['load'][0]:.1f} to {s['load'][1]:.1f})", flush=True)


if __name__ == "__main__":
    main()
