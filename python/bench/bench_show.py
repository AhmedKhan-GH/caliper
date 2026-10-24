"""caliper.show() in a training loop, per transport: throughput, per-call cost,
step-to-screen latency, memory.

    PYTHONPATH=cmake-build-notorch/py .venv/bin/python python/bench/bench_show.py \\
        --mode window|attach --device mps|cpu --size 256|1024|2048 [--seconds 8]

One (mode, device, size) per invocation (under 2 minutes); the log goes to
python/bench/results/P-<mode>-<device>-<size>.txt as well as stdout.

Workload (A's): the TinyCNN of examples/python/train_show.py on 64-image
batches plus a weight W (size x size, f32, updated in place every step);
``caliper.show(W, "W")`` once per step, autoscaled (GPU aminmax).

Rows (each --seconds long, after a warm-up):
  baseline-A / baseline-B   no caliper (before / after; their mean is 100 %)
  control-record60          MPS: torch.mps.Event.record() at 60 Hz and nothing else.
                            Recording an event commits torch's command buffer, which by
                            itself RAISES this small model's throughput, so a snapshot
                            sampler is compared with this row, not only with the baseline.
  sampler-60Hz              show(W) every step, default rate (60 snapshots/s)
  sampler-off               show(W, rate=0): a snapshot whenever a slot is free
  sampler-60Hz+run()        window mode: training on a worker thread, main thread in
                            the AppKit loop (caliper.run)

Latency (step-to-screen, show() stamp -> frame drawn): window mode from the render
thread (every drawn snapshot) and from caliper.live_view_info() sampled at 10 Hz;
attach mode from the viewer's own samples (caliper-viewer --latency-log, same
CLOCK_UPTIME_RAW clock). Memory: torch.mps.current_allocated_memory and RSS deltas
over each caliper row.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


class TinyCNN(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def make_batch(n, device):
    labels = torch.randint(0, 4, (n,), device=device)
    cy = 7.0 + 14.0 * (labels // 2).float() + torch.randn(n, device=device) * 2
    cx = 7.0 + 14.0 * (labels % 2).float() + torch.randn(n, device=device) * 2
    g = torch.arange(28, device=device, dtype=torch.float32)
    yy = (g[None, :, None] - cy[:, None, None]) ** 2
    xx = (g[None, None, :] - cx[:, None, None]) ** 2
    img = torch.exp(-(yy + xx) / 18.0) + 0.15 * torch.randn(n, 28, 28, device=device)
    return img[:, None], labels


class Workload:
    def __init__(self, device, size):
        torch.manual_seed(0)
        self.device = torch.device(device)
        self.model = TinyCNN().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.W = torch.randn(size, size, device=self.device)

    def step(self):
        x, y = make_batch(64, self.device)
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.W.add_(0.001)

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True).stdout
    return int(out.strip()) / 1024.0


def mps_mem():
    return torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0


def visible_fraction(log, t0, t1):
    """Fraction of [t0, t1] the window was not occluded (occlusion_log entries)."""
    if not log or t1 <= t0:
        return None
    vis_time, state, t = 0, log[0][1], t0
    for ts, v in log:
        if ts <= t0:
            state = v
            continue
        if ts >= t1:
            break
        if state:
            vis_time += ts - t
        t, state = ts, v
    if state:
        vis_time += t1 - t
    return round(vis_time / (t1 - t0), 3)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def run_steps(wl, seconds, call=None, timings=None):
    """Train for `seconds`; call(W) every step. Returns (steps, elapsed incl. the
    final device sync)."""
    steps = 0
    pc = time.perf_counter_ns
    t0 = time.perf_counter()
    t_end = t0 + seconds
    while time.perf_counter() < t_end:
        wl.step()
        steps += 1
        if call is not None:
            a = pc()
            took = call(wl.W)
            if timings is not None:
                timings[bool(took)].append(pc() - a)
    wl.sync()
    return steps, time.perf_counter() - t0


class InfoSampler:
    """live_view_info("W") at 10 Hz on its own thread: a core query waits for
    the render thread (it runs between frames), so the training loop never makes
    one. The GIL is released while it waits."""

    def __init__(self, caliper):
        import threading
        self.caliper = caliper
        self.lat = []
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(0.1):
            try:
                self.lat.append(self.caliper.live_view_info("W")["latency_ms"])
            except RuntimeError:
                pass

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("window", "attach"), required=True)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--activate", action="store_true",
                    help="window mode: bring the window to the front (default: behind the terminal)")
    ap.add_argument("--tag", default="", help="suffix of the result file name")
    args = ap.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("no MPS device", file=sys.stderr)
        return 2
    os.makedirs(RESULTS, exist_ok=True)
    out = Tee(os.path.join(RESULTS, f"P-{args.mode}-{args.device}-{args.size}{args.tag}.txt"))
    import caliper
    dev, S, secs, warm = args.device, args.size, args.seconds, args.warmup
    out(f"# bench_show.py --mode {args.mode} --device {dev} --size {S} --seconds {secs:g}  "
        f"({time.strftime('%Y-%m-%d %H:%M:%S')})")
    out(f"torch {torch.__version__}, W {S}x{S} f32 ({S * S * 4 / 2**20:.1f} MB), show(W) every step, "
        f"autoscaled, {secs:g} s per row after {warm:g} s warm-up, load average "
        f"{' '.join(f'{x:.1f}' for x in os.getloadavg())} (18 cores)")
    wl = Workload(dev, S)
    rows = []

    win = sess = None
    lat_log = None
    if args.mode == "window":
        win = caliper.window(900, 600, title="bench_show", activate=args.activate)
        win.core.set_window_pos(40, 40)
    else:
        sess = caliper.session()
        lat_log = tempfile.mktemp(prefix="caliper-bench-lat-", suffix=".txt")
        sess.launch_viewer(args=("--stats", "--latency-log", lat_log),
                           log=lat_log + ".stderr")
        if not sess.wait_attached(15.0):
            out("the viewer did not attach")
            return 3

    def ring():
        return win.ring if win is not None else caliper._session_route.ring

    def row(name, steps, el, timings=None, extra=None):
        r = {"row": name, "steps": steps, "seconds": round(el, 3),
             "steps_per_s": round(steps / el, 1)}
        if timings is not None:
            snap, skip = timings[True], timings[False]
            r.update(shows=len(snap) + len(skip), snapshots=len(snap),
                     snap_us_med=round(q(snap, 0.5) / 1e3, 1), snap_us_p95=round(q(snap, 0.95) / 1e3, 1),
                     skip_us_med=round(q(skip, 0.5) / 1e3, 2) if skip else None,
                     skip_us_p95=round(q(skip, 0.95) / 1e3, 2) if skip else None)
        if extra:
            r.update(extra)
        rows.append(r)
        out("RESULT " + json.dumps(r))

    def caliper_row(name, rate, use_run=False):
        show = (lambda W: caliper.show(W, "W")) if rate is None else \
            (lambda W: caliper.show(W, "W", rate=rate))
        if use_run:
            caliper.run(run_steps, wl, warm, show)
        else:
            run_steps(wl, warm, show)
        caliper.flush()
        time.sleep(0.1)
        c0 = dict(ring().counters)
        if win is not None:
            win.core.render_stats(True)
        m0, r0 = mps_mem(), rss_mb()
        t_ns0 = time.monotonic_ns()
        info_lat = []
        t = {True: [], False: []}
        if win is not None:
            with InfoSampler(caliper) as smp:
                if use_run:
                    s, e = caliper.run(run_steps, wl, secs, show, t)
                else:
                    s, e = run_steps(wl, secs, show, t)
            info_lat = smp.lat
        else:
            s, e = run_steps(wl, secs, show, t)
        t_ns1 = time.monotonic_ns()
        caliper.flush()
        time.sleep(0.1)
        m1, r1 = mps_mem(), rss_mb()
        c1 = dict(ring().counters)
        cnt = {k: c1.get(k, 0) - c0.get(k, 0) for k in ("taken", "published", "ring_full",
                                                         "superseded", "skipped")}
        extra = {"ring": cnt, "d_mps_MB": round((m1 - m0) / 2**20, 2), "d_rss_MB": round(r1 - r0, 2),
                 "t_ns": [t_ns0, t_ns1]}
        if win is not None:
            st = win.core.render_stats(True)
            lat = st["latency_ms"]
            extra.update(frames=st["frames"], fps=round(st["frames"] / e, 1), drawn=st["drawn"],
                         errors=st["errors"], last_error=st["last_error"],
                         lat_ms_med=round(q(lat, 0.5), 2), lat_ms_p95=round(q(lat, 0.95), 2),
                         lat_n=len(lat), info_lat_ms_med=round(q(info_lat, 0.5), 2),
                         info_lat_n=len(info_lat), max_frame_gap_ms=round(st["max_frame_gap_ms"], 1))
            i = caliper.view_info("W")
            extra.update(path=i["path"], imported=i["imported"], creates=i["creates"],
                         visible_frac=visible_fraction(win.core.occlusion_log(), t_ns0, t_ns1))
        row(name, s, e, t, extra)

    run_steps(wl, warm)
    s, e = run_steps(wl, secs)
    row("baseline-A", s, e)
    if dev == "mps":
        ev = torch.mps.Event()
        last = [0.0]

        def commit_only(W):
            now = time.perf_counter()
            if now - last[0] >= 1.0 / 60.0:
                last[0] = now
                ev.record()
                return True
            return False
        s, e = run_steps(wl, secs, commit_only)
        row("control-record60", s, e)
    caliper_row("sampler-60Hz", None)
    caliper_row("sampler-off", 0)
    if win is not None:
        caliper_row("sampler-60Hz+run()", None, use_run=True)
    caliper.close()        # the window closes; the session's viewer keeps its last frame
    viewer_lat = []
    if sess is not None:
        if sess.viewer is not None:
            sess.viewer.terminate()          # SIGTERM: final stats line, latency log flushed
            try:
                sess.viewer.wait(10)
            except subprocess.TimeoutExpired:
                sess.viewer.kill()
        try:
            with open(lat_log) as f:
                viewer_lat = [tuple(float(x) for x in ln.split()) for ln in f if ln.strip()]
        except OSError:
            pass
        for r in rows:
            if "t_ns" in r:
                lo, hi = r["t_ns"]
                xs = [ms for t_ns, ms in viewer_lat if lo <= t_ns <= hi]
                r.update(lat_ms_med=round(q(xs, 0.5), 2), lat_ms_p95=round(q(xs, 0.95), 2),
                         lat_n=len(xs))
                out("VIEWER " + json.dumps({"row": r["row"], "lat_ms_med": r["lat_ms_med"],
                                            "lat_ms_p95": r["lat_ms_p95"], "lat_n": r["lat_n"]}))
    s, e = run_steps(wl, secs)
    row("baseline-B", s, e)

    out(f"load average at the end {' '.join(f'{x:.1f}' for x in os.getloadavg())}")
    base = (rows[0]["steps_per_s"] + rows[-1]["steps_per_s"]) / 2
    ctrl = next((r["steps_per_s"] for r in rows if r["row"] == "control-record60"), None)
    out(f"\n{args.mode} {dev} {S}x{S}: baseline mean {base:.1f} steps/s "
        f"(A {rows[0]['steps_per_s']}, B {rows[-1]['steps_per_s']})"
        + (f", commit control {ctrl:.1f}" if ctrl else ""))
    out("| row | steps/s | % baseline | % control | snapshot us med (p95) | skip us med | "
        "snapshots | ring full | latency ms med (p95) [n] | d MPS MB | d RSS MB |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pct_c = f"{100 * r['steps_per_s'] / ctrl:.1f}" if ctrl else "-"
        snap = f"{r['snap_us_med']} ({r['snap_us_p95']})" if "snap_us_med" in r else "-"
        lat = (f"{r['lat_ms_med']} ({r['lat_ms_p95']}) [{r['lat_n']}]" if "lat_ms_med" in r else "-")
        out(f"| {r['row']} | {r['steps_per_s']:.1f} | {100 * r['steps_per_s'] / base:.1f} | {pct_c} | "
            f"{snap} | {r.get('skip_us_med', '-')} | {r.get('snapshots', '-')} | "
            f"{r.get('ring', {}).get('ring_full', '-')} | {lat} | {r.get('d_mps_MB', '-')} | "
            f"{r.get('d_rss_MB', '-')} |")
    for r in rows:
        if "frames" in r:
            out(f"  {r['row']}: render {r['fps']} fps, drawn {r['drawn']}, errors {r['errors']}, "
                f"max frame gap {r['max_frame_gap_ms']} ms, window visible {r['visible_frac']} of "
                f"the row, live_view_info latency med {r['info_lat_ms_med']} ms "
                f"[{r['info_lat_n']}], {r['path']} imported={r['imported']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
