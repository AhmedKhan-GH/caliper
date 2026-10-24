"""Evidence for the in-process window transport (caliper.window).

    PYTHONPATH=cmake-build-notorch/py .venv/bin/python python/bench/bench_inproc.py <cmd> [...]

Commands (each runs in its own process; `all` runs every one and writes logs to
python/bench/results/P-*.txt):

    clock                       time.monotonic_ns() vs the render thread's clock
    liveness --mode spike|none|run
                                screenshots at t=1 s and t=3 s of a 5 s stall in the
                                training loop, plus main-thread probe latencies
    tearing [--control]         torn-frame detector (uniform tensors, texture readbacks)

Throughput, per-call cost and latency rows moved to bench_show.py. The A-*.txt
results were measured with prototype A (caliper._inproc.InprocViewer, its
SIGALRM pump and the v1.2 view route); the pump is not part of the package.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# ---------------------------------------------------------------------------
# the training workload (examples/python/train_show.py's TinyCNN + a weight W)
# ---------------------------------------------------------------------------
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


def run_steps(wl, seconds, show=None, timings=None):
    """Train for `seconds`; show(W) every step when `show` is given. Returns
    (steps, elapsed incl. the final device sync)."""
    steps = 0
    t0 = time.perf_counter()
    t_end = t0 + seconds
    pc = time.perf_counter_ns
    while time.perf_counter() < t_end:
        wl.step()
        steps += 1
        if show is not None:
            a = pc()
            took = show(wl.W, "W")
            if timings is not None:
                timings[bool(took)].append(pc() - a)
    wl.sync()
    return steps, time.perf_counter() - t0


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True).stdout
    return int(out.strip()) / 1024.0


def emit(obj):
    print("RESULT " + json.dumps(obj), flush=True)


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------
def cmd_clock(args):
    from caliper import _native
    pairs = []
    for _ in range(5):
        a = time.monotonic_ns()
        b = _native.monotonic_ns()
        c = time.monotonic_ns()
        pairs.append((a, b, c))
    for a, b, c in pairs:
        print(f"time.monotonic_ns {a}  native CLOCK_UPTIME_RAW {b}  time.monotonic_ns {c}  "
              f"(native - first = {b - a} ns, bracketed: {a <= b <= c})")
    emit({"cmd": "clock", "bracketed": all(a <= b <= c for a, b, c in pairs)})


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------
def read_bmp(path):
    import numpy as np
    with open(path, "rb") as f:
        b = f.read()
    off = struct.unpack_from("<I", b, 10)[0]
    w, h = struct.unpack_from("<ii", b, 18)
    bpp = struct.unpack_from("<H", b, 28)[0]
    row = ((w * bpp // 8) + 3) & ~3
    a = np.frombuffer(b, dtype=np.uint8, count=row * abs(h), offset=off).reshape(abs(h), row)
    a = a[:, : w * bpp // 8].reshape(abs(h), w, bpp // 8)
    return a[::-1] if h > 0 else a


def png_diff(p1, p2):
    import numpy as np
    bmps = []
    for p in (p1, p2):
        bmp = p[:-4] + ".bmp"
        subprocess.run(["sips", "-s", "format", "bmp", p, "--out", bmp], capture_output=True, check=True)
        bmps.append(read_bmp(bmp))
        os.remove(bmp)
    a, b = bmps
    if a.shape != b.shape:
        return {"same_shape": False, "shape1": a.shape, "shape2": b.shape}
    d = (a != b).any(-1)
    ys, xs = np.nonzero(d)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    return {"same_shape": True, "w": int(a.shape[1]), "h": int(a.shape[0]),
            "changed_px": int(d.sum()), "changed_frac": round(float(d.mean()), 5), "bbox": box}


def cmd_liveness(args):
    """A training loop that shows W for 1 s, then stalls 5 s (time.sleep), then
    shows for 1 s more. Screenshots at t=1 s and t=3 s of the stall; main-thread
    probes posted 0.5 s into the stall (run-loop block, synthetic mouse move) and
    a close probe 3.5 s in. Modes: spike (the spike's synchronous show, no render
    thread), none (caliper.window, training on the main thread), run
    (caliper.window + caliper.run: training on a worker thread)."""
    import caliper
    mode = args.mode
    tag = f"{args.prefix}-liveness-{mode}"
    wl = Workload("mps", 512)
    stall = args.stall

    if mode == "spike":
        # The spike's caliper.show model, inactive window: drain, synchronous
        # update, and poll + frame (60 Hz) inside show(); nothing between calls.
        import tempfile
        from caliper import _native
        core = _native.Core(tempfile.mkdtemp(prefix="caliper-py-"))
        core.open_window(800, 500, "caliper", False)
        last = [0.0]

        def core_show(t, name):
            lo, hi = torch.aminmax(t)
            core.view_tensor(name, t.__dlpack__(), 0, 0, lo.item(), hi.item(),
                             _native.VIEW_FLAG_WAIT_GPU)
            core.poll()
            now = time.perf_counter()
            if now - last[0] >= 1.0 / 60.0:
                last[0] = now
                core.frame()
        viewer = None
    else:
        viewer = caliper.window(800, 500, activate=False, heartbeat=True)
        core = viewer.core
        core_show = caliper.show
    core.set_window_pos(60, 60)
    wid = core.window_number
    print(f"mode {mode}: window id {wid}")

    shots = {}
    marks = {}

    def helper(t_stall0):
        def at(t):
            d = t_stall0 + t - time.monotonic()
            if d > 0:
                time.sleep(d)
        at(0.5)
        for k in (0, 1):
            core.post_probe(k)
        marks["probe_posted"] = time.monotonic() - t_stall0
        for t in (1.0, 3.0):
            at(t)
            path = os.path.join(RESULTS, f"{tag}-t{int(t)}s.png")
            subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), path], check=False)
            shots[t] = path
            if viewer is not None and not viewer.closed:
                marks[f"frames_at_{t}"] = viewer.core.render_stats()["frames"]
        at(3.5)
        core.post_probe(2)
        marks["close_posted"] = time.monotonic() - t_stall0

    def train(seconds):
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            wl.step()
            core_show(wl.W, "W")
        wl.sync()

    def stall_fn():
        time.sleep(stall)

    def body():
        train(1.0)
        t0 = time.monotonic()
        th = threading.Thread(target=helper, args=(t0,), daemon=True)
        th.start()
        f0 = viewer.core.render_stats()["frames"] if viewer is not None else None
        stall_fn()
        marks["stall_s"] = time.monotonic() - t0
        if viewer is not None and not viewer.closed:
            marks["frames_during_stall"] = viewer.core.render_stats()["frames"] - f0
        th.join()
        return t0

    if viewer is not None:
        viewer.stats(reset=True)
    if mode == "run":
        t0 = caliper.run(body)
    else:
        t0 = body()
    t_after = time.monotonic()
    # after the stall: pump 0.5 s so queued probes get handled (the realistic
    # "next show()/pump" moment for modes without a pump)
    end = time.perf_counter() + 0.5
    while time.perf_counter() < end:
        if viewer is None:
            core.poll()
        elif not viewer.closed:
            viewer.pump()
        time.sleep(0.005)
    probes = core.probe_results()
    if viewer is not None:
        st = viewer.stats() if not viewer.closed else {}
    else:
        st = {}
    names = {0: "runloop-block", 1: "mouse-move", 2: "close"}
    t0_ns = int(t0 * 1e9)
    pr = [{"probe": names[k], "posted_s": round((p - t0_ns) / 1e9, 3),
           "handled_s": round((h - t0_ns) / 1e9, 3), "latency_ms": round((h - p) / 1e6, 1)}
          for k, p, h in probes]
    closed = viewer.closed if viewer is not None else core.should_close()
    res = {"cmd": "liveness", "mode": mode, "stall_s": round(marks.get("stall_s", 0), 3),
           "stall_end_s": round(t_after - t0, 3), "probes": pr, "window_closed": closed,
           "frames_during_stall": marks.get("frames_during_stall"),
           "frames_at_1s": marks.get("frames_at_1.0"), "frames_at_3s": marks.get("frames_at_3.0"),
           "max_frame_gap_ms": round(st.get("max_frame_gap_ms", 0), 1) if st else None}
    if 1.0 in shots and 3.0 in shots and os.path.exists(shots[1.0]) and os.path.exists(shots[3.0]):
        res["screens"] = [os.path.relpath(shots[1.0]), os.path.relpath(shots[3.0])]
        res["diff_t1_t3"] = png_diff(shots[1.0], shots[3.0])
    emit(res)
    for p in pr:
        print(f"  probe {p['probe']:14s} posted t={p['posted_s']:.2f}s handled t={p['handled_s']:.2f}s "
              f"({p['latency_ms']:.1f} ms)")
    print(f"  stall {res['stall_s']} s; frames during stall {res['frames_during_stall']}; "
          f"screens differ: {res.get('diff_t1_t3')}")
    if viewer is not None:
        caliper.close()
    else:
        core.close()


# ---------------------------------------------------------------------------
# tearing
# ---------------------------------------------------------------------------
def cmd_tearing(args):
    """Every snapshot is a uniform tensor (value k, cycling), shown with the fixed
    range [0, 255], so each k maps to its own LUT colour. The producer rewrites as
    fast as the GPU allows (rate=0); the view texture is read back about
    `args.checks` times per second and decoded:
      torn   = more than one value in one texture (a read overlapped a write)
      wrong  = a single value that was never published (the slot was rewritten
               between publication and the core's read)
    --control makes the ring always rewrite slot 0, claimed or not (the unsafe
    case), to show what the detector sees when the slot rule is broken."""
    import numpy as np
    import caliper
    S = args.size
    viewer = caliper.window(512, 512, offscreen=True, rate=0)
    lut = {int(c): i for i, c in enumerate(caliper.colormap_lut("viridis"))}
    ring = viewer.ring
    cur_k = [0]
    slot_k, published = {}, set()
    real_pick, real_publish = ring._pick, viewer.transport.publish

    def pick(name, v):
        if args.control:
            ring.poll()           # publish slot 0 if its last write completed
            v.pending.clear()     # slot 0's single event now tracks the newest write
            idx = 0               # always, whatever the core reads
        else:
            idx = real_pick(name, v)
        if idx is not None:
            slot_k[idx] = cur_k[0]
        return idx

    def publish(name, slot, meta):
        published.add(slot_k.get(slot.index))
        real_publish(name, slot, meta)
    ring._pick = pick
    viewer.transport.publish = publish

    W = torch.zeros(S, S, device="mps")
    it = checks = torn = wrong = snaps = 0
    t_end = time.perf_counter() + args.seconds
    next_check = time.perf_counter()
    while time.perf_counter() < t_end:
        cur_k[0] = (cur_k[0] + 7) % 256
        W.fill_(float(cur_k[0]))
        snaps += bool(caliper.show(W, "T", vmin=0, vmax=255))
        it += 1
        if it % 8 == 0:
            torch.mps.synchronize()   # back-pressure, like a loop's periodic .item()
        if time.perf_counter() >= next_check and published:
            try:
                rb = viewer.core.view_readback("T")
            except RuntimeError:      # published, not yet applied by the render thread
                continue
            vals = np.unique(rb.view(np.uint32).ravel())
            checks += 1
            if len(vals) > 1:
                torn += 1
            elif lut.get(int(vals[0])) not in published:
                wrong += 1
            next_check = time.perf_counter() + 1.0 / args.checks
    torch.mps.synchronize()
    st = viewer.stats()
    caliper.close()
    r = {"cmd": "tearing", "size": S, "control": args.control, "iterations": it,
         "snapshots": snaps, "published": len(published), "drawn": st["drawn"],
         "checks": checks, "torn": torn, "wrong_content": wrong, "ring": st["ring"],
         "heatmap_live_apply": st.get("heatmap_live_apply")}
    emit(r)
    print(f"  size {S} control={args.control}: {it} iterations, {snaps} snapshots, "
          f"{st['drawn']} drawn, {checks} readbacks: {torn} torn, {wrong} showing a "
          f"never-published value")


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------
def cmd_all(args):
    os.makedirs(RESULTS, exist_ok=True)
    py = sys.executable
    me = os.path.abspath(__file__)
    pre = args.prefix
    jobs = [(f"{pre}-clock", ["clock"])]
    for mode in ("spike", "none", "run"):
        jobs.append((f"{pre}-liveness-{mode}", ["liveness", "--mode", mode, "--prefix", pre]))
    jobs.append((f"{pre}-tearing", ["tearing"]))
    jobs.append((f"{pre}-tearing-control", ["tearing", "--control"]))
    only = set(args.only.split(",")) if args.only else None
    for tag, argv in jobs:
        if only and not any(o in tag for o in only):
            continue
        log = os.path.join(RESULTS, tag + ".txt")
        print(f"== {tag}: {' '.join(argv)}", flush=True)
        with open(log, "w") as f:
            f.write(f"# {' '.join(['bench_inproc.py'] + argv)}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.flush()
            p = subprocess.run([py, me] + argv, stdout=f, stderr=subprocess.STDOUT, cwd=os.getcwd())
            f.write(f"# exit {p.returncode}\n")
        print(f"   exit {p.returncode} -> {os.path.relpath(log)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clock")
    lv = sub.add_parser("liveness")
    lv.add_argument("--mode", choices=["spike", "none", "run"], default="none")
    lv.add_argument("--stall", type=float, default=5.0)
    lv.add_argument("--prefix", default="P")
    te = sub.add_parser("tearing")
    te.add_argument("--control", action="store_true")
    te.add_argument("--size", type=int, default=2048)
    te.add_argument("--seconds", type=float, default=10.0)
    te.add_argument("--checks", type=float, default=100.0)
    al = sub.add_parser("all")
    al.add_argument("--only", default="")
    al.add_argument("--prefix", default="P")
    args = ap.parse_args()
    globals()["cmd_" + args.cmd.replace("-", "_")](args)


if __name__ == "__main__":
    main()
