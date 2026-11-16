"""What caliper.serve() costs bench_net.py's training loop, measured in pairs.

    PYTHONPATH=python CALIPER_SESSION=0 .venv/bin/python python/bench/bench_net_pairs.py \\
        [--arms heatmap-1024:1,all:2,...] [--pairs 12] [--seconds 1.0] [--tag=-run1]

For each arm, a kind of view call with K viewers, it trains in pairs of short
segments: the view calls every step against a reference, in alternating order
within each pair, and takes the ratio of the two throughputs within each pair,
so that a disturbance from the rest of the machine lasting longer than a pair
affects both sides alike. With viewers the reference is the commit-matched
control (an MPS event recorded at 60 Hz and no view call, see bench_net.py);
with no viewer it is the plain loop, since a call then does no device work.

Kinds: heatmap-1024 (show of a 1024 x 1024 weight), series (the loss), points-100k
(100,000 3-D points with 4-class labels), all (the three), all-heavy30 (the three,
the heatmap and the points with rate=30), points-100k-r30 (the points with rate=30). Each viewer is net_client.py in its
own process, subscribed throughout. The workload is bench_net.py's.

A pair counts only when the 1-minute load average, read before and after it,
is at most --max-load (8 by default, this process and its viewers included):
before a pair it waits up to --max-wait seconds for that, and a pair that
ends above it is discarded and run again. When the wait runs out, the pair is
kept and marked over_load, so that a busy machine still gives a result and
says so.

Per arm it prints and records: the median ratio, its spread, the step time p99
of both sides, the load averages, the pairs discarded, the mean of the step times shorter than 3 ms
as a ratio (a check that rare long steps do not decide the result), the share
of the time the training thread spends in the view calls, and the ticks and
snapshots per second. The records go to python/bench/results/
N-pairs-<sender>-mps-1024<tag>.json, where <sender> is native or python.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KINDS = {"heatmap-1024": ["W"], "series": ["loss"], "points-100k": ["emb100k"],
         "all": ["W", "loss", "emb100k"], "all-heavy30": ["W30", "loss", "emb100k30"],
         "points-100k-r30": ["emb100k30"]}
DEFAULT_ARMS = [f"{k}:{n}" for k in ("heatmap-1024", "series", "points-100k", "all") for n in (1, 2)] + \
               ["all-heavy30:1", "all-heavy30:2"] + [f"{k}:0" for k in ("heatmap-1024", "series", "points-100k", "all")]
TRIM_US = 3000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated <kind>:<viewers>")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=1.0, help="length of each segment")
    ap.add_argument("--tag", default="", help="suffix of the result file name")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--tmp", default=None, help="folder for the viewers' records (default: a new temporary one)")
    ap.add_argument("--max-load", type=float, default=8.0, help="highest 1-minute load average of a pair")
    ap.add_argument("--max-wait", type=float, default=600.0, help="seconds to wait for it before a pair")
    a = ap.parse_args()
    if sys.platform == "darwin":                     # as a training script at the front would run
        ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(0x21, 0)
    import numpy as np
    import torch
    import bench_net as B
    import caliper

    arms = [(k, int(n)) for k, n in (w.split(":") for w in a.arms.split(","))]
    wl = B.Workload("mps", 1024, 100_000, 3)
    srv = caliper.serve(port=0)
    sender = "native" if type(srv).__name__ == "NativeServer" else "python"
    print(f"sender {sender}", flush=True)
    tmp = a.tmp or tempfile.mkdtemp(prefix="caliper-pairs-")
    os.makedirs(tmp, exist_ok=True)

    def hook_for(views):
        fs = []
        for v in views:
            if v == "W":
                fs.append(lambda: caliper.show(wl.W, "W"))
            elif v == "W30":
                fs.append(lambda: caliper.show(wl.W, "W", rate=30))
            elif v == "loss":
                fs.append(lambda: caliper.series(wl.loss, "loss"))
            elif v == "emb100k30":
                fs.append(lambda: caliper.points(wl.E, wl.labels, "emb100k", classes=4, rate=30))
            else:
                fs.append(lambda: caliper.points(wl.E, wl.labels, "emb100k", classes=4))

        def h():
            for f in fs:
                f()
        return h

    def control():
        st = [0.0]

        def h():
            now = time.perf_counter()
            if now - st[0] >= 1 / 60:
                st[0] = now
                torch.mps.Event().record()
        return h

    def seg(hook):
        marks, spent = [], [0]
        pc = time.perf_counter_ns

        def timed():
            t = pc()
            hook()
            spent[0] += pc() - t
        n, el = B.train(wl, a.seconds, timed if hook is not None else None, marks)
        dt = np.diff(np.asarray(marks, dtype=np.int64)) / 1e3
        keep = dt[dt < TRIM_US]
        return n / el, float(np.percentile(dt, 99)), spent[0] / 1e9 / el * 100, float(keep.mean())

    for _ in range(50):
        wl.step()
    wl.sync()
    rows = []
    for kind, k in arms:
        views = KINDS[kind]
        names = [{"W30": "W", "emb100k30": "emb100k"}.get(v, v) for v in views]
        procs = B.launch_viewers(srv.stream_url, names, "wire", k, tmp, f"{kind}-{k}",
                                 lambda: (wl.step(), hook_for(views)())) if k else []
        seg(hook_for(views))
        ref = (lambda: None) if k == 0 else control
        rec = []
        discarded = 0
        while len(rec) < a.pairs:
            i = len(rec)
            waited = time.monotonic()
            while os.getloadavg()[0] > a.max_load and time.monotonic() - waited < a.max_wait:
                time.sleep(5)
            over = os.getloadavg()[0] > a.max_load            # the wait ran out
            load0 = os.getloadavg()[0]
            k0 = dict(srv.ring.counters)
            if i % 2 == 0:
                c = seg(ref())
                t = seg(hook_for(views))
            else:
                t = seg(hook_for(views))
                c = seg(ref())
            k1 = dict(srv.ring.counters)
            load1 = os.getloadavg()[0]
            if not over and load1 > a.max_load:
                discarded += 1
                continue
            rec.append({"ratio": t[0] / c[0] * 100, "load_before": load0, "load_after": load1,
                        "over_load": over, "trimmed": c[3] / t[3] * 100, "steps_per_s": t[0],
                        "control_steps_per_s": c[0], "p99_us": t[1], "control_p99_us": c[1],
                        "calls_pct": t[2], "control_calls_pct": c[2],
                        "ticks": (k1.get("ticks", 0) - k0.get("ticks", 0)) / a.seconds,
                        "snapshots": (k1.get("taken", 0) - k0.get("taken", 0)) / a.seconds})
        B.stop_viewers(procs)
        r = np.array([x["ratio"] for x in rec])
        med = lambda key: float(np.median([x[key] for x in rec]))  # noqa: E731
        row = {"kind": kind, "viewers": k, "median": float(np.median(r)), "q1": float(np.percentile(r, 25)),
               "q3": float(np.percentile(r, 75)), "trimmed_median": med("trimmed"), "p99_us": med("p99_us"),
               "control_p99_us": med("control_p99_us"), "calls_pct": med("calls_pct"),
               "control_calls_pct": med("control_calls_pct"), "ticks_per_s": med("ticks") if k else 0.0,
               "load": [min(x["load_before"] for x in rec), max(max(x["load_before"], x["load_after"]) for x in rec)],
               "discarded": discarded, "over_load": sum(x["over_load"] for x in rec), "pairs": rec}
        rows.append(row)
        print(f"{kind:13s} {k} viewer(s): {row['median']:6.1f} % (IQR {row['q1']:.1f} to {row['q3']:.1f});"
              f" trimmed {row['trimmed_median']:.1f} %; step p99 {row['p99_us']:.0f} µs"
              f" (reference {row['control_p99_us']:.0f}); calls {row['calls_pct'] - row['control_calls_pct']:.2f} %"
              f" of the time; {med('ticks'):.0f} ticks/s, {med('snapshots'):.0f} snapshots/s;"
              f" load {row['load'][0]:.1f} to {row['load'][1]:.1f}; {discarded} pairs discarded,"
              f" {row['over_load']} over the load limit", flush=True)
    srv.close()
    os.makedirs(a.results, exist_ok=True)
    out = os.path.join(a.results, f"N-pairs-{sender}-mps-1024{a.tag}.json")
    with open(out, "w") as f:
        json.dump({"sender": sender, "pairs": a.pairs, "seconds": a.seconds, "max_load": a.max_load,
                   "arms": rows}, f, indent=1)
    print(f"wrote {os.path.relpath(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
