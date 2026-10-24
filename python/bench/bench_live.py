"""Benchmark the live session (attach to app): caliper.show routed to
caliper.live.SessionTransport, drawn by a separate caliper-viewer process over
shared GPU memory.

Training loop: the TinyCNN of examples/python/train_show.py (batch 64, Adam)
plus a synthetic weight W (S x S f32) on the same device, rewritten in place
every step (W.add_(0.001)) and shown every step with caliper.show. A
loss.item() every 100 steps (identical in every row) stands in for logging.

  bench_live.py row --mode {baseline,sampler,nosampler,unattached,...} --size S [--device mps|cpu]
  bench_live.py table [--size 1024] [--repeat 2]   -> results/S-table-<dev><S>.txt
  bench_live.py noviewer [--size 1024]             -> results/S-noviewer.txt  (near-free show)
  bench_live.py mem [--size 2048] [--n 200]        -> results/S-mem-<S>.txt
  bench_live.py liveness                           -> results/S-liveness.txt + PNGs
  bench_live.py clock                              -> results/S-clock.txt

Rows: baseline = no show, no session. sampler = show every step with the 60 Hz
per-view sampler (default). nosampler = show(rate=0): every call snapshots.
unattached = the session is open but no viewer is attached: show() must check
and return (no snapshot). record60 / record = the commit-cadence-matched
controls (torch.mps.Event.record at 60 Hz / every step, no snapshot). Each row
runs in a fresh process: 1 s warm-up (plus however long the viewer takes to
attach), then a 10 s measured window ending with a device sync. Latency comes
from the viewer's --latency-log (step-to-screen per update, viewer
CLOCK_UPTIME_RAW at encode minus the show() stamp), restricted to the measured
window. Session files go to a private $CALIPER_DATA_DIR (a temp dir).

The public path (caliper/__init__.py routing, caliper/_ring.py SnapshotRing,
caliper/live.py SessionTransport) is what every row measures.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_train_show():
    spec = importlib.util.spec_from_file_location("train_show", REPO / "examples" / "python" / "train_show.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # imports caliper (lazy: no window, no native load)
    return mod


class _RUsage(ctypes.Structure):   # rusage_info_v0 (libproc)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


def mem_of(pid: int) -> tuple[float, float]:
    """(RSS MB, phys_footprint MB) of a process."""
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    ru = _RUsage()
    if libproc.proc_pid_rusage(pid, 0, ctypes.byref(ru)) != 0:
        return float("nan"), float("nan")
    return ru.resident_size / 2**20, ru.phys_footprint / 2**20


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    pos = q * (len(v) - 1)
    i = int(pos)
    f = pos - i
    return v[i] * (1 - f) + v[i + 1] * f if i + 1 < len(v) else v[i]


def parse_stats_lines(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "] stats " not in line:
            continue
        d = {}
        for tok in line.split("] stats ", 1)[1].split():
            if "=" in tok and ":" not in tok.split("=", 1)[0]:
                k, v = tok.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            elif ":" in tok:   # per-view  name:upd=..,path=..
                name, rest = tok.split(":", 1)
                d.setdefault("per_view", {})[name] = dict(kv.split("=", 1) for kv in rest.split(",") if "=" in kv)
        out.append(d)
    return out


_SPAWNED: list = []   # viewer Popen handles this process started


def teardown_viewer():
    """Benchmarks clean up after themselves: close the session and terminate
    every viewer this process spawned (users keep theirs; benchmarks must not)."""
    try:
        import caliper
        caliper.close()
    except Exception:
        pass
    for p in _SPAWNED:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def open_session(viewer_args=(), viewer_log=None, launch=True):
    """The script's live session (what the first caliper.show opens), plus a
    viewer process with the benchmark's flags."""
    import caliper
    tr = caliper.session()
    if launch:
        tr.launch_viewer(args=viewer_args, log=viewer_log)
        _SPAWNED.append(tr.viewer)
    return tr


def close_session(tr) -> None:
    """caliper.close() (session file + shm go), then SIGTERM the viewer
    (graceful close + its final stats line)."""
    import caliper
    v = tr.viewer
    caliper.close()
    if v is not None and v.poll() is None:
        v.terminate()
        try:
            v.wait(10)
        except subprocess.TimeoutExpired:
            v.kill()


def wait_viewer_attached(tr, timeout=15.0, step=None):
    """Keep training (step()) until the spawned viewer has attached."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if tr.attached():
            return time.perf_counter() - t0
        if step:
            step()
        else:
            time.sleep(0.01)
    raise RuntimeError("viewer did not attach")


def view_stats(tr, name):
    import caliper
    st = caliper.stats().get("session", {})
    v = tr.stats()["views"].get(name, {})
    return st.get("ring", {}), v


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------
def run_row(mode: str, size: int, device: str, seconds: float, warmup: float, tag: str,
            sync_every: int = 100) -> dict:
    import torch
    import torch.nn.functional as F
    ts = load_train_show()
    import caliper
    dev = torch.device(device)
    torch.manual_seed(0)
    model = ts.TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    W = torch.randn(size, size, device=dev)
    use_show = mode not in ("baseline", "record", "record60", "idleviewer", "record60viewer", "recordviewer")
    use_viewer = mode not in ("baseline", "record", "record60", "unattached")
    rate = None if mode in ("sampler", "spike2", "unattached") else 0.0
    acts = {}
    if mode == "spike2":   # the spike's train_show.py views: fc1.weight + conv1 activation mosaic
        model.conv1.register_forward_hook(lambda m, i, o: acts.__setitem__("conv1", o.detach()))
    RESULTS.mkdir(exist_ok=True)
    vlog = RESULTS / f"S-raw-{tag}-viewer.txt"
    llog = RESULTS / f"S-raw-{tag}-latency.txt"
    tr = None
    if use_viewer:
        for p in (vlog, llog):
            p.unlink(missing_ok=True)
        tr = open_session(("--stats", "--latency-log", str(llog), "--exit-after", str(int(warmup + seconds + 60))),
                          str(vlog))
    elif mode == "unattached":
        tr = open_session(launch=False)

    state = {"loss": None, "steps": 0, "last_rec": 0.0}
    rec_ev = torch.mps.Event() if (mode.startswith("record") and dev.type == "mps") else None

    def step():
        x, y = ts.make_batch(64, dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        W.add_(0.001)
        state["steps"] += 1
        if mode in ("record", "recordviewer"):   # commit cadence of nosampler, without the snapshot
            rec_ev.record()
        elif mode in ("record60", "record60viewer"):   # commit cadence of the 60 Hz sampler, no snapshot
            now = time.perf_counter()
            if now - state["last_rec"] >= 1.0 / 60.0:
                state["last_rec"] = now
                rec_ev.record()
        if state["steps"] % sync_every == 0:
            loss.item()

    snap_us, skip_us = [], []

    def loop(dur, record):
        n = 0
        t0 = time.perf_counter()
        end = t0 + dur
        while time.perf_counter() < end:
            step()
            n += 1
            if use_show:
                a = time.perf_counter()
                if mode == "spike2":
                    r = caliper.show(model.fc1.weight, name="fc1.weight", cmap="rdbu", rate=rate)
                    caliper.show(acts["conv1"], name="act", rate=rate)
                else:
                    r = caliper.show(W, name="W", cmap="rdbu", rate=rate)
                b = time.perf_counter()
                if record:
                    (snap_us if r else skip_us).append((b - a) * 1e6)
        if dev.type == "mps":
            torch.mps.synchronize()
        return n, time.perf_counter() - t0

    loop(warmup, False)
    attach_s = None
    vname = "fc1.weight" if mode == "spike2" else "W"
    if mode in ("idleviewer", "record60viewer", "recordviewer"):
        attach_s = wait_viewer_attached(tr, step=step)   # a session and a viewer; no show()
        loop(0.5, False)
    if use_show and use_viewer:
        src0 = (lambda: model.fc1.weight) if mode == "spike2" else (lambda: W)
        attach_s = wait_viewer_attached(tr, step=lambda: (step(), caliper.show(src0(), name=vname, rate=rate)))
        loop(0.5, False)
    alloc0 = torch.mps.current_allocated_memory() if dev.type == "mps" else 0
    rss0 = mem_of(os.getpid())
    ring0, view0 = view_stats(tr, vname) if tr is not None else ({}, {})
    load0 = os.getloadavg()[0]
    t_start = time.monotonic_ns()
    steps, el = loop(seconds, True)
    t_end = time.monotonic_ns()
    load1 = os.getloadavg()[0]
    alloc1 = torch.mps.current_allocated_memory() if dev.type == "mps" else 0
    rss1 = mem_of(os.getpid())
    res = {"tag": tag, "mode": mode, "sync_every": sync_every, "size": size, "device": device, "seconds": round(el, 3),
           "steps": steps, "steps_per_s": steps / el, "torch": torch.__version__,
           "mps_alloc_delta_mb": (alloc1 - alloc0) / 2**20, "rss_delta_mb": rss1[0] - rss0[0],
           "loadavg_1m_start_end": [round(load0, 2), round(load1, 2)]}
    if mode == "unattached":
        ring1, view1 = view_stats(tr, vname)
        res.update({
            "attached": tr.attached(), "shows": len(snap_us) + len(skip_us),
            "snapshots_in_window": len(snap_us),
            "show_unattached_us_median": pct(skip_us, 0.5), "show_unattached_us_p95": pct(skip_us, 0.95),
            "ring_taken": ring1.get("taken", 0), "views_created": len(tr.stats()["views"]),
        })
        close_session(tr)
    if use_show and use_viewer:
        ring1, vw = view_stats(tr, vname)
        vpid = tr.viewer.pid
        res.update({
            "viewer_attach_s": attach_s,
            "shows": len(snap_us) + len(skip_us), "snapshots_in_window": len(snap_us),
            "show_snapshot_us_median": pct(snap_us, 0.5), "show_snapshot_us_p95": pct(snap_us, 0.95),
            "show_skipped_us_median": pct(skip_us, 0.5), "show_skipped_us_p95": pct(skip_us, 0.95),
            "published_in_window": ring1.get("published", 0) - ring0.get("published", 0),
            "ring_full_in_window": ring1.get("ring_full", 0) - ring0.get("ring_full", 0),
            "busy_dropped_total": vw.get("dropped_client", 0),
            "viewer_seq": vw.get("viewer_seq"), "viewer_skipped_total": vw.get("viewer_skipped"),
            "viewer_rss_mb_end": mem_of(vpid)[0] if vpid else None,
        })
        time.sleep(2.5)   # one more viewer stats window (flushes the latency log)
        close_session(tr)   # SIGTERM -> graceful close + final stats line
        lat = []
        if llog.exists():
            for line in llog.read_text().split("\n"):
                if line.strip():
                    t, ms = line.split()
                    if t_start <= int(t) <= t_end:
                        lat.append(float(ms))
        stats_lines = parse_stats_lines(vlog)
        fps = [d["fps"] for d in stats_lines[1:-1] if "fps" in d]
        paths = {f"{n}:{pv.get('path')}" for d in stats_lines for n, pv in d.get("per_view", {}).items()}
        imported = {pv.get("imported") for d in stats_lines for pv in d.get("per_view", {}).values()}
        vrss = [d.get("rss_mb") for d in stats_lines if "rss_mb" in d]
        vfp = [d.get("footprint_mb") for d in stats_lines if "footprint_mb" in d]
        res.update({
            "latency_n": len(lat), "latency_ms_median": pct(lat, 0.5), "latency_ms_p95": pct(lat, 0.95),
            "latency_ms_max": max(lat) if lat else float("nan"),
            "viewer_fps_median": statistics.median(fps) if fps else float("nan"),
            "viewer_fps_min": min(fps) if fps else float("nan"),
            "viewer_paths": sorted(p for p in paths if p and not p.endswith(":-")),
            "viewer_imported": sorted(i for i in imported if i),
            "viewer_rss_mb_first_last": [vrss[0], vrss[-1]] if vrss else None,
            "viewer_footprint_mb_first_last": [vfp[0], vfp[-1]] if vfp else None,
        })
    if mode in ("idleviewer", "record60viewer", "recordviewer"):
        res["viewer_attach_s"] = attach_s
        close_session(tr)
    return res


# ---------------------------------------------------------------------------
# table driver
# ---------------------------------------------------------------------------
def run_child(args_list) -> dict:
    env = dict(os.environ)
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args_list],
                       capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"row failed: {' '.join(args_list)}\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")


TABLE_MODES = ("baseline", "unattached", "record60", "sampler", "record", "nosampler")
MATCHED = {"sampler": "record60", "nosampler": "record", "spike2": "record60"}


def fmt_row(r, base, ctl):
    """base: plain baseline steps/s; ctl: steps/s of the commit-cadence-matched control."""
    pb = 100.0 * r["steps_per_s"] / base if base else float("nan")
    pc = 100.0 * r["steps_per_s"] / ctl if ctl else float("nan")
    head = f"| {r['device']} | {r['size']}² | {r['mode']} | {r['steps_per_s']:.1f} | {pb:.1f} % "
    if r["mode"] == "unattached":
        return head + (f"| - | - | {r['show_unattached_us_median']:.2f} (unattached) | - | - "
                       f"| {r['snapshots_in_window']} / - | no viewer |")
    if "show_snapshot_us_median" not in r:
        return head + "| - | - | - | - | - | - | - |"
    return (head + f"| {pc:.1f} % | {r['show_snapshot_us_median']:.1f} | {r['show_skipped_us_median']:.1f} "
            f"| {r['latency_ms_median']:.2f} / {r['latency_ms_p95']:.2f} (n={r['latency_n']}) "
            f"| {r['viewer_fps_median']:.1f} | {r['snapshots_in_window']} / {r['busy_dropped_total']} "
            f"| {','.join(r['viewer_paths'])} |")


def load_tag(r):
    return f" load {r['loadavg_1m_start_end'][0]:.1f}->{r['loadavg_1m_start_end'][1]:.1f}"


def run_table(device, size, modes, repeat, seconds, warmup, out_name):
    """One size per invocation (keeps a run under ~2 min); modes interleaved per repeat."""
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for rep in range(1, repeat + 1):
        for m in modes:
            tag = f"{device}{size}-{m}-r{rep}"
            print(f"[table] {tag} ...", flush=True)
            r = run_child(["row", "--mode", m, "--size", str(size), "--device", device,
                           "--seconds", str(seconds), "--warmup", str(warmup), "--tag", tag])
            r["repeat"] = rep
            rows.append(r)
            print(f"    {r['steps_per_s']:.1f} steps/s", flush=True)
    hdr = ("| device | W | mode | steps/s | % of baseline | % of commit-matched control "
           "| show µs snapshot (median) | show µs skipped (median) | step-to-screen ms median / p95 "
           "| viewer fps | snapshots / busy-dropped | viewer path |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [f"# caliper.show -> live session (S) {device} {size}², {time.strftime('%Y-%m-%d %H:%M:%S')}, "
             f"{seconds:.0f} s measured after {warmup:.0f} s warm-up, {repeat} repeat(s); "
             f"matched control: record60 for sampler/spike2, record for nosampler", hdr]
    for rep in range(1, repeat + 1):
        by = {r["mode"]: r["steps_per_s"] for r in rows if r["repeat"] == rep}
        for r in rows:
            if r["repeat"] == rep:
                lines.append(fmt_row(r, by.get("baseline"), by.get(MATCHED.get(r["mode"], ""), None))
                             + f" r{rep}" + load_tag(r))
    lines += ["", "# raw rows (JSON)"] + [json.dumps(r) for r in rows]
    out = RESULTS / out_name
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2 + len(rows)]))
    print(f"wrote {out}")


DIAG_MODES = ("baseline", "record60", "sampler", "record", "nosampler", "idleviewer")
DIAG2_MODES = ("record60viewer", "sampler", "recordviewer", "nosampler")


def run_diag(size: int, repeat: int, seconds: float, warmup: float, modes=DIAG_MODES, name="diag"):
    """Why can show() rows beat the baseline? Same loop, equal warm-up, modes
    interleaved: baseline / record (Event.record per step, no snapshot) /
    idleviewer (viewer attached, no show) / sampler / nosampler."""
    rows = []
    for rep in range(1, repeat + 1):
        for m in modes:
            tag = f"{name}-mps{size}-{m}-r{rep}"
            print(f"[diag] {tag} ...", flush=True)
            r = run_child(["row", "--mode", m, "--size", str(size), "--seconds", str(seconds),
                           "--warmup", str(warmup), "--tag", tag])
            r["repeat"] = rep
            rows.append(r)
            print(f"    {r['steps_per_s']:.1f} steps/s  load {r['loadavg_1m_start_end']}", flush=True)
    lines = [f"# live session (S) diagnostic: baseline vs commit cadence vs idle viewer, MPS W {size}², "
             f"{seconds:.0f} s after {warmup:.0f} s warm-up, {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "| mode | " + " | ".join(f"r{i} steps/s" for i in range(1, repeat + 1)) + " |",
             "|---|" + "---|" * repeat]
    for m in modes:
        vals = [f"{r['steps_per_s']:.1f} (load {r['loadavg_1m_start_end'][0]:.1f})" for r in rows if r["mode"] == m]
        lines.append(f"| {m} | " + " | ".join(vals) + " |")
    lines += ["", "# raw rows (JSON)"] + [json.dumps(r) for r in rows]
    (RESULTS / f"S-{name}-{size}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


def run_mem(size: int, n: int):
    import torch
    import caliper
    RESULTS.mkdir(exist_ok=True)
    vlog = RESULTS / f"S-raw-mem{size}-viewer.txt"
    tr = open_session(("--stats", "--exit-after", "90"), str(vlog))
    W = torch.randn(size, size, device="mps")
    wait_viewer_attached(tr, step=lambda: (W.add_(0.001), caliper.show(W, name="W", rate=0)))
    for _ in range(50):
        W.add_(0.001)
        caliper.show(W, name="W", rate=0)
    caliper.flush()
    time.sleep(0.5)
    vpid = tr.viewer.pid
    a0, d0 = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    r0, f0 = mem_of(os.getpid())
    vr0, vf0 = mem_of(vpid)
    times, busy_times = [], []
    calls = 0
    deadline = time.perf_counter() + 30.0
    # n snapshots actually taken (busy calls are dropped). A sync every 10 calls
    # bounds the CPU run-ahead like any logging loop does: a sync-free loop of
    # W.add_ lets torch queue commands without limit (B-backlog.txt), and the
    # pending snapshots then wait behind that backlog.
    while len(times) < n and time.perf_counter() < deadline:
        W.add_(0.001)
        t = time.perf_counter()
        took = caliper.show(W, name="W", rate=0)
        (times if took else busy_times).append((time.perf_counter() - t) * 1e6)
        calls += 1
        if calls % 10 == 0:
            torch.mps.synchronize()
    caliper.flush()
    time.sleep(1.0)
    a1, d1 = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    r1, f1 = mem_of(os.getpid())
    vr1, vf1 = mem_of(vpid)
    ring, st = view_stats(tr, "W")
    time.sleep(2.2)
    close_session(tr)
    paths = {d["per_view"]["W"].get("path") + "/imported=" + d["per_view"]["W"].get("imported")
             for d in parse_stats_lines(vlog) if "per_view" in d and "W" in d["per_view"]}
    lines = [
        f"# live session (S) memory over {n} snapshots of a {size}x{size} f32 MPS tensor (sampler off), "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"trainer torch.mps.current_allocated_memory delta: {(a1 - a0) / 2**20:+.3f} MB  ({a0} -> {a1} bytes)",
        f"trainer torch.mps.driver_allocated_memory delta:  {(d1 - d0) / 2**20:+.3f} MB",
        f"trainer RSS delta: {r1 - r0:+.2f} MB ({r0:.1f} -> {r1:.1f}); phys_footprint delta {f1 - f0:+.2f} MB",
        f"viewer  RSS delta: {vr1 - vr0:+.2f} MB ({vr0:.1f} -> {vr1:.1f}); phys_footprint delta {vf1 - vf0:+.2f} MB",
        f"show() per snapshot: median {pct(times, 0.5):.1f} us, p95 {pct(times, 0.95):.1f} us; "
        f"{calls} calls for {n} snapshots ({len(busy_times)} busy-dropped, median {pct(busy_times, 0.5):.1f} us)",
        f"published {ring.get('published')} ring_full {ring.get('ring_full', 0)} viewer_seq {st.get('viewer_seq')}",
        f"viewer view path(s): {sorted(paths)}",
        f"shm segment {st.get('segment')} bytes {st.get('segment_bytes')}",
    ]
    out = RESULTS / f"S-mem-{size}.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_noviewer(size: int, n: int = 100_000):
    """The near-free path: a session is open, no viewer attached. Per-call
    cost of attached() and of caliper.show() (MPS and CPU tensors), timed
    singly (median, p99) and in a loop (mean); plus what it did NOT do."""
    import torch
    import caliper
    tr = open_session(launch=False)
    clock = time.perf_counter_ns

    def single(fn, k):
        ts = []
        for _ in range(k):
            a = clock()
            fn()
            ts.append(clock() - a)
        ts.sort()
        return ts[len(ts) // 2] / 1e3, ts[int(len(ts) * 0.99)] / 1e3

    def looped(fn, k):
        a = clock()
        for _ in range(k):
            fn()
        return (clock() - a) / k / 1e3

    lines = [f"# near-free show with no viewer attached ({time.strftime('%Y-%m-%d %H:%M:%S')}), "
             f"torch {torch.__version__}, {n} calls per row; session {tr.session}",
             "| call | median us (timed singly) | p99 us | mean us (loop) |", "|---|---|---|---|"]
    rows = [("SessionTransport.attached()", tr.attached)]
    Wm = torch.randn(size, size, device="mps") if torch.backends.mps.is_available() else None
    Wc = torch.randn(256, 256)
    if Wm is not None:
        rows.append((f"caliper.show(W {size}x{size} mps)", lambda: caliper.show(Wm, name="W")))
        rows.append((f"caliper.show(W {size}x{size} mps, rate=0)", lambda: caliper.show(Wm, name="W", rate=0)))
    rows.append(("caliper.show(W 256x256 cpu)", lambda: caliper.show(Wc, name="Wc")))
    rows.append(("caliper.series(1.0)", lambda: caliper.series(1.0, name="loss")))
    for label, fn in rows:
        fn()
        med, p99 = single(fn, n)
        mean = looped(fn, n)
        lines.append(f"| {label} | {med:.3f} | {p99:.3f} | {mean:.3f} |")
    ring, _ = view_stats(tr, "W")
    lines += ["", f"attached() = {tr.attached()}; ring counters after all rows: {dict(ring)}; "
              f"views created in the session: {list(tr.stats()['views'])} (none: no snapshot, no slot, no GPU work)"]
    close_session(tr)
    (RESULTS / "S-noviewer.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# liveness: the trainer sleeps 5 s while the viewer is showing live content
# ---------------------------------------------------------------------------
def _cg():
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cg.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
    cg.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    return cg, cf


class _CGRect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("w", ctypes.c_double), ("h", ctypes.c_double)]


def find_windows(pid: int) -> list[dict]:
    """CGWindowListCopyWindowInfo, filtered by owner pid: [{id, bounds}]."""
    cg, cf = _cg()
    key = lambda n: ctypes.c_void_p.in_dll(cg, n)
    k_num, k_pid, k_bounds = key("kCGWindowNumber"), key("kCGWindowOwnerPID"), key("kCGWindowBounds")
    cg.CGRectMakeWithDictionaryRepresentation.restype = ctypes.c_bool
    cg.CGRectMakeWithDictionaryRepresentation.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CGRect)]
    arr = cg.CGWindowListCopyWindowInfo(1 | 16, 0)   # on-screen only, exclude desktop elements
    out = []
    for i in range(cf.CFArrayGetCount(arr)):
        d = cf.CFArrayGetValueAtIndex(arr, i)
        v = ctypes.c_int64()
        cf.CFNumberGetValue(cf.CFDictionaryGetValue(d, k_pid), 4, ctypes.byref(v))
        if v.value != pid:
            continue
        cf.CFNumberGetValue(cf.CFDictionaryGetValue(d, k_num), 4, ctypes.byref(v))
        r = _CGRect()
        cg.CGRectMakeWithDictionaryRepresentation(cf.CFDictionaryGetValue(d, k_bounds), ctypes.byref(r))
        out.append({"id": v.value, "bounds": (r.x, r.y, r.w, r.h)})
    cf.CFRelease(arr)
    return out


def _trainer_sleep(size: int, before: float, sleep_s: float, after: float):
    """Child process for `liveness`: train + show, then time.sleep, then resume."""
    import torch
    import torch.nn.functional as F
    import caliper
    ts = load_train_show()
    dev = torch.device("mps")
    model = ts.TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    W = torch.randn(size, size, device=dev)
    vlog = RESULTS / "S-liveness-viewer.txt"
    tr = open_session(("--stats", "--exit-after", "60"), str(vlog))

    def step():
        x, y = ts.make_batch(64, dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        W.mul_(0.999).add_(torch.randn_like(W), alpha=0.05)   # visibly changing content
        caliper.show(W, name="W", cmap="rdbu")
        caliper.show(model.conv1.weight[:, 0], name="conv1")

    step()
    wait_viewer_attached(tr, step=step)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < before:
        step()
    print(f"VIEWER {tr.viewer.pid}", flush=True)
    print(f"SLEEP_START {time.monotonic_ns()}", flush=True)
    time.sleep(sleep_s)                    # the "training loop" blocks here
    print(f"SLEEP_END {time.monotonic_ns()}", flush=True)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < after:
        step()
        n += 1
    print(f"RESUMED steps={n} stats={json.dumps(view_stats(tr, 'W')[1])}", flush=True)
    time.sleep(2.2)
    close_session(tr)


def run_liveness(size: int):
    RESULTS.mkdir(exist_ok=True)
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_trainer_sleep",
                              "--size", str(size)], stdout=subprocess.PIPE, text=True)
    log = []

    def note(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.append(line)

    vpid = None
    while True:
        line = child.stdout.readline()
        if not line:
            raise RuntimeError("trainer exited early")
        if line.startswith("VIEWER "):
            vpid = int(line.split()[1])
        if line.startswith("SLEEP_START"):
            t_sleep = time.perf_counter()
            break
    wins = find_windows(vpid)
    note(f"viewer pid {vpid}; CGWindowList windows of that pid: {wins}")
    wid = max(wins, key=lambda w: w["bounds"][2] * w["bounds"][3])["id"]

    def at(t):
        d = t_sleep + t - time.perf_counter()
        if d > 0:
            time.sleep(d)

    shots = []
    at(1.0)
    p1 = RESULTS / "S-liveness-t1.png"
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(p1)], check=True)
    note(f"t=1.0 s into the trainer's sleep: screencapture -l {wid} -> {p1}")
    shots.append(p1)
    at(1.5)
    os.kill(vpid, signal.SIGUSR1)   # SIGUSR1: viewer posts a mouse-move NSEvent onto its status window
    note("t=1.5 s: SIGUSR1 -> viewer posts move+click on its status window title bar through its own AppKit event queue")
    at(3.0)
    p3 = RESULTS / "S-liveness-t3.png"
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(p3)], check=True)
    note(f"t=3.0 s: screencapture -> {p3}")
    shots.append(p3)
    at(3.5)
    os.kill(vpid, signal.SIGUSR2)   # SIGUSR2: mouse-move to the window centre (status window no longer hovered)
    note("t=3.5 s: SIGUSR2 -> mouse-move to the window centre")
    at(4.5)
    p4 = RESULTS / "S-liveness-t4.5.png"
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(p4)], check=True)
    note(f"t=4.5 s: screencapture -> {p4}")
    rest = child.stdout.read()
    child.wait()
    for line in rest.splitlines():
        note("trainer: " + line)
    import hashlib
    hs = [hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in (p1, p3, p4)]
    note(f"png sha256 prefixes t1 {hs[0]} t3 {hs[1]} t4.5 {hs[2]} (differ = repainted)")
    vlog = RESULTS / "S-liveness-viewer.txt"
    for line in vlog.read_text().splitlines():
        if "stats " in line or "input-probe" in line:
            note("viewer: " + line.split("] ", 1)[1][:220])
    (RESULTS / "S-liveness.txt").write_text("\n".join(log) + "\n")


def run_clock():
    from caliper import _live
    lines = ["# Python time.monotonic_ns() vs caliper_live_now_ns() (CLOCK_UPTIME_RAW), same process"]
    import time as _t
    lines.append(f"time.get_clock_info('monotonic') = {_t.get_clock_info('monotonic')}")
    for _ in range(5):
        a = _t.monotonic_ns()
        b = _live.now_ns()
        c = _t.monotonic_ns()
        lines.append(f"monotonic_ns {a}  uptime_raw {b}  (uptime_raw - monotonic) {b - a} ns  next monotonic {c - b:+d} ns")
    (RESULTS / "S-clock.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    import atexit
    import shutil
    import tempfile
    if not os.environ.get("CALIPER_DATA_DIR"):
        # Session files of the benchmark go to a private root (children inherit it).
        d = tempfile.mkdtemp(prefix="caliper-bench-")
        os.environ["CALIPER_DATA_DIR"] = d
        atexit.register(shutil.rmtree, d, True)
    atexit.register(teardown_viewer)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("row")
    r.add_argument("--mode", choices=["baseline", "sampler", "nosampler", "spike2", "record", "record60", "idleviewer",
                            "record60viewer", "recordviewer", "unattached"],
                   required=True)
    r.add_argument("--size", type=int, default=1024)
    r.add_argument("--device", default="mps")
    r.add_argument("--seconds", type=float, default=10.0)
    r.add_argument("--warmup", type=float, default=1.0)
    r.add_argument("--tag", default="row")
    r.add_argument("--sync-every", type=int, default=100, help="loss.item() every N steps")
    t = sub.add_parser("table")
    t.add_argument("--size", type=int, default=1024)
    t.add_argument("--device", default="mps")
    t.add_argument("--modes", nargs="+", default=list(TABLE_MODES))
    t.add_argument("--repeat", type=int, default=1)
    t.add_argument("--out", default=None)
    t.add_argument("--seconds", type=float, default=10.0)
    t.add_argument("--warmup", type=float, default=1.0)
    dg = sub.add_parser("diag")
    dg.add_argument("--size", type=int, default=1024)
    dg.add_argument("--repeat", type=int, default=2)
    dg.add_argument("--seconds", type=float, default=10.0)
    dg.add_argument("--warmup", type=float, default=3.0)
    dg.add_argument("--matched", action="store_true", help="viewer-matched controls (record60viewer, recordviewer)")
    m = sub.add_parser("mem")
    m.add_argument("--size", type=int, default=2048)
    m.add_argument("--n", type=int, default=200)
    lv = sub.add_parser("liveness")
    lv.add_argument("--size", type=int, default=1024)
    ts_ = sub.add_parser("_trainer_sleep")
    ts_.add_argument("--size", type=int, default=1024)
    sub.add_parser("clock")
    nv = sub.add_parser("noviewer")
    nv.add_argument("--size", type=int, default=1024)
    nv.add_argument("--n", type=int, default=100_000)
    a = ap.parse_args()
    if a.cmd == "row":
        res = run_row(a.mode, a.size, a.device, a.seconds, a.warmup, a.tag, a.sync_every)
        print("RESULT " + json.dumps(res), flush=True)
    elif a.cmd == "table":
        run_table(a.device, a.size, a.modes, a.repeat, a.seconds, a.warmup,
                  a.out or f"S-table-{a.device}{a.size}.txt")
    elif a.cmd == "diag":
        if a.matched:
            run_diag(a.size, a.repeat, a.seconds, a.warmup, DIAG2_MODES, "diag-matched")
        else:
            run_diag(a.size, a.repeat, a.seconds, a.warmup)
    elif a.cmd == "mem":
        run_mem(a.size, a.n)
    elif a.cmd == "liveness":
        run_liveness(a.size)
    elif a.cmd == "_trainer_sleep":
        _trainer_sleep(a.size, before=2.0, sleep_s=5.0, after=2.0)
    elif a.cmd == "clock":
        run_clock()
    elif a.cmd == "noviewer":
        run_noviewer(a.size, a.n)


if __name__ == "__main__":
    main()
