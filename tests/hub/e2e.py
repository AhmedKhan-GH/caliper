"""The hub end to end, headless: `caliper serve`, two training scripts, a Python
viewer and the native viewer, as separate processes on 127.0.0.1.

    PYTHONPATH=python CALIPER_SESSION=0 python tests/hub/e2e.py --build cmake-build-release

It needs a build with the `caliper` and `caliper_viewer` targets, and a
Python environment with torch, numpy and websockets. It checks, in order:

1. `caliper serve --port 0` starts and writes its hub file (mode 0600).
2. Two scripts (tests/hub/producer.py) connect to it as producers and train
   small models; both sessions are listed as live.
3. A Python viewer (tests/hub/viewer.py) attaches to each session and receives
   snapshots of both weight heatmaps and the loss series from the start.
4. The native viewer, `caliper_viewer --headless --url <link>&session=<id>
   --exit-after 5 --stats`, attaches to each session and draws every view.
5. The test tells the second script to end: its session is listed as ended, and a viewer that
   attaches afterwards still gets its last snapshots and its whole loss series.
6. The hub's resident memory does not grow with time: it grows no more in a
   12 s window than in the 6 s window before it, but for what is in flight.
7. Protocol 3.1 controls. Script e2e-a registers the parameter "lr" and the
   action "save"; a viewer that announced `controls` sees them through the hub
   but its CONTROLs are answered `forbidden`, since the hub runs without
   --allow-control. A second hub started with --allow-control forwards them:
   the script sets "lr" at its next checkpoint and runs "save", and both
   answers reach the viewer.
8. SIGTERM stops the hub with exit code 0, it removes its hub file, and no
   process this script started is left.

With --connect the two scripts are tests/hub/producer_connect.py instead,
which stream with the package's own caliper.connect() (the native sender
must be built: python scripts/build_netio.py), and their sessions are found
in the session list by pid. They take the controls of step 7 through the
package: caliper.param(), caliper.action() and caliper.step(), and a viewer
pauses one of them through the hub: the run pauses at a step and the viewer
then holds exactly that step's loss values. --hub-pause-order also checks the
protocol's order through the hub, that the paused state reached the viewer
after those values (a hub delivers a run state after the data it received
before it); without it the order seen is printed. With --bench it then measures the relay with a
512 x 512 heatmap at 60 Hz to four Python viewers in other processes. It
prints one line per check and exits with 0 when all of them pass.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viewer  # noqa: E402  (tests/hub/viewer.py)

FAILED = []


def check(ok: bool, what: str) -> bool:
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILED.append(what)
    return ok


def rss_mib(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0


def wait_hub_file(path: str, timeout: float = 30.0) -> dict | None:
    """The hub file's content, once the hub has written it whole."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with open(path) as f:
                h = json.load(f)
            if h.get("stream_url") and h.get("token"):
                return h
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    return None


def wait_session_state(url: str, sid: str, state: str, timeout: float = 60.0) -> str | None:
    """Polls the hub's session list until session `sid` has `state`."""
    end = time.monotonic() + timeout
    got = None
    while time.monotonic() < end:
        got = {s["id"]: s["state"] for s in asyncio.run(viewer.sessions(url))}.get(sid)
        if got == state:
            break
        time.sleep(0.1)
    return got


def stop(proc, stop_file: str, timeout: float = 60.0) -> int | None:
    """Tells a test script to end (it trains until its --stop-file exists) and
    waits for it to exit."""
    open(stop_file, "w").close()
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def native_viewer(exe: str, link: str, data_dir: str, seconds: int = 5) -> dict:
    r = subprocess.run([exe, "--headless", "--url", link, "--exit-after", str(seconds), "--stats",
                        "--data-dir", data_dir], capture_output=True, text=True, timeout=seconds + 30)
    views = {}
    session = None
    for line in r.stderr.splitlines():
        m = re.search(r"received view=(\S+) kind=(\S+) updates=(\d+) seq=(\d+) series=(\d+)", line)
        if m:
            views[m.group(1)] = {"kind": m.group(2), "updates": int(m.group(3)), "series": int(m.group(5))}
        m = re.search(r"received session=(\S+) .* alive=(\d)", line)
        if m:
            session = (m.group(1), int(m.group(2)))
    return {"code": r.returncode, "views": views, "session": session, "stderr": r.stderr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="cmake-build-release", help="the CMake build folder")
    ap.add_argument("--work", help="a folder for the hub file and logs (default: a new temporary one)")
    ap.add_argument("--bench", action="store_true", help="also measure the relay at 60 Hz to four viewers")
    ap.add_argument("--connect", action="store_true",
                    help="the scripts stream with caliper.connect() (tests/hub/producer_connect.py)")
    ap.add_argument("--hub-pause-order", action="store_true",
                    help="with --connect, check that the hub delivers a pause's run state after the "
                         "data it received before it")
    a = ap.parse_args()
    caliper = os.path.join(a.build, "caliper")
    caliper_viewer = os.path.join(a.build, "caliper_viewer")
    for exe in (caliper, caliper_viewer):
        if not os.access(exe, os.X_OK):
            print(f"missing {exe}: build the caliper and caliper_viewer targets", file=sys.stderr)
            return 2
    work = a.work or tempfile.mkdtemp(prefix="caliper-hub-e2e-")
    os.makedirs(work, exist_ok=True)
    hub_file = os.path.join(work, "hub.json")
    env = dict(os.environ, CALIPER_HUB_FILE=hub_file, CALIPER_SESSION="0")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                                   "python"), env.get("PYTHONPATH")]))
    procs: list[subprocess.Popen] = []
    rss: list[tuple[float, float]] = []

    def spawn(args, log, extra_env=None):
        p = subprocess.Popen(args, env=dict(env, **(extra_env or {})), stdout=open(os.path.join(work, log), "w"),
                             stderr=subprocess.STDOUT)
        procs.append(p)
        return p

    hub = None
    try:
        # 1. the hub
        hub = spawn([caliper, "serve", "--port", "0"], "hub.log")
        h = wait_hub_file(hub_file)
        if not check(h is not None, "caliper serve writes its hub file"):
            return 1
        check(stat.S_IMODE(os.stat(hub_file).st_mode) == 0o600, "the hub file is readable by its owner only")
        url = f"{h['stream_url']}?token={h['token']}"
        rss.append((time.monotonic(), rss_mib(hub.pid)))

        # 2. two scripts, which train until the test tells them to end
        stop_a, stop_b = os.path.join(work, "stop-a"), os.path.join(work, "stop-b")
        if a.connect:
            pa = spawn([sys.executable, os.path.join(HERE, "producer_connect.py"), "--seconds", "600",
                        "--stop-file", stop_a, "--controls"], "producer-a.log")
            pb = spawn([sys.executable, os.path.join(HERE, "producer_connect.py"), "--seconds", "600",
                        "--stop-file", stop_b], "producer-b.log")
        else:
            pa = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file, "--id", "e2e-a",
                        "--seconds", "600", "--stop-file", stop_a, "--controls"], "producer-a.log")
            pb = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file, "--id", "e2e-b",
                        "--seconds", "600", "--stop-file", stop_b], "producer-b.log")
        ida, idb = "e2e-a", "e2e-b"
        listed = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            got = asyncio.run(viewer.sessions(url))
            if a.connect:                       # the package names its session; find it by pid
                by_pid = {s.get("pid"): s["id"] for s in got}
                ida, idb = by_pid.get(pa.pid, ida), by_pid.get(pb.pid, idb)
            listed = {s["id"]: s["state"] for s in got}
            # caliper.connect() lists a session before its views exist: they come
            # with the script's first view calls, so wait for all three
            nviews = {s["id"]: s.get("views", 0) for s in got}
            if listed.get(ida) == "live" and listed.get(idb) == "live" and \
                    (not a.connect or (nviews.get(ida) == 3 and nviews.get(idb) == 3)):
                break
            time.sleep(0.3)
        check(listed.get(ida) == "live" and listed.get(idb) == "live",
              f"both scripts are listed as live sessions: {listed}")

        # 3. a Python viewer on each session. The first snapshot of a view can be the newest one
        # that the script had already made; a second one proves that the stream goes on.
        enough = lambda snaps, vals: (snaps.get("fc1.weight", 0) >= 2 and snaps.get("fc2.weight", 0) >= 2  # noqa: E731
                                      and vals.get("loss", 0) >= 1)
        for sid in (ida, idb):
            r = asyncio.run(viewer.watch(url, sid, 60.0, until=enough))
            rss.append((time.monotonic(), rss_mib(hub.pid)))
            snaps = r["snapshots"]
            loss = r["series"].get("loss", {})
            check(r["session"] == sid and not r["errors"], f"the Python viewer attaches to {sid} ({r['errors']})")
            check(snaps.get("fc1.weight", 0) >= 2 and snaps.get("fc2.weight", 0) >= 2,
                  f"it receives snapshots of both heatmaps from {sid}: {snaps}")
            check(loss.get("first") == 0 and loss.get("contiguous") and loss.get("values", 0) >= 1,
                  f"it receives the loss series of {sid} from index 0: {loss}")

        # 4. the native viewer on each session
        link = h["url"]
        for sid in (ida, idb):
            n = native_viewer(caliper_viewer, f"{link}&session={sid}", work)
            rss.append((time.monotonic(), rss_mib(hub.pid)))
            views = n["views"]
            check(n["code"] == 0 and n["session"] == (sid, 1),
                  f"caliper_viewer --headless attaches to the live session {sid} (exit {n['code']}, {n['session']})")
            check(all(views.get(v, {}).get("updates", 0) > 0 for v in ("fc1.weight", "fc2.weight", "loss"))
                  and views.get("loss", {}).get("series", 0) > 0,
                  f"it draws every view of {sid}: {views}")
            if FAILED and FAILED[-1].startswith("it draws"):
                print(n["stderr"], file=sys.stderr)

        # 5. the second script ends
        code_b = stop(pb, stop_b)
        check(code_b == 0, f"script {idb} finished (exit {code_b})")
        with open(os.path.join(work, "producer-b.log")) as f:
            tail = [l for l in f.read().splitlines() if l.startswith("{")]
        b_done = json.loads(tail[-1]) if tail else {}
        wait_session_state(url, idb, "ended")
        whole = lambda snaps, vals: (snaps.get("fc1.weight", 0) >= 1 and snaps.get("fc2.weight", 0) >= 1  # noqa: E731
                                     and vals.get("loss", 0) >= b_done.get("loss_values", 1))
        r = asyncio.run(viewer.watch(url, idb, 30.0, until=whole))
        rss.append((time.monotonic(), rss_mib(hub.pid)))
        check(r["state"] == "ended", f"its session is listed as ended: {r['state']}")
        check(r["snapshots"].get("fc1.weight", 0) >= 1 and r["snapshots"].get("fc2.weight", 0) >= 1,
              f"a viewer that attaches afterwards gets its last snapshots: {r['snapshots']}")
        loss = r["series"].get("loss", {})
        check(loss.get("first") == 0 and loss.get("contiguous") and loss.get("values") == b_done.get("loss_values"),
              f"and its whole loss series: {loss.get('values')} of {b_done.get('loss_values')} values")
        n = native_viewer(caliper_viewer, f"{link}&session={idb}", work, seconds=3)
        check(n["code"] == 0 and n["views"].get("fc1.weight", {}).get("updates", 0) > 0,
              f"the native viewer shows the ended session too: {n['views']} {n['session']}")
        r = asyncio.run(viewer.watch(url, ida, 30.0, until=lambda snaps, vals: snaps.get("fc1.weight", 0) > 0))
        check(r["state"] == "live" and r["snapshots"].get("fc1.weight", 0) > 0, "the other session goes on")

        # 6. the hub's memory: a sample now, after a window of 6 s and after the window of 12 s
        # that follows it. A leak of r MiB/s grows the second window by 6 r more than the first,
        # and memory that settles grows the first window only. What may still differ between the
        # windows is what the hub holds and frees, and what it retains by design: a second copy
        # of each session's two heatmaps while a newer one replaces them (64 x 32 and 2 x 64
        # floats), and 6 s more of the loss series it keeps, one float a step at no more than
        # 250 steps a second (the 4 ms sleep of each step of both producers). The resident size
        # counts whole pages, and an allocation can touch one page more than its size fills.
        # Measured on 2026-11-28 on Apple silicon over 5 runs: the 6 s window grew 0 to
        # 0.031 MiB and the 12 s window 0 to 0.016 MiB.
        page = os.sysconf("SC_PAGE_SIZE")
        pages = lambda n: (-(-n // page) + 1) * page  # noqa: E731  (the bytes of the pages n bytes can touch)
        in_flight = 2 * (pages(64 * 32 * 4) + pages(2 * 64 * 4) + pages(6 * 250 * 4)) / 2**20
        m0 = rss_mib(hub.pid)
        time.sleep(6.0)
        m1 = rss_mib(hub.pid)
        time.sleep(12.0)
        m2 = rss_mib(hub.pid)
        grow_short, grow_long = m1 - m0, m2 - m1
        if m0 > 0:
            check(grow_long - grow_short <= in_flight,
                  f"the hub's resident memory does not grow with time: {grow_short:+.3f} MiB in 6 s, then "
                  f"{grow_long:+.3f} MiB in 12 s ({m0:.1f} MiB at the first sample, "
                  f"{rss[0][1]:.1f} MiB at start; at most {in_flight:.3f} MiB in flight)")

        # optional: the relay at 60 Hz to four viewers in other processes
        if a.bench:
            bench(spawn, url, hub_file, work)

        # 7. controls (protocol 3.1): refused by this hub, forwarded by one with --allow-control
        controls(spawn, url, caliper, work, ida, a.connect, a.hub_pause_order)

        # 8. stop
        if stop(pa, stop_a, timeout=30) is None:
            pa.terminate()
        hub.send_signal(signal.SIGTERM)
        try:
            code = hub.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        check(code == 0, f"SIGTERM stops the hub with exit code 0 ({code})")
        check(not os.path.exists(hub_file), "the hub removes its hub file")
        with open(os.path.join(work, "hub.log")) as f:
            log = f.read()
        check(f"session {idb} ended" in log and "hub stopped" in log, "the hub logged the sessions and its stop")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        left = [p.pid for p in procs if p.poll() is None]
        check(not left, "no process started by this test is left")
        if not a.work:
            shutil.rmtree(work, ignore_errors=True)
    print(f"{'all checks pass' if not FAILED else str(len(FAILED)) + ' checks failed'}", flush=True)
    return 0 if not FAILED else 1


def controls(spawn, url: str, caliper: str, work: str, session: str, connect: bool,
             hub_pause_order: bool = False) -> None:
    r = asyncio.run(viewer.control(url, session, 0.05, seconds=60.0))
    check("controls" in r["capabilities"] and r["control"] is False,
          f"the hub announces controls and says this viewer may not control: {r['capabilities']} {r['control']}")
    check(r["controls"] >= 1 and r["lr"] == 0.001,
          f"the viewer sees the script's run state and its parameter lr = {r['lr']}")
    refused = [x for k in (1, 2) for x in r["replies"][k]]
    check(len(refused) == 2 and all(x["error"] == "forbidden" for x in refused),
          f"without --allow-control both CONTROLs are answered forbidden: {r['replies']}")

    hub_file2 = os.path.join(work, "hub2.json")
    hub2 = spawn([caliper, "serve", "--port", "0", "--allow-control"], "hub2.log", {"CALIPER_HUB_FILE": hub_file2})
    h2 = wait_hub_file(hub_file2)
    if not check(h2 is not None, "a hub started with --allow-control writes its hub file"):
        return
    url2 = f"{h2['stream_url']}?token={h2['token']}"
    stop_c = os.path.join(work, "stop-ctl")
    if connect:
        pc = spawn([sys.executable, os.path.join(HERE, "producer_connect.py"), "--seconds", "600",
                    "--stop-file", stop_c, "--controls"], "producer-ctl.log", {"CALIPER_HUB_FILE": hub_file2})
    else:
        pc = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file2, "--id",
                    "e2e-ctl", "--seconds", "600", "--stop-file", stop_c, "--controls"], "producer-ctl.log")
    sid = "e2e-ctl"
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120:
        got = asyncio.run(viewer.sessions(url2))
        if connect:
            sid = {s.get("pid"): s["id"] for s in got}.get(pc.pid, sid)
        if any(s["id"] == sid for s in got):
            break
        time.sleep(0.3)
    r = asyncio.run(viewer.control(url2, sid, 0.05, seconds=60.0))
    check(r["control"] is True, f"a hub started with --allow-control says the viewer may control: {r['control']}")
    set_r, run_r = r["replies"][1], r["replies"][2]
    check(len(set_r) == 1 and set_r[0]["status"] == "ok" and set_r[0]["value"] == 0.05,
          f"setting lr to 0.05 is accepted: {set_r}")
    check(r["lr"] == 0.05 and r["pending"] is None, f"and set at the script's next checkpoint: lr = {r['lr']}")
    check([x["status"] for x in run_r] == ["queued", "ok"] and (run_r[-1]["message"] or "").startswith("saved at step"),
          f"running save is queued, then answered with its result: {run_r}")
    if connect:
        # A pause through the hub: the script sends the paused step's data before its
        # run state, and the viewer ends up with exactly that step's loss values.
        o = asyncio.run(viewer.pause_order(url2, sid, seconds=60.0))
        check(o["step"] is not None and o["later"] == o["step"] and not o["errors"],
              f"a pause through the hub stops the run at step {o['step']}, and the viewer holds "
              f"that step's loss values: {o['later']} ({o['errors']})")
        # THE HUB'S ORDER, switched on with --hub-pause-order: the hub delivers the run
        # state after the data it received before it.
        if hub_pause_order:
            check(o["before"] == o["step"],
                  f"the hub delivered the paused state after those values: {o['before']} of "
                  f"{o['step']} had arrived before it")
        else:
            print(f"info  through the hub, {o['before']} of {o['step']} loss values arrived before "
                  "the paused state (checked with --hub-pause-order)", flush=True)
    stop(pc, stop_c)
    with open(os.path.join(work, "producer-ctl.log")) as f:
        tail = [l for l in f.read().splitlines() if l.startswith("{")]
    done = json.loads(tail[-1]) if tail else {}
    check(done.get("lr") == 0.05 and done.get("saves") == 1,
          f"the script trained with the new rate and saved once: {done}")
    hub2.send_signal(signal.SIGTERM)
    try:
        code = hub2.wait(timeout=10)
    except subprocess.TimeoutExpired:
        code = None
    check(code == 0 and not os.path.exists(hub_file2), f"the second hub stops cleanly ({code})")


def bench(spawn, url: str, hub_file: str, work: str) -> None:
    """A 512 x 512 heatmap at 60 Hz through the hub to four viewer processes."""
    prod = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file, "--id", "bench",
                  "--bench", "--rate", "60", "--seconds", "16"], "bench-producer.log")
    wait_session_state(url, "bench", "live")
    code = ("import asyncio, json, sys; sys.path.insert(0, %r); import viewer;"
            "r = asyncio.run(viewer.watch(%r, 'bench', 10.0, views=['bench']));"
            "print(json.dumps({'n': r['snapshots'].get('bench', 0), 'lat': r['latency_ms'], 'bytes': r['bytes']}))"
            % (HERE, url))
    vs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                           env=dict(os.environ, CALIPER_SESSION="0",
                                    PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(HERE)), "python")))
          for _ in range(4)]
    results = [json.loads(v.communicate(timeout=60)[0].strip().splitlines()[-1]) for v in vs]
    prod.wait(timeout=30)
    lat = sorted(x for r in results for x in r["lat"])
    n = sum(r["n"] for r in results)
    mib = sum(r["bytes"] for r in results) / 10.0 / (1 << 20)
    if lat:
        print(f"bench: {n} snapshots in 10 s to four viewers ({n / 40.0:.1f} per viewer per second), "
              f"{mib:.0f} MiB/s, latency p50 {lat[len(lat) // 2]:.2f} ms, p99 {lat[len(lat) * 99 // 100]:.2f} ms "
              f"(Python producer and viewers, decode included)", flush=True)
    check(n > 0, "bench: the four viewers receive the 512 x 512 heatmap")


if __name__ == "__main__":
    sys.exit(main())
