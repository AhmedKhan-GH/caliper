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
5. The shorter script exits: its session is listed as ended, and a viewer that
   attaches afterwards still gets its last snapshots and its whole loss series.
6. The hub's resident memory, sampled throughout, stays flat.
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
in the session list by pid. With --bench it then measures the relay with a
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
        t0 = time.monotonic()
        while not os.path.exists(hub_file) and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        if not check(os.path.exists(hub_file), "caliper serve writes its hub file"):
            return 1
        time.sleep(0.1)
        check(stat.S_IMODE(os.stat(hub_file).st_mode) == 0o600, "the hub file is readable by its owner only")
        with open(hub_file) as f:
            h = json.load(f)
        url = f"{h['stream_url']}?token={h['token']}"
        rss.append((time.monotonic(), rss_mib(hub.pid)))

        # 2. two scripts
        if a.connect:
            pa = spawn([sys.executable, os.path.join(HERE, "producer_connect.py"), "--seconds", "48"],
                       "producer-a.log")
            pb = spawn([sys.executable, os.path.join(HERE, "producer_connect.py"), "--seconds", "26"],
                       "producer-b.log")
        else:
            pa = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file, "--id", "e2e-a",
                        "--seconds", "48", "--controls"], "producer-a.log")
            pb = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file, "--id", "e2e-b",
                        "--seconds", "26"], "producer-b.log")
        ida, idb = "e2e-a", "e2e-b"
        listed = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < 30:
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

        # 3. a Python viewer on each session
        for sid in (ida, idb):
            r = asyncio.run(viewer.watch(url, sid, 3.0))
            rss.append((time.monotonic(), rss_mib(hub.pid)))
            snaps = r["snapshots"]
            loss = r["series"].get("loss", {})
            check(r["session"] == sid and not r["errors"], f"the Python viewer attaches to {sid} ({r['errors']})")
            check(snaps.get("fc1.weight", 0) >= 20 and snaps.get("fc2.weight", 0) >= 20,
                  f"it receives snapshots of both heatmaps from {sid}: {snaps}")
            check(loss.get("first") == 0 and loss.get("contiguous") and loss.get("values", 0) >= 50,
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

        # 5. the shorter script ends
        try:
            pb.wait(timeout=40)
        except subprocess.TimeoutExpired:
            pass
        check(pb.returncode == 0, f"script {idb} finished (exit {pb.returncode})")
        with open(os.path.join(work, "producer-b.log")) as f:
            tail = [l for l in f.read().splitlines() if l.startswith("{")]
        b_done = json.loads(tail[-1]) if tail else {}
        time.sleep(0.5)
        r = asyncio.run(viewer.watch(url, idb, 1.5))
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
        r = asyncio.run(viewer.watch(url, ida, 1.0))
        check(r["state"] == "live" and r["snapshots"].get("fc1.weight", 0) > 0, "the other session goes on")

        # 6. the hub's memory over the run
        while pa.poll() is None and time.monotonic() - rss[0][0] < 44:
            time.sleep(1.0)
            rss.append((time.monotonic(), rss_mib(hub.pid)))
        settled = [m for t, m in rss if t - rss[0][0] > 8]
        if settled:
            check(max(settled) - min(settled) < 16,
                  f"the hub's resident memory is flat: {min(settled):.1f} to {max(settled):.1f} MiB "
                  f"after warm-up ({len(settled)} samples, {rss[0][1]:.1f} MiB at start)")

        # optional: the relay at 60 Hz to four viewers in other processes
        if a.bench:
            bench(spawn, url, hub_file, work)

        # 7. controls (protocol 3.1): refused by this hub, forwarded by one with --allow-control
        if not a.connect:                   # the package's scripts expose no controls yet
            controls(spawn, url, caliper, work)

        # 8. stop
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


def controls(spawn, url: str, caliper: str, work: str) -> None:
    r = asyncio.run(viewer.control(url, "e2e-a", 0.05, seconds=6.0))
    check("controls" in r["capabilities"] and r["control"] is False,
          f"the hub announces controls and says this viewer may not control: {r['capabilities']} {r['control']}")
    check(r["controls"] >= 1 and r["lr"] == 0.001,
          f"the viewer sees the script's run state and its parameter lr = {r['lr']}")
    refused = [x for k in (1, 2) for x in r["replies"][k]]
    check(len(refused) == 2 and all(x["error"] == "forbidden" for x in refused),
          f"without --allow-control both CONTROLs are answered forbidden: {r['replies']}")

    hub_file2 = os.path.join(work, "hub2.json")
    hub2 = spawn([caliper, "serve", "--port", "0", "--allow-control"], "hub2.log", {"CALIPER_HUB_FILE": hub_file2})
    t0 = time.monotonic()
    while not os.path.exists(hub_file2) and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    time.sleep(0.1)
    with open(hub_file2) as f:
        h2 = json.load(f)
    url2 = f"{h2['stream_url']}?token={h2['token']}"
    pc = spawn([sys.executable, os.path.join(HERE, "producer.py"), "--hub-file", hub_file2, "--id", "e2e-ctl",
                "--seconds", "12", "--controls"], "producer-ctl.log")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        if any(s["id"] == "e2e-ctl" for s in asyncio.run(viewer.sessions(url2))):
            break
        time.sleep(0.3)
    r = asyncio.run(viewer.control(url2, "e2e-ctl", 0.05, seconds=8.0))
    check(r["control"] is True, f"a hub started with --allow-control says the viewer may control: {r['control']}")
    set_r, run_r = r["replies"][1], r["replies"][2]
    check(len(set_r) == 1 and set_r[0]["status"] == "ok" and set_r[0]["value"] == 0.05,
          f"setting lr to 0.05 is accepted: {set_r}")
    check(r["lr"] == 0.05 and r["pending"] is None, f"and set at the script's next checkpoint: lr = {r['lr']}")
    check([x["status"] for x in run_r] == ["queued", "ok"] and (run_r[-1]["message"] or "").startswith("saved at step"),
          f"running save is queued, then answered with its result: {run_r}")
    try:
        pc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
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
    time.sleep(1.5)
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
