#!/usr/bin/env python3
"""The check of programmable graphics: the Attention Arc Tower built from pieces.

It runs the steps of the programmable graphics acceptance test (tests/programmable_graphics/driver.py): the hidden
app with no script, and caliper-mcp as the agent's only way in. The agent starts examples/python/arc_tower_train.py
as a managed run, writes the pieces of demos/arc_tower into the workspace and places them in two scene panels, with
no built-in primitive in either frame. It then scrubs to the first kept step and the bump of the loss, picks an arc,
ablates the head that formed, forks the run from a past step with that head ablated, and stops a draw that is over
its time while training goes on. A piece of a foreign spec waits in review. The script prints each step as it
passes, then each measured result, and then the match of the library piece points@1 with the built-in points when the
build holds that test. No window opens.

usage: arc_tower_check.py [<capture folder>] [--app <caliper>] [--mcp <caliper-mcp>] [--python <python>]

All arguments have defaults: the folder is arc-tower-check in the build folder with the newest app, the programs are
the ones of that build folder, and the Python is the one that runs this script. It needs torch, numpy, the native
network sender and the producer module in that Python, and a build with the WGSL compiler library. It saves the
captures into the folder and exits with 1 on a failure and with 2 where the check cannot run. It stops after 14
minutes.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "programmable_graphics"))
import driver  # noqa: E402

# Where a build folder holds each program.
PLACES = {"caliper": "caliper", "caliper-mcp": "src/builder/mcp/caliper-mcp"}


def find(name):
    """The program of the newest build folder that holds both the app and caliper-mcp."""
    folders = [f for f in list(ROOT.glob("cmake-build-*")) + list(ROOT.glob("build*"))
               if all((f / place).is_file() for place in PLACES.values())]
    if not folders:
        return None
    newest = max(folders, key=lambda f: (f / PLACES["caliper"]).stat().st_mtime)
    return str(newest / PLACES[name])


def points_match(app, folder):
    """The match of points@1 with the built-in points primitive, when the build folder holds its test."""
    exe = Path(app).parent / "caliper_points_match_tests"
    if not exe.is_file():
        print("points@1 against the built-in points: the build holds no caliper_points_match_tests, not run")
        return True
    env = dict(os.environ, CALIPER_POINTS_MATCH_CAPTURE_DIR=str(folder))
    p = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
    if p.returncode == 77:
        print("points@1 against the built-in points: skipped (no Metal device)")
        return True
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    print(f"points@1 against the built-in points: {'matches' if p.returncode == 0 else 'DIFFERS'} ({tail})")
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    app = find("caliper")
    default_dir = str(Path(app).parent / "arc-tower-check") if app else "arc-tower-check"
    ap.add_argument("capture_dir", nargs="?", default=default_dir,
                    help="the folder that receives the captures (default: a folder in the build folder)")
    ap.add_argument("--app", default=app, help="the caliper executable")
    ap.add_argument("--mcp", default=find("caliper-mcp"), help="the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="a Python with torch")
    a = ap.parse_args()
    work = Path(a.capture_dir) / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=a.app, mcp=a.mcp, python=a.python, work=str(work),
                              capture_dir=a.capture_dir, limit_s=840)
    try:
        res = driver.run(args)
    except driver.Skip as e:
        print(f"The check cannot run here: {e}")
        return 2
    except driver.Fail as e:
        print(f"The check failed: {e}")
        return 1
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    print("Foreign piece: " + json.dumps(res.get("foreign")))
    print("Tower: " + str(res.get("tower_pixels")) + " pixels differ from the empty panel, no entity in the spec")
    print("Bump: " + json.dumps(res.get("bump")))
    print("Pick: " + json.dumps(res.get("pick")))
    print("Ablation: " + json.dumps(res.get("ablation")))
    print("Fork: " + json.dumps(res.get("fork")))
    print("Draw stopped: " + json.dumps(res.get("draw_stop")))
    print("Overhead share of the derive halves: " + json.dumps(res.get("overhead_share")))
    print("SHA-256 of the app, caliper-mcp and the Python package: the same before and after.")
    print("Captures:")
    for name, paths in (res.get("captures") or {}).items():
        print(f"  {name}: {', '.join(paths)}")
    print("No child process is left.")
    ok = points_match(a.app, Path(a.capture_dir))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
