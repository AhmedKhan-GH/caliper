#!/usr/bin/env python3
"""The check of GPU programs.

It runs the transcript of the GPU programs acceptance test (tests/gpu_programs/transcript.jsonl): the hidden app
with no script, and caliper-mcp as the agent's only way in. The agent starts a training script as a managed run
and writes WGSL programs over its live tensors. A geometry program draws points in a scene panel, a pixels program
shows in an image panel, and a material stage changes the look of the points. A program that declares a model
tensor writable is refused, and a program over its time limit stops while training goes on. A toggle bound to an
intervention raises the loss while it is on and releases it. No window opens.

usage: gpu_program_check.py [<capture folder>] [--app <caliper>] [--mcp <caliper-mcp>] [--python <python>]

All arguments have defaults: the folder is gpu-program-check in the build folder with the newest app, the programs
are the ones of that build folder, and the Python is the one that runs this script. It needs torch, numpy, the
native network sender and the producer module in that Python, and a build with the WGSL compiler library. It
saves the captures into the folder, prints the SHA-256 count, the cadence N and the capture paths, and exits with 1
on a failure and with 2 where the check cannot run. It stops after five minutes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu_programs"))
import driver  # noqa: E402


# Where a build folder holds each program.
PLACES = {"caliper": "caliper", "caliper-mcp": "src/builder/mcp/caliper-mcp"}


def find(name):
    """The program of the newest build folder that holds both the app and caliper-mcp.

    A checkout can hold several build folders, so the folder with the newest app wins, not the first folder
    by name."""
    folders = [f for f in list(ROOT.glob("cmake-build-*")) + list(ROOT.glob("build*"))
               if all((f / place).is_file() for place in PLACES.values())]
    if not folders:
        return None
    newest = max(folders, key=lambda f: (f / PLACES["caliper"]).stat().st_mtime)
    return str(newest / PLACES[name])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    app = find("caliper")
    default_dir = str(Path(app).parent / "gpu-program-check") if app else "gpu-program-check"
    ap.add_argument("capture_dir", nargs="?", default=default_dir,
                    help="the folder that receives the captures (default: a folder in the build folder)")
    ap.add_argument("--app", default=app, help="the caliper executable")
    ap.add_argument("--mcp", default=find("caliper-mcp"), help="the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="a Python with torch")
    a = ap.parse_args()
    work = Path(a.capture_dir) / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=a.app, mcp=a.mcp, python=a.python, work=str(work),
                              capture_dir=a.capture_dir, limit_s=290,
                              transcript=str(ROOT / "tests" / "gpu_programs" / "transcript.jsonl"))
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
    print(f"SHA-256 of {res.get('unchanged_files')} files (the app, caliper-mcp and the Python package): "
          "the same before and after.")
    print("Cadence of the geometry program: " + json.dumps(res.get("status")))
    print("Loss around the toggle: " + json.dumps(res.get("toggle")))
    print("Pixels that changed: geometry " + str(res.get("geometry_drawn")) + ", image panel "
          + str(res.get("pixels_shown")) + ", material " + json.dumps(res.get("material_changed")))
    print("Captures:")
    for name, paths in (res.get("captures") or {}).items():
        print(f"  {name}: {', '.join(paths)}")
    print("No child process is left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
