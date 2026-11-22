#!/usr/bin/env python3
"""The check of instrument building.

It runs the acceptance test of instrument building (tests/instrument_building/driver.py): the hidden app with no
script, and caliper-mcp as the agent's only way in. The agent starts a small transformer as a managed run and
builds an instrument on it with MCP tools only. The instrument holds a terrain from the surface operator over a
live module output, the loss as a line in the same scene, a pick that writes a state variable, a heatmap that
follows the pick, a range, a segmented group, a toggle and a slider, a time cursor with a transport, a bound text,
a threshold line and markers, and layout shares. No window opens.

usage: instrument_building_check.py [<capture folder>] [--app <caliper>] [--mcp <caliper-mcp>] [--python <python>]

All arguments have defaults: the folder is instrument-building-check in the build folder with the newest app, the
programs are the ones of that build folder, and the Python is the one that runs this script. It needs torch,
numpy, the native network sender and the producer module in that Python, and a Metal device. It saves the captures
into the folder, prints the time from the first tool call to the first complete frame, the SHA-256 count and the
capture paths, and exits with 1 on a failure and with 2 where the check cannot run. It stops after ten minutes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "instrument_building"))
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
    default_dir = str(Path(app).parent / "instrument-building-check") if app else "instrument-building-check"
    ap.add_argument("capture_dir", nargs="?", default=default_dir,
                    help="the folder that receives the captures (default: a folder in the build folder)")
    ap.add_argument("--app", default=app, help="the caliper executable")
    ap.add_argument("--mcp", default=find("caliper-mcp"), help="the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="a Python with torch")
    a = ap.parse_args()
    work = Path(a.capture_dir) / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=a.app, mcp=a.mcp, python=a.python, work=str(work), capture_dir=a.capture_dir,
                              limit_s=590, stop_after=None)
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
    print(f"Time from the first tool call to the first complete frame: "
          f"{res.get('seconds_to_first_terrain_render'):.2f} s")
    print(f"Terrain rows: {res['terrain_early']['rows']} early, {res['terrain_later']['rows']} 20 steps later, "
          f"{res['terrain_full']['rows']} full; narrow window {res['window_terrain']['rows']}")
    print(f"Pick: token {res['token_event']['value']}, detail version {res['detail_after_pick']['before']} -> "
          f"{res['detail_after_pick']['after']}")
    print(f"Cursor held at step {res['cursor']['held_step']}; bound text: {res['bound_text_held']}")
    print(f"Pixels of the terrain that the palette changed: {res['palette']['pixels_changed']}")
    print("Annotations: " + ", ".join(res["annotations"]) + "; label overlap: " + res["label_overlap_check"])
    print(f"SHA-256 of {res.get('unchanged_files')} files (the app, caliper-mcp and the Python package): "
          "the same before and after.")
    print("Captures:")
    for path in res.get("captures", []):
        print(f"  {path}")
    print("No child process is left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
