#!/usr/bin/env python3
"""The check of hosted runs.

It runs the transcript of the hosted runs acceptance test (tests/m4/transcript.jsonl): the hidden app
with no script, and caliper-mcp as the agent's only way in. The agent starts a training script as a
managed run, forks it with a lower learning rate, draws both loss curves in one panel, reads the log,
and stops both runs. Then it starts and stops an inference script. No window opens.

usage: hosted_runs_check.py [<capture folder>] [--app <caliper>] [--mcp <caliper-mcp>] [--python <python>]

All arguments have defaults: the folder is hosted-runs-check in the working folder, the programs are the
ones of the build folder next to the repository's cmake-build-release or build folder, and the Python is
the one that runs this script. It saves the capture of the two loss curves into the folder, prints the
states of the runs, and exits with 1 on a failure and with 2 where the check cannot run. It stops after
five minutes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "m4"))
import driver  # noqa: E402


def find(name):
    """The built program of the first build folder that has it."""
    for folder in sorted(ROOT.glob("cmake-build-*")) + sorted(ROOT.glob("build*")):
        for hit in folder.rglob(name):
            if hit.is_file():
                return str(hit)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_dir", nargs="?", default="hosted-runs-check", help="the folder that receives the capture")
    ap.add_argument("--app", default=find("caliper"), help="the caliper executable")
    ap.add_argument("--mcp", default=find("caliper-mcp"), help="the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="a Python with torch")
    a = ap.parse_args()
    work = Path(a.capture_dir) / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=a.app, mcp=a.mcp, python=a.python, work=str(work),
                              capture_dir=a.capture_dir, limit_s=290,
                              transcript=str(ROOT / "tests" / "m4" / "transcript.jsonl"))
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
    print(f"Capture: {res.get('capture')}")
    print("Runs after the fork: " + json.dumps(res.get("pair")))
    print(f"The app showed {res.get('attached')}; both curves came through run sources, on one step "
          "axis.")
    print("No child process is left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
