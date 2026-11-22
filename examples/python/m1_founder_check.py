#!/usr/bin/env python3
"""The founder check of milestone M1.

It runs the transcript of the acceptance test (tests/m1/transcript.jsonl): a training script with
one Caliper line, the hidden app on its stream, and caliper-mcp as the agent's only way in. The agent
adds a scene panel and a slider, changes the learning rate, and receives a pick. No window opens.

usage: m1_founder_check.py <capture folder> --app <caliper> --mcp <caliper-mcp> [--python <python>]

It saves the capture of the app with the scene panel and the slider into the folder, then prints the
learning rate that the script reports at its next step and the pick event. It stops after five minutes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "m1"))
import driver  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_dir", help="the folder that receives the capture")
    ap.add_argument("--app", required=True, help="the caliper executable")
    ap.add_argument("--mcp", required=True, help="the caliper-mcp executable")
    ap.add_argument("--python", default=sys.executable, help="a Python with torch")
    a = ap.parse_args()
    work = Path(a.capture_dir) / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=a.app, mcp=a.mcp, python=a.python, work=str(work),
                              capture_dir=a.capture_dir, limit_s=290,
                              transcript=str(ROOT / "tests" / "m1" / "transcript.jsonl"))
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
    print(f"The script reports learning rate {res['lr_at_next_step']} at step {res['lr_step']}.")
    print("Pick event: " + json.dumps(res["pick"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
