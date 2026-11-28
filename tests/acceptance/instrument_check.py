#!/usr/bin/env python3
"""The instrument check.

It runs the transcript of the acceptance test (tests/instruments/transcript.jsonl): a training script with
one Caliper line, the hidden app on its stream, and caliper-mcp as the agent's only way in. The agent
adds a scene panel and a slider, changes the learning rate, and receives a pick. No window opens.

usage: python3 tests/acceptance/instrument_check.py [capture folder] [--app <caliper>] [--mcp <caliper-mcp>]
                                                    [--python <python>]

With no argument it finds the app and caliper-mcp in cmake-build-release, then cmake-build-notorch,
of this repository. It uses the Python that runs it when that Python has torch, else the repository's
.venv. It writes the capture into a new folder under the build folder. It takes the capture after the
learning rate changed, so that the slider shows the new value. It prints one line per acceptance
point and stops after five minutes.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "instruments"))
import driver  # noqa: E402

BUILDS = ("cmake-build-release", "cmake-build-notorch")


def find_build():
    """The first build folder that holds both programs: (folder, app, mcp), or None."""
    for b in BUILDS:
        app = ROOT / b / "caliper"
        mcp = ROOT / b / "src" / "builder" / "mcp" / "caliper-mcp"
        if app.exists() and mcp.exists():
            return ROOT / b, app, mcp
    return None


def has_torch(python):
    try:
        return subprocess.run([str(python), "-c", "import torch"], capture_output=True).returncode == 0
    except OSError:
        return False


def find_python():
    for py in (sys.executable, ROOT / ".venv" / "bin" / "python"):
        if py and has_torch(py):
            return str(py)
    return sys.executable


def caliper_lines():
    """The lines of the training script that use Caliper, apart from its import."""
    code = (ROOT / "tests" / "instruments" / "train.py").read_text().split('"""')[2].splitlines()
    return [l.strip() for l in code if "caliper." in l and not l.lstrip().startswith(("#", "import "))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_dir", nargs="?", help="the folder that receives the capture")
    ap.add_argument("--app", help="the caliper executable")
    ap.add_argument("--mcp", help="the caliper-mcp executable")
    ap.add_argument("--python", help="a Python with torch")
    a = ap.parse_args()
    found = find_build()
    app = a.app or (found and str(found[1]))
    mcp = a.mcp or (found and str(found[2]))
    if not app or not mcp:
        print("The check cannot run here: no built caliper and caliper-mcp in cmake-build-release or "
              "cmake-build-notorch. Build them or give --app and --mcp.")
        return 2
    python = a.python or find_python()
    base = found[0] if found else ROOT
    out = Path(a.capture_dir) if a.capture_dir else base / time.strftime("instrument-check-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"     # the logs of the run; removed at the end
    args = argparse.Namespace(app=app, mcp=mcp, python=python, work=str(work),
                              capture_dir=str(out), limit_s=290,
                              transcript=str(ROOT / "tests" / "instruments" / "transcript.jsonl"))
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
    ok = set(res.get("ok_steps", []))
    lines = caliper_lines()
    points = [
        (len(lines) == 1, f"The script has one Caliper line: {lines[0] if lines else 'none'}"),
        ("scene" in ok and "state" in ok, "The scene panel \"world\" exists and the app is attached to the run."),
        ("slider" in ok, "The slider \"rate\" exists, bound to param:lr."),
        ("lr_at_next_step" in res,
         f"The script reports learning rate {res.get('lr_at_next_step')} at step {res.get('lr_step')}."),
        ("pick" in res, "Pick event: " + json.dumps(res.get("pick"))),
    ]
    for good, text in points:
        print(("PASS " if good else "FAIL ") + text)
    print(f"Capture (taken after the change, so the slider shows the new value): {res.get('capture')}")
    return 0 if all(g for g, _ in points) else 1


if __name__ == "__main__":
    sys.exit(main())
