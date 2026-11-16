#!/usr/bin/env python3
"""Build caliper._netio, the native sender of caliper.serve(), on its own.

It needs a C++17 compiler, CMake, Ninja (or make) and nanobind in the Python
that runs this script, and nothing else of the project: it compiles
python/src/_netio.cpp and src/host/net in a few seconds and copies the module
into python/caliper/, where ``PYTHONPATH=python`` finds it (git ignores it),
and into the caliper package that this Python has installed, if any (a wheel
in a virtual environment). Run it with the Python of the training script:

    .venv/bin/python scripts/build_netio.py            # Release
    .venv/bin/python scripts/build_netio.py --asan     # AddressSanitizer, plus the caliper_net tests

Reinstalling the wheel replaces the installed package; run it again after.
The wheel's own build of the module (python/CMakeLists.txt) is not tested yet.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def installed_package() -> str | None:
    """The folder of the caliper package this Python imports when it is not the
    one in this tree (found without importing it), or None."""
    try:
        spec = importlib.util.find_spec("caliper")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    d = os.path.realpath(list(spec.submodule_search_locations)[0])
    return None if d == os.path.realpath(os.path.join(ROOT, "python", "caliper")) else d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--asan", action="store_true", help="build with AddressSanitizer")
    ap.add_argument("--build-dir", default=None, help="default: build/netio (or build/netio-asan)")
    args = ap.parse_args()
    build = args.build_dir or os.path.join(ROOT, "build", "netio-asan" if args.asan else "netio")
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    cfg = ["cmake", "-S", os.path.join(ROOT, "python", "netio"), "-B", build, *gen,
           f"-DPython_EXECUTABLE={sys.executable}",
           f"-DCMAKE_BUILD_TYPE={'Debug' if args.asan else 'Release'}",
           f"-DNETIO_ASAN={'ON' if args.asan else 'OFF'}"]
    subprocess.run(cfg, check=True)
    subprocess.run(["cmake", "--build", build], check=True)
    built = glob.glob(os.path.join(build, "out", "_netio*"))
    if not built:
        print("the build produced no module", file=sys.stderr)
        return 1
    dests = [os.path.join(ROOT, "python", "caliper")]
    other = installed_package()
    if other is not None:
        dests.append(other)
    for dest in dests:
        for old in glob.glob(os.path.join(dest, "_netio*")):
            os.remove(old)
        for f in built:
            shutil.copy2(f, dest)
            print(os.path.join(dest, os.path.basename(f)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
