#!/usr/bin/env python3
"""Check that macOS binaries and wheels carry the minimum macOS Caliper builds for.

    python3 scripts/check_macos_min.py 14.0 cmake-build-release/caliper dist/caliper-*.whl

Each Mach-O file (an executable, a dylib, a Python extension) must say, in its
LC_BUILD_VERSION (or the older LC_VERSION_MIN_MACOSX), a minimum macOS of
exactly the version given. Each wheel must have the platform tag
macosx_<major>_<minor>_<arch> of that version, and every Mach-O file inside it
must carry it too. vtool reads the load commands; otool -l stands in when
vtool is missing. One line per file; the exit code is 1 when any check fails
or a named file is missing.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import zipfile

MACHO_MAGICS = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"}


def is_macho(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in MACHO_MAGICS
    except OSError:
        return False


def min_versions(path: str) -> list[str]:
    """The minimum macOS versions the load commands of a Mach-O file name (one per architecture)."""
    try:
        out = subprocess.run(["vtool", "-show-build", path], capture_output=True, text=True, check=True).stdout
        found = re.findall(r"^\s*minos\s+(\S+)", out, re.M)
        if found:
            return found
    except (OSError, subprocess.CalledProcessError):
        pass
    out = subprocess.run(["otool", "-l", path], capture_output=True, text=True).stdout
    found = re.findall(r"^\s*minos\s+(\S+)", out, re.M)
    # LC_VERSION_MIN_MACOSX says "version" right after the command.
    found += re.findall(r"cmd LC_VERSION_MIN_MACOSX\n\s*cmdsize \d+\n\s*version (\S+)", out)
    return found


def check_macho(path: str, want: str, label: str) -> bool:
    got = min_versions(path)
    ok = bool(got) and all(v == want for v in got)
    print(("ok    " if ok else "FAIL  ") + f"{label}: minos {', '.join(got) or 'not found'}")
    return ok


def check_wheel(path: str, want: str) -> bool:
    major, minor = (want.split(".") + ["0"])[:2]
    tag = os.path.basename(path)[:-4].split("-")[-1]
    ok = all(t.startswith(f"macosx_{major}_{minor}_") for t in tag.split("."))
    print(("ok    " if ok else "FAIL  ") + f"{os.path.basename(path)}: platform tag {tag}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        for root, _, files in os.walk(tmp):
            for f in sorted(files):
                p = os.path.join(root, f)
                if is_macho(p):
                    ok = check_macho(p, want, f"{os.path.basename(path)}!{os.path.relpath(p, tmp)}") and ok
    return ok


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().split("\n\n")[1])
        return 2
    want, paths = sys.argv[1], sys.argv[2:]
    ok = True
    for p in paths:
        if not os.path.exists(p):
            print(f"FAIL  {p}: missing")
            ok = False
        elif p.endswith(".whl"):
            ok = check_wheel(p, want) and ok
        elif is_macho(p):
            ok = check_macho(p, want, p) and ok
        else:
            print(f"FAIL  {p}: not a Mach-O file or a wheel")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
