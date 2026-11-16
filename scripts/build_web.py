#!/usr/bin/env python3
"""Build the browser viewer (the web/ package) and copy it into the Python package.

    python scripts/build_web.py              # npm ci, npm run build, copy web/dist/* to python/caliper/_web/
    python scripts/build_web.py --check      # exit 1 when the checked-in copies differ from a fresh build

The copies in python/caliper/_web/ are checked in, so installing the Python
package needs no Node. The script uses only the standard library and needs
Node.js with npm on PATH. Files that are no longer in the build are removed
from the copy; --check never changes the copy.

Caliper's own views compiled to WebAssembly (web/wasm) live in the same
folder as three files, caliper-views-loader.js, caliper-views.js and
caliper-views.wasm, which are build outputs and not checked in. This script
leaves them alone, and --check does not compare them:
`python scripts/build_wasm.py` builds them with Emscripten and
`python scripts/fetch_wasm.py` downloads CI's build. The script says when
they are missing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEB = ROOT / "web"
DEFAULT_OUT = ROOT / "python" / "caliper" / "_web"
REQUIRED = ("caliper-view.js", "caliper-widget.js", "caliper-controls-widget.js", "index.html")
# Not this script's to build, copy, remove or compare (build_wasm.py, fetch_wasm.py).
WASM_FILES = ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")
# Compared with line endings normalised, so a checkout that converts them
# (git's autocrlf on Windows) does not fail --check.
TEXT_SUFFIXES = {".js", ".mjs", ".html", ".css", ".json", ".map", ".svg", ".txt"}


def fail(msg: str) -> None:
    print(f"build_web: {msg}", file=sys.stderr)
    sys.exit(2)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join([Path(cmd[0]).name, *cmd[1:]]), f"(in {cwd.name}/)", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build(web: Path, install: bool) -> Path:
    """Install the package's dependencies and build it. Returns its dist/."""
    if not (web / "package.json").is_file():
        fail(f"no package.json in {web}")
    npm = shutil.which("npm")
    if npm is None:
        fail("npm is not on PATH (install Node.js)")
    if install:
        verb = "ci" if (web / "package-lock.json").is_file() else "install"
        run([npm, verb, "--no-audit", "--no-fund"], web)
    run([npm, "run", "build"], web)
    dist = web / "dist"
    missing = [n for n in REQUIRED if not (dist / n).is_file()]
    if missing:
        fail(f"the build did not write {', '.join(missing)} into {dist}")
    return dist


def files(root: Path) -> dict[str, Path]:
    """Every file under root by its POSIX relative name (hidden files left out)."""
    if not root.is_dir():
        return {}
    out = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not any(part.startswith(".") for part in rel.parts):
            out[rel.as_posix()] = p
    return out


def contents(p: Path) -> bytes:
    data = p.read_bytes()
    return data.replace(b"\r\n", b"\n") if p.suffix.lower() in TEXT_SUFFIXES else data


def built_files(dist: Path) -> dict[str, Path]:
    return {n: p for n, p in files(dist).items() if n not in WASM_FILES}


def copied_files(out: Path) -> dict[str, Path]:
    return {n: p for n, p in files(out).items() if n not in WASM_FILES}


def differences(dist: Path, out: Path) -> list[str]:
    built, copied = built_files(dist), copied_files(out)
    diffs = []
    for name in sorted(built.keys() | copied.keys()):
        if name not in copied:
            diffs.append(f"not checked in: {name}")
        elif name not in built:
            diffs.append(f"not in the build: {name}")
        elif contents(built[name]) != contents(copied[name]):
            diffs.append(f"differs from the build: {name}")
    return diffs


def copy(dist: Path, out: Path) -> None:
    built = built_files(dist)
    # The copy removes stale files, so refuse a folder that holds anything else.
    if (out / ".git").exists() or any(out.rglob("*.py")):
        fail(f"{out} holds Python files or a repository; it is not an asset folder")
    out.mkdir(parents=True, exist_ok=True)
    for name, p in copied_files(out).items():
        if name not in built:
            p.unlink()
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    for name, src in built.items():
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"build_web: copied {len(built)} files into {out}")


def note_wasm(out: Path) -> None:
    missing = [n for n in WASM_FILES if not (out / n).is_file()]
    if missing:
        print(f"build_web: the WebAssembly views are not in {out}, so browsers will draw with the "
              "TypeScript views; build them with `python scripts/build_wasm.py` or download CI's "
              "build with `python scripts/fetch_wasm.py`")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--web", type=Path, default=DEFAULT_WEB,
                    help="the browser package to build (default: web/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="where the built files go (default: python/caliper/_web/)")
    ap.add_argument("--check", action="store_true",
                    help="build, then exit 1 if the copies in --out differ from the build")
    ap.add_argument("--no-install", action="store_true",
                    help="skip `npm ci` (dependencies already installed)")
    args = ap.parse_args(argv)
    web, out = args.web.resolve(), args.out.resolve()

    try:
        dist = build(web, install=not args.no_install)
    except subprocess.CalledProcessError as e:
        fail(f"`{' '.join([Path(e.cmd[0]).name, *e.cmd[1:]])}` failed with exit code {e.returncode}")
    if args.check:
        diffs = differences(dist, out)
        if diffs:
            print(f"build_web: {out} does not match a fresh build of {web}:", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            print("build_web: run `python scripts/build_web.py` and commit the copies",
                  file=sys.stderr)
            return 1
        print(f"build_web: {out} matches the build")
        note_wasm(out)
        return 0
    copy(dist, out)
    note_wasm(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
