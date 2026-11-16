#!/usr/bin/env python3
"""Build the browser viewer (the web/ package) and copy it into the Python package.

    python scripts/build_web.py              # npm ci, npm run build, copy web/dist/* to python/caliper/_web/
    python scripts/build_web.py --check      # exit 1 when the checked-in copies differ from a fresh build

The copies in python/caliper/_web/ are checked in, so installing the Python
package needs no Node. The script uses only the standard library and needs
Node.js with npm on PATH. Files that are no longer in the build are removed
from the copy; --check never changes the copy.

Caliper's own views compiled to WebAssembly (web/wasm) ship beside them as
three files: caliper-views-loader.js, caliper-views.js and caliper-views.wasm.
Building those needs the Emscripten SDK, which most checkouts do not have, so
this script does not build them: it copies them from web/wasm's build folder
(--wasm-build, default web/wasm/build, as written by
`python scripts/build_wasm.py`) when all three are there, and otherwise
leaves the checked-in copies alone. --check compares them only when they are
there to compare with.
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
DEFAULT_WASM = ROOT / "web" / "wasm" / "build"
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


def wasm_files(build: Path | None) -> dict[str, Path]:
    """The WebAssembly views' three files in their build folder, or {} unless all three are there."""
    if build is None:
        return {}
    found = {name: build / name for name in WASM_FILES}
    return found if all(p.is_file() for p in found.values()) else {}


def built_files(dist: Path, wasm: dict[str, Path]) -> dict[str, Path]:
    return {**files(dist), **wasm}


def differences(dist: Path, out: Path, wasm: dict[str, Path] | None = None) -> list[str]:
    wasm = wasm or {}
    built, copied = built_files(dist, wasm), files(out)
    if not wasm:   # nothing to compare the checked-in WebAssembly views with
        copied = {n: p for n, p in copied.items() if n not in WASM_FILES}
    diffs = []
    for name in sorted(built.keys() | copied.keys()):
        if name not in copied:
            diffs.append(f"not checked in: {name}")
        elif name not in built:
            diffs.append(f"not in the build: {name}")
        elif contents(built[name]) != contents(copied[name]):
            diffs.append(f"differs from the build: {name}")
    return diffs


def copy(dist: Path, out: Path, wasm: dict[str, Path] | None = None) -> None:
    wasm = wasm or {}
    built = built_files(dist, wasm)
    # The copy removes stale files, so refuse a folder that holds anything else.
    if (out / ".git").exists() or any(out.rglob("*.py")):
        fail(f"{out} holds Python files or a repository; it is not an asset folder")
    out.mkdir(parents=True, exist_ok=True)
    for name, p in files(out).items():
        # Without a WebAssembly build, the checked-in views stay as they are.
        if name not in built and (wasm or name not in WASM_FILES):
            p.unlink()
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    for name, src in built.items():
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"build_web: copied {len(built)} files into {out}")


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
    ap.add_argument("--wasm-build", type=Path, default=DEFAULT_WASM,
                    help="web/wasm's build folder, whose three files are copied when all are there "
                         "(default: web/wasm/build)")
    args = ap.parse_args(argv)
    web, out = args.web.resolve(), args.out.resolve()
    wasm = wasm_files(args.wasm_build.resolve())
    if wasm:
        print(f"build_web: the WebAssembly views come from {args.wasm_build}")
    else:
        print(f"build_web: no WebAssembly build in {args.wasm_build}; the checked-in views are kept")

    try:
        dist = build(web, install=not args.no_install)
    except subprocess.CalledProcessError as e:
        fail(f"`{' '.join([Path(e.cmd[0]).name, *e.cmd[1:]])}` failed with exit code {e.returncode}")
    if args.check:
        diffs = differences(dist, out, wasm)
        if diffs:
            print(f"build_web: {out} does not match a fresh build of {web}:", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            print("build_web: run `python scripts/build_web.py` and commit the copies",
                  file=sys.stderr)
            return 1
        print(f"build_web: {out} matches the build")
        return 0
    copy(dist, out, wasm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
