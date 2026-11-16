#!/usr/bin/env python3
"""Build Caliper's views compiled to WebAssembly (web/wasm) and copy them into the Python package.

    python scripts/build_wasm.py                    # Release, into python/caliper/_web/
    python scripts/build_wasm.py --config Debug
    EMSDK=<emsdk> python scripts/build_wasm.py      # Emscripten from an emsdk checkout

It needs Emscripten at the version in web/wasm/emscripten-version.txt, CMake
and Ninja, and uses only the standard library. It takes emcc from $EMSDK, an
emsdk checkout with that version activated, when EMSDK is set, and otherwise
from PATH (an emsdk_env.sh that has been sourced puts it there). Another
version is refused unless --any-version is given.

The build goes into web/wasm/build (--build-dir), and the three files the
browser loads, caliper-views-loader.js, caliper-views.js and
caliper-views.wasm, are copied into python/caliper/_web/, where
PYTHONPATH=python and a wheel built from this tree find them (git ignores
them), and into the caliper package that this Python has installed, if any.
--out names other folders instead, and --no-copy only builds.

Without these files, caliper.serve() and the notebook widgets draw with the
views written in TypeScript. `python scripts/fetch_wasm.py` downloads the
build that CI made instead of building it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wasm_views  # noqa: E402

SOURCE = wasm_views.ROOT / "web" / "wasm"
DEFAULT_BUILD = SOURCE / "build"


class Fail(Exception):
    """A problem with the setup (exit 2) or the build (exit 1)."""

    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


def emscripten_env(env: dict[str, str]) -> dict[str, str]:
    """The environment with $EMSDK's Emscripten first on PATH, as emsdk_env.sh
    sets it up; unchanged when EMSDK is not set."""
    emsdk = env.get("EMSDK")
    if not emsdk:
        return env
    root = Path(emsdk).expanduser().resolve()
    if not (root / "upstream" / "emscripten").is_dir():
        raise Fail(f"EMSDK is {emsdk}, which holds no activated Emscripten (no upstream/emscripten); "
                   f"run `emsdk install {wasm_views.pinned_emscripten()}` and "
                   f"`emsdk activate {wasm_views.pinned_emscripten()}` there")
    out = dict(env)
    out["EMSDK"] = str(root)
    if (root / ".emscripten").is_file():
        out["EM_CONFIG"] = str(root / ".emscripten")
    out["PATH"] = os.pathsep.join([str(root / "upstream" / "emscripten"), str(root), env.get("PATH", "")])
    return out


def tool(name: str, env: dict[str, str], hint: str) -> str:
    found = shutil.which(name, path=env.get("PATH"))
    if found is None:
        raise Fail(f"{name} is not on PATH; {hint}")
    return found


def emscripten_version(emcc: str, env: dict[str, str]) -> str:
    try:
        r = subprocess.run([emcc, "--version"], env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Fail(f"`emcc --version` failed: {e}")
    first = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else ""
    m = re.search(r"\)\s+(\d+\.\d+\.\d+)", first) or re.search(r"\b(\d+\.\d+\.\d+)\b", first)
    if r.returncode != 0 or not m:
        raise Fail(f"`emcc --version` gave no version (exit {r.returncode}): {first!r}")
    return m.group(1)


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join([Path(cmd[0]).name, *cmd[1:]]), flush=True)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        raise Fail(f"`{Path(cmd[0]).name} {' '.join(cmd[1:])}` failed with exit code {e.returncode}", 1)


def build(build_dir: Path, config: str, any_version: bool, env: dict[str, str]) -> dict[str, bytes]:
    """Configure and build web/wasm in build_dir; returns the three files."""
    pin = wasm_views.pinned_emscripten()
    env = emscripten_env(env)
    emcc = tool("emcc", env, f"install Emscripten {pin} with emsdk and set EMSDK to the checkout, "
                             "or download CI's build with `python scripts/fetch_wasm.py`")
    version = emscripten_version(emcc, env)
    if version != pin:
        msg = f"emcc is Emscripten {version}, and web/wasm is built with {pin}"
        if not any_version:
            raise Fail(f"{msg}; run `emsdk install {pin}` and `emsdk activate {pin}`, "
                       "or pass --any-version to build with it anyway")
        print(f"build_wasm: {msg}; building with it anyway (--any-version)", file=sys.stderr)
    print(f"build_wasm: Emscripten {version} ({emcc})")
    emcmake = tool("emcmake", env, "it comes with Emscripten, beside emcc")
    cmake = tool("cmake", env, "install CMake")
    tool("ninja", env, "install Ninja")
    run([emcmake, cmake, "-S", str(SOURCE), "-B", str(build_dir), "-G", "Ninja",
         f"-DCMAKE_BUILD_TYPE={config}"], env)
    run([cmake, "--build", str(build_dir)], env)
    files = {}
    for name in wasm_views.FILES:
        p = build_dir / name
        if not p.is_file():
            raise Fail(f"the build did not write {name} into {build_dir}", 1)
        files[name] = p.read_bytes()
    bad = wasm_views.problems(files)
    if bad:
        raise Fail("the build's files are not usable: " + "; ".join(bad), 1)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD,
                    help="the CMake build folder (default: web/wasm/build)")
    ap.add_argument("--config", choices=("Release", "Debug"), default="Release")
    ap.add_argument("--out", type=Path, action="append",
                    help="a folder to copy the three files into, instead of python/caliper/_web/ "
                         "and the installed package (may be given more than once)")
    ap.add_argument("--no-copy", action="store_true", help="build only")
    ap.add_argument("--any-version", action="store_true",
                    help="build with an Emscripten other than the pinned one")
    args = ap.parse_args(argv)
    try:
        files = build(args.build_dir.resolve(), args.config, args.any_version, dict(os.environ))
    except Fail as e:
        print(f"build_wasm: {e}", file=sys.stderr)
        return e.code
    sizes = ", ".join(f"{n} {len(files[n]):,} bytes" for n in wasm_views.FILES)
    print(f"build_wasm: built {sizes}")
    if args.no_copy:
        return 0
    folders = wasm_views.destinations(args.out)
    wasm_views.install(files, folders)
    for folder in folders:
        print(f"build_wasm: copied into {wasm_views.describe(folder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
