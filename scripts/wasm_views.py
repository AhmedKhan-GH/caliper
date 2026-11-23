"""The files of Caliper's views compiled to WebAssembly (web/wasm), shared by
scripts/build_wasm.py and scripts/fetch_wasm.py: their names, the checks a
copy must pass, and where the copies go.

The three files are build outputs and are not checked in. They go into
python/caliper/_web/, where PYTHONPATH=python and a wheel built from this tree
find them (git ignores them there), and into the caliper package that the
running Python has installed, if it has one that is not this tree.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_WEB = ROOT / "python" / "caliper" / "_web"
# The Emscripten release the module is built with, in CI and by build_wasm.py.
PIN_FILE = ROOT / "web" / "wasm" / "emscripten-version.txt"
# The loader (hand-written, web/wasm/caliper-views-loader.js), the JavaScript
# glue and the module (both written by Emscripten). The loader travels with the
# other two because it is the module's interface to the page.
FILES = ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")
# Everything the module is compiled from. A build of one commit serves every
# later commit that changes none of these paths. The Browser views workflow
# runs when any of them changes, and fetch_wasm.py relies on that.
INPUTS = ("web/wasm", "src", "include", "sdk/include", "cmake/SternbildTokens.cmake",
          "third_party/imgui", "third_party/implot")
# Far above the real sizes (about 1.4 MB for the module), and a bound on what
# a download may write.
MAX_FILE_BYTES = 64 << 20


def pinned_emscripten() -> str:
    """The Emscripten version in web/wasm/emscripten-version.txt."""
    return PIN_FILE.read_text(encoding="utf-8").strip()


def problems(files: dict[str, bytes]) -> list[str]:
    """What is wrong with a set of the three files, or [] when they look like a
    build of the module."""
    out = [f"{name} is missing" for name in FILES if name not in files]
    out += [f"{name} is not one of the three files" for name in files if name not in FILES]
    for name, data in files.items():
        if not data:
            out.append(f"{name} is empty")
        elif len(data) > MAX_FILE_BYTES:
            out.append(f"{name} is larger than {MAX_FILE_BYTES >> 20} MB")
    wasm = files.get("caliper-views.wasm")
    if wasm and not wasm.startswith(b"\0asm"):
        out.append("caliper-views.wasm is not a WebAssembly module")
    loader = files.get("caliper-views-loader.js")
    if loader and b"mountCaliperViews" not in loader:
        out.append("caliper-views-loader.js does not export mountCaliperViews")
    glue = files.get("caliper-views.js")
    if glue and b"createCaliperViews" not in glue:
        out.append("caliper-views.js is not the module's glue (no createCaliperViews)")
    return out


def installed_web() -> Path | None:
    """The _web folder of the caliper package that this Python imports, when
    that package is not the one in this tree and has the folder (found without
    importing it), or None."""
    try:
        spec = importlib.util.find_spec("caliper")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(list(spec.submodule_search_locations)[0]).resolve()
    if pkg == PACKAGE_WEB.parent.resolve():
        return None
    web = pkg / "_web"
    return web if web.is_dir() else None


def destinations(out: list[Path] | None) -> list[Path]:
    """The folders the files go into: the ones named, else this tree's
    python/caliper/_web/ and the installed package's _web/, if any."""
    if out:
        return [p.resolve() for p in out]
    found = [PACKAGE_WEB]
    installed = installed_web()
    if installed is not None:
        found.append(installed)
    return found


def install(files: dict[str, bytes], folders: list[Path]) -> None:
    """Write the three files into each folder. Each file is written next to
    its place and then renamed over it, so a server never reads half a file.
    They get the mode a new file gets (mkstemp's own is owner-only, which a
    wheel would carry into every install)."""
    umask = os.umask(0)
    os.umask(umask)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=folder)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(files[name])
                os.chmod(tmp, 0o666 & ~umask)
                os.replace(tmp, folder / name)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise


def describe(folder: Path) -> str:
    """A folder for a message: relative to the repository when it is inside it."""
    try:
        return str(folder.resolve().relative_to(ROOT))
    except ValueError:
        return str(folder)
