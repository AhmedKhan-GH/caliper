"""The built browser viewer that ships inside the package (``caliper/_web/``).

The files come from the ``web/`` package: ``scripts/build_web.py`` builds it
and copies ``web/dist/*`` here, and the copies are checked in, so installing
the Python package needs no Node. The viewer page and the widget read them
through this module only:

    name = name_for_url_path(request_path)    # "/" -> "index.html"; ValueError if unsafe
    body = read_asset(name)                    # bytes; AssetMissing (a FileNotFoundError)
    headers = {"Content-Type": content_type(name)}

``asset_path()`` refuses every name that could leave the asset folder
(absolute paths, ``..``, backslashes, hidden files, symlinks pointing out).

``CALIPER_WEB_DIR`` in the environment points the package at another folder
of built assets, for example ``web/dist`` while working on the browser code.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

BUILD_COMMAND = "python scripts/build_web.py"
# The files every build has (fonts and other files the page references come
# beside them and are found the same way).
REQUIRED = ("caliper-view.js", "caliper-widget.js", "index.html")
# Caliper's own views compiled to WebAssembly (web/wasm), served beside them:
# the <caliper-view> element imports the loader from the stream's server and
# draws with these views the kinds they list, and with its own views without
# them. Built with the Emscripten SDK, so build_web.py copies them only from
# an existing build and otherwise keeps the checked-in copies.
WASM_VIEWS = ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")

# An explicit table rather than the mimetypes module, which reads the system's
# registry (on Windows it can map .js to text/plain, which browsers refuse to
# run as a module).
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".wasm": "application/wasm",
}


class AssetMissing(FileNotFoundError):
    """A browser asset is not in the asset folder."""


def asset_dir() -> Path:
    """The folder that holds the built browser assets."""
    override = os.environ.get("CALIPER_WEB_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent / "_web")


def _missing(name: str, root: Path) -> AssetMissing:
    return AssetMissing(
        f"caliper: the browser asset {name!r} is not in {root}. The assets are built "
        f"from the web/ package with `{BUILD_COMMAND}` in a source checkout "
        "(an installed wheel ships them).")


def _check_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid asset name {name!r}")
    if "\\" in name or "\0" in name or ":" in name:
        raise ValueError(f"invalid asset name {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("", ".", "..") or part.startswith(".")
                              for part in name.split("/")):
        raise ValueError(f"invalid asset name {name!r}")
    return p


def asset_path(name: str) -> Path:
    """The file of asset ``name`` (a relative POSIX path such as
    ``"caliper-view.js"`` or ``"fonts/x.woff2"``).

    Raises ValueError for a name that could leave the asset folder and
    AssetMissing when the file does not exist."""
    rel = _check_name(name)
    root = asset_dir()
    p = root.joinpath(*rel.parts).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"invalid asset name {name!r}")
    if not p.is_file():
        raise _missing(name, root)
    return p


def read_asset(name: str) -> bytes:
    """The bytes of asset ``name`` (same checks as :func:`asset_path`)."""
    return asset_path(name).read_bytes()


def content_type(name: str) -> str:
    """The HTTP Content-Type for asset ``name``."""
    return _CONTENT_TYPES.get(PurePosixPath(name).suffix.lower(), "application/octet-stream")


def asset_names() -> list[str]:
    """Every asset's name (sorted, POSIX separators, hidden files left out)."""
    root = asset_dir()
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not any(part.startswith(".") for part in rel.parts):
            out.append(rel.as_posix())
    return sorted(out)


def name_for_url_path(url_path: str) -> str:
    """The asset name an HTTP request path asks for: the query and fragment are
    dropped, percent escapes decoded once and ``/`` maps to ``index.html``.
    Raises ValueError for a path that could leave the asset folder."""
    path = url_path.split("?", 1)[0].split("#", 1)[0]
    path = unquote(path, errors="strict")
    if not path.startswith("/"):
        raise ValueError(f"invalid request path {url_path!r}")
    name = path[1:]
    if name == "" or name.endswith("/"):
        name += "index.html"
    _check_name(name)
    return name


def check_assets() -> None:
    """Raise AssetMissing unless every required asset is present."""
    root = asset_dir()
    missing = [n for n in REQUIRED if not (root / n).is_file()]
    if missing:
        raise _missing(", ".join(missing), root)
